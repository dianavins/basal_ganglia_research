"""
Cortico-Striatal Synaptic Plasticity Learning Rule
Implements Equation 3 from Ao et al. 2024
"""

import torch
import numpy as np
from collections import defaultdict


class CorticostriatalPlasticity:
    """
    Novel cortico-striatal synaptic plasticity learning rule with:
    1. Dopamine-modulated Hebbian learning (TD error)
    2. Direct reward modulation
    3. Exploration strategy (UCB-like)

    Learning rule (Equation 3):
        ΔW_PFC_D1m = λ_D1 * δ(t) * PFC * D1m + α * r * D1m + β * Φ * D1m
        ΔW_PFC_D2m = λ_D2 * δ(t) * PFC * D2m + α * r * D2m + β * Φ * D2m

    where:
        δ(t) = TD error from Critic
        PFC, D1m, D2m = spike counts over T timesteps
        Φ = sqrt(2 * ln(N) / n) = exploration bonus (Equation 4)
        N = total attempts for all actions in state
        n = attempts for specific action in state

    Args:
        n_states: Number of states
        n_actions: Number of actions
        alpha: Direct reward modulation rate (default: 0.19)
        lambda_d1_init: Initial learning rate for D1 pathway (default: 0.01)
        lambda_d2_init: Initial learning rate for D2 pathway (default: 0.01)
        beta_init: Initial exploration bonus weight (default: 0.05)
        gamma: TD discount factor (default: 0.99)
        decay_type: 'exponential' or 'cosine' for λ and β decay
    """

    def __init__(self, n_states=16, n_actions=4,
                 alpha=0.19,
                 lambda_d1_init=0.01,
                 lambda_d2_init=0.01,
                 beta_init=0.05,
                 gamma=0.99,
                 decay_type='exponential'):

        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.lambda_d1_init = lambda_d1_init
        self.lambda_d2_init = lambda_d2_init
        self.beta_init = beta_init
        self.gamma = gamma
        self.decay_type = decay_type

        # Current learning rates (decay over epochs)
        self.lambda_d1 = lambda_d1_init
        self.lambda_d2 = lambda_d2_init
        self.beta = beta_init

        # Track action attempts per state for exploration bonus
        # action_counts[state][action] = number of times action taken in state
        self.action_counts = defaultdict(lambda: np.zeros(n_actions))

        # Track total attempts per state
        self.state_attempts = np.zeros(n_states)

        # Episode counter for decay
        self.episode = 0

    def compute_exploration_bonus(self, state, action):
        """
        Compute exploration bonus Φ = sqrt(2 * ln(N) / n)  (Equation 4)

        Args:
            state: int, current state
            action: int, selected action

        Returns:
            phi: float, exploration bonus for this (state, action)
        """
        N = self.state_attempts[state] + 1  # Total attempts in this state
        n = self.action_counts[state][action] + 1  # Attempts for this action

        # Avoid log(0)
        if N <= 1:
            return 1.0

        phi = np.sqrt(2.0 * np.log(N) / n)
        return phi

    def update_action_counts(self, state, action):
        """Track action selection for exploration bonus"""
        self.action_counts[state][action] += 1
        self.state_attempts[state] += 1

    def compute_weight_update(self, pfc_spikes, d1_spikes, d2_spikes,
                             td_error, reward, state, action):
        """
        Compute cortico-striatal weight updates (Equation 3)

        Args:
            pfc_spikes: [n_states] spike counts for PFC neurons
            d1_spikes: [n_actions] spike counts for D1-MSNs
            d2_spikes: [n_actions] spike counts for D2-MSNs
            td_error: scalar, TD error δ(t)
            reward: scalar, immediate reward r
            state: int, current state
            action: int, selected action

        Returns:
            delta_w_d1: [n_states, n_actions] weight update for PFC→D1
            delta_w_d2: [n_states, n_actions] weight update for PFC→D2
        """
        # Compute exploration bonus
        phi = self.compute_exploration_bonus(state, action)

        # Convert to tensors if needed
        if not isinstance(pfc_spikes, torch.Tensor):
            pfc_spikes = torch.tensor(pfc_spikes)
        if not isinstance(d1_spikes, torch.Tensor):
            d1_spikes = torch.tensor(d1_spikes)
        if not isinstance(d2_spikes, torch.Tensor):
            d2_spikes = torch.tensor(d2_spikes)

        # Reshape for outer product: [n_states, 1] × [1, n_actions]
        pfc_spikes = pfc_spikes.view(-1, 1)
        d1_spikes = d1_spikes.view(1, -1)
        d2_spikes = d2_spikes.view(1, -1)

        # Three terms in Equation 3:
        # Term 1: Dopamine-modulated Hebbian (TD error)
        delta_w_d1_td = self.lambda_d1 * td_error * pfc_spikes * d1_spikes
        delta_w_d2_td = self.lambda_d2 * td_error * pfc_spikes * d2_spikes

        # Term 2: Direct reward modulation
        delta_w_d1_reward = self.alpha * reward * pfc_spikes * d1_spikes
        delta_w_d2_reward = self.alpha * reward * pfc_spikes * d2_spikes

        # Term 3: Exploration bonus
        delta_w_d1_explore = self.beta * phi * pfc_spikes * d1_spikes
        delta_w_d2_explore = self.beta * phi * pfc_spikes * d2_spikes

        # Total update
        delta_w_d1 = delta_w_d1_td + delta_w_d1_reward + delta_w_d1_explore
        delta_w_d2 = delta_w_d2_td + delta_w_d2_reward + delta_w_d2_explore

        return delta_w_d1, delta_w_d2

    def apply_weight_update(self, actor, pfc_spikes, d1_spikes, d2_spikes,
                           td_error, reward, state, action,
                           w_max=0.4, normalize=True):
        """
        Compute and apply weight updates to actor network

        Args:
            actor: BasalGangliaActor instance
            pfc_spikes: PFC spike counts
            d1_spikes: D1-MSN spike counts
            d2_spikes: D2-MSN spike counts
            td_error: TD error
            reward: immediate reward
            state: current state
            action: selected action
            w_max: maximum weight value (default: 0.4)
            normalize: whether to renormalize weights after update
        """
        # Compute updates
        delta_w_d1, delta_w_d2 = self.compute_weight_update(
            pfc_spikes, d1_spikes, d2_spikes,
            td_error, reward, state, action
        )

        # Apply updates (no autograd, manual update)
        with torch.no_grad():
            actor.w_pfc_d1.data += delta_w_d1
            actor.w_pfc_d2.data += delta_w_d2

        # Update action counts for exploration
        self.update_action_counts(state, action)

        # Clamp weights to [0, w_max]
        actor.clamp_weights(w_max)

        # Optional: renormalize rows
        if normalize:
            actor._normalize_weights()

    def decay_learning_rates(self, episode, total_episodes):
        """
        Decay λ and β over episodes

        Args:
            episode: current episode number
            total_episodes: total training episodes
        """
        self.episode = episode

        if self.decay_type == 'exponential':
            # Exponential decay with warmup period
            # No decay for first 20% of training, then gentle exponential decay
            warmup_episodes = int(0.2 * total_episodes)
            if episode < warmup_episodes:
                decay_factor = 1.0  # No decay during warmup
            else:
                # Gentle exponential decay: -2.0 instead of -5.0
                progress = (episode - warmup_episodes) / (total_episodes - warmup_episodes)
                decay_factor = np.exp(-2.0 * progress)

            self.lambda_d1 = self.lambda_d1_init * decay_factor
            self.lambda_d2 = self.lambda_d2_init * decay_factor
            self.beta = self.beta_init * decay_factor

        elif self.decay_type == 'cosine':
            # Cosine annealing with warmup
            warmup_episodes = int(0.2 * total_episodes)
            if episode < warmup_episodes:
                decay_factor = 1.0
            else:
                progress = (episode - warmup_episodes) / (total_episodes - warmup_episodes)
                decay_factor = 0.5 * (1.0 + np.cos(np.pi * progress))

            self.lambda_d1 = self.lambda_d1_init * decay_factor
            self.lambda_d2 = self.lambda_d2_init * decay_factor
            self.beta = self.beta_init * decay_factor

    def reset_exploration_counts(self):
        """Reset action count statistics (e.g., between training runs)"""
        self.action_counts = defaultdict(lambda: np.zeros(self.n_actions))
        self.state_attempts = np.zeros(self.n_states)

    def get_learning_rates(self):
        """Return current learning rates"""
        return {
            'lambda_d1': self.lambda_d1,
            'lambda_d2': self.lambda_d2,
            'beta': self.beta
        }
