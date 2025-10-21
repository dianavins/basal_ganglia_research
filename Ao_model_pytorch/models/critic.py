"""
Critic Network for State Value Estimation
Standard MLP as described in Ao et al. 2024 Section 3.2
"""

import torch
import torch.nn as nn
import numpy as np


class CriticNetwork(nn.Module):
    """
    Value network V(s) for Actor-Critic architecture.

    Architecture (from paper):
        Input: one-hot state [16]
        Linear(16, 64) → ReLU
        Linear(64, 64) → ReLU
        Linear(64, 1) → state value

    This is a standard ANN (not spiking) that estimates state values
    for TD error calculation.

    Args:
        n_states: Number of states (default: 16 for FrozenLake)
        hidden_dim: Hidden layer size (default: 64)
    """

    def __init__(self, n_states=16, hidden_dim=64):
        super().__init__()

        self.n_states = n_states
        self.hidden_dim = hidden_dim

        # Three-layer MLP
        self.network = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        """
        Compute state value V(s)

        Args:
            state: int or Tensor
                  If int: state index (0-15)
                  If Tensor: [batch_size, n_states] one-hot encoded
                            or [batch_size] state indices

        Returns:
            value: Tensor [batch_size, 1] or [1]
                  Estimated state value
        """
        # Convert state to one-hot if needed
        # Determine device from network parameters
        device = next(self.network.parameters()).device

        if isinstance(state, (int, np.integer)):
            state_onehot = torch.zeros(1, self.n_states, device=device)
            state_onehot[0, state] = 1.0
        elif isinstance(state, torch.Tensor):
            if state.dim() == 1 and state.dtype == torch.long:
                # State indices [batch_size]
                batch_size = state.size(0)
                state_onehot = torch.zeros(batch_size, self.n_states, device=device)
                state_onehot.scatter_(1, state.unsqueeze(1), 1.0)
            else:
                # Already one-hot encoded
                state_onehot = state.to(device)
        else:
            # Convert numpy or other types
            state_onehot = torch.zeros(1, self.n_states, device=device)
            state_onehot[0, int(state)] = 1.0

        # Forward through network
        value = self.network(state_onehot)

        return value

    def compute_td_error(self, state, reward, next_state, gamma=0.99):
        """
        Compute TD error: δ(t) = r + γ*V(s') - V(s)  (Equation 2)

        Args:
            state: current state
            reward: immediate reward r
            next_state: next state s'
            gamma: discount factor (default: 0.99)

        Returns:
            td_error: scalar, TD error δ(t)
            v_current: current state value V(s)
            v_next: next state value V(s')
        """
        with torch.no_grad():
            v_current = self.forward(state)
            v_next = self.forward(next_state)

        td_error = reward + gamma * v_next - v_current

        return td_error.item(), v_current.item(), v_next.item()

    def compute_td_target(self, reward, next_state, gamma=0.99, done=False):
        """
        Compute TD target: r + γ*V(s') * (1 - done)

        Args:
            reward: immediate reward
            next_state: next state
            gamma: discount factor
            done: episode termination flag

        Returns:
            target: TD target for training
        """
        with torch.no_grad():
            v_next = self.forward(next_state)

        # If episode done, target is just the reward
        if done:
            target = reward
        else:
            target = reward + gamma * v_next

        return target
