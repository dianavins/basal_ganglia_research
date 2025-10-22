"""
Training Loop for Basal Ganglia Model
Implements the experimental setup from Ao et al. 2024 (Section 4)
"""

import os
from matplotlib.pyplot import step
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from collections import defaultdict

from models import BasalGangliaActor, CriticNetwork, CorticostriatalPlasticity
from config import Config
from utils.visualization import (
    plot_firing_rates,
    plot_training_curves,
    plot_weight_heatmaps,
    save_firing_snapshot,
    plot_learning_rates,
    plot_action_distribution
)
from utils.replay_buffer import PrioritizedReplayBuffer
from utils.curriculum import CurriculumManager


class BasalGangliaTrainer:
    """
    Trainer for Basal Ganglia Actor-Critic Model on FrozenLake-v1

    Training follows Ao et al. 2024:
    - Actor: Spiking basal ganglia network (no backprop)
    - Critic: 3-layer MLP trained with Adam
    - Plasticity: Cortico-striatal weight updates (Equation 3)
    - Environment: FrozenLake-v1 (16 states, 4 actions)
    """

    def __init__(self, config=Config):
        """
        Initialize trainer with configuration

        Args:
            config: Config class with hyperparameters
        """
        self.config = config

        # Set random seeds
        torch.manual_seed(config.SEED)
        np.random.seed(config.SEED)

        # Initialize environment
        self.env = gym.make(
            config.ENV_NAME,
            is_slippery=config.IS_SLIPPERY,
            render_mode=config.RENDER_MODE
        )
        self.env.reset(seed=config.SEED)

        # Initialize networks
        self.actor = BasalGangliaActor(
            n_states=config.N_STATES,
            n_actions=config.N_ACTIONS,
            tau=config.TAU,
            v_reset=config.V_RESET,
            v_th=config.V_TH,
            dt=config.DT,
            T=config.T_SIMULATION
        ).to(config.DEVICE)

        self.critic = CriticNetwork(
            n_states=config.N_STATES,
            hidden_dim=config.CRITIC_HIDDEN_DIM
        ).to(config.DEVICE)

        # Initialize plasticity rule
        self.plasticity = CorticostriatalPlasticity(
            n_states=config.N_STATES,
            n_actions=config.N_ACTIONS,
            alpha=config.ALPHA,
            lambda_d1_init=config.LAMBDA_D1_INIT,
            lambda_d2_init=config.LAMBDA_D2_INIT,
            beta_init=config.BETA_INIT,
            gamma=config.GAMMA,
            decay_type=config.DECAY_TYPE
        )

        # Critic optimizer (Adam with lr=1e-3)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=config.CRITIC_LR
        )

        # Critic loss function (MSE on TD error)
        self.critic_loss_fn = nn.MSELoss()

        # Metrics tracking
        self.metrics = {
            'episodes': [],
            'episode_rewards': [],
            'episode_lengths': [],
            'success_rate': [],
            'critic_losses': [],
        }

        # Firing rate tracking for visualization
        self.firing_history = {}

        # Learning rate tracking
        self.learning_rate_history = {
            'episodes': [],
            'lambda_d1': [],
            'lambda_d2': [],
            'beta': []
        }

        # Initialize prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=10000,
            success_replay_ratio=10,
            batch_size=32
        )

        # Initialize curriculum learning manager
        self.curriculum = CurriculumManager(
            stages=config.CURRICULUM_STAGES,
            enabled=config.USE_CURRICULUM
        )

        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.RESULTS_DIR, exist_ok=True)
        os.makedirs(config.PLOTS_DIR, exist_ok=True)

    def train_step(self, state, action, reward, next_state, done):
        """
        Single training step (one environment transition)

        Args:
            state: current state
            action: selected action
            reward: immediate reward
            next_state: next state
            done: episode terminated

        Returns:
            td_error: TD error for this transition
            critic_loss: critic network loss
        """
        # Compute TD error: δ(t) = r + γV(s') - V(s)
        td_error, v_current, v_next = self.critic.compute_td_error(
            state, reward, next_state, gamma=self.config.GAMMA
        )

        # Move actor to same device
        self.actor = self.actor.to(self.config.DEVICE)

        # Get neural activities from last forward pass
        # (We need to run forward again to get activities for plasticity)
        _, _, activities = self.actor.forward(state, return_all_activities=True)

        # Apply cortico-striatal plasticity (Equation 3)
        self.plasticity.apply_weight_update(
            actor=self.actor,
            pfc_spikes=activities['pfc'],
            d1_spikes=activities['d1'],
            d2_spikes=activities['d2'],
            td_error=td_error,  # Already a float from compute_td_error
            reward=reward,
            state=state,
            action=action,
            w_max=self.config.W_MAX,
            normalize=False  # Don't normalize during training, only clamp
        )

        # Train Critic network
        # Target: r + γV(s') if not done, else r
        with torch.no_grad():
            if done:
                target = torch.tensor([[reward]], dtype=torch.float32, device=self.config.DEVICE)
            else:
                target_value = reward + self.config.GAMMA * v_next
                target = torch.tensor([[target_value]], dtype=torch.float32, device=self.config.DEVICE)

        # Critic loss: MSE between V(s) and target
        # Need to recompute V(s) with gradients enabled
        prediction = self.critic.forward(state)
        critic_loss = self.critic_loss_fn(prediction, target)

        # Backprop for critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return td_error, critic_loss.item()

    def get_epsilon(self, episode):
        """
        Epsilon decay schedule for exploration

        Start high (0.9) for exploration, decay to low (0.1) over first 1200 episodes
        This ensures early exploration while Φ bonus is still learning action counts

        Args:
            episode: current episode number

        Returns:
            epsilon: probability of random action
        """
        epsilon_start = 0.9
        epsilon_end = 0.1  # Keep higher final epsilon (was 0.05)
        epsilon_decay_episodes = 1200  # Decay over longer period (was 500)

        if episode >= epsilon_decay_episodes:
            return epsilon_end

        # Linear decay
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (episode / epsilon_decay_episodes)
        return epsilon

    def select_action(self, state, epsilon=0.0):
        """
        Select action with epsilon-greedy exploration

        Args:
            state: current state
            epsilon: probability of random action (0.0 = fully exploit)

        Returns:
            action: selected action
            pmc_spikes: PMC spike counts
        """
        if np.random.rand() < epsilon:
            # Random exploration
            action = np.random.randint(0, self.config.N_ACTIONS)
            # Still need to run forward pass for logging/plasticity
            _, pmc_spikes, _ = self.actor.forward(state, return_all_activities=True)
        else:
            # Exploit: use actor's decision (argmax of PMC spikes)
            action, pmc_spikes = self.actor.forward(state)

        return action, pmc_spikes

    def run_episode(self, episode_num, eval_mode=False):
        """
        Run one episode of training or evaluation

        Args:
            episode_num: current episode number
            eval_mode: if True, disable exploration and weight updates

        Returns:
            episode_reward: total reward
            episode_length: number of steps
            success: whether goal was reached
        """
        # Reset environment with curriculum learning (if enabled)
        if not eval_mode and self.curriculum.enabled:
            start_state = self.curriculum.get_start_state(episode_num, self.env)
            state = self.curriculum.reset_env_to_state(self.env, start_state)
        else:
            state, _ = self.env.reset()

        episode_reward = 0
        episode_length = 0
        done = False
        critic_losses = []

        # Collect episode transitions for replay buffer
        episode_transitions = []

        # Get epsilon for this episode
        if not eval_mode:
            epsilon = self.get_epsilon(episode_num)
        else:
            epsilon = self.config.EVAL_EPSILON  # Small exploration during eval to avoid getting stuck

        while not done and episode_length < self.config.MAX_STEPS_PER_EPISODE:
            # Actor selects action (with epsilon-greedy exploration)
            action, pmc_spikes = self.select_action(state, epsilon)

            # Environment step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Store transition for replay buffer
            if not eval_mode:
                episode_transitions.append((state, action, reward, next_state, done))

            # Training step (skip if eval mode)
            if not eval_mode:
                td_error, critic_loss = self.train_step(
                    state, action, reward, next_state, done
                )
                critic_losses.append(critic_loss)

                # Print detailed step info during training only
                if episode_length % 10 == 0:
                    print(f"step {episode_length} | state {state} | action {action} | reward {reward} | done {done} | loss {critic_loss:.5f} | td_error {td_error:.5f}")

            state = next_state

        # Check if goal was reached
        success = (reward > 0)  # FrozenLake gives +1 for reaching goal

        # Add episode to replay buffer and perform replay training
        if not eval_mode and len(episode_transitions) > 0:
            # Add episode to replay buffer
            self.replay_buffer.add_episode(episode_transitions, success=success)

            # Replay training: Sample from buffer and train on successful experiences
            # Do more replay training if we have successful transitions
            if len(self.replay_buffer.successful_transitions) > 0:
                n_replay_batches = 5 if success else 2  # More replay for successful episodes
            else:
                n_replay_batches = 1  # Minimal replay if no successes yet

            for _ in range(n_replay_batches):
                if len(self.replay_buffer) >= 16:  # Only replay if we have enough data
                    replay_batch = self.replay_buffer.sample_recent_and_success(
                        n_recent=8, n_success=8
                    )

                    # Train on replayed transitions
                    for replay_state, replay_action, replay_reward, replay_next_state, replay_done in replay_batch:
                        td_error, critic_loss = self.train_step(
                            replay_state, replay_action, replay_reward,
                            replay_next_state, replay_done
                        )
                        critic_losses.append(critic_loss)

        # Record firing snapshots for visualization states
        if not eval_mode and episode_num % self.config.FIRING_SNAPSHOT_INTERVAL == 0:
            for vis_state in self.config.PLOT_STATES:
                save_firing_snapshot(
                    self.actor, vis_state, self.firing_history, episode_num
                )

        return episode_reward, episode_length, success, np.mean(critic_losses) if critic_losses else 0.0

    def evaluate(self, n_episodes=None):
        """
        Evaluate current policy without exploration

        Args:
            n_episodes: number of episodes to evaluate (default: from config)

        Returns:
            eval_metrics: dict with success_rate, avg_reward, avg_length
        """
        if n_episodes is None:
            n_episodes = self.config.N_EVAL_EPISODES

        eval_rewards = []
        eval_lengths = []
        eval_successes = []

        for _ in range(n_episodes):
            reward, length, success, _ = self.run_episode(0, eval_mode=True)
            eval_rewards.append(reward)
            eval_lengths.append(length)
            eval_successes.append(success)

        return {
            'success_rate': np.mean(eval_successes),
            'avg_reward': np.mean(eval_rewards),
            'avg_length': np.mean(eval_lengths)
        }

    def train(self):
        """
        Main training loop (1000 episodes)
        """
        print("=" * 60)
        print("Training Basal Ganglia Model (Ao et al. 2024)")
        print("=" * 60)
        print(f"Environment: {self.config.ENV_NAME}")
        print(f"Episodes: {self.config.N_EPISODES}")
        print(f"Device: {self.config.DEVICE}")
        print(f"Simulation timesteps T: {self.config.T_SIMULATION}")
        if self.curriculum.enabled:
            print(f"Curriculum Learning: ENABLED ({len(self.curriculum.stages)} stages)")
        else:
            print("Curriculum Learning: DISABLED")
        print("=" * 60)

        for episode in range(self.config.N_EPISODES):
            # Run episode
            episode_reward, episode_length, success, avg_critic_loss = self.run_episode(episode)

            # Record metrics
            self.metrics['episodes'].append(episode)
            self.metrics['episode_rewards'].append(episode_reward)
            self.metrics['episode_lengths'].append(episode_length)
            self.metrics['critic_losses'].append(avg_critic_loss)

            # Decay learning rates
            self.plasticity.decay_learning_rates(episode, self.config.N_EPISODES)

            # Record learning rates
            if episode % 10 == 0:
                lr = self.plasticity.get_learning_rates()
                self.learning_rate_history['episodes'].append(episode)
                self.learning_rate_history['lambda_d1'].append(lr['lambda_d1'])
                self.learning_rate_history['lambda_d2'].append(lr['lambda_d2'])
                self.learning_rate_history['beta'].append(lr['beta'])

            # Logging
            if episode % self.config.LOG_INTERVAL == 0:
                recent_rewards = self.metrics['episode_rewards'][-self.config.LOG_INTERVAL:]
                recent_lengths = self.metrics['episode_lengths'][-self.config.LOG_INTERVAL:]
                epsilon = self.get_epsilon(episode)
                buffer_stats = self.replay_buffer.get_stats()
                curriculum_info = self.curriculum.get_stage_info(episode)
                print(f"Episode {episode}/{self.config.N_EPISODES} | "
                      f"Reward: {np.mean(recent_rewards):.3f} | "
                      f"Length: {np.mean(recent_lengths):.1f} | "
                      f"eps: {epsilon:.3f} | "
                      f"Critic Loss: {avg_critic_loss:.4f} | "
                      f"lambda_D1: {self.plasticity.lambda_d1:.5f} | "
                      f"Replay: {buffer_stats['successful_transitions']}/{buffer_stats['total_transitions']} | "
                      f"{curriculum_info}")

            # Evaluation
            if episode % self.config.EVAL_INTERVAL == 0 and episode > 0:
                eval_metrics = self.evaluate()
                self.metrics['success_rate'].append(eval_metrics['success_rate'])
                print(f"[EVAL] Episode {episode} | "
                      f"Success Rate: {eval_metrics['success_rate']:.2%} | "
                      f"Avg Reward: {eval_metrics['avg_reward']:.3f} | "
                      f"Avg Length: {eval_metrics['avg_length']:.1f}")

            # Save checkpoint
            if episode % self.config.SAVE_INTERVAL == 0 and episode > 0:
                self.save_checkpoint(episode)

        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)

        # Final evaluation
        final_eval = self.evaluate(n_episodes=100)
        print(f"Final Evaluation (100 episodes):")
        print(f"  Success Rate: {final_eval['success_rate']:.2%}")
        print(f"  Avg Reward: {final_eval['avg_reward']:.3f}")
        print(f"  Avg Length: {final_eval['avg_length']:.1f}")

        # Save final model
        self.save_checkpoint(self.config.N_EPISODES, final=True)

        # Generate visualizations
        self.generate_plots()

    def save_checkpoint(self, episode, final=False):
        """
        Save model checkpoint

        Args:
            episode: current episode number
            final: whether this is the final checkpoint
        """
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'plasticity_state': {
                'lambda_d1': self.plasticity.lambda_d1,
                'lambda_d2': self.plasticity.lambda_d2,
                'beta': self.plasticity.beta,
                'action_counts': dict(self.plasticity.action_counts),
                'state_attempts': self.plasticity.state_attempts
            },
            'metrics': self.metrics,
            'config': self.config
        }

        if final:
            path = os.path.join(self.config.CHECKPOINT_DIR, 'final_model.pth')
        else:
            path = os.path.join(self.config.CHECKPOINT_DIR, f'checkpoint_ep{episode}.pth')

        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint

        Args:
            checkpoint_path: path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

        # Restore plasticity state
        pstate = checkpoint['plasticity_state']
        self.plasticity.lambda_d1 = pstate['lambda_d1']
        self.plasticity.lambda_d2 = pstate['lambda_d2']
        self.plasticity.beta = pstate['beta']
        self.plasticity.action_counts = defaultdict(lambda: np.zeros(self.config.N_ACTIONS),
                                                     pstate['action_counts'])
        self.plasticity.state_attempts = pstate['state_attempts']

        self.metrics = checkpoint['metrics']

        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from episode {checkpoint['episode']}")

    def generate_plots(self):
        """
        Generate all visualization plots after training
        """
        print("\nGenerating visualizations...")

        # 1. Firing rate plots (Figure 3 style)
        if self.config.PLOT_FIRING_RATES and len(self.firing_history) > 0:
            firing_path = os.path.join(self.config.PLOTS_DIR, 'firing_rates.png')
            plot_firing_rates(
                self.firing_history,
                states=self.config.PLOT_STATES,
                save_path=firing_path,
                smooth=True
            )

        # 2. Training curves (Figure 4 style)
        curves_path = os.path.join(self.config.PLOTS_DIR, 'training_curves.png')
        plot_training_curves(self.metrics, save_path=curves_path, smooth=True)

        # 3. Weight heatmaps
        weights_path = os.path.join(self.config.PLOTS_DIR, 'weight_heatmaps.png')
        plot_weight_heatmaps(self.actor, save_path=weights_path)

        # 4. Learning rate decay
        lr_path = os.path.join(self.config.PLOTS_DIR, 'learning_rates.png')
        plot_learning_rates(self.learning_rate_history, save_path=lr_path)

        # 5. Action distribution
        action_path = os.path.join(self.config.PLOTS_DIR, 'action_distribution.png')
        plot_action_distribution(self.plasticity.action_counts, save_path=action_path)

        print(f"All plots saved to {self.config.PLOTS_DIR}/")


def main():
    """
    Main entry point for training
    """
    # Print configuration
    Config.print_config()

    # Initialize trainer
    trainer = BasalGangliaTrainer(config=Config)

    # Run training
    trainer.train()

    print("\nTraining session complete!")


if __name__ == '__main__':
    main()
