"""
Visualization utilities for Basal Ganglia Model
Recreates Figure 3 and Figure 4 from Ao et al. 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os


def plot_firing_rates(firing_history, states=[0, 8], save_path=None, smooth=True):
    """
    Recreate Figure 3 from paper: Firing rates of D1-MSN, D2-MSN, and PMC over training

    Args:
        firing_history: dict with structure:
            {state: {
                'steps': [episode_numbers],
                'd1_firing': [[d1_0, d1_1, d1_2, d1_3], ...],  # spike counts per neuron
                'd2_firing': [[d2_0, d2_1, d2_2, d2_3], ...],
                'pmc_firing': [[pmc_0, pmc_1, pmc_2, pmc_3], ...]
            }}
        states: list of states to visualize (default: [0, 8] as in paper)
        save_path: path to save figure (default: None, just display)
        smooth: whether to apply Savitzky-Golay smoothing (default: True)
    """
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Activity Status of D1-MSNs, D2-MSNs and PMC (Ao et al. 2024 - Figure 3)',
                 fontsize=14, fontweight='bold')

    neuron_types = ['d1', 'd2', 'pmc']
    neuron_labels = ['D1-MSN', 'D2-MSN', 'PMC']
    colors = ['blue', 'green', 'red', 'orange']

    for row, (neuron_type, label) in enumerate(zip(neuron_types, neuron_labels)):
        for col, state in enumerate(states):
            ax = axes[row, col]

            if state not in firing_history:
                ax.text(0.5, 0.5, f'No data for state {state}',
                       ha='center', va='center')
                ax.set_title(f'{label} - State {state}')
                continue

            data = firing_history[state]
            steps = np.array(data['steps'])
            firing_key = f'{neuron_type}_firing'

            if firing_key not in data:
                continue

            firing = np.array(data[firing_key])  # [num_snapshots, n_actions]

            # Plot each action neuron
            for action_idx in range(firing.shape[1]):
                y = firing[:, action_idx]

                # Apply smoothing if requested
                if smooth and len(y) > 11:  # Need at least 11 points for savgol
                    window_length = min(11, len(y) if len(y) % 2 == 1 else len(y) - 1)
                    if window_length >= 5:  # Minimum for polyorder=3
                        y = savgol_filter(y, window_length, 3)

                ax.plot(steps, y, label=f'{label}{action_idx+1}',
                       color=colors[action_idx], linewidth=2, alpha=0.8)

            ax.set_xlabel('Step', fontsize=10)
            ax.set_ylabel('Firing rate', fontsize=10)
            ax.set_title(f'{label} - State {state}', fontsize=11)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Firing rate plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_training_curves(metrics, save_path=None, smooth=True):
    """
    Recreate Figure 4 from paper: Training curves (episodic length and reward)

    Args:
        metrics: dict with keys:
            'episodes': [episode numbers]
            'episode_lengths': [length per episode]
            'episode_rewards': [reward per episode]
            'success_rate': [success rate over time] (optional)
        save_path: path to save figure
        smooth: whether to apply Savitzky-Golay smoothing
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves (Ao et al. 2024 - Figure 4 style)',
                 fontsize=14, fontweight='bold')

    episodes = np.array(metrics['episodes'])
    episode_lengths = np.array(metrics['episode_lengths'])
    episode_rewards = np.array(metrics['episode_rewards'])

    # Apply smoothing
    if smooth and len(episodes) > 11:
        window_length = min(51, len(episodes) if len(episodes) % 2 == 1 else len(episodes) - 1)
        if window_length >= 5:
            episode_lengths_smooth = savgol_filter(episode_lengths, window_length, 3)
            episode_rewards_smooth = savgol_filter(episode_rewards, window_length, 3)
        else:
            episode_lengths_smooth = episode_lengths
            episode_rewards_smooth = episode_rewards
    else:
        episode_lengths_smooth = episode_lengths
        episode_rewards_smooth = episode_rewards

    # Plot episodic length
    axes[0].plot(episodes, episode_lengths, alpha=0.3, color='blue', linewidth=0.5)
    axes[0].plot(episodes, episode_lengths_smooth, color='blue', linewidth=2, label='Our model')
    axes[0].set_xlabel('Episode', fontsize=12)
    axes[0].set_ylabel('Episodic length', fontsize=12)
    axes[0].set_title('(a) Episode Length', fontsize=11)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot reward
    axes[1].plot(episodes, episode_rewards, alpha=0.3, color='green', linewidth=0.5)
    axes[1].plot(episodes, episode_rewards_smooth, color='green', linewidth=2, label='Our model')
    axes[1].set_xlabel('Episode', fontsize=12)
    axes[1].set_ylabel('Reward', fontsize=12)
    axes[1].set_title('(b) Reward', fontsize=11)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_weight_heatmaps(actor, save_path=None):
    """
    Visualize cortico-striatal weight matrices as heatmaps

    Args:
        actor: BasalGangliaActor instance
        save_path: path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Cortico-Striatal Weight Matrices', fontsize=14, fontweight='bold')

    # W_PFC_D1
    w_d1 = actor.w_pfc_d1.detach().cpu().numpy()
    im1 = axes[0].imshow(w_d1, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[0].set_title('W_PFC_D1 (Direct Pathway)', fontsize=11)
    axes[0].set_xlabel('D1-MSN neurons', fontsize=10)
    axes[0].set_ylabel('PFC neurons (states)', fontsize=10)
    axes[0].set_xticks(range(w_d1.shape[1]))
    axes[0].set_yticks(range(0, w_d1.shape[0], 2))
    plt.colorbar(im1, ax=axes[0], label='Weight')

    # W_PFC_D2
    w_d2 = actor.w_pfc_d2.detach().cpu().numpy()
    im2 = axes[1].imshow(w_d2, aspect='auto', cmap='viridis', interpolation='nearest')
    axes[1].set_title('W_PFC_D2 (Indirect Pathway)', fontsize=11)
    axes[1].set_xlabel('D2-MSN neurons', fontsize=10)
    axes[1].set_ylabel('PFC neurons (states)', fontsize=10)
    axes[1].set_xticks(range(w_d2.shape[1]))
    axes[1].set_yticks(range(0, w_d2.shape[0], 2))
    plt.colorbar(im2, ax=axes[1], label='Weight')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Weight heatmaps saved to {save_path}")
    else:
        plt.show()

    plt.close()


def save_firing_snapshot(actor, state, firing_history, episode):
    """
    Record a snapshot of neuron firing rates for a given state

    Args:
        actor: BasalGangliaActor instance
        state: int, state to record
        firing_history: dict to update with firing data
        episode: current episode number
    """
    # Get neural activities
    action, pmc_spikes, activities = actor.forward(state, return_all_activities=True)

    # Initialize history for this state if needed
    if state not in firing_history:
        firing_history[state] = {
            'steps': [],
            'd1_firing': [],
            'd2_firing': [],
            'pmc_firing': []
        }

    # Record firing rates (convert spike counts to rates or keep as counts)
    # Paper Figure 3 shows firing rates over training steps
    firing_history[state]['steps'].append(episode)
    firing_history[state]['d1_firing'].append(activities['d1'].detach().cpu().numpy())
    firing_history[state]['d2_firing'].append(activities['d2'].detach().cpu().numpy())
    firing_history[state]['pmc_firing'].append(pmc_spikes.detach().cpu().numpy())


def plot_learning_rates(learning_rate_history, save_path=None):
    """
    Plot learning rate decay over episodes

    Args:
        learning_rate_history: dict with keys:
            'episodes': [episode numbers]
            'lambda_d1': [lambda_d1 values]
            'lambda_d2': [lambda_d2 values]
            'beta': [beta values]
        save_path: path to save figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    episodes = learning_rate_history['episodes']

    ax.plot(episodes, learning_rate_history['lambda_d1'],
           label='λ_D1 (D1 learning rate)', linewidth=2, color='blue')
    ax.plot(episodes, learning_rate_history['lambda_d2'],
           label='λ_D2 (D2 learning rate)', linewidth=2, color='green')
    ax.plot(episodes, learning_rate_history['beta'],
           label='β (exploration bonus)', linewidth=2, color='red')

    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Decay', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Learning rate plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_action_distribution(action_counts, save_path=None):
    """
    Plot action selection frequency per state

    Args:
        action_counts: dict[state][action] = count
        save_path: path to save figure
    """
    states = sorted(action_counts.keys())
    n_states = len(states)
    n_actions = 4

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Create heatmap of action selections
    action_matrix = np.zeros((n_states, n_actions))
    for i, state in enumerate(states):
        for action in range(n_actions):
            action_matrix[i, action] = action_counts[state][action]

    im = ax.imshow(action_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Action', fontsize=12)
    ax.set_ylabel('State', fontsize=12)
    ax.set_title('Action Selection Distribution', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_actions))
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(states)
    plt.colorbar(im, ax=ax, label='Count')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Action distribution plot saved to {save_path}")
    else:
        plt.show()

    plt.close()
