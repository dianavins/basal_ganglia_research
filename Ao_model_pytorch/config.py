"""
Configuration and Hyperparameters for Ao et al. 2024 Basal Ganglia Model
All parameters from the paper
"""

import torch


class Config:
    """Hyperparameters for the Basal Ganglia Actor-Critic model"""

    # ==================== Environment ====================
    ENV_NAME = 'FrozenLake-v1'
    N_STATES = 16  # 4x4 grid
    N_ACTIONS = 4  # Left, Down, Right, Up
    IS_SLIPPERY = False  # Deterministic (False) or stochastic (True)
    RENDER_MODE = None  # None, 'human', or 'rgb_array'

    # ==================== LIF Neuron Parameters (Section 4.1) ====================
    TAU = 1.0  # Time constant
    V_RESET = 0.0  # Reset potential
    V_TH = 1.0  # Firing threshold
    DT = 1.0  # Timestep (ms)

    # ==================== Network Architecture ====================
    # PFC: 16 neurons (state encoding)
    N_PFC = 16
    # Striatum: 4 D1-MSNs + 4 D2-MSNs
    N_D1_MSN = 4
    N_D2_MSN = 4
    # Other BG nuclei: 4 each (one per action channel)
    N_GPE = 4
    N_STN = 4
    N_GPI = 4
    N_THAL = 4
    N_PMC = 4

    # Critic architecture
    CRITIC_HIDDEN_DIM = 64

    # ==================== Temporal Dynamics ====================
    T_SIMULATION = 500  # Simulation timesteps per decision (from paper Fig. 3)

    # ==================== Fixed Pathway Weights ====================
    # Direct pathway
    W_D1_GPI = -1.0  # Inhibitory
    # Indirect pathway
    W_D2_GPE = -1.0  # Inhibitory
    W_GPE_STN = -0.8  # Inhibitory
    W_STN_GPI = 1.2  # Excitatory
    # Output pathway
    W_GPI_THAL = -1.0  # Inhibitory
    W_THAL_PMC = 1.0  # Excitatory

    # ==================== Cortico-Striatal Weight Initialization ====================
    # Template values (Section 4.1, Equation 5)
    W1_TEMPLATE = [0.20, 0.12, 0.07, 0.03]  # Descending (for D1)
    W2_TEMPLATE = [0.03, 0.07, 0.12, 0.20]  # Ascending (for D2, reversed W1)
    INIT_NOISE_STD = 0.01  # Gaussian noise std
    TARGET_L1_NORM = 0.42  # L1 normalization target per row

    # Weight constraints
    W_MIN = 0.0
    W_MAX = 0.4

    # ==================== Plasticity Learning Rates (Section 4.1) ====================
    ALPHA = 0.19  # Direct reward modulation (fixed)
    LAMBDA_D1_INIT = 0.01  # Initial D1 pathway learning rate
    LAMBDA_D2_INIT = 0.01  # Initial D2 pathway learning rate
    BETA_INIT = 0.05  # Initial exploration bonus weight
    GAMMA = 0.99  # TD discount factor

    DECAY_TYPE = 'exponential'  # 'exponential' or 'cosine'

    # ==================== Training Parameters ====================
    N_EPISODES = 2000  # Total training episodes (increased for sparse rewards)
    MAX_STEPS_PER_EPISODE = 100  # Max steps per episode

    # Critic training
    CRITIC_LR = 1e-3  # Adam learning rate for Critic (increased for faster learning)
    CRITIC_TRAIN_FREQ = 1  # Train critic every N steps

    # ==================== Curriculum Learning ====================
    USE_CURRICULUM = True  # Enable/disable curriculum learning
    # Curriculum stages: gradually increase difficulty by expanding start states
    # Each stage: {'episodes': max_episode_for_stage, 'start_states': [allowed states]}
    # start_states=None means any state (full problem)
    CURRICULUM_STAGES = [
        {'episodes': 500, 'start_states': [10, 11, 14]},  # Stage 1: Very close to goal (1-2 steps)
        {'episodes': 1000, 'start_states': [6, 7, 9, 10, 11, 13, 14]},  # Stage 2: Medium distance (2-4 steps)
        {'episodes': 2000, 'start_states': None}  # Stage 3: Full problem (any starting state)
    ]

    # ==================== Logging and Evaluation ====================
    LOG_INTERVAL = 10  # Log every N episodes
    EVAL_INTERVAL = 50  # Evaluate every N episodes
    SAVE_INTERVAL = 100  # Save model every N episodes
    N_EVAL_EPISODES = 10  # Number of episodes for evaluation

    # Visualization
    PLOT_FIRING_RATES = True  # Plot neuron firing rates (like Figure 3)
    PLOT_STATES = [0, 8]  # States to visualize (from paper Figure 3)
    FIRING_SNAPSHOT_INTERVAL = 50  # Save firing rates every N episodes

    # ==================== Paths ====================
    CHECKPOINT_DIR = './checkpoints'
    RESULTS_DIR = './results'
    PLOTS_DIR = './plots'

    # ==================== Device ====================
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================== Random Seed ====================
    SEED = 42

    @classmethod
    def print_config(cls):
        """Print all configuration parameters"""
        print("=" * 60)
        print("Basal Ganglia Model Configuration (Ao et al. 2024)")
        print("=" * 60)
        print(f"Environment: {cls.ENV_NAME}")
        print(f"States: {cls.N_STATES}, Actions: {cls.N_ACTIONS}")
        print(f"Device: {cls.DEVICE}")
        print()
        print("LIF Parameters:")
        print(f"  τ={cls.TAU}, V_reset={cls.V_RESET}, V_th={cls.V_TH}, dt={cls.DT}")
        print(f"  Simulation timesteps T={cls.T_SIMULATION}")
        print()
        print("Network Architecture:")
        print(f"  PFC: {cls.N_PFC}, D1-MSN: {cls.N_D1_MSN}, D2-MSN: {cls.N_D2_MSN}")
        print(f"  GPe: {cls.N_GPE}, STN: {cls.N_STN}, GPi: {cls.N_GPI}")
        print(f"  Thalamus: {cls.N_THAL}, PMC: {cls.N_PMC}")
        print(f"  Critic hidden: {cls.CRITIC_HIDDEN_DIM}")
        print()
        print("Learning Parameters:")
        print(f"  α={cls.ALPHA}, λ_D1={cls.LAMBDA_D1_INIT}, λ_D2={cls.LAMBDA_D2_INIT}")
        print(f"  β={cls.BETA_INIT}, γ={cls.GAMMA}")
        print(f"  Decay type: {cls.DECAY_TYPE}")
        print(f"  Critic LR: {cls.CRITIC_LR}")
        print()
        print("Training:")
        print(f"  Episodes: {cls.N_EPISODES}")
        print(f"  Max steps per episode: {cls.MAX_STEPS_PER_EPISODE}")
        print("=" * 60)


if __name__ == '__main__':
    # Print configuration when run as script
    Config.print_config()
