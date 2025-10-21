"""
Quick sanity test for training loop
Runs 50 episodes to verify all components work together
"""

import torch
import sys

# Override config for quick testing
from config import Config

# Reduce episodes for quick test
Config.N_EPISODES = 50
Config.LOG_INTERVAL = 5
Config.EVAL_INTERVAL = 20
Config.SAVE_INTERVAL = 25
Config.FIRING_SNAPSHOT_INTERVAL = 10

from trainer import BasalGangliaTrainer

def test_training():
    """Quick sanity check for training"""
    print("=" * 60)
    print("Sanity Test: Training Loop (50 episodes)")
    print("=" * 60)

    try:
        # Initialize trainer
        trainer = BasalGangliaTrainer(config=Config)
        print("[PASS] Trainer initialized successfully")

        # Test single episode
        print("\nTesting single episode...")
        episode_reward, episode_length, success, critic_loss = trainer.run_episode(0)
        print(f"[PASS] Episode completed:")
        print(f"  Reward: {episode_reward}, Length: {episode_length}, Success: {success}")
        print(f"  Critic Loss: {critic_loss:.4f}")

        # Test full training
        print("\nRunning 50-episode training...")
        trainer.train()

        print("\n[SUCCESS] All training components working!")
        print(f"Final metrics:")
        print(f"  Total episodes: {len(trainer.metrics['episodes'])}")
        print(f"  Average reward (last 10): {sum(trainer.metrics['episode_rewards'][-10:])/10:.3f}")

        return True

    except Exception as e:
        print(f"\n[FAIL] Training test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_training()
    sys.exit(0 if success else 1)
