"""
Curriculum Learning Manager
Progressively increases task difficulty by expanding starting state space
"""

import numpy as np


class CurriculumManager:
    """
    Manages curriculum learning stages for sparse reward environments.

    Starts with easier tasks (close to goal) and gradually increases difficulty
    by expanding the set of allowed starting states.

    Args:
        stages: List of curriculum stages, each with:
            - 'episodes': Maximum episode number for this stage
            - 'start_states': List of allowed starting states (None = any state)
        enabled: Whether curriculum learning is enabled

    Example stages:
        [
            {'episodes': 500, 'start_states': [10, 11, 14]},  # Close to goal
            {'episodes': 1000, 'start_states': [6, 7, 9, 10, 11, 13, 14]},  # Medium
            {'episodes': 2000, 'start_states': None}  # Full problem
        ]
    """

    def __init__(self, stages, enabled=True):
        self.stages = stages if enabled else []
        self.enabled = enabled
        self.current_stage_idx = 0

    def get_current_stage(self, episode):
        """
        Get current curriculum stage based on episode number

        Args:
            episode: Current episode number

        Returns:
            stage_dict: Current stage configuration
            stage_number: Stage index (0, 1, 2, ...)
        """
        if not self.enabled or len(self.stages) == 0:
            return {'episodes': float('inf'), 'start_states': None}, -1

        # Find which stage we're in
        for idx, stage in enumerate(self.stages):
            if episode < stage['episodes']:
                return stage, idx

        # If past all stages, use final stage
        return self.stages[-1], len(self.stages) - 1

    def get_start_state(self, episode, env):
        """
        Sample a starting state according to current curriculum stage

        Args:
            episode: Current episode number
            env: Gymnasium environment

        Returns:
            start_state: State index to start episode from (as Python int)
        """
        if not self.enabled:
            # No curriculum: use environment's default reset
            state, _ = env.reset()
            return int(state)

        stage, stage_idx = self.get_current_stage(episode)

        if stage['start_states'] is None:
            # Full problem: any starting state
            state, _ = env.reset()
            return int(state)
        else:
            # Curriculum: sample from allowed states
            start_state = int(np.random.choice(stage['start_states']))
            return start_state

    def reset_env_to_state(self, env, state):
        """
        Reset environment to a specific state

        Args:
            env: Gymnasium environment
            state: Target state index

        Returns:
            state: Actual state after reset (as Python int)
        """
        # Reset environment first
        env.reset()

        # For FrozenLake, we can directly set the state
        # This is a bit of a hack but works for discrete state spaces
        if hasattr(env, 'unwrapped'):
            env.unwrapped.s = int(state)
        else:
            env.s = int(state)

        return int(state)

    def get_stage_info(self, episode):
        """
        Get human-readable info about current stage

        Args:
            episode: Current episode number

        Returns:
            info_str: Stage information string
        """
        if not self.enabled:
            return "Curriculum: Disabled"

        stage, stage_idx = self.get_current_stage(episode)

        if stage['start_states'] is None:
            states_str = "Any state"
        else:
            states_str = f"{len(stage['start_states'])} states: {stage['start_states']}"

        return f"Stage {stage_idx + 1}/{len(self.stages)} | {states_str}"

    def is_final_stage(self, episode):
        """Check if we've reached the final curriculum stage"""
        if not self.enabled:
            return True

        stage, stage_idx = self.get_current_stage(episode)
        return stage_idx == len(self.stages) - 1
