"""
Prioritized Experience Replay Buffer
Stores and prioritizes successful transitions for sparse reward environments
"""

import numpy as np
from collections import deque
import random


class PrioritizedReplayBuffer:
    """
    Experience replay buffer with prioritization for successful episodes.

    In sparse reward environments like FrozenLake, successful episodes are rare
    and valuable. This buffer stores all transitions but samples successful ones
    more frequently to accelerate learning.

    Args:
        capacity: Maximum number of transitions to store
        success_replay_ratio: How many times more to replay successful transitions (default: 10)
        batch_size: Number of transitions to sample per batch
    """

    def __init__(self, capacity=10000, success_replay_ratio=10, batch_size=32):
        self.capacity = capacity
        self.success_replay_ratio = success_replay_ratio
        self.batch_size = batch_size

        # Storage for transitions
        self.buffer = deque(maxlen=capacity)

        # Separate tracking for successful transitions (reached goal)
        self.successful_transitions = deque(maxlen=capacity // 2)

        # Statistics
        self.n_successes = 0
        self.n_total = 0

    def add_episode(self, transitions, success=False):
        """
        Add an entire episode to the buffer

        Args:
            transitions: List of (state, action, reward, next_state, done) tuples
            success: Whether this episode reached the goal
        """
        for transition in transitions:
            self.buffer.append(transition)

            if success:
                # Store successful transitions separately for prioritized sampling
                self.successful_transitions.append(transition)

        self.n_total += len(transitions)
        if success:
            self.n_successes += 1

    def add_transition(self, state, action, reward, next_state, done):
        """
        Add a single transition

        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state
            done: episode terminated
        """
        transition = (state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.n_total += 1

        # If this transition has positive reward, it's valuable
        if reward > 0:
            self.successful_transitions.append(transition)
            self.n_successes += 1

    def sample(self, batch_size=None):
        """
        Sample a batch of transitions with prioritization

        Returns 50% from successful transitions and 50% from general buffer
        to balance learning from successes while maintaining diversity

        Args:
            batch_size: Number of transitions to sample (default: self.batch_size)

        Returns:
            List of (state, action, reward, next_state, done) tuples
        """
        if batch_size is None:
            batch_size = self.batch_size

        if len(self.buffer) < batch_size:
            # Not enough transitions yet, return all we have
            return list(self.buffer)

        # Calculate how many to sample from each pool
        if len(self.successful_transitions) > 0:
            # Sample 70% from successful transitions, 30% from general buffer
            # This heavily prioritizes successful experiences
            n_success_samples = min(
                int(batch_size * 0.7),
                len(self.successful_transitions)
            )
            n_general_samples = batch_size - n_success_samples

            success_batch = random.sample(list(self.successful_transitions), n_success_samples)
            general_batch = random.sample(list(self.buffer), n_general_samples)

            batch = success_batch + general_batch
        else:
            # No successful transitions yet, sample from general buffer
            batch = random.sample(list(self.buffer), batch_size)

        return batch

    def sample_recent_and_success(self, n_recent=16, n_success=16):
        """
        Sample a mix of recent transitions and successful transitions

        This ensures the model learns from both:
        - Recent experiences (current policy behavior)
        - Successful experiences (goal-reaching behavior)

        Args:
            n_recent: Number of recent transitions to include
            n_success: Number of successful transitions to include

        Returns:
            List of transitions
        """
        batch = []

        # Get recent transitions
        recent_transitions = list(self.buffer)[-n_recent:] if len(self.buffer) >= n_recent else list(self.buffer)
        batch.extend(recent_transitions)

        # Get successful transitions
        if len(self.successful_transitions) > 0:
            n_success_actual = min(n_success, len(self.successful_transitions))
            success_batch = random.sample(list(self.successful_transitions), n_success_actual)
            batch.extend(success_batch)

        return batch

    def __len__(self):
        return len(self.buffer)

    def get_stats(self):
        """Return buffer statistics"""
        return {
            'total_transitions': len(self.buffer),
            'successful_transitions': len(self.successful_transitions),
            'total_added': self.n_total,
            'success_rate': len(self.successful_transitions) / max(1, len(self.buffer))
        }

    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.successful_transitions.clear()
        self.n_successes = 0
        self.n_total = 0
