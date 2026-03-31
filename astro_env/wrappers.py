"""Action space wrappers for compatibility with different RL algorithms."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class FlattenMultiDiscreteToDiscrete(gym.ActionWrapper):
    """Converts a MultiDiscrete action space to a single Discrete space.

    Required for DQN which only supports Discrete action spaces.
    Uses mixed-radix encoding to map between flat integer and multi-action array.

    For MultiDiscrete([5, 3, 3, 3, 4, 2]):
        Total combinations = 5 * 3 * 3 * 3 * 4 * 2 = 1080
        Flat action 0 -> [0, 0, 0, 0, 0, 0]
        Flat action 1079 -> [4, 2, 2, 2, 3, 1]
    """

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete), (
            "FlattenMultiDiscreteToDiscrete requires a MultiDiscrete action space"
        )
        self.nvec = env.action_space.nvec
        self.total_actions = int(np.prod(self.nvec))
        self.action_space = spaces.Discrete(self.total_actions)

    def action(self, flat_action: int) -> np.ndarray:
        """Decode a flat integer action to a multi-action array."""
        multi_action = np.zeros(len(self.nvec), dtype=np.int64)
        remaining = flat_action
        for i in range(len(self.nvec) - 1, -1, -1):
            multi_action[i] = remaining % self.nvec[i]
            remaining //= self.nvec[i]
        return multi_action

    def reverse_action(self, multi_action: np.ndarray) -> int:
        """Encode a multi-action array to a flat integer."""
        flat = 0
        multiplier = 1
        for i in range(len(self.nvec) - 1, -1, -1):
            flat += int(multi_action[i]) * multiplier
            multiplier *= self.nvec[i]
        return flat
