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


class ReducedDiscreteWrapper(gym.ActionWrapper):
    """Reduces MultiDiscrete([5,3,3,3,4,2]) -> Discrete(360) for DQN.

    Fixes roll to neutral (index 1 = 0 degrees), dropping the roll dimension.
    Resulting nvec: [5, 3, 3, 4, 2] -> 5*3*3*4*2 = 360 actions (vs 1080).
    This gives DQN 3x more samples per action at the same training budget,
    which significantly speeds up Q-value convergence.

    Action layout after reduction:
        [thrust(5), pitch(3), yaw(3), instrument(4), comm(2)]
    Roll is always 1 (neutral, 0 degrees).
    """

    # Index of roll in the original MultiDiscrete: [thrust, pitch, yaw, roll, instrument, comm]
    _ROLL_IDX = 3
    _ROLL_NEUTRAL = 1  # index 1 in ROTATION_DELTAS=[-5, 0, 5] -> 0 degrees

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.action_space, spaces.MultiDiscrete), (
            "ReducedDiscreteWrapper requires a MultiDiscrete action space"
        )
        full_nvec = env.action_space.nvec  # [5, 3, 3, 3, 4, 2]
        # Drop roll dimension
        self.reduced_nvec = np.array(
            [n for i, n in enumerate(full_nvec) if i != self._ROLL_IDX], dtype=np.int64
        )
        self.total_actions = int(np.prod(self.reduced_nvec))
        self.action_space = spaces.Discrete(self.total_actions)

    def action(self, flat_action: int) -> np.ndarray:
        """Decode flat -> reduced multi-action -> full 6-dim action."""
        reduced = np.zeros(len(self.reduced_nvec), dtype=np.int64)
        remaining = flat_action
        for i in range(len(self.reduced_nvec) - 1, -1, -1):
            reduced[i] = remaining % self.reduced_nvec[i]
            remaining //= self.reduced_nvec[i]
        # Reinsert neutral roll at position _ROLL_IDX
        full = np.insert(reduced, self._ROLL_IDX, self._ROLL_NEUTRAL).astype(np.int64)
        return full

    def reverse_action(self, multi_action: np.ndarray) -> int:
        """Encode full 6-dim action -> flat (drops roll)."""
        reduced = np.delete(multi_action, self._ROLL_IDX)
        flat = 0
        multiplier = 1
        for i in range(len(self.reduced_nvec) - 1, -1, -1):
            flat += int(reduced[i]) * multiplier
            multiplier *= self.reduced_nvec[i]
        return flat
