"""Reward calculator for the AstroExploration environment."""

import numpy as np
from astro_env.celestial_bodies import BIOSIGNATURE_REWARDS


class RewardCalculator:
    """Computes rewards based on environment state transitions."""

    def __init__(
        self,
        step_fuel_penalty: float = 0.01,
        step_time_penalty: float = 0.001,
        collision_penalty: float = -1000.0,
        orbital_insertion_bonus: float = 100.0,
        transmission_bonus: float = 50.0,
        proximity_scale: float = 0.1,
    ):
        self.step_fuel_penalty = step_fuel_penalty
        self.step_time_penalty = step_time_penalty
        self.collision_penalty = collision_penalty
        self.orbital_insertion_bonus = orbital_insertion_bonus
        self.transmission_bonus = transmission_bonus
        self.proximity_scale = proximity_scale

    def compute(self, state: dict) -> tuple[float, dict]:
        """Compute total reward and breakdown info dict.

        Args:
            state: Dictionary containing:
                - fuel_used: float, fuel consumed this step
                - collision: bool, whether collision occurred
                - new_biosignatures: list[str], newly detected biosignatures
                - new_transmissions: list[str], newly transmitted biosignatures
                - orbital_insertion: bool, whether orbital insertion achieved
                - min_target_distance: float, distance to nearest target body
                - out_of_bounds: bool

        Returns:
            (total_reward, info_dict) where info_dict breaks down each component.
        """
        info = {}
        total = 0.0

        # Dense step penalty (fuel + time)
        fuel_penalty = -(self.step_fuel_penalty * state.get("fuel_used", 0.0))
        time_penalty = -self.step_time_penalty
        info["reward_step_fuel"] = fuel_penalty
        info["reward_step_time"] = time_penalty
        total += fuel_penalty + time_penalty

        # Sparse: biosignature detection rewards
        for biosig in state.get("new_biosignatures", []):
            reward = BIOSIGNATURE_REWARDS.get(biosig, 0.0)
            info[f"reward_detect_{biosig}"] = reward
            total += reward

        # Sparse: transmission bonus
        for biosig in state.get("new_transmissions", []):
            info[f"reward_transmit_{biosig}"] = self.transmission_bonus
            total += self.transmission_bonus

        # Sparse: orbital insertion
        if state.get("orbital_insertion", False):
            info["reward_orbital_insertion"] = self.orbital_insertion_bonus
            total += self.orbital_insertion_bonus

        # Sparse: collision penalty
        if state.get("collision", False) or state.get("out_of_bounds", False):
            info["reward_collision"] = self.collision_penalty
            total += self.collision_penalty

        # Dense: proximity shaping (small bonus for approaching targets)
        min_dist = state.get("min_target_distance", 50.0)
        if min_dist < 5.0:
            proximity_reward = self.proximity_scale * (1.0 / (min_dist + 0.1) - 1.0 / 5.1)
            proximity_reward = max(0.0, proximity_reward)
            info["reward_proximity"] = proximity_reward
            total += proximity_reward

        info["reward_total"] = total
        return total, info
