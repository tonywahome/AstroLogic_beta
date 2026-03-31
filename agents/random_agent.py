"""Random agent demo - spacecraft taking random actions without any model.

This script demonstrates the AstroExploration environment visualization
with a random policy. No training is involved.

Usage:
    python -m agents.random_agent
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astro_env  # noqa: F401 - registers the environment
import gymnasium as gym
from astro_env.celestial_bodies import INSTRUMENTS


def main():
    print("=" * 60)
    print("AstroLogic - Random Agent Demo")
    print("Spacecraft taking random actions (no trained model)")
    print("=" * 60)

    env = gym.make("AstroExploration-v0", render_mode="human")
    obs, info = env.reset(seed=42)

    print(f"\nEnvironment: AstroExploration-v0")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"\nStarting position: ({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}) AU")
    print(f"Initial fuel: {info['fuel']:.1%}")
    print(f"Initial battery: {info['battery']:.1%}")
    print(f"\nRunning random agent for up to 2000 steps...")
    print(f"Close the pygame window to stop early.\n")

    total_reward = 0.0
    max_steps = 2000

    for step in range(1, max_steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the environment
        env.render()

        # Print status every 100 steps
        if step % 100 == 0:
            instrument = INSTRUMENTS[action[4]]["name"]
            comm = "TX" if action[5] == 1 else "OFF"
            print(
                f"Step {step:4d} | "
                f"Reward: {reward:+8.4f} | "
                f"Fuel: {info['fuel']:5.1%} | "
                f"Batt: {info['battery']:5.1%} | "
                f"Inst: {instrument:14s} | "
                f"Comm: {comm:3s} | "
                f"Found: {len(info['biosig_found'])} | "
                f"TX: {len(info['biosig_transmitted'])}"
            )

        if terminated or truncated:
            reason = "SUCCESS" if info.get("success") else (
                "COLLISION" if info.get("collision") else (
                    "OUT OF BOUNDS" if info.get("out_of_bounds") else (
                        "RESOURCES DEPLETED" if info.get("resource_depleted") else
                        "MAX STEPS"
                    )
                )
            )
            print(f"\n--- Episode ended: {reason} ---")
            break

    print(f"\n{'=' * 60}")
    print(f"Episode Summary")
    print(f"{'=' * 60}")
    print(f"Total steps:              {step}")
    print(f"Total reward:             {total_reward:.4f}")
    print(f"Final fuel:               {info['fuel']:.1%}")
    print(f"Final battery:            {info['battery']:.1%}")
    print(f"Biosignatures found:      {info['biosig_found']}")
    print(f"Biosignatures transmitted: {info['biosig_transmitted']}")
    print(f"Final position:           ({obs[0]:.3f}, {obs[1]:.3f}, {obs[2]:.3f}) AU")
    print(f"{'=' * 60}")

    env.close()


if __name__ == "__main__":
    main()
