#!/usr/bin/env python
"""Run a trained agent with pygame rendering in the foreground."""

import sys
import os
import argparse

# Ensure proper path setup for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO
from agents.reinforce_policy import REINFORCEPolicy
import environment  # noqa: F401
from environment.custom_env import FlattenMultiDiscreteToDiscrete


def run_sb3_model(model_path: str, algorithm: str, n_episodes: int = 5):
    """Run an SB3 model (DQN or PPO) with rendering."""
    print(f"\nLoading {algorithm} model from: {model_path}")

    # Initialize environment with rendering
    env = gym.make("AstroExploration-v0", render_mode="human")

    if algorithm == "DQN":
        env = FlattenMultiDiscreteToDiscrete(env)
        model = DQN.load(model_path)
    elif algorithm == "PPO":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    print(f"Running {n_episodes} episodes with rendering...\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        print(f"Episode {ep + 1}/{n_episodes}", end="", flush=True)

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()  # Render each frame
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f" | Reward: {total_reward:.2f} | Steps: {steps} | "
              f"Found: {len(info.get('biosig_found', []))} | "
              f"TX: {len(info.get('biosig_transmitted', []))}")

    env.close()
    print("\nRendering complete!")


def run_reinforce_model(model_path: str, n_episodes: int = 5,
                        hidden_sizes: list = None):
    """Run a REINFORCE policy with rendering."""
    print(f"\nLoading REINFORCE model from: {model_path}")

    # Initialize environment with rendering
    env = gym.make("AstroExploration-v0", render_mode="human")

    policy = REINFORCEPolicy(
        obs_dim=23,
        action_nvec=[5, 3, 3, 3, 4, 2],
        hidden_sizes=hidden_sizes or [128, 64],
    )
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    print(f"Running {n_episodes} episodes with rendering...\n")

    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        print(f"Episode {ep + 1}/{n_episodes}", end="", flush=True)

        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                action, _ = policy.get_action(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()  # Render each frame
            total_reward += reward
            steps += 1
            done = terminated or truncated

        print(f" | Reward: {total_reward:.2f} | Steps: {steps} | "
              f"Found: {len(info.get('biosig_found', []))} | "
              f"TX: {len(info.get('biosig_transmitted', []))}")

    env.close()
    print("\nRendering complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run trained agent with pygame rendering"
    )
    parser.add_argument("--model", type=str, required=True,
                        help="Path to model file")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["DQN", "PPO", "REINFORCE"])
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    if args.algorithm == "REINFORCE":
        run_reinforce_model(args.model, args.episodes)
    else:
        run_sb3_model(args.model, args.algorithm, args.episodes)
