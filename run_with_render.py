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


def _infer_reinforce_policy_config(state_dict: dict) -> tuple[int, list[int], list[int]]:
    """Infer obs_dim, action_nvec, and hidden_sizes from a saved REINFORCE checkpoint."""
    backbone_weight_keys = sorted(
        (key for key in state_dict if key.startswith("backbone.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    hidden_sizes = [state_dict[key].shape[0] for key in backbone_weight_keys]
    obs_dim = state_dict[backbone_weight_keys[0]].shape[1]

    head_weight_keys = sorted(
        (key for key in state_dict if key.startswith("heads.") and key.endswith(".weight")),
        key=lambda key: int(key.split(".")[1]),
    )
    action_nvec = [state_dict[key].shape[0] for key in head_weight_keys]

    return obs_dim, action_nvec, hidden_sizes


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

    state_dict = torch.load(model_path, weights_only=True)
    obs_dim, action_nvec, inferred_hidden_sizes = _infer_reinforce_policy_config(state_dict)

    policy = REINFORCEPolicy(
        obs_dim=obs_dim,
        action_nvec=action_nvec,
        hidden_sizes=hidden_sizes or inferred_hidden_sizes,
    )
    policy.load_state_dict(state_dict)
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
