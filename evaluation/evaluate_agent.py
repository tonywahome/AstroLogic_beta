"""Evaluate a trained agent on the AstroExploration environment."""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import astro_env  # noqa: F401
import gymnasium as gym
import torch
from stable_baselines3 import DQN, PPO
from astro_env.wrappers import FlattenMultiDiscreteToDiscrete
from agents.reinforce_policy import REINFORCEPolicy


def evaluate_sb3_model(model_path: str, algorithm: str, n_episodes: int = 10,
                       render: bool = False) -> list:
    """Evaluate an SB3 model (DQN or PPO)."""
    render_mode = "human" if render else None
    env = gym.make("AstroExploration-v0", render_mode=render_mode)

    if algorithm == "DQN":
        env = FlattenMultiDiscreteToDiscrete(env)
        model = DQN.load(model_path)
    elif algorithm == "PPO":
        model = PPO.load(model_path)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episodes.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "biosig_found": len(info.get("biosig_found", [])),
            "biosig_transmitted": len(info.get("biosig_transmitted", [])),
            "success": info.get("success", False),
            "final_fuel": info.get("fuel", 0),
            "final_battery": info.get("battery", 0),
        })
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, "
              f"found={episodes[-1]['biosig_found']}, tx={episodes[-1]['biosig_transmitted']}")

    env.close()
    return episodes


def evaluate_reinforce_model(model_path: str, n_episodes: int = 10,
                             hidden_sizes: list = None, render: bool = False) -> list:
    """Evaluate a trained REINFORCE policy."""
    render_mode = "human" if render else None
    env = gym.make("AstroExploration-v0", render_mode=render_mode)

    policy = REINFORCEPolicy(
        obs_dim=23,
        action_nvec=[5, 3, 3, 3, 4, 2],
        hidden_sizes=hidden_sizes or [128, 64],
    )
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()

    episodes = []
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs)
                action, _ = policy.get_action(obs_tensor)
            obs, reward, terminated, truncated, info = env.step(action)
            if render:
                env.render()
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episodes.append({
            "episode": ep,
            "reward": total_reward,
            "length": steps,
            "biosig_found": len(info.get("biosig_found", [])),
            "biosig_transmitted": len(info.get("biosig_transmitted", [])),
            "success": info.get("success", False),
            "final_fuel": info.get("fuel", 0),
            "final_battery": info.get("battery", 0),
        })
        print(f"  Episode {ep+1}: reward={total_reward:.2f}, steps={steps}, "
              f"found={episodes[-1]['biosig_found']}, tx={episodes[-1]['biosig_transmitted']}")

    env.close()
    return episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--algorithm", type=str, required=True,
                        choices=["DQN", "PPO", "REINFORCE"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--output", type=str, default=None, help="Save results to JSON")
    args = parser.parse_args()

    print(f"\nEvaluating {args.algorithm} model: {args.model}")
    print(f"Running {args.episodes} episodes...\n")

    if args.algorithm == "REINFORCE":
        results = evaluate_reinforce_model(args.model, args.episodes, render=args.render)
    else:
        results = evaluate_sb3_model(args.model, args.algorithm, args.episodes, args.render)

    # Summary
    rewards = [e["reward"] for e in results]
    lengths = [e["length"] for e in results]
    found = [e["biosig_found"] for e in results]
    tx = [e["biosig_transmitted"] for e in results]

    print(f"\n{'='*50}")
    print(f"Evaluation Summary ({args.episodes} episodes)")
    print(f"{'='*50}")
    print(f"Mean Reward:     {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"Mean Length:     {np.mean(lengths):.0f} +/- {np.std(lengths):.0f}")
    print(f"Mean Biosig Found: {np.mean(found):.1f}")
    print(f"Mean Biosig TX:    {np.mean(tx):.1f}")
    print(f"Success Rate:    {sum(e['success'] for e in results)}/{len(results)}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
