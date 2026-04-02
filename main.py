"""Run the best trained model with pygame visualization."""

import os, glob, argparse
import numpy as np
import gymnasium as gym

import environment  # noqa: F401  — registers AstroExploration-v0
from environment.custom_env import FlattenMultiDiscreteToDiscrete


def find_best_model(algorithm):
    """Walk model directories and return the path to the best saved model."""
    if algorithm == "dqn":
        search_dir = "models/dqn"
    else:
        search_dir = "models/pg"

    best_path = None
    for root, dirs, files in os.walk(search_dir):
        if "best_model.zip" in files:
            best_path = os.path.join(root, "best_model.zip")
            break
        if "final_model.zip" in files and best_path is None:
            best_path = os.path.join(root, "final_model.zip")
        if "policy.pt" in files and best_path is None:
            best_path = os.path.join(root, "policy.pt")

    if best_path is None:
        raise FileNotFoundError(
            f"No trained model found in {search_dir}/. Train first.")
    return best_path


def run_episodes(algorithm, model_path, n_episodes=3):
    """Load a trained model and run episodes with rendering."""
    env = gym.make("AstroExploration-v0", render_mode="human")

    if algorithm == "dqn":
        from stable_baselines3 import DQN
        wrapped = FlattenMultiDiscreteToDiscrete(env)
        model = DQN.load(model_path, env=wrapped)
        for ep in range(n_episodes):
            obs, _ = wrapped.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = wrapped.step(action)
                wrapped.render()
                total_reward += reward
                done = terminated or truncated
            print(f"Episode {ep+1}: reward={total_reward:.2f}")
        wrapped.close()

    elif algorithm == "ppo":
        from stable_baselines3 import PPO
        model = PPO.load(model_path, env=env)
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                total_reward += reward
                done = terminated or truncated
            print(f"Episode {ep+1}: reward={total_reward:.2f}")
        env.close()

    elif algorithm == "reinforce":
        import torch
        from training.pg_training import REINFORCEPolicy
        obs_dim = env.observation_space.shape[0]
        action_nvec = list(env.action_space.nvec)
        policy = REINFORCEPolicy(obs_dim=obs_dim, action_nvec=action_nvec)
        policy.load_state_dict(torch.load(model_path, weights_only=True))
        policy.eval()
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            total_reward = 0.0
            while not done:
                with torch.no_grad():
                    obs_t = torch.FloatTensor(obs)
                    action, _ = policy.get_action(obs_t)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                total_reward += reward
                done = terminated or truncated
            print(f"Episode {ep+1}: reward={total_reward:.2f}")
        env.close()

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run trained AstroLogic agent")
    p.add_argument("--algorithm", choices=["dqn", "ppo", "reinforce"],
                   default="dqn")
    p.add_argument("--model-path", default=None,
                   help="Path to model file. Auto-detected if omitted.")
    p.add_argument("--episodes", type=int, default=3)
    args = p.parse_args()

    path = args.model_path or find_best_model(args.algorithm)
    print(f"Loading {args.algorithm.upper()} model from {path}")
    run_episodes(args.algorithm, path, args.episodes)
