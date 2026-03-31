"""Train REINFORCE agent on AstroExploration environment."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astro_env  # noqa: F401
import gymnasium as gym
from agents.reinforce_agent import REINFORCEAgent
from training.hyperparams import REINFORCE_CONFIGS, REINFORCE_EPISODES


def train_reinforce(run_idx: int, seed: int = 42) -> dict:
    """Train a single REINFORCE run with the given config index."""
    config = REINFORCE_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}")
    print(f"Training REINFORCE - {run_name} (Run {run_idx})")
    print(f"{'='*60}")

    save_dir = f"results/models/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    env = gym.make("AstroExploration-v0")

    import time
    start_time = time.time()

    agent = REINFORCEAgent(
        env=env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        hidden_sizes=config["hidden_sizes"],
        baseline=config["baseline"],
        seed=seed,
    )

    history = agent.train(
        num_episodes=REINFORCE_EPISODES,
        log_interval=100,
        save_dir=save_dir,
    )

    wall_time = time.time() - start_time
    print(f"\nTraining completed in {wall_time:.1f}s")

    env.close()

    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "REINFORCE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE on AstroExploration")
    parser.add_argument("--run", type=int, default=0, help="Config index (0-9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_reinforce(args.run, args.seed)
