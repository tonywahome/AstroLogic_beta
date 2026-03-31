"""Train DQN agent on AstroExploration environment."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astro_env  # noqa: F401
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from astro_env.wrappers import FlattenMultiDiscreteToDiscrete
from training.hyperparams import DQN_CONFIGS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES


def train_dqn(run_idx: int, seed: int = 42) -> dict:
    """Train a single DQN run with the given config index."""
    config = DQN_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}")
    print(f"Training DQN - {run_name} (Run {run_idx})")
    print(f"{'='*60}")

    # Create directories
    log_dir = f"results/logs/{run_name}"
    model_dir = f"results/models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create environment with DQN wrapper (MultiDiscrete -> Discrete)
    env = gym.make("AstroExploration-v0")
    env = FlattenMultiDiscreteToDiscrete(env)
    env = Monitor(env, log_dir)

    # Evaluation environment
    eval_env = gym.make("AstroExploration-v0")
    eval_env = FlattenMultiDiscreteToDiscrete(eval_env)
    eval_env = Monitor(eval_env)

    # Extract config params (excluding non-DQN keys)
    model_params = {k: v for k, v in config.items() if k != "name"}

    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"results/logs/{run_name}_tb",
        **model_params,
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    # Train
    import time
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )
    wall_time = time.time() - start_time

    # Save final model
    model.save(os.path.join(model_dir, "final_model"))
    print(f"\nTraining completed in {wall_time:.1f}s")
    print(f"Model saved to {model_dir}/final_model.zip")

    env.close()
    eval_env.close()

    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "DQN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on AstroExploration")
    parser.add_argument("--run", type=int, default=0, help="Config index (0-9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dqn(args.run, args.seed)
