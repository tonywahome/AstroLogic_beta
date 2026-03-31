"""Train PPO agent on AstroExploration environment."""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astro_env  # noqa: F401
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from training.hyperparams import PPO_CONFIGS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES


def train_ppo(run_idx: int, seed: int = 42) -> dict:
    """Train a single PPO run with the given config index."""
    config = PPO_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}")
    print(f"Training PPO - {run_name} (Run {run_idx})")
    print(f"{'='*60}")

    log_dir = f"results/logs/{run_name}"
    model_dir = f"results/models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # PPO supports MultiDiscrete natively - no wrapper needed
    env = gym.make("AstroExploration-v0")
    env = Monitor(env, log_dir)

    eval_env = gym.make("AstroExploration-v0")
    eval_env = Monitor(eval_env)

    model_params = {k: v for k, v in config.items() if k != "name"}

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=seed,
        tensorboard_log=f"results/logs/{run_name}_tb",
        **model_params,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    import time
    start_time = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        progress_bar=True,
    )
    wall_time = time.time() - start_time

    model.save(os.path.join(model_dir, "final_model"))
    print(f"\nTraining completed in {wall_time:.1f}s")
    print(f"Model saved to {model_dir}/final_model.zip")

    env.close()
    eval_env.close()

    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "PPO"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on AstroExploration")
    parser.add_argument("--run", type=int, default=0, help="Config index (0-9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(args.run, args.seed)
