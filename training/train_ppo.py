"""Train PPO agent on AstroExploration environment."""

import sys
import os
import argparse
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astro_env  # noqa: F401
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from training.hyperparams import PPO_CONFIGS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES

# Keys that are NOT PPO constructor kwargs and must be stripped before passing to PPO().
_NON_MODEL_KEYS = {"name", "env_kwargs"}


class MissionKPITrainCallback(BaseCallback):
    """Logs per-episode mission KPIs (success, detections, distance) to CSV."""

    def __init__(self, log_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.rows = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for info, done in zip(infos, dones):
            if done:
                self.rows.append({
                    "timesteps": int(self.num_timesteps),
                    "success": int(bool(info.get("success", False))),
                    "detected_count": int(info.get("detected_count", 0)),
                    "transmitted_count": int(info.get("transmitted_count", 0)),
                    "min_target_distance": float(info.get("min_target_distance", 0.0)),
                    "instrument_compatible": int(bool(info.get("instrument_compatible", False))),
                    "collision": int(bool(info.get("collision", False))),
                    "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
                })
        return True

    def _on_training_end(self) -> None:
        if not self.rows:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "kpi_train.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


class CurriculumProgressCallback(BaseCallback):
    """Updates environment curriculum progress based on training step."""

    def __init__(self, env_list, verbose: int = 0):
        super().__init__(verbose)
        self.env_list = env_list

    def _on_step(self) -> bool:
        # Progress from 0 → 1 over TOTAL_TIMESTEPS
        progress = min(1.0, self.num_timesteps / max(1, TOTAL_TIMESTEPS))
        for env in self.env_list:
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'set_curriculum_progress'):
                env.unwrapped.set_curriculum_progress(progress)
        return True


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

    # Extract env_kwargs so PPO env gets rebalanced rewards,
    # easier success criteria, and larger detection zones.
    env_kwargs = config.get("env_kwargs", {})

    # PPO supports MultiDiscrete natively - no wrapper needed
    train_env = gym.make("AstroExploration-v0", **env_kwargs)
    train_env = Monitor(train_env, log_dir)

    eval_env = gym.make("AstroExploration-v0", **env_kwargs)
    eval_env = Monitor(eval_env)

    # Build PPO constructor kwargs — strip housekeeping keys.
    model_params = {k: v for k, v in config.items() if k not in _NON_MODEL_KEYS}

    model = PPO(
        "MlpPolicy",
        train_env,
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

    kpi_callback = MissionKPITrainCallback(log_dir)

    # If curriculum is enabled, track progress so horizon shrinks as agent improves.
    callbacks = [eval_callback, kpi_callback]
    if env_kwargs.get("enable_curriculum", False):
        curriculum_callback = CurriculumProgressCallback([train_env, eval_env])
        callbacks.insert(0, curriculum_callback)

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        progress_bar=True,
    )
    wall_time = time.time() - start

    model.save(os.path.join(model_dir, "final_model"))
    print(f"\nTraining completed in {wall_time:.1f}s")
    print(f"Model saved to {model_dir}/final_model.zip")

    train_env.close()
    eval_env.close()

    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "PPO"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO on AstroExploration")
    parser.add_argument("--run", type=int, default=0, help="Config index (0-9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_ppo(args.run, args.seed)
