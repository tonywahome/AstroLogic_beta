"""Train DQN agent on AstroExploration environment."""

import sys
import os
import argparse
import csv
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import astro_env  # noqa: F401
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from astro_env.wrappers import ReducedDiscreteWrapper
from training.hyperparams import DQN_CONFIGS, TOTAL_TIMESTEPS, EVAL_FREQ, N_EVAL_EPISODES

# Prioritized Experience Replay — requires sb3-contrib.
# Falls back to standard ReplayBuffer if not installed.
try:
    from sb3_contrib.common.buffers import PrioritizedReplayBuffer
    _HAS_PER = True
except ImportError:
    PrioritizedReplayBuffer = None
    _HAS_PER = False


# Keys that are NOT DQN constructor kwargs and must be stripped before passing to DQN().
_NON_MODEL_KEYS = {"name", "env_kwargs", "use_per", "per_alpha", "per_beta"}


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


def _make_env(env_kwargs: dict) -> gym.Env:
    """Create and wrap AstroExploration env for DQN."""
    env = gym.make("AstroExploration-v0", **env_kwargs)
    return ReducedDiscreteWrapper(env)


def train_dqn(run_idx: int, seed: int = 42) -> dict:
    """Train a single DQN run with the given config index."""
    config = DQN_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}")
    print(f"Training DQN - {run_name} (Run {run_idx})")
    print(f"{'='*60}")

    log_dir = f"results/logs/{run_name}"
    model_dir = f"results/models/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    env_kwargs = config.get("env_kwargs", {})
    train_env = Monitor(_make_env(env_kwargs), log_dir)
    eval_env = Monitor(_make_env(env_kwargs))

    # Build DQN constructor kwargs — strip housekeeping keys.
    model_params = {k: v for k, v in config.items() if k not in _NON_MODEL_KEYS}

    # Wire up Prioritized Experience Replay when requested and available.
    use_per = config.get("use_per", False)
    if use_per and _HAS_PER:
        model_params["replay_buffer_class"] = PrioritizedReplayBuffer
        model_params["replay_buffer_kwargs"] = {
            "alpha": config.get("per_alpha", 0.6),
            "beta": config.get("per_beta", 0.4),
        }
        # PER is incompatible with optimize_memory_usage.
        model_params.pop("optimize_memory_usage", None)
        print(f"  PER enabled (alpha={config.get('per_alpha', 0.6)}, "
              f"beta={config.get('per_beta', 0.4)})")
    elif use_per and not _HAS_PER:
        print("  WARNING: use_per=True but sb3-contrib not installed. "
              "Falling back to standard ReplayBuffer.")

    model = DQN(
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

    start = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, kpi_callback],
        progress_bar=True,
    )
    wall_time = time.time() - start

    model.save(os.path.join(model_dir, "final_model"))
    print(f"\nTraining completed in {wall_time:.1f}s")
    print(f"Model saved to {model_dir}/final_model.zip")

    train_env.close()
    eval_env.close()

    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "DQN"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on AstroExploration")
    parser.add_argument("--run", type=int, default=0, help="Config index (0-9)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_dqn(args.run, args.seed)
