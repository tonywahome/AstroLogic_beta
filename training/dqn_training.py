"""DQN training with 10 hyperparameter configurations.

High-reward exploitation strategy:
  - ALL configs use Prioritized Experience Replay (PER) when sb3-contrib is
    available.  PER samples transitions with high TD-error more often, which
    correlates with surprising / high-reward transitions.  This lets the agent
    learn more from rare positive outcomes and forget ineffective transitions.
"""

import os, csv, time, sys
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

# Ensure imports work when running this script from inside the training folder.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import environment  # noqa: F401  — registers AstroExploration-v0
from environment.custom_env import FlattenMultiDiscreteToDiscrete

try:
    from sb3_contrib.common.buffers import PrioritizedReplayBuffer
    _HAS_PER = True
except ImportError:
    PrioritizedReplayBuffer = None
    _HAS_PER = False

# ============================================================
# Training constants
# ============================================================
TOTAL_TIMESTEPS = 500_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5

# Shared reward rebalance: delta-based approach and heading alignment rewards
# replace the static proximity signal.  All 10 DQN runs use the same reward
# so comparisons across hyperparams are fair.
_REWARD_KWARGS = {
    "step_fuel_penalty": 0.001,
    "step_time_penalty": 0.0001,
    "collision_penalty": -100.0,
    "orbital_insertion_bonus": 100.0,
    "transmission_bonus": 200.0,
    "approach_scale": 200.0,
    "heading_scale": 0.5,
}

_NON_MODEL_KEYS = {"name", "reward_kwargs", "use_per", "per_alpha", "per_beta"}

# ============================================================
# 10 DQN hyperparameter configurations
# ============================================================
DQN_CONFIGS = [
    {   # 0  Baseline + PER
        "name": "dqn_baseline",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 1  Smaller network
        "name": "dqn_small_net",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [128, 128]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 2  Higher learning rate
        "name": "dqn_high_lr",
        "learning_rate": 5e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 3  Lower learning rate
        "name": "dqn_low_lr",
        "learning_rate": 1e-5,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 4  Large buffer + strong PER prioritization
        "name": "dqn_large_buffer",
        "learning_rate": 1e-4,
        "buffer_size": 300_000,
        "batch_size": 128,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.65,
        "exploration_final_eps": 0.02,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.7, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 5  Soft target update
        "name": "dqn_soft_update",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 0.005,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 6  Lower gamma (near-term rewards)
        "name": "dqn_low_gamma",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.95,
        "exploration_fraction": 0.65,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 7  Deeper network
        "name": "dqn_deep_net",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256, 128]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 8  Medium lr + large batch
        "name": "dqn_med_lr_batch",
        "learning_rate": 3e-4,
        "buffer_size": 200_000,
        "batch_size": 256,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.6,
        "exploration_final_eps": 0.05,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.6, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 9  Long exploration + maximum PER alpha
        "name": "dqn_long_explore",
        "learning_rate": 1e-4,
        "buffer_size": 100_000,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.995,
        "exploration_fraction": 0.75,
        "exploration_final_eps": 0.01,
        "learning_starts": 3000,
        "policy_kwargs": {"net_arch": [256, 256]},
        "use_per": True, "per_alpha": 0.8, "per_beta": 0.4,
        "reward_kwargs": _REWARD_KWARGS,
    },
]


# ============================================================
# Callbacks
# ============================================================

class MissionKPICallback(BaseCallback):
    """Logs per-episode mission KPIs to CSV."""
    def __init__(self, log_dir, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.rows = []

    def _on_step(self):
        for info, done in zip(self.locals.get("infos", []),
                              self.locals.get("dones", [])):
            if done:
                self.rows.append({
                    "timesteps": int(self.num_timesteps),
                    "success": int(bool(info.get("success", False))),
                    "collision": int(bool(info.get("collision", False))),
                    "out_of_bounds": int(bool(info.get("out_of_bounds", False))),
                })
        return True

    def _on_training_end(self):
        if not self.rows:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "kpi_train.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)


# ============================================================
# Training function
# ============================================================

def _make_env(reward_kwargs):
    env = gym.make("AstroExploration-v0", reward_kwargs=reward_kwargs)
    return FlattenMultiDiscreteToDiscrete(env)


def train_dqn(run_idx, seed=42):
    config = DQN_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}\nTraining DQN — {run_name}  (run {run_idx})\n{'='*60}")

    log_dir = f"models/dqn/{run_name}/logs"
    model_dir = f"models/dqn/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    rk = config.get("reward_kwargs", {})
    train_env = Monitor(_make_env(rk), log_dir)
    eval_env = Monitor(_make_env(rk))

    model_params = {k: v for k, v in config.items() if k not in _NON_MODEL_KEYS}

    use_per = config.get("use_per", False)
    if use_per and _HAS_PER:
        model_params["replay_buffer_class"] = PrioritizedReplayBuffer
        model_params["replay_buffer_kwargs"] = {
            "alpha": config.get("per_alpha", 0.6),
            "beta": config.get("per_beta", 0.4),
        }
        model_params.pop("optimize_memory_usage", None)
        print(f"  PER enabled (alpha={config.get('per_alpha')}, beta={config.get('per_beta')})")
    elif use_per and not _HAS_PER:
        print("  WARNING: PER requested but sb3-contrib not installed — standard buffer used.")

    model = DQN("MlpPolicy", train_env, verbose=1, seed=seed,
                tensorboard_log=os.path.join(model_dir, "tb"), **model_params)

    eval_cb = EvalCallback(eval_env, best_model_save_path=model_dir,
                           log_path=log_dir, eval_freq=EVAL_FREQ,
                           n_eval_episodes=N_EVAL_EPISODES, deterministic=True)
    kpi_cb = MissionKPICallback(log_dir)

    start = time.time()
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[eval_cb, kpi_cb],
                progress_bar=True)
    wall_time = time.time() - start

    model.save(os.path.join(model_dir, "final_model"))
    print(f"  Done in {wall_time:.1f}s — saved to {model_dir}")

    train_env.close()
    eval_env.close()
    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "DQN"}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    train_dqn(args.run, args.seed)
