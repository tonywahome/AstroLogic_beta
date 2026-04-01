"""PPO and REINFORCE training with 10 hyperparameter configurations each.

High-reward exploitation strategies:
  - PPO: Advantage normalization ensures above-average returns get positive
    advantages (policy strengthened) while below-average returns get negative
    advantages (policy moves away).  Higher n_epochs = more exploitation per
    rollout.
  - REINFORCE: Baseline subtraction (mean or running) serves the same role —
    returns above the baseline produce positive policy gradients, returns
    below baseline produce negative gradients.
"""

import os, csv, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import environment  # noqa: F401  — registers AstroExploration-v0

# ============================================================
# Training constants
# ============================================================
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5
REINFORCE_EPISODES = 1_000

# Shared reward rebalance — identical to dqn_training.py for fair comparison
_REWARD_KWARGS = {
    "step_fuel_penalty": 0.001,
    "step_time_penalty": 0.0001,
    "collision_penalty": -100.0,
    "orbital_insertion_bonus": 100.0,
    "transmission_bonus": 200.0,
    "proximity_scale": 0.5,
}

_NON_MODEL_KEYS_PPO = {"name", "reward_kwargs"}

# ============================================================
# 10 PPO hyperparameter configurations
# ============================================================
PPO_CONFIGS = [
    {   # 0  Baseline
        "name": "ppo_baseline",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 1  High entropy (more exploration)
        "name": "ppo_high_entropy",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.05,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 2  Lower learning rate
        "name": "ppo_low_lr",
        "learning_rate": 1.5e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 3  Tight clipping + larger batch
        "name": "ppo_tight_clip",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "clip_range": 0.15,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 4  Wide clipping + larger batch
        "name": "ppo_wide_clip",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 10,
        "clip_range": 0.3,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 5  More epochs + deeper net + higher GAE
        "name": "ppo_more_epochs",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 15,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256, 128]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 6  Smaller network
        "name": "ppo_small_net",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [128, 128]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 7  Large rollout + bigger batch + higher GAE
        "name": "ppo_large_rollout",
        "learning_rate": 3e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 8  Lower gamma (near-term rewards)
        "name": "ppo_low_gamma",
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 9  High learning rate + deeper net
        "name": "ppo_high_lr_deep",
        "learning_rate": 5e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "gamma": 0.995,
        "gae_lambda": 0.95,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256, 128]},
        "reward_kwargs": _REWARD_KWARGS,
    },
]

# ============================================================
# 10 REINFORCE hyperparameter configurations
# ============================================================
REINFORCE_CONFIGS = [
    {   # 0  Baseline (mean baseline)
        "name": "reinforce_baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 1  No baseline (vanilla REINFORCE)
        "name": "reinforce_no_baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "none",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 2  Low learning rate
        "name": "reinforce_low_lr",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 3  High learning rate
        "name": "reinforce_high_lr",
        "learning_rate": 5e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 4  Large network
        "name": "reinforce_large_net",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [256, 128],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 5  Small network
        "name": "reinforce_small_net",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [64, 32],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 6  Low gamma
        "name": "reinforce_low_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 7  Very low gamma
        "name": "reinforce_very_low_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.90,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 8  Deep network
        "name": "reinforce_deep_net",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [256, 128, 64],
        "baseline": "mean",
        "reward_kwargs": _REWARD_KWARGS,
    },
    {   # 9  Running baseline
        "name": "reinforce_running_baseline",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "running",
        "reward_kwargs": _REWARD_KWARGS,
    },
]


# ============================================================
# REINFORCE Policy Network (inlined)
# ============================================================

class REINFORCEPolicy(nn.Module):
    """Multi-head policy network for MultiDiscrete action space.

    Architecture:
        Input (26) -> Shared Backbone (hidden layers + ReLU)
                   -> Head 0: Linear -> Categorical (5 actions: thrust)
                   -> Head 1: Linear -> Categorical (3 actions: pitch)
                   -> Head 2: Linear -> Categorical (3 actions: yaw)
                   -> Head 3: Linear -> Categorical (4 actions: instrument)
                   -> Head 4: Linear -> Categorical (2 actions: comm)
    """

    def __init__(self, obs_dim: int = 26, action_nvec: list = None,
                 hidden_sizes: list = None):
        super().__init__()

        if action_nvec is None:
            action_nvec = [5, 3, 3, 4, 2]
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        self.action_nvec = action_nvec

        layers = []
        prev_size = obs_dim
        for h_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, h_size))
            layers.append(nn.ReLU())
            prev_size = h_size
        self.backbone = nn.Sequential(*layers)

        self.heads = nn.ModuleList([
            nn.Linear(prev_size, n) for n in action_nvec
        ])

    def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
        features = self.backbone(obs)
        return [head(features) for head in self.heads]

    def get_action(self, obs: torch.Tensor) -> tuple:
        logits_list = self.forward(obs)
        actions = []
        total_log_prob = torch.tensor(0.0)

        for logits in logits_list:
            dist = Categorical(logits=logits)
            action = dist.sample()
            total_log_prob = total_log_prob + dist.log_prob(action)
            actions.append(action.item())

        return np.array(actions, dtype=np.int64), total_log_prob

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor) -> torch.Tensor:
        logits_list = self.forward(obs)
        total_log_prob = torch.zeros(obs.shape[0])

        for i, logits in enumerate(logits_list):
            dist = Categorical(logits=logits)
            total_log_prob = total_log_prob + dist.log_prob(actions[:, i])

        return total_log_prob


# ============================================================
# REINFORCE Agent (inlined)
# ============================================================

class REINFORCEAgent:
    """REINFORCE algorithm with multi-head policy for MultiDiscrete actions.

    Supports three baseline modes for high-reward exploitation:
        - 'none': No baseline (vanilla REINFORCE)
        - 'mean': Subtract mean of returns per episode — above-mean returns
          produce positive gradients, below-mean are suppressed
        - 'running': Subtract exponential moving average — adapts over training
    """

    def __init__(self, env, learning_rate=1e-3, gamma=0.99,
                 hidden_sizes=None, baseline="mean", seed=42):
        self.env = env
        self.gamma = gamma
        self.baseline = baseline

        torch.manual_seed(seed)
        np.random.seed(seed)

        obs_dim = env.observation_space.shape[0]
        action_nvec = list(env.action_space.nvec)

        self.policy = REINFORCEPolicy(
            obs_dim=obs_dim,
            action_nvec=action_nvec,
            hidden_sizes=hidden_sizes or [128, 64],
        )
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        self.running_return = 0.0
        self.running_count = 0

    def collect_episode(self):
        obs, info = self.env.reset()
        log_probs = []
        rewards = []
        done = False

        while not done:
            obs_tensor = torch.FloatTensor(obs)
            action, log_prob = self.policy.get_action(obs_tensor)
            log_probs.append(log_prob)

            obs, reward, terminated, truncated, info = self.env.step(action)
            rewards.append(reward)
            done = terminated or truncated

        return log_probs, rewards, len(rewards), info

    def compute_returns(self, rewards):
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        returns = torch.FloatTensor(returns)

        if self.baseline == "mean":
            if returns.std() > 1e-8:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            else:
                returns = returns - returns.mean()

        elif self.baseline == "running":
            self.running_count += 1
            alpha = 0.05
            self.running_return = (
                (1 - alpha) * self.running_return + alpha * returns.mean().item()
            )
            returns = returns - self.running_return

        return returns

    def update(self, log_probs, returns):
        policy_loss = torch.tensor(0.0)
        for log_prob, G in zip(log_probs, returns):
            policy_loss = policy_loss + (-log_prob * G)

        policy_loss = policy_loss / len(log_probs)

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss.item()

    def train(self, num_episodes=1000, log_interval=100, save_dir=None):
        history = []
        reward_window = []

        for episode in range(1, num_episodes + 1):
            log_probs, rewards, ep_len, info = self.collect_episode()
            ep_reward = sum(rewards)
            returns = self.compute_returns(rewards)
            loss = self.update(log_probs, returns)

            history.append((ep_reward, ep_len))
            reward_window.append(ep_reward)
            if len(reward_window) > 100:
                reward_window.pop(0)

            if episode % log_interval == 0:
                avg_reward = np.mean(reward_window)
                avg_len = np.mean([h[1] for h in history[-100:]])
                print(
                    f"Episode {episode:5d} | "
                    f"Avg Reward: {avg_reward:10.2f} | "
                    f"Avg Length: {avg_len:8.1f} | "
                    f"Loss: {loss:10.4f}"
                )

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            torch.save(self.policy.state_dict(),
                       os.path.join(save_dir, "policy.pt"))

            csv_path = os.path.join(save_dir, "rewards.csv")
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "episode_length"])
                for i, (r, l) in enumerate(history, 1):
                    writer.writerow([i, r, l])

            print(f"\nModel saved to {save_dir}/policy.pt")
            print(f"Rewards saved to {csv_path}")

        return history


# ============================================================
# Callbacks
# ============================================================

class MissionKPICallback(BaseCallback):
    """Logs per-episode mission KPIs to CSV (same format as dqn_training)."""
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
# PPO training function
# ============================================================

def _make_env(reward_kwargs):
    return gym.make("AstroExploration-v0", reward_kwargs=reward_kwargs)


def train_ppo(run_idx, seed=42):
    config = PPO_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}\nTraining PPO — {run_name}  (run {run_idx})\n{'='*60}")

    log_dir = f"models/pg/{run_name}/logs"
    model_dir = f"models/pg/{run_name}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    rk = config.get("reward_kwargs", {})
    train_env = Monitor(_make_env(rk), log_dir)
    eval_env = Monitor(_make_env(rk))

    model_params = {k: v for k, v in config.items()
                    if k not in _NON_MODEL_KEYS_PPO}

    model = PPO("MlpPolicy", train_env, verbose=1, seed=seed,
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
    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "PPO"}


# ============================================================
# REINFORCE training function
# ============================================================

def train_reinforce(run_idx, seed=42):
    config = REINFORCE_CONFIGS[run_idx]
    run_name = config["name"]
    print(f"\n{'='*60}\nTraining REINFORCE — {run_name}  (run {run_idx})\n{'='*60}")

    model_dir = f"models/pg/{run_name}"
    os.makedirs(model_dir, exist_ok=True)

    rk = config.get("reward_kwargs", {})
    env = _make_env(rk)

    agent = REINFORCEAgent(
        env=env,
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        hidden_sizes=config["hidden_sizes"],
        baseline=config["baseline"],
        seed=seed,
    )

    start = time.time()
    history = agent.train(
        num_episodes=REINFORCE_EPISODES,
        log_interval=100,
        save_dir=model_dir,
    )
    wall_time = time.time() - start

    print(f"  Done in {wall_time:.1f}s — saved to {model_dir}")
    env.close()
    return {"run_name": run_name, "wall_time": wall_time, "algorithm": "REINFORCE"}


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=["ppo", "reinforce"], required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.algo == "ppo":
        train_ppo(args.run, args.seed)
    else:
        train_reinforce(args.run, args.seed)
