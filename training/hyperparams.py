"""Hyperparameter configurations for all 30 experiments.

10 configurations per algorithm, each varying 1-2 parameters from a baseline.
"""

# Total training timesteps for SB3 algorithms
TOTAL_TIMESTEPS = 100_000  # Reduced from 500k for faster exploration training
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 5

# REINFORCE training episodes
REINFORCE_EPISODES = 1000  # Reduced for faster experimentation

# ============================================================
# Shared DQN environment kwargs
# - success_count=1: only need 1 transmitted biosignature to win (easier early goal)
# - detection_zone_scale=2.0: larger target zones make early detections reachable
# - enable_curriculum=True: episode horizon shrinks from 10K -> 3K as agent improves
# - reward_kwargs: rebalanced so collision (-80) no longer drowns out success (+2000)
# ============================================================
_DQN_ENV_KWARGS = {
    "enable_curriculum": True,
    "success_count": 1,
    "detection_zone_scale": 2.0,
    "initial_fuel": 3000.0,  # Increased fuel for longer exploration
    "initial_battery": 1500.0,  # Increased battery
    "fuel_consumption_rate": 0.8,  # Slower fuel consumption
    "reward_kwargs": {
        "step_fuel_penalty": 0.001,
        "step_time_penalty": 0.0001,
        "direction_bonus_scale": 4.0,
        "moving_away_penalty": 0.01,  # REDUCED from 0.05 - encourage exploration away from Earth
        "compatible_instrument_bonus": 8.0,
        "in_zone_hold_bonus": 2.0,
        "proximity_scale": 1.0,
        "transmission_bonus": 600.0,
        "success_completion_bonus": 2000.0,
        "collision_penalty": -80.0,
        # NEW: Exploration bonuses
        "distance_bonus_scale": 10.0,  # Reward for distance from Earth
        "new_planet_visited_bonus": 500.0,  # Bonus for visiting Mars, Europa, etc.
        "energy_efficiency_bonus": 50.0,  # Bonus for low fuel consumption
    },
}

# ============================================================
# DQN Configurations (10 runs)
# All runs share _DQN_ENV_KWARGS so reward balance and zone scale are consistent.
# use_per=True enables PrioritizedReplayBuffer (sb3-contrib) — graceful fallback
#   if sb3-contrib is absent.
# n_steps>1 propagates sparse rewards back N steps (SB3 DQN supports this).
# exploration_fraction >= 0.6 because Discrete(360) still needs wide coverage.
# ============================================================
DQN_CONFIGS = [
    {   # Run 0: Baseline + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 1: Smaller network + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 2: Higher learning rate + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 3: Lower learning rate + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 4: Large buffer + PER with high alpha (strong prioritization)
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
        "use_per": True,
        "per_alpha": 0.7,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 5: Soft target update + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 6: Lower gamma (prioritises near-term navigation rewards)
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 7: Deeper network + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 8: Medium lr + large batch + PER
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
        "use_per": True,
        "per_alpha": 0.6,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 9: Long exploration + very high PER alpha (maximum exploitation of rare wins)
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
        "use_per": True,
        "per_alpha": 0.8,
        "per_beta": 0.4,
        "env_kwargs": _DQN_ENV_KWARGS,
    },
]

# ============================================================
# PPO Configurations (10 runs)
# All runs share the same rebalanced reward & environment settings as DQN:
# - success_count=1: only need 1 biosignature detected to win
# - detection_zone_scale=2.0: larger target zones
# - enable_curriculum=True: progressive difficulty
# - reward_kwargs: balanced so +2000 success > -80 collision
# PPO parameters vary: lr, n_steps (rollout), batch, clipping, entropy
# ============================================================
PPO_CONFIGS = [
    {   # Run 0: Baseline
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 1: Higher entropy (wider exploration)
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 2: Lower learning rate
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 3: Tighter clipping (safer policy updates)
        "name": "ppo_tight_clip",
        "learning_rate": 2e-4,
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 4: Wider clipping (bolder updates)
        "name": "ppo_wide_clip",
        "learning_rate": 2e-4,
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 5: More epochs (exploit rollouts more)
        "name": "ppo_more_epochs",
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 128,
        "n_epochs": 15,
        "clip_range": 0.2,
        "ent_coef": 0.015,
        "vf_coef": 0.6,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256, 128]},
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 6: Smaller network
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 7: Larger rollout (longer trajectory collection)
        "name": "ppo_large_rollout",
        "learning_rate": 2e-4,
        "n_steps": 4096,
        "batch_size": 256,
        "n_epochs": 10,
        "clip_range": 0.2,
        "ent_coef": 0.015,
        "vf_coef": 0.6,
        "gamma": 0.995,
        "gae_lambda": 0.97,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": [256, 256]},
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 8: Lower gamma (prioritise near-term)
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
    {   # Run 9: Higher lr + deeper network
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
        "env_kwargs": _DQN_ENV_KWARGS,
    },
]

# ============================================================
# REINFORCE Configurations (10 runs)
# Baseline: lr=1e-3, gamma=0.99, hidden=[128,64], baseline=mean
# ============================================================
REINFORCE_CONFIGS = [
    {   # Run 0: Baseline
        "name": "reinforce_baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
    },
    {   # Run 1: No baseline
        "name": "reinforce_no_baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "none",
    },
    {   # Run 2: Lower learning rate
        "name": "reinforce_low_lr",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
    },
    {   # Run 3: Higher learning rate
        "name": "reinforce_high_lr",
        "learning_rate": 5e-3,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
    },
    {   # Run 4: Larger network
        "name": "reinforce_large_net",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [256, 128],
        "baseline": "mean",
    },
    {   # Run 5: Smaller network
        "name": "reinforce_small_net",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [64, 32],
        "baseline": "mean",
    },
    {   # Run 6: Lower gamma
        "name": "reinforce_low_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.95,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
    },
    {   # Run 7: Very low gamma
        "name": "reinforce_very_low_gamma",
        "learning_rate": 1e-3,
        "gamma": 0.90,
        "hidden_sizes": [128, 64],
        "baseline": "mean",
    },
    {   # Run 8: Deeper network
        "name": "reinforce_deep_net",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "hidden_sizes": [256, 128, 64],
        "baseline": "mean",
    },
    {   # Run 9: Running baseline
        "name": "reinforce_running_baseline",
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "hidden_sizes": [128, 64],
        "baseline": "running",
    },
]
