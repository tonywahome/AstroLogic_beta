"""Compare all trained models and generate comparison plots.

Reads training logs from results/logs/ and generates:
1. Learning curves per algorithm (2x2 grid)
2. Best-run comparison overlay
3. Final performance bar chart
4. Biosignature discovery rate
5. Training efficiency scatter
6. Summary CSV table

Usage:
    python evaluation/compare_models.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from training.hyperparams import (
    DQN_CONFIGS, PPO_CONFIGS, REINFORCE_CONFIGS,
)

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
ALGORITHM_COLORS = {
    "DQN": "#e74c3c",
    "PPO": "#2ecc71",
    "REINFORCE": "#9b59b6",
}


def load_monitor_data(log_dir: str) -> pd.DataFrame | None:
    """Load SB3 Monitor CSV data from a log directory."""
    csv_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(csv_path):
        # Try finding any .monitor.csv file
        for f in os.listdir(log_dir):
            if f.endswith(".monitor.csv"):
                csv_path = os.path.join(log_dir, f)
                break
        else:
            return None

    try:
        df = pd.read_csv(csv_path, skiprows=1)
        if "r" in df.columns:
            df = df.rename(columns={"r": "reward", "l": "length", "t": "time"})
        return df
    except Exception:
        return None


def load_reinforce_data(model_dir: str) -> pd.DataFrame | None:
    """Load REINFORCE training rewards CSV."""
    csv_path = os.path.join(model_dir, "rewards.csv")
    if not os.path.exists(csv_path):
        return None
    try:
        return pd.read_csv(csv_path)
    except Exception:
        return None


def smooth_curve(values, window=100):
    """Apply rolling average smoothing."""
    if len(values) < window:
        window = max(1, len(values) // 5)
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def plot_learning_curves():
    """Plot 1: Learning curves per algorithm (2x2 grid)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Learning Curves by Algorithm", fontsize=16, fontweight="bold")

    algo_configs = [
        ("DQN", DQN_CONFIGS, axes[0]),
        ("PPO", PPO_CONFIGS, axes[1]),
        ("REINFORCE", REINFORCE_CONFIGS, axes[2]),
    ]

    for algo_name, configs, ax in algo_configs:
        ax.set_title(algo_name, fontsize=14)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Episode Reward")
        ax.grid(True, alpha=0.3)

        colors = plt.cm.tab10(np.linspace(0, 1, len(configs)))

        for i, config in enumerate(configs):
            run_name = config["name"]
            if algo_name == "REINFORCE":
                df = load_reinforce_data(f"results/models/{run_name}")
            else:
                df = load_monitor_data(f"results/logs/{run_name}")

            if df is not None and "reward" in df.columns:
                rewards = df["reward"].values
                smoothed = smooth_curve(rewards)
                ax.plot(smoothed, color=colors[i], alpha=0.7,
                        label=run_name.replace(f"{algo_name.lower()}_", ""), linewidth=0.8)

        ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "learning_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Generated: learning_curves.png")


def plot_best_comparison():
    """Plot 2: Best run from each algorithm overlaid."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title("Best Run Comparison Across Algorithms", fontsize=14, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Episode Reward (smoothed)")
    ax.grid(True, alpha=0.3)

    algo_configs = [
        ("DQN", DQN_CONFIGS),
        ("PPO", PPO_CONFIGS),
        ("REINFORCE", REINFORCE_CONFIGS),
    ]

    for algo_name, configs in algo_configs:
        best_mean = -float("inf")
        best_data = None

        for config in configs:
            run_name = config["name"]
            if algo_name == "REINFORCE":
                df = load_reinforce_data(f"results/models/{run_name}")
            else:
                df = load_monitor_data(f"results/logs/{run_name}")

            if df is not None and "reward" in df.columns:
                mean_reward = df["reward"].tail(100).mean()
                if mean_reward > best_mean:
                    best_mean = mean_reward
                    best_data = df["reward"].values

        if best_data is not None:
            smoothed = smooth_curve(best_data)
            color = ALGORITHM_COLORS[algo_name]
            ax.plot(smoothed, color=color, linewidth=2,
                    label=f"{algo_name} (best: {best_mean:.1f})")

    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "best_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Generated: best_comparison.png")


def plot_final_performance():
    """Plot 3: Final performance bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Final Performance by Algorithm", fontsize=14, fontweight="bold")

    algo_names = ["DQN", "PPO", "REINFORCE"]
    all_configs = [DQN_CONFIGS, PPO_CONFIGS, REINFORCE_CONFIGS]

    means, stds, best_means = [], [], []

    for algo_name, configs in zip(algo_names, all_configs):
        run_means = []
        for config in configs:
            run_name = config["name"]
            if algo_name == "REINFORCE":
                df = load_reinforce_data(f"results/models/{run_name}")
            else:
                df = load_monitor_data(f"results/logs/{run_name}")

            if df is not None and "reward" in df.columns:
                final_mean = df["reward"].tail(50).mean()
                run_means.append(final_mean)

        if run_means:
            means.append(np.mean(run_means))
            stds.append(np.std(run_means))
            best_means.append(np.max(run_means))
        else:
            means.append(0)
            stds.append(0)
            best_means.append(0)

    x = np.arange(len(algo_names))
    width = 0.35

    colors = [ALGORITHM_COLORS[a] for a in algo_names]
    bars1 = ax.bar(x - width/2, means, width, yerr=stds, label="All Runs (mean +/- std)",
                   color=colors, alpha=0.6, capsize=5)
    bars2 = ax.bar(x + width/2, best_means, width, label="Best Run",
                   color=colors, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(algo_names)
    ax.set_ylabel("Final Episode Reward")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "final_performance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Generated: final_performance.png")


def plot_training_efficiency():
    """Plot 5: Training efficiency scatter (wall time vs reward)."""
    # Load experiment index
    index_path = os.path.join(RESULTS_DIR, "experiment_index.json")
    if not os.path.exists(index_path):
        print("  Skipped: training_efficiency.png (no experiment_index.json)")
        return

    with open(index_path) as f:
        index = json.load(f)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title("Training Efficiency: Wall Time vs Final Reward", fontsize=14, fontweight="bold")
    ax.set_xlabel("Wall Clock Time (seconds)")
    ax.set_ylabel("Final Mean Reward")

    for result in index.get("results", []):
        if result.get("status") != "completed":
            continue
        algo = result.get("algorithm", "")
        wall_time = result.get("wall_time", 0)
        run_name = result.get("run_name", "")

        # Load final reward
        if algo == "REINFORCE":
            df = load_reinforce_data(f"results/models/{run_name}")
        else:
            df = load_monitor_data(f"results/logs/{run_name}")

        if df is not None and "reward" in df.columns:
            final_reward = df["reward"].tail(50).mean()
            color = ALGORITHM_COLORS.get(algo, "gray")
            ax.scatter(wall_time, final_reward, color=color, s=60, alpha=0.7,
                       label=algo if algo not in ax.get_legend_handles_labels()[1] else "")

    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_efficiency.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Generated: training_efficiency.png")


def generate_summary_table():
    """Generate summary CSV with all experiment results."""
    rows = []
    algo_names = ["DQN", "PPO", "REINFORCE"]
    all_configs = [DQN_CONFIGS, PPO_CONFIGS, REINFORCE_CONFIGS]

    for algo_name, configs in zip(algo_names, all_configs):
        for config in configs:
            run_name = config["name"]
            if algo_name == "REINFORCE":
                df = load_reinforce_data(f"results/models/{run_name}")
            else:
                df = load_monitor_data(f"results/logs/{run_name}")

            row = {
                "algorithm": algo_name,
                "run_name": run_name,
                "learning_rate": config.get("learning_rate", ""),
                "gamma": config.get("gamma", ""),
            }

            if df is not None and "reward" in df.columns:
                row["mean_reward"] = df["reward"].tail(50).mean()
                row["std_reward"] = df["reward"].tail(50).std()
                row["max_reward"] = df["reward"].max()
                row["total_episodes"] = len(df)
            else:
                row["mean_reward"] = "N/A"
                row["std_reward"] = "N/A"
                row["max_reward"] = "N/A"
                row["total_episodes"] = 0

            rows.append(row)

    df_summary = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "comparison_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"  Generated: comparison_summary.csv")
    print(f"\n{df_summary.to_string()}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("\nGenerating comparison plots...")
    print("=" * 50)

    plot_learning_curves()
    plot_best_comparison()
    plot_final_performance()
    plot_training_efficiency()
    generate_summary_table()

    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
