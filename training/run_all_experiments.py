"""Master script to run all 30 hyperparameter experiments.

Usage:
    python training/run_all_experiments.py
    python training/run_all_experiments.py --algorithm dqn
    python training/run_all_experiments.py --algorithm ppo --start 3 --end 5
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.train_dqn import train_dqn
from training.train_ppo import train_ppo
from training.train_reinforce import train_reinforce
from training.hyperparams import (
    DQN_CONFIGS, PPO_CONFIGS, REINFORCE_CONFIGS,
)


def run_all(algorithm=None, start=0, end=10, seed=42):
    """Run all experiments or a subset filtered by algorithm."""
    results = []
    os.makedirs("results", exist_ok=True)

    experiments = []
    if algorithm is None or algorithm == "dqn":
        for i in range(start, min(end, len(DQN_CONFIGS))):
            experiments.append(("DQN", i, train_dqn))
    if algorithm is None or algorithm == "ppo":
        for i in range(start, min(end, len(PPO_CONFIGS))):
            experiments.append(("PPO", i, train_ppo))
    if algorithm is None or algorithm == "reinforce":
        for i in range(start, min(end, len(REINFORCE_CONFIGS))):
            experiments.append(("REINFORCE", i, train_reinforce))

    total = len(experiments)
    print(f"\n{'='*60}")
    print(f"Running {total} experiments")
    print(f"{'='*60}\n")

    overall_start = time.time()

    for idx, (algo_name, run_idx, train_fn) in enumerate(experiments, 1):
        print(f"\n[{idx}/{total}] Starting {algo_name} run {run_idx}...")
        try:
            result = train_fn(run_idx, seed=seed)
            result["status"] = "completed"
            results.append(result)
        except Exception as e:
            print(f"ERROR in {algo_name} run {run_idx}: {e}")
            results.append({
                "run_name": f"{algo_name.lower()}_run{run_idx}",
                "algorithm": algo_name,
                "wall_time": 0,
                "status": "failed",
                "error": str(e),
            })

    overall_time = time.time() - overall_start

    # Save experiment index
    index = {
        "total_experiments": total,
        "total_wall_time_seconds": overall_time,
        "seed": seed,
        "results": results,
    }
    index_path = "results/experiment_index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All experiments completed in {overall_time:.1f}s")
    print(f"Results index saved to {index_path}")
    print(f"{'='*60}")

    # Summary table
    print(f"\n{'Algorithm':<12} {'Run Name':<30} {'Time (s)':<12} {'Status'}")
    print("-" * 70)
    for r in results:
        print(f"{r.get('algorithm',''):<12} {r['run_name']:<30} {r['wall_time']:<12.1f} {r['status']}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all RL experiments")
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["dqn", "ppo", "reinforce"],
                        help="Run only this algorithm")
    parser.add_argument("--start", type=int, default=0, help="Start run index")
    parser.add_argument("--end", type=int, default=10, help="End run index (exclusive)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_all(args.algorithm, args.start, args.end, args.seed)
