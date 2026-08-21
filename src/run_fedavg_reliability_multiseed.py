"""Run the existing FedAvg reliability experiment across multiple seeds."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_fedavg_reliability.py"
RESULTS_ROOT = (SCRIPT_DIR / "../results/federated/dataset2_fedavg_reliability").resolve()
SCENARIOS = ["clean", "sensor_noise", "stale_updates", "persistent_failure", "mixed", "mixed_full"]
DEFAULT_SEEDS = [42, 123, 2024, 3407, 7777]


def parse_args():
    parser = argparse.ArgumentParser(description="Run reliability FedAvg across multiple seeds.")
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--seeds", default="42,123,2024,3407,7777")
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--keep-existing", action="store_true")
    return parser.parse_args()


def parse_seeds(raw):
    seeds = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    return list(dict.fromkeys(seeds))


def scenario_dir(scenario):
    return RESULTS_ROOT / scenario


def seed_dir(scenario, seed):
    return scenario_dir(scenario) / "seeds" / f"seed_{seed}"


def run_one_seed(scenario, seed, args):
    output_dir = scenario_dir(scenario)
    saved_dir = seed_dir(scenario, seed)
    (output_dir / "seeds").mkdir(parents=True, exist_ok=True)

    if saved_dir.exists():
        if args.keep_existing:
            print(f"Keeping existing results: {saved_dir}")
            return
        shutil.rmtree(saved_dir)

    for filename in ["fedavg_reliability_round_history.csv", "fedavg_reliability_final_metrics.csv", "fedavg_reliability_config.csv"]:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    for png in output_dir.glob("*.png"):
        png.unlink()

    command = [
        sys.executable, str(RUNNER),
        "--scenario", scenario,
        "--rounds", str(args.rounds),
        "--local-epochs", str(args.local_epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--seed", str(seed),
    ]
    print("\n" + "=" * 88)
    print(f"RUNNING reliability scenario={scenario}, seed={seed}")
    print("=" * 88)
    result = subprocess.run(command, cwd=SCRIPT_DIR, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Reliability experiment failed for seed {seed}: exit code {result.returncode}")

    final_file = output_dir / "fedavg_reliability_final_metrics.csv"
    if not final_file.exists():
        raise FileNotFoundError(f"Missing final metrics: {final_file}")

    saved_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["fedavg_reliability_round_history.csv", "fedavg_reliability_final_metrics.csv", "fedavg_reliability_config.csv"]:
        source = output_dir / filename
        if source.exists():
            shutil.copy2(source, saved_dir / filename)
    for png in output_dir.glob("*.png"):
        shutil.copy2(png, saved_dir / png.name)


def load_results(scenario, seeds):
    rows = []
    for seed in seeds:
        path = seed_dir(scenario, seed) / "fedavg_reliability_final_metrics.csv"
        df = pd.read_csv(path)
        metrics = dict(zip(df["metric"].astype(str), df["value"]))
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "accuracy": float(metrics["final_accuracy"]),
            "precision": float(metrics["final_precision"]),
            "recall": float(metrics["final_recall"]),
            "f1": float(metrics["final_f1"]),
            "roc_auc": float(metrics["final_roc_auc"]),
            "global_test_loss": float(metrics["final_test_loss"]),
            "total_communication_bytes": float(metrics["total_communication_bytes"]),
            "total_round_time_seconds": float(metrics["total_round_time_seconds"]),
        })
    return pd.DataFrame(rows)


def save_summary(per_seed, scenario, seeds, args):
    output_dir = scenario_dir(scenario)
    per_seed.to_csv(output_dir / "multi_seed_per_seed_results.csv", index=False)
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "global_test_loss", "total_communication_bytes", "total_round_time_seconds"]
    summary = []
    for metric in metrics:
        values = per_seed[metric].to_numpy(float)
        summary.append({
            "scenario": scenario,
            "seeds": ",".join(map(str, seeds)),
            "n_seeds": len(seeds),
            "metric": metric,
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        })
    pd.DataFrame(summary).to_csv(output_dir / "multi_seed_summary.csv", index=False)

    compact = {"scenario": scenario, "n_seeds": len(seeds)}
    for metric in metrics:
        values = per_seed[metric].to_numpy(float)
        compact[f"{metric}_mean"] = float(np.mean(values))
        compact[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    pd.DataFrame([compact]).to_csv(output_dir / "multi_seed_compact_summary.csv", index=False)

    pd.DataFrame({
        "parameter": ["scenario", "seeds", "n_seeds", "rounds", "local_epochs", "batch_size", "learning_rate", "client_partition", "reliability_schedule"],
        "value": [scenario, ",".join(map(str, seeds)), len(seeds), args.rounds, args.local_epochs, args.batch_size, args.learning_rate, "fixed unbalanced non-IID", "fixed across seeds"],
    }).to_csv(output_dir / "multi_seed_config.csv", index=False)

    print("\n" + "=" * 88)
    print("MULTI-SEED RELIABILITY EXPERIMENT COMPLETE")
    print("=" * 88)
    print(f"Scenario: {scenario}")
    print(f"Seeds   : {seeds}")
    print("\nFinal-round mean +/- std:")
    for metric in metrics:
        values = per_seed[metric].to_numpy(float)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0
        print(f"  {metric:28s}: {np.mean(values):.6f} +/- {std:.6f}")
    print(f"\nPer-seed results : {output_dir / 'multi_seed_per_seed_results.csv'}")
    print(f"Summary          : {output_dir / 'multi_seed_summary.csv'}")
    print(f"Compact summary  : {output_dir / 'multi_seed_compact_summary.csv'}")
    print(f"Configuration    : {output_dir / 'multi_seed_config.csv'}")
    print("=" * 88)


def main():
    args = parse_args()
    seeds = parse_seeds(args.seeds)
    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing runner: {RUNNER}")
    for seed in seeds:
        run_one_seed(args.scenario, seed, args)
    per_seed = load_results(args.scenario, seeds)
    save_summary(per_seed, args.scenario, seeds, args)


if __name__ == "__main__":
    main()
