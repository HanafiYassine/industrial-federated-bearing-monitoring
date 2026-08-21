"""
run_fedavg_reliability_multiseed.py

Run standard FedAvg under one industrial reliability scenario
across multiple training seeds while preserving each run's outputs.

This wrapper intentionally reuses the existing validated
run_fedavg_reliability.py without changing its experiment logic.

Example from src/:
    python run_fedavg_reliability_multiseed.py --scenario sensor_noise --seeds 42,123,2024,3407,7777
    python run_fedavg_reliability_multiseed.py --scenario stale_updates --seeds 42,123,2024,3407,7777
    python run_fedavg_reliability_multiseed.py --scenario persistent_failure --seeds 42,123,2024,3407,7777
    python run_fedavg_reliability_multiseed.py --scenario mixed --seeds 42,123,2024,3407,7777
    python run_fedavg_reliability_multiseed.py --scenario mixed_full --seeds 42,123,2024,3407,7777

The reliability schedule is kept fixed across seeds. Therefore this
experiment measures training stochasticity while the reliability
scenario and client partition remain unchanged.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_fedavg_reliability.py"

RESULTS_ROOT = (
    SCRIPT_DIR
    / "../results/federated/dataset2_fedavg_reliability"
).resolve()


DEFAULT_SEEDS = [42, 123, 2024, 3407, 7777]

SCENARIOS = [
    "clean",
    "sensor_noise",
    "stale_updates",
    "persistent_failure",
    "mixed",
    "mixed_full",
]


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedAvg reliability experiments across multiple seeds."
    )

    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        required=True,
        help="Industrial reliability scenario.",
    )

    parser.add_argument(
        "--seeds",
        default="42,123,2024,3407,7777",
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=15,
        help="Number of FL rounds.",
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Local epochs per client per round.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Local training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Local Adam learning rate.",
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Keep existing seed result folders instead of deleting them.",
    )

    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            seed = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid seed '{item}'. Seeds must be integers."
            ) from exc

        seeds.append(seed)

    if not seeds:
        raise ValueError("At least one seed is required.")

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(seeds))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def seed_output_dir(scenario: str, seed: int) -> Path:
    return RESULTS_ROOT / scenario / "seeds" / f"seed_{seed}"


def live_output_dir(scenario: str) -> Path:
    return RESULTS_ROOT / scenario


def run_one_seed(
    scenario: str,
    seed: int,
    args: argparse.Namespace,
) -> None:

    if not RUNNER.exists():
        raise FileNotFoundError(
            f"Missing runner:\n{RUNNER}"
        )

    output_dir = live_output_dir(scenario)
    seeds_root = (
        output_dir / "seeds"
    )
    saved_dir = (
        seeds_root / f"seed_{seed}"
    )

    # The existing runner always writes to:
    #   dataset2_fedavg_reliability/<scenario>/
    #
    # Do NOT move output_dir itself because saved_dir is inside it.
    # Instead, preserve only this seed's generated result files.

    seeds_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if saved_dir.exists():
        if args.keep_existing:
            print(
                f"Keeping existing results: {saved_dir}"
            )
            return

        shutil.rmtree(saved_dir)

    # Remove only previous top-level generated result files.
    # Never delete the seeds directory or its contents.
    generated_files = [
        "fedavg_reliability_round_history.csv",
        "fedavg_reliability_final_metrics.csv",
        "fedavg_reliability_config.csv",
    ]

    for filename in generated_files:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    # Remove top-level PNGs generated by the experiment.
    for png_file in output_dir.glob("*.png"):
        png_file.unlink()

    command = [
        sys.executable,
        str(RUNNER),
        "--scenario",
        scenario,
        "--rounds",
        str(args.rounds),
        "--local-epochs",
        str(args.local_epochs),
        "--batch-size",
        str(args.batch_size),
        "--learning-rate",
        str(args.learning_rate),
        "--seed",
        str(seed),
    ]

    print("\n" + "=" * 88)
    print(f"RUNNING SEED {seed}")
    print("=" * 88)
    print(
        " ".join(
            f'"{x}"' if " " in x else x
            for x in command
        )
    )

    completed = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Reliability experiment failed for seed {seed} "
            f"with exit code {completed.returncode}."
        )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Expected runner output not found:\n"
            f"{output_dir}"
        )

    # Create the seed-specific directory.
    saved_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy the generated files into the seed directory.
    for filename in generated_files:
        source = output_dir / filename

        if source.exists():
            shutil.copy2(
                source,
                saved_dir / filename,
            )

    # Preserve generated figures for that seed.
    for png_file in output_dir.glob("*.png"):
        shutil.copy2(
            png_file,
            saved_dir / png_file.name,
        )

    print(
        f"Saved seed {seed} results to:\n"
        f"{saved_dir}"
    )


def load_seed_final_metrics(
    scenario: str,
    seeds: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []

    for seed in seeds:
        seed_dir = seed_output_dir(
            scenario,
            seed,
        )

        final_file = (
            seed_dir
            / "fedavg_reliability_final_metrics.csv"
        )

        if not final_file.exists():
            raise FileNotFoundError(
                f"Missing final metrics for seed {seed}:\n{final_file}"
            )

        df = pd.read_csv(final_file)

        if df.empty:
            raise ValueError(
                f"Final metrics file is empty:\n{final_file}"
            )

        # The existing runner stores metric/value rows.
        metrics = dict(
            zip(
                df["metric"].astype(str),
                df["value"],
            )
        )

        rows.append(
            {
                "scenario": scenario,
                "seed": seed,
                "accuracy": float(metrics["final_accuracy"]),
                "precision": float(metrics["final_precision"]),
                "recall": float(metrics["final_recall"]),
                "f1": float(metrics["final_f1"]),
                "roc_auc": float(metrics["final_roc_auc"]),
                "global_test_loss": float(
                    metrics["final_test_loss"]
                ),
                "total_communication_bytes": float(
                    metrics["total_communication_bytes"]
                ),
                "total_round_time_seconds": float(
                    metrics["total_round_time_seconds"]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_summary(
    per_seed: pd.DataFrame,
    scenario: str,
    seeds: list[int],
    args: argparse.Namespace,
) -> None:
    output_dir = RESULTS_ROOT / scenario
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_seed_file = (
        output_dir
        / "multi_seed_per_seed_results.csv"
    )
    per_seed.to_csv(
        per_seed_file,
        index=False,
    )

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "global_test_loss",
        "total_communication_bytes",
        "total_round_time_seconds",
    ]

    summary_rows = []

    for metric in metric_columns:
        values = per_seed[metric].to_numpy(
            dtype=float
        )

        summary_rows.append(
            {
                "scenario": scenario,
                "seeds": ",".join(str(s) for s in seeds),
                "n_seeds": len(seeds),
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary_file = (
        output_dir
        / "multi_seed_summary.csv"
    )
    summary.to_csv(
        summary_file,
        index=False,
    )

    # A compact one-row-per-scenario table for manuscript use.
    compact = {
        "scenario": scenario,
        "n_seeds": len(seeds),
    }

    for metric in metric_columns:
        values = per_seed[metric].to_numpy(
            dtype=float
        )

        compact[f"{metric}_mean"] = float(
            np.mean(values)
        )
        compact[f"{metric}_std"] = float(
            np.std(values, ddof=1)
        ) if len(values) > 1 else 0.0

    compact_file = (
        output_dir
        / "multi_seed_compact_summary.csv"
    )
    pd.DataFrame([compact]).to_csv(
        compact_file,
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "scenario",
                "seeds",
                "n_seeds",
                "rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "partition",
                "global_test",
                "schedule_policy",
            ],
            "value": [
                scenario,
                ",".join(str(s) for s in seeds),
                len(seeds),
                args.rounds,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                "noniid",
                "fixed global test set",
                "fixed reliability schedule across seeds",
            ],
        }
    )

    config_file = (
        output_dir
        / "multi_seed_config.csv"
    )
    config.to_csv(
        config_file,
        index=False,
    )

    print("\n" + "=" * 88)
    print("MULTI-SEED RELIABILITY EXPERIMENT COMPLETE")
    print("=" * 88)
    print(f"Scenario: {scenario}")
    print(f"Seeds   : {seeds}")
    print()
    print("Final-round mean ± std:")

    for metric in metric_columns:
        values = per_seed[metric].to_numpy(
            dtype=float
        )
        mean = np.mean(values)
        std = np.std(values, ddof=1) if len(values) > 1 else 0.0

        print(
            f"  {metric:30s}: "
            f"{mean:.6f} ± {std:.6f}"
        )

    print()
    print(f"Per-seed results : {per_seed_file}")
    print(f"Summary          : {summary_file}")
    print(f"Compact summary  : {compact_file}")
    print(f"Configuration    : {config_file}")
    print("=" * 88)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0.")

    if args.local_epochs <= 0:
        raise ValueError("--local-epochs must be > 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0.")

    seeds = parse_seeds(args.seeds)

    # Sanity checks.
    if not RUNNER.exists():
        raise FileNotFoundError(
            f"Cannot find run_fedavg_reliability.py:\n{RUNNER}"
        )

    for seed in seeds:
        run_one_seed(
            scenario=args.scenario,
            seed=seed,
            args=args,
        )

    per_seed = load_seed_final_metrics(
        scenario=args.scenario,
        seeds=seeds,
    )

    save_summary(
        per_seed=per_seed,
        scenario=args.scenario,
        seeds=seeds,
        args=args,
    )


if __name__ == "__main__":
    main()
