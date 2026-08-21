"""
run_fedavg_multiseed.py

Multi-seed wrapper for the validated run_fedavg.py experiment.

Purpose
-------
Run the existing FedAvg experiment repeatedly with independent random seeds
without changing the original run_fedavg.py implementation or overwriting
its seed-42 outputs.

The underlying experiment remains exactly the same:
    - same client CSV files
    - same held-out global test set
    - same global StandardScaler fitting procedure
    - same model and FedAvg implementation
    - same number of clients / rounds / local epochs unless overridden

Each seed is saved under:
    <output-root>/<partition>/seed_<seed>/

A cross-seed summary is also written to:
    <output-root>/<partition>/multi_seed_summary.csv
    <output-root>/<partition>/multi_seed_round_summary.csv
    <output-root>/<partition>/multi_seed_config.csv

Example (from src/):
    python run_fedavg_multiseed.py --partition iid --seeds 42,123,2024,3407,7777

Single-seed verification:
    python run_fedavg_multiseed.py --partition iid --seeds 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_fedavg import (
    BATCH_SIZE,
    DEVICE,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    ROUNDS,
    N_CLIENTS,
    OUTPUT_ROOT,
    create_figures,
    run_experiment,
)

DEFAULT_SEEDS = [42, 123, 2024, 3407, 7777]


# ============================================================
# ARGUMENTS
# ============================================================


def parse_seed_list(value: str) -> list[int]:
    """Parse a comma-separated list of integer seeds."""
    raw_values = [item.strip() for item in value.split(",")]

    if not raw_values or any(item == "" for item in raw_values):
        raise argparse.ArgumentTypeError(
            "Seeds must be a comma-separated list such as 42,123,2024."
        )

    try:
        seeds = [int(item) for item in raw_values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Every seed must be an integer."
        ) from exc

    if len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError(
            "Seed values must be unique."
        )

    return seeds



def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the existing FedAvg experiment across multiple seeds."
    )

    parser.add_argument(
        "--partition",
        choices=["iid", "noniid", "balanced_noniid"],
        required=True,
        help="Client partition to evaluate.",
    )

    parser.add_argument(
        "--seeds",
        type=parse_seed_list,
        default=DEFAULT_SEEDS,
        help="Comma-separated random seeds.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="Number of FL communication rounds.",
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=LOCAL_EPOCHS,
        help="Local epochs per client per round.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Local training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
        help="Local Adam learning rate.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for multi-seed outputs.",
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================


def validate_args(args):
    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0.")

    if args.local_epochs <= 0:
        raise ValueError("--local-epochs must be > 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    if args.learning_rate <= 0:
        raise ValueError("--learning-rate must be > 0.")

    if not args.seeds:
        raise ValueError("At least one seed is required.")


# ============================================================
# PER-SEED OUTPUTS
# ============================================================


def save_seed_results(
    history: pd.DataFrame,
    partition: str,
    seed: int,
    args,
) -> Path:
    """Save one seed's complete round history and final metrics."""
    output_dir = (
        args.output_root
        / partition
        / f"seed_{seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    history = history.copy()
    history["seed"] = seed

    history.to_csv(
        output_dir / "fedavg_round_history.csv",
        index=False,
    )

    final = history.iloc[-1]

    final_metrics = pd.DataFrame(
        {
            "metric": [
                "partition",
                "seed",
                "rounds",
                "clients",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "device",
                "final_accuracy",
                "final_precision",
                "final_recall",
                "final_f1",
                "final_roc_auc",
                "final_test_loss",
                "total_communication_bytes",
                "total_round_time_seconds",
            ],
            "value": [
                partition,
                seed,
                args.rounds,
                N_CLIENTS,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                str(DEVICE),
                final["accuracy"],
                final["precision"],
                final["recall"],
                final["f1"],
                final["roc_auc"],
                final["global_test_loss"],
                history["total_communication_bytes"].sum(),
                history["round_time_seconds"].sum(),
            ],
        }
    )

    final_metrics.to_csv(
        output_dir / "fedavg_final_metrics.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "partition",
                "seed",
                "clients",
                "rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "device",
                "model",
                "input_features",
                "hidden_layers",
                "dropout",
            ],
            "value": [
                partition,
                seed,
                N_CLIENTS,
                args.rounds,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                str(DEVICE),
                "FederatedMLP",
                36,
                "128-64-32",
                0.40,
            ],
        }
    )

    config.to_csv(
        output_dir / "fedavg_config.csv",
        index=False,
    )

    # Reuse the established publication figures for each seed.
    create_figures(
        history.drop(columns=["seed"]),
        partition,
        output_dir,
    )

    return output_dir


# ============================================================
# CROSS-SEED SUMMARY
# ============================================================


def build_seed_summary(seed_histories: list[pd.DataFrame]) -> pd.DataFrame:
    """Build final-round per-seed metrics plus mean/std across seeds."""
    rows: list[dict] = []

    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "global_test_loss",
        "total_communication_bytes",
        "round_time_seconds",
    ]

    for history in seed_histories:
        final = history.iloc[-1]
        rows.append(
            {
                "seed": int(final["seed"]),
                **{
                    metric: float(final[metric])
                    for metric in metric_columns
                },
            }
        )

    per_seed = pd.DataFrame(rows).sort_values("seed")

    summary_rows = []

    for metric in metric_columns:
        values = per_seed[metric].to_numpy(dtype=float)

        summary_rows.append(
            {
                "metric": metric,
                "n_seeds": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        )

    summary = pd.DataFrame(summary_rows)

    return per_seed, summary



def build_round_summary(seed_histories: list[pd.DataFrame]) -> pd.DataFrame:
    """Aggregate round-level metrics across seeds."""
    frames = []

    for history in seed_histories:
        df = history.copy()
        if "seed" not in df.columns:
            raise ValueError("Seed column is missing from round history.")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    metric_columns = [
        "mean_local_loss",
        "global_test_loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "round_time_seconds",
        "total_communication_bytes",
    ]

    rows = []

    grouped = combined.groupby("round", sort=True)

    for round_number, group in grouped:
        row = {
            "round": int(round_number),
            "n_seeds": int(group["seed"].nunique()),
        }

        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows)



def save_cross_seed_summary(
    seed_histories: list[pd.DataFrame],
    partition: str,
    seeds: list[int],
    args,
):
    output_dir = args.output_root / partition
    output_dir.mkdir(parents=True, exist_ok=True)

    per_seed, summary = build_seed_summary(seed_histories)
    round_summary = build_round_summary(seed_histories)

    per_seed.to_csv(
        output_dir / "multi_seed_per_seed_results.csv",
        index=False,
    )

    summary.to_csv(
        output_dir / "multi_seed_summary.csv",
        index=False,
    )

    round_summary.to_csv(
        output_dir / "multi_seed_round_summary.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "partition",
                "seeds",
                "n_seeds",
                "clients",
                "rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "device",
                "client_partitions_fixed_across_seeds",
            ],
            "value": [
                partition,
                ",".join(str(seed) for seed in seeds),
                len(seeds),
                N_CLIENTS,
                args.rounds,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                str(DEVICE),
                True,
            ],
        }
    )

    config.to_csv(
        output_dir / "multi_seed_config.csv",
        index=False,
    )

    return per_seed, summary, round_summary


# ============================================================
# MAIN
# ============================================================


def main():
    args = parse_args()
    validate_args(args)

    print("=" * 80)
    print("MULTI-SEED FEDAVG VALIDATION")
    print("=" * 80)
    print(f"Partition       : {args.partition}")
    print(f"Seeds           : {args.seeds}")
    print(f"Rounds          : {args.rounds}")
    print(f"Local epochs    : {args.local_epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.learning_rate}")
    print(f"Device          : {DEVICE}")
    print("=" * 80)

    seed_histories: list[pd.DataFrame] = []

    for seed in args.seeds:
        print()
        print("#" * 80)
        print(f"RUNNING SEED {seed}")
        print("#" * 80)

        history = run_experiment(
            partition=args.partition,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=seed,
        )

        history = history.copy()
        history["seed"] = seed
        seed_histories.append(history)

        save_seed_results(
            history,
            args.partition,
            seed,
            args,
        )

    per_seed, summary, round_summary = save_cross_seed_summary(
        seed_histories,
        args.partition,
        args.seeds,
        args,
    )

    print()
    print("=" * 80)
    print("MULTI-SEED FEDAVG COMPLETE")
    print("=" * 80)
    print(f"Partition: {args.partition}")
    print(f"Seeds    : {args.seeds}")
    print()
    print("Final-round mean ± std:")

    for _, row in summary.iterrows():
        metric = row["metric"]
        mean_value = row["mean"]
        std_value = row["std"]
        print(
            f"  {metric:28s}: {mean_value:.6f} ± {std_value:.6f}"
        )

    output_dir = args.output_root / args.partition
    print()
    print("Per-seed results :", output_dir / "multi_seed_per_seed_results.csv")
    print("Summary          :", output_dir / "multi_seed_summary.csv")
    print("Round summary    :", output_dir / "multi_seed_round_summary.csv")
    print("Configuration    :", output_dir / "multi_seed_config.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
