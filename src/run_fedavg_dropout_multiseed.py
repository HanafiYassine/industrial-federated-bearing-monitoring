"""
run_fedavg_dropout_multiseed.py

Run the existing deterministic FedAvg client-dropout experiment
across five training seeds without changing the dropout schedules.

The underlying run_fedavg_dropout.py remains untouched.

Example:
    python run_fedavg_dropout_multiseed.py --rate 0.2 --seeds 42,123,2024,3407,7777
    python run_fedavg_dropout_multiseed.py --rate 0.4 --seeds 42,123,2024,3407,7777
    python run_fedavg_dropout_multiseed.py --rate 0.5 --seeds 42,123,2024,3407,7777
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
RUNNER = SCRIPT_DIR / "run_fedavg_dropout.py"

RESULTS_ROOT = (
    SCRIPT_DIR
    / "../results/federated/dataset2_fedavg_dropout"
).resolve()

DEFAULT_SEEDS = [42, 123, 2024, 3407, 7777]
VALID_RATES = [0.0, 0.2, 0.4, 0.5]

FINAL_FILE = "fedavg_dropout_final_metrics.csv"
HISTORY_FILE = "fedavg_dropout_round_history.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FedAvg client-dropout experiments across multiple seeds."
    )

    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Dropout rate: 0.0, 0.2, 0.4 or 0.5.",
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
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
    )

    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not rerun seeds whose saved result already exists.",
    )

    return parser.parse_args()


def parse_seeds(raw: str) -> list[int]:
    seeds = []

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        try:
            seed = int(item)
        except ValueError as exc:
            raise ValueError(
                f"Invalid seed '{item}'."
            ) from exc

        seeds.append(seed)

    if not seeds:
        raise ValueError("At least one seed is required.")

    return list(dict.fromkeys(seeds))


def rate_percentage(rate: float) -> int:
    return int(round(rate * 100))


def scenario_dir(rate: float) -> Path:
    return RESULTS_ROOT / f"dropout_{rate_percentage(rate)}pct"


def seed_dir(rate: float, seed: int) -> Path:
    return (
        scenario_dir(rate)
        / "seeds"
        / f"seed_{seed}"
    )


def run_one_seed(
    rate: float,
    seed: int,
    args: argparse.Namespace,
) -> None:

    if not RUNNER.exists():
        raise FileNotFoundError(
            f"Missing runner:\n{RUNNER}"
        )

    output_dir = scenario_dir(rate)
    seeds_root = output_dir / "seeds"
    saved_dir = seed_dir(rate, seed)

    seeds_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    if saved_dir.exists():
        if args.keep_existing:
            print(f"Keeping existing results: {saved_dir}")
            return
        shutil.rmtree(saved_dir)

    # Remove only the previous top-level outputs produced by
    # run_fedavg_dropout.py. Never remove the seeds directory.
    for filename in [
        HISTORY_FILE,
        FINAL_FILE,
    ]:
        path = output_dir / filename
        if path.exists():
            path.unlink()

    for png_file in output_dir.glob("*.png"):
        png_file.unlink()

    command = [
        sys.executable,
        str(RUNNER),
        "--rate",
        str(rate),
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
    print(f"RUNNING DROPOUT RATE {rate:.2f}, SEED {seed}")
    print("=" * 88)

    completed = subprocess.run(
        command,
        cwd=SCRIPT_DIR,
        check=False,
    )

    if completed.returncode != 0:
        raise RuntimeError(
            f"Dropout experiment failed for rate={rate}, "
            f"seed={seed}, exit code={completed.returncode}."
        )

    if not output_dir.exists():
        raise FileNotFoundError(
            f"Expected output directory not found:\n{output_dir}"
        )

    saved_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Copy the final metrics and round history into the seed folder.
    for filename in [
        HISTORY_FILE,
        FINAL_FILE,
    ]:
        source = output_dir / filename

        if source.exists():
            shutil.copy2(
                source,
                saved_dir / filename,
            )
        else:
            raise FileNotFoundError(
                f"Expected output file missing:\n{source}"
            )

    # Preserve figures for this seed.
    for png_file in output_dir.glob("*.png"):
        shutil.copy2(
            png_file,
            saved_dir / png_file.name,
        )

    print(f"Saved seed {seed} results to:")
    print(saved_dir)


def load_final_results(
    rate: float,
    seeds: list[int],
) -> pd.DataFrame:

    rows = []

    for seed in seeds:
        final_file = (
            seed_dir(rate, seed)
            / FINAL_FILE
        )

        if not final_file.exists():
            raise FileNotFoundError(
                f"Missing result for seed {seed}:\n{final_file}"
            )

        df = pd.read_csv(final_file)

        if df.empty:
            raise ValueError(
                f"Empty result file:\n{final_file}"
            )

        metrics = dict(
            zip(
                df["metric"].astype(str),
                df["value"],
            )
        )

        rows.append(
            {
                "dropout_rate": rate,
                "dropout_percentage": rate_percentage(rate),
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
                "total_communication_mib": float(
                    metrics["total_communication_mib"]
                ),
                "total_round_time_seconds": float(
                    metrics["total_round_time_seconds"]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_summary(
    per_seed: pd.DataFrame,
    rate: float,
    seeds: list[int],
    args: argparse.Namespace,
) -> None:

    output_dir = scenario_dir(rate)
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
        "total_communication_mib",
        "total_round_time_seconds",
    ]

    rows = []

    for metric in metric_columns:
        values = per_seed[metric].to_numpy(
            dtype=float
        )

        rows.append(
            {
                "dropout_rate": rate,
                "dropout_percentage": rate_percentage(rate),
                "n_seeds": len(seeds),
                "metric": metric,
                "mean": float(np.mean(values)),
                "std": float(
                    np.std(values, ddof=1)
                ) if len(values) > 1 else 0.0,
            }
        )

    summary = pd.DataFrame(rows)

    summary_file = (
        output_dir
        / "multi_seed_summary.csv"
    )
    summary.to_csv(
        summary_file,
        index=False,
    )

    compact = {
        "dropout_rate": rate,
        "dropout_percentage": rate_percentage(rate),
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
                "dropout_rate",
                "dropout_percentage",
                "seeds",
                "n_seeds",
                "rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "client_partition",
                "participation_schedule",
                "global_test",
            ],
            "value": [
                rate,
                rate_percentage(rate),
                ",".join(str(s) for s in seeds),
                len(seeds),
                args.rounds,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                "unbalanced non-IID",
                "fixed across seeds",
                "fixed global test set",
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
    print("MULTI-SEED DROPOUT EXPERIMENT COMPLETE")
    print("=" * 88)
    print(
        f"Dropout rate: {rate:.2f} "
        f"({rate_percentage(rate)}%)"
    )
    print(f"Seeds      : {seeds}")
    print()
    print("Final-round mean ± std:")

    for metric in metric_columns:
        values = per_seed[metric].to_numpy(
            dtype=float
        )

        mean = np.mean(values)
        std = (
            np.std(values, ddof=1)
            if len(values) > 1
            else 0.0
        )

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


def main() -> None:
    args = parse_args()

    if not any(
        abs(args.rate - valid) < 1e-9
        for valid in VALID_RATES
    ):
        raise ValueError(
            "Dropout rate must be one of: "
            "0.0, 0.2, 0.4, 0.5"
        )

    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0.")

    if args.local_epochs <= 0:
        raise ValueError("--local-epochs must be > 0.")

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0.")

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    seeds = parse_seeds(args.seeds)

    for seed in seeds:
        run_one_seed(
            rate=args.rate,
            seed=seed,
            args=args,
        )

    per_seed = load_final_results(
        rate=args.rate,
        seeds=seeds,
    )

    save_summary(
        per_seed=per_seed,
        rate=args.rate,
        seeds=seeds,
        args=args,
    )


if __name__ == "__main__":
    main()
