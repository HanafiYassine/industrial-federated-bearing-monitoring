"""
evaluate_fedavg_stress_tests.py

Consolidate the controlled FedAvg stress tests for NASA IMS Dataset 2.

Inputs:
    ../results/federated/dataset2_fedavg/
        iid/
        balanced_noniid/
        noniid/

    ../results/federated/dataset2_fedavg_dropout/
        dropout_20pct/
        dropout_40pct/
        dropout_50pct/

    ../results/federated/dataset2_fedavg_reliability/
        sensor_noise/
        stale_updates/
        persistent_failure/
        mixed/
        mixed_full/

Outputs:
    ../results/federated/dataset2_fedavg/stress_evaluation/
        fedavg_stress_summary.csv
        fedavg_stress_round_history.csv
        fedavg_stress_report.txt
        final_f1_comparison.png
        final_auc_comparison.png
        final_accuracy_comparison.png
        communication_comparison.png
        dropout_curve.png
        reliability_comparison.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE = Path("../results/federated")

FEDAVG_ROOT = BASE / "dataset2_fedavg"
DROPOUT_ROOT = BASE / "dataset2_fedavg_dropout"
RELIABILITY_ROOT = BASE / "dataset2_fedavg_reliability"

OUTPUT = FEDAVG_ROOT / "stress_evaluation"


# ============================================================
# EXPERIMENT REGISTRY
# ============================================================

EXPERIMENTS = [
    {
        "name": "IID",
        "category": "distribution",
        "path": FEDAVG_ROOT / "iid" / "fedavg_round_history.csv",
    },
    {
        "name": "Balanced Non-IID",
        "category": "distribution",
        "path": FEDAVG_ROOT / "balanced_noniid" / "fedavg_round_history.csv",
    },
    {
        "name": "Unbalanced Non-IID",
        "category": "distribution",
        "path": FEDAVG_ROOT / "noniid" / "fedavg_round_history.csv",
    },
    {
        "name": "Random Dropout 20%",
        "category": "dropout",
        "path": DROPOUT_ROOT / "dropout_20pct" / "fedavg_dropout_round_history.csv",
    },
    {
        "name": "Random Dropout 40%",
        "category": "dropout",
        "path": DROPOUT_ROOT / "dropout_40pct" / "fedavg_dropout_round_history.csv",
    },
    {
        "name": "Random Dropout 50%",
        "category": "dropout",
        "path": DROPOUT_ROOT / "dropout_50pct" / "fedavg_dropout_round_history.csv",
    },
    {
        "name": "Sensor Noise",
        "category": "reliability",
        "path": RELIABILITY_ROOT / "sensor_noise" / "fedavg_reliability_round_history.csv",
    },
    {
        "name": "Stale Updates",
        "category": "reliability",
        "path": RELIABILITY_ROOT / "stale_updates" / "fedavg_reliability_round_history.csv",
    },
    {
        "name": "Persistent Failure",
        "category": "reliability",
        "path": RELIABILITY_ROOT / "persistent_failure" / "fedavg_reliability_round_history.csv",
    },
    {
        "name": "Noise + Stale",
        "category": "reliability",
        "path": RELIABILITY_ROOT / "mixed" / "fedavg_reliability_round_history.csv",
    },
    {
        "name": "Full Mixed",
        "category": "reliability",
        "path": RELIABILITY_ROOT / "mixed_full" / "fedavg_reliability_round_history.csv",
    },
]


# ============================================================
# LOAD
# ============================================================

def load_history(item):
    path = item["path"]

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = {
        "round",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "global_test_loss",
        "total_communication_bytes",
        "round_time_seconds",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["experiment"] = item["name"]
    df["category"] = item["category"]

    return df.sort_values("round").reset_index(drop=True)


def build_histories():
    histories = {}

    for item in EXPERIMENTS:
        df = load_history(item)

        if df is not None:
            histories[item["name"]] = df
        else:
            print(
                f"WARNING: missing {item['name']} -> {item['path']}"
            )

    if not histories:
        raise RuntimeError("No FedAvg stress-test history files found.")

    return histories


# ============================================================
# SUMMARY
# ============================================================

def build_summary(histories):
    rows = []

    for name, df in histories.items():
        final = df.iloc[-1]

        best_f1 = df.loc[df["f1"].idxmax()]
        best_auc = df.loc[df["roc_auc"].idxmax()]
        best_acc = df.loc[df["accuracy"].idxmax()]

        rows.append(
            {
                "experiment": name,
                "category": df["category"].iloc[0],
                "rounds": int(df["round"].max()),
                "final_accuracy": float(final["accuracy"]),
                "final_precision": float(final["precision"]),
                "final_recall": float(final["recall"]),
                "final_f1": float(final["f1"]),
                "final_roc_auc": float(final["roc_auc"]),
                "final_test_loss": float(final["global_test_loss"]),
                "best_f1": float(best_f1["f1"]),
                "best_f1_round": int(best_f1["round"]),
                "best_roc_auc": float(best_auc["roc_auc"]),
                "best_roc_auc_round": int(best_auc["round"]),
                "best_accuracy": float(best_acc["accuracy"]),
                "best_accuracy_round": int(best_acc["round"]),
                "total_communication_mib": float(
                    df["total_communication_bytes"].sum() / (1024 ** 2)
                ),
                "total_round_time_seconds": float(
                    df["round_time_seconds"].sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def save_bar(
    summary,
    metric,
    ylabel,
    title,
    filename,
):
    fig, ax = plt.subplots(figsize=(13, 6))

    values = summary[metric].to_numpy()
    names = summary["experiment"].tolist()
    x = np.arange(len(names))

    ax.bar(x, values)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_dropout_curve(summary):
    rows = summary[
        summary["category"] == "dropout"
    ].copy()

    baseline_match = summary[
        summary["experiment"] == "Unbalanced Non-IID"
    ]

    if not baseline_match.empty:
        base = baseline_match.iloc[0]

        rows = pd.concat(
            [
                pd.DataFrame(
                    [{
                        "dropout_rate": 0.0,
                        "final_f1": base["final_f1"],
                        "final_roc_auc": base["final_roc_auc"],
                    }]
                ),
                rows.assign(
                    dropout_rate=rows["experiment"]
                    .str.extract(r"(\d+)")
                    .astype(float)[0]
                    / 100.0
                )[
                    [
                        "dropout_rate",
                        "final_f1",
                        "final_roc_auc",
                    ]
                ],
            ],
            ignore_index=True,
        )

        rows = rows.sort_values("dropout_rate")

        fig, ax = plt.subplots(figsize=(9, 6))

        ax.plot(
            rows["dropout_rate"] * 100,
            rows["final_f1"],
            marker="o",
            label="F1",
        )

        ax.plot(
            rows["dropout_rate"] * 100,
            rows["final_roc_auc"],
            marker="o",
            label="ROC-AUC",
        )

        ax.set_xlabel("Random client dropout (%)")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.set_title("FedAvg Robustness to Random Client Dropout")
        ax.grid(True, alpha=0.25)
        ax.legend()

        fig.tight_layout()
        fig.savefig(
            OUTPUT / "dropout_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_reliability_comparison(summary):
    rows = summary[
        summary["category"] == "reliability"
    ].copy()

    if rows.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(rows))
    width = 0.36

    ax.bar(
        x - width / 2,
        rows["final_f1"],
        width,
        label="F1",
    )

    ax.bar(
        x + width / 2,
        rows["final_roc_auc"],
        width,
        label="ROC-AUC",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        rows["experiment"],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("FedAvg Under Industrial Client Reliability Stress")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        OUTPUT / "reliability_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ============================================================
# REPORT
# ============================================================

def write_report(summary):
    lines = [
        "=" * 90,
        "FEDAVG STRESS-TEST EVALUATION",
        "=" * 90,
        "",
        "All values are our controlled experimental results.",
        "Published-paper values are intentionally not mixed into this table.",
        "",
    ]

    for _, row in summary.iterrows():
        lines.extend(
            [
                row["experiment"],
                "-" * 90,
                f"Final accuracy       : {row['final_accuracy']:.6f}",
                f"Final precision      : {row['final_precision']:.6f}",
                f"Final recall         : {row['final_recall']:.6f}",
                f"Final F1             : {row['final_f1']:.6f}",
                f"Final ROC-AUC        : {row['final_roc_auc']:.6f}",
                f"Final test loss      : {row['final_test_loss']:.6f}",
                f"Best F1              : {row['best_f1']:.6f} "
                f"(round {int(row['best_f1_round'])})",
                f"Best ROC-AUC         : {row['best_roc_auc']:.6f} "
                f"(round {int(row['best_roc_auc_round'])})",
                f"Communication        : {row['total_communication_mib']:.4f} MiB",
                "",
            ]
        )

    # Explicit mixed-full degradation relative to clean unbalanced.
    clean = summary[
        summary["experiment"] == "Unbalanced Non-IID"
    ]
    mixed = summary[
        summary["experiment"] == "Full Mixed"
    ]

    if not clean.empty and not mixed.empty:
        c = clean.iloc[0]
        m = mixed.iloc[0]

        lines.extend(
            [
                "KEY CONTROLLED COMPARISON",
                "-" * 90,
                f"Full Mixed ΔF1 vs Clean Non-IID   : "
                f"{m['final_f1'] - c['final_f1']:+.6f}",
                f"Full Mixed ΔRecall vs Clean      : "
                f"{m['final_recall'] - c['final_recall']:+.6f}",
                f"Full Mixed ΔROC-AUC vs Clean     : "
                f"{m['final_roc_auc'] - c['final_roc_auc']:+.6f}",
                f"Full Mixed ΔTest Loss vs Clean   : "
                f"{m['final_test_loss'] - c['final_test_loss']:+.6f}",
                "",
            ]
        )

    path = OUTPUT / "fedavg_stress_report.txt"
    path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    histories = build_histories()

    print("=" * 90)
    print("FEDAVG STRESS-TEST EVALUATION")
    print("=" * 90)

    print(
        "Loaded:",
        ", ".join(histories.keys()),
    )

    summary = build_summary(
        histories
    )

    round_history = pd.concat(
        histories.values(),
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT / "fedavg_stress_summary.csv",
        index=False,
    )

    round_history.to_csv(
        OUTPUT / "fedavg_stress_round_history.csv",
        index=False,
    )

    save_bar(
        summary,
        "final_f1",
        "F1 score",
        "FedAvg Final F1 Across Stress Tests",
        "final_f1_comparison.png",
    )

    save_bar(
        summary,
        "final_roc_auc",
        "ROC-AUC",
        "FedAvg Final ROC-AUC Across Stress Tests",
        "final_auc_comparison.png",
    )

    save_bar(
        summary,
        "final_accuracy",
        "Accuracy",
        "FedAvg Final Accuracy Across Stress Tests",
        "final_accuracy_comparison.png",
    )

    save_bar(
        summary,
        "total_communication_mib",
        "Communication (MiB)",
        "FedAvg Communication Across Stress Tests",
        "communication_comparison.png",
    )

    save_dropout_curve(
        summary
    )

    save_reliability_comparison(
        summary
    )

    write_report(
        summary
    )

    print()
    print(
        summary[
            [
                "experiment",
                "final_accuracy",
                "final_recall",
                "final_f1",
                "final_roc_auc",
                "final_test_loss",
                "total_communication_mib",
            ]
        ].to_string(index=False)
    )

    print()
    print("=" * 90)
    print(
        "Evaluation complete:",
        OUTPUT.resolve(),
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
