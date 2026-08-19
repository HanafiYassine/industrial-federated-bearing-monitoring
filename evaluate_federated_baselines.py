"""
evaluate_federated_baselines.py

Compare the three Dataset-2 FedAvg experiments:

    1. IID
    2. Balanced Non-IID
    3. Unbalanced Non-IID

Expected input structure:
    ../results/federated/dataset2_fedavg/
        iid/
            fedavg_round_history.csv
            fedavg_final_metrics.csv
            fedavg_config.csv
        balanced_noniid/
            ...
        noniid/
            ...

Outputs:
    ../results/federated/dataset2_fedavg/evaluation/
        fedavg_baseline_comparison.csv
        fedavg_round_comparison.csv
        fedavg_best_rounds.csv
        fedavg_communication_summary.csv
        auc_vs_round.png
        f1_vs_round.png
        accuracy_vs_round.png
        test_loss_vs_round.png
        communication_vs_round.png
        final_metrics_comparison.png
        baseline_comparison_report.txt

The script uses the same global test set by construction because all
three FedAvg experiments were configured with the same test file and
evaluation protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(
    "../results/federated/dataset2_fedavg"
)

OUTPUT = (
    ROOT / "evaluation"
)

PARTITIONS = [
    "iid",
    "balanced_noniid",
    "noniid",
]

DISPLAY_NAMES = {
    "iid": "IID",
    "balanced_noniid": "Balanced Non-IID",
    "noniid": "Unbalanced Non-IID",
}


# ============================================================
# LOAD
# ============================================================

def load_history(partition: str) -> pd.DataFrame:

    path = (
        ROOT
        / partition
        / "fedavg_round_history.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing round history:\n{path.resolve()}"
        )

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
            f"{path} is missing columns: {sorted(missing)}"
        )

    df = df.copy()

    df["partition"] = partition
    df["method"] = DISPLAY_NAMES[partition]

    return df.sort_values(
        "round"
    ).reset_index(drop=True)


# ============================================================
# FINAL METRICS
# ============================================================

def build_final_comparison(
    histories,
) -> pd.DataFrame:

    rows = []

    for partition, df in histories.items():

        final = df.iloc[-1]

        best_auc = df.loc[
            df["roc_auc"].idxmax()
        ]

        best_f1 = df.loc[
            df["f1"].idxmax()
        ]

        best_accuracy = df.loc[
            df["accuracy"].idxmax()
        ]

        rows.append(
            {
                "partition": partition,
                "method": DISPLAY_NAMES[partition],

                "rounds": int(
                    df["round"].max()
                ),

                "clients": int(
                    final["clients"]
                ),

                "local_epochs": int(
                    final["local_epochs"]
                ),

                "batch_size": int(
                    final["batch_size"]
                ),

                "learning_rate": float(
                    final["learning_rate"]
                ),

                "final_accuracy": float(
                    final["accuracy"]
                ),

                "final_precision": float(
                    final["precision"]
                ),

                "final_recall": float(
                    final["recall"]
                ),

                "final_f1": float(
                    final["f1"]
                ),

                "final_roc_auc": float(
                    final["roc_auc"]
                ),

                "final_test_loss": float(
                    final["global_test_loss"]
                ),

                "best_accuracy": float(
                    best_accuracy["accuracy"]
                ),

                "best_accuracy_round": int(
                    best_accuracy["round"]
                ),

                "best_f1": float(
                    best_f1["f1"]
                ),

                "best_f1_round": int(
                    best_f1["round"]
                ),

                "best_roc_auc": float(
                    best_auc["roc_auc"]
                ),

                "best_roc_auc_round": int(
                    best_auc["round"]
                ),

                "total_communication_bytes": float(
                    df[
                        "total_communication_bytes"
                    ].sum()
                ),

                "total_communication_mib": float(
                    df[
                        "total_communication_bytes"
                    ].sum()
                    / (1024 ** 2)
                ),

                "total_round_time_seconds": float(
                    df[
                        "round_time_seconds"
                    ].sum()
                ),

                "mean_round_time_seconds": float(
                    df[
                        "round_time_seconds"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# ROUND COMPARISON
# ============================================================

def build_round_comparison(
    histories,
) -> pd.DataFrame:

    return pd.concat(
        histories.values(),
        ignore_index=True,
    )


# ============================================================
# BEST ROUNDS
# ============================================================

def build_best_rounds(
    histories,
) -> pd.DataFrame:

    rows = []

    for partition, df in histories.items():

        for metric in [
            "accuracy",
            "f1",
            "roc_auc",
        ]:

            idx = df[
                metric
            ].idxmax()

            row = df.loc[idx]

            rows.append(
                {
                    "partition": partition,
                    "method": DISPLAY_NAMES[partition],
                    "metric": metric,
                    "best_round": int(
                        row["round"]
                    ),
                    "best_value": float(
                        row[metric]
                    ),
                    "accuracy_at_round": float(
                        row["accuracy"]
                    ),
                    "f1_at_round": float(
                        row["f1"]
                    ),
                    "roc_auc_at_round": float(
                        row["roc_auc"]
                    ),
                    "test_loss_at_round": float(
                        row["global_test_loss"]
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# COMMUNICATION SUMMARY
# ============================================================

def build_communication_summary(
    final_comparison,
) -> pd.DataFrame:

    df = final_comparison[
        [
            "partition",
            "method",
            "rounds",
            "clients",
            "total_communication_bytes",
            "total_communication_mib",
            "total_round_time_seconds",
            "mean_round_time_seconds",
        ]
    ].copy()

    df["communication_per_round_mib"] = (
        df["total_communication_mib"]
        / df["rounds"]
    )

    return df


# ============================================================
# PLOTS
# ============================================================

def plot_metric(
    histories,
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    ylim=None,
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for partition in PARTITIONS:

        if partition not in histories:
            continue

        df = histories[partition]

        ax.plot(
            df["round"],
            df[metric],
            marker="o",
            linewidth=1.5,
            label=DISPLAY_NAMES[partition],
        )

    ax.set_xlabel(
        "Federated round"
    )

    ax.set_ylabel(
        ylabel
    )

    ax.set_title(
        title
    )

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_communication(
    histories,
):

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    for partition in PARTITIONS:

        if partition not in histories:
            continue

        df = histories[partition]

        cumulative = (
            df[
                "total_communication_bytes"
            ].cumsum()
            / (1024 ** 2)
        )

        ax.plot(
            df["round"],
            cumulative,
            marker="o",
            linewidth=1.5,
            label=DISPLAY_NAMES[partition],
        )

    ax.set_xlabel(
        "Federated round"
    )

    ax.set_ylabel(
        "Cumulative communication (MiB)"
    )

    ax.set_title(
        "FedAvg Cumulative Communication Cost"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        OUTPUT / "communication_vs_round.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_final_metrics(
    final_comparison,
):

    metrics = [
        (
            "final_accuracy",
            "Accuracy",
        ),
        (
            "final_precision",
            "Precision",
        ),
        (
            "final_recall",
            "Recall",
        ),
        (
            "final_f1",
            "F1",
        ),
        (
            "final_roc_auc",
            "ROC-AUC",
        ),
    ]

    methods = final_comparison[
        "method"
    ].tolist()

    x = np.arange(
        len(methods)
    )

    width = 0.15

    fig, ax = plt.subplots(
        figsize=(13, 6)
    )

    for i, (
        column,
        label,
    ) in enumerate(metrics):

        values = (
            final_comparison[column]
            .to_numpy()
        )

        ax.bar(
            x
            + (i - 2)
            * width,

            values,

            width,

            label=label,
        )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        methods
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0,
        1.05,
    )

    ax.set_title(
        "Final FedAvg Metrics Comparison"
    )

    ax.grid(
        True,
        axis="y",
        alpha=0.25,
    )

    ax.legend(
        ncol=3
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT
        / "final_metrics_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# REPORT
# ============================================================

def write_report(
    final_comparison,
):

    lines = []

    lines.append(
        "=" * 78
    )

    lines.append(
        "FEDERATED BASELINE COMPARISON"
    )

    lines.append(
        "=" * 78
    )

    lines.append("")

    for _, row in final_comparison.iterrows():

        lines.append(
            row["method"]
        )

        lines.append(
            "-" * 78
        )

        lines.append(
            f"Final accuracy      : "
            f"{row['final_accuracy']:.6f}"
        )

        lines.append(
            f"Final precision     : "
            f"{row['final_precision']:.6f}"
        )

        lines.append(
            f"Final recall        : "
            f"{row['final_recall']:.6f}"
        )

        lines.append(
            f"Final F1            : "
            f"{row['final_f1']:.6f}"
        )

        lines.append(
            f"Final ROC-AUC       : "
            f"{row['final_roc_auc']:.6f}"
        )

        lines.append(
            f"Final test loss     : "
            f"{row['final_test_loss']:.6f}"
        )

        lines.append(
            f"Best accuracy       : "
            f"{row['best_accuracy']:.6f} "
            f"(round {int(row['best_accuracy_round'])})"
        )

        lines.append(
            f"Best F1             : "
            f"{row['best_f1']:.6f} "
            f"(round {int(row['best_f1_round'])})"
        )

        lines.append(
            f"Best ROC-AUC        : "
            f"{row['best_roc_auc']:.6f} "
            f"(round {int(row['best_roc_auc_round'])})"
        )

        lines.append(
            f"Communication       : "
            f"{row['total_communication_mib']:.4f} MiB"
        )

        lines.append(
            f"Total round time    : "
            f"{row['total_round_time_seconds']:.4f} s"
        )

        lines.append("")

    # --------------------------------------------------------
    # Pairwise AUC differences
    # --------------------------------------------------------

    lookup = final_comparison.set_index(
        "partition"
    )

    if "iid" in lookup.index:

        iid_auc = lookup.loc[
            "iid",
            "final_roc_auc",
        ]

        for partition in [
            "balanced_noniid",
            "noniid",
        ]:

            if partition in lookup.index:

                auc = lookup.loc[
                    partition,
                    "final_roc_auc",
                ]

                delta = auc - iid_auc

                lines.append(
                    f"AUC difference "
                    f"{DISPLAY_NAMES[partition]} "
                    f"vs IID: {delta:+.6f}"
                )

        lines.append("")

    lines.append(
        "Interpretation note:"
    )

    lines.append(
        "These experiments use the same controlled feature, label, "
        "model and global-test protocol. The IID/non-IID partitions "
        "are our experimental design, not a claim that the published "
        "paper used these exact partitions."
    )

    lines.append("")

    path = (
        OUTPUT
        / "baseline_comparison_report.txt"
    )

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

    histories = {}

    for partition in PARTITIONS:

        history_path = (
            ROOT
            / partition
            / "fedavg_round_history.csv"
        )

        if not history_path.exists():

            print(
                f"WARNING: missing "
                f"{partition}: "
                f"{history_path}"
            )

            continue

        histories[
            partition
        ] = load_history(
            partition
        )

    if not histories:

        raise RuntimeError(
            "No FedAvg history files were found."
        )

    print("=" * 78)
    print("FEDERATED BASELINE EVALUATION")
    print("=" * 78)

    print()
    print(
        "Loaded partitions:",
        ", ".join(
            DISPLAY_NAMES[p]
            for p in histories
        ),
    )

    # --------------------------------------------------------
    # Tables
    # --------------------------------------------------------

    final_comparison = (
        build_final_comparison(
            histories
        )
    )

    round_comparison = (
        build_round_comparison(
            histories
        )
    )

    best_rounds = (
        build_best_rounds(
            histories
        )
    )

    communication = (
        build_communication_summary(
            final_comparison
        )
    )

    final_comparison.to_csv(
        OUTPUT
        / "fedavg_baseline_comparison.csv",
        index=False,
    )

    round_comparison.to_csv(
        OUTPUT
        / "fedavg_round_comparison.csv",
        index=False,
    )

    best_rounds.to_csv(
        OUTPUT
        / "fedavg_best_rounds.csv",
        index=False,
    )

    communication.to_csv(
        OUTPUT
        / "fedavg_communication_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Figures
    # --------------------------------------------------------

    plot_metric(
        histories,
        metric="roc_auc",
        ylabel="ROC-AUC",
        title="FedAvg ROC-AUC vs Communication Round",
        filename="auc_vs_round.png",
        ylim=(0, 1.05),
    )

    plot_metric(
        histories,
        metric="f1",
        ylabel="F1 score",
        title="FedAvg F1 vs Communication Round",
        filename="f1_vs_round.png",
        ylim=(0, 1.05),
    )

    plot_metric(
        histories,
        metric="accuracy",
        ylabel="Accuracy",
        title="FedAvg Accuracy vs Communication Round",
        filename="accuracy_vs_round.png",
        ylim=(0, 1.05),
    )

    plot_metric(
        histories,
        metric="global_test_loss",
        ylabel="Test loss",
        title="FedAvg Test Loss vs Communication Round",
        filename="test_loss_vs_round.png",
    )

    plot_communication(
        histories
    )

    plot_final_metrics(
        final_comparison
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    write_report(
        final_comparison
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print(
        final_comparison[
            [
                "method",
                "final_accuracy",
                "final_precision",
                "final_recall",
                "final_f1",
                "final_roc_auc",
                "final_test_loss",
                "total_communication_mib",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "=" * 78
    )

    print(
        "Evaluation complete."
    )

    print(
        "Output:",
        OUTPUT.resolve()
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()
