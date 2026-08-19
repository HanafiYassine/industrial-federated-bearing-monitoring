"""
run_fedavg.py

Experiment runner for the first federated-learning baseline.

Runs:
    - IID FedAvg
    - Non-IID FedAvg

Default configuration:
    10 clients
    15 communication rounds
    1 local epoch
    batch size 512
    Adam lr 0.001
    seed 42

The same held-out global test set is used for every partition.

Important:
    A StandardScaler is fitted ONLY on the global training pool and
    then shared with all clients and the global test set. This avoids
    feature-space mismatch between clients while preventing test leakage.

Usage from src/:

    python run_fedavg.py --partition iid
    python run_fedavg.py --partition noniid

Optional:
    python run_fedavg.py --partition iid --rounds 15 --local-epochs 1 --seed 42
"""

from __future__ import annotations

import argparse
import random
import time
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from federated_model import build_model
from fedavg import train_local_model, fedavg, get_feature_columns


def parameter_count(state_dict) -> int:
    """Count floating-point parameters in a model state dict."""
    return sum(
        tensor.numel()
        for tensor in state_dict.values()
        if torch.is_floating_point(tensor)
    )


def model_size_bytes(
    state_dict,
    bytes_per_parameter: int = 4,
) -> int:
    """Estimate model size assuming FP32 parameters."""
    return (
        parameter_count(state_dict)
        * bytes_per_parameter
    )



# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

BASE_CLIENT_DIR = Path(
    "../results/federated/dataset2_clients"
)

OUTPUT_ROOT = Path(
    "../results/federated/dataset2_fedavg"
)

N_CLIENTS = 10
ROUNDS = 15
LOCAL_EPOCHS = 1

BATCH_SIZE = 512
LEARNING_RATE = 0.001

SEED = 42

PARTITIONS = {
    "iid": BASE_CLIENT_DIR / "iid",
    "noniid": BASE_CLIENT_DIR / "noniid",
    "balanced_noniid": BASE_CLIENT_DIR / "balanced_noniid",
}

GLOBAL_TEST_FILE = (
    BASE_CLIENT_DIR / "global_test.csv"
)

TRAIN_POOL_FILE = (
    BASE_CLIENT_DIR / "train_pool.csv"
)

# Model has no BatchNorm, so this is primarily a
# reproducibility/reporting configuration.
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Keep results as deterministic as practical.
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run centralized-test FedAvg experiment."
    )

    parser.add_argument(
        "--partition",
        choices=["iid", "noniid", "balanced_noniid"],
        required=True,
        help="Client partition to use.",
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
        "--seed",
        type=int,
        default=SEED,
        help="Global random seed.",
    )

    return parser.parse_args()


# ============================================================
# LOAD CLIENTS
# ============================================================

def load_clients(
    partition: str,
) -> dict[int, pd.DataFrame]:

    directory = PARTITIONS[partition]

    if not directory.exists():
        raise FileNotFoundError(
            f"Client directory not found:\n"
            f"{directory.resolve()}"
        )

    clients = {}

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        path = (
            directory
            / f"client_{client_id:02d}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing client file:\n"
                f"{path.resolve()}"
            )

        df = pd.read_csv(path)

        if "label" not in df.columns:
            raise ValueError(
                f"{path} has no 'label' column."
            )

        clients[client_id] = df

    return clients


# ============================================================
# FEATURE SCALING
# ============================================================

def fit_global_scaler(
    train_pool: pd.DataFrame,
):
    feature_columns = get_feature_columns(
        train_pool
    )

    scaler = StandardScaler()

    scaler.fit(
        train_pool[
            feature_columns
        ]
    )

    return scaler, feature_columns


def apply_scaler(
    df: pd.DataFrame,
    scaler,
    feature_columns,
) -> pd.DataFrame:

    result = df.copy()

    result.loc[
        :,
        feature_columns
    ] = scaler.transform(
        df[feature_columns]
    )

    return result


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_global_model(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    feature_columns,
):
    model.eval()

    X = test_df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y = test_df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():

        logits = model(
            X_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()

    y_pred = (
        predictions
        .cpu()
        .numpy()
    )

    y_prob = (
        probabilities
        .cpu()
        .numpy()
    )

    accuracy = accuracy_score(
        y,
        y_pred,
    )

    precision = precision_score(
        y,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0,
    )

    if len(
        np.unique(y)
    ) == 2:

        auc = roc_auc_score(
            y,
            y_prob,
        )

    else:

        auc = np.nan

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
    }


# ============================================================
# GLOBAL TEST LOSS
# ============================================================

def evaluate_global_loss(
    model: torch.nn.Module,
    test_df: pd.DataFrame,
    feature_columns,
):
    model.eval()

    X = torch.tensor(
        test_df[
            feature_columns
        ].to_numpy(
            dtype=np.float32
        ),
        dtype=torch.float32,
        device=DEVICE,
    )

    y = torch.tensor(
        test_df[
            "label"
        ].to_numpy(
            dtype=np.int64
        ),
        dtype=torch.long,
        device=DEVICE,
    )

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        logits = model(X)
        loss = criterion(
            logits,
            y,
        )

    return float(
        loss.item()
    )


# ============================================================
# ROUND LOGGING
# ============================================================

def round_communication_bytes(
    model_state,
    n_clients: int,
):
    """
    Analytical communication estimate.

    Each round:
        client uploads 1 model
        server broadcasts 1 model to each client

    Therefore:
        upload   = N_clients * model_size
        download = N_clients * model_size
        total    = 2 * N_clients * model_size

    This excludes framework/protocol overhead.
    """

    model_bytes = model_size_bytes(
        model_state
    )

    upload = (
        n_clients
        * model_bytes
    )

    download = upload

    total = (
        upload
        + download
    )

    return (
        model_bytes,
        upload,
        download,
        total,
    )


# ============================================================
# TRAIN ONE FEDERATED EXPERIMENT
# ============================================================

def run_experiment(
    partition: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):
    set_seed(seed)

    print("=" * 78)
    print("FEDERATED AVERAGING EXPERIMENT")
    print("=" * 78)

    print(
        f"Partition       : {partition}"
    )

    print(
        f"Clients         : {N_CLIENTS}"
    )

    print(
        f"Rounds          : {rounds}"
    )

    print(
        f"Local epochs    : {local_epochs}"
    )

    print(
        f"Batch size      : {batch_size}"
    )

    print(
        f"Learning rate   : {learning_rate}"
    )

    print(
        f"Seed            : {seed}"
    )

    print(
        f"Device          : {DEVICE}"
    )

    # --------------------------------------------------------
    # Load global training pool and test set.
    # --------------------------------------------------------

    if not TRAIN_POOL_FILE.exists():
        raise FileNotFoundError(
            f"Missing train_pool.csv:\n"
            f"{TRAIN_POOL_FILE.resolve()}"
        )

    if not GLOBAL_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing global_test.csv:\n"
            f"{GLOBAL_TEST_FILE.resolve()}"
        )

    train_pool = pd.read_csv(
        TRAIN_POOL_FILE
    )

    global_test = pd.read_csv(
        GLOBAL_TEST_FILE
    )

    # --------------------------------------------------------
    # Load clients.
    # --------------------------------------------------------

    clients = load_clients(
        partition
    )

    # Verify total client samples.
    total_client_samples = sum(
        len(df)
        for df in clients.values()
    )

    if total_client_samples != len(
        train_pool
    ):
        raise ValueError(
            "Client samples do not match train pool size:\n"
            f"Clients: {total_client_samples}\n"
            f"Train pool: {len(train_pool)}"
        )

    # --------------------------------------------------------
    # Shared feature normalization.
    # --------------------------------------------------------

    scaler, feature_columns = (
        fit_global_scaler(
            train_pool
        )
    )

    scaled_clients = {}

    for client_id, df in clients.items():

        scaled_clients[
            client_id
        ] = apply_scaler(
            df,
            scaler,
            feature_columns,
        )

    scaled_test = apply_scaler(
        global_test,
        scaler,
        feature_columns,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    input_size = len(
        feature_columns
    )

    model = build_model(
        input_size=input_size
    ).to(DEVICE)

    global_state = OrderedDict(
        (
            name,
            tensor.detach().cpu().clone()
        )
        for name, tensor
        in model.state_dict().items()
    )

    print()
    print(
        "Input features  :",
        input_size
    )

    print(
        "Model parameters :",
        parameter_count(
            global_state
        )
    )

    print(
        "Model size       :",
        model_size_bytes(
            global_state
        ),
        "bytes"
    )

    # --------------------------------------------------------
    # Metrics collected after every round.
    # --------------------------------------------------------

    history = []

    # --------------------------------------------------------
    # Federated rounds
    # --------------------------------------------------------

    for round_number in range(
        1,
        rounds + 1,
    ):

        print()
        print("-" * 78)
        print(
            f"ROUND {round_number}/{rounds}"
        )
        print("-" * 78)

        round_start = time.perf_counter()

        local_states = []
        local_counts = []
        local_losses = []

        # ----------------------------------------------------
        # Local client training.
        # ----------------------------------------------------

        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):

            client_df = scaled_clients[
                client_id
            ]

            # Deterministic but distinct client seed.
            client_seed = (
                seed
                + round_number * 1000
                + client_id
            )

            set_seed(
                client_seed
            )

            local_start = (
                time.perf_counter()
            )

            (
                local_state,
                local_loss,
                sample_count,
            ) = train_local_model(
                global_state=global_state,
                client_df=client_df,
                device=DEVICE,
                input_size=input_size,
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )

            local_time = (
                time.perf_counter()
                - local_start
            )

            local_states.append(
                local_state
            )

            local_counts.append(
                sample_count
            )

            local_losses.append(
                local_loss
            )

            print(
                f"Client {client_id:02d}: "
                f"samples={sample_count:3d} "
                f"loss={local_loss:.6f} "
                f"time={local_time:.3f}s"
            )

        # ----------------------------------------------------
        # FedAvg
        # ----------------------------------------------------

        aggregation_start = (
            time.perf_counter()
        )

        global_state = fedavg(
            local_states,
            local_counts,
        )

        aggregation_time = (
            time.perf_counter()
            - aggregation_start
        )

        model.load_state_dict(
            global_state
        )

        # ----------------------------------------------------
        # Global evaluation
        # ----------------------------------------------------

        global_loss = (
            evaluate_global_loss(
                model,
                scaled_test,
                feature_columns,
            )
        )

        metrics = (
            evaluate_global_model(
                model,
                scaled_test,
                feature_columns,
            )
        )

        # ----------------------------------------------------
        # Communication
        # ----------------------------------------------------

        (
            model_bytes,
            upload_bytes,
            download_bytes,
            total_bytes,
        ) = round_communication_bytes(
            global_state,
            N_CLIENTS,
        )

        round_time = (
            time.perf_counter()
            - round_start
        )

        mean_local_loss = float(
            np.mean(
                local_losses
            )
        )

        row = {
            "partition": partition,
            "round": round_number,
            "clients": N_CLIENTS,
            "local_epochs": local_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "mean_local_loss": mean_local_loss,
            "global_test_loss": global_loss,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "roc_auc": metrics["roc_auc"],
            "model_parameters": parameter_count(
                global_state
            ),
            "model_size_bytes": model_bytes,
            "upload_bytes": upload_bytes,
            "download_bytes": download_bytes,
            "total_communication_bytes": total_bytes,
            "aggregation_time_seconds": aggregation_time,
            "round_time_seconds": round_time,
        }

        history.append(
            row
        )

        print()
        print(
            f"Global test loss : "
            f"{global_loss:.6f}"
        )

        print(
            f"Accuracy         : "
            f"{metrics['accuracy']:.6f}"
        )

        print(
            f"Precision        : "
            f"{metrics['precision']:.6f}"
        )

        print(
            f"Recall           : "
            f"{metrics['recall']:.6f}"
        )

        print(
            f"F1               : "
            f"{metrics['f1']:.6f}"
        )

        print(
            f"ROC-AUC          : "
            f"{metrics['roc_auc']:.6f}"
        )

        print(
            f"Round time       : "
            f"{round_time:.3f}s"
        )

        print(
            f"Communication    : "
            f"{total_bytes / 1024:.2f} KB"
        )

    return pd.DataFrame(
        history
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    history: pd.DataFrame,
    partition: str,
    args,
):
    output_dir = (
        OUTPUT_ROOT
        / partition
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Round history
    # --------------------------------------------------------

    history.to_csv(
        output_dir
        / "fedavg_round_history.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    final = history.iloc[-1].copy()

    final_metrics = pd.DataFrame(
        {
            "metric": [
                "partition",
                "rounds",
                "clients",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "seed",
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
                args.rounds,
                N_CLIENTS,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                args.seed,
                str(DEVICE),
                final["accuracy"],
                final["precision"],
                final["recall"],
                final["f1"],
                final["roc_auc"],
                final["global_test_loss"],
                history[
                    "total_communication_bytes"
                ].sum(),
                history[
                    "round_time_seconds"
                ].sum(),
            ],
        }
    )

    final_metrics.to_csv(
        output_dir
        / "fedavg_final_metrics.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = pd.DataFrame(
        {
            "parameter": [
                "partition",
                "clients",
                "rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "seed",
                "device",
                "model",
                "input_features",
                "hidden_layers",
                "dropout",
            ],

            "value": [
                partition,
                N_CLIENTS,
                args.rounds,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                args.seed,
                str(DEVICE),
                "FederatedMLP",
                36,
                "128-64-32",
                0.40,
            ],
        }
    )

    config.to_csv(
        output_dir
        / "fedavg_config.csv",
        index=False,
    )

    return output_dir


# ============================================================
# PLOT TRAINING CURVES
# ============================================================

def create_figures(
    history: pd.DataFrame,
    partition: str,
    output_dir: Path,
):
    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        history["round"],
        history["global_test_loss"],
        marker="o",
        label="Global test loss",
    )

    ax.plot(
        history["round"],
        history["mean_local_loss"],
        marker="o",
        label="Mean local training loss",
    )

    ax.set_xlabel(
        "Federated round"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        f"FedAvg Loss — {partition.upper()}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "fedavg_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for column, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]:

        ax.plot(
            history["round"],
            history[column],
            marker="o",
            label=label,
        )

    ax.set_xlabel(
        "Federated round"
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_ylim(
        0.0,
        1.05,
    )

    ax.set_title(
        f"FedAvg Evaluation Metrics — {partition.upper()}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "fedavg_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    cumulative = (
        history[
            "total_communication_bytes"
        ].cumsum()
        / (1024 * 1024)
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        history["round"],
        cumulative,
        marker="o",
    )

    ax.set_xlabel(
        "Federated round"
    )

    ax.set_ylabel(
        "Cumulative communication (MiB)"
    )

    ax.set_title(
        f"FedAvg Communication Cost — {partition.upper()}"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "fedavg_communication.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be > 0."
        )

    if args.local_epochs <= 0:
        raise ValueError(
            "--local-epochs must be > 0."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be > 0."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be > 0."
        )

    set_seed(
        args.seed
    )

    history = run_experiment(
        partition=args.partition,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    output_dir = save_results(
        history,
        args.partition,
        args,
    )

    create_figures(
        history,
        args.partition,
        output_dir,
    )

    print()
    print("=" * 78)
    print("FEDAVG EXPERIMENT COMPLETE")
    print("=" * 78)

    final = history.iloc[-1]

    print(
        f"Partition      : {args.partition}"
    )

    print(
        f"Final accuracy  : "
        f"{final['accuracy']:.6f}"
    )

    print(
        f"Final precision : "
        f"{final['precision']:.6f}"
    )

    print(
        f"Final recall    : "
        f"{final['recall']:.6f}"
    )

    print(
        f"Final F1        : "
        f"{final['f1']:.6f}"
    )

    print(
        f"Final ROC-AUC   : "
        f"{final['roc_auc']:.6f}"
    )

    print(
        f"Total communication: "
        f"{history['total_communication_bytes'].sum() / (1024 * 1024):.4f} MiB"
    )

    print(
        f"Total round time: "
        f"{history['round_time_seconds'].sum():.3f}s"
    )

    print(
        "Results:",
        output_dir.resolve(),
    )

    print("=" * 78)


if __name__ == "__main__":
    main()