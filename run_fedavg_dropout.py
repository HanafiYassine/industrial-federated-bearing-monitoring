"""
run_fedavg_dropout.py

Run the standard FedAvg baseline under deterministic client dropout.

Purpose:
    Measure how ordinary FedAvg degrades when industrial clients become
    unavailable during training.

This is a BASELINE stress test. It does not add:
    - client selection
    - adaptive aggregation
    - reliability weighting
    - checkpointing
    - optimization

Those mechanisms will be added later.

Default:
    Dataset 2
    unbalanced non-IID clients
    10 clients
    15 rounds
    local epochs = 1
    batch size = 512
    Adam lr = 0.001
    seed = 42

Examples:

    python run_fedavg_dropout.py --rate 0.0
    python run_fedavg_dropout.py --rate 0.2
    python run_fedavg_dropout.py --rate 0.4
    python run_fedavg_dropout.py --rate 0.5

The participation schedule is read from:
    ../results/federated/dataset2_dropout/dropout_Xpct/client_participation.csv
"""

from __future__ import annotations

import argparse
import random
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from federated_model import build_model
from fedavg import (
    train_local_model,
    fedavg,
    get_feature_columns,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_CLIENT_DIR = Path(
    "../results/federated/dataset2_clients/noniid"
)

DROPOUT_ROOT = Path(
    "../results/federated/dataset2_dropout"
)

GLOBAL_TEST_FILE = Path(
    "../results/federated/dataset2_clients/global_test.csv"
)

OUTPUT_ROOT = Path(
    "../results/federated/dataset2_fedavg_dropout"
)

N_CLIENTS = 10

ROUNDS = 15

LOCAL_EPOCHS = 1

BATCH_SIZE = 512

LEARNING_RATE = 0.001

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def parameter_count(state_dict) -> int:
    return sum(
        tensor.numel()
        for tensor in state_dict.values()
        if torch.is_floating_point(tensor)
    )


def model_size_bytes(
    state_dict,
    bytes_per_parameter: int = 4,
) -> int:
    return parameter_count(
        state_dict
    ) * bytes_per_parameter


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run FedAvg with client dropout."
    )

    parser.add_argument(
        "--rate",
        type=float,
        required=True,
        help="Dropout rate: 0, 0.2, 0.4 or 0.5.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=LOCAL_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=LEARNING_RATE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    return parser.parse_args()


# ============================================================
# LOAD DATA
# ============================================================

def load_clients():

    clients = {}

    for client_id in range(
        1,
        N_CLIENTS + 1,
    ):

        path = (
            BASE_CLIENT_DIR
            / f"client_{client_id:02d}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing client file:\n"
                f"{path.resolve()}"
            )

        clients[client_id] = pd.read_csv(
            path
        )

    return clients


def load_global_test():

    if not GLOBAL_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing global test file:\n"
            f"{GLOBAL_TEST_FILE.resolve()}"
        )

    return pd.read_csv(
        GLOBAL_TEST_FILE
    )


def load_participation_schedule(
    rate: float,
    rounds: int,
):

    pct = int(
        round(rate * 100)
    )

    path = (
        DROPOUT_ROOT
        / f"dropout_{pct}pct"
        / "client_participation.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Participation schedule not found:\n"
            f"{path.resolve()}\n\n"
            "Run simulate_client_dropout.py first."
        )

    schedule = pd.read_csv(
        path
    )

    required = {
        "round",
        "client_id",
        "participates",
    }

    missing = required - set(
        schedule.columns
    )

    if missing:
        raise ValueError(
            f"Schedule is missing columns: {sorted(missing)}"
        )

    schedule["round"] = schedule[
        "round"
    ].astype(int)

    schedule["client_id"] = schedule[
        "client_id"
    ].astype(int)

    schedule["participates"] = schedule[
        "participates"
    ].astype(bool)

    schedule = schedule[
        schedule["round"] <= rounds
    ]

    return schedule


# ============================================================
# SCALING
# ============================================================

def prepare_data(
    clients,
    global_test,
):

    train_pool = pd.concat(
        list(clients.values()),
        ignore_index=True,
    )

    feature_columns = get_feature_columns(
        train_pool
    )

    scaler = StandardScaler()

    scaler.fit(
        train_pool[
            feature_columns
        ]
    )

    scaled_clients = {}

    for client_id, df in clients.items():

        scaled = df.copy()

        scaled[
            feature_columns
        ] = scaler.transform(
            df[
                feature_columns
            ]
        )

        scaled_clients[
            client_id
        ] = scaled

    scaled_test = global_test.copy()

    scaled_test[
        feature_columns
    ] = scaler.transform(
        global_test[
            feature_columns
        ]
    )

    return (
        scaled_clients,
        scaled_test,
        feature_columns,
    )


# ============================================================
# GLOBAL EVALUATION
# ============================================================

def evaluate_model(
    model,
    test_df,
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

    y = test_df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
        device=DEVICE,
    )

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():

        logits = model(X)

        loss = criterion(
            logits,
            y_tensor,
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

    return {
        "global_test_loss": float(
            loss.item()
        ),
        "accuracy": float(
            accuracy_score(
                y,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y,
                y_prob,
            )
        ),
    }


# ============================================================
# ROUND
# ============================================================

def participating_clients(
    schedule,
    round_number,
):

    rows = schedule[
        schedule["round"] == round_number
    ]

    clients = rows[
        rows["participates"]
    ]["client_id"].tolist()

    return [
        int(client_id)
        for client_id in clients
    ]


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_experiment(
    rate: float,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
):

    set_seed(seed)

    clients = load_clients()

    global_test = load_global_test()

    schedule = load_participation_schedule(
        rate,
        rounds,
    )

    (
        scaled_clients,
        scaled_test,
        feature_columns,
    ) = prepare_data(
        clients,
        global_test,
    )

    model = build_model(
        input_size=len(
            feature_columns
        )
    ).to(DEVICE)

    global_state = OrderedDict(
        (
            name,
            tensor.detach().cpu().clone()
        )
        for name, tensor
        in model.state_dict().items()
    )

    model_bytes = model_size_bytes(
        global_state
    )

    history = []

    print("=" * 78)
    print("FEDAVG CLIENT-DROPOUT EXPERIMENT")
    print("=" * 78)

    print(
        f"Dropout rate    : {rate:.2f}"
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
        f"Device          : {DEVICE}"
    )

    print(
        f"Model size      : "
        f"{model_bytes} bytes"
    )

    for round_number in range(
        1,
        rounds + 1,
    ):

        print()
        print("-" * 78)
        print(
            f"ROUND {round_number}/{rounds}"
        )

        selected = participating_clients(
            schedule,
            round_number,
        )

        if len(selected) == 0:
            raise RuntimeError(
                f"No participating clients in round {round_number}."
            )

        print(
            "Participating clients:",
            selected,
        )

        round_start = time.perf_counter()

        local_states = []
        local_counts = []
        local_losses = []

        for client_id in selected:

            client_seed = (
                seed
                + round_number * 1000
                + client_id
            )

            set_seed(
                client_seed
            )

            client = scaled_clients[
                client_id
            ]

            (
                local_state,
                local_loss,
                sample_count,
            ) = train_local_model(
                global_state=global_state,
                client_df=client,
                device=DEVICE,
                input_size=len(
                    feature_columns
                ),
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
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
                f"loss={local_loss:.6f}"
            )

        aggregation_start = time.perf_counter()

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

        metrics = evaluate_model(
            model,
            scaled_test,
            feature_columns,
        )

        round_time = (
            time.perf_counter()
            - round_start
        )

        participating = len(
            selected
        )

        # Only participating clients transmit
        # this round.
        upload_bytes = (
            participating
            * model_bytes
        )

        download_bytes = (
            participating
            * model_bytes
        )

        total_communication = (
            upload_bytes
            + download_bytes
        )

        history.append(
            {
                "dropout_rate": rate,
                "round": round_number,
                "participating_clients": participating,
                "dropped_clients": (
                    N_CLIENTS - participating
                ),
                "mean_local_loss": float(
                    np.mean(local_losses)
                ),
                "global_test_loss": metrics[
                    "global_test_loss"
                ],
                "accuracy": metrics[
                    "accuracy"
                ],
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "roc_auc": metrics[
                    "roc_auc"
                ],
                "model_parameters": parameter_count(
                    global_state
                ),
                "model_size_bytes": model_bytes,
                "upload_bytes": upload_bytes,
                "download_bytes": download_bytes,
                "total_communication_bytes": (
                    total_communication
                ),
                "aggregation_time_seconds": (
                    aggregation_time
                ),
                "round_time_seconds": (
                    round_time
                ),
            }
        )

        print(
            f"Global test loss : "
            f"{metrics['global_test_loss']:.6f}"
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
            f"Communication    : "
            f"{total_communication / (1024 ** 2):.4f} MiB"
        )

    return pd.DataFrame(
        history
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    history,
    rate,
    rounds,
    local_epochs,
    batch_size,
    learning_rate,
    seed,
):

    percentage = int(
        round(rate * 100)
    )

    output_dir = (
        OUTPUT_ROOT
        / f"dropout_{percentage}pct"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        output_dir
        / "fedavg_dropout_round_history.csv",
        index=False,
    )

    final = history.iloc[-1]

    final_metrics = pd.DataFrame(
        {
            "metric": [
                "dropout_rate",
                "dropout_percentage",
                "rounds",
                "clients",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "seed",
                "final_accuracy",
                "final_precision",
                "final_recall",
                "final_f1",
                "final_roc_auc",
                "final_test_loss",
                "total_communication_bytes",
                "total_communication_mib",
                "total_round_time_seconds",
            ],
            "value": [
                rate,
                percentage,
                rounds,
                N_CLIENTS,
                local_epochs,
                batch_size,
                learning_rate,
                seed,
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
                    "total_communication_bytes"
                ].sum()
                / (1024 ** 2),
                history[
                    "round_time_seconds"
                ].sum(),
            ],
        }
    )

    final_metrics.to_csv(
        output_dir
        / "fedavg_dropout_final_metrics.csv",
        index=False,
    )

    return output_dir


# ============================================================
# PLOTS
# ============================================================

def create_figures(
    history,
    output_dir,
    rate,
):

    pct = int(
        round(rate * 100)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for metric, label in [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
    ]:

        ax.plot(
            history["round"],
            history[metric],
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
        0,
        1.05,
    )

    ax.set_title(
        f"FedAvg under {pct}% Client Dropout"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "fedavg_dropout_metrics.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

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
        f"FedAvg Loss under {pct}% Client Dropout"
    )

    ax.grid(
        True,
        alpha=0.25,
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "fedavg_dropout_loss.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    if not 0.0 <= args.rate < 1.0:
        raise ValueError(
            "--rate must satisfy 0 <= rate < 1."
        )

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be > 0."
        )

    history = run_experiment(
        rate=args.rate,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    output_dir = save_results(
        history=history,
        rate=args.rate,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    create_figures(
        history,
        output_dir,
        args.rate,
    )

    final = history.iloc[-1]

    print()
    print("=" * 78)
    print("FEDAVG DROPOUT EXPERIMENT COMPLETE")
    print("=" * 78)

    print(
        f"Dropout          : "
        f"{args.rate * 100:.0f}%"
    )

    print(
        f"Final accuracy   : "
        f"{final['accuracy']:.6f}"
    )

    print(
        f"Final precision  : "
        f"{final['precision']:.6f}"
    )

    print(
        f"Final recall     : "
        f"{final['recall']:.6f}"
    )

    print(
        f"Final F1         : "
        f"{final['f1']:.6f}"
    )

    print(
        f"Final ROC-AUC    : "
        f"{final['roc_auc']:.6f}"
    )

    print(
        f"Final test loss  : "
        f"{final['global_test_loss']:.6f}"
    )

    print(
        f"Total communication: "
        f"{history['total_communication_bytes'].sum() / (1024 ** 2):.4f} MiB"
    )

    print(
        "Output:",
        output_dir.resolve(),
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
