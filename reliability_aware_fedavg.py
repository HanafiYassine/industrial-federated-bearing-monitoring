"""
reliability_aware_fedavg.py

Reliability-Aware FedAvg (RA-FedAvg), first research version.

Purpose:
    Replace sample-count-only FedAvg weighting with a reliability-aware
    aggregation rule based only on observable client-side quantities.

Standard FedAvg:
    alpha_k = n_k / sum_j(n_j)

RA-FedAvg:
    alpha_k =
        n_k * R_k / sum_j(n_j * R_j)

Client reliability score:

    R_k =
        DATA_WEIGHT     * data_score
      + LOSS_WEIGHT     * loss_score
      + STABILITY_WEIGHT * stability_score
      + AVAILABILITY_WEIGHT * availability_score

where:

    data_score:
        sqrt(n_k / max_n)  [bounded in (0,1]]

    loss_score:
        exp(-normalized_local_loss)

    stability_score:
        1 / (1 + update_distance)

    availability_score:
        recent participation rate

Important:
    - No hidden simulator reliability_score is used.
    - No future test information is used.
    - The global test set is NEVER used to compute aggregation weights.
    - The method uses only information available at/after local training.
    - Client dropout is handled naturally: unavailable clients do not participate.

For this first version, the score weights are configurable. The default is:

    data        = 0.15
    loss        = 0.35
    stability   = 0.35
    availability= 0.15

This is an initial proposed method, not yet the final optimized version.
Later we can optimize these coefficients against multiple objectives.

Usage:
    python reliability_aware_fedavg.py --scenario mixed_full

Scenarios:
    clean
    sensor_noise
    stale_updates
    persistent_failure
    mixed
    mixed_full

Outputs:
    ../results/federated/dataset2_ra_fedavg/<scenario>/
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from federated_model import build_model
from fedavg import (
    get_feature_columns,
    train_local_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_DIR = Path(
    "../results/federated/dataset2_clients/noniid"
)

RELIABILITY_ROOT = Path(
    "../results/federated/dataset2_reliability"
)

GLOBAL_TEST_FILE = Path(
    "../results/federated/dataset2_clients/global_test.csv"
)

OUTPUT_ROOT = Path(
    "../results/federated/dataset2_ra_fedavg"
)

N_CLIENTS = 10

DEFAULT_ROUNDS = 15
DEFAULT_LOCAL_EPOCHS = 1
DEFAULT_BATCH_SIZE = 512
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_SEED = 42

DATA_WEIGHT = 0.15
LOSS_WEIGHT = 0.35
STABILITY_WEIGHT = 0.35
AVAILABILITY_WEIGHT = 0.15

EPS = 1e-12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run reliability-aware FedAvg."
    )

    parser.add_argument(
        "--scenario",
        choices=[
            "clean",
            "sensor_noise",
            "stale_updates",
            "persistent_failure",
            "mixed",
            "mixed_full",
        ],
        required=True,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=DEFAULT_LOCAL_EPOCHS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--data-weight",
        type=float,
        default=DATA_WEIGHT,
    )

    parser.add_argument(
        "--loss-weight",
        type=float,
        default=LOSS_WEIGHT,
    )

    parser.add_argument(
        "--stability-weight",
        type=float,
        default=STABILITY_WEIGHT,
    )

    parser.add_argument(
        "--availability-weight",
        type=float,
        default=AVAILABILITY_WEIGHT,
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_weights(args):
    weights = [
        args.data_weight,
        args.loss_weight,
        args.stability_weight,
        args.availability_weight,
    ]

    if any(w < 0 for w in weights):
        raise ValueError(
            "Reliability weights must be non-negative."
        )

    total = sum(weights)

    if total <= 0:
        raise ValueError(
            "At least one reliability weight must be positive."
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

    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


# ============================================================
# STATE / PARAMETER UTILITIES
# ============================================================

def copy_state(state):
    return OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in state.items()
    )


def parameter_count(state):
    return sum(
        tensor.numel()
        for tensor in state.values()
        if torch.is_floating_point(tensor)
    )


def model_size_bytes(
    state,
    bytes_per_parameter=4,
):
    return (
        parameter_count(state)
        * bytes_per_parameter
    )


def state_distance(
    state_a,
    state_b,
):
    """
    Relative L2 distance between two model states.

        ||a-b|| / (||b|| + eps)
    """

    numerator = 0.0
    denominator = 0.0

    for name in state_a:

        a = state_a[name].float()
        b = state_b[name].float()

        numerator += torch.sum(
            (a - b) ** 2
        ).item()

        denominator += torch.sum(
            b ** 2
        ).item()

    return float(
        np.sqrt(numerator)
        / (
            np.sqrt(denominator)
            + EPS
        )
    )


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
            CLIENT_DIR
            / f"client_{client_id:02d}.csv"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing client file:\n"
                f"{path.resolve()}"
            )

        clients[client_id] = pd.read_csv(path)

    return clients


def load_test():
    if not GLOBAL_TEST_FILE.exists():
        raise FileNotFoundError(
            f"Missing global test file:\n"
            f"{GLOBAL_TEST_FILE.resolve()}"
        )

    return pd.read_csv(
        GLOBAL_TEST_FILE
    )


def load_schedule(
    scenario,
    rounds,
):
    path = (
        RELIABILITY_ROOT
        / scenario
        / "reliability_schedule.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Reliability schedule not found:\n"
            f"{path.resolve()}"
        )

    schedule = pd.read_csv(
        path
    )

    required = {
        "round",
        "client_id",
        "available",
        "sensor_noise_multiplier",
        "stale_update",
        "reliability_score",
    }

    missing = required - set(
        schedule.columns
    )

    if missing:
        raise ValueError(
            f"Schedule missing columns: {sorted(missing)}"
        )

    schedule["round"] = schedule[
        "round"
    ].astype(int)

    schedule["client_id"] = schedule[
        "client_id"
    ].astype(int)

    schedule["available"] = schedule[
        "available"
    ].astype(bool)

    schedule["sensor_noise_multiplier"] = schedule[
        "sensor_noise_multiplier"
    ].astype(float)

    schedule["stale_update"] = schedule[
        "stale_update"
    ].astype(bool)

    schedule["reliability_score"] = schedule[
        "reliability_score"
    ].astype(float)

    schedule = schedule[
        schedule["round"] <= rounds
    ].copy()

    return schedule.sort_values(
        ["round", "client_id"]
    ).reset_index(drop=True)


# ============================================================
# DATA PREPARATION
# ============================================================

def apply_scaling(
    clients,
    test,
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

    scaled_test = test.copy()

    scaled_test[
        feature_columns
    ] = scaler.transform(
        test[
            feature_columns
        ]
    )

    return (
        scaled_clients,
        scaled_test,
        feature_columns,
    )


# ============================================================
# SENSOR NOISE
# ============================================================

def apply_sensor_noise(
    df,
    feature_columns,
    multiplier,
    seed,
):
    if multiplier <= 0:
        return df

    rng = np.random.default_rng(
        seed
    )

    result = df.copy()

    values = result[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    std = np.std(
        values,
        axis=0,
        ddof=0,
    )

    noise = rng.normal(
        0.0,
        multiplier * (
            std + EPS
        ),
        size=values.shape,
    ).astype(np.float32)

    result[
        feature_columns
    ] = values + noise

    return result


# ============================================================
# EVALUATION
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

        probability = torch.softmax(
            logits,
            dim=1,
        )[:, 1]

        prediction = (
            probability >= 0.5
        ).long()

    pred = prediction.cpu().numpy()
    prob = probability.cpu().numpy()

    return {
        "global_test_loss": float(
            loss.item()
        ),
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "precision": float(
            precision_score(
                y, pred, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y, pred, zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y, pred, zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y, prob)
        ),
    }


# ============================================================
# RELIABILITY SCORE
# ============================================================

def normalize_weights(args):
    total = (
        args.data_weight
        + args.loss_weight
        + args.stability_weight
        + args.availability_weight
    )

    return (
        args.data_weight / total,
        args.loss_weight / total,
        args.stability_weight / total,
        args.availability_weight / total,
    )


def compute_reliability_scores(
    client_records,
    global_state,
    previous_global_state,
    recent_participation,
    max_samples,
    args,
):
    """
    Compute observable per-client reliability scores.

    client_records: list of dicts containing:
        client_id
        local_state
        local_loss
        sample_count

    The unavailable clients are not in client_records and therefore
    cannot contribute.

    The score is normalized across the currently participating clients.
    """

    if not client_records:
        raise RuntimeError(
            "No client records to score."
        )

    (
        w_data,
        w_loss,
        w_stability,
        w_availability,
    ) = normalize_weights(args)

    # --------------------------------------------------------
    # Raw quantities
    # --------------------------------------------------------

    raw = []

    for record in client_records:

        client_id = record["client_id"]
        local_state = record["local_state"]
        local_loss = float(
            record["local_loss"]
        )
        sample_count = int(
            record["sample_count"]
        )

        data_score = (
            np.sqrt(
                sample_count
                / max(
                    max_samples,
                    1
                )
            )
        )

        loss_score = float(
            np.exp(
                -max(
                    local_loss,
                    0.0
                )
            )
        )

        distance = state_distance(
            local_state,
            global_state,
        )

        stability_score = float(
            1.0
            / (
                1.0
                + distance
            )
        )

        availability_score = float(
            recent_participation.get(
                client_id,
                1.0,
            )
        )

        raw.append(
            {
                "client_id": client_id,
                "data_score": data_score,
                "loss_score": loss_score,
                "stability_score": stability_score,
                "availability_score": availability_score,
            }
        )

    raw_df = pd.DataFrame(raw)

    # --------------------------------------------------------
    # Aggregate score
    # --------------------------------------------------------

    raw_df["raw_reliability"] = (
        w_data
        * raw_df["data_score"]
        + w_loss
        * raw_df["loss_score"]
        + w_stability
        * raw_df["stability_score"]
        + w_availability
        * raw_df["availability_score"]
    )

    # Prevent all-zero scores.
    raw_df["raw_reliability"] = (
        raw_df["raw_reliability"]
        .clip(lower=EPS)
    )

    # --------------------------------------------------------
    # Normalize so aggregation weights sum to 1.
    # --------------------------------------------------------

    sample_lookup = {
        r["client_id"]: r["sample_count"]
        for r in client_records
    }

    raw_df["sample_count"] = raw_df[
        "client_id"
    ].map(sample_lookup)

    raw_df["weighted_score"] = (
        raw_df["raw_reliability"]
        * raw_df["sample_count"]
    )

    total = raw_df[
        "weighted_score"
    ].sum()

    raw_df["aggregation_weight"] = (
        raw_df["weighted_score"]
        / (
            total
            + EPS
        )
    )

    return raw_df


# ============================================================
# AGGREGATION
# ============================================================

def reliability_aggregate(
    client_records,
    reliability_df,
):
    if not client_records:
        raise RuntimeError(
            "No client records supplied."
        )

    lookup = {
        int(row["client_id"]): float(
            row["aggregation_weight"]
        )
        for _, row in reliability_df.iterrows()
    }

    global_state = OrderedDict()

    first_state = client_records[0]["local_state"]

    for name in first_state:

        ref = first_state[name]

        if torch.is_floating_point(ref):

            aggregate = torch.zeros_like(
                ref,
                dtype=torch.float32,
            )

            for record in client_records:

                client_id = record[
                    "client_id"
                ]

                weight = lookup[
                    client_id
                ]

                aggregate += (
                    record["local_state"][
                        name
                    ].float()
                    * weight
                )

            global_state[name] = aggregate

        else:

            # Current model is entirely floating-point.
            # For robustness, keep the first tensor for
            # non-floating buffers.
            global_state[name] = ref.clone()

    return global_state


# ============================================================
# EXPERIMENT
# ============================================================

def run_experiment(
    scenario,
    rounds,
    local_epochs,
    batch_size,
    learning_rate,
    seed,
    args,
):
    set_seed(seed)

    clients = load_clients()
    global_test = load_test()

    schedule = load_schedule(
        scenario,
        rounds,
    )

    (
        scaled_clients,
        scaled_test,
        feature_columns,
    ) = apply_scaling(
        clients,
        global_test,
    )

    model = build_model(
        input_size=len(feature_columns)
    ).to(DEVICE)

    global_state = copy_state(
        model.state_dict()
    )

    model_bytes = model_size_bytes(
        global_state
    )

    # Participation history used only as a rolling availability feature.
    recent_participation = {
        client_id: 1.0
        for client_id in range(
            1,
            N_CLIENTS + 1,
        )
    }

    # Store previous client states for stale simulation.
    previous_client_states = {}

    history = []
    all_score_rows = []

    print("=" * 82)
    print("RELIABILITY-AWARE FEDAVG")
    print("=" * 82)

    print(
        f"Scenario             : {scenario}"
    )

    print(
        f"Clients              : {N_CLIENTS}"
    )

    print(
        f"Rounds               : {rounds}"
    )

    print(
        f"Device               : {DEVICE}"
    )

    print(
        "Reliability weights  : "
        f"data={args.data_weight:.3f}, "
        f"loss={args.loss_weight:.3f}, "
        f"stability={args.stability_weight:.3f}, "
        f"availability={args.availability_weight:.3f}"
    )

    for round_number in range(
        1,
        rounds + 1,
    ):
        round_start = time.perf_counter()

        round_schedule = schedule[
            schedule["round"] == round_number
        ]

        available_rows = round_schedule[
            round_schedule["available"]
        ]

        participating_ids = [
            int(x)
            for x in available_rows[
                "client_id"
            ].tolist()
        ]

        if not participating_ids:
            raise RuntimeError(
                f"No participating clients in round {round_number}."
            )

        local_records = []

        actual_noisy = 0
        actual_stale = 0

        print()
        print("-" * 82)
        print(
            f"ROUND {round_number}/{rounds}"
        )

        print(
            "Participating clients:",
            participating_ids,
        )

        for client_id in participating_ids:

            schedule_row = round_schedule[
                round_schedule["client_id"]
                == client_id
            ].iloc[0]

            client_seed = (
                seed
                + round_number * 1000
                + client_id
            )

            set_seed(
                client_seed
            )

            client_df = scaled_clients[
                client_id
            ]

            noise_multiplier = float(
                schedule_row[
                    "sensor_noise_multiplier"
                ]
            )

            if noise_multiplier > 0:

                actual_noisy += 1

                client_df = apply_sensor_noise(
                    client_df,
                    feature_columns,
                    noise_multiplier,
                    seed
                    + round_number * 10000
                    + client_id,
                )

            (
                current_state,
                local_loss,
                sample_count,
            ) = train_local_model(
                global_state=global_state,
                client_df=client_df,
                device=DEVICE,
                input_size=len(feature_columns),
                local_epochs=local_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
            )

            previous_state = previous_client_states.get(
                client_id
            )

            previous_client_states[
                client_id
            ] = copy_state(
                current_state
            )

            stale_requested = bool(
                schedule_row["stale_update"]
            )

            submitted_state = current_state

            stale_used = False

            if (
                stale_requested
                and previous_state is not None
            ):
                submitted_state = copy_state(
                    previous_state
                )
                stale_used = True
                actual_stale += 1

            local_records.append(
                {
                    "client_id": client_id,
                    "local_state": submitted_state,
                    "current_state": current_state,
                    "local_loss": float(local_loss),
                    "sample_count": int(sample_count),
                }
            )

        # ----------------------------------------------------
        # Reliability scores
        # ----------------------------------------------------

        max_samples = max(
            r["sample_count"]
            for r in local_records
        )

        # Update rolling availability from current schedule.
        for client_id in range(
            1,
            N_CLIENTS + 1,
        ):
            available = bool(
                round_schedule[
                    round_schedule["client_id"]
                    == client_id
                ]["available"].iloc[0]
            )

            recent_participation[
                client_id
            ] = (
                0.8
                * recent_participation.get(
                    client_id,
                    1.0,
                )
                + 0.2
                * float(available)
            )

        reliability_df = compute_reliability_scores(
            client_records=local_records,
            global_state=global_state,
            previous_global_state=global_state,
            recent_participation=recent_participation,
            max_samples=max_samples,
            args=args,
        )

        reliability_df[
            "round"
        ] = round_number

        reliability_df[
            "scenario"
        ] = scenario

        all_score_rows.append(
            reliability_df
        )

        # ----------------------------------------------------
        # Reliability-aware aggregation
        # ----------------------------------------------------

        aggregation_start = time.perf_counter()

        global_state = reliability_aggregate(
            local_records,
            reliability_df,
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
            participating_ids
        )

        upload_bytes = (
            participating
            * model_bytes
        )

        download_bytes = upload_bytes

        total_communication = (
            upload_bytes
            + download_bytes
        )

        reliability_mean = float(
            reliability_df[
                "raw_reliability"
            ].mean()
        )

        history.append(
            {
                "scenario": scenario,
                "round": round_number,
                "available_clients": participating,
                "failed_clients": N_CLIENTS - participating,
                "noisy_clients": actual_noisy,
                "stale_clients": actual_stale,
                "mean_raw_reliability": reliability_mean,
                "mean_local_loss": float(
                    np.mean(
                        [
                            r["local_loss"]
                            for r in local_records
                        ]
                    )
                ),
                "global_test_loss": metrics[
                    "global_test_loss"
                ],
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
                "total_communication_bytes": total_communication,
                "aggregation_time_seconds": aggregation_time,
                "round_time_seconds": round_time,
            }
        )

        print(
            f"Test loss     : {metrics['global_test_loss']:.6f}"
        )
        print(
            f"Accuracy      : {metrics['accuracy']:.6f}"
        )
        print(
            f"Precision     : {metrics['precision']:.6f}"
        )
        print(
            f"Recall        : {metrics['recall']:.6f}"
        )
        print(
            f"F1            : {metrics['f1']:.6f}"
        )
        print(
            f"ROC-AUC       : {metrics['roc_auc']:.6f}"
        )
        print(
            f"Reliability μ : {reliability_mean:.6f}"
        )

        print(
            "Aggregation weights:"
        )

        print(
            reliability_df[
                [
                    "client_id",
                    "sample_count",
                    "data_score",
                    "loss_score",
                    "stability_score",
                    "availability_score",
                    "raw_reliability",
                    "aggregation_weight",
                ]
            ].round(4).to_string(
                index=False
            )
        )

    return (
        pd.DataFrame(history),
        pd.concat(
            all_score_rows,
            ignore_index=True,
        ),
    )


# ============================================================
# SAVE
# ============================================================

def save_results(
    history,
    score_history,
    scenario,
    args,
):
    output_dir = (
        OUTPUT_ROOT / scenario
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history.to_csv(
        output_dir
        / "ra_fedavg_round_history.csv",
        index=False,
    )

    score_history.to_csv(
        output_dir
        / "ra_fedavg_client_scores.csv",
        index=False,
    )

    final = history.iloc[-1]

    total_communication = float(
        history[
            "total_communication_bytes"
        ].sum()
    )

    final_metrics = pd.DataFrame(
        {
            "metric": [
                "scenario",
                "rounds",
                "clients",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "seed",
                "data_weight",
                "loss_weight",
                "stability_weight",
                "availability_weight",
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
                scenario,
                args.rounds,
                N_CLIENTS,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                args.seed,
                args.data_weight,
                args.loss_weight,
                args.stability_weight,
                args.availability_weight,
                final["accuracy"],
                final["precision"],
                final["recall"],
                final["f1"],
                final["roc_auc"],
                final["global_test_loss"],
                total_communication,
                total_communication / (1024 ** 2),
                history[
                    "round_time_seconds"
                ].sum(),
            ],
        }
    )

    final_metrics.to_csv(
        output_dir
        / "ra_fedavg_final_metrics.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "scenario",
                "clients",
                "rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "seed",
                "data_weight",
                "loss_weight",
                "stability_weight",
                "availability_weight",
                "aggregation",
            ],
            "value": [
                scenario,
                N_CLIENTS,
                args.rounds,
                args.local_epochs,
                args.batch_size,
                args.learning_rate,
                args.seed,
                args.data_weight,
                args.loss_weight,
                args.stability_weight,
                args.availability_weight,
                "Reliability-aware weighted aggregation",
            ],
        }
    )

    config.to_csv(
        output_dir
        / "ra_fedavg_config.csv",
        index=False,
    )

    return output_dir


# ============================================================
# FIGURES
# ============================================================

def create_figures(
    history,
    score_history,
    scenario,
    output_dir,
):
    fig, ax = plt.subplots(figsize=(10, 6))

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

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"RA-FedAvg Metrics — {scenario}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        output_dir / "metrics_vs_round.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    # Aggregate-weight evolution
    fig, ax = plt.subplots(figsize=(11, 6))

    for client_id in sorted(
        score_history["client_id"].unique()
    ):
        client = score_history[
            score_history["client_id"] == client_id
        ]

        ax.plot(
            client["round"],
            client["aggregation_weight"],
            marker="o",
            linewidth=1.0,
            label=f"Client {client_id}",
        )

    ax.set_xlabel("Federated round")
    ax.set_ylabel("Aggregation weight")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        f"RA-FedAvg Client Aggregation Weights — {scenario}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend(
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(
        output_dir / "client_weights_vs_round.png",
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
        raise ValueError("--rounds must be > 0.")

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

    validate_weights(args)

    set_seed(args.seed)

    (
        history,
        score_history,
    ) = run_experiment(
        scenario=args.scenario,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        args=args,
    )

    output_dir = save_results(
        history,
        score_history,
        args.scenario,
        args,
    )

    create_figures(
        history,
        score_history,
        args.scenario,
        output_dir,
    )

    final = history.iloc[-1]

    print()
    print("=" * 82)
    print("RA-FEDAVG EXPERIMENT COMPLETE")
    print("=" * 82)

    print(
        f"Scenario        : {args.scenario}"
    )

    print(
        f"Final accuracy  : {final['accuracy']:.6f}"
    )

    print(
        f"Final precision : {final['precision']:.6f}"
    )

    print(
        f"Final recall    : {final['recall']:.6f}"
    )

    print(
        f"Final F1        : {final['f1']:.6f}"
    )

    print(
        f"Final ROC-AUC   : {final['roc_auc']:.6f}"
    )

    print(
        f"Final test loss : {final['global_test_loss']:.6f}"
    )

    print(
        f"Total communication: "
        f"{history['total_communication_bytes'].sum() / (1024 ** 2):.4f} MiB"
    )

    print(
        "Output:",
        output_dir.resolve(),
    )

    print("=" * 82)


if __name__ == "__main__":
    main()
