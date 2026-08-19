"""
create_balanced_noniid.py

Create a BALANCED non-IID client partition for NASA IMS Dataset 2.

Goal:
    Isolate class-distribution heterogeneity while keeping client sizes
    approximately equal.

Fixed:
    - Same global train/test split already created.
    - Same 787-record training pool.
    - Same 197-record global test set.
    - 10 clients.
    - Client sizes differ by at most one sample.
    - Every client contains both normal and anomaly samples.

Non-IID mechanism:
    - Draw target anomaly proportions from a Dirichlet distribution.
    - Convert proportions into exact anomaly counts subject to each
      client's capacity.
    - Allocate records without duplication.

Output:
    ../results/federated/dataset2_clients/balanced_noniid/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path("../results/federated/dataset2_clients")

TRAIN_POOL = BASE_DIR / "train_pool.csv"
GLOBAL_TEST = BASE_DIR / "global_test.csv"

OUTPUT_DIR = BASE_DIR / "balanced_noniid"

N_CLIENTS = 10
SEED = 42

DIRICHLET_ALPHA = 0.30

MIN_ANOMALIES_PER_CLIENT = 2
MIN_NORMALS_PER_CLIENT = 2


def load_data():
    if not TRAIN_POOL.exists():
        raise FileNotFoundError(
            f"Missing:\n{TRAIN_POOL.resolve()}"
        )

    train = pd.read_csv(TRAIN_POOL)

    if "label" not in train.columns:
        raise ValueError("train_pool.csv must contain 'label'.")

    train["label"] = train["label"].astype(int)

    if set(train["label"].unique()) != {0, 1}:
        raise ValueError("Expected binary labels 0 and 1.")

    return train


def client_sizes(total, n_clients):
    """
    Split total samples as evenly as possible.
    Example: 787 / 10 -> [79,79,79,79,79,79,79,78,78,78].
    """
    base = total // n_clients
    remainder = total % n_clients

    sizes = [
        base + (1 if i < remainder else 0)
        for i in range(n_clients)
    ]

    return sizes


def bounded_integer_allocation(
    proportions,
    total,
    capacities,
    minimums,
    rng,
):
    """
    Convert continuous proportions to integer counts.

    Constraints:
        minimums[i] <= counts[i] <= capacities[i]
        sum(counts) == total
    """

    n = len(capacities)

    if sum(minimums) > total:
        raise ValueError(
            "Minimum allocations exceed total."
        )

    if sum(capacities) < total:
        raise ValueError(
            "Capacities cannot accommodate total."
        )

    proportions = np.asarray(
        proportions,
        dtype=float,
    )

    proportions = proportions / proportions.sum()

    ideal = proportions * total

    counts = np.floor(ideal).astype(int)

    counts = np.maximum(
        counts,
        np.asarray(minimums),
    )

    counts = np.minimum(
        counts,
        np.asarray(capacities),
    )

    # Adjust until exact total is reached.
    while counts.sum() < total:

        candidates = [
            i
            for i in range(n)
            if counts[i] < capacities[i]
        ]

        if not candidates:
            raise RuntimeError(
                "Could not increase allocation to total."
            )

        # Largest remaining fractional need first.
        scores = {
            i: ideal[i] - counts[i]
            for i in candidates
        }

        best_score = max(
            scores.values()
        )

        best = [
            i
            for i, score in scores.items()
            if np.isclose(
                score,
                best_score,
            )
        ]

        chosen = rng.choice(best)

        counts[chosen] += 1

    while counts.sum() > total:

        candidates = [
            i
            for i in range(n)
            if counts[i] > minimums[i]
        ]

        if not candidates:
            raise RuntimeError(
                "Could not reduce allocation to total."
            )

        scores = {
            i: counts[i] - ideal[i]
            for i in candidates
        }

        best_score = max(
            scores.values()
        )

        best = [
            i
            for i, score in scores.items()
            if np.isclose(
                score,
                best_score,
            )
        ]

        chosen = rng.choice(best)

        counts[chosen] -= 1

    return counts


def create_partition(train):
    rng = np.random.default_rng(SEED)

    n = len(train)

    sizes = client_sizes(
        n,
        N_CLIENTS,
    )

    total_anomalies = int(
        (train["label"] == 1).sum()
    )

    total_normal = int(
        (train["label"] == 0).sum()
    )

    # Draw heterogeneous target proportions.
    proportions = rng.dirichlet(
        np.full(
            N_CLIENTS,
            DIRICHLET_ALPHA,
        )
    )

    # Each client must contain both classes.
    min_anomaly = np.full(
        N_CLIENTS,
        MIN_ANOMALIES_PER_CLIENT,
        dtype=int,
    )

    max_anomaly = (
        np.asarray(sizes)
        - MIN_NORMALS_PER_CLIENT
    )

    anomaly_counts = bounded_integer_allocation(
        proportions=proportions,
        total=total_anomalies,
        capacities=max_anomaly,
        minimums=min_anomaly,
        rng=rng,
    )

    normal_counts = (
        np.asarray(sizes)
        - anomaly_counts
    )

    if normal_counts.sum() != total_normal:
        raise RuntimeError(
            "Normal/anomaly totals do not match."
        )

    anomaly_pool = (
        train[
            train["label"] == 1
        ]
        .sample(
            frac=1.0,
            random_state=SEED + 1,
        )
        .reset_index(drop=True)
    )

    normal_pool = (
        train[
            train["label"] == 0
        ]
        .sample(
            frac=1.0,
            random_state=SEED + 2,
        )
        .reset_index(drop=True)
    )

    clients = {}

    anomaly_pos = 0
    normal_pos = 0

    for client_idx in range(N_CLIENTS):

        n_anomaly = int(
            anomaly_counts[client_idx]
        )

        n_normal = int(
            normal_counts[client_idx]
        )

        anomaly_part = anomaly_pool.iloc[
            anomaly_pos:
            anomaly_pos + n_anomaly
        ]

        normal_part = normal_pool.iloc[
            normal_pos:
            normal_pos + n_normal
        ]

        anomaly_pos += n_anomaly
        normal_pos += n_normal

        client = pd.concat(
            [
                normal_part,
                anomaly_part,
            ],
            ignore_index=True,
        )

        # Shuffle within client.
        client = client.sample(
            frac=1.0,
            random_state=SEED + 100 + client_idx,
        ).reset_index(drop=True)

        clients[
            client_idx + 1
        ] = client

    # Verify exact coverage.
    original_ids = set(
        train["file_index"]
    )

    assigned_ids = set()

    for client in clients.values():
        assigned_ids.update(
            client["file_index"]
        )

    if original_ids != assigned_ids:
        raise RuntimeError(
            "Client partition does not contain exactly "
            "the full training pool."
        )

    if sum(
        len(client)
        for client in clients.values()
    ) != len(train):

        raise RuntimeError(
            "Client sample count mismatch."
        )

    return clients, sizes, anomaly_counts, normal_counts


def save_partition(
    clients,
    sizes,
    anomaly_counts,
    normal_counts,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for client_id, client in clients.items():

        output = (
            OUTPUT_DIR
            / f"client_{client_id:02d}.csv"
        )

        client.to_csv(
            output,
            index=False,
        )

        n = len(client)
        anomalies = int(
            client["label"].sum()
        )

        normal = int(
            (client["label"] == 0).sum()
        )

        rows.append(
            {
                "partition": "balanced_noniid",
                "client_id": client_id,
                "samples": n,
                "normal_samples": normal,
                "anomaly_samples": anomalies,
                "normal_percentage": normal / n,
                "anomaly_percentage": anomalies / n,
                "min_file_index": int(
                    client["file_index"].min()
                ),
                "max_file_index": int(
                    client["file_index"].max()
                ),
            }
        )

    statistics = pd.DataFrame(rows)

    statistics.to_csv(
        OUTPUT_DIR / "client_statistics.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "n_clients",
                "seed",
                "dirichlet_alpha",
                "client_size_min",
                "client_size_max",
                "minimum_anomalies_per_client",
                "minimum_normals_per_client",
            ],
            "value": [
                N_CLIENTS,
                SEED,
                DIRICHLET_ALPHA,
                min(sizes),
                max(sizes),
                MIN_ANOMALIES_PER_CLIENT,
                MIN_NORMALS_PER_CLIENT,
            ],
        }
    )

    config.to_csv(
        OUTPUT_DIR / "partition_config.csv",
        index=False,
    )

    return statistics


def main():

    print("=" * 70)
    print("BALANCED NON-IID CLIENT PARTITION")
    print("=" * 70)

    train = load_data()

    print(
        "Training pool:",
        train.shape,
    )

    print(
        "Class distribution:"
    )

    print(
        train["label"]
        .value_counts()
        .sort_index()
    )

    clients, sizes, anomaly_counts, normal_counts = (
        create_partition(train)
    )

    statistics = save_partition(
        clients,
        sizes,
        anomaly_counts,
        normal_counts,
    )

    print()
    print("=" * 70)
    print("BALANCED NON-IID CLIENT STATISTICS")
    print("=" * 70)

    print(
        statistics.to_string(
            index=False,
        )
    )

    print()
    print(
        "Total client samples:",
        statistics["samples"].sum(),
    )

    print(
        "Total anomaly samples:",
        statistics["anomaly_samples"].sum(),
    )

    print(
        "Total normal samples:",
        statistics["normal_samples"].sum(),
    )

    print()
    print(
        "Output:",
        OUTPUT_DIR.resolve(),
    )

    print("=" * 70)
    print("PARTITION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
