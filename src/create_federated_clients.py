"""
create_federated_clients.py

Create IID and non-IID federated client datasets for the
NASA IMS Dataset 2 experiment.

Input:
    ../results/federated/dataset2_fl_dataset.csv

Protocol:
    1. Stratified global train/test split: 80/20.
    2. Test set is global and NEVER assigned to clients.
    3. Training pool is partitioned among 10 clients.
    4. IID clients: approximately equal class distributions.
    5. Non-IID clients: Dirichlet class allocation.
    6. Every client is required to contain both classes.

This is our controlled FL experimental setup. It is not claimed
to reproduce an undisclosed client partition from Marfo et al.

Outputs:
    ../results/federated/dataset2_clients/
        global_test.csv
        train_pool.csv

        iid/
            client_01.csv
            ...
            client_10.csv
            client_statistics.csv

        noniid/
            client_01.csv
            ...
            client_10.csv
            client_statistics.csv

        partition_summary.csv
        partition_config.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    "../results/federated/dataset2_fl_dataset.csv"
)

OUTPUT_DIR = Path(
    "../results/federated/dataset2_clients"
)

N_CLIENTS = 10

TEST_SIZE = 0.20

SEED = 42

# Smaller alpha -> stronger non-IID heterogeneity.
DIRICHLET_ALPHA = 0.30

MIN_CLASSES_PER_CLIENT = 2

LABEL_COLUMN = "label"

# Columns that identify a recording rather than model features.
METADATA_COLUMNS = {
    "dataset",
    "file_index",
    "timestamp",
    "quantization_error",
    "label",
    "is_baseline",
    "phase",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file not found:\n"
            f"{INPUT_FILE.resolve()}\n\n"
            f"Run create_fl_dataset.py and create_labels.py first."
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "file_index",
        "timestamp",
        LABEL_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:

        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    df[LABEL_COLUMN] = (
        df[LABEL_COLUMN]
        .astype(int)
    )

    labels = sorted(
        df[LABEL_COLUMN].unique()
    )

    if labels != [0, 1]:

        raise ValueError(
            f"Expected binary labels [0, 1], got {labels}"
        )

    return df


# ============================================================
# GLOBAL STRATIFIED SPLIT
# ============================================================

def create_global_split(df):

    train_pool, global_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=df[LABEL_COLUMN],
        shuffle=True,
    )

    train_pool = train_pool.sort_values(
        "file_index"
    ).reset_index(drop=True)

    global_test = global_test.sort_values(
        "file_index"
    ).reset_index(drop=True)

    return train_pool, global_test


# ============================================================
# IID PARTITION
# ============================================================

def create_iid_clients(
    train_pool,
    n_clients,
):

    rng = np.random.default_rng(SEED)

    shuffled = train_pool.sample(
        frac=1.0,
        random_state=SEED,
    ).reset_index(drop=True)

    client_parts = [
        []
        for _ in range(n_clients)
    ]

    for index, (_, row) in enumerate(
        shuffled.iterrows()
    ):

        client_id = index % n_clients

        client_parts[
            client_id
        ].append(row)

    clients = {}

    for client_id, rows in enumerate(
        client_parts,
        start=1,
    ):

        client_df = pd.DataFrame(
            rows
        ).reset_index(
            drop=True
        )

        # Shuffle independently after assignment.
        client_df = client_df.sample(
            frac=1.0,
            random_state=SEED + client_id,
        ).reset_index(drop=True)

        clients[
            client_id
        ] = client_df

    return clients


# ============================================================
# NON-IID DIRICHLET PARTITION
# ============================================================

def create_noniid_clients(
    train_pool,
    n_clients,
    alpha,
    max_attempts=1000,
):

    rng = np.random.default_rng(
        SEED + 1000
    )

    labels = sorted(
        train_pool[LABEL_COLUMN].unique()
    )

    # We retry until every client receives at least one
    # sample from every class.
    for attempt in range(
        max_attempts
    ):

        client_indices = [
            []
            for _ in range(n_clients)
        ]

        for label in labels:

            label_indices = np.where(
                train_pool[LABEL_COLUMN].to_numpy()
                == label
            )[0]

            rng.shuffle(
                label_indices
            )

            proportions = rng.dirichlet(
                np.full(
                    n_clients,
                    alpha,
                    dtype=float,
                )
            )

            # Convert proportions to integer counts while
            # preserving the exact number of class samples.
            raw_counts = (
                proportions
                * len(label_indices)
            )

            counts = np.floor(
                raw_counts
            ).astype(int)

            remainder = (
                len(label_indices)
                - int(counts.sum())
            )

            if remainder > 0:

                fractional = (
                    raw_counts
                    - counts
                )

                order = np.argsort(
                    fractional
                )[::-1]

                for j in order[
                    :remainder
                ]:

                    counts[j] += 1

            start = 0

            for client_id, count in enumerate(
                counts
            ):

                selected = label_indices[
                    start:start + count
                ]

                client_indices[
                    client_id
                ].extend(
                    selected.tolist()
                )

                start += count

        # Verify every client has every class.
        valid = True

        for indices in client_indices:

            if len(indices) == 0:

                valid = False
                break

            client_labels = set(
                train_pool.iloc[
                    indices
                ][LABEL_COLUMN].tolist()
            )

            if len(client_labels) < MIN_CLASSES_PER_CLIENT:

                valid = False
                break

        if valid:

            clients = {}

            for client_id, indices in enumerate(
                client_indices,
                start=1,
            ):

                client_df = train_pool.iloc[
                    indices
                ].copy()

                client_df = client_df.sample(
                    frac=1.0,
                    random_state=SEED + client_id,
                ).reset_index(
                    drop=True
                )

                clients[
                    client_id
                ] = client_df

            return clients

    raise RuntimeError(
        "Could not create a valid non-IID partition "
        f"after {max_attempts} attempts. "
        f"Try increasing DIRICHLET_ALPHA."
    )


# ============================================================
# CLIENT STATISTICS
# ============================================================

def client_statistics(
    clients,
    partition_name,
):

    rows = []

    for client_id, df in clients.items():

        n = len(df)

        normal = int(
            (df[LABEL_COLUMN] == 0).sum()
        )

        anomaly = int(
            (df[LABEL_COLUMN] == 1).sum()
        )

        normal_pct = (
            normal / n
            if n
            else 0.0
        )

        anomaly_pct = (
            anomaly / n
            if n
            else 0.0
        )

        rows.append(
            {
                "partition": partition_name,
                "client_id": client_id,
                "samples": n,
                "normal_samples": normal,
                "anomaly_samples": anomaly,
                "normal_percentage": normal_pct,
                "anomaly_percentage": anomaly_pct,
                "min_file_index": int(
                    df["file_index"].min()
                ),
                "max_file_index": int(
                    df["file_index"].max()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SAVE PARTITION
# ============================================================

def save_partition(
    clients,
    partition_name,
    root_output,
):

    output = (
        root_output
        / partition_name
    )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    statistics = client_statistics(
        clients,
        partition_name,
    )

    for client_id, df in clients.items():

        path = (
            output
            / f"client_{client_id:02d}.csv"
        )

        df.to_csv(
            path,
            index=False,
        )

    statistics.to_csv(
        output
        / "client_statistics.csv",
        index=False,
    )

    return statistics


# ============================================================
# CREATE SUMMARY
# ============================================================

def create_partition_summary(
    train_pool,
    global_test,
    iid_clients,
    noniid_clients,
):

    rows = []

    def add_global(
        name,
        df,
    ):

        rows.append(
            {
                "partition": name,
                "client_id": "global",
                "samples": len(df),
                "normal_samples": int(
                    (df[LABEL_COLUMN] == 0).sum()
                ),
                "anomaly_samples": int(
                    (df[LABEL_COLUMN] == 1).sum()
                ),
                "normal_percentage": (
                    (df[LABEL_COLUMN] == 0).mean()
                ),
                "anomaly_percentage": (
                    (df[LABEL_COLUMN] == 1).mean()
                ),
            }
        )

    add_global(
        "train_pool",
        train_pool,
    )

    add_global(
        "global_test",
        global_test,
    )

    for partition_name, clients in [
        ("iid", iid_clients),
        ("noniid", noniid_clients),
    ]:

        for client_id, df in clients.items():

            rows.append(
                {
                    "partition": partition_name,
                    "client_id": client_id,
                    "samples": len(df),
                    "normal_samples": int(
                        (df[LABEL_COLUMN] == 0).sum()
                    ),
                    "anomaly_samples": int(
                        (df[LABEL_COLUMN] == 1).sum()
                    ),
                    "normal_percentage": (
                        (df[LABEL_COLUMN] == 0).mean()
                    ),
                    "anomaly_percentage": (
                        (df[LABEL_COLUMN] == 1).mean()
                    ),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# VALIDATION
# ============================================================

def validate_partition(
    train_pool,
    clients,
    partition_name,
):

    original_ids = set(
        train_pool["file_index"]
    )

    assigned_ids = set()

    for client_id, df in clients.items():

        client_ids = set(
            df["file_index"]
        )

        overlap = (
            assigned_ids
            & client_ids
        )

        if overlap:

            raise AssertionError(
                f"{partition_name}: duplicate records "
                f"across clients {client_id}: {overlap}"
            )

        assigned_ids.update(
            client_ids
        )

    if original_ids != assigned_ids:

        missing = (
            original_ids
            - assigned_ids
        )

        extra = (
            assigned_ids
            - original_ids
        )

        raise AssertionError(
            f"{partition_name}: partition mismatch.\n"
            f"Missing records: {len(missing)}\n"
            f"Extra records: {len(extra)}"
        )

    print(
        f"{partition_name}: "
        f"all {len(original_ids)} training records "
        f"assigned exactly once."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FEDERATED CLIENT PARTITIONING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    print()
    print(
        "Full dataset:",
        df.shape
    )

    print(
        "Class distribution:"
    )

    print(
        df[LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Global split
    # --------------------------------------------------------

    train_pool, global_test = (
        create_global_split(df)
    )

    print()
    print("=" * 70)
    print("GLOBAL TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        "Train pool:",
        len(train_pool)
    )

    print(
        "Global test:",
        len(global_test)
    )

    print()
    print(
        "Train class distribution:"
    )

    print(
        train_pool[LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "Test class distribution:"
    )

    print(
        global_test[LABEL_COLUMN]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # IID
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING IID CLIENTS")
    print("=" * 70)

    iid_clients = create_iid_clients(
        train_pool,
        N_CLIENTS,
    )

    validate_partition(
        train_pool,
        iid_clients,
        "IID",
    )

    # --------------------------------------------------------
    # Non-IID
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CREATING NON-IID CLIENTS")
    print("=" * 70)

    print(
        f"Dirichlet alpha = "
        f"{DIRICHLET_ALPHA}"
    )

    noniid_clients = create_noniid_clients(
        train_pool,
        N_CLIENTS,
        DIRICHLET_ALPHA,
    )

    validate_partition(
        train_pool,
        noniid_clients,
        "NON-IID",
    )

    # --------------------------------------------------------
    # Output directories
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save global data
    # --------------------------------------------------------

    train_pool.to_csv(
        OUTPUT_DIR / "train_pool.csv",
        index=False,
    )

    global_test.to_csv(
        OUTPUT_DIR / "global_test.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Save clients
    # --------------------------------------------------------

    iid_stats = save_partition(
        iid_clients,
        "iid",
        OUTPUT_DIR,
    )

    noniid_stats = save_partition(
        noniid_clients,
        "noniid",
        OUTPUT_DIR,
    )

    # --------------------------------------------------------
    # Save configuration
    # --------------------------------------------------------

    config = pd.DataFrame(
        {
            "parameter": [
                "input_file",
                "total_samples",
                "test_size",
                "train_samples",
                "test_samples",
                "n_clients",
                "random_seed",
                "noniid_method",
                "dirichlet_alpha",
                "minimum_classes_per_client",
            ],
            "value": [
                str(INPUT_FILE),
                len(df),
                TEST_SIZE,
                len(train_pool),
                len(global_test),
                N_CLIENTS,
                SEED,
                "Dirichlet",
                DIRICHLET_ALPHA,
                MIN_CLASSES_PER_CLIENT,
            ],
        }
    )

    config.to_csv(
        OUTPUT_DIR / "partition_config.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = create_partition_summary(
        train_pool,
        global_test,
        iid_clients,
        noniid_clients,
    )

    summary.to_csv(
        OUTPUT_DIR / "partition_summary.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Print client statistics
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("IID CLIENT STATISTICS")
    print("=" * 70)

    print(
        iid_stats.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("NON-IID CLIENT STATISTICS")
    print("=" * 70)

    print(
        noniid_stats.to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("PARTITIONING COMPLETE")
    print("=" * 70)

    print(
        "Output:",
        OUTPUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()