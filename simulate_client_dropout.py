"""
simulate_client_dropout.py

Create deterministic per-round client participation schedules for
federated-learning robustness experiments.

The simulator DOES NOT modify client data. It only determines which
clients participate in each communication round.

Supported dropout rates:
    0.0
    0.2
    0.4
    0.5

For each round:
    - a fixed fraction of clients is randomly unavailable;
    - at least one client always participates;
    - the same random seed makes the schedule reproducible.

The output schedule can later be consumed by run_fedavg.py.

Default client source:
    ../results/federated/dataset2_clients/noniid

Outputs:
    ../results/federated/dataset2_dropout/
        dropout_0pct/
            client_participation.csv
            dropout_config.csv
        dropout_20pct/
            ...
        dropout_40pct/
            ...
        dropout_50pct/
            ...

Each participation CSV contains one row per client per round:
    round
    client_id
    participates
    dropout_rate
    seed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# DEFAULT CONFIGURATION
# ============================================================

CLIENT_DIR = Path(
    "../results/federated/dataset2_clients/noniid"
)

OUTPUT_ROOT = Path(
    "../results/federated/dataset2_dropout"
)

N_CLIENTS = 10

ROUNDS = 15

SEED = 42

DEFAULT_RATES = [
    0.0,
    0.20,
    0.40,
    0.50,
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic client dropout schedules "
            "for federated-learning experiments."
        )
    )

    parser.add_argument(
        "--rates",
        nargs="+",
        type=float,
        default=DEFAULT_RATES,
        help=(
            "Dropout rates to generate. "
            "Example: --rates 0 0.2 0.4 0.5"
        ),
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
        help="Number of FL communication rounds.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Base random seed.",
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=N_CLIENTS,
        help="Number of participating clients.",
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_rates(
    rates,
):

    for rate in rates:

        if not 0.0 <= rate < 1.0:
            raise ValueError(
                f"Invalid dropout rate: {rate}. "
                "Use 0 <= rate < 1."
            )


def validate_client_directory(
    client_dir: Path,
    n_clients: int,
):

    if not client_dir.exists():

        raise FileNotFoundError(
            f"Client directory not found:\n"
            f"{client_dir.resolve()}"
        )

    missing = []

    for client_id in range(
        1,
        n_clients + 1,
    ):

        file = (
            client_dir
            / f"client_{client_id:02d}.csv"
        )

        if not file.exists():
            missing.append(
                file.name
            )

    if missing:

        raise FileNotFoundError(
            "Missing client files:\n"
            + "\n".join(missing)
        )


# ============================================================
# PARTICIPATING CLIENT COUNT
# ============================================================

def number_of_dropped_clients(
    n_clients: int,
    dropout_rate: float,
) -> int:
    """
    Determine a deterministic integer number of dropped clients.

    We round to the nearest integer, but always keep at least
    one client participating.
    """

    if dropout_rate == 0.0:
        return 0

    dropped = int(
        round(
            n_clients
            * dropout_rate
        )
    )

    dropped = min(
        dropped,
        n_clients - 1,
    )

    return max(
        dropped,
        1,
    )


# ============================================================
# CREATE ONE SCHEDULE
# ============================================================

def create_schedule(
    n_clients: int,
    rounds: int,
    dropout_rate: float,
    seed: int,
):
    """
    Create a deterministic client participation matrix.
    """

    rng = np.random.default_rng(
        seed
    )

    rows = []

    dropped_per_round = (
        number_of_dropped_clients(
            n_clients,
            dropout_rate,
        )
    )

    all_clients = np.arange(
        1,
        n_clients + 1,
    )

    for round_number in range(
        1,
        rounds + 1,
    ):

        if dropped_per_round == 0:

            dropped = set()

        else:

            dropped = set(
                rng.choice(
                    all_clients,
                    size=dropped_per_round,
                    replace=False,
                ).tolist()
            )

        for client_id in all_clients:

            participates = (
                client_id
                not in dropped
            )

            rows.append(
                {
                    "round": round_number,
                    "client_id": int(client_id),
                    "participates": bool(participates),
                    "dropout_rate": dropout_rate,
                    "seed": seed,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# SAVE SCHEDULE
# ============================================================

def save_schedule(
    schedule: pd.DataFrame,
    dropout_rate: float,
    output_root: Path,
    n_clients: int,
    rounds: int,
    seed: int,
):

    percentage = int(
        round(
            dropout_rate * 100
        )
    )

    output_dir = (
        output_root
        / f"dropout_{percentage}pct"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule_path = (
        output_dir
        / "client_participation.csv"
    )

    schedule.to_csv(
        schedule_path,
        index=False,
    )

    # Round-level summary
    round_summary = (
        schedule
        .groupby("round")
        .agg(
            participating_clients=(
                "participates",
                "sum",
            ),
            dropped_clients=(
                "participates",
                lambda x: int(
                    (~x.astype(bool)).sum()
                ),
            ),
        )
        .reset_index()
    )

    round_summary[
        "dropout_rate"
    ] = dropout_rate

    round_summary.to_csv(
        output_dir
        / "round_participation_summary.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "dropout_rate",
                "dropout_percentage",
                "clients",
                "rounds",
                "seed",
                "dropped_clients_per_round",
                "participating_clients_per_round",
            ],
            "value": [
                dropout_rate,
                percentage,
                n_clients,
                rounds,
                seed,
                number_of_dropped_clients(
                    n_clients,
                    dropout_rate,
                ),
                n_clients
                - number_of_dropped_clients(
                    n_clients,
                    dropout_rate,
                ),
            ],
        }
    )

    config.to_csv(
        output_dir
        / "dropout_config.csv",
        index=False,
    )

    return output_dir


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_rates(
        args.rates
    )

    if args.clients < 2:

        raise ValueError(
            "--clients must be >= 2."
        )

    if args.rounds <= 0:

        raise ValueError(
            "--rounds must be > 0."
        )

    validate_client_directory(
        CLIENT_DIR,
        args.clients,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("CLIENT DROPOUT SIMULATOR")
    print("=" * 78)

    print(
        f"Clients : {args.clients}"
    )

    print(
        f"Rounds  : {args.rounds}"
    )

    print(
        f"Seed    : {args.seed}"
    )

    print(
        f"Rates   : {args.rates}"
    )

    print(
        f"Client source: {CLIENT_DIR.resolve()}"
    )

    print()

    all_summaries = []

    for rate_index, dropout_rate in enumerate(
        args.rates
    ):

        # Use a different deterministic stream for each rate.
        schedule_seed = (
            args.seed
            + (rate_index * 10000)
        )

        schedule = create_schedule(
            n_clients=args.clients,
            rounds=args.rounds,
            dropout_rate=dropout_rate,
            seed=schedule_seed,
        )

        output_dir = save_schedule(
            schedule=schedule,
            dropout_rate=dropout_rate,
            output_root=OUTPUT_ROOT,
            n_clients=args.clients,
            rounds=args.rounds,
            seed=schedule_seed,
        )

        round_summary = (
            schedule
            .groupby("round")["participates"]
            .sum()
        )

        all_summaries.append(
            {
                "dropout_rate": dropout_rate,
                "dropout_percentage": (
                    dropout_rate * 100
                ),
                "average_participating_clients": (
                    float(
                        round_summary.mean()
                    )
                ),
                "minimum_participating_clients": (
                    int(
                        round_summary.min()
                    )
                ),
                "maximum_participating_clients": (
                    int(
                        round_summary.max()
                    )
                ),
            }
        )

        print(
            f"Dropout {dropout_rate * 100:.0f}%:"
        )

        print(
            "  dropped/round:",
            number_of_dropped_clients(
                args.clients,
                dropout_rate,
            )
        )

        print(
            "  participating/round:",
            args.clients
            - number_of_dropped_clients(
                args.clients,
                dropout_rate,
            )
        )

        print(
            "  output:",
            output_dir
        )

        print()

    # Overall summary
    summary = pd.DataFrame(
        all_summaries
    )

    summary.to_csv(
        OUTPUT_ROOT
        / "dropout_experiment_summary.csv",
        index=False,
    )

    print("=" * 78)
    print("DROPOUT SCHEDULES CREATED")
    print("=" * 78)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Output:",
        OUTPUT_ROOT.resolve()
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
