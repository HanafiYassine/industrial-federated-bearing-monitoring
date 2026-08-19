"""
simulate_industrial_client_reliability.py

Create deterministic industrial-client reliability scenarios for the
NASA IMS Dataset-2 federated-learning study.

This simulator does NOT modify the original client CSV files.
It creates a per-round reliability profile that the next FedAvg runner
can consume.

Reliability dimensions:
    1. Availability / persistent failure
    2. Sensor noise
    3. Stale updates
    4. A continuous reliability score

Scenarios created by default:

    clean
        All clients available, no degradation.

    sensor_noise
        A small persistent subset of clients has noisy feature values.

    stale_updates
        A persistent subset of clients produces stale model updates.

    persistent_failure
        A persistent subset of clients becomes unavailable.

    mixed
        Combines persistent failure, noisy clients and stale clients.

The default client population is the existing UNBALANCED NON-IID
partition, because that was our strongest heterogeneous baseline.

Important:
    This simulator is OUR experimental stress-test design. It is
    inspired by the industrial CPS issues discussed by Marfo et al.
    (sensor reliability variation and node failures), but it does not
    claim to reproduce their hidden reliability-generation process.

Inputs:
    ../results/federated/dataset2_clients/noniid/client_01.csv
    ...
    ../results/federated/dataset2_clients/noniid/client_10.csv

Outputs:
    ../results/federated/dataset2_reliability/
        clean/
        sensor_noise/
        stale_updates/
        persistent_failure/
        mixed/

Each scenario contains:
    reliability_schedule.csv
    scenario_config.csv
    client_reliability_summary.csv
    reliability_experiment_summary.csv

The schedule contains one row per client per FL round:

    round
    client_id
    available
    sensor_noise_multiplier
    stale_update
    staleness_rounds
    reliability_score
    scenario
    seed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

CLIENT_DIR = Path(
    "../results/federated/dataset2_clients/noniid"
)

OUTPUT_ROOT = Path(
    "../results/federated/dataset2_reliability"
)

N_CLIENTS = 10
ROUNDS = 15
SEED = 42

# Persistent problematic-client fraction.
DEFAULT_BAD_CLIENT_FRACTION = 0.20

# Sensor-noise multiplier expressed relative to the feature
# standard deviation. 1.0 means noise_std == feature_std.
DEFAULT_NOISE_MULTIPLIER = 0.50

# Probability that a selected stale client sends an update that
# is one round old. The next runner will interpret stale_update.
DEFAULT_STALE_PROBABILITY = 1.0

# Persistent failure starts at this round.
DEFAULT_FAILURE_START = 5

SCENARIOS = [
    "clean",
    "sensor_noise",
    "stale_updates",
    "persistent_failure",
    "mixed",
]


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create industrial client reliability schedules."
    )

    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=SCENARIOS,
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=ROUNDS,
    )

    parser.add_argument(
        "--clients",
        type=int,
        default=N_CLIENTS,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
    )

    parser.add_argument(
        "--bad-client-fraction",
        type=float,
        default=DEFAULT_BAD_CLIENT_FRACTION,
        help="Fraction of clients designated unreliable.",
    )

    parser.add_argument(
        "--noise-multiplier",
        type=float,
        default=DEFAULT_NOISE_MULTIPLIER,
        help="Noise std as a multiple of the feature std.",
    )

    parser.add_argument(
        "--stale-probability",
        type=float,
        default=DEFAULT_STALE_PROBABILITY,
        help="Probability of a stale update for stale clients.",
    )

    parser.add_argument(
        "--failure-start",
        type=int,
        default=DEFAULT_FAILURE_START,
        help="First FL round where persistent failed clients become unavailable.",
    )

    return parser.parse_args()


# ============================================================
# VALIDATION
# ============================================================

def validate_args(args):

    if args.clients < 2:
        raise ValueError("--clients must be >= 2.")

    if args.rounds <= 0:
        raise ValueError("--rounds must be > 0.")

    if not 0.0 < args.bad_client_fraction < 1.0:
        raise ValueError(
            "--bad-client-fraction must be between 0 and 1."
        )

    if args.noise_multiplier < 0:
        raise ValueError(
            "--noise-multiplier must be >= 0."
        )

    if not 0.0 <= args.stale_probability <= 1.0:
        raise ValueError(
            "--stale-probability must be between 0 and 1."
        )

    if not 1 <= args.failure_start <= args.rounds:
        raise ValueError(
            "--failure-start must be inside the FL round range."
        )


def validate_clients(
    client_dir: Path,
    n_clients: int,
):
    if not client_dir.exists():
        raise FileNotFoundError(
            f"Client directory not found:\n{client_dir.resolve()}"
        )

    missing = []

    for client_id in range(1, n_clients + 1):
        path = client_dir / f"client_{client_id:02d}.csv"

        if not path.exists():
            missing.append(path.name)

    if missing:
        raise FileNotFoundError(
            "Missing client files:\n" +
            "\n".join(missing)
        )


# ============================================================
# CLIENT PROFILE
# ============================================================

def load_client_summary(
    client_dir: Path,
    n_clients: int,
):
    rows = []

    for client_id in range(1, n_clients + 1):

        path = (
            client_dir
            / f"client_{client_id:02d}.csv"
        )

        df = pd.read_csv(path)

        if "label" not in df.columns:
            raise ValueError(
                f"{path} does not contain label."
            )

        feature_columns = [
            c for c in df.columns
            if c.startswith("ch")
        ]

        if not feature_columns:
            raise ValueError(
                f"{path} contains no feature columns."
            )

        normal = int(
            (df["label"] == 0).sum()
        )

        anomaly = int(
            (df["label"] == 1).sum()
        )

        rows.append(
            {
                "client_id": client_id,
                "samples": len(df),
                "normal_samples": normal,
                "anomaly_samples": anomaly,
                "anomaly_percentage": (
                    anomaly / len(df)
                    if len(df)
                    else 0.0
                ),
                "feature_count": len(feature_columns),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SELECT UNRELIABLE CLIENTS
# ============================================================

def choose_bad_clients(
    n_clients: int,
    fraction: float,
    rng: np.random.Generator,
):
    count = max(
        1,
        int(
            round(
                n_clients * fraction
            )
        ),
    )

    count = min(
        count,
        n_clients - 1,
    )

    selected = rng.choice(
        np.arange(1, n_clients + 1),
        size=count,
        replace=False,
    )

    return sorted(
        int(x)
        for x in selected
    )


def choose_specialized_clients(
    bad_clients,
    seed,
):
    """
    Split bad clients deterministically into:
        - noisy
        - stale
        - persistent failure

    The mixed scenario uses all three roles.
    """

    if not bad_clients:
        return {
            "noise": [],
            "stale": [],
            "failure": [],
        }

    rng = np.random.default_rng(seed)

    shuffled = list(bad_clients)

    rng.shuffle(shuffled)

    n = len(shuffled)

    # Try to distribute roles across the available unreliable
    # clients. With two clients, each receives a different role;
    # with three or more, use all roles.
    noise = []
    stale = []
    failure = []

    for i, client_id in enumerate(shuffled):

        role = i % 3

        if role == 0:
            noise.append(client_id)

        elif role == 1:
            stale.append(client_id)

        else:
            failure.append(client_id)

    return {
        "noise": sorted(noise),
        "stale": sorted(stale),
        "failure": sorted(failure),
    }


# ============================================================
# SCHEDULE GENERATION
# ============================================================

def generate_schedule(
    scenario: str,
    n_clients: int,
    rounds: int,
    seed: int,
    bad_client_fraction: float,
    noise_multiplier: float,
    stale_probability: float,
    failure_start: int,
):
    master_rng = np.random.default_rng(seed)

    bad_clients = choose_bad_clients(
        n_clients,
        bad_client_fraction,
        master_rng,
    )

    roles = choose_specialized_clients(
        bad_clients,
        seed + 1,
    )

    rows = []

    for round_number in range(
        1,
        rounds + 1,
    ):

        for client_id in range(
            1,
            n_clients + 1,
        ):

            available = True
            sensor_noise_multiplier = 0.0
            stale_update = False
            staleness_rounds = 0

            if scenario == "clean":
                pass

            elif scenario == "sensor_noise":

                if client_id in bad_clients:
                    sensor_noise_multiplier = noise_multiplier

            elif scenario == "stale_updates":

                if client_id in bad_clients:

                    stale_update = (
                        master_rng.random()
                        < stale_probability
                    )

                    if stale_update:
                        staleness_rounds = 1

            elif scenario == "persistent_failure":

                if (
                    client_id in bad_clients
                    and round_number >= failure_start
                ):
                    available = False

            elif scenario == "mixed":

                if client_id in roles["noise"]:
                    sensor_noise_multiplier = (
                        noise_multiplier
                    )

                if client_id in roles["stale"]:

                    stale_update = (
                        master_rng.random()
                        < stale_probability
                    )

                    if stale_update:
                        staleness_rounds = 1

                if (
                    client_id in roles["failure"]
                    and round_number >= failure_start
                ):
                    available = False

            else:
                raise ValueError(
                    f"Unknown scenario: {scenario}"
                )

            # Continuous reliability score used later by
            # the optimization method. It is NOT used by
            # baseline FedAvg.
            score = 1.0

            if not available:
                score *= 0.0

            if sensor_noise_multiplier > 0:
                score *= 1.0 / (
                    1.0
                    + sensor_noise_multiplier
                )

            if stale_update:
                score *= 0.70

            score = float(
                np.clip(
                    score,
                    0.0,
                    1.0,
                )
            )

            rows.append(
                {
                    "scenario": scenario,
                    "round": round_number,
                    "client_id": client_id,
                    "available": bool(available),
                    "sensor_noise_multiplier": (
                        sensor_noise_multiplier
                    ),
                    "stale_update": bool(
                        stale_update
                    ),
                    "staleness_rounds": (
                        staleness_rounds
                    ),
                    "reliability_score": score,
                    "seed": seed,
                }
            )

    schedule = pd.DataFrame(rows)

    return schedule, bad_clients, roles


# ============================================================
# SAVE
# ============================================================

def save_scenario(
    scenario,
    schedule,
    client_summary,
    bad_clients,
    roles,
    args,
):
    output_dir = (
        OUTPUT_ROOT / scenario
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule.to_csv(
        output_dir / "reliability_schedule.csv",
        index=False,
    )

    # Client-level summary.
    client_rows = []

    for client_id in range(
        1,
        args.clients + 1,
    ):

        client_schedule = schedule[
            schedule["client_id"] == client_id
        ]

        client_rows.append(
            {
                "client_id": client_id,
                "samples": int(
                    client_summary.loc[
                        client_summary["client_id"] == client_id,
                        "samples",
                    ].iloc[0]
                ),
                "bad_client": client_id in bad_clients,
                "noise_role": (
                    client_id in roles["noise"]
                ),
                "stale_role": (
                    client_id in roles["stale"]
                ),
                "failure_role": (
                    client_id in roles["failure"]
                ),
                "participating_rounds": int(
                    client_schedule["available"].sum()
                ),
                "availability_rate": float(
                    client_schedule["available"].mean()
                ),
                "mean_reliability_score": float(
                    client_schedule["reliability_score"].mean()
                ),
                "noise_rounds": int(
                    (
                        client_schedule[
                            "sensor_noise_multiplier"
                        ] > 0
                    ).sum()
                ),
                "stale_rounds": int(
                    client_schedule[
                        "stale_update"
                    ].sum()
                ),
                "failed_rounds": int(
                    (
                        ~client_schedule["available"]
                    ).sum()
                ),
            }
        )

    reliability_summary = pd.DataFrame(
        client_rows
    )

    reliability_summary.to_csv(
        output_dir
        / "client_reliability_summary.csv",
        index=False,
    )

    # Round summary.
    round_summary = (
        schedule
        .groupby("round")
        .agg(
            available_clients=(
                "available",
                "sum",
            ),
            unavailable_clients=(
                "available",
                lambda x: int(
                    (~x.astype(bool)).sum()
                ),
            ),
            noisy_clients=(
                "sensor_noise_multiplier",
                lambda x: int(
                    (x > 0).sum()
                ),
            ),
            stale_clients=(
                "stale_update",
                "sum",
            ),
            mean_reliability=(
                "reliability_score",
                "mean",
            ),
        )
        .reset_index()
    )

    round_summary.to_csv(
        output_dir
        / "round_reliability_summary.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "scenario",
                "clients",
                "rounds",
                "seed",
                "bad_client_fraction",
                "bad_clients",
                "noise_multiplier",
                "stale_probability",
                "failure_start",
                "noise_clients",
                "stale_clients",
                "failure_clients",
            ],
            "value": [
                scenario,
                args.clients,
                args.rounds,
                args.seed,
                args.bad_client_fraction,
                ",".join(
                    str(x)
                    for x in bad_clients
                ),
                args.noise_multiplier,
                args.stale_probability,
                args.failure_start,
                ",".join(
                    str(x)
                    for x in roles["noise"]
                ),
                ",".join(
                    str(x)
                    for x in roles["stale"]
                ),
                ",".join(
                    str(x)
                    for x in roles["failure"]
                ),
            ],
        }
    )

    config.to_csv(
        output_dir / "scenario_config.csv",
        index=False,
    )

    return output_dir, reliability_summary, round_summary


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    validate_args(args)

    validate_clients(
        CLIENT_DIR,
        args.clients,
    )

    client_summary = load_client_summary(
        CLIENT_DIR,
        args.clients,
    )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("INDUSTRIAL CLIENT RELIABILITY SIMULATOR")
    print("=" * 78)

    print(
        f"Clients              : {args.clients}"
    )

    print(
        f"Rounds               : {args.rounds}"
    )

    print(
        f"Seed                 : {args.seed}"
    )

    print(
        f"Bad-client fraction  : "
        f"{args.bad_client_fraction:.2f}"
    )

    print(
        f"Noise multiplier     : "
        f"{args.noise_multiplier:.2f}"
    )

    print(
        f"Stale probability    : "
        f"{args.stale_probability:.2f}"
    )

    print(
        f"Failure starts       : "
        f"round {args.failure_start}"
    )

    print()

    overall_rows = []

    for index, scenario in enumerate(
        args.scenarios
    ):

        scenario_seed = (
            args.seed
            + index * 10000
        )

        (
            schedule,
            bad_clients,
            roles,
        ) = generate_schedule(
            scenario=scenario,
            n_clients=args.clients,
            rounds=args.rounds,
            seed=scenario_seed,
            bad_client_fraction=args.bad_client_fraction,
            noise_multiplier=args.noise_multiplier,
            stale_probability=args.stale_probability,
            failure_start=args.failure_start,
        )

        (
            output_dir,
            reliability_summary,
            round_summary,
        ) = save_scenario(
            scenario=scenario,
            schedule=schedule,
            client_summary=client_summary,
            bad_clients=bad_clients,
            roles=roles,
            args=args,
        )

        overall_rows.append(
            {
                "scenario": scenario,
                "bad_clients": len(bad_clients),
                "unavailable_round_events": int(
                    (~schedule["available"]).sum()
                ),
                "mean_available_clients": float(
                    round_summary["available_clients"].mean()
                ),
                "mean_reliability": float(
                    schedule["reliability_score"].mean()
                ),
                "noise_client_rounds": int(
                    (
                        schedule[
                            "sensor_noise_multiplier"
                        ] > 0
                    ).sum()
                ),
                "stale_client_rounds": int(
                    schedule[
                        "stale_update"
                    ].sum()
                ),
            }
        )

        print(
            f"Scenario: {scenario}"
        )

        print(
            "  bad clients:",
            bad_clients,
        )

        print(
            "  noise clients:",
            roles["noise"],
        )

        print(
            "  stale clients:",
            roles["stale"],
        )

        print(
            "  failure clients:",
            roles["failure"],
        )

        print(
            "  mean available clients:",
            f"{round_summary['available_clients'].mean():.2f}",
        )

        print(
            "  mean reliability:",
            f"{schedule['reliability_score'].mean():.3f}",
        )

        print(
            "  output:",
            output_dir,
        )

        print()

    overall = pd.DataFrame(
        overall_rows
    )

    overall.to_csv(
        OUTPUT_ROOT
        / "reliability_experiment_summary.csv",
        index=False,
    )

    print("=" * 78)
    print("RELIABILITY SCENARIOS CREATED")
    print("=" * 78)

    print(
        overall.to_string(
            index=False
        )
    )

    print()
    print(
        "Output:",
        OUTPUT_ROOT.resolve(),
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
