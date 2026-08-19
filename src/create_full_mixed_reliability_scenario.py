"""
create_full_mixed_reliability_scenario.py

Create a deterministic FULL mixed industrial reliability scenario.

Roles:
    - one persistent noisy client
    - one persistent stale-update client
    - one persistent-failure client

Default deterministic assignment:
    noise   -> client 3
    stale   -> client 7
    failure -> client 9

The scenario uses the existing unbalanced Non-IID client partition
and produces a schedule compatible with run_fedavg_reliability.py.

Failure starts at round 5.
Noise multiplier = 0.5.
Stale probability = 1.0.

Output:
    ../results/federated/dataset2_reliability/mixed_full/
        reliability_schedule.csv
        scenario_config.csv
        client_reliability_summary.csv
        round_reliability_summary.csv
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


CLIENT_DIR = Path(
    "../results/federated/dataset2_clients/noniid"
)

OUTPUT_DIR = Path(
    "../results/federated/dataset2_reliability/mixed_full"
)

N_CLIENTS = 10
ROUNDS = 15
SEED = 42

NOISE_CLIENT = 3
STALE_CLIENT = 7
FAILURE_CLIENT = 9

NOISE_MULTIPLIER = 0.5
STALE_PROBABILITY = 1.0
FAILURE_START = 5


def validate_client_files():
    if not CLIENT_DIR.exists():
        raise FileNotFoundError(
            f"Client directory not found:\n{CLIENT_DIR.resolve()}"
        )

    missing = []

    for client_id in range(1, N_CLIENTS + 1):
        path = CLIENT_DIR / f"client_{client_id:02d}.csv"
        if not path.exists():
            missing.append(path.name)

    if missing:
        raise FileNotFoundError(
            "Missing client files:\n" + "\n".join(missing)
        )


def build_schedule():
    rows = []

    for round_number in range(1, ROUNDS + 1):
        for client_id in range(1, N_CLIENTS + 1):

            available = True
            sensor_noise_multiplier = 0.0
            stale_update = False
            staleness_rounds = 0

            # -----------------------------
            # Sensor noise role
            # -----------------------------
            if client_id == NOISE_CLIENT:
                sensor_noise_multiplier = NOISE_MULTIPLIER

            # -----------------------------
            # Stale-update role
            # -----------------------------
            if client_id == STALE_CLIENT:
                stale_update = True
                staleness_rounds = 1

            # -----------------------------
            # Persistent-failure role
            # -----------------------------
            if (
                client_id == FAILURE_CLIENT
                and round_number >= FAILURE_START
            ):
                available = False

            # The reliability score is only a logged signal.
            # Standard FedAvg does not use it.
            score = 1.0

            if not available:
                score = 0.0

            elif sensor_noise_multiplier > 0:
                score *= 1.0 / (
                    1.0 + sensor_noise_multiplier
                )

            if stale_update:
                score *= 0.70

            score = max(0.0, min(1.0, score))

            rows.append(
                {
                    "scenario": "mixed_full",
                    "round": round_number,
                    "client_id": client_id,
                    "available": available,
                    "sensor_noise_multiplier": sensor_noise_multiplier,
                    "stale_update": stale_update,
                    "staleness_rounds": staleness_rounds,
                    "reliability_score": score,
                    "seed": SEED,
                }
            )

    return pd.DataFrame(rows)


def save_outputs(schedule):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    schedule.to_csv(
        OUTPUT_DIR / "reliability_schedule.csv",
        index=False,
    )

    # Client-level summary
    summary_rows = []

    for client_id in range(1, N_CLIENTS + 1):
        client_path = (
            CLIENT_DIR
            / f"client_{client_id:02d}.csv"
        )

        df = pd.read_csv(client_path)

        s = schedule[
            schedule["client_id"] == client_id
        ]

        summary_rows.append(
            {
                "client_id": client_id,
                "samples": len(df),
                "anomaly_samples": int(
                    df["label"].sum()
                ),
                "noise_role": client_id == NOISE_CLIENT,
                "stale_role": client_id == STALE_CLIENT,
                "failure_role": client_id == FAILURE_CLIENT,
                "participating_rounds": int(
                    s["available"].sum()
                ),
                "failed_rounds": int(
                    (~s["available"]).sum()
                ),
                "noise_rounds": int(
                    (s["sensor_noise_multiplier"] > 0).sum()
                ),
                "stale_rounds": int(
                    s["stale_update"].sum()
                ),
                "mean_reliability": float(
                    s["reliability_score"].mean()
                ),
            }
        )

    client_summary = pd.DataFrame(
        summary_rows
    )

    client_summary.to_csv(
        OUTPUT_DIR / "client_reliability_summary.csv",
        index=False,
    )

    # Round-level summary
    round_summary = (
        schedule
        .groupby("round")
        .agg(
            available_clients=(
                "available",
                "sum",
            ),
            failed_clients=(
                "available",
                lambda x: int((~x.astype(bool)).sum()),
            ),
            noisy_clients=(
                "sensor_noise_multiplier",
                lambda x: int((x > 0).sum()),
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
        OUTPUT_DIR / "round_reliability_summary.csv",
        index=False,
    )

    config = pd.DataFrame(
        {
            "parameter": [
                "scenario",
                "clients",
                "rounds",
                "seed",
                "noise_client",
                "stale_client",
                "failure_client",
                "noise_multiplier",
                "stale_probability",
                "failure_start",
            ],
            "value": [
                "mixed_full",
                N_CLIENTS,
                ROUNDS,
                SEED,
                NOISE_CLIENT,
                STALE_CLIENT,
                FAILURE_CLIENT,
                NOISE_MULTIPLIER,
                STALE_PROBABILITY,
                FAILURE_START,
            ],
        }
    )

    config.to_csv(
        OUTPUT_DIR / "scenario_config.csv",
        index=False,
    )

    return client_summary, round_summary


def validate(schedule):
    # Exactly one noise client each round.
    assert (
        schedule[
            "sensor_noise_multiplier"
        ].gt(0).sum()
        == ROUNDS
    )

    # Exactly one stale client each round.
    assert (
        schedule[
            "stale_update"
        ].sum()
        == ROUNDS
    )

    # One failure client, unavailable from round 5 onward.
    expected_failed = (
        ROUNDS - FAILURE_START + 1
    )

    assert (
        (~schedule["available"]).sum()
        == expected_failed
    )

    # Every round has at least nine available clients.
    available_per_round = (
        schedule
        .groupby("round")["available"]
        .sum()
    )

    assert (
        available_per_round.min()
        >= N_CLIENTS - 1
    )


def main():
    validate_client_files()

    schedule = build_schedule()

    validate(schedule)

    client_summary, round_summary = save_outputs(
        schedule
    )

    print("=" * 78)
    print("FULL MIXED INDUSTRIAL RELIABILITY SCENARIO")
    print("=" * 78)

    print("Noise client   :", NOISE_CLIENT)
    print("Stale client   :", STALE_CLIENT)
    print("Failure client :", FAILURE_CLIENT)
    print("Failure starts :", f"round {FAILURE_START}")
    print("Noise multiplier:", NOISE_MULTIPLIER)
    print("Rounds         :", ROUNDS)
    print("Clients        :", N_CLIENTS)

    print()
    print("Client summary:")
    print(
        client_summary.to_string(index=False)
    )

    print()
    print("Round summary:")
    print(
        round_summary.to_string(index=False)
    )

    print()
    print("Output:")
    print(
        OUTPUT_DIR.resolve()
    )

    print()
    print("Validation: PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()
