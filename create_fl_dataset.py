from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew


NASA_DIR = Path("../data/NASA")
RESULT_DIR = Path("../results/federated")


FEATURES = [
    "mean",
    "std",
    "rms",
    "min",
    "max",
    "peak_to_peak",
    "kurtosis",
    "skewness",
    "crest_factor",
]


def extract_channel_features(x):
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))

    return {
        "mean": np.mean(x),
        "std": np.std(x),
        "rms": rms,
        "min": np.min(x),
        "max": np.max(x),
        "peak_to_peak": np.ptp(x),
        "kurtosis": kurtosis(x),
        "skewness": skew(x),
        "crest_factor": peak / (rms + 1e-12),
    }


def extract_recording_features(data):
    result = {}

    for channel in range(data.shape[1]):
        values = extract_channel_features(data[:, channel])

        for name, value in values.items():
            result[f"ch{channel + 1}_{name}"] = value

    return result


def process_dataset(dataset_name):

    dataset_dir = NASA_DIR / dataset_name

    files = sorted(
        f for f in dataset_dir.iterdir()
        if f.is_file()
    )

    rows = []

    for index, file in enumerate(files):

        data = np.loadtxt(file)

        features = extract_recording_features(data)

        features["dataset"] = dataset_name
        features["file_index"] = index
        features["timestamp"] = file.name

        rows.append(features)

        if (
            index == 0
            or (index + 1) % 100 == 0
            or index + 1 == len(files)
        ):
            print(
                f"{dataset_name}: "
                f"{index + 1}/{len(files)}"
            )

    return pd.DataFrame(rows)


def main():

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Start with Dataset 2.
    df = process_dataset("2nd_test")

    output = (
        RESULT_DIR
        / "dataset2_features.csv"
    )

    df.to_csv(
        output,
        index=False
    )

    print()
    print("=" * 60)
    print("FL DATASET CREATED")
    print("=" * 60)
    print("Shape:", df.shape)
    print("Output:", output)


if __name__ == "__main__":
    main()