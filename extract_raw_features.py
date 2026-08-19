from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

DATASET = Path("../data/NASA")
OUTPUT = Path("../results/raw_features.csv")


def extract_features(data):
    features = {}

    for ch in range(data.shape[1]):
        x = data[:, ch]

        rms = np.sqrt(np.mean(x ** 2))
        peak = np.max(np.abs(x))

        values = {
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

        for name, value in values.items():
            features[f"ch{ch+1}_{name}"] = value

    return features


rows = []

for test_folder in ["1st_test", "2nd_test", "3rd_test"]:

    folder = DATASET / test_folder

    files = sorted(
        [f for f in folder.iterdir() if f.is_file()]
    )

    print(f"{test_folder}: {len(files)} files")

    for i, file_path in enumerate(files):

        data = np.loadtxt(file_path)

        features = extract_features(data)

        features["dataset"] = test_folder
        features["file_index"] = i
        features["timestamp"] = file_path.name

        rows.append(features)

        if (i + 1) % 100 == 0:
            print(f"  processed {i + 1}/{len(files)}")


df = pd.DataFrame(rows)

# Put identification columns first
id_columns = [
    "dataset",
    "file_index",
    "timestamp",
]

feature_columns = [
    c for c in df.columns
    if c not in id_columns
]

df = df[id_columns + feature_columns]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(OUTPUT, index=False)

print()
print("=" * 60)
print("Finished")
print("Rows:", len(df))
print("Features:", len(feature_columns))
print("Output:", OUTPUT)
print("=" * 60)