from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("../results/raw_features.csv")
OUTPUT = Path("../results/feature_ranking.csv")

df = pd.read_csv(INPUT)

feature_columns = [
    c for c in df.columns
    if c not in [
        "dataset",
        "file_index",
        "timestamp"
    ]
]

results = []

for dataset_name in ["1st_test", "2nd_test", "3rd_test"]:

    data = df[df["dataset"] == dataset_name].copy()

    n = len(data)

    # First 60% = normal baseline according to the paper
    split = int(0.60 * n)

    baseline = data.iloc[:split]
    later = data.iloc[split:]

    print()
    print("=" * 70)
    print(dataset_name)
    print("Total:", n)
    print("Baseline:", split)
    print("Later:", n - split)

    for feature in feature_columns:

        x = data[feature].to_numpy()

        # Correlation with time
        time_index = np.arange(n)

        correlation = np.corrcoef(
            time_index,
            x
        )[0, 1]

        # Baseline statistics
        baseline_mean = baseline[feature].mean()
        baseline_std = baseline[feature].std()

        later_mean = later[feature].mean()

        # Standardized change
        change = abs(
            later_mean - baseline_mean
        ) / (baseline_std + 1e-12)

        results.append({
            "dataset": dataset_name,
            "feature": feature,
            "abs_time_correlation": abs(correlation),
            "standardized_change": change,
        })


ranking = pd.DataFrame(results)

# Combined score
ranking["score"] = (
    ranking["abs_time_correlation"]
    * ranking["standardized_change"]
)

ranking = ranking.sort_values(
    ["dataset", "score"],
    ascending=[True, False]
)

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

ranking.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print("TOP FEATURES")
print("=" * 70)

for dataset_name in [
    "1st_test",
    "2nd_test",
    "3rd_test"
]:

    print()
    print(dataset_name)

    subset = ranking[
        ranking["dataset"] == dataset_name
    ]

    print(
        subset.head(20).to_string(
            index=False
        )
    )

print()
print("Saved:", OUTPUT)