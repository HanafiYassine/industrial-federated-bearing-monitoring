from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


INPUT = Path("../results/raw_features.csv")
OUTPUT = Path("../results")

df = pd.read_csv(INPUT)

print("=" * 60)
print("Dataset summary")
print("=" * 60)

print("Total recordings:", len(df))
print("Total features:", len(df.columns) - 3)

print("\nRecordings per dataset:")
print(df["dataset"].value_counts())

# ---------------------------------------------------------
# 1. Show the features with the largest variation
# ---------------------------------------------------------

feature_columns = [
    c for c in df.columns
    if c not in ["dataset", "file_index", "timestamp"]
]

std_values = df[feature_columns].std().sort_values(ascending=False)

print("\nTop 20 features by standard deviation:")
print(std_values.head(20).to_string())


# ---------------------------------------------------------
# 2. Plot RMS for all 8 channels
# ---------------------------------------------------------

for dataset_name in ["1st_test", "2nd_test", "3rd_test"]:

    subset = df[df["dataset"] == dataset_name].copy()

    fig, axes = plt.subplots(
        8, 1,
        figsize=(14, 16),
        sharex=True
    )

    for ch in range(1, 9):

        column = f"ch{ch}_rms"

        axes[ch - 1].plot(
            subset["file_index"],
            subset[column],
            linewidth=0.8
        )

        axes[ch - 1].set_ylabel(f"CH {ch}")
        axes[ch - 1].grid(True)

    axes[-1].set_xlabel("Recording index")

    fig.suptitle(
        f"{dataset_name} - RMS evolution"
    )

    plt.tight_layout()

    filename = OUTPUT / f"{dataset_name}_rms.png"

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    print("Saved:", filename)


# ---------------------------------------------------------
# 3. Plot kurtosis for all 8 channels
# ---------------------------------------------------------

for dataset_name in ["1st_test", "2nd_test", "3rd_test"]:

    subset = df[df["dataset"] == dataset_name].copy()

    fig, axes = plt.subplots(
        8, 1,
        figsize=(14, 16),
        sharex=True
    )

    for ch in range(1, 9):

        column = f"ch{ch}_kurtosis"

        axes[ch - 1].plot(
            subset["file_index"],
            subset[column],
            linewidth=0.8
        )

        axes[ch - 1].set_ylabel(f"CH {ch}")
        axes[ch - 1].grid(True)

    axes[-1].set_xlabel("Recording index")

    fig.suptitle(
        f"{dataset_name} - Kurtosis evolution"
    )

    plt.tight_layout()

    filename = OUTPUT / f"{dataset_name}_kurtosis.png"

    plt.savefig(
        filename,
        dpi=150
    )

    plt.close()

    print("Saved:", filename)


print("\nAnalysis complete.")