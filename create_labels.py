from pathlib import Path

import pandas as pd


FEATURE_FILE = Path(
    "../results/federated/dataset2_features.csv"
)

SOM_RESULTS = Path(
    "../results/baseline_som_t2/dataset2_som_results.csv"
)

OUTPUT = Path(
    "../results/federated/dataset2_fl_dataset.csv"
)


def main():

    features = pd.read_csv(
        FEATURE_FILE
    )

    som = pd.read_csv(
        SOM_RESULTS
    )

    merged = features.merge(
        som[
            [
                "file_index",
                "quantization_error"
            ]
        ],
        on="file_index",
        how="inner"
    )

    # Use the frozen SOM threshold.
    baseline = merged[
        merged["file_index"] < 590
    ]

    threshold = (
        baseline["quantization_error"].mean()
        + 3 * baseline["quantization_error"].std(
            ddof=1
        )
    )

    merged["label"] = (
        merged["quantization_error"]
        > threshold
    ).astype(int)

    merged.to_csv(
        OUTPUT,
        index=False
    )

    print("=" * 60)
    print("LABELED FL DATASET")
    print("=" * 60)

    print("Threshold:", threshold)

    print(
        merged["label"]
        .value_counts()
        .sort_index()
    )

    print(
        "Output:",
        OUTPUT
    )


if __name__ == "__main__":
    main()