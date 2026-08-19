"""
baseline_som.py

NASA IMS Bearing Dataset
Baseline SOM reproduction

Current target:
    2nd_test

Pipeline:
    NASA IMS raw vibration data
        ↓
    60% baseline / 40% monitoring
        ↓
    deterministic representative sampling
        ↓
    Min-Max normalization [0, 1]
        ↓
    50 x 50 SOM
        ↓
    Gaussian neighborhood
        ↓
    Quantization Error
        ↓
    μ + 3σ anomaly threshold
        ↓
    anomaly timeline

IMPORTANT:
    NASA IMS datasets do not all have the same number of channels.

    1st_test : 8 channels
    2nd_test : 4 channels
    3rd_test : 4 channels

The script therefore detects the number of channels automatically.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


# ============================================================
# CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

# Current reproduction target
DATASET_DIR = Path("../data/NASA/3rd_test")

# Results
RESULTS_DIR = Path("../results/baseline_som")


# ------------------------------------------------------------
# NASA IMS parameters
# ------------------------------------------------------------

EXPECTED_SAMPLES_PER_FILE = 20480

SAMPLING_RATE = 20_000


# ------------------------------------------------------------
# Baseline / monitoring split
# ------------------------------------------------------------

TRAIN_RATIO = 0.60


# ------------------------------------------------------------
# SOM configuration
# ------------------------------------------------------------

SOM_ROWS = 50
SOM_COLS = 50

N_ITERATIONS = 50

SIGMA = 1.0

LEARNING_RATE = 0.5

RANDOM_SEED = 42


# ------------------------------------------------------------
# Computational control
# ------------------------------------------------------------
#
# The full Dataset 2 baseline contains:
#
#     590 files × 20,480 samples
#     = 12,083,200 samples
#
# Training a 50×50 SOM directly on all of them is expensive.
#
# For the first reproduction run we use a fixed number of
# representative samples from every file.
#
# Later we can run a full-data version.
# ------------------------------------------------------------

SAMPLES_PER_FILE = 100


# ------------------------------------------------------------
# Anomaly threshold
# ------------------------------------------------------------

THRESHOLD_SIGMA = 3.0


# ============================================================
# LOAD ONE NASA FILE
# ============================================================

def load_file(path):
    """
    Load one NASA IMS vibration recording.

    Parameters
    ----------
    path : Path
        Path to NASA recording.

    Returns
    -------
    data : np.ndarray
        Shape:
            (20480, 4) for Dataset 2/3
            (20480, 8) for Dataset 1
    """

    data = np.loadtxt(path)

    if data.ndim != 2:

        raise ValueError(
            f"Unexpected data dimensions in:\n"
            f"{path}\n"
            f"Shape: {data.shape}"
        )

    if data.shape[0] != EXPECTED_SAMPLES_PER_FILE:

        print(
            f"WARNING: {path.name} contains "
            f"{data.shape[0]} samples "
            f"(expected {EXPECTED_SAMPLES_PER_FILE})"
        )

    return data.astype(np.float32)


# ============================================================
# FIND NASA FILES
# ============================================================

def get_files():

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{DATASET_DIR.resolve()}"
        )

    files = sorted(
        [
            f
            for f in DATASET_DIR.iterdir()
            if f.is_file()
        ]
    )

    if len(files) == 0:

        raise RuntimeError(
            f"No files found in:\n"
            f"{DATASET_DIR.resolve()}"
        )

    return files


# ============================================================
# DETECT DATASET STRUCTURE
# ============================================================

def inspect_dataset(files):

    print()
    print("=" * 70)
    print("DATASET INSPECTION")
    print("=" * 70)

    first_file = files[0]

    first_data = load_file(first_file)

    n_samples = first_data.shape[0]
    n_channels = first_data.shape[1]

    print("First file:")
    print(first_file.name)

    print()
    print("Samples per file :", n_samples)
    print("Channels         :", n_channels)

    # Check a few files to make sure the dimensionality
    # is consistent.

    check_count = min(10, len(files))

    for path in files[:check_count]:

        data = load_file(path)

        if data.shape[1] != n_channels:

            raise ValueError(
                "Inconsistent number of channels detected.\n"
                f"{first_file.name}: {n_channels}\n"
                f"{path.name}: {data.shape[1]}"
            )

    print()
    print("Channel structure verified.")

    return n_samples, n_channels


# ============================================================
# COLLECT TRAINING SAMPLES
# ============================================================

def collect_training_samples(
    files,
    train_count,
    n_channels
):

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    samples = []

    print()
    print("=" * 70)
    print("COLLECTING SOM TRAINING SAMPLES")
    print("=" * 70)

    print(
        f"Training files     : {train_count}"
    )

    print(
        f"Samples per file   : {SAMPLES_PER_FILE}"
    )

    print(
        f"Input dimensions   : {n_channels}"
    )

    print()

    for i, path in enumerate(
        files[:train_count]
    ):

        data = load_file(path)

        if data.shape[1] != n_channels:

            raise ValueError(
                f"Channel mismatch in {path.name}"
            )

        # ----------------------------------------------------
        # Select samples
        # ----------------------------------------------------

        if SAMPLES_PER_FILE is None:

            selected = data

        else:

            sample_count = min(
                SAMPLES_PER_FILE,
                len(data)
            )

            indices = rng.choice(
                len(data),
                size=sample_count,
                replace=False
            )

            selected = data[indices]

        samples.append(selected)

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            (i + 1) % 50 == 0
            or i == 0
            or i + 1 == train_count
        ):

            print(
                f"Processed "
                f"{i + 1:4d}/{train_count} files"
            )

    samples = np.vstack(samples)

    print()
    print("Training matrix:")
    print(
        f"Shape: {samples.shape}"
    )

    print(
        f"Samples: {samples.shape[0]}"
    )

    print(
        f"Features/channels: {samples.shape[1]}"
    )

    return samples


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_training_data(
    training_samples
):

    print()
    print("=" * 70)
    print("MIN-MAX NORMALIZATION")
    print("=" * 70)

    scaler = MinMaxScaler(
        feature_range=(0.0, 1.0)
    )

    normalized = scaler.fit_transform(
        training_samples
    )

    normalized = normalized.astype(
        np.float32
    )

    print(
        "Normalized shape:",
        normalized.shape
    )

    print(
        "Minimum:",
        normalized.min(axis=0)
    )

    print(
        "Maximum:",
        normalized.max(axis=0)
    )

    return normalized, scaler


# ============================================================
# TRAIN SOM
# ============================================================

def train_som(
    training_data,
    n_channels
):

    print()
    print("=" * 70)
    print("TRAINING SOM")
    print("=" * 70)

    print(
        f"SOM size          : "
        f"{SOM_ROWS} x {SOM_COLS}"
    )

    print(
        f"Number of neurons : "
        f"{SOM_ROWS * SOM_COLS}"
    )

    print(
        f"Input dimension   : {n_channels}"
    )

    print(
        f"Iterations        : {N_ITERATIONS}"
    )

    print(
        f"Learning rate     : {LEARNING_RATE}"
    )

    print(
        f"Gaussian sigma    : {SIGMA}"
    )

    print(
        f"Random seed       : {RANDOM_SEED}"
    )

    print()

    som = MiniSom(
        x=SOM_ROWS,
        y=SOM_COLS,
        input_len=n_channels,
        sigma=SIGMA,
        learning_rate=LEARNING_RATE,
        neighborhood_function="gaussian",
        random_seed=RANDOM_SEED
    )

    # PCA initialization gives a stable starting
    # configuration for the SOM.

    print("Initializing SOM with PCA...")

    som.pca_weights_init(
        training_data
    )

    print("Starting SOM training...")

    som.train_random(
        training_data,
        N_ITERATIONS,
        verbose=True
    )

    print()
    print("SOM training completed.")

    return som


# ============================================================
# QUANTIZATION ERROR
# ============================================================

def calculate_file_q_error(
    path,
    scaler,
    som
):

    data = load_file(path)

    normalized = scaler.transform(
        data
    ).astype(np.float32)

    # Find BMU for each sample
    bmus = np.array(
        [
            som.winner(x)
            for x in normalized
        ]
    )

    # Get corresponding neuron weights
    weights = som.get_weights()

    quantized = weights[
        bmus[:, 0],
        bmus[:, 1]
    ]

    # Euclidean distance between
    # sample and BMU weight

    errors = np.linalg.norm(
        normalized - quantized,
        axis=1
    )

    return float(
        np.mean(errors)
    )


# ============================================================
# CALCULATE ALL Q-ERRORS
# ============================================================

def calculate_quantization_errors(
    files,
    train_count,
    scaler,
    som
):

    print()
    print("=" * 70)
    print("CALCULATING QUANTIZATION ERRORS")
    print("=" * 70)

    results = []

    total = len(files)

    for i, path in enumerate(files):

        q_error = calculate_file_q_error(
            path,
            scaler,
            som
        )

        results.append(
            {
                "file_index": i,

                "file": path.name,

                "timestamp": path.name,

                "quantization_error": q_error,

                "is_baseline": (
                    i < train_count
                )
            }
        )

        if (
            (i + 1) % 25 == 0
            or i == 0
            or i + 1 == total
        ):

            print(
                f"Processed "
                f"{i + 1:4d}/{total} files"
            )

    return pd.DataFrame(results)


# ============================================================
# CALCULATE ANOMALY THRESHOLD
# ============================================================

def calculate_threshold(
    results
):

    baseline_errors = (
        results[
            results["is_baseline"]
        ]["quantization_error"]
        .to_numpy()
    )

    mean_error = np.mean(
        baseline_errors
    )

    std_error = np.std(
        baseline_errors,
        ddof=1
    )

    threshold = (
        mean_error
        + THRESHOLD_SIGMA * std_error
    )

    return (
        mean_error,
        std_error,
        threshold
    )


# ============================================================
# DETECT ANOMALIES
# ============================================================

def detect_anomalies(
    results,
    threshold
):

    results = results.copy()

    results["anomaly"] = (
        results["quantization_error"]
        > threshold
    )

    results["anomaly_score"] = (
        results["quantization_error"]
        / threshold
    )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results,
    mean_error,
    std_error,
    threshold,
    n_channels
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Main results
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "dataset2_som_results.csv"
    )

    results.to_csv(
        results_path,
        index=False
    )

    # --------------------------------------------------------
    # Experiment configuration
    # --------------------------------------------------------

    statistics = pd.DataFrame(
        {
            "parameter": [
                "dataset",
                "total_files",
                "baseline_files",
                "monitoring_files",
                "channels",
                "samples_per_file",
                "sampling_rate_hz",
                "som_rows",
                "som_cols",
                "som_neurons",
                "iterations",
                "learning_rate",
                "sigma",
                "threshold_sigma",
                "baseline_mean_q_error",
                "baseline_std_q_error",
                "threshold",
                "samples_per_file_used"
            ],

            "value": [
                "2nd_test",
                len(results),
                int(
                    results[
                        "is_baseline"
                    ].sum()
                ),
                int(
                    (~results[
                        "is_baseline"
                    ]).sum()
                ),
                n_channels,
                EXPECTED_SAMPLES_PER_FILE,
                SAMPLING_RATE,
                SOM_ROWS,
                SOM_COLS,
                SOM_ROWS * SOM_COLS,
                N_ITERATIONS,
                LEARNING_RATE,
                SIGMA,
                THRESHOLD_SIGMA,
                mean_error,
                std_error,
                threshold,
                SAMPLES_PER_FILE
            ]
        }
    )

    statistics_path = (
        RESULTS_DIR
        / "som_statistics.csv"
    )

    statistics.to_csv(
        statistics_path,
        index=False
    )

    print()
    print("Results saved:")
    print(
        results_path.resolve()
    )

    print(
        statistics_path.resolve()
    )


# ============================================================
# PLOT QUANTIZATION ERROR
# ============================================================

def plot_results(
    results,
    mean_error,
    threshold
):

    plt.figure(
        figsize=(16, 7)
    )

    # --------------------------------------------------------
    # Q-error
    # --------------------------------------------------------

    plt.plot(
        results["file_index"],
        results["quantization_error"],
        linewidth=0.8,
        label="Quantization error"
    )

    # --------------------------------------------------------
    # Baseline mean
    # --------------------------------------------------------

    plt.axhline(
        mean_error,
        linestyle="--",
        label="Baseline mean"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    plt.axhline(
        threshold,
        linestyle="--",
        label="μ + 3σ threshold"
    )

    # --------------------------------------------------------
    # Baseline boundary
    # --------------------------------------------------------

    baseline_end = (
        results[
            results["is_baseline"]
        ]["file_index"]
        .max()
    )

    plt.axvline(
        baseline_end,
        linestyle=":",
        label="60% baseline boundary"
    )

    # --------------------------------------------------------
    # Anomalies
    # --------------------------------------------------------

    anomalies = results[
        results["anomaly"]
    ]

    plt.scatter(
        anomalies["file_index"],
        anomalies["quantization_error"],
        s=12,
        label="Detected anomaly"
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    plt.xlabel(
        "Recording index"
    )

    plt.ylabel(
        "Quantization error"
    )

    plt.title(
        "NASA IMS Dataset 2 - "
        "SOM Quantization Error"
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    output_path = (
        RESULTS_DIR
        / "dataset2_quantization_error.png"
    )

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()

    print()
    print(
        "Plot saved:"
    )

    print(
        output_path.resolve()
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    results,
    mean_error,
    std_error,
    threshold
):

    anomaly_count = int(
        results["anomaly"].sum()
    )

    total = len(results)

    monitoring = results[
        ~results["is_baseline"]
    ]

    monitoring_anomalies = int(
        monitoring["anomaly"].sum()
    )

    print()
    print("=" * 70)
    print("BASELINE SOM SUMMARY")
    print("=" * 70)

    print(
        f"Total recordings       : {total}"
    )

    print(
        f"Baseline recordings    : "
        f"{results['is_baseline'].sum()}"
    )

    print(
        f"Monitoring recordings  : "
        f"{(~results['is_baseline']).sum()}"
    )

    print()

    print(
        f"Mean Q-error            : "
        f"{mean_error:.8f}"
    )

    print(
        f"Std Q-error             : "
        f"{std_error:.8f}"
    )

    print(
        f"Anomaly threshold       : "
        f"{threshold:.8f}"
    )

    print()

    print(
        f"All detected anomalies : "
        f"{anomaly_count}"
    )

    print(
        f"Monitoring anomalies   : "
        f"{monitoring_anomalies}"
    )

    print(
        f"Anomaly percentage     : "
        f"{100 * anomaly_count / total:.2f}%"
    )

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NASA IMS BEARING DATASET")
    print("BASELINE SOM REPRODUCTION")
    print("=" * 70)

    print()
    print(
        "Dataset directory:"
    )

    print(
        DATASET_DIR.resolve()
    )

    # --------------------------------------------------------
    # Get files
    # --------------------------------------------------------

    files = get_files()

    print()
    print(
        "Total recordings found:",
        len(files)
    )

    # --------------------------------------------------------
    # Inspect dataset
    # --------------------------------------------------------

    (
        n_samples,
        n_channels
    ) = inspect_dataset(
        files
    )

    # --------------------------------------------------------
    # Baseline / monitoring split
    # --------------------------------------------------------

    total_files = len(files)

    train_count = int(
        TRAIN_RATIO * total_files
    )

    monitoring_count = (
        total_files - train_count
    )

    print()
    print("=" * 70)
    print("EXPERIMENT SPLIT")
    print("=" * 70)

    print(
        f"Total files      : {total_files}"
    )

    print(
        f"Baseline (60%)   : {train_count}"
    )

    print(
        f"Monitoring (40%) : {monitoring_count}"
    )

    # --------------------------------------------------------
    # Collect training samples
    # --------------------------------------------------------

    training_samples = (
        collect_training_samples(
            files,
            train_count,
            n_channels
        )
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    (
        training_data,
        scaler
    ) = normalize_training_data(
        training_samples
    )

    # --------------------------------------------------------
    # Train SOM
    # --------------------------------------------------------

    som = train_som(
        training_data,
        n_channels
    )

    # --------------------------------------------------------
    # Quantization errors
    # --------------------------------------------------------

    results = (
        calculate_quantization_errors(
            files,
            train_count,
            scaler,
            som
        )
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    (
        mean_error,
        std_error,
        threshold
    ) = calculate_threshold(
        results
    )

    # --------------------------------------------------------
    # Anomaly detection
    # --------------------------------------------------------

    results = detect_anomalies(
        results,
        threshold
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        results,
        mean_error,
        std_error,
        threshold,
        n_channels
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    plot_results(
        results,
        mean_error,
        threshold
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print_summary(
        results,
        mean_error,
        std_error,
        threshold
    )

    print()
    print("=" * 70)
    print("BASELINE SOM COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()