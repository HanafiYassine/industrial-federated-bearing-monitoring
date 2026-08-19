from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


DATASET = Path("../data/NASA")


def spectrum(x, fs):
    """
    Calculate one-sided amplitude spectrum.
    """
    n = len(x)

    x = x - np.mean(x)

    fft = np.fft.rfft(x)

    amplitude = np.abs(fft) / n

    frequencies = np.fft.rfftfreq(
        n,
        d=1 / fs
    )

    return frequencies, amplitude


# ---------------------------------------------------------
# Compare an early and late recording from each dataset
# ---------------------------------------------------------

fs = 20_000

for dataset_name in [
    "1st_test",
    "2nd_test",
    "3rd_test"
]:

    folder = DATASET / dataset_name

    files = sorted(
        [
            f for f in folder.iterdir()
            if f.is_file()
        ]
    )

    early_file = files[0]
    late_file = files[-1]

    early = np.loadtxt(early_file)
    late = np.loadtxt(late_file)

    print()
    print("=" * 60)
    print(dataset_name)
    print("Early:", early_file.name)
    print("Late :", late_file.name)

    # Compare channel 1
    ch = 0

    f_early, a_early = spectrum(
        early[:, ch],
        fs
    )

    f_late, a_late = spectrum(
        late[:, ch],
        fs
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(14, 9)
    )

    # Time domain
    axes[0].plot(
        np.arange(len(early[:, ch])) / fs,
        early[:, ch],
        linewidth=0.6
    )

    axes[0].set_title(
        f"{dataset_name} - Early recording - CH1"
    )

    axes[0].set_xlabel("Time (seconds)")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Frequency domain
    axes[1].plot(
        f_early,
        a_early,
        label="Early"
    )

    axes[1].plot(
        f_late,
        a_late,
        label="Late"
    )

    axes[1].set_xlim(0, 10_000)

    axes[1].set_title(
        f"{dataset_name} - CH1 Frequency Spectrum"
    )

    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    output = Path(
        "../results/"
        f"{dataset_name}_frequency_comparison.png"
    )

    plt.savefig(
        output,
        dpi=150
    )

    plt.close()

    print("Saved:", output)