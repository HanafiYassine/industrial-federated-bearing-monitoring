from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATASET = Path("../data/NASA")

file_path = sorted((DATASET / "1st_test").iterdir())[0]

data = np.loadtxt(file_path)

print("File:", file_path.name)
print("Shape:", data.shape)

# Sampling frequency
fs = 20_000

time = np.arange(data.shape[0]) / fs

fig, axes = plt.subplots(8, 1, figsize=(14, 14), sharex=True)

for i, ax in enumerate(axes):
    ax.plot(time, data[:, i])
    ax.set_ylabel(f"CH {i+1}")
    ax.grid(True)

axes[-1].set_xlabel("Time (seconds)")

plt.suptitle("NASA IMS Bearing Dataset - First Measurement")
plt.tight_layout()

plt.savefig("../results/first_measurement_channels.png", dpi=150)
plt.show()