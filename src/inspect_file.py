from pathlib import Path
import numpy as np

DATASET = Path("../data/NASA")

# Take the first file from 1st_test
file_path = sorted((DATASET / "1st_test").iterdir())[0]

print("=" * 60)
print("File:", file_path)
print("Size:", file_path.stat().st_size, "bytes")

# NASA IMS files are whitespace-separated text files
data = np.loadtxt(file_path)

print("=" * 60)
print("Shape:", data.shape)
print("Number of samples:", data.shape[0])
print("Number of columns:", data.shape[1])

print("\nFirst 5 rows:")
print(data[:5])

print("\nLast 5 rows:")
print(data[-5:])

print("\nData type:", data.dtype)

print("\nMin:")
print(data.min(axis=0))

print("\nMax:")
print(data.max(axis=0))

print("\nMean:")
print(data.mean(axis=0))

print("\nStd:")
print(data.std(axis=0))