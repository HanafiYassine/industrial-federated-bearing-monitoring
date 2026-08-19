# NASA IMS Bearing Dataset

This directory contains the documentation for the public NASA IMS Bearing Dataset used by this research project.

## Dataset source

The IMS Bearing Dataset is a public bearing-experiment dataset provided by the Center for Intelligent Maintenance Systems (IMS), University of Cincinnati. NASA currently hosts the dataset through the NASA Open Data Portal as **IMS Bearings**, with the downloadable resource provided as `IMS.zip`.

Official source:

https://data.nasa.gov/dataset/ims-bearings

The NASA Open Data Portal identifies the dataset as public and describes it as experiments on bearings provided by IMS, University of Cincinnati.

## Do not commit the dataset to GitHub

The raw dataset is large (more than 2 GB in the prepared research workspace), so it is intentionally **not included in this repository**.

Download the dataset separately and place the extracted files in the local project directory described below.

## Expected directory structure

After downloading and extracting the dataset, the project should contain:

```text
industrial-federated-bearing-monitoring/
├── data/
│   ├── README.md
│   └── NASA/
│       ├── 1st_test/
│       ├── 2nd_test/
│       └── 3rd_test/
├── src/
├── results/
└── ...
```

Each of the three test folders should contain the original NASA IMS recording files. The filenames are timestamp-like strings, for example:

```text
2003.10.22.12.24.13
2003.10.22.12.29.13
...
```

## Dataset organization used in this project

The prepared workspace contains:

| Dataset | Recordings | Channel structure used by the project |
|---|---:|---:|
| `1st_test` | 786 | 8 channels |
| `2nd_test` | 984 | 4 channels |
| `3rd_test` | 6324 | 4 channels |
| **Total** | **8094** | — |

Each recording contains 20,480 vibration samples in the data files processed by this project.

The preprocessing code detects the channel count rather than assuming one fixed number of channels across all three runs.

## Research pipeline using the dataset

The repository processes the raw recordings in the following order:

```text
NASA IMS raw recordings
        |
        v
Per-recording statistical feature extraction
        |
        v
Centralized SOM anomaly-detection baseline
        |
        v
Dataset-2 federated feature table
        |
        v
SOM-derived anomaly labels
        |
        v
IID / Non-IID client partitions
        |
        v
FedAvg and industrial reliability stress tests
```

## Feature extraction

The current feature-extraction pipeline computes the following time-domain statistics independently for each available channel:

- mean
- standard deviation
- RMS
- minimum
- maximum
- peak-to-peak value
- kurtosis
- skewness
- crest factor

The script `src/extract_raw_features.py` detects the number of channels from each recording and produces a combined feature table.

## Centralized SOM experiment

The SOM baseline uses the recordings in temporal order and divides each run into:

```text
60% baseline period
40% monitoring period
```

The current SOM implementation uses a 50 x 50 map, deterministic sampling, Min-Max normalization, and a `mean + 3*std` baseline quantization-error threshold.

The implementation is in:

```text
src/baseline_som.py
```

## Federated-learning experiment

The main federated-learning experiments currently use `2nd_test` because it provides a consistent four-channel representation for the controlled Dataset-2 study.

For each of the four channels, nine time-domain features are used:

```text
mean
std
rms
min
max
peak_to_peak
kurtosis
skewness
crest_factor
```

This produces:

```text
4 channels x 9 features = 36 features per recording
```

The anomaly labels used by the current FL classification experiment are derived from the frozen SOM threshold. These labels should therefore be understood as **SOM-derived surrogate anomaly labels**, not independently verified physical failure labels.

## Reproducing the preprocessing

From the repository root:

```powershell
python src/extract_raw_features.py
python src/analyze_features.py
python src/analyze_frequency.py
python src/baseline_som.py
```

Then create the federated Dataset-2 table and labels:

```powershell
python src/create_fl_dataset.py
python src/create_labels.py
```

The client partitions can then be created with:

```powershell
python src/create_federated_clients.py
python src/create_balanced_noniid.py
```

## Data provenance and citation

The dataset used in this project originates from the IMS Center, University of Cincinnati, and is distributed through the NASA Open Data Portal.

When publishing results based on these data, cite the original dataset/reference requested by the dataset provider and cite the scientific papers that use or describe the IMS bearing experiments as appropriate.

For the reproducible software used in this project, cite the GitHub repository release using the information in `CITATION.cff`. A future archived release DOI should be preferred once the repository is deposited with Zenodo.

## License / usage

The NASA Open Data Portal currently lists the dataset with an `other-license-specified` license designation. Users should review the source dataset's current terms before redistribution or commercial use.

This repository therefore does **not** redistribute the raw NASA IMS recordings. It provides code, documentation, configurations, and research outputs needed to reproduce the analysis after the user obtains the dataset from the official source.

## Official links

- NASA IMS Bearings dataset: https://data.nasa.gov/dataset/ims-bearings
- NASA Open Data Portal: https://data.nasa.gov/
