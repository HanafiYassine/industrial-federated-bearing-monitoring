# Experimental Results

Generated experiment outputs are intentionally excluded from Git tracking by the repository `.gitignore` because they can become large and machine-specific.

The main reproducible result groups are:

```text
results/
├── raw_features.csv
├── baseline_som/
├── federated/
│   ├── dataset2_features.csv
│   ├── dataset2_fl_dataset.csv
│   ├── dataset2_clients/
│   ├── dataset2_fedavg/
│   ├── dataset2_dropout/
│   ├── dataset2_fedavg_dropout/
│   ├── dataset2_reliability/
│   ├── dataset2_fedavg_reliability/
│   └── dataset2_ra_fedavg/
└── ...
```

For the current research release, compact result tables and publication figures should be copied into dedicated versioned locations when needed rather than committing entire raw experiment directories.

The README and source scripts document how to regenerate these outputs from the NASA IMS dataset.
