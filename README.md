# Federated Learning for Industrial Bearing Condition Monitoring

Reproducible research code for federated learning, anomaly detection, and industrial client-reliability experiments using the NASA IMS Bearing Dataset.

## Research pipeline

```text
NASA IMS raw vibration data
        |
        v
Feature extraction -> centralized SOM
        |
        v
SOM-derived anomaly labels
        |
        v
Dataset-2 federated dataset
        |
        +---------------------------+
        |                           |
        v                           v
IID / Non-IID clients       Reliability stress tests
        |                           |
        +------------+--------------+
                     v
                   FedAvg
                     |
                     v
          Reliability-aware FedAvg v1
                     |
                     v
       Future optimized reliability-aware FL
```

## Completed experiments

- NASA IMS `1st_test`, `2nd_test`, and `3rd_test` preprocessing;
- centralized 50x50 SOM anomaly detection with a `mu + 3 sigma` threshold;
- Dataset-2 federated feature construction and SOM-derived labels;
- centralized MLP baseline;
- IID, balanced non-IID, and unbalanced non-IID client partitions;
- FedAvg with 10 clients and 15 rounds;
- five-seed FedAvg validation with seeds `42, 123, 2024, 3407, 7777`;
- 20%, 40%, and 50% random client dropout, each validated across the same five seeds;
- sensor noise, stale updates, persistent failure, noise+stale, and deterministic full-mixed reliability scenarios, each validated across the same five seeds;
- preliminary reliability-aware FedAvg aggregation.

## Main five-seed Dataset-2 results

Clean unbalanced non-IID FedAvg, final round, mean +/- standard deviation over five seeds:

- Accuracy: `0.861929 +/- 0.059997`
- Recall: `0.531034 +/- 0.203784`
- F1: `0.671892 +/- 0.205383`
- ROC-AUC: `0.969834 +/- 0.015358`

Full-mixed reliability FedAvg:

- Accuracy: `0.777665 +/- 0.054008`
- Recall: `0.244828 +/- 0.183440`
- F1: `0.364955 +/- 0.241667`
- ROC-AUC: `0.947680 +/- 0.023187`

Relative to the clean unbalanced non-IID baseline, full-mixed reliability produces mean decreases of approximately:

- Accuracy: `8.43` percentage points
- Recall: `28.62` percentage points
- F1: `30.69` percentage points
- ROC-AUC: `2.22` percentage points

The full-mixed degradation occurs for all five training seeds in the paired comparison.

### Dropout validation

Final-round mean +/- standard deviation over five seeds:

| Dropout | Accuracy | Recall | F1 | ROC-AUC | Communication |
|---|---:|---:|---:|---:|---:|
| 0% | 0.8619 +/- 0.0600 | 0.5310 +/- 0.2038 | 0.6719 +/- 0.2054 | 0.9698 +/- 0.0154 | 17.324 MiB |
| 20% | 0.8548 +/- 0.0633 | 0.5069 +/- 0.2149 | 0.6473 +/- 0.2237 | 0.9679 +/- 0.0160 | 13.859 MiB |
| 40% | 0.8609 +/- 0.0706 | 0.5276 +/- 0.2397 | 0.6588 +/- 0.2551 | 0.9695 +/- 0.0157 | 10.394 MiB |
| 50% | 0.8284 +/- 0.0624 | 0.4172 +/- 0.2120 | 0.5588 +/- 0.2520 | 0.9640 +/- 0.0182 | 8.662 MiB |

## Reliability stress results

Final-round mean +/- standard deviation over five seeds:

| Scenario | Accuracy | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|
| Clean Unbalanced Non-IID | 0.8619 +/- 0.0600 | 0.5310 +/- 0.2038 | 0.6719 +/- 0.2054 | 0.9698 +/- 0.0154 |
| Sensor noise | 0.8619 +/- 0.0600 | 0.5310 +/- 0.2038 | 0.6719 +/- 0.2054 | 0.9696 +/- 0.0155 |
| Stale updates | 0.8619 +/- 0.0655 | 0.5310 +/- 0.2226 | 0.6670 +/- 0.2303 | 0.9688 +/- 0.0160 |
| Persistent failure | 0.9180 +/- 0.0689 | 0.7793 +/- 0.2680 | 0.8260 +/- 0.1972 | 0.9791 +/- 0.0116 |
| Mixed | 0.8731 +/- 0.0870 | 0.5897 +/- 0.3124 | 0.6813 +/- 0.3172 | 0.9593 +/- 0.0188 |
| Full mixed | 0.7767 +/- 0.0540 | 0.2448 +/- 0.1834 | 0.3650 +/- 0.2417 | 0.9477 +/- 0.0232 |

Sensor noise and stale updates have little effect on final thresholded classification under the tested schedules. The full-mixed condition is substantially more damaging, especially for recall and F1.

## Preliminary RA-FedAvg v1

The first reliability-aware aggregation rule is intentionally retained as a negative-result baseline. It did not outperform standard FedAvg on the original full-mixed run.

Original seed-42 comparison:

- FedAvg Accuracy: `0.837563`
- FedAvg Recall: `0.448276`
- FedAvg F1: `0.619048`
- FedAvg ROC-AUC: `0.934384`

- RA-FedAvg v1 Accuracy: `0.822335`
- RA-FedAvg v1 Recall: `0.396552`
- RA-FedAvg v1 F1: `0.567901`
- RA-FedAvg v1 ROC-AUC: `0.933887`

RA-FedAvg v1 did not outperform standard FedAvg. This negative result is intentionally retained as part of the research record.

## Repository structure

```text
industrial-federated-bearing-monitoring/
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── src/
│   ├── baseline_som.py
│   ├── analyze_features.py
│   ├── analyze_frequency.py
│   ├── extract_raw_features.py
│   ├── create_fl_dataset.py
│   ├── create_labels.py
│   ├── create_federated_clients.py
│   ├── create_balanced_noniid.py
│   ├── centralized_mlp.py
│   ├── federated_model.py
│   ├── fedavg.py
│   ├── run_fedavg.py
│   ├── run_fedavg_multiseed.py
│   ├── evaluate_federated_baselines.py
│   ├── simulate_client_dropout.py
│   ├── run_fedavg_dropout.py
│   ├── run_fedavg_dropout_multiseed.py
│   ├── simulate_industrial_client_reliability.py
│   ├── create_full_mixed_reliability_scenario.py
│   ├── run_fedavg_reliability.py
│   ├── run_fedavg_reliability_multiseed.py
│   ├── evaluate_fedavg_stress_tests.py
│   └── reliability_aware_fedavg.py
├── figures/
└── results/
```

## Dataset

The 2+ GB NASA IMS dataset is **not stored in this repository**. Download it separately and follow `data/README.md` for the expected directory structure.

```text
data/NASA/
├── 1st_test/
├── 2nd_test/
└── 3rd_test/
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

```bash
pip install -r requirements.txt
```

## Reproduction order

```powershell
python src/extract_raw_features.py
python src/analyze_features.py
python src/analyze_frequency.py

python src/baseline_som.py

python src/create_fl_dataset.py
python src/create_labels.py
python src/create_federated_clients.py
python src/create_balanced_noniid.py

python src/centralized_mlp.py

# Single-seed baseline
python src/run_fedavg.py --partition iid
python src/run_fedavg.py --partition balanced_noniid
python src/run_fedavg.py --partition noniid

# Five-seed distribution validation
python src/run_fedavg_multiseed.py --partition iid --seeds 42,123,2024,3407,7777
python src/run_fedavg_multiseed.py --partition balanced_noniid --seeds 42,123,2024,3407,7777
python src/run_fedavg_multiseed.py --partition noniid --seeds 42,123,2024,3407,7777

# Dropout schedules
python src/simulate_client_dropout.py

# Five-seed dropout validation
python src/run_fedavg_dropout_multiseed.py --rate 0.2 --seeds 42,123,2024,3407,7777
python src/run_fedavg_dropout_multiseed.py --rate 0.4 --seeds 42,123,2024,3407,7777
python src/run_fedavg_dropout_multiseed.py --rate 0.5 --seeds 42,123,2024,3407,7777

# Reliability schedules
python src/simulate_industrial_client_reliability.py
python src/create_full_mixed_reliability_scenario.py

# Five-seed reliability validation
python src/run_fedavg_reliability_multiseed.py --scenario sensor_noise --seeds 42,123,2024,3407,7777
python src/run_fedavg_reliability_multiseed.py --scenario stale_updates --seeds 42,123,2024,3407,7777
python src/run_fedavg_reliability_multiseed.py --scenario persistent_failure --seeds 42,123,2024,3407,7777
python src/run_fedavg_reliability_multiseed.py --scenario mixed --seeds 42,123,2024,3407,7777
python src/run_fedavg_reliability_multiseed.py --scenario mixed_full --seeds 42,123,2024,3407,7777

# Optional analysis
python src/evaluate_fedavg_stress_tests.py
python src/reliability_aware_fedavg.py --scenario mixed_full
```

## Federated configuration

- clients: 10
- rounds: 15
- local epochs: 1
- batch size: 512
- Adam learning rate: 0.001
- MLP: `36 -> 128 -> 64 -> 32 -> 2`
- dropout probability: 0.40
- global held-out test set: 197 recordings
- validation seeds: `42, 123, 2024, 3407, 7777`
- client partitions and reliability/dropout schedules are fixed across seeds for the multi-seed validation

## Research status

Current stage: **validated FedAvg baseline + five-seed industrial reliability stress testing**.

Completed multi-seed validation covers client heterogeneity, client dropout, sensor noise, stale updates, persistent failure, mixed reliability, and full-mixed reliability.

Planned next stage: statistical significance analysis, stronger threshold/operating-point analysis, and a substantially improved reliability-aware federated aggregation method.

## Citation

The repository includes `CITATION.cff`. After a stable release is archived with Zenodo, cite the release DOI in research publications.
