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
- 20%, 40%, and 50% random client dropout;
- sensor noise, stale updates, persistent failure, noise+stale, and deterministic full-mixed reliability scenarios;
- preliminary reliability-aware FedAvg aggregation.

## Main Dataset-2 results

Clean unbalanced non-IID FedAvg:

- Accuracy: `0.898477`
- Recall: `0.655172`
- F1: `0.791667`
- ROC-AUC: `0.956835`

Full-mixed reliability FedAvg:

- Accuracy: `0.837563`
- Recall: `0.448276`
- F1: `0.619048`
- ROC-AUC: `0.934384`

Preliminary RA-FedAvg v1 on the same full-mixed condition:

- Accuracy: `0.822335`
- Recall: `0.396552`
- F1: `0.567901`
- ROC-AUC: `0.933887`

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
│   ├── evaluate_federated_baselines.py
│   ├── simulate_client_dropout.py
│   ├── run_fedavg_dropout.py
│   ├── simulate_industrial_client_reliability.py
│   ├── create_full_mixed_reliability_scenario.py
│   ├── run_fedavg_reliability.py
│   ├── evaluate_fedavg_stress_tests.py
│   └── reliability_aware_fedavg.py
├── figures/
├── results/
└── paper/
    ├── main.tex
    └── references.bib
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

python src/run_fedavg.py --partition iid
python src/run_fedavg.py --partition balanced_noniid
python src/run_fedavg.py --partition noniid
python src/evaluate_federated_baselines.py

python src/simulate_client_dropout.py
python src/run_fedavg_dropout.py --rate 0.2
python src/run_fedavg_dropout.py --rate 0.4
python src/run_fedavg_dropout.py --rate 0.5

python src/simulate_industrial_client_reliability.py
python src/create_full_mixed_reliability_scenario.py
python src/run_fedavg_reliability.py --scenario sensor_noise
python src/run_fedavg_reliability.py --scenario stale_updates
python src/run_fedavg_reliability.py --scenario persistent_failure
python src/run_fedavg_reliability.py --scenario mixed
python src/run_fedavg_reliability.py --scenario mixed_full

python src/evaluate_fedavg_stress_tests.py
python src/reliability_aware_fedavg.py --scenario mixed_full
```

## Federated configuration

- clients: 10
- rounds: 15
- seed: 42
- local epochs: 1
- batch size: 512
- Adam learning rate: 0.001
- MLP: `36 -> 128 -> 64 -> 32 -> 2`
- dropout probability: 0.40
- global held-out test set: 197 recordings

## Research status

Current stage: reproduction + FedAvg baseline + industrial reliability stress testing.

Planned next stage: multi-seed validation, statistical testing, and an optimized reliability-aware federated aggregation method.

## Citation

The repository includes `CITATION.cff`. After a stable release is archived with Zenodo, cite the release DOI in research publications.
