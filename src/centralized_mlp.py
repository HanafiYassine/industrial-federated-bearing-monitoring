from pathlib import Path
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

DATA = Path(
    "../results/federated/dataset2_fl_dataset.csv"
)

RESULTS_DIR = Path(
    "../results/federated/dataset2"
)

SEED = 42

TRAIN_COUNT = 590

BATCH_SIZE = 512
LEARNING_RATE = 0.001
DROPOUT = 0.40

EPOCHS = 20


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# MODEL
# ============================================================

class MLP(nn.Module):

    def __init__(self, input_size):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                input_size,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                DROPOUT
            ),

            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),

            nn.Linear(
                32,
                2
            )
        )

    def forward(self, x):

        return self.network(x)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CENTRALIZED MLP BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA
    )

    print(
        "Dataset shape:",
        df.shape
    )

    # --------------------------------------------------------
    # Feature columns
    # --------------------------------------------------------

    feature_columns = [
        c
        for c in df.columns
        if c.startswith("ch")
    ]

    print(
        "Feature count:",
        len(feature_columns)
    )

    # --------------------------------------------------------
    # Temporal split
    #
    # Same 590/394 split used by SOM.
    # --------------------------------------------------------

    train = df.iloc[
        :TRAIN_COUNT
    ].copy()

    test = df.iloc[
        TRAIN_COUNT:
    ].copy()

    print(
        "Training:",
        len(train)
    )

    print(
        "Testing :",
        len(test)
    )

    # --------------------------------------------------------
    # Scaling
    #
    # Fit ONLY on training data.
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[
            feature_columns
        ]
    )

    X_test = scaler.transform(
        test[
            feature_columns
        ]
    )

    y_train = train[
        "label"
    ].to_numpy()

    y_test = test[
        "label"
    ].to_numpy()

    # --------------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    loader = DataLoader(

        TensorDataset(
            X_train,
            y_train
        ),

        batch_size=BATCH_SIZE,

        shuffle=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MLP(
        input_size=len(
            feature_columns
        )
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    start_time = time.perf_counter()

    for epoch in range(
        EPOCHS
    ):

        model.train()

        total_loss = 0.0

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(
                X
            )

            loss = criterion(
                logits,
                y
            )

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * len(X)
            )

        epoch_loss = (
            total_loss
            / len(train)
        )

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{EPOCHS} "
            f"| loss = "
            f"{epoch_loss:.6f}"
        )

    training_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(
            X_test.to(device)
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()

    y_true = y_test.cpu().numpy()
    y_pred = predictions.cpu().numpy()
    y_prob = probabilities.cpu().numpy()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    if len(
        np.unique(y_true)
    ) == 2:

        auc = roc_auc_score(
            y_true,
            y_prob
        )

    else:

        auc = np.nan

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CENTRALIZED MLP RESULTS")
    print("=" * 70)

    print(
        f"Accuracy        : {accuracy:.6f}"
    )

    print(
        f"Precision       : {precision:.6f}"
    )

    print(
        f"Recall          : {recall:.6f}"
    )

    print(
        f"F1              : {f1:.6f}"
    )

    print(
        f"ROC-AUC         : {auc:.6f}"
    )

    print(
        f"Training time   : {training_time:.3f} s"
    )

    print()
    print(
        "Confusion matrix:"
    )

    print(
        cm
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    metrics = pd.DataFrame(
        {
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "training_time_seconds",
                "epochs",
                "batch_size",
                "learning_rate",
                "dropout",
                "train_samples",
                "test_samples",
                "feature_count",
                "seed",
            ],

            "value": [
                accuracy,
                precision,
                recall,
                f1,
                auc,
                training_time,
                EPOCHS,
                BATCH_SIZE,
                LEARNING_RATE,
                DROPOUT,
                len(train),
                len(test),
                len(feature_columns),
                SEED,
            ]
        }
    )

    metrics.to_csv(
        RESULTS_DIR
        / "centralized_mlp_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = test[
        [
            "file_index",
            "timestamp",
            "label",
        ]
    ].copy()

    predictions_df[
        "probability_anomaly"
    ] = y_prob

    predictions_df[
        "prediction"
    ] = y_pred

    predictions_df.to_csv(
        RESULTS_DIR
        / "centralized_mlp_predictions.csv",
        index=False
    )

    print()
    print(
        "Saved results to:"
    )

    print(
        RESULTS_DIR.resolve()
    )

    print("=" * 70)


if __name__ == "__main__":
    main()