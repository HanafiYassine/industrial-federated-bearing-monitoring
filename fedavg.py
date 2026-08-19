"""
fedavg.py

Baseline Federated Averaging implementation.

Responsibilities:
    - Load one client dataset.
    - Train a local FederatedMLP.
    - Return local model parameters.
    - Compute weighted FedAvg aggregation.
    - Copy aggregated parameters to the global model.

This module intentionally does NOT contain:
    - adaptive aggregation
    - client selection
    - client reliability weighting
    - Weibull checkpointing
    - optimization

Those will be implemented later as separate components.

FedAvg:

    w_global = Σ_k (n_k / N) * w_k

where:
    n_k = number of samples at client k
    N   = total samples across participating clients
"""


from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from federated_model import (
    FederatedMLP,
    build_model,
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 512

LOCAL_EPOCHS = 1

LEARNING_RATE = 0.001


# ============================================================
# FEATURE IDENTIFICATION
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> List[str]:
    """
    Find the 36 numerical feature columns.

    Feature columns are the columns beginning with 'ch'.
    """

    columns = [
        column
        for column in df.columns
        if column.startswith("ch")
    ]

    if len(columns) == 0:
        raise ValueError(
            "No feature columns beginning with 'ch' were found."
        )

    return columns


# ============================================================
# DATASET PREPARATION
# ============================================================

def dataframe_to_loader(
    df: pd.DataFrame,
    feature_columns: List[str],
    batch_size: int = BATCH_SIZE,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convert one client DataFrame into a PyTorch DataLoader.
    """

    if "label" not in df.columns:
        raise ValueError(
            "Client dataset must contain a 'label' column."
        )

    X = df[
        feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    y = df[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    if len(X) == 0:
        raise ValueError(
            "Cannot build DataLoader from an empty client."
        )

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y,
        dtype=torch.long,
    )

    dataset = TensorDataset(
        X_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
    )


# ============================================================
# LOCAL TRAINING
# ============================================================

def train_local_model(
    global_state: OrderedDict,
    client_df: pd.DataFrame,
    device: torch.device,
    input_size: int,
    local_epochs: int = LOCAL_EPOCHS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
) -> Tuple[OrderedDict, float, int]:
    """
    Train a copy of the global model on one client.

    Parameters
    ----------
    global_state:
        Parameters received from the server.

    client_df:
        Client's local training data.

    device:
        CPU/GPU device.

    input_size:
        Number of input features.

    local_epochs:
        Number of local epochs.

    learning_rate:
        Adam learning rate.

    batch_size:
        Local batch size.

    Returns
    -------
    local_state:
        Updated local model parameters.

    mean_loss:
        Mean local training loss.

    sample_count:
        Number of local training samples.
    """

    if local_epochs <= 0:
        raise ValueError(
            "local_epochs must be >= 1."
        )

    model = build_model(
        input_size=input_size,
    ).to(device)

    model.load_state_dict(
        global_state
    )

    model.train()

    feature_columns = get_feature_columns(
        client_df
    )

    loader = dataframe_to_loader(
        client_df,
        feature_columns,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    losses = []

    for _ in range(local_epochs):

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(X)

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            losses.append(
                float(loss.item())
            )

    local_state = OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor
        in model.state_dict().items()
    )

    mean_loss = (
        float(np.mean(losses))
        if losses
        else float("nan")
    )

    sample_count = len(
        client_df
    )

    return (
        local_state,
        mean_loss,
        sample_count,
    )


# ============================================================
# FEDAVG AGGREGATION
# ============================================================

def fedavg(
    client_states: Iterable[OrderedDict],
    client_sample_counts: Iterable[int],
) -> OrderedDict:
    """
    Standard weighted FedAvg.

        w = Σ_k (n_k / N) w_k

    Parameters
    ----------
    client_states:
        Local model parameter dictionaries.

    client_sample_counts:
        Number of training samples at each client.

    Returns
    -------
    global_state:
        Weighted-average model parameters.
    """

    states = list(
        client_states
    )

    counts = list(
        client_sample_counts
    )

    if len(states) == 0:
        raise ValueError(
            "No client states were provided."
        )

    if len(states) != len(counts):
        raise ValueError(
            "Number of client states and sample counts differ."
        )

    if any(
        count <= 0
        for count in counts
    ):
        raise ValueError(
            "Every client must have at least one sample."
        )

    total_samples = sum(
        counts
    )

    if total_samples <= 0:
        raise ValueError(
            "Total number of samples must be positive."
        )

    # --------------------------------------------------------
    # Initialize output state from the first client.
    # --------------------------------------------------------

    global_state = OrderedDict()

    for name in states[0]:

        reference = states[0][name]

        # Floating-point tensors can be averaged.
        if torch.is_floating_point(
            reference
        ):

            aggregate = torch.zeros_like(
                reference,
                dtype=torch.float32,
            )

            for state, count in zip(
                states,
                counts,
            ):

                weight = (
                    count
                    / total_samples
                )

                aggregate += (
                    state[name].float()
                    * weight
                )

            global_state[name] = aggregate

        else:
            # Integer/buffer tensors are copied from the
            # largest client contribution. This is appropriate
            # for the current simple MLP, whose state is floating
            # point, but keeps the function robust.
            largest = int(
                np.argmax(
                    counts
                )
            )

            global_state[name] = (
                states[largest][name]
                .clone()
            )

    return global_state


# ============================================================
# MODEL PARAMETER UTILITIES
# ============================================================

def clone_state_dict(
    state: Dict[str, torch.Tensor],
) -> OrderedDict:
    """
    Return an independent CPU copy.
    """

    return OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in state.items()
    )


def parameter_count(
    state: Dict[str, torch.Tensor],
) -> int:
    """
    Count scalar parameters represented by a state dict.
    """

    return sum(
        tensor.numel()
        for tensor in state.values()
        if torch.is_floating_point(tensor)
    )


def model_size_bytes(
    state: Dict[str, torch.Tensor],
    bytes_per_parameter: int = 4,
) -> int:
    """
    Estimate serialized FP32 model size.

    This is an analytical communication estimate,
    not a network benchmark.
    """

    return (
        parameter_count(state)
        * bytes_per_parameter
    )


# ============================================================
# QUICK SELF-TEST
# ============================================================

def self_test():

    print("=" * 70)
    print("FEDAVG MODULE SELF-TEST")
    print("=" * 70)

    model = build_model(
        input_size=36
    )

    state = clone_state_dict(
        model.state_dict()
    )

    params = parameter_count(
        state
    )

    size_bytes = model_size_bytes(
        state
    )

    print(
        "Parameters :",
        params,
    )

    print(
        "FP32 model:",
        size_bytes,
        "bytes",
    )

    # Create two artificial local models.
    state_a = clone_state_dict(
        state
    )

    state_b = clone_state_dict(
        state
    )

    # Make the models intentionally different.
    for name in state_b:

        if torch.is_floating_point(
            state_b[name]
        ):

            state_b[name] = (
                state_b[name]
                + 1.0
            )

    aggregated = fedavg(
        [state_a, state_b],
        [1, 1],
    )

    # For equal weights:
    # aggregated = state_a + 0.5
    for name in state_a:

        if torch.is_floating_point(
            state_a[name]
        ):

            expected = (
                state_a[name]
                + 0.5
            )

            if not torch.allclose(
                aggregated[name],
                expected,
            ):
                raise AssertionError(
                    f"FedAvg failed for parameter {name}"
                )

    print(
        "FedAvg weighted averaging: PASS"
    )

    print("=" * 70)
    print("SELF-TEST PASSED")
    print("=" * 70)


if __name__ == "__main__":
    self_test()