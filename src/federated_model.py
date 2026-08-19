"""
federated_model.py

Neural-network model used by the federated-learning experiments.

Architecture:

    36 input features
        ↓
      Dense 128
        ↓
       ReLU
        ↓
    Dropout 0.40
        ↓
      Dense 64
        ↓
       ReLU
        ↓
    Dropout 0.40
        ↓
      Dense 32
        ↓
       ReLU
        ↓
      Dense 2
        ↓
    Normal / Anomaly

The architecture follows the configuration described in the
Marfo et al. 2025 federated-learning experiment, while the
36-dimensional feature representation is our controlled
NASA Dataset-2 representation.

This file intentionally contains NO federated aggregation logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FederatedMLP(nn.Module):
    """
    MLP used by all centralized/federated experiments.

    Parameters
    ----------
    input_size : int
        Number of input features.
    hidden_1 : int
        First hidden layer size.
    hidden_2 : int
        Second hidden layer size.
    hidden_3 : int
        Third hidden layer size.
    dropout : float
        Dropout probability.
    num_classes : int
        Number of output classes.
    """

    def __init__(
        self,
        input_size: int = 36,
        hidden_1: int = 128,
        hidden_2: int = 64,
        hidden_3: int = 32,
        dropout: float = 0.40,
        num_classes: int = 2,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "input_size must be greater than zero."
            )

        if hidden_1 <= 0 or hidden_2 <= 0 or hidden_3 <= 0:
            raise ValueError(
                "Hidden-layer sizes must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)."
            )

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        self.input_size = input_size
        self.hidden_1 = hidden_1
        self.hidden_2 = hidden_2
        self.hidden_3 = hidden_3
        self.dropout_probability = dropout
        self.num_classes = num_classes

        self.network = nn.Sequential(
            nn.Linear(
                input_size,
                hidden_1,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_1,
                hidden_2,
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout,
            ),

            nn.Linear(
                hidden_2,
                hidden_3,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_3,
                num_classes,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape: [batch_size, input_size]

        Returns
        -------
        torch.Tensor
            Raw logits with shape:
            [batch_size, num_classes]
        """

        if x.ndim != 2:
            raise ValueError(
                "Expected input shape "
                "[batch_size, features], "
                f"got {tuple(x.shape)}."
            )

        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} input features, "
                f"got {x.shape[1]}."
            )

        return self.network(x)

    def parameter_count(
        self,
    ) -> int:
        """
        Return total number of trainable parameters.
        """

        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

    def trainable_state_dict(self):
        """
        Return a detached copy of model parameters.

        Useful for sending a local model to the federated server.
        """

        return {
            name: tensor.detach().cpu().clone()
            for name, tensor in self.state_dict().items()
        }


def build_model(
    input_size: int = 36,
    dropout: float = 0.40,
) -> FederatedMLP:
    """
    Convenience constructor.
    """

    return FederatedMLP(
        input_size=input_size,
        hidden_1=128,
        hidden_2=64,
        hidden_3=32,
        dropout=dropout,
        num_classes=2,
    )


if __name__ == "__main__":

    # --------------------------------------------------------
    # Simple self-test
    # --------------------------------------------------------

    print("=" * 70)
    print("FEDERATED MODEL SELF-TEST")
    print("=" * 70)

    model = build_model()

    print(model)

    print()
    print(
        "Trainable parameters:",
        model.parameter_count(),
    )

    # Test a batch of 8 observations.
    x = torch.randn(
        8,
        36,
    )

    output = model(x)

    print(
        "Input shape :",
        tuple(x.shape),
    )

    print(
        "Output shape:",
        tuple(output.shape),
    )

    assert output.shape == (
        8,
        2,
    )

    print()
    print("SELF-TEST PASSED")
    print("=" * 70)