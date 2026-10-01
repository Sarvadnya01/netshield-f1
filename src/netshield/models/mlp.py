"""Configurable MLP for 8-class IoT attack classification.

Architecture: input → [Linear → LayerNorm → GELU → Dropout] × N → Linear → output
Uses LayerNorm (never BatchNorm).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with LayerNorm, GELU, and Dropout."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 8,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout

        layers: list[tuple[str, nn.Module]] = []
        prev = input_dim
        for i, h in enumerate(hidden_dims):
            layers.append((f"linear_{i}", nn.Linear(prev, h)))
            layers.append((f"norm_{i}", nn.LayerNorm(h)))
            layers.append((f"act_{i}", nn.GELU()))
            layers.append((f"drop_{i}", nn.Dropout(dropout)))
            prev = h
        layers.append(("head", nn.Linear(prev, num_classes)))

        self.net = nn.Sequential(OrderedDict(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    # -- FL helpers -----------------------------------------------------------

    def get_weights(self) -> dict[str, torch.Tensor]:
        """Return a deep copy of model weights on CPU."""
        return {k: v.clone().cpu() for k, v in self.state_dict().items()}

    def set_weights(self, weights: dict[str, torch.Tensor]) -> None:
        """Load weights (moves to current device automatically)."""
        self.load_state_dict(weights)
