"""Federated client: local training with optional FedProx proximal term."""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

from netshield.common.hardware import get_device, should_put_data_on_gpu
from netshield.common.labels import NUM_CLASSES
from netshield.models.mlp import MLP

logger = logging.getLogger(__name__)


class FLClient:
    """A single federated learning client."""

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_dim: int,
        hidden_dims: list[int],
        dropout: float,
        local_epochs: int,
        local_lr: float,
        local_batch_size: int,
        mu: float = 0.0,
    ):
        self.client_id = client_id
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.local_epochs = local_epochs
        self.local_lr = local_lr
        self.local_batch_size = local_batch_size
        self.mu = mu
        self.n_samples = len(X_train)

        self.device = torch.device(get_device())

        # Decide data placement
        data_bytes = X_train.nbytes + y_train.nbytes + X_val.nbytes + y_val.nbytes
        self.data_on_gpu = should_put_data_on_gpu(data_bytes)

        pin = self.device.type == "cuda" and not self.data_on_gpu

        self.train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_train).float(),
                torch.from_numpy(y_train).long(),
            ),
            batch_size=local_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=pin,
        )
        self.val_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val).float(),
                torch.from_numpy(y_val).long(),
            ),
            batch_size=local_batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=pin,
        )

    def train_round(
        self, global_weights: dict[str, torch.Tensor],
    ) -> dict:
        """Run E local epochs starting from global_weights.

        Returns dict with keys: weights, n_samples, loss, val_macro_f1.
        """
        model = MLP(self.input_dim, NUM_CLASSES, self.hidden_dims, self.dropout)
        model.set_weights(global_weights)
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=self.local_lr)
        criterion = nn.CrossEntropyLoss()

        # Keep a copy of global weights for FedProx proximal term
        if self.mu > 0:
            global_params = {
                k: v.clone().to(self.device) for k, v in global_weights.items()
            }

        total_loss = 0.0
        total_samples = 0

        for _epoch in range(self.local_epochs):
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)

                # FedProx proximal term: (mu/2) * ||w - w_global||^2
                if self.mu > 0:
                    prox = 0.0
                    for name, param in model.named_parameters():
                        prox += ((param - global_params[name]) ** 2).sum()
                    loss = loss + (self.mu / 2.0) * prox

                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(xb)
                total_samples += len(xb)

        avg_loss = total_loss / max(total_samples, 1)

        # Evaluate on local val
        val_f1 = self._evaluate(model)

        weights = model.get_weights()

        return {
            "weights": weights,
            "n_samples": self.n_samples,
            "loss": avg_loss,
            "val_macro_f1": val_f1,
        }

    def _evaluate(self, model: MLP) -> float:
        """Evaluate model on local validation set, return macro-F1."""
        model.eval()
        all_preds = []
        all_true = []
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb = xb.to(self.device)
                logits = model(xb)
                all_preds.append(logits.argmax(dim=1).cpu().numpy())
                all_true.append(yb.numpy())

        y_pred = np.concatenate(all_preds)
        y_true = np.concatenate(all_true)
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
