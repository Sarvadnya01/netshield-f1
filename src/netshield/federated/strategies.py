"""FL strategies: fedavg, fedprox, local_only.

Each strategy function takes partition data, config, and global test set,
runs the appropriate training, and returns (global_weights, history, test_metrics).
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from netshield.common.config import get_config
from netshield.common.hardware import get_device
from netshield.common.labels import NUM_CLASSES
from netshield.federated.client import FLClient
from netshield.federated.server import _evaluate_global, run_fl
from netshield.models.mlp import MLP

logger = logging.getLogger(__name__)


def _build_clients(
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: list[np.ndarray],
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    local_epochs: int,
    local_lr: float,
    local_batch_size: int,
    mu: float,
    val_fraction: float,
    seed: int,
) -> list[FLClient]:
    """Build FLClient instances with local train/val splits."""
    rng = np.random.RandomState(seed)
    clients = []

    for k, idx in enumerate(client_indices):
        X_k = X_train[idx]
        y_k = y_train[idx]

        # Split into local train / val
        n = len(X_k)
        n_val = max(1, int(n * val_fraction))
        perm = rng.permutation(n)
        val_idx = perm[:n_val]
        train_idx = perm[n_val:]

        client = FLClient(
            client_id=k,
            X_train=X_k[train_idx],
            y_train=y_k[train_idx],
            X_val=X_k[val_idx],
            y_val=y_k[val_idx],
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
            local_epochs=local_epochs,
            local_lr=local_lr,
            local_batch_size=local_batch_size,
            mu=mu,
        )
        clients.append(client)
        logger.info(
            "Client %d: %d train, %d val samples",
            k, len(train_idx), len(val_idx),
        )

    return clients


def run_fedavg(
    run_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: list[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    """Run FedAvg (mu=0)."""
    return _run_strategy(
        run_name, X_train, y_train, client_indices,
        X_val, y_val, X_test, y_test, mu=0.0, seed=seed,
    )


def run_fedprox(
    run_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: list[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mu: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    """Run FedProx with given mu."""
    return _run_strategy(
        run_name, X_train, y_train, client_indices,
        X_val, y_val, X_test, y_test, mu=mu, seed=seed,
    )


def run_local_only(
    run_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: list[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, torch.Tensor] | None, list[dict], dict]:
    """Train each client locally (no aggregation). Report mean/worst client F1 on global test.

    Uses the same total compute budget as FL: rounds * local_epochs per client.
    """
    cfg = get_config()
    fl_cfg = cfg.get("fl", {})
    mlp_cfg = cfg.get("mlp", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64])
    dropout = mlp_cfg.get("dropout", 0.3)
    local_epochs = fl_cfg.get("local_epochs", 3)
    local_lr = fl_cfg.get("local_lr", 0.001)
    local_batch_size = fl_cfg.get("local_batch_size", 2048)
    num_rounds = fl_cfg.get("rounds", 20)
    val_fraction = fl_cfg.get("local_val_fraction", 0.1)
    input_dim = X_train.shape[1]

    # Total local epochs = rounds * local_epochs (same compute budget)
    total_local_epochs = num_rounds * local_epochs

    clients = _build_clients(
        X_train, y_train, client_indices,
        input_dim, hidden_dims, dropout,
        total_local_epochs, local_lr, local_batch_size,
        mu=0.0, val_fraction=val_fraction, seed=seed,
    )

    device = torch.device(get_device())
    torch.manual_seed(seed)
    init_model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout)
    init_weights = init_model.get_weights()

    client_test_f1s = []
    history = []

    for k, client in enumerate(clients):
        result = client.train_round(init_weights)
        # Evaluate this client's model on global test
        model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout)
        model.set_weights(result["weights"])
        model.to(device)
        model.eval()

        with torch.no_grad():
            X_t = torch.from_numpy(X_test).float().to(device)
            logits = model(X_t).cpu()
        y_pred = logits.argmax(dim=1).numpy()
        from sklearn.metrics import f1_score
        test_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        client_test_f1s.append(test_f1)

        logger.info(
            "Local-only client %d: local_val_f1=%.4f  global_test_f1=%.4f",
            k, result["val_macro_f1"], test_f1,
        )

    history.append({
        "round": 1,
        "client_test_f1s": client_test_f1s,
        "mean_client_test_f1": float(np.mean(client_test_f1s)),
        "worst_client_test_f1": float(np.min(client_test_f1s)),
    })

    test_metrics = {
        "macro_f1": float(np.mean(client_test_f1s)),
        "worst_client_f1": float(np.min(client_test_f1s)),
        "accuracy": None,
        "per_client_test_f1": client_test_f1s,
    }

    return None, history, test_metrics


def _run_strategy(
    run_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    client_indices: list[np.ndarray],
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mu: float,
    seed: int,
) -> tuple[dict[str, torch.Tensor], list[dict], dict]:
    """Shared implementation for fedavg/fedprox."""
    cfg = get_config()
    fl_cfg = cfg.get("fl", {})
    mlp_cfg = cfg.get("mlp", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64])
    dropout = mlp_cfg.get("dropout", 0.3)
    local_epochs = fl_cfg.get("local_epochs", 3)
    local_lr = fl_cfg.get("local_lr", 0.001)
    local_batch_size = fl_cfg.get("local_batch_size", 2048)
    num_rounds = fl_cfg.get("rounds", 20)
    fraction_fit = fl_cfg.get("fraction_fit", 1.0)
    checkpoint_every = fl_cfg.get("checkpoint_every", 5)
    val_fraction = fl_cfg.get("local_val_fraction", 0.1)
    input_dim = X_train.shape[1]

    clients = _build_clients(
        X_train, y_train, client_indices,
        input_dim, hidden_dims, dropout,
        local_epochs, local_lr, local_batch_size,
        mu=mu, val_fraction=val_fraction, seed=seed,
    )

    global_weights, history = run_fl(
        run_name=run_name,
        clients=clients,
        X_val=X_val,
        y_val=y_val,
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
        num_rounds=num_rounds,
        fraction_fit=fraction_fit,
        checkpoint_every=checkpoint_every,
        seed=seed,
    )

    # Final test evaluation
    device = torch.device(get_device())
    test_metrics = _evaluate_global(
        global_weights, X_test, y_test,
        input_dim, hidden_dims, dropout, device,
    )

    return global_weights, history, test_metrics
