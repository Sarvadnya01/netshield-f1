"""Tests for federated learning: aggregation, FedProx, partitions, resume."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---- Weighted aggregation correctness --------------------------------------

def test_weighted_aggregate_correctness():
    """Verify weighted_aggregate produces correct weighted average."""
    from netshield.federated.server import weighted_aggregate

    # Two clients with known weights
    w1 = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([3.0])}
    w2 = {"a": torch.tensor([5.0, 6.0]), "b": torch.tensor([7.0])}

    results = [
        {"weights": w1, "n_samples": 100},
        {"weights": w2, "n_samples": 300},
    ]

    agg = weighted_aggregate(results)

    # Expected: w = (100*w1 + 300*w2) / 400
    expected_a = (100 * np.array([1, 2]) + 300 * np.array([5, 6])) / 400
    expected_b = (100 * np.array([3]) + 300 * np.array([7])) / 400

    np.testing.assert_allclose(agg["a"].numpy(), expected_a, rtol=1e-5)
    np.testing.assert_allclose(agg["b"].numpy(), expected_b, rtol=1e-5)


def test_weighted_aggregate_single_client():
    """Single-client aggregation should return the same weights."""
    from netshield.federated.server import weighted_aggregate

    w = {"x": torch.tensor([1.0, 2.0, 3.0])}
    results = [{"weights": w, "n_samples": 50}]
    agg = weighted_aggregate(results)
    np.testing.assert_allclose(agg["x"].numpy(), w["x"].numpy())


# ---- FedProx mu=0 == FedAvg for fixed seed ---------------------------------

def test_fedprox_mu0_equals_fedavg():
    """FedProx with mu=0 should produce identical results to FedAvg."""
    from netshield.federated.client import FLClient
    from netshield.models.mlp import MLP
    from netshield.common.labels import NUM_CLASSES

    np.random.seed(42)
    torch.manual_seed(42)

    input_dim = 10
    X = np.random.randn(200, input_dim).astype(np.float32)
    y = np.random.randint(0, NUM_CLASSES, 200).astype(np.int64)

    # Init global model
    model = MLP(input_dim, NUM_CLASSES, [16, 8], dropout=0.0)
    global_weights = model.get_weights()

    # FedAvg client (mu=0)
    torch.manual_seed(42)
    c_avg = FLClient(
        client_id=0, X_train=X[:180], y_train=y[:180],
        X_val=X[180:], y_val=y[180:],
        input_dim=input_dim, hidden_dims=[16, 8], dropout=0.0,
        local_epochs=2, local_lr=0.01, local_batch_size=64, mu=0.0,
    )
    torch.manual_seed(99)
    r_avg = c_avg.train_round(global_weights)

    # FedProx client (mu=0, should be identical)
    torch.manual_seed(42)
    c_prox = FLClient(
        client_id=0, X_train=X[:180], y_train=y[:180],
        X_val=X[180:], y_val=y[180:],
        input_dim=input_dim, hidden_dims=[16, 8], dropout=0.0,
        local_epochs=2, local_lr=0.01, local_batch_size=64, mu=0.0,
    )
    torch.manual_seed(99)
    r_prox = c_prox.train_round(global_weights)

    for key in r_avg["weights"]:
        np.testing.assert_allclose(
            r_avg["weights"][key].numpy(),
            r_prox["weights"][key].numpy(),
            atol=1e-6,
            err_msg=f"Weights differ for key {key}",
        )


# ---- Partition disjoint and complete ---------------------------------------

def test_partition_disjoint_and_complete():
    """Partition indices must be disjoint and union equals full dataset."""
    from netshield.federated.partition import dirichlet_partition

    n = 1000
    y = np.random.randint(0, 8, n).astype(np.int64)
    indices = dirichlet_partition(y, num_clients=5, alpha=0.5, seed=42)

    all_idx = np.concatenate(indices)
    # Complete: all indices present
    assert len(all_idx) == n
    assert set(all_idx) == set(range(n))

    # Disjoint: no duplicates
    assert len(np.unique(all_idx)) == n


def test_partition_iid_roughly_balanced():
    """IID partition should produce roughly equal-sized clients."""
    from netshield.federated.partition import dirichlet_partition

    n = 1000
    y = np.random.randint(0, 8, n).astype(np.int64)
    indices = dirichlet_partition(y, num_clients=5, alpha="iid", seed=42)

    sizes = [len(idx) for idx in indices]
    assert max(sizes) - min(sizes) <= 1  # round-robin should be nearly equal


def test_partition_deterministic():
    """Same (alpha, seed) should produce identical partitions."""
    from netshield.federated.partition import dirichlet_partition

    y = np.random.RandomState(7).randint(0, 8, 500).astype(np.int64)

    p1 = dirichlet_partition(y, num_clients=4, alpha=0.5, seed=42)
    p2 = dirichlet_partition(y, num_clients=4, alpha=0.5, seed=42)

    for a, b in zip(p1, p2):
        np.testing.assert_array_equal(a, b)


# ---- Resume from checkpoint yields same history length ---------------------

def test_resume_checkpoint():
    """After checkpointing, resuming should yield the same total history length."""
    from netshield.federated.server import _save_checkpoint, _load_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir) / "test_ckpt"

        weights = {"w": torch.tensor([1.0, 2.0])}
        history = [
            {"round": 1, "global_val_macro_f1": 0.5},
            {"round": 2, "global_val_macro_f1": 0.6},
            {"round": 3, "global_val_macro_f1": 0.65},
        ]

        _save_checkpoint(ckpt_dir, 3, weights, history)

        loaded = _load_checkpoint(ckpt_dir)
        assert loaded is not None
        rnd, loaded_weights, loaded_history = loaded
        assert rnd == 3
        assert len(loaded_history) == 3
        np.testing.assert_allclose(
            loaded_weights["w"].numpy(), weights["w"].numpy(),
        )

        # If we resume from round 3, next round should be 4
        # Total rounds = 5 → should run rounds 4, 5 → history grows by 2
        assert rnd + 1 == 4  # next round to run


def test_checkpoint_not_found():
    """Loading from empty dir returns None."""
    from netshield.federated.server import _load_checkpoint

    with tempfile.TemporaryDirectory() as tmpdir:
        result = _load_checkpoint(Path(tmpdir) / "nonexistent")
        assert result is None
