"""Dirichlet label-skew partitioning for federated learning.

Partitions TRAIN data among K clients using Dirichlet(alpha) per-label allocation.
Deterministic from (alpha, seed) so the lab regenerates identical partitions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from netshield.common.config import repo_root
from netshield.common.labels import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)


def dirichlet_partition(
    y: np.ndarray,
    num_clients: int,
    alpha: float | str,
    seed: int = 42,
    min_samples: int = 50,
) -> list[np.ndarray]:
    """Partition sample indices among clients using Dirichlet label skew.

    Args:
        y: Label array (int64 class indices).
        num_clients: Number of clients K.
        alpha: Dirichlet concentration. Use "iid" for uniform split.
        seed: Random seed for reproducibility.
        min_samples: Minimum samples per client (redistributed if needed).

    Returns:
        List of K index arrays, one per client.
    """
    rng = np.random.RandomState(seed)
    n = len(y)

    if alpha == "iid":
        indices = rng.permutation(n)
        return [indices[i::num_clients] for i in range(num_clients)]

    alpha_val = float(alpha)
    # Group indices by class
    class_indices = [np.where(y == c)[0] for c in range(NUM_CLASSES)]

    # Dirichlet allocation per class
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]

    for c in range(NUM_CLASSES):
        idx_c = class_indices[c]
        rng.shuffle(idx_c)

        # Draw proportions from Dirichlet
        proportions = rng.dirichlet(np.full(num_clients, alpha_val))
        # Convert to counts
        counts = (proportions * len(idx_c)).astype(int)
        # Fix rounding: give remainder to random clients
        remainder = len(idx_c) - counts.sum()
        for _ in range(remainder):
            counts[rng.randint(num_clients)] += 1

        start = 0
        for k in range(num_clients):
            client_indices[k].extend(idx_c[start : start + counts[k]])
            start += counts[k]

    # Ensure minimum samples per client by redistribution
    result = [np.array(ci, dtype=np.int64) for ci in client_indices]
    # If any client has fewer than min_samples, steal from the largest
    for _ in range(10):  # max iterations to prevent infinite loop
        sizes = [len(r) for r in result]
        min_idx = int(np.argmin(sizes))
        if sizes[min_idx] >= min_samples:
            break
        max_idx = int(np.argmax(sizes))
        deficit = min_samples - sizes[min_idx]
        transfer = min(deficit, sizes[max_idx] - min_samples)
        if transfer <= 0:
            break
        result[min_idx] = np.concatenate([result[min_idx], result[max_idx][-transfer:]])
        result[max_idx] = result[max_idx][:-transfer]

    # Shuffle each client's indices
    for r in result:
        rng.shuffle(r)

    return result


def create_partition(
    y: np.ndarray,
    alpha: float | str,
    num_clients: int,
    seed: int = 42,
    min_samples: int = 50,
    name: str | None = None,
) -> tuple[list[np.ndarray], dict]:
    """Create a named partition and save artifacts.

    Args:
        y: Training labels.
        alpha: Dirichlet alpha or "iid".
        num_clients: Number of clients.
        seed: Random seed.
        min_samples: Minimum samples per client.
        name: Partition name (auto-generated if None).

    Returns:
        (client_indices, partition_info) tuple.
    """
    if name is None:
        alpha_str = "iid" if alpha == "iid" else f"a{alpha}"
        name = f"K{num_clients}_{alpha_str}_s{seed}"

    client_indices = dirichlet_partition(y, num_clients, alpha, seed, min_samples)

    root = repo_root()

    # Save partition npz
    part_dir = root / "data" / "partitions"
    part_dir.mkdir(parents=True, exist_ok=True)
    npz_path = part_dir / f"{name}.npz"
    save_dict = {f"client_{k}": idx for k, idx in enumerate(client_indices)}
    np.savez(npz_path, **save_dict)
    logger.info("Partition saved: %s (%d clients)", npz_path, num_clients)

    # Compute distribution stats
    dist = np.zeros((num_clients, NUM_CLASSES), dtype=int)
    for k, idx in enumerate(client_indices):
        for c in range(NUM_CLASSES):
            dist[k, c] = int(np.sum(y[idx] == c))

    info = {
        "name": name,
        "alpha": alpha,
        "num_clients": num_clients,
        "seed": seed,
        "min_samples": min_samples,
        "total_samples": int(len(y)),
        "client_sizes": [int(len(idx)) for idx in client_indices],
        "class_distribution": dist.tolist(),
    }

    # Save partition info JSON
    fl_dir = root / "reports" / "fl_runs"
    fl_dir.mkdir(parents=True, exist_ok=True)
    info_path = fl_dir / f"partition_{name}.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    # Save heatmap
    fig_dir = root / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _save_heatmap(dist, name, fig_dir)

    return client_indices, info


def _save_heatmap(dist: np.ndarray, name: str, save_dir: Path) -> None:
    """Save a client × class distribution heatmap."""
    fig, ax = plt.subplots(figsize=(10, max(4, dist.shape[0] * 0.6)))
    sns.heatmap(
        dist, annot=True, fmt="d", cmap="YlOrRd",
        xticklabels=CLASS_NAMES,
        yticklabels=[f"Client {k}" for k in range(dist.shape[0])],
        ax=ax,
    )
    ax.set_title(f"Partition: {name}")
    ax.set_xlabel("Class")
    ax.set_ylabel("Client")
    fig.tight_layout()
    fig.savefig(save_dir / f"partition_{name}.png", dpi=150)
    plt.close(fig)
    logger.info("Partition heatmap saved: %s", save_dir / f"partition_{name}.png")


def load_partition(name: str) -> list[np.ndarray]:
    """Load a saved partition by name."""
    root = repo_root()
    npz_path = root / "data" / "partitions" / f"{name}.npz"
    data = np.load(npz_path)
    num_clients = len(data.files)
    return [data[f"client_{k}"] for k in range(num_clients)]
