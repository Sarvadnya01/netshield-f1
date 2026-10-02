"""Evaluation utilities: metrics, confusion matrix, inference latency."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from netshield.common.labels import CLASS_NAMES

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray | None = None,
) -> dict:
    """Compute classification metrics.

    Returns dict with accuracy, macro_f1, weighted_f1, and per-class P/R/F1.
    """
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }

    logger.info("accuracy=%.4f  macro_f1=%.4f  weighted_f1=%.4f", acc, macro_f1, weighted_f1)
    return result


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_dir: Path,
    prefix: str = "cm",
) -> None:
    """Save confusion matrix as PNG and JSON."""
    save_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))

    # JSON
    cm_dict = {
        "matrix": cm.tolist(),
        "labels": CLASS_NAMES,
    }
    with open(save_dir / f"{prefix}.json", "w") as f:
        json.dump(cm_dict, f, indent=2)

    # PNG
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(save_dir / f"{prefix}.png", dpi=150)
    plt.close(fig)
    logger.info("Confusion matrix saved to %s", save_dir)


def measure_inference_latency(
    predict_fn,
    X: np.ndarray,
    batch_sizes: list[int] | None = None,
    warmup: int = 5,
    repeats: int = 20,
) -> dict[str, float]:
    """Measure inference latency in ms for various batch sizes.

    Args:
        predict_fn: Callable taking a float32 ndarray and returning predictions.
        X: Sample data to benchmark with.
        batch_sizes: List of batch sizes to test.
        warmup: Number of warmup iterations.
        repeats: Number of timed iterations.

    Returns:
        Dict mapping "batch_{n}" to mean latency in ms.
    """
    if batch_sizes is None:
        batch_sizes = [1, 256, 4096]

    results = {}
    for bs in batch_sizes:
        batch = X[:bs].copy()
        # Warmup
        for _ in range(warmup):
            predict_fn(batch)
        # Timed
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            predict_fn(batch)
            times.append((time.perf_counter() - t0) * 1000)
        mean_ms = float(np.mean(times))
        results[f"batch_{bs}"] = round(mean_ms, 3)
        logger.info("Latency batch=%d: %.3f ms", bs, mean_ms)

    return results
