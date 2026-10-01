"""Data loading: parquet → preprocessed numpy arrays for training."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from netshield.common.config import get_config, repo_root
from netshield.common.labels import CLASS_TO_IDX
from netshield.common.preprocess import transform

logger = logging.getLogger(__name__)


def _load_stats(drop_features: list[str] | None = None) -> dict[str, Any]:
    """Load feature_stats.json, optionally dropping features."""
    root = repo_root()
    cfg = get_config()
    stats_path = root / cfg["data"]["feature_stats"]
    with open(stats_path) as f:
        stats = json.load(f)

    if drop_features:
        keep = [i for i, f in enumerate(stats["feature_names"]) if f not in set(drop_features)]
        stats = {
            "feature_names": [stats["feature_names"][i] for i in keep],
            "mean": [stats["mean"][i] for i in keep],
            "std": [stats["std"][i] for i in keep],
        }
        logger.info("Dropped %d features, %d remaining", len(drop_features), len(keep))

    return stats


def load_split(
    split: str,
    max_rows: int | None = None,
    drop_features: list[str] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a data split from parquet, preprocess, return (X_float32, y_int64).

    Args:
        split: One of "train", "val", "test".
        max_rows: If set, stratified subsample to at most this many rows.
        drop_features: Feature names to exclude (e.g. ["iat"]).
        seed: Random seed for subsampling.

    Returns:
        (X, y) where X is float32 and y is int64 (class indices).
    """
    root = repo_root()
    cfg = get_config()
    split_path = root / cfg["data"]["processed_dir"] / split

    mem_before = psutil.Process().memory_info().rss / (1024**2)

    # Read parquet
    df = pd.read_parquet(split_path)
    logger.info("Loaded %s: %d rows, %d columns", split, len(df), len(df.columns))

    # Stratified subsample if needed
    if max_rows is not None and len(df) > max_rows:
        frac = max_rows / len(df)
        df = (
            df.groupby("label_8", group_keys=False)
            .apply(lambda g: g.sample(frac=min(frac, 1.0), random_state=seed))
        )
        logger.info("Subsampled %s to %d rows (max_rows=%d)", split, len(df), max_rows)

    # Load stats and transform
    stats = _load_stats(drop_features=drop_features)
    X = transform(df, stats)

    # Map labels to int indices
    y = df["label_8"].map(CLASS_TO_IDX).values.astype(np.int64)

    mem_after = psutil.Process().memory_info().rss / (1024**2)
    logger.info(
        "%s ready: X %s (%.0f MB), y %s | process RSS %.0f MB (+%.0f)",
        split, X.shape, X.nbytes / 1e6, y.shape, mem_after, mem_after - mem_before,
    )
    return X, y


def get_feature_names(drop_features: list[str] | None = None) -> list[str]:
    """Return the feature names used after optional dropping."""
    stats = _load_stats(drop_features=drop_features)
    return stats["feature_names"]
