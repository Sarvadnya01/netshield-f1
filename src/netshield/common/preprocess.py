"""Feature preprocessing: sign-safe log1p + standardization.

All preprocessing logic lives here. No Spark dependency.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fit_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute feature statistics (mean, std) after sign-safe log1p transform.

    Returns a dict with keys: feature_names, mean, std (all lists).
    """
    feature_names = sorted(df.select_dtypes(include=[np.number]).columns.tolist())
    arr = df[feature_names].values.astype(np.float64)

    # Sign-safe log1p: sign(x) * log1p(|x|)
    transformed = np.sign(arr) * np.log1p(np.abs(arr))

    mean = np.nanmean(transformed, axis=0).tolist()
    std = np.nanstd(transformed, axis=0).tolist()

    # Replace zero std with 1.0 to avoid division by zero
    std = [s if s > 1e-10 else 1.0 for s in std]

    return {
        "feature_names": feature_names,
        "mean": mean,
        "std": std,
    }


def transform(data: pd.DataFrame | np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    """Apply sign-safe log1p + standardization using precomputed stats.

    Args:
        data: DataFrame with named columns or ndarray with columns in stats order.
        stats: Dict from fit_stats() with feature_names, mean, std.

    Returns:
        float32 ndarray of transformed features.
    """
    feature_names = stats["feature_names"]
    mean = np.array(stats["mean"], dtype=np.float64)
    std = np.array(stats["std"], dtype=np.float64)

    if isinstance(data, pd.DataFrame):
        arr = data[feature_names].values.astype(np.float64)
    else:
        arr = np.asarray(data, dtype=np.float64)

    # Sign-safe log1p
    transformed = np.sign(arr) * np.log1p(np.abs(arr))

    # Standardize
    result = (transformed - mean) / std

    return result.astype(np.float32)


def load_active_model() -> tuple[Path, dict, dict]:
    """Load the active serving model info.

    Returns:
        (onnx_path, model_card, feature_stats)
    """
    from netshield.common.config import repo_root

    root = repo_root()
    serving_dir = root / "models" / "serving"

    # Load active model metadata
    active_path = serving_dir / "active.json"
    with open(active_path) as f:
        active = json.load(f)

    onnx_path = serving_dir / active["model_file"]
    model_card = active.get("model_card", {})

    # Load feature stats
    stats_path = serving_dir / "feature_stats.json"
    with open(stats_path) as f:
        feature_stats = json.load(f)

    return onnx_path, model_card, feature_stats
