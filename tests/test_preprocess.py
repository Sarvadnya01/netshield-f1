"""Tests for netshield.common.preprocess."""

import numpy as np
import pandas as pd

from netshield.common.preprocess import fit_stats, transform


def test_fit_stats_basic():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    stats = fit_stats(df)
    assert stats["feature_names"] == ["a", "b"]
    assert len(stats["mean"]) == 2
    assert len(stats["std"]) == 2
    # Std should be positive
    assert all(s > 0 for s in stats["std"])


def test_transform_dataframe():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    stats = fit_stats(df)
    result = transform(df, stats)
    assert result.dtype == np.float32
    assert result.shape == (3, 2)
    # After standardization, mean should be ~0
    assert abs(result.mean()) < 0.5


def test_transform_ndarray():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    stats = fit_stats(df)
    arr = df[stats["feature_names"]].values
    result = transform(arr, stats)
    assert result.dtype == np.float32
    assert result.shape == (3, 2)


def test_sign_safe_log1p():
    """Negative values should be handled correctly via sign-safe log1p."""
    df = pd.DataFrame({"x": [-100.0, 0.0, 100.0]})
    stats = fit_stats(df)
    result = transform(df, stats)
    assert result.dtype == np.float32
    # sign(-100)*log1p(100) should be negative, 0 stays ~0, positive stays positive
    assert result[0, 0] < result[1, 0] < result[2, 0]


def test_zero_std_handling():
    """A constant column should get std=1.0 to avoid division by zero."""
    df = pd.DataFrame({"const": [5.0, 5.0, 5.0], "vary": [1.0, 2.0, 3.0]})
    stats = fit_stats(df)
    # The constant column should have std replaced with 1.0
    const_idx = stats["feature_names"].index("const")
    assert stats["std"][const_idx] == 1.0
    # Transform should not produce NaN/Inf
    result = transform(df, stats)
    assert np.all(np.isfinite(result))
