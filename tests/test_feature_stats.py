"""Test that the Spark feature-stats algorithm matches preprocess.fit_stats.

The Spark code (split.compute_feature_stats) computes:
  1. sign-safe log1p: signum(x) * log1p(abs(x))
  2. population mean and stddev

This must produce the same results as preprocess.fit_stats (numpy-based).
"""

import numpy as np
import pandas as pd

from netshield.common.preprocess import fit_stats


def _compute_stats_like_spark(df: pd.DataFrame) -> dict:
    """Reproduce the Spark feature-stats algorithm in pure numpy.

    This mirrors what compute_feature_stats does in Spark:
    - signum(x) * log1p(abs(x))
    - mean (population)
    - stddev_pop (population std, ddof=0)
    """
    feature_names = sorted(df.select_dtypes(include=[np.number]).columns.tolist())
    arr = df[feature_names].values.astype(np.float64)

    transformed = np.sign(arr) * np.log1p(np.abs(arr))

    mean = np.mean(transformed, axis=0).tolist()
    std = np.std(transformed, axis=0, ddof=0).tolist()
    std = [s if s > 1e-10 else 1.0 for s in std]

    return {"feature_names": feature_names, "mean": mean, "std": std}


def test_parity_basic():
    """fit_stats and the Spark algorithm agree on simple data."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "feat_c": [-5.0, 0.0, 5.0, 10.0, 15.0],
    })
    pandas_stats = fit_stats(df)
    spark_stats = _compute_stats_like_spark(df)

    assert pandas_stats["feature_names"] == spark_stats["feature_names"]
    np.testing.assert_allclose(pandas_stats["mean"], spark_stats["mean"], atol=1e-12)
    np.testing.assert_allclose(pandas_stats["std"], spark_stats["std"], atol=1e-12)


def test_parity_with_negatives():
    """Sign-safe log1p handles negative values correctly."""
    df = pd.DataFrame({
        "x": [-1000.0, -1.0, 0.0, 1.0, 1000.0],
        "y": [0.001, 0.01, 0.1, 1.0, 10.0],
    })
    pandas_stats = fit_stats(df)
    spark_stats = _compute_stats_like_spark(df)

    assert pandas_stats["feature_names"] == spark_stats["feature_names"]
    np.testing.assert_allclose(pandas_stats["mean"], spark_stats["mean"], atol=1e-12)
    np.testing.assert_allclose(pandas_stats["std"], spark_stats["std"], atol=1e-12)


def test_parity_constant_column():
    """Constant columns get std=1.0 in both implementations."""
    df = pd.DataFrame({
        "const": [7.0, 7.0, 7.0, 7.0],
        "vary": [1.0, 2.0, 3.0, 4.0],
    })
    pandas_stats = fit_stats(df)
    spark_stats = _compute_stats_like_spark(df)

    const_idx = pandas_stats["feature_names"].index("const")
    assert pandas_stats["std"][const_idx] == 1.0
    assert spark_stats["std"][const_idx] == 1.0
    np.testing.assert_allclose(pandas_stats["mean"], spark_stats["mean"], atol=1e-12)


def test_parity_large_random():
    """Parity on a larger random dataset."""
    rng = np.random.RandomState(42)
    n = 10000
    df = pd.DataFrame({
        f"f{i}": rng.randn(n) * 100 for i in range(20)
    })
    pandas_stats = fit_stats(df)
    spark_stats = _compute_stats_like_spark(df)

    assert pandas_stats["feature_names"] == spark_stats["feature_names"]
    np.testing.assert_allclose(pandas_stats["mean"], spark_stats["mean"], atol=1e-10)
    np.testing.assert_allclose(pandas_stats["std"], spark_stats["std"], atol=1e-10)
