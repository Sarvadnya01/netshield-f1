"""Stratified splitting and feature stats computation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from netshield.common.config import get_config, repo_root
from netshield.etl.clean import get_feature_cols

logger = logging.getLogger(__name__)


def stratified_split(df: DataFrame, seed: int = 42) -> dict[str, DataFrame]:
    """Split a cleaned DataFrame into stream / train / val / test.

    Uses per-class rand() + row_number for exact stratification.

    Returns:
        Dict with keys "stream", "train", "val", "test".
    """
    cfg = get_config()
    split_cfg = cfg["data"]["split"]

    stream_frac = split_cfg["stream_fraction"]
    train_frac = split_cfg["train"]
    val_frac = split_cfg["val"]
    # test_frac is the remainder

    train_cap = cfg["data"].get("train_cap_per_class")
    vt_cap = cfg["data"].get("val_test_total_cap")

    # -- Assign per-class rank via rand + row_number --------------------------
    df = df.withColumn("_rand", F.rand(seed=seed))
    w = Window.partitionBy("label_8").orderBy("_rand")
    df = df.withColumn("_rn", F.row_number().over(w))

    # Per-class counts (8 rows, safe to collect)
    counts = df.groupBy("label_8").agg(F.count("*").alias("_cc"))
    df = df.join(F.broadcast(counts), on="label_8")
    df = df.withColumn("_frac", F.col("_rn") / F.col("_cc"))

    # -- Compute thresholds ---------------------------------------------------
    # _frac in (0, 1], first stream_frac goes to stream
    train_end = stream_frac + (1 - stream_frac) * train_frac
    val_end = train_end + (1 - stream_frac) * val_frac

    stream = df.filter(F.col("_frac") <= stream_frac)
    train = df.filter((F.col("_frac") > stream_frac) & (F.col("_frac") <= train_end))
    val = df.filter((F.col("_frac") > train_end) & (F.col("_frac") <= val_end))
    test = df.filter(F.col("_frac") > val_end)

    drop_cols = ["_rand", "_rn", "_cc", "_frac"]

    # -- Cap train per class --------------------------------------------------
    if train_cap is not None:
        w2 = Window.partitionBy("label_8").orderBy("_rand")
        train = train.withColumn("_rn2", F.row_number().over(w2))
        train = train.filter(F.col("_rn2") <= train_cap).drop("_rn2")

    # -- Cap val/test total with stratified sampling --------------------------
    for name, split_df in [("val", val), ("test", test)]:
        if vt_cap is not None:
            split_count = split_df.count()
            if split_count > vt_cap:
                frac = vt_cap / split_count
                class_fracs = {
                    row["label_8"]: min(frac, 1.0)
                    for row in split_df.groupBy("label_8").count().collect()
                }
                split_df = split_df.stat.sampleBy("label_8", class_fracs, seed=seed)
                logger.info("Capped %s from %d to ~%d", name, split_count, vt_cap)
        if name == "val":
            val = split_df
        else:
            test = split_df

    return {
        "stream": stream.drop(*drop_cols),
        "train": train.drop(*drop_cols),
        "val": val.drop(*drop_cols),
        "test": test.drop(*drop_cols),
    }


def write_splits(splits: dict[str, DataFrame]) -> dict[str, int]:
    """Write split DataFrames to parquet; return per-split row counts."""
    cfg = get_config()
    root = repo_root()
    processed = root / cfg["data"]["processed_dir"]
    stream_dir = root / cfg["data"]["stream_dir"]

    split_counts: dict[str, int] = {}

    # Stream holdout — coalesce to few files
    stream_path = str(stream_dir / "stream.parquet")
    splits["stream"].coalesce(4).write.parquet(stream_path, mode="overwrite")
    logger.info("Wrote stream holdout to %s", stream_path)

    # Train / val / test
    for name in ("train", "val", "test"):
        out = str(processed / name)
        splits[name].write.parquet(out, mode="overwrite")
        logger.info("Wrote %s to %s", name, out)

    return split_counts


def compute_feature_stats(spark: SparkSession, train_path: str) -> dict:
    """Compute feature stats on the training set in Spark.

    Applies sign-safe log1p then computes mean and population stddev.
    Result is compatible with netshield.common.preprocess.fit_stats.
    """
    df = spark.read.parquet(train_path)
    feature_cols = get_feature_cols(df)

    # Apply sign-safe log1p: signum(x) * log1p(abs(x))
    transformed = df
    for col in feature_cols:
        c = F.col(col)
        transformed = transformed.withColumn(col, F.signum(c) * F.log1p(F.abs(c)))

    # Aggregate mean and stddev_pop (population std, matches numpy default)
    agg_exprs = []
    for col in feature_cols:
        c = F.col(col)
        agg_exprs.append(F.mean(c).alias(f"mean__{col}"))
        agg_exprs.append(F.stddev_pop(c).alias(f"std__{col}"))

    row = transformed.agg(*agg_exprs).collect()[0]

    means = []
    stds = []
    for col in feature_cols:
        m = float(row[f"mean__{col}"]) if row[f"mean__{col}"] is not None else 0.0
        s = float(row[f"std__{col}"]) if row[f"std__{col}"] is not None else 0.0
        means.append(m)
        stds.append(s if s > 1e-10 else 1.0)

    stats = {
        "feature_names": feature_cols,
        "mean": means,
        "std": stds,
    }
    return stats


def write_feature_stats(stats: dict) -> Path:
    """Write feature stats to models/serving/feature_stats.json."""
    root = repo_root()
    out = root / "models" / "serving" / "feature_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Feature stats written: %s (%d features)", out, len(stats["feature_names"]))
    return out
