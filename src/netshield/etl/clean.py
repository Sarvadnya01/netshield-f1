"""Data cleaning: null/inf removal, deduplication, label mapping."""

from __future__ import annotations

import logging

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from netshield.common.config import get_config
from netshield.common.labels import _EXPLICIT_MAP, _PREFIX_RULES

logger = logging.getLogger(__name__)


def get_feature_cols(df: DataFrame) -> list[str]:
    """Return sorted list of numeric feature column names."""
    exclude = {"label", "label_34", "label_8", "source_file"}
    return sorted(c for c in df.columns if c not in exclude)


def drop_nulls_and_infs(df: DataFrame) -> DataFrame:
    """Drop rows with any null, NaN, or +/-inf in feature columns."""
    feature_cols = get_feature_cols(df)

    conditions = []
    for col in feature_cols:
        c = F.col(col)
        conditions.append(c.isNotNull() & ~F.isnan(c) & (F.abs(c) != float("inf")))

    combined = conditions[0]
    for cond in conditions[1:]:
        combined = combined & cond

    # Also require label is not null
    combined = combined & F.col("label").isNotNull() & (F.trim(F.col("label")) != "")

    return df.filter(combined)


def deduplicate(df: DataFrame, scope: str = "global") -> DataFrame:
    """Deduplicate rows.

    scope: "global" — full dropDuplicates on feature+label cols.
           "per_file" — dedup within each source_file (cheaper).
    """
    feature_cols = get_feature_cols(df)
    dedup_cols = feature_cols + ["label"]

    if scope == "global":
        logger.info("Global dedup on %d columns", len(dedup_cols))
        return df.dropDuplicates(dedup_cols)
    elif scope == "per_file":
        logger.info("Per-file dedup on %d columns", len(dedup_cols) + 1)
        return df.dropDuplicates(["source_file"] + dedup_cols)
    else:
        raise ValueError(f"Unknown dedupe_scope: {scope!r}")


def map_labels(df: DataFrame, spark: SparkSession) -> DataFrame:
    """Map fine-grained labels to 8-class using broadcast join.

    Renames 'label' → 'label_34', adds 'label_8'.
    Raises on unmapped labels.
    """
    df = df.withColumnRenamed("label", "label_34")

    # Build mapping (explicit + prefix rules combined)
    full_mapping = dict(_EXPLICIT_MAP)

    # Check for any labels present in the data that aren't in the explicit map
    distinct_labels = [
        row["label_34"]
        for row in df.select("label_34").distinct().collect()
    ]
    for lbl in distinct_labels:
        if lbl not in full_mapping:
            matched = False
            for prefix, cls in _PREFIX_RULES:
                if lbl.startswith(prefix):
                    full_mapping[lbl] = cls
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unmapped label: {lbl!r}")

    logger.info("Label mapping: %d distinct labels → 8 classes", len(full_mapping))

    mapping_rows = [(k, v) for k, v in full_mapping.items()]
    mapping_df = spark.createDataFrame(mapping_rows, ["label_34", "label_8"])

    df = df.join(F.broadcast(mapping_df), on="label_34", how="inner")

    return df


def clean(df: DataFrame, spark: SparkSession) -> DataFrame:
    """Full cleaning pipeline: drop nulls/infs → dedup → map labels."""
    cfg = get_config()
    dedupe_scope = cfg.get("etl", {}).get("dedupe_scope", "global")

    df = drop_nulls_and_infs(df)
    df = deduplicate(df, scope=dedupe_scope)
    df = map_labels(df, spark)

    return df
