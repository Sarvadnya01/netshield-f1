"""Data profiling: class counts, feature summary, before/after report."""

from __future__ import annotations

import json
import logging

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from netshield.common.config import repo_root
from netshield.etl.clean import get_feature_cols

logger = logging.getLogger(__name__)


def build_profile(df: DataFrame, stage_counts: dict | None = None) -> dict:
    """Compute a data profile from a cleaned DataFrame.

    Args:
        df: Cleaned DataFrame with label_34, label_8, and feature columns.
        stage_counts: Optional dict of row counts at each pipeline stage.

    Returns:
        Profile dict with class counts, feature summary, and stage counts.
    """
    profile: dict = {}

    # Row counts by label_8
    class8_rows = df.groupBy("label_8").count().orderBy("label_8").collect()
    profile["class_8_counts"] = {row["label_8"]: row["count"] for row in class8_rows}
    profile["total_rows"] = sum(profile["class_8_counts"].values())

    # Row counts by label_34
    class34_rows = df.groupBy("label_34").count().orderBy("label_34").collect()
    profile["class_34_counts"] = {row["label_34"]: row["count"] for row in class34_rows}

    # Feature summary (min, max, mean, stddev, nulls) — single pass
    feature_cols = get_feature_cols(df)
    agg_exprs = []
    for col in feature_cols:
        c = F.col(col)
        agg_exprs.extend([
            F.min(c).alias(f"{col}__min"),
            F.max(c).alias(f"{col}__max"),
            F.mean(c).alias(f"{col}__mean"),
            F.stddev_pop(c).alias(f"{col}__std"),
            F.count(F.when(c.isNull() | F.isnan(c), True)).alias(f"{col}__nulls"),
        ])
    stats_row = df.agg(*agg_exprs).collect()[0]

    feature_summary = {}
    for col in feature_cols:
        def _val(suffix: str):
            v = stats_row[f"{col}__{suffix}"]
            return float(v) if v is not None else None

        feature_summary[col] = {
            "min": _val("min"),
            "max": _val("max"),
            "mean": _val("mean"),
            "std": _val("std"),
            "nulls": int(stats_row[f"{col}__nulls"]),
        }
    profile["feature_summary"] = feature_summary
    profile["num_features"] = len(feature_cols)
    profile["feature_names"] = feature_cols

    if stage_counts:
        profile["stage_counts"] = stage_counts

    return profile


def _format_markdown(profile: dict) -> str:
    """Render profile dict as markdown."""
    lines = ["# Data Profile\n"]

    if "stage_counts" in profile:
        lines.append("## Pipeline Stage Counts\n")
        lines.append("| Stage | Rows |")
        lines.append("|-------|------|")
        for stage, count in profile["stage_counts"].items():
            lines.append(f"| {stage} | {count:,} |")
        lines.append("")

    lines.append("## Summary\n")
    lines.append(f"- **Total rows (cleaned):** {profile['total_rows']:,}")
    lines.append(f"- **Features:** {profile['num_features']}")
    lines.append(f"- **8-class labels:** {len(profile['class_8_counts'])}")
    lines.append(f"- **34-class labels:** {len(profile['class_34_counts'])}\n")

    lines.append("## Class Distribution (8-class)\n")
    lines.append("| Class | Count | Fraction |")
    lines.append("|-------|-------|----------|")
    total = profile["total_rows"]
    for cls, cnt in sorted(profile["class_8_counts"].items()):
        frac = cnt / total if total > 0 else 0
        lines.append(f"| {cls} | {cnt:,} | {frac:.4f} |")
    lines.append("")

    lines.append("## Class Distribution (34-class)\n")
    lines.append("| Label | Count |")
    lines.append("|-------|-------|")
    for lbl, cnt in sorted(profile["class_34_counts"].items()):
        lines.append(f"| {lbl} | {cnt:,} |")
    lines.append("")

    return "\n".join(lines)


def write_profile(profile: dict) -> None:
    """Write profile to reports/tables/ as JSON and markdown."""
    root = repo_root()
    out_dir = root / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "data_profile.json"
    with open(json_path, "w") as f:
        json.dump(profile, f, indent=2, default=str)
    logger.info("Profile JSON written: %s", json_path)

    md_path = out_dir / "data_profile.md"
    with open(md_path, "w") as f:
        f.write(_format_markdown(profile))
    logger.info("Profile markdown written: %s", md_path)
