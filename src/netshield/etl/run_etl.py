"""Staged, resumable ETL pipeline for CICIoT2023.

Stages:
  1. ingest_clean  — CSV ingest + cleaning → data/interim/cleaned/
  2. profile       — data profiling → reports/tables/data_profile.json
  3. split         — stratified split → data/processed/{train,val,test}/ + data/stream/
  4. feature_stats — feature statistics on train → models/serving/feature_stats.json

Usage:
  python -m netshield.etl.run_etl [--max-files N] [--force STAGE ...]
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from pathlib import Path

from netshield.common.config import get_config, repo_root

logger = logging.getLogger(__name__)

STAGE_ORDER = ["ingest_clean", "profile", "split", "feature_stats"]


def _setup_logging() -> None:
    """Configure root logger with console + file handlers."""
    root = repo_root()
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"etl_{ts}.log"

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    fh = logging.FileHandler(str(log_file))
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    logger.info("Logging to %s", log_file)


def _stage_done(marker: Path) -> bool:
    return marker.exists()


def _clear_output(path: Path) -> None:
    """Remove a directory tree or file."""
    if path.is_dir():
        shutil.rmtree(path)
        logger.info("Cleared %s", path)
    elif path.is_file():
        path.unlink()
        logger.info("Removed %s", path)


def _log_split_summary(spark, label: str, path: str) -> int:
    """Read a parquet path, log per-class counts, return total."""
    df = spark.read.parquet(path)
    total = df.count()
    class_counts = (
        df.groupBy("label_8").count().orderBy("label_8").collect()
    )
    breakdown = ", ".join(f"{r['label_8']}={r['count']:,}" for r in class_counts)
    logger.info("  %s: %d rows  [%s]", label, total, breakdown)
    return total


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser(description="NetShield-FL Spark ETL pipeline")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Limit number of CSV files to ingest (dev mode)")
    parser.add_argument("--force", nargs="*", default=[],
                        help="Force re-run of completed stage(s)")
    args = parser.parse_args()

    _setup_logging()

    cfg = get_config()
    root = repo_root()
    seed = cfg.get("etl", {}).get("seed", 42)

    interim_dir = root / cfg["data"]["interim_dir"]
    cleaned_dir = interim_dir / "cleaned"
    processed_dir = root / cfg["data"]["processed_dir"]
    stream_dir = root / cfg["data"]["stream_dir"]
    stats_path = root / cfg["data"]["feature_stats"]
    profile_path = root / "reports" / "tables" / "data_profile.json"

    # Markers for stage completion
    markers = {
        "ingest_clean": cleaned_dir / "_SUCCESS",
        "profile": profile_path,
        "split": processed_dir / "test" / "_SUCCESS",
        "feature_stats": stats_path,
    }

    # Force-clear requested stages
    force_set = set(args.force) if args.force else set()
    for stage in force_set:
        if stage not in STAGE_ORDER:
            parser.error(f"Unknown stage: {stage}. Choose from {STAGE_ORDER}")

    clear_targets = {
        "ingest_clean": [cleaned_dir],
        "profile": [profile_path, root / "reports" / "tables" / "data_profile.md"],
        "split": [processed_dir / "train", processed_dir / "val", processed_dir / "test",
                  stream_dir / "stream.parquet"],
        "feature_stats": [stats_path],
    }
    for stage in force_set:
        for t in clear_targets[stage]:
            _clear_output(t)

    # ---- Lazy imports (pyspark only available inside container) ----
    from netshield.etl.spark_session import get_spark

    spark = get_spark("netshield-etl")
    pipeline_start = time.time()
    stage_counts: dict[str, int] = {}

    # ================================================================
    # Stage 1: INGEST + CLEAN
    # ================================================================
    if _stage_done(markers["ingest_clean"]) and "ingest_clean" not in force_set:
        logger.info("Stage ingest_clean: SKIPPED (already done)")
    else:
        t0 = time.time()
        logger.info("Stage ingest_clean: STARTING")

        from netshield.etl.clean import clean
        from netshield.etl.ingest import ingest

        df_raw = ingest(spark, max_files=args.max_files)
        count_raw = df_raw.count()
        stage_counts["raw_ingested"] = count_raw
        logger.info("Raw rows ingested: %d", count_raw)

        df_clean = clean(df_raw, spark)

        # Write partitioned by label_8
        cleaned_dir.parent.mkdir(parents=True, exist_ok=True)
        df_clean.write.partitionBy("label_8").parquet(str(cleaned_dir), mode="overwrite")

        # Count from parquet
        count_clean = spark.read.parquet(str(cleaned_dir)).count()
        stage_counts["after_cleaning"] = count_clean
        logger.info(
            "Clean rows: %d (dropped %d, %.1f%%)",
            count_clean, count_raw - count_clean,
            100 * (count_raw - count_clean) / max(count_raw, 1),
        )
        logger.info("Stage ingest_clean: DONE in %.1fs", time.time() - t0)

    # Save stage counts metadata
    meta_path = interim_dir / "metadata.json"
    if stage_counts:
        with open(meta_path, "w") as f:
            json.dump(stage_counts, f, indent=2)
    elif meta_path.exists():
        with open(meta_path) as f:
            stage_counts = json.load(f)

    # ================================================================
    # Stage 2: PROFILE
    # ================================================================
    if _stage_done(markers["profile"]) and "profile" not in force_set:
        logger.info("Stage profile: SKIPPED (already done)")
    else:
        t0 = time.time()
        logger.info("Stage profile: STARTING")

        from netshield.etl.profile import build_profile, write_profile

        df_cleaned = spark.read.parquet(str(cleaned_dir))
        profile = build_profile(df_cleaned, stage_counts=stage_counts)
        write_profile(profile)

        logger.info("Stage profile: DONE in %.1fs", time.time() - t0)

    # ================================================================
    # Stage 3: SPLIT
    # ================================================================
    if _stage_done(markers["split"]) and "split" not in force_set:
        logger.info("Stage split: SKIPPED (already done)")
    else:
        t0 = time.time()
        logger.info("Stage split: STARTING")

        from netshield.etl.split import stratified_split, write_splits

        df_cleaned = spark.read.parquet(str(cleaned_dir))
        splits = stratified_split(df_cleaned, seed=seed)
        write_splits(splits)

        logger.info("Stage split: DONE in %.1fs", time.time() - t0)

    # ================================================================
    # Stage 4: FEATURE STATS
    # ================================================================
    if _stage_done(markers["feature_stats"]) and "feature_stats" not in force_set:
        logger.info("Stage feature_stats: SKIPPED (already done)")
    else:
        t0 = time.time()
        logger.info("Stage feature_stats: STARTING")

        from netshield.etl.split import compute_feature_stats, write_feature_stats

        train_path = str(processed_dir / "train")
        stats = compute_feature_stats(spark, train_path)
        write_feature_stats(stats)

        logger.info("Stage feature_stats: DONE in %.1fs", time.time() - t0)

    # ================================================================
    # Final summary
    # ================================================================
    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("ETL COMPLETE in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Per-split summary
    logger.info("Split summary:")
    for name, subdir in [
        ("stream", str(stream_dir / "stream.parquet")),
        ("train", str(processed_dir / "train")),
        ("val", str(processed_dir / "val")),
        ("test", str(processed_dir / "test")),
    ]:
        try:
            _log_split_summary(spark, name, subdir)
        except Exception:
            logger.info("  %s: not available", name)

    # Disk usage
    logger.info("Disk usage:")
    for label, path in [
        ("interim/cleaned", cleaned_dir),
        ("processed/train", processed_dir / "train"),
        ("processed/val", processed_dir / "val"),
        ("processed/test", processed_dir / "test"),
        ("stream", stream_dir / "stream.parquet"),
    ]:
        logger.info("  %s: %.1f MB", label, _dir_size_mb(path))

    spark.stop()


if __name__ == "__main__":
    main()
