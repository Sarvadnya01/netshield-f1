"""Spark session factory with profile-based configuration."""

from __future__ import annotations

import logging
import os

from pyspark.sql import SparkSession

from netshield.common.config import get_config

logger = logging.getLogger(__name__)


def get_spark(app_name: str = "netshield-etl") -> SparkSession:
    """Create or get a SparkSession with settings from the merged config profile."""
    cfg = get_config()
    spark_cfg = cfg.get("spark", {})

    driver_memory = spark_cfg.get("driver_memory", "4g")
    master = spark_cfg.get("master", "local[*]")
    shuffle_partitions = spark_cfg.get("shuffle_partitions", 200)

    # Use SPARK_LOCAL_DIR env var if set (Docker spark-tmp volume)
    local_dir = os.environ.get("SPARK_LOCAL_DIR", "/tmp/spark")

    builder = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.local.dir", local_dir)
        # Arrow for pandas interop
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Adaptive query execution
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        # Tune for many CSV files
        .config("spark.sql.files.maxPartitionBytes", "67108864")   # 64 MB
        .config("spark.sql.files.openCostInBytes", "4194304")      # 4 MB
        # Memory / spill tuning for constrained environments
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.windowExec.buffer.in.memory.threshold", "4096")
        .config("spark.sql.autoBroadcastJoinThreshold", "10485760")  # 10 MB
    )

    spark = builder.getOrCreate()
    logger.info(
        "SparkSession created: master=%s, driver_memory=%s, shuffle_partitions=%s",
        master, driver_memory, shuffle_partitions,
    )
    return spark
