"""Spark Structured Streaming: ONNX inference on iot.flows -> iot.alerts + iot.metrics.

Runs inside the spark-stream Docker container (kafka:9092).

Usage:
  python -m netshield.streaming.spark_inference
"""

from __future__ import annotations

import json
import logging
import os

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DoubleType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
)

from netshield.streaming.score import score_batch

logger = logging.getLogger(__name__)

# ---- Schema for iot.flows JSON ----
FLOW_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("ts_event", LongType()),
    StructField("org_id", StringType()),
    StructField("device_id", StringType()),
    StructField("features", MapType(StringType(), DoubleType())),
    StructField("true_label", StringType()),
])

# ---- Schema for alert output ----
ALERT_SCHEMA = StructType([
    StructField("event_id", StringType()),
    StructField("ts_event", LongType()),
    StructField("ts_scored", LongType()),
    StructField("latency_ms", DoubleType()),
    StructField("org_id", StringType()),
    StructField("device_id", StringType()),
    StructField("pred_class", StringType()),
    StructField("confidence", DoubleType()),
    StructField("probs", StringType()),  # JSON string of dict
    StructField("is_attack", BooleanType()),
    StructField("true_label", StringType()),
    StructField("model_version", StringType()),
])


def _load_model_info(
    serving_dir: str,
) -> tuple[str, list[str], list[float], list[float], list[str], str]:
    """Load active model info from serving directory.

    Returns (onnx_path, feature_names, mean, std, class_names, model_version).
    """
    with open(os.path.join(serving_dir, "active.json")) as f:
        active = json.load(f)

    onnx_path = os.path.join(serving_dir, active["model_file"])
    model_version = f"{active.get('kind', 'unknown')}:{active['model_file']}"

    with open(os.path.join(serving_dir, "feature_stats.json")) as f:
        stats = json.load(f)

    from netshield.common.labels import CLASS_NAMES
    return (
        onnx_path, stats["feature_names"], stats["mean"],
        stats["std"], CLASS_NAMES, model_version,
    )


# ---- Global session cache for ONNX (one per Python worker) ----
_CACHED_MODEL_INFO: dict | None = None


def _get_model_info() -> dict:
    """Get or cache model info for the current worker."""
    global _CACHED_MODEL_INFO
    if _CACHED_MODEL_INFO is None:
        serving_dir = "/workspace/models/serving"
        info = _load_model_info(serving_dir)
        onnx_path, feature_names, mean, std, class_names, model_version = info
        _CACHED_MODEL_INFO = {
            "onnx_path": onnx_path,
            "feature_names": feature_names,
            "mean": mean,
            "std": std,
            "class_names": class_names,
            "model_version": model_version,
        }
    return _CACHED_MODEL_INFO


def _score_partition(iterator):
    """mapInPandas function: score each partition batch."""
    info = _get_model_info()
    for pdf in iterator:
        yield score_batch(pdf, **info)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
    checkpoint_base = "/workspace/checkpoints/streaming"

    logger.info("Starting Spark Structured Streaming inference")
    logger.info("Kafka bootstrap: %s", bootstrap)

    spark = (
        SparkSession.builder
        .appName("NetShield-Streaming")
        .master("local[2]")
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .config("spark.executor.memory", "1g")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # Read from iot.flows
    flows_raw = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("subscribe", "iot.flows")
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # Parse JSON
    flows = (
        flows_raw
        .select(F.from_json(F.col("value").cast("string"), FLOW_SCHEMA).alias("data"))
        .select("data.*")
    )

    # ---- Query 1: Alerts (1s trigger) ----
    alerts = flows.mapInPandas(_score_partition, schema=ALERT_SCHEMA)

    # Convert to JSON and write to iot.alerts
    (
        alerts
        .select(
            F.col("device_id").cast("string").alias("key"),
            F.to_json(F.struct("*")).alias("value"),
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("topic", "iot.alerts")
        .option("checkpointLocation", f"{checkpoint_base}/alerts")
        .trigger(processingTime="1 second")
        .start()
    )
    logger.info("Alerts query started (1s trigger)")

    # ---- Query 2: Metrics (10s tumbling window, 30s watermark) ----
    # Re-read alerts from the scored stream for metrics aggregation
    # We use flows directly with scoring to avoid reading iot.alerts back
    scored_for_metrics = flows.mapInPandas(_score_partition, schema=ALERT_SCHEMA)

    metrics_agg = (
        scored_for_metrics
        .withColumn("event_time", (F.col("ts_event") / 1000).cast("timestamp"))
        .withWatermark("event_time", "30 seconds")
        .groupBy(
            F.window("event_time", "10 seconds"),
            F.col("org_id"),
        )
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("is_attack"), 1).otherwise(0)).alias("attacks"),
            # Per-class counts as a JSON string
            F.to_json(
                F.map_from_arrays(
                    F.collect_list("pred_class"),
                    F.collect_list(F.lit(1)),
                )
            ).alias("by_class_raw"),
        )
    )

    # Build the metrics event
    metrics_events = (
        metrics_agg
        .select(
            (F.unix_timestamp(F.col("window.start")) * 1000).cast("long").alias("window_start"),
            (F.unix_timestamp(F.col("window.end")) * 1000).cast("long").alias("window_end"),
            F.col("org_id"),
            F.col("total"),
            F.col("attacks"),
            F.col("by_class_raw").alias("by_class"),
        )
    )

    (
        metrics_events
        .select(
            F.col("org_id").cast("string").alias("key"),
            F.to_json(F.struct("*")).alias("value"),
        )
        .writeStream
        .format("kafka")
        .option("kafka.bootstrap.servers", bootstrap)
        .option("topic", "iot.metrics")
        .option("checkpointLocation", f"{checkpoint_base}/metrics")
        .trigger(processingTime="10 seconds")
        .outputMode("update")
        .start()
    )
    logger.info("Metrics query started (10s window, 30s watermark)")

    # Wait for termination
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
