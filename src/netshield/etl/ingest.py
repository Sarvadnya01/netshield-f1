"""CSV ingestion for CICIoT2023 dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

from netshield.common.config import get_config, repo_root

logger = logging.getLogger(__name__)


def _to_snake_case(name: str) -> str:
    """Convert a column name to snake_case."""
    name = name.strip().replace(" ", "_")
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = name.lower()
    name = re.sub(r"_+", "_", name)
    return name


def _discover_schema(spark: SparkSession, sample_path: str) -> tuple[StructType, str]:
    """Read one CSV header to discover columns.

    Returns (explicit_schema, original_label_column_name).
    All columns except the label are DoubleType.
    """
    sample = spark.read.csv(sample_path, header=True, inferSchema=False).limit(0)
    original_cols = sample.columns

    # Find label column (case-insensitive)
    label_col = None
    for col in original_cols:
        if col.strip().lower() == "label":
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"No 'label' column found. Columns: {original_cols}")

    fields = []
    for col in original_cols:
        if col == label_col:
            fields.append(StructField(col, StringType(), nullable=True))
        else:
            fields.append(StructField(col, DoubleType(), nullable=True))

    return StructType(fields), label_col


def ingest(spark: SparkSession, max_files: int | None = None) -> DataFrame:
    """Read CICIoT2023 CSVs into a DataFrame with snake_case columns.

    Args:
        spark: Active SparkSession.
        max_files: If set, only read this many files (for dev).

    Returns:
        DataFrame with snake_case column names and a source_file column.
    """
    cfg = get_config()
    root = repo_root()
    raw_dir = root / cfg["data"]["raw_dir"]

    csv_files = sorted(str(p) for p in Path(raw_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_dir}")

    total_files = len(csv_files)
    if max_files is not None:
        csv_files = csv_files[:max_files]
    logger.info("CSV files: %d available, using %d", total_files, len(csv_files))

    # Discover schema from first file
    schema, label_col = _discover_schema(spark, csv_files[0])
    logger.info(
        "Schema from %s: %d columns, label=%r",
        Path(csv_files[0]).name, len(schema.fields), label_col,
    )

    # Read all files with explicit schema
    df = (
        spark.read
        .schema(schema)
        .option("header", "true")
        .option("mode", "DROPMALFORMED")
        .csv(csv_files)
        .withColumn("source_file", F.input_file_name())
    )

    # Snake-case all column names
    rename_map = {}
    for col in df.columns:
        new_name = _to_snake_case(col)
        if new_name != col:
            rename_map[col] = new_name
    for old, new in rename_map.items():
        df = df.withColumnRenamed(old, new)

    logger.info("Ingested schema: %s", [f.name for f in df.schema.fields])
    return df
