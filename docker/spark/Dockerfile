FROM python:3.11-slim-bookworm

# Install Java 17
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jre-headless && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

WORKDIR /workspace

# Install Python dependencies
COPY requirements/spark.txt /tmp/spark.txt
RUN pip install --no-cache-dir -r /tmp/spark.txt

# Pre-fetch the Spark SQL Kafka connector into Ivy cache at build time
# Must match the pyspark version (3.5.x) and Scala 2.12
RUN python -c "\
from pyspark.sql import SparkSession; \
spark = SparkSession.builder \
    .master('local[1]') \
    .config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.5') \
    .config('spark.driver.memory', '512m') \
    .getOrCreate(); \
spark.stop(); \
print('Kafka connector cached successfully')"

CMD ["python", "--version"]
