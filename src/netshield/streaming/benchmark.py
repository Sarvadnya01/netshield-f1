"""Streaming benchmark: measure throughput and latency through the pipeline.

Sends N events at max rate via producer, consumes iot.alerts, reports stats.

Usage:
  python -m netshield.streaming.benchmark --events 1000
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid

import numpy as np
import pandas as pd
from confluent_kafka import Consumer, Producer

from netshield.common.config import get_config, repo_root
from netshield.common.hardware import get_cuda_info, get_system_ram_gb
from netshield.common.preprocess import load_active_model

logger = logging.getLogger(__name__)


def run_benchmark(
    n_events: int = 1000,
    bootstrap: str = "localhost:9094",
    timeout_s: float = 60.0,
) -> dict:
    """Run the streaming benchmark.

    Args:
        n_events: Number of events to send.
        bootstrap: Kafka bootstrap servers.
        timeout_s: Max time to wait for all alerts.

    Returns:
        Dict with throughput and latency stats.
    """
    cfg = get_config()
    topics = cfg.get("kafka", {}).get("topics", {})
    flows_topic = topics.get("flows", "iot.flows")
    alerts_topic = topics.get("alerts", "iot.alerts")

    # Load stream data for sending
    root = repo_root()
    stream_path = root / "data" / "stream" / "stream.parquet"
    df = pd.read_parquet(stream_path)
    feature_names = sorted(df.select_dtypes(include=[np.number]).columns.tolist())

    # Get model info
    _, model_card, _ = load_active_model()
    model_kind = model_card.get("kind", "unknown") if isinstance(model_card, dict) else "unknown"

    # Create unique group_id for this benchmark run
    group_id = f"benchmark-{uuid.uuid4().hex[:8]}"

    producer = Producer({
        "bootstrap.servers": bootstrap,
        "linger.ms": 1,
        "batch.num.messages": 500,
    })
    consumer = Consumer({
        "bootstrap.servers": bootstrap,
        "group.id": group_id,
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([alerts_topic])

    # Drain any existing messages
    logger.info("Draining existing alerts...")
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        consumer.poll(0.5)

    # Send events
    logger.info("Sending %d events to %s...", n_events, flows_topic)
    send_start = time.monotonic()
    event_ids = set()

    for i in range(n_events):
        row = df.iloc[i % len(df)]
        features = {name: float(row[name]) for name in feature_names}
        eid = str(uuid.uuid4())
        event_ids.add(eid)
        event = {
            "event_id": eid,
            "ts_event": int(time.time() * 1000),
            "org_id": f"org-{(i % 8) + 1}",
            "device_id": f"bench-{i % 16}",
            "features": features,
            "true_label": str(row.get("label_8", "")),
        }
        producer.produce(
            flows_topic,
            value=json.dumps(event).encode("utf-8"),
            key=event["device_id"].encode("utf-8"),
        )
        if i % 200 == 0:
            producer.poll(0)

    producer.flush(10)
    send_elapsed = time.monotonic() - send_start
    send_eps = n_events / max(send_elapsed, 0.001)
    logger.info("Sent %d events in %.2fs (%.0f eps)", n_events, send_elapsed, send_eps)

    # Consume alerts
    logger.info("Consuming alerts (timeout %.0fs)...", timeout_s)
    latencies = []
    received = 0
    consume_start = time.monotonic()

    while received < n_events and (time.monotonic() - consume_start) < timeout_s:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        try:
            alert = json.loads(msg.value().decode("utf-8"))
            if alert.get("event_id") in event_ids:
                lat = alert.get("latency_ms", 0)
                latencies.append(lat)
                received += 1
        except Exception:
            pass

    consumer.close()
    consume_elapsed = time.monotonic() - consume_start

    if len(latencies) == 0:
        logger.warning("No alerts received! Is spark-stream running?")
        return {"error": "no alerts received", "events_sent": n_events}

    latencies_arr = np.array(latencies)
    throughput = received / max(consume_elapsed, 0.001)

    # Hardware info
    cuda_info = get_cuda_info()
    ram_gb = get_system_ram_gb()

    result = {
        "events_sent": n_events,
        "events_received": received,
        "receive_rate": round(received / n_events * 100, 1),
        "send_throughput_eps": round(send_eps, 1),
        "end_to_end_throughput_eps": round(throughput, 1),
        "latency_ms": {
            "p50": round(float(np.percentile(latencies_arr, 50)), 1),
            "p95": round(float(np.percentile(latencies_arr, 95)), 1),
            "p99": round(float(np.percentile(latencies_arr, 99)), 1),
            "mean": round(float(latencies_arr.mean()), 1),
            "min": round(float(latencies_arr.min()), 1),
            "max": round(float(latencies_arr.max()), 1),
        },
        "model_kind": model_kind,
        "gpu": cuda_info.get("device_name", "none"),
        "ram_gb": round(ram_gb, 1),
        "duration_s": round(consume_elapsed, 1),
    }

    logger.info(
        "Benchmark: %d/%d received, p50=%.0fms p95=%.0fms p99=%.0fms, %.0f eps",
        received, n_events,
        result["latency_ms"]["p50"],
        result["latency_ms"]["p95"],
        result["latency_ms"]["p99"],
        throughput,
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming benchmark")
    parser.add_argument("--events", type=int, default=1000, help="Number of events")
    parser.add_argument("--timeout", type=float, default=60, help="Timeout seconds")
    parser.add_argument("--bootstrap", default=None, help="Kafka bootstrap servers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = get_config()
    bootstrap = args.bootstrap or cfg.get("kafka", {}).get("bootstrap_servers", "localhost:9094")

    # Load active model info for filename
    _, model_card, _ = load_active_model()
    active_path = repo_root() / "models" / "serving" / "active.json"
    with open(active_path) as f:
        active = json.load(f)
    kind = active.get("kind", "unknown")

    result = run_benchmark(
        n_events=args.events,
        bootstrap=bootstrap,
        timeout_s=args.timeout,
    )

    # Save result
    out_dir = repo_root() / "reports" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stream_benchmark_{kind}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Results saved: %s", out_path)


if __name__ == "__main__":
    main()
