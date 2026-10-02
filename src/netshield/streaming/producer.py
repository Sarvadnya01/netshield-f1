"""Kafka producer: replay stream data with attack injection via iot.control.

Usage:
  python -m netshield.streaming.producer --rate 200
  python -m netshield.streaming.producer --rate 200 --mode scenario
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import threading
import time
import uuid

import numpy as np
import pandas as pd
from confluent_kafka import Consumer, Producer

from netshield.common.config import get_config, repo_root
from netshield.common.labels import CLASS_NAMES

logger = logging.getLogger(__name__)


class TokenBucket:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate: float):
        self.rate = rate
        self.tokens = 0.0
        self.max_tokens = rate * 2
        self.last = time.monotonic()

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self.last
            self.tokens = min(self.max_tokens, self.tokens + elapsed * self.rate)
            self.last = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            time.sleep(0.001)


class Burst:
    """Tracks an active attack injection burst."""

    def __init__(self, org_id: str, attack_class: str, rate_eps: float,
                 duration_s: float, attack_rows: pd.DataFrame, feature_names: list[str]):
        self.org_id = org_id
        self.attack_class = attack_class
        self.rate_eps = rate_eps
        self.end_time = time.monotonic() + duration_s
        self.attack_rows = attack_rows
        self.feature_names = feature_names
        self.bucket = TokenBucket(rate_eps)
        self.idx = 0

    @property
    def active(self) -> bool:
        return time.monotonic() < self.end_time

    def next_event(self) -> dict | None:
        if not self.active or len(self.attack_rows) == 0:
            return None
        self.bucket.acquire()
        row = self.attack_rows.iloc[self.idx % len(self.attack_rows)]
        self.idx += 1
        features = {name: float(row[name]) for name in self.feature_names}
        device_id = f"burst-{self.org_id}-{self.attack_class}"
        return {
            "event_id": str(uuid.uuid4()),
            "ts_event": int(time.time() * 1000),
            "org_id": self.org_id,
            "device_id": device_id,
            "features": features,
            "true_label": self.attack_class,
        }


class StreamProducer:
    """Kafka producer with replay and scenario modes."""

    def __init__(self, rate: float, mode: str = "replay", bootstrap: str = "localhost:9094"):
        cfg = get_config()
        kafka_cfg = cfg.get("kafka", {})
        topics = kafka_cfg.get("topics", {})
        self.flows_topic = topics.get("flows", "iot.flows")
        self.control_topic = topics.get("control", "iot.control")
        self.bootstrap = bootstrap

        self.rate = rate
        self.mode = mode
        self.bucket = TokenBucket(rate)
        self.running = True
        self.bursts: list[Burst] = []
        self.bursts_lock = threading.Lock()

        # Stats
        self.sent_count = 0
        self.burst_count = 0
        self.start_time = time.monotonic()

        # Load stream data
        root = repo_root()
        stream_path = root / "data" / "stream" / "stream.parquet"
        self.df = pd.read_parquet(stream_path)
        self.feature_names = sorted(
            self.df.select_dtypes(include=[np.number]).columns.tolist()
        )
        logger.info(
            "Loaded stream data: %d rows, %d features",
            len(self.df), len(self.feature_names),
        )

        # Assign orgs and devices
        self.orgs = [f"org-{i}" for i in range(1, 9)]
        self.devices_per_org: dict[str, list[str]] = {}
        rng = np.random.RandomState(42)
        for org in self.orgs:
            n_devices = rng.randint(3, 6)
            self.devices_per_org[org] = [f"{org}-dev-{j}" for j in range(1, n_devices + 1)]

        # Group rows by label for scenario/inject
        self.rows_by_label: dict[str, pd.DataFrame] = {}
        if "label_8" in self.df.columns:
            for label in CLASS_NAMES:
                mask = self.df["label_8"] == label
                if mask.any():
                    self.rows_by_label[label] = self.df[mask].reset_index(drop=True)

        # Kafka producer
        self.producer = Producer({
            "bootstrap.servers": self.bootstrap,
            "linger.ms": 5,
            "batch.num.messages": 100,
        })

        # Control consumer thread
        self.control_thread = threading.Thread(
            target=self._control_listener, daemon=True,
        )

    def _delivery_callback(self, err, msg):
        if err:
            logger.error("Delivery failed: %s", err)

    def _make_replay_event(self, idx: int) -> dict:
        """Build a flow event from replay data."""
        row = self.df.iloc[idx % len(self.df)]
        org = self.orgs[idx % len(self.orgs)]
        devices = self.devices_per_org[org]
        device = devices[idx % len(devices)]
        features = {name: float(row[name]) for name in self.feature_names}
        true_label = row.get("label_8", None)
        return {
            "event_id": str(uuid.uuid4()),
            "ts_event": int(time.time() * 1000),
            "org_id": org,
            "device_id": device,
            "features": features,
            "true_label": str(true_label) if true_label is not None else None,
        }

    def _make_scenario_event(self, idx: int) -> dict:
        """Build a mostly-benign event for scenario mode."""
        benign_rows = self.rows_by_label.get("Benign", self.df)
        row = benign_rows.iloc[idx % len(benign_rows)]
        org = self.orgs[idx % len(self.orgs)]
        devices = self.devices_per_org[org]
        device = devices[idx % len(devices)]
        features = {name: float(row[name]) for name in self.feature_names}
        return {
            "event_id": str(uuid.uuid4()),
            "ts_event": int(time.time() * 1000),
            "org_id": org,
            "device_id": device,
            "features": features,
            "true_label": "Benign",
        }

    def _control_listener(self) -> None:
        """Background thread that consumes iot.control for inject/stop commands."""
        consumer = Consumer({
            "bootstrap.servers": self.bootstrap,
            "group.id": "producer-control",
            "auto.offset.reset": "latest",
        })
        consumer.subscribe([self.control_topic])
        logger.info("Control listener started on %s", self.control_topic)

        while self.running:
            msg = consumer.poll(1.0)
            if msg is None or msg.error():
                continue
            try:
                cmd = json.loads(msg.value().decode("utf-8"))
                self._handle_control(cmd)
            except Exception as e:
                logger.error("Control parse error: %s", e)

        consumer.close()

    def _handle_control(self, cmd: dict) -> None:
        """Handle an inject or stop command."""
        command = cmd.get("command", "")
        org_id = cmd.get("org_id", "org-1")

        if command == "inject":
            attack_class = cmd.get("attack_class", "DDoS")
            rate_eps = cmd.get("rate_eps", 50)
            duration_s = cmd.get("duration_s", 10)

            attack_rows = self.rows_by_label.get(attack_class, pd.DataFrame())
            if len(attack_rows) == 0:
                logger.warning("No rows for attack class '%s'", attack_class)
                return

            burst = Burst(org_id, attack_class, rate_eps, duration_s,
                          attack_rows, self.feature_names)
            with self.bursts_lock:
                self.bursts.append(burst)
            logger.info(
                "INJECT: %s on %s at %.0f eps for %.0fs",
                attack_class, org_id, rate_eps, duration_s,
            )

        elif command == "stop":
            with self.bursts_lock:
                before = len(self.bursts)
                self.bursts = [b for b in self.bursts if b.org_id != org_id]
                stopped = before - len(self.bursts)
            logger.info("STOP: removed %d bursts for %s", stopped, org_id)

    def _send_burst_events(self) -> None:
        """Send events from active bursts (non-blocking: one event per burst)."""
        with self.bursts_lock:
            self.bursts = [b for b in self.bursts if b.active]
            active = list(self.bursts)

        for burst in active:
            event = burst.next_event()
            if event:
                value = json.dumps(event).encode("utf-8")
                self.producer.produce(
                    self.flows_topic, value=value,
                    key=event["device_id"].encode("utf-8"),
                    callback=self._delivery_callback,
                )
                self.burst_count += 1

    def run(self) -> None:
        """Main producer loop."""
        self.control_thread.start()
        idx = 0
        last_stats = time.monotonic()

        logger.info(
            "Producer started: mode=%s, rate=%d eps, bootstrap=%s",
            self.mode, self.rate, self.bootstrap,
        )

        while self.running:
            # Rate-limited main event
            self.bucket.acquire()

            if self.mode == "replay":
                event = self._make_replay_event(idx)
            else:
                event = self._make_scenario_event(idx)

            value = json.dumps(event).encode("utf-8")
            self.producer.produce(
                self.flows_topic, value=value,
                key=event["device_id"].encode("utf-8"),
                callback=self._delivery_callback,
            )
            self.sent_count += 1
            idx += 1

            # Send burst events (non-blocking)
            self._send_burst_events()

            # Periodic flush and stats
            if idx % 100 == 0:
                self.producer.poll(0)

            now = time.monotonic()
            if now - last_stats >= 10.0:
                elapsed = now - self.start_time
                eps = self.sent_count / max(elapsed, 1)
                logger.info(
                    "Stats: %d sent, %d bursts, %.0f eps, %.0fs elapsed",
                    self.sent_count, self.burst_count, eps, elapsed,
                )
                last_stats = now

        self.producer.flush(5)
        logger.info(
            "Producer stopped. Total: %d sent, %d burst events",
            self.sent_count, self.burst_count,
        )

    def stop(self) -> None:
        self.running = False


def main() -> None:
    parser = argparse.ArgumentParser(description="NetShield Kafka producer")
    parser.add_argument("--rate", type=int, default=200, help="Events per second")
    parser.add_argument("--mode", choices=["replay", "scenario"], default="replay",
                        help="replay=natural distribution, scenario=benign background + inject")
    parser.add_argument("--bootstrap", default=None,
                        help="Kafka bootstrap servers (default from config)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = get_config()
    bootstrap = args.bootstrap or cfg.get("kafka", {}).get("bootstrap_servers", "localhost:9094")

    producer = StreamProducer(rate=args.rate, mode=args.mode, bootstrap=bootstrap)

    def _sigint(sig, frame):
        logger.info("Ctrl+C received, stopping...")
        producer.stop()

    signal.signal(signal.SIGINT, _sigint)
    producer.run()


if __name__ == "__main__":
    main()
