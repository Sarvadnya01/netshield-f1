"""Background Kafka consumers for iot.alerts and iot.metrics.

Supports --mock / MOCK_MODE=1 for development without Kafka.
SCORING_MODE=api enables fallback scoring: the API consumes iot.flows
and scores with the active ONNX model directly, bypassing Spark.
"""

from __future__ import annotations

import json
import logging
import os
import random
import threading
import time
import uuid

from netshield.api.state import LiveState, ModelState
from netshield.common.labels import CLASS_NAMES

logger = logging.getLogger(__name__)


def is_mock_mode() -> bool:
    return os.environ.get("MOCK_MODE", "0") == "1"


def scoring_mode() -> str:
    """Return 'spark' or 'api'. Default 'spark'."""
    return os.environ.get("SCORING_MODE", "spark").lower()


class MockGenerator:
    """Generate synthetic AlertEvents for development."""

    def __init__(self, state: LiveState, rate_eps: float = 20.0) -> None:
        self._state = state
        self._rate_eps = rate_eps
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._orgs = [f"org-{i}" for i in range(1, 9)]
        self._devices = {
            org: [f"dev-{org[-1]}-{j}" for j in range(1, random.randint(3, 6))]
            for org in self._orgs
        }

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="mock-gen")
        self._thread.start()
        logger.info("Mock generator started (%.0f eps)", self._rate_eps)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        interval = 1.0 / self._rate_eps if self._rate_eps > 0 else 0.05
        while not self._stop.is_set():
            org = random.choice(self._orgs)
            device = random.choice(self._devices[org])
            is_attack = random.random() < 0.3
            pred_class = random.choice(CLASS_NAMES[1:]) if is_attack else "Benign"

            probs = {c: random.random() * 0.1 for c in CLASS_NAMES}
            probs[pred_class] = 0.6 + random.random() * 0.35
            total = sum(probs.values())
            probs = {c: v / total for c, v in probs.items()}

            now_ms = int(time.time() * 1000)
            latency = random.uniform(50, 500)

            alert = {
                "event_id": str(uuid.uuid4()),
                "ts_event": int(now_ms - latency),
                "ts_scored": now_ms,
                "latency_ms": round(latency, 1),
                "org_id": org,
                "device_id": device,
                "pred_class": pred_class,
                "confidence": round(probs[pred_class], 4),
                "probs": probs,
                "is_attack": is_attack,
                "true_label": None,
                "model_version": "mock:synthetic",
            }
            self._state.push_alert(alert)
            self._stop.wait(interval)


class KafkaAlertConsumer:
    """Background consumer for iot.alerts topic."""

    def __init__(self, state: LiveState, bootstrap: str, topic: str) -> None:
        self._state = state
        self._bootstrap = bootstrap
        self._topic = topic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kafka-alerts"
        )
        self._thread.start()
        logger.info("Kafka alert consumer started (%s)", self._topic)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        from confluent_kafka import Consumer

        consumer = Consumer({
            "bootstrap.servers": self._bootstrap,
            "group.id": "netshield-api-alerts",
            "auto.offset.reset": "latest",
        })
        consumer.subscribe([self._topic])

        try:
            while not self._stop.is_set():
                msg = consumer.poll(1.0)
                if msg is None or msg.error():
                    continue
                try:
                    data = json.loads(msg.value().decode("utf-8"))
                    # Parse probs from JSON string if needed
                    if isinstance(data.get("probs"), str):
                        data["probs"] = json.loads(data["probs"])
                    self._state.push_alert(data)
                except Exception:
                    logger.warning("Bad alert message", exc_info=True)
        finally:
            consumer.close()


class KafkaMetricsConsumer:
    """Background consumer for iot.metrics topic."""

    def __init__(self, state: LiveState, bootstrap: str, topic: str) -> None:
        self._state = state
        self._bootstrap = bootstrap
        self._topic = topic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="kafka-metrics"
        )
        self._thread.start()
        logger.info("Kafka metrics consumer started (%s)", self._topic)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        from confluent_kafka import Consumer

        consumer = Consumer({
            "bootstrap.servers": self._bootstrap,
            "group.id": "netshield-api-metrics",
            "auto.offset.reset": "latest",
        })
        consumer.subscribe([self._topic])

        try:
            while not self._stop.is_set():
                msg = consumer.poll(1.0)
                if msg is None or msg.error():
                    continue
                try:
                    data = json.loads(msg.value().decode("utf-8"))
                    if isinstance(data.get("by_class"), str):
                        data["by_class"] = json.loads(data["by_class"])
                    self._state.push_metrics(data)
                except Exception:
                    logger.warning("Bad metrics message", exc_info=True)
        finally:
            consumer.close()


class KafkaFlowScorer:
    """Fallback scorer: consume iot.flows, score with ONNX, produce iot.alerts.

    Used when SCORING_MODE=api (Spark streaming is not running).
    """

    def __init__(
        self,
        state: LiveState,
        model: ModelState,
        bootstrap: str,
        flows_topic: str,
        alerts_topic: str,
    ) -> None:
        self._state = state
        self._model = model
        self._bootstrap = bootstrap
        self._flows_topic = flows_topic
        self._alerts_topic = alerts_topic
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="api-flow-scorer"
        )
        self._thread.start()
        logger.info(
            "API flow scorer started (%s -> %s)", self._flows_topic, self._alerts_topic
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        from confluent_kafka import Consumer, Producer

        consumer = Consumer({
            "bootstrap.servers": self._bootstrap,
            "group.id": "netshield-api-scorer",
            "auto.offset.reset": "latest",
        })
        consumer.subscribe([self._flows_topic])

        producer = Producer({
            "bootstrap.servers": self._bootstrap,
            "linger.ms": 5,
        })

        try:
            while not self._stop.is_set():
                msg = consumer.poll(0.5)
                if msg is None or msg.error():
                    continue
                try:
                    flow = json.loads(msg.value().decode("utf-8"))
                    features = flow.get("features", {})
                    result = self._model.predict(features)

                    ts_scored = int(time.time() * 1000)
                    ts_event = int(flow.get("ts_event", ts_scored))

                    alert = {
                        "event_id": flow.get("event_id", str(uuid.uuid4())),
                        "ts_event": ts_event,
                        "ts_scored": ts_scored,
                        "latency_ms": float(ts_scored - ts_event),
                        "org_id": flow.get("org_id", ""),
                        "device_id": flow.get("device_id", ""),
                        "pred_class": result["pred_class"],
                        "confidence": result["confidence"],
                        "probs": result["probs"],
                        "is_attack": result["is_attack"],
                        "true_label": flow.get("true_label"),
                        "model_version": result["model_version"],
                    }

                    # Push to live state
                    self._state.push_alert(alert)

                    # Produce to iot.alerts
                    alert_out = dict(alert)
                    alert_out["probs"] = json.dumps(alert_out["probs"])
                    producer.produce(
                        self._alerts_topic,
                        value=json.dumps(alert_out).encode("utf-8"),
                        key=alert["device_id"].encode("utf-8"),
                    )
                    producer.poll(0)
                except Exception:
                    logger.warning("Flow scoring error", exc_info=True)
        finally:
            producer.flush(5)
            consumer.close()
