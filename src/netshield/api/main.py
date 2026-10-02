"""FastAPI application: live alerts, control, FL results, experiments, model endpoints."""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from netshield.api.kafka_consumers import (
    KafkaAlertConsumer,
    KafkaFlowScorer,
    KafkaMetricsConsumer,
    MockGenerator,
    is_mock_mode,
    scoring_mode,
)
from netshield.api.state import LiveState, ModelState
from netshield.common.config import get_config, repo_root
from netshield.common.schemas import ControlEvent, FlowEvent

logger = logging.getLogger(__name__)

# ---- Global state (initialized in lifespan) ----
_live: LiveState | None = None
_model: ModelState | None = None
_consumers: list[Any] = []


def _get_live() -> LiveState:
    assert _live is not None
    return _live


def _get_model() -> ModelState:
    assert _model is not None
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _live, _model, _consumers

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    root = repo_root()
    cfg = get_config()

    _live = LiveState()
    _model = ModelState(root / "models" / "serving")

    if is_mock_mode():
        mock = MockGenerator(_live)
        mock.start()
        _consumers.append(mock)
        logger.info("Running in MOCK mode")
    else:
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
        topics = cfg.get("kafka", {}).get("topics", {})
        alerts_topic = topics.get("alerts", "iot.alerts")
        metrics_topic = topics.get("metrics", "iot.metrics")
        flows_topic = topics.get("flows", "iot.flows")
        mode = scoring_mode()

        if mode == "api":
            # Fallback: API scores iot.flows directly (no Spark needed)
            scorer = KafkaFlowScorer(
                _live, _model, bootstrap, flows_topic, alerts_topic
            )
            scorer.start()
            _consumers.append(scorer)
            logger.info("API-mode scoring started (bypass Spark)")
        else:
            # Normal: consume iot.alerts produced by Spark streaming
            alert_consumer = KafkaAlertConsumer(_live, bootstrap, alerts_topic)
            alert_consumer.start()
            _consumers.append(alert_consumer)

        metrics_consumer = KafkaMetricsConsumer(_live, bootstrap, metrics_topic)
        metrics_consumer.start()
        _consumers.append(metrics_consumer)
        logger.info("Kafka consumers started (bootstrap=%s, scoring=%s)", bootstrap, mode)

    yield

    for c in _consumers:
        c.stop()
    _consumers.clear()
    logger.info("Shutdown complete")


app = FastAPI(
    title="NetShield-FL API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS for Streamlit dashboard
cors_origins = get_config().get("api", {}).get("cors_origins", ["http://localhost:8501"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Health ----
@app.get("/health")
def health():
    model = _get_model()
    return {
        "status": "ok",
        "mock_mode": is_mock_mode(),
        "scoring_mode": scoring_mode(),
        "model_loaded": model.onnx_session is not None,
        "model_version": model.model_version,
    }


# ---- Live endpoints ----
@app.get("/live/summary")
def live_summary():
    return _get_live().get_summary()


@app.get("/live/alerts")
def live_alerts(
    limit: int = Query(50, ge=1, le=500),
    only_attacks: bool = Query(False),
    org_id: str | None = Query(None),
):
    return _get_live().get_alerts(limit=limit, only_attacks=only_attacks, org_id=org_id)


@app.get("/live/orgs")
def live_orgs():
    return _get_live().get_orgs()


# ---- Control endpoints ----
@app.post("/control/inject")
def control_inject(event: ControlEvent):
    if is_mock_mode():
        return {"status": "ignored", "reason": "mock mode"}
    try:
        from confluent_kafka import Producer

        bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
        cfg = get_config()
        topic = cfg.get("kafka", {}).get("topics", {}).get("control", "iot.control")

        cmd = event.model_dump()
        cmd["command"] = "inject"
        producer = Producer({"bootstrap.servers": bootstrap})
        producer.produce(topic, value=json.dumps(cmd).encode("utf-8"))
        producer.flush(5)
        return {"status": "sent", "command": cmd}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.post("/control/stop")
def control_stop(event: ControlEvent):
    if is_mock_mode():
        return {"status": "ignored", "reason": "mock mode"}
    try:
        from confluent_kafka import Producer

        bootstrap = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
        cfg = get_config()
        topic = cfg.get("kafka", {}).get("topics", {}).get("control", "iot.control")

        cmd = {"command": "stop", "org_id": event.org_id}
        producer = Producer({"bootstrap.servers": bootstrap})
        producer.produce(topic, value=json.dumps(cmd).encode("utf-8"))
        producer.flush(5)
        return {"status": "sent", "command": cmd}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ---- FL endpoints ----
def _fl_runs_dir() -> Path:
    return repo_root() / "reports" / "fl_runs"


@app.get("/fl/runs")
def fl_runs():
    d = _fl_runs_dir()
    if not d.exists():
        return []
    runs = []
    for f in sorted(d.glob("fl_*.json")):
        try:
            data = json.loads(f.read_text())
            runs.append({
                "run_name": data.get("run_name", f.stem),
                "method": data.get("method"),
                "alpha": data.get("alpha"),
                "mu": data.get("mu"),
                "seed": data.get("seed"),
                "profile": data.get("profile"),
                "rounds": data.get("rounds"),
                "macro_f1": data.get("test_metrics", {}).get("macro_f1"),
            })
        except Exception:
            continue
    return runs


@app.get("/fl/runs/{name}")
def fl_run_detail(name: str):
    path = _fl_runs_dir() / f"{name}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Run {name!r} not found")
    return json.loads(path.read_text())


@app.get("/fl/partitions")
def fl_partitions():
    d = _fl_runs_dir()
    if not d.exists():
        return []
    parts = []
    for f in sorted(d.glob("partition_*.json")):
        try:
            parts.append(json.loads(f.read_text()))
        except Exception:
            continue
    return parts


# ---- Experiments endpoint ----
@app.get("/experiments/summary")
def experiments_summary():
    tables_dir = repo_root() / "reports" / "tables"
    if not tables_dir.exists():
        return {"rows": [], "smoke_only": True}

    rows = []
    for f in sorted(tables_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            rows.append(data)
        except Exception:
            continue

    # Also include FL results CSV if present
    fl_csv = tables_dir / "fl_results.csv"
    if fl_csv.exists():
        import csv
        with open(fl_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append({k: _try_numeric(v) for k, v in row.items()})

    # Prefer rows with profile="lab", flag smoke-only
    has_lab = any(r.get("profile") == "lab" for r in rows)

    return {
        "rows": rows,
        "smoke_only": not has_lab,
    }


def _try_numeric(v: str) -> Any:
    """Try to parse a string as int or float."""
    try:
        return int(v)
    except (ValueError, TypeError):
        pass
    try:
        return float(v)
    except (ValueError, TypeError):
        return v


# ---- Model endpoints ----
@app.get("/model/active")
def model_active():
    model = _get_model()
    model.maybe_reload()
    return model.active_info


@app.post("/predict")
def predict(event: FlowEvent):
    model = _get_model()
    try:
        result = model.predict(event.features)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "event_id": event.event_id,
        "org_id": event.org_id,
        "device_id": event.device_id,
        **result,
    }
