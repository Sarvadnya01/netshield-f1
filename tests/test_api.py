"""Tests for the FastAPI backend (mock mode, no Kafka)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch):
    """Run all API tests in mock mode."""
    monkeypatch.setenv("MOCK_MODE", "1")


@pytest.fixture()
def client():
    """Create a TestClient with mock mode enabled."""
    os.environ["MOCK_MODE"] = "1"
    from fastapi.testclient import TestClient
    from netshield.api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["mock_mode"] is True
    assert data["model_loaded"] is True
    assert "model_version" in data


def test_live_summary(client):
    # Wait briefly for mock generator to produce some events
    time.sleep(0.3)
    r = client.get("/live/summary")
    assert r.status_code == 200
    data = r.json()
    assert "events_per_s" in data
    assert "threats_last_60s" in data
    assert "by_class" in data
    assert "latency_p50_ms" in data


def test_live_alerts(client):
    time.sleep(0.3)
    r = client.get("/live/alerts?limit=10")
    assert r.status_code == 200
    alerts = r.json()
    assert isinstance(alerts, list)
    if alerts:
        a = alerts[0]
        assert "event_id" in a
        assert "pred_class" in a
        assert "confidence" in a
        assert "probs" in a


def test_live_alerts_only_attacks(client):
    time.sleep(0.3)
    r = client.get("/live/alerts?limit=100&only_attacks=true")
    assert r.status_code == 200
    alerts = r.json()
    for a in alerts:
        assert a["is_attack"] is True


def test_live_orgs(client):
    time.sleep(0.3)
    r = client.get("/live/orgs")
    assert r.status_code == 200
    orgs = r.json()
    assert isinstance(orgs, list)


def test_control_inject_mock(client):
    r = client.post("/control/inject", json={
        "command": "inject",
        "org_id": "org-1",
        "attack_class": "DDoS",
        "rate_eps": 50,
        "duration_s": 10,
    })
    assert r.status_code == 200
    assert r.json()["status"] == "ignored"


def test_control_stop_mock(client):
    r = client.post("/control/stop", json={
        "command": "stop",
        "org_id": "org-1",
    })
    assert r.status_code == 200
    assert r.json()["status"] == "ignored"


def test_fl_runs(client):
    r = client.get("/fl/runs")
    assert r.status_code == 200
    runs = r.json()
    assert isinstance(runs, list)
    # We should have smoke FL runs from P3
    if runs:
        assert "run_name" in runs[0]
        assert "method" in runs[0]


def test_fl_run_detail(client):
    # Get list first
    runs = client.get("/fl/runs").json()
    if not runs:
        pytest.skip("No FL runs available")
    name = runs[0]["run_name"]
    r = client.get(f"/fl/runs/{name}")
    assert r.status_code == 200
    data = r.json()
    assert "test_metrics" in data
    assert "history" in data


def test_fl_run_not_found(client):
    r = client.get("/fl/runs/nonexistent_run")
    assert r.status_code == 404


def test_fl_partitions(client):
    r = client.get("/fl/partitions")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_experiments_summary(client):
    r = client.get("/experiments/summary")
    assert r.status_code == 200
    data = r.json()
    assert "rows" in data
    assert "smoke_only" in data
    assert isinstance(data["rows"], list)


def test_model_active(client):
    r = client.get("/model/active")
    assert r.status_code == 200
    data = r.json()
    assert "model_file" in data
    assert "kind" in data


def test_predict(client):
    # Build a features dict with all zeros
    from netshield.common.labels import CLASS_NAMES

    features = {f"f{i}": 0.0 for i in range(46)}
    # Use actual feature names from the model
    r = client.get("/model/active")
    model_card = r.json().get("model_card", {})
    if "features" in model_card:
        features = {name: 0.1 for name in model_card["features"]}

    r = client.post("/predict", json={
        "event_id": "test-001",
        "ts_event": int(time.time() * 1000),
        "org_id": "org-test",
        "device_id": "dev-test",
        "features": features,
    })
    assert r.status_code == 200
    data = r.json()
    assert "pred_class" in data
    assert data["pred_class"] in CLASS_NAMES
    assert "confidence" in data
    assert 0 <= data["confidence"] <= 1
    assert "probs" in data
    assert "is_attack" in data
    assert "model_version" in data


def test_model_hot_reload(client, tmp_path):
    """Verify model state detects active.json mtime change."""
    from netshield.api.main import _get_model

    model = _get_model()
    old_version = model.model_version

    # Force a reload check by resetting last_check and interval
    model._last_check = 0
    model._check_interval = 0

    # Touch active.json to change mtime
    active_path = model._active_path
    original_mtime = active_path.stat().st_mtime

    # Write same content but with new timestamp
    content = active_path.read_text()
    time.sleep(0.05)  # Ensure mtime changes
    active_path.write_text(content)

    reloaded = model.maybe_reload()
    assert reloaded is True
    # Version should still be the same content, but reload happened
    assert model.model_version == old_version


def test_openapi_schema(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    schema = r.json()
    paths = schema["paths"]
    expected = [
        "/health", "/live/summary", "/live/alerts", "/live/orgs",
        "/control/inject", "/control/stop",
        "/fl/runs", "/fl/runs/{name}", "/fl/partitions",
        "/experiments/summary", "/model/active", "/predict",
    ]
    for ep in expected:
        assert ep in paths, f"Missing endpoint: {ep}"
