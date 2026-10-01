"""Tests for netshield.common.schemas."""

from netshield.common.schemas import (
    ActiveModel,
    AlertEvent,
    ControlEvent,
    FlowEvent,
    MetricsEvent,
)


def test_flow_event():
    e = FlowEvent(
        event_id="abc-123",
        ts_event=1700000000000,
        org_id="org1",
        device_id="dev1",
        features={"f1": 1.0, "f2": 2.5},
        true_label="DDoS",
    )
    assert e.event_id == "abc-123"
    assert e.ts_event == 1700000000000
    assert e.features["f2"] == 2.5
    d = e.model_dump()
    assert isinstance(d, dict)
    assert d["true_label"] == "DDoS"


def test_flow_event_defaults():
    e = FlowEvent(event_id="x", ts_event=0, features={})
    assert e.org_id == "default"
    assert e.device_id == ""
    assert e.true_label is None


def test_alert_event():
    e = AlertEvent(
        event_id="abc-123",
        ts_event=1700000000000,
        ts_scored=1700000000050,
        latency_ms=50.0,
        pred_class="DDoS",
        confidence=0.95,
        probs={"DDoS": 0.95, "Benign": 0.05},
        is_attack=True,
        model_version="v1",
    )
    assert e.is_attack is True
    assert e.latency_ms == 50.0
    d = e.model_dump()
    assert "probs" in d


def test_metrics_event():
    e = MetricsEvent(
        window_start=1700000000000,
        window_end=1700000030000,
        total=1000,
        attacks=150,
        by_class={"DDoS": 100, "DoS": 50},
    )
    assert e.total == 1000
    assert e.by_class["DDoS"] == 100


def test_control_event():
    e = ControlEvent(
        command="inject",
        org_id="org1",
        attack_class="DDoS",
        duration_s=60.0,
        rate_eps=100.0,
    )
    assert e.command == "inject"
    assert e.duration_s == 60.0


def test_control_event_stop():
    e = ControlEvent(command="stop")
    assert e.attack_class == ""
    assert e.rate_eps == 0.0


def test_active_model():
    m = ActiveModel(
        model_file="mlp_smoke.onnx",
        model_card={"macro_f1": 0.85},
        kind="smoke",
        updated_at="2024-01-01T00:00:00Z",
    )
    assert m.kind == "smoke"
    assert m.model_card["macro_f1"] == 0.85
