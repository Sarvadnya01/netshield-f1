"""Pydantic v2 message schemas for Kafka topics and serving metadata."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FlowEvent(BaseModel):
    """A network flow event sent to iot.flows."""

    event_id: str
    ts_event: int = Field(description="Epoch milliseconds")
    org_id: str = "default"
    device_id: str = ""
    features: dict[str, float]
    true_label: str | None = None


class AlertEvent(BaseModel):
    """A scored alert sent to iot.alerts."""

    event_id: str
    ts_event: int = Field(description="Epoch milliseconds of the original flow")
    ts_scored: int = Field(description="Epoch milliseconds when scoring completed")
    latency_ms: float
    org_id: str = "default"
    device_id: str = ""
    pred_class: str
    confidence: float
    probs: dict[str, float]
    is_attack: bool
    true_label: str | None = None
    model_version: str = ""


class MetricsEvent(BaseModel):
    """Aggregated metrics over a time window, sent to iot.metrics."""

    window_start: int = Field(description="Epoch milliseconds")
    window_end: int = Field(description="Epoch milliseconds")
    org_id: str = "default"
    total: int
    attacks: int
    by_class: dict[str, int]


class ControlEvent(BaseModel):
    """Control command sent to iot.control for attack injection."""

    command: str = Field(description="'inject' or 'stop'")
    org_id: str = "default"
    attack_class: str = ""
    duration_s: float = 0.0
    rate_eps: float = 0.0


class ActiveModel(BaseModel):
    """Metadata for the currently active serving model."""

    model_file: str
    model_card: dict = Field(default_factory=dict)
    kind: str = Field(description="'smoke' or 'production'")
    updated_at: str
