"""Tests for streaming: score_batch and producer burst logic."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


# ---- score_batch unit test (no Spark) ----------------------------------------

def test_score_batch_basic():
    """score_batch should produce correct alert fields from a small DataFrame."""
    pytest.importorskip("onnxruntime")
    from netshield.common.labels import CLASS_NAMES, NUM_CLASSES
    from netshield.models.mlp import MLP
    from netshield.streaming.score import score_batch

    # Create a tiny ONNX model
    input_dim = 10
    model = MLP(input_dim, NUM_CLASSES, hidden_dims=[16, 8], dropout=0.0)
    model.eval()

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = str(Path(tmpdir) / "test.onnx")
        dummy = torch.randn(1, input_dim)
        torch.onnx.export(
            model, dummy, onnx_path,
            input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )

        feature_names = [f"f{i}" for i in range(input_dim)]
        mean = [0.0] * input_dim
        std = [1.0] * input_dim

        # Build input DataFrame
        n = 5
        features = [{f"f{i}": float(np.random.randn()) for i in range(input_dim)} for _ in range(n)]
        pdf = pd.DataFrame({
            "event_id": [f"evt-{i}" for i in range(n)],
            "ts_event": [1000000 + i for i in range(n)],
            "org_id": ["org-1"] * n,
            "device_id": ["dev-1"] * n,
            "features": features,
            "true_label": ["Benign"] * n,
        })

        result = score_batch(
            pdf, onnx_path, feature_names, mean, std,
            CLASS_NAMES, "test:model.onnx",
        )

        # Check shape and columns
        assert len(result) == n
        assert "event_id" in result.columns
        assert "pred_class" in result.columns
        assert "confidence" in result.columns
        assert "is_attack" in result.columns
        assert "model_version" in result.columns
        assert "probs" in result.columns

        # Check types
        assert all(result["pred_class"].isin(CLASS_NAMES))
        assert all(result["confidence"] >= 0)
        assert all(result["confidence"] <= 1)
        assert result["model_version"].iloc[0] == "test:model.onnx"

        # Probs should be valid JSON dicts
        for p in result["probs"]:
            d = json.loads(p)
            assert isinstance(d, dict)
            assert len(d) == NUM_CLASSES


def test_score_batch_empty():
    """score_batch should handle empty DataFrame."""
    from netshield.streaming.score import score_batch

    pdf = pd.DataFrame(columns=[
        "event_id", "ts_event", "org_id", "device_id", "features", "true_label",
    ])
    result = score_batch(pdf, "/nonexistent.onnx", [], [], [], [], "v1")
    assert len(result) == 0


# ---- Producer burst logic tests ---------------------------------------------

def test_burst_generates_events():
    """Burst should generate events with correct fields."""
    from netshield.streaming.producer import Burst

    feature_names = ["f0", "f1", "f2"]
    attack_rows = pd.DataFrame({
        "f0": [1.0, 2.0],
        "f1": [3.0, 4.0],
        "f2": [5.0, 6.0],
    })

    burst = Burst(
        org_id="org-1", attack_class="DDoS",
        rate_eps=10000,  # high rate to not block
        duration_s=10,
        attack_rows=attack_rows,
        feature_names=feature_names,
    )

    assert burst.active
    event = burst.next_event()
    assert event is not None
    assert event["org_id"] == "org-1"
    assert event["true_label"] == "DDoS"
    assert "features" in event
    assert set(event["features"].keys()) == set(feature_names)


def test_burst_expires():
    """Burst should become inactive after duration."""
    from netshield.streaming.producer import Burst

    burst = Burst(
        org_id="org-1", attack_class="DoS",
        rate_eps=10000, duration_s=0.0,  # expires immediately
        attack_rows=pd.DataFrame({"f0": [1.0]}),
        feature_names=["f0"],
    )

    assert not burst.active


def test_token_bucket_rate():
    """TokenBucket should respect approximate rate."""
    import time

    from netshield.streaming.producer import TokenBucket

    bucket = TokenBucket(rate=1000)
    count = 0
    start = time.monotonic()
    while time.monotonic() - start < 0.1:
        bucket.acquire()
        count += 1

    # Should be roughly 100 tokens in 0.1s at 1000/s
    assert 50 < count < 200, f"Expected ~100, got {count}"


def test_producer_control_handler():
    """_handle_control should create and remove bursts."""
    from unittest.mock import patch

    from netshield.streaming.producer import StreamProducer

    # Patch everything the constructor needs
    with patch("netshield.streaming.producer.get_config") as mock_cfg, \
         patch("netshield.streaming.producer.repo_root") as mock_root, \
         patch("netshield.streaming.producer.Producer"), \
         patch("pandas.read_parquet") as mock_parquet:

        mock_cfg.return_value = {
            "kafka": {
                "bootstrap_servers": "localhost:9094",
                "topics": {"flows": "iot.flows", "control": "iot.control"},
            }
        }
        mock_root.return_value = Path(tempfile.mkdtemp())

        # Create mock stream data
        mock_df = pd.DataFrame({
            "f0": [1.0, 2.0], "f1": [3.0, 4.0],
            "label_8": ["DDoS", "Benign"],
        })
        mock_parquet.return_value = mock_df

        producer = StreamProducer(rate=100, mode="replay", bootstrap="localhost:9094")
        # Don't start threads
        producer.control_thread = MagicMock()

        # Inject command
        producer._handle_control({
            "command": "inject",
            "org_id": "org-1",
            "attack_class": "DDoS",
            "rate_eps": 50,
            "duration_s": 10,
        })
        assert len(producer.bursts) == 1
        assert producer.bursts[0].org_id == "org-1"

        # Stop command
        producer._handle_control({
            "command": "stop",
            "org_id": "org-1",
        })
        assert len(producer.bursts) == 0
