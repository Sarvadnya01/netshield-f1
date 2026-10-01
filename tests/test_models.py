"""Tests for models: MLP shape, no BatchNorm, ONNX parity, active.json."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")


# ---- MLP tests -------------------------------------------------------------

def test_mlp_forward_shape():
    from netshield.models.mlp import MLP

    model = MLP(input_dim=46, num_classes=8, hidden_dims=[256, 128, 64], dropout=0.3)
    x = torch.randn(32, 46)
    out = model(x)
    assert out.shape == (32, 8)


def test_mlp_single_sample():
    from netshield.models.mlp import MLP

    model = MLP(input_dim=10, num_classes=4, hidden_dims=[16, 8])
    x = torch.randn(1, 10)
    out = model(x)
    assert out.shape == (1, 4)


def test_mlp_no_batchnorm():
    """MLP must use LayerNorm, never BatchNorm."""
    from netshield.models.mlp import MLP

    model = MLP(input_dim=46, num_classes=8, hidden_dims=[256, 128, 64])
    for name, module in model.named_modules():
        assert not isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)), (
            f"Found BatchNorm at {name}"
        )
    # Verify LayerNorm is present
    ln_count = sum(1 for _, m in model.named_modules() if isinstance(m, torch.nn.LayerNorm))
    assert ln_count == 3  # one per hidden layer


def test_mlp_get_set_weights():
    from netshield.models.mlp import MLP

    model = MLP(input_dim=10, num_classes=4, hidden_dims=[16, 8])
    weights = model.get_weights()
    assert isinstance(weights, dict)
    assert all(isinstance(v, torch.Tensor) for v in weights.values())
    assert all(v.device == torch.device("cpu") for v in weights.values())

    # Modify and set back
    model2 = MLP(input_dim=10, num_classes=4, hidden_dims=[16, 8])
    model2.set_weights(weights)

    for k in weights:
        assert torch.allclose(model.state_dict()[k], model2.state_dict()[k])


# ---- ONNX parity test ------------------------------------------------------

def test_onnx_parity_random():
    """Export a small MLP to ONNX and verify parity with PyTorch."""
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from netshield.models.mlp import MLP

    model = MLP(input_dim=20, num_classes=8, hidden_dims=[32, 16], dropout=0.0)
    model.eval()

    X = np.random.randn(100, 20).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_out = model(torch.from_numpy(X)).numpy()

    # Export to ONNX
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "test_model.onnx"
        dummy = torch.randn(1, 20)

        export_kwargs = dict(
            input_names=["features"], output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=17,
        )
        try:
            torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)

        # ONNX Runtime inference
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"features": X})[0]

    max_diff = np.max(np.abs(pt_out - ort_out))
    assert max_diff < 1e-4, f"ONNX parity failed: max_diff={max_diff}"


# ---- active.json round-trip -------------------------------------------------

def test_active_json_roundtrip():
    from netshield.common.schemas import ActiveModel

    active = ActiveModel(
        model_file="netshield_mlp_smoke.onnx",
        model_card={"macro_f1": 0.85, "accuracy": 0.90},
        kind="smoke",
        updated_at="2024-01-01T00:00:00Z",
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "active.json"
        with open(path, "w") as f:
            json.dump(active.model_dump(), f)

        with open(path) as f:
            loaded = json.load(f)

        reloaded = ActiveModel(**loaded)
        assert reloaded.model_file == "netshield_mlp_smoke.onnx"
        assert reloaded.kind == "smoke"
        assert reloaded.model_card["macro_f1"] == 0.85
