"""ONNX export for MLP models with parity check and latency benchmarking.

Usage:
  python -m netshield.models.export_onnx \\
      --run-name mlp_smoke --onnx-name netshield_mlp_smoke --set-active smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from netshield.common.config import active_profile, repo_root
from netshield.common.labels import CLASS_NAMES

logger = logging.getLogger(__name__)


def export_mlp_onnx(
    run_name: str,
    onnx_name: str,
    opset: int = 17,
) -> Path:
    """Export a trained MLP checkpoint to ONNX.

    Args:
        run_name: Checkpoint directory name under checkpoints/.
        onnx_name: Output filename stem (without .onnx).
        opset: ONNX opset version.

    Returns:
        Path to the exported .onnx file.
    """
    import torch
    from netshield.models.mlp import MLP

    root = repo_root()
    ckpt_dir = root / "checkpoints" / run_name

    # Load model metadata
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    model = MLP(
        input_dim=meta["input_dim"],
        num_classes=meta["num_classes"],
        hidden_dims=meta["hidden_dims"],
        dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    out_dir = root / "models" / "serving"
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / f"{onnx_name}.onnx"

    dummy = torch.randn(1, meta["input_dim"])

    # TorchScript-based export (dynamo=False where supported)
    export_kwargs = dict(
        input_names=["features"],
        output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
    )
    try:
        torch.onnx.export(model, dummy, str(onnx_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older PyTorch without dynamo parameter
        torch.onnx.export(model, dummy, str(onnx_path), **export_kwargs)

    size_mb = onnx_path.stat().st_size / (1024 * 1024)
    logger.info("ONNX exported: %s (%.2f MB, opset %d)", onnx_path, size_mb, opset)
    return onnx_path


def verify_parity(
    run_name: str,
    onnx_path: Path,
    n_samples: int = 10000,
    atol: float = 1e-4,
    drop_features: list[str] | None = None,
) -> float:
    """Check ONNX parity against PyTorch on val data.

    Returns max absolute difference.
    """
    import onnxruntime as ort
    import torch
    from netshield.models.mlp import MLP

    root = repo_root()
    ckpt_dir = root / "checkpoints" / run_name
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    # Load PyTorch model
    model = MLP(
        input_dim=meta["input_dim"], num_classes=meta["num_classes"],
        hidden_dims=meta["hidden_dims"], dropout=meta["dropout"],
    )
    model.load_state_dict(torch.load(ckpt_dir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()

    # Try to load real val data; fall back to random
    try:
        from netshield.models.data import load_split
        X_val, _ = load_split("val", drop_features=drop_features)
        X_sample = X_val[:n_samples]
    except Exception:
        logger.warning("Could not load val data, using random input for parity check")
        X_sample = np.random.randn(n_samples, meta["input_dim"]).astype(np.float32)

    # PyTorch inference
    with torch.no_grad():
        pt_out = model(torch.from_numpy(X_sample)).numpy()

    # ONNX Runtime inference
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"features": X_sample})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    logger.info("ONNX parity: max_abs_diff=%.2e (threshold=%.0e)", max_diff, atol)

    if max_diff > atol:
        logger.error("ONNX parity FAILED: %.2e > %.0e", max_diff, atol)
    else:
        logger.info("ONNX parity PASSED")

    return max_diff


def benchmark_ort(
    onnx_path: Path,
    input_dim: int,
    batch_sizes: list[int] | None = None,
) -> dict[str, float]:
    """Benchmark ONNX Runtime CPU latency."""
    import onnxruntime as ort
    from netshield.models.evaluate import measure_inference_latency

    if batch_sizes is None:
        batch_sizes = [1, 256, 4096]

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    max_bs = max(batch_sizes)
    X = np.random.randn(max_bs, input_dim).astype(np.float32)

    def predict(x: np.ndarray) -> np.ndarray:
        return sess.run(None, {"features": x})[0]

    return measure_inference_latency(predict, X, batch_sizes=batch_sizes)


def write_model_card(
    onnx_name: str,
    run_name: str,
    onnx_path: Path,
    latency: dict,
    parity_diff: float,
    drop_features: list[str] | None = None,
) -> Path:
    """Write <name>.model_card.json next to the ONNX file."""
    root = repo_root()
    ckpt_dir = root / "checkpoints" / run_name

    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    # Load result JSON for metrics
    result_path = root / "reports" / "tables" / f"centralized_{run_name}.json"
    metrics = {}
    feature_names = []
    if result_path.exists():
        with open(result_path) as f:
            result = json.load(f)
        metrics = result.get("test_metrics", {})
        feature_names = result.get("feature_names", [])

    card = {
        "model_name": onnx_name,
        "model_type": meta.get("model_type", "mlp"),
        "onnx_file": onnx_path.name,
        "opset": 17,
        "input_dim": meta["input_dim"],
        "hidden_dims": meta.get("hidden_dims"),
        "num_classes": meta["num_classes"],
        "features": feature_names,
        "classes": CLASS_NAMES,
        "drop_features": drop_features or [],
        "metrics": {
            "macro_f1": metrics.get("macro_f1"),
            "accuracy": metrics.get("accuracy"),
            "weighted_f1": metrics.get("weighted_f1"),
        },
        "ort_latency_ms": latency,
        "parity_max_diff": parity_diff,
        "profile": active_profile(),
        "version": "v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    card_path = onnx_path.with_suffix(".model_card.json")
    with open(card_path, "w") as f:
        json.dump(card, f, indent=2)
    logger.info("Model card written: %s", card_path)
    return card_path


def set_active(onnx_name: str, kind: str) -> Path:
    """Write models/serving/active.json pointing to the given model."""
    root = repo_root()
    serving = root / "models" / "serving"

    card_path = serving / f"{onnx_name}.model_card.json"
    card = {}
    if card_path.exists():
        with open(card_path) as f:
            card = json.load(f)

    active = {
        "model_file": f"{onnx_name}.onnx",
        "model_card": card,
        "kind": kind,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out = serving / "active.json"
    with open(out, "w") as f:
        json.dump(active, f, indent=2)
    logger.info("Active model set: %s (kind=%s)", onnx_name, kind)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Export MLP to ONNX")
    parser.add_argument("--run-name", required=True, help="Checkpoint dir name")
    parser.add_argument("--onnx-name", required=True, help="Output ONNX filename stem")
    parser.add_argument("--set-active", choices=["smoke", "production"], default=None,
                        help="Set as active serving model")
    parser.add_argument("--drop-features", nargs="*", default=None,
                        help="Features that were dropped during training")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    root = repo_root()
    ckpt_dir = root / "checkpoints" / args.run_name
    with open(ckpt_dir / "meta.json") as f:
        meta = json.load(f)

    # Export
    onnx_path = export_mlp_onnx(args.run_name, args.onnx_name)

    # Parity check
    parity_diff = verify_parity(args.run_name, onnx_path,
                                drop_features=args.drop_features)

    # Latency benchmark
    latency = benchmark_ort(onnx_path, input_dim=meta["input_dim"])

    # Model card
    write_model_card(args.onnx_name, args.run_name, onnx_path, latency,
                     parity_diff, drop_features=args.drop_features)

    # Set active
    if args.set_active:
        set_active(args.onnx_name, args.set_active)

    logger.info("Export complete: %s", onnx_path)


if __name__ == "__main__":
    main()
