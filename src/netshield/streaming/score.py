"""Pure-pandas scoring function: preprocess + ONNX inference.

No Spark or Kafka dependency — importable on the host for testing.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd


def score_batch(
    pdf: pd.DataFrame,
    onnx_path: str,
    feature_names: list[str],
    mean: list[float],
    std: list[float],
    class_names: list[str],
    model_version: str,
) -> pd.DataFrame:
    """Score a batch of flow events with the ONNX model.

    Args:
        pdf: DataFrame with columns: event_id, ts_event, org_id, device_id,
             features (dict), true_label.
        onnx_path: Path to the ONNX model file.
        feature_names: Ordered feature names.
        mean: Per-feature means for standardization.
        std: Per-feature stds for standardization.
        class_names: Ordered class names.
        model_version: Model identifier string.

    Returns:
        DataFrame with alert fields.
    """
    import onnxruntime as ort

    if len(pdf) == 0:
        return pd.DataFrame(columns=[
            "event_id", "ts_event", "ts_scored", "latency_ms",
            "org_id", "device_id", "pred_class", "confidence",
            "probs", "is_attack", "true_label", "model_version",
        ])

    mean_arr = np.array(mean, dtype=np.float64)
    std_arr = np.array(std, dtype=np.float64)

    features_list = pdf["features"].tolist()
    X = np.array(
        [[f.get(name, 0.0) for name in feature_names] for f in features_list],
        dtype=np.float64,
    )

    # Sign-safe log1p + standardize
    X_transformed = np.sign(X) * np.log1p(np.abs(X))
    X_normed = ((X_transformed - mean_arr) / std_arr).astype(np.float32)

    # ONNX inference
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"features": X_normed})[0]

    # Softmax
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs_arr = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    pred_idx = probs_arr.argmax(axis=1)
    confidence = probs_arr.max(axis=1)
    ts_scored = int(time.time() * 1000)

    results = []
    for i in range(len(pdf)):
        row = pdf.iloc[i]
        pred_class = class_names[pred_idx[i]]
        prob_dict = {class_names[j]: float(probs_arr[i, j]) for j in range(len(class_names))}
        latency = ts_scored - int(row["ts_event"])

        results.append({
            "event_id": str(row["event_id"]),
            "ts_event": int(row["ts_event"]),
            "ts_scored": ts_scored,
            "latency_ms": float(latency),
            "org_id": str(row["org_id"]),
            "device_id": str(row["device_id"]),
            "pred_class": pred_class,
            "confidence": float(confidence[i]),
            "probs": json.dumps(prob_dict),
            "is_attack": pred_class != "Benign",
            "true_label": row.get("true_label"),
            "model_version": model_version,
        })

    return pd.DataFrame(results)
