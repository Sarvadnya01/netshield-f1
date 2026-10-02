"""Thread-safe in-memory state: ring buffers and rolling counters for live data."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from netshield.common.labels import CLASS_NAMES


class LiveState:
    """Thread-safe ring buffers and rolling counters for the dashboard."""

    def __init__(self, maxlen: int = 5000, window_s: float = 60.0) -> None:
        self._lock = threading.Lock()
        self.alerts: deque[dict] = deque(maxlen=maxlen)
        self.metrics: deque[dict] = deque(maxlen=maxlen)
        self._window_s = window_s

        # Rolling counters
        self._event_times: deque[float] = deque(maxlen=maxlen)
        self._threat_times: deque[float] = deque(maxlen=maxlen)
        self._class_counts: dict[str, int] = {c: 0 for c in CLASS_NAMES}
        self._org_counts: dict[str, int] = {}
        self._latencies: deque[float] = deque(maxlen=maxlen)

    def push_alert(self, alert: dict) -> None:
        now = time.time()
        with self._lock:
            self.alerts.appendleft(alert)
            self._event_times.append(now)
            self._latencies.append(alert.get("latency_ms", 0.0))
            pred = alert.get("pred_class", "")
            if pred in self._class_counts:
                self._class_counts[pred] += 1
            org = alert.get("org_id", "")
            self._org_counts[org] = self._org_counts.get(org, 0) + 1
            if alert.get("is_attack"):
                self._threat_times.append(now)

    def push_metrics(self, event: dict) -> None:
        with self._lock:
            self.metrics.appendleft(event)

    def get_summary(self) -> dict:
        now = time.time()
        cutoff = now - self._window_s
        with self._lock:
            recent_events = sum(1 for t in self._event_times if t > cutoff)
            recent_threats = sum(1 for t in self._threat_times if t > cutoff)
            eps = recent_events / self._window_s if self._window_s > 0 else 0

            lats = list(self._latencies)
            if lats:
                lat_arr = np.array(lats[-500:])
                p50 = float(np.percentile(lat_arr, 50))
                p95 = float(np.percentile(lat_arr, 95))
            else:
                p50 = 0.0
                p95 = 0.0

            return {
                "events_per_s": round(eps, 1),
                "threats_last_60s": recent_threats,
                "total_alerts": len(self.alerts),
                "by_class": dict(self._class_counts),
                "by_org": dict(self._org_counts),
                "latency_p50_ms": round(p50, 1),
                "latency_p95_ms": round(p95, 1),
            }

    def get_alerts(
        self,
        limit: int = 50,
        only_attacks: bool = False,
        org_id: str | None = None,
    ) -> list[dict]:
        with self._lock:
            out: list[dict] = []
            for a in self.alerts:
                if only_attacks and not a.get("is_attack"):
                    continue
                if org_id and a.get("org_id") != org_id:
                    continue
                out.append(a)
                if len(out) >= limit:
                    break
            return out

    def get_orgs(self) -> list[dict]:
        with self._lock:
            return [
                {"org_id": org, "alert_count": cnt}
                for org, cnt in sorted(self._org_counts.items())
            ]


class ModelState:
    """Hot-reloadable ONNX model state."""

    def __init__(self, serving_dir: Path) -> None:
        self.serving_dir = serving_dir
        self._lock = threading.Lock()
        self._active_path = serving_dir / "active.json"
        self._stats_path = serving_dir / "feature_stats.json"
        self._mtime: float = 0.0
        self._last_check: float = 0.0
        self._check_interval = 10.0

        self.onnx_session: ort.InferenceSession | None = None
        self.feature_names: list[str] = []
        self.mean: list[float] = []
        self.std: list[float] = []
        self.model_version: str = ""
        self.active_info: dict = {}

        self._load()

    def _load(self) -> None:
        if not self._active_path.exists():
            return
        with open(self._active_path) as f:
            active = json.load(f)
        self.active_info = active
        onnx_path = self.serving_dir / active["model_file"]
        self.model_version = f"{active.get('kind', 'unknown')}:{active['model_file']}"

        if self._stats_path.exists():
            with open(self._stats_path) as f:
                stats = json.load(f)
            self.feature_names = stats["feature_names"]
            self.mean = stats["mean"]
            self.std = stats["std"]

        if onnx_path.exists():
            self.onnx_session = ort.InferenceSession(
                str(onnx_path), providers=["CPUExecutionProvider"]
            )
        self._mtime = self._active_path.stat().st_mtime

    def maybe_reload(self) -> bool:
        """Check if active.json changed and reload if needed. Returns True if reloaded."""
        now = time.time()
        if now - self._last_check < self._check_interval:
            return False
        self._last_check = now

        if not self._active_path.exists():
            return False
        current_mtime = self._active_path.stat().st_mtime
        if current_mtime == self._mtime:
            return False

        with self._lock:
            self._load()
        return True

    def predict(self, features: dict[str, float]) -> dict[str, Any]:
        """Run inference on a single flow event's features dict."""
        self.maybe_reload()

        with self._lock:
            if self.onnx_session is None:
                raise RuntimeError("No ONNX model loaded")

            mean_arr = np.array(self.mean, dtype=np.float64)
            std_arr = np.array(self.std, dtype=np.float64)

            x = np.array(
                [[features.get(name, 0.0) for name in self.feature_names]],
                dtype=np.float64,
            )
            x_transformed = np.sign(x) * np.log1p(np.abs(x))
            x_normed = ((x_transformed - mean_arr) / std_arr).astype(np.float32)

            ts_start = time.time() * 1000
            logits = self.onnx_session.run(None, {"features": x_normed})[0]
            ts_scored = int(time.time() * 1000)

            exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs_arr = exp_logits / exp_logits.sum(axis=1, keepdims=True)

            pred_idx = int(probs_arr.argmax(axis=1)[0])
            confidence = float(probs_arr.max(axis=1)[0])
            pred_class = CLASS_NAMES[pred_idx]
            prob_dict = {
                CLASS_NAMES[j]: float(probs_arr[0, j])
                for j in range(len(CLASS_NAMES))
            }

            return {
                "pred_class": pred_class,
                "confidence": confidence,
                "probs": prob_dict,
                "is_attack": pred_class != "Benign",
                "model_version": self.model_version,
                "inference_ms": round(ts_scored - ts_start, 2),
            }
