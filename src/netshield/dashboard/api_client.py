"""Thin HTTP client for the NetShield API."""

from __future__ import annotations

import os
from typing import Any

import requests

API_URL = os.environ.get("API_URL", "http://localhost:8000")
_TIMEOUT = 5


def _get(path: str, **params: Any) -> Any:
    try:
        r = requests.get(f"{API_URL}{path}", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except Exception:
        return None


def _post(path: str, body: dict) -> Any:
    try:
        r = requests.post(f"{API_URL}{path}", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.ConnectionError:
        return None
    except Exception:
        return None


def health() -> dict | None:
    return _get("/health")


def live_summary() -> dict | None:
    return _get("/live/summary")


def live_alerts(
    limit: int = 50, only_attacks: bool = False, org_id: str | None = None,
) -> list | None:
    params: dict[str, Any] = {"limit": limit, "only_attacks": only_attacks}
    if org_id:
        params["org_id"] = org_id
    return _get("/live/alerts", **params)


def live_orgs() -> list | None:
    return _get("/live/orgs")


def control_inject(
    org_id: str, attack_class: str, rate_eps: float, duration_s: float,
) -> dict | None:
    return _post("/control/inject", {
        "command": "inject",
        "org_id": org_id,
        "attack_class": attack_class,
        "rate_eps": rate_eps,
        "duration_s": duration_s,
    })


def control_stop(org_id: str) -> dict | None:
    return _post("/control/stop", {"command": "stop", "org_id": org_id})


def fl_runs() -> list | None:
    return _get("/fl/runs")


def fl_run_detail(name: str) -> dict | None:
    return _get(f"/fl/runs/{name}")


def fl_partitions() -> list | None:
    return _get("/fl/partitions")


def experiments_summary() -> dict | None:
    return _get("/experiments/summary")


def model_active() -> dict | None:
    return _get("/model/active")


def predict(event: dict) -> dict | None:
    return _post("/predict", event)
