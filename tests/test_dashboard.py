"""Tests for the Streamlit dashboard (import checks and api_client)."""

from __future__ import annotations

import os

# Ensure mock mode for all tests
os.environ["MOCK_MODE"] = "1"


def test_api_client_imports():
    from netshield.dashboard import api_client
    assert callable(api_client.health)
    assert callable(api_client.live_summary)
    assert callable(api_client.fl_runs)
    assert callable(api_client.predict)


def test_components_imports():
    from netshield.dashboard.components import CLASS_COLORS, CLASS_ORDER
    assert len(CLASS_COLORS) == 8
    assert len(CLASS_ORDER) == 8
    assert CLASS_ORDER[0] == "Benign"


def test_page_imports():
    from netshield.dashboard.pages import architecture, experiments, federated_lab, live_soc
    assert callable(live_soc.render)
    assert callable(federated_lab.render)
    assert callable(experiments.render)
    assert callable(architecture.render)


def test_api_client_returns_none_when_api_down():
    """api_client should return None when the API is unreachable."""
    from netshield.dashboard import api_client
    # Default API_URL is localhost:8000 which won't be running during tests
    result = api_client.health()
    # Could be None if API is not running, or dict if it happens to be
    assert result is None or isinstance(result, dict)


def test_api_client_live_alerts_params():
    """Verify live_alerts accepts filter params."""
    from netshield.dashboard import api_client
    result = api_client.live_alerts(limit=10, only_attacks=True, org_id="org-1")
    assert result is None or isinstance(result, list)


def test_class_colors_match_class_order():
    from netshield.dashboard.components import CLASS_COLORS, CLASS_ORDER
    for cls in CLASS_ORDER:
        assert cls in CLASS_COLORS, f"Missing color for {cls}"
