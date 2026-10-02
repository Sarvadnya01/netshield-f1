"""Shared dashboard components and constants."""

from __future__ import annotations

import streamlit as st

from netshield.dashboard import api_client

CLASS_COLORS = {
    "Benign": "#2ecc71",
    "DDoS": "#e74c3c",
    "DoS": "#e67e22",
    "Mirai": "#9b59b6",
    "Recon": "#3498db",
    "Spoofing": "#f39c12",
    "Web": "#1abc9c",
    "BruteForce": "#e84393",
}

CLASS_ORDER = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]


def model_badge() -> None:
    """Show model badge in the header."""
    info = api_client.model_active()
    if info is None:
        st.warning("API unreachable")
        return
    kind = info.get("kind", "unknown")
    model_file = info.get("model_file", "unknown")
    if kind == "smoke":
        st.markdown(
            f'<span style="background:#d4a017;color:#000;padding:4px 12px;'
            f'border-radius:4px;font-weight:bold;font-size:0.85em">'
            f'SMOKE MODEL &mdash; {model_file}</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span style="background:#27ae60;color:#fff;padding:4px 12px;'
            f'border-radius:4px;font-weight:bold;font-size:0.85em">'
            f'PRODUCTION (Federated) &mdash; {model_file}</span>',
            unsafe_allow_html=True,
        )


def api_down_banner() -> bool:
    """Show a banner if the API is unreachable. Returns True if API is down."""
    h = api_client.health()
    if h is None:
        st.error(
            "**NetShield API is unreachable.** "
            "Start it with: `MOCK_MODE=1 uvicorn netshield.api.main:app --port 8000`"
        )
        return True
    return False
