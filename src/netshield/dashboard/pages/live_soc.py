"""Page 1: Live SOC — real-time alerts, KPIs, attack simulator."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from netshield.dashboard import api_client
from netshield.dashboard.components import (
    CLASS_COLORS,
    CLASS_ORDER,
    api_down_banner,
    model_badge,
)


def render() -> None:
    st.header("Live SOC")
    model_badge()

    if api_down_banner():
        return

    _sidebar_controls()
    _live_dashboard()


def _sidebar_controls() -> None:
    """Attack simulator in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Attack Simulator")

    org = st.sidebar.selectbox("Target Org", [f"org-{i}" for i in range(1, 9)], key="sim_org")
    attack = st.sidebar.selectbox(
        "Attack Class",
        ["DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"],
        key="sim_attack",
    )
    duration = st.sidebar.slider("Duration (s)", 5, 120, 30, key="sim_dur")
    rate = st.sidebar.slider("Rate (eps)", 10, 500, 100, key="sim_rate")

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Inject", use_container_width=True, type="primary"):
            result = api_client.control_inject(org, attack, rate, duration)
            if result:
                st.sidebar.success(f"Injecting {attack} at {rate} eps")
            else:
                st.sidebar.error("Failed to send inject")
    with col2:
        if st.button("Stop", use_container_width=True):
            result = api_client.control_stop(org)
            if result:
                st.sidebar.info(f"Stop sent to {org}")
            else:
                st.sidebar.error("Failed to send stop")


@st.fragment(run_every=2)
def _live_dashboard() -> None:
    """Auto-refreshing dashboard fragment."""
    summary = api_client.live_summary()
    if summary is None:
        st.warning("Could not fetch live summary")
        return

    # KPI cards
    cols = st.columns(5)
    cols[0].metric("Events/s", f"{summary.get('events_per_s', 0):.1f}")
    cols[1].metric("Threats (60s)", summary.get("threats_last_60s", 0))

    by_org = summary.get("by_org", {})
    cols[2].metric("Active Orgs", len(by_org))
    cols[3].metric("p95 Latency", f"{summary.get('latency_p95_ms', 0):.0f} ms")

    h = api_client.health()
    cols[4].metric("Model", (h or {}).get("model_version", "?").split(":")[-1][:20])

    # Two-column layout
    left, right = st.columns([2, 1])

    with left:
        _alert_table()

    with right:
        _class_donut(summary)
        _org_bars(summary)


def _alert_table() -> None:
    """Live alert table colored by class."""
    alerts = api_client.live_alerts(limit=50)
    if not alerts:
        st.info("No alerts yet")
        return

    df = pd.DataFrame(alerts)
    display_cols = ["pred_class", "confidence", "org_id", "device_id", "latency_ms", "is_attack"]
    available = [c for c in display_cols if c in df.columns]
    if not available:
        return

    df_show = df[available].copy()
    if "confidence" in df_show.columns:
        df_show["confidence"] = df_show["confidence"].round(3)
    if "latency_ms" in df_show.columns:
        df_show["latency_ms"] = df_show["latency_ms"].round(1)

    def _color_class(val: str) -> str:
        color = CLASS_COLORS.get(val, "#888")
        return f"color: {color}; font-weight: bold"

    subset = ["pred_class"] if "pred_class" in df_show.columns else []
    styled = df_show.style.map(_color_class, subset=subset)
    st.dataframe(styled, use_container_width=True, height=400)


def _class_donut(summary: dict) -> None:
    """Class distribution donut chart."""
    by_class = summary.get("by_class", {})
    if not any(by_class.values()):
        return

    labels = [c for c in CLASS_ORDER if by_class.get(c, 0) > 0]
    values = [by_class[c] for c in labels]
    colors = [CLASS_COLORS[c] for c in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.5,
        marker=dict(colors=colors),
        textinfo="label+percent",
        textfont_size=11,
    )])
    fig.update_layout(
        title="Class Distribution",
        showlegend=False,
        height=280,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    st.plotly_chart(fig, use_container_width=True)


def _org_bars(summary: dict) -> None:
    """Per-org threat bar chart."""
    by_org = summary.get("by_org", {})
    if not by_org:
        return

    df = pd.DataFrame([
        {"org_id": k, "alerts": v}
        for k, v in sorted(by_org.items())
    ])
    fig = px.bar(
        df, x="alerts", y="org_id", orientation="h",
        color_discrete_sequence=["#00d4aa"],
        title="Alerts by Org",
    )
    fig.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
        yaxis_title="",
        xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)
