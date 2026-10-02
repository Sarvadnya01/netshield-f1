"""Page 2: Federated Lab — FL run analysis, partition heatmaps, privacy card."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from netshield.dashboard import api_client
from netshield.dashboard.components import (
    CLASS_ORDER,
    api_down_banner,
    model_badge,
)


def render() -> None:
    st.header("Federated Lab")
    model_badge()

    if api_down_banner():
        return

    runs = api_client.fl_runs()
    if runs is None:
        st.error("Could not fetch FL runs")
        return

    if not runs:
        st.info("No FL runs found. Run the federated experiment first.")
        return

    # Check if all runs are smoke-only
    all_smoke = all(r.get("profile") == "laptop" for r in runs)
    if all_smoke:
        st.warning(
            "Only **smoke-profile** runs available (laptop, reduced data). "
            "Full-scale results will appear after the lab session."
        )

    _run_detail_section(runs)
    st.markdown("---")
    _comparison_section(runs)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        _partition_section()
    with col2:
        _privacy_card(runs)


def _run_detail_section(runs: list[dict]) -> None:
    """Run selector with round slider and convergence animation."""
    run_names = [r["run_name"] for r in runs]
    selected = st.selectbox("Select FL Run", run_names)
    if not selected:
        return

    detail = api_client.fl_run_detail(selected)
    if detail is None:
        st.error("Could not load run detail")
        return

    history = detail.get("history", [])
    if not history:
        st.info("No round history available")
        return

    max_round = len(history)
    round_idx = st.slider("Round", 1, max_round, max_round, key="fl_round") - 1

    # Global macro-F1 convergence
    rounds = list(range(1, max_round + 1))
    f1s = [h.get("global_val_macro_f1", 0) for h in history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rounds, y=f1s, mode="lines+markers",
        name="Global Val Macro-F1",
        line=dict(color="#00d4aa", width=2),
        marker=dict(size=8),
    ))
    # Highlight selected round
    fig.add_trace(go.Scatter(
        x=[rounds[round_idx]], y=[f1s[round_idx]],
        mode="markers", marker=dict(size=14, color="#e74c3c", symbol="star"),
        name=f"Round {round_idx + 1}",
    ))
    fig.update_layout(
        title="Global Macro-F1 Convergence",
        xaxis_title="Round", yaxis_title="Macro-F1",
        height=300, margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-client F1 bars at selected round
    rd = history[round_idx]
    client_f1s = rd.get("client_f1s", [])
    if client_f1s:
        df_clients = pd.DataFrame({
            "Client": [f"Client {i}" for i in range(len(client_f1s))],
            "Macro-F1": client_f1s,
        })
        fig2 = px.bar(
            df_clients, x="Client", y="Macro-F1",
            color_discrete_sequence=["#3498db"],
            title=f"Per-Client F1 at Round {round_idx + 1}",
        )
        fig2.update_layout(
            height=280, margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0", yaxis_range=[0, max(client_f1s) * 1.3 + 0.01],
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Round stats
    cols = st.columns(4)
    cols[0].metric("Global F1", f"{rd.get('global_val_macro_f1', 0):.4f}")
    cols[1].metric("Mean Client F1", f"{rd.get('mean_client_f1', 0):.4f}")
    cols[2].metric("Worst Client F1", f"{rd.get('worst_client_f1', 0):.4f}")
    cols[3].metric("MB Communicated", f"{rd.get('mb_communicated', 0):.2f}")


def _comparison_section(runs: list[dict]) -> None:
    """FedAvg vs FedProx vs centralized comparison curves."""
    st.subheader("Method Comparison")

    # Gather all FL run convergence curves
    fig = go.Figure()
    method_colors = {"fedavg": "#3498db", "fedprox": "#e74c3c", "local_only": "#f39c12"}

    for run in runs:
        detail = api_client.fl_run_detail(run["run_name"])
        if detail is None:
            continue
        history = detail.get("history", [])
        if not history:
            continue
        rounds = [h["round"] for h in history]
        f1s = [h.get("global_val_macro_f1", 0) for h in history]
        method = run.get("method", "unknown")
        alpha = run.get("alpha", "?")
        mu = run.get("mu", 0)
        label = f"{method} (a={alpha}"
        if method == "fedprox" and mu:
            label += f", mu={mu}"
        label += ")"
        color = method_colors.get(method, "#888")
        fig.add_trace(go.Scatter(
            x=rounds, y=f1s, mode="lines+markers", name=label,
            line=dict(color=color, width=2),
        ))

    # Add centralized baselines from experiments
    exp = api_client.experiments_summary()
    if exp and exp.get("rows"):
        for row in exp["rows"]:
            if row.get("model_type") in ("mlp", "xgb"):
                f1 = row.get("test_metrics", {}).get("macro_f1")
                if f1 is None:
                    f1 = row.get("test_macro_f1")
                if f1 is not None:
                    name = row.get("run_name", row.get("model_type", "?"))
                    fig.add_hline(
                        y=float(f1), line_dash="dash",
                        annotation_text=f"Centralized {name}: {float(f1):.4f}",
                        annotation_position="top left",
                        line_color="#2ecc71",
                    )

    fig.update_layout(
        title="Convergence: Federated vs Centralized",
        xaxis_title="Round", yaxis_title="Val Macro-F1",
        height=350, margin=dict(l=40, r=20, t=40, b=40),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", legend=dict(x=0.01, y=0.99),
    )
    st.plotly_chart(fig, use_container_width=True)


def _partition_section() -> None:
    """Partition heatmap visualization."""
    st.subheader("Data Partition")

    parts = api_client.fl_partitions()
    if not parts:
        st.info("No partition data available")
        return

    part = parts[0]  # Show first partition
    dist = part.get("class_distribution", [])
    if not dist:
        return

    n_clients = len(dist)
    n_classes = len(CLASS_ORDER)
    arr = np.zeros((n_classes, n_clients))
    for c_idx in range(n_clients):
        for cls_idx in range(min(n_classes, len(dist[c_idx]))):
            arr[cls_idx, c_idx] = dist[c_idx][cls_idx]

    fig = px.imshow(
        arr,
        labels=dict(x="Client", y="Class", color="Samples"),
        x=[f"C{i}" for i in range(n_clients)],
        y=CLASS_ORDER[:n_classes],
        color_continuous_scale="YlOrRd",
        aspect="auto",
        title=f"Partition (alpha={part.get('alpha', '?')}, K={n_clients})",
    )
    fig.update_layout(
        height=350, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    st.plotly_chart(fig, use_container_width=True)


def _privacy_card(runs: list[dict]) -> None:
    """Privacy card: data transmitted vs model updates."""
    st.subheader("Privacy Summary")

    total_mb = 0.0
    total_rounds = 0
    for run in runs:
        detail = api_client.fl_run_detail(run["run_name"])
        if detail is None:
            continue
        for h in detail.get("history", []):
            total_mb += h.get("mb_communicated", 0)
            total_rounds += 1

    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Raw data transmitted | **0 bytes** |
    | Model updates exchanged | **{total_mb:.2f} MB** |
    | Total FL rounds | **{total_rounds}** |
    | Avg update size | **{total_mb / max(total_rounds, 1):.2f} MB/round** |

    > Federated learning keeps raw data on each client device.
    > Only model weight updates are shared with the server.
    """)
