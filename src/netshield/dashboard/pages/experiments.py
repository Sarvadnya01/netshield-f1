"""Page 3: Experiments — centralized + FL results, per-class F1, ONNX latency."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from netshield.dashboard import api_client
from netshield.dashboard.components import (
    CLASS_ORDER,
    api_down_banner,
    model_badge,
)


def render() -> None:
    st.header("Experiments")
    model_badge()

    if api_down_banner():
        return

    exp = api_client.experiments_summary()
    if exp is None:
        st.error("Could not fetch experiment data")
        return

    rows = exp.get("rows", [])
    smoke_only = exp.get("smoke_only", True)

    if smoke_only:
        st.warning(
            "Only **smoke-profile** results available. "
            "Full-scale lab results will appear after PL2."
        )

    if not rows:
        st.info("No experiment results found.")
        return

    # Filter by profile
    profiles = sorted({r.get("profile", "unknown") for r in rows})
    if len(profiles) > 1:
        idx = profiles.index("lab") if "lab" in profiles else 0
        selected_profile = st.selectbox("Profile", profiles, index=idx)
        rows = [r for r in rows if r.get("profile") == selected_profile]

    _results_table(rows)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        _per_class_heatmap(rows)
    with col2:
        _macro_f1_comparison(rows)

    st.markdown("---")
    _onnx_latency()
    _streaming_benchmark()


def _results_table(rows: list[dict]) -> None:
    """Summary results table."""
    st.subheader("Results Summary")

    table_rows = []
    for r in rows:
        metrics = r.get("test_metrics", {})
        macro_f1 = metrics.get("macro_f1") or r.get("test_macro_f1")
        accuracy = metrics.get("accuracy") or r.get("test_accuracy")
        table_rows.append({
            "Run": r.get("run_name", r.get("method", "?")),
            "Type": r.get("model_type", r.get("method", "?")),
            "Macro-F1": f"{float(macro_f1):.4f}" if macro_f1 is not None else "?",
            "Accuracy": f"{float(accuracy):.4f}" if accuracy is not None else "?",
            "Rows": r.get("train_rows", r.get("num_clients", "?")),
            "Profile": r.get("profile", "?"),
        })

    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)


def _per_class_heatmap(rows: list[dict]) -> None:
    """Per-class F1 heatmap across experiments."""
    st.subheader("Per-Class F1")

    data = []
    run_names = []
    for r in rows:
        metrics = r.get("test_metrics", {})
        per_class = metrics.get("per_class", {})
        if not per_class:
            continue
        name = r.get("run_name", r.get("method", "?"))
        run_names.append(name)
        row_data = []
        for cls in CLASS_ORDER:
            f1 = per_class.get(cls, {}).get("f1", 0)
            row_data.append(float(f1) if f1 is not None else 0)
        data.append(row_data)

    if not data:
        st.info("No per-class data available")
        return

    arr = np.array(data)
    fig = px.imshow(
        arr, x=CLASS_ORDER, y=run_names,
        color_continuous_scale="RdYlGn",
        labels=dict(color="F1"),
        aspect="auto",
        title="Per-Class F1 Scores",
        text_auto=".3f",
    )
    fig.update_layout(
        height=max(200, 60 * len(run_names) + 80),
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0",
    )
    st.plotly_chart(fig, use_container_width=True)


def _macro_f1_comparison(rows: list[dict]) -> None:
    """Macro-F1 bar comparison across experiments."""
    st.subheader("Macro-F1 Comparison")

    bar_data = []
    for r in rows:
        metrics = r.get("test_metrics", {})
        f1 = metrics.get("macro_f1") or r.get("test_macro_f1")
        if f1 is None:
            continue
        name = r.get("run_name", r.get("method", "?"))
        model_type = r.get("model_type", r.get("method", "?"))
        bar_data.append({"Run": name, "Macro-F1": float(f1), "Type": model_type})

    if not bar_data:
        st.info("No F1 data available")
        return

    df = pd.DataFrame(bar_data)
    color_map = {"mlp": "#3498db", "xgb": "#2ecc71", "fedavg": "#e74c3c", "fedprox": "#e67e22"}
    fig = px.bar(
        df, x="Run", y="Macro-F1", color="Type",
        color_discrete_map=color_map,
        title="Macro-F1 by Experiment",
    )
    fig.update_layout(
        height=350, margin=dict(l=40, r=20, t=40, b=80),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e0e0e0", xaxis_tickangle=-30,
    )
    st.plotly_chart(fig, use_container_width=True)


def _onnx_latency() -> None:
    """ONNX inference latency from the active model card."""
    info = api_client.model_active()
    if info is None:
        return

    card = info.get("model_card", {})
    latency = card.get("ort_latency_ms", {})
    if not latency:
        return

    st.subheader("ONNX Inference Latency")
    data = [{"Batch Size": k.replace("batch_", ""), "Latency (ms)": v} for k, v in latency.items()]
    df = pd.DataFrame(data)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with col2:
        fig = px.bar(
            df, x="Batch Size", y="Latency (ms)",
            color_discrete_sequence=["#00d4aa"],
            title="ORT Latency by Batch Size",
        )
        fig.update_layout(
            height=250, margin=dict(l=40, r=20, t=40, b=40),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e0e0e0",
        )
        st.plotly_chart(fig, use_container_width=True)


def _streaming_benchmark() -> None:
    """Show streaming benchmark results if available."""
    exp = api_client.experiments_summary()
    if exp is None:
        return

    # Look for stream_benchmark files in the results
    rows = exp.get("rows", [])
    bench = [r for r in rows if "stream_benchmark" in str(r.get("run_name", ""))]
    if not bench:
        # Check if any row has throughput fields
        bench = [r for r in rows if "end_to_end_throughput_eps" in r]

    if not bench:
        return

    st.subheader("Streaming Benchmark")
    for b in bench:
        cols = st.columns(4)
        cols[0].metric("Throughput", f"{b.get('end_to_end_throughput_eps', '?')} eps")
        cols[1].metric("p50 Latency", f"{b.get('latency_ms', {}).get('p50', '?')} ms")
        cols[2].metric("p95 Latency", f"{b.get('latency_ms', {}).get('p95', '?')} ms")
        cols[3].metric("p99 Latency", f"{b.get('latency_ms', {}).get('p99', '?')} ms")
