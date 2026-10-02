"""Page 4: Architecture — system diagram and module mapping."""

from __future__ import annotations

import streamlit as st

from netshield.dashboard.components import api_down_banner, model_badge


def render() -> None:
    st.header("Architecture")
    model_badge()

    if api_down_banner():
        return

    _system_diagram()
    st.markdown("---")
    _module_mapping()


def _system_diagram() -> None:
    """Graphviz diagram of the full system."""
    st.subheader("System Overview")

    dot = """
    digraph NetShield {
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="filled,rounded", fontname="Courier",
              fontsize=11, fontcolor="#e0e0e0", color="#444"];
        edge [color="#888", fontcolor="#aaa", fontsize=9];

        subgraph cluster_data {
            label="Data Pipeline";
            style="dashed"; color="#555"; fontcolor="#aaa";
            CSV [label="CICIoT2023\\nCSVs", fillcolor="#2c3e50"];
            ETL [label="Spark ETL\\n(Docker)", fillcolor="#2c3e50"];
            Parquet [label="Parquet\\nSplits", fillcolor="#2c3e50"];
            CSV -> ETL -> Parquet;
        }

        subgraph cluster_training {
            label="Training (Two Machines)";
            style="dashed"; color="#555"; fontcolor="#aaa";

            subgraph cluster_laptop {
                label="Dev Laptop\\n(RTX 4050)";
                style="dotted"; color="#666"; fontcolor="#aaa";
                Smoke [label="Smoke\\nBaselines", fillcolor="#1a3a5c"];
                SmokeFl [label="Smoke\\nFL Grid", fillcolor="#1a3a5c"];
            }

            subgraph cluster_lab {
                label="Lab PC\\n(RTX 4090)";
                style="dotted"; color="#666"; fontcolor="#aaa";
                FullTrain [label="Full\\nBaselines", fillcolor="#3a1a5c"];
                FullFL [label="Full FL\\nGrid", fillcolor="#3a1a5c"];
            }

            Bundle [label="git bundle\\n+ zip", fillcolor="#2c3e50", shape=diamond];
            Parquet -> Smoke;
            Parquet -> SmokeFl;
            Bundle -> FullTrain [label="code+data"];
            Bundle -> FullFL;
            FullTrain -> Bundle [label="results"];
            FullFL -> Bundle [label="results"];
        }

        subgraph cluster_serving {
            label="Model Serving";
            style="dashed"; color="#555"; fontcolor="#aaa";
            ONNX [label="ONNX\\nExport", fillcolor="#1a5c3a"];
            Active [label="active.json\\n(hot-reload)", fillcolor="#1a5c3a"];
            MLflow [label="MLflow\\nTracking", fillcolor="#1a5c3a"];
            Smoke -> ONNX;
            SmokeFl -> ONNX;
            FullTrain -> ONNX;
            FullFL -> ONNX;
            ONNX -> Active;
            ONNX -> MLflow;
        }

        subgraph cluster_stream {
            label="Streaming Pipeline (Docker)";
            style="dashed"; color="#555"; fontcolor="#aaa";
            Producer [label="Producer\\n(Host)", fillcolor="#5c3a1a"];
            Kafka [label="Kafka\\n(KRaft)", fillcolor="#5c3a1a"];
            Spark [label="Spark\\nStreaming", fillcolor="#5c3a1a"];
            Alerts [label="iot.alerts", fillcolor="#5c3a1a"];
            Metrics [label="iot.metrics", fillcolor="#5c3a1a"];
            Control [label="iot.control", fillcolor="#5c3a1a"];
            Producer -> Kafka [label="iot.flows"];
            Kafka -> Spark;
            Active -> Spark [label="ONNX model"];
            Spark -> Alerts;
            Spark -> Metrics;
            Control -> Producer [label="inject/stop"];
        }

        subgraph cluster_app {
            label="Application Layer";
            style="dashed"; color="#555"; fontcolor="#aaa";
            API [label="FastAPI\\n:8000", fillcolor="#1a3a5c"];
            Dashboard [label="Streamlit\\n:8501", fillcolor="#1a3a5c"];
            Alerts -> API;
            Metrics -> API;
            API -> Control [label="inject/stop"];
            API -> Dashboard [label="HTTP"];
            Active -> API [label="hot-reload"];
        }
    }
    """
    st.graphviz_chart(dot, use_container_width=True)


def _module_mapping() -> None:
    """Component to source module mapping table."""
    st.subheader("Module Mapping")

    data = [
        ("Data Pipeline", "Spark ETL", "`netshield.etl.run_etl`"),
        ("Data Pipeline", "Feature Stats", "`netshield.common.preprocess`"),
        ("Data Pipeline", "Label Mapping", "`netshield.common.labels`"),
        ("Training", "MLP Baseline", "`netshield.models.mlp`"),
        ("Training", "XGBoost Baseline", "`netshield.models.train_centralized`"),
        ("Training", "ONNX Export", "`netshield.models.train_centralized`"),
        ("Federated", "Dirichlet Partition", "`netshield.federated.partition`"),
        ("Federated", "FL Client", "`netshield.federated.client`"),
        ("Federated", "FL Server (FedAvg/Prox)", "`netshield.federated.server`"),
        ("Federated", "Strategy Runner", "`netshield.federated.strategies`"),
        ("Federated", "Experiment Grid", "`netshield.federated.run_experiment`"),
        ("Streaming", "Kafka Producer", "`netshield.streaming.producer`"),
        ("Streaming", "Score Batch (ONNX)", "`netshield.streaming.score`"),
        ("Streaming", "Spark Inference", "`netshield.streaming.spark_inference`"),
        ("Streaming", "Benchmark", "`netshield.streaming.benchmark`"),
        ("Application", "FastAPI Backend", "`netshield.api.main`"),
        ("Application", "Live State", "`netshield.api.state`"),
        ("Application", "Kafka Consumers", "`netshield.api.kafka_consumers`"),
        ("Application", "Dashboard", "`netshield.dashboard.app`"),
        ("Config", "Profiles & Merge", "`netshield.common.config`"),
        ("Config", "Message Schemas", "`netshield.common.schemas`"),
        ("Lab Transfer", "Pack / Unpack", "`scripts/pack_for_lab.ps1`"),
        ("Lab Transfer", "MLflow Rewrite", "`scripts/rewrite_mlflow_paths.py`"),
    ]

    st.markdown("| Category | Component | Module |")
    st.markdown("|----------|-----------|--------|")
    for cat, comp, mod in data:
        st.markdown(f"| {cat} | {comp} | {mod} |")
