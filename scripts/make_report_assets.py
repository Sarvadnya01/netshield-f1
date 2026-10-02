"""Generate report-ready tables and figures from canonical results.

Usage:
  python scripts/make_report_assets.py [--profile lab]

Reads from reports/tables/, reports/fl_runs/. Writes to reports/figures/.
Prefers profile="lab" results; falls back to "laptop" (smoke) with a warning.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = REPO_ROOT / "reports" / "tables"
FL_RUNS_DIR = REPO_ROOT / "reports" / "fl_runs"
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
CLASS_ORDER = ["Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"]
DPI = 300


def _load_centralized(profile: str) -> list[dict]:
    """Load centralized experiment JSONs, preferring the given profile."""
    rows = []
    for f in sorted(TABLES_DIR.glob("centralized_*.json")):
        data = json.loads(f.read_text())
        rows.append(data)

    # Filter to profile if available
    lab = [r for r in rows if r.get("profile") == profile]
    if lab:
        return lab
    if profile == "lab":
        logger.warning("No lab-profile centralized results; using smoke data")
    return rows


def _load_fl_runs(profile: str) -> list[dict]:
    """Load FL run JSONs, preferring the given profile."""
    rows = []
    for f in sorted(FL_RUNS_DIR.glob("fl_*.json")):
        data = json.loads(f.read_text())
        rows.append(data)

    lab = [r for r in rows if r.get("profile") == profile]
    if lab:
        return lab
    if profile == "lab":
        logger.warning("No lab-profile FL results; using smoke data")
    return rows


def _load_partitions() -> list[dict]:
    parts = []
    for f in sorted(FL_RUNS_DIR.glob("partition_*.json")):
        parts.append(json.loads(f.read_text()))
    return parts


def _load_stream_benchmark() -> dict | None:
    for f in sorted(TABLES_DIR.glob("stream_benchmark_*.json")):
        return json.loads(f.read_text())
    return None


# ---- Table: Main Results ----

def make_main_table(centralized: list[dict], fl_runs: list[dict]) -> pd.DataFrame:
    """method x alpha -> macro-F1, accuracy, client-F1 std, rounds to 95%, MB."""
    rows = []

    # Centralized baselines
    for c in centralized:
        m = c.get("test_metrics", {})
        rows.append({
            "Method": c.get("model_type", "?").upper(),
            "Alpha": "-",
            "Macro-F1": f"{m.get('macro_f1', 0):.4f}",
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "Client-F1 Std": "-",
            "Rounds to 95%": "-",
            "MB Communicated": "-",
            "Profile": c.get("profile", "?"),
        })

    # FL runs - group by method+alpha, average over seeds
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for r in fl_runs:
        key = (r.get("method", "?"), r.get("alpha", "?"))
        groups[key].append(r)

    for (method, alpha), runs in sorted(groups.items()):
        f1s = [r.get("test_metrics", {}).get("macro_f1", 0) for r in runs]
        accs = [r.get("test_metrics", {}).get("accuracy", 0) for r in runs]

        # Client-F1 std from last round
        client_stds = []
        for r in runs:
            hist = r.get("history", [])
            if hist:
                client_stds.append(hist[-1].get("std_client_f1", 0))

        # Rounds to reach 95% of best centralized macro-F1
        best_cent_f1 = max(
            (c.get("test_metrics", {}).get("macro_f1", 0) for c in centralized),
            default=0,
        )
        target = best_cent_f1 * 0.95
        rounds_95_list = []
        for r in runs:
            hist = r.get("history", [])
            reached = None
            for h in hist:
                if h.get("global_val_macro_f1", 0) >= target:
                    reached = h["round"]
                    break
            rounds_95_list.append(reached)

        # Total MB communicated
        total_mb = []
        for r in runs:
            mb = sum(h.get("mb_communicated", 0) for h in r.get("history", []))
            total_mb.append(mb)

        mean_f1 = np.mean(f1s)
        std_f1 = np.std(f1s) if len(f1s) > 1 else 0

        r95 = [x for x in rounds_95_list if x is not None]
        r95_str = f"{np.mean(r95):.0f}" if r95 else "not reached"

        method_label = method.upper()
        if method == "fedprox":
            mu = runs[0].get("mu", 0)
            method_label = f"FedProx (mu={mu})"

        rows.append({
            "Method": method_label,
            "Alpha": str(alpha),
            "Macro-F1": f"{mean_f1:.4f}" + (f" +/- {std_f1:.4f}" if std_f1 > 0 else ""),
            "Accuracy": f"{np.mean(accs):.4f}",
            "Client-F1 Std": f"{np.mean(client_stds):.4f}" if client_stds else "-",
            "Rounds to 95%": r95_str,
            "MB Communicated": f"{np.mean(total_mb):.2f}",
            "Profile": runs[0].get("profile", "?"),
        })

    df = pd.DataFrame(rows)
    return df


# ---- Figures ----

def fig_macro_f1_vs_alpha(fl_runs: list[dict]) -> None:
    """Macro-F1 vs alpha per method."""
    from collections import defaultdict
    data: dict[str, dict[float, list]] = defaultdict(lambda: defaultdict(list))
    for r in fl_runs:
        method = r.get("method", "?")
        alpha = r.get("alpha", 0)
        f1 = r.get("test_metrics", {}).get("macro_f1", 0)
        data[method][alpha].append(f1)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"fedavg": "#3498db", "fedprox": "#e74c3c", "local_only": "#f39c12"}
    for method, alpha_dict in sorted(data.items()):
        alphas = sorted(alpha_dict.keys())
        means = [np.mean(alpha_dict[a]) for a in alphas]
        stds = [np.std(alpha_dict[a]) for a in alphas]
        ax.errorbar(alphas, means, yerr=stds, marker="o", label=method,
                    color=colors.get(method, "#888"), capsize=4, linewidth=2)

    ax.set_xlabel("Dirichlet Alpha (non-IID severity)")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Macro-F1 vs Non-IID Severity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "macro_f1_vs_alpha.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved macro_f1_vs_alpha.png")


def fig_convergence(fl_runs: list[dict], centralized: list[dict]) -> None:
    """Convergence curves: global macro-F1 per round."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"fedavg": "#3498db", "fedprox": "#e74c3c", "local_only": "#f39c12"}

    for r in fl_runs:
        hist = r.get("history", [])
        if not hist:
            continue
        rounds = [h["round"] for h in hist]
        f1s = [h.get("global_val_macro_f1", 0) for h in hist]
        method = r.get("method", "?")
        alpha = r.get("alpha", "?")
        mu = r.get("mu", 0)
        label = f"{method} (a={alpha})"
        if method == "fedprox":
            label += f" mu={mu}"
        ax.plot(rounds, f1s, marker="o", label=label,
                color=colors.get(method, "#888"), linewidth=2)

    # Centralized horizontal lines
    for c in centralized:
        f1 = c.get("test_metrics", {}).get("macro_f1", 0)
        name = c.get("run_name", c.get("model_type", "?"))
        ax.axhline(y=f1, linestyle="--", color="#2ecc71", alpha=0.7,
                   label=f"Centralized {name}: {f1:.4f}")

    ax.set_xlabel("FL Round")
    ax.set_ylabel("Val Macro-F1")
    ax.set_title("Federated Learning Convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "convergence_curves.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved convergence_curves.png")


def fig_client_fairness(fl_runs: list[dict]) -> None:
    """Client-fairness boxplots: per-client F1 at final round."""
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = []
    data = []

    for r in fl_runs:
        hist = r.get("history", [])
        if not hist:
            continue
        last = hist[-1]
        client_f1s = last.get("client_f1s", [])
        if client_f1s:
            method = r.get("method", "?")
            alpha = r.get("alpha", "?")
            label = f"{method}\na={alpha}"
            labels.append(label)
            data.append(client_f1s)

    if data:
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        colors = ["#3498db", "#e74c3c", "#f39c12", "#2ecc71"]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    ax.set_ylabel("Client Macro-F1")
    ax.set_title("Client Fairness (Final Round)")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "client_fairness_boxplots.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved client_fairness_boxplots.png")


def fig_mb_vs_f1(fl_runs: list[dict]) -> None:
    """MB communicated vs macro-F1 scatter."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"fedavg": "#3498db", "fedprox": "#e74c3c", "local_only": "#f39c12"}

    for r in fl_runs:
        total_mb = sum(h.get("mb_communicated", 0) for h in r.get("history", []))
        f1 = r.get("test_metrics", {}).get("macro_f1", 0)
        method = r.get("method", "?")
        ax.scatter(total_mb, f1, s=100, color=colors.get(method, "#888"),
                   label=f"{method} (a={r.get('alpha', '?')})", edgecolors="white", zorder=5)

    ax.set_xlabel("Total MB Communicated")
    ax.set_ylabel("Test Macro-F1")
    ax.set_title("Communication Cost vs Performance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mb_vs_f1.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved mb_vs_f1.png")


def fig_per_class_heatmap(centralized: list[dict], fl_runs: list[dict]) -> None:
    """Per-class F1 heatmap: centralized vs federated."""
    methods = []
    matrix = []

    for c in centralized:
        pc = c.get("test_metrics", {}).get("per_class", {})
        name = c.get("run_name", c.get("model_type", "?"))
        methods.append(f"Cent. {name}")
        matrix.append([pc.get(cls, {}).get("f1", 0) for cls in CLASS_ORDER])

    # Use lowest-alpha runs for FL (most non-IID)
    for r in fl_runs:
        pc = r.get("test_metrics", {}).get("per_class", {})
        method = r.get("method", "?")
        alpha = r.get("alpha", "?")
        methods.append(f"{method} a={alpha}")
        matrix.append([pc.get(cls, {}).get("f1", 0) for cls in CLASS_ORDER])

    if not matrix:
        return

    arr = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, max(3, len(methods) * 0.6 + 1)))
    im = ax.imshow(arr, cmap="RdYlGn", aspect="auto", vmin=0, vmax=max(0.3, arr.max()))

    ax.set_xticks(range(len(CLASS_ORDER)))
    ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=9)

    for i in range(len(methods)):
        for j in range(len(CLASS_ORDER)):
            ax.text(j, i, f"{arr[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if arr[i, j] < 0.1 else "black")

    ax.set_title("Per-Class F1 Scores")
    fig.colorbar(im, ax=ax, label="F1", shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "per_class_f1_heatmap.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved per_class_f1_heatmap.png")


def fig_iat_ablation(centralized: list[dict]) -> None:
    """IAT ablation: with vs without IAT feature."""
    with_iat = [c for c in centralized if "no_iat" not in c.get("run_name", "")
                and c.get("model_type") == "mlp"]
    without_iat = [c for c in centralized if "no_iat" in c.get("run_name", "")]

    if not with_iat or not without_iat:
        logger.info("IAT ablation: insufficient data, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = CLASS_ORDER
    x = np.arange(len(labels))
    width = 0.35

    w_pc = with_iat[0].get("test_metrics", {}).get("per_class", {})
    wo_pc = without_iat[0].get("test_metrics", {}).get("per_class", {})

    f1_with = [w_pc.get(c, {}).get("f1", 0) for c in labels]
    f1_without = [wo_pc.get(c, {}).get("f1", 0) for c in labels]

    ax.bar(x - width / 2, f1_with, width, label="With IAT", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, f1_without, width, label="Without IAT", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Class")
    ax.set_ylabel("F1 Score")
    ax.set_title("IAT Feature Ablation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Annotation: overall macro-F1
    w_f1 = with_iat[0].get("test_metrics", {}).get("macro_f1", 0)
    wo_f1 = without_iat[0].get("test_metrics", {}).get("macro_f1", 0)
    ax.text(0.02, 0.98, f"Macro-F1: with={w_f1:.4f}, without={wo_f1:.4f}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "iat_ablation.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved iat_ablation.png")


def fig_streaming_latency(bench: dict | None) -> None:
    """Streaming latency + throughput from benchmark JSON."""
    if bench is None:
        logger.info("No streaming benchmark data, skipping latency figure")
        return

    lat = bench.get("latency_ms", {})
    if not lat:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Latency bar chart
    keys = ["min", "p50", "p95", "p99", "max"]
    vals = [lat.get(k, 0) for k in keys]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#e74c3c"]
    ax1.bar(keys, vals, color=colors, alpha=0.8)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("End-to-End Latency Distribution")
    ax1.grid(True, alpha=0.3, axis="y")

    # Throughput summary
    info_text = (
        f"Events sent: {bench.get('events_sent', '?')}\n"
        f"Events received: {bench.get('events_received', '?')}\n"
        f"Receive rate: {bench.get('receive_rate', '?')}%\n"
        f"Send throughput: {bench.get('send_throughput_eps', '?')} eps\n"
        f"E2E throughput: {bench.get('end_to_end_throughput_eps', '?')} eps\n"
        f"GPU: {bench.get('gpu', 'none')}\n"
        f"RAM: {bench.get('ram_gb', '?')} GB"
    )
    ax2.text(0.5, 0.5, info_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment="center", horizontalalignment="center",
             fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#1a1f2e", edgecolor="#444"))
    ax2.set_title("Streaming Benchmark Summary")
    ax2.axis("off")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "streaming_latency.png", dpi=DPI)
    plt.close(fig)
    logger.info("Saved streaming_latency.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report assets")
    parser.add_argument("--profile", default="lab", help="Preferred profile (default: lab)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    profile = args.profile
    centralized = _load_centralized(profile)
    fl_runs = _load_fl_runs(profile)
    partitions = _load_partitions()
    bench = _load_stream_benchmark()

    logger.info("Loaded %d centralized, %d FL runs, %d partitions",
                len(centralized), len(fl_runs), len(partitions))

    # Main table
    table = make_main_table(centralized, fl_runs)
    table.to_csv(TABLES_DIR / "main_results.csv", index=False)
    table.to_markdown(TABLES_DIR / "main_results.md", index=False)
    logger.info("Saved main_results.csv and main_results.md")
    print("\n=== Main Results Table ===")
    print(table.to_string(index=False))

    # Figures
    fig_macro_f1_vs_alpha(fl_runs)
    fig_convergence(fl_runs, centralized)
    fig_client_fairness(fl_runs)
    fig_mb_vs_f1(fl_runs)
    fig_per_class_heatmap(centralized, fl_runs)
    fig_iat_ablation(centralized)
    fig_streaming_latency(bench)

    # Summary
    figures = list(FIGURES_DIR.glob("*.png"))
    logger.info("Done. %d figures in %s", len(figures), FIGURES_DIR)
    for f in sorted(figures):
        logger.info("  %s (%.0f KB)", f.name, f.stat().st_size / 1024)


if __name__ == "__main__":
    main()
