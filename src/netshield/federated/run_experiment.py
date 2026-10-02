"""FL experiment grid runner.

Iterates over methods × alphas × mus × seeds from the merged config profile.
Each run is logged to MLflow and saved as JSON. Skips runs whose final JSON exists.

Usage:
  python -m netshield.federated.run_experiment
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path

import numpy as np

from netshield.common.config import active_profile, get_config, repo_root
from netshield.common.labels import NUM_CLASSES
from netshield.federated.partition import create_partition
from netshield.federated.strategies import run_fedavg, run_fedprox, run_local_only
from netshield.models.data import load_split

logger = logging.getLogger(__name__)


def _setup_logging() -> None:
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S",
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root_logger.addHandler(ch)

    log_dir = repo_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(str(log_dir / "fl_experiment.log"))
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)


def _set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def _run_name(method: str, alpha: float | str, mu: float, seed: int) -> str:
    """Generate a canonical run name."""
    alpha_str = "iid" if alpha == "iid" else f"a{alpha}"
    if method == "fedprox":
        return f"fl_{method}_{alpha_str}_mu{mu}_s{seed}"
    return f"fl_{method}_{alpha_str}_s{seed}"


def _build_grid(fl_cfg: dict) -> list[dict]:
    """Build the experiment grid from config."""
    methods = fl_cfg.get("methods", ["fedavg", "fedprox"])
    alphas = fl_cfg.get("alphas", [0.5])
    mus = fl_cfg.get("mus", [0.01])
    seeds = fl_cfg.get("seeds", [42])

    grid = []
    for seed in seeds:
        for alpha in alphas:
            for method in methods:
                if method == "fedprox":
                    for mu in mus:
                        grid.append({
                            "method": method, "alpha": alpha,
                            "mu": mu, "seed": seed,
                        })
                else:
                    grid.append({
                        "method": method, "alpha": alpha,
                        "mu": 0.0, "seed": seed,
                    })
    return grid


def _time_one_round(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    fl_cfg: dict, mlp_cfg: dict, seed: int,
) -> float:
    """Time a single FL round to estimate ETA for the grid."""
    from netshield.federated.partition import dirichlet_partition
    from netshield.federated.strategies import _build_clients

    num_clients = fl_cfg.get("num_clients", 8)
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64])
    dropout = mlp_cfg.get("dropout", 0.3)
    local_epochs = fl_cfg.get("local_epochs", 3)
    local_lr = fl_cfg.get("local_lr", 0.001)
    local_batch_size = fl_cfg.get("local_batch_size", 2048)
    val_fraction = fl_cfg.get("local_val_fraction", 0.1)
    input_dim = X_train.shape[1]

    # Quick partition
    indices = dirichlet_partition(y_train, num_clients, 0.5, seed=seed)
    clients = _build_clients(
        X_train, y_train, indices,
        input_dim, hidden_dims, dropout,
        local_epochs, local_lr, local_batch_size,
        mu=0.0, val_fraction=val_fraction, seed=seed,
    )

    # Time one round only
    import torch
    torch.manual_seed(seed)
    from netshield.models.mlp import MLP
    model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout)
    global_weights = model.get_weights()

    t0 = time.time()
    results = []
    for c in clients:
        results.append(c.train_round(global_weights))
    from netshield.federated.server import weighted_aggregate
    weighted_aggregate(results)
    round_time = time.time() - t0
    return round_time


def _append_to_csv(csv_path: Path, row: dict) -> None:
    """Append a row to the FL results CSV, creating it if needed."""
    exists = csv_path.exists()
    fieldnames = [
        "run_name", "method", "alpha", "mu", "seed", "profile",
        "num_clients", "rounds", "final_global_val_f1", "test_macro_f1",
        "test_accuracy", "test_weighted_f1", "worst_client_f1",
        "wall_time_s",
    ]
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def select_best(profile: str) -> dict | None:
    """Select the best FL run by VAL macro-F1 at alpha=0.5 among fedavg/fedprox.

    Returns the run info dict or None if no eligible runs.
    """
    root = repo_root()
    fl_dir = root / "reports" / "fl_runs"
    if not fl_dir.exists():
        return None

    best = None
    best_f1 = -1.0

    for p in fl_dir.glob("fl_fed*.json"):
        if p.name.startswith("partition_"):
            continue
        with open(p) as f:
            run = json.load(f)
        if run.get("profile") != profile:
            continue
        alpha = run.get("alpha")
        if alpha != 0.5:
            continue
        method = run.get("method", "")
        if method not in ("fedavg", "fedprox"):
            continue
        # Use last round's global_val_macro_f1
        history = run.get("history", [])
        if not history:
            continue
        final_f1 = history[-1].get("global_val_macro_f1", 0)
        if final_f1 > best_f1:
            best_f1 = final_f1
            best = run

    return best


def main() -> None:
    _setup_logging()

    cfg = get_config()
    profile = active_profile()
    fl_cfg = cfg.get("fl", {})
    mlp_cfg = cfg.get("mlp", {})
    train_cfg = cfg.get("training", {})
    seed_base = train_cfg.get("seed", 42)
    train_max_rows = train_cfg.get("train_max_rows")

    num_clients = fl_cfg.get("num_clients", 8)
    num_rounds = fl_cfg.get("rounds", 20)
    min_samples = fl_cfg.get("min_samples_per_client", 50)

    logger.info("=== FL Grid Experiment (profile=%s) ===", profile)

    # Load data
    X_train, y_train = load_split("train", max_rows=train_max_rows, seed=seed_base)
    X_val, y_val = load_split("val", seed=seed_base)
    X_test, y_test = load_split("test", seed=seed_base)

    # Build grid
    grid = _build_grid(fl_cfg)
    logger.info("Grid: %d runs", len(grid))

    # Check which runs are already done
    root = repo_root()
    fl_dir = root / "reports" / "fl_runs"
    fl_dir.mkdir(parents=True, exist_ok=True)
    csv_path = root / "reports" / "tables" / "fl_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    pending = []
    for spec in grid:
        name = _run_name(spec["method"], spec["alpha"], spec["mu"], spec["seed"])
        result_path = fl_dir / f"{name}.json"
        if result_path.exists():
            logger.info("Skipping %s — result exists", name)
        else:
            pending.append(spec)

    if not pending:
        logger.info("All %d runs already complete", len(grid))
        return

    # Time one round and print ETA
    logger.info("Timing one round for ETA estimation...")
    round_time = _time_one_round(X_train, y_train, X_val, y_val, fl_cfg, mlp_cfg, seed_base)
    logger.info("One round took %.2fs", round_time)

    total_rounds = 0
    for spec in pending:
        if spec["method"] == "local_only":
            total_rounds += 1  # local_only is one batch
        else:
            total_rounds += num_rounds
    eta_s = round_time * total_rounds
    logger.info(
        "ETA for %d pending runs (%d total rounds): %.0fs (~%.1f min)",
        len(pending), total_rounds, eta_s, eta_s / 60,
    )

    # MLflow setup
    import mlflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(f"{cfg['mlflow']['experiment_prefix']}-federated")

    for spec in pending:
        name = _run_name(spec["method"], spec["alpha"], spec["mu"], spec["seed"])
        _set_deterministic(spec["seed"])

        logger.info("--- Starting: %s ---", name)

        # Create partition (deterministic from alpha, seed)
        alpha_str = "iid" if spec["alpha"] == "iid" else f"a{spec['alpha']}"
        part_name = f"K{num_clients}_{alpha_str}_s{spec['seed']}"
        client_indices, part_info = create_partition(
            y_train, alpha=spec["alpha"], num_clients=num_clients,
            seed=spec["seed"], min_samples=min_samples, name=part_name,
        )

        t0 = time.time()

        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                "method": spec["method"],
                "alpha": str(spec["alpha"]),
                "mu": spec["mu"],
                "seed": spec["seed"],
                "profile": profile,
                "num_clients": num_clients,
                "num_rounds": num_rounds,
                "train_rows": len(X_train),
                "input_dim": X_train.shape[1],
            })

            if spec["method"] == "fedavg":
                gw, history, test_metrics = run_fedavg(
                    name, X_train, y_train, client_indices,
                    X_val, y_val, X_test, y_test, seed=spec["seed"],
                )
            elif spec["method"] == "fedprox":
                gw, history, test_metrics = run_fedprox(
                    name, X_train, y_train, client_indices,
                    X_val, y_val, X_test, y_test,
                    mu=spec["mu"], seed=spec["seed"],
                )
            elif spec["method"] == "local_only":
                gw, history, test_metrics = run_local_only(
                    name, X_train, y_train, client_indices,
                    X_val, y_val, X_test, y_test, seed=spec["seed"],
                )
            else:
                raise ValueError(f"Unknown method: {spec['method']}")

            wall_time = time.time() - t0

            # Log metrics to MLflow
            if test_metrics.get("macro_f1") is not None:
                mlflow.log_metric("test_macro_f1", test_metrics["macro_f1"])
            if test_metrics.get("accuracy") is not None:
                mlflow.log_metric("test_accuracy", test_metrics["accuracy"])
            mlflow.log_metric("wall_time_s", wall_time)
            if history:
                last = history[-1]
                if "global_val_macro_f1" in last:
                    mlflow.log_metric("final_val_f1", last["global_val_macro_f1"])

        # Save result JSON
        result = {
            "run_name": name,
            "method": spec["method"],
            "alpha": spec["alpha"],
            "mu": spec["mu"],
            "seed": spec["seed"],
            "profile": profile,
            "num_clients": num_clients,
            "rounds": num_rounds,
            "train_rows": len(X_train),
            "input_dim": X_train.shape[1],
            "wall_time_s": round(wall_time, 1),
            "test_metrics": test_metrics,
            "history": history,
        }
        result_path = fl_dir / f"{name}.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Result saved: %s", result_path)

        # Append to CSV
        final_val_f1 = history[-1].get("global_val_macro_f1", None) if history else None
        _append_to_csv(csv_path, {
            "run_name": name,
            "method": spec["method"],
            "alpha": str(spec["alpha"]),
            "mu": spec["mu"],
            "seed": spec["seed"],
            "profile": profile,
            "num_clients": num_clients,
            "rounds": num_rounds,
            "final_global_val_f1": final_val_f1,
            "test_macro_f1": test_metrics.get("macro_f1"),
            "test_accuracy": test_metrics.get("accuracy"),
            "test_weighted_f1": test_metrics.get("weighted_f1"),
            "worst_client_f1": test_metrics.get(
                "worst_client_f1",
                min(history[-1].get("client_f1s", [0]))
                if history else None,
            ),
            "wall_time_s": round(wall_time, 1),
        })

        logger.info(
            "Done: %s  test_f1=%.4f  wall=%.1fs",
            name, test_metrics.get("macro_f1", 0), wall_time,
        )

    # Selection + export
    logger.info("=== Selecting best model ===")
    best = select_best(profile)
    if best is not None:
        logger.info(
            "Best run: %s (val_f1=%.4f at alpha=0.5)",
            best["run_name"], best["history"][-1]["global_val_macro_f1"],
        )
        _export_best(best, profile)
    else:
        logger.warning("No eligible runs found for selection at alpha=0.5")

    logger.info("=== FL Grid complete ===")


def _export_best(best_run: dict, profile: str) -> None:
    """Export the best FL model to ONNX."""
    import torch

    from netshield.models.export_onnx import (
        benchmark_ort,
        export_mlp_onnx,
        set_active,
        verify_parity,
        write_model_card,
    )
    from netshield.models.mlp import MLP

    root = repo_root()
    cfg = get_config()
    mlp_cfg = cfg.get("mlp", {})
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64])
    dropout = mlp_cfg.get("dropout", 0.3)

    run_name = best_run["run_name"]
    input_dim = best_run["input_dim"]

    # Load global weights from checkpoint
    ckpt_dir = root / "checkpoints" / "fl" / run_name
    weights_path = ckpt_dir / "global_weights.pt"
    if not weights_path.exists():
        logger.warning("No checkpoint found for %s, skipping export", run_name)
        return

    global_weights = torch.load(weights_path, map_location="cpu", weights_only=True)

    # Build model and load weights
    model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout)
    model.set_weights(global_weights)

    # Save as a standard checkpoint so export_onnx can load it
    export_ckpt_dir = root / "checkpoints" / run_name
    export_ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), export_ckpt_dir / "model.pt")
    meta = {
        "model_type": "mlp",
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "num_classes": NUM_CLASSES,
        "dropout": dropout,
    }
    import json
    with open(export_ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    if profile == "lab":
        onnx_name = "netshield_fl_global"
    else:
        onnx_name = "netshield_fl_smoke"

    onnx_path = export_mlp_onnx(run_name, onnx_name)
    parity = verify_parity(run_name, onnx_path)
    latency = benchmark_ort(onnx_path, input_dim)
    write_model_card(onnx_name, run_name, onnx_path, latency, parity)

    if profile == "lab":
        set_active(onnx_name, "production")
        logger.info("Set active: %s (production)", onnx_name)
    else:
        logger.info("Exported %s (smoke, not setting active)", onnx_name)


if __name__ == "__main__":
    main()
