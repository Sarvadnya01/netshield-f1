"""Centralized training for MLP and XGBoost baselines.

Usage:
  python -m netshield.models.train_centralized --model mlp --run-name mlp_smoke
  python -m netshield.models.train_centralized --model xgb --run-name xgb_smoke
  python -m netshield.models.train_centralized --model mlp \
    --drop-features iat --run-name mlp_no_iat_smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time

import numpy as np

from netshield.common.config import active_profile, get_config, repo_root
from netshield.common.labels import NUM_CLASSES
from netshield.models.data import get_feature_names, load_split
from netshield.models.evaluate import compute_metrics, save_confusion_matrix

logger = logging.getLogger(__name__)


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
    fh = logging.FileHandler(str(log_dir / "train_centralized.log"))
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)


# ---------- MLP training ----------------------------------------------------

def _train_mlp(
    run_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: dict, seed: int,
) -> dict:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from netshield.common.hardware import get_device
    from netshield.models.mlp import MLP

    device = torch.device(get_device())
    train_cfg = cfg["training"]
    mlp_cfg = cfg.get("mlp", {})

    batch_size = train_cfg["batch_size"]
    max_epochs = train_cfg["max_epochs"]
    lr = train_cfg.get("learning_rate", 0.001)
    patience = train_cfg.get("early_stopping_patience", 5)
    hidden_dims = mlp_cfg.get("hidden_dims", [256, 128, 64])
    dropout = mlp_cfg.get("dropout", 0.3)
    input_dim = X_train.shape[1]

    def _run(bs: int) -> tuple:
        model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        use_amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        train_ds = TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).long(),
        )
        val_ds = TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).long(),
        )
        train_loader = DataLoader(
            train_ds, batch_size=bs, shuffle=True,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds, batch_size=bs * 2, shuffle=False,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )

        best_f1 = -1.0
        best_state = None
        wait = 0

        for epoch in range(max_epochs):
            # Train
            model.train()
            train_loss = 0.0
            n = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item() * len(xb)
                n += len(xb)

            # Validate
            model.eval()
            all_preds = []
            all_true = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        logits = model(xb)
                    all_preds.append(logits.argmax(dim=1).cpu().numpy())
                    all_true.append(yb.numpy())

            val_preds = np.concatenate(all_preds)
            val_true = np.concatenate(all_true)
            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_true, val_preds, average="macro", zero_division=0)

            logger.info(
                "Epoch %d/%d  train_loss=%.4f  val_macro_f1=%.4f  (best=%.4f)",
                epoch + 1, max_epochs, train_loss / n, val_f1, best_f1,
            )

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break

        model.load_state_dict(best_state)
        model.to(device)

        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
            logger.info("Peak VRAM: %.2f GB", peak_vram)

        return model, bs

    # Try training; catch OOM and retry with half batch
    try:
        model, actual_bs = _run(batch_size)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        new_bs = batch_size // 2
        logger.warning("CUDA OOM at batch_size=%d, retrying with %d", batch_size, new_bs)
        model, actual_bs = _run(new_bs)

    # Save checkpoint
    ckpt_dir = repo_root() / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "model.pt")

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        X_t = torch.from_numpy(X_test).float().to(device)
        logits = model(X_t).cpu()
    y_probs = torch.softmax(logits, dim=1).numpy()
    y_pred = logits.argmax(dim=1).numpy()
    test_metrics = compute_metrics(y_test, y_pred, y_probs)

    # Confusion matrix
    fig_dir = repo_root() / "reports" / "figures"
    save_confusion_matrix(y_test, y_pred, fig_dir, prefix=f"cm_{run_name}")

    # Save model metadata for export_onnx
    meta = {
        "model_type": "mlp",
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "num_classes": NUM_CLASSES,
        "dropout": dropout,
        "batch_size_used": actual_bs,
    }
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return test_metrics


# ---------- XGBoost training -------------------------------------------------

def _train_xgb(
    run_name: str,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    cfg: dict, seed: int,
) -> dict:
    from netshield.models.xgb_baseline import train_xgb

    model = train_xgb(X_train, y_train, X_val, y_val, seed=seed)

    # Save checkpoint
    ckpt_dir = repo_root() / "checkpoints" / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(ckpt_dir / "model.ubj"))

    # Evaluate on test
    y_probs = model.predict_proba(X_test)
    y_pred = np.argmax(y_probs, axis=1)
    test_metrics = compute_metrics(y_test, y_pred, y_probs)

    fig_dir = repo_root() / "reports" / "figures"
    save_confusion_matrix(y_test, y_pred, fig_dir, prefix=f"cm_{run_name}")

    meta = {"model_type": "xgb", "best_iteration": int(model.best_iteration)}
    with open(ckpt_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return test_metrics


# ---------- CLI entry point --------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Centralized baseline training")
    parser.add_argument("--model", choices=["mlp", "xgb"], required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--drop-features", nargs="*", default=None,
                        help="Feature names to exclude (e.g. iat)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if result JSON exists")
    args = parser.parse_args()

    _setup_logging()

    cfg = get_config()
    profile = active_profile()
    seed = cfg["training"]["seed"]
    train_max_rows = cfg["training"].get("train_max_rows")

    # Resumability check
    result_path = repo_root() / "reports" / "tables" / f"centralized_{args.run_name}.json"
    if result_path.exists() and not args.force:
        logger.info("Skipping %s — result exists. Use --force to rerun.", args.run_name)
        return

    _set_deterministic(seed)

    logger.info("=== Centralized training: %s (profile=%s) ===", args.run_name, profile)
    logger.info("Model: %s | drop_features: %s", args.model, args.drop_features)

    # Load data
    X_train, y_train = load_split("train", max_rows=train_max_rows,
                                  drop_features=args.drop_features, seed=seed)
    X_val, y_val = load_split("val", drop_features=args.drop_features, seed=seed)
    X_test, y_test = load_split("test", drop_features=args.drop_features, seed=seed)

    feature_names = get_feature_names(drop_features=args.drop_features)

    # MLflow
    import mlflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    experiment_name = f"{cfg['mlflow']['experiment_prefix']}-centralized"
    mlflow.set_experiment(experiment_name)

    t_wall = time.time()

    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model_type": args.model,
            "profile": profile,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "test_rows": len(X_test),
            "input_dim": X_train.shape[1],
            "drop_features": str(args.drop_features or []),
            "seed": seed,
        })

        if args.model == "mlp":
            test_metrics = _train_mlp(
                args.run_name, X_train, y_train, X_val, y_val, X_test, y_test, cfg, seed,
            )
        else:
            test_metrics = _train_xgb(
                args.run_name, X_train, y_train, X_val, y_val, X_test, y_test, cfg, seed,
            )

        wall_time = time.time() - t_wall

        # Log flat metrics to MLflow
        for k, v in test_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"test_{k}", v)

        mlflow.log_metric("wall_time_s", wall_time)

    # Save result JSON
    result = {
        "run_name": args.run_name,
        "model_type": args.model,
        "profile": profile,
        "drop_features": args.drop_features or [],
        "feature_names": feature_names,
        "train_rows": len(X_train),
        "input_dim": X_train.shape[1],
        "seed": seed,
        "wall_time_s": round(wall_time, 1),
        "test_metrics": test_metrics,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Result saved: %s", result_path)
    logger.info("Done in %.1fs. macro_f1=%.4f", wall_time, test_metrics["macro_f1"])


if __name__ == "__main__":
    main()
