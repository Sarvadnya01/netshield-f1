"""Federated server: orchestrates rounds, aggregation, checkpointing."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from netshield.common.config import repo_root
from netshield.common.hardware import get_device
from netshield.common.labels import NUM_CLASSES
from netshield.models.evaluate import compute_metrics
from netshield.models.mlp import MLP

logger = logging.getLogger(__name__)


def weighted_aggregate(
    client_results: list[dict],
) -> dict[str, torch.Tensor]:
    """FedAvg-style weighted aggregation of client weights.

    Each client_result has keys: weights (dict), n_samples (int).
    Returns aggregated weights dict.
    """
    total_samples = sum(r["n_samples"] for r in client_results)
    agg: dict[str, torch.Tensor] = {}

    for key in client_results[0]["weights"]:
        weighted_sum = torch.zeros_like(client_results[0]["weights"][key], dtype=torch.float64)
        for r in client_results:
            w = r["n_samples"] / total_samples
            weighted_sum += r["weights"][key].double() * w
        agg[key] = weighted_sum.float()

    return agg


def _save_checkpoint(
    ckpt_dir: Path,
    rnd: int,
    global_weights: dict[str, torch.Tensor],
    history: list[dict],
) -> None:
    """Save a checkpoint for resumption."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(global_weights, ckpt_dir / "global_weights.pt")
    with open(ckpt_dir / "history.json", "w") as f:
        json.dump({"round": rnd, "history": history}, f, indent=2)
    logger.info("Checkpoint saved at round %d -> %s", rnd, ckpt_dir)


def _load_checkpoint(ckpt_dir: Path) -> tuple[int, dict[str, torch.Tensor], list[dict]] | None:
    """Load checkpoint if it exists. Returns (round, weights, history) or None."""
    weights_path = ckpt_dir / "global_weights.pt"
    history_path = ckpt_dir / "history.json"
    if not weights_path.exists() or not history_path.exists():
        return None
    global_weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    with open(history_path) as f:
        data = json.load(f)
    logger.info("Resuming from round %d checkpoint", data["round"])
    return data["round"], global_weights, data["history"]


def run_fl(
    run_name: str,
    clients: list,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    num_rounds: int,
    fraction_fit: float = 1.0,
    checkpoint_every: int = 5,
    seed: int = 42,
) -> tuple[dict[str, torch.Tensor], list[dict]]:
    """Run the federated learning loop.

    Args:
        run_name: Name for this FL run.
        clients: List of FLClient instances.
        X_val: Global validation features.
        y_val: Global validation labels.
        input_dim: Input feature dimension.
        hidden_dims: MLP hidden layer dims.
        dropout: Dropout rate.
        num_rounds: Total communication rounds.
        fraction_fit: Fraction of clients sampled each round.
        checkpoint_every: Checkpoint interval in rounds.
        seed: Random seed.

    Returns:
        (final_global_weights, history) tuple.
    """
    rng = np.random.RandomState(seed)
    device = torch.device(get_device())
    num_clients = len(clients)

    ckpt_dir = repo_root() / "checkpoints" / "fl" / run_name

    # Try to resume from checkpoint
    checkpoint = _load_checkpoint(ckpt_dir)
    if checkpoint is not None:
        start_round, global_weights, history = checkpoint
        start_round += 1  # resume from next round
    else:
        # Initialize global model
        model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout)
        torch.manual_seed(seed)
        model.apply(lambda m: None)  # ensure init is seeded
        global_weights = model.get_weights()
        history = []
        start_round = 1

    for rnd in range(start_round, num_rounds + 1):
        t0 = time.time()

        # Sample clients
        num_selected = max(1, int(fraction_fit * num_clients))
        selected = rng.choice(num_clients, size=num_selected, replace=False).tolist()

        # Train selected clients
        client_results = []
        for k in selected:
            result = clients[k].train_round(global_weights)
            client_results.append(result)

        # Aggregate
        global_weights = weighted_aggregate(client_results)

        # Evaluate global model on global val set
        val_metrics = _evaluate_global(
            global_weights, X_val, y_val, input_dim, hidden_dims, dropout, device,
        )

        # Compute per-client stats
        client_f1s = [r["val_macro_f1"] for r in client_results]
        client_losses = [r["loss"] for r in client_results]

        # Estimate bytes communicated (weights up + down)
        weight_bytes = sum(v.numel() * 4 for v in global_weights.values())  # float32
        mb_communicated = (weight_bytes * (num_selected + 1)) / (1024 ** 2)

        wall_time = time.time() - t0

        round_info = {
            "round": rnd,
            "global_val_macro_f1": val_metrics["macro_f1"],
            "global_val_accuracy": val_metrics["accuracy"],
            "mean_client_f1": float(np.mean(client_f1s)),
            "std_client_f1": float(np.std(client_f1s)),
            "worst_client_f1": float(np.min(client_f1s)),
            "mean_client_loss": float(np.mean(client_losses)),
            "client_f1s": client_f1s,
            "mb_communicated": round(mb_communicated, 2),
            "wall_time_s": round(wall_time, 2),
            "selected_clients": selected,
        }
        history.append(round_info)

        logger.info(
            "Round %d/%d  global_f1=%.4f  mean_client_f1=%.4f±%.4f  "
            "worst_f1=%.4f  %.1fs  %.1f MB",
            rnd, num_rounds,
            val_metrics["macro_f1"],
            np.mean(client_f1s), np.std(client_f1s),
            np.min(client_f1s),
            wall_time, mb_communicated,
        )

        # Checkpoint
        if rnd % checkpoint_every == 0 or rnd == num_rounds:
            _save_checkpoint(ckpt_dir, rnd, global_weights, history)

    return global_weights, history


def _evaluate_global(
    weights: dict[str, torch.Tensor],
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    hidden_dims: list[int],
    dropout: float,
    device: torch.device,
) -> dict:
    """Evaluate global model on validation set."""
    model = MLP(input_dim, NUM_CLASSES, hidden_dims, dropout)
    model.set_weights(weights)
    model.to(device)
    model.eval()

    with torch.no_grad():
        X_t = torch.from_numpy(X_val).float().to(device)
        logits = model(X_t).cpu()

    y_pred = logits.argmax(dim=1).numpy()
    y_probs = torch.softmax(logits, dim=1).numpy()
    return compute_metrics(y_val, y_pred, y_probs)
