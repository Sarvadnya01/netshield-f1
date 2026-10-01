"""XGBoost baseline for 8-class IoT attack classification."""

from __future__ import annotations

import logging
import time

import numpy as np
import xgboost as xgb

from netshield.common.config import get_config
from netshield.common.hardware import get_device
from netshield.common.labels import NUM_CLASSES

logger = logging.getLogger(__name__)


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> xgb.XGBClassifier:
    """Train an XGBoost classifier with early stopping on validation.

    Device and hyper-parameters come from the merged config profile.
    """
    cfg = get_config()
    xgb_cfg = cfg.get("xgb", {})
    device = get_device()  # "cuda" or "cpu"

    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 20)

    model = xgb.XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 500),
        max_depth=xgb_cfg.get("max_depth", 8),
        learning_rate=xgb_cfg.get("learning_rate", 0.1),
        objective="multi:softprob",
        num_class=NUM_CLASSES,
        tree_method="hist",
        device=device,
        random_state=seed,
        n_jobs=-1,
        verbosity=1,
        early_stopping_rounds=early_stopping_rounds,
    )

    logger.info(
        "XGB training: n_estimators=%d, max_depth=%d, device=%s, early_stop=%d",
        model.n_estimators, model.max_depth, device, early_stopping_rounds,
    )

    t0 = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    elapsed = time.time() - t0
    logger.info("XGB training done in %.1fs, best iteration: %d", elapsed, model.best_iteration)

    return model
