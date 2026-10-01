"""Configuration loader with profile-based deep merge and env-var overrides."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONFIG_FILES = ["data.yaml", "train.yaml", "fl.yaml", "stream.yaml"]

_ENV_OVERRIDES: dict[str, list[str]] = {
    "KAFKA_BOOTSTRAP": ["kafka", "bootstrap_servers"],
    "MLFLOW_TRACKING_URI": ["mlflow", "tracking_uri"],
}


def repo_root() -> Path:
    """Return the repository root (parent of src/)."""
    return Path(__file__).resolve().parents[3]


def active_profile() -> str:
    """Return the active profile name from NETSHIELD_PROFILE env var."""
    return os.environ.get("NETSHIELD_PROFILE", "laptop")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _apply_env_overrides(cfg: dict) -> dict:
    """Apply environment variable overrides to the config dict."""
    for env_var, key_path in _ENV_OVERRIDES.items():
        value = os.environ.get(env_var)
        if value is not None:
            d = cfg
            for key in key_path[:-1]:
                d = d.setdefault(key, {})
            d[key_path[-1]] = value
    return cfg


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, returning empty dict if file is empty."""
    if not path.exists():
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    """Load all config files, deep-merge the active profile, apply env overrides.

    Returns the merged configuration dictionary.
    """
    root = repo_root()
    config_dir = root / "configs"

    # Load base configs
    cfg: dict[str, Any] = {}
    for fname in _CONFIG_FILES:
        cfg = _deep_merge(cfg, _load_yaml(config_dir / fname))

    # Deep-merge profile
    profile = active_profile()
    profile_path = config_dir / "profiles" / f"{profile}.yaml"
    if profile_path.exists():
        profile_cfg = _load_yaml(profile_path)
        cfg = _deep_merge(cfg, profile_cfg)
        logger.info("Loaded profile: %s (%s)", profile, profile_path)
    else:
        logger.warning("Profile '%s' not found at %s, using base config only", profile, profile_path)

    # Apply env-var overrides
    cfg = _apply_env_overrides(cfg)

    return cfg
