"""Tests for netshield.common.config."""

import os

import pytest

from netshield.common.config import _deep_merge, active_profile, get_config, repo_root


def test_repo_root():
    root = repo_root()
    assert root.exists()
    assert (root / "pyproject.toml").exists()


def test_active_profile_default(monkeypatch):
    monkeypatch.delenv("NETSHIELD_PROFILE", raising=False)
    assert active_profile() == "laptop"


def test_active_profile_env(monkeypatch):
    monkeypatch.setenv("NETSHIELD_PROFILE", "lab")
    assert active_profile() == "lab"


def test_deep_merge_basic():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99, "e": 5}, "f": 6}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": {"c": 99, "d": 3, "e": 5}, "f": 6}


def test_deep_merge_no_mutate():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}}
    _deep_merge(base, override)
    assert "c" not in base["a"]


def test_get_config_returns_dict(monkeypatch):
    monkeypatch.setenv("NETSHIELD_PROFILE", "laptop")
    get_config.cache_clear()
    cfg = get_config()
    assert isinstance(cfg, dict)
    # Should have merged the laptop profile
    assert cfg.get("training", {}).get("train_max_rows") == 300000


def test_get_config_lab_profile(monkeypatch):
    monkeypatch.setenv("NETSHIELD_PROFILE", "lab")
    get_config.cache_clear()
    cfg = get_config()
    assert cfg.get("training", {}).get("train_max_rows") is None
    assert cfg.get("training", {}).get("batch_size") == 8192


def test_env_override_kafka(monkeypatch):
    monkeypatch.setenv("NETSHIELD_PROFILE", "laptop")
    monkeypatch.setenv("KAFKA_BOOTSTRAP", "custom-host:9999")
    get_config.cache_clear()
    cfg = get_config()
    assert cfg["kafka"]["bootstrap_servers"] == "custom-host:9999"


def test_env_override_mlflow(monkeypatch):
    monkeypatch.setenv("NETSHIELD_PROFILE", "laptop")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    get_config.cache_clear()
    cfg = get_config()
    assert cfg["mlflow"]["tracking_uri"] == "http://mlflow:5000"
