"""Logging configuration for NetShield-FL."""

from __future__ import annotations

import logging

from netshield.common.config import repo_root


def setup_logging(name: str = "netshield", level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger that writes to console and logs/ directory."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    log_dir = repo_root() / "logs"
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / f"{name}.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
