"""Hardware detection: CUDA device, GPU name, VRAM, system RAM."""

from __future__ import annotations

import logging
import platform
import sys

import psutil

logger = logging.getLogger(__name__)


def get_system_ram_gb() -> float:
    """Return total system RAM in GB."""
    return psutil.virtual_memory().total / (1024**3)


def get_cuda_info() -> dict:
    """Return CUDA device info: available, device_name, vram_gb, device_index."""
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            vram_bytes = getattr(props, "total_memory", None) or props.total_mem
            vram_gb = vram_bytes / (1024**3)
            return {
                "available": True,
                "device_index": idx,
                "device_name": name,
                "vram_gb": round(vram_gb, 2),
            }
    except ImportError:
        pass
    return {"available": False, "device_index": None, "device_name": None, "vram_gb": 0.0}


def get_device() -> str:
    """Return 'cuda' if CUDA is available, else 'cpu'."""
    info = get_cuda_info()
    return "cuda" if info["available"] else "cpu"


def should_put_data_on_gpu(n_bytes: int) -> bool:
    """Decide whether to put a dataset on GPU based on size and config.

    Uses the training.data_on_gpu config value:
    - true: always GPU
    - false: always CPU
    - auto: GPU if dataset fits in 60% of VRAM
    """
    from netshield.common.config import get_config

    cfg = get_config()
    setting = cfg.get("training", {}).get("data_on_gpu", "auto")

    if setting is True:
        return True
    if setting is False:
        return False

    # auto: check if data fits in 60% of VRAM
    info = get_cuda_info()
    if not info["available"]:
        return False
    vram_bytes = info["vram_gb"] * (1024**3)
    return n_bytes < vram_bytes * 0.6


def print_report() -> None:
    """Print hardware report to stdout."""
    from netshield.common.config import active_profile

    profile = active_profile()
    ram = get_system_ram_gb()
    cuda = get_cuda_info()

    print(f"Profile:    {profile}")
    print(f"Platform:   {platform.system()} {platform.release()}")
    print(f"Python:     {sys.version}")
    print(f"System RAM: {ram:.1f} GB")
    print(f"CUDA:       {'available' if cuda['available'] else 'not available'}")

    if cuda["available"]:
        print(f"GPU:        {cuda['device_name']}")
        print(f"VRAM:       {cuda['vram_gb']:.1f} GB")

        if profile == "lab" and cuda["vram_gb"] < 16:
            logger.warning(
                "Profile is 'lab' but VRAM is only %.1f GB (expected >= 16 GB)", cuda["vram_gb"]
            )
            print("WARNING: Profile=lab but VRAM < 16 GB!")
        elif profile == "laptop" and cuda["vram_gb"] >= 16:
            logger.warning(
                "Profile is 'laptop' but GPU has %.1f GB VRAM — consider 'lab' profile",
                cuda["vram_gb"],
            )
            print("WARNING: Profile=laptop on a large GPU — consider 'lab' profile.")
    else:
        print("WARNING: No CUDA GPU detected.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print_report()
