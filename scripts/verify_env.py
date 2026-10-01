"""Verify the NetShield-FL environment: versions, CUDA, GPU, VRAM, profile."""

from __future__ import annotations

import sys


def main() -> None:
    print("=== NetShield-FL Environment Verification ===\n")

    # Python version
    print(f"Python:        {sys.version}")

    # Key packages
    pkgs = {
        "numpy": "numpy",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
        "pydantic": "pydantic",
        "scikit-learn": "sklearn",
        "xgboost": "xgboost",
        "onnx": "onnx",
        "onnxruntime": "onnxruntime",
        "mlflow": "mlflow",
        "fastapi": "fastapi",
        "streamlit": "streamlit",
        "pyyaml": "yaml",
    }
    for name, mod in pkgs.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"{name:15s}{ver}")
        except ImportError:
            print(f"{name:15s}NOT INSTALLED")

    # PyTorch (optional)
    try:
        import torch

        print(f"{'torch':15s}{torch.__version__}")
        print(f"{'CUDA available':15s}{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"{'CUDA version':15s}{torch.version.cuda}")
    except ImportError:
        print(f"{'torch':15s}NOT INSTALLED")

    print()

    # Hardware report
    try:
        from netshield.common.hardware import print_report

        print_report()
    except Exception as e:
        print(f"Hardware check failed: {e}")


if __name__ == "__main__":
    main()
