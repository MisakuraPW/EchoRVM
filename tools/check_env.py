"""Check the local or AutoDL runtime for the MAE project."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from pathlib import Path


def status(ok: bool, label: str, detail: str) -> None:
    tag = "[OK]" if ok else "[WARN]"
    print(f"{tag} {label}: {detail}")


def import_version(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - diagnostic path
        return False, str(exc)
    return True, getattr(module, "__version__", "installed")


def check_torch() -> bool:
    try:
        import torch
    except Exception as exc:
        status(False, "PyTorch", f"not importable ({exc})")
        return False

    status(True, "PyTorch", torch.__version__)
    cuda_ok = torch.cuda.is_available()
    status(cuda_ok, "CUDA available", str(cuda_ok))
    if cuda_ok:
        status(True, "CUDA version", str(torch.version.cuda))
        status(True, "GPU", torch.cuda.get_device_name(0))
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        status(True, "GPU memory", f"{total_gb:.1f} GB")
    return cuda_ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check MAE project environment.")
    parser.add_argument("--data-root", default="/root/autodl-tmp/datasets")
    parser.add_argument("--output-root", default="/root/autodl-tmp/outputs")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when critical GPU/runtime checks fail.")
    args = parser.parse_args()

    project_root = Path.cwd()
    print("===== Environment Check =====")
    status(True, "Project root", str(project_root))
    status((project_root / "docs").exists(), "docs directory", str(project_root / "docs"))
    status((project_root / "configs").exists(), "configs directory", str(project_root / "configs"))
    status(True, "Python", sys.version.replace("\n", " "))
    status(True, "Platform", platform.platform())

    critical_ok = check_torch()
    for package in [
        "numpy",
        "pandas",
        "yaml",
        "tqdm",
        "matplotlib",
        "cv2",
        "skimage",
        "SimpleITK",
        "timm",
        "tensorboard",
    ]:
        ok, detail = import_version(package)
        status(ok, package, detail)
        critical_ok = critical_ok and ok if package in {"numpy", "yaml", "tqdm"} else critical_ok

    for path_label, path_value in [
        ("data_root", args.data_root),
        ("output_root", args.output_root),
    ]:
        path = Path(path_value)
        status(path.exists(), path_label, str(path))

    if os.name == "nt":
        status(False, "Runtime note", "Windows is fine for development; AutoDL training should run on Linux.")

    print("===== Check Finished =====")
    return 0 if critical_ok or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
