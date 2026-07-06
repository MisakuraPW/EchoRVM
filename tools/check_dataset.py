"""Lightweight dataset layout checker for EchoNet-Dynamic, CAMUS, and EchoRisk."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_ROOTS = {
    "echonet": Path("/root/autodl-tmp/datasets/EchoNet-Dynamic"),
    "camus": Path("/root/autodl-tmp/datasets/CAMUS"),
    "echorisk": Path("/root/autodl-tmp/datasets/EchoRisk"),
}


def status(ok: bool, label: str, detail: str) -> None:
    tag = "[OK]" if ok else "[WARN]"
    print(f"{tag} {label}: {detail}")


def count_files(root: Path, patterns: list[str], limit: int = 1_000_000) -> int:
    if not root.exists():
        return 0
    total = 0
    for pattern in patterns:
        for _ in root.rglob(pattern):
            total += 1
            if total >= limit:
                return total
    return total


def check_echonet(root: Path) -> bool:
    print(f"\n===== EchoNet-Dynamic: {root} =====")
    ok = root.exists()
    status(ok, "root exists", str(root))
    status((root / "FileList.csv").exists(), "FileList.csv", str(root / "FileList.csv"))
    status((root / "VolumeTracings.csv").exists(), "VolumeTracings.csv", str(root / "VolumeTracings.csv"))
    video_dirs = [root / "Videos", root / "videos", root]
    avi_count = sum(count_files(path, ["*.avi"]) for path in video_dirs if path.exists())
    status(avi_count > 0, "AVI videos", str(avi_count))
    npy_count = count_files(root, ["*.npy"])
    status(npy_count > 0, "preprocessed npy files", str(npy_count))
    return ok


def check_camus(root: Path) -> bool:
    print(f"\n===== CAMUS: {root} =====")
    ok = root.exists()
    status(ok, "root exists", str(root))
    mhd_count = count_files(root, ["*.mhd"])
    raw_count = count_files(root, ["*.raw"])
    nii_count = count_files(root, ["*.nii", "*.nii.gz"])
    status(mhd_count > 0 or nii_count > 0, "image metadata files", f"mhd={mhd_count}, nii={nii_count}")
    status(raw_count > 0 or nii_count > 0, "raw image payloads", f"raw={raw_count}, nii={nii_count}")
    cfg_count = count_files(root, ["Info_*.cfg", "*.cfg"])
    status(cfg_count > 0, "patient cfg files", str(cfg_count))
    return ok


def check_echorisk(root: Path) -> bool:
    print(f"\n===== EchoRisk: {root} =====")
    ok = root.exists()
    status(ok, "root exists", str(root))
    dcm_count = count_files(root, ["*.dcm", "*.dicom"])
    csv_count = count_files(root, ["*.csv"])
    status(dcm_count > 0, "DICOM files", str(dcm_count))
    status(csv_count > 0, "metadata csv files", str(csv_count))
    if not ok or dcm_count == 0:
        status(False, "access note", "EchoRisk may be unavailable until Synapse permission is approved.")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Check dataset roots for the MAE project.")
    parser.add_argument("--dataset", choices=["all", "echonet", "camus", "echorisk"], default="all")
    parser.add_argument("--root", default=None, help="Override root when checking a single dataset.")
    args = parser.parse_args()

    targets = ["echonet", "camus", "echorisk"] if args.dataset == "all" else [args.dataset]
    all_existing = True
    for name in targets:
        root = Path(args.root) if args.root and len(targets) == 1 else DEFAULT_ROOTS[name]
        if name == "echonet":
            all_existing = check_echonet(root) and all_existing
        elif name == "camus":
            all_existing = check_camus(root) and all_existing
        elif name == "echorisk":
            all_existing = check_echorisk(root) and all_existing

    print("\n===== Dataset Check Finished =====")
    return 0 if all_existing else 1


if __name__ == "__main__":
    raise SystemExit(main())
