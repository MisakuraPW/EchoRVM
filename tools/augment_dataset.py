"""Offline ultrasound augmentation for EchoNet-Dynamic and CAMUS.

This script writes augmented .npy caches. It is intended for AutoDL workflows
where raw data lives on /root/autodl-fs and code runs from /root/autodl-tmp.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from augment.ultrasound import EchoAugmentConfig, EchoClipAugmentor, ensure_video_array


VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv"}
ARRAY_SUFFIXES = {".npy", ".npz"}
VOLUME_SUFFIXES = {".mhd", ".nii", ".gz"}


def read_video(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def read_array(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        key = data.files[0]
        return data[key]
    return np.load(path)


def read_medical_volume(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except Exception as exc:  # pragma: no cover - cloud diagnostic path
        raise RuntimeError("Reading CAMUS .mhd/.nii requires SimpleITK. Install requirements.txt first.") from exc
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return arr


def read_sample(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        return read_video(path)
    if suffix in ARRAY_SUFFIXES:
        return read_array(path)
    if suffix in {".mhd", ".nii"} or path.name.lower().endswith(".nii.gz"):
        return read_medical_volume(path)
    raise ValueError(f"Unsupported input file: {path}")


def candidate_files(dataset: str, input_root: Path) -> list[Path]:
    if dataset == "echonet":
        roots = [
            input_root / "Videos",
            input_root / "videos",
            input_root / "a4c-video-dir",
            input_root / "npy",
            input_root,
        ]
        suffixes = VIDEO_SUFFIXES | ARRAY_SUFFIXES
    elif dataset == "camus":
        roots = [input_root]
        suffixes = {".mhd", ".nii", ".gz"} | ARRAY_SUFFIXES
    else:
        roots = [input_root]
        suffixes = VIDEO_SUFFIXES | ARRAY_SUFFIXES | {".mhd", ".nii", ".gz"}

    files: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            name = path.name.lower()
            suffix = path.suffix.lower()
            if dataset == "camus" and ("_gt" in name or "sequence_gt" in name):
                continue
            if suffix in suffixes or name.endswith(".nii.gz"):
                resolved = path.resolve()
                if resolved not in seen:
                    files.append(path)
                    seen.add(resolved)
    return sorted(files)


def output_path_for(path: Path, input_root: Path, output_root: Path, variant: int) -> Path:
    try:
        rel = path.relative_to(input_root)
    except ValueError:
        rel = Path(path.name)
    stem = rel.with_suffix("")
    if path.name.lower().endswith(".nii.gz"):
        stem = rel.with_name(rel.name[:-7])
    return output_root / f"variant_{variant:02d}" / stem.with_suffix(".npy")


def write_manifest(rows: Iterable[dict[str, str]], output_root: Path) -> None:
    rows = list(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_root / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "variant", "source", "output", "shape", "applied"])
        writer.writeheader()
        writer.writerows(rows)


def build_config(args: argparse.Namespace) -> EchoAugmentConfig:
    return EchoAugmentConfig(
        img_size=args.img_size,
        tgc_prob=args.tgc_prob,
        gamma_contrast_prob=args.gamma_contrast_prob,
        brightness_prob=args.brightness_prob,
        zoom_prob=args.zoom_prob,
        blur_prob=args.blur_prob,
        shadow_prob=args.shadow_prob,
        speckle_prob=args.speckle_prob,
        preserve_dtype=not args.float32,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline ultrasound augmentation for EchoNet/CAMUS.")
    parser.add_argument("--dataset", choices=["echonet", "camus", "generic"], required=True)
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--float32", action="store_true", help="Save float32 [0, 1] instead of preserving source dtype.")
    parser.add_argument("--tgc-prob", type=float, default=0.4)
    parser.add_argument("--gamma-contrast-prob", type=float, default=0.4)
    parser.add_argument("--brightness-prob", type=float, default=0.3)
    parser.add_argument("--zoom-prob", type=float, default=0.3)
    parser.add_argument("--blur-prob", type=float, default=0.15)
    parser.add_argument("--shadow-prob", type=float, default=0.1)
    parser.add_argument("--speckle-prob", type=float, default=0.4)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    files = candidate_files(args.dataset, input_root)
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"[ERROR] No supported files found under {input_root}", file=sys.stderr)
        return 2

    cfg = build_config(args)
    manifest: list[dict[str, str]] = []
    failures = 0
    for variant in range(args.variants):
        augmentor = EchoClipAugmentor(cfg, seed=args.seed + variant)
        for src in tqdm(files, desc=f"{args.dataset} variant {variant:02d}"):
            dst = output_path_for(src, input_root, output_root, variant)
            meta_path = dst.with_suffix(".json")
            if dst.exists() and not args.overwrite:
                continue
            try:
                sample = read_sample(src)
                sample = ensure_video_array(sample)
                augmented, meta = augmentor(sample, return_meta=True)
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst, augmented)
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(meta | {"source": str(src)}, f, indent=2, ensure_ascii=False)
                manifest.append(
                    {
                        "dataset": args.dataset,
                        "variant": str(variant),
                        "source": str(src),
                        "output": str(dst),
                        "shape": "x".join(map(str, augmented.shape)),
                        "applied": ",".join(meta["applied"]),
                    }
                )
            except Exception as exc:
                failures += 1
                print(f"[WARN] Failed {src}: {exc}", file=sys.stderr)

    write_manifest(manifest, output_root)
    print(f"Finished. wrote={len(manifest)} failures={failures} output={output_root}")
    return 1 if failures and not manifest else 0


if __name__ == "__main__":
    raise SystemExit(main())
