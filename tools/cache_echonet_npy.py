"""Cache EchoNet-Dynamic videos as fast .npy arrays.

This script does not apply augmentation. It only decodes AVI/MP4 once and
writes plain uint8 NumPy arrays for faster online MAE training. Training-time
A4 augmentation still happens in the Dataset/DataLoader path.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


VIDEO_DIRS = ("Videos", "videos", "a4c-video-dir")
VIDEO_SUFFIXES = (".avi", ".mp4", ".mov", ".mkv")


def find_source_video(root: Path, file_name: str) -> Path | None:
    stem = Path(str(file_name)).stem
    for dirname in VIDEO_DIRS:
        for suffix in VIDEO_SUFFIXES:
            path = root / dirname / f"{stem}{suffix}"
            if path.exists():
                return path
    for suffix in VIDEO_SUFFIXES:
        path = root / f"{stem}{suffix}"
        if path.exists():
            return path
    for suffix in VIDEO_SUFFIXES:
        matches = list(root.rglob(f"{stem}{suffix}"))
        if matches:
            return matches[0]
    return None


def read_video(path: Path, grayscale: bool) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError(f"No frames read from {path}")
    return np.stack(frames, axis=0)


def cache_one(task: dict[str, Any]) -> dict[str, Any]:
    src = Path(task["src"])
    dst = Path(task["dst"])
    grayscale = bool(task["grayscale"])
    overwrite = bool(task["overwrite"])
    if dst.exists() and not overwrite:
        arr = np.load(dst, mmap_mode="r")
        return {"status": "skipped", "src": str(src), "dst": str(dst), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    arr = read_video(src, grayscale=grayscale)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, arr)
    tmp.replace(dst)
    return {"status": "written", "src": str(src), "dst": str(dst), "shape": list(arr.shape), "dtype": str(arr.dtype)}


def copy_metadata(input_root: Path, output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("FileList.csv", "VolumeTracings.csv"):
        src = input_root / name
        if src.exists():
            dst = output_root / name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)


def build_tasks(input_root: Path, output_root: Path, limit: int | None, grayscale: bool, overwrite: bool) -> list[dict[str, Any]]:
    filelist = input_root / "FileList.csv"
    if not filelist.exists():
        raise FileNotFoundError(f"EchoNet FileList.csv not found: {filelist}")
    df = pd.read_csv(filelist)
    if "FileName" not in df.columns:
        raise ValueError(f"FileList.csv must contain FileName column: {filelist}")
    if limit is not None:
        df = df.iloc[:limit]
    tasks = []
    missing = []
    for file_name in df["FileName"].astype(str):
        stem = Path(file_name).stem
        src = find_source_video(input_root, stem)
        if src is None:
            missing.append(stem)
            continue
        tasks.append(
            {
                "src": str(src),
                "dst": str(output_root / "npy" / f"{stem}.npy"),
                "grayscale": grayscale,
                "overwrite": overwrite,
            }
        )
    if missing:
        preview = ", ".join(missing[:10])
        print(f"[WARN] missing videos: {len(missing)} preview={preview}", file=sys.stderr)
    if not tasks:
        raise RuntimeError(f"No EchoNet videos found under {input_root}")
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Decode EchoNet-Dynamic videos into npy cache for fast training.")
    parser.add_argument("--input-root", required=True, help="Original EchoNet-Dynamic root, usually on /root/autodl-fs.")
    parser.add_argument("--output-root", required=True, help="Cache root, usually /root/autodl-tmp/datasets/EchoNet-Dynamic.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rgb", action="store_true", help="Save RGB [T,H,W,3]. Default saves grayscale [T,H,W].")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    grayscale = not args.rgb
    copy_metadata(input_root, output_root)
    tasks = build_tasks(input_root, output_root, args.limit, grayscale=grayscale, overwrite=args.overwrite)

    written = 0
    skipped = 0
    failed = 0
    manifest_path = output_root / "npy_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as manifest:
        if args.num_workers <= 1:
            iterator = (cache_one(task) for task in tasks)
            for result in tqdm(iterator, total=len(tasks), desc="cache echonet npy"):
                manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                written += result["status"] == "written"
                skipped += result["status"] == "skipped"
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
                futures = [pool.submit(cache_one, task) for task in tasks]
                for future in tqdm(as_completed(futures), total=len(futures), desc="cache echonet npy"):
                    try:
                        result = future.result()
                        manifest.write(json.dumps(result, ensure_ascii=False) + "\n")
                        written += result["status"] == "written"
                        skipped += result["status"] == "skipped"
                    except Exception as exc:
                        failed += 1
                        print(f"[WARN] cache failed: {exc}", file=sys.stderr)

    print(
        f"Finished EchoNet npy cache: written={written} skipped={skipped} failed={failed} "
        f"output={output_root} manifest={manifest_path}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
