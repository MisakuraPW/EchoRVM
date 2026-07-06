"""Dataset IO helpers for EchoNet-Dynamic and CAMUS proxy experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def read_video(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        return data[data.files[0]]
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


def read_medical_image(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except Exception as exc:
        raise RuntimeError("CAMUS .mhd/.nii reading requires SimpleITK.") from exc
    image = sitk.ReadImage(str(path))
    arr = sitk.GetArrayFromImage(image)
    return np.asarray(arr)


def to_grayscale(video: np.ndarray) -> np.ndarray:
    arr = np.asarray(video)
    if arr.ndim == 2:
        return arr[None]
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1)[None]
        return arr
    if arr.ndim == 4:
        if arr.shape[-1] in (3, 4):
            return arr[..., :3].mean(axis=-1)
        if arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
            return arr[..., :3].mean(axis=-1)
    raise ValueError(f"Unsupported array shape for grayscale conversion: {arr.shape}")


def normalize_float(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    max_seen = float(np.nanmax(arr)) if arr.size else 1.0
    if max_seen > 2.0:
        arr = arr / 255.0 if max_seen <= 255.0 else arr / max_seen
    return np.clip(arr, 0.0, 1.0)


def sample_frames(video: np.ndarray, num_frames: int) -> np.ndarray:
    video = np.asarray(video)
    t = video.shape[0]
    if t <= 0:
        raise ValueError("Cannot sample from empty video")
    if t == num_frames:
        return video
    idx = np.linspace(0, t - 1, num_frames)
    idx = np.rint(idx).astype(np.int64)
    return video[idx]


def find_echonet_video(root: Path, file_name: str) -> Path | None:
    stem = Path(str(file_name)).stem
    candidates = [
        root / "Videos" / f"{stem}.avi",
        root / "videos" / f"{stem}.avi",
        root / "a4c-video-dir" / f"{stem}.avi",
        root / "video" / f"{stem}.npy",
        root / "npy" / f"{stem}.npy",
        root / f"{stem}.avi",
        root / f"{stem}.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob(f"{stem}.avi")) + list(root.rglob(f"{stem}.npy"))
    return matches[0] if matches else None


def load_echonet_filelist(root: Path, split: str = "TRAIN") -> pd.DataFrame:
    csv_path = root / "FileList.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"EchoNet FileList.csv not found: {csv_path}")
    df = pd.read_csv(csv_path)
    split_col = "Split" if "Split" in df.columns else None
    if split_col:
        df = df[df[split_col].astype(str).str.upper() == split.upper()]
    return df.reset_index(drop=True)


def camus_pairs(root: Path, split: str = "training") -> list[dict[str, str]]:
    search_roots = [root / split, root] if (root / split).exists() else [root]
    rows: list[dict[str, str]] = []
    for base in search_roots:
        for img in sorted(base.rglob("*.mhd")) + sorted(base.rglob("*.nii")) + sorted(base.rglob("*.nii.gz")):
            name = img.name
            lower = name.lower()
            if "_gt" in lower or "sequence" in lower:
                continue
            if not any(tag in lower for tag in ("_ed", "_es")):
                continue
            if name.endswith(".nii.gz"):
                mask = img.with_name(name[:-7] + "_gt.nii.gz")
            else:
                mask = img.with_name(img.stem + "_gt" + img.suffix)
            if mask.exists():
                rows.append({"image": str(img), "mask": str(mask), "dataset": "camus"})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
