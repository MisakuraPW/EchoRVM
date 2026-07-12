"""Dataset IO helpers for EchoNet-Dynamic and CAMUS proxy experiments."""

from __future__ import annotations

import json
import re
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
        root / "video" / f"{stem}.npy",
        root / "npy" / f"{stem}.npy",
        root / f"{stem}.npy",
        root / "Videos" / f"{stem}.avi",
        root / "videos" / f"{stem}.avi",
        root / "a4c-video-dir" / f"{stem}.avi",
        root / f"{stem}.avi",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = list(root.rglob(f"{stem}.npy")) + list(root.rglob(f"{stem}.avi"))
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


_CAMUS_SPLIT_ALIASES = {
    "train": {"train", "training", "tr"},
    "val": {"val", "valid", "validation", "dev"},
    "test": {"test", "testing", "te"},
}


def _canonical_camus_split(split: str) -> str:
    split_l = str(split).lower()
    for canonical, aliases in _CAMUS_SPLIT_ALIASES.items():
        if split_l in aliases:
            return canonical
    return split_l


def _strip_medical_suffix(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".nii.gz"):
        return name[:-7]
    return Path(name).stem


def _camus_patient_id(path: Path) -> str:
    text = f"{path.parent.name}_{path.name}".lower()
    match = re.search(r"patient[_-]?(\d+)", text)
    if match:
        return f"patient{int(match.group(1)):04d}"
    stem = _strip_medical_suffix(path.name).lower()
    for tag in ("_2ch", "_4ch", "_ed", "_es"):
        if tag in stem:
            stem = stem.split(tag)[0]
    return stem


def _normalise_camus_patient_token(value: str) -> str | None:
    token = str(value).strip().lower()
    if not token:
        return None
    match = re.search(r"patient[_-]?(\d+)", token)
    if match:
        return f"patient{int(match.group(1)):04d}"
    if re.fullmatch(r"\d{1,4}", token):
        return f"patient{int(token):04d}"
    return None


def _camus_view_and_frame(path: Path) -> tuple[str, str]:
    name = _strip_medical_suffix(path.name).lower()
    view = "2CH" if "2ch" in name else "4CH" if "4ch" in name else ""
    frame = "ED" if "_ed" in name else "ES" if "_es" in name else ""
    return view, frame


def _extract_patient_ids(text: str) -> set[str]:
    ids = set()
    for match in re.finditer(r"patient[_-]?(\d+)", text.lower()):
        ids.add(f"patient{int(match.group(1)):04d}")
    for token in re.split(r"[\s,;]+", text):
        patient = _normalise_camus_patient_token(token)
        if patient:
            ids.add(patient)
    return ids


def _load_camus_split_ids(root: Path, split: str) -> set[str] | None:
    split_dir = root / "database_split"
    if not split_dir.exists():
        return None
    canonical = _canonical_camus_split(split)
    aliases = _CAMUS_SPLIT_ALIASES.get(canonical, {canonical})
    ids: set[str] = set()

    # Common layouts: train.txt / validation.csv / testing.json, or nested dirs.
    files = [p for p in split_dir.rglob("*") if p.is_file()]
    named_files = [
        p for p in files
        if any(alias in p.stem.lower() or alias in str(p.parent.name).lower() for alias in aliases)
    ]
    for path in named_files:
        try:
            ids.update(_extract_patient_ids(path.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            continue
    if ids:
        return ids

    # Fallback for a single CSV/TSV table with split and patient columns.
    for path in files:
        if path.suffix.lower() not in {".csv", ".tsv", ".txt"}:
            continue
        try:
            sep = "\t" if path.suffix.lower() == ".tsv" else None
            df = pd.read_csv(path, sep=sep, engine="python")
        except Exception:
            continue
        lower_cols = {str(col).lower(): col for col in df.columns}
        split_col = next((lower_cols[c] for c in ("split", "set", "subset") if c in lower_cols), None)
        id_col = next((lower_cols[c] for c in ("patient", "patient_id", "id", "name") if c in lower_cols), None)
        if split_col is None or id_col is None:
            continue
        mask = df[split_col].astype(str).str.lower().isin(aliases)
        for value in df.loc[mask, id_col].astype(str):
            extracted = _extract_patient_ids(value)
            patient = _normalise_camus_patient_token(value)
            ids.update(extracted or ({patient} if patient else {_camus_patient_id(Path(value))}))
    return ids


def _camus_search_roots(root: Path, split: str) -> tuple[list[Path], str]:
    split_path = root / split
    if split_path.exists():
        return [split_path], "directory"
    canonical = _canonical_camus_split(split)
    for alias in _CAMUS_SPLIT_ALIASES.get(canonical, {canonical}):
        alias_path = root / alias
        if alias_path.exists():
            return [alias_path], "directory"
    nifti = root / "database_nifti"
    if nifti.exists():
        return [nifti], "database_nifti"
    return [root], "scan"


def camus_pairs(root: Path, split: str = "training") -> list[dict[str, str]]:
    split_ids = _load_camus_split_ids(root, split)
    search_roots, split_source = _camus_search_roots(root, split)
    rows: list[dict[str, str]] = []
    for base in search_roots:
        for img in sorted(base.rglob("*.mhd")) + sorted(base.rglob("*.nii")) + sorted(base.rglob("*.nii.gz")):
            name = img.name
            lower = name.lower()
            if "_gt" in lower or "sequence" in lower:
                continue
            if not any(tag in lower for tag in ("_ed", "_es")):
                continue
            patient = _camus_patient_id(img)
            if split_ids is not None and patient not in split_ids:
                continue
            if name.endswith(".nii.gz"):
                mask = img.with_name(name[:-7] + "_gt.nii.gz")
            else:
                mask = img.with_name(img.stem + "_gt" + img.suffix)
            if mask.exists():
                view, frame = _camus_view_and_frame(img)
                rows.append(
                    {
                        "image": str(img),
                        "mask": str(mask),
                        "dataset": "camus",
                        "patient": patient,
                        "view": view,
                        "frame": frame,
                        "split": _canonical_camus_split(split),
                        "split_source": "database_split" if split_ids is not None else split_source,
                    }
                )
    seen = set()
    unique = []
    for row in rows:
        key = row["image"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
