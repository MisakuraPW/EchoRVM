"""Downstream datasets for EchoRMAE fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from augment.ultrasound import resize_with_pad
from echo_aug_validation.augment_recipes import augment_image_mask, augment_video, resize_mask_with_pad
from echo_aug_validation.io_utils import (
    camus_pairs,
    find_echonet_video,
    load_echonet_filelist,
    normalize_float,
    read_medical_image,
    read_video,
    sample_frames,
    to_grayscale,
)


def _row_stem(value: object) -> str:
    return Path(str(value)).stem


def _patient_key(row: dict[str, str]) -> str:
    path = Path(row["image"])
    name = path.name.lower()
    for tag in ("_2ch", "_4ch", "_ed", "_es"):
        if tag in name:
            return name.split(tag)[0]
    return path.parent.name.lower()


def _split_rows(rows: list[dict[str, str]], split: str, val_fraction: float, seed: int) -> list[dict[str, str]]:
    split_l = split.lower()
    if split_l in {"all", "full"}:
        return rows
    keys = sorted({_patient_key(row) for row in rows})
    if not keys:
        return rows
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(keys, dtype=object)
    rng.shuffle(shuffled)
    val_n = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 1
    val_keys = set(str(x) for x in shuffled[:val_n])
    want_val = split_l in {"val", "valid", "validation"}
    out = [row for row in rows if (_patient_key(row) in val_keys) == want_val]
    return out if out else rows


def _video_to_tensor(video: np.ndarray, frames: int, img_size: int) -> torch.Tensor:
    sampled = sample_frames(video, frames)
    sampled = resize_with_pad(sampled, img_size)
    if sampled.ndim == 4 and sampled.shape[-1] == 1:
        gray = sampled[..., 0]
    elif sampled.ndim == 4 and sampled.shape[-1] in (3, 4):
        gray = sampled[..., :3].mean(axis=-1)
    elif sampled.ndim == 4 and sampled.shape[1] in (1, 3, 4):
        gray = sampled[:, :3].mean(axis=1)
    else:
        gray = to_grayscale(sampled)
    gray = normalize_float(gray)
    return torch.from_numpy(gray[:, None]).float()


def _image_to_tensor(image: np.ndarray, img_size: int) -> torch.Tensor:
    arr = resize_with_pad(image, img_size)
    gray = normalize_float(to_grayscale(arr)[0])
    return torch.from_numpy(gray[None]).float()


def _fill_trace_mask(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import binary_fill_holes

        return binary_fill_holes(mask > 0).astype(np.uint8)
    except Exception:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros_like(mask, dtype=np.uint8)
        cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
        return filled


def _rasterize_echonet_trace(group: pd.DataFrame, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    required = {"X1", "Y1", "X2", "Y2"}
    if not required.issubset(group.columns):
        raise ValueError(f"VolumeTracings.csv must contain columns {sorted(required)}")
    if len(group) < 2:
        return mask

    # Match the official EchoNet-Dynamic loader and EchoCardMAE conversion
    # script exactly: the two traced LV borders are joined into one polygon,
    # skipping the first paired point before rasterization.
    x1 = group["X1"].to_numpy(dtype=np.float32)
    y1 = group["Y1"].to_numpy(dtype=np.float32)
    x2 = group["X2"].to_numpy(dtype=np.float32)
    y2 = group["Y2"].to_numpy(dtype=np.float32)
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    try:
        import skimage.draw

        r, c = skimage.draw.polygon(np.rint(y).astype(np.int64), np.rint(x).astype(np.int64), shape)
        mask[r, c] = 1
    except Exception:
        contour = np.stack([x, y], axis=1)
        contour[:, 0] = np.clip(contour[:, 0], 0, w - 1)
        contour[:, 1] = np.clip(contour[:, 1], 0, h - 1)
        cv2.fillPoly(mask, [np.rint(contour).astype(np.int32)], 1)
    return mask


class EchoNetEFDataset(Dataset):
    """EchoNet-Dynamic EF regression dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        frames: int,
        img_size: int,
        aug_cfg=None,
        per_frame_random: bool = False,
        seed: int = 0,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.df = load_echonet_filelist(self.root, split)
        if "EF" not in self.df.columns:
            raise ValueError(f"EF column not found in {self.root / 'FileList.csv'}")
        if limit is not None:
            self.df = self.df.iloc[: int(limit)]
        self.frames = int(frames)
        self.img_size = int(img_size)
        self.aug_cfg = aug_cfg
        self.per_frame_random = per_frame_random
        self.seed = int(seed)
        if len(self.df) == 0:
            raise RuntimeError(f"EchoNet EF split {split!r} is empty under {self.root}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        file_name = str(row["FileName"])
        path = find_echonet_video(self.root, file_name)
        if path is None:
            raise FileNotFoundError(f"Cannot find EchoNet video for {file_name!r} under {self.root}")
        video = read_video(path)
        video = sample_frames(video, self.frames)
        if self.aug_cfg is not None:
            video = augment_video(video, self.aug_cfg, self.seed + index * 997, self.per_frame_random)
        x = _video_to_tensor(video, self.frames, self.img_size)
        y = torch.tensor(float(row["EF"]), dtype=torch.float32)
        return {"video": x, "target": y, "id": file_name, "source_path": str(path)}


class EchoNetSegmentationDataset(Dataset):
    """EchoNet-Dynamic LV segmentation dataset from VolumeTracings.csv."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int,
        aug_cfg=None,
        seed: int = 0,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.aug_cfg = aug_cfg
        self.seed = int(seed)
        file_df = load_echonet_filelist(self.root, split)
        allowed = {_row_stem(v) for v in file_df["FileName"].tolist()}
        trace_path = self.root / "VolumeTracings.csv"
        if not trace_path.exists():
            raise FileNotFoundError(f"EchoNet VolumeTracings.csv not found: {trace_path}")
        traces = pd.read_csv(trace_path)
        required = {"FileName", "Frame", "X1", "Y1", "X2", "Y2"}
        if not required.issubset(traces.columns):
            raise ValueError(f"VolumeTracings.csv must contain columns {sorted(required)}")
        traces["_stem"] = traces["FileName"].map(_row_stem)
        traces = traces[traces["_stem"].isin(allowed)]
        samples: list[dict[str, Any]] = []
        for (stem, frame), group in traces.groupby(["_stem", "Frame"], sort=True):
            samples.append({"stem": str(stem), "frame": int(frame), "trace": group.reset_index(drop=True)})
        if limit is not None:
            samples = samples[: int(limit)]
        if not samples:
            raise RuntimeError(f"No EchoNet segmentation traces found for split {split!r} under {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        path = find_echonet_video(self.root, sample["stem"])
        if path is None:
            raise FileNotFoundError(f"Cannot find EchoNet video for {sample['stem']!r} under {self.root}")
        video = read_video(path)
        gray_video = to_grayscale(video)
        frame_idx = min(max(int(sample["frame"]), 0), gray_video.shape[0] - 1)
        image = gray_video[frame_idx]
        mask = _rasterize_echonet_trace(sample["trace"], image.shape[:2])
        if self.aug_cfg is not None:
            image, mask = augment_image_mask(image, mask, self.aug_cfg, self.seed + index * 997, allow_zoom=True)
        else:
            image = resize_with_pad(image, self.img_size)[0, :, :, 0]
            mask = resize_mask_with_pad(mask, self.img_size)
        image = normalize_float(image)
        return {
            "image": torch.from_numpy(image[None]).float(),
            "mask": torch.from_numpy(mask.astype(np.int64)).long(),
            "id": f"{sample['stem']}:{frame_idx}",
            "source_path": str(path),
        }


class CAMUSSegmentationDataset(Dataset):
    """CAMUS ED/ES segmentation dataset."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        img_size: int,
        aug_cfg=None,
        seed: int = 0,
        limit: int | None = None,
        val_fraction: float = 0.15,
    ):
        self.root = Path(root)
        self.img_size = int(img_size)
        self.aug_cfg = aug_cfg
        self.seed = int(seed)
        rows = camus_pairs(self.root, "training")
        if not rows:
            rows = camus_pairs(self.root, split)
        rows = _split_rows(rows, split, val_fraction=val_fraction, seed=seed)
        if limit is not None:
            rows = rows[: int(limit)]
        if not rows:
            raise RuntimeError(f"No CAMUS segmentation pairs found for split {split!r} under {self.root}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = np.squeeze(read_medical_image(Path(row["image"])))
        mask = np.squeeze(read_medical_image(Path(row["mask"]))).astype(np.uint8)
        if self.aug_cfg is not None:
            image, mask = augment_image_mask(image, mask, self.aug_cfg, self.seed + index * 997, allow_zoom=True)
        else:
            image = resize_with_pad(image, self.img_size)[0, :, :, 0]
            mask = resize_mask_with_pad(mask, self.img_size)
        image = normalize_float(image)
        return {
            "image": torch.from_numpy(image[None]).float(),
            "mask": torch.from_numpy(mask.astype(np.int64)).long(),
            "id": row["image"],
            "source_path": row["image"],
        }
