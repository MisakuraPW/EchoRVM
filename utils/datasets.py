"""Real EchoNet/CAMUS datasets for RMAE pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from augment.ultrasound import resize_with_pad
from echo_aug_validation.io_utils import (
    camus_pairs,
    find_echonet_video,
    load_echonet_filelist,
    normalize_float,
    read_medical_image,
    read_video,
    sample_frames,
)


def _as_video_tensor(video: np.ndarray, frames: int, img_size: int) -> torch.Tensor:
    video = sample_frames(video, frames)
    video = resize_with_pad(video, img_size)
    if video.ndim == 4:
        if video.shape[-1] == 1:
            gray = video[..., 0]
        elif video.shape[-1] in (3, 4):
            gray = video[..., :3].mean(axis=-1)
        else:
            raise ValueError(f"Unsupported channel count for video shape {video.shape}")
    elif video.ndim == 3:
        gray = video
    else:
        raise ValueError(f"Unsupported video shape {video.shape}")
    gray = normalize_float(gray)
    return torch.from_numpy(gray[:, None]).float()


def _split_subset(dataset: Dataset, split: str, val_fraction: float = 0.15, seed: int = 42) -> Dataset:
    split_l = split.lower()
    if split_l in {"all", "full", "training"}:
        return dataset
    n = len(dataset)
    if n == 0:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_n = max(1, int(round(n * val_fraction))) if n > 1 else 1
    val_idx = np.sort(indices[:val_n]).tolist()
    train_idx = np.sort(indices[val_n:]).tolist()
    if split_l in {"val", "valid", "validation"}:
        return Subset(dataset, val_idx)
    if split_l in {"train"}:
        return Subset(dataset, train_idx if train_idx else val_idx)
    raise ValueError(f"Unsupported split {split!r}")


class EchoNetRMAEDataset(Dataset):
    def __init__(self, root: str | Path, split: str, frames: int, img_size: int, limit: int | None = None):
        self.root = Path(root)
        self.df = load_echonet_filelist(self.root, split)
        if limit is not None:
            self.df = self.df.iloc[: int(limit)]
        self.frames = int(frames)
        self.img_size = int(img_size)
        if len(self.df) == 0:
            raise RuntimeError(f"EchoNet split {split!r} is empty under {self.root}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        file_name = str(row["FileName"])
        path = find_echonet_video(self.root, file_name)
        if path is None:
            raise FileNotFoundError(f"Cannot find EchoNet video for {file_name!r} under {self.root}")
        video = _as_video_tensor(read_video(path), self.frames, self.img_size)
        return {"video": video, "id": file_name, "dataset": "echonet", "source_path": str(path)}


class CAMUSRMAEDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        frames: int,
        img_size: int,
        limit: int | None = None,
        val_fraction: float = 0.15,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.frames = int(frames)
        self.img_size = int(img_size)
        base_split = "training"
        rows = camus_pairs(self.root, base_split)
        if not rows:
            rows = camus_pairs(self.root, split)
        if limit is not None:
            rows = rows[: int(limit)]
        self.rows = rows
        if len(self.rows) == 0:
            raise RuntimeError(f"No CAMUS ED/ES images found under {self.root}")
        subset = _split_subset(self, split, val_fraction=val_fraction, seed=seed)
        if isinstance(subset, Subset):
            self.rows = [self.rows[i] for i in subset.indices]
        if len(self.rows) == 0:
            raise RuntimeError(f"CAMUS split {split!r} is empty under {self.root}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = np.squeeze(read_medical_image(Path(row["image"])))
        video = _as_video_tensor(image[None], self.frames, self.img_size)
        return {"video": video, "id": row["image"], "dataset": "camus", "source_path": row["image"]}


def build_rmae_dataset(data_cfg: dict[str, Any], model_cfg: dict[str, Any], split: str, seed: int = 42) -> Dataset:
    frames = int(model_cfg.get("frames", 16))
    img_size = int(model_cfg.get("img_size", 112))
    limit = data_cfg.get("limit", None)
    if limit is not None:
        limit = int(limit)
    if "datasets" in data_cfg:
        datasets = []
        for item in data_cfg.get("datasets", []):
            if not item.get("enabled", True):
                continue
            sub_cfg = dict(data_cfg)
            sub_cfg.update(item)
            sub_cfg.pop("datasets", None)
            datasets.append(build_rmae_dataset(sub_cfg, model_cfg, split, seed=seed))
        if not datasets:
            raise RuntimeError("No enabled datasets configured")
        return datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)

    name = str(data_cfg.get("dataset_name") or data_cfg.get("name") or "").lower()
    root = data_cfg.get("data_root")
    if not root:
        raise ValueError(f"data_root is required for real dataset {name!r}")
    split_name = data_cfg.get(f"{split}_split", split)
    if "echonet" in name:
        return EchoNetRMAEDataset(root, str(split_name), frames, img_size, limit=limit)
    if "camus" in name:
        return CAMUSRMAEDataset(
            root,
            str(split_name),
            frames,
            img_size,
            limit=limit,
            val_fraction=float(data_cfg.get("val_fraction", 0.15)),
            seed=seed,
        )
    raise ValueError(f"Unsupported dataset_name/name {name!r}")
