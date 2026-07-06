"""Torch datasets for augmentation proxy experiments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .augment_recipes import augment_image_mask, augment_video
from .io_utils import (
    camus_pairs,
    find_echonet_video,
    load_echonet_filelist,
    normalize_float,
    read_jsonl,
    read_medical_image,
    read_video,
    sample_frames,
    to_grayscale,
)


class EchoNetEFDataset(Dataset):
    def __init__(self, root: str | Path, split: str, num_frames: int, aug_cfg=None, per_frame_random: bool = False, seed: int = 0, limit: int | None = None):
        self.root = Path(root)
        self.df = load_echonet_filelist(self.root, split)
        if limit:
            self.df = self.df.iloc[:limit]
        self.num_frames = num_frames
        self.aug_cfg = aug_cfg
        self.per_frame_random = per_frame_random
        self.seed = seed

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        file_name = row["FileName"]
        path = find_echonet_video(self.root, str(file_name))
        if path is None:
            raise FileNotFoundError(f"Cannot find EchoNet video for {file_name}")
        video = read_video(path)
        video = sample_frames(video, self.num_frames)
        if self.aug_cfg is not None:
            video = augment_video(video, self.aug_cfg, self.seed + idx * 997, self.per_frame_random)
        gray = normalize_float(to_grayscale(video))
        x = torch.from_numpy(gray[:, None]).float()
        y = torch.tensor(float(row["EF"]), dtype=torch.float32)
        return {"video": x, "target": y, "id": str(file_name)}


class EchoNetMAEDataset(EchoNetEFDataset):
    def __getitem__(self, idx: int):
        item = super().__getitem__(idx)
        return {"video": item["video"], "id": item["id"]}


class CAMUSSegDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "training", aug_cfg=None, seed: int = 0, limit: int | None = None):
        rows = camus_pairs(Path(root), split)
        self.rows = rows[:limit] if limit else rows
        self.aug_cfg = aug_cfg
        self.seed = seed

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        image = read_medical_image(Path(row["image"]))
        mask = read_medical_image(Path(row["mask"]))
        image = np.squeeze(image)
        mask = np.squeeze(mask).astype(np.uint8)
        if self.aug_cfg is not None:
            image, mask = augment_image_mask(image, mask, self.aug_cfg, self.seed + idx * 997)
        else:
            from augment.ultrasound import resize_with_pad
            from .augment_recipes import resize_mask_with_pad
            image = resize_with_pad(image, 112)[0, :, :, 0]
            mask = resize_mask_with_pad(mask, 112)
        image = normalize_float(image)
        return {
            "image": torch.from_numpy(image[None]).float(),
            "mask": torch.from_numpy(mask.astype(np.int64)).long(),
            "id": row["image"],
        }


class MixedMAEDataset(Dataset):
    def __init__(self, echonet_root: str | Path | None, camus_root: str | Path | None, split: str, num_frames: int, aug_cfg=None, per_frame_random: bool = False, seed: int = 0, limit: int | None = None):
        self.items: list[tuple[str, int]] = []
        self.echonet = None
        self.camus = None
        self.num_frames = num_frames
        if echonet_root:
            self.echonet = EchoNetMAEDataset(echonet_root, split, num_frames, aug_cfg, per_frame_random, seed, limit)
            self.items.extend(("echonet", i) for i in range(len(self.echonet)))
        if camus_root:
            self.camus = CAMUSSegDataset(camus_root, "training", aug_cfg, seed, limit)
            self.items.extend(("camus", i) for i in range(len(self.camus)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        name, inner = self.items[idx]
        if name == "echonet":
            return self.echonet[inner]
        item = self.camus[inner]
        video = item["image"].unsqueeze(0).repeat(self.num_frames, 1, 1, 1)
        return {"video": video, "id": item["id"]}
