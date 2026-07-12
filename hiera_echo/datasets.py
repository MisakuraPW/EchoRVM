"""Frame datasets for Hiera-T echo MAE pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils.datasets import build_rmae_dataset


class SyntheticHieraFrameDataset(Dataset):
    def __init__(self, length: int = 32, img_size: int = 128):
        self.length = int(length)
        self.img_size = int(img_size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, Any]:
        gen = torch.Generator().manual_seed(int(index))
        image = torch.rand(1, self.img_size, self.img_size, generator=gen)
        return {"image": image, "valid_mask": torch.ones_like(image), "id": f"synthetic_{index:06d}"}


class RMAEVideoToFrameDataset(Dataset):
    """Reuse existing EchoNet/CAMUS readers, but expose single-frame samples."""

    def __init__(self, base: Dataset, img_size: int, frame_policy: str = "middle"):
        self.base = base
        self.img_size = int(img_size)
        self.frame_policy = str(frame_policy)

    def __len__(self) -> int:
        return len(self.base)

    def _select_index(self, video: torch.Tensor, index: int) -> int:
        t = int(video.shape[0])
        if self.frame_policy == "random":
            return int(torch.randint(0, t, (1,)).item())
        if self.frame_policy == "first":
            return 0
        return t // 2

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.base[index])
        video = sample["video"].float()
        frame_idx = self._select_index(video, index)
        image = video[frame_idx]
        if image.shape[-1] != self.img_size or image.shape[-2] != self.img_size:
            image = F.interpolate(image[None], size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)[0]
        valid_mask = torch.ones(1, self.img_size, self.img_size, dtype=image.dtype)
        return {
            "image": image,
            "valid_mask": valid_mask,
            "id": sample.get("id", str(index)),
            "source_path": sample.get("source_path", ""),
            "dataset": sample.get("dataset", ""),
        }


def build_hiera_frame_dataset(data_cfg: dict[str, Any], model_cfg: dict[str, Any], split: str, seed: int = 42) -> Dataset:
    loader = str(data_cfg.get("loader", data_cfg.get("dataset_name", ""))).lower()
    img_size = int(model_cfg.get("img_size", 192))
    if loader in {"synthetic", "debug", "smoke"} or bool(data_cfg.get("synthetic", False)):
        return SyntheticHieraFrameDataset(
            length=int(data_cfg.get(f"synthetic_{split}_samples", data_cfg.get("synthetic_samples", 32))),
            img_size=img_size,
        )

    # Reuse existing video readers with T=1 unless a caller explicitly asks for
    # more candidate frames before selecting one.
    rmae_model_cfg = dict(model_cfg)
    rmae_model_cfg["frames"] = int(data_cfg.get("source_frames", 1))
    rmae_model_cfg["in_chans"] = 1
    base = build_rmae_dataset(data_cfg, rmae_model_cfg, split, seed=seed)
    return RMAEVideoToFrameDataset(base, img_size=img_size, frame_policy=str(data_cfg.get("frame_policy", "middle")))
