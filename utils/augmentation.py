"""Training-time ultrasound augmentation adapters.

This module keeps the trainer's tensor contract stable while reusing the
project's NumPy/OpenCV ultrasound augmentations.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from augment.ultrasound import EchoAugmentConfig
from echo_aug_validation.augment_recipes import RECIPE_DEFAULTS, augment_video


_AUGMENT_CONFIG_KEYS = {field.name for field in fields(EchoAugmentConfig)}
_PRESET_ALIASES = {
    "A4": "A4_tgc_zoom_speckle",
    "a4": "A4_tgc_zoom_speckle",
    "echo_clip_consistent": "A4_tgc_zoom_speckle",
}


def build_echo_augment_config(augment_cfg: dict[str, Any] | None, img_size: int = 112) -> tuple[EchoAugmentConfig, bool, str]:
    augment_cfg = dict(augment_cfg or {})
    preset = str(augment_cfg.get("preset", "A4_tgc_zoom_speckle"))
    preset = _PRESET_ALIASES.get(preset, preset)
    params: dict[str, Any] = {}
    if preset in RECIPE_DEFAULTS:
        params.update(RECIPE_DEFAULTS[preset])
    params.update({key: value for key, value in augment_cfg.items() if key in _AUGMENT_CONFIG_KEYS})
    params.setdefault("img_size", img_size)
    params["preserve_dtype"] = False
    per_frame_random = bool(augment_cfg.get("per_frame_random", params.pop("per_frame_random", False)))
    cfg = EchoAugmentConfig(**{key: value for key, value in params.items() if key in _AUGMENT_CONFIG_KEYS})
    return cfg, per_frame_random, preset


def _video_tensor_to_numpy(video: torch.Tensor) -> np.ndarray:
    arr = video.detach().cpu().numpy()
    if arr.ndim == 4:
        if arr.shape[1] in (1, 3, 4):
            arr = np.transpose(arr, (0, 2, 3, 1))
    elif arr.ndim == 3:
        pass
    else:
        raise ValueError(f"Expected video tensor [T,C,H,W] or [T,H,W], got shape {tuple(video.shape)}")
    return arr


def _numpy_to_video_tensor(video: np.ndarray, channels: int = 1) -> torch.Tensor:
    arr = np.asarray(video)
    if arr.ndim == 3:
        arr = arr[:, :, :, None]
    if arr.ndim != 4:
        raise ValueError(f"Expected augmented video [T,H,W,C], got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.shape[-1] != channels:
        if channels == 1:
            arr = arr[..., :3].mean(axis=-1, keepdims=True)
        elif arr.shape[-1] == 1:
            arr = np.repeat(arr, channels, axis=-1)
        else:
            arr = arr[..., :channels]
    arr = np.clip(arr, 0.0, 1.0)
    arr = np.transpose(arr, (0, 3, 1, 2)).copy()
    return torch.from_numpy(arr)


class EchoVideoAugmenter:
    """Apply A4-style online augmentation to one clip tensor."""

    def __init__(self, augment_cfg: dict[str, Any] | None, img_size: int = 112, channels: int = 1):
        self.enabled = bool((augment_cfg or {}).get("enabled", False))
        self.cfg, self.per_frame_random, self.preset = build_echo_augment_config(augment_cfg, img_size=img_size)
        self.channels = int(channels)

    def __call__(self, video: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return video.float()
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        arr = _video_tensor_to_numpy(video)
        aug = augment_video(arr, cfg=self.cfg, seed=seed, per_frame_random=self.per_frame_random)
        return _numpy_to_video_tensor(aug, channels=self.channels)


class AugmentedVideoDataset(Dataset):
    """Wrap any dataset returning a ``video`` item and augment it online."""

    def __init__(self, dataset: Dataset, augmenter: EchoVideoAugmenter):
        self.dataset = dataset
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        sample["video"] = self.augmenter(sample["video"])
        return sample
