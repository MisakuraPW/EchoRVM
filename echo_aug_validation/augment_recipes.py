"""Augmentation recipes A0-A7 for proxy experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from augment.ultrasound import EchoAugmentConfig, EchoClipAugmentor, ensure_video_array, resize_with_pad


RECIPE_DEFAULTS: dict[str, dict[str, Any]] = {
    "A0_no_aug": dict(tgc_prob=0.0, gamma_contrast_prob=0.0, brightness_prob=0.0, zoom_prob=0.0, blur_prob=0.0, shadow_prob=0.0, speckle_prob=0.0, per_frame_random=False),
    "A1_basic_photometric": dict(tgc_prob=0.0, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.0, blur_prob=0.2, shadow_prob=0.0, speckle_prob=0.0, per_frame_random=False),
    "A2_basic_tgc": dict(tgc_prob=0.4, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.0, blur_prob=0.2, shadow_prob=0.0, speckle_prob=0.0, per_frame_random=False),
    "A3_tgc_zoom": dict(tgc_prob=0.4, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.3, blur_prob=0.2, shadow_prob=0.0, speckle_prob=0.0, per_frame_random=False),
    "A4_tgc_zoom_speckle": dict(tgc_prob=0.4, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.3, blur_prob=0.2, shadow_prob=0.0, speckle_prob=0.4, per_frame_random=False),
    "A5_shadow": dict(tgc_prob=0.4, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.3, blur_prob=0.2, shadow_prob=0.1, speckle_prob=0.4, per_frame_random=False),
    "A6_per_frame_random": dict(tgc_prob=0.4, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.3, blur_prob=0.2, shadow_prob=0.0, speckle_prob=0.4, per_frame_random=True),
    "A7_clip_consistent": dict(tgc_prob=0.4, gamma_contrast_prob=0.5, brightness_prob=0.3, zoom_prob=0.3, blur_prob=0.2, shadow_prob=0.0, speckle_prob=0.4, per_frame_random=False),
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_recipe(path_or_id: str | Path | None, img_size: int = 112) -> tuple[str, EchoAugmentConfig, bool]:
    if path_or_id is None:
        path_or_id = "A7_clip_consistent"
    path = Path(path_or_id)
    if path.exists():
        data = load_yaml(path)
        recipe_id = data.get("id", path.stem)
        params = data.get("params", {})
        per_frame = bool(data.get("per_frame_random", params.pop("per_frame_random", False)))
    else:
        recipe_id = str(path_or_id)
        params = dict(RECIPE_DEFAULTS.get(recipe_id, RECIPE_DEFAULTS["A7_clip_consistent"]))
        per_frame = bool(params.pop("per_frame_random", False))
    params.setdefault("img_size", img_size)
    cfg = EchoAugmentConfig(**{k: v for k, v in params.items() if k in asdict(EchoAugmentConfig()).keys()})
    return recipe_id, cfg, per_frame


def augment_video(video: np.ndarray, cfg: EchoAugmentConfig, seed: int, per_frame_random: bool = False) -> np.ndarray:
    video = ensure_video_array(video)
    if per_frame_random:
        frames = []
        for i in range(video.shape[0]):
            aug = EchoClipAugmentor(cfg, seed=seed + i)
            frames.append(aug(video[i], return_meta=False)[0])
        return np.stack(frames, axis=0)
    return EchoClipAugmentor(cfg, seed=seed)(video)


def _zoom_pair(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    img = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    msk = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    if scale >= 1.0:
        y0 = (nh - h) // 2
        x0 = (nw - w) // 2
        return img[y0 : y0 + h, x0 : x0 + w], msk[y0 : y0 + h, x0 : x0 + w]
    out_i = np.zeros_like(image)
    out_m = np.zeros_like(mask, dtype=np.uint8)
    y0 = (h - nh) // 2
    x0 = (w - nw) // 2
    out_i[y0 : y0 + nh, x0 : x0 + nw] = img
    out_m[y0 : y0 + nh, x0 : x0 + nw] = msk
    return out_i, out_m


def resize_mask_with_pad(mask: np.ndarray, size: int = 112) -> np.ndarray:
    mask = np.asarray(mask)
    h, w = mask.shape[:2]
    if h == size and w == size:
        return mask.astype(np.uint8)
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(mask.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def augment_image_mask(image: np.ndarray, mask: np.ndarray, cfg: EchoAugmentConfig, seed: int, allow_zoom: bool = True) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    image = resize_with_pad(image, cfg.img_size)[0]
    mask = resize_mask_with_pad(mask, cfg.img_size)
    if allow_zoom and cfg.zoom_prob > 0 and rng.random() < cfg.zoom_prob:
        scale = float(rng.uniform(cfg.zoom_min, cfg.zoom_max))
        image, mask = _zoom_pair(image, mask, scale)
        no_zoom_cfg = EchoAugmentConfig(**(asdict(cfg) | {"zoom_prob": 0.0}))
        image = EchoClipAugmentor(no_zoom_cfg, seed=seed + 17)(image)[0]
    else:
        image = EchoClipAugmentor(cfg, seed=seed)(image)[0]
    return image, mask.astype(np.int64)
