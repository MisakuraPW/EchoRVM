"""Ultrasound-specific, temporally consistent video augmentations.

The defaults follow the project overview and the "Speckle and Shadows" paper:
small zoom, time-gain compensation, gamma/contrast changes, weak speckle noise,
and low-probability acoustic shadows. Geometry-breaking transforms such as
flips, large rotations, CutMix, and elastic warps are intentionally absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class EchoAugmentConfig:
    img_size: int = 112
    tgc_prob: float = 0.4
    gamma_contrast_prob: float = 0.4
    brightness_prob: float = 0.3
    zoom_prob: float = 0.3
    blur_prob: float = 0.15
    shadow_prob: float = 0.1
    speckle_prob: float = 0.4
    tgc_min: float = 0.75
    tgc_max: float = 1.35
    gamma_min: float = 0.75
    gamma_max: float = 1.35
    contrast_min: float = 0.85
    contrast_max: float = 1.20
    brightness_delta: float = 0.08
    zoom_min: float = 0.92
    zoom_max: float = 1.12
    speckle_sigma_min: float = 0.03
    speckle_sigma_max: float = 0.10
    shadow_width_min: float = 0.08
    shadow_width_max: float = 0.22
    shadow_strength_min: float = 0.35
    shadow_strength_max: float = 0.75
    shadow_softness: int = 17
    valid_threshold: float = 0.02
    preserve_dtype: bool = True


def _to_float01(arr: Array) -> tuple[Array, np.dtype, float]:
    dtype = arr.dtype
    if np.issubdtype(dtype, np.integer):
        max_value = float(np.iinfo(dtype).max)
        return arr.astype(np.float32) / max_value, dtype, max_value
    arr_f = arr.astype(np.float32)
    max_seen = float(np.nanmax(arr_f)) if arr_f.size else 1.0
    scale = 255.0 if max_seen > 2.0 else 1.0
    return np.clip(arr_f / scale, 0.0, 1.0), dtype, scale


def _from_float01(arr: Array, dtype: np.dtype, scale: float, preserve_dtype: bool) -> Array:
    arr = np.clip(arr, 0.0, 1.0)
    if not preserve_dtype:
        return arr.astype(np.float32)
    if np.issubdtype(dtype, np.integer):
        return np.rint(arr * scale).astype(dtype)
    return (arr * scale).astype(dtype)


def ensure_video_array(video: Array) -> Array:
    """Return video as [T, H, W, C], accepting common grayscale/RGB layouts."""
    video = np.asarray(video)
    if video.ndim == 2:
        video = video[None, :, :, None]
    elif video.ndim == 3:
        if video.shape[-1] in (1, 3, 4):
            video = video[None, :, :, :]
        else:
            video = video[:, :, :, None]
    elif video.ndim == 4:
        if video.shape[1] in (1, 3, 4) and video.shape[-1] not in (1, 3, 4):
            video = np.transpose(video, (0, 2, 3, 1))
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D video array, got shape {video.shape}")
    if video.shape[-1] == 4:
        video = video[..., :3]
    return video


def restore_original_rank(video: Array, original_shape: tuple[int, ...]) -> Array:
    """Collapse singleton dimensions when the input was an image."""
    if len(original_shape) == 2:
        return video[0, :, :, 0]
    if len(original_shape) == 3 and original_shape[-1] not in (1, 3, 4):
        return video[:, :, :, 0]
    if len(original_shape) == 3 and original_shape[-1] in (1, 3, 4):
        return video[0]
    return video


def resize_with_pad(video: Array, size: int = 112) -> Array:
    """Resize each frame into a square canvas without changing aspect ratio."""
    video = ensure_video_array(video)
    t, h, w, c = video.shape
    if h == size and w == size:
        return video.copy()
    scale = min(size / h, size / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    out = np.zeros((t, size, size, c), dtype=video.dtype)
    y0 = (size - nh) // 2
    x0 = (size - nw) // 2
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    for idx in range(t):
        frame = cv2.resize(video[idx], (nw, nh), interpolation=interpolation)
        if c == 1 and frame.ndim == 2:
            frame = frame[:, :, None]
        out[idx, y0 : y0 + nh, x0 : x0 + nw] = frame
    return out


def estimate_valid_region(video: Array, threshold: float = 0.02) -> Array:
    """Estimate non-background ultrasound region from temporal mean intensity."""
    video_f, _, _ = _to_float01(ensure_video_array(video))
    gray = video_f.mean(axis=(0, 3))
    mask = gray > max(threshold, float(np.percentile(gray, 55)) * 0.15)
    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)
    return mask.astype(bool)


def _sample_flag(rng: np.random.Generator, prob: float) -> bool:
    return bool(rng.random() < prob)


def _apply_tgc(video: Array, rng: np.random.Generator, cfg: EchoAugmentConfig) -> Array:
    h = video.shape[1]
    anchors = rng.uniform(cfg.tgc_min, cfg.tgc_max, size=10).astype(np.float32)
    gains = np.interp(np.linspace(0, 9, h), np.arange(10), anchors).astype(np.float32)
    gains = cv2.GaussianBlur(gains[:, None], (1, 9), 0).reshape(1, h, 1, 1)
    return video * gains


def _apply_gamma_contrast(video: Array, rng: np.random.Generator, cfg: EchoAugmentConfig) -> Array:
    gamma = float(rng.uniform(cfg.gamma_min, cfg.gamma_max))
    contrast = float(rng.uniform(cfg.contrast_min, cfg.contrast_max))
    mean = video.mean(axis=(1, 2, 3), keepdims=True)
    video = np.power(np.clip(video, 0.0, 1.0), gamma)
    return (video - mean) * contrast + mean


def _apply_brightness(video: Array, rng: np.random.Generator, cfg: EchoAugmentConfig) -> Array:
    delta = float(rng.uniform(-cfg.brightness_delta, cfg.brightness_delta))
    return video + delta


def _apply_zoom(video: Array, rng: np.random.Generator, cfg: EchoAugmentConfig) -> Array:
    scale = float(rng.uniform(cfg.zoom_min, cfg.zoom_max))
    t, h, w, c = video.shape
    resized_h, resized_w = max(1, round(h * scale)), max(1, round(w * scale))
    out = np.empty_like(video)
    for idx in range(t):
        frame = cv2.resize(video[idx], (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        if c == 1 and frame.ndim == 2:
            frame = frame[:, :, None]
        if scale >= 1.0:
            y0 = (resized_h - h) // 2
            x0 = (resized_w - w) // 2
            out[idx] = frame[y0 : y0 + h, x0 : x0 + w]
        else:
            canvas = np.zeros_like(video[idx])
            y0 = (h - resized_h) // 2
            x0 = (w - resized_w) // 2
            canvas[y0 : y0 + resized_h, x0 : x0 + resized_w] = frame
            out[idx] = canvas
    return out


def _apply_blur(video: Array) -> Array:
    out = np.empty_like(video)
    for idx in range(video.shape[0]):
        frame = cv2.GaussianBlur(video[idx], (3, 3), 0)
        if video.shape[-1] == 1 and frame.ndim == 2:
            frame = frame[:, :, None]
        out[idx] = frame
    return out


def _apply_speckle(video: Array, rng: np.random.Generator, cfg: EchoAugmentConfig) -> Array:
    sigma = float(rng.uniform(cfg.speckle_sigma_min, cfg.speckle_sigma_max))
    noise = rng.normal(loc=0.0, scale=sigma, size=video.shape).astype(np.float32)
    return video + video * noise


def _shadow_mask(shape: tuple[int, int], rng: np.random.Generator, cfg: EchoAugmentConfig) -> Array:
    h, w = shape
    center = float(rng.uniform(0.25 * w, 0.75 * w))
    width_top = float(rng.uniform(cfg.shadow_width_min, cfg.shadow_width_max) * w)
    width_bottom = width_top * float(rng.uniform(1.4, 2.6))
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.arange(w, dtype=np.float32)[None, :]
    center_drift = center + (yy - 0.5) * float(rng.uniform(-0.15, 0.15) * w)
    half_width = (width_top * (1.0 - yy) + width_bottom * yy) / 2.0
    mask = (np.abs(xx - center_drift) <= half_width).astype(np.float32)
    if cfg.shadow_softness > 1:
        k = cfg.shadow_softness if cfg.shadow_softness % 2 == 1 else cfg.shadow_softness + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)
    depth_ramp = np.clip((yy - 0.10) / 0.90, 0.0, 1.0)
    return mask * depth_ramp


def _apply_shadow(video: Array, rng: np.random.Generator, cfg: EchoAugmentConfig, valid_mask: Array | None) -> Array:
    mask = _shadow_mask(video.shape[1:3], rng, cfg)
    if valid_mask is not None:
        mask = mask * valid_mask.astype(np.float32)
    strength = float(rng.uniform(cfg.shadow_strength_min, cfg.shadow_strength_max))
    mask = mask[None, :, :, None]
    shadow_floor = rng.gamma(shape=0.8, scale=0.08, size=video.shape).astype(np.float32)
    darkened = video * (1.0 - strength * mask)
    return np.where(mask > 0, np.maximum(darkened, shadow_floor * mask), video)


def median_blur_target(video: Array, kernel_size: int = 3) -> Array:
    video = ensure_video_array(video)
    out = np.empty_like(video)
    for idx in range(video.shape[0]):
        frame = cv2.medianBlur(video[idx], kernel_size)
        if video.shape[-1] == 1 and frame.ndim == 2:
            frame = frame[:, :, None]
        out[idx] = frame
    return out


def nakagami_shape_map(image: Array, window: int = 20, eps: float = 1e-6) -> Array:
    """Approximate a Nakagami shape parameter map from a B-mode image."""
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[..., :3].mean(axis=-1)
    image_f, _, _ = _to_float01(image)
    power = image_f**2
    mean_power = cv2.blur(power, (window, window))
    mean_power_sq = cv2.blur(power**2, (window, window))
    var_power = np.maximum(mean_power_sq - mean_power**2, eps)
    shape = (mean_power**2) / var_power
    high = np.percentile(shape, 99)
    return np.clip(shape / max(float(high), eps), 0.0, 1.0).astype(np.float32)


class EchoClipAugmentor:
    """Apply ultrasound augmentations with clip-level temporal consistency."""

    def __init__(self, cfg: EchoAugmentConfig | None = None, seed: int | None = None):
        self.cfg = cfg or EchoAugmentConfig()
        self.rng = np.random.default_rng(seed)

    def __call__(self, video: Array, return_meta: bool = False) -> Array | tuple[Array, dict[str, Any]]:
        original_shape = tuple(np.asarray(video).shape)
        arr = ensure_video_array(video)
        arr = resize_with_pad(arr, self.cfg.img_size)
        arr_f, dtype, scale = _to_float01(arr)
        valid = estimate_valid_region(arr_f, self.cfg.valid_threshold)

        applied: list[str] = ["resize_with_pad"]
        if _sample_flag(self.rng, self.cfg.zoom_prob):
            arr_f = _apply_zoom(arr_f, self.rng, self.cfg)
            applied.append("zoom")
        if _sample_flag(self.rng, self.cfg.tgc_prob):
            arr_f = _apply_tgc(arr_f, self.rng, self.cfg)
            applied.append("tgc")
        if _sample_flag(self.rng, self.cfg.gamma_contrast_prob):
            arr_f = _apply_gamma_contrast(arr_f, self.rng, self.cfg)
            applied.append("gamma_contrast")
        if _sample_flag(self.rng, self.cfg.brightness_prob):
            arr_f = _apply_brightness(arr_f, self.rng, self.cfg)
            applied.append("brightness")
        if _sample_flag(self.rng, self.cfg.blur_prob):
            arr_f = _apply_blur(arr_f)
            applied.append("blur")
        if _sample_flag(self.rng, self.cfg.shadow_prob):
            arr_f = _apply_shadow(arr_f, self.rng, self.cfg, valid)
            applied.append("shadow")
        if _sample_flag(self.rng, self.cfg.speckle_prob):
            arr_f = _apply_speckle(arr_f, self.rng, self.cfg)
            applied.append("speckle")

        out = _from_float01(arr_f, dtype, scale, self.cfg.preserve_dtype)
        meta = {
            "original_shape": original_shape,
            "output_shape": tuple(out.shape),
            "applied": applied,
            "img_size": self.cfg.img_size,
        }
        return (out, meta) if return_meta else out


def augment_clip(video: Array, cfg: EchoAugmentConfig | None = None, seed: int | None = None) -> Array:
    return EchoClipAugmentor(cfg=cfg, seed=seed)(video)
