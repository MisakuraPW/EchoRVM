"""Explicit image-content protocol, independent of the model's input channels."""

from pathlib import Path

import cv2
import numpy as np

from echo_aug_validation.io_utils import read_video


INPUT_PROTOCOLS = ('rgb', 'gray_repeat3')


def read_echo_input(path: Path, input_protocol: str) -> np.ndarray:
    if input_protocol not in INPUT_PROTOCOLS:
        raise ValueError(f'Unknown EchoNet input protocol: {input_protocol}')
    path = Path(path)
    cached = path.suffix.lower() in ('.npy', '.npz')
    raw = (np.load(path, mmap_mode='r', allow_pickle=False)
           if path.suffix.lower() == '.npy' else read_video(path))
    if raw.ndim == 4 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim not in (3, 4) or (raw.ndim == 4 and raw.shape[-1] != 3):
        raise ValueError(f'{path}: expected [T,H,W] or [T,H,W,3], got {raw.shape}')
    if len(raw) == 0:
        raise ValueError(f'{path}: empty video')
    if input_protocol == 'gray_repeat3':
        if raw.ndim == 4:
            # Match the existing cache tool's BGR2GRAY, not an arithmetic RGB mean.
            conversion = cv2.COLOR_RGB2GRAY if cached else cv2.COLOR_BGR2GRAY
            return np.stack([cv2.cvtColor(frame, conversion) for frame in raw])
        return raw  # Leave grayscale NPY memory-mapped; sample before expanding channels.
    if raw.ndim != 4:
        raise ValueError(f'{path}: RGB protocol does not accept grayscale. Use gray_repeat3 explicitly.')
    return raw if cached else raw[..., ::-1].copy()
