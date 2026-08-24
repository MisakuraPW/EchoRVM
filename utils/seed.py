"""Randomness control."""

from __future__ import annotations

import random
import warnings

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _as_cpu_byte_tensor(value) -> torch.Tensor | None:
    """Normalize RNG state loaded through any map_location to CPU uint8."""

    if value is None:
        return None
    try:
        if torch.is_tensor(value):
            return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        if isinstance(value, np.ndarray):
            return torch.from_numpy(np.asarray(value, dtype=np.uint8)).clone()
        return torch.as_tensor(value, dtype=torch.uint8, device="cpu").contiguous()
    except (TypeError, ValueError, RuntimeError):
        return None


def set_rng_state(state: dict | None) -> None:
    if not state:
        return

    try:
        if "python" in state:
            random.setstate(state["python"])
        if "numpy" in state:
            np.random.set_state(state["numpy"])
    except (TypeError, ValueError) as exc:
        warnings.warn(f"Could not restore Python/NumPy RNG state: {exc}", RuntimeWarning)

    torch_state = _as_cpu_byte_tensor(state.get("torch"))
    if torch_state is not None:
        try:
            torch.set_rng_state(torch_state)
        except RuntimeError as exc:
            warnings.warn(f"Could not restore PyTorch CPU RNG state: {exc}", RuntimeWarning)

    if not torch.cuda.is_available():
        return

    cuda_state = state.get("cuda")
    if cuda_state is None:
        return
    if torch.is_tensor(cuda_state) or isinstance(cuda_state, np.ndarray):
        cuda_state = [cuda_state]

    try:
        cuda_states = list(cuda_state)
    except TypeError:
        warnings.warn("Could not restore CUDA RNG state: invalid checkpoint value.", RuntimeWarning)
        return

    for device_index, raw_state in enumerate(cuda_states[: torch.cuda.device_count()]):
        normalized = _as_cpu_byte_tensor(raw_state)
        if normalized is None:
            warnings.warn(
                f"Could not restore CUDA RNG state for device {device_index}: invalid value.",
                RuntimeWarning,
            )
            continue
        try:
            torch.cuda.set_rng_state(normalized, device=device_index)
        except (TypeError, RuntimeError) as exc:
            warnings.warn(
                f"Could not restore CUDA RNG state for device {device_index}: {exc}",
                RuntimeWarning,
            )
