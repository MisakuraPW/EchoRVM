"""Inspect an official Hiera checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hiera_echo.models import _as_state_dict


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    args = p.parse_args()
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = _as_state_dict(ckpt)
    print(f"checkpoint={args.checkpoint}")
    print(f"top_keys={list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}")
    print(f"state_tensors={len(state)}")
    for key in ("pos_embed", "decoder_pos_embed", "patch_embed.proj.weight", "decoder_pred.weight"):
        value = state.get(key)
        print(f"{key}: {tuple(value.shape) if value is not None else 'MISSING'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
