"""Convert official 224 Hiera-T MAE weights to an echo baseline resolution."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from hiera_echo.models import EchoHieraMAE, _as_state_dict, _load_torch, convert_hiera_state_for_model


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target_size", type=int, default=192)
    p.add_argument("--hiera_repo", default=None)
    args = p.parse_args()
    model = EchoHieraMAE(img_size=args.target_size, hiera_repo=args.hiera_repo).model
    state = _as_state_dict(_load_torch(args.source))
    converted, report = convert_hiera_state_for_model(state, model)
    missing_bad = [k for k in report["missing"] if k not in {"pos_embed", "decoder_pos_embed"}]
    if report["skipped_shape"]:
        raise RuntimeError(f"Unexpected shape mismatches: {report['skipped_shape']}")
    model.load_state_dict(converted, strict=False)
    payload = {
        "model_state": model.state_dict(),
        "source_checkpoint": str(Path(args.source).resolve()),
        "source_size": 224,
        "target_size": int(args.target_size),
        "conversion_report": report,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, ensure_ascii=False, indent=2))
    print(f"saved={out}")
    if missing_bad:
        print(f"warning_missing_nonpos={missing_bad[:10]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
