"""Check CAMUS dataset discovery for old MHD/RAW and new NIfTI layouts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from echo_aug_validation.io_utils import camus_pairs


def summarise(root: Path, split: str) -> None:
    rows = camus_pairs(root, split)
    patients = {row.get("patient", "") for row in rows}
    views = Counter(row.get("view", "") or "unknown" for row in rows)
    frames = Counter(row.get("frame", "") or "unknown" for row in rows)
    sources = Counter(row.get("split_source", "") or "unknown" for row in rows)
    print(f"{split}: samples={len(rows)} patients={len(patients)} views={dict(views)} frames={dict(frames)} split_source={dict(sources)}")
    if rows:
        print(f"  first_image={rows[0]['image']}")
        print(f"  first_mask ={rows[0]['mask']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect CAMUS ED/ES image-mask discovery.")
    parser.add_argument("--root", default="/root/autodl-fs/datasets/CAMUS")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()
    root = Path(args.root)
    print(f"root={root}")
    print(f"database_nifti={(root / 'database_nifti').exists()} database_split={(root / 'database_split').exists()}")
    for split in args.splits:
        summarise(root, split)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
