"""Prepare lightweight JSONL indexes for augmentation validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .io_utils import camus_pairs, find_echonet_video, load_echonet_filelist, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description="Build EchoNet/CAMUS indexes for augmentation validation.")
    parser.add_argument("--echonet-root", default="/root/autodl-fs/datasets/EchoNet-Dynamic")
    parser.add_argument("--camus-root", default="/root/autodl-fs/datasets/CAMUS")
    parser.add_argument("--output-dir", default="/root/autodl-tmp/aug_validation_indexes")
    parser.add_argument("--split", default="TRAIN")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    echonet_root = Path(args.echonet_root)
    if echonet_root.exists():
        df = load_echonet_filelist(echonet_root, args.split)
        rows = []
        for _, row in df.iterrows():
            video = find_echonet_video(echonet_root, str(row["FileName"]))
            if video is not None:
                rows.append({"file_name": str(row["FileName"]), "video": str(video), "ef": float(row["EF"]), "dataset": "echonet"})
            if args.limit and len(rows) >= args.limit:
                break
        write_jsonl(out / "echonet_ef.jsonl", rows)
        print(f"EchoNet index: {len(rows)} samples -> {out / 'echonet_ef.jsonl'}")

    camus_root = Path(args.camus_root)
    if camus_root.exists():
        rows = camus_pairs(camus_root, "training")
        if args.limit:
            rows = rows[: args.limit]
        write_jsonl(out / "camus_seg.jsonl", rows)
        print(f"CAMUS index: {len(rows)} samples -> {out / 'camus_seg.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
