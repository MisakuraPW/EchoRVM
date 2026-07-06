"""Train a tiny reconstruction proxy to screen augmentation recipes."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from .augment_recipes import load_recipe
from .datasets import MixedMAEDataset
from .losses import mae_recon_loss
from .metrics import MetricsCSV
from .models import SmallMAE
from .train_common import build_dataloader, configure_torch_runtime, device_from_config, load_config, make_run_dir, save_checkpoint, save_run_metadata, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="small-MAE augmentation screening.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--augmentation_config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_torch_runtime()
    seed = int(args.seed if args.seed is not None else cfg.get("experiment", {}).get("seed", 42))
    seed_everything(seed)
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    aug_id, aug_cfg, per_frame = load_recipe(args.augmentation_config or cfg.get("augmentation_config"), model_cfg.get("input_size", 112))

    output_root = args.output_dir or cfg.get("experiment", {}).get("output_root", "/root/autodl-tmp/outputs/aug_validation")
    run_dir = make_run_dir(output_root, cfg.get("experiment", {}).get("name", "small_mae"), aug_id, seed)
    save_run_metadata(run_dir, args.config, cfg, aug_id, seed)

    limit = train_cfg.get("debug_limit", 32) if args.debug else data_cfg.get("limit")
    dataset = MixedMAEDataset(
        data_cfg.get("echonet_root"),
        data_cfg.get("camus_root"),
        data_cfg.get("train_split", "TRAIN"),
        model_cfg.get("num_frames", 8),
        aug_cfg,
        per_frame,
        seed,
        limit,
    )
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 samples for train/val split.")
    val_size = max(1, int(len(dataset) * float(data_cfg.get("val_fraction", 0.2))))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = build_dataloader(train_ds, train_cfg, default_batch_size=8, shuffle=True)
    val_loader = build_dataloader(val_ds, train_cfg, default_batch_size=8, shuffle=False)

    device = device_from_config()
    model = SmallMAE(base=model_cfg.get("base_channels", 32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    metrics = MetricsCSV(run_dir / "logs" / "metrics.csv")
    best = float("inf")
    global_step = 0
    max_steps = args.max_steps or train_cfg.get("max_steps")

    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        pbar = tqdm(train_loader, desc=f"small-mae epoch {epoch}")
        for batch in pbar:
            video = batch["video"].to(device, non_blocking=True)  # B,T,C,H,W
            b, t, c, h, w = video.shape
            x = video.reshape(b * t, c, h, w)
            pred, mask = model(x, float(model_cfg.get("mask_ratio", 0.75)))
            loss = mae_recon_loss(pred, x, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
            seen += x.size(0)
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            if max_steps and global_step >= int(max_steps):
                break
        train_loss /= max(seen, 1)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                video = batch["video"].to(device, non_blocking=True)
                b, t, c, h, w = video.shape
                x = video.reshape(b * t, c, h, w)
                pred, mask = model(x, float(model_cfg.get("mask_ratio", 0.75)))
                loss = mae_recon_loss(pred, x, mask)
                val_loss += loss.item() * x.size(0)
                val_seen += x.size(0)
        val_loss /= max(val_seen, 1)
        row = {"epoch": epoch, "step": global_step, "augmentation_id": aug_id, "model_type": "small_mae", "seed": seed, "train_loss": train_loss, "val_loss": val_loss}
        metrics.append(row)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, opt, epoch, best)
        if val_loss < best:
            best = val_loss
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, opt, epoch, best)
        if max_steps and global_step >= int(max_steps):
            break
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
