"""Train lightweight U-Net on CAMUS to validate boundary safety."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import random_split
from tqdm import tqdm

from .augment_recipes import load_recipe
from .datasets import CAMUSSegDataset
from .losses import seg_loss
from .metrics import MetricsCSV, dice_per_class
from .models import LightUNet
from .train_common import build_dataloader, configure_torch_runtime, device_from_config, load_config, make_run_dir, save_checkpoint, save_run_metadata, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="Light U-Net CAMUS segmentation proxy.")
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
    aug_id, aug_cfg, _ = load_recipe(args.augmentation_config or cfg.get("augmentation_config"), model_cfg.get("input_size", 112))
    output_root = args.output_dir or cfg.get("experiment", {}).get("output_root", "/root/autodl-tmp/outputs/aug_validation")
    run_dir = make_run_dir(output_root, cfg.get("experiment", {}).get("name", "unet_seg_proxy"), aug_id, seed)
    save_run_metadata(run_dir, args.config, cfg, aug_id, seed)

    limit = train_cfg.get("debug_limit", 32) if args.debug else data_cfg.get("limit")
    dataset = CAMUSSegDataset(data_cfg["camus_root"], data_cfg.get("split", "training"), aug_cfg, seed, limit)
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 CAMUS segmentation samples.")
    val_size = max(1, int(len(dataset) * float(data_cfg.get("val_fraction", 0.2))))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = build_dataloader(train_ds, train_cfg, default_batch_size=8, shuffle=True)
    val_loader = build_dataloader(val_ds, train_cfg, default_batch_size=8, shuffle=False)

    device = device_from_config()
    num_classes = int(model_cfg.get("num_classes", 4))
    model = LightUNet(num_classes=num_classes, base=int(model_cfg.get("base_channels", 16))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    metrics = MetricsCSV(run_dir / "logs" / "metrics.csv")
    best = -1.0
    global_step = 0
    max_steps = args.max_steps or train_cfg.get("max_steps")

    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"unet epoch {epoch}"):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["mask"].to(device, non_blocking=True).clamp(0, num_classes - 1)
            logits = model(x)
            loss = seg_loss(logits, y, num_classes)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
            seen += x.size(0)
            global_step += 1
            if max_steps and global_step >= int(max_steps):
                break
        train_loss /= max(seen, 1)

        model.eval()
        val_loss = 0.0
        val_seen = 0
        dice_accum = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                y = batch["mask"].to(device, non_blocking=True).clamp(0, num_classes - 1)
                logits = model(x)
                loss = seg_loss(logits, y, num_classes)
                val_loss += loss.item() * x.size(0)
                val_seen += x.size(0)
                dice_accum.append(dice_per_class(logits.cpu(), y.cpu(), num_classes))
        val_loss /= max(val_seen, 1)
        dice_mean = float(sum(d["dice_mean"] for d in dice_accum) / max(len(dice_accum), 1))
        row = {"epoch": epoch, "step": global_step, "augmentation_id": aug_id, "model_type": "unet_seg_proxy", "seed": seed, "train_loss": train_loss, "val_loss": val_loss, "dice_mean": dice_mean}
        metrics.append(row)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, opt, epoch, best)
        if dice_mean > best:
            best = dice_mean
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, opt, epoch, best)
        if max_steps and global_step >= int(max_steps):
            break
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
