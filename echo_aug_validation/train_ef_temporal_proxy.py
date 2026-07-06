"""Train CNN-GRU EF proxy on EchoNet-Dynamic."""

from __future__ import annotations

import argparse

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .augment_recipes import load_recipe
from .datasets import EchoNetEFDataset
from .metrics import MetricsCSV, ef_metrics
from .models import CNNGRUEF
from .train_common import build_dataloader, configure_torch_runtime, device_from_config, load_config, make_run_dir, save_checkpoint, save_run_metadata, seed_everything


def main() -> int:
    parser = argparse.ArgumentParser(description="CNN-GRU EF temporal augmentation proxy.")
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
    run_dir = make_run_dir(output_root, cfg.get("experiment", {}).get("name", "ef_temporal_proxy"), aug_id, seed)
    save_run_metadata(run_dir, args.config, cfg, aug_id, seed)

    limit = train_cfg.get("debug_limit", 32) if args.debug else data_cfg.get("limit")
    train_ds = EchoNetEFDataset(data_cfg["echonet_root"], data_cfg.get("train_split", "TRAIN"), model_cfg.get("num_frames", 32), aug_cfg, per_frame, seed, limit)
    val_ds = EchoNetEFDataset(data_cfg["echonet_root"], data_cfg.get("val_split", "VAL"), model_cfg.get("num_frames", 32), None, False, seed, limit)
    train_loader = build_dataloader(train_ds, train_cfg, default_batch_size=4, shuffle=True)
    val_loader = build_dataloader(val_ds, train_cfg, default_batch_size=4, shuffle=False)

    device = device_from_config()
    model = CNNGRUEF(feat_dim=int(model_cfg.get("feat_dim", 128)), hidden_dim=int(model_cfg.get("hidden_dim", 128))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(train_cfg.get("lr", 1e-3)), weight_decay=float(train_cfg.get("weight_decay", 1e-4)))
    metrics = MetricsCSV(run_dir / "logs" / "metrics.csv")
    best = float("inf")
    global_step = 0
    max_steps = args.max_steps or train_cfg.get("max_steps")

    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        model.train()
        train_loss = 0.0
        seen = 0
        for batch in tqdm(train_loader, desc=f"ef epoch {epoch}"):
            x = batch["video"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y)
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
        preds, targets = [], []
        val_loss = 0.0
        val_seen = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["video"].to(device, non_blocking=True)
                y = batch["target"].to(device, non_blocking=True)
                pred = model(x)
                loss = F.smooth_l1_loss(pred, y)
                val_loss += loss.item() * x.size(0)
                val_seen += x.size(0)
                preds.extend(pred.detach().cpu().tolist())
                targets.extend(y.detach().cpu().tolist())
        val_loss /= max(val_seen, 1)
        em = ef_metrics(preds, targets)
        row = {"epoch": epoch, "step": global_step, "augmentation_id": aug_id, "model_type": "ef_temporal_proxy", "seed": seed, "train_loss": train_loss, "val_loss": val_loss, **em}
        metrics.append(row)
        save_checkpoint(run_dir / "checkpoints" / "last.pt", model, opt, epoch, best)
        if em["mae"] < best:
            best = em["mae"]
            save_checkpoint(run_dir / "checkpoints" / "best.pt", model, opt, epoch, best)
        if max_steps and global_step >= int(max_steps):
            break
    print(f"run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
