"""Low-cost representation audit for echocardiography MAE checkpoints.

The audit never updates the pretrained backbone.  It combines reconstruction,
feature geometry, augmentation/temporal diagnostics, EF ridge/kNN probes and a
patch-level linear segmentation probe.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models import build_echo_rmae, build_echo_single_frame_mae, build_echo_videomae
from models.downstream import EchoRMAEBackbone, EchoVideoMAEBackbone
from utils.datasets import build_rmae_dataset
from utils.downstream_datasets import EchoNetEFDataset, EchoNetSegmentationDataset
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen representation audit for one MAE checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--audit_frames", type=int, default=16)
    parser.add_argument("--recon_samples", type=int, default=512)
    parser.add_argument("--feature_samples", type=int, default=1024)
    parser.add_argument("--ef_train_samples", type=int, default=2000)
    parser.add_argument("--ef_val_samples", type=int, default=512)
    parser.add_argument("--seg_train_samples", type=int, default=512)
    parser.add_argument("--seg_val_samples", type=int, default=256)
    parser.add_argument("--seg_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def checkpoint_state(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if isinstance(ckpt.get(key), dict):
                ckpt = ckpt[key]
                break
    out = {}
    for key, value in ckpt.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module.") or key.startswith("_orig_mod."):
            key = key.split(".", 1)[1]
        out[key] = value
    return out


def load_model(path: str, device: torch.device) -> tuple[nn.Module, dict[str, Any], dict[str, Any]]:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = dict(cfg.get("model", {})) if isinstance(cfg, dict) else {}
    name = str(model_cfg.get("name", "echo_rmae")).lower()
    if name in {"echo_videomae", "videomae", "video_mae"}:
        model = build_echo_videomae(model_cfg)
    elif name in {"echo_single_frame_mae", "single_frame_mae", "videomae_single_frame"}:
        model = build_echo_single_frame_mae(model_cfg)
    else:
        model = build_echo_rmae(model_cfg)
    missing, unexpected = model.load_state_dict(checkpoint_state(ckpt), strict=False)
    model.to(device).eval()
    meta = {
        "checkpoint": str(path),
        "epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt, dict) else -1,
        "global_step": int(ckpt.get("global_step", -1)) if isinstance(ckpt, dict) else -1,
        "model_name": name,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "checkpoint_mb": Path(path).stat().st_size / (1024**2),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
    }
    return model, model_cfg, meta


def make_backbone(model: nn.Module) -> nn.Module:
    if hasattr(model, "forward_features") and hasattr(model, "tubelet_size"):
        return EchoVideoMAEBackbone(model)
    return EchoRMAEBackbone(model)


def loader_kwargs(batch_size: int, workers: int) -> dict[str, Any]:
    out = {"batch_size": batch_size, "shuffle": False, "num_workers": workers, "pin_memory": torch.cuda.is_available()}
    if workers > 0:
        out.update(persistent_workers=True, prefetch_factor=4)
    return out


def sample_limit(loader: DataLoader, maximum: int):
    seen = 0
    for batch in loader:
        if seen >= maximum:
            break
        yield batch
        key = "video" if "video" in batch else "image"
        seen += int(batch[key].shape[0])


def to_video(batch: dict[str, Any], device: torch.device) -> torch.Tensor:
    if "video" in batch:
        return batch["video"].to(device, non_blocking=True)
    return batch["image"].unsqueeze(1).to(device, non_blocking=True)


def adapt_video_for_model(video: torch.Tensor, model: nn.Module) -> torch.Tensor:
    expected = int(getattr(model, "frames", video.shape[1]))
    if hasattr(model, "frame_mae"):
        return video
    if video.shape[1] == expected:
        return video
    if video.shape[1] == 1:
        return video.expand(-1, expected, -1, -1, -1)
    ids = torch.linspace(0, video.shape[1] - 1, expected, device=video.device).round().long()
    return video.index_select(1, ids)


@torch.inference_mode()
def feature_sequence(backbone: nn.Module, model: nn.Module, video: torch.Tensor) -> torch.Tensor:
    video = adapt_video_for_model(video, model)
    return backbone.forward_tokens(video)["outputs"]


def perturb_clip(video: torch.Tensor) -> torch.Tensor:
    """A fixed clip-consistent photometric/speckle view for invariance auditing."""
    gamma = 1.15
    contrast = 1.08
    brightness = 0.03
    x = video.clamp(0, 1).pow(gamma)
    mean = x.mean(dim=(-1, -2), keepdim=True)
    x = (x - mean) * contrast + mean + brightness
    noise = torch.randn_like(x[:, :1]) * 0.05
    noise = noise.expand_as(x)
    return (x + x * noise).clamp(0, 1)


def safe_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    if a.numel() < 2 or float(a.std()) < 1e-8 or float(b.std()) < 1e-8:
        return 0.0
    return float(torch.corrcoef(torch.stack((a, b)))[0, 1])


@torch.inference_mode()
def benchmark_compute(
    model: nn.Module,
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    repeats: int = 10,
) -> dict[str, float]:
    batch = next(iter(loader))
    video = to_video(batch, device)
    model_video = adapt_video_for_model(video, model)
    warmup = 3 if device.type == "cuda" else 1
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(warmup):
        model(model_video)
        feature_sequence(backbone, model, video)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    import time

    start = time.perf_counter()
    for _ in range(repeats):
        model(model_video)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reconstruction_ms = (time.perf_counter() - start) * 1000.0 / repeats

    start = time.perf_counter()
    for _ in range(repeats):
        feature_sequence(backbone, model, video)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    encoder_ms = (time.perf_counter() - start) * 1000.0 / repeats
    peak = torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
    batch_size = int(video.shape[0])
    return {
        "encoder_latency_ms_batch": encoder_ms,
        "encoder_samples_per_second": batch_size * 1000.0 / max(encoder_ms, 1e-9),
        "reconstruction_latency_ms_batch": reconstruction_ms,
        "reconstruction_samples_per_second": batch_size * 1000.0 / max(reconstruction_ms, 1e-9),
        "inference_peak_memory_gb": peak,
        "benchmark_batch_size": float(batch_size),
    }


@torch.inference_mode()
def audit_unsupervised(
    model: nn.Module,
    backbone: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_samples: int,
) -> dict[str, float]:
    pooled, reconstruction = [], []
    adjacent_cos, temporal_var, reverse_delta, aug_cos, dynamics_corr = [], [], [], [], []
    for batch in sample_limit(loader, max_samples):
        video = to_video(batch, device)
        model_video = adapt_video_for_model(video, model)
        out = model(model_video)
        reconstruction.append((float(out["loss"]), int(video.shape[0])))
        seq = feature_sequence(backbone, model, video)
        frame = seq.mean(dim=2)
        global_feature = frame.mean(dim=1)
        pooled.append(global_feature.cpu())
        if frame.shape[1] > 1:
            adjacent_cos.append(F.cosine_similarity(frame[:, 1:], frame[:, :-1], dim=-1).mean().cpu())
            temporal_var.append(frame.var(dim=1, unbiased=False).mean().cpu())
            pixel_delta = model_video[:, 1:].sub(model_video[:, :-1]).square().mean(dim=(2, 3, 4))
            feature_delta = frame[:, 1:].sub(frame[:, :-1]).square().mean(dim=-1)
            if pixel_delta.shape[1] != feature_delta.shape[1]:
                pixel_delta = F.interpolate(pixel_delta[:, None], size=feature_delta.shape[1], mode="linear", align_corners=False)[:, 0]
            dynamics_corr.append(torch.tensor(safe_corr(pixel_delta.cpu(), feature_delta.cpu())))
            reversed_feature = feature_sequence(backbone, model, video.flip(1)).mean(dim=(1, 2))
            reverse_delta.append((1.0 - F.cosine_similarity(global_feature, reversed_feature, dim=-1)).mean().cpu())
        augmented = perturb_clip(video)
        aug_feature = feature_sequence(backbone, model, augmented).mean(dim=(1, 2))
        aug_cos.append(F.cosine_similarity(global_feature, aug_feature, dim=-1).mean().cpu())

    features = torch.cat(pooled, dim=0).float()
    centered = features - features.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    probability = variance / variance.sum().clamp_min(1e-12)
    effective_rank = float(torch.exp(-(probability * probability.clamp_min(1e-12).log()).sum()))
    participation = float(variance.sum().square() / variance.square().sum().clamp_min(1e-12))
    normalized = F.normalize(features, dim=-1)
    count = min(normalized.shape[0], 512)
    pair = normalized[:count] @ normalized[:count].T
    off_diag = pair[~torch.eye(count, dtype=torch.bool)]
    recon_sum = sum(value * n for value, n in reconstruction)
    recon_n = sum(n for _, n in reconstruction)
    return {
        "reconstruction_loss": recon_sum / max(1, recon_n),
        "feature_std_mean": float(centered.std(dim=0).mean()),
        "collapsed_dim_fraction": float((centered.std(dim=0) < 1e-4).float().mean()),
        "effective_rank": effective_rank,
        "participation_ratio": participation,
        "mean_pairwise_cosine": float(off_diag.mean()) if off_diag.numel() else 1.0,
        "augmentation_cosine": float(torch.stack(aug_cos).mean()),
        "adjacent_frame_cosine": float(torch.stack(adjacent_cos).mean()) if adjacent_cos else math.nan,
        "temporal_feature_variance": float(torch.stack(temporal_var).mean()) if temporal_var else math.nan,
        "temporal_reverse_delta": float(torch.stack(reverse_delta).mean()) if reverse_delta else math.nan,
        "pixel_feature_dynamics_corr": float(torch.stack(dynamics_corr).mean()) if dynamics_corr else math.nan,
    }


@torch.inference_mode()
def extract_ef(
    dataset: EchoNetEFDataset,
    model: nn.Module,
    backbone: nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(dataset, **loader_kwargs(batch_size, workers))
    features, targets = [], []
    for batch in sample_limit(loader, maximum):
        video = batch["video"].to(device, non_blocking=True)
        seq = feature_sequence(backbone, model, video)
        features.append(seq.mean(dim=(1, 2)).cpu())
        targets.append(batch["target"].float())
    return torch.cat(features).float(), torch.cat(targets).float()


def ridge_predict(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, alpha: float = 10.0) -> torch.Tensor:
    mean = x_train.mean(0, keepdim=True)
    std = x_train.std(0, keepdim=True).clamp_min(1e-5)
    x = (x_train - mean) / std
    xv = (x_val - mean) / std
    y_mean = y_train.mean()
    y = y_train - y_mean
    # Solve in sample space when it is cheaper; this avoids a large D x D inverse.
    if x.shape[0] <= x.shape[1]:
        dual = torch.linalg.solve(x @ x.T + alpha * torch.eye(x.shape[0]), y)
        weight = x.T @ dual
    else:
        weight = torch.linalg.solve(x.T @ x + alpha * torch.eye(x.shape[1]), x.T @ y)
    return xv @ weight + y_mean


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    error = pred - target
    ss_res = error.square().sum()
    ss_tot = (target - target.mean()).square().sum().clamp_min(1e-8)
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error.square().mean().sqrt()),
        "r2": float(1.0 - ss_res / ss_tot),
        "corr": safe_corr(pred, target),
    }


def ef_probes(
    model: nn.Module,
    backbone: nn.Module,
    root: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    frames = int(getattr(model, "frames", max(args.audit_frames, int(cfg.get("frames", 1)))))
    image_size = int(cfg.get("img_size", 112))
    train = EchoNetEFDataset(root, "train", frames, image_size, limit=args.ef_train_samples)
    val = EchoNetEFDataset(root, "val", frames, image_size, limit=args.ef_val_samples)
    x_train, y_train = extract_ef(train, model, backbone, device, args.batch_size, args.num_workers, args.ef_train_samples)
    x_val, y_val = extract_ef(val, model, backbone, device, args.batch_size, args.num_workers, args.ef_val_samples)
    out = {}
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(x_train.shape[0], generator=generator)
    for fraction in (0.01, 0.1, 1.0):
        n = max(16, min(x_train.shape[0], int(round(x_train.shape[0] * fraction))))
        pred = ridge_predict(x_train[order[:n]], y_train[order[:n]], x_val)
        for key, value in regression_metrics(pred, y_val).items():
            out[f"ef_ridge_{int(fraction * 100):03d}pct_{key}"] = value

    train_norm = F.normalize(x_train, dim=-1)
    val_norm = F.normalize(x_val, dim=-1)
    k = min(5, x_train.shape[0])
    predictions = []
    for start in range(0, x_val.shape[0], 128):
        similarity = val_norm[start : start + 128] @ train_norm.T
        indices = similarity.topk(k, dim=1).indices
        predictions.append(y_train[indices].mean(dim=1))
    knn = torch.cat(predictions)
    for key, value in regression_metrics(knn, y_val).items():
        out[f"ef_knn5_{key}"] = value
    return out


@torch.inference_mode()
def extract_seg_tokens(
    dataset: EchoNetSegmentationDataset,
    model: nn.Module,
    backbone: nn.Module,
    device: torch.device,
    batch_size: int,
    workers: int,
    maximum: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    loader = DataLoader(dataset, **loader_kwargs(batch_size, workers))
    feature_rows, label_rows = [], []
    grid_size = int(backbone.grid_size)
    for batch in sample_limit(loader, maximum):
        video = to_video(batch, device)
        outputs = feature_sequence(backbone, model, video)
        if "target_index" in batch:
            index = batch["target_index"].to(device).clamp(0, outputs.shape[1] - 1)
            tokens = outputs[torch.arange(outputs.shape[0], device=device), index]
        else:
            tokens = outputs[:, outputs.shape[1] // 2]
        masks = batch["mask"].to(device)
        labels = F.interpolate(masks[:, None].float(), size=(grid_size, grid_size), mode="nearest")[:, 0].long()
        feature_rows.append(tokens.cpu())
        label_rows.append(labels.flatten(1).cpu())
    return torch.cat(feature_rows), torch.cat(label_rows), grid_size


def segmentation_linear_probe(
    model: nn.Module,
    backbone: nn.Module,
    root: str,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    image_size = int(cfg.get("img_size", 112))
    train = EchoNetSegmentationDataset(root, "train", image_size, limit=args.seg_train_samples)
    val = EchoNetSegmentationDataset(root, "val", image_size, limit=args.seg_val_samples)
    x_train, y_train, grid = extract_seg_tokens(
        train, model, backbone, device, args.batch_size, args.num_workers, args.seg_train_samples
    )
    x_val, y_val, _ = extract_seg_tokens(
        val, model, backbone, device, args.batch_size, args.num_workers, args.seg_val_samples
    )
    dim = x_train.shape[-1]
    head = nn.Linear(dim, 2).to(device)
    positives = y_train.sum().item()
    negatives = y_train.numel() - positives
    weights = torch.tensor([1.0, negatives / max(1.0, positives)], device=device).clamp(max=20.0)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-2, weight_decay=1e-3)
    flat_x = x_train.reshape(-1, dim)
    flat_y = y_train.reshape(-1)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.seg_steps):
        ids = torch.randint(0, flat_x.shape[0], (min(8192, flat_x.shape[0]),), generator=generator)
        logits = head(flat_x[ids].to(device))
        loss = F.cross_entropy(logits, flat_y[ids].to(device), weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.inference_mode():
        pred = []
        flat_val = x_val.reshape(-1, dim)
        for start in range(0, flat_val.shape[0], 16384):
            pred.append(head(flat_val[start : start + 16384].to(device)).argmax(dim=1).cpu())
        pred = torch.cat(pred).reshape_as(y_val)
    intersection = ((pred == 1) & (y_val == 1)).sum().item()
    denominator = (pred == 1).sum().item() + (y_val == 1).sum().item()
    dice = 2.0 * intersection / max(1, denominator)
    accuracy = float((pred == y_val).float().mean())
    return {
        "seg_linear_patch_dice": float(dice),
        "seg_linear_patch_accuracy": accuracy,
        "seg_linear_grid_size": float(grid),
        "seg_linear_train_samples": float(x_train.shape[0]),
    }


def main() -> int:
    args = parse_args()
    if args.smoke:
        args.recon_samples = min(args.recon_samples, 8)
        args.feature_samples = min(args.feature_samples, 8)
        args.ef_train_samples = 16
        args.ef_val_samples = 8
        args.seg_train_samples = 8
        args.seg_val_samples = 8
        args.seg_steps = 2
        args.num_workers = 0
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_cfg, metrics = load_model(args.checkpoint, device)
    backbone = make_backbone(model).to(device).eval()

    audit_cfg = dict(model_cfg)
    audit_cfg["frames"] = int(getattr(model, "frames", args.audit_frames))
    data_cfg = {
        "dataset_name": "EchoNet-Dynamic",
        "data_root": args.data_root,
        "train_split": "train",
        "val_split": "val",
    }
    dataset = build_rmae_dataset(data_cfg, audit_cfg, "val", seed=args.seed)
    loader = DataLoader(dataset, **loader_kwargs(args.batch_size, args.num_workers))
    metrics.update(benchmark_compute(model, backbone, loader, device, repeats=2 if args.smoke else 10))
    metrics.update(audit_unsupervised(model, backbone, loader, device, min(args.recon_samples, args.feature_samples)))
    metrics.update(ef_probes(model, backbone, args.data_root, model_cfg, args, device))
    metrics.update(segmentation_linear_probe(model, backbone, args.data_root, model_cfg, args, device))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "representation_metrics.json"
    output.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
