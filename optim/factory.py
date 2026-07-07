"""Optimizer construction and parameter grouping."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .muon import SingleDeviceMuonWithAuxAdam


@dataclass
class OptimizerStats:
    muon_tensors: int
    muon_params: int
    adam_tensors: int
    adam_params: int


def _is_adamw_name(name: str) -> bool:
    lowered = name.lower()
    adam_keywords = [
        "bias",
        "norm",
        "pos_embed",
        "mask_token",
        "background_token",
        "cls_token",
        "scale",
        "head",
    ]
    return any(key in lowered for key in adam_keywords)


def split_muon_adamw_params(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter], OptimizerStats]:
    muon_params: list[torch.nn.Parameter] = []
    adam_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and not _is_adamw_name(name):
            muon_params.append(param)
        else:
            adam_params.append(param)
    stats = OptimizerStats(
        muon_tensors=len(muon_params),
        muon_params=sum(p.numel() for p in muon_params),
        adam_tensors=len(adam_params),
        adam_params=sum(p.numel() for p in adam_params),
    )
    return muon_params, adam_params, stats


def build_optimizer(model: torch.nn.Module, cfg: dict) -> tuple[torch.optim.Optimizer, OptimizerStats]:
    name = str(cfg.get("name", "muon_adamw_hybrid")).lower()
    lr = float(cfg.get("lr", 1e-4))
    weight_decay = float(cfg.get("weight_decay", 0.05))
    if name in {"adamw", "adam"}:
        params = [p for p in model.parameters() if p.requires_grad]
        stats = OptimizerStats(0, 0, len(params), sum(p.numel() for p in params))
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=tuple(cfg.get("betas", (0.9, 0.95)))), stats
    if name not in {"muon_adamw_hybrid", "muon", "hybrid_muon"}:
        raise ValueError(f"Unknown optimizer: {cfg.get('name')}")
    muon_params, adam_params, stats = split_muon_adamw_params(model)
    groups = []
    if adam_params:
        groups.append(
            {
                "params": adam_params,
                "lr": float(cfg.get("adamw_lr", lr)),
                "betas": tuple(cfg.get("betas", (0.9, 0.95))),
                "eps": float(cfg.get("eps", 1e-8)),
                "weight_decay": weight_decay,
                "use_muon": False,
            }
        )
    if muon_params:
        groups.append(
            {
                "params": sorted(muon_params, key=lambda p: p.numel(), reverse=True),
                "lr": float(cfg.get("muon_lr", lr)),
                "momentum": float(cfg.get("muon_momentum", 0.95)),
                "weight_decay": weight_decay,
                "ns_steps": int(cfg.get("muon_ns_steps", 5)),
                "nesterov": bool(cfg.get("muon_nesterov", True)),
                "use_muon": True,
            }
        )
    return SingleDeviceMuonWithAuxAdam(groups), stats
