"""Single-device Muon + AdamW hybrid optimizer.

Adapted from the provided Muon reference with a narrow, single-device scope for
this project. Muon is applied to hidden matrix weights; AdamW handles tokens,
normalization, biases, and other non-hidden parameters.
"""

from __future__ import annotations

import torch


def zeropower_via_newtonschulz5(grad: torch.Tensor, steps: int = 5) -> torch.Tensor:
    if grad.ndim < 2:
        raise ValueError("Muon update expects tensors with ndim >= 2")
    original_shape = grad.shape
    x = grad.flatten(1) if grad.ndim > 2 else grad
    if x.shape[0] > x.shape[1]:
        x = x.t()
        transposed = True
    else:
        transposed = False
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        xtx = x @ x.t()
        x = a * x + (b * xtx + c * xtx @ xtx) @ x
    if transposed:
        x = x.t()
    return x.reshape(original_shape)


def muon_update(grad: torch.Tensor, momentum: torch.Tensor, beta: float = 0.95, ns_steps: int = 5, nesterov: bool = True) -> torch.Tensor:
    momentum.mul_(beta).add_(grad)
    update = grad.add(momentum, alpha=beta) if nesterov else momentum
    return zeropower_via_newtonschulz5(update, steps=ns_steps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """Hybrid optimizer controlled by param group flag ``use_muon``."""

    def __init__(self, param_groups: list[dict]):
        for group in param_groups:
            if group.get("use_muon", False):
                group.setdefault("momentum", 0.95)
                group.setdefault("weight_decay", 0.0)
                group.setdefault("ns_steps", 5)
                group.setdefault("nesterov", True)
            else:
                group.setdefault("betas", (0.9, 0.95))
                group.setdefault("eps", 1e-8)
                group.setdefault("weight_decay", 0.0)
        super().__init__(param_groups, defaults={})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon", False):
                lr = group["lr"]
                wd = group["weight_decay"]
                momentum_beta = group["momentum"]
                ns_steps = group["ns_steps"]
                nesterov = group["nesterov"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    if grad.ndim < 2:
                        continue
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(grad, state["momentum_buffer"], beta=momentum_beta, ns_steps=ns_steps, nesterov=nesterov)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    scale = max(1.0, p.shape[0] / max(1, p.shape[1]))**0.5
                    p.add_(update, alpha=-lr * scale)
            else:
                lr = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                wd = group["weight_decay"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    grad = p.grad
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.0, device=p.device)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    state["step"].add_(1)
                    step = int(state["step"].item())
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step
                    step_size = lr * (bias_correction2**0.5) / bias_correction1
                    p.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)
        return loss
