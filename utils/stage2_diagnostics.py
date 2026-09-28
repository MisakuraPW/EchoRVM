"""Frozen interface diagnostics, separate from training and legacy audit protocols."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def tubelet_weights(valid, tubelet):
    if valid.shape[1] % tubelet:
        raise ValueError('Validity length must be divisible by tubelet size')
    grouped = valid.bool().reshape(len(valid), -1, tubelet)
    return grouped.all(-1), grouped.any(-1) & ~grouped.all(-1)


def paired_bootstrap(rows_a, rows_b, field, seed=42, repetitions=2000):
    """Average repeated views/ED/ES within patient before paired resampling."""
    a, b = {r['id']: r for r in rows_a}, {r['id']: r for r in rows_b}
    if not a or len(a) != len(rows_a) or len(b) != len(rows_b) or a.keys() != b.keys():
        raise ValueError('Paired rows require unique identical sample IDs')
    grouped = {}
    for key in a:
        if a[key]['patient'] != b[key]['patient']:
            raise ValueError('Patient mismatch')
        grouped.setdefault(a[key]['patient'], []).append(float(a[key][field]) - float(b[key][field]))
    values = np.asarray([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    means = np.asarray([values[rng.integers(len(values), size=len(values))].mean()
                        for _ in range(repetitions)])
    return dict(delta=float(values.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), patients=len(values))


@torch.inference_mode()
def extract_batch(model, video, valid, device, target_indices=None, amp=True):
    """Reduce each native window immediately; never retain full-video token maps."""
    if getattr(model, 'frame_readout', 'repeat') != 'repeat':
        raise ValueError('Legacy tubelet audit cannot evaluate learned frame expansion; use tools/evaluate_stage2.py')
    batch, frames = valid.shape
    if frames % model.frames:
        raise ValueError('Audit frame count must be a multiple of the checkpoint native window')
    if target_indices is not None and (torch.any(target_indices < 0) or torch.any(target_indices >= frames)):
        raise ValueError('Target index out of range')
    sums = {key: torch.zeros(batch, model.embed_dim, device=device) for key in ('local', 'fused')}
    denominator = torch.zeros(batch, device=device)
    maps = {key: None for key in ('local', 'fused', 'local_mean', 'fused_mean')}
    complete_total = torch.zeros(batch, dtype=torch.long)
    partial_total = complete_total.clone()
    max_abs, squared, elements = 0., 0., 0
    target_indices = target_indices.to(device) if target_indices is not None else None
    for start in range(0, frames, model.frames):
        clip = video[:, start:start+model.frames].to(device, non_blocking=True)
        fv = valid[:, start:start+model.frames].to(device, non_blocking=True)
        good, partial = tubelet_weights(fv, model.tubelet_size)
        with torch.autocast(device.type, enabled=amp and device.type == 'cuda'):
            result = model.diagnostic_features(clip, fv)
        local, fused = result['local_features'], result['features']
        complete_total += good.sum(1).cpu()
        partial_total += partial.sum(1).cpu()
        denominator += good.sum(1)
        delta = fused.float()-local.float()
        max_abs = max(max_abs, float(delta.abs().max()))
        squared += float(delta.square().sum())
        elements += delta.numel()
        for name, tokens in (('local', local), ('fused', fused)):
            sums[name] += (tokens.float().mean(2)*good[:, :, None]).sum(1)
            if target_indices is None:
                continue
            in_window = (target_indices >= start) & (target_indices < start+model.frames)
            if not bool(in_window.any()):
                continue
            idx = ((target_indices-start).clamp(0, model.frames-1)//model.tubelet_size)
            chosen = tokens[torch.arange(batch, device=device), idx].float()
            # Mean of the SAME local clip containing the target, not a whole-video mean.
            local_t = model.local_frames//model.tubelet_size
            clip_ids = idx//local_t
            positions = clip_ids[:, None]*local_t + torch.arange(local_t, device=device)[None]
            group = tokens[torch.arange(batch, device=device)[:, None], positions].float()
            group_valid = good[torch.arange(batch, device=device)[:, None], positions]
            mean = (group*group_valid[:, :, None, None]).sum(1)/group_valid.sum(1)[:, None, None].clamp_min(1)
            for key, value in ((name, chosen), (name+'_mean', mean)):
                if maps[key] is None:
                    maps[key] = torch.zeros_like(value)
                maps[key][in_window] = value[in_window]
        del result, local, fused, delta, clip
    if bool((denominator == 0).any()):
        raise ValueError('A case has no complete valid tubelet; cannot compute a meaningful EF feature')
    pooled = {name:(value/denominator[:, None]).cpu() for name,value in sums.items()}
    # Quantify the old denominator dilution without another model pass.
    factor = denominator.cpu()*model.tubelet_size/valid.sum(1).clamp_min(1)
    pooled['legacy_fused'] = pooled['fused']*factor[:, None]
    if target_indices is not None:
        gh, gw = model.token_grid[1:]
        maps = {name:value.transpose(1,2).reshape(batch,model.embed_dim,gh,gw).cpu()
                for name,value in maps.items()}
    return dict(pooled=pooled, maps=maps, complete=complete_total, partial=partial_total,
                fusion_max_abs=max_abs, fusion_square_sum=squared, fusion_elements=elements)


@torch.inference_mode()
def tune_batch(model, sample, device, maximum=32, memory_fraction=.80, amp=True):
    """Short forward-only throughput search. Does not tune scientific hyperparameters."""
    if device.type != 'cuda':
        return 1, [dict(batch_size=1, status='cpu', samples_per_second=None)]
    trials = []
    sizes = sorted({1, maximum, *(n for n in (2,4,8,16,32) if n <= maximum)})
    for size in sizes:
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(device)
        allowance = min(total*memory_fraction, free*.9+torch.cuda.memory_allocated(device))
        torch.cuda.reset_peak_memory_stats(device)
        video = sample['video'][None,:model.frames].expand(size,-1,-1,-1,-1)
        valid = sample['frame_valid'][None,:model.frames].expand(size,-1)
        try:
            for _ in range(1):
                extract_batch(model,video,valid,device,amp=amp)
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(2):
                extract_batch(model,video,valid,device,amp=amp)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            trials.append(dict(batch_size=size, status='ok' if peak < allowance else 'headroom',
                               samples_per_second=2*size/(time.perf_counter()-start), peak_bytes=peak))
            if peak >= allowance:
                break
        except torch.cuda.OutOfMemoryError:
            trials.append(dict(batch_size=size,status='oom'))
            break
    torch.cuda.empty_cache()
    valid_trials = [r for r in trials if r['status']=='ok']
    if not valid_trials:
        raise RuntimeError('No batch passed the GPU memory reserve; free GPU memory and retry')
    best = max(r['samples_per_second'] for r in valid_trials)
    selected = min((r for r in valid_trials if r['samples_per_second'] >= best*.97), key=lambda r:r['batch_size'])
    return selected['batch_size'], trials


class OffsetReadout(nn.Module):
    """Two equal-budget controls: branch averaging vs position-based routing."""

    def __init__(self, channels, branches=1, routed=False):
        super().__init__()
        self.proj = nn.Conv2d(channels, 2*branches, 1)
        self.branches, self.routed = branches, routed

    def forward(self, x, offsets):
        out = self.proj(x).reshape(len(x),self.branches,2,*x.shape[-2:])
        if self.routed:
            return out[torch.arange(len(x),device=x.device), offsets]
        return out.mean(1)
