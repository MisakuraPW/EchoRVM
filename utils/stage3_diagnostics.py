"""Read-only memory interventions; no architecture or checkpoint changes."""

from collections import deque

import torch
from torch.nn import functional as F


def conditions(prefixes):
    maximum = max(prefixes)
    return [(f'history_{p}', p, 'clean') for p in prefixes] + [
        ('repeat_prefix', maximum, 'repeat_prefix'),
        ('zero_prefix_clip', maximum, 'zero_prefix_clip'),
        ('degraded_no_history', 0, 'degraded_recent'),
        ('degraded_history', maximum, 'degraded_recent')]


def intervention_video(video, prefix, recent, kind, local_frames):
    """All conditions end on exactly the same recent frames, never resample."""
    if prefix < 0 or recent < 1 or prefix + recent > video.shape[1]:
        raise ValueError('Insufficient real context')
    if prefix % local_frames or recent % local_frames:
        raise ValueError('Context must contain complete clips')
    clip = video[:, -(prefix + recent):].clone()
    if kind == 'repeat_prefix':
        if prefix < local_frames:
            raise ValueError('Repeat requires a prefix')
        clip[:, :prefix] = clip[:, :local_frames].repeat(1, prefix // local_frames, 1, 1, 1)
    elif kind == 'zero_prefix_clip':
        if prefix < local_frames:
            raise ValueError('Corruption requires a prefix')
        clip[:, prefix-local_frames:prefix] = 0
    elif kind == 'degraded_recent':
        h, w = clip.shape[-2:]
        clip[:, prefix:, :, h//4:3*h//4, w//4:3*w//4] = 0
    elif kind != 'clean':
        raise ValueError(kind)
    return clip


@torch.no_grad()
def stream_audit(model, video, recent, observe=False):
    """Bounded FIFO of descriptors; optional per-clip compression/update traces."""
    length = model.local_frames
    if video.shape[1] < recent or video.shape[1] % length or recent % length:
        raise ValueError('Invalid stream/window length')
    fifo = deque(maxlen=recent//length)
    state = short = boundary = None
    trace, captured = [], {}
    handles = []
    compressed, updated = deque(maxlen=recent//length), deque(maxlen=recent//length)
    if model.memory_mode != 'none':
        def before_memory(module, inputs):
            tokens, old = inputs
            old = torch.zeros_like(tokens) if old is None else old
            captured['pooled'] = tokens.detach()
            captured['old'] = old.detach()
            if observe:
                captured['update'] = torch.sigmoid(module.update_x(tokens) + module.update_s(old)).detach()
                captured['reset'] = torch.sigmoid(module.reset_x(tokens) + module.reset_s(old)).detach()
        handles.append(model.memory.register_forward_pre_hook(before_memory))
    try:
        for index, clip in enumerate(video.split(length, 1)):
            if index * length == video.shape[1] - recent:
                boundary = state
            out = model.stream_clip(clip, state, short)
            state, short = out['final_state'], out['final_short']
            local = model.frame_features(out['local_features']).mean(2)
            fused = model.frame_features(out['features']).mean(2)
            fifo.append((local, fused))
            if state is not None:
                compressed.append(captured['pooled'].mean(1))
                updated.append(state.mean(1))
            if observe:
                row = dict(clip=index, in_recent=index*length >= video.shape[1]-recent,
                           fusion_rms=float((fused.float()-local.float()).square().mean().sqrt()))
                if state is not None:
                    old, pooled = captured['old'].float(), captured['pooled'].float()
                    update, reset = captured['update'].float(), captured['reset'].float()
                    row.update(state_rms=float(state.float().square().mean().sqrt()),
                        state_change_rms=float((state.float()-old).square().mean().sqrt()),
                        pooled_rms=float(pooled.square().mean().sqrt()),
                        update_mean=float(update.mean()), reset_mean=float(reset.mean()),
                        update_saturation=float(((update<.05)|(update>.95)).float().mean()),
                        slot_variance=float(state.float().var(1, unbiased=False).mean()))
                trace.append(row)
    finally:
        for handle in handles:
            handle.remove()
    local = torch.cat([v[0] for v in fifo], 1)
    fused = torch.cat([v[1] for v in fifo], 1)
    slots = 1 if model.memory_mode == 'global' else model.memory_grid**2
    empty = local.new_zeros(len(video), slots, model.embed_dim)
    boundary = empty if boundary is None else boundary.to(local)
    outputs = dict(cache=fused, local=local,
                local_history=torch.cat((local, boundary), 1),
                local_empty=torch.cat((local, empty), 1))
    if compressed:
        outputs.update(compressed=torch.stack(list(compressed),1), updated=torch.stack(list(updated),1))
    return outputs, trace


def gradient_audit(model, video, seed=42):
    """Last-clip masked loss sensitivity to earlier INPUTS, not target gradients.

    This is not a parameter-gradient norm or evidence of causal clinical value.
    No optimizer is constructed, no parameter/RNG is changed.
    """
    flags = [p.requires_grad for p in model.parameters()]
    device = video.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    try:
        model.requires_grad_(False)
        with torch.random.fork_rng(devices=devices), torch.enable_grad():
            torch.manual_seed(seed)
            x = video[:, -model.frames:].detach().float().requires_grad_(True)
            # FP32 and B=1 deliberately keep this diagnostic independent of AMP scaling.
            out = model(x)
            n = model.patch_embed.num_patches
            error = (out['pred'][:, -n:].float() - out['target'][:, -n:].detach().float()).square().mean(-1)
            mask = out['mask'][:, -n:].float()
            loss = (error*mask).sum()/mask.sum().clamp_min(1)
            gradient, = torch.autograd.grad(loss, x)
            return dict(last_clip_loss=float(loss.detach()),
                input_gradient_rms=[float(g.square().mean().sqrt()) for g in gradient.split(model.local_frames, 1)],
                local_frames=model.local_frames, native_frames=model.frames,
                note='Last-clip masked reconstruction input sensitivity, fixed mask seed; not a quality score.')
    finally:
        for p, flag in zip(model.parameters(), flags):
            p.requires_grad_(flag)
