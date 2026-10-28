"""Active-clip batching for deterministic, complete-clip SSL windows."""

import torch

from models.final_temporal_mae import FinalTemporalMAE
from models.temporal_mae import temporal_mask
from models.video_mae import tubelet_patchify


def supported(model):
    # Stochastic layers depend on execution order; keep their reference path.
    return (isinstance(model, FinalTemporalMAE) and model.research_mask in {'tube', 'complementary'}
            and not any(getattr(layer, 'p', 0) or getattr(layer, 'drop_prob', 0)
                        for layer in model.modules()))


def local_objective(model, clip, mask, indices, state):
    encoded, raw = model.encode_current(clip, mask, state, visible_indices=indices)
    writing = raw if model.memory_write_source == 'local' else encoded
    if (model.memory_mode in {'global', 'none'} and model.memory_compression == 'mean'
            and getattr(model, 'forward_variant', None) != 'content_write'):
        pooled = writing.mean(1, keepdim=True)
    else:
        pooled = model.write_summary(writing, mask, torch.ones_like(mask), state)
    pred = model._decode(encoded, mask, state, visible_indices=indices)
    target = tubelet_patchify(model._normalize_input(clip), model.tubelet_size, model.patch_size)
    if model.norm_pix_loss:
        target = (target-target.mean(-1, keepdim=True))/(target.var(-1, keepdim=True, unbiased=False)+1e-6).sqrt()
    error = ((pred.float()-target.float()).square().mean(-1)*mask).sum(1)
    if model.memory_mode != 'none':
        _, state = model._call(model.memory, pooled, state)
    return error, mask.float().sum(1), state


def enable_compiled_clips(model):
    """Compile only tensor work; parameter/state-dict names remain unchanged."""
    if (not supported(model) or model.frame_readout != 'repeat' or model.memory_mode != 'global'
            or model.memory_compression != 'mean' or model.gradient_checkpointing
            or getattr(model, 'forward_variant', None) == 'content_write'):
        raise ValueError('Compiled clips require deterministic B0/A global-repeat execution without recomputation')
    import functools
    model._pipeline_compiled_clip = torch.compile(functools.partial(local_objective, model),
                                                   dynamic=True, fullgraph=True)


def active_clip_loss(model, samples, seed):
    """Same per-window objective, but batch real clips across unequal lengths.

    A finished window leaves the active batch. No padding, state detachment,
    prefix truncation, sample reordering or cross-patient state reuse is used.
    """
    if not samples or not supported(model):
        raise ValueError('Active-clip execution requires a deterministic temporal MAE')
    device = next(model.parameters()).device
    lengths = [len(sample['video']) for sample in samples]
    if any(t < model.local_frames or t % model.local_frames for t in lengths):
        raise ValueError('SSL windows must contain real complete local clips')
    if any(tuple(s['video'].shape[-2:]) != (model.img_size, model.img_size) for s in samples):
        raise ValueError('MAE data/model spatial contract mismatch')
    gt, gh, gw = model.token_grid
    orders = []
    for sample in samples:
        generator = torch.Generator(device=device).manual_seed(seed + int(sample['record_index'])*1009)
        orders.append(torch.rand(1, gh*gw, generator=generator, device=device).argsort(-1))
    orders = torch.cat(orders)
    # Preserve the loader's pinned storage instead of concatenating pageable CPU
    # clips at every recurrent step. Device-side stacking stays asynchronous.
    videos = [sample['video'].to(device, non_blocking=True) for sample in samples]
    keep = gt*max(1, int(gh*gw*(1-model.mask_ratio)))
    state, old_active = None, list(range(len(samples)))
    numerator = torch.zeros(len(samples), device=device)
    denominator = torch.zeros_like(numerator)
    for step in range(max(lengths)//model.local_frames):
        start = step*model.local_frames
        active = [i for i, length in enumerate(lengths) if start < length]
        ids = torch.tensor(active, device=device)
        if state is not None and active != old_active:
            positions = torch.tensor([old_active.index(i) for i in active], device=device)
            state = state.index_select(0, positions)
        clip = torch.stack([videos[i][start:start+model.local_frames] for i in active])
        if model.in_chans == 1:
            clip = clip[:, :, :1]
        elif clip.shape[2] == 1:
            clip = clip.expand(-1, -1, model.in_chans, -1, -1)
        mask = temporal_mask(len(active), model.token_grid, model.mask_ratio, model.research_mask,
                             device, step, orders.index_select(0, ids))
        indexed = model.frame_readout == 'repeat'
        indices = mask.to(torch.int8).argsort(dim=1, stable=True)[:, :keep] if indexed else None
        step_function = getattr(model, '_pipeline_compiled_clip', None)
        error, count, state = (step_function(clip, mask, indices, state) if step_function is not None else
                               local_objective(model, clip, mask, indices, state))
        included = torch.tensor([start+model.local_frames > lengths[i]-min(lengths[i], model.reconstruction_recent_frames)
                                 for i in active], device=device)
        numerator = numerator.index_add(0, ids, error*included)
        denominator = denominator.index_add(0, ids, count*included)
        old_active = active
    pixel = (numerator/denominator.clamp_min(1)).mean()
    orth = model.frame_expansion.orthogonal_loss() if model.frame_readout in {'factorized', 'soft_factorized'} else pixel.new_zeros(())
    return pixel + model.dynamic_orthogonal_weight*orth
