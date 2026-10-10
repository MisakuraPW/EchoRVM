"""Differentiable task exits from a single legal chronological forward."""

from __future__ import annotations

import torch


def encode_window(model, video, prefix_frames=0, recent_frames=64, target_index=None):
    prefix = int(prefix_frames)
    if prefix < 0 or prefix % model.local_frames or video.shape[1] % model.local_frames:
        raise ValueError('Task windows require complete real clips')
    if prefix >= video.shape[1]:
        raise ValueError('History must leave real recent observations')
    if video.shape[2] == 1 and model.in_chans == 3:
        video = video.expand(-1, -1, 3, -1, -1)
    if video.shape[1] - prefix > recent_frames:
        raise ValueError('Explicit recent capacity was exceeded')
    targets = None if target_index is None else torch.as_tensor(target_index, device=video.device).long()
    if targets is not None and (targets.shape != (len(video),) or bool((targets < prefix).any())
                               or bool((targets >= video.shape[1]).any())):
        raise ValueError('Target indices must refer to observed recent source frames')
    state = boundary = None
    sequences = dict(final=[], base=[], local=[], local_final=[])
    maps = {}
    for start in range(0, video.shape[1], model.local_frames):
        if start == prefix:
            boundary = state
        out = model.stream_clip(video[:, start:start + model.local_frames], state)
        state = out['final_state']
        if start < prefix:
            continue
        exits = dict(final=out['frame_outputs'], base=out['frame_base_outputs'],
                     local=out['local_base_outputs'], local_final=out['local_frame_outputs'])
        for name, value in exits.items():
            sequences[name].append(value.mean(2))
            if targets is not None:
                choose = (targets >= start) & (targets < start + model.local_frames)
                if choose.any():
                    if name not in maps:
                        maps[name] = value.new_zeros(len(video), value.shape[2], value.shape[3])
                    rows = choose.nonzero().flatten()
                    maps[name][rows] = value[rows, targets[rows] - start]
    return dict(sequences={key:torch.cat(values, 1) for key, values in sequences.items()},
                maps=maps, boundary=boundary, final_state=state)


def task_features(exits, task, exit_name='final', append_boundary=False, memory_slots=0):
    if exit_name not in {'final', 'base', 'local', 'local_final'}:
        raise ValueError('Unknown registered exit')
    if task == 'seg':
        return exits['maps'][exit_name], 0
    sequence = exits['sequences'][exit_name]
    if not append_boundary:
        return sequence, 0
    boundary = exits['boundary']
    if boundary is None:
        boundary = sequence.new_zeros(len(sequence), memory_slots, sequence.shape[-1])
    return torch.cat((sequence, boundary.to(sequence)), 1), boundary.shape[1]
