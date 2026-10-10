"""Bounded inference cache. No implicit patient reset, no retained training graphs."""

from collections import deque

import torch


class StreamingFeatureCache:
    def __init__(self, model, capacity=4):
        if capacity < 1:
            raise ValueError('capacity must be positive')
        self.model, self.capacity = model, int(capacity)
        self.reset()

    def reset(self, patient_ids=None):
        self.patient_ids = None if patient_ids is None else tuple(patient_ids)
        self.state = self.short_state = None
        self.entries = deque(maxlen=self.capacity)
        self.next_index = None
        self.observed_clips = 0

    @torch.inference_mode()
    def update(self, video, patient_ids, frame_indices):
        if self.model.training:
            raise RuntimeError('Inference FIFO requires model.eval(); training graphs are not cached')
        ids = tuple(patient_ids)
        if len(ids) != len(video) or len(set(ids)) != len(ids):
            raise ValueError('Each batch slot needs a unique patient ID')
        if self.patient_ids is not None and self.patient_ids != ids:
            raise ValueError('Patient/batch order changed: call reset(patient_ids) explicitly')
        indices = torch.as_tensor(frame_indices, device=video.device, dtype=torch.long)
        if indices.shape != video.shape[:2] or bool((indices < 0).any()):
            raise ValueError('Streaming requires real frame indices; padding is not history')
        if bool((indices[:, 1:] - indices[:, :-1] != 1).any()):
            raise ValueError('This streaming protocol requires dense consecutive frames')
        if self.next_index is not None and not torch.equal(indices[:, 0], self.next_index):
            raise ValueError('Duplicated, skipped or reordered frames; reset at discontinuities')
        previous = self.state
        out = self.model.stream_clip(video, self.state, self.short_state)
        self.patient_ids = ids
        self.state, self.short_state = out['final_state'], out['final_short']
        local = out['local_frame_outputs'] if 'local_frame_outputs' in out else self.model.frame_features(out['local_features'])
        fused = out['frame_outputs'] if 'frame_outputs' in out else self.model.frame_features(out['features'])
        self.entries.append(dict(local=local.mean(2).detach(),
            fused=fused.mean(2).detach(),
            before=previous, indices=indices.detach().clone()))
        self.next_index = indices[:, -1] + 1
        self.observed_clips += 1
        return dict(frame_features=fused,
                    frame_indices=indices, **self.read())

    def read(self):
        """Read does not advance memory. Boundary is BEFORE the oldest retained clip."""
        if not self.entries:
            raise RuntimeError('Empty stream')
        items = list(self.entries)
        return dict(local=torch.cat([e['local'] for e in items], 1),
                    fused=torch.cat([e['fused'] for e in items], 1),
                    last=items[-1]['fused'], boundary=items[0]['before'],
                    final_state=self.state,
                    source_indices=torch.cat([e['indices'] for e in items], 1),
                    older_clips=self.observed_clips-len(items))


def ef_sequences(readout):
    """Same cache and same head slots for real vs empty older-history conditions."""
    fused, local = readout['fused'], readout['local']
    b, _, d = fused.shape
    boundary, final = readout['boundary'], readout['final_state']
    if boundary is None or final is None or readout['older_clips'] < 1:
        raise ValueError('EF history comparison needs a real prefix and recurrent state')
    # Keep all spatial memory slots; the task head decides how to pool them.
    zero = torch.zeros_like(boundary)
    return dict(last=readout['last'], cache=fused, state=final,
        cache_empty=torch.cat((fused, zero), 1),
        cache_history=torch.cat((fused, boundary), 1),
        local_empty=torch.cat((local, zero), 1),
        local_history=torch.cat((local, boundary), 1))
