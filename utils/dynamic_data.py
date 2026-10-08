"""Matched real-video windows containing the two traced events, without padding."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from echo_aug_validation.io_utils import load_echonet_filelist, find_echonet_video
from utils.datasets import _as_video_tensor
from utils.downstream_datasets import _rasterize_echonet_trace
from utils.echo_input import read_echo_input


def build_dynamic_manifest(root, frames, train_cases, val_cases, seed):
    if frames < 64 or frames % 16 or min(train_cases, val_cases) < 1:
        raise ValueError('Need positive case budgets and complete L16 windows >=64 frames')
    root = Path(root)
    traces = pd.read_csv(root / 'VolumeTracings.csv')
    traces['_stem'] = traces.FileName.map(lambda s: Path(str(s)).stem)
    labels = {}
    for (case, frame), group in traces.groupby(['_stem', 'Frame']):
        labels.setdefault(case, []).append((int(frame), int(_rasterize_echonet_trace(group, (112, 112)).sum())))
    manifest, excluded = {}, []
    for split, budget in (('train', train_cases), ('val', val_cases)):
        filelist = load_echonet_filelist(root, split).sample(frac=1, random_state=seed).reset_index(drop=True)
        rows = []
        for index, row in filelist.iterrows():
            case = Path(str(row.FileName)).stem
            events = labels.get(case, [])
            path = find_echonet_video(root, str(row.FileName))
            fps = float(row.get('FPS', float('nan')))
            reason = None
            if path is None or len(events) != 2:
                reason = 'missing_video_or_two_traced_events'
            elif not np.isfinite(fps) or fps <= 0:
                reason = 'missing_positive_fps'
            elif events[0][1] == events[1][1]:
                reason = 'ambiguous_equal_traced_area'
            if reason is None:
                raw = read_echo_input(path, 'gray_repeat3')
                n = len(raw)
                ed, es = max(events, key=lambda e: e[1])[0], min(events, key=lambda e: e[1])[0]
                lower, upper = max(0, max(ed, es) - frames + 1), min(min(ed, es), n - frames)
                if lower > upper or min(ed, es) < 0 or max(ed, es) >= n:
                    reason = 'no_complete_window_containing_both_events'
                else:
                    rng = np.random.default_rng(seed + index * 997)
                    start = int(rng.integers(lower, upper + 1))
                    stat = path.stat()
                    rows.append(dict(patient=case, source_path=str(path.resolve()),
                                     source_bytes=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                                     start=start, frames=frames, fps=fps, ed_frame=ed, es_frame=es,
                                     ef=float(row.EF), raw_frames=n))
                del raw
            if reason:
                excluded.append(dict(split=split, patient=case, reason=reason))
            if len(rows) == budget:
                break
        if len(rows) != budget:
            raise ValueError(f'{split}: only {len(rows)}/{budget} eligible videos; reduce explicit case budget')
        manifest[split] = rows
    if {r['patient'] for r in manifest['train']} & {r['patient'] for r in manifest['val']}:
        raise ValueError('Training/validation patient overlap')
    return dict(version=1, seed=seed, frames=frames, splits=manifest, excluded=excluded,
                label_source='Larger/smaller official traced-mask area; sparse ED/ES proxy, not dense phase GT')


class DynamicWindowDataset(Dataset):
    def __init__(self, records, input_protocol='gray_repeat3', channels=3):
        self.records = records
        self.input_protocol, self.channels = input_protocol, channels

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        path = Path(r['source_path'])
        stat = path.stat()
        if stat.st_size != r['source_bytes'] or stat.st_mtime_ns != r['source_mtime_ns']:
            raise ValueError(f'Data changed after manifest creation: {path}')
        raw = read_echo_input(path, self.input_protocol)
        clip = raw[r['start']:r['start'] + r['frames']]
        if len(clip) != r['frames']:
            raise ValueError('Incomplete real window; padding is forbidden')
        video = _as_video_tensor(clip, r['frames'], 112, channels=self.channels)
        return dict(video=video, frame_valid=torch.ones(r['frames'], dtype=torch.bool),
                    id=r['patient'], record_index=index)
