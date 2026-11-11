"""Restore annotated frames without changing the recurrent encoder or task head."""

import math
from pathlib import Path
import numpy as np
import torch

from .artifacts import digest, file_hash
from .echo import IdentifiedDataset
from .spatial_baseline import build as build_spatial
from utils.final_temporal_data import (WindowDataset, _close, _file_provenance,
                                      _prepare_video, _rng, _trace_mask)
from utils.echo_input import read_echo_input


class CoverageDataset(WindowDataset):
    def __init__(self, manifest, split, *, training=False, aug_cfg=None, seed=42,
                 recent_frames=64, local_frames=16, max_prefix=0, prefix=0,
                 positions='all', allow_test=False):
        if max_prefix or prefix:
            raise ValueError('Coverage repair is defined for W64 H0 only')
        super().__init__(manifest, split, task='seg', recent_frames=recent_frames,
                         local_frames=local_frames, max_prefix=0, prefix=0,
                         training=training, aug_cfg=aug_cfg, seed=seed,
                         positions=positions, records=[], allow_test=allow_test)
        self.legacy = WindowDataset(manifest, split, task='seg', recent_frames=recent_frames,
                                    local_frames=local_frames, max_prefix=0, prefix=0,
                                    training=training, aug_cfg=aug_cfg, seed=seed,
                                    positions=positions, allow_test=allow_test)
        legacy_by_frame = {}
        for i, r in enumerate(self.legacy.records):
            legacy_by_frame.setdefault((r['patient'], r['target_frame']), []).append((i, r))
        self.records = []
        for case in self.cases.values():
            for trace in case['traces']:
                matches = legacy_by_frame.get((case['patient'], trace['frame']), [])
                if matches:
                    self.records.extend(dict(r, legacy_index=i) for i, r in matches)
                else:
                    self.records.append(dict(patient=case['patient'], target_frame=trace['frame'],
                                             target_position='causal_boundary'))
        self.excluded = []

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.legacy.set_epoch(epoch)

    def __getitem__(self, index):
        record = self.records[index]
        if 'legacy_index' in record:
            sample = self.legacy[record['legacy_index']]
            sample['coverage_kind'] = 'legacy_complete_context'
            return sample
        case, frame = self.cases[record['patient']], record['target_frame']
        length = min(self.recent_frames, max(self.local_frames,
                     math.ceil((frame+1)/self.local_frames)*self.local_frames))
        start = frame-length+1
        indices = np.maximum(np.arange(start, frame+1, dtype=np.int64), 0)
        before = _file_provenance(Path(case['path']))
        if any(before[k] != case[k] for k in ('source_bytes', 'source_mtime_ns')):
            raise ValueError('Video changed after coverage manifest creation')
        raw = read_echo_input(Path(case['path']), 'gray_repeat3')
        try:
            if len(raw) != case['frames'] or tuple(raw.shape[1:3]) != (112, 112):
                raise ValueError('Video shape changed')
            clip = np.asarray(raw[indices]).copy()
        finally:
            _close(raw)
        if _file_provenance(Path(case['path'])) != before:
            raise ValueError('Video changed during reading')
        trace = next(t for t in case['traces'] if t['frame'] == frame)
        aug_seed = int(_rng(self.seed, 'coverage_augmentation', index,
                           self.epoch.value if self.training else 0).integers(2**32))
        video, mask = _prepare_video(clip, _trace_mask(trace), self.aug_cfg, aug_seed)
        return dict(video=video, mask=mask, frame_indices=torch.from_numpy(indices),
                    patient=case['patient'], id=case['patient'], prefix_frames=0,
                    target_index=length-1, target_position=length-1,
                    target_recent_index=length-1, target_frame=frame, fps=case['fps'],
                    source_path=case['path'], recent_start=start, record_index=index,
                    full_context=False, input_protocol='gray_repeat3',
                    coverage_kind='short_causal_context',
                    left_padding_frames=max(0, -start))

    def calibration_samples(self, batch_size):
        candidates = [i for i, r in enumerate(self.records) if 'legacy_index' in r]
        sample = self[candidates[0] if candidates else 0]
        return [sample] * batch_size


class DrawBudgetDataset(torch.utils.data.Dataset):
    """Equal updates/check interval; visit the entire pool before starting a new permutation."""
    shuffle_draws = False
    def __init__(self, dataset, draws):
        if draws < 1 or not len(dataset):
            raise ValueError('Nonempty draw budget and supervision pool required')
        self.dataset, self.draws = dataset, draws
        self.epoch = dataset.epoch
        self.records, self.cases = dataset.records, dataset.cases
        self.excluded = dataset.excluded
        self._cycle, self._order = None, None

    def __len__(self):
        return self.draws

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        if not 0 <= index < self.draws:
            raise IndexError(index)
        cycle, offset = divmod(int(self.epoch.value)*self.draws+index, len(self.dataset))
        if self._cycle != cycle:
            self._order = np.random.default_rng(self.dataset.seed+cycle*1000003).permutation(len(self.dataset))
            self._cycle = cycle
        return self.dataset[int(self._order[offset])]

    def calibration_samples(self, batch_size):
        return self.dataset.calibration_samples(batch_size)


def evaluation_dataset(cfg, manifest, split, task):
    if task != 'seg':
        raise ValueError('Coverage repair is a segmentation-only intervention')
    return CoverageDataset(manifest, split, seed=cfg['data'].get('seed', 42),
                           allow_test=split == 'test', positions='all')


def build(config, stage, source, device):
    mode = stage.get('mode')
    if stage['kind'] != 'seg' or mode not in {'finetune', 'frozen_linear'}:
        raise ValueError('Coverage repair requires segmentation FT or a frozen linear probe')
    data = config['data']
    # Limit records after eligibility, not the first N videos before trace screening.
    limits = {key:data[key] for key in ('limit_train','limit_val') if data.get(key) is not None}
    reference = dict(config, data={key:value for key,value in data.items() if key not in limits})
    exp = build_spatial(reference, stage, source, device)
    if 'limit_val' in limits:
        records = exp.val.dataset.records
        exp.val.dataset.records = records[:limits['limit_val']]
        exp.metadata.update(val_ids=sorted({r['patient'] for r in exp.val.dataset.records}),
                            val_records_sha256=digest(exp.val.dataset.records))
    finetune = mode == 'finetune'
    dataset = CoverageDataset(data['coverage_manifest'], 'train', training=finetune,
                              aug_cfg=data.get('augmentation') if finetune else None, seed=data.get('seed', 42),
                              positions='balanced')
    if data.get('limit_train') is not None:
        dataset.records = dataset.records[:data['limit_train']]
    draws = min(len(exp.train),limits.get('limit_train',len(exp.train))) if finetune else len(dataset)
    exp.train = IdentifiedDataset(DrawBudgetDataset(dataset, draws) if finetune else dataset, 'train')
    exp.metadata['data_files'][str(data['coverage_manifest'])] = file_hash(data['coverage_manifest'])
    exp.metadata.update(train_ids=sorted({r['patient'] for r in dataset.records}),
                        train_records_sha256=digest(dataset.records),
                        coverage_manifest=str(data['coverage_manifest']),
                        coverage=dict(policy='legacy positions unchanged; recovered frames use causal short context',
                                      unique_source_frames=len({(r['patient'], r['target_frame']) for r in dataset.records}),
                                      supervised_pool=len(dataset), draws_per_check=draws,
                                      budget='same samples/optimizer-updates per validation check as legacy' if finetune
                                             else 'one unaugmented window per available annotated frame per epoch',
                                      boundary='repeat first source frame only to complete earliest local clip',
                                      validation='unchanged common legacy cohort for selection'))
    return exp
