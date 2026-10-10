"""Real EchoNet windows for the final Q1-Q3 v2 protocol (no frame padding).

``start`` always names the first frame of the recent window, not the prefix.
Segmentation target_index includes the prefix; target_position does not.
EF targets remain percentage points. MAE samples have no supervised target.
Source hashes cover metadata and stat provenance, not entire video contents.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from augment.ultrasound import EchoAugmentConfig, resize_with_pad
from echo_aug_validation.augment_recipes import _zoom_pair, augment_video
from echo_aug_validation.io_utils import find_echonet_video
from .augmentation import build_echo_augment_config
from .downstream_datasets import _rasterize_echonet_trace
from .echo_input import read_echo_input


_POINT_COLUMNS = ['X1', 'Y1', 'X2', 'Y2']


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def _rng(seed, *key):
    return np.random.default_rng(int(_hash([int(seed), *key])[:16], 16))


def _file_provenance(path, content=False):
    stat = path.stat()
    record = dict(path=str(path.resolve()), source_bytes=stat.st_size,
                  source_mtime_ns=stat.st_mtime_ns)
    if content:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        record['sha256'] = digest.hexdigest()
    return record


def _dimensions(recent_frames, local_frames, max_prefix):
    if (any(int(v) != v for v in (recent_frames, local_frames, max_prefix))
            or local_frames < 1 or recent_frames < local_frames or recent_frames % local_frames
            or max_prefix < 0 or max_prefix % local_frames):
        raise ValueError('Require complete local clips, recent >= local, and a nonnegative clip-multiple prefix')


def _close(raw):
    if isinstance(raw, np.memmap):
        raw._mmap.close()


def _trace_mask(trace):
    points = np.asarray(trace['points'], dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 4 or len(points) < 2:
        raise ValueError('A trace requires at least two [x1,y1,x2,y2] rows')
    if not np.isfinite(points).all():
        raise ValueError('EchoNet GT coordinates must be finite')
    # Official traces can cross the canvas border. Let the official polygon
    # rasterizer clip them; never clamp vertices or rescale cached GT.
    return _rasterize_echonet_trace(pd.DataFrame(points, columns=_POINT_COLUMNS), (112, 112))


def _complete_trace(case, frame, recent, local, prefix=0):
    # The latest target position needs the most past; the earliest needs the most future.
    return frame - (recent - 1) >= prefix and frame + local <= case['frames']


def build_manifest(root, seed=42, recent_frames=64, local_frames=16,
                   max_prefix=128, smoke=False):
    """Audit TRAIN/VAL; smoke caps each split at eight eligible real videos.

    Short videos are excluded from W64, not by H128 availability. All official
    traces are kept on eligible cases; task-level warm-up exclusions are listed
    separately. TEST video files are never opened. Duplicate IDs are fatal.
    """
    _dimensions(recent_frames, local_frames, max_prefix)
    root = Path(root)
    filelist, tracefile = root / 'FileList.csv', root / 'VolumeTracings.csv'
    metadata = dict(filelist=_file_provenance(filelist, True),
                    traces=_file_provenance(tracefile, True))
    files = pd.read_csv(filelist)
    required = {'FileName', 'Split', 'FPS', 'EF'}
    if not required.issubset(files.columns):
        raise ValueError(f'FileList.csv requires {sorted(required)}')
    files['_patient'] = files.FileName.map(lambda x: Path(str(x)).stem)
    if files['_patient'].duplicated().any():
        raise ValueError('Duplicate patient IDs / split overlap in FileList.csv')
    files['_split'] = files.Split.astype(str).str.strip().str.lower()
    if not set(files['_split']).issubset({'train', 'val', 'test'}):
        raise ValueError('Require official TRAIN/VAL/TEST splits')
    traces = pd.read_csv(tracefile)
    if not {'FileName', 'Frame', *_POINT_COLUMNS}.issubset(traces.columns):
        raise ValueError('VolumeTracings.csv is missing official trace columns')
    traces['_patient'] = traces.FileName.map(lambda x: Path(str(x)).stem)
    selected_ids = set(files.loc[files['_split'].isin(['train', 'val']), '_patient'])
    labels = defaultdict(list)
    for (patient, frame), group in traces[traces['_patient'].isin(selected_ids)].groupby(
            ['_patient', 'Frame'], sort=True):
        if not np.isfinite(frame) or int(frame) != frame or frame < 0:
            raise ValueError(f'{patient}: invalid original trace frame {frame}')
        trace = dict(frame=int(frame), points=group[_POINT_COLUMNS].to_numpy(dtype=float).tolist())
        if not _trace_mask(trace).any():
            raise ValueError(f'{patient}: empty official mask at frame {frame}')
        labels[patient].append(trace)

    result = dict(version=2, seed=int(seed), root=str(root.resolve()),
                  recent_frames=int(recent_frames), local_frames=int(local_frames),
                  max_prefix=int(max_prefix), smoke=bool(smoke), train=[], val=[], excluded=[])
    counts = {}
    for split in ('train', 'val'):
        rows = files[files['_split'] == split].to_dict('records')
        rows.sort(key=lambda row: _hash([int(seed), split, row['_patient']]))
        reasons = Counter()
        warmup = 0
        dimension_metadata_mismatches = 0
        for row in rows:
            patient = row['_patient']
            if smoke and len(result[split]) >= 8:
                reasons['smoke_budget'] += 1
                result['excluded'].append(dict(split=split, patient=patient, reason='smoke_budget'))
                continue
            path = find_echonet_video(root, str(row['FileName']))
            reason = None
            if path is None:
                reason = 'missing_video'
            elif not np.isfinite(float(row['FPS'])) or float(row['FPS']) <= 0:
                reason = 'missing_positive_fps'
            if reason:
                reasons[reason] += 1
                result['excluded'].append(dict(split=split, patient=patient, reason=reason))
                continue
            provenance = _file_provenance(path)
            declared_shape = [float(row[column]) if pd.notna(row.get(column)) else None
                              for column in ('FrameHeight', 'FrameWidth')]
            if path.suffix.lower() == '.npy':
                raw = np.load(path, mmap_mode='r', allow_pickle=False)
            else:
                raw = read_echo_input(path, 'gray_repeat3')
            try:
                shape, dtype = list(raw.shape), str(raw.dtype)
                if (raw.ndim not in (3, 4) or (raw.ndim == 4 and raw.shape[-1] not in (1, 3))
                        or tuple(raw.shape[1:3]) != (112, 112)):
                    raise ValueError(f'{path}: require [T,112,112] or [T,112,112,1/3]; GT coordinate mismatch: {shape}')
                if raw.dtype != np.uint8 and not np.issubdtype(raw.dtype, np.floating):
                    raise ValueError(f'{path}: require uint8 or floating-point pixels, got {dtype}')
                n = len(raw)
            finally:
                _close(raw)
            if _file_provenance(path) != provenance:
                raise ValueError(f'Data changed while building manifest: {path}')
            if any(t['frame'] >= n for t in labels[patient]):
                raise ValueError(f'{patient}: trace frame outside actual video/cache length {n}')
            case = dict(patient=patient, frames=n, fps=float(row['FPS']),
                        ef=float(row['EF']) if np.isfinite(float(row['EF'])) else None,
                        traces=labels[patient], shape=shape, dtype=dtype, **provenance)
            case['declared_spatial_shape'] = declared_shape
            case['dimension_metadata_mismatch'] = any(value is not None and value != 112 for value in declared_shape)
            dimension_metadata_mismatches += int(case['dimension_metadata_mismatch'])
            case['source_fingerprint'] = _hash(provenance)
            case['declared_frames'] = (int(row['NumberOfFrames'])
                                       if pd.notna(row.get('NumberOfFrames')) else None)
            if n < recent_frames:
                reasons['short_recent'] += 1
                result['excluded'].append(dict(split=split, reason='short_recent', **case))
                continue
            result[split].append(case)
            for trace in case['traces']:
                if not _complete_trace(case, trace['frame'], recent_frames, local_frames):
                    warmup += 1
                    result['excluded'].append(dict(split=split, patient=patient, task='seg',
                                                   frame=trace['frame'], reason='seg_not_complete_all_positions'))
        counts[split] = dict(filelist_cases=len(rows), eligible_cases=len(result[split]),
                             dimension_metadata_mismatches=dimension_metadata_mismatches,
                             short_recent_cases=reasons['short_recent'],
                             warmup_excluded_traces=warmup, exclusion_counts=dict(reasons))
    result['counts'] = counts
    for name, path in (('filelist', filelist), ('traces', tracefile)):
        if _file_provenance(path) != {k: v for k, v in metadata[name].items() if k != 'sha256'}:
            raise ValueError(f'Metadata changed while building manifest: {path}')
    result['provenance'] = dict(**metadata,
                                source_hash_policy='SHA256 metadata contents; video stat/shape/dtype only',
                                input_protocol='gray_repeat3', gt_shape=[112, 112],
                                dimension_policy='Actual pixel shape must be 112x112; CSV dimensions are recorded, never used to rescale GT',
                                rasterizer='utils.downstream_datasets._rasterize_echonet_trace')
    result['manifest_sha256'] = _hash(result)
    return result


def _load_manifest(manifest):
    if isinstance(manifest, (str, Path)):
        with Path(manifest).open(encoding='utf-8') as handle:
            manifest = json.load(handle)
    if manifest.get('version') != 2:
        raise ValueError('Expected final temporal manifest version 2')
    digest = manifest.get('manifest_sha256')
    if digest and _hash({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != digest:
        raise ValueError('Manifest hash mismatch')
    train = [r['patient'] for r in manifest['train']]
    val = [r['patient'] for r in manifest['val']]
    if len(set(train + val)) != len(train + val):
        raise ValueError('Duplicate patients or train/val overlap')
    return manifest


def _normalized_video(raw):
    arr = np.asarray(raw).astype(np.float32)
    if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 255:
        raise ValueError('Pixels must be finite uint8 / float [0,1] / float [0,255]')
    if raw.dtype == np.uint8 or arr.max() > 1:
        arr /= 255.0
    return arr


def _prepare_video(raw, mask, aug_cfg, seed):
    clip = resize_with_pad(_normalized_video(raw), 112)
    if aug_cfg is not None:
        rng = np.random.default_rng(seed)
        if aug_cfg.zoom_prob > 0 and rng.random() < aug_cfg.zoom_prob:
            scale = float(rng.uniform(aug_cfg.zoom_min, aug_cfg.zoom_max))
            placeholder = np.zeros((112, 112), dtype=np.uint8) if mask is None else mask
            frames = []
            for image in clip:
                image, transformed = _zoom_pair(image, placeholder, scale)
                frames.append(image)
            clip = np.stack(frames)
            if mask is not None:
                mask = transformed
            seed += 17
        # Geometry is sampled once above; all photometric parameters are clip-consistent.
        clip = augment_video(clip, replace(aug_cfg, zoom_prob=0.0, preserve_dtype=False), seed, False)
    video = torch.from_numpy(np.clip(clip[..., 0], 0, 1).copy())[:, None].repeat(1, 3, 1, 1)
    return video, None if mask is None else torch.from_numpy(mask.astype(np.int64))


class WindowDataset(Dataset):
    """W64 task windows, optionally preceded by real H clips.

    prefix=None draws H uniformly from available multiples, prefix=0 disables
    history. ``all`` expands every retained trace across the last local_frames
    positions. ``balanced`` visits each position once per local_frames epochs.
    limit counts cases before trace/position expansion (records in explicit
    mode). set_epoch is shared with
    persistent workers. Optional records fix patient/recent_start/H (and
    target_frame for seg); prefix_kind='repeat_prefix' labels HRh explicitly,
    repeating the first real prefix clip (the existing stage3 convention),
    still requiring genuine H availability in that same-window cohort.
    """

    def __init__(self, manifest, split, task='ef', recent_frames=64, local_frames=16,
                 max_prefix=128, seed=42, training=False, positions='balanced',
                 prefix=0, aug_cfg=None, limit=None, *, records=None):
        _dimensions(recent_frames, local_frames, max_prefix)
        if task not in ('ef', 'seg', 'mae') or split not in ('train', 'val'):
            raise ValueError('Require task ef/seg/mae and split train/val; TEST is sealed')
        if positions not in ('balanced', 'all'):
            raise ValueError('positions must be balanced or all')
        if prefix is not None and (int(prefix) != prefix or prefix < 0 or prefix > max_prefix or prefix % local_frames):
            raise ValueError('prefix must be None or an available clip multiple <= max_prefix')
        if limit is not None and (int(limit) != limit or limit < 0):
            raise ValueError('limit must be nonnegative')
        self.manifest = _load_manifest(manifest)
        self.recent_frames, self.local_frames, self.max_prefix = recent_frames, local_frames, max_prefix
        self.seed, self.training, self.task = int(seed), bool(training), task
        self.positions, self.prefix = positions, prefix
        self.epoch = multiprocessing.Value('q', 0)
        self.aug_cfg = None
        if training and aug_cfg is not None:
            if isinstance(aug_cfg, EchoAugmentConfig):
                self.aug_cfg = replace(aug_cfg, preserve_dtype=False)
            elif aug_cfg.get('enabled', True):
                self.aug_cfg, per_frame, _ = build_echo_augment_config(aug_cfg, 112)
                if per_frame:
                    raise ValueError('Final temporal augmentation must be clip-consistent')
            if self.aug_cfg is not None and self.aug_cfg.img_size != 112:
                raise ValueError('EchoNet GT requires img_size=112')
        self.cases = {r['patient']: r for r in self.manifest[split]}
        self.excluded, self.records = [], []
        if records is not None:
            self.records = [dict(r) for r in records]
            for r in self.records:
                if 'recent_start' in r and 'start' in r and r['recent_start'] != r['start']:
                    raise ValueError('Conflicting explicit recent_start/start')
                r['start'] = r.get('recent_start', r.get('start'))
                if r['start'] is None:
                    raise ValueError('Explicit record requires recent_start')
                r['prefix_kind'] = r.get('prefix_kind', 'clean')
                self._validate_record(r)
            if limit is not None:
                self.records = self.records[:limit]
            self.explicit_records = True
            return
        self.explicit_records = False
        eligible = []
        for case in self.manifest[split]:
            if task == 'ef' and case['ef'] is None:
                self.excluded.append(dict(patient=case['patient'], reason='missing_ef'))
                continue
            if case['frames'] < recent_frames + (prefix or 0):
                self.excluded.append(dict(patient=case['patient'], reason='short_real_context'))
                continue
            eligible.append(case)
        if limit is not None:
            eligible = eligible[:limit]
        for case in eligible:
            if task != 'seg':
                self.records.append(dict(patient=case['patient']))
                continue
            for trace in case['traces']:
                frame = trace['frame']
                if not _complete_trace(case, frame, recent_frames, local_frames, prefix or 0):
                    self.excluded.append(dict(patient=case['patient'], frame=frame,
                                             reason='seg_not_complete_all_positions'))
                    continue
                targets = range(recent_frames - local_frames, recent_frames) if positions == 'all' else [None]
                for position in targets:
                    self.records.append(dict(patient=case['patient'], target_frame=frame,
                                             target_position=position))

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        if epoch < 0:
            raise ValueError('epoch must be nonnegative')
        self.epoch.value = int(epoch)

    def _validate_record(self, record):
        if record['patient'] not in self.cases:
            raise ValueError('Explicit record is not in the requested split')
        case = self.cases[record['patient']]
        start, h = record['start'], record.get('H', record.get('prefix_frames', 0))
        if 'H' in record and 'prefix_frames' in record and record['H'] != record['prefix_frames']:
            raise ValueError('Conflicting explicit H/prefix_frames')
        if (int(start) != start or int(h) != h or h < 0 or h > self.max_prefix
                or h % self.local_frames or start < h or start + self.recent_frames > case['frames']):
            raise ValueError('Explicit record lacks complete real history/window')
        if record.get('prefix_kind', 'clean') not in ('clean', 'repeat_prefix'):
            raise ValueError('Unknown explicit history intervention')
        if record.get('prefix_kind') == 'repeat_prefix' and h < self.local_frames:
            raise ValueError('repeat_prefix requires at least one real prefix clip')
        if self.task == 'ef' and case['ef'] is None:
            raise ValueError('Explicit EF record requires a real EF label')
        if self.task == 'seg':
            frame = record['target_frame']
            position = frame - start
            if not self.recent_frames - self.local_frames <= position < self.recent_frames:
                raise ValueError('Segmentation target must be in the last local clip')
            if not _complete_trace(case, frame, self.recent_frames, self.local_frames, h):
                raise ValueError('Segmentation label is not complete at ALL target positions')
            if not any(t['frame'] == frame for t in case['traces']):
                raise ValueError('Segmentation target must be an original annotated frame')

    def get_record(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        base = self.records[index]
        case = self.cases[base['patient']]
        if self.explicit_records:
            self._validate_record(base)
            h = int(base.get('H', base.get('prefix_frames', 0)))
            record = dict(case, **base)
            record.update(H=h, prefix_frames=h, recent_start=record['start'])
            if self.task == 'seg':
                record['target_position'] = record['target_frame'] - record['start']
            return record
        epoch = self.epoch.value if self.training else 0
        rng = _rng(self.seed, self.task, base['patient'], index, epoch)
        if self.task == 'seg':
            position = base['target_position']
            if position is None:
                offset = int(_rng(self.seed, 'positions').integers(self.local_frames))
                position = self.recent_frames - self.local_frames + (index + offset + epoch) % self.local_frames
            start = base['target_frame'] - position
        else:
            start = int(rng.integers(self.prefix or 0, case['frames'] - self.recent_frames + 1))
            position = -1
        h = self.prefix
        if h is None:
            h = int(rng.integers(min(self.max_prefix, start) // self.local_frames + 1)) * self.local_frames
        record = dict(case, **base)
        record.update(start=start, recent_start=start, H=h, prefix_frames=h,
                      target_position=position, prefix_kind='clean')
        return record

    def _sample_record(self, record, index):
        case = self.cases[record['patient']]
        path = Path(case['path'])
        provenance = _file_provenance(path)
        if any(provenance[k] != case[k] for k in ('source_bytes', 'source_mtime_ns')):
            raise ValueError(f'Data changed after manifest creation: {path}')
        h, start = record['H'], record['start']
        indices = np.arange(start - h, start + self.recent_frames, dtype=np.int64)
        history = record.get('prefix_kind', 'clean')
        if history == 'repeat_prefix' and h:
            indices[:h] = start - h + np.arange(h) % self.local_frames
        raw = read_echo_input(path, 'gray_repeat3')
        try:
            if len(raw) != case['frames'] or tuple(raw.shape[1:3]) != (112, 112):
                raise ValueError('Video/cache shape changed; original GT coordinates required')
            clip = np.asarray(raw[indices]).copy()
        finally:
            _close(raw)
        if _file_provenance(path) != provenance:
            raise ValueError(f'Data changed while reading: {path}')
        mask = None
        position = record.get('target_position', -1)
        if self.task == 'seg':
            trace = next(t for t in case['traces'] if t['frame'] == record['target_frame'])
            mask = _trace_mask(trace)
        epoch = self.epoch.value if self.training else 0
        aug_seed = int(_rng(self.seed, 'augmentation', index, epoch).integers(2**32))
        video, mask = _prepare_video(clip, mask, self.aug_cfg, aug_seed)
        sample = dict(video=video, frame_indices=torch.from_numpy(indices), patient=case['patient'],
                      id=case['patient'], prefix_frames=h, target_index=h + position if position >= 0 else -1,
                      target_position=position, target_recent_index=position,
                      fps=case['fps'], source_path=case['path'],
                      full_context=True, recent_start=start, input_protocol='gray_repeat3',
                      history_intervention=history, prefix_kind=history, record_index=index)
        if self.task == 'ef':
            sample['target'] = torch.tensor(case['ef'], dtype=torch.float32)
        elif self.task == 'seg':
            sample.update(mask=mask, target_frame=record['target_frame'])
        return sample

    def __getitem__(self, index):
        return self._sample_record(self.get_record(index), index)


class WarmPlanDataset(WindowDataset):
    """Index-keyed MAE draws, independent of micro-batch size or resume cursor."""

    def __init__(self, manifest, total_updates=1500, effective_batch=32,
                 recent_frames=64, local_frames=16, max_prefix=128, seed=42):
        if (int(total_updates) != total_updates or int(effective_batch) != effective_batch
                or total_updates < 1 or effective_batch < 1):
            raise ValueError('total_updates and effective_batch must be positive')
        super().__init__(manifest, 'train', task='mae', recent_frames=recent_frames,
                         local_frames=local_frames, max_prefix=max_prefix, seed=seed, prefix=None)
        self.total_updates, self.effective_batch = int(total_updates), int(effective_batch)
        self.eligible_cases = [self.cases[r['patient']] for r in self.records]
        if not self.eligible_cases:
            raise ValueError('No real recent windows available for MAE adaptation')

    def __len__(self):
        return self.total_updates * self.effective_batch

    def get_record(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        rng = _rng(self.seed, 'warm-plan', int(index))
        case = self.eligible_cases[int(rng.integers(len(self.eligible_cases)))]
        start = int(rng.integers(case['frames'] - self.recent_frames + 1))
        h = int(rng.integers(min(self.max_prefix, start) // self.local_frames + 1)) * self.local_frames
        return dict(case, start=start, recent_start=start, H=h, prefix_frames=h,
                    prefix_kind='clean', target_position=-1,
                    update_index=index // self.effective_batch,
                    sample_index=index % self.effective_batch, record_index=index)

    def __getitem__(self, index):
        record = self.get_record(index)
        sample = self._sample_record(record, index)
        sample['update_index'] = record['update_index']
        sample['sample_index'] = record['sample_index']
        return sample


class WarmBatchSampler(Sampler):
    """Group each optimizer update by H, then split without padding/dropping.

    start_update is the next *successful optimizer update*, not a micro-batch
    cursor. Partial updates must be replayed on resume. Ragged micro-batches
    require loss weighting by their true size / effective_batch in the caller.
    """

    def __init__(self, dataset, micro_batch, start_update=0):
        if int(micro_batch) != micro_batch or micro_batch < 1:
            raise ValueError('micro_batch must be a positive integer')
        if int(start_update) != start_update or not 0 <= start_update <= dataset.total_updates:
            raise ValueError('start_update must be a valid optimizer-update cursor')
        self.dataset, self.micro_batch, self.start_update = dataset, int(micro_batch), int(start_update)

    def __iter__(self):
        for update in range(self.start_update, self.dataset.total_updates):
            groups = defaultdict(list)
            first = update * self.dataset.effective_batch
            for index in range(first, first + self.dataset.effective_batch):
                groups[self.dataset.get_record(index)['H']].append(index)
            for h in sorted(groups):
                indices = groups[h]
                for offset in range(0, len(indices), self.micro_batch):
                    yield indices[offset:offset + self.micro_batch]

    def __len__(self):
        return sum(1 for _ in self)
