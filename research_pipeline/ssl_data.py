"""Label-free, all-split EchoNet SSL inventory and reproducible online windows."""

from __future__ import annotations

from collections import Counter
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import digest, file_hash
from utils.final_temporal_data import _dimensions, _file_provenance, _prepare_video, _rng
from utils.augmentation import build_echo_augment_config


EXPOSURE = 'transductive_all_splits_unlabeled'


def build_ssl_manifest(root, local_frames=16, recent_frames=64, max_prefix=128, seed=42):
    """Read only non-label CSV columns and NPY headers; never open tracings."""
    _dimensions(recent_frames, local_frames, max_prefix)
    root = Path(root).resolve()
    source = root / 'FileList.csv'
    before = _file_provenance(source)
    files = pd.read_csv(source, usecols=lambda name: name in {'FileName', 'Split', 'FPS'})
    if not {'FileName', 'Split'}.issubset(files.columns):
        raise ValueError('SSL inventory requires FileName and official Split, not EF or traces')
    result = dict(format='echo_ssl_unlabeled_v1', exposure=EXPOSURE, root=str(root), seed=int(seed),
                  local_frames=local_frames, recent_frames=recent_frames, max_prefix=max_prefix,
                  videos=[], excluded=[], label_columns_loaded=[], tracing_file_opened=False,
                  sampling='one new real consecutive window per eligible video per epoch; no padded frames')
    patients = set()
    for row in files.to_dict('records'):
        patient = Path(str(row['FileName'])).stem
        split = str(row['Split']).strip().lower()
        if patient in patients or split not in {'train', 'val', 'test'}:
            raise ValueError('Duplicate IDs or unknown official split in SSL inventory')
        patients.add(patient)
        path = root / 'npy' / f'{patient}.npy'
        if not path.is_file():
            # Missing caches are actionable, not silently omitted from "all data".
            raise FileNotFoundError(f'{path}: prepare the existing grayscale NPY cache first')
        provenance = _file_provenance(path)
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            shape, dtype = list(raw.shape), str(raw.dtype)
            if raw.ndim not in (3, 4) or tuple(raw.shape[1:3]) != (112, 112) or (raw.ndim == 4 and raw.shape[-1] != 1):
                raise ValueError(f'{path}: clean SSL requires native grayscale [T,112,112,(1)], got {shape}')
            if raw.dtype != np.uint8:
                raise ValueError(f'{path}: require the existing uint8 grayscale cache, got {dtype}')
            frames = len(raw)
        finally:
            raw._mmap.close()
        if _file_provenance(path) != provenance:
            raise ValueError(f'NPY changed during inventory: {path}')
        if frames < local_frames:
            result['excluded'].append(dict(patient=patient, split=split, frames=frames, reason='shorter_than_one_real_clip'))
            continue
        fps = row.get('FPS')
        fps = float(fps) if pd.notna(fps) and np.isfinite(float(fps)) and float(fps) > 0 else None
        result['videos'].append(dict(patient=patient, split=split, frames=frames, fps=fps,
                                     shape=shape, dtype=dtype, **provenance))
    if not result['videos']:
        raise ValueError('No complete real clips in SSL inventory')
    result['videos'].sort(key=lambda case: digest([seed, case['patient']]))
    result['counts'] = dict(eligible_by_official_split=dict(Counter(c['split'] for c in result['videos'])),
                            eligible_videos=len(result['videos']), excluded_videos=len(result['excluded']))
    if _file_provenance(source) != before:
        raise ValueError('FileList changed during inventory')
    result['filelist_sha256'] = file_hash(source)
    result['hash_policy'] = 'CSV content SHA256 (not stored labels); NPY size/mtime/shape/dtype, not content SHA256'
    result['manifest_sha256'] = digest(result)
    return result


def load_ssl_manifest(path):
    value = json.loads(Path(path).read_text(encoding='utf-8')) if isinstance(path, (str, Path)) else path
    if value.get('format') != 'echo_ssl_unlabeled_v1' or value.get('exposure') != EXPOSURE:
        raise ValueError('Expected explicit all-split unlabeled SSL protocol')
    if digest({k:v for k,v in value.items() if k != 'manifest_sha256'}) != value.get('manifest_sha256'):
        raise ValueError('SSL manifest hash mismatch')
    if value.get('label_columns_loaded') != [] or value.get('tracing_file_opened') is not False:
        raise ValueError('SSL inventory must not contain supervision')
    allowed = {'patient', 'split', 'frames', 'fps', 'shape', 'dtype', 'path', 'source_bytes', 'source_mtime_ns'}
    ids = [case['patient'] for case in value['videos']]
    if len(set(ids)) != len(ids) or any(set(case)-allowed for case in value['videos']):
        raise ValueError('Duplicate videos or label/unrecognized fields in SSL inventory')
    return value


class SSLWindowDataset(torch.utils.data.Dataset):
    """One deterministic seed/epoch/video draw, backed by mmap, not saved windows."""
    def __init__(self, manifest, training, seed=42, diagnostic_cases=64, limit=None, augmentation=None):
        self.manifest = load_ssl_manifest(manifest)
        self.training, self.seed = bool(training), int(seed)
        self.local_frames = int(self.manifest['local_frames'])
        self.recent_frames = int(self.manifest['recent_frames'])
        self.max_prefix = int(self.manifest['max_prefix'])
        self.epoch = multiprocessing.Value('q', 0)
        cases = self.manifest['videos']
        if not training:
            if diagnostic_cases < 1:
                raise ValueError('Diagnostic population must be nonempty')
            cases = cases[:diagnostic_cases]
        if limit is not None:
            cases = cases[:limit]
        self.records = [dict(case) for case in cases]
        self.excluded = self.manifest['excluded']
        self.aug_cfg = None
        if training and augmentation and augmentation.get('enabled', True):
            self.aug_cfg, per_frame, _ = build_echo_augment_config(augmentation, 112)
            if per_frame:
                raise ValueError('SSL augmentation must be consistent across the complete observed window')

    def __len__(self):
        return len(self.records)

    def set_epoch(self, epoch):
        self.epoch.value = int(epoch)

    def __getitem__(self, index):
        case = self.records[index]
        rng = _rng(self.seed, 'ssl_window', case['patient'], self.epoch.value if self.training else 0)
        local = self.local_frames
        recent = min(self.recent_frames, case['frames']//local*local)
        start = int(rng.integers(case['frames']-recent+1))
        prefix = int(rng.integers(min(self.max_prefix, start)//local+1))*local
        return self._sample(index, start, recent, prefix)

    def _sample(self, index, start, recent, prefix):
        case = self.records[index]
        path = Path(case['path'])
        provenance = _file_provenance(path)
        if any(provenance[k] != case[k] for k in ('source_bytes', 'source_mtime_ns')):
            raise ValueError(f'Video changed after SSL inventory: {path}')
        indices = np.arange(start-prefix, start+recent, dtype=np.int64)
        raw = np.load(path, mmap_mode='r', allow_pickle=False)
        try:
            if list(raw.shape) != case['shape'] or str(raw.dtype) != case['dtype']:
                raise ValueError('NPY shape/dtype changed')
            clip = np.array(raw[indices], copy=True)
        finally:
            raw._mmap.close()
        if _file_provenance(path) != provenance:
            raise ValueError('NPY changed during window read')
        epoch = self.epoch.value if self.training else 0
        aug_seed = int(_rng(self.seed, 'ssl_aug', index, epoch).integers(2**32))
        video, _ = _prepare_video(clip, None, self.aug_cfg, aug_seed)
        return dict(video=video, frame_indices=torch.from_numpy(indices), patient=case['patient'],
                    prefix_frames=prefix, recent_start=start, recent_frames=recent, target_index=-1,
                    fps=case['fps'], source_path=case['path'], record_index=index,
                    input_protocol='gray_repeat3', data_exposure=EXPOSURE)

    def calibration_samples(self, batch_size):
        # Probe the longest admissible real context, not a lucky short draw.
        local = self.local_frames
        index = max(range(len(self)), key=lambda i: min(self.records[i]['frames']//local*local,
                                                       self.recent_frames+self.max_prefix))
        frames = self.records[index]['frames']
        recent = min(self.recent_frames, frames//local*local)
        prefix = min(self.max_prefix, (frames-recent)//local*local)
        return [self._sample(index, prefix, recent, prefix)]*batch_size
