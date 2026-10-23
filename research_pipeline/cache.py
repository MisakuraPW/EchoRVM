"""Bounded disk-backed frozen features, with immutable input/encoder identity."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .artifacts import digest, file_hash, write_json


class FeatureDataset(Dataset):
    def __init__(self, directory):
        self.directory = Path(directory)
        self.records = [json.loads(line) for line in (self.directory / 'records.jsonl').read_text(encoding='utf-8').splitlines()]
        self._features = self._targets = None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        if self._features is None:
            self._features = np.load(self.directory / 'features.npy', mmap_mode='r')
            self._targets = np.load(self.directory / 'targets.npy', mmap_mode='r')
        record = dict(self.records[index])
        record['features'] = torch.from_numpy(np.array(self._features[index], copy=True))
        record['mask' if record.pop('_target_key') == 'mask' else 'target'] = torch.from_numpy(np.array(self._targets[index], copy=True))
        return record

    def __getstate__(self):
        return dict(self.__dict__, _features=None, _targets=None)

    def close(self):
        for value in (getattr(self, '_features', None), getattr(self, '_targets', None)):
            mapping = getattr(value, '_mmap', None)
            if mapping is not None:
                mapping.close()
        self._features = self._targets = None

    def __del__(self):
        self.close()


def _metadata(sample):
    value = {}
    for key, item in sample.items():
        if key in {'video', 'mask', 'target', 'features', 'frame_indices'}:
            continue
        if torch.is_tensor(item):
            item = item.tolist()
        if isinstance(item, np.generic):
            item = item.item()
        value[key] = item
    return value


def prepare_cache(experiment, directory, identity, budget_bytes, min_free_bytes, heartbeat=None, batch_size=4, budget_root=None):
    if not experiment.cacheable:
        raise ValueError('Frozen caching requires an adapter declaring deterministic unaugmented features')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if budget_root is not None:
        used = sum(p.stat().st_size for p in Path(budget_root).rglob('*') if p.is_file())
        if used > budget_bytes:
            raise OSError('Existing run cache exceeds cache_gb; explicitly clean unneeded cache or increase its budget')
    required_total = 0
    for split in ('train', 'val'):
        dataset = getattr(experiment, split)
        destination = directory / split
        destination.mkdir(parents=True, exist_ok=True)
        key = digest(dict(identity=identity, split=split, samples=len(dataset), precision='fp32'))
        done = destination / 'DONE.json'
        if done.exists():
            metadata = json.loads(done.read_text(encoding='utf-8'))
            if metadata['identity'] != key:
                raise ValueError('Frozen feature cache identity changed; use a new run tag')
            for name, stamp in metadata['files'].items():
                path = destination / name
                if not path.is_file() or path.stat().st_size != stamp['bytes'] or path.stat().st_mtime_ns != stamp['mtime_ns']:
                    raise ValueError(f'Frozen feature cache changed: {path}')
            required_total += metadata['required_bytes']
            if required_total > budget_bytes:
                raise OSError('Existing frozen cache exceeds the configured stage cache budget')
            setattr(experiment, split, FeatureDataset(destination))
            continue
        experiment.model.eval()
        with torch.no_grad():
            first = experiment.features([dataset[0]]).detach().float().cpu().numpy()[0]
        sample = dataset[0]
        target_key = 'mask' if 'mask' in sample else 'target'
        target = np.asarray(sample[target_key])
        expected = len(dataset) * (first.nbytes + target.nbytes) + 1024 * 1024
        required_total += expected
        budget_root = Path(budget_root) if budget_root else directory
        other_bytes = sum(p.stat().st_size for p in budget_root.rglob('*')
                          if p.is_file() and not p.is_relative_to(destination))
        if required_total > budget_bytes or other_bytes+expected > budget_bytes:
            raise OSError(f'Frozen cache needs ~{required_total/1024**3:.2f} GiB, exceeding cache_gb; '
                          'increase disk cache budget or explicitly disable cache_frozen (slower)')
        if shutil.disk_usage(directory).free < expected + min_free_bytes:
            raise OSError('Insufficient free disk for frozen features plus safety reserve')
        features = np.lib.format.open_memmap(destination / 'features.npy', mode='w+', dtype=np.float32,
                                            shape=(len(dataset), *first.shape))
        targets = np.lib.format.open_memmap(destination / 'targets.npy', mode='w+', dtype=target.dtype,
                                           shape=(len(dataset), *target.shape))
        # Stream extraction: neither the full video population nor all features enter RAM/VRAM.
        with (destination / 'records.jsonl').open('w', encoding='utf-8') as handle, torch.no_grad():
            index, batch_size = 0, max(1, int(batch_size))
            while index < len(dataset):
                samples = [dataset[i] for i in range(index, min(len(dataset), index+batch_size))]
                try:
                    values = experiment.features(samples).detach().float().cpu().numpy()
                except torch.cuda.OutOfMemoryError:
                    if len(samples) == 1:
                        raise
                    batch_size = max(1, len(samples)//2)
                    torch.cuda.empty_cache()
                    continue
                if tuple(values.shape[1:]) != first.shape or not np.isfinite(values).all():
                    raise ValueError('Frozen features must have one finite fixed shape; use an adapter with explicit padding/masks otherwise')
                for offset, sample in enumerate(samples):
                    features[index+offset], targets[index+offset] = values[offset], np.asarray(sample[target_key])
                    record = dict(_metadata(sample), _target_key=target_key)
                    handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                index += len(samples)
                if heartbeat:
                    heartbeat(phase='feature_cache', split=split, completed=index, total=len(dataset), batch_size=batch_size)
        features.flush(); targets.flush()
        del features, targets
        files = {}
        for name in ('features.npy', 'targets.npy', 'records.jsonl'):
            path = destination / name
            files[name] = dict(bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=file_hash(path))
        write_json(done, dict(identity=key, files=files, required_bytes=expected,
                             reuse_check='size+mtime; SHA256 recorded at creation'))
        setattr(experiment, split, FeatureDataset(destination))
    return dict(bytes=required_total, directory=str(directory), precision='fp32',
                note='Deterministic frozen features only; no augmentation or encoder updates')
