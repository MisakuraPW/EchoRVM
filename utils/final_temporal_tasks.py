"""Strong frozen and fully differentiable task fits for the final v2 study.

Only ``last.pt`` carries optimizer state. It records the next effective-batch
cursor, including on interruption; partially accumulated gradients are replayed.
Frozen cache entries contain one window's descriptors and supervision, never a
video. Cache and resume identities include the source checkpoint's actual bytes.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import copy
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

from models.downstream import ViTPatchSegDecoder
from models.ef_readout import TemporalEFReadout
from models.final_temporal_mae import load_final_model
from .checkpoint import atomic_torch_save, tensor_payload_bytes
from .final_temporal_data import WindowDataset, _load_manifest
from .final_temporal_execution import encode_window, task_features
from .seed import get_rng_state, seed_everything, set_rng_state


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(path)


def _json(path, value):
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _csv(path, rows):
    stream = io.StringIO(newline='')
    if rows:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_text(path, stream.getvalue())


def _log(path, message):
    with Path(path).open('a', encoding='utf-8') as handle:
        handle.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}\n')


class _FeatureCache:
    """Bounded, disposable LRU shards; a namespace never trusts another run."""

    def __init__(self, root, identity, disk_bytes=8 * 1024 ** 3, ram_bytes=128 * 1024 ** 2):
        self.root = Path(root) / identity
        self.identity = identity
        self.disk_bytes, self.ram_bytes = int(disk_bytes), int(ram_bytes)
        if min(self.disk_bytes, self.ram_bytes) < 0:
            raise ValueError('Cache budgets must be nonnegative')
        self.root.mkdir(parents=True, exist_ok=True)
        self.ram, self.ram_used = OrderedDict(), 0
        self.hits = self.misses = self.writes = 0
        self._prune()

    def _path(self, key):
        return self.root / (key + '.pt')

    def __getstate__(self):
        # Spawned readers get their own bounded LRU, not a serialized warm RAM cache.
        state = dict(self.__dict__)
        state.update(ram=OrderedDict(), ram_used=0)
        return state

    def set_ram_budget(self, budget):
        self.ram_bytes = int(budget)
        if self.ram_bytes < 0:
            raise ValueError('RAM cache budget must be nonnegative')
        while self.ram and self.ram_used > self.ram_bytes:
            self.ram_used -= tensor_payload_bytes(self.ram.popitem(last=False)[1])

    def _remember(self, key, value):
        size = tensor_payload_bytes(value)
        if size > self.ram_bytes:
            return
        if key in self.ram:
            self.ram_used -= tensor_payload_bytes(self.ram.pop(key))
        while self.ram and self.ram_used + size > self.ram_bytes:
            self.ram_used -= tensor_payload_bytes(self.ram.popitem(last=False)[1])
        self.ram[key] = value
        self.ram_used += size

    def get(self, key):
        if key in self.ram:
            value = self.ram.pop(key)
            self.ram[key] = value
            self.hits += 1
            return value
        path = self._path(key)
        try:
            value = torch.load(path, map_location='cpu', weights_only=False)
        except FileNotFoundError:
            self.misses += 1
            return None
        if value.get('identity') != self.identity or value.get('key') != key:
            raise ValueError('Frozen cache identity mismatch')
        self.hits += 1
        self._remember(key, value['sample'])
        # Recency is advisory; another reader may race an eviction.
        try:
            path.touch()
        except FileNotFoundError:
            pass
        return value['sample']

    def put(self, key, sample):
        if 'video' in sample:
            raise ValueError('Videos must never enter the feature cache')
        sample = {k: v.detach().cpu().clone() if torch.is_tensor(v) else v
                  for k, v in sample.items()}
        self._remember(key, sample)
        if self.disk_bytes and tensor_payload_bytes(sample) <= self.disk_bytes:
            path = self._path(key)
            atomic_torch_save(dict(identity=self.identity, key=key, sample=sample), path)
            self.writes += 1
            self._prune()

    def _prune(self):
        files = []
        for path in self.root.parent.glob('*/*.pt'):
            try:
                stat = path.stat()
                files.append((stat.st_mtime_ns, path, stat.st_size))
            except FileNotFoundError:
                continue
        total = sum(size for _, _, size in files)
        for _, path, size in sorted(files):
            if total <= self.disk_bytes:
                break
            path.unlink(missing_ok=True)
            total -= size


class _Inputs(Dataset):
    def __init__(self, source, cache=None, split='train'):
        self.source, self.cache, self.split = source, cache, split

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):
        index, cursor, size, last = item if isinstance(item, tuple) else (item, -1, 0, False)
        record = self.source.get_record(index)
        _check_source(record)
        key = _digest(dict(split=self.split, record=record))
        sample = self.cache.get(key) if self.cache else None
        if sample is None:
            sample = self.source[index]
        sample = dict(sample)
        sample.update(cache_key=key, update_cursor=cursor, update_size=size, update_last=last)
        return sample


def _check_source(case):
    stat = Path(case['path']).stat()
    if stat.st_size != case['source_bytes'] or stat.st_mtime_ns != case['source_mtime_ns']:
        raise ValueError('Data changed after manifest creation; cached features cannot bypass provenance')


def _evaluation_population(dataset):
    """Canonical identities for the exact VAL draws, not just cohort counts."""
    records = []
    for index in range(len(dataset)):
        record = dataset.get_record(index)
        start, h = int(record['recent_start']), int(record['prefix_frames'])
        position = int(record.get('target_position', -1))
        identity = dict(patient=record['patient'], recent_start=start, H=h,
                        source_frame=int(record.get('target_frame', -1)), target_position=position,
                        target_index=h + position if position >= 0 else -1,
                        window_source_start=start - h, window_source_end=start + dataset.recent_frames - 1,
                        full_context=True)
        records.append(dict(id=_digest(identity), **identity))
    records.sort(key=lambda value: (value['patient'], value['source_frame'], value['target_position'],
                                     value['recent_start'], value['H']))
    sources = []
    for patient in sorted({record['patient'] for record in records}):
        case = dataset.cases[patient]
        sources.append(dict(patient=patient, path=str(Path(case['path']).resolve()), frames=case['frames'],
                            fps=case['fps'], source_bytes=case['source_bytes'], source_mtime_ns=case['source_mtime_ns'],
                            source_fingerprint=case.get('source_fingerprint'), shape=case.get('shape'), dtype=case.get('dtype'),
                            label_sha256=_digest(case['traces'] if dataset.task == 'seg' else dict(ef=case['ef']))))
    return dict(version=2, split='val', task=dataset.task, dataset_seed=dataset.seed,
                recent_frames=dataset.recent_frames, local_frames=dataset.local_frames,
                records=records, source_provenance=sources,
                video_provenance_policy='Manifest stat/shape/dtype fingerprint; video content not rehashed')


class _UpdateBatches(Sampler):
    """Effective updates own their permutation; micro-batches cannot change it."""

    def __init__(self, source, micro, effective, seed, epoch, start=0, training=True):
        self.source, self.micro, self.effective = source, int(micro), int(effective)
        self.seed, self.epoch, self.start, self.training = seed, epoch, start, training

    def __iter__(self):
        if self.training:
            generator = torch.Generator().manual_seed(self.seed + self.epoch * 1000003)
            order = torch.randperm(len(self.source), generator=generator).tolist()
        else:
            order = list(range(len(self.source)))
        for cursor, offset in enumerate(range(0, len(order), self.effective)):
            if cursor < self.start:
                continue
            update = order[offset:offset + self.effective]
            groups = defaultdict(list)
            for index in update:
                groups[self.source.get_record(index)['prefix_frames']].append(index)
            batches = [indices[i:i + self.micro] for _, indices in sorted(groups.items())
                       for i in range(0, len(indices), self.micro)]
            for i, batch in enumerate(batches):
                yield [(index, cursor, len(update), i == len(batches) - 1) for index in batch]

    def __len__(self):
        return sum(1 for _ in self)


def _collate(samples):
    return samples


def task_worker_init(worker_id):
    """Runtime-only CPU thread limits for each spawned data reader."""
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)


class _RepeatedBatches(Sampler):
    """Finite I/O-only repetition of the real sampler, including tiny datasets."""

    def __init__(self, sampler, count):
        self.sampler, self.count = sampler, int(count)

    def __iter__(self):
        emitted = 0
        while emitted < self.count:
            produced = False
            for batch in self.sampler:
                produced = True
                yield batch
                emitted += 1
                if emitted == self.count:
                    return
            if not produced:
                raise ValueError('Cannot benchmark an empty task loader')

    def __len__(self):
        return self.count


def _loader(inputs, micro, effective, seed, epoch=0, start=0, training=False, workers=0, benchmark_batches=None):
    sampler = _UpdateBatches(inputs.source, micro, effective, seed, epoch, start, training)
    if benchmark_batches is not None:
        sampler = _RepeatedBatches(sampler, benchmark_batches)
    options = dict(persistent_workers=True, prefetch_factor=4, worker_init_fn=task_worker_init) if workers else {}
    return DataLoader(inputs, batch_sampler=sampler, num_workers=workers, collate_fn=_collate,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(seed + epoch), **options)


def _tune_workers(inputs, job, micro):
    """Train-only real loader I/O: four prewarm batches, then twenty timed batches."""
    requested = int(job['num_workers'])
    patients = {case['patient'] for case in inputs.source.manifest['train']}
    if inputs.split != 'train' or any(record['patient'] not in patients for record in inputs.source.records):
        raise ValueError('Worker timing may only use TRAIN records')
    cache = inputs.cache
    if not job['autotune'] or requested == 0:
        if cache:
            cache.set_ram_budget(job['ram_cache_bytes'] // (requested + 1))
        return requested, dict(mode='explicit', requested_workers=requested, selected_workers=requested, trials=[])
    rng = get_rng_state()
    trials = []
    original_budget = cache.ram_bytes if cache else None
    chosen = None
    candidates = sorted({0, min(4, requested), requested})
    loader = iterator = batch = None
    try:
        for count in candidates:
            budget = job['ram_cache_bytes'] // (count + 1)
            if cache:
                cache.set_ram_budget(budget)
            try:
                loader = _loader(inputs, micro, job['effective_batch'], job['seed'],
                                 epoch=int(inputs.source.epoch.value), training=True, workers=count,
                                 benchmark_batches=24)
                iterator = iter(loader)
                for _ in range(4):
                    batch = next(iterator)
                started = time.perf_counter()
                samples = 0
                for _ in range(20):
                    batch = next(iterator)
                    samples += len(batch)
                elapsed = time.perf_counter() - started
                trials.append(dict(num_workers=count, status='ok', prewarm_batches=4, measured_batches=20,
                                   measured_samples=samples, seconds=elapsed,
                                   samples_per_second=samples / max(elapsed, 1e-9),
                                   ram_cache_bytes_per_process=budget, ram_cache_processes=count + 1))
            except (RuntimeError, OSError) as exc:
                if count == 0:
                    raise
                trials.append(dict(num_workers=count, status='unavailable', error=str(exc)[:1500],
                                   ram_cache_bytes_per_process=budget, ram_cache_processes=count + 1))
            finally:
                iterator = loader = batch = None
                gc.collect()
        valid = [row for row in trials if row['status'] == 'ok']
        chosen = max(valid, key=lambda row: (row['samples_per_second'], -row['num_workers']))['num_workers']
        if cache:
            cache.set_ram_budget(job['ram_cache_bytes'] // (chosen + 1))
        return chosen, dict(mode='train_loader_io_only', requested_workers=requested, selected_workers=chosen,
                            trials=trials, accuracy_consulted=False, effective_batch=job['effective_batch'], micro_batch=micro,
                            ram_cache_total_budget_bytes=job['ram_cache_bytes'])
    finally:
        iterator = loader = batch = None
        if cache and chosen is None:
            cache.set_ram_budget(original_budget)
        set_rng_state(rng)
        gc.collect()


def _video(sample, backbone, device):
    video = sample['video'].to(device, non_blocking=True)
    if video.shape[-2:] != (backbone.img_size, backbone.img_size):
        video = F.interpolate(video, (backbone.img_size, backbone.img_size), mode='bilinear',
                              align_corners=False)
    if backbone.in_chans == 1:
        video = video[:, :1]
    return video


def _extract(backbone, samples, job, device):
    groups = defaultdict(list)
    for index, sample in enumerate(samples):
        groups[(int(sample['prefix_frames']), len(sample['video']))].append(index)
    result = [None] * len(samples)
    slots = None
    for (prefix, _), indices in groups.items():
        video = torch.stack([_video(samples[i], backbone, device) for i in indices])
        targets = ([int(samples[i]['target_index']) for i in indices] if job['task'] == 'seg' else None)
        exits = encode_window(backbone, video, prefix_frames=prefix,
                              recent_frames=job['recent_frames'], target_index=targets)
        features, history = task_features(exits, job['task'], job['exit_name'],
                                          job['append_boundary'], backbone.memory_slots)
        if slots is not None and history != slots:
            raise ValueError('A batch must use one fixed boundary slot count')
        slots = history
        for row, index in enumerate(indices):
            result[index] = features[row]
    return torch.stack(result), int(slots or 0)


def _features(backbone, samples, job, device, cache=None):
    if not job['freeze']:
        return _extract(backbone, samples, job, device)
    missing = [i for i, sample in enumerate(samples) if 'features' not in sample]
    if missing:
        with torch.no_grad(), torch.autocast(device.type, enabled=False):
            values, slots = _extract(backbone, [samples[i] for i in missing], job, device)
        for row, index in enumerate(missing):
            sample = {k: v for k, v in samples[index].items()
                      if k not in ('video', 'frame_indices', 'update_cursor', 'update_size', 'update_last')}
            sample.update(features=values[row].detach().cpu(), history_slots=slots)
            if cache:
                cache.put(sample['cache_key'], sample)
            samples[index].update(features=sample['features'], history_slots=slots)
    slots = {int(sample['history_slots']) for sample in samples}
    if len(slots) != 1:
        raise ValueError('Inconsistent cached boundary slots')
    return torch.stack([sample['features'] for sample in samples]).to(device), slots.pop()


class _TaskModel(nn.Module):
    def __init__(self, backbone, job, normalization):
        super().__init__()
        self.backbone = backbone
        self.task = job['task']
        self.exit_name, self.append_boundary = job['exit_name'], job['append_boundary']
        self.recent_frames = job['recent_frames']
        self.grid = backbone.token_grid[1]
        self.head = (TemporalEFReadout(backbone.embed_dim, hidden=job['head_dim']) if self.task == 'ef'
                     else ViTPatchSegDecoder(backbone.embed_dim, 2, self.grid, backbone.patch_size,
                                            job['head_dim'], job['head_depth'], job['head_heads']))
        self.register_buffer('feature_mean', torch.tensor(normalization['feature_mean']))
        self.register_buffer('feature_std', torch.tensor(normalization['feature_std']))

    def read(self, features, slots=0):
        values = (features.float() - self.feature_mean) / self.feature_std
        return self.head(values, slots) if self.task == 'ef' else self.head(values, self.grid)

    def forward(self, video, prefix_frames=0, recent_frames=None, target_index=None):
        exits = encode_window(self.backbone, video, prefix_frames=prefix_frames,
                              recent_frames=self.recent_frames if recent_frames is None else recent_frames,
                              target_index=target_index)
        values, slots = task_features(exits, self.task, self.exit_name, self.append_boundary,
                                       self.backbone.memory_slots)
        return self.read(values, slots)


def load_frozen_task_head(path, model, protocol_or_metrics=None):
    """Return an eval-only task module reusing a frozen fit's exact trained head.

    ``model`` is the source backbone already loaded onto the desired device.
    The optional third argument is a protocol dict/path or the returned metrics
    dict. Call the result on native-resolution [B,T,C,H,W], supplying absolute
    ``target_index`` for seg and real ``prefix_frames`` for history experiments.
    EF outputs remain normalized; denormalize with ``result.normalization``.
    """
    saved = torch.load(Path(path), map_location='cpu', weights_only=False)
    protocol = saved.get('protocol')
    if not protocol or not protocol['freeze']:
        raise ValueError('load_frozen_task_head requires a frozen task best checkpoint')
    if protocol_or_metrics is not None:
        expected = protocol_or_metrics
        if isinstance(expected, dict) and 'protocol_path' in expected:
            expected = expected['protocol_path']
        if isinstance(expected, (str, Path)):
            expected = json.loads(Path(expected).read_text(encoding='utf-8'))
        if _resume_identity(expected) != _resume_identity(protocol):
            raise ValueError('Frozen head protocol mismatch')
    head = protocol['head']
    job = dict(task=protocol['task'], exit_name=protocol['exit_name'],
               append_boundary=protocol['append_boundary'], recent_frames=protocol['recent_frames'],
               head_dim=head['dim'], head_depth=head['depth'] or 4, head_heads=head['heads'] or 3)
    rng = get_rng_state()
    try:
        result = _TaskModel(model, job, saved['normalization']).to(next(model.parameters()).device)
        allowed = {k for k in result.state_dict() if k.startswith('head.')} | {'feature_mean', 'feature_std'}
        if set(saved['model_state_dict']) != allowed:
            raise ValueError('Frozen head checkpoint contains unexpected task/backbone tensors')
        result.load_state_dict(saved['model_state_dict'], strict=False)
        result.normalization = copy.deepcopy(saved['normalization'])
        result.task_protocol = copy.deepcopy(protocol)
        result.requires_grad_(False)
        return result.eval()
    finally:
        set_rng_state(rng)


def _fit_normalization(backbone, inputs, job, device, cache):
    backbone.eval()
    count, mean, m2 = 0, torch.zeros(backbone.embed_dim, dtype=torch.float64), torch.zeros(
        backbone.embed_dim, dtype=torch.float64)
    loader = _loader(inputs, job.get('micro_batch') or 1, job['effective_batch'], job['seed'],
                     workers=job['num_workers'])
    # Float64 parallel Welford statistics bound RAM and avoid E[x^2]-E[x]^2 cancellation.
    with torch.no_grad():
        for samples in loader:
            values, _ = _features(backbone, samples, dict(job, freeze=True), device, cache)
            values = values.detach().cpu().double().reshape(-1, backbone.embed_dim)
            n = len(values)
            batch_mean = values.mean(0)
            delta = batch_mean - mean
            m2 += (values - batch_mean).square().sum(0) + delta.square() * count * n / (count + n)
            mean += delta * n / (count + n)
            count += n
    cases = {r['patient']: inputs.source.cases[r['patient']] for r in inputs.source.records}
    targets = np.array([case['ef'] for case in cases.values()], dtype=np.float64) if job['task'] == 'ef' else None
    return dict(version=2, fit_split='train', fit_patients=sorted(cases), feature_count=count,
                feature_mean=mean.tolist(), feature_std=(m2 / count).sqrt().clamp_min(1e-6).tolist(),
                target_mean=float(targets.mean()) if targets is not None else 0.0,
                target_std=(float(targets.std()) if float(targets.std()) > 1e-6 else 1.0)
                if targets is not None else 1.0,
                target_units='percentage_points' if targets is not None else 'binary_mask',
                target_transform='(EF - train_mean) / train_std; no sigmoid' if targets is not None else 'none')


_MAE_ONLY = ('decoder_embed.', 'decoder_blocks.', 'decoder_norm.', 'decoder_pred.',
             'memory_to_decoder.', 'memory_fusion.', 'mask_token')


def _active_parameters(model, samples, job, device):
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad_(not job['freeze'] and not name.startswith(_MAE_ONLY))
    if not job['freeze']:
        rng = get_rng_state()
        model.eval()
        try:
            values, slots = _extract(model.backbone, samples, job, device)
            model.read(values, slots).float().sum().backward()
            for parameter in model.backbone.parameters():
                parameter.requires_grad_(parameter.grad is not None)
        finally:
            model.zero_grad(set_to_none=True)
            set_rng_state(rng)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _task_state(model, active):
    allowed = set(active) | {'feature_mean', 'feature_std'}
    return {key: value for key, value in model.state_dict().items() if key in allowed or key.startswith('head.')}


def _restore_task_state(model, state, active):
    if set(state) != set(_task_state(model, active)):
        raise ValueError('Task checkpoint active parameter mismatch')
    model.load_state_dict(state, strict=False)


def _loss(prediction, samples, task, normalization):
    if task == 'ef':
        targets = torch.stack([s['target'] for s in samples]).to(prediction.device).float()
        targets = (targets - normalization['target_mean']) / normalization['target_std']
        return F.smooth_l1_loss(prediction.float(), targets)
    masks = torch.stack([s['mask'] for s in samples]).to(prediction.device).long()
    logits = F.interpolate(prediction.float(), masks.shape[-2:], mode='bilinear', align_corners=False)
    ce = F.cross_entropy(logits, masks, reduction='none').mean((1, 2))
    probability = logits.softmax(1)[:, 1]
    truth = masks.float()
    dice = (2 * (probability * truth).sum((1, 2)) + 1e-6) / (
        probability.sum((1, 2)) + truth.sum((1, 2)) + 1e-6)
    return (ce + 1 - dice).mean()


@torch.no_grad()
def _training_monitor(prediction, samples, task, normalization):
    if task == 'ef':
        targets = torch.stack([s['target'] for s in samples]).to(prediction.device).float()
        physical = prediction.float() * normalization['target_std'] + normalization['target_mean']
        return float((physical - targets).abs().mean())
    masks = torch.stack([s['mask'] for s in samples]).to(prediction.device).bool()
    hard = F.interpolate(prediction.float(), masks.shape[-2:], mode='bilinear', align_corners=False).argmax(1).bool()
    dice = (2 * (hard & masks).sum((1, 2)).float() + 1e-6) / (
        hard.sum((1, 2)) + masks.sum((1, 2)) + 1e-6)
    return float(dice.mean())


def _amp(job, device):
    enabled = device.type == 'cuda' and job['precision'] != 'fp32'
    dtype = torch.bfloat16 if job['precision'] == 'bf16' else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _train_mode(model, job):
    model.train()
    if job['freeze']:
        model.backbone.eval()


def _scaler(job, device):
    return torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and job['precision'] == 'fp16',
                               init_scale=128.0)


def _autotune(model, samples, job, normalization, device):
    """Benchmark a disposable clone with synthetic targets, without tuning accuracy."""
    if job.get('micro_batch') is not None:
        return int(job['micro_batch']), dict(mode='explicit', trials=[])
    if not job['autotune']:
        return 1, dict(mode='disabled', trials=[])
    rng = get_rng_state()
    original_device = next(model.parameters()).device
    probe = initial = optimizer = scaler = values = loss = batch = None
    trials, chosen, best_rate = [], None, -1.0
    candidates = sorted(set([1, job['effective_batch']] + [2 ** k for k in range(
        1, int(math.log2(job['effective_batch'])) + 1)])) if device.type == 'cuda' else [1]
    try:
        # Do not count two GPU copies against a job's available memory.
        model.cpu()
        probe = copy.deepcopy(model).to(device)
        initial = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
        _train_mode(probe, job)
        for micro in candidates:
            probe.load_state_dict(initial)
            set_rng_state(rng)
            probe.zero_grad(set_to_none=True)
            optimizer = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=job['lr'],
                                           weight_decay=job['weight_decay'])
            scaler = _scaler(job, device)
            peak = 0
            started = time.perf_counter()
            feasible = True
            successful = attempts = 0
            measured_updates = 2 if device.type == 'cuda' else 1
            if device.type == 'cuda':
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                free_before, _ = torch.cuda.mem_get_info(device)
                baseline = torch.cuda.memory_reserved(device)
            try:
                while successful < measured_updates and attempts < measured_updates + 8:
                    probe.zero_grad(set_to_none=True)
                    for offset in range(0, job['effective_batch'], micro):
                        n = min(micro, job['effective_batch'] - offset)
                        batch = [dict(samples[i % len(samples)]) for i in range(n)]
                        for sample in batch:
                            if job['task'] == 'ef':
                                sample['target'] = torch.tensor(normalization['target_mean'])
                            else:
                                sample['mask'] = torch.zeros_like(sample['mask'])
                        with _amp(job, device):
                            values, slots = _features(probe.backbone, batch, job, device)
                            loss = _loss(probe.read(values, slots), batch, job['task'], normalization)
                        scaler.scale(loss * n / job['effective_batch']).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    before = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    successful += int(scaler.get_scale() >= before)
                    attempts += 1
                feasible = successful == measured_updates
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    peak = torch.cuda.max_memory_reserved(device)
                    reserve = max(job['tune_reserve_bytes'], int(free_before * job['tune_reserve_fraction']))
                    feasible = feasible and peak - baseline + reserve <= free_before
            except torch.cuda.OutOfMemoryError:
                feasible = False
            elapsed = time.perf_counter() - started
            rate = job['effective_batch'] * successful / max(elapsed, 1e-9)
            trials.append(dict(micro_batch=micro, seconds=elapsed, samples_per_second=rate,
                               peak_cuda_bytes=peak, successful_updates=successful,
                               attempts=attempts, feasible=feasible))
            if feasible and rate > best_rate:
                chosen, best_rate = micro, rate
            probe.zero_grad(set_to_none=True)
            optimizer = scaler = values = loss = batch = None
            if not feasible:
                break
        if chosen is None:
            raise RuntimeError('No micro-batch fits the requested CUDA memory reserve')
        return chosen, dict(mode='real_forward_backward_clone', trials=trials,
                            synthetic_labels=True, precision=job['precision'],
                            workload='cached_head' if job['freeze'] and all('features' in sample for sample in samples)
                            else 'encoder_and_head',
                            gradient_checkpointing=bool(model.backbone.gradient_checkpointing))
    finally:
        probe = initial = optimizer = scaler = values = loss = batch = None
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model.to(original_device)
        set_rng_state(rng)


def _aggregate(rows, task, expected_positions=None):
    sources = defaultdict(list)
    for row in rows:
        if not row['full_context']:
            raise ValueError('Validation requires complete-context source intersection')
        sources[(row['patient'], row['source_frame'])].append(row)
    patients = defaultdict(list)
    for (patient, frame), group in sorted(sources.items()):
        if task == 'seg' and expected_positions is not None:
            positions = [r['position'] for r in group]
            if len(positions) != len(set(positions)) or set(positions) != set(expected_positions):
                raise ValueError('Incomplete/duplicate validation target positions')
        source = dict(patient=patient, source_frame=frame, windows=len(group),
                      full_context_windows=sum(r['full_context'] for r in group))
        if task == 'ef':
            if len({r['target'] for r in group}) != 1:
                raise ValueError('Conflicting EF labels within a source')
            source.update(prediction=float(np.mean([r['prediction'] for r in group])), target=group[0]['target'])
        else:
            source['dice'] = float(np.mean([r['dice'] for r in group]))
        patients[patient].append(source)
    result = []
    for patient, group in sorted(patients.items()):
        row = dict(patient=patient, task=task, source_frames=len(group),
                   positions=sum(r['windows'] for r in group), windows=sum(r['windows'] for r in group),
                   full_context_windows=sum(r['full_context_windows'] for r in group),
                   full_context_sources=len(group))
        if task == 'ef':
            row.update(prediction=float(np.mean([r['prediction'] for r in group])),
                       target=float(np.mean([r['target'] for r in group])))
            row['absolute_error_pp'] = abs(row['prediction'] - row['target'])
        else:
            row['dice'] = float(np.mean([r['dice'] for r in group]))
        result.append(row)
    if not result:
        raise ValueError('No complete validation patients')
    if task == 'ef':
        errors = np.array([r['prediction'] - r['target'] for r in result])
        metrics = dict(mae_pp=float(np.abs(errors).mean()), rmse_pp=float(np.sqrt((errors ** 2).mean())))
    else:
        metrics = dict(patient_dice=float(np.mean([r['dice'] for r in result])))
    metrics.update(patients=len(result), source_frames=len(sources), windows=len(rows),
                   full_context_windows=len(rows), full_context_sources=len(sources))
    return metrics, result


def _evaluate(model, inputs, job, normalization, device, cache, micro):
    model.eval()
    rows, loss_sum, count = [], 0.0, 0
    started = time.perf_counter()
    with torch.no_grad():
        for samples in _loader(inputs, micro, job['effective_batch'], job['seed'], workers=job['num_workers']):
            with _amp(job, device):
                values, slots = _features(model.backbone, samples, job, device, inputs.cache)
                prediction = model.read(values, slots)
                loss = _loss(prediction, samples, job['task'], normalization)
            loss_sum += float(loss) * len(samples)
            count += len(samples)
            if job['task'] == 'ef':
                physical = prediction.float().cpu() * normalization['target_std'] + normalization['target_mean']
            else:
                masks = torch.stack([s['mask'] for s in samples]).to(device)
                hard = F.interpolate(prediction.float(), masks.shape[-2:], mode='bilinear',
                                     align_corners=False).argmax(1).bool()
                truth = masks.bool()
                physical = ((2 * (hard & truth).sum((1, 2)).float() + 1e-6) /
                            (hard.sum((1, 2)) + truth.sum((1, 2)) + 1e-6)).cpu()
            for sample, value in zip(samples, physical.tolist()):
                row = dict(patient=sample['patient'], source_frame=int(sample.get('target_frame', -1)),
                           position=int(sample['target_position']), full_context=bool(sample['full_context']))
                row.update(dict(prediction=value, target=float(sample['target'])) if job['task'] == 'ef'
                           else dict(dice=value))
                rows.append(row)
    positions = range(job['recent_frames'] - job['local_frames'], job['recent_frames'])
    metrics, patients = _aggregate(rows, job['task'], positions if job['task'] == 'seg' else None)
    metrics.update(val_loss=loss_sum / count, evaluation_seconds=time.perf_counter() - started)
    return metrics, patients


def _memory(device):
    try:
        import psutil
        rss = psutil.Process().memory_info().rss
    except ImportError:
        rss = 0
    return dict(rss_bytes=rss, peak_cuda_bytes=torch.cuda.max_memory_allocated(device)
                if device.type == 'cuda' else 0)


def _plot(history, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(7, 4), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    axes.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='Train', color='#0072b2')
    axes.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='Validation', color='#d55e00',
              linestyle='--')
    axes.set(xlabel='Epoch', ylabel='Task loss')
    axes.grid(alpha=.2)
    axes.legend()
    temporary = Path(path).with_suffix('.tmp.png')
    figure.savefig(temporary, dpi=150)
    temporary.replace(path)


def _publish_history(output, history):
    _csv(output / 'logs' / 'metrics.csv', history)
    _atomic_text(output / 'logs' / 'metrics.jsonl', ''.join(json.dumps(row, allow_nan=False) + '\n' for row in history))
    _plot(history, output / 'plots' / 'loss_latest.png')


def _resolve_job(job, backbone):
    job = copy.deepcopy(dict(job))
    task = job.get('task')
    if task not in ('ef', 'seg'):
        raise ValueError('task must be ef or seg')
    freeze = bool(job.get('freeze', True))
    defaults = dict(freeze=freeze, seed=42, dataset_seed=42, epochs=60 if freeze else 80,
                    patience=12 if freeze else 20, num_workers=8,
                    effective_batch=12 if task == 'ef' else 64, autotune=True, smoke=False,
                    exit_name='final', append_boundary=False, recent_frames=64,
                    local_frames=backbone.local_frames, max_prefix=128, prefix=0,
                    head_dim=64 if task == 'ef' else 192, head_depth=4, head_heads=3,
                    precision='fp16', tune_reserve_bytes=512 * 1024 ** 2,
                    tune_reserve_fraction=.1, disk_cache_bytes=8 * 1024 ** 3,
                    ram_cache_bytes=128 * 1024 ** 2)
    for key, value in defaults.items():
        job.setdefault(key, value)
    if int(job['dataset_seed']) != job['dataset_seed']:
        raise ValueError('dataset_seed must be an integer')
    job['lr'] = (1e-3 if task == 'ef' else 3e-4) if freeze else (5e-5 if task == 'ef' else 1e-4)
    job['weight_decay'] = .01 if freeze else 1e-4
    job['warmup_epochs'] = 3 if freeze else 5
    job.setdefault('min_free_gb', 0 if job['smoke'] else 5)
    for key in ('epochs', 'patience', 'effective_batch', 'head_dim', 'head_depth', 'head_heads'):
        if int(job[key]) != job[key] or job[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    if int(job['num_workers']) != job['num_workers'] or not 0 <= job['num_workers'] <= 8:
        raise ValueError('num_workers must be an integer in [0,8]')
    if job.get('micro_batch') is not None and (int(job['micro_batch']) != job['micro_batch'] or
                                              not 1 <= job['micro_batch'] <= job['effective_batch']):
        raise ValueError('micro_batch must be in [1, effective_batch]')
    if job['exit_name'] not in ('final', 'base', 'local', 'local_final'):
        raise ValueError('Unknown task exit')
    if job['precision'] not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16, or bf16')
    if job['local_frames'] != backbone.local_frames:
        raise ValueError('local_frames must match the loaded backbone')
    if not job['smoke'] and (backbone.img_size != 112 or backbone.local_frames != 16 or
                             job['recent_frames'] != 64 or job['head_dim'] != defaults['head_dim'] or
                             job['head_depth'] != 4 or job['head_heads'] != 3):
        raise ValueError('Nonstandard model/window/head dimensions require smoke=True')
    if task == 'seg' and (job['head_dim'] % job['head_heads'] or job['head_dim'] % 4):
        raise ValueError('Segmentation decoder dimension must be divisible by heads and four')
    if not 0 <= job['tune_reserve_fraction'] < 1 or job['tune_reserve_bytes'] < 0:
        raise ValueError('Invalid tuning memory reserve')
    if job['min_free_gb'] < 0:
        raise ValueError('min_free_gb must be nonnegative')
    return job


def _resume_identity(protocol):
    # Paths, worker count and disposable cache budgets are not scientific state.
    return _digest({k: v for k, v in protocol.items() if k not in ('runtime', 'load_report', 'autotune')})


def _reconcile_best(path, pending, progress, identity):
    """Finish an interrupted two-file checkpoint publication without losing old best."""
    if progress['best_epoch'] is None:
        pending.unlink(missing_ok=True)
        return
    for candidate in (pending, path):
        if not candidate.exists():
            continue
        value = torch.load(candidate, map_location='cpu', weights_only=False)
        if (value.get('identity') == identity and value.get('epoch') == progress['best_epoch']
                and value.get('metric') == progress['best_metric']):
            if candidate == pending:
                pending.replace(path)
            else:
                pending.unlink(missing_ok=True)
            return
    raise ValueError('Observed best checkpoint does not match the durable resume state')


def _completed_metrics(output, checkpoints, identity, resumed):
    path = output / 'DONE'
    if not path.exists():
        return None
    done = json.loads(path.read_text(encoding='utf-8'))
    if done.get('protocol_sha256') != identity:
        raise ValueError('Completed task protocol mismatch')
    for name, digest in done['artifacts'].items():
        artifact = output / name
        if not artifact.is_file() or _file_digest(artifact) != digest:
            raise ValueError('Completed task has missing or altered outputs')
    if resumed is None or not resumed['progress']['finished']:
        raise ValueError('Completed task has no finished resume checkpoint')
    metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    if metrics['checkpoint_dir'] != str(checkpoints.resolve()):
        raise ValueError('Completed task checkpoint directory changed')
    for name, digest in done['checkpoints'].items():
        artifact = checkpoints / name
        if not artifact.is_file() or _file_digest(artifact) != digest:
            raise ValueError('Completed task has missing or altered checkpoints')
    return metrics


def _tensorboard(output, job):
    if not job.get('tensorboard', False):
        return None
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        message = 'TensorBoard requested but unavailable; CSV/JSONL logging remains enabled'
        warnings.warn(message, RuntimeWarning)
        _log(output / 'logs' / 'train.log', message)
        return None
    return SummaryWriter(log_dir=str(output / 'logs' / 'tensorboard'))


def _calibration_sample(clean, manifest, arguments, task):
    """Stress the longest admissible TRAIN context, without looking at labels."""
    if clean.prefix is not None:
        return clean[0]
    candidates = []
    for index in range(len(clean)):
        record = clean.get_record(index)
        case = clean.cases[record['patient']]
        if task == 'ef':
            start = min(clean.max_prefix, case['frames'] - clean.recent_frames)
            start = start // clean.local_frames * clean.local_frames
        else:
            start = record['target_frame'] - (clean.recent_frames - clean.local_frames)
        available = start if task == 'ef' else record['target_frame'] - (clean.recent_frames - 1)
        h = min(clean.max_prefix, available) // clean.local_frames * clean.local_frames
        candidates.append(dict(patient=record['patient'], recent_start=start, H=h,
                               **({'target_frame': record['target_frame']} if task == 'seg' else {})))
    record = max(candidates, key=lambda value: value['H'])
    return WindowDataset(manifest, 'train', records=[record], **arguments)[0]


def run_task_job(job, manifest, device):
    """Fit approved EF/seg tasks, resume exactly, and publish the observed best.

    ``subset={'train': N, 'val': N}`` is the only implicit-case restriction;
    ``smoke`` permits small model/head/window dimensions but never truncates data.
    Existing last.pt is resumed automatically. No TEST samples are constructed.
    """
    started = time.perf_counter()
    device = torch.device(device)
    manifest = _load_manifest(manifest)
    seed = int(job.get('seed', 42))
    seed_everything(seed)
    overrides = dict(gradient_checkpointing=True)
    overrides.update(job.get('overrides') or {})
    backbone, config, report = load_final_model(job['checkpoint'], overrides=overrides, seed=seed)
    job = _resolve_job(job, backbone)
    output, checkpoints = Path(job['output_dir']), Path(job['checkpoint_dir'])
    if output.resolve() == checkpoints.resolve():
        raise ValueError('Results and checkpoints must have separate directories')
    if Path(job['checkpoint']).resolve() in {(checkpoints / name).resolve() for name in ('last.pt', 'best.pt')}:
        raise ValueError('Source checkpoint cannot be overwritten by task checkpoints')
    for path in (output, output / 'logs', output / 'plots', checkpoints):
        path.mkdir(parents=True, exist_ok=True)
    last_path, best_path = checkpoints / 'last.pt', checkpoints / 'best.pt'
    pending_best = checkpoints / 'best.pt.pending'
    subset = job.get('subset') or {}
    if not isinstance(subset, dict) or set(subset) - {'train', 'val'}:
        raise ValueError('subset must contain only train/val case limits')
    augmentation = None if job['freeze'] else dict(enabled=True, preset='A4_tgc_zoom_speckle', per_frame_random=False)
    arguments = dict(task=job['task'], recent_frames=job['recent_frames'], local_frames=job['local_frames'],
                     max_prefix=job['max_prefix'], seed=int(job['dataset_seed']), prefix=job['prefix'])
    clean = WindowDataset(manifest, 'train', training=False, positions='balanced', limit=subset.get('train'), **arguments)
    train = clean if job['freeze'] else WindowDataset(manifest, 'train', training=True, positions='balanced',
                                                     aug_cfg=augmentation, limit=subset.get('train'), **arguments)
    val = WindowDataset(manifest, 'val', training=False, positions='all', limit=subset.get('val'), **arguments)
    if not len(train) or not len(val):
        raise ValueError('Tasks require nonempty eligible TRAIN and VAL sources')
    for dataset in (clean, val):
        for patient in {record['patient'] for record in dataset.records}:
            _check_source(dataset.cases[patient])
    evaluation_population = _evaluation_population(val)
    evaluation_sha256 = _digest(evaluation_population)
    protocol = dict(version=2, task=job['task'], freeze=job['freeze'], seed=seed, dataset_seed=job['dataset_seed'],
                    evaluation_population_sha256=evaluation_sha256,
                    source_checkpoint_sha256=_file_digest(job['checkpoint']), manifest_sha256=_digest(manifest),
                    model=config['model'], exit_name=job['exit_name'], append_boundary=job['append_boundary'],
                    recent_frames=job['recent_frames'], local_frames=job['local_frames'], prefix=job['prefix'],
                    max_prefix=job['max_prefix'], subset=subset, smoke=job['smoke'],
                    head=dict(name='TemporalEFReadout' if job['task'] == 'ef' else 'ViTPatchSegDecoder',
                              dim=job['head_dim'], depth=job['head_depth'] if job['task'] == 'seg' else None,
                              heads=job['head_heads'] if job['task'] == 'seg' else None),
                    optimizer=dict(name='AdamW', lr=job['lr'], weight_decay=job['weight_decay'], clip_norm=1.0),
                    schedule=dict(name='cosine', epochs=job['epochs'], warmup_epochs=job['warmup_epochs'],
                                  successful_updates_only=True), patience=job['patience'],
                    effective_batch=job['effective_batch'], precision=job['precision'] if device.type == 'cuda' else 'fp32',
                    augmentation=augmentation, frozen_train_positions='one_fixed_balanced_per_source',
                    frozen_feature_precision='fp32',
                    validation_positions=('all_local_positions_complete_context_intersection' if job['task'] == 'seg'
                                          else 'one_fixed_real_window_per_patient'),
                    validation_cache='none_streaming' if job['task'] == 'seg' else 'bounded_frozen_features',
                    aggregation='position -> original source frame -> patient, equally weighted sources/patients',
                    train_samples=len(train), val_samples=len(val), train_excluded=clean.excluded, val_excluded=val.excluded,
                    code_sha256={name: _file_digest(Path(__file__).parents[1] / name) for name in (
                        'utils/final_temporal_tasks.py', 'utils/final_temporal_data.py',
                        'utils/final_temporal_execution.py', 'models/final_temporal_mae.py',
                        'models/ef_readout.py', 'models/downstream.py')},
                    load_report=report, runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                                                    device=str(device), cuda=torch.version.cuda, workers=job['num_workers'],
                                                    min_free_checkpoint_gb=job['min_free_gb']))
    cache_identity = _digest({k: protocol[k] for k in ('source_checkpoint_sha256', 'manifest_sha256', 'model',
                              'task', 'exit_name', 'append_boundary', 'recent_frames', 'local_frames', 'prefix',
                              'max_prefix', 'seed', 'dataset_seed', 'subset', 'code_sha256', 'frozen_feature_precision')})
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding='utf-8'))
        previous = {k: v for k, v in previous.items() if k not in (
            'micro_batch', 'active_parameters', 'normalization_sha256', 'cache_identity')}
        if _resume_identity(previous) != _resume_identity(protocol):
            raise ValueError('Task output protocol mismatch; use a new output_dir')
    # One RAM bound across main process plus all worker-local readers.
    cache = _FeatureCache(job['cache_dir'], cache_identity, job['disk_cache_bytes'],
                          job['ram_cache_bytes'] // (job['num_workers'] + 1)) if job['freeze'] else None
    clean_inputs, train_inputs, val_inputs = (_Inputs(clean, cache, 'train'), _Inputs(train, cache, 'train'),
                                             _Inputs(val, cache if job['task'] == 'ef' else None, 'val'))
    resumed = torch.load(last_path, map_location='cpu', weights_only=False) if last_path.exists() else None
    if resumed and resumed['identity'] != _resume_identity(protocol):
        raise ValueError('Task resume protocol mismatch; use a new checkpoint_dir')
    complete = _completed_metrics(output, checkpoints, _resume_identity(protocol), resumed)
    if complete is not None:
        return complete
    _json(output / 'evaluation_population.json', evaluation_population)
    if resumed:
        normalization = resumed['normalization']
    else:
        normalization = _fit_normalization(backbone.to(device), clean_inputs, job, device, cache)
    _json(output / 'normalization.json', normalization)
    model = _TaskModel(backbone, job, normalization).to(device)
    calibration = [clean_inputs[0] if job['freeze'] else _calibration_sample(clean, manifest, arguments, job['task'])]
    if job['freeze']:
        _features(backbone, calibration, job, device, cache)
        calibration = [{k: v for k, v in sample.items() if k not in ('video', 'frame_indices')}
                       for sample in calibration]
    # Active detection needs real pixels even when calibration has a cached descriptor.
    active = _active_parameters(model, [_calibration_sample(clean, manifest, arguments, job['task'])], job, device)
    if resumed:
        micro, tune_report = resumed['micro_batch'], resumed['autotune']
        if job.get('micro_batch') is not None and int(job['micro_batch']) != micro:
            raise ValueError('Exact resume requires the saved micro_batch')
        _restore_task_state(model, resumed['model_state_dict'], active)
    else:
        micro, tune_report = _autotune(model, calibration, job, normalization, device)
    requested_workers = job['num_workers']
    previous_worker_tune = tune_report.get('worker_tuning')
    if resumed and previous_worker_tune and previous_worker_tune['requested_workers'] == requested_workers:
        workers = previous_worker_tune['selected_workers']
        worker_report = previous_worker_tune
        if cache:
            cache.set_ram_budget(job['ram_cache_bytes'] // (workers + 1))
    else:
        workers, worker_report = _tune_workers(train_inputs, job, micro)
    tune_report = dict(tune_report, worker_tuning=worker_report)
    job['num_workers'] = workers
    protocol['runtime'].update(workers=workers, requested_workers=requested_workers,
                               ram_cache_bytes_per_process=job['ram_cache_bytes'] // (workers + 1))
    protocol.update(micro_batch=micro, active_parameters=active, autotune=tune_report,
                    normalization_sha256=_digest(normalization), cache_identity=cache_identity)
    # The early guard excludes fields learned once from calibration/resume.
    identity = _resume_identity({k: v for k, v in protocol.items() if k not in (
        'micro_batch', 'active_parameters', 'normalization_sha256', 'cache_identity')})
    if resumed and active != resumed['active_parameters']:
        raise ValueError('Active backbone modules changed across resume')
    _json(output / 'protocol.json', protocol)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=job['lr'],
                                 weight_decay=job['weight_decay'])
    updates_per_epoch = math.ceil(len(train) / job['effective_batch'])
    warmup, total = job['warmup_epochs'] * updates_per_epoch, job['epochs'] * updates_per_epoch

    def multiplier(step):
        if step < warmup:
            return (step + 1) / warmup
        return .5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = _scaler(job, device)
    progress = dict(epoch=0, next_update=0, global_step=0, bad_epochs=0, best_metric=None, best_epoch=None,
                    best_predictions=[], best_metrics={}, history=[], train_loss_sum=0.0, train_samples=0,
                    skipped_updates=0, data_wait_seconds=0.0, train_seconds=0.0, finished=False)
    if resumed:
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
        scheduler.load_state_dict(resumed['scheduler_state_dict'])
        scaler.load_state_dict(resumed['scaler_state_dict'])
        progress.update(resumed['progress'])
        set_rng_state(resumed['rng_state'])
        _reconcile_best(best_path, pending_best, progress, identity)
    boundary_rng = get_rng_state()
    safe = True
    setup_seconds = time.perf_counter() - started
    _log(output / 'logs' / 'train.log', f'{"resume" if resumed else "start"} task={job["task"]} '
         f'freeze={job["freeze"]} micro={micro} effective={job["effective_batch"]} active={len(active)}')
    (output / 'DONE').unlink(missing_ok=True)
    writer = _tensorboard(output, job)
    bar = None
    loader = iterator = None

    def save_last():
        atomic_torch_save(dict(version=2, identity=identity, model_state_dict=_task_state(model, active),
                               optimizer_state_dict=optimizer.state_dict(), scheduler_state_dict=scheduler.state_dict(),
                               scaler_state_dict=scaler.state_dict(), rng_state=boundary_rng,
                               progress=copy.deepcopy(progress), normalization=normalization, micro_batch=micro,
                               active_parameters=active, autotune=tune_report, config=config,
                               source_checkpoint=str(Path(job['checkpoint']).resolve())), last_path,
                          min_free_gb=job['min_free_gb'])

    try:
        if not resumed:
            save_last()
        while progress['epoch'] < job['epochs'] and not progress['finished']:
            epoch = progress['epoch']
            train.set_epoch(epoch)
            _train_mode(model, job)
            optimizer.zero_grad(set_to_none=True)
            loader = _loader(train_inputs, micro, job['effective_batch'], seed, epoch,
                             progress['next_update'], True, job['num_workers'])
            iterator = iter(loader)
            pending_loss = pending_count = 0
            pending_monitor = 0.0
            update_started = time.perf_counter()
            update_wait = 0.0
            bar = tqdm(total=updates_per_epoch, initial=progress['next_update'],
                       desc=f'{job["task"]} {"Frozen" if job["freeze"] else "FT"} epoch {epoch + 1}/{job["epochs"]}',
                       unit='update', dynamic_ncols=True, disable=bool(job.get('quiet', False)))
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            while True:
                waiting = time.perf_counter()
                try:
                    samples = next(iterator)
                except StopIteration:
                    break
                update_wait += time.perf_counter() - waiting
                with _amp(job, device):
                    values, slots = _features(model.backbone, samples, job, device, cache)
                    prediction = model.read(values, slots)
                    loss = _loss(prediction, samples, job['task'], normalization)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite task loss')
                size = samples[0]['update_size']
                scaler.scale(loss * len(samples) / size).backward()
                pending_loss += float(loss.detach()) * len(samples)
                pending_count += len(samples)
                if not progress['history']:
                    pending_monitor += _training_monitor(prediction, samples, job['task'], normalization) * len(samples)
                if not samples[0]['update_last']:
                    continue
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                if not scaler.is_enabled() and not bool(torch.isfinite(grad_norm)):
                    raise FloatingPointError('Nonfinite task gradients')
                # An interruption inside AdamW may mutate only part of the model.
                # Never overwrite the last durable boundary in that narrow region.
                safe = False
                before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                succeeded = scaler.get_scale() >= before
                if succeeded:
                    scheduler.step()
                    progress['global_step'] += 1
                    progress['train_loss_sum'] += pending_loss
                    progress['train_samples'] += pending_count
                else:
                    progress['skipped_updates'] += 1
                optimizer.zero_grad(set_to_none=True)
                progress['next_update'] = samples[0]['update_cursor'] + 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                progress['data_wait_seconds'] += update_wait
                step_seconds = time.perf_counter() - update_started
                progress['train_seconds'] += step_seconds
                boundary_rng = get_rng_state()
                safe = True
                last_val = progress['history'][-1]['monitor'] if progress['history'] else None
                monitor_value = last_val if last_val is not None else pending_monitor / pending_count
                monitor_name = ('MAE_val_pp' if last_val is not None else 'MAE_train_pp') if job['task'] == 'ef' else (
                    'Dice_val' if last_val is not None else 'Dice_train')
                bar.update(1)
                bar.set_postfix(loss=f'{pending_loss / pending_count:.4f}',
                                **{monitor_name: f'{monitor_value:.4f}'},
                                data=f'{update_wait:.3f}s', step=f'{step_seconds:.3f}s',
                                lr=f'{optimizer.param_groups[0]["lr"]:.2e}',
                                GPU=f'{torch.cuda.memory_allocated(device) / 1024 ** 3:.2f}G'
                                if device.type == 'cuda' else '0.00G', stepped=succeeded)
                if writer and succeeded:
                    writer.add_scalar('train/update_loss', pending_loss / pending_count, progress['global_step'])
                    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], progress['global_step'])
                pending_loss = pending_count = 0
                pending_monitor = 0.0
                update_wait = 0.0
                update_started = time.perf_counter()
            bar.close()
            bar = None
            # Retire persistent readers before validation or a new epoch starts.
            iterator = loader = None
            gc.collect()
            if not progress['train_samples']:
                raise RuntimeError('Epoch had no successful optimizer updates')
            validation, patients = _evaluate(model, val_inputs, job, normalization, device, cache, micro)
            metric = validation['mae_pp'] if job['task'] == 'ef' else validation['patient_dice']
            improved = (progress['best_metric'] is None or
                        (metric < progress['best_metric'] if job['task'] == 'ef' else metric > progress['best_metric']))
            safe = False
            if improved:
                progress.update(best_metric=metric, best_epoch=epoch + 1, best_predictions=patients,
                                best_metrics=validation, bad_epochs=0)
                atomic_torch_save(dict(version=2, identity=identity, model_state_dict=_task_state(model, active),
                                       epoch=epoch + 1, metric=metric, normalization=normalization,
                                       active_parameters=active, config=config, protocol=protocol,
                                       source_checkpoint=str(Path(job['checkpoint']).resolve())), pending_best,
                                  min_free_gb=job['min_free_gb'])
                _csv(output / 'patient_predictions.csv', patients)
            else:
                progress['bad_epochs'] += 1
            row = dict(epoch=epoch + 1, train_loss=progress['train_loss_sum'] / progress['train_samples'],
                       val_loss=validation['val_loss'], monitor=metric, best_metric=progress['best_metric'],
                       lr=optimizer.param_groups[0]['lr'], successful_updates=progress['global_step'],
                       skipped_updates=progress['skipped_updates'], train_samples=progress['train_samples'],
                       train_seconds=progress['train_seconds'], data_wait_seconds=progress['data_wait_seconds'],
                       compute_seconds=max(0.0, progress['train_seconds'] - progress['data_wait_seconds']),
                       evaluation_seconds=validation['evaluation_seconds'],
                       samples_per_second=progress['train_samples'] / max(progress['train_seconds'], 1e-9),
                       **_memory(device))
            progress['history'].append(row)
            progress.update(epoch=epoch + 1, next_update=0, train_loss_sum=0.0, train_samples=0,
                            skipped_updates=0, data_wait_seconds=0.0, train_seconds=0.0,
                            finished=epoch + 1 >= job['epochs'] or progress['bad_epochs'] >= job['patience'])
            boundary_rng = get_rng_state()
            save_last()
            _reconcile_best(best_path, pending_best, progress, identity)
            safe = True
            _publish_history(output, progress['history'])
            if writer:
                for key in ('train_loss', 'val_loss', 'monitor', 'lr'):
                    writer.add_scalar(f'epoch/{key}', row[key], epoch + 1)
                writer.flush()
            _log(output / 'logs' / 'train.log', f'epoch={epoch + 1} monitor={metric:.6g} '
                 f'best={progress["best_metric"]:.6g} successful_updates={progress["global_step"]}')
    except (KeyboardInterrupt, Exception) as exc:
        if bar is not None:
            bar.close()
        optimizer.zero_grad(set_to_none=True)
        if safe:
            set_rng_state(boundary_rng)
            save_last()
        _log(output / 'logs' / 'train.log', f'interrupted {type(exc).__name__}; '
             f'{"saved optimizer boundary" if safe else "kept previous durable checkpoint"}')
        raise
    finally:
        iterator = loader = None
        gc.collect()
        if writer:
            writer.close()
    if not progress['best_predictions'] or not best_path.exists():
        raise RuntimeError('Training finished without a durable observed best')
    _csv(output / 'patient_predictions.csv', progress['best_predictions'])
    _publish_history(output, progress['history'])
    metrics = dict(progress['best_metrics'], task=job['task'], freeze=job['freeze'],
                   best_epoch=progress['best_epoch'], best_metric=progress['best_metric'],
                   epochs_completed=progress['epoch'], successful_updates=progress['global_step'],
                   stopped_early=progress['epoch'] < job['epochs'], micro_batch=micro,
                   effective_batch=job['effective_batch'], protocol_sha256=identity,
                   checkpoint_dir=str(checkpoints.resolve()), runtime=protocol['runtime'],
                   best_checkpoint=str(best_path.resolve()), last_checkpoint=str(last_path.resolve()),
                   normalization_path=str((output / 'normalization.json').resolve()),
                   patient_predictions_path=str((output / 'patient_predictions.csv').resolve()),
                   protocol_path=str(protocol_path.resolve()),
                   evaluation_population_sha256=evaluation_sha256,
                   evaluation_population_provenance=dict(
                       path=str((output / 'evaluation_population.json').resolve()), split='val', dataset_seed=job['dataset_seed'],
                       records=len(evaluation_population['records']),
                       source_provenance=evaluation_population['source_provenance']),
                   source_checkpoint=str(Path(job['checkpoint']).resolve()),
                   checkpoint_format='task_active_state; reconstruct backbone from source_checkpoint',
                   setup_seconds=setup_seconds, run_wall_seconds=time.perf_counter() - started,
                   **_memory(device),
                   total_train_seconds=sum(r['train_seconds'] for r in progress['history']),
                   total_evaluation_seconds=sum(r['evaluation_seconds'] for r in progress['history']),
                   cache=dict(identity=cache_identity, disk_budget_bytes=job['disk_cache_bytes'],
                              ram_budget_bytes=job['ram_cache_bytes'], hits=cache.hits if cache else 0,
                              misses=cache.misses if cache else 0, writes=cache.writes if cache else 0))
    metrics.update(dict(mae=metrics['mae_pp']) if job['task'] == 'ef'
                   else dict(dice_patient_mean=metrics['patient_dice']))
    _json(output / 'metrics.json', metrics)
    _json(output / 'DONE', dict(protocol_sha256=identity, artifacts={name: _file_digest(output / name) for name in (
        'metrics.json', 'patient_predictions.csv', 'normalization.json', 'protocol.json', 'evaluation_population.json',
        'logs/metrics.csv', 'logs/metrics.jsonl', 'plots/loss_latest.png')},
        checkpoints={name: _file_digest(checkpoints / name) for name in ('last.pt', 'best.pt')}))
    _log(output / 'logs' / 'train.log', f'DONE best_epoch={progress["best_epoch"]} best={progress["best_metric"]:.6g}')
    return metrics
