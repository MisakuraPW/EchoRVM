"""Read-only, train-only Q1.4 recurrence and reconstruction-path diagnostics.

Norms and gate statistics describe numerical behavior, not clinical semantics or
the fraction of history used. No rank objective, optimizer, or weight update is
introduced. Only the single gradient probe may shorten its real prefix on OOM.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import copy
import csv
import gc
import hashlib
import io
import json
import math
from pathlib import Path
import time

import torch
from tqdm import tqdm

from models.temporal_mae import temporal_mask
from utils.final_temporal_data import WindowDataset, _load_manifest
from utils.final_temporal_training import native_video, write_json, guard_job, complete_job, file_digest
from utils.seed import get_rng_state, set_rng_state, seed_everything


_ARTIFACTS = ('protocol.json', 'state_traces.csv', 'gradients.json', 'metrics.json')
_CONTRACT = ('memory_mode', 'memory_slots', 'memory_grid', 'memory_compression',
             'memory_write_source', 'memory_read_location', 'frame_readout', 'soft_beta',
             'dynamic_orthogonal_weight', 'mask_ratio', 'research_mask', 'norm_pix_loss',
             'img_size', 'local_frames', 'patch_size', 'tubelet_size', 'embed_dim',
             'in_chans', 'token_grid', 'reconstruction_recent_frames')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def _finite_value(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _model_hash(model):
    digest = hashlib.sha256()
    values = list(model.named_parameters()) + list(model.named_buffers())
    for name, value in sorted(values):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@contextmanager
def _preserve(model):
    modes = [(module, module.training) for module in model.modules()]
    flags = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    checkpointing = model.gradient_checkpointing
    suffix = model.reconstruction_recent_frames
    rng = get_rng_state()
    try:
        yield
    finally:
        for module, training in modes:
            module.training = training
        for parameter, enabled in flags:
            parameter.requires_grad_(enabled)
        model.gradient_checkpointing = checkpointing
        model.reconstruction_recent_frames = suffix
        set_rng_state(rng)


def _config(config, model):
    cfg = copy.deepcopy(config or {})
    smoke = bool(cfg.setdefault('smoke', False))
    cfg.setdefault('seed', 42)
    cfg.setdefault('recent_frames', 64)
    cfg.setdefault('max_prefix', 128)
    cfg.setdefault('num_windows', 2 if smoke else 32)
    cfg.setdefault('saturation_epsilon', .01)
    cfg.setdefault('prediction_atol', 1e-6)
    cfg.setdefault('prediction_rtol', 1e-5)
    for key in ('seed', 'recent_frames', 'max_prefix', 'num_windows'):
        if int(cfg[key]) != cfg[key]:
            raise ValueError(f'{key} must be an integer')
        cfg[key] = int(cfg[key])
    if cfg['num_windows'] != (2 if smoke else 32):
        raise ValueError('Q1.4 fixes 32 TRAIN windows (2 for smoke)')
    if (cfg['recent_frames'] < model.local_frames or cfg['recent_frames'] % model.local_frames
            or not 0 <= cfg['max_prefix'] <= 128 or cfg['max_prefix'] % model.local_frames):
        raise ValueError('Require real complete clips and max_prefix <= 128')
    if not smoke and (model.img_size != 112 or model.local_frames != 16 or cfg['recent_frames'] != 64):
        raise ValueError('Formal Q1.4 requires native112/L16/W64; tiny dimensions require smoke=True')
    if not math.isfinite(cfg['saturation_epsilon']) or not 0 < cfg['saturation_epsilon'] < .5:
        raise ValueError('saturation_epsilon must lie in (0,.5)')
    for name in ('prediction_atol', 'prediction_rtol'):
        if not math.isfinite(cfg[name]) or cfg[name] < 0:
            raise ValueError('Prediction tolerances must be finite and nonnegative')
    return cfg


def _plan(manifest, model, cfg):
    recent, local, maximum = cfg['recent_frames'], model.local_frames, cfg['max_prefix']
    cases = [case for case in manifest['train'] if case['frames'] >= recent]
    if not cases:
        raise ValueError('Q1.4 requires real eligible TRAIN windows')

    def available(case):
        return min(maximum, (case['frames'] - recent) // local * local)

    cases.sort(key=lambda case: (available(case), _digest([cfg['seed'], case['patient']])))
    records = []
    count = cfg['num_windows']
    for index in range(count):
        case = cases[round(index * (len(cases) - 1) / (count - 1))]
        capacity = available(case)
        prefix = (0, capacity, capacity // (2 * local) * local, min(local, capacity))[index % 4]
        key = int(_digest([cfg['seed'], 'Q1.4-window', index, case['patient']])[:16], 16)
        start = prefix + key % (case['frames'] - recent - prefix + 1)
        records.append(dict(patient=case['patient'], recent_start=start, H=prefix))
    longest = max(cases, key=lambda case: (available(case), case['frames'],
                                         _digest([cfg['seed'], case['patient']])))
    h = available(longest)
    representative = dict(patient=longest['patient'], recent_start=h, H=h)
    return records, representative


def _dataset(manifest, records, model, cfg):
    return WindowDataset(manifest, 'train', task='mae', recent_frames=cfg['recent_frames'],
                         local_frames=model.local_frames, max_prefix=cfg['max_prefix'],
                         seed=cfg['seed'], training=False, prefix=None, records=records)


def _sources(manifest, records):
    cases = {case['patient']: case for case in manifest['train']}
    result = []
    for patient in sorted({record['patient'] for record in records}):
        case = cases[patient]
        path = Path(case['path'])
        stat = path.stat()
        if stat.st_size != case['source_bytes'] or stat.st_mtime_ns != case['source_mtime_ns']:
            raise ValueError('Source changed after TRAIN manifest creation')
        result.append(dict(patient=patient, path=str(path.resolve()), bytes=stat.st_size,
                           mtime_ns=stat.st_mtime_ns, sha256=file_digest(path)))
    return result


def _fixed_masks(model, clips, seed, strategy=None):
    """Generate on CPU so masking never depends on CUDA/global RNG state."""
    strategy = strategy or model.research_mask
    generator = torch.Generator().manual_seed(seed)
    spatial = model.token_grid[1] * model.token_grid[2]
    order = torch.rand(1, spatial, generator=generator).argsort(-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return torch.stack([temporal_mask(1, model.token_grid, model.mask_ratio, strategy,
                                          'cpu', index, order)[0] for index in range(clips)])[None]


def _video(sample, model, cfg, device):
    video = native_video(sample['video'][None].to(device=device, dtype=torch.float32), model, cfg['smoke'])
    if model.in_chans == 1:
        video = video[:, :, :1]
    elif video.shape[2] == 1 and model.in_chans == 3:
        video = video.expand(-1, -1, 3, -1, -1)
    return video


def _summary(value):
    value = value.detach().float().cpu()
    finite = bool(torch.isfinite(value).all())
    if not finite:
        return dict(finite=False, l2=None, rms=None, max_abs=None, slot_variance=None)
    return dict(finite=True, l2=float(value.double().norm()), rms=float(value.double().square().mean().sqrt()),
                max_abs=float(value.abs().max()),
                slot_variance=float(value.double().var(1, unbiased=False).mean()) if value.ndim == 3 else None)


def _gate(value, epsilon):
    gate = value.detach().float().cpu().sigmoid()
    if not bool(torch.isfinite(gate).all()):
        return dict(finite=False, low_saturation=None, high_saturation=None,
                    **{name: None for name in ('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max')})
    quantiles = torch.quantile(gate.flatten(), torch.tensor([0., .05, .25, .5, .75, .95, 1.])).tolist()
    return dict(finite=True, low_saturation=float((gate <= epsilon).float().mean()),
                high_saturation=float((gate >= 1 - epsilon).float().mean()),
                **dict(zip(('min', 'p05', 'p25', 'median', 'p75', 'p95', 'max'), quantiles)))


class _StateRecorder:
    """Observe actual forward outputs; never approximate or replace the RVM."""

    def __init__(self, model, sample, index, mode, cfg):
        self.model, self.sample, self.index, self.mode, self.cfg = model, sample, index, mode, cfg
        self.handles, self.rows, self.current = [], [], {}

    def __enter__(self):
        if self.model.memory_mode == 'none':
            return self
        memory = self.model.memory
        self.handles.append(memory.register_forward_pre_hook(self._before))
        for name in ('update_x', 'update_s', 'reset_x', 'reset_s', 'norm', 'candidate_up'):
            if hasattr(memory, name):
                self.handles.append(getattr(memory, name).register_forward_hook(self._capture(name)))
        self.handles.append(memory.register_forward_hook(self._after))
        return self

    def _before(self, module, args):
        tokens, state = args
        self.current = dict(previous=torch.zeros_like(tokens) if state is None else state, pooled=tokens)

    def _capture(self, name):
        def capture(module, args, output):
            self.current[name] = output
        return capture

    def _metadata(self, clip):
        local, sample = self.model.local_frames, self.sample
        return dict(window=self.index, patient=sample['patient'], mode=self.mode,
                    prefix_frames=int(sample['prefix_frames']), recent_start=int(sample['recent_start']),
                    clip_index=clip, source_start=int(sample['frame_indices'][clip * local]),
                    source_end=int(sample['frame_indices'][(clip + 1) * local - 1]),
                    is_prefix=clip * local < sample['prefix_frames'], memory_mode=self.model.memory_mode,
                    state_slots=self.model.memory_slots, embed_dim=self.model.embed_dim)

    def _after(self, module, args, output):
        state = output[1]
        candidate = self.current.get('candidate_up', self.current['norm'])
        row = self._metadata(len(self.rows))
        for name, value in (('state', state), ('previous', self.current['previous']),
                            ('pooled', self.current['pooled']), ('candidate', candidate),
                            ('candidate_pre_bottleneck', self.current['norm']),
                            ('update', state - self.current['previous'])):
            row.update({f'{name}_{key}': item for key, item in _summary(value).items()})
        for name in ('update', 'reset'):
            row.update({f'{name}_gate_{key}': item for key, item in _gate(
                self.current[name + '_x'] + self.current[name + '_s'], self.cfg['saturation_epsilon']).items()})
        self.rows.append(row)
        self.current = {}

    def no_memory(self, result):
        for clip, descriptor in enumerate(result['states'].unbind(1)):
            row = self._metadata(clip)
            row.update(memory_applicable=False, numerical_finite=bool(torch.isfinite(descriptor).all()),
                       no_memory_descriptor_l2=_summary(descriptor)['l2'])
            self.rows.append(row)

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.current = {}


def _write_traces(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix('.csv.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8')
    temporary.replace(path)


def _content_loss(result):
    weight = result['mask'].float()
    return (((result['pred'].float() - result['target'].detach().float()).square().mean(-1) * weight).sum()
            / weight.sum())


def _parameters(model):
    result = []
    for name, parameter in model.named_parameters():
        if (name.startswith(('memory.', 'frame_expansion.', 'compression_score.', 'feature_fusion.'))
                or name in ('frame_gamma', 'memory_type')):
            result.append((name, parameter))
    return result


def _gradient_stats(value):
    if value is None:
        return dict(connected=False, finite=True, l2=0., max_abs=0., nonzero_elements=0)
    value = value.detach().float()
    finite = bool(torch.isfinite(value).all())
    return dict(connected=True, finite=finite, l2=float(value.double().norm()) if finite else None,
                max_abs=float(value.abs().max()) if finite else None,
                nonzero_elements=int(torch.count_nonzero(value)))


def _group_norm(records, predicate):
    selected = [record for name, record in records.items() if predicate(name)]
    finite = all(record['finite'] for record in selected)
    return dict(parameters=len(selected), connected_parameters=sum(record['connected'] for record in selected),
                finite=finite, l2=math.sqrt(sum(record['l2'] ** 2 for record in selected)) if finite else None,
                nonzero_parameters=sum(record['nonzero_elements'] > 0 for record in selected))


def _orthogonal_gradients(model, names, reconstruction_gradients):
    if model.frame_readout not in ('factorized', 'soft_factorized'):
        return dict(applicable=False, reason='frame module has no registered orthogonal loss')
    frame = [(name, parameter) for name, parameter in names if name.startswith('frame_expansion.')]
    parameters = [parameter for _, parameter in frame]
    orth = model.frame_expansion.orthogonal_loss()
    weight = float(model.dynamic_orthogonal_weight)
    raw = torch.autograd.grad(orth, parameters, allow_unused=True, retain_graph=True)
    weighted = torch.autograd.grad(weight * orth, parameters, allow_unused=True)
    raw_stats = {name: _gradient_stats(value) for (name, _), value in zip(frame, raw)}
    weighted_stats = {name: _gradient_stats(value) for (name, _), value in zip(frame, weighted)}
    recon_stats = {name: _gradient_stats(reconstruction_gradients.get(name)) for name, _ in frame}
    raw_norm = _group_norm(raw_stats, lambda _: True)
    weighted_norm = _group_norm(weighted_stats, lambda _: True)
    recon_norm = _group_norm(recon_stats, lambda _: True)
    dot = 0.0
    for (name, _), value in zip(frame, weighted):
        content = reconstruction_gradients.get(name)
        if value is not None and content is not None:
            dot += float((value.detach().double() * content.double().to(value.device)).sum())
    denominator = (weighted_norm['l2'] or 0) * (recon_norm['l2'] or 0)
    cosine = max(-1., min(1., dot / denominator)) if denominator and math.isfinite(dot) else None
    return dict(applicable=True, loss=_finite_value(orth.detach()), loss_finite=bool(torch.isfinite(orth)), coefficient=weight,
                weighted_loss=_finite_value((weight * orth).detach()), raw_gradient=raw_norm,
                weighted_gradient=weighted_norm, reconstruction_frame_gradient=recon_norm,
                weighted_to_reconstruction_ratio=weighted_norm['l2'] / recon_norm['l2']
                if recon_norm['l2'] and weighted_norm['l2'] is not None else None,
                weighted_vs_reconstruction_cosine=cosine, raw_parameters=raw_stats,
                weighted_parameters=weighted_stats,
                interpretation='Actual weighted gradient measured; a small scalar orth loss does not imply no gradient')


def _gradient_once(model, sample, cfg, device, actual_h, full_masks):
    requested = int(sample['prefix_frames'])
    trim = requested - actual_h
    cropped = dict(sample, video=sample['video'][trim:])
    video = _video(cropped, model, cfg, device).detach().requires_grad_(True)
    masks = full_masks[:, trim // model.local_frames:].to(device)
    names = _parameters(model)
    result = model(video, masks=masks)
    content = _content_loss(result)
    gradients = torch.autograd.grad(content, [video] + [parameter for _, parameter in names], allow_unused=True)
    input_gradient = gradients[0]
    # CPU copies release the reconstruction graph before the independent regularizer probe.
    parameter_gradients = {name: None if value is None else value.detach().cpu()
                           for (name, _), value in zip(names, gradients[1:])}
    records = {name: _gradient_stats(value) for name, value in parameter_gradients.items()}
    clips = []
    for start in range(0, actual_h, model.local_frames):
        value = None if input_gradient is None else input_gradient[:, start:start + model.local_frames]
        clips.append(dict(clip_index=start // model.local_frames,
                          source_start=int(sample['frame_indices'][trim + start]),
                          source_end=int(sample['frame_indices'][trim + start + model.local_frames - 1]),
                          lag_frames_from_recent_start=actual_h - start, **_gradient_stats(value)))
    loss_value = float(content.detach())
    input_stats = _gradient_stats(input_gradient)
    del result, content, gradients, input_gradient, video
    orth = _orthogonal_gradients(model, names, parameter_gradients)
    groups = dict(memory=_group_norm(records, lambda name: name.startswith('memory.')),
                  memory_update_gates=_group_norm(records, lambda name: name.startswith(('memory.update_x.', 'memory.update_s.'))),
                  memory_reset_gates=_group_norm(records, lambda name: name.startswith(('memory.reset_x.', 'memory.reset_s.'))),
                  memory_candidate=_group_norm(records, lambda name: name.startswith(
                      ('memory.integration.', 'memory.norm.', 'memory.candidate_down.', 'memory.candidate_up.'))),
                  frame=_group_norm(records, lambda name: name.startswith('frame_expansion.') or name == 'frame_gamma'),
                  compression=_group_norm(records, lambda name: name.startswith('compression_score.')),
                  feature_fusion=_group_norm(records, lambda name: name.startswith('feature_fusion.')))
    return dict(content_loss=_finite_value(loss_value), loss_finite=math.isfinite(loss_value), input_gradient=input_stats,
                prefix_clips=clips, parameters=records, groups=groups, orthogonal=orth)


def _gradient_probe(model, sample, cfg, device):
    requested = int(sample['prefix_frames'])
    clips = len(sample['video']) // model.local_frames
    masks = _fixed_masks(model, clips, cfg['seed'] + 7001)
    attempts, actual = [], requested
    model.train()
    model.gradient_checkpointing = True
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    while True:
        seed_everything(cfg['seed'] + 7001)
        try:
            with torch.enable_grad(), torch.autocast(device.type, enabled=False):
                measured = _gradient_once(model, sample, cfg, device, actual, masks)
            attempts.append(dict(prefix_frames=actual, status='verified'))
            return dict(measured, status='verified' if actual == requested else 'verified_reduced_prefix',
                        requested_prefix_frames=requested, actual_prefix_frames=actual,
                        requested_total_frames=requested + cfg['recent_frames'],
                        actual_total_frames=actual + cfg['recent_frames'],
                        unverified_prefix_frames=requested - actual, prefix_coverage_complete=actual == requested,
                        prefix_applicable=actual > 0, no_prefix_reason='H0 has no old prefix; not a failure' if actual == 0 else None,
                        patient=sample['patient'], recent_start=int(sample['recent_start']), attempts=attempts,
                        precision='fp32', gradient_checkpointing=True,
                        loss_definition='masked recent reconstruction only; target detached; orth excluded')
        except torch.cuda.OutOfMemoryError:
            attempts.append(dict(prefix_frames=actual, status='oom_unverified'))
        # Run after the exception traceback has released the failed graph.
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        if actual == 0:
            return dict(status='oom_unverified', requested_prefix_frames=requested, actual_prefix_frames=None,
                        actual_total_frames=None, unverified_prefix_frames=requested,
                        prefix_coverage_complete=False, patient=sample['patient'], attempts=attempts,
                        precision='fp32', gradient_checkpointing=True)
        actual = actual // (2 * model.local_frames) * model.local_frames


def _hidden_pixels(mask, model):
    batch, clips, _ = mask.shape
    gt, gh, gw = model.token_grid
    hidden = mask.reshape(batch, clips * gt, gh, gw)
    hidden = hidden.repeat_interleave(model.tubelet_size, 1)
    hidden = hidden.repeat_interleave(model.patch_size, 2).repeat_interleave(model.patch_size, 3)
    return hidden[:, :, None]


@torch.no_grad()
def _hidden_isolation(model, sample, cfg, device):
    model.eval()
    video = _video(sample, model, cfg, device)
    masks = _fixed_masks(model, len(sample['video']) // model.local_frames, cfg['seed'] + 9001, 'tube').to(device)
    hidden = _hidden_pixels(masks, model).expand_as(video)
    changed = torch.where(hidden, (video + .37).remainder(1.), video)
    with torch.autocast(device.type, enabled=False):
        before = model(video, masks=masks)
        after = model(changed, masks=masks)
    finite = bool(torch.isfinite(before['pred']).all() and torch.isfinite(after['pred']).all())
    delta = (before['pred'].float() - after['pred'].float()).abs()
    return dict(mask_strategy='fixed_tube', patient=sample['patient'], prefix_frames=int(sample['prefix_frames']),
                total_frames=len(sample['video']), changed_pixel_elements=int(torch.count_nonzero(changed - video)),
                visible_pixel_max_difference=float((changed - video)[~hidden].abs().max()),
                only_fully_hidden_tubelet_patches_changed=bool(torch.equal(changed[~hidden], video[~hidden])),
                prediction_finite=finite, prediction_max_abs_difference=float(delta.max()) if finite else None,
                prediction_invariant=finite and bool(torch.allclose(before['pred'], after['pred'],
                    atol=cfg['prediction_atol'], rtol=cfg['prediction_rtol'])),
                target_max_abs_difference=_finite_value((before['target'] - after['target']).abs().max()),
                state_finite=bool(torch.isfinite(before['states']).all() and torch.isfinite(after['states']).all()))


@torch.no_grad()
def _future_causality(model, sample, cfg, device):
    model.eval()
    video = _video(sample, model, cfg, device)
    changed = video.clone()
    changed[:, model.local_frames:] = (changed[:, model.local_frames:] + .37).remainder(1.)

    def first_exits(sequence):
        state = None
        snapshot = None
        for clip in sequence.split(model.local_frames, 1):
            output = model.stream_clip(clip, state)
            state = output['final_state']
            if snapshot is None:
                snapshot = {key: None if output[key] is None else output[key].detach().cpu().clone()
                            for key in ('frame_outputs', 'frame_base_outputs', 'local_base_outputs', 'final_state')}
        return snapshot

    with torch.autocast(device.type, enabled=False):
        before, after = first_exits(video), first_exits(changed)
    finite = all(value is None or bool(torch.isfinite(value).all()) for value in (*before.values(), *after.values()))
    difference = max(float((before[key] - after[key]).abs().max()) for key in before if before[key] is not None) if finite else None
    invariant = finite and all(before[key] is None or torch.allclose(before[key], after[key],
                              atol=cfg['prediction_atol'], rtol=cfg['prediction_rtol']) for key in before)
    return dict(applicable=video.shape[1] > model.local_frames,
                first_clip_source_start=int(sample['frame_indices'][0]),
                first_clip_frames=model.local_frames, future_frames=video.shape[1] - model.local_frames,
                future_changed_elements=int(torch.count_nonzero(changed[:, model.local_frames:] - video[:, model.local_frames:])),
                first_clip_finite=finite, first_clip_invariant=bool(invariant), max_abs_difference=difference,
                protocol='Each stream_clip receives only its current complete clip; no future window tensor is passed',
                boundary='Within-clip attention remains bidirectional; this is inter-clip causality, not frame causality')


def run_mechanism_audit(model, manifest, output_dir, config, device):
    """Audit 32 fixed TRAIN windows (smoke2) and one real maximal prefix probe.

    Reuse requires identical model/data/code/protocol and intact DONE artifacts.
    A caller may be in eval/no_grad with frozen parameters; its modes, parameter
    flags, existing .grad tensors, checkpointing, suffix and RNG are preserved.
    """
    device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    manifest = _load_manifest(manifest)
    cfg = _config(config, model)
    records, representative = _plan(manifest, model, cfg)
    sources = _sources(manifest, records + [representative])
    source_hash = _model_hash(model)
    root = Path(__file__).resolve().parents[1]
    protocol = dict(version=2, audit='Q1.4', split='train', config=cfg, windows=records,
                    representative=representative, model_sha256=source_hash,
                    model_contract={name: getattr(model, name, None) for name in _CONTRACT},
                    model_class=f'{type(model).__module__}.{type(model).__qualname__}',
                    manifest_sha256=_digest(manifest), data_sha256=_digest(sources), source_files=sources,
                    precision='fp32', gradient_checkpointing=True, device_type=device.type,
                    code_sha256={name: file_digest(root / name) for name in (
                        'utils/final_temporal_mechanisms.py', 'models/final_temporal_mae.py',
                        'models/temporal_mae.py', 'models/rvm_core.py', 'models/frame_readout.py',
                        'models/video_mae.py', 'utils/final_temporal_data.py', 'utils/final_temporal_training.py')},
                    interpretation='Numerical/content-gradient audit only; no clinical-semantic or high-rank objective claim')
    # Persisted JSON turns tuples (notably token_grid) into lists.
    protocol = json.loads(json.dumps(protocol, allow_nan=False))
    output = Path(output_dir)
    if guard_job(output, protocol, done_files=_ARTIFACTS):
        return json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
    started = time.perf_counter()
    # Normal callers already supply FP32 on device. A clone protects other
    # dtype/device callers from precision conversion or parameter migration.
    tensors = list(model.parameters()) + list(model.buffers())
    compatible = all(value.device == device and (not value.is_floating_point() or value.dtype == torch.float32)
                     for value in tensors)
    audited = model if compatible else copy.deepcopy(model).to(device).float()
    rows = []
    with _preserve(audited):
        seed_everything(cfg['seed'])
        audited.eval()
        audited.gradient_checkpointing = True
        audited.reconstruction_recent_frames = cfg['recent_frames']
        dataset = _dataset(manifest, records, audited, cfg)
        for index in tqdm(range(len(dataset)), desc='Q1.4 TRAIN traces', disable=bool(cfg.get('quiet', False))):
            sample = dataset[index]
            video = _video(sample, audited, cfg, device)
            masks = _fixed_masks(audited, len(sample['video']) // audited.local_frames, cfg['seed'] + index).to(device)
            with torch.no_grad(), torch.autocast(device.type, enabled=False):
                for mode in ('masked', 'unmasked'):
                    with _StateRecorder(audited, sample, index, mode, cfg) as recorder:
                        result = audited(video, masks=masks) if mode == 'masked' else audited.diagnostic_features(video)
                        if audited.memory_mode == 'none':
                            recorder.no_memory(result)
                        rows.extend(recorder.rows)
                    del result
            del sample, video, masks
        sample = _dataset(manifest, [representative], audited, cfg)[0]
        gradients = _gradient_probe(audited, sample, cfg, device)
        isolation = _hidden_isolation(audited, sample, cfg, device)
        causality = _future_causality(audited, sample, cfg, device)
    unchanged = _model_hash(model) == source_hash
    if not unchanged:
        raise RuntimeError('Mechanism audit unexpectedly changed source weights or buffers')
    if _sources(manifest, records + [representative]) != sources:
        raise ValueError('TRAIN source changed during mechanism audit')
    trace_finite = all(all(value for name, value in row.items() if name.endswith('_finite')) for row in rows)
    grad_verified = gradients['status'].startswith('verified')
    gradient_finite = (gradients['loss_finite'] and gradients['input_gradient']['finite']
                       and all(item['finite'] for item in gradients['parameters'].values())) if grad_verified else None
    if grad_verified and gradients['orthogonal']['applicable']:
        orth = gradients['orthogonal']
        gradient_finite = gradient_finite and orth['loss_finite'] and orth['raw_gradient']['finite'] and orth['weighted_gradient']['finite']
    findings = []
    if not trace_finite:
        findings.append('nonfinite_state_or_gate_trace')
    if grad_verified and not gradient_finite:
        findings.append('nonfinite_content_loss_or_gradient')
    if not isolation['prediction_finite']:
        findings.append('hidden_pixel_invariance_unverified_nonfinite')
    elif not isolation['prediction_invariant']:
        findings.append('hidden_pixel_prediction_dependency')
    if not causality['first_clip_finite']:
        findings.append('future_causality_unverified_nonfinite')
    elif causality['applicable'] and not causality['first_clip_invariant']:
        findings.append('preceding_clip_future_dependency')
    if grad_verified and gradients['groups']['frame']['parameters'] and not gradients['groups']['frame']['nonzero_parameters']:
        findings.append('no_measured_frame_content_gradient')
    if grad_verified and gradients['actual_prefix_frames'] > 0 and audited.memory_mode != 'none':
        if not any(item['nonzero_elements'] for item in gradients['prefix_clips']):
            findings.append('no_measured_prefix_content_gradient')
        if not gradients['groups']['memory_update_gates']['nonzero_parameters']:
            findings.append('no_measured_memory_update_content_gradient')
    metrics = dict(audit='Q1.4', split='train', windows=len(records), unique_patients=len({r['patient'] for r in records}),
                   prefix_window_counts=dict(Counter(str(r['H']) for r in records)), state_trace_rows=len(rows),
                   memory_mode=audited.memory_mode, memory_slots=audited.memory_slots,
                   state_traces_finite=trace_finite, content_gradient_verified=grad_verified,
                   content_gradients_finite=gradient_finite, gradient_status=gradients['status'],
                   requested_gradient_prefix_frames=gradients['requested_prefix_frames'],
                   actual_gradient_prefix_frames=gradients['actual_prefix_frames'],
                   unverified_gradient_prefix_frames=gradients['unverified_prefix_frames'],
                   hidden_pixel_isolation=isolation, preceding_clip_future_causality=causality,
                   source_weights_unchanged=unchanged, technical_findings=findings,
                   protocol_sha256=_digest(protocol), model_sha256=source_hash, data_sha256=protocol['data_sha256'],
                   precision='fp32', gradient_checkpointing=True, wall_seconds=time.perf_counter() - started,
                   limitations=['Gate means do not measure the percentage of history used.',
                                'Finite gradients do not establish clinical semantics or useful long-range retention.',
                                'No high-rank objective or rank-based superiority is asserted.',
                                'Unavailable or OOM-reduced history is unverified, not synthesized.'])
    _write_traces(output / 'state_traces.csv', rows)
    write_json(output / 'gradients.json', gradients)
    write_json(output / 'metrics.json', metrics)
    complete_job(output, files=_ARTIFACTS)
    return metrics
