"""Discarded train-only timing workloads; budgets are locked before selection."""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from models.final_temporal_mae import load_final_model
from utils.final_temporal_data import WindowDataset
from utils.final_temporal_training import (run_warm_job, write_json, guard_job, complete_job,
                                          job_protocol, file_digest)
from utils.final_temporal_tasks import (_resolve_job, _TaskModel, _active_parameters, _features,
                                       _loss, _amp, _scaler, _train_mode)
from utils.seed import get_rng_state, set_rng_state, seed_everything


def _task_timing(job, manifest, device, task, frozen):
    device = torch.device(device)
    rng = get_rng_state()
    backbone = model = optimizer = scaler = parameters = values = prediction = loss = samples = batch = None

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    try:
        seed_everything(int(job.get('seed', 42)))
        overrides = dict(gradient_checkpointing=True)
        overrides.update(job.get('overrides') or {})
        backbone = load_final_model(job['checkpoint'], overrides, job.get('seed', 42))[0].to(device)
        resolved = _resolve_job(dict(job, task=task, freeze=frozen, micro_batch=1), backbone)
        dataset = WindowDataset(manifest, 'train', task=task, recent_frames=resolved['recent_frames'],
                                local_frames=backbone.local_frames, max_prefix=resolved['max_prefix'],
                                seed=resolved['dataset_seed'], training=False, prefix=resolved['prefix'])
        if not len(dataset):
            raise ValueError('No real TRAIN labels available for timing')
        tick = time.perf_counter()
        sample = dataset[0]
        data_seconds = time.perf_counter() - tick
        cases = {r['patient']: dataset.cases[r['patient']] for r in dataset.records}
        targets = np.asarray([case['ef'] for case in cases.values()], dtype=np.float64) if task == 'ef' else None
        normalization = dict(feature_mean=[0.] * backbone.embed_dim, feature_std=[1.] * backbone.embed_dim,
                             target_mean=float(targets.mean()) if task == 'ef' else 0.,
                             target_std=(float(targets.std()) if targets.std() > 1e-6 else 1.) if task == 'ef' else 1.)
        model = _TaskModel(backbone, resolved, normalization).to(device)
        active = _active_parameters(model, [sample], resolved, device)
        model.eval()
        samples = [dict(sample)]
        with torch.no_grad():
            synchronize()
            tick = time.perf_counter()
            values, _ = _features(backbone, samples, dict(resolved, freeze=True), device)
            synchronize()
            extract_time = time.perf_counter() - tick
        # Frozen measurements use precisely the cached-head path, not encoder work.
        if frozen:
            samples = [{key: value for key, value in samples[0].items() if key not in ('video', 'frame_indices')}]
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=resolved['lr'], weight_decay=resolved['weight_decay'])
        scaler = _scaler(resolved, device)
        _train_mode(model, resolved)
        timings, successes, attempts = [], 0, 0
        iterations = 2 if job.get('smoke') else 20
        warmup = 2
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        while successes < warmup + iterations and attempts < 4 * (warmup + iterations) + 8:
            optimizer.zero_grad(set_to_none=True)
            synchronize()
            tick = time.perf_counter()
            with _amp(resolved, device):
                values, slots = _features(backbone, samples, resolved, device)
                prediction = model.read(values, slots)
                loss = _loss(prediction, samples, task, normalization)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite timing workload loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.)
            if not scaler.is_enabled() and not bool(torch.isfinite(norm)):
                raise FloatingPointError('Nonfinite timing workload gradients')
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            synchronize()
            seconds = time.perf_counter() - tick
            attempts += 1
            if scaler.get_scale() >= before:
                successes += 1
                if successes > warmup:
                    timings.append(seconds)
            values = prediction = loss = None
        if len(timings) != iterations:
            raise FloatingPointError('Timing workload did not complete the required successful updates')
        model.eval()
        evaluation = []
        # Still TRAIN inputs/labels: measure the validation computation, not a score.
        with torch.no_grad():
            for _ in range(2 if job.get('smoke') else 5):
                batch = [dict(sample)]
                synchronize()
                tick = time.perf_counter()
                with _amp(resolved, device):
                    values, slots = _features(backbone, batch, resolved, device)
                    prediction = model.read(values, slots)
                    loss = _loss(prediction, batch, task, normalization)
                synchronize()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError('Nonfinite timing-only evaluation')
                evaluation.append(time.perf_counter() - tick)
                batch = values = prediction = loss = None
        return dict(task=task, freeze=frozen, measured_successful_updates=len(timings), warmup_successful_updates=warmup,
                    skipped_updates=attempts - successes, measurement_batch_size=1, measurement_effective_batch=1,
                    single_sample_update_seconds=float(np.median(timings)),
                    encoder_extraction_seconds_per_sample=extract_time,
                    evaluation_seconds_per_sample=float(np.median(evaluation)), data_wait_seconds_per_sample=data_seconds,
                    labels_split='train', trial_weights_discarded=True, validation_scores_consulted=False,
                    gradient_checkpointing=bool(backbone.gradient_checkpointing),
                    precision=resolved['precision'] if device.type == 'cuda' else 'fp32',
                    head=type(model.head).__name__, active_parameters=active,
                    peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                    note='Actual B=1 forward/backward and full evaluation workload; identity feature normalization for timing only')
    finally:
        if model is not None:
            model.zero_grad(set_to_none=True)
        # Release graph/optimizer owners before collecting or emptying CUDA caches,
        # including when a timing workload raises and its traceback survives.
        backbone = model = optimizer = scaler = parameters = values = prediction = loss = samples = batch = None
        set_rng_state(rng)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()


def run_preflight_job(job, manifest, device):
    device = torch.device(device); out = Path(job['output_dir'])
    protocol = job_protocol(job, manifest)
    protocol.update(preflight_code=file_digest(__file__), task_trainer_code=file_digest(
        Path(__file__).with_name('final_temporal_tasks.py')))
    if guard_job(out, protocol):
        return json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    overrides = dict(gradient_checkpointing=True, reconstruction_recent_frames=job.get('recent_frames', 64))
    overrides.update(job.get('overrides') or {})
    warm = dict(job, kind='adapt', output_dir=str(out / 'warm'),
                checkpoint_dir=str(Path(job['checkpoint_dir']) / 'warm'),
                updates=2 if job.get('smoke') else 20, effective_batch=2 if job.get('smoke') else 32,
                measurement_only=True, micro_batch=1, autotune=False, overrides=overrides)
    if Path(job['checkpoint']).resolve() in {
            (Path(warm['checkpoint_dir']) / name).resolve() for name in ('last.pt', 'final.pt')}:
        raise ValueError('Measurement checkpoint paths cannot overwrite the source')
    rng = get_rng_state()
    try:
        adaptation = run_warm_job(warm, manifest, device)
        adaptation = dict(adaptation, measurement_micro_batch=1, validation_scores_consulted=False)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        tasks = [_task_timing(job, manifest, device, task, freeze) for freeze in (True, False) for task in ('ef', 'seg')]
    finally:
        set_rng_state(rng)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        for name in ('last.pt', 'final.pt'):
            (Path(warm['checkpoint_dir']) / name).unlink(missing_ok=True)
        (Path(warm['output_dir']) / 'DONE').unlink(missing_ok=True)
    records = {}
    for task in ('ef', 'seg'):
        records[task] = {split:len(WindowDataset(manifest, split, task=task,
                            recent_frames=job.get('recent_frames', 64), local_frames=manifest['local_frames'],
                            max_prefix=job.get('max_prefix', 128), positions='balanced' if split == 'train' else 'all'))
                         for split in ('train', 'val')}
    metrics = dict(measurement_only=True, adaptation=adaptation, task_timings=tasks, task_records=records,
                   validation_scores_consulted=False, measurement_micro_batch=1,
                   notes=['No validation metric consulted; measurement weights never become a study source.',
                          'Wall estimates include conservative B1 compute; cache/I/O and early stopping change actual duration.'])
    write_json(out / 'metrics.json', metrics); complete_job(out, files=('metrics.json', 'protocol.json'))
    return metrics


def estimate_budget(preflight, updates=1500, frozen_epochs=60, ft_epochs=80, optional_ft=True, second_seed=True):
    def positive(value, label):
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f'Missing/invalid measured {label}; cannot estimate a zero-cost budget')
        return float(value)

    for count in (updates, frozen_epochs, ft_epochs):
        if int(count) != count or count < 1:
            raise ValueError('Future update/epoch budgets must be positive integers')
    warm_seconds = positive(preflight['adaptation'].get('optimizer_update_seconds_median'), 'MAE update time') * updates
    timing = {(x['task'], x['freeze']):x for x in preflight['task_timings']}
    if len(timing) != 4 or len(preflight['task_timings']) != 4:
        raise ValueError('Require one timing record per task/adaptation mode')
    totals = {}
    # 3 historical checkpoints + 7 mandatory adaptations; bounded branches add 2.
    for frozen, models, epochs in ((True, 10, frozen_epochs), (False, 3, ft_epochs)):
        seconds = 0.
        for task in ('ef', 'seg'):
            counts = preflight['task_records'][task]; speed = timing[(task, frozen)]
            for split in ('train', 'val'):
                if int(counts[split]) != counts[split] or counts[split] < 1:
                    raise ValueError('Budget requires positive eligible TRAIN/VAL sample counts')
            step = positive(speed.get('single_sample_update_seconds'), 'task update time')
            extraction = positive(speed.get('encoder_extraction_seconds_per_sample'), 'encoder extraction time')
            # Older records lack a full evaluation measurement: charge extraction
            # plus a training step, not an assumed cache-hit discount.
            evaluation_cost = positive(speed.get('evaluation_seconds_per_sample', extraction + step), 'evaluation time')
            data = float(speed.get('data_wait_seconds_per_sample', 0))
            if not math.isfinite(data) or data < 0:
                raise ValueError('Invalid measured data wait')
            train = counts['train'] * epochs * (step + data)
            evaluation = counts['val'] * epochs * (evaluation_cost + data)
            setup = (counts['train'] + counts['val']) * (extraction + data)
            seconds += models * (train + evaluation + setup)
        totals['frozen_tasks' if frozen else 'calibration_finetuning'] = seconds
    totals['mandatory_adaptation'] = warm_seconds * 7
    optional = warm_seconds * 2 + totals['frozen_tasks'] / 10 * 2
    if optional_ft:
        optional += totals['calibration_finetuning'] / 3 * 2
    if second_seed:
        optional += totals['frozen_tasks'] / 10 * 2
    mandatory = sum(totals.values())
    totals['conditionals_and_confirmations'] = optional
    # Extra branch/linear/history/stream diagnostics have a declared accounting margin.
    totals['diagnostic_and_io_margin'] = sum(totals.values()) * .15
    return dict(components_seconds=totals, mandatory_hours=mandatory * 1.15 / 3600,
                worst_case_hours=sum(totals.values()) / 3600, updates=updates,
                frozen_epochs=frozen_epochs, ft_epochs=ft_epochs, optional_ft=bool(optional_ft), second_seed=bool(second_seed),
                validation_scores_consulted=False, assumes_cache_hits=False, measurement_batch_size=1,
                note='Measured, conservative planning estimate, not a completion ETA or guaranteed GPU maximum')
