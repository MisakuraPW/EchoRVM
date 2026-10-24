"""One training/evaluation loop for all adapters; epoch-boundary recovery."""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib
import inspect
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from utils.checkpoint import atomic_torch_save
from utils.seed import get_rng_state, seed_everything, set_rng_state
from .artifacts import code_identity, digest, event, file_hash, layout, mark_done, provenance, verify_done, write_json
from .cache import prepare_cache
from .metrics import linear_cka, representation_summary, stratify, summarize
from .runtime import loader, select_runtime


def build_experiment(cfg, source, device):
    module, name = cfg['adapter'].split(':', 1)
    factory = getattr(importlib.import_module(module), name)
    seed_everything(cfg['training']['seed'])
    return factory(cfg, cfg['stage'], source, device)


class Heartbeat:
    def __init__(self, output, interval):
        self.path, self.interval, self.last = output / 'status.json', interval, 0.

    def __call__(self, force=False, **value):
        if force or time.monotonic()-self.last >= self.interval:
            import socket
            write_json(self.path, dict(pid=os.getpid(), hostname=socket.gethostname(), time=time.time(), **value))
            self.last = time.monotonic()


def cache_identity(exp, cfg, source):
    metadata = {k:v for k,v in exp.input_digests().items() if k != 'head'}
    return dict(adapter=cfg['adapter'], model=cfg['model'], data=cfg['data'], metadata=metadata,
                source_sha256=file_hash(source) if source else None, seed=cfg['training']['seed'],
                task=exp.task, exit_name=cfg['stage'].get('exit_name', 'final'),
                append_boundary=cfg['stage'].get('append_boundary', False), code=code_identity(), fp32=True)


def prepare_features(exp, cfg, source, heartbeat):
    if cfg['stage'].get('mode', '').startswith('frozen') and cfg['evaluation']['cache_frozen']:
        key = cache_identity(exp, cfg, source)
        root = layout(cfg)['cache']
        directory = root / 'frozen' / digest(key)
        return prepare_cache(exp, directory, key,
                             int(cfg['runtime']['cache_gb']*1024**3), int(cfg['runtime']['min_free_gb']*1024**3),
                             heartbeat, batch_size=cfg['runtime']['micro_batch'], budget_root=root,
                             evict_completed=cfg['runtime']['evict_completed_feature_caches'])
    return None


def amp_context(cfg, device):
    precision = cfg['runtime']['precision']
    if device.type == 'cpu' and precision == 'fp16':
        raise ValueError('CPU fp16 training is unsupported; explicitly use fp32')
    if device.type == 'cuda' and precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('Requested bf16 is unsupported; choose an explicit supported precision')
    return torch.autocast(device.type, dtype=torch.bfloat16 if precision == 'bf16' else torch.float16,
                          enabled=precision != 'fp32')


def make_scaler(cfg, device):
    return torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and cfg['runtime']['precision'] == 'fp16', init_scale=128.)


def update_step(model, optimizer, scaler, clip_norm):
    scaler.unscale_(optimizer)
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, clip_norm if clip_norm else float('inf'))
    if not scaler.is_enabled() and not bool(torch.isfinite(norm)):
        raise FloatingPointError('Nonfinite gradients; no successful update recorded')
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return scaler.get_scale() >= before


def csv_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def publish_history(output, history, diagnostic=False):
    csv_rows(output / 'metrics.csv', history)
    if not history:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = [row['epoch'] for row in history]
    axes[0].plot(epochs, [row['train_loss'] for row in history], label='train', color='#0072B2', linestyle='-', marker='.')
    if any(row.get('val_loss') is not None for row in history):
        axes[0].plot(epochs, [row.get('val_loss', np.nan) for row in history], label='seen-data diagnostic' if diagnostic else 'validation', color='#D55E00', marker='o', linestyle='--')
    axes[0].set(xlabel='Epoch', ylabel='Loss'); axes[0].legend()
    key = next((k for k in ('val_mae_pp', 'val_dice_mean', 'val_reconstruction_loss') if any(k in row for row in history)), None)
    if key:
        axes[1].plot(epochs, [row.get(key, np.nan) for row in history], marker='.')
        axes[1].set(xlabel='Epoch', ylabel=key)
        if key == 'val_dice_mean':
            axes[1].set_ylim(-.02, 1.02)
    if len(epochs) == 1:
        for axis in axes:
            axis.set_xlim(epochs[0]-.5, epochs[0]+.5)
            axis.set_xticks(epochs)
    fig.tight_layout()
    fig.savefig(output / 'curves.png', dpi=150)
    plt.close(fig)


def evaluate(exp, cfg, selection, heartbeat, validation_loader=None):
    rng = get_rng_state()
    rows, loss_sum, count = [], 0., 0
    exp.model.eval()
    # A deterministic validation mask seed makes MAE loss diagnostic, not random mask difficulty.
    seed_everything(cfg['training']['seed'] + 9173)
    try:
        with torch.no_grad():
            for samples in validation_loader or loader(exp.val, cfg, selection, training=False):
                with amp_context(cfg, torch.device(cfg['runtime']['device'])):
                    loss, values = exp.evaluate_batch(samples, boundaries=cfg['evaluation']['boundaries'])
                    rows.extend(values)
                loss_sum += float(loss)*len(samples)
                count += len(samples)
                heartbeat(phase='validation', completed=count, total=len(exp.val))
        metrics, patients = summarize(rows, exp.task)
        if not math.isfinite(loss_sum):
            raise FloatingPointError('Nonfinite validation loss')
        return loss_sum/count, metrics, patients, rows
    finally:
        set_rng_state(rng)


def run_training(cfg, source):
    started = time.perf_counter()
    device = torch.device(cfg['runtime']['device'])
    paths = layout(cfg, cfg['stage']['id'])
    output, checkpoints = paths['result'], paths['ckpt']
    for directory in (output, checkpoints):
        directory.mkdir(parents=True, exist_ok=True)
    beat = Heartbeat(output, cfg['runtime']['heartbeat_seconds'])
    beat(force=True, phase='initializing')
    exp = build_experiment(cfg, source, device)
    module = importlib.import_module(cfg['adapter'].split(':')[0])
    identity = dict(version=1, adapter=cfg['adapter'], stage=cfg['stage'], data=cfg['data'], model=cfg['model'],
                    training=cfg['training'], evaluation=cfg['evaluation'], precision=cfg['runtime']['precision'],
                    source=file_hash(source) if source else None, inputs=exp.input_digests(),
                    code=code_identity([inspect.getfile(module)]))
    identity_sha = digest(identity)
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8'))['sha256'] != identity_sha:
        raise ValueError('Scientific/code/data protocol changed under the same run tag; use a new run')
    write_json(protocol_path, dict(sha256=identity_sha, protocol=identity, environment=provenance()))
    if verify_done(output):
        beat(force=True, phase='done', reused=True)
        return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    last_path = checkpoints / 'last.pt'
    resumed = torch.load(last_path, map_location='cpu', weights_only=False) if last_path.exists() else None
    if resumed and resumed['identity'] != identity_sha:
        raise ValueError('Checkpoint protocol mismatch')
    preview_samples = [exp.val[i] for i in range(min(2, len(exp.val)))]
    cache_report = prepare_features(exp, cfg, source, beat)
    if cache_report:
        write_json(output / 'cache.json', cache_report)
    selection = select_runtime(cfg, exp, output, source, identity_sha, beat)
    if resumed and resumed['selection'] != selection:
        raise ValueError('Exact epoch-boundary resume requires the saved execution grouping')
    exp.model.to(device)
    exp.set_checkpointing(selection['gradient_checkpointing'])
    params = [p for p in exp.model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No trainable parameters')
    train = cfg['training']
    optimizer = torch.optim.AdamW(params, lr=train['lr'], weight_decay=train['weight_decay'], betas=tuple(train['betas']))
    per_epoch = math.ceil(len(exp.train)/train['effective_batch'])
    total, warmup = train['epochs']*per_epoch, train['warmup_epochs']*per_epoch

    def schedule(step):
        if step < warmup:
            return (step+1)/max(1, warmup)
        return .5*(1+math.cos(math.pi*min(1., (step-warmup)/max(1, total-warmup))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    scaler = make_scaler(cfg, device)
    progress = dict(epoch=0, global_step=0, best=None, best_epoch=None, best_file=None,
                    bad_checks=0, history=[], stopped=False, active_names=[])
    if resumed:
        exp.model.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        scaler.load_state_dict(resumed['scaler'])
        progress = resumed['progress']
        set_rng_state(resumed['rng'])
        if progress['best_file'] and not (checkpoints / progress['best_file']).is_file():
            raise ValueError('Resume refers to a missing best checkpoint')
    diagnostic = bool(exp.metadata.get('diagnostic_overlap'))
    publish_history(output, progress['history'], diagnostic)
    writer = None
    if train['tensorboard']:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(output / 'tensorboard'), purge_step=progress['global_step']+1)
        except ImportError:
            event(output / 'events.jsonl', phase='warning', message='TensorBoard unavailable; CSV/JSON monitoring remains active')
    event(output / 'events.jsonl', phase='resume' if resumed else 'start', epoch=progress['epoch'], selection=selection)
    training_loader = validation_loader = None

    def weights(epoch):
        return dict(version=1, identity=identity_sha, model=exp.model.state_dict(),
                    epoch=epoch, source=source, **exp.model_export())

    def save_last():
        atomic_torch_save(dict(version=1, identity=identity_sha, model=exp.model.state_dict(),
                               optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                               rng=get_rng_state(), progress=copy.deepcopy(progress), selection=selection),
                          last_path, min_free_gb=cfg['runtime']['min_free_gb'])

    try:
        if 0 in train['save_epochs'] and not (checkpoints / 'epoch_0000.pt').exists():
            if progress['epoch']:
                raise ValueError('Initial checkpoint is missing; cannot reconstruct epoch 0 from trained weights')
            atomic_torch_save(weights(0), checkpoints / 'epoch_0000.pt', min_free_gb=cfg['runtime']['min_free_gb'])
        if exp.task == 'mae' and 0 in train['save_epochs'] and progress['epoch'] == 0 and not (output / 'initial_diagnostic.json').exists():
            loss, metrics, _, _ = evaluate(exp, cfg, selection, beat)
            write_json(output / 'initial_diagnostic.json', dict(epoch=0, loss=loss, metrics=metrics,
                       diagnostic_overlap=diagnostic, selection='never used to select or stop SSL'))
            with torch.no_grad():
                exp.preview(preview_samples, output / 'previews' / 'epoch_0000')
        if not resumed:
            save_last()
        while progress['epoch'] < train['epochs'] and not progress['stopped']:
            epoch = progress['epoch']
            if hasattr(exp.train, 'set_epoch'):
                exp.train.set_epoch(epoch)
            if training_loader is None:
                training_loader = loader(exp.train, cfg, selection, epoch)
                validation_loader = loader(exp.val, cfg, selection, training=False)
            else:
                training_loader.batch_sampler.epoch = epoch
                training_loader.generator.manual_seed(train['seed']+epoch)
            exp.train_mode()
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            optimizer.zero_grad(set_to_none=True)
            begin, seen, pending, loss_sum, wait_sum = time.perf_counter(), 0, 0, 0., 0.
            sampled_frames, sampled_clips, window_digest = 0, 0, hashlib.sha256()
            prefix_histogram = {}
            monitor_sum, pending_monitors = {}, {}
            pending_loss = None
            effective = train['effective_batch']
            window_size = min(effective, len(exp.train))
            iterator = iter(training_loader)
            while True:
                waiting = time.perf_counter()
                samples = next(iterator, None)
                wait_sum += time.perf_counter()-waiting
                if samples is None:
                    break
                for sample in samples:
                    if 'video' in sample and sample['video'].ndim == 4:
                        frames = len(sample['video'])
                        sampled_frames += frames
                        sampled_clips += frames//int(cfg['data'].get('local_frames', frames))
                    if 'frame_indices' in sample:
                        indices = sample['frame_indices'].tolist()
                        window_digest.update(digest([sample['patient'], indices, sample.get('prefix_frames', 0)]).encode())
                        key = str(sample.get('prefix_frames', 0))
                        prefix_histogram[key] = prefix_histogram.get(key, 0)+1
                with amp_context(cfg, device):
                    loss = exp.loss(samples)
                for name, metric in exp.last_metrics.items():
                    value = metric.detach()*len(samples)
                    pending_monitors[name] = pending_monitors.get(name, 0)+value
                scaler.scale(loss*len(samples)/window_size).backward()
                value = loss.detach()*len(samples)
                pending_loss = value if pending_loss is None else pending_loss+value
                pending += len(samples)
                seen += len(samples)
                if pending < window_size:
                    continue
                scalar = float(pending_loss)
                if not math.isfinite(scalar):
                    raise FloatingPointError('Nonfinite training loss; previous durable checkpoint retained')
                if not update_step(exp.model, optimizer, scaler, train['clip_norm']):
                    raise FloatingPointError('AMP skipped an optimizer update; abort rather than silently spend the scientific budget')
                active = {name for name, param in exp.model.named_parameters() if param.grad is not None}
                progress['active_names'] = sorted(set(progress['active_names']) | active)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                progress['global_step'] += 1
                loss_sum += scalar
                for name, value in pending_monitors.items():
                    monitor_sum[name] = monitor_sum.get(name, 0.)+float(value)
                pending_monitors = {}
                pending, pending_loss = 0, None
                window_size = min(effective, len(exp.train)-seen)
                elapsed = time.perf_counter()-begin
                beat(phase='training', epoch=epoch+1, epochs=train['epochs'], completed=seen, total=len(exp.train),
                     loss=loss_sum/seen, successful_updates=progress['global_step'], samples_per_second=seen/elapsed,
                     data_wait_fraction=wait_sum/elapsed,
                     allocated_bytes=torch.cuda.memory_allocated(device) if device.type == 'cuda' else 0,
                     reserved_bytes=torch.cuda.memory_reserved(device) if device.type == 'cuda' else 0)
            if pending or seen != len(exp.train):
                raise RuntimeError('Epoch sample accounting mismatch')
            train_seconds = time.perf_counter()-begin
            row = dict(epoch=epoch+1, train_loss=loss_sum/seen, train_seconds=train_seconds,
                       samples_per_second=seen/train_seconds, data_wait_seconds=wait_sum,
                       lr=optimizer.param_groups[0]['lr'], successful_updates=progress['global_step'])
            row.update(visited_windows=seen, observed_frames=sampled_frames, observed_clips=sampled_clips,
                       frames_per_second=sampled_frames/train_seconds, window_sequence_sha256=window_digest.hexdigest(),
                       prefix_histogram_json=json.dumps(prefix_histogram, sort_keys=True))
            if device.type == 'cuda':
                row.update(train_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           train_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            row.update({f'train_{name}': value/seen for name, value in monitor_sum.items()})
            improved, previous_best = False, progress['best_file']
            evaluated = (epoch+1) % train['eval_every'] == 0 or epoch+1 == train['epochs']
            if evaluated:
                val_started = time.perf_counter()
                val_loss, metrics, patients, observations = evaluate(exp, cfg, selection, beat, validation_loader)
                row.update(val_loss=val_loss, val_seconds=time.perf_counter()-val_started,
                           **{f'val_{k}': v for k, v in metrics.items() if isinstance(v, (int, float))})
                score = metrics['mae_pp'] if exp.task == 'ef' else -metrics['dice_mean'] if exp.task == 'seg' else val_loss
                improved = exp.task != 'mae' and (progress['best'] is None or score < progress['best']-train['min_delta'])
                if improved:
                    progress.update(best=score, best_epoch=epoch+1, best_file=f'best_epoch_{epoch+1:04d}.pt', bad_checks=0)
                    atomic_torch_save(weights(epoch+1), checkpoints / progress['best_file'], min_free_gb=cfg['runtime']['min_free_gb'])
                elif exp.task != 'mae':
                    progress['bad_checks'] += 1
                if exp.task != 'mae' and train['patience'] is not None and progress['bad_checks'] >= train['patience']:
                    progress['stopped'] = True
            progress['epoch'] = epoch+1
            progress['history'].append(row)
            # Commit recovery metadata before pruning any previous best it references.
            if improved or (epoch+1) % train['checkpoint_every'] == 0 or progress['stopped'] or epoch+1 == train['epochs']:
                beat(force=True, phase='checkpoint', epoch=epoch+1)
                save_last()
                if previous_best and previous_best != progress['best_file']:
                    (checkpoints / previous_best).unlink(missing_ok=True)
            if (epoch+1) % train['checkpoint_every'] == 0 or epoch+1 in train['save_epochs']:
                atomic_torch_save(weights(epoch+1), checkpoints / f'epoch_{epoch+1:04d}.pt', min_free_gb=cfg['runtime']['min_free_gb'])
                unpinned = [p for p in sorted(checkpoints.glob('epoch_*.pt')) if int(p.stem.split('_')[-1]) not in train['save_epochs']]
                for path in unpinned[:max(0, len(unpinned)-train['keep_snapshots'])]:
                    path.unlink()
            csv_rows(output / 'metrics.csv', progress['history'])
            if evaluated:
                publish_history(output, progress['history'], diagnostic)
                if exp.task == 'mae':
                    with torch.no_grad():
                        exp.preview(preview_samples, output / 'previews' / f'epoch_{epoch+1:04d}')
            event(output / 'events.jsonl', phase='epoch_completed', **row)
            print(json.dumps(dict(stage=cfg['stage']['id'], **row), ensure_ascii=False), flush=True)
            if writer:
                for key, value in row.items():
                    if isinstance(value, (float, int)):
                        writer.add_scalar(key, value, progress['global_step'])
        atomic_torch_save(weights(progress['epoch']), checkpoints / 'final.pt', min_free_gb=cfg['runtime']['min_free_gb'])
        selected_name = progress['best_file'] or 'final.pt'
        selected = torch.load(checkpoints / selected_name, map_location='cpu', weights_only=False)
        exp.model.load_state_dict(selected['model'])
        _, metrics, patients, observations = evaluate(exp, cfg, selection, beat, validation_loader)
        with torch.no_grad():
            exp.preview(preview_samples, output / 'previews')
        write_json(output / 'observations.json', observations)
        write_json(output / 'strata.json', stratify(observations, exp.task))
        write_json(output / 'patients.json', patients)
        csv_rows(output / 'patients.csv', patients)
        publish_history(output, progress['history'], diagnostic)
        summary = dict(stage=cfg['stage']['id'], kind=exp.task, mode=cfg['stage'].get('mode'),
                       metrics=metrics, epoch=progress['epoch'], selected_epoch=selected['epoch'],
                       stopped_early=progress['stopped'], selection='final fixed epoch' if exp.task == 'mae' else 'VAL only',
                       smoke=bool(cfg.get('smoke')),
                       synthetic=bool(exp.metadata.get('synthetic')), protocol_sha256=identity_sha,
                       model_parameters=sum(p.numel() for p in exp.model.parameters()),
                       trainable_parameters=sum(p.numel() for p in exp.model.parameters() if p.requires_grad),
                       parameters_with_observed_gradients=sum(p.numel() for name,p in exp.model.named_parameters()
                                                             if name in progress['active_names']),
                       parameter_count_scope='full adapter including inactive reconstruction decoder; observed gradients reported separately',
                       checkpoints={name: dict(path=str(checkpoints / name), sha256=file_hash(checkpoints / name),
                                               bytes=(checkpoints / name).stat().st_size)
                                    for name in {selected_name, 'final.pt', *[p.name for p in checkpoints.glob('epoch_*.pt')]}},
                       best_file=selected_name, wall_seconds=time.perf_counter()-started,
                       completed_train_seconds=sum(r['train_seconds'] for r in progress['history']),
                       completed_validation_seconds=sum(r.get('val_seconds', 0.) for r in progress['history']),
                       scope=exp.metadata.get('evaluation_scope', 'TRAIN/VAL development only; no TEST opened by this adapter'),
                       ssl_data_exposure=exp.metadata.get('ssl_data_exposure'),
                       diagnostic_overlap=diagnostic, cache=cache_report)
        write_json(output / 'summary.json', summary)
        mark_done(output, ['summary.json', 'protocol.json', 'metrics.csv', 'patients.json', 'observations.json', 'strata.json'])
        beat(force=True, phase='done', epoch=progress['epoch'], metrics=metrics)
        return summary
    except BaseException as exc:
        beat(force=True, phase='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
             error=repr(exc), resume='last committed epoch boundary; incomplete epoch is replayed')
        event(output / 'events.jsonl', phase='failed', error=repr(exc))
        raise
    finally:
        if writer:
            writer.close()
        training_loader = validation_loader = None
        for dataset in (exp.train, exp.val):
            if hasattr(dataset, 'close'):
                dataset.close()


def run_representation(cfg, source):
    paths = layout(cfg, cfg['stage']['id'])
    output = paths['result']; output.mkdir(parents=True, exist_ok=True)
    beat = Heartbeat(output, cfg['runtime']['heartbeat_seconds'])
    exp = build_experiment(cfg, source, torch.device(cfg['runtime']['device']))
    identity = digest(dict(features=cache_identity(exp, cfg, source), evaluation=cfg['evaluation'], stage=cfg['stage']))
    protocol = output / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))['sha256'] != identity:
        raise ValueError('Representation protocol changed')
    write_json(protocol, dict(sha256=identity, input=exp.metadata))
    if verify_done(output):
        return json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    exp.model.eval()
    rows, vectors, names, alternatives, timings = [], [], [], {}, []
    maximum = min(len(exp.val), cfg['evaluation']['representation_max_samples'])
    with torch.no_grad():
        for index in range(maximum):
            sample = exp.val[index]
            device = torch.device(cfg['runtime']['device'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            begin = time.perf_counter()
            feature = exp.features([sample]).float().cpu().numpy()[0]
            timings.append((time.perf_counter()-begin)*1000)
            # One vector per sample; pooling is explicit, not a frame-geometry claim.
            vector = feature.reshape(-1, feature.shape[-1]).mean(0)
            vectors.append(vector)
            names.append(dict(patient=str(sample['patient']), sample_id=sample.get('sample_id', str(index))))
            if cfg['evaluation']['representation_interventions'] and 'video' in sample and sample['video'].ndim == 4:
                video = sample['video']
                target = int(sample.get('target_index', -1))
                pivot = target if target >= 0 else len(video)//2
                interventions = dict(repeat_pivot=video[pivot:pivot+1].expand_as(video).clone())
                if target >= 0 and len(video) > 1:
                    replacement = video.clone()
                    replacement[target] = video[target-1 if target else 1]
                    interventions['target_substitution'] = replacement
                prefix = int(sample.get('prefix_frames', 0))
                if prefix:
                    changed = video.clone(); changed[:prefix] = video[:1]
                    interventions['repeat_history_only'] = changed
                for name, changed in interventions.items():
                    value = exp.features([dict(sample, video=changed)]).float().cpu().numpy()[0]
                    descriptor = value.reshape(-1, value.shape[-1]).mean(0)
                    alternatives.setdefault(name, []).append(descriptor)
                    rows.append(dict(sample_id=names[-1]['sample_id'], patient=names[-1]['patient'],
                                     intervention=name, l2=float(np.linalg.norm(descriptor-vector))))
            beat(phase='representation', completed=index+1, total=maximum)
    matrix = np.stack(vectors)
    summary = dict(stage=cfg['stage']['id'], kind='representation', metrics=representation_summary(matrix),
                   synthetic=bool(exp.metadata.get('synthetic')), scope='VAL bounded sample descriptor geometry; not task quality')
    summary['metrics'].update(descriptor_forward_ms_p50=float(np.percentile(timings, 50)),
                              descriptor_forward_ms_p95=float(np.percentile(timings, 95)),
                              timing_scope='single-sample descriptor extraction + transfer + CPU materialization; no data decode/acquisition; not clinical inference FPS')
    summary['interventions'] = {}
    for name, values in alternatives.items():
        if len(values) == len(matrix):
            summary['interventions'][name] = dict(linear_cka=linear_cka(matrix, np.stack(values)),
                                                  mean_l2=float(np.mean([r['l2'] for r in rows if r['intervention'] == name])),
                                                  interpretation='Counterfactual sensitivity; larger/smaller is not automatically better')
    paths['cache'].mkdir(parents=True, exist_ok=True)
    np.save(paths['cache'] / 'representation.npy', matrix)
    write_json(output / 'sample_order.json', names)
    write_json(output / 'interventions.json', rows)
    write_json(output / 'summary.json', summary)
    mark_done(output, ['summary.json', 'sample_order.json', 'protocol.json', 'interventions.json'])
    beat(force=True, phase='done')
    return summary


def profile_experiment(cfg, source, output):
    """Opt-in operator trace on throwaway weights, not a training continuation."""
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg['runtime']['device'])
    exp = build_experiment(cfg, source, device)
    selection = dict(micro_batch=cfg['runtime']['micro_batch'], num_workers=min(cfg['runtime']['worker_candidates']),
                     gradient_checkpointing=False)
    exp.train_mode()
    optimizer = torch.optim.AdamW([p for p in exp.model.parameters() if p.requires_grad], lr=cfg['training']['lr'])
    scaler = make_scaler(cfg, device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    data = loader(exp.train, cfg, selection)
    iterator = iter(data)
    times = []
    with torch.profiler.profile(activities=activities, record_shapes=True, profile_memory=True) as prof:
        for _ in range(3):
            begin = time.perf_counter()
            with torch.profiler.record_function('pipeline/data_wait'):
                samples = next(iterator, None)
                if samples is None:
                    iterator = iter(data); samples = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            with torch.profiler.record_function('pipeline/forward_loss'), amp_context(cfg, device):
                loss = exp.loss(samples)
            with torch.profiler.record_function('pipeline/backward'):
                scaler.scale(loss).backward()
            with torch.profiler.record_function('pipeline/optimizer'):
                update_step(exp.model, optimizer, scaler, cfg['training']['clip_norm'])
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            times.append(time.perf_counter()-begin)
            prof.step()
    prof.export_chrome_trace(str(output / 'trace.json'))
    (output / 'operators.txt').write_text(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=40), encoding='utf-8')
    write_json(output / 'profile.json', dict(selection=selection, step_seconds=times,
                                            scope='three disposable micro-batch forward/backward steps; profiler overhead included; not an inference speed claim'))
