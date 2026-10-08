"""Frozen medical-motion and frame-content diagnostics on matched real windows."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.temporal_mae import TemporalMAE
from tools.diagnose_stage2_interfaces import save_json, file_hash, write_csv, make_loader
from tools.evaluate_representation_quality import load_model
from tools.evaluate_stage3 import tune_runtime
from tools.evaluate_temporal_mae import ridge_fit, ridge_apply
from tools.run_temporal_research import archive_analysis
from utils.dynamic_data import DynamicWindowDataset, build_dynamic_manifest
from utils.dynamic_latent import (trajectory_statistics, fit_motion_axis, motion_signal,
                                  detect_events, event_scores, pair_identity, safe_correlation, motion_agreement,
                                  clip_boundary_ratio)
from utils.seed import seed_everything


def evaluator_hash():
    files = [Path(__file__), ROOT/'utils/dynamic_data.py', ROOT/'utils/dynamic_latent.py',
             ROOT/'models/temporal_mae.py', ROOT/'models/frame_readout.py', ROOT/'models/video_mae.py',
             ROOT/'models/vit_blocks.py', ROOT/'models/rvm_core.py', ROOT/'utils/echo_input.py']
    files += [ROOT/'utils/datasets.py', ROOT/'utils/downstream_datasets.py',
              ROOT/'echo_aug_validation/io_utils.py', ROOT/'tools/evaluate_temporal_mae.py']
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()


@torch.inference_mode()
def trajectories(model, video, device):
    """Release local spatial maps immediately; retain only descriptors, not graphs."""
    columns = dict(local=[], fused=[], frame=[], cache=[])
    if model.memory_mode != 'none':
        columns.update(state=[], compressed=[])
    if model.frame_readout == 'factorized':
        columns.update(dynamic=[], structure=[])
    state = short = None
    recent = []
    for clip in video.split(model.local_frames, 1):
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            out = model.stream_clip(clip, state, short)
            local, fused = out['local_features'], out['features']
            columns['local'].append(local.mean(2).repeat_interleave(model.tubelet_size, 1))
            columns['fused'].append(fused.mean(2).repeat_interleave(model.tubelet_size, 1))
            frames = model.frame_features(fused).mean(2)
            columns['frame'].append(frames)
            recent.append(frames)
            recent = recent[-4:]
            columns['cache'].append(torch.cat(recent, 1).mean(1, keepdim=True))
            state, short = out['final_state'], out['final_short']
            if state is not None:
                columns['state'].append(state.mean(1, keepdim=True))
                source = local if model.memory_write_source == 'local' else fused
                valid = torch.ones(source.shape[:3], device=device)
                compressed = model._pool(source.flatten(1, 2), None, valid.flatten(1))
                columns['compressed'].append(compressed.mean(1, keepdim=True))
            if model.frame_readout == 'factorized':
                parts = model.frame_expansion.components(fused)
                columns['dynamic'].append(parts['coefficients'].mean(2))
                columns['structure'].append(parts['reference'].mean(2))
        del out, local, fused, clip
    return {k:torch.cat(v, 1).float().cpu().numpy() for k, v in columns.items()}


def branch_seconds(branch, record, length):
    if length == record['frames']:
        indices = np.arange(length)
    else:
        indices = np.arange(length) * 16 + 15
    return (record['start'] + indices) / record['fps']


def read_case(path):
    with np.load(path, allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files}


def extract_cases(model, records, split, args, device, cache, identity):
    cache.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    pending = []
    for i, record in enumerate(records):
        p = cache / (record['patient'] + '.npz')
        if p.exists():
            with np.load(p, allow_pickle=False) as z:
                if str(z['signature'].item()) != signature:
                    raise ValueError('Changed feature-cache protocol; use a fresh run_tag')
        else:
            pending.append((i, record))
    dataset = DynamicWindowDataset([r for _, r in pending], args.input_protocol, model.in_chans)
    progress = tqdm(make_loader(dataset, args), desc='dynamic extract ' + split)
    for batch in progress:
        video = batch['video'].to(device, non_blocking=True)
        begin = time.perf_counter()
        arrays = trajectories(model, video, device)
        images = F.adaptive_avg_pool2d(video.mean(2).flatten(0, 1)[:, None], 8).reshape(len(video), -1, 64)
        arrays['images'] = images.float().cpu().numpy()
        arrays['brightness'] = video.float().mean((2, 3, 4)).cpu().numpy()
        selected = [j for j,i in enumerate(batch['record_index'].tolist())
                    if split == 'val' and pending[i][0] < args.nuisance_cases]
        extras = {}
        if selected:
            subset = video[selected]
            transformed = trajectories(model, (subset * 1.1 + .03).clamp(0, 1), device)
            swapped = subset.reshape(len(subset), -1, 2, *subset.shape[2:]).flip(2).reshape_as(subset)
            exchange = trajectories(model, swapped, device)
            static = trajectories(model, subset[:, subset.shape[1]//2:subset.shape[1]//2+1].expand_as(subset), device)
            extras.update({'bright_' + k:v for k,v in transformed.items()})
            extras.update({'static_' + k:v for k,v in static.items()})
            extras['swap_frame'] = exchange['frame']
        for j, index in enumerate(batch['record_index'].tolist()):
            _, r = pending[index]
            item = {k:v[j] for k,v in arrays.items()}
            if j in selected:
                item.update({k:v[selected.index(j)] for k,v in extras.items()})
            if not all(np.isfinite(v).all() for v in item.values()):
                raise ValueError('Nonfinite extracted representation')
            path = cache / (r['patient'] + '.npz')
            with path.with_suffix('.tmp').open('wb') as f:
                np.savez_compressed(f, signature=signature, **item)
            path.with_suffix('.tmp').replace(path)
        progress.set_postfix(batch=args.batch_size, workers=args.num_workers,
                             seconds=f'{time.perf_counter()-begin:.2f}')


def plot_example(out, case, record, axes, index):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    branches = list(axes)
    fig, plot = plt.subplots(len(branches), 2, figsize=(10, max(3, len(branches) * 2.1)), squeeze=False)
    for i, branch in enumerate(branches):
        x = case[branch]
        t = branch_seconds(branch, record, len(x))
        centered = x - x.mean(0)
        u, s, _ = np.linalg.svd(centered, full_matrices=False)
        p = u[:, :2] * s[:2]
        if p.shape[1] == 2:
            plot[i, 0].plot(p[:, 0], p[:, 1], linewidth=.7)
            plot[i, 0].scatter(p[:, 0], p[:, 1], c=t, s=7, cmap='viridis')
        raw, filtered = motion_signal(x, t, axes[branch])
        plot[i, 1].plot(t, raw, alpha=.4, label='raw')
        plot[i, 1].plot(t, filtered, label='filtered')
        events = detect_events(filtered, t)
        for phase, color in (('ed', 'green'), ('es', 'red')):
            plot[i, 1].axvline(record[phase + '_frame'] / record['fps'], color=color, linestyle=':', label='traced '+phase.upper())
            indices = events[phase]
            plot[i, 1].scatter(t[indices], filtered[indices], color=color,
                               marker='^' if phase=='ed' else 'v', s=15)
        plot[i, 0].set(title=branch + ' within-video PCA', xlabel='PC1', ylabel='PC2')
        plot[i, 1].set(title='Training-fitted motion axis', xlabel='Source time (s)', ylabel=branch)
    plot[0, 1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / f'trajectory_{index:02d}.png', dpi=130)
    plt.close(fig)


def summarize_rows(rows):
    summaries = []
    for branch in sorted({r['branch'] for r in rows}):
        values = [r for r in rows if r['branch'] == branch]
        m = dict(branch=branch, patients=len(values))
        for field in ('ed_error_ms', 'es_error_ms', 'ed_within100ms', 'es_within100ms',
                      'ed_detected', 'es_detected', 'candidates_per_second', 'effective_rank',
                      'explained_variance_2', 'temporal_variance', 'collapsed', 'recurrence_score',
                      'pair_accuracy', 'brightness_delta_correlation', 'bright_motion_agreement',
                      'clip_boundary_ratio', 'source_boundary_ratio', 'max_emission_delay_ms',
                      'static_motion_energy_ratio', 'static_recurrence_score'):
            present = [r[field] for r in values if r.get(field) is not None]
            m[field] = float(np.mean(present)) if present else None
        m['ed_error_evaluated_patients'] = sum(r['ed_detected'] for r in values)
        m['es_error_evaluated_patients'] = sum(r['es_detected'] for r in values)
        m['static_control_evaluated_patients'] = sum(r.get('static_motion_energy_ratio') is not None for r in values)
        summaries.append(m)
    return summaries


def evaluate(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    out, cache = Path(args.output_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, meta = load_model(args.checkpoint, device)
    if not isinstance(model, TemporalMAE) or model.local_frames != 16 or model.frames != 64 or model.img_size != 112:
        raise ValueError('Dynamic audit requires the L16/T64/112px TemporalMAE protocol')
    model.requires_grad_(False).eval()
    model.gradient_checkpointing = False
    args.input_protocol = meta['input_protocol']
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    manifest_hash = file_hash(args.manifest)
    identity = dict(version=1, checkpoint_sha256=file_hash(args.checkpoint), evaluator_sha256=evaluator_hash(),
                    manifest_sha256=manifest_hash, seed=args.seed, nuisance_cases=args.nuisance_cases,
                    ridge_alpha=args.ridge_alpha, metadata=meta,
                    event_protocol='One shared training-fitted motion PCA axis; sign from training traced-area labels; no validation swap',
                    event_scope='Offline filtered trajectory; ED/ES are sparse area-derived proxies; no dense phase GT',
                    memory_scope='State/compressed descriptors are spatial means, not sufficient-state proofs')
    guard = out / 'protocol.json'
    if guard.exists() and json.loads(guard.read_text(encoding='utf-8')) != identity:
        raise ValueError('Audit protocol changed; use a new run_tag')
    save_json(guard, identity)
    if (out / 'DONE').exists():
        if not (out / 'metrics.json').exists():
            raise ValueError('DONE without metrics')
        return
    datasets = {s:DynamicWindowDataset(r, args.input_protocol, model.in_chans) for s, r in manifest['splits'].items()}
    def actual_trial(video):
        trajectories(model, video, device)
        if args.nuisance_cases:
            trajectories(model, (video * 1.1 + .03).clamp(0, 1), device)
            swapped = video.reshape(len(video), -1, 2, *video.shape[2:]).flip(2).reshape_as(video)
            trajectories(model, swapped, device)
            trajectories(model, video[:, video.shape[1]//2:video.shape[1]//2+1].expand_as(video), device)
    runtime = tune_runtime(model, datasets['train'], args, device, benchmark_fn=actual_trial)
    runtime['trial_path'] = 'Actual descriptors and optional nuisance passes; no labels or score search'
    runtime['software'] = dict(torch=str(torch.__version__), numpy=np.__version__, python=sys.version)
    runtime.update(recent_frames=args.recent_frames, cpu_threads=args.cpu_threads)
    save_json(out / 'runtime.json', runtime)
    start = time.perf_counter()
    for split in ('train', 'val'):
        extract_cases(model, manifest['splits'][split], split, args, device, cache / split, identity)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    train_records, val_records = manifest['splits']['train'], manifest['splits']['val']
    first = read_case(cache / 'train' / (train_records[0]['patient'] + '.npz'))
    branches = [k for k in first if k not in {'signature', 'images', 'brightness'}]
    axes, pixel_readouts = {}, {}
    for branch in tqdm(branches, desc='fit training-only diagnostics'):
        training, x, y = [], [], []
        for r in train_records:
            case = read_case(cache / 'train' / (r['patient'] + '.npz'))
            values = case[branch]
            training.append(dict(features=values, seconds=branch_seconds(branch, r, len(values)),
                                 ed_seconds=r['ed_frame'] / r['fps'], es_seconds=r['es_frame'] / r['fps']))
            if len(values) == r['frames']:
                x.append(values)
                y.append(case['images'])
        axes[branch] = fit_motion_axis(training)
        if x:
            pixel_readouts[branch] = ridge_fit(torch.from_numpy(np.concatenate(x)),
                                             torch.from_numpy(np.concatenate(y)), args.ridge_alpha)
    save_json(out / 'training_axes.json', {k:v.tolist() for k, v in axes.items()})
    rows, controls = [], []
    for index, r in enumerate(tqdm(val_records, desc='medical and identity scores')):
        case = read_case(cache / 'val' / (r['patient'] + '.npz'))
        for branch in branches:
            values = case[branch]
            t = branch_seconds(branch, r, len(values))
            _, signal = motion_signal(values, t, axes[branch])
            stats = trajectory_statistics(values, t)
            recurrence = stats.pop('recurrence')
            stats.update(recurrence_score=None if recurrence is None else recurrence['score'],
                         recurrence_lag_seconds=None if recurrence is None else recurrence['lag_seconds'])
            events = event_scores(detect_events(signal, t), t, r['ed_frame'] / r['fps'], r['es_frame'] / r['fps'], r['fps'])
            row = dict(id=r['patient'] + ':' + branch, patient=r['patient'], branch=branch,
                       fps=r['fps'], start=r['start'], frames=r['frames'],
                       **stats, **events, pair_accuracy=None, eligible_pairs=None,
                       excluded_near_identical_pairs=None, tied_fraction=None, mean_assignment_margin=None,
                       brightness_delta_correlation=None, bright_motion_agreement=None,
                       clip_boundary_ratio=None, source_boundary_ratio=None,
                       static_motion_energy_ratio=None, static_recurrence_score=None,
                       max_emission_delay_ms=15000/r['fps'] if len(values)==r['frames'] else 0.)
            if branch in pixel_readouts:
                prediction = ridge_apply(pixel_readouts[branch], torch.from_numpy(values)).numpy()
                row.update(pair_identity(prediction, case['images']))
                speed = np.linalg.norm(np.diff(values, axis=0), axis=1)
                row['brightness_delta_correlation'] = safe_correlation(speed, abs(np.diff(case['brightness'])))
                row['clip_boundary_ratio'] = clip_boundary_ratio(values)
                row['source_boundary_ratio'] = clip_boundary_ratio(case['images'])
            if 'bright_' + branch in case:
                row['bright_motion_agreement'] = motion_agreement(values, case['bright_' + branch])
            if 'static_' + branch in case:
                static = case['static_' + branch]
                energy = np.mean(np.diff(values, axis=0) ** 2)
                row['static_motion_energy_ratio'] = float(np.mean(np.diff(static, axis=0) ** 2) / energy) if energy > 1e-12 else None
                recurrence = trajectory_statistics(static, t)['recurrence']
                row['static_recurrence_score'] = None if recurrence is None else recurrence['score']
            rows.append(row)
        if 'swap_frame' in case:
            prediction = ridge_apply(pixel_readouts['frame'], torch.from_numpy(case['swap_frame'])).numpy()
            target = case['images'].reshape(-1, 2, 64)[:, ::-1].reshape(-1, 64)
            controls.append(dict(patient=r['patient'], **pair_identity(prediction, target)))
        if index < args.plot_cases:
            plot_example(out, case, r, axes, index)
        write_csv(out / 'patient_metrics.csv', rows)
        save_json(out / 'status.json', dict(stage='scoring', completed=index + 1, total=len(val_records)))
    summaries = summarize_rows(rows)
    write_csv(out / 'summary.csv', summaries)
    write_csv(out / 'within_tubelet_swap.csv', controls)
    save_json(out / 'metrics.json', dict(summary=summaries, wall_seconds=time.perf_counter() - start,
                                        parameters=meta['parameters'], frame_readout=cfg.get('frame_readout', 'repeat'),
                                        memory_compression=cfg.get('memory_compression', 'mean')))
    (out / 'DONE').write_text('Frozen matched medical diagnostics completed\n')
    archive_analysis(out)
    if not args.keep_cache:
        for split in ('train', 'val'):
            for r in manifest['splits'][split]:
                (cache / split / (r['patient'] + '.npz')).unlink()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'output_dir', 'cache_dir', 'manifest'):
        p.add_argument('--' + key, required=True)
    for key, value in dict(seed=42, batch_size=0, max_batch_size=32, num_workers=8, prefetch_factor=2,
                           cpu_threads=4, nuisance_cases=8, plot_cases=3, recent_frames=64).items():
        p.add_argument('--' + key, type=int, default=value)
    p.add_argument('--ridge_alpha', type=float, default=10.)
    p.add_argument('--auto_workers', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--keep_cache', action='store_true')
    args = p.parse_args()
    if min(args.max_batch_size, args.cpu_threads, args.ridge_alpha) <= 0 or min(args.batch_size, args.num_workers, args.nuisance_cases, args.plot_cases) < 0:
        p.error('Invalid runtime or diagnostic budget')
    evaluate(args)


if __name__ == '__main__':
    main()
