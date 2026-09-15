"""Cheap frozen EF screening and opt-in, history-aware segmentation probes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from tools.evaluate_temporal_mae import (
    load_model, TemporalEchoDataset, TemporalMAE, ef_features, ridge_fit, ridge_apply,
    regression_metrics, bootstrap_mean, ProbeSegDataset, seg_features, segmentation_probe, write_csv)
from utils.seed import seed_everything


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_output(output, identity):
    path = output / 'screen_protocol.json'
    if path.exists() and json.loads(path.read_text()) != identity:
        raise RuntimeError('Different screening protocol/checkpoint in this directory; use a separate output directory.')
    path.write_text(json.dumps(identity, indent=2), encoding='utf-8')


def validate_seg_context(model, args):
    if isinstance(model, TemporalMAE) and model.memory_mode != 'none':
        within = args.seg_target_index % model.frames
        if within < model.local_frames:
            raise ValueError('Target falls in first local clip after a memory reset. Choose a later target index.')


def quick_ef(model, args, device, output, identity):
    train = TemporalEchoDataset(args.data_root, 'train', args.audit_frames, channels=model.in_chans,
                               limit=args.ef_train_cases, seed=args.seed, random_start=False,
                               input_protocol=args.input_protocol)
    val = TemporalEchoDataset(args.data_root, 'val', args.audit_frames, channels=model.in_chans,
                             limit=args.ef_val_cases, seed=args.seed, random_start=False,
                             input_protocol=args.input_protocol)
    if set(train.ids) & set(val.ids):
        raise ValueError('Train/validation overlap')
    if max(args.ef_budgets) > len(train):
        raise ValueError('EF budget exceeds actual training cases')
    cache = Path(args.feature_cache) if args.feature_cache else output/'features.npz'
    signature = dict(identity, ef_train_ids=train.ids, ef_val_ids=val.ids)
    signature.pop('ef_budgets', None)
    signature = json.dumps(signature, sort_keys=True)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data['signature'].item()) != signature:
                raise RuntimeError('Feature cache does not match data/protocol/checkpoint. Use a new cache path.')
            arrays = {k: torch.from_numpy(data[k].copy()) for k in data.files if k != 'signature'}
        cache_hit = True
    else:
        tr, y, _, _, tc, _ = ef_features(model, train, args, device, reverse=False)
        va, yv, _, _, vc, _ = ef_features(model, val, args, device, reverse=False)
        arrays = dict(train_x=tr['normal'], val_x=va['normal'], train_y=y, val_y=yv,
                      train_lengths=torch.tensor(tc), val_lengths=torch.tensor(vc))
        if 'state' in tr:
            arrays.update(train_state=tr['state'], val_state=va['state'])
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix('.tmp')
        with temporary.open('wb') as handle:
            np.savez_compressed(handle, signature=np.asarray(signature),
                                **{k: v.cpu().numpy() for k, v in arrays.items()})
        temporary.replace(cache)
        cache_hit = False
    result, predictions = dict(ef={}, state_only_ef={}, cache_hit=cache_hit), []
    for n in args.ef_budgets:
        for mode, tx, vx, field in [('normal', 'train_x', 'val_x', 'ef'),
                                    ('state', 'train_state', 'val_state', 'state_only_ef')]:
            if tx not in arrays:
                continue
            pred = ridge_apply(ridge_fit(arrays[tx][:n], arrays['train_y'][:n]), arrays[vx])
            result[field][str(n)] = regression_metrics(pred, arrays['val_y'])
            result[field][str(n)]['mae_ci95'] = bootstrap_mean((pred-arrays['val_y']).abs().numpy(), args.seed)
            for i, case in enumerate(val.ids):
                predictions.append(dict(id=case, train_cases=n, mode=mode,
                                        target=float(arrays['val_y'][i]), prediction=float(pred[i]),
                                        valid_frames=int(arrays['val_lengths'][i])))
    length_pred = ridge_apply(ridge_fit(arrays['train_lengths'][:, None], arrays['train_y']),
                             arrays['val_lengths'][:, None])
    result['length_only_ef_control'] = regression_metrics(length_pred, arrays['val_y'])
    result['case_ids'] = dict(ef_train=train.ids, ef_val=val.ids)
    write_csv(output/'ef_predictions.csv', predictions)
    return result


def summarize_quick(root):
    rows = []
    for path in sorted(Path(root).glob('*/quick_audit/epoch_*/metrics.json')):
        if not (path.parent/'DONE').exists():
            continue
        m = json.loads(path.read_text())
        budget = max(map(int, m['ef']))
        seg_path = path.parents[2] / 'seg_history_v1' / path.parent.name / 'metrics.json'
        seg = json.loads(seg_path.read_text()) if seg_path.exists() and (seg_path.parent/'DONE').exists() else None
        rows.append(dict(method=path.parents[2].name, epoch=m['metadata']['epoch'],
                         profile=m['profile'], train_cases=budget,
                         ef_mae=m['ef'][str(budget)]['mae'],
                         state_ef_mae=m['state_only_ef'].get(str(budget), {}).get('mae'),
                         seg_dice=seg['segmentation']['dice_mean'] if seg else None,
                         seg_wall_seconds=seg['wall_seconds'] if seg else None,
                         wall_seconds=m['wall_seconds']))
    if rows:
        write_csv(Path(root)/'quick_comparison.csv', rows)
        lines = ['# Quick EF screening', '',
                 'Frozen validation probes; normal EF is primary, state EF is diagnostic. Dice is opt-in; NA means not run.', '',
                 '| Method | Epoch | EF MAE | State EF MAE | Optional Dice | EF seconds |', '|---|---:|---:|---:|---:|---:|']
        for row in rows:
            state = row['state_ef_mae']
            state = f'{state:.4f}' if state is not None else 'NA'
            dice = f"{row['seg_dice']:.4f}" if row['seg_dice'] is not None else 'NA'
            lines.append(f"| {row['method']} | {row['epoch']} | {row['ef_mae']:.4f} | {state} | {dice} | {row['wall_seconds']:.1f} |")
        (Path(root)/'quick_comparison.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        paired_quick(Path(root), rows)


def paired_quick(root, rows):
    comparisons = []
    available = {(r['method'], r['epoch']): r for r in rows}
    controls = dict(hier_global='clip_mae_pool64', hier_spatial='clip_mae_pool64',
                    hier_dual='clip_mae_pool64', frame_rvm64='frame_mae_pool64',
                    echocardmae_video_port='videomae_matched', tsf_frequency_memory='tsf_spatial_control')
    seg_comparisons = []
    for (method, epoch), row in available.items():
        control = controls.get(method)
        if (control, epoch) not in available:
            continue
        budget = row['train_cases']
        protocols, predictions, manifests = [], [], []
        for name in (method, control):
            path = root/name/'quick_audit'/f'epoch_{epoch:04d}'
            protocol = json.loads((path/'screen_protocol.json').read_text())
            protocol.pop('checkpoint_sha256')
            protocols.append(protocol)
            manifests.append(json.loads((path/'metrics.json').read_text())['case_ids'])
            with (path/'ef_predictions.csv').open() as handle:
                predictions.append({r['id']: r for r in csv.DictReader(handle)
                                    if r['mode'] == 'normal' and int(r['train_cases']) == budget})
        if protocols[0] != protocols[1] or manifests[0] != manifests[1]:
            raise RuntimeError('Quick paired comparison requires identical probe protocols and patient selections.')
        if not predictions[0] or predictions[0].keys() != predictions[1].keys():
            raise RuntimeError('Quick paired comparison requires matching validation IDs and budgets.')
        delta = []
        for case, a in predictions[0].items():
            b = predictions[1][case]
            if float(a['target']) != float(b['target']):
                raise RuntimeError('Paired EF targets differ.')
            delta.append(abs(float(a['prediction'])-float(a['target'])) -
                         abs(float(b['prediction'])-float(b['target'])))
        ci = bootstrap_mean(delta, seed=42, repetitions=10000)
        comparisons.append(dict(method=method, control=control, epoch=epoch, train_cases=budget,
                                mae_difference=ci['mean'], ci95_low=ci['low'], ci95_high=ci['high'],
                                cases=ci['n'], meaning='negative favors method; unadjusted patient bootstrap'))
        seg_pair = paired_seg(root, method, control, epoch)
        if seg_pair:
            seg_comparisons.append(seg_pair)
    write_csv(root/'quick_paired_comparisons.csv', comparisons)
    write_csv(root/'quick_paired_seg.csv', seg_comparisons)


def paired_seg(root, method, control, epoch):
    paths = [root/name/'seg_history_v1'/f'epoch_{epoch:04d}' for name in (method, control)]
    if not all((p/'DONE').exists() for p in paths):
        return None
    protocols, manifests, values = [], [], []
    for path in paths:
        protocol = json.loads((path/'screen_protocol.json').read_text())
        protocol.pop('checkpoint_sha256')
        protocols.append(protocol)
        manifests.append(json.loads((path/'metrics.json').read_text())['case_ids'])
        by_frame = {}
        with (path/'seg_predictions.csv').open() as handle:
            for row in csv.DictReader(handle):
                by_frame[row['id']] = float(row['dice'])
        values.append(by_frame)
    if protocols[0] != protocols[1] or manifests[0] != manifests[1] or values[0].keys() != values[1].keys():
        raise RuntimeError('Paired Dice requires identical protocols and labeled frames.')
    patients = {}
    for frame, dice in values[0].items():
        patients.setdefault(frame.rsplit(':', 1)[0], []).append(dice-values[1][frame])
    ci = bootstrap_mean([np.mean(x) for x in patients.values()], repetitions=10000)
    return dict(method=method, control=control, epoch=epoch, dice_difference=ci['mean'],
                ci95_low=ci['low'], ci95_high=ci['high'], patients=ci['n'],
                meaning='positive favors method; paired patient bootstrap, ED/ES averaged per patient')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--profile', choices=['ef', 'seg'], default='ef')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--audit_frames', type=int)
    parser.add_argument('--seg_target_index', type=int, default=48)
    parser.add_argument('--ef_train_cases', type=int, default=512)
    parser.add_argument('--ef_val_cases', type=int, default=256)
    parser.add_argument('--ef_budgets', nargs='+', type=int, default=[512])
    parser.add_argument('--seg_train_cases', type=int, default=64)
    parser.add_argument('--seg_val_cases', type=int, default=64)
    parser.add_argument('--seg_steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--feature_cache')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.audit_frames is None:
        args.audit_frames = 256 if args.profile == 'ef' else 64
    args.ef_budgets = sorted(set(args.ef_budgets))
    if min(args.batch_size, args.audit_frames, args.ef_train_cases, args.ef_val_cases,
           args.seg_train_cases, args.seg_val_cases, args.seg_steps, *args.ef_budgets) < 1:
        parser.error('Batch, frame counts, cases, budgets and steps must be positive.')
    if args.num_workers < 0:
        parser.error('num_workers must be nonnegative')
    if args.profile == 'seg' and not 0 <= args.seg_target_index < args.audit_frames:
        parser.error('seg_target_index must lie inside audit_frames')
    if args.smoke:
        args.ef_train_cases, args.ef_val_cases, args.ef_budgets = 8, 4, [8]
        args.seg_train_cases = args.seg_val_cases = 2
        args.seg_steps, args.num_workers = 2, 0
    begin = time.perf_counter()
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, metadata = load_model(args.checkpoint, device)
    model.requires_grad_(False).eval()
    seed_everything(args.seed)
    args.input_protocol = metadata['input_protocol']
    identity = {k: v for k, v in vars(args).items()
                if k not in {'output_dir', 'feature_cache', 'num_workers', 'checkpoint'}}
    identity.update(version=1, checkpoint_sha256=file_hash(args.checkpoint),
                    filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'))
    if args.profile == 'seg':
        identity['tracings_sha256'] = file_hash(Path(args.data_root)/'VolumeTracings.csv')
        validate_seg_context(model, args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    check_output(output, identity)
    if (output/'DONE').exists() and (output/'metrics.json').exists():
        print(f'Already completed: {output}')
        return
    if args.profile == 'ef':
        metrics = quick_ef(model, args, device, output, identity)
    else:
        args.skip_seg_auxiliary = True
        tr = seg_features(model, ProbeSegDataset(args.data_root, 'train', args, model.in_chans), args, device)
        va = seg_features(model, ProbeSegDataset(args.data_root, 'val', args, model.in_chans), args, device)
        if {x.rsplit(':', 1)[0] for x in tr[3]} & {x.rsplit(':', 1)[0] for x in va[3]}:
            raise ValueError('Segmentation train/validation overlap')
        values, predictions = segmentation_probe(tr, va, args, device)
        metrics = dict(segmentation=values, case_ids=dict(seg_train=tr[3], seg_val=va[3]))
        write_csv(output/'seg_predictions.csv', predictions)
    metrics.update(metadata=metadata, protocol=vars(args), profile=args.profile,
                   wall_seconds=time.perf_counter()-begin)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    (output/'DONE').write_text('screening complete\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
