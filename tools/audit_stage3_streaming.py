"""Bounded final checks on frozen models; no MAE training or test-set selection."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.temporal_mae import TemporalMAE
from tools.diagnose_stage2_interfaces import (
    SegViews, load_model, save_json, file_hash, write_csv, seg_probe)
from tools.evaluate_stage2 import extract, seg_batch, code_hash
from tools.evaluate_stage3 import tune_runtime
from tools.run_temporal_research import archive_analysis
from utils.seed import seed_everything
from utils.streaming_features import StreamingFeatureCache
from utils.temporal_data import TemporalEchoDataset


def resident_bytes(cache):
    values = [v for e in cache.entries for v in e.values() if torch.is_tensor(v)]
    values += [v for v in (cache.state, cache.short_state, cache.next_index)
               if torch.is_tensor(v)]
    stores = {v.untyped_storage().data_ptr(): v.untyped_storage().nbytes() for v in values}
    return sum(stores.values())


def expect_rejected(fn):
    try:
        fn()
    except ValueError:
        return True
    raise AssertionError('Invalid stream update was accepted')


@torch.inference_mode()
def inspect_stream(model, video, patient, fps):
    """Measure a real contiguous stream, with isolated FP32 correctness checks."""
    length, capacity = model.local_frames, model.frames // model.local_frames
    device = video.device
    sync = lambda: torch.cuda.synchronize(device) if device.type == 'cuda' else None
    cache = StreamingFeatureCache(model, capacity)
    storage, timings, state_norm, frame_pair_distance = [], [], [], []
    for start in range(0, video.shape[1], length):
        clip = video[:, start:start+length]
        idx = torch.arange(start, start+length, device=device)[None]
        sync()
        tick = time.perf_counter()
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            result = cache.update(clip, [patient], idx)
        sync()
        if start >= model.frames:
            timings.append((time.perf_counter()-tick)*1000)
        assert torch.isfinite(result['frame_features']).all()
        assert len(cache.entries) <= capacity
        assert result['source_indices'].shape[1] <= model.frames
        assert not result['fused'].requires_grad
        if cache.state is not None:
            assert torch.isfinite(cache.state).all()
            state_norm.append(float(cache.state.float().square().mean().sqrt()))
        frame_pair_distance.append(float((result['frame_features'][:, 0] -
                                         result['frame_features'][:, 1]).abs().max()))
        storage.append(resident_bytes(cache))
    count, before = cache.observed_clips, cache.read()['fused'].clone()
    for _ in range(3):
        assert torch.equal(cache.read()['fused'], before)
    assert cache.observed_clips == count
    last = video[:, -length:]
    idx = torch.arange(video.shape[1]-length, video.shape[1], device=device)[None]
    expect_rejected(lambda: cache.update(last, [patient], idx))
    expect_rejected(lambda: cache.update(last, ['different_patient'], idx+length))
    expect_rejected(lambda: cache.update(last, [patient], idx+length-1))
    expect_rejected(lambda: cache.update(last, [patient], idx+length+1))

    # These tests run FP32, separately from timed AMP inference.
    native = video[:, :model.frames]
    cache.reset()
    first = cache.update(native[:, :length], [patient], torch.arange(length, device=device)[None])
    fresh = first['frame_features'].clone()
    cache.reset(['new_patient'])
    reset = cache.update(native[:, :length], ['new_patient'], torch.arange(length, device=device)[None])
    torch.testing.assert_close(fresh, reset['frame_features'], rtol=1e-5, atol=1e-6)
    cache.reset()
    chunks = []
    for start in range(0, model.frames, length):
        out = cache.update(native[:, start:start+length], [patient],
                           torch.arange(start, start+length, device=device)[None])
        chunks.append(out['frame_features'])
    offline = model.frame_features(model.diagnostic_features(native)['features'])
    torch.testing.assert_close(torch.cat(chunks, 1), offline, rtol=1e-5, atol=1e-6)
    changed = native.clone()
    changed[:, -length:] = 1-changed[:, -length:]
    other = model.frame_features(model.diagnostic_features(changed)['features'])
    torch.testing.assert_close(offline[:, :-length], other[:, :-length], rtol=1e-5, atol=1e-6)
    # Full FIFO includes distinct pre-clip states only after its first eviction.
    plateau = storage[capacity:]
    assert plateau and len(set(plateau)) == 1, 'Persistent stream storage grew after FIFO filled'
    return dict(patient=patient, source_fps=fps, frames=video.shape[1],
        source_span_seconds=(video.shape[1]-1)/fps, recent_span_seconds=(model.frames-1)/fps,
        max_acquisition_wait_seconds=(length-1)/fps,
        clip_update_ms_p50=float(np.median(timings)), clip_update_ms_p95=float(np.quantile(timings,.95)),
        persistent_bytes=plateau[-1], storage_bytes_by_clip=storage,
        state_rms_by_clip=state_norm, adjacent_same_tubelet_max_abs=frame_pair_distance,
        checks='finite,bounded,read_idempotent,reset,duplicate_overlap_gap_patient_rejected,offline_match,no_interclip_future',
        note='Compute excludes I/O and task head. Within-clip attention is bidirectional. '
             'Finite state is not proof of useful long-horizon memory or accurate arbitrary-frame segmentation.')


def position_summary(rows):
    # Restrict comparisons to identical labeled source frames complete at EVERY tested position.
    positions = sorted({int(r['target_index']) for r in rows})
    valid = {}
    for r in rows:
        key = (r['patient'], r['source_frame'])
        if r['full_context']:
            valid.setdefault(key, set()).add(int(r['target_index']))
    eligible = {k for k, v in valid.items() if len(v) == len(positions)}
    summaries = []
    for pos in positions:
        selected = [r for r in rows if int(r['target_index']) == pos and
                    (r['patient'], r['source_frame']) in eligible]
        by_patient = {}
        for r in selected:
            by_patient.setdefault(r['patient'], []).append(r['dice'])
        summaries.append(dict(target_index=pos, labeled_frames=len(selected), patients=len(by_patient),
            dice_patient_mean=float(np.mean([np.mean(v) for v in by_patient.values()])) if by_patient else None))
    return summaries


def evaluate(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, meta = load_model(args.checkpoint, device)
    if not isinstance(model, TemporalMAE) or model.local_frames != 16 or model.frames != 64:
        raise ValueError('This bounded check requires the selected L16/T64 protocol')
    model.eval().requires_grad_(False)
    model.gradient_checkpointing = False
    args.input_protocol = meta['input_protocol']
    args.seg_frames, args.seg_target_index, args.offset_probe = 64, 62, True
    out, cache_dir = Path(args.output_dir), Path(args.cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(version=1, checkpoint_sha256=file_hash(args.checkpoint),
        code_sha256=code_hash(), audit_sha256=file_hash(__file__), seed=args.seed,
        seg_train_cases=args.seg_train_cases, seg_val_cases=args.seg_val_cases,
        stream_cases=args.stream_cases, stream_frames=args.stream_frames,
        seg_steps=args.seg_steps, positions=list(range(48,64)), meta=meta,
        train_all_positions=args.train_all_positions,
        filelist_sha256=file_hash(Path(args.data_root)/'FileList.csv'),
        traces_sha256=file_hash(Path(args.data_root)/'VolumeTracings.csv'))
    guard = out/'protocol.json'
    if guard.exists() and json.loads(guard.read_text()) != identity:
        raise ValueError('Changed audit protocol; use a new output directory')
    save_json(guard, identity)
    if (out/'DONE').exists():
        return
    train = SegViews(args.data_root, 'train', args, model.in_chans, 2)
    val = SegViews(args.data_root, 'val', args, model.in_chans, 2)
    assert not set(train.patient_ids) & set(val.patient_ids)
    sample_data = TemporalEchoDataset(args.data_root, 'val', 64, channels=model.in_chans,
        input_protocol=args.input_protocol, seed=args.seed, random_start=False)
    save_json(out/'runtime.json', tune_runtime(model, sample_data, args, device, train[0]))
    training = []
    for target in range(48,64,2) if args.train_all_positions else (62,):
        cfg = copy.copy(args)
        cfg.seg_target_index = target
        dataset = SegViews(args.data_root, 'train', cfg, model.in_chans, 2)
        item = extract(model, dataset, args, device, 'seg', 'real', cache_dir/f'train_{target}.npz', identity)
        training.append({k:item[k] for k in ('fused','y','rows')})
        (cache_dir/f'train_{target}.npz').unlink()
    tr = {k:np.concatenate([v[k] for v in training]) for k in ('fused','y')}
    tr['rows'] = [r for v in training for r in v['rows']]
    del training, item
    values = []
    for target in range(48, 64, 2):
        cfg = copy.copy(args)
        cfg.seg_target_index = target
        dataset = SegViews(args.data_root, 'val', cfg, model.in_chans, 2)
        va = extract(model, dataset, args, device, 'seg', 'real', cache_dir/f'val_{target}.npz', identity)
        for row in va['rows']:
            row['id'] += f':position{target}'
        values.append(va)
        (cache_dir/f'val_{target}.npz').unlink()
    combined = {k:np.concatenate([v[k] for v in values]) for k in ('fused', 'y')}
    combined['rows'] = [r for v in values for r in v['rows']]
    _, rows, losses = seg_probe(tr, combined, 'fused', args, device, model.tubelet_size)
    write_csv(out/'seg_positions.csv', rows)
    write_csv(out/'seg_position_summary.csv', position_summary(rows))
    write_csv(out/'seg_head_loss.csv', losses)
    del tr, combined, values, va

    data = TemporalEchoDataset(args.data_root, 'val', args.stream_frames, channels=model.in_chans,
        input_protocol=args.input_protocol, seed=args.seed, random_start=False)
    streams = []
    excluded = 0
    for i in range(len(data)):
        if float(data.df.iloc[i].get('NumberOfFrames', 0)) < args.stream_frames:
            excluded += 1
            continue
        sample = data[i]
        fps = float(data.df.iloc[i]['FPS'])
        if not sample['frame_valid'].all() or not np.isfinite(fps) or fps <= 0:
            excluded += 1
            continue
        result = inspect_stream(model, sample['video'][None].to(device), sample['id'], fps)
        result.update(source_start=int(sample['frame_indices'][0]), source_end=int(sample['frame_indices'][-1]))
        streams.append(result)
        save_json(out/'stream_checks.json', dict(cases=streams, excluded=excluded))
        if len(streams) >= args.stream_cases:
            break
    if len(streams) < args.stream_cases:
        raise ValueError('Insufficient real long videos; never pad this test')
    (out/'DONE').write_text('Frozen position and stream checks completed\n')
    archive_analysis(out)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint','output_dir','cache_dir'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--data_root', default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    for key, value in dict(seed=42, seg_train_cases=64, seg_val_cases=64, seg_steps=200,
        probe_batch_size=32, stream_cases=8, stream_frames=512, cpu_threads=4,
        batch_size=0, max_batch_size=32, num_workers=8, prefetch_factor=2, recent_frames=64).items():
        p.add_argument('--'+key, type=int, default=value)
    p.add_argument('--seg_lr', type=float, default=.01)
    p.add_argument('--train_all_positions', action='store_true',
                   help='Keep head/steps fixed; balance training across all16 local positions')
    p.add_argument('--auto_workers', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.stream_frames <= 64 or args.stream_frames % 16 or min(args.stream_cases,args.seg_train_cases,args.seg_val_cases,args.seg_steps) < 1:
        p.error('Need positive budgets and stream_frames > 64 divisible by16')
    evaluate(args)


if __name__ == '__main__':
    main()
