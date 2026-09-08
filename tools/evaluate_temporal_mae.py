"""Fixed-budget frozen probes. Never fit the backbone or select with test data."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.temporal_mae import TemporalMAE
from tools.evaluate_representation_quality import load_model, regression_metrics
from utils.temporal_data import TemporalEchoDataset
from utils.downstream_datasets import EchoNetSegmentationDataset, _rasterize_echonet_trace
from utils.datasets import _as_video_tensor
from echo_aug_validation.io_utils import find_echonet_video, read_video
from utils.seed import seed_everything


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ridge_fit(x, y, alpha=10.):
    x, y = x.double(), y.double()
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-5)
    x = (x - mean) / std
    ym = y.mean(0)
    if len(x) <= x.shape[1]:
        weight = x.T @ torch.linalg.solve(x @ x.T + alpha * torch.eye(len(x)), y - ym)
    else:
        weight = torch.linalg.solve(x.T @ x + alpha * torch.eye(x.shape[1]), x.T @ (y - ym))
    return mean, std, weight, ym


def ridge_apply(fit, x):
    mean, std, weight, ym = fit
    return (((x.double() - mean) / std) @ weight + ym).float()


def bootstrap_mean(values, seed=42, repetitions=1000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.quantile(means, .025)),
                high=float(np.quantile(means, .975)), n=len(values))


def plot_diagnostics(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path = output / 'trajectory_0.npz'
    if not path.exists():
        return
    with np.load(path, allow_pickle=False) as data:
        frame = data['features'][0]
        indices = data['frame_indices'][0]
        valid = indices >= 0
        frame = frame[valid]
        state = data['states']
        trajectory = state[0] if state.size else frame
    centered = trajectory - trajectory.mean(0)
    u, singular, _ = np.linalg.svd(centered, full_matrices=False)
    projected = u[:, :2] * singular[:2]
    fig, axes = plt.subplots(1,2,figsize=(11,4))
    axes[0].plot(indices[valid][1:],np.linalg.norm(np.diff(frame,axis=0),axis=1))
    axes[0].set(xlabel='Source frame',ylabel='Adjacent feature L2',title='Unmasked features')
    if projected.shape[1] >= 2:
        axes[1].plot(projected[:,0],projected[:,1],alpha=.4)
        axes[1].scatter(projected[:,0],projected[:,1],c=np.arange(len(projected)),cmap='viridis',s=12)
    axes[1].set(title='Within-case state/feature PCA (diagnostic)',xlabel='PC1',ylabel='PC2')
    fig.tight_layout()
    fig.savefig(output/'temporal_diagnostics.png',dpi=140)
    plt.close(fig)


@torch.inference_mode()
def inference_benchmark(model, dataset, args, device):
    batch = next(iter(make_loader(dataset,args)))
    video, valid = batch['video'].to(device), batch['frame_valid'].to(device)
    for _ in range(2):
        stream(model,video,valid)
    if device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(3 if args.smoke else 10):
        stream(model,video,valid)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = (time.perf_counter()-start)/(3 if args.smoke else 10)
    return dict(normal_encoder_ms_per_batch=seconds*1000, batch_size=len(video),
                input_frames=video.shape[1], videos_per_second=len(video)/seconds,
                peak_allocated_gb=torch.cuda.max_memory_allocated()/2**30 if device.type=='cuda' else 0,
                device=str(device), amp=device.type=='cuda')


@torch.inference_mode()
def stream(model, video, valid, intervention='normal', reset_interval=0):
    """Every method sees the same raw frames; native windows reset at boundaries."""
    native = model.frames
    outputs, state_rows = [], []
    for start in range(0, video.shape[1], native):
        clip, fv = video[:, start:start + native], valid[:, start:start + native]
        actual = clip.shape[1]
        if actual < native:
            clip = F.pad(clip, (0, 0, 0, 0, 0, 0, 0, native - actual))
            fv = F.pad(fv, (0, native - actual), value=False)
        with torch.autocast(video.device.type, enabled=video.device.type == 'cuda'):
            if isinstance(model, TemporalMAE):
                result = model.state_trajectory(clip, fv, intervention, reset_interval)
                encoded = result['features']
                state_rows.append(result['states'].float())
            else:
                encoded = model.forward_features(clip)
        encoded = encoded.float().repeat_interleave(model.tubelet_size, dim=1)[:, :actual]
        outputs.append(encoded)
    sequence = torch.cat(outputs, 1)
    pooled = (sequence.mean(2) * valid[:, :, None]).sum(1) / valid.sum(1)[:, None].clamp_min(1)
    return sequence, pooled, torch.cat(state_rows, 1) if state_rows else None


def make_loader(dataset, args):
    kwargs = dict(batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                  pin_memory=torch.cuda.is_available())
    if args.num_workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


@torch.inference_mode()
def ef_features(model, dataset, args, device, modes=('normal',), reverse=False, diagnostic=None):
    features = {mode: [] for mode in modes}
    state_features = []
    targets, ids, reversed_features, valid_counts = [], [], [], []
    trajectory = []
    elapsed, examples = 0., 0
    for batch_index, batch in enumerate(tqdm(make_loader(dataset, args), desc='frozen EF')):
        video, valid = batch['video'].to(device), batch['frame_valid'].to(device)
        ids.extend(batch['id'])
        targets.append(batch['target'])
        valid_counts.extend(valid.sum(1).cpu().tolist())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        for mode in modes:
            if mode == 'shuffle' and len(video) == 1:
                # Duplicate is not a cross-patient intervention. Omit this case.
                features[mode].append(torch.full((1, model.embed_dim), float('nan')))
                continue
            intervention = 'normal' if mode.startswith('reset_') else mode
            interval = int(mode.split('_')[1]) if mode.startswith('reset_') else 0
            seq, pooled, states = stream(model, video, valid, intervention, interval)
            features[mode].append(pooled.cpu())
            if mode == 'normal' and states is not None:
                k = model.local_frames
                count = (video.shape[1] + k - 1) // k
                weights = F.pad(valid,(0,count*k-valid.shape[1])).reshape(len(video),count,k).sum(-1)
                state_features.append(((states[:,:count] * weights[:,:,None]).sum(1) /
                                       weights.sum(1)[:,None].clamp_min(1)).cpu())
            if mode == 'normal' and diagnostic and batch_index < 2:
                frame = seq.mean(2).cpu()
                for bi, case in enumerate(batch['id']):
                    length = int(valid[bi].sum())
                    for ti in range(1, length):
                        trajectory.append(dict(id=case, sampled_frame=ti,
                            source_frame=int(batch['frame_indices'][bi, ti]),
                            feature_l2=float((frame[bi, ti] - frame[bi, ti - 1]).norm()),
                            feature_cos=float(F.cosine_similarity(frame[bi, ti], frame[bi, ti - 1], dim=0))))
                np.savez_compressed(diagnostic / f'trajectory_{batch_index}.npz',
                    ids=np.asarray(batch['id']), frame_indices=batch['frame_indices'].numpy(),
                    features=frame.numpy(), states=np.empty(0) if states is None else states.cpu().numpy())
        if reverse:
            # Reverse only real frames, keeping padding at the end.
            rv = video.clone()
            for bi in range(len(video)):
                n = int(valid[bi].sum())
                rv[bi, :n] = video[bi, :n].flip(0)
            reversed_features.append(stream(model, rv, valid)[1].cpu())
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - start
        examples += len(video)
    if diagnostic:
        write_csv(diagnostic / 'temporal_distances.csv', trajectory)
    if state_features:
        features['state'] = state_features
    return ({key: torch.cat(value).clone() for key, value in features.items()}, torch.cat(targets).clone(), ids,
            torch.cat(reversed_features).clone() if reverse else None, valid_counts,
            dict(total_feature_seconds=elapsed, cases=examples, modes=list(modes), includes_reverse=reverse))


class ProbeSegDataset(EchoNetSegmentationDataset):
    def __init__(self, root, split, args, channels):
        super().__init__(root, split, 112)
        self.context = args.audit_frames
        self.channels = channels
        rng = np.random.default_rng(args.seed)
        # Select patients, then retain both annotated frames, not a prefix of frames.
        ids = sorted({sample['stem'] for sample in self.samples})
        rng.shuffle(ids)
        limit = args.seg_train_cases if split == 'train' else args.seg_val_cases
        keep = set(ids[:limit])
        self.samples = [sample for sample in self.samples if sample['stem'] in keep]
        area = {}
        for sample in self.samples:
            area[(sample['stem'], sample['frame'])] = int(_rasterize_echonet_trace(sample['trace'], (112, 112)).sum())
        self.ed_frame = {case: max((s for s in self.samples if s['stem'] == case),
                                  key=lambda s: area[(case, s['frame'])])['frame'] for case in keep}

    def __getitem__(self, index):
        sample = self.samples[index]
        path = find_echonet_video(self.root, sample['stem'])
        raw = np.load(path, mmap_mode='r') if path.suffix == '.npy' else read_video(path)
        if self.channels == 3 and (raw.ndim != 4 or raw.shape[-1] != 3):
            raise ValueError('Frozen probes require RGB inputs: use original AVI or RGB NPY, not grayscale NPY.')
        if path.suffix not in {'.npy','.npz'} and raw.ndim == 4:
            raw = raw[..., ::-1].copy()
        center = sample['frame']
        if not 0 <= center < len(raw):
            raise ValueError(f'Tracing frame out of bounds: {sample["stem"]}, {center}')
        # Centered temporal context, with explicit invalid padding, never repeated motion.
        indices = center + np.arange(self.context) - self.context // 2
        valid = (indices >= 0) & (indices < len(raw))
        clip = np.zeros((self.context, *raw.shape[1:]), dtype=raw.dtype)
        clip[valid] = raw[indices[valid]]
        video = _as_video_tensor(clip, self.context, 112, channels=self.channels)
        if raw.shape[1:3] != (112, 112):
            raise ValueError('EchoNet tracing probes require the native 112x112 coordinate system.')
        mask = _rasterize_echonet_trace(sample['trace'], (112, 112))
        return dict(video=video, frame_valid=torch.from_numpy(valid), mask=torch.from_numpy(mask).long(),
                    target_index=self.context // 2, phase=int(center == self.ed_frame[sample['stem']]),
                    id=sample['stem'] + ':' + str(center))


@torch.inference_mode()
def seg_features(model, dataset, args, device):
    features, masks, phase, ids, state_features = [], [], [], [], []
    for batch in tqdm(make_loader(dataset, args), desc='frozen segmentation'):
        seq, _, states = stream(model, batch['video'].to(device), batch['frame_valid'].to(device))
        token = seq[:, args.audit_frames // 2]
        side = int(token.shape[1] ** .5)
        fmap = token.transpose(1, 2).reshape(len(token), -1, side, side)
        features.append(F.interpolate(fmap, (14, 14), mode='bilinear', align_corners=False).cpu())
        masks.append(batch['mask'])
        phase.append(batch['phase'])
        ids.extend(batch['id'])
        if states is not None:
            state_index = min(args.audit_frames//2//model.local_frames,states.shape[1]-1)
            state_features.append(states[:,state_index].cpu())
    return (torch.cat(features).clone(), torch.cat(masks).clone(), torch.cat(phase).clone(), ids,
            torch.cat(state_features).clone() if state_features else None)


def segmentation_probe(train, val, args, device):
    torch.manual_seed(args.seed + 2001)
    x, y, phase, _, state = train
    xv, yv, phasev, ids, statev = val
    mean, std = x.mean((0, 2, 3), keepdim=True), x.std((0, 2, 3), keepdim=True).clamp_min(1e-5)
    x, xv = (x - mean) / std, (xv - mean) / std
    head = nn.Conv2d(x.shape[1], 2, 1).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.001)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.seg_steps):
        indices = torch.randint(len(x), (min(16, len(x)),), generator=generator)
        logits = F.interpolate(head(x[indices].to(device)), (112, 112), mode='bilinear', align_corners=False)
        loss = F.cross_entropy(logits, y[indices].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    rows, inter, denom = [], 0., 0.
    with torch.no_grad():
        for start in range(0, len(xv), 16):
            pred = F.interpolate(head(xv[start:start+16].to(device)), (112, 112),
                                 mode='bilinear', align_corners=False).argmax(1).cpu()
            target = yv[start:start+16]
            intersection = ((pred == 1) & (target == 1)).sum((1, 2))
            denominator = (pred == 1).sum((1, 2)) + (target == 1).sum((1, 2))
            dice = (2 * intersection + 1e-6) / (denominator + 1e-6)
            inter += float(intersection.sum())
            denom += float(denominator.sum())
            rows.extend(dict(id=ids[start+i], dice=float(v)) for i, v in enumerate(dice))
    # ED/ES is an area-derived binary proxy, not a dense cardiac phase annotation.
    fit = ridge_fit(x.mean((2, 3)), F.one_hot(phase.long(), 2).float())
    phase_prediction = ridge_apply(fit, xv.mean((2, 3))).argmax(1)
    metrics = dict(dice_mean=float(np.mean([r['dice'] for r in rows])),
                dice_global=2 * inter / max(1, denom),
                area_derived_ed_es_accuracy=float((phase_prediction == phasev).float().mean()),
                steps=args.seg_steps, grid=14)
    if state is not None:
        fit = ridge_fit(state,F.one_hot(phase.long(),2).float())
        metrics['state_area_derived_ed_es_accuracy'] = float(
            (ridge_apply(fit,statev).argmax(1)==phasev).float().mean())
        metrics.update(state_segmentation_probe(state,statev,y,yv,args,device))
    return metrics, rows


def state_segmentation_probe(state,statev,target,targetv,args,device):
    # Global state -> spatial logits is linear. This is a separate diagnostic:
    # it has more head parameters than the shared per-token linear probe.
    torch.manual_seed(args.seed+3001)
    mean, std = state.mean(0), state.std(0,unbiased=False).clamp_min(1e-5)
    state, statev = (state-mean)/std, (statev-mean)/std
    head = nn.Linear(state.shape[-1],2*14*14).to(device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.01,weight_decay=.001)
    generator = torch.Generator().manual_seed(args.seed)
    for _ in range(args.seg_steps):
        index = torch.randint(len(state),(min(16,len(state)),),generator=generator)
        logits = head(state[index].to(device)).reshape(-1,2,14,14)
        logits = F.interpolate(logits,(112,112),mode='bilinear',align_corners=False)
        loss = F.cross_entropy(logits,target[index].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scores = []
    with torch.no_grad():
        for start in range(0,len(statev),16):
            logits = head(statev[start:start+16].to(device)).reshape(-1,2,14,14)
            pred = F.interpolate(logits,(112,112),mode='bilinear',align_corners=False).argmax(1).cpu()==1
            truth = targetv[start:start+16]==1
            scores.extend(((2*(pred&truth).sum((1,2))+1e-6) /
                           (pred.sum((1,2))+truth.sum((1,2))+1e-6)).tolist())
    return dict(state_only_seg_dice=float(np.mean(scores)),state_seg_head_parameters=sum(p.numel() for p in head.parameters()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--audit_frames', type=int, default=256)
    parser.add_argument('--ef_train_cases', type=int, default=512)
    parser.add_argument('--ef_val_cases', type=int, default=256)
    parser.add_argument('--seg_train_cases', type=int, default=64)
    parser.add_argument('--seg_val_cases', type=int, default=64)
    parser.add_argument('--seg_steps', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    try:
        import skimage.draw
    except ImportError as exc:
        raise RuntimeError('Frozen segmentation probes require: python -m pip install scikit-image') from exc
    if args.batch_size < 2:
        raise ValueError('Use batch_size >= 2 for patient-shuffle diagnostics.')
    if args.smoke:
        args.ef_train_cases, args.ef_val_cases = 8, 4
        args.seg_train_cases = args.seg_val_cases = 2
        args.seg_steps, args.num_workers = 2, 0
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, cfg, metadata = load_model(args.checkpoint, device)
    model.requires_grad_(False).eval()
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train = TemporalEchoDataset(args.data_root, 'train', args.audit_frames, channels=model.in_chans,
                               limit=args.ef_train_cases, seed=args.seed, random_start=False)
    val = TemporalEchoDataset(args.data_root, 'val', args.audit_frames, channels=model.in_chans,
                             limit=args.ef_val_cases, seed=args.seed, random_start=False)
    if set(train.ids) & set(val.ids):
        raise ValueError('Train/validation case overlap.')
    modes = ('normal', 'reset', 'shuffle', 'reset_2', 'reset_4') if isinstance(model, TemporalMAE) and model.memory_mode != 'none' else ('normal',)
    tr, y, train_ids, tr_reverse, train_valid_counts, _ = ef_features(model, train, args, device, reverse=True)
    va, yv, val_ids, va_reverse, valid_counts, runtime = ef_features(model, val, args, device, modes, True, output)
    metrics = dict(metadata=metadata, protocol=vars(args), runtime=runtime, ef={}, interventions={})
    metrics['inference'] = inference_benchmark(model,val,args,device)
    if 'state' in tr:
        metrics['state_only_ef'] = regression_metrics(
            ridge_apply(ridge_fit(tr['state'],y),va['state']),yv)
    length_fit = ridge_fit(torch.tensor(train_valid_counts)[:,None],y)
    metrics['length_only_ef_control'] = regression_metrics(
        ridge_apply(length_fit,torch.tensor(valid_counts)[:,None]),yv)
    rows = []
    for n in sorted({min(64, len(y)), min(256, len(y)), len(y)}):
        fit = ridge_fit(tr['normal'][:n], y[:n])
        prediction = ridge_apply(fit, va['normal'])
        metrics['ef'][str(n)] = regression_metrics(prediction, yv)
        metrics['ef'][str(n)]['mae_ci95'] = bootstrap_mean((prediction-yv).abs().numpy(), args.seed)
        for mode in modes:
            pred = ridge_apply(fit, va[mode])
            for i, case in enumerate(val_ids):
                rows.append(dict(id=case, train_cases=n, mode=mode, target=float(yv[i]),
                                 prediction=float(pred[i]), valid_frames=valid_counts[i]))
            if n == len(y) and mode != 'normal':
                delta = (pred - yv).abs() - (prediction - yv).abs()
                delta = delta[torch.isfinite(delta)]
                metrics['interventions'][mode] = bootstrap_mean(delta.numpy(), args.seed)
    write_csv(output / 'ef_predictions.csv', rows)
    order_x = torch.cat((tr['normal'], tr_reverse))
    order_y = torch.cat((torch.ones(len(y)), torch.zeros(len(y))))
    order_fit = ridge_fit(order_x, order_y)
    op = ridge_apply(order_fit, torch.cat((va['normal'], va_reverse)))
    oy = torch.cat((torch.ones(len(yv)), torch.zeros(len(yv))))
    metrics['temporal_order_accuracy'] = float(((op >= .5) == oy.bool()).float().mean())
    x = tr['normal'].float()
    sv = torch.linalg.svdvals(x - x.mean(0)).square()
    p = sv / sv.sum().clamp_min(1e-12)
    metrics['effective_rank'] = float((-(p * p.clamp_min(1e-12).log()).sum()).exp())
    st = seg_features(model, ProbeSegDataset(args.data_root, 'train', args, model.in_chans), args, device)
    sv = seg_features(model, ProbeSegDataset(args.data_root, 'val', args, model.in_chans), args, device)
    metrics['segmentation'], seg_rows = segmentation_probe(st, sv, args, device)
    write_csv(output / 'seg_predictions.csv', seg_rows)
    metrics['case_ids'] = dict(ef_train=train_ids, ef_val=val_ids, seg_train=st[3], seg_val=sv[3])
    plot_diagnostics(output)
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    (output / 'DONE').write_text('frozen probes completed\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
