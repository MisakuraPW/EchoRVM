"""Reference adapter: existing chronological encoder and official EchoNet labels."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .artifacts import digest, file_hash
from .contracts import Experiment
from .metrics import segmentation_row


class TemporalTask(nn.Module):
    def __init__(self, backbone, task, mode, recent, exit_name, append_boundary, hidden=64, depth=2, heads=4):
        super().__init__()
        from models.downstream import ViTPatchSegDecoder
        from models.ef_readout import TemporalEFReadout
        self.backbone, self.task, self.mode = backbone, task, mode
        self.recent, self.exit_name, self.append_boundary = recent, exit_name, append_boundary
        self.linear = mode == 'frozen_linear'
        if task == 'ef':
            self.head = nn.Linear(backbone.embed_dim, 1) if self.linear else TemporalEFReadout(backbone.embed_dim, hidden)
        else:
            self.head = nn.Linear(backbone.embed_dim, 2) if self.linear else ViTPatchSegDecoder(
                backbone.embed_dim, 2, backbone.token_grid[1], backbone.patch_size, hidden, depth, heads)
        if mode.startswith('frozen'):
            backbone.requires_grad_(False)

    def extract(self, samples):
        from utils.final_temporal_execution import encode_window, task_features
        groups, values = defaultdict(list), [None]*len(samples)
        device = next(self.head.parameters()).device
        for index, sample in enumerate(samples):
            groups[(int(sample['prefix_frames']), len(sample['video']))].append(index)
        for (prefix, _), indices in groups.items():
            video = torch.stack([samples[i]['video'] for i in indices]).to(device, non_blocking=True)
            if video.shape[-2:] != (self.backbone.img_size, self.backbone.img_size):
                raise ValueError('No silent resizing in the reference adapter; model and data resolution must match')
            if self.backbone.in_chans == 1:
                video = video[:, :, :1]
            targets = [int(samples[i]['target_index']) for i in indices] if self.task == 'seg' else None
            exits = encode_window(self.backbone, video, prefix, self.recent, targets)
            features, _ = task_features(exits, self.task, self.exit_name, self.append_boundary, self.backbone.memory_slots)
            for row, index in enumerate(indices):
                values[index] = features[row]
        return torch.stack(values)

    def forward(self, samples):
        device = next(self.head.parameters()).device
        if all('features' in sample for sample in samples):
            features = torch.stack([s['features'] for s in samples]).to(device, non_blocking=True)
        else:
            with torch.set_grad_enabled(not self.mode.startswith('frozen') and torch.is_grad_enabled()):
                features = self.extract(samples)
        if self.task == 'ef':
            if self.linear:
                return self.head(features.mean(1)).squeeze(-1)
            slots = self.backbone.memory_slots if self.append_boundary else 0
            return self.head(features, slots)
        grid = self.backbone.token_grid[1]
        if self.linear:
            logits = self.head(features).transpose(1, 2).reshape(len(samples), 2, grid, grid)
            return F.interpolate(logits, size=(self.backbone.img_size, self.backbone.img_size),
                                 mode='bilinear', align_corners=False)
        return self.head(features, grid)


class EchoExperiment(Experiment):
    def _mae(self, samples):
        from utils.final_temporal_training import deterministic_masks
        device = next(self.model.parameters()).device
        video = torch.stack([s['video'] for s in samples]).to(device, non_blocking=True)
        if self.model.in_chans == 1:
            video = video[:, :, :1]
        elif video.shape[2] == 1:
            video = video.expand(-1, -1, 3, -1, -1)
        if video.shape[-2:] != (self.model.img_size, self.model.img_size):
            raise ValueError('MAE data/model spatial contract mismatch')
        epoch = self.train.dataset.epoch.value if self.model.training else 0
        seed = self.seed + (int(epoch)*1000003 if self.model.training else 9173)
        masks = deterministic_masks(self.model, dict(video=video,
                                    record_index=torch.tensor([int(s['record_index']) for s in samples])), seed)
        return self.model(video, masks=masks)

    def train_mode(self):
        self.model.train()
        if isinstance(self.model, TemporalTask) and self.model.mode.startswith('frozen'):
            self.model.backbone.eval()

    def set_checkpointing(self, enabled):
        self.backbone.gradient_checkpointing = bool(enabled)

    @property
    def backbone(self):
        return self.model.backbone if isinstance(self.model, TemporalTask) else self.model

    def features(self, samples):
        if self.task == 'mae':
            raise ValueError('Use an EF/seg readout stage for unmasked representation extraction')
        return self.model.extract(samples)

    def calibration_samples(self, batch_size):
        if not isinstance(self.train, IdentifiedDataset):
            return [self.train[0]] * batch_size
        from utils.final_temporal_data import WindowDataset
        data = self.train.dataset
        if hasattr(data, 'calibration_samples'):
            return data.calibration_samples(batch_size)
        candidates = []
        for index, record in enumerate(data.records):
            case = data.cases[record['patient']]
            if self.task == 'seg':
                target = record['target_frame']
                start = target - (data.recent_frames-data.local_frames)
                available = max(0, target-(data.recent_frames-1))
            else:
                target = None
                start = case['frames']-data.recent_frames
                available = start
            prefix = data.prefix if data.prefix is not None else min(data.max_prefix, available)//data.local_frames*data.local_frames
            candidates.append((prefix, index, dict(patient=record['patient'], recent_start=start, H=prefix,
                                                    **({'target_frame': target} if target is not None else {}))))
        _, index, record = max(candidates, key=lambda item:item[0])
        stress = WindowDataset(data.manifest, 'train', task=self.task, recent_frames=data.recent_frames,
                               local_frames=data.local_frames, max_prefix=data.max_prefix, records=[record])
        sample = stress[0]
        sample.update(sample_id=f'stress:{index}', record_index=index)
        return [sample]*batch_size

    def loss(self, samples):
        if self.task == 'mae':
            groups = defaultdict(list)
            for sample in samples:
                groups[len(sample['video'])].append(sample)
            loss = None
            for values in groups.values():
                value = self._mae(values)['loss'] * len(values)/len(samples)
                loss = value if loss is None else loss+value
            return loss
        return self._task_loss(self.model(samples), samples)

    def _task_loss(self, output, samples):
        if self.task == 'ef':
            y = torch.stack([s['target'] for s in samples]).to(output, non_blocking=True)/100.
            self.last_metrics = dict(batch_mae_pp=(output.detach()-y).abs().mean()*100.)
            return F.smooth_l1_loss(output, y)
        y = torch.stack([s['mask'] for s in samples]).to(device=output.device, dtype=torch.long, non_blocking=True)
        ce = F.cross_entropy(output, y, reduction='none').flatten(1).mean(1)
        probability = output.softmax(1)[:, 1]
        truth = (y == 1).to(probability)
        dice = (2*(probability*truth).flatten(1).sum(1)+1e-6)/(probability.flatten(1).sum(1)+truth.flatten(1).sum(1)+1e-6)
        hard = (output.detach().argmax(1) == 1).to(truth)
        self.last_metrics = dict(batch_dice=((2*(hard*truth).flatten(1).sum(1)+1e-6)/
                                             (hard.flatten(1).sum(1)+truth.flatten(1).sum(1)+1e-6)).mean())
        return (ce + 1-dice).mean()

    def predict(self, samples, boundaries=False):
        if self.task == 'mae':
            return self.evaluate_batch(samples, boundaries)[1]
        return self._rows(self.model(samples), samples, boundaries)

    def _rows(self, prediction, samples, boundaries=False):
        rows = []
        for index, sample in enumerate(samples):
            if self.task == 'ef':
                rows.append(dict(patient=str(sample['patient']), target=float(sample['target']),
                                 prediction=float(prediction[index])*100., sample_id=sample['sample_id']))
            else:
                rows.append(segmentation_row(prediction[index].argmax(0).cpu().numpy(),
                                             np.asarray(sample['mask']) == 1, patient=sample['patient'],
                                             source_frame=sample['target_frame'], class_id=1,
                                             spacing=sample.get('spacing'), boundaries=boundaries,
                                             target_position=int(sample['target_position']), sample_id=sample['sample_id']))
        return rows

    def evaluate_batch(self, samples, boundaries=False):
        if self.task != 'mae':
            prediction = self.model(samples)
            return self._task_loss(prediction, samples), self._rows(prediction, samples, boundaries)
        groups, rows, total = defaultdict(list), [], None
        for index, sample in enumerate(samples):
            groups[len(sample['video'])].append((index, sample))
        for values in groups.values():
            result = self._mae([s for _, s in values])
            loss = result['loss']*len(values)/len(samples)
            total = loss if total is None else total+loss
            errors = (result['pred'].float()-result['target'].float()).square().mean(-1)
            reconstruction = (errors*result['mask']).sum(1)/result['mask'].sum(1).clamp_min(1)
            for offset, (index, sample) in enumerate(values):
                rows.append((index, dict(patient=str(sample['patient']), loss=float(reconstruction[offset]))))
        return total, [row for _, row in sorted(rows)]

    def preview(self, samples, directory):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from .artifacts import write_json
        directory.mkdir(parents=True, exist_ok=True)
        if self.task == 'ef':
            return
        self.model.eval()
        for index, sample in enumerate(samples):
            if self.task == 'seg':
                pred = self.model([sample])[0].argmax(0).cpu().numpy()
                image = sample['video'][int(sample['target_index']), 0].numpy()
                truth = np.asarray(sample['mask'])
                fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')
                for axis, title in zip(axes, ('Observed target frame', 'Ground truth', 'Prediction')):
                    axis.imshow(image, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                    axis.set_title(title); axis.axis('off')
                if truth.any():
                    axes[1].contour(truth, levels=[.5], colors=['#009E73'])
                if pred.any():
                    axes[2].contour(pred, levels=[.5], colors=['#D55E00'])
            else:
                result = self._mae([sample])
                q, p, grid, channels = self.model.tubelet_size, self.model.patch_size, self.model.token_grid[1], self.model.in_chans
                frames = result['pred'].shape[1] // (grid*grid) * q
                def unpatchify(value):
                    return value.reshape(1, frames//q, grid, grid, q, p, p, channels).permute(0, 1, 4, 7, 2, 5, 3, 6).reshape(1, frames, channels, grid*p, grid*p)
                target_patches, predicted_patches = result['target'], result['pred']
                if self.model.norm_pix_loss:
                    from models.video_mae import tubelet_patchify
                    video = sample['video'][None].to(predicted_patches)
                    if self.model.in_chans == 1:
                        video = video[:, :, :1]
                    raw = tubelet_patchify(self.model._normalize_input(video[:, -frames:]), q, p)
                    mean, std = raw.mean(-1, keepdim=True), (raw.var(-1, keepdim=True, unbiased=False)+1e-6).sqrt()
                    target_patches = target_patches*std+mean
                    predicted_patches = predicted_patches*std+mean
                    write_json(directory / 'preview_policy.json', dict(
                        policy='GT per-patch mean/std invert normalized targets for visualization only; NOT raw-pixel reconstruction quality',
                        displayed_frame='middle reconstructed frame', clipping='display range [0,1]'))
                target, prediction = unpatchify(target_patches), unpatchify(predicted_patches)
                if self.model.input_mean is not None:
                    target = target*self.model.input_std.to(target)+self.model.input_mean.to(target)
                    prediction = prediction*self.model.input_std.to(prediction)+self.model.input_mean.to(prediction)
                mask = unpatchify(result['mask'][..., None].expand_as(result['pred']).float())
                observed = target*(1-mask)
                combined = target*(1-mask)+prediction*mask
                frame = frames//2
                fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout='constrained')
                for axis, value, title in zip(axes, (target, observed, combined), ('Target', 'Visible pixels', 'Visible + masked prediction')):
                    axis.imshow(value[0, frame].mean(0).cpu().numpy(), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
                    axis.set_title(title); axis.axis('off')
            fig.savefig(directory / f'fixed_case_{index:02d}.png', dpi=150)
            plt.close(fig)
        write_json(directory / 'selection.json', dict(policy='first two deterministic diagnostic/VAL records, never best-looking cases',
                                                     ids=[s['sample_id'] for s in samples]))

    def model_export(self):
        return dict(model_state_dict=self.backbone.state_dict(),
                    config=dict(model=self.native_config, pretraining_data_protocol=self.pretraining_protocol))


class IdentifiedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, split):
        self.dataset, self.split = dataset, split

    def __len__(self):
        return len(self.dataset)

    def set_epoch(self, epoch):
        self.dataset.set_epoch(epoch)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['sample_id'] = f'{self.split}:{index}'
        return sample


def build(config, stage, source, device):
    from models.final_temporal_mae import FinalTemporalMAE, load_final_model
    from utils.final_temporal_data import WindowDataset, _load_manifest
    data, model = config['data'], config['model']
    kind = stage['kind']
    task = stage.get('readout_task', 'ef') if kind == 'representation' else kind
    if task not in {'ef', 'seg', 'mae'}:
        raise ValueError('Reference adapter supports MAE, EF, segmentation and representation')
    seed = config['training']['seed']
    if source:
        backbone, saved, report = load_final_model(source, model.get('overrides'), seed)
        native = saved['model']
        pretraining_protocol = saved.get('pretraining_data_protocol', {})
    else:
        pretraining_protocol = {}
        native = model.get('config')
        if not native:
            raise ValueError('Provide model.config for initialization or a native checkpoint source')
        if native.get('forward_variant'):
            raise ValueError('Candidates must explicitly copy shared tensors from a B0 epoch-0 checkpoint')
        backbone, report = FinalTemporalMAE(**native), dict(initialization='random, explicitly configured')
        if model.get('init_checkpoint'):
            from utils.pretrained_init import load_videomae_init
            report = load_videomae_init(backbone, model['init_checkpoint'])
            report['sha256'] = file_hash(model['init_checkpoint'])
            if report.get('missing_encoder_keys'):
                raise ValueError(f'Incomplete encoder initialization: {report["missing_encoder_keys"]}')
    recent = int(data.get('recent_frames', 64))
    local = backbone.local_frames
    if data.get('local_frames', local) != local or recent % local:
        raise ValueError('Dataset and encoder local_frames must match; recent window must contain complete clips')
    frozen = stage.get('mode', 'finetune').startswith('frozen') or kind == 'representation'
    augmentation = data.get('augmentation') if not frozen else None
    arguments = dict(task=task, recent_frames=recent, local_frames=local,
                     max_prefix=data.get('max_prefix', 128), prefix=data.get('prefix', 0), seed=data.get('seed', seed))
    all_ssl = task == 'mae' and bool(data.get('ssl_manifest'))
    if all_ssl:
        from .ssl_data import SSLWindowDataset, load_ssl_manifest
        manifest = load_ssl_manifest(data['ssl_manifest'])
        if (manifest['local_frames'], manifest['recent_frames'], manifest['max_prefix']) != (local, recent, arguments['max_prefix']):
            raise ValueError('SSL inventory and model/data window dimensions disagree')
        if backbone.reconstruction_recent_frames < recent+arguments['max_prefix']:
            raise ValueError('Clean all-window SSL must reconstruct every observed clip, including its prefix')
        train = SSLWindowDataset(manifest, True, arguments['seed'], limit=data.get('limit_train'), augmentation=augmentation)
        val = SSLWindowDataset(manifest, False, arguments['seed'], data.get('diagnostic_cases', 64), data.get('limit_val'))
        used_manifest = data['ssl_manifest']
        pretraining_protocol = dict(exposure=manifest['exposure'], manifest_sha256=digest(manifest),
                                    counts=manifest['counts'], labels_used=False,
                                    window_policy=manifest['sampling'])
    else:
        manifest = _load_manifest(data['manifest'])
        train = WindowDataset(manifest, 'train', training=not frozen, positions='balanced',
                              aug_cfg=augmentation, limit=data.get('limit_train'), **arguments)
        val = WindowDataset(manifest, 'val', training=False, positions=data.get('val_positions', 'all'),
                            limit=data.get('limit_val'), **arguments)
        used_manifest = data['manifest']
    if not len(train) or not len(val):
        raise ValueError('No eligible SSL/diagnostic videos' if all_ssl else 'No eligible TRAIN/VAL records')
    train_ids = {r['patient'] for r in train.records}
    val_ids = {r['patient'] for r in val.records}
    if train_ids & val_ids and not all_ssl:
        raise ValueError('Patient leakage between TRAIN and VAL')
    if task != 'mae':
        backbone = TemporalTask(backbone, task, stage.get('mode', 'frozen_linear'), recent,
                                stage.get('exit_name', 'final'), stage.get('append_boundary', False),
                                stage.get('head_dim', 64), stage.get('head_depth', 2), stage.get('head_heads', 4))
    metadata = dict(dataset='EchoNet-Dynamic', manifest_sha256=digest(manifest),
                    native_config=native, initialization=report, checkpointing=True,
                    feature_exit=dict(name=stage.get('exit_name', 'final'), layer=len((backbone.backbone if isinstance(backbone, TemporalTask) else backbone).blocks),
                                      norm='final encoder norm', history_read_before_exit=native.get('forward_variant') == 'mid_read' or stage.get('exit_name', 'final') not in {'local','local_final'},
                                      local_exit_semantics='post-read final layer, NOT baseline H' if native.get('forward_variant') == 'mid_read' else 'history-free final-layer H'),
                    train_ids=sorted(train_ids), val_ids=sorted(val_ids),
                    train_excluded=train.excluded, val_excluded=val.excluded,
                    train_records_sha256=digest(train.records), val_records_sha256=digest(val.records),
                    data_files={str(Path(used_manifest)): file_hash(used_manifest)},
                    input='native112 gray_repeat3, no RGB disk cache',
                    head=type(backbone.head).__name__ if task != 'mae' else None,
                    loss='masked pixel objective from native model' if task == 'mae' else
                         'Huber on EF/100; metrics in pp' if task == 'ef' else 'per-sample CE + foreground soft Dice',
                    aggregation='positions -> source frame -> patient', test_access=all_ssl,
                    ssl_data_exposure=pretraining_protocol.get('exposure', 'source-dependent; inspect native source provenance'),
                    diagnostic_overlap=all_ssl,
                    evaluation_scope='Seen-population reconstruction diagnostic; no held-out SSL validation or checkpoint selection' if all_ssl else
                        'TRAIN supervised fitting, VAL selection; TEST labels unopened. Source SSL may have seen VAL/TEST images; see ssl_data_exposure',
                    mask_policy='record-index + epoch keyed train masks; fixed record-index VAL masks')
    exp = EchoExperiment(backbone.to(device), IdentifiedDataset(train, 'train'), IdentifiedDataset(val, 'val'),
                         task, metadata, microbatch_safe=True, cacheable=frozen and task != 'mae')
    exp.native_config = native
    exp.pretraining_protocol = pretraining_protocol
    exp.seed = seed
    return exp
