"""Official-weight single-frame anchor using the verified local ViTSeg2D port."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from models.echocardmae_official import load_echocardmae_official_seg
from .artifacts import digest, file_hash
from .contracts import Experiment
from .coverage_repair import CoverageDataset
from .echo import EchoExperiment, IdentifiedDataset
from .readout_calibration import CalibratedExperiment
from utils.final_temporal_data import _close, _file_provenance, _prepare_video, _rng, _trace_mask
from utils.echo_input import read_echo_input


class FrameDataset(CoverageDataset):
    def __init__(self, manifest, split, training=False, seed=42):
        super().__init__(manifest, split, training=training, positions='balanced',
                         allow_test=split == 'test', seed=seed)
        self.records = [dict(patient=case['patient'], target_frame=t['frame'])
                        for case in self.cases.values() for t in case['traces']]

    def __getitem__(self, index):
        r = self.records[index]
        case = self.cases[r['patient']]
        path = Path(case['path'])
        before = _file_provenance(path)
        if any(before[k] != case[k] for k in ('source_bytes', 'source_mtime_ns')):
            raise ValueError('Anchor video changed after manifest')
        raw = read_echo_input(path, 'gray_repeat3')
        try:
            clip = np.asarray(raw[r['target_frame']:r['target_frame']+1]).copy()
        finally:
            _close(raw)
        if _file_provenance(path) != before:
            raise ValueError('Anchor video changed while reading')
        trace = next(t for t in case['traces'] if t['frame'] == r['target_frame'])
        video, mask = _prepare_video(clip, _trace_mask(trace), None, 0)
        if self.training:
            # Pad after official normalization with zeros, expressed in raw pixels
            # so the model remains the sole normalization owner.
            mean = video.new_tensor([.1257, .1271, .1292])[None, :, None, None]
            padded = mean.expand(1, 3, 136, 136).clone()
            padded[:, :, 12:124, 12:124] = video
            target = F.pad(mask, (12, 12, 12, 12))
            rng = _rng(self.seed, 'official_crop', index, self.epoch.value)
            y, x = rng.integers(0, 25, size=2)
            video, mask = padded[:, :, y:y+112, x:x+112], target[y:y+112, x:x+112]
        return dict(video=video, mask=mask, patient=r['patient'], id=r['patient'],
                    target_frame=r['target_frame'], target_index=0, target_position=0,
                    prefix_frames=0, record_index=index, coverage_kind='single_frame',
                    frame_indices=torch.tensor([r['target_frame']]))


class FrameModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.official = model

    def forward(self, samples):
        device = next(self.parameters()).device
        return self.official(torch.stack([s['video'][0] for s in samples]).to(device, non_blocking=True))


class AnchorExperiment(Experiment):
    _task_loss = CalibratedExperiment._task_loss
    _rows = EchoExperiment._rows
    evaluate_batch = EchoExperiment.evaluate_batch
    predict = EchoExperiment.predict
    preview = EchoExperiment.preview

    def loss(self, samples):
        return self._task_loss(self.model(samples), samples)


def evaluation_dataset(cfg, manifest, split, task):
    if task != 'seg':
        raise ValueError('Official anchor only supports segmentation')
    return FrameDataset(manifest, split, seed=cfg['data'].get('seed', 42))


def build(config, stage, source, device):
    if stage['kind'] != 'seg' or stage.get('mode') != 'finetune' or not source:
        raise ValueError('Anchor requires official weights and segmentation FT')
    from utils.final_temporal_data import _load_manifest
    manifest = _load_manifest(config['data']['coverage_manifest'])
    model, report = load_echocardmae_official_seg(source, config['model']['config'])
    allowed = {'encoder.patch_embed.proj.weight', 'decoder.head.weight', 'decoder.head.bias'}
    # The released pretraining checkpoint omits the two sinusoidal position
    # parameters; the author also initializes these in the segmentation model.
    missing_allowed = allowed | {'pos_embed', 'encoder.pos_embed'}
    if set(report['missing']) - missing_allowed or set(report['skipped_shape']) - allowed:
        raise ValueError(f'Unexpected official checkpoint mismatch: {report}')
    seed = config['data'].get('seed', 42)
    train, val = FrameDataset(manifest, 'train', True, seed), FrameDataset(manifest, 'val', False, seed)
    for dataset, key in ((train, 'limit_train'), (val, 'limit_val')):
        if config['data'].get(key) is not None:
            dataset.records = dataset.records[:config['data'][key]]
    metadata = dict(train_ids=sorted({r['patient'] for r in train.records}),
                    val_ids=sorted({r['patient'] for r in val.records}),
                    train_records_sha256=digest(train.records), val_records_sha256=digest(val.records),
                    initialization=report, source_sha256=file_hash(source),
                    input='single real frame; official mean/std; normalized-zero pad12 + random crop',
                    loss='foreground/background soft Dice smooth1e-5',
                    evaluation_scope='Official-weight local implementation anchor; VAL selection, TEST separate',
                    scope_warning='Not the author training runtime; sample-tail and execution precision are explicit',
                    ssl_data_exposure='released official weights; inspect upstream pretraining split',
                    data_files={str(config['data']['coverage_manifest']): file_hash(config['data']['coverage_manifest'])})
    if set(metadata['train_ids']) & set(metadata['val_ids']):
        raise ValueError('Supervised split overlap')
    return AnchorExperiment(FrameModel(model).to(device), IdentifiedDataset(train, 'train'),
                            IdentifiedDataset(val, 'val'), 'seg', metadata, microbatch_safe=True)
