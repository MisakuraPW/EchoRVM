"""Small deterministic integration fixture. Never a scientific benchmark."""

from __future__ import annotations

import torch
from torch import nn

from .contracts import Experiment
from .metrics import segmentation_row


class Samples(torch.utils.data.Dataset):
    def __init__(self, count, split, task):
        self.count, self.split, self.task = count, split, task

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        x = torch.rand(1, 8, 8, generator=torch.Generator().manual_seed(index + (1000 if self.split == 'val' else 0)))
        item = dict(video=x, patient=f'{self.split}_{index}', sample_id=f'{self.split}:{index}', target_frame=index)
        item['mask' if self.task == 'seg' else 'target'] = (x[0] > .5).long() if self.task == 'seg' else x.mean()*100
        return item


class ToyModel(nn.Module):
    def __init__(self, task, frozen=False):
        super().__init__()
        self.backbone = nn.Conv2d(1, 4, 1)
        self.head = nn.Conv2d(4, 2 if task == 'seg' else 1, 1)
        self.gradient_checkpointing = False
        if frozen:
            self.backbone.requires_grad_(False)

    def forward(self, samples):
        device = self.head.weight.device
        features = (torch.stack([s['features'] for s in samples]).to(device).permute(0, 3, 1, 2) if 'features' in samples[0] else
                    self.backbone(torch.stack([s['video'] for s in samples]).to(device)))
        return self.head(features)


class ToyExperiment(Experiment):
    def set_checkpointing(self, enabled):
        self.model.gradient_checkpointing = enabled

    def features(self, samples):
        return self.model.backbone(torch.stack([s['video'] for s in samples]).to(self.model.head.weight.device)).permute(0, 2, 3, 1)

    def loss(self, samples):
        pred = self.model(samples)
        if self.task == 'seg':
            return nn.functional.cross_entropy(pred, torch.stack([s['mask'] for s in samples]).to(pred.device))
        target = torch.stack([s['target'] for s in samples]).to(pred)/100
        return (pred.mean((1, 2, 3))-target).square().mean()

    def predict(self, samples, boundaries=False):
        pred = self.model(samples).detach().cpu()
        if self.task == 'seg':
            return [segmentation_row(p.argmax(0).numpy(), s['mask'].numpy(), patient=s['patient'],
                                     source_frame=s['target_frame'], boundaries=boundaries) for p, s in zip(pred, samples)]
        if self.task == 'mae':
            return [dict(patient=s['patient'], loss=float(self.loss([s]))) for s in samples]
        return [dict(patient=s['patient'], target=float(s['target']), prediction=float(p.mean()*100)) for p, s in zip(pred, samples)]


def build(config, stage, source, device):
    task = stage.get('readout_task', 'ef') if stage['kind'] == 'representation' else stage['kind']
    frozen = stage.get('mode', '').startswith('frozen')
    model = ToyModel(task, frozen).to(device)
    if source:
        state = torch.load(source, map_location='cpu', weights_only=False)
        prefix = {k.removeprefix('backbone.'): v for k, v in state['model'].items() if k.startswith('backbone.')}
        model.backbone.load_state_dict(prefix)
    return ToyExperiment(model, Samples(config['data'].get('limit_train', 7), 'train', task),
                         Samples(config['data'].get('limit_val', 3), 'val', task), task,
                         dict(synthetic=True, train_ids=['train'], val_ids=['val'], checkpointing=True),
                         microbatch_safe=True, cacheable=frozen)
