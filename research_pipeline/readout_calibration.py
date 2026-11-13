"""Two bounded task adaptations of existing B0 weights, not new SSL."""

import torch
from torch import nn
from torch.nn import functional as F

from models.downstream import ViTPatchSegDecoder
from .echo import EchoExperiment, build as build_reference


def transfer_seg_decoder(backbone, head):
    """Copy every shared decoder tensor; convert separate q/v bias exactly."""
    source = dict(backbone.decoder_blocks.state_dict())
    for index in range(len(head.blocks)):
        prefix = f'{index}.attn.'
        if prefix+'q_bias' in source:
            q, v = source.pop(prefix+'q_bias'), source.pop(prefix+'v_bias')
            source[prefix+'qkv.bias'] = torch.cat((q, torch.zeros_like(q), v))
    head.blocks.load_state_dict(source, strict=True)
    head.norm.load_state_dict(backbone.decoder_norm.state_dict(), strict=True)
    head.encoder_to_decoder.load_state_dict(backbone.decoder_embed.state_dict(), strict=True)
    return dict(blocks=len(head.blocks), dim=head.encoder_to_decoder.out_features,
                shared_tensors=len(source)+len(backbone.decoder_norm.state_dict())+
                len(backbone.decoder_embed.state_dict()), pixel_classifier='new Xavier initialization',
                position='new trainable 2D sinusoid; no video-position interpolation')


class CalibratedExperiment(EchoExperiment):
    def _task_loss(self, output, samples):
        if self.task == 'ef':
            target = torch.stack([s['target'] for s in samples]).to(output)
            error = output*100-target
            self.last_metrics = dict(batch_mae_pp=error.detach().abs().mean())
            return F.l1_loss(output*100, target)
        target = torch.stack([s['mask'] for s in samples]).to(device=output.device, dtype=torch.long)
        truth = F.one_hot(target, 2).permute(0, 3, 1, 2).float()
        prob = output.float().softmax(1)
        overlap = (prob*truth).sum((2, 3))
        loss = 1-(2*overlap+1e-5)/(prob.sum((2, 3))+truth.sum((2, 3))+1e-5)
        hard = output.detach().argmax(1) == 1
        foreground = target == 1
        self.last_metrics = dict(batch_dice=((2*(hard & foreground).sum((1, 2))+1e-6)/
                                             (hard.sum((1, 2))+foreground.sum((1, 2))+1e-6)).mean())
        return loss.mean()


def build(config, stage, source, device):
    valid_mode = stage.get('mode') == 'finetune' or (
        stage['kind'] == 'ef' and stage.get('mode') == 'frozen_lite' and stage.get('calibrated_ef'))
    if stage['kind'] not in {'ef', 'seg'} or not valid_mode or not source:
        raise ValueError('Calibrated readout requires native weights and an applicable supervised task mode')
    reference = build_reference(config, stage, source, device)
    exp = CalibratedExperiment(reference.model, reference.train, reference.val, reference.task,
                               reference.metadata, reference.microbatch_safe, reference.cacheable)
    exp.seed, exp.native_config = reference.seed, reference.native_config
    exp.pretraining_protocol = reference.pretraining_protocol
    if exp.task == 'seg':
        backbone = exp.backbone
        head = ViTPatchSegDecoder(backbone.embed_dim, 2, backbone.token_grid[1],
                                  backbone.patch_size, 192, 4, 3)
        # Match native decoder LayerNorm epsilon, not the generic head default.
        for layer in head.modules():
            if isinstance(layer, nn.LayerNorm):
                layer.eps = backbone.decoder_norm.eps
        position = head.pos_embed.detach().clone()
        del head.pos_embed
        head.pos_embed = nn.Parameter(position)
        nn.init.xavier_uniform_(head.head.weight)
        nn.init.zeros_(head.head.bias)
        transfer = transfer_seg_decoder(backbone, head)
        exp.model.head = head.to(device)
        exp.metadata.update(head='ViTPatchSegDecoder192x4', head_transfer=transfer,
                            loss='mean foreground/background soft Dice, smooth=1e-5')
    else:
        head = exp.model.head
        nn.init.zeros_(head.regress[-1].weight)
        nn.init.constant_(head.regress[-1].bias, .556)
        exp.metadata.update(head='TemporalEFReadout, zero last weight, bias=.556',
                            loss='L1 on EF percentage points',
                            head_transfer=dict(policy='new task head; same cache sequence, no state-only exit'))
        if stage.get('ef_temporal_relation'):
            exp.metadata['head'] += ', one residual temporal relation block'
    exp.metadata['calibration_scope'] = 'task recipe bundle; not single-factor causal attribution or official reproduction'
    return exp
