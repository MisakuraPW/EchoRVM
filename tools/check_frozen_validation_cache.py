"""Read-only real-data certificate for frozen validation feature reuse."""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def certify(job, manifest, checkpoint, output, device='cuda'):
    import torch
    from torch.nn import functional as F
    from models.final_temporal_mae import load_final_model
    from utils.final_temporal_data import WindowDataset
    from utils.final_temporal_tasks import (_FeatureCache, _Inputs, _features, _loss, _amp,
                                          _resolve_job, load_frozen_task_head)
    from utils.final_temporal_training import write_json, file_digest
    torch.set_num_threads(4)
    device = torch.device(device)
    backbone = load_final_model(job['checkpoint'])[0].to(device).eval()
    head = load_frozen_task_head(checkpoint, backbone)
    settings = _resolve_job(dict(job, micro_batch=1), backbone)
    assert settings['freeze'] and settings['task'] == 'seg'
    normalization = head.normalization
    dataset = WindowDataset(manifest, 'val', task='seg', training=False, positions='all', limit=2,
                            recent_frames=settings['recent_frames'], local_frames=backbone.local_frames,
                            max_prefix=settings['max_prefix'], prefix=settings['prefix'], seed=settings['dataset_seed'])
    if not len(dataset):
        raise ValueError('No real validation windows for cache certificate')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix='cache_certificate_') as directory:
        cache = _FeatureCache(directory, file_digest(checkpoint), disk_bytes=64 * 1024**2, ram_bytes=0)
        inputs = _Inputs(dataset, cache, 'val')
        count = min(4, len(inputs))
        first = [inputs[index] for index in range(count)]
        with torch.no_grad():
            values, slots = _features(backbone, first, settings, device, cache)
            with _amp(settings, device):
                predicted = head.read(values, slots)
                loss = _loss(predicted, first, 'seg', normalization)
            calls = []
            hook = backbone.patch_embed.register_forward_hook(lambda *args: calls.append(True))
            try:
                replay = [inputs[index] for index in range(count)]
                cached, cached_slots = _features(backbone, replay, settings, device, cache)
            finally:
                hook.remove()
            with _amp(settings, device):
                replay_predicted = head.read(cached, cached_slots)
                replay_loss = _loss(replay_predicted, replay, 'seg', normalization)
            torch.testing.assert_close(values, cached, rtol=0, atol=0)
            torch.testing.assert_close(predicted, replay_predicted, rtol=0, atol=0)
            torch.testing.assert_close(loss, replay_loss, rtol=0, atol=0)
            masks = torch.stack([sample['mask'] for sample in first]).to(device)
            def dice(logits):
                hard = F.interpolate(logits.float(), masks.shape[-2:], mode='bilinear', align_corners=False).argmax(1).bool()
                truth = masks.bool()
                return (2 * (hard & truth).sum((1,2)).float() + 1e-6) / (hard.sum((1,2)) + truth.sum((1,2)) + 1e-6)
            torch.testing.assert_close(dice(predicted), dice(replay_predicted), rtol=0, atol=0)
        if calls:
            raise AssertionError('Validation replay unexpectedly re-encoded video')
        report = dict(passed=True, samples=count, device=str(device), encoder_calls_on_replay=0,
                      feature_precision=str(cached.dtype), exact_feature_match=True, exact_prediction_match=True,
                      exact_loss_match=True, exact_dice_match=True, head_checkpoint_sha256=file_digest(checkpoint),
                      scope='Execution equivalence certificate, not a new performance experiment')
        write_json(output, report)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    print(json.dumps(certify(json.loads(Path(args.job).read_text()), json.loads(Path(args.manifest).read_text()),
                             args.checkpoint, args.output, args.device)))


if __name__ == '__main__':
    main()
