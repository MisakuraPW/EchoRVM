import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import torch

from models.final_temporal_mae import FinalTemporalMAE
from research_pipeline.config import load, stage_config
from research_pipeline.mae_execution import active_clip_loss
from research_pipeline.ssl_data import build_ssl_manifest, SSLWindowDataset
from utils.pretrained_init import load_videomae_init


class Patch8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_recipe_budget_and_smoke_references(self):
        cfg = load('configs/pipeline/clip_recurrent_patch8.yaml')
        stage = stage_config(cfg, cfg['stages'][0])
        self.assertEqual(stage['training']['epochs'], 400)
        self.assertIsNone(stage['training']['patience'])
        self.assertEqual(stage['training']['effective_batch'], 64)
        self.assertEqual(cfg['model']['config']['patch_size'], 8)
        self.assertEqual(cfg['data']['max_prefix'], 0)
        self.assertEqual(len(cfg['stages']), 7)
        smoke = load('configs/pipeline/clip_recurrent_patch8.yaml', smoke=True)
        self.assertEqual(smoke['stages'][1]['source'], '@pretrain/epoch_0000')
        self.assertEqual(smoke['stages'][-1]['source'], '@pretrain/final')

    def test_one_real_window_per_video_without_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            rows = []
            for patient, split, frames in [('a', 'TRAIN', 90), ('b', 'VAL', 40), ('c', 'TEST', 7)]:
                np.save(root/'npy'/f'{patient}.npy', np.zeros((frames, 112, 112), dtype=np.uint8))
                rows.append(dict(FileName=patient, Split=split, FPS=50))
            pd.DataFrame(rows).to_csv(root/'FileList.csv', index=False)
            inventory = build_ssl_manifest(root, max_prefix=0)
            dataset = SSLWindowDataset(inventory, True)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(len(inventory['excluded']), 1)
            for index in range(len(dataset)):
                sample = dataset[index]
                self.assertEqual(sample['prefix_frames'], 0)
                self.assertEqual(len(sample['video']), 64 if sample['patient']=='a' else 32)
                self.assertTrue(bool((sample['frame_indices'].diff()==1).all()))
                self.assertNotIn('target', sample)
            self.assertEqual(len(dataset.calibration_samples(1)[0]['video']), 64)

    def test_patch_conversion_mask_budget_and_recurrent_gradients(self):
        cfg = dict(img_size=112, local_frames=16, clip_count=4, in_chans=3,
                   embed_dim=24, depth=1, num_heads=4, decoder_embed_dim=24,
                   decoder_depth=1, decoder_num_heads=4, norm_pix_loss=True,
                   reconstruction_recent_frames=64, memory_mode='global', mask_ratio=.9)
        source = FinalTemporalMAE(**cfg, patch_size=16)
        model = FinalTemporalMAE(**cfg, patch_size=8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'initial.pt'
            torch.save({'model': source.state_dict()}, path)
            report = load_videomae_init(model, path)
        self.assertFalse(report['missing_encoder_keys'])
        self.assertEqual(set(report['skipped']), {'decoder_pred.weight', 'decoder_pred.bias'})
        self.assertEqual(model.token_grid, (8, 14, 14))
        sample = dict(video=torch.rand(64, 3, 112, 112), record_index=0)
        loss = active_clip_loss(model, [sample], 42)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.patch_embed.proj.weight.grad.abs().sum()), 0)
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0 for p in model.memory.parameters()))


if __name__ == '__main__':
    unittest.main()
