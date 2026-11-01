import tempfile
import json
import subprocess
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from models.final_temporal_mae import FinalTemporalMAE
from research_pipeline.echo import EchoExperiment, TemporalTask
from research_pipeline.readout_calibration import build
from research_pipeline.config import load, stage_config
from research_pipeline.engine import run_training
from research_pipeline.artifacts import layout


class ReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def reference(self, task):
        cfg = dict(img_size=112, patch_size=16, in_chans=3, local_frames=4,
                   clip_count=2, embed_dim=24, depth=1, num_heads=4,
                   decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3,
                   separate_qv_bias=True, decoder_embed_bias=False,
                   memory_mode='global', norm_eps=1e-6, frame_readout='repeat')
        backbone = FinalTemporalMAE(**cfg)
        task_model = TemporalTask(backbone, task, 'finetune', 8, 'final', False)
        exp = EchoExperiment(task_model, None, None, task, {}, True, False)
        exp.seed, exp.native_config, exp.pretraining_protocol = 42, cfg, {}
        return exp

    def test_seg_shared_weights_and_gradient(self):
        reference = self.reference('seg')
        with patch('research_pipeline.readout_calibration.build_reference', return_value=reference):
            exp = build({}, dict(kind='seg', mode='finetune'), 'fixture', torch.device('cpu'))
        self.assertEqual(len(exp.model.head.blocks), 4)
        self.assertEqual(exp.model.head.norm.normalized_shape, (192,))
        torch.testing.assert_close(exp.model.head.encoder_to_decoder.weight, exp.backbone.decoder_embed.weight)
        src = exp.backbone.decoder_blocks[0].attn
        dst = exp.model.head.blocks[0].attn
        torch.testing.assert_close(dst.qkv.bias, torch.cat((src.q_bias, torch.zeros_like(src.q_bias), src.v_bias)))
        sample = dict(video=torch.rand(8, 3, 112, 112), prefix_frames=0, target_index=6,
                      mask=torch.zeros(112, 112, dtype=torch.long))
        sample['mask'][25:70, 30:65] = 1
        loss = exp.loss([sample])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(exp.backbone.patch_embed.proj.weight.grad.abs().sum()), 0)
        self.assertGreater(float(exp.model.head.pos_embed.grad.abs().sum()), 0)
        self.assertIsNone(exp.backbone.decoder_pred.weight.grad)

    def test_ef_initialization_and_loss_units(self):
        with patch('research_pipeline.readout_calibration.build_reference', return_value=self.reference('ef')):
            exp = build({}, dict(kind='ef', mode='finetune'), 'fixture', torch.device('cpu'))
        features = torch.randn(2, 8, 24)
        pred = exp.model.head(features)
        torch.testing.assert_close(pred, torch.full((2,), .556))
        loss = exp._task_loss(pred, [dict(target=torch.tensor(60.)), dict(target=torch.tensor(50.))])
        self.assertAlmostEqual(float(loss.detach()), 5., places=4)
        loss.backward()
        self.assertGreater(float(exp.model.head.regress[-1].weight.grad.abs().sum()), 0)

    def test_constant_schedule_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load('configs/pipeline/synthetic.yaml', root=tmp, tag='constant')
            cfg['training'].update(lr_schedule='constant', epochs=2, warmup_epochs=0, tensorboard=False)
            current = stage_config(cfg, cfg['stages'][0])
            run_training(current, None)
            saved = torch.load(layout(cfg, current['stage']['id'])['ckpt']/'last.pt', weights_only=False)
            self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], cfg['training']['lr'])

    def test_launcher_file_entrypoint_outside_repository(self):
        script = Path(__file__).resolve().parents[1]/'tools/queue_readout_calibration.py'
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'execution_selection.json').write_text(json.dumps(dict(stages=[])))
            result = subprocess.run([sys.executable, str(script), '--wait-pid', '999999999',
                                     '--previous-result', tmp], cwd=tmp, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Previous comparison missing', result.stderr)
            self.assertNotIn('ModuleNotFoundError', result.stderr)


if __name__ == '__main__':
    unittest.main()
