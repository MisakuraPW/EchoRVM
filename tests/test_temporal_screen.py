import csv
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.temporal_mae import TemporalMAE
from tools.evaluate_temporal_mae import ProbeSegDataset, stream
from tools.evaluate_temporal_screen import check_output, validate_seg_context, summarize_quick, quick_ef


ROOT = Path(__file__).resolve().parents[1]


def config(mode='global', size=16, patch_size=4):
    return dict(name='temporal_mae', img_size=size, patch_size=patch_size,
                local_frames=4, clip_count=2, frames=8, tubelet_size=2, in_chans=3,
                embed_dim=24, depth=1, num_heads=3, decoder_embed_dim=24,
                decoder_depth=1, decoder_num_heads=3, memory_mode=mode, memory_grid=2,
                core_depth=1, separate_qv_bias=True, position_embedding='flat_sinusoid')


class ScreenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_memory_target_is_not_reset_boundary(self):
        for mode in ('global', 'spatial', 'dual'):
            torch.manual_seed(42)
            model = TemporalMAE(**config(mode)).eval()
            with self.assertRaisesRegex(ValueError, 'memory reset'):
                validate_seg_context(model, SimpleNamespace(seg_target_index=8))
            validate_seg_context(model, SimpleNamespace(seg_target_index=6))
            video = torch.rand(2, 16, 3, 16, 16)
            valid = torch.ones(2, 16, dtype=torch.bool)
            with torch.no_grad():
                normal = stream(model, video, valid)[0]
                reset = stream(model, video, valid, intervention='reset')[0]
            torch.testing.assert_close(normal[:, 8], reset[:, 8])
            self.assertGreater(float((normal[:, 6]-reset[:, 6]).abs().max()), 1e-5)

    def test_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            check_output(output, {'version': 1})
            check_output(output, {'version': 1})
            with self.assertRaises(RuntimeError):
                check_output(output, {'version': 2})

    def test_quick_cli_cache_and_optional_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            files, traces = ['FileName,EF,Split'], ['FileName,Frame,X1,Y1,X2,Y2']
            for i in range(12):
                name = f'case{i:02d}'
                files.append(f'{name},{40+2*i},{"TRAIN" if i < 8 else "VAL"}')
                raw = np.random.default_rng(i).integers(0, 255, (16,112,112), dtype=np.uint8)
                np.save(root/'npy'/f'{name}.npy', raw)
                for frame in (6, 12):
                    for y in (20,30,50,70,90):
                        traces.append(f'{name}.avi,{frame},30,{y},80,{y}')
            (root/'FileList.csv').write_text('\n'.join(files))
            cfg = config(size=112, patch_size=8)
            checkpoint = root/'epoch.pt'
            torch.save(dict(model_state_dict=TemporalMAE(**cfg).state_dict(), epoch=0,
                            config=dict(model=cfg, data=dict(input_protocol='gray_repeat3'))), checkpoint)
            env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
            common = [sys.executable, 'tools/evaluate_temporal_screen.py', '--checkpoint', str(checkpoint),
                      '--data_root', str(root), '--batch_size', '2', '--audit_frames', '8', '--smoke']
            output = root/'result/clip_mae_pool64/quick_audit/epoch_0000'

            def run(extra):
                result = subprocess.run(common+extra, cwd=ROOT, env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout+result.stderr)

            # EF screening does not require tracing annotations or run segmentation.
            run(['--output_dir', str(output)])
            metrics = json.loads((output/'metrics.json').read_text())
            self.assertNotIn('segmentation', metrics)
            self.assertNotIn('interventions', metrics)
            self.assertFalse(metrics['cache_hit'])
            self.assertTrue(metrics['state_only_ef'])
            with (output/'ef_predictions.csv').open() as handle:
                self.assertEqual({r['mode'] for r in csv.DictReader(handle)}, {'normal','state'})
            cached = root/'cached'
            run(['--output_dir', str(cached), '--feature_cache', str(output/'features.npz')])
            again = json.loads((cached/'metrics.json').read_text())
            self.assertTrue(again['cache_hit'])
            self.assertEqual(metrics['ef'], again['ef'])
            budget_args = SimpleNamespace(**metrics['protocol'])
            budget_args.feature_cache = str(output/'features.npz')
            budget_args.ef_budgets = [2,4,8]
            identity = json.loads((output/'screen_protocol.json').read_text())
            identity['ef_budgets'] = [2,4,8]
            budget_out = root/'budgets'
            budget_out.mkdir()
            with patch('tools.evaluate_temporal_screen.ef_features', side_effect=AssertionError('Cache should avoid encoding')):
                extra = quick_ef(TemporalMAE(**cfg), budget_args, torch.device('cpu'), budget_out, identity)
            self.assertEqual(set(extra['ef']), {'2','4','8'})
            with self.assertRaisesRegex(RuntimeError, 'cache does not match'):
                quick_ef(TemporalMAE(**cfg), budget_args, torch.device('cpu'), budget_out,
                         dict(identity, checkpoint_sha256='changed'))
            shutil.copytree(output, root/'result/hier_global/quick_audit/epoch_0000')
            summarize_quick(root/'result')
            self.assertTrue((root/'result/quick_comparison.csv').exists())
            with (root/'result/quick_paired_comparisons.csv').open() as handle:
                paired = list(csv.DictReader(handle))
            self.assertEqual(float(paired[0]['mae_difference']), 0.)
            self.assertEqual(float(paired[0]['ci95_high']), 0.)

            (root/'VolumeTracings.csv').write_text('\n'.join(traces))
            args = SimpleNamespace(audit_frames=8, seg_target_index=6, input_protocol='gray_repeat3',
                                   seed=42, seg_train_cases=2, seg_val_cases=2)
            dataset = ProbeSegDataset(root, 'train', args, 3)
            item, sample = dataset[0], dataset.samples[0]
            expected = np.load(root/'npy'/f'{sample["stem"]}.npy')[sample['frame']]
            torch.testing.assert_close(item['video'][6, 0], torch.from_numpy(expected).float()/255)
            self.assertEqual(item['target_index'], 6)
            segout = root/'seg'
            run(['--output_dir', str(segout), '--profile', 'seg', '--seg_target_index', '6'])
            seg = json.loads((segout/'metrics.json').read_text())
            self.assertIn('dice_mean', seg['segmentation'])
            self.assertNotIn('state_only_seg_dice', seg['segmentation'])
            self.assertNotIn('ef', seg)

    def test_runner_dry_run_profiles(self):
        common = [sys.executable, 'tools/run_temporal_research.py', '--dry_run', '--only', 'hier_dual']
        for profile, epochs in [('quick', '[0, 100, 200, 400]'),
                                ('full', '[0, 50, 100, 150, 200, 250, 300, 350, 400]')]:
            result = subprocess.run(common+['--audit_profile',profile], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertIn('epochs='+epochs, result.stdout)
            self.assertIn('Selected experiments=1/9', result.stdout)


if __name__ == '__main__':
    unittest.main()
