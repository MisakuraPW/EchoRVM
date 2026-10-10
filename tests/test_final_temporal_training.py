import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from models.final_temporal_mae import FinalTemporalMAE
from test_final_temporal_tasks import _fixture, _config
from utils.final_temporal_training import run_warm_job, model_digest
import utils.final_temporal_training as training


class WarmTrainingTests(unittest.TestCase):
    def test_success_boundaries_resume_identically_and_preserve_counts(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = _fixture(root)
            cfg = _config()
            source = root / 'source.pt'
            torch.manual_seed(19)
            torch.save(dict(model_state_dict=FinalTemporalMAE(**cfg).state_dict(),
                            config=dict(model=cfg), epoch=100), source)

            def job(name):
                return dict(checkpoint=str(source), output_dir=str(root/name/'result'),
                            checkpoint_dir=str(root/name/'ckpt'), updates=3, effective_batch=3,
                            micro_batch=1, num_workers=0, autotune=False, smoke=True,
                            recent_frames=8, max_prefix=8, save_every_updates=1, min_free_gb=0,
                            overrides=dict(gradient_checkpointing=True), seed=42)

            reference = job('reference')
            a = run_warm_job(reference, manifest, 'cpu')
            interrupted = job('interrupted')
            original = training.atomic_torch_save

            def failing_save(payload, path, *args, **kwargs):
                if Path(path).name == 'last.pt' and payload['cursor'] == 2:
                    raise KeyboardInterrupt()
                return original(payload, path, *args, **kwargs)

            with mock.patch.object(training, 'atomic_torch_save', side_effect=failing_save):
                with self.assertRaises(KeyboardInterrupt):
                    run_warm_job(interrupted, manifest, 'cpu')
            b = run_warm_job(interrupted, manifest, 'cpu')
            astate = torch.load(Path(reference['checkpoint_dir'])/'final.pt', weights_only=False)['model_state_dict']
            bstate = torch.load(Path(interrupted['checkpoint_dir'])/'final.pt', weights_only=False)['model_state_dict']
            for key in astate:
                torch.testing.assert_close(astate[key], bstate[key], rtol=0, atol=0)
            self.assertEqual(a['prefix_sample_counts'], b['prefix_sample_counts'])
            self.assertEqual(sum(b['prefix_sample_counts'].values()), 9)
            self.assertEqual(a['training_coverage'], b['training_coverage'])
            self.assertEqual(b['training_coverage']['samples'], 9)
            self.assertEqual(b['training_coverage']['recent_reconstruction_frames'], 9 * 8)
            self.assertEqual(b['training_coverage']['prefix_frames'],
                             sum(int(h) * count for h, count in b['prefix_sample_counts'].items()))
            rows = [json.loads(row) for row in (root/'interrupted/result/logs/train_metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['step'] for row in rows], [1,2,3])
            self.assertIn('weighted_orthogonal_loss', rows[-1])


if __name__ == '__main__':
    unittest.main()
