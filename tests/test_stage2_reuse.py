import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from tools.run_stage2_questions import main, stage_config, validate_reused_control


class Stage2ReuseTests(unittest.TestCase):
    def setUp(self):
        args = SimpleNamespace(seed=42, data_root='/data/echo', num_workers=8,
            init_checkpoint='ckpt/mae/videomae_vit_s.pth', epochs=100, smoke=False,
            local_frames=16)
        self.cfg = stage_config(args, 'repeat', Path('/output/ckpt/repeat'))

    def validate(self, config, epoch=100, partial=False):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'epoch_0100.pt'
            torch.save(dict(config=config, epoch=epoch, partial_epoch=partial,
                            model_state_dict={'sentinel': torch.ones(1)}), path)
            return validate_reused_control(path, self.cfg)

    def test_accepts_stage1_name_and_runtime_changes(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['experiment']['name'] = 'spatial_l16'
        cfg['model'].pop('frame_readout')
        cfg['model']['gradient_checkpointing'] = False
        cfg['data']['num_workers'] = 0
        cfg['checkpoint']['dir'] = '/old/ckpt'
        cfg['train'].update(batch_size=32, grad_accum_steps=1, epoch_sample_batch=8)
        result = self.validate(cfg)
        self.assertEqual(result['epoch'], 100)
        self.assertEqual(result['training_contract']['train']['effective_batch'], 32)

    def test_rejects_changed_scientific_settings(self):
        for section, key, value in [('model', 'local_frames', 8),
                ('model', 'memory_mode', 'global'), ('model', 'frame_readout', 'learned'),
                ('optimizer', 'lr', .0002), ('data', 'train_split', 'val'),
                ('experiment', 'seed', 7), ('train', 'batch_size', 16),
                ('train', 'epoch_sample_batch', 32), ('train', 'epochs', 400)]:
            with self.subTest(key=key):
                cfg = copy.deepcopy(self.cfg)
                cfg[section][key] = value
                with self.assertRaisesRegex(ValueError, 'protocol mismatch'):
                    self.validate(cfg)

    def test_rejects_wrong_epoch_and_partial(self):
        for epoch, partial in [(400, False), (100, True)]:
            with self.assertRaisesRegex(ValueError, 'complete checkpoint'):
                self.validate(self.cfg, epoch, partial)

    def test_reuse_queue_never_launches_training_or_copies_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root/'data'
            data.mkdir()
            for name in ('FileList.csv', 'VolumeTracings.csv'):
                (data/name).write_text('fixture\n')
            init = root/'init.pt'
            init.write_bytes(b'fixture')
            args = SimpleNamespace(seed=42, data_root=str(data), num_workers=8,
                init_checkpoint=str(init), epochs=100, smoke=False, local_frames=16,
                phase='train', variants=['repeat'], checkpoint=None,
                reuse_control=str(root/'control.pt'), output_root=str(root/'outputs'),
                run_tag='reuse', ef_head='attention', eval_batch_size=0,
                autotune=True, dry_run=False)
            cfg = stage_config(args, 'repeat', root/'old_ckpt')
            torch.save(dict(config=cfg, epoch=100,
                model_state_dict={'sentinel': torch.ones(1)}), args.reuse_control)
            with patch('tools.run_stage2_questions.parse_args', return_value=args), \
                 patch('tools.run_stage2_questions.run_command') as run, \
                 patch('tools.run_stage2_questions.summarize'), \
                 patch('tools.run_stage2_questions.archive_analysis'):
                main()
            self.assertEqual(run.call_count, 1)
            cmd, stage = run.call_args.args[:2]
            self.assertEqual(stage, 'repeat/endpoint')
            self.assertIn('tools/evaluate_stage2.py', cmd)
            self.assertEqual(cmd[cmd.index('--checkpoint')+1], str(Path(args.reuse_control).resolve()))
            self.assertFalse((root/'outputs/reuse/ckpt').exists())
            self.assertTrue((root/'outputs/reuse/result/repeat/reused_control.json').exists())


if __name__ == '__main__':
    unittest.main()
