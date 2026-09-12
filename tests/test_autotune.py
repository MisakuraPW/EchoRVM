import copy
import logging
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, RandomSampler

from utils.autotune import (SampleBudgetBatchSampler, apply_selection,
                            batch_candidates, choose_trial, fingerprint)


def config():
    return dict(model=dict(name='temporal_mae', frames=64, gradient_checkpointing=True),
                train=dict(batch_size=8, grad_accum_steps=4, epochs=400,
                           mixed_precision=False, stop_on_nan=True),
                data=dict(sampling_protocol='temporal_v1', num_workers=8),
                optimizer=dict(lr=.0001), logging=dict(use_tqdm=False))


class AutotuneTests(unittest.TestCase):
    def test_safe_divisors_and_contrastive_lock(self):
        c = config()
        self.assertEqual(batch_candidates(c), [1, 2, 4, 8, 16, 32])
        c['model']['two_views'] = True
        self.assertEqual(batch_candidates(c), [8])
        with self.assertRaises(ValueError):
            apply_selection(c, dict(batch_size=16, gradient_checkpointing=True, num_workers=4))

    def test_selection_only_changes_runtime(self):
        c = config()
        before = copy.deepcopy(c)
        tuned = apply_selection(c, dict(batch_size=16, gradient_checkpointing=False, num_workers=4))
        self.assertEqual(c, before)
        self.assertEqual(tuned['train']['batch_size'] * tuned['train']['grad_accum_steps'], 32)
        self.assertEqual(tuned['train']['epoch_sample_batch'], 8)
        self.assertEqual(tuned['model']['frames'], 64)
        self.assertEqual(tuned['optimizer'], c['optimizer'])
        self.assertEqual(fingerprint(c), fingerprint(before))

    def test_selection_uses_speed_with_headroom_tie_break(self):
        trials = [dict(status='oom'),
                  dict(status='ok', samples_per_second=100, peak_reserved_bytes=100, num_workers=4),
                  dict(status='ok', samples_per_second=99, peak_reserved_bytes=80, num_workers=8),
                  dict(status='unsafe', samples_per_second=1000)]
        self.assertEqual(choose_trial(trials)['peak_reserved_bytes'], 80)
        with self.assertRaises(RuntimeError):
            choose_trial([dict(status='oom')])

    def test_sample_budget_same_order_and_tail(self):
        for micro in (1, 8, 16, 32):
            gen = torch.Generator().manual_seed(42)
            sampler = SampleBudgetBatchSampler(RandomSampler(range(41), generator=gen), 40, micro)
            batches = list(sampler)
            indices = [i for b in batches for i in b]
            self.assertEqual(len(indices), 40)
            if micro == 1:
                reference = indices
            self.assertEqual(indices, reference)
            self.assertEqual(len(sampler), len(batches))

    def test_weighted_tail_matches_original_optimizer_updates(self):
        from trainers.train_rmae import run_epoch

        class Net(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(.5))

            def forward(self, video):
                return dict(loss=((video * self.weight - 1) ** 2).mean())

        data = [dict(video=torch.tensor([i / 41.])) for i in range(41)]
        final = []
        for micro in (8, 16, 32):
            c = apply_selection(config(), dict(batch_size=micro, gradient_checkpointing=True, num_workers=0))
            sampler = SampleBudgetBatchSampler(RandomSampler(data, generator=torch.Generator()), 40, micro)
            loader = DataLoader(data, batch_sampler=sampler)
            model = Net()
            optimizer = torch.optim.SGD(model.parameters(), lr=.1)
            scaler = torch.amp.GradScaler('cuda', enabled=False)
            _, steps = run_epoch(model, loader, optimizer, None, scaler, torch.device('cpu'), c,
                                 1, 0, True, None, logging.getLogger('autotune-test'))
            self.assertEqual(steps, 2)
            final.append(model.weight.detach())
        for actual in final[1:]:
            torch.testing.assert_close(actual, final[0])

    def test_cpu_calibration_does_not_change_config(self):
        from tools.tune_rmae_runtime import calibrate
        c = config()
        with patch('torch.cuda.is_available', return_value=False):
            self.assertIs(calibrate(c, 'unused'), c)

    def test_calibration_cache_is_reused_and_rejects_protocol_change(self):
        from tools.tune_rmae_runtime import calibrate

        class FakeProcess:
            returncode = 0

            def __init__(self, command, **kwargs):
                request = json.loads(Path(command[command.index('--trial') + 1]).read_text())
                selection = request['selection']
                result = dict(selection, status='ok',
                              samples_per_second=selection['batch_size'] * 10,
                              peak_reserved_bytes=100)
                Path(command[command.index('--result') + 1]).write_text(json.dumps(result))

            def wait(self, **kwargs):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            with patch('torch.cuda.is_available', return_value=True), \
                 patch('tools.tune_rmae_runtime.hardware', return_value={'gpu': 'fake'}):
                with patch('tools.tune_rmae_runtime.subprocess.Popen', side_effect=FakeProcess):
                    first = calibrate(config(), directory)
                self.assertEqual(first['train']['batch_size'], 32)
                with patch('tools.tune_rmae_runtime.subprocess.Popen') as launch:
                    second = calibrate(config(), directory)
                    launch.assert_not_called()
                self.assertEqual(first, second)
                changed = config()
                changed['optimizer']['lr'] *= 2
                with self.assertRaises(RuntimeError):
                    calibrate(changed, directory)


if __name__ == '__main__':
    unittest.main()
