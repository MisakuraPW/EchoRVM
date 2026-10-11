import json
from pathlib import Path
import tempfile
import unittest

from tools.poll_temporal_final import snapshot


class MonitorPathsTests(unittest.TestCase):
    def test_current_stage_logs_and_handoffs_resolve_under_result_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'result/run'
            phase = root / 'C/ft/ef'
            (phase / 'logs').mkdir(parents=True)
            (root / 'operations').mkdir()
            (root / 'current_stage.json').write_text(json.dumps(dict(stage='C/ft/ef', status='running')))
            (phase / 'logs/train.log').write_text('actual task log\n')
            (phase / 'logs/metrics.csv').write_text('epoch,successful_updates\n1,619\n2,1238\n')
            (phase / 'protocol.json').write_text(json.dumps(dict(patience=10, runtime=dict(workers=8))))
            (root / 'operations/early_stopping_status.json').write_text(json.dumps(dict(state='handoff_complete')))
            result = snapshot(root, Path(folder) / 'no_proc', gpu=False)
            self.assertEqual(result['errors'], [])
            self.assertEqual(result['stage_result_dir'], str(phase.resolve()))
            self.assertEqual(result['latest_epoch']['successful_updates'], '1238')
            self.assertEqual(result['runtime']['patience'], 10)
            self.assertEqual(result['runtime']['workers'], 8)
            self.assertEqual(result['handoffs']['early_stopping']['state'], 'handoff_complete')
            self.assertIn('actual task log', result['train_log_tail'])
            self.assertFalse(result['final_complete'])

    def test_wrong_root_is_reported_not_silently_used_as_project(self):
        with tempfile.TemporaryDirectory() as folder:
            result = snapshot(folder, Path(folder) / 'no_proc', gpu=False)
            self.assertIsNone(result['stage'])
            self.assertIn('wrong result directory', result['errors'][0]['error'])

    def test_stage_cannot_escape_result_dir(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'current_stage.json').write_text(json.dumps(dict(stage='../elsewhere', status='running')))
            result = snapshot(root, root / 'no_proc', gpu=False)
            self.assertNotIn('stage_result_dir', result)
            self.assertEqual(result['errors'][0]['error'], 'unsafe stage path')


if __name__ == '__main__':
    unittest.main()
