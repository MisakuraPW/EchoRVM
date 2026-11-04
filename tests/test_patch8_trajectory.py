import json
from pathlib import Path
import tempfile
import unittest

from research_pipeline.config import load, validate
from tools.run_patch8_trajectory import build_config, parent_failed


class TrajectoryTests(unittest.TestCase):
    def test_only_existing_snapshot_frozen_probes(self):
        plan = load('configs/pipeline/clip_recurrent_patch8.yaml')
        plan['stages'] = [dict(id=f'{task}_linear_1600', kind=task,
                              mode='frozen_linear', source='/old/epoch_1600.pt')
                          for task in ('ef', 'seg')]
        original = json.dumps(plan, sort_keys=True)
        cfg = validate(build_config(plan, Path('/outputs'), 'parent'))
        self.assertEqual(json.dumps(plan, sort_keys=True), original)
        self.assertEqual(cfg['run']['tag'], 'parent_trajectory')
        self.assertEqual([s['id'] for s in cfg['stages']],
                         ['ef_linear_0400', 'seg_linear_0400',
                          'ef_linear_0800', 'seg_linear_0800'])
        for stage in cfg['stages']:
            self.assertEqual(stage['mode'], 'frozen_linear')
            self.assertEqual(Path(stage['source']).name,
                             f"epoch_{stage['id'][-4:]}.pt")
            self.assertEqual(Path(stage['source']).parent.name, 'pretrain')
        self.assertEqual(cfg['model'], plan['model'])
        self.assertEqual(cfg['data'], plan['data'])
        self.assertEqual(cfg['training'], plan['training'])

    def test_parent_failure_is_not_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(parent_failed(root))
            (root / 'queue.jsonl').write_text('{"state":"failed"}\n')
            self.assertTrue(parent_failed(root))
            (root / 'queue.jsonl').write_text('{"state":"starting"}\n')
            self.assertFalse(parent_failed(root))
            stage = root / 'ef_l1_ft'
            stage.mkdir()
            (stage / 'status.json').write_text('{"phase":"failed"}')
            self.assertTrue(parent_failed(root))


if __name__ == '__main__':
    unittest.main()
