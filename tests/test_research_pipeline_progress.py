"""Read-only progress rendering and queue display must not require CUDA or torch."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.watch_pipeline_progress import latest_metrics, render_stage, run_view, StageDisplay


class ProgressTests(unittest.TestCase):
    def test_training_and_autotune_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'status.json').write_text(json.dumps(dict(phase='training',epoch=2,epochs=1600,
                completed=500,total=1000,loss=.8,lr=1e-5,samples_per_second=10,time=100)),encoding='utf-8')
            text = render_stage(root,'pretrain',now=105)
            for expected in ('epoch 2/1600','50.0%','loss=0.800000','epoch ETA~00:50','heartbeat_age=5s','lr=1e-05'):
                self.assertIn(expected,text)
            (root/'status.json').write_text(json.dumps(dict(phase='autotune',trial=3,
                selection=dict(micro_batch=8,num_workers=4,gradient_checkpointing=False))),encoding='utf-8')
            self.assertIn('trial=3 batch=8 workers=4',render_stage(root,'pretrain'))

    def test_csv_quotes_partial_files_and_one_shot_view(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stage = root/'result'/'tag'/'pretrain'
            stage.mkdir(parents=True)
            (stage/'metrics.csv').write_text('epoch,lr,prefix_histogram_json\n1,0.001,"{""0"": 4}"\n',encoding='utf-8-sig')
            self.assertEqual(latest_metrics(stage/'metrics.csv')['epoch'],'1')
            (stage/'status.json').write_text('{',encoding='utf-8')
            self.assertIn('waiting_for_worker',render_stage(stage,'pretrain'))
            (stage.parent/'queue.jsonl').write_text('{"stage":"pretrain","state":"starting"}\n{partial',encoding='utf-8')
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                run_view(root,'tag',once=True)
            self.assertIn('stage: pretrain',output.getvalue())
            self.assertNotIn('?/?',output.getvalue())

    def test_observer_does_not_change_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'status.json'
            original = '{"phase":"paused_at_budget","epoch":400}'
            path.write_text(original)
            with contextlib.redirect_stdout(io.StringIO()):
                with StageDisplay(tmp,'A',interval=.01):
                    pass
            self.assertEqual(path.read_text(),original)


if __name__=='__main__':
    unittest.main()
