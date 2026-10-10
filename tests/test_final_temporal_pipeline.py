import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import gc

import torch

from models.final_temporal_mae import FinalTemporalMAE
from models.temporal_mae import TemporalMAE
from test_final_temporal_tasks import _fixture, _config


class FinalPipelineTests(unittest.TestCase):
    def test_real_file_tiny_full_queue_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            arguments = [sys.executable, 'tools/run_temporal_final.py', '--smoke', '--device','cpu',
                         '--run_tag','test_smoke', '--data_root', str(root),
                         '--output_root',str(root/'outputs'), '--min_free_gb','0',
                         '--cache_disk_gb','0.01', '--cache_ram_gb','0.01']
            for name, frame in (('P','repeat'), ('C','learned'), ('F','factorized')):
                cfg = dict(_config(), frame_readout=frame, dynamic_rank=4, memory_compression='temporal_attention')
                if name == 'P':
                    cfg.pop('frame_readout')
                torch.manual_seed(42)
                path = root/(name+'.pt')
                torch.save(dict(model_state_dict=TemporalMAE(**cfg).state_dict(),
                                config=dict(model=cfg), epoch=100), path)
                arguments.extend(['--checkpoint',name+'='+str(path)])
            from tools import run_temporal_final as queue
            from tools import final_temporal_worker as worker

            def dispatch(command, label, result_root, times):
                # Same real worker dispatcher, with imports cached for CPU CI.
                with mock.patch.object(sys, 'argv', command[1:]):
                    worker.main()
                gc.collect()

            for attempt in range(2):
                with mock.patch.object(sys, 'argv', arguments[1:]), mock.patch.object(
                        queue, 'run_command', side_effect=dispatch if not attempt else
                        AssertionError('A completed queue must not dispatch workers again')):
                    queue.main()
            out = root/'outputs/result/test_smoke'
            self.assertTrue(json.loads((out/'SMOKE_DONE').read_text())['passed'])
            self.assertFalse(json.loads((out/'SMOKE_DONE').read_text())['scientific_questions_closed'])
            self.assertTrue((out/'B7/adapt/DONE').exists())
            self.assertTrue((out/'B8/adapt/DONE').exists())
            self.assertTrue(list(out.glob('*/streaming/DONE')))
            self.assertFalse(list(out.rglob('*.pt')))
            self.assertFalse(list((root/'outputs/cache').rglob('*.pt')))


if __name__ == '__main__':
    unittest.main()
