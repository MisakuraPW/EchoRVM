import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tools import handoff_temporal_validation_cache as handoff
from tools.check_frozen_validation_cache import certify
from models.final_temporal_mae import FinalTemporalMAE
from test_final_temporal_tasks import _fixture,_config
from utils.final_temporal_tasks import run_task_job
from utils.final_temporal_training import write_json,file_digest


class CacheHandoffTests(unittest.TestCase):
    def test_controller_identity_and_suspension_required(self):
        record=dict(controller_pid=123,controller_start_ticks='456',controller_command='owned')
        current=dict(start='456',command='owned',cwd=str(handoff.ROOT),state='T')
        with mock.patch.object(handoff,'process',return_value=current):
            handoff.verified_controller(record)
        for field,value in (('start','reused'),('command','other_project'),('cwd','elsewhere'),('state','R')):
            changed=dict(current,**{field:value})
            with mock.patch.object(handoff,'process',return_value=changed),self.assertRaises(ValueError):
                handoff.verified_controller(record)

    def test_boundary_seal_required_no_partial_or_corrupted_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); out=root/'P/frozen/seg'; out.mkdir(parents=True)
            job=dict(output_dir=str(out),checkpoint_dir=str(root/'weights'))
            write_json(root/'jobs/P__frozen__seg.json',job)
            write_json(out/'metrics.json',dict(metric=.9))
            write_json(out/'DONE',dict(artifacts={'metrics.json':file_digest(out/'metrics.json')}))
            record=dict(boundary_stage='P/frozen/seg')
            self.assertEqual(handoff.verify_boundary(root,record),job)
            write_json(out/'metrics.json',dict(metric=0))
            with self.assertRaises(ValueError):handoff.verify_boundary(root,record)

    def test_real_file_cpu_equivalence_certificate(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); manifest=_fixture(root); cfg=_config()
            path=root/'source.pt'
            torch.save(dict(model_state_dict=FinalTemporalMAE(**cfg).state_dict(),config=dict(model=cfg),epoch=100),path)
            job=dict(checkpoint=str(path),output_dir=str(root/'result'),checkpoint_dir=str(root/'weights'),
                     cache_dir=str(root/'cache'),task='seg',freeze=True,epochs=1,patience=12,
                     seed=42,num_workers=0,micro_batch=2,effective_batch=3,autotune=False,smoke=True,
                     head_dim=12,head_depth=1,head_heads=3,recent_frames=8,max_prefix=8,precision='fp32',quiet=True)
            metrics=run_task_job(job,manifest,'cpu')
            output=root/'operations/certificate.json'
            result=certify(job,manifest,metrics['best_checkpoint'],output,'cpu')
            self.assertTrue(result['passed'])
            self.assertEqual(result['encoder_calls_on_replay'],0)
            self.assertTrue(result['exact_dice_match'])


if __name__=='__main__':unittest.main()
