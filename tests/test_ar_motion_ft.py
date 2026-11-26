import copy
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from models.forward_temporal_mae import ForwardTemporalMAE
from research_pipeline.ar_motion_ft import MotionExperiment, MotionTask, TrainAreas, area_log_targets, window_target
from research_pipeline.echo import TemporalTask
from tools.run_ar_motion_aux import check_draws, reference_windows
from tools.run_ar_motion_ft import evaluate_arm, second_arm


def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    b = ForwardTemporalMAE(img_size=16,patch_size=8,local_frames=16,clip_count=4,embed_dim=16,
        depth=7,num_heads=2,decoder_embed_dim=16,decoder_depth=1,decoder_num_heads=2,tubelet_size=2,
        in_chans=3,core_depth=1,memory_grid=1,memory_mode='global',memory_read_location='tokens',
        memory_write_source='fused',memory_compression='mean',frame_readout='repeat',
        reconstruction_recent_frames=64,mask_ratio=.9,forward_variant='mid_read',read_after_block=6)
    base = TemporalTask(b,'ef','finetune',64,'final',False)
    rows = [dict(video=torch.rand(64,3,16,16),prefix_frames=0,fps=50.+i,target=torch.tensor(42.+i),
                 motion_target=(5,12,math.log(.5))) for i in range(2)]
    return base,rows


class MotionTests(unittest.TestCase):
    def test_new_arms_align_to_ar_and_have_no_future_leakage(self):
        base, rows = fixture()
        for arm in ('M1','M1_aux','M2'):
            model = MotionTask(copy.deepcopy(base),arm).eval()
            with torch.no_grad():
                torch.testing.assert_close(model(rows),base(rows),rtol=0,atol=0)
                bare = [{k:v for k,v in r.items() if k not in ('target','motion_target')} for r in rows]
                torch.testing.assert_close(model(bare),model(rows),rtol=0,atol=0)
                if arm=='M2':
                    model.sparse.out.weight.normal_(0,.04)
                if arm=='M1':
                    model.motion.write.weight.normal_(0,.04)
                capture = []
                hook = model.head.register_forward_pre_hook(lambda m,a:capture.append(a[0]))
                model(rows)
                later = [dict(r,video=torch.cat((r['video'][:32],torch.zeros_like(r['video'][32:])))) for r in rows]
                model(later)
                hook.remove()
                torch.testing.assert_close(capture[0][:,:32],capture[1][:,:32],rtol=0,atol=0)

    def test_gradient_from_ef_alone_and_aux_only_control(self):
        base, rows = fixture()
        for arm in ('M1','M1_aux','M2'):
            m = MotionTask(copy.deepcopy(base),arm)
            exp = MotionExperiment(m,None,None,'ef',{})
            optim = torch.optim.AdamW(m.parameters(),lr=1e-4)
            for _ in range(2):
                optim.zero_grad(set_to_none=True)
                exp.loss(rows).backward()
                optim.step()
            if arm=='M1_aux':
                self.assertIsNone(m.motion.write.weight.grad)
                self.assertGreater(float(m.motion.area.weight.grad.abs().sum()),0)
            else:
                optim.zero_grad(set_to_none=True)
                m(rows).square().sum().backward()
                weight = m.motion.write.weight if arm=='M1' else m.sparse.out.weight
                project = m.motion.project[1].weight if arm=='M1' else m.sparse.keys.weight
                self.assertGreater(float(weight.grad.abs().sum()),0)
                self.assertGreater(float(project.grad.abs().sum()),0)

    def test_only_real_train_pairs_and_fixed_branch(self):
        first,second = dict(frame=3,area=20),dict(frame=18,area=10)
        import numpy as np
        values = area_log_targets({'a':dict(traces=[second,first]),
                                   'b':dict(traces=[dict(first,area=5),second])},
                                 lambda trace:np.ones(trace['area']))
        self.assertEqual(set(values),{'a'})
        self.assertEqual(window_target(values['a'],2),(0,8,math.log(.5)))
        self.assertIsNone(window_target(values['a'],4))
        self.assertIsNone(window_target((3,4,-.4),3))
        class NotTrain:
            split = 'val'
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            TrainAreas(NotTrain())
        self.assertEqual(second_arm(.1),'M1_aux')
        self.assertEqual(second_arm(.099),'M2')

    def test_aux_runner_matches_archived_training_and_val_windows(self):
        with TemporaryDirectory() as directory:
            p = Path(directory)
            old,new = (p/name for name in ('old','new'))
            old.mkdir()
            new.mkdir()
            rows = 'epoch,window_sequence_sha256\n1,same\n'
            (old/'metrics.csv').write_text(rows,encoding='utf-8')
            (new/'metrics.csv').write_text(rows+'2,extra\n',encoding='utf-8')
            self.assertEqual(check_draws(old,new),[1])
            (new/'metrics.csv').write_text(rows.replace('same','different'),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'different'):
                check_draws(old,new)
            base = [dict(patient='p',start=4,target=50.,prediction=55.)]
            for name in ('single','five'):
                folder = old/name
                folder.mkdir()
                (folder/'candidate_windows.csv').write_text(
                    'patient,recent_start,target,prediction\np,4,50.0,54.0\n',encoding='utf-8-sig')
            self.assertEqual(reference_windows(old,(base,base))[0][0]['prediction'],54.)
            with self.assertRaisesRegex(ValueError,'differ'):
                reference_windows(old,(base,[dict(base[0],start=5)]))
            with self.assertRaisesRegex(ValueError,'development-only'):
                evaluate_arm(None,None,'test',None,None,old,residual=False)
