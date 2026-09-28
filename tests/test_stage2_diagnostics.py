import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from models.temporal_mae import TemporalMAE
from utils.stage2_diagnostics import extract_batch, tubelet_weights, paired_bootstrap, OffsetReadout, tune_batch
from tools.diagnose_stage2_interfaces import SegViews, validity_rows

ROOT = Path(__file__).resolve().parents[1]


def config(mode='spatial', size=16):
    return dict(name='temporal_mae', img_size=size,patch_size=4 if size==16 else 8,
                local_frames=4,clip_count=2,frames=8,tubelet_size=2,in_chans=3,
                embed_dim=24,depth=1,num_heads=3,decoder_embed_dim=24,decoder_depth=1,
                decoder_num_heads=3,memory_mode=mode,memory_grid=2,core_depth=1,
                separate_qv_bias=True,position_embedding='flat_sinusoid')


class Stage2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_instrumentation_preserves_forward_and_state(self):
        for mode in ('none','global','spatial','dual'):
            torch.manual_seed(3)
            model = TemporalMAE(**config(mode)).eval().requires_grad_(False)
            before = {k:v.clone() for k,v in model.state_dict().items()}
            video = torch.rand(2,8,3,16,16)
            valid = torch.ones(2,8,dtype=torch.bool)
            valid[1,-1] = False
            with torch.inference_mode():
                old = model.state_trajectory(video,valid)
                diag = model.diagnostic_features(video,valid)
                one = model.diagnostic_features(video[:1],valid[:1])
            torch.testing.assert_close(old['features'],diag['features'],rtol=0,atol=0)
            torch.testing.assert_close(old['states'],diag['states'],rtol=0,atol=0)
            torch.testing.assert_close(one['features'],diag['features'][:1])
            torch.testing.assert_close(diag['local_features'][:,:2],diag['features'][:,:2])
            if mode=='none':
                torch.testing.assert_close(diag['local_features'],diag['features'],rtol=0,atol=0)
            else:
                self.assertGreater(float((diag['features'][:,2:]-diag['local_features'][:,2:]).abs().max()),1e-5)
            for key,value in before.items():
                torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_reduction_and_partial_tubelet(self):
        model = TemporalMAE(**config()).eval()
        video = torch.rand(2,16,3,16,16)
        valid = torch.ones(2,16,dtype=torch.bool)
        valid[1,-1] = False
        result = extract_batch(model,video,valid,torch.device('cpu'),torch.tensor([6,14]))
        with torch.inference_mode():
            tokens = torch.cat([model.state_trajectory(video[:,i:i+8],valid[:,i:i+8])['features'] for i in (0,8)],1)
        good,partial = tubelet_weights(valid,2)
        manual = (tokens.mean(2)*good[:,:,None]).sum(1)/good.sum(1)[:,None]
        torch.testing.assert_close(result['pooled']['fused'],manual)
        torch.testing.assert_close(result['pooled']['legacy_fused'][1],manual[1]*14/15)
        self.assertEqual(result['partial'].tolist(),[0,1])
        self.assertEqual(float(result['maps']['fused'][1].abs().sum()),0.)
        torch.testing.assert_close(result['maps']['fused'][0].flatten(1).T,tokens[0,3])
        torch.testing.assert_close(result['maps']['fused_mean'][0].flatten(1).T,tokens[0,2:4].mean(0))
        with self.assertRaisesRegex(ValueError,'no complete'):
            extract_batch(model,video[:1,:8],torch.zeros(1,8,dtype=torch.bool),torch.device('cpu'))

    def test_bootstrap_clusters_views_by_patient(self):
        a = [dict(id='p:1:v0',patient='p',dice=.8),dict(id='p:1:v1',patient='p',dice=.6),
             dict(id='q:2:v0',patient='q',dice=.4)]
        b = [dict(r,dice=.4) for r in a]
        result = paired_bootstrap(a,b,'dice',repetitions=100)
        self.assertEqual(result['patients'],2)
        self.assertAlmostEqual(result['delta'],.15)
        with self.assertRaises(ValueError):
            paired_bootstrap(a,b[:1],'dice')

    def test_offset_heads_equal_budget_and_frame_identity(self):
        torch.manual_seed(5)
        a = OffsetReadout(24,2,False)
        torch.manual_seed(5)
        b = OffsetReadout(24,2,True)
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        x = torch.rand(1,24,4,4).repeat(2,1,1,1)
        offset = torch.tensor([0,1])
        torch.testing.assert_close(a(x,offset)[0],a(x,offset)[1])
        self.assertGreater(float((b(x,offset)[0]-b(x,offset)[1]).detach().abs().max()),1e-5)

    def test_forward_only_batch_tuning(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = TemporalMAE(**config()).to(device).eval().requires_grad_(False)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        sample = dict(video=torch.rand(8,3,16,16),frame_valid=torch.ones(8,dtype=torch.bool))
        batch,trials = tune_batch(model,sample,device,maximum=2)
        self.assertIn(batch,(1,2))
        self.assertTrue(trials)
        for key,value in before.items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_cli_real_data_smoke_cache_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            files = ['FileName,EF,Split']
            traces = ['FileName,Frame,X1,Y1,X2,Y2']
            for i in range(12):
                name = f'case{i:02d}'
                files.append(f'{name},{40+i*2},{"TRAIN" if i<8 else "VAL"}')
                video = np.arange(17,dtype=np.uint8)[:,None,None]*np.ones((17,112,112),np.uint8)
                np.save(root/'npy'/f'{name}.npy',video)
                for frame in (2,12):
                    for y in (20,30,50,70,90):
                        traces.append(f'{name}.avi,{frame},30,{y},80,{y}')
            (root/'FileList.csv').write_text('\n'.join(files))
            (root/'VolumeTracings.csv').write_text('\n'.join(traces))
            checkpoints = []
            for mode in ('none','spatial'):
                cfg = config(mode,112)
                path = root/f'{mode}.pt'
                torch.save(dict(model_state_dict=TemporalMAE(**cfg).state_dict(),epoch=400,
                                config=dict(model=cfg,data=dict(input_protocol='gray_repeat3'))),path)
                checkpoints.append(f'{mode}={path}')
            args = SimpleNamespace(seg_frames=8,seg_target_index=4,offset_probe=True,
                                   input_protocol='gray_repeat3',seg_train_cases=2,seg_val_cases=2,seed=42)
            ds = SegViews(root,'val',args,3,2)
            self.assertEqual(len(ds),8)
            for index in range(len(ds)):
                item = ds[index]
                target = item['target_index']
                self.assertAlmostEqual(float(item['video'][target].mean()),item['source_frame']/255,places=6)
                self.assertIn(target,(4,5))
            cmd = [sys.executable,'tools/diagnose_stage2_interfaces.py','--checkpoints',*checkpoints,
                   '--data_root',str(root),'--output_root',str(root/'output'),'--run_tag','test',
                   '--ef_frames','16','--seg_frames','8','--seg_target_index','4','--smoke','--keep_cache']
            env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONUTF8='1')
            def run(extra=(),ok=True):
                result = subprocess.run(cmd+list(extra),cwd=ROOT,env=env,capture_output=True,text=True,encoding='utf-8')
                if ok:
                    self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                else:
                    self.assertNotEqual(result.returncode,0)
                return result
            run()
            out = root/'output/result/smoke_test'
            m = json.loads((out/'spatial/metrics.json').read_text())
            self.assertEqual(m['boundaries']['mae_optimizer_steps'],0)
            self.assertEqual(set(m['seg']),{'local','fused','local_mean','fused_mean','shared_bank','offset_bank'})
            self.assertEqual(m['seg']['offset_bank']['parameters'],m['seg']['shared_bank']['parameters'])
            self.assertTrue((out/'analysis.zip').is_file())
            self.assertTrue((out/'cross_checkpoint_differences.csv').is_file())
            control = json.loads((out/'none/metrics.json').read_text())
            self.assertEqual(control['ef']['local'],control['ef']['fused'])
            self.assertEqual(control['seg']['local'],control['seg']['fused'])
            self.assertFalse(list((root/'output').rglob('*.pt')))
            self.assertTrue(list((root/'output/cache').rglob('*.npz')))
            before = (out/'spatial/metrics.json').read_bytes()
            self.assertIn('Skip completed spatial',run().stdout)
            self.assertEqual(before,(out/'spatial/metrics.json').read_bytes())
            (out/'spatial/DONE').unlink()
            run()
            self.assertEqual(m['seg'],json.loads((out/'spatial/metrics.json').read_text())['seg'])
            self.assertIn('Protocol changed',run(['--ridge_alpha','20'],ok=False).stderr)


if __name__=='__main__':
    unittest.main()
