import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from models.temporal_mae import TemporalMAE
from utils.stage3_diagnostics import intervention_video, stream_audit, gradient_audit

ROOT=Path(__file__).resolve().parents[1]


def config(mode='spatial',size=16):
    return dict(name='temporal_mae',img_size=size,patch_size=4 if size==16 else 8,
        local_frames=4,clip_count=2,tubelet_size=2,in_chans=3,embed_dim=24,depth=1,
        num_heads=3,decoder_embed_dim=24,decoder_depth=1,decoder_num_heads=3,
        mask_ratio=.5,memory_mode=mode,memory_grid=2,core_depth=1,norm_pix_loss=False,
        separate_qv_bias=True,position_embedding='flat_sinusoid')


class Stage3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prefix_conditions_leave_recent_window_unchanged(self):
        video=torch.arange(24.).reshape(1,24,1,1,1).expand(-1,-1,-1,16,16).clone()
        for p in (0,8,16):
            value=intervention_video(video,p,8,'clean',4)
            torch.testing.assert_close(value[:,-8:],video[:,-8:])
        for kind in ('repeat_prefix','zero_prefix_clip'):
            value=intervention_video(video,16,8,kind,4)
            torch.testing.assert_close(value[:,-8:],video[:,-8:])
        with self.assertRaises(ValueError):
            intervention_video(video,32,8,'clean',4)
        torch.testing.assert_close(video[:,0],torch.zeros_like(video[:,0]))

    def test_stream_matches_native_all_four_modes_and_local_is_history_independent(self):
        for mode in ('none','global','spatial','dual'):
            model=TemporalMAE(**config(mode)).eval()
            video=torch.rand(2,16,3,16,16)
            with torch.no_grad():
                native=model.diagnostic_features(video[:,-8:])
            short,_=stream_audit(model,video[:,-8:],8)
            long,trace=stream_audit(model,video,8,observe=True)
            torch.testing.assert_close(short['cache'],model.frame_features(native['features']).mean(2))
            torch.testing.assert_close(short['local'],long['local'])
            if mode=='none':
                torch.testing.assert_close(short['cache'],long['cache'])
            else:
                self.assertEqual(len(trace),4)
                self.assertIn('update_mean',trace[0])
                self.assertEqual(len(model.memory._forward_pre_hooks),0)
                slots=1 if mode=='global' else 4
                self.assertEqual(long['local_history'].shape,(2,8+slots,24))
                torch.testing.assert_close(long['local_history'][:,:8],long['local_empty'][:,:8])
            again,_=stream_audit(model,video[:,-8:],8)
            torch.testing.assert_close(short['cache'],again['cache'])

    def test_gradient_probe_preserves_parameters_flags_rng_and_detects_no_memory_path(self):
        for mode in ('none','spatial'):
            model=TemporalMAE(**config(mode)).eval()
            video=torch.rand(1,8,3,16,16)
            before={k:v.clone() for k,v in model.state_dict().items()}
            rng=torch.get_rng_state().clone()
            result=gradient_audit(model,video)
            torch.testing.assert_close(torch.get_rng_state(),rng)
            self.assertTrue(all(p.requires_grad for p in model.parameters()))
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,before[key])
            self.assertEqual(len(result['input_gradient_rms']),2)
            if mode=='none':
                self.assertEqual(result['input_gradient_rms'][0],0)
            else:
                self.assertGreater(result['input_gradient_rms'][0],0)

    def test_file_backed_smoke_resume_cache_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'npy').mkdir()
            files=['FileName,EF,Split']
            traces=['FileName,Frame,X1,Y1,X2,Y2']
            rng=np.random.default_rng(8)
            for i in range(8):
                name=f'case{i}'
                files.append(f'{name},{40+3*i},{"TRAIN" if i<4 else "VAL"}')
                n=12 if i in (0,4) else 32
                np.save(root/'npy'/f'{name}.npy',rng.integers(0,256,(n,112,112),dtype=np.uint8))
                for frame in (2,10):
                    for y in (30,40,50,60,70):
                        traces.append(f'{name}.avi,{frame},35,{y},75,{y}')
            (root/'FileList.csv').write_text('\n'.join(files)+'\n')
            (root/'VolumeTracings.csv').write_text('\n'.join(traces)+'\n')
            cfg=config(size=112)
            model=TemporalMAE(**cfg)
            path=root/'model.pt'
            torch.save(dict(model=model.state_dict(),epoch=400,config=dict(model=cfg,data=dict(input_protocol='gray_repeat3'))),path)
            out=root/'result'
            cmd=[sys.executable,str(ROOT/'tools/evaluate_stage3.py'),'--checkpoint',str(path),
                 '--expected_epoch','400','--expected_memory','spatial','--data_root',str(root),
                 '--output_dir',str(out),'--cache_dir',str(root/'cache'),
                 '--prefixes','0','8','16','--recent_frames','8','--smoke','--keep_cache']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTORCH_NVML_BASED_CUDA_CHECK='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            done=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=240)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            result=json.loads((out/'metrics.json').read_text())
            self.assertEqual(result['train_patients'],3)
            self.assertEqual(result['val_patients'],3)
            self.assertEqual(len(result['ef']),11)
            with (out/'recovery_patient.csv').open(encoding='utf-8') as handle:
                recovery=list(csv.DictReader(handle))
            self.assertTrue(recovery)
            self.assertTrue(all(float(r['state_distance_rms'])==0 for r in recovery
                                if r['condition']=='zero_prefix_clip' and int(r['clip'])<3))
            self.assertIn('dice_patient_mean',result['seg'])
            with np.load(root/'cache'/'val.npz') as f:
                self.assertEqual(len(json.loads(str(f['excluded'].item()))),1)
                self.assertEqual(f['history_0_cache'].shape,f['history_16_cache'].shape)
            before=(out/'metrics.json').read_bytes()
            done=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            self.assertEqual(before,(out/'metrics.json').read_bytes())
            # Interrupted finalization uses validated cached features; stale scientific settings fail.
            (out/'DONE').unlink()
            done=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            self.assertTrue((out/'DONE').is_file())
            done=subprocess.run(cmd+['--seed','43'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
            self.assertNotEqual(done.returncode,0)
            self.assertIn('Protocol/data/code changed',done.stderr)
            # The previously unsupported no-memory segmented control must really run.
            cfg=config(mode='none',size=112)
            path_none=root/'none.pt'
            torch.save(dict(model=TemporalMAE(**cfg).state_dict(),epoch=400,
                            config=dict(model=cfg,data=dict(input_protocol='gray_repeat3'))),path_none)
            cmd_none=cmd.copy()
            for flag,value in (('--checkpoint',str(path_none)),('--expected_memory','none'),
                               ('--output_dir',str(root/'none_result')),('--cache_dir',str(root/'none_cache'))):
                cmd_none[cmd_none.index(flag)+1]=value
            done=subprocess.run(cmd_none+['--no-with_seg'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            result=json.loads((root/'none_result'/'metrics.json').read_text())
            self.assertEqual(result['ef']['history_0_cache']['mae'],result['ef']['history_16_cache']['mae'])
            self.assertEqual(result['ef']['history_16_cache']['mae'],result['ef']['repeat_prefix_cache']['mae'])
            eligible=cmd_none.copy()
            eligible.remove('--smoke')
            for flag,value in (('--output_dir',str(root/'eligible_result')),('--cache_dir',str(root/'eligible_cache'))):
                eligible[eligible.index(flag)+1]=value
            done=subprocess.run(eligible+['--no-with_seg','--eligible_budget','--ef_train_cases','3',
                '--ef_val_cases','3','--ef_steps','2','--batch_size','2','--num_workers','0','--no-auto_workers'],
                cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            value=json.loads((root/'eligible_result'/'metrics.json').read_text())
            self.assertEqual(value['train_patients'],3)
            self.assertEqual(value['val_patients'],3)

    def test_queue_dry_run_selection_and_missing_checkpoint_fail_closed(self):
        cmd=[sys.executable,str(ROOT/'tools/run_stage3_memory.py'),'--only','hier_spatial','--dry_run']
        result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,timeout=90)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('hier_spatial: reuse',result.stdout)
        self.assertNotIn('hier_global: reuse',result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            result=subprocess.run(cmd[:-1]+['--checkpoint_root',tmp,'--output_root',str(Path(tmp)/'output')],
                                  cwd=ROOT,capture_output=True,text=True,timeout=90)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('never substitute random weights',result.stderr)
            self.assertFalse((Path(tmp)/'output').exists())


if __name__=='__main__':
    unittest.main()
