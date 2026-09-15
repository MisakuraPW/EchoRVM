import contextlib
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.frequency import haar2, patch_bands, band_descriptor, multiband_error
from models.temporal_mae import TemporalMAE, temporal_mask
from models.video_mae import EchoVideoMAE
from utils.autotune import batch_candidates, apply_selection
from utils.checkpoint import save_checkpoint, load_checkpoint
from tools import run_temporal_research as runner


def tiny(**changes):
    cfg = dict(img_size=16, patch_size=4, local_frames=4, clip_count=2, frames=8,
               tubelet_size=2, in_chans=3, embed_dim=24, depth=1, num_heads=3,
               decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=3,
               memory_mode='spatial', memory_grid=2, core_depth=1, norm_pix_loss=False,
               frequency_conditioned=True, frequency_loss_weight=.1, separate_qv_bias=True)
    cfg.update(changes)
    return cfg


class FrequencyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_haar_energy_and_inverse(self):
        x = torch.randn(3, 8, 8)
        ll, lh, hl, hh = haar2(x)
        restored = torch.empty_like(x)
        restored[...,0::2,0::2] = (ll+lh+hl+hh)/2
        restored[...,0::2,1::2] = (ll+lh-hl-hh)/2
        restored[...,1::2,0::2] = (ll-lh+hl-hh)/2
        restored[...,1::2,1::2] = (ll-lh-hl+hh)/2
        torch.testing.assert_close(restored, x)
        torch.testing.assert_close(x.square().sum(), sum(b.square().sum() for b in (ll,lh,hl,hh)))
        flat = torch.randn(5, 2*8*8*3)
        bands = patch_bands(flat,2,8,3)
        torch.testing.assert_close(flat.square().sum(), sum(b.square().sum() for b in bands))
        q = band_descriptor(torch.ones_like(flat),2,8,3)
        torch.testing.assert_close(q[:,0], torch.ones(5))
        torch.testing.assert_close(q[:,1:], torch.zeros(5,6))
        pred = flat.clone().requires_grad_()
        loss = multiband_error(pred,torch.zeros_like(flat),2,8,3).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertGreater(float(pred.grad.abs().sum()),0)

    def test_visible_only_and_future_leakage(self):
        torch.manual_seed(42)
        model = TemporalMAE(**tiny()).eval()
        # Activate conditioning so a hidden-data leak cannot hide behind zero initialization.
        with torch.no_grad():
            model.frequency_gates.read_band.weight.normal_(0,.2)
            model.frequency_gates.write_band.weight.normal_(0,.2)
        x = torch.rand(2,8,3,16,16)
        masks = torch.stack([temporal_mask(2,model.token_grid,.5,'tube',x.device) for _ in range(2)],1)
        pixel_mask = masks.reshape(2,4,4,4).repeat_interleave(2,1).repeat_interleave(4,2).repeat_interleave(4,3)[:,:,None]
        hidden_changed = torch.where(pixel_mask, x+torch.rand_like(x)*20, x)
        with torch.no_grad():
            base = model(x,masks=masks)['pred']
            torch.testing.assert_close(base, model(hidden_changed,masks=masks)['pred'])
            future = x.clone()
            future[:,4:] += 10
            torch.testing.assert_close(base[:,:32],model(future,masks=masks)['pred'][:,:32])

    def test_amp_gradients_resume_and_padding(self):
        torch.manual_seed(42)
        model = TemporalMAE(**tiny())
        optimizer = torch.optim.AdamW(model.parameters(),lr=.001)
        x = torch.rand(2,8,3,16,16)
        valid = torch.ones(2,8,dtype=torch.bool)
        valid[0,4:] = False
        with torch.autocast('cpu',dtype=torch.bfloat16):
            out = model(x,valid)
        out['loss'].backward()
        self.assertTrue(torch.isfinite(out['loss']))
        self.assertGreater(float(model.frequency_gates.write_band.weight.grad.abs().sum()),0)
        torch.testing.assert_close(out['loss'].detach(),out['loss_recon']+out['loss_frequency_weighted'])
        optimizer.step()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'last.pt'
            save_checkpoint(path,model,optimizer,None,None,1,1,.5,dict(model=tiny()))
            restored = TemporalMAE(**tiny())
            opt = torch.optim.AdamW(restored.parameters(),lr=.001)
            load_checkpoint(path,restored,opt)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key])

    def test_gate_gradient_on_later_clip_and_legacy_checkpoint(self):
        model = TemporalMAE(**tiny())
        model(torch.rand(2,8,3,16,16))['loss'].backward()
        self.assertGreater(float(model.frequency_gates.read_band.weight.grad.abs().sum()),0)
        self.assertGreater(float(model.memory.update_x.weight.grad.abs().sum()),0)
        legacy = TemporalMAE(**tiny(frequency_conditioned=False,frequency_loss_weight=0))
        self.assertFalse(any(k.startswith('frequency_gates') for k in legacy.state_dict()))
        legacy.load_state_dict(legacy.state_dict(),strict=True)

    def test_two_run_configuration_and_autotune_budget(self):
        matrix = runner.experiment_matrix('tsf')
        self.assertEqual([x[0] for x in matrix],['tsf_spatial_control','tsf_frequency_memory'])
        cfg = dict(model=dict(name='temporal_mae',frequency_conditioned=True),
                   train=dict(batch_size=8,grad_accum_steps=4),data=dict(sampling_protocol='temporal_v1'))
        self.assertIn(32,batch_candidates(cfg))
        selected = apply_selection(cfg,dict(batch_size=32,gradient_checkpointing=False,num_workers=8))
        self.assertEqual(selected['train']['grad_accum_steps'],1)
        self.assertTrue(selected['model']['frequency_conditioned'])

    def test_two_run_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            files, traces = ['FileName,EF,Split'], ['FileName,Frame,X1,Y1,X2,Y2']
            for i in range(12):
                case = f'case{i:02d}'
                files.append(f'{case},{40+2*i},{"TRAIN" if i<8 else "VAL"}')
                np.save(root/'npy'/f'{case}.npy',np.random.default_rng(i).integers(0,255,(64,112,112),dtype=np.uint8))
                for frame in (24,48):
                    for y in (20,30,50,70,90):
                        traces.append(f'{case}.avi,{frame},30,{y},80,{y}')
            (root/'FileList.csv').write_text('\n'.join(files))
            (root/'VolumeTracings.csv').write_text('\n'.join(traces))
            init = root/'init.pt'
            dimensions = dict(img_size=112,patch_size=8,local_frames=16,clip_count=4,frames=64,
                              embed_dim=24,depth=1,num_heads=3,decoder_embed_dim=24,
                              decoder_depth=1,decoder_num_heads=3,memory_grid=2)
            torch.save({'model': EchoVideoMAE(**tiny(**dict(dimensions,frames=16))).state_dict()},init)
            original = runner.make_config

            def small_config(*args):
                cfg = original(*args)
                cfg['model'].update(dimensions)
                cfg['model']['gradient_checkpointing'] = False
                cfg['model']['require_init_min_tensors'] = 1
                cfg['train']['mixed_precision'] = False
                cfg['checkpoint']['min_free_gb'] = 0
                cfg['logging'].update(use_tqdm=False,use_tensorboard=False)
                return cfg

            env = dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            commands = []

            def execute(command,label,output,timings):
                commands.append(label)
                result = subprocess.run(command,cwd=runner.ROOT,env=env,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stdout+result.stderr)

            argv = ['run_temporal_research.py','--suite','tsf','--run_tag','smoke_tsf','--smoke',
                    '--audit_profile','quick','--screen_seg_final','--autotune','--num_workers','0',
                    '--init_checkpoint',str(init),'--data_root',str(root),'--output_root',str(root/'out')]
            with patch.object(sys,'argv',argv), patch.object(runner,'make_config',side_effect=small_config), \
                 patch.object(runner,'run_command',side_effect=execute), contextlib.redirect_stdout(io.StringIO()):
                runner.main()
                runner.main()  # Completed stages are not trained or evaluated again.
            self.assertEqual(len(commands),8)  # 2 x (train + EF0 + EF1 + final Dice).
            result_root = root/'out/smoke_tsf/result'
            weights_root = root/'out/smoke_tsf/ckpt'
            self.assertTrue((result_root/'quick_paired_seg.csv').exists())
            self.assertTrue((result_root/'quick_paired_comparisons.csv').exists())
            for name, _, _ in runner.experiment_matrix('tsf'):
                self.assertTrue((weights_root/name/'last.pt').exists())
                self.assertTrue((weights_root/name/'epoch_0000.pt').exists())
                self.assertTrue((weights_root/name/'epoch_0001.pt').exists())
                self.assertFalse(list((result_root/name).rglob('*.pt')))
            with (result_root/'tsf_frequency_memory/logs/metrics.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(float(rows[-1]['train_loss_frequency']),0)


if __name__ == '__main__':
    unittest.main()
