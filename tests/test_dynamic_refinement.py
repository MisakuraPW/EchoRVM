import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml

from models.frame_readout import FactorizedFrameExpansion
from models.temporal_mae import TemporalMAE, temporal_mask
from utils.dynamic_latent import (fit_motion_axis, motion_signal, detect_events, event_scores,
                                  pair_identity, trajectory_statistics)
from utils.dynamic_data import build_dynamic_manifest
from utils.streaming_features import StreamingFeatureCache
from tools.evaluate_dynamic_latent import evaluate
from tools.run_dynamic_refinement import configuration, queue_lock, run_endpoint


def tiny_config(readout, image_size=16):
    return dict(name='temporal_mae',img_size=image_size,patch_size=4 if image_size==16 else 8,
                local_frames=4 if image_size==16 else 16,clip_count=2 if image_size==16 else 4,
                tubelet_size=2,in_chans=3,embed_dim=24,depth=1,num_heads=3,
                decoder_embed_dim=24,decoder_depth=1,decoder_num_heads=3,mask_ratio=.5,
                memory_mode='spatial',memory_grid=2,core_depth=1,norm_pix_loss=False,
                frame_readout=readout,dynamic_rank=6,separate_qv_bias=True,
                dynamic_orthogonal_weight=.001 if readout=='factorized' else 0.)


class DynamicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_structure_low_rank_and_validity(self):
        layer=FactorizedFrameExpansion(24,2,6)
        x=torch.randn(2,4,5,24)
        valid=torch.ones(2,4,5,dtype=torch.bool)
        valid[:,:,0]=False
        parts=layer.components(x,valid)
        self.assertEqual(parts['features'].shape,(2,8,5,24))
        self.assertEqual(float(parts['features'][:,:,0].detach().abs().sum()),0.)
        for patch in range(1,5):
            d=parts['dynamic'][0,:,patch].detach()
            self.assertLessEqual(int(torch.linalg.matrix_rank(d,tol=1e-5)),6)
        torch.testing.assert_close(parts['features'].mean(1,keepdim=True),
                                   parts['reference']+parts['dynamic'].mean(1,keepdim=True),atol=1e-6,rtol=1e-5)
        changed=x.clone();changed[:,1:]+=3
        other=layer.components(changed,valid)
        torch.testing.assert_close(parts['coefficients'][:,:2],other['coefficients'][:,:2])
        self.assertLess(float(layer.orthogonal_loss().detach()),1e-10)

    def test_matched_shared_initialization(self):
        torch.manual_seed(42)
        anchor=TemporalMAE(**tiny_config('learned'))
        for kind in ('factorized','query'):
            torch.manual_seed(42)
            candidate=TemporalMAE(**tiny_config(kind))
            for key,value in anchor.state_dict().items():
                if not key.startswith('frame_expansion.'):
                    torch.testing.assert_close(value,candidate.state_dict()[key])
            for key,value in anchor.frame_expansion.state_dict().items():
                torch.testing.assert_close(value,candidate.frame_expansion.base.state_dict()[key])

    def test_completed_endpoints_are_reused_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp)/'endpoint';output.mkdir()
            (output/'DONE').write_text('complete')
            (output/'metrics.json').write_text('{}')
            times=[]
            with patch('tools.run_dynamic_refinement.run_command') as launch:
                run_endpoint(['python','evaluate.py'],'baseline/ef',Path(temp),times,output,('metrics.json',))
                launch.assert_not_called()
            with self.assertRaisesRegex(RuntimeError,'missing required outputs'):
                run_endpoint(['python','evaluate.py'],'broken',Path(temp),times,output,('predictions.csv',))

    def test_masked_pixels_never_leak_amp_gradients_roundtrip_and_stream(self):
        for readout in ('learned','factorized','query'):
            cfg=tiny_config(readout)
            model=TemporalMAE(**cfg).eval()
            video=torch.rand(2,8,3,16,16)
            mask=torch.stack([temporal_mask(2,model.token_grid,.5,'tube','cpu') for _ in range(2)],1)
            hidden=mask.reshape(2,4,4,4).repeat_interleave(2,1).repeat_interleave(4,2).repeat_interleave(4,3)[:,:,None]
            with torch.no_grad():
                torch.testing.assert_close(model(video,masks=mask)['pred'],
                    model(torch.where(hidden,video+5,video),masks=mask)['pred'])
                native=model.frame_features(model.forward_features(video))
                stream=StreamingFeatureCache(model,capacity=2)
                chunks=[stream.update(video[:,i:i+4],['a','b'],torch.arange(i,i+4)[None].expand(2,-1))['frame_features']
                        for i in (0,4)]
                torch.testing.assert_close(torch.cat(chunks,1),native)
                other=video.clone();other[:,4:]+=5
                torch.testing.assert_close(model.frame_features(model.forward_features(other))[:,:4],native[:,:4])
            model.train()
            with torch.autocast('cpu',dtype=torch.bfloat16):
                out=model(video,masks=mask)
            out['loss'].backward()
            self.assertEqual(out['pred'].shape,out['target'].shape)
            self.assertTrue(torch.isfinite(out['loss']))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.frame_expansion.parameters()))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            restored=TemporalMAE(**cfg)
            restored.load_state_dict(model.state_dict(),strict=True)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_fp16_training_gradients(self):
        for readout in ('factorized','query'):
            cfg=tiny_config(readout)
            cfg['memory_compression']='temporal_attention'
            model=TemporalMAE(**cfg).cuda().train()
            with torch.autocast('cuda',dtype=torch.float16):
                loss=model(torch.rand(2,8,3,16,16,device='cuda'))['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            self.assertGreater(float(model.compression_score[-1].weight.grad.abs().sum()),0.)
            if readout=='factorized':
                self.assertGreater(float(model.frame_expansion.basis.grad.abs().sum()),0.)

    def test_event_detector_no_oracle_swap_misses_and_physical_time(self):
        t=np.arange(200)/50
        x=np.c_[np.cos(2*np.pi*t),np.sin(2*np.pi*t)]
        train=[dict(features=x,seconds=t,ed_seconds=1.,es_seconds=1.5)]
        axis=fit_motion_axis(train)
        _,s=motion_signal(x,t,axis)
        events=detect_events(s,t)
        original=copy.deepcopy(events)
        score=event_scores(events,t,1.,1.5,50)
        event_scores(events,t,1.5,1.,50)
        self.assertEqual(events,original)
        self.assertTrue(score['ed_detected'])
        self.assertTrue(score['es_detected'])
        missed=event_scores(detect_events(np.zeros(len(t)),t),t,1.,1.5,50)
        self.assertIsNone(missed['ed_error_ms'])
        self.assertFalse(missed['ed_within100ms'])
        stats=trajectory_statistics(x,t)
        self.assertAlmostEqual(stats['median_interval_ms'],20)
        self.assertGreater(stats['explained_variance_2'],.99)
        self.assertTrue(trajectory_statistics(np.ones_like(x),t)['collapsed'])

    def test_identity_chance_credit_content_swap_and_near_identical_exclusion(self):
        y=np.array([[1.,0.],[0.,1.],[2.,0.],[0.,2.]])
        self.assertEqual(pair_identity(y,y)['pair_accuracy'],1.)
        self.assertEqual(pair_identity(y.reshape(2,2,2)[:,::-1].reshape(4,2),y)['pair_accuracy'],0.)
        same=np.repeat(y.reshape(2,2,2).mean(1),2,axis=0)
        self.assertEqual(pair_identity(same,y)['pair_accuracy'],.5)
        self.assertIsNone(pair_identity(np.ones_like(y),np.ones_like(y))['pair_accuracy'])

    def test_config_preserves_budget_and_no_duplicate_queue(self):
        args=SimpleNamespace(seed=42,data_root='data',num_workers=8,init_checkpoint='init.pt',
            epochs=100,smoke=False,dynamic_rank=16,orthogonal_weight=.001,save_last_every=10,min_free_gb=3.)
        for name in ('combined','factorized','query'):
            cfg=configuration(args,name,Path('/owned')/name)
            self.assertEqual(cfg['train']['batch_size']*cfg['train']['grad_accum_steps'],32)
            self.assertEqual(cfg['checkpoint']['save_epochs'],[100])
            self.assertFalse(cfg['checkpoint']['save_initial'])
            self.assertFalse(cfg['early_stopping']['enabled'])
        with tempfile.TemporaryDirectory() as temp:
            with queue_lock(Path(temp)/'lock'):
                with self.assertRaisesRegex(RuntimeError,'running queue'):
                    with queue_lock(Path(temp)/'lock'):
                        pass

    def test_real_interface_frozen_pipeline_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';(data/'npy').mkdir(parents=True)
            files,traces=[],[]
            yy,xx=np.mgrid[:112,:112]
            for i,split in enumerate(('TRAIN','TRAIN','VAL','VAL')):
                name=f'case{i}'
                radius=20.5+4.5*np.cos(2*np.pi*(np.arange(208)-20)/160)
                video=((xx[None]-56)**2+(yy[None]-56)**2 < radius[:,None,None]**2).astype('uint8')*180
                np.save(data/'npy'/f'{name}.npy',video)
                files.append(dict(FileName=name,Split=split,EF=50+i,FPS=50,NumberOfFrames=208))
                for frame,r in ((20,25),(100,16)):
                    for y in np.linspace(56-r,56+r,12):
                        half=np.sqrt(max(0,r*r-(y-56)**2))
                        traces.append(dict(FileName=name+'.avi',Frame=frame,X1=56-half,Y1=y,X2=56+half,Y2=y))
            pd.DataFrame(files).to_csv(data/'FileList.csv',index=False)
            pd.DataFrame(traces).to_csv(data/'VolumeTracings.csv',index=False)
            manifest=build_dynamic_manifest(data,192,2,2,42)
            self.assertFalse({r['patient'] for r in manifest['splits']['train']} & {r['patient'] for r in manifest['splits']['val']})
            mf=root/'manifest.json';mf.write_text(json.dumps(manifest))
            cfg=tiny_config('factorized',112)
            model=TemporalMAE(**cfg)
            ckpt=root/'model.pt'
            torch.save(dict(model=model.state_dict(),epoch=100,config=dict(model=cfg,data=dict(input_protocol='gray_repeat3'))),ckpt)
            args=SimpleNamespace(seed=42,cpu_threads=2,output_dir=str(root/'result'),cache_dir=str(root/'cache'),
                checkpoint=str(ckpt),manifest=str(mf),batch_size=0,max_batch_size=1,num_workers=0,prefetch_factor=2,
                auto_workers=False,recent_frames=64,nuisance_cases=1,plot_cases=1,ridge_alpha=10.,keep_cache=False)
            evaluate(args)
            self.assertTrue((root/'result/DONE').exists())
            self.assertTrue((root/'result/trajectory_00.png').exists())
            metrics=json.loads((root/'result/metrics.json').read_text())
            self.assertTrue(any(r['branch']=='dynamic' for r in metrics['summary']))
            self.assertFalse(list((root/'cache').rglob('*.npz')))
            evaluate(args)
            setup=SimpleNamespace(seed=42,data_root=str(data),num_workers=0,init_checkpoint=str(ckpt),
                epochs=1,smoke=True,dynamic_rank=6,orthogonal_weight=.001,save_last_every=10,min_free_gb=0.)
            training=configuration(setup,'factorized',root/'weights')
            training['model'].update(cfg)
            training['train'].update(batch_size=1,grad_accum_steps=1,cpu_threads=2)
            requested=root/'train.yaml';requested.write_text(yaml.safe_dump(training))
            project=Path(__file__).resolve().parents[1]
            def cli(script, flags):
                proc=subprocess.run([sys.executable,str(project/script),*map(str,flags)],cwd=project,
                                    capture_output=True,text=True,encoding='utf-8',errors='replace',env=os.environ.copy())
                self.assertEqual(proc.returncode,0,proc.stdout[-2500:]+proc.stderr[-2500:])
            flags=['--config',requested,'--output_dir',root/'train']
            cli('trainers/train_rmae.py',flags)
            last=root/'weights/last.pt';final=root/'weights/epoch_0001.pt'
            state=torch.load(last,map_location='cpu',weights_only=False)
            self.assertEqual(state['global_step'],2)
            self.assertTrue(final.exists())
            cli('trainers/train_rmae.py',flags+['--resume',last])
            cli('tools/evaluate_stage3.py',['--checkpoint',final,'--output_dir',root/'ef',
                '--cache_dir',root/'ef_cache','--data_root',data,'--no-with_seg','--eligible_budget',
                '--ef_train_cases','2','--ef_val_cases','2','--ef_steps','2',
                '--batch_size','1','--num_workers','0','--cpu_threads','2','--no-auto_workers'])
            cli('tools/audit_stage3_streaming.py',['--checkpoint',final,'--output_dir',root/'positions',
                '--cache_dir',root/'position_cache','--data_root',data,'--train_all_positions',
                '--seg_train_cases','2','--seg_val_cases','2','--seg_steps','2','--stream_cases','1',
                '--stream_frames','80','--batch_size','1','--num_workers','0',
                '--cpu_threads','2','--no-auto_workers'])
            self.assertTrue((root/'ef/DONE').exists())
            self.assertTrue((root/'positions/DONE').exists())


if __name__=='__main__':
    unittest.main()
