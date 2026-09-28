import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from models.temporal_mae import TemporalMAE, temporal_mask
from models.frame_readout import FrameExpansion
from models.ef_readout import TemporalEFReadout, Stage2EFFineTuner
from models.downstream import EchoVideoMAEBackbone, load_pretrained_rmae
from utils.streaming_features import StreamingFeatureCache, ef_sequences
from utils.pretrained_init import load_videomae_init
from tools.evaluate_stage2 import ef_batch, seg_batch
from tools.run_stage2_questions import stage_config

ROOT=Path(__file__).resolve().parents[1]


def config(mode='spatial', readout='repeat', tubelet=2, size=16):
    return dict(name='temporal_mae',img_size=size,patch_size=4 if size==16 else 8,
        frames=8,local_frames=4,clip_count=2,tubelet_size=tubelet,in_chans=3,
        embed_dim=24,depth=1,num_heads=3,decoder_embed_dim=24,decoder_depth=1,
        decoder_num_heads=3,mask_ratio=.5,memory_mode=mode,memory_grid=2,
        core_depth=1,frame_readout=readout,norm_pix_loss=False,
        separate_qv_bias=True,position_embedding='flat_sinusoid')


class Stage2QuestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stream_equals_native_all_modes_and_patient_reset(self):
        for mode in ('none','global','spatial','dual'):
            model=TemporalMAE(**config(mode)).eval()
            video=torch.rand(2,8,3,16,16)
            with torch.no_grad():
                native=model.diagnostic_features(video)
                stream=StreamingFeatureCache(model,capacity=2)
                outputs=[]
                for start in (0,4):
                    out=stream.update(video[:,start:start+4],['a','b'],
                                      torch.arange(start,start+4)[None].expand(2,-1))
                    outputs.append(out['frame_features'])
                torch.testing.assert_close(torch.cat(outputs,1),model.frame_features(native['features']))
                if mode!='none':
                    torch.testing.assert_close(stream.state.mean(1),native['states'][:,-1])
                state=stream.state
                stream.read()
                self.assertIs(stream.state,state)
                with self.assertRaisesRegex(ValueError,'Patient'):
                    stream.update(video[:,:4],['b','a'],torch.arange(8,12)[None].expand(2,-1))
                with self.assertRaisesRegex(ValueError,'Duplicated'):
                    stream.update(video[:,:4],['a','b'],torch.arange(4)[None].expand(2,-1))
                stream.reset()
                first=stream.update(video[:,:4],['c','d'],torch.arange(4)[None].expand(2,-1))
                torch.testing.assert_close(first['frame_features'],outputs[0])

    def test_fifo_boundary_is_older_not_final_state(self):
        model=TemporalMAE(**config()).eval()
        stream=StreamingFeatureCache(model,capacity=2)
        video=torch.rand(1,16,3,16,16)
        snapshots=[]
        for start in range(0,16,4):
            out=stream.update(video[:,start:start+4],['a'],torch.arange(start,start+4)[None])
            snapshots.append(out['final_state'])
        self.assertEqual(len(stream.entries),2)
        torch.testing.assert_close(out['boundary'],snapshots[1])
        self.assertEqual(out['source_indices'].tolist(),[list(range(8,16))])
        self.assertEqual(out['older_clips'],2)
        seq=ef_sequences(out)
        self.assertEqual(seq['cache_history'].shape,seq['cache_empty'].shape)
        torch.testing.assert_close(seq['cache_history'][:,:8],seq['cache_empty'][:,:8])
        self.assertEqual(float(seq['cache_empty'][:,8:].abs().sum()),0.)
        self.assertTrue(all(not v.requires_grad for v in seq.values()))
        stream.reset()
        out=stream.update(video[:,:4],['a'],torch.arange(4)[None])
        with self.assertRaisesRegex(ValueError,'real prefix'):
            ef_sequences(out)

    def test_expansion_order_and_gradients_no_hidden_leak(self):
        layer=FrameExpansion(2)
        with torch.no_grad():
            layer.projections[0].bias.fill_(10)
            layer.projections[1].bias.fill_(20)
        x=torch.arange(12.).reshape(1,3,2,2)
        y=layer(x)
        torch.testing.assert_close(y[:,0::2],x+10)
        torch.testing.assert_close(y[:,1::2],x+20)
        for tubelet,readout in ((2,'repeat'),(2,'learned'),(1,'repeat')):
            model=TemporalMAE(**config(readout=readout,tubelet=tubelet)).eval()
            video=torch.rand(2,8,3,16,16)
            masks=torch.stack([temporal_mask(2,model.token_grid,.5,'tube',video.device) for _ in range(2)],1)
            hidden=masks.reshape(2,8//tubelet,4,4).repeat_interleave(tubelet,1).repeat_interleave(4,2).repeat_interleave(4,3)[:,:,None]
            with torch.no_grad():
                pred=model(video,masks=masks)['pred']
                torch.testing.assert_close(pred,model(torch.where(hidden,video+10,video),masks=masks)['pred'])
            with torch.autocast('cpu',dtype=torch.bfloat16):
                out=model(video,masks=masks)
            self.assertEqual(out['pred'].shape,out['target'].shape)
            out['loss'].backward()
            self.assertTrue(torch.isfinite(out['loss']))
            if readout=='learned':
                self.assertTrue(all(p.grad is not None and p.grad.abs().sum()>0 for p in model.frame_expansion.parameters()))
                with torch.no_grad():
                    features=EchoVideoMAEBackbone(model).forward_tokens(video)['outputs']
                self.assertEqual(features.shape,(2,8,16,24))
                self.assertGreater(float((features[:,0]-features[:,1]).abs().max()),0.)

    def test_ef_recent_reset_and_context_masks(self):
        model=TemporalMAE(**config()).eval()
        x=torch.rand(2,16,3,16,16)
        idx=torch.arange(16)[None].expand(2,-1)
        seq=ef_batch(model,x,idx,8)
        changed=x.clone()
        changed[:,:8]+=3
        other=ef_batch(model,changed,idx,8)
        torch.testing.assert_close(seq['cache'],other['cache'])
        self.assertGreater(float((seq['cache_history']-other['cache_history']).abs().sum()),0.)
        head=TemporalEFReadout(24,16)
        loss=head(seq['cache_history'].clone(),4).square().mean()
        loss.backward()
        self.assertIsNotNone(head.project.weight.grad)
        learned=TemporalMAE(**config(readout='learned')).eval()
        valid=torch.ones(2,8,dtype=torch.bool)
        valid[1,-1]=False
        maps,_=seg_batch(learned,x[:,:8],valid,torch.tensor([6,6]),'real')
        self.assertEqual(float(maps['fused'][1].abs().sum()),0.)

    def test_checkpoint_roundtrip_and_tubelet1_init(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'model.pt'
            cfg=config(readout='learned')
            model=TemporalMAE(**cfg)
            torch.save(dict(model=model.state_dict(),config=dict(model=cfg)),path)
            loaded,_,report=load_pretrained_rmae(path)
            self.assertFalse(report['missing'] or report['unexpected'])
            self.assertEqual(loaded.frame_readout,'learned')
            src=TemporalMAE(**config())
            torch.save(dict(model=src.state_dict()),path)
            cfg=config(tubelet=1)
            cfg['patch_init_temporal_sum']=True
            dst=TemporalMAE(**cfg)
            report=load_videomae_init(dst,path)
            torch.testing.assert_close(dst.patch_embed.proj.weight,src.patch_embed.proj.weight.sum(2,keepdim=True))
            self.assertFalse(report['missing_encoder_keys'])

    def test_full_finetune_has_backbone_gradients(self):
        for mode in ('last','cache','cache_empty','cache_history','local_empty','local_history','state'):
            backbone=TemporalMAE(**config(readout='learned'))
            head=Stage2EFFineTuner(backbone,mode,8,8,16)
            prediction=head(torch.rand(2,16,3,16,16))
            prediction.square().mean().backward()
            self.assertEqual(prediction.shape,(2,))
            self.assertGreater(float(backbone.patch_embed.proj.weight.grad.abs().sum()),0.)
            if mode not in ('local_empty',):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in backbone.memory.parameters()))

    def test_presets_budget_and_structure(self):
        args=SimpleNamespace(seed=42,data_root='data',num_workers=8,init_checkpoint='init.pth',
                             epochs=100,smoke=False,local_frames=16)
        for name in ('repeat','learned','tubelet1','joint'):
            cfg=stage_config(args,name,Path('ckpt')/name)
            self.assertEqual(cfg['model']['frames'],64)
            self.assertEqual(cfg['train']['batch_size']*cfg['train']['grad_accum_steps'],32)
            self.assertEqual(cfg['checkpoint']['save_epochs'],[100])
            self.assertFalse(cfg['early_stopping']['enabled'])
            self.assertEqual(cfg['model']['sampling_rate'],1)
            self.assertFalse(cfg['augment']['enabled'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_amp_training_and_streaming(self):
        for tubelet,readout in ((2,'repeat'),(2,'learned'),(1,'repeat')):
            model=TemporalMAE(**config(readout=readout,tubelet=tubelet)).cuda()
            video=torch.rand(2,8,3,16,16,device='cuda')
            opt=torch.optim.AdamW(model.parameters(),lr=1e-4)
            scaler=torch.amp.GradScaler('cuda',init_scale=128.)
            with torch.autocast('cuda'):
                loss=model(video)['loss']
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            scaler.step(opt)
            scaler.update()
            model.eval()
            with torch.autocast('cuda'):
                cache=StreamingFeatureCache(model,2)
                for start in (0,4):
                    result=cache.update(video[:,start:start+4],['a','b'],torch.arange(start,start+4,device='cuda')[None].expand(2,-1))
            self.assertTrue(torch.isfinite(result['frame_features']).all())

    def test_end_to_end_file_backed_smoke_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'npy').mkdir()
            files=['FileName,EF,Split']
            traces=['FileName,Frame,X1,Y1,X2,Y2']
            rng=np.random.default_rng(8)
            for i in range(8):
                name=f'case{i}'
                files.append(f'{name},{40+3*i},{"TRAIN" if i<4 else "VAL"}')
                n=12 if i in (0,4) else 20
                np.save(root/'npy'/f'{name}.npy',rng.integers(0,256,(n,112,112),dtype=np.uint8))
                for frame in (2,10):
                    for y in (30,40,50,60,70):
                        traces.append(f'{name}.avi,{frame},35,{y},75,{y}')
            (root/'FileList.csv').write_text('\n'.join(files)+'\n')
            (root/'VolumeTracings.csv').write_text('\n'.join(traces)+'\n')
            cfg=config(readout='learned',size=112)
            model=TemporalMAE(**cfg)
            path=root/'model.pt'
            torch.save(dict(model=model.state_dict(),config=dict(model=cfg,data=dict(input_protocol='gray_repeat3'))),path)
            out=root/'result'
            cmd=[sys.executable,str(ROOT/'tools/evaluate_stage2.py'),'--checkpoint',str(path),
                '--data_root',str(root),'--output_dir',str(out),'--cache_dir',str(root/'cache'),
                '--prefix_frames','8','--recent_frames','8','--smoke','--keep_cache']
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTORCH_NVML_BASED_CUDA_CHECK='0',
                     OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            done=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            self.assertTrue((out/'DONE').exists())
            result=json.loads((out/'metrics.json').read_text())
            self.assertEqual(len(result['ef']),6)
            self.assertEqual(len(result['seg']),4)
            self.assertTrue((out/'real'/'overlay_fused_000.png').exists())
            with np.load(root/'cache'/'ef_train_real.npz') as cache:
                rows=json.loads(str(cache['rows'].item()))
                excluded=json.loads(str(cache['excluded'].item()))
                self.assertEqual(len(rows),3)
                self.assertEqual(len(excluded),1)
            before=(out/'metrics.json').read_bytes()
            done=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=90)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            self.assertEqual(before,(out/'metrics.json').read_bytes())
            # Exercise the production fine-tuning loader/trainer, including
            # real-history filtering and online augmentation, not only the head.
            ft=yaml.safe_load((ROOT/'configs/finetune/stage2_ef.yaml').read_text())
            ft['data'].update(data_root=str(root),num_workers=0)
            ft['model'].update(backbone_checkpoint=str(path),frames=16,prefix_frames=8,recent_frames=8,
                               head_hidden_dim=16,stage2_ef_readout='cache_history')
            ft['train'].update(epochs=1,max_steps=1,batch_size=2,grad_accum_steps=1,mixed_precision=False)
            ft['checkpoint'].update(dir=str(root/'ft_ckpt'))
            ft['logging']['use_tensorboard']=False
            ft_path=root/'ft.yaml'
            ft_path.write_text(yaml.safe_dump(ft))
            cmd=[sys.executable,str(ROOT/'trainers/train_finetune.py'),'--task','echonet_ef',
                 '--config',str(ft_path),'--output_dir',str(root/'ft_result')]
            done=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True,timeout=180)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            self.assertTrue((root/'ft_ckpt'/'last.pt').exists())
            manifest=json.loads((root/'ft_result'/'history_dataset_manifest.json').read_text())
            self.assertEqual(len(manifest['train']['excluded']),1)


if __name__=='__main__':
    unittest.main()
