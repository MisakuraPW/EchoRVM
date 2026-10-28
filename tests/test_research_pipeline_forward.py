"""Forward candidates and fixed-schedule budget pause contracts."""

import copy
import tempfile
from pathlib import Path
import unittest
from unittest import mock
from types import SimpleNamespace
import json

import torch

from models.final_temporal_mae import FinalTemporalMAE, load_final_model
from models.forward_temporal_mae import ForwardTemporalMAE, ContentWriter
from research_pipeline.config import load, stage_config
from research_pipeline.artifacts import layout, verify_done
from research_pipeline.cli import configure, source_path
from research_pipeline.engine import run_training
from research_pipeline.forward_diagnostics import blocked_reads, gradient_credit, stream
from research_pipeline.forward_suite import build_plan


def native():
    return dict(img_size=16,patch_size=8,local_frames=4,clip_count=4,embed_dim=24,depth=4,num_heads=4,
                decoder_embed_dim=24,decoder_depth=1,decoder_num_heads=4,in_chans=3,tubelet_size=2,
                mask_ratio=.5,core_depth=1,memory_mode='global',frame_readout='repeat',
                reconstruction_recent_frames=32,norm_pix_loss=True)


class ForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_baseline_helper_is_exact_legacy_path(self):
        model = FinalTemporalMAE(**native()).eval()
        clip,state = torch.rand(2,4,3,16,16),torch.rand(2,1,24)
        raw,_ = model.encode_video(clip,None)
        expected = model.feature_fusion(raw,state.to(raw))
        actual,local = model.encode_current(clip,None,state)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(local,raw,rtol=0,atol=0)

    def test_shared_init_copied_and_a_reads_once_at_midpoint(self):
        cfg = native()
        model = FinalTemporalMAE(**cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'zero.pt'
            torch.save(dict(model_state_dict=model.state_dict(),config=dict(model=cfg),epoch=0),path)
            for variant in ('mid_read','content_write'):
                candidate,_,report = load_final_model(path,dict(forward_variant=variant,read_after_block=2))
                candidate.eval()
                self.assertFalse(report['discarded_keys'])
                for key,value in model.state_dict().items():
                    torch.testing.assert_close(value,candidate.state_dict()[key],rtol=0,atol=0)
                clip = torch.rand(1,4,3,16,16)
                torch.testing.assert_close(model.stream_clip(clip)['features'],candidate.stream_clip(clip)['features'],rtol=0,atol=0)
                if variant=='content_write':
                    torch.testing.assert_close(model.stream_clip(clip)['final_state'],candidate.stream_clip(clip)['final_state'],rtol=1e-6,atol=1e-6)
                if variant=='mid_read':
                    count,calls = [0],[]
                    handles = [block.register_forward_hook(lambda *args:count.__setitem__(0,count[0]+1)) for block in candidate.blocks]
                    handles.append(candidate.feature_fusion.register_forward_pre_hook(lambda *args:calls.append(count[0])))
                    out1 = candidate.stream_clip(clip)
                    out2 = candidate.stream_clip(clip,out1['final_state'])
                    self.assertEqual(calls,[6])
                    self.assertEqual(count[0],8)
                    for handle in handles: handle.remove()
                    self.assertEqual(out2['final_state'].shape,(1,1,24))

    def test_writer_uniform_validity_and_delayed_gradients(self):
        writer = ContentWriter(24,8)
        x,state = torch.randn(2,7,24),torch.randn(2,1,24)
        visible = torch.ones(2,7,dtype=torch.bool)
        visible[:,-2:] = False
        expected = x[:,:5].mean(1,keepdim=True)
        torch.testing.assert_close(writer(x,state,visible),expected,rtol=1e-6,atol=1e-6)
        bad = x.clone(); bad[:,-2:] = 1e8
        torch.testing.assert_close(writer(bad,state,visible),writer(x,state,visible),rtol=0,atol=0)
        optimizer = torch.optim.SGD(writer.parameters(),lr=.1)
        for step in range(3):
            optimizer.zero_grad()
            writer(x,state,visible).square().sum().backward()
            if step==0:
                self.assertGreater(float(writer.score.weight.grad.norm()),0)
            if step==1:
                self.assertGreater(float(writer.feature.weight.grad.norm()),0)
                self.assertGreater(float(writer.state.weight.grad.norm()),0)
            optimizer.step()
        # Nonlinear state conditioning must not collapse to an additive constant in softmax.
        self.assertFalse(torch.allclose(writer(x,state,visible),writer(x,torch.zeros_like(state),visible)))

    def test_independent_edges_and_time_specific_credit(self):
        model = FinalTemporalMAE(**native()).eval()
        video = torch.rand(1,12,3,16,16)
        masks = torch.tensor([[[False,False,True,True]*2]*3])
        normal = gradient_credit(model,video,masks)
        neither = gradient_credit(model,video,masks,True,True)
        self.assertGreater(normal['states'][0]['norm'],0)
        self.assertTrue(neither['states'][0]['norm'] in (None,0))
        self.assertFalse(normal['states'][-1]['connected'])
        self.assertEqual(len(normal['time_specific_writes']),3)
        reference = stream(model,video,4)[-1]['sequence']
        with blocked_reads(model,decoder=True):
            torch.testing.assert_close(stream(model,video,4)[-1]['sequence'],reference,rtol=0,atol=0)
        with blocked_reads(model,feature=True):
            self.assertFalse(torch.equal(stream(model,video,4)[-1]['sequence'],reference))

    def test_future_isolation_and_both_cache_reset_protocols(self):
        model = ForwardTemporalMAE(**native(),forward_variant='mid_read',read_after_block=2).eval()
        video = torch.rand(1,24,3,16,16)
        with torch.no_grad():
            clean = stream(model,video,8)
            changed = video.clone(); changed[:,-4:] = 0
            other = stream(model,changed,8)
            torch.testing.assert_close(clean[1]['state'],other[1]['state'],rtol=0,atol=0)
            a = stream(model,video,8,reset_recent=True)[-1]['sequence']
            b = stream(model,video,8,clear_last=True)[-1]['sequence']
            self.assertFalse(torch.equal(a,b))

    def test_candidates_checkpointed_bf16_backward(self):
        for variant in ('mid_read','content_write'):
            model = ForwardTemporalMAE(**native(),forward_variant=variant,read_after_block=2).train()
            model.gradient_checkpointing = True
            with torch.autocast('cpu',dtype=torch.bfloat16):
                loss = model(torch.rand(1,12,3,16,16))['loss']
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))

    def test_pause_resume_matches_full_schedule_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load('configs/pipeline/synthetic.yaml',root=tmp,tag='continuous')
            cfg['training'].update(epochs=4,schedule_epochs=4,run_until_epoch=4,warmup_epochs=0,
                                    patience=None,checkpoint_every=1,keep_snapshots=0,save_epochs=[0])
            cfg = stage_config(cfg,dict(id='pretrain',kind='mae'))
            configure(cfg)
            run_training(cfg,None)
            split = copy.deepcopy(cfg); split['run']['tag']='split'; split['training']['run_until_epoch']=2
            paused = run_training(split,None)
            output = layout(split,'pretrain')['result']
            self.assertEqual(paused['phase'],'paused_at_budget')
            self.assertFalse(verify_done(output))
            with self.assertRaisesRegex(ValueError,'not completed'):
                source_path(split,dict(source='@pretrain/final'))
            self.assertTrue(Path(source_path(split,dict(source='@pretrain/epoch_0002'))).exists())
            snapshot = torch.load(layout(split,'pretrain')['ckpt']/'last.pt',weights_only=False)
            self.assertGreater(snapshot['optimizer']['param_groups'][0]['lr'],0)
            split['training']['run_until_epoch']=4
            run_training(split,None)
            for name in ('model','optimizer','scheduler'):
                a = torch.load(layout(cfg,'pretrain')['ckpt']/'last.pt',weights_only=False)[name]
                b = torch.load(layout(split,'pretrain')['ckpt']/'last.pt',weights_only=False)[name]
                if name=='model':
                    for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                elif name=='optimizer':
                    self.assertEqual(a['param_groups'],b['param_groups'])
                    for parameter in a['state']:
                        for key in a['state'][parameter]:
                            torch.testing.assert_close(a['state'][parameter][key],b['state'][parameter][key],rtol=0,atol=0)
                elif name=='scheduler': self.assertEqual(a,b)
            self.assertTrue((layout(split,'pretrain')['ckpt']/'epoch_0002.pt').exists())
            durable = torch.load(layout(split,'pretrain')['ckpt']/'resume_epoch_0002.pt',weights_only=False)
            self.assertEqual(durable['progress']['epoch'],2)
            self.assertIn('optimizer',durable)
            self.assertTrue(verify_done(output))

    def test_default_queue_is_bounded_and_w_is_opt_in(self):
        cfg,selected = build_plan(load('configs/pipeline/clip_recurrent_global.yaml'))
        self.assertIn('A',selected)
        self.assertNotIn('W',selected)
        self.assertNotIn('A_ef_0800',selected)
        a = stage_config(cfg,next(s for s in cfg['stages'] if s['id']=='A'))
        self.assertEqual(a['training']['schedule_epochs'],1600)
        self.assertEqual(a['training']['run_until_epoch'],400)
        with self.assertRaises(ValueError):
            build_plan(load('configs/pipeline/clip_recurrent_global.yaml'),endpoint=600)

    def test_file_backed_queue_diagnostics_probes_candidates_and_compare(self):
        from test_research_pipeline_clean import fixture
        from research_pipeline.artifacts import write_json
        from research_pipeline.cli import execute, work
        from research_pipeline.forward_suite import compare
        from utils.final_temporal_data import build_manifest
        import pandas as pd
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            files = pd.read_csv(root/'FileList.csv'); files['EF'] = 60
            files.to_csv(root/'FileList.csv',index=False)
            traces = [dict(FileName=patient,Frame=12,X1=a,Y1=b,X2=c,Y2=d)
                      for patient in ('train','val') for a,b,c,d in ((0,0,112,112),(30,25,80,25),(25,55,85,55),(40,85,70,85))]
            pd.DataFrame(traces).to_csv(root/'VolumeTracings.csv',index=False)
            write_json(root/'task.json',build_manifest(root,recent_frames=8,local_frames=4,max_prefix=8))
            base = load('configs/pipeline/clip_recurrent_global.yaml',root=str(root/'out'),tag='queue')
            base['runtime'].update(device='cpu',precision='fp32',autotune=False,micro_batch=1,
                                   worker_candidates=[0],min_free_gb=0,cache_gb=1)
            base['training'].update(epochs=1,warmup_epochs=0,effective_batch=2,tensorboard=False)
            base['model'] = dict(config=dict(native(),img_size=112,patch_size=16,depth=12),overrides={})
            base['data'] = dict(manifest=str(root/'task.json'),ssl_manifest=str(root/'ssl.json'),
                                local_frames=4,recent_frames=8,max_prefix=8,prefix=None,diagnostic_cases=2)
            base['stages'] = [dict(id='pretrain',kind='mae',training=dict(epochs=4,patience=None,
                                      warmup_epochs=0,checkpoint_every=1,save_epochs=[0,1,2,4]))]
            cfg,selected = build_plan(base,with_w=True)
            cfg['smoke'] = True  # Permits bounded legacy audits at test-only L4.
            def in_process(command,**kwargs):
                request = json.loads(Path(command[-1]).read_text())
                work(request)
                return SimpleNamespace(returncode=0)
            with mock.patch('research_pipeline.cli.subprocess.run',in_process):
                execute(cfg,selected)
            self.assertFalse(verify_done(layout(cfg,'A')['result']))
            self.assertTrue((layout(cfg,'A')['result']/'PAUSED.json').exists())
            for stage in ('D_paths_history','A_diagnostics','W_diagnostics'):
                directory = layout(cfg,stage)['result']
                self.assertTrue(verify_done(directory))
                self.assertTrue(json.loads((directory/'summary.json').read_text())['source_unchanged'])
            cfg['smoke'] = False
            compare(cfg,selected)
            rows = json.loads((layout(cfg)['result']/'forward_comparison.json').read_text())['rows']
            self.assertEqual(len(rows),4)
            self.assertTrue(all(row['status']=='matched' for row in rows))


if __name__=='__main__':
    unittest.main()
