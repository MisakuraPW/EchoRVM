import tempfile
import unittest
import importlib
import sys
import types
import subprocess
import os
import json
import logging
import shutil
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.temporal_mae import TemporalMAE, temporal_mask
from models.echocardmae_video import EchoCardMAEVideo, _median_blur_video
from models.video_mae import flat_sinusoid, EchoVideoMAE, tubelet_patchify
from models.downstream import EchoEFFineTuner, EchoSegFineTuner, segmentation_metrics
from utils.datasets import build_rmae_dataset
from utils.temporal_data import TemporalEchoDataset
from utils.echo_input import read_echo_input
from utils.pretrained_init import load_videomae_init
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.config import load_config
from tools.evaluate_temporal_mae import stream, ridge_fit, ridge_apply
from tools.run_temporal_research import experiment_matrix, summarize, archive_analysis


def tiny(mode='global', local_frames=4, clip_count=3):
    return dict(img_size=16, patch_size=4, local_frames=local_frames, clip_count=clip_count,
                frames=local_frames*clip_count, tubelet_size=min(2, local_frames), in_chans=3,
                embed_dim=24, depth=1, num_heads=3, decoder_embed_dim=24, decoder_depth=1,
                decoder_num_heads=3, mask_ratio=.5, memory_mode=mode, memory_grid=2,
                core_depth=1, separate_qv_bias=True, position_embedding='flat_sinusoid')


class TemporalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)

    def test_raw_avi_matches_rgb_cache(self):
        import cv2
        from tools.cache_echonet_npy import read_video as cache_decode
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_root, cache_root = root/'raw', root/'rgb'
            (raw_root/'Videos').mkdir(parents=True)
            (cache_root/'npy').mkdir(parents=True)
            for directory in (raw_root, cache_root):
                (directory/'FileList.csv').write_text('FileName,EF,Split\na,55,VAL\n')
            path = raw_root/'Videos/a.avi'
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (16,16))
            self.assertTrue(writer.isOpened())
            for i in range(6):
                frame = np.zeros((16,16,3),dtype=np.uint8)
                frame[...,0], frame[...,1], frame[...,2] = 20+i,80,180
                writer.write(frame)
            writer.release()
            np.save(cache_root/'npy/a.npy',cache_decode(path,grayscale=False))
            a = TemporalEchoDataset(raw_root,'val',4,img_size=16,channels=3)[0]
            b = TemporalEchoDataset(cache_root,'val',4,img_size=16,channels=3)[0]
            torch.testing.assert_close(a['video'],b['video'],rtol=0,atol=0)
            torch.testing.assert_close(a['frame_indices'],b['frame_indices'])
            self.assertGreater(float(a['video'][:,0].mean()),float(a['video'][:,2].mean()))
            gray_root = root/'gray'
            (gray_root/'npy').mkdir(parents=True)
            (gray_root/'FileList.csv').write_text('FileName,EF,Split\na,55,VAL\n')
            np.save(gray_root/'npy/a.npy',cache_decode(path,grayscale=True))
            from_raw = read_echo_input(path,'gray_repeat3')
            from_cache = read_echo_input(gray_root/'npy/a.npy','gray_repeat3')
            self.assertIsInstance(from_cache,np.memmap)
            self.assertFalse(from_cache.flags.writeable)
            np.testing.assert_array_equal(from_raw,from_cache)
            gray = TemporalEchoDataset(gray_root,'val',4,img_size=16,channels=3,
                                       two_views=True,input_protocol='gray_repeat3')[0]
            for key in ('video','video_view2'):
                torch.testing.assert_close(gray[key][:,0],gray[key][:,1],rtol=0,atol=0)
                torch.testing.assert_close(gray[key][:,0],gray[key][:,2],rtol=0,atol=0)
            self.assertEqual(gray['input_protocol'],'gray_repeat3')
            torch.testing.assert_close(gray['video'][:,0],
                                       torch.from_numpy(from_raw[gray['frame_indices'].numpy()]).float()/255)
            with self.assertRaisesRegex(ValueError,'RGB protocol'):
                read_echo_input(gray_root/'npy/a.npy','rgb')
            from_cache._mmap.close()

    def test_modes_backward_amp_and_reset(self):
        for mode in ('none', 'global', 'spatial', 'dual'):
            with self.subTest(mode=mode):
                model = TemporalMAE(**tiny(mode))
                video = torch.rand(2, 12, 3, 16, 16)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    out = model(video)
                out['loss'].backward()
                self.assertTrue(torch.isfinite(out['loss']))
                if mode != 'none':
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.memory.parameters() if p.grad is not None), 0)
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.feature_fusion.parameters() if p.grad is not None), 0)
                model.eval()
                with torch.no_grad():
                    a = model.forward_features(video)
                    b = model.forward_features(video)
                    torch.testing.assert_close(a, b)
                    permuted = model.forward_features(video.flip(0)).flip(0)
                    torch.testing.assert_close(a, permuted)
                    reset = model.forward_features(video, intervention='reset')
                    if mode != 'none':
                        self.assertGreater(float((a-reset).abs().sum()), 0)

    def test_downstream_gradient_and_per_image_dice(self):
        for kind in ('ef','seg'):
            backbone = TemporalMAE(**tiny())
            video = torch.rand(2,12,1,16,16)
            if kind == 'ef':
                head = EchoEFFineTuner(backbone,hidden_dim=24)
                prediction = head(video)
            else:
                head = EchoSegFineTuner(backbone,2,decoder_embed_dim=24,decoder_depth=1,decoder_num_heads=3)
                prediction = head.forward_video(video)
            prediction.square().mean().backward()
            self.assertGreater(float(backbone.patch_embed.proj.weight.grad.abs().sum()),0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in backbone.memory.parameters()))
        target = torch.zeros(2,4,4,dtype=torch.long)
        target[0] = 1
        target[1,0,0] = 1
        predicted = target.clone()
        predicted[1] = 0
        logits = F.one_hot(predicted,2).permute(0,3,1,2).float()*10
        result = segmentation_metrics(logits,target,2)
        self.assertAlmostEqual(result['dice_mean'],.5)

    def test_hidden_pixels_and_future_cannot_leak(self):
        model = TemporalMAE(**tiny()).eval()
        video = torch.rand(2, 12, 3, 16, 16)
        masks = torch.stack([temporal_mask(2, model.token_grid, .5, 'tube', video.device) for _ in range(3)], 1)
        pixel_mask = masks.reshape(2, 6, 4, 4).repeat_interleave(2, 1).repeat_interleave(4, 2).repeat_interleave(4, 3)[:, :, None]
        changed = torch.where(pixel_mask, video + 20, video)
        with torch.no_grad():
            base = model(video, masks=masks)['pred']
            hidden_changed = model(changed, masks=masks)['pred']
            torch.testing.assert_close(base, hidden_changed)
            future = video.clone()
            future[:, 8:] += 10
            other = model(future, masks=masks)['pred']
            torch.testing.assert_close(base[:, :64], other[:, :64])

    def test_mask_budgets(self):
        for strategy in ('tube', 'random_frame', 'temporal_block', 'complementary'):
            mask = temporal_mask(4, (4, 4, 4), .75, strategy, torch.device('cpu'))
            self.assertEqual(mask.sum(1).tolist(), [48] * 4)
            if strategy == 'tube':
                self.assertTrue(torch.equal(mask[:, :16], mask[:, 16:32]))

    def test_videomae_channelwise_target(self):
        cfg = tiny()
        cfg.update(norm_pix_loss=True,target_normalization='videomae')
        model = EchoVideoMAE(**cfg)
        video = torch.rand(2,12,3,16,16)
        raw = tubelet_patchify(video,2,4).reshape(2,96,32,3)
        expected = ((raw-raw.mean(-2,keepdim=True)) /
                    (raw.var(-2,unbiased=True,keepdim=True).sqrt()+1e-6)).flatten(2)
        torch.testing.assert_close(model(video)['target'],expected)

    def test_frame_model_padding_and_stream(self):
        model = TemporalMAE(**tiny(local_frames=1, clip_count=4)).eval()
        video = torch.rand(2, 7, 3, 16, 16)
        valid = torch.ones(2, 7, dtype=torch.bool)
        seq, pooled, states = stream(model, video, valid)
        self.assertEqual(seq.shape, (2, 7, 16, 24))
        self.assertEqual(pooled.shape, (2, 24))
        self.assertEqual(states.shape[0], 2)

    def test_roi_official_rounding_and_bias_load(self):
        cfg = tiny()
        cfg.update(frames=16, img_size=112, patch_size=8, mask_ratio=.75,
                   input_mean=[.1257,.1271,.1292], input_std=[.1951,.1957,.1974])
        model = EchoCardMAEVideo(**cfg)
        fg = model._foreground(2, torch.device('cpu'))
        count = int(fg[0].sum()) // 8
        self.assertEqual(model._roi_mask(fg).sum(1).tolist(), [(count-int(count*.25))*8]*2)
        self.assertIsNone(model.blocks[0].attn.qkv.bias)
        self.assertIsNone(model.decoder_embed.bias)
        self.assertEqual(model.norm.eps, 1e-6)
        self.assertEqual(float(model.background_token.detach().abs().sum()), 0)
        torch.testing.assert_close(model.pos_embed, flat_sinusoid(24, 8*196))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'init.pt'
            q = torch.arange(24).float()
            torch.save({'model': {'encoder.blocks.0.attn.q_bias': q,
                                   'encoder.blocks.0.attn.v_bias': -q}}, path)
            report = load_videomae_init(model, path)
            self.assertEqual(report['loaded_tensors'], 2)
            torch.testing.assert_close(model.blocks[0].attn.q_bias, q)

    def test_available_initializer_full_encoder_coverage(self):
        root = Path(__file__).resolve().parents[1]
        path = root/'ckpt/mae/videomae_vit_s.pth'
        if not path.exists():
            self.skipTest('Optional local initialization checkpoint is absent.')
        cfg = load_config(root/'configs/pretrain/stage0_echonet_echocardmae_400.yaml')['model']
        for kind in ('official','frame','hierarchy'):
            if kind == 'official':
                model = EchoCardMAEVideo(**cfg)
            else:
                adapted = dict(cfg,local_frames=1 if kind=='frame' else 16,clip_count=4,
                               tubelet_size=1 if kind=='frame' else 2,memory_mode='global')
                model = TemporalMAE(**adapted)
            report = load_videomae_init(model,path)
            self.assertEqual(report['missing_encoder_keys'],[],(kind,report))
            del model

    def test_median_matches_opencv(self):
        import cv2
        raw = np.arange(2*16*16, dtype=np.uint8).reshape(2,16,16)
        video = torch.from_numpy(raw.copy()).float()[None,:,None]
        expected = np.stack([cv2.medianBlur(x,3) for x in raw])
        np.testing.assert_array_equal(_median_blur_video(video,3).numpy()[0,:,0], expected)

    def test_official_forward_parity_when_reference_present(self):
        reference = Path(__file__).resolve().parents[1] / '资料/EchoCardMAE-main/EchoCardMAE-main/model'
        if not reference.exists():
            self.skipTest('Optional local official source is not distributed with the project.')
        package = types.ModuleType('_official_echocard_reference')
        package.__path__ = [str(reference)]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + '.echocardmae_vit')
        official = module.EchoCardMAEViT(num_frames=16, img_size=112, patch_size=8,
            encoder_embed_dim=24, encoder_depth=1, encoder_num_heads=3,
            decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=3,
            qkv_bias=True, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6)).eval()
        cfg = tiny()
        cfg.update(frames=16, img_size=112, patch_size=8, mask_ratio=.75,
                   input_mean=[.1257,.1271,.1292], input_std=[.1951,.1957,.1974])
        ours = EchoCardMAEVideo(**cfg).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'official.pt'
            torch.save({'model':official.state_dict()},path)
            report = load_videomae_init(ours,path)
            self.assertEqual(report['missing_after_partial_load'],0)
        video = torch.rand(2,16,3,112,112)
        foreground = ours._foreground(2,video.device)
        mask = ours._roi_mask(foreground)
        with torch.no_grad():
            predicted, feature = official(ours._normalize(video).transpose(1,2),mask,foreground)
            _, actual, actual_feature = ours._forward_view(video,mask,foreground)
        torch.testing.assert_close(actual,predicted,atol=2e-5,rtol=2e-5)
        torch.testing.assert_close(actual_feature,feature,atol=2e-5,rtol=2e-5)

    def test_real_npy_sampling_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            (root/'FileList.csv').write_text('FileName,EF,Split\na,60,TRAIN\nb,45,VAL\n')
            raw = np.arange(40,dtype=np.uint8)[:,None,None,None] * np.ones((40,16,16,3),dtype=np.uint8)
            raw[:,0,0] = [10,30,200]
            np.save(root/'npy/a.npy', raw)
            np.save(root/'npy/b.npy', raw)
            data = TemporalEchoDataset(root,'train',8,16,3,2)
            data.set_epoch(4)
            a, b = data[0], data[0]
            torch.testing.assert_close(a['video'],b['video'])
            torch.testing.assert_close(a['video'][0,:,0,0],torch.tensor([10,30,200])/255)
            self.assertEqual(torch.diff(a['frame_indices']).tolist(),[2]*7)
            legacy = build_rmae_dataset(dict(dataset_name='echonet', data_root=str(root)),
                                       dict(frames=4,img_size=16,in_chans=3,sampling_rate=2,two_views=True),'train')
            self.assertEqual(legacy[0]['video_view2'].shape,(4,3,16,16))
            padded = TemporalEchoDataset(root,'val',48,16,3)[0]
            self.assertEqual(int(padded['frame_valid'].sum()),40)
            model = TemporalMAE(**tiny())
            opt = torch.optim.AdamW(model.parameters(),lr=.001)
            model(torch.rand(2,12,3,16,16))['loss'].backward()
            opt.step()
            path = root/'last.pt'
            save_checkpoint(path,model,opt,None,None,1,1,.5,dict(model=tiny()))
            random_expected = torch.rand(2)
            restored = TemporalMAE(**tiny())
            ropt = torch.optim.AdamW(restored.parameters(),lr=.001)
            saved = load_checkpoint(path,restored,ropt)
            torch.testing.assert_close(torch.rand(2),random_expected)
            self.assertEqual(saved['global_step'],1)
            for key,value in model.state_dict().items():
                torch.testing.assert_close(value,restored.state_dict()[key])

    def test_ridge_and_manifest(self):
        x = torch.randn(30,8)
        y = x[:,0]*2
        prediction = ridge_apply(ridge_fit(x,y,alpha=.001),x)
        self.assertLess(float((prediction-y).abs().mean()),.001)
        entries = experiment_matrix('full')
        names = [row[0] for row in entries]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(len(experiment_matrix('core')),9)
        self.assertEqual(len(entries),37)

    def test_training_to_audit_integration(self):
        import yaml
        project = Path(__file__).resolve().parents[1]
        env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'npy').mkdir()
            files, traces = ['FileName,EF,Split'], ['FileName,Frame,X1,Y1,X2,Y2']
            for i in range(12):
                case = f'case{i:02d}'
                split = 'TRAIN' if i < 8 else 'VAL'
                files.append(f'{case},{40+i*2},{split}')
                raw = np.random.default_rng(i).integers(0,255,(32,112,112),dtype=np.uint8)
                np.save(root/'npy'/f'{case}.npy',raw)
                for frame, width in ((8,25),(24,15)):
                    for y in (20,30,50,70,90):
                        traces.append(f'{case}.avi,{frame},{56-width},{y},{56+width},{y}')
            (root/'FileList.csv').write_text('\n'.join(files))
            (root/'VolumeTracings.csv').write_text('\n'.join(traces))
            model_cfg = tiny(local_frames=4,clip_count=2)
            model_cfg.update(name='temporal_mae',img_size=112,patch_size=8,gradient_checkpointing=True)
            cfg = dict(experiment=dict(seed=42), model=model_cfg,
                       data=dict(loader='echonet',sampling_protocol='temporal_v1',input_protocol='gray_repeat3',data_root=str(root),
                                 num_workers=0,drop_last=True),
                       train=dict(epochs=1,batch_size=2,max_steps=2,val_interval=1,plot_interval=1,
                                  grad_accum_steps=1,mixed_precision=False),
                       optimizer=dict(name='adamw',lr=.001),scheduler=dict(name='none'),
                       checkpoint=dict(auto_resume=True,save_epochs=[1],save_initial=True,
                                       save_every_n_epochs=0,epoch_name_width=4),
                       early_stopping=dict(enabled=False),logging=dict(use_tqdm=False,use_tensorboard=False))
            path = root/'config.yaml'
            path.write_text(yaml.safe_dump(cfg))
            run = root/'result/run'
            ckpt = root/'ckpt/run'
            command = [sys.executable,'trainers/train_rmae.py','--config',str(path),'--output_dir',str(run),
                       '--checkpoint_dir',str(ckpt),'--save_last_every','10']
            subprocess.run(command,cwd=project,env=env,check=True,capture_output=True,text=True)
            self.assertTrue((run/'plots/loss_latest.png').exists())
            self.assertFalse((run/'checkpoints').exists())
            self.assertTrue((ckpt/'epoch_0000.pt').exists())
            first = torch.load(ckpt/'last.pt',weights_only=False)
            self.assertEqual(first['config']['data']['input_protocol'],'gray_repeat3')
            subprocess.run(command+['--epochs','2','--checkpoint_epochs','1 2'],cwd=project,env=env,
                           check=True,capture_output=True,text=True)
            second = torch.load(ckpt/'last.pt',weights_only=False)
            self.assertEqual(second['epoch'],2)
            self.assertEqual(second['global_step'],first['global_step']+2)
            audit = root/'audit'
            command = [sys.executable,'tools/evaluate_temporal_mae.py','--checkpoint',
                       str(ckpt/'epoch_0002.pt'),'--data_root',str(root),
                       '--output_dir',str(audit),'--smoke','--audit_frames','16','--batch_size','2']
            result = subprocess.run(command,cwd=project,env=env,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            metrics = json.loads((audit/'metrics.json').read_text())
            self.assertEqual(metrics['protocol']['input_protocol'],'gray_repeat3')
            self.assertEqual(metrics['metadata']['missing_keys'],0)
            self.assertIn('reset',metrics['interventions'])
            self.assertIn('state_only_ef',metrics)
            self.assertIn('state_only_seg_dice',metrics['segmentation'])
            self.assertTrue((audit/'ef_predictions.csv').exists())
            fine_cfg = dict(experiment=dict(seed=42),model=dict(img_size=112,frames=8),
                            data=dict(data_root=str(root),num_workers=0,input_protocol='gray_repeat3'),
                            train=dict(epochs=1,batch_size=2,max_steps=2,mixed_precision=False),
                            optimizer=dict(name='adamw',lr=.001),scheduler=dict(name='none'),
                            checkpoint=dict(auto_resume=True,save_every_n_epochs=0,best_weights_only=True),
                            early_stopping=dict(enabled=False),
                            logging=dict(use_tqdm=False,use_tensorboard=False))
            fine_path = root/'fine.yaml'
            fine_path.write_text(yaml.safe_dump(fine_cfg))
            fine_result, fine_ckpt = root/'result/fine',root/'ckpt/fine'
            command = [sys.executable,'trainers/train_finetune.py','--task','echonet_ef','--config',str(fine_path),
                       '--pretrained',str(ckpt/'epoch_0002.pt'),'--output_dir',str(fine_result),
                       '--checkpoint_dir',str(fine_ckpt),'--save_last_every','10']
            fine = subprocess.run(command,cwd=project,env=env,capture_output=True,text=True)
            self.assertEqual(fine.returncode,0,fine.stdout+fine.stderr)
            self.assertFalse(list((root/'result').rglob('*.pt')))
            self.assertTrue((fine_ckpt/'last.pt').is_file())
            self.assertIsNone(torch.load(fine_ckpt/'best.pt',weights_only=False)['optimizer_state_dict'])
            report_root = root/'reports'
            for name in ('clip_mae_pool64','hier_global'):
                shutil.copytree(audit,report_root/name/'audit/epoch_0002')
            summarize(report_root)
            self.assertTrue((report_root/'paired_comparisons.csv').exists())
            archive_analysis(report_root)
            with zipfile.ZipFile(report_root/'analysis.zip') as bundle:
                self.assertTrue(any(name.endswith('metrics.json') for name in bundle.namelist()))
                self.assertFalse(any(name.endswith(('.pt','.npz')) for name in bundle.namelist()))

    def test_partial_accumulation_window(self):
        from trainers.train_rmae import run_epoch
        class ScalarMAE(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))
            def forward(self,video):
                return {'loss':self.weight*video.mean()}
        model = ScalarMAE()
        loader = torch.utils.data.DataLoader([{'video':torch.tensor([float(i)])} for i in (1,3)],batch_size=1)
        opt = torch.optim.SGD(model.parameters(),lr=.1)
        scaler = torch.amp.GradScaler('cuda',enabled=False)
        cfg = dict(train=dict(grad_accum_steps=3,mixed_precision=False),logging=dict(use_tqdm=False))
        _,steps = run_epoch(model,loader,opt,None,scaler,torch.device('cpu'),cfg,1,0,True,None,logging.getLogger('test'))
        self.assertEqual(steps,1)
        self.assertAlmostEqual(float(model.weight.detach()),.8,places=6)


if __name__ == '__main__':
    unittest.main()
