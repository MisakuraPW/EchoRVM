"""CPU file-backed history controls; fixture heads are not research results."""

import copy
import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from models.final_temporal_mae import FinalTemporalMAE
from utils.final_temporal_data import build_manifest, WindowDataset
from utils.final_temporal_tasks import _TaskModel, load_frozen_task_head
from utils.final_temporal_training import native_video
import utils.final_temporal_history as history


POINTS = [[0, 0, 112, 112], [35, 30, 75, 30], [25, 55, 85, 55], [42, 85, 68, 85]]


def config(size=16, local=4, recent=8):
    return dict(name='temporal_final', img_size=size, patch_size=size//2, local_frames=local,
                clip_count=recent//local, tubelet_size=2, in_chans=3, embed_dim=12, depth=1,
                num_heads=3, decoder_embed_dim=12, decoder_depth=1, decoder_num_heads=3,
                memory_mode='spatial', memory_grid=2, core_depth=1, frame_readout='learned',
                memory_write_source='local', norm_pix_loss=False, mask_ratio=.5,
                gradient_checkpointing=False, position_embedding='flat_sinusoid')


def head_files(root, cfg, recent=8):
    """Known fixture tensors exercising the actual frozen-checkpoint format."""
    torch.manual_seed(42)
    model = FinalTemporalMAE(**cfg).eval()
    source = root/'source.pt'
    torch.save(dict(model_state_dict=model.state_dict(), config=dict(model=cfg), epoch=0), source)
    paths = {}
    for task in ('ef', 'seg'):
        normalization = dict(feature_mean=[0.]*12, feature_std=[1.]*12,
                             target_mean=50., target_std=10., fixture=True)
        job = dict(task=task, exit_name='base', append_boundary=False, recent_frames=recent,
                   head_dim=12, head_depth=1, head_heads=3)
        task_model = _TaskModel(model, job, normalization).eval()
        protocol = dict(task=task, freeze=True, exit_name='base', append_boundary=False,
                        recent_frames=recent, fixture=True, head=dict(dim=12, depth=1, heads=3))
        checkpoint = root/(task+'_head.pt')
        state = {k:v for k,v in task_model.state_dict().items()
                 if k.startswith('head.') or k in ('feature_mean', 'feature_std')}
        torch.save(dict(model_state_dict=state, normalization=normalization, protocol=protocol), checkpoint)
        paths[task] = str(checkpoint)
    return model, source, paths


def fixture(root):
    (root/'npy').mkdir()
    rows, traces = [], []
    for i, (name, split) in enumerate((('train_a', 'TRAIN'), ('train_b', 'TRAIN'),
                                       ('val_a', 'VAL'), ('val_b', 'VAL'))):
        video = np.broadcast_to((np.arange(32)*7+i).astype(np.uint8)[:, None, None], (32,112,112)).copy()
        np.save(root/'npy'/(name+'.npy'), video)
        rows.append(dict(FileName=name, Split=split, EF=40.+20*(i % 2), FPS=40+i*10, NumberOfFrames=32))
        for frame in (6, 15, 23, 29):
            for x1,y1,x2,y2 in POINTS:
                traces.append(dict(FileName=name+'.avi', Frame=frame, X1=x1,Y1=y1,X2=x2,Y2=y2))
    pd.DataFrame(rows).to_csv(root/'FileList.csv', index=False)
    pd.DataFrame(traces).to_csv(root/'VolumeTracings.csv', index=False)
    return build_manifest(root, recent_frames=8, local_frames=4, max_prefix=32)


class HistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = fixture(self.root)
        self.model, self.source, self.heads = head_files(self.root, config())

    def job(self, name='run', **updates):
        result = dict(checkpoint=str(self.source), output_dir=str(self.root/name/'result'),
                      checkpoint_dir=str(self.root/name/'weights'), recent_frames=8, max_prefix=32,
                      seed=42, smoke=True, train_cases=2, val_cases=2, micro_batch=4,
                      num_workers=0, head_epochs=1, hidden=12, quiet=True, bootstrap_repetitions=40,
                      ef_head_checkpoint=self.heads['ef'], seg_head_checkpoint=self.heads['seg'])
        result.update(updates)
        return result

    def test_official_trace_intersection_all_positions_and_true_prefix(self):
        records = history.seg_history_records(self.manifest, 'val', 8, recent=8, local=4)
        self.assertEqual(len(records), 16)
        self.assertEqual({r['target_frame'] for r in records}, {15, 23})
        self.assertEqual({r['target_position'] for r in records}, {4,5,6,7})
        limited = history.seg_history_records(self.manifest, 'val', 8, 8, 4, limit=1)
        self.assertEqual(len(limited), 8)
        self.assertEqual(len({r['patient'] for r in limited}), 1)
        self.assertEqual(history.seg_history_records(self.manifest, 'val', 32, 8, 4), [])
        data = [WindowDataset(self.manifest, 'val', task='seg', recent_frames=8, local_frames=4,
                              max_prefix=32, records=changed) for changed in (
            records, [dict(r,H=0) for r in records], [dict(r,prefix_kind='repeat_prefix') for r in records])]
        for index, record in enumerate(records):
            samples = [d[index] for d in data]
            for sample in samples:
                self.assertEqual(sample['target_recent_index'], record['target_position'])
                self.assertEqual(sample['target_index'], sample['prefix_frames']+record['target_position'])
                self.assertEqual(int(sample['frame_indices'][sample['target_index']]), record['target_frame'])
                self.assertTrue(sample['full_context'])
                torch.testing.assert_close(sample['video'][-8:], samples[0]['video'][-8:])
                torch.testing.assert_close(sample['mask'], samples[0]['mask'])
            np.testing.assert_array_equal(samples[2]['frame_indices'][:8].numpy(),
                                          np.tile(samples[0]['frame_indices'][:4].numpy(), 2))

    def test_fixed_frozen_seg_reuses_exit_normalization_and_absolute_index(self):
        head = load_frozen_task_head(self.heads['seg'], self.model)
        records = history.seg_history_records(self.manifest, 'val', 8, 8, 4, 1)
        rows = history.fixed_head_predictions(self.model, head, self.manifest, 'val', records,
                                              'seg', self.job(), torch.device('cpu'))
        self.assertEqual(len(rows), 8)
        self.assertFalse(any(p.requires_grad for p in head.parameters()))
        sample = WindowDataset(self.manifest, 'val', task='seg', recent_frames=8, local_frames=4,
                               max_prefix=32, records=records)[0]
        with torch.no_grad():
            video = native_video(sample['video'][None], self.model, True)
            logits = head(video, prefix_frames=8, target_index=[sample['target_index']])
            mask = sample['mask'].bool()[None]
            hard = torch.nn.functional.interpolate(logits, (112,112), mode='bilinear',
                                                    align_corners=False).argmax(1).bool()
            expected = float((2*(hard & mask).sum()+1e-6)/(hard.sum()+mask.sum()+1e-6))
        self.assertAlmostEqual(rows[0]['dice'], expected, places=6)
        self.assertEqual(rows[0]['position']+8, rows[0]['target_index'])
        self.assertAlmostEqual(rows[0]['prefix_seconds'], 8/rows[0]['fps'])

    def test_fixed_ef_denormalizes_and_keeps_existing_recent_window(self):
        head = load_frozen_task_head(self.heads['ef'], self.model)
        records = history.history_records(self.manifest, 'val', 8, 8)
        smaller = history.history_records(self.manifest, 'val', 4, 8)
        self.assertEqual([(r['patient'],r['recent_start']) for r in records],
                         [(r['patient'],r['recent_start']) for r in smaller])
        rows = history.fixed_head_predictions(self.model, head, self.manifest, 'val', records,
                                              'ef', self.job(), torch.device('cpu'))
        sample = WindowDataset(self.manifest, 'val', recent_frames=8, local_frames=4,
                               max_prefix=32, records=records)[0]
        with torch.no_grad():
            expected = float(head(native_video(sample['video'][None], self.model, True), prefix_frames=8)*10+50)
        self.assertAlmostEqual(rows[0]['prediction'], expected, places=5)

    def test_strict_patient_window_source_position_pairing_and_aggregation(self):
        a = [dict(patient='a', recent_start=10, source_frame=15, position=5, dice=.8),
             dict(patient='a', recent_start=9, source_frame=15, position=6, dice=.6),
             dict(patient='a', recent_start=18, source_frame=23, position=5, dice=.5),
             dict(patient='b', recent_start=10, source_frame=15, position=5, dice=.9)]
        b = [dict(r,dice=r['dice']-.1) for r in reversed(a)]
        paired = history.paired_delta(a,b,'dice',repetitions=50)
        self.assertAlmostEqual(paired['delta'], .1)
        self.assertEqual(paired['patients'], 2)
        for changed in (b[:-1], b+[b[0]], [dict(b[0], position=7), *b[1:]]):
            with self.assertRaises(ValueError):
                history.paired_delta(a,changed,'dice',repetitions=50)
        self.assertAlmostEqual(history._patient_mean(a,'dice'), (.6+.9)/2)
        left = [dict(r,source_path='original.npy') for r in a]
        right = [dict(r,source_path='changed.npy') for r in b]
        with self.assertRaisesRegex(ValueError,'pairing differs'):
            history.paired_delta(left,right,'dice',repetitions=50)

    def test_registered_segmentation_primary_patient_boundary(self):
        for patients in (0,1,63):
            self.assertFalse(history._seg_primary(patients))
        self.assertTrue(history._seg_primary(64))
        self.assertTrue(history._seg_primary(65))
        self.assertFalse(history._seg_primary(0,smoke=True))
        self.assertTrue(history._seg_primary(1,smoke=True))

    def test_anchor_insufficient_formal_still_evaluates_fixed_ef_and_seg(self):
        self.model, self.source, self.heads = head_files(self.root, config(size=112))
        job = self.job(smoke=False)
        with patch.object(history, 'fit_head', side_effect=AssertionError('Insufficient population must not fit')):
            metrics = history.run_history_job(job,self.manifest,'cpu')
        self.assertIsNone(metrics['anchor'])
        self.assertEqual(metrics['capability_status'], 'insufficient_anchor_population')
        self.assertEqual(metrics['status'], 'complete')
        self.assertEqual(metrics['capabilities'], {})
        for task in ('ef','seg'):
            self.assertEqual(metrics['fixed_head_coverage'][task]['8']['evaluation_status'], 'evaluated')
            self.assertEqual(metrics['fixed_head_coverage'][task]['32']['selected']['patients'], 0)
            self.assertEqual(metrics['fixed_head_coverage'][task]['32']['evaluation_status'], 'no_real_cohort')
        seg = metrics['fixed_head_coverage']['seg']['8']
        self.assertEqual(seg['population_scope'], 'auxiliary')
        self.assertEqual(seg['primary_patient_minimum'],64)
        self.assertEqual(seg['selected']['patients'],2)
        pairs = [p for p in metrics['paired'] if p['task']=='seg' and p['prefix']==8]
        self.assertEqual(len(pairs),2)
        self.assertTrue(all(p['patients']==2 and p['windows']==16 and p['source_frames']==4 for p in pairs))
        self.assertTrue(all(p['better_direction']=='positive' for p in pairs))
        output = Path(job['output_dir'])
        counts = json.loads((output/'cohort_counts.json').read_text())
        self.assertEqual(counts['fixed_head_coverage'],metrics['fixed_head_coverage'])
        with (output/'seg_fixed_head_h8_real.csv').open(newline='') as handle:
            real = list(csv.DictReader(handle))
        for condition in ('h0','repeat'):
            with (output/f'seg_fixed_head_h8_{condition}.csv').open(newline='') as handle:
                control = list(csv.DictReader(handle))
            self.assertEqual([(r['patient'],r['recent_start'],r['source_frame'],r['position']) for r in real],
                             [(r['patient'],r['recent_start'],r['source_frame'],r['position']) for r in control])
        with patch.object(history,'load_final_model',side_effect=AssertionError('Completed result should reuse')):
            self.assertEqual(history.run_history_job(job,self.manifest,'cpu'), metrics)
        from matplotlib.image import imread
        pixels = imread(output/'plots/paired_history.png')
        self.assertGreater(float(pixels[...,:3].std()),.03)
        self.assertIn('no cross-population absolute trend',metrics['note'])

    def test_smoke_registered_capability_counts_and_loss_plot(self):
        job = self.job(max_prefix=8)
        metrics = history.run_history_job(job,self.manifest,'cpu')
        self.assertEqual(metrics['capability_status'],'complete')
        self.assertEqual(set(metrics['capabilities']),{'anchor_h0','anchor_real','local_empty','local_history'})
        self.assertEqual(metrics['capability_minimum'],dict(train=2,val=1))
        self.assertEqual(metrics['fixed_head_coverage']['seg']['8']['population_scope'],'primary')
        self.assertTrue((Path(job['output_dir'])/'plots/capability_loss.png').is_file())
        self.assertEqual(metrics['fixed_head_curve'][0]['scope'],'fixed_anchor_head')

    def test_selected_anchor_capability_threshold_not_bypassed_by_limit(self):
        job = self.job(train_cases=1, max_prefix=8)
        with patch.object(history,'fit_head',side_effect=AssertionError('Do not bypass selected threshold')):
            metrics = history.run_history_job(job,self.manifest,'cpu')
        self.assertEqual(metrics['anchor'],8)
        self.assertEqual(metrics['capability_status'],'insufficient_selected_anchor_population')
        self.assertEqual(metrics['fixed_head_coverage']['ef']['8']['evaluation_status'],'evaluated')
        self.assertEqual(metrics['fixed_head_curve'][0]['scope'],'fixed_frozen_ef_head')

    def test_no_history_or_no_head_reports_explicit_counts(self):
        job = self.job(train_cases=0,max_prefix=8,seg_head_checkpoint=None,ef_head_checkpoint=None)
        metrics = history.run_history_job(job,self.manifest,'cpu')
        self.assertEqual(metrics['status'],'insufficient_history_evidence')
        self.assertTrue(all(c['mae'] is None for c in metrics['fixed_head_curve']))
        self.assertTrue(all(c['patient_dice'] is None for c in metrics['seg_fixed_head_curve']))
        self.assertEqual(metrics['fixed_head_coverage']['seg']['8']['evaluation_status'],'missing_fixed_head')
        self.assertEqual(metrics['fixed_head_coverage']['seg']['8']['selected']['patients'],2)
        self.assertEqual(metrics['paired'],[])
        self.assertFalse(metrics['history_content_supported'])
        self.assertEqual(history.fixed_head_predictions(None,None,self.manifest,'val',[],'seg',job,
                                                       torch.device('cpu')),[])

    def test_frozen_protocol_shape_mismatch_is_explicit(self):
        head = load_frozen_task_head(self.heads['seg'],self.model)
        rows = history.seg_history_records(self.manifest,'val',4,8,4,1)
        with self.assertRaisesRegex(ValueError,'protocol mismatch'):
            history.fixed_head_predictions(self.model,head,self.manifest,'val',rows,'ef',self.job(),torch.device('cpu'))
        with self.assertRaisesRegex(ValueError,'silently resize'):
            history.fixed_head_predictions(self.model,head,self.manifest,'val',rows,'seg',self.job(smoke=False),
                                           torch.device('cpu'))


class RealHistoryAudit(unittest.TestCase):
    def test_real_npy_cache_original_label_all_positions_paired_controls(self):
        from echo_aug_validation.io_utils import find_echonet_video
        from utils.echo_input import read_echo_input
        root = Path(os.environ.get('FINAL_TEMPORAL_DATA_ROOT','G:/SRTP/dataset/EchoNet-Dynamic'))
        if not (root/'FileList.csv').exists():
            self.skipTest('Set FINAL_TEMPORAL_DATA_ROOT for the real-video audit')
        torch.set_num_threads(2)
        files = pd.read_csv(root/'FileList.csv'); traces = pd.read_csv(root/'VolumeTracings.csv')
        chosen = None
        for _, row in files[files.Split=='VAL'].iterrows():
            name = Path(str(row.FileName)).stem
            labels = traces.loc[traces.FileName==name+'.avi','Frame'].unique()
            if any(f>=95 and f+16<=row.NumberOfFrames for f in labels):
                raw = read_echo_input(find_echonet_video(root,name),'gray_repeat3')
                if any(f>=95 and f+16<=len(raw) for f in labels):
                    chosen = row; break
        self.assertIsNotNone(chosen)
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp); (cache/'npy').mkdir()
            name = Path(str(chosen.FileName)).stem
            np.save(cache/'npy'/(name+'.npy'),raw)
            pd.DataFrame([chosen]).to_csv(cache/'FileList.csv',index=False)
            traces[traces.FileName==name+'.avi'].to_csv(cache/'VolumeTracings.csv',index=False)
            manifest = build_manifest(cache,max_prefix=32)
            model,_,heads = head_files(cache,config(local=16,recent=64),recent=64)
            head = load_frozen_task_head(heads['seg'],model)
            records = history.seg_history_records(manifest,'val',32,64,16)
            self.assertGreaterEqual(len(records),16)
            job = dict(recent_frames=64,max_prefix=32,smoke=True,micro_batch=4,num_workers=0)
            controls = []
            for altered in (records,[dict(r,H=0) for r in records],
                            [dict(r,prefix_kind='repeat_prefix') for r in records]):
                controls.append(history.fixed_head_predictions(model,head,manifest,'val',altered,'seg',job,
                                                                torch.device('cpu')))
            for rows in controls:
                self.assertEqual({r['position'] for r in rows},set(range(48,64)))
                self.assertTrue(all(r['full_context'] for r in rows))
                self.assertTrue(all(r['target_index']==r['prefix']+r['position'] for r in rows))
            for control in controls[1:]:
                pair = history.paired_delta(controls[0],control,'dice',repetitions=50)
                self.assertEqual(pair['patients'],1)


if __name__=='__main__':
    unittest.main()
