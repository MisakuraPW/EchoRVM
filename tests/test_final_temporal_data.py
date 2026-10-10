"""CPU, file-backed tests; set FINAL_TEMPORAL_DATA_ROOT for a real-data audit."""

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import pandas as pd
from skimage.draw import polygon
import torch
from torch.utils.data import DataLoader

from augment.ultrasound import EchoAugmentConfig
from echo_aug_validation.augment_recipes import _zoom_pair
from utils.downstream_datasets import _rasterize_echonet_trace
from utils.echo_input import read_echo_input
from utils.final_temporal_data import build_manifest, WindowDataset, WarmPlanDataset, WarmBatchSampler
from utils.stage3_diagnostics import intervention_video


POINTS = [[0, 0, 112, 112], [31.4, 23.7, 75.6, 24.4],
          [25.5, 55.6, 83.2, 54.7], [42.4, 87.5, 66.6, 88.2]]


def fixture(root):
    (root / 'npy').mkdir()
    rows, traces = [], []
    cases = [('short', 'TRAIN', 63, [30]), ('exact', 'TRAIN', 64, [48]),
             ('edge', 'TRAIN', 80, [62, 63, 64, 65]),
             ('long', 'TRAIN', 256, [127, 192]), ('middle', 'TRAIN', 112, [80]),
             ('valid', 'VAL', 192, [100, 150]), ('val_exact', 'VAL', 64, [50]),
             ('sealed', 'TEST', 200, [])]
    for name, split, n, frames in cases:
        # Known frame intensities make source-index and normalization errors observable.
        arr = np.broadcast_to((np.arange(n) % 255).astype(np.uint8)[:, None, None],
                              (n, 112, 112)).copy()
        if split != 'TEST':
            np.save(root / 'npy' / f'{name}.npy', arr)
        rows.append(dict(FileName=name, Split=split, FPS=40 if name == 'middle' else 50,
                         EF=55.25, NumberOfFrames=n + 3, FrameHeight=112, FrameWidth=112))
        for frame in frames:
            for x1, y1, x2, y2 in POINTS:
                traces.append(dict(FileName=name + '.avi', Frame=frame,
                                   X1=x1, Y1=y1, X2=x2, Y2=y2))
    pd.DataFrame(rows).to_csv(root / 'FileList.csv', index=False)
    pd.DataFrame(traces).to_csv(root / 'VolumeTracings.csv', index=False)


class FinalTemporalDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture(self.root)
        self.manifest = build_manifest(self.root)

    def test_manifest_json_provenance_actual_lengths_and_exclusions(self):
        self.assertEqual(self.manifest, build_manifest(self.root))
        self.assertEqual(json.loads(json.dumps(self.manifest, allow_nan=False)), self.manifest)
        self.assertEqual(self.manifest['version'], 2)
        self.assertFalse({c['patient'] for c in self.manifest['train']} &
                         {c['patient'] for c in self.manifest['val']})
        exact = next(c for c in self.manifest['train'] if c['patient'] == 'exact')
        self.assertEqual(exact['frames'], 64)
        self.assertEqual(exact['declared_frames'], 67)
        self.assertEqual(exact['shape'], [64, 112, 112])
        self.assertEqual(exact['dtype'], 'uint8')
        self.assertEqual(len(exact['source_fingerprint']), 64)
        self.assertEqual(len(self.manifest['provenance']['filelist']['sha256']), 64)
        self.assertEqual(self.manifest['counts']['train']['short_recent_cases'], 1)
        self.assertEqual(self.manifest['counts']['train']['warmup_excluded_traces'], 3)
        self.assertEqual(self.manifest['counts']['val']['warmup_excluded_traces'], 1)
        self.assertFalse(any(r['patient'] == 'sealed' for r in self.manifest['excluded']))
        edge = next(c for c in self.manifest['train'] if c['patient'] == 'edge')
        self.assertEqual([t['frame'] for t in edge['traces']], [62, 63, 64, 65])
        self.assertEqual(edge['traces'][0]['points'], POINTS)

    def test_hash_tampering_file_changes_and_split_leak_rejected(self):
        tampered = copy.deepcopy(self.manifest)
        tampered['train'][0]['ef'] += 1
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            WindowDataset(tampered, 'train')
        overlap = copy.deepcopy(self.manifest)
        overlap.pop('manifest_sha256')
        overlap['val'].append(copy.deepcopy(overlap['train'][0]))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            WindowDataset(overlap, 'train')
        ds = WindowDataset(self.manifest, 'train')
        case = ds.get_record(0)
        path = Path(case['path'])
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 10000000))
        with self.assertRaisesRegex(ValueError, 'Data changed'):
            ds[0]

    def test_missing_split_duplicate_invalid_gt_and_out_of_range_trace(self):
        path = self.root / 'FileList.csv'
        frame = pd.read_csv(path)
        frame.drop(columns='Split').to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'requires'):
            build_manifest(self.root)
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            build_manifest(self.root)
        frame.to_csv(path, index=False)
        np.save(self.root / 'npy' / 'exact.npy', np.zeros((64, 56, 56), dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, 'GT coordinate mismatch'):
            build_manifest(self.root)
        np.save(self.root / 'npy' / 'exact.npy', np.zeros((64, 112, 112), dtype=np.uint8))
        tracefile = self.root / 'VolumeTracings.csv'
        trace = pd.read_csv(tracefile)
        trace.loc[trace.FileName == 'exact.avi', 'Frame'] = 64
        trace.to_csv(tracefile, index=False)
        with self.assertRaisesRegex(ValueError, 'outside actual'):
            build_manifest(self.root)
        trace.loc[trace.FileName == 'exact.avi', 'Frame'] = 48
        trace.loc[0, 'X1'] = np.nan
        trace.to_csv(tracefile, index=False)
        with self.assertRaisesRegex(ValueError, 'coordinates must be finite'):
            build_manifest(self.root)
        trace.loc[0, 'X1'] = 0
        trace.to_csv(tracefile, index=False)
        frame.loc[frame.FileName == 'exact', 'FrameWidth'] = 224
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, 'GT coordinate mismatch'):
            build_manifest(self.root)

    def test_official_out_of_canvas_vertices_are_clipped_not_changed(self):
        tracefile = self.root / 'VolumeTracings.csv'
        traces = pd.read_csv(tracefile)
        target = (traces.FileName == 'long.avi') & (traces.Frame == 127)
        points = np.asarray(POINTS, dtype=float)
        points[1, :2] = [-2.32, -3.73]
        points[2, 2:] = [119.88, 116.08]
        traces.loc[target, ['X1', 'Y1', 'X2', 'Y2']] = points
        traces.to_csv(tracefile, index=False)
        manifest = build_manifest(self.root)
        ds = WindowDataset(manifest, 'train', task='seg', records=[
            dict(patient='long', recent_start=64, H=32, target_frame=127)])
        expected = _rasterize_echonet_trace(traces[target], (112, 112))
        np.testing.assert_array_equal(ds[0]['mask'].numpy(), expected)
        self.assertEqual(next(c for c in manifest['train'] if c['patient'] == 'long')['traces'][0]['points'],
                         points.tolist())

    def test_ef_mae_real_indices_and_low_uint8_normalization(self):
        ds = WindowDataset(self.manifest, 'train', task='ef', prefix=None)
        for i in range(len(ds)):
            sample = ds[i]
            idx = sample['frame_indices']
            h = sample['prefix_frames']
            self.assertEqual(sample['video'].shape, (64 + h, 3, 112, 112))
            self.assertTrue(torch.all(idx[1:] - idx[:-1] == 1))
            torch.testing.assert_close(sample['video'][:, 0, 0, 0], idx.float().remainder(255) / 255)
            torch.testing.assert_close(sample['video'][:, 0], sample['video'][:, 1])
            self.assertEqual(float(sample['target']), 55.25)
            self.assertEqual(sample['target_index'], -1)
            self.assertTrue(sample['full_context'])
            self.assertEqual(sample['recent_start'], int(idx[h]))
        # Even a very dark integer cache must divide by 255, not by max_seen.
        np.save(self.root / 'npy' / 'exact.npy', np.ones((64, 112, 112), dtype=np.uint8))
        manifest = build_manifest(self.root)
        ds = WindowDataset(manifest, 'train', task='mae', records=[dict(patient='exact', recent_start=0, H=0)])
        self.assertAlmostEqual(float(ds[0]['video'].max()), 1 / 255)
        self.assertNotIn('target', ds[0])

    def test_seg_all_positions_is_original_frame_intersection_and_exact_mask(self):
        ds = WindowDataset(self.manifest, 'train', task='seg', positions='all')
        edge = [i for i, r in enumerate(ds.records) if r['patient'] == 'edge']
        self.assertEqual(len(edge), 32)
        self.assertEqual({ds.records[i]['target_frame'] for i in edge}, {63, 64})
        expected = np.zeros((112, 112), dtype=np.uint8)
        points = np.asarray(POINTS, dtype=np.float32)
        x = np.r_[points[1:, 0], points[1:, 2][::-1]]
        y = np.r_[points[1:, 1], points[1:, 3][::-1]]
        r, c = polygon(np.rint(y).astype(int), np.rint(x).astype(int), expected.shape)
        expected[r, c] = 1
        self.assertEqual(expected[0, 0], 0)
        for i in edge:
            sample = ds[i]
            target = sample['target_index']
            self.assertEqual(sample['target_recent_index'], sample['target_position'])
            self.assertEqual(int(sample['frame_indices'][target]), sample['target_frame'])
            self.assertIn(sample['target_position'], range(48, 64))
            np.testing.assert_array_equal(sample['mask'].numpy(), expected)
            self.assertEqual(sample['mask'].dtype, torch.int64)
        # f=63 is valid at every W64 position, but not with H16 at every position.
        prefixed = WindowDataset(self.manifest, 'train', task='seg', positions='all', prefix=16)
        self.assertFalse(any(r['patient'] == 'edge' for r in prefixed.records))
        self.assertEqual({r['patient'] for r in prefixed.records}, {'long', 'middle'})
        self.assertTrue(all(prefixed[i]['target_index'] == 16 + prefixed[i]['target_recent_index']
                            for i in range(len(prefixed))))

    def test_balanced_positions_rotate_by_epoch_and_eval_stays_fixed(self):
        ds = WindowDataset(self.manifest, 'train', task='seg', training=True)
        coverage = [set() for _ in range(len(ds))]
        for epoch in range(16):
            ds.set_epoch(epoch)
            for i in range(len(ds)):
                record = ds.get_record(i)
                coverage[i].add(record['target_position'])
                self.assertEqual(record['start'] + record['target_position'], record['target_frame'])
        self.assertTrue(all(c == set(range(48, 64)) for c in coverage))
        ds.set_epoch(5)
        first = [ds.get_record(i) for i in range(len(ds))]
        ds.set_epoch(5)
        self.assertEqual(first, [ds.get_record(i) for i in range(len(ds))])
        val = WindowDataset(self.manifest, 'val', task='seg', positions='all')
        first = [val.get_record(i) for i in range(len(val))]
        val.set_epoch(99)
        self.assertEqual(first, [val.get_record(i) for i in range(len(val))])
        self.assertEqual(len(val), 32)
        limited = WindowDataset(self.manifest, 'val', task='seg', positions='all', limit=1)
        self.assertIn(len(limited), (0, 32))

    def test_explicit_same_window_h0_hh_hrh_matches_existing_intervention(self):
        records = [dict(patient='long', recent_start=80, H=h, prefix_kind=kind)
                   for h, kind in ((0, 'clean'), (64, 'clean'), (64, 'repeat_prefix'))]
        ds = WindowDataset(self.manifest, 'train', task='ef', records=records,
                           prefix=None, training=True)
        samples = [ds[i] for i in range(3)]
        for sample in samples:
            self.assertEqual(sample['recent_start'], 80)
            np.testing.assert_array_equal(sample['frame_indices'][-64:].numpy(), np.arange(80, 144))
            torch.testing.assert_close(sample['video'][-64:], samples[0]['video'])
        expected = intervention_video(samples[1]['video'][None], 64, 64, 'repeat_prefix', 16)[0]
        torch.testing.assert_close(samples[2]['video'], expected)
        np.testing.assert_array_equal(samples[2]['frame_indices'][:64].numpy(), np.tile(np.arange(16, 32), 4))
        self.assertEqual(samples[2]['prefix_kind'], 'repeat_prefix')
        ds.set_epoch(17)
        self.assertEqual(ds.get_record(2)['recent_start'], 80)
        seg = WindowDataset(self.manifest, 'train', task='seg', records=[
            dict(patient='long', recent_start=64, H=32, target_frame=127)])
        sample = seg[0]
        self.assertEqual(sample['target_recent_index'], 63)
        self.assertEqual(sample['target_index'], 95)
        self.assertEqual(int(sample['frame_indices'][95]), 127)

    def test_explicit_records_reject_fake_history_and_label_or_split_changes(self):
        invalid = [dict(patient='exact', recent_start=0, H=16),
                   dict(patient='long', recent_start=12, H=32),
                   dict(patient='long', recent_start=80, H=17),
                   dict(patient='long', recent_start=240, H=0),
                   dict(patient='valid', recent_start=80, H=0),
                   dict(patient='long', recent_start=80, H=0, prefix_kind='repeat_prefix'),
                   dict(patient='long', recent_start=80, start=64, H=0)]
        for record in invalid:
            with self.subTest(record=record), self.assertRaises(ValueError):
                WindowDataset(self.manifest, 'train', records=[record])
        for record in [dict(patient='edge', recent_start=14, H=0, target_frame=62),
                       dict(patient='long', recent_start=64, H=0, target_frame=126)]:
            with self.subTest(record=record), self.assertRaises(ValueError):
                WindowDataset(self.manifest, 'train', task='seg', records=[record])

    def test_a4_zoom_shared_geometry_and_float_no_double_normalization(self):
        frame = pd.DataFrame(POINTS, columns=['X1', 'Y1', 'X2', 'Y2'])
        mask = _rasterize_echonet_trace(frame, (112, 112))
        np.save(self.root / 'npy' / 'long.npy', np.repeat(mask[None], 256, axis=0).astype(np.float32))
        manifest = build_manifest(self.root)
        cfg = EchoAugmentConfig(tgc_prob=0, gamma_contrast_prob=0, brightness_prob=0,
                                zoom_prob=1, blur_prob=0, shadow_prob=0, speckle_prob=0,
                                zoom_min=1.12, zoom_max=1.12, preserve_dtype=False)
        records = [dict(patient='long', recent_start=64, H=32, target_frame=127)]
        ds = WindowDataset(manifest, 'train', task='seg', records=records, aug_cfg=cfg, training=True)
        sample = ds[0]
        image, expected_mask = _zoom_pair(mask.astype(np.float32), mask, 1.12)
        torch.testing.assert_close(sample['video'][:, 0], torch.from_numpy(image)[None].repeat(96, 1, 1))
        np.testing.assert_array_equal(sample['mask'].numpy(), expected_mask)
        self.assertEqual(float(sample['video'].max()), 1)
        torch.testing.assert_close(ds[0]['video'], sample['video'])
        # Zoom-out geometry and uint8/float255 are normalized exactly once too.
        for dtype, scale in [(np.uint8, 255), (np.float32, 255), (np.float32, 1)]:
            np.save(self.root / 'npy' / 'long.npy', np.repeat(mask[None], 256, axis=0).astype(dtype) * scale)
            ds = WindowDataset(build_manifest(self.root), 'train', task='seg', records=records,
                               aug_cfg=replace(cfg, zoom_min=.92, zoom_max=.92), training=True)
            image, expected_mask = _zoom_pair(mask.astype(np.float32), mask, .92)
            torch.testing.assert_close(ds[0]['video'][32, 0], torch.from_numpy(image))
            np.testing.assert_array_equal(ds[0]['mask'].numpy(), expected_mask)
        val = WindowDataset(build_manifest(self.root), 'train', task='seg', records=records,
                            aug_cfg=cfg, training=False)
        np.testing.assert_array_equal(val[0]['mask'].numpy(), mask)

    def test_rgb_and_single_channel_npy_layouts_and_nonfinite_pixels(self):
        gray = np.arange(64, dtype=np.uint8)[:, None, None]
        rgb = np.zeros((64, 112, 112, 3), dtype=np.uint8)
        rgb[..., 0] = gray
        rgb[..., 1] = 90
        rgb[..., 2] = 180
        path = self.root / 'npy' / 'exact.npy'
        np.save(path, rgb)
        ds = WindowDataset(build_manifest(self.root), 'train', records=[dict(patient='exact', recent_start=0, H=0)])
        sample = ds[0]
        expected = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in rgb])
        torch.testing.assert_close(sample['video'][:, 0], torch.from_numpy(expected.astype(np.float32) / 255))
        np.save(path, expected[..., None])
        ds = WindowDataset(build_manifest(self.root), 'train', records=[dict(patient='exact', recent_start=0, H=0)])
        self.assertEqual(ds[0]['video'].shape, (64, 3, 112, 112))
        invalid = expected.astype(np.float32)
        invalid[0, 0, 0] = np.nan
        np.save(path, invalid)
        ds = WindowDataset(build_manifest(self.root), 'train', records=[dict(patient='exact', recent_start=0, H=0)])
        with self.assertRaisesRegex(ValueError, 'finite'):
            ds[0]

    def test_warm_plan_all_w64_patients_and_available_h(self):
        ds = WarmPlanDataset(self.manifest, total_updates=100, effective_batch=32)
        self.assertEqual(len(ds), 3200)
        patients, lengths = set(), set()
        for i in range(len(ds)):
            record = ds.get_record(i)
            patients.add(record['patient'])
            lengths.add(record['H'])
            self.assertEqual(record['H'] % 16, 0)
            self.assertLessEqual(record['H'], min(128, record['start']))
            self.assertLessEqual(record['start'] + 64, record['frames'])
            self.assertEqual(record['update_index'], i // 32)
            self.assertEqual(record['sample_index'], i % 32)
            if record['patient'] == 'exact':
                self.assertEqual((record['start'], record['H']), (0, 0))
        self.assertEqual(patients, {c['patient'] for c in self.manifest['train']})
        self.assertEqual(lengths, set(range(0, 129, 16)))
        other = WarmPlanDataset(json.loads(json.dumps(self.manifest)), total_updates=100)
        for i in (0, 31, 32, 1059, 3199):
            self.assertEqual(ds.get_record(i), other.get_record(i))
        sample = ds[35]
        self.assertEqual(sample['update_index'], 1)
        self.assertEqual(sample['sample_index'], 3)
        self.assertNotIn('target', sample)

    def test_sampler_exact_update_buckets_no_padding_and_resume(self):
        ds = WarmPlanDataset(self.manifest, total_updates=7, effective_batch=19)
        reference = None
        for micro in (1, 3, 8, 64):
            sampler = WarmBatchSampler(ds, micro)
            batches = list(sampler)
            self.assertEqual(len(sampler), len(batches))
            draws = {}
            per_update = {}
            for batch in batches:
                self.assertTrue(0 < len(batch) <= micro)
                records = [ds.get_record(i) for i in batch]
                self.assertEqual(len({r['H'] for r in records}), 1)
                self.assertEqual(len({r['update_index'] for r in records}), 1)
                update = records[0]['update_index']
                per_update.setdefault(update, []).extend(batch)
                for i, r in zip(batch, records):
                    draws[i] = (r['patient'], r['recent_start'], r['H'])
            for update, indices in per_update.items():
                self.assertEqual(sorted(indices), list(range(update * 19, (update + 1) * 19)))
            self.assertEqual(len(draws), len(ds))
            if reference is None:
                reference = draws
            self.assertEqual(draws, reference)
            resumed = list(WarmBatchSampler(ds, micro, start_update=3))
            self.assertEqual(resumed, [b for b in batches if b[0] // 19 >= 3])
        self.assertEqual(list(WarmBatchSampler(ds, 4, start_update=7)), [])
        ds.set_epoch(99)
        self.assertEqual(ds.get_record(12), WarmPlanDataset(self.manifest, total_updates=7,
                                                          effective_batch=19).get_record(12))

    def test_file_backed_dataloader_collation_and_json_manifest_path(self):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(self.manifest), encoding='utf-8')
        ds = WarmPlanDataset(path, total_updates=2, effective_batch=7)
        loader = DataLoader(ds, batch_sampler=WarmBatchSampler(ds, micro_batch=3), num_workers=0)
        count = 0
        for batch in loader:
            self.assertEqual(len(set(batch['prefix_frames'].tolist())), 1)
            self.assertEqual(len(set(batch['update_index'].tolist())), 1)
            self.assertEqual(batch['video'].dtype, torch.float32)
            count += len(batch['id'])
        self.assertEqual(count, 14)

    def test_persistent_worker_observes_epoch_and_preserves_source_target(self):
        ds = WindowDataset(self.manifest, 'val', task='seg', training=True, prefix=16)
        loader = DataLoader(ds, batch_size=1, num_workers=1, persistent_workers=True)
        try:
            ds.set_epoch(0)
            first = next(iter(loader))
            ds.set_epoch(1)
            second = next(iter(loader))
            self.assertNotEqual(first['target_recent_index'].item(), second['target_recent_index'].item())
            self.assertEqual(first['target_frame'].item(), second['target_frame'].item())
            for batch in (first, second):
                target = batch['target_index'].item()
                self.assertEqual(target, 16 + batch['target_recent_index'].item())
                self.assertEqual(batch['frame_indices'][0, target].item(), batch['target_frame'].item())
        finally:
            if loader._iterator is not None:
                loader._iterator._shutdown_workers()

    def test_unlabelled_mae_eligible_and_missing_sources_never_fabricated(self):
        path = self.root / 'FileList.csv'
        rows = pd.read_csv(path)
        rows.loc[rows.FileName == 'exact', 'EF'] = np.nan
        rows.to_csv(path, index=False)
        traces = pd.read_csv(self.root / 'VolumeTracings.csv')
        traces[traces.FileName != 'exact.avi'].to_csv(self.root / 'VolumeTracings.csv', index=False)
        manifest = build_manifest(self.root)
        self.assertTrue(any(c['patient'] == 'exact' for c in WarmPlanDataset(manifest).eligible_cases))
        self.assertFalse(any(r['patient'] == 'exact' for r in WindowDataset(manifest, 'train').records))
        (self.root / 'npy' / 'middle.npy').unlink()
        manifest = build_manifest(self.root)
        self.assertTrue(any(r['patient'] == 'middle' and r['reason'] == 'missing_video'
                            for r in manifest['excluded']))
        self.assertFalse(any(c['patient'] == 'middle' for c in WarmPlanDataset(manifest).eligible_cases))

    def test_argument_validation_and_smoke_are_explicit(self):
        smoke = build_manifest(self.root, smoke=True)
        self.assertTrue(smoke['smoke'])
        self.assertLessEqual(len(smoke['train']), 8)
        for kw in (dict(recent_frames=63), dict(max_prefix=127), dict(local_frames=0),
                   dict(recent_frames=64.5)):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                build_manifest(self.root, **kw)
        for kw in (dict(prefix=1), dict(positions='random'), dict(task='other')):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                WindowDataset(self.manifest, 'train', **kw)
        with self.assertRaises(ValueError):
            WindowDataset(self.manifest, 'test')
        for micro, cursor in ((0, 0), (1, -1), (1, 3)):
            with self.assertRaises(ValueError):
                WarmBatchSampler(WarmPlanDataset(self.manifest, total_updates=2), micro, cursor)


class RealEchoNetAudit(unittest.TestCase):
    def test_real_original_video_and_npy_cache_gt_equivalence(self):
        root = Path(os.environ.get('FINAL_TEMPORAL_DATA_ROOT', 'G:/SRTP/dataset/EchoNet-Dynamic'))
        if not (root / 'FileList.csv').is_file():
            self.skipTest('Set FINAL_TEMPORAL_DATA_ROOT for real-file audit')
        files = pd.read_csv(root / 'FileList.csv')
        traces = pd.read_csv(root / 'VolumeTracings.csv')
        from echo_aug_validation.io_utils import find_echonet_video
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache / 'npy').mkdir()
            selected = []
            for split in ('TRAIN', 'VAL'):
                for _, row in files[files.Split == split].iterrows():
                    name = Path(str(row.FileName)).stem
                    targets = traces.loc[traces.FileName == name + '.avi', 'Frame'].unique()
                    if not any(f >= 63 and f + 16 <= row.NumberOfFrames for f in targets):
                        continue
                    path = find_echonet_video(root, name)
                    self.assertIsNotNone(path)
                    raw = read_echo_input(path, 'gray_repeat3')
                    self.assertEqual(raw.shape[1:], (112, 112))
                    if not any(f >= 63 and f + 16 <= len(raw) for f in targets):
                        continue
                    np.save(cache / 'npy' / f'{name}.npy', raw)
                    selected.append(row)
                    break
                else:
                    self.fail(f'No real complete segmentation case in {split}')
            pd.DataFrame(selected).to_csv(cache / 'FileList.csv', index=False)
            names = {Path(str(r.FileName)).stem for r in selected}
            chosen = traces[traces.FileName.map(lambda x: Path(str(x)).stem).isin(names)]
            chosen.to_csv(cache / 'VolumeTracings.csv', index=False)
            manifest = build_manifest(cache)
            total = 0
            for split in ('train', 'val'):
                ds = WindowDataset(manifest, split, task='seg', positions='all')
                for i in range(len(ds)):
                    sample = ds[i]
                    group = chosen[(chosen.FileName == sample['patient'] + '.avi') &
                                   (chosen.Frame == sample['target_frame'])]
                    np.testing.assert_array_equal(sample['mask'].numpy(),
                                                  _rasterize_echonet_trace(group, (112, 112)))
                    self.assertEqual(int(sample['frame_indices'][sample['target_index']]), sample['target_frame'])
                    total += 1
            self.assertGreater(total, 0)


if __name__ == '__main__':
    unittest.main()
