import unittest

import torch

from models.temporal_mae import TemporalMAE
from tools.audit_stage3_streaming import inspect_stream, position_summary


class StreamingAuditTests(unittest.TestCase):
    def test_stream_contract_and_tubelet_identity(self):
        torch.set_num_threads(2)
        for mode in ('none', 'spatial'):
            model = TemporalMAE(img_size=16, patch_size=4, local_frames=4, clip_count=2,
                tubelet_size=2, in_chans=3, embed_dim=24, depth=1, num_heads=3,
                decoder_embed_dim=24, decoder_depth=1, decoder_num_heads=3,
                mask_ratio=.5, memory_mode=mode, memory_grid=2, core_depth=1,
                norm_pix_loss=False, position_embedding='flat_sinusoid').eval()
            row = inspect_stream(model, torch.rand(1,32,3,16,16), 'case', 50.)
            self.assertEqual(row['frames'], 32)
            self.assertEqual(row['max_acquisition_wait_seconds'], .06)
            self.assertEqual(max(row['adjacent_same_tubelet_max_abs']), 0.)
            self.assertGreater(row['persistent_bytes'], 0)

    def test_position_comparison_excludes_incomplete_source_across_all_positions(self):
        rows = [dict(patient='a', source_frame=20, target_index=p, full_context=True, dice=.8)
                for p in (48,49)]
        rows += [dict(patient='b', source_frame=25, target_index=p, full_context=p==48, dice=.2)
                 for p in (48,49)]
        result = position_summary(rows)
        self.assertTrue(all(r['patients']==1 and r['dice_patient_mean']==.8 for r in result))


if __name__ == '__main__':
    unittest.main()
