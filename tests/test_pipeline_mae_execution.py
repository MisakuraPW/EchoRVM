import copy
import unittest

import torch

from models.final_temporal_mae import FinalTemporalMAE
from models.forward_temporal_mae import ForwardTemporalMAE
from research_pipeline.mae_execution import active_clip_loss, supported
from utils.final_temporal_training import deterministic_masks


class ActiveClipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_losses_gradients_and_update_match_length_groups(self):
        for variant in ('baseline', 'mid_read', 'content_write'):
            for strategy in ('tube', 'complementary'):
                with self.subTest(variant=variant, strategy=strategy):
                    torch.manual_seed(71)
                    config = dict(img_size=16, patch_size=8, local_frames=4, clip_count=4,
                                  embed_dim=24, depth=2, num_heads=4, decoder_embed_dim=24,
                                  decoder_depth=1, decoder_num_heads=4, in_chans=3, tubelet_size=2,
                                  mask_ratio=.5, core_depth=1, memory_mode='global', frame_readout='repeat',
                                  reconstruction_recent_frames=8, norm_pix_loss=True, research_mask=strategy)
                    model = FinalTemporalMAE(**config) if variant == 'baseline' else ForwardTemporalMAE(
                        **config, forward_variant=variant, read_after_block=1)
                    fast = copy.deepcopy(model)
                    samples = [dict(video=torch.rand(t, 3, 16, 16), record_index=i+5)
                               for i, t in enumerate((12, 4, 8, 12))]
                    reference = None
                    for length in (12, 4, 8):
                        group = [s for s in samples if len(s['video']) == length]
                        video = torch.stack([s['video'] for s in group])
                        mask = deterministic_masks(model, dict(video=video, record_index=torch.tensor(
                            [s['record_index'] for s in group])), 42)
                        term = model(video, masks=mask)['loss']*len(group)/len(samples)
                        reference = term if reference is None else reference+term
                    actual = active_clip_loss(fast, samples, 42)
                    torch.testing.assert_close(actual, reference, rtol=2e-6, atol=2e-6)
                    reference.backward(); actual.backward()
                    for (name, p), (_, q) in zip(model.named_parameters(), fast.named_parameters()):
                        self.assertEqual(p.grad is None, q.grad is None, name)
                        if p.grad is not None:
                            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=3e-6, msg=name)
                    a = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    b = torch.optim.AdamW(fast.parameters(), lr=1e-4)
                    a.step(); b.step()
                    for name, p in model.named_parameters():
                        # Adam can amplify rounding in nearly zero bias gradients.
                        torch.testing.assert_close(p, dict(fast.named_parameters())[name], rtol=1e-4, atol=1e-5)

    def test_random_mask_and_stochastic_layers_use_reference(self):
        model = torch.nn.Sequential(torch.nn.Dropout(.1))
        self.assertFalse(supported(model))


if __name__ == '__main__':
    unittest.main()
