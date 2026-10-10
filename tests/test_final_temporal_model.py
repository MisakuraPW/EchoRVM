import tempfile
from pathlib import Path
import unittest

import torch

from models.temporal_mae import TemporalMAE
from models.final_temporal_mae import FinalTemporalMAE, load_final_model
from utils.streaming_features import StreamingFeatureCache
from utils.final_temporal_execution import encode_window


def tiny_config(**changes):
    value = dict(name='temporal_mae', img_size=16, patch_size=4, local_frames=4, clip_count=2,
                 frames=8, embed_dim=24, depth=1, num_heads=4, decoder_embed_dim=24,
                 decoder_depth=1, decoder_num_heads=4, tubelet_size=2, in_chans=3,
                 memory_grid=2, core_depth=1, memory_mode='spatial', memory_compression='temporal_attention',
                 frame_readout='learned', mask_ratio=.5, norm_pix_loss=False, reconstruction_recent_frames=8)
    value.update(changes)
    return value


class FinalTemporalModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_legacy_learned_and_factorized_are_compatible(self):
        for readout in ('repeat', 'learned', 'factorized'):
            cfg = tiny_config(frame_readout=readout, dynamic_rank=4)
            old = TemporalMAE(**cfg).eval()
            new = FinalTemporalMAE(**cfg).eval(); new.load_state_dict(old.state_dict())
            video = torch.rand(2, 8, 3, 16, 16)
            mask = torch.zeros(2, 2, 32, dtype=torch.bool); mask[:, :, ::2] = True
            with torch.no_grad():
                torch.testing.assert_close(old(video, masks=mask)['pred'], new(video, masks=mask)['pred'], rtol=1e-5, atol=1e-6)
                expected = old.frame_features(old.diagnostic_features(video)['features'])
                torch.testing.assert_close(expected, new.diagnostic_features(video)['frame_outputs'], rtol=1e-5, atol=1e-6)

    def test_suffix_gradient_and_hidden_pixel_isolation(self):
        for mode in ('global', 'spatial', 'spatial_global'):
            for read in ('tokens', 'frames'):
                model = FinalTemporalMAE(**tiny_config(memory_mode=mode, frame_readout='soft_factorized',
                         dynamic_rank=4, memory_read_location=read, memory_write_source='local' if read == 'frames' else 'fused'))
                video = torch.rand(1, 12, 3, 16, 16, requires_grad=True)
                mask = torch.zeros(1, 3, 32, dtype=torch.bool); mask[:, :, ::2] = True
                result = model(video, masks=mask)
                self.assertEqual(result['pred'].shape[1], 64)
                error = (result['pred'] - result['target'].detach()).square().mean(-1)
                loss = (error * result['mask']).sum() / result['mask'].sum()
                loss.backward()
                self.assertGreater(float(video.grad[:, :4].abs().sum()), 0.)
                hidden = mask.reshape(1, 6, 4, 4).repeat_interleave(2, 1).repeat_interleave(4, 2).repeat_interleave(4, 3)[:, :, None]
                with torch.no_grad():
                    changed = torch.where(hidden, video + 20, video)
                    torch.testing.assert_close(result['pred'], model(changed, masks=mask)['pred'], rtol=1e-5, atol=1e-6)

    def test_all_mechanisms_stream_and_frame_index(self):
        for mode, slots in (('global', 1), ('spatial', 4), ('spatial_global', 5)):
            for frame in ('learned', 'shrink', 'factorized', 'soft_factorized'):
                model = FinalTemporalMAE(**tiny_config(memory_mode=mode, frame_readout=frame, dynamic_rank=4,
                                       candidate_rank=4, memory_read_location='frames', memory_write_source='local')).eval()
                video = torch.rand(1, 24, 3, 16, 16)
                stream = StreamingFeatureCache(model, 2)
                outputs = []
                for start in range(0, 24, 4):
                    result = stream.update(video[:, start:start+4], ['p'], torch.arange(start, start+4)[None])
                    outputs.append(result['frame_features'])
                    self.assertEqual(result['final_state'].shape, (1, slots, 24))
                    self.assertLessEqual(result['fused'].shape[1], 8)
                with torch.no_grad():
                    all_frames = model.diagnostic_features(video)['frame_outputs']
                    torch.testing.assert_close(torch.cat(outputs, 1), all_frames)
                    task = encode_window(model, video, 16, 8, torch.tensor([22]))
                    torch.testing.assert_close(task['maps']['final'], all_frames[:, 22])
                    self.assertEqual(task['sequences']['final'].shape, (1, 8, 24))

    def test_means_soft_shrink_and_expansion_not_doubled(self):
        values = torch.rand(2, 4, 16, 24)
        hard = FinalTemporalMAE(**tiny_config(frame_readout='factorized', dynamic_rank=4))
        soft = FinalTemporalMAE(**tiny_config(frame_readout='soft_factorized', dynamic_rank=4))
        soft.load_state_dict(hard.state_dict())
        h = hard.frame_base_features(values)
        f = hard.frame_transform(h)
        torch.testing.assert_close(h.mean(1), f.mean(1))
        torch.testing.assert_close(soft.frame_transform(h), h + .5 * (f - h))
        shrink = FinalTemporalMAE(**tiny_config(frame_readout='shrink'))
        torch.testing.assert_close(shrink.frame_transform(h), h)
        self.assertEqual(f.shape[1], 8)

    def test_source_loading_initializes_only_registered_new_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'source.pt'
            cfg = tiny_config(); source = TemporalMAE(**cfg)
            torch.save(dict(model_state_dict=source.state_dict(), config=dict(model=cfg), epoch=100), path)
            for overrides in (dict(memory_mode='global'), dict(memory_mode='spatial_global'),
                              dict(frame_readout='factorized', dynamic_rank=4), dict(frame_readout='shrink'), dict(candidate_rank=4)):
                model, _, report = load_final_model(path, overrides)
                torch.testing.assert_close(source.patch_embed.proj.weight, model.patch_embed.proj.weight)
                self.assertFalse(report['discarded_keys'])
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_final_model(path, dict(embed_dim=48))

    def test_generic_backbone_keeps_variable_real_context(self):
        from models.downstream import EchoVideoMAEBackbone
        model = FinalTemporalMAE(**tiny_config()).eval()
        backbone = EchoVideoMAEBackbone(model)
        with torch.no_grad():
            for length in (4, 8, 12, 20):
                value = backbone.forward_tokens(torch.rand(1, length, 1, 16, 16))['outputs']
                self.assertEqual(value.shape[1], length)
            with self.assertRaises(ValueError):
                backbone.forward_tokens(torch.rand(1, 1, 1, 16, 16))

    def test_legacy_omitted_readout_and_memory_defaults_load_exactly(self):
        with tempfile.TemporaryDirectory() as directory:
            for default_memory in (False, True):
                cfg = tiny_config()
                cfg.pop('frame_readout')
                if default_memory:
                    cfg.pop('memory_mode')
                    cfg['memory_compression'] = 'mean'
                old = TemporalMAE(**cfg).eval()
                self.assertEqual(old.frame_readout, 'repeat')
                direct = FinalTemporalMAE(**cfg).eval()
                direct.load_state_dict(old.state_dict(), strict=True)
                path = Path(directory) / ('legacy_default_' + str(default_memory) + '.pt')
                torch.save(dict(model_state_dict=old.state_dict(),config=dict(model=cfg),epoch=100),path)
                restored, saved_config, report = load_final_model(path)
                restored.eval()
                self.assertEqual(restored.frame_readout, 'repeat')
                self.assertEqual(restored.memory_mode, old.memory_mode)
                self.assertEqual(saved_config['model']['frame_readout'], 'repeat')
                self.assertEqual(report['initialized_keys'], [])
                self.assertEqual(report['discarded_keys'], [])
                self.assertFalse(any(key.startswith('frame_expansion.') for key in restored.state_dict()))
                video = torch.rand(1,8,3,16,16)
                mask = torch.zeros(1,2,32,dtype=torch.bool); mask[:,:,::2]=True
                with torch.no_grad():
                    torch.testing.assert_close(old(video,masks=mask)['pred'], restored(video,masks=mask)['pred'],rtol=1e-5,atol=1e-6)
                    torch.testing.assert_close(old.frame_features(old.diagnostic_features(video)['features']),
                        restored.diagnostic_features(video)['frame_outputs'],rtol=1e-5,atol=1e-6)


if __name__ == '__main__':
    unittest.main()
