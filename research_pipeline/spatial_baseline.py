"""Shared pipeline with calibrated full task heads and unchanged linear probes."""

from .echo import build as build_reference
from .readout_calibration import build as build_calibrated


def build(config, stage, source, device):
    calibrated = stage['kind'] in {'seg', 'ef'} and (
        stage.get('mode') == 'finetune' or (stage['kind'] == 'ef' and stage.get('calibrated_ef')))
    factory = build_calibrated if calibrated else build_reference
    exp = factory(config, stage, source, device)
    if exp.backbone.patch_size != 8 or exp.backbone.img_size != 112:
        raise ValueError('Spatial recipe requires native112 patch8 forward, not patch16 or resized images')
    exp.metadata['development_recipe'] = dict(
        spatial_grid=[14, 14], max_window_frames=64, extra_prefix_frames=0,
        interpretation='Spatial/window/schedule recipe bundle, not an isolated patch-size effect')
    return exp
