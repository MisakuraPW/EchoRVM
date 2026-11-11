"""Fixed-L16 Stage A adapter; the public engine still owns all training."""

from . import coverage_repair, spatial_baseline


def build(config, stage, source, device):
    factory = coverage_repair.build if stage['kind'] == 'seg' else spatial_baseline.build
    exp = factory(config, stage, source, device)
    model = exp.backbone
    if (model.local_frames, model.tubelet_size, model.memory_slots, model.frame_readout) != (16, 2, 1, 'repeat'):
        raise ValueError('Stage A requires L16/tubelet2/global/repeat')
    if not config.get('smoke'):
        for key,value in stage.get('matched_inputs',{}).items():
            if exp.metadata[key] != value:
                raise ValueError(f'Stage A input changed before fitting: {key}')
    if stage['kind'] == 'mae':
        from .artifacts import file_hash

        if file_hash(source) != config['stage_a_baseline']['epoch0']:
            raise ValueError('B8 epoch-0 checkpoint changed')
        init = exp.metadata['initialization']
        variant = getattr(model, 'forward_variant', None)
        added = set(init.get('initialized_keys', ()))
        writer = {k for k in model.state_dict() if k.startswith('content_writer.')}
        expected = writer if variant == 'content_write' else set()
        if (variant not in {'mid_read', 'content_write'} or init.get('source_epoch') != 0
                or added != expected or init.get('discarded_keys')
                or init['loaded_tensors'] != len(model.state_dict()) - len(expected)):
            raise ValueError('Stage A requires exact B8 epoch-0 shared initialization')
        if variant == 'mid_read' and model.read_after_block != 6:
            raise ValueError('A-R has one predeclared insertion after block 6')
        if variant == 'content_write' and model.content_writer.score.weight.detach().count_nonzero():
            raise ValueError('A-W must start with uniform attention')
    return exp


def evaluation_dataset(cfg, manifest, split, task):
    if task == 'seg':
        return coverage_repair.evaluation_dataset(cfg, manifest, split, task)
    if task != 'ef' or split != 'test':
        raise ValueError('Only the fixed-window EF TEST protocol is supported')
    from utils.final_temporal_data import WindowDataset

    data = cfg['data']
    return WindowDataset(manifest, split, task='ef', allow_test=True, recent_frames=64,
                         local_frames=16, max_prefix=0, prefix=0, seed=data['seed'])
