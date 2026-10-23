"""Validated experiment DAG with separate scientific and execution controls."""

from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml


DEFAULTS = {
    'version': 1,
    'run': {'root': '/root/autodl-tmp/outputs_cr', 'tag': None},
    'adapter': 'research_pipeline.echo:build',
    'data': {}, 'model': {},
    'runtime': {'device': 'cuda', 'autotune': True, 'micro_batch': 4,
                'batch_candidates': [1, 2, 4, 8, 16, 32, 64],
                'worker_candidates': [0, 2, 4, 8], 'checkpoint_candidates': [False, True],
                'precision': 'fp32', 'prefetch_factor': 2, 'threads': 2,
                'probe_warmup_updates': 1, 'probe_updates': 2, 'probe_timeout_seconds': 240,
                'memory_headroom_fraction': .1, 'memory_headroom_gb': .5,
                'heartbeat_seconds': 15, 'loader_timeout_seconds': 180,
                'cache_gb': 24, 'min_free_gb': 2},
    'training': {'seed': 42, 'epochs': 100, 'effective_batch': 32, 'lr': .0001,
                 'weight_decay': .05, 'warmup_epochs': 5, 'clip_norm': 1.,
                 'eval_every': 1, 'patience': 10, 'min_delta': 0.,
                 'checkpoint_every': 10, 'keep_snapshots': 2, 'save_epochs': [],
                 'tensorboard': True},
    'evaluation': {'boundaries': False, 'bootstrap_repetitions': 2000,
                   'representation_max_samples': 128, 'representation_interventions': False, 'cache_frozen': True},
    'stages': [],
}


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def safe_name(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError(f'Unsafe or missing run/stage name: {value!r}')
    return value


def stage_config(config, stage):
    result = copy.deepcopy(config)
    for key in ('data', 'model', 'runtime', 'training', 'evaluation'):
        result[key] = merge(config[key], stage.get(key, {}))
    result['stage'] = {k: v for k, v in stage.items() if k not in ('data', 'model', 'runtime', 'training', 'evaluation')}
    return result


def validate(config):
    if config['version'] != 1:
        raise ValueError('Unsupported pipeline version')
    safe_name(config['run']['tag'])
    known = set()
    for stage in config['stages']:
        name = safe_name(stage['id'])
        if name in known:
            raise ValueError(f'Duplicate stage: {name}')
        kind = stage.get('kind')
        if kind not in {'mae', 'ef', 'seg', 'representation', 'streaming', 'mechanisms'}:
            raise ValueError(f'Unsupported stage kind: {kind}')
        if kind == 'streaming' and set(stage.get('audit', {})) - {'max_frames', 'max_cases', 'atol', 'rtol'}:
            raise ValueError('Unsupported streaming audit override')
        if stage.get('mode', 'finetune') not in {'finetune', 'frozen_linear', 'frozen_lite'}:
            raise ValueError('Unknown head/training mode')
        source = stage.get('source')
        if isinstance(source, str) and source.startswith('@'):
            parts = source[1:].split('/')
            if len(parts) != 2 or parts[0] not in known or parts[1] not in {'final', 'best'}:
                raise ValueError('A source must name an earlier stage and final/best')
        cfg = stage_config(config, stage)
        train, run = cfg['training'], cfg['runtime']
        for key in ('epochs', 'effective_batch', 'eval_every', 'checkpoint_every'):
            if not isinstance(train[key], int) or train[key] < 1:
                raise ValueError(f'training.{key} must be a positive integer')
        if train['patience'] is not None and (not isinstance(train['patience'], int) or train['patience'] < 1):
            raise ValueError('patience must be a positive number of validation checks or null')
        if kind == 'mae' and (train['patience'] is not None or train['eval_every'] < 1):
            raise ValueError('MAE must set training.patience: null; its epoch budget is fixed')
        if not 0 <= train['warmup_epochs'] <= train['epochs']:
            raise ValueError('warmup_epochs must be between zero and epochs')
        if train['lr'] <= 0 or train['weight_decay'] < 0 or train['min_delta'] < 0:
            raise ValueError('Invalid optimizer/early-stopping settings')
        if run['precision'] not in {'fp32', 'fp16', 'bf16'}:
            raise ValueError('Use an explicit precision: fp32, fp16, bf16')
        if not 0 <= run['memory_headroom_fraction'] < 1 or min(run['cache_gb'], run['min_free_gb']) < 0:
            raise ValueError('Invalid memory/storage budget')
        if any(not isinstance(n, int) or n < 0 for n in run['worker_candidates']):
            raise ValueError('Worker candidates must be nonnegative integers')
        if not run['worker_candidates'] or not run['batch_candidates'] or not run['checkpoint_candidates']:
            raise ValueError('Runtime candidate lists cannot be empty')
        if any(not isinstance(n, int) or n < 1 for n in run['batch_candidates']):
            raise ValueError('Batch candidates must be positive integers')
        known.add(name)
    if not known:
        raise ValueError('No stages configured')
    return config


def load(path, overrides=(), tag=None, root=None, smoke=False):
    config = merge(DEFAULTS, yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {})
    for expression in overrides:
        key, sep, value = expression.partition('=')
        if not sep:
            raise ValueError('--set requires dotted.key=YAML_value')
        target = config
        inherited, stage_override = {}, False
        pieces = key.split('.')
        for part in pieces[:-1]:
            if isinstance(target, list):
                matches = [stage for stage in target if stage.get('id') == part]
                if len(matches) != 1:
                    raise ValueError(f'Unknown stage ID in path: {key}')
                target = matches[0]
                stage_override = True
                continue
            if part == 'stages' and isinstance(target.get(part), list):
                target = target[part]
                continue
            if stage_override and part in ('data', 'model', 'runtime', 'training', 'evaluation'):
                target.setdefault(part, {})
                inherited = config[part]
            elif inherited and isinstance(inherited.get(part), dict):
                inherited = inherited[part]
            if not isinstance(target.get(part), dict):
                raise ValueError(f'Unknown configuration path: {key}')
            target = target[part]
        if pieces[-1] not in target and pieces[-1] not in inherited:
            raise ValueError(f'Unknown configuration key: {key}')
        target[pieces[-1]] = yaml.safe_load(value)
    if tag:
        config['run']['tag'] = tag
    if root:
        config['run']['root'] = root
    if smoke:
        config['run']['tag'] = safe_name(config['run']['tag']) + '_smoke'
        config['smoke'] = True
        config['data'].update(limit_train=4, limit_val=2)
        config['runtime'].update(autotune=False, micro_batch=1, worker_candidates=[0])
        config['training'].update(epochs=1, warmup_epochs=0, effective_batch=2, tensorboard=False)
        for stage in config['stages']:
            stage.setdefault('training', {}).update(epochs=1, warmup_epochs=0, effective_batch=2)
            stage.setdefault('data', {}).update(limit_train=4, limit_val=2)
            if stage['kind'] == 'streaming':
                audit = stage.setdefault('audit', {})
                audit['max_frames'] = min(audit.get('max_frames', 512), 8*config['data'].get('local_frames', 16))
    return validate(config)
