"""Execute the approved finite Q1-Q3 study. Q4 and formal long training remain pending."""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.final_temporal_training import digest, file_digest, write_json
from utils.final_temporal_data import build_manifest, _hash, _complete_trace
from utils.final_temporal_preflight import estimate_budget
from utils.final_temporal_selection import select_reference, mixed_trigger, soft_trigger, write_final_report, task_vector
from tools.run_dynamic_refinement import queue_lock
from tools.run_temporal_research import run_command, archive_analysis


DEFAULT_SOURCES = dict(
    P='/root/autodl-tmp/outputs_stage3/stage3_mechanisms_20260929/ckpt/temporal_pool/epoch_0100.pt',
    C='/root/autodl-tmp/outputs_dynamic/dynamic_refinement_20261008/ckpt/combined/epoch_0100.pt',
    F='/root/autodl-tmp/outputs_dynamic/dynamic_refinement_20261008/ckpt/factorized/epoch_0100.pt')
QUESTION_IDS = tuple(f'Q{i}.{j}' for i in (1, 2, 3) for j in (1, 2, 3, 4))
LEGACY_CACHE_EXECUTION = {
    'utils/final_temporal_tasks.py': '83391f6467c01f79dfc235391fb71fa6b23e3bd764cf886b53f3ba0a39ea1b66',
    'tools/run_temporal_final.py': 'eaad03a58918eb8ddb29897a2af1e4a6193f31d036b3beb036d56366e71c8bf1'}
VALIDATION_CACHE_TASK_SHA256 = '0b242e80117ded82f8d8f3703dabe2c0f863c57772373acfaf8d39f279b14502'


def scientific_code():
    paths = ['models/final_temporal_mae.py', 'models/temporal_mae.py', 'models/frame_readout.py',
             'models/rvm_core.py', 'models/video_mae.py', 'models/vit_blocks.py',
             'utils/final_temporal_data.py', 'utils/final_temporal_execution.py',
             'utils/final_temporal_training.py', 'utils/final_temporal_tasks.py',
             'utils/final_temporal_history.py', 'utils/final_temporal_representation.py',
             'utils/final_temporal_mechanisms.py', 'utils/final_temporal_selection.py',
             'utils/final_temporal_streaming.py',
             'utils/final_temporal_preflight.py', 'tools/run_temporal_final.py']
    paths.append('tools/final_temporal_worker.py')
    paths.extend(['models/downstream.py', 'models/ef_readout.py', 'utils/seed.py',
                  'utils/checkpoint.py', 'utils/streaming_features.py', 'utils/downstream_datasets.py',
                  'utils/echo_input.py', 'utils/datasets.py', 'utils/augmentation.py',
                  'augment/ultrasound.py', 'echo_aug_validation/augment_recipes.py',
                  'echo_aug_validation/io_utils.py', 'tools/evaluate_temporal_mae.py'])
    return {name:file_digest(ROOT / name) for name in paths}


def smoke_manifest(manifest):
    result = copy.deepcopy(manifest)
    recent, local = result['recent_frames'], result['local_frames']
    for split in ('train', 'val'):
        usable = [case for case in result[split] if case['ef'] is not None and
                  any(_complete_trace(case, trace['frame'], recent, local) for trace in case['traces'])]
        if len(usable) < 2:
            raise ValueError(f'Smoke needs two real {split} patients with complete official labeled context')
        result[split] = usable[:2]
        result['counts'][split]['eligible_cases'] = 2
    result['smoke_selection'] = 'two real complete-label patients per split; no synthetic fallback'
    result.pop('manifest_sha256', None); result['manifest_sha256'] = _hash(result)
    return result


def validate_sources(sources, smoke=False):
    from models.final_temporal_mae import checkpoint_payload, load_final_model
    records = {}
    for name, path in sources.items():
        if not Path(path).is_file():
            raise FileNotFoundError(f'{name} source missing: {path}; pass --checkpoint {name}=PATH; no automatic retraining')
        payload, config, state = checkpoint_payload(path)
        model = config['model']
        if not smoke and (payload.get('epoch') != 100 or model.get('local_frames') != 16
                          or model.get('img_size') != 112 or model.get('patch_size') != 8
                          or model.get('embed_dim') != 384 or model.get('tubelet_size') != 2
                          or model.get('depth', 12) != 12 or model.get('num_heads', 6) != 6
                          or model.get('in_chans', 3) != 3 or model.get('memory_grid', 4) != 4
                          or model.get('memory_compression') != 'temporal_attention'):
            raise ValueError(f'{name}: source is not the registered L16/112/ViT-S/100-epoch checkpoint')
        expected = dict(P='repeat', C='learned', F='factorized')[name]
        if model.get('frame_readout', 'repeat') != expected or model.get('memory_mode') != 'spatial':
            raise ValueError(f'{name}: source role/readout mismatch; names alone do not certify source identity')
        # Validate the actual tensor contract before any timing/training worker.
        verified, _, report = load_final_model(path)
        if report['initialized_keys']:
            raise ValueError(f'{name}: calibration source must load every tensor without fresh modules')
        del verified
        records[name] = dict(path=str(Path(path).resolve()), sha256=file_digest(path),
                             epoch=payload.get('epoch'), model=model, tensor_count=len(state),
                             source_tensor_contract_verified=True)
    keys = ('img_size','patch_size','local_frames','tubelet_size','in_chans','embed_dim','depth','num_heads',
            'memory_grid','core_depth','memory_compression','position_embedding')
    canonical = {key:records['C']['model'].get(key) for key in keys}
    for name in ('P','F'):
        actual = {key:records[name]['model'].get(key) for key in keys}
        if actual != canonical:
            raise ValueError(f'{name}: historical source architecture differs from the shared calibration contract')
    return records


class Queue:
    def __init__(self, args, sources, contracts, manifest):
        self.args, self.sources, self.contracts, self.manifest = args, sources, contracts, manifest
        root = Path(args.output_root)
        self.result = root / 'result' / args.run_tag
        self.ckpt = root / 'ckpt' / args.run_tag
        self.cache = root / 'cache' / args.run_tag
        for path in (self.result, self.ckpt, self.cache, self.result / 'jobs'):
            path.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.result / 'manifest.json'
        self.local = int(contracts['C']['model']['local_frames'])
        self.recent = int(contracts['C']['model'].get('frames', self.local * contracts['C']['model'].get('clip_count', 4))) if args.smoke else 64
        self.max_prefix = min(args.max_prefix, self.recent * 2) if args.smoke else args.max_prefix
        self.times = []
        if (self.result / 'stage_times.csv').exists():
            with (self.result / 'stage_times.csv').open() as f:
                self.times = list(csv.DictReader(f))
        self.scores, self.decisions, self.model_configs = {}, {}, {}
        self.populations = {}
        self.code_identity = None
        self.memory_target_identity = None
        self.updates, self.optional_ft, self.second_seed = args.updates, args.optional_ft, args.second_head_seed

    def guard(self):
        identity = dict(version=2, sources=self.contracts, manifest=self.manifest['manifest_sha256'],
                        code=scientific_code(), config={key:value for key,value in vars(self.args).items()
                        if key not in {'status', 'dry_run', 'preflight_only', 'device', 'num_workers', 'adopt_validation_cache'}})
        path = self.result / 'protocol.json'
        if path.exists():
            previous = json.loads(path.read_text(encoding='utf-8'))
            if previous != identity:
                self.adopt_validation_cache(previous, identity)
        write_json(path, identity); write_json(self.manifest_path, self.manifest)
        self.code_identity = identity['code']
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        except subprocess.CalledProcessError:
            commit = 'unavailable'
        write_json(self.result / 'implementation.json', dict(commit=commit, code=identity['code'],
                    plan='docs/时域最后一轮探索_Q1-Q3实验计划_20261010.md', parent_pid=os.getpid()))

    def adopt_validation_cache(self, previous, identity):
        ledger = self.result / 'operations' / 'validation_cache_upgrade.json'
        if not getattr(self.args, 'adopt_validation_cache', False) or ledger.exists():
            raise ValueError('Run-tag protocol/source/data/code changed; use a new RUN_TAG')
        old, new = copy.deepcopy(previous), copy.deepcopy(identity)
        old_code, new_code = old.pop('code'), new.pop('code')
        old_budget = old['config'].pop('cache_disk_gb')
        new_budget = new['config'].pop('cache_disk_gb')
        if old != new or new_budget < old_budget or new_budget > 16:
            raise ValueError('Execution upgrade cannot change data, source weights, training or evaluation settings')
        changed = {key for key in set(old_code) | set(new_code) if old_code.get(key) != new_code.get(key)}
        if changed != set(LEGACY_CACHE_EXECUTION) or any(old_code.get(key) != value for key,value in LEGACY_CACHE_EXECUTION.items()):
            raise ValueError('Only the registered legacy cache implementation can be upgraded once')
        if new_code.get('utils/final_temporal_tasks.py') != VALIDATION_CACHE_TASK_SHA256:
            raise ValueError('Task implementation is not the regression-verified cache upgrade')
        completed = []
        for request in sorted((self.result / 'jobs').glob('*.json')):
            job = json.loads(request.read_text(encoding='utf-8'))
            output = Path(job['output_dir'])
            if self.result.resolve() not in output.resolve().parents:
                raise ValueError('Job is outside this run')
            if not (output / 'DONE').exists():
                raise ValueError('Upgrade requires a completed task boundary, not a partial task resume')
            self.verify_done(job)
            completed.append(dict(job=request.name, done_sha256=file_digest(output/'DONE')))
        if not completed:
            raise ValueError('No completed legacy results to preserve')
        write_json(self.result / 'operations' / 'protocol_before_validation_cache.json', previous)
        write_json(ledger, dict(version=1, reason='User-authorized frozen validation feature reuse only',
            scientific_protocol_unchanged=True, precision='FP32 features, unchanged task heads and metrics',
            previous_protocol_sha256=digest(previous), upgraded_protocol_sha256=digest(identity),
            old_code=old_code, new_code=new_code, old_cache_disk_gb=old_budget, new_cache_disk_gb=new_budget,
            preserved_completed_jobs=completed,
            runtime_comparison_boundary='Uncached and cached validation timings must not be attributed to model architecture'))

    def verify_done(self, job):
        output = Path(job['output_dir'])
        done = json.loads((output/'DONE').read_text(encoding='utf-8'))
        for name,expected in done.get('artifacts',{}).items():
            path = output/name
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError(f'Missing/corrupted completed artifact: {path}')
        for name,expected in done.get('checkpoints',{}).items():
            path = Path(job['checkpoint_dir'])/name
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError(f'Missing/corrupted completed checkpoint: {path}')
        if not (output/'metrics.json').is_file():
            raise ValueError('DONE without metrics')

    def job(self, label, kind, checkpoint, **extra):
        job = dict(kind=kind, checkpoint=str(checkpoint), output_dir=str(self.result / label),
                   checkpoint_dir=str(self.ckpt / label), cache_dir=str(self.cache / label),
                   recent_frames=self.recent, max_prefix=self.max_prefix, seed=42,
                   num_workers=0 if self.args.smoke else self.args.num_workers,
                   micro_batch=1 if self.args.smoke else None, autotune=not self.args.smoke,
                   min_free_gb=self.args.min_free_gb, smoke=self.args.smoke,
                   overrides=dict(gradient_checkpointing=True))
        job.update(extra)
        return job

    def run(self, label, job):
        if self.code_identity is not None and scientific_code() != self.code_identity:
            raise ValueError('Implementation changed while the queue was running; no mixed-code next stage is allowed')
        request = self.result / 'jobs' / (label.replace('/', '__') + '.json')
        output = Path(job['output_dir'])
        metrics = output / 'metrics.json'
        if request.exists():
            registered = json.loads(request.read_text(encoding='utf-8'))
            if registered != job:
                allowed = {'disk_cache_bytes','ram_cache_bytes'}
                if not (output/'DONE').exists() or {k:v for k,v in registered.items() if k not in allowed} != {k:v for k,v in job.items() if k not in allowed}:
                    raise ValueError(f'{label}: registered job changed')
        else:
            write_json(request, job)
        if (output / 'DONE').exists():
            self.verify_done(job)
            print(f'========== {label}: reuse verified completed outputs ==========', flush=True)
            self.cleanup_cache(job)
            return json.loads(metrics.read_text(encoding='utf-8'))
        run_command([sys.executable, str(ROOT / 'tools/final_temporal_worker.py'), '--job', str(request),
                     '--manifest', str(self.manifest_path), '--device', self.args.device], label, self.result, self.times)
        if not (output / 'DONE').exists() or not metrics.exists():
            raise RuntimeError(f'{label}: worker exited without valid completion')
        self.cleanup_cache(job)
        return json.loads(metrics.read_text(encoding='utf-8'))

    def cleanup_cache(self, job):
        # Only completed jobs' disposable features, never source data or weights.
        path = Path(job['cache_dir'])
        root = self.cache.resolve()
        if not path.exists():
            return
        resolved = path.resolve()
        if root not in resolved.parents or path.is_symlink():
            raise ValueError('Refuse cache cleanup outside this run-owned root')
        if any(item.is_symlink() for item in path.rglob('*')):
            raise ValueError('Refuse cache cleanup containing links')
        shutil.rmtree(resolved)

    def budget(self):
        path = self.result / 'budget.json'
        if path.exists():
            value = json.loads(path.read_text(encoding='utf-8'))
            self.updates = value['updates']; self.optional_ft = value['optional_ft']; self.second_seed = value['second_seed']
            return value
        preflight = self.run('preflight', self.job('preflight', 'preflight', self.sources['C'],
                                  effective_batch=2 if self.args.smoke else 32))
        value = estimate_budget(preflight, self.updates, self.args.frozen_epochs, self.args.ft_epochs,
                                self.optional_ft, self.second_seed)
        if self.args.max_gpu_hours is not None and value['worst_case_hours'] > self.args.max_gpu_hours:
            self.optional_ft = self.second_seed = False
            value = estimate_budget(preflight, self.updates, self.args.frozen_epochs, self.args.ft_epochs, False, False)
            while self.updates > 750 and value['worst_case_hours'] > self.args.max_gpu_hours:
                self.updates = max(750, self.updates - 250)
                value = estimate_budget(preflight, self.updates, self.args.frozen_epochs, self.args.ft_epochs, False, False)
            if value['worst_case_hours'] > self.args.max_gpu_hours:
                write_json(self.result / 'budget_blocked.json', value)
                raise RuntimeError('Measured conservative budget exceeds requested limit even after permitted reductions; no scored run started')
        value.update(optional_ft=self.optional_ft, second_seed=self.second_seed,
                     locked_before_validation_selection=True, module_limit=9, max_prefix=self.max_prefix)
        write_json(path, value)
        print(f"Locked planning budget: mandatory~{value['mandatory_hours']:.2f}h; worst-case~{value['worst_case_hours']:.2f}h (conservative, not ETA)", flush=True)
        return value

    def task(self, model_id, checkpoint, freeze, task, *, exit_name='final', group=None, seed=42):
        style = 'frozen' if freeze else 'ft'
        label = f'{group or model_id}/{style}/{task}' + (f'_{exit_name}' if exit_name != 'final' else '') + (f'_seed{seed}' if seed != 42 else '')
        job = self.job(label, 'task', checkpoint, task=task, freeze=freeze, exit_name=exit_name, seed=seed,
                       epochs=1 if self.args.smoke else self.args.frozen_epochs if freeze else self.args.ft_epochs,
                       patience=12 if freeze else 20,
                       effective_batch=2 if self.args.smoke else 12 if task == 'ef' else 64,
                       disk_cache_bytes=int(self.args.cache_disk_gb * 1024**3),
                       ram_cache_bytes=int(self.args.cache_ram_gb * 1024**3))
        if self.args.smoke:
            job['subset'] = dict(train=2, val=2)
            if self.contracts['C']['model']['img_size'] != 112:
                job.update(head_dim=24, head_depth=1, head_heads=3)
        value = self.run(label, job)
        with Path(value['patient_predictions_path']).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        value['patient_keys'] = sorted(row['patient'] for row in rows)
        # Counts do not encode predictions: only the matched evaluation population.
        value['population_sha256'] = value['evaluation_population_sha256']
        expected = self.populations.setdefault(task, value['population_sha256'])
        if value['population_sha256'] != expected:
            raise ValueError(f'{label}: task patient/source/position evaluation population changed')
        return value

    def endpoint(self, model_id, checkpoint):
        entry = self.scores.setdefault(model_id, {})
        for task in ('ef', 'seg'):
            entry[task] = self.task(model_id, checkpoint, True, task)
        from models.final_temporal_mae import checkpoint_payload
        _, config, _ = checkpoint_payload(checkpoint)
        mode = config['model'].get('memory_mode', 'spatial')
        entry.update(memory_mode=mode, memory_slots=1 if mode == 'global' else
                     config['model'].get('memory_grid', 4)**2 + int(mode == 'spatial_global'),
                     checkpoint=str(checkpoint), attempted=True, status='complete')
        self.representation(model_id, checkpoint, detailed=model_id in {'C', 'F', 'B0', 'B1'})
        write_json(self.result / 'observed_scores.json', self.scores)

    def representation(self, model_id, checkpoint, detailed=False, memory=False):
        selected_mixed = model_id == 'B8' and self.decisions.get('organization', {}).get('selected') == 'B8'
        label = f'{model_id}/representation' + ('_memory' if memory else '_detailed' if detailed and model_id == 'B8' else '')
        config = dict(model_id={'C':'C100','F':'F100'}.get(model_id, model_id), seed=42,
                      recent_frames=self.recent, max_train=2 if self.args.smoke else 512,
                      max_val=2 if self.args.smoke else 256, max_seg_train=2 if self.args.smoke else 128,
                      max_seg_val=2 if self.args.smoke else 128, attention_patients=1 if self.args.smoke else 32,
                      mixed_selected=selected_mixed, reference_id='C100')
        if self.args.smoke:
            config.update(token_reservoir=32, anatomy_train_tokens=64, anatomy_val_tokens=64,
                          bootstrap_repetitions=20, tokens_per_window=8)
            if self.contracts['C']['model']['img_size'] != 112:
                config.update(layers=[1], attention_layers=[1])
        if memory:
            anchor = self.scores[model_id].get('history', {}).get('anchor')
            if not anchor:
                return None
            if self.args.smoke and self.contracts['C']['model'].get('depth', 12) < 9:
                self.scores[model_id]['representation_memory'] = dict(
                    status='tiny_smoke_has_no_canonical_layer9', audit_completed=False,
                    explanation='R5 target remains canonical layer9; no substitute target is invented')
                return None
            config['memory_prefix'] = anchor
        job = self.job(label, 'representation_memory' if memory else 'representation', checkpoint,
                       representation=config, detailed=detailed, memory_prediction=memory,
                       reference_checkpoint=str(self.sources['C']) if memory else None)
        value = self.run(label, job)
        entry = self.scores.setdefault(model_id, {})
        if memory:
            representation = entry['representation']
            representation['memory_prediction_delta'] = value.get('memory_prediction_delta')
            report = value['memory_prediction']
            representation['memory_prediction'] = report
            representation['memory_prediction_delta_convention'] = 'zero_minus_true'
            representation['memory_patient_observations'] = [dict(row, memory_prediction_delta=row['delta'])
                for row in value.get('patient_observations', [])]
            if report.get('status') == 'complete':
                identity = {key:report[key] for key in ('cohort_hash','target_projection_hash','history_anchor','cache_frames')}
                if self.memory_target_identity is None:
                    self.memory_target_identity = identity
                elif identity != self.memory_target_identity:
                    raise ValueError('R5 cohort/canonical target projection differs; no cross-model R5 selection')
            by_patient = {row['patient']:row for row in representation['memory_patient_observations']}
            for row in representation.get('patient_observations', []):
                if row['patient'] in by_patient:
                    row['memory_prediction_delta'] = by_patient[row['patient']]['memory_prediction_delta']
            entry['representation_memory'] = value
        else:
            entry['representation'] = value
        return value

    def adapt(self, model_id, overrides):
        cfg = dict(memory_mode='spatial', memory_compression='temporal_attention',
                   memory_write_source='fused', memory_read_location='tokens', frame_readout='learned',
                   candidate_rank=0, dynamic_orthogonal_weight=0., gradient_checkpointing=True,
                   reconstruction_recent_frames=self.recent)
        cfg.update(overrides); self.model_configs[model_id] = cfg
        label = f'{model_id}/adapt'
        job = self.job(label, 'adapt', self.sources['C'], overrides=cfg, updates=self.updates,
                       effective_batch=2 if self.args.smoke else 32, save_every_updates=300)
        self.run(label, job)
        checkpoint = self.ckpt / label / 'final.pt'
        if not checkpoint.is_file():
            raise FileNotFoundError('Missing final adaptation checkpoint')
        self.endpoint(model_id, checkpoint)
        audit_label = f'{model_id}/mechanisms'
        self.scores[model_id]['mechanisms'] = self.run(audit_label, self.job(audit_label, 'mechanisms', checkpoint,
                  audit=dict(num_windows=2 if self.args.smoke else 32, recent_frames=self.recent,
                             max_prefix=self.max_prefix, smoke=self.args.smoke)))
        audit = self.scores[model_id]['mechanisms']
        if audit.get('technical_findings'):
            raise RuntimeError(f'{model_id}: technical mechanism findings require review before selection: '
                               + str(audit['technical_findings']))
        return checkpoint

    def history(self, model_id):
        label = f'{model_id}/history'
        entry = self.scores[model_id]
        job = self.job(label, 'history', entry['checkpoint'], hidden=64,
                       train_cases=2 if self.args.smoke else 1024, val_cases=2 if self.args.smoke else 512,
                       head_epochs=1 if self.args.smoke else 60,
                       ef_head_checkpoint=entry['ef']['best_checkpoint'],
                       ef_head_metrics=entry['ef'],
                       seg_head_checkpoint=entry['seg']['best_checkpoint'],
                       seg_head_metrics=entry['seg'], head_patience=12)
        entry['history'] = self.run(label, job)

    def execute(self):
        self.budget()
        if self.args.preflight_only:
            write_json(self.result / 'current_stage.json', dict(stage='preflight_complete', status='completed'))
            return
        for model_id in ('P', 'C', 'F'):
            self.endpoint(model_id, self.sources[model_id])
            ft = {}
            for task in ('ef', 'seg'):
                ft[task] = self.task(model_id, self.sources[model_id], False, task)
            self.scores[model_id + '_FT'] = dict(**ft, checkpoint=self.sources[model_id], status='complete')
        exits = {'C':{'fused_final':dict(ef=self.scores['C']['ef']), 'fused_base':dict(ef=self.scores['C']['ef'])},
                 'F':{'fused_final':dict(ef=self.scores['F']['ef'])}}
        for model_id in ('C', 'F'):
            exits[model_id]['local_base'] = dict(ef=self.task(model_id, self.sources[model_id], True, 'ef', exit_name='local'))
            exits[model_id]['local_base']['seg'] = self.task(model_id, self.sources[model_id], True, 'seg', exit_name='local')
        exits['F']['fused_base'] = dict(ef=self.task('F', self.sources['F'], True, 'ef', exit_name='base'))
        exits['F']['fused_base']['seg'] = self.task('F', self.sources['F'], True, 'seg', exit_name='base')
        exits['C']['fused_base']['seg'] = self.scores['C']['seg']
        for model_id in ('C', 'F'):
            exits[model_id]['fused_final']['seg'] = self.scores[model_id]['seg']
        write_json(self.result / 'exit_scores.json', exits)
        self.adapt('B0', {})
        self.adapt('B1', dict(memory_mode='global'))
        for model_id in ('B0', 'B1'):
            self.history(model_id)
        a, b = task_vector(self.scores['B0']), task_vector(self.scores['B1'])
        if self.args.smoke or (abs(a[0] - b[0]) <= .3 and abs(a[1] - b[1]) <= .005) or ((a[0] - b[0]) * (a[1] - b[1]) > 0):
            for model_id in ('B0', 'B1'):
                self.representation(model_id, self.scores[model_id]['checkpoint'], memory=True)
        trigger, reason = mixed_trigger('B0', 'B1', self.scores)
        if self.args.smoke:
            trigger, reason = True, 'SMOKE: exercise conditional mixed implementation, not scientific selection'
        self.decisions['mixed_branch'] = dict(triggered=trigger, reason=reason)
        organization_candidates = ['B1']
        if trigger:
            self.adapt('B8', dict(memory_mode='spatial_global'))
            organization_candidates.append('B8')
        self.decisions['organization'] = select_reference('B0', organization_candidates, self.scores)
        org = self.decisions['organization']['selected']
        if org == 'B8':
            self.history('B8')
            self.representation('B8', self.scores['B8']['checkpoint'], detailed=True)
            self.representation('B8', self.scores['B8']['checkpoint'], memory=True)
        background = self.model_configs[org]
        self.adapt('B2', dict(background, memory_write_source='local'))
        self.adapt('B3', dict(background, memory_write_source='local', memory_read_location='frames'))
        self.decisions['write_only'] = select_reference(org, ['B2'], self.scores)
        self.decisions['read_only'] = select_reference('B2', ['B3'], self.scores)
        self.decisions['read_write'] = select_reference(org, ['B2', 'B3'], self.scores)
        ref = self.decisions['read_write']['selected']; background = self.model_configs[ref]
        self.adapt('B4', dict(background, candidate_rank=32 if not self.args.smoke else
                            min(4, self.contracts['C']['model']['embed_dim'])))
        self.adapt('B5', dict(background, frame_readout='factorized', dynamic_rank=16 if not self.args.smoke else 4,
                             dynamic_orthogonal_weight=.001))
        self.adapt('B6', dict(background, frame_readout='shrink'))
        self.decisions['memory_candidate'] = select_reference(ref, ['B4'], self.scores)
        trigger, reason = soft_trigger('C', 'F', ref, 'B5', 'B6', self.scores, exits)
        cft, fft = task_vector(self.scores['C_FT']), task_vector(self.scores['F_FT'])
        if fft[0] - cft[0] <= .3:
            trigger, reason = False, 'Full fine-tuning recovered F EF within tolerance; no frame correction needed'
        if self.args.smoke:
            trigger, reason = True, 'SMOKE: exercise fixed-beta soft mechanism, not scientific selection'
        self.decisions['soft_branch'] = dict(triggered=trigger, reason=reason)
        frame_candidates = ['B5', 'B6']
        if trigger:
            self.adapt('B7', dict(background, frame_readout='soft_factorized', dynamic_rank=16 if not self.args.smoke else 4,
                                  dynamic_orthogonal_weight=.001, soft_beta=.5))
            frame_candidates.append('B7')
        self.decisions['frame'] = select_reference(ref, frame_candidates, self.scores)
        selected = self.decisions['frame']['selected']
        confirm_ref, confirm_candidate = ref, selected
        confirm_axis = 'frame'
        if selected == ref:
            confirm_candidate = self.decisions['memory_candidate']['selected']
            confirm_axis = 'memory_candidate'
        if confirm_candidate == confirm_ref:
            confirm_ref, confirm_candidate = org, ref
            confirm_axis = 'read_write'
        if confirm_candidate == confirm_ref:
            confirm_ref, confirm_candidate = 'B0', org
            confirm_axis = 'organization'
        self.decisions['confirmation'] = dict(reference=confirm_ref, candidate=confirm_candidate,
                                             axis=confirm_axis, run=False, tasks=0, seed=None)
        if self.optional_ft and confirm_candidate != confirm_ref:
            for model_id in (confirm_ref, confirm_candidate):
                ft = {task:self.task(model_id, self.scores[model_id]['checkpoint'], False, task,
                                    group='confirm_' + model_id) for task in ('ef', 'seg')}
                self.scores['confirm_' + model_id] = dict(**ft, checkpoint=self.scores[model_id]['checkpoint'])
            self.decisions['confirmation'].update(run=True, tasks=4)
        if self.second_seed and confirm_candidate != confirm_ref:
            for model_id in sorted({confirm_ref, confirm_candidate}):
                seed_score = {task:self.task(model_id, self.scores[model_id]['checkpoint'], True, task,
                                             seed=43) for task in ('ef', 'seg')}
                self.scores[model_id + '_head43'] = seed_score
            self.decisions['confirmation']['seed'] = 43
        if self.second_seed and confirm_candidate != confirm_ref:
            confirmed = select_reference(confirm_ref + '_head43', [confirm_candidate + '_head43'], self.scores)
            self.decisions['confirmation']['seed_decision'] = confirmed
            self.decisions['confirmation']['stable_primary_choice'] = confirmed['selected'] == confirm_candidate + '_head43'
            if not self.decisions['confirmation']['stable_primary_choice']:
                # No extra run: retain the matched reference provisionally.
                self.decisions['confirmation']['provisional_fallback'] = confirm_ref
        if self.optional_ft and confirm_candidate != confirm_ref:
            a, b = task_vector(self.scores['confirm_' + confirm_ref]), task_vector(self.scores['confirm_' + confirm_candidate])
            unacceptable = b[0] - a[0] > .3 or a[1] - b[1] > .005
            self.decisions['confirmation']['ft_unacceptable_tradeoff'] = unacceptable
            if unacceptable:
                self.decisions['confirmation']['provisional_fallback'] = confirm_ref
        fallback = self.decisions['confirmation'].get('provisional_fallback')
        if fallback is not None:
            changed = self.decisions[confirm_axis]
            changed.update(primary_selected=changed['selected'], selected=fallback,
                           confirmation_fallback=True)
            if confirm_axis == 'organization':
                org = ref = selected = fallback
                for axis in ('read_write', 'memory_candidate', 'frame'):
                    self.decisions[axis].update(primary_selected=self.decisions[axis]['selected'], selected=fallback,
                                                confirmation_fallback=True)
            elif confirm_axis == 'read_write':
                ref = selected = fallback
                for axis in ('memory_candidate', 'frame'):
                    self.decisions[axis].update(primary_selected=self.decisions[axis]['selected'], selected=fallback,
                                                confirmation_fallback=True)
            elif confirm_axis == 'frame':
                selected = fallback
        self.decisions['conditional_design'] = dict(organization=org, read_write=ref,
                    memory=self.decisions['memory_candidate']['selected'], frame=selected,
                    local_frames=self.local, recent_frames=self.recent, application_prefix='any observed complete clip count',
                    q4_pending=True, not_a_merged_checkpoint=True)
        self.decisions['existing_exit_readout'] = {
            model_id:self.exit_decision(model_id, exits[model_id]) for model_id in ('C', 'F')}
        write_json(self.result / 'observed_scores.json', self.scores)
        if self.args.smoke:
            self.streaming(sorted({selected, self.decisions['memory_candidate']['selected']}))
            write_json(self.result / 'smoke_decisions.json', self.decisions)
            write_json(self.result / 'SMOKE_DONE', dict(passed=True, scientific_questions_closed=False,
                       cases_real=True, conditional_paths_exercised=True))
        else:
            self.close_questions(exits)
        write_json(self.result / 'current_stage.json', dict(stage='smoke_completed' if self.args.smoke else 'Q1-Q3_completed_Q4_pending', status='completed'))
        archive_analysis(self.result)

    def streaming(self, model_ids):
        for model_id in model_ids:
            label = f'{model_id}/streaming'
            self.scores[model_id]['streaming'] = self.run(label, self.job(label, 'streaming',
                self.scores[model_id]['checkpoint'], stream_frames=512, stream_cases=2))

    def exit_decision(self, model_id, exits):
        # A legal per-task readout decision, not a second encoder or a merged MAIN.
        chosen = dict(ef='fused_final', seg='fused_final')
        for task, tolerance in (('ef', .3), ('seg', .005)):
            candidates = []
            for name in ('local_base', 'fused_base', 'fused_final'):
                pair = task_vector(exits.get(name, {}))
                metric = pair[0 if task == 'ef' else 1]
                if metric is not None:
                    candidates.append((metric, name))
            best, name = (min(candidates) if task == 'ef' else max(candidates))
            final = task_vector(exits['fused_final'])[0 if task == 'ef' else 1]
            if (final - best if task == 'ef' else best - final) > tolerance:
                chosen[task] = name
        return dict(model=model_id, task_exits=chosen, scores=exits,
                    boundary='Separate task heads from the same pretrained checkpoint; prospective new backgrounds need Q4 confirmation')

    def close_questions(self, exits):
        org, rw = self.decisions['organization']['selected'], self.decisions['read_write']['selected']
        memory, frame = self.decisions['memory_candidate']['selected'], self.decisions['conditional_design']['frame']
        questions = {}
        evidence = {
            'Q1.1':['B0/history', 'B1/history', 'B0/representation', 'B1/representation'],
            'Q1.2':['B0/frozen', 'B1/frozen', 'B0/representation', 'B1/representation'],
            'Q1.3':[f'{org}/frozen', 'B2/frozen', 'B3/frozen'],
            'Q1.4':[f'{rw}/mechanisms', 'B4/mechanisms', 'B4/frozen'],
            'Q2.1':['C/frozen', 'F/frozen', 'C/ft', 'F/ft'],
            'Q2.2':['exit_scores.json', 'C/representation', 'F/representation'],
            'Q2.3':[f'{rw}/frozen', 'B5/frozen', 'B6/frozen'],
            'Q2.4':['exit_scores.json', 'observed_scores.json'],
            'Q3.1':['C/representation', 'F/representation', f'{frame}/representation'],
            'Q3.2':['C/frozen/seg', 'F/frozen/seg', f'{frame}/frozen/seg'],
            'Q3.3':['B0/history', 'B1/history', f'{rw}/representation'],
            'Q3.4':[f'{frame}/streaming', f'{memory}/streaming']}
        choices = {
            'Q1.1':'Use variable observed history; task utility is conditional on matched cohorts and length, H128 never required.',
            'Q1.2':f'Conditional organization: {org}; dual excluded.',
            'Q1.3':f'Conditional read/write configuration: {rw}.',
            'Q1.4':f'Candidate-update choice: {memory}; unconfirmed rank32 benefit defaults to registered original update.',
            'Q2.1':dict(C_frozen=task_vector(self.scores['C']), F_frozen=task_vector(self.scores['F']),
                       C_ft=task_vector(self.scores['C_FT']), F_ft=task_vector(self.scores['F_FT'])),
            'Q2.2':dict(exits=exits, explanation='Compare local/base/final readable information; do not force unique causal explanation.'),
            'Q2.3':f'Conditional frame mechanism: {frame}; geometry alone does not establish low-rank contribution.',
            'Q2.4':self.decisions['soft_branch'],
            'Q3.1':self.scores[frame].get('representation', {}).get('identity', {}),
            'Q3.2':dict(frame_exit='final', metric=self.scores[frame]['seg'], population='original complete labeled frame intersection'),
            'Q3.3':'Reuse paired history/exit information. No dense physiological phase claim.',
            'Q3.4':'Clip-end L16 output, bounded four-clip FIFO, patient reset and real-frame identity registered.'}
        self.streaming(sorted({frame, memory}))
        for question in QUESTION_IDS:
            present = all((self.result / path).exists() for path in evidence[question])
            coverage = {}
            if question in {'Q1.1', 'Q3.3'}:
                coverage = {key:self.scores[key]['history'].get('status', 'complete') for key in ('B0','B1')}
            elif question == 'Q1.4':
                coverage = {key:self.scores[key]['mechanisms']['gradient_status'] for key in {rw, 'B4'}}
            elif question == 'Q3.4':
                coverage = {key:self.scores[key]['streaming'].get('status') for key in {frame, memory}}
            questions[question] = dict(evidence=evidence[question], decision=choices[question], closed=present,
                   coverage=coverage,
                   boundary='One pretrained seed; validation development decision. Closure ends this finite design search, not proof of every hypothesis. Missing/short history and reduced gradient coverage remain unverified: ' + str(coverage))
        self.decisions['questions'] = questions
        write_final_report(self.result, self.decisions, self.scores,
                           dict(sources=self.contracts, model_configs=self.model_configs,
                                budget=json.loads((self.result / 'budget.json').read_text(encoding='utf-8')),
                                real_history='no fixed startup prefix, no padding', q4='pending'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_tag', default='temporal_final_20261010')
    p.add_argument('--data_root', default='/root/autodl-tmp/datasets/EchoNet-Dynamic')
    p.add_argument('--output_root', default='/root/autodl-tmp/outputs_temporal_final')
    p.add_argument('--checkpoint', action='append', default=[], help='P=PATH, C=PATH, F=PATH')
    p.add_argument('--num_workers', type=int, default=8)
    p.add_argument('--updates', type=int, default=1500)
    p.add_argument('--frozen_epochs', type=int, default=60)
    p.add_argument('--ft_epochs', type=int, default=80)
    p.add_argument('--max_prefix', type=int, default=128)
    p.add_argument('--min_free_gb', type=float, default=5.)
    p.add_argument('--cache_disk_gb', type=float, default=16.)
    p.add_argument('--cache_ram_gb', type=float, default=4.)
    p.add_argument('--max_gpu_hours', type=float)
    p.add_argument('--optional_ft', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--second_head_seed', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--preflight_only', action='store_true')
    p.add_argument('--status', action='store_true')
    p.add_argument('--adopt_validation_cache', action='store_true', help='One verified execution-only upgrade at a completed legacy task boundary')
    p.add_argument('--device', default=None)
    args = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', args.run_tag):
        p.error('run_tag requires a single safe directory name')
    if args.num_workers < 0 or not 0 <= args.max_prefix <= 128 or args.max_prefix % 16:
        p.error('Workers/prefix exceed the registered bounded protocol')
    if not args.smoke and not (750 <= args.updates <= 1500 and 20 <= args.frozen_epochs <= 60 and 20 <= args.ft_epochs <= 80):
        p.error('Budget outside the registered scope; use explicit --smoke for tiny checks')
    if min(args.min_free_gb, args.cache_disk_gb, args.cache_ram_gb) < 0:
        p.error('Storage/cache budgets must be nonnegative')
    if args.max_gpu_hours is not None and (not math.isfinite(args.max_gpu_hours) or args.max_gpu_hours <= 0):
        p.error('GPU-hours budget must be finite and positive')
    if args.smoke:
        args.updates, args.frozen_epochs, args.ft_epochs = 2, 1, 1
        args.optional_ft, args.second_head_seed = False, False
    return args


def main():
    args = parse_args()
    sources = dict(DEFAULT_SOURCES)
    for value in args.checkpoint:
        name, sep, path = value.partition('=')
        if not sep or name not in sources or not path:
            raise ValueError('--checkpoint must be P/C/F=PATH')
        sources[name] = path
    if args.status:
        folder = Path(args.output_root) / 'result' / args.run_tag
        for name in ('current_stage.json', 'budget.json', 'decisions.json', 'SMOKE_DONE'):
            path = folder / name
            if path.exists():
                print(name + ':\n' + path.read_text(encoding='utf-8'))
        return
    if args.dry_run:
        print(json.dumps(dict(sources=sources, mandatory=['B0','B1','B2','B3','B4','B5','B6'],
              conditional=['B8-spatial_global','B7-soft_factorized'], source='shared C100 initialization, not chained extra updates',
              prefix='available real H=0..128 in L16 multiples; no fixed application prefix',
              calibration='P/C/F each frozen+FT EF/seg', q4='pending', smoke=args.smoke), indent=2))
        return
    contracts = validate_sources(sources, args.smoke)
    local = int(contracts['C']['model']['local_frames'])
    recent = int(contracts['C']['model'].get('frames', local * contracts['C']['model'].get('clip_count', 4))) if args.smoke else 64
    maximum = min(args.max_prefix, recent * 2) if args.smoke else args.max_prefix
    manifest = build_manifest(args.data_root, recent_frames=recent, local_frames=local,
                              max_prefix=maximum, smoke=args.smoke)
    if args.smoke:
        manifest = smoke_manifest(manifest)
    import torch
    args.device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    queue = Queue(args, sources, contracts, manifest)
    with queue_lock(queue.result / 'queue.lock'):
        queue.guard()
        queue.execute()


if __name__ == '__main__':
    main()
