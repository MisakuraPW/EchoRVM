"""Storage-only changes and safe retention for the temporal research runner."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml


STORAGE_KEYS = {'dir', 'save_last_every_n_epochs', 'min_free_gb', 'best_weights_only'}


def training_identity(config):
    result = copy.deepcopy(config)
    for key in STORAGE_KEYS:
        result.get('checkpoint', {}).pop(key, None)
    return result


def check_protocol(destination: Path, config: dict) -> str:
    digest = hashlib.sha256(json.dumps(training_identity(config), sort_keys=True).encode()).hexdigest()
    identity = destination / 'protocol.sha256'
    if identity.exists() and identity.read_text().strip() != digest:
        previous = destination / 'requested_config.yaml'
        if not previous.exists() or training_identity(yaml.safe_load(previous.read_text(encoding='utf-8'))) != training_identity(config):
            raise RuntimeError(f'{destination.name}: changed training protocol under same run_tag. Use a new run_tag.')
    return digest


def checked_path(path: Path, root: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    if resolved == root or not resolved.is_relative_to(root) or path.is_symlink():
        raise RuntimeError(f'Refusing storage operation outside owned directory: {path}')
    return resolved


def move_owned(source: Path, target: Path, root: Path):
    checked_path(source, root)
    checked_path(target, root)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite during layout migration: {target}')
    if source.is_dir() and any(p.is_symlink() for p in source.rglob('*')):
        raise RuntimeError(f'Refusing to move a symlink-containing tree: {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)


def prepare_layout(run_root: Path, names, *, migrate: bool = False):
    results, checkpoints = run_root / 'result', run_root / 'ckpt'
    legacy = [run_root / name for name in names if (run_root / name).is_dir()]
    if legacy and not migrate:
        raise RuntimeError('Legacy output layout detected. Stop old training first, then rerun with '
                           '--migrate_legacy_layout (moves files without copying weights).')
    if migrate:
        results.mkdir(parents=True, exist_ok=True)
        checkpoints.mkdir(parents=True, exist_ok=True)
        for source in legacy:
            model_ckpt = checkpoints / source.name
            old = source / 'checkpoints'
            if old.exists():
                model_ckpt.mkdir(parents=True, exist_ok=True)
                for item in list(old.iterdir()):
                    move_owned(item, model_ckpt / item.name, run_root)
                old.rmdir()
            for anchor in (source / 'full_finetune').glob('*'):
                old = anchor / 'checkpoints'
                if old.exists():
                    target = model_ckpt / 'full_finetune' / anchor.name
                    target.mkdir(parents=True, exist_ok=True)
                    for item in list(old.iterdir()):
                        move_owned(item, target / item.name, run_root)
                    old.rmdir()
            target = results / source.name
            target.mkdir(parents=True, exist_ok=True)
            for item in list(source.iterdir()):
                move_owned(item, target / item.name, run_root)
            source.rmdir()
        # Move only known runner reports, never arbitrary files from the run root.
        for name in ('plan.json', 'invocations.jsonl', 'input_contract.json', 'stage_times.csv',
                     'comparison.csv', 'comparison.md', 'paired_comparisons.csv',
                     'full_finetune.csv', 'representation_trajectory.png', 'analysis.zip'):
            source = run_root / name
            if source.exists():
                move_owned(source, results / name, run_root)
    return results, checkpoints


def remove_owned_checkpoint(path: Path, ckpt_root: Path, result_dir: Path, reason: str):
    checked_path(path, ckpt_root)
    if not path.is_file():
        return
    size = path.stat().st_size
    path.unlink()
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / 'checkpoint_cleanup.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(path=str(path), bytes=size, reason=reason,
                                     time=datetime.now().isoformat())) + '\n')


def prune_audited_snapshot(ckpt_dir, results, epoch, final_epoch, *, keep=False):
    audit = results / 'audit' / f'epoch_{epoch:04d}'
    if not keep and epoch != final_epoch and (audit / 'DONE').is_file() and (audit / 'metrics.json').is_file():
        remove_owned_checkpoint(ckpt_dir / f'epoch_{epoch:04d}.pt', ckpt_dir, results,
                                'stage audit completed; final model retained')


def prune_completed_resume(ckpt_dir, results, final_epoch, audit_epochs, *, keep=False, audited=True):
    if keep or not audited or not (results / 'PRETRAIN_DONE').is_file():
        return
    if not (ckpt_dir / f'epoch_{final_epoch:04d}.pt').is_file():
        raise RuntimeError('Refusing cleanup without final checkpoint')
    if not all((results / 'audit' / f'epoch_{e:04d}' / 'DONE').is_file()
               and (results / 'audit' / f'epoch_{e:04d}' / 'metrics.json').is_file() for e in audit_epochs):
        return
    for name in ('last.pt', 'interrupt.pt'):
        remove_owned_checkpoint(ckpt_dir / name, ckpt_dir, results,
                                'pretraining and all requested audits completed')
