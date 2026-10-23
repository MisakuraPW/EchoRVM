"""Atomic result metadata, ownership locks, and result-only exports."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import time
import zipfile
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def event(path, **value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(dict(time=time.time(), **value), ensure_ascii=False, allow_nan=False) + '\n')


def layout(config, stage=None):
    root, tag = Path(config['run']['root']), config['run']['tag']
    suffix = Path(tag) / stage if stage else Path(tag)
    return {key: root / key / suffix for key in ('result', 'ckpt', 'cache')}


def code_identity(extra=()):
    files = list((ROOT / 'research_pipeline').glob('*.py'))
    files += list((ROOT / 'models').glob('*.py')) + list((ROOT / 'utils').glob('*.py'))
    files += list((ROOT / 'augment').glob('*.py')) + list((ROOT / 'echo_aug_validation').glob('*.py'))
    files += [Path(p).resolve() for p in extra]
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): file_hash(path)
            for path in sorted(set(files)) if path.is_file()}


def provenance():
    import torch
    def git(*args):
        try:
            return subprocess.check_output(['git', *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                platform=platform.platform(), hostname=socket.gethostname(),
                git_commit=git('rev-parse', 'HEAD'), git_dirty=bool(git('status', '--porcelain')))


def alive(pid):
    if os.name == 'nt':
        import ctypes
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'RUNNING.lock'
    value = dict(pid=os.getpid(), hostname=socket.gethostname(), time=time.time())
    try:
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle)
    except FileExistsError:
        previous = json.loads(path.read_text(encoding='utf-8'))
        if previous['hostname'] != socket.gethostname() or alive(previous['pid']):
            raise RuntimeError(f'Another owner may be running: {previous}; no process was stopped')
        # Only recover a stale lock on this host, without touching workloads.
        path.unlink()
        with path.open('x', encoding='utf-8') as handle:
            json.dump(value, handle)
    try:
        yield
    finally:
        if path.exists() and json.loads(path.read_text(encoding='utf-8')) == value:
            path.unlink()


def mark_done(directory, names):
    write_json(directory / 'DONE.json', {name: file_hash(directory / name) for name in names})


def verify_done(directory):
    path = directory / 'DONE.json'
    if not path.exists():
        return False
    for name, expected in json.loads(path.read_text(encoding='utf-8')).items():
        file = directory / name
        if not file.is_file() or file_hash(file) != expected:
            raise RuntimeError(f'Completed artifact missing or modified: {file}')
    return True


def export_results(directory, destination):
    directory, destination = Path(directory).resolve(), Path(destination).resolve()
    files = sorted(p for p in directory.rglob('*') if p.is_file() and p != destination)
    blocked = {'.pt', '.pth', '.ckpt', '.safetensors', '.npy', '.npz', '.zip'}
    files = [p for p in files if p.suffix.lower() not in blocked and not p.is_symlink()
             and not p.name.endswith(('.tmp', '.lock'))]
    if destination.is_relative_to(directory):
        raise ValueError('Write exports outside the result tree')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        index = []
        for path in files:
            name = str(path.relative_to(directory)).replace('\\', '/')
            archive.write(path, name)
            index.append(dict(path=name, bytes=path.stat().st_size, sha256=file_hash(path)))
        archive.writestr('EXPORT_MANIFEST.json', json.dumps(index, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError('Export ZIP integrity check failed')
    write_json(destination.with_suffix('.sha256.json'), dict(sha256=file_hash(destination), files=len(files)))
    return dict(path=str(destination), files=len(files), bytes=destination.stat().st_size)
