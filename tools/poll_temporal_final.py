"""Small read-only monitoring snapshot; result paths never depend on the cwd."""

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess


def read_json(path, errors):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        errors.append(dict(path=str(path), error=str(error)))
        return None


def tail(path, size):
    try:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - size))
            return handle.read(size).decode('utf-8', errors='replace')
    except FileNotFoundError:
        return ''


def file_marker(path):
    try:
        stat = path.stat()
        return dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    except FileNotFoundError:
        return None


def process(pid, proc_root=Path('/proc')):
    try:
        path = proc_root / str(int(pid))
        fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
        command = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        result = dict(pid=int(pid), ppid=int(fields[1]), state=fields[0], start_ticks=fields[19],
                      cpu_ticks=int(fields[11]) + int(fields[12]), command=command)
        fd = path / 'fd'
        result['open_fds'] = sum(1 for _ in fd.iterdir()) if fd.exists() else None
        for line in (path / 'limits').read_text().splitlines():
            if line.startswith('Max open files'):
                result['nofile_soft'] = line.split()[3]
                break
        return result
    except (OSError, ValueError, IndexError, TypeError):
        return None


def snapshot(result_dir, proc_root=Path('/proc'), gpu=True):
    root = Path(result_dir).resolve()
    errors = []
    result = dict(checked_at=datetime.now(timezone.utc).isoformat(), result_dir=str(root), errors=errors)
    stage = read_json(root / 'current_stage.json', errors)
    result['stage'] = stage
    if stage is None:
        errors.append(dict(path=str(root / 'current_stage.json'), error='stage file unavailable; wrong result directory or run missing'))
    name = stage.get('stage') if isinstance(stage, dict) else None
    if name:
        path = (root / name).resolve()
        if root not in path.parents:
            errors.append(dict(path=str(path), error='unsafe stage path'))
        else:
            result['stage_result_dir'] = str(path)
            files = {key: path / value for key, value in {
                'train_log': 'logs/train.log', 'metrics': 'logs/metrics.csv',
                'normalization': 'normalization.json', 'protocol': 'protocol.json', 'done': 'DONE'}.items()}
            result['markers'] = {key: file_marker(value) for key, value in files.items()}
            result['train_log_tail'] = tail(files['train_log'], 600)
            if files['metrics'].exists():
                with files['metrics'].open(newline='', encoding='utf-8') as handle:
                    latest = deque(csv.DictReader(handle), maxlen=1)
                result['latest_epoch'] = list(latest)[0] if latest else None
            protocol = read_json(files['protocol'], errors) or {}
            result['runtime'] = {key: protocol.get(key) for key in
                                 ('task', 'freeze', 'micro_batch', 'effective_batch', 'patience')}
            result['runtime']['workers'] = protocol.get('runtime', {}).get('workers')
            result['runtime']['autotune_mode'] = protocol.get('autotune', {}).get('mode')
    result['queue_log_tail'] = tail(root / 'operations/continued_queue_spawn.log', 1600)
    result['handoffs'] = {}
    for key in ('followup_audit', 'early_stopping'):
        value = read_json(root / ('operations/' + key + '_status.json'), errors)
        if value:
            result['handoffs'][key] = value
            if value.get('state') == 'failed':
                result['handoffs'][key]['log_tail'] = tail(root / ('operations/' + key + '_guardian.log'), 600)
    implementation = read_json(root / 'implementation.json', errors) or {}
    controller = process(implementation.get('parent_pid'), proc_root)
    result['controller'] = controller
    workers, readers = [], []
    if controller and proc_root.exists():
        processes = []
        for item in proc_root.iterdir():
            if item.name.isdigit():
                observed = process(item.name, proc_root)
                if observed:
                    processes.append(observed)
        workers = [item for item in processes if item['ppid'] == controller['pid'] and
                   str(root) in item['command'] and 'final_temporal_worker.py' in item['command']]
        parents = {item['pid'] for item in workers}
        readers = [item for item in processes if item['ppid'] in parents]
    result['workers'] = workers
    result['readers'] = {key: value for key, value in {
        'count': len(readers), 'cpu_ticks': sum(item['cpu_ticks'] for item in readers),
        'states': dict((state, sum(item['state'] == state for item in readers)) for state in {item['state'] for item in readers})}.items()}
    result['final_complete'] = bool(stage and stage.get('stage') == 'Q1-Q3_completed_Q4_pending' and
        stage.get('status') == 'completed' and all((root / name).exists() for name in
        ('Q1-Q3_closure.md', 'comparison.csv', 'decisions.json')))
    if root.exists():
        disk = shutil.disk_usage(root)
        result['disk_free_gib'] = round(disk.free / 1024**3, 2)
    if gpu:
        try:
            reading = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
            result['gpu'] = reading.stdout.strip() if reading.returncode == 0 else 'unavailable: ' + reading.stderr[-250:]
        except (OSError, subprocess.TimeoutExpired) as error:
            result['gpu'] = 'unavailable: ' + str(error)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result_dir', required=True)
    args = parser.parse_args()
    # Compact, bounded stdout; no model imports, writes, or process signals.
    value = snapshot(args.result_dir)
    print(json.dumps(value, ensure_ascii=True, separators=(',', ':')), flush=True)


if __name__ == '__main__':
    main()
