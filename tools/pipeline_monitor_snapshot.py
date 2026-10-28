"""Read-only stdlib snapshot, sent over stdin; never imports or edits training code."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError,ValueError):
        return None


def tail(path, size=12000):
    try:
        with path.open('rb') as handle:
            handle.seek(max(0,path.stat().st_size-size))
            return handle.read().decode('utf-8',errors='replace')
    except OSError:
        return ''


def snapshot(root='/root/autodl-tmp/outputs_cr_clean', tag='cr_forward_v1', project='/root/autodl-tmp/EchoRVM'):
    now = time.time()
    processes = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit(): continue
        try:
            if os.readlink(directory/'cwd') != project: continue
            command = (directory/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            if 'research_pipeline' not in command and 'run_clip_recurrent' not in command: continue
            processes.append(dict(pid=int(directory.name),command=command[:800],
                                  state=(directory/'stat').read_text().split(') ',1)[1].split()[0]))
        except OSError: pass
    runs = []
    for name in (tag+'_smoke',tag):
        result = Path(root)/'result'/name
        if not result.exists(): continue
        events = []
        for line in tail(result/'queue.jsonl',24000).splitlines():
            try: events.append(json.loads(line))
            except ValueError: pass
        latest = events[-1] if events else None
        current = result/latest['stage'] if latest else None
        status = read_json(current/'status.json') if current else None
        metrics = []
        if current and (current/'metrics.csv').exists():
            with (current/'metrics.csv').open(encoding='utf-8-sig') as handle:
                metrics = list(csv.DictReader(handle))[-2:]
        log = current/'worker.log' if current else None
        completed = read_json(result/'forward_comparison.json')
        runs.append(dict(tag=name,last_event=latest,status=status,
                         heartbeat_age_seconds=None if not status else round(now-status['time'],1),
                         recent_epochs=metrics,
                         worker_log_tail=tail(log,5000) if log else '',
                         log_mtime=None if not log or not log.exists() else log.stat().st_mtime,
                         stage_terminal_counts={kind:sum(1 for p in result.glob('*/'+kind)) for kind in ('DONE.json','PAUSED.json')},
                         comparison_present=completed is not None,
                         comparison_mtime=(result/'forward_comparison.json').stat().st_mtime if completed is not None else None))
    disk = shutil.disk_usage(root)
    gpu = subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],
                         capture_output=True,text=True,timeout=15)
    return dict(remote_time=now,processes=processes,runs=runs,disk_free_gib=round(disk.free/1024**3,2),gpu=gpu.stdout.strip(),
                note='PAUSED at declared budget is expected; GPU percentage alone is not a stall detector')


if __name__=='__main__':
    print(json.dumps(snapshot(tag=os.environ.get('MAE_RUN_TAG', 'cr_forward_fast_v3')),ensure_ascii=False))
