"""Stdlib-only live progress reader; safe beside an already running checkout."""

import argparse
import csv
import io
import json
from pathlib import Path
import sys
import threading
import time


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {}


def tail(path, size=16000):
    try:
        with Path(path).open('rb') as handle:
            handle.seek(max(0,Path(path).stat().st_size-size))
            return handle.read().decode('utf-8',errors='replace')
    except OSError:
        return ''


def latest_metrics(path):
    try:
        with Path(path).open(encoding='utf-8-sig',newline='') as handle:
            header = next(csv.reader(handle))
        lines = tail(path).splitlines()
        for line in reversed(lines):
            row = next(csv.reader([line]))
            if len(row)==len(header) and row[header.index('epoch')].isdigit():
                return dict(zip(header,row))
    except (OSError,ValueError,StopIteration,csv.Error):
        pass
    return {}


def duration(seconds):
    seconds = max(0,int(seconds))
    days,seconds = divmod(seconds,86400)
    hours,seconds = divmod(seconds,3600)
    minutes,seconds = divmod(seconds,60)
    return (f'{days}d ' if days else '')+(f'{hours:02d}:' if hours or days else '')+f'{minutes:02d}:{seconds:02d}'


def render_stage(directory, stage, now=None):
    directory = Path(directory)
    status = read_json(directory/'status.json')
    now = time.time() if now is None else now
    phase = status.get('phase','waiting_for_worker')
    parts = [stage,phase]
    if 'epoch' in status:
        parts.append(f"epoch {status['epoch']}/{status.get('epochs', '?')}")
    count,total = status.get('completed'),status.get('total')
    if isinstance(count,(int,float)) and isinstance(total,(int,float)) and total>0:
        fraction = min(1,max(0,count/total))
        parts.append('['+'#'*int(fraction*20)+'-'*(20-int(fraction*20))+f'] {count}/{total} ({fraction:.1%})')
    if phase=='autotune':
        choice = status.get('selection',{})
        parts.append(f"trial={status.get('trial','?')} batch={choice.get('micro_batch','?')} workers={choice.get('num_workers','?')} checkpointing={choice.get('gradient_checkpointing','?')}")
    if isinstance(status.get('loss'),(float,int)):
        parts.append(f"loss={status['loss']:.6f}")
    if isinstance(status.get('lr'),(float,int)):
        parts.append(f"lr={status['lr']:.6g}")
    speed = status.get('samples_per_second')
    if speed:
        parts.append(f'{speed:.2f} windows/s')
        if phase=='training' and isinstance(total,(float,int)) and isinstance(count,(float,int)):
            parts.append('epoch ETA~'+duration((total-count)/speed))
    if 'successful_updates' in status:
        parts.append(f"updates={status['successful_updates']}")
    if 'allocated_bytes' in status:
        parts.append(f"memory={status['allocated_bytes']/1024**3:.2f}/{status.get('reserved_bytes',0)/1024**3:.2f} GiB allocated/reserved")
    if 'data_wait_fraction' in status:
        parts.append(f"data_wait={status['data_wait_fraction']:.1%}")
    metrics = latest_metrics(directory/'metrics.csv')
    if metrics:
        parts.append(f"last_epoch={metrics['epoch']} time={duration(float(metrics.get('train_seconds') or 0))}")
        for name in ('lr','val_dice_mean','val_mae_pp','val_loss'):
            if metrics.get(name):
                if name=='lr' and 'lr' in status:
                    continue
                label = 'last_lr' if name=='lr' else name
                parts.append(f'{label}={float(metrics[name]):.6g}')
    if status.get('time'):
        age = max(0,now-status['time'])
        parts.append(f'heartbeat_age={age:.0f}s'+(' (inspect logs; not proof of a stall)' if age>900 and phase not in {'done','paused_at_budget'} else ''))
    if status.get('error'):
        parts.append('ERROR: '+str(status['error']))
    return ' | '.join(parts)


class StageDisplay:
    """Observe the child without consuming its stdout or changing its training RNG."""
    def __init__(self,directory,stage,interval=10):
        self.directory,self.stage,self.interval = Path(directory),stage,interval
        self.stop = threading.Event()
        self.thread = None
        self.previous = None

    def publish(self):
        text = render_stage(self.directory,self.stage)
        if text != self.previous:
            print('[progress] '+text,flush=True)
            self.previous = text

    def __enter__(self):
        self.publish()
        def observe():
            while not self.stop.wait(self.interval):
                try:
                    self.publish()
                except (OSError,ValueError):
                    pass  # A transient read/display failure must never kill a training worker.
        self.thread = threading.Thread(target=observe,daemon=True,name='pipeline-progress')
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set()
        self.thread.join(timeout=2)
        self.publish()


def run_view(root,tag,once=False,interval=2,plain=False):
    directory = Path(root)/'result'/tag
    previous = None
    while True:
        events = []
        for line in tail(directory/'queue.jsonl').splitlines():
            try: events.append(json.loads(line))
            except ValueError: pass
        latest = events[-1] if events else {}
        stage = latest.get('stage')
        selection = read_json(directory/'execution_selection.json').get('stages',[])
        index = f' [{selection.index(stage)+1}/{len(selection)}]' if stage in selection else ''
        lines = [f'Run: {tag} | stage: {stage or "waiting"}{index}',
                 render_stage(directory/stage,stage) if stage else f'Waiting for queue: {directory}',
                 f'worker log: {directory/stage/"worker.log"}' if stage else '',
                 'Read-only display. Ctrl+C closes this view ONLY, not the experiment.']
        text = '\n'.join(lines)
        if not plain and sys.stdout.isatty() and not once:
            print('\033[2J\033[H'+text,flush=True)
        elif text != previous or once:
            print(text,flush=True)
        previous = text
        if once: return
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='/root/autodl-tmp/outputs_cr_clean')
    parser.add_argument('--tag',default='cr_forward_v1')
    parser.add_argument('--interval',type=float,default=2)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--plain',action='store_true')
    args = parser.parse_args()
    if args.interval<=0: parser.error('--interval must be positive')
    try:
        run_view(args.root,args.tag,args.once,args.interval,args.plain)
    except KeyboardInterrupt:
        pass


if __name__=='__main__':
    main()
