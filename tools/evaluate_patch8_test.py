"""Evaluate VAL-selected full task models on TEST, without training/selection."""

import argparse
import copy
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def locked_checkpoint(root, tag, stage):
    from research_pipeline.artifacts import file_hash, verify_done
    directory = root / 'result' / tag / stage
    if not verify_done(directory):
        raise ValueError(f'Training must finish before TEST: {directory}')
    summary = json.loads((directory / 'summary.json').read_text())
    if summary.get('selection') != 'VAL only' or summary.get('mode') != 'finetune':
        raise ValueError('TEST entry requires a VAL-selected full fine-tuned model')
    name = summary['best_file']
    if Path(name).name != name:
        raise ValueError('Invalid checkpoint filename')
    path = root / 'ckpt' / tag / stage / name
    if file_hash(path) != summary['checkpoints'][name]['sha256']:
        raise ValueError('Selected checkpoint hash mismatch')
    request = json.loads((directory / 'request.json').read_text())
    return directory, summary, path, request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/outputs_cr_spatial'))
    parser.add_argument('--tag', default='cr_patch8_w64_1600_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    parser.add_argument('--stages', nargs='+', default=['seg_decoder_transfer_ft', 'ef_l1_ft'])
    parser.add_argument('--split', choices=['test', 'val_full'], default='test')
    args = parser.parse_args()
    import torch
    from research_pipeline.artifacts import file_hash, mark_done, run_lock, verify_done, write_json
    from research_pipeline.cli import configure
    from research_pipeline.echo import IdentifiedDataset
    from research_pipeline.engine import Heartbeat, build_experiment, csv_rows, evaluate
    from research_pipeline.metrics import stratify, summarize
    from tools.watch_pipeline_progress import StageDisplay
    from utils.final_temporal_data import build_manifest, WindowDataset

    stages = tuple(args.stages)
    # Lock both selected models before opening any TEST labels.
    locked = {s: locked_checkpoint(args.root, args.tag, s) for s in stages}
    output = args.root / 'result' / (args.tag + '_' + args.split)
    with run_lock(output):
        binding = {s: dict(checkpoint=str(v[2]), sha256=file_hash(v[2]),
                           selected_epoch=v[1]['selected_epoch']) for s, v in locked.items()}
        selection_path = output / 'selection_lock.json'
        if selection_path.exists() and json.loads(selection_path.read_text()) != binding:
            raise ValueError('TEST selection changed; do not overwrite prior evaluation')
        write_json(selection_path, binding)
        results = []
        for stage, (directory, selected, path, request) in locked.items():
            dest = output / stage
            if verify_done(dest):
                results.append(json.loads((dest / 'summary.json').read_text()))
                continue
            beat = Heartbeat(dest, 10)
            beat(force=True, phase='initializing')
            try:
                started = time.perf_counter()
                cfg = copy.deepcopy(request['config'])
                configure(cfg)
                # Rebuild the exact trained architecture, then load the whole task model strictly.
                exp = build_experiment(cfg, request['source'], torch.device(cfg['runtime']['device']))
                payload = torch.load(path, map_location='cpu', weights_only=False)
                exp.model.load_state_dict(payload['model'], strict=True)
                exp.model.requires_grad_(False)
                data = cfg['data']
                manifest = build_manifest(args.data_root, seed=data.get('seed', 42),
                                          recent_frames=data['recent_frames'], local_frames=data['local_frames'],
                                          max_prefix=data['max_prefix'],
                                          splits=('test',) if args.split == 'test' else ('train', 'val'),
                                          minimum_frames=data.get('evaluation_minimum_frames'))
                split = 'test' if args.split == 'test' else 'val'
                import importlib
                adapter = importlib.import_module(cfg['adapter'].split(':')[0])
                if hasattr(adapter, 'evaluation_dataset'):
                    test = adapter.evaluation_dataset(cfg, manifest, split, exp.task)
                else:
                    if args.split != 'test':
                        raise ValueError('Full VAL evaluation requires an explicit coverage adapter')
                    test = WindowDataset(manifest, split, task=exp.task, allow_test=True,
                                     recent_frames=data['recent_frames'], local_frames=data['local_frames'],
                                     max_prefix=data['max_prefix'], prefix=data.get('prefix', 0),
                                     seed=data.get('seed', 42), training=False,
                                     positions=data.get('val_positions', 'all'))
                if not len(test):
                    raise ValueError('No eligible evaluation records')
                test_ids = {r['patient'] for r in test.records}
                forbidden = set(exp.metadata['train_ids'])
                if args.split == 'test':
                    forbidden |= set(exp.metadata['val_ids'])
                if test_ids & forbidden:
                    raise ValueError('TEST overlaps supervised TRAIN/VAL')
                exp.val = IdentifiedDataset(test, split)
                runtime = json.loads((directory / 'runtime.json').read_text())['selected']
                label = 'TEST' if args.split == 'test' else 'VAL_FULL_NO_SELECTION'
                write_json(dest / 'protocol.json', dict(split=label, selection=binding[stage],
                           parent_protocol_sha256=selected['protocol_sha256'], config=cfg,
                           runtime=runtime, manifest_sha256=manifest['manifest_sha256'],
                           manifest_counts=manifest['counts'], task_excluded=test.excluded,
                           training=False, test_tuning=False,
                           ssl_data_exposure=selected.get('ssl_data_exposure')))
                with StageDisplay(dest, stage + '/' + label):
                    loss, metrics, patients, observations = evaluate(exp, cfg, runtime, beat)
                summary = dict(stage=stage, split=label, metrics=metrics, loss=loss,
                               selection='VAL-selected best checkpoint; no TEST tuning',
                               selected_epoch=selected['selected_epoch'], checkpoint_sha256=file_hash(path),
                               ssl_data_exposure=selected.get('ssl_data_exposure'),
                               wall_seconds=time.perf_counter()-started)
                common = [row for row in observations if row.get('coverage_kind') == 'legacy_complete_context']
                if common:
                    summary['legacy_common_context_metrics'] = summarize(common, exp.task)[0]
                    summary['common_context_note'] = 'Original all-positions eligibility; do not compare full-cohort score against this subset'
                write_json(dest / 'summary.json', summary)
                write_json(dest / 'patients.json', patients)
                write_json(dest / 'observations.json', observations)
                write_json(dest / 'strata.json', stratify(observations, exp.task))
                write_json(dest / 'coverage_strata.json', stratify(observations, exp.task,
                           fields=('coverage_kind', 'left_padding_frames')))
                csv_rows(dest / 'patients.csv', patients)
                mark_done(dest, ['summary.json', 'protocol.json', 'patients.json', 'observations.json',
                                 'strata.json', 'coverage_strata.json'])
                beat(force=True, phase='done')
                results.append(summary)
                del exp, payload
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except BaseException as exc:
                beat(force=True, phase='failed', error=repr(exc))
                raise
        write_json(output / 'comparison.json', results)
        mark_done(output, ['selection_lock.json', 'comparison.json'])
        print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
