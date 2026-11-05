"""Stage-one gap diagnosis: no new MAE pretraining, no automatic later phases."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import ROOT, run_lock, verify_done, write_json
from research_pipeline.config import validate


def build_plans(parent, source, coverage_manifest, root, tag, official_checkpoint):
    coverage = copy.deepcopy(parent)
    coverage['run'] = dict(root=str(root), tag=tag + '_coverage')
    coverage['adapter'] = 'research_pipeline.coverage_repair:build'
    coverage['data'].update(coverage_manifest=str(coverage_manifest), evaluation_minimum_frames=1)
    old = next(s for s in parent['stages'] if s['id'] == 'seg_decoder_transfer_ft')
    stage = copy.deepcopy(old)
    stage.update(id='seg_full_coverage_ft', source=str(source))
    coverage['stages'] = [stage]
    anchor = copy.deepcopy(coverage)
    anchor['run']['tag'] = tag + '_official_anchor'
    anchor['adapter'] = 'research_pipeline.official_anchor:build'
    anchor['data'].update(seed=0, augmentation={'enabled': False})
    anchor['runtime'].update(precision='fp16', checkpoint_candidates=[False])
    anchor['training'].update(seed=0, effective_batch=64, epochs=1000, lr=1e-4,
                             weight_decay=1e-4, betas=[.9, .999], lr_schedule='constant',
                             warmup_epochs=0, patience=25, min_epochs=25,
                             monitor='val_loss', min_delta=0.0, clip_norm=0)
    anchor['stages'] = [dict(id='official_seg_ft', kind='seg', mode='finetune',
                            source=str(official_checkpoint))]
    return dict(coverage=validate(coverage), anchor=validate(anchor))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=['plan', 'smoke', 'run'], default='plan')
    parser.add_argument('--source-root', type=Path, default=Path('/root/autodl-fs/outputs_cr_spatial'))
    parser.add_argument('--source-tag', default='cr_patch8_w64_400_v1')
    parser.add_argument('--current-tag', default='cr_patch8_w64_1600_v1')
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-fs/outputs_cr_gap'))
    parser.add_argument('--tag', default='seg_gap_stage1_v1')
    parser.add_argument('--data-root', type=Path, default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    parser.add_argument('--official-checkpoint', type=Path, default=ROOT/'ckpt/mae/EchoCardMAE.pt')
    parser.add_argument('--reuse-anchor', type=Path,
                        help='Existing completed pipeline official-anchor result directory, not a log alone')
    parser.add_argument('--development-only', action='store_true',
                        help='Explicitly omit NEW candidates TEST; run will be marked development-only')
    args = parser.parse_args()
    parent_dir = args.source_root / 'result' / args.source_tag
    parent = json.loads((parent_dir/'plan.json').read_text())
    source = args.source_root / 'ckpt' / args.source_tag / 'pretrain/epoch_0400.pt'
    manifest = args.root/'manifests/full_supervision.json'
    plans = build_plans(parent, source, manifest, args.root, args.tag, args.official_checkpoint.resolve())
    control = args.root/'control'/args.tag
    if args.action == 'plan':
        print(json.dumps(dict(scope='Stage1 only; no MAE runs; mask/ROI require a later decision',
                              current_test=dict(root=str(args.source_root), tag=args.current_tag),
                              plans=plans, new_test=not args.development_only,
                              anchor_reuse=str(args.reuse_anchor) if args.reuse_anchor else None), indent=2))
        return
    if not verify_done(parent_dir) or not source.is_file():
        raise ValueError('Completed source run and its original 400-epoch SSL checkpoint are required')
    if not args.reuse_anchor and not args.official_checkpoint.is_file():
        raise FileNotFoundError('Official checkpoint absent; supply --official-checkpoint before allocating GPU')
    if args.reuse_anchor and not verify_done(args.reuse_anchor):
        raise ValueError('Anchor reuse requires verified DONE; a training log is not sufficient')
    if args.reuse_anchor:
        reference = json.loads((args.reuse_anchor/'protocol.json').read_text())['protocol']
        if reference['adapter'] != 'research_pipeline.official_anchor:build':
            raise ValueError('Reused anchor must match this implementation/protocol, not an arbitrary DONE')
    from utils.final_temporal_data import build_manifest, _load_manifest
    if manifest.exists():
        value = _load_manifest(manifest)
        if value.get('minimum_frames') != 1 or Path(value['root']).resolve() != args.data_root.resolve():
            raise ValueError('Coverage manifest protocol changed')
    else:
        write_json(manifest, build_manifest(args.data_root, max_prefix=0, minimum_frames=1))
    def command(*arguments):
        subprocess.run([sys.executable, '-u', *map(str, arguments)], cwd=ROOT, check=True)
    with run_lock(control):
        def action(name, function):
            write_json(control/'status.json', dict(phase='running', task=name,
                       pid=os.getpid(), time=time.time()))
            print(f'========== {name} ==========', flush=True)
            function()
        try:
            if args.action == 'run':
                action('current_VAL_selected_models_TEST', lambda: command(
                    ROOT/'tools/evaluate_patch8_test.py', '--root', args.source_root, '--tag',
                    args.current_tag, '--data-root', args.data_root))
            for name in ('anchor', 'coverage'):
                if name == 'anchor' and args.reuse_anchor:
                    continue
                path = control/(name+'.json')
                write_json(path, plans[name])
                action(name, lambda path=path: command('-m', 'research_pipeline',
                    'smoke' if args.action == 'smoke' else 'run', '--config', path))
            if args.action == 'run':
                cfg = plans['coverage']
                action('coverage_FULL_VAL_not_for_selection', lambda: command(
                    ROOT/'tools/evaluate_patch8_test.py', '--root', args.root, '--tag', cfg['run']['tag'],
                    '--stages', 'seg_full_coverage_ft', '--split', 'val_full', '--data-root', args.data_root))
                if not args.development_only:
                    for name, stage in (('coverage', 'seg_full_coverage_ft'), ('anchor', 'official_seg_ft')):
                        test_root, test_tag, test_stage = args.root, plans[name]['run']['tag'], stage
                        if name == 'anchor' and args.reuse_anchor:
                            test_root = args.reuse_anchor.resolve().parents[2]
                            test_tag, test_stage = args.reuse_anchor.parent.name, args.reuse_anchor.name
                        action(name+'_locked_TEST', lambda root=test_root, tag=test_tag, stage=test_stage: command(
                            ROOT/'tools/evaluate_patch8_test.py', '--root', root,
                            '--tag', tag, '--stages', stage, '--data-root', args.data_root))
                baseline = json.loads((parent_dir/'seg_decoder_transfer_ft/summary.json').read_text())
                candidate_dir = args.root/'result'/plans['coverage']['run']['tag']/'seg_full_coverage_ft'
                candidate = json.loads((candidate_dir/'summary.json').read_text())
                comparison = dict(reference=baseline, candidate=candidate,
                                  evaluation='same legacy VAL cohort; full VAL is supplemental',
                                  interpretation='coverage + necessary boundary handling bundle; not an architecture ablation')
                write_json(control/'comparison.json', comparison)
                text = ['# Segmentation gap: stage 1', '',
                        '| Common VAL metric | Original supervision | Restored supervision |',
                        '| --- | --- | --- |']
                for key in ('dice_mean', 'dice_global_mean'):
                    text.append(f"| {key} | {baseline['metrics'][key]:.6f} | {candidate['metrics'][key]:.6f} |")
                text += ['', 'This comparison uses VAL, not TEST. Do not substitute full-cohort scores into this table.',
                         'No new MAE pretraining. Mask/ROI are not queued.',
                         'TEST completed for new models: '+str(not args.development_only)]
                (control/'comparison.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
            write_json(control/'status.json', dict(phase='done', action=args.action,
                       pid=os.getpid(), time=time.time(),
                       new_test_completed=not args.development_only and args.action=='run',
                       limitation='No mask/ROI pretraining or architecture changes queued'))
        except BaseException as exc:
            write_json(control/'status.json', dict(phase='failed', error=repr(exc),
                       pid=os.getpid(), time=time.time()))
            raise


if __name__ == '__main__':
    main()
