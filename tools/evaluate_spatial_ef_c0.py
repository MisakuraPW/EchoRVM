"""Supplement an already completed spatial EF comparison with frozen C0 TEST."""

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import file_hash, layout, mark_done, run_lock, verify_done, write_json
from research_pipeline.cli import configure
from research_pipeline.config import safe_name
from tools import run_spatial_ef_readout as frozen
from tools import run_patch8_stage_a as a


def evaluate(args):
    import torch
    from research_pipeline.echo import IdentifiedDataset
    from research_pipeline.engine import Heartbeat, amp_context, build_experiment, csv_rows
    from research_pipeline.metrics import summarize
    from research_pipeline.runtime import loader
    from research_pipeline.spatial_ef import tensor_digest
    from tools.watch_pipeline_progress import StageDisplay
    from utils.final_temporal_data import WindowDataset, build_manifest

    cfg,c0,source=frozen.plan(args)
    control=args.root/'control'/args.tag
    if not verify_done(control) or not verify_done(control/'test'):
        raise ValueError('C1/C2 TEST must already be complete and sealed')
    prior=a.read(control/'comparison.json')
    original=a.read(control/'protocol.json')
    if original['config']!=cfg or original['source_sha256']!=file_hash(source):
        raise ValueError('C0 must share the original C1/C2 protocol and SSL source')
    summary=a.completed(c0)
    request=a.read(c0/'request.json')
    settings=request['config']
    name=summary['best_file']
    if Path(name).name!=name:
        raise ValueError('Invalid C0 checkpoint name')
    checkpoint=layout(settings,settings['stage']['id'])['ckpt']/name
    sha=file_hash(checkpoint)
    if (sha!=summary['checkpoints'][name]['sha256'] or
            original['c0_summary_sha256']!=file_hash(c0/'summary.json') or request['source']!=source):
        raise ValueError('C0 VAL checkpoint or recorded source has changed')
    data=cfg['data']
    root=Path(a.read(data['manifest'])['root']).resolve()
    if args.data_root.resolve()!=root:
        raise ValueError('C0 must use the original TEST dataset root')
    dest=control/'c0_test'
    with run_lock(dest):
        a.lock_selection(dest/'selection_lock.json',dict(checkpoint=str(checkpoint),sha256=sha,
                         selected_epoch=summary['selected_epoch'],ssl_sha256=file_hash(source),
                         original_comparison_sha256=file_hash(control/'comparison.json'),
                         selection='original frozen VAL checkpoint; inference only'))
        if verify_done(dest):
            return a.read(dest/'comparison_all.json')
        configure(settings)
        device=torch.device(settings['runtime']['device'])
        exp=build_experiment(settings,source,device)
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if payload['epoch']!=summary['selected_epoch'] or payload['source']!=source:
            raise ValueError('Wrong C0 endpoint epoch or SSL source')
        exp.model.load_state_dict(payload['model'],strict=True)
        exp.model.requires_grad_(False).eval()
        backbone=tensor_digest(exp.backbone)
        for arm in frozen.ARMS:
            integrity=a.read(layout(cfg,arm)['result']/'frozen_integrity.json')
            if integrity['backbone_digest']!=backbone:
                raise ValueError('C0 frozen encoder differs from C1/C2')

        started=time.perf_counter()
        manifest=build_manifest(root,seed=data['seed'],recent_frames=64,local_frames=16,
                                max_prefix=0,splits=('test',))
        if manifest['manifest_sha256']!=prior['test']['manifest_sha256']:
            raise ValueError('TEST dataset manifest does not match the completed evaluation')
        dataset=IdentifiedDataset(WindowDataset(manifest,'test',task='ef',allow_test=True,
                   recent_frames=64,local_frames=16,max_prefix=0,prefix=0,seed=data['seed']), 'test')
        if {r['patient'] for r in dataset.dataset.records} & (set(exp.metadata['train_ids'])|set(exp.metadata['val_ids'])):
            raise ValueError('TEST patient overlap')
        beat=Heartbeat(dest,settings['runtime']['heartbeat_seconds'])
        rows=[]
        with StageDisplay(dest,'C0 frozen EF TEST'),torch.no_grad():
            for samples in loader(dataset,settings,dict(micro_batch=8,num_workers=0),training=False):
                # Match the query comparison: FP32 frozen encoding, then BF16 readout.
                features=exp.features(samples)
                with amp_context(settings,device):
                    pred=exp.model.head(features,0)
                batch=exp._rows(pred,samples)
                for row,sample in zip(batch,samples):
                    row.update(recent_start=int(sample['recent_start']),frames=len(sample['video']),
                               prefix_frames=int(sample['prefix_frames']))
                rows.extend(batch)
                beat(phase='validation',completed=len(rows),total=len(dataset))
            metrics,patients=summarize(rows,'ef')
            deltas={}
            for arm in frozen.ARMS:
                frozen.matched_observations(rows,a.read(control/'test'/f'{arm}_observations.json'))
                deltas[f'{arm}_minus_C0']=frozen.paired(patients,a.read(control/'test'/f'{arm}_patients.json'))
            write_json(dest/'observations.json',rows)
            write_json(dest/'patients.json',patients)
            csv_rows(dest/'patients.csv',patients)
            test=dict(metrics=metrics,selected_epoch=summary['selected_epoch'],checkpoint_sha256=sha,
                      backbone_digest=backbone,manifest_sha256=manifest['manifest_sha256'],
                      wall_seconds=time.perf_counter()-started,split='TEST',evaluation_only=True,
                      precision='FP32 backbone, inherited BF16 head',shared_cache_bytes=0,
                      ssl_data_exposure=exp.metadata['ssl_data_exposure'])
            write_json(dest/'summary.json',test)
            combined=dict(val=prior['val'],test=dict(prior['test'],
                          metrics=dict(C0=metrics,**prior['test']['metrics']),C0=test,
                          paired_by_arm=dict(C2_minus_C1=prior['test']['paired'],**deltas)),
                          decision=prior['decision'])
            combined['val']['C0_scope']='Original frozen C0 VAL plus separately sealed, matching C0 TEST; no retraining'
            write_json(dest/'comparison_all.json',combined)
            mark_done(dest,['selection_lock.json','observations.json','patients.json','patients.csv',
                            'summary.json','comparison_all.json'])
            beat(force=True,phase='done')
    return combined


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-fs/outputs_ef_spatial_readout'))
    parser.add_argument('--cache-root',type=Path,default=Path('/root/autodl-tmp/outputs_ef_spatial_readout_cache'))
    parser.add_argument('--data-root',type=Path,default=Path('/root/autodl-tmp/datasets/EchoNet-Dynamic'))
    parser.add_argument('--tag',default='ar400_spatial_ef_20261116')
    parser.add_argument('--a2-root',type=Path,default=Path('/root/autodl-fs/outputs_cr_stage_a2'))
    parser.add_argument('--a2-tag',default='stage_a2_recurrent_20261114')
    args=parser.parse_args()
    safe_name(args.tag)
    safe_name(args.a2_tag)
    result=evaluate(args)
    print({arm:metrics['mae_pp'] for arm,metrics in result['test']['metrics'].items()})


if __name__=='__main__':
    main()
