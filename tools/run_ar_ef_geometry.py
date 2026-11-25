"""Locked A-R EF five-window replay plus automatic segmentation geometry."""

import argparse
import copy
import csv
import json
from pathlib import Path
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.artifacts import code_identity, digest, file_hash, mark_done, run_lock, verify_done, write_json
from research_pipeline.config import safe_name
from tools.evaluate_patch8_test import locked_checkpoint
from tools.run_ar_ef_pairing import cohort, dataset, reduce_results, uniform_starts
from tools.run_patch8_stage_a import lock_selection, read, verify_inputs


def verify_models(args):
    models = {s:locked_checkpoint(args.stage_a_root,args.stage_a_tag,s)
              for s in ('ef_l1_ft','seg_full_coverage_ft')}
    epochs = (21,48)
    source = []
    for (stage,(directory,summary,path,request)),epoch in zip(models.items(),epochs):
        protocol = verify_inputs(directory)
        stamp = file_hash(request['source'])
        cfg = request['config']
        if (summary['selected_epoch']!=epoch or summary.get('smoke') or summary.get('synthetic')
            or stamp!=protocol['source'] or not request['source'].endswith('/epoch_0400.pt')
            or protocol['inputs']['native_config'].get('forward_variant')!='mid_read'
            or (cfg['data']['local_frames'],cfg['data']['recent_frames'],cfg['data']['prefix'])!=(16,64,0)):
            raise ValueError(f'{stage}: require sealed original A-R FT endpoint')
        source.append(stamp)
    parent = args.e0
    if not verify_done(parent/'e0_val'):
        raise ValueError('Missing sealed E0 VAL')
    e0 = read(parent/'e0_val'/'summary.json')
    protocol = read(parent/'protocol.json')
    cfg = models['ef_l1_ft'][3]['config']
    manifest = read(cfg['data']['manifest'])
    if (len(set(source))!=1 or protocol['ssl_sha256']!=source[0] or e0['epoch']!=21
        or e0['checkpoint_sha256']!=file_hash(models['ef_l1_ft'][2])
        or protocol['manifest_sha256']!=file_hash(cfg['data']['manifest'])
        or 'test' in manifest or Path(manifest['root']).resolve()!=args.data_root.resolve()):
        raise ValueError('E0 and the segmentation/EF runs must agree on cohort and A-R source')
    return models,manifest


def windows(source, name):
    with (source/(name+'_windows.csv')).open(encoding='utf-8',newline='') as handle:
        return [dict(patient=r['patient'],start=int(r['recent_start']),fps=float(r['fps']),
                     target=float(r['target']),prediction=float(r['prediction'])) for r in csv.DictReader(handle)]


def locked_windows(directory, data):
    single,five = (windows(directory,name) for name in ('single','five'))
    truth = cohort(data)
    recs = [data.get_record(i) for i in range(len(data))]
    for policy,rows in [('single',single),('five',five)]:
        current = [(r['patient'],r['start'],r['target'],r['fps']) for r in rows]
        expected = [(r['patient'],s,truth[r['patient']],r['fps']) for r in recs for s in
                    ([r['recent_start']] if policy=='single' else uniform_starts(r['frames']))]
        if current!=expected:
            raise ValueError(f'{policy}: E0 patient/order/target/actual source windows do not match')
    from research_pipeline.metrics import summarize
    for name,rows in [('single',single),('five',five)]:
        expected = read(directory/'summary.json')['metrics'][name]['mae_pp']
        if abs(summarize(rows,'ef')[0]['mae_pp']-expected)>1e-9:
            raise ValueError(f'{name}: stored E0 predictions did not reproduce reported MAE')
    return single,five


def open_model(info, device):
    import torch
    from research_pipeline.engine import build_experiment
    _,summary,path,request = info
    model = build_experiment(request['config'],request['source'],device).model
    payload = torch.load(path,map_location='cpu',weights_only=False)
    if payload['epoch']!=summary['selected_epoch'] or file_hash(payload['source'])!=file_hash(request['source']):
        raise ValueError('Checkpoint contents disagree with its locked source or epoch')
    model.load_state_dict(payload['model'],strict=True)
    return model.requires_grad_(False).eval()


def samples(manifest, split, seed, rows):
    return dataset(manifest,split,seed,records=[dict(patient=r['patient'],recent_start=r['start'],H=0) for r in rows])


def visual_inputs(dataset, start, end):
    # No outcomes, clinical measures or annotations survive into the model.
    keys = ('patient','video','frame_indices','fps','prefix_frames','recent_start')
    return [{k:r[k] for k in keys} for r in (dataset[i] for i in range(start,end))]


def source_video(batch, device):
    import torch
    for r in batch:
        expected = torch.arange(r['recent_start'],r['recent_start']+64)
        if len(r['video'])!=64 or r['prefix_frames'] or not torch.equal(r['frame_indices'],expected):
            raise ValueError('Only the original unpadded, consecutive W64 source indices are allowed')
    return torch.stack([r['video'] for r in batch]).to(device)


def decode_window(model, video, decode_batch, audit=False):
    """One chronological pass; decode each native tubelet only once."""
    import torch
    maps,areas,confidence,hist = [],[],[],None
    b = model.backbone
    if (b.local_frames,b.tubelet_size,b.frame_readout,model.exit_name,model.append_boundary)!=(16,2,'repeat','final',False):
        raise ValueError('Dense adapter requires the original A-R segmentation exit')
    grid = b.token_grid[1]
    for clip in video.split(16,dim=1):
        out = b.stream_clip(clip,hist)
        hist = out['final_state']
        native = out['frame_outputs']
        if audit:
            torch.testing.assert_close(native[:,::2],native[:,1::2],atol=0,rtol=0)
        tokens = native[:,::2].reshape(len(video)*8,grid*grid,b.embed_dim)
        a,q,logits = [],[],[]
        for chunk in tokens.split(decode_batch):
            logit = model.head(chunk,grid)
            margin = logit[:,1].float()-logit[:,0].float()
            fg = margin>0
            count = fg.sum((1,2))
            a.append(count)
            q.append((margin.sigmoid()*fg).sum((1,2))/count.clamp_min(1))
            if audit:
                logits.append(logit)
        areas.append(torch.cat(a).reshape(len(video),8))
        confidence.append(torch.cat(q).reshape(len(video),8))
        if audit:
            clip_maps = torch.cat(logits)
            maps.append(clip_maps.reshape(len(video),8,*clip_maps.shape[1:]))
    if audit:
        return torch.cat(maps,dim=1)
    values = torch.stack((torch.cat(areas,1),torch.cat(confidence,1)),dim=1).cpu().tolist()
    return values


def replay(models, manifest, split, seed, single, five, device):
    import torch
    from research_pipeline.engine import amp_context
    cfg = models['ef_l1_ft'][3]['config']
    ef = open_model(models['ef_l1_ft'],device)
    report = {}
    with torch.inference_mode():
        for name,rows in [('single',single),('five',five)]:
            chosen = rows[:8]  # two original batches; full CSV/cohort checks run separately
            ds = samples(manifest,split,seed,chosen)
            pred = []
            for i in range(0,len(chosen),4):
                batch = visual_inputs(ds,i,min(i+4,len(chosen)))
                source_video(batch,'cpu')
                with amp_context(cfg,device):
                    pred += [v*100 for v in ef(batch).float().tolist()]
            error = [abs(p-r['prediction']) for p,r in zip(pred,chosen)]
            if max(error)>.8:
                raise ValueError(f'{name}: locked E0 forward replay diverged: {max(error)} pp')
            report[name] = dict(windows=len(error),max_abs_pp=max(error))
    return report


def equivalence(seg, batch, device):
    import torch
    with torch.inference_mode():
        video = source_video(batch,device)
        dense = decode_window(seg,video,8,audit=True)
        maximum, disagreements = 0.,0
        for pos in (0,1,15,16,31,32,47,48,63):
            # Compare against the original target-frame route, also across clip boundaries.
            reference = seg([dict(r,target_index=pos) for r in batch]).float()
            candidate = dense[:,pos//2].float()
            torch.testing.assert_close(candidate,reference,rtol=2e-3,atol=2e-3)
            maximum = max(maximum,float((candidate-reference).abs().max()))
            disagreements += int((candidate.argmax(1)!=reference.argmax(1)).sum())
        if disagreements:
            raise ValueError('Batched dense route did not reproduce original hard segmentation')
    return dict(max_abs_logit=maximum,mask_disagreements=disagreements,positions=[0,1,15,16,31,32,47,48,63])


def extract(seg, manifest, split, seed, rows, dest, device, batch, decode_batch):
    import torch
    from research_pipeline.engine import Heartbeat
    from tools.watch_pipeline_progress import StageDisplay
    ds = samples(manifest,split,seed,rows)
    groups = []
    beat = Heartbeat(dest,5)
    started = time.perf_counter()
    with StageDisplay(dest,'predicted LV area '+split),torch.inference_mode():
        for left in range(0,len(ds),128):
            right = min(left+128,len(ds))
            path = dest/f'area_{left:05d}.json'
            key = digest([(r['patient'],r['start'],r['fps']) for r in rows[left:right]])
            if path.exists():
                part = read(path)
                if part['key']!=key or len(part['rows'])!=right-left:
                    raise ValueError('Existing partial area shard is incompatible')
                group = part['rows']
            else:
                group = []
                for offset in range(left,right,batch):
                    values = visual_inputs(ds,offset,min(offset+batch,right))
                    v = source_video(values,device)
                    for x,(a,q) in zip(values,decode_window(seg,v,decode_batch)):
                        group.append(dict(patient=x['patient'],start=int(x['recent_start']),
                                          fps=float(x['fps']),area=[int(k) for k in a],confidence=q))
                    beat(phase='feature_cache',split=split,completed=left+len(group),total=len(ds))
                write_json(path,dict(key=key,rows=group))
            groups.extend(group)
        beat(force=True,phase='done',completed=len(groups),total=len(ds))
    return groups,time.perf_counter()-started


def develop_rows(five, areas):
    from collections import defaultdict
    from statistics import mean
    from research_pipeline.ef_geometry import case_geometry
    by_case = defaultdict(list)
    for r,a in zip(five,areas,strict=True):
        if (r['patient'],r['start'],r['fps'])!=(a['patient'],a['start'],a['fps']):
            raise ValueError('Area rows must match every E0 source window')
        by_case[r['patient']].append((r,a))
    return [dict(patient=p,target=g[0][0]['target'],G0=mean(r['prediction'] for r,_ in g),
                 **case_geometry([a for _,a in g])) for p,g in sorted(by_case.items())]


def annotations(data, areas, outcomes):
    """Post-prediction diagnosis; unannotated cycles are not considered errors."""
    from collections import defaultdict
    from statistics import median
    from utils.final_temporal_data import _trace_mask
    by_case = defaultdict(list)
    pairs = {r['patient']:r['beats'] for r in outcomes}
    for row in areas:
        by_case[row['patient']].append(row)
    result = []
    for patient,values in sorted(by_case.items()):
        c = data.cases[patient]
        if len(c['traces'])!=2:
            continue
        ed,es = sorted(c['traces'],key=lambda t:t['frame'])
        ea,sa = int(_trace_mask(ed).sum()),int(_trace_mask(es).sum())
        i,j = ed['frame'],es['frame']
        if not ea>sa:
            continue
        covered = [r for r in values if r['start']<=i<j<r['start']+64]
        if not covered:
            continue
        oracle = [(r['area'][(i-r['start'])//2],r['area'][(j-r['start'])//2]) for r in covered]
        match = [p for p in pairs[patient] if max(abs(p['ed']-i),abs(p['es']-j))/c['fps']<=.16]
        best = min(match,key=lambda p:abs(p['ed']-i)+abs(p['es']-j),default=None)
        result.append(dict(patient=patient,windows=len(covered),true_ed_area=ea,true_es_area=sa,
             ref_time_ed_abs_px=median(abs(a-ea) for a,b in oracle),ref_time_es_abs_px=median(abs(b-sa) for a,b in oracle),
             auto_ed_abs_px=abs(best['ed_area']-ea) if best else None,
             auto_es_abs_px=abs(best['es_area']-sa) if best else None,
             ed_abs_ms=abs(best['ed']-i)*1000/c['fps'] if best else None,
             es_abs_ms=abs(best['es']-j)*1000/c['fps'] if best else None))
    return result


def choose_runtime(seg, ds, dest, device, smoke):
    import torch
    path = dest/'runtime.json'
    if path.exists():
        return read(path)['selected']
    trials = []
    with torch.inference_mode():
        for micro, decode in ([(2,16)] if smoke or device.type=='cpu' else [(2,16),(4,32),(8,64),(12,48)]):
            if len(ds)<micro:
                continue
            frames = source_video(visual_inputs(ds,0,micro),device)
            started = None
            try:
                decode_window(seg,frames,decode)
                if device.type=='cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for _ in range(3):
                    decode_window(seg,frames,decode)
                elapsed = time.perf_counter()-started
                peak = torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0
                trials.append(dict(micro=micro,decode=decode,status='ok',windows_per_second=3*micro/elapsed,
                                   peak_reserved_bytes=peak))
            except torch.cuda.OutOfMemoryError:
                trials.append(dict(micro=micro,decode=decode,status='oom'))
            finally:
                del frames
                if device.type=='cuda':
                    torch.cuda.empty_cache()
    valid = [x for x in trials if x['status']=='ok']
    if not valid:
        raise ValueError('No safe area extraction candidate; inspect GPU resources')
    best = max(valid,key=lambda r:r['windows_per_second'])
    write_json(path,dict(selected=best,trials=trials,scope='fixed inference math, cached input, no label use'))
    print(json.dumps(dict(phase='area_runtime',selected=best)),flush=True)
    return best


def save_split(dest, data, inputs, outputs, fits, timing, *, annotation=None):
    from research_pipeline.engine import csv_rows
    from tools.run_ar_ef_pairing import population_strata
    arms = ('G0','Gcal','Gauto')
    groups = {k:v for k,v in population_strata(data).items() if k.startswith('EF_')}
    groups['auto_available'] = {r['patient'] for r in outputs if r['g'] is not None}
    groups['fallback'] = {r['patient'] for r in outputs if r['fallback']}
    rows = {arm:[dict(patient=r['patient'],target=r['target'],prediction=r[arm]) for r in outputs] for arm in arms}
    summary, patients = reduce_results(rows,groups,[('G0','Gcal'),('G0','Gauto'),('Gcal','Gauto')])
    for arm in arms:
        csv_rows(dest/f'{arm}_patients.csv',patients[arm])
    details = []
    for r in outputs:
        row = {k:v for k,v in r.items() if k!='beats'}
        row.update(beats=len(r['beats']),Gauto_minus_G0_abs_pp=abs(r['Gauto']-r['target'])-abs(r['G0']-r['target']),
                   Gauto_minus_Gcal_abs_pp=abs(r['Gauto']-r['target'])-abs(r['Gcal']-r['target']))
        details.append(row)
    csv_rows(dest/'paired_cases.csv',details)
    write_json(dest/'beats.json',[dict(patient=r['patient'],beats=r['beats']) for r in inputs])
    if annotation is not None:
        write_json(dest/'annotation_diagnostic.json',annotation)
        if annotation:
            csv_rows(dest/'annotation_diagnostic.csv',annotation)
    summary.update(eligible_cases=len(outputs),available_cases=len(groups['auto_available']),
                   fallback_cases=len(groups['fallback']),deduplicated_beats=sum(len(r['beats']) for r in outputs),
                   raw_pairs=sum(r['raw_pairs'] for r in outputs),wall=timing,fits=fits,
                   diagnosis_count=len(annotation) if annotation is not None else None,
                   matched_annotation_count=sum(r['ed_abs_ms'] is not None for r in annotation) if annotation is not None else None)
    write_json(dest/'summary.json',summary)
    files = ['summary.json','paired_cases.csv','beats.json',*(f'{arm}_patients.csv' for arm in arms)]
    if annotation is not None:
        files.append('annotation_diagnostic.json')
    mark_done(dest,files)
    print(json.dumps(dict(phase='split_done',path=str(dest),mae={a:summary['metrics'][a]['mae_pp'] for a in arms},
                          available=summary['available_cases'],cases=len(outputs))),flush=True)
    return summary


def run(args):
    import torch
    from research_pipeline.cli import configure
    from research_pipeline.ef_geometry import RULE, case_geometry, coefficients, cross_fitted, predict
    from utils.final_temporal_data import build_manifest
    models,manifest = verify_models(args)
    if args.action=='plan':
        print(json.dumps(dict(checkpoints={k:str(v[2]) for k,v in models.items()},rule=RULE,
                              test_labels_opened=False,selection='VAL cross-fit diagnostic; final calibration locked before TEST')))
        return
    smoke = args.action=='smoke'
    device = torch.device(args.device)
    if args.action=='run' and device.type!='cuda':
        raise ValueError('Formal run requires the supervised CUDA checkpoint precision')
    cfg = copy.deepcopy(models['seg_full_coverage_ft'][3]['config'])
    cfg['runtime']['device'] = args.device
    configure(cfg)
    dest = args.root/(args.tag+('_smoke' if smoke else ''))
    binding = dict(code=code_identity([__file__]),rule=RULE,models={k:file_hash(v[2]) for k,v in models.items()},
                   manifest=digest(manifest),source_e0=str(args.e0),
                   e0_val_sha256=file_hash(args.e0/'e0_val'/'DONE.json'),smoke=smoke,
                   seed=cfg['data']['seed'],device=args.device,torch=str(torch.__version__))
    if device.type=='cuda':
        binding['cuda_device'] = torch.cuda.get_device_name(device)
    with run_lock(dest):
        lock_selection(dest/'protocol.json',binding)
        if verify_done(dest):
            print(dest/'comparison.json')
            return
        if shutil.disk_usage(dest).free<2*1024**3:
            raise OSError('Need 2 GiB free reserve for results and resumable area shards')
        seed = cfg['data']['seed']
        val = dataset(manifest,'val',seed)
        first,all_five = locked_windows(args.e0/'e0_val',val)
        five = all_five
        if smoke:
            val = dataset(manifest,'val',seed,limit=12)
            ids = set(cohort(val))
            first,five = [[r for r in arm if r['patient'] in ids] for arm in (first,five)]
        replay_report = replay(models,manifest,'val',seed,first,five,device)
        seg = open_model(models['seg_full_coverage_ft'],device)
        ds = samples(manifest,'val',seed,five)
        equiv = equivalence(seg,visual_inputs(ds,0,2),device)
        runtime = choose_runtime(seg,ds,dest,device,smoke)
        batch, decode = runtime['micro'],runtime['decode']
        write_json(dest/'preflight.json',dict(ef_replay=replay_report,dense_equivalence=equiv,runtime=runtime))
        train = dataset(manifest,'train',seed,limit=4 if smoke else 48)
        train_windows = [dict(patient=r['patient'],start=s,fps=r['fps'])
                         for r in (train.get_record(i) for i in range(len(train))) for s in uniform_starts(r['frames'])]
        train_areas,seconds = extract(seg,manifest,'train',seed,train_windows,dest/'train_panel',device,batch,decode)
        by_case = {}
        for row in train_areas:
            by_case.setdefault(row['patient'],[]).append(row)
        geom = {p:case_geometry(rows) for p,rows in by_case.items()}
        write_json(dest/'train_panel'/'summary.json',dict(cases=len(geom),covered=sum(r['g'] is not None for r in geom.values()),
                   windows=len(train_areas),wall_seconds=seconds,policy='Fixed predeclared rule; no TRAIN EF fitting and no rule search'))
        va,seconds = extract(seg,manifest,'val',seed,five,dest/'val'/'areas',device,batch,decode)
        val_rows = develop_rows(five,va)
        out,folds = cross_fitted(val_rows)
        cand = coefficients(val_rows)
        val_summary = save_split(dest/'val',val,val_rows,out,dict(oof_folds=folds),
                                 dict(area_seconds=seconds,windows=len(va)),
                                 annotation=annotations(val,va,val_rows))
        result = dict(val_oof=val_summary,test=None,
                      val_scope='Five-fold patient-level development diagnostic; not independent validation',
                      fitting_population='matching automatically covered VAL patients in each fold; no geometry -> G0 in both arms')
        delta = val_summary['pairs']['Gauto_minus_Gcal']['candidate_minus_reference']
        gain = -val_summary['pairs']['Gauto_minus_G0']['candidate_minus_reference']
        result['val_decision'] = dict(Gauto_gain_vs_G0=gain,Gauto_minus_Gcal=delta,
                                      signal=gain>=.10 and delta<0,no_auto_followup=True)
        if not smoke:
            lock_selection(dest/'TEST_LOCK.json',dict(final_fit=cand,rule_sha256=digest(RULE),checkpoints=binding['models'],
                selected_on='VAL-only final fit; no TEST tuning',val_decision=result['val_decision']))
            prior = args.e0/'e0_test'
            if (not verify_done(prior) or read(prior/'summary.json')['checkpoint_sha256']!=binding['models']['ef_l1_ft']
                or read(args.e0/'TEST_LOCK.json')['ft_sha256']!=binding['models']['ef_l1_ft']):
                raise ValueError('E0 TEST source changed')
            tm = build_manifest(args.data_root,seed=seed,recent_frames=64,local_frames=16,max_prefix=0,splits=('test',))
            if read(args.e0/'test_manifest_lock.json')['sha256']!=tm['manifest_sha256']:
                raise ValueError('TEST manifest changed relative to locked E0')
            test = dataset(tm,'test',seed)
            if set(cohort(test)) & (set(cohort(val))|{r['patient'] for r in manifest['train']}):
                raise ValueError('Supervised split overlap')
            test_single,test_five = locked_windows(prior,test)
            test_replay = replay(models,tm,'test',seed,test_single,test_five,device)
            ta,seconds = extract(seg,tm,'test',seed,test_five,dest/'test'/'areas',device,batch,decode)
            test_rows = develop_rows(test_five,ta)
            result['test'] = save_split(dest/'test',test,test_rows,[predict(r,cand) for r in test_rows],
                                        dict(locked=cand),dict(area_seconds=seconds,windows=len(ta),ef_replay=test_replay))
        write_json(dest/'comparison.json',result)
        files = ['protocol.json','preflight.json','runtime.json','comparison.json','val/DONE.json','train_panel/summary.json']
        if not smoke:
            files += ['TEST_LOCK.json','test/DONE.json']
        mark_done(dest,files)
        print(dest/'comparison.json',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','smoke','run'])
    parser.add_argument('--tag',type=safe_name,required=True)
    parser.add_argument('--device',default='cuda')
    defaults = dict(stage_a_root='/root/autodl-fs/outputs_cr_stage_a',
                    e0='/root/autodl-fs/outputs_ar_ef_pairing/result/ef_pair_20261125/ef_pairing',
                    data_root='/root/autodl-tmp/datasets/EchoNet-Dynamic',
                    root='/root/autodl-fs/outputs_ar_ef_geometry/result')
    for k,v in defaults.items():
        parser.add_argument('--'+k.replace('_','-'),type=Path,default=Path(v))
    parser.add_argument('--stage-a-tag',type=safe_name,default='patch8_stage_a_v1_AR')
    run(parser.parse_args())


if __name__=='__main__':
    main()
