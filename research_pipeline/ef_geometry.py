"""Label-free same-window contraction extraction and matched L1 fusion."""

import hashlib

import numpy as np
from scipy import sparse
from scipy.optimize import linprog


RULE = dict(min_seconds=.16, max_seconds=.65, cold_groups=2, min_excursion=.12,
            min_area=32, max_area=4500, min_fg_probability=.72, duplicate_seconds=.10,
            boundary_min_jump=.18, boundary_jump_factor=2.5)


def contractions(row):
    """One source timestamp per native tubelet. No annotations are inputs."""
    area = np.asarray(row['area'], dtype=float)
    conf = np.asarray(row['confidence'], dtype=float)
    fps = float(row['fps'])
    if area.shape != (32,) or conf.shape != (32,) or fps <= 0 or not np.isfinite([*area,*conf,fps]).all():
        raise ValueError('Require 32 finite native tubelet observations and positive FPS')
    good = (area >= RULE['min_area']) & (area <= RULE['max_area']) & (conf >= RULE['min_fg_probability'])
    good[:RULE['cold_groups']] = False
    smoothed = np.convolve(np.pad(area,1,mode='edge'),[.25,.5,.25],mode='valid')
    # Split at suspicious state/context discontinuities instead of explaining them as contractions.
    jumps = np.abs(np.diff(area))/np.maximum(1,np.maximum(area[:-1],area[1:]))
    boundaries = {7,15,23}
    broken = {i for i in boundaries if jumps[i] >= max(RULE['boundary_min_jump'],
               RULE['boundary_jump_factor']*np.median(jumps[[j for j in range(i-2,i+3) if j!=i]]))}
    for i in broken:
        good[i:i+2] = False
    peaks = [i for i in range(2,len(area)-1) if good[i-1:i+2].all()
             and smoothed[i-1]<smoothed[i]>=smoothed[i+1]]
    valleys = [i for i in range(2,len(area)-1) if good[i-1:i+2].all()
               and smoothed[i-1]>smoothed[i]<=smoothed[i+1]]
    pairs = []
    for ed in peaks:
        for es in valleys:
            elapsed = (es-ed)*2/fps
            if not RULE['min_seconds']<=elapsed<=RULE['max_seconds'] or not good[ed:es+1].all():
                continue
            if any(ed<other<es and smoothed[other]>=.95*smoothed[ed] for other in peaks):
                continue
            g = 100*(1-area[es]/area[ed])
            if g < 100*RULE['min_excursion']:
                continue
            pairs.append(dict(ed=int(row['start'])+2*ed+.5, es=int(row['start'])+2*es+.5,
                              ed_area=area[ed], es_area=area[es], g=g,
                              quality=float(np.min(conf[ed:es+1])), start=int(row['start'])))
            break
    # A nearby shoulder on the same descent is not a second beat.
    pairs.sort(key=lambda p:(-p['g'],-p['quality'],p['ed']))
    kept = []
    for pair in pairs:
        if not any(pair['ed']<p['es'] and p['ed']<pair['es'] for p in kept):
            kept.append(pair)
    return sorted(kept,key=lambda p:p['ed']), len(broken)


def deduplicate(pairs, fps):
    """Cluster near-identical source times across overlapping windows, then vote once per beat."""
    clusters = []
    tolerance = RULE['duplicate_seconds']*fps
    for p in sorted(pairs,key=lambda q:(q['ed'],q['es'],q['start'])):
        compatible = [group for group in clusters if all(
            max(abs(p['ed']-v['ed']),abs(p['es']-v['es'])) <= tolerance for v in group)]
        if compatible:
            group = min(compatible,key=lambda g:abs(p['ed']-np.median([v['ed'] for v in g])))
            group.append(p)
        else:
            clusters.append([p])
    # Keep an actual observed same-window pair as each representative, never combine end points.
    reps = []
    for group in clusters:
        ed,es,g = [np.median([p[key] for p in group]) for key in ('ed','es','g')]
        chosen = min(group,key=lambda p:(abs(p['ed']-ed)+abs(p['es']-es),abs(p['g']-g),-p['quality'],p['start']))
        reps.append(dict(chosen, observations=len(group), g_median=float(g)))
    return reps


def case_geometry(rows):
    if not rows or any(r['fps']!=rows[0]['fps'] for r in rows):
        raise ValueError('Require same-case windows with consistent FPS')
    found = [contractions(r) for r in rows]
    raw = [p for pairs,_ in found for p in pairs]
    beats = deduplicate(raw,rows[0]['fps'])
    return dict(g=float(np.median([p['g_median'] for p in beats])) if beats else None,
                raw_pairs=len(raw), beats=beats, windows=len(rows),
                discontinuities=sum(count for _,count in found))


def coefficients(rows):
    """Fit both arms on exactly the same covered cases. No outcome-dependent exclusions."""
    covered = [r for r in rows if r['g'] is not None]
    if len(covered) < 12:
        return dict(Gcal=None,Gauto=None,fitted_n=len(covered))
    ef = np.array([r['G0'] for r in covered])
    y = np.array([r['target'] for r in covered]) - ef
    g = np.array([r['g'] for r in covered])
    design = np.stack([np.ones(len(y)),(ef-50)/10,(g-50)/10],axis=1)
    result = {}
    for name,width in [('Gcal',2),('Gauto',3)]:
        x = sparse.csr_matrix(design[:,:width])
        if np.linalg.matrix_rank(x.toarray()) < width:
            return dict(Gcal=None,Gauto=None,fitted_n=len(covered))
        neg_eye = -sparse.eye(len(y),format='csr')
        fitted = linprog(np.r_[np.zeros(width),np.ones(len(y))],
                         A_ub=sparse.bmat([[x,neg_eye],[-x,neg_eye]],format='csr'), b_ub=np.r_[y,-y],
                         bounds=[(None,None)]*width+[(0,None)]*len(y),method='highs')
        if not fitted.success:
            raise RuntimeError(f'L1 calibration failed: {fitted.message}')
        result[name] = fitted.x[:width].tolist()
    return dict(result,fitted_n=len(covered))


def predict(row, coeff):
    result = dict(row, Gcal=row['G0'], Gauto=row['G0'], fallback=True)
    if row['g'] is not None and coeff['Gauto'] is not None:
        x = np.array([1,(row['G0']-50)/10,(row['g']-50)/10])
        for name in ('Gauto','Gcal'):
            result[name] = float(np.clip(row['G0']+np.dot(x[:len(coeff[name])],coeff[name]),0,100))
        result['fallback'] = False
    return result


def cross_fitted(rows):
    ordered = sorted(rows,key=lambda r:hashlib.sha256(f"42:{r['patient']}".encode()).hexdigest())
    if len({r['patient'] for r in rows}) != len(rows):
        raise ValueError('Cross-fitting requires one row per patient')
    out, fits = [], []
    for fold in range(5):
        coeff = coefficients([r for i,r in enumerate(ordered) if i%5!=fold])
        fits.append(coeff)
        out.extend(dict(predict(r,coeff),fold=fold) for i,r in enumerate(ordered) if i%5==fold)
    return sorted(out,key=lambda r:r['patient']), fits
