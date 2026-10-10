"""Deterministic, run-free decisions for the final temporal study v2.

EF is MAE in percentage points (lower is better); Dice is a fraction (higher
is better). Error deltas default to real-state MSE minus zero-slot MSE (negative
is better). The legacy native R5 zero-minus-true convention is also supported
when explicitly labelled or identifiable from its reported component errors.
Missing metrics are None, never zero. CI crossing is
descriptive, not a veto. No models, trainers, or datasets are imported here.

select_reference returns a JSON-ready dict. Branch triggers return (bool,
reason). Explicit closure records belong in decisions['questions']['Q1.1'],
etc.; the report does not infer completed research from aggregate scores.
"""

from __future__ import annotations

from collections.abc import Mapping
import csv
import json
import math
from pathlib import Path
import re


_EPS = 1e-12
_DOMAINS = ('anatomy_linear_score', 'content_identity_accuracy', 'memory_prediction_delta')
_OBS_ALIASES = {
    'anatomy_linear_score': ('anatomy_F', 'anatomy_linear_score'),
    'content_identity_accuracy': ('F_real_margin', 'content_identity_accuracy'),
    'memory_prediction_delta': ('memory_prediction_delta',),
}
_QUESTIONS = {
    'Q1.1': ('Cache-external history', 'Effects apply within matched C_h cohorts; H0 still has recent-window memory.'),
    'Q1.2': ('State organization', 'Capacity and organization are confounded; unverified mixed roles are not clinical causality.'),
    'Q1.3': ('Read and write placement', 'The decision applies to the measured organization/frame background, not every combination.'),
    'Q1.4': ('Update and low-rank candidate', 'Only the registered candidate-rank intervention is assessed; gate means are not percentages of history used.'),
    'Q2.1': ('Calibrated factorized trade-off', 'A fitted endpoint is budget-specific; recovery by unperformed fitting is not assumed.'),
    'Q2.2': ('Exit and bottleneck', 'Same-forward base/final exits must be distinguished; a unique causal bottleneck need not be established.'),
    'Q2.3': ('Hard factorization versus shrink', 'Single-clip rank plots do not isolate shared low-rank or orthogonality contributions.'),
    'Q2.4': ('Single soft correction', 'B7 is at most one registered beta=0.5 branch; no corrective search loop.'),
    'Q3.1': ('Content identity', 'Static offset separation is not content identity; FP32 real/repeat/swap evidence is required.'),
    'Q3.2': ('Target anatomy', 'Only original sparse labels and matched last-16 positions are covered, not unlabelled dense phases.'),
    'Q3.3': ('History complement or interference', 'Fixed-head dependence and separately fitted capability are different claims; lengths cannot mix cohorts.'),
    'Q3.4': ('Streaming and bounded storage', 'Finite streaming checks do not establish clinical deployment, zero latency, or hour-long operation.'),
}
_BOUNDARIES = [
    'This is a development-selection report, not clinical validation or a clinical noninferiority claim.',
    'L16/W64 are the registered protocol. H128 is a diagnostic anchor, not a startup requirement or the default population.',
    'Historical effects are paired within each available C_h; no fabricated history or cross-cohort length curve.',
    'Clip-internal bidirectionality entails acquisition waiting; zero-latency, unlimited-duration, and time-loop claims are not established.',
    'Geometry, sink heatmaps, rank, CKA, and CI crossing alone neither promote nor veto a candidate.',
    'Dual is excluded. B8 and B7 are bounded optional branches, not permission to launch runs or reopen searches.',
    'Q4 remains pending. Conditional Q1-Q3 choices are not a completed MAIN or a verified integrated combination.',
    'Artifact information is caller-supplied provenance, not verification that checkpoints or experiments exist.',
]


def _number(value, name):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f'{name}: boolean is not a metric')
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{name}: expected a numeric metric') from exc
    return number if math.isfinite(number) else None


def _alias(metrics, keys, name):
    values = [_number(metrics[key], name) for key in keys if metrics.get(key) is not None]
    values = [value for value in values if value is not None]
    if not values:
        return None
    if any(not math.isclose(value, values[0], rel_tol=1e-9, abs_tol=_EPS) for value in values[1:]):
        raise ValueError(f'{name}: conflicting metric aliases')
    return values[0]


def task_vector(model, scores=None):
    """Return (ef_mae_pp, patient_dice_fraction), allowing None components.

    Accept a model's metrics dict directly or task_vector(model_id, scores).
    Percentage Dice is rejected rather than silently divided by 100.
    """
    record = model if scores is None else scores.get(model, {})
    if not isinstance(record, Mapping):
        raise ValueError('task_vector requires a model metrics mapping')
    ef = _alias(record.get('ef') or {}, ('mae_pp', 'mae'), 'EF MAE')
    dice = _alias(record.get('seg') or {}, ('dice_patient_mean', 'patient_dice'), 'patient Dice')
    if ef is not None and ef < 0:
        raise ValueError('EF MAE must be nonnegative percentage points')
    if dice is not None and not 0 <= dice <= 1:
        raise ValueError('Patient Dice must be a fraction in [0,1]')
    return ef, dice


def _patient_map(observations):
    if isinstance(observations, Mapping):
        rows = []
        for patient, value in observations.items():
            if not isinstance(value, Mapping):
                raise ValueError('Patient observations must map to metric dictionaries')
            if 'patient' in value and str(value['patient']) != str(patient):
                raise ValueError('Patient observation key conflicts with patient field')
            rows.append(dict(value, patient=str(patient)))
    elif isinstance(observations, (list, tuple)):
        rows = observations
    else:
        raise ValueError('Missing patient observations')
    result = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get('patient'):
            raise ValueError('Patient observation requires an explicit patient key')
        patient = str(row['patient'])
        if patient in result:
            raise ValueError('Duplicate patient key; aggregate within patient first')
        result[patient] = row
    if not result:
        raise ValueError('Empty patient observations')
    return result


def matched_patient_delta(reference, candidate, field):
    """Strict candidate-minus-reference mean; never silently intersect patients.

    Inputs are patient-keyed mappings or already patient-aggregated row lists.
    Every patient must have a finite field value. No CI or significance test is
    inferred, and duplicated patients/windows are not independent observations.
    """
    left, right = _patient_map(reference), _patient_map(candidate)
    if set(left) != set(right):
        raise ValueError('Patient keys differ; silent intersection is forbidden')
    deltas = []
    for patient in sorted(left):
        a, b = _number(left[patient].get(field), field), _number(right[patient].get(field), field)
        if a is None or b is None:
            raise ValueError(f'{field}: missing finite observation for {patient}')
        deltas.append(b - a)
    return dict(delta=math.fsum(deltas) / len(deltas), patients=len(deltas),
                patient_keys=sorted(left), positive_patients=sum(d > _EPS for d in deltas),
                negative_patients=sum(d < -_EPS for d in deltas))


def _representation(record):
    return record.get('representation') or {}


def _memory_sign(representation):
    """Convert an R5 delta to improvement-positive without guessing from sign."""
    report = representation.get('memory_prediction') or {}
    convention = representation.get('memory_prediction_delta_convention', report.get('delta_convention'))
    if convention is None:
        true = _number((report.get('true_state_error') or {}).get('mean'), 'true_state_error')
        zero = _number((report.get('zero_slots_error') or {}).get('mean'), 'zero_slots_error')
        delta = _number(representation.get('memory_prediction_delta'), 'memory_prediction_delta')
        # Native R5 currently exports zero-true; verify the identity rather than
        # interpreting every positive/negative number as a particular convention.
        if None not in (true, zero, delta) and abs(zero-true) > _EPS:
            if math.isclose(delta, zero-true, rel_tol=1e-9, abs_tol=_EPS):
                convention = 'zero_minus_true'
            elif not math.isclose(delta, true-zero, rel_tol=1e-9, abs_tol=_EPS):
                raise ValueError('R5 delta disagrees with reported component errors')
    if convention in (None, 'true_minus_zero', 'error_delta'):
        return -1
    if convention in ('zero_minus_true', 'improvement_positive'):
        return 1
    raise ValueError('Unknown memory_prediction_delta convention')


def _memory_observations(rows, sign):
    result = {}
    for patient, row in rows.items():
        value = _number(row.get('memory_prediction_delta'), 'memory_prediction_delta')
        result[patient] = dict(row, memory_prediction_delta=None if value is None else value*sign)
    return result


def _observed_pair(left, right, fields, observations_key='patient_observations'):
    """Pair each independently observed domain cohort, never their intersection.

    Prefer the registered field if present anywhere on either side. Do not
    switch aliases to conceal missing candidate values in that registered field.
    """
    coverage = dict(observations_key=observations_key, field=None, reference_total=0,
                    candidate_total=0, reference_observed=0, candidate_observed=0,
                    reference_patient_keys=[], candidate_patient_keys=[], paired_patients=0)
    try:
        maps, issues = [], []
        for record in (left, right):
            try:
                maps.append(_patient_map(record.get(observations_key)))
            except ValueError as exc:
                maps.append({}); issues.append(str(exc))
        a, b = maps
        coverage.update(reference_total=len(a), candidate_total=len(b))
        field = next((key for key in fields if any(key in row for row in [*a.values(), *b.values()])), None)
        coverage['field'] = field
        if field is None:
            return {}, {}, None, coverage, 'Registered domain field unavailable'
        aa = {p:row for p,row in a.items() if _number(row.get(field), field) is not None}
        bb = {p:row for p,row in b.items() if _number(row.get(field), field) is not None}
        coverage.update(reference_observed=len(aa), candidate_observed=len(bb),
                        reference_patient_keys=sorted(aa), candidate_patient_keys=sorted(bb))
        if issues:
            return aa, bb, field, coverage, '; '.join(issues)
        if set(aa) != set(bb):
            return aa, bb, field, coverage, 'Patient keys differ in observed domain cohort; silent intersection is forbidden'
        if not aa:
            return aa, bb, field, coverage, 'No finite observed domain cohort'
        coverage['paired_patients'] = len(aa)
        return aa, bb, field, coverage, None
    except ValueError as exc:
        return {}, {}, None, coverage, str(exc)


def _memory_identity_issue(left, right, dedicated):
    reports = [record.get('memory_prediction') or {} for record in (left, right)]
    keys = ('cohort_hash', 'target_projection_hash')
    if dedicated or any(report.get(key) is not None for report in reports for key in keys):
        for key in keys:
            a, b = (report.get(key) for report in reports)
            if not isinstance(a, str) or not a or not isinstance(b, str) or not b:
                return f'R5 missing required {key} on one or both sides'
            if a != b:
                return f'R5 {key} differs; cohort/projection pairing is required'
    return None


def _evidence(reference, candidate):
    """Require concordant aggregate AND strictly matched patient support.

    Near-tie promotion needs two readable domains, including anatomy and at
    least one of content/information. All observed domains must be non-worse;
    a positive matched mean also needs a majority of patients improving.
    Each domain uses its own finite observed cohort, requiring identical sets
    independently on both sides. A union's unobserved patients are not GT.
    Dedicated R5 observations also require identical cohort/projection hashes.
    Missing domains remain unavailable and cannot substitute for evidence.
    """
    left, right = _representation(reference), _representation(candidate)
    result = dict(supported=False, domains={}, reasons=[])
    gains, regressions = [], []
    for domain in _DOMAINS:
        av, bv = _number(left.get(domain), domain), _number(right.get(domain), domain)
        entry = dict(reference=av, candidate=bv, aggregate_delta=None, paired=None, supported=False)
        result['domains'][domain] = entry
        dedicated = domain == 'memory_prediction_delta' and any('memory_patient_observations' in r for r in (left, right))
        observation_key = 'memory_patient_observations' if dedicated else 'patient_observations'
        aa, bb, field, coverage, issue = _observed_pair(left, right, _OBS_ALIASES[domain], observation_key)
        entry['paired_coverage'] = coverage
        if av is None or bv is None:
            entry['reason'] = 'Aggregate R metric unavailable'
            continue
        if domain != 'memory_prediction_delta' and not (0 <= av <= 1 and 0 <= bv <= 1):
            raise ValueError(f'{domain} must be in [0,1]')
        if domain == 'memory_prediction_delta':
            identity_issue = _memory_identity_issue(left, right, dedicated)
            if identity_issue:
                issue = identity_issue
                coverage['paired_patients'] = 0
            sa, sb = _memory_sign(left), _memory_sign(right)
            av, bv = av*sa, bv*sb
            aa, bb = _memory_observations(aa, sa), _memory_observations(bb, sb)
            entry['comparison_convention'] = 'improvement_positive; original scores preserved'
        entry['aggregate_delta'] = bv - av
        if issue:
            entry['reason'] = issue
            result['reasons'].append(f'{domain}: {issue}')
        else:
            try:
                entry['paired'] = matched_patient_delta(aa, bb, field)
            except ValueError as exc:
                entry['reason'] = str(exc)
            paired = entry['paired']
            entry['supported'] = bool(paired and bv - av > _EPS and paired['delta'] > _EPS
                                      and paired['positive_patients'] > paired['patients'] / 2)
        if entry['supported']:
            gains.append(domain)
        if bv < av - _EPS or (entry['paired'] and entry['paired']['delta'] < -_EPS):
            regressions.append(domain)
    result['supported'] = ('anatomy_linear_score' in gains and len(gains) >= 2 and not regressions)
    result['gains'], result['regressions'] = gains, regressions
    result['reasons'].append('Concordant matched anatomy plus content/information support'
                             if result['supported'] else 'No concordant matched multi-domain support')
    return result


def _tolerances(ef, dice):
    if any(_number(v, 'tolerance') is None or v < 0 for v in (ef, dice)):
        raise ValueError('Tolerances must be finite and nonnegative')


def _role(model_id, record):
    role = record.get('memory_mode') or (record.get('config') or {}).get('memory_mode')
    if role:
        return role
    token = re.split(r'[_-]', str(model_id))[0]
    return {'B0': 'spatial', 'B1': 'global', 'B8': 'spatial_global'}.get(token)


def _slots(model_id, record):
    value = record.get('memory_slots', (record.get('cost') or {}).get('memory_slots'))
    if value is not None:
        value = _number(value, 'memory_slots')
        if value is None or value < 1 or int(value) != value:
            raise ValueError('memory_slots must be a positive observed integer')
        return int(value)
    # These are protocol-defined capacities, not invented measured runtimes.
    return {'spatial': 16, 'global': 1, 'spatial_global': 17}.get(_role(model_id, record))


def _dual(model_id, record):
    return _role(model_id, record) == 'dual' or bool(re.search(r'(^|[_-])dual([_-]|$)', str(model_id), re.I))


def _cohort_issue(reference, candidate, tasks=('ef', 'seg')):
    for task in tasks:
        a, b = reference.get(task) or {}, candidate.get(task) or {}
        if 'patient_keys' not in a and 'patient_keys' not in b:
            continue
        x, y = a.get('patient_keys'), b.get('patient_keys')
        if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
            return f'{task}: patient keys supplied on only one side'
        if len(set(x)) != len(x) or len(set(y)) != len(y) or set(x) != set(y) or not x:
            return f'{task}: patient keys differ or are duplicated/empty'
    return None


def _resolve_candidates(models, entries, scores, ef_tolerance, dice_tolerance):
    gains = {model: (entries[model]['ef_gain'], entries[model]['dice_gain']) for model in models}

    def dominates(a, b):
        return (all(x >= y - _EPS for x, y in zip(gains[a], gains[b]))
                and any(x > y + _EPS for x, y in zip(gains[a], gains[b])))

    frontier = [model for model in models if not any(dominates(other, model) for other in models)]
    if len(frontier) == 1:
        return frontier[0], 'Unique Pareto candidate in the highest qualifying decision class'
    pairs = [(a, b) for i, a in enumerate(frontier) for b in frontier[i + 1:]]
    for a, b in pairs:
        ef, dice = gains[a][0] - gains[b][0], gains[a][1] - gains[b][1]
        if (ef * dice < 0 and abs(ef) > _EPS and abs(dice) > _EPS
                and abs(ef) >= ef_tolerance - _EPS and abs(dice) >= dice_tolerance - _EPS):
            return None, 'Retain registered reference: unresolved material EF/Dice trade-off on the Pareto frontier; no composite score'

    def materially_better(a, b):
        ef, dice = gains[a][0] - gains[b][0], gains[a][1] - gains[b][1]
        return ((ef > _EPS and ef >= ef_tolerance - _EPS and dice >= -dice_tolerance - _EPS)
                or (dice > _EPS and dice >= dice_tolerance - _EPS and ef >= -ef_tolerance - _EPS))

    winners = [a for a in frontier if all(a == b or materially_better(a, b) for b in frontier)]
    if len(winners) == 1:
        return winners[0], 'Material advantage on one task; other-task differences remain within tolerance'
    near = all(abs(gains[a][0] - gains[b][0]) <= ef_tolerance + _EPS
               and abs(gains[a][1] - gains[b][1]) <= dice_tolerance + _EPS for a, b in pairs)
    readable = {model: tuple(_number(_representation(scores[model]).get(key), key)
                             for key in _DOMAINS) for model in frontier}
    equal_r = all(readable[a] == readable[b] for a, b in pairs)
    capacities = {model: _slots(model, scores[model]) for model in frontier}
    if near and equal_r and all(value is not None for value in capacities.values()):
        smallest = min(capacities.values())
        cheapest = [model for model in frontier if capacities[model] == smallest]
        if len(cheapest) == 1:
            return cheapest[0], 'Task near tie and identical reported R metrics: uniquely fewer persistent memory slots; runtime advantage not inferred'
    if (equal_r and all(all(abs(x - y) <= _EPS for x, y in zip(gains[a], gains[b]))
                        and capacities[a] == capacities[b] for a, b in pairs)):
        return min(frontier), 'Exact task/R/capacity tie: lexical ID is an administrative tie-break, not a quality ranking'
    return None, 'Retain registered reference: unresolved frontier or insufficient evidence for a task/cost tie-break'


def select_reference(reference, candidateIds, scores, ef_tolerance=.3, dice_tolerance=.005):
    """Select against a fixed reference, independent of candidate input order.

    Reject any task degradation beyond tolerance first. Eligible priority is
    meaningful task gain, then concordant matched R gain, then the B0 -> B1
    smaller-state near-tie rule. Within the highest class use Pareto comparisons,
    then a material one-task advantage with only tolerated other-task differences.
    Opposing material advantages retain the registered reference. A slot-cost
    tie requires both task gaps within tolerance, identical reported R metrics,
    all capacities known, and a unique smaller state. Lexical ID breaks only
    exact task/R/capacity ties. Never combine EF and Dice into a score; CI is
    never a gate.
    Task cohort equality is checked when explicit patient_keys are supplied;
    otherwise common protocol/cohort compatibility remains a caller obligation.
    """
    _tolerances(ef_tolerance, dice_tolerance)
    base = scores.get(reference) or {}
    ref_ef, ref_dice = task_vector(base)
    result = dict(selected=reference, reference=reference, candidates={}, reasons=[],
                  ef_tolerance=ef_tolerance, dice_tolerance=dice_tolerance,
                  units=dict(ef='percentage_points', dice='fraction'), ci_policy='descriptive_only',
                  comparison_assumption='Task protocol/cohort compatibility is caller-supplied unless patient_keys are provided')
    if _dual(reference, base):
        raise ValueError('Dual cannot be a final-study reference')
    eligible = []
    for model in sorted(set(candidateIds)):
        if model == reference:
            continue
        score = scores.get(model) or {}
        ef, dice = task_vector(score)
        entry = dict(eligible=False, reason=None, ef_gain=None, dice_gain=None, evidence=None)
        result['candidates'][model] = entry
        if _dual(model, score):
            entry['reason'] = 'Dual is outside the v2 candidate set'
            continue
        if None in (ef, dice, ref_ef, ref_dice):
            entry['reason'] = 'Insufficient task scores; missing values are not success'
            continue
        issue = _cohort_issue(base, score)
        if issue:
            entry['reason'] = issue
            continue
        ef_gain, dice_gain = ref_ef - ef, dice - ref_dice
        entry.update(ef_gain=ef_gain, dice_gain=dice_gain)
        if ef_gain < -ef_tolerance - _EPS or dice_gain < -dice_tolerance - _EPS:
            entry['reason'] = 'Unacceptable task trade-off; geometry or R evidence cannot override task cost'
            continue
        meaningful = ((ef_gain > _EPS and ef_gain >= ef_tolerance - _EPS)
                      or (dice_gain > _EPS and dice_gain >= dice_tolerance - _EPS))
        if meaningful:
            priority, entry['reason'] = 3, 'Meaningful task improvement with acceptable other-task cost'
        else:
            evidence = _evidence(base, score)
            entry['evidence'] = evidence
            if evidence['supported']:
                priority, entry['reason'] = 2, 'Within task tolerances with matched content/anatomy/information gain'
            elif (_role(reference, base) == 'spatial' and _role(model, score) == 'global'
                  and _slots(model, score) < _slots(reference, base)
                  and not evidence.get('gains') and not evidence.get('regressions')):
                priority, entry['reason'] = 1, 'Organization near tie without observed quality/evidence gain: fewer global slots'
            else:
                entry['reason'] = 'No meaningful task gain or concordant matched functional evidence; geometry alone is insufficient'
                continue
        entry['eligible'] = True
        entry['decision_class'] = {3: 'task_gain', 2: 'matched_evidence', 1: 'organization_cost'}[priority]
        eligible.append((priority, model))
    if eligible:
        highest = max(priority for priority, _ in eligible)
        models = [model for priority, model in eligible if priority == highest]
        selected, reason = _resolve_candidates(models, result['candidates'], scores,
                                                ef_tolerance, dice_tolerance)
        if selected is not None:
            result['selected'] = selected
            result['reasons'].append(result['candidates'][selected]['reason'])
        result['reasons'].append(reason)
    else:
        result['reasons'].append('Retain reference; no qualifying candidate. This does not establish equivalence or clinical noninferiority.')
    return result


def _already_run(scores, model):
    record = scores.get(model) or {}
    return bool(record.get('attempted') is True or record.get('status') in ('complete', 'failed')
                or all(value is not None for value in task_vector(record)))


def _r2_ef_support(spatial, global_):
    """Lower fixed-exit R2 EF MAE plus strict per-patient error improvements.

    Canonical scalar: representation.ef_probe_mae_pp (aliases ef_probe_mae,
    ef_linear_mae_pp/ef_linear_mae). Native probes[task=ef,exit=F] are accepted.
    ef_probe_exit defaults to F and must match; never search for the best exit.
    Observations use ef_probe_error or the native ef_F_error (etc.) column.
    """
    left, right = _representation(spatial), _representation(global_)
    exit_name = left.get('ef_probe_exit', 'F')
    if exit_name != right.get('ef_probe_exit', 'F'):
        return False, 'R2 fixed exits differ'

    def aggregate(record):
        value = _alias(record, ('ef_probe_mae_pp', 'ef_probe_mae', 'ef_linear_mae_pp', 'ef_linear_mae'), 'R2 EF MAE')
        probes = [probe for probe in record.get('probes', [])
                  if probe.get('task') == 'ef' and probe.get('exit') == exit_name]
        if len(probes) > 1:
            raise ValueError('R2 requires one fixed-exit EF probe, not a layer/exit search')
        if probes:
            native = _alias(probes[0], ('mae_pp', 'mae'), 'R2 EF MAE')
            if value is not None and native is not None and not math.isclose(value, native, rel_tol=1e-9, abs_tol=_EPS):
                raise ValueError('Conflicting fixed R2 EF probe scores')
            value = value if value is not None else native
        if value is not None and value < 0:
            raise ValueError('R2 EF MAE must be nonnegative')
        return value

    a, b = aggregate(left), aggregate(right)
    if a is None or b is None or b >= a - _EPS:
        return False, 'No observed global fixed-R2 EF probe improvement'
    try:
        patients_a, patients_b, field, _, issue = _observed_pair(
            left, right, ('ef_probe_error', 'ef_linear_error', f'ef_{exit_name}_error'))
        if issue:
            return False, issue
        paired = matched_patient_delta(patients_a, patients_b, field)
    except ValueError as exc:
        return False, str(exc)
    return (paired['delta'] < -_EPS and paired['negative_patients'] > paired['patients'] / 2,
            'Strictly paired fixed-R2 EF probe errors')


def mixed_trigger(B0, B1, scores):
    """One B8 for matched R5/anatomy complement, or task trade-off plus R2.

    The R2 alternative requires global primary EF better and spatial primary
    Dice better, plus readable matched global R2 EF and spatial R anatomy
    increments. Complementary task signs alone never suffice. R5 remains a
    valid alternative, not a prerequisite for the R2 + task route.
    """
    if _already_run(scores, 'B8'):
        return False, 'B8 already attempted/completed; no mixed retry loop'
    spatial, global_ = scores.get(B0) or {}, scores.get(B1) or {}
    if _dual(B0, spatial) or _dual(B1, global_):
        return False, 'Dual is excluded'
    s_ef, s_dice = task_vector(spatial)
    g_ef, g_dice = task_vector(global_)
    if None in (s_ef, s_dice, g_ef, g_dice):
        return False, 'Insufficient B0/B1 task evidence'
    issue = _cohort_issue(spatial, global_)
    if issue:
        return False, issue
    functional = _evidence(spatial, global_)['domains'].get('memory_prediction_delta', {})
    anatomy = _evidence(global_, spatial)['domains'].get('anatomy_linear_score', {})
    if functional.get('supported') and anatomy.get('supported'):
        return True, 'Matched global R5 cache-external functional information and spatial anatomy increments are complementary; B8 allowed once, not selected or validated'
    complement = g_ef < s_ef - _EPS and s_dice > g_dice + _EPS
    r2, r2_reason = _r2_ef_support(spatial, global_) if complement else (False, 'No complementary primary global-EF/spatial-Dice signs')
    if complement and r2 and anatomy.get('supported'):
        return True, 'Primary global-EF/spatial-Dice trade-off plus strictly matched fixed-R2 functional and R anatomy support; B8 allowed once without requiring R5, not selected or validated'
    return False, ('No matched readable global-function/spatial-anatomy complement; '
                   f'{r2_reason}; sink heatmaps or geometry alone cannot trigger B8')


def _exit_metrics(exit_scores, model):
    entries = exit_scores.get(model) or {}
    if not isinstance(entries, Mapping):
        raise ValueError('exit_scores[model] must map exit names to metrics')
    result = {}
    for canonical, aliases in (('local_base', ('local_base', 'local')),
                               ('fused_base', ('fused_base', 'base')),
                               ('fused_final', ('fused_final', 'final'))):
        values = [entries[name] for name in aliases if name in entries]
        if len(values) > 1 and values[0] != values[1]:
            raise ValueError('Conflicting exit aliases')
        if values:
            value = values[0]
            result[canonical] = value if 'ef' in value else dict(ef=value)
    return result


def soft_trigger(Ccalib, Fcalib, Bref, B5, B6, scores, exit_scores):
    """One B7, not a loop: preserved gain, unresolved EF cost, hard/shrink control.

    Ccalib/Fcalib denote caller-certified common strong-calibration endpoints.
    The three existing F exits must have fitted EF results; missing exits cannot
    be called failed rescue. B5/B6 must be the registered same-background hard
    and shrink endpoints. This function does not verify training/checkpoints.
    """
    if _already_run(scores, 'B7'):
        return False, 'B7 already attempted/completed; no soft correction loop'
    models = [scores.get(model) or {} for model in (Ccalib, Fcalib, Bref, B5, B6)]
    if any(_dual(model, score) for model, score in zip((Ccalib, Fcalib, Bref, B5, B6), models)):
        return False, 'Dual is excluded'
    vectors = [task_vector(score) for score in models]
    if any(None in vector for vector in vectors):
        return False, 'Insufficient calibration or registered hard/shrink task scores'
    c, f, ref, hard, shrink = vectors
    for a, b in ((models[0], models[1]), (models[2], models[3]), (models[2], models[4])):
        issue = _cohort_issue(a, b)
        if issue:
            return False, issue
    if f[0] - c[0] <= .3 + _EPS:
        return False, 'Calibrated F has no unacceptable EF cost; no correction needed'
    exits = _exit_metrics(exit_scores or {}, Fcalib)
    for name, record in exits.items():
        value = task_vector(record)[0]
        issue = _cohort_issue(models[0], record, ('ef',))
        if issue:
            return False, f'{name}: {issue}'
        if value is not None and value - c[0] <= .3 + _EPS:
            return False, f'Existing {name} exit resolves EF cost; use a legal exit instead of B7'
    if any(name not in exits or task_vector(exits[name])[0] is None
           for name in ('local_base', 'fused_base', 'fused_final')):
        return False, 'Insufficient fitted existing-exit evidence; missing scores are not failed rescue'
    gain = f[1] - c[1] >= .005 - _EPS or _evidence(models[0], models[1])['supported']
    if not gain:
        return False, 'Calibrated F has no preserved meaningful anatomy/content gain'
    hard_gain = hard[1] - ref[1] >= .005 - _EPS or _evidence(models[2], models[3])['supported']
    consistent = (hard_gain and hard[0] - ref[0] > .3 + _EPS
                  and hard[0] - shrink[0] >= .3 - _EPS
                  and shrink[0] - ref[0] <= .3 + _EPS
                  and shrink[1] >= hard[1] - .005 - _EPS
                  and shrink[1] >= ref[1] - .005 - _EPS)
    if not consistent:
        return False, 'Registered B5/B6 evidence does not support a hard restriction cost relieved by shrink with anatomy retained'
    return True, 'Preserved calibrated F gain, EF cost after all fitted exits, and matched-background hard/shrink evidence permit one beta=0.5 B7; not proof of unique low-rank causality'


def _json_safe(value):
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f'Unsupported report value: {type(value).__name__}; provide JSON data')


def _display(value):
    if isinstance(value, str):
        return value
    return json.dumps(_json_safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False)


def write_final_report(result_root, decisions, scores, artifacts_info=None, *, artifacts=None):
    """Write a truthful deterministic closure bundle; return paths and summary.

    Each questions[Qx.y] entry supplies evidence, decision, boundary, closed.
    Negative results may close an objective. A missing component or explicit
    closed=False remains data_insufficiency_limited. Paths/metrics alone do not
    certify completion. No timestamp, experiments, or workbook writes occur.
    """
    if artifacts_info is not None and artifacts is not None:
        raise ValueError('Supply artifacts_info or artifacts, not both')
    artifact_data = _json_safe(artifacts_info if artifacts_info is not None else artifacts or {})
    decisions, scores = _json_safe(decisions), _json_safe(scores)
    supplied = decisions.get('questions', decisions.get('subquestions', decisions))
    questions = {}
    for key, (title, default_boundary) in _QUESTIONS.items():
        entry = supplied.get(key) or {}
        evidence, decision, boundary = entry.get('evidence'), entry.get('decision'), entry.get('boundary')
        requested = entry.get('closed') is True
        complete = bool(evidence and decision and boundary and requested)
        questions[key] = dict(title=title, evidence=evidence or [],
                              decision=decision or 'No evidence-backed decision supplied; retain registered reference provisionally.',
                              boundary=boundary or default_boundary, closed=complete,
                              status='objective_closed' if complete else 'data_insufficiency_limited')
    complete = all(entry['closed'] for entry in questions.values())
    summary = dict(version=2, status='objective_closed' if complete else 'data_insufficiency_limited',
                   questions=questions, decisions=decisions, scores=scores, artifacts=artifact_data,
                   q4=dict(status='pending', main_complete=False), boundaries=_BOUNDARIES,
                   metric_units=dict(ef='percentage_points', dice='fraction'),
                   missing_metric_policy='null/blank, not zero or inferred success')
    root = Path(result_root)
    root.mkdir(parents=True, exist_ok=True)
    report, comparison, decision_file = (root / name for name in ('Q1-Q3_closure.md', 'comparison.csv', 'decisions.json'))
    columns = ['model_id', 'ef_mae_pp', 'patient_dice_fraction', *_DOMAINS,
               'memory_slots', 'history_metrics', 'representation_patient_count']
    rows = []
    for model in sorted(scores):
        record = scores[model] or {}
        ef, dice = task_vector(record)
        representation = _representation(record)
        row = dict(model_id=model, ef_mae_pp=ef, patient_dice_fraction=dice,
                   memory_slots=_slots(model, record),
                   history_metrics=_display(record['history']) if record.get('history') else '')
        row.update({key: _number(representation.get(key), key) for key in _DOMAINS})
        observations = representation.get('patient_observations')
        row['representation_patient_count'] = len(_patient_map(observations)) if observations else None
        rows.append(row)
    lines = ['# Q1-Q3 Closure', '', f"Status: {summary['status']}.",
             'Q4: pending. MAIN: not complete.', '', '## Observed Comparisons', '',
             '| Model | EF MAE (pp) | Patient Dice (fraction) |', '|---|---:|---:|']
    for row in rows:
        model = row['model_id'].replace('|', '\\|').replace('\n', ' ')
        ef = 'unavailable' if row['ef_mae_pp'] is None else f"{row['ef_mae_pp']:.6g}"
        dice = 'unavailable' if row['patient_dice_fraction'] is None else f"{row['patient_dice_fraction']:.6g}"
        lines.append(f'| {model} | {ef} | {dice} |')
    for key, entry in questions.items():
        lines.extend(['', f"## {key} {entry['title']}", '',
                      f"Status: {entry['status']}; closed: {str(entry['closed']).lower()}.",
                      'Evidence: ' + (_display(entry['evidence']) if entry['evidence'] else 'Not supplied.'),
                      'Decision: ' + _display(entry['decision']), 'Boundary: ' + _display(entry['boundary'])])
    lines.extend(['', '## Selection Decisions', '', '```json',
                  json.dumps(decisions, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False), '```',
                  '', '## Claim Boundaries', ''])
    lines.extend('- ' + boundary for boundary in _BOUNDARIES)
    lines.extend(['', '## Supplied Artifact Information', '', '```json',
                  json.dumps(artifact_data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False), '```', ''])
    # Validate/serialize before writing any of the three final files.
    serialized = json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    report.write_text('\n'.join(lines), encoding='utf-8')
    with comparison.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    decision_file.write_text(serialized, encoding='utf-8')
    return dict(report=str(report.resolve()), comparison=str(comparison.resolve()),
                decisions=str(decision_file.resolve()), summary=summary)
