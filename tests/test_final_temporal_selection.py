"""Run-free decisions and file-backed reports, with no scientific-score defaults."""

import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from utils.final_temporal_selection import (
    task_vector, matched_patient_delta, select_reference, mixed_trigger,
    soft_trigger, write_final_report,
)


def score(ef=5.0, dice=.85, anatomy=None, content=None, memory=None, **extra):
    result = dict(ef=dict(mae_pp=ef), seg=dict(patient_dice=dice),
                  representation=dict(anatomy_linear_score=anatomy,
                                      content_identity_accuracy=content,
                                      memory_prediction_delta=memory), **extra)
    if any(v is not None for v in (anatomy, content, memory)):
        if memory is not None:
            result['representation']['memory_prediction_delta_convention'] = 'zero_minus_true'
        result['representation']['patient_observations'] = [
            dict(patient=p, anatomy_linear_score=anatomy,
                 content_identity_accuracy=content, memory_prediction_delta=memory)
            for p in ('patient_a', 'patient_b', 'patient_c')]
    return result


def soft_inputs():
    scores = dict(C=score(), F=score(5.5, .87), ref=score(),
                  hard=score(5.5, .872), shrink=score(5.1, .870))
    exits = dict(F=dict(local_base=dict(mae_pp=5.5), fused_base=dict(mae_pp=5.4),
                        fused_final=dict(mae_pp=5.5)))
    return scores, exits


class TaskAndSelectionTests(unittest.TestCase):
    def test_task_vector_aliases_missing_and_units(self):
        self.assertEqual(task_vector(dict(ef=dict(mae=6.1), seg=dict(dice_patient_mean=.86))), (6.1, .86))
        self.assertEqual(task_vector('model', dict(model=score())), (5.0, .85))
        self.assertEqual(task_vector({}), (None, None))
        self.assertEqual(task_vector(dict(ef=dict(mae=float('nan')))), (None, None))
        self.assertEqual(task_vector(dict(ef=dict(mae=None, mae_pp=5.1))), (5.1, None))
        for record in (dict(ef=dict(mae=5, mae_pp=6)), dict(seg=dict(patient_dice=85)),
                       dict(ef=dict(mae=-1)), dict(ef=dict(mae=True)),
                       dict(seg=dict(patient_dice=.85, dice_patient_mean=.9))):
            with self.subTest(record=record), self.assertRaises(ValueError):
                task_vector(record)

    def test_meaningful_task_gain_and_ci_crossing_not_veto(self):
        candidate = score(4.6, .849)
        candidate['ef']['paired_interval'] = [-.8, .2]
        scores = dict(ref=score(), candidate=candidate)
        result = select_reference('ref', ['candidate'], scores)
        self.assertEqual(result['selected'], 'candidate')
        self.assertEqual(result['candidates']['candidate']['decision_class'], 'task_gain')
        self.assertEqual(result['ci_policy'], 'descriptive_only')
        self.assertIn('acceptable', result['reasons'][0])

    def test_cost_veto_precedes_any_geometric_or_representation_signal(self):
        reference = score(anatomy=.7, content=.65, memory=.03)
        ef_good_seg_bad = score(4.3, .844, .9, .9, .8)
        seg_good_ef_bad = score(5.31, .90, .9, .9, .8)
        for candidate in (ef_good_seg_bad, seg_good_ef_bad):
            candidate['representation']['effective_rank'] = 99999
            result = select_reference('ref', ['bad'], dict(ref=reference, bad=candidate))
            self.assertEqual(result['selected'], 'ref')
            self.assertIn('Unacceptable', result['candidates']['bad']['reason'])
            self.assertFalse(result['candidates']['bad']['eligible'])

    def test_fraction_tolerance_boundary_and_meaningful_threshold(self):
        for candidate in (score(4.7, .845), score(5.3, .855)):
            result = select_reference('ref', ['candidate'], dict(ref=score(), candidate=candidate))
            self.assertEqual(result['selected'], 'candidate')
        result = select_reference('ref', ['candidate'], dict(ref=score(), candidate=score(4.7, .844999)))
        self.assertEqual(result['selected'], 'ref')
        with self.assertRaises(ValueError):
            select_reference('ref', [], {}, ef_tolerance=-.1)
        with self.assertRaises(ValueError):
            select_reference('ref', [], {}, dice_tolerance=float('nan'))

    def test_near_tie_requires_readable_matched_evidence_not_geometry(self):
        reference = score(anatomy=.7, content=.65, memory=.02)
        supported = score(5.05, .851, .75, .7, .04)
        supported['representation']['anatomy_linear_interval'] = dict(low=-.1, high=.2)
        result = select_reference('ref', ['candidate'], dict(ref=reference, candidate=supported))
        self.assertEqual(result['selected'], 'candidate')
        self.assertEqual(result['candidates']['candidate']['decision_class'], 'matched_evidence')
        geometry = score(5.0, .851)
        geometry['representation'].update(effective_rank=200, cka=.01, sink_removed=True)
        result = select_reference('ref', ['geometry'], dict(ref=reference, geometry=geometry))
        self.assertEqual(result['selected'], 'ref')
        aggregate_only = copy.deepcopy(supported)
        aggregate_only['representation'].pop('patient_observations')
        result = select_reference('ref', ['aggregate'], dict(ref=reference, aggregate=aggregate_only))
        self.assertEqual(result['selected'], 'ref')

    def test_missing_domains_not_defaulted_and_single_signal_insufficient(self):
        reference = score(anatomy=.70, content=.65)
        candidate = score(5.01, .851, .75, .70)
        result = select_reference('ref', ['candidate'], dict(ref=reference, candidate=candidate))
        self.assertEqual(result['selected'], 'candidate')
        domain = result['candidates']['candidate']['evidence']['domains']['memory_prediction_delta']
        self.assertIsNone(domain['reference'])
        self.assertIsNone(domain['candidate'])
        result = select_reference('ref', ['candidate'], dict(ref=score(anatomy=.70),
                                                           candidate=score(5.01, .851, anatomy=.75)))
        self.assertEqual(result['selected'], 'ref')

    def test_patient_keys_duplicates_missing_fields_and_no_intersection(self):
        reference = [dict(patient='a', value=.1), dict(patient='b', value=.3)]
        candidate = [dict(patient='b', value=.5), dict(patient='a', value=.2)]
        result = matched_patient_delta(reference, candidate, 'value')
        self.assertAlmostEqual(result['delta'], .15)
        self.assertEqual(result['patient_keys'], ['a', 'b'])
        self.assertEqual(result['patients'], 2)
        bad_inputs = [candidate[:1], candidate + [candidate[0]],
                      [dict(patient='a', value=None), candidate[0]],
                      [dict(patient='a', value=float('inf')), candidate[0]]]
        for invalid in bad_inputs:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                matched_patient_delta(reference, invalid, 'value')
        ref, cand = score(anatomy=.7, content=.7), score(anatomy=.8, content=.8)
        cand['representation']['patient_observations'].pop()
        result = select_reference('ref', ['candidate'], dict(ref=ref, candidate=cand))
        self.assertEqual(result['selected'], 'ref')
        self.assertIn('Patient keys differ', result['candidates']['candidate']['evidence']['reasons'][0])

    def test_mismatched_or_discordant_patient_evidence_cannot_promote(self):
        reference = score(anatomy=.7, content=.7)
        candidate = score(anatomy=.8, content=.8)
        for row in candidate['representation']['patient_observations']:
            row['anatomy_linear_score'] = .6
        result = select_reference('ref', ['candidate'], dict(ref=reference, candidate=candidate))
        self.assertEqual(result['selected'], 'ref')
        candidate = score(anatomy=.8, content=.8)
        for row in candidate['representation']['patient_observations'][:2]:
            row['anatomy_linear_score'] = .699
        result = select_reference('ref', ['candidate'], dict(ref=reference, candidate=candidate))
        self.assertEqual(result['selected'], 'ref')

    def test_real_representation_observation_aliases(self):
        reference = score(anatomy=.70, content=.65, memory=.02)
        candidate = score(anatomy=.75, content=.7, memory=.04)
        for record in (reference, candidate):
            for row in record['representation']['patient_observations']:
                row['anatomy_F'] = row.pop('anatomy_linear_score')
                row['F_real_margin'] = row.pop('content_identity_accuracy')
        result = select_reference('ref', ['candidate'], dict(ref=reference, candidate=candidate))
        self.assertEqual(result['selected'], 'candidate')

    def test_domain_observed_subsets_not_the_union_or_a_silent_intersection(self):
        reference, candidate = score(anatomy=.7,content=.65), score(anatomy=.8,content=.75)
        for record in (reference,candidate):
            r = record['representation']
            for i,row in enumerate(r['patient_observations']):
                row['anatomy_F'] = r['anatomy_linear_score'] if i<2 else None
                row['F_real_margin'] = r['content_identity_accuracy']
            # Different unobserved union members are not part of either domain.
            r['patient_observations'].append(dict(patient='extra_'+str(r['anatomy_linear_score']),
                                                  anatomy_F=None,F_real_margin=float('nan')))
        result = select_reference('ref',['candidate'],dict(ref=reference,candidate=candidate))
        self.assertEqual(result['selected'],'candidate')
        domains = result['candidates']['candidate']['evidence']['domains']
        self.assertEqual(domains['anatomy_linear_score']['paired_coverage']['field'],'anatomy_F')
        self.assertEqual(domains['anatomy_linear_score']['paired_coverage']['paired_patients'],2)
        self.assertEqual(domains['content_identity_accuracy']['paired_coverage']['paired_patients'],3)
        self.assertEqual(domains['anatomy_linear_score']['paired_coverage']['reference_total'],4)
        asymmetric = copy.deepcopy(candidate)
        asymmetric['representation']['patient_observations'][0]['anatomy_F'] = None
        result = select_reference('ref',['candidate'],dict(ref=reference,candidate=asymmetric))
        self.assertEqual(result['selected'],'ref')
        anatomy = result['candidates']['candidate']['evidence']['domains']['anatomy_linear_score']
        self.assertEqual(anatomy['paired_coverage']['reference_observed'],2)
        self.assertEqual(anatomy['paired_coverage']['candidate_observed'],1)
        self.assertEqual(anatomy['paired_coverage']['paired_patients'],0)
        self.assertIn('silent intersection',anatomy['reason'])

    def test_dedicated_r5_cohort_and_projection_hashes(self):
        reference,candidate = score(anatomy=.7,memory=.02),score(anatomy=.8,memory=.04)
        for record in (reference,candidate):
            r = record['representation']
            r['memory_patient_observations'] = [dict(patient='r5_only',memory_prediction_delta=r['memory_prediction_delta'])]
            r['memory_prediction'] = dict(cohort_hash='same_cohort',target_projection_hash='same_projection')
            for row in r['patient_observations']:
                row['memory_prediction_delta'] = None
        result = select_reference('ref',['candidate'],dict(ref=reference,candidate=candidate))
        self.assertEqual(result['selected'],'candidate')
        coverage = result['candidates']['candidate']['evidence']['domains']['memory_prediction_delta']['paired_coverage']
        self.assertEqual(coverage['observations_key'],'memory_patient_observations')
        self.assertEqual(coverage['paired_patients'],1)
        self.assertEqual(coverage['reference_patient_keys'],['r5_only'])
        for key in ('cohort_hash','target_projection_hash'):
            changed = copy.deepcopy(candidate)
            changed['representation']['memory_prediction'][key] = 'different'
            result = select_reference('ref',['candidate'],dict(ref=reference,candidate=changed))
            self.assertEqual(result['selected'],'ref')
            memory = result['candidates']['candidate']['evidence']['domains']['memory_prediction_delta']
            self.assertIn(key,memory['reason'])
            self.assertEqual(memory['paired_coverage']['paired_patients'],0)
            del changed['representation']['memory_prediction'][key]
            self.assertEqual(select_reference('ref',['candidate'],dict(ref=reference,candidate=changed))['selected'],'ref')
        changed = copy.deepcopy(candidate)
        changed['representation'].pop('memory_patient_observations')
        self.assertEqual(select_reference('ref',['candidate'],dict(ref=reference,candidate=changed))['selected'],'ref')
        changed = copy.deepcopy(candidate)
        changed['representation']['memory_patient_observations'][0]['memory_prediction_delta'] = None
        self.assertEqual(select_reference('ref',['candidate'],dict(ref=reference,candidate=changed))['selected'],'ref')

    def test_org_global_simplicity_near_tie_spatial_when_global_degraded(self):
        scores = dict(B0=score(), B1=score(5.1, .848))
        result = select_reference('B0', ['B1'], scores)
        self.assertEqual(result['selected'], 'B1')
        self.assertEqual(result['candidates']['B1']['decision_class'], 'organization_cost')
        self.assertIn('fewer global slots', result['reasons'][0])
        scores['B1'] = score(5.31, .85)
        self.assertEqual(select_reference('B0', ['B1'], scores)['selected'], 'B0')
        self.assertEqual(select_reference('B1', ['B0'], scores)['selected'], 'B0')
        scores = dict(B0=score(anatomy=.75, content=.7, memory=.04),
                      B1=score(5.0, .85, .7, .65, .02))
        self.assertEqual(select_reference('B0', ['B1'], scores)['selected'], 'B0')

    def test_deterministic_order_fixed_reference_and_no_mutation(self):
        scores = dict(ref=score(), z=score(4.4, .85), a=score(5, .858), bad=score(5.31, .9))
        original = copy.deepcopy(scores)
        expected = select_reference('ref', ['a', 'z', 'bad'], scores)
        self.assertEqual(expected['selected'], 'ref')
        self.assertIn('unresolved material EF/Dice trade-off', expected['reasons'][0])
        self.assertEqual(expected, select_reference('ref', ['bad', 'z', 'a', 'z'], scores))
        self.assertEqual(scores, original)
        self.assertEqual(select_reference('ref', ['a', 'b'], dict(ref=score(), a=score(4.6),
                                                               b=score(4.6)))['selected'], 'a')

    def test_pareto_joint_improvement_resolves_opposing_material_gains(self):
        scores = dict(ref=score(), ef_best=score(4.4, .85), seg_best=score(5, .858),
                      joint=score(4.3, .859))
        result = select_reference('ref', ['seg_best', 'joint', 'ef_best'], scores)
        self.assertEqual(result['selected'], 'joint')
        self.assertIn('Unique Pareto', result['reasons'][-1])
        self.assertNotIn('net', json.dumps(result))
        self.assertEqual(result, select_reference('ref', ['ef_best', 'seg_best', 'joint'], scores))

    def test_material_tradeoff_not_overridden_by_fewer_slots_or_r_geometry(self):
        scores = dict(ref=score(), global_=score(4.4, .85, memory_slots=1),
                      spatial=score(5, .858, memory_slots=16))
        scores['global_']['representation']['effective_rank'] = 999
        result = select_reference('ref', ['global_', 'spatial'], scores)
        self.assertEqual(result['selected'], 'ref')
        self.assertTrue(all(entry['eligible'] for entry in result['candidates'].values()))
        self.assertIn('no composite score', result['reasons'][0])

    def test_cost_tie_only_near_tasks_equal_reported_r_known_capacities(self):
        scores = dict(ref=score(), spatial=score(4.4, .85, memory_slots=16),
                      global_=score(4.5, .852, memory_slots=1))
        result = select_reference('ref', ['spatial', 'global_'], scores)
        self.assertEqual(result['selected'], 'global_')
        self.assertIn('uniquely fewer persistent memory slots', result['reasons'][-1])
        scores['global_'].pop('memory_slots')
        self.assertEqual(select_reference('ref', ['spatial', 'global_'], scores)['selected'], 'ref')
        scores['global_']['memory_slots'] = 1
        scores['global_']['representation']['anatomy_linear_score'] = .8
        self.assertEqual(select_reference('ref', ['spatial', 'global_'], scores)['selected'], 'ref')

    def test_single_task_material_advantage_before_cost_and_class_priority(self):
        scores = dict(ref=score(), ef_better=score(4.1, .85, memory_slots=16),
                      cheaper=score(4.6, .852, memory_slots=1))
        result = select_reference('ref', ['cheaper', 'ef_better'], scores)
        self.assertEqual(result['selected'], 'ef_better')
        self.assertIn('Material advantage on one task', result['reasons'][-1])
        reference = score(anatomy=.7, content=.65, memory=.02)
        candidate = score(5.01, .851, .8, .75, .04)
        result = select_reference('ref', ['evidence', 'task'],
                                  dict(ref=reference, evidence=candidate, task=score(4.5, .85)))
        self.assertEqual(result['selected'], 'task')
        self.assertEqual(result['candidates']['task']['decision_class'], 'task_gain')

    def test_task_cohort_keys_and_missing_tasks_and_dual(self):
        reference, candidate = score(), score(4.0)
        reference['ef']['patient_keys'] = ['a', 'b']
        candidate['ef']['patient_keys'] = ['a', 'c']
        result = select_reference('ref', ['candidate'], dict(ref=reference, candidate=candidate))
        self.assertEqual(result['selected'], 'ref')
        self.assertIn('patient keys differ', result['candidates']['candidate']['reason'])
        self.assertEqual(select_reference('ref', ['missing'], dict(ref=score(), missing={'ef': {'mae': 3}}))['selected'], 'ref')
        result = select_reference('B0', ['candidate_dual'], dict(B0=score(), candidate_dual=score(3, .9)))
        self.assertEqual(result['selected'], 'B0')
        with self.assertRaises(ValueError):
            select_reference('old', [], dict(old=score(memory_mode='dual')))


class BranchTests(unittest.TestCase):
    def test_negative_error_delta_is_better_and_native_legacy_is_explicit(self):
        scores = dict(B0=score(5.05, .86, .8, .7, -.02),
                      B1=score(5.0, .85, .75, .7, -.04))
        for record in scores.values():
            record['representation'].pop('memory_prediction_delta_convention')
        self.assertTrue(mixed_trigger('B0', 'B1', scores)[0])
        worse = copy.deepcopy(scores)
        r = worse['B1']['representation']; r['memory_prediction_delta'] = -.01
        for row in r['patient_observations']:
            row['memory_prediction_delta'] = -.01
        self.assertFalse(mixed_trigger('B0', 'B1', worse)[0])
        reference, candidate = score(anatomy=.7, memory=-.02), score(anatomy=.8, memory=-.04)
        for record in (reference, candidate):
            record['representation'].pop('memory_prediction_delta_convention')
        self.assertEqual(select_reference('ref', ['candidate'], dict(ref=reference, candidate=candidate))['selected'],
                         'candidate')
        missing = copy.deepcopy(candidate)
        missing['representation']['patient_observations'][0]['memory_prediction_delta'] = float('nan')
        self.assertEqual(select_reference('ref', ['candidate'], dict(ref=reference,candidate=missing))['selected'], 'ref')
        native = dict(B0=score(5.05, .86, .8, .7, .02), B1=score(5.0, .85, .75, .7, .04))
        for record in native.values():
            r = record['representation']; r.pop('memory_prediction_delta_convention')
            r['memory_prediction'] = dict(true_state_error={'mean': 1-r['memory_prediction_delta']},
                                          zero_slots_error={'mean': 1})
        self.assertTrue(mixed_trigger('B0', 'B1', native)[0])

    def r2_scores(self, native=False):
        scores = dict(B0=score(5.0, .87, anatomy=.8), B1=score(4.6, .85, anatomy=.75))
        for model, mae in (('B0', 6.0), ('B1', 5.5)):
            r = scores[model]['representation']
            if native:
                r['probes'] = [dict(task='ef', exit='F', mae=mae, low=mae-1, high=mae+1),
                               dict(task='ef', exit='H', mae=mae-2)]
            else:
                r['ef_probe_mae_pp'] = mae
            for row in r['patient_observations']:
                row['ef_F_error'] = mae
        return scores

    def test_mixed_primary_task_complement_plus_fixed_r2_without_r5(self):
        for native in (False, True):
            scores = self.r2_scores(native)
            before = copy.deepcopy(scores)
            trigger, reason = mixed_trigger('B0', 'B1', scores)
            self.assertTrue(trigger)
            self.assertIn('without requiring R5', reason)
            self.assertIn('fixed-R2', reason)
            self.assertEqual(scores, before)
            # Selection must not trade away material Dice solely for EF.
            self.assertEqual(select_reference('B0', ['B1'], scores)['selected'], 'B0')

    def test_mixed_complementary_tasks_alone_or_unpaired_r2_do_not_trigger(self):
        scores = dict(B0=score(5.0, .87), B1=score(4.6, .85))
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores = self.r2_scores()
        scores['B1']['representation']['patient_observations'].pop()
        trigger, reason = mixed_trigger('B0', 'B1', scores)
        self.assertFalse(trigger)
        self.assertIn('Patient keys differ', reason)
        scores = self.r2_scores()
        for row in scores['B1']['representation']['patient_observations']:
            row['ef_F_error'] = 6.5
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores = self.r2_scores()
        scores['B1']['representation']['patient_observations'][0]['ef_F_error'] = None
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])

    def test_mixed_r2_requires_spatial_anatomy_and_primary_task_signs(self):
        scores = self.r2_scores()
        scores['B0']['representation']['anatomy_linear_score'] = .74
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores = self.r2_scores()
        scores['B1']['ef']['mae_pp'] = 5.1
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores = self.r2_scores()
        scores['B1']['seg']['patient_dice'] = .88
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])

    def test_mixed_r2_fixed_exit_not_best_exit_and_no_retry(self):
        scores = self.r2_scores(native=True)
        scores['B1']['representation']['probes'][0]['mae'] = 6.1
        # The H row is better but cannot rescue a fixed F comparison.
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores = self.r2_scores()
        scores['B1']['representation']['ef_probe_exit'] = 'H'
        trigger, reason = mixed_trigger('B0', 'B1', scores)
        self.assertFalse(trigger)
        self.assertIn('fixed exits differ', reason)
        scores = self.r2_scores()
        scores['B8'] = dict(attempted=True)
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])

    def test_mixed_r2_scalar_aliases_and_conflicts(self):
        scores = self.r2_scores()
        for record in scores.values():
            r = record['representation']
            r['ef_linear_mae'] = r.pop('ef_probe_mae_pp')
        self.assertTrue(mixed_trigger('B0', 'B1', scores)[0])
        scores['B1']['representation']['ef_probe_mae_pp'] = 7
        with self.assertRaisesRegex(ValueError, 'conflicting metric aliases'):
            mixed_trigger('B0', 'B1', scores)

    def test_mixed_only_matched_function_anatomy_complement(self):
        scores = dict(B0=score(5.05, .86, .8, .7, .02),
                      B1=score(5.0, .85, .75, .7, .04))
        trigger, reason = mixed_trigger('B0', 'B1', scores)
        self.assertTrue(trigger)
        self.assertIn('allowed once', reason)
        before = copy.deepcopy(scores)
        self.assertEqual(mixed_trigger('B0', 'B1', scores), (trigger, reason))
        self.assertEqual(scores, before)
        scores['B8'] = dict(status='complete')
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])

    def test_mixed_sink_only_missing_or_dominant_single_organization_skip(self):
        scores = dict(B0=score(), B1=score())
        scores['B0']['representation']['sink_heatmap'] = 'pretty.png'
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores = dict(B0=score(anatomy=.8, content=.7, memory=.02),
                      B1=score(anatomy=.8, content=.75, memory=.04))
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])
        scores['B0']['representation']['anatomy_linear_score'] = .85
        scores['B0']['representation']['patient_observations'].pop()
        self.assertFalse(mixed_trigger('B0', 'B1', scores)[0])

    def test_soft_complete_registered_evidence_and_bounded_retry(self):
        scores, exits = soft_inputs()
        original = copy.deepcopy((scores, exits))
        trigger, reason = soft_trigger('C', 'F', 'ref', 'hard', 'shrink', scores, exits)
        self.assertTrue(trigger)
        self.assertIn('beta=0.5', reason)
        self.assertIn('not proof', reason)
        self.assertEqual((scores, exits), original)
        scores['B7'] = dict(attempted=True, status='failed')
        self.assertFalse(soft_trigger('C', 'F', 'ref', 'hard', 'shrink', scores, exits)[0])

    def test_soft_base_exit_rescue_skips_even_if_other_exits_missing(self):
        scores, _ = soft_inputs()
        for exits in (dict(F=dict(base=dict(mae_pp=5.2))),
                      dict(F=dict(local=dict(ef=dict(mae=5.0)))),
                      dict(F=dict(final=dict(mae_pp=5.3)))):
            trigger, reason = soft_trigger('C', 'F', 'ref', 'hard', 'shrink', scores, exits)
            self.assertFalse(trigger)
            self.assertIn('resolves EF cost', reason)

    def test_soft_missing_evidence_no_geometry_or_fake_success(self):
        scores, exits = soft_inputs()
        invalid = []
        for model, field, value in (('F', 'seg', {'patient_dice': .851}),
                                    ('F', 'ef', {'mae_pp': 5.2}),
                                    ('hard', 'ef', {'mae_pp': 5.2}),
                                    ('shrink', 'ef', {'mae_pp': 5.4}),
                                    ('shrink', 'seg', {'patient_dice': .84}),
                                    ('shrink', 'ef', {})):
            changed = copy.deepcopy(scores)
            changed[model][field] = value
            changed[model]['representation'].update(rank=999, sink_removed=True)
            invalid.append(changed)
        for changed in invalid:
            with self.subTest(changed=changed):
                self.assertFalse(soft_trigger('C', 'F', 'ref', 'hard', 'shrink', changed, exits)[0])
        for missing in ({}, dict(F=dict(base={'mae': 5.5})),
                        dict(F=dict(local={'mae': 5.5}, base={'mae': None}, final={'mae': 5.5}))):
            trigger, reason = soft_trigger('C', 'F', 'ref', 'hard', 'shrink', scores, missing)
            self.assertFalse(trigger)
            self.assertIn('Insufficient fitted', reason)

    def test_soft_exit_aliases_and_ef_only_cohort_matching(self):
        scores, exits = soft_inputs()
        for task in ('ef', 'seg'):
            scores['C'][task]['patient_keys'] = ['a', 'b']
            scores['F'][task]['patient_keys'] = ['a', 'b']
        exits = dict(F={name: dict(mae_pp=5.5, patient_keys=['a', 'b']) for name in ('local', 'base', 'final')})
        self.assertTrue(soft_trigger('C', 'F', 'ref', 'hard', 'shrink', scores, exits)[0])
        exits['F']['base']['patient_keys'] = ['a', 'c']
        trigger, reason = soft_trigger('C', 'F', 'ref', 'hard', 'shrink', scores, exits)
        self.assertFalse(trigger)
        self.assertIn('patient keys differ', reason)


class ReportTests(unittest.TestCase):
    def test_three_files_all_twelve_sections_missing_scores_and_determinism(self):
        with tempfile.TemporaryDirectory() as temp:
            decisions = dict(organization=select_reference('B0', ['B1'], dict(B0=score(), B1=score())),
                             questions={'Q1.2': dict(evidence=['Task near tie; fewer registered slots'],
                                                    decision='Use global conditionally',
                                                    boundary='No integrated MAIN verified', closed=True)})
            scores = dict(B0=score(), B1=score(), absent=dict(representation={'memory_prediction_delta': None}))
            artifacts = dict(checkpoint='not_verified.pt', runtime_seconds=None, tested_h=[0, 16, 64, 128])
            result = write_final_report(temp, decisions, scores, artifacts)
            self.assertEqual({p.name for p in Path(temp).iterdir()},
                             {'Q1-Q3_closure.md', 'comparison.csv', 'decisions.json'})
            summary = json.loads(Path(result['decisions']).read_text(encoding='utf-8'))
            self.assertEqual(len(summary['questions']), 12)
            self.assertEqual(summary['status'], 'data_insufficiency_limited')
            self.assertEqual(summary['q4'], dict(status='pending', main_complete=False))
            self.assertTrue(summary['questions']['Q1.2']['closed'])
            self.assertFalse(summary['questions']['Q1.1']['closed'])
            for entry in summary['questions'].values():
                self.assertEqual(set(entry), {'title', 'evidence', 'decision', 'boundary', 'closed', 'status'})
            report = Path(result['report']).read_text(encoding='utf-8')
            self.assertIn('H128 is a diagnostic anchor', report)
            self.assertIn('Q4: pending. MAIN: not complete.', report)
            self.assertIn('Dual is excluded', report)
            self.assertIn('not clinical validation', report)
            self.assertIn('caller-supplied provenance', report)
            for q in summary['questions']:
                self.assertEqual(report.count(f'## {q} '), 1)
            with Path(result['comparison']).open(newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
            absent = next(row for row in rows if row['model_id'] == 'absent')
            self.assertEqual(absent['ef_mae_pp'], '')
            self.assertEqual(absent['patient_dice_fraction'], '')
            self.assertEqual(absent['memory_prediction_delta'], '')
            before = {p.name: p.read_bytes() for p in Path(temp).iterdir()}
            write_final_report(temp, decisions, dict(reversed(list(scores.items()))), artifacts=artifacts)
            self.assertEqual(before, {p.name: p.read_bytes() for p in Path(temp).iterdir()})

    def test_complete_objectives_can_close_negative_results_not_main(self):
        questions = {f'Q{q}.{i}': dict(evidence={'observed_gain': False, 'scope': 'budget complete'},
                                     decision='Retain reference / do not adopt',
                                     boundary='No positive mechanism claim', closed=True)
                     for q in (1, 2, 3) for i in (1, 2, 3, 4)}
        with tempfile.TemporaryDirectory() as temp:
            result = write_final_report(temp, dict(questions=questions), {})
            summary = result['summary']
            self.assertEqual(summary['status'], 'objective_closed')
            self.assertTrue(all(entry['closed'] for entry in summary['questions'].values()))
            self.assertFalse(summary['q4']['main_complete'])
            self.assertEqual(summary['q4']['status'], 'pending')

    def test_incomplete_closure_flags_not_inferred_and_nan_is_missing(self):
        decisions = {'Q1.1': dict(evidence=['measured'], decision='retain', closed=True),
                     'Q1.2': dict(evidence=['measured'], decision='retain', boundary='limited'),
                     'Q1.3': dict(evidence=[], decision='retain', boundary='limited', closed=True)}
        scores = dict(missing=dict(ef={'mae': float('nan')}, seg={'patient_dice': None}))
        with tempfile.TemporaryDirectory() as temp:
            result = write_final_report(temp, decisions, scores, dict(checkpoint=Path('unknown.pt')))
            self.assertTrue(all(not entry['closed'] for entry in result['summary']['questions'].values()))
            self.assertIsNone(result['summary']['scores']['missing']['ef']['mae'])
            self.assertNotIn('NaN', Path(result['decisions']).read_text())
            with self.assertRaises(ValueError):
                write_final_report(temp, {}, {}, artifacts_info={}, artifacts={})

    def test_invalid_metrics_fail_before_writing_final_files(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                write_final_report(temp, {}, dict(bad=dict(seg={'patient_dice': 86})))
            self.assertEqual(list(Path(temp).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
