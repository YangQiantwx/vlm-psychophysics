"""Finite allocation acceptance, including a read-only frozen-source integration.

CVPR_BENCHMARK_ROOT enables the integration tests against the existing local
source metadata. No video decode, generation, model answers or remote calls.
"""
from copy import deepcopy
import os
from pathlib import Path
import unittest

import numpy as np

from classic_precision import natural_t2_rebalance as rebalance


def synthetic():
    deltas = (.05,.06,.07,.08,.09,.10,.11,.12,.13,.14,.15,.16,.17,.18,.19,.20,.25,.30,.35,.40)
    rng = np.random.default_rng(5)
    values = rng.normal(size=(20,24,18))
    values -= values.mean(axis=1,keepdims=True)
    features = values.reshape(480,18)
    features[3*24,15] = -20
    features[13*24,15] = 20
    blocks = [dict(block=i,delta_Y=deltas[i//24],object_pair=f'pair_{i:03}') for i in range(480)]
    return blocks,features,deltas


class SelectionTests(unittest.TestCase):
    def test_single_exchange_restores_gates_without_changing_global_moments(self):
        blocks,features,deltas=synthetic();original=features.copy()
        before,mu,sd=rebalance.standardized(features)
        self.assertGreater(abs(before[:,12:]).max(),.5)
        result=rebalance.choose_single_swap(blocks,features,deltas,set())
        self.assertLessEqual(abs(result['after']).max(),.5+1e-12)
        np.testing.assert_array_equal(features,original)
        np.testing.assert_allclose(result['revised_features'].mean(0),mu,atol=1e-14)
        np.testing.assert_allclose(result['revised_features'].std(0),sd,atol=1e-14)
        changed=np.flatnonzero(np.any(features!=result['revised_features'],axis=1))
        self.assertEqual(changed.tolist(),sorted(result['blocks']))
        self.assertEqual({blocks[i]['delta_Y'] for i in changed},set(rebalance.TARGETS))
        np.testing.assert_array_equal(result['revised_features'][:,:8],features[:,:8])

    def test_selection_is_deterministic(self):
        args=synthetic()
        a=rebalance.choose_single_swap(*args,set());b=rebalance.choose_single_swap(*args,set())
        self.assertEqual(a['blocks'],b['blocks'])
        self.assertEqual(a['standardized_movement_L2'],b['standardized_movement_L2'])

    def test_forbidden_objects_cannot_enter_candidate(self):
        blocks,features,deltas=synthetic()
        with self.assertRaisesRegex(ValueError,'No admissible single exchange'):
            rebalance.choose_single_swap(blocks,features,deltas,{b['object_pair'] for b in blocks})

    def test_duplicate_objects_in_target_cell_are_rejected(self):
        blocks,features,deltas=synthetic()
        for i in range(24):blocks[13*24+i]['object_pair']=blocks[3*24+i]['object_pair']
        with self.assertRaisesRegex(ValueError,'No admissible single exchange'):
            rebalance.choose_single_swap(blocks,features,deltas,set())

    def test_original_photometric_gate_cannot_be_bypassed(self):
        blocks,features,deltas=synthetic()
        features[:24,0]+=10  # BG gate failure outside the authorized target cells.
        with self.assertRaisesRegex(ValueError,'No admissible single exchange'):
            rebalance.choose_single_swap(blocks,features,deltas,set())


@unittest.skipUnless(os.environ.get('CVPR_BENCHMARK_ROOT'),'Set CVPR_BENCHMARK_ROOT for source-bound integration')
class CurrentSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['CVPR_BENCHMARK_ROOT'])
        cls.plan=rebalance.build_plan(cls.root)
        cls.inputs=rebalance.load_inputs(cls.root)

    def test_current_candidate_has_minimal_complete_semantic_exchange(self):
        p=self.plan
        self.assertEqual(p['selected_exchange']['blocks'],[93,330])
        self.assertEqual(p['selected_exchange']['before_object_pairs'],['obj_043','obj_052'])
        self.assertEqual(len(p['exact_changed_videos']),16)
        self.assertEqual(p['invariants']['non_target_delta_videos_unchanged'],3456)
        self.assertTrue(p['invariants']['all_global_object_use_counts_identical'])
        self.assertTrue(p['invariants']['same_object_semantic_pairs_across_four_groups'])
        self.assertLessEqual(p['maxima']['original12_after'],.5)
        self.assertLessEqual(p['maxima']['shape6_after'],.5)
        self.assertLessEqual(p['maxima']['target_shape6_after'],.45)
        self.assertAlmostEqual(p['maxima']['shape6_before'],.5523938937389758,places=12)

    def test_new_pixels_receive_new_ids_and_frozen_parents(self):
        current={r['sample_id']:r for r in self.inputs['current']}
        children=self.plan['exact_changed_videos']
        self.assertEqual(len({r['sample_id'] for r in children}),16)
        self.assertTrue({r['sample_id'] for r in children}.isdisjoint(current))
        self.assertFalse(self.plan['parent_sample_ids_are_aliases'])
        for child in children:
            parent=current[child['parent_sample_id']]
            self.assertEqual(child['parent_video_sha256'],parent['video_sha256'])
            self.assertEqual(child['parent_decoded_stream_sha256'],parent['decoded_stream_sha256'])
            self.assertNotIn(child['object_id'],rebalance.FORBIDDEN_PHOTO_OBJECTS)
            self.assertNotIn(child['old_object_id'],rebalance.FORBIDDEN_PHOTO_OBJECTS)
            self.assertFalse(child['parent_is_alias'])
            self.assertIsNone(child['video_sha256'])

    def test_validation_rejects_missing_lr_member(self):
        with self.assertRaises(AssertionError):
            rebalance.validate_assignments(self.inputs,self.plan['exact_changed_videos'][:-1])

    def test_validation_rejects_unbalanced_object_substitution(self):
        changed=deepcopy(self.plan['exact_changed_videos'])
        changed[0]['object_id']='gpt_obj_001'
        with self.assertRaises(AssertionError):
            rebalance.validate_assignments(self.inputs,changed)

    def test_plan_is_reproducible_and_source_files_stay_unchanged(self):
        self.assertEqual(rebalance.build_plan(self.root),self.plan)
        self.assertFalse(self.plan['frozen_for_generation'])
        for row in self.plan['source_snapshots'].values():
            self.assertEqual(rebalance.sha(row['path']),row['sha256'])

    def test_final_gate_recomputes_declared_photometry_and_current_spoon(self):
        overrides={aid:deepcopy(self.inputs['prep'][aid]) for aid in rebalance.FORBIDDEN_PHOTO_OBJECTS}
        eligibility=rebalance.read(self.root/'plans/natural-t13-20261029/object-eligibility.json')
        t3={r['asset_id']:deepcopy(r) for r in eligibility['objects'] if r['asset_id']=='photo_obj_025'}
        result=rebalance.evaluate_final_gates(self.root,self.plan,overrides,t3)
        self.assertTrue(result['T2']['PASS']);self.assertTrue(result['T3']['PASS'])
        self.assertAlmostEqual(result['T2']['original12_max'],self.plan['maxima']['original12_after'],places=12)
        self.assertAlmostEqual(result['T2']['shape6_max'],self.plan['maxima']['shape6_after'],places=12)
        self.assertAlmostEqual(result['T3']['original_object_max'],.38293890451015855,places=12)

    def test_final_gate_rejects_geometry_change_or_incomplete_icc_inputs(self):
        overrides={aid:deepcopy(self.inputs['prep'][aid]) for aid in rebalance.FORBIDDEN_PHOTO_OBJECTS}
        with self.assertRaises(AssertionError):
            rebalance.evaluate_final_gates(self.root,self.plan,{})
        overrides['photo_obj_025']['rotation_degrees']+=1
        with self.assertRaises(AssertionError):
            rebalance.evaluate_final_gates(self.root,self.plan,overrides)


if __name__=='__main__':unittest.main()
