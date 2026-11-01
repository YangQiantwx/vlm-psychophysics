import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from classic_precision import gemma_x_recheck_recorder_repair as repair
from classic_precision import gemma_x_recheck_trace_v2 as trace


class RecorderRepairTests(unittest.TestCase):
    def fixture(self, active=True, capture_fails=False, native_fails=False):
        calls = []
        result = object()
        def native(*args, **kwargs):
            calls.append(('native', args, kwargs))
            if native_fails: raise RuntimeError('native failure')
            return result
        module = SimpleNamespace(gumbel_sample=native)
        def sample(*args, **kwargs):
            calls.append(('sample', args, kwargs))
            return module.gumbel_sample(*args, **kwargs)
        sampler = SimpleNamespace(sample=sample)
        state = {'active': active, 'sample_step': -1, 'records': []}
        def capture(kind, tensor, row, args, kwargs):
            calls.append((kind, tensor))
            if capture_fails: raise RuntimeError('observer failure')
            return {'kind': kind}
        trace.attach_sampler(state, sampler, module, capture)
        return state, sampler, module, calls, result, sample, native

    def test_native_arguments_result_identity_and_call_count_preserved(self):
        s, sampler, module, calls, result, _, _ = self.fixture()
        tensor, mapping = object(), object()
        self.assertIs(sampler.sample(tensor, mapping, apply_temperature=False), result)
        native = [x for x in calls if x[0] == 'native']
        self.assertEqual(native, [('native', (tensor, mapping), {'apply_temperature': False})])
        self.assertEqual([x[0] for x in calls], ['raw_sampler_logits', 'sample',
            'actual_greedy_logits', 'native', 'actual_greedy_result'])
        self.assertEqual(s['sample_step'], 0)

    def test_observer_failure_does_not_suppress_native_answer(self):
        s, sampler, _, calls, result, _, _ = self.fixture(capture_fails=True)
        self.assertIs(sampler.sample(object()), result)
        self.assertEqual(len([x for x in calls if x[0] == 'native']), 1)
        self.assertEqual(len(s['records']), 3)

    def test_native_exception_is_not_retried_or_swallowed(self):
        _, sampler, _, calls, _, _, _ = self.fixture(native_fails=True)
        with self.assertRaisesRegex(RuntimeError, 'native failure'): sampler.sample(object())
        self.assertEqual(len([x for x in calls if x[0] == 'native']), 1)

    def test_inactive_and_restore(self):
        s, sampler, module, calls, result, sample, native = self.fixture(active=False)
        self.assertIs(sampler.sample(object()), result)
        self.assertEqual([x[0] for x in calls], ['sample', 'native'])
        s['restore_sampler']()
        self.assertIs(sampler.sample, sample); self.assertIs(module.gumbel_sample, native)

    def test_exact_v2_source_binding_rejects_legacy_or_different_sdk(self):
        cls = 'vllm.v1.worker.gpu.sample.sampler.Sampler'
        trace.validate_sources(cls, {'sha256': trace.SAMPLER_SHA}, {'sha256': trace.GUMBEL_SHA})
        for c, s, g in [('vllm.v1.sample.sampler.Sampler', trace.SAMPLER_SHA, trace.GUMBEL_SHA),
                        (cls, 'changed', trace.GUMBEL_SHA), (cls, trace.SAMPLER_SHA, 'changed')]:
            with self.subTest(c=c, s=s, g=g), self.assertRaises(ValueError):
                trace.validate_sources(c, {'sha256': s}, {'sha256': g})

    def test_namespace_keeps_off_engine_observer_and_child_entrypoint(self):
        prior = repair.core.NAME, repair.core.MODULE, repair.off.NAME, repair.trace.install
        with repair.namespace('radon0'):
            self.assertIs(repair.trace.install, trace.install)
            self.assertIs(repair.trace.close, trace.close)
            self.assertIs(repair.trace.AdapterObservation, repair.off.Observation)
            self.assertIs(repair.core.full.worker_identity, repair.off.identity)
            with repair.recovery.namespace():
                self.assertEqual(repair.core.NAME, repair.name('radon0'))
                self.assertEqual(repair.core.MODULE, repair.MODULE)
                self.assertIs(repair.core.science, repair.science)
        self.assertEqual(prior, (repair.core.NAME, repair.core.MODULE, repair.off.NAME, repair.trace.install))

    def test_fourier_namespace_does_not_turn_autotune_off(self):
        original = repair.core.full.worker_identity, repair.trace.AdapterObservation
        with repair.namespace('fourier0'):
            self.assertEqual(original, (repair.core.full.worker_identity, repair.trace.AdapterObservation))
            self.assertEqual(repair.core.NAME, repair.name('fourier0'))
            self.assertIs(repair.trace.install, trace.install)

    def test_post_load_closure_requires_zero_requests_real_wait_and_fence(self):
        p = Path('/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility') / repair.BASE / 'autotune-off1-zero-request-closed.json'
        value = json.loads(p.read_text()); repair.validate_closure(value)
        mutations = [lambda v: v['workers'][0].update(claims=1),
                     lambda v: v['workers'][0].update(raw=1),
                     lambda v: v['workers'][0].update(scientific_INTENT=1),
                     lambda v: v['workers'][0]['real_wait'].update(owned_engines_exited=False),
                     lambda v: v['workers'][0].update(original_same_inode_lease_free=False),
                     lambda v: v['workers'][1].update(real_wait={}),
                     lambda v: v['outer_real_wait'].update(real_wait_observed=False),
                     lambda v: v['controls'][0]['after'].update(paused=False)]
        for mutation in mutations:
            bad = copy.deepcopy(value); mutation(bad)
            with self.assertRaises(ValueError): repair.validate_closure(bad)

    def test_science_routes_only_authorized_flag_and_does_not_mutate(self):
        for card in ['radon0', 'fourier0', 'fourier1']:
            p = {'card': card, 'recheck': {'observation_repair': {
                'entrypoint': {'sha256': repair.file_sha256(repair.__file__)},
                'trace_v2': {'sha256': repair.file_sha256(trace.__file__)},
                'scientific_changes_from_donor': [], 'donor_closed': 'closure'}}}
            before = copy.deepcopy(p)
            with (patch.object(repair, 'validate_closure'), patch.object(repair, 'OFF_SCIENCE') as off,
                  patch.object(repair, 'ON_SCIENCE') as on):
                repair.science(p, lambda x: {})
                self.assertEqual(off.call_count, int(card == 'radon0'))
                self.assertEqual(on.call_count, int(card != 'radon0'))
            self.assertEqual(p, before)


if __name__ == '__main__': unittest.main()
