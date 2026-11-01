import copy
import gc
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

from classic_precision import gemma_x_recheck_memory_safe as m

B = Path('/Users/qianyang/Desktop/cvpr2027_benchmark')
RUN = B/'runs/reproducibility'/m.BASE/'autotune-off3'


class FinishedProfiler:
    def __init__(self):
        self.action_map = {'exit': [self.stop]}
        self.profiler = bytearray(256 * 1024)
    def stop(self):
        pass


class MemoryRepairTests(unittest.TestCase):
    def test_old_cycle_retains_buffers_without_collection_new_release_does_not(self):
        old_enabled = gc.isenabled()
        gc.disable()
        try:
            old = FinishedProfiler(); old_ref = weakref.ref(old); del old
            self.assertIsNotNone(old_ref())
            m.release_finished_profiler(old_ref())
            self.assertIsNone(old_ref())
            for _ in range(96):
                p = FinishedProfiler(); ref = weakref.ref(p)
                release = m.release_finished_profiler(p); del p
                self.assertTrue(release['detached_native_event_owner'])
                self.assertIsNone(ref())
            self.assertFalse(gc.isenabled())
        finally:
            if old_enabled: gc.enable()

    def test_finish_preserves_result_and_releases_on_success_or_exception(self):
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                p = FinishedProfiler(); ref = weakref.ref(p)
                state = {'profiler': p, 'sid': 'saved-original-id', 'root': Path(directory)}
                del p
                w = SimpleNamespace(_gemma_recheck_trace=state)
                original_result = {'original_tokens': [31884, 106], 'binary_sha': 'immutable'}
                def original(worker):
                    worker._gemma_recheck_trace.pop('profiler')
                    if fail: raise RuntimeError('export failure after raw saved')
                    return original_result
                with patch.object(m, 'ORIGINAL_FINISH', original):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'export failure'): m.finish(w)
                    else:
                        result = m.finish(w)
                        self.assertEqual({k: result[k] for k in original_result}, original_result)
                self.assertIsNone(ref())
                rows = (Path(directory)/'memory-lifecycle.jsonl').read_text().splitlines()
                self.assertEqual(len(rows), 1)
                self.assertFalse(json.loads(rows[0])['math_sampling_or_global_gc_changed'])

    def terminal(self):
        closed = json.loads((RUN/'stalled-worker-closed-evidence.json').read_text())
        signal = json.loads((RUN/'stalled-worker-termination-signal-receipt.json').read_text())
        values = {'closed': closed, 'signal': signal, 'receipt': {
            'status': 'PASS_COMPLETE_ABNORMAL_TERMINAL_ALL_FILE_SHA_RETURN',
            'all_file_SHA_verified': True, 'total_raw': 71}}
        records = {'outer_wait': copy.deepcopy(closed['outer_wait']),
                   'outer_owner': {'status': 'OWNER_PROCESS_EXITED', 'alive': False}, 'workers': []}
        for expected in closed['workers']:
            profile = json.loads((B/'plans'/m.BASE/'autotune-off3'/Path(expected['profile']['path']).name).read_text())
            actual = copy.deepcopy(expected)
            actual.update(profile_digest=expected['wait']['profile_sha256'] if expected['wait'] else 'unstarted',
                          live_owner_count=0, assigned_ids=profile['assigned_ids'])
            records['workers'].append(actual)
        spec = {'authorized_abnormal_closure': 'closed',
                'authorized_single_worker_termination': 'signal', 'terminal_return_receipt': 'receipt'}
        return records, spec, values

    def test_accept_only_actual_authorized_abnormal_closure_without_rewriting_wait(self):
        records, spec, values = self.terminal(); before = copy.deepcopy(records)
        with patch.object(m, 'verified', side_effect=values.__getitem__):
            m.validate_forced_terminal(records, spec)
        self.assertEqual(records, before)
        self.assertEqual(records['workers'][0]['wait']['returncode'], -9)
        self.assertEqual(records['outer_wait']['returncode'], 1)

    def test_reject_bad_terminal_conditions(self):
        for change in ('live', 'no_wait', 'wrong_pid', 'wrong_boot', 'no_archive',
                       'lease', 'live_engine', 'raw_count', 'extra_claim', 'cold2_started', 'fake_zero'):
            with self.subTest(change=change):
                r, s, v = self.terminal(); w = r['workers'][0]
                if change == 'live': r['outer_owner']['alive'] = True
                if change == 'no_wait': r['outer_wait']['real_wait_observed'] = False
                if change == 'wrong_pid': v['signal']['authorized_pid'] = 1
                if change == 'wrong_boot': v['signal']['boot_id'] = 'other'
                if change == 'no_archive': v['receipt']['all_file_SHA_verified'] = False
                if change == 'lease': w['same_inode_lease_free'] = False
                if change == 'live_engine': w['live_owner_count'] = 1
                if change == 'raw_count': w['raw'] = 71
                if change == 'extra_claim': w['claimed_ids'].append('foreign-id')
                if change == 'cold2_started': r['workers'][1]['started'] = True
                if change == 'fake_zero': w['wait']['returncode'] = 0
                with patch.object(m, 'verified', side_effect=v.__getitem__), self.assertRaises(ValueError):
                    m.validate_forced_terminal(r, s)

    def test_existing_scientific_profiles_are_not_resealed(self):
        paths = [B/'plans'/m.BASE/'on-confirm1/radon0-recheck-cold1-profile.json']
        for card in ('fourier0', 'fourier1'):
            paths.extend(B/'plans'/m.BASE/'recorder-safe'/f'{card}-recheck-cold{cold}-profile.json'
                         for cold in (1, 2))
        for path in paths:
            profile = json.loads(path.read_text())
            self.assertEqual(profile['maximum_model_calls'], 96)
            self.assertEqual(profile['engine_kwargs']['kv_cache_memory_bytes'], 48 * 1024**3)
            self.assertEqual(profile['engine_kwargs']['max_model_len'], 65536)
            self.assertEqual(profile['engine_kwargs']['max_num_seqs'], 1)
            # Original ON profiles use the backend default, unlike explicit OFF.
            self.assertIsNot(profile['engine_kwargs'].get('kernel_config', {}).get(
                'enable_flashinfer_autotune'), False)

    def test_overlay_routes_original_gpu_queues_and_restores_all_hooks(self):
        previous = (m.observer.begin, m.observer.finish, m.on.MODULE,
                    m.continuation.MODULE, m.on.validate_terminal, m.core.write_new)
        for card in ('radon0', 'fourier0', 'fourier1'):
            with self.subTest(card=card), patch.object(m, 'admission', return_value={}):
                with m.overlay('/existing/root', card):
                    namespace = m.on.namespace() if card == 'radon0' else m.continuation.namespace(card)
                    with namespace:
                        self.assertEqual(m.core.MODULE, m.MODULE)
                        self.assertIs(m.core.trace.finish, m.finish)
                        self.assertIs(m.observer.begin, m.begin)
                        self.assertEqual(m.core.NAME, m.namespace_name(card))
            self.assertEqual((m.observer.begin, m.observer.finish, m.on.MODULE,
                              m.continuation.MODULE, m.on.validate_terminal, m.core.write_new), previous)

    def test_memory_guard_stops_only_after_original_artifacts_are_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = {'profiler': FinishedProfiler(), 'root': root, 'sid': 'original',
                     'memory_initial_request': {'VmRSS': 1000}}
            def original(worker):
                worker._gemma_recheck_trace.pop('profiler')
                (root/'saved-answer').write_bytes(b'original unchanged answer')
                return {'saved': True}
            with patch.object(m, 'ORIGINAL_FINISH', original), patch.object(
                    m, 'process_memory', return_value={'VmRSS': 1000 + 17 * 1024**2}):
                with self.assertRaisesRegex(ValueError, 'Recorder RSS'):
                    m.finish(SimpleNamespace(_gemma_recheck_trace=state))
            self.assertEqual((root/'saved-answer').read_bytes(), b'original unchanged answer')
            self.assertTrue((root/'memory-lifecycle.jsonl').exists())


if __name__ == '__main__': unittest.main()
