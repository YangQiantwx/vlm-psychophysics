"""Offline request, identity, billing, and no-inference guard regression tests."""
import base64
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

from vlm_psychophysics.api import astra as a


def png():
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 768, 512, 16, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\0' + bytes(768 * 6)) * 512)) + chunk(b'IEND', b''))


class DirectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        p, out = a.paths(self.root)
        (p / 'prompts').mkdir(parents=True)
        for t in ('T1', 'T2', 'T3'):
            (p / 'prompts' / f'{t}-prompt.txt').write_text('Inspect the sequence. Return LEFT or RIGHT.')
        self.frame = self.root / 'original.png'
        self.frame.write_bytes(png())
        self.cid = 'a' * 32
        a.write(p / 'samples.json', [{'case_id': self.cid, 'task': 'T1', 'sample_id': 'PRIVATE_TRUTH_LEFT',
                                     'frames_directory': '/tmp/private', 'video_relative': 'T1/secret-left.mkv',
                                     'truth': 'LEFT', 'historical_answer': 'LEFT'}])
        self.plan = {'samples_sha256': a.sha(p / 'samples.json'), 'implementation_sha256': a.sha(Path(a.__file__)),
                     'prompts': {t: a.sha(p / 'prompts' / f'{t}-prompt.txt') for t in ('T1', 'T2', 'T3')},
                     'max_output_tokens': 4096, 'input_tokens_reservation': 100000,
                     'budget': {'ceiling_usd': 38}, 'standard_pilot_ids': [self.cid], 'batch_ids': []}
        a.write(p / 'plan.json', self.plan)
        self.catalog = {self.cid: [{'index': i, 'path': str(self.frame), 'sha256': a.sha(self.frame)} for i in range(180)]}
        a.write(out / 'frame-catalog.json', self.catalog)
        a.write(out / 'preflight.json', {'samples_sha256': self.plan['samples_sha256'], 'frame_catalog_sha256': a.sha(out / 'frame-catalog.json')})


    def test_preflight_uses_selected_root_and_preserves_source_checks(self):
        p, out = a.paths(self.root)
        video = self.root / "datasets/classic-t1-t3-v0.4/T1/secret-left.mkv"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"source-byte-fixture")
        rollout = self.root / "runs/reproducibility" / a.OLD / "case/rollout.jsonl"
        rollout.parent.mkdir(parents=True)
        image_url = "data:image/png;base64," + base64.b64encode(self.frame.read_bytes()).decode()
        rollout.write_text(json.dumps({
            "type": "response_item",
            "payload": {"output": [{"type": "input_image", "image_url": image_url}]},
        }) + "\n")
        rows = a.read(p / "samples.json")
        rows[0].update(source_sha256=a.sha(video), historical_case_relative="case",
                       historical_rollout_sha256=a.sha(rollout), historical_returned_images=1)
        (p / "samples.json").write_text(json.dumps(rows))
        self.plan["samples_sha256"] = a.sha(p / "samples.json")
        self.save_plan()
        (out / "frame-catalog.json").unlink()
        (out / "preflight.json").unlink()
        with patch.object(a, "frame_files", return_value=[self.frame] * 180), \
                patch("urllib.request.urlopen") as network:
            receipt = a.preflight(self.root)
        network.assert_not_called()
        self.assertEqual(receipt["status"], "PASS_OFFLINE_INPUT_PREFLIGHT")
        self.assertEqual(receipt["frames"], 180)
        self.assertEqual(receipt["model_calls"], 0)
        self.assertEqual(len(a.read(out / "frame-catalog.json")[self.cid]), 180)
        video.write_bytes(b"changed-source")
        with self.assertRaisesRegex(ValueError, "Source bytes changed"):
            a.preflight(self.root)

    def response(self):
        return {'status': 'completed', 'model': a.MODEL, 'service_tier': 'default', 'id': 'test',
                'output': [{'type': 'reasoning', 'summary': []},
                           {'type': 'message', 'content': [{'type': 'output_text', 'text': 'LEFT'}]}],
                'usage': {'input_tokens': 85000, 'output_tokens': 1000,
                          'input_tokens_details': {'cached_tokens': 0, 'cache_write_tokens': 0}}}

    def test_real_payload_order_image_bytes_no_tools_no_private_path(self):
        with patch('urllib.request.urlopen') as network:
            body, evidence = a.build_request(self.root, self.cid)
        network.assert_not_called()
        self.assertEqual(body['tools'], [])
        self.assertEqual(body['tool_choice'], 'none')
        self.assertEqual(body['prompt_cache_options'], {'mode': 'explicit'})
        content = body['input'][0]['content']
        images = [x for x in content if x['type'] == 'input_image']
        self.assertEqual(len(images), 180)
        self.assertEqual(evidence['frames'], 180)
        self.assertEqual(base64.b64decode(images[-1]['image_url'].split(',')[1]), self.frame.read_bytes())
        self.assertEqual(content[1]['text'], 'Frame 000; t=0.000000 s')
        self.assertEqual(content[-2]['text'], 'Frame 179; t=2.983333 s')
        self.assertTrue(all(x['detail'] == 'original' for x in images))
        serialized = json.dumps(body)
        self.assertNotIn('PRIVATE_TRUTH', serialized)
        self.assertNotIn('/tmp/private', serialized)
        self.assertNotIn('temperature', body)
        self.assertNotIn('seed', body)

    def test_mutated_frame_refuses_before_network(self):
        self.frame.write_bytes(png() + b'changed')
        with self.assertRaisesRegex(ValueError, 'Frame bytes changed'):
            a.build_request(self.root, self.cid)

    def test_bad_order_even_if_catalog_rehashed_refuses(self):
        p, out = a.paths(self.root)
        self.catalog[self.cid][7]['index'] = 8
        (out / 'frame-catalog.json').write_text(json.dumps(self.catalog))
        (out / 'preflight.json').write_text(json.dumps({'samples_sha256': self.plan['samples_sha256'], 'frame_catalog_sha256': a.sha(out / 'frame-catalog.json')}))
        with self.assertRaisesRegex(ValueError, 'Frame order/count'):
            a.build_request(self.root, self.cid)

    def test_tool_response_and_incomplete_are_not_answers(self):
        r = self.response()
        r['output'].append({'type': 'function_call', 'name': 'shell'})
        with self.assertRaisesRegex(ValueError, 'Unexpected tool'):
            a.parse_response(r)
        r = self.response()
        r['status'] = 'incomplete'
        with self.assertRaisesRegex(ValueError, 'Incomplete'):
            a.parse_response(r)

    def test_reasoning_output_cost_not_double_counted_and_format_separate(self):
        r = self.response()
        r['usage']['output_tokens_details'] = {'reasoning_tokens': 999}
        self.assertAlmostEqual(a.parse_response(r)['standard_cost_usd'], .9)
        r['output'][1]['content'][0]['text'] = 'It is LEFT.'
        parsed = a.parse_response(r)
        self.assertFalse(parsed['strict_format'])
        self.assertIsNone(parsed['answer'])

    def test_cache_reads_are_accounted_without_rejecting_visual_answer(self):
        r = self.response()
        r['usage']['input_tokens_details']['cached_tokens'] = 10000
        parsed = a.parse_response(r)
        self.assertEqual(parsed['answer'], 'LEFT')
        self.assertEqual(parsed['usage'], r['usage'])
        self.assertAlmostEqual(parsed['standard_cost_usd'], .81)
        self.assertEqual(parsed['billing']['cached_input_tokens'], 10000)

    def test_cache_writes_use_disjoint_lifetimes_or_conservative_upper_bound(self):
        r = self.response()
        r['usage'] = {'input_tokens': 12000, 'output_tokens': 1000, 'input_tokens_details': {
            'cached_tokens': 4000, 'cache_write_tokens': 7000,
            'cache_write_tokens_details': {'ttl_30m_tokens': 3000, 'ttl_12h_tokens': 4000}}}
        parsed = a.parse_response(r)
        self.assertAlmostEqual(parsed['standard_cost_usd'], .2015)
        self.assertEqual(parsed['billing']['ordinary_input_tokens'], 1000)
        del r['usage']['input_tokens_details']['cache_write_tokens_details']
        bounded = a.parse_response(r)
        self.assertAlmostEqual(bounded['standard_cost_usd'], .239)
        self.assertIn('upper_bound', bounded['billing']['basis'])

    def test_invalid_usage_and_other_model_are_not_silent(self):
        for details in [{'cached_tokens': -1}, {'cached_tokens': 85001},
                        {'cached_tokens': 84000, 'cache_write_tokens': 2000},
                        {'cache_write_tokens': 1000, 'cache_write_tokens_details': {'ttl_30m_tokens': 999}},
                        {'cached_tokens': True}]:
            with self.subTest(details=details):
                r = self.response()
                r['usage']['input_tokens_details'] = details
                with self.assertRaisesRegex(ValueError, 'cache'):
                    a.parse_response(r)
        r = self.response()
        r['model'] = 'different-model'
        with self.assertRaisesRegex(ValueError, 'Served model'):
            a.parse_response(r)

    def save_plan(self):
        p, _ = a.paths(self.root)
        (p / 'plan.json').write_text(json.dumps(self.plan))

    def test_budget_and_default_refuse_without_auth_or_network(self):
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}) as env:
            for budget in (0, -1, 39, float('nan')):
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, 'Explicit budget'):
                    a.run_standard(self.root, [self.cid], budget, True)
            with self.assertRaisesRegex(ValueError, 'Budget reservation exhausted'):
                a.run_standard(self.root, [self.cid], .1, True)
        network.assert_not_called()
        _, out = a.paths(self.root)
        self.assertFalse((out / 'claims').exists())

    def test_existing_unresolved_claim_is_never_resent(self):
        _, out = a.paths(self.root)
        a.write(out / 'claims' / (self.cid + '.json'), {'reserved_usd': 1.2048})
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}):
            with self.assertRaisesRegex(ValueError, 'unresolved claim'):
                a.run_standard(self.root, [self.cid], 38, True)
        network.assert_not_called()

    def test_reservations_include_other_unresolved_attempts(self):
        _, out = a.paths(self.root)
        a.write(out / 'claims' / ('b' * 32 + '.json'), {'reserved_usd': 1.2048})
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}):
            with self.assertRaisesRegex(ValueError, 'Budget reservation exhausted'):
                a.run_standard(self.root, [self.cid], 2, True)
        network.assert_not_called()

    def test_cross_process_budget_lock_refuses_before_auth_or_network(self):
        _, out = a.paths(self.root)
        child = subprocess.Popen([sys.executable, '-c',
            "import fcntl,sys; f=open(sys.argv[1],'a'); fcntl.flock(f,fcntl.LOCK_EX); "
            "print('READY',flush=True); sys.stdin.readline()", str(out / 'execution.lock')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'READY')
            with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}):
                with self.assertRaisesRegex(ValueError, 'budget lock'):
                    a.run_standard(self.root, [self.cid], 38, True)
            network.assert_not_called()
        finally:
            child.communicate('done\n', timeout=10)
        self.assertEqual(child.returncode, 0)

    def test_mock_success_saved_and_existing_success_skipped_without_auth(self):
        _, out = a.paths(self.root)
        wire = json.dumps(self.response()).encode()
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {'OPENAI_API_KEY': 'offline-placeholder'}):
            network.return_value.__enter__.return_value.read.return_value = wire
            network.return_value.__enter__.return_value.headers = {'x-request-id': 'offline-fixture'}
            a.run_standard(self.root, [self.cid], 38, True)
            self.assertEqual(network.call_count, 1)  # Mock only; no provider connection.
        record = a.read(out / 'cases' / self.cid / 'result.json')
        self.assertEqual(record['response_sha256'], a.digest(wire))
        claim = a.read(out / 'claims' / (self.cid + '.json'))
        self.assertEqual(record['request_sha256'], claim['request_sha256'])
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}):
            a.run_standard(self.root, [self.cid], 38, True)
        network.assert_not_called()
        self.assertEqual(len(list((out / 'claims').glob('*.json'))), 1)

    def test_uncertain_network_attempt_retains_claim_and_blocks_retry(self):
        _, out = a.paths(self.root)
        with patch('urllib.request.urlopen', side_effect=TimeoutError) as network, patch.object(a.os, 'environ', {'OPENAI_API_KEY': 'offline-placeholder'}):
            with self.assertRaisesRegex(RuntimeError, 'HOLD_NO_RETRY'):
                a.run_standard(self.root, [self.cid], 38, True)
            self.assertEqual(network.call_count, 1)
        self.assertTrue((out / 'claims' / (self.cid + '.json')).exists())
        self.assertEqual(a.read(out / 'cases' / self.cid / 'failure.json')['status'], 'HOLD_NO_RETRY')
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}):
            with self.assertRaisesRegex(ValueError, 'failed/uncertain'):
                a.run_standard(self.root, [self.cid], 38, True)
        network.assert_not_called()

    def test_batch_partition_export_is_local_once_and_excludes_pilot(self):
        p, out = a.paths(self.root)
        bid = 'b' * 32
        rows = a.read(p / 'samples.json')
        rows.append(dict(rows[0], case_id=bid))
        (p / 'samples.json').write_text(json.dumps(rows))
        self.plan['samples_sha256'] = a.sha(p / 'samples.json')
        self.plan['batch_ids'] = [bid]
        self.save_plan()
        a.write(out / 'cases' / self.cid / 'result.json', {'strict_format': True, 'zero_tool_calls': True, 'standard_cost_usd': .9})
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}), patch.object(a, 'build_request', return_value=({'service_tier': 'default', 'tools': []}, {})) as build:
            receipt = a.export_batch(self.root)
            self.assertEqual(receipt['batch_ids'], [bid])
            self.assertEqual(receipt['excluded_standard_pilot_ids'], [self.cid])
            line = json.loads(Path(receipt['shards'][0]['path']).read_text())
            self.assertEqual(line['custom_id'], bid)
            self.assertNotIn('service_tier', line['body'])
            self.assertEqual(receipt['model_calls'], 0)
            with self.assertRaisesRegex(ValueError, 'already prepared'):
                a.export_batch(self.root)
        build.assert_called_once_with(self.root, bid)
        network.assert_not_called()

    def test_batch_cannot_reinclude_standard_pilot(self):
        self.plan['batch_ids'] = [self.cid]
        self.save_plan()
        with patch('urllib.request.urlopen') as network, self.assertRaisesRegex(ValueError, 'Duplicate/missing'):
            a.export_batch(self.root)
        network.assert_not_called()

    def test_refresh_refuses_any_existing_attempt(self):
        _, out = a.paths(self.root)
        a.write(out / 'claims' / (self.cid + '.json'), {'reserved_usd': 1.2048})
        with patch('urllib.request.urlopen') as network, self.assertRaisesRegex(ValueError, 'Existing attempts'):
            a.refresh_preparation(self.root)
        network.assert_not_called()

    def test_unadmitted_maintenance_plan_cannot_execute(self):
        p, out = a.paths(self.root)
        maintained = dict(self.plan, maintenance={'model_calls': 0, 'original_plan_sha256': a.sha(p / 'plan.json'), 'unchanged_input_bindings': {}})
        a.write(p / 'plan-maintained.json', maintained)
        with patch('urllib.request.urlopen') as network, patch.object(a.os, 'environ', {}), self.assertRaises(FileNotFoundError):
            a.run_standard(self.root, [self.cid], 38, True)
        network.assert_not_called()

    def test_requires_execute_and_stays_inside_budget(self):
        with patch('urllib.request.urlopen') as network:
            with self.assertRaisesRegex(ValueError, '--execute'):
                a.run_standard(self.root, [self.cid], 40, False)
        network.assert_not_called()
        self.assertAlmostEqual(a.cost(100000, 4096) * (3 + 57 / 2), 37.9512)
        self.assertLess(a.cost(100000, 4096) * (3 + 57 / 2), 38)


if __name__ == '__main__':
    unittest.main()
