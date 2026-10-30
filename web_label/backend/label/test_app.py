"""Synthetic storage concurrency/invariants; no scientific or participant records."""
import copy
import importlib.util
import json
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.api_core.exceptions import PreconditionFailed

spec = importlib.util.spec_from_file_location('label_app', Path(__file__).with_name('app.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class LabelWorkflow(unittest.TestCase):
    def setUp(self):
        self.objects = {}
        self.lock = threading.Lock()
        def read(key):
            with self.lock:
                value, generation = self.objects.get(key, (None, 0))
                return copy.deepcopy(value), generation
        def write(key, value, generation=0):
            with self.lock:
                old, actual = self.objects.get(key, (None, 0))
                if actual != generation:
                    raise PreconditionFailed('generation mismatch')
                self.objects[key] = (copy.deepcopy(value), actual + 1)
        m.read, m.write = read, write
        m.cached_manifest.cache_clear()
        items = [{'id': f'synthetic-{i}', 'object': f'fixture/{i}', 'sha256': '0'*64, 'width': 768, 'height': 512, 'duration': 3, 'stratum': {'side': i % 2, 'condition': i % 3}} for i in range(25)]
        write('manifests/test/fixture.json', {'version': 'fixture', 'tasks': {'T1': {'title': 'Motion', 'prompt': 'Direction?', 'choices': ['LEFT', 'RIGHT'], 'items': items}}})
        write('manifests/current-test.json', {'object': 'manifests/test/fixture.json'})
        self.client = m.app.test_client()
        result = self.client.post('/api/session', json={'email': 'synthetic@example.invalid', 'mode': 'test'})
        self.auth = {'Authorization': 'Bearer ' + result.json['token']}

    def post(self, route, value):
        return self.client.post(route, json=value, headers=self.auth)

    def test_concurrent_claim_recovery_immutable_answers_and_next_batch(self):
        def claim(_):
            with m.app.test_client() as c:
                return c.post('/api/claim', json={'task': 'T1'}, headers=self.auth).json['batch']
        with ThreadPoolExecutor(max_workers=8) as pool:
            batches = list(pool.map(claim, range(8)))
        bid = batches[0]['id']
        self.assertEqual(len({b['id'] for b in batches}), 1)
        self.assertEqual(batches[0]['total'], 20)
        text = json.dumps(batches[0])
        self.assertNotIn('synthetic-', text)
        self.assertNotIn('stratum', text)
        self.assertEqual(self.post('/api/complete', {'batch_id': bid}).status_code, 409)
        for i in range(20):
            payload = {'batch_id': bid, 'index': i}
            self.assertEqual(self.post('/api/answer', {**payload, 'answer': 'LEFT'}).status_code, 409)
            self.assertEqual(self.post('/api/start', payload).status_code, 200)
            self.assertEqual(self.post('/api/start', payload).status_code, 409)
            self.assertEqual(self.post('/api/playback', {**payload, 'evidence': {'ended': True, 'rate': 1, 'media_time': 3}}).status_code, 409)
            for key, (value, generation) in list(self.objects.items()):
                if key.endswith(f'{i:03d}-start.json'):
                    value['server_epoch'] -= 4
            self.assertEqual(self.post('/api/playback', {**payload, 'evidence': {'ended': True, 'rate': 1, 'media_time': 3}}).status_code, 200)
            self.assertEqual(self.post('/api/answer', {**payload, 'answer': 'LEFT'}).status_code, 200)
            self.assertTrue(self.post('/api/answer', {**payload, 'answer': 'LEFT'}).json['duplicate'])
            self.assertEqual(self.post('/api/answer', {**payload, 'answer': 'RIGHT'}).status_code, 409)
        recovered = self.client.get('/api/state', headers=self.auth).json['batch']
        self.assertEqual(recovered['id'], bid)
        self.assertTrue(recovered['complete'])
        records = [v[0] for k, v in self.objects.items() if k.endswith('-answer.json')]
        self.assertTrue(all(r['allocation_seed'] and r['order_algorithm'] == 'stratum_round_robin_v1' for r in records))
        self.assertEqual(self.post('/api/complete', {'batch_id': bid}).status_code, 200)
        self.assertTrue(self.post('/api/complete', {'batch_id': bid}).json['duplicate'])
        self.assertTrue(self.post('/api/answer', {'batch_id': bid, 'index': 0, 'answer': 'LEFT'}).json['duplicate'])
        self.assertEqual(self.post('/api/answer', {'batch_id': bid, 'index': 0, 'answer': 'RIGHT'}).status_code, 409)
        self.assertTrue(self.post('/api/playback', {'batch_id': bid, 'index': 0}).json['duplicate'])
        second = self.post('/api/claim', {'task': 'T1'}).json['batch']
        self.assertEqual(second['total'], 5)
        self.assertEqual(second['offset'], 20)
        state = next(v[0] for k, v in self.objects.items() if k.endswith('state.json'))
        first = next(v[0] for k, v in self.objects.items() if '/completed/' in k)['items']
        self.assertFalse(set(first).intersection(state['active']['items']))
        self.assertEqual(len([k for k in self.objects if k.endswith('-answer.json')]), 20)
        self.assertEqual(self.post('/api/start', {'batch_id': second['id'], 'index': 0}).status_code, 200)
        # New preview release preserves all prior response objects and skips unchanged answered IDs.
        old_manifest = copy.deepcopy(self.objects['manifests/test/fixture.json'][0])
        old_manifest['version'] = 'fixture-v2'
        old_manifest['tasks']['T1']['items'].append({'id': 'replacement-new-id', 'object': 'fixture/new', 'source_sha256': '1'*64, 'sha256': '1'*64, 'width': 768, 'height': 512, 'duration': 3})
        m.write('manifests/test/fixture-v2.json', old_manifest)
        m.write('manifests/current-test.json', {'object': 'manifests/test/fixture-v2.json'}, 1)
        refreshed = self.client.get('/api/state', headers=self.auth).json
        self.assertIsNone(refreshed['batch'])
        self.assertEqual(refreshed['tasks'][0]['remaining'], 5)
        self.assertTrue(any('/releases/fixture/' in k for k in self.objects))
        self.assertEqual(len([k for k in self.objects if k.endswith('-answer.json')]), 20)
        future = self.post('/api/claim', {'task': 'T1'}).json['batch']
        self.assertEqual(future['total'], 5)


    def test_test_identity_and_human_collection_gate(self):
        self.assertEqual(self.client.post('/api/session', json={'email': 'person@university.edu', 'mode': 'test'}).status_code, 200)
        self.assertEqual(self.client.post('/api/session', json={'email': 'person@university.edu', 'mode': 'human'}).status_code, 403)
        self.assertEqual(self.client.get('/api/state').status_code, 401)

    def test_trial_reference_binds_to_assigned_source_and_stays_blind(self):
        fixture = copy.deepcopy(self.objects['manifests/test/fixture.json'][0])
        source = fixture['tasks']['T1']['items'][0]
        items = []
        for i in range(2):
            items.append({**source, 'id': 'ref-fixture-' + str(i), 'reference': {'object': 'references/ref-' + str(i) + '.png', 'sha256': str(i)*64, 'width': 160, 'height': 160}})
        fixture['tasks']['T5'] = {'title': 'Target', 'prompt': 'Match?', 'choices': ['YES', 'NO'], 'items': items}
        m.write('manifests/test/fixture.json', fixture, 1)
        result = self.client.post('/api/session', json={'email': 'reference@example.invalid', 'mode': 'test'})
        headers = {'Authorization': 'Bearer ' + result.json['token']}
        result = self.client.post('/api/claim', json={'task': 'T5'}, headers=headers).json['batch']
        self.assertEqual(len({r['reference']['sha256'] for r in result['trials']}), 2)
        self.assertNotIn('references/ref-', json.dumps(result))
        old_bucket = m.bucket
        self.addCleanup(setattr, m, 'bucket', old_bucket)
        class FakeBucket:
            def blob(self, name):
                class Blob:
                    def download_as_bytes(self):
                        return name.encode()
                return Blob()
        m.bucket = lambda: FakeBucket()
        state = next(v[0] for k, v in self.objects.items() if k.endswith('state.json') and (v[0].get('active') or {}).get('task') == 'T5')
        first_id = state['active']['items'][0]
        ref = next(item['reference']['object'] for item in items if item['id'] == first_id)
        response = self.client.get(result['trials'][0]['reference']['media'], headers=headers)
        self.assertEqual(response.data, ref.encode())
        self.assertEqual(response.content_type, 'image/png')

    def test_accuracy_eligibility_is_preserved_in_private_response(self):
        fixture = copy.deepcopy(self.objects['manifests/test/fixture.json'][0])
        for row in fixture['tasks']['T1']['items']:
            row['accuracy_eligible'] = False
        m.write('manifests/test/fixture.json', fixture, 1)
        bid = self.post('/api/claim', {'task': 'T1'}).json['batch']['id']
        payload = {'batch_id': bid, 'index': 0}
        self.post('/api/start', payload)
        self.post('/api/answer', {**payload, 'answer': None, 'reason': 'playback_interrupted'})
        record = next(v[0] for k, v in self.objects.items() if k.endswith('-answer.json'))
        self.assertFalse(record['accuracy_eligible'])

    def test_source_policy_closes_new_batches_and_public_media_is_allowlisted(self):
        m.write('manifests/collection-policy.json', {'disabled_tasks': ['T1']})
        self.assertEqual(self.post('/api/claim', {'task': 'T1'}).status_code, 409)
        tasks = self.client.get('/api/state', headers=self.auth).json['tasks']
        self.assertTrue(tasks[0]['disabled'])
        self.assertEqual(self.client.get('/api/example-media/' + 'a'*32).status_code, 404)
        self.assertEqual(self.client.get('/api/example-media/records').status_code, 404)

    def test_interrupted_playback_remains_nonresponse(self):
        bid = self.post('/api/claim', {'task': 'T1'}).json['batch']['id']
        payload = {'batch_id': bid, 'index': 0}
        self.post('/api/start', payload)
        self.assertEqual(self.post('/api/answer', {**payload, 'answer': None, 'reason': 'playback_interrupted'}).status_code, 200)
        record = next(v[0] for k, v in self.objects.items() if k.endswith('-answer.json'))
        self.assertFalse(record['playback_complete'])
        self.assertIsNone(record['answer'])

    def test_large_media_is_bounded_stream_without_content_length(self):
        bid = self.post('/api/claim', {'task': 'T1'}).json['batch']['id']
        old_bucket = m.bucket
        self.addCleanup(setattr, m, 'bucket', old_bucket)
        reads, closed, chunk_sizes = [], [], []
        class Reader:
            remaining = 40 * 1024 * 1024
            def __enter__(self):return self
            def __exit__(self, *args):closed.append(True)
            def read(self, size):
                reads.append(size)
                if not self.remaining:return b''
                self.remaining -= size
                return b'x' * size
        class Blob:
            def open(self, mode, chunk_size):
                self.mode = mode
                chunk_sizes.append(chunk_size)
                return Reader()
        class Bucket:
            def blob(self, name):return Blob()
        m.bucket = lambda: Bucket()
        result = self.client.get('/api/media/' + bid + '/0', headers=self.auth, buffered=False)
        self.assertTrue(result.is_streamed)
        self.assertNotIn('Content-Length', result.headers)
        self.assertLessEqual(len(reads), 1)
        self.assertEqual(sum(len(chunk) for chunk in result.response), 40 * 1024 * 1024)
        self.assertTrue(all(size == 1024 * 1024 for size in reads))
        self.assertEqual(chunk_sizes, [4 * 1024 * 1024])
        self.assertEqual(closed, [True])


if __name__ == '__main__':
    unittest.main()
