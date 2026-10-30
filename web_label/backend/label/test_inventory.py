"""Inventory qualification and blinding gates with synthetic metadata only."""
import copy
import hashlib
import importlib.util
import json
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('inventory', Path(__file__).with_name('sync_inventory.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class InventoryGates(unittest.TestCase):
    def setUp(self):
        self.png = b'\x89PNG\r\n\x1a\n' + struct.pack('>I', 13) + b'IHDR' + struct.pack('>II', 160, 160) + b'fixture'
        refsha = hashlib.sha256(self.png).hexdigest()
        self.ref = {'object': 'references/fixture.png', 'sha256': refsha, 'bytes': len(self.png), 'width': 160, 'height': 160}
        self.rows = []
        self.objects = {}
        for i, tid in enumerate(m.TASKS):
            source = hashlib.sha256((tid+'source').encode()).hexdigest()
            body = hashlib.sha256((tid+'body').encode()).hexdigest()
            row = {'task': tid, 'id': tid+'-fixture', 'source_id': tid+'-fixture', 'domain': 'classic', 'dataset': 'synthetic', 'object': 'browser/'+tid+'.webm', 'sha256': body, 'source_sha256': source, 'qa_sha256': 'a'*64, 'width': 768, 'height': 512, 'fps': 60, 'frames': 180, 'duration': 3, 'bytes': 123, 'accuracy_eligible': tid != 'T4', 'stratum': {'condition': 1, 'truth': 'LEAK_SENTINEL'}, 'validation': {'status': m.VALIDATION, 'source_route': 'official_OpenCV_RGB8_X', 'source_RGB8_sha256': source, 'derivative_RGB8_sha256': source, 'source_master_bits': 8}, 'reference': self.ref if tid == 'T5' else None}
            self.rows.append(row)
            self.objects[row['object']] = SimpleNamespace(size=123, metadata={'sha256': body})
        self.objects[self.ref['object']] = SimpleNamespace(size=len(self.png), metadata={'sha256': refsha})
        raw = self.png
        self.bucket = SimpleNamespace(blob=lambda key: SimpleNamespace(download_as_bytes=lambda: raw))

    def build(self, rows=None):
        return m.build_manifest([{'items': rows or self.rows}], [{'sha256': 'b'*64}], self.objects, self.bucket, list(m.TASKS))

    def test_all_tasks_blind_private_manifest_and_zero_control_denominator(self):
        manifest, receipt = self.build()
        self.assertEqual(set(manifest['tasks']), set(m.TASKS))
        self.assertEqual(receipt['distinct_references'], 1)
        self.assertEqual(receipt['accuracy_eligible_counts']['T4'], 0)
        self.assertNotIn('LEAK_SENTINEL', json.dumps(manifest))
        self.assertNotIn('truth', json.dumps(manifest))
        self.assertEqual(manifest['tasks']['T5']['items'][0]['reference']['width'], 160)

    def test_rejects_missing_task(self):
        with self.assertRaisesRegex(ValueError, 'required task'):
            self.build(self.rows[:-1])

    def test_accepts_sealed_all_identity_status_and_rejects_missing_inventory(self):
        index = {'schema': 'validated-browser-library-v1', 'version': 'fixture', 'status': 'ALL_5_BROWSER_AND_GCS_IDENTITIES_VERIFIED', 'items': self.rows, 'expected_counts': {tid: 1 for tid in m.TASKS}, 'count': 5, 'missing': []}
        class Blob:
            name = 'manifests/fixture.json'
            generation = 1
            def download_as_bytes(self):return json.dumps(index).encode()
        bucket = SimpleNamespace(name='fixture', blob=lambda key: Blob())
        loaded, identity = m.load_index(bucket, 'gs://fixture/manifests/fixture.json')
        self.assertEqual(len(loaded['items']), 5)
        index['missing'] = ['unresolved']
        with self.assertRaisesRegex(ValueError, 'missing items'):
            m.load_index(bucket, 'gs://fixture/manifests/fixture.json')

    def test_rejects_unqualified_or_changed_rgb(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['validation']['derivative_RGB8_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'RGB stream'):
            self.build(rows)
        rows[0]['validation']['status'] = 'PENDING'
        with self.assertRaisesRegex(ValueError, 'validation'):
            self.build(rows)

    def test_rejects_remote_sha_or_reference_dimensions(self):
        self.objects[self.rows[0]['object']].metadata['sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'SHA metadata'):
            self.build()
        self.objects[self.rows[0]['object']].metadata['sha256'] = self.rows[0]['sha256']
        rows = copy.deepcopy(self.rows); rows[-1]['reference']['width'] = 159
        with self.assertRaisesRegex(ValueError, '160'):
            self.build(rows)


if __name__ == '__main__':
    unittest.main()
