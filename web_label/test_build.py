"""Synthetic publication gates: incomplete science cannot overwrite current UI."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

import build
import workspace_publish


class PublicBuildGates(unittest.TestCase):
    def test_input_paths_are_bounded(self):
        for name in ('../records.json', '/private.json', 'a/../../record.json'):
            with self.assertRaises(ValueError):
                build.relative_path(name)

    def test_capture_requires_versioned_code_and_rejects_private_records(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'unversioned.js').write_text('fixture')
            with self.assertRaisesRegex(ValueError, 'code must be versioned'):
                build.capture(root, 'https://example.invalid')
            (root / 'unversioned.js').unlink()
            (root / 'records').mkdir()
            (root / 'records/test.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'private data'):
                build.capture(root, 'https://example.invalid')

    def test_only_complete_four_file_release_updates_t45(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            analysis = root / 'runs/reproducibility/cosmos-x-t45-20261030/analysis'
            analysis.mkdir(parents=True)
            review = root / 'review'
            review.mkdir()
            stage = root / 'stage'
            (stage / 'experiments').mkdir(parents=True)
            retained = stage / 'experiments/current.json'
            retained.write_bytes(b'unchanged T1-T3 fixture')
            config = {'tasks': {tid: {'result': {'status': 'pending'}} for tid in ('T4', 'T5')}}
            (review / 't45-final.json').write_text(json.dumps(config))
            (analysis / 'summary.json').write_text('{"status":"RUNNING"}')
            with self.assertRaisesRegex(ValueError, 'not COMPLETE'):
                workspace_publish.refresh_t45(root, review, stage)
            self.assertFalse((stage / 'experiments/t45-final.json').exists())
            (analysis / 'summary.json').write_text('{"status":"COMPLETE","fixture":true}')
            with self.assertRaisesRegex(ValueError, 'missing a final figure'):
                workspace_publish.refresh_t45(root, review, stage)
            (analysis / 'summary.json').write_text('{"status":"COMPLETE_COSMOS_X_T45_1200","fixture":true}')
            (analysis / 'cells.csv').write_text('task,fixture\nT4,true\n')
            for tid in ('T4', 'T5'):
                Image.new('RGB', (2, 2)).save(analysis / (tid + '-classic-natural.png'))
            workspace_publish.refresh_t45(root, review, stage)
            published = json.loads((stage / 'experiments/t45-final.json').read_text())
            self.assertEqual(published['tasks']['T4']['result']['status'], 'complete')
            self.assertEqual(retained.read_bytes(), b'unchanged T1-T3 fixture')


if __name__ == '__main__':
    unittest.main()
