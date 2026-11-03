import json
import unittest
from pathlib import Path
from export_records import build_binding, joined_answer


class PrivateExport(unittest.TestCase):
    def setUp(self):
        self.manifest = {'version': 'v1', 'tasks': {'T3': {'items': [
            {'id': 'canonical-video', 'source_sha256': 'a'*64, 'sha256': 'b'*64,
             'object': 'browser/clip.webm', 'domain': 'natural', 'dataset': 'strict-LR',
             'accuracy_eligible': True}]}}}
        self.catalog = {'items': {'old-model-id': {
            'row': {'sample_id': 'old-model-id', 'task': 'T3', 'truth': 'RIGHT',
                    'source_parameters': {'angle': 0.5, 'onset': 18, 'physical_tilt_sign': -1}},
            'content': [{'type': 'video', 'sha256': 'a'*64, 'path': '/source/T3_0.5deg.mp4'}]}}}
        self.identity = {'mode': 'test', 'email': 'qa@example.invalid', 'participant_id': 'participant'}
        self.answer = {'mode': 'test', 'task': 'T3', 'source_id': 'canonical-video',
                       'manifest_version': 'v1', 'answer': 'LEFT'}

    def binding(self):
        raw = json.dumps(self.manifest).encode()
        return build_binding(self.manifest, raw, 'manifests/v1.json',
                             [(Path('catalog.json'), json.dumps(self.catalog).encode())])

    def test_alias_by_exact_video_sha_preserves_answer_and_parameters(self):
        b = self.binding()
        row = joined_answer(self.answer, self.identity, {'v1': b}, 'records/test/participant/trial-answer.json')
        self.assertEqual((row['email'], row['source_id'], row['model_sample_id']),
                         ('qa@example.invalid', 'canonical-video', 'old-model-id'))
        self.assertEqual((row['source_filename'], row['angle_deg'], row['answer'], row['truth']),
                         ('T3_0.5deg.mp4', 0.5, 'LEFT', 'RIGHT'))
        self.assertEqual(json.loads(row['source_parameters_json'])['physical_tilt_sign'], -1)

    def test_missing_or_conflicting_source_is_rejected(self):
        self.manifest['tasks']['T3']['items'][0]['source_sha256'] = 'c'*64
        with self.assertRaises(ValueError): self.binding()

    def test_no_cross_release_or_cross_participant_join(self):
        for bindings, path in [({}, 'records/test/participant/trial-answer.json'),
                               ({'v1': self.binding()}, 'records/test/someone-else/trial-answer.json')]:
            with self.assertRaises(ValueError): joined_answer(self.answer, self.identity, bindings, path)
        with self.assertRaises(ValueError):
            joined_answer({**self.answer, 'mode': 'human'}, self.identity, {'v1': self.binding()},
                          'records/test/participant/trial-answer.json')


if __name__ == '__main__':
    unittest.main()
