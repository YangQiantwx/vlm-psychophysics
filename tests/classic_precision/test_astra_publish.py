import copy
import unittest

from classic_precision.astra_publish import cells_from_records


def fixture():
    return [{'sample_id': str(i), 'task': 'T1', 'cell': ['T1', 4, .1], 'truth': 'LEFT',
             'answer': 'LEFT' if i < 3 else 'RIGHT', 'correct': i < 3, 'strict_format': True,
             'visual_only_trace_status': 'PASS_OBSERVED', 'model_effort_verified': True,
             'returned_images': 180 if i == 0 else 60,
             'protocol_status': 'FULL180_TRACE_PASS' if i == 0 else 'EARLY_ANSWER_TRACE_PASS'} for i in range(4)]


class AstraPublicCounts(unittest.TestCase):
    def test_early_answers_stay_in_primary_denominator(self):
        cell = cells_from_records(fixture())[0]
        self.assertEqual((cell['correct'], cell['n'], cell['full180'], cell['right']), (3, 4, 1, 1))
        self.assertEqual(cell['observed_frames']['min'], 60)

    def test_rejects_singleton_and_repeated_sample(self):
        with self.assertRaisesRegex(AssertionError, 'four answers'):
            cells_from_records(fixture()[:1])
        rows = fixture(); rows[1]['sample_id'] = rows[0]['sample_id']
        with self.assertRaisesRegex(AssertionError, 'Repeated'):
            cells_from_records(rows)

    def test_auxiliary_angles_are_not_primary(self):
        rows = fixture()
        for row in rows:
            row['task'] = 'T3'; row['cell'] = ['T3', 1.75]
        with self.assertRaisesRegex(AssertionError, 'Auxiliary'):
            cells_from_records(rows)
