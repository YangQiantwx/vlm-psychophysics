import unittest
import argparse
import json
import math
import tempfile
from pathlib import Path
import numpy as np

from classic_precision.dataset_content_audit import (
    classify_axes, match_alpha, measured_side, parse_name, rgb8, signed_axis,
    compare, run, sha,
)


class ContentAuditTest(unittest.TestCase):
    def patch(self, theta):
        yy, xx = np.indices((320, 320), dtype=float)
        xx -= 159.5; yy -= 159.5
        t = np.deg2rad(theta)
        minor = xx * np.cos(t) + yy * np.sin(t)
        major = -xx * np.sin(t) + yy * np.cos(t)
        a = np.exp(-.5 * ((minor / 8)**2 + (major / 40)**2))
        return np.dstack([a * .5, a * .4, a * .3, a])

    def test_axis_sign_and_horizontal(self):
        for angle in [-32, -.5, 0, .5, 32]:
            self.assertAlmostEqual(signed_axis(self.patch(angle)[:, :, 3])['signed_deviation_deg'], angle, places=3)
        self.assertAlmostEqual(signed_axis(self.patch(90)[:, :, 3])['deviation_deg'], 90, places=4)

    def test_blind_candidate_binding_and_ambiguity(self):
        bg = np.ones((512, 768, 3)) * .12
        base, event_float = rgb8(bg), bg.copy()
        candidates = []
        for angle in [-.5, 0, .5]:
            z = self.patch(angle)
            candidates.append((str(angle), str(angle), 'source', z, signed_axis(z[:, :, 3])))
        for x, angle in [(64, 0), (384, .5)]:
            z = self.patch(angle)
            event_float[144:464, x:x + 320] = z[:, :, :3] + (1 - z[:, :, 3, None]) * bg[144:464, x:x + 320]
        event = rgb8(event_float)
        result = match_alpha(base, event, bg, candidates)
        self.assertEqual(result['vertical_side'], 'LEFT')
        self.assertEqual(result['upper_end_direction'], 'right')
        with self.assertRaisesRegex(ValueError, 'unique alpha/RGB'):
            match_alpha(base, event, bg, candidates + [candidates[1]])
        damaged = event.copy(); damaged[300, 500, 0] ^= 1
        with self.assertRaisesRegex(ValueError, 'unique alpha/RGB'):
            match_alpha(base, damaged, bg, candidates)

    def test_side_from_pixels(self):
        b = np.zeros((512, 768, 3), np.uint8); e = b.copy()
        e[200:210, 500:510] = 1
        self.assertEqual(measured_side(b, e)['side'], 'RIGHT')
        e[200:210, 200:210] = 1
        with self.assertRaises(ValueError):
            measured_side(b, e)

    def test_name_semantics(self):
        n = parse_name('T3/T3_verticalLeft_tiltRight_0.5deg_bg-na_036_obj-na_032_trial01.mp4', 'natural', 'T3')
        self.assertEqual((n['side'], n['direction'], n['angle']), ('LEFT', 'right', .5))
        n = parse_name('T2/T2_left_bg0.2_delta0.35_30frames_trial05.mkv', 'classic', 'T2')
        self.assertEqual((n['delta'], n['duration'], n['trial']), (.35, 30, 5))

    def test_simultaneous_name_condition_phase_corruption_rejected(self):
        mapping = {'domain': 'classic', 'task': 'T1', 'truth': 'LEFT',
                   'proposed_public_path': 'T1/T1_left_sf0.5_contrast0.05_trial02.mkv',
                   'condition_key': json.dumps(['T1', '.5', '.05', 1, 'LEFT'])}
        observed = {'side': 'LEFT', 'sf': .5, 'm': .05, 'phase_error_rad': 0,
                    'initial_phase_rad': math.radians(7.5)}
        with self.assertRaisesRegex(ValueError, 'Measured phase/name'):
            compare(mapping, {}, observed, {'sf_absolute': .001, 'm_absolute': .0001, 'phase_radians': .001})

    def test_resume_rejects_changed_input(self):
        from classic_precision import dataset_content_audit as module
        with tempfile.TemporaryDirectory() as d:
            p = Path(d); source = p / 'input.jsonl'
            source.write_text(json.dumps({'mapping': {'legacy_sample_id': 'one'}}) + '\n')
            (p / 'per-video.jsonl').write_text(json.dumps({'status': 'PASS', 'legacy_sample_id': 'one',
                'code_sha256': sha(module.__file__), 'input_sha256': 'not-current'}) + '\n')
            args = argparse.Namespace(input=source, out=p, root=p, stage=p, support=p,
                                      workers=1, pilot=False, resume=True)
            with self.assertRaisesRegex(ValueError, 'unchanged code/input'):
                run(args)


if __name__ == '__main__':
    unittest.main()
