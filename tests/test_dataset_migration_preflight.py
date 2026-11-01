import contextlib
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

from classic_precision.dataset_migration_preflight import (
    extract_points, file_identity, physical_tilt_sign, prepare, verify,
)


class MigrationPreflightTests(unittest.TestCase):
    def test_tilt_labels_follow_rendered_upper_endpoint(self):
        # Image coordinates: classic vertices use R(-theta), natural affine R(theta).
        for angle in (.1, .7, 45, 89):
            for source_sign in (-1, 1):
                theta = math.radians(angle * source_sign)
                upper_x = {"classic": -math.sin(theta), "natural": math.sin(theta)}
                for domain, x in upper_x.items():
                    expected = 1 if x < 0 else -1
                    self.assertEqual(physical_tilt_sign(domain, angle, source_sign), expected)
        self.assertEqual(physical_tilt_sign("classic", 90, 1), 0)
        self.assertEqual(physical_tilt_sign("natural", 90, -1), 0)

    def test_nested_cell_metrics_are_not_hidden_by_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"n": 4, "correct": 3, "cells": [
                {"cell": ["T1", 4, .6], "n": 2, "correct": 1},
                {"cell": ["T1", 4, .8], "n": 2, "correct": 2},
            ]}))
            rows = extract_points([path])
            self.assertEqual([r["source_record"] for r in rows], ["", "/cells/0", "/cells/1"])

    def test_baseline_cannot_be_silently_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                prepare(path, path, path)

    def test_verifier_detects_changed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            evidence = path / "original.json"
            evidence.write_text('{"answer":"LEFT"}\n')
            (path / "preparation.json").write_text(json.dumps({
                "immutable_evidence_files": [file_identity(evidence)], "artifacts": [],
            }))
            with contextlib.redirect_stdout(io.StringIO()):
                verify(path)
            evidence.write_text('{"answer":"RIGHT"}\n')
            with self.assertRaisesRegex(ValueError, "changed"):
                verify(path)


if __name__ == "__main__":
    unittest.main()
