"""Small CPU checks; set CLASSIC_T1_SOURCE_ROOT to the downloaded T1 source run."""
from copy import deepcopy
import math
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from classic_precision.source_audit_t1 import T1Audit, extract_line


@pytest.fixture
def audit(tmp_path):
    source = os.environ.get('CLASSIC_T1_SOURCE_ROOT')
    if not source:
        pytest.skip('CLASSIC_T1_SOURCE_ROOT is required for SHA-bound original-source tests')
    return T1Audit(Path(source), tmp_path / 'audit', max_cached_banks=1)


def row_for(audit, sf=.5, depth=.05, phase=0, direction='RIGHT'):
    return next(row for row in audit.rows if
                row['params']['sf_cycles_per_aperture'] == sf and
                row['params']['modulation_depth'] == depth and
                row['phase_index'] == phase and row['correct_answer'] == direction)


def core_lines(audit, row, *, tf=5.):
    """Independent closed form in the flat aperture; no raster/basis/state call."""
    params = row['params']
    x = np.arange(194, 574, dtype=np.float64) + .5 - 384
    t = np.arange(180, dtype=np.float64) / 60
    sf = params['sf_cycles_per_aperture']
    phase = 2 * np.pi * sf * x[None, :] / 448 + params['phase_rad']
    phase = phase - params['direction_sign'] * 2 * np.pi * tf * t[:, None]
    ideal = .31 * (1 + params['modulation_depth'] * np.sinc(sf / 448) * np.cos(phase))
    return audit._producer.quantize(ideal)


def test_complete_scientific_grid_and_frozen_row_hashes(audit):
    value = audit.verify_parameters()
    assert value['pass'] and value['videos'] == value['scientific_spec_sha256_checked'] == 3840


def test_changed_scientific_spec_rejected(audit):
    audit.rows[0]['prompt'] += ' changed'
    with pytest.raises(ValueError, match='scientific specification hash changed'):
        audit.verify_parameters()


def test_mislabeled_phase_or_direction_rejected(audit):
    audit.rows[0]['params']['direction_sign'] = 1
    with pytest.raises(ValueError, match='direction_sign'):
        audit.verify_parameters()


def test_missing_condition_rejected(audit):
    audit.rows.pop()
    with pytest.raises(ValueError, match='complete 3840'):
        audit.verify_parameters()


@pytest.mark.parametrize('sf', [.5, 64.])
def test_cached_frames_are_original_producer_codes_with_exact_12_state_period(audit, sf):
    c, s = audit._producer.build_basis(sf)
    audit._basis[sf] = (c, s)
    audit._producer._BASIS[sf] = (c, s)
    for depth, phase, direction in [(.05, 0, 'RIGHT'), (1., 1, 'LEFT')]:
        row = row_for(audit, sf, depth, phase, direction)
        original = audit._producer.states(row)
        qa = audit.begin_video(row)
        assert qa['max_source_linear_Y_quantization_error'] <= .00003
        for index in range(180):
            expected = audit.expected_frame(row, index)
            assert np.array_equal(expected, original[index % 12])
            assert not expected.flags.writeable
    assert len(audit._banks) == 1


@pytest.mark.parametrize('sf,depth,phase,direction', [
    (.5, .05, 0, 'RIGHT'), (64., 1., 23, 'LEFT'),
    (2., 1., 7, 'RIGHT'), (32., .05, 13, 'LEFT')])
def test_blind_readout_passes_independent_flat_core_formula(audit, sf, depth, phase, direction):
    row = row_for(audit, sf, depth, phase, direction)
    value = audit.pixel_readout(row, core_lines(audit, row))
    assert value['pass'] and value['estimated']['direction'] == direction
    assert set(value['errors']) == set(value['limits'])


@pytest.mark.parametrize('change', ['direction', 'phase', 'sf', 'depth', 'tf'])
def test_blind_readout_rejects_wrong_source_parameter(audit, change):
    row = row_for(audit, 4., .4, 7, 'RIGHT')
    codes = core_lines(audit, row, tf=6. if change == 'tf' else 5.)
    target = deepcopy(row)
    if change == 'direction':
        target['correct_answer'] = 'LEFT'
    elif change == 'phase':
        target['params']['phase_rad'] += math.radians(15)
    elif change == 'sf':
        target['params']['sf_cycles_per_aperture'] = 8.
    elif change == 'depth':
        target['params']['modulation_depth'] = .6
    assert not audit.pixel_readout(target, codes)['pass']


def test_extract_line_is_copy_and_refuses_non_rgb48():
    rgb = np.zeros((512, 768, 3), dtype=np.uint16)
    rgb[255, 194:574, 0] = np.arange(380)
    line = extract_line(rgb)
    assert np.array_equal(line, np.arange(380))
    line[0] = 123
    assert rgb[255, 194, 0] == 0
    with pytest.raises(ValueError, match='RGB48'):
        extract_line(rgb.astype(np.uint8))


def test_partial_or_wrong_precision_core_lines_rejected(audit):
    row = row_for(audit)
    with pytest.raises(ValueError, match='all 180x380'):
        audit.pixel_readout(row, np.zeros((179, 380), dtype=np.uint16))
    with pytest.raises(ValueError, match='all 180x380'):
        audit.pixel_readout(row, np.zeros((180, 380), dtype=np.float64))
    with pytest.raises(ValueError, match='frame index'):
        audit.expected_frame(row, 180)


def test_output_inside_original_run_is_rejected(audit):
    with pytest.raises(ValueError, match='outside the original source'):
        T1Audit(audit.source_root, audit.source_root / 'audit')


def test_constructor_does_not_write_source_bytecode_even_without_python_B(audit, tmp_path):
    copied = tmp_path / 'source-without-basis'
    copied.mkdir()
    for name in ('producer.py', 'manifest.json', 'protocol.json'):
        shutil.copyfile(audit.source_root / name, copied / name)
    before = set(copied.iterdir())
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = False
    try:
        T1Audit(copied)
    finally:
        sys.dont_write_bytecode = previous
    assert set(copied.iterdir()) == before


def test_fresh_basis_bit_difference_is_rejected_without_source_write(audit, tmp_path):
    basis_root = tmp_path / 'source-copy'
    basis_root.mkdir()
    audit.source_root = basis_root
    # Parameter checks are separately exercised on all real 3840 rows above.
    audit.verify_parameters = lambda: {'pass': True}
    c = np.zeros((512, 768), dtype=np.float64)
    s = c.copy()
    audit._producer.build_basis = lambda sf, order=16: (c.copy(), s.copy())
    audit._producer.reference_pixels = lambda: [(0, 0)]
    (basis_root / 'basis').mkdir()
    changed = c.copy()
    changed[0, 0] = np.nextafter(0., 1.)
    path = basis_root / 'basis' / 'sf0p5.npz'
    np.savez(path, cos=changed, sin=s, sf=.5)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='basis differs'):
        audit.verify_parameters_and_basis()
    assert path.read_bytes() == original


def zero_basis_fixture(audit, tmp_path):
    root = tmp_path / 'saved-source-basis'
    (root / 'basis').mkdir(parents=True)
    audit.source_root = root
    audit.verify_parameters = lambda: {'pass': True}
    c = np.zeros((512, 768), dtype=np.float64)
    audit._producer.build_basis = lambda sf, order=16: (c.copy(), c.copy())
    audit._producer.reference_pixels = lambda: [(0, 0), (384, 256)]
    for sf in (.5, 1, 2, 4, 8, 16, 32, 64):
        name = format(float(sf), '.10g').replace('.', 'p')
        np.savez_compressed(root / 'basis' / f'sf{name}.npz', cos=c, sin=c, sf=sf)
    return root


def test_basis_report_retains_each_independent_sample_and_separates_new_outputs(audit, tmp_path):
    root = zero_basis_fixture(audit, tmp_path)
    before = {path.name: path.read_bytes() for path in (root / 'basis').iterdir()}
    calls = []

    def integral(sf, x, y):
        calls.append((sf, x, y))
        return np.zeros(2)

    audit._producer.reference_integral = integral
    report = audit.verify_parameters_and_basis()
    assert report['pass'] and len(report['pixel_integrals']) == 8
    assert len(calls) == 16
    assert all(len(entry['errors']) == entry['pixels'] == 2 for entry in report['pixel_integrals'])
    rebuilt = Path(report['rebuilt_basis_root'])
    assert len(list(rebuilt.glob('sf*.npz'))) == 8 and root not in rebuilt.parents
    assert before == {path.name: path.read_bytes() for path in (root / 'basis').iterdir()}
    worker = T1Audit(os.environ['CLASSIC_T1_SOURCE_ROOT'], basis_root=rebuilt)
    assert np.array_equal(worker._load_basis(64)[0], audit._basis[64][0])


def test_nonfinite_independent_integral_is_rejected(audit, tmp_path):
    zero_basis_fixture(audit, tmp_path)
    audit._producer.reference_integral = lambda sf, x, y: np.array([0., np.nan])
    with pytest.raises(ValueError, match='integral is invalid'):
        audit.verify_parameters_and_basis()


def test_real_adaptive_integral_does_not_use_production_window_or_gl(audit):
    c, s = audit._producer.build_basis(64)

    def forbidden(*args, **kwargs):
        raise AssertionError('Independent integrator touched the production window or GL nodes')

    audit._producer.source_window = forbidden
    original = np.polynomial.legendre.leggauss
    np.polynomial.legendre.leggauss = forbidden
    try:
        for x, y in [(160, 243), (375, 63), (384, 256)]:
            independent = audit._producer.reference_integral(64, x, y)
            assert max(abs(c[y, x] - independent[0]), abs(s[y, x] - independent[1])) <= 2e-8
    finally:
        np.polynomial.legendre.leggauss = original
