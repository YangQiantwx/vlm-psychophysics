"""Read-only CPU helpers for a fresh T1 RGB16 source audit.

The caller owns video decoding and the per-ID result journal.  Source mathematics
and the blind estimator are loaded from the SHA-bound original producer without
running its CLI.  Rebuilt basis files, if requested, go only to the new audit root.
Independent adaptive quadrature checks a documented pixel sample, not all pixels.
"""
from __future__ import annotations

from collections import Counter, OrderedDict
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np


WIDTH, HEIGHT, FPS, FRAMES = 768, 512, 60, 180
MEAN = .31
SFS = (.5, 1., 2., 4., 8., 16., 32., 64.)
DEPTHS = (.05, .1, .2, .3, .4, .6, .7, .8, .9, 1.)
LINE_ROW, LINE_START, LINE_STOP = 255, 194, 574
COEFFICIENT_ERROR_LIMIT = 2e-8
QUANTIZATION_ERROR_LIMIT = .00003


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _token(value):
    return format(float(value), '.10g').replace('.', 'p')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def extract_line(rgb):
    """Copy the flat-core line needed by the blind estimator from one RGB frame."""
    rgb = np.asarray(rgb)
    _require(rgb.dtype == np.uint16 and rgb.shape == (HEIGHT, WIDTH, 3),
             'T1 line extraction requires a 512x768x3 uint16 RGB48 frame')
    return rgb[LINE_ROW, LINE_START:LINE_STOP, 0].copy()


class T1Audit:
    """No decoding, GPU calls, source writes, or expensive work at construction.

    Call ``verify_parameters_and_basis`` once in the audit coordinator.  Workers
    can then set ``basis_root`` to its returned ``rebuilt_basis_root`` and load
    those arrays lazily without repeating quadrature or basis construction.
    Order videos by (SF, modulation_depth) to keep the small state cache useful.
    ``expected_frame`` returns one grayscale plane; compare all three RGB planes.
    """

    def __init__(self, source_root, output_root=None, *, basis_root=None,
                 max_cached_banks=2):
        self.source_root = Path(source_root).expanduser().resolve()
        self.output_root = None if output_root is None else Path(output_root).expanduser().resolve()
        if self.output_root is not None:
            _require(self.output_root != self.source_root and
                     self.source_root not in self.output_root.parents,
                     'T1 audit output must be outside the original source run')
        _require(isinstance(max_cached_banks, int) and max_cached_banks > 0,
                 'max_cached_banks must be a positive integer')
        self.max_cached_banks = max_cached_banks
        self.protocol = _read(self.source_root / 'protocol.json')
        self.manifest = _read(self.source_root / 'manifest.json')
        self.rows = self.manifest['samples']
        self.basis_root = self.source_root / 'basis' if basis_root is None else Path(basis_root).resolve()
        producer_path = self.source_root / 'producer.py'
        producer_bytes = producer_path.read_bytes()
        self.producer_sha256 = hashlib.sha256(producer_bytes).hexdigest()
        _require(self.producer_sha256 == self.protocol['implementation']['producer_sha256'],
                 'T1 producer differs from the frozen protocol')
        spec = importlib.util.spec_from_file_location(
            '_t1_audit_source_' + self.producer_sha256[:16], producer_path)
        _require(spec is not None and spec.loader is not None, 'Cannot load T1 producer')
        self._producer = importlib.util.module_from_spec(spec)
        # SourceFileLoader.exec_module may write __pycache__ into the original
        # run when the caller omits -B.  Execute the already hashed bytes directly.
        exec(compile(producer_bytes, str(producer_path), 'exec'), self._producer.__dict__)
        self._basis = {}
        self._banks = OrderedDict()

    def verify_parameters(self):
        """Check the complete 3840-row grid and each row's frozen scientific hash."""
        p = self.protocol
        _require(p['schema'] == 't1-rgb16-v1' and p['count'] == 3840,
                 'Unexpected T1 protocol schema/count')
        _require(self.manifest['schema'] == 't1-rgb16-manifest-v1' and
                 self.manifest['count'] == len(self.rows) == 3840,
                 'T1 manifest is not the complete 3840-video source grid')
        _require(p['grid'] == {
            'sf_cycles_per_aperture': list(SFS), 'modulation_depth': list(DEPTHS),
            'phase_degrees': [7.5 + 15 * k for k in range(24)]},
            'T1 protocol grid changed')
        source = p['source']
        for key, value in {'canvas': [WIDTH, HEIGHT],
                           'aperture_center_continuous_xy': [384, 256],
                           'flat_radius_px': 192, 'outer_radius_px': 224,
                           'mean_linear_Y': MEAN, 'independent_states_per_video': 12,
                           'source_frames_written': FRAMES,
                           'phase_bank_at_frame': '(phase_index-2*direction_sign*frame)%24'}.items():
            _require(source[key] == value, f'T1 source parameter changed: {key}')
        for key, value in {'container': 'matroska', 'codec': 'FFV1', 'level': 3,
                           'pixel_format': 'gbrp16le', 'width': WIDTH, 'height': HEIGHT,
                           'fps': FPS, 'frames': FRAMES, 'no_8bit_intermediate': True}.items():
            _require(p['encoding'][key] == value, f'T1 encoding parameter changed: {key}')
        seen = Counter()
        for row in self.rows:
            params = row['params']
            index, direction = row['phase_index'], row['correct_answer']
            _require(type(index) is int and 0 <= index < 24 and direction in ('LEFT', 'RIGHT'),
                     f'T1 phase/direction invalid: {row["sample_id"]}')
            sf, depth = params['sf_cycles_per_aperture'], params['modulation_depth']
            _require(sf in SFS and depth in DEPTHS, 'T1 SF/depth outside source grid')
            expected_id = f't1rgb16_i{index:02d}_sf{_token(sf)}_m{_token(depth)}_{direction.lower()}'
            _require(row['sample_id'] == expected_id and row['video_path'] == f'media/{expected_id}.mkv',
                     'T1 sample ID/path does not encode its parameters')
            _require(row['task'] == row['task_id'] == 'T1' and
                     row['allowed_answers'] == ['LEFT', 'RIGHT'], 'T1 labels changed')
            for key, value in {'tf_hz': 5, 'phase_degrees': 7.5 + 15 * index,
                               'bank_start_index': index, 'direction': direction,
                               'direction_sign': 1 if direction == 'RIGHT' else -1,
                               'aperture_diameter_px': 448, 'classic_mean_Y': MEAN,
                               'multiplier_depth': depth}.items():
                _require(params[key] == value, f'T1 row parameter mismatch: {key}')
            _require(abs(params['phase_rad'] - math.radians(params['phase_degrees'])) < 1e-14,
                     'T1 initial phase is inconsistent')
            for key, value in {'frames': FRAMES, 'fps': FPS, 'duration_s': 3,
                               'width': WIDTH, 'height': HEIGHT,
                               'dimensions_wh': [WIDTH, HEIGHT]}.items():
                _require(row[key] == value, f'T1 row geometry/time changed: {key}')
            signed = {k: v for k, v in row.items() if k != 'scientific_spec_sha256'}
            _require(self._producer.object_sha(signed) == row['scientific_spec_sha256'],
                     f'T1 scientific specification hash changed: {expected_id}')
            seen[(sf, depth, index, direction)] += 1
        expected = Counter(itertools.product(SFS, DEPTHS, range(24), ('LEFT', 'RIGHT')))
        _require(seen == expected, 'T1 source grid has duplicates or missing conditions')
        return {'pass': True, 'videos': len(self.rows), 'unique_parameter_combinations': len(seen),
                'scientific_spec_sha256_checked': len(self.rows),
                'producer_sha256': self.producer_sha256,
                'protocol_sha256': _sha(self.source_root / 'protocol.json'),
                'manifest_sha256': _sha(self.source_root / 'manifest.json')}

    def verify_parameters_and_basis(self):
        """Rebuild all FP64 basis pixels and independently integrate sample pixels.

        Return every sample coordinate/error and fresh-versus-saved comparisons.
        A failing comparison raises; no source file is repaired or regenerated.
        """
        parameters = self.verify_parameters()
        report = {'schema': 't1-source-reaudit-basis-v1', 'pass': False,
                  'parameters': parameters, 'pixel_integrals': [],
                  'independent_integration_scope': 'sampled pixels per SF; all sample coordinates and errors retained',
                  'all_basis_pixels_recomputed': True,
                  'basis_algorithm': 'original FP64 analytic core sinc and 16x16 Gauss-Legendre taper'}
        rebuilt_root = None if self.output_root is None else self.output_root / 't1-basis'
        if rebuilt_root is not None:
            rebuilt_root.mkdir(parents=True, exist_ok=True)
        coords = self._producer.reference_pixels()
        for sf in SFS:
            started = time.monotonic()
            c, s = self._producer.build_basis(sf, order=16)
            saved_path = self.source_root / 'basis' / f'sf{_token(sf)}.npz'
            with np.load(saved_path, allow_pickle=False) as data:
                _require(float(data['sf']) == sf, 'Saved T1 basis SF identity mismatch')
                differences = []
                for name, array in [('cos', c), ('sin', s)]:
                    saved = data[name]
                    _require(saved.dtype == array.dtype == np.float64 and
                             saved.shape == array.shape == (HEIGHT, WIDTH),
                             'Saved T1 basis shape or FP64 dtype mismatch')
                    _require(np.isfinite(saved).all() and np.isfinite(array).all(),
                             'T1 basis contains non-finite values')
                    maximum = float(np.max(np.abs(saved - array)))
                    differences.append(maximum)
                    _require(np.array_equal(saved, array),
                             f'Rebuilt T1 {name} basis differs at SF={sf}: max error {maximum}')
            errors = []
            for x, y in coords:
                independent = np.asarray(self._producer.reference_integral(sf, x, y))
                _require(independent.shape == (2,) and np.isfinite(independent).all(),
                         f'Independent T1 integral is invalid SF={sf}, pixel=({x},{y})')
                errors.append({'pixel_xy': [x, y], 'coefficient_error': [
                    abs(float(c[y, x]) - float(independent[0])),
                    abs(float(s[y, x]) - float(independent[1]))]})
            maximum = max(max(item['coefficient_error']) for item in errors)
            _require(math.isfinite(maximum) and maximum <= COEFFICIENT_ERROR_LIMIT,
                     f'Independent T1 integral failed SF={sf}: {maximum}')
            c.flags.writeable = s.flags.writeable = False
            self._basis[sf] = (c, s)
            entry = {'sf': sf, 'pass': True, 'basis_sha256': _sha(saved_path),
                     'fresh_saved_max_absolute_error_cos_sin': differences,
                     'fresh_saved_all_values_equal': True, 'pixels': len(coords),
                     'max_coefficient_error': maximum,
                     'coefficient_error_limit': COEFFICIENT_ERROR_LIMIT,
                     'max_possible_linear_Y_error_at_m1': MEAN * maximum * math.sqrt(2),
                     'independent_algorithm': 'adaptive nested quadrature; explicit circle-intersection splits; no production window, GL nodes or raster mask',
                     'errors': errors, 'seconds': time.monotonic() - started}
            if rebuilt_root is not None:
                path = rebuilt_root / saved_path.name
                # These are new audit artifacts.  Existing source NPZs are read only.
                with path.open('wb') as stream:
                    np.savez_compressed(stream, cos=c, sin=s, sf=np.float64(sf))
                entry['rebuilt_basis_file'] = path.name
                entry['rebuilt_basis_sha256'] = _sha(path)
            report['pixel_integrals'].append(entry)
        report['pass'] = True
        report['rebuilt_basis_root'] = None if rebuilt_root is None else str(rebuilt_root)
        return report

    def _load_basis(self, sf):
        sf = float(sf)
        if sf not in self._basis:
            path = self.basis_root / f'sf{_token(sf)}.npz'
            with np.load(path, allow_pickle=False) as data:
                _require(float(data['sf']) == sf, 'T1 readback basis SF mismatch')
                c, s = data['cos'], data['sin']
            _require(c.dtype == s.dtype == np.float64 and
                     c.shape == s.shape == (HEIGHT, WIDTH), 'T1 readback basis dtype/shape mismatch')
            c.flags.writeable = s.flags.writeable = False
            self._basis[sf] = c, s
        return self._basis[sf]

    def _state(self, sf, depth, index):
        """Cache original floating operations for one (SF,m,24-phase) bank."""
        key = (float(sf), float(depth))
        if key not in self._banks:
            self._banks[key] = {}
            while len(self._banks) > self.max_cached_banks:
                self._banks.popitem(last=False)
        self._banks.move_to_end(key)
        bank = self._banks[key]
        if index not in bank:
            c, s = self._load_basis(sf)
            phase = math.radians(7.5 + 15 * index)
            # Keep the producer's operation order.  Do not replace sin/cos with
            # symmetries or a continuous f%period shortcut before quantization.
            ideal = MEAN * (1 + depth * (c * math.cos(phase) - s * math.sin(phase)))
            code = self._producer.quantize(ideal)
            maximum = float(np.max(np.abs(self._producer.linear_decode(code) - ideal)))
            _require(maximum <= QUANTIZATION_ERROR_LIMIT,
                     f'T1 source quantization failed: {maximum}')
            code.flags.writeable = False
            bank[index] = (code, maximum)
        return bank[index]

    def expected_frame(self, row, frame_index):
        _require(type(frame_index) is int and 0 <= frame_index < FRAMES,
                 'T1 frame index must be an integer in [0,180)')
        params = row['params']
        index = (row['phase_index'] - 2 * params['direction_sign'] * (frame_index % 12)) % 24
        return self._state(params['sf_cycles_per_aperture'], params['modulation_depth'], index)[0]

    def begin_video(self, row):
        """Generate the 12 source states once and return photometric QA metrics."""
        params = row['params']
        errors = []
        for frame_index in range(12):
            index = (row['phase_index'] - 2 * params['direction_sign'] * frame_index) % 24
            errors.append(self._state(params['sf_cycles_per_aperture'],
                                      params['modulation_depth'], index)[1])
        return {'source_states': 12,
                'max_source_linear_Y_quantization_error': max(errors),
                'source_linear_Y_quantization_error_limit': QUANTIZATION_ERROR_LIMIT}

    @staticmethod
    def extract_line(rgb):
        return extract_line(rgb)

    def pixel_readout(self, row, lines):
        """Blindly recover SF, m, 5Hz, direction, phase and mean from all 180 lines."""
        codes = np.asarray(lines)
        _require(codes.dtype == np.uint16 and codes.shape == (FRAMES, LINE_STOP - LINE_START),
                 'T1 blind readout requires all 180x380 uint16 decoded core lines')
        result = self._producer.measure_line(codes, row)
        # Treat a numerical failure as invalid even if a future implementation
        # accidentally relaxes a comparison involving NaN.
        _require(all(math.isfinite(float(value)) for section in ('estimated', 'errors')
                     for value in result[section].values() if not isinstance(value, str)),
                 'T1 blind readout produced non-finite estimates')
        return result
