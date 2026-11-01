"""Two finite RGB16 T2/T3 examples; rendering runs on the workstation.

Disk coverage integrates circle/box intersections analytically. Independent
SciPy quadrature checks every boundary pixel. T3 reuses the audited strip kernel.
MP4/PNG outputs are viewing copies, never the scientific model inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from PIL import Image
from scipy.integrate import quad

import t3_precision_reference as t3


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def circle_pixel_area(x0, x1, y0, y1, radius):
    """Integral of max(0,min(y1,h)-max(y0,-h)), h=sqrt(r²-x²)."""
    lo, hi = max(x0, -radius), min(x1, radius)
    if lo >= hi:
        return 0.0
    cuts = [lo, hi]
    for y in (y0, y1):
        if abs(y) < radius:
            crossing = math.sqrt(max(0.0, radius * radius - y * y))
            cuts.extend(x for x in (-crossing, crossing) if lo < x < hi)
    cuts = sorted(set(cuts))

    def primitive(x):
        h = math.sqrt(max(0.0, radius * radius - x * x))
        return .5 * (x * h + radius * radius * math.asin(np.clip(x / radius, -1, 1)))

    area = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        h = math.sqrt(max(0.0, radius * radius - ((a + b) / 2) ** 2))
        if min(y1, h) <= max(y0, -h):
            continue
        top_const, top_h = (y1, 0) if y1 <= h else (0, 1)
        bot_const, bot_h = (y0, 0) if y0 >= -h else (0, -1)
        area += (top_const - bot_const) * (b - a) + (top_h - bot_h) * (primitive(b) - primitive(a))
    return float(np.clip(area, 0, 1))


def disk_coverage(shape=(512, 768), center=(224., 302.), area=5118.7529296875):
    radius = math.sqrt(area / math.pi)
    alpha = np.zeros(shape, np.float64)
    boundary_errors = []
    cx, cy = center
    for y in range(max(0, math.floor(cy - radius)), min(shape[0], math.ceil(cy + radius))):
        for x in range(max(0, math.floor(cx - radius)), min(shape[1], math.ceil(cx + radius))):
            dx, dy = max(abs(x - cx), abs(x + 1 - cx)), max(abs(y - cy), abs(y + 1 - cy))
            if dx * dx + dy * dy <= radius * radius:
                alpha[y, x] = 1.
                continue
            x0, x1, y0, y1 = x - cx, x + 1 - cx, y - cy, y + 1 - cy
            value = circle_pixel_area(x0, x1, y0, y1, radius)
            alpha[y, x] = value
            if 1e-12 < value < 1 - 1e-12:
                lo, hi = max(x0, -radius), min(x1, radius)
                cuts = []
                for edge in (y0, y1):
                    if abs(edge) < radius:
                        q = math.sqrt(radius * radius - edge * edge)
                        cuts.extend(v for v in (-q, q) if lo < v < hi)
                def integrand(v):
                    h = math.sqrt(max(0., radius * radius - v * v))
                    return max(0., min(y1, h) - max(y0, -h))
                independent, _ = quad(integrand, lo, hi, points=sorted(set(cuts)), epsabs=1e-11, epsrel=1e-11)
                boundary_errors.append(abs(value - independent))
    area_error = abs(float(alpha.sum()) - area)
    assert area_error < 1e-8, area_error
    assert max(boundary_errors, default=0) < 1e-8
    return alpha, {'radius_px': radius, 'diameter_px': 2 * radius,
                   'target_area_px2': area, 'actual_alpha_area_px2': float(alpha.sum()),
                   'area_absolute_error_px2': area_error, 'independent_boundary_pixels': len(boundary_errors),
                   'quadrature_max_coverage_error': max(boundary_errors, default=0)}


def encode_linear(values):
    assert np.min(values) >= 0 and np.max(values) <= 1
    gray = np.rint(t3.srgb(values) * 65535).astype(np.uint16)
    return np.repeat(gray[:, :, None], 3, axis=2)


def write_and_verify(out, stem, baseline, event, spec):
    master = out / f'{stem}_RGB16.mkv'
    preview = out / f'{stem}_preview.mp4'
    on, off = spec['onset_frame'], spec['offset_frame']
    source_hash = hashlib.sha256()
    with av.open(str(master), 'w') as container:
        stream = container.add_stream('ffv1', rate=60)
        stream.width, stream.height = 768, 512
        stream.pix_fmt = 'gbrp16le'
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = 0
        stream.options = {'level': '3', 'coder': '1', 'context': '1', 'slicecrc': '1'}
        for i in range(180):
            rgb = event if on <= i < off else baseline
            source_hash.update(rgb.tobytes())
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb48le')
            frame.pts, frame.time_base = i, Fraction(1, 60)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    actual_hash = hashlib.sha256()
    times, changed = [], []
    with av.open(str(master)) as container:
        pixel_format = container.streams.video[0].codec_context.format.name
        assert pixel_format == 'gbrp16le'
        for i, frame in enumerate(container.decode(video=0)):
            actual = frame.to_ndarray(format='rgb48le')
            expected = event if on <= i < off else baseline
            assert actual.dtype == np.uint16 and np.array_equal(actual, expected), i
            actual_hash.update(actual.tobytes())
            times.append(float(frame.pts * frame.time_base))
            if not np.array_equal(actual, baseline):
                changed.append(i)
            if i == on:
                decoded_event = actual
    assert len(times) == 180 and changed == list(range(on, off))
    assert source_hash.hexdigest() == actual_hash.hexdigest()
    pts_error = max(abs(t - i / 60) for i, t in enumerate(times))
    assert pts_error <= .001
    base8, event8 = [np.rint(rgb.astype(np.float64) / 65535 * 255).astype(np.uint8) for rgb in (baseline, event)]
    Image.fromarray(event8).save(out / f'{stem}_event.png')
    with av.open(str(preview), 'w') as container:
        stream = container.add_stream('libx264', rate=60)
        stream.width, stream.height, stream.pix_fmt = 768, 512, 'yuv420p'
        stream.options = {'crf': '18', 'preset': 'fast', 'threads': '2'}
        for i in range(180):
            frame = av.VideoFrame.from_ndarray(event8 if on <= i < off else base8, format='rgb24')
            frame.pts, frame.time_base = i, Fraction(1, 60)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return decoded_event, {**spec, 'master': master.name, 'preview': preview.name,
        'source_wh': [768, 512], 'fps': 60, 'frames': 180,
        'master_bits_per_channel': 16, 'master_codec': 'ffv1', 'master_pixel_format': pixel_format,
        'max_decoded_channel_error': 0, 'source_RGB_stream_sha256': source_hash.hexdigest(),
        'decoded_RGB_stream_sha256': actual_hash.hexdigest(), 'max_PTS_error_s': pts_error,
        'preview_bits_per_channel': 8, 'preview_is_scientific_input': False,
        'files': {f.name: {'sha256': digest(f), 'bytes': f.stat().st_size} for f in
                  (master, preview, out / f'{stem}_event.png')}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--delta-y', type=float, default=.09)
    parser.add_argument('--angle-deg', type=float, default=.5)
    args = parser.parse_args()
    delta_y, angle_deg = args.delta_y, args.angle_deg
    assert 0 < delta_y <= .5 and 0 < angle_deg <= 90
    def label(value):
        return format(value, '.12g').replace('.', 'p')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'examples_qualification.json').exists(), 'Preserve completed example receipts.'
    left, circle_qa = disk_coverage()
    right = np.roll(left, 320, axis=1)
    assert not np.any((left > 0) & (right > 0))
    baseline_Y = .20 + (left + right) * (.50 - .20)
    event_Y = baseline_Y + left * delta_y
    baseline, event = encode_linear(baseline_Y), encode_linear(event_Y)
    decoded, t2_record = write_and_verify(out, 't2_rect_dY' + label(delta_y), baseline, event,
        {'task': 'T2', 'background_Y': .20, 'initial_dot_Y': .50, 'delta_Y': delta_y,
         'centers_xy': [[224, 302], [544, 302]], 'ground_truth': 'LEFT',
         'onset_frame': 63, 'offset_frame': 93, 'circle': circle_qa})
    actual_delta = t3.linear(decoded[:, :, 0].astype(np.float64) / 65535) - t3.linear(baseline[:, :, 0].astype(np.float64) / 65535)
    core = left == 1
    roi = left > 0
    slope = float(np.dot(left[roi], actual_delta[roi]) / np.dot(left[roi], left[roi]))
    core_error = float(np.max(np.abs(actual_delta[core] - delta_y)))
    outside_changed = int(np.count_nonzero(decoded[~roi] != baseline[~roi]))
    assert core_error < .0001 and abs(slope - delta_y) < .0001 and outside_changed == 0
    t2_record['decoded_photometry'] = {'core_increment': float(actual_delta[core].mean()),
        'core_max_increment_error': core_error, 'alpha_regression_increment': slope,
        'alpha_regression_increment_error': abs(slope - delta_y),
        'max_ROI_increment_residual': float(np.max(np.abs(actual_delta[roi] - left[roi] * delta_y))),
        'outside_ROI_changed_channel_values': outside_changed}
    a = t3.strip_coverage(0, shape=(512, 768), center=(224, 304))
    a += t3.strip_coverage(angle_deg, shape=(512, 768), center=(544, 304))
    expected = t3.geos_coverage(0, shape=(512, 768), center=(224, 304))
    expected += t3.geos_coverage(angle_deg, shape=(512, 768), center=(544, 304))
    geos_difference = float(np.max(np.abs(a - expected)))
    assert geos_difference < 1e-9
    event = encode_linear(.10 + (.82 - .10) * a)
    baseline = encode_linear(np.full((512, 768), .10))
    decoded, t3_record = write_and_verify(out, 't3_rect_' + label(angle_deg) + 'deg', baseline, event,
        {'task': 'T3', 'background_Y': .10, 'foreground_Y': .82, 'bar_wh': [20, 190],
         'centers_xy': [[224, 304], [544, 304]], 'nominal_angle_deg': angle_deg,
         'tilted_side': 'RIGHT', 'ground_truth': 'LEFT', 'onset_frame': 63, 'offset_frame': 93})
    axes = []
    for cx, nominal in ((224, 0.), (544, angle_deg)):
        roi_codes = decoded[176:432, cx - 128:cx + 128, 0]
        pca = t3.pca_axis(t3.decode(roi_codes, 16))
        blind = t3.quantized_edge_axis(roi_codes, 16)
        assert blind['status'] == 'ok'
        record = {'nominal_angle_deg': nominal, 'PCA_angle_deg': pca,
                  'PCA_error_deg': t3.angle_error(pca, nominal),
                  'blind_angle_deg': blind['axis_deg'], 'blind_error_deg': t3.angle_error(blind['axis_deg'], nominal)}
        assert record['PCA_error_deg'] < .001 and record['blind_error_deg'] < .001
        axes.append(record)
    t3_record['decoded_axis_verification'] = axes
    t3_record['independent_GEOS_max_coverage_difference'] = geos_difference
    receipt = {'status': 'COMPLETE', 'scope': 'Two examples only; no formal batch or VLM calls',
               'new_model_calls': 0, 'source_file': str(Path(__file__).resolve()),
               'source_sha256': digest(__file__), 'T3_kernel_sha256': digest(t3.__file__),
               'records': [t2_record, t3_record]}
    (out / 'examples_qualification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
