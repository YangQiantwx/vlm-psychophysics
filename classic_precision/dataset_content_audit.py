"""Read-only decoded-content checks and byte-preserving T1--T3 release staging.

The immutable input joins source records, the proposed mapping, and SHA-bound
full-frame QA. Existing sufficient measurements are reused. Missing spatial
measurements are made from decoded RGB before comparing labels. No inference.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time

import av
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def array_sha(a):
    return hashlib.sha256(a.tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def linear(a):
    return np.where(a <= .04045, a / 12.92, ((a + .055) / 1.055)**2.4)


def rgb8(a):
    require(np.isfinite(a).all() and a.min() >= -1e-10 and a.max() <= 1 + 1e-10,
            'Invalid source-over RGB range')
    a = np.clip(a, 0, 1)
    return np.rint(255 * np.where(a <= .0031308, a * 12.92,
                                1.055 * a**(1 / 2.4) - .055)).astype(np.uint8)


def signed_axis(alpha):
    """Full alpha second moments. Positive: upper end right, x right/y down."""
    import cv2
    cv2.setNumThreads(1)
    m = cv2.moments(np.asarray(alpha, np.float64), binaryImage=False)
    require(m['m00'] > 0, 'Empty alpha')
    cov = np.array([[m['mu20'], m['mu11']], [m['mu11'], m['mu02']]]) / m['m00']
    vals, vec = np.linalg.eigh(cov)
    v = vec[:, -1]
    # Undirected axis modulo 180, with vertical zero and horizontal -90.
    angle = (math.degrees(math.atan2(v[1], v[0])) - 90 + 90) % 180 - 90
    return {'signed_deviation_deg': angle, 'deviation_deg': abs(angle),
            'centroid_xy': [m['m10'] / m['m00'], m['m01'] / m['m00']],
            'alpha_area_px2': m['m00'], 'axis_ratio': float(vals[-1] / vals[0])}


def classify_axes(axes, tolerance):
    ordered = sorted(axes, key=lambda x: x['deviation_deg'])
    require(ordered[0]['deviation_deg'] <= tolerance, 'Neither object is vertical')
    require(ordered[1]['deviation_deg'] > ordered[0]['deviation_deg'] + tolerance,
            'Vertical side is ambiguous')
    tilted = ordered[1]
    direction = ('horizontal' if abs(tilted['deviation_deg'] - 90) <= tolerance else
                 'right' if tilted['signed_deviation_deg'] > 0 else 'left')
    return {'axes': axes, 'vertical_side': ordered[0]['side'],
            'tilted_side': tilted['side'], 'angle_deg': tilted['deviation_deg'],
            'upper_end_direction': direction}


def decode_states(path, event_index, domain):
    """Decode baseline plus event; full sequence/timing remains bound by old QA."""
    fmt = 'rgb48le' if domain == 'classic' else 'rgb24'
    arrays, pts = {}, {}
    with av.open(str(path)) as c:
        require(len(c.streams.video) == 1, 'Expected one video stream')
        s = c.streams.video[0]
        s.codec_context.thread_count = 1
        require((s.width, s.height) == (768, 512) and s.average_rate == 60,
                'Dimensions/FPS changed')
        codec, pixfmt = s.codec_context.name, s.codec_context.format.name
        require(codec == ('ffv1' if domain == 'classic' else 'h264'), 'Codec changed')
        for i, f in enumerate(c.decode(video=0)):
            if i in (0, event_index):
                arrays[i] = f.to_ndarray(format=fmt)
                pts[i] = float(f.pts * f.time_base)
            if i == event_index:
                break
        require(set(arrays) == {0, event_index}, 'Missing baseline/event frame')
        require(all(abs(pts[i] - i / 60) <= (.001 if domain == 'classic' else .000001)
                    for i in arrays), 'Selected frame PTS changed')
    return arrays[0], arrays[event_index], {'decoded_frame_indices': [0, event_index],
        'decoded_frame_sha256': [array_sha(arrays[0]), array_sha(arrays[event_index])],
        'selected_PTS_s': [pts[0], pts[event_index]], 'codec': codec,
        'pixel_format': pixfmt, 'width': 768, 'height': 512, 'fps': 60}


def measured_side(base, event):
    changed = np.any(base != event, axis=-1)
    left, right = int(changed[:, :384].sum()), int(changed[:, 384:].sum())
    require((left > 0) != (right > 0), 'Change is missing or appears on both sides')
    yy, xx = np.where(changed)
    return {'side': 'LEFT' if left else 'RIGHT', 'changed_pixels_by_side': [left, right],
            'changed_bbox_xyxy': [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]}


@lru_cache(maxsize=24)
def patches(directory, object_id):
    """Every stored orientation for this source object, never target angle/sign."""
    out, seen = [], set()
    for p in sorted(Path(directory).glob(object_id + '_*.npy')):
        h = sha(p)
        if h in seen:
            continue
        seen.add(h)
        meta = read(p.with_suffix('.json'))
        require(h == meta['patch_sha256'], 'Patch bytes differ from original source QA')
        z = np.load(p, allow_pickle=False)
        require(z.shape == (320, 320, 4) and z.dtype == np.float64, 'Patch type changed')
        out.append((p.name, h, meta['source_sha256'], z, signed_axis(z[:, :, 3])))
    require(out, 'No source alpha candidates')
    return out


@lru_cache(maxsize=24)
def background(root, asset_id):
    p = Path(root) / 'datasets/natural-t13-20261029/backgrounds' / (asset_id + '.npy')
    meta = read(p.with_suffix('.json'))
    require(sha(p) == meta['prepared_sha256'], 'Background prepared SHA differs')
    return np.load(p, allow_pickle=False), meta


def match_alpha(base, event, bg, candidates):
    require(np.array_equal(rgb8(bg), base), 'Decoded baseline/background mismatch')
    axes, masks = [], []
    for side, x in [('LEFT', 64), ('RIGHT', 384)]:
        b, observed = bg[144:464, x:x + 320], event[144:464, x:x + 320]
        matches = []
        # A cheap grid rejects candidates. The survivor is checked at every RGB code.
        for name, h, source_sha, z, measurement in candidates:
            small = z[::5, ::5]
            if not np.array_equal(rgb8(small[:, :, :3] + (1 - small[:, :, 3, None]) * b[::5, ::5]),
                                  observed[::5, ::5]):
                continue
            predicted = rgb8(z[:, :, :3] + (1 - z[:, :, 3, None]) * b)
            if np.array_equal(predicted, observed):
                matches.append((name, h, source_sha, z, measurement))
        require(len(matches) == 1, f'{side}: unique alpha/RGB binding failed ({len(matches)} matches)')
        name, h, source_sha, z, measurement = matches[0]
        axes.append({**measurement, 'side': side, 'patch_file': name,
                     'patch_sha256': h, 'source_object_sha256': source_sha,
                     'candidate_count': len(candidates), 'exact_matches': len(matches),
                     'RGB_max_code_error': 0,
                     'canvas_centroid_xy': [measurement['centroid_xy'][0] + x,
                                            measurement['centroid_xy'][1] + 144]})
        mask = np.zeros(base.shape[:2], bool)
        mask[144:464, x:x + 320] = z[:, :, 3] > 0
        masks.append(mask)
    require(np.array_equal(base[~(masks[0] | masks[1])], event[~(masks[0] | masks[1])]),
            'Pixels changed outside the independently matched silhouettes')
    return classify_axes(axes, .01)


def parse_name(path, domain, task):
    name = Path(path).name
    # Source-pixel revisions retain the readable condition and carry a short
    # revision suffix; catalog SHA and parent lineage remain authoritative.
    if domain == 'natural':
        name = re.sub(r'_rev[0-9a-f]{8}(?=\.mp4$)', '', name)
    number = r'(\d+(?:\.\d+)?)'
    if task == 'T1':
        p = r'T1_(left|right)_' + (r'bg-(na_\d{3}|gpt_\d{3})_' if domain == 'natural' else '')
        m = re.fullmatch(p + r'sf' + number + r'_contrast' + number + r'_trial(\d{2})\.(?:mkv|mp4)', name)
        require(m, 'Unrecognized T1 filename')
        a = list(m.groups()); side = a.pop(0).upper()
        bg = a.pop(0) if domain == 'natural' else None
        return {'side': side, 'background': bg, 'sf': float(a[0]), 'm': float(a[1]), 'trial': int(a[2])}
    if task == 'T2':
        source = r'bg-(na_\d{3}|gpt_\d{3})_obj-(na_\d{3}|gpt_\d{3})' if domain == 'natural' else 'bg' + number
        m = re.fullmatch(r'T2_(left|right)_' + source + r'_delta' + number + r'_(\d+)frames_trial(\d{2})\.(?:mkv|mp4)', name)
        require(m, 'Unrecognized T2 filename')
        a = list(m.groups()); side = a.pop(0).upper(); bg = a.pop(0)
        obj = a.pop(0) if domain == 'natural' else None
        return {'side': side, 'background': bg, 'object': obj, 'delta': float(a[0]), 'duration': int(a[1]), 'trial': int(a[2])}
    m = re.fullmatch(r'T3_vertical(Left|Right)_(tiltLeft|tiltRight|horizontal)_' + number + r'deg_' +
                     (r'bg-(na_\d{3}|gpt_\d{3})_obj-(na_\d{3}|gpt_\d{3})_' if domain == 'natural' else '') +
                     r'trial(\d{2})\.(?:mkv|mp4)', name)
    require(m, 'Unrecognized T3 filename')
    a = list(m.groups()); side, direction, angle = a[:3]; a = a[3:]
    bg, obj = (a.pop(0), a.pop(0)) if domain == 'natural' else (None, None)
    return {'side': side.upper(), 'direction': {'tiltLeft': 'left', 'tiltRight': 'right', 'horizontal': 'horizontal'}[direction],
            'angle': float(angle), 'background': bg, 'object': obj, 'trial': int(a[0])}


def alias(s):
    return s.replace('photo_', 'na_').replace('bg_', '').replace('obj_', '')


def compare(mapping, source, observed, tolerances):
    domain, task = mapping['domain'], mapping['task']
    n = parse_name(mapping['proposed_public_path'], domain, task)
    c = json.loads(mapping['condition_key'])
    require(n['side'] == mapping['truth'] == (observed['vertical_side'] if task == 'T3' else observed['side']), 'Measured side disagrees with name/GT')
    require(c[-1] == n['side'], 'Condition side disagrees')
    if domain == 'natural':
        require(n['background'] == alias(source['background_id']), 'Background source alias changed')
        if task != 'T1':
            require(n['object'] == alias(source['object_id']), 'Object source alias changed')
    if task == 'T1':
        require(abs(observed['sf'] - n['sf']) <= tolerances['sf_absolute'], 'SF/name mismatch')
        require(abs(observed['m'] - n['m']) <= tolerances['m_absolute'], 'Modulation/name mismatch')
        require(n['sf'] == float(c[1]) and n['m'] == float(c[2]) and n['trial'] - 1 == c[3], 'T1 condition mismatch')
        require(abs(observed['phase_error_rad']) <= tolerances['phase_radians'], 'Phase/trial mismatch')
        if domain == 'classic':
            expected_phase = math.radians(7.5 + 15 * (n['trial'] - 1))
            difference = (observed['initial_phase_rad'] - expected_phase + math.pi) % (2 * math.pi) - math.pi
            require(abs(difference) <= tolerances['phase_radians'], 'Measured phase/name trial mismatch')
        else:
            require(n['trial'] - 1 == source['phase_index'], 'Trial differs from phase-bound source measurement')
    elif task == 'T2':
        error = abs(observed['delta_Y'] - n['delta'])
        require(error <= tolerances['delta_absolute'], 'Brightness/name mismatch')
        if domain == 'natural':
            require(error / n['delta'] <= tolerances['delta_relative'], 'Relative brightness error')
        else:
            require(abs(observed['background_Y'] - float(n['background'])) <= .0001, 'Background/name mismatch')
        require(n['delta'] == float(c[1]) and n['duration'] == observed['offset_frame'] - observed['onset_frame'] == c[3], 'T2 duration/condition mismatch')
        require(18 + 5 * (n['trial'] - 1) == observed['onset_frame'] == c[2], 'T2 onset/trial mismatch')
    else:
        require(abs(observed['angle_deg'] - n['angle']) <= tolerances['axis_deg'], 'Axis angle/name mismatch')
        require(observed['upper_end_direction'] == n['direction'] == mapping['physical_upper_end_direction'], 'Physical direction/name mismatch')
        sign = {'horizontal': 0, 'left': 1, 'right': -1}[observed['upper_end_direction']]
        require(sign == c[2] and n['angle'] == float(c[1]), 'T3 physical condition mismatch')
        require(observed['onset_frame'] == 18 + 5 * (n['trial'] - 1) == c[3], 'T3 onset/trial mismatch')
    return {'filename': True, 'GT': True, 'condition': True, 'source_aliases': True}


def audit(job):
    item, root, stage, support, version, input_digest = job
    start = time.monotonic()
    m, s, old = item['mapping'], item['source'], item['prior_qa']
    domain, task = m['domain'], m['task']
    r = {'schema': 'source-content-naming-v1', 'domain': domain, 'task': task,
         'legacy_sample_id': m['legacy_sample_id'], 'sample_id': m['proposed_sample_id'],
         'old_path': m['legacy_path'], 'path': m['proposed_public_path'],
         'source_relative': item['source_relative'], 'source_sha256': m['video_sha256'],
         'video_sha256': None, 'source_record_sha256': m['source_record_sha256'],
         'code_sha256': version, 'input_sha256': input_digest, 'status': 'FAIL', 'failed_reason': None}
    try:
        source = Path(root) / item['source_relative']
        target = Path(stage) / domain / m['proposed_public_path']
        require(source.is_file(), 'Missing source: ' + item['source_relative'])
        before = source.stat()
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.link(source.resolve(), target)
        require(not target.is_symlink(), 'Staged media must be a regular file')
        h = sha(target)
        require(h == m['video_sha256'] == s['video_sha256'], 'Actual target SHA differs from source identity')
        after = source.stat()
        require((after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) ==
                (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns), 'Source changed during read')
        expected_bytes = old['source_stat']['size'] if old else s.get('bytes', s.get('video_bytes'))
        require(target.stat().st_size == before.st_size == expected_bytes, 'Media byte length changed')
        r.update(video_sha256=h, bytes=before.st_size, source_target_same_inode=os.path.samefile(source, target))
        if old:
            require(old['status'] == 'PASS' and old['video_sha256'] == h and old['decoded_frames'] == 180 and old['every_decoded_RGB_code_compared'], 'Inherited full-frame QA mismatch')
            stream_sha = old['decoded_stream_sha256']
            inherited = {'file': old['inherited_QA_file'], 'file_sha256': old['inherited_QA_file_sha256'],
                         'identity': old['identity'], 'measured_from': 'all180 decoded RGB codes',
                         'max_PTS_error_s': old['max_PTS_error_s']}
        else:
            require(s.get('frames_byte_equal', s.get('decoded_frames_exact')) == 180, 'Missing full-frame source binding')
            stream_sha = s.get('RGB_stream_sha256', s.get('stream_sha256'))
            inherited = {'source_record_sha256': m['source_record_sha256'], 'qa_sha256': s.get('qa_sha256'),
                         'measured_from': 'source states equal to all180 decoded RGB frames',
                         'render_code_sha256': s.get('render_implementation_sha256')}
        require(stream_sha == m['decoded_stream_sha256'], 'Decoded stream identity mismatch')
        r.update(decoded_stream_sha256=stream_sha, inherited_full_frame_QA=inherited, frames=180, fps=60,
                 width=768, height=512, codec='ffv1' if domain == 'classic' else 'h264',
                 pixel_format='gbrp16le' if domain == 'classic' else 'gbrp')
        if task == 'T1':
            if domain == 'classic':
                q = old['independent_pixel_measurements']; e = q['estimated']
                require(q['pass'], 'Prior blind T1 measurement failed')
                observed = {'side': e['direction'], 'sf': e['sf_cycles_per_aperture'],
                            'm': e['modulation_depth'], 'phase_error_rad': q['errors']['initial_phase_rad'],
                            'initial_phase_rad': e['initial_phase_rad'],
                            'temporal_frequency_hz': e['temporal_frequency_hz'], 'mean_linear_Y': e['mean_linear_Y']}
                tol = {'sf_absolute': .001, 'm_absolute': .0001, 'phase_radians': .001,
                       'temporal_hz': .00001}
                require(abs(e['temporal_frequency_hz'] - 5) <= tol['temporal_hz'], 'T1 temporal frequency mismatch')
                method = q['method'] + '; reused SHA-bound all-frame decoded measurement'
            else:
                q = s['parameter_measurement']; require(q['passed'], 'Natural T1 prior QA failed')
                observed = {'side': 'RIGHT' if q['direction_sign'] > 0 else 'LEFT',
                            'sf': q['measured_sf'], 'm': q['measured_m'], 'phase_error_rad': q['phase_error_rad']}
                tol = {'sf_absolute': .015, 'm_absolute': .003, 'phase_radians': .05}
                method = q['method'] + '; reused source-state measurement with all180 decoded equality'
        elif task == 'T2':
            temporal = old['event_frame_interval_half_open'] if old else [s['onset_frame'], s['offset_frame']]
            base, event, decoded = decode_states(target, temporal[0], domain)
            observed = {**measured_side(base, event), 'onset_frame': temporal[0], 'offset_frame': temporal[1],
                        'temporal_method': 'inherited same-SHA full180 event-state equality', **decoded}
            if domain == 'classic':
                q = old['independent_pixel_measurements']
                observed.update(delta_Y=q['core_increment'], alpha_regression_delta_Y=q['alpha_regression_increment'],
                                background_Y=float(linear(base[0, 0, 0] / 65535.)))
                require(q['core_max_error'] < .0001 and q['alpha_regression_error'] < .0001 and q['changed_outside_target'] == 0, 'Classic T2 source photometry failed')
                tol = {'delta_absolute': .0001, 'background_absolute': .0001}
            else:
                require(array_sha(base) == s['baseline_rgb8_sha256'] and array_sha(event) == s['event_rgb8_sha256'], 'T2 selected decoded frame hash differs')
                measured_delta = float(((linear(event / 255.) - linear(base / 255.)) @ np.array([.2126, .7152, .0722])).sum() / s['alpha_area_px2'])
                require(abs(measured_delta - s['decoded_delta_Y']) < 1e-12 and s['outside_target_identical'], 'T2 original delta measurement changed')
                observed.update(delta_Y=measured_delta, alpha_area_px2=s['alpha_area_px2'])
                tol = {'delta_absolute': .002, 'delta_relative': .02}
            method = 'decoded baseline/event independent half-image change; SHA-bound full180 temporal/photometric QA'
        else:
            if domain == 'classic':
                q = old['independent_pixel_measurements']
                axes = [{'side': a['side'], 'signed_deviation_deg': -a['PCA_deg'],
                         'deviation_deg': abs(a['PCA_deg']), 'PCA_deg_classic_convention': a['PCA_deg'],
                         'halfplane_deg_classic_convention': a['halfplane_deg']} for a in q['axes']]
                require(all(a['PCA_error_deg'] < .001 and a['halfplane_error_deg'] < .001 for a in q['axes']), 'Inherited T3 geometry failed')
                observed = classify_axes(axes, .001)
                temporal = old['event_frame_interval_half_open']
                tol = {'axis_deg': .001}
                method = 'independent full-silhouette PCA and halfplane fit; reused all180 decoded RGB binding'
            else:
                temporal = [s['onset'], s['parameter_measurement']['offset']]
                base, event, decoded = decode_states(target, temporal[0], domain)
                bg, bgmeta = background(root, s['background_id'])
                replacement = s['source_dataset'] == 'natural-t3-spoon-20261030'
                directory = Path(support) / ('patches-spoon' if replacement else 'patches-original')
                candidates = patches(str(directory), s['object_id'])
                observed = match_alpha(base, event, bg, candidates)
                prior = item['natural_qa']
                expected_patches = {x['patch_sha256'] for x in prior['patches']}
                require({x['patch_sha256'] for x in observed['axes']} == expected_patches,
                        'Measured alpha differs from original full-frame source QA')
                require(bgmeta['source_sha256'] == prior['source_background_sha256'],
                        'Background source differs from original QA')
                observed.update(decoded, background_source_sha256=bgmeta['source_sha256'],
                                background_prepared_sha256=bgmeta['prepared_sha256'])
                tol = {'axis_deg': .01, 'RGB_code_error': 0, 'unique_RGBA_matches': 1}
                method = 'all-source-orientation candidates, unique exact decoded RGB source-over match per side; independent full-alpha OpenCV moments'
            observed.update(onset_frame=temporal[0], offset_frame=temporal[1],
                            temporal_method='inherited same-SHA full180 event-state equality')
        r.update(measured=observed, method=method, tolerances=tol)
        r['comparison'] = compare(m, s, observed, tol)
        r['status'] = 'PASS'
    except Exception as e:
        r['failed_reason'] = type(e).__name__ + ': ' + str(e)
    r['seconds'] = time.monotonic() - start
    return r


def run(args):
    items = [json.loads(line) for line in args.input.open()]
    version = sha(__file__)
    input_digest = sha(args.input)
    if args.pilot:
        selected = []
        for domain in ['classic', 'natural']:
            for task in ['T1', 'T2', 'T3']:
                group = [x for x in items if x['mapping']['domain'] == domain and x['mapping']['task'] == task]
                selected.extend([group[0], group[-1]])
        # Cover replacement source, both signs, near vertical and horizontal.
        for sign in ['left', 'right', 'horizontal']:
            group = [x for x in items if x['mapping']['domain'] == 'natural' and x['mapping']['task'] == 'T3' and x['mapping']['physical_upper_end_direction'] == sign]
            selected.extend(group[:2])
            selected.extend([x for x in group if x['source']['source_dataset'] == 'natural-t3-spoon-20261030'][:2])
        items = list({x['mapping']['legacy_sample_id']: x for x in selected}.values())
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / ('pilot.jsonl' if args.pilot else 'per-video.jsonl')
    require(not output.exists() or args.resume, 'Refusing to overwrite an existing content audit')
    existing = [json.loads(l) for l in output.open()] if output.exists() else []
    require(all(r['status'] == 'PASS' and r['code_sha256'] == version and
                r.get('input_sha256') == input_digest for r in existing),
            'Resume requires unchanged code/input and passing records')
    done = {r['legacy_sample_id'] for r in existing}
    expected_ids = {x['mapping']['legacy_sample_id'] for x in items}
    require(len(expected_ids) == len(items), 'Duplicate input sample IDs')
    require(len(done) == len(existing) and done <= expected_ids, 'Duplicate or foreign existing records')
    todo = [x for x in items if x['mapping']['legacy_sample_id'] not in done]
    start = time.monotonic(); records = existing[:]
    summary_path = args.out / ('pilot-summary.json' if args.pilot else 'summary.json')
    def summary(status):
        return {'schema': 'source-content-naming-v1', 'status': status, 'expected': len(items),
                'completed': len(records), 'counts': dict(Counter(r['domain'] + '/' + r['task'] for r in records)),
                'passed': sum(r['status'] == 'PASS' for r in records),
                'failures': [{'sample_id': r['sample_id'], 'reason': r['failed_reason']} for r in records if r['status'] != 'PASS'],
                'code_sha256': version, 'input_sha256': input_digest, 'workers': args.workers,
                'seconds_this_dispatch': time.monotonic() - start, 'model_calls': 0,
                'media_generated_or_reencoded': 0, 'source_media_mutations': 0,
                'measurement_policy': 'All targets hashed. Sufficient same-video-SHA decoded QA reused; missing spatial fields measured from actual decoded RGB.'}
    write(summary_path, summary('RUNNING'))
    jobs = [(x, str(args.root), str(args.stage), str(args.support), version, input_digest) for x in todo]
    with output.open('a') as f, ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(audit, j) for j in jobs]
        for future in as_completed(futures):
            r = future.result(); records.append(r); f.write(json.dumps(r, separators=(',', ':'), allow_nan=False) + '\n'); f.flush()
            if r['status'] != 'PASS':
                print(json.dumps({'failure': r['sample_id'], 'reason': r['failed_reason']}), flush=True)
            if len(records) % 100 == 0:
                write(summary_path, summary('RUNNING'))
                print(json.dumps({'completed': len(records), 'total': len(items), 'seconds': round(time.monotonic() - start, 1)}), flush=True)
    require(len(records) == len(items) and {r['legacy_sample_id'] for r in records} == expected_ids,
            'Incomplete or duplicate result set')
    result = summary('PASS' if all(r['status'] == 'PASS' for r in records) else 'FAIL')
    result['per_video_sha256'] = sha(output)
    write(summary_path, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'failures'}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--support', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--resume', action='store_true')
    run(p.parse_args())


if __name__ == '__main__':
    main()
