"""Read-only, resumable CPU audit of the released classic T1/T2/T3 sources.

Every encoded frame is decoded as RGB48 and compared with freshly reconstructed,
numerically measured source states. Repeated event onsets share a source-state
measurement only after exact pixel equality has bound each video to that state.
This is a source audit, not a model-processor or display calibration certificate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import av
import numpy as np

from .manifest import atomic_json, canonical, digest, file_sha, state_key, validate_manifest
from .source_audit_t1 import T1Audit
from .stimuli import make_state
from .verify import source_panel


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def stat_identity(path):
    s = Path(path).stat()
    return {key: getattr(s, 'st_' + key) for key in ('dev', 'ino', 'size', 'mtime_ns')}


def safe_child(root, relative):
    root = Path(root).resolve()
    p = root / relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,
            'Source path must be a relative child')
    require(p.is_file() and not p.is_symlink() and root in p.resolve().parents,
            'Missing source file, symlink, or path escaping source root')
    return p


def code_identity():
    names = ['source_audit.py', 'source_audit_t1.py', 'manifest.py', 'stimuli.py',
             'raster.py', 'verify.py']
    return {name: file_sha(Path(__file__).with_name(name)) for name in names}


def fp32_codebook():
    q = np.arange(65536, dtype=np.uint16)
    entry = q.astype(np.float32) * np.float32(255. / 65535.)
    restored = np.rint(entry * np.float32(65535. / 255.)).astype(np.uint16)
    require(np.array_equal(q, restored), 'RGB16 -> FP32 [0,255] codebook is not reversible')
    return {'codes': 65536, 'roundtrip_max_code_error': 0,
            'scope': 'exhaustive scalar input mapping; not an official processor or GPU test'}


def inventory(config):
    items, panels, seen = [], [], set()
    for panel in config['panels']:
        root = Path(panel['root']).resolve()
        manifest_path = root / 'manifest.json'
        require(file_sha(manifest_path) == panel['manifest_sha256'], 'Frozen manifest SHA mismatch')
        manifest = json.loads(manifest_path.read_text())
        t1 = panel['task_family'] == 'T1'
        rows = manifest['samples'] if t1 else validate_manifest(manifest)['rows']
        require(dict(Counter(r['task'] for r in rows)) == panel['expected_counts'],
                'Panel task denominators differ')
        if t1:
            readies = [json.loads(p.read_text()) for p in sorted((root / 'ready').glob('*.json'))]
            T1Audit(root).verify_parameters()
        else:
            readies = [json.loads(line) for line in (root / 'ready.jsonl').read_text().splitlines() if line]
        lookup = {r['sample_id']: r for r in readies}
        require(len(lookup) == len(readies) == len(rows), 'Duplicate or missing ready IDs')
        require(set(lookup) == {r['sample_id'] for r in rows}, 'Ready and manifest IDs differ')
        for row in rows:
            sid = row['sample_id']
            require(sid not in seen, 'Scientific ID repeated across source panels')
            seen.add(sid)
            ready = lookup[sid]
            if t1:
                require(ready['params'] == row['params'] and
                        ready['scientific_spec_sha256'] == row['scientific_spec_sha256'] and
                        ready['correct_answer'] == row['correct_answer'] and
                        ready['manifest_sha256'] == panel['manifest_sha256'] and
                        ready['protocol_sha256'] == file_sha(root / 'protocol.json') and
                        ready['producer_sha256'] == file_sha(root / 'producer.py'),
                        'T1 ready/scientific identity mismatch')
                relative = ready['video_path']
                require(relative == row['video_path'], 'T1 source path changed')
            else:
                require(ready['source_parameters'] == row['source_parameters'], 'Ready parameters changed')
                require(row['truth'] == row['source_parameters'][
                    'target_side' if row['task'] == 'T2' else 'reference_side'], 'Wrong side label')
                relative = ready['video_relative']
                require(relative == 'videos/' + sid + '.mkv', 'Unexpected source video path')
            video = safe_child(root, relative)
            items.append({'sample_id': sid, 'panel': panel['key'], 'root': str(root),
                          'row': row, 'ready': ready, 'path': str(video),
                          'manifest_sha256': panel['manifest_sha256']})
        panels.append({'key': panel['key'], 'manifest_sha256': panel['manifest_sha256'],
                       'counts': panel['expected_counts']})
    require(dict(Counter(x['row']['task'] for x in items)) == config['expected_counts'],
            'Global task denominators differ')
    items.sort(key=lambda x: (x['panel'],
               x['row'].get('params', {}).get('sf_cycles_per_aperture', 0),
               x['row'].get('params', {}).get('modulation_depth', 0), x['sample_id']))
    return items, panels


def prepare(config, items, output, identity):
    qa_path = output / 'source-qa.json'
    if qa_path.exists():
        report = json.loads(qa_path.read_text())
        require(report['identity'] == identity and report['status'] == 'PASS', 'Source QA identity changed')
        for path, metadata in report['source_files'].items():
            require(stat_identity(path) == metadata['stat'], 'Previously checked source bank changed')
        for entry in report['t1']['pixel_integrals']:
            path = Path(report['t1']['rebuilt_basis_root']) / entry['rebuilt_basis_file']
            require(file_sha(path) == entry['rebuilt_basis_sha256'], 'Audit basis artifact changed')
        return report
    started = time.monotonic()
    atomic_json(output / 'progress.json', {'status': 'REBUILDING_SOURCE_STATES', 'started_utc': utc(),
                'identity': identity, 'expected_videos': len(items), 'completed_videos': 0})
    t1root = next(x['root'] for x in items if x['row']['task'] == 'T1')
    t1 = T1Audit(t1root, output).verify_parameters_and_basis()
    locations, files = {}, {}
    for panel in config['panels']:
        if panel['task_family'] == 'T1':
            continue
        root = Path(panel['root']) / 'state_cache'
        if not (root / 'states.json').exists():
            continue
        index = json.loads((root / 'states.json').read_text())
        bankpath = root / 'states.npy'
        bank = np.load(bankpath, mmap_mode='r', allow_pickle=False)
        require(bank.dtype == np.uint16 and bank.shape[1:] == (512, 768, 3), 'Source bank dtype/shape')
        for path in (root / 'states.json', bankpath):
            files[str(path)] = {'stat': stat_identity(path), 'sha256': file_sha(path)}
        for key, entry in index['states'].items():
            require(key not in locations, 'Duplicate state key in independent source banks')
            locations[key] = {'bank': str(bankpath), **entry}
        del bank
    unique = {}
    for item in items:
        if item['row']['task'] != 'T1':
            unique.setdefault(state_key(item['row']), item['row'])
    require(len(unique) == config['expected_unique_T2_T3_states'], 'Unexpected number of unique visual states')
    states, banks, records = {}, {}, []
    for i, (key, row) in enumerate(sorted(unique.items()), 1):
        require(key in locations, 'No source bank for required state')
        location = locations[key]
        if location['bank'] not in banks:
            banks[location['bank']] = np.load(location['bank'], mmap_mode='r', allow_pickle=False)
        bank = banks[location['bank']]
        base, event, qa = make_state(row)
        for array, position, name in zip((base, event), location['indices'], ('baseline_sha256', 'event_sha256')):
            require(np.array_equal(array, bank[position]) and
                    hashlib.sha256(array.tobytes()).hexdigest() == location[name],
                    'Freshly rebuilt state differs from frozen source pixels')
        states[key] = {k: location[k] for k in ('bank', 'indices', 'baseline_sha256', 'event_sha256')}
        states[key].update({'source_parameters': row['source_parameters'], 'measurements': qa,
                            'fresh_rebuild_all_codes_equal': True})
        records.append({'source_parameters': row['source_parameters'], **qa})
        if i % 4 == 0:
            atomic_json(output / 'progress.json', {'status': 'REBUILDING_SOURCE_STATES',
                        'completed_source_states': i, 'expected_source_states': len(unique),
                        'expected_videos': len(items), 'completed_videos': 0, 'updated_utc': utc()})
    report = {'schema': 'classic-source-qa-v1', 'identity': identity, 'status': 'PASS',
              't1': t1, 'states': states, 'state_panel': source_panel(records),
              'FP32_entry_codebook': fp32_codebook(), 'source_files': files,
              'seconds': time.monotonic() - started, 'completed_utc': utc(),
              'scope': 'fresh numeric source measurement, no model/processor/GPU inference'}
    atomic_json(qa_path, report)
    return report


_QA = _T1 = _BANKS = None


def initialize_worker(qa, t1root):
    global _QA, _T1, _BANKS
    _QA, _BANKS = qa, {}
    _T1 = T1Audit(t1root, basis_root=qa['t1']['rebuilt_basis_root'])


def check_frame(actual, expected, index, pts, previous, fps=60, n=180):
    require(index < n, 'Extra encoded frame')
    require(actual.dtype == np.uint16 and actual.shape == (512, 768, 3), 'Frame shape or bit depth changed')
    if expected.ndim == 2:
        require(all(np.array_equal(actual[..., c], expected) for c in range(3)),
                f'Decoded source pixel mismatch at frame {index}')
    else:
        require(np.array_equal(actual, expected), f'Decoded source pixel mismatch at frame {index}')
    require(pts is not None and np.isfinite(pts), 'Missing/non-finite frame timestamp')
    require(previous is None or pts > previous, 'Nonmonotonic frame timestamp')
    error = abs(pts - index / fps)
    require(error <= .001, f'Frame timestamp out of tolerance: {error}')
    return error


def audit_video(item):
    start = time.monotonic()
    result = {'sample_id': item['sample_id'], 'task': item['row']['task'], 'panel': item['panel'],
              'manifest_sha256': item['manifest_sha256'], 'row_sha256': digest(item['row']),
              'ready_sha256': digest(item['ready']), 'status': 'REVIEW_REQUIRED'}
    try:
        path, row, ready = Path(item['path']), item['row'], item['ready']
        before = stat_identity(path)
        sha = file_sha(path)
        require(sha == ready['video_sha256'], 'Encoded file SHA differs from ready receipt')
        if 'video_bytes' in ready:
            require(before['size'] == ready['video_bytes'], 'File length differs from ready receipt')
        t1 = row['task'] == 'T1'
        if t1:
            photometric = _T1.begin_video(row)
            lines = []
        else:
            key = state_key(row)
            state = _QA['states'][key]
            if state['bank'] not in _BANKS:
                _BANKS[state['bank']] = np.load(state['bank'], mmap_mode='r', allow_pickle=False)
            base, event = [_BANKS[state['bank']][i] for i in state['indices']]
            params = row['source_parameters']
        h, count, previous, max_pts, first, last = hashlib.sha256(), 0, None, 0., None, None
        with av.open(str(path)) as container:
            require(len(container.streams.video) == 1, 'Expected exactly one video stream')
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            require(stream.codec_context.name == 'ffv1' and stream.codec_context.format.name == 'gbrp16le',
                    'Unexpected codec or source pixel format')
            require((stream.width, stream.height) == (768, 512) and stream.average_rate == 60,
                    'Encoded dimensions/rate changed')
            for i, frame in enumerate(container.decode(video=0)):
                actual = frame.to_ndarray(format='rgb48le')
                expected = _T1.expected_frame(row, i) if t1 else (
                    event if params['onset_frame'] <= i < params['offset_frame'] else base)
                pts = None if frame.pts is None else float(frame.pts * frame.time_base)
                max_pts = max(max_pts, check_frame(actual, expected, i, pts, previous))
                if first is None:
                    first = pts
                last, previous, count = pts, pts, i + 1
                h.update(actual.tobytes())
                if t1:
                    lines.append(_T1.extract_line(actual))
        require(count == 180, 'Missing encoded frames')
        require('stream_sha256' not in ready or h.hexdigest() == ready['stream_sha256'],
                'Decoded stream SHA differs from ready receipt')
        require(before == stat_identity(path), 'Source video changed during audit')
        result.update({'status': 'PASS', 'video_sha256': sha, 'source_stat': before,
                       'decoded_stream_sha256': h.hexdigest(), 'decoded_frames': count,
                       'every_decoded_RGB_code_compared': True, 'max_code_error': 0,
                       'max_PTS_error_s': max_pts, 'first_last_PTS_s': [first, last],
                       'codec': 'ffv1', 'pixel_format': 'gbrp16le', 'decoded_format': 'rgb48le uint16',
                       'dimensions_wh': [768, 512], 'fps': 60,
                       'parameters': row['params'] if t1 else params,
                       'truth': row['correct_answer'] if t1 else row['truth']})
        if t1:
            result.update({'photometric_QA': photometric,
                           'independent_pixel_measurements': _T1.pixel_readout(row, np.asarray(lines))})
        else:
            result.update({'source_state_key': key, 'independent_pixel_measurements': state['measurements'],
                           'measurement_scope': 'fresh unique-state measurement bound by every decoded RGB code',
                           'event_frame_interval_half_open': [params['onset_frame'], params['offset_frame']],
                           'event_frame_count': params['offset_frame'] - params['onset_frame']})
    except Exception as error:
        result['status'] = 'REVIEW_REQUIRED'
        result['error'] = type(error).__name__ + ': ' + str(error)
    result.update({'audit_seconds': time.monotonic() - start, 'completed_utc': utc(), 'model_calls': 0})
    return result


def pilot_items(items):
    chosen = []
    for key in sorted({x['panel'] + '/' + x['row']['task'] for x in items}):
        group = [x for x in items if x['panel'] + '/' + x['row']['task'] == key]
        chosen.extend((group[0], group[-1]))
    return chosen


def summarize(records, total, config, qa, identity, wall, mode):
    counts = Counter(r['task'] for r in records if r['status'] == 'PASS')
    failures = [r['sample_id'] for r in records if r['status'] != 'PASS']
    complete = len(records) == total
    status = ('REVIEW_REQUIRED' if failures else 'COMPLETE' if complete else
              'PILOT_COMPLETE' if mode == 'pilot' else 'RUNNING')
    t1 = [r['independent_pixel_measurements'] for r in records if r['status'] == 'PASS' and r['task'] == 'T1']
    t2 = [s['measurements'] for s in qa['states'].values() if s['measurements']['task'] == 'T2']
    axes = [a for s in qa['states'].values() if s['measurements']['task'] == 'T3' for a in s['measurements']['axes']]
    numeric = {'T1_max_blind_error': {key: max(r['errors'][key] for r in t1) for key in t1[0]['errors']} if t1 else {},
               'T2_max_core_linear_Y_error': max((r['core_max_error'] for r in t2), default=None),
               'T2_max_alpha_regression_linear_Y_error': max((r['alpha_regression_error'] for r in t2), default=None),
               'T2_changed_outside_target': sum(r['changed_outside_target'] for r in t2),
               'T3_max_PCA_error_deg': max((a['PCA_error_deg'] for a in axes), default=None),
               'T3_max_halfplane_error_deg': max((a['halfplane_error_deg'] for a in axes), default=None)}
    return {'schema': 'classic-source-audit-v1', 'status': status, 'identity': identity,
            'updated_utc': utc(), 'expected_counts': config['expected_counts'], 'completed_counts': dict(counts),
            'completed_videos': len(records), 'expected_videos': total, 'failed_IDs': failures,
            'decoded_frames': sum(r.get('decoded_frames', 0) for r in records),
            'expected_frames': total * 180, 'source_qa_status': qa['status'],
            'max_decoded_code_error': max((r.get('max_code_error', 0) for r in records), default=None),
            'max_PTS_error_s': max((r.get('max_PTS_error_s', 0) for r in records), default=None),
            'mode': mode, 'this_session_wall_seconds': wall,
            'video_worker_seconds': sum(r['audit_seconds'] for r in records),
            'source_qa_seconds': qa['seconds'], 'new_model_calls': 0,
            'numeric_measurement_summary': numeric,
            'limits': {'source_code_error': 0, 'PTS_seconds': .001,
                       'T2_linear_Y_error': .0001, 'T3_axis_degrees': .001},
            'scope': 'source mathematics and all encoded RGB16 frames; not processor/display or arbitrary hardware'}


def write_csv(output, records):
    fields = ['sample_id', 'task', 'panel', 'status', 'truth', 'video_sha256',
              'decoded_stream_sha256', 'decoded_frames', 'max_code_error', 'max_PTS_error_s',
              'source_state_key', 'parameters', 'independent_pixel_measurements', 'error']
    temp = output / 'per-id.csv.partial'
    with temp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for record in sorted(records, key=lambda r: r['sample_id']):
            row = dict(record)
            for key in ('parameters', 'independent_pixel_measurements'):
                row[key] = json.dumps(row.get(key), ensure_ascii=False, sort_keys=True, allow_nan=False)
            writer.writerow(row)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, output / 'per-id.csv')


def run(config_path, mode='full'):
    started = time.monotonic()
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text())
    output = Path(config['output']).resolve()
    for panel in config['panels']:
        require(Path(panel['root']).resolve() not in output.parents and output != Path(panel['root']).resolve(),
                'Audit artifacts must not be written into original source roots')
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / '.audit.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = digest({'config_file_sha256': file_sha(config_path), 'code': code_identity()})
    items, panels = inventory(config)
    qa = prepare(config, items, output, identity)
    journal = output / 'per-id.jsonl'
    records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    done = {r['sample_id']: r for r in records}
    require(len(done) == len(records), 'Repeated IDs in audit journal')
    lookup = {x['sample_id']: x for x in items}
    for sid, record in done.items():
        require(sid in lookup and record['identity'] == identity and
                record['row_sha256'] == digest(lookup[sid]['row']) and
                record['ready_sha256'] == digest(lookup[sid]['ready']), 'Resume identity changed')
        require(record['status'] == 'PASS' and record['source_stat'] == stat_identity(lookup[sid]['path']),
                'Resume requires unchanged, successfully checked source files')
    selected = pilot_items(items) if mode == 'pilot' else items
    pending = [item for item in selected if item['sample_id'] not in done]
    atomic_json(output / 'inventory.json', {'identity': identity, 'panels': panels,
                'counts': config['expected_counts'], 'videos': len(items),
                'source_qa_sha256': file_sha(output / 'source-qa.json'),
                'code': code_identity(), 'new_model_calls': 0,
                'runtime': {'python': sys.version, 'executable': sys.executable,
                            'numpy': np.__version__, 'pyav': av.__version__,
                            'libav': {k: list(v) for k, v in av.library_versions.items()},
                            'platform': platform.platform()}})
    t1root = next(x['root'] for x in items if x['row']['task'] == 'T1')
    workers = config['workers']
    require(type(workers) is int and 1 <= workers <= 8, 'CPU audit worker limit is 1..8')
    def publish():
        summary = summarize(records, len(items), config, qa, identity, time.monotonic()-started, mode)
        atomic_json(output / 'progress.json', summary)
        return summary
    publish()
    with journal.open('ab', buffering=0) as stream, concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(qa, t1root)) as pool:
        iterator, active, exhausted = iter(pending), set(), False
        while active or not exhausted:
            # Admit one bounded video per worker; no unbounded queues or large frame buffers.
            meminfo = Path('/proc/meminfo')
            available = (next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                              if line.startswith('MemAvailable:')) if meminfo.exists() else float('inf'))
            while not exhausted and len(active) < workers and available >= config['minimum_available_GiB'] * 1024**3:
                item = next(iterator, None)
                if item is None:
                    exhausted = True
                    break
                active.add(pool.submit(audit_video, item))
            if not active:
                if exhausted:
                    break
                atomic_json(output / 'progress.json', {**publish(), 'status': 'WAITING_MEMORY_RESERVE'})
                time.sleep(10)
                continue
            finished, active = concurrent.futures.wait(active, timeout=10,
                                return_when=concurrent.futures.FIRST_COMPLETED)
            for future in finished:
                record = future.result()
                record['identity'] = identity
                stream.write(canonical(record) + b'\n')
                os.fsync(stream.fileno())
                records.append(record)
            publish()
    summary = publish()
    write_csv(output, records)
    atomic_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--mode', choices=['pilot', 'full'], default='full')
    args = parser.parse_args()
    result = run(args.config, args.mode)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result['status'] in ('COMPLETE', 'PILOT_COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
