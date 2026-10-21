"""Finite RGB16 construction with reusable states and resumable atomic videos."""
from __future__ import annotations

import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import shutil
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av
import numpy as np

from . import raster, verify
from .manifest import atomic_json, canonical, digest, file_sha, state_key


def rgb16(y):
    if not np.isfinite(y).all() or np.min(y) < 0 or np.max(y) > 1:
        raise ValueError('Invalid/clipped source luminance')
    q = np.rint(raster.srgb(y) * 65535).astype(np.uint16)
    return np.repeat(q[..., None], 3, axis=-1)


@lru_cache(maxsize=4)
def disk(area):
    return raster.disk_coverage(area=area)


def make_state(row):
    p = row['source_parameters']
    shape = (p['height'], p['width'])
    if shape != (512, 768):
        raise ValueError('This protocol fixes source dimensions at768x512')
    if row['task'] == 'T2':
        if p['centers_xy'] != [[224, 302], [544, 302]]:
            raise ValueError('Unexpected disk center')
        left, circle_qa = disk(p['disk_alpha_area_px2'])
        if abs(circle_qa['diameter_px'] - p['disk_diameter_px']) > 1e-9:
            raise ValueError('Declared disk area/diameter disagree')
        right = np.roll(left, 320, axis=1)
        base_y = p['background_Y'] + (left + right) * (p['initial_dot_Y'] - p['background_Y'])
        target = left if p['target_side'] == 'LEFT' else right
        baseline, event = rgb16(base_y), rgb16(base_y + target * p['delta_Y'])
        qa = verify.source_state(row, baseline, event, target)
        qa['independent_circle_validation'] = circle_qa
        return baseline, event, qa
    angles = [0. if side == p['reference_side'] else p['angle_deg'] * p['tilt_sign']
              for side in ('LEFT', 'RIGHT')]
    alpha = np.zeros(shape, np.float64)
    independent = np.zeros(shape, np.float64)
    for center, angle in zip(p['centers_xy'], angles):
        kw = {'shape': shape, 'center': tuple(center),
              'width': p['bar_wh'][0], 'height': p['bar_wh'][1]}
        alpha += raster.strip_coverage(angle, **kw)
        independent += raster.geos_coverage(angle, **kw)
    diff = float(np.max(abs(alpha - independent)))
    if diff >= 1e-9 or np.max(alpha) > 1 + 1e-12:
        raise ValueError(('Independent coverage or overlap failure', diff))
    baseline = rgb16(np.full(shape, p['background_Y']))
    event = rgb16(p['background_Y'] + (p['foreground_Y'] - p['background_Y']) * alpha)
    qa = verify.source_state(row, baseline, event, alpha)
    qa['independent_GEOS_max_coverage_difference'] = diff
    return baseline, event, qa


def prepare_states(manifest, work):
    """Serialize cache publication shared by standalone generation/validation."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    with (work/'.state-cache.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _prepare_states_locked(manifest, work)


def _prepare_states_locked(manifest, work):
    """Cache two source states per unique condition, not per onset/duration."""
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    unique = {}
    for row in manifest['rows']:
        unique.setdefault(state_key(row), row)
    index = {'protocol_id': manifest['protocol_id'], 'states': {}}
    bank_path, index_path = work/'states.npy', work/'states.json'
    if index_path.exists():
        old = json.loads(index_path.read_text())
        if old['protocol_id'] != manifest['protocol_id'] or not bank_path.is_file():
            raise ValueError('State cache identity mismatch')
        if set(old['states']) != set(unique):
            raise ValueError('State cache keys differ from immutable manifest')
        bank = np.load(bank_path, mmap_mode='r')
        if bank.dtype != np.uint16 or bank.shape != (len(unique)*2, 512, 768, 3):
            raise ValueError('State cache shape/precision mismatch')
        used = []
        for item in old['states'].values():
            used.extend(item['indices'])
            for pos, field in zip(item['indices'], ['baseline_sha256', 'event_sha256']):
                if hashlib.sha256(bank[pos].tobytes()).hexdigest() != item[field]:
                    raise ValueError('State cache changed after pixel QA')
        if sorted(used) != list(range(len(unique)*2)):
            raise ValueError('State cache indices overlap or are incomplete')
        del bank
        return bank_path, old
    temp = work/'states.npy.partial'
    bank = np.lib.format.open_memmap(temp, mode='w+', dtype=np.uint16,
                                    shape=(len(unique) * 2, 512, 768, 3))
    qa = []
    for i, (key, row) in enumerate(unique.items()):
        base, event, record = make_state(row)
        bank[2*i], bank[2*i+1] = base, event
        index['states'][key] = {'indices': [2*i, 2*i+1],
            'baseline_sha256': hashlib.sha256(base.tobytes()).hexdigest(),
            'event_sha256': hashlib.sha256(event.tobytes()).hexdigest()}
        qa.append({'state_key': key, 'source_parameters': row['source_parameters'], **record})
    index['validation'] = verify.source_panel(qa)
    index['quality_records'] = qa
    bank.flush()
    del bank
    os.replace(temp, bank_path)
    atomic_json(index_path, index)
    return bank_path, index


def validate_video(row, path, base, event):
    """Existing frame-by-frame RGB16/source equality check, without model arrays."""
    p = row['source_parameters']
    on, off, n = p['onset_frame'], p['offset_frame'], p['frames']
    actual_hash = hashlib.sha256()
    count, pts_error = 0, 0.
    with av.open(str(path)) as container:
        if container.streams.video[0].codec_context.format.name != 'gbrp16le':
            raise ValueError('Encoder dropped channel precision')
        for count, frame in enumerate(container.decode(video=0), 1):
            i = count-1
            actual = frame.to_ndarray(format='rgb48le')
            expected = event if on <= i < off else base
            if actual.dtype != np.uint16 or not np.array_equal(actual, expected):
                raise ValueError(('Lossless frame/sequence mismatch', row['sample_id'], i))
            actual_hash.update(actual.tobytes())
            pts_error = max(pts_error, abs(float(frame.pts*frame.time_base)-i/p['fps']))
    if count != n or pts_error > .001:
        raise ValueError(('Encoded timing/stream mismatch', row['sample_id']))
    return {'stream_sha256': actual_hash.hexdigest(), 'frames': count,
            'max_PTS_error_s': pts_error, 'max_decoded_code_error': 0}


def _encode_one(task):
    row, state, bank_path, run_dir, minimum_free_bytes = task
    run_dir = Path(run_dir)
    if shutil.disk_usage(run_dir).free < minimum_free_bytes:
        raise RuntimeError('Disk reserve reached; completed ready records are preserved')
    bank = np.load(bank_path, mmap_mode='r')
    base, event = [bank[i] for i in state['indices']]
    p = row['source_parameters']
    video = run_dir/'videos'/(row['sample_id'] + '.mkv')
    partial = video.with_name(video.name + '.partial')
    video.parent.mkdir(exist_ok=True)
    on, off, n = p['onset_frame'], p['offset_frame'], p['frames']
    expected_hash = hashlib.sha256()
    with av.open(str(partial), 'w', format='matroska') as container:
        stream = container.add_stream('ffv1', rate=p['fps'])
        stream.width, stream.height = p['width'], p['height']
        stream.pix_fmt = 'gbrp16le'
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = 0
        stream.options = {'level': '3', 'coder': '1', 'context': '1', 'slicecrc': '1'}
        for i in range(n):
            rgb = event if on <= i < off else base
            expected_hash.update(rgb.tobytes())
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb48le')
            frame.pts, frame.time_base = i, Fraction(1, p['fps'])
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    qa = validate_video(row, partial, base, event)
    if qa['stream_sha256'] != expected_hash.hexdigest():
        raise ValueError(('Encoded timing/stream mismatch', row['sample_id']))
    with partial.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(partial, video)
    ready = {'sample_id': row['sample_id'], 'video_relative': str(video.relative_to(run_dir)),
             'video_sha256': file_sha(video), 'stream_sha256': qa['stream_sha256'],
             'source_parameters': p, 'frames': qa['frames'], 'master_bits_per_channel': 16,
             'max_decoded_code_error': 0, 'max_PTS_error_s': qa['max_PTS_error_s']}
    # A small sidecar closes the encoded-video -> ready append recovery window.
    atomic_json(video.with_suffix('.ready.json'), ready)
    return ready


def generate(manifest, run_dir, workers=None, disk_floor_gib=10, shard_count=3,
             cpu_memory_gib=32, ram_reserve_gib=4):
    from .ready import (ReadyCatalog, append_ready, finish_generation, generation_order,
                        publication_lock)
    from .cpu import allocation, available_cores, ready_buffer
    import time
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock = (run_dir/'.generation.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        ready_path = run_dir/'ready.jsonl'
        catalog = ReadyCatalog(manifest, ready_path, run_dir, file_sha(run_dir/'manifest.json'),
                               allow_failed=True)
        done = catalog.refresh()
        if catalog.complete:
            return json.loads((run_dir/'generation.json').read_text())
        with publication_lock(ready_path, exclusive=True):
            atomic_json(run_dir/'generation.json', {'status': 'RUNNING',
                'protocol_id': manifest['protocol_id'],
                'manifest_sha256': file_sha(run_dir/'manifest.json'), 'new_VLM_calls': 0})
        bank_path, index = prepare_states(manifest, run_dir/'state_cache')
        tasks = []
        for row in generation_order(manifest, shard_count):
            sid = row['sample_id']
            if sid in done:
                continue
            sidecar = run_dir/'videos'/(sid + '.ready.json')
            if sidecar.exists():
                # Includes the video-SHA check before recovering a publication.
                record = json.loads(sidecar.read_text())
                append_ready(manifest, run_dir, record)
            else:
                tasks.append((row, index['states'][state_key(row)], str(bank_path), str(run_dir),
                              int(disk_floor_gib * 1024**3)))
        # Admission controls in-flight tasks, not the lifetime of idle pool
        # processes. No GPU query or unrelated process inspection is involved.
        ceiling = min(workers or max(1, available_cores()-2), available_cores())
        with concurrent.futures.ProcessPoolExecutor(max_workers=ceiling) as pool:
            next_index = 0
            active = set()
            last_budget, target, last_buffer, backlog = 0., 0, 0., 0
            while active or next_index < len(tasks):
                if time.monotonic()-last_budget >= 2 or not active:
                    activity_path = run_dir/'cpu-activity.json'
                    activity = json.loads(activity_path.read_text()) if activity_path.exists() else {}
                    if time.monotonic()-last_buffer >= 10:
                        backlog = ready_buffer(manifest, run_dir, shard_count)
                        last_buffer = time.monotonic()
                    plan = allocation(workers, cpu_memory_gib, ram_reserve_gib,
                        calibration_active=activity.get('calibration_active', False),
                        buffer_rows=backlog,
                        disk_free_gib=shutil.disk_usage(run_dir).free/2**30,
                        disk_floor_gib=disk_floor_gib)
                    atomic_json(run_dir/'cpu-budget.json', plan)
                    if plan['reason'] == 'DISK_RESERVE_REACHED':
                        raise RuntimeError('Disk reserve reached; preserve qualified inputs')
                    target, last_budget = plan['workers'], time.monotonic()
                while next_index < len(tasks) and len(active) < target:
                    active.add(pool.submit(_encode_one, tasks[next_index]))
                    next_index += 1
                if not active:
                    time.sleep(10)
                    continue
                finished, _ = concurrent.futures.wait(active, timeout=2,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    active.remove(future)
                    append_ready(manifest, run_dir, future.result(), verified=True)
        catalog.refresh()
        return finish_generation(manifest, run_dir)
    except BaseException as error:
        # Current status is explicit. Preserve the prior failure in a small
        # durable record when a resumed generator later changes the status.
        receipt = {'status': 'FAILED', 'protocol_id': manifest['protocol_id'],
                   'manifest_sha256': file_sha(run_dir/'manifest.json'),
                   'error_type': type(error).__name__, 'error': str(error), 'new_VLM_calls': 0}
        with publication_lock(run_dir/'ready.jsonl', exclusive=True):
            atomic_json(run_dir/'generation.failure.json', receipt)
            atomic_json(run_dir/'generation.json', receipt)
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _validate_one(task):
    row, item, state, bank_path, root = task
    bank = np.load(bank_path, mmap_mode='r')
    base, event = [bank[i] for i in state['indices']]
    receipt = validate_video(row, Path(root)/item['video_relative'], base, event)
    if receipt['stream_sha256'] != item['stream_sha256']:
        raise ValueError('Independent decode stream differs from published ready')
    return row['sample_id']


def validate_inputs(run_dir, workers=None, require_complete=False, cpu_memory_gib=32, ram_reserve_gib=4):
    from .manifest import validate_manifest
    from .ready import ReadyCatalog
    from .cpu import allocation, available_cores
    import time
    run_dir = Path(run_dir)
    manifest = validate_manifest(json.loads((run_dir/'manifest.json').read_text()))
    catalog = ReadyCatalog(manifest, run_dir/'ready.jsonl', run_dir, file_sha(run_dir/'manifest.json'))
    ready = catalog.refresh()
    if require_complete and not catalog.complete:
        raise ValueError('Independent validation requires the complete frozen input panel')
    checked, plan = [], allocation(workers, cpu_memory_gib, ram_reserve_gib)
    if ready:
        bank, index = prepare_states(manifest, run_dir/'state_cache')
        tasks = [(row, ready[row['sample_id']], index['states'][state_key(row)], str(bank), str(run_dir))
                 for row in manifest['rows'] if row['sample_id'] in ready]
        ceiling = min(workers or max(1, available_cores()-2), available_cores())
        with concurrent.futures.ProcessPoolExecutor(max_workers=ceiling) as pool:
            active, next_index = set(), 0
            while active or next_index < len(tasks):
                plan = allocation(workers, cpu_memory_gib, ram_reserve_gib)
                while next_index < len(tasks) and len(active) < plan['workers']:
                    active.add(pool.submit(_validate_one, tasks[next_index]))
                    next_index += 1
                if not active:
                    atomic_json(run_dir/'validation-wait.json', {'status': 'WAITING_FOR_RAM', **plan})
                    time.sleep(10)
                    continue
                finished, _ = concurrent.futures.wait(active, timeout=2,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    active.remove(future)
                    checked.append(future.result())
    result = {'status': 'COMPLETE' if catalog.complete else 'INCOMPLETE',
              'validation': 'PASS', 'checked': len(checked), 'expected': len(manifest['rows']),
              'protocol_id': manifest['protocol_id'], 'row_ids': checked,
              'source_code_max_error': 0, 'new_VLM_calls': 0, 'cpu_budget': plan}
    atomic_json(run_dir/'validation.json', result)
    return result
