"""Resumable CPU browser encoding with OpenCV RGB8 admission and FFmpeg readback.

RGB16 masters are preserved. Their derivatives represent the official OpenCV
uint8 decode, not a lossless preservation of sixteen-bit values.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from fractions import Fraction
import hashlib
import json
import os
import shutil
from pathlib import Path
import resource
import subprocess
import time
import zlib


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def convert(job):
    root, row = job
    root = Path(root)
    dataset = row['dataset']
    src = root / row['path'].split('cvpr2027_benchmark/', 1)[-1]
    assert src.is_file() and file_sha(src) == row['sha256']
    sid = row.get('id', src.stem)
    key = hashlib.sha256(f'{dataset}:{sid}'.encode()).hexdigest()
    out = root / 'runs/browser-library-20261030' / dataset
    out.mkdir(parents=True, exist_ok=True)
    video, qa = out / f'{key}.webm', out / f'{key}.json'
    if qa.exists() and video.exists():
        old = json.loads(qa.read_text())
        assert old['source_sha256'] == row['sha256'] and file_sha(video) == old['sha256']
        return {**old, 'reused': True}
    # The first 25 already-qualified T4 derivatives are immutable and reusable.
    prior = root / 'runs/stimulus-preparation/t45-final-20261030/browser'
    if dataset == 't45-final-20261030' and (prior / f'{key}.json').exists():
        old = json.loads((prior / f'{key}.json').read_text())
        assert old['source_sha256'] == row['sha256'] and file_sha(prior / f'{key}.webm') == old['sha256']
        shutil.copyfile(prior / f'{key}.webm', video)
        old['reused_previous_qualification'] = True
        qa.write_text(json.dumps(old, indent=2) + '\n')
        return old
    import av
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    begin = time.monotonic()
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    part = video.with_suffix(f'.partial-{os.getpid()}.webm')
    cap = cv2.VideoCapture(str(src), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    assert cap.isOpened()
    original_crc, stream = [], hashlib.sha256()
    with av.open(str(part), 'w', format='webm') as container:
        encoder = container.add_stream('libvpx-vp9', rate=60)
        encoder.width, encoder.height, encoder.pix_fmt = 768, 512, 'gbrp'
        encoder.options = {'lossless': '1', 'deadline': 'realtime', 'cpu-used': '8', 'threads': '1'}
        encoder.codec_context.thread_count = 1
        encoder.codec_context.colorspace = 0
        for index in range(180):
            ok, bgr = cap.read()
            assert ok and bgr.dtype == np.uint8 and bgr.shape == (512, 768, 3)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raw = rgb.tobytes()
            original_crc.append(zlib.crc32(raw))
            stream.update(raw)
            frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            frame.pts, frame.time_base = index, Fraction(1, 60)
            for packet in encoder.encode(frame):
                container.mux(packet)
        assert not cap.read()[0]
        for packet in encoder.encode():
            container.mux(packet)
    cap.release()
    if row.get('source_RGB8_sha256'):
        assert stream.hexdigest() == row['source_RGB8_sha256'], f'Official decode/source RGB8 mismatch: {sid}'
    decoder = subprocess.Popen(['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(part),
        '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-threads', '1', '-'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    returned_crc, decoded = [], hashlib.sha256()
    frame_bytes = 768 * 512 * 3
    for index in range(180):
        raw = decoder.stdout.read(frame_bytes)
        assert len(raw) == frame_bytes
        returned_crc.append(zlib.crc32(raw))
        decoded.update(raw)
    assert not decoder.stdout.read(1)
    error = decoder.stderr.read()
    assert decoder.wait() == 0, f'FFmpeg readback error: {sid}: {error[-500:]!r}'
    assert returned_crc == original_crc and decoded.hexdigest() == stream.hexdigest()
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,pix_fmt,width,height:frame=best_effort_timestamp_time',
        '-show_frames', '-of', 'json', str(part)]))
    pts = [float(f['best_effort_timestamp_time']) for f in probe['frames']]
    assert len(pts) == 180 and max(abs(t-i/60) for i,t in enumerate(pts)) <= .000501
    assert probe['streams'][0]['codec_name'] == 'vp9' and probe['streams'][0]['pix_fmt'] == 'gbrp'
    assert file_sha(src) == row['sha256']
    part.replace(video)
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    receipt = {'id': sid, 'dataset': dataset, 'object': f'browser/{dataset}/{key}.webm',
        'sha256': file_sha(video), 'source_sha256': row['sha256'], 'bytes': video.stat().st_size,
        'width': 768, 'height': 512, 'duration': 3, 'stratum': row.get('stratum', {}),
        'qa': {'status': 'ALL_180_RGB8_CRC_SHA_AND_TIMESTAMPS_PASS', 'source_decoder': 'OpenCV CAP_FFMPEG uint8 BGR to RGB',
               'OpenCV_version': cv2.__version__, 'independent_readback': 'FFmpeg RGB24',
               'source_RGB8_sha256': stream.hexdigest(), 'derivative_RGB8_sha256': decoded.hexdigest(),
               'frame_crc32': original_crc, 'max_PTS_error_s': max(abs(t-i/60) for i,t in enumerate(pts)),
               'wall_seconds': time.monotonic()-begin,
               'CPU_seconds': usage1.ru_utime+usage1.ru_stime-usage0.ru_utime-usage0.ru_stime,
               'master_bit_depth_preserved': dataset != 'classic-t1-t3-v0.4'}}
    qa.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--control', type=Path, help='Optional JSON {workers:N} for bounded live CPU concurrency.')
    args = ap.parse_args()
    rows = json.loads(args.inventory.read_text())['items']
    rows = [r for i,r in enumerate(rows) if i % args.shards == args.shard]
    if args.limit:
        rows = rows[:args.limit]
    out = args.root / 'runs/browser-library-20261030'
    out.mkdir(parents=True, exist_ok=True)
    result = out / f'shard-{args.shard}-of-{args.shards}.jsonl'
    count, failures = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, result.open('a') as log:
        pending, next_index = {}, 0
        while pending or next_index < len(rows):
            active = args.workers
            if args.control and args.control.exists():
                active = max(1, min(args.workers, int(json.loads(args.control.read_text())['workers'])))
            while len(pending) < active and next_index < len(rows):
                row = rows[next_index]
                pending[pool.submit(convert, (str(args.root), row))] = row
                next_index += 1
            done, _ = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            if not done:
                continue
            for future in done:
                row = pending.pop(future)
                try:
                    receipt = future.result()
                    log.write(json.dumps(receipt) + '\n')
                    log.flush()
                    count += 1
                except Exception as exc:
                    failures.append({'object': row['object'], 'error_type': type(exc).__name__, 'error': str(exc)[:500]})
            if (count + len(failures)) % 50 == 0:
                print(json.dumps({'pass': count, 'failures': len(failures), 'total': len(rows)}), flush=True)
            progress = {'expected': len(rows), 'pass': count, 'failures': failures, 'active_workers': active,
                        'status': 'COMPLETE' if count == len(rows) else 'RUNNING_OR_PARTIAL'}
            temp = out / f'shard-{args.shard}-of-{args.shards}-progress.tmp'
            temp.write_text(json.dumps(progress, indent=2) + '\n')
            temp.replace(temp.with_suffix('.json'))
    print(json.dumps({'pass': count, 'failed': len(failures), 'expected': len(rows)}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
