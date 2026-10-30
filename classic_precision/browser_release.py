"""Create browser derivatives; admit only full decoded RGB/timestamp parity.

No stimulus reconstruction or model calls. Source remains authoritative.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def decode(path: Path) -> tuple[str, int]:
    cmd = ['ffmpeg', '-v', 'error', '-threads', '1', '-i', str(path),
           '-an', '-pix_fmt', 'rgb24', '-f', 'rawvideo', '-threads', '1', '-']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    h = hashlib.sha256()
    n = 0
    assert p.stdout is not None
    for chunk in iter(lambda: p.stdout.read(1 << 20), b''):
        h.update(chunk)
        n += len(chunk)
    assert p.stderr is not None
    err = p.stderr.read().decode(errors='replace')
    if p.wait() != 0:
        raise RuntimeError(f'Decode failed: {path}: {err[-1500:]}')
    return h.hexdigest(), n


def frames(path: Path) -> dict:
    raw = subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=codec_name,pix_fmt,width,height,avg_frame_rate:frame=best_effort_timestamp_time',
           '-show_frames', '-of', 'json', str(path)])
    data = json.loads(raw)
    times = [float(f['best_effort_timestamp_time']) for f in data['frames']]
    assert len(times) == 180
    assert max(abs(t - i / 60) for i, t in enumerate(times)) <= .000501
    st = data['streams'][0]
    assert [st['width'], st['height']] == [768, 512]
    return {'stream': st, 'frames': len(times),
            'max_timestamp_error_s': max(abs(t - i / 60) for i, t in enumerate(times))}


def convert(item: dict, root: Path, out: Path, protocol: str) -> dict:
    sid = item['sample_id']
    src = Path(next(c['path'] for c in item['content'] if c['type'] == 'video'))
    if not src.exists():
        src = root / 'datasets' / protocol / 'videos' / src.name
    expected = item['ready']['video_sha256']
    assert sha(src) == expected, f'Source identity changed: {sid}'
    key = hashlib.sha256(f'{protocol}:{sid}'.encode()).hexdigest()
    dest = out / f'{key}.webm'
    qa = out / f'{key}.json'
    if qa.exists() and dest.exists():
        previous = json.loads(qa.read_text())
        if previous['source_sha256'] == expected and sha(dest) == previous['sha256']:
            return previous
        raise RuntimeError(f'Existing derivative identity mismatch: {sid}')
    cmd = ['ffmpeg', '-v', 'error', '-nostdin', '-threads', '1', '-i', str(src),
           '-an', '-c:v', 'libvpx-vp9', '-lossless', '1', '-pix_fmt', 'gbrp',
           '-colorspace', 'rgb', '-color_range', 'pc', '-threads', '1',
           '-deadline', 'realtime', '-cpu-used', '8', '-n', str(dest)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f'Encode failed: {sid}: {result.stderr[-2000:]}')
    source_rgb, source_n = decode(src)
    target_rgb, target_n = decode(dest)
    assert source_n == target_n == 180 * 768 * 512 * 3
    assert source_rgb == target_rgb, f'Full RGB parity failed: {sid}'
    source_timing, target_timing = frames(src), frames(dest)
    assert target_timing['stream']['codec_name'] == 'vp9'
    assert target_timing['stream']['pix_fmt'] == 'gbrp'
    params = item['row']['source_parameters']
    receipt = {'id': sid, 'object': f'browser/{protocol}/{key}.webm',
               'sha256': sha(dest), 'bytes': dest.stat().st_size,
               'width': 768, 'height': 512, 'duration': 3,
               'source_sha256': expected,
               'stratum': {'task': item['task'], 'protocol': protocol,
                           'domain': params['domain'], 'coherence': params.get('coherence')},
               'qa': {'status': 'FULL_180_RGB_BYTES_AND_TIMESTAMPS_PASS',
                      'source_RGB_sha256': source_rgb, 'derivative_RGB_sha256': target_rgb,
                      'decoded_bytes': source_n, 'source': source_timing,
                      'derivative': target_timing, 'encoder_command': cmd}}
    qa.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--protocol', default='t45-final-20261030')
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    source = args.root / 'plans' / args.protocol / 'fourier-sources.json'
    candidates = list(json.loads(source.read_text())['items'].values())
    candidates = [x for x in candidates if x['task'] == 'T4'
                  and x['row']['source_parameters']['domain'] == 'classic'
                  and x['row']['source_parameters']['diameter_px'] == 56
                  and x['row']['source_parameters']['coherence'] > 0]
    candidates.sort(key=lambda x: (x['row']['source_parameters']['seed'],
                                  x['row']['source_parameters']['coherence'],
                                  x['row']['source_parameters']['direction']))
    selected, seen = [], set()
    for item in candidates:
        identity = item['ready']['video_sha256']
        if identity not in seen:
            selected.append(item)
            seen.add(identity)
        if len(selected) == args.limit:
            break
    assert len(selected) == args.limit, (len(selected), args.limit)
    out = args.root / 'runs/stimulus-preparation' / args.protocol / 'browser'
    out.mkdir(parents=True, exist_ok=True)
    def work(x):
        r = convert(x, args.root, out, args.protocol)
        print(json.dumps({'status': 'PASS', 'id': r['id'], 'bytes': r['bytes']}), flush=True)
        return r
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        receipts = list(pool.map(work, selected))
    manifest = {'protocol': args.protocol, 'status': 'FULL_DECODED_RGB_PARITY_PASS',
                'source_index_sha256': sha(source), 'count': len(receipts),
                'items': [{k: v for k, v in r.items() if k != 'qa'} for r in receipts]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
