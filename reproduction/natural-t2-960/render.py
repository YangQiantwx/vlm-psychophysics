#!/usr/bin/env python3
"""Render the frozen Natural T2 panel directly from source images."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import av
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import PIL
import scipy

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE/'source'), str(HERE/'source/src/stimuli')]
import natural_t2_stimuli as t2
from natural_signal_reference import background
from prepare_next_tasks import sample_affine


def source_file(spec, repository, asset_cache):
    if 'bundle_path' in spec:
        path = HERE/spec['bundle_path']
    elif asset_cache and (asset_cache/(spec['id']+'.png')).is_file():
        path = asset_cache/(spec['id']+'.png')
    else:
        path = hf_hub_download(repo_type='dataset', revision=repository['revision'],
                               repo_id=repository['repo_id'], filename=spec['repository_path'], token=False)
    if t2.sha(path) != spec['sha256']:
        raise ValueError(f"Source SHA mismatch for {spec['id']}")
    return Path(path)


def prepare(args):
    work, spec, path = args
    dest = work/(spec['id']+'.npy')
    if spec['kind'] == 'background':
        array, _ = background(path)
    elif 'source_to_output_matrix' in spec:
        with Image.open(path) as image:
            rgba = np.asarray(image.convert('RGBA'), np.float64) / 255
        alpha = rgba[..., 3:]
        premultiplied = np.concatenate((t2.linear(rgba[..., :3])*alpha, alpha), axis=-1)
        array = sample_affine(premultiplied, np.asarray(spec['source_to_output_matrix']), size=384, samples=8)
    else:
        # The replacement spoon uses its own source geometry.
        raw = work/'assets/library'
        raw.mkdir(parents=True, exist_ok=True)
        filename = spec['id']+'.png'
        (raw/filename).symlink_to(path)
        t2.asset_job((work, {**spec, 'path': filename}, 'object'))
        with np.load(work/'datasets'/t2.NAME/'prepared-vertical-v2'/(spec['id']+'.npz')) as packed:
            array = np.concatenate((packed['pm'], packed['alpha'][..., None]), axis=-1)
    np.save(dest, array)


def render(args):
    work, output, row = args
    onset, offset, delta = (row[key] for key in ('onset_frame', 'offset_frame', 'delta_Y'))
    bg = np.load(work/(row['background_id']+'.npy'))
    obj = np.load(work/(row['object_id']+'.npy'))
    pm, alpha = obj[..., :3], obj[..., 3]
    area = alpha.sum()
    gain = delta/(1-(pm@t2.Y).sum()/area)
    assert 0 < gain < 1 and offset-onset == 30
    for side in ('LEFT','RIGHT'):
        t2.stamp(bg, pm, alpha, side)
    base = t2.rgb8(bg)
    records = []
    for case in row['cases']:
        altered = bg.copy()
        cx, cy = t2.CENTERS[['LEFT','RIGHT'].index(case['truth'])]
        y, x = slice(cy-192,cy+192), slice(cx-192,cx+192)
        altered[y,x] += gain*(alpha[..., None]-pm)
        assert abs(((altered-bg)@t2.Y).sum()/area-delta) < 1e-12
        event = t2.rgb8(altered)
        observed = ((t2.linear(event/255)-t2.linear(base/255))@t2.Y).sum()/area
        assert abs(observed-delta) < min(.002,.02*delta)
        roi = np.zeros(base.shape[:2], dtype=bool)
        roi[y,x] = alpha>0
        assert np.any(base!=event) and np.array_equal(base[~roi],event[~roi])
        path = output/Path(case['path']).name
        temp = path.with_suffix('.partial.mp4')
        t2.encode(temp, base, event, onset, offset)
        digest = hashlib.sha256()
        count = 0
        with av.open(str(temp)) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            assert stream.codec_context.name == 'h264'
            assert (stream.width, stream.height, stream.average_rate) == (768, 512, 60)
            for i, frame in enumerate(container.decode(stream)):
                rgb = frame.to_ndarray(format='rgb24')
                assert frame.pts*frame.time_base == Fraction(i,60)
                assert np.array_equal(rgb, event if onset<=i<offset else base)
                digest.update(rgb.tobytes())
                count += 1
        stream_sha = digest.hexdigest()
        if count != 180 or stream_sha != case['decoded_stream_sha256']:
            raise ValueError(f"{case['sample_id']} decoded stream mismatch ({count} frames, {stream_sha})")
        video_sha = t2.sha(temp)
        temp.replace(path)
        records.append({'sample_id':case['sample_id'],'video_sha256':video_sha,'decoded_stream_sha256':stream_sha,
                        'published_container_equal':video_sha==case['video_sha256'],'frames':count,
                        'decoded_delta_Y':float(observed)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--asset-cache', type=Path, help='Optional offline PNGs named by frozen asset ID')
    parser.add_argument('--representative', action='store_true', help='One pair per source variant')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    manifest = json.loads((HERE/'combinations.json').read_bytes())
    for rel, expected in manifest['source_files'].items():
        assert t2.sha(HERE/rel)==expected, rel
    rows = manifest['combinations']
    ids = [case['sample_id'] for row in rows for case in row['cases']]
    assert len(ids)==len(set(ids))==manifest['stimuli']
    assert hashlib.sha256(('\n'.join(ids)+'\n').encode()).hexdigest()==manifest['ordered_ids_sha256']
    if args.representative:
        rows = list({row['source_variant']:row for row in reversed(rows)}.values())
    output = args.output.resolve()
    output.mkdir(parents=True)
    video = output/'videos'
    video.mkdir()
    from tempfile import TemporaryDirectory
    assets = [manifest['assets'][k] for k in sorted({row[field] for row in rows for field in ('background_id','object_id')})]
    with TemporaryDirectory(prefix='.prepared-', dir=output) as dirname:
        work = Path(dirname)
        (work/'src').symlink_to(HERE/'source/src', target_is_directory=True)
        with ThreadPoolExecutor(max_workers=min(12,args.workers*2)) as pool:
            sources = list(pool.map(lambda spec: source_file(spec,manifest['asset_repository'],args.asset_cache),assets))
        print(f'{len(sources)} SHA-verified assets', flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(prepare, ((work,spec,path) for spec,path in zip(assets,sources,strict=True))))
            records = []
            for pair in pool.map(render, ((work,video,row) for row in rows)):
                records.extend(pair)
                if len(records)%48==0:
                    print(f'{len(records)}/{2*len(rows)} videos passed', flush=True)
    t2.write(output/'acceptance.json', {
        'status':'PASS', 'mode':'representative' if args.representative else 'full',
        'combination_sha256':t2.sha(HERE/'combinations.json'),
        'runtime':{'Python':sys.version.split()[0],'PyAV':av.__version__,'NumPy':np.__version__,
                   'SciPy':scipy.__version__,'Pillow':PIL.__version__,'FFmpeg':av.library_versions},
        'stimuli':len(records),'decoded_frames_and_timestamps':180*len(records),
        'published_container_matches':sum(row['published_container_equal'] for row in records),
        'cases':records})
    print(f'{len(records)} videos passed, {sum(row["published_container_equal"] for row in records)} published container SHAs matched')


if __name__ == '__main__':
    main()
