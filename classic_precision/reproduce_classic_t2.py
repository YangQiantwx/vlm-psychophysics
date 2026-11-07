"""Rebuild Classic T2 videos from the published per-video metadata."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

from . import manifest, stimuli


def load_trials(metadata, metadata_sha256=None, *, sample_ids=None, background_y=None):
    raw = Path(metadata).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if metadata_sha256 is not None and digest != metadata_sha256:
        raise ValueError('Classic T2 metadata SHA-256 differs')
    trials = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = {trial['sample_id'] for trial in trials}
    if len(ids) != len(trials):
        raise ValueError('Repeated trial ID in Classic T2 metadata')
    if sample_ids:
        unknown = set(sample_ids) - ids
        if unknown:
            raise ValueError(f'Unknown Classic T2 trial IDs: {", ".join(sorted(unknown))}')
    selected = []
    names = set()
    for trial in trials:
        sid = trial['sample_id']
        if sample_ids and sid not in sample_ids:
            continue
        p = trial['parameters']
        if background_y is not None and p['background_Y'] != background_y:
            continue
        if (trial['domain'] != 'classic' or trial['task'] != 'T2' or p['task'] != 'T2'
                or sid != 't2_' + manifest.digest(p)[:20]
                or trial['truth'] not in ('LEFT', 'RIGHT') or p['target_side'] != trial['truth']
                or not trial['prompt'].strip()
                or any(trial[k] != p[k] for k in ('width', 'height', 'frames', 'fps'))
                or (p['width'], p['height'], p['frames'], p['fps'], p['bits']) != (768, 512, 180, 60, 16)
                or not 0 <= p['onset_frame'] < p['offset_frame'] <= p['frames']
                or any(not re.fullmatch('[0-9a-f]{64}', trial[k])
                       for k in ('video_sha256', 'decoded_stream_sha256'))):
            raise ValueError(f'Invalid Classic T2 source record: {sid}')
        path = PurePosixPath(trial['path'])
        name = path.name
        if (not path.is_relative_to('classic/t2') or '..' in path.parts
                or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*\.mkv', name)
                or name.casefold() in names):
            raise ValueError(f'Invalid or repeated video name for {sid}: {path}')
        names.add(name.casefold())
        row = dict(sample_id=sid, task='T2', truth=trial['truth'], prompt=trial['prompt'], source_parameters=p)
        selected.append((row, name, trial['video_sha256'], trial['decoded_stream_sha256']))
    if not selected:
        raise ValueError('No Classic T2 trials selected')
    return digest, selected


def reproduce(metadata, output, *, sample_ids=None, background_y=None, metadata_sha256=None, workers=1):
    if workers < 1:
        raise ValueError('Workers must be positive')
    digest, trials = load_trials(metadata, metadata_sha256, sample_ids=sample_ids, background_y=background_y)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    media_dir = output / 'videos'
    media_dir.mkdir(exist_ok=True)
    for _, name, _, _ in trials:
        path = media_dir / name
        if path.exists() or path.is_symlink():
            raise ValueError(f'Output already exists: {path}')

    with TemporaryDirectory(dir=output, prefix='.classic-t2-') as temp:
        work = Path(temp)
        bank, index = stimuli.prepare_states({'protocol_id': digest, 'rows': [t[0] for t in trials]},
                                              work / 'state_cache')
        jobs = [(row, index['states'][manifest.state_key(row)], bank, work, 0) for row, _, _, _ in trials]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for (row, name, video_sha, stream_sha), item in zip(trials, pool.map(stimuli._encode_one, jobs)):
                sid = row['sample_id']
                if item['stream_sha256'] != stream_sha:
                    raise ValueError(f'Decoded RGB16 SHA-256 differs for {sid}')
                video = work / item['video_relative']
                output_path = video.replace(media_dir / name)
                yield dict(sample_id=sid, metadata_sha256=digest, video_relative=output_path.relative_to(output).as_posix(),
                           decoded_stream_sha256=stream_sha, frames=item['frames'],
                           max_pts_error_s=item['max_PTS_error_s'],
                           published_video_sha256=video_sha, reproduced_video_sha256=item['video_sha256'],
                           container_match=item['video_sha256'] == video_sha)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True, help='Published classic/t2/metadata.jsonl')
    parser.add_argument('--metadata-sha256', help='Verify the metadata against a known SHA-256 before rendering')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=1)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--sample-id', action='append', help='Repeat to render several trials')
    selection.add_argument('--background-y', type=float, choices=(.05, .20, .30, .40))
    selection.add_argument('--all', action='store_true', help='Render every trial in the metadata')
    args = parser.parse_args(argv)
    try:
        for record in reproduce(args.metadata, args.output, sample_ids=args.sample_id,
                                background_y=args.background_y, metadata_sha256=args.metadata_sha256,
                                workers=args.workers):
            print(json.dumps(record, sort_keys=True), flush=True)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')


if __name__ == '__main__':
    main()
