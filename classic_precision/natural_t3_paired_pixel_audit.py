"""CPU-only decoded-keyframe audit, identifying orientations before reading labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .dataset_content_audit import decode_states, match_alpha, patches, sha


NAME = 'natural-t3-classic-paired-20261102'


@lru_cache(maxsize=24)
def read_bg(path, expected):
    assert sha(path) == expected
    return np.load(path, allow_pickle=False)


def audit_object(job):
    root, object_id, rows, catalog = job
    root = Path(root); dataset = root/'datasets'/NAME
    patches.cache_clear()
    candidates = patches(str(dataset/'patches'), object_id)
    assert len(candidates) == 58
    records=[]
    for row in rows:
        source = catalog[row['sample_id']]
        video = dataset/'videos'/row['path']
        assert sha(video) == source['video_sha256']
        bg = read_bg(str(root/row['background_prepared_path']), row['background_prepared_sha256'])
        base, event, decoded = decode_states(video, row['onset_frame'], 'natural')
        # Candidate lookup knows object identity only, not expected angle/sign/side.
        observed = match_alpha(base, event, bg, candidates)
        assert observed['vertical_side'] == row['truth']
        sign = {'left':1,'right':-1,'horizontal':0}[observed['upper_end_direction']]
        assert sign == row['physical_tilt_sign']
        tilted = next(a for a in observed['axes'] if a['side']==observed['tilted_side'])
        vertical = next(a for a in observed['axes'] if a['side']==observed['vertical_side'])
        relative = abs((tilted['signed_deviation_deg']-vertical['signed_deviation_deg']+90)%180-90)
        assert abs(observed['angle_deg']-row['angle_deg']) <= .01
        assert abs(relative-row['angle_deg']) <= .02
        assert all(a['source_object_sha256']==row['object_source_sha256'] for a in observed['axes'])
        records.append(dict(status='PASS',sample_id=row['sample_id'],classic_sample_id=row['classic_sample_id'],
                            video_sha256=source['video_sha256'],expected_angle_deg=row['angle_deg'],
                            relative_angle_deg=relative,observed=observed,decoded=decoded))
    return records


def main(root, workers):
    root=Path(root); plan=root/'plans'/NAME;run=root/'runs/stimulus-preparation'/NAME
    allocation=json.loads((plan/'allocation.json').read_text())
    catalog={r['sample_id']:r for r in map(json.loads,(plan/'catalogs/T3.jsonl').read_text().splitlines())}
    byobject=defaultdict(list)
    for row in allocation['rows']:byobject[row['object_id']].append(row)
    started=time.monotonic();records=[]
    with ProcessPoolExecutor(workers) as pool:
        futures=[pool.submit(audit_object,(str(root),obj,rows,catalog)) for obj,rows in byobject.items()]
        for future in as_completed(futures):
            records.extend(future.result())
            progress=dict(status='RUNNING',videos=len(records),total=2736,elapsed_seconds=time.monotonic()-started,workers=workers)
            (run/'decoded-keyframe-progress.json').write_text(json.dumps(progress)+'\n')
            print(json.dumps(progress),flush=True)
    assert len(records)==len({r['sample_id'] for r in records})==2736
    records.sort(key=lambda r:r['sample_id'])
    target=run/'decoded-keyframe-audit.jsonl'
    target.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    small=[r['relative_angle_deg'] for r in records if r['expected_angle_deg']==.5]
    summary=dict(status='PASS_ALL2736_DECODED_KEYFRAME_UNIQUE_RGB_ORIENTATION',videos=2736,decoded_keyframes=5472,
                 exact_orientation_matches_per_side=1,candidate_orientations_per_object=58,model_calls=0,
                 label_independent_match='Object identity only selects all58 candidate orientations; exact decoded RGB match chooses each side before expected angle/sign/GT are checked.',
                 independent_measurement='OpenCV full-alpha moments on the uniquely RGB-matched source; RGB does not reveal a unique alpha without the known source assets/background.',
                 prior_full_video_QA='All180 decoded frames/PTS already accepted; this audit reads baseline and first event frame of every video, plus every complete video SHA.',
                 relative_angle_error_max_deg=max(abs(r['relative_angle_deg']-r['expected_angle_deg']) for r in records),
                 angle_0p5_relative_range_deg=[min(small),max(small)],
                 pixels_dropped_to_identical_vertical_orientation=0,
                 records=dict(path=str(target),sha256=sha(target)),
                 allocation_sha256=sha(plan/'allocation.json'),catalog_sha256=sha(plan/'catalogs/T3.jsonl'),
                 implementation_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,workers=workers)
    (run/'decoded-keyframe-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--workers',type=int,default=24)
    a=p.parse_args();main(a.root,a.workers)
