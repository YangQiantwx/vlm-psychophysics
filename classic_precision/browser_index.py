"""Seal existing per-video QA into a private browser index and upload inventory."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--inventory', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--require-count', type=int)
    ap.add_argument('--subset-ids', type=Path)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    args = ap.parse_args()
    source = json.loads(args.inventory.read_text())
    rows = source['items']
    rows = [r for i,r in enumerate(rows) if i % args.shards == args.shard]
    selected = None
    if args.subset_ids:
        selected = set(json.loads(args.subset_ids.read_text())['site_subset_ids'])
        rows = [r for r in rows if r['id'] in selected]
    refs = {r['sha256']: r for r in source.get('references', [])}
    index, upload, used_refs, site, missing = [], [], {}, [], []
    for row in rows:
        sid, dataset = row['id'], row['dataset']
        key = hashlib.sha256(f'{dataset}:{sid}'.encode()).hexdigest()
        qa = args.root / 'runs/browser-library-20261030' / dataset / f'{key}.json'
        media = qa.with_suffix('.webm')
        if not qa.exists() or not media.exists():
            missing.append(sid)
            continue
        r = json.loads(qa.read_text())
        assert r['id'] == sid and r['source_sha256'] == row['sha256']
        assert media.stat().st_size == r['bytes']
        q = r['qa']
        assert q['status'] in ('ALL_180_RGB8_CRC_SHA_AND_TIMESTAMPS_PASS',
                               'FULL_180_RGB_BYTES_AND_TIMESTAMPS_PASS')
        if 'frame_crc32' in q:
            assert len(q['frame_crc32']) == 180
        original = q.get('source_RGB8_sha256') or q.get('source_RGB_sha256')
        decoded = q.get('derivative_RGB8_sha256') or q.get('derivative_RGB_sha256')
        assert original == decoded
        if row.get('source_RGB8_sha256'):
            assert original == row['source_RGB8_sha256']
        stratum = row.get('stratum', {})
        task = row.get('task') or stratum.get('task')
        if task is None:
            task = row['object'].split('/')[2]
        assert task in ('T1', 'T2', 'T3', 'T4', 'T5')
        domain = row.get('domain') or ('classic' if dataset.startswith('classic') else 'natural')
        entry = {k: r[k] for k in ('id', 'object', 'sha256', 'bytes', 'source_sha256', 'width', 'height', 'duration')}
        entry.update(source_id=sid, task=task, domain=domain, dataset=dataset, fps=60, frames=180,
                     accuracy_eligible=row.get('accuracy_eligible', not (task == 'T4' and stratum.get('coherence') == 0)),
                     stratum=stratum, reference=None, qa_sha256=digest(qa), validation={
                     'status': 'PASS_ALL_180_RGB8_CRC_SHA_TIMESTAMPS',
                     'source_route': 'official_OpenCV_RGB8_X', 'source_RGB8_sha256': original,
                     'derivative_RGB8_sha256': decoded,
                     'source_master_bits': 16 if dataset.startswith('classic-t1-t3') else 8})
        example = {'alias': r['sha256'], 'task': task, 'domain': domain, 'object': r['object'],
                   'sha256': r['sha256'], 'status': 'current', 'source_id': sid,
                   **{k: v for k,v in stratum.items() if k in ('seed','coherence','direction','k','position_pair','scenario','mask')}}
        if row.get('reference_sha256'):
            from PIL import Image
            h = row['reference_sha256']
            ref = refs[h]
            p = args.root / ref['path'].split('cvpr2027_benchmark/', 1)[-1]
            assert p.exists() and digest(p) == h
            assert Image.open(p).size == (160, 160)
            entry['reference'] = {'object': ref['object'], 'sha256': h, 'width': 160,
                                  'height': 160, 'bytes': p.stat().st_size}
            used_refs[h] = {**ref, 'path': str(p), 'bytes': p.stat().st_size}
            example['reference_alias'] = h
        index.append(entry)
        site.append(example)
        upload.append({'path': str(media), 'object': r['object'], 'sha256': r['sha256'],
                       'bytes': r['bytes'], 'source_sha256': row['sha256'], 'dataset': dataset,
                       'content_type': 'video/webm'})
    if args.require_count is not None:
        assert len(index) == args.require_count and not missing, (len(index), args.require_count, len(missing))
    for h,ref in used_refs.items():
        site.append({'alias': h, 'task': 'T5', 'domain': ref['domain'], 'object': ref['object'],
                     'sha256': h, 'status': 'current'})
    assert len({r['id'] for r in index}) == len(index)
    doc = {'schema': 'validated-browser-library-v1', 'version': '20261030-fixed-current',
           'status': 'COMPLETE_SELECTED_SCOPE' if not missing else 'PARTIAL_NOT_READY_FOR_FULL_ACTIVATION',
           'expected_counts': dict(Counter((r.get('task') or r.get('stratum',{}).get('task') or r['object'].split('/')[2]) for r in rows)),
           'count': len(index), 'missing': len(missing), 'items': index}
    args.out.mkdir(parents=True, exist_ok=True)
    for name,body in [('index.json',doc),('uploads.json',{'items':upload}),
                      ('references.json',{'items':list(used_refs.values())}),
                      ('site-release.json',{'version':'20261030-fixed-current','items':site})]:
        (args.out/name).write_text(json.dumps(body, indent=2) + '\n')
    print(json.dumps({'count':len(index),'references':len(used_refs),'missing':len(missing),
                      'bytes':sum(r['bytes'] for r in index)}))


if __name__ == '__main__':
    main()
