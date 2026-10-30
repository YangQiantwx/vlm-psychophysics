"""Verify a large assigned media response using an isolated synthetic test identity.

A deterministic QA seed is selected before this identity's first claim so its
batch contains the specified video. All task orders still use the production
allocation algorithm and remain reconstructible. No global pointer is changed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time

import requests
from google.cloud import storage

from app import ordered


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', required=True)
    p.add_argument('--bucket', required=True)
    p.add_argument('--api', required=True)
    p.add_argument('--contract', type=Path, required=True, help='One qualified browser item')
    p.add_argument('--private-manifest', type=Path, required=True)
    p.add_argument('--receipt', type=Path, required=True)
    args = p.parse_args()
    contract = json.loads(args.contract.read_text())
    manifest = json.loads(args.private_manifest.read_text())
    tid, sid = contract['task'], contract['id']
    item = next(row for row in manifest['tasks'][tid]['items'] if row['id'] == sid)
    assert item['sha256'] == contract['sha256'] and item['bytes'] == contract['bytes']
    assert item['bytes'] > 32 * 1024 * 1024
    for trial in range(10000):
        seed = hashlib.sha256(('large-transfer-qa-' + str(trial)).encode()).hexdigest()[:32]
        order = ordered(manifest['tasks'][tid]['items'], seed + tid)
        if sid in order[:20]:
            break
    else:
        raise RuntimeError('No bounded QA allocation was found')
    email = 'large-transfer-qa-' + secrets.token_hex(8) + '@example.invalid'
    response = requests.post(args.api + '/api/session', json={'email': email, 'mode': 'test'}, timeout=60)
    response.raise_for_status()
    auth = {'Authorization': 'Bearer ' + response.json()['token']}
    bucket = storage.Client(project=args.project).bucket(args.bucket)
    identity = json.loads(bucket.blob('identities/test/' + hashlib.sha256(email.encode()).hexdigest() + '.json').download_as_bytes())
    key = 'records/test/' + identity['participant_id'] + '/state.json'
    blob = bucket.blob(key)
    initial = blob.download_as_bytes()
    state = json.loads(initial)
    assert state['active'] is None and all(task['cursor'] == 0 for task in state['tasks'].values())
    assert state['manifest_version'] == manifest['version']
    bucket.blob(key.removesuffix('state.json') + 'qa-initial-state.json').upload_from_string(initial, content_type='application/json', if_generation_match=0)
    state['seed'] = seed
    state['qa_seed_selection'] = {'test': True, 'purpose': 'large transfer SHA proof', 'source_id': sid, 'selection_trial': trial}
    for task_id, task in manifest['tasks'].items():
        state['tasks'][task_id]['order'] = ordered(task['items'], seed + task_id)
        state['tasks'][task_id]['allocation_seed'] = seed + task_id
    blob.upload_from_string(json.dumps(state), content_type='application/json', if_generation_match=int(blob.generation))
    response = requests.post(args.api + '/api/claim', headers=auth, json={'task': tid}, timeout=60)
    response.raise_for_status()
    batch = response.json()['batch']
    index = state['tasks'][tid]['order'].index(sid)
    assigned = batch['trials'][index]
    assert assigned['sha256'] == item['sha256']
    start = time.monotonic()
    size, digest = 0, hashlib.sha256()
    with requests.get(args.api + assigned['media'], headers=auth, stream=True, timeout=(30, 300)) as media:
        media.raise_for_status()
        for chunk in media.iter_content(1024 * 1024):
            size += len(chunk)
            digest.update(chunk)
        headers = {key: media.headers.get(key) for key in ('Content-Type', 'Content-Length', 'Transfer-Encoding')}
    assert size == item['bytes'] and digest.hexdigest() == item['sha256']
    assert headers['Content-Length'] is None
    receipt = {'status': 'PASS_ACTUAL_CLOUD_LARGE_STREAM_SHA', 'test': True, 'research_collection': False,
               'manifest_version': manifest['version'], 'source_id': sid, 'bytes': size, 'sha256': digest.hexdigest(),
               'elapsed_seconds': round(time.monotonic() - start, 3), 'response_headers': headers,
               'participant_id': identity['participant_id'], 'batch_id': batch['id'], 'index': index,
               'seed': seed, 'seed_selection': state['qa_seed_selection'],
               'scope': 'Actual assigned API bytes and streaming transport; no perceptual or browser pixel judgment. Synthetic QA seed selected before first claim.'}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
