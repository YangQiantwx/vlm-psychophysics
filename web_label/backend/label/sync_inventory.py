"""Build a private, blind preview inventory from completed GCS browser indexes.

qy Python. Trigger once after an index is sealed; activation uses GCS generation
preconditions. This never opens task policy or starts participant collection.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import struct
from pathlib import Path
from urllib.parse import urlsplit

from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT', 'access-cis230202-570037')
BUCKET = os.environ.get('LABEL_BUCKET', 'cvpr2027-label-access-cis230202-570037-us-west1')
VALIDATION = 'PASS_ALL_180_RGB8_CRC_SHA_TIMESTAMPS'
TASKS = {
    'T1': ('Motion', 'Which direction do the stripes move?', ['LEFT', 'RIGHT']),
    'T2': ('Brightness', 'Which side becomes brighter?', ['LEFT', 'RIGHT']),
    'T3': ('Orientation', 'Which object is closer to vertical?', ['LEFT', 'RIGHT']),
    'T4': ('Global motion', 'Which horizontal direction is the coherent motion signal moving?', ['LEFT', 'RIGHT']),
    'T5': ('Target detection', 'Does either event match the target orientation?', ['YES', 'NO']),
}


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value)


def load_index(bucket, uri):
    url = urlsplit(uri)
    need(url.scheme == 'gs' and url.netloc == bucket.name, 'Index must be in the isolated CVPR bucket')
    blob = bucket.blob(url.path.lstrip('/'))
    raw = blob.download_as_bytes()
    index = json.loads(raw)
    need(index.get('schema') == 'validated-browser-library-v1', 'Unexpected browser index schema')
    need(str(index.get('status', '')).startswith('COMPLETE'), 'Browser index has not completed')
    items = index['items']
    counts = collections.Counter(row['task'] for row in items)
    if 'count' in index:
        need(int(index['count']) == len(items), 'Browser index top-level count disagrees with its items')
    expected = {k: int(v) for k, v in index['expected_counts'].items() if k in TASKS}
    need(dict(counts) == expected, 'Browser index expected_counts disagree with its items')
    return index, {'object': blob.name, 'generation': str(blob.generation), 'sha256': sha(raw), 'version': index['version']}


def remote_objects(client, bucket, prefixes):
    objects = {}
    for prefix in prefixes:
        for blob in client.list_blobs(bucket, prefix=prefix, page_size=1000, fields='items(name,size,metadata,generation,md5Hash),nextPageToken'):
            objects[blob.name] = blob
    return objects


def build_manifest(indexes, inputs, objects, bucket, required_tasks, version=None):
    rows = [row for index in indexes for row in index['items']]
    tasks = {}
    identity, physical = set(), set()
    references = {}
    for row in rows:
        tid = row['task']
        need(tid in TASKS, 'Unknown task')
        sid = str(row['id'])
        need(sid and row.get('source_id', sid) == sid and (tid, sid) not in identity, 'Repeated or inconsistent source identity')
        identity.add((tid, sid))
        need(valid_sha(row['sha256']) and valid_sha(row['source_sha256']) and valid_sha(row['qa_sha256']), 'Missing content or source QA hash')
        need((tid, row['source_sha256']) not in physical, 'Two canonical source IDs repeat the same physical video within a task')
        physical.add((tid, row['source_sha256']))
        need((row['width'], row['height'], row['fps'], row['frames'], row['duration']) == (768, 512, 60, 180, 3), 'Video raster or timing differs from current protocol')
        validation = row['validation']
        need(validation.get('status') == VALIDATION, 'A source-to-browser all-frame validation is missing')
        need(validation.get('source_route') == 'official_OpenCV_RGB8_X', 'Browser derivative is not bound to the current official RGB8 reader route')
        need(valid_sha(validation.get('source_RGB8_sha256')) and validation['source_RGB8_sha256'] == validation.get('derivative_RGB8_sha256'), 'Source/browser decoded RGB stream differs')
        need(validation.get('source_master_bits') in (8, 16), 'Source bit depth is not identified')
        need(isinstance(row.get('accuracy_eligible'), bool), 'Accuracy eligibility must be explicit, including zero-signal controls')
        need(row['domain'] in ('classic', 'natural'), 'Index includes a non-current domain')
        name = row['object']
        need(name.endswith('.webm') and name.startswith('browser/') and '..' not in name.split('/'), 'Unexpected browser object path')
        remote = objects.get(name)
        need(remote and int(remote.size) == int(row['bytes']), 'Remote video is absent or has different bytes: ' + sid)
        need((remote.metadata or {}).get('sha256') == row['sha256'], 'Remote video SHA metadata differs: ' + sid)
        title, prompt, choices = TASKS[tid]
        task = tasks.setdefault(tid, {'title': title, 'prompt': prompt, 'choices': choices, 'items': []})
        # Allocation retains condition balancing without exposing GT or parameter labels.
        group = sha(encoded(row.get('stratum', {})))[:24]
        item = {k: row[k] for k in ('object', 'sha256', 'bytes', 'source_sha256', 'width', 'height', 'duration')}
        item.update(id=sid, fps=60, frames=180, source_RGB8_sha256=validation['source_RGB8_sha256'], source_master_bits=validation['source_master_bits'], accuracy_eligible=row['accuracy_eligible'], dataset=row['dataset'], domain=row['domain'], qa_sha256=row['qa_sha256'], stratum={'allocation_group': group})
        ref = row.get('reference')
        if tid == 'T5':
            need(isinstance(ref, dict), 'T5 is missing its current reference')
            need((ref.get('width'), ref.get('height')) == (160, 160) and valid_sha(ref.get('sha256')), 'T5 reference identity or160×160 dimensions differ')
            need(ref['object'].startswith('references/') and '..' not in ref['object'].split('/'), 'Unexpected reference object path')
            remote_ref = objects.get(ref['object'])
            need(remote_ref and int(remote_ref.size) == int(ref['bytes']) and (remote_ref.metadata or {}).get('sha256') == ref['sha256'], 'Reference upload identity differs')
            if ref['object'] not in references:
                raw = bucket.blob(ref['object']).download_as_bytes()
                need(sha(raw) == ref['sha256'] and raw[:8] == b'\x89PNG\r\n\x1a\n' and struct.unpack('>II', raw[16:24]) == (160, 160), 'Current PNG reference bytes are invalid')
                references[ref['object']] = ref['sha256']
            item['reference'] = {k: ref[k] for k in ('object', 'sha256', 'bytes', 'width', 'height')}
        else:
            need(ref is None, 'Unexpected reference on a non-T5 trial')
        task['items'].append(item)
    need(set(required_tasks).issubset(tasks), 'A required task is absent')
    signature = sha(encoded(inputs))[:16]
    release = version or '20261030-browser-' + signature
    need(re.fullmatch(r'[A-Za-z0-9_.-]+', release), 'Invalid release version')
    manifest = {'schema': 'blind-label-inventory-v2', 'version': release, 'mode': 'test', 'qualification': 'Sealed browser indexes; all180 decoded RGB8/CRC/SHA/PTS per video; GCS size/SHA metadata checked for every object; each distinct T5 PNG SHA and160×160 IHDR independently checked. Browser playback/pixel inspection remains separately recorded.', 'inputs': inputs, 'tasks': tasks}
    return manifest, {'counts': {tid: len(task['items']) for tid, task in tasks.items()}, 'accuracy_eligible_counts': {tid: sum(item['accuracy_eligible'] for item in task['items']) for tid, task in tasks.items()}, 'distinct_references': len(references)}


def create_or_verify(bucket, key, raw):
    blob = bucket.blob(key)
    try:
        blob.upload_from_string(raw, content_type='application/json', if_generation_match=0)
    except PreconditionFailed:
        need(blob.download_as_bytes() == raw, 'Immutable manifest key already contains different bytes')


def activate(bucket, key):
    blob = bucket.blob('manifests/current-test.json')
    try:
        old = json.loads(blob.download_as_bytes())
        generation = int(blob.generation)
        if old.get('object') == key:
            return False
    except NotFound:
        generation = 0
    blob.upload_from_string(encoded({'object': key}), content_type='application/json', if_generation_match=generation)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=PROJECT)
    parser.add_argument('--bucket', default=BUCKET)
    parser.add_argument('--index', action='append', required=True, help='Sealed gs:// isolated-bucket browser index; repeat to combine indexes')
    parser.add_argument('--require-tasks', default='T1,T2,T3,T4,T5')
    parser.add_argument('--version')
    parser.add_argument('--expected-counts', help='Optional JSON object of required final task counts')
    parser.add_argument('--activate', action='store_true')
    parser.add_argument('--retain-other-tasks', action='store_true', help='Keep currently qualified task subsets absent from these indexes; receipt reports their exact counts')
    parser.add_argument('--output', type=Path, required=True, help='Private manifest; never put under site/')
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    need('site' not in args.output.parts, 'Private inventory cannot be written under site/')
    client = storage.Client(project=args.project)
    bucket = client.bucket(args.bucket)
    indexes, inputs = [], []
    for uri in args.index:
        index, identity = load_index(bucket, uri)
        indexes.append(index); inputs.append(identity)
    retained = {}
    if args.retain_other_tasks:
        try:
            pointer = json.loads(bucket.blob('manifests/current-test.json').download_as_bytes())
            previous = json.loads(bucket.blob(pointer['object']).download_as_bytes())
            incoming = {row['task'] for index in indexes for row in index['items']}
            retained = {tid: task for tid, task in previous['tasks'].items() if tid not in incoming}
            for task in retained.values():
                for item in task['items']:
                    if set(item.get('stratum', {})) != {'allocation_group'}:
                        item['stratum'] = {'allocation_group': sha(encoded(item.get('stratum', {})))[:24]}
            if retained:
                inputs.append({'retained_task_subsets_sha256': sha(encoded(retained)), 'counts': {tid: len(task['items']) for tid, task in retained.items()}})
        except NotFound:
            pass
    objects = remote_objects(client, bucket, ('browser/', 'references/'))
    manifest, counts = build_manifest(indexes, inputs, objects, bucket, [] if retained else args.require_tasks.split(','), args.version)
    if retained:
        manifest['tasks'].update(retained)
        manifest['retained_task_subsets'] = {tid: len(task['items']) for tid, task in retained.items()}
        manifest['qualification'] += ' Retained task subsets preserve their earlier qualification scope; they are not asserted to cover a full index.'
        need(set(args.require_tasks.split(',')).issubset(manifest['tasks']), 'A required task is absent after retaining qualified subsets')
        counts['counts'] = {tid: len(task['items']) for tid, task in manifest['tasks'].items()}
        counts['accuracy_eligible_counts'] = {tid: sum(item.get('accuracy_eligible', True) for item in task['items']) for tid, task in manifest['tasks'].items()}
        counts['retained_subsets'] = manifest['retained_task_subsets']
    if args.expected_counts:
        need(counts['counts'] == json.loads(args.expected_counts), 'Verified task counts disagree with the requested final library')
    raw = encoded(manifest)
    key = 'manifests/test/' + manifest['version'] + '.json'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    create_or_verify(bucket, key, raw)
    pointer_changed = activate(bucket, key) if args.activate else False
    receipt = {'status': 'COMPLETE_PRIVATE_INVENTORY_SYNC', 'bucket': args.bucket, 'manifest_object': key, 'manifest_sha256': sha(raw), **counts, 'activated': args.activate, 'pointer_changed': pointer_changed, 'inputs': inputs, 'task_policy': 'Controlled separately; source/browser review precedes openingT4', 'mode': 'test'}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
