"""Private, version-bound label export. Never include these outputs in a public stage."""
import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()


def private_write(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        os.chmod(path, 0o600)
        f.write(raw)


def build_binding(manifest, manifest_raw, manifest_object, catalogs):
    sources, provenance = {}, []
    for path, raw in catalogs:
        provenance.append({'file': str(path), 'sha256': digest(raw)})
        for item in json.loads(raw)['items'].values():
            row = item['row']
            video = next(c for c in item['content'] if c['type'] == 'video')
            key = (row['task'], video['sha256'])
            entry = {'source_filename': Path(video['path']).name,
                     'source_sha256': video['sha256'], 'model_sample_id': row['sample_id'],
                     'truth': row['truth'], 'source_parameters': row['source_parameters']}
            if key in sources:
                old = sources[key]
                if any(old[k] != entry[k] for k in ('truth', 'source_parameters', 'source_filename')):
                    raise ValueError('Conflicting source metadata for ' + str(key))
            sources[key] = entry
    items = {}
    for task, group in manifest['tasks'].items():
        for trial in group['items']:
            source = sources.get((task, trial['source_sha256']))
            if source is None:
                raise ValueError('Missing source metadata: ' + task + '/' + trial['id'])
            key = task + '/' + trial['id']
            if key in items:
                raise ValueError('Duplicate trial identity: ' + key)
            params = source['source_parameters']
            angle = params.get('angle_deg', params.get('angle'))
            if task == 'T3' and angle is None:
                raise ValueError('T3 is missing its angle: ' + key)
            items[key] = {**source, 'source_id': trial['id'], 'task': task,
                          'domain': trial['domain'], 'dataset': trial['dataset'],
                          'browser_sha256': trial['sha256'], 'browser_object': trial['object'],
                          'angle_deg': angle, 'accuracy_eligible': trial['accuracy_eligible']}
    return {'schema': 'private-label-source-binding-v1', 'manifest_version': manifest['version'],
            'manifest_object': manifest_object, 'manifest_sha256': digest(manifest_raw),
            'source_catalogs': provenance, 'count': len(items), 'items': items}


def joined_answer(answer, identity, bindings, record_object):
    mode = identity['mode']
    if answer['mode'] != mode:
        raise ValueError('Identity/answer mode mismatch')
    if not record_object.startswith(f"records/{mode}/{identity['participant_id']}/"):
        raise ValueError('Answer belongs to another participant')
    binding = bindings.get(answer['manifest_version'])
    if binding is None:
        raise ValueError('No pinned source binding for release ' + answer['manifest_version'])
    source = binding['items'].get(answer['task'] + '/' + answer['source_id'])
    if source is None:
        raise ValueError('Response source identity is absent from its pinned release')
    return {**answer, **source, 'participant_id': identity['participant_id'],
            'email': identity['email'], 'email_verified': False,
            'manifest_object': binding['manifest_object'], 'manifest_sha256': binding['manifest_sha256'],
            'source_parameters_json': json.dumps(source['source_parameters'], ensure_ascii=False, sort_keys=True),
            'record_object': record_object}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default='access-cis230202-570037')
    p.add_argument('--bucket', default='cvpr2027-label-access-cis230202-570037-us-west1')
    sub = p.add_subparsers(dest='action', required=True)
    build = sub.add_parser('bind')
    build.add_argument('--manifest', type=Path, required=True)
    build.add_argument('--manifest-object', required=True)
    build.add_argument('--catalog', type=Path, action='append', required=True)
    build.add_argument('--output', type=Path, required=True)
    build.add_argument('--upload', action='store_true', help='Create a private immutable binding, without changing the live inventory')
    export = sub.add_parser('export')
    export.add_argument('--binding', type=Path, action='append', required=True)
    export.add_argument('--email', required=True, help='Export only this identity, without scanning other participants')
    export.add_argument('--mode', choices=['test', 'human'], required=True)
    export.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.action == 'bind':
        raw = args.manifest.read_bytes()
        binding = build_binding(json.loads(raw), raw, args.manifest_object,
                                [(x, x.read_bytes()) for x in args.catalog])
        payload = encode(binding)
        object_name = f"manifests/analysis/{binding['manifest_version']}-{digest(payload)}.json"
        if args.upload:
            from google.cloud import storage
            from google.api_core.exceptions import PreconditionFailed
            bucket = storage.Client(project=args.project).bucket(args.bucket)
            if digest(bucket.blob(args.manifest_object).download_as_bytes()) != binding['manifest_sha256']:
                raise ValueError('Cloud inventory differs from the pinned local manifest')
            blob = bucket.blob(object_name)
            blob.metadata = {'sha256': digest(payload)}
            try:
                blob.upload_from_string(payload, content_type='application/json', if_generation_match=0)
            except PreconditionFailed:
                if blob.download_as_bytes() != payload:
                    raise ValueError('Existing binding bytes differ')
            if digest(blob.download_as_bytes()) != digest(payload):
                raise ValueError('Binding cloud readback differs')
        private_write(args.output, payload)
        print(json.dumps({'count': binding['count'], 'binding_sha256': digest(payload),
                          'private_cloud_object': object_name if args.upload else None}))
        return
    from google.cloud import storage
    bucket = storage.Client(project=args.project).bucket(args.bucket)
    bindings = {}
    for path in args.binding:
        binding = json.loads(path.read_text())
        if binding['schema'] != 'private-label-source-binding-v1':
            raise ValueError('Unknown source binding schema')
        if binding['manifest_version'] in bindings:
            raise ValueError('Duplicate release binding')
        if digest(bucket.blob(binding['manifest_object']).download_as_bytes()) != binding['manifest_sha256']:
            raise ValueError('Pinned manifest no longer matches its SHA')
        bindings[binding['manifest_version']] = binding
    email = args.email.strip().lower()
    identity = json.loads(bucket.blob(f'identities/{args.mode}/{digest(email.encode())}.json').download_as_bytes())
    if identity['email'] != email or identity['mode'] != args.mode:
        raise ValueError('Identity does not match requested email/mode')
    prefix = f"records/{args.mode}/{identity['participant_id']}/"
    objects = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith('-answer.json')]
    def join(blob):
        return joined_answer(json.loads(blob.download_as_bytes()), identity, bindings, blob.name)
    with ThreadPoolExecutor(max_workers=8) as workers:
        rows = list(workers.map(join, objects))
    rows.sort(key=lambda x: (x['server_at'], x['record_object']))
    columns = ['participant_id', 'email', 'email_verified', 'mode', 'test', 'research_collection',
               'manifest_version', 'manifest_object', 'manifest_sha256', 'task', 'domain', 'dataset',
               'source_id', 'model_sample_id', 'source_filename', 'source_sha256', 'browser_sha256',
               'browser_object', 'angle_deg', 'truth', 'answer', 'reason', 'accuracy_eligible',
               'playback_complete', 'reaction_ms', 'client_at', 'server_at', 'batch_id', 'batch_index',
               'trial_position', 'participant_seed', 'allocation_seed', 'order_algorithm',
               'reference_sha256', 'source_parameters_json', 'record_object']
    import io
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction='ignore')
    writer.writeheader(); writer.writerows(rows)
    private_write(args.output, buf.getvalue().encode('utf-8'))
    print(json.dumps({'rows': len(rows), 'all_records_joined': True, 'mode': args.mode,
                      'output_sha256': digest(args.output.read_bytes())}))


if __name__ == '__main__':
    main()
