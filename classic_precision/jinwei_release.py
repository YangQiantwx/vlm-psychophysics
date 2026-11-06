"""Read and verify the frozen media/metadata used by the provider evaluation."""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from urllib.request import urlopen

REPO = 'DrQY/vlm-psychophysics'
REVISION = '20baa977b183c8ecf56b5e4f7da569c5a93e9464'
HF_URL = f'https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/'
INDEX_SHA = '096a675cfce1beeede63fdd3c1fec0f49a3c5deeafe7956fb9de96ee45897731'
PRIVATE_SHA = '8eb931fa5c8c0103a6c2efbf9b0f27717147a26632f034eaa8e9f06c29c3f964'
PRIVATE_GENERATION = '1793982719470388'
MIME = {'.png': 'image/png', '.mp4': 'video/mp4', '.mkv': 'video/x-matroska'}
# Logical video rows, independent static cases, video manifest SHA, static manifest SHA.
SCOPES = {
    'classic/t1': (4224, 4224, '0573885dca13d2de8b9c80e08bde6072cd1591b31a347a1ded27b96f7ff3675b',
                   '7de354148944fbdef8d4097a71d33571f1dde30c63e884f6fe1044ef34941783'),
    'classic/t2': (2880, 120, '4f558ed5de53722d8082ce5b061b22d160ccecdb50ca3e843510efbfe83fd50e',
                   '777fd0bfa827ac3c3a3aa17552d6e1b5911b93a0eaa3236c95474d9ef76bd8b4'),
    'classic/t3': (2736, 114, '8067cf3d1c61212dc986fa1a3b95200c41e19d10ba43f321f8c8fba889bfc761',
                   '6123cbce7d69b8595eb47cbdca3ee660022bf28ae2101488a65de225ee5ca131'),
    'classic/t4': (768, 768, 'c22f7c6e1139910c9b4c45e8b9dde9d44eec67a25e4a45386ad3d43b478195fa',
                   'b5992e1a8cc65464caa59be7655cf514a9d61e465f29b57f3a7bf3cefd573926'),
    'classic/t5': (96, 0, '27655bda3db6f2f96ffa2a075fc6e3a5fd8e2a8986cfe0d9b860de6c4b2bdc59',
                   '7a0903b3e3de8372427f3cfaa0cc13da67661da926e1e3f3df687ac9f27c58fe'),
    'natural/t1': (4224, 4224, 'fac90aa3a3fb4bef5fcd460df012f5c47aa546d72cc7990bfd7786d6f4308ee4',
                   '5a09bc7961cdcbf2dc71d479837056a053e196a5185fa925c170df972d2557cb'),
    'natural/t2': (960, 960, 'cca3ff95ba07e9cd3afb820c8035aa769cdbfef9c1697ba857e92ae5e97ff337',
                   'e3abe55e58b069c691650277526079bb9defc48e361e5d12d73ee946c8a60d33'),
    'natural/t3': (2736, 2736, 'e22b40af55214ce03cd0b78cb2af032e855b264977cb4645a4e0e00985be9eb9',
                   '02a711cd6757c51d13eda01b3e763ba29d7b009a7def1bc3c230a9a99cc284de'),
    'natural/t4': (1536, 1536, 'e35b96b874d2ce2efa97df89eacb29e644d77690f6137b4ba6565e8aae0583c1',
                   'f37c891886819bf9804823b60a314a618959a696d3b0ef7ed3c76abe193fb412'),
    'natural/t5': (96, 0, 'af0b8a6a53c6f40b427202f6a15653b483be2b978059498a8f44da103d269c9c',
                   '8dab0b48dc501bcb48afa96c6255d3a39e014dff52aeab9ef8608d876a0e1bd9'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=False, allow_nan=False) + '\n').encode()


def checked(data, expected, name):
    require(sha(data) == expected, f'SHA-256 mismatch for {name}')
    return data


def relative(path, scope):
    value = PurePosixPath(path)
    require(not value.is_absolute() and '..' not in value.parts
            and value.is_relative_to(scope), f'Unexpected relative path {value}')
    return value.as_posix()


def media(path, digest, scope, **fields):
    path = relative(path, scope)
    require(re.fullmatch(r'[0-9a-f]{64}', digest), f'Missing SHA-256 for {path}')
    return dict(path=path, sha256=digest, mime=MIME[PurePosixPath(path).suffix], **fields)


def trial(scope, video, protocol, manifest_sha, static=None, private=False):
    domain, name = scope.split('/')
    task = name.upper()
    sid = video['sample_id']
    require(video['domain'] == domain and video['task'] == task, f'Wrong domain or task for {sid}')
    choices = ['YES', 'NO'] if task == 'T5' else ['LEFT', 'RIGHT']
    truth = video['truth']
    require((truth is None or truth in choices) and video.get('choices', choices) == choices
            and video.get('accuracy_eligible', truth is not None) == (truth is not None),
            f'Unexpected ground truth or choices for {sid}')
    prompt = video['prompt']
    require(isinstance(prompt, str) and bool(prompt.strip()), f'Missing prompt for {sid}')
    path = video.get('video_path') or video['path']
    parameters = video.get('parameters') or video.get('source_parameters') or video.get('frozen_row', {}).get('source_parameters')
    require(isinstance(parameters, dict), f'Missing parameters for {sid}')
    mask = video.get('mask', parameters.get('mask')) if task == 'T5' else None
    original = media(path, video['video_sha256'], scope, role='video')
    source = 'private' if private else 'public'
    mode = 'static' if static is not None else 'video'
    inspection = mode == 'static' and task == 'T5'
    case = sid
    if static is not None:
        require((static['sample_id'], static['truth'], static['video_sha256'], static['video_path'])
                == (sid, truth, original['sha256'], original['path']), f'Wrong static parent for {sid}')
        if inspection:
            require(static['purpose'] == 'stimulus inspection' and not static['model_evaluated'],
                    f'Unexpected T5 static purpose for {sid}')
        else:
            require(static['choices'] == choices and isinstance(static['prompt'], str)
                    and bool(static['prompt'].strip()), f'Invalid static prompt or choices for {sid}')
            prompt = static['prompt']
            case = static['static_case_id']
        ordered = [media(item['path'], item['sha256'], scope,
                         **{k: v for k, v in item.items() if k not in ('path', 'sha256')})
                   for item in static['images']]
        expected = {'T1': 4, 'T2': 1, 'T3': 1, 'T4': 2, 'T5': 3}[task]
        require(len(ordered) == expected and all(x['mime'] == 'image/png' for x in ordered),
                f'Unexpected static inputs for {sid}')
    elif task == 'T5':
        ordered = [media(video['reference_path'], video['reference_sha256'], scope, role='reference'), original]
    else:
        ordered = [original]
    request_id = None if inspection else sha(encoded([scope, mode, case]))
    profile = '/'.join([scope, mode] + (['masked' if private else 'unmasked'] if task == 'T5' else []))
    return dict(scope=scope, domain=domain, task=task, input=mode, sample_id=sid,
                source=source, source_manifest_sha256=manifest_sha, protocol=protocol,
                source_version=video.get('dataset_revision') or video.get('source_version') or video.get('protocol_id'),
                source_video=original, original_video_ID=sid, legacy_sample_id=video.get('legacy_sample_id'),
                source_record_sha256=video.get('source_record_sha256') or video.get('source_manifest_sha256'),
                condition_key=video.get('condition_key'), parameters=parameters, mask=mask,
                background_id=video.get('background_id', video.get('frozen_row', {}).get('background_id')),
                object_id=video.get('object_id', video.get('frozen_row', {}).get('object_id')),
                object_ids_in_point_order=video.get('object_ids_in_point_order'),
                reference=next((item for item in ordered if item.get('role') == 'reference'), None),
                media=ordered, static_case_id=case if static is not None and not inspection else None,
                model_input_sha256=(static or {}).get('model_input_sha256'),
                static_alias_ids=(static or {}).get('static_alias_ids'),
                frame_indices=(static or {}).get('frame_indices'), source_video_format={
                    k: video[k] for k in ('frames', 'fps', 'width', 'height')},
                request_id=request_id, profile=profile, prompt=prompt, prompt_sha256=sha(prompt.encode()),
                choices=choices, truth=truth, accuracy_eligible=not inspection and truth is not None,
                planned_status='INSPECTION_ONLY' if inspection else 'CALLABLE')


def row_map(raw, count):
    rows = [json.loads(line) for line in raw.splitlines()]
    by_id = {r['sample_id']: r for r in rows}
    require(len(by_id) == len(rows) == count, f'Expected {count} unique manifest IDs')
    return by_id


def public_plan(root=None):
    def get(path, digest):
        if root is not None:
            data = (root / path).read_bytes()
        else:
            with urlopen(HF_URL + path, timeout=120) as response:
                data = response.read()
        return checked(data, digest, path)

    index = json.loads(get('dataset-index.json', INDEX_SHA))
    require(len(index['scopes']) == len(SCOPES), 'Unexpected public scope count')

    def scope_plan(entry):
        name, info = entry
        scope = '/'.join(name.split('/')[:2])
        count, unique_static, video_sha, static_sha = SCOPES[scope]
        require(info['media_count'] == info['static_trial_count'] == count,
                f'Scope count differs for {scope}')
        video = row_map(get(relative(info['metadata'], scope), video_sha), count)
        static = row_map(get(relative(info['static_metadata'], scope), static_sha), count)
        require(video.keys() == static.keys(), f'Public video and static IDs differ for {scope}')
        protocol = info.get('protocol') or {
            'classic/t1': 'classic-rgb16-v0.4-plus-m05',
            'classic/t2': 'classic-rgb16-v0.4',
            'classic/t3': 'classic-rgb16-v0.4',
            'classic/t4': 'classic-t4-strict-v1',
            'natural/t1': 'natural-t1-rgb-20261030-plus-m05',
            'natural/t3': 'natural-t3-lr-paired-20261102',
        }[scope]
        items = []
        for sid in sorted(video):
            items.append(trial(scope, video[sid], protocol, video_sha))
            items.append(trial(scope, video[sid], protocol + '/static', static_sha, static[sid]))
        require(len({r['request_id'] for r in items if r['input'] == 'static' and r['request_id']})
                == unique_static, f'Independent static count differs for {scope}')
        return items

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = [row for items in pool.map(scope_plan, sorted(index['scopes'].items())) for row in items]
    return rows


class PrivateDelta:
    """Generation-pinned, explicitly selected GCS supplement. No bucket is built in."""

    def __init__(self, uri, metadata_root=None):
        self.root = metadata_root
        self.bucket = None
        self.locations = {}
        if uri:
            from google.cloud import storage
            require(uri.startswith('gs://') and uri.endswith('#' + PRIVATE_GENERATION),
                    'Use the authorized private index URL with its exact generation suffix')
            bucket, name = uri[5:].rsplit('#', 1)[0].split('/', 1)
            self.bucket = storage.Client().bucket(bucket)
        else:
            require(metadata_root is not None, 'An authorized private index or local metadata root is required')
        if metadata_root is None:
            blob = self.bucket.blob(name, generation=int(PRIVATE_GENERATION))
            raw = blob.download_as_bytes(if_generation_match=int(PRIVATE_GENERATION),
                                         raw_download=True, timeout=60)
        else:
            raw = (metadata_root / 'index.json').read_bytes()
        self.index = json.loads(checked(raw, PRIVATE_SHA, 'private index'))
        require(self.index['hf_release_revision'] == REVISION
                and self.index['t5_private_masked_trials'] == 192, 'Unexpected private release')
        if self.bucket:
            require(self.index['bucket'] == self.bucket.name, 'Private bucket differs from pinned index')

    def object_bytes(self, entry):
        generation = int(entry['generation'])
        blob = self.bucket.blob(entry['object'], generation=generation)
        raw = blob.download_as_bytes(if_generation_match=generation, raw_download=True, timeout=60)
        require(len(raw) == entry['bytes'], 'Private object length differs')
        return checked(raw, entry['sha256'], 'private object')

    def plan(self, public):
        rows = []
        for scope in ('classic/t5', 'natural/t5'):
            index = self.index
            path = index['scopes'][scope]['metadata']
            entry = index['files'][path]
            raw = ((self.root / path).read_bytes() if self.root else self.object_bytes(entry))
            records = row_map(gzip.decompress(checked(raw, entry['sha256'], scope)), 192)
            current = {r['sample_id']: r for r in public if r['scope'] == scope and r['input'] == 'video'}
            unmasked = {sid for sid, record in records.items() if record['source']['mask'] == 'none'}
            require(unmasked == current.keys(), f'Private/public unmasked IDs differ for {scope}')
            for sid, record in sorted(records.items()):
                source = record['source']
                require(source['sample_id'] == sid and record['truth'] == source['truth']
                        and record['video']['sha256'] == source['video_sha256']
                        and record['reference']['sha256'] == source['reference_sha256'],
                        f'Private source mismatch for {sid}')
                if sid in unmasked:
                    expected = current[sid]
                    require((expected['source_video']['sha256'], expected['reference']['sha256'],
                             expected['truth'], expected['prompt']) ==
                            (source['video_sha256'], source['reference_sha256'], source['truth'], source['prompt']),
                            f'Private/public unmasked identity differs for {sid}')
                    continue
                require(record['static'] is None and not source['publicly_redistributable'],
                        f'Unexpected private static or redistribution status for {sid}')
                for key in ('video', 'reference'):
                    item = record[key]
                    require(item['path'] == source[key + '_path'], f'Private path differs for {sid}')
                    self.locations[item['path']] = item
                rows.append(trial(scope, source, record['protocol'], entry['sha256'], private=True))
        require(len(rows) == 192, 'Expected 192 masked T5 trials')
        return rows

    def media_bytes(self, item):
        require(self.bucket is not None, 'Use an authorized --private-index or provide a verified local media root')
        location = self.locations[item['path']]
        require('direct' in location, 'Expected a direct object for T5 media')
        raw = self.object_bytes(location['direct'])
        require(len(raw) == location['bytes'], 'Private T5 media length differs')
        return raw


def public_media(item):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=REPO, repo_type='dataset', revision=REVISION,
                           filename=item['path'], token=False)
    return Path(path).read_bytes()


def unique_requests(rows):
    requests = {}
    signatures = {}
    identities = set()
    for row in rows:
        identity = row['scope'], row['input'], row['sample_id']
        require(identity not in identities, f'Duplicate release trial {identity}')
        identities.add(identity)
        key = row['request_id']
        if key is None:
            continue
        signature = (row['profile'], row['prompt'], tuple(row['choices']), row['truth'],
                     tuple((item['mime'], item['sha256']) for item in row['media']))
        if key in signatures:
            require(signatures[key] == signature, f'Inconsistent static alias {key}')
        else:
            requests[key] = row
            signatures[key] = signature
    return requests


def counts(rows):
    grouped = {}
    for row in rows:
        group = grouped.setdefault(row['profile'], Counter())
        group['logical_rows'] += 1
        group['planned_scorable_rows'] += row['accuracy_eligible']
        group['inspection_rows'] += row['request_id'] is None
    for row in unique_requests(rows).values():
        grouped[row['profile']]['unique_requests'] += 1
    return {name: dict(value) for name, value in sorted(grouped.items())}
