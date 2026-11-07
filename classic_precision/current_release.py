"""Fetch and verify inputs from the unified Classic/Natural T1–T5 release."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import inspect
import json
from pathlib import Path, PurePosixPath
import re


REPO_ID = 'DrQY/vlm-psychophysics'
REVISION = '3ede52f068164d2db4889d6a84cb153037da1a45'
SCOPES = tuple(f'{domain}/t{task}' for domain in ('classic', 'natural') for task in range(1, 6))


def payload_path(scope, name):
    path = PurePosixPath(name)
    if ('\\' in name or path.as_posix() != name or path.is_absolute()
            or '..' in path.parts or not path.is_relative_to(PurePosixPath(scope))
            or path == PurePosixPath(scope) or any(char in name for char in '*?[]')):
        raise ValueError(f'Unexpected {scope} release path: {name}')
    return name


def media_refs(row, modality):
    if modality == 'static':
        return [(image['path'], image['sha256']) for image in row['images']]
    refs = [(row.get('video_path') or row['path'], row['video_sha256'])]
    if row['task'] == 'T5':
        refs.append((row['reference_path'], row['reference_sha256']))
    return refs


def download(output, scopes=SCOPES, modality='both', *, revision=REVISION, limit=None, workers=8):
    from huggingface_hub import hf_hub_download

    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('Use a full immutable Hugging Face commit SHA.')
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError('Worker and per-scope limit values must be positive.')
    if not scopes or len(set(scopes)) != len(scopes) or any(scope not in SCOPES for scope in scopes):
        raise ValueError('Choose one or more distinct Classic/Natural task scopes.')
    if modality not in ('video', 'static', 'both'):
        raise ValueError('Input must be video, static, or both.')

    output = Path(output)
    hub = dict(repo_id=REPO_ID, repo_type='dataset', revision=revision, local_dir=output, token=False)
    if 'local_dir_use_symlinks' in inspect.signature(hf_hub_download).parameters:
        hub['local_dir_use_symlinks'] = False
    index = json.loads(Path(hf_hub_download(filename='dataset-index.json', **hub)).read_text())
    entries = index['scopes']
    refs = {}
    receipts = []
    for scope in scopes:
        index_key = scope
        if scope not in entries and scope.endswith('/t4'):
            index_key = f'{scope}/strict-v1'
        entry = entries[index_key]
        for source in ('video', 'static') if modality == 'both' else (modality,):
            manifest_key, count_key = ('metadata', 'media_count') if source == 'video' else (
                'static_metadata', 'static_trial_count')
            manifest = payload_path(scope, entry[manifest_key])
            path = Path(hf_hub_download(filename=manifest, **hub))
            with path.open(encoding='utf-8') as stream:
                rows = [json.loads(line) for line in stream if line.strip()]
            if len(rows) != entry[count_key] or len({row['sample_id'] for row in rows}) != len(rows):
                raise ValueError(f'Manifest count or sample IDs differ: {manifest}')
            rows = rows[:limit]
            for row in rows:
                if row['domain'] != scope.split('/')[0] or row['task'].lower() != scope.split('/')[1]:
                    raise ValueError(f'Trial scope differs: {row["sample_id"]}')
                for name, sha in media_refs(row, source):
                    name = payload_path(scope, name)
                    if not re.fullmatch('[0-9a-f]{64}', sha) or refs.setdefault(name, sha) != sha:
                        raise ValueError(f'Invalid or conflicting SHA-256: {name}')
            receipts.append(dict(scope=scope, input=source, selected_trials=len(rows), manifest=manifest))

    def fetch(item):
        name, expected = item
        path = Path(hf_hub_download(filename=name, **hub))
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'SHA-256 mismatch: {name}')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(fetch, refs.items()):
            pass
    return dict(repo_id=REPO_ID, revision=revision, scopes=receipts, files_sha256_verified=len(refs))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope', action='append', choices=SCOPES,
                        help='Repeat to select multiple scopes. Default is all ten scopes.')
    parser.add_argument('--input', choices=('video', 'static', 'both'), default='both')
    parser.add_argument('--revision', default=REVISION)
    parser.add_argument('--limit', type=int, help='First N trials per scope and input, for a small check.')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args(argv)
    try:
        result = download(args.output, args.scope or SCOPES, args.input, revision=args.revision,
                          limit=args.limit, workers=args.workers)
    except (OSError, ValueError) as error:
        parser.exit(2, f'{error}\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
