"""Build the public frontend from versioned code and SHA-pinned public inputs."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def relative_path(name):
    path = Path(name)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError('Unsafe public input path')
    return path


def capture(stage, base_url):
    if urlsplit(base_url).scheme != 'https':
        raise ValueError('A public HTTPS deployment URL is required')
    rows = []
    for source in sorted(stage.rglob('*')):
        if source.is_symlink():
            raise ValueError('Snapshot inputs cannot be symlinks')
        if not source.is_file():
            continue
        relative = source.relative_to(stage)
        if relative.parts[0] in ('records', 'sessions', 'private'):
            raise ValueError('Participant or private data cannot be public inputs')
        if (HERE / 'frontend' / relative).is_file():
            continue
        if source.suffix in ('.js', '.css', '.html') or source.name in ('_headers', '_routes.json'):
            raise ValueError('Public code must be versioned under frontend/: ' + str(relative))
        if source.name.startswith('.'):
            continue
        raw = source.read_bytes()
        rows.append({'path': str(relative), 'bytes': len(raw), 'sha256': digest(raw)})
    payload = {'schema': 'public-site-inputs-v1', 'base_url': base_url.rstrip('/'),
               'scope': 'Published machine-result aggregates, stimulus metadata, attribution and review assets. No participant records. Lossless label videos remain private GCS API inputs.',
               'files': rows}
    (HERE / 'public-inputs.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'captured_inputs': len(rows), 'bytes': sum(r['bytes'] for r in rows)}))


def build(output, label_only=False, api=None, analysis_root=None):
    source = HERE / 'frontend'
    if output == HERE or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Output cannot contain or overwrite frontend source')
    output.mkdir(parents=True, exist_ok=True)
    marker = output / '.web-build.json'
    if any(output.iterdir()) and not marker.is_file():
        raise ValueError('Choose an empty output directory or a previous web build')
    marker.write_text(json.dumps({'schema': 'web-build-v1', 'status': 'building'}) + '\n')
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Frontend source cannot be a symlink')
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if label_only and relative.parts[0] != 'label':
            continue
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    inputs = json.loads((HERE / 'public-inputs.json').read_text())
    fetched = 0
    for row in inputs['files']:
        relative = relative_path(row['path'])
        if label_only and relative.parts[0] != 'label':
            continue
        target = output / relative
        if target.is_file() and target.stat().st_size == row['bytes'] and digest(target.read_bytes()) == row['sha256']:
            continue
        request = Request(inputs['base_url'] + '/' + relative.as_posix(), headers={'User-Agent': 'Mozilla/5.0 (compatible; VisionTasksBuild/1.0)'})
        with urlopen(request, timeout=90) as response:
            raw = response.read()
        if len(raw) != row['bytes'] or digest(raw) != row['sha256']:
            raise ValueError('Public input differs from its pinned SHA256: ' + str(relative))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + '.pending')
        temporary.write_bytes(raw)
        temporary.replace(target)
        fetched += 1
        if fetched % 25 == 0:
            print(json.dumps({'verified_public_inputs': fetched}), flush=True)
    if api:
        if urlsplit(api).scheme not in ('http', 'https'):
            raise ValueError('API must be an HTTP(S) origin')
        for name in ('label/config.json', 'experiments/t45-final.json'):
            path = output / name
            if path.is_file():
                value = json.loads(path.read_text())
                value['api'] = api.rstrip('/')
                path.write_text(json.dumps(value, separators=(',', ':')) + '\n')
    if analysis_root:
        from workspace_publish import refresh_t45
        refresh_t45(analysis_root, output / 'experiments', output)
    if label_only:
        (output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/label/"><a href="/label/">Label</a>\n')
    marker.write_text(json.dumps({'schema': 'web-build-v1', 'label_only': label_only, 'public_inputs_sha256': digest((HERE / 'public-inputs.json').read_bytes())}) + '\n')
    print(json.dumps({'status': 'BUILT', 'output': str(output), 'fetched_inputs': fetched, 'mode': 'test'}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=HERE / 'dist')
    p.add_argument('--label-only', action='store_true')
    p.add_argument('--api', help='Override the public API origin in generated configuration')
    p.add_argument('--analysis-root', type=Path, help='Benchmark workspace containing the sealed COMPLETE T45 analysis')
    p.add_argument('--capture-stage', type=Path, help='Update the public-input reference manifest after publishing a stage')
    p.add_argument('--base-url', help='Immutable public deployment URL used with --capture-stage')
    args = p.parse_args()
    if args.capture_stage:
        if not args.base_url:
            p.error('--capture-stage requires --base-url')
        capture(args.capture_stage.resolve(), args.base_url)
    else:
        build(args.output.resolve(), args.label_only, args.api, args.analysis_root.resolve() if args.analysis_root else None)


if __name__ == '__main__':
    main()
