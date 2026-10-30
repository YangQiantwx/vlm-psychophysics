"""Stage current canonical natural sources, then publish and verify every HF file.

This release operation does not render stimuli, alter scientific records or run models.
The input release manifest is private; only selected, sanitized scientific fields are published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_blob(path: Path) -> str:
    h = hashlib.sha1(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def link(source: Path, dest: Path, expected: str | None = None) -> None:
    assert source.is_file() and not source.is_symlink(), source
    if expected:
        assert sha(source) == expected, f'Identity mismatch: {source}'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        assert not dest.is_symlink(), f'Staging destination cannot alias another path: {dest}'
        assert sha(dest) == sha(source)
        assert dest.stat().st_nlink == 1, f'Staging must be independent of active source inodes: {dest}'
    else:
        shutil.copyfile(source, dest)


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items()
                if k not in ('card', 'output_hint', 'original_generated_path', 'reused_from')}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        for prefix in ('/Users/qianyang/Desktop/cvpr2027_benchmark/',
                       '/home/yang/cvpr2027_benchmark/', '/mnt/sda/q/cvpr2027_benchmark/'):
            value = value.replace(prefix, '')
    return value


def stage(root: Path, manifest: dict, task: str) -> tuple[Path, dict]:
    d = next(d for d in manifest['datasets'] if d['task'] == task)
    source_rows = d.get('public_rows', d['rows'])
    output = root / 'runs/source-releases' / d.get('public_release_id', d['version'])
    payload = output / 'payload'
    payload.mkdir(parents=True, exist_ok=True)
    assert not any(p.is_symlink() or (p.is_file() and p.stat().st_nlink > 1)
                   for p in payload.rglob('*')), 'Existing payload must first be isolated from active sources'
    credits = []
    for asset in d['assets']:
        a = sanitize(asset.copy())
        a['workspace_input_paths'] = {k: a[k] for k in ('path', 'raw_path', 'mask_path') if a.get(k)}
        for field, sha_field in [('path', 'sha256'), ('raw_path', 'raw_sha256'), ('mask_path', 'mask_sha256')]:
            if not a.get(field):
                continue
            if field == 'raw_path' and a['license'] != 'GFDL 1.2':
                # The exact editable image used by the generator is already under
                # path/. Original public-photo download identity remains provenance;
                # duplicating multi-gigabyte panoramas is not required to rebuild it.
                a['recorded_original_source_sha256'] = a.get('raw_sha256')
                a[field] = None
                continue
            rel = Path(a[field])
            assert not rel.is_absolute() and '..' not in rel.parts
            source = root / 'assets/library' / rel
            if not source.exists() and field != 'path':
                a[field] = None
                continue
            dest_rel = Path('assets') / a['id'] / field / source.name
            # Primary stimulus inputs have authoritative SHA; historical raw metadata
            # may identify a different representation and is retained separately.
            link(source, payload / dest_rel, a.get(sha_field) if field != 'raw_path' else None)
            if field == 'raw_path' and a.get(sha_field):
                a['recorded_original_source_sha256'] = a[sha_field]
            a[field] = str(dest_rel)
            a[sha_field] = sha(source)
        credits.append(a)
    (payload / 'attribution.json').write_text(json.dumps(credits, ensure_ascii=False, indent=2) + '\n')
    reproduction = payload / 'reproduction'
    code_files = ['classic_precision/__init__.py', 'classic_precision/natural_t2_stimuli.py',
                  'classic_precision/natural_t13_stimuli.py', 'classic_precision/natural_spoon_revision.py',
                  'src/stimuli/natural_signal_reference.py', 'src/stimuli/prepare_next_tasks.py',
                  'src/stimuli/build_precise_protocol_examples.py', 'src/stimuli/t3_precision_reference.py']
    frozen = 'plans/natural-t2-qwen-x-20261029/cpu-generation-original.py'
    if d.get('public_code'):
        code_files = [r['relative_path'] for r in d['public_code']]
        code_files += ['src/stimuli/natural_signal_reference.py', 'src/stimuli/prepare_next_tasks.py',
                       'src/stimuli/build_precise_protocol_examples.py', 'src/stimuli/t3_precision_reference.py']
        code_files += d.get('public_runner_files', [])
    elif task == 'T2':
        code_files.append(frozen)
    code_identity = []
    for name in code_files:
        snapshot = root / 'runs/source-releases/code-snapshot-20261030' / name
        source = snapshot if snapshot.exists() else root / name
        bound = next((r for r in d.get('public_code', []) if r['relative_path'] == name), None)
        if bound:
            assert sha(source) == bound['sha256'], name
        link(source, reproduction / name)
        code_identity.append({'path': name, 'sha256': sha(source)})
    for bound in d.get('frozen_code', []):
        source = root / bound['relative_path']
        assert sha(source) == bound['sha256']
        dest = payload / 'evidence/frozen-execution' / bound['relative_path']
        link(source, dest, bound['sha256'])
        code_identity.append({'path': str(dest.relative_to(payload)), 'sha256': bound['sha256'],
                              'role': 'historical execution evidence; not the default public recipe'})
    plan_names = ['plans/natural-t2-qwen-x-20261029/allocation.json',
                  'plans/natural-t2-qwen-x-20261029/allocation-criteria.json',
                  'plans/natural-t2-qwen-x-20261029/asset-preparation.json',
                  'plans/natural-t13-20261029/allocation.json',
                  'plans/natural-t13-20261029/criteria.json',
                  'plans/natural-t13-20261029/catalog.json',
                  'plans/natural-t13-20261029/object-eligibility.json',
                  'plans/natural-t13-20261029/numerical-refinement-policy.json']
    for name in plan_names:
        snapshot = root / 'runs/source-releases/code-snapshot-20261030' / name
        source = snapshot if snapshot.exists() else root / name
        if source.exists():
            dest = reproduction / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(json.dumps(sanitize(json.loads(source.read_text())), ensure_ascii=False, indent=2) + '\n')
            code_identity.append({'path': name, 'original_sha256': sha(source),
                                  'published_sha256': sha(dest), 'sanitized_paths': True})
    for name in d.get('public_plan_files', []):
        source = root / name
        dest = reproduction / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Canonical path-free plans retain card selectors and their exact SHA.
        shutil.copyfile(source, dest)
        code_identity.append({'path': name, 'original_sha256': sha(source), 'published_sha256': sha(dest)})
    prepared_index = d.get('public_prepared_index')
    if prepared_index:
        source_index = root / prepared_index
        prepared = json.loads(source_index.read_text())
        for filename, record in prepared['items'].items():
            source = root / prepared['relative_directory'] / filename
            dest = reproduction / prepared['relative_directory'] / filename
            link(source.resolve(strict=True), dest, record['sha256'])
        assert len(prepared['items']) == 300
    for index_name in d.get('public_additional_input_indexes', []):
        for relative, record in json.loads((root / index_name).read_text())['items'].items():
            source = root / relative
            link(source.resolve(strict=True), reproduction / relative, record['sha256'])
    for destination, index_name in d.get('public_extra_indexes', {}).items():
        link(root / index_name, payload / destination)
    (reproduction / 'source-identities.json').write_text(json.dumps(code_identity, indent=2) + '\n')
    wanted_references = {r.get('reference_sha256') for r in source_rows}
    for reference in d.get('references', []):
        if reference['sha256'] not in wanted_references:
            continue
        source = root / reference['path'].split('cvpr2027_benchmark/', 1)[-1]
        link(source, payload / 'references' / f'{reference["sha256"]}.png', reference['sha256'])
    for name in d.get('scientific_records', []):
        source = root / name
        link(source, payload / 'audit' / 'independent' / source.name)
    asset_map = {a['id']: a for a in credits}
    published = []
    for row in source_rows:
        source = root / row['video_root_relative']
        rel = Path('videos') / f'{row["sample_id"]}.mp4'
        link(source, payload / rel, row['video_sha256'])
        record = {k: v for k, v in row.items()
                  if k not in ('video_root_relative', 'video_relative', 'prompt_file', 'card')}
        record['video_path'] = str(rel)
        record['source_assets'] = list(dict.fromkeys([row['background_id'], row['object_id']]))
        record['source_licenses'] = [asset_map[x]['license'] for x in record['source_assets']]
        record['source_bytes'] = source.stat().st_size
        record.update(frames=180, width=768, height=512, fps=60,
                      choices=row.get('choices', ['YES', 'NO'] if row['task'] == 'T5' else ['LEFT', 'RIGHT']),
                      source_version=d['version'], current1200_manifest_sha256=d.get('current1200_manifest_sha256'))
        if record.get('reference_sha256'):
            record['reference_path'] = f'references/{record["reference_sha256"]}.png'
        published.append(sanitize(record))
    if d.get('public_metadata_index_file'):
        exact_index = root / d['public_metadata_index_file']
        admitted = [json.loads(line) for line in exact_index.read_text().splitlines()]
        by_id = {r['sample_id']: r for r in published}
        assert len(admitted) == len(by_id) and {r['sample_id'] for r in admitted} == set(by_id)
        for row in admitted:
            original = by_id[row['sample_id']]
            for field in ('video_path','video_sha256','prompt','choices','truth','accuracy_eligible','frames','width','height','fps'):
                assert row[field] == original[field], (row['sample_id'], field)
        shutil.copyfile(exact_index, payload / 'metadata.jsonl')
        published = admitted
    else:
        (payload / 'metadata.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in published))
    for i, name in enumerate(d['source_audits']):
        source = root / name
        if not source.exists():
            raise FileNotFoundError(f'Required scientific audit missing: {source}')
        dest = payload / 'audit' / f'{i}-{source.name}'
        dest.parent.mkdir(exist_ok=True)
        doc = {'original_sha256': sha(source), 'original_workspace_path': name,
               'published_path_policy': 'Machine-specific prefixes and execution-card labels removed.',
               'record': sanitize(json.loads(source.read_text()))}
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + '\n')
    licenses = payload / 'LICENSES'
    licenses.mkdir(exist_ok=True)
    system_license = Path('/usr/share/common-licenses/GFDL-1.2')
    if system_license.exists():
        license_text = system_license.read_text()
    else:
        import requests
        response = requests.get('https://www.gnu.org/licenses/old-licenses/fdl-1.2.txt', timeout=60)
        response.raise_for_status()
        license_text = response.text
    assert 'GNU Free Documentation License' in license_text and 'Version 1.2' in license_text
    (licenses / 'GFDL-1.2.txt').write_text(license_text)
    (payload / 'LICENSE').write_text(
        'MIXED SOURCE LICENSES\n\nThis collection has no blanket CC0 or permissive media license. '
        'Consult attribution.json for each source, and metadata.jsonl for each video\'s source association. '
        'Original author attributions, license links and transformation descriptions are retained. '
        'Third-party rights remain with their respective authors.\n\n'
        'Videos derived from CC BY-SA materials retain the corresponding share-alike requirements. '
        'Videos derived from photo_obj_008 (Prathyush Thomas) or photo_obj_010 (Fir0002/Flagstaffotos) '
        'are distributed under GFDL version 1.2 only, with no invariant sections, front-cover or back-cover texts. '
        'The full license and editable/transparent source inputs are included. '
        'These videos do not use the Pexels background.\n\n'
        'Pexels and Unsplash materials retain their linked provider licenses. This release is a scientific '
        'stimulus dataset, not a stock-photo or wallpaper service. Generated assets are explicitly identified '
        'and are not represented as recorded photographs; no blanket license over third-party content is asserted.\n')
    model_note = ('A separate completed Qwen X evaluation scored 3756/3840; source release is not a new inference run.'
                  if task == 'T2' else ('This source release performs no inference; final-model results are released separately.'
                  if task == 'T45' else 'No model inference has been completed on this current natural T3 release.'))
    if task == 'T45':
        prompts = payload / 'prompts'
        prompts.mkdir(exist_ok=True)
        for task_name in sorted({r['task'] for r in published}):
            values = {r['prompt'] for r in published if r['task'] == task_name}
            assert len(values) == 1
            (prompts / f'{task_name}.txt').write_text(next(iter(values)).rstrip() + '\n')
    else:
        prompt = (published[0].get('prompt') or (root / 'plans/natural-t13-20261029/T3-prompt.txt').read_text())
        (payload / 'prompt.txt').write_text(prompt.rstrip() + '\n')
    replacement_note = ('This canonical index excludes the 350 superseded classic T4 videos and all superseded natural '
                        'envelope candidates. It retains source-identical classic clips by verified identity. '
                        'The complete parent release passed independent pixel audits on 216,000 source frames. '
                        if task == 'T45' else 'This canonical index includes every current replacement and excludes the superseded spoon sources. ')
    (payload / 'README.md').write_text(
        '---\nlicense: other\nlicense_name: mixed-source-licenses\nlicense_link: LICENSE\n'
        'task_categories:\n- video-classification\nlanguage:\n- en\n---\n\n' +
        (f'# {d.get("public_title", "Classic and natural T4/T5" if task == "T45" else "Natural " + task)} visual psychophysics · current source release\n\n') +
        f'{len(published):,} validated, unique source videos. Version: `{d["version"]}`. '
        'Each video is 768×512, 60 fps, 180 frames (3 s), encoded as lossless RGB H264. '
        'Browser rendering requires an independently verified derivative; this encoding claim is not a browser-pixel claim.\n\n' +
        replacement_note +
        'Old sources and original model answers remain in private scientific history. '
        f'{model_note}\n\n'
        '`metadata.jsonl` records the sample ID, exact encoded-video SHA256, source assets, conditions, '
        'ground truth and numerical source QA. Ground truth is for analysis and must be withheld from participant interfaces. '
        '`attribution.json`, `assets/` and `LICENSES/` retain source attribution and reproducible editable inputs. '
        'Digital luminance is not calibrated cd/m²; generated scenes are labeled explicitly.\n\n'
        'Reproduction code: [vlm-psychophysics](https://github.com/DrQY/vlm-psychophysics/tree/classic-precision), '
        'Maintained reproduction entrypoints are bundled under `reproduction/`; frozen execution evidence is kept separately. '
        'Exact generator identities and source SHA256 values are retained in metadata and audit records. '
        'No meetings, personal email, credentials, model weights, environments or old retired videos are included.\n')
    if task == 'T45':
        with (payload / 'README.md').open('a') as f:
            f.write('\nThis public revision includes all 816 T4 videos. ' +
                    ('It also includes all 192 unmasked T5 videos and their 25 reference images; '
                     'the 192 persistent-mask videos await clarification of the original mask redistribution permission. '
                     if any(r['task'] == 'T5' for r in published) else '') +
                    'Natural T4 uses a 56px enclosing support; '
                    'cutouts do not have equal visible area or contrast. The 300 prepared NPY inputs '
                    '(200 backgrounds and 100 T4 object arrays) are pinned individually by the prepared-file index. '
                    'These commands require no T5 mask or reference inputs:\n\n'
                    '```bash\npython reproduction/src/stimuli/prepare_t45_final.py render --root reproduction --task T4 --domain all --host fourier --workers 4\n'
                    'python reproduction/src/stimuli/prepare_t45_final.py render --root reproduction --task T4 --domain all --host radon --workers 4\n```\n')
            f.write('\nT4 is a direction-discrimination assay. A CPU-only metadata diagnostic predicts RIGHT when the final '
                    'centroid x exceeds 384 and obtains 360/384 correct labels on the classic nonzero-coherence sources. '
                    'This geometric rule is not a VLM score; accuracy alone does not establish cross-frame integration.\n')
            if any(r['task'] == 'T5' for r in published):
                f.write('\nUnmasked T5 replay (the additional-input index pins 24 T5 arrays and 25 reference PNGs):\n\n'
                        '```bash\npython reproduction/src/stimuli/prepare_t45_final.py render --root reproduction --task T5 --domain all --mask none --host fourier --workers 4\n'
                        'python reproduction/src/stimuli/prepare_t45_final.py render --root reproduction --task T5 --domain all --mask none --host radon --workers 4\n```\n'
                        '\nT5 tests reference-target detection across two sequential events. The first event uses left/right positions '
                        'and the second top/bottom positions: first/second contrasts mix spatial axis with recency and do not identify a pure temporal-order effect.\n')
            if d.get('public_runner_files'):
                catalog_sha = sha(payload / 'metadata.jsonl')
                f.write('\nThe bundled portable Cosmos X runner uses its adjacent qualified recipe. The catalog admission command '
                        'checks the published source set without importing vLLM or calling a model:\n\n'
                        f'```bash\npython reproduction/classic_precision/cosmos_x.py check-index --data-root . --catalog metadata.jsonl --catalog-sha256 {catalog_sha}\n```\n'
                        '\nThe original full-frame X engine/request settings and T5 reference-first message order are retained. '
                        'The public entrypoint contract was checked on CPU; this source publication does not claim a new GPU replay.\n')
    files = []
    for p in sorted(payload.rglob('*')):
        if p.is_file():
            files.append({'path': str(p.relative_to(payload)), 'bytes': p.stat().st_size,
                          'sha256': sha(p), 'git_blob_sha1': git_blob(p)})
    release = {'task': task, 'repository': d['hf_repository'], 'version': d['version'],
               'videos': len(published), 'source_manifest_sha256': sha(root / 'plans/video-library-20261030/release-manifest.json'),
               'files': files, 'model_calls': 0}
    (output / 'manifest.json').write_text(json.dumps(release, indent=2) + '\n')
    return payload, release


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--task', choices=['T2', 'T3', 'T45'], required=True)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()
    manifest = json.loads((args.root / 'plans/video-library-20261030/release-manifest.json').read_text())
    payload, release = stage(args.root, manifest, args.task)
    print(json.dumps({'status': 'STAGED_IDENTITIES_PASS', 'videos': release['videos'],
                      'files': len(release['files']), 'bytes': sum(r['bytes'] for r in release['files'])}), flush=True)
    if not args.execute:
        return
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(release['repository'], repo_type='dataset', private=False, exist_ok=True)
    commit = api.upload_folder(repo_id=release['repository'], repo_type='dataset', folder_path=payload,
                               commit_message=f'Publish verified canonical {args.task} source revision with per-source licenses')
    remote = {r.path: r for r in api.list_repo_tree(release['repository'], repo_type='dataset',
                                                  revision=commit.oid, recursive=True) if hasattr(r, 'size')}
    for row in release['files']:
        r = remote[row['path']]
        assert r.size == row['bytes'], row['path']
        if r.lfs:
            assert r.lfs.sha256 == row['sha256'], row['path']
        else:
            assert r.blob_id == row['git_blob_sha1'], row['path']
    result = {'status': 'PUBLISHED_AND_ALL_FILE_IDENTITIES_VERIFIED', 'repository': release['repository'],
              'commit': commit.oid, 'url': f'https://huggingface.co/datasets/{release["repository"]}',
              'videos': release['videos'], 'verified_files': len(release['files']), 'model_calls': 0}
    (payload.parent / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
