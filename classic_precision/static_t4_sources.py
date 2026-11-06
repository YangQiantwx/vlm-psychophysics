"""T4 first/last RGB8 projection. CPU only; original 180-frame QA is reused."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import time

NAME = 'static-t4-first-last-20261104'
FROOT = str(Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve())
PARENT = 't4-region-balanced-20261104'
INDICES = [0, 179]
CONTRACT = {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}
REPLACEMENTS = [
    {'old': 'Watch the moving elements against the stationary background and follow their positions over time.',
     'new': 'Compare the two images, shown in chronological order as the first and last frames, and follow the elements relative to the stationary background.', 'expected_occurrences': 1},
    {'old': 'across the video', 'new': 'between the two images', 'expected_occurrences': 1},
    {'old': 'across successive moments', 'new': 'between the first and last frames', 'expected_occurrences': 1},
]


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    return sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode())


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def descriptor(path):
    return {'path': str(Path(path).absolute()), 'sha256': file_sha(path)}


def verified(ref):
    require(file_sha(ref['path']) == ref['sha256'], 'Changed bound file: ' + ref['path'])
    return read(ref['path'])


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()
    if path.exists():
        require(path.read_bytes() == data, 'Refuse overwrite of existing evidence: ' + str(path))
    else:
        with path.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
    return descriptor(path)


def logical_ref(path, root, logical_root):
    ref = descriptor(path)
    ref['path'] = str(Path(logical_root) / Path(path).relative_to(root))
    return ref


def protocol(original):
    prompt = original
    for change in REPLACEMENTS:
        require(prompt.count(change['old']) == change['expected_occurrences'], 'Prompt replacement cardinality differs')
        prompt = prompt.replace(change['old'], change['new'])
    return {'schema': 'static-t4-first-last-protocol-v1', 'status': 'CPU_PREP_ONLY', 'experiment': NAME,
            'domain': 'classic', 'task': 'T4', 'frame_indices': INDICES, 'images_per_question': 2,
            'video_contract': CONTRACT, 'image_contract': {'count': 2, 'width': 768, 'height': 512},
            'source_video_prompt': original, 'source_video_prompt_sha256': sha(original.encode()),
            'prompt': prompt, 'prompt_sha256': sha(prompt.encode()), 'minimal_replacements': REPLACEMENTS,
            'choices': ['LEFT', 'RIGHT'], 'source_aliases': 816, 'scored_aliases': 768, 'bias_only_aliases': 48,
            'maximum_new_model_calls': 3264, 'extra_qualification_calls': 0, 'automatic_retry': False,
            'interpretation': 'Ordered first/last-frame judgment; not single-image ability or an isolated interface effect',
            'deduplication': 'Ordered RGB8 first/last pixels plus exact UTF8 prompt and choices; GT and scoreability must agree',
            'display_preselection': {'layout_index': 0, 'N': [8, 16], 'coherence': 1.0, 'directions': ['LEFT', 'RIGHT'],
                                     'expected_source_videos': 4, 'expected_unique_PNGs': 6}}


def build_manifest(root, logical_root=FROOT):
    root = Path(root)
    ready_path = root / 'datasets' / PARENT / 'catalog-all.json'
    admission_path = root / 'plans' / PARENT / 'source-admission.json'
    ready, admission = read(ready_path), read(admission_path)
    require(ready['full_816'] is True and ready['count'] == len(ready['items']) == 816,
            'Exact complete original T4 source required')
    require(admission['status'] == 'PASS_T4_CLASSIC816_SOURCES'
            and admission['ready_catalog']['sha256'] == file_sha(ready_path), 'Original source admission differs')
    original = next(iter(ready['items'].values()))['row']['prompt']; spec = protocol(original)
    items, groups, selected = {}, defaultdict(list), []
    for old_id, source in sorted(ready['items'].items()):
        require(source['row']['prompt'] == original and source['content'][0]['type'] == 'video', 'Parent modality/prompt changed')
        qa_path = root / Path(source['source_qa_path']).relative_to(logical_root)
        qa = read(qa_path)
        require(qa['status'] == 'PASS_ALL180_RGB8_PTS_OPENCV' and qa['sample_id'] == old_id
                and len(qa['decoded_RGB8_frame_hashes']) == 180 and qa['max_PTS_error_seconds'] == 0
                and qa['video_sha256'] == source['content'][0]['sha256']
                and qa['stream_sha256'] == source['ready']['stream_sha256'], 'Original full RGB8/PTS evidence mismatch')
        pixels = [qa['decoded_RGB8_frame_hashes'][i] for i in INDICES]
        sid = old_id.replace('classic-t4b-', 'classic-st4-')
        row = copy.deepcopy(source['row']); row.update(sample_id=sid, prompt=spec['prompt'], domain='classic')
        row['source_parameters'] = copy.deepcopy(source['row']['source_parameters'])
        expected_truth = qa['row']['truth']; eligible = qa['row']['accuracy_eligible']
        require(row['truth'] == expected_truth and row['accuracy_eligible'] is eligible
                and ((eligible and expected_truth in ('LEFT', 'RIGHT')) or (not eligible and expected_truth is None)),
                'GT/scoreability changed')
        item = {'sample_id': sid, 'case_id': sid, 'task': 'T4', 'domain': 'classic', 'row': row,
                'original_source': {'path': source['content'][0]['path'], 'sha256': source['content'][0]['sha256']},
                'source_qa': logical_ref(qa_path, root, logical_root), 'source_frame_RGB8_sha256': pixels,
                'video_contract': CONTRACT, 'image_contract': spec['image_contract'], 'frame_indices': INDICES,
                'pair_key': row['source_parameters']['paired_block'],
                'original_video_aliases': [{'sample_id': old_id, 'domain': 'classic', 'task': 'T4',
                    'truth': row['truth'], 'accuracy_eligible': eligible, 'source_parameters': row['source_parameters'],
                    'source': {'path': source['content'][0]['path'], 'sha256': source['content'][0]['sha256']},
                    'source_qa': logical_ref(qa_path, root, logical_root)}]}
        key = digest([pixels, spec['prompt'], spec['choices']]); item['model_input_sha256'] = key
        items[sid] = item; groups[key].append(sid)
        params = row['source_parameters']
        if params['layout_index'] == 0 and params['coherence'] == 1.0:
            selected.append(sid)
    final, mapping, conflicts, collapsed = {}, {}, [], []
    for key, ids in sorted(groups.items()):
        labels = {(items[s]['row']['truth'], items[s]['row']['accuracy_eligible']) for s in ids}
        if len(labels) != 1:
            conflicts.append({'input_sha256': key, 'case_ids': ids, 'GT_and_scoreability': sorted(labels, key=str)})
            continue
        keep = min(ids); item = copy.deepcopy(items[keep]); item['preprojection_case_aliases'] = sorted(ids)
        item['original_video_aliases'] = [alias for s in sorted(ids) for alias in items[s]['original_video_aliases']]
        item['pair_keys'] = sorted({items[s]['pair_key'] for s in ids})
        final[keep] = item
        for s in ids:
            mapping[s] = keep
        if len(ids) > 1:
            collapsed.append({'retained_case_id': keep, 'case_ids': sorted(ids), 'model_input_sha256': key})
    require(not conflicts, 'HOLD: true ordered two-image input conflicts with GT/scoreability: ' + json.dumps(conflicts))
    selected = sorted(set(mapping[s] for s in selected))
    require(len(selected) == 4 and len({h for s in selected for h in final[s]['source_frame_RGB8_sha256']}) == 6,
            'Preselected four source examples/six image identities differ')
    return {'schema': 'static-t4-first-last-source-manifest-v1', 'status': 'CPU_METADATA_PASS_EXPORT_PENDING',
            'experiment': NAME, 'protocol': spec, 'source_ready_catalog': logical_ref(ready_path, root, logical_root),
            'source_admission': logical_ref(admission_path, root, logical_root),
            'implementation_sha256': file_sha(__file__), 'items': dict(sorted(final.items())),
            'source_alias_count': 816, 'scored_alias_count': 768, 'bias_only_alias_count': 48,
            'unique_inputs': len(final), 'unique_first_images': len({x['source_frame_RGB8_sha256'][0] for x in items.values()}),
            'unique_endpoint_images': len({h for x in items.values() for h in x['source_frame_RGB8_sha256']}),
            'mapping': mapping, 'duplicate_groups': collapsed, 'GT_scoreability_conflicts': conflicts,
            'preselected_case_ids': selected, 'new_model_call_cap': 4 * len(final), 'model_calls': 0}


def exact_endpoints(item):
    """Seek each exact source PTS; compare the reused dual-decoder frame SHA."""
    import av
    import numpy as np
    source = item['original_source']; path = Path(source['path'])
    require(path.is_file() and file_sha(path) == source['sha256'], 'Original video SHA changed')
    qa = verified(item['source_qa'])
    require(qa['status'] == 'PASS_ALL180_RGB8_PTS_OPENCV' and qa['video_sha256'] == source['sha256'], 'Unaccepted video source')
    arrays, receipts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.codec_context.thread_count = 1
        for index, expected in zip(INDICES, item['source_frame_RGB8_sha256']):
            target = Fraction(index, 60)
            container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
            found, decoded = None, 0
            for frame in container.decode(stream):
                decoded += 1; timestamp = Fraction(frame.pts) * frame.time_base
                if timestamp < target:
                    continue
                require(timestamp == target, 'Exact first/last source PTS missing')
                found = frame.to_ndarray(format='rgb24'); break
            require(found is not None and found.dtype == np.uint8 and found.shape == (512, 768, 3), 'Native RGB8 shape missing')
            require(sha(found.tobytes()) == expected == qa['decoded_RGB8_frame_hashes'][index], 'Endpoint RGB8 does not equal accepted full-video QA')
            arrays.append(found)
            receipts.append({'frame_index': index, 'PTS_seconds_fraction': str(target), 'decoded_to_reach_target': decoded,
                             'pixel_sha256': expected, 'matches_accepted_dual_decoder_QA': True})
    return arrays, receipts


def export_case(args):
    item, output_root = args; start = time.monotonic()
    from PIL import Image
    import numpy as np
    arrays, receipts = exact_endpoints(item); entries = []
    for pixels, receipt in zip(arrays, receipts):
        pixel_sha = receipt['pixel_sha256']; path = Path(output_root) / 'images' / pixel_sha[:2] / (pixel_sha + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            # Different cases can share the first frame; exclusive CAS placement is atomic.
            import io
            encoded = io.BytesIO(); Image.fromarray(pixels, 'RGB').save(encoded, format='PNG', compress_level=3)
            import tempfile
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.endpoint-', delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(encoded.getvalue()); stream.flush(); os.fsync(stream.fileno())
            try:
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    pass
            finally:
                temporary.unlink()
        with Image.open(path) as image:
            require(image.mode == 'RGB' and image.size == (768, 512) and not image.info
                    and sha(np.asarray(image).tobytes()) == pixel_sha, 'Exported PNG identity/geometry changed')
        entries.append({'type': 'image', **descriptor(path), 'pixel_sha256': pixel_sha,
                        'frame_index': receipt['frame_index'], 'source_PTS_seconds_fraction': receipt['PTS_seconds_fraction']})
    out = copy.deepcopy(item); out['content'] = entries + [{'type': 'text', 'text': item['row']['prompt']}]
    out['endpoint_export'] = {'status': 'PASS_EXACT_ENDPOINT_RGB8', 'source_qa': item['source_qa'],
                              'source_sha256': item['original_source']['sha256'], 'endpoints': receipts,
                              'resize_crop_calls': 0, 'elapsed_seconds': time.monotonic() - start}
    return out


def prepare(manifest_path, output_root, pilot=False, workers=2):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU export requires CUDA_VISIBLE_DEVICES empty')
    require(platform.system() == 'Linux', 'Formal endpoint export is execution-host Linux only, never Mac')
    require(1 <= workers <= 4, 'Finite CPU export concurrency must be 1..4')
    manifest = read(manifest_path); require(manifest['implementation_sha256'] == file_sha(__file__), 'Source code changed after manifest')
    out = Path(output_root).resolve(); out.mkdir(parents=True, exist_ok=True)
    require(out.stat().st_dev == Path(next(iter(manifest['items'].values()))['original_source']['path']).stat().st_dev,
            'Endpoint source/output must stay on the same admitted NVMe filesystem')
    ids = manifest['preselected_case_ids'][::2] if pilot else sorted(manifest['items'])
    require(not pilot or len(ids) == 2, 'Pilot must cover two preselected cases')
    if not pilot:
        previous = read(out / 'receipt-pilot.json')
        require(previous['manifest'] == descriptor(manifest_path) and previous['status'] == 'PASS_ENDPOINT_EXPORT'
                and previous['count'] == 2, 'Actual two-case CPU throughput pilot must precede full export')
    start = time.monotonic(); existing = {}
    if not pilot and (out / 'catalog-pilot.json').exists():
        existing = read(out / 'catalog-pilot.json')['items']
    todo = [s for s in ids if s not in existing]
    args = [(manifest['items'][s], str(out)) for s in todo]
    if workers == 1:
        done = [export_case(arg) for arg in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = list(pool.map(export_case, args))
    prepared = {s: existing[s] for s in ids if s in existing}; prepared.update({x['sample_id']: x for x in done})
    elapsed = time.monotonic() - start; phase = 'pilot' if pilot else 'all'
    receipt = {'schema': 'static-t4-endpoint-export-receipt-v1', 'status': 'PASS_ENDPOINT_EXPORT', 'selection': phase,
               'manifest': descriptor(manifest_path), 'count': len(prepared), 'source_alias_count': 816,
               'unique_inputs': manifest['unique_inputs'], 'newly_exported_cases': len(done), 'reused_pilot_cases': len(prepared)-len(done),
               'seconds': elapsed, 'cases_per_second': len(done) / elapsed, 'workers': workers,
               'estimated_all_seconds_at_measured_rate': manifest['unique_inputs'] * elapsed / max(1, len(done)),
               'source_full_QA_repeated': False, 'resize_crop_calls': 0, 'model_calls': 0, 'CUDA_visible': False,
               'output_realpath': str(out), 'output_st_dev': out.stat().st_dev,
               'source_implementation_sha256': file_sha(__file__),
               'cases': {s: prepared[s]['endpoint_export'] for s in sorted(prepared)}}
    rref = write_new(out / ('receipt-' + phase + '.json'), receipt)
    catalog = {k: copy.deepcopy(v) for k,v in manifest.items() if k != 'items'}
    catalog.update(schema='static-t4-ready-catalog-v1', status='PASS_ENDPOINT_EXPORT', selection=phase,
                   manifest=descriptor(manifest_path), source_prepare_receipt=rref, items=dict(sorted(prepared.items())))
    cref = write_new(out / ('catalog-' + phase + '.json'), catalog)
    return {'status': receipt['status'], 'catalog': cref, 'receipt': rref, 'count': len(prepared),
            'seconds': elapsed, 'estimated_all_seconds': receipt['estimated_all_seconds_at_measured_rate'], 'model_calls': 0}


def main():
    p = argparse.ArgumentParser(); p.add_argument('command', choices=['plan', 'prepare'])
    p.add_argument('--root'); p.add_argument('--logical-root', default=FROOT); p.add_argument('--output')
    p.add_argument('--manifest'); p.add_argument('--output-root'); p.add_argument('--workers', type=int, default=2)
    p.add_argument('--pilot', action='store_true'); p.add_argument('--all', action='store_true')
    a=p.parse_args()
    if a.command == 'plan':
        v=build_manifest(a.root,a.logical_root); ref=write_new(a.output,v)
        result={k:v[k] for k in ['status','unique_inputs','unique_first_images','unique_endpoint_images','source_alias_count','new_model_call_cap','preselected_case_ids','duplicate_groups','GT_scoreability_conflicts']};result['manifest']=ref
    else:
        require(a.pilot != a.all, 'Select exactly --pilot or --all')
        result=prepare(a.manifest,a.output_root,a.pilot,a.workers)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
