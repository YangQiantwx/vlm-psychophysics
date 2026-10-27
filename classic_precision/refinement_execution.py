"""Finite append-only refinement sources, using the qualified RGB16 renderer."""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from .manifest import atomic_json, canonical, digest, file_sha, state_key, validate_manifest


def generate(manifest_path, root, workers=8):
    from .cpu import allocation
    from .ready import ReadyCatalog, append_ready, finish_generation
    from .stimuli import _encode_one, prepare_states
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    if manifest['counts'] != {'T2': 2016, 'T3': 1152} or manifest['models']:
        raise ValueError('Only the sealed 3168-source refinement is supported')
    target = root/'manifest.json'
    if target.exists() and target.read_bytes() != manifest_path.read_bytes():
        raise ValueError('Source manifest changed')
    if not target.exists():
        with target.open('xb') as f:
            f.write(manifest_path.read_bytes()); f.flush(); os.fsync(f.fileno())
    with (root/'.generation.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        catalog = ReadyCatalog(manifest, root/'ready.jsonl', root, file_sha(target), allow_failed=True)
        done = catalog.refresh()
        if catalog.complete:
            return json.loads((root/'generation.json').read_text())
        start = time.monotonic()
        def progress(status):
            atomic_json(root/'generation.json', {
                'status': status, 'protocol_id': manifest['protocol_id'],
                'manifest_sha256': file_sha(target), 'videos': len(done),
                'expected_videos': len(manifest['rows']), 'new_VLM_calls': 0,
                'this_session_monotonic_seconds': time.monotonic()-start})
        progress('RUNNING')
        bank, states = prepare_states(manifest, root/'state_cache')
        tasks = []
        for row in manifest['rows']:
            sid = row['sample_id']
            if sid in done:
                continue
            sidecar = root/'videos'/(sid+'.ready.json')
            if sidecar.exists():
                record = json.loads(sidecar.read_text())
                append_ready(manifest, root, record)
                done[sid] = record
            else:
                tasks.append((row, states['states'][state_key(row)], str(bank), str(root), 50*1024**3))
        try:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
                iterator, active, exhausted = iter(tasks), set(), False
                while active or not exhausted:
                    budget = allocation(workers, memory_gib=16, reserve_gib=16,
                        disk_free_gib=shutil.disk_usage(root).free/2**30, disk_floor_gib=50)
                    if budget['reason'] == 'DISK_RESERVE_REACHED':
                        raise RuntimeError('Source disk reserve reached')
                    while len(active) < budget['workers'] and not exhausted:
                        task = next(iterator, None)
                        if task is None:
                            exhausted = True
                        else:
                            active.add(pool.submit(_encode_one, task))
                    if not active and not exhausted:
                        time.sleep(10)
                        continue
                    finished, active = concurrent.futures.wait(active, timeout=10,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in finished:
                        record = future.result()
                        append_ready(manifest, root, record, verified=True)
                        done[record['sample_id']] = record
                    progress('RUNNING')
            # _encode_one has compared every decoded code and precise event frame
            # against the numerically checked source bank before publishing ready.
            result = finish_generation(manifest, root,
                unique_source_states=len(states['states']),
                this_session_monotonic_seconds=time.monotonic()-start,
                implementation_sha256=file_sha(__file__))
            result['unique_source_states'] = len(states['states'])
            atomic_json(root/'generation.json', result)
            catalog.refresh()
            return result
        except BaseException as error:
            progress('FAILED')
            atomic_json(root/'generation-error.json', {'error': str(error), 'type': type(error).__name__})
            raise


def audit(manifest_path, root, output, workers=8):
    """Independent fresh source-state measurements and full new-media audit."""
    import numpy as np
    from . import source_audit as sa
    from .stimuli import make_state
    from .verify import source_panel
    from .ready import ReadyCatalog
    root, output = Path(root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    catalog = ReadyCatalog(manifest, root/'ready.jsonl', root, file_sha(manifest_path))
    readies = catalog.refresh()
    if not catalog.complete:
        raise ValueError('Source generation is incomplete')
    config = {'panels': [{'key': 'classic-refinement', 'root': str(root),
        'manifest_sha256': file_sha(manifest_path), 'task_family': 'T2T3',
        'expected_counts': manifest['counts']}], 'expected_counts': manifest['counts']}
    items, _ = sa.inventory(config)
    identity = digest({'manifest': file_sha(manifest_path), 'code': sa.code_identity(),
                       'refinement_execution': file_sha(__file__)})
    with (output/'.audit.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cached = json.loads((root/'state_cache/states.json').read_text())
        bankpath = root/'state_cache/states.npy'
        bank = np.load(bankpath, mmap_mode='r')
        unique = {state_key(r): r for r in manifest['rows']}
        states, records = {}, []
        for key, row in unique.items():
            base, event, qa = make_state(row)
            location = cached['states'][key]
            for array, position in zip((base, event), location['indices']):
                if not np.array_equal(array, bank[position]):
                    raise ValueError('Freshly rebuilt source differs')
            states[key] = {**location, 'bank': str(bankpath), 'measurements': qa,
                           'source_parameters': row['source_parameters']}
            records.append({'source_parameters': row['source_parameters'], **qa})
        qa = {'identity': identity, 'status': 'PASS', 'states': states,
              'state_panel': source_panel(records), 'FP32_entry_codebook': sa.fp32_codebook(),
              't1': {'status': 'NOT_IN_THIS_EXTENSION'}, 'seconds': 0}
        atomic_json(output/'source-qa.json', qa)
        journal = output/'per-id.jsonl'
        records = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
        done = {r['sample_id']: r for r in records}
        if len(done) != len(records):
            raise ValueError('Repeated audit ID')
        lookup = {x['sample_id']: x for x in items}
        for sid, row in done.items():
            if (sid not in lookup or row['identity'] != identity or row['status'] != 'PASS'
                    or row['source_stat'] != sa.stat_identity(lookup[sid]['path'])):
                raise ValueError('Audit resume identity differs')
        start = time.monotonic()
        def progress():
            result = sa.summarize(records, len(items), config, qa, identity, time.monotonic()-start, 'full')
            atomic_json(output/'progress.json', result)
            return result
        progress()
        with journal.open('ab', buffering=0) as stream, concurrent.futures.ProcessPoolExecutor(
                max_workers=workers, initializer=_audit_initialize, initargs=(qa,)) as pool:
            iterator = iter(x for x in items if x['sample_id'] not in done)
            active, exhausted = set(), False
            while active or not exhausted:
                while len(active) < workers and not exhausted:
                    item = next(iterator, None)
                    if item is None:
                        exhausted = True
                    else:
                        active.add(pool.submit(sa.audit_video, item))
                finished, active = concurrent.futures.wait(active, timeout=10,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in finished:
                    row = future.result(); row['identity'] = identity
                    stream.write(canonical(row)+b'\n'); os.fsync(stream.fileno())
                    records.append(row)
                progress()
        result = progress()
        sa.write_csv(output, records)
        atomic_json(output/'summary.json', result)
        if result['status'] != 'COMPLETE':
            raise ValueError('New source audit failed')
        return result


def _audit_initialize(qa):
    from . import source_audit as sa
    sa._QA, sa._T1, sa._BANKS = qa, None, {}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('generate', 'audit', 'source'))
    p.add_argument('--manifest', required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--output')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if not 1 <= a.workers <= 8:
        p.error('Use 1..8 bounded CPU workers')
    if a.command == 'generate':
        result = generate(a.manifest, a.root, a.workers)
    else:
        if not a.output:
            p.error('audit requires --output')
        if a.command == 'source':
            generate(a.manifest, a.root, a.workers)
        result = audit(Path(a.root)/'manifest.json', a.root, a.output, a.workers)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
