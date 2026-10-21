"""One finite model queue per physical GPU; no LLM or exclusive-GPU assumption."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .manifest import atomic_json, validate_manifest, selected_rows
from .summary import model_snapshot, technical_passed
from .storage import MAX_ATTEMPTS, RunStore, digest


def capacity(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.free,memory.total,utilization.gpu',
                             '--format=csv,noheader,nounits'], check=True, text=True, capture_output=True)
    free, total, utilization = [float(v.strip()) for v in result.stdout.strip().split(',')]
    mem = {p[0].rstrip(':'): int(p[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
           if len(p := line.split()) >= 2}
    return {'free_gpu_gib': free/1024, 'total_gpu_gib': total/1024,
            'utilization_percent': utilization, 'available_ram_gib': mem['MemAvailable']/2**30}


def choose_model(pending, specs, snapshot):
    fitting = [name for name in pending
               if snapshot['free_gpu_gib'] >= specs[name]['resources']['min_free_gpu_gib']
               and snapshot['available_ram_gib'] >= specs[name]['resources']['min_available_ram_gib']]
    # On a shared card, start with the fitting queue with the smallest VRAM budget.
    # This is a scheduling choice, never an accuracy/model-selection rule.
    return min(fitting, key=lambda x: (specs[x]['resources']['min_free_gpu_gib'], pending.index(x))) if fitting else None


def progress(output_dir, family, selected_ids, expected_identity=None,
             expected_rows=None, ready=None, retry_failed=True, retry_interrupted=True):
    """Read one bound snapshot; returned raw counts never imply technical PASS."""
    directory = Path(output_dir)/family
    selected_ids = set(selected_ids)
    rows, events, raw_tail, event_tail = [], [], False, False
    identity_path = directory/'identity.json'
    if identity_path.exists():
        identity, rows, events, raw_tail, event_tail = model_snapshot(directory)
        if expected_identity is None:
            raise ValueError('An expected worker/SDK/checkpoint binding is required for supervised progress')
        wanted = {**expected_identity, 'storage_all_row_ids_sha256': expected_identity['all_row_ids_sha256']}
        if identity != wanted:
            raise ValueError('Supervised output differs from current manifest/model/runner/storage/SDK/checkpoint binding: '+family)
    elif any(p.exists() and p.stat().st_size for p in (directory/'outputs.jsonl', directory/'events.jsonl')):
        raise ValueError('Durable journals exist without a frozen run identity: '+family)
    by_id = {}
    for row in rows:
        sid = row['sample_id']
        if sid in by_id:
            raise ValueError('Duplicate durable output in supervised model journal')
        if expected_rows is not None:
            expected = expected_rows.get(sid)
            if expected is None or any(row.get(key) != expected[key] for key in ('task', 'truth', 'source_parameters')):
                raise ValueError('Supervised raw output source/GT differs from the frozen scientific row')
            if ready is not None and row.get('row_identity') != digest({'row': expected, 'ready': ready[sid]}):
                raise ValueError('Supervised raw output video/prompt identity differs from its frozen ready row')
            from .inference import GENERATION
            if any(row.get('effective_generation', {}).get(key) != value for key, value in GENERATION.items()):
                raise ValueError('Supervised raw output effective generation differs from the fixed protocol')
        by_id[sid] = row
    attempts, latest = {}, {}
    for event in events:
        sid = event.get('sample_id')
        if event.get('event') == 'INFLIGHT':
            attempts.setdefault(sid, []).append(event['attempt_id'])
        if event.get('event') in ('INFLIGHT', 'FAILED', 'INTERRUPTED'):
            latest[sid] = event['event']
    if any(len(ids) > MAX_ATTEMPTS for ids in attempts.values()):
        raise ValueError('Persisted model attempts exceed the fixed initial-plus-one-retry budget')
    for row in rows:
        if row['attempt_id'] not in attempts.get(row['sample_id'], []):
            raise ValueError('Durable output has no corresponding persisted INFLIGHT attempt')
    completed = selected_ids & set(by_id)
    passed = completed & technical_passed(rows, events)
    missing = selected_ids - completed
    counts = {sid: len(attempts.get(sid, [])) for sid in selected_ids}
    exhausted = {sid for sid in missing if counts[sid] >= MAX_ATTEMPTS}
    pending = {sid for sid in missing - exhausted
               if (retry_failed or latest.get(sid) != 'FAILED')
               and (retry_interrupted or latest.get(sid) not in ('INFLIGHT', 'INTERRUPTED'))}
    retryable = {sid for sid in pending if 0 < counts[sid] < MAX_ATTEMPTS}
    unresolved = completed - passed
    if expected_rows is not None:
        from .inference import qualification_order
        selected_rows_ordered = [row for row in expected_rows.values() if row['sample_id'] in selected_ids]
        _, qualification = qualification_order(selected_rows_ordered)
        for row in qualification:
            raw = by_id.get(row['sample_id'])
            if raw is not None and (raw.get('finish_reason') != 'stop' or raw.get('raw_answer') not in ('LEFT', 'RIGHT')):
                unresolved.add(row['sample_id'])
    needs_recovery = bool(unresolved or raw_tail or event_tail)
    complete = len(completed) == len(selected_ids) and not needs_recovery
    return {'expected': len(selected_ids), 'completed': len(completed), 'technical_pass': len(passed),
            'complete': complete, 'technical_unresolved_ids': sorted(unresolved),
            'pending_ids': sorted(pending), 'held_ids': sorted(missing-pending),
            'retryable_ids': sorted(retryable), 'exhausted_ids': sorted(exhausted),
            'attempt_counts': counts, 'attempt_ids': {sid: attempts.get(sid, []) for sid in selected_ids},
            'needs_recovery': needs_recovery, 'raw_tail': raw_tail, 'event_tail': event_tail}


def _reconcile_no_model(output_dir, family, identity, selected, all_rows, shard_id,
                        retry_failed=True, retry_interrupted=True):
    """Run the worker's durable recovery and receipt guards without a GPU load."""
    from .inference import reconcile_technical_receipts, qualification_order, _qualification_marker
    with RunStore(output_dir, identity, family, shard_id, [row['sample_id'] for row in selected],
                  all_row_ids=[row['sample_id'] for row in all_rows]) as store:
        store.pending_ids(retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        unresolved = reconcile_technical_receipts(store, family)
        if unresolved:
            raise RuntimeError('Durable raw outputs have unresolved technical evidence; no new model calls: '+str(unresolved))
        _, qualification = qualification_order(selected)
        if len(qualification) != 2:
            raise ValueError('A production shard must include both T2 and T3 qualification rows')
        _qualification_marker(store, identity, family, shard_id, qualification)
        return store.summary(retry_failed=retry_failed, retry_interrupted=retry_interrupted)


def supervise(manifest_path, output_dir, model_root, gpu, shard_index, shard_count=3,
              ready_path=None, asset_root=None, cosmos_package=None,
              worker_python=None, wait_seconds=60, retry_failed=True, retry_interrupted=True):
    """Exactly one subprocess at a time; error exits, resource wait does not spin GPUs.

    Run this command independently for shards 0/1/2 on three GPUs. No wall deadline,
    durable raw answers are terminal. An unfinished request may have only one
    retry, counted from persisted INFLIGHT events across process restarts.
    """
    import fcntl
    if wait_seconds < 10:
        raise ValueError('Use a bounded, non-busy resource poll interval >=10 seconds')
    manifest_path, output_dir = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    from .inference import build_run_identity, load_run_inputs
    manifest, _, ready, _, ready_path = load_run_inputs(manifest_path, ready_path, asset_root)
    selected = selected_rows(manifest, shard_index, shard_count)
    ids = {r['sample_id'] for r in selected}
    expected_rows = {row['sample_id']: row for row in manifest['rows']}
    models = list(manifest['models'])
    bindings = {model: build_run_identity(manifest_path, ready_path, manifest, model, model_root,
                                         cosmos_package=cosmos_package)[0] for model in models}
    if worker_python:
        probe = subprocess.run([worker_python, '-B', '-c',
                    'import json; from classic_precision.inference import _versions; print(json.dumps(_versions()))'],
                    check=True, text=True, capture_output=True)
        worker_versions = json.loads(probe.stdout)
        for identity in bindings.values():
            identity['runtime_versions'] = worker_versions
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = (output_dir/f'.supervisor-gpu-{gpu}.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output_dir/f'supervisor-shard-{shard_index}.json'
    shard_id = f'onset-{shard_index}-of-{shard_count}'
    def state(model):
        return progress(output_dir, model, ids, bindings[model], expected_rows, ready,
                        retry_failed=retry_failed, retry_interrupted=retry_interrupted)
    def recover(model):
        try:
            return _reconcile_no_model(output_dir, model, bindings[model], selected, manifest['rows'],
                                       shard_id, retry_failed, retry_interrupted)
        except BaseException as error:
            atomic_json(status_path, {'status': 'HOLD_TECHNICAL', 'model': model,
                        'error_type': type(error).__name__, 'error': str(error),
                        'time_unix': time.time(), 'supervisor_pid': os.getpid()})
            raise
    try:
        while True:
            pending = []
            counts = {}
            for model in models:
                current = state(model)
                if current['needs_recovery'] or current['exhausted_ids']:
                    recover(model)
                    current = state(model)
                counts[model] = current['completed']
                if current['exhausted_ids'] or current['held_ids']:
                    atomic_json(status_path, {'status': 'HOLD_ATTEMPTS', 'model': model,
                                'exhausted_ids': current['exhausted_ids'], 'held_ids': current['held_ids'],
                                'time_unix': time.time(), 'supervisor_pid': os.getpid()})
                    raise RuntimeError('Unfinished IDs are exhausted or retry is explicitly disabled: '+model)
                if not current['complete']:
                    pending.append(model)
            status = {'shard_index': shard_index, 'shard_count': shard_count, 'gpu': str(gpu),
                      'expected_per_model': len(ids), 'completed': counts, 'time_unix': time.time(),
                      'supervisor_pid': os.getpid()}
            if not pending:
                atomic_json(status_path, {**status, 'status': 'COMPLETE'})
                return status
            snapshot = capacity(gpu)
            model = choose_model(pending, manifest['models'], snapshot)
            atomic_json(status_path, {**status, 'capacity': snapshot,
                        'status': 'STARTING' if model else 'WAITING_FOR_FITTING_MODEL', 'model': model})
            if model is None:
                time.sleep(wait_seconds)
                continue
            before = state(model)
            command = [worker_python or sys.executable, '-B', '-m', 'classic_precision', 'run',
                       '--manifest', str(manifest_path), '--output', str(output_dir), '--model', model,
                       '--model-root', str(Path(model_root).resolve()), '--gpu', str(gpu),
                       '--shard-index', str(shard_index), '--shard-count', str(shard_count)]
            command.extend(['--retry-failed' if retry_failed else '--no-retry-failed',
                            '--retry-interrupted' if retry_interrupted else '--no-retry-interrupted'])
            for name, value in [('--ready', ready_path), ('--assets', asset_root), ('--cosmos-package', cosmos_package)]:
                if value:
                    command.extend([name, str(Path(value).resolve())])
            console = output_dir/f'console-{model}-shard-{shard_index}.log'
            with console.open('ab', buffering=0) as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=False)
            after = state(model)
            if after['needs_recovery'] or after['exhausted_ids']:
                recover(model)
                after = state(model)
            if after['complete']:
                continue
            new_retryable = [sid for sid in after['retryable_ids']
                             if len(after['attempt_ids'][sid]) > len(before['attempt_ids'][sid])]
            if new_retryable:
                atomic_json(status_path, {**status, 'status': 'RETRY_ONCE', 'model': model,
                            'retry_ids': new_retryable, 'worker_returncode': result.returncode,
                            'max_attempts': MAX_ATTEMPTS, 'console': str(console)})
                continue
            atomic_json(status_path, {**status, 'status': 'HOLD', 'model': model,
                        'worker_returncode': result.returncode, 'console': str(console),
                        'exhausted_ids': after['exhausted_ids'], 'held_ids': after['held_ids'],
                        'technical_unresolved_ids': after['technical_unresolved_ids']})
            raise RuntimeError('Worker has no permitted new retry; preserve its real failure and stop: '+str(console))
    finally:
        lock.close()


def download_models(model_root, families=None):
    """Pinned snapshots only. Existing snapshots are reused by huggingface_hub."""
    from huggingface_hub import snapshot_download
    from .manifest import load_protocol
    protocol = load_protocol()
    selected = families or list(protocol['models'])
    paths = {}
    for family in selected:
        spec = protocol['models'][family]
        path = Path(model_root)/spec['checkpoint_relative']
        snapshot_download(repo_id=spec['repo'], revision=spec['revision'], local_dir=path)
        paths[family] = str(path)
    return paths
