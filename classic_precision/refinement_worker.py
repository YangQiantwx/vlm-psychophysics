"""Source-bound single-attempt refinement workers; qualified sessions unchanged."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import uuid

from .storage import RunStore, atomic_json, digest, file_sha256, owner_identity, _append


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        json.dump(value, f, sort_keys=True, separators=(',', ':')); f.write('\n')
        f.flush(); os.fsync(f.fileno())


def descriptor(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_sha256(path)}


def verified(ref):
    require(file_sha256(ref['path']) == ref['sha256'], 'Bound file changed: '+ref['path'])
    return read(ref['path'])


class Boundary:
    def __init__(self, profile):
        self.profile, self.requested = profile, False
    def signal(self, *_):
        self.requested = True
    def check(self):
        p = Path(self.profile['pause_file'])
        if p.exists():
            require(read(p) == {'worker_token': self.profile['worker_token'], 'action': 'PAUSE'}, 'Foreign pause file')
            self.requested = True
        return self.requested


def claim(profile, sid):
    write_new(Path(profile['claim_root'])/(sid+'.json'), {
        'sample_id': sid, 'worker_token': profile['worker_token'],
        'profile_sha256': digest(profile), 'output': profile['output']})


@contextmanager
def physical_lease(profile, torch):
    from .inference import resource_snapshot
    require(torch.cuda.device_count() == 1, 'Exactly one physical GPU required')
    device = str(torch.cuda.get_device_properties(0).uuid)
    require(device.lower().removeprefix('gpu-') == profile['gpu_uuid'].lower().removeprefix('gpu-'), 'Wrong GPU')
    lock = profile['physical_lock']
    path = Path(lock['path'])
    require(path.is_file() and not path.is_symlink(), 'Original physical lease missing')
    with path.open('r+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stat = os.fstat(f.fileno())
        require((stat.st_dev, stat.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original lease inode changed')
        admission = resource_snapshot(torch, profile['resources'])
        require(admission['admitted'], 'Insufficient live GPU/RAM headroom')
        yield {**lock, 'device': device, 'admission': admission}
        require((path.stat().st_dev, path.stat().st_ino) == (stat.st_dev, stat.st_ino), 'Physical lease replaced')


class Sink:
    def __init__(self, store):
        self.store, self.passed = store, set()
    def event(self, event, **fields):
        if event == 'TECHNICAL_PASS':
            raw = self.store.outputs[fields['sample_id']]
            fields['raw_sha256'] = digest(raw)
            self.passed.add((fields['sample_id'], fields['attempt_id'], fields['raw_sha256']))
        self.store.event(event, **fields)


def generic_run(profile, boundary):
    from . import inference
    from . import gemma_speed as lifecycle
    model = profile['model']
    if model in ('qwen', 'nemo'):
        require(inference.checkpoint_asset_identity(profile['checkpoint']) == profile['checkpoint_assets'], 'Checkpoint identity changed')
    else:
        from .nvfp4_cosmos_official import runtime_identity
        current = runtime_identity(profile['runtime'])
        require(current == profile['runtime_identity'], 'Cosmos qualified runtime identity changed')
    qsa = None
    if model == 'qwen':
        from . import qsa_only_reproducibility as qsa
        qsa.runtime.activate('qwen', 'qwen_qsa_only', profile['qualified_profile']['baseline_environment'])
        qsa._validate_profile(qsa.runtime.status(), 'qwen', require_applied=False)
    import torch
    items = verified(profile['catalog'])['items']
    out = Path(profile['output'])
    require(not out.exists(), 'No automatic worker resume')
    identity = {'schema': profile['schema'], 'profile': profile, 'profile_sha256': digest(profile),
                'implementation': profile['implementation'], 'owner': owner_identity()}
    completed = []
    with RunStore(out, identity, model, profile['worker'], profile['assigned_ids']) as store:
        sink, session, owned = Sink(store), None, []
        with physical_lease(profile, torch) as lease:
            atomic_json(out/'lease.json', lease)
            store.event('LEASE_ACQUIRED', lease=lease)
            try:
                if model == 'cosmos':
                    from .nvfp4_cosmos_official import OfficialSession
                    session = OfficialSession(profile['runtime'], sink)
                else:
                    session = inference.ModelSession(model, profile['checkpoint'], sink, resources=profile['resources'])
                owned = session.engine.collective_rpc(lifecycle._worker_identity)
                atomic_json(out/'engine-owner.json', {'owned': owned})
                if qsa:
                    qsa.runtime.prepare_native('qwen')
                    snapshot = qsa.runtime.runtime_snapshot()
                    qsa.validate_runtime_policy(snapshot, 'qwen')
                    store.event('QUALIFIED_QSA_RUNTIME', snapshot=snapshot)
                for ordinal, sid in enumerate(profile['assigned_ids']):
                    if boundary.check():
                        break
                    require(file_sha256(__file__) == profile['implementation']['refinement_worker.py'], 'Worker source changed')
                    item = items[sid]; row, ready = item['row'], item['ready']
                    claim(profile, sid)
                    attempt = store.begin(sid, digest({'row': row, 'ready': ready}))
                    started = time.monotonic()
                    session.context['deep'] = ordinal < 2
                    def commit(record):
                        return store.commit_output(sid, {**record, 'profile_sha256': digest(profile),
                            'source_audit_identity': item['source_audit_identity'], 'source_stream_sha256': item['source_stream_sha256'],
                            'source_video_sha256': ready['video_sha256'], 'source_family': item['source_family'],
                            'source_row_sha256': item['row_sha256'], 'source_ready_sha256': item['ready_sha256'],
                            'overall_attempt_number': 1, 'deep': ordinal < 2,
                            'full_request_to_raw_seconds': time.monotonic()-started})
                    session.generate(row, ready, Path(profile['source_root']), commit)
                    raw = store.outputs[sid]
                    require((sid, raw['attempt_id'], digest(raw)) in sink.passed, 'Raw lacks bound technical pass')
                    store.event('ITERATION_COMPLETE', sample_id=sid, attempt_id=attempt,
                        raw_sha256=digest(raw), request_through_technical_pass_seconds=time.monotonic()-started,
                        scope='prepare through durable raw and technical-pass journal; excludes this timing record and progress write')
                    _append(out/'request-timings.jsonl', {'sample_id':sid,'deep':ordinal<2,
                        'full_request_seconds':time.monotonic()-started,
                        'scope':'prepare through raw and technical-pass persistence; excludes timing/progress writes'})
                    completed.append(sid)
                    if ordinal == 1:
                        atomic_json(out/'first-two.json', {'status': 'PASS', 'sample_ids': completed[:2],
                            'profile_sha256': digest(profile), 'extra_calls': 0})
                    atomic_json(out/'progress.json', {'status': 'RUNNING', 'completed': len(completed),
                        'expected': len(profile['assigned_ids']), 'last_sample_id': sid})
            except BaseException as error:
                store.event('FAILED_NO_RETRY', error_type=type(error).__name__, error=str(error))
                raise
            finally:
                if session is not None:
                    lifecycle.shutdown_owned_engine(session.engine, owned, out)
                    session.engine = None
                    session.close()
                store.event('OWNED_ENGINES_EXITED_BEFORE_LEASE_RELEASE')
        store.event('LEASE_RELEASED')
    result = {'status': 'PAUSED' if boundary.requested else 'COMPLETE', 'completed_ids': completed,
              'expected': len(profile['assigned_ids']), 'profile_sha256': digest(profile), 'automatic_retry': False}
    atomic_json(out/'result.json', result)
    return result


def gemma_run(profile, boundary):
    from . import gemma_nvfp4_continuation as c
    from . import gemma_production as p
    from . import gemma_speed as g
    catalog = verified(profile['catalog'])
    anchor = verified(profile['qualified_anchor'])
    reference = verified(anchor['references']['20'])

    class References:
        def __init__(self):
            self.current = {}
        def expected(self, sid, deep):
            return self.current.get(sid, self.ordinary(sid)) if deep else self.ordinary(sid)
        def ordinary(self, sid):
            item = catalog['items'][sid]
            return {**catalog['templates'][item['template_key']]['controls'],
                    'source_stream_sha256': item['source_stream_sha256']}
        def verify(self, receipt, sid, deep):
            g.verify_input(receipt, self.ordinary(sid), False)
            if deep:
                # Fresh first-two CPU prepare is compared to the actual GPU
                # bytes by the unchanged qualified observer, within this call.
                self.current[sid] = receipt

    class Worker(c.Worker):
        def __init__(self):
            self.profile, self.catalog = profile, catalog
            self.admission = read(profile['admission_file'])
            self.refs, self.boundary = References(), boundary
            self.inbox = c.AssignedItems(profile, catalog, self.refs, boundary)
            self.output, self.pending, self.proofs, self.service_times = None, None, [], []
        def reference_gate(self, *_):
            require(catalog['counts'] == {'T2': 2016, 'T3': 1152}, 'New source universe differs')
        def prepare(self, session, item, streamlined, deep):
            return g.prepare(session, item, {item['task']: profile['source_root']}, streamlined, deep)
        def intent(self, item):
            claim(profile, item['sample_id'])
            p.Worker.intent(self, item)
        def commit(self, result, started):
            p.Worker.commit(self, result, started)
            g.append(self.output/'request-timings.jsonl', {**self.service_times[-1],
                'scope':'prepare through raw, observer, technical pass and call-proof persistence; excludes timing write'})

    # AssignedItems asks for a reference before prepare. For new deep IDs the
    # ordinary geometry controls are sufficient until prepare records the hashes.
    c.sources(); c.runtime_check(profile['runtime'])
    require(g.runtime_sources() == profile['runtime_sources'], 'Qualified runtime sources changed')
    require(c.inference.checkpoint_asset_identity(profile['runtime']['checkpoint']) == profile['checkpoint_assets'], 'Gemma checkpoint assets changed')
    require(c.nv.effective_checkpoint_contract(profile['runtime']['checkpoint'], profile['runtime']['processor_checkpoint']) == profile['checkpoint_contract'], 'Gemma NVFP4 lineage changed')
    worker = Worker()
    args = argparse.Namespace(profile=profile['profile_file'], reference=anchor['references']['20']['path'],
        admission=profile['admission_file'], output=profile['output'])
    with c.overlay(profile), ExitStack() as stack:
        state = stack.enter_context(c.cpu.installed_overlay(profile))
        runner, derived = worker.compile(); runner.__globals__['ENTRY_SCHEMA'] = profile['schema']
        state.code, state.line = runner.__code__, derived['gc_line']
        result = runner(args)
    atomic_json(Path(profile['output'])/'continuation.json', {
        'status': worker.status(), 'completed_ids': [x['sample_id'] for x in worker.proofs],
        'profile_sha256': digest(profile), 'automatic_retry': False,
        'service_times': worker.service_times, 'uncertain_id': worker.pending})
    return result


def run(profile_path):
    profile = read(profile_path)
    require('torch' not in sys.modules and 'vllm' not in sys.modules, 'Fresh process required')
    require(socket.gethostname() == profile['hostname'], 'Host differs')
    require(profile['schema'] == 'classic-refinement-worker-v1', 'Worker schema differs')
    require(len(profile['assigned_ids']) == len(set(profile['assigned_ids'])) == profile['maximum_model_calls'], 'Finite scope differs')
    require(profile['automatic_retry'] is False, 'Automatic retry forbidden')
    for name, value in profile['environment'].items():
        require(os.environ.get(name) == value, 'Launch environment differs: '+name)
    for name in profile.get('cleared_environment', []):
        require(name not in os.environ, 'Unexpected inherited environment: '+name)
    for name, sha in profile['implementation'].items():
        path = Path(__file__).with_name(name) if '/' not in name else Path(__file__).parent.parent/name
        require(file_sha256(path) == sha, 'Qualified implementation changed: '+name)
    catalog = verified(profile['catalog'])
    qa = verified(profile['source_QA'])
    require(qa['status'] == 'COMPLETE' and not qa['failed_IDs']
        and qa['completed_counts'] == {'T2':2016,'T3':1152}, 'New source QA binding differs')
    alloc = verified(profile['allocation'])
    require(alloc['assignments'][profile['worker']] == profile['assigned_ids'], 'Frozen allocation changed')
    require(set(profile['assigned_ids']) <= set(catalog['items']), 'Unknown source ID')
    require(not any((Path(profile['claim_root'])/(sid+'.json')).exists() for sid in profile['assigned_ids']), 'An assigned ID is already claimed; reconcile without retry')
    boundary = Boundary(profile)
    handlers = {s: signal.signal(s, boundary.signal) for s in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM)}
    try:
        return gemma_run(profile, boundary) if profile['model'] == 'gemma' else generic_run(profile, boundary)
    finally:
        for s, handler in handlers.items():
            signal.signal(s, handler)


def supervise(profile_path):
    profile = read(profile_path)
    env = dict(os.environ)
    for k in profile.get('cleared_environment', []):
        env.pop(k, None)
    env.update(profile['environment'])
    output = Path(profile['output']); output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix('.log').open('xb') as log:
        started = time.monotonic()
        child = subprocess.Popen([sys.executable, '-B', '-m', 'classic_precision.refinement_worker',
            'run', '--profile', str(profile_path)], env=env, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL)
        write_new(output.with_suffix('.launch.json'), {'owner': owner_identity(), 'pid': child.pid,
            'profile_sha256': digest(profile), 'automatic_retry': False})
        code = child.wait()
    # Inspect only the owned identities returned by our engine, never other jobs.
    from .gemma_speed import owned_worker_alive
    engine_file = output/'engine-owner.json'
    exit_file = output/'exit.json'
    owned = read(engine_file)['owned'] if engine_file.exists() else []
    if exit_file.exists():
        owned = [{'worker': x} for x in read(exit_file).get('owned_workers', [])] or owned
    exited = all(not owned_worker_alive(x['worker']) for x in owned)
    lock = profile['physical_lock']; free = False
    with Path(lock['path']).open('r+b') as f:
        state = os.fstat(f.fileno())
        if (state.st_dev, state.st_ino) == (lock['st_dev'], lock['st_ino']):
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); free = True
            except BlockingIOError:
                pass
    result = {'returncode': code, 'real_wait_observed': True, 'owned_engines_exited': exited,
        'original_physical_lease_released': free, 'profile_sha256': digest(profile),
        'supervisor_monotonic_seconds': time.monotonic()-started, 'automatic_retry': False}
    terminal = output/('continuation.json' if profile['model']=='gemma' else 'result.json')
    completion = read(terminal) if terminal.exists() else {}
    result['terminal_status'] = completion.get('status', 'FAILED_OR_UNCERTAIN')
    result['completed_ids'] = completion.get('completed_ids', [])
    write_new(output.with_suffix('.wait.json'), result)
    require(code == 0 and exited and free, 'Worker failed or shutdown/lease release incomplete')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('run', 'supervise', 'pause'))
    p.add_argument('--profile', required=True)
    a = p.parse_args()
    if a.command == 'pause':
        q = read(a.profile)
        write_new(q['pause_file'], {'worker_token': q['worker_token'], 'action': 'PAUSE'})
        result = {'status': 'PAUSE_REQUESTED', 'profile': a.profile, 'drain_current_request': True}
    else:
        result = run(a.profile) if a.command == 'run' else supervise(a.profile)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
