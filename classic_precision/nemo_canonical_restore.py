"""Finite, explicitly authorized restoration of Nemo's legacy runtime.

Runs the existing scientific workers unchanged. Original successful answers,
claims, pauses, and queues remain immutable runtime variants. This entry only
selects the public legacy KernelConfig and gates the actual selected kernels.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

from .nemo_kernel_replay import CONTRACT, inventory, validate_inventory
from .refinement_worker import read, require, write_new, descriptor
from .storage import atomic_json, digest, file_sha256

NAME = 'reader-route-smoke-20261026'
AUTHORIZATION = '三卡都用上，赶紧reproduce，完了以后给我建议，比如要不要完全按照官方load，然后no resize'
ROOTS = ('/Users/qianyang/Desktop/cvpr2027_benchmark',
         '/home/yang/cvpr2027_benchmark', '/mnt/sda/q/cvpr2027_benchmark')
MUTABLE = {'worker', 'worker_token', 'output', 'pause_file', 'claim_root',
           'implementation', 'maximum_model_calls', 'assigned_ids', 'restoration',
           'environment', 'cleared_environment', 'hostname', 'gpu_uuid',
           'physical_lock', 'resources', 'allocation', 'assignment_sha256',
           'profile_file', 'admission_file', 'catalog', 'source_QA'}


def relocate(value, root):
    if isinstance(value, dict): return {k: relocate(v, root) for k, v in value.items()}
    if isinstance(value, list): return [relocate(v, root) for v in value]
    if isinstance(value, str):
        for old in ROOTS:
            if value.startswith(old): return str(root) + value[len(old):]
    return value


def load_bound(root, ref):
    path = relocate(ref['path'], root)
    require(file_sha256(path) == ref['sha256'], 'Preserved evidence changed: ' + path)
    return read(path)


def records(path):
    return [json.loads(x) for x in Path(path).open()]


def split_formal(ids):
    require(len(ids) == len(set(ids)) == 836, 'Exactly the frozen 836 formal IDs required')
    # Sealed order only; no answer-dependent allocation. The two cards are equal.
    return {'fourier0': ids[::2], 'fourier1': ids[1::2]}


def preserved(root, block):
    parent = load_bound(root, block['parent_profile'])
    out = Path(relocate(parent['output'], root))
    raw_path = out/'raw.jsonl' if block['kind'] == 'smoke' else out/'nemo/outputs.jsonl'
    expected_sha = block.get('old_raw_sha256', block.get('raw', {}).get('sha256'))
    require(file_sha256(raw_path) == expected_sha, 'Original raw changed')
    rows = records(raw_path)
    require(len(rows) == len(block['sample_ids']) and
            {r['sample_id']: digest(r) for r in rows} == block['previous_raw_digest'],
            'Original per-ID raw identity differs')
    w = read(out.with_suffix('.wait.json'))
    require(w['returncode'] == 0 and w['real_wait_observed'] and w['owned_engines_exited']
            and w['original_physical_lease_released'] and w['profile_sha256'] == digest(parent)
            and w['completed_ids'] == block['sample_ids'], 'Original execution not closed')
    claims = Path(relocate(parent['claim_root'], root))
    refs = {sid: descriptor(claims/(sid+'.json')) for sid in block['sample_ids']}
    if block['kind'] == 'formal':
        require({sid: ref['sha256'] for sid, ref in refs.items()} ==
                {sid: ref['sha256'] for sid, ref in block['preserved_claims'].items()},
                'Original formal claim changed')
        require(read(relocate(parent['pause_file'], root)) ==
                {'worker_token': parent['worker_token'], 'action': 'PAUSE'}, 'Original PAUSE changed')
        for key in ('actual_wait', 'pause_audit'):
            load_bound(root, block[key])
    return parent, rows, refs


def validate_profile(p, root):
    r = p['restoration']; plan = Path(root)/'plans'/NAME
    auth = load_bound(root, r['authorization'])
    require(auth['human_message'] == AUTHORIZATION and auth['maximum_additional_calls'] == 1268
            and not auth['automatic_retry'], 'Current finite repair authorization missing')
    intent = load_bound(root, auth['intent'])
    block = intent['blocks'][r['block_index']]
    parent, _, refs = preserved(root, block)
    require({s: ref['sha256'] for s, ref in refs.items()} == r['preserved_claim_sha256'],
            'Original claims changed after sealing')
    expected = relocate(parent, root)
    require({k: v for k, v in p.items() if k not in MUTABLE} ==
            {k: v for k, v in expected.items() if k not in MUTABLE},
            'Scientific fields changed beyond legacy backend restoration')
    allocation = load_bound(root, r['sealed_allocation'])
    require(p['assigned_ids'] == allocation['assignments'][p['worker']]
            and p['maximum_model_calls'] == len(p['assigned_ids'])
            and set(p['assigned_ids']) <= set(block['sample_ids']), 'Finite restored ID scope differs')
    require(p['model'] == 'nemo' and p['output'] != relocate(parent['output'], root)
            and p['claim_root'] != relocate(parent['claim_root'], root), 'Original journals cannot be reused')
    launch = load_bound(root, r['qualified_launch'])
    env = dict(launch['environment']); env['CUDA_VISIBLE_DEVICES'] = r['gpu_index']
    require(p['environment'] == env and p['cleared_environment'] == launch['cleared_environment'],
            'Qualified complete launch environment changed')
    for key, value in {'CLASSIC_PRECISION_KV_COMPAT': '1',
                       'CLASSIC_PRECISION_NEMO_DUMMY_CONFIG_COMPAT': '1',
                       'CLASSIC_PRECISION_DEFER_MODELS': 'gemma'}.items():
        require(env.get(key) == value, 'Missing qualified Nemo launch switch: ' + key)
    require(p['checkpoint_assets'] == launch['checkpoint_assets'], 'Checkpoint/processor identity changed')
    for name, sha in r['execution_source_sha256'].items():
        require(file_sha256(Path(__file__).with_name(name)) == sha, 'Sealed entry changed: ' + name)
    require(p['hostname'] == socket.gethostname(), 'Wrong execution host')
    require(not Path(p['output']).exists(), 'A restoration attempt cannot be resumed automatically')
    return {'status': 'PASS', 'worker': p['worker'], 'maximum_new_calls': len(p['assigned_ids']), 'model_calls': 0}


def prepare(root, host):
    root = Path(root); plan = root/'plans'/NAME
    auth_path = plan/'nemo-restore1-authorization.json'; auth = read(auth_path)
    require(auth['human_message'] == AUTHORIZATION and auth['maximum_additional_calls'] == 1268,
            'Explicit updated finite budget required')
    intent = load_bound(root, auth['intent'])
    require(intent['new_scientific_calls_requiring_budget_extension'] == 1268, 'Intent changed')
    formal = next(b for b in intent['blocks'] if b['kind'] == 'formal')
    blocks = [(i, b) for i, b in enumerate(intent['blocks'])
              if (host == 'radon') == (b['kind'] == 'smoke')]
    require(host in ('fourier', 'radon'), 'Unknown host')
    launch_path = root/'runs/reproducibility/nemo-reproduction-diagnosis-20261027'/(
        'nemo-fourier0-cutlass-replay-profile.json' if host == 'fourier' else 'nemo-radon0-cutlass-replay-profile.json')
    launch = read(launch_path)
    assignment = {}; prepared = []; originals = {}
    for index, block in blocks:
        parent, oldrows, claims = preserved(root, block)
        cards = split_formal(formal['sample_ids']) if host == 'fourier' else {'radon0': block['sample_ids']}
        for card, ids in cards.items():
            kind = 'formal' if host == 'fourier' else block['arm']
            worker = f'nemo-{card}-{kind}-restore1'; p = relocate(deepcopy(parent), root)
            out_root = root/'runs/reproducibility'/('nemo-canonical-restoration-20261027' if host == 'fourier' else NAME)
            anchor = read(root/'plans/classic-refinement'/(
                'qwen-fourier0-profile.json' if card == 'fourier0' else
                'gemma-fourier1-profile.json' if card == 'fourier1' else 'nemo-radon0-v2-profile.json'))
            lock = anchor['physical_lock']
            with Path(lock['path']).open('r+b') as f:
                st = os.fstat(f.fileno())
                require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original lease replaced')
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            environment = dict(launch['environment']); environment['CUDA_VISIBLE_DEVICES'] = card[-1]
            p.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids),
                     worker_token=digest({'authorization': descriptor(auth_path), 'worker': worker, 'ids': ids}),
                     output=str(out_root/worker), pause_file=str(plan/(worker+'-PAUSE.json')),
                     claim_root=str(out_root/'claims-restore1'/('formal' if host == 'fourier' else block['arm'])),
                     environment=environment, cleared_environment=launch['cleared_environment'],
                     hostname=anchor['hostname'], gpu_uuid=anchor['gpu_uuid'], physical_lock=lock)
            if host == 'fourier':
                p['profile_file'] = str(plan/(worker+'-profile.json'))
                p['admission_file'] = str(plan/(worker+'-admission.json'))
                p['assignment_sha256'] = digest(ids)
                for field in ('catalog', 'source_QA'):
                    p[field] = descriptor(p[field]['path'])
                catalog = read(p['catalog']['path'])['items']
                by_id = {r['sample_id']: r for r in oldrows}
                for sid in ids:
                    item = catalog[sid]; old = by_id[sid]
                    require(item['ready']['video_sha256'] == old['source_video_sha256']
                            and item['row']['truth'] == old['truth']
                            and item['row_sha256'] == old['source_row_sha256'], 'Host source identity differs')
            else:
                p['implementation'] = descriptor(Path(__file__).with_name('reader_route_smoke.py'))
            p['restoration'] = {'authorization': descriptor(auth_path), 'block_index': index,
                'public_kernel_config': CONTRACT, 'gpu_index': card[-1],
                'qualified_launch': descriptor(launch_path),
                'preserved_claim_sha256': {sid: ref['sha256'] for sid, ref in claims.items()},
                'execution_source_sha256': {n: file_sha256(Path(__file__).with_name(n)) for n in
                    ('nemo_canonical_restore.py', 'nemo_kernel_replay.py', 'reader_route_smoke.py', 'refinement_worker.py')}}
            assignment[worker] = ids; prepared.append((plan/(worker+'-profile.json'), p))
            originals[worker] = {'parent_profile': block['parent_profile'], 'kind': block['kind'],
                                 'logical_worker': parent['worker'], 'arm': block.get('arm')}
    allocation_path = plan/f'nemo-restore1-{host}-allocation.json'
    write_new(allocation_path, {'assignments': assignment, 'authorization': descriptor(auth_path), 'automatic_retry': False})
    for path, p in prepared:
        p['restoration']['sealed_allocation'] = descriptor(allocation_path)
        if host == 'fourier': p['allocation'] = descriptor(allocation_path)
        validate_profile(p, root); write_new(path, p)
    queues = {}
    for card in ('fourier0', 'fourier1') if host == 'fourier' else ('radon0',):
        path = plan/(card+'-nemo-restore1-queue.json')
        profiles = [str(path_) for path_, p in prepared if f'-{card}-' in p['worker']]
        write_new(path, {'status': 'SEALED', 'profiles': profiles, 'automatic_retry': False,
                         'authorization': descriptor(auth_path), 'formal_auto_resume': False})
        queues[card] = str(path)
    manifest = {'schema': 'nemo-canonical-restoration-manifest-v1', 'authorization': descriptor(auth_path),
                'host': host, 'profiles': {p['worker']: {'profile': descriptor(path), **originals[p['worker']]}
                for path, p in prepared}, 'queues': queues, 'maximum_new_calls': sum(map(len, assignment.values())),
                'retained_original_answers': True, 'automatic_retry': False}
    write_new(plan/f'nemo-restore1-{host}-manifest.json', manifest)
    return {'status': 'PREPARED_NOT_LAUNCHED', 'queues': queues, 'maximum_new_calls': manifest['maximum_new_calls']}


@contextmanager
def legacy_session(output):
    # Importing inference is CPU-only. Delay vLLM until the original worker has
    # verified its fresh-process, source, allocation, and environment guards.
    from . import inference, gemma_speed as g
    original_init = inference.ModelSession.__init__
    def initialize(session, name, *args, **kwargs):
        require(name == 'nemo', 'Only Nemo may use the restoration context')
        import vllm
        original_llm = vllm.LLM
        def create(**settings):
            require('kernel_config' not in settings, 'Do not override another explicit kernel contract')
            engine = original_llm(**settings, kernel_config=deepcopy(CONTRACT))
            owned = engine.collective_rpc(g._worker_identity)
            try:
                selected = engine.collective_rpc(inventory)
                atomic_json(Path(output)/'restored-kernels.json', selected)
                validate_inventory(selected)
            except BaseException:
                g.shutdown_owned_engine(engine, owned, Path(output)); raise
            return engine
        vllm.LLM = create
        try:
            original_init(session, name, *args, **kwargs)
            from .reader_route_smoke import generation_receipt
            atomic_json(Path(output)/'restored-generation.json', generation_receipt(session.sampling))
        finally:
            vllm.LLM = original_llm
    inference.ModelSession.__init__ = initialize
    try: yield
    finally: inference.ModelSession.__init__ = original_init


def run(path):
    p = read(path); root = Path(path).parents[2]
    validate_profile(p, root)
    from .refinement_dispatch import compiler_preflight
    compiler_preflight(p['environment'])
    from . import reader_route_smoke, refinement_worker
    with legacy_session(p['output']):
        return (reader_route_smoke if 'arm' in p else refinement_worker).run(path)


def supervise(path):
    from . import refinement_worker as worker
    original = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([s.replace('classic_precision.refinement_worker',
                                  'classic_precision.nemo_canonical_restore') for s in argv], **kwargs)
    worker.subprocess.Popen = popen
    try: return worker.supervise(path)
    finally: worker.subprocess.Popen = original


def queue(path):
    q = read(path); require(q['status'] == 'SEALED', 'No automatic queue replay')
    q.update(status='RUNNING'); atomic_json(path, q); reports = []
    try:
        for pp in q['profiles']:
            p = read(pp); require(not Path(p['pause_file']).exists(), 'Queued restoration paused')
            child = subprocess.Popen([sys.executable, '-B', '-m', 'classic_precision.nemo_canonical_restore',
                                      'supervise', '--profile', pp])
            code = child.wait(); w = read(Path(p['output']).with_suffix('.wait.json')); reports.append(w)
            require(code == 0 and w['returncode'] == 0 and w['real_wait_observed'] and w['owned_engines_exited']
                    and w['original_physical_lease_released'] and w['terminal_status'] == 'COMPLETE'
                    and w['completed_ids'] == p['assigned_ids'], 'Restoration failed/partial; successors held without retry')
        q.update(status='COMPLETE', reports=reports); atomic_json(path, q)
    except BaseException as error:
        q.update(status='HELD_NO_RETRY', error=str(error), reports=reports); atomic_json(path, q); raise
    return {'status': q['status'], 'workers': len(reports)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'validate', 'run', 'supervise', 'queue'])
    parser.add_argument('--root'); parser.add_argument('--host'); parser.add_argument('--profile'); parser.add_argument('--queue')
    a = parser.parse_args()
    result = (prepare(a.root, a.host) if a.command == 'prepare' else
              validate_profile(read(a.profile), Path(a.profile).parents[2]) if a.command == 'validate' else
              queue(a.queue) if a.command == 'queue' else
              supervise(a.profile) if a.command == 'supervise' else run(a.profile))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
