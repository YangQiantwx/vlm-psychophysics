"""Finite T4 CPU scheduling: no GPU queries, retries, or implicit new work.

Reuse the frozen scheduling utilities, never its static-specific plan policy.
Model supervisors retain ownership of resource admission and scientific audits.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time

from . import static_x_queue as lifecycle
from .refinement_worker import descriptor, read, require, verified
from .storage import atomic_json, digest, file_sha256

MODULE = 'classic_precision.t4_balanced_queue'
OPERATIONS = 'classic_precision.t4_balanced_operations'
ORDER = ['cosmos', 'qwen_x_qsa', 'nemo', 'gemma_x_full']
COMMON = ('t4_authorization', 'source_catalog', 't4_protocol', 't4_allocation_plan')


def allocation_check(profiles, global_profiles, card):
    require(len(profiles) == 4 and [p['model'] for p in profiles] == ORDER,
            'T4 card must run Cosmos, Qwen, Nemo, then Gemma exactly once')
    require(len(global_profiles) == 8 and len({p['output'] for p in global_profiles}) == 8,
            'T4 global allocation requires eight distinct workers')
    require({p['card'] for p in global_profiles} == {'fourier0', 'fourier1'},
            'This frozen allocation permits only its two Fourier cards')
    require(all(p['card'] == card for p in profiles), 'Queue contains another card')
    require(all(p['assigned_ids'] and len(p['assigned_ids']) == len(set(p['assigned_ids']))
                == p['maximum_model_calls'] and p['automatic_retry'] is False
                and p['formal_auto_resume'] is False for p in global_profiles),
            'Worker allocation is duplicate, unbounded, or retryable')
    by_model = {}
    for model in ORDER:
        workers = [p for p in global_profiles if p['model'] == model]
        require(len(workers) == 2 and {p['card'] for p in workers} == {'fourier0', 'fourier1'},
                'Each model needs exactly one worker on each Fourier card')
        first, second = map(lambda p: set(p['assigned_ids']), workers)
        require(len(first) == len(second) == 408 and not first.intersection(second),
                'T4 model has overlapping or incorrectly sized allocations')
        by_model[model] = first | second
    require(all(ids == by_model[ORDER[0]] for ids in by_model.values()),
            'Four T4 models must cover the same 816 source IDs')
    return by_model[ORDER[0]]


def validate_plan(path, owner):
    plan = read(path)
    require(plan['schema'] == 't4-region-balanced-finite-queue-v1'
            and plan['owner_chat'] == owner and owner,
            'Unknown T4 plan or foreign owner')
    require(plan['queue_entry'] == descriptor(__file__), 'T4 queue source changed')
    ops = importlib.import_module(OPERATIONS)
    ops_sha = file_sha256(lifecycle.source_path(OPERATIONS))
    refs, global_refs = plan['profiles'], plan['global_profiles']
    require(len({r['path'] for r in global_refs}) == len(global_refs) == 8
            and len({r['path'] for r in refs}) == len(refs) == 4
            and all(r in global_refs for r in refs), 'Duplicate or foreign global profile')
    profiles, all_profiles = [verified(r) for r in refs], [verified(r) for r in global_refs]
    own_ids = allocation_check(profiles, all_profiles, plan['card'])
    for card in ('fourier0', 'fourier1'):
        card_profiles = [p for p in all_profiles if p['card'] == card]
        require([p['model'] for p in card_profiles] == ORDER, 'Global T4 peer ordering differs')
    require(refs == [r for r, p in zip(global_refs, all_profiles) if p['card'] == plan['card']],
            'Own T4 ordered profiles differ from global peer binding')
    for ref, p in zip(global_refs, all_profiles):
        require(ref['supervisor_module'] == OPERATIONS and ref['supervisor_sha256'] == ops_sha
                and p['execution_policy'] == {'module': OPERATIONS, 'source_sha256': ops_sha}
                and p['t4_final_review'] == plan['final_review'],
                'T4 supervisor binding changed')
        require(all(p[key] == plan[key] for key in COMMON), 'T4 common source/authority differs')
        auth, allocation = ops.authorization(p)
        require(allocation['assigned_ids'] == p['assigned_ids']
                and allocation['output'] == p['output'], 'T4 allocation identity differs')
        catalog = ops.configuration(p)
        require(set(catalog['items']) == own_ids, 'T4 catalog differs from global allocation')
        require(Path(p['python']).is_absolute() and Path(p['output']).is_absolute(),
                'T4 interpreter/output must be absolute')
    require(len({p['python'] for p in profiles}) == 1
            and all(p['gpu_uuid'] == plan['gpu_uuid'] for p in profiles)
            and all(p['physical_lock'] == profiles[0]['physical_lock'] for p in profiles),
            'T4 card interpreter, GPU, or physical lease differs')
    review = verified(plan['final_review'])
    require(review['schema'] == 't4-final-execution-review-v1'
            and review['status'] == 'PASS_FINAL_T4_EXECUTION_REVIEW'
            and review['owner_chat'] == owner
            and all(review[k] == plan[k] for k in ('t4_authorization', 'source_catalog', 't4_protocol')),
            'Final independent T4 preflight is absent or belongs to another source')
    code_refs = {x['path']: x['sha256'] for x in review['code_bundle']}
    for source in (__file__, lifecycle.source_path(OPERATIONS)):
        require(code_refs.get(str(source)) == file_sha256(source), 'Final review does not bind execution source')
    non_gemma = [r for r, p in zip(global_refs, all_profiles) if p['model'] != 'gemma_x_full']
    require(len(non_gemma) == 6, 'Global T4 Gemma-last barrier requires six workers')
    gemma = profiles[-1]['t4_pre_gemma_intent']
    require(gemma['owner_chat'] == owner
            and all(gemma[k] == plan[k] for k in COMMON[:3])
            and gemma['expected_non_gemma_profiles'] == non_gemma
            and str(Path(gemma['path']).resolve()) == str((Path(plan['output']) / 'pre-gemma-global-closure.json').resolve()),
            'Gemma preflight intent differs from exact global six-worker allocation')
    expected_peers = {p['card']: str(Path(p['t4_pre_gemma_intent']['path']).parent)
                      for p in all_profiles if p['model'] == 'gemma_x_full'}
    require(plan['peer_queue_outputs'] == expected_peers
            and plan['output'] == expected_peers[plan['card']], 'T4 peer queue namespace binding differs')
    predecessors = plan['predecessors']
    require(len(predecessors) == len({x['path'] for x in predecessors}) == 2,
            'T4 requires both original static queue predecessor bindings')
    pred_plans = [verified(x) for x in predecessors]
    require({p['card'] for p in pred_plans} == {'fourier0', 'fourier1'}
            and all(p['schema'] == 'static-x-finite-queue-v1' and p['phase'] == 'full'
                    and p['owner_chat'] == owner for p in pred_plans),
            'T4 predecessor is not the authorized static full phase')
    require(len({p['output'] for p in pred_plans}) == 2, 'T4 predecessor outputs duplicate')
    require(isinstance(plan['event_wait_timeout_seconds'], (int, float))
            and 0 < plan['event_wait_timeout_seconds'] <= 86400,
            'T4 event wait must have an explicit finite deadline')
    lock = plan['queue_lease']; physical = profiles[0]['physical_lock']
    expected_lock = Path(physical['path']).with_name('t4-cpu-queue-' + plan['gpu_uuid'].lower().removeprefix('gpu-') + '.lock')
    require(lock['path'] == str(expected_lock) and Path(lock['path']).is_file()
            and not Path(lock['path']).is_symlink(), 'T4 CPU queue lease is not its stable regular file')
    st = Path(lock['path']).stat()
    require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'T4 CPU queue lease inode changed')
    require(Path(plan['output']).is_absolute(), 'T4 queue output is not absolute')
    return plan, profiles


def peer_snapshot(plan, mode):
    """Fail promptly on a known terminal failure; allow real parent-wait windows."""
    identities, paths = [], []
    if mode == 'predecessors':
        peers = [(ref, verified(ref)) for ref in plan['predecessors']]
    else:
        peers = [(None, {'output': output, 'card': card})
                 for card, output in plan['peer_queue_outputs'].items()]
    for expected_ref, expected in peers:
        out = Path(expected['output'])
        names = ('launcher-failure.json', 'queue-result.json', 'launch-intent.json',
                 'queue-identity.json', 'outer-wait.json')
        paths.extend(out / name for name in names)
        require(not (out / 'launcher-failure.json').exists(), 'Known peer queue launcher failed')
        result = lifecycle.optional_json(out / 'queue-result.json')
        require(result is None or result['status'] == 'COMPLETE', 'Known peer queue failed or is uncertain')
        launch = lifecycle.optional_json(out / 'launch-intent.json')
        identity = lifecycle.optional_json(out / 'queue-identity.json')
        outer = lifecycle.optional_json(out / 'outer-wait.json')
        if launch:
            peer = verified(launch['plan'])
            require(peer['output'] == str(out) and peer['owner_chat'] == plan['owner_chat'],
                    'Known peer output/owner does not match its original queue')
            if expected_ref is not None:
                require(launch['plan'] == expected_ref, 'Static predecessor plan changed')
            else:
                require(peer['card'] == expected['card'] and peer['queue_entry'] == plan['queue_entry']
                        and all(peer[k] == plan[k] for k in COMMON)
                        and peer['global_profiles'] == plan['global_profiles']
                        and peer['profiles'] == [r for r in plan['global_profiles']
                            if verified(r)['card'] == expected['card']]
                        and peer['peer_queue_outputs'] == plan['peer_queue_outputs'],
                        'Known T4 peer authorization/allocation/source differs')
            launcher = lifecycle.owner_format(launch['launcher'])
            if lifecycle.process_alive(launcher): identities.append(launcher)
            elif outer is None:
                require(False, 'Known queue launcher exited without its real outer wait')
        if identity:
            require(launch is not None and identity['plan'] == launch['plan']
                    and identity['launcher'] == launch['launcher'], 'Known peer identity lacks matching launch intent')
            owner = lifecycle.owner_format(identity['owner'])
            if lifecycle.process_alive(owner): identities.append(owner)
            else:
                require(result is not None, 'Known queue child exited without terminal result')
                # A COMPLETE child may have exited just before its live launcher
                # writes outer-wait. The launcher pidfd/filesystem event is watched.
        if mode != 'predecessors':
            card_refs = [r for r in plan['global_profiles'] if verified(r)['card'] == expected['card']]
            for ordinal, ref in enumerate(card_refs):
                p = verified(ref)
                if p['model'] == 'gemma_x_full': continue
                scheduling_wait = out / f'profile-{ordinal:02d}-wait.json'
                paths.append(scheduling_wait)
                wait = lifecycle.optional_json(scheduling_wait)
                if wait is not None:
                    require(wait['profile'] == ref and wait['returncode'] == 0
                            and wait['real_wait_observed'] is True and wait['known_supervisor_exited'] is True,
                            'Known non-Gemma supervisor failed or is uncertain')
                    require(lifecycle.optional_json(Path(p['output']).with_suffix('.wait.json')) is not None,
                            'Known successful supervisor has no actual worker wait')
    return identities, paths


def barrier(plan, mode):
    """Listen only to filesystem events/known owned PID exits, holding no GPU lock."""
    if mode == 'predecessors':
        refs = plan['predecessors']; entries = [verified(r) for r in refs]
        targets = [Path(p['output']) / 'outer-wait.json' for p in entries]
        failure_paths = [Path(p['output']) / 'launcher-failure.json' for p in entries]
        closure_name = 'predecessors-closed.json'
    else:
        refs = [r for r in plan['global_profiles'] if verified(r)['model'] != 'gemma_x_full']
        entries = [verified(r) for r in refs]
        targets = [Path(p['output']).with_suffix('.wait.json') for p in entries]
        failure_paths = [Path(p['output']) / 'failure.json' for p in entries]
        closure_name = 'pre-gemma-global-closure.json'
    _, peer_paths = peer_snapshot(plan, mode)
    deadline = time.monotonic() + plan['event_wait_timeout_seconds']
    with lifecycle.PhaseEvents(targets + failure_paths + peer_paths) as events:
        while True:
            peer_identities, _ = peer_snapshot(plan, mode)
            require(not any(p.exists() for p in failure_paths), 'Known predecessor/worker failed; no continuation')
            receipts = [lifecycle.optional_json(p) for p in targets]
            if all(x is not None for x in receipts):
                if mode == 'predecessors':
                    closures = [lifecycle.closed(r['path'], plan['owner_chat']) for r in refs]
                else:
                    closures = [lifecycle.profile_closed(r, p) for r, p in zip(refs, entries)]
                record = {'schema': 't4-global-pre-gemma-closure-v1' if mode != 'predecessors' else 't4-static-predecessor-closure-v1',
                    'status': 'PASS_ALL_NON_GEMMA_T4_REAL_CLOSED' if mode != 'predecessors' else 'PASS_BOTH_STATIC_QUEUES_REAL_CLOSED',
                    'owner_chat': plan['owner_chat'], 'closures': closures,
                    'physical_GPU_lease_held': False, 'GPU_queries': 0,
                    **{k: plan[k] for k in COMMON[:3]}}
                record['expected_non_gemma_profiles' if mode != 'predecessors' else 'predecessors'] = refs
                lifecycle.durable_new(Path(plan['output']) / closure_name, record)
                return descriptor(Path(plan['output']) / closure_name)
            identities = list(peer_identities)
            for p in entries:
                launch_path = Path(p['output']) / 'queue-identity.json' if mode == 'predecessors' else Path(p['output']).with_suffix('.launch.json')
                value = lifecycle.optional_json(launch_path)
                if value:
                    identity = lifecycle.owner_format(value['owner'])
                    if lifecycle.process_alive(identity): identities.append(identity)
            # Register first, then recheck files, closing the filesystem race.
            if not events.refresh(identities): continue
            if any(lifecycle.optional_json(p) is not None and old is None for p, old in zip(targets, receipts)): continue
            require(not any(p.exists() for p in failure_paths), 'Known predecessor/worker failed during barrier setup')
            peer_snapshot(plan, mode)
            events.wait(deadline)


def run(path, owner):
    plan, profiles = validate_plan(path, owner); out = Path(plan['output'])
    require(sys.executable == profiles[0]['python'] and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
            'T4 CPU queue must use the sealed interpreter with CUDA invisible')
    intent = read(out / 'launch-intent.json')
    require(intent['plan'] == descriptor(path) and intent['queue_entry'] == plan['queue_entry']
            and intent['owner_chat'] == owner and os.getppid() == intent['launcher']['pid']
            and lifecycle.process_identity(os.getppid()) == intent['launcher'], 'T4 queue is not its original direct child')
    lifecycle.durable_new(out / 'queue-identity.json', {'plan': intent['plan'], 'owner': lifecycle.process_identity(),
        'launcher': intent['launcher'], 'queue_entry': plan['queue_entry']})
    completed, pending = [], None
    try:
        with Path(plan['queue_lease']['path']).open('r+b') as stream:
            st = os.fstat(stream.fileno()); lock = plan['queue_lease']
            require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'T4 CPU lease replaced')
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lifecycle.durable_new(out / 'queue-lease.json', {'lease': lock, 'owner': lifecycle.process_identity(),
                'physical_GPU_lease_held_by_CPU_queue': False})
            barrier(plan, 'predecessors')
            for ordinal, (ref, p) in enumerate(zip(plan['profiles'], profiles)):
                require(descriptor(path) == intent['plan'] and descriptor(__file__) == plan['queue_entry'], 'T4 plan/code changed at boundary')
                require(lifecycle.process_alive(intent['launcher']), 'Original T4 launcher exited; no successor dispatch')
                verified(ref)
                require(file_sha256(lifecycle.source_path(OPERATIONS)) == ref['supervisor_sha256'], 'T4 supervisor changed')
                if ordinal: lifecycle.profile_closed(plan['profiles'][ordinal - 1], profiles[ordinal - 1])
                gate = barrier(plan, 'gemma') if p['model'] == 'gemma_x_full' else None
                pending = ref; stem = f'profile-{ordinal:02d}'
                lifecycle.durable_new(out / (stem + '-intent.json'), {'ordinal': ordinal, 'profile': ref,
                    'queue_owner': lifecycle.process_identity(), 'automatic_retry': False, 'pre_gemma_barrier': gate})
                env = dict(os.environ)
                for key in p.get('cleared_environment', []): env.pop(key, None)
                env.update(p['environment'])
                argv = [p['python'], '-B', '-m', OPERATIONS, 'supervise', '--profile', ref['path']]
                with (out / (stem + '.log')).open('xb') as log:
                    identity, code = lifecycle.spawn_and_wait(argv, log=log,
                        cwd=str(Path(p['implementation']['path']).parent.parent), env=env,
                        child_receipt=out / (stem + '-child.json'),
                        receipt_fields={'ordinal': ordinal, 'profile': ref, 'parent': lifecycle.process_identity()}, identity_key='supervisor')
                wait = {'ordinal': ordinal, 'profile': ref, 'supervisor': identity, 'returncode': code,
                    'real_wait_observed': True, 'known_supervisor_exited': not lifecycle.process_alive(identity), 'automatic_retry': False}
                lifecycle.durable_new(out / (stem + '-wait.json'), wait)
                require(code == 0 and wait['known_supervisor_exited'], 'T4 supervisor failed/uncertain; no retry')
                closure = lifecycle.profile_closed(ref, p)
                completed.append({'ordinal': ordinal, 'profile': ref, 'queue_wait': descriptor(out / (stem + '-wait.json')), 'closure': closure})
                pending = None
                atomic_json(out / 'progress.json', {'status': 'RUNNING', 'completed': len(completed), 'expected': 4, 'automatic_retry': False})
            atomic_json(out / 'queue-result.json', {'status': 'COMPLETE', 'plan': intent['plan'], 'completed': completed,
                'UNCERTAIN_profiles': [], 'automatic_retry': False, 'next_phase_launched': False})
        return {'status': 'COMPLETE', 'completed_profiles': 4}
    except BaseException as error:
        atomic_json(out / 'queue-result.json', {'status': 'FAILED_OR_UNCERTAIN', 'plan': intent['plan'], 'completed': completed,
            'UNCERTAIN_profiles': [pending] if pending else [], 'error': str(error), 'automatic_retry': False,
            'created_child_wait': getattr(error, 'static_queue_child_wait', None), 'next_phase_launched': False})
        raise


def launch_once(path, owner):
    plan, profiles = validate_plan(path, owner); out = Path(plan['output'])
    require(sys.platform.startswith('linux'), 'T4 launch needs actual Linux identities')
    parent = lifecycle.process_identity()
    lifecycle.durable_new(out / 'launch-intent.json', {'schema': 't4-exclusive-queue-launch-v1', 'plan': descriptor(path),
        'queue_entry': plan['queue_entry'], 'owner_chat': owner, 'launcher': parent, 'automatic_retry': False})
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = ''
    argv = [profiles[0]['python'], '-B', '-m', MODULE, 'run', '--plan', str(path), '--owner-chat', owner]
    try:
        with (out / 'queue-run.log').open('xb') as log:
            identity, code = lifecycle.spawn_and_wait(argv, log=log, cwd=str(Path(__file__).parent.parent), env=env,
                child_receipt=out / 'queue-child.json', receipt_fields={'launcher': parent, 'plan': descriptor(path)}, identity_key='queue_child')
        result = read(out / 'queue-result.json')
        record = {'schema': 't4-real-queue-outer-wait-v1', 'plan': descriptor(path), 'launcher': parent,
            'queue_child': identity, 'returncode': code, 'real_wait_observed': True,
            'known_queue_child_exited': not lifecycle.process_alive(identity), 'terminal_status': result['status'], 'automatic_retry': False}
        lifecycle.durable_new(out / 'outer-wait.json', record)
        require(code == 0 and record['known_queue_child_exited'] and result['status'] == 'COMPLETE', 'T4 outer failed/uncertain')
        return {'status': 'COMPLETE', 'outer_wait': descriptor(out / 'outer-wait.json')}
    except BaseException as error:
        lifecycle.durable_new(out / 'launcher-failure.json', {'plan': descriptor(path), 'launcher': parent, 'error': str(error),
            'created_child_wait': getattr(error, 'static_queue_child_wait', None), 'automatic_retry': False})
        raise


def closed(path, owner):
    plan, profiles = validate_plan(path, owner); out = Path(plan['output'])
    intent, child, identity, outer, result = [read(out / name) for name in
        ('launch-intent.json', 'queue-child.json', 'queue-identity.json', 'outer-wait.json', 'queue-result.json')]
    require(intent['plan'] == child['plan'] == identity['plan'] == outer['plan'] == result['plan'] == descriptor(path)
            and intent['launcher'] == child['launcher'] == identity['launcher'] == outer['launcher']
            and child['queue_child'] == identity['owner'] == outer['queue_child']
            and intent['owner_chat'] == owner and intent['queue_entry'] == identity['queue_entry'] == plan['queue_entry'],
            'T4 queue lifecycle identities differ')
    require(outer['returncode'] == 0 and outer['real_wait_observed'] is True and outer['known_queue_child_exited'] is True
            and outer['terminal_status'] == result['status'] == 'COMPLETE' and result['automatic_retry'] is False
            and not result['UNCERTAIN_profiles'] and len(result['completed']) == 4
            and not lifecycle.process_alive(identity['owner']) and not (out / 'launcher-failure.json').exists(),
            'T4 lacks real successful outer closure')
    for ordinal, (ref, p, saved) in enumerate(zip(plan['profiles'], profiles, result['completed'])):
        stem = f'profile-{ordinal:02d}'; own_intent, own_child, wait = [read(out / (stem + suffix)) for suffix in
            ('-intent.json', '-child.json', '-wait.json')]
        require(saved['ordinal'] == own_intent['ordinal'] == own_child['ordinal'] == wait['ordinal'] == ordinal
                and saved['profile'] == own_intent['profile'] == own_child['profile'] == wait['profile'] == ref
                and own_intent['queue_owner'] == own_child['parent'] == identity['owner']
                and own_intent['pre_gemma_barrier'] == (descriptor(out / 'pre-gemma-global-closure.json')
                    if p['model'] == 'gemma_x_full' else None)
                and own_child['supervisor'] == wait['supervisor'] and wait['returncode'] == 0
                and wait['real_wait_observed'] is True and wait['known_supervisor_exited'] is True
                and not lifecycle.process_alive(wait['supervisor'])
                and saved['closure'] == lifecycle.profile_closed(ref, p)
                and saved['queue_wait'] == descriptor(out / (stem + '-wait.json')), 'T4 worker lifecycle differs')
    lease = read(out / 'queue-lease.json'); lock = plan['queue_lease']; st = Path(lock['path']).stat()
    require(lease['lease'] == lock and lease['owner'] == identity['owner']
            and lease['physical_GPU_lease_held_by_CPU_queue'] is False
            and (st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'T4 CPU lease proof differs')
    predecessor = read(out / 'predecessors-closed.json')
    require(predecessor['status'] == 'PASS_BOTH_STATIC_QUEUES_REAL_CLOSED'
            and predecessor['predecessors'] == plan['predecessors']
            and predecessor['owner_chat'] == owner
            and all(predecessor[k] == plan[k] for k in COMMON[:3])
            and predecessor['physical_GPU_lease_held'] is False and predecessor['GPU_queries'] == 0
            and predecessor['closures'] == [lifecycle.closed(r['path'], owner) for r in plan['predecessors']], 'T4 static predecessor proof differs')
    gate = read(out / 'pre-gemma-global-closure.json')
    non_gemma = [r for r in plan['global_profiles'] if verified(r)['model'] != 'gemma_x_full']
    require(gate['status'] == 'PASS_ALL_NON_GEMMA_T4_REAL_CLOSED' and gate['expected_non_gemma_profiles'] == non_gemma
            and gate['closures'] == [lifecycle.profile_closed(r, verified(r)) for r in non_gemma]
            and all(gate[k] == plan[k] for k in COMMON[:3]) and gate['owner_chat'] == owner
            and gate['physical_GPU_lease_held'] is False and gate['GPU_queries'] == 0,
            'T4 Gemma-last proof differs')
    return {'status': 'COMPLETE_PASS_T4_REAL_QUEUE_CLOSURE', 'plan': descriptor(path),
        'outer_wait': descriptor(out / 'outer-wait.json'), 'completed_profiles': 4, 'model_calls_by_queue_module': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'run', 'launch_once', 'closed'))
    parser.add_argument('--plan', required=True); parser.add_argument('--owner-chat', required=True)
    args = parser.parse_args()
    value = ({'status': 'PASS_T4_QUEUE_METADATA', 'profiles': len(validate_plan(args.plan, args.owner_chat)[1])}
             if args.command == 'validate' else globals()[args.command](args.plan, args.owner_chat))
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__': main()
