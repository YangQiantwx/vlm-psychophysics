"""Finite SST4 CPU scheduling: no GPU queries, retries, or implicit new work.

Reuse the frozen scheduling utilities, never its static-specific plan policy.
Model supervisors retain ownership of resource admission and scientific audits.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import json
import os
from pathlib import Path
import sys
import time

from . import static_x_queue as lifecycle
from . import t4_balanced_queue as video_queue
from .refinement_worker import descriptor, read, require, verified
from .storage import atomic_json, digest, file_sha256

MODULE = 'classic_precision.static_t4_strict_queue'
OPERATIONS = 'classic_precision.static_t4_strict_operations'
ORDER = ['cosmos', 'qwen_x_qsa', 'nemo', 'gemma_x_full']
OWNER = os.environ.get('PSYCHOPHYSICS_QUEUE_OWNER', 'unbound-current-queue')
NAME = 'static-t4-strict-first-last'
VIDEO_OWNER = os.environ.get('PSYCHOPHYSICS_PARENT_QUEUE_OWNER', 'unbound-parent-queue')
COMMON = ('st4_authorization', 'source_catalog', 'st4_protocol', 'st4_parent_index', 'st4_allocation_plan')


def allocation_check(profiles, global_profiles, card):
    require(len(profiles) == 4 and [p['model'] for p in profiles] == ORDER,
            'ST4 card must run Cosmos, Qwen, Nemo, then Gemma exactly once')
    require(len(global_profiles) == 8 and len({p['output'] for p in global_profiles}) == 8,
            'ST4 global allocation requires eight distinct workers')
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
        require(not first.intersection(second) and 0 < len(first | second) <= 768,
                'ST4 model has overlapping or incorrectly sized allocations')
        by_model[model] = first | second
    require(all(ids == by_model[ORDER[0]] for ids in by_model.values()),
            'Four ST4 models must cover the same deduplicated source IDs')
    return by_model[ORDER[0]]


def validate_plan(path, owner):
    plan = read(path)
    require(plan['schema'] == 'static-t4-first-last-finite-queue-v1'
            and plan['owner_chat'] == owner == OWNER,
            'Unknown ST4 plan or foreign owner')
    require(plan['queue_entry'] == descriptor(__file__), 'ST4 queue source changed')
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
        require([p['model'] for p in card_profiles] == ORDER, 'Global ST4 peer ordering differs')
    require(refs == [r for r, p in zip(global_refs, all_profiles) if p['card'] == plan['card']],
            'Own ST4 ordered profiles differ from global peer binding')
    for ref, p in zip(global_refs, all_profiles):
        require(ref['supervisor_module'] == OPERATIONS and ref['supervisor_sha256'] == ops_sha
                and p['execution_policy'] == {'module': OPERATIONS, 'source_sha256': ops_sha}
                and p['st4_final_review'] == plan['final_review'],
                'ST4 supervisor binding changed')
        require(all(p[key] == plan[key] for key in COMMON), 'ST4 common source/authority differs')
        auth, allocation = ops.authorization(p)
        require(allocation['assigned_ids'] == p['assigned_ids']
                and allocation['output'] == p['output'], 'ST4 allocation identity differs')
        catalog = ops.configuration(p)
        require(set(catalog['items']) == own_ids, 'ST4 catalog differs from global allocation')
        require(Path(p['python']).is_absolute() and Path(p['output']).is_absolute(),
                'ST4 interpreter/output must be absolute')
    require(len({p['python'] for p in profiles}) == 1
            and all(p['gpu_uuid'] == plan['gpu_uuid'] for p in profiles)
            and all(p['physical_lock'] == profiles[0]['physical_lock'] for p in profiles),
            'ST4 card interpreter, GPU, or physical lease differs')
    review = verified(plan['final_review'])
    require(review['schema'] == 'static-t4-final-execution-review-v1'
            and review['status'] == 'PASS_STATIC_T4_EXECUTION_PREFLIGHT'
            and review['owner_chat'] == owner
            and all(review[k] == plan[k] for k in COMMON),
            'Final independent ST4 preflight is absent or belongs to another source')
    code_refs = {x['path']: x['sha256'] for x in review['codes'].values()}
    for source in (__file__, lifecycle.source_path(OPERATIONS)):
        require(code_refs.get(str(source)) == file_sha256(source), 'Final review does not bind execution source')
    non_gemma = [r for r, p in zip(global_refs, all_profiles) if p['model'] != 'gemma_x_full']
    require(len(non_gemma) == 6, 'Global ST4 Gemma-last barrier requires six workers')
    gemma = profiles[-1]['st4_pre_gemma_intent']
    require(gemma['owner_chat'] == owner
            and all(gemma[k] == plan[k] for k in COMMON[:3])
            and gemma['expected_non_gemma_profiles'] == non_gemma
            and str(Path(gemma['path']).resolve()) == str((Path(plan['output']) / 'pre-gemma-global-closure.json').resolve()),
            'Gemma preflight intent differs from exact global six-worker allocation')
    expected_peers = {p['card']: str(Path(p['st4_pre_gemma_intent']['path']).parent)
                      for p in all_profiles if p['model'] == 'gemma_x_full'}
    require(plan['peer_queue_outputs'] == expected_peers
            and plan['output'] == expected_peers[plan['card']], 'ST4 peer queue namespace binding differs')
    predecessors = plan['predecessors']
    require(len(predecessors) == len({x['path'] for x in predecessors}) == 2,
            'ST4 requires both original T4 video queue predecessor bindings')
    pred_plans = [verified(x) for x in predecessors]
    require({p['card'] for p in pred_plans} == {'fourier0', 'fourier1'}
            and all(p['schema'] == 't4-region-balanced-finite-queue-v1'
                    and p['owner_chat'] == VIDEO_OWNER for p in pred_plans),
            'ST4 predecessor is not the authorized T4 video phase')
    require(len({p['output'] for p in pred_plans}) == 2, 'ST4 predecessor outputs duplicate')
    require(isinstance(plan['event_wait_timeout_seconds'], (int, float))
            and 0 < plan['event_wait_timeout_seconds'] <= 86400,
            'ST4 event wait must have an explicit finite deadline')
    lock = plan['queue_lease']; physical = profiles[0]['physical_lock']
    expected_lock = Path(physical['path']).with_name('st4-strict-cpu-queue-' + plan['gpu_uuid'].lower().removeprefix('gpu-') + '.lock')
    require(lock['path'] == str(expected_lock) and Path(lock['path']).is_file()
            and not Path(lock['path']).is_symlink(), 'ST4 CPU queue lease is not its stable regular file')
    st = Path(lock['path']).stat()
    require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'ST4 CPU queue lease inode changed')
    require(Path(plan['output']) == Path(profiles[0]['workspace_root']) / 'runs/reproducibility' / NAME / ('queue-' + plan['card']),
            'ST4 queue output namespace differs')
    require(all(Path(p['output']).parent == Path(plan['output']).parent and Path(p['claim_root']).parent == Path(plan['output']).parent
                for p in all_profiles), 'ST4 worker namespace differs')
    return plan, profiles


def start_gate(p):
    """Model-process boundary: only this armed queue's known supervisor may enter."""
    out = Path(p['workspace_root']) / 'runs/reproducibility' / NAME / ('queue-' + p['card'])
    launch = read(out / 'launch-intent.json')
    require(launch['owner_chat'] == OWNER and launch['scope'] == 'ST4_ARM_CPU_EVENT_WAIT_ONLY'
            and launch['physical_GPU_lease_held'] is False and launch['GPU_queries'] == 0,
            'No exact one-time ST4 CPU arming intent')
    plan, profiles = validate_plan(launch['plan']['path'], OWNER)
    require(launch['plan'] == descriptor(launch['plan']['path']) and plan['output'] == str(out)
            and launch['queue_entry'] == plan['queue_entry'], 'ST4 arming source/plan changed')
    matches = [i for i, own in enumerate(profiles) if own == p]
    require(len(matches) == 1, 'Unsealed/mutated ST4 model profile at start gate')
    ordinal = matches[0]; ref = plan['profiles'][ordinal]
    identity = read(out / 'queue-identity.json'); intent = read(out / f'profile-{ordinal:02d}-intent.json')
    child = read(out / f'profile-{ordinal:02d}-child.json')
    require(identity['plan'] == launch['plan'] and identity['queue_entry'] == plan['queue_entry']
            and identity['launcher'] == launch['launcher'] and lifecycle.process_alive(identity['owner'])
            and lifecycle.process_alive(launch['launcher']), 'Original ST4 CPU owner/launcher is absent')
    require(intent['ordinal'] == child['ordinal'] == ordinal and intent['profile'] == child['profile'] == ref
            and intent['queue_owner'] == child['parent'] == identity['owner'] and intent['automatic_retry'] is False,
            'ST4 supervisor was not dispatched by its original queue')
    supervisor = lifecycle.owner_format(child['supervisor'])
    require(supervisor in (lifecycle.process_identity(), lifecycle.process_identity(os.getppid()))
            and lifecycle.process_alive(supervisor), 'ST4 caller is not its known supervisor/direct model child')
    predecessor = read(out / 'predecessors-closed.json')
    require(predecessor['status'] == 'PASS_BOTH_T4_VIDEO_QUEUES_REAL_CLOSED'
            and predecessor['owner_chat'] == OWNER and predecessor['predecessors'] == plan['predecessors']
            and predecessor['closures'] == [video_queue.closed(r['path'], VIDEO_OWNER) for r in plan['predecessors']]
            and all(predecessor[k] == plan[k] for k in COMMON[:3])
            and predecessor['physical_GPU_lease_held'] is False and predecessor['GPU_queries'] == 0,
            'Original T4 video predecessors lack both real outer closures')
    gate = None
    if p['model'] == 'gemma_x_full':
        non_gemma = [r for r in plan['global_profiles'] if verified(r)['model'] != 'gemma_x_full']
        value = read(out / 'pre-gemma-global-closure.json'); policy = p['st4_pre_gemma_intent']
        require(value['schema'] == 'st4-global-pre-gemma-closure-v1' and value['status'] == 'PASS_ALL_NON_GEMMA_ST4_REAL_CLOSED'
                and value['owner_chat'] == OWNER and value['expected_non_gemma_profiles'] == non_gemma == policy['expected_non_gemma_profiles']
                and len(non_gemma) == 6 and all(value[k] == plan[k] == policy[k] for k in COMMON[:3])
                and value['closures'] == [lifecycle.profile_closed(r, verified(r)) for r in non_gemma]
                and value['physical_GPU_lease_held'] is False and value['GPU_queries'] == 0,
                'Gemma lacks this ST4 allocation\'s six actual non-Gemma closures')
        gate = descriptor(out / 'pre-gemma-global-closure.json')
    require(intent['pre_gemma_barrier'] == gate, 'Supervisor intent lacks its exact ST4 barrier')
    return {'status': 'PASS_ST4_ARM_AND_REAL_PREDECESSOR_GATE', 'profile': ref, 'queue': launch['plan'],
            'predecessors': descriptor(out / 'predecessors-closed.json'), 'pre_gemma_barrier': gate,
            'GPU_queries': 0, 'model_calls': 0}


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
            require(peer['output'] == str(out) and peer['owner_chat'] == (VIDEO_OWNER if expected_ref else plan['owner_chat']),
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
                        'Known ST4 peer authorization/allocation/source differs')
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
                    closures = [video_queue.closed(r['path'], VIDEO_OWNER) for r in refs]
                else:
                    closures = [lifecycle.profile_closed(r, p) for r, p in zip(refs, entries)]
                record = {'schema': 'st4-global-pre-gemma-closure-v1' if mode != 'predecessors' else 'st4-video-predecessor-closure-v1',
                    'status': 'PASS_ALL_NON_GEMMA_ST4_REAL_CLOSED' if mode != 'predecessors' else 'PASS_BOTH_T4_VIDEO_QUEUES_REAL_CLOSED',
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
            'ST4 CPU queue must use the sealed interpreter with CUDA invisible')
    intent = read(out / 'launch-intent.json')
    require(intent['plan'] == descriptor(path) and intent['queue_entry'] == plan['queue_entry']
            and intent['owner_chat'] == owner and os.getppid() == intent['launcher']['pid']
            and lifecycle.process_identity(os.getppid()) == intent['launcher'], 'ST4 queue is not its original direct child')
    lifecycle.durable_new(out / 'queue-identity.json', {'plan': intent['plan'], 'owner': lifecycle.process_identity(),
        'launcher': intent['launcher'], 'queue_entry': plan['queue_entry']})
    completed, pending = [], None
    try:
        with Path(plan['queue_lease']['path']).open('r+b') as stream:
            st = os.fstat(stream.fileno()); lock = plan['queue_lease']
            require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'ST4 CPU lease replaced')
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lifecycle.durable_new(out / 'queue-lease.json', {'lease': lock, 'owner': lifecycle.process_identity(),
                'physical_GPU_lease_held_by_CPU_queue': False})
            barrier(plan, 'predecessors')
            for ordinal, (ref, p) in enumerate(zip(plan['profiles'], profiles)):
                require(descriptor(path) == intent['plan'] and descriptor(__file__) == plan['queue_entry'], 'ST4 plan/code changed at boundary')
                require(lifecycle.process_alive(intent['launcher']), 'Original ST4 launcher exited; no successor dispatch')
                verified(ref)
                require(file_sha256(lifecycle.source_path(OPERATIONS)) == ref['supervisor_sha256'], 'ST4 supervisor changed')
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
                require(code == 0 and wait['known_supervisor_exited'], 'ST4 supervisor failed/uncertain; no retry')
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
    require(sys.platform.startswith('linux'), 'ST4 launch needs actual Linux identities')
    parent = lifecycle.process_identity()
    lifecycle.durable_new(out / 'launch-intent.json', {'schema': 'st4-exclusive-queue-launch-v1', 'plan': descriptor(path),
        'scope': 'ST4_ARM_CPU_EVENT_WAIT_ONLY', 'queue_entry': plan['queue_entry'], 'owner_chat': owner, 'launcher': parent,
        'physical_GPU_lease_held': False, 'GPU_queries': 0, 'automatic_retry': False})
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = ''
    argv = [profiles[0]['python'], '-B', '-m', MODULE, 'run', '--plan', str(path), '--owner-chat', owner]
    try:
        with (out / 'queue-run.log').open('xb') as log:
            identity, code = lifecycle.spawn_and_wait(argv, log=log, cwd=str(Path(__file__).parent.parent), env=env,
                child_receipt=out / 'queue-child.json', receipt_fields={'launcher': parent, 'plan': descriptor(path)}, identity_key='queue_child')
        result = read(out / 'queue-result.json')
        record = {'schema': 'st4-real-queue-outer-wait-v1', 'plan': descriptor(path), 'launcher': parent,
            'queue_child': identity, 'returncode': code, 'real_wait_observed': True,
            'known_queue_child_exited': not lifecycle.process_alive(identity), 'terminal_status': result['status'], 'automatic_retry': False}
        lifecycle.durable_new(out / 'outer-wait.json', record)
        require(code == 0 and record['known_queue_child_exited'] and result['status'] == 'COMPLETE', 'ST4 outer failed/uncertain')
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
            'ST4 queue lifecycle identities differ')
    require(outer['returncode'] == 0 and outer['real_wait_observed'] is True and outer['known_queue_child_exited'] is True
            and outer['terminal_status'] == result['status'] == 'COMPLETE' and result['automatic_retry'] is False
            and not result['UNCERTAIN_profiles'] and len(result['completed']) == 4
            and not lifecycle.process_alive(identity['owner']) and not (out / 'launcher-failure.json').exists(),
            'ST4 lacks real successful outer closure')
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
                and saved['queue_wait'] == descriptor(out / (stem + '-wait.json')), 'ST4 worker lifecycle differs')
    lease = read(out / 'queue-lease.json'); lock = plan['queue_lease']; st = Path(lock['path']).stat()
    require(lease['lease'] == lock and lease['owner'] == identity['owner']
            and lease['physical_GPU_lease_held_by_CPU_queue'] is False
            and (st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'ST4 CPU lease proof differs')
    predecessor = read(out / 'predecessors-closed.json')
    require(predecessor['status'] == 'PASS_BOTH_T4_VIDEO_QUEUES_REAL_CLOSED'
            and predecessor['predecessors'] == plan['predecessors']
            and predecessor['owner_chat'] == owner
            and all(predecessor[k] == plan[k] for k in COMMON[:3])
            and predecessor['physical_GPU_lease_held'] is False and predecessor['GPU_queries'] == 0
            and predecessor['closures'] == [video_queue.closed(r['path'], VIDEO_OWNER) for r in plan['predecessors']], 'ST4 video predecessor proof differs')
    gate = read(out / 'pre-gemma-global-closure.json')
    non_gemma = [r for r in plan['global_profiles'] if verified(r)['model'] != 'gemma_x_full']
    require(gate['status'] == 'PASS_ALL_NON_GEMMA_ST4_REAL_CLOSED' and gate['expected_non_gemma_profiles'] == non_gemma
            and gate['closures'] == [lifecycle.profile_closed(r, verified(r)) for r in non_gemma]
            and all(gate[k] == plan[k] for k in COMMON[:3]) and gate['owner_chat'] == owner
            and gate['physical_GPU_lease_held'] is False and gate['GPU_queries'] == 0,
            'ST4 Gemma-last proof differs')
    return {'status': 'COMPLETE_PASS_ST4_REAL_QUEUE_CLOSURE', 'plan': descriptor(path),
        'outer_wait': descriptor(out / 'outer-wait.json'), 'completed_profiles': 4, 'model_calls_by_queue_module': 0}


def self_test():
    """Small synthetic scheduling tests; no current entry, GPU, PID or remote read."""
    tests = []
    def rejected(label, values):
        try:
            allocation_check(values[:4], values, 'fourier0')
        except ValueError:
            tests.append(label)
        else:
            raise AssertionError('Accepted: ' + label)
    values = [{'model': model, 'card': card, 'output': '/synthetic/' + model + '-' + card,
        'assigned_ids': ['u0', 'u1', 'u2'] if card == 'fourier0' else ['u3'],
        'maximum_model_calls': 3 if card == 'fourier0' else 1, 'automatic_retry': False, 'formal_auto_resume': False}
        for card in ('fourier0', 'fourier1') for model in ORDER]
    require(allocation_check(values[:4], values, 'fourier0') == {'u0','u1','u2','u3'}, 'Dynamic unequal allocation failed')
    tests.append('dynamic_unique_calls_not_alias_or_408')
    bad = copy.deepcopy(values);bad[4]['assigned_ids'] = ['u2'];rejected('overlap_rejected',bad)
    bad = copy.deepcopy(values);bad[4]['assigned_ids'] = ['different'];rejected('different_model_source_set_rejected',bad)
    bad = copy.deepcopy(values);bad[1]['automatic_retry'] = True;rejected('retry_rejected',bad)
    bad = copy.deepcopy(values);bad[1]['assigned_ids'] = ['u0','u0','u2'];rejected('duplicate_claim_ID_rejected',bad)
    bad = copy.deepcopy(values);bad[0],bad[1] = bad[1],bad[0];rejected('wrong_order_rejected',bad)
    bad = copy.deepcopy(values);bad[-1]['output'] = bad[0]['output'];rejected('duplicate_worker_namespace_rejected',bad)
    bad = copy.deepcopy(values);bad[0]['maximum_model_calls'] = 816;rejected('unbounded_or_alias_budget_rejected',bad)
    # Start-gate dependencies are controlled metadata fixtures, not claims of real closure.
    import types
    original = {k: globals()[k] for k in ('read','descriptor','verified','validate_plan','lifecycle','video_queue')}
    try:
        base = Path('/synthetic/ST4');qout = base/'runs/reproducibility'/NAME/'queue-fourier0'
        refs = [{'path': str(base/(p['model']+'-'+p['card']+'.json')), 'sha256': digest(p)} for p in values]
        all_profiles = copy.deepcopy(values)
        common = {k: {'path': str(base/(k+'.json')), 'sha256': 'a'*64} for k in COMMON}
        non_gemma = [r for r,p in zip(refs,all_profiles) if p['model'] != 'gemma_x_full']
        for p in all_profiles:p.update(common,workspace_root=str(base))
        p = all_profiles[3];p['st4_pre_gemma_intent'] = {'path': str(qout/'pre-gemma-global-closure.json'),
            'expected_non_gemma_profiles': non_gemma, **{k: common[k] for k in COMMON[:3]}}
        qref = {'path': str(base/'queue.json'), 'sha256': 'b'*64}
        pred_refs = [{'path': str(base/f'video-{i}.json'), 'sha256': str(i)*64} for i in range(2)]
        queue = {'output': str(qout), 'profiles': refs[:4], 'global_profiles': refs, 'predecessors': pred_refs,
            'queue_entry': {'path': __file__, 'sha256': 'c'*64}, **common}
        owner,launcher,supervisor = ({'pid': i, 'boot_id': 'synthetic', 'startticks': i} for i in (11,12,13))
        pred_closed = [{'status': 'SYNTHETIC_ORIGINAL_VIDEO_OUTER_CLOSED','path':r['path']} for r in pred_refs]
        native_closed = lambda r,p: {'status':'SYNTHETIC_REAL_PROFILE_CLOSURE','profile':r}
        records = {str(qout/'launch-intent.json'): {'owner_chat':OWNER,'scope':'ST4_ARM_CPU_EVENT_WAIT_ONLY',
            'physical_GPU_lease_held':False,'GPU_queries':0,'plan':qref,'queue_entry':queue['queue_entry'],'launcher':launcher},
            str(qout/'queue-identity.json'): {'plan':qref,'queue_entry':queue['queue_entry'],'launcher':launcher,'owner':owner},
            str(qout/'profile-03-intent.json'): {'ordinal':3,'profile':refs[3],'queue_owner':owner,'automatic_retry':False,
                'pre_gemma_barrier':{'path':str(qout/'pre-gemma-global-closure.json'),'sha256':'d'*64}},
            str(qout/'profile-03-child.json'): {'ordinal':3,'profile':refs[3],'parent':owner,'supervisor':supervisor},
            str(qout/'predecessors-closed.json'): {'status':'PASS_BOTH_T4_VIDEO_QUEUES_REAL_CLOSED','owner_chat':OWNER,
                'predecessors':pred_refs,'closures':pred_closed,'physical_GPU_lease_held':False,'GPU_queries':0,
                **{k:common[k] for k in COMMON[:3]}},
            str(qout/'pre-gemma-global-closure.json'): {'schema':'st4-global-pre-gemma-closure-v1','status':'PASS_ALL_NON_GEMMA_ST4_REAL_CLOSED',
                'owner_chat':OWNER,'expected_non_gemma_profiles':non_gemma,'closures':[native_closed(r,None) for r in non_gemma],
                'physical_GPU_lease_held':False,'GPU_queries':0,**{k:common[k] for k in COMMON[:3]}}}
        globals().update(read=lambda path: records[str(path)], descriptor=lambda path: qref if str(path)==qref['path'] else
            {'path':str(path),'sha256':'d'*64},verified=lambda r:all_profiles[refs.index(r)],validate_plan=lambda path,owner:(queue,all_profiles[:4]),
            lifecycle=types.SimpleNamespace(process_alive=lambda value:value in (owner,launcher,supervisor),owner_format=lambda value:value,
                process_identity=lambda *args:supervisor,profile_closed=native_closed),
            video_queue=types.SimpleNamespace(closed=lambda path,owner:next(c for c in pred_closed if c['path']==path)))
        require(start_gate(p)['status']=='PASS_ST4_ARM_AND_REAL_PREDECESSOR_GATE','Synthetic start gate failed');tests.append('synthetic_exact_ST4_arm_and_six_worker_gate')
        def deny(label, path, key, value):
            old=records[str(path)][key];records[str(path)][key]=value
            try:
                try:start_gate(p)
                except ValueError:tests.append(label)
                else:raise AssertionError('Accepted: '+label)
            finally:records[str(path)][key]=old
        deny('foreign_arm_owner_rejected',qout/'launch-intent.json','owner_chat','foreign')
        deny('video_closure_not_ST4_gemma_gate',qout/'pre-gemma-global-closure.json','status','PASS_ALL_NON_GEMMA_T4_REAL_CLOSED')
        deny('missing_one_new_ST4_worker_rejected',qout/'pre-gemma-global-closure.json','expected_non_gemma_profiles',non_gemma[:-1])
        deny('unknown_supervisor_rejected',qout/'profile-03-child.json','supervisor',{'pid':999})
        deny('predecessor_not_really_closed_rejected',qout/'predecessors-closed.json','closures',[])
    finally:
        globals().update(original)
    return {'status':'PASS_SYNTHETIC_ST4_CPU_QUEUE_ONLY','tests':tests,'SSH_calls':0,'GPU_queries':0,'model_calls':0,'real_entry_writes':0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('validate', 'run', 'launch_once', 'closed', 'self-test'))
    parser.add_argument('--plan'); parser.add_argument('--owner-chat', default=OWNER)
    args = parser.parse_args()
    if args.command == 'self-test':
        print(json.dumps(self_test()));return
    require(args.plan, 'Exact sealed plan path is required')
    value = ({'status': 'PASS_T4_QUEUE_METADATA', 'profiles': len(validate_plan(args.plan, args.owner_chat)[1])}
             if args.command == 'validate' else globals()[args.command](args.plan, args.owner_chat))
    print(json.dumps(value, ensure_ascii=False))


if __name__ == '__main__': main()
