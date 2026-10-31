"""CPU status and one-shot, predecessor-fenced Fourier takeover launch.

Status never imports torch or queries a GPU. Launch resource admission executes
the existing inference.resource_snapshot API in the sealed GPU interpreter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from . import cosmos_x_full_status as generic
from . import gemma_x_full as full
from .refinement_worker import read, require, write_new
from .storage import digest, file_sha256, owner_identity

MODULE = 'classic_precision.gemma_x_fourier_operations'
TAKEOVER = 'classic_precision.gemma_x_fourier_takeover'


def descriptor(path):
    return {'path': str(Path(path).absolute()), 'sha256': file_sha256(path)}


def inside(arc, value):
    path = arc.path(value)
    require(path.resolve().is_relative_to(Path(arc.root).resolve()), 'Evidence lies outside selected root')
    return path


def bound(arc, ref):
    path = inside(arc, ref['path'])
    require(file_sha256(path) == ref['sha256'], 'Bound identity changed: ' + str(path))
    return read(path)


def boot_id():
    path = Path('/proc/sys/kernel/random/boot_id')
    return path.read_text().strip() if path.exists() else None


def observe_owner(owner):
    if owner and owner.get('host') != socket.gethostname():
        return {'status': 'FOREIGN_HOST_SNAPSHOT', 'alive': None, 'identity': owner}
    if not boot_id():
        return {'status': 'HOST_IDENTITY_UNAVAILABLE', 'alive': None, 'identity': owner}
    value = generic.queue_owner_observation(owner, boot_id())
    value['identity'] = owner
    return value


def ledger(arc, p):
    rows, partial = generic.journal(inside(arc, p['output']) / 'events.jsonl')
    passes = [x['sample_id'] for x in rows if x.get('event') == 'TECHNICAL_PASS']
    intents = {x['sample_id'] for x in rows if x.get('event') == 'INTENT'}
    return passes, intents, partial


def ancestor_claims(arc, child, seen=None):
    """Only SHA-bound, cleanly closed completed prefixes can be excluded."""
    seen = set() if seen is None else seen
    if 'continuation_parent' not in child:
        require('continuation_wait' not in child, 'Wait without continuation parent')
        return {}
    ref = child['continuation_parent']
    require(ref['sha256'] not in seen, 'Cyclic continuation ancestry')
    seen.add(ref['sha256'])
    parent = bound(arc, ref)
    parent_digest = digest(parent)
    wait = bound(arc, child['continuation_wait'])
    out = inside(arc, parent['output'])
    require(arc.path(child['continuation_wait']['path']) == out.with_suffix('.wait.json'),
            'Continuation wait belongs to another worker')
    require(parent['workspace_root'] == child['workspace_root']
            and parent['card'] == child['card'] and parent['claim_root'] == child['claim_root']
            and parent['gpu_uuid'] == child['gpu_uuid']
            and parent['physical_lock'] == child['physical_lock']
            and child['control_generation'] == parent['control_generation'] + 1,
            'Continuation changed card, claim namespace, lease or generation')
    require(wait['returncode'] == 0 and wait['real_wait_observed']
            and wait['owned_engines_exited'] and wait['original_physical_lease_released']
            and wait['profile_sha256'] == parent_digest, 'Ancestor did not safely close')
    result = read(out / 'result.json')
    passes, intents, partial = ledger(arc, parent)
    done = set(passes)
    require(not partial and len(done) == len(passes) and done <= intents
            and result['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD', 'COMPLETE')
            and wait['terminal_status'] == result['status'] and not result.get('UNCERTAIN_ids')
            and not (out / 'failure.json').exists()
            and passes == result['completed_ids'] == wait['completed_ids']
            and passes == parent['assigned_ids'][:len(passes)], 'Ancestor durable PASS/prefix differs')
    require(child['assigned_ids'] == [s for s in parent['assigned_ids'] if s not in done],
            'Continuation is not the exact unclaimed parent suffix')
    known = ancestor_claims(arc, parent, seen)
    require(not done.intersection(known), 'Ancestor completed IDs overlap')
    for sid in done:
        claim = read(inside(arc, parent['claim_root']) / (sid + '.json'))
        require(claim['sample_id'] == sid and claim['profile_sha256'] == parent_digest
                and claim['worker_token'] == parent['worker_token']
                and claim.get('output', parent['output']) == parent['output'], 'Ancestor claim identity changed')
        known[sid] = ref
    return known


def outer_paths(arc, queue_path, queue):
    p = bound(arc, queue['profiles'][0])
    base = arc.path(p['output']).parent / Path(queue_path).stem
    return {k: base.with_suffix('.' + k + '.json')
            for k in ('launch-intent', 'owner', 'launch', 'wait')}


def inspect_outer(arc, queue_path):
    path = arc.path(queue_path); q = read(path)
    require(q['status'] == 'SEALED' and q['automatic_retry'] is False, 'Unsealed status queue')
    for ref in q['profiles']: bound(arc, ref)
    paths = outer_paths(arc, path, q)
    return {'queue': descriptor(path), 'profiles': q['profiles'],
            'attempted': any(p.exists() for p in paths.values()) or arc.path(q['state_path']).exists(),
            'owner': observe_owner(read(paths['owner']) if paths['owner'].exists() else None),
            'wait': read(paths['wait']) if paths['wait'].exists() else None}


def status(root, profiles, queues=()):
    arc = full.Archive(root); workers = []
    require(profiles, 'Explicit profiles required')
    require(len(set(map(str, profiles))) == len(profiles), 'Duplicate status profile')
    for path in profiles:
        path = arc.path(path); p = read(path); expected = set(p['assigned_ids'])
        profile_digest = digest(p)
        inside(arc, path)
        require(len(expected) == len(p['assigned_ids']) == p['maximum_model_calls'], 'Invalid allocation')
        known = ancestor_claims(arc, p)
        passes, intents, partial = ledger(arc, p); done = set(passes)
        claims = {x.stem for x in inside(arc, p['claim_root']).glob('*.json')}
        scoped = claims - set(known); issues = []
        if (done | intents | scoped) - expected: issues.append('FOREIGN_ID_IN_LEDGER')
        if len(passes) != len(done): issues.append('DUPLICATE_TECHNICAL_PASS')
        if not done <= scoped: issues.append('COMPLETED_ID_WITHOUT_CLAIM')
        if not done <= intents: issues.append('COMPLETED_ID_WITHOUT_INTENT')
        for sid in scoped & expected:
            claim = read(arc.path(p['claim_root']) / (sid + '.json'))
            if (claim.get('sample_id') != sid or claim.get('profile_sha256') != profile_digest
                    or claim.get('worker_token') != p['worker_token']
                    or claim.get('output', p['output']) != p['output']): issues.append('CLAIM_IDENTITY_MISMATCH')
        out = inside(arc, p['output']); wait_path = out.with_suffix('.wait.json')
        wait = read(wait_path) if wait_path.exists() else None
        if wait:
            if not (wait.get('returncode') == 0 and wait.get('real_wait_observed')
                    and wait.get('owned_engines_exited') and wait.get('original_physical_lease_released')
                    and wait.get('profile_sha256') == profile_digest):
                issues.append('WAIT_OR_PROFILE_OR_EXIT_OR_LEASE_NOT_SUCCESSFUL')
            if scoped != done: issues.append('TERMINAL_CLAIMS_WITHOUT_PASS')
            if wait.get('terminal_status') == 'COMPLETE' and done != expected:
                issues.append('COMPLETE_WITH_INCOMPLETE_ID_LEDGER')
        if (out / 'failure.json').exists(): issues.append('WORKER_FAILURE_RETAINED')
        catalog = p['source_catalog']
        catalog_ok = arc.path(catalog['path']).exists() and file_sha256(arc.path(catalog['path'])) == catalog['sha256']
        if not catalog_ok: issues.append('SOURCE_CATALOG_IDENTITY_MISMATCH')
        inflight = sorted(scoped - done)
        if len(inflight) > 1: issues.append('MORE_THAN_ONE_INFLIGHT_CLAIM')
        window = generic.timing_window(out / 'request-timings.jsonl', 20)
        progress = read(out / 'progress.json') if (out / 'progress.json').exists() else {}
        state = ('FAILED_OR_NEEDS_REVIEW' if issues else
                 ('CLOSED_' + wait['terminal_status'] if wait else progress.get('status', 'STARTING')))
        actual_host = p['hostname'] == socket.gethostname()
        if not actual_host and not wait and not issues:
            state = 'ARCHIVED_SNAPSHOT_NOT_LIVE'
        originals = generic.inspect_profile(path)
        workers.append({'worker': p['worker'], 'card': p['card'], 'status': state,
            'profile': descriptor(path), 'actual_host_observation': actual_host,
            'expected': len(expected), 'technical_pass_count': len(done),
            'claimed': len(scoped), 'ancestor_completed_claims_excluded': len(known),
            'claimed_without_pass': inflight, 'one_inflight': len(inflight) == 1,
            'remaining': len(expected - done), 'unattempted': len(expected - scoped),
            'partial_event_tail': partial, 'recent_ordinary': window,
            'eta_remaining_seconds': len(expected - done) * window['mean_seconds']
                if actual_host and not wait and not issues and window.get('n') else None,
            'wait': wait, 'issues': sorted(set(issues)), 'generic_original': originals})
    selected = list(queues)
    if not selected:
        for path in profiles:
            p = read(arc.path(path)); out = arc.path(p['output'])
            for stem in (p['worker'] + '-queue', p['card'] + '-full-queue'):
                launch = out.parent / (stem + '.launch.json')
                if launch.exists():
                    ref = read(launch)['queue']; q = bound(arc, ref)
                    if any(x['sha256'] == file_sha256(arc.path(path)) for x in q['profiles']):
                        selected.append(ref['path'])
    outers = [inspect_outer(arc, q) for q in dict.fromkeys(selected)]
    for worker in workers:
        matches = [q for q in outers if any(ref['sha256'] == worker['profile']['sha256']
                   and arc.path(ref['path']) == Path(worker['profile']['path']) for ref in q['profiles'])]
        worker['outer_queues'] = matches
        if worker['wait']: continue
        if len(matches) == 1 and not matches[0]['attempted'] and not worker['issues']:
            worker['status'] = 'NOT_LAUNCHED'
        elif worker['actual_host_observation']:
            if len(matches) != 1:
                if not matches and worker['claimed'] == worker['technical_pass_count'] == 0 and not worker['issues']:
                    worker['status'] = 'NOT_LAUNCHED'
                else: worker['issues'].append('MISSING_OR_DUPLICATE_QUEUE_OWNER')
            elif matches[0]['owner']['alive'] is not True:
                worker['issues'].append('QUEUE_OWNER_' + matches[0]['owner']['status'])
            if worker['issues']:
                worker['status'] = 'FAILED_OR_NEEDS_REVIEW'; worker['eta_remaining_seconds'] = None
    return {'schema': 'gemma-fourier-scoped-status-v1', 'workers': workers,
            'host': socket.gethostname(), 'boot_id': boot_id(),
            'observed_monotonic_seconds': time.monotonic(), 'observed_unix': time.time(),
            'queues': outers,
            'model_calls': 0, 'mutations': 0, 'GPU_queries': 0}


def queue_profiles(arc, path):
    path = arc.path(path); q = read(path)
    require(q['status'] == 'SEALED' and q['automatic_retry'] is False and q['profiles'], 'Not a finite sealed queue')
    ps = [bound(arc, ref) for ref in q['profiles']]
    require(len({p['card'] for p in ps}) == len(ps), 'Repeated card in launch queue')
    for p in ps:
        require(p['hostname'] == socket.gethostname() and Path(p['workspace_root']).absolute() == Path(arc.root).absolute(),
                'Launch must use the actual host and canonical root')
    return path, q, ps


def predecessor_gate(arc, successor, predecessor):
    from . import gemma_x_fourier_takeover as takeover
    path, q, ps = queue_profiles(arc, successor)
    old_path, old_q, old_ps = queue_profiles(arc, predecessor)
    require(len(ps) == len(old_ps) == 1, 'Launch one physical card independently')
    p, old = ps[0], old_ps[0]
    require(p['card'] == old['card'] and 'fourier_takeover_adapter' in p
            and p['takeover_predecessor']['sha256'] == old_q['profiles'][0]['sha256']
            and arc.path(p['takeover_predecessor']['path']) == arc.path(old_q['profiles'][0]['path']),
            'Successor/predecessor queue profile mapping differs')
    takeover.predecessor_closed(p)
    paths = outer_paths(arc, old_path, old_q)
    wait = read(paths['wait']); owner = read(paths['owner']); launch = read(paths['launch'])
    old_ref = descriptor(old_path)
    require(wait['returncode'] == 0 and wait['real_wait_observed']
            and wait['queue_sha256'] == old_ref['sha256']
            and arc.path(wait['queue']) == old_path
            and launch['queue']['sha256'] == old_ref['sha256']
            and arc.path(launch['queue']['path']) == old_path, 'Predecessor outer wait/queue identity differs')
    require(observe_owner(owner)['alive'] is False, 'Predecessor outer owner has not demonstrably exited')
    for key in ('pid', 'boot_id'):
        require(launch.get(key) is not None and launch.get(key) == owner.get(key),
                'Predecessor outer owner/launch identity differs')
    ticks = [launch[k] for k in ('process_start_ticks', 'start_ticks') if k in launch]
    require(ticks and owner.get('process_start_ticks') is not None
            and all(str(x) == str(owner['process_start_ticks']) for x in ticks),
            'Predecessor outer start ticks missing or inconsistent')
    state = read(arc.path(old_q['state_path']))
    require(state['status'] == 'COMPLETE' and state['plan']['sha256'] == old_ref['sha256']
            and arc.path(state['plan']['path']) == old_path, 'Predecessor queue is not COMPLETE')
    require(len(state['workers']) == 1
            and state['workers'][0]['profile'] == old_q['profiles'][0]
            and state['workers'][0]['returncode'] == 0
            and state['workers'][0]['real_supervisor_wait'] is True
            and state['workers'][0]['terminal_status'] == 'COMPLETE', 'Predecessor queue worker mapping differs')
    require(state['owner']['pid'] == wait['child_pid']
            and observe_owner(state['owner'])['alive'] is False,
            'Predecessor inner queue owner has not exited or differs from waited child')
    return path, q, p


def environment(p):
    env = dict(os.environ)
    for name in p.get('cleared_environment', []): env.pop(name, None)
    env.update(p['environment'])
    code = Path(p['workspace_root']) / 'code/vlm-psychophysics'
    require(code.is_dir() and env.get('PYTHONPATH', '').split(os.pathsep)[0] == str(code), 'Wrong sealed code root/PYTHONPATH')
    return env, code


def resource_preflight(p):
    env, cwd = environment(p)
    env['CUDA_VISIBLE_DEVICES'] = p['gpu_uuid']
    code = ('import json,sys,torch; from classic_precision.inference import resource_snapshot; '
            'from classic_precision.refinement_worker import require; '
            'p=json.load(sys.stdin); v=resource_snapshot(torch,p["resources"]); '
            'require(torch.cuda.device_count()==1,"Expected one physical GPU"); '
            'require(str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix("gpu-") == '
            'p["gpu_uuid"].lower().removeprefix("gpu-"),"Wrong admission GPU"); print(json.dumps(v))')
    result = subprocess.run([p['python'], '-B', '-c', code], input=json.dumps(p),
                            text=True, capture_output=True, env=env, cwd=cwd, check=True, timeout=60)
    value = json.loads(result.stdout.strip().splitlines()[-1])
    require(value['admitted'], 'Existing resource_snapshot thresholds not met')
    return value


def launch(root, queue, predecessor_queue):
    arc = full.Archive(root); path, q, p = predecessor_gate(arc, queue, predecessor_queue)
    paths = outer_paths(arc, path, q)
    require(not any(x.exists() for x in paths.values()) and not arc.path(q['state_path']).exists(), 'Queue already attempted; no retry')
    admission = resource_preflight(p)
    intent = {'status': 'INTENT_EXCLUSIVE_NO_RETRY', 'queue': descriptor(path),
              'predecessor_queue': descriptor(arc.path(predecessor_queue)),
              'operations': descriptor(__file__), 'caller': owner_identity(), 'resource_admission': admission}
    write_new(paths['launch-intent'], intent)
    from . import gemma_x_fourier_takeover as takeover
    takeover.dispatch('activate', str(path))
    env, cwd = environment(p)
    with paths['launch'].with_suffix('.log').open('xb') as log:
        child = subprocess.Popen([p['python'], '-B', '-m', MODULE, 'queue-supervisor',
            '--root', str(root), '--queue', str(path), '--predecessor-queue', str(predecessor_queue),
            '--intent-sha256', file_sha256(paths['launch-intent'])], env=env, cwd=cwd,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    value = {'status': 'LAUNCHED_ONCE', 'queue': descriptor(path), 'pid': child.pid,
             'operations': descriptor(__file__), 'python': p['python'], 'automatic_retry': False}
    value.update(boot_id=boot_id(), process_start_ticks=
                 Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19])
    write_new(paths['launch'], value)
    return value


def queue_supervisor(root, queue, predecessor_queue, intent_sha256):
    arc = full.Archive(root); path, q, p = predecessor_gate(arc, queue, predecessor_queue)
    paths = outer_paths(arc, path, q)
    require(file_sha256(paths['launch-intent']) == intent_sha256, 'Launch intent changed')
    intent = read(paths['launch-intent'])
    require(intent['queue'] == descriptor(path) and intent['operations']['sha256'] == file_sha256(__file__)
            and intent['predecessor_queue'] == descriptor(arc.path(predecessor_queue)), 'Intent source/queue binding differs')
    require(Path(sys.executable).resolve() == Path(p['python']).resolve(), 'Supervisor must use sealed GPU interpreter')
    write_new(paths['owner'], owner_identity())
    return supervise_child(p, path, paths['wait'])


def supervise_child(p, queue_path, wait_path):
    """A real Popen.wait(), never a synthetic terminal receipt or retry."""
    env, cwd = environment(p); started = time.monotonic()
    child = subprocess.Popen([p['python'], '-B', '-m', TAKEOVER, 'queue', '--plan', str(queue_path)],
                             env=env, cwd=cwd, stdin=subprocess.DEVNULL)
    code = child.wait()
    value = {'queue': str(queue_path), 'queue_sha256': file_sha256(queue_path),
             'operations_sha256': file_sha256(__file__), 'owner': owner_identity(),
             'child_pid': child.pid, 'returncode': code, 'real_wait_observed': True,
             'elapsed_monotonic_seconds': time.monotonic() - started, 'automatic_retry': False}
    write_new(wait_path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('status', 'launch', 'queue-supervisor'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--profile', action='append', default=[])
    parser.add_argument('--queue', action='append', default=[])
    parser.add_argument('--predecessor-queue')
    parser.add_argument('--intent-sha256')
    args = parser.parse_args()
    if args.command == 'status': value = status(args.root, args.profile, args.queue)
    else:
        require(len(args.queue) == 1 and args.predecessor_queue, 'Exactly one successor/predecessor queue required')
        value = (launch(args.root, args.queue[0], args.predecessor_queue) if args.command == 'launch'
                 else queue_supervisor(args.root, args.queue[0], args.predecessor_queue, args.intent_sha256))
    print(json.dumps(value), flush=True)
    if args.command == 'queue-supervisor': sys.exit(value['returncode'])


if __name__ == '__main__': main()
