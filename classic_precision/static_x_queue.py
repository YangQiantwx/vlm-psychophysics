"""Single-launch CPU queues for one sealed static qualification/full phase.

Only the root controller creates and launches plans. The queue calls each
already-bound supervisor, observes real subprocess waits, and stops on any
failure or uncertainty. It never queries a GPU, retries, resumes, or kills.
"""
from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import errno
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import socket
import select
import struct
import subprocess
import sys
import time

from .refinement_worker import descriptor, read, require, verified
from .storage import atomic_json, digest, file_sha256, _mkdir_durable, _sync_directory

MODULE = 'classic_precision.static_x_queue'
SUPERVISORS = {'qualification': 'classic_precision.static_x_admission',
               'full': 'classic_precision.static_x_operations'}
MODEL_COUNTS = {'cosmos': 1, 'qwen_x_qsa': 2, 'nemo': 1, 'gemma_x_full': 1}
MODEL_ORDER = ['cosmos', 'qwen_x_qsa', 'qwen_x_qsa', 'nemo', 'gemma_x_full']


def durable_new(path, value):
    path = Path(path)
    _mkdir_durable(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0), 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    _sync_directory(path.parent)


def process_identity(pid=None):
    """Read only our own or a known directly-created child's /proc identity."""
    require(sys.platform.startswith('linux'), 'Live queue execution requires Linux process identities')
    pid = os.getpid() if pid is None else int(pid)
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'host': socket.gethostname(), 'pid': pid, 'startticks': fields[19],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def process_alive(identity):
    require(identity['host'] == socket.gethostname(), 'Queue identity belongs to another host')
    if identity['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return False
    path = Path('/proc') / str(identity['pid']) / 'stat'
    try:
        fields = path.read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return False
    return fields[19] == str(identity['startticks']) and fields[0] != 'Z'


def native_pidfd_open(pid, libc=None):
    """Use the stdlib binding, or the same Linux API exposed by libc."""
    native = getattr(os, 'pidfd_open', None)
    if native is not None:
        return native(pid, 0)
    libc = ctypes.CDLL(None, use_errno=True) if libc is None else libc
    native = getattr(libc, 'pidfd_open', None)
    require(native is not None, 'Linux libc has no pidfd_open; no polling fallback')
    native.argtypes = [ctypes.c_int, ctypes.c_uint]
    native.restype = ctypes.c_int
    fd = native(int(pid), 0)
    if fd < 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    return fd


def queue_lock_path(physical_lock, gpu_uuid):
    uuid = gpu_uuid.lower().removeprefix('gpu-')
    require(uuid and all(c in '0123456789abcdef-' for c in uuid), 'GPU UUID is not a stable filename identity')
    return Path(physical_lock['path']).with_name('static-x-cpu-queue-' + uuid + '.lock')


def source_path(module):
    spec = importlib.util.find_spec(module)
    require(spec is not None and spec.origin, 'Supervisor source cannot be resolved')
    return Path(spec.origin)


def validate_plan(plan_path, owner_chat):
    plan = read(plan_path)
    require(plan['schema'] == 'static-x-finite-queue-v1' and plan['owner_chat'] == owner_chat
            and owner_chat and plan['phase'] in SUPERVISORS, 'Unknown phase or foreign queue owner')
    require(plan['queue_entry'] == descriptor(__file__), 'Queue source differs from the sealed plan')
    refs = plan['profiles']; require(len(refs) == 5, 'Each card phase must have exactly five finite model/task workers')
    require(len({x['path'] for x in refs}) == len(refs), 'Duplicate profile in queue')
    supervisor = SUPERVISORS[plan['phase']]
    expected_sha = file_sha256(source_path(supervisor))
    # These are existing metadata-only validators, not model/processor calls.
    policy = importlib.import_module(supervisor)
    common_fields = ['static_authorization', 'source_catalog', 'static_protocol']
    if plan['phase'] == 'full': common_fields += ['full_authorization', 'qualification_summary']
    profiles, allocations, qwen_tasks = [], [], []
    for ref in refs:
        require(ref['supervisor_module'] == supervisor and ref['supervisor_sha256'] == expected_sha,
                'Profile supervisor/phase source SHA differs')
        p = verified(ref)
        require(all(plan[field] == p[field] for field in common_fields), 'Queue mixes authorization/catalog/protocol/qualification refs')
        require(p['gpu_uuid'] == plan['gpu_uuid'] and p['static_card'] == plan['card']
                and (p.get('card') is None or p['card'] == plan['card'])
                and p['execution_policy'] == {'module': supervisor, 'source_sha256': expected_sha}
                and p['automatic_retry'] is False and p['formal_auto_resume'] is False,
                'Profile belongs to another card, phase, policy or retry mode')
        require(Path(p['python']).is_absolute() and p['assigned_ids']
                and len(p['assigned_ids']) == len(set(p['assigned_ids'])) == p['maximum_model_calls'],
                'Profile is not an explicit finite allocation')
        require(file_sha256(p['implementation']['path']) == p['implementation']['sha256'], 'Base implementation changed')
        auth, allocation = policy.authorization(p)
        allocations.append(digest(allocation))
        if p['model'] == 'qwen_x_qsa':
            catalog = policy.configuration(p)
            own_ids = allocation.get('normal_ids', p['assigned_ids'])
            qwen_tasks.append(frozenset(catalog['items'][sid]['task'] for sid in own_ids))
        profiles.append(p)
    require([p['model'] for p in profiles] == MODEL_ORDER
            and qwen_tasks == [frozenset({'T1'}), frozenset({'T2', 'T3'})],
            'Required order is Cosmos, Qwen T1, Qwen T23, Nemo, then Gemma')
    require(len({p['python'] for p in profiles}) == 1, 'All queue workers must share their sealed interpreter')
    expected_allocations = [digest(x) for x in auth['allocations'] if x['gpu_uuid'] == plan['gpu_uuid']]
    require(len(allocations) == len(set(allocations)) == len(expected_allocations) == 5
            and set(allocations) == set(expected_allocations)
            and set(qwen_tasks) == {frozenset({'T1'}), frozenset({'T2', 'T3'})},
            'Queue must cover its common authorization exactly once, including both distinct Qwen arms')
    physical = profiles[0]['physical_lock']
    require(all(p['physical_lock'] == physical for p in profiles), 'Queue profiles have different physical leases')
    expected_lock = queue_lock_path(physical, plan['gpu_uuid'])
    lock = plan['queue_lease']; require(lock['path'] == str(expected_lock), 'CPU queue lease is not globally stable for this GPU')
    require(Path(lock['path']).is_file() and not Path(lock['path']).is_symlink(), 'CPU queue lease must already be a regular file')
    stat = Path(lock['path']).stat()
    require((stat.st_dev, stat.st_ino) == (lock['st_dev'], lock['st_ino']), 'CPU queue lease inode changed')
    require(Path(plan['output']).is_absolute(), 'Queue output must be explicit and absolute')
    require(len({p['output'] for p in profiles}) == len(profiles), 'Queue profiles share output')
    # Child cwd follows the already-sealed package, not a new runtime/environment.
    roots = {str(Path(p['implementation']['path']).parent.parent) for p in profiles}
    require(len(roots) == 1 and source_path(supervisor).parent.parent == Path(next(iter(roots))),
            'CPU queue and worker supervisor resolve different code roots')
    validate_pre_gemma(plan, policy, common_fields, auth)
    return plan, profiles


def ref_identity(ref):
    return tuple(ref[key] for key in ('path', 'sha256', 'supervisor_module', 'supervisor_sha256'))


def validate_pre_gemma(plan, policy, common_fields, auth):
    timeout = plan['gemma_barrier_timeout_seconds']
    require(isinstance(timeout, (int, float)) and not isinstance(timeout, bool)
            and math.isfinite(timeout) and timeout > 0, 'Gemma barrier needs a positive finite deadline')
    expected = [x for x in auth['allocations'] if x['model'] != 'gemma']
    require(len(expected) == 8 and len({x['gpu_uuid'] for x in expected}) == 2,
            'This phase barrier is limited to the exact two-Fourier authorization')
    direct, deferred = 'pre_gemma_profiles' in plan, 'pre_gemma_manifest_intent' in plan
    require(direct != deferred, 'Choose exactly one direct or deferred Gemma barrier binding')
    if direct:
        validate_pre_gemma_refs(plan, plan['pre_gemma_profiles'], policy, common_fields, auth)
    else:
        intent = plan['pre_gemma_manifest_intent']
        require(intent['schema'] == 'static-x-pre-gemma-manifest-intent-v1'
                and Path(intent['path']).is_absolute() and intent['owner_chat'] == plan['owner_chat']
                and all(intent[field] == plan[field] for field in common_fields)
                and intent['expected_allocation_digests'] == sorted(digest(x) for x in expected),
                'Deferred Gemma intent does not bind all exact authorized non-Gemma roles')
    outputs = plan['phase_queue_outputs']
    require(set(outputs) == {x['gpu_uuid'] for x in expected}
            and len(set(outputs.values())) == 2 and all(Path(x).is_absolute() for x in outputs.values())
            and outputs[plan['gpu_uuid']] == plan['output'], 'Exact known phase queue outputs are missing')


def validate_pre_gemma_refs(plan, refs, policy, common_fields, auth):
    require(len(refs) == len({ref_identity(x) for x in refs}) == 8, 'Global pre-Gemma barrier requires exactly eight unique profiles')
    expected = [x for x in auth['allocations'] if x['model'] != 'gemma']
    found, tasks = [], {}
    for ref in refs:
        require(ref['supervisor_module'] == SUPERVISORS[plan['phase']]
                and ref['supervisor_sha256'] == file_sha256(source_path(ref['supervisor_module'])),
                'Global barrier supervisor source changed')
        p = verified(ref)
        require(p['model'] != 'gemma_x_full' and all(p[field] == plan[field] for field in common_fields)
                and p['execution_policy'] == {'module': ref['supervisor_module'], 'source_sha256': ref['supervisor_sha256']},
                'Global barrier mixes phases, authorization, source or policy')
        _, allocation = policy.authorization(p)
        found.append(digest(allocation))
        if p['model'] == 'qwen_x_qsa':
            catalog = policy.configuration(p)
            own = allocation.get('normal_ids', p['assigned_ids'])
            tasks.setdefault(p['gpu_uuid'], set()).add(frozenset(catalog['items'][sid]['task'] for sid in own))
    require(len(set(found)) == 8 and set(found) == {digest(x) for x in expected}
            and all(value == {frozenset({'T1'}), frozenset({'T2', 'T3'})} for value in tasks.values())
            and len(tasks) == 2, 'Global barrier does not exactly cover both cards non-Gemma allocations')
    require({ref_identity(x) for x in plan['profiles'][:-1]} <= {ref_identity(x) for x in refs},
            'Own preceding models are absent from the global Gemma barrier')


def barrier_binding(plan):
    if 'pre_gemma_profiles' in plan:
        return {'direct': sorted(ref_identity(x) for x in plan['pre_gemma_profiles'])}
    return {'deferred': plan['pre_gemma_manifest_intent']}


def resolve_pre_gemma(plan):
    if 'pre_gemma_profiles' in plan:
        return plan['pre_gemma_profiles'], None
    intent = plan['pre_gemma_manifest_intent']; path = Path(intent['path'])
    if not path.exists(): return None, None
    require(path.is_file() and not path.is_symlink(), 'HOLD: global manifest is not a regular file')
    before = path.stat(); data = path.read_bytes(); after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'HOLD: global manifest changed during its initial byte read')
    if not data.endswith(b'\n'): return None, None
    manifest = json.loads(data)
    require(isinstance(manifest, dict), 'HOLD: global manifest is not a JSON object')
    manifest_ref = {'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest()}
    fields = ['owner_chat', 'static_authorization', 'source_catalog', 'static_protocol']
    if plan['phase'] == 'full': fields += ['full_authorization', 'qualification_summary']
    require(manifest['schema'] == 'static-x-global-pre-gemma-profiles-v1' and manifest['phase'] == plan['phase']
            and all(manifest[field] == intent[field] for field in fields)
            and manifest['expected_allocation_digests'] == intent['expected_allocation_digests'],
            'HOLD: sealed global non-Gemma manifest differs from its original intent')
    policy = importlib.import_module(SUPERVISORS[plan['phase']])
    common_fields = ['static_authorization', 'source_catalog', 'static_protocol']
    if plan['phase'] == 'full': common_fields += ['full_authorization', 'qualification_summary']
    validate_pre_gemma_refs(plan, manifest['profiles'], policy, common_fields, verified(plan['static_authorization']))
    require(descriptor(intent['path']) == manifest_ref, 'HOLD: global non-Gemma manifest changed while reading')
    return manifest['profiles'], manifest_ref


def optional_json(path):
    path = Path(path)
    if not path.exists(): return None
    require(path.is_file() and not path.is_symlink(), 'Peer receipt is not a regular file: ' + str(path))
    data = path.read_bytes()
    if not data.endswith(b'\n'): return None  # Exclusive supervisor receipt may still be writing.
    value = json.loads(data)
    require(isinstance(value, dict), 'Peer receipt is not a complete JSON object')
    return value


def owner_format(identity):
    value = {'host': identity['host'], 'pid': identity['pid'], 'boot_id': identity['boot_id'],
             'startticks': identity.get('startticks', identity.get('process_start_ticks'))}
    require(value['startticks'] is not None and value['boot_id'], 'Known peer owner lacks startticks/boot identity')
    return value


def gemma_barrier_snapshot(plan, refs):
    closures, pending, identities = [], [], []
    for gpu_uuid, output in plan['phase_queue_outputs'].items():
        out = Path(output)
        require(not (out / 'launcher-failure.json').exists(), 'HOLD: peer queue launcher failed: ' + output)
        result = optional_json(out / 'queue-result.json')
        require(result is None or result['status'] == 'COMPLETE', 'HOLD: peer phase queue failed or is uncertain: ' + output)
        launch = optional_json(out / 'launch-intent.json')
        identity = optional_json(out / 'queue-identity.json')
        if launch:
            peer = verified(launch['plan'])
            fields = ['owner_chat', 'phase', 'static_authorization', 'source_catalog', 'static_protocol', 'queue_entry', 'phase_queue_outputs']
            if plan['phase'] == 'full': fields += ['full_authorization', 'qualification_summary']
            require(peer['gpu_uuid'] == gpu_uuid and peer['output'] == output
                    and all(peer[field] == plan[field] for field in fields)
                    and barrier_binding(peer) == barrier_binding(plan),
                    'HOLD: known peer queue belongs to another owner/phase/authorization')
        if identity:
            require(launch is not None and identity['plan'] == launch['plan'], 'HOLD: peer queue identity lacks its original intent')
            owner = owner_format(identity['owner'])
            alive = process_alive(owner)
            require(alive or result is not None, 'HOLD: known peer queue exited without a terminal result')
            if alive: identities.append(owner)
    for ref in refs:
        p = verified(ref); out = Path(p['output'])
        require(not (out / 'failure.json').exists(), 'HOLD: non-Gemma peer failed: ' + ref['path'])
        wait = optional_json(out.with_suffix('.wait.json'))
        if wait is not None:
            closures.append(profile_closed(ref, p))
            continue
        result = optional_json(out / 'result.json')
        require(result is None or result['status'] == 'COMPLETE', 'HOLD: non-Gemma peer is not COMPLETE: ' + ref['path'])
        launch = optional_json(out.with_suffix('.launch.json'))
        if launch:
            require(launch['profile_sha256'] == digest(p), 'HOLD: peer launch belongs to a different profile')
            owner = owner_format(launch['owner'])
            require(process_alive(owner), 'HOLD: peer supervisor exited without a real successful wait')
            identities.append(owner)
        pending.append({'path': ref['path'], 'sha256': ref['sha256']})
    return closures, pending, identities


class PhaseEvents:
    """Linux filesystem events plus pidfds for only the known owned peers."""
    MASK = 0x00000008 | 0x00000080 | 0x00000100 | 0x00000200 | 0x00000400 | 0x00000800
    def __init__(self, paths):
        require(sys.platform.startswith('linux'), 'Barrier requires Linux inotify/pidfd, no polling fallback')
        self.libc = ctypes.CDLL(None, use_errno=True)
        self.libc.inotify_init1.argtypes = [ctypes.c_int]; self.libc.inotify_init1.restype = ctypes.c_int
        self.libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        self.libc.inotify_add_watch.restype = ctypes.c_int
        self.libc.inotify_rm_watch.argtypes = [ctypes.c_int, ctypes.c_int]
        self.fd = self.libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        require(self.fd >= 0, 'Cannot initialize filesystem barrier events')
        self.paths = list(map(Path, paths)); self.watches = {}; self.names = {}; self.pidfds = {}
    def __enter__(self): return self
    def __exit__(self, *_):
        for fd in self.pidfds.values(): os.close(fd)
        os.close(self.fd)
    def refresh(self, identities):
        desired = {}
        for path in self.paths:
            folder = path.parent
            while not folder.is_dir(): folder = folder.parent
            desired.setdefault(str(folder), set()).add(path.relative_to(folder).parts[0])
        for folder in list(self.watches):
            if folder not in desired:
                wd = self.watches.pop(folder); self.libc.inotify_rm_watch(self.fd, wd)
                self.names.pop(wd, None)
        for folder, names in desired.items():
            if folder not in self.watches:
                wd = self.libc.inotify_add_watch(self.fd, os.fsencode(folder), self.MASK)
                require(wd >= 0, 'Cannot watch known phase receipt directory: ' + folder)
                self.watches[folder] = wd
            self.names[self.watches[folder]] = names
        wanted = {(x['pid'], str(x['startticks']), x['boot_id']): x for x in identities}
        for key in list(self.pidfds):
            if key not in wanted: os.close(self.pidfds.pop(key))
        for key, identity in wanted.items():
            if key not in self.pidfds:
                try: fd = native_pidfd_open(identity['pid'], self.libc)
                except OSError as error:
                    if error.errno == errno.ESRCH: return False
                    raise
                if not process_alive(identity): os.close(fd); return False
                self.pidfds[key] = fd
        return True
    def wait(self, deadline):
        while True:
            remaining = deadline - time.monotonic()
            require(remaining > 0, 'HOLD: Gemma-last barrier deadline expired')
            ready, _, _ = select.select([self.fd, *self.pidfds.values()], [], [], remaining)
            require(ready, 'HOLD: Gemma-last barrier deadline expired')
            if any(fd != self.fd for fd in ready): return
            relevant = False
            while True:
                try: data = os.read(self.fd, 65536)
                except BlockingIOError: break
                require(data, 'Filesystem event descriptor unexpectedly closed')
                offset = 0
                while offset < len(data):
                    wd, mask, _, size = struct.unpack_from('iIII', data, offset)
                    name = os.fsdecode(data[offset + 16:offset + 16 + size].split(b'\0', 1)[0])
                    offset += 16 + size
                    if mask & (0x00000400 | 0x00000800 | 0x00008000):
                        for folder, watched in list(self.watches.items()):
                            if watched == wd: self.watches.pop(folder)
                        self.names.pop(wd, None)
                        relevant = True
                    elif mask & 0x00004000: relevant = True
                    elif name in self.names.get(wd, set()): relevant = True
            if relevant: return


def gemma_barrier(plan, plan_ref, output):
    paths = []
    if 'pre_gemma_manifest_intent' in plan: paths.append(Path(plan['pre_gemma_manifest_intent']['path']))
    for output_root in plan['phase_queue_outputs'].values():
        paths += [Path(output_root) / name for name in ('launch-intent.json', 'queue-identity.json',
                  'queue-result.json', 'launcher-failure.json')]
    started = time.monotonic(); deadline = started + plan['gemma_barrier_timeout_seconds']
    refs, manifest_ref, profile_paths_added = None, None, False
    with PhaseEvents(paths) as events:
        events.refresh([])  # Install filesystem watches before the first snapshot.
        while True:
            require(descriptor(__file__) == plan['queue_entry'], 'Queue source changed during barrier')
            if refs is None:
                refs, manifest_ref = resolve_pre_gemma(plan)
            elif manifest_ref is not None:
                require(descriptor(manifest_ref['path']) == manifest_ref, 'HOLD: bound non-Gemma manifest changed during barrier')
            if refs is not None and not profile_paths_added:
                for ref in refs:
                    out = Path(verified(ref)['output'])
                    events.paths += [out.with_suffix('.wait.json'), out.with_suffix('.launch.json')]
                    events.paths += [out / name for name in ('result.json', 'failure.json', 'exit.json', 'lease.json')]
                events.refresh([])  # Newly known peer files are watched before their snapshot.
                profile_paths_added = True
            closures, pending, identities = gemma_barrier_snapshot(plan, refs or [])
            if refs is None:
                pending.append({'awaiting_global_manifest': plan['pre_gemma_manifest_intent']['path']})
            if not pending:
                receipt = {'schema': 'static-x-global-gemma-last-barrier-v1', 'status': 'PASS',
                    'plan': plan_ref, 'static_authorization': plan['static_authorization'],
                    'pre_gemma_profiles': refs, 'global_manifest': manifest_ref, 'closures': closures,
                    'seconds': time.monotonic() - started, 'physical_GPU_lease_held': False,
                    'waiting_method': 'inotify filtered known receipt names + known-owner pidfds', 'GPU_queries': 0}
                durable_new(Path(output) / 'pre-gemma-barrier.json', receipt)
                return receipt
            atomic_json(Path(output) / 'progress.json', {'status': 'WAITING_GLOBAL_NON_GEMMA',
                'phase': plan['phase'], 'pending_profiles': pending, 'completed_non_gemma': len(closures),
                'deadline_seconds_remaining': max(0, deadline - time.monotonic()), 'GPU_queries': 0})
            if events.refresh(identities): events.wait(deadline)


def profile_closed(ref, p):
    """Scheduling boundary only; scientific audits stay in the bound supervisors."""
    require(file_sha256(ref['path']) == ref['sha256'], 'Sealed profile changed during supervision')
    out = Path(p['output']); wait_path = out.with_suffix('.wait.json')
    wait = read(wait_path); result = read(out / 'result.json'); exit_record = read(out / 'exit.json')
    require(wait['profile_sha256'] == digest(p) and wait['returncode'] == 0 and wait['real_wait_observed'] is True
            and wait['owned_engines_exited'] is True and wait['original_physical_lease_released'] is True
            and wait['terminal_status'] == 'COMPLETE' and wait['completed_ids'] == p['assigned_ids']
            and result['status'] == 'COMPLETE' and result['completed_ids'] == p['assigned_ids']
            and not result.get('UNCERTAIN_ids') and not (out / 'failure.json').exists(),
            'Previous finite worker is not actually COMPLETE/wait0/owned-exited/lease-free')
    require(exit_record['owned_engine_workers_exited'] is True and exit_record['shutdown_error'] is None,
            'Owned model engine shutdown is incomplete')
    lock = p['physical_lock']; path = Path(lock['path']); lease = read(out / 'lease.json')
    require(path.is_file() and not path.is_symlink()
            and (path.stat().st_dev, path.stat().st_ino) == (lock['st_dev'], lock['st_ino'])
            and all(lease[key] == value for key, value in lock.items()), 'Original physical lease inode binding changed')
    return {'profile': {'path': ref['path'], 'sha256': ref['sha256']},
            'wait': descriptor(wait_path), 'result': descriptor(out / 'result.json'),
            'exit': descriptor(out / 'exit.json'), 'lease': descriptor(out / 'lease.json'),
            'status': 'COMPLETE', 'new_calls': len(p['assigned_ids'])}


def validate_launch(plan_path, plan, owner_chat):
    out = Path(plan['output']); intent = read(out / 'launch-intent.json')
    require(intent['plan'] == descriptor(plan_path) and intent['queue_entry'] == plan['queue_entry']
            and intent['owner_chat'] == owner_chat and intent['automatic_retry'] is False,
            'Missing or foreign exclusive launch intent')
    parent = intent['launcher']
    require(os.getppid() == parent['pid'] and process_identity(parent['pid']) == parent and process_alive(parent),
            'Queue run is not the direct child of its real original launcher')
    return intent


def wait_created_child(child):
    """A created process is always reaped, even after a receipt/interrupt error."""
    interrupted = None
    while True:
        try:
            return child.wait(), interrupted
        except KeyboardInterrupt as error:
            interrupted = error


def spawn_and_wait(argv, *, log, cwd, env, child_receipt, receipt_fields, identity_key):
    child = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=subprocess.STDOUT)
    identity, receipt_error = None, None
    try:
        identity = process_identity(child.pid)
        durable_new(child_receipt, {**receipt_fields, identity_key: identity, 'argv': argv})
    except BaseException as error:
        receipt_error = error
    finally:
        code, interrupted = wait_created_child(child)
    if receipt_error is not None:
        receipt_error.static_queue_child_wait = {'pid': child.pid, 'identity': identity,
            'returncode': code, 'real_wait_observed': True}
        raise receipt_error
    if interrupted is not None:
        interrupted.static_queue_child_wait = {'pid': child.pid, 'identity': identity,
            'returncode': code, 'real_wait_observed': True}
        raise interrupted
    return identity, code


def run(plan_path, owner_chat):
    plan, profiles = validate_plan(plan_path, owner_chat)
    require(sys.executable == profiles[0]['python'], 'Queue child must use its sealed worker interpreter')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU queue process must have no visible CUDA devices')
    intent = validate_launch(plan_path, plan, owner_chat); out = Path(plan['output'])
    durable_new(out / 'queue-identity.json', {'plan': descriptor(plan_path), 'owner': process_identity(),
                'launcher': intent['launcher'], 'queue_entry': plan['queue_entry']})
    completed, pending = [], None
    try:
        with Path(plan['queue_lease']['path']).open('r+b') as stream:
            state = os.fstat(stream.fileno()); lock = plan['queue_lease']
            require((state.st_dev, state.st_ino) == (lock['st_dev'], lock['st_ino']), 'CPU lease changed before lock')
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            durable_new(out / 'queue-lease.json', {'lease': lock, 'owner': process_identity(),
                         'physical_GPU_lease_held_by_CPU_queue': False})
            for ordinal, (ref, p) in enumerate(zip(plan['profiles'], profiles)):
                # Recheck frozen identities at each boundary; never select a new model/profile.
                require(descriptor(plan_path) == intent['plan'] and descriptor(__file__) == plan['queue_entry'],
                        'Queue plan/implementation changed')
                require(file_sha256(source_path(ref['supervisor_module'])) == ref['supervisor_sha256'],
                        'Supervisor changed before successor')
                require(file_sha256(ref['path']) == ref['sha256'], 'Profile changed before successor')
                require((Path(lock['path']).stat().st_dev, Path(lock['path']).stat().st_ino)
                        == (lock['st_dev'], lock['st_ino']), 'Stable CPU queue lease was replaced')
                if ordinal:
                    profile_closed(plan['profiles'][ordinal - 1], profiles[ordinal - 1])
                barrier_ref = None
                if p['model'] == 'gemma_x_full':
                    try:
                        gemma_barrier(plan, intent['plan'], out)
                    except BaseException as error:
                        error.static_gemma_barrier_hold = True
                        atomic_json(out / 'pre-gemma-barrier-HOLD.json', {'status': 'HOLD',
                            'plan': intent['plan'], 'error_type': type(error).__name__, 'error': str(error),
                            'Gemma_engine_started_by_this_queue': False, 'new_Gemma_calls': 0})
                        raise
                    barrier_ref = descriptor(out / 'pre-gemma-barrier.json')
                pending = ref
                stem = f'profile-{ordinal:02d}'
                durable_new(out / (stem + '-intent.json'), {'ordinal': ordinal, 'profile': ref,
                            'queue_owner': process_identity(), 'automatic_retry': False,
                            'pre_gemma_barrier': barrier_ref})
                env = dict(os.environ)
                for key in p.get('cleared_environment', []): env.pop(key, None)
                env.update(p['environment'])
                argv = [p['python'], '-B', '-m', ref['supervisor_module'], 'supervise', '--profile', ref['path']]
                with (out / (stem + '.log')).open('xb') as log:
                    child_identity, code = spawn_and_wait(argv, log=log,
                        cwd=str(Path(p['implementation']['path']).parent.parent), env=env,
                        child_receipt=out / (stem + '-child.json'),
                        receipt_fields={'ordinal': ordinal, 'profile': ref, 'parent': process_identity()},
                        identity_key='supervisor')
                exited = not process_alive(child_identity)
                wait = {'ordinal': ordinal, 'profile': ref, 'supervisor': child_identity,
                        'returncode': code, 'real_wait_observed': True, 'known_supervisor_exited': exited,
                        'automatic_retry': False}
                durable_new(out / (stem + '-wait.json'), wait)
                require(code == 0 and exited, 'Known supervisor failed or did not exit; queue stops without retry')
                closure = profile_closed(ref, p)
                completed.append({'ordinal': ordinal, 'profile': ref,
                                  'queue_wait': descriptor(out / (stem + '-wait.json')), 'closure': closure})
                pending = None
                atomic_json(out / 'progress.json', {'status': 'RUNNING', 'completed': len(completed),
                    'expected': len(profiles), 'last_profile': ref, 'automatic_retry': False})
            atomic_json(out / 'queue-result.json', {'status': 'COMPLETE', 'plan': descriptor(plan_path),
                'completed': completed, 'UNCERTAIN_profiles': [], 'automatic_retry': False,
                'next_phase_launched': False})
        return {'status': 'COMPLETE', 'completed_profiles': len(completed), 'next_phase_launched': False}
    except BaseException as error:
        atomic_json(out / 'queue-result.json', {'status': 'HOLD_PRE_GEMMA' if getattr(error, 'static_gemma_barrier_hold', False)
                    else 'FAILED_OR_UNCERTAIN', 'plan': intent['plan'],
            'completed': completed, 'UNCERTAIN_profiles': [pending] if pending else [],
            'error_type': type(error).__name__, 'error': str(error), 'automatic_retry': False,
            'created_child_wait': getattr(error, 'static_queue_child_wait', None),
            'next_phase_launched': False})
        raise


def launch_once(plan_path, owner_chat):
    plan, profiles = validate_plan(plan_path, owner_chat); out = Path(plan['output'])
    require(sys.platform.startswith('linux'), 'Launch requires actual Linux PID/startticks/boot identity')
    parent = process_identity()
    intent = {'schema': 'static-x-queue-exclusive-launch-v1', 'plan': descriptor(plan_path),
              'queue_entry': plan['queue_entry'], 'owner_chat': owner_chat, 'launcher': parent,
              'automatic_retry': False, 'formal_auto_resume': False}
    durable_new(out / 'launch-intent.json', intent)
    env = dict(os.environ); env['CUDA_VISIBLE_DEVICES'] = ''
    argv = [profiles[0]['python'], '-B', '-m', MODULE, 'run', '--plan', str(plan_path), '--owner-chat', owner_chat]
    started = time.monotonic()
    try:
        with (out / 'queue-run.log').open('xb') as log:
            identity, code = spawn_and_wait(argv, log=log, cwd=str(Path(__file__).parent.parent), env=env,
                child_receipt=out / 'queue-child.json',
                receipt_fields={'launcher': parent, 'plan': intent['plan']}, identity_key='queue_child')
        exited = not process_alive(identity)
        result_path = out / 'queue-result.json'
        result = read(result_path) if result_path.exists() else {'status': 'FAILED_OR_UNCERTAIN'}
        outer = {'schema': 'static-x-queue-real-outer-wait-v1', 'plan': intent['plan'],
                 'launcher': parent, 'queue_child': identity, 'returncode': code,
                 'real_wait_observed': True, 'known_queue_child_exited': exited,
                 'terminal_status': result['status'], 'seconds': time.monotonic() - started,
                 'automatic_retry': False}
        durable_new(out / 'outer-wait.json', outer)
        require(code == 0 and exited and result['status'] == 'COMPLETE', 'Queue failed or is uncertain; no retry or phase continuation')
        return {'status': 'COMPLETE', 'outer_wait': descriptor(out / 'outer-wait.json'),
                'plan': intent['plan'], 'next_phase_launched': False}
    except BaseException as error:
        durable_new(out / 'launcher-failure.json', {'plan': intent['plan'], 'launcher': parent,
            'error_type': type(error).__name__, 'error': str(error), 'automatic_retry': False,
            'created_child_wait': getattr(error, 'static_queue_child_wait', None),
            'unknown_wait_is_not_completion': True})
        raise


def closed(plan_path, owner_chat):
    plan, profiles = validate_plan(plan_path, owner_chat); out = Path(plan['output'])
    intent = read(out / 'launch-intent.json'); child = read(out / 'queue-child.json')
    identity = read(out / 'queue-identity.json'); outer = read(out / 'outer-wait.json')
    result = read(out / 'queue-result.json')
    require(intent['plan'] == child['plan'] == identity['plan'] == outer['plan'] == result['plan'] == descriptor(plan_path)
            and intent['owner_chat'] == owner_chat and intent['queue_entry'] == identity['queue_entry'] == plan['queue_entry']
            and intent['launcher'] == child['launcher'] == identity['launcher'] == outer['launcher']
            and child['queue_child'] == identity['owner'] == outer['queue_child'], 'Queue process/plan identity binding differs')
    require(outer['returncode'] == 0 and outer['real_wait_observed'] is True
            and outer['known_queue_child_exited'] is True and outer['terminal_status'] == result['status'] == 'COMPLETE'
            and not process_alive(child['queue_child']) and not (out / 'launcher-failure.json').exists()
            and not result['UNCERTAIN_profiles'] and result['automatic_retry'] is False
            and result['next_phase_launched'] is False, 'Queue lacks its real successful parent wait')
    require(len(result['completed']) == len(profiles), 'Incomplete finite queue')
    barrier_path = out / 'pre-gemma-barrier.json'; barrier = read(barrier_path)
    pre_gemma_refs, global_manifest = resolve_pre_gemma(plan)
    require(barrier['schema'] == 'static-x-global-gemma-last-barrier-v1' and barrier['status'] == 'PASS'
            and barrier['plan'] == descriptor(plan_path) and barrier['static_authorization'] == plan['static_authorization']
            and pre_gemma_refs is not None and barrier['pre_gemma_profiles'] == pre_gemma_refs
            and barrier['global_manifest'] == global_manifest
            and barrier['physical_GPU_lease_held'] is False and barrier['GPU_queries'] == 0
            and barrier['closures'] == [profile_closed(ref, verified(ref)) for ref in pre_gemma_refs]
            and not (out / 'pre-gemma-barrier-HOLD.json').exists(), 'Global Gemma-last barrier proof differs')
    for ordinal, (ref, p, completed) in enumerate(zip(plan['profiles'], profiles, result['completed'])):
        stem = f'profile-{ordinal:02d}'; worker_child = read(out / (stem + '-child.json'))
        worker_intent = read(out / (stem + '-intent.json')); worker_wait = read(out / (stem + '-wait.json'))
        require(worker_intent['ordinal'] == worker_child['ordinal'] == worker_wait['ordinal'] == completed['ordinal'] == ordinal
                and worker_intent['profile'] == worker_child['profile'] == worker_wait['profile'] == completed['profile'] == ref
                and worker_intent['queue_owner'] == worker_child['parent'] == identity['owner']
                and worker_intent['pre_gemma_barrier'] == (descriptor(barrier_path) if p['model'] == 'gemma_x_full' else None)
                and worker_child['supervisor'] == worker_wait['supervisor'] and worker_wait['returncode'] == 0
                and worker_wait['real_wait_observed'] is True and worker_wait['known_supervisor_exited'] is True
                and not process_alive(worker_child['supervisor'])
                and completed['queue_wait'] == descriptor(out / (stem + '-wait.json'))
                and completed['closure'] == profile_closed(ref, p), 'Finite profile scheduling boundary failed closure')
    lease = read(out / 'queue-lease.json')
    require(lease['lease'] == plan['queue_lease'] and lease['owner'] == identity['owner']
            and lease['physical_GPU_lease_held_by_CPU_queue'] is False, 'CPU queue lease binding differs')
    return {'status': 'COMPLETE_PASS_REAL_QUEUE_CLOSURE', 'plan': descriptor(plan_path),
            'outer_wait': descriptor(out / 'outer-wait.json'), 'completed_profiles': len(profiles),
            'next_phase_launched': False, 'model_calls_by_queue_module': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'supervise', 'launch_once', 'closed'))
    parser.add_argument('--plan', required=True); parser.add_argument('--owner-chat', required=True)
    args = parser.parse_args()
    if args.command == 'run': result = run(args.plan, args.owner_chat)
    elif args.command == 'closed': result = closed(args.plan, args.owner_chat)
    else: result = launch_once(args.plan, args.owner_chat)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
