"""Bounded, read-only status for a sealed full Cosmos X allocation.

This module uses only the standard library. It never loads a model, modifies a
claim, pauses a worker, or retries a request. A live snapshot is not final audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import shutil
import socket
import statistics
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def profile_digest(profile):
    encoded = json.dumps(profile, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def journal(path):
    """Ignore only a not-yet-durable final line of a live journal."""
    p = Path(path)
    if not p.exists():
        return [], False
    data = p.read_bytes()
    partial = bool(data and not data.endswith(b'\n'))
    lines = data.splitlines()
    if partial:
        lines = lines[:-1]
    return [json.loads(line) for line in lines if line.strip()], partial


def timing_window(path, count=20):
    """Tail a bounded amount of the small timing journal, not the large raw."""
    p = Path(path)
    if not p.exists():
        return {}
    with p.open('rb') as f:
        size = f.seek(0, os.SEEK_END)
        start = max(0, size - 256 * 1024)
        f.seek(start)
        data = f.read()
    lines = data.splitlines()
    if start:
        lines = lines[1:]
    if data and not data.endswith(b'\n'):
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines if line.strip()]
    values = [r['end_to_end_durable_seconds'] for r in rows
              if not r.get('deep', False)
              and isinstance(r.get('end_to_end_durable_seconds'), (int, float))
              and math.isfinite(r['end_to_end_durable_seconds'])
              and r['end_to_end_durable_seconds'] > 0][-count:]
    if not values:
        return {'n': 0}
    ordered = sorted(values)
    pos = (len(values) - 1) * .9
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return {'n': len(values), 'mean_seconds': statistics.mean(values),
            'median_seconds': statistics.median(values),
            'p90_seconds': ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo),
            'clock': 'same-worker monotonic; request-to-durable-raw-and-PASS; ordinary only'}


def inspect_profile(path):
    p = read(path)
    out = Path(p['output'])
    expected = p['assigned_ids']
    expected_set = set(expected)
    issues = []
    if len(expected_set) != len(expected) or len(expected) != p['maximum_model_calls']:
        issues.append('SEALED_ALLOCATION_NOT_UNIQUE_OR_QUOTA_MISMATCH')
    events, partial = journal(out / 'events.jsonl')
    passes = [x['sample_id'] for x in events if x.get('event') == 'TECHNICAL_PASS']
    intents = {x['sample_id'] for x in events if x.get('event') == 'INTENT'}
    done = set(passes)
    claims = {f.stem for f in Path(p['claim_root']).glob('*.json')}
    if len(passes) != len(done):
        issues.append('DUPLICATE_TECHNICAL_PASS')
    if not (done | intents | claims) <= expected_set:
        issues.append('FOREIGN_ID_IN_LEDGER')
    if not done <= claims:
        issues.append('COMPLETED_ID_WITHOUT_CLAIM')
    if not done <= intents:
        issues.append('COMPLETED_ID_WITHOUT_INTENT')
    progress = read(out / 'progress.json') if (out / 'progress.json').exists() else {}
    failure = read(out / 'failure.json') if (out / 'failure.json').exists() else None
    wait = read(out.with_suffix('.wait.json')) if out.with_suffix('.wait.json').exists() else None
    if failure:
        issues.append('WORKER_FAILURE_RETAINED')
    if wait and (wait.get('returncode') != 0 or not all(wait.get(k) for k in
            ('real_wait_observed', 'owned_engines_exited', 'original_physical_lease_released'))):
        issues.append('WAIT_OR_EXIT_OR_LEASE_NOT_SUCCESSFUL')
    if wait and wait.get('profile_sha256') != profile_digest(p):
        issues.append('WAIT_PROFILE_IDENTITY_MISMATCH')
    if wait and claims != done:
        issues.append('TERMINAL_CLAIMS_WITHOUT_PASS')
    if wait and wait.get('terminal_status') == 'COMPLETE' and done != expected_set:
        issues.append('COMPLETE_WITH_INCOMPLETE_ID_LEDGER')
    ref = p['source_catalog']
    catalog_ok = Path(ref['path']).exists() and sha(ref['path']) == ref['sha256']
    if not catalog_ok:
        issues.append('SOURCE_CATALOG_IDENTITY_MISMATCH')
    window = timing_window(out / 'request-timings.jsonl')
    remaining = len(expected_set - done)
    state = progress.get('status', 'STARTING')
    if wait:
        if wait.get('terminal_status') == 'COMPLETE' and done == expected_set:
            state = 'CLOSED_PENDING_FINAL_RAW_AUDIT'
        elif wait.get('terminal_status') == 'PAUSED':
            state = 'PAUSED_CLOSED'
        else:
            state = 'CLOSED_INCOMPLETE_NEEDS_REVIEW'
    if issues:
        state = 'FAILED_OR_NEEDS_REVIEW'
    return {'worker': p['worker'], 'status': state, 'profile_sha256': sha(path),
            'expected': len(expected), 'technical_pass_count': len(done),
            'claimed': len(claims), 'claimed_without_pass': sorted(claims - done),
            'unattempted': len(expected_set - claims), 'remaining': remaining,
            'progress_completed': progress.get('completed'),
            'last_completed': progress.get('last'), 'partial_event_tail': partial,
            'source_catalog_identity_pass': catalog_ok,
            'recent_ordinary': window,
            'eta_remaining_seconds': remaining * window['mean_seconds']
                if not wait and not issues and window.get('n') else None,
            'eta_basis': 'Recent request mean only; excludes load and between-request bookkeeping.',
            'failure': {k: failure.get(k) for k in
                ('error_type', 'error', 'UNCERTAIN_id', 'completed')} if failure else None,
            'wait': {k: wait.get(k) for k in ('returncode', 'terminal_status',
                'real_wait_observed', 'owned_engines_exited',
                'original_physical_lease_released')} if wait else None,
            'issues': issues, 'raw_full_audit_performed': False,
            'snapshot_note': 'Live files are read sequentially; one in-flight request is normal.'}


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(),
                'stderr': p.stderr.strip()[:500]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'error': str(error)}


def resources(root):
    mem = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            mem[key + '_GiB'] = int(value.split()[0]) / 1024**2
    def cpu():
        return [int(x) for x in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
    before = cpu()
    time.sleep(.2)
    after = cpu()
    delta = [b - a for a, b in zip(before, after)]
    total = sum(delta)
    disk = shutil.disk_usage(root)
    gpu_processes = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                             '--format=csv,noheader'])
    accounts = {}
    for line in gpu_processes.get('stdout', '').splitlines():
        try:
            pid = int(line.split(',')[1].strip())
            accounts[str(pid)] = pwd.getpwuid(Path('/proc', str(pid)).stat().st_uid).pw_name
        except (IndexError, ValueError, OSError, KeyError):
            continue  # Process may exit between nvidia-smi and the /proc lookup.
    return {'gpu': command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu',
                             '--format=csv,noheader']),
            'gpu_processes': gpu_processes, 'gpu_process_accounts': accounts,
            'memory': mem, 'load_average': os.getloadavg(),
            'cpu_busy_percent': 100 * (total - delta[3] - delta[4]) / total if total else None,
            'psi': {name: (Path('/proc/pressure') / name).read_text().strip()
                    for name in ('cpu', 'memory', 'io') if (Path('/proc/pressure') / name).exists()},
            'disk_GiB': {'total': disk.total / 1024**3, 'free': disk.free / 1024**3}}


def status(root, plan_dir, include_resources=True):
    root, plan_dir = Path(root), Path(plan_dir)
    plan = read(plan_dir / 'plan.json')
    workers = []
    for card in plan['allocation']:
        p = plan_dir / (card + '-profile.json')
        if p.exists() and read(p)['hostname'] == socket.gethostname():
            expected_sha = plan['allocation'][card]['sealed_profile']['sha256']
            if sha(p) != expected_sha:
                raise ValueError('Sealed profile identity mismatch: ' + str(p))
            workers.append(inspect_profile(p))
    if not workers:
        raise ValueError('No sealed profiles for this exact host; refusing to report an empty healthy state')
    return {'schema': 'cosmos-X-full-live-status-v1', 'host': socket.gethostname(),
            'observed_UTC': datetime.now(timezone.utc).isoformat(),
            'observed_monotonic_seconds': time.monotonic(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'workers': workers,
            'resources': resources(root) if include_resources and any(not w['wait'] for w in workers)
                else {'skipped': 'disabled or all local workers terminal'},
            'model_calls': 0, 'mutations': 0}


def queue_owner_observation(owner, boot_id, proc_root=Path('/proc')):
    """Distinguish durable RUNNING text from the lifetime of its actual owner."""
    if not owner:
        return {'status': 'NOT_OBSERVED', 'alive': None}
    if owner.get('boot_id') != boot_id:
        return {'status': 'PREVIOUS_HOST_BOOT', 'alive': False,
                'recorded_boot_id': owner.get('boot_id'), 'current_boot_id': boot_id}
    try:
        fields = (proc_root / str(int(owner['pid'])) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return {'status': 'OWNER_PROCESS_EXITED', 'alive': False}
    except (OSError, ValueError, KeyError, IndexError):
        return {'status': 'OWNER_IDENTITY_UNREADABLE', 'alive': None}
    if len(fields) <= 19 or str(owner.get('process_start_ticks')) != fields[19]:
        return {'status': 'OWNER_PID_IDENTITY_CHANGED', 'alive': False}
    return {'status': 'LIVE', 'alive': True}


def queue_status(root, queue_paths, include_resources=True):
    """Inspect exact sealed queues; never discover or resume other experiments."""
    root = Path(root)
    queues, workers, seen = [], [], set()
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    for path in queue_paths:
        queue = read(path)
        if queue.get('status') != 'SEALED' or queue.get('automatic_retry') is not False:
            raise ValueError('Expected an explicitly sealed finite queue')
        state_path = Path(queue['state_path'])
        state = read(state_path) if state_path.exists() else {'status': 'NOT_LAUNCHED'}
        owner = queue_owner_observation(state.get('owner'), boot_id)
        queues.append({'path': str(path), 'sha256': sha(path),
                       'state_path': str(state_path), 'state': state,
                       'owner_observation': owner})
        for ref in queue['profiles']:
            if ref['path'] in seen:
                raise ValueError('Repeated profile across requested queues')
            seen.add(ref['path'])
            if sha(ref['path']) != ref['sha256']:
                raise ValueError('Sealed profile identity mismatch')
            profile = read(ref['path'])
            if profile['hostname'] != socket.gethostname() or Path(profile['workspace_root']) != root:
                raise ValueError('Queue belongs to another host or workspace')
            worker = inspect_profile(ref['path'])
            worker['queue_owner_observation'] = owner
            if not worker['wait'] and owner['alive'] is False:
                rebooted = owner['status'] == 'PREVIOUS_HOST_BOOT'
                worker['issues'].append('QUEUE_OWNER_' + owner['status'])
                worker['status'] = ('INTERRUPTED_BY_HOST_REBOOT' if rebooted
                                    else 'QUEUE_OWNER_GONE_NEEDS_REVIEW')
                worker['eta_remaining_seconds'] = None
                worker['snapshot_note'] = 'Durable progress is not evidence of a live worker; no retry authorized.'
            worker.update({key: profile.get(key) for key in ('card', 'phase', 'resource_group')})
            workers.append(worker)
    if not workers:
        raise ValueError('No explicit sealed profiles')
    return {'schema': 'official-X-queue-live-status-v1', 'host': socket.gethostname(),
            'observed_UTC': datetime.now(timezone.utc).isoformat(),
            'observed_monotonic_seconds': time.monotonic(),
            'boot_id': boot_id,
            'queues': queues, 'workers': workers,
            'resources': resources(root) if include_resources and any(not w['wait'] for w in workers)
                else {'skipped': 'disabled or all requested profiles terminal'},
            'model_calls': 0, 'mutations': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--plan-dir')
    parser.add_argument('--queue', action='append', default=[])
    parser.add_argument('--no-resources', action='store_true')
    args = parser.parse_args()
    plan_dir = args.plan_dir or str(Path(args.root) / 'plans/cosmos-x-full-20261028')
    result = queue_status(args.root, args.queue, not args.no_resources) if args.queue else status(args.root, plan_dir, not args.no_resources)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
