"""Read-only workload telemetry; never launches a load or changes GPU settings.

One collector per host records UUID-keyed GPU measurements and host pressure.
GPU board power is not wall power, rack power, or a UPS load measurement.
The JSONL is append-only; the small latest JSON is replaced atomically.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import xml.etree.ElementTree as ET

from .storage import atomic_json, owner_identity


def command(args, timeout=5):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {'returncode': None, 'error': str(e)}


def tree(element):
    if not list(element):
        return (element.text or '').strip()
    result = {}
    for child in element:
        value = tree(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(value)
        else:
            result[child.tag] = value
    return result


def gpu_snapshot():
    response = command(['nvidia-smi', '-q', '-x'])
    if response['returncode'] != 0:
        return {'status': 'UNAVAILABLE', 'detail': response}
    try:
        root = ET.fromstring(response['stdout'])
        fields = ('product_name', 'uuid', 'performance_state', 'fb_memory_usage',
                  'utilization', 'temperature', 'power_readings', 'gpu_power_readings',
                  'clocks', 'clocks_event_reasons', 'clocks_throttle_reasons',
                  'ecc_errors', 'retired_pages', 'processes', 'pci')
        return {'status': 'READ_OK', 'driver_version': root.findtext('driver_version'),
                'gpus': [{k: tree(g.find(k)) for k in fields if g.find(k) is not None}
                         for g in root.findall('gpu')]}
    except (ET.ParseError, ValueError) as e:
        return {'status': 'PARSE_ERROR', 'error': str(e)}


def host_snapshot(workspace):
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError as e:
            return 'UNAVAILABLE: ' + str(e)
    mem = {k: v.strip() for k, v in
           (line.split(':', 1) for line in read('/proc/meminfo').splitlines() if ':' in line)}
    vm = {parts[0]: int(parts[1]) for line in read('/proc/vmstat').splitlines()
          if (parts := line.split()) and parts[0] in ('pswpin', 'pswpout', 'pgmajfault')}
    disk = os.statvfs(workspace)
    return {'loadavg': read('/proc/loadavg'), 'cpu_counters': read('/proc/stat').splitlines()[0],
            'memory': {k: mem.get(k) for k in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree')},
            'vm_counters': vm, 'pressure': {x: read('/proc/pressure/' + x) for x in ('cpu', 'memory', 'io')},
            'disk_available_bytes': disk.f_bavail * disk.f_frsize}


def kernel_events():
    r = command(['journalctl', '-k', '--since', '-70 seconds', '--no-pager', '-n', '200', '-o', 'short-iso'])
    if r['returncode'] != 0 or 'permission' in r.get('stderr', '').lower():
        return {'status': 'UNAVAILABLE', 'detail': r}
    lines = [x for x in r.get('stdout', '').splitlines()
             if ('NVRM' in x or 'Xid' in x or 'GPU has fallen' in x or 'Out of memory' in x)]
    return {'status': 'READ_OK_BOUNDED_WINDOW', 'events': lines,
            'limit': 'Last70s, at most200 kernel lines; not a complete system-event audit.'}


def terminal(paths):
    if not paths:
        return False
    statuses = []
    for path in paths:
        try:
            statuses.append(json.loads(Path(path).read_text()).get('status'))
        except (OSError, ValueError):
            return False
    return all(x in ('COMPLETE', 'HELD_NO_RETRY', 'PAUSED', 'PAUSED_BEFORE_SUCCESSOR') for x in statuses)


def collect(args):
    assert args.interval >= 5 and 0 < args.max_seconds <= 172800
    out = Path(args.output).resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    assert not out.exists(), 'Use a new telemetry filename; never overwrite a prior session'
    lock_path = out.parent / 'collector.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'RUNNING', 'owner': owner_identity(), 'host': socket.gethostname(),
                 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                 'kernel': os.uname().release, 'interval_seconds': args.interval,
                 'output': str(out), 'samples': 0, 'gpu_read_errors': 0,
                 'interpretation': 'Observed inference workload, not an extra stress workload. GPU board power is not UPS or wall power.',
                 'ups': 'NOT_MEASURED: no UPS telemetry connection configured.', 'gpu_power_peak_watts': {}}
        stop = []
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda number, frame: stop.append(number))
        start = time.monotonic(); last_kernel = float('-inf'); events = {}
        atomic_json(out.with_suffix('.owner.json'), state)
        with out.open('x') as journal:
            while True:
                now = time.monotonic()
                if now - last_kernel >= 60:
                    events = kernel_events(); last_kernel = now
                gpu = gpu_snapshot()
                sample = {'utc': datetime.now(timezone.utc).isoformat(), 'monotonic_seconds': now,
                          'boot_id': state['boot_id'], 'gpu': gpu, 'host': host_snapshot(args.workspace),
                          'kernel_events_last_observation': events}
                journal.write(json.dumps(sample, ensure_ascii=False) + '\n')
                journal.flush()
                state['samples'] += 1; state['latest'] = sample
                state['gpu_read_errors'] += gpu['status'] != 'READ_OK'
                for g in gpu.get('gpus', []):
                    powers = g.get('gpu_power_readings', g.get('power_readings', {}))
                    value = powers.get('instant_power_draw', powers.get('power_draw', 'N/A'))
                    try:
                        watts = float(value.split()[0]); uuid = g['uuid']
                        state['gpu_power_peak_watts'][uuid] = max(watts, state['gpu_power_peak_watts'].get(uuid, 0))
                    except (ValueError, TypeError, KeyError):
                        pass
                state['elapsed_monotonic_seconds'] = time.monotonic() - start
                reason = ('SIGNAL' if stop else 'STOP_FILE' if args.stop_file and Path(args.stop_file).exists()
                          else 'BOUND_REACHED' if state['elapsed_monotonic_seconds'] >= args.max_seconds
                          else 'QUEUES_TERMINAL' if terminal(args.terminal_state) else None)
                if args.once:
                    reason = 'ONE_READ_COMPLETE'
                if reason:
                    state.update(status='STOPPED', stop_reason=reason)
                atomic_json(out.with_suffix('.latest.json'), state)
                if reason:
                    os.fsync(journal.fileno()); break
                if state['samples'] % 6 == 0:
                    os.fsync(journal.fileno())
                # This daemon sleeps locally, independent of an SSH/Mac connection.
                deadline = now + args.interval
                while not stop and time.monotonic() < deadline:
                    time.sleep(min(1, max(0, deadline-time.monotonic())))
        return {k: state[k] for k in ('status', 'host', 'samples', 'gpu_read_errors', 'stop_reason')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', required=True); p.add_argument('--output', required=True)
    p.add_argument('--interval', type=float, default=10); p.add_argument('--max-seconds', type=float, default=86400)
    p.add_argument('--terminal-state', action='append', default=[]); p.add_argument('--stop-file')
    p.add_argument('--once', action='store_true')
    print(json.dumps(collect(p.parse_args())), flush=True)


if __name__ == '__main__':
    main()
