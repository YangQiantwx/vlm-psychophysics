"""Local ordinary-process orchestration of the existing finite commands."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .cpu import allocation, available_memory_gib, single_thread_environment
from .manifest import atomic_json, file_sha, selected_rows, validate_manifest
from .ready import ReadyCatalog, representatives


def process_owner(pid):
    path = Path('/proc')/str(pid)/'stat'
    try:
        return {'pid': pid, 'startticks': path.read_text().rsplit(')', 1)[1].split()[19]}
    except (FileNotFoundError, ProcessLookupError):
        if Path('/proc').exists():
            return {'pid': pid, 'startticks': None}
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'lstart='], capture_output=True, text=True)
        return {'pid': pid, 'startticks': result.stdout.strip() or None}


def owned_group_members(owner):
    """Only sessions/groups originally created by this parent, including orphans."""
    leader = process_owner(owner['pid'])
    if leader['startticks'] is not None and leader != owner:
        return []  # The numeric PID was reused; it no longer identifies our session.
    members = []
    if Path('/proc').exists():
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():
                continue
            try:
                fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
                if (int(fields[3]) == owner['pid'] and fields[0] != 'Z'
                        and path.stat().st_uid == os.getuid()):
                    members.append((int(path.name), int(fields[2])))
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
    else:
        result = subprocess.run(['ps', '-axo', 'pid=,pgid=,uid=,stat='], capture_output=True, text=True, check=True)
        for line in result.stdout.splitlines():
            pid, group, uid, status = line.split()
            if int(group) == owner['pid'] and int(uid) == os.getuid() and not status.startswith('Z'):
                members.append((int(pid), int(group)))
    return members


def terminate_owned_session(owner, grace_seconds=15):
    def signal_groups(signum):
        for group in {group for _, group in owned_group_members(owner)}:
            try:
                os.killpg(group, signum)
            except ProcessLookupError:
                pass
    signal_groups(signal.SIGTERM)
    deadline = time.monotonic()+grace_seconds
    while owned_group_members(owner) and time.monotonic() < deadline:
        time.sleep(.05)
    if owned_group_members(owner):
        signal_groups(signal.SIGKILL)
    deadline = time.monotonic()+5
    while owned_group_members(owner) and time.monotonic() < deadline:
        time.sleep(.05)
    return not owned_group_members(owner)


def checkpoint_prepared(path):
    """A missing pending shard is preparation, not a model-load failure."""
    path = Path(path)
    index = path/'model.safetensors.index.json'
    if not (path/'config.json').is_file():
        return False
    if not isinstance(json.loads((path/'config.json').read_text()), dict):
        raise ValueError('Invalid checkpoint config')
    if not any((path/name).is_file() for name in ('preprocessor_config.json', 'processor_config.json', 'video_preprocessor_config.json')):
        return False
    if not (path/'tokenizer_config.json').is_file() or not (
            (path/'tokenizer.json').is_file() or (path/'tokenizer.model').is_file()
            or ((path/'vocab.json').is_file() and (path/'merges.txt').is_file())):
        return False
    if index.exists():
        from .ready import inside
        mapping = json.loads(index.read_text()).get('weight_map')
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError('Invalid checkpoint index')
        return all(inside(path, relative).is_file() for relative in set(mapping.values()))
    return (path/'model.safetensors').is_file()


def pipeline(run_dir, model_root, gpu_shards, shard_count=3, produce=False, output_dir=None,
             worker_python=None, cosmos_package=None, cpu_max_workers=None, cpu_memory_gib=32,
             ram_reserve_gib=4, start_ready=100, input_poll_seconds=10,
             sync_destination_root=None, destination_ssh=None, destination_python=None,
             destination_code_root=None):
    import fcntl
    from .calibration import calibration_identity, calibration_ready
    if cpu_memory_gib < 8 or input_poll_seconds < 1:
        raise ValueError('Pipeline needs an8GiB CPU calibration allowance and positive poll')
    allocation(cpu_max_workers, cpu_memory_gib, ram_reserve_gib)
    run_dir = Path(run_dir).resolve()
    output_dir = Path(output_dir or run_dir/'outputs').resolve()
    worker_python = worker_python or sys.executable
    manifest_path = run_dir/'manifest.json'
    manifest = validate_manifest(json.loads(manifest_path.read_text()))
    pairs = [(str(gpu), int(shard)) for gpu, shard in gpu_shards]
    if (not pairs or len({gpu for gpu, _ in pairs}) != len(pairs)
            or len({shard for _, shard in pairs}) != len(pairs)):
        raise ValueError('One explicit GPU per distinct shard is required')
    for gpu, shard in pairs:
        if ',' in gpu:
            raise ValueError('A worker uses exactly one GPU')
        selected_rows(manifest, shard, shard_count)
    lock = (run_dir/'.pipeline.lock').open('a+b')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, **single_thread_environment()}
    catalog = ReadyCatalog(manifest, run_dir/'ready.jsonl', run_dir, file_sha(manifest_path), allow_failed=produce)
    logs = run_dir/'pipeline-logs'
    logs.mkdir(exist_ok=True)
    children, identities = {}, {}
    calibration_active = None
    status_path = run_dir/'pipeline.json'
    def launch(stage, arguments, python=None, cpu=False):
        child_env = {**env, **({'CUDA_VISIBLE_DEVICES': ''} if cpu else {})}
        with (logs/(stage+'.log')).open('ab', buffering=0) as stream:
            process = subprocess.Popen([python or sys.executable, '-B', '-m', 'classic_precision', *arguments],
                stdout=stream, stderr=subprocess.STDOUT, env=child_env, start_new_session=True)
        children[stage] = {'process': process, 'owner': process_owner(process.pid), 'done': False, 'session_stopped': False}
    def snapshot(status, **extra):
        value = {'status': status, 'producer': bool(produce), 'expected_inputs': len(manifest['rows']),
                 'gpu_shards': pairs, 'time_unix': time.time(), **extra,
                 'children': {name: {'owner': job['owner'], 'returncode': job['process'].returncode}
                              for name, job in children.items()}}
        atomic_json(status_path, value)
        return value
    def stop(signum, frame):
        raise KeyboardInterrupt('Pipeline received signal '+str(signum))
    previous_term = signal.signal(signal.SIGTERM, stop)
    try:
        versions = None
        if worker_python != sys.executable:
            result = subprocess.run([worker_python, '-B', '-c',
                'import json;from classic_precision.inference import _versions;print(json.dumps(_versions()))'],
                check=True, capture_output=True, text=True, env=env, timeout=60)
            versions = json.loads(result.stdout)
        ready = catalog.refresh()
        if produce and not catalog.complete:
            command = ['generate', '--run', str(run_dir), '--shard-count', str(shard_count),
                       '--cpu-memory-gib', str(cpu_memory_gib), '--ram-reserve-gib', str(ram_reserve_gib)]
            if cpu_max_workers is not None:
                command += ['--workers', str(cpu_max_workers)]
            launch('generate', command, cpu=True)
        while True:
            for name, job in children.items():
                members = [] if job['session_stopped'] else owned_group_members(job['owner'])
                code = job['process'].poll()
                if code is not None and not members:
                    job['session_stopped'] = True
                if code is not None:
                    if code:
                        raise RuntimeError('Pipeline child failed: '+name+'; inspect '+str(logs/(name+'.log')))
                    job['done'] = True
            generation = run_dir/'generation.json'
            marker = json.loads(generation.read_text()) if generation.exists() else {}
            if marker.get('status') == 'FAILED':
                if produce and 'generate' in children and not children['generate']['done']:
                    # Explicit producer resume will atomically publish RUNNING.
                    time.sleep(input_poll_seconds)
                    continue
                raise RuntimeError('Input producer/relay FAILED: '+marker.get('error', 'unknown'))
            ready = catalog.refresh()
            if sync_destination_root and 'sync-ready' not in children:
                command = ['sync-ready', '--manifest', str(manifest_path), '--source-root', str(run_dir),
                           '--destination-root', str(sync_destination_root), '--poll-seconds', str(input_poll_seconds)]
                for option, value in [('--destination-ssh', destination_ssh), ('--destination-python', destination_python),
                                      ('--destination-code-root', destination_code_root)]:
                    if value:
                        command += [option, str(value)]
                launch('sync-ready', command, cpu=True)
            for gpu, shard in pairs:
                stage = 'supervise-'+str(shard)
                if stage not in children:
                    command = ['supervise', '--manifest', str(manifest_path), '--output', str(output_dir),
                        '--model-root', str(model_root), '--gpu', gpu, '--shard-index', str(shard),
                        '--shard-count', str(shard_count), '--worker-python', worker_python,
                        '--start-ready', str(start_ready), '--input-poll-seconds', str(input_poll_seconds)]
                    if cosmos_package:
                        command += ['--cosmos-package', str(cosmos_package)]
                    launch(stage, command)
            if calibration_active and children[calibration_active]['done']:
                calibration_active = None
            atomic_json(run_dir/'cpu-activity.json', {'calibration_active': calibration_active is not None})
            completed, waiting = [], {}
            reps_ready = all(row['sample_id'] in ready for row in representatives(manifest))
            for model, spec in manifest['models'].items():
                checkpoint = Path(model_root)/spec['checkpoint_relative']
                if not checkpoint_prepared(checkpoint):
                    waiting[model] = 'WAITING_FOR_CHECKPOINT'
                    continue
                if model not in identities:
                    identities[model] = calibration_identity(manifest_path, manifest, model, model_root,
                                                            cosmos_package, versions)
                if calibration_ready(run_dir/'calibration', model, identities[model], manifest, ready):
                    completed.append(model)
                    continue
                if not reps_ready:
                    waiting[model] = 'WAITING_FOR_128_REPRESENTATIVES'
                elif calibration_active:
                    waiting[model] = 'WAITING_FOR_CPU_CALIBRATOR'
                elif available_memory_gib()-ram_reserve_gib < 8:
                    waiting[model] = 'WAITING_FOR_CALIBRATION_RAM'
                else:
                    calibration_active = 'calibrate-'+model
                    atomic_json(run_dir/'cpu-activity.json', {'calibration_active': True})
                    command = ['calibrate', '--manifest', str(manifest_path), '--model-root', str(model_root),
                               '--model', model, '--output', str(run_dir/'calibration'),
                               '--input-poll-seconds', str(input_poll_seconds)]
                    if cosmos_package:
                        command += ['--cosmos-package', str(cosmos_package)]
                    launch(calibration_active, command, worker_python, cpu=True)
            if len(completed) == len(manifest['models']) and all(job['done'] for job in children.values()):
                return snapshot('COMPLETE', ready=len(ready), calibration=completed)
            snapshot('RUNNING', ready=len(ready), generation_complete=catalog.complete,
                     calibration_complete=completed, waiting=waiting)
            time.sleep(input_poll_seconds)
    except BaseException as error:
        snapshot('INTERRUPTED' if isinstance(error, KeyboardInterrupt) else 'FAILED',
                 error_type=type(error).__name__, error=str(error))
        raise
    finally:
        # Only children we created in their own process sessions are signalled.
        cleaned = []
        for job in children.values():
            cleaned.append(job['session_stopped'] or terminate_owned_session(job['owner']))
            try:
                job['process'].wait(timeout=5)
            except subprocess.TimeoutExpired:
                cleaned[-1] = False
        if status_path.exists():
            final = json.loads(status_path.read_text())
            final['owned_sessions_stopped'] = all(cleaned)
            if not all(cleaned):
                final['original_status'] = final['status']
                final['status'] = 'FAILED_CLEANUP'
            final['children'] = {name: {'owner': job['owner'], 'returncode': job['process'].returncode}
                                 for name, job in children.items()}
            atomic_json(status_path, final)
        atomic_json(run_dir/'cpu-activity.json', {'calibration_active': False})
        signal.signal(signal.SIGTERM, previous_term)
        lock.close()
        if not all(cleaned) and sys.exc_info()[0] is None:
            raise RuntimeError('Owned session cleanup incomplete; inspect pipeline.json')
