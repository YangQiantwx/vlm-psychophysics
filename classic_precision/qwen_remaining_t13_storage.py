"""One explicitly authorized Radon SSD restart using the frozen Qwen runner.

The only engine-setting addition is the official local-media allowlist. Profiles
retain the actual setting, raw/profile bindings and all frozen science checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
import tempfile

from . import qwen_remaining_t13 as q
from .refinement_worker import read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

NAME = q.NAME
RESTART = 'ssd-restart1'
ALLOWED_ROOT = '/home/q/cvpr2027_datasets_ssd'
COUNTS = {'T3': 738, 'T1': 1038}
MODULE = 'classic_precision.qwen_remaining_t13_storage'
_science = q.science


def descriptor(path):
    # Keep canonical paths portable when the run directory is an SSD symlink.
    return {'path': str(path), 'sha256': file_sha256(path)}


def worker_name(group, generation):
    return 'radon0-full-' + group + '-' + RESTART + '-segment' + str(generation)


def validate_storage(p, resolve=verified):
    entry = p['storage_entry']
    require(set(entry) == {'implementation', 'authorization', 'restart'}
            and entry['restart'] == RESTART
            and entry['implementation']['sha256'] == file_sha256(__file__)
            and Path(entry['implementation']['path']).name == Path(__file__).name,
            'Storage entry/source differs')
    require(p['card'] == 'radon0' and p['phase'] == 'full'
            and p['resource_group'] in COUNTS, 'SSD restart is only Radon classic T1/T3')
    require(p['engine_kwargs'].get('allowed_local_media_path') == ALLOWED_ROOT,
            'Official media allowlist is not the exact authorized SSD root')
    auth = resolve(entry['authorization'])
    require(auth['schema'] == NAME + '/radon-ssd-restart-v1' and auth['status'] == 'AUTHORIZED'
            and auth['restart'] == RESTART and auth['cards'] == ['radon0']
            and auth['maximum_new_model_calls'] == 1776 and auth['task_counts'] == COUNTS
            and auth['experiment_plan_sha256'] == p['experiment_plan']['sha256']
            and auth['allowed_local_media_path'] == ALLOWED_ROOT
            and auth['automatic_retry'] is False and auth['previous_outputs_preserved'] is True
            and isinstance(auth.get('user_message'), str) and auth['user_message'].strip(),
            'Explicit finite SSD restart authorization differs')
    worker = worker_name(p['resource_group'], p['control_generation'])
    require(p['worker'] == worker and Path(p['output']) == Path(p['workspace_root']) /
            'runs/reproducibility' / NAME / worker, 'Restart output/worker identity differs')
    return auth


def science(p, resolve=verified):
    if 'storage_entry' not in p:
        return _science(p, resolve)

    def normalized(value):
        if isinstance(value, dict) and 'storage_entry' in value and 'engine_kwargs' in value:
            validate_storage(value, resolve)
            value = copy.deepcopy(value)
            del value['engine_kwargs']['allowed_local_media_path']
        return value

    # The actual profile is never rewritten. Normalize exactly this directory
    # key, including bound ancestors, solely for the frozen science comparison.
    return _science(normalized(p), lambda ref: normalized(resolve(ref)))


@contextmanager
def science_context():
    previous = q.science
    q.science = science
    try:
        yield
    finally:
        q.science = previous


def check_media_roots(paths, allowed_root):
    root = Path(allowed_root).resolve(strict=True)
    require(root.is_dir(), 'Authorized media root is not a directory')
    for path in paths:
        real = Path(path).resolve(strict=True)
        require(real.is_file() and real.is_relative_to(root), 'Scientific source escapes authorized SSD root')
    return root


def source_admission(p):
    validate_storage(p); catalog = verified(p['source_catalog'])['items']; paths = []
    for sid in p['assigned_ids']:
        for media in q.base.media_entries(p, sid, catalog[sid]):
            if media['type'] in ('video', 'image'):
                paths.append(media['path'])
    root = check_media_roots(paths, ALLOWED_ROOT)
    require(str(root) == ALLOWED_ROOT, 'Authorized allowlist root is not its physical canonical path')
    checkpoint = Path(p['checkpoint']).resolve(strict=True)
    return {'status': 'PASS_EXACT_OFFICIAL_SSD_MEDIA_ROOT', 'storage_entry': p['storage_entry'],
        'profile_sha256': digest(p), 'allowed_local_media_path': ALLOWED_ROOT,
        'resolved_media_root': str(root), 'source_files': len(paths),
        'source_filesystem_devices': sorted({Path(x).stat().st_dev for x in paths}),
        'checkpoint_resolved': str(checkpoint),
        'checkpoint_shard_filesystem_devices': sorted({(checkpoint / x).stat().st_dev
            for x in p['checkpoint_assets']['weight_shard_bytes']}),
        'scope': 'Exact media containment and filesystem identity. Physical SSD type is checked by the separate host preflight; this receipt makes no physical-device-type claim. No GPU call.'}


@contextmanager
def execution_context():
    native = subprocess.Popen

    def popen(argv, **kwargs):
        if isinstance(argv, (list, tuple)):
            argv = list(argv)
            if '-m' in argv:
                index = argv.index('-m') + 1
                if index < len(argv) and argv[index] == 'classic_precision.qwen_remaining_t13':
                    argv[index] = MODULE
        return native(argv, **kwargs)

    with science_context():
        subprocess.Popen = popen
        try:
            yield
        finally:
            subprocess.Popen = native


def seal(draft, output):
    p = read(draft); science(p); source_admission(p)
    with science_context():
        return q.seal(draft, output)


def run(profile):
    p = read(profile); science(p); admission = source_admission(p)
    write_new(Path(p['output']).with_suffix('.storage.json'), admission)
    with science_context():
        return q.run(profile)


def supervise(profile):
    p = read(profile); science(p)
    require('storage_entry' in p, 'Storage supervisor requires an explicit storage profile')
    with execution_context():
        return q.supervise(profile)


def activate(queue_path):
    queue = read(queue_path)
    for ref in queue['profiles']:
        p = verified(ref); science(p); source_admission(p)
    with science_context():
        return q.activate(queue_path)


def queue(queue_path):
    for ref in read(queue_path)['profiles']:
        p = verified(ref); science(p)
        require('storage_entry' in p, 'Storage queue contains a nonstorage profile')
    with execution_context():
        return q.queue(queue_path)


def closed(profile):
    with science_context():
        return q.closed(profile)


def prepare_resume(profile, output):
    """Keep the frozen closed/remaining/seal rules and the restart worker name."""
    p = read(profile); science(p); validate_storage(p)
    with q.lifecycle.controller_lock(str(Path(p['control_file']).with_suffix('.lock'))):
        p, result, raw = closed(profile); control = read(p['control_file'])
        require(control['paused'] and control['generation'] == p['control_generation'],
                'Only an unbranched explicitly paused restart can continue')
        ids = q.lifecycle.remaining(p, result, raw); require(ids, 'Nothing remains')
        require(not any((Path(p['claim_root']) / (sid + '.json')).exists() for sid in ids),
                'Unresolved claim in continuation suffix')
        generation = control['generation'] + 1; worker = worker_name(p['resource_group'], generation)
        value = copy.deepcopy(p)
        for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper'):
            value.pop(key, None)
        value.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
            output=str(Path(p['output']).parent / worker), pause_file=str(Path(output).with_suffix('.PAUSE.json')),
            continuation_parent=descriptor(profile), continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')),
            retained_prefix_length=len(raw))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(value, f); temp = f.name
        try:
            answer = seal(temp, output)
        finally:
            Path(temp).unlink()
        queue_path = Path(output).with_name(worker + '-queue.json')
        q.write_queue(queue_path, [descriptor(output)], p['workspace_root'])
        control.update(generation=generation, paused=True, continuation_profile=descriptor(output))
        atomic_json(p['control_file'], control)
    return {**answer, 'status': 'CONTINUATION_PREPARED_PAUSED', 'queue': descriptor(queue_path), 'model_calls': 0}


def prepare_resume_queue(queue_path, output):
    previous = read(queue_path); state = read(previous['state_path'])
    require(state['status'] in ('PAUSED', 'PAUSED_BEFORE_SUCCESSOR'), 'Only a clean paused queue can continue')
    require(not Path(output).exists(), 'Continuation queue exists')
    profiles = []
    with q.lifecycle.controller_lock(previous['controller_lock']):
        for ref in previous['profiles']:
            p = verified(ref); science(p); validate_storage(p)
            require(read(p['control_file'])['paused'], 'Pause both resource groups before continuation')
            if Path(p['output']).exists():
                _, terminal, _ = closed(ref['path'])
                if terminal['status'] == 'COMPLETE':
                    continue
                worker = worker_name(p['resource_group'], p['control_generation'] + 1)
                target = Path(ref['path']).with_name(worker + '-profile.json')
                resumed = prepare_resume(ref['path'], target)
                profiles.extend(verified(resumed['queue'])['profiles'])
            else:
                q.fresh(p); profiles.append(ref)
        require(profiles, 'Nothing remains')
        q.write_queue(output, profiles, read(profiles[0]['path'])['workspace_root'])
    return {'status': 'CONTINUATION_QUEUE_PREPARED_PAUSED', 'queue': descriptor(output), 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal', 'activate', 'queue', 'run', 'supervise', 'pause',
        'prepare-resume', 'prepare-resume-queue', 'audit-closed'))
    for key in ('draft', 'output', 'profile', 'plan', 'control'):
        parser.add_argument('--' + key)
    a = parser.parse_args()
    if a.command == 'seal': result = seal(a.draft, a.output)
    elif a.command == 'activate': result = activate(a.plan)
    elif a.command == 'queue': result = queue(a.plan)
    elif a.command == 'run': result = run(a.profile)
    elif a.command == 'supervise': result = supervise(a.profile)
    elif a.command == 'pause': result = q.lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': result = prepare_resume(a.profile, a.output)
    elif a.command == 'prepare-resume-queue': result = prepare_resume_queue(a.plan, a.output)
    else:
        _, terminal, raw = closed(a.profile)
        result = {'status': 'CLOSED_PASS', 'worker_status': terminal['status'], 'completed': len(raw)}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
