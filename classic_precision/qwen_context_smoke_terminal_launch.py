"""Authorized operational handoff after the safely closed Gemma mismatch.

The sealed Qwen science, profile, budget, queue and worker stay unchanged.
Only this CPU launch process substitutes the predecessor admission rule.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

from . import qwen_context_smoke as smoke
from .refinement_worker import read, require, verified
from .storage import digest, file_sha256

ANSWER = '继续Qwen context 48题（推荐）'
QUEUE_SHA = 'f013de1dcefc53d7b6f719484b192ff2cd507207c68db630a68e375c3e234d3e'
PROFILE_SHA = '29a497adbbc19bd7b4554a9f3fcc3e1a8f335c7a63bf1a6727acce8e31d9ff8f'
COUNTS = (96, 37)


def validate_records(auth, p, outer, documents, owner):
    """CPU-only admission, including exact completed prefixes and claims."""
    require(auth['status'] == 'AUTHORIZED' and auth['user_answer'] == ANSWER
            and auth['maximum_new_model_calls'] == 48
            and auth['queue']['sha256'] == QUEUE_SHA
            and auth['profile']['sha256'] == PROFILE_SHA
            and auth['scientific_configuration_unchanged'] is True
            and auth['preserve_Gemma_failure'] is True
            and auth['automatic_retry'] is False, 'Terminal handoff authorization differs')
    require(outer['returncode'] == 1 and outer['real_wait_observed'] is True
            and outer['queue_sha256'] == smoke.GEMMA_QUEUE_SHA
            and outer['queue'] == auth['predecessor_queue']['path'],
            'Expected real Gemma outer wait1 after the known pause')
    require(owner['alive'] is False and len(documents) == 2,
            'Gemma owner is live or workers differ')
    for i, (record, count) in enumerate(zip(documents, COUNTS), 1):
        gp, wait, result, control = (record[k] for k in ('profile', 'wait', 'result', 'control'))
        require(gp['worker'] == f'radon0-qualification-cold{i}'
                and gp['physical_lock'] == p['physical_lock']
                and len(gp['assigned_ids']) == gp['maximum_model_calls'] == 96,
                'Gemma worker/allocation/physical lease differs')
        require(wait['returncode'] == 0 and wait['real_wait_observed'] is True
                and wait['owned_engines_exited'] is True
                and wait['original_physical_lease_released'] is True
                and wait['profile_sha256'] == digest(gp), 'Gemma worker did not close safely')
        expected_status = 'COMPLETE' if i == 1 else 'PAUSED'
        completed = gp['assigned_ids'][:count]
        require(result['status'] == wait['terminal_status'] == expected_status
                and result['completed_ids'] == wait['completed_ids'] == completed
                and result['unattempted_ids'] == gp['assigned_ids'][count:]
                and not result.get('UNCERTAIN_ids') and not record['failure_exists']
                and control['paused'] is True
                and control['generation'] == gp['control_generation'],
                'Gemma terminal prefix/control differs or has uncertain work')
        claims = record['claims']
        require(len(claims) == count and {x['sample_id'] for x in claims} == set(completed)
                and all(x['profile_sha256'] == digest(gp) and x['worker_token'] == gp['worker_token']
                        for x in claims), 'Gemma claims are not the exact closed prefix')


def predecessor_admission(auth, plan, p):
    require(plan['predecessor_queue'] == auth['predecessor_queue'], 'Different predecessor')
    queue = verified(plan['predecessor_queue']); outer = verified(auth['outer_wait'])
    failure = verified(auth['failure_summary'])
    require(failure['status'] == 'FAILED_GEMMA_X_REPRODUCTION_CLOSED'
            and failure['R_raw'] == 133 and failure['R_differences'] == 1,
            'Known failure record changed')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    owner = verified(auth['outer_owner'])
    require(owner['boot_id'] == boot, 'Predecessor boot changed; review actual closure')
    observed = smoke.queue_owner_observation(owner, boot)
    documents = []
    for ref, evidence in zip(queue['profiles'], auth['workers'], strict=True):
        require(ref == evidence['profile'], 'Predecessor profile binding differs')
        gp = verified(ref); out = Path(gp['output'])
        require(evidence['wait']['path'] == str(out.with_suffix('.wait.json'))
                and evidence['result']['path'] == str(out / 'result.json')
                and evidence['raw']['path'] == str(out / 'raw.jsonl')
                and file_sha256(out / 'raw.jsonl') == evidence['raw']['sha256'],
                'Closure path or final raw binding differs')
        documents.append({'profile': gp, 'wait': verified(evidence['wait']),
            'result': verified(evidence['result']), 'control': read(gp['control_file']),
            'claims': [read(x) for x in Path(gp['claim_root']).glob('*.json')],
            'failure_exists': (out / 'failure.json').exists()})
    validate_records(auth, p, outer, documents, observed)
    fd = os.open(p['physical_lock']['path'], os.O_RDWR)
    try:
        st = os.fstat(fd)
        require((st.st_dev, st.st_ino) == (p['physical_lock']['st_dev'], p['physical_lock']['st_ino']),
                'Original physical lease inode changed')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return {'ready': True, 'status': 'AUTHORIZED_AFTER_KNOWN_GEMMA_FAILURE_SAFELY_CLOSED',
            'authorization': auth['_descriptor'], 'outer_wait': auth['outer_wait'],
            'workers': auth['workers'], 'failure_preserved': auth['failure_summary'],
            'same_inode_lease_free': True, 'scientific_configuration_changed': False}


def launch(plan_path, authorization_path, authorization_sha256):
    auth = verified({'path': authorization_path, 'sha256': authorization_sha256})
    auth['_descriptor'] = {'path': authorization_path, 'sha256': authorization_sha256}
    require(auth['launcher_sha256'] == file_sha256(__file__)
            and smoke.descriptor(plan_path) == auth['queue']
            and read(plan_path)['profiles'] == [auth['profile']], 'Launch/code binding changed')
    p = verified(auth['profile']); run_root = Path(p['output']).parent.resolve(strict=True)
    source = next(iter(verified(p['source_catalog'])['items'].values()))['content'][0]['path']
    require(str(run_root) == '/home/q/cvpr2027_runs_ssd/' + smoke.NAME
            and {run_root.stat().st_dev, Path(source).stat().st_dev,
                 Path(p['checkpoint']).stat().st_dev} == {66306},
            'Qwen weights/data/run must remain on the selected NVMe')
    original = smoke.predecessor_closed
    smoke.predecessor_closed = lambda plan, p: predecessor_admission(auth, plan, p)
    try:
        # Original launch retains science/freshness/SSD/resource checks, exclusive
        # intent, and the original GPU supervisor in a fresh Python process.
        return smoke.launch(plan_path)
    finally:
        smoke.predecessor_closed = original


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'authorization', 'authorization-sha256'):
        parser.add_argument('--' + key, required=True)
    args = parser.parse_args()
    print(json.dumps(launch(args.plan, args.authorization, args.authorization_sha256)), flush=True)
