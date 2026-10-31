"""One explicitly authorized retry of the reboot-interrupted Qwen sample.

Reuse the frozen reader, QSA, runtime, lifecycle and scientific result audit.
Only this deployment namespace/allocation changes. The original claim and
PARTIAL result remain immutable; this is never an automatic retry mechanism.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import socket
import subprocess

from . import qwen_remaining_t13 as q
from . import qwen_remaining_t13_results as results
from . import qwen_remaining_t13_archive as archive
from .refinement_worker import read, require, verified, descriptor
from .storage import file_sha256

TARGET = 't3_2e29ac8a4e90ee6c0f7f'
BATCH = 'reboot1-authorized-retry1'
MODULE = 'classic_precision.qwen_remaining_t13_single_retry'
WORKER = 'fourier0-full-T3-' + BATCH
_original_science = q.science


def science(p, resolve=None):
    resolve = archive.portable_resolver(p, resolve)
    if 'single_retry' not in p:
        return _original_science(p, resolve)
    entry = p['single_retry']
    require(entry['implementation']['sha256'] == file_sha256(__file__), 'Retry adapter changed')
    auth = resolve(entry['authorization'])
    require(auth['status'] == 'AUTHORIZED_SINGLE_RETRY'
            and auth['sample_ids'] == [TARGET] and auth['maximum_new_model_calls'] == 1
            and auth['user_answer'] == '仅补跑这1条，保留原记录（推荐）'
            and auth['preserve_original_claim'] is True, 'Missing exact human retry authorization')
    require(entry['original_claim'] == auth['original_claim']
            and entry['original_partial_summary'] == auth['original_partial_summary'], 'Unbound retry parent')
    claim = resolve(entry['original_claim']); partial = resolve(entry['original_partial_summary'])
    require(claim['sample_id'] == TARGET and claim['attempt'] == 1
            and partial['status'] == 'PARTIAL_CLASSIC_QWEN_T13_X_UNRESOLVED_CLAIM'
            and partial['missing_ids'] == partial['UNCERTAIN_ids'] == [TARGET]
            and partial['unique_videos'] == 6575, 'Original unresolved evidence changed')
    template = resolve(entry['runtime_profile'])
    _original_science(template, resolve)
    require(template['card'] == 'fourier0' and template['phase'] == 'full'
            and template['resource_group'] == 'T3', 'Wrong qualified runtime')
    operational = {'assigned_ids', 'maximum_model_calls', 'worker', 'output', 'claim_root',
                   'control_file', 'pause_file', 'source_catalog', 'single_retry',
                   'implementation', 'worker_token', 'media_files', 'operational_wrapper'}
    require(all(p.get(k) == template.get(k) for k in p.keys() | template.keys()
                if k not in operational), 'Retry changed a qualified scientific/deployment field')
    require(p['assigned_ids'] == [TARGET] and p['maximum_model_calls'] == 1
            and p['control_generation'] == 0 and not p['automatic_retry']
            and not p['formal_auto_resume'], 'Retry scope changed')
    root = Path(p['workspace_root']); plan = root / 'plans' / q.NAME / BATCH
    require(p['worker'] == WORKER
            and Path(p['output']) == root / 'runs/reproducibility' / q.NAME / WORKER
            and Path(p['claim_root']) == root / 'runs/reproducibility' / q.NAME / ('claims-' + BATCH)
            and Path(p['control_file']) == plan / 'control.json'
            and Path(p['pause_file']) == plan / 'PAUSE.json', 'Retry namespace changed')
    canonical = resolve(p['canonical_catalog'])['items']
    catalog = resolve(p['source_catalog'])['items']
    require(set(catalog) == {TARGET} and catalog[TARGET] == q.rebase(canonical[TARGET], str(root)),
            'Retry source/GT/prompt is not the original canonical sample')
    q.validate_source(TARGET, catalog[TARGET], 'T3')
    q.check_implementation(p['implementation_sources'])


@contextmanager
def context(execution=False):
    old_science = q.science; native = subprocess.Popen
    def popen(argv, **kwargs):
        if isinstance(argv, (list, tuple)):
            argv = list(argv)
            if '-m' in argv:
                i = argv.index('-m') + 1
                if argv[i] == 'classic_precision.qwen_remaining_t13': argv[i] = MODULE
        return native(argv, **kwargs)
    q.science = science
    if execution: subprocess.Popen = popen
    try: yield
    finally:
        q.science = old_science
        subprocess.Popen = native


def seal(draft, output):
    p = read(draft); science(p)
    require(p['hostname'] == socket.gethostname(), 'Seal on the actual execution host')
    with context(): return q.seal(draft, output)


def audit(root, profile):
    with context(): return results.audit_segment(root, profile)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal', 'activate', 'queue', 'supervise', 'run', 'audit-closed', 'pause'))
    for key in ('draft', 'output', 'plan', 'profile', 'root', 'control'): parser.add_argument('--' + key)
    a = parser.parse_args()
    if a.command == 'seal': answer = seal(a.draft, a.output)
    elif a.command == 'audit-closed':
        rows, proof = audit(a.root, a.profile)
        answer = {'status': 'CLOSED_PASS', 'records': rows, 'proof': proof}
    elif a.command == 'pause': answer = q.lifecycle.pause(a.control)
    else:
        if a.command in ('activate', 'queue'):
            plan = read(a.plan)
            require(len(plan['profiles']) == 1 and 'single_retry' in verified(plan['profiles'][0]),
                    'Only the one authorized retry queue is allowed')
        else:
            require('single_retry' in read(a.profile), 'Only the authorized retry profile is allowed')
        with context(execution=True):
            answer = getattr(q, a.command)(a.plan if a.command in ('activate', 'queue') else a.profile)
    print(json.dumps(answer), flush=True)


if __name__ == '__main__': main()
