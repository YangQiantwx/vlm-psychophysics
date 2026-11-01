"""Versioned V2 observation repair; no change to the selected scientific path.

Radon retains the authorized OFF192 comparison. Fourier retains ON384 and waits
for its existing formal predecessor. Retired deployment queues never restart.
"""
import argparse
from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import sys

from . import gemma_x_recheck as core
from . import gemma_x_recheck_autotune_off as off
from . import gemma_x_recheck_recovery as recovery
from . import gemma_x_recheck_trace as trace
from . import gemma_x_recheck_trace_v2 as v2
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, file_sha256

BASE = core.NAME
MODULE = 'classic_precision.gemma_x_recheck_recorder_repair'
OFF_SCIENCE, ON_SCIENCE, ON_RUN = off.science, core.science, core.run


def name(card):
    require(card in core.full.CARDS, 'Unknown card')
    return BASE + ('/autotune-off2' if card == 'radon0' else '/recorder-v2')


def validate_closure(value):
    require(value['status'] == 'ZERO_REQUEST_RECORDER_FAILURE_CLOSED_FENCED'
            and value['new_model_calls'] == 0 and value['model_load_started'] is True
            and value['exact_error'] == 'Sampler source changed; inspect before recording',
            'Not the observed post-load, zero-request recorder failure')
    require(value['queue']['sha256'] == '4d951420b0015e60c3d4d92ee1022e1420ca0a57496e1f2cad0ec4da9e1edea9'
            and value['outer_owner']['status'] == 'OWNER_PROCESS_EXITED'
            and not value['outer_owner']['alive']
            and value['outer_real_wait']['returncode'] == 1
            and value['outer_real_wait']['real_wait_observed'], 'Failed donor outer not closed')
    require([w['worker'] for w in value['workers']] ==
            ['radon0-recheck-cold1', 'radon0-recheck-cold2'], 'Wrong donor workers')
    for i, w in enumerate(value['workers']):
        require(w['claims'] == w['raw'] == w['scientific_INTENT'] == 0
                and w['original_same_inode_lease_free'], 'Claimed donor cannot be retried')
        wait = w['real_wait']
        if i == 0:
            require(w['model_load_started'] and wait['returncode'] == 1
                    and wait['real_wait_observed'] and wait['owned_engines_exited']
                    and wait['original_physical_lease_released'] and not wait['completed_ids'],
                    'Post-load worker not closed')
        else:
            require(not w['model_load_started'] and wait is None, 'Cold2 unexpectedly started')
    require(len(value['controls']) == 2 and all(c['after']['paused']
            and c['after']['generation'] == 1 for c in value['controls']), 'Donor not fenced')


def science(p, resolve=verified):
    spec = p['recheck']['observation_repair']
    require(spec['entrypoint']['sha256'] == file_sha256(__file__)
            and spec['trace_v2']['sha256'] == file_sha256(v2.__file__)
            and spec['scientific_changes_from_donor'] == [], 'Recorder version changed')
    if p['card'] == 'radon0':
        validate_closure(resolve(spec['donor_closed']))
        OFF_SCIENCE(p, resolve)
    else:
        ON_SCIENCE(p, resolve)


def run(profile):
    p = read(profile); science(p)
    if p['card'] == 'radon0': return off.run(profile)
    return ON_RUN(profile)


@contextmanager
def namespace(card):
    old_off = off.NAME, off.MODULE, off.science
    if card == 'radon0': off.NAME, off.MODULE = name(card), MODULE
    try:
        with off.namespace() if card == 'radon0' else nullcontext():
            old_core = core.NAME, core.MODULE, core.science, core.run
            old_recovery = recovery.NAME, recovery.MODULE, recovery.science
            old_trace = trace.install, trace.close
            core.NAME, core.MODULE, core.science, core.run = name(card), MODULE, science, run
            recovery.NAME, recovery.MODULE, recovery.science = name(card), MODULE, science
            if card == 'radon0': off.science = science
            trace.install, trace.close = v2.install, v2.close
            try: yield
            finally:
                core.NAME, core.MODULE, core.science, core.run = old_core
                recovery.NAME, recovery.MODULE, recovery.science = old_recovery
                trace.install, trace.close = old_trace
    finally:
        off.NAME, off.MODULE, off.science = old_off


def prepare(root, card):
    root = Path(root); dest = root/'plans'/name(card)
    require(not (dest/(card+'-recheck-queue.json')).exists(), 'Already sealed; never reseal')
    parent = root/'plans'/BASE
    donor = parent/('autotune-off1' if card == 'radon0' else '')/(card+'-recheck-queue.json')
    closure = root/'runs/reproducibility'/BASE/'autotune-off1-zero-request-closed.json'
    if card == 'radon0': validate_closure(read(closure))
    else:
        with namespace_original():
            require(not any(p.exists() for p in core.paths(root, card).values()),
                    'Original Fourier diagnostic was launched')
    profiles = [verified(ref) for ref in read(donor)['profiles']]
    controls = []
    for p in profiles:
        out = Path(p['output']); control = read(p['control_file'])
        require(not any(Path(p['claim_root']).rglob('*.json'))
                and not (out/'raw.jsonl').exists() and control['paused'],
                'Donor has requests or is active')
        if card != 'radon0':
            require(control['generation'] == 0 and not out.exists(), 'Fourier donor used')
        else: require(control['generation'] == 1, 'OFF1 donor not fenced')
    dest.mkdir(parents=True, exist_ok=True)
    for p in profiles:
        if card != 'radon0':
            before = read(p['control_file']); after = {**before, 'generation': 1,
                'fence_reason': 'Unlaunched legacy-sampler recorder replaced by V2 observation adapter'}
            atomic_json(p['control_file'], after)
            controls.append({'path': p['control_file'], 'before': before, 'after': after})
        p = copy.deepcopy(p); worker = p['worker']; p.pop('worker_token', None)
        p.update(output=str(root/'runs/reproducibility'/name(card)/worker),
                 claim_root=str(root/'runs/reproducibility'/name(card)/'claims'/worker),
                 control_file=str(dest/(worker+'-control.json')), pause_file=str(dest/(worker+'.pause')))
        p['recheck']['observation_repair'] = {'entrypoint': descriptor(__file__),
            'trace_v2': descriptor(v2.__file__), 'scientific_changes_from_donor': [],
            'donor_queue': descriptor(donor)}
        if card == 'radon0': p['recheck']['observation_repair']['donor_closed'] = descriptor(closure)
        write_new(dest/(worker+'-draft.json'), p)
    if card == 'radon0':
        (dest/'context-predecessor-summary-ref.json').write_bytes((parent/'context-predecessor-summary-ref.json').read_bytes())
    with namespace(card): receipt = core.stage(root, card)
    receipt['donor_fences'] = controls
    atomic_json(root/'runs/reproducibility'/BASE/(card+'-recorder-v2-preparation.json'), receipt)
    return receipt


@contextmanager
def namespace_original():
    previous = core.NAME; core.NAME = BASE
    try: yield
    finally: core.NAME = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'resource-window', 'launch',
                         'queue-supervisor', 'queue', 'supervise', 'run'])
    for key in ['root', 'card', 'plan', 'profile', 'window']: parser.add_argument('--'+key)
    a = parser.parse_args()
    card = a.card or (read(a.profile)['card'] if a.profile else
                     read(a.plan)['recheck']['card'] if a.plan else 'radon0')
    if a.command == 'prepare': value = prepare(a.root, card)
    else:
        with namespace(card):
            if a.command == 'resource-window':
                require(card == 'radon0', 'Resource window is Radon-only')
                value = recovery.resource_window(a.root)
            elif a.command == 'launch':
                value = recovery.launch(a.root, a.window) if card == 'radon0' else core.launch(a.root, card)
            else: core.main(); return
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
