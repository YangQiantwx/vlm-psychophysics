"""Versioned zero-call deployment recovery; original Gemma science stays frozen."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from . import gemma_x_recheck as core
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

BASE = 'gemma-x-three-card-recheck-20261031'
NAME = BASE + '/execution-repair1'
MODULE = 'classic_precision.gemma_x_recheck_recovery'
ORIGINAL_SCIENCE = core.science
DONOR_QUEUE_SHA = 'c6d192dde2a84e809f12edae5bddc6191487d6bb99e5724138b3e213f2605497'


def validate_closure(value):
    require(value['status'] == 'ZERO_CALL_STARTUP_RESOURCE_FAILURE_CLOSED_FENCED'
            and value['new_model_calls'] == 0 and not value['model_load_started'],
            'Recovery is only for the observed zero-call startup failure')
    require(value['outer_owner']['status'] == 'OWNER_PROCESS_EXITED'
            and value['outer_owner']['alive'] is False
            and value['outer_real_wait']['returncode'] == 1
            and value['outer_real_wait']['real_wait_observed'], 'Donor outer is not closed')
    require([w['worker'] for w in value['workers']] ==
            ['radon0-recheck-cold1', 'radon0-recheck-cold2'], 'Wrong donor workers')
    for w in value['workers']:
        require(w['claims'] == w['raw'] == w['scientific_INTENT'] == 0
                and not w['model_load_started'] and w['original_same_inode_lease_free'],
                'A claimed or model-loaded donor cannot use this recovery')
    w = value['workers'][0]['real_wait']
    require(w['returncode'] == 1 and w['real_wait_observed'] and w['owned_engines_exited']
            and w['original_physical_lease_released'] and not w['completed_ids'], 'Donor worker not closed')
    require(value['workers'][1]['real_wait'] is None, 'Donor cold2 was started')
    require(len(value['controls']) == 2 and all(c['after']['paused']
            and c['after']['generation'] == 1 for c in value['controls']), 'Donor controls not fenced')


def science(p, resolve=verified):
    require(p['card'] == 'radon0', 'Recovery is Radon-only')
    entry = p['recheck']['startup_recovery']
    require(entry['entrypoint']['sha256'] == file_sha256(__file__), 'Recovery entrypoint changed')
    validate_closure(resolve(entry['closure']))
    donor = resolve(entry['donor_profile'])
    require({k: v for k, v in p.items() if k not in core.OPERATIONAL} ==
            {k: v for k, v in donor.items() if k not in core.OPERATIONAL},
            'Recovery altered the sealed scientific configuration')
    require(p['assigned_ids'] == donor['assigned_ids'] and len(p['assigned_ids']) == 96,
            'Recovery panel or budget differs')
    ORIGINAL_SCIENCE(p, resolve)


@contextmanager
def namespace():
    old = core.NAME, core.MODULE, core.science
    core.NAME, core.MODULE, core.science = NAME, MODULE, science
    try:
        yield
    finally:
        core.NAME, core.MODULE, core.science = old


def prepare(root):
    root = Path(root); old = root / 'plans' / BASE; dest = root / 'plans' / NAME
    closure = root / 'runs/reproducibility' / BASE / 'radon0-zero-call-startup-closed.json'
    validate_closure(read(closure))
    qp = old / 'radon0-recheck-queue.json'
    require(file_sha256(qp) == DONOR_QUEUE_SHA, 'Donor queue changed')
    require(not dest.exists(), 'Recovery already prepared; never reseal')
    donors = [(ref, verified(ref)) for ref in read(qp)['profiles']]
    for ref, p in donors:
        c = read(p['control_file']); out = Path(p['output'])
        require(c['paused'] and c['generation'] == 1, 'Donor control revived')
        require(not any(Path(p['claim_root']).glob('*.json')), 'Donor has claims')
        for name in ('raw.jsonl', 'events.jsonl'):
            f = out / name
            require(not f.exists() or not f.read_text().strip(), 'Donor has scientific records')
        require(not any((out / name).exists() for name in
                ('public-settings.json', 'engine-owner.json', 'load.json', 'runtime.json')),
                'Donor reached model loading')
    require(read(Path(donors[0][1]['output']).with_suffix('.wait.json'))['profile_sha256']
            == digest(donors[0][1]), 'Donor wait/profile mismatch')
    dest.mkdir(parents=True)
    for ref, donor in donors:
        p = copy.deepcopy(donor); worker = p['worker']
        p.update(output=str(root/'runs/reproducibility'/NAME/worker),
                 claim_root=str(root/'runs/reproducibility'/NAME/'claims'/worker),
                 control_file=str(dest/(worker+'-control.json')),
                 pause_file=str(dest/(worker+'.pause')))
        p.pop('worker_token', None)
        p['recheck']['startup_recovery'] = {'entrypoint': descriptor(__file__),
            'closure': descriptor(closure), 'donor_profile': ref,
            'new_scientific_call_budget': 0, 'unchanged_remaining_Radon_budget': 192}
        write_new(dest/(worker+'-draft.json'), p)
    # The accepted T proof still points to its existing immutable copy.
    (dest/'context-predecessor-summary-ref.json').write_bytes(
        (old/'context-predecessor-summary-ref.json').read_bytes())
    with namespace():
        receipt = core.stage(root, 'radon0')
    receipt['original_zero_call_closure'] = descriptor(closure)
    return receipt


def window_admitted(value, profile, now, boot):
    samples = value['samples']; budgets = profile['resources']
    return (value['boot_id'] == boot and value['profile_sha256'] == digest(profile)
            and len(samples) >= 3 and samples[-1]['monotonic']-samples[0]['monotonic'] >= 20
            and 0 <= now-samples[-1]['monotonic'] <= 60
            and all(s['gpu_uuid'] == profile['gpu_uuid']
                    and s['free_gpu_gib'] >= budgets['min_free_gpu_gib']
                    and s['available_ram_gib'] >= budgets['min_available_ram_gib'] for s in samples))


def resource_window(root):
    root = Path(root)
    with namespace():
        _, profiles = core.queue_check(root/'plans'/NAME/'radon0-recheck-queue.json')
        require(not any(p.exists() for p in core.paths(root,'radon0').values()),
                'Already attempted; do not poll or relaunch this deployment')
    p = profiles[0]; samples = []; start = time.monotonic()
    # One bounded 21-second CPU observation per monitoring period, no model/context.
    for index in range(22):
        if index: time.sleep(max(0, start+index-time.monotonic()))
        result = subprocess.run(['nvidia-smi','-i',p['gpu_uuid'],
            '--query-gpu=uuid,memory.free','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=3)
        require(result.returncode == 0, 'Bounded resource observation failed')
        uuid, free = [s.strip() for s in result.stdout.strip().split(',')]
        available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                             if x.startswith('MemAvailable:')))/1024**2
        samples.append({'monotonic':time.monotonic(),'gpu_uuid':uuid,
                        'free_gpu_gib':float(free)/1024,'available_ram_gib':available})
    value = {'profile_sha256':digest(p),'boot_id':core.boot_id(),'samples':samples,
             'new_model_calls':0,'thresholds':p['resources']}
    value['admitted'] = window_admitted(value,p,time.monotonic(),core.boot_id())
    directory = root/'runs/reproducibility'/NAME
    path = directory/('resource-window-'+str(time.time_ns())+'.json')
    write_new(path,value)
    return {'status':'READY_FOR_UNIQUE_LAUNCH' if value['admitted'] else 'HOLD_RESOURCES',
            'window':descriptor(path),'minimum_free_gpu_gib':min(s['free_gpu_gib'] for s in samples),
            'maximum_free_gpu_gib':max(s['free_gpu_gib'] for s in samples),'new_model_calls':0}


def launch(root, window):
    root = Path(root)
    with namespace():
        _, profiles = core.queue_check(root/'plans'/NAME/'radon0-recheck-queue.json')
        require(window_admitted(read(window),profiles[0],time.monotonic(),core.boot_id()),
                'No fresh continuous original-threshold resource window')
        return core.launch(root,'radon0')


def recorded_snapshot(original, path):
    def wrapped(*args, **kwargs):
        value = original(*args, **kwargs)
        with Path(path).open('a') as f:
            f.write(json.dumps({'monotonic':time.monotonic(),'snapshot':value})+'\n')
            f.flush()
        return value
    return wrapped


def main():
    command = sys.argv[1]
    if command in ('prepare','resource-window','launch'):
        parser=argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command');parser.add_argument('--root',required=True)
        parser.add_argument('--window');a=parser.parse_args()
        value = prepare(a.root) if command=='prepare' else (
            resource_window(a.root) if command=='resource-window' else launch(a.root,a.window))
        print(json.dumps(value));return
    require(command in ('queue-supervisor','queue','supervise','run'), 'Unsupported recovery command')
    with namespace():
        if command != 'run':
            core.main();return
        from . import inference
        profile=read(sys.argv[sys.argv.index('--profile')+1]);science(profile)
        original=inference.resource_snapshot
        inference.resource_snapshot=recorded_snapshot(original,
            Path(profile['output']).with_suffix('.resource-observations.jsonl'))
        try: core.main()
        finally: inference.resource_snapshot=original


if __name__ == '__main__':
    main()
