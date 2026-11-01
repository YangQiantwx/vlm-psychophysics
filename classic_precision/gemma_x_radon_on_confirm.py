"""One newly authorized Radon ON96 attempt after the existing OFF queue closes.

Reuses the frozen ON engine and the accepted observational recorder. This module
adds only an independent budget, namespace and a terminal predecessor gate.
"""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import sys
import time

from . import gemma_x_recheck as core
from . import gemma_x_recheck_trace_v2 as v2
from . import gemma_x_recheck_observer_safe as observer
from . import gemma_x_recheck_recovery as recovery
from .cosmos_x_full_status import queue_owner_observation
from .gemma_speed import owned_worker_alive
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

BASE = 'gemma-x-three-card-recheck-20261031'
NAME = BASE + '/on-confirm1'
MODULE = 'classic_precision.gemma_x_radon_on_confirm'
PREDECESSOR_SHA = '6a21615f5cba4c5422d31b80a244042dbdeea59eab1d4374503f49d26d10531b'
ORIGINAL_SCIENCE = core.ORIGINAL_SCIENCE
trace = core.trace


def science(p, resolve=verified):
    spec = p['recheck']; template = resolve(spec['template_profile'])
    # Reuse the historical scientific validator with its own bound authorization.
    # The real dispatched profile below retains the NEW one-attempt96 authority.
    projected = copy.deepcopy(p)
    projected['execution_authorization'] = template['execution_authorization']
    ORIGINAL_SCIENCE(projected, resolve)
    core.qual.verify_code(p)
    require(spec['operations']['sha256'] == file_sha256(__file__)
            and spec['trace']['sha256'] == file_sha256(trace.__file__)
            and spec['trace_v2']['sha256'] == file_sha256(v2.__file__)
            and spec['observer_safe']['sha256'] == file_sha256(observer.__file__),
            'ON confirmation or observational implementation changed')
    require(p['card'] == 'radon0' and p['cold'] == 1 and p['maximum_model_calls'] == 96
            and len(p['assigned_ids']) == len(set(p['assigned_ids'])) == 96,
            'This authorization is exactly one Radon ON96 attempt')
    require({k:v for k,v in p.items() if k not in core.OPERATIONAL} ==
            {k:v for k,v in template.items() if k not in core.OPERATIONAL},
            'ON confirmation changed the existing ON scientific configuration')
    root = Path(p['workspace_root']); worker = 'radon0-recheck-cold1'
    require(p['worker'] == worker and p['output'] == str(root/'runs/reproducibility'/NAME/worker)
            and p['claim_root'] == str(root/'runs/reproducibility'/NAME/'claims'/worker)
            and p['control_file'] == str(root/'plans'/NAME/(worker+'-control.json')),
            'ON confirmation escaped its independent namespace')
    auth = resolve(p['execution_authorization'])
    require(auth['status'] == 'EXPLICIT_USER_AUTHORIZED_ON_REPEAT'
            and auth['independent_new_call_budget'] == 96 and auth['cold_runs'] == 1
            and auth['configuration_changes_from_original_ON'] == []
            and auth['preserve_all_original_answers'] and auth['point_deg'] == 1.5
            and auth['reference'] == spec['reference']
            and auth['predecessor_queue'] == spec['predecessor_queue']
            and auth['stop_on_first_same_configuration_output_difference'],
            'ON96 authorization or comparison policy changed')
    require(spec['predecessor_queue']['sha256'] == PREDECESSOR_SHA,
            'Different predecessor queue')
    reference = resolve(spec['reference']); table = resolve(p['reference_prompt_catalog'])
    require(set(reference['by_id']) == set(p['assigned_ids'])
            and reference['actual_prompt_token_ids'] == table['prompt_token_ids']['T3']
            and sum(x['raw_answer'].strip() == x['truth'] for x in reference['by_id'].values()) == 74,
            'Original ON96 reference changed')


def queue_check(path):
    q = read(path)
    require(q['recheck']['card'] == 'radon0' and q['recheck']['operations_sha256'] == file_sha256(__file__)
            and q['maximum_new_calls'] == 96 and not q['automatic_retry']
            and len(q['profiles']) == 1, 'Wrong finite ON96 queue')
    profiles = [verified(q['profiles'][0])]
    science(profiles[0])
    return q, profiles


def queue(path):
    q, profiles = queue_check(path)
    require(q['status'] == 'SEALED', 'Unsealed ON96 queue')
    p = profiles[0]
    require(not read(p['control_file'])['paused'], 'ON96 remains paused')
    result = core.supervise(q['profiles'][0]['path'])
    require(result['returncode'] == 0 and result['terminal_status'] == 'COMPLETE'
            and result['real_wait_observed'] and result['owned_engines_exited']
            and result['original_physical_lease_released'], 'ON96 did not close successfully; no retry')
    return {'status':'COMPLETE_ON_CONFIRMATION_QUEUE','profiles':q['profiles'],'worker_wait':result}


def validate_terminal(records):
    outer = records['outer_wait']; owner = records['outer_owner']
    require(outer['real_wait_observed'] and outer['returncode'] in (0, 1)
            and not owner.get('alive') and owner['status'] == 'OWNER_PROCESS_EXITED',
            'OFF queue has not really closed')
    require(len(records['workers']) == 2, 'Wrong OFF predecessor workers')
    for item in records['workers']:
        require(item['same_inode_lease_free'], 'OFF physical lease is not free')
        if not item['started']:
            require(item['claims'] == item['raw'] == 0 and item['wait'] is None,
                    'Unstarted predecessor has calls or a wait')
            continue
        w = item['wait']
        require(w and w['real_wait_observed'] and w['returncode'] in (0, 1)
                and w['owned_engines_exited'] and w['original_physical_lease_released']
                and w['profile_sha256'] == item['profile_digest']
                and item['live_owner_count'] == 0,
                'OFF worker closure missing or still owns a process')
        if outer['returncode'] == 0:
            require(item['result_status'] == 'COMPLETE' and w['returncode'] == 0
                    and item['claimed_ids'] == item['assigned_ids']
                    and set(w['completed_ids']) == set(item['assigned_ids']),
                    'Successful OFF queue does not have complete workers')
    require(any(x['started'] for x in records['workers']), 'OFF queue never started')
    if outer['returncode'] == 0:
        require(all(x['started'] for x in records['workers']), 'OFF success omitted a worker')


def predecessor(root, card):
    require(card == 'radon0', 'ON confirmation is Radon-only')
    root = Path(root); qp = root/'plans'/BASE/'autotune-off3/radon0-recheck-queue.json'
    require(file_sha256(qp) == PREDECESSOR_SHA, 'OFF predecessor changed')
    q = read(qp); run = root/'runs/reproducibility'/BASE/'autotune-off3'
    waitpath = run/'radon0-recheck-queue.wait.json'
    require(waitpath.exists(), 'HOLD_OFF_QUEUE_NOT_CLOSED')
    outer = read(waitpath)
    require(outer['queue'] == descriptor(qp), 'OFF outer wait/queue mismatch')
    owner = queue_owner_observation(read(run/'radon0-recheck-queue.owner.json'), core.boot_id())
    records = {'queue':descriptor(qp), 'outer_wait':outer, 'outer_owner':owner, 'workers':[]}
    for ref in q['profiles']:
        p = verified(ref); out = Path(p['output']); wp = out.with_suffix('.wait.json')
        claims = [read(x) for x in Path(p['claim_root']).glob('*.json')]
        require(all(x['profile_sha256'] == digest(p) for x in claims), 'Foreign predecessor claim')
        raw = [json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines()] if (out/'raw.jsonl').exists() else []
        lock = p['physical_lock']
        with open(lock['path'], 'rb') as f:
            st = os.fstat(f.fileno())
            require(st.st_dev == lock['st_dev'] and st.st_ino == lock['st_ino'], 'Original lease replaced')
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        started = out.with_suffix('.launch.json').exists() or wp.exists()
        engine_file = out/'engine-owner.json';exit_file = out/'exit.json'
        owned = read(engine_file)['owned'] if engine_file.exists() else []
        if exit_file.exists():
            owned = [{'worker':x} for x in read(exit_file).get('owned_workers',[])] or owned
        records['workers'].append({'profile':ref, 'profile_digest':digest(p), 'started':started, 'claims':len(claims), 'raw':len(raw),
            'claimed_ids':sorted(x['sample_id'] for x in claims), 'assigned_ids':sorted(p['assigned_ids']),
            'wait':read(wp) if wp.exists() else None, 'same_inode_lease_free':True,
            'live_owner_count':sum(owned_worker_alive(x['worker']) for x in owned),
            'result_status':read(out/'result.json')['status'] if (out/'result.json').exists() else None})
    validate_terminal(records)
    fences = []
    for ref in q['profiles']:
        p = verified(ref); before = read(p['control_file'])
        if not before['paused']:
            after = {**before, 'generation':before['generation']+1, 'paused':True,
                     'reason':'Closed OFF queue fenced before separately authorized ON96'}
            atomic_json(p['control_file'], after); fences.append({'path':p['control_file'],'before':before,'after':after})
    target = root/'runs/reproducibility'/NAME/'off-predecessor-closed-admission.json'
    records['controls_fenced'] = (read(target).get('controls_fenced',[]) if target.exists() else []) + fences
    atomic_json(target, records)
    return records


@contextmanager
def namespace():
    old = core.NAME, core.MODULE, core.science, core.queue_check, core.predecessor
    old_queue = core.qual.queue
    old_trace = trace.install, trace.close, trace.finish, observer.ORIGINAL_ADAPTER.prepare
    core.NAME, core.MODULE, core.science, core.queue_check, core.predecessor = NAME, MODULE, science, queue_check, predecessor
    core.qual.queue = queue
    trace.install, trace.close, trace.finish, observer.ORIGINAL_ADAPTER.prepare = v2.install, v2.close, observer.finish, observer.prepare_observed
    try: yield
    finally:
        core.NAME, core.MODULE, core.science, core.queue_check, core.predecessor = old
        core.qual.queue = old_queue
        trace.install, trace.close, trace.finish, observer.ORIGINAL_ADAPTER.prepare = old_trace


def prepare(root, authorization):
    root = Path(root); dest = root/'plans'/NAME; run = root/'runs/reproducibility'/NAME
    require(not dest.exists(), 'ON96 already prepared; never reseal')
    template_path = root/'plans'/core.G/'radon0-qualification-cold1-profile.json'
    p = copy.deepcopy(read(template_path)); worker = 'radon0-recheck-cold1';p.pop('worker_token', None)
    dest.mkdir(parents=True); run.mkdir(parents=True)
    require(run.resolve().is_relative_to((root/'runs/reproducibility'/BASE).resolve()), 'Unexpected ON storage')
    authpath = dest/'authorization.json';write_new(authpath, authorization)
    p.update(worker=worker, output=str(run/worker), claim_root=str(run/'claims'/worker),
             control_file=str(dest/(worker+'-control.json')), pause_file=str(dest/(worker+'.pause')),
             execution_authorization=descriptor(authpath))
    p['recheck']={'operations':descriptor(__file__), 'trace':descriptor(trace.__file__),
        'trace_v2':descriptor(v2.__file__), 'observer_safe':descriptor(observer.__file__),
        'template_profile':descriptor(template_path), 'reference':authorization['reference'],
        'predecessor_queue':authorization['predecessor_queue']}
    science(p); draft=dest/(worker+'-draft.json');profile=dest/(worker+'-profile.json')
    write_new(draft,p); storage=core.full.storage_admission(p)
    with namespace(), core.dispatch_context():core.qual.seal(str(draft),str(profile))
    science(read(profile));write_new(p['control_file'],{'generation':0,'paused':True,'reason':'Wait for OFF queue true terminal closure'})
    qp=dest/'radon0-recheck-queue.json'
    write_new(qp,{'status':'SEALED','automatic_retry':False,'maximum_new_calls':96,
        'profiles':[descriptor(profile)],'recheck':{'card':'radon0','operations_sha256':file_sha256(__file__)}})
    receipt={'status':'SEALED_PAUSED_NOT_LAUNCHED','queue':descriptor(qp),'profile':descriptor(profile),
             'authorization':descriptor(authpath),'physical_storage':storage,'new_model_calls':0}
    write_new(run/'stage-receipt.json',receipt);return receipt


def launch(root):
    with namespace():
        files=core.paths(root,'radon0')
        require(not any(p.exists() for p in files.values()), 'Already attempted; never redispatch')
        predecessor(root,'radon0')
        old=recovery.NAME,recovery.MODULE,recovery.science
        recovery.NAME,recovery.MODULE,recovery.science=NAME,MODULE,science
        try:window=recovery.resource_window(root)
        finally:recovery.NAME,recovery.MODULE,recovery.science=old
        require(window['status']=='READY_FOR_UNIQUE_LAUNCH','HOLD_RESOURCES')
        _, profiles=queue_check(Path(root)/'plans'/NAME/'radon0-recheck-queue.json')
        require(recovery.window_admitted(verified(window['window']),profiles[0],time.monotonic(),core.boot_id()),'Stale resource window')
        return core.launch(root,'radon0')


def main():
    command=sys.argv[1]
    if command in ('prepare','launch'):
        parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command')
        parser.add_argument('--root',required=True);parser.add_argument('--authorization');a=parser.parse_args()
        value=prepare(a.root,read(a.authorization)) if command=='prepare' else launch(a.root)
        print(json.dumps(value),flush=True);return
    require(command in ('queue-supervisor','queue','supervise','run'),'Unsupported command')
    with namespace():core.main()


if __name__=='__main__':main()
