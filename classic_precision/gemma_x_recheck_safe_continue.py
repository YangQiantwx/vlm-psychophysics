"""Continue only unclaimed diagnostic IDs after a retained-answer recorder error."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys
from . import gemma_x_recheck_recorder_repair as repair
from . import gemma_x_recheck_observer_safe as safe
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

core, off, recovery, trace = repair.core, repair.off, repair.recovery, repair.trace
BASE = repair.BASE
MODULE = 'classic_precision.gemma_x_recheck_safe_continue'
SCIENCE = repair.science
COLD1_REFERENCE = off.cold1_reference


def name(card):
    require(card in core.full.CARDS, 'Unknown card')
    return BASE + ('/autotune-off3' if card == 'radon0' else '/recorder-safe')


def validate_prefix(closed, prefix, donor):
    require(closed['status'] == 'ONE_SAVED_RAW_RECORDER_FAILURE_CLOSED_FENCED'
            and closed['model_calls'] == 1 and closed['never_claimed_remainder'] == [95, 96]
            and closed['do_not_repeat_retained_id']
            and closed['outer_owner']['status'] == 'OWNER_PROCESS_EXITED'
            and not closed['outer_owner']['alive'] and closed['outer_real_wait']['returncode'] == 1
            and closed['outer_real_wait']['real_wait_observed'], 'Saved-answer donor not closed')
    require([w['claims'] for w in closed['workers']] == [1, 0]
            and [w['raw'] for w in closed['workers']] == [1, 0]
            and all(w['original_same_inode_lease_free'] for w in closed['workers'])
            and closed['workers'][1]['real_wait'] is None, 'Unexpected donor calls')
    w = closed['workers'][0]['real_wait']
    require(w['returncode'] == 1 and w['real_wait_observed'] and w['owned_engines_exited']
            and w['original_physical_lease_released'], 'Donor worker not closed')
    require(len(closed['controls']) == 2 and all(c['after']['paused']
            and c['after']['generation'] == 1 for c in closed['controls']), 'Donor not fenced')
    require(prefix['status'] == 'SAVED_RAW_AND_INPUT_RECOVERED_WITHOUT_NEW_CALL'
            and prefix['extra_model_calls'] == 0 and prefix['original_failure_and_wait_unchanged']
            and prefix['actual_CPU_NATIVE_GPU_180_bytes_equal']
            and prefix['actual_60_layers_and_native_greedy_observed']
            and prefix['full_output_comparison']['accepted']
            and prefix['raw']['sample_id'] == closed['retained_id'] == donor['assigned_ids'][0]
            and prefix['raw']['profile_sha256'] == digest(donor), 'Retained answer not independently bound')


def science(p, resolve=verified):
    spec = p['recheck']['safe_observer']
    require(spec['entrypoint']['sha256'] == file_sha256(__file__)
            and spec['observer']['sha256'] == file_sha256(safe.__file__)
            and spec['scientific_changes_from_donor'] == [], 'Safe observer version changed')
    projected = p
    if p['card'] == 'radon0':
        closed, prefix, donor = (resolve(spec[k]) for k in ['donor_closed', 'retained_prefix', 'prefix_profile'])
        validate_prefix(closed, prefix, donor)
        require(p['assigned_ids'] == donor['assigned_ids'][1 if p['cold'] == 1 else 0:]
                and p['maximum_model_calls'] == (95 if p['cold'] == 1 else 96),
                'Continuation repeats a claimed ID or changes the remaining budget')
        projected = copy.deepcopy(p)
        projected['assigned_ids'] = donor['assigned_ids']
        projected['maximum_model_calls'] = 96
    SCIENCE(projected, resolve)


def cold1_reference(p):
    result = COLD1_REFERENCE(p)
    spec = p['recheck']['safe_observer']
    prefix = verified(spec['retained_prefix']); donor = verified(spec['prefix_profile'])
    validate_prefix(verified(spec['donor_closed']), prefix, donor)
    sid = prefix['raw']['sample_id']
    require(sid not in result and len(result) == 95, 'Retained answer duplicated or continuation incomplete')
    result[sid] = prefix['raw']
    require(set(result) == set(donor['assigned_ids']), 'Combined first group is not original96')
    return result


@contextmanager
def namespace(card):
    old_repair = repair.name, repair.MODULE, repair.science
    repair.name, repair.MODULE, repair.science = name, MODULE, science
    try:
        with repair.namespace(card):
            old = trace.finish, safe.ORIGINAL_ADAPTER.prepare, off.cold1_reference
            trace.finish, safe.ORIGINAL_ADAPTER.prepare = safe.finish, safe.prepare_observed
            if card == 'radon0': off.cold1_reference = cold1_reference
            try: yield
            finally: trace.finish, safe.ORIGINAL_ADAPTER.prepare, off.cold1_reference = old
    finally: repair.name, repair.MODULE, repair.science = old_repair


def prepare(root, card):
    root = Path(root); base = root/'plans'/BASE
    variant = 'autotune-off2' if card == 'radon0' else 'recorder-v2'
    parent = base/variant; donor = parent/(card+'-recheck-queue.json')
    dest = root/'plans'/name(card)
    require(not (dest/(card+'-recheck-queue.json')).exists(), 'Already sealed; never reseal')
    q = read(donor); profiles = [verified(x) for x in q['profiles']]
    closed = root/'runs/reproducibility'/BASE/'autotune-off2-recorder-failure-closed.json'
    prefix = root/'runs/reproducibility'/BASE/'autotune-off2-retained-first-answer.json'
    if card == 'radon0': validate_prefix(read(closed), read(prefix), profiles[0])
    else:
        donor_run = root/'runs/reproducibility'/BASE/variant
        require(not any(donor_run.glob(card+'-recheck-queue.*.json')), 'Fourier diagnostic donor launched')
    for i,p in enumerate(profiles):
        c = read(p['control_file'])
        require(c['paused'] and c['generation'] == (1 if card == 'radon0' else 0), 'Donor control differs')
        claims = list(Path(p['claim_root']).glob('*.json'))
        require(len(claims) == (1 if card == 'radon0' and i == 0 else 0), 'Unexpected donor claim')
        if claims: require(read(claims[0])['sample_id'] == profiles[0]['assigned_ids'][0], 'Wrong retained ID')
    dest.mkdir(parents=True, exist_ok=True); fences = []
    for p in profiles:
        if card != 'radon0':
            before = read(p['control_file']); after = {**before, 'generation':1,
                'fence_reason':'Unused diagnostic superseded by isolated telemetry and JSON-safe recorder'}
            atomic_json(p['control_file'], after)
            fences.append({'path':p['control_file'],'before':before,'after':after})
        p = copy.deepcopy(p); worker=p['worker']; p.pop('worker_token',None)
        p.update(output=str(root/'runs/reproducibility'/name(card)/worker),
                 claim_root=str(root/'runs/reproducibility'/name(card)/'claims'/worker),
                 control_file=str(dest/(worker+'-control.json')),pause_file=str(dest/(worker+'.pause')))
        spec={'entrypoint':descriptor(__file__),'observer':descriptor(safe.__file__),
              'scientific_changes_from_donor':[],'donor_queue':descriptor(donor)}
        if card == 'radon0':
            spec.update(donor_closed=descriptor(closed),retained_prefix=descriptor(prefix),prefix_profile=q['profiles'][0],
                first_group='retained1 plus never-claimed95; separate engine starts disclosed',new_calls=191)
            if p['cold'] == 1: p['assigned_ids']=p['assigned_ids'][1:];p['maximum_model_calls']=95
        p['recheck']['safe_observer']=spec
        write_new(dest/(worker+'-draft.json'),p)
    if card == 'radon0':
        (dest/'context-predecessor-summary-ref.json').write_bytes((base/'context-predecessor-summary-ref.json').read_bytes())
    with namespace(card): receipt=core.stage(root,card)
    receipt['donor_fences']=fences
    atomic_json(root/'runs/reproducibility'/BASE/(card+'-safe-observer-preparation.json'),receipt)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','resource-window','launch','queue-supervisor','queue','supervise','run'])
    for key in ['root','card','plan','profile','window']:parser.add_argument('--'+key)
    a=parser.parse_args();card=a.card or (read(a.profile)['card'] if a.profile else read(a.plan)['recheck']['card'] if a.plan else 'radon0')
    if a.command=='prepare':value=prepare(a.root,card)
    else:
        with namespace(card):
            if a.command=='resource-window':
                require(card=='radon0','Resource window is Radon-only');value=recovery.resource_window(a.root)
            elif a.command=='launch':value=recovery.launch(a.root,a.window) if card=='radon0' else core.launch(a.root,card)
            else:core.main();return
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
