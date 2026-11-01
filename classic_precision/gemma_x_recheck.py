"""Finite, independently claimed three-card recheck of the existing T3 1.5 point.

Uses the frozen Gemma qualification engine/input/sampling implementation. This
module owns allocation, real queue waits, observational diagnostics and admission.
"""
import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from . import gemma_x_full as full
from . import gemma_x_qualification as qual
from . import gemma_x_fourier_operations as ops
from . import gemma_x_recheck_trace as trace
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, file_sha256, owner_identity, digest


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

NAME = 'gemma-x-three-card-recheck-20261031'
MODULE = 'classic_precision.gemma_x_recheck'
G = 'gemma-x-full-20261030'
H = 'gemma-x-provisional-20261031'
L = 'gemma-x-fourier-takeover-20261031'
T = 'qwen-x-context-smoke-20261031'
ORIGINAL_SCIENCE = qual.science
OPERATIONAL = {'worker', 'output', 'claim_root', 'control_file', 'pause_file',
               'worker_token', 'media_files', 'execution_authorization', 'recheck'}


def science(p, resolve=verified):
    ORIGINAL_SCIENCE(p, resolve)
    qual.verify_code(p)
    spec = p['recheck']; template = resolve(spec['template_profile'])
    require(spec['operations']['sha256'] == file_sha256(__file__)
            and spec['trace']['sha256'] == file_sha256(trace.__file__), 'Recheck implementation changed')
    require({k: v for k, v in p.items() if k not in OPERATIONAL} ==
            {k: v for k, v in template.items() if k not in OPERATIONAL},
            'Recheck changed the existing X scientific configuration')
    root = Path(p['workspace_root']); worker = p['card'] + '-recheck-cold' + str(p['cold'])
    require(p['worker'] == worker and p['output'] == str(root/'runs/reproducibility'/NAME/worker)
            and p['claim_root'] == str(root/'runs/reproducibility'/NAME/'claims'/worker)
            and p['control_file'] == str(root/'plans'/NAME/(worker+'-control.json')),
            'Recheck escaped its independent output/claim/control namespace')
    auth = resolve(p['execution_authorization'])
    require(auth['independent_new_call_budget'] == 576 and auth['point_deg'] == 1.5
            and auth['preserve_all_original_answers'] and auth['configuration_changes'] == []
            and auth['observation_only'] and auth['reference'] == spec['reference'],
            'Recheck authorization or baseline differs')
    reference = resolve(spec['reference'])
    table = resolve(p['reference_prompt_catalog'])
    require(set(reference['by_id']) == set(p['assigned_ids']) and len(reference['by_id'])==96
            and reference['actual_prompt_token_ids'] == table['prompt_token_ids']['T3']
            and sum(x['raw_answer'].strip()==x['truth'] for x in reference['by_id'].values())==74,
            'Existing full96 Fourier reference changed')


@contextmanager
def dispatch_context():
    old_science, old_supervise = qual.science, qual.supervise
    qual.science, qual.supervise = science, supervise
    try: yield
    finally: qual.science, qual.supervise = old_science, old_supervise


def run(profile):
    p = read(profile); science(p)
    with dispatch_context(), trace.observing(p): return qual.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    original = worker.subprocess.Popen
    def launch(argv, *args, **kwargs):
        argv = [MODULE if x == 'classic_precision.refinement_worker' else x for x in argv]
        return original(argv, *args, **kwargs)
    worker.subprocess.Popen = launch
    try: return worker.supervise(profile)
    finally: worker.subprocess.Popen = original


def prepare(root):
    root = Path(root); arc = full.Archive(root)
    directory = root/'plans'/NAME; runroot = root/'runs/reproducibility'/NAME
    require(not directory.exists() and not runroot.exists(), 'Recheck already prepared')
    directory.mkdir(); runroot.mkdir()
    original_plan = read(root/'plans'/G/'qualification-plan-exec1.json')
    selection = arc.ref(original_plan['selection']); ids = selection['assigned_ids']
    template = arc.ref(original_plan['workers'][0]['profile'])
    records = [json.loads(x) for x in (arc.path(template['output'])/'raw.jsonl').read_text().splitlines()]
    require([r['sample_id'] for r in records] == ids and len(records) == 96,
            'Original Fourier reference is not the full96')
    prompts = records[0]['actual_prompt_token_ids']
    require(all(r['actual_prompt_token_ids'] == prompts for r in records), 'Reference prompts differ')
    keys = ['raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason', 'truth', 'question', 'generation']
    reference = {'source': descriptor(arc.path(template['output'])/'raw.jsonl'),
                 'actual_prompt_token_ids': prompts,
                 'by_id': {r['sample_id']: {k: r[k] for k in keys} for r in records}}
    write_new(directory/'reference.json', reference)
    draft_workers = []
    for card in full.CARDS:
        entries = [w for w in original_plan['workers'] if w['card'] == card]
        remote = arc.ref(entries[0]['profile'])['workspace_root']; remote_dir = Path(remote)/'plans'/NAME
        ref = {**descriptor(directory/'reference.json'), 'path': str(remote_dir/'reference.json')}
        auth = {'status': 'AUTHORIZED', 'maximum_qualification_calls': 576,
                'independent_new_call_budget': 576, 'point_deg': 1.5,
                'qualification_selection': arc.ref(entries[0]['profile'])['qualification_selection'],
                'reference': ref, 'configuration_changes': [], 'observation_only': True,
                'preserve_all_original_answers': True,
                'human_instructions': ['三个都得一致吧', '选1.5°：复查已知74/96的同一点',
                                       '最好的是a协议也有的点', '你这次测的时候监测所有东西，看看什么导致gemma没法reproduce'],
                'selection_note': 'User chose the existing A67/96 and FourierX74/96 point; no1.75/2.25 or2deg panel',
                'formal_denominator_unchanged': 9456, 'extra_inference_probes': 0}
        auth_file = directory/(card+'-authorization.json'); write_new(auth_file, auth)
        auth_ref = {**descriptor(auth_file), 'path': str(remote_dir/auth_file.name)}
        for w in entries:
            p = copy.deepcopy(arc.ref(w['profile'])); worker = card+'-recheck-cold'+str(w['cold'])
            p.update(worker=worker, output=str(Path(remote)/'runs/reproducibility'/NAME/worker),
                claim_root=str(Path(remote)/'runs/reproducibility'/NAME/'claims'/worker),
                control_file=str(remote_dir/(worker+'-control.json')),
                pause_file=str(remote_dir/(worker+'.pause')), execution_authorization=auth_ref,
                recheck={'template_profile': w['profile'], 'reference': ref,
                    'operations': {'path': str(Path(remote)/'code/vlm-psychophysics/classic_precision/gemma_x_recheck.py'),
                                   'sha256': file_sha256(__file__)},
                    'trace': {'path': str(Path(remote)/'code/vlm-psychophysics/classic_precision/gemma_x_recheck_trace.py'),
                              'sha256': file_sha256(trace.__file__)}})
            p.pop('worker_token', None)
            path = directory/(worker+'-draft.json'); write_new(path, p)
            draft_workers.append({'card': card, 'cold': w['cold'], 'draft': descriptor(path)})
    plan = {'schema': full.QUALIFICATION_SCHEMA, 'maximum_model_calls': 576,
            'unique_inputs': 96, 'selection': original_plan['selection'], 'draft_workers': draft_workers,
            'status': 'PREPARED_NOT_LAUNCHED', 'original_plan': descriptor(root/'plans'/G/'qualification-plan-exec1.json'),
            'formal_results_unchanged': True,
            'scheduling': 'R after accepted T closure; F each after current H segment1 COMPLETE, before L'}
    write_new(directory/'plan.json', plan)
    write_new(runroot/'preparation-current.json', {'status': plan['status'], 'plan': descriptor(directory/'plan.json'),
        'point_deg':1.5, 'A_correct':67, 'Fourier_X_reference_correct':74, 'unique_ids':96,
        'maximum_new_model_calls':576, 'actual_new_model_calls':0, 'diagnostics':descriptor(trace.__file__),
        'scheduling':plan['scheduling'], 'monitor':'cvpr-4', 'original_failures_retained':True})
    return {'status': plan['status'], 'drafts': len(draft_workers), 'point_deg':1.5, 'new_model_calls':0}


def stage(root, card):
    root = Path(root); directory = root/'plans'/NAME; runroot = root/'runs/reproducibility'/NAME
    require(card in full.CARDS, 'Unknown card')
    if card == 'radon0':
        physical = (root/'runs/reproducibility'/T).resolve().parent/NAME
        physical.mkdir(exist_ok=True)
        if not runroot.exists(): runroot.symlink_to(physical, target_is_directory=True)
        require(runroot.resolve() == physical, 'Radon recheck run must use the accepted NVMe run parent')
    else: runroot.mkdir(exist_ok=True)
    refs = []; proofs = []
    for cold in (1, 2):
        worker = card+'-recheck-cold'+str(cold)
        draft = directory/(worker+'-draft.json'); p = read(draft)
        science(p); proofs.append(full.storage_admission(p))
        profile = directory/(worker+'-profile.json')
        with dispatch_context(): qual.seal(str(draft), str(profile))
        science(read(profile)); refs.append(descriptor(profile))
        write_new(p['control_file'], {'generation':0, 'paused':True,
                  'reason':'Await exact predecessor closure and once-only recheck launch'})
    queue = directory/(card+'-recheck-queue.json')
    write_new(queue, {'status':'SEALED', 'automatic_retry':False, 'profiles':refs,
        'recheck': {'card':card, 'operations_sha256':file_sha256(__file__)}})
    receipt = {'status':'SEALED_PAUSED_NOT_LAUNCHED', 'queue':descriptor(queue), 'profiles':refs,
               'physical_storage':proofs, 'model_calls':0}
    write_new(runroot/(card+'-stage-receipt.json'), receipt)
    return receipt


def queue_check(path):
    q = read(path); require(q['recheck']['operations_sha256'] == file_sha256(__file__), 'Operations changed')
    profiles = [verified(r) for r in q['profiles']]
    require(len(profiles) == 2 and [p['cold'] for p in profiles] == [1,2]
            and {p['card'] for p in profiles} == {q['recheck']['card']}, 'Wrong finite queue')
    for p in profiles: science(p)
    return q, profiles


def paths(root, card):
    run = Path(root)/'runs/reproducibility'/NAME; prefix = card+'-recheck-queue'
    return {name:run/(prefix+'.'+name+'.json') for name in ['launch-intent','launch','owner','wait']}


def predecessor(root, card):
    root = Path(root); arc = full.Archive(root)
    if card != 'radon0':
        _, _, p = ops.predecessor_gate(arc, root/'plans'/L/(card+'-full-queue.json'),
            root/'plans'/H/'resume1'/(card+'-full-segment1-queue.json'))
        return {'status':'COMPLETE_FORMAL_PREDECESSOR_ACCEPTED', 'profile':p['worker']}
    qpath = root/'plans'/T/'radon-queue.json'
    require(file_sha256(qpath) == 'f013de1dcefc53d7b6f719484b192ff2cd507207c68db630a68e375c3e234d3e',
            'Context predecessor queue changed')
    summary = verified(read(root/'plans'/NAME/'context-predecessor-summary-ref.json'))
    require(summary['status'] == 'PASS_QWEN_T1_CONTEXT_POINT' and summary['new_model_calls'] == 48,
            'Context48 has not passed its existing raw audit')
    q = read(qpath); p = verified(q['profiles'][0]); out = Path(p['output'])
    result = read(out/'result.json'); require(result['status'] == 'COMPLETE', 'Context not COMPLETE')
    exit_record = read(out.with_suffix('.wait.json'))
    require(exit_record['returncode'] == 0 and exit_record['real_wait_observed']
            and exit_record['owned_engines_exited'] and exit_record['original_physical_lease_released']
            and exit_record['profile_sha256']==digest(p)
            and exit_record['completed_ids']==p['assigned_ids'], 'Context worker closure missing')
    outer = read(root/'runs/reproducibility'/T/'queue-wait.json')
    require(outer['returncode'] == 0 and outer['real_wait_observed'], 'Context outer wait missing')
    from .cosmos_x_full_status import queue_owner_observation
    owner = read(root/'runs/reproducibility'/T/'queue-owner.json')
    observation = queue_owner_observation(owner, boot_id())
    require(not observation.get('alive'), 'Context supervisor still alive')
    lock = p['physical_lock']
    with open(lock['path'], 'rb') as f:
        st = os.fstat(f.fileno())
        require(st.st_dev == lock['st_dev'] and st.st_ino == lock['st_ino'], 'Original physical lock replaced')
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return {'status':'CONTEXT48_AUDITED_AND_CLOSED', 'outer_wait':outer, 'owner':observation}


def launch(root, card):
    root = Path(root); queue = root/'plans'/NAME/(card+'-recheck-queue.json')
    q, profiles = queue_check(queue); files = paths(root, card)
    require(not any(p.exists() for p in files.values()), 'Queue already attempted; no redispatch')
    admission = predecessor(root, card)
    for p in profiles:
        require(read(p['control_file'])['paused'], 'Recheck must still be paused before launch')
        full.storage_admission(p)
    resources = ops.resource_preflight(profiles[0])
    write_new(files['launch-intent'], {'status':'INTENT_EXCLUSIVE_NO_RETRY', 'queue':descriptor(queue),
        'predecessor':admission, 'resources':resources, 'owner':owner_identity()})
    with dispatch_context(): qual.activate(str(queue))
    p = profiles[0]; env, cwd = ops.environment(p)
    with files['launch'].with_suffix('.log').open('xb') as log:
        child = subprocess.Popen([p['python'],'-B','-m',MODULE,'queue-supervisor','--root',str(root),'--card',card],
            env=env,cwd=cwd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    value = {'status':'LAUNCHED_ONCE', 'queue':descriptor(queue),'pid':child.pid,'boot_id':boot_id(),
             'python':p['python'], 'process_start_ticks':Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19],
             'automatic_retry':False}
    write_new(files['launch'],value);return value


def queue_supervisor(root, card):
    root = Path(root); queue = root/'plans'/NAME/(card+'-recheck-queue.json')
    q, profiles = queue_check(queue); files = paths(root,card)
    require(read(files['launch-intent'])['queue'] == descriptor(queue), 'Launch intent binding differs')
    p = profiles[0]; require(sys.executable == p['python'], 'Supervisor must use sealed GPU interpreter')
    write_new(files['owner'],owner_identity());env,cwd=ops.environment(p)
    child=subprocess.Popen([p['python'],'-B','-m',MODULE,'queue','--plan',str(queue)],env=env,cwd=cwd,stdin=subprocess.DEVNULL)
    code=child.wait();write_new(files['wait'],{'returncode':code,'real_wait_observed':True,
        'child_pid':child.pid,'queue':descriptor(queue),'owner':owner_identity()})
    return code


def summarize(root, plan_path):
    """Use the original strict six-worker audit, writing only this recheck root."""
    root=Path(root);arc=full.Archive(root);plan=read(plan_path)
    require(len(plan['workers'])==6,'Expected six explicitly sealed new workers')
    old_name,old_writer=qual.NAME,qual.atomic_json
    pending={}
    def capture(path,value):
        require(Path(path).parent==root/'runs/reproducibility'/NAME,'Audit escaped recheck scope')
        pending[Path(path).name]=value
    try:
        qual.NAME=NAME
        qual.atomic_json=capture
        with dispatch_context():qual.summarize(root,plan_path)
    finally:qual.NAME,qual.atomic_json=old_name,old_writer
    out=root/'runs/reproducibility'/NAME
    summary=pending['qualification-summary.json']
    diagnostics=[]
    for card in full.CARDS:
        queue=root/'plans'/NAME/(card+'-recheck-queue.json')
        wait=read(paths(root,card)['wait'])
        require(wait['returncode']==0 and wait['real_wait_observed']
                and wait['queue']['sha256']==file_sha256(queue),'Real outer queue closure missing')
    for selected in plan['workers']:
        p=arc.ref(selected['profile']);worker=arc.path(p['output'])
        reference=arc.ref(p['recheck']['reference'])
        for raw in (json.loads(x) for x in (worker/'raw.jsonl').read_text().splitlines()):
            expected=reference['by_id'][raw['sample_id']]
            require(all(raw[k]==v for k,v in expected.items())
                    and raw['actual_prompt_token_ids']==reference['actual_prompt_token_ids'],
                    'Independent replay found an original-reference difference')
        rows=[json.loads(x) for x in (worker/'output-equivalence.jsonl').read_text().splitlines()]
        require([x['sample_id'] for x in rows]==p['assigned_ids']
                and all(x['full_output_prompt_equal'] for x in rows),'Original reference mismatch retained')
        inputs=[json.loads(x) for x in (worker/'diagnostic-input-equivalence.jsonl').read_text().splitlines()]
        require([x['sample_id'] for x in inputs]==p['assigned_ids']
                and all(x['actual_CPU_GPU_per_frame_bytes_equal'] for x in inputs),'Missing whole-point GPU byte observations')
        requests=[json.loads(x) for x in (worker/'diagnostic-requests.jsonl').read_text().splitlines()]
        require([x['sample_id'] for x in requests]==p['assigned_ids'],'Missing diagnostic requests')
        kernel_count=0
        for row in requests:
            for w in row['workers']:
                evidence=arc.ref(w['proof']);require(evidence['sample_id']==row['sample_id'],'Mixed diagnostic IDs')
                require(file_sha256(arc.path(w['tensor_bundle']['path']))==w['tensor_bundle']['sha256'],
                        'Observed tensor bundle changed')
                kernel_count+=int(evidence['actual_CUDA_kernel_events']>0)
        before=read(worker/'diagnostics/after-load.json')['weights']['tensors']
        after=read(worker/'diagnostics/before-exit.json')['weights']['tensors']
        b={(x['kind'],x['name']):x for x in before};a={(x['kind'],x['name']):x for x in after}
        changes=[list(k) for k in sorted(set(a)|set(b)) if a.get(k)!=b.get(k)]
        diagnostics.append({'worker':p['worker'],'requests':len(requests),
            'requests_with_actual_CUDA_kernel_events':kernel_count,'weight_or_buffer_changes':changes})
    summary.update(status='PASS_GEMMA_X_RECHECK_THREE_CARDS_TWO_COLD'
        if not summary['full_output_differences'] and not summary['prompt_token_differences'] else summary['status'],
        diagnostic_coverage=diagnostics,original_reference_equal_calls=576,
        previous_failed_runs_retained=True,formal_9456_replaced_or_rerun=False)
    summary.pop('limits',None)
    atomic_json(out/'qualification-A-X-pairs.json',pending['qualification-A-X-pairs.json'])
    atomic_json(out/'qualification-summary.json',summary)
    return {k:summary[k] for k in ['status','executed_model_calls','correct_by_worker','diagnostic_coverage']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','stage','launch','queue-supervisor','queue','supervise','run','summarize'])
    for key in ['root','card','plan','profile']:parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare':value=prepare(a.root)
    elif a.command=='stage':value=stage(a.root,a.card)
    elif a.command=='launch':value=launch(a.root,a.card)
    elif a.command=='queue-supervisor':sys.exit(queue_supervisor(a.root,a.card))
    elif a.command=='run':value=run(a.profile)
    elif a.command=='supervise':value=supervise(a.profile)
    elif a.command=='summarize':value=summarize(a.root,a.plan)
    else:
        queue_check(a.plan)
        with dispatch_context():value=qual.queue(a.plan)
    print(json.dumps(value),flush=True)


if __name__=='__main__':main()
