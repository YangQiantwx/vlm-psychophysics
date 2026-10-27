"""Continue only audited unattempted formal IDs after the reader smoke is reported.

Creates fresh profiles/queues/output directories and retains every old PAUSE,
claim and original answer. Qualified refinement workers and model math stay intact.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import fcntl
import os
from pathlib import Path
import subprocess
import sys

from .refinement_worker import require, read, write_new, descriptor, verified
from .storage import atomic_json, digest, file_sha256

AUTHORIZATION = '好了以后生成汇报和你的判断，然后继续我们没完成的补点'
ORDERS = {'fourier0': ['qwen-fourier0', 'gemma-fourier0'],
          'fourier1': ['gemma-fourier1', 'cosmos-fourier1'],
          'radon0': ['nemo-radon0-v2', 'gemma-radon0']}


def remaining_ids(profile, audit, current_claims):
    """Reject ambiguity; a held claim is never converted back to an unattempted ID."""
    assigned=profile['assigned_ids'];universe=set(assigned)
    require(len(universe)==len(assigned), 'Duplicate original allocation')
    require(not audit['UNCERTAIN_ids'], 'UNCERTAIN IDs require diagnosis, never automatic continuation')
    done=set(audit['completed_ids']);unattempted=set(audit['unattempted_ids'])
    require(not done & unattempted and done | unattempted==universe, 'Original ID partition differs')
    require(done==set(audit['claimed_ids'])==set(current_claims)&universe, 'Claims changed since pause audit')
    require(len(audit['completed_ids'])==len(done) and len(audit['unattempted_ids'])==len(unattempted), 'Duplicate audited ID')
    if done:
        w=audit['real_wait']
        require(w and w['real_wait_observed'] and w['returncode']==0 and w['owned_engines_exited']
            and w['original_physical_lease_released'] and w['terminal_status']=='PAUSED'
            and w['completed_ids']==audit['completed_ids'], 'Original pause did not close')
    return [sid for sid in assigned if sid in unattempted]


def validate_handoff(summary, decision, summary_sha):
    require(summary['status']=='COMPLETE_PASS' and summary['expected']==summary['technical_pass']==1728,
            'Reader smoke is incomplete; do not start formal continuation')
    require(decision['schema']=='reader-route-decision-v1' and decision['status']=='REPORTED'
        and decision['summary_sha256']==summary_sha
        and decision['formal_action']=='CONTINUE_EXISTING_CONTROLLED_PROTOCOL'
        and decision['authorization_message']==AUTHORIZATION
        and decision.get('judgment') and decision.get('reported_HST'), 'Bound report/judgment or current user authorization missing')
    require(len(summary['closures'])==15 and all(w['returncode']==0 and w['real_wait_observed']
        and w['owned_engines_exited'] and w['original_physical_lease_released']
        and w['terminal_status']=='COMPLETE' for w in summary['closures']), 'Smoke engines did not all close')


def prepare(root, host, report_path, decision_path):
    root=Path(root);plan=root/'plans/classic-refinement'
    hold=root/'plans/reader-route-smoke-20261026/nemo-reproduction-HOLD.json'
    require(not hold.exists() or read(hold)['status']=='RESOLVED',
            'Nemo reproduction diagnosis is unresolved; formal continuation held')
    summary,decision=read(report_path),read(decision_path)
    validate_handoff(summary,decision,file_sha256(report_path))
    formal=root/'runs/reproducibility/classic-refinement-20261026'
    audit_path=formal/f'pause-for-reader-smoke-{host}.json';audit=read(audit_path)
    require(audit['status']=='PAUSED_USER_REQUEST' and not audit['formal_auto_resume'], 'Wrong original pause audit')
    cards=['fourier0','fourier1'] if host=='fourier' else ['radon0']
    allocation={'schema':'classic-refinement-continuation-allocation-v1','assignments':{},
        'parent_pause_audit':descriptor(audit_path),'smoke_report':descriptor(report_path),
        'reported_decision':descriptor(decision_path),'automatic_retry':False}
    profiles={}
    for card in cards:
        current=summary.get('card_completion',{}).get(card)
        queue_path=Path(current['queue_path']) if current else root/'plans/reader-route-smoke-20261026'/(card+'-queue.json')
        if current:require(file_sha256(queue_path)==current['queue_sha256'], 'Current smoke queue changed since report')
        smoke_queue=read(queue_path)
        require(smoke_queue['status']=='COMPLETE', 'Card smoke queue is not complete')
        last=read(smoke_queue['profiles'][-1]);last_wait=Path(last['output']).with_suffix('.wait.json')
        w=read(last_wait)
        require(w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited']
            and w['original_physical_lease_released'] and w['terminal_status']=='COMPLETE'
            and w['completed_ids']==last['assigned_ids'], 'Latest card smoke closure differs')
        for name in ORDERS[card]:
            old_path=plan/(name+'-profile.json');p=read(old_path);a=audit['profiles'][name]
            require(read(p['pause_file'])=={'worker_token':p['worker_token'],'action':'PAUSE'}, 'Original PAUSE missing or changed')
            claims={x.stem for x in Path(p['claim_root']).glob('*.json')}
            ids=remaining_ids(p,a,claims)
            if Path(p['output']).exists():
                require(read(Path(p['output']).with_suffix('.wait.json'))==a['real_wait'], 'Original wait changed')
                require(a['real_wait']['profile_sha256']==digest(p), 'Original profile binding differs')
            lock=p['physical_lock']
            with Path(lock['path']).open('r+b') as f:
                st=os.fstat(f.fileno())
                require((st.st_dev,st.st_ino)==(lock['st_dev'],lock['st_ino']), 'Original lease identity changed')
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if not ids: continue
            new=deepcopy(p);worker=name+'-cont1';path=plan/(worker+'-profile.json')
            new.update(assigned_ids=ids, maximum_model_calls=len(ids),
                worker=p['worker']+'-cont1',
                profile_file=str(path), output=str(formal/worker),
                pause_file=str(plan/(worker+'-PAUSE.json')),
                admission_file=str(plan/(worker+'-admission.json')),
                worker_token=digest({'parent':descriptor(old_path),'decision':descriptor(decision_path)}),
                assignment_sha256=digest(ids),
                continuation={'parent_profile':descriptor(old_path),'pause_audit':descriptor(audit_path),
                    'original_worker':p['worker'],
                    'completed_before':len(a['completed_ids']),'unattempted_only':True,
                    'reported_decision':descriptor(decision_path),'same_original_GPU':True},
                preceding_smoke_wait=descriptor(last_wait))
            if p['model']=='nemo':
                from .nemo_kernel_replay import CONTRACT
                canonical=root/'runs/reproducibility/nemo-canonical-restoration-20261027/canonical-audit.json'
                require(read(canonical)['status']=='COMPLETE_PASS', 'Nemo canonical restoration not accepted')
                new['continuation']['nemo_legacy_runtime']={
                    'kernel_config':CONTRACT,'canonical_audit':descriptor(canonical),
                    'entry_source_sha256':{name:file_sha256(Path(__file__).with_name(name)) for name in
                        ('nemo_canonical_restore.py','nemo_kernel_replay.py')}}
            require(not Path(new['output']).exists() and not Path(new['pause_file']).exists(), 'Continuation already exists')
            allocation['assignments'][new['worker']]=ids
            profiles[name]=(path,new)
    allocation_path=plan/f'continuation1-{host}-allocation.json'
    require(sum(map(len,allocation['assignments'].values()))=={'fourier':7399,'radon':3124}[host],
            'Frozen10523-call continuation scope changed')
    write_new(allocation_path,allocation)
    for path,p in profiles.values():
        p['allocation']=descriptor(allocation_path);write_new(path,p)
    queues={}
    for card in cards:
        stages=[str(profiles[n][0])for n in ORDERS[card]if n in profiles]
        path=plan/(card+'-cont1-queue.json')
        write_new(path,{'status':'SEALED','stages':stages,'automatic_retry':False,
            'reported_decision':descriptor(decision_path),'smoke_report':descriptor(report_path)})
        queues[card]=str(path)
    return {'status':'PREPARED_NOT_LAUNCHED','host':host,'queues':queues,
        'new_unattempted_calls':sum(map(len,allocation['assignments'].values())),'automatic_retry':False}


def queue(queue_path):
    q=read(queue_path);require(q['status']=='SEALED', 'Existing queue cannot be replayed automatically')
    decision=verified(q['reported_decision']);summary=verified(q['smoke_report'])
    validate_handoff(summary,decision,q['smoke_report']['sha256'])
    from .refinement_dispatch import compiler_preflight
    reports=[];prior_wait=None;q['status']='RUNNING';atomic_json(queue_path,q)
    try:
        for path in q['stages']:
            p=read(path)
            require(not Path(p['pause_file']).exists(), 'Queued continuation paused')
            require(not Path(p['output']).exists(), 'Continuation output already exists')
            require(not any((Path(p['claim_root'])/(s+'.json')).exists()for s in p['assigned_ids']), 'An ID was claimed after preparation; do not retry')
            if p['environment'].get('FLASHINFER_NVCC'):compiler_preflight(p['environment'])
            if p['model']=='gemma':
                previous=prior_wait or p['preceding_smoke_wait']['path'];w=read(previous)
                require(w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited']
                    and w['original_physical_lease_released'] and w['terminal_status']=='COMPLETE', 'Predecessor did not close')
                write_new(p['admission_file'],{'plan_sha256':p['plan_sha256'],'profile_sha256':digest(p),
                    'qwen_t1_complete':3840,'all_owned_engines_exited':True,'physical_gpu_locks_released':True,
                    'preceding_wait':descriptor(previous),'scope':'Latest owned predecessor on the same GPU; native same-inode lease/live headroom rechecked at load'})
            module='classic_precision.refinement_continue' if p['model']=='nemo' else 'classic_precision.refinement_worker'
            command='supervise-nemo' if p['model']=='nemo' else 'supervise'
            child=subprocess.Popen([sys.executable,'-B','-m',module,command,'--profile',str(path)])
            code=child.wait();prior_wait=Path(p['output']).with_suffix('.wait.json');w=read(prior_wait);reports.append(w)
            require(code==0 and w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited']
                and w['original_physical_lease_released'], 'Worker failed or closure incomplete; no retry')
            if w['terminal_status']=='PAUSED':
                q.update(status='PAUSED',reports=reports);atomic_json(queue_path,q);return {'status':'PAUSED'}
            require(w['terminal_status']=='COMPLETE' and w['completed_ids']==p['assigned_ids'], 'Finite ID completion differs')
        q.update(status='COMPLETE',reports=reports);atomic_json(queue_path,q)
    except BaseException as error:
        q.update(status='HELD_NO_RETRY',reports=reports,error=str(error));atomic_json(queue_path,q);raise
    return {'status':'COMPLETE','stages':len(reports)}


def run_nemo(path):
    """Keep the original formal worker, with the accepted legacy runtime gate."""
    p=read(path);require(p['model']=='nemo','Nemo continuation only')
    from .nemo_kernel_replay import CONTRACT
    runtime=p['continuation']['nemo_legacy_runtime']
    require(runtime['kernel_config']==CONTRACT,'Nemo legacy backend contract changed')
    audit=verified(runtime['canonical_audit'])
    require(audit['status']=='COMPLETE_PASS' and audit['technical_pass']==1268,
            'Nemo canonical restoration not accepted')
    for name,sha in runtime['entry_source_sha256'].items():
        require(file_sha256(Path(__file__).with_name(name))==sha,'Qualified Nemo runtime entry changed')
    decision=verified(p['continuation']['reported_decision'])
    report=Path(runtime['canonical_audit']['path']).parents[1]/'reader-route-smoke-20261026/comparison/summary.json'
    validate_handoff(read(report),decision,file_sha256(report))
    root=Path(path).parents[2]
    require(read(root/'plans/reader-route-smoke-20261026/nemo-reproduction-HOLD.json')['status']=='RESOLVED',
            'Nemo scientific hold remains')
    from .nemo_canonical_restore import legacy_session
    from . import refinement_worker
    with legacy_session(p['output']):return refinement_worker.run(path)


def supervise_nemo(path):
    from . import refinement_worker as worker
    original=worker.subprocess.Popen
    def popen(argv,**kwargs):
        argv=list(argv)
        at=argv.index('classic_precision.refinement_worker')
        argv[at:at+2]=['classic_precision.refinement_continue','run-nemo']
        return original(argv,**kwargs)
    worker.subprocess.Popen=popen
    try:return worker.supervise(path)
    finally:worker.subprocess.Popen=original


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','queue','run-nemo','supervise-nemo'])
    p.add_argument('--root');p.add_argument('--host',choices=['fourier','radon'])
    p.add_argument('--report');p.add_argument('--decision');p.add_argument('--queue');p.add_argument('--profile')
    a=p.parse_args()
    result=(prepare(a.root,a.host,a.report,a.decision) if a.command=='prepare' else
            queue(a.queue) if a.command=='queue' else run_nemo(a.profile) if a.command=='run-nemo'
            else supervise_nemo(a.profile))
    import json
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
