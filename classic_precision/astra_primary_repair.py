"""Explicitly authorized60 failed primary cases; preserve every original attempt."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
import copy
import fcntl
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import uuid
from . import astra_video_eval as base
from .astra_video_results import audit_case

AUTHORIZATION='只补主协议60条（推荐）'
VERSION='primary-repair1'
CAPACITY_AUTHORIZATION='model at capacity，这种你开个新的cli重跑就行'


def locations(root,version=VERSION):
    p,o=base.paths(root)
    base.require(version==VERSION or re.fullmatch(r'capacity-retry[1-9]\d*',version),'Unknown repair version')
    return p/version,o/'repairs'/version


def eligible(samples,audits):
    selected=[]
    for s in samples:
        if s['task']=='T3' and s['cell'][1] in [1.75,2.25]:continue
        a=audits[s['case_id']]
        if a['visual_only_trace_status']=='PASS_OBSERVED':continue
        base.require(a['protocol_status'] in ['ENVIRONMENT_FAILED_NO_VALID_VISUAL_EVALUATION','EXECUTION_FAILED'],
                     'Only terminal failed cases are eligible')
        selected.append(s)
    base.require(Counter(s['task'] for s in selected)=={'T1':20,'T2':22,'T3':18},'Exact60 primary gap changed')
    return selected


def prepare(root):
    oldp,oldo,oldplan,samples=base.load(root);p,o=locations(root)
    closure=base.read(oldo/'segments/quota-resume1/completion-closure.json')
    base.require(closure['status']=='COMPLETE_DRAINED' and closure['active']==0,'Original scheduler still active')
    ledger=base.closed_ledger(oldo);base.require(len(ledger)==924,'Original attempt ledger incomplete')
    audits={s['case_id']:base.read(oldo/'trace-audits'/(s['case_id']+'.json')) for s in samples}
    selected=eligible(samples,audits)
    base.require(not p.exists() and not o.exists(),'Repair already exists; never overwrite/replay')
    p.mkdir();o.mkdir(parents=True);new=[]
    for old in selected:
        base.require(audits[old['case_id']]['identity']['result_sha256']==ledger[old['case_id']]['result'],
                     'Audited failure changed')
        case=copy.deepcopy(old);case['prior_case_id']=old['case_id'];case['attempt']=2
        case['case_id']=uuid.uuid4().hex;directory=Path('/tmp')/uuid.uuid4().hex;directory.mkdir()
        video=directory/(case['case_id']+'.mkv')
        base.require(base.sha(old['video_path'])==old['source_sha256'],'Original neutral source changed')
        shutil.copyfile(old['video_path'],video);base.require(base.sha(video)==old['source_sha256'],'Source copy changed')
        case.update(public_directory=str(directory),video_path=str(video));new.append(case)
    for spec in oldplan['prompts'].values():shutil.copyfile(oldp/spec['file'],p/spec['file'])
    base.write(p/'samples.json',new,exclusive=True)
    plan=copy.deepcopy(oldplan);plan.update(maximum_sessions=60,counts={'T1':20,'T2':22,'T3':18},
        samples_sha256=base.sha(p/'samples.json'),version=VERSION,authorization=AUTHORIZATION)
    base.write(p/'plan.json',plan,exclusive=True)
    manifest={'schema':'astra-primary-repair-v1','status':'SEALED','human_authorization':AUTHORIZATION,
        'maximum_new_sessions':60,'automatic_retry':False,'model':base.MODEL,'effort':base.EFFORT,
        'old_ledger_sha256':base.digest(ledger),'original_plan_sha256':base.sha(oldp/'plan.json'),
        'original_samples_sha256':base.sha(oldp/'samples.json'),'plan_sha256':base.sha(p/'plan.json'),
        'samples_sha256':base.sha(p/'samples.json'),'original_case_ids':[s['case_id'] for s in selected],
        'excluded_auxiliary_angles':[1.75,2.25],'original_answers_preserved':924,
        'new_attempt':2,'same_scientific_prompt':True,'maximum_concurrent':20,
        'worker_sha256':base.sha(base.__file__),'controller_sha256':base.sha(__file__), 'created_HST':base.hst()}
    base.write(p/'manifest.json',manifest,exclusive=True)
    shutil.copyfile(base.__file__,o/'execution-worker.py');shutil.copyfile(__file__,o/'execution-controller.py')
    return manifest


def load(root,version=VERSION):
    p,o=locations(root,version);m=base.read(p/'manifest.json');plan=base.read(p/'plan.json');samples=base.read(p/'samples.json')
    capacity=m.get('kind')=='provider_capacity_only';budget=m['maximum_new_sessions']
    base.require(m['human_authorization']==(CAPACITY_AUTHORIZATION if capacity else AUTHORIZATION)
                 and (1<=budget<=60 if capacity else budget==60) and not m['automatic_retry'],
                 'Explicit finite authorization mismatch')
    base.require(base.sha(p/'plan.json')==m['plan_sha256'] and base.sha(p/'samples.json')==m['samples_sha256'],
                 'Sealed repair inputs changed')
    base.require(len(samples)==budget and len({s['prior_case_id'] for s in samples})==budget and
        all((s['attempt']>=3 if capacity else s['attempt']==2) and not(s['task']=='T3' and s['cell'][1] in [1.75,2.25]) for s in samples),
        'Attempt/primary scope mismatch')
    return p,o,m,plan,samples


async def run(root,codex,version=VERSION):
    p,o,m,plan,samples=load(root,version);oldp,oldo,_,_=base.load(root);budget=len(samples)
    base.require(base.sha(base.__file__)==m['worker_sha256'] and base.sha(__file__)==m['controller_sha256'],
                 'Frozen implementation changed')
    base.require(base.digest(base.closed_ledger(oldo))==m['old_ledger_sha256'],'Original attempts changed')
    base.require(base.sha(oldp/'plan.json')==m['original_plan_sha256'],'Original plan changed')
    if m.get('kind')=='provider_capacity_only':
        parent=Path(m['parent_output'])
        base.require(base.digest(base.closed_ledger(parent))==m['parent_ledger_sha256'],'Capacity-failed parent changed')
        for s in samples:
            result=base.read(parent/'cases'/s['prior_attempt_case_id']/'result.json')
            base.require(result['returncode']!=0 and result['answer'] is None
                and any('Selected model is at capacity' in str(e) for e in result['errors']),
                'Only exact provider-capacity failures can be retried')
    with (oldo/'scheduler.lock').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.require(not (o/'scheduler-started.json').exists() and not (o/'PAUSE.json').exists(),'Already started/paused')
        base.require(base.read(oldo/'cpu-admission/receipt.json')['status']=='PASS','CLI CPU admission missing')
        base.write(o/'scheduler-started.json',{'pid':os.getpid(),'model':base.MODEL,'effort':base.EFFORT,
            'maximum_concurrent':min(20,budget),'scheduled':budget,'sandbox':'danger-full-access','HST':base.hst(),
            'codex_version':subprocess.check_output([codex,'--version'],text=True).strip(),
            'lock_identity':[os.fstat(lock.fileno()).st_dev,os.fstat(lock.fileno()).st_ino]},exclusive=True)
        def pause():base.write(o/'PAUSE.json',{'reason':'Operator requested safe drain','HST':base.hst()})
        for sig in [signal.SIGINT,signal.SIGTERM]:asyncio.get_running_loop().add_signal_handler(sig,pause)
        def paused():
            if shutil.disk_usage('/tmp').free<plan['minimum_free_disk_GiB']*2**30:
                base.write(o/'PAUSE.json',{'reason':'Reserved disk headroom','HST':base.hst()})
            return (o/'PAUSE.json').exists()
        def progress(left,active,closed):base.write(o/'progress.json',{'remaining_unclaimed':left,
            'active_sessions':active,'closed_sessions':closed,'HST':base.hst()})
        async def execute(case,slot):
            try:return await base.one(root,plan,case,codex,slot,o,run_paths=(p,o))
            except Exception as exc:
                base.write(o/'PAUSE.json',{'reason':'Coordinator failure','case_id':case['case_id'],'error':repr(exc)})
                raise
        done,pending=await base.dynamic_pool(samples,min(20,budget),execute,paused,progress)
        progress(len(pending),0,len(done))
        base.write(o/'scheduler-final.json',{'status':'COMPLETE' if len(done)==budget and not pending else 'PAUSED_DRAINED',
            'closed':len(done),'unclaimed':len(pending),'all_children_waited':True,'HST':base.hst()},exclusive=True)


def analyze(root,version=VERSION):
    p,o,m,plan,samples=load(root,version);_,oldo,_,original=base.load(root);budget=len(samples)
    base.require(base.digest(base.closed_ledger(oldo))==m['old_ledger_sha256'],'Original attempts changed')
    if m.get('kind')=='provider_capacity_only':
        base.require(base.digest(base.closed_ledger(Path(m['parent_output'])))==m['parent_ledger_sha256'],'Parent attempts changed')
    rows=[audit_case(root,s,run_paths=(p,o,plan)) for s in samples if (o/'cases'/s['case_id']/'result.json').exists()]
    mapping={s['sample_id']:s for s in samples};current=[]
    by_new={a['case_id']:a for a in rows}
    parent_records={a['sample_id']:a for a in base.read(Path(m['parent_output'])/'canonical-primary-records.json')} if m.get('kind')=='provider_capacity_only' else {}
    for s in original:
        if s['task']=='T3' and s['cell'][1] in [1.75,2.25]:continue
        new=mapping.get(s['sample_id']);a=by_new.get(new['case_id']) if new else None
        if a is None:a=parent_records.get(s['sample_id']) or base.read(oldo/'trace-audits'/(s['case_id']+'.json'))
        current.append(a)
    good=[a for a in current if a['visual_only_trace_status']=='PASS_OBSERVED' and a['strict_format']]
    terminal=False
    if len(rows)==budget and (o/'completion-closure.json').exists():
        close=base.read(o/'completion-closure.json');wait=base.read(o/'supervisor.wait.json')
        terminal=(close['status']=='COMPLETE_DRAINED' and close['scheduler_lock_free']
            and close['scheduler_pid_exited'] and wait['real_wait'] and wait['returncode']==0
            and close['source_final_sha256']==base.sha(o/'scheduler-final.json'))
        base.require(len(base.closed_ledger(o))==budget,'New attempt ledger incomplete')
    summary={'status':'COMPLETE_PRIMARY_VISUAL_ANSWERS' if len(good)==916 and terminal else 'PARTIAL_PRIMARY_REPAIR',
        'repair_closed':len(rows),'repair_pass':sum(a['visual_only_trace_status']=='PASS_OBSERVED' for a in rows),
        'primary_planned':916,'primary_valid':len(good),'primary_missing':916-len(good),
        'by_task':{t:{'valid':sum(a['task']==t for a in good),'correct':sum(a['task']==t and a['correct'] for a in good)} for t in ['T1','T2','T3']},
        'full180':sum(a['protocol_status']=='FULL180_TRACE_PASS' for a in good),'preserved_original_attempts':924,
        'terminal_closure_verified':terminal}
    base.write(o/'summary.json',summary);base.write(o/'canonical-primary-records.json',current)
    return summary


def prepare_capacity(root,version,parent_version):
    base.require(version!=VERSION and version!=parent_version,'New capacity version required')
    pp,po,pm,parent_plan,parent_samples=load(root,parent_version);p,o=locations(root,version)
    base.require(base.read(po/'completion-closure.json')['status']=='COMPLETE_DRAINED','Parent not closed')
    records=base.read(po/'canonical-primary-records.json');by_id={s['case_id']:s for s in parent_samples};selected=[]
    for a in records:
        if a['visual_only_trace_status']=='PASS_OBSERVED':continue
        base.require(a['case_id'] in by_id,'Unresolved case is not in the closed parent')
        old=by_id[a['case_id']];result=base.read(po/'cases'/old['case_id']/'result.json')
        base.require(result['returncode']!=0 and result['answer'] is None
            and any('Selected model is at capacity' in str(e) for e in result['errors']),
            'Non-capacity scientific/protocol failure requires separate handling')
        selected.append(old)
    base.require(0<len(selected)<=60 and not p.exists() and not o.exists(),'No new finite capacity batch or already exists')
    p.mkdir();o.mkdir(parents=True);new=[]
    for old in selected:
        case=copy.deepcopy(old);case.update(prior_attempt_case_id=old['case_id'],attempt=old['attempt']+1,case_id=uuid.uuid4().hex)
        directory=Path('/tmp')/uuid.uuid4().hex;directory.mkdir();video=directory/(case['case_id']+'.mkv')
        base.require(base.sha(old['video_path'])==old['source_sha256'],'Source changed');shutil.copyfile(old['video_path'],video)
        base.require(base.sha(video)==old['source_sha256'],'Fresh source copy changed')
        case.update(public_directory=str(directory),video_path=str(video));new.append(case)
    for spec in parent_plan['prompts'].values():shutil.copyfile(pp/spec['file'],p/spec['file'])
    base.write(p/'samples.json',new,exclusive=True);plan=copy.deepcopy(parent_plan)
    plan.update(maximum_sessions=len(new),counts=dict(Counter(s['task'] for s in new)),samples_sha256=base.sha(p/'samples.json'),
                version=version,authorization=CAPACITY_AUTHORIZATION)
    base.write(p/'plan.json',plan,exclusive=True);manifest=copy.deepcopy(pm)
    manifest.update(kind='provider_capacity_only',human_authorization=CAPACITY_AUTHORIZATION,
        maximum_new_sessions=len(new),parent_output=str(po),parent_version=parent_version,
        parent_ledger_sha256=base.digest(base.closed_ledger(po)),plan_sha256=base.sha(p/'plan.json'),
        samples_sha256=base.sha(p/'samples.json'),original_case_ids=[s['prior_case_id'] for s in new],
        new_attempt=new[0]['attempt'],maximum_concurrent=min(20,len(new)),worker_sha256=base.sha(base.__file__),
        controller_sha256=base.sha(__file__),created_HST=base.hst())
    base.write(p/'manifest.json',manifest,exclusive=True);shutil.copyfile(base.__file__,o/'execution-worker.py')
    shutil.copyfile(__file__,o/'execution-controller.py');return manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','prepare-capacity','run','analyze']);p.add_argument('--root',required=True)
    p.add_argument('--version',default=VERSION);p.add_argument('--parent-version',default=VERSION)
    p.add_argument('--codex',default='/home/yang/.local/bin/codex');a=p.parse_args()
    if a.action=='run':asyncio.run(run(a.root,a.codex,a.version))
    else:
        result=prepare(a.root) if a.action=='prepare' else prepare_capacity(a.root,a.version,a.parent_version) if a.action=='prepare-capacity' else analyze(a.root,a.version)
        print(__import__('json').dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
