"""One-shot predecessor-gated finite successor; no scientific retry."""
from __future__ import annotations
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

from . import cosmos_t45 as study
from .refinement_worker import read, verified, require, descriptor, write_new
from .storage import atomic_json, owner_identity, file_sha256


def qualified_request_check(p):
    """Guard the effective override, not just the request default."""
    old=verified(p['qualified_profile'])
    require(p.get('request_mm_processor_kwargs_by_task',{})
            ==old.get('request_mm_processor_kwargs_by_task',{}),
            'Unqualified task-specific processor override')
    for task in ('T4','T5'):
        actual=p.get('request_mm_processor_kwargs_by_task',{}).get(task,p['request_mm_processor_kwargs'])
        expected=old.get('request_mm_processor_kwargs_by_task',{}).get(task,old['request_mm_processor_kwargs'])
        require(actual==expected,'Effective official processor setting changed')


def resume(profile_path, output_path):
    """Resume a clean prefix; preserve a zero-call pre-load pause explicitly."""
    p,terminal,raw=study.closed(profile_path)
    if terminal['status']!='PAUSED_BEFORE_LOAD':
        return study.resume(profile_path,output_path)
    require(not raw and not terminal['completed_ids'] and read(p['control_file'])['paused'],
            'Zero-call resume requires an explicitly paused empty closed attempt')
    qualified_request_check(p)
    output_path=Path(output_path);q=copy.deepcopy(p)
    for key in ('implementation','observer_entry','worker_token','media_files','operational_wrapper','continuation_parent','continuation_wait'):
        q.pop(key,None)
    # All original IDs remain never claimed. This is a fresh initial profile,
    # with an explicit audit link, not a successful or uncertain request retry.
    worker=output_path.stem.removesuffix('-profile')
    control_path=output_path.with_name(worker+'-control.json')
    write_new(control_path,{'generation':0,'paused':True,'status':'ZERO_CALL_RESUME_PREPARED'})
    q.update(worker=worker,output=str(Path(p['output']).parent/worker),control_file=str(control_path),
        control_generation=0,pause_file=str(output_path.with_suffix('.PAUSE.json')),
        zero_call_predecessor=descriptor(profile_path),
        zero_call_predecessor_wait=descriptor(Path(p['output']).with_suffix('.wait.json')))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(q,f);draft=f.name
    try:answer=study.seal(draft,output_path)
    finally:Path(draft).unlink()
    return {**answer,'status':'SEALED_PAUSED_NOT_LAUNCHED','preserved_zero_call_predecessor':descriptor(profile_path)}


def prepare(plan_path, shard):
    """Prepare a finite source-revision shard without changing a running worker."""
    plan_path=Path(plan_path);plan=read(plan_path);spec=plan['shards'][shard]
    require(spec['run_new_calls'] is True,'This shard only retains previously valid answers')
    root=Path(plan['workspace_root']);plans=plan_path.parent
    ids=spec['assigned_ids'];ready=read(spec['ready_catalog_path'])
    require(set(ids)<=set(ready['items']),'Source shard not ready')
    pixel=read(spec['pixel_audit_path'])
    require(pixel['status']=='PASS_INDEPENDENT_PIXELS_COMPLETE_CARD'
            and pixel['card']==spec['card'] and set(ids)<=set(pixel['sample_ids'])
            and pixel['source_catalog_sha256']==file_sha256(spec['ready_catalog_path']),
            'Independent decoded-pixel source measurement missing or changed')
    measured={r['sample_id']:r for r in pixel['measurements']}
    items={s:ready['items'][s] for s in ids};bindings=[]
    for sid,item in items.items():
        qa=read(item['source_qa_path'])
        require(qa['status']=='PASS_ALL180_RGB8_PTS_OPENCV' and qa['sample_id']==sid
                and len(qa['decoded_RGB8_frame_hashes'])==180
                and qa['stream_sha256']==qa['opencv_stream_sha256']==item['ready']['stream_sha256']
                and qa['video_sha256']==item['ready']['video_sha256'],'CPU source identity differs')
        require(qa['row']['card']==spec['card'] and qa['row']['domain']==spec['domain'],'Card/domain changed')
        require(measured[sid]['status']=='PASS_INDEPENDENT_PIXEL_AUDIT'
                and measured[sid]['video_sha256']==item['ready']['video_sha256']
                and measured[sid]['stream_sha256']==item['ready']['stream_sha256'],
                'Independent pixels are not bound to this exact input')
        bindings.append(descriptor(item['source_qa_path']))
    old=verified(spec['previous_classic_profile'])
    _,_,raw=study.closed(spec['previous_classic_profile']['path'])
    require(not(set(ids)&set(raw)),'An already answered valid ID would be repeated')
    require(not any((Path(old['claim_root'])/(sid+'.json')).exists() for sid in ids),'Existing original claim cannot be reassigned')
    source_path=plans/(shard+'-sources.json')
    write_new(source_path,{'schema':'immutable-T45-revised-shard-sources-v1','items':items})
    acceptance_path=plans/(shard+'-source-acceptance.json')
    write_new(acceptance_path,{'status':'PASS_FROZEN_SHARD_SOURCES','assigned_ids':ids,
        'catalog_sha256':file_sha256(source_path),'source_QA':bindings,'source_frames':len(ids)*180,
        'new_model_calls':0,'source_revision_plan':descriptor(plan_path),
        'independent_pixel_audit':descriptor(spec['pixel_audit_path'])})
    parent_path=root/'plans/cosmos-x-full-20261028'/(spec['card']+'-profile.json')
    p=read(parent_path);runtime_path=Path(p['output'])/'runtime.json'
    for key in ('implementation','observer_entry','worker_token','media_files','sources','parent_validated_profile','scope_change_only','cold'):
        p.pop(key,None)
    p['engine_kwargs']['limit_mm_per_prompt']={'audio':0,'image':1,'video':1}
    worker=shard+'-segment0';control_path=plans/(shard+'-control.json')
    write_new(control_path,{'generation':0,'paused':True,'status':'PREPARED_NOT_LAUNCHED'})
    p.update(worker=worker,shard=shard,card=spec['card'],phase=spec['domain'],assigned_ids=ids,
        maximum_model_calls=len(ids),qualified_profile=descriptor(parent_path),qualified_runtime=descriptor(runtime_path),
        execution_plan=descriptor(plan_path),source_catalog=descriptor(source_path),source_acceptance=descriptor(acceptance_path),
        output=str(root/'runs/reproducibility'/study.NAME/worker),claim_root=str(root/'runs/reproducibility'/study.NAME/'claims'/shard),
        pause_file=str(plans/(worker+'-PAUSE.json')),control_file=str(control_path),control_generation=0,
        protocol_name='Cosmos-X-T45-20261030-source-revision',choices_by_task={'T4':['LEFT','RIGHT'],'T5':['YES','NO']},
        automatic_retry=False,formal_auto_resume=False)
    qualified_request_check(p)
    draft=plans/(worker+'-draft.json');write_new(draft,p)
    return study.seal(draft,plans/(worker+'-profile.json'))


def execute(plan_path):
    plan=read(plan_path);p=verified(plan['successor']);parent=verified(plan['predecessor'])
    require(plan['automatic_retry'] is False and p['card']==parent['card']
            and p['physical_lock']==parent['physical_lock']
            and p['phase']=='natural' and parent['phase']=='classic', 'Wrong queue handoff')
    state_path=Path(plan['state_path']);lock_path=Path(plan['controller_lock'])
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state={'status':'WAITING_PREDECESSOR','owner':owner_identity(),'queue_plan':descriptor(plan_path)}
        write_new(state_path,state)
        try:
            deadline=time.monotonic()+plan['maximum_wait_seconds']
            wait_path=Path(parent['output']).with_suffix('.wait.json')
            while not wait_path.exists():
                control=read(p['control_file'])
                if control['paused'] or control['generation']!=p['control_generation']:
                    state['status']='PAUSED_BEFORE_SUCCESSOR';return state
                require(time.monotonic()<deadline,'Predecessor deadline exceeded; no model launched')
                time.sleep(30)
            _,terminal,_=study.closed(plan['predecessor']['path'])
            require(terminal['status']=='COMPLETE','Predecessor did not complete; no successor or retry')
            control=read(p['control_file'])
            if control['paused'] or control['generation']!=p['control_generation']:
                state['status']='PAUSED_BEFORE_SUCCESSOR';return state
            state['accepted_predecessor_wait']=descriptor(wait_path)
            qualified_request_check(p)
            study.activate(plan['successor']['path'])
            env=dict(os.environ)
            for key in p.get('cleared_environment',[]):env.pop(key,None)
            env.update(p['environment'])
            state['status']='SUCCESSOR_SUPERVISOR_RUNNING';atomic_json(state_path,state)
            with Path(p['output']).with_suffix('.supervisor.log').open('xb') as log:
                child=subprocess.Popen([p['python'],'-B','-m','classic_precision.cosmos_t45','supervise',
                    '--profile',plan['successor']['path']],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                state['supervisor_pid']=child.pid;atomic_json(state_path,state)
                rc=child.wait()
            state.update(real_supervisor_wait=True,supervisor_returncode=rc)
            require(rc==0,'Successor failed; no retry')
            _,terminal,_=study.closed(plan['successor']['path'])
            state['status']='COMPLETE' if terminal['status']=='COMPLETE' else 'PAUSED'
            return state
        except BaseException as error:
            state.update(status='HOLD_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
        finally:
            atomic_json(state_path,state)


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--plan');a.add_argument('--resume-profile');a.add_argument('--output');a.add_argument('--prepare-shard');p=a.parse_args()
    require(bool(p.plan)!=bool(p.resume_profile),'Choose queue/prepare plan or explicit clean resume')
    result=prepare(p.plan,p.prepare_shard) if p.prepare_shard else execute(p.plan) if p.plan else resume(p.resume_profile,p.output)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
