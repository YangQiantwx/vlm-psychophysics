"""Nemo X: frozen native runner, runtime admission, finite queues and safe continuation.

No processor/forward/weight patch. This wrapper adds observer assertions and
request-boundary control only. A continuation never reissues an existing claim.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from . import official_x_runner as base
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity, _append

NAME = 'nemo-x-full-20261029'
ORIGINAL_IDENTITY = base.worker_identity
ORIGINAL_BOUNDARY = base.Boundary
EXPECTED_KERNELS = {
 'kernel:vllm.model_executor.kernels.linear.nvfp4.flashinfer.FlashInferCutlassNvFp4LinearKernel':80,
 'kernel:vllm.model_executor.kernels.linear.scaled_mm.flashinfer.FlashInferFP8ScaledMMLinearKernel':97,
 'moe_kernel:vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel':40,
}


def validate_runtime(value):
    require(value['model_class'].endswith('.NemotronH_Nano_VL_V2'), 'Different Nemo model implementation')
    require(all(value['method_kernel_counts'].get(k)==v for k,v in EXPECTED_KERNELS.items()), 'Actual Nemo kernels differ')
    require(value['actual_backends']=={'nvfp4_backend:NvFp4MoeBackend.FLASHINFER_CUTLASS':40}, 'Actual MoE differs')
    config=value['kernel_config']
    require('enable_flashinfer_autotune=False' in config and "moe_backend='auto'" in config and "linear_backend='auto'" in config, 'Official kernel settings differ')
    require(value['mm_device_do_normalize'] is False, 'Native normalization path differs')
    return value


def worker_identity(worker, model):
    require(model=='nemo','This entry is only for Nemo')
    return validate_runtime(ORIGINAL_IDENTITY(worker,model))


def science(p):
    require(p['model']=='nemo' and not p.get('bootstrap') and not p.get('request_adapter'), 'No custom model/input patches allowed')
    e=p['engine_kwargs'];v=e['hf_overrides']['vision_config']
    require(e['kernel_config']=={'enable_flashinfer_autotune':False}, 'Wrong official kernel configuration')
    require(v['video_target_num_patches']==1536 and v['video_maintain_aspect_ratio'] is True, 'Native no-resize geometry differs')
    require(e['max_model_len']==65536 and e['kv_cache_memory_bytes']==2*1024**3 and e['max_num_batched_tokens']==4096, 'Validated resource configuration differs')
    require(e['enforce_eager'] and not e['enable_prefix_caching'] and e['mm_processor_cache_gb']==0 and e['video_pruning_rate']==0 and e['max_num_seqs']==1, 'Validated inference policy differs')
    require(p['vision_observation']['rule']=='nchw_frames' and not p.get('request_mm_processor_kwargs'), 'Native Nemo request differs')
    require(p.get('media_io_kwargs',base.MEDIA)==base.MEDIA, 'Official all-frame reader differs')


class Boundary(ORIGINAL_BOUNDARY):
    def check(self):
        control=read(self.profile['control_file'])
        if control['paused'] or control['generation']!=self.profile['control_generation']:
            self.requested=True
        return super().check()


def seal(draft, output):
    p=read(draft);science(p);p['operational_wrapper']=descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(p,f);name=f.name
    try:return base.seal(name,output)
    finally:Path(name).unlink()


def run(profile):
    p=read(profile);science(p)
    require(p['operational_wrapper']['sha256']==file_sha256(__file__),'Operational wrapper changed')
    if p['phase']=='full':
        gate=verified(p['qualification_gate'])
        require(gate['status']=='PASS_ALL_THREE_CARDS' and gate['calls']==72,'Qualification incomplete')
    base.worker_identity=worker_identity;base.Boundary=Boundary
    return base.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    native=worker.subprocess.Popen
    def popen(argv,**kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv],**kwargs)
    worker.subprocess.Popen=popen
    try:return worker.supervise(profile)
    finally:worker.subprocess.Popen=native


def rows(path):
    if not Path(path).exists():return
    with Path(path).open() as f:
        for line in f:
            if line.strip():yield json.loads(line)


def closed(profile_path, allowed=('COMPLETE','PAUSED','PAUSED_BEFORE_LOAD')):
    p=read(profile_path);out=Path(p['output']);w=read(out.with_suffix('.wait.json'));r=read(out/'result.json')
    require(w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited'] and w['original_physical_lease_released'], 'No safe worker closure')
    require(w['profile_sha256']==digest(p) and r['status'] in allowed and not r.get('UNCERTAIN_ids') and not (out/'failure.json').exists(), 'Failure or uncertain request; no automatic retry')
    passed={};raw={}
    for x in rows(out/'events.jsonl'):
        if x['event']=='TECHNICAL_PASS':
            require(x['sample_id'] not in passed,'Duplicate PASS');passed[x['sample_id']]=x['raw_digest']
    for x in rows(out/'raw.jsonl'):
        sid=x['sample_id'];require(sid not in raw and sid in p['assigned_ids'],'Duplicate or unassigned raw')
        require(x['profile_sha256']==digest(p) and passed.get(sid)==digest(x),'Unqualified raw retained; cannot continue')
        claim=read(Path(p['claim_root'])/(sid+'.json'))
        require(claim['profile_sha256']==digest(p) and claim['worker_token']==p['worker_token'],'Claim identity differs')
        raw[sid]={'sample_id':sid,'raw_answer':x['raw_answer'],'output_token_ids':x['output_token_ids'],'finish_reason':x['finish_reason'],'stop_reason':x['stop_reason'],'actual_prompt_token_ids':x['actual_prompt_token_ids'],'source_media_sha256':[q['sha256'] for q in x['source']['media']]}
    require(list(raw)==r['completed_ids'] and set(raw)==set(passed),'Raw/PASS/result disagreement')
    claims={s for s in p['assigned_ids'] if (Path(p['claim_root'])/(s+'.json')).exists()}
    require(claims==set(raw),'Claim without qualified answer; refuse retry')
    if r['status']=='COMPLETE':require(r['completed_ids']==p['assigned_ids'],'Incomplete COMPLETE')
    return p,r,raw


@contextmanager
def controller_lock(path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);yield


def queue(plan_path):
    plan=read(plan_path);require(plan['status']=='SEALED' and plan['automatic_retry'] is False,'Wrong finite queue')
    state_path=Path(plan['state_path']);require(not state_path.exists(),'Queue already attempted')
    state={'status':'RUNNING','owner':owner_identity(),'plan':descriptor(plan_path),'workers':[]}
    with controller_lock(plan['controller_lock']):
        atomic_json(state_path,state)
        try:
            for ref in plan['profiles']:
                p=verified(ref);control=read(p['control_file'])
                if control['paused'] or control['generation']!=p['control_generation']:
                    state['status']='PAUSED_BEFORE_SUCCESSOR';break
                state['current_profile']=ref;atomic_json(state_path,state)
                env=dict(os.environ)
                for k in p.get('cleared_environment',[]):env.pop(k,None)
                env.update(p['environment'])
                child=subprocess.Popen([p['python'],'-B','-m',__spec__.name,'supervise','--profile',ref['path']],env=env,stdin=subprocess.DEVNULL)
                code=child.wait()
                state['workers'].append({'profile':ref,'supervisor_pid':child.pid,'real_supervisor_wait':True,'returncode':code})
                require(code==0,'Supervisor failure; no retry or successor')
                _,r,_=closed(ref['path']);state['workers'][-1]['terminal_status']=r['status'];atomic_json(state_path,state)
                if r['status']!='COMPLETE':state['status']='PAUSED';break
            else:state['status']='COMPLETE'
        except BaseException as e:
            state.update(status='HELD_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
        finally:atomic_json(state_path,state)
    return {'status':state['status'],'workers':len(state['workers'])}


def pause(control_path):
    p=Path(control_path);c=read(p)
    c['paused']=True;c['pause_requested_by']='human-or-authorized-controller';c['pause_request_owner']=owner_identity()
    _append(p.with_suffix('.events.jsonl'),{'event':'PAUSE_REQUESTED','generation':c['generation'],'owner':owner_identity()})
    atomic_json(p,c)
    return {'status':'PAUSE_REQUESTED','control':str(p),'drain_current_request':True,'kill':False}


def remaining(p,result,claimed):
    require(result['status'] in ('PAUSED','PAUSED_BEFORE_LOAD') and not result.get('UNCERTAIN_ids'),'Only a clean pause can resume')
    done=result['completed_ids'];require(len(done)==len(set(done)) and set(done)==set(claimed),'Only qualified complete claims may be retained')
    require(set(done)<=set(p['assigned_ids']),'Foreign completion')
    return [s for s in p['assigned_ids'] if s not in set(done)]


def prepare_resume(profile_path,output_path):
    p,result,raw=closed(profile_path)
    require(p['phase']=='full','Qualification is never automatically repeated')
    control=read(p['control_file']);require(control['paused'],'Explicit pause required before continuation')
    ids=remaining(p,result,raw);require(ids,'Nothing remains')
    q=copy.deepcopy(p);generation=control['generation']+1;worker=p['card']+'-full-segment'+str(generation)
    for key in ('implementation','worker_token','media_files','operational_wrapper'):q.pop(key,None)
    q.update(worker=worker,assigned_ids=ids,maximum_model_calls=len(ids),control_generation=generation,
        output=str(Path(p['output']).parent/worker),pause_file=str(Path(output_path).with_suffix('.PAUSE.json')),
        continuation_parent=descriptor(profile_path),continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(q,f);draft=f.name
    try:answer=seal(draft,output_path)
    finally:Path(draft).unlink()
    _append(Path(p['control_file']).with_suffix('.events.jsonl'),{'event':'CONTINUATION_PREPARED','generation':generation,'parent':descriptor(profile_path),'next':descriptor(output_path),'remaining':len(ids)})
    # Activation is explicit and happens after the full old raw/claim/exit audit.
    control.update(generation=generation,paused=False,continuation_profile=descriptor(output_path));atomic_json(p['control_file'],control)
    return answer


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['seal','run','supervise','queue','pause','prepare-resume','audit-closed'])
    for name in ('draft','output','profile','plan','control'):a.add_argument('--'+name)
    q=a.parse_args()
    if q.command=='seal':v=seal(q.draft,q.output)
    elif q.command=='run':v=run(q.profile)
    elif q.command=='supervise':v=supervise(q.profile)
    elif q.command=='queue':v=queue(q.plan)
    elif q.command=='pause':v=pause(q.control)
    elif q.command=='prepare-resume':v=prepare_resume(q.profile,q.output)
    else:
        p,r,raw=closed(q.profile);v={'status':'CLOSED_PASS','worker_status':r['status'],'completed':len(raw)}
    print(json.dumps(v),flush=True)


if __name__=='__main__':main()
