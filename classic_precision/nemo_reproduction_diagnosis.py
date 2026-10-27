"""Bounded old-entry/current-entry Nemo diagnosis; never replaces science raw.

Only observational hooks are added. No model forward, processor, sampling or
Torch numerical policy is replaced. Original worker closure/lease checks apply.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .refinement_worker import read, require, write_new, verified, physical_lease, Boundary
from .storage import atomic_json, _append, digest, file_sha256, owner_identity


def fingerprint(tensor):
    import torch
    value=tensor.detach().contiguous().cpu()
    return {'shape':list(value.shape),'dtype':str(value.dtype),
        'sha256':hashlib.sha256(memoryview(value.view(torch.uint8).numpy())).hexdigest()}


def install_probe(worker):
    model=worker.get_model();target=model.vision_model
    worker._nemo_diagnostic_receipts={'inputs':[],'outputs':[]}
    def before(module,args,kwargs):
        worker._nemo_diagnostic_receipts['inputs'].append(fingerprint(args[0]))
    def after(module,args,kwargs,result):
        import torch
        def collect(value):
            if isinstance(value,torch.Tensor):return fingerprint(value)
            if isinstance(value,(list,tuple)):return [collect(v)for v in value[:4]]
            if isinstance(value,dict):return {k:collect(v)for k,v in value.items()}
            return str(type(value))
        worker._nemo_diagnostic_receipts['outputs'].append(collect(result))
    worker._nemo_diagnostic_handles=[target.register_forward_pre_hook(before,with_kwargs=True),
        target.register_forward_hook(after,with_kwargs=True)]
    return True


def pop_probe(worker):
    result=worker._nemo_diagnostic_receipts
    worker._nemo_diagnostic_receipts={'inputs':[],'outputs':[]}
    return result


def run(path):
    from . import inference,gemma_speed as g
    from .reader_route_smoke import decode,metadata,prompt,file_loader_settings,Events,generation_receipt
    from .refinement_dispatch import compiler_preflight
    p=read(path);out=Path(p['output'])
    require(p['maximum_model_calls']==len(p['cases'])==len(p['assigned_ids'])==6,'Only six calls per cold')
    require(not out.exists() and socket.gethostname()==p['hostname'],'No resume/wrong host')
    require(file_sha256(__file__)==p['implementation']['sha256'],'Diagnosis source changed')
    require(not any(Path(p['claim_root']).glob('*.json')),'An existing diagnostic intent cannot be retried')
    for k,v in p['environment'].items():require(os.environ.get(k)==v,'Environment differs: '+k)
    for k in p['cleared_environment']:require(k not in os.environ,'Environment must be cleared: '+k)
    require(inference.checkpoint_asset_identity(p['checkpoint'])==p['checkpoint_assets'],'Checkpoint differs')
    compiler_preflight(p['environment'])
    import torch
    from runtime_compat.classic_precision_config_only import _worker_runtime,_torch_flags
    catalog=verified(p['source_catalog']);out.mkdir(parents=True);sink=Events(out)
    atomic_json(out/'identity.json',{'profile':p,'profile_sha256':digest(p),'owner':owner_identity()})
    done=[];pending=None;session=None;owned=[];boundary=Boundary(p)
    with physical_lease(p,torch) as lease,ExitStack() as stack:
        atomic_json(out/'lease.json',lease)
        if p['file_media']:stack.enter_context(file_loader_settings(Path(path).parents[2]))
        try:
            start=time.monotonic();session=inference.ModelSession('nemo',p['checkpoint'],sink,resources=p['resources'])
            engine=session.engine;owned=engine.collective_rpc(g._worker_identity)
            atomic_json(out/'engine-owner.json',{'owned':owned})
            atomic_json(out/'load.json',{'seconds':time.monotonic()-start,'generation':generation_receipt(session.sampling),
                'parent_flags':_torch_flags(torch),'worker_runtime':engine.collective_rpc(_worker_runtime),
                'effective_environment':{k:os.environ.get(k) for k in sorted(set(p['environment'])|set(p['cleared_environment']))}})
            require(all(engine.collective_rpc(install_probe)),'Vision observer not installed')
            original_generate=engine.generate;capture={}
            def observe_request(request,*args,**kwargs):
                frames,meta=request['multi_modal_data']['video'][0]
                capture.update(submitted_FP32_sha256=hashlib.sha256(memoryview(frames)).hexdigest(),
                    submitted_shape=list(frames.shape),submitted_dtype=str(frames.dtype),metadata=meta,
                    chat_text=request['prompt'])
                answer=original_generate(request,*args,**kwargs)
                capture['actual_prompt_token_ids']=list(answer[0].prompt_token_ids)
                return answer
            engine.generate=observe_request
            for case in p['cases']:
                if boundary.check():break
                key=case['case_id'];sid=case['sample_id'];pending=key;item=catalog['items'][sid]
                write_new(Path(p['claim_root'])/(key+'.json'),{'case_id':key,'sample_id':sid,'profile_sha256':digest(p)})
                capture.clear();began=time.monotonic()
                engine.collective_rpc(pop_probe)
                if case['route']=='OLD':
                    ready=item['ready'];source=Path(p['sources'][sid]['path']);asset=source
                    for _ in Path(ready['video_relative']).parts:asset=asset.parent
                    session.context['deep']=False
                    def commit(raw):
                        record={**raw,'case_id':key,'route':'OLD','attempt_id':key,'profile_sha256':digest(p)}
                        _append(out/'raw.jsonl',record);return record
                    session.generate(item['row'],ready,asset,commit)
                else:
                    session.context.update(active=False,deep=False)
                    frames,receipt,_=decode(p['sources'][sid]['path'],'A')
                    answer=engine.generate({'prompt':prompt(session,item),'multi_modal_data':{'video':[(frames,metadata())]}},
                        session.sampling,use_tqdm=False)[0];choice=answer.outputs[0]
                    raw={'case_id':key,'sample_id':sid,'route':'SMOKE_A','raw_answer':choice.text,
                        'output_token_ids':list(choice.token_ids),'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason,
                        'truth':item['row']['truth'],'profile_sha256':digest(p),'reader':receipt}
                    _append(out/'raw.jsonl',raw)
                    require(choice.finish_reason=='stop' and choice.text.strip()in('LEFT','RIGHT'),'Unexpected completion retained')
                gpu=engine.collective_rpc(pop_probe)
                require(gpu and sum(v['shape'][0]for v in gpu[0]['inputs'])==180,'GPU did not receive180 frames')
                record={'event':'DIAGNOSTIC_PASS','case_id':key,'sample_id':sid,'route':case['route'],
                    'request':dict(capture),'GPU_vision':gpu,'seconds':time.monotonic()-began}
                _append(out/'events.jsonl',record);done.append(key);pending=None
                atomic_json(out/'progress.json',{'status':'RUNNING','completed':len(done),'expected':6,'last':key})
        except BaseException as e:
            atomic_json(out/'failure.json',{'error':str(e),'UNCERTAIN':pending,'automatic_retry':False});raise
        finally:
            if session is not None:
                g.shutdown_owned_engine(session.engine,owned,out);session.engine=None;session.close()
    result={'status':'COMPLETE'if done==p['assigned_ids']else'PAUSED','completed_ids':done,'automatic_retry':False}
    atomic_json(out/'result.json',result);return result


def supervise(path):
    import classic_precision.refinement_worker as w
    original=w.subprocess.Popen
    def popen(argv,**kwargs):
        return original([s.replace('classic_precision.refinement_worker','classic_precision.nemo_reproduction_diagnosis')for s in argv],**kwargs)
    w.subprocess.Popen=popen
    try:return w.supervise(path)
    finally:w.subprocess.Popen=original


def queue(path):
    p=read(path);require(p['status']=='SEALED','No automatic queue replay')
    p['status']='RUNNING';atomic_json(path,p);reports=[]
    try:
        for profile in p['profiles']:
            child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.nemo_reproduction_diagnosis','supervise','--profile',profile])
            code=child.wait();out=Path(read(profile)['output']);w=read(out.with_suffix('.wait.json'));reports.append(w)
            require(code==w['returncode']==0 and w['real_wait_observed']and w['owned_engines_exited']
                and w['original_physical_lease_released']and w['terminal_status']=='COMPLETE','Diagnostic not safely closed; no retry')
        p.update(status='COMPLETE',reports=reports);atomic_json(path,p)
    except BaseException as e:
        p.update(status='HELD_NO_RETRY',reports=reports,error=str(e));atomic_json(path,p);raise
    return {'status':'COMPLETE','cold_loads':len(reports),'calls':sum(len(w['completed_ids'])for w in reports)}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','supervise','queue']);p.add_argument('--profile');p.add_argument('--queue')
    a=p.parse_args();result=queue(a.queue)if a.command=='queue'else supervise(a.profile)if a.command=='supervise'else run(a.profile)
    print(__import__('json').dumps(result,sort_keys=True))


if __name__=='__main__':main()
