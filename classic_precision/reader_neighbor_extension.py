"""User-authorized T2/T3 neighbors for the remaining model/route pairs.

Uses the unchanged finite reader worker, qualified model profiles, and Nemo's
already-tested public CUTLASS context. Original held runs are never resumed.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ctypes
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import time

from .refinement_worker import read,require,write_new,descriptor,verified
from .storage import digest,file_sha256,atomic_json

NAME='reader-neighbors-extension-20261027'
PREVIOUS='cosmos-t3-neighbors-20261027'
SCIENCE=('checkpoint','checkpoint_assets','runtime','runtime_identity','resources','environment',
    'cleared_environment','qualified_profile','qualified_anchor','engine_settings','kernel','runtime_sources',
    'checkpoint_contract','cpu_execution_overlay','entry_implementation','qualified_host_profile')


def seal(root,host):
    from . import reader_route_smoke as worker
    root=Path(root);plan=root/'plans'/NAME;source=read(plan/'sources.json');alloc=read(plan/'allocation.json')
    experiment=read(plan/'plan.json')
    require(experiment['maximum_new_scientific_calls']==960 and alloc['total_B_answers']==1152,'Quota changed')
    for name,sha in experiment['files'].items():require(file_sha256(plan/name)==sha,'Sealed input changed')
    cards=['fourier0','fourier1'] if host=='fourier' else ['radon0']
    base_dir=root/'plans/reader-route-smoke-20261026'
    anchor=verified(read(root/'plans/classic-refinement'/('gemma-'+cards[0]+'-profile.json'))['qualified_anchor'])
    roots=anchor['source_roots'];files={}
    for sid,item in source['items'].items():
        path=Path(roots[item['source_family']])/item['ready']['video_relative']
        require(file_sha256(path)==item['ready']['video_sha256'],'Source changed')
        files[sid]={'path':str(path),'stat':worker.state(path)}
    queues=[]
    for card in cards:
        predecessor=root/'plans'/PREVIOUS/(card+'-queue.json');prior=read(predecessor)
        pps=[descriptor(p) for p in prior['profiles']]
        inherited_card=read(prior['profiles'][-1]);profiles=[]
        for segment in alloc['cards'][card]:
            model=segment['model']
            for arm in ['C']:
                suffix='-restore1' if model=='nemo' else ''
                original=base_dir/(model+'-'+card+'-'+arm+suffix+'-profile.json')
                if model=='qwen' and arm=='C':original=base_dir/'qwen-fourier1-C-repair1-profile.json'
                if model=='cosmos' and arm=='C':original=base_dir/'cosmos-fourier1-C-repair1-profile.json'
                base=read(original);p={k:deepcopy(base[k]) for k in SCIENCE if k in base}
                ids=segment['sample_ids'];name=model+'-'+card+'-'+arm+'-neighbors2';out=root/'runs/reproducibility'/NAME/name
                p.update(schema='reader-neighbors-extension-worker-v1',model=model,arm=arm,user_route='A' if arm=='A' else 'B',
                    worker=name,hostname=socket.gethostname(),worker_token=digest({'allocation':descriptor(plan/'allocation.json'),'worker':name}),
                    assigned_ids=ids,maximum_model_calls=len(ids),automatic_retry=False,formal_auto_resume=False,
                    source_catalog=descriptor(plan/'sources.json'),allocation=descriptor(plan/'allocation.json'),
                    experiment=descriptor(plan/'plan.json'),sources={s:files[s] for s in ids},
                    gpu_uuid=inherited_card['gpu_uuid'],physical_lock=inherited_card['physical_lock'],
                    output=str(out),claim_root=str(out.parent/'claims'/model/arm),pause_file=str(plan/(name+'-PAUSE.json')),
                    implementation=descriptor(worker.__file__),qualified_neighbor_parent=descriptor(original),
                    neighbor_entry=descriptor(__file__),route_label=('controlled RGB16 to FP32; original downstream' if arm=='A'
                    else 'official uint8/native no resize; Gemma shared bottom16 spatial/input adapter retained' if model=='gemma'
                    else 'official uint8/native no resize, no secondary sampling'))
                require(p['environment']['CUDA_VISIBLE_DEVICES']==card[-1],'Scientific parent bound to wrong card')
                if model=='nemo':
                    p['legacy_kernel_context']={name:descriptor(Path(__file__).with_name(name)) for name in
                        ['nemo_canonical_restore.py','nemo_kernel_replay.py']}
                path=plan/(name+'-profile.json');write_new(path,p);profiles.append(str(path))
        q=plan/(card+'-queue.json');write_new(q,{'status':'SEALED','profiles':profiles,
            'predecessor_queue':str(predecessor),'predecessor_profiles':pps,'predecessor_card':card,
            'automatic_retry':False,'formal_auto_resume':False});queues.append(str(q))
    return {'status':'SEALED','queues':queues,'model_calls':0}


def validate(p):
    parent=verified(p['qualified_neighbor_parent']);exp=verified(p['experiment']);allocation=verified(p['allocation'])
    require(exp['maximum_new_scientific_calls']==960 and exp['historical_A_reuse'],'Unapproved scope')
    require({k:p[k] for k in SCIENCE if k in p}=={k:parent[k] for k in SCIENCE if k in parent},'Qualified science/resource/environment drift')
    card=next(c for c,segs in allocation['cards'].items() if any(s['model']==p['model'] for s in segs))
    segment=next(s for s in allocation['cards'][card] if s['model']==p['model'])
    require(p['assigned_ids']==segment['sample_ids'] and p['maximum_model_calls']==len(segment['sample_ids']), 'Unsealed sample IDs')
    require(p['arm']=='C' and p['user_route']=='B' and segment['arms']==['C'] and p['neighbor_entry']['sha256']==file_sha256(__file__),'Wrong route or entry changed')
    require(p['output']!=parent['output'] and p['claim_root']!=parent['claim_root'],'Old journals reused')
    if p['model']=='nemo':
        for ref in p['legacy_kernel_context'].values():require(file_sha256(ref['path'])==ref['sha256'],'Legacy kernel context changed')
    return {'status':'PASS','model':p['model'],'arm':p['arm'],'maximum_model_calls':p['maximum_model_calls'],'model_calls':0}


def cpu(path):
    p=read(path);validate(p)
    if p['model']=='gemma':
        for k in p.get('cleared_environment',[]):os.environ.pop(k,None)
        os.environ.update(p['environment']);os.environ['CUDA_VISIBLE_DEVICES']=''
        from .reader_route_smoke import cpu as check
        return check(path)
    # Pixel values changed, but geometry/reader/processor contracts did not.
    # Deep CPU processor evidence is reused from qualified parent; first real
    # GPU observation per task/arm remains inside this new finite quota.
    return {**validate(p),'status':'QUALIFIED_CONTRACT_REUSED','model_calls':0,'profile':str(path)}


def run(path):
    p=read(path);validate(p)
    from . import reader_route_smoke
    if p['model']=='nemo':
        from .nemo_canonical_restore import legacy_session
        with legacy_session(p['output']):return reader_route_smoke.run(path)
    return reader_route_smoke.run(path)


def supervise(path):
    from . import refinement_worker as worker
    original=worker.subprocess.Popen
    def popen(argv,**kwargs):
        return original([s.replace('classic_precision.refinement_worker','classic_precision.reader_neighbor_extension') for s in argv],**kwargs)
    worker.subprocess.Popen=popen
    try:return worker.supervise(path)
    finally:worker.subprocess.Popen=original


def wait_predecessor(q,path):
    """Wait on filesystem completion events, not repeated GPU/health polling."""
    watched=Path(q['predecessor_queue']);lib=ctypes.CDLL(None,use_errno=True)
    fd=lib.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);require(fd>=0,'Cannot watch predecessor')
    require(lib.inotify_add_watch(fd,os.fsencode(watched.parent),0x8|0x80)>=0,'Cannot watch predecessor directory')
    deadline=time.monotonic()+4*3600
    try:
        while True:
            prior=read(watched)
            require(prior['status'] in ['SEALED','RUNNING','COMPLETE'],'Predecessor held; do not dispatch')
            if prior['status']=='COMPLETE':break
            for pp in q['profiles']:require(not Path(read(pp)['pause_file']).exists(),'Queued successor paused')
            ready,_,_=select.select([fd],[],[],min(60,max(0,deadline-time.monotonic())))
            require(time.monotonic()<deadline,'Predecessor wait deadline; no retry')
            if ready:
                try:os.read(fd,65536)
                except BlockingIOError:pass
        require(prior['profiles']==[d['path'] for d in q['predecessor_profiles']],'Predecessor queue changed')
        proofs=[]
        for ref in q['predecessor_profiles']:
            p=verified(ref);w=read(Path(p['output']).with_suffix('.wait.json'))
            require(w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited'] and
                w['original_physical_lease_released'] and w['terminal_status']=='COMPLETE' and
                w['profile_sha256']==digest(p) and w['completed_ids']==p['assigned_ids'],'Predecessor closure incomplete')
            proofs.append(w)
        last=verified(q['predecessor_profiles'][-1]);new=read(q['profiles'][0])
        require(last['physical_lock']==new['physical_lock'] and last['gpu_uuid']==new['gpu_uuid'],'Cross-card handoff')
        write_new(Path(path).with_name(q['predecessor_card']+'-predecessor-accepted.json'),{'status':'PASS','proofs':proofs})
    finally:os.close(fd)


def queue(path):
    q=read(path);require(q['status']=='SEALED','No automatic queue replay')
    for pp in q['profiles']:
        p=read(pp);validate(p)
        if p['model']=='gemma':require(read(Path(pp).with_suffix('.cpu.json'))['profile_sha256']==digest(p),'Gemma CPU binding missing')
    q.update(status='WAITING_PREDECESSOR');atomic_json(path,q);reports=[]
    try:
        wait_predecessor(q,path);q['status']='RUNNING';atomic_json(path,q)
        for pp in q['profiles']:
            p=read(pp);require(not Path(p['pause_file']).exists(),'Queued profile paused')
            child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.reader_neighbor_extension','supervise','--profile',pp])
            code=child.wait();w=read(Path(p['output']).with_suffix('.wait.json'));reports.append(w)
            require(code==0 and w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited'] and
                w['original_physical_lease_released'] and w['terminal_status']=='COMPLETE' and w['completed_ids']==p['assigned_ids'],
                'Worker partial/failure; hold successor without retry')
        q.update(status='COMPLETE',reports=reports);atomic_json(path,q)
    except BaseException as e:
        q.update(status='HELD_NO_RETRY',reports=reports,error=str(e));atomic_json(path,q);raise
    return {'status':q['status'],'closed_workers':len(reports)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['seal','validate','cpu','run','supervise','queue'])
    p.add_argument('--root');p.add_argument('--host');p.add_argument('--profile');p.add_argument('--queue');a=p.parse_args()
    value=(seal(a.root,a.host) if a.command=='seal' else queue(a.queue) if a.command=='queue'
        else validate(read(a.profile)) if a.command=='validate' else globals()[a.command](a.profile))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
