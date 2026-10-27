"""Seal new source/index identities and finite qualified refinement allocations."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import json
import os
from pathlib import Path
import shutil
import socket
import statistics
import subprocess
import sys
import time
import uuid

from .manifest import digest, file_sha, state_key
from .refinement_worker import descriptor, read, require, verified, write_new

HOSTS = {'fourier': '/home/yang/cvpr2027_benchmark', 'radon': '/mnt/sda/q/cvpr2027_benchmark'}
GPUS = {'fourier0': 'GPU-c89c0c8c-8081-5ba9-c73f-d08af5875193',
        'fourier1': 'GPU-94d2f97e-b0db-db79-5423-467860a66e2f',
        'radon0': 'GPU-25d52640-5cbc-8521-f1f9-baaf52852f5d'}


def allocation(manifest):
    rows = manifest['rows']
    groups = defaultdict(list)
    for row in rows:
        p = dict(row['source_parameters'])
        for k in ('target_side', 'reference_side', 'tilt_sign'):
            p.pop(k, None)
        groups[digest(p)].append(row)
    gemma = {k: [] for k in GPUS}
    by_cell = defaultdict(list)
    for key, block in groups.items():
        task = block[0]['task']
        require(len(block) == (2 if task == 'T2' else 4), 'Incomplete minimum contrast unit')
        p = dict(block[0]['source_parameters'])
        for k in ('target_side', 'reference_side', 'tilt_sign', 'onset_frame', 'offset_frame', 'onset_index'):
            p.pop(k, None)
        # Duration remains a condition even when absolute event times are removed.
        p['duration_frames'] = block[0]['source_parameters']['offset_frame']-block[0]['source_parameters']['onset_frame']
        by_cell[digest(p)].append((key, block))
    for cell, blocks in sorted(by_cell.items()):
        require(len(blocks) == 24, 'Each cell must retain all 24 event onsets')
        blocks.sort(key=lambda x: digest(['classic-refinement-20261026', cell, x[0]]))
        owners = ['fourier0']*9+['fourier1']*9+['radon0']*6
        for (_, block), owner in zip(blocks, owners):
            gemma[owner].extend(r['sample_id'] for r in sorted(block, key=lambda r:r['sample_id']))
    ids = [r['sample_id'] for r in rows]
    lookup = {r['sample_id']: r for r in rows}
    def first_two(values):
        # One T2 and one T3 within the formal allocation. They are counted once.
        first = [next(s for s in values if lookup[s]['task'] == t) for t in ('T2','T3')]
        return first+[s for s in values if s not in first]
    assignments = {'qwen-fourier0': first_two(ids), 'cosmos-fourier1': first_two(ids),
                   'nemo-radon0': first_two(ids), **{'gemma-'+k:first_two(v) for k,v in gemma.items()}}
    flat = sum(gemma.values(), [])
    require(len(flat) == len(set(flat)) == 3168 and set(flat) == set(ids), 'Gemma allocation gap/overlap')
    return {'schema':'classic-refinement-allocation-v1', 'manifest_sha256':digest(manifest),
        'assignments':assignments, 'maximum_model_calls':12672, 'max_attempts_per_id_per_model':1,
        'automatic_retry':False, 'Gemma_onsets_per_cell':{'fourier0':9,'fourier1':9,'radon0':6},
        'rationale':'Score-independent SHA ordering of complete LR/sign onset units. Qwen then Gemma/F0, Gemma then Cosmos/F1, Nemo then Gemma/R0. The more specialized Gemma bridge is exercised immediately. Gemma allocation follows measured prior ordinary request costs about22/22/29s.'}


def sources(root):
    from .dataset import index_from_catalog, load_index, validate_index
    from .refinement import extend_index
    root=Path(root); run=root/'runs/classic-refinement-20261026';op=root/'runs/reproducibility/operator'
    qa=op/'classic-refinement-source-qa-20261026'
    summary=read(qa/'summary.json')
    require(summary['status']=='COMPLETE' and not summary['failed_IDs']
        and summary['completed_counts']=={'T2':2016,'T3':1152}
        and summary['decoded_frames']==570240, 'New source QA incomplete')
    manifest=read(run/'manifest.json')
    ready={r['sample_id']:r for r in map(json.loads,(run/'ready.jsonl').read_text().splitlines())}
    audits={r['sample_id']:r for r in map(json.loads,(qa/'per-id.jsonl').read_text().splitlines())}
    require(len(ready)==len(audits)==len(manifest['rows'])==3168,'Source IDs missing/duplicate')
    anchor=read(op/'gemma-nvfp4-formal-tail3-fourier0-profile-20261026.json')
    old=verified(anchor['catalog'])
    items={}
    for row in manifest['rows']:
        sid=row['sample_id'];r=ready[sid];a=audits[sid]
        require(a['status']=='PASS' and a['row_sha256']==digest(row) and a['ready_sha256']==digest(r)
            and a['video_sha256']==r['video_sha256'] and a['decoded_stream_sha256']==r['stream_sha256'], 'Source binding differs')
        require(file_sha(run/r['video_relative'])==r['video_sha256'],'Source changed after QA')
        key=digest({'task':row['task'],'exact_prompt':row['prompt']})
        require(key in old['templates'],'Prompt differs from qualified geometry controls')
        items[sid]={'sample_id':sid,'task':row['task'],'row':row,'ready':r,
            'row_sha256':digest(row),'ready_sha256':digest(r),'source_family':'classic-refinement',
            'source_stream_sha256':r['stream_sha256'],'source_audit_identity':a['identity'],'template_key':key}
    catalog={**{k:v for k,v in old.items() if k in ('schema','CPU_reference_value_sha256')},
        'counts':{'T2':2016,'T3':1152},'items':items,'templates':old['templates'],
        'new_source_audit':descriptor(qa/'summary.json'),
        'reference_policy':'Existing geometry/prompt templates; first two fresh CPU inputs bound to actual GPU within counted formal requests'}
    path=root/'plans/classic-refinement/execution-catalog.json';write_new(path,catalog)
    addition=index_from_catalog(catalog)
    names={r['sample_id']:r['video_relative'] for r in csv.DictReader((root/'plans/classic-refinement/video-names.csv').open())}
    for sid,item in addition['items'].items():item['video_relative']=names[sid]
    addition.pop('index_id');addition['index_id']=digest(addition);validate_index(addition)
    write_new(root/'plans/classic-refinement/new-index.json',addition)
    base=load_index(root/'datasets/current/dataset-index.json')
    full=extend_index(base,addition)
    dataset=root/'datasets/classic-t1-t3-v0.4';dataset.mkdir(exist_ok=False)
    for sid,e in full['items'].items():
        target=dataset/e['video_relative'];target.parent.mkdir(parents=True,exist_ok=True)
        source=run/ready[sid]['video_relative'] if sid in ready else root/'datasets/current'/e['video_relative']
        target.symlink_to(source.resolve())
    write_new(dataset/'dataset-index.json',full)
    write_new(root/'plans/classic-refinement/allocation.json',allocation(manifest))
    return {'status':'SOURCE_READY','counts':full['counts'],'index_id':full['index_id'],
            'dataset_root':str(dataset),'source_QA':summary['identity'],'new_GPU_calls':0}


def seal(root, host):
    from .inference import checkpoint_asset_identity
    root=Path(root);op=root/'runs/reproducibility/operator';plan=root/'plans/classic-refinement'
    alloc=read(plan/'allocation.json');catalog=read(plan/'execution-catalog.json')
    oldmanifest=read(root/'runs/t2_t3_classic_rgb16_production_20261020/manifest.json')
    qwen=read(op/'qwen-t2-extension-single-20261024-plan.json') if host=='fourier' else None
    cosmos=read(op/('cosmos-t1t2-nvfp4-'+host+'-runtime-20261025.json'))
    cosmos_anchor=read(op/('cosmos-t1t2-nvfp4-'+host+'-sealed-20261025.json'))
    files={p.name:file_sha(p) for p in (root/'code/vlm-psychophysics/classic_precision').glob('*.py')
           if p.name in ('refinement_worker.py','inference.py','nvfp4_cosmos_official.py','nvfp4_cosmos.py',
               'gemma_speed.py','gemma_cpu_copy.py','gemma_production.py','gemma_nvfp4_continuation.py','nvfp4_gemma.py')}
    profiles=[]
    names=[k for k in alloc['assignments'] if (k.endswith('radon0') if host=='radon' else 'fourier' in k)]
    for name in names:
        model,card=name.split('-',1);gpu=card[-1]
        anchorpath=op/('gemma-nvfp4-formal-tail3-'+card+'-profile-20261026.json')
        anchor=read(anchorpath);runtime=deepcopy(anchor['runtime'])
        code=str(root/'code/vlm-psychophysics')
        env=dict(runtime['environment']) if model in ('gemma','cosmos') else {
            'CLASSIC_PRECISION_KV_COMPAT':'1','CLASSIC_PRECISION_NEMO_DUMMY_CONFIG_COMPAT':'1',
            'CLASSIC_PRECISION_DEFER_MODELS':'gemma','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'NUMPY_MADVISE_HUGEPAGE':'0','VLLM_ALLOW_INSECURE_SERIALIZATION':'1'}
        cleared=[]
        if model=='qwen':
            env.update(qwen['profile']['environment']);cleared=qwen['profile']['cleared_environment']
        elif model=='nemo':
            env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        elif model=='cosmos':
            for k,v in cosmos_anchor['runtime_identity']['environment'].items():
                if v is None:
                    env.pop(k,None);cleared.append(k)
                else:env[k]=v
        env.update(CUDA_VISIBLE_DEVICES=gpu, PYTHONPATH=code+'/runtime_compat:'+code, PYTHONDONTWRITEBYTECODE='1')
        runtime['environment']=dict(env)
        ids=alloc['assignments'][name]
        path=plan/(name+'-profile.json');output=root/'runs/reproducibility/classic-refinement-20261026'/name
        # Bind the exact pre-existing physical lease, not a new namespace.
        lock=cosmos['physical_locks'][gpu if host=='fourier' else '0']
        require(Path(lock['path']).stat().st_ino==lock['st_ino'],'Native physical lease changed')
        profile={'schema':'classic-refinement-worker-v1','worker':name,'model':model,'hostname':socket.gethostname(),
            'worker_token':uuid.uuid4().hex,'assigned_ids':ids,'assignment_sha256':digest(ids),
            'maximum_model_calls':len(ids),'first_deep_calls':2,'automatic_retry':False,
            'catalog':descriptor(plan/'execution-catalog.json'),'allocation':descriptor(plan/'allocation.json'),
            'source_root':str(root/'runs/classic-refinement-20261026'),'output':str(output),
            'profile_file':str(path),'admission_file':str(plan/(name+'-admission.json')),
            'claim_root':str(root/'runs/reproducibility/classic-refinement-20261026/claims'/model),
            'pause_file':str(plan/(name+'-PAUSE.json')),'environment':env,'cleared_environment':cleared,
            'gpu_uuid':GPUS[card],'physical_lock':lock,'implementation':files,'entry_implementation':files,
            'source_QA':descriptor(op/'classic-refinement-source-qa-20261026/summary.json'),'input_protocol':'controlled-rgb16',
            'generation':oldmanifest['generation'],'runtime':runtime}
        if model in ('qwen','nemo'):
            spec=oldmanifest['models'][model]
            profile['checkpoint']=str(root/'runs/inference_models_20261003'/spec['checkpoint_relative'])
            profile['checkpoint_assets']=checkpoint_asset_identity(profile['checkpoint'])
            profile['qualified_model_spec']=spec;profile['resources']=spec['resources']
            if model=='qwen':
                profile['qualified_profile']=qwen['profile'];profile['qualified_anchor']=descriptor(op/'qwen-t2-extension-single-20261024-plan.json')
                require(profile['checkpoint_assets']==qwen['checkpoint_assets'],'Qwen current assets differ')
        elif model=='cosmos':
            profile['runtime']=deepcopy(cosmos)
            profile['runtime']['asset_roots']['classic-refinement']=profile['source_root']
            profile['resources']=cosmos['resources'];profile['checkpoint']=cosmos['checkpoint']
            from .nvfp4_cosmos_official import runtime_identity,engine_settings
            from .nvfp4_cosmos_official import ENV_KEYS
            prior={k:os.environ.get(k) for k in ENV_KEYS}
            try:
                for k in ENV_KEYS:
                    if k in env:os.environ[k]=env[k]
                    else:os.environ.pop(k,None)
                profile['runtime_identity']=runtime_identity(profile['runtime'])
            finally:
                for k,v in prior.items():
                    if v is None:os.environ.pop(k,None)
                    else:os.environ[k]=v
            profile['engine_settings']=engine_settings(profile['runtime'])
            require(profile['runtime_identity']==cosmos_anchor['runtime_identity']
                and profile['engine_settings']==cosmos_anchor['engine_settings'], 'Qualified Cosmos identity/settings differ')
        else:
            for key in ('candidate','variant','implementation','checkpoint_assets','checkpoint_contract','runtime_sources',
                        'kernel','cpu_execution_overlay','cpu_copy','engine_settings','execution_signature','plan_sha256','references'):
                profile[key]=deepcopy(anchor[key])
            profile['qualified_anchor']=descriptor(anchorpath)
            profile['resources']={'min_free_gpu_gib':94,'min_available_ram_gib':64}
            profile['mode']='production';profile['panel']={'items':[]}
            profile['inbox']=str(plan);profile['inbox_timeout_seconds']=1
            require(profile['engine_settings']==__import__('classic_precision.gemma_nvfp4_continuation',fromlist=['settings']).settings(runtime,profile['candidate']), 'Gemma numerical settings differ')
        write_new(path,profile);profiles.append(str(path))
    return {'status':'SEALED_NO_GPU_CALLS','profiles':profiles}


def queue(root, card):
    """Two sealed stages, actual wait between them, never retries or reassigns."""
    root=Path(root);plan=root/'plans/classic-refinement'
    first={'fourier0':'qwen','fourier1':'cosmos','radon0':'nemo'}[card]
    order=('gemma',first) if card=='fourier1' else (first,'gemma')
    stages=[plan/(model+'-'+card+'-profile.json') for model in order]
    state=plan/(card+'-queue.json')
    require(not state.exists(),'Existing queue requires reconciliation, no automatic resume')
    write_new(state,{'status':'RUNNING','stages':list(map(str,stages)),'automatic_retry':False})
    reports=[]
    from .storage import atomic_json
    try:
        for path in stages:
            profile=read(path)
            require(not Path(profile['pause_file']).exists(),'Sealed next worker is paused')
            if profile['model']=='gemma':
                if reports:
                    prior=reports[-1]
                    require(prior['terminal_status']=='COMPLETE' and prior['real_wait_observed']
                        and prior['owned_engines_exited'] and prior['original_physical_lease_released'],
                        'Preceding assigned GPU has not closed cleanly')
                    prior_path=Path(read(stages[0])['output']).with_suffix('.wait.json')
                else:
                    prior_path=root/'runs/reproducibility/operator'/('gemma-nvfp4-formal-tail3-'+card+'-20261026-exit.json')
                    prior=read(prior_path)
                    require(prior['actual_wait_observed'] and prior['returncode']==0,'Historical worker actual wait missing')
                    closure=read(Path(prior['output'])/'exit.json')
                    require(closure['owned_engine_workers_exited'] and closure['lease_released_after_engine_exit'], 'Original closure missing')
                write_new(profile['admission_file'],{
                    'plan_sha256':profile['plan_sha256'],'profile_sha256':digest(profile),
                    'qwen_t1_complete':3840,'all_owned_engines_exited':True,'physical_gpu_locks_released':True,
                    'scope':'Preceding owned worker on this exact GPU; old Qwen T1 completion retained; live headroom and same-inode lease checked at new load',
                    'preceding_wait':descriptor(prior_path)})
            child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.refinement_worker',
                'supervise','--profile',str(path)])
            code=child.wait()
            report=read(Path(profile['output']).with_suffix('.wait.json'))
            require(code==0 and report['returncode']==0,'Finite worker failed; queue held without retry')
            reports.append(report)
            if report['terminal_status']=='PAUSED':
                atomic_json(state,{'status':'PAUSED','reports':reports,'automatic_retry':False})
                return {'status':'PAUSED'}
            require(report['terminal_status']=='COMPLETE'
                and report['completed_ids']==profile['assigned_ids'],'Per-ID completion differs')
        atomic_json(state,{'status':'COMPLETE','reports':reports,'automatic_retry':False})
        return {'status':'COMPLETE'}
    except BaseException as error:
        atomic_json(state,{'status':'HELD_NO_RETRY','reports':reports,'error':str(error),'automatic_retry':False})
        raise


def _tail_records(path, limit=64, maximum_bytes=32*1024**2):
    path=Path(path)
    if not path.exists():return []
    with path.open('rb') as f:
        size=f.seek(0,2);start=max(0,size-maximum_bytes);f.seek(start);data=f.read()
    if start:data=data.split(b'\n',1)[-1]
    return [json.loads(line) for line in data.split(b'\n')[:-1][-limit:] if line]


def status(root):
    """One bounded operational snapshot, without rereading scientific raw arrays."""
    root=Path(root);plan=root/'plans/classic-refinement'
    result={'host':socket.gethostname(),'workers':{},'queues':{}}
    for p in sorted(plan.glob('*-profile.json')):
        q=read(p);out=Path(q['output']);timing=out/'request-timings.jsonl'
        # Small scalar-only journal, bounded by the sealed finite allocation.
        complete=timing.read_bytes().count(b'\n') if timing.exists() else 0
        recent=[x for x in _tail_records(timing) if not x.get('deep')][-20:]
        mean=statistics.mean(x['full_request_seconds'] for x in recent) if recent else None
        wait=out.with_suffix('.wait.json')
        result['workers'][q['worker']]={'completed_with_durable_technical_proof':complete,
            'expected':q['maximum_model_calls'],'last_ordinary_n':len(recent),
            'last_ordinary_mean_request_seconds':mean,'request_timing_scope':recent[-1]['scope'] if recent else None,
            'remaining_seconds_at_this_worker_rate':mean*(q['maximum_model_calls']-complete) if mean else None,
            'wait':read(wait) if wait.exists() else None,
            'progress':read(out/'progress.json') if (out/'progress.json').exists() else None,
            'failure':read(out/'failure.json') if (out/'failure.json').exists() else None}
        if result['workers'][q['worker']]['wait']:
            result['workers'][q['worker']]['wait'].pop('completed_ids',None)
    for p in plan.glob('*-queue.json'):
        d=read(p);result['queues'][p.stem]={'status':d['status'],'error':d.get('error')}
    for name,path in [('generation',root/'runs/classic-refinement-20261026/generation.json'),
                      ('source_QA',root/'runs/reproducibility/operator/classic-refinement-source-qa-20261026/progress.json')]:
        if path.exists():result[name]=read(path)
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
    result['memory_GiB']={k:mem[k]/1024**2 for k in ('MemAvailable','MemTotal','SwapFree','SwapTotal')}
    result['load_average']=os.getloadavg();result['cpu_count']=os.cpu_count()
    def counters():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    before=counters();time.sleep(.2);after=counters();delta=[b-a for a,b in zip(before,after)]
    result['CPU_busy_percent']=100*(1-(delta[3]+delta[4])/sum(delta)) if sum(delta) else 0
    result['PSI']={k:(Path('/proc/pressure')/k).read_text().strip() for k in ('cpu','memory','io')}
    disk=shutil.disk_usage(root);result['disk_GiB']={'free':disk.free/2**30,'total':disk.total/2**30}
    result['GPU']=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader'],check=True,capture_output=True,text=True).stdout.strip().splitlines()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('sources','seal','queue','status'))
    p.add_argument('--root',required=True);p.add_argument('--host',choices=HOSTS);p.add_argument('--card',choices=GPUS)
    a=p.parse_args()
    result=sources(a.root) if a.command=='sources' else seal(a.root,a.host) if a.command=='seal' else status(a.root) if a.command=='status' else queue(a.root,a.card)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
