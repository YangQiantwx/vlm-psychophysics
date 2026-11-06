"""Finite T1 m=.5 extension of the verified static X executor."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.metadata
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from . import official_x_runner as base
from . import static_x_admission as native
from . import static_x_operations as parent
from . import static_x_queue as lifecycle
from . import t4_balanced_operations as supervisor
from .static_t4_sources import descriptor as ref, digest, read, require, verified, write_new

NAME='static-t1-m05'
MODULE='classic_precision.static_t1_m05'
ROOT=Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve()
CODE=ROOT/'code/vlm-psychophysics'
PLAN=ROOT/'plans'/NAME
OUT=ROOT/'runs/reproducibility'/NAME
CARDS=('fourier0','fourier1')
MODELS=('cosmos','qwen','nemo','gemma')


def launch_argv(p, command, *args):
    return [p['python'],'-B','-m',MODULE,command,*map(str,args)]


def environment(p):
    env=dict(os.environ)
    for k in p.get('cleared_environment',[]):env.pop(k,None)
    return env|p['environment']


def configuration(p):
    authority=native.cached_verified(p['static_authorization'])
    require(p['static_authorization']==p['full_authorization'] and authority['status']=='AUTHORIZED_M05_ONLY'
            and authority['maximum_new_model_calls']==3072 and p['source_catalog']==authority['catalog']
            and p['static_protocol']==authority['protocol'], 'Wrong extension authority')
    model=native.model_key(p)
    require(model in MODELS and p['card'] in CARDS and p['worker']==f"{model}-{p['card']}",'Wrong model/card')
    catalog=native.cached_verified(p['source_catalog'])
    items=catalog['items']
    require(catalog['status']=='PASS_EXACT_T1_RGB8_KEYFRAMES' and catalog['count']==len(items)==768,'Wrong source scope')
    require(p['assigned_ids']==authority['assigned_ids'][p['card']]
            and p['maximum_model_calls']==len(p['assigned_ids'])==384,'Wrong finite allocation')
    require(p['output']==str(OUT/p['worker']) and p['claim_root']==p['output']+'-claims'
            and p['automatic_retry'] is False and p['formal_auto_resume'] is False,'Wrong owned output or retry policy')
    old=native.cached_verified(p['parent_validated_profile'])
    require(native.deployment_signature(old)==native.deployment_signature(p)
            and all(p.get(k)==old.get(k) for k in ('gpu_uuid','physical_lock','hostname','minimum_kv_capacity',
                'request_mm_processor_kwargs_by_task','parent_runtime','bootstrap')),'Static deployment changed')
    require(p['choices_by_task']=={'T1':['LEFT','RIGHT']} and p['static_cold'] is None,'Wrong T1 response contract')
    if model=='qwen':
        require(p['engine_kwargs']['max_model_len']==98304 and p['engine_kwargs']['kv_cache_memory_bytes']==8*2**30,
                'Wrong qualified four-image Qwen deployment')
    protocol=native.cached_verified(p['static_protocol'])
    for sid in p['assigned_ids']:
        item=items[sid]
        require(item['sample_id']==item['case_id']==sid and item['task']=='T1','Wrong T1 identity')
        native.check_item(item,protocol)
        parent.input_identity(item)
    return catalog


def policy(p):
    catalog=configuration(p)
    require(p['execution_policy']=={'module':MODULE,'source_sha256':ref(__file__)['sha256']}
            and p['static_entry']==ref(native.__file__) and p['static_dependencies']==native.dependencies()
            and p['implementation']==ref(base.__file__),'Frozen execution code changed')
    parent.verify_cpu(p,catalog)
    parent.qualified_deployment(p)
    return catalog


def prepare():
    cpath=ROOT/'datasets'/NAME/'catalog.json'
    catalog=read(cpath)
    require(catalog['count']==768 and len(catalog['items'])==768,'T1 source not complete')
    parent_dir=ROOT/'plans/static-x-t123-20261104/full'
    by_domain=defaultdict(list)
    for sid,item in sorted(catalog['items'].items()):
        by_domain[item['domain']].append(sid)
    require({k:len(v) for k,v in by_domain.items()}=={'classic':384,'natural':384},'Wrong domain counts')
    allocations={card:sorted(sid for values in by_domain.values() for sid in values[i::2])
                 for i,card in enumerate(CARDS)}
    require(len(set(allocations[CARDS[0]])|set(allocations[CARDS[1]]))==768,'Invalid card split')
    authority=write_new(PLAN/'authorization.json',{'status':'AUTHORIZED_M05_ONLY','maximum_new_model_calls':3072,
        'assigned_ids':allocations,'catalog':ref(cpath),'protocol':catalog['protocol'],
        'scope':'Only the currently missing Classic and Natural T1 m=0.5 static X inputs'})
    for card in CARDS:
        for model in MODELS:
            label=model+'-'+card
            filename=('qwen_T1' if model=='qwen' else model)+'-'+card+'-profile.json'
            old_path=parent_dir/filename
            old=read(old_path)
            lifecycle.profile_closed(ref(old_path),old)
            p=copy.deepcopy(old)
            for k in ('worker_token','media_files','static_cpu_budget','static_deep_ids','static_cpu_template_map',
                      'static_start_gate','final_review_draft','final_static_review','full_allocation_id','cold'):
                p.pop(k,None)
            p.update(worker=label,card=card,source_catalog=ref(cpath),static_authorization=authority,full_authorization=authority,
                     static_protocol=catalog['protocol'],phase='full',static_cold=None,choices_by_task={'T1':['LEFT','RIGHT']},
                     assigned_ids=allocations[card],maximum_model_calls=384,parent_validated_profile=ref(old_path),
                     output=str(OUT/label),claim_root=str(OUT/(label+'-claims')),pause_file=str(OUT/(label+'-pause.json')),
                     execution_policy={'module':MODULE,'source_sha256':ref(__file__)['sha256']},
                     automatic_retry=False,formal_auto_resume=False)
            configuration(p)
            write_new(PLAN/(label+'-draft.json'),p)
    print('Eight unchanged static X deployments drafted',flush=True)


def cpu(label):
    import torch
    p=read(PLAN/(label+'-draft.json'))
    catalog=configuration(p)
    require(sys.executable==p['python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),
            'CPU admission requires the frozen deployment interpreter and invisible CUDA')
    for package,version in p['expected_packages'].items():
        require(importlib.metadata.version(package)==version,'Native package changed: '+package)
    torch.set_num_threads(p.get('torch_num_threads') or 1)
    reps,mapping=parent.representative_map(p,catalog)
    require({catalog['items'][s]['domain'] for s in reps}=={'classic','natural'},'Missing representative domain')
    attempts=[native.cpu_sample(p,catalog['items'][s]) for s in reps]
    require(not torch.cuda.is_initialized() and all(x['image_count']==4 for x in attempts),'Native four-frame check failed')
    write_new(PLAN/(label+'-cpu.json'),{'schema':'static-x-full-native-cpu-v1','status':'PASS','model_calls':0,
        'weights_loaded':False,'CUDA_initialized':False,**parent.budget_binding(p),
        'representative_ids':reps,'sid_to_reference':mapping,'attempts':attempts,
        'assigned_item_digests':{s:digest(catalog['items'][s]) for s in p['assigned_ids']}})
    print(label,'CPU PASS, 2 representatives',flush=True)


def cpu_all():
    def run_cpu(label):
        p=read(PLAN/(label+'-draft.json'))
        with (PLAN/(label+'-cpu.log')).open('xb') as log:
            subprocess.run(launch_argv(p,'cpu',label),env=environment(p)|{'CUDA_VISIBLE_DEVICES':''},
                           cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(label,'CPU PASS',flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_cpu,[m+'-'+c for m in MODELS for c in CARDS]))


def seal():
    profiles={}
    from .gemma_x_full import storage_admission
    for card in CARDS:
        profiles[card]=[]
        for model in MODELS:
            label=model+'-'+card
            p=read(PLAN/(label+'-draft.json'))
            budget=PLAN/(label+'-cpu.json')
            b=read(budget)
            p.update(static_cpu_budget=ref(budget),static_deep_ids=b['representative_ids'],
                     static_cpu_template_map=b['sid_to_reference'])
            policy(p)
            storage_admission(p)
            supervisor.claims_gate(p)
            target=PLAN/(label+'-profile.json')
            with tempfile.TemporaryDirectory(prefix='static-t1-m05-seal-') as temp:
                draft=Path(temp)/'profile.json'
                write_new(draft,p)
                base.seal(draft,target)
            profiles[card].append(ref(target))
    write_new(PLAN/'sealed.json',{'status':'PASS_NATIVE_CPU_AND_SEALED','profiles':profiles,
        'runner':ref(__file__),'catalog':ref(ROOT/'datasets'/NAME/'catalog.json'),
        'source_code':ref(PLAN/'sources.py'),'model_calls':0})
    print('Eight workers sealed after 16 native four-image CPU admissions',flush=True)


def start_gate(p):
    launch=read(OUT/'launch.json')
    require(lifecycle.process_alive(launch['owner']) and launch['seal']==ref(PLAN/'sealed.json'),'Missing original dispatcher')
    q=read(OUT/p['card']/'identity.json')
    require(lifecycle.process_alive(q['owner']) and q['launcher']==launch['owner'],'Missing original card queue')
    profile_ref=ref(PLAN/(p['worker']+'-profile.json'))
    require(profile_ref in verified(launch['seal'])['profiles'][p['card']]
            and verified(profile_ref)==p,'Unsealed worker')
    child=read(OUT/p['card']/(p['worker']+'-supervisor.json'))
    require(child['queue']==q['owner'] and child['profile']==profile_ref
            and lifecycle.process_identity(os.getppid())==child['supervisor'],'Unknown model supervisor')
    return {'status':'PASS_EXCLUSIVE_T1_M05_DISPATCH','queue':q['owner'],'profile':profile_ref}


def runtime_validator(p,runtime):
    return parent.qualified_deployment(p,runtime)


def deep_policy(p,item,ordinal):
    return parent.representative_deep(p,item,ordinal)


def input_validator(p,item,raw,deep):
    return parent.full_input(p,item,raw,deep)


def run(path):
    old=native.start_gate
    native.start_gate=start_gate
    try:
        return native.execute_bound_profile(path,policy_validator=policy,input_validator=input_validator,
                                            deep_policy=deep_policy,runtime_validator=runtime_validator)
    finally:
        native.start_gate=old


def supervise(path):
    p=read(path)
    policy(p)
    old=supervisor.MODULE
    supervisor.MODULE=MODULE
    try:
        return supervisor.supervise(path)
    finally:
        supervisor.MODULE=old


def queue(card):
    launch=read(OUT/'launch.json')
    sealed=verified(launch['seal'])
    require(lifecycle.process_identity(os.getppid())==launch['owner'],'Wrong queue dispatcher')
    out=OUT/card
    out.mkdir()
    identity=lifecycle.process_identity()
    write_new(out/'identity.json',{'owner':identity,'launcher':launch['owner']})
    completed=[]
    try:
        for profile in sealed['profiles'][card]:
            p=verified(profile)
            if native.model_key(p)=='gemma':
                deadline=time.monotonic()+3600
                others=[ref for refs in sealed['profiles'].values() for ref in refs[:-1]]
                while not all(Path(verified(x)['output']).with_suffix('.wait.json').exists() for x in others):
                    require(time.monotonic()<deadline and not any((OUT/c/'failure.json').exists() for c in CARDS),
                            'Non-Gemma peer failed or missed deadline')
                    time.sleep(2)
                for x in others:lifecycle.profile_closed(x,verified(x))
            require(lifecycle.process_alive(launch['owner']),'Original dispatcher stopped')
            with (out/(p['worker']+'.log')).open('xb') as log:
                child, code=lifecycle.spawn_and_wait(launch_argv(p,'supervise',profile['path']),log=log,cwd=CODE,
                    env=environment(p),child_receipt=out/(p['worker']+'-supervisor.json'),
                    receipt_fields={'queue':identity,'profile':profile},identity_key='supervisor')
            require(code==0 and not lifecycle.process_alive(child),'Model supervisor failed, no retry')
            completed.append(lifecycle.profile_closed(profile,p))
        write_new(out/'result.json',{'status':'COMPLETE','completed':completed})
    except Exception as error:
        write_new(out/'failure.json',{'status':'FAILED','error':str(error),'completed':completed,'automatic_retry':False})
        raise


def close(sealed,waits):
    rows=Counter()
    cells=defaultdict(Counter)
    files={}
    def keep(path):
        d=ref(path)
        files[d['path']]=d['sha256']
    for card in CARDS:
        require(read(OUT/card/'result.json')['status']=='COMPLETE','No card terminal')
        for f in (OUT/card).glob('*.json'):keep(f)
        for profile in sealed['profiles'][card]:
            p=verified(profile)
            lifecycle.profile_closed(profile,p)
            supervisor.claims_gate(p,complete=True)
            catalog=policy(p)
            path=Path(p['output'])
            raw=native.strict_records(path/'raw.jsonl')
            proof=native.strict_records(path/'input-admissions.jsonl')
            require([x['sample_id'] for x in raw]==[x['sample_id'] for x in proof]==p['assigned_ids'],'Raw coverage differs')
            for answer, evidence in zip(raw,proof,strict=True):
                sid=answer['sample_id']; item=catalog['items'][sid]
                require(evidence['raw_digest']==digest(answer) and answer['truth']==item['row']['truth']
                        and answer['question']==item['row']['prompt'] and answer['finish_reason']=='stop'
                        and [x['pixel_sha256'] for x in answer['native_reader']]==[x['pixel_sha256'] for x in native.image_entries(item)],
                        'Raw/source/native input binding differs')
                model=native.model_key(p); domain=item['domain']
                rows[model,domain]+=1
                key=(model,domain,item['row']['condition_key'][1])
                value=cells[key]
                value['n']+=1; value['correct']+=int(answer['raw_answer'].strip()==answer['truth'])
            for f in path.iterdir():
                if f.is_file() and f.suffix in ('.json','.jsonl'):keep(f)
            for f in Path(p['claim_root']).iterdir():keep(f)
            for f in (Path(profile['path']),path.with_suffix('.wait.json'),path.with_suffix('.launch.json'),
                      Path(p['static_cpu_budget']['path'])):keep(f)
    require(rows==Counter({(m,d):384 for m in MODELS for d in ('classic','natural')})
            and len(cells)==64 and all(r['n']==48 for r in cells.values()),'Wrong exact scientific denominator')
    for p in ('sealed.json','authorization.json','sources.py','release-binding.json'):keep(PLAN/p)
    for d in (sealed['catalog'],sealed['runner']):keep(d['path'])
    for d in parent.final_review_codes().values():keep(d['path'])
    keep(OUT/'launch.json')
    result={'status':'COMPLETE_PASS','model_calls':3072,'unique_inputs':768,'queue_real_waits':waits,
            'cells':[{'model':m,'domain':d,'sf':sf,**n} for (m,d,sf),n in sorted(cells.items())],
            'return_files':files}
    write_new(OUT/'terminal.json',result)
    print('COMPLETE_PASS 3072 raw answers, 64 cells, two real outer exits',flush=True)


def launch():
    seal_ref=ref(PLAN/'sealed.json')
    sealed=verified(seal_ref)
    require(sealed['runner']==ref(__file__) and os.environ.get('CUDA_VISIBLE_DEVICES')=='','Invalid controller')
    OUT.mkdir(parents=True)
    owner=lifecycle.process_identity()
    write_new(OUT/'launch.json',{'owner':owner,'seal':seal_ref,'automatic_retry':False})
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),
            'Another GPU workload is present')
    children={}
    for card in CARDS:
        p=verified(sealed['profiles'][card][0])
        with (OUT/(card+'.log')).open('xb') as log:
            process=subprocess.Popen(launch_argv(p,'queue',card),cwd=CODE,env=dict(os.environ),
                                     stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        children[card]=(process,lifecycle.process_identity(process.pid))
    waits={c:{'child':i,'returncode':p.wait(),'real_wait_observed':True} for c,(p,i) in children.items()}
    write_new(OUT/'outer-waits.json',waits)
    require(all(w['returncode']==0 and not lifecycle.process_alive(w['child']) for w in waits.values()),'Card queue failure')
    close(sealed,waits)


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument('command',choices=['prepare','cpu','cpu_all','seal','run','supervise','queue','launch'])
    cli.add_argument('arg',nargs='?')
    cli.add_argument('--profile')
    a=cli.parse_args()
    arg=a.profile or a.arg
    if arg: globals()[a.command](arg)
    else: globals()[a.command]()

if __name__=='__main__':
    main()
