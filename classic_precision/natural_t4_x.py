"""Native X execution for the finite Natural T4 video and endpoint panels."""
import argparse
from collections import Counter,defaultdict
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
from . import static_x_admission as static
from . import static_x_operations as static_parent
from . import static_x_queue as life
from . import t4_balanced_operations as previous
from .static_t4_sources import descriptor as ref, digest, read, require, verified, write_new

MODULE='classic_precision.natural_t4_x'
ROOT=Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve()
CODE=ROOT/'code/vlm-psychophysics'
NAME='t4-natural-strict-v1'
PLAN=ROOT/'plans'/NAME
DATA=ROOT/'datasets'/NAME
RUN=ROOT/'runs/reproducibility'/NAME
CARDS=('fourier0','fourier1')
MODELS=('cosmos','qwen','nemo','gemma')
MODALITIES=('video','static')
CONTRACT={'frames':180,'fps':60,'width':768,'height':512}
INHERIT=('checkpoint','checkpoint_assets','engine_kwargs','environment','cleared_environment',
         'expected_packages','python','hostname','gpu_uuid','physical_lock','workspace_root',
         'request_mm_processor_kwargs','request_mm_processor_kwargs_by_task','media_io_kwargs',
         'vision_observation','minimum_kv_capacity','torch_num_threads','resources','processor_checkpoint',
         'input_dependencies','adapter_image_max_soft_tokens','request_adapter','vision_validation_hook','bootstrap')


def video_route(p):
    if static.model_key(p)=='qwen':
        from . import official_x_runner_qsa as route
    else:
        from . import official_x_runner_v2 as route
    return route


def check_static(item, protocol):
    require(item['task']=='T4' and item['domain']=='natural' and item['image_contract']=={'count':2,'width':768,'height':512}
            and item['video_contract']==CONTRACT,'Wrong two-image contract')
    require([e['type'] for e in item['content']]==['image','image','text']
            and [e['frame_index'] for e in item['content'][:2]]==[0,179],'Wrong endpoint order')
    require(item['row']['prompt']==item['content'][2]['text']==protocol['prompt']
            and item['row']['choices']==['LEFT','RIGHT'] and item['row']['truth'] in ('LEFT','RIGHT'),'Prompt or GT changed')
    require(static_parent.input_identity(item)==item['model_input_sha256'],'Wrong exact image input identity')
    return 2


def source_catalogs():
    a=read(DATA/'source-acceptance.json'); plan=verified(a['plan'])
    require(a['status']=='PASS' and a['count']==a['unique_endpoint_pairs']==len(a['items'])==1536
            and a['decoded_frames']==1536*180,'Full source acceptance is missing')
    protocol=read(ROOT/'plans/static-t4-strict-first-last/protocol.json')
    prompt=plan['video_prompt']
    require(prompt==protocol['source_video_prompt'],'Classic and Natural video prompts differ')
    revised=prompt
    for sub in protocol['minimal_replacements']:
        require(revised.count(sub['old'])==sub['expected_occurrences'],'Prompt substitution cardinality')
        revised=revised.replace(sub['old'],sub['new'])
    require(revised==protocol['prompt'],'Natural endpoints must retain the accepted T4 static prompt')
    refs={}
    static_spec={'prompt':revised,'source_video_prompt':prompt,'frame_indices':[0,179],'scope':NAME,
                 'parent':ref(ROOT/'plans/static-t4-strict-first-last/protocol.json')}
    write_new(PLAN/'static-protocol.json',static_spec)
    catalogs={m:{} for m in MODALITIES}
    for b in plan['blocks']:
        n,k,l=b['N'],b['level'],b['layout']
        for appearance in ('same','distinct'):
            for t in b['trials']:
                sid=f"natural-t4-strict-v1-n{n}-{appearance}-l{l:02d}-c{k:02d}-{t['truth'].lower()}"
                accepted=a['items'][sid]
                require(accepted['status']=='PASS_BOTH_DECODERS_ALL180' and accepted['truth']==t['truth'],'Wrong accepted video')
                params={'N':n,'coherence':k/8,'layout':l,'appearance':appearance,
                        'bg_kind':b['bg_kind'],'obj_kind':b['obj_kind'],'classic_id':t['classic_id']}
                video=dict(type='video',**accepted['video'])
                row=dict(sample_id=sid,task='T4',truth=t['truth'],choices=['LEFT','RIGHT'],
                         accuracy_eligible=True,prompt=prompt,source_parameters=params)
                common=dict(sample_id=sid,case_id=sid,task='T4',domain='natural',video_contract=CONTRACT,
                            parameters=params,accuracy_eligible=True,original_source=accepted['video'],
                            source_receipt=ref(DATA/'media'/sid/'accepted.json'))
                catalogs['video'][sid]=common|dict(row=row,content=[video,dict(type='text',text=prompt)])
                images=accepted['endpoint_images']
                catalogs['static'][sid]=common|dict(row=row|{'prompt':revised},
                    content=images+[dict(type='text',text=revised)],image_contract={'count':2,'width':768,'height':512},
                    model_input_sha256=digest([[x['pixel_sha256'] for x in images],revised,['LEFT','RIGHT']]))
    require(set(catalogs['video'])==set(catalogs['static'])==set(a['items']),'Catalog loses accepted IDs')
    for modality,items in catalogs.items():
        path=PLAN/(modality+'-catalog.json')
        write_new(path,{'schema':'natural-t4-X-catalog-v1','modality':modality,'items':items,
                        'source_acceptance':ref(DATA/'source-acceptance.json'),'plan':a['plan']})
        refs[modality]=ref(path)
    return refs


def expected_ids(catalog,card):
    return sorted(s for s,r in catalog['items'].items() if r['parameters']['layout']%2==int(card[-1]))


def config(p):
    auth=static.cached_verified(p['full_authorization'])
    mode,card=p['modality'],p['card']
    model=static.model_key(p)
    require(mode in MODALITIES and card in CARDS and model in MODELS and p['worker']==model+'-'+card,
            'Invalid model allocation')
    require(auth['status']=='AUTHORIZED_NATURAL_T4_ONLY' and auth['budgets']=={'video':6144,'static':6144}
            and p['source_catalog']==auth['catalogs'][mode] and p['static_authorization']==p['full_authorization'],
            'Source/authority differs')
    require(p['output']==str(RUN/mode/p['worker']) and p['claim_root']==p['output']+'-claims'
            and p['automatic_retry'] is False and p['formal_auto_resume'] is False,'Wrong output or retry policy')
    old=static.cached_verified(p['parent_validated_profile'])
    require(all(p.get(k)==old.get(k) for k in INHERIT) and p['model']==old['model']
            and p['static_cold'] is None,'Inherited deployment changed')
    if mode=='static':
        require(static.deployment_signature(old)==static.deployment_signature(p),'Static numerical configuration changed')
    require(p['choices_by_task']=={'T4':['LEFT','RIGHT']} and p['engine_kwargs']['max_model_len']==old['engine_kwargs']['max_model_len'],
            'Wrong task or context')
    catalog=static.cached_verified(p['source_catalog'])
    require(len(catalog['items'])==1536 and p['assigned_ids']==expected_ids(catalog,card)
            and p['maximum_model_calls']==len(p['assigned_ids'])==768,'Wrong finite ID allocation')
    if mode=='static':
        spec=static.cached_verified(p['static_protocol'])
        for sid in p['assigned_ids']:check_static(catalog['items'][sid],spec)
    return catalog


def prepare():
    catalogs=source_catalogs()
    authorization=PLAN/'authorization.json'
    write_new(authorization,{'status':'AUTHORIZED_NATURAL_T4_ONLY','budgets':{'video':6144,'static':6144},
              'catalogs':catalogs,'reviewed_code':ref(PLAN/'review.json'),'scope':NAME})
    for mode in MODALITIES:
        folder=PLAN/mode;folder.mkdir(exist_ok=True)
        for card in CARDS:
            for model in MODELS:
                worker=model+'-'+card
                if mode=='video':
                    refdir=ROOT/'runs/reproducibility/t1-m05-t4-classic/t4-prep'
                    old_path=refdir/('t4_classic-'+worker+'-profile.json')
                else:
                    refdir=ROOT/'runs/reproducibility/static-t4-strict-first-last/Mac-return'
                    # The complete strict static profiles are on Fourier in plans/.
                    old_path=ROOT/'plans/static-t4-strict-first-last'/(worker+'-profile.json')
                old=read(old_path)
                if mode=='video':
                    wait=read(Path(old['output']).with_suffix('.wait.json'))
                    require(wait['returncode']==0 and wait['real_wait_observed'] and wait['owned_engines_exited']
                            and wait['original_physical_lease_released']
                            and read(Path(old['output'])/'result.json')['completed_ids']==old['assigned_ids'],'Video parent did not close')
                else:
                    life.profile_closed(ref(old_path),old)
                p=copy.deepcopy(old)
                for field in ('worker_token','media_files','static_cpu_budget','static_deep_ids','static_cpu_template_map',
                              'static_start_gate','native_CPU','draft_digest','reference_prompt_catalog'):
                    p.pop(field,None)
                auth=ref(authorization); path=RUN/mode/worker
                p.update(worker=worker,modality=mode,card=card,source_catalog=catalogs[mode],
                         assigned_ids=expected_ids(static.cached_verified(catalogs[mode]),card),maximum_model_calls=768,
                         full_authorization=auth,static_authorization=auth,
                         static_protocol=ref(PLAN/'static-protocol.json'),static_cold=None,phase=old['phase'],
                         output=str(path),claim_root=str(path.with_name(worker+'-claims')),pause_file=str(path.with_name(worker+'-pause.json')),
                         parent_validated_profile=ref(old_path),choices_by_task={'T4':['LEFT','RIGHT']},
                         execution_policy={'module':MODULE,'source_sha256':ref(__file__)['sha256']},
                         automatic_retry=False,formal_auto_resume=False)
                config(p)
                write_new(folder/(worker+'-draft.json'),p)


def environment(p):
    out=dict(os.environ)
    for k in p.get('cleared_environment',[]):out.pop(k,None)
    return out|p['environment']


def cpu(path):
    import torch
    p=read(path); c=config(p)
    require(sys.executable==p['python'] and os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(),
            'Native CPU requires the frozen interpreter and invisible CUDA')
    for name,version in p['expected_packages'].items():
        require(importlib.metadata.version(name)==version,'Wrong version: '+name)
    torch.set_num_threads(p.get('torch_num_threads') or 1)
    dest=Path(path).with_name(p['worker']+'-cpu.json')
    if p['modality']=='static':
        representatives,mapping=static_parent.representative_map(p,c)
        attempts=[static.cpu_sample(p,c['items'][sid]) for sid in representatives]
        require(all(x['image_count']==2 for x in attempts),'Wrong native image count')
        result={'schema':'static-x-full-native-cpu-v1','status':'PASS','model_calls':0,
            'weights_loaded':False,'CUDA_initialized':False,**static_parent.budget_binding(p),
            'representative_ids':representatives,'sid_to_reference':mapping,'attempts':attempts,
            'assigned_item_digests':{sid:digest(c['items'][sid]) for sid in p['assigned_ids']}}
    else:
        sid=p['assigned_ids'][0]; item=c['items'][sid]; table=None
        if static.model_key(p)=='gemma':
            from . import gemma_x_full_input as adapter
            from vllm.multimodal.media.image import ImageMediaIO
            from vllm.multimodal.media.video import VideoMediaIO
            target=dest.with_suffix('.gemma-prompts.json')
            original=adapter.TASKS
            try:
                adapter.TASKS=('T4',)
                adapter.build_prompt_catalog(p,{'items':{sid:item}},target)
            finally:
                adapter.TASKS=original
            table=ref(target); ids=verified(table)['prompt_token_ids']['T4']
            with base.NativeObserver(p) as observer:
                observer.active,observer.deep,observer.contract=True,True,CONTRACT
                VideoMediaIO(ImageMediaIO(),**base.MEDIA['video']).load_file(Path(item['content'][0]['path']))
            video_reader=observer.reads[0]
        else:
            native=previous.cpu_video_sample(p,item)
            require(native['fits_context'],'Video context exceeded')
            ids=native['prompt_token_ids'];video_reader=native['source_reader'][0]
        q=verified(item['source_receipt']); render=verified(q['render'])
        check_video_reader(video_reader)
        require(video_reader['per_frame_sha256']==render['per_frame']
                and ids==p['donor']['prompt_ids'] and len(ids)+16384<=p['engine_kwargs']['max_model_len'],
                'Actual all-frame or prompt token CPU mismatch')
        result={'status':'PASS_NATIVE_CPU_NO_MODEL','draft':ref(path),'sample_id':sid,'item_digest':digest(item),
                'reader':video_reader,'prompt_token_ids':ids,'gemma_prompt_table':table}
    require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
    write_new(dest,result)
    print(p['modality'],p['worker'],'native CPU passed',flush=True)


def argv(p,action,*args):
    return [p['python'],'-B','-m',MODULE,action,*map(str,args)]


def cpu_all():
    def job(draft):
        p=read(draft)
        with draft.with_suffix('.log').open('xb') as log:
            subprocess.run(argv(p,'cpu',draft),env=environment(p)|{'CUDA_VISIBLE_DEVICES':''},cwd=CODE,
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print(p['modality'],p['worker'],'PASS',flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex:
        list(ex.map(job,sorted(p for m in MODALITIES for p in (PLAN/m).glob('*-draft.json'))))


def policy(p):
    c=config(p)
    require(p['execution_policy']=={'module':MODULE,'source_sha256':ref(__file__)['sha256']},'Adapter code changed')
    verify_review()
    if p['modality']=='static':
        require(p['static_entry']==ref(static.__file__) and p['static_dependencies']==static.dependencies(),'Static code changed')
        static_parent.verify_cpu(p,c)
        static_parent.qualified_deployment(p)
    else:
        evidence=verified(p['native_CPU'])
        require(evidence['status']=='PASS_NATIVE_CPU_NO_MODEL' and evidence['sample_id']==p['assigned_ids'][0]
                and evidence['item_digest']==digest(c['items'][evidence['sample_id']])
                and evidence['prompt_token_ids']==p['donor']['prompt_ids'],'Video CPU proof differs')
    return c


def seal():
    from .gemma_x_full import storage_admission
    sets={}
    for mode in MODALITIES:
        sets[mode]={}
        for card in CARDS:
            sets[mode][card]=[]
            for m in MODELS:
                worker=m+'-'+card
                path=PLAN/mode
                p=read(path/(worker+'-draft.json'))
                budget=ref(path/(worker+'-cpu.json'));b=verified(budget)
                if mode=='static':
                    p.update(static_cpu_budget=budget,static_deep_ids=b['representative_ids'],
                             static_cpu_template_map=b['sid_to_reference'])
                else:
                    p['native_CPU']=budget
                    if b['gemma_prompt_table']:p['reference_prompt_catalog']=b['gemma_prompt_table']
                policy(p);storage_admission(p);previous.claims_gate(p)
                target=path/(worker+'-profile.json')
                with tempfile.TemporaryDirectory(prefix='natural-t4-seal-') as tmp:
                    draft=Path(tmp)/'p.json'
                    write_new(draft,p)
                    (base if mode=='static' else video_route(p)).seal(draft,target)
                sets[mode][card].append(ref(target))
    write_new(PLAN/'sealed.json',{'status':'PASS_SCIENCE_NATIVE_AND_SEALED','sets':sets,'code':ref(__file__),
              'review':ref(PLAN/'review.json')})
    print('16 workers sealed after real native CPU checks',flush=True)


def verify_review():
    review=read(PLAN/'review.json')
    require(review['status']=='PASS_ASTRA_XHIGH_CODE_REVIEW','Independent code approval missing')
    for f in review['files']:
        require(ref(f['path'])==f,'Reviewed file changed')
    require(ref(__file__) in review['files'],'Execution module was not reviewed')


def start_gate(p):
    launch=read(RUN/'launch.json')
    require(life.process_alive(launch['owner']) and launch['seal']==ref(PLAN/'sealed.json'),'Original dispatcher absent')
    out=RUN/p['modality']/p['card']
    queue=read(out/'identity.json')
    require(life.process_alive(queue['owner']) and queue['launcher']==launch['owner'],'Card queue absent')
    profile=ref(PLAN/p['modality']/(p['worker']+'-profile.json'))
    require(profile in verified(launch['seal'])['sets'][p['modality']][p['card']] and verified(profile)==p,
            'Worker not in approved queue')
    child=read(out/(p['worker']+'-supervisor.json'))
    require(child['queue']==queue['owner'] and child['profile']==profile
            and child['supervisor']==life.process_identity(os.getppid()),'Wrong model supervisor')
    return {'status':'PASS_ORIGINAL_NATURAL_T4_QUEUE','profile':profile,'queue':queue['owner']}


def runtime(p,actual):
    if p['modality']=='static':
        return static_parent.qualified_deployment(p,actual)
    old=verified(p['parent_validated_profile'])
    expected=read(Path(old['output'])/'runtime.json')
    require([previous.runtime_signature(x) for x in actual]==[previous.runtime_signature(x) for x in expected],
            'Actual video deployment differs from the accepted Classic T4 runtime')


def input_validator(p,item,raw,deep):
    return static_parent.full_input(p,item,raw,deep)


def deep_policy(p,item,ordinal):
    return static_parent.representative_deep(p,item,ordinal)


def run(path):
    p=read(path)
    if p['modality']=='static':
        protocol = verified(p['static_protocol'])
        require(protocol['scope'] == NAME, 'Wrong Natural T4 static protocol')
        old_check,old_gate=static.check_item,static.start_gate

        def scoped_check(item, candidate):
            if candidate == protocol:
                return check_static(item, candidate)
            return old_check(item, candidate)

        static.check_item,static.start_gate=scoped_check,start_gate
        try:
            return static.execute_bound_profile(path,policy_validator=policy,input_validator=input_validator,
                                                deep_policy=deep_policy,runtime_validator=runtime)
        finally:
            static.check_item,static.start_gate=old_check,old_gate
    policy(p); start_gate(p)
    from .gemma_x_full import storage_admission
    write_new(Path(p['output']).with_suffix('.storage.json'),storage_admission(p))
    route=video_route(p)
    native=route.v2 if static.model_key(p)=='qwen' else route
    identity=native.base.worker_identity
    if static.model_key(p)=='nemo':
        from .nemo_x_full import worker_identity as identity
    elif static.model_key(p)=='gemma':
        from .gemma_x_full import worker_identity as identity
    def checked(worker,model):
        actual=identity(worker,model)
        runtime(p,[actual])
        return actual
    native.base.worker_identity=checked
    return native.run(path)


def supervise(path):
    p=read(path);policy(p)
    old=previous.MODULE
    previous.MODULE=MODULE
    try:
        return previous.supervise(path)
    finally:
        previous.MODULE=old


def queue(mode,card):
    launch=read(RUN/'launch.json')
    sealed=verified(launch['seal'])
    require(life.process_identity(os.getppid())==launch['owner'],'Wrong dispatcher')
    out=RUN/mode/card;out.mkdir(parents=True)
    owner=life.process_identity()
    write_new(out/'identity.json',{'owner':owner,'launcher':launch['owner']})
    completed=[]
    try:
        for profile in sealed['sets'][mode][card]:
            p=verified(profile)
            if static.model_key(p)=='gemma':
                other=[r for jobs in sealed['sets'][mode].values() for r in jobs[:-1]]
                waits=[Path(verified(r)['output']).with_suffix('.wait.json') for r in other]
                deadline=time.monotonic()+43200
                while not all(f.exists() for f in waits):
                    require(time.monotonic()<deadline and not any((RUN/mode/c/'failure.json').exists() for c in CARDS),'Peer failed')
                    time.sleep(3)
                for r in other:life.profile_closed(r,verified(r))
            require(life.process_alive(launch['owner']),'Controller exited')
            with (out/(p['worker']+'.log')).open('xb') as log:
                child,rc=life.spawn_and_wait(argv(p,'supervise',profile['path']),log=log,cwd=CODE,
                    env=environment(p),child_receipt=out/(p['worker']+'-supervisor.json'),
                    receipt_fields={'queue':owner,'profile':profile},identity_key='supervisor')
            require(rc==0 and not life.process_alive(child),'Worker failed; preserve original attempt')
            completed.append(life.profile_closed(profile,p))
        write_new(out/'result.json',{'status':'COMPLETE','completed':completed})
    except Exception as exc:
        write_new(out/'failure.json',{'error':str(exc),'completed':completed,'automatic_retry':False})
        raise


def check_video_reader(r):
    require(r['shape']==[180,512,768,3] and r['dtype']=='uint8'
            and r['metadata']['frames_indices']==list(range(180)) and r['metadata']['fps']==60
            and r['metadata']['duration']==3.0,'Actual video frame count, geometry or timing differs')


def finish(mode,sets,waits):
    cells=defaultdict(Counter);seen=set();files={}
    def keep(path):
        f=ref(path)
        files[f['path']]=f['sha256']
    for card in CARDS:
        qdir=RUN/mode/card
        require(read(qdir/'result.json')['status']=='COMPLETE','Missing card terminal')
        for f in qdir.glob('*.json'):keep(f)
        for profile in sets[card]:
            p=verified(profile);life.profile_closed(profile,p);previous.claims_gate(p,complete=True)
            items=policy(p)['items'];out=Path(p['output'])
            records=static.strict_records(out/'raw.jsonl')
            journal=static.strict_records(out/'events.jsonl')
            profile_digest=digest(p)
            video_tokens=verified(p['native_CPU'])['prompt_token_ids'] if mode=='video' else None
            static_tokens={a['sample_id']:a['prompt_token_ids'] for a in
                           verified(p['static_cpu_budget'])['attempts']} if mode=='static' else {}
            ids=p['assigned_ids']
            require([r['sample_id'] for r in records]==ids and [(r['event'],r['sample_id']) for r in journal]
                    ==[(event,s) for s in ids for event in ('INTENT','TECHNICAL_PASS')],'Wrong raw or event coverage')
            for ordinal,row in enumerate(records):
                sid=row['sample_id'];item=items[sid];gt=item['row']['truth'];m=static.model_key(p)
                claim=read(Path(p['claim_root'])/(sid+'.json'))
                require((m,sid) not in seen and row['truth']==gt and row['question']==item['row']['prompt']
                        and row['finish_reason']=='stop' and row['raw_answer'].strip() in ('LEFT','RIGHT')
                        and row['profile_sha256']==claim['profile_sha256']==profile_digest
                        and row['attempt']==claim['attempt']==1 and row['ordinal']==ordinal
                        and row['model']==p['model'] and row['task']=='T4'
                        and claim['sample_id']==sid and claim['worker_token']==p['worker_token']
                        and row['source']['catalog']==p['source_catalog'] and row['source']['item_digest']==digest(item)
                        and row['request']==base.messages(p,sid,item)
                        and journal[2*ordinal+1]['raw_digest']==digest(row),'Wrong accepted model answer')
                require(row['actual_prompt_token_ids']==(video_tokens if mode=='video'
                        else static_tokens[p['static_cpu_template_map'][sid]]),'Actual full prompt token IDs differ')
                seen.add((m,sid))
                static_parent.generation_contract(row['generation'])
                if mode=='video':
                    require(len(row['native_reader'])==1,'Wrong native video count')
                    r=row['native_reader'][0];check_video_reader(r)
                    if row['deep']:
                        q=verified(item['source_receipt'])
                        require(r['per_frame_sha256']==verified(q['render'])['per_frame'],'Actual GPU request source pixels differ')
                else:
                    require([r['pixel_sha256'] for r in row['native_reader']]==[r['pixel_sha256'] for r in item['content'][:2]],
                            'Actual static endpoint pixels differ')
                k=item['parameters']
                counts=cells[m,k['N'],k['appearance'],k['coherence'],k['bg_kind'],k['obj_kind']]
                counts['n']+=1;counts[gt]+=1;counts['correct']+=int(row['raw_answer'].strip()==gt)
                counts['answers_RIGHT']+=int(row['raw_answer'].strip()=='RIGHT')
            for f in out.iterdir():
                if f.is_file() and f.suffix in ('.json','.jsonl'):keep(f)
            for f in Path(p['claim_root']).iterdir():keep(f)
            for f in (profile['path'],out.with_suffix('.wait.json'),out.with_suffix('.launch.json'),
                      p['source_catalog']['path'],p.get('native_CPU',p.get('static_cpu_budget'))['path']):keep(f)
    require(len(seen)==6144 and len(cells)==512 and all(c['n']==12 and c['LEFT']==c['RIGHT']==6 for c in cells.values()),
            'Wrong stratified denominators')
    keep(RUN/mode/'outer-waits.json')
    summary={'status':'COMPLETE_PASS','modality':mode,'model_calls':6144,'unique_inputs':1536,'outer_waits':waits,
              'cells':[dict(model=m,N=n,appearance=a,coherence=k,bg_kind=b,obj_kind=o,**v)
                       for (m,n,a,k,b,o),v in sorted(cells.items())],
              'return_files':files}
    write_new(RUN/mode/'terminal.json',summary)
    print(mode,'COMPLETE_PASS 6144 raw answers',flush=True)


def launch():
    verify_review()
    sealed_ref=ref(PLAN/'sealed.json');sealed=verified(sealed_ref)
    require(sealed['code']==ref(__file__) and sealed['review']==ref(PLAN/'review.json'),'Sealed code/review changed')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','Controller must be CPU-only')
    RUN.mkdir(parents=True)
    write_new(RUN/'launch.json',{'owner':life.process_identity(),'seal':sealed_ref,'automatic_retry':False})
    require(not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).strip(),
            'Another compute workload is present')
    for mode in MODALITIES:
        children={}
        (RUN/mode).mkdir()
        for card in CARDS:
            p=verified(sealed['sets'][mode][card][0])
            with (RUN/mode/(card+'.log')).open('xb') as log:
                child=subprocess.Popen(argv(p,'queue',mode,card),cwd=CODE,env=dict(os.environ),
                                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            children[card]=(child,life.process_identity(child.pid))
        waits={c:{'child':i,'returncode':p.wait(),'real_wait_observed':True} for c,(p,i) in children.items()}
        write_new(RUN/mode/'outer-waits.json',waits)
        require(all(w['returncode']==0 and not life.process_alive(w['child']) for w in waits.values()),'Card queue failure')
        finish(mode,sealed['sets'][mode],waits)
    write_new(RUN/'terminal.json',{'status':'COMPLETE_PASS','model_calls':12288,
              'video':ref(RUN/'video/terminal.json'),'static':ref(RUN/'static/terminal.json'),
              'source':ref(DATA/'source-acceptance.json'),'review':ref(PLAN/'review.json')})
    print('VIDEO + STATIC COMPLETE_PASS',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('action',choices=('prepare','cpu','cpu_all','seal','run','supervise','queue','launch'))
    cli.add_argument('args',nargs='*')
    cli.add_argument('--profile')
    a=cli.parse_args()
    globals()[a.action](*(a.args if a.profile is None else [a.profile]))

if __name__=='__main__':main()
