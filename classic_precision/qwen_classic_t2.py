"""Finite classic T2 scan on the unchanged, cross-card-qualified Qwen X runner.

This module changes experiment admission and lifecycle only. No reader,
processor, forward, selector mathematics, sampling or kernel policy changes.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import socket
import tempfile

from . import nemo_x_full as lifecycle
from . import qwen_x_crosscard as qualified
from .refinement_worker import descriptor,read,require,verified,write_new
from .storage import atomic_json,digest,file_sha256
NAME = 'qwen-x-classic-t2-20261029'
ROOTS = {'fourier':'/home/yang/cvpr2027_benchmark','radon':'/mnt/sda/q/cvpr2027_benchmark'}

qsa=qualified.qsa
base=qsa.base

def science(p,resolve=verified):
    qualified.fixed_implementation()
    template=resolve(p['qualified_profile'])
    for key in ('model','base_model','checkpoint','checkpoint_assets','engine_kwargs','expected_packages',
                'cleared_environment','environment','request_mm_processor_kwargs','vision_observation',
                'minimum_kv_capacity','torch_num_threads','gpu_uuid','physical_lock','python'):
        require(p[key]==template[key],'Qualified Qwen X setting changed: '+key)
    require(not p.get('bootstrap') and not p.get('request_adapter'),'Unexpected input adaptation')
    require(p.get('media_io_kwargs',base.MEDIA)==base.MEDIA,'Official full-frame reader changed')
    require(p['phase']=='classic_T2' and not p['automatic_retry'] and not p['formal_auto_resume'],'Wrong phase/retry')
    plan=resolve(p['experiment_plan']);auth=resolve(p['execution_authorization'])
    require(plan['schema']=='classic-T2-Qwen-X-execution-v1' and plan['maximum_new_model_calls']==2880,'Wrong finite scope')
    require(auth['status']=='AUTHORIZED' and auth['experiment_plan_sha256']==p['experiment_plan']['sha256']
            and auth['maximum_new_model_calls']==2880 and auth['user_message'],'Missing bound authorization')
    gate=resolve(p['qualification_summary'])
    require(p['qualification_summary']['sha256']==plan['qualification_sha256']
            and gate['status']=='PASS_FIXED144_THREE_CARDS' and gate['technical_status']=='COMPLETE_PASS'
            and gate['executed_model_calls']==432 and gate['unique_video_ids']==144
            and gate['full_output_differences']==[],'Existing qualification differs')
    full=plan['cards'][p['card']]['assigned_ids'];ids=p['assigned_ids']
    require(p['source_catalog']==plan['cards'][p['card']]['source_catalog'],'Catalog not bound to plan')
    require(len(ids)==len(set(ids))==p['maximum_model_calls'] and set(ids)<=set(full),'Repeated/foreign IDs')
    if p.get('continuation_parent'):
        parent=resolve(p['continuation_parent'])
        require(p['control_generation']==parent['control_generation']+1,'Resume generation differs')
        if resolve is verified:
            parent,_,raw=closed(p['continuation_parent']['path'])
            require(ids==[sid for sid in parent['assigned_ids'] if sid not in raw],'Resume is not the unattempted suffix')
    else:require(ids==full and p['control_generation']==0,'Initial allocation changed')
    root=Path(p['workspace_root'])
    require(Path(p['output']).parent==root/'runs/reproducibility'/NAME
            and Path(p['claim_root'])==root/'runs/reproducibility'/NAME/'claims'/p['card']
            and Path(p['control_file'])==root/'plans'/NAME/(p['card']+'-control.json'),'Run namespace differs')
    require(p['canonical_catalog']['sha256']==plan['canonical_catalog_sha256'],'Canonical identity changed')
    canonical=resolve(p['canonical_catalog'])['items'];catalog=resolve(p['source_catalog'])['items']
    for sid in ids:
        item=catalog[sid];original=canonical[sid];row=item['row'];v=item['content'][0];params=row['source_parameters']
        require(item==rebase(original,p['workspace_root']),'Source differs from accepted classic dataset')
        require(item['task']==row['task']=='T2' and row['accuracy_eligible']
                and row['choices']==['LEFT','RIGHT'],'Wrong scientific task')
        # These retained digests identify the ORIGINAL full QA records, not
        # the deliberately reduced runtime row/ready dictionaries.
        public=item['public_index']
        require(all(item[k]==public[k] for k in ('source_row_sha256','source_ready_sha256','source_audit_identity'))
                and item['ready']['video_sha256']==public['video_sha256']==v['sha256']
                and item['ready']['stream_sha256']==public['stream_sha256']==item['source_stream_sha256']
                and params==public['nominal_parameters'] and row['truth']==public['truth']
                and row['prompt']==public['prompt'],'Source identity differs')
        require(params['offset_frame']-params['onset_frame']==30 and params['initial_dot_Y']==.5
                and params['background_Y'] in (.05,.2,.4) and params['delta_Y'] in plan['deltas'],'Wrong classic condition')
        require(row['truth']==params['target_side'] and row['prompt']==plan['prompt']
                and item['content'][1]=={'type':'text','text':plan['prompt']},'GT/prompt differs')
        require(item['video_contract']=={'fps':60,'frames':180,'height':512,'width':768},'Video contract differs')


def rebase(value,remote):
    if isinstance(value,str):
        for old in ROOTS.values():
            if value.startswith(old+'/'):return remote+value[len(old):]
        return value
    if isinstance(value,list):return [rebase(v,remote) for v in value]
    if isinstance(value,dict):return {k:rebase(v,remote) for k,v in value.items()}
    return value

class Boundary(lifecycle.Boundary):pass

def seal(draft,output):
    p=read(draft);science(p);p['operational_wrapper']=descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:json.dump(p,f);temp=f.name
    try:return qsa.seal(temp,output)
    finally:Path(temp).unlink()

def run(profile):
    p=read(profile);science(p)
    for key,path in [('operational_wrapper',__file__),('qsa_entry',qsa.__file__),('observer_entry',qsa.v2.__file__),('selector_source',qsa.selector.__file__)]:
        require(file_sha256(path)==p[key]['sha256'],'Frozen execution source changed: '+key)
    base.worker_identity=qualified.worker_identity;base.Boundary=Boundary
    return qsa.v2.run(profile)

def supervise(profile):
    from . import refinement_worker as worker
    native=worker.subprocess.Popen
    def popen(argv,**kwargs):return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv],**kwargs)
    worker.subprocess.Popen=popen
    try:return worker.supervise(profile)
    finally:worker.subprocess.Popen=native

def closed(profile):
    p,result,raw=lifecycle.closed(profile)
    require(result['completed_ids']==p['assigned_ids'][:len(raw)],'Not a clean ordered prefix')
    if raw:
        from .official_x_results import Archive,audit_worker
        audit_worker(Archive(p['workspace_root']),{'profile':descriptor(profile),'phase':'classic_T2',
            'variant':'X_QSA_NO_FLASHINFER_AUTOTUNE','closed_prefix_only':result['status']=='PAUSED'})
    return p,result,raw

def activate(profile):
    p=read(profile);science(p);c=read(p['control_file'])
    require(c['generation']==p['control_generation'] and not Path(p['output']).exists()
            and not Path(p['output']).with_suffix('.launch.json').exists(),'Attempted/stale worker')
    require(not any((Path(p['claim_root'])/(s+'.json')).exists() for s in p['assigned_ids']),'Already claimed')
    c.update(paused=False,activated_profile=descriptor(profile));atomic_json(p['control_file'],c)
    return {'status':'ACTIVATED_NOT_LAUNCHED','profile':descriptor(profile)}

def resume(profile,output):
    p,result,raw=closed(profile);science(p);c=read(p['control_file']);require(c['paused'],'Explicit pause required')
    ids=lifecycle.remaining(p,result,raw);require(ids,'No unattempted inputs remain')
    q=copy.deepcopy(p);generation=c['generation']+1;worker=p['card']+'-segment'+str(generation)
    for key in ('implementation','worker_token','media_files','operational_wrapper'):q.pop(key,None)
    q.update(worker=worker,assigned_ids=ids,maximum_model_calls=len(ids),control_generation=generation,
        output=str(Path(p['output']).parent/worker),pause_file=str(Path(output).with_suffix('.PAUSE.json')),
        continuation_parent=descriptor(profile),continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')))
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:json.dump(q,f);temp=f.name
    try:r=seal(temp,output)
    finally:Path(temp).unlink()
    c.update(generation=generation,paused=True,continuation_profile=descriptor(output));atomic_json(p['control_file'],c)
    return {**r,'status':'SEALED_PAUSED_NOT_LAUNCHED'}

def prepare(root):
    """Bind the already validated classic source union and existing qualification; CPU only."""
    from collections import defaultdict,Counter
    from .official_x_results import Archive
    root=Path(root).resolve();plans=root/'plans'/NAME;require(not plans.exists(),'Package already exists')
    arc=Archive(root);allocation=read(root/'plans/nemo-x-full-20261029/full-allocation.json')
    require(allocation['task_counts']=={'T1':3840,'T2':2880,'T3':2736},'Accepted classic scope differs')
    items={};bindings=[]
    for card in ('fourier0','radon0'):
        ref=allocation['cards'][card]['source_catalog'];old=arc.ref(ref);bindings.append(ref)
        for sid,item in old['items'].items():
            if item['task']=='T2':
                require(sid not in items,'Duplicate canonical ID');items[sid]=item
    require(len(items)==2880,'Classic T2 count differs')
    cells=defaultdict(dict);prompts=set()
    for sid,item in items.items():
        q=item['row']['source_parameters'];key=(q['background_Y'],q['delta_Y']);sub=(q['onset_index'],q['target_side'])
        require(sub not in cells[key] and q['offset_frame']-q['onset_frame']==30,'Duplicate condition or short exposure')
        cells[key][sub]=sid;prompts.add(item['row']['prompt'])
    require(len(cells)==60 and all(len(v)==48 for v in cells.values()),'Incomplete condition grid')
    require(len(prompts)==1,'Multiple prompts');prompt=prompts.pop()
    require(prompt==read(root/'plans/natural-t2-qwen-x-20261029/plan.json')['prompt'],'Classic/natural prompt differs')
    assigned={c:[] for c in ('fourier0','fourier1','radon0')}
    for ci,(cell,rows) in enumerate(sorted(cells.items())):
        onsets=[(j+7*ci)%24 for j in range(24)];n=8 if ci%3==0 else 9
        for card,selected in zip(assigned,(onsets[:n],onsets[n:2*n],onsets[2*n:])):
            assigned[card]+=[rows[i,side] for i in selected for side in ('LEFT','RIGHT')]
    require([len(x) for x in assigned.values()]==[1040,1040,800],'Allocation differs')
    require(len({s for x in assigned.values() for s in x})==2880,'Allocation union differs')
    plans.mkdir(parents=True);write_new(plans/'canonical-sources.json',{'items':items,'accepted_catalogs':bindings})
    qualification=root/'runs/reproducibility/qwen-x-crosscard-t23-20261029/analysis/summary.json'
    write_new(plans/'qualification-summary.json',read(qualification))
    cards={}
    for host,remote in ROOTS.items():
        selected={sid:rebase(items[sid],remote) for card,ids in assigned.items() if (card=='radon0')==(host=='radon') for sid in ids}
        path=plans/(host+'-sources.json');write_new(path,{'items':selected,'schema':'classic-T2-Qwen-X-sources-v1'})
        for card,ids in assigned.items():
            if (card=='radon0')==(host=='radon'):
                cards[card]={'assigned_ids':ids,'count':len(ids),'source_catalog':qualified.remote_ref(path,root,remote)}
        (plans/(host+'-video-files.txt')).write_text(''.join(str(Path(v['content'][0]['path']).relative_to(remote))+'\n' for v in selected.values()))
    plan={'schema':'classic-T2-Qwen-X-execution-v1','status':'PREPARED_NOT_LAUNCHED','maximum_new_model_calls':2880,
          'cards':cards,'prompt':prompt,'deltas':sorted({k[1] for k in cells}),
          'canonical_catalog_sha256':file_sha256(plans/'canonical-sources.json'),
          'qualification_sha256':file_sha256(plans/'qualification-summary.json'),
          'source_bindings':bindings,'automatic_retry':False,'no_automatic_successor':True,'selection_uses_model_answers':False,
          'allocation_rule':'Whole LR/onset pairs; cell-index rotation. 20 cells split8/8/8 pairs;40 split9/9/6. No answer-based assignment.',
          'qualification_reused_calls':432,'new_qualification_calls':0,'qualified_outputs_in_full_curve':False}
    write_new(plans/'plan.json',plan)
    write_new(plans/'authorization.json',{'status':'AUTHORIZED','maximum_new_model_calls':2880,
        'experiment_plan_sha256':file_sha256(plans/'plan.json'),
        'user_message':'我可能搞错了？我们有的是nemo和cosmos是吧？如果没跑的话，把qwen x协议在t2的经典上验证扫一下，跑之前好好核实一下哦！就是必须达到我要的那种程度，你懂的，官方load，最小化我们自己加东西等等，和A的区别load autoyune之类的',
        'scope':'Classic T2 only:3 backgrounds x20 deltas x24 onsets xLR;500ms;no new qualification or automatic retries.'})
    for card in assigned:
        host='radon' if card=='radon0' else 'fourier';remote=ROOTS[host]
        original=root/'plans/qwen-x-crosscard-t23-20261029'/(card+'-t23-cold1-profile.json');p=read(original)
        for key in ('implementation','worker_token','media_files','operational_wrapper','panel','prepared_plan_sha256','cold'):p.pop(key,None)
        worker=card+'-segment0';control=plans/(card+'-control.json');write_new(control,{'generation':0,'paused':True,'status':'PREPARED_NOT_LAUNCHED'})
        p.update(worker=worker,phase='classic_T2',resource_group='T23',assigned_ids=assigned[card],maximum_model_calls=len(assigned[card]),
            source_catalog=cards[card]['source_catalog'],qualified_profile=qualified.remote_ref(original,root,remote),
            canonical_catalog=qualified.remote_ref(plans/'canonical-sources.json',root,remote),
            qualification_summary=qualified.remote_ref(plans/'qualification-summary.json',root,remote),
            experiment_plan=qualified.remote_ref(plans/'plan.json',root,remote),execution_authorization=qualified.remote_ref(plans/'authorization.json',root,remote),
            output=str(Path(remote)/'runs/reproducibility'/NAME/worker),claim_root=str(Path(remote)/'runs/reproducibility'/NAME/'claims'/card),
            pause_file=str(Path(remote)/'plans'/NAME/(worker+'-PAUSE.json')),control_file=str(Path(remote)/'plans'/NAME/control.name),
            control_generation=0,automatic_retry=False,formal_auto_resume=False)
        science(p,arc.ref);write_new(plans/(worker+'-draft.json'),p)
    return {'status':'PREPARED_NOT_LAUNCHED','maximum_new_model_calls':2880,'cards':{c:v['count'] for c,v in cards.items()},'new_qualification_calls':0}

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','seal','activate','run','supervise','pause','prepare-resume','audit-closed'])
    for key in ('root','draft','output','profile','control'):parser.add_argument('--'+key)
    a=parser.parse_args()
    if a.command=='prepare':r=prepare(a.root)
    elif a.command=='seal':r=seal(a.draft,a.output)
    elif a.command=='activate':r=activate(a.profile)
    elif a.command=='run':r=run(a.profile)
    elif a.command=='supervise':r=supervise(a.profile)
    elif a.command=='pause':r=lifecycle.pause(a.control)
    elif a.command=='prepare-resume':r=resume(a.profile,a.output)
    else:
        p,result,raw=closed(a.profile);science(p);r={'status':'CLOSED_PASS','worker_status':result['status'],'completed':len(raw)}
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
