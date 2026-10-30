"""Bound natural-T2 allocation on the unchanged qualified Qwen X runner.

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
from .natural_t2_stimuli import NAME

qsa=qualified.qsa
base=qsa.base

def science(p,resolve=verified):
    qualified.fixed_implementation()
    template=resolve(p['qualified_profile'])
    for key in ('model','base_model','checkpoint','checkpoint_assets','engine_kwargs','expected_packages',
                'cleared_environment','environment','request_mm_processor_kwargs','vision_observation',
                'minimum_kv_capacity','torch_num_threads','gpu_uuid','physical_lock','python'):
        require(p[key]==template[key],'Qualified Qwen X setting changed: '+key)
    require(not p.get('bootstrap') and not p.get('request_adapter'),'Unapproved inference adaptation')
    require(p['phase']=='natural_T2' and not p['automatic_retry'] and not p['formal_auto_resume'],'Wrong phase/retry policy')
    plan=resolve(p['experiment_plan']);auth=resolve(p['execution_authorization'])
    require(plan['schema']=='natural-T2-Qwen-X-execution-v1' and plan['maximum_new_model_calls']==3840,
            'Wrong finite natural experiment')
    require(auth['status']=='AUTHORIZED' and auth['experiment_plan_sha256']==p['experiment_plan']['sha256']
            and auth['maximum_new_model_calls']==3840 and auth['user_message'],'Missing bound user authorization')
    full=plan['cards'][p['card']]['assigned_ids'];ids=p['assigned_ids']
    require(p['source_catalog']==plan['cards'][p['card']]['source_catalog'],'Catalog not bound to authorization')
    require(len(ids)==len(set(ids))==p['maximum_model_calls'] and set(ids)<=set(full),'Unallocated/repeated IDs')
    if p.get('continuation_parent'):
        parent=resolve(p['continuation_parent'])
        require(p['control_generation']==parent['control_generation']+1,'Resume generation differs')
        if resolve is verified:
            parent,_,raw=closed(p['continuation_parent']['path'])
            require(ids==[s for s in parent['assigned_ids'] if s not in raw],'Resume must contain only never-claimed IDs')
    else:require(ids==full and p['control_generation']==0,'Initial allocation changed')
    catalog=resolve(p['source_catalog'])
    for sid in ids:
        item=catalog['items'][sid];row=item['row'];v=item['content'][0];qa=item['natural_qa']
        require(item['task']==row['task']=='T2' and row['accuracy_eligible'] and row['choices']==['LEFT','RIGHT'],'Wrong task')
        require(digest(row)==item['source_row_sha256'] and item['source_audit_identity']==digest(qa),'Source binding differs')
        require(v['sha256']==qa['video_sha256'] and row['truth']==qa['truth']
                and qa['decoded_frames_exact']==180 and qa['outside_target_identical'],'Unqualified stimulus')
        require(qa['sample_id']==sid and qa['render_implementation_sha256'] in plan['accepted_stimulus_implementation_sha256']
                and qa['stimulus_recipe_sha256']==plan['stimulus_recipe_sha256'],'Stimulus version changed')
        require(abs(qa['decoded_delta_Y']-qa['delta_Y'])<min(.002,.02*qa['delta_Y'])
                and abs(qa['float64_delta_Y']-qa['delta_Y'])<1e-10
                and qa['offset_frame']-qa['onset_frame']==30,'Event differs')
        require(item['video_contract']=={'fps':60,'frames':180,'height':512,'width':768},'Video contract differs')
        require(row['prompt']==plan['prompt'] and item['content'][1]=={'type':'text','text':plan['prompt']},'Prompt differs')

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
    """Create finite drafts only after all lossless source QA has completed."""
    from collections import Counter
    from .natural_t2_results import source_audit
    root=Path(root).resolve();plans=root/'plans'/NAME
    qa_summary,qa=source_audit(root);write_new(plans/'source-acceptance.json',qa_summary)
    allocation=read(plans/'allocation.json');criteria=read(plans/'allocation-criteria.json')
    template_dir=root/'plans/qwen-x-crosscard-t23-20261029';cards={};prompts=set()
    roots={'fourier':'/home/yang/cvpr2027_benchmark','radon':'/mnt/sda/q/cvpr2027_benchmark'}
    from .official_x_results import Archive
    arc=Archive(root)
    for host,remote in roots.items():
        tp=read(template_dir/('radon0-t23-cold1-profile.json' if host=='radon' else 'fourier0-t23-cold1-profile.json'))
        old=arc.ref(tp['source_catalog'])['items']
        prompt=next(i['row']['prompt'] for i in old.values() if i['task']=='T2');prompts.add(prompt)
        items={}
        for sid,q in qa.items():
            if (q['card']=='radon0')!=(host=='radon'):continue
            row={'sample_id':sid,'task':'T2','truth':q['truth'],'choices':['LEFT','RIGHT'],
                 'accuracy_eligible':True,'prompt':prompt,'source_parameters':{k:q[k] for k in
                 ('block','group','background_id','object_id','delta_Y','onset_frame','offset_frame','truth','card')},
                 'source_origin':'natural-T2-randomized-balanced-transfer'}
            ready={k:q[k] for k in ('video_sha256','stream_sha256','baseline_rgb8_sha256','event_rgb8_sha256')}
            items[sid]={'sample_id':sid,'task':'T2','row':row,'natural_qa':q,'ready':ready,
                'content':[{'type':'video','path':str(Path(remote)/'datasets'/NAME/'videos'/(sid+'.mp4')),'sha256':q['video_sha256']},
                           {'type':'text','text':prompt}],
                'source_row_sha256':digest(row),'source_ready_sha256':digest(ready),
                'source_stream_sha256':q['stream_sha256'],'source_audit_identity':digest(q),
                'video_contract':{'fps':60,'frames':180,'height':512,'width':768}}
        write_new(plans/(host+'-sources.json'),{'schema':'natural-T2-source-catalog-v1','items':items})
    require(len(prompts)==1,'Qualified prompts differ');prompt=prompts.pop()
    for card in ('fourier0','fourier1','radon0'):
        host='radon' if card=='radon0' else 'fourier';remote=roots[host]
        ids=[s for s,q in qa.items() if q['card']==card]
        require(len(ids)==(1088 if card=='radon0' else 1376),'Card count differs')
        cards[card]={'assigned_ids':ids,'count':len(ids),
                    'source_catalog':qualified.remote_ref(plans/(host+'-sources.json'),root,remote)}
    plan={'schema':'natural-T2-Qwen-X-execution-v1','status':'PREPARED_NOT_LAUNCHED',
          'maximum_new_model_calls':3840,'cards':cards,'prompt':prompt,
          'stimulus_recipe_sha256':file_sha256(plans/'allocation-criteria.json'),
          'accepted_stimulus_implementation_sha256':read(plans/'source-quality-policy.json')['accepted_generator_sha256'],
          'source_quality_policy':descriptor(plans/'source-quality-policy.json'),
          'source_acceptance':descriptor(plans/'source-acceptance.json'),
          'allocation':descriptor(plans/'allocation.json'),'automatic_retry':False,
          'no_automatic_successor':True,'selection_uses_model_answers':False,
          'interpretation':'Four balanced BG x OBJ source combinations; absolute mean linear-RGB Y increments; no attribution to background alone.'}
    write_new(plans/'plan.json',plan)
    auth={'status':'AUTHORIZED','maximum_new_model_calls':3840,'experiment_plan_sha256':file_sha256(plans/'plan.json'),
          'user_message':'先做t2；真实/生成各维持50个obj，可替换工业或world-model常见物；四种混搭数量一致；三张卡跑Qwen，抽查起始和关键帧。',
          'scope':'20 deltas x 24 onsets x LR x four BG/OBJ source groups; first GPU admission included; no extra inference/retry.'}
    write_new(plans/'authorization.json',auth)
    for card in cards:
        host='radon' if card=='radon0' else 'fourier';remote=roots[host]
        original=template_dir/(card+'-t23-cold1-profile.json');p=read(original);worker=card+'-segment0'
        for key in ('implementation','worker_token','media_files','operational_wrapper','panel','prepared_plan_sha256','cold'):
            p.pop(key,None)
        control=plans/(card+'-control.json');write_new(control,{'generation':0,'paused':True,'status':'PREPARED_NOT_LAUNCHED'})
        p.update(worker=worker,phase='natural_T2',resource_group='natural_T2',assigned_ids=cards[card]['assigned_ids'],
            maximum_model_calls=cards[card]['count'],source_catalog=cards[card]['source_catalog'],
            qualified_profile=qualified.remote_ref(original,root,remote),
            experiment_plan=qualified.remote_ref(plans/'plan.json',root,remote),
            execution_authorization=qualified.remote_ref(plans/'authorization.json',root,remote),
            output=str(Path(remote)/'runs/reproducibility'/NAME/worker),
            claim_root=str(Path(remote)/'runs/reproducibility'/NAME/'claims'/card),
            pause_file=str(Path(remote)/'plans'/NAME/(worker+'-PAUSE.json')),
            control_file=str(Path(remote)/'plans'/NAME/control.name),control_generation=0,
            automatic_retry=False,formal_auto_resume=False)
        science(p,arc.ref);write_new(plans/(worker+'-draft.json'),p)
        (plans/(card+'-video-files.txt')).write_text(''.join(s+'.mp4\n' for s in p['assigned_ids']))
    return {'status':'PREPARED_NOT_LAUNCHED','maximum_new_model_calls':3840,'cards':{c:v['count'] for c,v in cards.items()}}

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
