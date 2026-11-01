"""CPU contracts: source refusal, finite constructors and no-repeat pause/resume."""
import copy
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from classic_precision import natural_x_operations as op
from classic_precision import natural_x_runner as runner


def put(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + '\n'); return op.descriptor(path)


def journal(path, values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, sort_keys=True) + '\n' for x in values))


def owner(pid=111):
    return {'pid': pid, 'host': socket.gethostname(), 'boot_id': 'cpu-fixture-boot', 'process_start_ticks': '20'}


@pytest.fixture
def study(tmp_path, monkeypatch):
    # Production10416 remains a hard constant; a three-source fixture tests the
    # actual contracts without generating media or running any model.
    monkeypatch.setattr(runner, 'COUNTS', {'T1': 1, 'T2': 1, 'T3': 1})
    root = tmp_path; ns = 'natural-x-cpu-test'; plans = root/'plans'/ns; run = root/'runs/reproducibility'/ns
    plans.mkdir(parents=True);run.mkdir(parents=True);(root/'code/vlm-psychophysics').mkdir(parents=True)
    lock = root/'physical.lock';lock.write_bytes(b'');st = lock.stat()
    template = {'model': 'nemo', 'base_model': 'nemo', 'python': '/fixture/gpu/bin/python', 'hostname': socket.gethostname(),
        'workspace_root': str(root), 'checkpoint': str(root/'official-mixed-NVFP4'), 'checkpoint_assets': {},
        'engine_kwargs': {'max_model_len': 65536, 'kv_cache_memory_bytes': 2*1024**3, 'quantization': 'modelopt'},
        'expected_packages': {}, 'environment': {'PYTHONPATH': str(root/'code/vlm-psychophysics')}, 'cleared_environment': [],
        'gpu_uuid': 'fixture-physical-card', 'card': 'fourier0',
        'physical_lock': {'path': str(lock), 'st_dev': st.st_dev, 'st_ino': st.st_ino}}
    tref = put(plans/'qualified-template.json', template);items = {};rows = []
    for n, task in enumerate(('T1','T2','T3'), 1):
        sid = 'natural-source-'+str(n);media = root/'media'/task/(sid+'.mp4');media.parent.mkdir(parents=True)
        media.write_bytes(('CPU CONTRACT NOT A REAL VIDEO '+sid).encode());sha = op.file_sha256(media)
        items[sid] = {'sample_id': sid, 'task': task, 'row': {'sample_id': sid, 'task': task,
            'truth': 'LEFT', 'prompt': 'Which target?', 'choices': ['LEFT','RIGHT'], 'accuracy_eligible': True},
            'content': [{'type': 'video', 'path': str(media), 'sha256': sha}, {'type': 'text', 'text': 'Which target?'}],
            'video_contract': {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}}
        rows.append({'dataset_revision': 'cpu-source-v1', 'domain': 'natural', 'task': task,
            'canonical_id': sid, 'legacy_sample_id': sid, 'path': task+'/'+sid+'.mp4', 'legacy_path': 'videos/'+sid+'.mp4', 'video_sha256': sha})
    cat = put(plans/'catalog.json', {'items': items});reg = put(plans/'registry.json', {'rows': rows})
    approval = {'schema': 'natural-x-source-approval-v1', 'status': 'ROOT_APPROVED_NATURAL_X_SOURCE_VERSION',
        'source_version': 'cpu-source-v1', 'dataset_revision': 'cpu-source-v1', 'counts': runner.COUNTS,
        'model_order': list(runner.ORDER), 'registry': reg, 'catalogs': [cat], 'run_namespace': ns,
        'templates': {model: [tref] for model in runner.ORDER}, 'reuse': {model: [] for model in runner.ORDER}}
    aref = put(plans/'approval.json', approval);ids = list(items)
    reused = put(plans/'reuse.json', {'status': 'PASS_EXACT_CURRENT_SOURCE_REUSE_CPU', 'model': 'nemo',
        'source_approval': aref, 'catalog': cat, 'accepted_ids': [], 'remaining_ids': ids})
    allocation = put(plans/'allocation.json', {'schema': 'natural-x-approved-allocation-v1', 'status': 'ROOT_APPROVED_EXACT_REMAINING',
        'model': 'nemo', 'source_approval': aref, 'catalog': cat, 'accepted_reuse_ids': [], 'remaining_ids': ids,
        'shards': {'fourier0': ids}, 'reuse_receipt': reused})
    p = runner.draft(tref, aref, cat, allocation, 'fourier0', 'worker0', run/'worker0', run/'claims/worker0', plans/'worker0-control.json', plans/'worker0.pause')
    p['worker_token'] = 'cpu-fixture-token';pref = put(plans/'worker0-profile.json', p)
    put(p['control_file'], {'generation': 0, 'paused': True})
    fence = plans/'fence.json';put(fence, {'generation': 0, 'paused': False})
    predecessor = put(plans/'prior-model.json', {'status': 'COMPLETE_PASS', 'model': 'qwen', 'unique_videos': 10416, 'source_approval': aref})
    q = {'schema': 'natural-x-finite-queue-v1', 'status': 'SEALED', 'automatic_retry': False, 'model': 'nemo',
        'workspace_root': str(root), 'cpu_python': sys.executable, 'source_approval': aref, 'profiles': [pref],
        'operator': op.descriptor(op.__file__),
        'output': str(run/'queue0'), 'state_path': str(run/'queue0/state.json'),
        'controller_lock': str(plans/'queue0.controller.lock'), 'fence': str(fence), 'fence_generation': 0,
        'dispatch_lock': str(plans/'dispatch.lock'), 'predecessors': [{'receipt': predecessor, 'required': {'status': 'COMPLETE_PASS'}}]}
    qp = plans/'queue0.json';qref = put(qp,q)
    rs = plans/'run-set.json';put(rs, {'schema': 'natural-x-run-set-v1', 'queues': [qref], 'source_approval': aref,
        'dispatch_lock': q['dispatch_lock'], 'fence': str(fence)})
    monkeypatch.setattr(op, 'live_owner', lambda value: {'alive': False})
    # Native/GPU audits themselves are unchanged existing functions. This test
    # supplies their CPU result and exercises all new ledger/source boundaries.
    def audited(arc, spec):
        profile = arc.ref(spec['profile']);raws = op.journal(arc.path(profile['output'])/'raw.jsonl')
        return [{'sample_id': r['sample_id']} for r in raws], {'runtime': [{'CPU_fixture_only': True}]}
    monkeypatch.setattr(op.evidence, 'audit_worker', audited)
    monkeypatch.setattr(runner, 'validate_runtime', lambda *args: None)
    return SimpleNamespace(root=root,plans=plans,run=run,approval=approval,aref=aref,cat=cat,template=tref,
        allocation=allocation,p=p,pref=pref,q=q,qp=qp,qref=qref,rs=rs,items=items)


def finished(p, completed=1, status='PAUSED'):
    out = Path(p['output']);out.mkdir(parents=True,exist_ok=True);ids = p['assigned_ids'][:completed];raws=[];events=[]
    catalog = op.verified(p['source_catalog'])['items']
    for ordinal,sid in enumerate(ids):
        item=catalog[sid];raw={'sample_id':sid,'ordinal':ordinal,'profile_sha256':op.digest(p),
            'source':{'media':[item['content'][0]]},'truth':item['row']['truth'],'question':item['row']['prompt'],
            'raw_answer':'LEFT','finish_reason':'stop','stop_reason':1,'output_token_ids':[1],'actual_prompt_token_ids':[2,3]}
        raws.append(raw);events += [{'event':'INTENT','sample_id':sid},{'event':'TECHNICAL_PASS','sample_id':sid,'raw_digest':op.digest(raw)}]
        put(Path(p['claim_root'])/(sid+'.json'), {'sample_id':sid,'profile_sha256':op.digest(p),'worker_token':p['worker_token']})
    journal(out/'raw.jsonl',raws);journal(out/'events.jsonl',events)
    put(out/'result.json',{'status':status,'completed_ids':ids,'unattempted_ids':p['assigned_ids'][completed:],'UNCERTAIN_ids':[]})
    put(out.with_suffix('.wait.json'),{'returncode':0,'real_wait_observed':True,'owned_engines_exited':True,
        'original_physical_lease_released':True,'profile_sha256':op.digest(p),'terminal_status':status,'completed_ids':ids})
    put(out.with_suffix('.launch.json'),{'owner':owner(222),'pid':333,'profile_sha256':op.digest(p)})
    return raws


def close_queue(s, status='PAUSED'):
    ps=op.paths(s.qp,s.q);qref=op.descriptor(s.qp)
    put(ps['launch-intent'],{'queue':qref});put(ps['launch'],{'queue':qref,'pid':111,'boot_id':owner()['boot_id'],'process_start_ticks':'20'})
    put(ps['owner'],owner());put(ps['wait'],{'returncode':0,'real_wait_observed':True,'queue':qref,'child_pid':222,'owner':owner()})
    workers=[]
    for ref in s.q['profiles']:
        profile=op.verified(ref);out=Path(profile['output'])
        if out.exists():workers.append({'profile':ref,'returncode':0,'real_supervisor_wait':True,
                                       'terminal_status':op.read(out/'result.json')['status']})
    put(s.q['state_path'],{'status':status,'owner':owner(222),'plan':qref,'workers':workers})


def test_production_scope_and_no_extra_protocol_panel():
    assert runner.COUNTS == {'T1':3840,'T2':3840,'T3':2736}
    assert sum(runner.COUNTS.values()) == 10416 and runner.ORDER == ('cosmos','qwen','nemo','gemma')


@pytest.mark.parametrize('change',['pending','changed_sha','KV','hostname','input_dependencies','classic_control','allocation','parent_alias'])
def test_unapproved_or_changed_science_never_executes(study,change):
    s=study;p=copy.deepcopy(s.p)
    if change=='pending':
        a=copy.deepcopy(s.approval);a['status']='WAITING_SOURCE_REVIEW';p['natural_x']['source_approval']=put(s.plans/'unapproved.json',a)
    if change=='changed_sha':p['source_catalog']={**s.cat,'sha256':'0'*64}
    if change=='KV':p['engine_kwargs']['kv_cache_memory_bytes'] += 1
    if change=='hostname':p['hostname']='wrong-physical-host'
    if change=='input_dependencies':p['input_dependencies']=[{'module':'changed','source_sha256':'wrong'}]
    if change=='classic_control':p['control_file']=str(s.root/'plans/gemma-x-provisional-20261031/control.json')
    if change=='allocation':p['assigned_ids']=['unapproved-new-id'];p['maximum_model_calls']=1
    if change=='parent_alias':
        catalog=op.read(s.cat['path']);first=next(iter(catalog['items']));catalog['items'][first]['row']['sample_id']='historical-spoon-parent'
        p['source_catalog']=put(s.plans/'wrong-parent-catalog.json',catalog)
    with pytest.raises((ValueError,KeyError)):runner.science(p)


def test_pause_fences_all_successors_and_finishes_current_whole_request(study):
    s=study;original_profile=Path(s.pref['path']).read_bytes()
    put(s.p['control_file'],{'generation':0,'paused':False})
    answer=finished(s.p,1);result=op.pause(s.rs,'Switch to T4/T5 later')
    assert result['signals_sent']==0 and op.read(s.q['fence'])['paused']
    assert op.read(s.p['control_file'])=={'generation':1,'paused':True,'reason':'Switch to T4/T5 later'}
    assert runner.Boundary(s.p).check() is True
    assert op.journal(Path(s.p['output'])/'raw.jsonl')==answer
    assert Path(s.pref['path']).read_bytes()==original_profile
    assert op.pause(s.rs,'same request')['status']=='PAUSE_ALREADY_REQUESTED'
    with pytest.raises(ValueError,match='fenced'):op.gates(s.q)


def test_resume_keeps_exact_suffix_and_preserves_original_raw_wait(study):
    s=study;finished(s.p,1);close_queue(s);op.pause(s.rs,'pause')
    protected={str(p):p.read_bytes() for p in [Path(s.p['output'])/'raw.jsonl',Path(s.p['output']).with_suffix('.wait.json')]}
    result=op.resume(s.qp,s.plans/'resume1.json');new=result['draft_profiles'][0]
    assert result['uncertain_claims']==0 and result['accepted_ids']==s.p['assigned_ids'][:1]
    assert new['assigned_ids']==s.p['assigned_ids'][1:] and new['maximum_model_calls']==2
    assert new['claim_root']!=s.p['claim_root'] and new['output']!=s.p['output'] and 'worker_token' not in new
    assert all(Path(path).read_bytes()==value for path,value in protected.items())
    with pytest.raises(ValueError,match='already exists'):op.resume(s.qp,s.plans/'resume1.json')


@pytest.mark.parametrize('change',['unknown_claim','partial_raw','wrong_GT','worker_wait1','outer_wait1','owner_live','lease_inode','failure'])
def test_resume_refuses_every_uncertain_or_failed_boundary(study,monkeypatch,change):
    s=study;raw=finished(s.p,1);close_queue(s);op.pause(s.rs,'pause');out=Path(s.p['output'])
    if change=='unknown_claim':put(Path(s.p['claim_root'])/'unknown.json',{'unknown':True})
    if change=='partial_raw':(out/'raw.jsonl').write_bytes((out/'raw.jsonl').read_bytes()+b'{')
    if change=='wrong_GT':
        raw[0]['truth']='RIGHT';journal(out/'raw.jsonl',raw)
        events=op.journal(out/'events.jsonl');events[-1]['raw_digest']=op.digest(raw[0]);journal(out/'events.jsonl',events)
    if change=='worker_wait1':
        p=out.with_suffix('.wait.json');v=op.read(p);v['returncode']=1;put(p,v)
    if change=='outer_wait1':
        p=op.paths(s.qp,s.q)['wait'];v=op.read(p);v['returncode']=1;put(p,v)
    if change=='owner_live':monkeypatch.setattr(op,'live_owner',lambda owner:{'alive':True})
    if change=='lease_inode':s.p['physical_lock']['st_ino']+=1;put(s.pref['path'],s.p);s.q['profiles']=[op.descriptor(s.pref['path'])];put(s.qp,s.q);close_queue(s)
    if change=='failure':put(out/'failure.json',{'error':'genuine failure'})
    with pytest.raises((ValueError,BlockingIOError)):op.resume(s.qp,s.plans/'must-not-exist.json')
    assert not (s.plans/'must-not-exist.json').exists()


def test_two_pauses_reconcile_every_ancestor_without_reusing_claim_space(study):
    s=study;finished(s.p,1);close_queue(s);op.pause(s.rs,'first')
    first=op.resume(s.qp,s.plans/'resume1.json')['draft_profiles'][0]
    first['worker_token']='fresh-one';first_ref=put(s.plans/'child-profile.json',first)
    put(first['control_file'],{'generation':2,'paused':True});finished(first,1)
    ancestry=op.continuation(op.Archive(s.root),first)
    assert [row['sample_id'] for row in ancestry]==s.p['assigned_ids'][:1]
    q=copy.deepcopy(s.q);q['profiles']=[first_ref];q['output']=str(s.run/'queue1');q['state_path']=str(s.run/'queue1/state.json')
    qp=s.plans/'queue1.json';put(qp,q);child=SimpleNamespace(qp=qp,q=q);close_queue(child)
    result=op.resume(qp,s.plans/'resume2.json')
    assert result['accepted_ids']==s.p['assigned_ids'][:2]
    assert result['draft_profiles'][0]['assigned_ids']==s.p['assigned_ids'][2:]


def test_existing_reuse_requires_identical_full_raw_source_and_scientific_config(study):
    s=study;raw=finished(s.p,1);out=Path(s.p['output'])
    audit=put(s.plans/'old-audit.json',{'status':'EXISTING_PASS'})
    receipt={'schema':'natural-x-existing-raw-reuse-v1','status':'ROOT_ACCEPTED_ORIGINAL_RAW_AUDIT','model':'nemo',
        'original_audit':audit,'segments':[{'profile':s.pref,'raw':op.descriptor(out/'raw.jsonl'),
                                        'record_digests':{raw[0]['sample_id']:op.digest(raw[0])}}]}
    a=copy.deepcopy(s.approval);a['reuse']['nemo']=[put(s.plans/'old-reuse.json',receipt)];aref=put(s.plans/'approval-reuse.json',a)
    result=op.reusable(aref,s.cat,'nemo',s.root)
    assert result['accepted_ids']==s.p['assigned_ids'][:1] and len(result['remaining_ids'])==2
    receipt['segments'][0]['record_digests'][raw[0]['sample_id']]='0'*64
    a['reuse']['nemo']=[put(s.plans/'bad-reuse.json',receipt)];bad=put(s.plans/'bad-approval.json',a)
    with pytest.raises(ValueError,match='raw changed'):op.reusable(bad,s.cat,'nemo',s.root)


def test_real_finite_constructors_keep_CPU_parent_and_sealed_worker(study,monkeypatch):
    s=study;Path(s.q['output']).mkdir(parents=True);calls=[]
    class Child:
        pid=222
        def __init__(self,argv):self.argv=argv
        def wait(self):
            if 'supervise' in self.argv:finished(s.p,3,'COMPLETE')
            return 0
    def popen(argv,**kwargs):calls.append((argv,kwargs));return Child(argv)
    monkeypatch.setattr(op.subprocess,'Popen',popen);monkeypatch.setattr(op,'owner_identity',owner)
    from classic_precision import gemma_x_fourier_operations as resource
    monkeypatch.setattr(resource,'resource_preflight',lambda p:{'admitted':True})
    actual=Path
    monkeypatch.setattr(op,'Path',lambda *parts:SimpleNamespace(read_text=lambda:
        ('222 (worker) '+' '.join(['S']+['0']*18+['20'])) if str(parts[0])=='/proc' else owner()['boot_id'])
        if parts and (str(parts[0])=='/proc' or str(parts[0])=='/proc/sys/kernel/random/boot_id') else actual(*parts))
    op.launch(s.qp);op.queue_supervisor(s.qp);op.execute(s.qp)
    assert len(calls)==3
    assert calls[0][0][:4]==[s.q['cpu_python'],'-B','-m',op.MODULE]
    assert calls[1][0][:4]==[s.q['cpu_python'],'-B','-m',op.MODULE]
    assert calls[2][0][:4]==[s.p['python'],'-B','-m',op.WORKER]
    assert op.read(s.q['state_path'])['status']=='COMPLETE'


def test_seal_resume_unpause_and_no_branch_preserve_old_state(study,monkeypatch):
    s=study;finished(s.p,1);close_queue(s);op.pause(s.rs,'pause')
    receipt_path=s.plans/'resume.json';op.resume(s.qp,receipt_path)
    def cpu_seal(draft,target):
        p=op.read(draft);runner.science(p);p['worker_token']='new-segment-token';put(target,p)
    monkeypatch.setattr(runner,'seal',cpu_seal)
    qpath=s.plans/'next-queue.json';result=op.seal_resume(receipt_path,qpath)
    q=op.read(qpath);p=op.verified(q['profiles'][0])
    assert p['assigned_ids']==s.p['assigned_ids'][1:]
    assert op.read(p['control_file'])=={'generation':1,'paused':True}
    assert op.read(s.q['fence'])['paused'] and q['fence_generation']==1
    with pytest.raises((ValueError,FileExistsError)):op.seal_resume(receipt_path,s.plans/'duplicate-branch.json')
    newset=s.plans/'next-run-set.json';old=op.read(s.rs);put(newset,{**old,'queues':[result['queue']]})
    assert op.unpause(newset)['status'].startswith('NEW_GENERATION_READY')
    assert not op.read(s.q['fence'])['paused']
    # Old profiles remain fenced even after the new generation is enabled.
    with pytest.raises(ValueError,match='stale'):op.gates(s.q)
    assert op.read(Path(s.p['output']).with_suffix('.wait.json'))['returncode']==0


def test_canonical_allocation_owner_rejects_a_second_initial_branch(study):
    s=study;op.reserve_allocation(s.pref,s.p)
    with pytest.raises(ValueError,match='already dispatched'):op.reserve_allocation(s.pref,s.p)
    p=copy.deepcopy(s.p);p['worker']='different';p['output']=str(s.run/'different')
    pref=put(s.plans/'different.json',p)
    with pytest.raises(ValueError,match='another branch'):op.reserve_allocation(pref,p)


def test_profile_worker_constructor_is_the_original_true_wait_supervisor(study,monkeypatch):
    s=study;finished(s.p,3,'COMPLETE');calls=[]
    Path(s.p['output']).with_suffix('.wait.json').unlink()
    Path(s.p['output']).with_suffix('.launch.json').unlink()
    class Child:
        pid=444
        def wait(self):return 0
    monkeypatch.setattr(sys,'executable',s.p['python'])
    monkeypatch.setattr(op.subprocess,'Popen',lambda argv,**kwargs:(calls.append((argv,kwargs)) or Child()))
    result=runner.supervise(s.pref['path'])
    assert calls[0][0]==[s.p['python'],'-B','-m','classic_precision.natural_x_runner','run','--profile',s.pref['path']]
    assert result['real_wait_observed'] and result['returncode']==0
    assert result['owned_engines_exited'] and result['original_physical_lease_released']


def test_real_CLI_pause_status_resume_with_the_production10416_universe(study):
    """Real separate CPU processes; metadata fixtures, no media decoding/model."""
    import os
    import subprocess
    s=study;counts={'T1':3840,'T2':3840,'T3':2736};items={};registry=[]
    skeleton=next(iter(s.items.values()))
    for task,n in counts.items():
        for i in range(n):
            sid=f'cpu-only-{task}-{i:05d}';sha=hashlib.sha256(sid.encode()).hexdigest();item=copy.deepcopy(skeleton)
            item.update(sample_id=sid,task=task);item['row'].update(sample_id=sid,task=task);item['content'][0]['sha256']=sha
            items[sid]=item;registry.append({'domain':'natural','task':task,'canonical_id':sid,'legacy_sample_id':sid,
                'dataset_revision':'cpu-source-v1','path':task+'/'+sid+'.mp4','video_sha256':sha})
    cat=put(s.plans/'full-catalog.json',{'items':items});reg=put(s.plans/'full-registry.json',{'rows':registry})
    approval=copy.deepcopy(s.approval);approval.update(counts=counts,registry=reg,catalogs=[cat]);aref=put(s.plans/'full-approval.json',approval)
    ids=list(items);reuse=put(s.plans/'full-reuse.json',{'status':'PASS_EXACT_CURRENT_SOURCE_REUSE_CPU','model':'nemo',
        'source_approval':aref,'catalog':cat,'accepted_ids':[],'remaining_ids':ids})
    allocation=op.read(s.allocation['path']);allocation.update(source_approval=aref,catalog=cat,
        remaining_ids=ids,shards={'fourier0':ids},reuse_receipt=reuse)
    alloc=put(s.plans/'full-allocation.json',allocation)
    profile=copy.deepcopy(s.p);profile.update(source_catalog=cat,assigned_ids=ids,maximum_model_calls=10416)
    profile['natural_x'].update(source_approval=aref,allocation=alloc);pref=put(s.pref['path'],profile)
    q={**s.q,'source_approval':aref,'profiles':[pref]};put(s.qp,q)
    runset=op.read(s.rs);runset.update(source_approval=aref,queues=[op.descriptor(s.qp)]);put(s.rs,runset)
    worktree=Path(runner.__file__).resolve().parents[1];env={**os.environ,'PYTHONPATH':str(worktree)}
    def cli(*args,success=True):
        result=subprocess.run([sys.executable,'-B','-m',op.MODULE,*args],cwd=worktree,env=env,capture_output=True,text=True,timeout=20)
        assert (result.returncode==0)==success,result.stderr
        return json.loads(result.stdout) if success else result
    before=Path(pref['path']).read_bytes()
    assert cli('status','--queue',str(s.qp))['profiles'][0]['expected']==10416
    assert cli('pause','--run-set',str(s.rs))['signals_sent']==0
    receipt=s.plans/'real-cli-resume.json';assert cli('resume','--queue',str(s.qp),'--output',str(receipt))['status'].startswith('CPU_RECONCILED')
    assert op.read(receipt)['draft_profiles'][0]['assigned_ids']==ids
    cli('launch','--queue',str(s.qp),success=False)
    assert not op.paths(s.qp,q)['launch-intent'].exists()
    assert Path(pref['path']).read_bytes()==before


def test_gemma_requires_approved_natural_prompt_table_and_preserves_dependencies(study):
    s=study;template=op.read(s.template['path'])
    classic=put(s.plans/'classic-prompts.json',{'task_contracts':{task:{'question':'Classic-only question'} for task in runner.COUNTS}})
    natural=put(s.plans/'natural-prompts.json',{'task_contracts':{task:{'question':'Which target?'} for task in runner.COUNTS}})
    template.update(model='gemma_x_full',base_model='gemma',reference_prompt_catalog=classic,
        operational_wrapper={'path':'sealed-original.py','sha256':'original-full-wrapper'},
        full_dependencies={'native':'unchanged'},input_dependencies=[{'module':'original','source_sha256':'unchanged'}])
    template['engine_kwargs']['kv_cache_memory_bytes']=48*1024**3
    tref=put(s.plans/'gemma-template.json',template)
    approval=copy.deepcopy(s.approval);approval['templates']['gemma']=[tref];approval['prompt_catalogs']=[natural]
    aref=put(s.plans/'gemma-approval.json',approval)
    reuse=op.read(s.plans/'reuse.json');reuse.update(model='gemma',source_approval=aref)
    reuse_ref=put(s.plans/'gemma-reuse.json',reuse)
    allocation=op.read(s.allocation['path']);allocation.update(model='gemma',source_approval=aref,reuse_receipt=reuse_ref)
    allocation_ref=put(s.plans/'gemma-allocation.json',allocation)
    args=(tref,aref,s.cat,allocation_ref,'fourier0','gemma-natural',s.run/'gemma',s.run/'claims/gemma',s.plans/'gemma-control.json',s.plans/'gemma.pause')
    with pytest.raises(ValueError,match='Gemma task prompt differs'):runner.draft(*args)
    profile=runner.draft(*args,prompt_catalog=natural)
    assert profile['reference_prompt_catalog']==natural
    for key in ('operational_wrapper','full_dependencies','input_dependencies'):assert profile[key]==template[key]


def test_cross_host_model_gate_uses_same_exact_source_identity(study):
    s=study;approval,_,_=runner.admission(s.aref,s.cat)
    remote=put(s.plans/'remote-model-complete.json',{'status':'COMPLETE_PASS','model':'qwen','unique_videos':10416,
        'source_approval':{'path':'different-host-approved.json','sha256':'host-specific'},
        'source_version':approval['source_version'],'source_identity_sha256':approval['computed_source_identity_sha256']})
    q={**s.q,'predecessors':[{'receipt':remote,'required':{'status':'COMPLETE_PASS'}}]}
    op.gates(q)
    record=op.read(remote['path']);record['source_identity_sha256']='wrong-video-or-GT-prompt-version'
    q['predecessors']=[{'receipt':put(s.plans/'wrong-remote-model.json',record),'required':{'status':'COMPLETE_PASS'}}]
    with pytest.raises(ValueError,match='source version'):op.gates(q)


def test_template_without_card_needs_an_explicit_operation_label(study):
    s=study;template=op.read(s.template['path']);template.pop('card')
    tref=put(s.plans/'no-card-template.json',template)
    approval=copy.deepcopy(s.approval);approval['templates']['nemo']=[tref]
    aref=put(s.plans/'no-card-approval.json',approval)
    reuse=op.read(s.plans/'reuse.json');reuse['source_approval']=aref;reuse_ref=put(s.plans/'no-card-reuse.json',reuse)
    allocation=op.read(s.allocation['path']);allocation.update(source_approval=aref,reuse_receipt=reuse_ref)
    alloc=put(s.plans/'no-card-allocation.json',allocation)
    args=(tref,aref,s.cat,alloc,'fourier0','no-card-worker',s.run/'no-card-worker',s.run/'claims/no-card',s.plans/'no-card-control.json',s.plans/'no-card.pause')
    with pytest.raises(ValueError,match='no operation card label'):runner.draft(*args)
    p=runner.draft(*args,card='fourier0')
    assert p['card']=='fourier0' and p['hostname']==template['hostname']
    assert p['gpu_uuid']==template['gpu_uuid'] and p['physical_lock']==template['physical_lock']
