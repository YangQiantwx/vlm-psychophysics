"""Finite ST4 science, CPU admission, exact deployment reuse, and closed audit.

All execution remains in the frozen static image executor. This new namespace
cannot authorize probes/retries or change the inherited numerical deployment.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

from . import static_t4_strict_sources as source
from . import static_t4_strict_runner as runner
from . import static_x_admission as native
from . import static_x_operations as parent_ops
from . import official_x_runner as base
from . import t4_balanced_operations as video_ops
from .storage import digest, file_sha256

NAME = source.NAME
MODULE = 'classic_precision.static_t4_strict_operations'
OWNER = os.environ.get('PSYCHOPHYSICS_QUEUE_OWNER', 'unbound-current-queue')
MODELS = ('cosmos','qwen','nemo','gemma')
COMMON = ('st4_authorization','source_catalog','st4_protocol','st4_parent_index','st4_allocation_plan')
require, read, descriptor, verified, write_new = source.require, source.read, source.descriptor, source.verified, source.write_new
claims_gate = video_ops.claims_gate
records = native.strict_records
_CACHE = set()


def signature(p):
    value = copy.deepcopy(p)
    for key in ('schema','worker_token','media_files','static_cpu_budget','static_deep_ids','static_cpu_template_map',
                'st4_final_review','st4_reviewed_draft_signature'):
        value.pop(key,None)
    if value.get('st4_pre_gemma_intent'):
        value['st4_pre_gemma_intent'].pop('expected_non_gemma_profiles',None)
    return digest(value)


def draft_signature(p):
    return signature(p)


def final_review_codes():
    codes = parent_ops.final_review_codes()
    for ref in video_ops.dependency_refs():
        codes[Path(ref['path']).stem] = ref
    for name in ('t4_balanced_operations','t4_balanced_queue','static_t4_strict_sources','static_t4_strict_runner','static_t4_strict_operations','static_t4_strict_queue'):
        codes[name] = descriptor(Path(__file__).with_name(name+'.py'))
    return codes


def cpu_codes():
    return {name: descriptor(Path(__file__).with_name(name+'.py'))
            for name in ('static_t4_strict_sources','static_t4_strict_runner','static_t4_strict_operations','static_x_admission')}


def authorization(p):
    auth = native.cached_verified(p['st4_authorization'])
    require(auth['schema'] == 'static-t4-first-last-user-authorization-v1' and auth['owner_chat'] == OWNER
            and auth['experiment'] == NAME and auth['models'] == list(MODELS)
            and auth['domain'] == 'classic_strict_T4_only' and auth['model_call_cap_upper_bound'] == 3072
            and auth['extra_qualification_or_probe_calls'] == 0, 'Wrong finite user authorization')
    require(p['static_authorization'] == p['st4_authorization'] and p['static_protocol'] == p['st4_protocol']
            and p['automatic_retry'] is False and p['formal_auto_resume'] is False and p.get('static_cold') is None,
            'Retry/cold or mixed static/ST4 identity')
    terminal = native.cached_verified(auth['strict_video_terminal'])
    require(terminal['status']=='COMPLETE_PASS' and terminal['completed']==3072
            and len(terminal['worker_closures'])==8, 'Strict source video run is not closed')
    catalog = native.cached_verified(p['source_catalog']); items = catalog['items']
    allocation = native.cached_verified(p['st4_allocation_plan'])
    require(allocation['schema'] == 'static-t4-global-allocation-v1' and allocation['owner_chat'] == OWNER
            and all(allocation[k] == p[k] for k in ('st4_authorization','source_catalog','st4_protocol','st4_parent_index')),
            'Global allocation binding differs')
    rows = allocation['allocations']; require(len(rows) == 8, 'ST4 must use exact eight new workers')
    slots, outputs, claims = set(), set(), set()
    for row in rows:
        require(row['model'] in MODELS and row['card'] in ('fourier0','fourier1') and row['assigned_ids']
                and len(row['assigned_ids']) == len(set(row['assigned_ids'])), 'Invalid finite owner allocation')
        require(row['output'] not in outputs and row['claim_root'] not in claims, 'Shared output/claim namespace')
        outputs.add(row['output']); claims.add(row['claim_root'])
        for sid in row['assigned_ids']:
            require(sid in items and (row['model'],sid) not in slots, 'Unknown/duplicate global claim slot')
            require(all(alias['source_parameters']['layout_index'] % 2 == int(row['card'][-1])
                        for alias in items[sid]['original_video_aliases']), 'Paired layout split across cards')
            slots.add((row['model'],sid))
    require(slots == {(m,s) for m in MODELS for s in items} and len(slots) <= 3072
            and allocation['maximum_new_model_calls'] == len(slots), 'Global unique-call coverage/cap differs')
    own = [x for x in rows if x['allocation_id'] == p['st4_allocation_id']]
    require(len(own) == 1, 'Unknown ST4 allocation')
    own = own[0]
    require(own['model'] == p['st4_model'] == native.model_key(p)
            and all(own[k] == p[k] for k in ('card','assigned_ids','output','claim_root'))
            and p['maximum_model_calls'] == len(own['assigned_ids']), 'ST4 worker differs from finite owner')
    return auth,own


def inherited_card(p,parent):
    # Cosmos static parents identify the card only as static_card; their
    # missing card field is historical metadata, not a deployment change.
    require(p['card']==parent['static_card']
            and (parent.get('card') is None or p['card']==parent['card']),
            'Inherited static physical card assignment changed')


def configuration(p):
    parent = native.cached_verified(p['parent_validated_profile'])
    require(native.model_key(parent) == p['st4_model']
            and native.deployment_signature(parent) == native.deployment_signature(p),
            'ST4 changed a qualified static deployment setting')
    for field in ('checkpoint','checkpoint_assets','processor_checkpoint','python','engine_kwargs','environment',
                  'cleared_environment','expected_packages','vision_observation','bootstrap','minimum_kv_capacity',
                  'media_io_kwargs','gpu_uuid','physical_lock','hostname','resources','torch_num_threads',
                  'request_mm_processor_kwargs','request_mm_processor_kwargs_by_task','request_adapter',
                  'selector_source','parent_runtime','runtime_template'):
        require(p.get(field) == parent.get(field), 'Inherited static field changed: '+field)
    inherited_card(p,parent)
    require(p['engine_kwargs']['limit_mm_per_prompt'] == {'image':4,'video':0,'audio':0}
            and p['request_mm_processor_kwargs'] == native.native_kwargs(p)
            and p['choices_by_task'] == {'T4':['LEFT','RIGHT']}, 'Static2 route or answer contract changed')
    if p['st4_model'] == 'qwen':
        require(p['engine_kwargs']['max_model_len'] == p['minimum_kv_capacity'] == 65536
                and p['engine_kwargs']['kv_cache_memory_bytes'] == 4*2**30,
                'Qwen must inherit the already-qualified static T23 deployment unchanged')
    catalog = native.cached_verified(p['source_catalog']); spec = native.cached_verified(p['st4_protocol'])
    authorization(p)
    for sid in p['assigned_ids']:
        item = catalog['items'][sid]
        require(item['sample_id'] == item['case_id'] == sid, 'ST4 canonical identity differs')
        runner.check_item(item,spec)
    return catalog


def science(p):
    catalog = configuration(p); items = catalog['items']
    receipt = native.cached_verified(p['st4_source_admission'])
    manifest = native.cached_verified(catalog['manifest'])
    require(catalog['schema'] == 'static-t4-ready-catalog-v1' and catalog['selection'] == 'all'
            and catalog['status'] == receipt['status'] == 'PASS_ENDPOINT_EXPORT'
            and catalog['source_prepare_receipt'] == p['st4_source_admission']
            and receipt['manifest'] == catalog['manifest'] and receipt['selection'] == 'all'
            and len(items) == catalog['unique_inputs'] == receipt['count'] == manifest['unique_ordered_pairs']
            and set(items) == set(manifest['items']) and not catalog['GT_scoreability_conflicts']
            and catalog['protocol'] == native.cached_verified(p['st4_protocol'])
            and receipt['source_implementation_sha256'] == file_sha256(source.__file__), 'Complete endpoint source admission missing')
    key = (p['source_catalog']['sha256'],p['st4_source_admission']['sha256'])
    if key not in _CACHE:
        aliases=[]; hashes=set()
        for sid,item in items.items():
            require({k:item[k] for k in manifest['items'][sid]} == manifest['items'][sid]
                    and item['endpoint_export'] == receipt['cases'][sid], 'Export changed source/GT/alias metadata')
            require(item['model_input_sha256'] not in hashes, 'Ready catalog repeats a true model input')
            hashes.add(item['model_input_sha256']);aliases.extend(item['original_video_aliases'])
        require(len(aliases) == len({x['sample_id'] for x in aliases}) == 768
                and sum(x['accuracy_eligible'] for x in aliases) == 768
                and sum(not x['accuracy_eligible'] for x in aliases) == 0
                and catalog['source_alias_count'] == manifest['count'] == 768,
                'Scientific source aliases/scored/bias denominator differs')
        cells=Counter((x['source_parameters']['N'],x['source_parameters']['coherence']) for x in aliases)
        require(cells == Counter({(n,c/8):48 for n in (8,16) for c in range(1,9)}),
                'Complete N/coherence source cells differ')
        _CACHE.add(key)
    qualified_deployment(p)
    return catalog


def qualified_deployment(p, runtime=None):
    index = native.cached_verified(p['st4_parent_index']); rec = index['parents'][p['st4_allocation_id']]
    require(rec['profile'] == p['parent_validated_profile'] and rec['qualification_summary'] == p['qualification_summary'],
            'Wrong already-qualified parent binding')
    parent = native.cached_verified(rec['profile'])
    summary = parent_ops.qualification_summary(rec['qualification_summary'])
    matches=[]
    for ref in summary['profiles']:
        candidate = native.cached_verified(ref)
        if candidate['gpu_uuid'] == p['gpu_uuid'] and native.deployment_signature(candidate) == native.deployment_signature(p):
            matches.append(ref)
    require(len(matches)==1 and matches[0] == rec['qualification_profile'], 'No unique existing same-card/deployment qualification')
    wait,result = native.cached_verified(rec['completed_wait']),native.cached_verified(rec['completed_result'])
    require(wait['returncode']==0 and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released'] and wait['profile_sha256']==digest(parent)
            and result['status']=='COMPLETE' and result['completed_ids']==parent['assigned_ids']
            and not result.get('UNCERTAIN_ids'), 'Selected static parent not truly closed')
    expected=native.cached_verified(rec['runtime'])
    if runtime is not None:
        require([native.runtime_signature(x) for x in runtime] == [native.runtime_signature(x) for x in expected],
                'ST4 actual runtime differs from qualified same-card static parent')
    return rec['qualification_profile']


def budget_binding(p):
    return {'draft_signature':signature(p),'source_catalog':p['source_catalog'],'static_protocol':p['static_protocol'],
            'parent_validated_profile':p['parent_validated_profile'],'engine_kwargs_digest':digest(p['engine_kwargs']),
            'cpu_sources':cpu_codes(),'static_dependencies':native.dependencies()}


def cpu(draft,output):
    import torch
    p=read(draft); catalog=science(p)
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not torch.cuda.is_initialized(), 'CPU admission requires invisible uninitialized CUDA')
    require(sys.executable==p['python'], 'CPU native processing must use exact sealed deployment interpreter')
    for package,version in p['expected_packages'].items():
        require(importlib.metadata.version(package)==version,'CPU package changed: '+package)
    torch.set_num_threads(p.get('torch_num_threads') or 1)
    sid=p['assigned_ids'][0]; attempt=runner.cpu_sample(p,catalog['items'][sid])
    require(attempt['image_count']==2 and not torch.cuda.is_initialized(), 'Real two-image native CPU admission failed')
    value={'schema':'static-t4-native-cpu-admission-v1','status':'PASS','model_calls':0,'weights_loaded':False,
           'CUDA_initialized':False,**budget_binding(p),'representative_ids':[sid],
           'sid_to_reference':{x:sid for x in p['assigned_ids']},
           'assigned_item_digests':{x:digest(catalog['items'][x]) for x in p['assigned_ids']},'attempts':[attempt]}
    ref=write_new(output,value)
    return {'status':'CPU_PASS_GPU_PENDING','static_cpu_budget':ref,'static_deep_ids':[sid],
            'mapped_items':len(p['assigned_ids']),'prompt_tokens':attempt['prompt_tokens'],'model_calls':0}


def native_cpu(draft,output):
    return cpu(draft,output)


def verify_cpu(p,catalog=None):
    catalog = science(p) if catalog is None else catalog
    budget=native.cached_verified(p['static_cpu_budget'])
    require(budget['schema']=='static-t4-native-cpu-admission-v1' and budget['status']=='PASS'
            and budget['model_calls']==0 and budget['weights_loaded'] is False and budget['CUDA_initialized'] is False,
            'Actual native CPU admission missing')
    require(all(budget[k]==v for k,v in budget_binding(p).items()),'Native CPU admitted source/config/code changed')
    sid=p['assigned_ids'][0]
    require(budget['representative_ids']==p['static_deep_ids']==[sid]
            and budget['sid_to_reference']==p['static_cpu_template_map']=={x:sid for x in p['assigned_ids']}
            and budget['assigned_item_digests']=={x:digest(catalog['items'][x]) for x in p['assigned_ids']}
            and len(budget['attempts'])==1,'Native CPU representative/finite input coverage differs')
    value=budget['attempts'][0]
    require(value['sample_id']==sid and value['item_digest']==digest(catalog['items'][sid])
            and value['image_count']==len(value['mm_placeholders']['image'])==2
            and [x['frame_index'] for x in value['source_reader']]==[0,179]
            and all(x['matches_original_video_RGB8'] for x in value['source_reader'])
            and value['prompt_tokens']==len(value['prompt_token_ids'])
            and value['prompt_tokens']+16384<=p['engine_kwargs']['max_model_len']
            and value['fits_context'] and value['processor_resize_crop_calls']==0 and not value['CUDA_initialized'],
            'Native two-image shape/prompt/context proof incomplete')
    for ref in value['official_source_identity']:
        require(native.official_source(ref['path'],ref['distribution'])==ref,'Official processor source changed')
    return budget


def final_review(p):
    review=native.cached_verified(p['st4_final_review'])
    require(review['schema']=='static-t4-final-execution-review-v1'
            and review['status']=='PASS_STATIC_T4_EXECUTION_PREFLIGHT'
            and review['owner_chat']==OWNER and review['codes']==final_review_codes()
            and all(review[k]==p[k] for k in COMMON)
            and review['draft_signatures'][p['st4_allocation_id']]==signature(p)
            and review['cpu_budgets'][p['st4_allocation_id']]==p['static_cpu_budget']
            and review['new_qualification_calls']==0, 'Missing exact final scientific/native/operations preflight')
    return review


def verify_code(p):
    require(p['execution_policy']=={'module':MODULE,'source_sha256':file_sha256(__file__)}
            and p['static_entry']==descriptor(native.__file__) and p['static_dependencies']==native.dependencies()
            and p['st4_runner_entry']==descriptor(runner.__file__) and p['st4_source_entry']==descriptor(source.__file__)
            and p['implementation']['sha256']==file_sha256(base.__file__), 'Frozen runtime code changed')


def full_policy(p):
    catalog=science(p);verify_code(p);verify_cpu(p,catalog);final_review(p)
    return catalog


def deep_policy(p,item,ordinal):
    return ordinal==0


def full_input(p,item,raw,deep,persist=True):
    sid=item['sample_id']; is_deep=sid==p['assigned_ids'][0]
    require(raw['profile_sha256']==digest(p) and raw['source']['catalog']==p['source_catalog']
            and raw['source']['item_digest']==digest(item)
            and raw['source']['media']==[p['media_files'][x['path']] for x in native.image_entries(item)]
            and raw['deep'] is is_deep and bool(deep) is is_deep
            and raw['variant']=='CURRENT_X_STATIC_T4_STRICT_FIRST_LAST'
            and raw['accuracy_eligible'] is item['row']['accuracy_eligible']
            and raw['correct']==(raw['raw_answer'].strip()==raw['truth'] if raw['accuracy_eligible'] else None),
            'ST4 raw source/zero/deep mapping differs')
    parent_ops.generation_contract(raw['generation'])
    require(raw['request_mm_processor_kwargs']==native.native_kwargs(p),'Native image kwargs changed')
    with runner.dispatch(p):
        expected=native.budget_for(p,sid)
        if is_deep:
            qualified_deployment(p,[x['runtime'] for x in deep])
            proof=native.validate_input(p,item,raw,deep,persist=False)
        else:
            proof=native.ordinary_input(p,item,raw,expected)
    if persist:
        base._append(Path(p['output'])/'input-admissions.jsonl', {'sample_id':sid,'raw_digest':digest(raw),
            'observations_digest':digest(deep),'representative_sid':expected['sample_id'],'deep':is_deep,
            'profile_sha256':digest(p),'source_item_digest':digest(item),'proof':proof})
    return proof


def seal(draft,output,*,cpu_budget=None,final_review_ref=None,non_gemma_profiles=None):
    p=read(draft)
    if cpu_budget is not None:
        budget=verified(cpu_budget);p['static_cpu_budget']=cpu_budget
        p['static_deep_ids']=budget['representative_ids'];p['static_cpu_template_map']=budget['sid_to_reference']
    if final_review_ref is not None:
        p['st4_final_review']=final_review_ref
    if non_gemma_profiles is not None:
        require(p['st4_model']=='gemma' and len(non_gemma_profiles)==6, 'Only Gemma binds exactly six new ST4 predecessors')
        p['st4_pre_gemma_intent']['expected_non_gemma_profiles']=non_gemma_profiles
    catalog=science(p); verify_cpu(p,catalog);final_review(p);verify_code(p);claims_gate(p)
    from .gemma_x_full import storage_admission
    storage_admission(p)
    lock=p['physical_lock'];state=Path(lock['path']).stat()
    require(not Path(lock['path']).is_symlink() and (state.st_dev,state.st_ino)==(lock['st_dev'],lock['st_ino']), 'Historical same-inode lease changed')
    p['st4_reviewed_draft_signature']=signature(p)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as stream:
        json.dump(p,stream); temporary=Path(stream.name)
    try:
        return base.seal(temporary,output)
    finally:
        temporary.unlink()


def run(profile):
    return runner.execute(profile)


def supervise(profile):
    # Reuse the reviewed real-wait/owned-exit/same-inode-release implementation.
    p=read(profile);full_policy(p);claims_gate(p)
    old=video_ops.MODULE;video_ops.MODULE=MODULE
    try:
        return video_ops.supervise(profile)
    finally:
        video_ops.MODULE=old


def closed(profile):
    p=read(profile);catalog=full_policy(p);claims_gate(p,complete=True);out=Path(p['output'])
    wait,result,exit_record=read(out.with_suffix('.wait.json')),read(out/'result.json'),read(out/'exit.json')
    require(wait['profile_sha256']==digest(p) and wait['returncode']==0 and wait['real_wait_observed'] is True
            and wait['owned_engines_exited'] is True and wait['original_physical_lease_released'] is True
            and wait['completed_ids']==p['assigned_ids'] and wait['terminal_status']=='COMPLETE'
            and result['status']=='COMPLETE' and result['completed_ids']==p['assigned_ids'] and not result.get('UNCERTAIN_ids')
            and result['automatic_retry'] is False and result['formal_auto_resume'] is False
            and not (out/'failure.json').exists() and exit_record['owned_engine_workers_exited'] is True
            and exit_record['shutdown_error'] is None,'No real clean ST4 worker closure; unknown claims remain HOLD')
    identity=read(out/'identity.json');storage=read(out/'storage-admission.json');lease=read(out/'lease.json')
    require(identity['profile']==p and identity['profile_sha256']==digest(p)
            and storage['status']=='PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN' and storage['profile_sha256']==digest(p)
            and all(lease[k]==v for k,v in p['physical_lock'].items()),'Worker identity/physical SSD/original lease evidence differs')
    # Historical release is in the original real-wait record. Do not try to
    # acquire the lease now: another legitimate successor can already hold it.
    state=Path(p['physical_lock']['path']).stat()
    require(not Path(p['physical_lock']['path']).is_symlink()
            and (state.st_dev,state.st_ino)==(p['physical_lock']['st_dev'],p['physical_lock']['st_ino']), 'Original lease inode replaced')
    runtime=read(out/'runtime.json');qualified_deployment(p,runtime);native.validate_parent_runtime(p,runtime)
    raws,proofs,events=records(out/'raw.jsonl'),records(out/'input-admissions.jsonl'),records(out/'events.jsonl')
    observations,installs=records(out/'deep-observations.jsonl'),records(out/'deep-install.jsonl')
    ids=p['assigned_ids'];require([r['sample_id'] for r in raws]==[r['sample_id'] for r in proofs]==ids
        and [r['sample_id'] for r in observations]==[r['sample_id'] for r in installs]==[ids[0]]
        and [(r['event'],r['sample_id']) for r in events]==[(event,s) for s in ids for event in ('INTENT','TECHNICAL_PASS')],
        'ST4 exact finite raw/PASS/deep journals differ')
    require(observations[0]['raw_digest']==digest(raws[0]),'First native proof not bound to first formal raw')
    for ordinal,(raw,proof) in enumerate(zip(raws,proofs)):
        sid=ids[ordinal];item=catalog['items'][sid];claim=read(Path(p['claim_root'])/(sid+'.json'))
        require(claim=={'sample_id':sid,'profile_sha256':digest(p),'worker_token':p['worker_token'],
                       'attempt':1,'cold':None,'authorization':p['static_authorization']},'Wrong single-attempt durable claim')
        require(raw['model']==p['model'] and raw['task']=='T4' and raw['domain']=='classic' and raw['cold'] is None
                and raw['attempt']==1 and raw['ordinal']==events[2*ordinal]['ordinal']==events[2*ordinal+1]['ordinal']==ordinal
                and raw['truth']==item['row']['truth'] and raw['question']==item['row']['prompt']
                and raw['finish_reason']=='stop' and raw['raw_answer'].strip() in ('LEFT','RIGHT')
                and proof['raw_digest']==events[2*ordinal+1]['raw_digest']==digest(raw)
                and proof['profile_sha256']==digest(p) and proof['source_item_digest']==digest(item)
                and proof['representative_sid']==ids[0] and proof['deep'] is (ordinal==0), 'ST4 original raw/claim/proof metadata differs')
        deep=observations[0]['observations'] if ordinal==0 else []
        require(proof['observations_digest']==digest(deep)
                and proof['proof']==full_input(p,item,raw,deep,persist=False), 'ST4 native evidence does not reproduce')
    return {'status':'COMPLETE_PASS','profile':descriptor(profile),'model':p['st4_model'],'card':p['card'],
            'completed':len(raws),'alias_count':sum(len(catalog['items'][s]['original_video_aliases']) for s in ids),
            'wait':descriptor(out.with_suffix('.wait.json')),'result':descriptor(out/'result.json'),
            'same_inode_release_recorded':True,'model_calls':0}


def return_manifest(profile,output):
    closure=closed(profile);p=read(profile);out=Path(p['output'])
    paths={Path(profile),out.with_suffix('.wait.json'),out.with_suffix('.launch.json')}
    paths.update(path for path in out.iterdir() if path.is_file() and path.suffix in ('.json','.jsonl'))
    paths.update(Path(p['claim_root'])/(sid+'.json') for sid in p['assigned_ids'])
    paths.update(Path(p[k]['path']) for k in COMMON+('st4_source_admission','static_cpu_budget','st4_final_review','parent_validated_profile','qualification_summary'))
    paths.update(Path(ref['path']) for ref in final_review_codes().values())
    catalog=native.cached_verified(p['source_catalog'])
    paths.update(Path(catalog[k]['path']) for k in ('manifest','source_prepare_receipt','source_ready_catalog','source_admission'))
    if p.get('st4_pre_gemma_intent'):
        paths.add(Path(p['st4_pre_gemma_intent']['path']))
    index=native.cached_verified(p['st4_parent_index'])['parents'][p['st4_allocation_id']]
    paths.update(Path(index[k]['path']) for k in ('runtime','completed_wait','completed_result','qualification_profile'))
    # Include the exact scientific alias source QA metadata; formal PNG media is retained on NVMe.
    paths.update(Path(a['source_qa']['path']) for sid in p['assigned_ids'] for a in catalog['items'][sid]['original_video_aliases'])
    value={'schema':'static-t4-closed-worker-full-return-manifest-v1','status':'READY_FULL_SHA_RETURN_NOT_YET_RETURNED',
           'owner_chat':OWNER,'closure':closure,'files':[descriptor(x) for x in sorted(paths)],
           'media_or_weights_included':False,'model_calls':0}
    return {'status':value['status'],'manifest':write_new(output,value),'files':len(paths),'model_calls':0}


def summarize(plan,output):
    plan_value=read(plan);refs=plan_value.get('global_profiles',plan_value.get('profiles'))
    require(isinstance(refs,list) and len(refs)==8,'Exact eight ST4 global profiles required')
    answers={};cells=defaultdict(list);closures=[];common=None;all_aliases=[]
    for ref in refs:
        p=verified(ref);closure=closed(ref['path']);closures.append(closure)
        current={k:p[k] for k in COMMON}
        require(common is None or common==current,'Global summary mixes ST4 scientific versions');common=current
        catalog=native.cached_verified(p['source_catalog'])
        for raw in records(Path(p['output'])/'raw.jsonl'):
            sid=raw['sample_id'];key=(p['st4_model'],sid)
            require(key not in answers,'A unique ST4 input was attempted twice');answers[key]=raw
            for alias in catalog['items'][sid]['original_video_aliases']:
                params=alias['source_parameters']; scored=alias['accuracy_eligible']
                record={'model':p['st4_model'],'sample_id':alias['sample_id'],'case_id':sid,
                        'raw_answer':raw['raw_answer'],'truth':alias['truth'],'accuracy_eligible':scored,
                        'correct':raw['raw_answer'].strip()==alias['truth'] if scored else None}
                cells[p['st4_model'],params['N'],params['coherence']].append(record);all_aliases.append(record)
    require(len(answers)==4*catalog['unique_inputs']<=3072
            and set(answers)=={(m,s) for m in MODELS for s in catalog['items']}
            and len(all_aliases)==3072,'Complete four-model unique/alias denominator missing')
    rows=[]
    for (model,n,coherence),records_ in sorted(cells.items()):
        eligible=[x for x in records_ if x['accuracy_eligible']]
        require(len(records_)==(24 if coherence==0 else 48) and len(eligible)==(0 if coherence==0 else 48),'Wrong scored/bias cell denominator')
        correct=sum(x['correct'] for x in eligible)
        rows.append({'model':model,'N':n,'coherence':coherence,'calls':len({x['case_id'] for x in records_}),
                     'scientific_aliases':len(records_),'accuracy_n':len(eligible),'correct':correct,
                     'accuracy':correct/len(eligible) if eligible else None,
                     'answers':dict(Counter(x['raw_answer'].strip() for x in records_)),'zero_is_bias_only':coherence==0})
    result={'schema':'static-t4-four-model-summary-v1','status':'COMPLETE_ORIGINAL_RAW_CLOSED_AUDIT_PASS','owner_chat':OWNER,
            **common,'executed_model_calls':len(answers),'unique_answers':len(answers),'scientific_aliases':len(all_aliases),
            'scored':sum(x['accuracy_eligible'] for x in all_aliases),'zero_bias':sum(not x['accuracy_eligible'] for x in all_aliases),
            'extra_qualification_calls':0,'closures':closures,'cells':rows,
            'full_SHA_return_required_before_publication':True,'model_calls_during_summary':0}
    return {'status':result['status'],'summary':write_new(output,result),'model_calls':0}


def prepare(root,ready_catalog,protocol,authorization_path):
    """CPU metadata only; create new allocation and eight unchanged-static drafts."""
    root=Path(root);folder=root/'plans'/NAME
    catalog=read(ready_catalog);items=catalog['items'];cref=descriptor(ready_catalog);pref=descriptor(protocol);aref=descriptor(authorization_path)
    require(catalog['selection']=='all' and catalog['status']=='PASS_ENDPOINT_EXPORT'
            and catalog['protocol']==read(protocol),'Actual all-source endpoint catalog/protocol required')
    parents={};parent_base=root/'plans/static-x-t123-20261104/full'
    for model in MODELS:
        group='qwen_T23' if model=='qwen' else model
        for card in ('fourier0','fourier1'):
            profile=parent_base/(group+'-'+card+'-profile.json');p=read(profile);out=Path(p['output'])
            summary=parent_ops.qualification_summary(p['qualification_summary']);matches=[]
            for ref in summary['profiles']:
                candidate=verified(ref)
                if candidate['gpu_uuid']==p['gpu_uuid'] and native.deployment_signature(candidate)==native.deployment_signature(p):
                    matches.append(ref)
            require(len(matches)==1,'Parent static deployment lacks unique accepted qualification')
            parents[model+'-'+card]={'profile':descriptor(profile),'qualification_summary':p['qualification_summary'],
                'qualification_profile':matches[0],'runtime':descriptor(out/'runtime.json'),
                'completed_wait':descriptor(out.with_suffix('.wait.json')),'completed_result':descriptor(out/'result.json'),
                'parent_group':group,'selection_reason':'Existing same-card static deployment unchanged; Qwen uses already-qualified static T23 65536/4GiB'}
    parent_ref=write_new(folder/'static-parent-bindings.json',{'schema':'static-t4-parent-bindings-v1','owner_chat':OWNER,'parents':parents})
    allocations=[]
    for card in ('fourier0','fourier1'):
        ids=sorted(s for s,item in items.items() if all(a['source_parameters']['layout_index']%2==int(card[-1]) for a in item['original_video_aliases']))
        for model in MODELS:
            label=model+'-'+card;out=root/'runs/reproducibility'/NAME/label
            allocations.append({'allocation_id':label,'model':model,'card':card,'assigned_ids':ids,
                                'output':str(out),'claim_root':str(out.with_name(label+'-claims'))})
    allocation_ref=write_new(folder/'allocation.json',{'schema':'static-t4-global-allocation-v1','owner_chat':OWNER,
        'st4_authorization':aref,'source_catalog':cref,'st4_protocol':pref,'st4_parent_index':parent_ref,
        'maximum_new_model_calls':4*len(items),'allocations':allocations})
    drafts=[]
    for allocation in allocations:
        label=allocation['allocation_id'];rec=parents[label];parent=verified(rec['profile']);p=copy.deepcopy(parent)
        for key in list(p):
            if key.startswith(('full_','static_')) or key in ('worker_token','media_files','final_review_draft','final_static_review',
                'execution_authorization','experiment_plan','canonical_catalog','reference_panel','qualification_gate','control_file','control_generation'):
                p.pop(key,None)
        p.update(worker=label,st4_model=allocation['model'],st4_allocation_id=label,card=allocation['card'],
            workspace_root=str(root),phase='full',assigned_ids=allocation['assigned_ids'],maximum_model_calls=len(allocation['assigned_ids']),
            output=allocation['output'],claim_root=allocation['claim_root'],sources={},automatic_retry=False,formal_auto_resume=False,
            pause_file=allocation['output']+'-pause.json',choices_by_task={'T4':['LEFT','RIGHT']},
            st4_authorization=aref,static_authorization=aref,st4_protocol=pref,static_protocol=pref,source_catalog=cref,
            st4_parent_index=parent_ref,st4_allocation_plan=allocation_ref,st4_source_admission=catalog['source_prepare_receipt'],
            parent_validated_profile=rec['profile'],qualification_summary=rec['qualification_summary'],static_cold=None,
            static_entry=descriptor(native.__file__),static_dependencies=native.dependencies(),
            st4_runner_entry=descriptor(runner.__file__),st4_source_entry=descriptor(source.__file__),
            execution_policy={'module':MODULE,'source_sha256':file_sha256(__file__)})
        if allocation['model']=='gemma':
            p['st4_pre_gemma_intent']={'path':str(root/'runs/reproducibility'/NAME/('queue-'+p['card'])/'pre-gemma-global-closure.json'),
                'owner_chat':OWNER,**{k:p[k] for k in ('st4_authorization','source_catalog','st4_protocol')},'expected_non_gemma_profiles':[]}
        science(p);drafts.append(write_new(folder/(label+'-draft.json'),p))
    value={'schema':'static-t4-draft-plan-v1','status':'CPU_DRAFTS_ONLY_NOT_SEALED','owner_chat':OWNER,
           'st4_authorization':aref,'source_catalog':cref,'st4_protocol':pref,'st4_parent_index':parent_ref,
           'st4_allocation_plan':allocation_ref,'drafts':drafts,'model_calls':0}
    return {'status':value['status'],'plan':write_new(folder/'draft-plan.json',value),'drafts':drafts,'model_calls':0}


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','science','validate','cpu','native_cpu','seal','run','supervise','closed','return-manifest','return_manifest','summarize'])
    for key in ('root','ready-catalog','protocol','authorization','profile','draft','output','plan','cpu-budget','final-review','non-gemma-profiles'):
        p.add_argument('--'+key)
    a=p.parse_args()
    if a.command=='prepare':v=prepare(a.root,a.ready_catalog,a.protocol,a.authorization)
    elif a.command in ('science','validate'):
        value=read(a.profile or a.draft);catalog=science(value);v={'status':'PASS_CPU_SCIENCE_METADATA','unique_inputs':len(catalog['items']),'model_calls':0}
    elif a.command in ('cpu','native_cpu'):v=cpu(a.draft,a.output)
    elif a.command=='seal':
        v=seal(a.draft,a.output,cpu_budget=descriptor(a.cpu_budget) if a.cpu_budget else None,
               final_review_ref=descriptor(a.final_review) if a.final_review else None,
               non_gemma_profiles=read(a.non_gemma_profiles) if a.non_gemma_profiles else None)
    elif a.command=='run':v=run(a.profile)
    elif a.command=='supervise':v=supervise(a.profile)
    elif a.command=='closed':v=closed(a.profile)
    elif a.command in ('return-manifest','return_manifest'):v=return_manifest(a.profile,a.output)
    else:v=summarize(a.plan,a.output)
    print(json.dumps(v,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
