"""Explicit finite repair of zero-output reader failures; never replay old queues.

All original profiles, claims, raw answers and failed waits remain immutable.
Only the five previously empty logical arms are admitted, on their original GPU.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import socket

from .reader_route_smoke import NAME
from .refinement_worker import read, require, write_new, descriptor, verified
from .storage import digest, file_sha256

AUTHORIZATION = '修复漏洞，确保所有答案都是严格从我们固定协议来的，比如180帧全送进去，比如temp greedy fix，比如不碰procssor和后面的东西，比如all cam rerodudece等等，之前几千条怎么跑的，这次除了和unit8有关的，几乎都得一致吧？'
WORKERS = {'fourier': ['qwen-fourier1-C', 'cosmos-fourier1-A', 'cosmos-fourier1-B', 'cosmos-fourier1-C'],
           'radon': ['cosmos-radon0-C']}
EXCEPTIONS = {'qwen-fourier1-C': 't2_14765cd72ee89fc7cd98',
              'cosmos-radon0-C': 't3_01c9a08668e7506dc483'}
EXECUTION_FIELDS = {'worker', 'worker_token', 'output', 'claim_root', 'pause_file', 'implementation', 'repair'}


def records(path):
    return [__import__('json').loads(x) for x in Path(path).open()] if Path(path).exists() else []


def check_parent(parent, preserved_claims):
    out = Path(parent['output'])
    require(not records(out/'raw.jsonl'), 'Original raw exists; never replace a scientific answer')
    assigned = set(parent['assigned_ids'])
    actual = {x.stem for x in Path(parent['claim_root']).glob('*.json')} & assigned
    require(actual == set(preserved_claims), 'Original claim set changed; no implicit retry')
    for sid, ref in preserved_claims.items():
        require(sid in assigned and file_sha256(ref['path']) == ref['sha256'], 'Original claim changed')
    wait = out.with_suffix('.wait.json')
    if preserved_claims:
        w = read(wait)
        require(w['profile_sha256'] == digest(parent) and w['returncode'] == 1 and not w['completed_ids']
            and w['real_wait_observed'] and w['owned_engines_exited'] and w['original_physical_lease_released'],
            'Original failed attempt is not safely closed')
    else:
        require(not out.exists() and not wait.exists(), 'Unattempted arm has an execution record')


def validate_repair_profile(p):
    r = p['repair']; auth = verified(r['authorization']); parent = verified(r['parent_profile'])
    require(auth['human_message'] == AUTHORIZATION and not auth['automatic_retry'], 'Finite human repair authority missing')
    require(p['worker'] == parent['worker']+'-repair1' and parent['worker'] in auth['workers'], 'Unexpected repair worker')
    require({k:v for k,v in p.items() if k not in EXECUTION_FIELDS} ==
            {k:v for k,v in parent.items() if k not in EXECUTION_FIELDS}, 'Scientific or resource settings changed')
    require(p['claim_root'] != parent['claim_root'] and p['output'] != parent['output'], 'Old journals must be retained')
    require(set(r['preserved_claims']) == ({EXCEPTIONS[parent['worker']]} if parent['worker'] in EXCEPTIONS else set()),
            'Only the two zero-output failures may receive attempt 2')
    require(len(p['assigned_ids']) == auth['workers'][parent['worker']]['maximum_new_answers'], 'Quota changed')
    check_parent(parent, r['preserved_claims'])
    for ref in r['CPU_evidence']:
        evidence = verified(ref)
        require(evidence.get('CUDA_initialized', evidence.get('CUDA_initialized_after')) is False, 'CPU evidence initialized CUDA')
    return parent


def cpu(profile_path):
    """Actual installed vLLM processor, no engine or model weights, one/task."""
    import os
    import sys
    p=read(profile_path)
    require(p['worker'] in sum(WORKERS.values(), []), 'CPU check outside fixed repair')
    os.environ.update(p['environment']);os.environ['CUDA_VISIBLE_DEVICES']=''
    for path in reversed(p['environment'].get('PYTHONPATH','').split(':')):
        if path:sys.path.insert(0,path)
    import torch
    torch.set_num_threads(6);torch.set_grad_enabled(False)
    if p['model']=='cosmos':
        sys.path.insert(0,p['runtime']['cosmos_package'])
        import transformers_cosmos3
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from .reader_route_smoke import decode, metadata, NativeObservation
    processor=AutoProcessor.from_pretrained(p['checkpoint'],local_files_only=True,trust_remote_code=False)
    tokenizer_path=p['runtime'].get('original_checkpoint',p['checkpoint']) if p['model']=='cosmos' else p['checkpoint']
    config=ModelConfig(model=p['checkpoint'],tokenizer=tokenizer_path,dtype='bfloat16',max_model_len=65536,
        limit_mm_per_prompt={'video':1,'image':0,'audio':0},mm_processor_cache_gb=0,
        mm_processor_kwargs={'do_resize':False,'do_sample_frames':False})
    native=MULTIMODAL_REGISTRY.create_processor(config,tokenizer=processor.tokenizer)
    source=verified(p['source_catalog']);seen={};observations=[]
    for sid in p['assigned_ids']:
        item=source['items'][sid]
        if item['task'] in seen:continue
        frames,receipt,_=decode(p['sources'][sid]['path'],p['arm'])
        text=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},
            {'type':'text','text':item['row']['prompt']}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        # Match the real route's metadata, including the official reader's True
        # flag: the explicit request False must override it for native C.
        meta=receipt['official_metadata'] if p['arm']=='C' else metadata()
        parsed=native.info.parse_mm_data({'video':(frames,meta)})
        with NativeObservation(p['model']) as obs:
            obs.active=True
            result=native.apply(ProcessorInputs(prompt=processor.tokenizer.encode(text,add_special_tokens=False),
                mm_data_items=parsed,hf_processor_mm_kwargs={'do_resize':False,'do_sample_frames':False} if p['arm']=='C' else {}),
                TimingContext(enabled=False))
            require(obs.processors, 'Actual installed vLLM processor was not observed')
            actual=obs.processors
        seen[item['task']]=digest(result['prompt_token_ids'])
        observations.append({'task':item['task'],'sample_id':sid,'reader':receipt,
            'actual_native_processor':actual,'prompt_tokens':len(result['prompt_token_ids']),
            'request_mm_kwargs':{'do_resize':False,'do_sample_frames':False} if p['arm']=='C' else {}})
        del frames,result
    require(not torch.cuda.is_initialized(), 'CPU admission initialized CUDA')
    record={'status':'PASS','GPU_calls':0,'weights_loaded':False,'CUDA_initialized':False,
        'parent_profile_sha256':digest(p),'prompt_tokens_by_task':seen,'observations':observations,
        'implementation':descriptor(__file__)}
    write_new(Path(profile_path).with_name(p['worker']+'-repair1-cpu.json'),record)
    return {'status':'PASS','worker':p['worker'],'tasks':len(seen),'model_calls':0}


def prepare(root, host):
    root = Path(root); plan = root/'plans'/NAME; run = root/'runs/reproducibility'/NAME
    require(host in WORKERS, 'Unknown host')
    auth_path = plan/f'repair1-{host}-authorization.json'
    require(not auth_path.exists(), 'Repair already prepared; inspect actual ledger, never overwrite')
    parents = {w: read(plan/(w+'-profile.json')) for w in WORKERS[host]}
    require(all(p['hostname'] == socket.gethostname() for p in parents.values()), 'Wrong host')
    for card in ({'fourier1'} if host == 'fourier' else {'radon0'}):
        require(read(plan/(card+'-queue.json'))['status'] == 'HELD_NO_RETRY', 'Old queue state changed')
    preserved = {}
    for w,p in parents.items():
        preserved[w] = {EXCEPTIONS[w]: descriptor(Path(p['claim_root'])/(EXCEPTIONS[w]+'.json'))} if w in EXCEPTIONS else {}
        check_parent(p, preserved[w])
        a=read(plan/(w+'-repair1-cpu.json'))
        require(a['status']=='PASS' and a['GPU_calls']==0 and a['CUDA_initialized'] is False
            and a['parent_profile_sha256']==digest(p), 'Task-specific CPU admission missing or changed')
    auth = {'schema':'reader-route-finite-repair-authorization-v1', 'human_message':AUTHORIZATION,
        'interpretation':'Repair instruction authorizes only the two retained zero-output CPU failures and 382 unclaimed tuples; no completed answer is rerun.',
        'original_total_budget':1728, 'retained_complete_answers':1344, 'total_maximum_new_answers':384,
        'automatic_retry':False, 'workers':{w:{'parent_profile':descriptor(plan/(w+'-profile.json')),
            'maximum_new_answers':len(p['assigned_ids']), 'preserved_claims':preserved[w]} for w,p in parents.items()}}
    require(sum(len(p['assigned_ids']) for p in parents.values()) == {'fourier':288,'radon':96}[host], 'Finite repair quota differs')
    write_new(auth_path, auth)
    profiles = []; replacements = {}
    for w,parent in parents.items():
        # A prompt expansion is independent of these fixed pixels; bind every
        # actual repair prompt to the already retained A tokens for the task.
        reference = run/(w.rsplit('-',1)[0]+'-A')/'raw.jsonl'
        if not reference.exists():
            reference = run/'cosmos-radon0-A/raw.jsonl'  # Cosmos T2 uses the sealed CPU control below.
        prompt_tokens = {}
        for raw in records(reference):
            h = digest(raw['actual_prompt_token_ids'])
            require(raw['task'] not in prompt_tokens or prompt_tokens[raw['task']] == h, 'A prompt layout is inconsistent')
            prompt_tokens[raw['task']] = h
        # Host-local CPU admission writes a task-specific prompt hash when the
        # original card has no completed A (Cosmos T2). No inference is added.
        cpu_path = run/('cpu-vllm-request-fixed-'+parent['model']+'.json')
        cpu = read(cpu_path)
        require(cpu['CUDA_initialized_after'] is False, 'CPU admission required')
        admission_path = plan/(w+'-repair1-cpu.json')
        require(admission_path.exists(), 'Task-specific CPU admission missing: '+w)
        admission = read(admission_path)
        require(admission['status']=='PASS' and admission['GPU_calls']==0
            and admission['parent_profile_sha256']==digest(parent), 'CPU admission binding differs')
        for task,h in admission['prompt_tokens_by_task'].items():
            require(task not in prompt_tokens or prompt_tokens[task]==h, 'CPU prompt differs from retained A')
            prompt_tokens[task]=h
        worker=w+'-repair1'; p=deepcopy(parent)
        p.update(worker=worker, worker_token=digest({'parent':descriptor(plan/(w+'-profile.json')), 'auth':descriptor(auth_path)}),
            output=str(run/worker), claim_root=str(run/'claims-repair1'/parent['model']/parent['arm']),
            pause_file=str(plan/(worker+'-PAUSE.json')), implementation=descriptor(Path(__file__).with_name('reader_route_smoke.py')),
            repair={'parent_profile':descriptor(plan/(w+'-profile.json')), 'authorization':descriptor(auth_path),
                'preserved_claims':preserved[w], 'prompt_tokens_by_task':prompt_tokens,
                'CPU_evidence':[descriptor(cpu_path), descriptor(admission_path)]})
        validate_repair_profile(p)
        path=plan/(worker+'-profile.json');write_new(path,p);profiles.append(str(path))
        replacements[w]={'profile':descriptor(path), 'output_worker':worker, 'parent_profile':descriptor(plan/(w+'-profile.json'))}
    card='fourier1' if host=='fourier' else 'radon0'
    queue_path=plan/(card+'-repair1-queue.json')
    write_new(queue_path,{'status':'SEALED','profiles':profiles,'automatic_retry':False,'formal_auto_resume':False})
    manifest={'schema':'reader-route-repair-manifest-v1','authorization':descriptor(auth_path),
        'replacements':replacements, 'card':card, 'queue':descriptor(queue_path), 'maximum_new_answers':sum(len(p['assigned_ids']) for p in parents.values())}
    write_new(plan/f'repair1-{host}-manifest.json',manifest)
    return {'status':'PREPARED_NOT_LAUNCHED','queue':str(queue_path),'maximum_new_answers':manifest['maximum_new_answers']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root');p.add_argument('--host');p.add_argument('--cpu-profile')
    a=p.parse_args();print(__import__('json').dumps(cpu(a.cpu_profile) if a.cpu_profile else prepare(a.root,a.host),sort_keys=True))


if __name__=='__main__':main()
