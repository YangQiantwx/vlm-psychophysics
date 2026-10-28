"""Finite official native-file X experiments; audit hooks never alter model math.

Profiles bind sources, public LLM/request/generation settings and physical leases.
New attempts require new profiles/claim directories; there is no automatic retry.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .storage import atomic_json, _append, digest, file_sha256, owner_identity
from .refinement_worker import Boundary, descriptor, physical_lease, read, require, verified, write_new

MEDIA = {'video': {'video_backend':'opencv', 'num_frames':180, 'fps':-1}}


def stat(path):
    x = Path(path).stat()
    return [x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns]


def records(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def tensor_description(value, deep=False):
    import torch
    if isinstance(value, torch.Tensor):
        d = {'shape':list(value.shape), 'dtype':str(value.dtype), 'device':str(value.device)}
        if deep:
            d['sha256'] = hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
        if not value.is_floating_point() and value.numel() <= 600:
            d['values'] = value.tolist()
        return d
    if isinstance(value, (list, tuple)):
        return [tensor_description(x, deep) for x in value[:8]]
    return None


def first_tensor(args, kwargs):
    import torch
    def search(x):
        if isinstance(x, torch.Tensor): return x
        if isinstance(x, (list, tuple)):
            for v in x:
                r = search(v)
                if r is not None: return r
        if isinstance(x, dict):
            for v in x.values():
                r = search(v)
                if r is not None: return r
        return None
    return search(args) if args else search(kwargs)


def media_entries(p, sid, item):
    content = item.get('content')
    if content is None:
        content = [{'type':'video', 'path':p['sources'][sid]['path'],
                    'sha256':item['ready']['video_sha256']},
                   {'type':'text', 'text':item['row']['prompt']}]
    return [dict(x) for x in content]


def messages(p, sid, item):
    content = []
    for x in media_entries(p, sid, item):
        kind = x['type']
        if kind == 'text': content.append({'type':'text', 'text':x['text']})
        elif kind in ('video', 'image'):
            content.append({'type':kind+'_url', kind+'_url':{'url':Path(x['path']).as_uri()}})
        else:
            require(kind in ('video_url','image_url'), 'Unsupported native content type')
            content.append(x)
    return [{'role':'user', 'content':content}]


def choices(p, item):
    result = item['row'].get('choices', p.get('choices_by_task', {}).get(item['task']))
    require(isinstance(result, list) and len(result) >= 2 and all(isinstance(x,str) for x in result),
            'Explicit task choices missing')
    return result


def video_contract(p, item):
    c = item.get('video_contract', p.get('video_contract', {}))
    require(c.get('frames') == 180 and c.get('width') and c.get('height'), 'Explicit full180 source geometry missing')
    return c


def load_hook(ref):
    module = importlib.import_module(ref['module'])
    require(file_sha256(module.__file__) == ref['source_sha256'], 'Bound extension source changed')
    return getattr(module, ref.get('function',ref.get('factory')))


def seal(draft_path, output_path):
    p = read(draft_path)
    require(not Path(output_path).exists(), 'Refuse to overwrite a sealed profile')
    catalog = verified(p['source_catalog'])
    ids = p['assigned_ids']
    require(len(ids) == len(set(ids)) == p['maximum_model_calls'], 'Finite allocation differs')
    files = {}
    for sid in ids:
        item = catalog['items'][sid]
        choices(p, item); video_contract(p, item)
        for entry in media_entries(p, sid, item):
            if entry['type'] not in ('video','image'): continue
            path = entry['path']; expected = entry.get('sha256')
            require(expected and file_sha256(path) == expected, 'Source bytes differ: '+path)
            files[path] = {'path':path, 'sha256':expected, 'stat':stat(path), 'type':entry['type']}
    p.update(schema='official-x-worker-v1', hostname=socket.gethostname(),
             implementation=descriptor(__file__), media_files=files,
             automatic_retry=False, formal_auto_resume=False)
    p['worker_token'] = digest({'worker':p['worker'], 'ids':ids, 'implementation':p['implementation'],
                                'source_catalog':p['source_catalog'], 'environment':p['environment']})
    from .inference import checkpoint_asset_identity
    assets = checkpoint_asset_identity(p['checkpoint'])
    if p.get('checkpoint_assets'): require(p['checkpoint_assets'] == assets, 'Checkpoint assets differ')
    p['checkpoint_assets'] = assets
    write_new(output_path, p)
    return {'status':'SEALED', 'profile':descriptor(output_path), 'ids':len(ids), 'media_files':len(files), 'model_calls':0}


class NativeObserver:
    def __init__(self, p):
        self.p=p; self.active=False; self.deep=False; self.contract=None; self.clear()
    def clear(self):
        self.reads=[]; self.processors=[]; self.read_seconds=0.; self.processor_seconds=0.
    def describe(self, data):
        return {k:tensor_description(v,self.deep) for k,v in data.items()
                if 'pixel' in k or 'grid' in k or k in ('video_num_patches','video_frame_counts')}
    def __enter__(self):
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.processing.context import InputProcessingContext
        self.original_read=VideoMediaIO.load_file
        self.original_call=InputProcessingContext.call_hf_processor
        self.original_post=InputProcessingContext._postprocess_output
        obs=self
        def load(loader,path):
            t=time.monotonic(); result=obs.original_read(loader,path)
            if obs.active:
                frames, meta=result.media; c=obs.contract
                require(str(frames.dtype)=='uint8' and list(frames.shape)==[180,c['height'],c['width'],3]
                        and meta['frames_indices']==list(range(180)), 'Actual native video count/order/geometry differs')
                d={'path':str(path),'shape':list(frames.shape),'dtype':str(frames.dtype),'metadata':meta}
                if obs.deep:
                    d['decoded_sha256']=hashlib.sha256(frames.tobytes()).hexdigest()
                    d['per_frame_sha256']=[hashlib.sha256(x.tobytes()).hexdigest() for x in frames]
                obs.reads.append(d); obs.read_seconds += time.monotonic()-t
            return result
        def call(ctx,hf_processor,data,kwargs={}):
            t=time.monotonic(); result=obs.original_call(ctx,hf_processor,data,kwargs)
            if obs.active: obs.processor_seconds += time.monotonic()-t
            return result
        def post(ctx,output):
            before=obs.describe(output) if obs.active else None
            result=obs.original_post(ctx,output)
            if obs.active: obs.processors.append({'before_native_dtype':before,'after_native_dtype':obs.describe(result)})
            return result
        VideoMediaIO.load_file=load; InputProcessingContext.call_hf_processor=call
        InputProcessingContext._postprocess_output=post
        return self
    def __exit__(self,*_):
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.processing.context import InputProcessingContext
        VideoMediaIO.load_file=self.original_read; InputProcessingContext.call_hf_processor=self.original_call
        InputProcessingContext._postprocess_output=self.original_post


def qualified_name(value):
    cls=value if inspect.isclass(value) else type(value)
    return cls.__module__+'.'+cls.__name__


def worker_identity(worker, model):
    import torch,vllm
    counts=Counter(); backends=Counter(); examples=defaultdict(list)
    for name,module in worker.get_model().named_modules():
        method=getattr(module,'quant_method',None)
        if method is None: continue
        counts['method:'+qualified_name(method)]+=1
        for k,v in vars(method).items():
            if k.endswith('backend'): backends[k+':'+str(v)]+=1
            if 'kernel' in k or 'linear' in k:
                label=qualified_name(v)
                if label.startswith(('vllm.','flashinfer.')):
                    counts[k+':'+label]+=1
                    if len(examples[label])<2: examples[label].append(name)
    runner=worker.model_runner; config=worker.vllm_config
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    capacity,concurrency=get_kv_cache_capacity(config,runner.kv_cache_config)
    result={'pid':os.getpid(),'vllm_version':vllm.__version__,'torch_version':torch.__version__,
        'cuda_version':torch.version.cuda,'model_class':qualified_name(worker.get_model()),
        'runner_class':qualified_name(runner),'kernel_config':repr(config.kernel_config),
        'method_kernel_counts':dict(counts),'actual_backends':dict(backends),
        'representative_modules':dict(examples),'actual_kv_capacity':int(capacity),'kv_concurrency':concurrency,
        'mm_device_do_normalize':config.model_config.get_multimodal_config().mm_device_do_normalize}
    if model=='qwen':
        from .qwen_official_qsa_replay import native_qsa_evidence
        result['native_qsa']=native_qsa_evidence(worker)
    return result


def resolve_module(model, paths):
    for path in paths:
        value=model
        try:
            for segment in path.split('.'):
                value=value[int(segment)] if segment.isdigit() else getattr(value,segment)
        except (AttributeError,IndexError,KeyError,TypeError): continue
        if hasattr(value,'register_forward_pre_hook'): return path,value
    raise ValueError('No actual vision module matches '+repr(paths))


def install_probe(worker, spec, model):
    import torch
    network=worker.get_model(); state={'outer':[],'normalized':[],'handles':[],'active_outer':None}
    worker._official_x_probe=state
    outer_path,outer=resolve_module(network,spec['outer_paths'])
    normalized_path,normalized=resolve_module(network,spec['normalized_paths'])
    state.update(outer_path=outer_path,normalized_path=normalized_path)
    def outer_hook(module,args,kwargs):
        grid=kwargs.get('grid_thw')
        if isinstance(grid,torch.Tensor): grid=grid.tolist()
        state['active_outer']=len(state['outer'])
        state['outer'].append({'tensor':tensor_description(first_tensor(args,kwargs),True),
            'grid_thw':grid,'num_frames_argument':kwargs.get('num_frames')})
    def normalized_hook(module,args,kwargs):
        state['normalized'].append({'tensor':tensor_description(first_tensor(args,kwargs),True),
                                     'outer_index':state['active_outer']})
    state['handles'].append(outer.register_forward_pre_hook(outer_hook,with_kwargs=True))
    state['handles'].append(normalized.register_forward_pre_hook(normalized_hook,with_kwargs=True))
    if model=='qwen':
        state['profiler']=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        state['profiler'].__enter__()
    norm=getattr(outer,'input_norm',None)
    state['normalizer']={'class':qualified_name(norm),'input_dtype':str(getattr(norm,'input_dtype',None)),
        'source':descriptor(inspect.getsourcefile(type(norm))) if norm is not None and inspect.getsourcefile(type(norm)) else None,
        'weight':getattr(norm,'weight',torch.empty(0)).detach().cpu().tolist() if norm is not None else None,
        'bias':getattr(norm,'bias',torch.empty(0)).detach().cpu().tolist() if norm is not None else None}
    return {'outer_path':outer_path,'normalized_path':normalized_path,'normalizer':state['normalizer']}


def read_probe(worker, model):
    state=worker._official_x_probe
    for handle in state.pop('handles'): handle.remove()
    profiler=state.pop('profiler',None)
    if profiler:
        try:
            profiler.__exit__(None,None,None)
            state['actual_topk_operator_calls']=[{'key':x.key,'count':x.count} for x in profiler.key_averages()
                if 'persistent_topk' in x.key or 'cooperative_topk' in x.key]
        except BaseException as error: state['profiler_error']=repr(error)
    try: state['runtime']=worker_identity(worker,model)
    except BaseException as error: state['runtime_error']=repr(error)
    del worker._official_x_probe
    return state


def validate_deep(p,item,deep):
    c=video_contract(p,item); spec=p['vision_observation']; normalized=[]; outer=[]
    for worker in deep:
        require('runtime_error' not in worker and 'profiler_error' not in worker,'Deep observer could not complete')
        normalized.extend(worker['normalized']); outer.extend(worker['outer'])
        require(worker['runtime']['actual_kv_capacity']>=p.get('minimum_kv_capacity',65536),'Insufficient actual KV capacity')
        if p['model']=='qwen':
            require(any(x['count']>0 and 'persistent_topk' in x['key'] for x in worker['actual_topk_operator_calls']),
                    'No actual native persistent TopK observed')
    require(normalized,'No actual post-normalization vision input recorded')
    for row in normalized:
        tensor=row['tensor']; require(tensor and tensor['dtype']==spec.get('dtype','torch.bfloat16')
            and tensor['device'].startswith('cuda'),'Actual normalized vision dtype/device differs')
    rule=spec['rule']
    if rule=='flat_patches':
        expected=[c['frames']//spec.get('temporal_patch_size',2),c['height']//spec.get('patch_size',16),c['width']//spec.get('patch_size',16)]
        require(any(r.get('grid_thw')==[expected] or r.get('grid_thw')==expected for r in outer),'Actual full-video grid differs')
        require(any(r['tensor']['shape']==[expected[0]*expected[1]*expected[2],spec.get('feature_dim',1536)]
                    for r in normalized),'Normalized full video patch shape differs')
    elif rule=='nchw_frames':
        frames=sum(r['tensor']['shape'][0] for r in normalized if len(r['tensor']['shape'])==4
            and r['tensor']['shape'][1:]==[3,c['height'],c['width']])
        image_count=sum(x['type']=='image' for x in item.get('content',[]))
        require(c['frames']<=frames<=c['frames']+image_count,'Actual native video frame tensors differ')
    elif rule=='custom':
        result=load_hook(p['vision_validation_hook'])(p,item,deep)
        require(result.get('status')=='PASS','Model-specific actual vision validation failed')
    else: raise ValueError('Unknown vision observation rule')


def validate_processor(p,item,processors):
    require(processors,'No actual native/qualified HF processor observation')
    if p['vision_observation']['rule']=='flat_patches':
        c=video_contract(p,item);s=p['vision_observation']
        expected=[[c['frames']//s.get('temporal_patch_size',2),
                   c['height']//s.get('patch_size',16),c['width']//s.get('patch_size',16)]]
        values=[x[phase]['video_grid_thw'].get('values') for x in processors
                for phase in ('before_native_dtype','after_native_dtype') if 'video_grid_thw' in x[phase]]
        require(expected in values,'Native processor sampled/resized the video')


def engine_kwargs(p):
    kwargs=dict(p['engine_kwargs']);kwargs.setdefault('model',p['checkpoint'])
    kwargs.setdefault('media_io_kwargs',p.get('media_io_kwargs',MEDIA))
    kwargs.setdefault('allowed_local_media_path',p['workspace_root'])
    if isinstance(kwargs.get('cpu_offload_params'),list): kwargs['cpu_offload_params']=set(kwargs['cpu_offload_params'])
    return kwargs


def generation(choice):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    result=SamplingParams(n=1,temperature=0.,top_p=1.,seed=0,repetition_penalty=1.,max_tokens=16384,
                          structured_outputs=StructuredOutputsParams(choice=choice))
    receipt={k:getattr(result,k) for k in ('n','temperature','top_p','seed','repetition_penalty','max_tokens','ignore_eos','stop','stop_token_ids')}
    require(receipt['ignore_eos'] is False and receipt['stop']==[] and receipt['stop_token_ids']==[],'Native EOS changed')
    return result,{**receipt,'choice':choice,'thinking':False,'sampling_repr':repr(result),'native_EOS_override':False}


def run(profile_path):
    p=read(profile_path);out=Path(p['output']);require(not out.exists(),'No implicit resume')
    require(sys.executable==p['python'] and socket.gethostname()==p['hostname'],'Wrong python/host')
    require(file_sha256(__file__)==p['implementation']['sha256'],'Sealed runner changed')
    for k,v in p['environment'].items(): require(os.environ.get(k)==v,'Environment differs: '+k)
    for k in p.get('cleared_environment',[]): require(k not in os.environ,'Unexpected environment: '+k)
    require(len(p['assigned_ids'])==len(set(p['assigned_ids']))==p['maximum_model_calls'],'Finite quota differs')
    require(not any((Path(p['claim_root'])/(s+'.json')).exists() for s in p['assigned_ids']),'Claim exists; no retry')
    catalog=verified(p['source_catalog']);boundary=Boundary(p)
    out.mkdir(parents=True);atomic_json(out/'identity.json',{'profile':p,'profile_sha256':digest(p),'owner':owner_identity(),'python':sys.executable})
    if boundary.check():
        result={'status':'PAUSED_BEFORE_LOAD','completed_ids':[],'UNCERTAIN_ids':[],'model_load_started':False}
        atomic_json(out/'result.json',result);return result
    for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM): signal.signal(sig,boundary.signal)
    from . import gemma_speed as lifecycle
    from .inference import checkpoint_asset_identity
    import torch
    if p.get('torch_num_threads'): torch.set_num_threads(p['torch_num_threads'])
    require(checkpoint_asset_identity(p['checkpoint'])==p['checkpoint_assets'],'Checkpoint assets changed')
    versions={n:importlib.metadata.version(n) for n in ('vllm','torch','transformers','numpy')}
    for name,version in p.get('expected_packages',{}).items(): require(importlib.metadata.version(name)==version,'Package differs: '+name)
    atomic_json(out/'versions.json',versions)
    completed=[];pending=None;seen=set();sampling_cache={}
    with physical_lease(p,torch) as lease:
        atomic_json(out/'lease.json',lease)
        with ExitStack() as stack:
            if p.get('bootstrap'):
                atomic_json(out/'bootstrap.json',load_hook(p['bootstrap'])(p))
            from vllm import LLM
            start=time.monotonic();kwargs=engine_kwargs(p)
            atomic_json(out/'public-settings.json',{'LLM_kwargs':{k:sorted(v) if isinstance(v,set) else v for k,v in kwargs.items()},
                'request_mm_processor_kwargs':p.get('request_mm_processor_kwargs',{}),
                'request_mm_processor_kwargs_by_task':p.get('request_mm_processor_kwargs_by_task',{}),
                'chat_template_kwargs':{'enable_thinking':False},'generation_by_task':p.get('choices_by_task',{}),
                'audit_hooks_required_for_inference':False,'runtime_forward_patch':False,
                'runtime_processor_input_bridge':bool(p.get('request_adapter')),
                'request_adapter':p.get('request_adapter')})
            engine=LLM(**kwargs);owned=engine.collective_rpc(lifecycle._worker_identity)
            atomic_json(out/'engine-owner.json',{'owned':owned});stack.callback(lifecycle.shutdown_owned_engine,engine,owned,out)
            runtime=engine.collective_rpc(worker_identity,args=(p['model'],))
            atomic_json(out/'runtime.json',runtime)
            require(all(x['actual_kv_capacity']>=p.get('minimum_kv_capacity',65536) for x in runtime),'Actual KV capacity too small')
            atomic_json(out/'load.json',{'seconds':time.monotonic()-start,'native_eos':engine.get_tokenizer().eos_token_id})
            adapter=load_hook(p['request_adapter'])(p,engine) if p.get('request_adapter') else None
            if adapter is not None: stack.callback(adapter.close)
            for ref in p['media_files'].values():
                require(stat(ref['path'])==ref['stat'],'Source changed after sealing')
                with Path(ref['path']).open('rb') as f:
                    while f.read(8*1024**2): pass
            atomic_json(out/'cache-policy.json',{'source_file_page_cache_warmed':True,'decoded_cache':False,
                'prefix_cache':False,'fresh_engine':True,'mm_processor_cache_gb':kwargs.get('mm_processor_cache_gb')})
            obs=stack.enter_context(NativeObserver(p))
            try:
                for ordinal,sid in enumerate(p['assigned_ids']):
                    if boundary.check(): break
                    item=catalog['items'][sid];deep=item['task'] not in seen;seen.add(item['task']);pending=sid
                    for media in media_entries(p,sid,item):
                        if media['type'] in ('video','image'):
                            ref=p['media_files'][media['path']];require(stat(ref['path'])==ref['stat'],'Source changed')
                    write_new(Path(p['claim_root'])/(sid+'.json'),{'sample_id':sid,'profile_sha256':digest(p),
                        'worker_token':p['worker_token'],'attempt':1})
                    _append(out/'events.jsonl',{'event':'INTENT','sample_id':sid,'ordinal':ordinal,'deep':deep})
                    obs.clear();obs.active=True;obs.deep=deep;obs.contract=video_contract(p,item)
                    if deep:
                        install=engine.collective_rpc(install_probe,args=(p['vision_observation'],p['model']))
                        _append(out/'deep-install.jsonl',{'sample_id':sid,'observations':install})
                    choice=choices(p,item);key=tuple(choice)
                    if key not in sampling_cache: sampling_cache[key]=generation(choice)
                    sampling,gen=sampling_cache[key];request=messages(p,sid,item);prepared=None
                    request_kwargs=p.get('request_mm_processor_kwargs_by_task',{}).get(item['task'],p.get('request_mm_processor_kwargs',{}))
                    t0=time.monotonic();extra={'mm_processor_kwargs':request_kwargs} if request_kwargs else {}
                    if adapter is None:
                        answer=engine.chat(request,sampling_params=sampling,use_tqdm=False,
                            chat_template_kwargs={'enable_thinking':False},**extra)[0]
                    else:
                        prepared=adapter.prepare({**item,'sample_id':sid},request[0]['content'])
                        answer=engine.generate(prepared['request'],sampling,use_tqdm=False)[0]
                    request_seconds=time.monotonic()-t0;output=answer.outputs[0];eligible=item.get('accuracy_eligible',item['row'].get('accuracy_eligible',item['row'].get('truth') is not None))
                    raw={'sample_id':sid,'task':item['task'],'model':p['model'],'variant':'OFFICIAL_X','ordinal':ordinal,
                        'deep':deep,'attempt':1,'truth':item['row'].get('truth'),'accuracy_eligible':eligible,
                        'raw_answer':output.text,'correct':output.text.strip()==item['row']['truth'] if eligible else None,
                        'output_token_ids':list(output.token_ids),'finish_reason':output.finish_reason,'stop_reason':output.stop_reason,
                        'actual_prompt_token_ids':list(answer.prompt_token_ids),'question':item['row']['prompt'],
                        'request':request,'generation':gen,'request_mm_processor_kwargs':request_kwargs,
                        'profile_sha256':digest(p),'source':{'catalog':p['source_catalog'],'item_digest':digest(item),
                            'media':[p['media_files'][m['path']] for m in media_entries(p,sid,item) if m['type'] in ('video','image')]},
                        'native_reader':obs.reads or (prepared or {}).get('receipt',{}).get('native_reader',[]),
                        'native_processor':obs.processors or (prepared or {}).get('receipt',{}).get('native_processor',[]),
                        'request_adapter_receipt':(prepared or {}).get('receipt'),
                        'request_adapter_timing':(prepared or {}).get('timing'),
                        'timing':{'engine_request_inclusive_seconds':request_seconds,
                            'native_reader_nested_seconds':obs.read_seconds,'native_processor_nested_seconds':obs.processor_seconds}}
                    _append(out/'raw.jsonl',raw)
                    evidence=engine.collective_rpc(read_probe,args=(p['model'],)) if deep else None
                    if deep:
                        # Persist before every assertion, including dtype/geometry checks.
                        _append(out/'deep-observations.jsonl',{'sample_id':sid,'raw_digest':digest(raw),'observations':evidence})
                    require(output.finish_reason=='stop' and output.text.strip() in choice,'Unexpected scientific completion; raw retained')
                    require(raw['native_reader'],'No actual source reader observation')
                    validate_processor(p,item,raw['native_processor'])
                    if deep: validate_deep(p,item,evidence)
                    if adapter is not None: adapter.validate(answer)
                    _append(out/'events.jsonl',{'event':'TECHNICAL_PASS','sample_id':sid,'ordinal':ordinal,
                        'raw_digest':digest(raw),'deep':deep})
                    timing={**raw['timing'],'sample_id':sid,'task':item['task'],'deep':deep,
                            'end_to_end_durable_seconds':time.monotonic()-t0,'nested_not_additive':True}
                    _append(out/'request-timings.jsonl',timing);completed.append(sid);pending=None
                    atomic_json(out/'progress.json',{'status':'RUNNING','completed':len(completed),'expected':len(p['assigned_ids']),
                        'last':sid,'last_timing':timing})
                    if adapter is not None: adapter.clear()
                    del answer,raw,evidence,prepared
                    if (ordinal+1)%8==0: gc.collect()
            except BaseException as error:
                atomic_json(out/'failure.json',{'error_type':type(error).__name__,'error':str(error),
                    'UNCERTAIN_id':pending,'completed':len(completed),'automatic_retry':False,
                    'native_reader':obs.reads,'native_processor':obs.processors})
                raise
            finally:
                obs.active=False
                if adapter is not None: adapter.clear()
    result={'status':'COMPLETE' if completed==p['assigned_ids'] else 'PAUSED','completed_ids':completed,
        'UNCERTAIN_ids':[pending] if pending else [],'unattempted_ids':[s for s in p['assigned_ids'] if s not in completed and s!=pending],
        'automatic_retry':False,'formal_auto_resume':False}
    atomic_json(out/'result.json',result);return result


def supervise(profile_path):
    from . import refinement_worker as worker
    native=worker.subprocess.Popen
    def popen(argv,**kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv],**kwargs)
    worker.subprocess.Popen=popen
    try: return worker.supervise(profile_path)
    finally: worker.subprocess.Popen=native


def export(profile_path,output_path):
    p=read(profile_path);catalog=verified(p['source_catalog']);sid=p['assigned_ids'][0];item=catalog['items'][sid]
    _,gen=generation(choices(p,item))
    kwargs=engine_kwargs(p)
    value={'model':p['model'],'LLM_kwargs':{k:sorted(v) if isinstance(v,set) else v for k,v in kwargs.items()},
        'messages_example':messages(p,sid,item),'request_mm_processor_kwargs':p.get('request_mm_processor_kwargs',{}),
        'chat_template_kwargs':{'enable_thinking':False},'generation':gen,'bootstrap':p.get('bootstrap'),
        'request_adapter':p.get('request_adapter'),
        'audit_and_ledger_required_for_plain_inference':False,'legacy_ModelSession_required':False}
    write_new(output_path,value);return {'status':'EXPORTED','output':descriptor(output_path),'model_calls':0}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['seal','run','supervise','queue','export'])
    a.add_argument('--draft');a.add_argument('--output');a.add_argument('--profile');a.add_argument('--plan');args=a.parse_args()
    if args.command=='seal': value=seal(args.draft,args.output)
    elif args.command=='run': value=run(args.profile)
    elif args.command=='supervise': value=supervise(args.profile)
    elif args.command=='export': value=export(args.profile,args.output)
    else:
        plan=read(args.plan);require(plan['status']=='SEALED' and plan.get('automatic_retry') is False,'Wrong finite queue')
        for profile in plan['profiles']:
            result=supervise(profile);print(json.dumps(result),flush=True)
            require(result['returncode']==0 and result['terminal_status']=='COMPLETE'
                    and result['real_wait_observed'] and result['owned_engines_exited']
                    and result['original_physical_lease_released'],'Predecessor did not close COMPLETE; stop queue')
        value={'status':'COMPLETE','profiles':plan['profiles']}
    print(json.dumps(value))


if __name__=='__main__': main()
