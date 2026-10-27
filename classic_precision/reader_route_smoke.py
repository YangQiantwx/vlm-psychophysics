"""Finite paired reader experiment; independent journals, no retries or resume.

Uses the qualified loaders, GPU leases, QSA, Gemma adapter and shutdown helpers.
Only this module and experiment JSON are new; no shared SDK/model math is edited.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from .storage import atomic_json, _append, digest, file_sha256, owner_identity
from .refinement_worker import require, read, write_new, descriptor, verified, physical_lease, Boundary

NAME = 'reader-route-smoke-20261026'
MEDIA = {'video': {'video_backend': 'opencv', 'num_frames': 180, 'fps': -1}}


def metadata():
    return {'fps': 60., 'duration': 3., 'total_num_frames': 180,
            'frames_indices': list(range(180)), 'do_sample_frames': False}


def state(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def seal(root, host):
    """Bind immutable sources/config and qualified existing host environment, CPU only."""
    root = Path(root); plan = root/'plans'/NAME
    source = read(plan/'sources.json'); alloc = read(plan/'allocation.json')
    cards = ['fourier0', 'fourier1'] if host == 'fourier' else ['radon0']
    old = root/'plans/classic-refinement'
    gname = 'gemma-fourier0' if host == 'fourier' else 'gemma-radon0'
    gemma = read(old/(gname+'-profile.json'))
    anchor = verified(gemma['qualified_anchor'])
    roots = anchor['source_roots']
    files = {}
    for sid, item in source['items'].items():
        path = Path(roots[item['source_family']])/item['ready']['video_relative']
        require(file_sha256(path) == item['ready']['video_sha256'], 'Sealed source differs: '+sid)
        files[sid] = {'path': str(path), 'stat': state(path)}
    queues = {}
    for card in cards:
        gpu = read(old/((('gemma-'+card) if card != 'radon0' else 'nemo-radon0-v2')+'-profile.json'))
        profiles = []
        for segment in alloc['cards'][card]:
            model = segment['model']
            if model == 'gemma': base = gemma
            elif model == 'qwen': base = read(old/'qwen-fourier0-profile.json')
            elif model == 'nemo': base = read(old/'nemo-radon0-v2-profile.json')
            elif host == 'fourier': base = read(old/'cosmos-fourier1-profile.json')
            else:
                retained = root/'runs/reproducibility/operator/cosmos-t1t2-nvfp4-radon-sealed-20261025.json'
                prior = read(retained)
                env = dict(gemma['environment'])
                cleared = []
                for k, v in prior['runtime_identity']['environment'].items():
                    if v is None: env.pop(k, None); cleared.append(k)
                    else: env[k] = v
                base = {'model': 'cosmos', 'checkpoint': prior['runtime']['checkpoint'],
                        'runtime': prior['runtime'], 'runtime_identity': prior['runtime_identity'],
                        'resources': prior['runtime']['resources'], 'environment': env,
                        'cleared_environment': cleared, 'qualified_anchor': descriptor(retained)}
            # Retain all scientific/resource settings and their original qualified binding.
            keep = ['checkpoint', 'checkpoint_assets', 'runtime', 'runtime_identity', 'resources',
                    'environment', 'cleared_environment', 'qualified_profile', 'qualified_anchor',
                    'engine_settings', 'kernel', 'runtime_sources', 'checkpoint_contract',
                    'cpu_execution_overlay', 'entry_implementation']
            inherited = {k: base[k] for k in keep if k in base}
            env = dict(inherited['environment']); env['CUDA_VISIBLE_DEVICES'] = card[-1]
            inherited['environment'] = env
            for arm in segment['arms']:
                worker = model+'-'+card+'-'+arm
                out = root/'runs/reproducibility'/NAME/worker
                profile = {**inherited, 'schema': 'reader-route-worker-v1', 'model': model,
                    'worker': worker, 'hostname': socket.gethostname(), 'arm': arm,
                    'worker_token': digest({'allocation': descriptor(plan/'allocation.json'), 'worker': worker}),
                    'assigned_ids': segment['sample_ids'], 'maximum_model_calls': len(segment['sample_ids']),
                    'automatic_retry': False, 'formal_auto_resume': False,
                    'source_catalog': descriptor(plan/'sources.json'),
                    'allocation': descriptor(plan/'allocation.json'), 'experiment': descriptor(plan/'plan.json'),
                    'sources': {s: files[s] for s in segment['sample_ids']},
                    'gpu_uuid': gpu['gpu_uuid'], 'physical_lock': gpu['physical_lock'],
                    'output': str(out), 'claim_root': str(out.parent/'claims'/model/arm),
                    'pause_file': str(plan/(worker+'-PAUSE.json')),
                    'implementation': descriptor(__file__),
                    'qualified_host_profile': base.get('profile_file', base.get('qualified_anchor')),
                    'route_label': ('official uint8 + shared spatial/input adapter; HF video processor, '
                                    '280 budget, native post-processor BF16; not unmodified vLLM file route')
                                    if model == 'gemma' and arm == 'C' else arm}
                path = plan/(worker+'-profile.json'); write_new(path, profile); profiles.append(str(path))
        queue = plan/(card+'-queue.json')
        write_new(queue, {'status': 'SEALED', 'profiles': profiles, 'automatic_retry': False,
                          'formal_auto_resume': False})
        queues[card] = str(queue)
    return {'status': 'SEALED', 'queues': queues, 'source_files_verified': len(files), 'model_calls': 0}


def decode(path, arm, gemma=False):
    """Same qualified RGB16 numeric order; B differs solely at the reader boundary."""
    import numpy as np
    start = time.monotonic(); cast_seconds = 0.
    if arm == 'A':
        import av
        pts = []; chunks = []
        frames = np.empty((180, 528, 768, 3), np.float32) if gemma else None
        with av.open(str(path)) as container:
            stream = container.streams.video[0]; stream.codec_context.thread_count = 1
            require(stream.codec_context.name == 'ffv1' and stream.codec_context.format.name == 'gbrp16le', 'RGB16 codec differs')
            for i, frame in enumerate(container.decode(video=0)):
                require(i < 180, 'Extra frame')
                data = frame.to_ndarray(format='rgb48le')
                require(data.dtype == np.uint16 and data.shape == (512,768,3), 'RGB16 dimensions differ')
                t = time.monotonic()
                if gemma:
                    if i == 0: padding = data[0,0].astype(np.float32)
                    frames[i,:512], frames[i,512:] = data, padding
                    np.multiply(frames[i], 255./65535., out=frames[i])
                else: chunks.append(data)
                cast_seconds += time.monotonic()-t
                pts.append(float(frame.pts*frame.time_base))
        require(len(pts) == 180 and max(abs(t-i/60) for i,t in enumerate(pts)) <= .001, 'Frame count/PTS differs')
        t = time.monotonic()
        if not gemma: frames = np.stack(chunks).astype(np.float32)*(255./65535.)
        cast_seconds += time.monotonic()-t
        receipt = {'reader': 'PyAV RGB48/gbrp16le', 'reader_dtype': 'uint16', 'reader_shape': [180,512,768,3],
                   'frames_indices': list(range(180)), 'source_pts_s': pts}
    else:
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.media.image import ImageMediaIO
        loader = VideoMediaIO(ImageMediaIO(), **MEDIA['video'])
        result = loader.load_file(Path(path)); values, meta = result.media
        require(values.dtype == np.uint8 and values.shape == (180,512,768,3)
                and meta['frames_indices'] == list(range(180)), 'Official reader changed frames/geometry')
        t = time.monotonic()
        frames = values.astype(np.float32) if arm == 'B' else values
        if gemma:
            padded = np.empty((180,528,768,3), frames.dtype)
            padded[:,:512] = frames; padded[:,512:] = frames[0,0,0]
            frames = padded
        cast_seconds += time.monotonic()-t
        receipt = {'reader': 'vLLM VideoMediaIO/OpenCV', 'reader_dtype': str(values.dtype),
                   'reader_shape': list(values.shape), 'frames_indices': meta['frames_indices'],
                   'official_metadata': meta}
    total = time.monotonic()-start
    receipt.update(submitted_shape=list(frames.shape), submitted_dtype=str(frames.dtype),
                   effective_geometry=[768,512], padding_bottom_rows=16 if gemma else 0,
                   padding_RGB=frames[0,0,0].tolist() if gemma else None)
    return frames, receipt, {'read_decode_seconds': total-cast_seconds,
                             'cast_pad_adapt_seconds': cast_seconds}


def prompt(session, item):
    impl = getattr(session, '_implementation', session)
    formatter = impl.tokenizer if impl.family == 'nemo' else impl.processor
    text = formatter.apply_chat_template([{'role':'user','content':[
        {'type':'video'}, {'type':'text','text':item['row']['prompt']}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require(not text.rstrip().endswith('<think>'), 'Thinking unexpectedly enabled')
    return text


@contextmanager
def gemma_processor_views():
    """Reuse the existing qualified singleton-stack optimization, no numerical change."""
    from .gemma_cpu_copy import installed_processor_source
    from transformers.models.gemma4 import video_processing_gemma4 as mod
    code, record = installed_processor_source(); ns = dict(vars(mod)); exec(code, ns)
    original = mod.Gemma4VideoProcessor._preprocess
    mod.Gemma4VideoProcessor._preprocess = ns['_preprocess']
    try: yield record
    finally: mod.Gemma4VideoProcessor._preprocess = original


def gemma_prepare(session, frames, text):
    """All arms share spatial adapter; C feeds official uint8 directly to HF processor."""
    import torch
    from transformers.video_utils import VideoMetadata
    t = time.monotonic()
    data = session.processor(text=[text], videos=[frames], videos_kwargs={
        'do_resize':False, 'do_sample_frames':False, 'max_soft_tokens':280,
        'video_metadata':[VideoMetadata(total_num_frames=180,fps=60.,duration=3.,frames_indices=list(range(180)))]},
        return_tensors='pt')
    processor_seconds = time.monotonic()-t; t = time.monotonic()
    positions = data[next(k for k in data if 'position_ids' in k and 'video' in k)][0]
    native_pixels = data['pixel_values_videos'][0]
    require(native_pixels.shape == (180,2520,768) and native_pixels.dtype == torch.float32, 'Gemma processor geometry/dtype')
    pixels = native_pixels.to(torch.bfloat16)  # Qualified A/B order, retained for C shared adapter.
    soft = (((positions!=-1).any(-1).sum(-1))//session.processor.video_processor.pooling_kernel_size**2).tolist()
    require(len(soft)==180 and set(soft)=={176}, 'Actual Gemma soft-token positions differ')
    ids = data['input_ids'][0].tolist()
    payload = {'pixel_values_videos':pixels, 'pixel_position_ids_videos':positions,
        'video_frame_counts':torch.tensor([180]), 'video_num_soft_tokens':[soft],
        'video_timestamps':[[i/60. for i in range(180)]]}
    return payload, {'processor_output_shape':list(native_pixels.shape), 'processor_output_dtype':str(native_pixels.dtype),
        'bridge_pixel_dtype':str(pixels.dtype), 'position_shape':list(positions.shape),
        'effective_visual_tokens_per_frame':soft, 'visual_tokens_total':sum(soft), 'configured_budget':280,
        'expected_prompt_token_ids':ids, 'post_processor_BF16':True}, {
        'processor_external_seconds':processor_seconds, 'post_processor_cast_payload_seconds':time.monotonic()-t}


class Events:
    def __init__(self, out): self.out = self.directory = out
    def event(self, event, **fields): _append(self.out/'events.jsonl', {'event':event, **fields})


class NativeObservation:
    """Observe existing reader and processor calls; nested times are explicitly inclusive."""
    def __init__(self, model):
        self.model = model; self.active = False; self.clear()
    def clear(self): self.reads=[]; self.processors=[]; self.read_seconds=0.; self.processor_seconds=0.
    def describe(self, data):
        import torch
        def desc(v):
            if isinstance(v, torch.Tensor):
                d={'shape':list(v.shape),'dtype':str(v.dtype)}
                if not v.is_floating_point() and v.numel() <= 600: d['values']=v.tolist()
                return d
            if isinstance(v,(list,tuple)): return [desc(x) for x in v[:4]]
            return type(v).__name__
        return {k:desc(v) for k,v in data.items() if 'pixel' in k or 'grid' in k
                or k in ('frames_indices','video_num_patches','video_frame_counts')}
    def __enter__(self):
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.processing.context import InputProcessingContext
        self.original_read=VideoMediaIO.load_file
        self.original_call=InputProcessingContext.call_hf_processor
        self.original_post=InputProcessingContext._postprocess_output
        obs=self
        def load(loader, path):
            t=time.monotonic(); result=obs.original_read(loader,path); elapsed=time.monotonic()-t
            if obs.active:
                frames, meta=result.media
                require(str(frames.dtype)=='uint8' and frames.shape==(180,512,768,3)
                        and meta['frames_indices']==list(range(180)), 'Native file reader frames/geometry')
                obs.read_seconds += elapsed
                obs.reads.append({'path':str(path),'shape':list(frames.shape),'dtype':str(frames.dtype),'metadata':meta})
            return result
        def call(ctx,hf_processor,data,kwargs={}):
            t=time.monotonic(); result=obs.original_call(ctx,hf_processor,data,kwargs)
            if obs.active:
                obs.processor_seconds += time.monotonic()-t
                if obs.model=='nemo' and 'video_num_patches' in result:
                    require(result['video_num_patches'].tolist()==[180], 'Nemo lost frames')
                elif obs.model in ('qwen','cosmos') and 'video_grid_thw' in result:
                    require(result['video_grid_thw'].tolist()==[[90,32,48]], 'Native resized/sampled')
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


@contextmanager
def file_loader_settings(root):
    """Add only native reader settings to the same qualified loader in every arm."""
    import vllm
    original=vllm.LLM
    def create(**kwargs):
        kwargs.update(media_io_kwargs=MEDIA, allowed_local_media_path=str(root))
        return original(**kwargs)
    vllm.LLM=create
    try: yield
    finally: vllm.LLM=original


def warm(profile, out):
    start=time.monotonic(); total=0
    for ref in profile['sources'].values():
        require(state(ref['path'])==ref['stat'], 'Sealed source stat changed')
        with Path(ref['path']).open('rb') as f:
            for block in iter(lambda:f.read(8*1024**2),b''): total+=len(block)
    atomic_json(out/'cache-policy.json', {'source_bytes_read':total,'page_cache_warm_seconds':time.monotonic()-start,
        'decoded_cache':False,'prefix_cache':False,'mm_processor_cache_gb':0,'fresh_engine_for_each_arm':True,
        'scope':'same source-file page-cache warm before each arm; outside request timing; no global cache drop'})


def cpu(profile_path):
    p=read(profile_path); require(p['model']=='gemma','CPU qualification currently Gemma shared adapter')
    from .inference import ProcessorSession
    import torch
    catalog=verified(p['source_catalog']); records=[]
    with ProcessorSession('gemma',p['runtime']['processor_checkpoint']) as session, gemma_processor_views():
        seen=set()
        for sid in p['assigned_ids']:
            item=catalog['items'][sid]
            if item['task'] in seen: continue
            seen.add(item['task'])
            frames, receipt, timing=decode(p['sources'][sid]['path'],p['arm'],True)
            payload, controls, phases=gemma_prepare(session,frames,prompt(session,item))
            require(not torch.cuda.is_initialized(),'CPU qualification initialized CUDA')
            template=catalog['templates'][item['template_key']]['controls']
            require(controls['expected_prompt_token_ids']==template['input_token_ids'],'Gemma prompt layout differs from qualified source template')
            records.append({'sample_id':sid,'task':item['task'],'reader':receipt,
                'controls':{k:v for k,v in controls.items() if k!='expected_prompt_token_ids'},
                'prompt_tokens':len(controls['expected_prompt_token_ids']),'timing':{**timing,**phases},'GPU_calls':0})
            del frames,payload;gc.collect()
    result={'status':'PASS','arm':p['arm'],'route_label':p['route_label'],'rows':records,
            'model_calls':0,'model_weights_loaded':False,'CUDA_initialized':False,'profile_sha256':digest(p)}
    write_new(Path(profile_path).with_suffix('.cpu.json'),result)
    return {'status':'PASS','arm':p['arm'],'CPU_rows':len(records),'model_calls':0}


def run(profile_path):
    p=read(profile_path); out=Path(p['output']); model=p['model']; arm=p['arm']
    require(socket.gethostname()==p['hostname'] and not out.exists(),'Wrong host or attempted implicit resume')
    require(file_sha256(__file__)==p['implementation']['sha256'],'Worker changed after sealing')
    for k,v in p['environment'].items():require(os.environ.get(k)==v,'Environment differs: '+k)
    for k in p.get('cleared_environment',[]):require(k not in os.environ,'Unexpected environment: '+k)
    catalog=verified(p['source_catalog']);verified(p['allocation']);verified(p['experiment'])
    require(len(set(p['assigned_ids']))==p['maximum_model_calls'] and not p['automatic_retry'],'Finite quota changed')
    require(not any((Path(p['claim_root'])/(s+'.json')).exists() for s in p['assigned_ids']),'Already claimed scientific tuple; reconcile without retry')
    if model=='gemma':
        check=read(Path(profile_path).with_suffix('.cpu.json'))
        require(check['status']=='PASS' and check['profile_sha256']==digest(p),'Gemma CPU path not qualified')
    # Activate exactly the retained QSA startup policy before importing torch/vLLM.
    qsa=None
    if model=='qwen':
        from . import qsa_only_reproducibility as qsa
        qsa.runtime.activate('qwen','qwen_qsa_only',p['qualified_profile']['baseline_environment'])
        qsa._validate_profile(qsa.runtime.status(),'qwen',require_applied=False)
    from . import inference, gemma_speed as g
    from .refinement_dispatch import compiler_preflight
    if p['environment'].get('FLASHINFER_NVCC'): compiler_preflight(p['environment'])
    import torch
    out.mkdir(parents=True);sink=Events(out);completed=[];pending=None;engine=None;owned=[]
    atomic_json(out/'identity.json',{'profile':p,'profile_sha256':digest(p),'owner':owner_identity(),'python':sys.executable})
    boundary=Boundary(p)
    for sig in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM):signal.signal(sig,boundary.signal)
    with physical_lease(p,torch) as lease:
        atomic_json(out/'lease.json',lease)
        with ExitStack() as stack:
            stack.enter_context(file_loader_settings(Path(profile_path).parents[2]))
            started=time.monotonic()
            if model=='gemma':
                session=stack.enter_context(inference.ProcessorSession('gemma',p['runtime']['processor_checkpoint']))
                stack.enter_context(gemma_processor_views());g._install_adapter(session,True)
                from vllm import LLM
                engine=LLM(**p['engine_settings'])
            elif model=='cosmos':
                from .nvfp4_cosmos_official import OfficialSession,runtime_identity
                require(runtime_identity(p['runtime'])==p['runtime_identity'],'Qualified Cosmos runtime changed')
                session=OfficialSession(p['runtime'],sink);engine=session.engine
            else:
                require(inference.checkpoint_asset_identity(p['checkpoint'])==p['checkpoint_assets'],'Checkpoint changed')
                session=inference.ModelSession(model,p['checkpoint'],sink,resources=p['resources']);engine=session.engine
            owned=engine.collective_rpc(g._worker_identity)
            atomic_json(out/'engine-owner.json',{'owned':owned})
            # Register cleanup before any post-load validation can fail.
            stack.callback(g.shutdown_owned_engine,engine,owned,out)
            if model!='gemma':
                def close_session(): session.engine=None;session.close()
                stack.callback(close_session)
            if model=='gemma':
                from .nvfp4_gemma import _placement
                placement=engine.collective_rpc(_placement,kwargs={'offload_gb':0})
                kernel=engine.collective_rpc(g._install_kernel,args=(p['kernel'],str(out)))
                engine.collective_rpc(g._install_observer)
            else: placement=kernel=None
            if qsa:
                qsa.runtime.prepare_native('qwen');snapshot=qsa.runtime.runtime_snapshot()
                qsa.validate_runtime_policy(snapshot,'qwen');atomic_json(out/'qsa-startup.json',snapshot)
            atomic_json(out/'load.json',{'seconds':time.monotonic()-started,'placement':placement,'kernel':kernel,
                'native_eos':engine.get_tokenizer().eos_token_id,'route_label':p['route_label']})
            from vllm import SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            sampling=SamplingParams(n=1,temperature=0.,top_p=1.,seed=0,repetition_penalty=1.,max_tokens=16384,
                structured_outputs=StructuredOutputsParams(choice=['LEFT','RIGHT']))
            warm(p,out);seen=set()
            observation=stack.enter_context(NativeObservation(model))
            try:
                for ordinal,sid in enumerate(p['assigned_ids']):
                    if boundary.check():break
                    item=catalog['items'][sid];ref=p['sources'][sid]
                    require(state(ref['path'])==ref['stat'],'Source changed after sealing')
                    observed=item['task'] not in seen;seen.add(item['task']);pending=sid
                    write_new(Path(p['claim_root'])/(sid+'.json'),{'sample_id':sid,'worker':p['worker'],'profile_sha256':digest(p)})
                    _append(out/'events.jsonl',{'event':'INTENT','sample_id':sid,'ordinal':ordinal,'observed':observed})
                    observation.clear();observation.active=True
                    if model=='gemma':engine.collective_rpc(g._set_deep,args=(False,))
                    elif observed:engine.collective_rpc(inference._install_vision_probe)
                    t0=time.monotonic();phases={};receipt={};controls={};frames=payload=None
                    text=prompt(session,item)
                    if arm!='C' or model=='gemma':
                        frames,receipt,phases=decode(ref['path'],arm,model=='gemma')
                        if model=='gemma':
                            payload,controls,more=gemma_prepare(session,frames,text);phases.update(more)
                            g.CTX.update(active=True,frames=frames,payload=payload,adapter_calls=0)
                        request={'prompt':text,'multi_modal_data':{'video':[(frames,metadata())]}}
                    else:
                        request=[{'role':'user','content':[{'type':'video_url','video_url':{'url':Path(ref['path']).as_uri()}},
                            {'type':'text','text':item['row']['prompt']}]}]
                    ts=time.monotonic()
                    if arm=='C' and model!='gemma':
                        answer=engine.chat(request,sampling_params=sampling,use_tqdm=False,
                            chat_template_kwargs={'enable_thinking':False})[0]
                    else:answer=engine.generate(request,sampling,use_tqdm=False)[0]
                    engine_seconds=time.monotonic()-ts;choice=answer.outputs[0];tp=time.monotonic()
                    raw={'sample_id':sid,'task':item['task'],'model':model,'arm':arm,'ordinal':ordinal,
                        'observed_or_cold':observed,'truth':item['row']['truth'],'raw_answer':choice.text,
                        'correct':choice.text.strip()==item['row']['truth'],'output_token_ids':list(choice.token_ids),
                        'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason,
                        'actual_prompt_token_ids':list(answer.prompt_token_ids),'question':item['row']['prompt'],
                        'controlled_template_text':text if arm!='C' or model=='gemma' else None,
                        'profile_sha256':digest(p),'source':{**ref,'video_sha256':item['ready']['video_sha256'],
                            'stream_sha256':item['source_stream_sha256'],'source_parameters':item['row']['source_parameters'],
                            'row_sha256':item['row_sha256'],'ready_sha256':item['ready_sha256']},
                        'reader':receipt,'input_controls':controls,'native_reader':observation.reads,
                        'native_processor':observation.processors if observed else [],
                        'timing':{**phases,'engine_request_inclusive_seconds':engine_seconds,
                            'native_reader_nested_seconds':observation.read_seconds,
                            'native_processor_nested_seconds':observation.processor_seconds},
                        'route_label':p['route_label'],'vllm_metrics':str(answer.metrics)}
                    # Durable raw precedes every post-output assertion/probe. Never retry it.
                    _append(out/'raw.jsonl',raw)
                    require(choice.finish_reason=='stop' and choice.text.strip() in ['LEFT','RIGHT'],'Unexpected scientific completion; raw retained')
                    require(len(answer.prompt_token_ids)+16384<=65536,'Context budget differs')
                    if model=='gemma':
                        gpu=engine.collective_rpc(g._read_observer)
                        require(list(answer.prompt_token_ids)==controls['expected_prompt_token_ids'],'Gemma actual prompt expansion differs')
                        require(g.CTX['adapter_calls']>=1,'Shared adapter did not run')
                        g.validate_gpu(gpu,{},False);g.CTX.clear()
                    elif observed:
                        gpu=engine.collective_rpc(inference._read_vision_probe)
                        inference._validate_forwards(model,[x for worker in gpu for x in worker])
                    else:gpu=None
                    if model!='gemma':require(observation.processors,'No actual native processor observed')
                    if arm=='C' and model!='gemma':require(observation.reads,'No actual native file reader observed')
                    if qsa and ordinal==0:
                        from .native_path_smoke import qsa_control
                        applied=engine.collective_rpc(qsa_control)
                        require(all(x['state']=='applied' for x in applied),'QSA not actually active')
                        atomic_json(out/'qsa-first-output.json',applied)
                    proof={'event':'TECHNICAL_PASS','sample_id':sid,'raw_digest':digest(raw),
                        'actual_GPU_input':gpu,'observed_or_cold':observed}
                    _append(out/'events.jsonl',proof)
                    td=time.monotonic()
                    timing={**raw['timing'],'sample_id':sid,'task':item['task'],'observed_or_cold':observed,
                        'validation_serialization_persistence_seconds':td-tp,'end_to_end_durable_seconds':td-t0,
                        'scope':'read start through durable raw and technical-pass; final timing/progress write excluded',
                        'nested_stages':['native_reader_nested_seconds','native_processor_nested_seconds'],
                        'bridge_IPC_transfer_seconds':None,'unseparated_engine_scope':'native processor, bridge/IPC/H2D, model and detokenization; nested stages are NOT additive',
                        'CUDA_existing_observer':gpu[0]['CUDA_seconds'] if model=='gemma' else None}
                    _append(out/'request-timings.jsonl',timing)
                    completed.append(sid);pending=None
                    atomic_json(out/'progress.json',{'status':'RUNNING','completed':len(completed),
                        'expected':len(p['assigned_ids']),'last':sid,'last_timing':timing})
                    del frames,payload,answer,raw
                    if (ordinal+1)%8==0:gc.collect()
            except BaseException as error:
                atomic_json(out/'failure.json',{'error_type':type(error).__name__,'error':str(error),
                    'UNCERTAIN_id':pending,'completed':len(completed),'automatic_retry':False})
                raise
            finally:
                observation.active=False;g.CTX.clear()
    result={'status':'COMPLETE' if completed==p['assigned_ids'] else 'PAUSED', 'completed_ids':completed,
            'unattempted_ids':[s for s in p['assigned_ids'] if s not in set(completed) and s!=pending],
            'UNCERTAIN_ids':[pending] if pending else [],'automatic_retry':False,'formal_auto_resume':False}
    atomic_json(out/'result.json',result)
    if model=='gemma': atomic_json(out/'continuation.json',result)
    return result


def supervise(profile_path):
    from .refinement_worker import supervise as original
    # Reuse actual child.wait, owned PID/startticks validation and native same-inode lease check.
    import classic_precision.refinement_worker as worker
    original_popen=worker.subprocess.Popen
    def popen(argv,**kwargs):
        argv=[x.replace('classic_precision.refinement_worker','classic_precision.reader_route_smoke') for x in argv]
        return original_popen(argv,**kwargs)
    worker.subprocess.Popen=popen
    try:return original(profile_path)
    finally:worker.subprocess.Popen=original_popen


def queue(queue_path):
    q=read(queue_path);require(q['status']=='SEALED','Existing queue requires explicit reconciliation, no retry')
    q['status']='RUNNING';atomic_json(queue_path,q);reports=[]
    try:
        for path in q['profiles']:
            p=read(path);require(not Path(p['pause_file']).exists(),'Queued smoke profile paused')
            child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.reader_route_smoke','supervise','--profile',path])
            code=child.wait();w=read(Path(p['output']).with_suffix('.wait.json'));reports.append(w)
            require(code==0 and w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited']
                    and w['original_physical_lease_released'],'Actual closure failed; no retry')
            require(w['terminal_status']=='COMPLETE' and w['completed_ids']==p['assigned_ids'],'Partial worker; hold successors')
        q.update(status='COMPLETE',reports=reports);atomic_json(queue_path,q)
    except BaseException as error:
        q.update(status='HELD_NO_RETRY',reports=reports,error=str(error));atomic_json(queue_path,q);raise
    return {'status':q['status'],'workers':len(reports)}


def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('command',choices=['seal','cpu','run','supervise','queue'])
    a.add_argument('--profile');a.add_argument('--root');a.add_argument('--host');a.add_argument('--queue')
    args=a.parse_args()
    if args.command=='seal':result=seal(args.root,args.host)
    elif args.command=='queue':result=queue(args.queue)
    else:result=globals()[args.command](args.profile)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
