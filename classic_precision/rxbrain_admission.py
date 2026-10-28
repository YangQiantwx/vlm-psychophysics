"""Bounded official RxBrain VQA admission; no retries or formal continuation.

The official ordered-image prompt/processor and model forward are unchanged.
The decoder follows upstream inference/vqa.py, adding only the benchmark's
LEFT/RIGHT language constraint, provenance, finite claims and durable records.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .embodied_admission import (append, decode, new, read, require, tensor_sha,
                                allowed_tokens)
from .storage import atomic_json, digest, file_sha256

NAME = 'embodied-points-20261027'


def configured(root):
    root = Path(root)
    p = read(root/'plans'/NAME/'rx-plan.json')
    require(p['per_model_maximum_calls'] == 36 and p['cold_starts'] == 2
            and len(p['samples']) == 18 and not p['automatic_retry'], 'Budget drift')
    index = read(root/'datasets/classic-t1-t3-v0.3/dataset-index.json')
    samples = []
    for s in p['samples']:
        entry = index['items'][s['sample_id']]
        require(entry['video_sha256'] == s['source_video_sha256']
                and entry['truth'] == s['truth'] and entry['task'] == s['task'], 'Source drift')
        samples.append({**s, 'prompt': entry['prompt'],
                        'path':str(root/'datasets/classic-t1-t3-v0.3'/s['video_relative'])})
    out = root/'runs/reproducibility'/NAME
    out.mkdir(parents=True, exist_ok=True)
    return p, samples, root/'models/Hy-Embodied-RxBrain-1.0', out


def vendor(root):
    import transformers.models
    root = Path(root)
    upstream = root/'code/vendor/transformers-hunyuan-9293856'
    repo = root/'code/vendor/Hy-Embodied-RxBrain-1.0'
    if str(upstream) not in transformers.models.__path__:
        transformers.models.__path__.append(str(upstream))
    sys.path.insert(0, str(repo))
    paths = list((upstream/'hunyuan_vl_mot').glob('*.py'))
    paths += list((repo/'model').glob('*.py'))
    paths += [repo/'inference/interleave.py',repo/'inference/vqa.py']
    require(len(paths) >= 18, 'Vendor source incomplete')
    return {str(p.relative_to(root)):file_sha256(p) for p in sorted(paths)}


def prepare(processor, item):
    import hashlib
    from PIL import Image
    from inference.interleave import build_conditioned_sequence, INPUT_IMAGE_PLACEHOLDER_IDS
    start = time.monotonic()
    frames = decode(item['path'])
    after_read = time.monotonic()
    ids, proc = build_conditioned_sequence(processor, [Image.fromarray(x) for x in frames], item['prompt'])
    pixels, grid = proc['pixel_values'], proc['image_grid_thw']
    require(grid.tolist() == [[1,32,48]]*180, 'Processor resize/frame drift')
    require(list(pixels.shape) == [276480,768], 'Patch geometry drift')
    placeholders = sum(t in INPUT_IMAGE_PLACEHOLDER_IDS for t in ids)
    require(placeholders == 69120, f'Visual placeholder count {placeholders}')
    receipt = {'reader':'OpenCV sequential BGR-to-RGB uint8',
        'reader_shape':list(frames.shape),'frame_indices':list(range(180)),
        'effective_geometry_wh':[768,512],'route':'180 ordered images, official VQA template; no video timestamps',
        'processor_grid':grid.tolist(),'processor_shape':list(pixels.shape),
        'processor_dtype':str(pixels.dtype),'processor_sha256':tensor_sha(pixels),
        'decoded_sha256':hashlib.sha256(frames.tobytes()).hexdigest(),
        'prompt_token_ids':ids,'prompt_text':processor.tokenizer.decode(ids,skip_special_tokens=False),
        'visual_tokens':placeholders,'read_decode_seconds':after_read-start,
        'processor_and_admission_seconds':time.monotonic()-after_read}
    return ids, proc, receipt


def cpu(root):
    import torch
    identity = vendor(root)
    from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor
    p,samples,model_path,out = configured(root)
    processor = HunYuanVLMoTProcessor.from_pretrained(model_path,local_files_only=True)
    processor.image_processor.do_resize = False
    records=[]
    for item in samples:
        require(file_sha256(item['path']) == item['source_video_sha256'], 'Source bytes changed')
        ids,proc,receipt = prepare(processor,item)
        records.append(dict(receipt,sample_id=item['sample_id']))
        del ids,proc
    require(not torch.cuda.is_initialized(), 'CPU check initialized CUDA')
    new(out/'rx-cpu-admission.json',{'status':'CPU_PASS','model_calls':0,
        'plan_sha256':digest(p),'implementation_sha256':file_sha256(__file__),
        'vendor':identity,'records':records})
    print(json.dumps({'status':'CPU_PASS','samples':len(records),'all180':True}),flush=True)


def generate(model, processor, prompt_ids, proc):
    # Same forward arguments and KV progression as pinned official vqa.answer.
    # The constraint operates on final logits only; no model/processor patch.
    import torch
    from transformers.cache_utils import DynamicCache
    from inference.interleave import INPUT_IMAGE_PLACEHOLDER_IDS
    device = torch.device('cuda')
    eos = model.config.eos_token_id
    require(isinstance(eos,int), 'Expected official scalar native EOS')
    inner = model.model
    pixels = proc['pixel_values'].to(device=device,dtype=torch.bfloat16)
    grid = proc['image_grid_thw'].to(device=device)
    length = len(prompt_ids)
    inp=torch.tensor([prompt_ids],dtype=torch.long,device=device)
    mod=torch.zeros(1,length,dtype=torch.long,device=device)
    iim=torch.zeros(1,length,dtype=torch.bool,device=device)
    for pid in INPUT_IMAGE_PLACEHOLDER_IDS:
        mask=inp[0]==pid;mod[0,mask]=1;iim[0,mask]=True
    empty=torch.zeros((0,2),dtype=torch.int32,device=device)
    prefix=allowed_tokens(processor.tokenizer,[eos],0)
    emitted=[]
    with torch.inference_mode():
        out=inner(input_ids=inp,position_ids=torch.arange(length,device=device).unsqueeze(0),
            pixel_values=pixels,image_grid_thw=grid,
            cu_seqlens=torch.tensor([0,length],dtype=torch.int32,device=device),
            sample_ids=torch.zeros(1,length,dtype=torch.int32,device=device),
            modality_mask=mod,input_image_mask=iim,flow_embeds=None,flow_positions=None,g_seqlens=empty,
            use_cache=True,past_key_values=DynamicCache(),cache_position=torch.arange(length,device=device))
        for _ in range(16384):
            choices=prefix(0,torch.tensor(emitted,dtype=torch.long))
            choice_tensor=torch.tensor(choices,device=device)
            nxt=int(choice_tensor[out.logits[0,-1,choice_tensor].argmax()].item())
            emitted.append(nxt)
            if nxt==eos:
                break
            pkv=out.past_key_values
            del out
            pos=length+len(emitted)-1
            out=inner(input_ids=torch.tensor([[nxt]],dtype=torch.long,device=device),
                position_ids=torch.tensor([[pos]],device=device),pixel_values=None,image_grid_thw=None,
                cu_seqlens=torch.tensor([0,pos+1],dtype=torch.int32,device=device),
                sample_ids=torch.zeros(1,1,dtype=torch.int32,device=device),
                modality_mask=torch.zeros(1,1,dtype=torch.long,device=device),
                input_image_mask=torch.zeros(1,1,dtype=torch.bool,device=device),
                flow_embeds=None,flow_positions=None,g_seqlens=empty,
                use_cache=True,past_key_values=pkv,cache_position=torch.tensor([pos],device=device))
    return emitted, eos


def run(root,cold):
    import random
    import numpy as np
    import torch
    identity=vendor(root)
    from transformers.models.hunyuan_vl_mot import HunYuanVLMoTProcessor
    from model import UnifiedMoTForConditionalGeneration, maybe_init_generation_path
    p,samples,model_path,out=configured(root)
    admission=read(out/'rx-cpu-admission.json')
    binding=read(out/'rx-prelaunch-source-binding.json')
    require(admission['plan_sha256']==digest(p) and admission['vendor']==identity
            and binding['cpu_admission_sha256']==file_sha256(out/'rx-cpu-admission.json')
            and binding['cpu_implementation_sha256']==admission['implementation_sha256']
            and binding['execution_implementation_sha256']==file_sha256(__file__)
            and binding['cpu_functions_unchanged'],'CPU admission stale')
    folder=out/f'rx-cold{cold}'
    require(not folder.exists(),'Attempt exists; no automatic retry')
    folder.mkdir();new(folder/'owner.json',{'pid':os.getpid(),'role':'single-process official HF model'})
    random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
    torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    require(torch.cuda.device_count()==1,'One visible GPU required')
    require(str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-')==
            p['gpu_uuid'].lower().removeprefix('gpu-'),'Wrong physical card')
    lock=p['physical_lock'];path=Path(lock['path'])
    require(not path.is_symlink(),'Symlink lease')
    try:
        with path.open('r+b') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            stat=os.fstat(handle.fileno())
            require((stat.st_dev,stat.st_ino)==(lock['st_dev'],lock['st_ino']),'Original lease changed')
            require(torch.cuda.mem_get_info()[0]>70*1024**3,'Less than70GiB free')
            start=time.monotonic()
            processor=HunYuanVLMoTProcessor.from_pretrained(model_path,local_files_only=True)
            processor.image_processor.do_resize=False
            model=UnifiedMoTForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,
                                                                     local_files_only=True)
            maybe_init_generation_path(model,model_load_path=str(model_path))
            model.to('cuda').eval();model.requires_grad_(False)
            new(folder/'load.json',{'seconds':time.monotonic()-start,'vendor':identity,
                'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','flash-attn','numpy']},
                'model_class':type(model).__name__,'config':model.config.to_dict(),
                'plan_sha256':digest(p),'implementation_sha256':file_sha256(__file__),
                'strict_deterministic_algorithms':True,'tf32':False,'seed':0,
                'decoder':'official VQA forward/argmax with fixed LEFT/RIGHT logits constraint and native EOS',
                'device':str(torch.cuda.get_device_properties(0))})
            observed=[]
            def observe(module,args,kwargs):
                x=args[0] if args else kwargs.get('pixel_values',kwargs.get('hidden_states'))
                # Upstream deliberately prepends one 64x64 zero image. Record it
                # separately from the 180 source frames; do not remove it.
                require(isinstance(x,list) and len(x)==181,'Official vision input list changed')
                observed.append({'shapes':[list(t.shape) for t in x],
                    'dtypes':[str(t.dtype) for t in x],'devices':[str(t.device) for t in x],
                    'official_zero_image':True})
            hook=model.model.visual.register_forward_pre_hook(observe,with_kwargs=True)
            completed=[]
            for item in samples:
                start=time.monotonic();sid=item['sample_id']
                new(folder/(sid+'.claim.json'),{'sample_id':sid,'cold':cold,'attempt':1,
                    'source_sha256':item['source_video_sha256'],'plan_sha256':digest(p)})
                ids,proc,receipt=prepare(processor,item)
                ref=next(x for x in admission['records'] if x['sample_id']==sid)
                require(receipt['processor_sha256']==ref['processor_sha256']
                        and receipt['prompt_token_ids']==ref['prompt_token_ids'],'Actual input drift')
                observed.clear();engine_start=time.monotonic()
                output,eos=generate(model,processor,ids,proc)
                engine=time.monotonic()-engine_start
                answer=processor.tokenizer.decode(output,skip_special_tokens=True)
                raw={'sample_id':sid,'task':item['task'],'truth':item['truth'],'answer':answer,
                    'output_token_ids':output,'finish_reason':'stop' if output[-1]==eos else 'length',
                    'stop_reason':eos if output[-1]==eos else None,'cold':cold,'attempt':1,
                    'source_sha256':item['source_video_sha256'],'plan_sha256':digest(p),
                    'input':receipt,'actual_visual_inputs':list(observed),
                    'engine_inclusive_seconds':engine,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                append(folder/'raw.jsonl',raw)
                require(answer in ['LEFT','RIGHT'] and raw['finish_reason']=='stop','Invalid output; raw retained')
                require(len(observed)==1 and observed[0]['shapes']==[[1,3,64,64]]+[[1,3,512,768]]*180
                    and all(x=='torch.bfloat16' for x in observed[0]['dtypes'])
                    and all(x.startswith('cuda') for x in observed[0]['devices']),
                    'Actual GPU source frames differ; raw retained')
                append(folder/'events.jsonl',{'event':'TECHNICAL_PASS','sample_id':sid,'raw_sha256':digest(raw)})
                append(folder/'timings.jsonl',{'sample_id':sid,'cold':cold,
                    'end_to_durable_raw_and_pass_seconds':time.monotonic()-start,
                    'engine_inclusive_seconds':engine,'deep_observation':True})
                completed.append(sid)
                print(json.dumps({'cold':cold,'completed':len(completed),'sample_id':sid,'answer':answer}),flush=True)
                del ids,proc
            hook.remove();new(folder/'result.json',{'status':'COMPLETE_PASS','completed_ids':completed})
    except BaseException as e:
        atomic_json(folder/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise


def supervise(root):
    p,samples,_,out=configured(root)
    new(out/'rx-launch.json',{'supervisor_pid':os.getpid(),'plan_sha256':digest(p),'automatic_retry':False})
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(p['gpu']),CUBLAS_WORKSPACE_CONFIG=':4096:8',
             OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for cold in [1,2]:
        with (out/f'rx-cold{cold}.log').open('xb') as f:
            child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.rxbrain_admission',
                                    'run','--root',str(root),'--cold',str(cold)],env=env,stdout=f,stderr=subprocess.STDOUT)
            code=child.wait()
        lock=p['physical_lock'];free=False
        with Path(lock['path']).open('r+b') as f:
            s=os.fstat(f.fileno())
            if (s.st_dev,s.st_ino)==(lock['st_dev'],lock['st_ino']):
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
                except BlockingIOError:pass
        new(out/f'rx-cold{cold}.wait.json',{'pid':child.pid,'returncode':code,'real_wait':True,
            'owned_hf_process_exited':not Path(f'/proc/{child.pid}').exists(),
            'original_same_inode_lease_free':free,'automatic_retry':False})
        require(code==0 and free,'Worker failed; successor held, no retry')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['cpu','run','supervise'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--cold',type=int,choices=[1,2])
    args=parser.parse_args()
    if args.command=='run':
        require(args.cold is not None,'Missing cold');run(args.root,args.cold)
    else:globals()[args.command](args.root)


if __name__=='__main__':main()
