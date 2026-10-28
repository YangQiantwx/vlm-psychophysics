"""Finite BF16 embodied-model admission; no retries or formal-queue continuation.

The supported Rynn route uses upstream HF processing/forward/generation. RxBrain
is audited separately: absence of its required upstream modules is not repaired
by silently replacing a shared Transformers/Torch installation.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .storage import atomic_json, file_sha256, digest

NAME = 'embodied-admission-20261027'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def new(path, value):
    path = Path(path)
    require(not path.exists(), f'Refuse to overwrite {path}')
    atomic_json(path, value)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, sort_keys=True) + '\n')
        f.flush()
        os.fsync(f.fileno())


def paths(root):
    root = Path(root)
    return root/'plans'/NAME/'plan.json', root/'runs/reproducibility'/NAME


def configured(root):
    plan_path, out = paths(root)
    p = read(plan_path)
    expected = {'embodied-admission-20261027': 6, 'embodied-points-20261027': 12}[NAME]
    require(p['per_model_maximum_calls'] == expected*2 and p['cold_starts'] == 2
            and len(p['samples']) == expected and not p['automatic_retry'], 'Budget/config changed')
    index = read(Path(root)/'datasets/classic-t1-t3-v0.3/dataset-index.json')
    samples = []
    for s in p['samples']:
        entry = index['items'][s['sample_id']]
        require(entry['video_sha256'] == s['source_video_sha256']
                and entry['truth'] == s['truth'] and entry['task'] == s['task'], 'Source index mismatch')
        samples.append({**s, 'prompt': entry['prompt'], 'source_index': entry,
                        'path': str(Path(root)/'datasets/classic-t1-t3-v0.3'/s['video_relative'])})
    model_path = Path(root)/'models/RynnBrain1.1-9B'
    out.mkdir(parents=True, exist_ok=True)
    return p, samples, model_path, out


def decode(path):
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), 'Cannot open source')
    require(abs(cap.get(cv2.CAP_PROP_FPS)-60) < 1e-6, 'Unexpected source fps')
    frames = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            require(len(frames) < 180 and bgr.shape == (512,768,3), 'Frame/geometry drift')
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        cap.release()
    require(len(frames) == 180, 'Reader did not preserve all180')
    return np.stack(frames)


def tensor_sha(tensor):
    import torch
    return hashlib.sha256(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()).hexdigest()


def prepare(processor, item):
    from transformers.video_utils import VideoMetadata
    start = time.monotonic()
    frames = decode(item['path'])
    after_read = time.monotonic()
    text = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'video'}, {'type': 'text', 'text': item['prompt']}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require(not text.rstrip().endswith('<think>'), 'Thinking active')
    metadata = VideoMetadata(total_num_frames=180, fps=60, width=768, height=512,
                             duration=3., video_backend='opencv', frames_indices=list(range(180)))
    inputs = processor(text=[text], videos=[frames], video_metadata=[metadata],
                       do_resize=False, do_sample_frames=False, return_tensors='pt')
    grid = inputs['video_grid_thw'].tolist()
    require(grid == [[90,32,48]], f'Resized/sampled processor grid: {grid}')
    require(inputs['pixel_values_videos'].shape[0] == 90*32*48, 'Patch count changed')
    token_ids = inputs['input_ids'][0].tolist()
    require(token_ids.count(processor.tokenizer.convert_tokens_to_ids('<|video_pad|>')) == 34560,
            'Expanded video token count differs')
    receipt = {'reader': 'OpenCV sequential BGR-to-RGB uint8', 'reader_shape': list(frames.shape),
               'frame_indices': list(range(180)), 'fps': 60, 'processor_grid': grid,
               'processor_shape': list(inputs['pixel_values_videos'].shape),
               'processor_dtype': str(inputs['pixel_values_videos'].dtype),
               'processor_sha256': tensor_sha(inputs['pixel_values_videos']),
               'prompt_token_ids': token_ids, 'prompt_text': text,
               'decoded_sha256': hashlib.sha256(frames.tobytes()).hexdigest(),
               'visual_tokens': 34560, 'effective_geometry_wh': [768,512],
               'read_decode_seconds': after_read-start,
               'processor_and_admission_seconds': time.monotonic()-after_read}
    return inputs, receipt


def cpu(root):
    import torch
    from transformers import AutoProcessor
    p, samples, model_path, out = configured(root)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    records = []
    for item in samples:
        require(file_sha256(item['path']) == item['source_video_sha256'], 'Source bytes changed')
        inputs, record = prepare(processor, item)
        record.update(sample_id=item['sample_id'], source_sha256=item['source_video_sha256'])
        records.append(record)
        del inputs
    require(not torch.cuda.is_initialized(), 'CPU admission initialized CUDA')
    result = {'status': 'CPU_PASS', 'model_calls': 0, 'plan_sha256': digest(p),
              'implementation_sha256': file_sha256(__file__), 'records': records}
    new(out/'cpu-admission.json', result)
    print(json.dumps({'status':'CPU_PASS','samples':len(records),'all180':True,'grid':[90,32,48]}), flush=True)


def allowed_tokens(tokenizer, eos, prompt_length):
    # Same LEFT/RIGHT answer language as the original benchmark. Native EOS IDs
    # are admitted only after a complete answer, not replaced by a custom stop.
    sequences = [tokenizer.encode(s, add_special_tokens=False) for s in ('LEFT','RIGHT')]
    require(all(sequences), 'Empty answer token sequence')
    def prefix(batch_id, ids):
        suffix = ids.tolist()[prompt_length:]
        result = []
        for seq in sequences:
            if suffix == seq:
                result.extend(eos)
            elif seq[:len(suffix)] == suffix:
                result.append(seq[len(suffix)])
        require(result, 'Answer escaped fixed language')
        return sorted(set(result))
    return prefix


def run(root, cold):
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    p, samples, model_path, out = configured(root)
    admission = read(out/'cpu-admission.json')
    require(admission['plan_sha256'] == digest(p)
            and admission['implementation_sha256'] == file_sha256(__file__), 'CPU admission stale')
    folder = out/f'rynn-cold{cold}'
    require(not folder.exists(), 'Cold attempt already exists; no automatic retry')
    folder.mkdir()
    new(folder/'owner.json', {'pid':os.getpid(),'role':'single-process HF model, no separate engine'})
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(8)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.device_count() == 1, 'Require one visible GPU')
    require(str(torch.cuda.get_device_properties(0).uuid).lower().removeprefix('gpu-') ==
            p['gpu_uuid'].lower().removeprefix('gpu-'), 'Wrong physical card')
    lock = p['physical_lock']; lock_path = Path(lock['path'])
    require(not lock_path.is_symlink(), 'Symlink lease')
    try:
        with lock_path.open('r+b') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            stat = os.fstat(handle.fileno())
            require((stat.st_dev,stat.st_ino) == (lock['st_dev'],lock['st_ino']), 'Original lease replaced')
            free,total = torch.cuda.mem_get_info()
            require(free > 70*1024**3, 'Need at least70GiB free before finite admission')
            started = time.monotonic()
            processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
                        attn_implementation='sdpa', local_files_only=True).to('cuda').eval()
            model.requires_grad_(False)
            eos = model.generation_config.eos_token_id
            eos = eos if isinstance(eos,list) else [eos]
            require(all(isinstance(x,int) for x in eos), 'Missing native EOS')
            new(folder/'load.json', {'seconds':time.monotonic()-started,'dtype':'bfloat16',
                'model_class':type(model).__name__,'model_config':model.config.to_dict(),
                'native_generation_config':model.generation_config.to_dict(),
                'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','numpy']},
                'device':str(torch.cuda.get_device_properties(0)), 'native_eos':eos,
                'plan_sha256':digest(p),'implementation_sha256':file_sha256(__file__),
                'sdpa':True,'strict_deterministic_algorithms':True,'tf32':False})
            visual = model.model.visual
            observations = []
            def observe(module, args, kwargs):
                x = args[0] if args else kwargs['hidden_states']
                observations.append({'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device),
                                     'grid':kwargs.get('grid_thw').tolist() if kwargs.get('grid_thw') is not None else None})
            hook = visual.register_forward_pre_hook(observe, with_kwargs=True)
            completed = []
            for item in samples:
                sid = item['sample_id']; request_start = time.monotonic()
                new(folder/(sid+'.claim.json'), {'sample_id':sid,'cold':cold,'attempt':1,
                    'plan_sha256':digest(p),'source_sha256':item['source_video_sha256']})
                inputs, receipt = prepare(processor,item)
                reference = next(x for x in admission['records'] if x['sample_id']==sid)
                require(receipt['processor_sha256']==reference['processor_sha256']
                    and receipt['prompt_token_ids']==reference['prompt_token_ids'], 'CPU/GPU input drift')
                inputs = inputs.to('cuda')
                length = inputs['input_ids'].shape[1]
                observations.clear()
                generator_start = time.monotonic()
                with torch.inference_mode():
                    result = model.generate(**inputs, do_sample=False, temperature=0., top_p=1.,
                        num_beams=1, repetition_penalty=1., max_new_tokens=16384,
                        prefix_allowed_tokens_fn=allowed_tokens(processor.tokenizer,eos,length),
                        return_dict_in_generate=True)
                ids = result.sequences[0,length:].tolist()
                engine_seconds = time.monotonic()-generator_start
                text = processor.tokenizer.decode(ids, skip_special_tokens=True)
                raw = {'sample_id':sid,'task':item['task'],'truth':item['truth'],'answer':text,
                    'output_token_ids':ids,'finish_reason':'stop' if ids and ids[-1] in eos else 'length',
                    'stop_reason':ids[-1] if ids and ids[-1] in eos else None,
                    'source_sha256':item['source_video_sha256'],'cold':cold,'attempt':1,
                    'plan_sha256':digest(p),'input':receipt,'actual_visual_inputs':list(observations),
                    'engine_inclusive_seconds':engine_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
                append(folder/'raw.jsonl',raw)
                require(text in ['LEFT','RIGHT'] and raw['finish_reason']=='stop', 'Invalid completion; raw retained')
                require(len(observations)==1 and observations[0]['shape'][0]==138240
                        and observations[0]['device'].startswith('cuda'), 'Actual visual input differs; raw retained')
                append(folder/'events.jsonl',{'event':'TECHNICAL_PASS','sample_id':sid,'raw_sha256':digest(raw)})
                append(folder/'timings.jsonl',{'sample_id':sid,'cold':cold,
                    'end_to_durable_raw_and_pass_seconds':time.monotonic()-request_start,
                    'engine_inclusive_seconds':engine_seconds,'input_seconds':receipt['read_decode_seconds']+
                    receipt['processor_and_admission_seconds'],'deep_observation':True})
                completed.append(sid)
                print(json.dumps({'cold':cold,'completed':len(completed),'sample_id':sid,'answer':text}),flush=True)
                del inputs,result
            hook.remove()
            new(folder/'result.json',{'status':'COMPLETE_PASS','completed_ids':completed,'count':len(completed)})
            require((lock_path.stat().st_dev,lock_path.stat().st_ino)==(stat.st_dev,stat.st_ino), 'Lease changed')
    except BaseException as e:
        atomic_json(folder/'failure.json',{'error':repr(e),'traceback':traceback.format_exc(),'automatic_retry':False})
        raise


def supervise(root):
    p, samples, model_path, out = configured(root)
    new(out/'launch.json',{'supervisor_pid':os.getpid(),'plan_sha256':digest(p),'automatic_retry':False})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(p['gpu']), CUBLAS_WORKSPACE_CONFIG=':4096:8',
               OMP_NUM_THREADS='8',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    for cold in (1,2):
        with (out/f'rynn-cold{cold}.log').open('xb') as f:
            child = subprocess.Popen([sys.executable,'-B','-m',__name__.replace('__main__','classic_precision.embodied_admission'),
                                      'run','--root',str(root),'--name',NAME,'--cold',str(cold)],env=env,stdout=f,stderr=subprocess.STDOUT)
            code = child.wait()
        free = False; lock=p['physical_lock']
        with Path(lock['path']).open('r+b') as f:
            s=os.fstat(f.fileno())
            if (s.st_dev,s.st_ino)==(lock['st_dev'],lock['st_ino']):
                try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
                except BlockingIOError: pass
        new(out/f'rynn-cold{cold}.wait.json',{'pid':child.pid,'returncode':code,'real_wait':True,
            'owned_hf_process_exited':not Path(f'/proc/{child.pid}').exists(),
            'original_same_inode_lease_free':free,'automatic_retry':False})
        require(code==0 and free,'Cold worker failed; successor held, no retry')
    first=[json.loads(s) for s in (out/'rynn-cold1/raw.jsonl').read_text().splitlines()]
    second=[json.loads(s) for s in (out/'rynn-cold2/raw.jsonl').read_text().splitlines()]
    require(len(first)==len(second)==len(samples),'Incomplete ID ledger')
    matches=[a['sample_id']==b['sample_id'] and all(a[k]==b[k] for k in
        ['answer','output_token_ids','finish_reason','stop_reason']) and
        a['input']['processor_sha256']==b['input']['processor_sha256'] and
        a['input']['prompt_token_ids']==b['input']['prompt_token_ids'] for a,b in zip(first,second)]
    new(out/'rynn-summary.json',{'status':'COMPLETE_REPLAY_MATCH' if all(matches) else 'COMPLETE_REPLAY_GAP',
        'model_calls':len(samples)*2,'unique_samples':len(samples),'same_card_cold_matches':sum(matches),
        'all180':True,'no_resize':True,'automatic_retry':False,'formal_auto_resume':False,
        'scope':f'{len(samples)} fixed samples, two cold loads on Fourier GPU0; not all-card/full-grid reproducibility'})


def main():
    global NAME
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['cpu','run','supervise'])
    parser.add_argument('--root',required=True)
    parser.add_argument('--name',default=NAME,choices=['embodied-admission-20261027','embodied-points-20261027'])
    parser.add_argument('--cold',type=int,choices=[1,2])
    args=parser.parse_args()
    NAME=args.name
    if args.command=='run':
        require(args.cold is not None,'Missing cold number')
        run(args.root,args.cold)
    else:
        globals()[args.command](args.root)


if __name__=='__main__':
    main()
