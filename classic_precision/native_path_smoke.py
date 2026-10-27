"""Finite native file-URL evaluation; no custom pixel transformation.

The four model engines run sequentially. A persisted intent is never retried.
Media observation records the native decoder return without modifying it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from .storage import atomic_json, digest, file_sha256


def read(path):
    return json.loads(Path(path).read_text())


def append(path, value):
    data = (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with Path(path).open('ab') as f:
        begin = f.tell(); f.write(data); f.flush(); os.fsync(f.fileno())
    return {'offset': begin, 'length': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def identity(pid=None):
    pid = os.getpid() if pid is None else pid
    p = Path('/proc') / str(pid) / 'stat'
    if not p.exists(): return None
    fields = p.read_text().split(') ', 1)[1].split()
    return {'pid': pid, 'startticks': fields[19], 'state': fields[0]}


def engine_identity(worker):
    return identity()


def validate(plan):
    paired = plan['schema'] in ('native-file-t3-resize-arm-v1', 'native-file-t1-resize-arm-v1',
                               'native-file-t2-qsa-resize-arm-v1')
    t2_no_resize = plan['schema'] == 'native-file-t2-no-resize-v1'
    task = ('T1' if plan['schema'] == 'native-file-t1-resize-arm-v1' else
            'T3' if plan['schema'] == 'native-file-t3-resize-arm-v1' else 'T2')
    assert paired or t2_no_resize or plan['schema'] == 'native-file-t2-four-model-smoke-v1'
    rows = plan['rows']; ids = [r['sample_id'] for r in rows]
    size = 96 if paired and task in ('T1', 'T3') else 48
    assert len(ids) == size and len(set(ids)) == size
    assert set(plan['models']) == ({'qwen','nemo','cosmos'} if paired else {'qwen','nemo','cosmos','gemma'})
    assert all(r['task'] == task and r['truth'] in ('LEFT','RIGHT') for r in rows)
    assert sum(r['truth'] == 'LEFT' for r in rows) == size//2
    if t2_no_resize:
        assert plan['arm']=='no_resize' and plan['qsa_determinism'] is False
        assert plan['automatic_retry'] is False and plan['expected_video_frames']==180
        assert set(plan['order']) <= {'qwen','nemo','cosmos'}
        assert all(r['source_parameters']['background_Y']==0.05
                   and r['source_parameters']['delta_Y']==0.15
                   and r['source_parameters']['offset_frame']-r['source_parameters']['onset_frame']==10 for r in rows)
    if paired:
        if task == 'T3':
            assert all(r['source_parameters']['angle_deg']==0.7 for r in rows)
            assert sum(r['source_parameters']['tilt_sign']==-1 for r in rows)==48
        elif task == 'T1':
            for sf in (4, 64):
                cell = [r for r in rows if r['source_parameters']['sf_cycles_per_aperture']==sf]
                assert len(cell)==48 and sum(r['truth']=='LEFT' for r in cell)==24
                pairs = {}
                for r in cell:
                    p=r['source_parameters']
                    assert p['modulation_depth']==0.6 and p['tf_hz']==5
                    pairs.setdefault(p['phase_index'], []).append(r['truth'])
                assert set(pairs)==set(range(24)) and all(sorted(v)==['LEFT','RIGHT'] for v in pairs.values())
        else:
            assert plan['qsa_determinism'] is True and plan['order']==['qwen']
            assert all(r['source_parameters']['background_Y']==0.05
                       and r['source_parameters']['delta_Y']==0.15
                       and r['source_parameters']['offset_frame']-r['source_parameters']['onset_frame']==10
                       for r in rows)
        assert plan['observe_first_n']==2 and plan['expected_video_frames']==180
        assert plan['arm'] in ('resize','no_resize')
        assert plan['automatic_retry'] is False
    assert len({r['prompt'] for r in rows}) == 1
    assert plan['source_sha256'] == file_sha256(__file__)
    for model_name, settings in plan['models'].items():
        assert 'hf_overrides' not in settings
        if plan.get('expected_video_frames') == 180:
            if model_name == 'nemo':
                assert not settings.get('mm_processor_kwargs')
            else:
                expected={'do_sample_frames':False}
                if paired: expected['do_resize']=plan['arm']=='resize'
                if t2_no_resize and model_name!='gemma': expected['do_resize']=False
                assert settings.get('mm_processor_kwargs') == expected
            assert settings['media_io_kwargs']['video'] == {'video_backend': 'opencv', 'num_frames': 180, 'fps': -1}
        else:
            assert 'mm_processor_kwargs' not in settings
        assert settings['limit_mm_per_prompt']['video'] == 1
        assert settings['kv_cache_dtype'] == 'bfloat16'
    if 'execution_ids' in plan:
        assert t2_no_resize and len(plan['order'])==1
        selected=plan['execution_ids']
        assert selected and len(selected)==len(set(selected)) and set(selected)<=set(ids)
        assert selected==[sid for sid in ids if sid in set(selected)]
        assert plan['max_new_calls']==len(selected)
        assert plan['continuation_base_plan_sha256']=='ea594c1f0a7ddd77e964cca9620ebec9fa5ef34ccecb18f1f84d316f23f86ee7'
        rows=[r for r in rows if r['sample_id'] in set(selected)]
    return rows


def nemo_original_geometry(config):
    # Native supported configuration only. No pixel transformation here.
    vision=getattr(config,'vision_config',None)
    if vision is None: return config  # vLLM also calls overrides on a dummy config.
    for k,v in {'video_target_num_patches':1536,'video_maintain_aspect_ratio':True}.items():
        if isinstance(vision,dict): vision[k]=v
        else: setattr(vision,k,v)
    return config


def qsa_control(worker):
    from runtime_compat import classic_precision_qsa_determinism as qsa
    assert file_sha256(qsa.__file__)=='bf3f16fc3092b4edc6a8308dbad49bc84efacc327a88e2f697ae6bb3b38b7624'
    qsa.install()
    return qsa.status()


def validate_pair(pair):
    assert pair['schema'] in ('native-file-t3-resize-pair-v1', 'native-file-t1-resize-pair-v1',
                             'native-file-t2-qsa-resize-pair-v1')
    assert pair['model'] in ('qwen','nemo','cosmos') and pair['automatic_retry'] is False
    a,b=(pair['arms'][k] for k in ('resize','no_resize'))
    assert a['schema']==b['schema']==pair['schema'].replace('-pair-', '-arm-')
    validate(a);validate(b)
    assert a['arm']=='resize' and b['arm']=='no_resize'
    assert a['rows']==b['rows'] and a['runtime']==b['runtime']
    assert a['order']==b['order']==[pair['model']]
    assert a['qsa_determinism']==b['qsa_determinism']==(pair['model']=='qwen')
    from copy import deepcopy
    aa,bb=deepcopy(a['models']),deepcopy(b['models'])
    for name in ('qwen','cosmos'):
        assert aa[name]['mm_processor_kwargs'].pop('do_resize') is True
        assert bb[name]['mm_processor_kwargs'].pop('do_resize') is False
    assert aa==bb, 'Only native resize geometry may differ between arms'
    return a,b


def pair_controller(plan_path, output):
    pair=read(plan_path);validate_pair(pair);root=Path(output);root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'pair-controller.json',{'identity':identity(),'plan_sha256':digest(pair),'model':pair['model']})
    for arm in ('resize','no_resize'):
        plan=pair['arms'][arm]
        if Path(plan['smoke_pause_file']).exists(): break
        target=root/('plan-'+arm+'.json');atomic_json(target,plan)
        env=os.environ.copy();env.update(plan['runtime']['environment'])
        with (root/(arm+'-controller.log')).open('xb') as log:
            began=time.monotonic();child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.native_path_smoke','controller','--plan',str(target),'--output',str(root/arm)],env=env,stdout=log,stderr=subprocess.STDOUT)
            atomic_json(root/(arm+'-controller-child.json'),{'identity':identity(child.pid)})
            rc=child.wait();atomic_json(root/(arm+'-controller-wait.json'),{'actual_wait_observed':True,'returncode':rc,'elapsed_monotonic_seconds':time.monotonic()-began})
        assert rc==0,'Arm failed; no automatic retry or following arm'
    complete=all((root/arm/(pair['model']+'-accepted.json')).exists() for arm in ('resize','no_resize'))
    atomic_json(root/'pair-complete.json',{'status':'COMPLETE_PASS' if complete else 'PAUSED',
        'count':2*len(pair['arms']['resize']['rows']) if complete else None,'automatic_resume':False})


def worker(plan_path, model, output):
    plan = read(plan_path); rows = validate(plan)
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    atomic_json(root/'launch.json', {'owner': identity(), 'plan_sha256': digest(plan),
        'model': model, 'source_sha256': file_sha256(__file__), 'automatic_retry': False})
    import torch
    from .gemma_speed import physical_lease
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.multimodal.media.video import VideoMediaIO
    engine = None; owned = []; completed = []; pending = None; started = time.monotonic()
    # Observation only: return the same object and never cast or resize pixels.
    original = VideoMediaIO.load_file
    media = []; processor_observations = []; active = [False]
    from vllm.multimodal.processing.context import InputProcessingContext
    original_postprocess = InputProcessingContext._postprocess_output
    original_processor_call = InputProcessingContext.call_hf_processor
    def tensor_description(value):
        if isinstance(value, torch.Tensor):
            result={'shape':list(value.shape),'dtype':str(value.dtype)}
            if not value.is_floating_point() and value.numel() <= 600: result['values']=value.tolist()
            return result
        if isinstance(value,(list,tuple)): return [tensor_description(x) for x in value[:4]]
        return str(type(value).__name__)
    def describe(data):
        return {k:tensor_description(v) for k,v in data.items()
            if 'pixel' in k or 'grid' in k or k in ('video_num_patches','frames_indices','video_frame_counts')}
    def observed_postprocess(ctx, output):
        before=describe(output) if active[0] else None
        result=original_postprocess(ctx, output)
        if before and active[0]: processor_observations.append({'before_native_vllm_dtype':before, 'after_native_vllm_dtype':describe(result)})
        return result
    def observed_processor_call(ctx, hf_processor, data, kwargs={}):
        result=original_processor_call(ctx,hf_processor,data,kwargs)
        if active[0] and plan.get('expected_video_frames') == 180:
            if model=='gemma' and 'pixel_values' in result:
                assert result['pixel_values'].shape[0] == 180, 'Gemma processor dropped frames'
            elif model=='nemo' and 'video_num_patches' in result:
                assert result['video_num_patches'].tolist() == [180], 'Nemo processor dropped frames'
            elif model in ('cosmos','qwen') and 'video_grid_thw' in result:
                temporal=int(hf_processor.video_processor.temporal_patch_size)
                assert int(result['video_grid_thw'][0,0])*temporal == 180, 'Qwen/Cosmos processor dropped frames'
        return result
    InputProcessingContext._postprocess_output=observed_postprocess
    InputProcessingContext.call_hf_processor=observed_processor_call
    def observed(loader, filepath):
        result = original(loader, filepath)
        frames, metadata = result.media
        if active[0] and plan.get('expected_video_frames') == 180:
            assert frames.shape[0] == 180 and metadata['frames_indices'] == list(range(180)), 'Native reader did not deliver every source frame'
        media.append({'path': str(filepath), 'dtype': str(frames.dtype),
            'shape': list(frames.shape), 'metadata': metadata,
            'loader_class': type(loader.video_loader).__name__, 'loader_kwargs': loader.kwargs})
        return result
    VideoMediaIO.load_file = observed
    with physical_lease(plan['runtime'], torch) as lease:
        atomic_json(root/'lease.json', lease)
        try:
            load_start = time.monotonic()
            settings=dict(plan['models'][model])
            if plan.get('arm')=='no_resize' and model=='nemo': settings['hf_overrides']=nemo_original_geometry
            engine = LLM(**settings)
            owned = engine.collective_rpc(engine_identity)
            qsa_receipt=engine.collective_rpc(qsa_control) if plan.get('qsa_determinism') else None
            if qsa_receipt: assert all(x and x['state'] in ('applied','pending_target_import') for x in qsa_receipt)
            tokenizer = engine.get_tokenizer()
            atomic_json(root/'load.json', {'seconds': time.monotonic()-load_start,
                'owned_engines': owned, 'settings': plan['models'][model],
                'native_eos_token_id': tokenizer.eos_token_id,
                'media_loader_observation_only': True,
                'custom_QSA_selector': bool(plan.get('qsa_determinism')), 'qsa_receipt':qsa_receipt,
                'native_nemo_original_geometry':model=='nemo' and plan.get('arm')=='no_resize',
                'custom_Gemma_attention_or_pixel_cast': False})
            sampling = SamplingParams(n=1, temperature=0., top_p=1., seed=0,
                repetition_penalty=1., max_tokens=16384,
                structured_outputs=StructuredOutputsParams(choice=['LEFT','RIGHT']))
            for index, row in enumerate(rows):
                if Path(plan['smoke_pause_file']).exists(): break
                pending = row['sample_id']; media.clear(); processor_observations.clear(); active[0]=index < plan.get('observe_first_n',len(rows))
                append(root/'events.jsonl', {'kind':'INTENT','ordinal':index,
                    'sample_id':pending, 'row_sha256':digest(row)})
                messages = [{'role':'user','content':[
                    {'type':'video_url','video_url':{'url':Path(row['path']).as_uri()}},
                    {'type':'text','text':row['prompt']}]}]
                t = time.monotonic()
                result = engine.chat(messages, sampling_params=sampling, use_tqdm=False,
                    chat_template_kwargs={'enable_thinking':False},
                    mm_processor_kwargs=plan['models'][model].get('mm_processor_kwargs'))[0]
                seconds = time.monotonic()-t; answer = result.outputs[0]
                record = {'model':model,'ordinal':index,'sample_id':pending,
                    'truth':row['truth'],'raw_answer':answer.text,
                    'output_token_ids':list(answer.token_ids),'finish_reason':answer.finish_reason,
                    'stop_reason':answer.stop_reason,'correct':answer.text.strip()==row['truth'],
                    'prompt_token_ids':result.prompt_token_ids,
                    'native_media':list(media),'native_processor':list(processor_observations),'request_seconds':seconds,
                    'source':row,'plan_sha256':digest(plan)}
                proof = append(root/'raw.jsonl', record)
                if plan.get('qsa_determinism') and index==0:
                    applied=engine.collective_rpc(qsa_control)
                    assert all(x and x['state']=='applied' for x in applied), 'QSA fixed-output control not actually applied'
                    atomic_json(root/'qsa-first-output.json',applied)
                # Preserve any unexpected output before stopping, without retry.
                assert answer.text.strip() in ('LEFT','RIGHT') and answer.finish_reason == 'stop'
                assert media and all(x['path']==row['path'] for x in media)
                if plan.get('expected_video_frames') == 180 and active[0]: assert processor_observations, 'No actual native processor observations'
                if plan.get('expected_video_frames') == 180:
                    assert all(x['shape'][0]==180 and x['metadata']['frames_indices']==list(range(180)) for x in media)
                append(root/'events.jsonl', {'kind':'PASS','sample_id':pending,
                    'ordinal':index,'raw_byteproof':proof,'raw_digest':digest(record)})
                completed.append(pending); pending = None
                print(json.dumps({'model':model,'completed':len(completed),'seconds':round(seconds,3)}),flush=True)
        finally:
            VideoMediaIO.load_file = original
            InputProcessingContext._postprocess_output=original_postprocess
            InputProcessingContext.call_hf_processor=original_processor_call
            if engine is not None:
                candidates = [engine, getattr(engine,'llm_engine',None)]
                candidates.append(getattr(candidates[-1],'engine_core',None))
                for candidate in candidates:
                    shutdown = getattr(candidate,'shutdown',None)
                    if callable(shutdown): shutdown(); break
            deadline=time.monotonic()+30
            while any((p:=identity(x['pid'])) and p['startticks']==x['startticks']
                      and p['state']!='Z' for x in owned) and time.monotonic()<deadline:
                time.sleep(.2)
            absent = all(not (p:=identity(x['pid'])) or p['startticks']!=x['startticks']
                         or p['state']=='Z' for x in owned)
            atomic_json(root/'closure.json', {'completed_ids':completed,'pending_id':pending,
                'owned_engines':owned,'owned_engines_exited':absent,
                'status':'COMPLETE' if len(completed)==len(rows) else 'PAUSED_OR_FAILED',
                'elapsed_monotonic_seconds':time.monotonic()-started})
            assert absent, 'Engine still active: do not release native lease'


def check_closed(root, ids):
    root=Path(root); closure=read(root/'closure.json')
    assert closure['completed_ids']==ids and closure['pending_id'] is None
    events=[json.loads(s) for s in (root/'events.jsonl').read_text().splitlines()]
    assert [(x['kind'],x['sample_id']) for x in events]==[
        (k,sid) for sid in ids for k in ('INTENT','PASS')]
    with (root/'raw.jsonl').open('rb') as f:
        for e in events[1::2]:
            p=e['raw_byteproof']; assert f.tell()==p['offset']
            b=f.read(p['length']); assert hashlib.sha256(b).hexdigest()==p['sha256']
            r=json.loads(b); assert digest(r)==e['raw_digest'] and r['sample_id']==e['sample_id']
        assert not f.read(1)
    return closure


def controller(plan_path, output):
    import fcntl
    plan=read(plan_path); rows=validate(plan); root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    atomic_json(root/'controller.json',{'identity':identity(),'plan_sha256':digest(plan)})
    env=os.environ.copy();env.update(plan['runtime']['environment'])
    for model in plan['order']:
        if Path(plan['smoke_pause_file']).exists(): break
        destination=root/model
        with (root/(model+'.log')).open('xb') as log:
            start=time.monotonic()
            child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.native_path_smoke','worker',
                '--plan',str(plan_path),'--model',model,'--output',str(destination)],env=env,
                stdout=log,stderr=subprocess.STDOUT)
            atomic_json(root/(model+'-child.json'),{'identity':identity(child.pid)})
            rc=child.wait()
            atomic_json(root/(model+'-wait.json'),{'returncode':rc,
                'actual_wait_observed':True,'elapsed_monotonic_seconds':time.monotonic()-start})
        assert rc==0, 'Child failed; no automatic retry or following model launch'
        closure=check_closed(destination,[r['sample_id'] for r in rows])
        for x in closure['owned_engines']:
            p=identity(x['pid']);assert not p or p['startticks']!=x['startticks'] or p['state']=='Z'
        lease=read(destination/'lease.json')
        with open(lease['path'],'r+b') as f:
            st=os.fstat(f.fileno());assert(st.st_dev,st.st_ino)==(lease['st_dev'],lease['st_ino'])
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
        atomic_json(root/(model+'-accepted.json'),{'status':'COMPLETE_PASS','count':len(rows),
            'actual_wait_rc':rc,'owned_exited':True,'same_inode_native_lock_free':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','controller','check','pair-controller','pair-check'])
    p.add_argument('--plan',required=True);p.add_argument('--output');p.add_argument('--model')
    a=p.parse_args()
    if a.mode=='pair-check': print(json.dumps({'status':'PAIR_PLAN_PASS','rows':len(validate_pair(read(a.plan))[0]['rows'])}))
    elif a.mode=='pair-controller': pair_controller(a.plan,a.output)
    elif a.mode=='check': print(json.dumps({'status':'PLAN_PASS','rows':len(validate(read(a.plan)))}))
    elif a.mode=='worker': worker(a.plan,a.model,a.output)
    else: controller(a.plan,a.output)


if __name__=='__main__':
    main()
