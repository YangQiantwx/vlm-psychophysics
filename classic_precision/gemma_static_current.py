"""Pinned Gemma NVFP4 on the existing 152-image, three-prompt control panel.

The retained image processor supplies FP32 patches directly to vLLM. The
native image policy is distinct from video: max_soft_tokens=280, one image.
There is no video CPU-copy overlay and no implicit retry after a partial run.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import json
import signal
import sys
import time
import types
import uuid
from pathlib import Path
from . import gemma_nvfp4_continuation as c
from . import gemma_speed as g
from . import nvfp4_gemma as nv
from . import static_current as sc
from .storage import RunStore, digest, file_sha256, atomic_json

SCHEMA='gemma-current-static-nvfp4-v1'


def image_payload(data):
    import torch
    # Match the retained control: BF16 vision input BEFORE official normalization.
    return {'pixel_values':data['pixel_values'].to(dtype=torch.bfloat16),
            'pixel_position_ids':data['image_position_ids']}


def settings(profile):
    value=c.settings(profile['runtime'],profile['candidate'])
    value['limit_mm_per_prompt']={'video':0,'image':1,'audio':0}
    # Preserve the original official image policy; never reuse video kwargs.
    value['mm_processor_kwargs']={'max_soft_tokens':280}
    return value


def seal(args):
    anchor=g.read(args.video_profile)
    g.require(anchor['candidate'] in ('qualified-o0','compiled-o0'),'A tested video candidate is required')
    static=Path(args.static_root);code=Path(args.static_code)
    manifest=g.read(static/'execution-manifest.json');rows=sc.ordered_rows(manifest)
    selected=g.read(args.ids) if args.ids else [r['sample_id'] for r in rows]
    g.require(selected and len(selected)==len(set(selected)) and set(selected)<={r['sample_id'] for r in rows},'Invalid image assignment')
    q={'schema':SCHEMA,'source_sha256':file_sha256(__file__),'shared_source_sha256':file_sha256(sc.__file__),
       'video_profile':sc.descriptor(args.video_profile),'runtime':anchor['runtime'],'candidate':anchor['candidate'],
       'kernel':anchor['kernel'],'settings':settings(anchor),'runtime_sources':anchor['runtime_sources'],
       'manifest':sc.descriptor(static/'execution-manifest.json'),'static_root':str(static),
       'static_sources':{n:sc.descriptor(code/n) for n in ['static_inputs.py','static_worker.py','static_extension.py']},
       'panel':sc.descriptor(static/'four-model-panel.json'),'sample_ids':selected,'maximum_calls':len(selected),
       'first_deep_calls':2,'automatic_retry':False,'worker_token':uuid.uuid4().hex,
       'pause_file':str(Path(args.pause_file).resolve()),'CPU_video_overlay':False}
    g.require(not Path(args.output).exists(),'No profile overwrite');atomic_json(args.output,q)
    return {'status':'SEALED','calls':len(selected),'profile_sha256':digest(q),'GPU_calls':0}


def install_observer(worker):
    model=worker.get_model();worker._gemma_image={'deep':True,'rows':[]}
    original=model._process_image_input
    def projection(layer,args):
        state=worker._gemma_image;x=args[0]
        row={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)}
        if state['deep']:row['tensor_sha256']=g.sha_tensor(x)
        state['projection']=row
    model.vision_tower.patch_embedder.input_proj.register_forward_pre_hook(projection)
    def image(this,inputs):
        state=worker._gemma_image;x=inputs['pixel_values'];pos=inputs['pixel_position_ids']
        if isinstance(x,list):
            import torch
            x=torch.stack(x);pos=torch.stack(pos)
        row={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)}
        if state['deep']:row.update(tensor_sha256=g.sha_tensor(x),positions_sha256=g.sha_tensor(pos))
        result=original(inputs);row['projection']=state.pop('projection');row['embedding_shapes']=[list(t.shape) for t in result]
        state['rows'].append(row);return result
    model._process_image_input=types.MethodType(image,model)
    return {'status':'INSTALLED','model':type(model).__name__}


def set_deep(worker,deep):worker._gemma_image.update(deep=deep,rows=[])
def read_observer(worker):return worker._gemma_image['rows']
def placement(worker):return nv._placement(worker,offload_gb=0)


def validate_output(raw,post):
    receipt=raw['input_receipt'];rows=post['actual_GPU_visual_forward']
    g.require(raw['actual_input_token_ids']==raw['expected_input_token_ids'],'Actual image prompt tokens differ')
    g.require(len(rows)==1,'Expected one actual image forward')
    row=rows[0]
    g.require(row['shape']==receipt['actual_processor_tensor_shape']==[1,2520,768]
              and row['dtype']=='torch.bfloat16' and row['device'].startswith('cuda'),'Actual image tensor differs')
    if receipt['deep_check']:
        g.require(row['tensor_sha256']==receipt['expected_BF16_GPU_sha256'] and
                  row['positions_sha256']==receipt['image_position_ids_sha256'],'Actual image pixels/positions differ')
    projection=row['projection']
    g.require(projection['shape']==[1,2520,768] and projection['dtype']=='torch.bfloat16' and projection['device'].startswith('cuda'),'Actual image projection differs')
    if receipt['deep_check']:g.require(projection['tensor_sha256']==receipt['expected_normalized_projection_BF16_sha256'],'Actual normalized BF16 projection differs')
    g.require(raw['bridge_calls']>=1 and raw['raw_answer'] in ('LEFT','RIGHT') and
              raw['finish_reason']=='stop' and raw['output_token_ids'][-1:]==[106],'Exact LR/native EOS required')


def run(args):
    g.require('torch' not in sys.modules and 'vllm' not in sys.modules,'Fresh process required')
    q=g.read(args.profile);anchor=sc.read_bound(q['video_profile'])
    g.require(q['schema']==SCHEMA and q['source_sha256']==file_sha256(__file__) and
              q['shared_source_sha256']==file_sha256(sc.__file__) and q['settings']==settings(anchor), 'Source/settings differ')
    g.require(not q['automatic_retry'] and q['runtime']==anchor['runtime'],'Runtime/retry changed')
    c.sources();c.runtime_check(q['runtime'])
    g.require(g.runtime_sources()==q['runtime_sources'],'SDK source changed')
    sc.read_bound(q['panel']);manifest=sc.read_bound(q['manifest'])
    rows={r['sample_id']:r for r in sc.ordered_rows(manifest)};images={r['image_id']:r for r in manifest['images']}
    root=Path(args.output);g.require(not root.exists(),'No implicit resume or overwrite');root.mkdir(parents=True)
    # Import original sources without changing their bytes or any model code.
    bridge=dict(q,runtime=dict(q['runtime'],original_checkpoint=q['runtime']['processor_checkpoint'],cosmos_package=''))
    bridge['static_sources']={k:v for k,v in q['static_sources'].items() if k!='static_extension.py'}
    inputs,_=sc.load_static(bridge)
    desc=q['static_sources']['static_extension.py'];g.require(file_sha256(desc['path'])==desc['sha256'],'Image processor source changed')
    spec=importlib.util.spec_from_file_location('static_extension',desc['path']);ext=importlib.util.module_from_spec(spec)
    sys.modules['static_extension']=ext;spec.loader.exec_module(ext);ext.RUN=Path(q['static_root'])
    boundary=c.Boundary(q);previous={s:signal.signal(s,boundary.signal) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGUSR1)}
    completed=[];owned=[];engine=None;context={'active':False}
    try:
        with RunStore(root,{'profile':q,'profile_sha256':digest(q)},'gemma','static-current',q['sample_ids']) as store:
            session=ext.GemmaSession(store)
            try:
                from PIL import Image
                from transformers.feature_extraction_utils import BatchFeature
                from vllm import LLM,SamplingParams
                from vllm.sampling_params import StructuredOutputsParams
                from vllm.model_executor.models.gemma4_mm import Gemma4MultiModalProcessor
                original=Gemma4MultiModalProcessor._apply_hf_processor_main;pil=Image.fromarray
                def no_pil(*a,**kw):
                    g.require(not context['active'],'RGB16 image attempted PIL conversion');return pil(*a,**kw)
                def prepared(this,mm_items,hf_kwargs):
                    if not context['active']:return original(this,mm_items,hf_kwargs)
                    import numpy as np
                    g.require(mm_items.get_all_counts()=={'image':1},'Image count differs')
                    actual=mm_items['image'].get_processor_data()['images'];g.require(len(actual)==1,'Image count differs')
                    actual=actual[0];expected=context['values']
                    g.require(isinstance(actual,np.ndarray) and actual.dtype==expected.dtype and
                              actual.shape==expected.shape and np.array_equal(actual,expected),'FP32 image entry changed')
                    context['calls']+=1;return BatchFeature(context['payload'])
                session._patch(Image,'fromarray',no_pil);session._patch(Gemma4MultiModalProcessor,'_apply_hf_processor_main',prepared)
                session.torch.set_num_threads(8)
                with g.physical_lease(q['runtime'],session.torch) as lease:
                    store.event('LEASE_ACQUIRED',lease=lease)
                    try:
                        started=time.monotonic();engine=LLM(**q['settings']);owned=engine.collective_rpc(g._worker_identity)
                        store.event('OWNED_ENGINE_IDENTITIES',identities=owned)
                        store.event('MODEL_LOADED',seconds=time.monotonic()-started,placement=engine.collective_rpc(placement),
                            kernel=engine.collective_rpc(g._install_kernel,args=(q['kernel'],str(root))),observer=engine.collective_rpc(install_observer))
                        sampling=SamplingParams(n=1,temperature=0.,top_p=1.,seed=0,repetition_penalty=1.,max_tokens=16384,
                                                structured_outputs=StructuredOutputsParams(choice=['LEFT','RIGHT']))
                        for sid in q['sample_ids']:
                            if boundary.check():break
                            row=rows[sid];image=images[row['image_id']];deep=len(completed)<2
                            store.begin(sid,digest({'row':row,'image':image}));began=time.monotonic()
                            data,receipt=session.prepare_image(row,image,deep);_,values=inputs.read_image(image)
                            if deep:
                                pixels=data['pixel_values']
                                g.require(g.sha_tensor(pixels.to(session.torch.bfloat16))==receipt['expected_BF16_GPU_sha256'],'Retained BF16 reference changed')
                                receipt['expected_normalized_projection_BF16_sha256']=g.sha_tensor(2*(pixels.to(session.torch.bfloat16)-0.5))
                            expected=data['input_ids'][0].tolist();text=receipt['chat_template_text']
                            context.update(active=True,values=values,calls=0,payload=image_payload(data))
                            engine.collective_rpc(set_deep,args=(deep,));infer=time.monotonic()
                            answer=engine.generate({'prompt':text,'multi_modal_data':{'image':values}},sampling,use_tqdm=False)[0]
                            choice=answer.outputs[0]
                            raw={'task':row['task'],'truth':row['truth'],'source_parameters':row['source_parameters'],
                                'image_id':row['image_id'],'prompt_id':row['prompt_id'],'prompt':row['prompt'],'prompt_sha256':digest(row['prompt']),
                                'image_identity':image,'input_receipt':receipt,'raw_answer':choice.text,'output_token_ids':list(choice.token_ids),
                                'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason,'actual_input_token_ids':list(answer.prompt_token_ids),
                                'expected_input_token_ids':expected,'actual_input_tokens':len(answer.prompt_token_ids),'bridge_calls':context['calls'],
                                'seconds':time.monotonic()-infer,'prepare_to_raw_seconds':time.monotonic()-began,'sampling_params_repr':repr(sampling)}
                            committed=store.commit_output(sid,raw)
                            post={'actual_GPU_visual_forward':[r for group in engine.collective_rpc(read_observer) for r in group]}
                            store.event('POST_OUTPUT_INPUT_RECEIPT',sample_id=sid,attempt_id=committed['attempt_id'],**post)
                            validate_output(raw,post)
                            store.event('TECHNICAL_PASS',sample_id=sid,attempt_id=committed['attempt_id'],raw_sha256=committed['raw_record_sha256'])
                            completed.append(sid);context.clear();context['active']=False;del data,values,answer;gc.collect()
                        store.event('SHARD_FINISHED' if len(completed)==len(q['sample_ids']) else 'PAUSED',completed=len(completed),expected=len(q['sample_ids']))
                    finally:
                        if engine is not None:
                            if not owned:owned=engine.collective_rpc(g._worker_identity)
                            g.shutdown_owned_engine(engine,owned,root)
                        store.event('MODEL_CLOSED_BEFORE_LEASE_RELEASE')
                store.event('LEASE_RELEASED')
            except BaseException as error:
                store.event('FAILED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
            finally:context['active']=False;session.close()
        result={'status':'COMPLETE' if len(completed)==len(q['sample_ids']) else 'PAUSED','completed_ids':completed,
                'maximum_calls':len(q['sample_ids']),'profile_sha256':digest(q),'automatic_resume':False}
        atomic_json(root/'result.json',result);return result
    finally:
        for s,h in previous.items():signal.signal(s,h)


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    s=sub.add_parser('seal')
    for name in ['video-profile','static-root','static-code','pause-file','output']:s.add_argument('--'+name,required=True)
    s.add_argument('--ids')
    s=sub.add_parser('run');s.add_argument('--profile',required=True);s.add_argument('--output',required=True)
    a=ap.parse_args();print(json.dumps(seal(a) if a.command=='seal' else run(a)))

if __name__=='__main__':main()
