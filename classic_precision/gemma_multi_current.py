"""Official Gemma NVFP4 on retained ordered T1 images; bounded, no retries."""
from __future__ import annotations
import argparse,gc,json,os,signal,time,uuid
from pathlib import Path
from types import SimpleNamespace
import types
from . import gemma_static_current as single
from . import gemma_nvfp4_continuation as c
from . import gemma_speed as g
from . import multi_current as multi
from . import static_current as sc
from . import inference
from .storage import RunStore,atomic_json,digest,file_sha256

SCHEMA='gemma-current-multi-nvfp4-v1'
def image_payload(data):
    import torch
    return {'pixel_values':data['pixel_values'].to(dtype=torch.bfloat16),'pixel_position_ids':data['image_position_ids']}
def install_observer(worker):
    # The accepted single-image observer, scoped here for uniform two/four-image
    # records even on hosts retaining an earlier single-image module release.
    model=worker.get_model();worker._gemma_image={'deep':True,'rows':[]};original=model._process_image_input
    def projection(layer,args):
        state=worker._gemma_image;x=args[0];row={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)}
        if state['deep']:row['tensor_sha256']=g.sha_tensor(x)
        state['projection']=row
    model.vision_tower.patch_embedder.input_proj.register_forward_pre_hook(projection)
    def image(this,inputs):
        state=worker._gemma_image;x=inputs['pixel_values'];pos=inputs['pixel_position_ids'];row={'shape':list(x.shape),'dtype':str(x.dtype),'device':str(x.device)}
        if state['deep']:row.update(tensor_sha256=g.sha_tensor(x),positions_sha256=g.sha_tensor(pos))
        result=original(inputs);row['projection']=state.pop('projection');row['embedding_shapes']=[list(t.shape) for t in result];state['rows'].append(row);return result
    model._process_image_input=types.MethodType(image,model)
    return {'status':'INSTALLED','input_and_native_projection_observed':True}
def settings(anchor):
    x=single.settings(anchor);x['limit_mm_per_prompt']={'video':0,'image':4,'audio':0};return x
def seal(a):
    anchor=g.read(a.video_profile);native=Path(a.native);manifest=native.parent/'evaluation_manifest.json';rows=g.read(manifest)['rows'];multi.validate_rows(rows)
    ids=g.read(a.ids);assert ids and len(ids)==len(set(ids)) and set(ids)<={r['sample_id'] for r in rows}
    selected=[r for r in rows if r['sample_id'] in set(ids)];ordered,deep=multi.load_native({'native':sc.descriptor(native)}).qualification_order(selected)
    q={'schema':SCHEMA,'source_sha256':file_sha256(__file__),'shared_sources':{Path(m.__file__).name:file_sha256(m.__file__) for m in [single,c,g,multi,sc]},'video_profile':sc.descriptor(a.video_profile),'runtime':anchor['runtime'],'candidate':anchor['candidate'],'kernel':anchor['kernel'],'runtime_sources':anchor['runtime_sources'],'settings':settings(anchor),'native':sc.descriptor(native),'manifest':sc.descriptor(manifest),'protocol':sc.descriptor(native.parent/'evaluation_protocol.json'),'sample_ids':[r['sample_id'] for r in ordered],'deep_ids':list(deep),'maximum_calls':len(ids),'automatic_retry':False,'worker_token':uuid.uuid4().hex,'pause_file':str(Path(a.pause_file).resolve())}
    assert not Path(a.output).exists();atomic_json(a.output,q);return {'status':'SEALED','calls':len(ids),'deep_calls_inside_budget':len(deep),'GPU_calls':0}
def validate(raw,post):
    r=raw['input_receipt'];n=r['images_per_context'];assert n in [2,4]
    assert r['source_frames']==([0,1,2,3] if n==4 else [0,3])
    assert raw['actual_prompt_token_ids']==raw['expected_prompt_token_ids'] and len(raw['actual_prompt_token_ids'])==r['input_tokens']
    assert len(post)==1 and post[0]['shape']==[n,2520,768] and post[0]['dtype']=='torch.bfloat16' and post[0]['device'].startswith('cuda')
    assert post[0]['projection']['shape']==[n,2520,768] and post[0]['projection']['dtype']=='torch.bfloat16' and post[0]['projection']['device'].startswith('cuda')
    assert raw['raw_answer'] in ['LEFT','RIGHT'] and raw['finish_reason']=='stop' and raw['output_token_ids'][-1:]==[106]
    if r['deep_check']:
        assert post[0]['tensor_sha256']==r['expected_BF16_GPU_sha256'] and post[0]['positions_sha256']==r['expected_position_ids_sha256']
        assert post[0]['projection']['tensor_sha256']==r['expected_normalized_projection_BF16_sha256']
def run(a):
    assert 'torch' not in __import__('sys').modules and 'vllm' not in __import__('sys').modules
    q=g.read(a.profile);assert q['schema']==SCHEMA and q['source_sha256']==file_sha256(__file__) and not q['automatic_retry']
    for m in [single,c,g,multi,sc]:assert q['shared_sources'][Path(m.__file__).name]==file_sha256(m.__file__)
    anchor=sc.read_bound(q['video_profile']);assert q['runtime']==anchor['runtime'] and q['settings']==settings(anchor)
    c.sources();c.runtime_check(q['runtime']);assert g.runtime_sources()==q['runtime_sources']
    manifest=sc.read_bound(q['manifest']);protocol=sc.read_bound(q['protocol']);rows={r['sample_id']:r for r in manifest['rows']}
    native=multi.load_native(q);spec=protocol['models']['gemma'];load_args=SimpleNamespace(model='gemma',models=str(Path(q['runtime']['processor_checkpoint']).parents[1]),cosmos_package='')
    root=Path(a.output);assert not root.exists();root.mkdir(parents=True)
    boundary=c.Boundary(q);old={s:signal.signal(s,boundary.signal) for s in [signal.SIGINT,signal.SIGTERM,signal.SIGUSR1]};context={'active':False};owned=[];engine=None;completed=[]
    try:
        with RunStore(root,{'profile':q,'profile_sha256':digest(q)},'gemma','multi-current',q['sample_ids']) as store:
            Processor=native.make_processor_class(inference);session=Processor('gemma',Path(q['runtime']['processor_checkpoint']),cosmos_package='')
            try:
                import numpy as np
                from PIL import Image
                from transformers.feature_extraction_utils import BatchFeature
                from vllm import LLM,SamplingParams
                from vllm.sampling_params import StructuredOutputsParams
                from vllm.model_executor.models.gemma4_mm import Gemma4MultiModalProcessor
                original=Gemma4MultiModalProcessor._apply_hf_processor_main;pil=Image.fromarray
                def no_pil(*args,**kwargs):
                    assert not context['active'],'Unexpected uint8/PIL conversion';return pil(*args,**kwargs)
                def prepared(this,mm_items,hf_kwargs):
                    if not context['active']:return original(this,mm_items,hf_kwargs)
                    expected=context['values'];assert mm_items.get_all_counts()=={'image':len(expected)}
                    actual=mm_items['image'].get_processor_data()['images'];assert len(actual)==len(expected)
                    for x,y in zip(actual,expected):assert isinstance(x,np.ndarray) and x.dtype==y.dtype and x.shape==y.shape and np.array_equal(x,y)
                    context['calls']+=1;return BatchFeature(context['payload'])
                session._patch(Image,'fromarray',no_pil);session._patch(Gemma4MultiModalProcessor,'_apply_hf_processor_main',prepared);session.torch.set_num_threads(8)
                with g.physical_lease(q['runtime'],session.torch) as lease:
                    store.event('LEASE_ACQUIRED',lease=lease)
                    try:
                        started=time.monotonic();engine=LLM(**q['settings']);owned=engine.collective_rpc(g._worker_identity);store.event('OWNED_ENGINE_IDENTITIES',identities=owned)
                        store.event('MODEL_LOADED',seconds=time.monotonic()-started,placement=engine.collective_rpc(single.placement),kernel=engine.collective_rpc(g._install_kernel,args=(q['kernel'],str(root))),observer=engine.collective_rpc(install_observer))
                        sampling=SamplingParams(n=1,temperature=0.,top_p=1.,seed=0,repetition_penalty=1.,max_tokens=16384,structured_outputs=StructuredOutputsParams(choice=['LEFT','RIGHT']))
                        for sid in q['sample_ids']:
                            if boundary.check():break
                            row=rows[sid];arrays,ready=native.read_images(row);deep=sid in q['deep_ids'];store.begin(sid,digest({'row':row,'ready':ready}));start=time.monotonic()
                            values,data,text,receipt,expected=session.prepare_images(row,arrays,deep)
                            if deep:receipt['expected_normalized_projection_BF16_sha256']=g.sha_tensor(2*(data['pixel_values'].to(session.torch.bfloat16)-.5))
                            context.update(active=True,values=values,payload=image_payload(data),calls=0);engine.collective_rpc(single.set_deep,args=(deep,));began=time.monotonic()
                            answer=engine.generate({'prompt':text,'multi_modal_data':{'image':values}},sampling,use_tqdm=False)[0];choice=answer.outputs[0]
                            raw={'task':'T1','truth':row['truth'],'arm':row['arm'],'prompt_id':row['prompt_id'],'parent_sample_id':row['parent_sample_id'],'prompt':row['prompt'],'prompt_sha256':digest(row['prompt']),'source_parameters':row['source_parameters'],'ready_record':ready,'input_receipt':receipt,'raw_answer':choice.text,'output_token_ids':list(choice.token_ids),'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason,'actual_prompt_token_ids':list(answer.prompt_token_ids),'expected_prompt_token_ids':expected,'actual_input_tokens':len(answer.prompt_token_ids),'bridge_calls':context['calls'],'seconds':time.monotonic()-began,'prepare_to_raw_seconds':time.monotonic()-start}
                            committed=store.commit_output(sid,raw);post=[r for group in engine.collective_rpc(single.read_observer) for r in group];store.event('POST_OUTPUT_INPUT_RECEIPT',sample_id=sid,attempt_id=committed['attempt_id'],actual_GPU_visual_forward=post)
                            validate(raw,post);store.event('TECHNICAL_PASS',sample_id=sid,attempt_id=committed['attempt_id'],raw_sha256=committed['raw_record_sha256']);completed.append(sid)
                            if len(completed)==len(q['deep_ids']):atomic_json(root/'qualification.json',{'status':'PASS','deep_ids':q['deep_ids'],'image_counts':[2,4],'full_input_prompt_position_projection_match':True,'extra_GPU_calls':0})
                            context.clear();context['active']=False;del arrays,values,data,answer;gc.collect()
                        store.event('SHARD_FINISHED' if len(completed)==len(q['sample_ids']) else 'PAUSED',completed=len(completed),expected=len(q['sample_ids']))
                    finally:
                        if engine is not None:g.shutdown_owned_engine(engine,owned,root)
                        store.event('MODEL_CLOSED_BEFORE_LEASE_RELEASE')
                store.event('LEASE_RELEASED')
            except BaseException as e:store.event('FAILED_NO_RETRY',error_type=type(e).__name__,error=str(e));raise
            finally:session.close()
        result={'status':'COMPLETE' if len(completed)==len(q['sample_ids']) else 'PAUSED','completed_ids':completed,'maximum_calls':len(q['sample_ids']),'automatic_resume':False};atomic_json(root/'result.json',result);return {'status':result['status'],'completed':len(completed)}
    finally:
        for s,h in old.items():signal.signal(s,h)
def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True);s=sub.add_parser('seal')
    for n in ['video-profile','native','ids','pause-file','output']:s.add_argument('--'+n,required=True)
    s=sub.add_parser('run');s.add_argument('--profile',required=True);s.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(seal(a) if a.command=='seal' else run(a)))
if __name__=='__main__':main()
