"""Current Cosmos NVFP4 on the existing, immutable RGB16 image panel.

Reuses the previously qualified static processor and visual-input probe. Only
the model backend is replaced with the qualified official Cosmos NVFP4 backend.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from . import nvfp4_cosmos_official as o
from . import gemma_speed as lifecycle
from .storage import RunStore, digest, file_sha256, atomic_json

SCHEMA = 'current-static-cosmos-nvfp4-v1'
QWEN_SCHEMA = 'current-static-qwen-qsa-v1'
COSMOS_SHA = '04cfb6e0f5bd581f8791d674c43aadaaeeb6da536256826f5c1b5a901359145e'


@contextmanager
def bounded_qwen_initialization(vllm):
    """Skip the failing dummy autotune; retain every native loader argument."""
    original=vllm.LLM;calls=[]
    def load(*args,**kwargs):
        o.require('kernel_config' not in kwargs,'Unexpected native kernel policy')
        kwargs=dict(kwargs,kernel_config={'enable_flashinfer_autotune':False})
        calls.append({'kernel_config':kwargs['kernel_config'],'max_num_batched_tokens':kwargs['max_num_batched_tokens']})
        return original(*args,**kwargs)
    vllm.LLM=load
    try:yield calls
    finally:vllm.LLM=original


def qwen_initialization_receipt(worker):
    config=worker.vllm_config
    return {'enable_flashinfer_autotune':config.kernel_config.enable_flashinfer_autotune,
            'max_num_batched_tokens':config.scheduler_config.max_num_batched_tokens,
            'max_model_len':config.model_config.max_model_len,'kv_cache_dtype':config.cache_config.cache_dtype}


def settings(runtime):
    value=o.engine_settings(runtime)
    value['limit_mm_per_prompt']={'video':0,'image':1,'audio':0}
    value['mm_processor_kwargs']={'do_resize':False}
    return value


def ordered_rows(manifest):
    rows=manifest['rows'];ids=[x['sample_id'] for x in rows]
    o.require(len(ids)==len(set(ids))==456,'Expected 456 unique image/prompt rows')
    o.require(Counter(x['task'] for x in rows)=={'T2':234,'T3':222},'Static task coverage differs')
    o.require(Counter(x['prompt_id'] for x in rows)=={'P0':152,'P1':152,'P2':152},'Prompt coverage differs')
    o.require(len(manifest['images'])==152,'Static image coverage differs')
    first=[next(x for x in rows if x['task']==task and x['prompt_id']=='P1') for task in ['T2','T3']]
    selected={x['sample_id'] for x in first}
    return first+[x for x in rows if x['sample_id'] not in selected]


def descriptor(path):return {'path':str(Path(path).resolve()),'sha256':file_sha256(path)}


def read_bound(value):
    o.require(file_sha256(value['path'])==value['sha256'],'Bound source changed')
    return json.loads(Path(value['path']).read_text())


def seal(args):
    original=Path(args.static_root);code=Path(args.static_code)
    manifest=json.loads((original/'execution-manifest.json').read_text());rows=ordered_rows(manifest)
    runtime=json.loads(Path(args.video_plan).read_text())['runtime']
    o.require(file_sha256(o.__file__)==COSMOS_SHA,'Qualified Cosmos source changed')
    value={'schema':SCHEMA,'source_sha256':file_sha256(__file__),'cosmos_source_sha256':COSMOS_SHA,
        'video_plan':descriptor(args.video_plan),'manifest':descriptor(original/'execution-manifest.json'),
        'protocol':descriptor(original/'protocol.json'),'static_root':str(original.resolve()),
        'static_sources':{n:descriptor(code/n) for n in ['static_inputs.py','static_worker.py']},
        'runtime':runtime,'settings':settings(runtime),'sample_ids':[x['sample_id'] for x in rows],
        'GPU':args.gpu,'maximum_calls':456,'first_deep_calls':2,'automatic_retry':False,
        'scope':'Existing 152 unique T2/T3 images, all three original prompts; no T1 static direction task'}
    o.require(not Path(args.output).exists(),'No seal overwrite');atomic_json(args.output,value)
    return {'status':'SEALED','calls':456,'images':152,'GPU_calls':0,'profile_sha256':digest(value)}


def load_static(q):
    # The retained modules use explicit filesystem constants. Repoint those
    # constants to the same existing host assets; do not rewrite their source.
    for name,desc in q['static_sources'].items():
        o.require(file_sha256(desc['path'])==desc['sha256'],'Static input/probe source changed')
        spec=importlib.util.spec_from_file_location(name[:-3],desc['path'])
        module=importlib.util.module_from_spec(spec);sys.modules[name[:-3]]=module;spec.loader.exec_module(module)
        if name=='static_inputs.py':
            module.RUN=Path(q['static_root'])
            module.MODELS=Path(q['runtime']['original_checkpoint']).parents[1]
            module.COSMOS=Path(q['runtime']['cosmos_package'])
            inputs=module
        else:worker=module
    return inputs,worker


def make_session(q,store,inputs,worker):
    class Session(inputs.ImageProcessor):
        def load_current(self):
            from vllm import LLM,SamplingParams
            from vllm.sampling_params import StructuredOutputsParams
            o._configure_torch(self.torch);self._owns_cuda=True
            self.engine=LLM(**q['settings'])
            actual=self.engine.collective_rpc(o._engine_receipt);o.validate_engine_receipt(actual)
            self.store.event('MODEL_LOADED',settings=q['settings'],actual=actual)
            self.sampling=SamplingParams(n=1,temperature=0.,top_p=1.,seed=0,repetition_penalty=1.,
                max_tokens=16384,structured_outputs=StructuredOutputsParams(choice=['LEFT','RIGHT']))
            self.effective_generation={**o.inference.GENERATION,'backend':'vllm','constraint':'StructuredOutputsParams(choice)',
                'native_eos_token_ids':[151645,151643],'sampling_params_repr':repr(self.sampling)}
            original=type(self.processor).__call__
            def checked(processor,*a,**kw):
                data=original(processor,*a,**kw)
                if self.context.get('engine_active') and 'pixel_values' in data:
                    x=data['pixel_values'];expected=self.context['receipt']
                    o.require(str(x.dtype)=='torch.float32' and list(x.shape)==expected['actual_processor_tensor_shape'],'Engine FP32 image differs')
                    actual=hashlib.sha256(x.contiguous().view(self.torch.uint8).numpy().tobytes()).hexdigest() if expected['deep_check'] else None
                    if expected['deep_check']:o.require(actual==self.context['processor_sha256'],'Engine processor values differ')
                    self.context['engine_receipt']={'shape':list(x.shape),'dtype':str(x.dtype),'deep_sha256':actual}
                return data
            self._patch(type(self.processor),'__call__',checked)

        def generate_current(self,row,image,deep):
            began=time.monotonic();values,data,text,receipt=self.prepare_image(row,image,deep)
            expected_ids=data['input_ids'][0].tolist();pixels=data['pixel_values']
            self.context.update(engine_active=True,receipt=receipt,engine_receipt=None,
                processor_sha256=hashlib.sha256(pixels.contiguous().view(self.torch.uint8).numpy().tobytes()).hexdigest() if deep else None)
            del data,pixels
            self.engine.collective_rpc(worker.install_probe,args=(deep,))
            infer_start=time.monotonic()
            answer=self.engine.generate({'prompt':text,'multi_modal_data':{'image':values}},self.sampling,use_tqdm=False)[0]
            output=answer.outputs[0];elapsed=time.monotonic()-infer_start
            raw={'task':row['task'],'truth':row['truth'],'source_parameters':row['source_parameters'],
                'image_id':row['image_id'],'prompt_id':row['prompt_id'],'prompt':row['prompt'],'prompt_sha256':digest(row['prompt']),
                'image_identity':image,'input_receipt':receipt,'effective_generation':self.effective_generation,
                'raw_answer':output.text,'output_token_ids':list(output.token_ids),'finish_reason':output.finish_reason,'stop_reason':output.stop_reason,
                'actual_input_tokens':len(answer.prompt_token_ids),'actual_input_token_ids':list(answer.prompt_token_ids),
                'all_prompt_token_ids_equal':list(answer.prompt_token_ids)==expected_ids,'seconds':elapsed,
                'prepare_to_raw_seconds':time.monotonic()-began,'deep':deep}
            committed=store.commit_output(row['sample_id'],raw)
            post={'actual_GPU_visual_forward':[x for group in self.engine.collective_rpc(worker.read_probe) for x in group],
                  'actual_engine_processor_receipt':self.context['engine_receipt']}
            store.event('POST_OUTPUT_INPUT_RECEIPT',sample_id=row['sample_id'],attempt_id=committed['attempt_id'],**post)
            worker.validate_post(raw,post,'cosmos')
            o.require(raw['all_prompt_token_ids_equal'] and self.context['engine_receipt'] is not None,'Actual processor/fullprompt differs')
            o.require(output.text in ('LEFT','RIGHT') and output.finish_reason=='stop' and output.token_ids[-1] in (151645,151643),'Not exact LR/native EOS')
            store.event('TECHNICAL_PASS',sample_id=row['sample_id'],attempt_id=committed['attempt_id'],raw_sha256=committed['raw_record_sha256'])
            self.context.update(engine_active=False,receipt=None);gc.collect()
    return Session('cosmos',store)


def run(args):
    q=json.loads(Path(args.profile).read_text())
    if q['schema']==QWEN_SCHEMA:return run_qwen(args,q)
    o.require(q['schema']==SCHEMA and q['source_sha256']==file_sha256(__file__) and file_sha256(o.__file__)==COSMOS_SHA,'Entry source differs')
    o.require(q['settings']==settings(q['runtime']) and not q['automatic_retry'],'Settings/retry changed')
    read_bound(q['video_plan']);read_bound(q['protocol']);manifest=read_bound(q['manifest']);rows=ordered_rows(manifest)
    o.require([x['sample_id'] for x in rows]==q['sample_ids'],'Assigned IDs differ')
    root=Path(args.output);o.require(not root.exists(),'No implicit retry or output overwrite');root.mkdir(parents=True)
    inputs,worker=load_static(q);images={x['image_id']:x for x in manifest['images']}
    identity={'profile':q,'profile_sha256':digest(q),'generation':o.inference.GENERATION}
    with RunStore(root,identity,'cosmos','static-current',q['sample_ids']) as store:
        session=make_session(q,store,inputs,worker);torch=session.torch;owned=[];engine=None
        try:
            with o.native.physical_lease(q['runtime'],torch) as lease:
                expected=q['runtime']['physical_locks'][str(q['GPU'])]
                o.require(all(lease[k]==expected[k] for k in ['path','st_dev','st_ino']),'Native lock identity differs')
                store.event('LEASE_ACQUIRED',lease=lease)
                try:
                    session.load_current();engine=session.engine;owned=engine.collective_rpc(lifecycle._worker_identity)
                    store.event('OWNED_ENGINE_IDENTITIES',identities=owned)
                    for i,row in enumerate(rows):
                        store.begin(row['sample_id'],digest({'row':row,'image':images[row['image_id']]}))
                        session.generate_current(row,images[row['image_id']],i<2)
                    store.event('SHARD_FINISHED',completed=len(store.outputs),expected=456)
                finally:
                    engine=getattr(session,'engine',None)
                    if engine is not None:
                        if not owned:owned=engine.collective_rpc(lifecycle._worker_identity)
                        lifecycle.shutdown_owned_engine(engine,owned,root);session.engine=None
                    session.close();store.event('MODEL_CLOSED_BEFORE_LEASE_RELEASE')
            store.event('LEASE_RELEASED')
        except BaseException as error:
            store.event('FAILED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    atomic_json(root/'result.json',{'status':'COMPLETE','calls':456,'profile_sha256':digest(q)})
    return {'status':'COMPLETE','calls':456}


def validate_native_stop(raw,tokenizer):
    eos=tokenizer.eos_token_id
    accepted=[eos] if isinstance(eos,int) else list(eos or [])
    o.require(accepted and raw['raw_answer'] in ('LEFT','RIGHT') and raw['finish_reason']=='stop'
              and raw['output_token_ids'] and raw['output_token_ids'][-1] in accepted,'Exact LR/native EOS required')


def qwen_remaining_rows(manifest,q):
    rows=ordered_rows(manifest);completed=[]
    if q.get('completed_reference'):
        receipt=read_bound(q['completed_reference'])
        o.require(receipt['status']=='PASS_POSTVALIDATOR_CORRECTED' and receipt['new_model_calls']==0,'Unverified prior output')
        completed=receipt['completed_ids']
        o.require(len(completed)==len(set(completed))==1,'Only explicit recovered one-output continuation allowed')
    selected=[r for r in rows if r['sample_id'] not in set(completed)]
    o.require([r['sample_id'] for r in selected]==q['sample_ids'] and len(selected)+len(completed)==456 and q['maximum_calls']==len(selected),'Assignment changed')
    return selected


def seal_qwen(args):
    """Keep the qualified native single-image loader and QSA-only policy."""
    from runtime_compat import classic_precision_adapter_qsa as helper
    original=Path(args.static_root);code=Path(args.static_code)
    manifest=json.loads((original/'execution-manifest.json').read_text())
    profile=json.loads(Path(args.native_profile).read_text())
    o.require(profile['mode']=='qwen_qsa_only' and profile['qsa_policy']=='reviewed_exact_selector_only',
              'Current QSA-only profile required')
    o.require(profile['environment']['CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB']=='16',
              'Retain native image placement16; video24 is a separate placement')
    runtime=json.loads(Path(args.video_plan).read_text())['runtime']
    sources={n:descriptor(code/n) for n in ['static_inputs.py','static_worker.py']}
    coverage=read_bound(descriptor(args.coverage))
    entry=next(e for e in coverage['routes'] if e['route_id']=='single')
    allowed={(Path(x['path']).name,x['sha256']) for x in entry['implementation_files']}
    o.require(all((n,d['sha256']) in allowed for n,d in sources.items()),'Unqualified image adapter')
    q={'schema':QWEN_SCHEMA,'source_sha256':file_sha256(__file__),
       'manifest':descriptor(original/'execution-manifest.json'),'protocol':descriptor(original/'protocol.json'),
       'static_root':str(original.resolve()),'static_sources':sources,'runtime':runtime,
       'video_plan':descriptor(args.video_plan),'coverage':descriptor(args.coverage),
       'native_profile':profile,'qsa_helper':descriptor(helper.__file__),
       'GPU':args.gpu,'sample_ids':[r['sample_id'] for r in ordered_rows(manifest)],
       'maximum_calls':456,'first_deep_calls':2,'automatic_retry':False,
       'initialization_policy':{'enable_flashinfer_autotune':False,'reason':'Native49152 dummy MoE autotune OOM; no scientific input change'},
       'placement_note':'Original qualified image expert CPU offload16GiB; video offload24GiB is not a precision change'}
    o.require(not Path(args.output).exists(),'No seal overwrite');atomic_json(args.output,q)
    return {'status':'SEALED','calls':456,'GPU_calls':0,'profile_sha256':digest(q)}


def run_qwen(args,q):
    from runtime_compat import classic_precision_adapter_qsa as helper
    o.require('torch' not in sys.modules and 'vllm' not in sys.modules,'Fresh QSA process required')
    o.require(q['source_sha256']==file_sha256(__file__) and not q['automatic_retry'],'Source/retry differs')
    o.require(file_sha256(helper.__file__)==q['qsa_helper']['sha256'],'QSA helper changed')
    profile=q['native_profile']
    o.require(all(os.environ.get(k)==v for k,v in profile['environment'].items()) and
              all(k not in os.environ for k in profile['cleared_environment']),'Native environment differs')
    read_bound(q['coverage']);read_bound(q['video_plan']);protocol=read_bound(q['protocol'])
    manifest=read_bound(q['manifest']);rows=qwen_remaining_rows(manifest,q)
    root=Path(args.output);o.require(not root.exists(),'No implicit retry');root.mkdir(parents=True)
    helper.activate(q['coverage']['path'],profile['baseline_environment'])
    before={k:os.environ.get(k) for k in helper.THREAD_ENVIRONMENT}
    inputs,worker=load_static(q)
    from .adapter_qsa_replay_ipc import register_native_module
    register_native_module(worker,q['static_sources']['static_worker.py']['sha256'])
    after={k:os.environ.get(k) for k in helper.THREAD_ENVIRONMENT}
    imports={'adapter_sha256':q['static_sources']['static_worker.py']['sha256'],
             'static_inputs_sha256':q['static_sources']['static_inputs.py']['sha256'],
             'before_import':before,'after_native_import':after,'before_native_processor':after}
    helper.bind_native_import('single',imports)
    images={x['image_id']:x for x in manifest['images']}
    with RunStore(root,{'profile':q,'profile_sha256':digest(q)},'qwen','static-current',q['sample_ids']) as store:
        session=worker.StaticSession('qwen',store);handle=None;owned=[]
        try:
            with o.native.physical_lease(q['runtime'],session.torch) as lease:
                expected=q['runtime']['physical_locks'][str(q['GPU'])]
                o.require(all(lease[k]==expected[k] for k in ['path','st_dev','st_ino']),'Native lock differs')
                store.event('LEASE_ACQUIRED',lease=lease)
                try:
                    import vllm
                    o.require(q['initialization_policy']['enable_flashinfer_autotune'] is False,'Initialization policy changed')
                    with bounded_qwen_initialization(vllm) as load_calls:
                        session.load(protocol['models']['qwen']['resources'])
                    actual_init=session.engine.collective_rpc(qwen_initialization_receipt)
                    o.require(len(load_calls)==1 and all(x['enable_flashinfer_autotune'] is False and
                              x['max_model_len']==65536 and x['kv_cache_dtype']=='bfloat16' for x in actual_init),
                              'Actual initialization or precision policy differs')
                    store.event('CURRENT_QSA_INITIALIZATION_POLICY',requested=load_calls,actual=actual_init)
                    owned=session.engine.collective_rpc(lifecycle._worker_identity)
                    store.event('OWNED_ENGINE_IDENTITIES',identities=owned)
                    handle=helper.instrument_session(session,'single',worker,q['coverage']['path'])
                    baseline=helper.snapshot()['comparison_identity']
                    store.event('CURRENT_QSA_RUNTIME',snapshot=baseline,imports=imports)
                    for i,row in enumerate(rows):
                        sid=row['sample_id'];store.begin(sid,digest({'row':row,'image':images[row['image_id']]}))
                        started=time.monotonic()
                        o.require(helper.snapshot()['comparison_identity']==baseline,'QSA runtime changed')
                        session.generate_image(row,images[row['image_id']],lambda raw:store.commit_output(sid,raw),deep=i<2)
                        raw=store.outputs[sid]
                        validate_native_stop(raw,session.tokenizer)
                        o.require(helper.snapshot()['comparison_identity']==baseline,'QSA runtime changed after call')
                        store.event('CURRENT_QSA_TECHNICAL_PASS',sample_id=sid,attempt_id=raw['attempt_id'],
                                    raw_sha256=digest(raw),full_request_seconds=time.monotonic()-started)
                        if i==1:handle.close();handle=None
                    store.event('SHARD_FINISHED',completed=len(store.outputs),expected=len(rows))
                finally:
                    if handle:handle.close()
                    engine=getattr(session,'engine',None)
                    if engine is not None:
                        if not owned:owned=engine.collective_rpc(lifecycle._worker_identity)
                        lifecycle.shutdown_owned_engine(engine,owned,root);session.engine=None
                    session.close();store.event('MODEL_CLOSED_BEFORE_LEASE_RELEASE')
            store.event('LEASE_RELEASED')
        except BaseException as error:
            store.event('FAILED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    atomic_json(root/'result.json',{'status':'COMPLETE','calls':len(rows),'profile_sha256':digest(q)})
    return {'status':'COMPLETE','calls':len(rows)}


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    s=sub.add_parser('seal')
    for name in ['static-root','static-code','video-plan','output']:s.add_argument('--'+name,required=True)
    s.add_argument('--gpu',type=int,default=0)
    s=sub.add_parser('seal-qwen')
    for name in ['static-root','static-code','video-plan','native-profile','coverage','output']:s.add_argument('--'+name,required=True)
    s.add_argument('--gpu',type=int,default=0)
    s=sub.add_parser('run');s.add_argument('--profile',required=True);s.add_argument('--output',required=True)
    a=ap.parse_args();print(json.dumps(seal(a) if a.command=='seal' else seal_qwen(a) if a.command=='seal-qwen' else run(a)))

if __name__=='__main__':main()
