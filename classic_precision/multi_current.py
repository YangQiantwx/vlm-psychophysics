"""Current Cosmos NVFP4 / Qwen QSA on the frozen 960 T1 image controls."""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from . import static_current as sc
from . import nvfp4_cosmos_official as o
from . import gemma_speed as g
from .storage import RunStore,digest,file_sha256,atomic_json


def validate_rows(rows):
    o.require(len(rows)==len({r['sample_id'] for r in rows})==960,'Expected960 unique existing image controls')
    o.require(Counter(r['arm'] for r in rows)=={'I2_P1':320,'I4_P1':320,'I2_P2':320},'Arm denominators differ')
    for r in rows:
        o.require(r['task']=='T1' and r['truth'] in ('LEFT','RIGHT'),'Task/answer differs')
        o.require([x['frame_index'] for x in r['frame_assets']]==([0,1,2,3] if r['arm']=='I4_P1' else [0,3]),'Ordered source images differ')


def seal(args):
    native=Path(args.native);root=native.parent;manifest=json.loads((root/'evaluation_manifest.json').read_text())
    validate_rows(manifest['rows']);runtime=json.loads(Path(args.video_plan).read_text())['runtime']
    q={'schema':'current-T1-multi-v1','source_sha256':file_sha256(__file__),'shared_source_sha256':file_sha256(sc.__file__),
       'family':args.family,'native':sc.descriptor(native),'manifest':sc.descriptor(root/'evaluation_manifest.json'),
       'protocol':sc.descriptor(root/'evaluation_protocol.json'),'video_plan':sc.descriptor(args.video_plan),'runtime':runtime,
       'maximum_calls':960,'automatic_retry':False,'GPU':args.gpu,'first_deep_per_image_count':2,
       'initialization_policy':{'enable_flashinfer_autotune':False},'native_profile':json.loads(Path(args.native_profile).read_text()) if args.native_profile else None,
       'coverage':sc.descriptor(args.coverage) if args.coverage else None}
    if args.family=='qwen':
        o.require(q['native_profile']['mode']=='qwen_qsa_only' and q['native_profile']['environment'].get('CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB','24')=='24','Native multi QSA24 placement required')
    o.require(not Path(args.output).exists(),'No seal overwrite');atomic_json(args.output,q)
    return {'status':'SEALED','calls':960,'profile_sha256':digest(q),'GPU_calls':0}


def load_native(q):
    d=q['native'];o.require(file_sha256(d['path'])==d['sha256'],'Native source changed')
    spec=importlib.util.spec_from_file_location('current_native_T1_images',d['path']);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    from .adapter_qsa_replay_ipc import register_native_module
    register_native_module(module,d['sha256']);return module


def cosmos_session(native,args,store,spec,runtime):
    session=native.make_session(o.inference,args,store,spec)
    def load(resources):
        import vllm
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        o._configure_torch(session.torch);session._owns_cuda=True
        settings=o.engine_settings(runtime);settings['limit_mm_per_prompt']={'image':4,'video':0,'audio':0}
        settings['mm_processor_kwargs']={'do_resize':False}
        began=time.monotonic();session.engine=vllm.LLM(**settings)
        actual=session.engine.collective_rpc(o._engine_receipt);o.validate_engine_receipt(actual)
        store.event('MODEL_LOADED',seconds=time.monotonic()-began,settings=settings,actual=actual)
        session.sampling=SamplingParams(n=1,temperature=0.,top_p=1.,repetition_penalty=1.,seed=0,max_tokens=16384,
                                       structured_outputs=StructuredOutputsParams(choice=['LEFT','RIGHT']))
        session.effective_generation={**o.inference.GENERATION,'backend':'vllm_official_NVFP4_images','sampling_params_repr':repr(session.sampling)}
    def generate(row,arrays,ready,durable,deep):
        began=time.monotonic();values,data,text,receipt,expected=session.prepare_images(row,arrays,deep)
        session.engine.collective_rpc(native.t4.install_probe,args=(deep,));infer=time.monotonic()
        answer=session.engine.generate({'prompt':text,'multi_modal_data':{'image':values}},session.sampling,use_tqdm=False)[0]
        choice=answer.outputs[0];actual=list(answer.prompt_token_ids)
        raw={'task':'T1','truth':row['truth'],'arm':row['arm'],'prompt_id':row['prompt_id'],'parent_sample_id':row['parent_sample_id'],
             'prompt':row['prompt'],'prompt_sha256':digest(row['prompt']),'source_parameters':row['source_parameters'],
             'row_identity':digest({'row':row,'ready':ready}),'ready_identity':digest(ready),'ready_record':ready,
             'input_receipt':receipt,'effective_generation':session.effective_generation,'raw_answer':choice.text,
             'output_token_ids':list(choice.token_ids),'actual_prompt_token_ids':actual,'actual_input_tokens':len(actual),
             'finish_reason':choice.finish_reason,'stop_reason':choice.stop_reason,'seconds':time.monotonic()-infer,
             'prepare_to_raw_seconds':time.monotonic()-began}
        committed=durable(raw);post={'actual_GPU_visual_forward':[x for group in session.engine.collective_rpc(native.t4.read_probe) for x in group],
            'actual_engine_processor_receipt':None,'actual_prompt_tokens':{'all_token_ids_equal':actual==expected,'sha256':digest(actual)}}
        store.event('POST_OUTPUT_INPUT_RECEIPT',sample_id=row['sample_id'],attempt_id=committed['attempt_id'],**post)
        native.validate_post(raw,post,'cosmos');store.event('TECHNICAL_PASS',sample_id=row['sample_id'],attempt_id=committed['attempt_id'])
        gc.collect()
    session.load=load;session.generate_images=generate;return session


def run(args):
    q=json.loads(Path(args.profile).read_text());o.require('torch' not in sys.modules and 'vllm' not in sys.modules,'Fresh process required')
    o.require(q['source_sha256']==file_sha256(__file__) and q['shared_source_sha256']==file_sha256(sc.__file__) and not q['automatic_retry'],'Source/retry differs')
    manifest=sc.read_bound(q['manifest']);protocol=sc.read_bound(q['protocol']);sc.read_bound(q['video_plan']);validate_rows(manifest['rows'])
    helper=None
    if q['family']=='qwen':
        from runtime_compat import classic_precision_adapter_qsa as helper
        profile=q['native_profile'];o.require(all(os.environ.get(k)==v for k,v in profile['environment'].items()) and all(k not in os.environ for k in profile['cleared_environment']),'Native QSA environment differs')
        sc.read_bound(q['coverage']);helper.activate(q['coverage']['path'],profile['baseline_environment'])
        before={k:os.environ.get(k) for k in helper.THREAD_ENVIRONMENT}
    native=load_native(q)
    if helper:
        after={k:os.environ.get(k) for k in helper.THREAD_ENVIRONMENT}
        helper.bind_native_import('multi',{'adapter_sha256':q['native']['sha256'],'before_import':before,'after_native_import':after,'before_native_processor':after})
    rows,deep_ids=native.qualification_order(manifest['rows']);ids=[r['sample_id'] for r in rows]
    root=Path(args.output);o.require(not root.exists(),'No implicit retry/output overwrite');root.mkdir(parents=True)
    family=q['family'];load_args=SimpleNamespace(model=family,models=str(Path(q['runtime']['original_checkpoint']).parents[1]),cosmos_package=q['runtime']['cosmos_package'])
    with RunStore(root,{'profile':q,'profile_sha256':digest(q)},family,'multi-current',ids) as store:
        session=cosmos_session(native,load_args,store,protocol['models'][family],q['runtime']) if family=='cosmos' else native.make_session(o.inference,load_args,store,protocol['models'][family])
        owned=[];handle=None
        try:
            with o.native.physical_lease(q['runtime'],session.torch) as lease:
                expected=q['runtime']['physical_locks'][str(q['GPU'])]
                o.require(all(lease[k]==expected[k] for k in ('path','st_dev','st_ino')),'Native lock differs');store.event('LEASE_ACQUIRED',lease=lease)
                try:
                    if helper:
                        import vllm
                        with sc.bounded_qwen_initialization(vllm):session.load(protocol['models'][family]['resources'])
                        actual=session.engine.collective_rpc(sc.qwen_initialization_receipt)
                        o.require(all(x['enable_flashinfer_autotune'] is False and x['kv_cache_dtype']=='bfloat16' and x['max_model_len']==65536 for x in actual),'Actual initialization differs')
                        store.event('CURRENT_QSA_INITIALIZATION_POLICY',actual=actual)
                        handle=helper.instrument_session(session,'multi',native,q['coverage']['path']);baseline=helper.snapshot()['comparison_identity']
                        store.event('CURRENT_QSA_RUNTIME',snapshot=baseline)
                    else:session.load(protocol['models'][family]['resources'])
                    owned=session.engine.collective_rpc(g._worker_identity);store.event('OWNED_ENGINE_IDENTITIES',identities=owned)
                    for i,row in enumerate(rows):
                        sid=row['sample_id'];arrays,ready=native.read_images(row);store.begin(sid,digest({'row':row,'ready':ready}));began=time.monotonic()
                        if helper:o.require(helper.snapshot()['comparison_identity']==baseline,'QSA runtime changed')
                        session.generate_images(row,arrays,ready,lambda raw:store.commit_output(sid,raw),deep=sid in deep_ids)
                        raw=store.outputs[sid];sc.validate_native_stop(raw,session.tokenizer)
                        if helper:o.require(helper.snapshot()['comparison_identity']==baseline,'QSA runtime changed after request')
                        store.event('CURRENT_TECHNICAL_PASS',sample_id=sid,attempt_id=raw['attempt_id'],raw_sha256=digest(raw),full_request_seconds=time.monotonic()-began)
                        if handle and i==len(deep_ids)-1:handle.close();handle=None
                    store.event('SHARD_FINISHED',completed=len(store.outputs),expected=960)
                finally:
                    if handle:handle.close()
                    engine=getattr(session,'engine',None)
                    if engine is not None:
                        if not owned:owned=engine.collective_rpc(g._worker_identity)
                        g.shutdown_owned_engine(engine,owned,root);session.engine=None
                    session.close();store.event('MODEL_CLOSED_BEFORE_LEASE_RELEASE')
            store.event('LEASE_RELEASED')
        except BaseException as error:
            store.event('FAILED_NO_RETRY',error_type=type(error).__name__,error=str(error));raise
    atomic_json(root/'result.json',{'status':'COMPLETE','calls':960,'profile_sha256':digest(q)});return {'status':'COMPLETE','calls':960}


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True);s=sub.add_parser('seal')
    for k in ['native','video-plan','output']:s.add_argument('--'+k,required=True)
    s.add_argument('--family',choices=['qwen','cosmos'],required=True);s.add_argument('--gpu',type=int,required=True)
    s.add_argument('--native-profile');s.add_argument('--coverage');s=sub.add_parser('run');s.add_argument('--profile',required=True);s.add_argument('--output',required=True)
    a=ap.parse_args();print(json.dumps(seal(a) if a.command=='seal' else run(a)))

if __name__=='__main__':main()
