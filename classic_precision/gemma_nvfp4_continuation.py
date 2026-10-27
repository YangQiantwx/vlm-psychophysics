"""Bounded Gemma NVFP4 measurements and request-boundary pause/resume.

The qualified generation loop and byte proofs are derived from gemma_production;
no model implementation, input precision, or generation setting is rewritten.
A resumed worker gets a new output directory and only proven unattempted IDs.
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid
from . import gemma_speed as g
from . import gemma_production as p
from . import gemma_cpu_copy as cpu
from . import nvfp4_gemma as nv
from . import inference

SCHEMA = 'gemma-nvfp4-continuation-v1'
FROZEN = {'gemma_speed.py':p.BASE_SHA,
          'gemma_production.py':'6a5b98a31caaee821b0994b70c33204de3592f58b4118eed7c4e7a4665103e86',
          'nvfp4_gemma.py':'0e00eea196188ab23d5b4893322b32dae01ea181eb0246d3661bf8c5095e16d9',
          'gemma_cpu_copy.py':nv.CPU_COPY_SHA,
          'inference.py':'2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3'}
VARIANTS = ('official-native', 'qualified-o0', 'native-flashinfer', 'compiled-o0')


def sources():
    root=Path(__file__).parent
    g.require(all(g.file_sha256(root/n)==h for n,h in FROZEN.items()),'Qualified source changed')
    return {**FROZEN,Path(__file__).name:g.file_sha256(__file__)}


def settings(runtime, variant):
    g.require(variant in VARIANTS,'Unknown candidate')
    out=nv.engine_settings(runtime['checkpoint'],runtime['processor_checkpoint'],'full-gpu')
    # Disabling dummy autotune is explicit; no scientific input is shortened.
    out['kernel_config']={'enable_flashinfer_autotune':False}
    if variant=='native-flashinfer': out['attention_config']={'backend':'FLASHINFER'}
    if variant=='compiled-o0': out['enforce_eager']=False
    return out


def runtime_check(runtime):
    g.require(Path(runtime['checkpoint']).name==nv.REVISION,'NVFP4 revision differs')
    g.require(Path(runtime['processor_checkpoint']).name==g.REVISION,'Original processor revision differs')
    versions={k:importlib.metadata.version(k) for k in runtime['expected_packages']}
    g.require(versions==runtime['expected_packages'],'Installed SDK versions differ')
    allowed=set(nv.COMPILER_ENV)|{'NUMPY_MADVISE_HUGEPAGE','CLASSIC_PRECISION_KV_COMPAT','PYTHONPATH',
        'VLLM_ALLOW_INSECURE_SERIALIZATION','OMP_NUM_THREADS','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
        'CUDA_VISIBLE_DEVICES','VLLM_WEIGHT_OFFLOADING_DISABLE_UVA','VLLM_WORKER_MULTIPROC_METHOD',
        'PYTHONDONTWRITEBYTECODE','TOKENIZERS_PARALLELISM','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'}
    g.require(set(runtime['environment'])<=allowed,'Unexpected environment field')
    for k,v in runtime['environment'].items(): g.require(os.environ.get(k)==v,'Environment differs: '+k)
    g.require(runtime['environment'].get('VLLM_WORKER_MULTIPROC_METHOD')=='spawn','Fresh spawned engine required')
    return versions


class Boundary:
    """Request is drained once yielded. Signals never interrupt its generation."""
    def __init__(self,profile):
        self.profile=profile; self.requested=False; self.reason=None
    def check(self):
        path=Path(self.profile['pause_file'])
        if path.exists():
            value=g.read(path)
            g.require(value=={'worker_token':self.profile['worker_token'],'action':'PAUSE'},'Foreign pause request')
            self.requested=True; self.reason='operator'
        deadline=self.profile.get('local_monotonic_deadline')
        if deadline is not None and time.monotonic()>=deadline:
            self.requested=True; self.reason='search_deadline'
        return self.requested
    def signal(self,*_): self.requested=True; self.reason='operator_signal'


class AssignedItems:
    def __init__(self,profile,catalog,refs,boundary):
        self.profile=profile; self.catalog=catalog; self.refs=refs; self.boundary=boundary
        self.ids=[]; self.by_id={}; self.done=None
    def __len__(self): return self.profile['maximum_model_calls']
    def __iter__(self):
        for sid in self.profile['assigned_ids']:
            if self.boundary.check(): break
            self.refs.expected(sid,len(self.ids)<2)
            self.ids.append(sid); self.by_id[sid]=self.profile['assignment_sha256']
            yield self.catalog['items'][sid]
        self.done={'event':'PAUSED' if self.boundary.requested else 'DONE','sample_ids':list(self.ids)}


class Worker(p.Worker):
    def __init__(self,profile,catalog,reference20,reference24,admission):
        super().__init__(profile,catalog,reference20,reference24,admission)
        self.refs.control=reference24 if profile['mode']=='24' else None
        if profile.get('resume_deep_reference'):
            self.refs.control=g.read(profile['resume_deep_reference']['path'])
            # Only the new first two have fresh deep references. Ordinary IDs
            # continue using the immutable catalog templates.
            original_verify=self.refs.verify
            def verify(receipt,sid,deep):
                if deep:original_verify(receipt,sid,deep)
                else:g.verify_input(receipt,self.refs.catalog[sid],False)
            self.refs.verify=verify
        self.boundary=Boundary(profile)
        self.inbox=AssignedItems(profile,catalog,self.refs,self.boundary)
    def reference_gate(self,*_):
        g.require(self.profile['catalog']['sha256']==p.CATALOG_SHA,'Catalog changed')
    def intent(self,item):
        # Atomic per-ID claim survives any crash before/after generate intent.
        path=Path(self.profile['claim_root'])/(item['sample_id']+'.json')
        p._write_new(path,{'sample_id':item['sample_id'],'worker_token':self.profile['worker_token'],
                          'profile_sha256':g.digest(self.profile),'output':str(self.output)})
        super().intent(item)
    def final(self,records):
        g.require(self.inbox.done and self.pending is None and
                  [x['sample_id'] for x in records]==self.inbox.ids==[x['sample_id'] for x in self.proofs],
                  'Durable call coverage incomplete')
        if not self.boundary.requested:
            g.require(self.inbox.ids==self.profile['assigned_ids'],'Assigned scope incomplete')
    def status(self): return 'PAUSED' if self.boundary.requested else 'COMPLETE'


def validate(profile,installed=False):
    g.require(profile['schema']==SCHEMA and profile['sources']==sources(),'Entry source changed')
    g.require(profile['engine_settings']==settings(profile['runtime'],profile['candidate']),'Settings changed')
    ids=profile['assigned_ids']
    g.require(ids and len(ids)==len(set(ids))==profile['maximum_model_calls'],'Duplicate or invalid assignment')
    g.require(profile['assignment_sha256']==g.digest(ids),'Assignment identity differs')
    g.require(profile['mode'] in ('24','production') and (len(ids)==24 if profile['mode']=='24' else len(ids)<=9360),'Budget differs')
    g.require(profile['first_deep_calls']==2 and profile['automatic_retry'] is False,'Qualification/retry changed')
    g.require(g.file_sha256(profile['catalog']['path'])==p.CATALOG_SHA,'Input catalog changed')
    g.require(profile['catalog']['sha256']==p.CATALOG_SHA,'Catalog descriptor differs')
    for descriptor in profile['references'].values():p.verified(descriptor)
    catalog=g.read(profile['catalog']['path'])
    g.require(set(ids)<=set(catalog['items']),'Unknown sample ID')
    if profile.get('resume_deep_reference'):
        p.verified(profile['resume_deep_reference']);ref=g.read(profile['resume_deep_reference']['path'])
        g.require(set(ref['rows'])==set(ids[:2]),'Resume references must cover exactly the next two IDs')
    else:ref=g.read(profile['references']['24']['path']) if profile['mode']=='24' else g.read(profile['references']['20']['path'])
    g.require(set(ids[:2])<=set(ref['rows']),'First two need existing deep references')
    if profile['mode']=='24':g.require(set(ids)==set(ref['rows']),'Fixed24 IDs differ')
    g.require(profile['variant']=='kernel','Original direct FP32 input bridge required')
    g.require(profile['cpu_copy']==(profile['candidate']!='official-native'),'CPU overlay differs')
    g.require(profile['cpu_execution_overlay']['source_sha256']==nv.CPU_COPY_SHA and
              profile['cpu_execution_overlay']['gc_cadence']==8,'Qualified CPU overlay changed')
    g.require(profile['candidate']!='native-flashinfer',
              'Native alternative backend awaits actual multimodal-mask observer compatibility; no blind GPU launch')
    if installed:
        runtime_check(profile['runtime'])
        g.require(g.runtime_sources()==profile['runtime_sources'],'Runtime source changed')
        g.require(inference.checkpoint_asset_identity(profile['runtime']['checkpoint'])==profile['checkpoint_assets'],'Assets changed')
        contract=nv.effective_checkpoint_contract(profile['runtime']['checkpoint'],profile['runtime']['processor_checkpoint'])
        g.require(contract==profile['checkpoint_contract'],'Pinned weight/input lineage changed')
    return catalog


def seal(args):
    anchor=g.read(args.anchor_identity)['profile']
    runtime=g.read(args.runtime) if args.runtime else deepcopy(anchor['runtime'])
    ids=g.read(args.ids)
    if isinstance(ids,dict):ids=ids['sample_ids']
    q={k:deepcopy(anchor[k]) for k in ['implementation','runtime_sources','checkpoint_assets','kernel','cpu_execution_overlay']}
    q.update(schema=SCHEMA,sources=sources(),candidate=args.candidate,variant='kernel',mode=args.mode,
        runtime=runtime,engine_settings=settings(runtime,args.candidate),plan_sha256=g.file_sha256(args.plan),
        catalog=p.descriptor(args.catalog),references={'20':p.descriptor(args.reference20),'24':p.descriptor(args.reference24)},
        source_roots=g.read(args.source_roots),assigned_ids=ids,assignment_sha256=g.digest(ids),
        worker=args.worker,worker_token=uuid.uuid4().hex,inbox=str(Path(args.output).parent),inbox_timeout_seconds=1,
        maximum_model_calls=len(ids),first_deep_calls=2,automatic_retry=False,panel={'items':[]},
        pause_file=str(Path(args.pause_file).resolve()),claim_root=str(Path(args.claim_root).resolve()),
        checkpoint_contract=deepcopy(anchor['nvfp4_overlay']['checkpoint_contract']),
        cpu_copy=args.candidate!='official-native',entry={'source_sha256':g.file_sha256(__file__)})
    if args.installed:
        q['checkpoint_contract']=nv.effective_checkpoint_contract(runtime['checkpoint'],runtime['processor_checkpoint'])
    if args.candidate in ('official-native','native-flashinfer'):
        q['kernel'].update(range_lookup='native',prefill_block_m=0)
    q['execution_signature']=signature(q)
    if args.resume_deep_reference:q['resume_deep_reference']=p.descriptor(args.resume_deep_reference)
    validate(q,installed=args.installed)
    p._write_new(args.output,q)
    return {'status':'SEALED','profile_sha256':g.digest(q),'calls':len(ids),'GPU_calls':0}


def signature(profile):
    return g.digest({'sources':profile['sources'],'settings':{k:v for k,v in profile['engine_settings'].items() if k not in ('model','tokenizer')},
        'kernel':profile['kernel'],'cpu_copy':profile['cpu_copy'],
        'cpu_execution_overlay':profile['cpu_execution_overlay'],'checkpoint_assets':profile['checkpoint_assets'],
        'processor_revision':g.REVISION,'nvfp4_revision':nv.REVISION,'runtime_sources':profile['runtime_sources']})


@contextmanager
def overlay(profile):
    original=(g.runtime_check,g.engine_settings,g._placement,inference.ProcessorSession)
    rt=profile['runtime']
    def processor(family,model,*a,**kw):
        g.require(family=='gemma' and str(model)==rt['checkpoint'],'Wrong processor request')
        return original[3](family,rt['processor_checkpoint'],*a,**kw)
    g.runtime_check=runtime_check
    g.engine_settings=lambda checkpoint:settings(rt,profile['candidate'])
    g._placement=lambda worker:nv._placement(worker,offload_gb=0)
    inference.ProcessorSession=processor
    try:yield
    finally:g.runtime_check,g.engine_settings,g._placement,inference.ProcessorSession=original


def run(args):
    g.require('torch' not in sys.modules and 'vllm' not in sys.modules,'Fresh process required')
    profile=g.read(args.profile);catalog=validate(profile,installed=True)
    g.require(profile['execution_signature']==signature(profile),'Signature differs')
    admission=g.read(args.admission)
    g.require(admission.get('profile_sha256')==g.digest(profile),'Admission bound to other profile')
    if profile['mode']=='production':
        gate=g.read(args.gate)
        actual=compare(gate['observation_paths'],cross_hardware=True)
        g.require(actual['status']=='MATCH' and actual==gate and
                  actual['execution_signature']==profile['execution_signature'],
                  'Same final entry three-card24 gate required')
    Path(profile['claim_root']).mkdir(parents=True,exist_ok=True)
    g.require(not any((Path(profile['claim_root'])/(sid+'.json')).exists() for sid in profile['assigned_ids']),
              'ID previously attempted; explicit reconciliation required')
    worker=Worker(profile,catalog,g.read(profile['references']['20']['path']),g.read(profile['references']['24']['path']),admission)
    if args.search_clock:
        clock_path=Path(args.search_clock)
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if not clock_path.exists():
            p._write_new(clock_path,{'boot_id':boot,'started_monotonic':time.monotonic(),'budget_seconds':3600})
        clock=g.read(clock_path)
        g.require(clock['boot_id']==boot and clock['budget_seconds']==3600,'Search clock changed or host rebooted')
        worker.boundary.profile=dict(profile,local_monotonic_deadline=clock['started_monotonic']+3600)
        g.require(not worker.boundary.check(),'One-hour search budget expired; no new model launch')
    previous={s:signal.signal(s,worker.boundary.signal) for s in (signal.SIGUSR1,signal.SIGINT,signal.SIGTERM)}
    args.reference=profile['references']['24' if profile['mode']=='24' else '20']['path']
    begin=time.monotonic();state=None;success=False
    try:
        with overlay(profile),ExitStack() as stack:
            if profile['cpu_copy']:state=stack.enter_context(cpu.installed_overlay(profile))
            runner,derived=worker.compile();runner.__globals__['ENTRY_SCHEMA']=SCHEMA
            if state:state.code,state.line=runner.__code__,derived['gc_line']
            result=runner(args)
        success=True
        return result
    finally:
        for s,h in previous.items():signal.signal(s,h)
        if worker.output:
            g.atomic_json(worker.output/'continuation.json',{'schema':SCHEMA,'status':worker.status() if success else 'UNCERTAIN_OR_FAILED',
                'profile_sha256':g.digest(profile),'execution_signature':signature(profile),'sources':sources(),
                'assigned_ids':profile['assigned_ids'],'completed_ids':[x['sample_id'] for x in worker.proofs],
                'uncertain_id':None if worker.pending is None else worker.pending['sample_id'],
                'unattempted_ids':[sid for sid in profile['assigned_ids'] if not (Path(profile['claim_root'])/(sid+'.json')).exists()],
                'worker_wall_seconds':time.monotonic()-begin,'pause_reason':worker.boundary.reason,
                'cpu_copy':profile['cpu_copy'],'gc_cadence':state.cadence if state else 0,'automatic_resume':False,
                'gc_final':state.final if state else None})


def reconcile(root):
    root=Path(root); identity=g.read(root/'identity.json');receipt=g.read(root/'continuation.json')
    g.require(receipt['status'] in ('COMPLETE','PAUSED'),'Unclean execution is not automatically resumable')
    g.require(receipt['profile_sha256']==identity['profile_sha256']==g.digest(identity['profile']),'Profile differs')
    exit=g.read(root/'exit.json')
    g.require(exit['status']=='COMPLETE' and exit['owned_engine_workers_exited'] and exit['lease_released_after_engine_exit'],'Owned engine/lease exit incomplete')
    ids=[]; attempts=set();raw_end=0;event_end=None
    for call in p._lines(root/'calls.jsonl') if (root/'calls.jsonl').exists() else []:
        g.require(call['raw_start']==raw_end and call['ordinal']==len(ids) and call['attempt_id'] not in attempts,
                  'Gap, duplicate or invalid proof ordinal')
        if event_end is None:event_end=call['prefix_end']
        g.require(call['event_start']==event_end,'Event proof gap')
        blobs={}
        for stem,name in [('raw','raw.jsonl'),('event','events.jsonl')]:
            b=p.byte_range(root/name,call[stem+'_start'],call[stem+'_end'])
            g.require(g.hashlib.sha256(b).hexdigest()==call[stem+'_sha256'],'Durable byte proof differs')
            blobs[stem]=[json.loads(line) for line in b.splitlines()]
        intents=[x for x in blobs['event'] if x.get('event')=='GENERATE_INTENT']
        passed=[x for x in blobs['event'] if x.get('event')=='TECHNICAL_PASS']
        g.require(len(blobs['raw'])==len(intents)==len(passed)==1,'Incomplete durable call')
        raw=blobs['raw'][0]
        g.require(raw['sample_id']==intents[0]['sample_id']==passed[0]['sample_id']==call['sample_id']
          and raw['entry_attempt_id']==intents[0]['attempt_id']==call['attempt_id']
          and raw['finish_reason']=='stop' and raw['raw_answer'] in ('LEFT','RIGHT')
          and raw['output_token_ids'] and passed[0]['status']=='PASS','Raw/intent/PASS binding differs')
        ids.append(call['sample_id'])
        attempts.add(call['attempt_id']);raw_end=call['raw_end'];event_end=call['event_end']
    g.require(raw_end==((root/'raw.jsonl').stat().st_size if (root/'raw.jsonl').exists() else 0),'Uncommitted raw tail')
    if event_end is not None:g.require(event_end==(root/'events.jsonl').stat().st_size,'Uncommitted event tail')
    g.require(ids==receipt['completed_ids'] and not receipt['uncertain_id'],'Calls/proofs incomplete')
    g.require(len(ids)==len(set(ids)) and not set(ids)&set(receipt['unattempted_ids']),'Resume duplicates completed ID')
    g.require(set(ids)|set(receipt['unattempted_ids'])==set(receipt['assigned_ids']),'Attempt gap')
    return {'status':'RESUMABLE' if receipt['status']=='PAUSED' else 'COMPLETE','completed_ids':ids,
            'remaining_ids':receipt['unattempted_ids'],'explicit_new_worker_required':True,'automatic_resume':False}


def resume_reference(args):
    previous=Path(args.previous);state=reconcile(previous)
    g.require(state['status']=='RESUMABLE' and state['remaining_ids'],'No unattempted resume IDs')
    q=g.read(previous/'identity.json')['profile'];catalog=p.verified(q['catalog'])
    refs=p.CatalogReferences(catalog,g.read(q['references']['20']['path']));rows={}
    with inference.ProcessorSession('gemma',q['runtime']['processor_checkpoint']) as session:
        for sid in state['remaining_ids'][:2]:
            item=catalog['items'][sid]
            frames,payload,text,receipt=g.prepare(session,item,refs.roots_for(sid,q['source_roots']),True,True)
            g.verify_input(receipt,refs[sid],False);rows[sid]=receipt
            del frames,payload
    value={'schema':'gemma-resume-two-deep-cpu-v1','prior_output':str(previous),'model_calls':0,
           'rows':rows,'catalog_sha256':p.CATALOG_SHA,'source_sha256':g.file_sha256(__file__)}
    p._write_new(args.output,value);return {'status':'CPU_REFERENCE_COMPLETE','new_GPU_calls':0,'IDs':list(rows)}


def compare(paths,cross_hardware=False):
    identities=[];runs=[];rows=[];masks=[]
    for path in paths:
        root=Path(path);r=reconcile(root);g.require(r['status']=='COMPLETE' and len(r['completed_ids'])==24,'Complete24 required')
        i=g.read(root/'identity.json');q=i['profile'];g.require(q['mode']=='24','Qualification mode required')
        g.require(q['execution_signature']==signature(q),'Signature differs')
        identities.append(i);rows.append(p._lines(root/'raw.jsonl'))
        passed=[x for x in p._lines(root/'events.jsonl') if x.get('event')=='TECHNICAL_PASS']
        g.require([x['sample_id'] for x in passed]==r['completed_ids'],'Mask observation coverage differs')
        masks.append([x['native_masks'] for x in passed])
        runs.append({'path':str(root),'identity_sha256':g.file_sha256(root/'identity.json'),
                     'calls_sha256':g.file_sha256(root/'calls.jsonl'),'exit_sha256':g.file_sha256(root/'exit.json')})
    g.require(len(rows)>=2,'At least two observations required')
    controls=['actual_input_token_ids'];output=['output_token_ids','raw_answer','finish_reason','stop_reason']
    differences=[]
    for n,other in enumerate(rows[1:],1):
        g.require([x['sample_id'] for x in other]==[x['sample_id'] for x in rows[0]],'Panel order differs')
        g.require(masks[n]==masks[0],'Actual native multimodal mask intervals differ')
        for a,b in zip(rows[0],other):
            g.require(all(a[k]==b[k] for k in controls),'Input tokens differ')
            for k in ['source_stream_sha256','prompt_sha256','position_sha256','HF_control_tensors']:
                g.require(a['input_receipt'][k]==b['input_receipt'][k],'Effective input differs: '+k)
            if any(a[k]!=b[k] for k in output):differences.append({'sample_id':a['sample_id'],'observation':n,
                'reference':{k:a[k] for k in output},'candidate':{k:b[k] for k in output}})
    cards=sorted({(i['host'],i['lease']['device']) for i in identities});hosts=sorted({i['host'] for i in identities})
    signatures={i['profile']['execution_signature'] for i in identities}
    if cross_hardware:g.require(len(cards)==3 and len(hosts)==2 and len(signatures)==1,'Three cards/two hosts/same final configuration required')
    return {'status':'DIFFERENT' if differences else 'MATCH','differences':differences,'count':24,
            'execution_signature':next(iter(signatures)) if len(signatures)==1 else None,
            'cards':[list(x) for x in cards],'hosts':hosts,'observation_paths':[str(Path(x)) for x in paths],'runs':runs}


def main():
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    s=sub.add_parser('seal')
    for key in ['anchor-identity','plan','catalog','reference20','reference24','source-roots','ids','worker','pause-file','claim-root','output']:s.add_argument('--'+key,required=True)
    s.add_argument('--runtime');s.add_argument('--resume-deep-reference');s.add_argument('--installed',action='store_true');s.add_argument('--mode',choices=['24','production'],default='24');s.add_argument('--candidate',choices=VARIANTS,required=True)
    s=sub.add_parser('run')
    for key in ['profile','admission','output']:s.add_argument('--'+key,required=True)
    s.add_argument('--gate');s.add_argument('--search-clock')
    s=sub.add_parser('pause');s.add_argument('--profile',required=True)
    s=sub.add_parser('reconcile');s.add_argument('root')
    s=sub.add_parser('compare');s.add_argument('roots',nargs='+');s.add_argument('--cross-hardware',action='store_true');s.add_argument('--output')
    s=sub.add_parser('resume-reference');s.add_argument('--previous',required=True);s.add_argument('--output',required=True)
    args=ap.parse_args()
    if args.command=='seal':result=seal(args)
    elif args.command=='run':result=run(args)
    elif args.command=='reconcile':result=reconcile(args.root)
    elif args.command=='resume-reference':result=resume_reference(args)
    elif args.command=='compare':
        result=compare(args.roots,args.cross_hardware)
        if args.output:p._write_new(args.output,result)
    else:
        q=g.read(args.profile);value={'worker_token':q['worker_token'],'action':'PAUSE'};path=Path(q['pause_file'])
        if path.exists():g.require(g.read(path)==value,'Pause request belongs to other worker')
        else:p.publish_message(path,value)
        result={'status':'PAUSE_REQUESTED','at':'after current durable request, before next intent','automatic_resume':False}
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
