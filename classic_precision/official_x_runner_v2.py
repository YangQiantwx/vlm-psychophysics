"""Observer-only X admission for official device-side fused normalization.

The frozen generic runner remains unchanged. CUDA uint8 ingress is admissible
when it exactly matches the native CPU patch payload and the official normalizer
source is verified. Internal normalized activations remain explicitly unobserved.
"""
from __future__ import annotations

import argparse
import base64
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

from . import official_x_runner as base
from .refinement_worker import descriptor, read, require, write_new
from .storage import digest, file_sha256

_original_validate=base.validate_deep
CERTIFICATE_SCHEMA='official-x-input-reconciliation/v1'


def official_file(path):
    path=Path(path).resolve();dist=importlib.metadata.distribution('vllm')
    entry=next((x for x in dist.files if Path(dist.locate_file(x)).resolve()==path),None)
    require(entry is not None and entry.hash and entry.hash.mode=='sha256','No official wheel RECORD binding')
    value=base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
    require(value==entry.hash.value,'Official source differs from installed wheel RECORD')
    return {'path':str(path),'sha256':file_sha256(path),'wheel_record_sha256':entry.hash.value,
            'distribution':'vllm','version':dist.version}


def validate_ingress(p,item,raw,deep):
    c=base.video_contract(p,item);spec=p['vision_observation']
    expected_grid=[c['frames']//spec.get('temporal_patch_size',2),c['height']//spec.get('patch_size',16),c['width']//spec.get('patch_size',16)]
    cpu=[]
    for record in raw['native_processor']:
        for name,value in record['after_native_dtype'].items():
            if 'pixel' in name and isinstance(value,dict) and value.get('sha256'):
                cpu.append(value)
    require(cpu,'No durable native CPU processor tensor hashes')
    readers=raw['native_reader'];require(readers,'No actual native source reader')
    require(all(x['shape']==[180,c['height'],c['width'],3] and x['dtype']=='uint8'
        and x['metadata']['frames_indices']==list(range(180)) and len(x.get('per_frame_sha256',[]))==180
        for x in readers),'Actual source count/order/geometry differs')
    outer=[];normalizers=[]
    for worker in deep:
        require(not worker.get('runtime_error') and not worker.get('profiler_error'),'Observer runtime failure')
        runtime=worker['runtime'];require(runtime['actual_kv_capacity']>=p.get('minimum_kv_capacity',65536),'KV capacity differs')
        if p['model']=='qwen':
            qsa=runtime['native_qsa']
            require(qsa['function_source_exact'] and not qsa['custom_selector_modules']
                and not qsa['custom_selector_import_hooks'] and qsa['binary_mapped'],'Native QSA identity differs')
            require(any('persistent_topk' in x['key'] and x['count']>0 for x in worker['actual_topk_operator_calls']),
                    'Actual native TopK was not observed')
        norm=worker['normalizer']
        require(norm['class'].endswith(('.FusedMMInputNorm','.IdentityInputNorm')) and norm.get('source'),
                'No identified official device normalization module')
        source=official_file(norm['source']['path']);require(source['sha256']==norm['source']['sha256'],'Normalizer source changed')
        normalizers.append({**norm,'official_source_verification':source})
        for record in worker['outer']:
            tensor=record['tensor'];require(tensor and tensor['dtype']=='torch.uint8'
                and tensor['device'].startswith('cuda'),'Actual raw GPU ingress differs')
            require(any(x['shape']==tensor['shape'] and x['dtype']==tensor['dtype']
                and x['sha256']==tensor['sha256'] for x in cpu),'CPU processor and real GPU ingress bytes differ')
            outer.append(record)
    require(any(x['grid_thw']==[expected_grid] or x['grid_thw']==expected_grid for x in outer),'Actual all-frame GPU grid differs')
    require(any(x['tensor']['shape']==[expected_grid[0]*expected_grid[1]*expected_grid[2],spec.get('feature_dim',1536)]
                for x in outer),'Actual full-video GPU patch shape differs')
    base.validate_processor(p,item,raw['native_processor'])
    return {'status':'PASS_OFFICIAL_RAW_GPU_INGRESS','actual_GPU_ingress':outer,'native_CPU_processor':cpu,
        'source_reader_all180':True,'source_frame_indices':list(range(180)),
        'CPU_to_GPU_patch_bytes_equal':True,'official_normalizers':normalizers,
        'internal_normalized_tensor_observed':any(x.get('normalized') for x in deep),
        'internal_observation_boundary':'Native internal normalized tensor was not separately observed; no dtype is invented. Admission uses complete raw CUDA ingress, exact CPU patch bytes, native grid and official normalizer source identity.'}


def validate_deep(p,item,deep):
    if any(worker.get('normalized') for worker in deep):
        return _original_validate(p,item,deep)
    require(p['vision_observation'].get('allow_official_fused_normalization') is True,
            'No normalized observation and no explicit official ingress admission')
    raw=base.records(Path(p['output'])/'raw.jsonl')[-1]
    proof=validate_ingress(p,item,raw,deep)
    base._append(Path(p['output'])/'input-admissions.jsonl',{'sample_id':raw['sample_id'],
        'raw_digest':digest(raw),'observations_digest':digest(deep),'proof':proof})


base.validate_deep=validate_deep


def seal(draft,output):
    value=read(draft);value['observer_entry']=descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w',suffix='.json',delete=False) as f:
        json.dump(value,f);path=f.name
    try: return base.seal(path,output)
    finally: Path(path).unlink()


def run(profile):
    p=read(profile);require(file_sha256(__file__)==p['observer_entry']['sha256'],'Sealed observer entry changed')
    return base.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    native=worker.subprocess.Popen
    def popen(argv,**kwargs):
        return native([x.replace('classic_precision.refinement_worker',__spec__.name) for x in argv],**kwargs)
    worker.subprocess.Popen=popen
    try: return worker.supervise(profile)
    finally: worker.subprocess.Popen=native


def admit_existing(profile_path,output_path):
    p=read(profile_path);out=Path(p['output']);rows=base.records(out/'raw.jsonl');deep_rows=base.records(out/'deep-observations.jsonl')
    require(len(rows)==len(deep_rows)==1,'Special reconciliation is limited to the one preserved first request')
    raw=rows[0];deep=deep_rows[0];sid=raw['sample_id'];catalog=base.verified(p['source_catalog'])
    require(raw['profile_sha256']==digest(p) and deep['raw_digest']==digest(raw) and deep['sample_id']==sid,'Raw/profile/deep binding differs')
    claim=read(Path(p['claim_root'])/(sid+'.json'))
    require(claim['profile_sha256']==digest(p) and claim['sample_id']==sid and claim['attempt']==1,'Claim binding differs')
    require(file_sha256(p['implementation']['path'])==p['implementation']['sha256'],'Frozen original runner changed')
    wait=read(out.with_suffix('.wait.json'));exit_record=read(out/'exit.json')
    require(wait['returncode']==1 and wait['real_wait_observed'] and wait['owned_engines_exited']
        and wait['original_physical_lease_released'] and exit_record['owned_engine_workers_exited']
        and exit_record['shutdown_error'] is None,'Original process/lease did not close')
    require(read(out/'failure.json')['error']=='No actual post-normalization vision input recorded','Different failure requires independent reconciliation')
    require(raw['finish_reason']=='stop' and raw['raw_answer'].strip() in base.choices(p,catalog['items'][sid]),'Scientific completion invalid')
    for media in raw['source']['media']:
        require(file_sha256(media['path'])==media['sha256'],'Scientific source bytes changed')
    proof=validate_ingress(p,catalog['items'][sid],raw,deep['observations'])
    lock=p['physical_lock'];same_inode=False;free=False
    with Path(lock['path']).open('r+b') as f:
        s=os.fstat(f.fileno());same_inode=(s.st_dev,s.st_ino)==(lock['st_dev'],lock['st_ino'])
        if same_inode:
            try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);free=True
            except BlockingIOError: pass
    require(same_inode,'Physical lock identity changed')
    certificate={'schema':CERTIFICATE_SCHEMA,'status':'ACCEPTED_EXISTING_RAW_INPUT',
        'sample_id':sid,'profile':descriptor(profile_path),'profile_digest':digest(p),'raw_digest':digest(raw),
        'bindings':{name:descriptor(path) for name,path in {'raw':out/'raw.jsonl','deep':out/'deep-observations.jsonl',
            'events':out/'events.jsonl','claim':Path(p['claim_root'])/(sid+'.json'),'wait':out.with_suffix('.wait.json'),
            'exit':out/'exit.json','failure':out/'failure.json','engine_owner':out/'engine-owner.json'}.items()},
        'original_wait':wait,'original_exit':exit_record,'same_physical_lock_inode':same_inode,'physical_lock_free_at_reconciliation':free,
        'input_proof':proof,'observer_entry':descriptor(__file__),
        'original_TECHNICAL_PASS_absent_preserved':True,'original_UNCERTAIN_and_failure_preserved':True,
        'new_scientific_calls':0,'request_reissued':False,
        'scope':'Only this durable raw is input-qualified by explicit observer-contract reconciliation. No cold repeatability or full-engine COMPLETE claim.'}
    write_new(output_path,certificate);return {'status':certificate['status'],'certificate':descriptor(output_path),'sample_id':sid,'model_calls':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['seal','run','supervise','queue','export','admit-existing'])
    parser.add_argument('--draft');parser.add_argument('--output');parser.add_argument('--profile');parser.add_argument('--plan');a=parser.parse_args()
    if a.command=='seal': result=seal(a.draft,a.output)
    elif a.command=='run': result=run(a.profile)
    elif a.command=='supervise': result=supervise(a.profile)
    elif a.command=='export': result=base.export(a.profile,a.output)
    elif a.command=='admit-existing': result=admit_existing(a.profile,a.output)
    else:
        plan=read(a.plan);require(plan['status']=='SEALED' and plan['automatic_retry'] is False,'Wrong finite queue')
        for profile in plan['profiles']:
            value=supervise(profile);print(json.dumps(value),flush=True)
            require(value['returncode']==0 and value['terminal_status']=='COMPLETE'
                and value['real_wait_observed'] and value['owned_engines_exited']
                and value['original_physical_lease_released'],'Predecessor did not close COMPLETE; stop queue')
        result={'status':'COMPLETE','profiles':plan['profiles']}
    print(json.dumps(result))


if __name__=='__main__': main()
