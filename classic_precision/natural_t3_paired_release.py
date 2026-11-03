"""Close the new T3 source and project a composite natural execution catalog.

No model calls. Existing T1/T2 source identities and execution items are retained.
Every revised T3 item is bound to its actual lossless video and classic counterpart.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path
import copy,json
from .natural_t13_stimuli import read,write,sha,digest
from .natural_t3_classic_paired import NAME,paths
from .natural_t3_paired_allocation import audit_rows,audit_features
from .dataset_content_audit import parse_name,alias
from .dataset_identity import IdentityRegistry,identity_digest
from . import natural_x_runner as runner

REVISION=NAME+'-v1'

def ref(p):return dict(path=str(p),sha256=sha(p))
def jsonl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(''.join(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n' for x in rows))

def check_one(job):
    root,row=job;p,d,r=paths(root);q=d/'qa'/(row['sample_id']+'.json');v=d/'videos'/row['path'];rec=read(q)
    assert rec['status']=='PASS' and rec['allocation_row_sha256']==digest(row)
    assert rec['sample_id']==row['sample_id'] and rec['classic_sample_id']==row['classic_sample_id']
    assert rec['frames_byte_equal']==rec['timestamps_checked']==180 and rec['bytes']==v.stat().st_size and rec['video_sha256']==sha(v)
    assert rec['onset_frame']==row['onset_frame'] and rec['offset_frame']==row['offset_frame']==row['onset_frame']+30
    assert rec['physical_tilt_sign_measured']==row['physical_tilt_sign']
    assert rec['unchanged_background_outside_objects'] and abs(rec['actual_angle_deg']-row['angle_deg'])<=.02
    assert rec['implementation_sha256']==sha(Path(__file__).with_name('natural_t3_classic_paired.py'))
    for patch in rec['patches']:
        assert patch['source_sha256']==row['object_source_sha256'] and abs(patch['axis_error_deg'])<=.01 and abs(patch['alpha_area']/3800-1)<=.001
    name=parse_name(row['path'],'natural','T3');physical={-1:'right',0:'horizontal',1:'left'}[row['physical_tilt_sign']]
    assert name==dict(side=row['truth'],direction=physical,angle=row['angle_deg'],background=alias(row['background_id']),object=alias(row['object_id']),trial=row['onset_index']+1)
    parameters=dict(angle=row['angle_deg'],physical_tilt_sign=row['physical_tilt_sign'],tilt_sign=1 if row['angle_deg']==90 else -row['physical_tilt_sign'],
        onset=row['onset_frame'],onset_index=row['onset_index'],offset_frame=row['offset_frame'],
        background_source=row['background_source'],object_source=row['object_source'],group=row['group'])
    output=dict(dataset_revision=REVISION,domain='natural',task='T3',sample_id=row['sample_id'],legacy_sample_id=row['sample_id'],
        path=row['path'],legacy_path=row['path'],video_sha256=rec['video_sha256'],decoded_stream_sha256=rec['RGB_stream_sha256'],
        full_identity_sha256=identity_digest('natural','T3',rec['video_sha256']),source_record_sha256=digest(row),
        truth=row['truth'],prompt=row['prompt'],parameters=parameters,
        condition_key=['T3',f'{row["angle_deg"]:g}',row['physical_tilt_sign'],row['onset_frame'],row['truth']],
        background_id=row['background_id'],object_id=row['object_id'],physical_upper_end_direction=physical,
        fps=60,frames=180,width=768,height=512,primary_T123_scope=True,
        classic_pair=dict(sample_id=row['classic_sample_id'],video_sha256=row['classic_video_sha256'],
            relation='One-to-one physical angle/sign/onset/duration/side configuration; natural visual assets substituted',is_alias=False),
        source_lineage=dict(preceding_version='natural-source-revision-20261101-v1',parent_is_alias=False,
            relation='New outcome-independent allocation; prior natural videos and answers remain historical',
            background_asset_sha256=row['background_source_sha256'],background_prepared_sha256=row['background_prepared_sha256'],object_asset_sha256=row['object_source_sha256']),
        generation=dict(allocation_row_sha256=digest(row),renderer_sha256=rec['implementation_sha256'],
            full_frame_qa=dict(path=f'qa/{row["sample_id"]}.json',sha256=sha(q),frames_byte_equal=180,timestamps_checked=180),
            physical_sign_independent_covariance_check=True,actual_angle_deg=rec['actual_angle_deg']))
    return output,dict(path=str(v.relative_to(Path(root))),sha256=rec['video_sha256'],bytes=rec['bytes']),dict(path=str(q.relative_to(Path(root))),sha256=sha(q),bytes=q.stat().st_size)

def finalize(root,workers):
    root=Path(root);p,d,r=paths(root);allocation=read(p/'allocation.json');assets=read(p/'assets.json');prep=read(p/'asset-preparation.json')
    assert allocation['implementation_sha256']==sha(Path(__file__).with_name('natural_t3_paired_allocation.py'))
    assert allocation['assets_sha256']==sha(p/'assets.json') and allocation['asset_preparation_sha256']==sha(p/'asset-preparation.json')
    assert allocation['classic_manifest_sha256']==sha(p/'classic-configurations.json')
    assert read(r/'render-finished.json')['status']=='PASS' and read(r/'render-finished.json')['count']==2736
    assert read(r/'render-finished.json')['allocation_sha256']==sha(p/'allocation.json')
    design=audit_rows(allocation['rows'],read(p/'classic-configurations.json')['rows']);design['measured_covariate_balance']=audit_features(allocation['rows'],assets,prep)
    # Recheck source dependencies once at closure, not redundantly per video.
    dependencies=[]
    for a in assets['objects']:dependencies.append((root/a['root_relative_path'],a['sha256']))
    for a in assets['backgrounds']:dependencies.append((root/a['prepared_path'],a['prepared_sha256']))
    for q in sorted((d/'patches').glob('*.json')):
        rec=read(q);dependencies.append((q.with_suffix('.npy'),rec['patch_sha256']))
    with ThreadPoolExecutor(workers) as pool:
        for actual,(_,expected) in zip(pool.map(lambda x:sha(x[0]),dependencies),dependencies):assert actual==expected
        checked=list(pool.map(check_one,[(root,x) for x in allocation['rows']]))
    rows=[x[0] for x in checked];IdentityRegistry.from_rows(rows)
    assert len(set(x['decoded_stream_sha256'] for x in rows))==2736
    jsonl(p/'catalogs/T3.jsonl',rows)
    write(p/'source-manifest.json',dict(schema='natural-T3-paired-source-files-v1',files=[z for x in checked for z in x[1:]],
        video_count=2736,video_bytes=sum(x[1]['bytes'] for x in checked),allocation=ref(p/'allocation.json'),
        catalog=ref(p/'catalogs/T3.jsonl'),implementation=ref(__file__)))
    summary={**design,'schema':'natural-T3-classic-paired-source-acceptance-v1','status':'COMPLETE_T3_CLASSIC_PAIRED_SOURCE_PASS',
        'source_version':REVISION,'decoded_frames':2736*180,'PTS_checked':2736*180,'all_videos_lossless_RGB8_byte_equal':True,
        'all_videos_canonical_filename_GT_physical_sign_onset_match':True,'originals_preserved':True,'model_calls':0,
        'video_bytes':sum(x[1]['bytes'] for x in checked),'source_dependencies_SHA_checked':len(dependencies),
        'allocation':ref(p/'allocation.json'),'source_manifest':ref(p/'source-manifest.json'),'catalog':ref(p/'catalogs/T3.jsonl'),
        'render_receipt':ref(r/'render-finished.json'),'renderer':ref(Path(__file__).with_name('natural_t3_classic_paired.py')),
        'allocator':ref(Path(__file__).with_name('natural_t3_paired_allocation.py')),'release_auditor':ref(__file__),
        'prior_T3_videos_reused':0,'new_T3_videos':2736,'T1_T2_source_changes':0}
    write(p/'source-acceptance.json',summary);print(json.dumps({k:summary[k] for k in ['status','natural_videos','missing','duplicates','decoded_frames','video_bytes','source_dependencies_SHA_checked']}),flush=True)

def project(root):
    root=Path(root);p,d,r=paths(root);oldp=root/'plans/natural-x-t123-20261101'
    acceptance=read(p/'source-acceptance.json');assert acceptance['status']=='COMPLETE_T3_CLASSIC_PAIRED_SOURCE_PASS'
    old_approval=read(oldp/'source-approval-fourier.json');old_catalog=read(oldp/'catalog-fourier.json')
    registry=runner.registry(old_approval['registry']);new_rows=[json.loads(x) for x in (p/'catalogs/T3.jsonl').read_text().splitlines()]
    identity_rows=[]
    for binding in registry.rows():
        if binding['task']=='T3':continue
        identity_rows.append({**binding,'dataset_revision':REVISION})
    identity_rows.extend(new_rows);new_registry=IdentityRegistry.from_rows(identity_rows);jsonl(p/'source-registry.jsonl',identity_rows)
    for sid,item in old_catalog['items'].items():
        if item['task']!='T3':continue
        try:new_registry.resolve(sid,domain='natural',dataset_revision=REVISION)
        except KeyError:pass
        else:raise AssertionError('Retired T3 ID must not alias a new input')
    new_items={sid:copy.deepcopy(x) for sid,x in old_catalog['items'].items() if x['task']!='T3'}
    assert len(new_items)==7680
    for row in new_rows:
        sid=row['sample_id'];assert sid not in old_catalog['items']
        new_items[sid]=dict(sample_id=sid,task='T3',row=dict(sample_id=sid,task='T3',truth=row['truth'],prompt=row['prompt'],choices=['LEFT','RIGHT'],accuracy_eligible=True,source_parameters=row['parameters']),
            video_contract={k:row[k] for k in ['fps','frames','height','width']},content=[dict(type='video',path=str(d/'videos'/row['path']),sha256=row['video_sha256']),dict(type='text',text=row['prompt'])],publication_row_sha256=digest(row))
    assert Counter(x['task'] for x in new_items.values())==runner.COUNTS
    unchanged=[sid for sid,x in new_items.items() if sid in old_catalog['items'] and x==old_catalog['items'][sid]];assert len(unchanged)==7680
    identities=[runner.source_identity(x,new_registry,REVISION) for x in new_items.values()]
    catalog=dict(schema='natural-x-publication-execution-catalog-v1',items=new_items,publication_refs={'T3':ref(p/'catalogs/T3.jsonl')},
        preceding_catalog=ref(oldp/'catalog-fourier.json'),unchanged_T1_T2_items=7680,model_calls=0)
    write(p/'catalog-fourier.json',catalog)
    # This is the source handoff, not a model launch approval or a fabricated reuse receipt.
    receipt=dict(schema='natural-T3-paired-consumer-projection-v1',status='PASS_EXACT_SOURCE_IDENTITY_CPU',source_version=REVISION,
        counts=runner.COUNTS,registry=ref(p/'source-registry.jsonl'),catalog=ref(p/'catalog-fourier.json'),source_acceptance=ref(p/'source-acceptance.json'),
        source_identity_sha256=digest(sorted(identities,key=lambda x:x['sample_id'])),unchanged_T1_T2_items=7680,new_T3_items=2736,
        retired_T3_aliases=0,current_model_launches=0,original_catalog_unchanged=True)
    write(p/'consumer-projection.json',receipt);print(json.dumps({k:receipt[k] for k in ['status','unchanged_T1_T2_items','new_T3_items','retired_T3_aliases','source_identity_sha256']}),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['finalize','project']);parser.add_argument('--root',required=True,type=Path);parser.add_argument('--workers',type=int,default=16);a=parser.parse_args()
    if a.action=='finalize':finalize(a.root,a.workers)
    else:project(a.root)
if __name__=='__main__':main()
