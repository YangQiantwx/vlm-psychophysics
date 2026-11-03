"""Strict LR source revision using the accepted renderer and immutable assets.

Full-video QA is reused only for byte-identical retained videos; changed videos
are rendered and checked against every RGB frame and PTS on the workstation.
"""
from __future__ import annotations
import argparse, copy, json, os, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
from .natural_t13_stimuli import read, write, sha, digest
from . import natural_t3_classic_paired as renderer
from . import natural_t3_paired_release as release
from . import natural_t3_paired_pixel_audit as pixel
from .natural_t3_lr_allocation import NAME, PRIOR, paths, audit, audit_features
from .dataset_identity import IdentityRegistry

REVISION=NAME+'-v1'

def configure():
    renderer.NAME=NAME
    release.REVISION=REVISION
    pixel.NAME=NAME

def setup(root):
    configure();root=Path(root);p,d,r=paths(root);old=root/'datasets'/PRIOR
    allocation=read(p/'allocation.json');rows={x['sample_id']:x for x in allocation['rows']}
    d.mkdir(parents=True,exist_ok=True);r.mkdir(parents=True,exist_ok=True)
    for folder in ['objects','patches']:
        target=d/folder
        if not target.exists():target.symlink_to(old/folder,target_is_directory=True)
        assert target.resolve()==(old/folder).resolve()
    dependencies={}
    for row in rows.values():
        dependencies[str(root/row['background_prepared_path'])]=row['background_prepared_sha256']
        for theta in [0.,row['render_theta_deg']]:
            q=d/'patches'/f'{row["object_id"]}_{theta:+g}.json';rec=read(q)
            assert rec['source_sha256']==row['object_source_sha256']
            dependencies[str(q.with_suffix('.npy'))]=rec['patch_sha256']
    with ThreadPoolExecutor(8) as pool:
        for actual,expected in zip(pool.map(sha,dependencies),dependencies.values()):assert actual==expected
    write(p/'source-dependencies-receipt.json',dict(status='PASS_IMMUTABLE_PREPARED_BG_AND_PATCH_SHA',
        files=len(dependencies),dependencies=dependencies,model_calls=0))
    prior={x['sample_id']:x for x in read(root/'plans'/PRIOR/'allocation.json')['rows']}
    receipt=[]
    for sid in allocation['reused_source_ids']:
        row=rows[sid];assert row==prior[sid]
        for rel in [Path('videos')/row['path'],Path('qa')/(sid+'.json')]:
            src=old/rel;dst=d/rel;dst.parent.mkdir(parents=True,exist_ok=True)
            if not dst.exists():os.link(src,dst)
            assert os.path.samefile(src,dst)
        q=read(d/'qa'/(sid+'.json'));assert sha(d/'videos'/row['path'])==q['video_sha256']
        receipt.append(dict(sample_id=sid,video_sha256=q['video_sha256'],qa_sha256=sha(d/'qa'/(sid+'.json')),
            original_video=str(old/'videos'/row['path']),reused_video=str(d/'videos'/row['path']),same_inode=True))
    write(p/'retained-source-receipt.json',dict(status='PASS_EXACT_IMMUTABLE_SOURCE_REUSE',count=len(receipt),
        records=receipt,allocation=release.ref(p/'allocation.json'),prior_allocation=release.ref(root/'plans'/PRIOR/'allocation.json')))
    print(json.dumps(dict(status='SETUP_PASS',same_inode_retained_videos=len(receipt),new_videos=2736-len(receipt))),flush=True)

def finalize(root,workers):
    configure();root=Path(root);p,d,r=paths(root);allocation=read(p/'allocation.json');assets=read(p/'assets.json');prep=read(p/'asset-preparation.json')
    assert allocation['implementation_sha256']==sha(Path(__file__).with_name('natural_t3_lr_allocation.py'))
    for field,filename in [('assets_sha256','assets.json'),('asset_preparation_sha256','asset-preparation.json'),('classic_manifest_sha256','classic-configurations.json')]:
        assert allocation[field]==sha(p/filename)
    finished=read(r/'render-finished.json');assert finished['status']=='PASS' and finished['count']==2736 and finished['allocation_sha256']==sha(p/'allocation.json')
    design=audit(allocation['rows'],read(p/'classic-configurations.json')['rows'])
    design['measured_covariate_balance']=audit_features(allocation['rows'],assets,prep)
    chosen=allocation['LR_pairs'];tables={}
    for field in ['object_id','background_id']:
        table={}
        for ident in sorted({x[field] for x in chosen}):
            sample=[x for x in chosen if x[field]==ident]
            angle=Counter(x['angle_deg'] for x in sample);onset=Counter(x['onset_index'] for x in sample)
            sign=Counter(x['physical_tilt_sign'] for x in sample)
            if field=='object_id':
                assert all(1<=angle[a]<=2 for a in renderer.ANGLES[:-1]) and angle[90]<=1
                assert all(1<=onset[t]<=2 for t in range(24))
                assert all(8<=sum(angle[a] for a in renderer.ANGLES[k*7:k*7+7])<=9 for k in range(4))
                assert abs(sign[1]-sign[-1])<=1
            else:
                assert len(angle)==len(sample) and max(angle.values())==1
                assert abs(sign[1]-sign[-1])<=3
            table[ident]=dict(LR_pairs=len(sample),videos=2*len(sample),angle_pairs=dict(angle),onset_pairs=dict(onset),physical_sign_pairs=dict(sign),LEFT=len(sample),RIGHT=len(sample))
        tables[field]=table
    tables['source_by_onset_pairs']={str(t):dict(Counter(x['group'] for x in chosen if x['onset_index']==t)) for t in range(24)}
    assert all(14<=n<=15 for cell in tables['source_by_onset_pairs'].values() for n in cell.values())
    write(p/'balance-tables.json',dict(schema='strict-LR-exact-marginal-counts-v1',unit='LR pairs unless explicitly videos',tables=tables,
        allocation=release.ref(p/'allocation.json'),model_outputs_used=False))
    retained=set(allocation['reused_source_ids']);replacements={x['new_sample_id']:x for x in allocation['replacements']}
    with ThreadPoolExecutor(workers) as pool:checked=list(pool.map(release.check_one,[(root,x) for x in allocation['rows']]))
    rows=[x[0] for x in checked]
    for row in rows:
        sid=row['sample_id'];row['source_lineage'].update(preceding_version=PRIOR+'-v1',
            relation='Exact unchanged source reused' if sid in retained else 'Opposite answer side with identical background, object, angle and event timing',
            retained_identity=sid in retained)
        if sid in replacements:row['source_lineage'].update(replacements[sid])
    IdentityRegistry.from_rows(rows);assert len({x['decoded_stream_sha256'] for x in rows})==2736
    release.jsonl(p/'catalogs/T3.jsonl',rows)
    write(p/'source-manifest.json',dict(schema='natural-T3-strict-LR-source-files-v1',files=[z for x in checked for z in x[1:]],
        video_count=2736,video_bytes=sum(x[1]['bytes'] for x in checked),allocation=release.ref(p/'allocation.json'),
        catalog=release.ref(p/'catalogs/T3.jsonl'),implementation=release.ref(__file__)))
    summary={**design,'schema':'natural-T3-strict-LR-source-acceptance-v1','status':'COMPLETE_T3_CLASSIC_PAIRED_SOURCE_PASS',
        'source_version':REVISION,'decoded_frames_verified':2736*180,'PTS_verified':2736*180,
        'newly_rendered_decoded_frames':(2736-len(retained))*180,'reused_prior_full_frame_QA':len(retained)*180,
        'all_videos_lossless_RGB8_byte_equal':True,'all_videos_canonical_filename_GT_physical_sign_onset_match':True,
        'originals_preserved':True,'model_calls':0,'video_bytes':sum(x[1]['bytes'] for x in checked),
        'allocation':release.ref(p/'allocation.json'),'source_manifest':release.ref(p/'source-manifest.json'),
        'catalog':release.ref(p/'catalogs/T3.jsonl'),'render_receipt':release.ref(r/'render-finished.json'),
        'renderer':release.ref(Path(renderer.__file__)),'allocator':release.ref(Path(__file__).with_name('natural_t3_lr_allocation.py')),
        'release_auditor':release.ref(__file__),'retained_source_receipt':release.ref(p/'retained-source-receipt.json'),
        'balance_tables':release.ref(p/'balance-tables.json'),'source_dependencies':release.ref(p/'source-dependencies-receipt.json'),
        'prior_T3_videos_reused':len(retained),'new_T3_videos':2736-len(retained),'T1_T2_source_changes':0}
    write(p/'source-acceptance.json',summary)
    print(json.dumps({k:summary[k] for k in ['status','natural_videos','strict_LR_pairs','prior_T3_videos_reused','new_T3_videos','video_bytes']}),flush=True)

def project(root):
    configure();root=Path(root);p,d,r=paths(root);release.project(root)
    # Preserve retained execution items in full, including actual original paths.
    prior=read(root/'plans'/PRIOR/'catalog-fourier.json');catalog=read(p/'catalog-fourier.json')
    allocation=read(p/'allocation.json')
    for sid in allocation['reused_source_ids']:
        old=prior['items'][sid];new=catalog['items'][sid]
        assert old['row']==new['row'] and old['video_contract']==new['video_contract']
        assert old['content'][0]['sha256']==new['content'][0]['sha256']
        catalog['items'][sid]=copy.deepcopy(old)
    catalog['unchanged_prior_paired_T3_items']=len(allocation['reused_source_ids'])
    write(p/'catalog-fourier.json',catalog);receipt=read(p/'consumer-projection.json')
    receipt.update(catalog=release.ref(p/'catalog-fourier.json'),new_T3_items=len(allocation['replacements']),
        retained_prior_T3_items=len(allocation['reused_source_ids']),strict_LR_pairs=1368)
    write(p/'consumer-projection.json',receipt)

def pixels(root,workers):
    configure();root=Path(root);p,d,r=paths(root);allocation=read(p/'allocation.json')
    retained=set(allocation['reused_source_ids']);prior_path=root/'runs/stimulus-preparation'/PRIOR/'decoded-keyframe-audit.jsonl'
    prior={x['sample_id']:x for x in map(json.loads,prior_path.read_text().splitlines())}
    catalog={x['sample_id']:x for x in map(json.loads,(p/'catalogs/T3.jsonl').read_text().splitlines())}
    records=[];byobject=defaultdict(list)
    for row in allocation['rows']:
        sid=row['sample_id']
        if sid in retained:
            old=prior[sid];assert old['status']=='PASS' and old['video_sha256']==catalog[sid]['video_sha256']
            assert old['classic_sample_id']==row['classic_sample_id'];records.append(copy.deepcopy(old))
        else:byobject[row['object_id']].append(row)
    started=time.monotonic()
    with ProcessPoolExecutor(workers) as pool:
        futures=[pool.submit(pixel.audit_object,(str(root),obj,rows,catalog)) for obj,rows in byobject.items()]
        for f in as_completed(futures):
            records.extend(f.result());progress=dict(status='RUNNING',verified_videos=len(records),retained_prior=len(retained),total=2736,seconds=time.monotonic()-started)
            write(r/'decoded-keyframe-progress.json',progress);print(json.dumps(progress),flush=True)
    assert len(records)==len({x['sample_id'] for x in records})==2736
    records.sort(key=lambda x:x['sample_id']);release.jsonl(r/'decoded-keyframe-audit.jsonl',records)
    small=[x['relative_angle_deg'] for x in records if x['expected_angle_deg']==.5]
    summary=dict(status='PASS_ALL2736_STRICT_LR_DECODED_KEYFRAME_ORIENTATION',videos=2736,
        newly_decoded_keyframes=2*(2736-len(retained)),retained_prior_keyframes=2*len(retained),
        exact_orientation_matches_per_side=1,candidate_orientations_per_object=58,
        label_independent_match='All58 orientations matched to decoded RGB before expected angle/sign/GT are checked.',
        prior_evidence=release.ref(prior_path),retained_source=release.ref(p/'retained-source-receipt.json'),
        relative_angle_error_max_deg=max(abs(x['relative_angle_deg']-x['expected_angle_deg']) for x in records),
        angle_0p5_relative_range_deg=[min(small),max(small)],pixels_dropped_to_identical_vertical_orientation=0,
        records=release.ref(r/'decoded-keyframe-audit.jsonl'),allocation=release.ref(p/'allocation.json'),
        catalog=release.ref(p/'catalogs/T3.jsonl'),implementation=release.ref(__file__),seconds=time.monotonic()-started,model_calls=0)
    write(r/'decoded-keyframe-summary.json',summary);print(json.dumps(summary),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['setup','render','finalize','project','pixels'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args();configure()
    if args.action in ['setup','project']:globals()[args.action](args.root)
    elif args.action=='render':renderer.render(args.root,args.workers)
    else:globals()[args.action](args.root,args.workers)

if __name__=='__main__':main()
