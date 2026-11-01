"""Project accepted natural revisions into the existing public reader contract.

Immutable renderer catalogs/QA remain untouched. The projection distinguishes
the revision recipe digest from the existing encoded-video identity digest.
No decoding, generation, inference, remote access or upload occurs here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from io import StringIO
import json
from pathlib import Path
import shutil

from .dataset_identity import IdentityRegistry, identity_digest, relative_path
from .natural_source_color import digest, read, sha, write

NAME = 'natural-source-revision-20261101-v1'
REPOSITORIES = {t:'vlm-psychophysics-natural-'+t.lower() for t in ['T1','T2','T3']}
COUNTS = {'T1':3840,'T2':3840,'T3':2736}


def lines(path):
    return Path(path).read_bytes().splitlines(keepends=True)


def jsonline(value):
    return (json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n').encode()


def project_row(row, record):
    assert row['sample_id'] == record['sample_id'] and row['video_sha256'] == record['video_sha256']
    assert row['parent_sample_id'] == record['parent_sample_id']
    assert row['full_identity_sha256'] == digest(record['identity_payload'])
    child = deepcopy(row)
    child['revision_identity_payload_sha256'] = child['full_identity_sha256']
    child['full_identity_sha256'] = identity_digest(child['domain'],child['task'],child['video_sha256'])
    # The published standalone reader indexes these two fields directly. Self
    # compatibility values introduce neither a historical parent nor extra media.
    child['legacy_sample_id'] = child['sample_id']
    child['legacy_path'] = child['path']
    assert not child.get('alias_paths') and not child.get('legacy_public_path')
    assert child['sample_id'] != child['parent_sample_id'] and child['parent_is_alias'] is False
    IdentityRegistry.from_rows([child])
    return child


def verify_return(revision):
    revision = Path(revision); manifest = read(revision/'return-manifest.json')
    assert len(manifest) == 235
    for name, expected in manifest.items():
        relative_path(name); path = revision/name
        assert path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'], name
    audit = read(revision/'source-audit.json'); records = read(revision/'source-records.json')
    plan = read(revision/'plan.json')
    assert digest({k:v for k,v in plan.items() if k!='plan_sha256'}) == plan['plan_sha256'] == audit['plan_sha256']
    assert audit['status'] == 'COMPLETE_200_SOURCE_REVISION_ALL180_PIXEL_AND_TIME_PASS'
    assert len(records) == len({r['sample_id'] for r in records}) == len({r['parent_sample_id'] for r in records}) == 200
    assert Counter(r['task'] for r in records) == {'T2':132,'T3':68}
    assert audit['total_natural_videos'] == 10416 and audit['decoded_frames'] == 36000
    assert audit['model_calls'] == 0 and audit['parents_are_aliases'] is False
    assert Counter(r['reason'] for r in records) == {'ICC':184,'local-balance':16}
    for r in records:
        assert r == read(revision/'qa'/(r['sample_id']+'.json'))
        assert r['frames_byte_equal'] == r['timestamps_checked'] == 180
        assert r['decoded_stream_sha256'] != r['parent_decoded_stream_sha256']
        if r['task']=='T2':
            assert r['outside_target_identical'] and abs(r['float64_delta_Y']-r['parameters']['delta_Y']) < 1e-12
            assert abs(r['decoded_delta_Y']-r['parameters']['delta_Y']) < min(.002,.02*r['parameters']['delta_Y'])
        else:
            assert r['outside_objects_identical'] and abs(r['actual_angle_deg']-r['parameters']['angle']) <= .02
    for task in COUNTS:
        assert sha(revision/'catalogs'/(task+'.jsonl')) == audit['final_catalog_sha256'][task]
    return audit,plan,records,manifest


def csvline(row, fields, newline='\r\n'):
    out = StringIO(newline=''); writer = csv.DictWriter(out,fieldnames=fields,lineterminator=newline)
    writer.writerow({k:json.dumps(row[k],sort_keys=True,separators=(',',':')) if isinstance(row.get(k),(dict,list))
                     else row.get(k,'') for k in fields})
    return out.getvalue().encode()


def projected_aliases(original, output, parents, projected):
    raw = lines(original); header = next(csv.reader([raw[0].decode()]))
    newline = '\r\n' if raw[0].endswith(b'\r\n') else '\n'
    out = [raw[0]]; unchanged = 0
    for line in raw[1:]:
        row = dict(zip(header,next(csv.reader([line.decode()]))))
        if row['sample_id'] not in parents:
            out.append(line); unchanged += 1
        else:
            child = projected[parents[row['sample_id']]]
            values = {**child,'legacy_sample_id':child['sample_id'],'legacy_path':child['path'],
                      'legacy_public_path':'','alias_paths':[]}
            out.append(csvline(values,header,newline))
    Path(output).write_bytes(b''.join(out))
    return unchanged


def source_content(row, record, qa_relative, qa_sha, plan, source_audit_sha):
    p = record['parameters']
    measured = dict(onset_frame=p.get('onset_frame',p.get('onset')),
                    offset_frame=p.get('offset_frame',p.get('onset',0)+30))
    if row['task']=='T2':
        measured.update(delta_Y=record['decoded_delta_Y'],float64_delta_Y=record['float64_delta_Y'],
                        alpha_area_px2=p['alpha_area_px2'],side=row['truth'])
    else:
        measured.update(angle_deg=record['actual_angle_deg'],vertical_side=row['truth'],
            upper_end_direction=row['physical_upper_end_direction'],
            angle_method='PCA of intended patch alpha; all180 decoded RGB frames match intended states')
    return dict(schema='source-content-revision-v1',status='PASS',domain='natural',task=row['task'],
        sample_id=row['sample_id'],legacy_sample_id=row['sample_id'],path=row['path'],
        video_sha256=row['video_sha256'],source_sha256=row['video_sha256'],
        parent_sample_id=row['parent_sample_id'],parent_video_sha256=row['parent_video_sha256'],parent_is_alias=False,
        source_record_sha256=row['source_record_sha256'],decoded_stream_sha256=row['decoded_stream_sha256'],
        frames=180,fps=60,width=768,height=512,bytes=record['bytes'],measured=measured,
        comparison={'GT':True,'condition':True,'filename':True,'source_aliases':True},
        method='Retained encoded-source QA: every RGB frame equals intended states and every PTS is checked; no consumer projection re-encoding',
        full_frame_QA=dict(path=qa_relative,sha256=qa_sha,frames_byte_equal=180,timestamps_checked=180,
            maximum_accepted_PTS_error_s=.001,decoded_stream_sha256=row['decoded_stream_sha256'],
            plan_sha256=plan['plan_sha256'],implementation_sha256=plan['implementation_sha256'],
            original_source_audit_sha256=source_audit_sha),
        tolerances={'delta_absolute':.002,'delta_relative':.02} if row['task']=='T2' else {'axis_deg':.01,'angle_separation_deg':.02},
        failed_reason=None)


def validate_identity(rows, aliases, parents):
    registry = IdentityRegistry.from_rows(rows); csv_registry = IdentityRegistry.from_csv(aliases)
    for row in rows:
        for key in [row['sample_id'],row['path']]:
            assert registry.resolve(key,domain='natural').canonical_id == row['sample_id']
            assert csv_registry.resolve(key,domain='natural').canonical_id == row['sample_id']
    for parent in parents:
        for reader in [registry,csv_registry]:
            try: reader.resolve(parent,domain='natural')
            except KeyError: pass
            else: raise AssertionError('A historical parent resolves in the new source view: '+parent)
    return len(registry.bindings)


def stage(baseline, revision, output, color_assets):
    baseline,revision,output,color_assets = map(Path,[baseline,revision,output,color_assets])
    audit,plan,records,returned = verify_return(revision)
    recby = {r['sample_id']:r for r in records}; parent_to_child = {r['parent_sample_id']:r['sample_id'] for r in records}
    color = read(revision/'color/source-catalog-delta.json'); deltas = {r['source_id']:r for r in color['changes']}
    assert set(deltas) == {'photo_obj_025','photo_obj_033','photo_obj_048'}
    output.mkdir(parents=True,exist_ok=True); datasets = []; allrows = []
    for task in COUNTS:
        old = baseline/REPOSITORIES[task]; oldlines = lines(old/'catalog.jsonl')
        oldby = {json.loads(line)['sample_id']:json.loads(line) for line in oldlines}
        assert sha(old/'catalog.jsonl') == plan['current_catalog_sha256'][task]
        if task=='T1':
            assert b''.join(oldlines) == (revision/'catalogs/T1.jsonl').read_bytes()
            allrows.extend(json.loads(line) for line in oldlines); continue
        dest = output/REPOSITORIES[task]; dest.mkdir(parents=True,exist_ok=True)
        for forbidden in ['catalog.jsonl','metadata.jsonl','aliases.csv','release.json']:
            assert not (dest/forbidden).exists(), 'Choose a new staging view; do not overwrite a prior projection'
        task_records = {sid:r for sid,r in recby.items() if r['task']==task}
        parents = {r['parent_sample_id']:sid for sid,r in task_records.items()}
        raw_catalog = lines(revision/'catalogs'/(task+'.jsonl')); projected = {}; catalog_lines = []
        for line in raw_catalog:
            row = json.loads(line)
            if row['sample_id'] in task_records:
                row = project_row(row,task_records[row['sample_id']]); projected[row['sample_id']] = row
                line = jsonline(row)
            catalog_lines.append(line)
        assert len(projected) == len(task_records)
        # Every line not belonging to the exact change set remains byte-for-byte.
        unchanged_old = [line for line in oldlines if json.loads(line)['sample_id'] not in parents]
        unchanged_new = [line for line in catalog_lines if json.loads(line)['sample_id'] not in projected]
        assert unchanged_old == unchanged_new
        (dest/'catalog.jsonl').write_bytes(b''.join(catalog_lines))
        metadata_lines = []
        for line in lines(old/'metadata.jsonl'):
            sid = json.loads(line)['sample_id']
            metadata_lines.append(jsonline(projected[parents[sid]]) if sid in parents else line)
        (dest/'metadata.jsonl').write_bytes(b''.join(metadata_lines))
        assert [json.loads(line) for line in metadata_lines] == [json.loads(line) for line in catalog_lines]
        unchanged_aliases = projected_aliases(old/'aliases.csv',dest/'aliases.csv',parents,projected)
        rows = [json.loads(line) for line in catalog_lines]; allrows.extend(rows)
        assert len(rows) == COUNTS[task] == validate_identity(rows,dest/'aliases.csv',parents)
        assert len(IdentityRegistry.from_jsonl(dest/'metadata.jsonl').bindings) == COUNTS[task]
        qa_lines = []
        for line in lines(old/'validation/source-content.jsonl'):
            oldqa = json.loads(line); sid = oldqa['sample_id']
            if sid not in parents: qa_lines.append(line); continue
            child = projected[parents[sid]]; record = task_records[child['sample_id']]
            qa_relative = 'validation/source-revision-20261101/qa/'+child['sample_id']+'.json'
            qa_dest = dest/qa_relative; qa_dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(revision/'qa'/(child['sample_id']+'.json'),qa_dest)
            qa_lines.append(jsonline(source_content(child,record,qa_relative,sha(qa_dest),plan,sha(revision/'source-audit.json'))))
        assert len(qa_lines) == COUNTS[task]
        assert {json.loads(line)['sample_id'] for line in qa_lines} == {r['sample_id'] for r in rows}
        (dest/'validation/source-content.jsonl').write_bytes(b''.join(qa_lines))
        assets = read(old/'source-assets.json'); changed_assets = []
        for asset in assets['assets']:
            delta = deltas.get(asset['source_id'])
            if delta is None: continue
            assert asset['sha256'] == delta['parent_sha256']
            public_path = 'assets/source-revision-20261101/'+asset['source_id']+'.png'
            source = color_assets/(asset['source_id']+'.png'); assert sha(source) == delta['sha256']
            target = dest/public_path; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
            asset.update(sha256=delta['sha256'],parent_source_sha256=delta['parent_sha256'],
                path=public_path,source_revision=NAME,color_management=delta['color_management'],
                alpha_byte_equal=True,native_downloaded_RGB_unchanged=False)
            changed_assets.append(asset['source_id'])
        assert set(changed_assets) == (set(deltas) if task=='T2' else {'photo_obj_025'})
        write(dest/'source-assets.json',assets)
        with (old/'metadata-demo.csv').open(newline='') as f: demo_fields = csv.DictReader(f).fieldnames
        demo = {}
        for row in projected.values(): demo.setdefault((task_records[row['sample_id']]['reason'],row['truth']),row)
        with (dest/'metadata-demo.csv').open('wb') as f:
            f.write(csvline(dict(zip(demo_fields,demo_fields)),demo_fields))
            for row in demo.values(): f.write(csvline(row,demo_fields))
        lineage = []
        for sid,row in projected.items():
            lineage.append({k:row[k] for k in ['sample_id','parent_sample_id','parent_video_sha256','parent_decoded_stream_sha256',
                'video_sha256','decoded_stream_sha256','path','full_identity_sha256','revision_identity_payload_sha256','parent_is_alias']})
        write(dest/'lineage.json',lineage)
        release = dict(schema='source-release-v2',dataset_revision=NAME,source_videos=COUNTS[task],
            changed_source_videos=len(projected),unchanged_source_videos=len(unchanged_old),
            catalog_sha256=sha(dest/'catalog.jsonl'),metadata_sha256=sha(dest/'metadata.jsonl'),
            aliases_sha256=sha(dest/'aliases.csv'),source_content_qa_sha256=sha(dest/'validation/source-content.jsonl'),
            source_assets_sha256=sha(dest/'source-assets.json'),lineage_sha256=sha(dest/'lineage.json'),
            parent_release_sha256=sha(old/'release.json'),parent_catalog_sha256=sha(old/'catalog.jsonl'),
            immutable_render_catalog_sha256=sha(revision/'catalogs'/(task+'.jsonl')),
            immutable_source_audit_sha256=sha(revision/'source-audit.json'),
            accepted_render_plan_sha256=plan['plan_sha256'],source_media_reencoded=False,
            newly_generated_source_videos=len(projected),consumer_projection_reencodes=0,
            parent_IDs_are_aliases=False,identity_contract='sample-identity-v1; immutable encoded video SHA',
            revision_identity_contract='revision_identity_payload_sha256 retains the renderer recipe digest')
        write(dest/'release.json',release)
        files = [{'path':str(p.relative_to(dest)),'local_path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='README.md']
        videos = [dict(sample_id=r['sample_id'],source_root_relative=r['video_root_relative'],
            path=projected[r['sample_id']]['path'],bytes=r['bytes'],sha256=r['video_sha256'],
            parent_sample_id=r['parent_sample_id'],superseded_current_path=oldby[r['parent_sample_id']]['path'])
            for r in task_records.values()]
        datasets.append(dict(task=task,repo_id='DrQY/'+REPOSITORIES[task],staging=str(dest.resolve()),
            files=files,videos=videos,source_videos=COUNTS[task],new_videos=len(videos),
            unchanged_catalog_lines_byte_equal=len(unchanged_old),unchanged_alias_lines_byte_equal=unchanged_aliases))
    assert len(allrows) == len(IdentityRegistry.from_rows(allrows).bindings) == 10416
    full = IdentityRegistry.from_rows(allrows)
    for parent in parent_to_child:
        try: full.resolve(parent,domain='natural')
        except KeyError: pass
        else: raise AssertionError('Parent unexpectedly resolves in final natural registry')
    manifest = dict(schema='natural-source-publication-projection-v1',status='PASS_CONSUMER_PROJECTION_NO_UPLOAD',
        revision=NAME,model_calls=0,total_natural_videos=10416,new_source_videos=200,unchanged_catalog_lines_byte_equal=10216,
        returned_metadata_files_SHA_verified=len(returned),source_audit_sha256=sha(revision/'source-audit.json'),
        source_return_manifest_sha256=sha(revision/'return-manifest.json'),
        projection_implementation_sha256=sha(__file__),datasets=datasets,
        T1_unchanged_catalog_sha256=sha(baseline/REPOSITORIES['T1']/'catalog.jsonl'),
        video_presence='200 accepted videos remain on execution host; this projection verifies their returned SHA-bound metadata, not a new local decode')
    write(output/'publication-manifest.json',manifest)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True); p.add_argument('--revision',required=True)
    p.add_argument('--output',required=True); p.add_argument('--color-assets',required=True)
    a = p.parse_args(); result = stage(a.baseline,a.revision,a.output,a.color_assets)
    print(json.dumps({k:v for k,v in result.items() if k!='datasets'},indent=2))


if __name__=='__main__': main()
