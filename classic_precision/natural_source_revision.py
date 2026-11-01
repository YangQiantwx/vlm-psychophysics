"""Prepare, render and accept exactly 200 versioned natural source revisions.

CPU only. No answers, inference, upload or mutation of original sources. Current
published catalogs are authoritative, including both existing spoon overlays.
Run preparation locally if desired; batch render and full decode belong on the
execution workstation, with its existing CPU environment and SSD paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from . import natural_source_color as color
from . import natural_t2_rebalance as balance
from . import natural_t2_stimuli as t2
from . import natural_t13_stimuli as t3

NAME = 'natural-source-revision-20261101-v1'
COUNTS = {'T1':3840, 'T2':3840, 'T3':2736}
read, write, sha, digest = t2.read, t2.write, t2.sha, t2.digest


def catalog_path(root, task):
    return Path(root)/color.STAGING/('vlm-psychophysics-natural-'+task.lower())/'catalog.jsonl'


def catalogs(root):
    result = {t:[json.loads(s) for s in catalog_path(root,t).read_text().splitlines()] for t in COUNTS}
    assert {t:len(rows) for t,rows in result.items()} == COUNTS
    assert len({r['sample_id'] for rows in result.values() for r in rows}) == 10416
    # These are the current corrected spoon sources, not the historical source-index.
    for task, count in [('T2',40), ('T3',72)]:
        spoon = [r for r in result[task] if r.get('object_id') == 'photo_obj_021']
        assert len(spoon) == count and all(r.get('parent_sample_id') for r in spoon)
    return result


def check_plan(root, output):
    plan = read(Path(output)/'plan.json')
    saved = plan.pop('plan_sha256'); assert digest(plan) == saved; plan['plan_sha256'] = saved
    assert plan['revision'] == NAME and plan['new_videos'] == 200
    assert Counter(r['task'] for r in plan['changes']) == {'T2':132, 'T3':68}
    assert len({r['sample_id'] for r in plan['changes']}) == len({r['parent_sample_id'] for r in plan['changes']}) == 200
    assert all(r['sample_id'] != r['parent_sample_id'] and not r['parent_is_alias'] for r in plan['changes'])
    for task, expected in plan['current_catalog_sha256'].items():
        assert sha(catalog_path(root,task)) == expected, 'Current source changed since preparation'
    for relative, expected in plan['implementation_sha256'].items():
        assert sha(Path(__file__).parent/relative) == expected, ('Implementation changed',relative)
    for relative, expected in plan['prepared_file_sha256'].items():
        assert sha(Path(output)/relative) == expected, ('Prepared file changed',relative)
    for relative, expected in plan['input_file_sha256'].items():
        assert sha(Path(root)/relative) == expected, ('Original prepared input changed',relative)
    return plan


def prepare_patches(root, output, color_dir, changes):
    root, output, color_dir = Path(root), Path(output), Path(color_dir)
    source = np.load(color_dir/'prepared/T3/photo_obj_025-source.npy')
    olddir = root/'datasets'/t3.NAME/'patches'; out = output/'patches'; out.mkdir(parents=True,exist_ok=True)
    measurements = []
    angles = sorted({0.} | {r['parent']['parameters']['tilt_sign']*r['parent']['parameters']['angle']
                            for r in changes if r['task']=='T3'})
    from .natural_t13_review import cv_axis
    for angle in angles:
        tag = f'photo_obj_025_{angle:+g}'
        oldmeta = read(olddir/(tag+'.json')); old = np.load(olddir/(tag+'.npy'), mmap_mode='r')
        assert sha(olddir/(tag+'.npy')) == oldmeta['patch_sha256']
        # The earlier frozen renderer used fixed 8x8 sampling and predates the
        # explicit attempts field. Exact historical alpha equality below is the
        # acceptance condition for either receipt format.
        samples = oldmeta.get('integration_attempts',[{'samples_per_axis':8}])[-1]['samples_per_axis']
        z = t3.affine(source, np.array(oldmeta['matrix']), samples=samples)
        assert np.array_equal(z[...,3],old[...,3]), ('T3 transformed alpha changed',tag)
        measured = cv_axis(z[...,3]); error = t3.angle_diff(measured,90+angle)
        assert abs(error) <= .01 and abs(z[...,3].sum()/3800-1) <= .001
        np.save(out/(tag+'.npy'),z)
        record = dict(tag=tag, theta=angle, matrix=oldmeta['matrix'], samples_per_axis=samples,
                      alpha_byte_equal=True, axis_error_deg=error, measured_axis_deg=measured,
                      parent_patch_sha256=oldmeta['patch_sha256'], patch_sha256=sha(out/(tag+'.npy')),
                      alpha_area=float(z[...,3].sum()))
        write(out/(tag+'.json'),record); measurements.append(record)
    return measurements


def build_plan(root, output, color_dir, balance_plan):
    root, output, color_dir = Path(root), Path(output), Path(color_dir)
    assert not (output/'plan.json').exists(), 'Use a new output directory; never replace a frozen plan'
    current = catalogs(root); lookup = {r['sample_id']:r for rows in current.values() for r in rows}
    creport = read(color_dir/'color-repair.json'); candidate = read(balance_plan)
    assert creport['parent_catalog_sha256'] == sha(root/color.CATALOG)
    cby = {r['asset_id']:r for r in creport['converted_assets']}; assert set(cby) == color.ASSETS
    changes = []
    for task, allowlist in creport['exact_current_video_allowlist'].items():
        actual = {r['sample_id'] for r in current[task] if r.get('object_id') in color.ASSETS}
        assert actual == {r['sample_id'] for r in allowlist}, 'ICC allowlist is not the current published identity set'
        for row in allowlist:
            parent = lookup[row['sample_id']]
            assert all(parent[k] == row[k] for k in ['sample_id','path','object_id','video_sha256'])
            asset = cby[parent['object_id']]
            identity = dict(revision=NAME, reason='embedded-source-ICC-to-sRGB', task=task,
                parent_sample_id=parent['sample_id'], parent_video_sha256=parent['video_sha256'],
                parent_decoded_stream_sha256=parent['decoded_stream_sha256'],
                source_asset_sha256=asset['source_sha256'], source_ICC_sha256=asset['ICC_sha256'],
                rendering_intent=color.INTENT, flags=color.FLAGS,
                prepared_sha256=asset['T2_preparation']['prepared_sha256'] if task=='T2' else asset['T3_preparation']['source_array_sha256'])
            child = ('nt2_' if task=='T2' else 'n')+digest(identity)[:24]
            changes.append(dict(task=task, reason='ICC', sample_id=child, parent_sample_id=parent['sample_id'],
                parent_video_sha256=parent['video_sha256'], parent=parent, object_id=parent['object_id'],
                source_asset_sha256=asset['source_sha256'], identity_payload=identity, parent_is_alias=False,
                proposed_path=str(Path(parent['path']).with_name(Path(parent['path']).stem+'_rev'+child[-8:]+'.mp4'))))
    assert Counter(r['task'] for r in changes) == {'T2':116,'T3':68}
    icc_parents = {r['parent_sample_id'] for r in changes}
    assert len(candidate['exact_changed_videos']) == 16
    for replacement in candidate['exact_changed_videos']:
        parent = lookup[replacement['parent_sample_id']]
        assert replacement['parent_sample_id'] not in icc_parents
        assert replacement['parent_video_sha256'] == parent['video_sha256']
        assert parent['object_id'] == replacement['old_object_id']
        assert replacement['object_id'] not in color.ASSETS
        changes.append(dict(task='T2', reason='local-balance', sample_id=replacement['sample_id'],
            parent_sample_id=parent['sample_id'], parent_video_sha256=parent['video_sha256'], parent=parent,
            object_id=replacement['object_id'], source_asset_sha256=replacement['new_source_asset_sha256'],
            identity_payload=replacement['identity_payload'], parent_is_alias=False,
            proposed_path=str(Path(replacement['proposed_path']).with_name(
                Path(replacement['proposed_path']).stem+'_rev'+replacement['sample_id'][-8:]+'.mp4'))))
    overrides2 = {aid:r['T2_preparation'] for aid,r in cby.items()}
    overrides3 = {aid:r['T3_preparation'] for aid,r in cby.items() if r['T3_preparation'] is not None}
    final_gates = balance.evaluate_final_gates(root,candidate,overrides2,overrides3)
    assert final_gates['status'] == 'PASS_DECLARED_PREPARATION_FEATURE_GATES', final_gates
    output.mkdir(parents=True,exist_ok=True)
    geometry = prepare_patches(root,output,color_dir,changes)
    write(output/'balance-candidate.json',candidate)
    bound = {}
    for p in sorted(color_dir.rglob('*')):
        if p.is_file(): bound[os.path.relpath(p,output)] = sha(p)
    for p in sorted((output/'patches').glob('*')):
        bound[str(p.relative_to(output))] = sha(p)
    bound['balance-candidate.json'] = sha(output/'balance-candidate.json')
    inputs = set()
    for c in changes:
        bg = root/'datasets'/(t2.NAME if c['task']=='T2' else t3.NAME)/(
            'prepared' if c['task']=='T2' else 'backgrounds')/(
            c['parent']['background_id']+('.npz' if c['task']=='T2' else '.npy'))
        meta = read(bg.with_suffix('.json'))
        assert sha(bg) == meta['prepared_sha256']
        inputs.update([bg,bg.with_suffix('.json')])
        if c['reason']=='local-balance':
            obj = root/'datasets'/t2.NAME/'prepared-vertical-v2'/(c['object_id']+'.npz')
            assert sha(obj) == read(obj.with_suffix('.json'))['prepared_sha256']
            inputs.update([obj,obj.with_suffix('.json')])
    plan = dict(schema='natural-finite-source-revision-v1', revision=NAME,
        status='FROZEN_200_SOURCE_CHANGES_PENDING_VIDEO_GENERATION', model_calls=0,
        current_catalog_sha256={t:sha(catalog_path(root,t)) for t in COUNTS},
        total_natural_videos=10416, unchanged_videos=10216, new_videos=200,
        counts=COUNTS, color_dir=os.path.relpath(color_dir,output), changes=changes,
        color_repair_sha256=sha(color_dir/'color-repair.json'), balance_candidate_sha256=sha(balance_plan),
        final_feature_gates=final_gates, T3_geometry=geometry,
        prepared_file_sha256=bound,
        input_file_sha256={str(p.relative_to(root)):sha(p) for p in sorted(inputs)},
        implementation_sha256={Path(m.__file__).name:sha(m.__file__) for m in [color,balance,t2,t3]},
        source_retention='Every parent video, original source, raw answer and earlier metadata remains immutable; parent IDs are lineage, never aliases.')
    plan['implementation_sha256'][Path(__file__).name] = sha(__file__)
    plan['plan_sha256'] = digest(plan)
    write(output/'plan.json',plan)
    write(output/'lineage.json',[{k:r[k] for k in ['task','reason','sample_id','parent_sample_id','parent_video_sha256','object_id','source_asset_sha256','proposed_path','parent_is_alias']} for r in changes])
    return plan


def banks(root, output, plan, change):
    root, output = Path(root), Path(output); parent = change['parent']; p = parent['parameters']
    if change['task'] == 'T2':
        b = np.load(root/'datasets'/t2.NAME/'prepared'/(parent['background_id']+'.npz'))['rgb'].copy()
        obj = (output/plan['color_dir']/'prepared/T2'/(change['object_id']+'.npz') if change['reason']=='ICC'
               else root/'datasets'/t2.NAME/'prepared-vertical-v2'/(change['object_id']+'.npz'))
        with np.load(obj) as z: pm,a = z['pm'],z['alpha']
        meta = read(obj.with_suffix('.json')); assert sha(obj) == meta['prepared_sha256']
        assert meta['source_sha256'] == change['source_asset_sha256']
        mu = float((pm@t2.Y).sum()/a.sum()); gain = p['delta_Y']/(1-mu); assert 0 < gain < 1
        t2.stamp(b,pm,a,'LEFT'); t2.stamp(b,pm,a,'RIGHT'); e = b.copy()
        cx,cy = t2.CENTERS[0 if parent['truth']=='LEFT' else 1]
        e[cy-192:cy+192,cx-192:cx+192] += gain*(a[...,None]-pm)
        delta = float(((e-b)@t2.Y).sum()/a.sum()); assert abs(delta-p['delta_Y']) < 1e-12
        base,event = t2.rgb8(b),t2.rgb8(e)
        observed = float(((t2.linear(event/255)-t2.linear(base/255))@t2.Y).sum()/a.sum())
        assert abs(observed-p['delta_Y']) < min(.002,.02*p['delta_Y'])
        roi = np.zeros((512,768),bool); roi[cy-192:cy+192,cx-192:cx+192] = a>0
        assert np.any(base!=event) and np.array_equal(base[~roi],event[~roi])
        params = {**p, 'baseline_object_Y':mu, 'white_blend_k':gain, 'alpha_area_px2':float(a.sum())}
        return [base,event], [int(p['onset_frame']<=i<p['offset_frame']) for i in range(180)], dict(
            parameters=params, decoded_delta_Y=observed,float64_delta_Y=delta,outside_target_identical=True)
    b = np.load(root/'datasets'/t3.NAME/'backgrounds'/(parent['background_id']+'.npy'))
    theta = p['tilt_sign']*p['angle']
    z0 = np.load(output/'patches/photo_obj_025_+0.npy'); zt = np.load(output/'patches'/f'photo_obj_025_{theta:+g}.npy')
    e = b.copy(); left,right = (z0,zt) if parent['truth']=='LEFT' else (zt,z0)
    t3.stamp(e,left,224,304); t3.stamp(e,right,544,304)
    base,event = t3.rgb8(b),t3.rgb8(e)
    roi = np.zeros((512,768),bool); roi[144:464,64:384] = left[...,3]>0; roi[144:464,384:704] = right[...,3]>0
    assert np.array_equal(base[~roi],event[~roi])
    from .natural_t13_review import cv_axis
    measured = abs(t3.angle_diff(cv_axis(zt[...,3]),cv_axis(z0[...,3])))
    assert abs(measured-p['angle']) <= .02
    return [base,event], [int(p['onset']<=i<p['onset']+30) for i in range(180)], dict(
        parameters=p,actual_angle_deg=measured,outside_objects_identical=True)


def accept_video(path, bank, indices):
    import av
    h = hashlib.sha256(); n = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]; stream.codec_context.thread_count = 1
        assert (stream.width,stream.height) == (768,512) and stream.average_rate == 60
        for frame in container.decode(stream):
            assert n < 180 and abs(float(frame.pts*frame.time_base)-n/60) <= .001
            rgb = frame.to_ndarray(format='rgb24')
            assert np.array_equal(rgb,bank[indices[n]]), ('Decoded pixel mismatch',str(path),n)
            h.update(rgb.tobytes()); n += 1
    assert n == 180
    return dict(video_sha256=sha(path), decoded_stream_sha256=h.hexdigest(),
                frames_byte_equal=n,timestamps_checked=n,bytes=Path(path).stat().st_size)


def video_job(args):
    root, output, plan, change, render = args; output = Path(output)
    bank,indices,metric = banks(root,output,plan,change)
    path = output/'videos'/change['task']/(change['sample_id']+'.mp4'); path.parent.mkdir(parents=True,exist_ok=True)
    if render and not path.exists():
        if change['task']=='T2':
            temporary = path.with_suffix('.partial.mp4')
            p = change['parent']['parameters']; t2.encode(temporary,bank[0],bank[1],p['onset_frame'],p['offset_frame'])
            accept_video(temporary,bank,indices); temporary.replace(path)
        else: t3.encode_verify(path,bank,indices)
    accepted = accept_video(path,bank,indices)
    assert accepted['decoded_stream_sha256'] != change['parent']['decoded_stream_sha256'], 'New identity has unchanged visible pixels'
    record = {k:v for k,v in change.items() if k!='parent'}
    record.update(accepted); record.update(metric)
    record['video_root_relative'] = os.path.relpath(path,root)
    record['parent_decoded_stream_sha256'] = change['parent']['decoded_stream_sha256']
    write(output/'qa'/(change['sample_id']+'.json'),record)
    return record


def execute(root, output, jobs, render=False):
    assert sys.platform != 'darwin', 'Batch video generation and full-frame audit run on the execution workstation only'
    root, output = Path(root), Path(output); plan = check_plan(root,output)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        records = []
        for i,record in enumerate(pool.map(video_job,[(str(root),str(output),plan,c,render) for c in plan['changes']])):
            records.append(record)
            if (i+1)%10 == 0: print(json.dumps({'accepted_videos':i+1,'total':200,'render':render}),flush=True)
    assert len(records) == 200 and len({r['decoded_stream_sha256'] for r in records}) == 200
    lookup = {r['parent_sample_id']:r for r in records}; final = output/'catalogs'; final.mkdir(exist_ok=True)
    for task in COUNTS:
        original_lines = catalog_path(root,task).read_bytes().splitlines(keepends=True); new_lines = []
        for line in original_lines:
            parent = json.loads(line); r = lookup.get(parent['sample_id'])
            if r is None: new_lines.append(line); continue
            child = deepcopy(parent)
            # No historical ID or historical path is advertised as an alias of changed pixels.
            for key in ['legacy_sample_id','legacy_path','legacy_public_path','alias_paths']:
                child.pop(key,None)
            child.update(sample_id=r['sample_id'],parent_sample_id=parent['sample_id'],
                parent_video_sha256=parent['video_sha256'],parent_decoded_stream_sha256=parent['decoded_stream_sha256'],
                parent_is_alias=False, path=r['proposed_path'],object_id=r['object_id'],
                dataset_revision=NAME, video_sha256=r['video_sha256'],decoded_stream_sha256=r['decoded_stream_sha256'],
                source_record_sha256=digest(r),full_identity_sha256=digest(r['identity_payload']),
                parameters=r['parameters'],video_root_relative=r['video_root_relative'],
                source_asset_sha256=r['source_asset_sha256'])
            new_lines.append((json.dumps(child,sort_keys=True,ensure_ascii=False)+'\n').encode())
        (final/(task+'.jsonl')).write_bytes(b''.join(new_lines))
        assert len(new_lines) == COUNTS[task]
    summary = dict(status='COMPLETE_200_SOURCE_REVISION_ALL180_PIXEL_AND_TIME_PASS',revision=NAME,
        counts=COUNTS, new_videos=200,unchanged_videos=10216,total_natural_videos=10416,
        changed_counts=dict(Counter(r['task'] for r in records)),decoded_frames=36000,model_calls=0,
        plan_sha256=plan['plan_sha256'],final_feature_gates=plan['final_feature_gates'],
        final_catalog_sha256={t:sha(final/(t+'.jsonl')) for t in COUNTS},
        old_sources_and_answers_preserved=True,parents_are_aliases=False,
        T2_max_absolute_delta_error=max(abs(r['decoded_delta_Y']-r['parameters']['delta_Y']) for r in records if r['task']=='T2'),
        T3_max_angle_error_deg=max(abs(r['actual_angle_deg']-r['parameters']['angle']) for r in records if r['task']=='T3'))
    write(output/'source-records.json',records); write(output/'source-audit.json',summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','render','audit']); p.add_argument('--root',required=True)
    p.add_argument('--output',required=True); p.add_argument('--color-dir'); p.add_argument('--balance-plan')
    p.add_argument('--jobs',type=int,default=4); a = p.parse_args()
    if a.command=='prepare':
        assert a.color_dir and a.balance_plan
        r = build_plan(a.root,a.output,a.color_dir,a.balance_plan)
        print(json.dumps({k:r[k] for k in ['status','new_videos','unchanged_videos','total_natural_videos','plan_sha256']},indent=2))
    else: print(json.dumps(execute(a.root,a.output,a.jobs,render=a.command=='render'),indent=2))


if __name__ == '__main__':
    main()
