"""Prepare a finite T2 allocation revision; never read answers or render media.

The candidate exchanges one complete photographic/generated object semantic pair
between the two prespecified delta cells. Source files and original manifests are
read only. Any new scientific video must use its proposed new ID and retain the
original current video as a parent, never as an alias.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = 'natural-t2-qwen-x-20261029'
REVISION = 'natural-t2-local-balance-20261101-v1'
TARGETS = (.08, .18)
FORBIDDEN_PHOTO_OBJECTS = frozenset({'photo_obj_025', 'photo_obj_033', 'photo_obj_048'})
GATE = .5
TARGET_MARGIN = .45
BNAMES = [f'{s}_background_{lr}_{stat}' for s in ['photo', 'gpt']
          for lr in ['left', 'right'] for stat in ['mean_Y', 'std_Y']]
ONAMES = [f'{s}_object_{stat}' for s in ['photo', 'gpt'] for stat in ['mean_Y', 'alpha_area']]
SNAMES = [f'{s}_{stat}' for s in ['photo', 'gpt']
          for stat in ['log_axis_eigenvalue_ratio', 'shape_cos2theta', 'shape_sin2theta']]
FEATURE_NAMES = BNAMES + ONAMES + SNAMES


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load_inputs(root):
    root = Path(root)
    paths = {
        'allocation': root/'plans'/NAME/'allocation.json',
        'criteria': root/'plans'/NAME/'allocation-criteria.json',
        'preparation': root/'plans'/NAME/'asset-preparation.json',
        'asset_catalog': root/'assets/library/extensions/t2-natural-20261029/catalog-at-preparation.json',
        'axes': root/'datasets'/NAME/'review/axis-algorithm-review.json',
        'spoon': root/'runs/stimulus-preparation/natural-spoon-replacement-20261030/balance-audit.json',
        'current_catalog': root/'plans/dataset-naming-migration/public-staging/vlm-psychophysics-natural-t2/catalog.jsonl',
    }
    data = {k: read(p) for k, p in paths.items() if k != 'current_catalog'}
    raw_lines = paths['current_catalog'].read_bytes().splitlines(keepends=True)
    current = [json.loads(line) for line in raw_lines if line.strip()]
    assert len(current) == 3840 and len({r['sample_id'] for r in current}) == 3840
    assert data['allocation']['criteria_sha256'] == sha(paths['criteria'])
    assert data['allocation']['asset_preparation_sha256'] == sha(paths['preparation'])
    assert data['axes']['catalog_sha256'] == sha(paths['asset_catalog'])
    assert data['axes']['current_preparation_sha256'] == sha(paths['preparation'])
    prep = {r['asset_id']: r for r in data['preparation']['assets']}
    prep['photo_obj_021'] = data['spoon']['T2_asset_preparation']
    axes = {r['asset_id']: r for r in data['axes']['assets']}
    spoon = data['spoon']['T3_asset_preparation']
    axes['photo_obj_021'] = {'source_sha256': spoon['source_sha256'],
                           'eigenvalue_ratio': spoon['ratio'],
                           'pca_axis_degrees_mod180': spoon['source_axis_deg']}
    lookup = {(r['pair_id'], r['id'].split('_')[0]): r['id']
              for r in data['asset_catalog']['objects']}
    for aid, axis in axes.items():
        assert axis['source_sha256'] == prep[aid]['source_sha256']
    blocks = data['allocation']['blocks']
    deltas = tuple(data['criteria']['deltas'])
    assert len(blocks) == 480 and len(deltas) == 20
    assert [b['block'] for b in blocks] == list(range(480))
    assert all(b['delta_Y'] == deltas[i//24] for i, b in enumerate(blocks))

    def object_features(pair):
        values = sum([prep[lookup[pair, src]]['features'] for src in ['photo', 'gpt']], [])
        for src in ['photo', 'gpt']:
            aid = lookup[pair, src]; a = axes[aid]
            ratio = a['eigenvalue_ratio']
            angle = np.deg2rad(a['pca_axis_degrees_mod180'] + prep[aid]['rotation_degrees'])
            anisotropy = (ratio-1)/(ratio+1)
            values.extend([float(np.log(ratio)), float(anisotropy*np.cos(2*angle)),
                           float(anisotropy*np.sin(2*angle))])
        return values

    features = np.array([sum([prep[f'{src}_{b["background_pair"]}']['features']
                             for src in ['photo', 'gpt']], []) + object_features(b['object_pair'])
                         for b in blocks])
    assert features.shape == (480, 18) and np.isfinite(features).all()
    # Bind each historical assignment to its current source, including the spoon
    # replacement IDs. Never infer current source identity from the old ID alone.
    def key(row):
        p = row['parameters']
        return (p['delta_Y'], p['group'], p['onset_frame'], row['background_id'], row['object_id'], row['truth'])
    current_by_key = {key(r): r for r in current}
    assert len(current_by_key) == 3840
    source_by_block = {}
    for b in blocks:
        rows = []
        for case in b['cases']:
            k = (case['delta_Y'], case['group'], case['onset_frame'], case['background_id'], case['object_id'], case['truth'])
            row = current_by_key[k]
            assert abs(row['parameters']['baseline_object_Y']-prep[row['object_id']]['features'][0]) < 1e-12
            assert abs(row['parameters']['alpha_area_px2']-prep[row['object_id']]['features'][1]) < 1e-8
            rows.append(row)
        source_by_block[b['block']] = rows
    return dict(paths=paths, data=data, prep=prep, lookup=lookup, blocks=blocks,
                deltas=deltas, features=features, current=current,
                source_by_block=source_by_block, raw_lines=raw_lines)


def standardized(features):
    sd = features.std(axis=0)
    assert np.all(sd > 1e-12)
    return ((features.reshape(20,24,-1).mean(axis=1)-features.mean(axis=0))/sd,
            features.mean(axis=0), sd)


def choose_single_swap(blocks, features, deltas, forbidden_pairs):
    """Minimum two changed blocks; deterministic minimum feature movement.

    Hard gate: original 12 and new six shape covariates <=0.5 SD in every
    delta cell. Among eligible single exchanges, prefer target shape <=0.45 SD,
    then minimize the L2 standardized change in the two target cell means;
    block numbers are the final tie break. No answers enter this ordering.
    """
    z, _, sd = standardized(features)
    cells = [deltas.index(d) for d in TARGETS]
    indices = [[i for i,b in enumerate(blocks) if b['delta_Y']==d] for d in TARGETS]
    pair_sets = [{blocks[i]['object_pair'] for i in group} for group in indices]
    candidates = []
    for ia in indices[0]:
        for ib in indices[1]:
            pa, pb = blocks[ia]['object_pair'], blocks[ib]['object_pair']
            if pa in forbidden_pairs or pb in forbidden_pairs or pa in pair_sets[1] or pb in pair_sets[0]:
                continue
            delta = (features[ib,8:]-features[ia,8:])/(24*sd[8:])
            after = z.copy(); after[cells[0],8:] += delta; after[cells[1],8:] -= delta
            if np.max(np.abs(after)) > GATE+1e-12:
                continue
            target = float(np.max(np.abs(after[cells,12:])))
            movement = float(np.linalg.norm(delta))
            candidates.append((target > TARGET_MARGIN, movement, ia, ib, target))
    if not candidates:
        raise ValueError('No admissible single exchange. Do not expand scope or generate media.')
    candidates.sort()
    _, movement, ia, ib, target = candidates[0]
    revised = features.copy()
    revised[[ia,ib],8:] = revised[[ib,ia],8:]
    after, _, _ = standardized(revised)
    assert np.max(np.abs(after)) <= GATE+1e-12
    return dict(blocks=[ia,ib], valid_single_exchange_candidates=len(candidates),
                standardized_movement_L2=movement, target_shape_max=target,
                before=z, after=after, revised_features=revised)


def replacements(inputs, chosen, recipe_sha):
    blocks, prep, lookup = inputs['blocks'], inputs['prep'], inputs['lookup']
    ia, ib = chosen['blocks']; output = []
    for here, donor in [(ia,ib),(ib,ia)]:
        new_pair = blocks[donor]['object_pair']
        for parent in inputs['source_by_block'][here]:
            src = parent['object_id'].split('_')[0]
            new_asset = lookup[new_pair, src]
            assert new_asset not in FORBIDDEN_PHOTO_OBJECTS
            asset = prep[new_asset]
            payload = dict(revision=REVISION, parent_sample_id=parent['sample_id'],
                parent_video_sha256=parent['video_sha256'], object_id=new_asset,
                source_asset_sha256=asset['source_sha256'], prepared_asset_sha256=asset['prepared_sha256'],
                recipe_sha256=recipe_sha)
            child_id = 'nt2_'+digest(payload)[:24]
            p = parent['parameters']; baseline = asset['features'][0]
            assert 0 <= baseline < 1 and 0 < p['delta_Y']/(1-baseline) < 1
            filename = Path(parent['path']).name.replace(parent['object_id'].replace('photo_obj_','obj-na_').replace('gpt_obj_','obj-gpt_'),
                                                       new_asset.replace('photo_obj_','obj-na_').replace('gpt_obj_','obj-gpt_'))
            assert filename != Path(parent['path']).name
            output.append(dict(block=here, donor_block=donor, delta_Y=p['delta_Y'], group=p['group'],
                truth=parent['truth'], onset_frame=p['onset_frame'], offset_frame=p['offset_frame'],
                card=blocks[here]['card'], background_id=parent['background_id'],
                old_object_id=parent['object_id'], object_id=new_asset,
                parent_sample_id=parent['sample_id'], parent_video_sha256=parent['video_sha256'],
                parent_decoded_stream_sha256=parent['decoded_stream_sha256'],
                parent_source_record_sha256=parent['source_record_sha256'],
                sample_id=child_id, proposed_path='T2/'+filename, identity_payload=payload,
                old_source_asset_sha256=prep[parent['object_id']]['source_sha256'],
                new_source_asset_sha256=asset['source_sha256'], new_prepared_asset_sha256=asset['prepared_sha256'],
                baseline_object_Y=baseline, alpha_area_px2=asset['features'][1],
                white_blend_k=p['delta_Y']/(1-baseline), video_sha256=None,
                parent_is_alias=False))
    assert len(output) == 16 and len({r['sample_id'] for r in output}) == 16
    return output


def validate_assignments(inputs, changed):
    current = inputs['current']; changes = {r['parent_sample_id']: r for r in changed}
    assert len(changes) == 16 and set(changes) <= {r['sample_id'] for r in current}
    before_objects, after_objects = Counter(), Counter()
    cells = defaultdict(list)
    for row in current:
        change = changes.get(row['sample_id'])
        assert change is None or row['parameters']['delta_Y'] in TARGETS
        oid = change['object_id'] if change else row['object_id']
        before_objects[row['object_id']] += 1; after_objects[oid] += 1
        cells[row['parameters']['group'],row['parameters']['delta_Y']].append((row,oid))
    assert before_objects == after_objects
    by_changed_block = defaultdict(list)
    for row in changed: by_changed_block[row['block']].append(row)
    assert len(by_changed_block)==2
    for rows in by_changed_block.values():
        assert len(rows)==8 and len({r['object_id'].split('_',1)[1] for r in rows})==1
        groups=defaultdict(list)
        for row in rows:groups[row['group']].append(row)
        assert len(groups)==4
        for members in groups.values():
            assert Counter(r['truth'] for r in members)=={'LEFT':1,'RIGHT':1}
            assert len({r['object_id'] for r in members})==1
    onsets = inputs['data']['criteria']['onsets']
    for rows in cells.values():
        assert len(rows)==48
        assert Counter(r['truth'] for r,_ in rows)=={'LEFT':24,'RIGHT':24}
        assert Counter(r['parameters']['onset_frame'] for r,_ in rows)==dict.fromkeys(onsets,2)
        assert len({r['background_id'] for r,_ in rows})==24
        assert len({oid for _,oid in rows})==24
    return dict(total_videos=3840, changed_videos=16, unchanged_videos=3824,
                non_target_delta_videos_unchanged=3456, group_delta_cells=80,
                videos_per_cell=48, LEFT_per_cell=24, RIGHT_per_cell=24,
                unique_onsets_per_cell=24, unique_BG_per_cell=24, unique_OBJ_per_cell=24,
                all_global_object_use_counts_identical=True,
                BG_onset_duration_truth_group_card_unchanged=True,
                same_object_semantic_pairs_across_four_groups=True,
                non_target_original_raw_catalog_lines_sha256=hashlib.sha256(b''.join(
                    line for line in inputs['raw_lines'] if json.loads(line)['parameters']['delta_Y'] not in TARGETS)).hexdigest())


def build_plan(root):
    inputs = load_inputs(root)
    forbidden = {pair for (pair,src),aid in inputs['lookup'].items() if aid in FORBIDDEN_PHOTO_OBJECTS}
    chosen = choose_single_swap(inputs['blocks'], inputs['features'], inputs['deltas'], forbidden)
    recipe = dict(revision=REVISION, base_criteria_sha256=sha(inputs['paths']['criteria']),
                  intervention='Exchange complete photo+generated object semantic pairs between two existing blocks.',
                  formula='Existing FP64 linear-RGB premultiplied alpha composite and absolute deltaY white blend; no photometric or geometry changes.',
                  grid_and_timing_unchanged=True, parents_are_lineage_not_aliases=True)
    changed = replacements(inputs, chosen, digest(recipe))
    invariants = validate_assignments(inputs, changed)
    before, after = chosen['before'], chosen['after']
    _, pooled_mean, pooled_sd = standardized(inputs['features'])
    detail = []
    for d in TARGETS:
        i = inputs['deltas'].index(d)
        detail.append(dict(delta_Y=d, features=[dict(name=name, pooled_mean=float(pooled_mean[j]),
             pooled_SD=float(pooled_sd[j]), before_mean=float(pooled_mean[j]+before[i,j]*pooled_sd[j]),
             after_mean=float(pooled_mean[j]+after[i,j]*pooled_sd[j]),
             before_signed_shift_SD=float(before[i,j]), after_signed_shift_SD=float(after[i,j]))
             for j,name in enumerate(FEATURE_NAMES)]))
    ia, ib = chosen['blocks']
    selected_assets = sorted({r['object_id'] for r in changed}|{r['background_id'] for r in changed})
    asset_inputs = []
    for aid in selected_assets:
        asset = inputs['prep'][aid]
        folder = 'prepared' if asset['kind']=='background' else 'prepared-vertical-v2'
        rel = f'datasets/{NAME}/{folder}/{aid}.npz'
        path = Path(root)/rel
        assert sha(path)==asset['prepared_sha256'],f'Prepared scientific input changed: {path}'
        asset_inputs.append(dict(asset_id=aid,kind=asset['kind'],source_sha256=asset['source_sha256'],
                                 prepared_relative=rel,prepared_sha256=asset['prepared_sha256']))
    plan = dict(schema='natural-t2-local-rebalance-candidate-v1', revision=REVISION,
        status='CANDIDATE_PASS_FEATURE_GATES_FINAL_ADMISSION_PENDING_ICC_DECISION',
        frozen_for_generation=False, new_model_calls=0, model_answers_read=False,
        source_media_written=False, original_manifests_written=False,
        planning_implementation_sha256=sha(__file__),
        selected_prepared_asset_inputs=asset_inputs,
        source_snapshots={k:dict(path=str(p),sha256=sha(p)) for k,p in inputs['paths'].items()},
        original_source_identity_set_sha256=digest(sorted((r['sample_id'],r['video_sha256'],r['decoded_stream_sha256']) for r in inputs['current'])),
        predeclared_policy=dict(target_deltas=list(TARGETS), max_standardized_shift_SD=GATE,
            shape_covariates=SNAMES, original_covariates=BNAMES+ONAMES,
            unit='480 semantic blocks, 24 per delta. Eight LR/source-group videos per block are matched observations.',
            standardization='Current 480-block pooled mean and population SD; unchanged because global asset counts are preserved.',
            shape_measurement='Source alpha covariance eigenvalue ratio; double-angle anisotropy after recorded placement rotation. Current spoon source geometry replaces historical spoon.',
            selection_order=['fewest changed semantic blocks: one exchange = two blocks',
                             'prefer target-cell six-shape maximum <=0.45 SD',
                             'minimum L2 standardized object+shape cell-mean movement', 'block numbers'],
            excluded_photo_objects=sorted(FORBIDDEN_PHOTO_OBJECTS),
            no_answer_dependent_selection=True, no_new_random_seed=True,
            final_photometric_gate='Recompute current preparation features after the separate ICC decision; this candidate does not approve ICC conversion.'),
        selected_exchange=dict(blocks=[ia,ib], before_object_pairs=[inputs['blocks'][i]['object_pair'] for i in [ia,ib]],
             after_object_pairs=[inputs['blocks'][i]['object_pair'] for i in [ib,ia]],
             valid_single_exchange_candidates=chosen['valid_single_exchange_candidates'],
             standardized_movement_L2=chosen['standardized_movement_L2']),
        maxima=dict(original12_before=float(abs(before[:,:12]).max()), original12_after=float(abs(after[:,:12]).max()),
             shape6_before=float(abs(before[:,12:]).max()), shape6_after=float(abs(after[:,12:]).max()),
             target_shape6_before=float(abs(before[[inputs['deltas'].index(d) for d in TARGETS],12:]).max()),
             target_shape6_after=chosen['target_shape_max']),
        target_before_after=detail, invariants=invariants, exact_changed_videos=changed,
        recipe=recipe, recipe_sha256=digest(recipe), parent_sample_ids_are_aliases=False,
        original_source_retention='Retain every original current video, metadata and model answer. New IDs identify new pixels; no old ID/path overwrite.')
    plan['candidate_plan_sha256'] = digest(plan)
    return plan


def evaluate_final_gates(root, candidate_plan, t2_prepared_overrides, t3_object_overrides=None):
    """Recheck the fixed exchange with declared ICC preparation receipts.

    This pure metadata check does not certify a color conversion or rendered
    pixels. The caller retains those independent receipts. Shape geometry must
    remain unchanged; all pooled photometric moments are recomputed.
    """
    inputs=load_inputs(root)
    assert set(t2_prepared_overrides)==FORBIDDEN_PHOTO_OBJECTS
    assert candidate_plan['original_source_identity_set_sha256']==digest(sorted(
        (r['sample_id'],r['video_sha256'],r['decoded_stream_sha256']) for r in inputs['current']))
    assert digest({k:v for k,v in candidate_plan.items() if k!='candidate_plan_sha256'})==candidate_plan['candidate_plan_sha256']
    assert {inputs['blocks'][i]['delta_Y'] for i in candidate_plan['selected_exchange']['blocks']}==set(TARGETS)
    validate_assignments(inputs,candidate_plan['exact_changed_videos'])
    for aid,new in t2_prepared_overrides.items():
        old=inputs['prep'][aid]
        assert new['asset_id']==aid and len(new['features'])==2
        for key in ['alpha_area_px2','rotation_degrees','source_axis_eigenvalue_ratio']:
            assert np.isclose(new[key],old[key],rtol=0,atol=1e-8),f'ICC changed T2 geometry: {aid}/{key}'
        assert np.isclose(new['features'][1],old['features'][1],rtol=0,atol=1e-8)
        assert np.isfinite(new['features']).all() and 0 <= new['features'][0] < 1
    revised=inputs['features'].copy()
    exchanges=candidate_plan['selected_exchange'];ia,ib=exchanges['blocks']
    revised[[ia,ib],8:]=revised[[ib,ia],8:]
    replacement_pairs=dict(zip(exchanges['blocks'],exchanges['after_object_pairs']))
    for i,b in enumerate(inputs['blocks']):
        pair=replacement_pairs.get(i,b['object_pair'])
        for j,src in enumerate(['photo','gpt']):
            aid=inputs['lookup'][pair,src]
            if aid in t2_prepared_overrides:
                revised[i,8+j*2:10+j*2]=t2_prepared_overrides[aid]['features']
    z,_,_=standardized(revised)
    result=dict(schema='natural-source-final-preparation-feature-gates-v1',
        status='PASS_DECLARED_PREPARATION_FEATURE_GATES' if abs(z).max()<=GATE+1e-12 else 'FAIL_T2_FEATURE_GATE',
        candidate_plan_sha256=candidate_plan['candidate_plan_sha256'],
        original_source_identity_set_sha256=candidate_plan['original_source_identity_set_sha256'],
        new_model_calls=0, model_answers_read=False, media_written=False,
        interpretation='Feature acceptance for declared prepared inputs; independent ICC/pixel/source receipts remain required.',
        T2=dict(original12_max=float(abs(z[:,:12]).max()),shape6_max=float(abs(z[:,12:]).max()),
                target_shape6_max=float(abs(z[[inputs['deltas'].index(d) for d in TARGETS],12:]).max()),
                gate=GATE,PASS=bool(abs(z).max()<=GATE+1e-12),
                failed_features=[dict(delta_Y=inputs['deltas'][i],feature=FEATURE_NAMES[j],signed_shift_SD=float(z[i,j]))
                                 for i,j in np.argwhere(abs(z)>GATE+1e-12)]),
        T2_prepared_override_receipts={aid:dict(source_sha256=r['source_sha256'],prepared_sha256=r['prepared_sha256'],
                                               features=r['features']) for aid,r in sorted(t2_prepared_overrides.items())})
    if t3_object_overrides is not None:
        assert set(t3_object_overrides)=={'photo_obj_025'}
        root=Path(root)
        eligibility_path=root/'plans/natural-t13-20261029/object-eligibility.json'
        allocation_path=root/'plans/natural-t13-20261029/allocation.json'
        eligibility=read(eligibility_path);allocation=read(allocation_path)
        objects={r['asset_id']:r for r in eligibility['objects']}
        objects['photo_obj_021']=inputs['data']['spoon']['T3_asset_preparation']
        for aid,new in t3_object_overrides.items():
            old=objects[aid]
            assert new['asset_id']==aid and len(new['features'])==4
            for key in ['ratio','source_axis_deg','scale','radius','area','baseline_axis_deg']:
                assert np.isclose(new[key],old[key],rtol=0,atol=1e-8),f'ICC changed T3 geometry: {aid}/{key}'
            np.testing.assert_allclose(new['features'][2:],old['features'][2:],rtol=0,atol=1e-8)
            objects[aid]=new
        pairs=eligibility['eligible_pairs']
        pool=np.array([objects['photo_'+p]['features']+objects['gpt_'+p]['features'] for p in pairs])
        standardized_pool=(pool-pool.mean(0))/np.maximum(pool.std(0),1e-10)
        pair_index={p:i for i,p in enumerate(pairs)}
        means=[]
        for cell in range(29):
            blocks=[b for b in allocation['blocks'] if b['task']=='T3' and b['cell']==cell and b['group']=='photo_photo']
            means.append(standardized_pool[[pair_index[b['object_id'].split('_',1)[1]] for b in blocks]].mean(0))
        maximum=float(abs(np.array(means)).max())
        result['T3']=dict(original_object_max=maximum,gate=GATE,PASS=maximum<=GATE+1e-12,
            eligible_object_pairs=len(pairs),angle_cells=29,
            input_sha256=dict(eligibility=sha(eligibility_path),allocation=sha(allocation_path)),
            prepared_override_receipts={aid:dict(source_sha256=r['source_sha256'],source_array_sha256=r['source_array_sha256'],
                                                features=r['features']) for aid,r in sorted(t3_object_overrides.items())})
        if maximum>GATE+1e-12: result['status']='FAIL_T3_FEATURE_GATE'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = build_plan(args.root)
    path=Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:plan[k] for k in ['status','selected_exchange','maxima','invariants','candidate_plan_sha256']},indent=2))


if __name__ == '__main__':
    main()
