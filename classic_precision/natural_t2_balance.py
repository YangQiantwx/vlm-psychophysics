"""Read-only allocation diagnostics; never inspect model answers or dispatch work."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

NAME = 'natural-t2-qwen-x-20261029'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def onehot(labels):
    levels = sorted(set(labels))
    return np.array([[v == k for k in levels] for v in labels], float), levels


def cell_means(values):
    return values.reshape(20, 24, -1).mean(axis=1)


def max_mean_shift(values):
    sd = values.std(axis=0)
    active = sd > 1e-12
    return float(np.max(np.abs((cell_means(values)[:, active]
                               - values.mean(axis=0)[active]) / sd[active])))


def max_trend(means, deltas):
    centered = means - means.mean(axis=0)
    x = deltas - deltas.mean()
    den = np.sqrt(np.sum(centered * centered, axis=0) * np.sum(x * x))
    corr = np.divide(x @ centered, den, out=np.zeros_like(den), where=den > 1e-12)
    return float(np.max(np.abs(corr)))


def max_tv(values):
    return float(np.max(np.sum(np.abs(cell_means(values) - values.mean(axis=0)), axis=1) / 2))


def audit(root, reference_seeds=10000):
    root = Path(root)
    plan = root / 'plans' / NAME
    paths = {
        'allocation': plan / 'allocation.json',
        'criteria': plan / 'allocation-criteria.json',
        'preparation': plan / 'asset-preparation.json',
        'catalog': root / 'assets/library/extensions/t2-natural-20261029/catalog-at-preparation.json',
        'axes': root / 'datasets' / NAME / 'review/axis-algorithm-review.json',
    }
    data = {k: read(p) for k, p in paths.items()}
    alloc, criteria = data['allocation'], data['criteria']
    assert alloc['criteria_sha256'] == sha(paths['criteria'])
    assert alloc['asset_preparation_sha256'] == sha(paths['preparation'])
    assert data['axes']['catalog_sha256'] == sha(paths['catalog'])
    assert data['axes']['current_preparation_sha256'] == sha(paths['preparation'])
    by = {x['asset_id']: x for x in data['preparation']['assets']}
    axes = {x['asset_id']: x for x in data['axes']['assets']}
    objects = data['catalog']['objects']
    lookup = {(a['pair_id'], a['id'].split('_')[0]): a for a in objects}
    objpairs = sorted({a['pair_id'] for a in objects})
    assert len(objpairs) == 50
    deltas = np.array(criteria['deltas'])
    bfeatures = np.array([[by[f'{src}_bg_{i+1:03}']['features']
                          for src in ['photo', 'gpt']] for i in range(100)]).reshape(100, -1)
    ofeatures = np.array([[by[lookup[pair, src]['id']]['features']
                          for src in ['photo', 'gpt']] for pair in objpairs]).reshape(50, -1)
    bnames = [f'{src}_background_{side}_{stat}' for src in ['photo', 'gpt']
              for side in ['left', 'right'] for stat in ['mean_Y', 'std_Y']]
    onames = [f'{src}_object_{stat}' for src in ['photo', 'gpt'] for stat in ['mean_Y', 'alpha_area']]
    snames = [f'{src}_{stat}' for src in ['photo', 'gpt']
              for stat in ['log_axis_eigenvalue_ratio', 'shape_cos2theta', 'shape_sin2theta']]
    shape = []
    for pair in objpairs:
        row = []
        for src in ['photo', 'gpt']:
            aid = lookup[pair, src]['id']
            ratio = axes[aid]['eigenvalue_ratio']
            angle = np.deg2rad(axes[aid]['pca_axis_degrees_mod180'] + by[aid]['rotation_degrees'])
            anisotropy = (ratio - 1) / (ratio + 1)
            row.extend([np.log(ratio), anisotropy * np.cos(2 * angle), anisotropy * np.sin(2 * angle)])
        shape.append(row)
    shape = np.array(shape)
    # Existing source families differ (e.g. photo "kitchen" versus generated "food").
    # Use one explicitly recorded semantic-family map for both sides of each pair.
    new_families = {'obj_051': 'kitchen', 'obj_052': 'workbench',
                    'obj_053': 'workbench', 'obj_054': 'workbench'}
    object_families = [lookup[p, 'gpt'].get('family', new_families.get(p)) for p in objpairs]
    assert all(object_families)
    oh, olevels = onehot(object_families)
    backgrounds = {x['id']: x for x in data['catalog']['backgrounds']}
    bfamily = [backgrounds[f'photo_bg_{i+1:03}']['family'] for i in range(100)]
    assert bfamily == [backgrounds[f'gpt_bg_{i+1:03}']['family'] for i in range(100)]
    bh, blevels = onehot(bfamily)
    placement = []
    for src in ['photo', 'gpt']:
        arr, _ = onehot([by[lookup[p, src]['id']]['placement_axis_rule'] for p in objpairs])
        placement.append(arr)

    blocks = alloc['blocks']
    assert len(blocks) == 480 and [x['block'] for x in blocks] == list(range(480))
    bi = np.array([int(x['background_pair'].split('_')[-1]) - 1 for x in blocks])
    oi = np.array([objpairs.index(x['object_pair']) for x in blocks])
    groups = defaultdict(list)
    for block in blocks:
        assert len(block['cases']) == 8
        assert len({c['group'] for c in block['cases']}) == 4
        assert all(c['delta_Y'] == block['delta_Y'] and c['onset_frame'] == block['onset_frame']
                   and c['object_id'] == lookup[block['object_pair'], c['object_id'].split('_')[0]]['id']
                   and c['background_id'].endswith(block['background_pair']) for c in block['cases'])
        for case in block['cases']:
            groups[case['group'], case['delta_Y']].append(case)
    assert len(groups) == 80
    for rows in groups.values():
        assert len(rows) == 48
        assert Counter(x['truth'] for x in rows) == {'LEFT': 24, 'RIGHT': 24}
        assert Counter(x['onset_frame'] for x in rows) == dict.fromkeys(criteria['onsets'], 2)
        assert len({x['background_id'] for x in rows}) == len({x['object_id'] for x in rows}) == 24

    def draw(seed):
        rng = np.random.default_rng(seed)
        bp, op = rng.permutation(100), rng.permutation(50)
        idx = np.arange(480)
        return bp[idx % 100], op[(idx + 11 * (idx // 100)) % 50]

    def gate(bg, obj):
        return max_mean_shift(np.concatenate([bfeatures[bg], ofeatures[obj]], axis=1))

    def metrics(bg, obj):
        return {
            'object_family_max_total_variation': max_tv(oh[obj]),
            'background_family_max_total_variation': max_tv(bh[bg]),
            'object_family_max_abs_delta_correlation': max_trend(cell_means(oh[obj]), deltas),
            'background_family_max_abs_delta_correlation': max_trend(cell_means(bh[bg]), deltas),
            'shape_orientation_max_mean_shift_SD': max_mean_shift(shape[obj]),
            'shape_orientation_max_abs_delta_correlation': max_trend(cell_means(shape[obj]), deltas),
            'placement_rule_max_total_variation': max(max_tv(x[obj]) for x in placement),
        }

    replay_bi, replay_oi = draw(alloc['selected_seed'])
    assert np.array_equal(bi, replay_bi) and np.array_equal(oi, replay_oi)
    observed = metrics(bi, oi)
    original_score = gate(bi, oi)
    assert original_score <= criteria['max_standardized_mean_deviation']
    simulations, accepted_seeds = [], []
    for seed in range(reference_seeds):
        if seed == alloc['selected_seed']:
            continue
        bg, obj = draw(seed)
        if gate(bg, obj) <= criteria['max_standardized_mean_deviation']:
            simulations.append(metrics(bg, obj))
            accepted_seeds.append(seed)
    assert len(simulations) >= 100, 'Reference too small; inspect without altering live allocation'
    comparison = {}
    for key, value in observed.items():
        vals = np.array([x[key] for x in simulations])
        comparison[key] = {'observed': value, 'reference_median': float(np.median(vals)),
                           'reference_p95': float(np.quantile(vals, .95)),
                           'reference_p99': float(np.quantile(vals, .99)),
                           'fraction_reference_ge_observed': float(np.mean(vals >= value - 1e-12)),
                           'above_reference_p99': bool(value > np.quantile(vals, .99) + 1e-12)}
    features = np.concatenate([bfeatures[bi], ofeatures[oi], shape[oi]], axis=1)
    names = bnames + onames + snames
    means = cell_means(features)
    ocounts, bcounts = cell_means(oh[oi]) * 24, cell_means(bh[bi]) * 24
    shifts = (means - features.mean(axis=0)) / features.std(axis=0)
    feature_details = []
    for j, name in enumerate(names):
        peak = int(np.argmax(np.abs(shifts[:, j])))
        feature_details.append({'feature': name, 'largest_shift_delta': float(deltas[peak]),
                                'signed_shift_SD': float(shifts[peak, j]),
                                'cell_mean': float(means[peak, j]),
                                'pooled_mean': float(features[:, j].mean()),
                                'pooled_SD': float(features[:, j].std()),
                                'max_abs_delta_correlation': max_trend(means[:, j:j+1], deltas)})
    rows = []
    for i, delta in enumerate(deltas):
        row = {'delta_Y': float(delta), 'semantic_blocks_per_group': 24,
               'videos_per_group': 48, 'left_per_group': 24, 'right_per_group': 24}
        row.update({f'object_family_{k}_blocks': int(round(v)) for k, v in zip(olevels, ocounts[i])})
        row.update({f'background_family_{k}_blocks': int(round(v)) for k, v in zip(blevels, bcounts[i])})
        row.update({k: float(v) for k, v in zip(names, means[i])})
        for source in ['photo', 'gpt']:
            row[f'{source}_elongated_objects_ratio_ge4'] = sum(
                axes[lookup[objpairs[int(j)], source]['id']]['eigenvalue_ratio'] >= 4
                for j in oi[i*24:(i+1)*24])
        rows.append(row)
    report = {
        'status': 'COMPLETE_READ_ONLY_ALLOCATION_DIAGNOSTICS',
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'new_model_calls': 0, 'live_allocation_changed': False, 'model_answers_read': False,
        'inputs': {k: {'path': str(p), 'sha256': sha(p)} for k, p in paths.items()},
        'implementation_sha256': sha(__file__),
        'unit': '480 semantic blocks; 8 matched LR/source-group videos per block are not independent draws',
        'hard_balance': {'group_delta_cells': 80, 'videos_each': 48, 'left_each': 24, 'right_each': 24,
                         'onsets_each': 24, 'unique_bg_each': 24, 'unique_obj_each': 24,
                         'same_semantic_blocks_across_four_groups': True},
        'original_feature_gate': {'observed': original_score, 'maximum': .5, 'PASS': True},
        'semantic_family_mapping': dict(zip(objpairs, object_families)),
        'semantic_family_mapping_provenance': 'Audit only: use existing generated counterpart family for both semantic pair members; four new assets assigned explicit kitchen/workbench families by object category. Does not modify catalog or stimuli.',
        'shape_features': 'Original alpha-area covariance eigenvalues; direction rotated by recorded similarity transform. cos/sin of double angle weighted by anisotropy. Geometric shape descriptor, not human-perceived or physical upright; not remeasured post-rasterization.',
        'reference': {'proposal_seed_count': reference_seeds - 1, 'accepted': len(accepted_seeds),
                      'proposal_seed_range': [0, reference_seeds - 1], 'excluded_observed_seed': alloc['selected_seed'],
                      'accepted_seed_digest': hashlib.sha256(json.dumps(accepted_seeds).encode()).hexdigest(),
                      'design': 'Exact original cyclic allocation and original max0.5SD feature acceptance; diagnostic candidate allocations only, never applied.',
                      'interpretation': 'Finite randomization reference, not probability of no confounding. A nonextreme rank does not prove absence of material imbalance. Diagnostics were not original admission criteria.'},
        'diagnostics': comparison,
        'feature_details': feature_details,
        'shape_cell_attention': [{'delta_Y': float(deltas[i]), 'feature': names[j],
                                  'signed_shift_SD': float(shifts[i, j])}
                                 for i in range(20) for j in range(len(bnames)+len(onames), len(names))
                                 if abs(shifts[i, j]) > comparison['shape_orientation_max_mean_shift_SD']['reference_p99']],
        'object_family_per_delta_count_ranges': {k: [int(round(ocounts[:, j].min())), int(round(ocounts[:, j].max()))] for j, k in enumerate(olevels)},
        'background_family_per_delta_count_ranges': {k: [int(round(bcounts[:, j].min())), int(round(bcounts[:, j].max()))] for j, k in enumerate(blevels)},
        'limits': ['No model difficulty or unobserved visual nuisance was measured.',
                   'Shape/semantic balance does not establish that the estimated performance curve is free from stimulus composition effects.',
                   'Result-level clustered or crossed-asset analysis remains separate from this allocation-only check.'],
    }
    out = root / 'runs/reproducibility' / NAME
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'allocation-balance-cells.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    target = out / 'allocation-balance-audit.json'
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n'); temporary.replace(target)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--reference-seeds', type=int, default=10000)
    args = parser.parse_args()
    result = audit(args.root, args.reference_seeds)
    print(json.dumps({k: result[k] for k in ['status', 'hard_balance', 'original_feature_gate',
                                           'reference', 'diagnostics', 'object_family_per_delta_count_ranges']},
                     ensure_ascii=False), flush=True)
