"""Plan an append-only T2/T3 source extension; never launches inference or uploads."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from pathlib import Path

from .dataset import load_index, validate_index
from .dataset_names import filename
from .manifest import atomic_json, digest, load_protocol, make_manifest

DELTA_LEVELS = (.16, .17, .18, .19, .25, .30, .35)
ANGLE_GRID = tuple(i / 10 for i in range(10, 26))
MODELS = ("qwen", "nemo", "cosmos", "gemma")


def build_plan(index):
    validate_index(index)
    if index['counts'] != {'T1': 3840, 'T2': 3744, 'T3': 1776}:
        raise ValueError('This extension is defined against the completed 9360-video panel')
    existing_angles = {e['nominal_parameters']['angle_deg'] for e in index['items'].values()
                       if e['task'] == 'T3'}
    angle_levels = tuple(a for a in ANGLE_GRID if a not in existing_angles)
    c = load_protocol()
    c['protocol_name'] = 'classic_t2_t3_transition_extension_v3'
    # This is a source-generation manifest. Legacy model defaults must not launch
    # a BF16 Cosmos/Gemma job; GPU plans are bound separately to qualified NVFP4.
    c['models'] = {}
    c['T2']['delta_Y'] = list(DELTA_LEVELS)
    c['T2']['control_10frame_delta_Y'] = list(DELTA_LEVELS)
    c['T2']['initial_dot_policy'] = 'Fixed object Y=.5 across all backgrounds'
    c['T3']['angle_degrees'] = list(angle_levels)
    t2_n = len(DELTA_LEVELS) * 3 * 2 * 24 * 2
    t3_n = len(angle_levels) * 24 * 2 * 2
    c['counts'] = {'T2_main': t2_n // 2, 'T2_167ms_control': t2_n // 2,
                   'T2_total': t2_n, 'T3_total': t3_n, 'total_per_model': t2_n+t3_n,
                   'four_models_total_calls': (t2_n+t3_n)*len(MODELS)}
    m = make_manifest(c)
    # All controls except the newly varied signal must match the existing panel.
    def control_key(task, p):
        if task == 'T2':
            return (task, p['background_Y'], p['offset_frame']-p['onset_frame'],
                    p['onset_index'], p['target_side'])
        return (task, p['onset_index'], p['reference_side'], p['tilt_sign'])

    def fixed_parameters(task, p):
        varied = 'delta_Y' if task == 'T2' else 'angle_deg'
        return {k: v for k, v in p.items() if k != varied}

    old = {}
    for e in index['items'].values():
        task = e['task']
        if task not in ('T2', 'T3'):
            continue
        p = e['nominal_parameters']
        key = control_key(task, p)
        value = (fixed_parameters(task, p), e['prompt'])
        if key in old and old[key] != value:
            raise ValueError('Existing rows disagree beyond the varied signal')
        old[key] = value
    names = []
    for r in m['rows']:
        task, p = r['task'], r['source_parameters']
        if old.get(control_key(task, p)) != (fixed_parameters(task, p), r['prompt']):
            raise ValueError('Extension would change an existing source/prompt control')
        if r['sample_id'] in index['items']:
            raise ValueError('Requested scientific ID already exists')
        entry = {'task': task, 'truth': r['truth'], 'nominal_parameters': p}
        names.append({'sample_id': r['sample_id'], 'video_relative': task + '/' + filename(entry)})
    if len({r['video_relative'] for r in names}) != len(names):
        raise ValueError('Readable filename collision')
    plan = {'schema': 'classic-refinement-plan-v3', 'status': 'PREPARED_NOT_RENDERED',
            'base_index_id': index['index_id'], 'base_videos': len(index['items']),
            'source_protocol_id': m['protocol_id'], 'models': list(MODELS),
            'model_precision': 'Existing qualified official NVFP4 mixed checkpoints; retain BF16 excluded layers/KV',
            'input_protocol': 'controlled-rgb16', 'delta_Y': list(DELTA_LEVELS),
            'angle_deg': list(angle_levels),
            'T3_regular_grid_deg': list(ANGLE_GRID), 'T3_grid_step_deg': .1,
            'T3_existing_grid_levels_reused': sorted(set(ANGLE_GRID) & existing_angles), 'new_videos_by_task': {'T2': t2_n, 'T3': t3_n},
            'new_videos': len(names), 'new_model_calls': len(names)*len(MODELS),
            'expected_combined_videos': len(index['items'])+len(names),
            'GPU_calls_started': 0, 'model_execution_plans': 'Bind qualified per-model runtime separately; do not use legacy default pipeline',
            'selection_basis': 'Common finite T2 grid without .225; T3 completes a regular 0.1-degree grid from1.0 to2.5, reusing existing levels and retaining extra historical points; selected before new answers',
            'stop_rule': 'No automatic additional levels or retries of completed/uncertain IDs',
            'id_policy': 'Existing IDs/paths/bytes unchanged; new ID = task prefix + canonical source-parameter digest[:20]',
            'source_acceptance': 'Render and validate only new videos before extending the public index',
            'independent_unit': '24 onset placements with paired sides; T3 also retains both signs. These are not independent scenes'}
    return m, plan, names


def extend_index(base, addition):
    """Merge an independently source-audited extension without rewriting old IDs."""
    validate_index(base)
    validate_index(addition)
    if set(base['items']) & set(addition['items']):
        raise ValueError('Extension overlaps existing IDs; never overwrite original observations')
    for e in addition['items'].values():
        if not e.get('source_audit_identity') or not e.get('source_ready_sha256'):
            raise ValueError('Extension lacks actual source audit/ready evidence')
    result = deepcopy(base)
    result.pop('index_id')
    result['items'].update(deepcopy(addition['items']))
    result['counts'] = dict(Counter(e['task'] for e in result['items'].values()))
    result['parent_index_ids'] = [base['index_id'], addition['index_id']]
    result['index_id'] = digest(result)
    return validate_index(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--index', type=Path, required=True)
    plan.add_argument('--output', type=Path, required=True)
    merge = sub.add_parser('extend-index')
    merge.add_argument('--base-index', type=Path, required=True)
    merge.add_argument('--new-index', type=Path, required=True)
    merge.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.command == 'plan':
        m, plan, names = build_plan(load_index(a.index))
        if a.output.exists():
            raise FileExistsError('Use a new source version; do not overwrite a prepared run')
        a.output.mkdir(parents=True)
        atomic_json(a.output/'manifest.json', m)
        atomic_json(a.output/'plan.json', plan)
        with (a.output/'video-names.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['sample_id', 'video_relative'])
            w.writeheader(); w.writerows(names)
        print(f"Prepared {len(names)} new video IDs / {plan['new_model_calls']} planned answers; no rendering or GPU calls")
    else:
        if a.output.exists():
            raise FileExistsError('Never overwrite a published index')
        result = extend_index(load_index(a.base_index), load_index(a.new_index))
        atomic_json(a.output, result)
        print(f"Validated append-only index: {len(result['items'])} videos")


if __name__ == '__main__':
    main()
