"""Plan an append-only T2 source extension; never launches inference or uploads."""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from pathlib import Path

from .dataset import load_index, validate_index
from .dataset_names import filename
from .manifest import atomic_json, digest, load_protocol, make_manifest

DELTA_LEVELS = (.16, .17, .18, .19, .225, .25, .30, .35)
MODELS = ("qwen", "nemo", "cosmos", "gemma")


def build_plan(index):
    validate_index(index)
    if index['counts'] != {'T1': 3840, 'T2': 3744, 'T3': 1776}:
        raise ValueError('This extension is defined against the completed 9360-video panel')
    c = load_protocol()
    c['protocol_name'] = 'classic_t2_transition_extension_v1'
    # This is a source-generation manifest. Legacy model defaults must not launch
    # a BF16 Cosmos/Gemma job; GPU plans are bound separately to qualified NVFP4.
    c['models'] = {}
    c['T2']['delta_Y'] = list(DELTA_LEVELS)
    c['T2']['control_10frame_delta_Y'] = list(DELTA_LEVELS)
    c['T2']['initial_dot_policy'] = 'Fixed object Y=.5 across all backgrounds'
    c['T3']['angle_degrees'] = []
    c['counts'] = {'T2_main': 1152, 'T2_167ms_control': 1152,
                   'T2_total': 2304, 'T3_total': 0, 'total_per_model': 2304,
                   'four_models_total_calls': 9216}
    m = make_manifest(c)
    # Bind unchanged parameters/prompt to an existing row of the same condition.
    old = {}
    for e in index['items'].values():
        if e['task'] != 'T2':
            continue
        p = e['nominal_parameters']
        key = (p['background_Y'], p['offset_frame'] - p['onset_frame'],
               p['onset_index'], p['target_side'])
        q = {k: v for k, v in p.items() if k != 'delta_Y'}
        value = (q, e['prompt'])
        if key in old and old[key] != value:
            raise ValueError('Existing T2 rows disagree beyond luminance increment')
        old[key] = value
    names = []
    for r in m['rows']:
        p = r['source_parameters']
        key = (p['background_Y'], p['offset_frame'] - p['onset_frame'],
               p['onset_index'], p['target_side'])
        if old.get(key) != ({k: v for k, v in p.items() if k != 'delta_Y'}, r['prompt']):
            raise ValueError('Extension would change an existing source/prompt control')
        if r['sample_id'] in index['items']:
            raise ValueError('Requested scientific ID already exists')
        entry = {'task': 'T2', 'truth': r['truth'], 'nominal_parameters': p}
        names.append({'sample_id': r['sample_id'], 'video_relative': 'T2/' + filename(entry)})
    if len({r['video_relative'] for r in names}) != len(names):
        raise ValueError('Readable filename collision')
    plan = {'schema': 'classic-refinement-plan-v1', 'status': 'PREPARED_NOT_RENDERED',
            'base_index_id': index['index_id'], 'base_videos': len(index['items']),
            'source_protocol_id': m['protocol_id'], 'models': list(MODELS),
            'model_precision': 'Existing qualified official NVFP4 mixed checkpoints; retain BF16 excluded layers/KV',
            'input_protocol': 'controlled-rgb16', 'delta_Y': list(DELTA_LEVELS),
            'new_videos': len(names), 'new_model_calls': len(names)*len(MODELS),
            'expected_combined_videos': len(index['items'])+len(names),
            'GPU_calls_started': 0, 'model_execution_plans': 'Bind qualified per-model runtime separately; do not use legacy default pipeline',
            'selection_basis': 'Common fixed grid covering the observed .15-.20 rise and .20-.40 gap; selected before new answers',
            'stop_rule': 'No automatic additional levels or retries of completed/uncertain IDs',
            'id_policy': 'Existing IDs/paths/bytes unchanged; new ID = t2_ + canonical source-parameter digest[:20]',
            'source_acceptance': 'Render and validate only new videos before extending the public index',
            'independent_unit': '24 onset placements with paired sides, not 48 independent scenes'}
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
