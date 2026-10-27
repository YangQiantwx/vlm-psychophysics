"""Finite T1/T2 completion with the qualified official Cosmos session unchanged.

Only catalog selection, paired three-card allocation and finite orchestration
are new. The original model, input bridge, engine, decoding, lease and durable
execution functions are called directly from nvfp4_cosmos_official.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import math
import os
from pathlib import Path
import re
import time

from . import nvfp4_cosmos_official as o
from .storage import RunStore, canonical_json, digest, file_sha256, owner_identity

SCHEMA = 'cosmos-official-nvfp4-classic-completion-v1'
BASE_SHA = '04cfb6e0f5bd581f8791d674c43aadaaeeb6da536256826f5c1b5a901359145e'
BASE_PLAN_SHA = '5cafb6898c4c24c044af745df4ec743599c0e71255680fea9272f9deeced1a2e'
SHARDS = ('fourier0', 'fourier1', 'radon0')
TARGET = {'T1': 3840, 'T2': 3744}
require, read, write_new = o.require, o.read, o.write_new


def parameters(item):
    row = item['row']
    return row.get('source_parameters', row.get('params'))


def truth(item):
    row = item['row']
    return row.get('truth', row.get('correct_answer'))


def cell_key(item):
    p = parameters(item)
    if item['task'] == 'T1':
        return ('T1', p['sf_cycles_per_aperture'], p['modulation_depth'])
    require(item['task'] == 'T2', 'Only T1/T2 are dispatched')
    return ('T2', p['background_Y'], p['delta_Y'], p['offset_frame'] - p['onset_frame'])


def pair_key(item):
    p = parameters(item)
    return cell_key(item) + (p['phase_degrees'] if item['task'] == 'T1' else p['onset_index'],)


def selected_cell(item):
    return canonical_json(cell_key(item))


def validate_reuse(reference, catalog):
    require(reference['schema'] == SCHEMA + '/reused-reference' and reference['status'] == 'COMPLETE_PASS',
            'Closed pilot reference required')
    reused = reference['items']
    require(len(reused) == 288 and Counter(catalog['items'][sid]['task'] for sid in reused)
            == {'T1': 144, 'T2': 144}, 'Reuse exactly the completed288 T1/T2 IDs')
    for sid, value in reused.items():
        item, raw = catalog['items'][sid], value['raw']
        require(raw['sample_id'] == sid and digest(raw) == value['raw_sha256'], 'Reused raw identity differs')
        require(raw['catalog_row_sha256'] == item['row_sha256']
                and raw['catalog_ready_sha256'] == item['ready_sha256']
                and raw['task'] == item['task'] and raw['truth'] == truth(item), 'Reused source differs')
        require(value['technical_pass_raw_sha256'] == value['raw_sha256']
                and raw['finish_reason'] == 'stop' and raw['raw_answer'] in ('LEFT', 'RIGHT'),
                'Reused output lacks its closed technical verdict')
        tokens = raw.get('output_token_ids')
        require(isinstance(tokens, list) and len(tokens) >= 2
                and tokens[-1] in raw['effective_generation']['native_eos_token_ids']
                and value['technical_pass_attempt_id'] == raw['attempt_id'], 'Reused terminal tokens/attempt differ')
    return set(reused)


def validate_allocation(allocation, catalog, reference):
    reused = validate_reuse(reference, catalog)
    target = {sid for sid, item in catalog['items'].items() if item['task'] in TARGET}
    require(Counter(catalog['items'][sid]['task'] for sid in target) == TARGET, 'Full target7584 differs')
    require(allocation['schema'] == SCHEMA + '/allocation' and set(allocation['new_assignments']) == set(SHARDS),
            'Three explicit physical shards required')
    require(set(allocation['reused_ids']) == reused and len(allocation['reused_ids']) == 288, 'Reused IDs changed')
    groups = allocation['new_assignments']; new = sum((groups[k] for k in SHARDS), [])
    require(len(new) == len(set(new)) == 7296 and not set(new) & reused
            and set(new) | reused == target, 'New/reused coverage or uniqueness differs')
    require(allocation['max_new_calls'] == 7296 and allocation['max_new_calls_per_id'] == 1
            and allocation['new_bf16_calls'] == 0
            and allocation['automatic_retry'] is False, 'Finite scientific call contract differs')
    all_cells = {cell_key(catalog['items'][sid]) for sid in new}
    for name, ids in groups.items():
        require(len(ids) >= 2 and [catalog['items'][sid]['task'] for sid in ids[:2]] == ['T1', 'T2'],
                'Each fresh worker first qualifies one real T1 and one real T2 ID')
        pairs = defaultdict(list)
        for sid in ids:
            pairs[pair_key(catalog['items'][sid])].append(truth(catalog['items'][sid]))
        require(all(sorted(v) == ['LEFT', 'RIGHT'] for v in pairs.values()), 'A left/right pair was split')
        require({cell_key(catalog['items'][sid]) for sid in ids} == all_cells,
                'Every remaining condition must be represented on each card')


def allocate(catalog, reference, costs, seed):
    reused = validate_reuse(reference, catalog)
    require(set(costs) == set(SHARDS) and all(math.isfinite(x) and x > 0 for x in costs.values()),
            'Three positive measured iteration proxies required')
    speed = {k: 1 / costs[k] for k in SHARDS}; total = sum(speed.values())
    share = {k: speed[k] / total for k in SHARDS}
    cells = defaultdict(lambda: defaultdict(list))
    for sid, item in catalog['items'].items():
        if item['task'] in TARGET and sid not in reused:
            cells[cell_key(item)][pair_key(item)].append(sid)
    groups = {k: [] for k in SHARDS}; pair_counts = Counter(); n_pairs = 0
    for cell in sorted(cells, key=lambda c: digest([seed, 'cell', c])):
        pairs = cells[cell]
        require(len(pairs) == 24 and all(len(v) == 2 for v in pairs.values()), 'A remaining48-cell is incomplete')
        quota = {k: math.floor(24 * share[k]) for k in SHARDS}
        for _ in range(24 - sum(quota.values())):
            chosen = max(SHARDS, key=lambda k: ((n_pairs + 24) * share[k] - pair_counts[k] - quota[k], k))
            quota[chosen] += 1
        owners = [(k, i) for k in SHARDS for i in range(quota[k])]
        owners.sort(key=lambda x: digest([seed, cell, 'owner', x]))
        ordered = sorted(pairs, key=lambda key: digest([seed, 'pair', key]))
        for key, (owner, _) in zip(ordered, owners):
            groups[owner].extend(sorted(pairs[key])); pair_counts[owner] += 1
        n_pairs += 24
    for name, ids in groups.items():
        first = [next(sid for sid in ids if catalog['items'][sid]['task'] == task) for task in ['T1', 'T2']]
        groups[name] = first + [sid for sid in ids if sid not in first]
    result = {'schema': SCHEMA + '/allocation', 'seed': seed, 'iteration_seconds_proxy': costs,
              'allocation_rule': 'Score-independent hashed conditions/onset pairs; cumulative rate-weighted quotas; LR pairs never split.',
              'new_assignments': groups, 'reused_ids': sorted(reused), 'max_new_calls': 7296,
              'max_new_calls_per_id': 1, 'automatic_retry': False, 'new_bf16_calls': 0,
              'by_shard_task': {k: dict(Counter(catalog['items'][sid]['task'] for sid in ids)) for k, ids in groups.items()},
              'predicted_worker_seconds': {k: len(ids) * costs[k] for k, ids in groups.items()}}
    validate_allocation(result, catalog, reference)
    return result


def validate_contract(plan, base, allocation, catalog, reference):
    validate_allocation(allocation, catalog, reference)
    names = ['fourier0', 'fourier1'] if plan['host'] == 'fourier' else ['radon0']
    require(plan['schema'] == SCHEMA and plan['host'] in ('fourier', 'radon')
            and plan['shard_names'] == names, 'Host/schema/shards differ')
    require(plan['assignments'] == [allocation['new_assignments'][k] for k in names], 'Host assignments differ')
    require(plan['generation'] == base['generation'] == o.inference.GENERATION, 'Generation contract differs')
    left, right = dict(base['engine_settings']), dict(plan['engine_settings'])
    for key in ('model', 'tokenizer'):
        left.pop(key); right.pop(key)
    require(left == right and plan['engine_settings'] == o.engine_settings(plan['runtime']), 'Qualified engine differs')
    require(plan['runtime']['resources'] == base['runtime']['resources'], 'Qualified resource margin differs')
    require(set(plan['runtime']['physical_locks']) == {str(i) for i in range(len(names))}, 'Native leases differ')
    ids = sum(plan['assignments'], [])
    require(plan['max_new_calls'] == len(ids) and plan['max_new_calls_per_id'] == plan['max_total_attempts_per_id'] == 1
            and plan['automatic_retry'] is False and plan['new_bf16_calls'] == 0
            and plan['deep_calls_per_fresh_worker'] == 2 and plan['prior_failed_nvfp4_attempts'] == {},
            'Finite one-attempt call contract differs')
    require([x['sample_id'] for x in plan['panel']['items']] == ids, 'Panel order differs')
    for item in plan['panel']['items']:
        copy = dict(item); label = copy.pop('selected_cell_id')
        require(copy == catalog['items'][item['sample_id']] and label == selected_cell(copy), 'Catalog input differs')


def validate(plan, physical=True):
    refs = plan['references']
    require(refs['base_plan']['sha256'] == BASE_PLAN_SHA and refs['catalog']['sha256'] == o.native.CATALOG_SHA,
            'Qualified reference anchor differs')
    base, allocation, catalog, reference = [read(o.verified(refs[k]))
                                           for k in ('base_plan', 'allocation', 'catalog', 'reuse_reference')]
    validate_contract(plan, base, allocation, catalog, reference)
    require(file_sha256(o.__file__) == BASE_SHA and file_sha256(__file__) == plan['extension_source_sha256'],
            'Qualified base or new orchestration source differs')
    if physical:
        for lock in plan['runtime']['physical_locks'].values():
            state = o.file_state(lock['path'])
            require(all(state[k] == lock[k] for k in ('st_dev', 'st_ino')), 'Physical lock inode changed')
        current = o.runtime_identity(plan['runtime'])
        require(current == plan['runtime_identity'], 'Sealed runtime changed')
        for key in ('packages', 'sdk_sources', 'implementation', 'environment', 'cosmos_package', 'native_t1_sha256'):
            require(current[key] == base['runtime_identity'][key], 'Pilot runtime differs: ' + key)


def seal(args):
    base, allocation, catalog, reference = [read(p) for p in
        (args.base_plan, args.allocation, args.catalog, args.reuse_reference)]
    runtime = read(args.runtime); names = ['fourier0', 'fourier1'] if args.host == 'fourier' else ['radon0']
    groups = [allocation['new_assignments'][k] for k in names]
    items = [{**catalog['items'][sid], 'selected_cell_id': selected_cell(catalog['items'][sid])}
             for sid in sum(groups, [])]
    plan = {'schema': SCHEMA, 'host': args.host, 'shard_names': names,
            'references': {k: o.descriptor(v) for k, v in
                (('base_plan', args.base_plan), ('allocation', args.allocation), ('catalog', args.catalog),
                 ('reuse_reference', args.reuse_reference))},
            'runtime': runtime, 'runtime_identity': o.runtime_identity(runtime),
            'extension_source_sha256': file_sha256(__file__), 'panel': {'items': items}, 'assignments': groups,
            'generation': base['generation'], 'engine_settings': o.engine_settings(runtime),
            'max_new_calls': sum(map(len, groups)), 'max_new_calls_per_id': 1, 'max_total_attempts_per_id': 1,
            'automatic_retry': False, 'new_bf16_calls': 0, 'deep_calls_per_fresh_worker': 2,
            'prior_failed_nvfp4_attempts': {}}
    validate(plan); write_new(args.output, plan)
    return {'status': 'SEALED', 'sha256': file_sha256(args.output), 'new_calls': plan['max_new_calls']}


def run(args):
    plan = read(args.plan); validate(plan)
    require(0 <= args.worker < len(plan['assignments']) and re.fullmatch(r'[A-Za-z0-9_.-]+', args.execution), 'Invalid worker')
    require(args.limit == len(plan['assignments'][args.worker]), 'Run exactly the assigned finite shard')
    admission = read(args.admission); plan_sha = file_sha256(args.plan)
    require(admission['plan_sha256'] == plan_sha and all(admission[k] is True for k in
            ('production_held', 'owned_engines_exited', 'physical_locks_released')), 'Actual launch admission missing')
    require(admission['workers'][str(args.worker)]['cuda_visible_devices'] == os.environ.get('CUDA_VISIBLE_DEVICES'),
            'Wrong physical GPU')
    root = Path(plan['runtime']['output_root']); directory = root / 'executions' / args.execution
    directory.mkdir(parents=True, exist_ok=False)
    binding = {'schema': SCHEMA, 'plan_sha256': plan_sha, 'worker': args.worker,
               'shard': plan['shard_names'][args.worker], 'execution': args.execution, 'owner': owner_identity(),
               'admission_sha256': file_sha256(args.admission), 'limit': args.limit}
    write_new(directory / 'binding.json', binding)
    identity = {'schema': SCHEMA, 'plan_sha256': plan_sha,
                'allocation_sha256': plan['references']['allocation']['sha256'],
                'reuse_reference_sha256': plan['references']['reuse_reference']['sha256']}
    status, started = 'FAILED', time.monotonic()
    try:
        with RunStore(root, identity, o.MODEL_KEY, 'worker' + str(args.worker),
                      plan['assignments'][args.worker], sum(plan['assignments'], [])) as store, o._signals():
            o.select_pending(store, plan['assignments'][args.worker], args.limit)
            os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
            import torch
            require(torch.version.cuda == '13.0', 'Qualified CUDA13.0 required'); o._configure_torch(torch)
            done = o.execute_segment(plan, plan_sha, args.worker, args.limit, store, torch, args.execution)
            require(done == plan['assignments'][args.worker], 'Finite shard incomplete')
            status = 'SEGMENT_COMPLETE'
            store.event(status, execution=args.execution, IDs=done, elapsed_seconds=time.monotonic() - started)
        write_new(directory / 'complete.json', {**binding, 'status': status, 'IDs': done, 'new_model_calls': len(done)})
    except BaseException as exc:
        write_new(directory / 'failed.json', {**binding, 'status': 'FAILED', 'error_type': type(exc).__name__,
                  'error': str(exc), 'automatic_retry': False, 'elapsed_seconds': time.monotonic() - started})
        raise
    finally:
        write_new(directory / 'worker_return.json', {**binding, 'status': status, 'process_exit_observed': False})
    return {'status': status, 'new_model_calls': len(done)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__); sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('allocate')
    for name in ('catalog', 'reuse-reference', 'costs', 'seed', 'output'): q.add_argument('--' + name, required=True)
    q = sub.add_parser('seal')
    for name in ('base-plan', 'allocation', 'catalog', 'reuse-reference', 'runtime', 'output'): q.add_argument('--' + name, required=True)
    q.add_argument('--host', choices=('fourier', 'radon'), required=True)
    q = sub.add_parser('check'); q.add_argument('--plan', required=True)
    q = sub.add_parser('run')
    for name in ('plan', 'admission', 'execution'): q.add_argument('--' + name, required=True)
    q.add_argument('--worker', type=int, required=True); q.add_argument('--limit', type=int, required=True)
    args = p.parse_args(argv)
    if args.command == 'allocate':
        result = allocate(read(args.catalog), read(args.reuse_reference), read(args.costs), args.seed)
        write_new(args.output, result); result = {'status': 'SEALED_ALLOCATION', 'new_calls': 7296}
    elif args.command == 'check':
        validate(read(args.plan)); result = {'status': 'CPU_METADATA_PASS', 'model_calls': 0}
    else:
        result = {'seal': seal, 'run': run}[args.command](args)
    print(canonical_json(result)); return result


if __name__ == '__main__':
    main()
