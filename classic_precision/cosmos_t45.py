"""Finite T4/T5 classic/natural execution on the already qualified Cosmos X.

Only admission, lifecycle and observer checks are added. The official native
reader, processor, model forward, kernels and sampling remain in the frozen
official_x_runner/v2. Each source shard is immutable before its first call.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile

from . import nemo_x_full as lifecycle
from . import official_x_runner_v2 as native
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

NAME = 'cosmos-x-t45-20261030'
base = native.base
ORIGINAL_DEEP = base.validate_deep
RUNTIME_FIELDS = ('model_class', 'method_kernel_counts', 'actual_backends',
                  'kernel_config', 'mm_device_do_normalize')
SETTINGS = ('model', 'checkpoint', 'checkpoint_assets', 'expected_packages',
            'environment', 'cleared_environment', 'request_mm_processor_kwargs',
            'vision_observation', 'minimum_kv_capacity', 'torch_num_threads',
            'gpu_uuid', 'physical_lock', 'python')


def science(p, resolve=verified):
    parent = resolve(p['qualified_profile'])
    require(p['model'] == 'cosmos' and not p.get('bootstrap') and not p.get('request_adapter'),
            'Only the qualified native Cosmos route is allowed')
    for key in SETTINGS:
        require(p[key] == parent[key], 'Qualified Cosmos X setting changed: ' + key)
    engine = copy.deepcopy(parent['engine_kwargs'])
    engine['limit_mm_per_prompt'] = {'audio': 0, 'image': 1, 'video': 1}
    require(p['engine_kwargs'] == engine, 'Only one T5 reference image may be admitted additionally')
    require(p.get('media_io_kwargs', base.MEDIA) == base.MEDIA, 'All-frame official reader changed')
    require(not p['automatic_retry'] and not p['formal_auto_resume'], 'Automatic retry forbidden')
    plan = resolve(p['execution_plan'])
    require(plan['maximum_model_calls'] == 1200 and plan['model'] == 'cosmos'
            and plan['user_authorized'] is True, 'No bound finite authorization')
    all_ids = [sid for shard in plan['shards'].values() for sid in shard['assigned_ids']]
    require(len(all_ids) == len(set(all_ids)) == 1200, 'Global finite allocation differs')
    group = plan['shards'][p['shard']]
    ids = p['assigned_ids']
    require(len(ids) == len(set(ids)) == p['maximum_model_calls']
            and set(ids) <= set(group['assigned_ids']), 'Foreign or duplicate source allocation')
    if not p.get('continuation_parent'):
        require(ids == group['assigned_ids'] and p['control_generation'] == 0, 'Initial shard changed')
    else:
        previous = resolve(p['continuation_parent'])
        require(p['control_generation'] == previous['control_generation'] + 1, 'Wrong continuation generation')
        if resolve is verified:
            _, terminal, raws = closed(p['continuation_parent']['path'])
            require(ids == terminal['unattempted_ids']
                    == [s for s in previous['assigned_ids'] if s not in raws], 'Resume repeats attempted input')
    catalog = resolve(p['source_catalog'])
    require(set(catalog['items']) == set(group['assigned_ids']), 'Frozen shard source set differs')
    source_acceptance = resolve(p['source_acceptance'])
    require(source_acceptance['status'] == 'PASS_FROZEN_SHARD_SOURCES'
            and source_acceptance['catalog_sha256'] == p['source_catalog']['sha256']
            and source_acceptance['assigned_ids'] == group['assigned_ids'], 'Source admission not bound')
    for sid in ids:
        item = catalog['items'][sid]
        row = item['row']; task = item['task']; media = item['content']
        require(task in ('T4', 'T5') and row['task'] == task, 'Wrong task')
        require({k: item['video_contract'][k] for k in ('fps', 'frames', 'height', 'width')}
                == {'fps': 60, 'frames': 180, 'height': 512, 'width': 768}, 'Source geometry changed')
        require(sum(m['type'] == 'video' for m in media) == 1
                and sum(m['type'] == 'image' for m in media) == int(task == 'T5'), 'Wrong native media modalities')
        require([m['text'] for m in media if m['type'] == 'text'] == [row['prompt']]
                and row['prompt'] == plan['prompts'][task], 'Scientific question changed')
        require(row['choices'] == (['YES', 'NO'] if task == 'T5' else ['LEFT', 'RIGHT']), 'Output choices changed')
        require(row['truth'] in row['choices'] if row['accuracy_eligible'] else task == 'T4' and row['truth'] is None,
                'Ground truth or zero-coherence eligibility differs')
    root = Path(p['workspace_root'])
    require(Path(p['output']).parent == root / 'runs/reproducibility' / NAME
            and Path(p['claim_root']) == root / 'runs/reproducibility' / NAME / 'claims' / p['shard'],
            'Run or claim namespace changed')


class Boundary(lifecycle.Boundary):
    pass


def validate_deep(p, item, deep):
    expected = verified(p['qualified_runtime'])
    require([{k: worker['runtime'][k] for k in RUNTIME_FIELDS} for worker in deep]
            == [{k: worker[k] for k in RUNTIME_FIELDS} for worker in expected],
            'Actual request runtime differs from the qualified Cosmos X')
    ORIGINAL_DEEP(p, item, deep)
    if item['task'] == 'T5':
        raw = base.records(Path(p['output']) / 'raw.jsonl')[-1]
        grids = [record[phase]['image_grid_thw'].get('values')
                 for record in raw['native_processor']
                 for phase in ('before_native_dtype', 'after_native_dtype')
                 if 'image_grid_thw' in record[phase]]
        require([[1, 10, 10]] in grids, 'Square T5 reference was resized or omitted')
        require(any(row.get('grid_thw') in ([1, 10, 10], [[1, 10, 10]])
                    for worker in deep for row in worker['outer']),
                'No actual GPU input for the square T5 reference')


def seal(draft, output):
    p = read(draft); science(p); p['operational_wrapper'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temp = stream.name
    try:
        return native.seal(temp, output)
    finally:
        Path(temp).unlink()


def run(profile):
    p = read(profile); science(p)
    require(p['operational_wrapper']['sha256'] == file_sha256(__file__), 'Execution wrapper changed')
    expected = verified(p['qualified_runtime'])
    # Runtime identity is checked in the first deep observation of each task.
    require(expected and all(k in expected[0] for k in RUNTIME_FIELDS), 'Qualified runtime receipt missing')
    base.Boundary = Boundary
    base.validate_deep = validate_deep
    return native.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    original = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = original


def closed(profile):
    p, terminal, raw = lifecycle.closed(profile)
    require(terminal['completed_ids'] == p['assigned_ids'][:len(raw)], 'Not an ordered clean prefix')
    if raw:
        runtime = read(Path(p['output']) / 'runtime.json')
        expected = verified(p['qualified_runtime'])
        require([{k: x[k] for k in RUNTIME_FIELDS} for x in runtime]
                == [{k: x[k] for k in RUNTIME_FIELDS} for x in expected], 'Actual Cosmos runtime differs')
    return p, terminal, raw


def activate(profile):
    p = read(profile); science(p); control = read(p['control_file'])
    require(control['generation'] == p['control_generation'] and not Path(p['output']).exists()
            and not Path(p['output']).with_suffix('.launch.json').exists(), 'Already attempted or stale profile')
    require(not any((Path(p['claim_root']) / (sid + '.json')).exists() for sid in p['assigned_ids']), 'Claim exists')
    control.update(paused=False, activated_profile=descriptor(profile))
    atomic_json(p['control_file'], control)
    return {'status': 'ACTIVATED_NOT_LAUNCHED', 'profile': descriptor(profile)}


def prepare_shard(root, shard):
    """Freeze one completed CPU source partition; never waits for other partitions."""
    root = Path(root).resolve(); plans = root / 'plans' / NAME
    plan_path = plans / 'execution-plan.json'; plan = read(plan_path)
    group = plan['shards'][shard]; card = group['card']; host = 'radon' if card == 'radon0' else 'fourier'
    catalog_ready = read(root / 'plans/t45-final-20261030' / (host + '-sources.json'))
    ids = group['assigned_ids']
    require(set(ids) <= set(catalog_ready['items']), 'Source partition still being generated')
    items = {sid: catalog_ready['items'][sid] for sid in ids}; qa_bindings = []
    for sid, item in items.items():
        qa = read(item['source_qa_path'])
        require(qa['status'] == 'PASS_ALL180_RGB8_PTS_OPENCV' and qa['sample_id'] == sid
                and qa['frames'] == 180 and qa['width'] == 768 and qa['height'] == 512
                and len(qa['decoded_RGB8_frame_hashes']) == 180
                and qa['stream_sha256'] == qa['opencv_stream_sha256'] == item['ready']['stream_sha256']
                and qa['video_sha256'] == item['ready']['video_sha256'], 'Source CPU admission differs')
        require(qa['row']['card'] == card and qa['row']['domain'] == group['domain'], 'Source assignment differs')
        qa_bindings.append(descriptor(item['source_qa_path']))
    source_path = plans / (shard + '-sources.json')
    write_new(source_path, {'schema': 'immutable-T45-shard-sources-v1', 'items': items})
    acceptance_path = plans / (shard + '-source-acceptance.json')
    write_new(acceptance_path, {'status': 'PASS_FROZEN_SHARD_SOURCES', 'assigned_ids': ids,
        'catalog_sha256': file_sha256(source_path), 'source_QA': qa_bindings,
        'source_frames': 180 * len(ids), 'new_model_calls': 0})
    parent_path = root / 'plans/cosmos-x-full-20261028' / (card + '-profile.json')
    p = read(parent_path)
    runtime_path = Path(p['output']) / 'runtime.json'
    for key in ('implementation', 'observer_entry', 'worker_token', 'media_files', 'sources',
                'parent_validated_profile', 'scope_change_only', 'cold'):
        p.pop(key, None)
    p['engine_kwargs']['limit_mm_per_prompt'] = {'audio': 0, 'image': 1, 'video': 1}
    worker = shard + '-segment0'; control_path = plans / (shard + '-control.json')
    write_new(control_path, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
    p.update(worker=worker, shard=shard, card=card, phase=group['domain'], assigned_ids=ids,
        maximum_model_calls=len(ids), qualified_profile=descriptor(parent_path),
        qualified_runtime=descriptor(runtime_path), execution_plan=descriptor(plan_path),
        source_catalog=descriptor(source_path), source_acceptance=descriptor(acceptance_path),
        output=str(root / 'runs/reproducibility' / NAME / worker),
        claim_root=str(root / 'runs/reproducibility' / NAME / 'claims' / shard),
        pause_file=str(plans / (worker + '-PAUSE.json')), control_file=str(control_path),
        control_generation=0, protocol_name='Cosmos-X-T45-20261030',
        choices_by_task={'T4': ['LEFT', 'RIGHT'], 'T5': ['YES', 'NO']},
        automatic_retry=False, formal_auto_resume=False)
    draft = plans / (worker + '-draft.json'); write_new(draft, p)
    return seal(draft, plans / (worker + '-profile.json'))


def resume(profile, output):
    p, terminal, raw = closed(profile); science(p); control = read(p['control_file'])
    require(control['paused'], 'Explicit pause required')
    ids = lifecycle.remaining(p, terminal, raw); require(ids, 'No unattempted source remains')
    q = copy.deepcopy(p); generation = control['generation'] + 1
    for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper'):
        q.pop(key, None)
    q.update(worker=p['shard'] + '-segment' + str(generation), assigned_ids=ids,
             maximum_model_calls=len(ids), control_generation=generation,
             output=str(Path(p['output']).parent / (p['shard'] + '-segment' + str(generation))),
             pause_file=str(Path(output).with_suffix('.PAUSE.json')),
             continuation_parent=descriptor(profile),
             continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(q, stream); temp = stream.name
    try:
        answer = seal(temp, output)
    finally:
        Path(temp).unlink()
    control.update(generation=generation, paused=True, continuation_profile=descriptor(output))
    atomic_json(p['control_file'], control)
    return {**answer, 'status': 'SEALED_PAUSED_NOT_LAUNCHED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-shard', 'seal', 'activate', 'run', 'supervise', 'pause', 'prepare-resume', 'audit-closed'])
    for key in ('draft', 'output', 'profile', 'control', 'root', 'shard'):
        parser.add_argument('--' + key)
    args = parser.parse_args()
    if args.command == 'prepare-shard': answer = prepare_shard(args.root, args.shard)
    elif args.command == 'seal': answer = seal(args.draft, args.output)
    elif args.command == 'activate': answer = activate(args.profile)
    elif args.command == 'run': answer = run(args.profile)
    elif args.command == 'supervise': answer = supervise(args.profile)
    elif args.command == 'pause': answer = lifecycle.pause(args.control)
    elif args.command == 'prepare-resume': answer = resume(args.profile, args.output)
    else:
        _, result, raw = closed(args.profile)
        answer = {'status': 'CLOSED_PASS', 'worker_status': result['status'], 'completed': len(raw)}
    print(json.dumps(answer), flush=True)


if __name__ == '__main__':
    main()
