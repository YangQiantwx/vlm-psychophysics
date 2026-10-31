"""Finite classic-only Qwen X continuation, gated by the completed T2 point.

Scientific reader/processor/QSA/forward code is reused byte-for-byte. This module
only binds the new finite allocation, admission, request-boundary pause and clean
continuation. Preparation performs no model call or remote operation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile

# Capture the original lifecycle boundary before installing the X-QSA observers.
from . import nemo_x_full as lifecycle
from . import qwen_x_crosscard as qualified
from .official_x_results import Archive, common_raw_checks, journal, unique
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity

NAME = 'qwen-x-remaining-t13-20261030'
CARDS = ('fourier0', 'fourier1', 'radon0')
GROUPS = ('T3', 'T1')
ROOTS = qualified.ROOTS
qsa = qualified.qsa
base = qsa.base
MM = qualified.MM
COUNTS = {'T1': 3840, 'T3': 2736}
TOTAL = 6576
DRIVER = '595.99.02'
KERNEL_PREFIX = '7.0.0-42'
COMPARE_FIELDS = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason',
                  'actual_prompt_token_ids', 'truth', 'question', 'source_media')
SETTING_KEYS = ('model', 'base_model', 'checkpoint', 'checkpoint_assets', 'engine_kwargs',
                'expected_packages', 'cleared_environment', 'environment', 'python',
                'request_mm_processor_kwargs', 'vision_observation', 'minimum_kv_capacity',
                'torch_num_threads', 'gpu_uuid', 'physical_lock', 'choices_by_task')
FORBIDDEN = ('bootstrap', 'request_adapter', 'request_mm_processor_kwargs_by_task',
             'vision_validation_hook', 'media_io_kwargs')


def rebase(value, remote):
    if isinstance(value, str):
        for old in ROOTS.values():
            if value.startswith(old + '/'):
                return remote + value[len(old):]
        return value
    if isinstance(value, list):
        return [rebase(x, remote) for x in value]
    if isinstance(value, dict):
        return {k: rebase(v, remote) for k, v in value.items()}
    return value


def remote_ref(path, root, remote):
    return {'path': str(Path(remote) / Path(path).relative_to(root)), 'sha256': file_sha256(path)}


def implementation():
    qualified.fixed_implementation()
    from . import refinement_worker, gemma_speed, storage, inference, official_x_results
    from . import qwen_remaining_t13_results as results
    modules = (qualified, lifecycle, refinement_worker, gemma_speed, storage, inference, official_x_results, results)
    values = {str(Path(x.__file__).resolve().relative_to(Path(__file__).resolve().parents[1])):
              file_sha256(x.__file__) for x in modules}
    values.update({str(Path(path).resolve().relative_to(Path(__file__).resolve().parents[1])): sha
                   for path, sha in qualified.fixed_implementation().items()})
    values['classic_precision/qwen_remaining_t13.py'] = file_sha256(__file__)
    return values


def check_implementation(expected):
    require(implementation() == expected, 'Bound scientific/lifecycle implementation changed')


def validate_template(template, group):
    e = template['engine_kwargs']
    expected = {'max_model_len': 98304 if group == 'T1' else 65536,
        'kv_cache_memory_bytes': (8 if group == 'T1' else 4) * 1024**3,
        'max_num_batched_tokens': 49152, 'cpu_offload_gb': 24, 'cpu_offload_params': ['experts'],
        'engram_config': {'cpu_offload': True}, 'max_num_seqs': 1, 'tensor_parallel_size': 1,
        'enforce_eager': True, 'compilation_config': {'mode': 0}, 'enable_prefix_caching': False,
        'mm_processor_cache_gb': 0, 'enable_chunked_prefill': True, 'kv_cache_dtype': 'bfloat16',
        'quantization': 'modelopt', 'seed': 0, 'trust_remote_code': True,
        'disable_custom_all_reduce': True, 'gpu_memory_utilization': .78,
        'kernel_config': {'enable_flashinfer_autotune': False}, 'mm_processor_kwargs': MM,
        'worker_extension_cls': 'classic_precision.official_x_qsa_selector.XQSAWorkerExtension',
        'limit_mm_per_prompt': {'audio': 0, 'image': 180 if group == 'T1' else 1, 'video': 1}}
    require(set(e) == set(expected) | {'model'} and all(e[k] == v for k, v in expected.items()),
            'Exact tested engine configuration differs')
    require(e['model'] == template['checkpoint'] and Path(template['checkpoint']).name ==
            'fc694b54fb0174e0913e6adf86691ef85a4ead47', 'Official checkpoint differs')
    require(template['model'] == 'qwen_x_qsa' and template['base_model'] == 'qwen'
            and template['request_mm_processor_kwargs'] == MM
            and template['minimum_kv_capacity'] == expected['max_model_len'], 'Tested template differs')
    require(template['vision_observation'] == {'allow_official_fused_normalization': True,
        'feature_dim': 1536, 'normalized_paths': ['visual.patch_embed'], 'outer_paths': ['visual'],
        'patch_size': 16, 'rule': 'flat_patches', 'temporal_patch_size': 2}, 'Vision observation differs')


def validate_source(sid, item, task):
    row = item['row']; public = item['public_index']; media = item['content']
    require(item['sample_id'] == row['sample_id'] == public['sample_id'] == sid
            and item['task'] == row['task'] == public['task'] == task, 'Source/task identity differs')
    require(row['accuracy_eligible'] is True and row['choices'] == ['LEFT', 'RIGHT']
            and row['truth'] in ('LEFT', 'RIGHT'), 'Task GT/choices differ')
    require(len(media) == 2 and media[0]['type'] == 'video'
            and media[1] == {'type': 'text', 'text': row['prompt']}, 'Unexpected media/prompt content')
    require(item['video_contract'] == {'fps': 60, 'frames': 180, 'height': 512, 'width': 768},
            'Full-frame source contract differs')
    require(row['prompt'] == public['prompt'] and row['truth'] == public['truth']
            and row['source_parameters'] == public['nominal_parameters'], 'Scientific source binding differs')
    require(media[0]['sha256'] == item['ready']['video_sha256'] == public['video_sha256']
            and item['ready']['stream_sha256'] == item['source_stream_sha256'] == public['stream_sha256']
            and all(item[k] == public[k] for k in ('source_audit_identity', 'source_row_sha256', 'source_ready_sha256')),
            'Source/stream/QA binding differs')
    params = row['source_parameters']
    if task == 'T1':
        require(row['truth'] == params['direction'] and sid.startswith('t1rgb16_')
                and '/datasets/classic-t1-t3-v0.4/T1/' in media[0]['path'], 'Nonclassic T1 source')
    elif task == 'T3':
        require(row['truth'] == params['reference_side']
                and '/datasets/classic-t1-t3-v0.4/T3/' in media[0]['path'], 'Nonclassic T3 source')
    else:
        require(task == 'T2' and params['background_Y'] == .4 and params['delta_Y'] == .06
                and params['offset_frame'] - params['onset_frame'] == 30, 'Qualification point differs')


def authorized(p, resolve=verified):
    a = resolve(p['execution_authorization'])
    require(a['status'] == 'AUTHORIZED' and a['plan_sha256'] == p['experiment_plan']['sha256']
            and a['maximum_qualification_calls'] == 144 and a['maximum_full_calls'] == TOTAL
            and a['cards'] == list(CARDS) and set(a['phases']) == {'qualification', 'full'}
            and isinstance(a['user_message'], str) and a['user_message'].strip(), 'Bound human authorization differs')


def science(p, resolve=verified):
    check_implementation(p['implementation_sources'])
    require(p['phase'] in ('qualification', 'full') and p['card'] in CARDS
            and p['resource_group'] in GROUPS, 'Foreign phase/card/resource group')
    require(all(k not in p for k in FORBIDDEN) and p['sources'] == {}
            and not p['automatic_retry'] and not p['formal_auto_resume'], 'Unexpected alternate input/retry entry')
    template = resolve(p['runtime_template']); validate_template(template, p['resource_group'])
    require(all(p.get(k) == template.get(k) for k in SETTING_KEYS), 'Tested runtime settings changed')
    plan = resolve(p['experiment_plan']); authorized(p, resolve)
    require(plan['schema'] == NAME + '/v1' and plan['full_counts'] == COUNTS
            and plan['maximum_full_calls'] == TOTAL and plan['maximum_qualification_calls'] == 144
            and plan['implementation_sources'] == p['implementation_sources'] and plan['driver'] == DRIVER,
            'Finite classic-only plan differs')
    group = p['resource_group']; phase = p['phase']; card = p['card']
    expected = plan['qualification_ids'] if phase == 'qualification' else plan['cards'][card]['groups'][group]
    require(phase != 'qualification' or group == 'T3', 'Qualification must use tested 65k/4GiB')
    ids = p['assigned_ids']
    require(len(ids) == len(set(ids)) == p['maximum_model_calls'] and ids, 'Empty/duplicate allocation')
    require(p['source_catalog'] == plan['cards'][card]['source_catalog'], 'Catalog not bound to allocation')
    require(p['reference_panel']['sha256'] == plan['reference_panel']['sha256']
            and p['driver'] == DRIVER, 'Reference/driver contract differs')
    if p.get('continuation_parent'):
        require(phase == 'full', 'Qualification cannot automatically resume')
        parent = resolve(p['continuation_parent'])
        require(p['control_generation'] == parent['control_generation'] + 1,
                'Continuation generation differs')
        require(p['assigned_ids'] == parent['assigned_ids'][p['retained_prefix_length']:]
                and p['retained_prefix_length'] >= 0, 'Continuation is not exact remaining suffix')
        operational = {'implementation', 'operational_wrapper', 'assigned_ids', 'maximum_model_calls', 'media_files', 'worker_token', 'worker', 'output',
            'pause_file', 'control_generation', 'continuation_parent', 'continuation_wait', 'retained_prefix_length'}
        require(all(parent.get(k) == p.get(k) for k in parent.keys() | p.keys() if k not in operational),
                'Continuation changed scientific configuration')
    else:
        require(ids == expected and p['control_generation'] == 0, 'Initial allocation changed')
    require(set(ids) <= set(expected), 'Foreign task or ID')
    root = Path(p['workspace_root']); namespace = phase + '-' + group
    require(Path(p['output']).parent == root / 'runs/reproducibility' / NAME
            and Path(p['claim_root']) == root / 'runs/reproducibility' / NAME / 'claims' / card / namespace
            and Path(p['control_file']) == root / 'plans' / NAME / (card + '-' + namespace + '-control.json'),
            'Foreign output/claim/control namespace')
    canonical = resolve(p['canonical_catalog'])['items']; catalog = resolve(p['source_catalog'])['items']
    require(p['canonical_catalog']['sha256'] == plan['canonical_catalog_sha256'], 'Canonical catalog differs')
    for sid in ids:
        require(catalog[sid] == rebase(canonical[sid], p['workspace_root']), 'Source changed from canonical')
        validate_source(sid, catalog[sid], 'T2' if phase == 'qualification' else group)


class Boundary(lifecycle.Boundary):
    pass


def seal(draft, output):
    p = read(draft); science(p)
    p['operational_wrapper'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temp = f.name
    try:
        return qsa.seal(temp, output)
    finally:
        Path(temp).unlink()


def run(profile):
    p = read(profile); science(p)
    require(file_sha256(__file__) == p['operational_wrapper']['sha256'], 'Execution wrapper changed')
    for key, path in (('qsa_entry', qsa.__file__), ('observer_entry', qsa.v2.__file__), ('selector_source', qsa.selector.__file__)):
        require(file_sha256(path) == p[key]['sha256'], 'Frozen QSA/observer changed')
    if p['phase'] == 'full':
        from .qwen_remaining_t13_results import validate_gate
        validate_gate(p, verified)
    driver = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,driver_version', '--format=csv,noheader'], text=True)
    require(any(uuid.strip().lower().removeprefix('gpu-') == p['gpu_uuid'].lower().removeprefix('gpu-')
                and version.strip() == DRIVER for uuid, version in (line.split(',') for line in driver.splitlines())),
            'Actual physical GPU driver differs')
    require(os.uname().release.startswith(KERNEL_PREFIX), 'Actual upgraded host kernel differs')
    write_new(Path(p['output']).with_suffix('.environment.json'), {'driver_query': driver,
        'kernel_release': os.uname().release, 'profile_sha256': digest(p), 'owner': owner_identity()})
    base.worker_identity = qualified.worker_identity; base.Boundary = Boundary
    return qsa.v2.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = native


def closed(profile):
    p, result, raw = lifecycle.closed(profile)
    science(p)
    require(result['completed_ids'] == p['assigned_ids'][:len(raw)]
            and read(Path(p['output']).with_suffix('.wait.json'))['completed_ids'] == list(raw), 'Not clean ordered prefix')
    if raw:
        from .qwen_remaining_t13_results import audit_segment
        audit_segment(p['workspace_root'], profile)
    return p, result, raw


def fresh(p):
    require(not Path(p['output']).exists() and not Path(p['output']).with_suffix('.launch.json').exists()
            and not Path(p['output']).with_suffix('.wait.json').exists(), 'Attempted worker cannot activate')
    require(not any((Path(p['claim_root']) / (s + '.json')).exists() for s in p['assigned_ids']),
            'Existing claim; never reissue')


def activate(queue_path):
    q = read(queue_path); require(q['status'] == 'SEALED', 'Unsealed queue')
    with lifecycle.controller_lock(q['controller_lock']):
        require(not Path(q['state_path']).exists(), 'Attempted queue cannot reactivate')
        for ref in q['profiles']:
            p = verified(ref); science(p); fresh(p)
            require(read(p['control_file'])['generation'] == p['control_generation'], 'Stale generation')
            if p.get('continuation_parent'):
                _, result, raw = closed(p['continuation_parent']['path'])
                require(p['assigned_ids'] == lifecycle.remaining(verified(p['continuation_parent']), result, raw),
                        'Continuation parent no longer clean')
            if p['phase'] == 'full':
                from .qwen_remaining_t13_results import validate_gate
                validate_gate(p, verified)
        for ref in q['profiles']:
            p = verified(ref); c = read(p['control_file']); c.update(paused=False, activated_queue=descriptor(queue_path))
            atomic_json(p['control_file'], c)
    return {'status': 'ACTIVATED_NOT_LAUNCHED', 'model_calls': 0}


def queue(queue_path):
    q = read(queue_path)
    require(q['status'] == 'SEALED' and q['automatic_retry'] is False, 'Wrong finite queue')
    with lifecycle.controller_lock(q['controller_lock']):
        state_path = Path(q['state_path']); require(not state_path.exists(), 'Queue already attempted')
        state = {'status': 'RUNNING', 'owner': owner_identity(), 'plan': descriptor(queue_path), 'workers': []}
        atomic_json(state_path, state)
        try:
            for ref in q['profiles']:
                p = verified(ref); science(p); c = read(p['control_file'])
                if c['paused'] or c['generation'] != p['control_generation']:
                    state['status'] = 'PAUSED_BEFORE_SUCCESSOR'; break
                fresh(p); env = dict(os.environ)
                for k in p['cleared_environment']: env.pop(k, None)
                env.update(p['environment']); state['current_profile'] = ref; atomic_json(state_path, state)
                child = subprocess.Popen([p['python'], '-B', '-m', __spec__.name, 'supervise', '--profile', ref['path']],
                                         env=env, stdin=subprocess.DEVNULL)
                code = child.wait()
                record = {'profile': ref, 'supervisor_pid': child.pid, 'real_supervisor_wait': True, 'returncode': code}
                state['workers'].append(record); require(code == 0, 'Supervisor failed; no retry/successor')
                _, result, _ = closed(ref['path']); record['terminal_status'] = result['status']
                atomic_json(state_path, state)
                if result['status'] != 'COMPLETE': state['status'] = 'PAUSED'; break
            else:
                state['status'] = 'COMPLETE'
        except BaseException as error:
            state.update(status='HELD_NO_RETRY', error_type=type(error).__name__, error=str(error)); raise
        finally:
            atomic_json(state_path, state)
    return {'status': state['status'], 'workers': len(state['workers'])}


def prepare_resume(profile, output):
    p = read(profile)
    with lifecycle.controller_lock(str(Path(p['control_file']).with_suffix('.lock'))):
        p, result, raw = closed(profile); c = read(p['control_file'])
        require(p['phase'] == 'full' and c['paused'] and c['generation'] == p['control_generation'],
                'Only unbranched explicitly paused full worker can continue')
        ids = lifecycle.remaining(p, result, raw); require(ids, 'Nothing remains')
        require(not any((Path(p['claim_root']) / (s + '.json')).exists() for s in ids), 'Unresolved claim in suffix')
        generation = c['generation'] + 1; worker = p['card'] + '-full-' + p['resource_group'] + '-segment' + str(generation)
        q = copy.deepcopy(p)
        for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper'): q.pop(key, None)
        q.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
            output=str(Path(p['output']).parent / worker), pause_file=str(Path(output).with_suffix('.PAUSE.json')),
            continuation_parent=descriptor(profile), continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')),
            retained_prefix_length=len(raw))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(q, f); temp = f.name
        try: answer = seal(temp, output)
        finally: Path(temp).unlink()
        queue_path = Path(output).with_name(worker + '-queue.json')
        write_queue(queue_path, [descriptor(output)], p['workspace_root'])
        c.update(generation=generation, paused=True, continuation_profile=descriptor(output)); atomic_json(p['control_file'], c)
    return {**answer, 'status': 'CONTINUATION_PREPARED_PAUSED', 'queue': descriptor(queue_path), 'model_calls': 0}


def prepare_resume_queue(queue_path, output):
    previous = read(queue_path); state = read(previous['state_path'])
    require(state['status'] in ('PAUSED', 'PAUSED_BEFORE_SUCCESSOR'), 'Only a clean paused queue can continue')
    require(not Path(output).exists(), 'Continuation queue exists')
    profiles = []
    with lifecycle.controller_lock(previous['controller_lock']):
        for ref in previous['profiles']:
            p = verified(ref); science(p)
            require(p['phase'] == 'full' and read(p['control_file'])['paused'], 'Pause every resource group before continuation')
            if Path(p['output']).exists():
                _, terminal, _ = closed(ref['path'])
                if terminal['status'] == 'COMPLETE': continue
                generation = p['control_generation'] + 1
                target = Path(ref['path']).with_name(p['card'] + '-full-' + p['resource_group'] + '-segment' + str(generation) + '-profile.json')
                resumed = prepare_resume(ref['path'], target)
                profiles.extend(verified(resumed['queue'])['profiles'])
            else:
                fresh(p); profiles.append(ref)
        require(profiles, 'Nothing remains')
        write_queue(output, profiles, read(profiles[0]['path'])['workspace_root'])
    return {'status': 'CONTINUATION_QUEUE_PREPARED_PAUSED', 'queue': descriptor(output), 'model_calls': 0}


def write_queue(path, profiles, root):
    write_new(path, {'status': 'SEALED', 'automatic_retry': False, 'profiles': profiles,
        'state_path': str(Path(root) / 'runs/reproducibility' / NAME / (Path(path).stem + '-state.json')),
        'controller_lock': str(Path(path).with_suffix('.lock'))})


def stage(plan_path, card, phase, authorization):
    plan = read(plan_path)
    require(plan['schema'] == NAME + '/v1' and card in CARDS and phase in ('qualification', 'full'), 'Wrong preparation')
    check_implementation(plan['implementation_sources']); profiles = []
    for ref in plan['profiles'][card][phase]:
        p = verified(ref); require(p['hostname'] == socket.gethostname(), 'Wrong staging host')
        p.update(experiment_plan=descriptor(plan_path), execution_authorization=descriptor(authorization))
        p.pop('status', None)
        if phase == 'full':
            gate = Path(p['workspace_root']) / 'runs/reproducibility' / NAME / 'qualification-summary.json'
            p['qualification_gate'] = descriptor(gate)
            from .qwen_remaining_t13_results import validate_gate
            validate_gate(p, verified)
        target = ref['path'].replace('-draft.json', '-profile.json')
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(p, f); temp = f.name
        try: seal(temp, target)
        finally: Path(temp).unlink()
        profiles.append(descriptor(target))
    remote = read(profiles[0]['path'])['workspace_root']
    target = Path(remote) / 'plans' / NAME / (card + '-' + phase + '-queue.json')
    write_queue(target, profiles, remote)
    return {'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(target), 'model_calls': 0}


def panel_reference(root, ids, items):
    """Extract full historical prompt/output tokens, bound to accepted raw/PASS/claims."""
    arc = Archive(root); run = Path(root) / 'runs/reproducibility/qwen-x-classic-t2-20261029'
    summary = read(run / 'summary.json')
    require(summary['status'] == 'COMPLETE_CLASSIC_QWEN_T2_X' and summary['overall']['n'] == 2880,
            'Classic T2 formal reference is not accepted')
    records = {}; proofs = []
    for proof in summary['worker_evidence']:
        p = arc.ref(proof['profile']); out = arc.path(p['output'])
        for key in ('raw', 'wait'): arcpath = arc.path(proof[key]['path']); require(file_sha256(arcpath) == proof[key]['sha256'], 'Historical evidence changed')
        wait = read(out.with_suffix('.wait.json'))
        require(wait['returncode'] == 0 and wait['terminal_status'] == 'COMPLETE' and wait['real_wait_observed']
                and wait['owned_engines_exited'] and wait['original_physical_lease_released']
                and wait['profile_sha256'] == digest(p), 'Historical reference lifecycle differs')
        selected = set(ids) & set(p['assigned_ids']); passes = unique([e for e in journal(out / 'events.jsonl') if e['event'] == 'TECHNICAL_PASS'])
        catalog = arc.ref(p['source_catalog'])['items']
        with (out / 'raw.jsonl').open() as f:
            for line in f:
                raw = json.loads(line); sid = raw['sample_id']
                if sid not in selected: continue
                common_raw_checks(p, catalog[sid], raw, p['assigned_ids'].index(sid))
                require(sid not in records and passes[sid]['raw_digest'] == digest(raw), 'Historical duplicate/unqualified raw')
                claim_path = arc.path(p['claim_root']) / (sid + '.json'); claim = read(claim_path)
                require(claim['profile_sha256'] == digest(p) and claim['worker_token'] == p['worker_token'], 'Historical claim differs')
                require(raw['source']['media'][0]['sha256'] == items[sid]['content'][0]['sha256']
                        and raw['truth'] == items[sid]['row']['truth'] and raw['question'] == items[sid]['row']['prompt'],
                        'Historical/canonical source or GT differs')
                records[sid] = {k: raw[k] for k in COMPARE_FIELDS if k != 'source_media'}
                records[sid]['source_media'] = [[m['type'], m['sha256']] for m in raw['source']['media']]
                records[sid]['raw_digest'] = digest(raw)
        proofs.append({k: proof[k] for k in ('profile', 'raw', 'wait')})
    require(set(records) == set(ids), 'Historical qualification point incomplete')
    return {'status': 'SEALED_EXISTING_FORMAL_X48', 'ids': ids, 'records': records,
            'summary': descriptor(run / 'summary.json'), 'historical_workers': proofs,
            'comparison_fields': list(COMPARE_FIELDS), 'new_model_calls': 0}


def allocate(items):
    cells = defaultdict(dict)
    for sid, item in items.items():
        t = item['task']; p = item['row']['source_parameters']
        key = (0, p['angle_deg'], p['tilt_sign']) if t == 'T3' else (1, p['sf_cycles_per_aperture'], p['modulation_depth'])
        pair = p['onset_index'] if t == 'T3' else p['bank_start_index']
        member = (pair, item['row']['truth']); require(member not in cells[key], 'Duplicate LR cell member')
        cells[key][member] = sid
    require(len(cells) == 137 and all(len(v) == 48 for v in cells.values()), 'Classic condition grid incomplete')
    assigned = {card: {g: [] for g in GROUPS} for card in CARDS}; special = 0
    for ci, (key, members) in enumerate(sorted(cells.items())):
        pairs = sorted({k[0] for k in members}); require(len(pairs) == 24, 'Missing LR pairs')
        pairs = pairs[(7 * ci) % 24:] + pairs[:(7 * ci) % 24]
        extra = ((ci + 1) * 66 // 137) > (ci * 66 // 137)
        n0, n1 = 9, 9
        if extra:
            if special % 2: n1 = 8
            else: n0 = 8
            special += 1
        chunks = (pairs[:n0], pairs[n0:n0+n1], pairs[n0+n1:]); group = 'T3' if key[0] == 0 else 'T1'
        for card, selected in zip(CARDS, chunks):
            assigned[card][group].extend(members[i, side] for i in selected for side in ('LEFT', 'RIGHT'))
    require([sum(map(len, assigned[c].values())) for c in CARDS] == [2400, 2400, 1776], 'Balanced allocation differs')
    require(len({s for groups in assigned.values() for ids in groups.values() for s in ids}) == TOTAL, 'Allocation duplicate/missing ID')
    return assigned


def prepare(root):
    root = Path(root).resolve(); plans = root / 'plans' / NAME
    require(not plans.exists(), 'Prepared package exists; do not overwrite')
    arc = Archive(root); accepted = read(root / 'plans/nemo-x-full-20261029/full-allocation.json'); items = {}; bindings = []
    require(accepted['task_counts'] == {'T1': 3840, 'T2': 2880, 'T3': 2736}, 'Accepted source universe differs')
    for card in ('fourier0', 'radon0'):
        ref = accepted['cards'][card]['source_catalog']; bindings.append(ref)
        for sid, item in arc.ref(ref)['items'].items():
            if item['task'] not in COUNTS: continue
            require(sid not in items, 'Repeated canonical source ID'); validate_source(sid, item, item['task']); items[sid] = item
    require(Counter(i['task'] for i in items.values()) == COUNTS, 'Missing classic T1/T3 sources')
    t2 = read(root / 'plans/qwen-x-classic-t2-20261029/canonical-sources.json')['items']
    qualification = {s: x for s, x in t2.items() if x['row']['source_parameters']['background_Y'] == .4
                     and x['row']['source_parameters']['delta_Y'] == .06}
    ids = sorted(qualification, key=lambda s: (qualification[s]['row']['source_parameters']['onset_index'], qualification[s]['row']['truth']))
    require(len(ids) == 48 and Counter(x['row']['truth'] for x in qualification.values()) == {'LEFT': 24, 'RIGHT': 24}, 'Qualification LR grid differs')
    for s, x in qualification.items(): validate_source(s, x, 'T2')
    reference = panel_reference(root, ids, qualification); assigned = allocate(items)
    canonical = {**items, **qualification}; plans.mkdir(parents=True)
    write_new(plans / 'canonical-sources.json', {'items': canonical, 'accepted_source_catalogs': bindings})
    write_new(plans / 'reference-panel.json', reference)
    sources = implementation(); cards = {}; profiles = {c: {'qualification': [], 'full': []} for c in CARDS}
    old_t1_path = root / 'plans/official-x-20261028/qwen-xqsa-noautotune-t1-video-cold1-profile.json'
    old_t1 = read(old_t1_path); validate_template(old_t1, 'T1')
    old_out = arc.path(old_t1['output']); old_runtime = read(old_out / 'runtime.json'); old_public = read(old_out / 'public-settings.json')
    require(all(r['actual_kv_capacity'] >= 98304 for r in old_runtime)
            and old_public['LLM_kwargs']['max_model_len'] == 98304
            and old_public['LLM_kwargs']['kv_cache_memory_bytes'] == 8 * 1024**3, 'T1 tested actual resource evidence differs')
    for card in CARDS:
        host = 'radon' if card == 'radon0' else 'fourier'; remote = ROOTS[host]
        native_path = root / 'plans/qwen-x-crosscard-t23-20261029' / (card + '-t23-cold1-profile.json')
        native = read(native_path); validate_template(native, 'T3')
        selected = {s: rebase(canonical[s], remote) for s in ids + assigned[card]['T3'] + assigned[card]['T1']}
        catalog = plans / (card + '-sources.json'); write_new(catalog, {'items': selected})
        (plans / (card + '-video-files.txt')).write_text(''.join(str(Path(x['content'][0]['path']).relative_to(remote)) + '\n' for x in selected.values()))
        cards[card] = {'groups': assigned[card], 'count': sum(map(len, assigned[card].values())), 'source_catalog': remote_ref(catalog, root, remote)}
        for group in GROUPS:
            template = {k: copy.deepcopy(native[k]) for k in SETTING_KEYS if k in native}
            if group == 'T1':
                template['engine_kwargs'] = rebase(copy.deepcopy(old_t1['engine_kwargs']), remote)
                template['engine_kwargs']['model'] = native['checkpoint']
                template['minimum_kv_capacity'] = 98304
                template['choices_by_task'] = copy.deepcopy(old_t1['choices_by_task'])
            validate_template(template, group)
            template_path = plans / (card + '-' + group + '-runtime-template.json'); write_new(template_path, template)
            for phase in ('qualification', 'full'):
                if phase == 'qualification' and group != 'T3': continue
                namespace = phase + '-' + group; worker = card + '-' + namespace + '-segment0'
                control = plans / (card + '-' + namespace + '-control.json')
                write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
                p = copy.deepcopy(template); selected_ids = ids if phase == 'qualification' else assigned[card][group]
                p.update(worker=worker, card=card, phase=phase, resource_group=group, workspace_root=remote,
                    hostname=native['hostname'], assigned_ids=selected_ids, maximum_model_calls=len(selected_ids), sources={},
                    source_catalog=cards[card]['source_catalog'], canonical_catalog=remote_ref(plans / 'canonical-sources.json', root, remote),
                    runtime_template=remote_ref(template_path, root, remote), reference_panel=remote_ref(plans / 'reference-panel.json', root, remote),
                    implementation_sources=sources, driver=DRIVER, resources={'min_available_ram_gib': 64, 'min_free_gpu_gib': 72},
                    output=str(Path(remote) / 'runs/reproducibility' / NAME / worker),
                    claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / 'claims' / card / namespace),
                    control_file=str(Path(remote) / 'plans' / NAME / control.name), control_generation=0,
                    pause_file=str(Path(remote) / 'plans' / NAME / (worker + '-PAUSE.json')),
                    automatic_retry=False, formal_auto_resume=False, status='PREPARED_NOT_LAUNCHED')
                target = plans / (worker + '-draft.json'); write_new(target, p)
                profiles[card][phase].append(remote_ref(target, root, remote))
    plan = {'schema': NAME + '/v1', 'status': 'PREPARED_NOT_LAUNCHED', 'maximum_qualification_calls': 144,
        'maximum_full_calls': TOTAL, 'full_counts': COUNTS, 'cards': cards, 'profiles': profiles,
        'qualification_ids': ids, 'reference_panel': remote_ref(plans / 'reference-panel.json', root, ROOTS['fourier']),
        'canonical_catalog_sha256': file_sha256(plans / 'canonical-sources.json'), 'implementation_sources': sources,
        'driver': DRIVER, 'resource_group_order': list(GROUPS), 'automatic_retry': False,
        'allocation_rule': 'Every one of 137 cells has 24 whole LR pairs. Rotated pair order; 71 cells split9/9/6, 66 cells alternate8/9/7 and9/8/7. No model answer is consulted.',
        'source_catalogs': bindings, 'T1_tested_profile': descriptor(old_t1_path),
        'T1_actual_runtime': descriptor(old_out / 'runtime.json'), 'T1_actual_public_settings': descriptor(old_out / 'public-settings.json'),
        'scope': 'Classic T1/T3 only. T2 point is qualification only; natural tasks remain paused.'}
    write_new(plans / 'plan.json', plan)
    return {'status': plan['status'], 'plan': descriptor(plans / 'plan.json'), 'counts': COUNTS,
            'cards': {c: cards[c]['count'] for c in CARDS}, 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'stage', 'seal', 'activate', 'queue', 'run', 'supervise', 'pause', 'prepare-resume', 'prepare-resume-queue', 'audit-closed'))
    for key in ('root', 'plan', 'card', 'phase', 'authorization', 'draft', 'output', 'profile', 'control'): parser.add_argument('--' + key)
    a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.root)
    elif a.command == 'stage': result = stage(a.plan, a.card, a.phase, a.authorization)
    elif a.command == 'seal': result = seal(a.draft, a.output)
    elif a.command == 'activate': result = activate(a.plan)
    elif a.command == 'queue': result = queue(a.plan)
    elif a.command == 'run': result = run(a.profile)
    elif a.command == 'supervise': result = supervise(a.profile)
    elif a.command == 'pause': result = lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': result = prepare_resume(a.profile, a.output)
    elif a.command == 'prepare-resume-queue': result = prepare_resume_queue(a.plan, a.output)
    else:
        p, terminal, raw = closed(a.profile); result = {'status': 'CLOSED_PASS', 'worker_status': terminal['status'], 'completed': len(raw)}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
