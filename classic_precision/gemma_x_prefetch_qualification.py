"""Finite official-prefetch qualification of the unclaimed Radon192 remainder.

The closed UVA prefix is retained separately. Only never-claimed cold1 inputs
and the untouched96 cold2 inputs may run; no retry or formal launch is implied.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import socket

from . import gemma_x_full as full
from . import gemma_x_qualification as qualification
from . import gemma_x_provisional as provisional
from . import gemma_x_prefetch_placement as placement
from . import official_x_runner as base
from . import refinement_worker as worker
from .refinement_worker import read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

NAME = 'gemma-x-prefetch-qualification-20261031'
MODULE = 'classic_precision.gemma_x_prefetch_qualification'
PASS = 'PASS_GEMMA_X_THREE_CARDS_PREFETCH_FINITE_POINT'
ADMISSION = 'PASS_CLOSED_UVA_PREFIX_PREFETCH_REMAINDER_ADMISSION'
OUTPUT_KEYS = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason', 'truth', 'question')
PREFETCH_FILES = ('raw.jsonl','events.jsonl','request-timings.jsonl','deep-observations.jsonl',
    'input-admissions.jsonl','runtime.json','public-settings.json','identity.json','result.json',
    'versions.json','load.json','prefetch-equivalence.jsonl')
PREFIX_FILES = ('raw.jsonl','events.jsonl','request-timings.jsonl','deep-observations.jsonl',
    'input-admissions.jsonl','runtime.json','result.json')
EXTRA_FIELDS = {'prefetch_adapter', 'prefetch_placement_helper', 'closed_uva_admission',
                'prefetch_deployment', 'original_qualification_profile', 'fourier_output_reference',
                'fourier_gate_helper'}
_SCIENCE, _VERIFY, _SEAL = qualification.science, qualification.verify_code, qualification.seal
_SUPERVISE = worker.supervise


def descriptor(path):
    """Keep canonical host aliases even when their physical target is an SSD."""
    return {'path': os.path.abspath(os.fspath(path)), 'sha256': file_sha256(path)}


def _rows(path):
    path = Path(path)
    if not path.exists(): return []
    text = path.read_text()
    require(not text or text.endswith('\n'), 'Incomplete journal tail: ' + str(path))
    return [json.loads(line) for line in text.splitlines() if line]


def _unique(rows, key='sample_id'):
    values = {x[key]: x for x in rows}
    require(len(values) == len(rows), 'Duplicate journal identity: ' + key)
    return values


def _bound(ref, path, arc):
    require(arc.path(ref['path']) == path and file_sha256(path) == ref['sha256'],
            'Closed admission descriptor differs: ' + str(path))
    return read(path)


def closed_admission(root, admission_ref):
    """Re-read claims, raw, PASS, result and actual wait; never trust ID counts."""
    arc = full.Archive(root); a = arc.ref(admission_ref); q = arc.ref(a['old_queue'])
    require(a['status'] == ADMISSION and a['original_radon_budget'] == 192
            and isinstance(a.get('authorization_user_message'), str)
            and a['authorization_user_message'].strip()
            and len(q['profiles']) == len(a['workers']) == 2 and q['automatic_retry'] is False,
            'Missing closed finite192-call admission or authorization')
    parents = {p['cold']: (p, ref) for ref in q['profiles'] for p in [arc.ref(ref)]}
    declared = {w['cold']: w for w in a['workers']}
    require(set(parents) == set(declared) == {1, 2}, 'Duplicate/missing old cold worker')
    outer = arc.ref(a['outer_wait'])
    require(outer['returncode'] == 1 and outer['real_wait_observed'],
            'Old COMPLETE-only queue must actually exit with the preserved rc1')
    first = parents[1][0]; ids = first['assigned_ids']
    require(len(ids) == len(set(ids)) == 96, 'Original Radon point is not96 unique inputs')
    for cold, (p, ref) in parents.items():
        require(declared[cold]['profile'] == ref and p['card'] == 'radon0'
                and p['assigned_ids'] == ids and p['maximum_model_calls'] == 96
                and p['cold'] == cold and p['automatic_retry'] is False
                and p['qualification_selection'] == first['qualification_selection'], 'Old profile identity/scope differs')
        control = read(arc.path(p['control_file']))
        require(control['paused'] is True and control['generation'] >= p['control_generation'],
                'Old control is active or stale')
    p = first; out = arc.path(p['output']); d = declared[1]
    result = _bound(d['result'], out / 'result.json', arc)
    wait = _bound(d['wait'], out.with_suffix('.wait.json'), arc)
    completed = result['completed_ids']; remaining = ids[len(completed):]
    require(completed and len(completed) < 96 and completed == ids[:len(completed)]
            and result['status'] == wait['terminal_status'] == 'PAUSED'
            and result['UNCERTAIN_ids'] == [] and result['unattempted_ids'] == remaining
            and not (out / 'failure.json').exists()
            and wait['returncode'] == 0 and wait['real_wait_observed']
            and wait['owned_engines_exited'] and wait['original_physical_lease_released']
            and wait['profile_sha256'] == digest(p) and wait['completed_ids'] == completed,
            'UVA prefix has an unresolved attempt or lacks actual clean closure')
    raw = _unique(_rows(out / 'raw.jsonl'))
    events = _rows(out / 'events.jsonl')
    intents = _unique([x for x in events if x['event'] == 'INTENT'])
    passes = _unique([x for x in events if x['event'] == 'TECHNICAL_PASS'])
    timings = _unique(_rows(out / 'request-timings.jsonl'))
    claim_dir = arc.path(p['claim_root']); claims = {x.stem: read(x) for x in claim_dir.glob('*.json')}
    require(list(raw) == completed and set(raw) == set(intents) == set(passes)
            == set(timings) == set(claims) == set(completed),
            'Claimed/UNCERTAIN/duplicate input cannot enter the prefetch remainder')
    for sid, value in raw.items():
        claim = claims[sid]
        require(value['profile_sha256'] == digest(p) and passes[sid]['raw_digest'] == digest(value)
                and claim['sample_id'] == sid and claim['profile_sha256'] == digest(p)
                and claim['worker_token'] == p['worker_token'], 'UVA raw/PASS/claim revision differs')
    for key in ('completed_ids', 'claimed_ids'):
        if key in d: require(d[key] == completed, 'Declared UVA prefix IDs differ')
    cold2 = parents[2][0]; second = arc.path(cold2['output'])
    require(declared[2].get('never_launched') is True and not second.exists()
            and not second.with_suffix('.launch.json').exists()
            and not second.with_suffix('.wait.json').exists()
            and not list(arc.path(cold2['claim_root']).glob('*.json')),
            'Old cold2 has already launched or acquired a claim')
    return {'admission': a, 'parents': parents, 'assigned_by_cold': {1: remaining, 2: ids},
            'completed_prefix_ids': completed, 'remaining_model_calls': 192 - len(completed),
            'prefix_raw': raw}


def science(p, resolve=verified):
    original = resolve(p['original_qualification_profile'])
    deployment = resolve(p['prefetch_deployment'])
    admission = resolve(p['closed_uva_admission'])
    expected_ids = deployment['assigned_by_cold'][str(p['cold'])]
    require(p['card'] == original['card'] == 'radon0' and p['cold'] == original['cold'] in (1, 2)
            and p['assigned_ids'] == expected_ids and len(expected_ids) == len(set(expected_ids))
            == p['maximum_model_calls'] and expected_ids,
            'Prefetch allocation is not the exact never-claimed finite remainder')
    require(deployment['status'] == 'AUTHORIZED_PREFETCH_REMAINDER_ONLY'
            and deployment['original_radon_budget'] == 192
            and admission['status'] == ADMISSION
            and deployment['closed_uva_admission'] == p['closed_uva_admission']
            and deployment['original_qualification_profiles'][str(p['cold'])]
                == p['original_qualification_profile']
            and deployment['engine_delta'] == placement.ENGINE_DELTA
            and deployment['resource_budget'] == placement.RESOURCES
            and deployment['automatic_retry'] is False
            and deployment['formal_launch_authorized'] is False,
            'Missing bound prefetch deployment authorization')
    done = deployment['completed_uva_prefix_ids']; original_ids = original['assigned_ids']
    require(done and done == original_ids[:len(done)] and len(original_ids) == 96
            and deployment['assigned_by_cold'] == {'1': original_ids[len(done):], '2': original_ids}
            and deployment['remaining_model_calls'] + len(done) == 192,
            'Prefetch allocation duplicates completed or uncertain original IDs')
    archive = getattr(resolve, '__self__', None)
    closed = closed_admission(archive.root if archive is not None else p['workspace_root'], p['closed_uva_admission'])
    require(expected_ids == closed['assigned_by_cold'][p['cold']]
            and done == closed['completed_prefix_ids']
            and p['original_qualification_profile'] == closed['parents'][p['cold']][0]['original_qualification_profile'],
            'Closed admission does not authorize this exact never-claimed profile revision')
    e = p['engine_kwargs']
    require(all(e.get(k) == v for k, v in placement.ENGINE_DELTA.items())
            and p['resources'] == placement.RESOURCES and p['environment'] == original['environment']
            and not p.get('bootstrap') and p['automatic_retry'] is False
            and p['formal_auto_resume'] is False and p['minimum_kv_capacity'] == 65536,
            'Prefetch engine/resources/environment/context changed')
    require(not any(k in p['environment'] for k in (
        'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA', 'VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY')),
        'Official pinned prefetch behavior must remain enabled')
    normalized = copy.deepcopy(p)
    for key in EXTRA_FIELDS: normalized.pop(key, None)
    normalized['assigned_ids'] = original_ids; normalized['maximum_model_calls'] = 96
    normalized['resources'] = original['resources']
    for key in placement.ENGINE_DELTA:
        if key in original['engine_kwargs']: normalized['engine_kwargs'][key] = original['engine_kwargs'][key]
        else: normalized['engine_kwargs'].pop(key, None)
    require(full.frozen_settings(normalized) == full.frozen_settings(original),
            'A scientific/input/runtime setting outside the explicit prefetch delta changed')
    _SCIENCE(normalized, resolve)
    reference = resolve(p['fourier_output_reference'])
    require(reference['status'] == 'PASS_FOURIER_FOUR_COLD_OUTPUT_REFERENCE'
            and reference['selection_sha256'] == p['qualification_selection']['sha256']
            and reference['assigned_ids'] == original_ids, 'Full Fourier reference differs')
    verify_extra_code(p)


def verify_extra_code(p):
    require(p['prefetch_adapter']['sha256'] == file_sha256(__file__)
            and p['prefetch_placement_helper']['sha256'] == file_sha256(placement.__file__)
            and p['fourier_gate_helper']['sha256'] == file_sha256(provisional.__file__),
            'Sealed prefetch lifecycle or placement helper changed')


def verify_code(p):
    _VERIFY(p); verify_extra_code(p)


@contextmanager
def adaptation():
    changes = [(qualification, 'science', science), (qualification, 'verify_code', verify_code),
               (qualification, 'supervise', supervise), (qualification, 'NAME', NAME),
               (full, 'worker_identity', placement.worker_identity), (base, 'read_probe', placement.read_probe)]
    prior = [(obj, key, getattr(obj, key)) for obj, key, _ in changes]
    identity, boundary = base.worker_identity, base.Boundary
    try:
        for obj, key, value in changes: setattr(obj, key, value)
        yield
    finally:
        base.worker_identity, base.Boundary = identity, boundary
        for obj, key, value in reversed(prior): setattr(obj, key, value)


def prepare(root, old_queue, admission):
    root = Path(os.path.abspath(root)); arc = full.Archive(root)
    admission_ref = descriptor(admission); closed = closed_admission(root, admission_ref)
    _bound(closed['admission']['old_queue'], Path(os.path.abspath(old_queue)), arc)
    plans = root / 'plans' / NAME; require(not plans.exists(), 'Refuse to overwrite prefetch preparation')
    parents = closed['parents']; remote = parents[1][0]['workspace_root']
    originals = {str(c): p['original_qualification_profile'] for c, (p, _) in parents.items()}
    original_plan = arc.ref(parents[1][0]['ram_deployment_authorization'])['original_qualification_plan']
    require(arc.ref(original_plan)['schema'] == full.QUALIFICATION_SCHEMA, 'Wrong original qualification plan')
    reference = arc.ref(parents[1][0]['fourier_output_reference'])
    provisional.gate(original_plan, reference['fourier_gate'], arc.ref)
    deployment = {'status': 'AUTHORIZED_PREFETCH_REMAINDER_ONLY', 'original_radon_budget': 192,
        'closed_uva_admission': full.remote_ref(admission, root, remote),
        'completed_uva_prefix_ids': closed['completed_prefix_ids'],
        'assigned_by_cold': {str(c): ids for c, ids in closed['assigned_by_cold'].items()},
        'remaining_model_calls': closed['remaining_model_calls'], 'original_qualification_profiles': originals,
        'engine_delta': placement.ENGINE_DELTA, 'resource_budget': placement.RESOURCES,
        'automatic_retry': False, 'formal_launch_authorized': False,
        'qualification_scope': 'Cold1 UVA prefix plus prefetch remainder; cold2 full96 official-prefetch point'}
    deployment_path = plans / 'prefetch-deployment.json'; write_new(deployment_path, deployment)
    drafts = []
    for cold, (old, _) in sorted(parents.items()):
        p = copy.deepcopy(arc.ref(originals[str(cold)]))
        for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper', 'full_dependencies'):
            p.pop(key, None)
        name = f'radon0-prefetch-qualification-cold{cold}'
        control = plans / (name + '-control.json')
        write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
        p.update(worker=name, control_generation=0, assigned_ids=closed['assigned_by_cold'][cold],
            maximum_model_calls=len(closed['assigned_by_cold'][cold]), resources=dict(placement.RESOURCES),
            output=str(Path(remote) / 'runs/reproducibility' / NAME / name),
            claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / 'claims' / name),
            control_file=str(Path(remote) / control.relative_to(root)),
            pause_file=str(Path(remote) / control.relative_to(root).with_suffix('.PAUSE.json')),
            original_qualification_profile=originals[str(cold)], prefetch_adapter=descriptor(__file__),
            prefetch_placement_helper=descriptor(placement.__file__),
            fourier_gate_helper=descriptor(provisional.__file__),
            closed_uva_admission=full.remote_ref(admission, root, remote),
            prefetch_deployment=full.remote_ref(deployment_path, root, remote),
            fourier_output_reference=old['fourier_output_reference'])
        p['engine_kwargs'].update(copy.deepcopy(placement.ENGINE_DELTA))
        science(p, arc.ref)
        path = plans / (name + '-draft.json'); write_new(path, p)
        drafts.append({'cold': cold, 'draft': descriptor(path)})
    spec = {'schema': NAME + '/preparation-v1', 'status': 'PREPARED_NOT_LAUNCHED',
        'drafts': drafts, 'original_qualification_plan': original_plan,
        'closed_uva_admission': deployment['closed_uva_admission'],
        'prefetch_deployment': full.remote_ref(deployment_path, root, remote),
        'remaining_model_calls': closed['remaining_model_calls'], 'original_radon_budget': 192,
        'automatic_retry': False, 'formal_launch_authorized': False}
    path = plans / 'radon-qualification-plan.json'; write_new(path, spec)
    return {'status': spec['status'], 'plan': descriptor(path), 'new_model_calls': 0,
            'remaining_model_calls': closed['remaining_model_calls'],
            'preserved_uva_prefix_calls': len(closed['completed_prefix_ids'])}


def seal(draft, output):
    p = read(draft); science(p); verify_extra_code(p); full.storage_admission(p)
    closed = closed_admission(p['workspace_root'], p['closed_uva_admission'])
    require(p['assigned_ids'] == closed['assigned_by_cold'][p['cold']], 'Old claims changed after preparation')
    catalog = verified(p['source_catalog'])['items']
    files = {x['path']: x['sha256'] for sid in p['assigned_ids']
             for x in catalog[sid]['content'] if x['type'] in ('video', 'image')}
    def hashed(item):
        path, expected = item; before = base.stat(path); value = file_sha256(path)
        require(value == expected and base.stat(path) == before, 'Source changed during parallel prefetch sealing')
        return path, (value, before)
    with ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 2))) as pool:
        admitted = dict(pool.map(hashed, files.items()))
    original_sha = base.file_sha256
    def cached_sha(path):
        if str(path) not in admitted: return original_sha(path)
        value, before = admitted[str(path)]; require(base.stat(path) == before, 'Source changed after hashing')
        return value
    try:
        with adaptation():
            base.file_sha256 = cached_sha
            result = _SEAL(draft, output)
    finally: base.file_sha256 = original_sha
    return {**result, 'profile': descriptor(output)}


def stage(plan_path):
    plan_path = Path(plan_path); spec = read(plan_path)
    require(spec['schema'] == NAME + '/preparation-v1' and spec['status'] == 'PREPARED_NOT_LAUNCHED'
            and len(spec['drafts']) == 2 and {x['cold'] for x in spec['drafts']} == {1, 2},
            'Wrong finite prefetch preparation')
    profiles = []
    for entry in sorted(spec['drafts'], key=lambda x: x['cold']):
        path = plan_path.parent / Path(entry['draft']['path']).name
        require(file_sha256(path) == entry['draft']['sha256'], 'Prefetch draft changed')
        p = read(path); require(p['hostname'] == socket.gethostname(), 'Wrong staging host')
        output = path.with_name(path.name.replace('-draft.json', '-profile.json'))
        seal(path, output); profiles.append(descriptor(output))
    path = plan_path.parent / 'radon-qualification-queue.json'
    write_new(path, {'status': 'SEALED', 'automatic_retry': False, 'profiles': profiles,
        'preparation': descriptor(plan_path), 'maximum_model_calls': spec['remaining_model_calls']})
    return {'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(path),
            'maximum_model_calls': spec['remaining_model_calls'], 'new_model_calls': 0}


def _queue_profiles(path):
    q = read(path)
    require(q['status'] == 'SEALED' and q['automatic_retry'] is False and len(q['profiles']) == 2,
            'Wrong sealed finite prefetch queue')
    profiles = [(verified(ref), ref) for ref in q['profiles']]
    require([p['cold'] for p, _ in profiles] == [1, 2]
            and sum(p['maximum_model_calls'] for p, _ in profiles) == q['maximum_model_calls'],
            'Prefetch queue has duplicate cold workers or changed budget')
    for p, _ in profiles:
        science(p); verify_code(p)
        closed = closed_admission(p['workspace_root'], p['closed_uva_admission'])
        require(p['assigned_ids'] == closed['assigned_by_cold'][p['cold']], 'Old claimed scope changed')
    return q, profiles


def activate(path):
    q, profiles = _queue_profiles(path)
    for p, _ in profiles:
        control = read(p['control_file']); out = Path(p['output'])
        require(control['paused'] is True and control['generation'] == p['control_generation']
                and not out.exists() and not out.with_suffix('.launch.json').exists()
                and not list(Path(p['claim_root']).glob('*.json')), 'Attempted, claimed or stale prefetch worker')
        full.storage_admission(p)
    for p, _ in profiles:
        control = read(p['control_file']); control.update(paused=False, activated_queue=descriptor(path))
        atomic_json(p['control_file'], control)
    return {'status': 'ACTIVATED_NOT_LAUNCHED', 'queue': descriptor(path), 'new_model_calls': 0}


def queue(path):
    q, profiles = _queue_profiles(path)
    for p, ref in profiles:
        require(read(p['control_file'])['paused'] is False, 'Prefetch qualification remains paused')
        result = supervise(ref['path']); print(json.dumps(result), flush=True)
        require(result['returncode'] == 0 and result['terminal_status'] == 'COMPLETE'
                and result['real_wait_observed'] and result['owned_engines_exited']
                and result['original_physical_lease_released'], 'Prefetch worker did not close; no retry')
    return {'status': 'COMPLETE_PREFETCH_QUALIFICATION_QUEUE', 'profiles': q['profiles']}


def run(profile):
    p = read(profile); science(p); verify_code(p)
    closed = closed_admission(p['workspace_root'], p['closed_uva_admission'])
    require(p['assigned_ids'] == closed['assigned_by_cold'][p['cold']], 'Original claimed scope changed')
    reference = verified(p['fourier_output_reference']); original_append = base._append
    def observed_append(path, value):
        original_append(path, value)  # Raw is durable before comparison or failure.
        if Path(path) != Path(p['output']) / 'raw.jsonl': return
        expected = reference['outputs'][value['sample_id']]
        differences = [k for k in OUTPUT_KEYS if value[k] != expected[k]]
        prompt_equal = value['actual_prompt_token_ids'] == reference['actual_prompt_token_ids']
        receipt = {'sample_id': value['sample_id'], 'raw_digest': digest(value),
            'differing_output_fields': differences, 'complete_prompt_token_ids_equal': prompt_equal,
            'match': not differences and prompt_equal, 'actual_output': {k: value[k] for k in OUTPUT_KEYS},
            'reference_output': expected, 'actual_prompt_sha256': digest(value['actual_prompt_token_ids']),
            'reference_prompt_sha256': digest(reference['actual_prompt_token_ids'])}
        original_append(Path(p['output']) / 'prefetch-equivalence.jsonl', receipt)
        if not receipt['match']:
            atomic_json(Path(p['output']) / 'reproduction-failure.json', {**receipt,
                'status': 'PAUSE_FOURIER_REQUIRED', 'original_scientific_output_preserved': True,
                'automatic_retry': False, 'new_model_calls_by_comparison': 0})
            raise ValueError('Prefetch output/prompt mismatch; pause Fourier; raw preserved')
    try:
        with adaptation():
            base._append = observed_append
            return qualification.run(profile)
    finally: base._append = original_append


def supervise(profile):
    original = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([x.replace('classic_precision.refinement_worker', MODULE) for x in argv], **kwargs)
    try:
        worker.subprocess.Popen = popen
        return _SUPERVISE(profile)
    finally: worker.subprocess.Popen = original


def summarize(root, plan, output=None):
    from . import official_x_results as evidence
    from . import gemma_x_full_results as checks
    root = Path(root); arc = full.Archive(root); spec = read(plan)
    q = read(Path(plan).parent / 'radon-qualification-queue.json')
    require(q['preparation']['sha256'] == file_sha256(plan), 'Queue/preparation changed')
    closed = closed_admission(root, spec['closed_uva_admission'])
    qp = arc.ref(spec['original_qualification_plan']); selection = arc.ref(qp['selection']); ids = selection['assigned_ids']
    require(spec['schema'] == NAME + '/preparation-v1' and len(q['profiles']) == 2, 'Wrong prefetch summary scope')
    reference = None; new = {}; proofs = []; diffs = []; prompt_diffs = []; bindings = []
    for ref in q['profiles']:
        p = arc.ref(ref); science(p, arc.ref); verify_code(p); out = arc.path(p['output'])
        require(p['assigned_ids'] == closed['assigned_by_cold'][p['cold']], 'Summary scope differs')
        reference = arc.ref(p['fourier_output_reference'])
        with checks.gemma_admission():
            _, proof = evidence.audit_worker(arc, {'profile': ref, 'phase': 'qualification', 'variant': 'X_NATIVE_COMPACT'})
        raw = _rows(out / 'raw.jsonl'); comparisons = _rows(out / 'prefetch-equivalence.jsonl')
        require([x['sample_id'] for x in comparisons] == p['assigned_ids']
                and len(raw) == len(comparisons) and not (out / 'reproduction-failure.json').exists(),
                'Missing per-request complete prefetch comparison')
        catalog = arc.ref(p['source_catalog'])['items']; table = arc.ref(p['reference_prompt_catalog'])
        deep = _unique(_rows(out / 'deep-observations.jsonl')); inputs = _unique(_rows(out / 'input-admissions.jsonl'))
        require(set(deep) == set(inputs) == {p['assigned_ids'][0]}, 'Missing first actual GPU input proof')
        for runtime in proof['runtime']:
            full.validate_runtime(runtime); placement.validate_placement(runtime['prefetch_placement'], 'after_load')
        for observed in deep[p['assigned_ids'][0]]['observations']:
            placement.validate_placement(observed['runtime']['prefetch_placement'], 'after_first_budgeted_request')
        storage = read(out.with_suffix('.storage.json'))
        require(storage['status'] == 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN'
                and storage['profile_sha256'] == digest(p), 'Missing actual SSD admission')
        for row, comparison in zip(raw, comparisons):
            sid = row['sample_id']; checks.native_input_admission(p, catalog[sid], row, deep.get(sid), inputs.get(sid), table)
            require(comparison['match'] and comparison['complete_prompt_token_ids_equal']
                    and not comparison['differing_output_fields'] and comparison['raw_digest'] == digest(row),
                    'Saved prefetch comparison does not bind the raw')
            if any(row[k] != reference['outputs'][sid][k] for k in OUTPUT_KEYS): diffs.append({'cold':p['cold'],'sample_id':sid})
            if row['actual_prompt_token_ids'] != reference['actual_prompt_token_ids']: prompt_diffs.append({'cold':p['cold'],'sample_id':sid})
        require(p['cold'] not in new, 'Duplicate prefetch cold worker')
        new[p['cold']] = {r['sample_id']: r for r in raw}; proofs.append(proof)
        bindings.extend(descriptor(out / name) for name in PREFETCH_FILES)
        bindings.extend((descriptor(out.with_suffix('.wait.json')),descriptor(out.with_suffix('.storage.json'))))
    require(set(new) == {1, 2} and list(new[2]) == ids and list(new[1]) == closed['assigned_by_cold'][1],
            'Final prefetch cold2 must cover all96; cold1 only its original remainder')
    provisional.gate(spec['original_qualification_plan'], reference['fourier_gate'], arc.ref)
    anchor = arc.ref(reference['anchor_profile']); anchor_path = arc.path(anchor['output']) / 'raw.jsonl'
    require(file_sha256(anchor_path) == reference['anchor_raw']['sha256']
            and reference['anchor_profile'] == next(w['profile'] for w in qp['workers']
                if w['card'] == 'fourier0' and w['cold'] == 1), 'Fourier anchor raw/profile changed')
    anchor_raw = _rows(anchor_path)
    require([r['sample_id'] for r in anchor_raw] == ids
            and all({k:r[k] for k in OUTPUT_KEYS} == reference['outputs'][r['sample_id']]
                and r['actual_prompt_token_ids'] == reference['actual_prompt_token_ids'] for r in anchor_raw),
            'Saved reference does not equal the original full Fourier anchor')
    prefix = closed['prefix_raw']
    old = closed['parents'][1][0]; old_out = arc.path(old['output'])
    old_catalog = arc.ref(old['source_catalog'])['items']; old_table = arc.ref(old['reference_prompt_catalog'])
    old_deep = _unique(_rows(old_out / 'deep-observations.jsonl'))
    old_inputs = _unique(_rows(old_out / 'input-admissions.jsonl'))
    require(set(old_deep) == set(old_inputs) == {ids[0]}, 'Preserved UVA prefix lacks original input proof')
    for ordinal, (sid, row) in enumerate(prefix.items()):
        with checks.gemma_admission(): evidence.common_raw_checks(old,old_catalog[sid],row,ordinal)
        checks.native_input_admission(old,old_catalog[sid],row,old_deep.get(sid),old_inputs.get(sid),old_table)
    old_runtime = read(old_out / 'runtime.json')
    require(old_runtime and all(full.validate_runtime(r) and r['actual_kv_capacity'] >= 65536
            for r in old_runtime), 'Preserved UVA runtime differs')
    old_storage = read(old_out.with_suffix('.storage.json'))
    require(old_storage['status'] == 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN'
            and old_storage['profile_sha256'] == digest(old), 'Preserved UVA storage proof differs')
    prefix_bindings = [descriptor(old_out / name) for name in PREFIX_FILES]
    prefix_bindings.extend((descriptor(old_out.with_suffix('.wait.json')),descriptor(old_out.with_suffix('.storage.json'))))
    bindings.extend(prefix_bindings)
    for sid, row in prefix.items():
        require(all(row[k] == reference['outputs'][sid][k] for k in OUTPUT_KEYS)
                and row['actual_prompt_token_ids'] == reference['actual_prompt_token_ids'], 'Preserved UVA prefix differs')
    overlap = [sid for sid in new[1] if any(new[1][sid][k] != new[2][sid][k]
                for k in OUTPUT_KEYS + ('actual_prompt_token_ids',))]
    require(not diffs and not prompt_diffs and not overlap, 'Finite-point reproduction failed')
    summary = {'schema': NAME + '/qualification-summary-v1', 'status': PASS,
        'plan_sha256': file_sha256(plan), 'selection_sha256': qp['selection']['sha256'],
        'unique_video_ids': 96, 'tasks': {'T3':96}, 'original_radon_budget':192,
        'radon_executed_model_calls': len(prefix) + sum(len(v) for v in new.values()),
        'prefetch_executed_model_calls': sum(len(v) for v in new.values()),
        'preserved_uva_prefix_calls': len(prefix), 'preserved_uva_prefix_ids': list(prefix),
        'prefetch_cold2_full96_vs_fourier_anchor': True,
        'prefetch_cold1_remainder_ids': list(new[1]), 'prefetch_overlap_differences': overlap,
        'full_output_differences': diffs, 'prompt_token_differences': prompt_diffs,
        'cold1_deployment': 'MIXED_UVA_PREFIX_AND_OFFICIAL_PREFETCH_REMAINDER',
        'two_full_prefetch_cold96_claimed': False, 'closed_uva_admission': spec['closed_uva_admission'],
        'prefetch_worker_proofs': proofs, 'input_admission_status':'PASS_PREFETCH_SEGMENTS',
        'fourier_gate':reference['fourier_gate'], 'queue':descriptor(Path(plan).parent / 'radon-qualification-queue.json'),
        'preserved_uva_prefix_proof':{'profile':closed['parents'][1][1], 'bindings':prefix_bindings,
            'input_admission_status':'PASS_ORIGINAL_UVA_PREFIX'}, 'bindings':bindings,
        'formal_launch_authorized':False, 'automatic_retry':False, 'new_model_calls_during_analysis':0}
    require(summary['radon_executed_model_calls'] == 192, 'Radon finite192 budget differs')
    path = Path(output) if output else root / 'runs/reproducibility' / NAME / 'qualification-summary.json'
    write_new(path, summary)
    return {'status':PASS, 'summary':descriptor(path), 'radon_executed_model_calls':192,
            'prefetch_executed_model_calls':summary['prefetch_executed_model_calls'], 'new_model_calls':0}


def gate(plan_ref, summary_ref, resolve=verified):
    """Accept this finite point without presenting two complete prefetch cold96s."""
    spec, summary = resolve(plan_ref), resolve(summary_ref)
    require(spec['schema'] == NAME + '/preparation-v1' and spec['original_radon_budget'] == 192
            and summary['schema'] == NAME + '/qualification-summary-v1' and summary['status'] == PASS
            and summary['plan_sha256'] == plan_ref['sha256'] and summary['unique_video_ids'] == 96
            and summary['tasks'] == {'T3':96} and summary['radon_executed_model_calls'] == 192
            and summary['prefetch_cold2_full96_vs_fourier_anchor'] is True
            and summary['two_full_prefetch_cold96_claimed'] is False
            and summary['cold1_deployment'] == 'MIXED_UVA_PREFIX_AND_OFFICIAL_PREFETCH_REMAINDER'
            and not summary['full_output_differences'] and not summary['prompt_token_differences']
            and not summary['prefetch_overlap_differences']
            and summary['closed_uva_admission'] == spec['closed_uva_admission']
            and summary['input_admission_status'] == 'PASS_PREFETCH_SEGMENTS'
            and summary['formal_launch_authorized'] is False and summary['automatic_retry'] is False,
            'Official-prefetch finite-point gate is not PASS')
    archive = getattr(resolve, '__self__', None)
    queue = resolve(summary['queue']); profiles = [resolve(ref) for ref in queue['profiles']]
    require(queue['preparation']['sha256'] == plan_ref['sha256'] and resolve(queue['preparation']) == spec
            and [p['cold'] for p in profiles] == [1,2]
            and summary['selection_sha256'] == profiles[0]['qualification_selection']['sha256'],
            'Gate queue/selection/cold workers differ')
    for p in profiles: science(p,resolve); verify_code(p)
    deployed = resolve(spec['prefetch_deployment']); done = deployed['completed_uva_prefix_ids']
    require(summary['preserved_uva_prefix_ids'] == done and summary['preserved_uva_prefix_calls'] == len(done)
            and summary['prefetch_cold1_remainder_ids'] == profiles[0]['assigned_ids']
            and summary['prefetch_executed_model_calls'] == 192-len(done) == spec['remaining_model_calls']
            and [proof['profile'] for proof in summary['prefetch_worker_proofs']] == queue['profiles']
            and summary['preserved_uva_prefix_proof']['input_admission_status'] == 'PASS_ORIGINAL_UVA_PREFIX',
            'Gate finite allocation or preserved-prefix evidence differs')
    provisional.gate(spec['original_qualification_plan'],summary['fourier_gate'],resolve)
    path_for = archive.path if archive is not None else Path
    closed = closed_admission(archive.root if archive is not None else profiles[0]['workspace_root'],
        spec['closed_uva_admission'])
    require(summary['preserved_uva_prefix_proof']['profile'] == closed['parents'][1][1],
            'Preserved-prefix gate profile changed')
    expected_paths = []
    for p, names in [(p,PREFETCH_FILES) for p in profiles] + [(closed['parents'][1][0],PREFIX_FILES)]:
        out = path_for(p['output'])
        expected_paths.extend([out / name for name in names] + [out.with_suffix('.wait.json'),out.with_suffix('.storage.json')])
    require(len(summary['bindings']) == len(expected_paths) == 37
            and {path_for(ref['path']) for ref in summary['bindings']} == set(expected_paths),
            'Incomplete or duplicate sealed finite-point evidence bindings')
    for ref in summary['bindings']:
        path = path_for(ref['path'])
        require(file_sha256(path) == ref['sha256'], 'Finite-point gate evidence changed: '+str(path))
    return spec,summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','stage','seal','activate','queue','run','supervise','summarize','audit-profile','audit-admission'))
    for name in ('root','old-queue','admission','plan','profile','draft','output'): parser.add_argument('--'+name)
    a = parser.parse_args()
    if a.command == 'prepare': value = prepare(a.root,a.old_queue,a.admission)
    elif a.command == 'stage': value = stage(a.plan)
    elif a.command == 'seal': value = seal(a.draft,a.output)
    elif a.command == 'activate': value = activate(a.plan)
    elif a.command == 'queue': value = queue(a.plan)
    elif a.command == 'run': value = run(a.profile)
    elif a.command == 'supervise': value = supervise(a.profile)
    elif a.command == 'summarize': value = summarize(a.root,a.plan,a.output)
    elif a.command == 'audit-admission':
        closed = closed_admission(a.root,descriptor(a.admission))
        value = {'status':'CPU_CLOSED_ADMISSION_PASS','remaining_model_calls':closed['remaining_model_calls'],'new_model_calls':0}
    else:
        p = read(a.profile); science(p,full.Archive(a.root or p['workspace_root']).ref)
        value = {'status':'CPU_PROFILE_PASS','profile':descriptor(a.profile),'new_model_calls':0}
    print(json.dumps(value),flush=True)


if __name__ == '__main__': main()
