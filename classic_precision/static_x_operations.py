"""Finite full-static policy over the frozen static_x_admission executor.

The root controller owns scheduling. This module never launches a successor,
retries a claim, or reconstructs a model forward/reader/adapter implementation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile

from . import static_x_admission as admission
from . import official_x_runner as base
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256

MODULE = 'classic_precision.static_x_operations'
MODELS = ('cosmos', 'qwen', 'nemo', 'gemma')
_QUALIFICATION_CACHE = {}
_FULL_RUNTIME_CACHE = {}
_COMPLETE_CATALOG_CACHE = set()


def input_identity(item):
    """Identity of the actual input; provenance aliases do not change it."""
    images = [x['pixel_sha256'] for x in admission.image_entries(item)]
    identity = [images, item['row']['prompt'], item['row']['choices']]
    checksum = admission.sha(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                        separators=(',', ':')).encode('utf-8'))
    require(item.get('model_input_sha256') == checksum, 'Actual ordered input signature differs')
    return checksum


def qualification_summary(ref):
    summary = admission.cached_verified(ref)
    require(summary['schema'] == 'static-x-15case-two-cold-summary-v1'
            and summary['status'] == 'PASS_STATIC_15CASE_TWO_COLD'
            and summary['executed_model_calls'] == 120
            and summary['exact_repeat_pairs'] == summary['total_pairs'] == 60
            and all(x['equal'] for x in summary['comparisons']),
            'All 120 fixed two-cold qualification calls must PASS before full execution')
    return summary


def complete_catalog(auth, catalog):
    receipt = admission.cached_verified(auth['source_prepare_receipt'])
    manifest = admission.cached_verified(auth['source_manifest'])
    cache_key = (auth['source_catalog']['sha256'], auth['source_prepare_receipt']['sha256'], auth['source_manifest']['sha256'])
    if cache_key in _COMPLETE_CATALOG_CACHE:
        return
    items = catalog['items']; n = len(items)
    require(receipt['status'] == 'PASS' and receipt['selection'] == 'all'
            and receipt['preprojection_cases'] == len(manifest['items']) == 14490
            and receipt['original_scientific_video_aliases'] == manifest['source_video_count'] == 19872
            and receipt['final_unique_cases'] == n
            and catalog['prepare_receipt_sha256'] == auth['source_prepare_receipt']['sha256']
            and catalog['manifest_sha256'] == receipt['manifest_sha256'] == auth['source_manifest']['sha256']
            and receipt['runtime']['implementation_sha256'] == file_sha256(Path(__file__).with_name('static_x_sources.py'))
            and not receipt['GT_conflicts'], 'Full catalog is not the complete accepted --all projection')
    require(any(ref['sha256'] == auth['static_protocol']['sha256'] and ref['path'] == auth['static_protocol']['path']
                for ref in manifest['source_evidence']), 'Projection protocol is not the full authorized protocol')
    mapping = receipt['case_id_mapping']
    require(set(mapping) == set(manifest['items']) and set(mapping.values()) == set(items), 'Projection mapping is incomplete')
    reverse = defaultdict(list)
    for old_sid, canonical_sid in mapping.items():
        reverse[canonical_sid].append(old_sid)
    originals = {}
    for old_sid, item in manifest['items'].items():
        for alias in item['original_video_aliases']:
            key = (alias['domain'], alias['sample_id'])
            require(key not in originals, 'Original manifest repeats a scientific video')
            originals[key] = (old_sid, alias)
    found = set()
    for sid, item in items.items():
        require(sorted(item['preprojection_case_aliases']) == sorted(reverse[sid]),
                'Canonical case aliases differ from full projection map')
        for alias in item['original_video_aliases']:
            key = (alias['domain'], alias['sample_id'])
            require(key in originals and key not in found, 'Missing/duplicate/unknown final scientific alias')
            old_sid, original = originals[key]
            require(mapping[old_sid] == sid and alias['preprojection_case_id'] == old_sid
                    and all(alias.get(k) == value for k, value in original.items()), 'Scientific alias metadata was lost or changed')
            found.add(key)
    require(found == set(originals) and len(found) == 19872, 'Full scientific source denominator is incomplete')
    require(receipt['counts'] == dict(Counter(item['domain'] + '/' + item['task'] for item in items.values())),
            'Full measured unique-input counts differ')
    _COMPLETE_CATALOG_CACHE.add(cache_key)


def authorization(p, catalog=None):
    require(p['full_authorization'] == p['static_authorization'], 'Full and executor authorization refs differ')
    auth = admission.cached_verified(p['full_authorization'])
    catalog = admission.cached_verified(p['source_catalog']) if catalog is None else catalog
    items = catalog['items']; n = len(items)
    require(auth['schema'] == 'static-x-full-authorization-v1' and auth['full_static_authorized'] is True
            and auth['automatic_retry'] is False and p.get('automatic_retry', False) is False
            and p.get('formal_auto_resume', False) is False,
            'Explicit finite full-static authorization is missing')
    require(auth['source_catalog'] == p['source_catalog'] and auth['static_protocol'] == p['static_protocol']
            and auth['qualification_summary'] == p['qualification_summary'], 'Mixed source/protocol/qualification')
    complete_catalog(auth, catalog)
    require(n <= 14490 and auth['normal_unique_answers'] == 4 * n
            and auth['accepted_reuse_count'] == 60 and auth['qualification_model_calls'] == 120
            and auth['maximum_formal_new_model_calls'] == 4 * n - 60
            and auth['maximum_new_model_calls'] == 4 * n + 60 <= 58020, 'Full call denominator/cap differs')
    summary = qualification_summary(p['qualification_summary'])
    panel = catalog['panel_case_ids']
    require(len(panel) == len(set(panel)) == 15
            and Counter(items[s]['task'] for s in panel) == {'T1': 5, 'T2': 5, 'T3': 5}, 'Final fixed panel differs')
    reuse = auth['accepted_reuse']; allocations = auth['allocations']
    require(len(reuse) == 60 and len({(x['model'], x['formal_case_id']) for x in reuse}) == 60
            and {(x['model'], x['formal_case_id']) for x in reuse}
            == {(model, sid) for model in MODELS for sid in panel}, 'Reuse must be exactly cold0 fixed-panel sixty')
    require(all(x['profile'] in summary['profiles'] for x in reuse), 'Reuse profile is outside completed qualification')
    require(len(allocations) == 10 and len({x['allocation_id'] for x in allocations}) == 10
            and Counter(x['model'] for x in allocations) == {'cosmos': 2, 'qwen': 4, 'nemo': 2, 'gemma': 2},
            'Expected ten finite workers, including Qwen T1/T23 on both cards')
    cards = {x['card']: x['gpu_uuid'] for x in allocations}
    require(len(cards) == len(set(cards.values())) == 2
            and all(cards[x['card']] == x['gpu_uuid'] for x in allocations)
            and all(Counter(x['gpu_uuid'] for x in allocations if x['model'] == model)
                    == {uuid: 2 if model == 'qwen' else 1 for uuid in cards.values()} for model in MODELS),
            'Full workers must cover the same two fixed cards with serial task-specific Qwen workers')
    owners, new_slots, reuse_slots, paths, claims = {}, set(), set(), set(), set()
    pair_cards = defaultdict(set)
    for allocation in allocations:
        model = allocation['model']
        normal, new, reused = allocation['normal_ids'], allocation['assigned_ids'], allocation['reused_ids']
        require(normal and new and len(normal) == len(set(normal))
                and len(new) == len(set(new)) and len(reused) == len(set(reused))
                and set(new).isdisjoint(reused) and set(normal) == set(new) | set(reused), 'Invalid finite owner partition')
        require(all(s in items for s in normal), 'Unknown static case in allocation')
        tasks = {items[s]['task'] for s in normal}
        require(model != 'qwen' or tasks in ({'T1'}, {'T2', 'T3'}), 'Qwen resource profiles were mixed')
        require(allocation['output'] not in paths, 'Workers share an output directory')
        paths.add(allocation['output'])
        for sid in normal:
            key = (model, sid)
            require(key not in owners, 'A static case has more than one formal owner')
            owners[key] = allocation
            for pair_key in items[sid].get('pair_keys', [items[sid]['pair_key']]):
                pair_cards[model, pair_key].add(allocation['gpu_uuid'])
        for sid in new:
            new_slots.add((model, sid))
            claim = (allocation['claim_root'], sid)
            require(claim not in claims, 'Formal calls share a durable claim')
            claims.add(claim)
        reuse_slots.update((model, sid) for sid in reused)
    expected = {(model, sid) for model in MODELS for sid in items}
    require(set(owners) == expected and len(new_slots) == auth['maximum_formal_new_model_calls']
            and reuse_slots == {(x['model'], x['formal_case_id']) for x in reuse}
            and new_slots.isdisjoint(reuse_slots) and new_slots | reuse_slots == expected,
            'Full owner coverage, accepted reuse or actual new-call denominator differs')
    require(all(len(cards) == 1 for cards in pair_cards.values()), 'LR pair-connected inputs must stay on one card')
    own = [x for x in allocations if x['allocation_id'] == p['full_allocation_id']]
    require(len(own) == 1, 'Unknown full allocation')
    allocation = own[0]
    require(allocation['model'] == admission.model_key(p) and allocation['assigned_ids'] == p['assigned_ids']
            and allocation['output'] == p['output'] and allocation['claim_root'] == p['claim_root']
            and allocation['gpu_uuid'] == p['gpu_uuid'] and allocation['card'] == p['static_card']
            and (p.get('card') is None or allocation['card'] == p['card'])
            and allocation['parent_profile_sha256'] == p['parent_validated_profile']['sha256']
            and p['maximum_model_calls'] == len(p['assigned_ids']) and p.get('static_cold') is None,
            'Profile is not its exact new-call full allocation')
    return auth, allocation


def configuration(p):
    parent = admission.cached_verified(p['parent_validated_profile'])
    model = admission.model_key(p)
    require(admission.model_key(parent) == model
            and p['engine_kwargs'] == admission.expected_engine(parent, model), 'Unapproved full engine change')
    for field in ('checkpoint', 'checkpoint_assets', 'processor_checkpoint', 'python', 'environment',
                  'cleared_environment', 'expected_packages', 'vision_observation', 'bootstrap',
                  'minimum_kv_capacity', 'media_io_kwargs', 'gpu_uuid', 'physical_lock', 'card',
                  'hostname', 'resources', 'torch_num_threads'):
        require(p.get(field) == parent.get(field), 'Inherited deployment changed: ' + field)
    require(p.get('request_mm_processor_kwargs', {}) == admission.native_kwargs(p)
            and not p.get('request_mm_processor_kwargs_by_task') and not p.get('bootstrap'), 'Full input kwargs differ')
    if model == 'gemma':
        require(p.get('request_adapter') == {'module': admission.MODULE, 'function': 'create_gemma',
                'source_sha256': file_sha256(admission.__file__)}, 'Full Gemma adapter is not frozen admission entry')
    else:
        require(not p.get('request_adapter'), 'Unexpected native image input adapter')
    if model == 'qwen':
        require(p.get('selector_source') == parent.get('selector_source'), 'Full X QSA source changed')
    catalog = admission.cached_verified(p['source_catalog'])
    protocol = admission.cached_verified(p['static_protocol'])
    _, allocation = authorization(p, catalog)
    if model == 'qwen':
        tasks = {catalog['items'][sid]['task'] for sid in allocation['normal_ids']}
        context, kv = (98304, 8 * 2**30) if tasks == {'T1'} else (65536, 4 * 2**30)
        require(p['engine_kwargs']['max_model_len'] == p['minimum_kv_capacity'] == context
                and p['engine_kwargs']['kv_cache_memory_bytes'] == kv, 'Qwen task-specific context/KV profile differs')
    for sid in allocation['normal_ids']:
        item = catalog['items'][sid]
        require(item['sample_id'] == item['case_id'] == sid, 'Static identity differs')
        admission.check_item(item, protocol)
        input_identity(item)
    return catalog


def representative_map(p, catalog):
    buckets = defaultdict(list)
    for sid in p['assigned_ids']:
        item = catalog['items'][sid]
        key = (item['task'], item['domain'], digest(item['image_contract']),
               item['row']['prompt'], tuple(item['row']['choices']))
        buckets[key].append(sid)
    representatives, mapping = [], {}
    for key, ids in sorted(buckets.items()):
        sid = min(ids, key=lambda x: digest(['full-static-cpu-representative-v1', x, input_identity(catalog['items'][x])]))
        representatives.append(sid)
        mapping.update({item_id: sid for item_id in ids})
    require(len(representatives) <= 6, 'Unexpected CPU reference strata')
    return representatives, mapping


def budget_binding(p):
    return {key: copy.deepcopy(p[key]) for key in ('source_catalog', 'static_protocol', 'parent_validated_profile',
            'full_authorization', 'qualification_summary')} | {
        'engine_kwargs_digest': digest(p['engine_kwargs']), 'assigned_ids_digest': digest(p['assigned_ids']),
        'entry': descriptor(__file__), 'admission_entry': descriptor(admission.__file__),
        'source_entry': descriptor(Path(__file__).with_name('static_x_sources.py')),
        'static_dependencies': admission.dependencies()}


def cpu(draft, output):
    import torch
    p = read(draft); catalog = configuration(p)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_initialized(), 'CPU budget requires invisible CUDA')
    require(sys.executable == p['python'], 'CPU processor must use the bound execution interpreter')
    for package, version in p['expected_packages'].items():
        require(importlib.metadata.version(package) == version, 'CPU package changed: ' + package)
    torch.set_num_threads(p.get('torch_num_threads') or 1)
    representatives, mapping = representative_map(p, catalog)
    attempts = [admission.cpu_sample(p, catalog['items'][sid]) for sid in representatives]
    require(not torch.cuda.is_initialized(), 'CPU processor initialized CUDA')
    value = {'schema': 'static-x-full-native-cpu-v1', 'status': 'PASS', 'model_calls': 0,
             'weights_loaded': False, 'CUDA_initialized': False, **budget_binding(p),
             'representative_ids': representatives, 'sid_to_reference': mapping,
             'assigned_item_digests': {s: digest(catalog['items'][s]) for s in p['assigned_ids']},
             'attempts': attempts}
    write_new(output, value)
    return {'status': 'CPU_PASS_GPU_PENDING', 'static_cpu_budget': descriptor(output),
            'static_deep_ids': representatives, 'template_map_in_budget': 'sid_to_reference',
            'mapped_items': len(mapping), 'representatives': len(representatives), 'model_calls': 0}


def verify_cpu(p, catalog=None):
    catalog = configuration(p) if catalog is None else catalog
    budget = admission.cached_verified(p['static_cpu_budget'])
    require(budget['schema'] == 'static-x-full-native-cpu-v1' and budget['status'] == 'PASS'
            and budget['model_calls'] == 0 and budget['weights_loaded'] is False
            and budget['CUDA_initialized'] is False, 'No real full CPU budget')
    for key, value in budget_binding(p).items():
        require(budget[key] == value, 'CPU source/configuration/code binding changed: ' + key)
    representatives, mapping = representative_map(p, catalog)
    require(budget['representative_ids'] == p['static_deep_ids'] == representatives
            and budget['sid_to_reference'] == p['static_cpu_template_map'] == mapping
            and budget['assigned_item_digests'] == {s: digest(catalog['items'][s]) for s in p['assigned_ids']}
            and [x['sample_id'] for x in budget['attempts']] == representatives, 'CPU representative or full sid map differs')
    for attempt in budget['attempts']:
        item = catalog['items'][attempt['sample_id']]
        require(attempt['item_digest'] == digest(item) and attempt['model'] == admission.model_key(p)
                and attempt['image_count'] == item['image_contract']['count']
                and len(attempt['mm_placeholders']['image']) == attempt['image_count']
                and attempt['prompt_tokens'] == len(attempt['prompt_token_ids'])
                and attempt['prompt_tokens'] + 16384 <= p['engine_kwargs']['max_model_len']
                and attempt['fits_context'] and not attempt['CUDA_initialized']
                and attempt['processor_resize_crop_calls'] == 0, 'CPU reference is incomplete or different')
        for ref in attempt['official_source_identity']:
            require(admission.official_source(ref['path'], ref['distribution']) == ref, 'Official processor source changed')
    return budget


def _qualification_closed(ref):
    p = admission.cached_verified(ref); out = Path(p['output'])
    files = [out.with_suffix('.wait.json'), out / 'result.json', out / 'exit.json', out / 'raw.jsonl',
             out / 'input-admissions.jsonl', out / 'events.jsonl', out / 'deep-observations.jsonl',
             out / 'deep-install.jsonl', out / 'runtime.json']
    files.extend(Path(p['claim_root']) / (sid + '.json') for sid in p['assigned_ids'])
    state = [(str(path), base.stat(path)) for path in files]
    key = (ref['path'], ref['sha256'])
    if key not in _QUALIFICATION_CACHE:
        profile, raws = admission.closed(ref['path'])
        _QUALIFICATION_CACHE[key] = (state, profile, {x['sample_id']: x for x in raws})
    saved, profile, raws = _QUALIFICATION_CACHE[key]
    require(state == saved, 'Accepted qualification evidence changed after closure verification')
    return profile, raws


def accepted_reuse(p, catalog=None):
    catalog = admission.cached_verified(p['source_catalog']) if catalog is None else catalog
    auth, allocation = authorization(p, catalog)
    rows = {}
    for reuse in auth['accepted_reuse']:
        sid = reuse['formal_case_id']
        if reuse['model'] != admission.model_key(p) or sid not in allocation['reused_ids']:
            continue
        old, old_raw = _qualification_closed(reuse['profile'])
        source_sid = reuse['qualification_case_id']; raw = old_raw[source_sid]
        source_item = admission.cached_verified(old['source_catalog'])['items'][source_sid]
        item = catalog['items'][sid]
        require(old['static_cold'] == 0 and admission.model_key(old) == admission.model_key(p)
                and old['gpu_uuid'] == p['gpu_uuid'] and raw['cold'] == 0,
                'Only exact cold0 answers on their formal owner card may be reused')
        require(input_identity(item) == input_identity(source_item) == reuse['model_input_sha256']
                and item['row']['truth'] == source_item['row']['truth'] == raw['truth'] == reuse['truth']
                and raw['question'] == item['row']['prompt'] and raw['raw_answer'].strip() in ('LEFT', 'RIGHT')
                and raw['finish_reason'] == 'stop' and digest(raw) == reuse['raw_digest'],
                'Accepted reuse pixels/order/prompt/choices/GT or original answer changed')
        require(admission.deployment_signature(p) == admission.deployment_signature(old),
                'Accepted cold0 deployment differs from full deployment')
        generation_contract(raw['generation'])
        rows[sid] = {'raw': raw, 'profile': reuse['profile'], 'qualification_case_id': source_sid,
                     'model_input_sha256': reuse['model_input_sha256']}
    require(set(rows) == set(allocation['reused_ids']), 'Missing exact accepted-reuse mapping')
    return rows


def qualified_deployment(p, runtime=None):
    """Bind this card/task profile to its already completed static qualification."""
    summary = qualification_summary(p['qualification_summary'])
    candidates = []
    for ref in summary['profiles']:
        old = admission.cached_verified(ref)
        if (admission.model_key(old) == admission.model_key(p)
                and old['gpu_uuid'] == p['gpu_uuid']
                and admission.deployment_signature(old) == admission.deployment_signature(p)):
            candidates.append(ref)
    require(len(candidates) == 1, 'No unique completed qualification for this exact card/deployment')
    old, _ = _qualification_closed(candidates[0])
    if runtime is not None:
        expected = read(Path(old['output']) / 'runtime.json')
        require([admission.runtime_signature(x) for x in runtime]
                == [admission.runtime_signature(x) for x in expected],
                'Actual full deployment differs from its same-card qualified runtime')
    return candidates[0]


def verify_code(p):
    require(p['execution_policy'] == {'module': MODULE, 'source_sha256': file_sha256(__file__)}
            and p['static_entry'] == descriptor(admission.__file__)
            and p['static_dependencies'] == admission.dependencies()
            and p['implementation']['sha256'] == file_sha256(base.__file__), 'Frozen full execution code changed')


def final_review_codes():
    codes = admission.dependencies()
    for name in ('static_x_sources', 'static_x_admission', 'static_x_operations', 'static_x_queue', 'static_x_plan'):
        codes[name] = descriptor(Path(__file__).with_name(name + '.py'))
    require(len(codes) == 18, 'Final review must cover the exact eighteen-file execution bundle')
    return codes


def final_preflight(p):
    """Require the final Astra code/data/120-call review, without new inference."""
    require(p.get('final_static_review') and p.get('final_review_draft'), 'Full static execution lacks its final Astra review')
    review = verified(p['final_static_review'])
    require(review['schema'] == 'static-x-final-full-preflight-v1'
            and review['status'] == 'PASS_FINAL_CODE_DS_AND_QUALIFICATION'
            and review['reviewer'] == 'astra1' and review['full_authorization'] == p['full_authorization']
            and review['code_bundle'] == final_review_codes(), 'Final Astra review/code bundle is missing, stale or for another scope')
    auth = admission.cached_verified(p['full_authorization'])
    for key in ('source_catalog', 'source_manifest', 'source_prepare_receipt', 'static_protocol', 'qualification_summary'):
        require(review[key] == auth[key], 'Final review source/qualification binding differs: ' + key)
    qualification_summary(review['qualification_summary'])
    catalog = admission.cached_verified(review['source_catalog'])
    complete_catalog(auth, catalog)
    workers = review['workers']
    require(len(workers) == 10 and len({x['draft']['path'] for x in workers}) == 10
            and len({x['cpu_budget']['path'] for x in workers}) == 10,
            'Final review must bind ten distinct original drafts and CPU budgets')
    allocation_ids, own = set(), []
    for worker in workers:
        draft = verified(worker['draft']); budget = verified(worker['cpu_budget'])
        allocation_id = draft['full_allocation_id']
        require(allocation_id not in allocation_ids, 'Final review repeats a full allocation')
        allocation_ids.add(allocation_id)
        require(draft['full_authorization'] == draft['static_authorization'] == review['full_authorization']
                and draft['source_catalog'] == review['source_catalog']
                and draft['static_protocol'] == review['static_protocol']
                and draft['qualification_summary'] == review['qualification_summary'], 'Reviewed draft belongs to another scope')
        require(budget['schema'] == 'static-x-full-native-cpu-v1' and budget['status'] == 'PASS'
                and budget['model_calls'] == 0 and budget['weights_loaded'] is False and budget['CUDA_initialized'] is False
                and budget['full_authorization'] == draft['full_authorization']
                and budget['source_catalog'] == draft['source_catalog'] and budget['static_protocol'] == draft['static_protocol']
                and budget['qualification_summary'] == draft['qualification_summary']
                and budget['parent_validated_profile'] == draft['parent_validated_profile']
                and budget['engine_kwargs_digest'] == digest(draft['engine_kwargs'])
                and budget['assigned_ids_digest'] == digest(draft['assigned_ids']), 'Reviewed CPU budget belongs to another draft/configuration')
        if worker['draft'] == p['final_review_draft']:
            own.append((worker, draft))
    require(allocation_ids == {x['allocation_id'] for x in auth['allocations']} and len(own) == 1,
            'Final review does not exactly cover all full allocations or this original draft')
    worker, approved = own[0]
    require(approved['full_allocation_id'] == p['full_allocation_id']
            and worker['cpu_budget'] == p['static_cpu_budget']
            and all(key in p and p[key] == value for key, value in approved.items()),
            'Sealing/execution changed an immutable reviewed draft field or replaced its CPU budget')
    return {'status': 'PASS_FINAL_CODE_DS_AND_QUALIFICATION', 'review': p['final_static_review'],
            'draft': p['final_review_draft'], 'reviewed_workers': 10, 'reviewed_code_files': 18}


def full_policy(p):
    final_preflight(p)
    catalog = configuration(p)
    verify_code(p); verify_cpu(p, catalog); qualified_deployment(p); accepted_reuse(p, catalog)
    return catalog


def representative_deep(p, item, ordinal):
    return item['sample_id'] in p['static_deep_ids']


def generation_contract(value):
    expected = {'n': 1, 'temperature': 0.0, 'top_p': 1.0, 'seed': 0, 'repetition_penalty': 1.0,
                'max_tokens': 16384, 'ignore_eos': False, 'stop': [], 'stop_token_ids': [],
                'choice': ['LEFT', 'RIGHT'], 'thinking': False, 'native_EOS_override': False}
    require(all(value.get(key) == expected_value for key, expected_value in expected.items()), 'Generation settings changed')


def full_input(p, item, raw, deep, persist=True):
    sid = item['sample_id']; is_deep = sid in p['static_deep_ids']
    require(raw['profile_sha256'] == digest(p) and raw['source']['catalog'] == p['source_catalog']
            and raw['source']['item_digest'] == digest(item)
            and raw['source']['media'] == [p['media_files'][x['path']] for x in admission.image_entries(item)]
            and raw['deep'] is is_deep and bool(deep) is is_deep, 'Actual raw source/profile/deep mapping differs')
    generation_contract(raw['generation'])
    require(raw['request_mm_processor_kwargs'] == admission.native_kwargs(p), 'Actual image kwargs changed')
    runtime_path = Path(p['output']) / 'runtime.json'
    runtime_key = (p['worker_token'], str(runtime_path)); state = base.stat(runtime_path)
    if runtime_key not in _FULL_RUNTIME_CACHE:
        qualified_deployment(p, read(runtime_path))
        _FULL_RUNTIME_CACHE[runtime_key] = state
    require(_FULL_RUNTIME_CACHE[runtime_key] == state, 'Actual runtime identity file changed during execution')
    expected = admission.budget_for(p, sid)
    if is_deep:
        require(expected['sample_id'] == sid, 'Deep sample must have its own actual CPU reference')
        qualified_deployment(p, [value['runtime'] for value in deep])
        proof = admission.validate_input(p, item, raw, deep, persist=False)
    else:
        proof = admission.ordinary_input(p, item, raw, expected)
    if persist:
        base._append(Path(p['output']) / 'input-admissions.jsonl', {
            'sample_id': sid, 'raw_digest': digest(raw), 'observations_digest': digest(deep),
            'representative_sid': expected['sample_id'], 'deep': is_deep, 'proof': proof})
    return proof


def seal(draft, output):
    p = read(draft); final_preflight(p); catalog = configuration(p)
    p.update(static_entry=descriptor(admission.__file__), static_dependencies=admission.dependencies(),
             execution_policy={'module': MODULE, 'source_sha256': file_sha256(__file__)},
             automatic_retry=False, formal_auto_resume=False)
    verify_cpu(p, catalog); accepted_reuse(p, catalog)
    from .gemma_x_full import storage_admission
    storage_admission(p)
    files = {entry['path']: entry['sha256'] for sid in p['assigned_ids']
             for entry in admission.image_entries(catalog['items'][sid])}
    def hashed(pair):
        path, expected = pair; before = base.stat(path)
        require(file_sha256(path) == expected and base.stat(path) == before, 'PNG changed during sealing')
        return path, (expected, before)
    with ThreadPoolExecutor(max_workers=4) as pool:
        admitted = dict(pool.map(hashed, files.items()))
    original_sha = base.file_sha256
    def admitted_sha(path):
        key = str(path)
        if key not in admitted:
            return original_sha(path)
        checksum, state = admitted[key]
        require(base.stat(path) == state, 'PNG changed after parallel SHA admission')
        return checksum
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); name = stream.name
    try:
        base.file_sha256 = admitted_sha
        return base.seal(name, output)
    finally:
        base.file_sha256 = original_sha
        Path(name).unlink()


def run(profile):
    return admission.execute_bound_profile(profile, policy_validator=full_policy,
        input_validator=full_input, deep_policy=representative_deep, runtime_validator=qualified_deployment)


def supervise(profile):
    from . import refinement_worker as worker
    full_policy(read(profile))
    original = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([x.replace('classic_precision.refinement_worker', MODULE) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = original


def closed(profile_path):
    p = read(profile_path); catalog = full_policy(p); out = Path(p['output'])
    wait = read(out.with_suffix('.wait.json')); result = read(out / 'result.json')
    require(wait['profile_sha256'] == digest(p) and wait['returncode'] == 0 and wait['real_wait_observed']
            and wait['owned_engines_exited'] and wait['original_physical_lease_released']
            and wait['completed_ids'] == p['assigned_ids'] and wait['terminal_status'] == 'COMPLETE'
            and result['status'] == 'COMPLETE' and result['completed_ids'] == p['assigned_ids']
            and not result.get('UNCERTAIN_ids') and not (out / 'failure.json').exists(), 'Full worker did not truly close COMPLETE')
    require(result['automatic_retry'] is False and result['formal_auto_resume'] is False, 'Unexpected retry/resume state')
    exit_record = read(out / 'exit.json'); lease = read(out / 'lease.json')
    lock = p['physical_lock']; lock_path = Path(lock['path'])
    require(exit_record['owned_engine_workers_exited'] and exit_record['shutdown_error'] is None
            and not lock_path.is_symlink() and (lock_path.stat().st_dev, lock_path.stat().st_ino)
            == (lock['st_dev'], lock['st_ino'])
            and all(lease[key] == value for key, value in lock.items()), 'Owned shutdown or original lease identity differs')
    identity = read(out / 'identity.json')
    require(identity['profile'] == p and identity['profile_sha256'] == digest(p), 'Worker profile identity changed')
    storage = read(out / 'storage-admission.json')
    require(storage['status'] == 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN'
            and storage['profile_sha256'] == digest(p), 'Physical SSD gate missing')
    runtime = read(out / 'runtime.json')
    require(runtime and all(x['gpu_uuid'].lower().removeprefix('gpu-')
            == p['gpu_uuid'].lower().removeprefix('gpu-')
            and x['actual_kv_capacity'] >= p.get('minimum_kv_capacity', p['engine_kwargs']['max_model_len']) for x in runtime),
            'Full actual GPU/KV runtime differs')
    admission.validate_parent_runtime(p, runtime)
    qualified_deployment(p, runtime)
    raws = admission.strict_records(out / 'raw.jsonl'); proofs = admission.strict_records(out / 'input-admissions.jsonl')
    events = admission.strict_records(out / 'events.jsonl'); observations = admission.strict_records(out / 'deep-observations.jsonl')
    installs = admission.strict_records(out / 'deep-install.jsonl')
    ids = p['assigned_ids']; expected_deep = [sid for sid in ids if sid in p['static_deep_ids']]
    require([x['sample_id'] for x in raws] == [x['sample_id'] for x in proofs] == ids
            and [x['sample_id'] for x in observations] == [x['sample_id'] for x in installs] == expected_deep,
            'Full raw/input/deep/install exact sets/order differ')
    require([(x['event'], x['sample_id']) for x in events]
            == [(event, sid) for sid in ids for event in ('INTENT', 'TECHNICAL_PASS')], 'INTENT/PASS journal differs')
    auth, _ = authorization(p, catalog)
    allowed = {sid for allocation in auth['allocations'] if allocation['claim_root'] == p['claim_root']
               for sid in allocation['assigned_ids']}
    require(set(ids) <= admission.claim_ids(p['claim_root'], allowed) <= allowed, 'Full claims missing or unauthorized')
    deep_by_id = {x['sample_id']: x for x in observations}
    _, generation = base.generation(['LEFT', 'RIGHT'])
    for ordinal, (raw, proof, intent, passed) in enumerate(zip(raws, proofs, events[::2], events[1::2])):
        sid = ids[ordinal]; item = catalog['items'][sid]; is_deep = sid in p['static_deep_ids']
        claim = read(Path(p['claim_root']) / (sid + '.json'))
        require(claim == {'sample_id': sid, 'profile_sha256': digest(p), 'worker_token': p['worker_token'],
                         'attempt': 1, 'cold': None, 'authorization': p['static_authorization']}, 'Full single-attempt claim differs')
        require(raw['sample_id'] == sid and raw['model'] == p['model'] and raw['variant'] == 'CURRENT_X_STATIC'
                and raw['cold'] is None and raw['attempt'] == 1 and raw['task'] == item['task']
                and raw['domain'] == item['domain'] and raw['truth'] == item['row']['truth']
                and raw['question'] == item['row']['prompt'] and raw['finish_reason'] == 'stop'
                and raw['raw_answer'].strip() in ('LEFT', 'RIGHT') and raw['generation'] == generation
                and raw['correct'] == (raw['raw_answer'].strip() == raw['truth'])
                and raw['ordinal'] == intent['ordinal'] == passed['ordinal'] == ordinal
                and raw['deep'] is intent['deep'] is passed['deep'] is is_deep
                and proof['raw_digest'] == passed['raw_digest'] == digest(raw)
                and proof['deep'] is is_deep and proof['representative_sid'] == p['static_cpu_template_map'][sid],
                'Full raw/generation/source/proof journal binding differs')
        observation = deep_by_id.get(sid); deep = observation['observations'] if observation else []
        require(proof['observations_digest'] == digest(deep), 'Deep observation digest differs')
        if is_deep:
            require(observation['raw_digest'] == digest(raw)
                    and [admission.runtime_signature(x['runtime']) for x in deep]
                    == [admission.runtime_signature(x) for x in runtime], 'Representative actual deployment drifted')
        require(proof['proof'] == full_input(p, item, raw, deep, persist=False), 'Full input proof cannot reproduce')
    return p, raws


def summarize(plan_path, output):
    plan = read(plan_path); auth = verified(plan['full_authorization'])
    answers, profiles, reused_count, new_count = {}, [], 0, 0
    catalog = verified(auth['source_catalog']); items = catalog['items']
    require(len(plan['profiles']) == len(auth['allocations']) == 10, 'Incomplete full profile plan')
    allocation_ids = set()
    for ref in plan['profiles']:
        verified(ref); p, raws = closed(ref['path'])
        require(p['full_authorization'] == plan['full_authorization'], 'Mixed full authorization in summary')
        require(p['full_allocation_id'] not in allocation_ids, 'Duplicate full allocation profile')
        allocation_ids.add(p['full_allocation_id']); profiles.append(ref)
        model = admission.model_key(p)
        for raw in raws:
            key = (model, raw['sample_id']); require(key not in answers, 'Duplicate formal answer')
            answers[key] = {'raw': raw, 'profile': ref, 'origin': 'formal'}; new_count += 1
        for sid, value in accepted_reuse(p, catalog).items():
            key = (model, sid); require(key not in answers, 'Cold0 reused twice or formally repeated')
            answers[key] = {**value, 'origin': 'qualification_cold0'}; reused_count += 1
    require(allocation_ids == {x['allocation_id'] for x in auth['allocations']}
            and set(answers) == {(model, sid) for model in MODELS for sid in items}
            and len(answers) == auth['normal_unique_answers'] and reused_count == 60
            and new_count == auth['maximum_formal_new_model_calls']
            and new_count + 120 == auth['maximum_new_model_calls'], 'Final unique/reuse/new-call totals differ')
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    unique_path = output.with_name(output.stem + '-unique-answers.jsonl')
    alias_path = output.with_name(output.stem + '-scientific-aliases.jsonl')
    unique_metrics, alias_metrics = defaultdict(Counter), defaultdict(Counter)
    with unique_path.open('x') as unique_file, alias_path.open('x') as alias_file:
        for (model, sid), value in sorted(answers.items()):
            item = items[sid]; raw = value['raw']; answer = raw['raw_answer'].strip()
            row = {'model': model, 'case_id': sid, 'task': item['task'], 'domain': item['domain'],
                   'truth': item['row']['truth'], 'answer': answer, 'correct': answer == item['row']['truth'],
                   'origin': value['origin'], 'profile': value['profile'], 'raw_digest': digest(raw),
                   'model_input_sha256': input_identity(item),
                   'original_scientific_video_aliases': len(item['original_video_aliases']),
                   'pair_keys': item.get('pair_keys', [item['pair_key']]),
                   'preprojection_case_aliases': item.get('preprojection_case_aliases', [sid])}
            unique_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
            key = model + '/' + item['domain'] + '/' + item['task']
            unique_metrics[key].update(total=1, correct=int(row['correct']))
            for alias in item['original_video_aliases']:
                alias_row = {**copy.deepcopy(alias), 'model': model, 'canonical_case_id': sid,
                             'model_input_sha256': row['model_input_sha256'], 'answer': answer,
                             'correct': answer == alias['truth'], 'origin': value['origin'],
                             'profile': value['profile'], 'raw_digest': row['raw_digest']}
                alias_file.write(json.dumps(alias_row, ensure_ascii=False, sort_keys=True) + '\n')
                key = model + '/' + alias['domain'] + '/' + alias['task']
                alias_metrics[key].update(total=1, correct=int(alias_row['correct']))
    result = {'schema': 'static-x-full-summary-v1', 'status': 'COMPLETE_PASS',
              'full_authorization': plan['full_authorization'], 'profiles': profiles,
              'source_catalog': auth['source_catalog'], 'qualification_summary': auth['qualification_summary'],
              'normal_unique_answers': len(answers), 'accepted_reused_cold0_answers': reused_count,
              'executed_formal_model_calls': new_count, 'qualification_model_calls': 120,
              'all_new_model_calls': new_count + 120, 'maximum_new_model_calls': auth['maximum_new_model_calls'],
              'unique_input_metrics': {k: dict(v) for k, v in unique_metrics.items()},
              'original_scientific_alias_metrics': {k: dict(v) for k, v in alias_metrics.items()},
              'unique_answers': descriptor(unique_path), 'scientific_aliases': descriptor(alias_path),
              'denominator_policy': 'Unique inference answers and original scientific aliases are separate. Alias rows preserve every source condition/onset and point to the shared unique answer; identical-input aliases are not independent model calls.',
              'automatic_retry': False, 'formal_auto_resume': False,
              'ordinary_scope': 'Every new raw has ordered PNG source/prompt/shape/placement proof; only explicit CPU representative cases have actual GPU-byte deep evidence.'}
    write_new(output, result)
    return {'status': result['status'], 'summary': descriptor(output), 'normal_unique_answers': len(answers),
            'new_model_calls': new_count + 120, 'model_calls_during_summary': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cpu', 'seal', 'run', 'supervise', 'closed', 'summarize'))
    for field in ('draft', 'profile', 'plan', 'output'):
        parser.add_argument('--' + field)
    args = parser.parse_args()
    if args.command == 'cpu': result = cpu(args.draft, args.output)
    elif args.command == 'seal': result = seal(args.draft, args.output)
    elif args.command == 'run': result = run(args.profile)
    elif args.command == 'supervise': result = supervise(args.profile)
    elif args.command == 'closed':
        p, rows = closed(args.profile)
        result = {'status': 'COMPLETE_PASS', 'profile': descriptor(args.profile), 'new_formal_calls': len(rows)}
    else: result = summarize(args.plan, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
