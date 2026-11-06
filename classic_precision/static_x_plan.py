"""Root-owned finite static allocation metadata; never starts a model."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import os
from pathlib import Path

from . import static_x_admission as admission
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256

OWNER = os.environ.get('PSYCHOPHYSICS_PARENT_QUEUE_OWNER', 'unbound-parent-queue')
NAME = 'static-x-t123-20261104'
GROUPS = ('cosmos', 'qwen_T1', 'qwen_T23', 'nemo', 'gemma')


def phase_dir(folder, phase, generation):
    require(type(generation) is int and generation >= 1, 'Positive preseal CPU generation required')
    base = Path(folder) / phase
    return base if generation == 1 else base / ('preseal-' + str(generation))


def preseal_only(root, folder, phase, generation):
    if generation == 1:
        return
    previous = Path(folder) / phase
    require(previous.is_dir(), 'Later CPU generation requires retained predecessor evidence')
    forbidden = [p for p in previous.rglob('*') if p.is_file() and
                 (p.name.endswith('-profile.json') or p.name == 'sealed-plan.json'
                  or p.name.startswith('queue-fourier') and p.suffix == '.json'
                  or 'launch-intent' in p.name)]
    runs = Path(root) / 'runs/reproducibility' / NAME / phase
    require(not forbidden and (not runs.exists() or not any(runs.iterdir())),
            'A later CPU generation cannot replace sealed/started workers or unknown claims')


def context(root):
    root = Path(root)
    folder = root / 'plans' / NAME
    scope = read(folder / 'authorization.json')
    require(scope['owner_chat'] == OWNER and scope['full_static_authorized'] is True
            and scope['maximum_new_model_calls'] == 58020 and scope['automatic_retry'] is False,
            'Latest finite user scope is missing')
    catalog_path = root / 'datasets' / NAME / 'catalog-all.json'
    receipt_path = root / 'datasets' / NAME / 'receipt-all.json'
    catalog = read(catalog_path)
    receipt = read(receipt_path)
    require(receipt['status'] == 'PASS' and receipt['selection'] == 'all'
            and receipt['original_scientific_video_aliases'] == 19872
            and receipt['preprojection_cases'] == 14490
            and receipt['final_unique_cases'] == len(catalog['items'])
            and catalog['prepare_receipt_sha256'] == file_sha256(receipt_path),
            'Complete physical-source projection is not accepted')
    protocol = descriptor(folder / 'protocol.json')
    require(scope['protocol']['sha256'] == protocol['sha256'], 'User protocol changed')
    bindings = read(folder / 'parent-bindings.json')['records']
    require(len(bindings) == 10 and len({(x['model_group'], x['card']) for x in bindings}) == 10,
            'Ten exact parent video profiles required')
    return root, folder, catalog, descriptor(catalog_path), descriptor(receipt_path), protocol, bindings


def ids_for(group, ids, catalog):
    return [sid for sid in ids if not group.startswith('qwen_')
            or (catalog['items'][sid]['task'] == 'T1') == (group == 'qwen_T1')]


def draft(parent_ref, runtime_ref, group, assigned, output, catalog, protocol):
    parent = verified(parent_ref)
    model = admission.model_key(parent)
    require(model == group.split('_')[0], 'Parent model/group differs')
    p = copy.deepcopy(parent)
    for key in ('worker_token', 'media_files', 'sources', 'request_adapter',
                'static_cpu_budget', 'static_entry', 'execution_policy', 'static_dependencies'):
        p.pop(key, None)
    p.update(worker=output.name, assigned_ids=assigned, maximum_model_calls=len(assigned),
             output=str(output), claim_root=str(output.with_name(output.name + '-claims')),
             pause_file=str(output.with_name(output.name + '-pause.json')),
             source_catalog=catalog, static_protocol=protocol, parent_validated_profile=parent_ref,
             automatic_retry=False, formal_auto_resume=False,
             engine_kwargs=admission.expected_engine(parent, model),
             request_mm_processor_kwargs=admission.native_kwargs(parent),
             request_mm_processor_kwargs_by_task={})
    if model == 'gemma':
        p['request_adapter'] = {'module': admission.MODULE, 'function': 'create_gemma',
                                'source_sha256': file_sha256(admission.__file__)}
    if model == 'cosmos':
        p['parent_runtime'] = runtime_ref
    return p


def qualification(root, generation=1):
    root, folder, catalog, cref, _, protocol, bindings = context(root)
    preseal_only(root, folder, 'qualification', generation)
    preparation = phase_dir(folder, 'qualification', generation)
    panel = catalog['panel_case_ids']
    require(len(set(panel)) == len(panel) == 15
            and Counter(catalog['items'][sid]['task'] for sid in panel) == {'T1': 5, 'T2': 5, 'T3': 5},
            'The fixed five-per-task panel changed')
    rows, drafts = [], []
    for binding in bindings:
        group, card = binding['model_group'], binding['card']
        cold = int(card[-1])
        assigned = ids_for(group, panel, catalog)
        output = root / 'runs/reproducibility' / NAME / 'qualification' / (group + '-' + card)
        p = draft(binding['profile'], binding['runtime'], group, assigned, output, cref, protocol)
        p['static_cold'] = cold
        p['static_card'] = card
        rows.append({'model': admission.model_key(p), 'cold': cold, 'assigned_ids': assigned,
                     'output': p['output'], 'claim_root': p['claim_root'],
                     'parent_profile_sha256': binding['profile']['sha256'], 'gpu_uuid': p['gpu_uuid']})
        drafts.append((preparation / (group + '-' + card + '-draft.json'), p))
    auth = {'schema': 'static-x-120-call-authorization-v1', 'maximum_model_calls': 120,
            'automatic_retry': False, 'full_static_authorized': False, 'scope_authorization': descriptor(folder / 'authorization.json'),
            'case_ids': panel, 'cold_gpu_uuids': {str(int(x['card'][-1])): x['gpu_uuid'] for x in bindings},
            'allocations': rows}
    auth_path = preparation / 'authorization.json'
    auth_path.parent.mkdir(parents=True, exist_ok=True)
    write_new(auth_path, auth)
    for path, p in drafts:
        p['static_authorization'] = descriptor(auth_path)
        admission.configuration(p)
        write_new(path, p)
    result = {'schema': 'static-x-qualification-draft-plan-v1', 'status': 'CPU_DRAFTS_ONLY',
              'preseal_generation': generation,
              'scope_authorization': descriptor(folder / 'authorization.json'), 'authorization': descriptor(auth_path),
              'drafts': [descriptor(path) for path, _ in drafts], 'model_calls': 0}
    write_new(preparation / 'draft-plan.json', result)
    return result


def paired_owners(catalog):
    """Connected components preserve every LR pair even after input collapse."""
    items = catalog['items']; parent = {sid: sid for sid in items}
    def find(sid):
        while parent[sid] != sid:
            parent[sid] = parent[parent[sid]]; sid = parent[sid]
        return sid
    def union(left, right):
        left, right = find(left), find(right)
        if left != right:
            parent[max(left, right)] = min(left, right)
    pairs = defaultdict(list)
    for sid, item in items.items():
        for key in item['pair_keys']:
            pairs[key].append(sid)
    for ids in pairs.values():
        require(len(ids) in (1, 2), 'A scientific pair has too many canonical inputs')
        for sid in ids[1:]: union(ids[0], sid)
    groups = defaultdict(list)
    for sid in items: groups[find(sid)].append(sid)
    panel = set(catalog['panel_case_ids']); totals = Counter(); owners = {}
    for _, ids in sorted(groups.items(), key=lambda x: (not bool(set(x[1]) & panel), digest(sorted(x[1])))):
        task = items[ids[0]]['task']
        require(all(items[s]['task'] == task for s in ids), 'Cross-task pair component')
        card = 'fourier0' if panel.intersection(ids) else min(('fourier0', 'fourier1'), key=lambda c: (totals[c, task], c))
        totals[card, task] += len(ids)
        owners.update({sid: card for sid in ids})
    require(all(owners[sid] == 'fourier0' for sid in panel), 'Cold0 answer reuse needs its qualified card')
    require(all(len({owners[sid] for sid in ids}) == 1 for ids in pairs.values()), 'Split LR pair')
    return owners


def full(root, summary_ref, generation=1):
    from . import static_x_operations as ops
    root, folder, catalog, cref, receipt, protocol, bindings = context(root)
    preseal_only(root, folder, 'full', generation)
    preparation = phase_dir(folder, 'full', generation)
    summary = ops.qualification_summary(summary_ref)
    owners = paired_owners(catalog)
    panel = set(catalog['panel_case_ids'])
    reuse = []
    for pref in summary['profiles']:
        p = verified(pref)
        if p['static_cold'] != 0: continue
        for raw in admission.strict_records(Path(p['output']) / 'raw.jsonl'):
            sid = raw['sample_id']; item = catalog['items'][sid]
            require(sid in panel and raw['source']['item_digest'] == digest(item), 'Cold0 input identity changed')
            reuse.append({'model': admission.model_key(p), 'formal_case_id': sid, 'qualification_case_id': sid,
                          'profile': pref, 'raw_digest': digest(raw),
                          'model_input_sha256': ops.input_identity(item), 'truth': item['row']['truth']})
    allocations, drafts = [], []
    for binding in bindings:
        group, card = binding['model_group'], binding['card']
        normal = ids_for(group, sorted(s for s in owners if owners[s] == card), catalog)
        reused = [s for s in normal if s in panel]
        assigned = [s for s in normal if s not in panel]
        output = root / 'runs/reproducibility' / NAME / 'full' / (group + '-' + card)
        p = draft(binding['profile'], binding['runtime'], group, assigned, output, cref, protocol)
        allocation_id = group + '-' + card
        p.update(static_cold=None, static_card=card, qualification_summary=summary_ref, full_allocation_id=allocation_id)
        allocations.append({'allocation_id': allocation_id, 'model': admission.model_key(p), 'card': card,
                            'gpu_uuid': p['gpu_uuid'], 'normal_ids': normal, 'assigned_ids': assigned, 'reused_ids': reused,
                            'output': p['output'], 'claim_root': p['claim_root'],
                            'parent_profile_sha256': binding['profile']['sha256']})
        drafts.append((preparation / (allocation_id + '-draft.json'), p))
    n = len(catalog['items'])
    auth = {'schema': 'static-x-full-authorization-v1', 'full_static_authorized': True, 'automatic_retry': False,
            'source_catalog': cref, 'source_manifest': descriptor(folder / 'source-cases.json'),
            'source_prepare_receipt': receipt, 'static_protocol': protocol, 'qualification_summary': summary_ref,
            'normal_unique_answers': 4*n, 'accepted_reuse_count': 60, 'qualification_model_calls': 120,
            'maximum_formal_new_model_calls': 4*n-60, 'maximum_new_model_calls': 4*n+60,
            'scope_authorization': descriptor(folder / 'authorization.json'),
            'allocations': allocations, 'accepted_reuse': reuse}
    path = preparation / 'authorization.json'; path.parent.mkdir(parents=True, exist_ok=True); write_new(path, auth)
    for path_draft, p in drafts:
        p['static_authorization'] = p['full_authorization'] = descriptor(path)
        ops.configuration(p)
        write_new(path_draft, p)
    result = {'schema': 'static-x-full-draft-plan-v1', 'status': 'FULL_CPU_DRAFTS_ONLY',
              'preseal_generation': generation,
              'authorization': descriptor(path), 'drafts': [descriptor(p) for p, _ in drafts], 'model_calls': 0}
    write_new(preparation / 'draft-plan.json', result)
    return result


def cpu_batch(root, phase, generation=1):
    """Two isolated CPU processes; no engine or CUDA visibility is allowed."""
    import os
    import subprocess
    import sys
    from concurrent.futures import ThreadPoolExecutor
    root = Path(root); folder = phase_dir(root / 'plans' / NAME, phase, generation)
    plan = read(folder / 'draft-plan.json')
    module = admission if phase == 'qualification' else __import__(
        'classic_precision.static_x_operations', fromlist=['cpu'])
    def one(ref):
        p = verified(ref); name = Path(ref['path']).stem.removesuffix('-draft')
        require(sys.executable == p['python'] and os.environ.get('CUDA_VISIBLE_DEVICES') == '',
                'CPU batch must use the existing bound interpreter with CUDA hidden')
        output = folder / 'cpu' / (name + '.json')
        output.parent.mkdir(parents=True, exist_ok=True)
        def checked_budget():
            check = dict(p, static_cpu_budget=descriptor(output))
            if phase == 'full':
                receipt = read(output)
                check['static_cpu_template_map'] = receipt['sid_to_reference']
                check['static_deep_ids'] = receipt['representative_ids']
            module.verify_cpu(check)
        if output.exists():
            checked_budget()
            return {'draft': ref, 'cpu_budget': descriptor(output), 'reused_verified_CPU': True}
        env = os.environ.copy()
        for key in p.get('cleared_environment', []): env.pop(key, None)
        env.update(p['environment'])
        env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        cmd = [p['python'], '-B', '-m', module.MODULE, 'cpu', '--draft', ref['path'], '--output', str(output)]
        with output.with_suffix('.log').open('ab') as log:
            result = subprocess.run(cmd, cwd=Path(admission.__file__).parent.parent,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
        require(result.returncode == 0 and output.is_file(), 'CPU admission failed: ' + str(output.with_suffix('.log')))
        checked_budget()
        return {'draft': ref, 'cpu_budget': descriptor(output), 'reused_verified_CPU': False}
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(one, plan['drafts']))
    result = {'schema': 'static-x-cpu-budget-plan-v1', 'status': 'PASS_NATIVE_CPU_GPU_PENDING',
              'preseal_generation': generation,
              'phase': phase, 'draft_plan': descriptor(folder / 'draft-plan.json'),
              'workers': results, 'model_calls': 0, 'weights_loaded': False, 'CUDA_visible': False}
    write_new(folder / 'native-cpu-plan.json', result)
    return result


def seal_queues(root, phase, gates, generation=1, cards=('fourier0', 'fourier1'), cpu_equivalence=None, final_review=None):
    """Seal only released cards; the global Gemma-last manifest can arrive later."""
    import os
    from . import static_x_queue as queue
    module = admission if phase == 'qualification' else __import__(
        'classic_precision.static_x_operations', fromlist=['seal'])
    root = Path(root); folder = phase_dir(root / 'plans' / NAME, phase, generation)
    budget_plan = read(folder / 'native-cpu-plan.json')
    require(budget_plan['status'] == 'PASS_NATIVE_CPU_GPU_PENDING' and budget_plan['phase'] == phase,
            'Complete native CPU budget plan absent')
    selected = set(cards)
    require(selected and selected <= {'fourier0', 'fourier1'} and selected <= set(gates), 'Actual gate needed for each selected card')
    require(phase != 'full' or final_review is not None, 'Formal sealing requires the final Astra code/DS/GPU-qualification review')
    profiles = []
    for worker in budget_plan['workers']:
        p = verified(worker['draft']); p['static_cpu_budget'] = worker['cpu_budget']
        if p['static_card'] not in selected: continue
        name = Path(worker['draft']['path']).stem.removesuffix('-draft')
        if admission.model_key(p) == 'gemma':
            p['request_adapter']['source_sha256'] = file_sha256(admission.__file__)
        p['execution_policy'] = {'module': module.MODULE, 'source_sha256': file_sha256(module.__file__)}
        gate_path = Path(gates[p['static_card']])
        gate = read(gate_path)
        if phase == 'qualification' and gate['schema'] == 'static-x-per-card-qualification-gate-template-v1':
            gate_value = copy.deepcopy(gate['gate_evidence'])
            require(gate_value['card'] == p['static_card'] and gate_value['gpu_uuid'] == p['gpu_uuid'], 'Wrong card evidence template')
            gate_value.update(static_cpu_budget=p['static_cpu_budget'], static_authorization=p['static_authorization'],
                              source_catalog=p['source_catalog'], source_template=descriptor(gate_path))
            gate_path = folder / (name + '-start-gate.json'); write_new(gate_path, gate_value)
        p['static_start_gate'] = descriptor(gate_path)
        if cpu_equivalence is not None:
            require(phase == 'qualification', 'Only qualified CPU references need the operational migration')
            p['static_cpu_code_equivalence'] = descriptor(cpu_equivalence)
        if phase == 'full':
            cpu = verified(worker['cpu_budget'])
            p['static_cpu_template_map'] = cpu['sid_to_reference']; p['static_deep_ids'] = cpu['representative_ids']
            p['final_static_review'] = descriptor(final_review)
            p['final_review_draft'] = worker['draft']
        admission.start_gate(p)
        bound_draft = folder / (name + '-admitted-draft.json')
        profile = folder / (name + '-profile.json')
        write_new(bound_draft, p); module.seal(bound_draft, profile); profiles.append(descriptor(profile))
    plans = []
    for card in sorted(selected):
        own = sorted((ref for ref in profiles if verified(ref)['static_card'] == card),
                     key=lambda ref: GROUPS.index(Path(ref['path']).name.removesuffix('-' + card + '-profile.json')))
        p = verified(own[0]); lock_path = queue.queue_lock_path(p['physical_lock'], p['gpu_uuid'])
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try: state = os.fstat(fd)
        finally: os.close(fd)
        plan = {'schema': 'static-x-finite-queue-v1', 'owner_chat': OWNER, 'phase': phase,
                'card': card, 'gpu_uuid': p['gpu_uuid'], 'output': str(root / 'runs/reproducibility' / NAME / phase / ('queue-' + card)),
                'queue_entry': descriptor(queue.__file__),
                'queue_lease': {'path': str(lock_path), 'st_dev': state.st_dev, 'st_ino': state.st_ino},
                'profiles': [{**ref, 'supervisor_module': module.MODULE,
                              'supervisor_sha256': file_sha256(module.__file__)} for ref in own]}
        for key in ('static_authorization', 'source_catalog', 'static_protocol'):
            plan[key] = p[key]
        if phase == 'full':
            for key in ('full_authorization', 'qualification_summary'): plan[key] = p[key]
        auth = verified(p['static_authorization'])
        expected = sorted(digest(x) for x in auth['allocations'] if x['model'] != 'gemma')
        require(len(expected) == 8, 'All eight non-Gemma roles must precede Gemma')
        plan['pre_gemma_manifest_intent'] = {
            'schema': 'static-x-pre-gemma-manifest-intent-v1', 'path': str(folder/'pre-gemma-global-profiles.json'),
            'owner_chat': OWNER, 'expected_allocation_digests': expected,
            **{key: plan[key] for key in ('static_authorization', 'source_catalog', 'static_protocol')}}
        if phase == 'full':
            for key in ('full_authorization', 'qualification_summary'): plan['pre_gemma_manifest_intent'][key] = plan[key]
        plan['phase_queue_outputs'] = {x['gpu_uuid']: str(root/'runs/reproducibility'/NAME/phase/('queue-'+x['card']))
                                       for x in read(root/'plans'/NAME/'parent-bindings.json')['records']}
        plan['gemma_barrier_timeout_seconds'] = 3600 if phase == 'qualification' else 86400
        path = folder / ('queue-' + card + '.json'); write_new(path, plan)
        queue.validate_plan(path, OWNER); plans.append(descriptor(path))
    result = {'schema': 'static-x-sealed-phase-plan-v1', 'phase': phase, 'status': 'SEALED_NOT_LAUNCHED',
              'preseal_generation': generation,
              'profiles': profiles, 'queues': plans, 'model_calls': 0}
    result['static_authorization' if phase == 'qualification' else 'full_authorization'] = verified(profiles[0])['static_authorization']
    suffix = '-'.join(sorted(selected))
    write_new(folder / ('sealed-plan-' + suffix + '.json'), result)
    all_profiles = []
    for worker in budget_plan['workers']:
        name = Path(worker['draft']['path']).stem.removesuffix('-draft')
        path = folder/(name+'-profile.json')
        if path.exists(): all_profiles.append(descriptor(path))
    if len(all_profiles) == 10:
        first = verified(all_profiles[0]); auth = verified(first['static_authorization'])
        manifest = {'schema': 'static-x-global-pre-gemma-profiles-v1', 'phase': phase, 'owner_chat': OWNER,
                    'expected_allocation_digests': sorted(digest(x) for x in auth['allocations'] if x['model'] != 'gemma'),
                    'profiles': [{**ref, 'supervisor_module': module.MODULE, 'supervisor_sha256': file_sha256(module.__file__)}
                                 for ref in all_profiles if admission.model_key(verified(ref)) != 'gemma'],
                    **{key: first[key] for key in ('static_authorization', 'source_catalog', 'static_protocol')}}
        if phase == 'full':
            for key in ('full_authorization', 'qualification_summary'): manifest[key] = first[key]
        write_new(folder/'pre-gemma-global-profiles.json', manifest)
        complete = dict(result, profiles=all_profiles, queues=[descriptor(folder/('queue-'+card+'.json'))
                        for card in ('fourier0','fourier1')])
        write_new(folder/'sealed-plan.json', complete)
    return result


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('qualification', 'full', 'cpu', 'seal'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--summary')
    parser.add_argument('--phase', choices=('qualification', 'full'))
    parser.add_argument('--gate0')
    parser.add_argument('--gate1')
    parser.add_argument('--generation', type=int, default=1, help='Preserves previous zero-model preseal evidence')
    parser.add_argument('--cards', nargs='+', choices=('fourier0','fourier1'), default=['fourier0','fourier1'])
    parser.add_argument('--cpu-equivalence')
    parser.add_argument('--final-review')
    args = parser.parse_args()
    if args.command == 'qualification': result = qualification(args.root, args.generation)
    elif args.command == 'full': result = full(args.root, descriptor(args.summary), args.generation)
    elif args.command == 'cpu':
        require(args.phase is not None, 'CPU phase required'); result = cpu_batch(args.root, args.phase, args.generation)
    else:
        require(args.phase is not None, 'Phase required')
        gates = {card: path for card, path in [('fourier0',args.gate0),('fourier1',args.gate1)] if path}
        result = seal_queues(args.root, args.phase, gates, args.generation, args.cards, args.cpu_equivalence, args.final_review)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
