"""Admit the original reserved Radon2256 after the real finite prefetch PASS.

Fourier retains its original profiles and dispatcher. Radon inherits the actual
complete prefetch cold2 template; cold1 remains explicitly UVA-prefix/prefetch-
remainder evidence. Qualification never enters the formal9456 denominator.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import socket
import tempfile

from . import gemma_x_full as full
from . import gemma_x_full_results as results
from . import gemma_x_prefetch_placement as placement
from . import gemma_x_prefetch_qualification as qualification
from . import gemma_x_provisional as provisional
from .refinement_worker import read, require, verified, write_new
from .storage import atomic_json, file_sha256

NAME = provisional.NAME
MODULE = 'classic_precision.gemma_x_prefetch_full'
POLICY = 'FOURIER_STARTED_PROVISIONALLY_RADON_AFTER_PREFETCH_FINITE_POINT'
ADAPTER_FIELDS = qualification.EXTRA_FIELDS | {
    'prefetch_full_adapter', 'prefetch_finite_plan', 'original_reserved_radon_draft',
    'provisional_adapter',
}
_SCIENCE = full.science
_VERIFY = full.verify_dependencies
_SEAL = full.seal
_CLOSED = full.closed
descriptor = qualification.descriptor


def radon_allocation(profile, plan):
    allocation = plan['cards']['radon0']['assigned_ids']; ids = profile['assigned_ids']
    require(profile['card'] == 'radon0' and len(allocation) == len(set(allocation)) == 2256
            and len(ids) == len(set(ids)) == profile['maximum_model_calls']
            and set(ids) <= set(allocation), 'Wrong reserved Radon2256 allocation')
    if not profile.get('continuation_parent'):
        require(profile['control_generation'] == 0 and ids == allocation,
                'Initial Radon worker must retain the original ordered2256 allocation')


def fresh_radon(profile, archive):
    out = archive.path(profile['output']); control = read(archive.path(profile['control_file']))
    require(not out.exists() and not out.with_suffix('.launch.json').exists()
            and not out.with_suffix('.wait.json').exists(), 'Radon formal worker was already attempted')
    require(control['paused'] and control['generation'] == profile['control_generation'] == 0,
            'Initial Radon control must remain paused and unfenced')
    require(not any((archive.path(profile['claim_root']) / (sid + '.json')).exists()
                    for sid in profile['assigned_ids']), 'Reserved Radon scope already contains a claim')


def finite_gate(plan_ref, summary_ref, resolve=verified):
    plan, summary = qualification.gate(plan_ref, summary_ref, resolve)
    original = resolve(plan['original_qualification_plan'])
    fourier = resolve(summary['fourier_gate'])
    prefix = summary['preserved_uva_prefix_calls']
    require(summary['status'] == qualification.PASS
            and summary['plan_sha256'] == plan_ref['sha256']
            and summary['selection_sha256'] == original['selection']['sha256']
            and fourier['executed_model_calls'] == 384 and fourier['closed_workers'] == 4
            and summary['radon_executed_model_calls'] == 192
            and summary['prefetch_executed_model_calls'] + prefix == 192
            and prefix == len(summary['preserved_uva_prefix_ids'])
            and 0 < prefix < 96
            and summary['prefetch_cold2_full96_vs_fourier_anchor'] is True
            and summary['two_full_prefetch_cold96_claimed'] is False
            and summary['closed_uva_admission'] == plan['closed_uva_admission'],
            'Real finite-point qualification is not the original Fourier384/Radon192 scope')
    proofs = summary['prefetch_worker_proofs']
    profiles = [(resolve(proof['profile']), proof['profile']) for proof in proofs]
    require(Counter(p['cold'] for p, _ in profiles) == {1: 1, 2: 1},
            'Missing actual prefetch cold worker')
    require(all(proof['accepted'] == p['maximum_model_calls'] == len(p['assigned_ids'])
                for proof, (p, _) in zip(proofs, profiles))
            and sum(proof['accepted'] for proof in proofs) == summary['prefetch_executed_model_calls'],
            'Actual prefetch worker ledger proofs do not reconstruct the finite remainder')
    for p, _ in profiles:
        _placement_evidence(p, resolve, require_complete=True)
    template, reference = next((p, ref) for p, ref in profiles if p['cold'] == 2)
    ids = resolve(original['selection'])['assigned_ids']
    require(template['card'] == 'radon0' and template['phase'] == 'qualification'
            and template['assigned_ids'] == ids and template['maximum_model_calls'] == 96
            and all(template['engine_kwargs'].get(k) == v for k, v in placement.ENGINE_DELTA.items())
            and template['resources'] == placement.RESOURCES,
            'Formal template must be the actually completed full96 prefetch cold2 profile')
    return plan, summary, template, reference


def _placement_evidence(profile, resolve, require_complete=False):
    archive = getattr(resolve, '__self__', None)
    path_for = archive.path if archive is not None else Path
    out = path_for(profile['output'])
    terminal = read(out / 'result.json')
    if require_complete:
        require(terminal['status'] == 'COMPLETE'
                and terminal['completed_ids'] == profile['assigned_ids']
                and not terminal['UNCERTAIN_ids'],
                'Actual prefetch qualification worker is not completely and cleanly closed')
    if terminal['status'] == 'PAUSED_BEFORE_LOAD' and not terminal['completed_ids']:
        require(terminal['model_load_started'] is False, 'Before-load pause unexpectedly loaded a model')
        return {'model_load_started': False}
    runtime_path = out / 'runtime.json'
    runtime = read(runtime_path)
    require(runtime, 'Missing actual prefetch runtime placement evidence')
    for value in runtime:
        full.validate_runtime(value)
        placement.validate_placement(value['prefetch_placement'], 'after_load')
    deep_path = out / 'deep-observations.jsonl'
    deep = list(full.lifecycle.rows(deep_path)) if deep_path.exists() else []
    require(not terminal['completed_ids'] or deep, 'Completed prefetch requests lack actual input/runtime admission')
    for row in deep:
        require(row['observations'], 'Empty actual prefetch deep observation')
        for observed in row['observations']:
            placement.validate_placement(observed['runtime']['prefetch_placement'], 'after_first_budgeted_request')
    result = {'runtime': descriptor(runtime_path), 'model_load_started': True}
    if deep_path.exists(): result['first_task_requests'] = descriptor(deep_path)
    return result


def science(profile, resolve=None, require_gate=True):
    if profile['card'] in provisional.FOURIER:
        with provisional.adaptation(): return provisional.science(profile, resolve, require_gate)
    p = profile
    resolver = resolve or full.Archive(p['workspace_root']).ref
    plan = resolver(p['experiment_plan']); radon_allocation(p, plan)
    require(p.get('prefetch_full_adapter', {}).get('sha256') == file_sha256(__file__),
            'Radon prefetch full adapter is not source-bound')
    qplan, summary, template, template_ref = finite_gate(
        p['prefetch_finite_plan'], p['qualification_summary'], resolver)
    require(p['qualification_plan']['sha256'] == plan['qualification_plan']['sha256']
            == qplan['original_qualification_plan']['sha256']
            and p['qualified_profile'] == template_ref
            and all(p.get(key) == template.get(key) for key in qualification.EXTRA_FIELDS),
            'Formal profile changed its original plan, actual cold2 template or sealed deployment bindings')
    reserved = resolver(p['original_reserved_radon_draft'])
    require(reserved['card'] == 'radon0' and reserved['phase'] == 'full'
            and reserved['experiment_plan'] == p['experiment_plan']
            and reserved['assigned_ids'] == plan['cards']['radon0']['assigned_ids'],
            'Original reserved Radon allocation differs')
    if not p.get('continuation_parent'):
        require(all(p[key] == reserved[key] for key in (
            'worker', 'output', 'claim_root', 'control_file', 'pause_file', 'control_generation',
            'source_catalog', 'canonical_catalog', 'assigned_ids', 'maximum_model_calls')),
            'Initial Radon formal paths or original2256 allocation changed')
    auth = resolver(p['execution_authorization'])
    require(auth['launch_policy'] == 'RADON_RESERVED_AFTER_REAL_PREFETCH_FINITE_PASS'
            and auth['allowed_launch_cards'] == ['radon0'] and auth['maximum_radon_full_calls'] == 2256
            and auth['prefetch_finite_plan'] == p['prefetch_finite_plan']
            and auth['prefetch_finite_summary'] == p['qualification_summary']
            and auth['closed_uva_admission'] == qplan['closed_uva_admission']
            and auth['fourier_gate'] == summary['fourier_gate']
            and auth['original_reserved_radon_draft'] == p['original_reserved_radon_draft']
            and auth['full_adapter']['sha256'] == file_sha256(__file__),
            'Missing bound conditional Radon2256 formal authorization')
    # The common auditor needs only selection/card-template fields. This view is
    # explicitly the new finite gate; it carries no old six-complete-worker PASS.
    view = {'schema': qualification.NAME + '/formal-gate-view-v1',
            'status': qualification.PASS, 'selection': resolver(qplan['original_qualification_plan'])['selection'],
            'cards': {'radon0': {'qualified_profile': template_ref}},
            'two_full_prefetch_cold96_claimed': False}
    previous = full.qualification_gate
    previous_operational = full.OPERATIONAL
    previous_name = full.NAME
    def bound_gate(original_ref, summary_ref, resolve=verified):
        require(original_ref['sha256'] == qplan['original_qualification_plan']['sha256']
                and summary_ref == p['qualification_summary'], 'Wrong finite qualification gate supplied')
        return view, summary
    try:
        full.NAME = NAME
        full.OPERATIONAL = previous_operational | ADAPTER_FIELDS
        full.qualification_gate = bound_gate
        return _SCIENCE(p, resolve, require_gate)
    finally:
        full.qualification_gate = previous
        full.OPERATIONAL = previous_operational
        full.NAME = previous_name


def verify_dependencies(profile):
    if profile['card'] in provisional.FOURIER: return provisional.verify_dependencies(profile)
    _VERIFY(profile)
    require(profile['prefetch_full_adapter']['sha256'] == file_sha256(__file__), 'Prefetch full source changed')
    qualification.verify_extra_code(profile)


def _seal(draft, output):
    p = read(draft); require(p['card'] == 'radon0', 'This adapter cannot reseal Fourier profiles')
    p['prefetch_full_adapter'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temporary = stream.name
    try: return _SEAL(temporary, output)
    finally: Path(temporary).unlink()


def closed(profile, allowed=('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD')):
    p, terminal, raw = _CLOSED(profile, allowed)
    if p['card'] == 'radon0': _placement_evidence(p, full.Archive(p['workspace_root']).ref)
    return p, terminal, raw


@contextmanager
def adaptation():
    changes = [(full, 'NAME', NAME), (full, 'MODULE', MODULE),
        (full, 'OPERATIONAL', full.OPERATIONAL | ADAPTER_FIELDS),
        (full, 'science', science), (full, 'verify_dependencies', verify_dependencies),
        (full, 'seal', _seal), (full, 'closed', closed),
        (full, 'worker_identity', placement.worker_identity), (full.base, 'read_probe', placement.read_probe),
        (results, 'NAME', NAME), (results, 'science', science), (results, 'verify_dependencies', verify_dependencies)]
    previous = [(obj, key, getattr(obj, key)) for obj, key, _ in changes]
    try:
        for obj, key, value in changes: setattr(obj, key, value)
        yield
    finally:
        for obj, key, value in reversed(previous): setattr(obj, key, value)


def prepare_radon(root, qualification_plan, qualification_summary, output_dir=None):
    root = Path(os.path.abspath(root)); archive = full.Archive(root)
    qref, sref = descriptor(qualification_plan), descriptor(qualification_summary)
    qplan, summary, template, template_ref = finite_gate(qref, sref, archive.ref)
    old_path = root / 'plans' / NAME / 'radon0-full-segment0-draft.json'; reserved = read(old_path)
    fresh_radon(reserved, archive)
    plans = Path(output_dir) if output_dir else old_path.parent / 'radon-prefetch-full'
    require(not plans.exists(), 'Refuse to overwrite Radon prefetch formal preparation')
    remote = reserved['workspace_root']
    qremote = full.remote_ref(qualification_plan, root, remote)
    sremote = full.remote_ref(qualification_summary, root, remote)
    with adaptation():
        p = {key: copy.deepcopy(value) for key, value in reserved.items() if key in full.OPERATIONAL}
        p.update({key: copy.deepcopy(value) for key, value in template.items() if key not in full.OPERATIONAL})
        p.update({key: copy.deepcopy(template[key]) for key in qualification.EXTRA_FIELDS})
    original_auth = archive.ref(template['execution_authorization'])
    original_ref = full.remote_ref(old_path, root, remote)
    authorization = {'status': 'AUTHORIZED', 'maximum_full_calls': 9456, 'cards': list(full.CARDS),
        'qualification_selection': archive.ref(qplan['original_qualification_plan'])['selection'],
        'user_message': original_auth['user_message'], 'original_authorization': template['execution_authorization'],
        'launch_policy': 'RADON_RESERVED_AFTER_REAL_PREFETCH_FINITE_PASS', 'allowed_launch_cards': ['radon0'],
        'maximum_radon_full_calls': 2256, 'prefetch_finite_plan': qremote, 'prefetch_finite_summary': sremote,
        'fourier_gate': summary['fourier_gate'], 'closed_uva_admission': qplan['closed_uva_admission'],
        'original_reserved_radon_draft': original_ref, 'full_adapter': descriptor(__file__),
        'automatic_retry': False, 'two_full_prefetch_cold96_claimed': False}
    auth_path = plans / 'authorization.json'; write_new(auth_path, authorization)
    p.update(qualified_profile=template_ref, qualification_summary=sremote, prefetch_finite_plan=qremote,
        prefetch_full_adapter=descriptor(__file__), original_reserved_radon_draft=original_ref,
        execution_authorization=full.remote_ref(auth_path, root, remote))
    with adaptation(): science(p, archive.ref)
    draft = plans / 'radon0-full-segment0-draft.json'; write_new(draft, p)
    return {'status': 'RADON2256_DRAFT_AFTER_REAL_PREFETCH_FINITE_PASS', 'draft': descriptor(draft),
        'authorization': descriptor(auth_path), 'assigned': 2256, 'new_model_calls': 0,
        'existing_Fourier_profiles_and_original_radon_draft_unchanged': True}


def stage(draft, output=None):
    p = read(draft); require(p['card'] == 'radon0' and p['hostname'] == socket.gethostname(), 'Wrong staging host/card')
    fresh_radon(p, full.Archive(p['workspace_root']))
    output = Path(output) if output else Path(draft).with_name('radon0-full-segment0-profile.json')
    with adaptation(): value = full.seal(draft, output)
    queue = output.with_name('radon0-full-queue.json')
    write_new(queue, {'status': 'SEALED', 'automatic_retry': False, 'profiles': [descriptor(output)],
        'state_path': str(Path(p['output']).parent / 'radon0-full-queue-state.json'),
        'controller_lock': str(queue.with_suffix('.lock'))})
    return {**value, 'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(queue)}


def dispatch(command, *args):
    if command in ('activate', 'queue'):
        profiles = read(args[0])['profiles']
        require(profiles and all(verified(ref)['card'] == 'radon0' for ref in profiles),
                'Fourier queues retain their original dispatcher')
    else:
        require(read(args[0])['card'] == 'radon0', 'Fourier profiles retain their original dispatcher')
    with adaptation(): return getattr(full, command)(*args)


def make_manifest(root, profiles, output):
    archive = full.Archive(root); selected = [descriptor(archive.path(p)) for p in profiles]
    require(len({r['sha256'] for r in selected}) == len(selected), 'Duplicate selected formal profile')
    cards = Counter()
    with adaptation():
        for ref in selected:
            p = archive.ref(ref); science(p, archive.ref); verify_dependencies(p)
            require(p['phase'] == 'full', 'Qualification cannot enter the formal denominator'); cards[p['card']] += 1
    require(set(cards) == set(full.CARDS), 'Manifest requires original Fourier7200 plus Radon2256')
    write_new(output, {'schema': NAME + '/segments-v1', 'status': 'READY_FOR_READ_ONLY_AUDIT',
        'profiles': selected, 'qualification_in_formal_denominator': False, 'execution_policy': POLICY})
    return {'status': 'EXPLICIT_MANIFEST_PREPARED', 'manifest': descriptor(output), 'new_model_calls': 0}


def _manifest_placement(root, manifest):
    archive = full.Archive(root); selected = read(archive.path(manifest))
    require(selected['execution_policy'] == POLICY, 'Wrong mixed-deployment formal manifest')
    proofs = []; gate = None; fourier = set()
    with adaptation():
        for ref in selected['profiles']:
            p = archive.ref(ref); science(p, archive.ref); verify_dependencies(p)
            if p['card'] in provisional.FOURIER:
                fourier.add(p['qualification_summary']['sha256']); continue
            current = (p['prefetch_finite_plan'], p['qualification_summary'])
            require(gate is None or gate == current, 'Radon formal segments use different finite-point gates')
            gate = current
            proofs.append({'profile': ref, **_placement_evidence(p, archive.ref)})
    require(gate is not None, 'No Radon prefetch formal segment selected')
    _, summary, _, _ = finite_gate(*gate, archive.ref)
    require(fourier == {summary['fourier_gate']['sha256']}, 'Formal Fourier profiles use a different original gate')
    return gate, summary, proofs


def audit_full(root, manifest):
    gate, finite, proofs = _manifest_placement(root, manifest)
    with adaptation(): value = results.full(root, manifest)
    path = Path(root) / 'runs/reproducibility' / NAME / 'analysis/summary.json'; summary = read(path)
    summary.update(qualification=gate[1], prefetch_finite_plan=gate[0], qualification_status=qualification.PASS,
        original_Fourier_launch_gate=finite['fourier_gate'], execution_policy=POLICY,
        prefetch_full_deployment_proofs=proofs,
        qualification_scope={'Fourier_calls': 384, 'Radon_total_calls': 192,
            'preserved_UVA_prefix_calls': finite['preserved_uva_prefix_calls'],
            'prefetch_calls': finite['prefetch_executed_model_calls'],
            'cold1': 'UVA_PREFIX_PLUS_PREFETCH_REMAINDER', 'cold2': 'COMPLETE_PREFETCH96',
            'two_full_prefetch_cold96_claimed': False})
    summary['limits'][0] = ('One formal pass; qualification is Fourier384 plus Radon192: '
        'cold1 preserves its closed UVA prefix and prefetch remainder; cold2 is complete prefetch96.')
    atomic_json(path, summary)
    return {**value, 'summary': descriptor(path), 'configs': summary['configs'], 'new_model_calls': 0}


def compare(root, qwen, cosmos, nemo):
    path = Path(root) / 'runs/reproducibility' / NAME / 'analysis/summary.json'; summary = read(path)
    require(summary['status'] == 'COMPLETE_PASS' and summary['execution_policy'] == POLICY
            and summary['qualification_status'] == qualification.PASS,
            'Mixed-deployment formal audit is not complete')
    manifest = summary['segment_manifest']
    archive = full.Archive(root); archive.ref(manifest)
    gate, _, proofs = _manifest_placement(root, manifest['path'])
    def canonical(value):
        if isinstance(value, dict):
            return {key: str(archive.path(item)) if key == 'path' and 'sha256' in value
                    else canonical(item) for key, item in value.items()}
        if isinstance(value, list): return [canonical(item) for item in value]
        return value
    require(summary['qualification'] == gate[1] and summary['prefetch_finite_plan'] == gate[0]
            and canonical(summary['prefetch_full_deployment_proofs']) == canonical(proofs),
            'Formal comparison lost its actual segment/gate/placement bindings')
    from . import gemma_x_ram_full as prior
    value = prior.compare(root, qwen, cosmos, nemo)
    comparison_path = Path(value['summary']['path']); comparison = read(comparison_path)
    comparison.update(gemma_execution_policy=POLICY, gemma_qualification=gate[1],
                      gemma_qualification_scope=summary['qualification_scope'])
    atomic_json(comparison_path, comparison)
    return {**value, 'summary': descriptor(comparison_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare-radon', 'stage', 'seal', 'activate', 'queue', 'run',
        'supervise', 'pause', 'prepare-resume', 'audit-closed', 'make-manifest', 'audit-full', 'compare'))
    for name in ('root', 'qualification-plan', 'qualification-summary', 'output-dir', 'draft', 'output',
                 'plan', 'profile', 'control', 'manifest', 'qwen', 'cosmos', 'nemo'):
        parser.add_argument('--' + name, action='append' if name == 'profile' else 'store')
    a = parser.parse_args()
    if a.command == 'prepare-radon': value = prepare_radon(a.root, a.qualification_plan, a.qualification_summary, a.output_dir)
    elif a.command == 'stage': value = stage(a.draft, a.output)
    elif a.command == 'make-manifest': value = make_manifest(a.root, a.profile, a.output)
    elif a.command == 'audit-full': value = audit_full(a.root, a.manifest)
    elif a.command == 'compare': value = compare(a.root, a.qwen, a.cosmos, a.nemo)
    elif a.command == 'seal': value = dispatch('seal', a.draft, a.output)
    elif a.command == 'pause':
        require(Path(a.control).name == 'radon0-control.json', 'Fourier controls retain their original dispatcher')
        value = full.lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': value = dispatch('prepare_resume', a.profile[0], a.output)
    elif a.command == 'audit-closed':
        _, terminal, raw = dispatch('closed', a.profile[0])
        value = {'status': 'CLOSED_PASS', 'terminal_status': terminal['status'], 'accepted': len(raw), 'new_model_calls': 0}
    else: value = dispatch(a.command, a.profile[0] if a.command in ('run', 'supervise') else a.plan)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
