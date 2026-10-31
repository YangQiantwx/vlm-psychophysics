"""Transfer the never-dispatched Radon2256 to two qualified Fourier cards.

The original Fourier7200 allocation, frozen runners, answers and qualification
gate are immutable. This independent namespace adds exactly1128 calls per card.
The Radon reproduction failure remains a failure, never a three-card PASS.
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
from . import gemma_x_provisional as provisional
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

NAME = 'gemma-x-fourier-takeover-20261031'
MODULE = 'classic_precision.gemma_x_fourier_takeover'
POLICY = 'FOURIER_ONLY_ORIGINAL7200_PLUS_NEVER_DISPATCHED_RADON2256'
FOURIER = provisional.FOURIER
DONOR_PASS = 'PASS_ZERO_FORMAL_CALLS_FENCED_RADON_RESERVED2256'
FOURIER_PASS = 'PASS_FOURIER_ONLY_TWO_CARDS_TWO_COLD'
FAILED_ID = 't3_248e40cecbc199eaf28f'
EXTRA = {'fourier_takeover_adapter', 'parent_experiment_plan', 'donor_admission',
         'takeover_predecessor'}
_SCIENCE = full.science
_VERIFY = full.verify_dependencies
_SEAL = full.seal


def _path(resolve, value):
    archive = getattr(resolve, '__self__', None)
    return archive.path(value) if archive is not None else Path(value)


def _same_ref(arc, left, right):
    return (left['sha256'] == right['sha256']
            and arc.path(left['path']).resolve() == arc.path(right['path']).resolve())


def portable_descriptor(path, root):
    """Hash actual bytes while retaining the canonical workspace alias."""
    root = Path(os.path.abspath(root)); path = Path(os.path.abspath(path))
    require(path.is_relative_to(root), 'Portable evidence must be under the canonical workspace root')
    return {'path': str(path), 'sha256': file_sha256(path)}


def _parent(plan, canonical=None):
    require(plan['schema'] == provisional.NAME + '/execution-v1'
            and plan['maximum_full_calls'] == 9456 and plan['task_counts'] == full.COUNTS
            and set(plan['cards']) == set(full.CARDS), 'Wrong original parent scope')
    assigned = {c: plan['cards'][c]['assigned_ids'] for c in full.CARDS}
    require({c: len(ids) for c, ids in assigned.items()} ==
            {'fourier0': 3600, 'fourier1': 3600, 'radon0': 2256}, 'Original allocation changed')
    flat = [sid for ids in assigned.values() for sid in ids]
    require(len(flat) == len(set(flat)) == 9456, 'Duplicate or overlapping original allocation')
    if canonical is not None:
        require(set(flat) == set(canonical)
                and Counter(v['task'] for v in canonical.values()) == full.COUNTS,
                'Original allocation is not the canonical9456')
    return assigned


def _failure(value):
    failed = value['reproduction_failure']; wait = value['wait']; outer = value['outer_wait']
    require(value['status'] == 'FAILED_REPRODUCTION_RAW_PRESERVED_REAL_CLOSURE'
            and value['raw_count'] == 8 and value['same_inode_lease_free'] is True
            and failed['sample_id'] == FAILED_ID and failed['match'] is False
            and failed['original_scientific_output_preserved'] is True
            and failed['complete_prompt_token_ids_equal'] is True
            and failed['actual_output']['raw_answer'] == 'LEFT'
            and failed['reference_output']['raw_answer'] == 'RIGHT'
            and failed['differing_output_fields'] == ['raw_answer', 'output_token_ids']
            and wait['returncode'] == outer['returncode'] == 1
            and wait['real_wait_observed'] and outer['real_wait_observed']
            and wait['owned_engines_exited'] and wait['original_physical_lease_released'],
            'Actual preserved Radon reproduction failure/real closure is missing')


def donor_zero_state(draft, resolve):
    """Inspect only this host's formal namespace, never qualification claims."""
    control_path = _path(resolve, draft['control_file']); control = read(control_path)
    require(draft['card'] == 'radon0' and draft['phase'] == 'full'
            and draft['control_generation'] == 0 and control['generation'] == 1
            and control['paused'] is True, 'Original donor must be fenced0->1 and paused')
    claim_root = _path(resolve, draft['claim_root'])
    require(not claim_root.is_symlink(), 'Donor claim directory cannot be a symlink')
    claims = list(claim_root.iterdir()) if claim_root.exists() else []
    require(not claims, 'Reserved Radon scope already contains a formal claim')
    run = _path(resolve, draft['output']).parent
    plans = control_path.parent
    # The initial draft/control are expected; any actual profile or dispatcher
    # artefact means this is no longer a never-dispatched donor.
    forbidden = []
    if run.exists(): forbidden.extend(p for p in run.iterdir() if p.name.startswith('radon0'))
    if plans.exists():
        forbidden.extend(p for p in plans.iterdir() if p.name.startswith('radon0')
                         and ('profile' in p.name or 'queue' in p.name or 'launch' in p.name))
    require(not forbidden, 'Radon formal dispatch/profile/output evidence exists')
    return {'formal_claims': 0, 'formal_outputs': 0, 'formal_dispatch_artifacts': 0,
            'control_generation': 1, 'paused': True}


def admit_donor(root, parent_plan, donor_draft, failure_evidence, user_message,
                output, fence_evidence=None):
    """Run once on Radon after the explicit fence; no Fourier archive is needed."""
    root = Path(os.path.abspath(root)); arc = full.Archive(root)
    ref = lambda path: portable_descriptor(arc.path(path), root)
    pref = ref(parent_plan); parent = arc.ref(pref)
    assigned = _parent(parent); dref = ref(donor_draft); donor = arc.ref(dref)
    require(donor['hostname'] == socket.gethostname(), 'Donor observation must run on its actual host')
    require(isinstance(user_message, str) and user_message.strip(), 'Explicit latest user message required')
    require(donor['experiment_plan']['sha256'] == pref['sha256']
            and donor['assigned_ids'] == assigned['radon0']
            and donor['maximum_model_calls'] == 2256
            and donor['source_catalog'] == parent['cards']['radon0']['source_catalog']
            and donor['canonical_catalog']['sha256'] == parent['canonical_catalog']['sha256'],
            'Donor draft is not the original reserved2256')
    state = donor_zero_state(donor, arc.ref)
    fref = ref(failure_evidence); _failure(arc.ref(fref))
    value = {'schema': NAME + '/donor-zero-call-fence-v1', 'status': DONOR_PASS,
        'parent_experiment_plan': pref, 'original_reserved_draft': dref,
        'fenced_control': ref(donor['control_file']),
        'failure_evidence': fref, 'observed_hostname': socket.gethostname(),
        'donor_card': 'radon0', 'reserved_ids_sha256': digest(assigned['radon0']),
        'reserved_calls': 2256, 'original_control_generation': 0,
        'observation': state, 'user_message': user_message,
        'new_model_calls': 0, 'qualification_calls_are_formal_calls': False,
        'three_card_qualification_pass_claimed': False,
        'fence_evidence': ref(fence_evidence) if fence_evidence else None}
    write_new(arc.path(output), value)
    return {'status': DONOR_PASS, 'donor_admission': ref(output), 'new_model_calls': 0}


def donor_gate(reference, parent_reference, resolve):
    proof = resolve(reference); parent = resolve(parent_reference); assigned = _parent(parent)
    donor = resolve(proof['original_reserved_draft']); control = resolve(proof['fenced_control'])
    require(proof['schema'] == NAME + '/donor-zero-call-fence-v1' and proof['status'] == DONOR_PASS
            and proof['parent_experiment_plan']['sha256'] == parent_reference['sha256']
            and proof['donor_card'] == donor['card'] == 'radon0'
            and donor['phase'] == 'full' and donor['assigned_ids'] == assigned['radon0']
            and donor['maximum_model_calls'] == proof['reserved_calls'] == 2256
            and proof['reserved_ids_sha256'] == digest(assigned['radon0'])
            and donor['experiment_plan']['sha256'] == parent_reference['sha256']
            and donor['source_catalog'] == parent['cards']['radon0']['source_catalog']
            and donor['canonical_catalog']['sha256'] == parent['canonical_catalog']['sha256']
            and proof['observed_hostname'] == donor['hostname']
            and proof['original_control_generation'] == donor['control_generation'] == 0
            and control['generation'] == 1 and control['paused'] is True
            and proof['three_card_qualification_pass_claimed'] is False
            and proof['qualification_calls_are_formal_calls'] is False
            and proof['new_model_calls'] == 0
            and isinstance(proof['user_message'], str) and proof['user_message'].strip(),
            'Invalid or stale never-dispatched donor proof')
    require(_path(resolve, proof['fenced_control']['path']).resolve() ==
            _path(resolve, donor['control_file']).resolve(),
            'Donor fence is bound to another control')
    require(proof['observation'] == donor_zero_state(donor, resolve),
            'Donor zero-formal-call snapshot is stale')
    _failure(resolve(proof['failure_evidence']))
    if proof.get('fence_evidence'): resolve(proof['fence_evidence'])
    return proof, parent


def paired_allocation(ids, items):
    """Keep stimulus LR pairs together; never inspect any model answer."""
    groups = {}
    for sid in ids:
        item = items[sid]; parameters = copy.deepcopy(item['row']['source_parameters'])
        for key in ('direction', 'direction_sign', 'target_side', 'reference_side'):
            parameters.pop(key, None)
        key = digest({'task': item['task'], 'parameters': parameters})
        groups.setdefault(key, []).append(sid)
    require(all(len(group) == 2 for group in groups.values()), 'Donor LR stimulus grouping is not paired')
    owners = {sid: FOURIER[index % 2] for index, group in enumerate(groups.values()) for sid in group}
    assigned = {card: [sid for sid in ids if owners[sid] == card] for card in FOURIER}
    require({c: len(s) for c, s in assigned.items()} == {'fourier0': 1128, 'fourier1': 1128},
            'Paired allocation is not1128+1128')
    return assigned


def _takeover_plan(plan, resolve):
    proof, parent = donor_gate(plan['donor_admission'], plan['parent_experiment_plan'], resolve)
    canonical = resolve(plan['canonical_catalog'])['items']; original = _parent(parent, canonical)
    require(plan['schema'] == NAME + '/execution-v1' and plan['maximum_full_calls'] == 9456
            and plan['maximum_new_model_calls'] == 2256 and plan['task_counts'] == full.COUNTS
            and set(plan['cards']) == set(full.CARDS) and plan['cards']['radon0']['assigned_ids'] == []
            and plan['canonical_catalog']['sha256'] == parent['canonical_catalog']['sha256']
            and plan['qualification_plan']['sha256'] == parent['qualification_plan']['sha256']
            and plan['execution_policy'] == POLICY and plan['automatic_retry'] is False
            and plan['selection_uses_model_answers'] is False,
            'Wrong isolated Fourier2256 takeover plan')
    expected = paired_allocation(original['radon0'], canonical)
    require(all(plan['cards'][c]['assigned_ids'] == expected[c] and plan['cards'][c]['count'] == 1128
                for c in FOURIER) and plan['cards']['radon0']['count'] == 0,
            'Changed, duplicate or foreign takeover allocation')
    flat = original['fourier0'] + original['fourier1'] + expected['fourier0'] + expected['fourier1']
    require(len(flat) == len(set(flat)) == 9456 and set(flat) == set(canonical),
            'Original7200 plus takeover2256 is not exactly canonical9456')
    qp, gate = provisional.gate(plan['qualification_plan'], plan['qualification_summary'], resolve)
    require(gate['three_card_qualification_pass_claimed'] is False, 'Three-card PASS is forbidden')
    return proof, parent, qp, gate


def science(p, resolve=None, require_gate=True):
    resolver = resolve or full.Archive(p['workspace_root']).ref
    require(p['card'] in FOURIER, 'Only the two qualified Fourier cards may run takeover calls')
    require(p.get('fourier_takeover_adapter', {}).get('sha256') == file_sha256(__file__),
            'Fourier takeover adapter is not source-bound')
    plan = resolver(p['experiment_plan']); proof, parent, qp, _ = _takeover_plan(plan, resolver)
    require(p['parent_experiment_plan'] == plan['parent_experiment_plan']
            and p['donor_admission'] == plan['donor_admission']
            and p['qualification_summary'] == plan['qualification_summary']
            and p['qualified_profile'] == qp['cards'][p['card']]['qualified_profile']
            and p['takeover_predecessor'] == plan['predecessor_profiles'][p['card']],
            'Takeover profile lost its original gate/donor/predecessor bindings')
    predecessor = resolver(p['takeover_predecessor'])
    require(predecessor['card'] == p['card'] and predecessor['phase'] == 'full'
            and predecessor['experiment_plan']['sha256'] == plan['parent_experiment_plan']['sha256']
            and predecessor['assigned_ids']
            and set(predecessor['assigned_ids']) <= set(parent['cards'][p['card']]['assigned_ids']),
            'Takeover predecessor is not the matching original Fourier allocation')
    if require_gate:
        auth = resolver(p['execution_authorization'])
        require(auth['status'] == 'AUTHORIZED' and auth['launch_policy'] == POLICY
                and auth['allowed_launch_cards'] == list(FOURIER)
                and auth['maximum_new_model_calls'] == 2256
                and auth['new_calls_by_card'] == {'fourier0': 1128, 'fourier1': 1128}
                and auth['user_message'] == proof['user_message']
                and auth['donor_admission'] == p['donor_admission']
                and auth['parent_experiment_plan'] == p['parent_experiment_plan']
                and auth['fourier_gate'] == p['qualification_summary']
                and auth['three_card_qualification_pass_claimed'] is False
                and auth['radon_launch_allowed'] is False and auth['automatic_retry'] is False
                and auth['takeover_adapter']['sha256'] == file_sha256(__file__),
                'Explicit isolated Fourier-only takeover authorization missing')
    # The frozen common science checker still owns all settings, dependencies,
    # source/GT/prompt, namespace and continuation contracts.
    return _SCIENCE(p, resolve, require_gate)


def verify_dependencies(p):
    _VERIFY(p)
    require(p['fourier_takeover_adapter']['sha256'] == file_sha256(__file__), 'Takeover adapter changed')


def _seal(draft, output):
    p = read(draft); p['fourier_takeover_adapter'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temporary = stream.name
    try: return _SEAL(temporary, output)
    finally: Path(temporary).unlink()


@contextmanager
def adaptation():
    changes = [(full, 'NAME', NAME), (full, 'MODULE', MODULE),
        (full, 'OPERATIONAL', full.OPERATIONAL | EXTRA),
        (full, 'qualification_gate', provisional.gate), (full, 'science', science),
        (full, 'verify_dependencies', verify_dependencies), (full, 'seal', _seal),
        (results, 'NAME', NAME), (results, 'science', science),
        (results, 'verify_dependencies', verify_dependencies)]
    previous = [(obj, key, getattr(obj, key)) for obj, key, _ in changes]
    try:
        for obj, key, value in changes: setattr(obj, key, value)
        yield
    finally:
        for obj, key, value in reversed(previous): setattr(obj, key, value)


def prepare(root, parent_plan, donor_admission, qualification_summary, predecessor_profiles):
    root = Path(root).resolve(); arc = full.Archive(root); plans = root / 'plans' / NAME
    require(not plans.exists(), 'Refuse to overwrite a takeover namespace')
    pref = descriptor(arc.path(parent_plan)); dref = descriptor(arc.path(donor_admission))
    proof, parent = donor_gate(dref, pref, arc.ref)
    sref = descriptor(arc.path(qualification_summary))
    qp, _ = provisional.gate(parent['qualification_plan'], sref, arc.ref)
    canonical = arc.ref(parent['canonical_catalog'])['items']; original = _parent(parent, canonical)
    assigned = paired_allocation(original['radon0'], canonical)
    predecessors = {}
    for path in predecessor_profiles:
        ref = descriptor(arc.path(path)); p = arc.ref(ref)
        require(p['card'] in FOURIER and p['card'] not in predecessors
                and p['phase'] == 'full' and p['experiment_plan']['sha256'] == pref['sha256'],
                'Each card needs one explicit original Fourier predecessor')
        with provisional.adaptation(): provisional.science(p, arc.ref); provisional.verify_dependencies(p)
        predecessors[p['card']] = ref
    require(set(predecessors) == set(FOURIER), 'Both explicit Fourier predecessor profiles are required')
    templates = {c: arc.ref(qp['cards'][c]['qualified_profile']) for c in FOURIER}
    remote_roots = {p['workspace_root'] for p in templates.values()}
    require(len(remote_roots) == 1, 'Fourier cards unexpectedly use different workspace roots')
    remote = next(iter(remote_roots))
    def ref_for(path): return full.remote_ref(arc.path(path).resolve(), root, remote)
    # All descriptors inside the new namespace target the execution root; the
    # donor evidence itself retains the original host paths for Archive mapping.
    cards = {}
    for card in FOURIER:
        path = plans / (card + '-sources.json')
        write_new(path, {'schema': NAME + '/sources-v1',
                        'items': {s: full.rebase(canonical[s], remote) for s in assigned[card]}})
        cards[card] = {'assigned_ids': assigned[card], 'count': 1128, 'source_catalog': ref_for(path)}
    cards['radon0'] = {'assigned_ids': [], 'count': 0, 'source_catalog': None}
    plan = {'schema': NAME + '/execution-v1', 'status': 'PREPARED_NOT_LAUNCHED',
        'maximum_full_calls': 9456, 'maximum_new_model_calls': 2256, 'task_counts': full.COUNTS,
        'cards': cards, 'canonical_catalog': ref_for(parent['canonical_catalog']['path']),
        'qualification_plan': ref_for(parent['qualification_plan']['path']),
        'qualification_summary': ref_for(qualification_summary),
        'qualification_summary_relative': str(arc.path(qualification_summary).resolve().relative_to(root)),
        'parent_experiment_plan': ref_for(parent_plan), 'donor_admission': ref_for(donor_admission),
        'predecessor_profiles': {c: ref_for(r['path']) for c, r in predecessors.items()},
        'execution_policy': POLICY, 'automatic_retry': False,
        'selection_uses_model_answers': False, 'qualified_outputs_in_full_curve': False,
        'three_card_qualification_pass_claimed': False}
    _takeover_plan(plan, arc.ref)
    write_new(plans / 'plan.json', plan)
    authorization = {'status': 'AUTHORIZED', 'maximum_full_calls': 9456, 'cards': list(full.CARDS),
        'qualification_selection': qp['selection'], 'launch_policy': POLICY,
        'allowed_launch_cards': list(FOURIER), 'maximum_new_model_calls': 2256,
        'new_calls_by_card': {'fourier0': 1128, 'fourier1': 1128}, 'user_message': proof['user_message'],
        'donor_admission': plan['donor_admission'], 'parent_experiment_plan': plan['parent_experiment_plan'],
        'fourier_gate': plan['qualification_summary'], 'takeover_adapter': descriptor(__file__),
        'radon_launch_allowed': False, 'three_card_qualification_pass_claimed': False,
        'automatic_retry': False, 'qualification_in_formal_denominator': False}
    write_new(plans / 'authorization.json', authorization)
    write_new(plans / 'full-allocation.json', {'schema': NAME + '/allocation-v1', 'cards': cards,
        'task_counts': full.COUNTS, 'maximum_new_full_calls': 2256,
        'full_combined_calls': 9456, 'original_Fourier_calls': 7200,
        'experiment_plan': ref_for(plans / 'plan.json'), 'parent_experiment_plan': plan['parent_experiment_plan']})
    for card in FOURIER:
        p = copy.deepcopy(templates[card])
        for key in full.OPERATIONAL | EXTRA: p.pop(key, None)
        worker = card + '-full-segment0'; control = plans / (card + '-control.json')
        write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
        p.update(worker=worker, phase='full', assigned_ids=assigned[card], maximum_model_calls=1128,
            source_catalog=cards[card]['source_catalog'], qualified_profile=qp['cards'][card]['qualified_profile'],
            canonical_catalog=plan['canonical_catalog'], qualification_plan=plan['qualification_plan'],
            qualification_summary=plan['qualification_summary'], experiment_plan=ref_for(plans / 'plan.json'),
            output=str(Path(remote) / 'runs/reproducibility' / NAME / worker),
            claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / 'claims' / card),
            pause_file=str(Path(remote) / 'plans' / NAME / (worker + '-PAUSE.json')),
            control_file=str(Path(remote) / 'plans' / NAME / control.name), control_generation=0,
            automatic_retry=False, formal_auto_resume=False, variant='X_NATIVE_COMPACT',
            execution_authorization=ref_for(plans / 'authorization.json'),
            parent_experiment_plan=plan['parent_experiment_plan'], donor_admission=plan['donor_admission'],
            takeover_predecessor=plan['predecessor_profiles'][card], fourier_takeover_adapter=descriptor(__file__))
        with adaptation(): science(p, arc.ref)
        write_new(plans / (worker + '-draft.json'), p)
    return {'status': 'PREPARED_NOT_LAUNCHED', 'plan': descriptor(plans / 'plan.json'),
        'authorization': descriptor(plans / 'authorization.json'),
        'new_calls': {'fourier0': 1128, 'fourier1': 1128}, 'new_model_calls': 0,
        'combined_exact_unique_scope': 9456, 'qualification_status': FOURIER_PASS}


def stage(plan, card):
    require(card in FOURIER, 'Radon cannot stage a takeover worker')
    plan_path = Path(plan); spec = read(plan_path); draft = plan_path.parent / (card + '-full-segment0-draft.json')
    p = read(draft); require(p['hostname'] == socket.gethostname(), 'Wrong staging host')
    require(p['experiment_plan']['sha256'] == file_sha256(plan_path), 'Draft belongs to another plan')
    require(spec['cards'][card]['assigned_ids'] == p['assigned_ids'], 'Draft allocation differs')
    with adaptation():
        return full.stage(plan, card, plan_path.parent / 'authorization.json',
                          full.Archive(p['workspace_root']).path(spec['qualification_summary']['path']))


def predecessor_closed(p):
    """Each card can hand off independently after its exact parent exits."""
    arc = full.Archive(p['workspace_root']); path = arc.path(p['takeover_predecessor']['path'])
    arc.ref(p['takeover_predecessor'])
    with provisional.adaptation(): parent, terminal, raw = full.closed(path)
    require(terminal['status'] == 'COMPLETE' and terminal['completed_ids'] == parent['assigned_ids']
            and len(raw) == parent['maximum_model_calls'], 'Original Fourier predecessor is not complete')
    wait = read(arc.path(parent['output']).with_suffix('.wait.json'))
    require(wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released'], 'Predecessor has not really exited and released its lease')
    require(parent['physical_lock'] == p['physical_lock'], 'Successor changed the original physical GPU lease')
    return {'profile': p['takeover_predecessor'], 'completed': len(raw), 'terminal_status': 'COMPLETE'}


def dispatch(command, *args):
    if command in ('activate', 'queue'):
        selected = read(args[0])['profiles']
        require(selected, 'Empty takeover queue')
        profiles = [verified(ref) for ref in selected]
    else: profiles = [read(args[0])]
    require(all(p['card'] in FOURIER and 'fourier_takeover_adapter' in p for p in profiles),
            'Dispatcher accepts only isolated Fourier takeover profiles')
    # Staging and continuation preparation are CPU-only and can precede the
    # handoff; activation and all execution entry points require real closure.
    if command in ('activate', 'queue', 'run', 'supervise'):
        for p in profiles: predecessor_closed(p)
    with adaptation(): return getattr(full, command)(*args)


def _profile_adaptation(p):
    return adaptation() if 'fourier_takeover_adapter' in p else provisional.adaptation()


def _selected(root, profiles):
    arc = full.Archive(root); selected = [descriptor(arc.path(path)) for path in profiles]
    require(len({x['sha256'] for x in selected}) == len(selected), 'Duplicate selected formal profile')
    namespaces = set(); parent_sha = set(); takeover_sha = set(); gate_sha = set()
    for ref in selected:
        p = arc.ref(ref)
        require(p['phase'] == 'full' and p['card'] in FOURIER, 'Qualification/Radon cannot enter formal selection')
        with _profile_adaptation(p): full.science(p, arc.ref); full.verify_dependencies(p)
        namespace = NAME if 'fourier_takeover_adapter' in p else provisional.NAME
        namespaces.add(namespace); gate_sha.add(p['qualification_summary']['sha256'])
        if namespace == NAME:
            parent_sha.add(p['parent_experiment_plan']['sha256']); takeover_sha.add(p['experiment_plan']['sha256'])
        else: parent_sha.add(p['experiment_plan']['sha256'])
    require(namespaces == {NAME, provisional.NAME} and len(parent_sha) == len(takeover_sha) == len(gate_sha) == 1,
            'Selection must bind one original7200, takeover2256 and original Fourier gate')
    return selected


def make_manifest(root, profiles, output):
    selected = _selected(root, profiles)
    write_new(output, {'schema': NAME + '/segments-v1', 'status': 'READY_FOR_READ_ONLY_AUDIT',
        'profiles': selected, 'execution_policy': POLICY, 'qualification_in_formal_denominator': False,
        'three_card_qualification_pass_claimed': False})
    return {'status': 'EXPLICIT_MANIFEST_PREPARED', 'manifest': descriptor(output), 'new_model_calls': 0}


def audit_full(root, manifest):
    """Reuse unchanged per-segment audit, reconstruct four complete streams."""
    root = Path(root); arc = full.Archive(root); selection = read(arc.path(manifest))
    require(selection['schema'] == NAME + '/segments-v1'
            and selection['status'] == 'READY_FOR_READ_ONLY_AUDIT'
            and selection['execution_policy'] == POLICY
            and selection['three_card_qualification_pass_claimed'] is False,
            'Explicit takeover segment selection required')
    # Resolve SHA-bound descriptors before creating any analysis artefact.
    for ref in selection['profiles']: arc.ref(ref)
    selected = _selected(root, [ref['path'] for ref in selection['profiles']])
    require([r['sha256'] for r in selected] == [r['sha256'] for r in selection['profiles']], 'Manifest changed')
    streams = {}; records = []; proofs = []; takeover = None
    for ref in selection['profiles']:
        p = arc.ref(ref); namespace = NAME if 'fourier_takeover_adapter' in p else provisional.NAME
        with _profile_adaptation(p): rows, proof = results.audit_segment(root, ref['path'])
        streams.setdefault((namespace, p['card']), []).append((p, ref, rows, proof))
        records.extend(rows); proofs.append(proof)
        if namespace == NAME: takeover = arc.ref(p['experiment_plan'])
    require(set(streams) == {(n, c) for n in (provisional.NAME, NAME) for c in FOURIER},
            'Four original/takeover card streams required')
    donor, parent, _, historical_gate = _takeover_plan(takeover, arc.ref)
    for (namespace, card), stream in streams.items():
        stream.sort(key=lambda value: value[0]['control_generation'])
        require([p['control_generation'] for p, _, _, _ in stream] == list(range(len(stream))),
                'Missing initial segment or continuation generation')
        allocation = (parent if namespace == provisional.NAME else takeover)['cards'][card]['assigned_ids']
        require(stream[0][0]['assigned_ids'] == allocation, 'Initial stream allocation changed')
        for index, (p, ref, _, proof) in enumerate(stream):
            if index:
                require(_same_ref(arc, p['continuation_parent'], stream[index - 1][1]),
                        'Continuation lineage differs')
            allowed = ('COMPLETE',) if index == len(stream) - 1 else ('PAUSED', 'PAUSED_BEFORE_LOAD')
            require(proof['terminal_status'] in allowed, 'Stream terminal sequence is incomplete')
        require([r['sample_id'] for _, _, rows, _ in stream for r in rows] == allocation,
                'Closed stream does not reconstruct its exact ordered allocation')
    for card in FOURIER:
        predecessor = streams[(provisional.NAME, card)][-1][1]
        require(all(_same_ref(arc, p['takeover_predecessor'], predecessor)
                    for p, _, _, _ in streams[(NAME, card)]),
                'Takeover does not follow the selected final original Fourier segment')
    canonical = arc.ref(takeover['canonical_catalog'])['items']; ids = [r['sample_id'] for r in records]
    require(len(ids) == len(set(ids)) == 9456 and set(ids) == set(canonical)
            and Counter(r['task'] for r in records) == full.COUNTS, 'Missing, foreign or duplicate formal answer')
    from .cosmos_x_full_results import cell_key
    cells = {}
    for row in records: cells.setdefault(cell_key(row), []).append(row)
    require(len(cells) == 169, 'Formal169-cell grid differs')
    out = root / 'runs/reproducibility' / NAME / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'canonical-records.json', records); results.write_csv(out / 'predictions.csv', records)
    results.write_csv(out / 'cells.csv', [{'cell': list(key), **results.basic(rows)}
                                         for key, rows in sorted(cells.items())])
    summary = {'schema': NAME + '/results-v1', 'status': 'COMPLETE_PASS', 'model': 'gemma',
        'precision': 'official-mixed-NVFP4', 'protocol': 'X_NATIVE_COMPACT', 'unique_videos': 9456,
        'configs': 169, 'new_model_calls': 0, 'execution_policy': POLICY,
        'tasks': {t: results.basic(r for r in records if r['task'] == t) for t in full.COUNTS},
        'segment_manifest': descriptor(arc.path(manifest)), 'workers': proofs,
        'parent_experiment_plan': takeover['parent_experiment_plan'],
        'qualification': takeover['qualification_summary'], 'qualification_status': FOURIER_PASS,
        'qualification_scope': {'cards': list(FOURIER), 'cold_workers': 4,
            'executed_model_calls': 384, 'unique_video_ids': 96,
            'tasks': {'T3': 96}, 'full_canonical_grid_qualification_claimed': False,
            'three_card_qualification_pass_claimed': False},
        'original_fourier_gate_status': historical_gate['status'],
        'donor_admission': takeover['donor_admission'], 'radon_failure_evidence': donor['failure_evidence'],
        'qualification_in_formal_denominator': False,
        'limits': ['One formal pass; Fourier-only qualification is four cold96 workers,384 calls.',
            'Radon reproduction failed; its qualification responses are preserved and excluded from formal scores.',
            'Native compact input accounting/padding removal is an explicit process-local adaptation.',
            'Historical A results are excluded from these X scores.']}
    summary['outputs'] = {p.name: descriptor(p) for p in out.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(out / 'summary.json', summary)
    return {'status': summary['status'], 'unique_videos': 9456, 'configs': 169,
            'tasks': summary['tasks'], 'summary': descriptor(out / 'summary.json'), 'new_model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('admit-donor', 'prepare', 'stage', 'seal', 'activate', 'queue',
        'run', 'supervise', 'pause', 'prepare-resume', 'audit-closed', 'make-manifest', 'audit-full'))
    for name in ('root', 'parent-plan', 'donor-draft', 'donor-admission', 'failure-evidence', 'fence-evidence',
                 'user-message', 'qualification-summary', 'plan', 'card', 'draft', 'output', 'control', 'manifest'):
        parser.add_argument('--' + name)
    parser.add_argument('--profile', action='append'); parser.add_argument('--predecessor-profile', action='append')
    a = parser.parse_args()
    if a.command == 'admit-donor':
        value = admit_donor(a.root, a.parent_plan, a.donor_draft, a.failure_evidence,
                            a.user_message, a.output, a.fence_evidence)
    elif a.command == 'prepare':
        value = prepare(a.root, a.parent_plan, a.donor_admission, a.qualification_summary, a.predecessor_profile or [])
    elif a.command == 'stage': value = stage(a.plan, a.card)
    elif a.command == 'seal': value = dispatch('seal', a.draft, a.output)
    elif a.command == 'pause':
        require(Path(a.control).parent.name == NAME and Path(a.control).name in
                {c + '-control.json' for c in FOURIER}, 'Only takeover Fourier controls may pause here')
        value = full.lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': value = dispatch('prepare_resume', a.profile[0], a.output)
    elif a.command == 'audit-closed':
        _, terminal, raw = dispatch('closed', a.profile[0])
        value = {'status': 'CLOSED_PASS', 'terminal_status': terminal['status'],
                 'accepted': len(raw), 'new_model_calls': 0}
    elif a.command == 'make-manifest': value = make_manifest(a.root, a.profile or [], a.output)
    elif a.command == 'audit-full': value = audit_full(a.root, a.manifest)
    else: value = dispatch(a.command, a.profile[0] if a.command in ('run', 'supervise') else a.plan)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
