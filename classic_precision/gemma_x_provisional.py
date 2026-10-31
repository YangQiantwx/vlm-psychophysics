"""Explicitly authorized Fourier-first Gemma X execution while Radon is pending.

The frozen full/input/result modules remain byte-identical. Only the launch gate,
namespace, and module dispatcher are changed within a restored process context.
Fourier's original two cold starts on each card must pass a complete read-only
ledger/input/runtime audit and exact whole-output/prompt-token comparison first.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile

from . import gemma_x_full as full
from . import gemma_x_qualification as qualification
from . import gemma_x_full_results as results
from . import official_x_results as evidence
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256

NAME = 'gemma-x-provisional-20261031'
MODULE = 'classic_precision.gemma_x_provisional'
FOURIER = ('fourier0', 'fourier1')
GATE_PASS = 'PROVISIONAL_FOURIER_PASS_RADON_PENDING'
USER_MESSAGE = '我预估是可以三卡reproduce，你fourier直接gemma x协议赶紧先跑起来，大不了发现randon没法复刻再停下来如何？'
_SCIENCE = full.science
_VERIFY = full.verify_dependencies
_SEAL = full.seal
_DEPENDENCIES = full.dependency_sources()


def audit_fourier(root, plan, output):
    """Audit existing durable evidence; no model or processor is initialized."""
    arc = full.Archive(root); qpath = arc.path(plan); qp = read(qpath)
    require(qp['schema'] == full.QUALIFICATION_SCHEMA and len(qp['workers']) == 6,
            'Original six-worker qualification plan required')
    require(Counter((w['card'], w['cold']) for w in qp['workers']) ==
            {(card, cold): 1 for card in full.CARDS for cold in (1, 2)},
            'Qualification plan has duplicate/missing original workers')
    selection = arc.ref(qp['selection']); ids = selection['assigned_ids']
    selected = [w for w in qp['workers'] if w['card'] in FOURIER]
    observed = {}; proofs = []; bindings = []
    for worker in selected:
        p = arc.ref(worker['profile'])
        qualification.science(p, arc.ref); qualification.verify_code(p)
        require(p['card'] == worker['card'] and p['cold'] == worker['cold'],
                'Original qualification worker identity differs')
        out = arc.path(p['output'])
        with results.gemma_admission():
            _, proof = evidence.audit_worker(arc, {'profile': worker['profile'],
                'phase': 'qualification', 'variant': 'X_NATIVE_COMPACT'})
        catalog = arc.ref(p['source_catalog'])['items']
        table = arc.ref(p['reference_prompt_catalog'])
        deep = evidence.unique(evidence.journal(out / 'deep-observations.jsonl'))
        admissions = evidence.unique(evidence.journal(out / 'input-admissions.jsonl'))
        rows = {}
        for raw in full.lifecycle.rows(out / 'raw.jsonl'):
            sid = raw['sample_id']
            results.native_input_admission(p, catalog[sid], raw, deep.get(sid),
                                           admissions.get(sid), table)
            rows[sid] = {k: raw[k] for k in ('raw_answer', 'output_token_ids',
                'finish_reason', 'stop_reason', 'truth', 'question', 'actual_prompt_token_ids')}
        require(list(rows) == ids and set(deep) == set(admissions) == {ids[0]},
                'Complete ordered96 or first-request input proof missing')
        for runtime in proof['runtime']: full.validate_runtime(runtime)
        storage_path = out.with_suffix('.storage.json'); storage = read(storage_path)
        require(storage['status'] == 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN'
                and storage['profile_sha256'] == digest(p)
                and storage['counts']['run'] == 1 and storage['counts']['source'] == 96
                and all(x['rotational'] == 0 for xs in storage['devices'].values() for x in xs),
                'Original physical SSD proof is incomplete')
        observed[(p['card'], p['cold'])] = rows
        proofs.append({'card': p['card'], 'cold': p['cold'], 'profile': worker['profile'],
                       'ledger_proof': proof, 'physical_storage': descriptor(storage_path)})
        for name in ('raw.jsonl', 'events.jsonl', 'request-timings.jsonl',
                     'deep-observations.jsonl', 'input-admissions.jsonl', 'runtime.json',
                     'public-settings.json', 'identity.json', 'result.json', 'versions.json'):
            bindings.append(descriptor(out / name))
        bindings.extend((descriptor(out.with_suffix('.wait.json')), descriptor(storage_path)))
    require(set(observed) == {(c, k) for c in FOURIER for k in (1, 2)},
            'Four complete original Fourier cold workers required')
    anchor = observed[('fourier0', 1)]; output_diffs = []; prompt_diffs = []
    output_keys = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason', 'truth', 'question')
    for (card, cold), rows in observed.items():
        for sid in ids:
            if any(rows[sid][k] != anchor[sid][k] for k in output_keys):
                output_diffs.append({'card': card, 'cold': cold, 'sample_id': sid})
            if rows[sid]['actual_prompt_token_ids'] != anchor[sid]['actual_prompt_token_ids']:
                prompt_diffs.append({'card': card, 'cold': cold, 'sample_id': sid})
    require(not output_diffs and not prompt_diffs,
            'Fourier whole-output or complete prompt-token mismatch; stop preparation')
    value = {'schema': NAME + '/fourier-gate-v1', 'status': GATE_PASS,
        'qualification_plan': descriptor(qpath), 'plan_sha256': file_sha256(qpath),
        'selection': qp['selection'], 'selection_sha256': qp['selection']['sha256'],
        'unique_video_ids': 96, 'executed_model_calls': 384, 'closed_workers': 4,
        'tasks': {'T3': 96}, 'full_output_differences': [], 'prompt_token_differences': [],
        'correct_by_worker': {f'{c}-cold{k}': sum(r['raw_answer'].strip() == r['truth']
            for r in rows.values()) for (c, k), rows in observed.items()},
        'input_admission_status': 'PASS_FOURIER_ALL_EXISTING_RECEIPTS',
        'allowed_launch_cards': list(FOURIER), 'radon_qualification_status': 'PENDING',
        'three_card_qualification_pass_claimed': False, 'worker_proofs': proofs,
        'bindings': bindings, 'frozen_dependencies': _DEPENDENCIES,
        'user_message': USER_MESSAGE, 'new_model_calls_during_analysis': 0,
        'scientific_inputs_and_generation_changed': False,
        'qualification_calls_enter_formal_denominator': False}
    write_new(output, value)
    return {'status': GATE_PASS, 'proof': descriptor(output), 'closed_workers': 4,
            'executed_model_calls': 384, 'full_output_differences': 0,
            'prompt_token_differences': 0, 'new_model_calls': 0}


def gate(plan_ref, summary_ref, resolve=verified):
    qp = resolve(plan_ref); proof = resolve(summary_ref)
    require(proof['schema'] == NAME + '/fourier-gate-v1' and proof['status'] == GATE_PASS
            and proof['plan_sha256'] == plan_ref['sha256']
            and proof['selection_sha256'] == qp['selection']['sha256']
            and proof['unique_video_ids'] == 96 and proof['executed_model_calls'] == 384
            and proof['closed_workers'] == 4 and proof['tasks'] == {'T3': 96}
            and not proof['full_output_differences'] and not proof['prompt_token_differences']
            and proof['allowed_launch_cards'] == list(FOURIER)
            and proof['radon_qualification_status'] == 'PENDING'
            and proof['three_card_qualification_pass_claimed'] is False
            and proof['user_message'] == USER_MESSAGE, 'Invalid explicit Fourier-first gate')
    workers = proof['worker_proofs']
    require(Counter((w['card'], w['cold']) for w in workers) ==
            {(c, k): 1 for c in FOURIER for k in (1, 2)}, 'Incomplete Fourier gate workers')
    originals = {(w['card'], w['cold']): w['profile'] for w in qp['workers']}
    for w in workers:
        require(w['profile'] == originals[(w['card'], w['cold'])], 'Gate changed original worker')
        resolve(w['profile'])
    actual = full.dependency_sources()
    require(set(proof['frozen_dependencies']) == set(actual)
            and all(proof['frozen_dependencies'][k]['sha256'] == v['sha256']
                    for k, v in actual.items()), 'Frozen dependency identity changed')
    archive = getattr(resolve, '__self__', None)
    for ref in proof['bindings']:
        path = archive.path(ref['path']) if archive is not None else Path(ref['path'])
        require(file_sha256(path) == ref['sha256'], 'Original gate evidence changed: ' + str(path))
    return qp, proof


def science(p, resolve=None, require_gate=True):
    resolver = resolve or full.Archive(p['workspace_root']).ref
    if require_gate:
        require(p['card'] in FOURIER, 'Radon is reserved and cannot launch under provisional Fourier gate')
        require(p.get('provisional_adapter', {}).get('sha256') == file_sha256(__file__),
                'Provisional adapter is not source-bound')
        auth = resolver(p['execution_authorization'])
        require(auth.get('launch_policy') == 'FOURIER_FIRST_RADON_PENDING'
                and auth.get('allowed_launch_cards') == list(FOURIER)
                and auth.get('radon_full_reserved_calls') == 2256
                and auth.get('user_message') == USER_MESSAGE,
                'Explicit Fourier-first execution authorization missing')
    return _SCIENCE(p, resolve, require_gate)


def verify_dependencies(p):
    _VERIFY(p)
    require(p['provisional_adapter']['sha256'] == file_sha256(__file__),
            'Provisional deployment adapter changed')


def _seal(draft, output):
    p = read(draft); p['provisional_adapter'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temporary = stream.name
    try: return _SEAL(temporary, output)
    finally: Path(temporary).unlink()


@contextmanager
def adaptation():
    """Scope the expressly changed gate; restore every frozen module global."""
    changes = [(full, 'NAME', NAME), (full, 'MODULE', MODULE),
        (full, 'OPERATIONAL', full.OPERATIONAL | {'provisional_adapter'}),
        (full, 'qualification_gate', gate), (full, 'science', science),
        (full, 'verify_dependencies', verify_dependencies), (full, 'seal', _seal),
        (results, 'NAME', NAME), (results, 'science', science),
        (results, 'verify_dependencies', verify_dependencies)]
    prior = [(obj, key, getattr(obj, key)) for obj, key, _ in changes]
    try:
        for obj, key, value in changes: setattr(obj, key, value)
        yield
    finally:
        for obj, key, value in reversed(prior): setattr(obj, key, value)


def prepare(root, qualification_plan, qualification_summary, allocation=None):
    arc = full.Archive(root)
    gate(descriptor(qualification_plan), descriptor(qualification_summary), arc.ref)
    with adaptation():
        value = full.prepare(root, qualification_plan, qualification_summary, allocation)
    path = Path(root) / 'plans' / NAME / 'full-allocation.json'; allocation_value = read(path)
    require({c: len(x['assigned_ids']) for c, x in allocation_value['cards'].items()} ==
            {'fourier0': 3600, 'fourier1': 3600, 'radon0': 2256}, 'Accepted allocation changed')
    return {**value, 'launch_policy': 'FOURIER_FIRST_RADON_PENDING',
            'allowed_launch_cards': list(FOURIER), 'radon_reserved_calls': 2256}


def authorize(root, qualification_plan, output):
    qp = read(qualification_plan)
    value = {'status': 'AUTHORIZED', 'maximum_full_calls': 9456, 'cards': list(full.CARDS),
        'qualification_selection': qp['selection'], 'user_message': USER_MESSAGE,
        'launch_policy': 'FOURIER_FIRST_RADON_PENDING', 'allowed_launch_cards': list(FOURIER),
        'radon_full_reserved_calls': 2256, 'automatic_retry': False,
        'stop_at_clean_request_boundary_if_radon_reproduction_fails': True,
        'qualification_in_formal_denominator': False}
    write_new(output, value)
    return {'status': 'AUTHORIZED_FOURIER_FIRST', 'authorization': descriptor(output), 'new_model_calls': 0}


def dispatch(command, *args, **kwargs):
    with adaptation(): return getattr(full, command)(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('audit-fourier', 'authorize', 'prepare', 'stage',
        'seal', 'activate', 'queue', 'run', 'supervise', 'pause', 'prepare-resume', 'audit-closed'))
    for name in ('root', 'qualification-plan', 'qualification-summary', 'allocation', 'plan',
                 'card', 'authorization', 'draft', 'output', 'profile', 'control'):
        parser.add_argument('--' + name)
    a = parser.parse_args()
    if a.command == 'audit-fourier': value = audit_fourier(a.root, a.plan, a.output)
    elif a.command == 'authorize': value = authorize(a.root, a.qualification_plan, a.output)
    elif a.command == 'prepare': value = prepare(a.root, a.qualification_plan, a.qualification_summary, a.allocation)
    elif a.command == 'stage': value = dispatch('stage', a.plan, a.card, a.authorization, a.qualification_summary)
    elif a.command == 'seal': value = dispatch('seal', a.draft, a.output)
    elif a.command == 'pause': value = full.lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': value = dispatch('prepare_resume', a.profile, a.output)
    elif a.command == 'audit-closed':
        _, terminal, raw = dispatch('closed', a.profile)
        value = {'status': 'CLOSED_PASS', 'worker_status': terminal['status'], 'completed': len(raw), 'new_model_calls': 0}
    else: value = dispatch(a.command, a.profile if a.command in ('run', 'supervise') else a.plan)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
