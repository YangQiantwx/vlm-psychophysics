"""Admit Radon's reserved2256 only after real six-worker RAM qualification.

Existing Fourier profiles,9456 allocation, source catalogs, controls and ledgers
are retained. This adapter changes only Radon's qualified deployment template
and gate, and dispatches the original full-run and result auditors by card.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import csv
import json
from pathlib import Path
import socket
import tempfile

from . import gemma_x_full as full
from . import gemma_x_full_results as results
from . import gemma_x_provisional as provisional
from . import gemma_x_ram_qualification as ram
from .cosmos_x_full_results import cell_key, pair_stats
from .official_x_results import basic, write_csv
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, file_sha256

NAME = provisional.NAME
MODULE = 'classic_precision.gemma_x_ram_full'
QUALIFIED_PLAN_SHA = '3e542749ad99d23c3ee3da765b0cd5215f22ee7d144e544f4e2d41c9a31c2fe9'
ADAPTER_FIELDS = {'ram_full_adapter', 'ram_six_worker_plan', 'original_reserved_radon_draft'}
_SCIENCE = full.science
_VERIFY = full.verify_dependencies
_SEAL = full.seal
_GATE = full.qualification_gate


def require_six_status(plan, summary):
    require(summary['status'] == full.QUALIFICATION_PASS
            and summary['executed_model_calls'] == 576 and summary['closed_workers'] == 6
            and summary['unique_video_ids'] == 96 and summary['tasks'] == {'T3': 96}
            and summary['full_output_differences'] == summary['prompt_token_differences'] == []
            and summary['input_admission_status'] == 'PASS_ALL_CARDS_TASKS'
            and summary.get('official_runtime_kernel_and_scientific_input_settings_preserved') is True,
            'Real six-worker RAM qualification is not PASS; Radon formal launch remains reserved')
    require(Counter((w['card'], w['cold']) for w in plan['workers']) ==
            {(c, k): 1 for c in full.CARDS for k in (1, 2)}, 'Incomplete six-worker qualification')
    require(Counter((w['card'], w['cold']) for w in summary['ram_deployment_proofs']) ==
            {('radon0', 1): 1, ('radon0', 2): 1}, 'Missing actual RAM deployment proofs')


def qualified_ram_template(plan, template, reference):
    require(plan['cards']['radon0']['qualified_profile'] == reference
            and template['card'] == 'radon0' and template['cold'] == 2
            and template['phase'] == 'qualification'
            and template.get('ram_adapter', {}).get('sha256') == file_sha256(ram.__file__)
            and template['engine_kwargs']['cpu_offload_gb'] == ram.OFFLOAD_GB
            and template['engine_kwargs']['gpu_memory_utilization'] == 0.8
            and template['engine_kwargs']['cpu_offload_params'] == ram.FILTER
            and template['resources'] == ram.RESOURCES,
            'Formal Radon template must be the actually qualified current RAM cold2 profile')


def six_gate(plan_ref, summary_ref, resolve=verified):
    require(plan_ref['sha256'] == QUALIFIED_PLAN_SHA,
            'Only the explicitly sealed exec3 six-worker plan can qualify Radon full')
    plan = resolve(plan_ref); summary = resolve(summary_ref)
    require_six_status(plan, summary)
    _GATE(plan_ref, summary_ref, resolve)
    ref = plan['cards']['radon0']['qualified_profile']; template = resolve(ref)
    qualified_ram_template(plan, template, ref); ram.science(template, resolve); ram.verify_code(template)
    for proof in summary['ram_deployment_proofs']:
        expected = next(w['profile'] for w in plan['workers'] if w['card'] == 'radon0' and w['cold'] == proof['cold'])
        require(proof['profile'] == expected and proof['actual_offloaded_parameter_bytes'] == ram.EXPECTED_BYTES,
                'RAM qualification proof/profile differs')
        for runtime in resolve(proof['runtime']): ram.validate_placement(runtime['ram_placement'], 'after_load')
        archive = getattr(resolve, '__self__', None)
        for key in ('first_request', 'comparisons'):
            path = archive.path(proof[key]['path']) if archive else Path(proof[key]['path'])
            require(file_sha256(path) == proof[key]['sha256'], 'Actual RAM evidence changed')
    return plan, summary


def radon_allocation(profile, plan):
    allocation = plan['cards']['radon0']['assigned_ids']; ids = profile['assigned_ids']
    require(profile['card'] == 'radon0' and len(allocation) == len(set(allocation)) == 2256
            and len(ids) == len(set(ids)) == profile['maximum_model_calls']
            and set(ids) <= set(allocation), 'Wrong reserved Radon2256 identity or denominator')
    if not profile.get('continuation_parent'):
        require(profile['control_generation'] == 0 and ids == allocation,
                'Initial Radon worker must contain the original ordered2256 allocation')


def science(profile, resolve=None, require_gate=True):
    if profile['card'] in provisional.FOURIER:
        with provisional.adaptation(): return provisional.science(profile, resolve, require_gate)
    p = profile; resolver = resolve or full.Archive(p['workspace_root']).ref
    plan = resolver(p['experiment_plan']); radon_allocation(p, plan)
    require(p.get('ram_full_adapter', {}).get('sha256') == file_sha256(__file__),
            'Radon full adapter is not source-bound')
    qplan, _ = six_gate(p['ram_six_worker_plan'], p['qualification_summary'], resolver)
    require(qplan['original_qualification_plan']['sha256'] == p['qualification_plan']['sha256']
            == plan['qualification_plan']['sha256'], 'RAM qualification is not descended from the existing full plan')
    qualified_ram_template(qplan, resolver(p['qualified_profile']), p['qualified_profile'])
    auth = resolver(p['execution_authorization'])
    require(auth['launch_policy'] == 'RADON_RESERVED_AFTER_REAL_SIX_WORKER_PASS'
            and auth['allowed_launch_cards'] == ['radon0'] and auth['maximum_radon_full_calls'] == 2256
            and auth['six_worker_plan'] == p['ram_six_worker_plan']
            and auth['six_worker_summary'] == p['qualification_summary'], 'Missing conditional Radon full authorization')
    native_gate = full.qualification_gate
    def bound_gate(original_ref, summary_ref, resolve=verified):
        require(original_ref['sha256'] == qplan['original_qualification_plan']['sha256']
                and summary_ref == p['qualification_summary'], 'Wrong qualification gate supplied')
        return qplan, resolver(summary_ref)
    try:
        full.qualification_gate = bound_gate
        return _SCIENCE(p, resolve, require_gate)
    finally: full.qualification_gate = native_gate


def verify_dependencies(profile):
    if profile['card'] in provisional.FOURIER: return provisional.verify_dependencies(profile)
    _VERIFY(profile)
    require(profile['ram_full_adapter']['sha256'] == file_sha256(__file__)
            and profile['ram_adapter']['sha256'] == file_sha256(ram.__file__), 'RAM deployment source changed')


def _seal(draft, output):
    p = read(draft); require(p['card'] == 'radon0', 'Existing Fourier profiles cannot be resealed')
    p['ram_full_adapter'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temporary = stream.name
    try: return _SEAL(temporary, output)
    finally: Path(temporary).unlink()


@contextmanager
def adaptation():
    changes = [(full, 'NAME', NAME), (full, 'MODULE', MODULE),
        (full, 'OPERATIONAL', full.OPERATIONAL | ADAPTER_FIELDS | {'provisional_adapter'}),
        (full, 'science', science), (full, 'verify_dependencies', verify_dependencies),
        (full, 'seal', _seal), (full, 'worker_identity', ram.ram_worker_identity),
        (full.base, 'read_probe', ram.ram_read_probe), (results, 'NAME', NAME),
        (results, 'science', science), (results, 'verify_dependencies', verify_dependencies)]
    prior = [(obj, key, getattr(obj, key)) for obj, key, _ in changes]
    try:
        for obj, key, value in changes: setattr(obj, key, value)
        yield
    finally:
        for obj, key, value in reversed(prior): setattr(obj, key, value)


def fresh_radon(profile, archive):
    out = archive.path(profile['output']); control = read(archive.path(profile['control_file']))
    require(not out.exists() and not out.with_suffix('.launch.json').exists()
            and not out.with_suffix('.wait.json').exists(), 'Radon formal worker was already attempted; no re-dispatch')
    require(control['paused'] and control['generation'] == profile['control_generation'] == 0,
            'Initial Radon control must remain paused and unfenced')
    require(not any((archive.path(profile['claim_root']) / (sid + '.json')).exists()
                    for sid in profile['assigned_ids']), 'Reserved Radon scope already contains a claim')


def prepare_radon(root, qualification_plan, qualification_summary, output_dir=None):
    root = Path(root).resolve(); archive = full.Archive(root)
    qref = descriptor(qualification_plan); sref = descriptor(qualification_summary)
    qplan, _ = six_gate(qref, sref, archive.ref)
    old_path = root / 'plans' / NAME / 'radon0-full-segment0-draft.json'; p = read(old_path)
    fresh_radon(p, archive)
    template_ref = qplan['cards']['radon0']['qualified_profile']; template = archive.ref(template_ref)
    # Replace only the frozen deployment settings; retain all original formal scope/path bindings.
    for key in list(p):
        if key not in full.OPERATIONAL: p.pop(key)
    p.update({key: copy.deepcopy(value) for key, value in template.items() if key not in full.OPERATIONAL})
    plans = Path(output_dir) if output_dir else old_path.parent / 'radon-ram-full'
    require(not plans.exists(), 'Radon full deployment preparation already exists; refuse overwrite')
    remote = p['workspace_root']; qremote = full.remote_ref(qualification_plan, root, remote)
    sremote = full.remote_ref(qualification_summary, root, remote)
    original_auth = archive.ref(template['execution_authorization'])
    authorization = {'status': 'AUTHORIZED', 'maximum_full_calls': 9456, 'cards': list(full.CARDS),
        'qualification_selection': qplan['selection'], 'user_message': original_auth['user_message'],
        'launch_policy': 'RADON_RESERVED_AFTER_REAL_SIX_WORKER_PASS', 'allowed_launch_cards': ['radon0'],
        'maximum_radon_full_calls': 2256, 'six_worker_plan': qremote, 'six_worker_summary': sremote,
        'original_authorization': template['execution_authorization'], 'automatic_retry': False}
    auth_path = plans / 'authorization.json'; write_new(auth_path, authorization)
    p.update(qualified_profile=template_ref, qualification_summary=sremote, ram_six_worker_plan=qremote,
        ram_full_adapter=descriptor(__file__), original_reserved_radon_draft=descriptor(old_path),
        execution_authorization=full.remote_ref(auth_path, root, remote))
    with adaptation(): science(p, archive.ref)
    draft = plans / 'radon0-full-segment0-draft.json'; write_new(draft, p)
    return {'status': 'RADON2256_DRAFT_PREPARED_AFTER_REAL_SIX_PASS', 'draft': descriptor(draft),
            'authorization': descriptor(auth_path), 'assigned': 2256, 'new_model_calls': 0,
            'existing_Fourier_profiles_and_original_radon_draft_unchanged': True}


def stage(draft, output=None):
    p = read(draft); require(p['card'] == 'radon0' and p['hostname'] == socket.gethostname(), 'Wrong staging card/host')
    fresh_radon(p, full.Archive(p['workspace_root']))
    output = Path(output) if output else Path(draft).with_name('radon0-full-segment0-profile.json')
    with adaptation(): value = full.seal(draft, output)
    queue = output.with_name('radon0-full-queue.json')
    write_new(queue, {'status': 'SEALED', 'automatic_retry': False, 'profiles': [descriptor(output)],
        'state_path': str(Path(p['output']).parent / 'radon0-full-queue-state.json'),
        'controller_lock': str(queue.with_suffix('.lock'))})
    return {**value, 'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(queue)}


def dispatch(command, *args):
    with adaptation(): return getattr(full, command)(*args)


def make_manifest(root, profiles, output):
    archive = full.Archive(root); selected = [descriptor(archive.path(p)) for p in profiles]
    require(len({r['sha256'] for r in selected}) == len(selected), 'Duplicate selected formal profile')
    cards = Counter()
    with adaptation():
        for ref in selected:
            p = archive.ref(ref); science(p, archive.ref); verify_dependencies(p)
            require(p['phase'] == 'full', 'Qualification cannot enter the formal manifest'); cards[p['card']] += 1
    require(set(cards) == set(full.CARDS), 'Manifest requires Fourier7200 plus Radon2256')
    value = {'schema': NAME + '/segments-v1', 'status': 'READY_FOR_READ_ONLY_AUDIT',
             'profiles': selected, 'qualification_in_formal_denominator': False}
    write_new(output, value)
    return {'status': 'EXPLICIT_MANIFEST_PREPARED', 'manifest': descriptor(output), 'new_model_calls': 0}


def audit_full(root, manifest):
    archive = full.Archive(root); selected = read(manifest); placement = []; qualified = None; initial_gate = None
    for ref in selected['profiles']:
        p = archive.ref(ref)
        if p['card'] in provisional.FOURIER:
            initial_gate = p['qualification_summary']; continue
        qualified = p['qualification_summary']; out = archive.path(p['output'])
        terminal = read(out / 'result.json')
        if terminal['status'] == 'PAUSED_BEFORE_LOAD' and not terminal['completed_ids']:
            placement.append({'profile': ref, 'model_load_started': False}); continue
        for runtime in read(out / 'runtime.json'): ram.validate_placement(runtime['ram_placement'], 'after_load')
        deep = list(full.lifecycle.rows(out / 'deep-observations.jsonl'))
        for row in deep:
            for observed in row['observations']:
                ram.validate_placement(observed['runtime']['ram_placement'], 'after_first_budgeted_request')
        placement.append({'profile': ref, 'runtime': descriptor(out / 'runtime.json'),
                          'first_task_requests': descriptor(out / 'deep-observations.jsonl')})
    with adaptation(): value = results.full(root, manifest)
    path = Path(root) / 'runs/reproducibility' / NAME / 'analysis/summary.json'; summary = read(path)
    summary.update(qualification=qualified, original_Fourier_launch_gate=initial_gate,
        execution_policy='FOURIER_STARTED_PROVISIONALLY_RADON_ADMITTED_AFTER_REAL_SIX_PASS', ram_full_deployment_proofs=placement)
    atomic_json(path, summary)
    return {**value, 'summary': descriptor(path), 'configs': summary['configs'], 'new_model_calls': 0}


def compare(root, qwen, cosmos, nemo):
    root = Path(root); archive = full.Archive(root)
    gemma_summary = root / 'runs/reproducibility' / NAME / 'analysis/summary.json'
    sources = {'gemma': gemma_summary, 'qwen': Path(qwen), 'cosmos': Path(cosmos), 'nemo': Path(nemo)}
    models = {}; bindings = {}
    for model, path in sources.items():
        summary = read(path); ref = summary['outputs']['canonical-records.json']; rows = archive.ref(ref)
        require(summary['status'] == 'COMPLETE_PASS' and len(rows) == 9456
                and len({r['sample_id'] for r in rows}) == 9456
                and all(r['model'] == model and r['eligible']
                        and r['correct'] == (r['answer'] == r['truth']) for r in rows),
                'Incomplete, mislabeled or unscored accepted peer archive: ' + model)
        models[model] = {r['sample_id']: r for r in rows}; bindings[model] = descriptor(path)
    lookup = models['gemma']; between = []; cells = defaultdict(list)
    require(all(set(records) == set(lookup) for records in models.values()), 'Four-model ID scope differs')
    for sid, x in lookup.items():
        row = {'sample_id': sid, 'task': x['task'], 'cell': list(cell_key(x)), 'truth': x['truth']}
        for model, records in models.items():
            peer = records[sid]
            require(peer['task'] == x['task'] and peer['truth'] == x['truth']
                    and peer['parameters'] == x['parameters']
                    and list(map(tuple, peer['source_media'])) == list(map(tuple, x['source_media'])),
                    'Four-model source, condition or GT differs')
            row[model] = peer['answer']; row[model + '_correct'] = peer['correct']
        between.append(row); cells[tuple(row['cell'])].append(row)
    require(len(cells) == 169, 'Four-model condition grid differs')
    a_dir = root / 'runs/reproducibility/current-nvfp4'; a_path = a_dir / 'video-predictions.csv'
    a_summary = read(a_dir / 'summary.json')
    require(a_summary['status'] == 'COMPLETE' and file_sha256(a_path) == a_summary['video_csv_sha256'], 'Original A export changed')
    with a_path.open() as stream: historical = [r for r in csv.DictReader(stream) if r['model'] == 'gemma']
    require(len({r['sample_id'] for r in historical}) == len(historical), 'Duplicate original Gemma A ID')
    pairs = []
    for a in historical:
        x = lookup.get(a['sample_id'])
        if x is None: continue
        require(a['protocol'] == 'controlled-rgb16' and a['task'] == x['task'] and a['truth'] == x['truth']
                and a['source_video_sha256'] in [sha for kind, sha in x['source_media'] if kind == 'video'],
                'Gemma A/X source or GT differs')
        require(bool(int(a['correct'])) == (a['answer'] == a['truth']), 'Historical correctness differs')
        pairs.append({'sample_id': x['sample_id'], 'task': x['task'], 'cell': list(cell_key(x)), 'truth': x['truth'],
            'A': a['answer'], 'X': x['answer'], 'A_correct': bool(int(a['correct'])), 'X_correct': x['correct'],
            'output_tokens_changed': json.loads(a['output_token_ids']) != x['output_token_ids'],
            'A_raw_sha256': a['raw_sha256'], 'X_raw_digest': x['raw_digest']})
    output = gemma_summary.parent / 'comparison'; output.mkdir(exist_ok=True)
    a_cells = defaultdict(list)
    for row in pairs: a_cells[tuple(row['cell'])].append(row)
    write_csv(output / 'paired-Gemma-A-X.csv', pairs)
    write_csv(output / 'Gemma-A-X-cells.csv', [{'cell': list(k), **pair_stats(rows)}
                                            for k, rows in sorted(a_cells.items())])
    write_csv(output / 'four-model-predictions.csv', between)
    write_csv(output / 'four-model-cells.csv', [{'cell': list(k), 'n': len(rows),
        **{m + '_correct': sum(r[m + '_correct'] for r in rows) for m in models}} for k, rows in sorted(cells.items())])
    summary = {'status': 'COMPLETE_PAIRED_FOUR_MODEL_COMPARISON', 'unique_videos_per_X_model': 9456,
        'configs': 169, 'accepted_archives': bindings, 'historical_A': descriptor(a_path),
        'four_model_tasks': {model: {t: basic(r for r in rows.values() if r['task'] == t)
                                    for t in full.COUNTS} for model, rows in models.items()},
        'Gemma_A_X_paired': [{'task': t, **pair_stats([p for p in pairs if p['task'] == t])} for t in full.COUNTS],
        'Gemma_A_unavailable_in_X_scope': 9456 - len(pairs), 'new_model_calls': 0,
        'qualification_in_formal_denominator': False,
        'scope': 'A/X compares protocol packages; repeated qualification is separate from the9456 X denominator.'}
    summary['outputs'] = {p.name: descriptor(p) for p in output.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(output / 'summary.json', summary)
    return {'status': summary['status'], 'summary': descriptor(output / 'summary.json'), 'new_model_calls': 0}


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
    elif a.command == 'pause': value = full.lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': value = dispatch('prepare_resume', a.profile[0], a.output)
    elif a.command == 'audit-closed':
        _, terminal, raw = dispatch('closed', a.profile[0])
        value = {'status': 'CLOSED_PASS', 'terminal_status': terminal['status'], 'accepted': len(raw), 'new_model_calls': 0}
    else: value = dispatch(a.command, a.profile[0] if a.command in ('run', 'supervise') else a.plan)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
