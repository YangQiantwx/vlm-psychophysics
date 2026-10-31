"""Read-only audits and exact-token gate for the finite classic Qwen T1/T3 run."""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import json
from pathlib import Path

from . import official_x_results as shared
from .official_x_results import Archive, audit_worker, basic, journal, unique, write_csv
from .refinement_worker import descriptor, read, require
from .storage import atomic_json, digest, file_sha256

NAME = 'qwen-x-remaining-t13-20261030'


def compared(raw):
    from .qwen_remaining_t13 import COMPARE_FIELDS
    result = {k: raw[k] for k in COMPARE_FIELDS if k != 'source_media'}
    result['source_media'] = [[m['type'], m['sha256']] for m in raw['source']['media']]
    return result


def compare(reference, raw):
    from .qwen_remaining_t13 import COMPARE_FIELDS
    require(raw['sample_id'] in reference['records'], 'Foreign qualification ID')
    actual = compared(raw); old = reference['records'][raw['sample_id']]
    return {k: actual[k] == old[k] for k in COMPARE_FIELDS}


@contextmanager
def prefix_admission():
    previous = shared.closed_prefix_selection
    def selection(archive, spec, p, wait, terminal, wait_path):
        require(p['model'] == 'qwen_x_qsa' and p['phase'] == 'full', 'Foreign paused protocol')
        ids = terminal['completed_ids']
        require(wait['returncode'] == 0 and wait['terminal_status'] == terminal['status'] == 'PAUSED'
                and not terminal['UNCERTAIN_ids'] and ids and ids == p['assigned_ids'][:len(ids)]
                and terminal['unattempted_ids'] == p['assigned_ids'][len(ids):], 'Unclean paused prefix')
        return ids
    shared.closed_prefix_selection = selection
    try: yield
    finally: shared.closed_prefix_selection = previous


def audit_segment(root, profile_path):
    from .qwen_remaining_t13 import science, qualified
    arc = Archive(root); path = arc.path(profile_path); p = read(path); science(p, arc.ref)
    out = arc.path(p['output']); terminal = read(out / 'result.json'); wait = read(out.with_suffix('.wait.json'))
    require(not (out / 'failure.json').exists() and not terminal['UNCERTAIN_ids'], 'Failed/uncertain output cannot be selected')
    require(terminal['status'] in ('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD') and wait['returncode'] == 0
            and wait['real_wait_observed'] and wait['owned_engines_exited'] and wait['original_physical_lease_released']
            and wait['profile_sha256'] == digest(p) and wait['terminal_status'] == terminal['status'], 'Lifecycle not cleanly closed')
    if p.get('continuation_parent'):
        parent = arc.ref(p['continuation_parent']); old_out = arc.path(parent['output']); old_terminal = read(old_out / 'result.json')
        old_wait = arc.ref(p['continuation_wait'])
        require(old_wait == read(old_out.with_suffix('.wait.json')) and old_terminal['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD')
                and old_wait['returncode'] == 0 and old_wait['real_wait_observed'] and old_wait['owned_engines_exited']
                and old_wait['original_physical_lease_released'] and not old_terminal['UNCERTAIN_ids']
                and p['retained_prefix_length'] == len(old_terminal['completed_ids'])
                and p['assigned_ids'] == old_terminal.get('unattempted_ids', parent['assigned_ids']), 'Continuation parent/closure differs')
    if p['phase'] == 'full': validate_gate(p, arc.ref)
    selected = terminal['completed_ids']
    require(selected == p['assigned_ids'][:len(selected)] and wait['completed_ids'] == selected, 'Wait/result ordered prefix differs')
    claims = {s for s in p['assigned_ids'] if (arc.path(p['claim_root']) / (s + '.json')).exists()}
    require(claims == set(selected), 'Claim without qualified raw; no retry')
    if not selected:
        require(terminal['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD')
                and not (out / 'raw.jsonl').exists() and not (out / 'events.jsonl').exists(), 'Empty prefix has dispatched evidence')
        return [], {'profile': descriptor(path), 'wait': descriptor(out.with_suffix('.wait.json')), 'accepted': 0}
    spec = {'profile': descriptor(path), 'phase': p['phase'], 'variant': 'X_QSA_NO_FLASHINFER_AUTOTUNE'}
    if terminal['status'] == 'PAUSED': spec['paused_prefix_successor'] = True
    with prefix_admission(): records, proof = audit_worker(arc, spec)
    for value in proof['runtime']: qualified.validate_runtime(value)
    public = proof['public_settings']
    require(public['request_mm_processor_kwargs'] == {'do_resize': False, 'do_sample_frames': False}
            and not public.get('request_mm_processor_kwargs_by_task') and not public.get('request_adapter')
            and public['LLM_kwargs']['media_io_kwargs'] == {'video': {'video_backend': 'opencv', 'num_frames': 180, 'fps': -1}},
            'Actual native reader/request policy differs')
    if p.get('operational_wrapper'):
        require(p['operational_wrapper']['sha256'] == p['implementation_sources']['classic_precision/qwen_remaining_t13.py'], 'Wrapper/source set differs')
    return records, proof


def validate_gate(p, resolve):
    from .qwen_remaining_t13 import science, COMPARE_FIELDS
    gate = resolve(p['qualification_gate']); reference = resolve(p['reference_panel'])
    require(gate['schema'] == NAME + '/qualification-v1' and gate['status'] == 'PASS_FORMAL_X48_THREE_CARDS'
            and gate['plan_sha256'] == p['experiment_plan']['sha256'] and gate['new_model_calls'] == 144
            and gate['cards'] == ['fourier0', 'fourier1', 'radon0'] and gate['unique_video_ids'] == 48
            and gate['compared_fields'] == reference['comparison_fields'] and gate['all_comparisons_equal'] is True
            and gate['differences'] == [] and gate['reference_panel_sha256'] == p['reference_panel']['sha256']
            and len(gate['workers']) == 3 and gate['technical_status'] == 'COMPLETE_PASS', 'Qualification gate is not exact and complete')
    # Every gate certifies the SHA-bound profile/raw/wait proof, not a score only.
    require({w['card'] for w in gate['workers']} == set(gate['cards'])
            and all(w['accepted'] == 48 and all(w[k].get('sha256') for k in ('profile', 'raw', 'wait'))
                    for w in gate['workers']), 'Qualification proof binding incomplete')
    # Both hosts retain the complete finite qualification evidence. Archive maps
    # the other host's original paths without rewriting sealed profile bytes.
    bound_archive = getattr(resolve, '__self__', None)
    arc = Archive(getattr(bound_archive, 'root', p['workspace_root']))
    for worker in gate['workers']:
        qp = arc.ref(worker['profile']); science(qp, arc.ref)
        require(qp['phase'] == 'qualification' and qp['card'] == worker['card']
                and qp['assigned_ids'] == reference['ids']
                and qp['experiment_plan']['sha256'] == p['experiment_plan']['sha256'], 'Foreign qualification proof')
        expected_out = arc.path(qp['output'])
        require(arc.path(worker['raw']['path']) == expected_out / 'raw.jsonl'
                and arc.path(worker['wait']['path']) == expected_out.with_suffix('.wait.json'), 'Qualification evidence path differs')
        require(file_sha256(arc.path(worker['raw']['path'])) == worker['raw']['sha256'], 'Qualification raw SHA changed')
        wait = arc.ref(worker['wait'])
        require(wait['profile_sha256'] == digest(qp) and wait['returncode'] == 0
                and wait['real_wait_observed'] and wait['owned_engines_exited']
                and wait['original_physical_lease_released'] and wait['terminal_status'] == 'COMPLETE'
                and wait['completed_ids'] == reference['ids'], 'Qualification wait/owned exit/lease differs')
        raw = journal(arc.path(worker['raw']['path']))
        require([x['sample_id'] for x in raw] == reference['ids']
                and all(x['profile_sha256'] == digest(qp) and all(compare(reference, x).values()) for x in raw),
                'Qualification full output/reference differs')
        require(len(COMPARE_FIELDS) == len(gate['compared_fields']), 'Incomplete comparison contract')


def qualification(root, manifest=None):
    from .qwen_remaining_t13 import CARDS, COMPARE_FIELDS, science
    root = Path(root); arc = Archive(root); plans = root / 'plans' / NAME; run = root / 'runs/reproducibility' / NAME
    plan = read(plans / 'plan.json'); ref = arc.ref(plan['reference_panel']); ids = plan['qualification_ids']
    require(ref['status'] == 'SEALED_EXISTING_FORMAL_X48' and ref['ids'] == ids and len(ids) == 48,
            'Sealed formal reference changed')
    paths = [plans / (c + '-qualification-T3-segment0-profile.json') for c in CARDS] if manifest is None else [arc.path(x['path']) for x in read(manifest)['profiles']]
    require(len(paths) == 3, 'Qualification must contain exactly three workers')
    proofs = []; pairs = []; seen = set(); bindings = []
    for path in paths:
        p = read(path); science(p, arc.ref)
        require(p['phase'] == 'qualification' and p['card'] not in seen and p['assigned_ids'] == ids
                and p['experiment_plan']['sha256'] == file_sha256(plans / 'plan.json'), 'Foreign/repeated qualification worker')
        seen.add(p['card']); records, proof = audit_segment(root, path)
        require(len(records) == 48, 'Qualification worker incomplete')
        out = arc.path(p['output']); raw = unique(journal(out / 'raw.jsonl'))
        for sid in ids:
            equal = compare(ref, raw[sid]); pairs.append({'card': p['card'], 'sample_id': sid,
                **{k + '_equal': v for k, v in equal.items()}, 'all_equal': all(equal.values()), 'raw_digest': digest(raw[sid])})
        proofs.append({**proof, 'card': p['card']})
        bindings.extend(proof[k] for k in ('profile', 'raw', 'wait'))
    require(seen == set(CARDS) and len(pairs) == 144, 'Qualification coverage differs')
    differences = [x for x in pairs if not x['all_equal']]
    result = {'schema': NAME + '/qualification-v1', 'status': 'HOLD_FULL_OUTPUT_DIFFERENCES' if differences else 'PASS_FORMAL_X48_THREE_CARDS',
        'technical_status': 'COMPLETE_PASS', 'new_model_calls': 144, 'unique_video_ids': 48, 'cards': list(CARDS),
        'plan_sha256': file_sha256(plans / 'plan.json'), 'reference_panel_sha256': plan['reference_panel']['sha256'],
        'compared_fields': list(COMPARE_FIELDS), 'all_comparisons_equal': not differences, 'differences': differences,
        'workers': proofs, 'evidence_bindings': bindings, 'automatic_retry': False,
        'scope': 'One fresh engine on each card; one fixed complete classic T2 point. This gate does not establish full T1/T3 reproducibility.'}
    # A Mac-generated gate must remain usable on both hosts. Preserve the sealed
    # profile contents and canonicalize only these derived evidence locations.
    from .qwen_remaining_t13 import ROOTS
    def portable(value):
        if isinstance(value, str) and value.startswith(str(root) + '/'):
            return ROOTS['fourier'] + value[len(str(root)):]
        if isinstance(value, list): return [portable(x) for x in value]
        if isinstance(value, dict): return {k: portable(x) for k, x in value.items()}
        return value
    result = portable(result)
    run.mkdir(parents=True, exist_ok=True); atomic_json(run / 'qualification-summary.json', result)
    write_csv(run / 'qualification-paired.csv', pairs)
    return {'status': result['status'], 'differences': len(differences), 'summary': descriptor(run / 'qualification-summary.json')}


def full(root, manifest):
    from .qwen_remaining_t13 import CARDS, GROUPS, COUNTS, TOTAL
    require(manifest, 'An explicit SHA-bound segment manifest is required')
    root = Path(root); arc = Archive(root); plans = root / 'plans' / NAME; run = root / 'runs/reproducibility' / NAME
    plan = read(plans / 'plan.json'); spec = read(manifest)
    require(spec['schema'] == NAME + '/segments-v1' and spec['plan_sha256'] == file_sha256(plans / 'plan.json'), 'Foreign segment manifest')
    selected = {}; records = []; proofs = []
    for ref in spec['profiles']:
        p = arc.ref(ref); key = (p['card'], p['resource_group']); require(p['phase'] == 'full', 'Qualification in main scores')
        selected.setdefault(key, []).append((p, ref))
    require(set(selected) == {(c, g) for c in CARDS for g in GROUPS}, 'Missing/foreign worker group')
    for (card, group), segments in selected.items():
        combined = []; ordered = sorted(segments, key=lambda pair: pair[0]['control_generation'])
        for n, (p, ref) in enumerate(ordered):
            require(p['control_generation'] == n, 'Segment generation gap')
            if n:
                require(arc.ref(p['continuation_parent']) == ordered[n-1][0], 'Segment parent differs')
            rows, proof = audit_segment(root, ref['path']); combined += rows; proofs.append(proof)
            terminal = read(arc.path(p['output']) / 'result.json')
            require(terminal['status'] == ('COMPLETE' if n == len(ordered)-1 else 'PAUSED')
                    or n < len(ordered)-1 and terminal['status'] == 'PAUSED_BEFORE_LOAD', 'Segment not terminal/closed')
        require([r['sample_id'] for r in combined] == plan['cards'][card]['groups'][group], 'Card/group union/order differs')
        require({f.stem for f in arc.path(p['claim_root']).glob('*.json')} == {r['sample_id'] for r in combined}, 'Foreign/unresolved claim ledger')
        records += combined
    require(len(records) == len({r['sample_id'] for r in records}) == TOTAL
            and Counter(r['task'] for r in records) == COUNTS, 'Main scope incomplete/repeated/foreign')
    out = run / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'canonical-records.json', records); write_csv(out / 'predictions.csv', records)
    summary = {'schema': NAME + '/results-v1', 'status': 'COMPLETE_CLASSIC_QWEN_T13_X', 'unique_videos': TOTAL,
        'tasks': {t: basic(r for r in records if r['task'] == t) for t in COUNTS}, 'workers': proofs,
        'segment_manifest': descriptor(manifest), 'plan': descriptor(plans / 'plan.json'),
        'qualification_outputs_included': False, 'natural_outputs_included': False}
    summary['outputs'] = {p.name: descriptor(p) for p in out.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(out / 'summary.json', summary)
    return {'status': summary['status'], 'unique_videos': TOTAL, 'tasks': summary['tasks'], 'summary': descriptor(out / 'summary.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', required=True)
    p.add_argument('--qualification', action='store_true'); p.add_argument('--manifest'); p.add_argument('--profile')
    a = p.parse_args()
    if a.profile:
        rows, proof = audit_segment(a.root, a.profile); result = {'status': 'CLOSED_PASS', 'n': len(rows), 'proof': proof}
    elif a.qualification: result = qualification(a.root, a.manifest)
    else: result = full(a.root, a.manifest)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
