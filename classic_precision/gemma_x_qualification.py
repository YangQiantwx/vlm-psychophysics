"""Finite, same-source Gemma X cross-card qualification before classic full run.

This wrapper adds admission, pause and real-wait handling to the existing X
runner. It does not alter generation, the processor, or model mathematics.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import tempfile

from . import official_x_runner as base
from . import nemo_x_full as lifecycle
from . import gemma_x_full as full
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

NAME = full.NAME


def science(p, resolve=verified):
    selection = resolve(p['qualification_selection'])
    auth = resolve(p['execution_authorization'])
    require(auth['status'] == 'AUTHORIZED' and auth['maximum_qualification_calls'] == 576
            and auth['qualification_selection'] == p['qualification_selection'], 'Unbound qualification scope')
    require(selection['task'] == 'T3' and selection['angle_deg'] == 1.5
            and selection['n'] == 96 and selection['correct'] == 67
            and selection['source_identity_verified'] and not selection['selection_uses_X_answers'],
            'Qualification point must be selected from historical A before X')
    require(p['assigned_ids'] == selection['assigned_ids'] and p['maximum_model_calls'] == 96
            and p['cold'] in (1, 2) and p['card'] in full.CARDS
            and p['phase'] == 'qualification' and p['model'] == 'gemma_x_full'
            and p['base_model'] == 'gemma' and not p['automatic_retry']
            and not p['formal_auto_resume'], 'Wrong finite qualification worker')
    e = p['engine_kwargs']
    require(e['max_model_len'] == 65536 and e['kv_cache_memory_bytes'] == 48 * 1024**3
            and e['max_num_batched_tokens'] == 49152 and e['max_num_seqs'] == 1
            and e['enforce_eager'] and e['skip_mm_profiling']
            and not e['enable_prefix_caching'] and e['mm_processor_cache_gb'] == 0
            and e['cpu_offload_gb'] == 0 and e['quantization'] == 'modelopt'
            and e['dtype'] == e['kv_cache_dtype'] == 'bfloat16'
            and not e.get('kernel_config'), 'Changed validated Gemma runtime policy')
    require(e['mm_processor_kwargs'] == p['request_mm_processor_kwargs'] == {
        'do_resize': False, 'do_sample_frames': False, 'max_soft_tokens': 140},
        'Native no-resize processor settings changed')
    require(p['request_adapter']['module'] == 'classic_precision.gemma_x_full_input'
            and p['request_adapter']['function'] == 'create'
            and p['vision_validation_hook']['module'] == 'classic_precision.gemma_x_full_input'
            and p['vision_validation_hook']['function'] == 'validate_vision'
            and not p.get('bootstrap'), 'Unexpected input/forward adaptation')
    require(p['choices_by_task'] == {t: ['LEFT', 'RIGHT'] for t in full.COUNTS}, 'Changed scientific choices')
    for k, v in {'vllm': '0.33.0rc1.dev175+gf4fd1ab68', 'torch': '2.14.0',
                 'transformers': '5.17.0', 'flashinfer-python': '0.7.2.post1'}.items():
        require(p['expected_packages'].get(k) == v, 'Changed SDK identity: ' + k)
    require(p.get('media_io_kwargs', base.MEDIA) == base.MEDIA, 'Changed official all-frame reader')
    catalog = resolve(p['source_catalog'])['items']
    for sid in p['assigned_ids']:
        v = catalog[sid]; full.source_check(v, v)
        require(v['task'] == 'T3' and v['row']['source_parameters']['angle_deg'] == 1.5,
                'Foreign qualification source')
    require(Counter((catalog[s]['row']['source_parameters']['tilt_sign'], catalog[s]['row']['truth'])
                    for s in p['assigned_ids']) == {(k, side): 24 for k in (-1, 1) for side in ('LEFT', 'RIGHT')},
            'Qualification side/sign/onset allocation differs')


def verify_code(p):
    require(p['operational_wrapper']['sha256'] == file_sha256(__file__), 'Qualification wrapper changed')
    for name, actual in full.dependency_sources().items():
        require(p['full_dependencies'][name]['sha256'] == actual['sha256'], 'Changed input dependency: ' + name)
    require(p['qualification_lifecycle']['sha256'] == file_sha256(full.__file__), 'Lifecycle admission changed')


def seal(draft, output):
    p = read(draft); science(p)
    p['operational_wrapper'] = descriptor(__file__)
    p['qualification_lifecycle'] = descriptor(full.__file__)
    p['full_dependencies'] = full.dependency_sources()
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temporary = f.name
    try: return base.seal(temporary, output)
    finally: Path(temporary).unlink()


def run(profile):
    p = read(profile); science(p); verify_code(p)
    storage = full.storage_admission(p)
    atomic_json(Path(p['output']).with_suffix('.storage.json'), storage)
    base.worker_identity = full.worker_identity
    base.Boundary = lifecycle.Boundary
    return base.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    original = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try: return worker.supervise(profile)
    finally: worker.subprocess.Popen = original


def activate(queue_path):
    q = read(queue_path)
    require(q['status'] == 'SEALED' and q['automatic_retry'] is False, 'Unsealed qualification queue')
    for ref in q['profiles']:
        p = verified(ref); science(p); verify_code(p)
        c = read(p['control_file']); out = Path(p['output'])
        require(c['generation'] == p['control_generation'] and not out.exists()
                and not out.with_suffix('.launch.json').exists()
                and not any((Path(p['claim_root']) / (s + '.json')).exists() for s in p['assigned_ids']),
                'Already attempted, stale or claimed qualification')
        full.storage_admission(p)
    for ref in q['profiles']:
        p = verified(ref); c = read(p['control_file'])
        c.update(paused=False, activated_queue=descriptor(queue_path)); atomic_json(p['control_file'], c)
    return {'status': 'ACTIVATED_NOT_LAUNCHED', 'queue': descriptor(queue_path), 'model_calls': 0}


def queue(path):
    q = read(path); require(q['status'] == 'SEALED' and len(q['profiles']) == 2, 'Two fresh starts required')
    for ref in q['profiles']:
        p = verified(ref); science(p); verify_code(p)
        require(not read(p['control_file'])['paused'], 'Qualification remains paused')
        result = supervise(ref['path']); print(json.dumps(result), flush=True)
        require(result['returncode'] == 0 and result['terminal_status'] == 'COMPLETE'
                and result['real_wait_observed'] and result['owned_engines_exited']
                and result['original_physical_lease_released'], 'Qualification did not close; no retry')
    return {'status': 'COMPLETE_QUALIFICATION_QUEUE', 'profiles': q['profiles']}


def summarize(root, plan_path):
    """Read all six original ledgers; compare whole outputs, never only scores."""
    from . import official_x_results as evidence
    from . import gemma_x_full_results as checks
    root = Path(root); arc = full.Archive(root); plan = read(plan_path)
    selection = arc.ref(plan['selection']); ids = selection['assigned_ids']
    require(plan['schema'] == full.QUALIFICATION_SCHEMA and len(plan['workers']) == 6,
            'Expected six explicitly selected cold workers')
    require(Counter((w['card'], w['cold']) for w in plan['workers']) ==
            {(card, cold): 1 for card in full.CARDS for cold in (1, 2)}, 'Duplicate/missing cold worker')
    observed = {}; proofs = []; first_deep = []
    for worker in plan['workers']:
        p = arc.ref(worker['profile']); science(p, arc.ref); verify_code(p)
        require(p['card'] == worker['card'] and p['cold'] == worker['cold'], 'Worker identity differs')
        out = arc.path(p['output'])
        require(read(out/'result.json')['status'] == 'COMPLETE', 'Qualification must fully close')
        with checks.gemma_admission():
            _, proof = evidence.audit_worker(arc, {'profile': worker['profile'],
                'phase': 'qualification', 'variant': 'X_NATIVE_COMPACT'})
        catalog = arc.ref(p['source_catalog'])['items']; table = arc.ref(p['reference_prompt_catalog'])
        deep = evidence.unique(evidence.journal(out/'deep-observations.jsonl'))
        admissions = evidence.unique(evidence.journal(out/'input-admissions.jsonl'))
        records = {}
        for raw in lifecycle.rows(out/'raw.jsonl'):
            sid = raw['sample_id']
            checks.native_input_admission(p, catalog[sid], raw, deep.get(sid), admissions.get(sid), table)
            records[sid] = {k: raw[k] for k in ('raw_answer', 'output_token_ids', 'finish_reason',
                'stop_reason', 'truth', 'question', 'actual_prompt_token_ids')}
        require(list(records) == ids and set(deep) == set(admissions) == {ids[0]},
                'Qualification has missing/extra raw or actual GPU admissions')
        for runtime in proof['runtime']: full.validate_runtime(runtime)
        storage = read(out.with_suffix('.storage.json'))
        require(storage['status'] == 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN'
                and storage['profile_sha256'] == digest(p), 'Qualification storage proof missing')
        observed[(p['card'], p['cold'])] = records; proofs.append(proof)
        first_deep.append({'worker': p['worker'], 'sample_id': ids[0],
                           'receipt': descriptor(out/'input-admissions.jsonl')})
    anchor = observed[('fourier0', 1)]; output_diffs = []; prompt_diffs = []
    signature_keys = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason', 'truth', 'question')
    for key, records in observed.items():
        for sid in ids:
            if any(records[sid][k] != anchor[sid][k] for k in signature_keys):
                output_diffs.append({'card': key[0], 'cold': key[1], 'sample_id': sid})
            if records[sid]['actual_prompt_token_ids'] != anchor[sid]['actual_prompt_token_ids']:
                prompt_diffs.append({'card': key[0], 'cold': key[1], 'sample_id': sid})
    a_path = arc.path(selection['historical_A']['path'])
    require(file_sha256(a_path) == selection['historical_A']['sha256'], 'Historical A table changed')
    a_rows = {r['sample_id']: r for r in csv.DictReader(a_path.open()) if r['sample_id'] in ids}
    require(len(a_rows) == 96 and sum(int(r['correct']) for r in a_rows.values()) == 67, 'A point differs')
    pairs = [{'sample_id': sid, 'truth': anchor[sid]['truth'], 'A_answer': a_rows[sid]['answer'],
              'X_answer': anchor[sid]['raw_answer'].strip(), 'A_correct': bool(int(a_rows[sid]['correct'])),
              'X_correct': anchor[sid]['raw_answer'].strip() == anchor[sid]['truth']} for sid in ids]
    out = root/'runs/reproducibility'/NAME
    status = full.QUALIFICATION_PASS if not output_diffs and not prompt_diffs else 'FAILED_GEMMA_X_REPRODUCTION'
    summary = {'status': status, 'schema': NAME+'/qualification-summary-v1',
        'plan_sha256': file_sha256(plan_path), 'selection_sha256': plan['selection']['sha256'],
        'unique_video_ids': 96, 'executed_model_calls': 576, 'tasks': {'T3': 96}, 'closed_workers': 6,
        'full_output_differences': output_diffs, 'prompt_token_differences': prompt_diffs,
        'input_admission_status': 'PASS_ALL_CARDS_TASKS', 'input_admissions': first_deep,
        'same_card_differences': {card: [sid for sid in ids if observed[(card,1)][sid] != observed[(card,2)][sid]] for card in full.CARDS},
        'correct_by_worker': {f'{card}-cold{cold}': sum(r['raw_answer'].strip() == r['truth'] for r in rows.values()) for (card,cold),rows in observed.items()},
        'A_X': {'n': 96, 'A_correct': 67, 'X_correct': sum(r['X_correct'] for r in pairs),
                'answer_changes': sum(r['A_answer'] != r['X_answer'] for r in pairs),
                'X_reference_worker': 'fourier0-cold1', 'repeat_calls_not_pooled_as_independent_samples': True},
        'worker_proofs': proofs, 'new_model_calls_during_analysis': 0,
        'limits': 'Finite selected T3 condition; not a guarantee on arbitrary hardware or inputs.'}
    atomic_json(out/'qualification-A-X-pairs.json', pairs)
    atomic_json(out/'qualification-summary.json', summary)
    return {k: summary[k] for k in ('status','unique_video_ids','executed_model_calls','correct_by_worker','A_X')}


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('command', choices=['seal', 'run', 'supervise', 'activate', 'queue', 'pause', 'audit-closed', 'summarize'])
    a.add_argument('--profile'); a.add_argument('--draft'); a.add_argument('--output'); a.add_argument('--plan'); a.add_argument('--control'); a.add_argument('--root')
    x = a.parse_args()
    if x.command == 'seal': value = seal(x.draft, x.output)
    elif x.command == 'run': value = run(x.profile)
    elif x.command == 'supervise': value = supervise(x.profile)
    elif x.command == 'activate': value = activate(x.plan)
    elif x.command == 'queue': value = queue(x.plan)
    elif x.command == 'summarize': value = summarize(x.root, x.plan)
    elif x.command == 'pause':
        c = read(x.control); c['paused'] = True; atomic_json(x.control, c)
        value = {'status': 'PAUSE_REQUESTED', 'control': x.control}
    else:
        p, r, records = lifecycle.closed(x.profile)
        value = {'status': 'CLOSED_LEDGER_PASS', 'completed': len(records), 'terminal_status': r['status']}
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
