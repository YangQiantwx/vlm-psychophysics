"""Read-only Gemma native-compact X ledgers and explicit classic9456 selection.

The common official-X closure/token/source auditor is reused. Only its Gemma
input-receipt branch and clean-paused-prefix dispatcher are specialized here;
no old profile, answer, claim, receipt or module is rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from . import official_x_results as evidence
from .official_x_results import basic, descriptor, journal, read, require, write_csv
from .storage import atomic_json, digest, file_sha256
from .gemma_x_full import NAME, CARDS, COUNTS, Archive, science, verify_dependencies


def native_raw_checks(p, item, raw, ordinal):
    """The generic official-X contracts plus the actual native compact route."""
    require(raw['profile_sha256'] == digest(p) and raw['ordinal'] == ordinal,
            'Raw profile/ordinal differs')
    require(raw['source']['item_digest'] == digest(item) and raw['source']['catalog'] == p['source_catalog']
            and evidence.receipt_media(raw) == evidence.media_identity(item), 'Raw source/reference differs')
    row = item['row']; require(raw['task'] == item['task'] and raw['truth'] == row['truth']
                              and raw['question'] == row['prompt'], 'Raw task/GT/prompt differs')
    content = []
    for part in item['content']:
        kind = part['type']
        if kind == 'text': content.append({'type': 'text', 'text': part['text']})
        else:
            require(kind in ('video', 'image'), 'Unknown full media content')
            content.append({'type': kind + '_url', kind + '_url': {'url': Path(part['path']).as_uri()}})
    require(raw['request'] == [{'role': 'user', 'content': content}], 'Actual native request differs')
    g = raw['generation']
    require(g['temperature'] == 0 and g['top_p'] == 1 and g['seed'] == 0 and g['n'] == 1
            and g['repetition_penalty'] == 1 and g['max_tokens'] == 16384
            and not g['ignore_eos'] and not g['stop'] and not g['stop_token_ids']
            and g['choice'] == row['choices'], 'Generation/EOS/choices changed')
    require(raw['finish_reason'] == 'stop' and raw['raw_answer'].strip() in row['choices']
            and raw['output_token_ids'] and raw['actual_prompt_token_ids'], 'Invalid or incomplete original output')
    eligible = item.get('accuracy_eligible', row.get('accuracy_eligible', raw['truth'] is not None))
    require(eligible is True and raw['accuracy_eligible'] == eligible
            and raw['correct'] == (raw['raw_answer'].strip() == raw['truth']), 'Accuracy/denominator differs')
    require(raw['native_reader'] and raw['native_processor'], 'Missing actual reader/processor evidence')
    contract = item['video_contract']
    for reader in raw['native_reader']:
        require(reader['dtype'] == 'uint8' and reader['shape'] == [180, contract['height'], contract['width'], 3]
                and reader['metadata']['frames_indices'] == list(range(180)), 'Reader source geometry/order differs')
    receipt = raw['request_adapter_receipt']
    require(receipt['route_label'] == 'X-native-compact-full-task-reference' and receipt['video_frames'] == 180
            and receipt['video_indices'] == list(range(180)) and receipt['video_max_soft_tokens'] == 280
            and receipt['video_effective_tokens_per_frame'] == 176
            and receipt['shared_SDK_files_modified'] is False
            and receipt['precomputed_tensor_bridge'] is False
            and receipt['model_forward_pooling_changed'] is False, 'Native compact route/geometry differs')
    require(receipt['expected_prompt_ids_sha256'] == digest(raw['actual_prompt_token_ids']),
            'Actual complete prompt differs from frozen task template')
    shapes = []
    for observed in raw['native_processor']:
        for phase in ('before_native_dtype', 'after_native_dtype'):
            # Official native Gemma calls the HF context with pixel_values;
            # it renames the modality and compacts masked slots afterwards.
            stage = observed.get(phase, {})
            for key in ('pixel_values', 'pixel_values_videos'):
                values = stage.get(key, [])
                for value in values if isinstance(values, list) else [values]:
                    if isinstance(value, dict) and 'shape' in value: shapes.append(value['shape'])
    require(shapes and all(len(s) == 3 and s[0] == 180 and s[1] in (2520, 1584) and s[2] == 768 for s in shapes),
            'Native processor changed frame/patch geometry')
    return eligible


def native_input_admission(p, item, raw, deep, admitted, table):
    """Reconcile durable independent CPU/native CPU/GPU chunks without runtime state."""
    sid = raw['sample_id']; task = item['task']; receipt = raw['request_adapter_receipt']
    question_sha = hashlib.sha256(item['row']['prompt'].encode()).hexdigest()
    expected_prompt = table['prompt_token_ids'][task]
    require(raw['actual_prompt_token_ids'] == expected_prompt
            and receipt['expected_prompt_ids_sha256'] == table['prompt_ids_sha256'][task] == digest(expected_prompt)
            and receipt['expected_prompt_tokens'] == len(expected_prompt)
            and receipt['scientific_question_sha256'] == table['task_contracts'][task]['question_sha256'] == question_sha
            and table['task_contracts'][task]['question'] == raw['question']
            and receipt['task_prompt_table'] == p['reference_prompt_catalog'], 'Sealed task prompt/source question differs')
    require(receipt['independent_GPU_hash_scope'] == 'first_sample_per_task_per_engine'
            and receipt['independent_CPU_reference_required'] == raw['deep'], 'Independent input audit scope differs')
    if not raw['deep']:
        require(admitted is None and deep is None, 'Ordinary request has an unexpected deep admission')
        return
    require(deep['raw_digest'] == digest(raw) and deep['sample_id'] == sid, 'Deep observation is not raw-bound')
    require(admitted and admitted['schema'] == 'gemma-x-full-input-admission-v1' and admitted['status'] == 'PASS'
            and admitted['sample_id'] == sid and admitted['task'] == task
            and admitted['profile_sha256'] == digest(p) and admitted['source_item_digest'] == digest(item)
            and admitted['source_catalog'] == p['source_catalog']
            and admitted['source_video_sha256'] == item['content'][0]['sha256']
            and admitted['scientific_question_sha256'] == question_sha
            and admitted['prompt_table'] == p['reference_prompt_catalog']
            and admitted['expected_prompt_ids_sha256'] == digest(expected_prompt), 'Independent admission identity differs')
    cpu = receipt['independent_CPU_reference']; native = admitted['native_CPU_observation']
    require(cpu['shape'] == [180, 1584, 768] and cpu['dtype'] == 'torch.bfloat16'
            and cpu['positions_shape'] == [180, 1584, 2] and cpu['valid_patch_slots_per_frame'] == 1584
            and cpu['soft_tokens_per_frame'] == 176 and cpu['video_max_soft_tokens'] == 280
            and cpu['reference_tensor_submitted_to_engine'] is False
            and admitted['expected_CPU_compact_sha256'] == native['actual_native_CPU_compact_sha256'] == cpu['sha256']
            and admitted['positions_sha256'] == native['actual_native_positions_sha256'] == cpu['positions_sha256']
            and native['valid_pixels_positions_exact_equal'] is True,
            'Independent official HF and actual native CPU bytes/positions differ')
    require(admitted['actual_GPU_chunks_equal_independent_CPU'] is True
            and admitted['reference_tensor_submitted_to_engine'] is False
            and admitted['observed_scope'] == 'first_sample_per_task_per_engine'
            and admitted['frame_count'] == 180 and admitted['patch_slots_per_frame'] == 1584
            and admitted['soft_tokens_per_frame'] == 176 and admitted['video_max_soft_tokens'] == 280
            and admitted['reader_metadata'] == receipt['reader_metadata'], 'Independent GPU scope/reader differs')
    metadata = admitted['reader_metadata']
    require(metadata['frames_indices'] == list(range(180)) and metadata['total_num_frames'] == 180
            and metadata['fps'] == 60 and metadata['duration'] == 3, 'Independent reader timing/order differs')
    actual = []
    for wi, worker in enumerate(deep['observations']):
        require(worker['outer_path'] == worker['normalized_path'] == 'vision_tower.patch_embedder',
                'Different actual Gemma GPU input observation point')
        for row in worker['normalized']:
            tensor = row['tensor']; outer_index = row['outer_index']; shape = tensor['shape']
            require(tensor['dtype'] == 'torch.bfloat16' and tensor['device'].startswith('cuda')
                    and len(shape) == 3 and shape[0] > 0 and shape[1:] == [1584, 768], 'Invalid GPU compact chunk')
            outer = worker['outer'][outer_index]['tensor']
            require(outer['sha256'] == tensor['sha256'] and outer['shape'] == shape,
                    'GPU outer and normalized observation disagree')
            actual.append((wi, outer_index, tensor))
    require(len(actual) == len(admitted['chunks']) and actual, 'GPU admission chunk count differs')
    cursor = 0
    for (wi, oi, tensor), chunk in zip(actual, admitted['chunks']):
        require(chunk['worker_index'] == wi and chunk['outer_index'] == oi
                and chunk['offset'] == cursor and chunk['frame_count'] == tensor['shape'][0]
                and chunk['shape'] == tensor['shape']
                and chunk['expected_CPU_sha256'] == chunk['actual_GPU_sha256'] == tensor['sha256'],
                'Independent CPU/GPU chunk hash, shape or ordering differs')
        cursor += chunk['frame_count']
    require(cursor == 180, 'GPU chunk coverage is not all180 frames')


def clean_prefix(archive, spec, p, wait, terminal, wait_path):
    require(p['model'] == 'gemma_x_full' and p['phase'] == 'full', 'Foreign paused protocol')
    selected = terminal['completed_ids']
    require(wait['returncode'] == 0 and wait['terminal_status'] == terminal['status'] == 'PAUSED'
            and not terminal['UNCERTAIN_ids'] and selected
            and selected == p['assigned_ids'][:len(selected)]
            and terminal['unattempted_ids'] == p['assigned_ids'][len(selected):], 'Unclean paused prefix')
    return selected


@contextmanager
def gemma_admission():
    original_raw = evidence.common_raw_checks; original_prefix = evidence.closed_prefix_selection
    original_journal = evidence.journal
    def journal_dispatch(path):
        # The generic auditor's admission dispatcher expects the unrelated
        # Qwen fused-normalization schema. Native Gemma receipts are audited
        # independently below, against the unchanged original file.
        if Path(path).name == 'input-admissions.jsonl': return []
        return original_journal(path)
    evidence.common_raw_checks = native_raw_checks; evidence.closed_prefix_selection = clean_prefix
    evidence.journal = journal_dispatch
    try: yield
    finally:
        evidence.common_raw_checks = original_raw; evidence.closed_prefix_selection = original_prefix
        evidence.journal = original_journal


def audit_segment(root, profile):
    arc = Archive(root); path = arc.path(profile); p = read(path)
    science(p, arc.ref); verify_dependencies(p); out = arc.path(p['output'])
    require(not (out / 'failure.json').exists(), 'Failure/UNCERTAIN cannot enter full analysis')
    terminal = read(out / 'result.json')
    if terminal['status'] in ('PAUSED_BEFORE_LOAD', 'PAUSED') and terminal['completed_ids'] == []:
        wait_path = out.with_suffix('.wait.json'); wait = read(wait_path)
        require(wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
                and wait['original_physical_lease_released'] and wait['profile_sha256'] == digest(p)
                and wait['terminal_status'] == terminal['status']
                and wait['completed_ids'] == terminal['completed_ids'] == []
                and not terminal['UNCERTAIN_ids'], 'Empty pause did not close cleanly')
        before_load = terminal['status'] == 'PAUSED_BEFORE_LOAD'
        if before_load: require(terminal['model_load_started'] is False, 'Before-load pause loaded a model')
        for name in ('raw.jsonl', 'events.jsonl'):
            require(not (out / name).exists() or not journal(out / name), 'Before-load pause contains attempts')
        for sid in p['assigned_ids']:
            claim_path = arc.path(p['claim_root']) / (sid + '.json')
            if claim_path.exists():
                require(read(claim_path)['profile_sha256'] != digest(p), 'Before-load pause owns a scientific claim')
        return [], {'profile': descriptor(path), 'wait': descriptor(wait_path), 'accepted': 0,
                    'terminal_status': terminal['status'], 'model_load_started': not before_load}
    if p.get('continuation_parent'):
        parent = arc.ref(p['continuation_parent']); wait = arc.ref(p['continuation_wait'])
        original_wait = read(arc.path(parent['output']).with_suffix('.wait.json'))
        previous = read(arc.path(parent['output']) / 'result.json')
        require(wait == original_wait and wait['profile_sha256'] == digest(parent)
                and wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
                and wait['original_physical_lease_released'] and previous['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD')
                and not previous['UNCERTAIN_ids'] and wait['completed_ids'] == previous['completed_ids']
                and p['assigned_ids'] == parent['assigned_ids'][len(previous['completed_ids']):],
                'Continuation does not preserve its original clean closure and suffix')
    spec = {'profile': descriptor(path), 'phase': 'full', 'variant': 'X_NATIVE_COMPACT'}
    if terminal['status'] == 'PAUSED': spec['paused_prefix_successor'] = True
    with gemma_admission(): records, proof = evidence.audit_worker(arc, spec)
    from .gemma_x_full import validate_runtime
    for runtime in proof['runtime']: validate_runtime(runtime)
    expected_deep = {}; catalog = arc.ref(p['source_catalog'])['items']
    for row in records: expected_deep.setdefault(row['task'], row['sample_id'])
    require({r['sample_id'] for r in records if r['deep']} == set(expected_deep.values()),
            'Every encountered task must have exactly its first actual GPU admission')
    deep = evidence.unique(journal(out / 'deep-observations.jsonl'))
    admissions = evidence.unique(journal(out / 'input-admissions.jsonl'))
    require(set(deep) == set(admissions) == set(expected_deep.values()), 'First-task input/deep admission set differs')
    table = arc.ref(p['reference_prompt_catalog'])
    require(table['schema'] == 'gemma-x-full-task-prompts-v1' and set(table['task_contracts']) == set(COUNTS),
            'Wrong three-task prompt reference')
    from .nemo_x_full import rows
    for raw in rows(out / 'raw.jsonl'):
        sid = raw['sample_id']
        native_input_admission(p, catalog[sid], raw, deep.get(sid), admissions.get(sid), table)
    claims = {s for s in p['assigned_ids'] if (arc.path(p['claim_root']) / (s + '.json')).exists()
              and read(arc.path(p['claim_root']) / (s + '.json'))['profile_sha256'] == digest(p)}
    require(claims == {r['sample_id'] for r in records}, 'Claim without qualified answer; no retry')
    storage_path = out.with_suffix('.storage.json'); storage = read(storage_path)
    require(storage['status'] == 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN'
            and storage['profile_sha256'] == digest(p) and storage['counts']['run'] == 1
            and storage['counts']['source'] == len(p['assigned_ids'])
            and all(leaf['rotational'] == 0 for leaves in storage['devices'].values() for leaf in leaves),
            'Physical SSD startup admission is absent or not profile-bound')
    proof['physical_storage'] = descriptor(storage_path); proof['terminal_status'] = terminal['status']
    return records, proof


def full(root, manifest):
    """Select every original closed segment explicitly; never latest-file glob."""
    root = Path(root); arc = Archive(root); plans = root / 'plans' / NAME; run = root / 'runs/reproducibility' / NAME
    selected = read(manifest)
    require(selected['schema'] == NAME + '/segments-v1'
            and selected['status'] == 'READY_FOR_READ_ONLY_AUDIT', 'Explicit SHA-bound segment selection required')
    allocation = read(plans / 'full-allocation.json'); plan = read(plans / 'plan.json')
    require(allocation['task_counts'] == plan['task_counts'] == COUNTS
            and allocation['maximum_new_full_calls'] == plan['maximum_full_calls'] == 9456, 'Classic scope changed')
    records = []; proofs = []; by_card = {c: [] for c in CARDS}; profile_refs = selected['profiles']
    require(len({x['sha256'] for x in profile_refs}) == len(profile_refs), 'Duplicate selected profile')
    for ref in profile_refs:
        p = arc.ref(ref); require(p['phase'] == 'full', 'Qualification cannot enter formal denominator')
        rows, proof = audit_segment(root, ref['path']); records.extend(rows); proofs.append(proof)
        by_card[p['card']].append((p, ref, rows, proof))
    for card in CARDS:
        segments = sorted(by_card[card], key=lambda x: x[0]['control_generation'])
        require(segments and [p['control_generation'] for p, _, _, _ in segments] == list(range(len(segments))),
                'Missing initial segment or continuation generation')
        require(segments[0][0]['assigned_ids'] == allocation['cards'][card]['assigned_ids'], 'Initial card allocation changed')
        for index, (p, ref, _, proof) in enumerate(segments):
            if index:
                require(p['continuation_parent'] == segments[index - 1][1], 'Continuation lineage differs')
            require(proof['terminal_status'] in (('COMPLETE',) if index == len(segments) - 1 else ('PAUSED', 'PAUSED_BEFORE_LOAD')),
                    'Nonfinal segment must retain its original clean PAUSED state')
        card_ids = [r['sample_id'] for _, _, rows, _ in segments for r in rows]
        require(card_ids == allocation['cards'][card]['assigned_ids'], 'Selected card ledgers do not reconstruct ordered allocation')
    ids = [r['sample_id'] for r in records]
    require(len(ids) == len(set(ids)) == 9456 and Counter(r['task'] for r in records) == COUNTS,
            'Missing, duplicate or foreign formal answer')
    out = run / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'canonical-records.json', records); write_csv(out / 'predictions.csv', records)
    from .cosmos_x_full_results import cell_key
    cells = {}
    for row in records: cells.setdefault(cell_key(row), []).append(row)
    require(len(cells) == 169, 'Formal condition grid differs from accepted169 cells')
    write_csv(out / 'cells.csv', [{'cell': list(k), **basic(rows)} for k, rows in sorted(cells.items())])
    summary = {'schema': NAME + '/results-v1', 'status': 'COMPLETE_PASS', 'model': 'gemma',
               'precision': 'official-mixed-NVFP4', 'protocol': 'X_NATIVE_COMPACT', 'unique_videos': 9456,
               'configs': len(cells),
               'new_model_calls': 0, 'tasks': {t: basic(r for r in records if r['task'] == t) for t in COUNTS},
               'segment_manifest': descriptor(manifest), 'allocation': descriptor(plans / 'full-allocation.json'),
               'qualification': arc.ref(profile_refs[0])['qualification_summary'], 'workers': proofs,
               'qualification_in_formal_denominator': False,
               'limits': ['One full pass; the fixed96 two-cold cross-card qualification is finite.',
                          'Native compact input accounting/padding removal is an explicit process-local adaptation.',
                          'Historical A results are excluded from these X scores.']}
    summary['outputs'] = {p.name: descriptor(p) for p in out.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(out / 'summary.json', summary)
    return {k: summary[k] for k in ('status', 'unique_videos', 'tasks')}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', required=True)
    parser.add_argument('--profile'); parser.add_argument('--manifest'); a = parser.parse_args()
    if a.profile:
        records, proof = audit_segment(a.root, a.profile)
        result = {'status': 'CLOSED_PASS', 'completed': len(records), 'terminal_status': proof['terminal_status'], 'new_model_calls': 0}
    else: result = full(a.root, a.manifest)
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
