"""Read-only reconciliation and paired reporting for the finite official-X study.

This module never imports an inference runner, changes claims, or calls a model.
An explicit analysis manifest selects accepted worker versions; retired attempts
remain evidence and cannot be selected by a latest-file heuristic.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .storage import atomic_json, digest, file_sha256

REMOTE_ROOTS = ('/home/yang/cvpr2027_benchmark', '/mnt/sda/q/cvpr2027_benchmark')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def journal(path):
    path = Path(path)
    require(path.exists(), 'Missing journal: ' + str(path))
    text = path.read_text()
    require(not text or text.endswith('\n'), 'Incomplete journal tail: ' + str(path))
    return [json.loads(line) for line in text.splitlines() if line]


def unique(rows, key='sample_id'):
    result = {x[key]: x for x in rows}
    require(len(result) == len(rows), 'Duplicate journal IDs: ' + key)
    return result


class Archive:
    def __init__(self, root):
        self.root = Path(root)

    def path(self, path):
        s = str(path)
        for prefix in REMOTE_ROOTS:
            if s.startswith(prefix + '/'):
                return self.root / s[len(prefix) + 1:]
        p = Path(s)
        return p if p.is_absolute() else self.root / p

    def ref(self, descriptor):
        path = self.path(descriptor['path'])
        require(file_sha256(path) == descriptor['sha256'], 'Identity mismatch: ' + str(path))
        return read(path)


def descriptor(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def media_identity(item):
    return [(x['type'], x['sha256']) for x in item['content'] if x['type'] in ('video', 'image')]


def receipt_media(raw):
    return [(x['type'], x['sha256']) for x in raw['source']['media']]


def common_raw_checks(profile, item, raw, ordinal):
    require(raw['profile_sha256'] == digest(profile), 'Raw/profile mismatch')
    require(raw['ordinal'] == ordinal, 'Ordinal differs from sealed allocation')
    require(raw['source']['item_digest'] == digest(item), 'Raw/catalog item mismatch')
    require(raw['source']['catalog'] == profile['source_catalog'], 'Raw/catalog binding differs')
    require(receipt_media(raw) == media_identity(item), 'Raw source/reference identity differs')
    require(raw['task'] == item['task'] and raw['truth'] == item['row'].get('truth'), 'Task or GT differs')
    require(raw['question'] == item['row']['prompt'], 'Scientific question changed')
    media = item.get('content')
    if media is None:
        media = [{'type': 'video', 'path': profile['sources'][raw['sample_id']]['path']},
                 {'type': 'text', 'text': item['row']['prompt']}]
    content = []
    for part in media:
        kind = part['type']
        if kind == 'text':
            content.append({'type': 'text', 'text': part['text']})
        elif kind in ('image', 'video'):
            content.append({'type': kind + '_url', kind + '_url': {'url': Path(part['path']).as_uri()}})
        else:
            require(kind in ('image_url', 'video_url'), 'Unknown native input content')
            content.append(part)
    require(raw['request'] == [{'role': 'user', 'content': content}], 'Actual request content/order differs')
    choices = item['row'].get('choices', profile.get('choices_by_task', {}).get(item['task']))
    gen = raw['generation']
    require(gen['temperature'] == 0 and gen['top_p'] == 1 and gen['seed'] == 0 and gen['n'] == 1,
            'Generation settings changed')
    require(not gen['ignore_eos'] and not gen['stop'] and not gen['stop_token_ids']
            and gen['choice'] == choices, 'Native EOS or output choices changed')
    require(raw['finish_reason'] == 'stop' and raw['raw_answer'].strip() in choices, 'Invalid final answer')
    require(raw['output_token_ids'] and raw['actual_prompt_token_ids'], 'Missing complete tokens')
    eligible = item.get('accuracy_eligible', item['row'].get('accuracy_eligible', raw['truth'] is not None))
    require(raw['accuracy_eligible'] == eligible, 'Accuracy denominator changed')
    expected = raw['raw_answer'].strip() == raw['truth'] if eligible else None
    require(raw['correct'] == expected, 'Saved correctness disagrees with raw and GT')
    require(raw['native_reader'] and raw['native_processor'], 'Missing actual reader/processor observations')
    c = item['video_contract']
    for reader in raw['native_reader']:
        require(reader['dtype'] == 'uint8', 'Not official uint8 decode')
        require(reader['shape'] == [180, c['height'], c['width'], 3], 'Source geometry/count differs')
        require(reader['metadata']['frames_indices'] == list(range(180)), 'Frame ordering/subsampling differs')
    model = profile.get('base_model', profile['model'])
    payloads = [observation.get(stage, {}) for observation in raw['native_processor']
                for stage in ('before_native_dtype', 'after_native_dtype')]
    if model in ('qwen', 'cosmos'):
        settings = profile['vision_observation']
        expected = [[180 // settings.get('temporal_patch_size', 2),
                     c['height'] // settings.get('patch_size', 16),
                     c['width'] // settings.get('patch_size', 16)]]
        grids = [p['video_grid_thw'].get('values') for p in payloads if 'video_grid_thw' in p]
        require(grids and all(g == expected for g in grids), 'Processor resized or subsampled the video')
    elif model == 'nemo':
        frames = [p['pixel_values_flat_video']['shape'] for p in payloads if 'pixel_values_flat_video' in p]
        require(frames and all(s == [180, 3, c['height'], c['width']] for s in frames),
                'Nemo processor geometry/count differs')
    elif model == 'gemma':
        adapter = raw['request_adapter_receipt']
        require(adapter and adapter['route_label'] == 'X-adapted' and not adapter['GPU_model_math_modified'],
                'Gemma adapter identity or scope differs')
        require(adapter['video_max_soft_tokens'] == 280
                and adapter['video_effective_tokens_per_frame'] == [196 if raw['task'] == 'T4' else 176] * 180,
                'Gemma visual token budget/count differs')
        require(adapter['expected_prompt_ids_sha256'] == digest(raw['actual_prompt_token_ids']),
                'Gemma logical prompt differs from official HF expansion')
        frames = [p['pixel_values_videos']['shape'][0] for p in payloads if 'pixel_values_videos' in p]
        require(frames and all(n == 180 for n in frames), 'Gemma processor frame count differs')
    return eligible


def closed_prefix_selection(archive, spec, profile, wait, terminal, wait_path):
    """Admit a preserved clean pause only with its SHA-bound fresh successor."""
    successor = archive.ref(spec['paused_prefix_successor'])
    require(profile['model'] == 'nemo' and profile['phase'] == 'full', 'Unexpected paused-prefix protocol')
    require(wait['returncode'] == 0 and wait['terminal_status'] == terminal['status'] == 'PAUSED'
            and not terminal['UNCERTAIN_ids'], 'Paused prefix was not cleanly closed')
    selected = terminal['completed_ids']
    require(selected and selected == profile['assigned_ids'][:len(selected)]
            and terminal['unattempted_ids'] == profile['assigned_ids'][len(selected):],
            'Paused ledger is not an exact ordered prefix')
    require(archive.ref(successor['continuation_parent']) == profile
            and archive.ref(successor['continuation_wait']) == wait
            and successor['continuation_wait']['sha256'] == file_sha256(wait_path),
            'Successor does not bind original profile and actual wait')
    require(successor['assigned_ids'] == terminal['unattempted_ids']
            and successor['control_generation'] == profile['control_generation'] + 1
            and successor['worker_token'] != profile['worker_token']
            and successor['output'] != profile['output'], 'Successor scope or fresh identity differs')
    operational = {'assigned_ids', 'media_files', 'maximum_model_calls', 'pause_file', 'worker_token',
                   'continuation_parent', 'continuation_wait', 'control_generation', 'output', 'worker'}
    require(all(profile.get(k) == successor.get(k) for k in profile.keys() | successor.keys()
                if k not in operational), 'Paused successor changed scientific configuration')
    return selected


def audit_worker(archive, spec):
    p = archive.ref(spec['profile'])
    out = archive.path(p['output'])
    wait_path = out.with_suffix('.wait.json')
    wait = read(wait_path)
    require(wait['profile_sha256'] == digest(p) and wait['real_wait_observed']
            and wait['owned_engines_exited'] and wait['original_physical_lease_released'],
            'Real wait/owned exit/original-inode lease incomplete: ' + p['worker'])
    raws = unique(journal(out / 'raw.jsonl'))
    events = journal(out / 'events.jsonl')
    intents = unique([x for x in events if x['event'] == 'INTENT'])
    passes = unique([x for x in events if x['event'] == 'TECHNICAL_PASS'])
    special = spec.get('independent_admission')
    if special:
        admission = archive.ref(special)
        require(admission['schema'] == 'official-x-input-reconciliation/v1'
                and admission['status'] == 'ACCEPTED_EXISTING_RAW_INPUT', 'Independent admission not accepted')
        require(admission['profile_digest'] == digest(p), 'Admission/profile mismatch')
        require(admission['original_wait'] == wait, 'Admission/wait mismatch')
        for binding in admission['bindings'].values():
            require(file_sha256(archive.path(binding['path'])) == binding['sha256'], 'Admission evidence changed')
        require(admission['new_scientific_calls'] == 0 and not admission['request_reissued']
                and admission['original_UNCERTAIN_and_failure_preserved']
                and admission['original_TECHNICAL_PASS_absent_preserved'], 'Original failure was not preserved')
        require(admission['same_physical_lock_inode'] and admission['physical_lock_free_at_reconciliation'],
                'Reconciliation lease evidence incomplete')
        proof = admission['input_proof']
        require(proof['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS'
                and proof['source_reader_all180'] and proof['source_frame_indices'] == list(range(180))
                and proof['CPU_to_GPU_patch_bytes_equal'], 'Reconciliation lacks full source/GPU proof')
        cpu = {x['sha256'] for x in proof['native_CPU_processor']}
        require(all(x['tensor']['sha256'] in cpu and x['tensor']['dtype'] == 'torch.uint8'
                    and x['tensor']['device'].startswith('cuda') for x in proof['actual_GPU_ingress']),
                'Recovered GPU bytes do not match native CPU payload')
        require(proof['official_normalizers'], 'No official normalizer identity')
        for norm in proof['official_normalizers']:
            require(norm['source']['sha256'] == norm['official_source_verification']['sha256'],
                    'Normalizer source identity mismatch')
        selected = [admission['sample_id']]
        require(set(selected) == set(raws) == set(intents), 'Independent admission has extra/unclosed attempts')
        require(wait['returncode'] != 0, 'Recovery admission must disclose original failure')
        timings = {}
        for sid in selected:
            require(admission['raw_digest'] == digest(raws[sid]), 'Admission/raw mismatch')
        event_status = 'INDEPENDENT_INPUT_ADMISSION_ORIGINAL_FAILURE_PRESERVED'
    elif spec.get('preclaim_refusal_admission'):
        # A stat-only source refusal happened before claim/INTENT/model dispatch.
        # Preserve its nonzero wait and missing result.json; admit only the
        # already durable TECHNICAL_PASS prefix, never the failed pending ID.
        admission = archive.ref(spec['preclaim_refusal_admission'])
        require(admission['status'] == 'PASS_PRECLAIM_SOURCE_STAT_RECONCILIATION'
                and admission['profile_digest'] == digest(p)
                and admission['original_wait'] == wait,
                'Preclaim reconciliation does not bind the original worker')
        require(p['model'] == 'cosmos' and p['phase'] == 'classic'
                and wait['returncode'] == 1 and wait['terminal_status'] == 'FAILED_OR_UNCERTAIN'
                and wait['completed_ids'] == [] and not (out / 'result.json').exists(),
                'Unexpected preclaim-refusal terminal state')
        for binding in admission['bindings'].values():
            require(file_sha256(archive.path(binding['path'])) == binding['sha256'],
                    'Preclaim reconciliation evidence changed')
        failure = read(out / 'failure.json')
        selected = admission['completed_ids']
        remaining = p['assigned_ids'][len(selected):]
        require(selected and selected == p['assigned_ids'][:len(selected)]
                and remaining == admission['never_dispatched_ids'] and remaining
                and failure['completed'] == len(selected)
                and failure['UNCERTAIN_id'] == remaining[0]
                and failure['error_type'] == 'ValueError' and failure['error'] == 'Source changed',
                'Refusal was not at the exact next source boundary')
        require(admission['new_scientific_calls'] == 0
                and admission['source_bytes_unchanged']
                and admission['stat_guard_before_claim_and_dispatch']
                and admission['same_physical_lock_inode']
                and admission['physical_lock_free_at_reconciliation'],
                'Preclaim/source/closure proof incomplete')
        claims = {s for s in p['assigned_ids']
                  if (archive.path(p['claim_root']) / (s + '.json')).exists()}
        timings = unique(journal(out / 'request-timings.jsonl'))
        require(claims == set(raws) == set(intents) == set(passes) == set(timings) == set(selected),
                'Refusal has a claimed, dispatched or uncertain remainder')
        event_status = 'TECHNICAL_PASS_PREFIX_PRECLAIM_REFUSAL_RC1_PRESERVED'
    else:
        terminal = read(out / 'result.json')
        if spec.get('paused_prefix_successor'):
            selected = closed_prefix_selection(archive, spec, p, wait, terminal, wait_path)
        elif spec.get('closed_prefix_only'):
            require(p['model']=='qwen_x_qsa' and p['phase']=='classic_T2', 'Unexpected standalone paused-prefix protocol')
            selected=terminal['completed_ids']
            require(wait['returncode']==0 and wait['terminal_status']==terminal['status']=='PAUSED'
                    and not terminal['UNCERTAIN_ids'] and selected
                    and selected==p['assigned_ids'][:len(selected)]
                    and terminal['unattempted_ids']==p['assigned_ids'][len(selected):], 'Unclean paused prefix')
            require({s for s in p['assigned_ids'] if (archive.path(p['claim_root'])/(s+'.json')).exists()}==set(selected),
                    'Paused prefix has an unresolved claim')
        else:
            require(wait['returncode'] == 0 and wait['terminal_status'] == 'COMPLETE', 'Worker not COMPLETE')
            selected = p['assigned_ids']
            require(terminal['status'] == 'COMPLETE' and terminal['completed_ids'] == selected
                    and not terminal['UNCERTAIN_ids'] and not terminal['unattempted_ids'], 'Terminal ID ledger incomplete')
        require(wait['completed_ids'] == selected, 'Wait/terminal completed IDs differ')
        timings = unique(journal(out / 'request-timings.jsonl'))
        require(set(raws) == set(intents) == set(passes) == set(timings) == set(selected), 'Per-ID ledger differs')
        event_status = 'PAUSED_PREFIX_TECHNICAL_PASS' if spec.get('paused_prefix_successor') or spec.get('closed_prefix_only') else 'WORKER_TECHNICAL_PASS'
    catalog = archive.ref(p['source_catalog'])
    identity = read(out / 'identity.json')
    require(identity['profile_sha256'] == digest(p) and identity['profile'] == p, 'Worker identity differs')
    versions = read(out / 'versions.json')
    runtime = read(out / 'runtime.json')
    public = read(out / 'public-settings.json')
    require(all(k in versions for k in ('vllm', 'torch', 'transformers', 'numpy')),
            'Core runtime version receipt incomplete')
    require(all(versions[k] == v for k, v in p['expected_packages'].items() if k in versions),
            'Recorded runtime package identity differs')
    # The frozen worker asserts every expected package before producing this
    # four-package receipt. Keep extra startup assertions explicit; they are not
    # separately recorded version strings and must not be presented as such.
    startup_asserted_packages = {k: v for k, v in p['expected_packages'].items() if k not in versions}
    require(runtime and all(r['vllm_version'] == versions['vllm']
                            and r['torch_version'].split('+')[0] == versions['torch'].split('+')[0]
                            and r['actual_kv_capacity'] >= p['minimum_kv_capacity'] for r in runtime),
            'Actual worker runtime or KV capacity differs')
    for key, value in p['engine_kwargs'].items():
        require(public['LLM_kwargs'][key] == (sorted(value) if key == 'cpu_offload_params' else value),
                'Actual public engine setting differs: ' + key)
    for key, value in p['engine_kwargs'].get('kernel_config', {}).items():
        require(all(f'{key}={value!r}' in r['kernel_config'] for r in runtime),
                'Requested official kernel configuration did not take effect: ' + key)
    deep = unique(journal(out / 'deep-observations.jsonl'))
    input_admissions = unique(journal(out / 'input-admissions.jsonl')) if (out / 'input-admissions.jsonl').exists() else {}
    selector_admissions = unique(journal(out / 'custom-selector-admissions.jsonl')) if p.get('selector_source') else {}
    records = []
    model = p.get('base_model', p['model'])
    for sid in selected:
        raw = raws[sid]
        item = catalog['items'][sid]
        ordinal = p['assigned_ids'].index(sid)
        eligible = common_raw_checks(p, item, raw, ordinal)
        claim = read(archive.path(p['claim_root']) / (sid + '.json'))
        require(claim['sample_id'] == sid and claim['profile_sha256'] == digest(p)
                and claim['worker_token'] == p['worker_token'], 'Claim/profile/raw mismatch')
        if not special:
            require(passes[sid]['raw_digest'] == digest(raw), 'Technical PASS does not bind raw')
            t = timings[sid]
            require(math.isfinite(t['end_to_end_durable_seconds']) and t['end_to_end_durable_seconds'] > 0,
                    'Invalid durable timing')
            require(t['engine_request_inclusive_seconds'] <= t['end_to_end_durable_seconds'], 'Inconsistent timing')
        if raw['deep']:
            require(sid in deep and deep[sid]['raw_digest'] == digest(raw), 'Deep proof does not bind raw')
            observations = deep[sid]['observations']
            require(observations and all('runtime_error' not in x and 'profiler_error' not in x for x in observations),
                    'Incomplete actual engine input evidence')
            if sid in input_admissions:
                accepted = input_admissions[sid]
                require(accepted['raw_digest'] == digest(raw) and accepted['observations_digest'] == digest(observations),
                        'Input admission does not bind actual deep evidence')
                require(accepted['proof']['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS'
                        and accepted['proof']['CPU_to_GPU_patch_bytes_equal']
                        and accepted['proof']['source_frame_indices'] == list(range(180)),
                        'Official raw GPU admission incomplete')
            elif not special:
                require(any(x.get('normalized') for x in observations), 'No actual normalized input or fused-input admission')
            if p.get('selector_source'):
                require(sid in selector_admissions, 'No actual custom QSA call admission')
                q = selector_admissions[sid]
                require(q['raw_digest'] == digest(raw) and q['variant'] == 'X_QSA'
                        and q['pure_official_config'] is False and q['actual_request_calls']
                        and all(n > 0 for n in q['actual_request_calls']), 'Custom selector not actually exercised')
                for receipt in q['selector_receipts']:
                    require(receipt['status'] == 'APPLIED_X_QSA'
                            and receipt['selector_source_sha256'] == p['selector_source']['sha256']
                            and receipt['actual_function_is_canonical_wrapper']
                            and receipt['native_function_verified_before_replacement']
                            and receipt['helper_status_after_import'] is None
                            and not receipt['helper_import_added_meta_path_hook']
                            and not receipt['SDK_files_modified']
                            and not receipt['forward_processor_or_weights_changed'],
                            'Custom selector identity or scope differs')
        params = item['row'].get('source_parameters', item.get('public_index', {}).get('nominal_parameters', {}))
        records.append({'model': model, 'variant': spec.get('variant', raw.get('variant', 'unspecified')),
                        'phase': spec['phase'], 'replicate': spec.get('replicate'),
                        'sample_id': sid, 'task': raw['task'], 'truth': raw['truth'],
                        'answer': raw['raw_answer'].strip(), 'raw_answer': raw['raw_answer'],
                        'correct': raw['correct'], 'eligible': eligible, 'parameters': params,
                        'analysis_group': item.get('analysis_group', 'main'),
                        'include_in_main_accuracy_curve': item.get('include_in_main_accuracy_curve', True),
                        'worker': p['worker'], 'profile_sha256': digest(p), 'raw_digest': digest(raw),
                        'source_media': receipt_media(raw), 'output_token_ids': raw['output_token_ids'],
                        'finish_reason': raw['finish_reason'], 'stop_reason': raw['stop_reason'],
                        'prompt_token_sha256': digest(raw['actual_prompt_token_ids']),
                        'prompt_tokens': len(raw['actual_prompt_token_ids']), 'deep': raw['deep'],
                        'timing': timings.get(sid), 'raw_timing': raw['timing'],
                        'adapter_timing': raw.get('request_adapter_timing'),
                        'admission': event_status})
    evidence = {'model': model, 'variant': spec.get('variant', records[0]['variant']),
                'worker': p['worker'], 'accepted': len(records),
                'profile': spec['profile'], 'raw': descriptor(out / 'raw.jsonl'),
                'wait': descriptor(wait_path), 'original_returncode': wait['returncode'],
                'admission': event_status, 'independent_admission': special,
                'preclaim_refusal_admission': spec.get('preclaim_refusal_admission'),
                'recorded_package_versions': versions,
                'extra_package_checks': {'scope': 'asserted_by_frozen_worker_before_first_request_not_separately_recorded',
                                         'expected': startup_asserted_packages},
                'runtime': runtime, 'load': read(out / 'load.json'),
                'public_settings': public}
    return records, evidence


def basic(rows):
    rows = list(rows)
    eligible = [x for x in rows if x['eligible']]
    correct = sum(x['correct'] is True for x in eligible)
    def choices(row):
        return ('YES', 'NO') if row.get('task') == 'T5' else ('LEFT', 'RIGHT')
    return {'n': len(rows), 'accuracy_n': len(eligible), 'correct': correct,
            'accuracy': correct / len(eligible) if eligible else None,
            'unscored_n': len(rows) - len(eligible),
            'invalid_n': sum(x['answer'] not in choices(x) for x in rows),
            'answers': dict(Counter(x['answer'] for x in rows)),
            'strict_format_n': sum(x['raw_answer'] in choices(x) for x in rows)}


def pair_stats(pairs):
    pairs = list(pairs)
    scored = []
    for a, b in pairs:
        require(a['truth'] == b['truth'], 'Paired GT differs')
        ae = a.get('eligible', a['truth'] is not None)
        be = b.get('eligible', b['truth'] is not None)
        require(ae == be, 'Paired accuracy eligibility differs')
        if ae:
            scored.append((a, b))
    before = sum(a['correct'] is True for a, _ in scored)
    after = sum(b['correct'] is True for _, b in scored)
    return {'n': len(pairs), 'accuracy_n': len(scored), 'unscored_n': len(pairs) - len(scored),
            'answer_changes': sum(a['answer'] != b['answer'] for a, b in pairs),
            'correct_to_wrong': sum(a['correct'] is True and b['correct'] is not True for a, b in scored),
            'wrong_to_correct': sum(a['correct'] is not True and b['correct'] is True for a, b in scored),
            'before_correct': before, 'after_correct': after,
            'before_accuracy': before / len(scored) if scored else None,
            'after_accuracy': after / len(scored) if scored else None,
            'accuracy_cost': (before - after) / len(scored) if scored else None}


def validation_tables(records, historical):
    lookup = {}
    for r in records:
        if r['phase'] == 'validation':
            key = (r['model'], r['replicate'], r['sample_id'])
            require(key not in lookup, 'Duplicate canonical validation tuple')
            lookup[key] = r
    history = {(r['model'], r['sample_id']): r for r in historical}
    summary, paired = [], []
    for model in sorted({k[0] for k in lookup}):
        variants = {r['variant'] for k, r in lookup.items() if k[0] == model}
        require(len(variants) == 1, 'Do not mix protocol variants inside a canonical model comparison')
        variant = next(iter(variants))
        for task in ('T2', 'T3'):
            a = {k[2]: v for k, v in lookup.items() if k[:2] == (model, 1) and v['task'] == task}
            b = {k[2]: v for k, v in lookup.items() if k[:2] == (model, 2) and v['task'] == task}
            require(set(a) == set(b) and len(a) == {'T2': 48, 'T3': 96}[task], 'Incomplete two-replicate cell')
            pairs = [(a[sid], b[sid]) for sid in a]
            for x, y in pairs:
                require(x['truth'] == y['truth'] and x['source_media'] == y['source_media'], 'Replicate source/GT differs')
            row = {'model': model, 'variant': variant, 'task': task, **pair_stats(pairs),
                   'complete_output_changes': sum((x['output_token_ids'], x['finish_reason'], x['stop_reason']) !=
                                                  (y['output_token_ids'], y['finish_reason'], y['stop_reason']) for x, y in pairs),
                   'prompt_token_changes': sum(x['prompt_token_sha256'] != y['prompt_token_sha256'] for x, y in pairs),
                   'replicate1_engines': len({x['worker'] for x in a.values()}),
                   'replicate2_engines': len({x['worker'] for x in b.values()})}
            old_pairs = []
            for sid, x in a.items():
                h = history[(model, sid)]
                require(h['truth'] == x['truth'], 'Historical GT differs')
                require(h['source_video_sha256'] in [sha for kind, sha in x['source_media'] if kind == 'video'],
                        'Historical/new source video differs')
                old_pairs.append((h, x))
                paired.append({'model': model, 'variant': variant, 'task': task, 'sample_id': sid, 'truth': x['truth'],
                               'C': h['answer'], 'X1': x['answer'], 'X2': b[sid]['answer'],
                               'C_correct': h['correct'], 'X1_correct': x['correct'], 'X2_correct': b[sid]['correct'],
                               'C_X1_output_equal': (h['output_token_ids'], h['finish_reason'], h['stop_reason']) ==
                                                      (x['output_token_ids'], x['finish_reason'], x['stop_reason']),
                               'C_X1_prompt_tokens_equal': h.get('prompt_token_sha256') == x['prompt_token_sha256'],
                               'X_output_equal': (x['output_token_ids'], x['finish_reason'], x['stop_reason']) ==
                                                 (b[sid]['output_token_ids'], b[sid]['finish_reason'], b[sid]['stop_reason'])})
            row['C_to_X1'] = pair_stats(old_pairs)
            row['C_to_X1']['complete_output_changes'] = sum(
                (h['output_token_ids'], h['finish_reason'], h['stop_reason']) !=
                (x['output_token_ids'], x['finish_reason'], x['stop_reason']) for h, x in old_pairs)
            row['C_to_X1']['prompt_token_changes'] = sum(
                h.get('prompt_token_sha256') != x['prompt_token_sha256'] for h, x in old_pairs)
            summary.append(row)
    return summary, paired


def validation_runtime_contracts(workers, archive):
    """Keep answer agreement separate from agreement of the tested deployment."""
    groups, seen = defaultdict(list), set()
    for worker in workers:
        profile = archive.ref(worker['profile'])
        require(worker['worker'] not in seen, 'Duplicate worker identity')
        seen.add(worker['worker'])
        groups[worker['model']].append((worker, profile))
    fields = ('actual_backends', 'actual_kv_capacity', 'cuda_version', 'kernel_config',
              'method_kernel_counts', 'mm_device_do_normalize', 'model_class',
              'runner_class', 'torch_version', 'vllm_version')
    contracts = []
    for model, entries in sorted(groups.items()):
        reference, profile = entries[0]
        runtime = [{key: part[key] for key in fields} for part in reference['runtime']]
        for other, other_profile in entries[1:]:
            require(other_profile['gpu_uuid'] == profile['gpu_uuid'], 'Repeat moved to another GPU: ' + model)
            require(other['public_settings'] == reference['public_settings'], 'Repeat inference settings differ: ' + model)
            require(other['recorded_package_versions'] == reference['recorded_package_versions'],
                    'Repeat package versions differ: ' + model)
            observed = [{key: part[key] for key in fields} for part in other['runtime']]
            require(observed == runtime, 'Repeat resolved kernel/runtime differs: ' + model)
        contracts.append({'model': model, 'status': 'MATCHED_WITHIN_VALIDATION',
                          'gpu_uuid': profile['gpu_uuid'], 'workers': len(entries),
                          'public_settings_sha256': digest(reference['public_settings']),
                          'recorded_package_versions': reference['recorded_package_versions'],
                          'resolved_runtime': runtime,
                          'scope': 'Selected T2/T3 repeated-input workers only; panel resource/host differences stay in individual worker records.'})
    return contracts


def panel_tables(records):
    rows = [r for r in records if r['phase'] == 'panel']
    require(len({(r['model'], r['sample_id']) for r in rows}) == len(rows), 'Duplicate canonical panel tuple')
    t4, t5, paired, costs = [], [], [], []
    for model in sorted({r['model'] for r in rows}):
        own = [r for r in rows if r['model'] == model]
        four = [r for r in own if r['task'] == 'T4']
        five = [r for r in own if r['task'] == 'T5']
        require(len(four) == 112 and len(five) == 192, 'Incomplete T4/T5 model panel')
        groups = defaultdict(list)
        for r in four:
            p = r['parameters']
            require((p['coherence_probability'] == 0) == (not r['eligible']), 'T4 zero eligibility differs')
            require(r['truth'] is None if not r['eligible'] else r['truth'] in ('LEFT', 'RIGHT'), 'T4 GT differs')
            kind = 'zero_bias' if not r['eligible'] else ('easy_joint' if not r['include_in_main_accuracy_curve'] else 'main')
            groups[(kind, r['parameters']['coherence_probability'])].append(r)
        expected = {('main', c): 16 for c in (.032, .064, .128, .256, .512, 1.)}
        expected.update({('zero_bias', 0.): 8, ('easy_joint', 1.): 8})
        require({k: len(v) for k, v in groups.items()} == expected, 'T4 main/zero/easy denominators differ')
        for (kind, coherence), group in sorted(groups.items()):
            seed_n = 4 if kind == 'easy_joint' else 8
            require(len({r['parameters']['seed'] for r in group}) == seed_n, 'T4 seed denominator differs')
            for r in group:
                p = r['parameters']
                require(p['dot_diameter_px'] == (48 if kind == 'easy_joint' else 3)
                        and p['speed_px_s'] == (32 if kind == 'easy_joint' else 300), 'T4 fixed size/speed differs')
            if kind != 'zero_bias':
                for seed in {r['parameters']['seed'] for r in group}:
                    require(Counter(r['truth'] for r in group if r['parameters']['seed'] == seed)
                            == {'LEFT': 1, 'RIGHT': 1}, 'T4 LR seed pair differs')
            t4.append({'model': model, 'group': kind, 'coherence': coherence, **basic(group),
                       'paired_seeds': len({r['parameters']['seed'] for r in group}),
                       'sampling_unit': 'initialization_seed',
                       'LEFT': basic([r for r in group if r['truth'] == 'LEFT']),
                       'RIGHT': basic([r for r in group if r['truth'] == 'RIGHT'])})
        lookup = {}
        for r in five:
            p = r['parameters'];key = (p['mask'], p['presentation'], p['paired_block'], r['truth'])
            require(key not in lookup, 'Duplicate T5 paired condition')
            require(p['probe_slot'] == int(p['presentation'][-1]) and p['truth'] == r['truth'], 'T5 slot or GT differs')
            require(p['exposure_frames'] == 30 and p['offset_to_onset_gap_frames'] == 30, 'T5 fixed timing differs')
            require(r['eligible'], 'T5 unexpectedly excluded from accuracy')
            lookup[key] = r
        blocks = sorted({k[2] for k in lookup})
        require(len(blocks) == 12, 'T5 onset-position block denominator differs')
        require(set(lookup) == {(mask, presentation, block, gt) for mask in ('none', 'persistent')
                for presentation in ('single1', 'dual1', 'single2', 'dual2') for block in blocks
                for gt in ('YES', 'NO')}, 'T5 paired factorial cells differ')
        for block in blocks:
            members = [r for k, r in lookup.items() if k[2] == block]
            require(len({(r['parameters']['k'], r['parameters']['position_level'],
                          r['parameters']['first_onset_frame']) for r in members}) == 1,
                    'T5 paired block changes onset/position')
        for mask in ('none', 'persistent'):
            for presentation in ('single1', 'dual1', 'single2', 'dual2'):
                group = [r for k, r in lookup.items() if k[:2] == (mask, presentation)]
                pos = [r for r in group if r['truth'] == 'YES']; neg = [r for r in group if r['truth'] == 'NO']
                require(len(pos) == len(neg) == 12, 'T5 hit/false-alarm denominator differs')
                tp = sum(r['answer'] == 'YES' for r in pos);tn = sum(r['answer'] == 'NO' for r in neg)
                fp = sum(r['answer'] == 'YES' for r in neg)
                fn = sum(r['answer'] == 'NO' for r in pos)
                t5.append({'model': model, 'mask': mask, 'presentation': presentation, **basic(group),
                           'probe_slot': int(presentation[-1]), 'paired_blocks': 12,
                           'sampling_unit': 'onset_position_block',
                           'positive_n': 12, 'negative_n': 12, 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                           'positive_invalid': len(pos) - tp - fn, 'negative_invalid': len(neg) - tn - fp,
                           'hit_rate': tp / 12, 'false_alarm_rate': fp / 12,
                           'specificity': tn / 12,
                           'balanced_accuracy': (tp / 12 + tn / 12) / 2})
        def cost_record(contrast, ps, **fields):
            stat = pair_stats(ps)
            require(stat['accuracy_n'] == 24, 'T5 paired scored denominator differs')
            positive = [(a, b) for a, b in ps if a['truth'] == 'YES']
            negative = [(a, b) for a, b in ps if a['truth'] == 'NO']
            require(len(positive) == len(negative) == 12, 'T5 paired class denominator differs')
            costs.append({'model': model, 'contrast': contrast, **fields, **stat,
                          'paired_blocks': 12, 'sampling_unit': 'onset_position_block',
                          'balanced_accuracy_cost': stat['accuracy_cost'],
                          'hit_rate_cost': sum((a['answer'] == 'YES') - (b['answer'] == 'YES') for a, b in positive) / 12,
                          'specificity_cost': sum((a['answer'] == 'NO') - (b['answer'] == 'NO') for a, b in negative) / 12,
                          'false_alarm_increase': sum((b['answer'] == 'YES') - (a['answer'] == 'YES') for a, b in negative) / 12})
            for a, b in ps:
                paired.append({'model': model, 'contrast': contrast, **fields,
                               'block': a['parameters']['paired_block'], 'truth': a['truth'],
                               'before_id': a['sample_id'], 'after_id': b['sample_id'],
                               'before': a['answer'], 'after': b['answer'],
                               'before_correct': a['correct'], 'after_correct': b['correct']})
        for mask in ('none', 'persistent'):
            for slot in (1, 2):
                ps = [(lookup[(mask, 'single' + str(slot), block, gt)],
                       lookup[(mask, 'dual' + str(slot), block, gt)]) for block in blocks for gt in ('YES', 'NO')]
                cost_record('single_to_dual', ps, mask=mask, probe_slot=slot)
        for presentation in ('single1', 'dual1', 'single2', 'dual2'):
            ps = [(lookup[('none', presentation, block, gt)], lookup[('persistent', presentation, block, gt)])
                  for block in blocks for gt in ('YES', 'NO')]
            cost_record('none_to_persistent', ps, presentation=presentation, probe_slot=int(presentation[-1]))
        for slot in (1, 2):
            block_effects = []
            for block in blocks:
                sequence = {}
                for mask in ('none', 'persistent'):
                    sequence[mask] = sum(
                        int(lookup[(mask, 'single' + str(slot), block, gt)]['correct'] is True)
                        - int(lookup[(mask, 'dual' + str(slot), block, gt)]['correct'] is True)
                        for gt in ('YES', 'NO')) / 2
                block_effects.append({'block': block, 'none_sequence_cost': sequence['none'],
                                      'persistent_sequence_cost': sequence['persistent'],
                                      'interaction': sequence['persistent'] - sequence['none']})
            costs.append({'model': model, 'contrast': 'mask_by_sequence_interaction', 'probe_slot': slot,
                          'paired_blocks': 12, 'paired_video_trials_per_condition': 24,
                          'sampling_unit': 'onset_position_block',
                          'none_sequence_cost': statistics.mean(x['none_sequence_cost'] for x in block_effects),
                          'persistent_sequence_cost': statistics.mean(x['persistent_sequence_cost'] for x in block_effects),
                          'interaction': statistics.mean(x['interaction'] for x in block_effects),
                          'positive_interaction': 'Persistent masking increases the single-to-dual cost.',
                          'block_effects': block_effects})
    return {'T4': t4, 'T5': t5, 'paired': paired, 'costs': costs}


def render_figures(summary, output):
    """Export observed T4/T5 figures from a completed, audited study summary.

    A clearly marked synthetic layout check is allowed only in the OS temporary
    directory. It is never accepted as a completed scientific study.
    """
    import tempfile
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    from .official_x_figures import render_repeatability, repeat_cells

    source = descriptor(Path(summary)) if isinstance(summary, (str, Path)) else {'sha256': digest(summary)}
    summary = read(summary) if isinstance(summary, (str, Path)) else summary
    output = Path(output).resolve()
    layout = summary.get('status') == 'SYNTHETIC_LAYOUT_CHECK_ONLY'
    require(summary.get('status') == 'COMPLETE_FINITE_STUDY'
            or (layout and output.is_relative_to(Path(tempfile.gettempdir()).resolve())),
            'Figures require a completed study; synthetic layout checks must stay temporary')
    repeat_cells(summary['validation'])
    models = ('qwen', 'cosmos', 'nemo', 'gemma')
    names = {'qwen': 'Qwen3.8 Flash Next', 'cosmos': 'Cosmos3 Super',
             'nemo': 'Nemotron 3.5 Super VL', 'gemma': 'Gemma 4 31B'}
    colors = {'qwen': '#3569ba', 'cosmos': '#147d73', 'nemo': '#a75b19', 'gemma': '#8b5a91'}
    markers = {'qwen': 'o', 'cosmos': 's', 'nemo': '^', 'gemma': 'D'}
    require({r['model'] for r in summary['T4']} == set(models)
            and {r['model'] for r in summary['T5']} == set(models), 'Four complete model panels required')
    four = {(r['model'], r['group'], r['coherence']): r for r in summary['T4']}
    five = {(r['model'], r['mask'], r['presentation']): r for r in summary['T5']}
    require(len(four) == 32 and len(five) == 32, 'Duplicate or incomplete figure cells')
    coherence = (.032, .064, .128, .256, .512, 1.)
    for model in models:
        for c in coherence:
            r = four[(model, 'main', c)]
            require(r['accuracy_n'] == 16 and r['paired_seeds'] == 8
                    and 0 <= r['correct'] <= 16, 'T4 main denominator differs')
        require(four[(model, 'zero_bias', 0.)]['accuracy_n'] == 0
                and four[(model, 'zero_bias', 0.)]['n'] == 8
                and four[(model, 'easy_joint', 1.)]['accuracy_n'] == 8, 'T4 control denominator differs')
        for mask in ('none', 'persistent'):
            for presentation in ('single1', 'dual1', 'single2', 'dual2'):
                r = five[(model, mask, presentation)]
                require(r['n'] == 24 and r['positive_n'] == r['negative_n'] == 12
                        and r['paired_blocks'] == 12, 'T5 figure denominator differs')
                require(abs(r['balanced_accuracy'] - (r['TP'] + r['TN']) / 24) < 1e-12
                        and abs(r['hit_rate'] - r['TP'] / 12) < 1e-12
                        and abs(r['false_alarm_rate'] - r['FP'] / 12) < 1e-12,
                        'T5 plotted rates disagree with counts')
    output.mkdir(parents=True, exist_ok=True)
    exports = []
    style = {'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.titlesize': 14,
             'axes.labelsize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.edgecolor': '#555555', 'axes.linewidth': .8,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}

    def title(fig, heading, subtitle):
        fig.suptitle(('SYNTHETIC LAYOUT CHECK · ' if layout else '') + heading,
                     x=.065, ha='left', y=.985, fontsize=20, fontweight='bold')
        fig.text(.065, .985 - .45 / fig.get_figheight(), subtitle,
                 ha='left', va='top', fontsize=12, color='#555555')

    def axis(ax, model, ylabel=None):
        ax.set_title(names[model], loc='left', color=colors[model], fontweight='bold', pad=10)
        ax.set_ylim(-3, 103);ax.set_yticks((0, 25, 50, 75, 100))
        ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
        ax.set_axisbelow(True);ax.grid(axis='y', color='#dedede', linewidth=.65)
        if ylabel:ax.set_ylabel(ylabel)

    def save(fig, name):
        paths = {}
        for ext in ('png', 'pdf'):
            path = output / (name + '.' + ext)
            fig.savefig(path, dpi=220, bbox_inches='tight', pad_inches=.14)
            paths[ext] = descriptor(path)
        plt.close(fig)
        exports.append({'name': name, **paths, 'source_summary': source,
                        'role': 'TEMPORARY_SYNTHETIC_LAYOUT' if layout else 'OBSERVED_FINITE_STUDY'})

    mask_legend = [Line2D([0], [0], color='#333333', marker='o', linewidth=2.1,
                          label='No mask'),
                   Line2D([0], [0], color='#333333', marker='s', markerfacecolor='none',
                          linestyle='--', linewidth=2.1, label='Persistent mask')]
    presentations = ('single1', 'dual1', 'single2', 'dual2')
    ticklabels = ('Single\nslot 1', 'Dual\nslot 1', 'Single\nslot 2', 'Dual\nslot 2')

    def condition_lines(ax, model, field):
        for mask, marker, line, fill in (('none', 'o', '-', colors[model]),
                                         ('persistent', 's', '--', 'none')):
            values = [100 * five[(model, mask, p)][field] for p in presentations]
            for indices in ((0, 1), (2, 3)):
                ax.plot(indices, [values[i] for i in indices], color=colors[model], linewidth=2.2,
                        marker=marker, markersize=7, markerfacecolor=fill,
                        markeredgewidth=1.6, linestyle=line)
        ax.set_xlim(-.28, 3.28);ax.set_xticks(range(4), ticklabels)
        ax.axvline(1.5, color='#dedede', linewidth=.8)

    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True, sharey=True)
        fig.subplots_adjust(left=.085, right=.98, bottom=.10, top=.85, hspace=.34, wspace=.20)
        title(fig, 'T4 · Coherent motion readout', 'n = 16 videos per level · 8 paired initialization seeds')
        for ax, model in zip(axes.flat, models):
            axis(ax, model, 'Accuracy' if model in ('qwen', 'nemo') else None)
            ax.plot([100*c for c in coherence], [100*four[(model, 'main', c)]['correct']/16 for c in coherence],
                    color=colors[model], linewidth=2.4, marker=markers[model], markersize=7)
            ax.axhline(50, color='#777777', linestyle=(0, (3, 3)), linewidth=1.0)
            ax.set_xscale('log', base=2);ax.set_xticks([100*c for c in coherence], ('3.2', '6.4', '12.8', '25.6', '51.2', '100'))
            ax.set_xlim(2.7, 117)
        for ax in axes[1]:ax.set_xlabel('Signal coherence (%) · log scale')
        save(fig, 'T4-coherence')

        fig, ax = plt.subplots(figsize=(12.8, 3.9));ax.axis('off')
        title(fig, 'T4 · Controls and direction balance', 'Zero: 8 directionless seeds · Easy: 4 paired seeds with jointly larger dots and slower motion')
        data = []
        for model in models:
            z = four[(model, 'zero_bias', 0.)];easy = four[(model, 'easy_joint', 1.)]
            main = [four[(model, 'main', c)] for c in coherence]
            zero_text = f"L {z['answers'].get('LEFT', 0)}/8  ·  R {z['answers'].get('RIGHT', 0)}/8"
            invalid = 8 - sum(z['answers'].get(a, 0) for a in ('LEFT', 'RIGHT'))
            if invalid:
                zero_text += f"  ·  Invalid {invalid}/8"
            data.append([names[model], zero_text,
                         f"{easy['correct']}/8", f"{sum(r['LEFT']['correct'] for r in main)}/48",
                         f"{sum(r['RIGHT']['correct'] for r in main)}/48"])
        table = ax.table(cellText=data, colLabels=('Model', 'Zero: response', 'Easy: correct\n48 px · 32 px/s',
                         'Main: LEFT\ncorrect', 'Main: RIGHT\ncorrect'), cellLoc='center',
                         colWidths=(.27, .24, .18, .155, .155), bbox=(.005, .02, .99, .73))
        table.auto_set_font_size(False);table.set_fontsize(11.5)
        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor('white');cell.set_linewidth(2)
            cell.set_facecolor('#e8e8e8' if row == 0 else ('#f4f4f4' if row % 2 else 'white'))
            if row == 0:cell.set_text_props(fontweight='bold')
            if row > 0 and col == 0:cell.set_text_props(color=colors[models[row-1]], fontweight='bold', ha='left')
        fig.subplots_adjust(left=.04, right=.98, bottom=.04, top=.88)
        save(fig, 'T4-controls')

        fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), sharex=True, sharey=True)
        fig.subplots_adjust(left=.085, right=.98, bottom=.10, top=.80, hspace=.44, wspace=.20)
        title(fig, 'T5 · Single and dual events', '24 trials · 12 paired onset–position blocks per condition')
        fig.legend(handles=mask_legend, loc='upper center', bbox_to_anchor=(.54, .902), ncol=2, frameon=False)
        for ax, model in zip(axes.flat, models):
            axis(ax, model, 'Balanced accuracy' if model in ('qwen', 'nemo') else None)
            condition_lines(ax, model, 'balanced_accuracy')
            ax.axhline(50, color='#777777', linestyle=(0, (3, 3)), linewidth=1.0)
            ax.tick_params(axis='x', labelbottom=True)
        save(fig, 'T5-balanced-accuracy')

        fig, axes = plt.subplots(2, 4, figsize=(16.0, 7.2), sharex=True, sharey=True)
        fig.subplots_adjust(left=.065, right=.99, bottom=.12, top=.80, hspace=.30, wspace=.16)
        title(fig, 'T5 · Hits and false alarms', '12 target-present + 12 target-absent trials · the same 12 paired blocks')
        fig.legend(handles=mask_legend, loc='upper center', bbox_to_anchor=(.54, .90), ncol=2, frameon=False)
        for column, model in enumerate(models):
            for row, field in enumerate(('hit_rate', 'false_alarm_rate')):
                ax = axes[row, column]
                axis(ax, model, ('Hit rate' if row == 0 else 'False alarm rate') if column == 0 else None)
                if row == 1:ax.set_title('', loc='left')
                condition_lines(ax, model, field)
                ax.set_yticks((0, 50, 100))
        save(fig, 'T5-detection-rates')
    exports.append(render_repeatability(summary, output, source_summary=source))
    return {'status': 'TEMPORARY_LAYOUT_ONLY' if layout else 'OBSERVED_FIGURES_EXPORTED',
            'source_summary': source, 'figures': exports,
            'uncertainty': 'Observed finite-panel rates. No independent-trial intervals or fitted thresholds are drawn.'}


def quantile(values, q):
    values = sorted(values);x = (len(values) - 1) * q;i = int(x);j = min(i + 1, len(values) - 1)
    return values[i] + (values[j] - values[i]) * (x - i)


def timing_tables(records):
    groups = defaultdict(list)
    for r in records:
        if r['timing'] and not r['deep']:
            groups[(r['model'], r['variant'], r['phase'], r['task'], r['replicate'])].append(r)
    output = []
    for key, rows in sorted(groups.items(), key=lambda x: str(x[0])):
        stages = ('end_to_end_durable_seconds', 'engine_request_inclusive_seconds',
                  'native_reader_nested_seconds', 'native_processor_nested_seconds',
                  'official_reader_external_inclusive_seconds', 'padding_and_reference_read_external_seconds',
                  'official_HF_processor_external_seconds', 'post_processor_cast_payload_external_seconds')
        for stage in stages:
            source = 'adapter_timing' if stage.endswith('_external_seconds') or stage == 'official_reader_external_inclusive_seconds' else 'timing'
            values = [(r.get(source) or {}).get(stage) for r in rows]
            values = [v for v in values if v is not None]
            if not values:
                continue
            # Gemma's input adapter separately measures the same reader plus its
            # external processor. Keep that accounting once; the inner reader
            # hook overlaps it, and an empty processor hook does not mean zero work.
            if stage in ('native_reader_nested_seconds', 'native_processor_nested_seconds') and all(r.get('adapter_timing') for r in rows):
                continue
            require(all(math.isfinite(v) and v >= 0 for v in values), 'Invalid stage timing')
            output.append({'model': key[0], 'variant': key[1], 'phase': key[2], 'task': key[3], 'replicate': key[4],
                           'stage': stage, 'n': len(values), 'mean': statistics.mean(values),
                           'median': statistics.median(values), 'p90': quantile(values, .9),
                           'videos_per_hour': 3600 / statistics.mean(values) if stage == 'end_to_end_durable_seconds' else None,
                           'nested_not_additive': stage != 'end_to_end_durable_seconds',
                           'scope': 'read_to_durable_result' if stage == 'end_to_end_durable_seconds' else
                                    ('whole_request_including_input_adapter' if stage == 'engine_request_inclusive_seconds'
                                     else 'nested_inside_whole_request')})
    return output


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields);w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v
                        for k, v in r.items()})


def add_historical_a(root, manifest_path):
    """Join original controlled A to already-audited C/X, without re-running workers."""
    archive = Archive(root)
    output = archive.root / 'runs/reproducibility/official-x-20261028/analysis'
    accepted = read(output / 'summary.json')
    require(accepted['status'] == 'COMPLETE_FINITE_STUDY', 'Completed X audit required')
    require(descriptor(manifest_path) == accepted['manifest'], 'Accepted manifest changed')
    plan = read(manifest_path)
    records = read(output / 'canonical-records.json')
    c_records = archive.ref(plan['historical_C'])['records']
    validation, _ = validation_tables(records, c_records)
    require(validation == accepted['validation'], 'Canonical records differ from accepted comparison')
    a_path = archive.root / 'plans/reader-route-smoke-20261026/historical.csv'
    with a_path.open() as f:
        a_rows = list(csv.DictReader(f))
    a_lookup = {(r['model'], r['sample_id']): r for r in a_rows}
    c_lookup = {(r['model'], r['sample_id']): r for r in c_records}
    x_lookup = {(r['model'], r['sample_id']): r for r in records
                if r['phase'] == 'validation' and r['replicate'] == 1}
    require(len(a_rows) == len(a_lookup) == len(c_records) == len(c_lookup) == len(x_lookup) == 576,
            'Expected exactly 576 unique model/source pairs in each route')
    require(set(a_lookup) == set(c_lookup) == set(x_lookup), 'Historical source IDs differ')
    paired, groups = [], defaultdict(list)
    for key, x in x_lookup.items():
        a, c = a_lookup[key], c_lookup[key]
        require(a['protocol'] == 'controlled-rgb16', 'A must be original controlled historical answers')
        require(a['task'] == c['task'] == x['task'] and a['truth'] == c['truth'] == x['truth'],
                'A/C/X task or ground truth differs')
        require(a['source_video_sha256'] == c['source_video_sha256']
                and a['source_video_sha256'] in [sha for kind, sha in x['source_media'] if kind == 'video'],
                'A/C/X source video bytes differ')
        answers = {'A': a['answer'], 'C': c['answer'], 'X': x['answer']}
        require(all(v in ('LEFT', 'RIGHT') for v in answers.values()), 'Invalid historical label')
        row = {'model': key[0], 'task': x['task'], 'sample_id': key[1], 'truth': x['truth'],
               'source_video_sha256': a['source_video_sha256'], **answers,
               **{k + '_correct': v == x['truth'] for k, v in answers.items()},
               'A_raw_sha256': a['raw_sha256'], 'X_raw_digest': x['raw_digest'],
               'A_reference_file': a['reference_file'], 'A_reference_row': a['reference_row']}
        require(row['A_correct'] == (a['correct'] == '1') and row['C_correct'] == c['correct']
                and row['X_correct'] == x['correct'], 'Saved correctness does not match labels')
        paired.append(row); groups[(key[0], x['task'])].append(row)
    comparison = []
    for (model, task), rows in sorted(groups.items()):
        require(len(rows) == {'T2': 48, 'T3': 96}[task], 'Incomplete fixed cell')
        cell = {'model': model, 'task': task, 'n': len(rows),
                **{route + '_correct': sum(r[route + '_correct'] for r in rows) for route in ('A', 'C', 'X')}}
        for before, after in (('A', 'C'), ('A', 'X'), ('C', 'X')):
            prefix = before + '_to_' + after
            cell.update({prefix + '_changes': sum(r[before] != r[after] for r in rows),
                         prefix + '_correct_to_wrong': sum(r[before + '_correct'] and not r[after + '_correct'] for r in rows),
                         prefix + '_wrong_to_correct': sum(not r[before + '_correct'] and r[after + '_correct'] for r in rows)})
        comparison.append(cell)
    summary = {'schema': 'official-x-historical-A-C-X-v1', 'status': 'COMPLETE_DESCRIPTIVE_COMPARISON',
               'new_model_calls': 0, 'unique_model_source_pairs': len(paired), 'comparison': comparison,
               'sources': {name: descriptor(path) for name, path in {
                   'historical_A': a_path, 'historical_C': archive.path(plan['historical_C']['path']),
                   'accepted_X_audit': output / 'summary.json', 'accepted_X_records': output / 'canonical-records.json'
               }.items()},
               'limits': ['A is the original controlled historical answer, not smoke A replay.',
                          'Nemo historical replay had seven cross-host differences on these 144 sources; causal attribution to decoding alone is not established.',
                          'C and X both use uint8, but differ in runtime stack; scores are not a bit-depth-only comparison.',
                          'X is the first accepted replicate; the second is separately verified identical.']}
    atomic_json(output / 'A-C-X-summary.json', summary)
    write_csv(output / 'A-C-X-comparison.csv', comparison)
    write_csv(output / 'A-C-X-paired.csv', paired)
    return {'status': summary['status'], 'new_model_calls': 0, 'pairs': len(paired), 'comparison': comparison}


def analyze(root, manifest_path):
    archive = Archive(root);manifest_path = Path(manifest_path);plan = read(manifest_path)
    require(plan['schema'] == 'official-x-analysis-manifest-v1', 'Explicit accepted-worker manifest required')
    records, workers = [], []
    for spec in plan['workers']:
        r, w = audit_worker(archive, spec);records.extend(r);workers.append(w)
    require(len(records) == plan['expected_effective_calls'], 'Effective call budget not complete')
    history = archive.ref(plan['historical_C'])['records']
    validation, vpaired = validation_tables(records, history)
    validation_ids = {archive.ref(spec['profile'])['worker'] for spec in plan['workers']
                      if spec['phase'] == 'validation'}
    contracts = validation_runtime_contracts([w for w in workers if w['worker'] in validation_ids], archive)
    require(all(row['answer_changes'] == 0 and row['complete_output_changes'] == 0
                and row['prompt_token_changes'] == 0 for row in validation),
            'Selected final configuration has not passed finite self-repeatability')
    panels = panel_tables(records);timing = timing_tables(records)
    require(len(validation) == 8 and len(panels['T5']) == 32 and len(panels['T4']) == 32, 'Missing model/cell results')
    require(len(vpaired) == 576 and len(records) == 2368, 'Canonical study count differs')
    output = archive.root / 'runs/reproducibility/official-x-20261028/analysis'
    output.mkdir(parents=True, exist_ok=True)
    summary = {'schema': 'official-x-study-results-v1', 'status': 'COMPLETE_FINITE_STUDY',
               'manifest': descriptor(manifest_path), 'effective_calls': len(records),
               'count_scope': 'Selected configuration only; retained defaults, failed starts, and qualification variants are separate physical attempts.',
               'validation': validation, 'validation_runtime_contracts': contracts,
               'T4': panels['T4'], 'T5': panels['T5'], 'T5_costs': panels['costs'],
               'workers': workers, 'timing': timing,
               'limits': ['Finite same-card repeated-input evidence, not all-hardware or future-version determinism.',
                          'Historical C is a descriptive reference, not a target score.',
                          'T4 paired seeds and T5 onset/position blocks are the replication units.',
                          'T4 easy controls jointly change size and speed and do not enter the coherence curve.',
                          'Fixed T5 SOA does not estimate an attentional-blink time window.',
                          'Reader and processor timings are nested in the engine request; do not add them.']}
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'canonical-records.json', records)
    write_csv(output / 'validation.csv', validation);write_csv(output / 'validation-paired.csv', vpaired)
    write_csv(output / 'T4.csv', panels['T4']);write_csv(output / 'T5.csv', panels['T5'])
    write_csv(output / 'T5-paired.csv', panels['paired']);write_csv(output / 'T5-costs.csv', panels['costs'])
    write_csv(output / 'timing.csv', timing)
    return {'status': summary['status'], 'effective_calls': len(records), 'summary': descriptor(output / 'summary.json')}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('--root', required=True)
    p.add_argument('--manifest', required=True)
    p.add_argument('--add-historical-a', action='store_true', help='Join original A to an existing completed X audit; no worker re-audit')
    a = p.parse_args()
    fn = add_historical_a if a.add_historical_a else analyze
    print(json.dumps(fn(a.root, a.manifest), ensure_ascii=False))


if __name__ == '__main__':
    main()
