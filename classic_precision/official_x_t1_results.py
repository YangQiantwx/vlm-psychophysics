"""Read-only acceptance of the finite 8-source x 2-arm x 2-cold T1 study.

The manifest explicitly selects four sealed profiles and their variant. No model,
runner, tokenizer, image decoder, process probe, or GPU library is imported.
Only complete 32-call evidence can produce COMPLETE_FINITE_T1_STUDY.

Manifest fields: schema, variant, expected_effective_calls=32, run_root,
selection/private_manifest/cpu_budget descriptors, and four workers containing
arm, cold and a sealed profile descriptor. The manifest chooses the accepted
variant explicitly; it never searches for the newest profile or unpauses work.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import tempfile
import uuid

from .official_x_results import Archive, descriptor, journal, read, unique, pair_stats, quantile
from .storage import atomic_json, digest

ARMS = ('video', 'images')
STAGES = ('end_to_end_durable_seconds', 'engine_request_inclusive_seconds',
          'native_reader_nested_seconds', 'native_processor_nested_seconds')
MANIFEST_SCHEMA = 'official-x-t1-analysis-manifest-v1'
RESULT_SCHEMA = 'official-x-t1-results-v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def refs(archive, plan):
    require(plan['schema'] == MANIFEST_SCHEMA, 'Explicit T1 analysis manifest required')
    require(plan['expected_effective_calls'] == 32, 'T1 allocation must remain exactly 32')
    require(isinstance(plan['variant'], str) and plan['variant'].strip(), 'Explicit accepted variant required')
    require(len(plan['workers']) == 4 and
            {(w['arm'], w['cold']) for w in plan['workers']} == {(a, c) for a in ARMS for c in (1, 2)},
            'Exactly video/images x cold1/cold2 profiles required')
    selection = archive.ref(plan['selection'])
    private = archive.ref(plan['private_manifest'])
    budget = archive.ref(plan['cpu_budget'])
    require(selection['maximum_new_calls'] == 32 and selection['model_calls'] == 0 and
            selection['status'] == 'FROZEN_BEFORE_NEW_MODEL_CALLS', 'Selection was not frozen before inference')
    require(selection['private_manifest']['sha256'] == plan['private_manifest']['sha256'], 'Frozen selection/private source binding differs')
    ids = selection['physical_ids']
    require(len(ids) == len(set(ids)) == 8, 'Eight unique physical IDs required')
    require(all(str(uuid.UUID(sid)) == sid for sid in ids), 'Source IDs must be anonymous UUIDs')
    chosen = unique(private['selected'], 'physical_id')
    require(set(chosen) == set(ids), 'Private/source selection differs')
    cells = Counter((r['source_parameters']['sf_cycles_per_aperture'],
                     r['source_parameters']['modulation_depth'],
                     r['source_parameters']['phase_index'], r['truth']) for r in chosen.values())
    require(cells == Counter({(sf, mod, phase, gt): 1 for sf, mod in ((4, .9), (8, .8))
                              for phase in (0, 12) for gt in ('LEFT', 'RIGHT')}), 'Frozen SF/contrast/phase/LR panel changed')
    require(budget['status'] == 'PASS' and budget['model_calls'] == 0 and budget['weights_loaded'] is False
            and budget['CUDA_initialized'] is False, 'Missing complete zero-model CPU qualification')
    attempts = {}
    for row in budget['attempts']:
        key = row['arm'], row['sample_id']
        require(key not in attempts, 'Duplicate CPU budget item')
        attempts[key] = row
    require(set(attempts) == {(a, sid) for a in ARMS for sid in ids}, 'CPU budget must cover all 16 arm/source inputs')
    for (arm, sid), row in attempts.items():
        require(row['source_all180_pixel_hashes_match'] and row['fits_context'] and
                row['context'] == budget['max_model_len'] and row['max_tokens'] == 16384 and
                row['prompt_tokens'] + row['max_tokens'] <= row['context'], 'CPU expansion/context qualification failed')
        require(len(row['pixel_payloads']) == (1 if arm == 'video' else 180), 'CPU native payload count differs')
        grids = [g['values'] for g in row['grids']]
        require(grids == ([[90, 32, 48]] if arm == 'video' else [[1, 32, 48]] * 180), 'CPU native temporal/spatial grids differ')
    return ids, chosen, attempts, budget


def source_checks(item, arm, sid, selected):
    require(item['sample_id'] == item['physical_id'] == sid and item['task'] == 'T1', 'Catalog physical identity differs')
    row = item['row']
    require(row['prompt'] == selected['prompt'] and row['truth'] == selected['truth'] and
            row['source_parameters'] == selected['source_parameters'] and row['choices'] == ['LEFT', 'RIGHT'],
            'Frozen scientific source/question/GT changed')
    require(item['video_contract'] == {'frames': 180, 'fps': 60, 'width': 768, 'height': 512} and
            item['image_contract'] == {'count': 180, 'width': 768, 'height': 512}, 'T1 all180/original geometry changed')
    require(item['ready']['video_sha256'] == selected['source']['sha256'] and
            item['ready']['source_qa'] == 'PASS', 'Source video binding differs')
    hashes = item['frame_pixel_sha256']
    require(len(hashes) == 180 and all(isinstance(x, str) and len(x) == 64 for x in hashes), 'Missing full frame-pixel sequence')
    require(len(item['decoded_video_sha256']) == 64, 'Missing ordered complete decoded-video hash')
    content = item['content']
    require(content[-1] == {'type': 'text', 'text': row['prompt']}, 'Original text must be the final content item')
    media = content[:-1]
    if arm == 'video':
        require(len(media) == 1 and media[0]['type'] == 'video' and
                media[0]['sha256'] == selected['source']['sha256'] and Path(media[0]['path']).stem == sid,
                'Video arm source/anonymous filename differs')
    else:
        require(len(media) == 180 and all(m['type'] == 'image' for m in media), 'Image arm must contain 180 logical images')
        for i, entry in enumerate(media):
            require(entry['frame_index'] == i and entry['pixel_sha256'] == hashes[i] and
                    Path(entry['path']).parent.name == sid and Path(entry['path']).name == f'{sid}_{i:03d}.png',
                    'PNG order, source-pixel binding, or anonymous naming differs')
    return media


def expected_messages(item):
    content = []
    for entry in item['content']:
        kind = entry['type']
        if kind == 'text':
            content.append({'type': 'text', 'text': entry['text']})
        else:
            content.append({'type': kind + '_url', kind + '_url': {'url': Path(entry['path']).as_uri()}})
    return [{'role': 'user', 'content': content}]


def processor_checks(raw, arm, expected):
    processors = raw['native_processor']
    require(processors, 'No actual native CPU processor observations')
    phase_values = []
    for observation in processors:
        for phase in ('before_native_dtype', 'after_native_dtype'):
            phase_values.append(observation[phase])
    grid_key = 'video_grid_thw' if arm == 'video' else 'image_grid_thw'
    wanted = [[90, 32, 48]] if arm == 'video' else [[1, 32, 48]] * 180
    # The observer records the processor's batched grids, not 180 GPU executions.
    require(any(p.get(grid_key, {}).get('values') == wanted for p in phase_values), 'Actual native CPU grid/count changed')
    pixel_key = 'pixel_values_videos' if arm == 'video' else 'pixel_values'
    shape = [138240, 1536] if arm == 'video' else [276480, 1536]
    payloads = [p[pixel_key] for p in phase_values if pixel_key in p]
    require(payloads and all(p['shape'] == shape and p['dtype'] == 'torch.uint8' for p in payloads),
            'Actual processor spatial/temporal payload or dtype changed')
    if raw['deep']:
        after = [o['after_native_dtype'][pixel_key] for o in processors if pixel_key in o['after_native_dtype']]
        if arm == 'video':
            require(any(p.get('sha256') == expected['pixel_payloads'][0]['sha256'] for p in after), 'CPU video patch bytes differ')
        else:
            wanted_hashes = [x['sha256'] for x in expected['pixel_payloads']]
            require(any(p.get('per_image_patch_sha256') == wanted_hashes for p in after), 'CPU logical image patch order differs')
    return {'grid_thw': wanted, 'pixel_shape': shape, 'dtype': 'torch.uint8'}


def selector_checks(profile, observation):
    runtime = observation['runtime']
    require(runtime['actual_kv_capacity'] >= profile['minimum_kv_capacity'], 'Actual KV capacity too small')
    if profile.get('selector_source'):
        q = runtime['custom_qsa_selector']
        require(q['status'] == 'APPLIED_X_QSA' and q['actual_function_is_canonical_wrapper'] and
                q['native_function_verified_before_replacement'] and
                q['selector_source_sha256'] == profile['selector_source']['sha256'] and
                q['helper_status_after_import'] is None and not q['helper_import_added_meta_path_hook'] and
                not q['SDK_files_modified'] and not q['forward_processor_or_weights_changed'] and
                observation['actual_custom_selector_calls_during_request'] > 0, 'Actual custom-selector identity/call evidence differs')
    else:
        q = runtime['native_qsa']
        require(q['function_source_exact'] and q['binary_mapped'] and not q['custom_selector_modules'] and
                not q['custom_selector_import_hooks'] and any('persistent_topk' in x['key'] and x['count'] > 0
                for x in observation['actual_topk_operator_calls']), 'Actual official selector evidence differs')


def deep_checks(profile, item, raw, observations, admission, arm, expected):
    require(observations and all(not o.get('runtime_error') and not o.get('profiler_error') for o in observations),
            'Incomplete actual GPU observations')
    for observation in observations:
        selector_checks(profile, observation)
        norm = observation['normalizer']
        require(norm['class'].endswith(('.FusedMMInputNorm', '.IdentityInputNorm')) and
                norm.get('source', {}).get('sha256'), 'No identified native normalization source')
    if arm == 'images':
        require(admission is not None and admission['raw_digest'] == digest(raw) and
                admission['observations_digest'] == digest(observations), 'Images admission does not bind raw/deep evidence')
        proof = admission['proof']
        wanted = [x['sha256'] for x in expected['pixel_payloads']]
        require(proof['status'] == 'PASS_NATIVE_ORDERED_IMAGE_LIST' and proof['logical_image_count'] == 180 and
                proof['logical_image_grid_thw'] == [[1, 32, 48]] * 180 and
                proof['logical_patch_sha256_in_order'] == wanted and proof['logical_prompt_token_ids_match_CPU'] and
                proof['actual_GPU_bytes_cover_all_logical_images'] and proof['native_duplicate_image_reuse_allowed'],
                'Images logical/source/GPU admission differs')
        actual = []
        batches = 0
        for observation in observations:
            for outer in observation['outer']:
                tensor = outer['tensor']
                hashes = outer.get('per_image_patch_sha256', [])
                require(hashes and tensor['shape'] == [len(hashes) * 1536, 1536] and
                        tensor['dtype'] == 'torch.uint8' and tensor['device'].startswith('cuda'), 'Actual image CUDA payload differs')
                grid = outer['grid_thw']
                require(grid == [[1, 32, 48]] * len(hashes) or (len(hashes) == 1 and grid == [1, 32, 48]),
                        'Actual image GPU grid differs')
                actual.extend(hashes)
                batches += 1
        require(actual and set(actual) == set(wanted) and proof['actual_GPU_encoded_patch_sha256'] == actual and
                proof['actual_GPU_encoded_image_count'] == len(actual) and
                proof['unique_logical_image_payload_count'] == len(set(wanted)), 'Actual encoded image coverage/count differs')
        indices = defaultdict(list)
        for i, sha in enumerate(actual):
            indices[sha].append(i)
        mapping = [{'logical_frame_index': i, 'source_pixel_sha256': item['frame_pixel_sha256'][i],
                    'native_patch_sha256': sha, 'actual_GPU_payload_indices': indices[sha]}
                   for i, sha in enumerate(wanted)]
        return {'scope': 'ACTUAL_GPU_DEEP_OBSERVATION', 'logical_image_count': 180,
                'unique_logical_patch_payloads': len(set(wanted)), 'actual_GPU_image_payloads': len(actual),
                'unique_actual_GPU_patch_payloads': len(set(actual)), 'observed_visual_forward_calls': batches,
                'logical_to_GPU_payload': mapping,
                'logical_content_multiplicity': dict(Counter(wanted)),
                'observed_GPU_content_multiplicity': dict(Counter(actual)),
                'mapping_semantics': 'Byte-equivalent observed payload positions. If equal bytes were encoded more than once, internal cached feature-instance routing is unobserved.',
                'internal_normalized_tensor_observed': any(o.get('normalized') for o in observations)}
    wanted = expected['pixel_payloads'][0]
    outer = [x for o in observations for x in o['outer']]
    require(outer and all(x['tensor']['shape'] == wanted['shape'] and x['tensor']['dtype'] == wanted['dtype'] and
                         x['tensor'].get('sha256') == wanted['sha256'] and x['tensor']['device'].startswith('cuda')
                         and x['grid_thw'] in ([90, 32, 48], [[90, 32, 48]]) for x in outer),
            'Actual video GPU patch bytes/grid differ from CPU budget')
    if admission is not None:
        proof = admission['proof']
        require(admission['raw_digest'] == digest(raw) and admission['observations_digest'] == digest(observations) and
                proof['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS' and proof['CPU_to_GPU_patch_bytes_equal'] and
                proof['source_reader_all180'] and proof['source_frame_indices'] == list(range(180)), 'Video ingress admission differs')
    else:
        require(any(o.get('normalized') for o in observations), 'Video has no normalized observation or ingress admission')
    return {'scope': 'ACTUAL_GPU_DEEP_OBSERVATION', 'source_frames': 180, 'temporal_grid': [90, 32, 48],
            'CPU_GPU_patch_sha256': wanted['sha256'], 'observed_visual_forward_calls': len(outer),
            'internal_normalized_tensor_observed': any(o.get('normalized') for o in observations)}


def backend_signature(runtime):
    keys = ('vllm_version', 'torch_version', 'cuda_version', 'model_class', 'runner_class', 'kernel_config',
            'method_kernel_counts', 'actual_backends', 'mm_device_do_normalize')
    require(runtime and all(all(k in row for k in keys) for row in runtime), 'Missing actual backend inventory')
    return digest([{k: row[k] for k in keys} for row in runtime])


def audit_worker(archive, spec, plan, ids, selected, attempts, budget):
    profile = archive.ref(spec['profile'])
    arm, cold = spec['arm'], spec['cold']
    require(profile['input_arm'] == arm and profile['cold'] == cold and profile['assigned_ids'] == ids and
            profile['maximum_model_calls'] == 8 and profile['automatic_retry'] is False and
            profile['formal_auto_resume'] is False, 'Worker allocation/arm/cold/retry contract differs')
    require(profile['t1_cpu_budget']['sha256'] == plan['cpu_budget']['sha256'], 'Profile/CPU qualification differs')
    settings = profile['engine_kwargs']
    require(profile.get('t1_entry') and profile.get('observer_entry'), 'Missing sealed T1/native-input observer source identity')
    require(settings['max_model_len'] == budget['max_model_len'] and
            profile['minimum_kv_capacity'] >= budget['max_model_len'] and
            settings['kv_cache_memory_bytes'] == budget['both_arms_same_KV_bytes'] and
            settings['limit_mm_per_prompt']['image'] == budget['both_arms_same_image_limit'] == 180 and
            settings['limit_mm_per_prompt']['video'] == 1 and settings['tensor_parallel_size'] == 1 and
            settings['max_num_seqs'] == 1 and settings['mm_processor_cache_gb'] == 0 and
            settings['enable_prefix_caching'] is False and not profile.get('request_adapter'), 'Scientific/resource input contract differs')
    require(profile['request_mm_processor_kwargs'] == {'do_resize': False, 'do_sample_frames': False},
            'Request changed all180/no-resize processing')
    catalog = archive.ref(profile['source_catalog'])
    require(catalog['assigned_ids'] == ids and catalog['selection_sha256'] == plan['selection']['sha256'] and
            catalog['private_manifest_sha256'] == plan['private_manifest']['sha256'], 'Catalog selection/GT provenance differs')
    out = archive.path(profile['output'])
    identity, wait = read(out / 'identity.json'), read(out.with_suffix('.wait.json'))
    terminal, exit_record = read(out / 'result.json'), read(out / 'exit.json')
    launch, lease = read(out.with_suffix('.launch.json')), read(out / 'lease.json')
    require(identity['profile'] == profile and identity['profile_sha256'] == digest(profile) and
            launch['profile_sha256'] == wait['profile_sha256'] == digest(profile) and
            launch['automatic_retry'] is False and launch['pid'] == identity['owner']['pid'], 'Worker/launch/wait identity differs')
    require(wait['returncode'] == 0 and wait['terminal_status'] == 'COMPLETE' and wait['real_wait_observed'] and
            wait['owned_engines_exited'] and wait['original_physical_lease_released'] and wait['automatic_retry'] is False,
            'Worker lacks real wait, owned exit, or original-inode release')
    require(terminal['status'] == 'COMPLETE' and terminal['completed_ids'] == wait['completed_ids'] == ids and
            not terminal['UNCERTAIN_ids'] and not terminal['unattempted_ids'], 'Missing/unattempted/uncertain calls remain')
    require(all(lease[k] == profile['physical_lock'][k] for k in ('path', 'st_dev', 'st_ino')) and
            lease['admission']['admitted'], 'Original physical lease inode/admission differs')
    owners = read(out / 'engine-owner.json')['owned']
    require(owners and exit_record['status'] == 'COMPLETE' and exit_record['owned_engine_workers_exited'] and
            exit_record['shutdown_error'] is None and exit_record['owned_workers'] == [x['worker'] for x in owners],
            'Exact owned engine identity did not close cleanly')
    raws = unique(journal(out / 'raw.jsonl'))
    events = journal(out / 'events.jsonl')
    require(all(e['event'] in ('INTENT', 'TECHNICAL_PASS') for e in events), 'Failed/unknown event cannot enter strict T1 acceptance')
    intents = unique([e for e in events if e['event'] == 'INTENT'])
    passes = unique([e for e in events if e['event'] == 'TECHNICAL_PASS'])
    timings = unique(journal(out / 'request-timings.jsonl'))
    require(set(raws) == set(intents) == set(passes) == set(timings) == set(ids), 'Raw/PASS/claim/timing ledger incomplete')
    deep_rows = unique(journal(out / 'deep-observations.jsonl'))
    require(set(deep_rows) == {ids[0]}, 'GPU deep scope differs from frozen first-task allocation')
    admissions_path = out / 'input-admissions.jsonl'
    admissions = unique(journal(admissions_path)) if admissions_path.exists() else {}
    require(set(admissions) <= {ids[0]}, 'Unexpected extra GPU input admissions')
    runtime = read(out / 'runtime.json')
    versions = read(out / 'versions.json')
    require(all(versions[k] == v for k, v in profile['expected_packages'].items()) and
            all(r['vllm_version'] == versions['vllm'] and
                r['torch_version'].split('+')[0] == versions['torch'].split('+')[0] for r in runtime),
            'Actual pinned runtime package identity differs')
    require(all(r['actual_kv_capacity'] >= profile['minimum_kv_capacity'] for r in runtime), 'Runtime KV qualification failed')
    public = read(out / 'public-settings.json')
    cache = read(out / 'cache-policy.json')
    require(cache['fresh_engine'] and cache['source_file_page_cache_warmed'] and cache['decoded_cache'] is False and
            cache['prefix_cache'] is False and cache['mm_processor_cache_gb'] == 0, 'Cold engine/source-cache policy differs')
    for key, value in settings.items():
        require(public['LLM_kwargs'][key] == (sorted(value) if key == 'cpu_offload_params' else value),
                'Saved actual engine setting differs: ' + key)
    require(public['request_mm_processor_kwargs'] == profile['request_mm_processor_kwargs'] and
            public['chat_template_kwargs'] == {'enable_thinking': False}, 'Actual request/template settings differ')
    records = []
    for ordinal, sid in enumerate(ids):
        item = catalog['items'][sid]
        media = source_checks(item, arm, sid, selected[sid])
        for entry in media:
            bound = profile['media_files'][entry['path']]
            require(bound['path'] == entry['path'] and bound['sha256'] == entry['sha256'] and
                    bound['type'] == entry['type'] and bound['stat'] == entry['stat'], 'Sealed PNG/video descriptor differs from source catalog')
        raw, timing = raws[sid], timings[sid]
        require(raw['sample_id'] == sid and raw['task'] == 'T1' and raw['profile_sha256'] == digest(profile) and
                raw['ordinal'] == ordinal and raw['attempt'] == 1 and raw['deep'] == (ordinal == 0), 'Raw allocation differs')
        require(raw['source']['catalog'] == profile['source_catalog'] and raw['source']['item_digest'] == digest(item) and
                raw['source']['media'] == [profile['media_files'][m['path']] for m in media], 'Raw source/video/PNG binding differs')
        require(raw['request'] == expected_messages(item) and raw['question'] == item['row']['prompt'] and
                raw['truth'] == item['row']['truth'] and raw['accuracy_eligible'] is True, 'Prompt, label privacy, order, or GT changed')
        require(raw['request_mm_processor_kwargs'] == profile['request_mm_processor_kwargs'], 'Actual request processing differs')
        gen = raw['generation']
        require(all(gen[k] == v for k, v in {'n': 1, 'temperature': 0., 'top_p': 1., 'seed': 0,
                'repetition_penalty': 1., 'max_tokens': 16384, 'ignore_eos': False, 'stop': [],
                'stop_token_ids': [], 'choice': ['LEFT', 'RIGHT'], 'thinking': False}.items()), 'Generation contract differs')
        require(gen['native_EOS_override'] is False, 'Native EOS was overridden')
        answer = raw['raw_answer'].strip()
        require(raw['finish_reason'] == 'stop' and answer in ('LEFT', 'RIGHT') and raw['output_token_ids'] and
                raw['correct'] == (answer == raw['truth']), 'Invalid/incorrectly scored scientific completion')
        expected = attempts[(arm, sid)]
        require(len(raw['actual_prompt_token_ids']) == expected['prompt_tokens'] and
                digest(raw['actual_prompt_token_ids']) == expected['prompt_token_ids_sha256'], 'Actual expanded prompt differs from CPU')
        readers = raw['native_reader']
        if arm == 'video':
            require(len(readers) == 1 and readers[0]['kind'] == 'video' and readers[0]['path'] == media[0]['path'] and
                    readers[0]['shape'] == [180, 512, 768, 3] and readers[0]['dtype'] == 'uint8' and
                    readers[0]['metadata']['frames_indices'] == list(range(180)) and
                    readers[0]['decoded_sha256'] == item['decoded_video_sha256'] and
                    readers[0]['per_frame_sha256'] == item['frame_pixel_sha256'], 'Native video pixel/order proof differs')
        else:
            require(len(readers) == 180 and len({r['frame_index'] for r in readers}) == 180, 'Missing/duplicated image reader observations')
            by_frame = {r['frame_index']: r for r in readers}
            require(set(by_frame) == set(range(180)), 'Actual image logical identities differ')
            for i, entry in enumerate(media):
                r = by_frame[i]
                require(r['kind'] == 'image' and r['path'] == entry['path'] and r['shape'] == [512, 768, 3] and
                        r['dtype'] == 'uint8' and r['pixel_sha256'] == entry['pixel_sha256'], 'Native PNG pixels/source binding differ')
        processor = processor_checks(raw, arm, expected)
        claim = read(archive.path(profile['claim_root']) / (sid + '.json'))
        require(claim['sample_id'] == sid and claim['profile_sha256'] == digest(profile) and
                claim['worker_token'] == profile['worker_token'] and claim['attempt'] == 1, 'Claim/raw identity differs')
        require(passes[sid]['raw_digest'] == digest(raw) and passes[sid]['ordinal'] == intents[sid]['ordinal'] == ordinal and
                passes[sid]['deep'] == intents[sid]['deep'] == raw['deep'], 'PASS/intent does not bind raw allocation')
        require(timing['task'] == 'T1' and timing['deep'] == raw['deep'] and timing['nested_not_additive'] is True,
                'Timing identity/scope differs')
        for stage in STAGES:
            require(math.isfinite(timing[stage]) and timing[stage] >= 0, 'Invalid observed timing')
            if stage in raw['timing']:
                require(timing[stage] == raw['timing'][stage], 'Raw/durable timing changed')
        require(0 < timing['engine_request_inclusive_seconds'] <= timing['end_to_end_durable_seconds'], 'Durable/engine timing inconsistent')
        gpu = {'scope': 'NOT_GPU_DEEP_OBSERVED_THIS_ID', 'logical_images': 180 if arm == 'images' else None,
               'CPU_expanded_prompt_and_all_source_reader_pixels_verified': True}
        if raw['deep']:
            deep = deep_rows[sid]
            require(deep['raw_digest'] == digest(raw), 'Deep/raw digest differs')
            require(backend_signature([o['runtime'] for o in deep['observations']]) == backend_signature(runtime),
                    'Deep observer/backend identity differs from loaded engine')
            gpu = deep_checks(profile, item, raw, deep['observations'], admissions.get(sid), arm, expected)
        params = item['row']['source_parameters']
        records.append({'sample_id': sid, 'arm': arm, 'cold': cold, 'variant': plan['variant'],
                        'profile_variant': profile.get('variant'), 'raw_variant': raw.get('variant'),
                        'truth': raw['truth'], 'eligible': True, 'answer': answer, 'correct': raw['correct'],
                        'sf': params['sf_cycles_per_aperture'], 'contrast': params['modulation_depth'],
                        'phase_index': params['phase_index'], 'output_token_ids': raw['output_token_ids'],
                        'finish_reason': raw['finish_reason'], 'stop_reason': raw['stop_reason'],
                        'prompt_tokens': len(raw['actual_prompt_token_ids']), 'prompt_token_sha256': digest(raw['actual_prompt_token_ids']),
                        'source_video_sha256': item['ready']['video_sha256'], 'decoded_video_sha256': item['decoded_video_sha256'],
                        'frame_pixel_sha256': item['frame_pixel_sha256'], 'processor': processor, 'GPU_evidence': gpu,
                        'deep': raw['deep'], 'timing': timing, 'worker': profile['worker'],
                        'profile_sha256': digest(profile), 'raw_digest': digest(raw),
                        'raw_journal': str(out / 'raw.jsonl'), 'ordinal': ordinal})
    load = read(out / 'load.json')['seconds']
    require(math.isfinite(load) and load > 0, 'Invalid cold engine load timing')
    evidence = {'arm': arm, 'cold': cold, 'variant': plan['variant'], 'worker': profile['worker'],
                'profile': spec['profile'], 'raw': descriptor(out / 'raw.jsonl'), 'wait': descriptor(out.with_suffix('.wait.json')),
                'exit': descriptor(out / 'exit.json'), 'lease': descriptor(out / 'lease.json'),
                'identity': descriptor(out / 'identity.json'), 'owned_workers': [o['worker'] for o in owners],
                'actual_backend_signature': backend_signature(runtime), 'runtime': descriptor(out / 'runtime.json'),
                'public_settings': descriptor(out / 'public-settings.json'), 'engine_settings_sha256': digest(settings),
                'entry_sources': {k: profile[k] for k in ('implementation', 'observer_entry', 't1_entry',
                                  't1_qsa_entry', 'qsa_entry', 'selector_source') if k in profile},
                'versions': descriptor(out / 'versions.json'),
                'cache_policy': descriptor(out / 'cache-policy.json'),
                'load_seconds': load, 'accepted_calls': len(records)}
    return records, evidence


def pair_table(records, ids):
    lookup = {(r['arm'], r['cold'], r['sample_id']): r for r in records}
    require(len(lookup) == len(records) == 32, 'Exactly 32 unique arm/cold/source results required')
    summaries, detailed = [], []
    contrasts = [('cold_repeat', a, 1, a, 2) for a in ARMS] + [('video_to_images', 'video', c, 'images', c) for c in (1, 2)]
    for kind, aa, ac, ba, bc in contrasts:
        pairs = [(lookup[(aa, ac, sid)], lookup[(ba, bc, sid)]) for sid in ids]
        for a, b in pairs:
            require(a['truth'] == b['truth'] and a['source_video_sha256'] == b['source_video_sha256'] and
                    a['frame_pixel_sha256'] == b['frame_pixel_sha256'] and a['decoded_video_sha256'] == b['decoded_video_sha256'],
                    'Paired source pixels or GT differ')
            if kind == 'cold_repeat':
                require(a['prompt_token_sha256'] == b['prompt_token_sha256'] and a['processor'] == b['processor'],
                        'Cold repeats changed logical input')
            detailed.append({'contrast': kind, 'before_arm': aa, 'before_cold': ac, 'after_arm': ba, 'after_cold': bc,
                             'sample_id': a['sample_id'], 'sf': a['sf'], 'contrast_depth': a['contrast'],
                             'phase_index': a['phase_index'], 'truth': a['truth'], 'before': a['answer'], 'after': b['answer'],
                             'answer_changed': a['answer'] != b['answer'], 'before_correct': a['correct'], 'after_correct': b['correct'],
                             'before_prompt_tokens': a['prompt_tokens'], 'after_prompt_tokens': b['prompt_tokens'],
                             'same_expanded_prompt': a['prompt_token_sha256'] == b['prompt_token_sha256'],
                             'complete_output_equal': (a['output_token_ids'], a['finish_reason'], a['stop_reason']) ==
                                                      (b['output_token_ids'], b['finish_reason'], b['stop_reason'])})
        for sf in (None, 4, 8):
            subset = pairs if sf is None else [(a, b) for a, b in pairs if a['sf'] == sf]
            summaries.append({'contrast': kind, 'before_arm': aa, 'before_cold': ac, 'after_arm': ba, 'after_cold': bc,
                              'sf': sf, **pair_stats(subset),
                              'prompt_token_changes': sum(a['prompt_token_sha256'] != b['prompt_token_sha256'] for a, b in subset),
                              'complete_output_changes': sum((a['output_token_ids'], a['finish_reason'], a['stop_reason']) !=
                                                             (b['output_token_ids'], b['finish_reason'], b['stop_reason']) for a, b in subset),
                              'sampling_unit': 'physical_source', 'shared_sources_across_cold_contrasts': True})
    return summaries, detailed


def timing_table(records, workers):
    output = []
    for arm in ARMS:
        for cold in (1, 2):
            for deep in (False, True):
                group = [r for r in records if (r['arm'], r['cold'], r['deep']) == (arm, cold, deep)]
                for stage in STAGES:
                    values = [r['timing'][stage] for r in group]
                    output.append({'arm': arm, 'cold': cold, 'stratum': 'deep_observer' if deep else 'ordinary',
                                   'stage': stage, 'n': len(values), 'mean_seconds': statistics.mean(values),
                                   'median_seconds': statistics.median(values), 'p90_seconds': quantile(values, .9),
                                   'requests_per_hour': 3600 / statistics.mean(values) if stage == 'end_to_end_durable_seconds' else None,
                                   'nested_not_additive': stage != 'end_to_end_durable_seconds'})
    for w in workers:
        output.append({'arm': w['arm'], 'cold': w['cold'], 'stratum': 'cold_load', 'stage': 'engine_load_seconds',
                       'n': 1, 'mean_seconds': w['load_seconds'], 'median_seconds': w['load_seconds'],
                       'p90_seconds': w['load_seconds'], 'nested_not_additive': False})
    return output


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, manifest_path, *, render=False):
    archive = Archive(root)
    manifest_path = Path(manifest_path)
    plan = read(manifest_path)
    ids, selected, attempts, budget = refs(archive, plan)
    records, workers = [], []
    for spec in plan['workers']:
        own, evidence = audit_worker(archive, spec, plan, ids, selected, attempts, budget)
        records.extend(own)
        workers.append(evidence)
    require(len(records) == 32 and len({(r['arm'], r['cold'], r['sample_id']) for r in records}) == 32,
            'Incomplete finite T1 study')
    require(len({w['actual_backend_signature'] for w in workers}) == 1 and
            len({w['engine_settings_sha256'] for w in workers}) == 1, 'Arms/cold engines changed backend/resource settings')
    require(len({digest(w['owned_workers']) for w in workers}) == 4 and
            len({w['worker'] for w in workers}) == 4, 'Independent cold engines were reused')
    pairs, detailed = pair_table(records, ids)
    accuracy = []
    for arm in ARMS:
        for cold in (1, 2):
            for sf in (None, 4, 8):
                own = [r for r in records if (r['arm'], r['cold']) == (arm, cold) and (sf is None or r['sf'] == sf)]
                accuracy.append({'arm': arm, 'cold': cold, 'sf': sf, 'n': len(own),
                                 'correct': sum(r['correct'] for r in own), 'sampling_unit': 'physical_source'})
    inputs = []
    for sid in ids:
        v, i = attempts[('video', sid)], attempts[('images', sid)]
        inputs.append({'sample_id': sid, 'video_prompt_tokens': v['prompt_tokens'], 'images_prompt_tokens': i['prompt_tokens'],
                       'prompt_token_delta_images_minus_video': i['prompt_tokens'] - v['prompt_tokens'],
                       'video_placeholder_counts': v['native_placeholder_counts'], 'images_placeholder_counts': i['native_placeholder_counts'],
                       'video_temporal_grid': [90, 32, 48], 'image_temporal_grid': [1, 32, 48], 'logical_image_count': 180,
                       'video_CPU_payload_sha256': v['pixel_payloads'][0]['sha256'],
                       'unique_image_CPU_patch_payloads': len({p['sha256'] for p in i['pixel_payloads']}),
                       'unique_source_frames': v['unique_source_frames'],
                       'video_template_text_sha256': v['template_text_sha256'], 'images_template_text_sha256': i['template_text_sha256'],
                       'video_expanded_prompt_sha256': v['prompt_token_ids_sha256'], 'images_expanded_prompt_sha256': i['prompt_token_ids_sha256']})
    output = archive.path(plan['run_root']) / 'analysis'
    synthetic = plan.get('evidence_kind') == 'SYNTHETIC_TEST_ONLY'
    require(not synthetic or output.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()), 'Synthetic fixtures must stay temporary')
    status = 'SYNTHETIC_TEST_ONLY' if synthetic else 'COMPLETE_FINITE_T1_STUDY'
    summary = {'schema': RESULT_SCHEMA, 'status': status, 'variant': plan['variant'], 'manifest': descriptor(manifest_path),
               'effective_calls': 32, 'physical_sources': 8, 'calls_per_arm_cold': 8, 'accuracy': accuracy,
               'paired_comparisons': pairs, 'input_comparison': inputs, 'workers': workers,
               'GPU_deep_observed_calls': sum(r['deep'] for r in records), 'ordinary_CPU_reader_prompt_verified_calls': sum(not r['deep'] for r in records),
               'input_organization': {'video': 'One native video: 180 decoded frames, 90 temporal pairs, native video timestamp layout.',
                                      'images': '180 ordered image_url items: 180 single-image grids, native single-image temporal padding, no video timestamp layout.',
                                      'timestamp_evidence': 'Actual expanded prompt IDs are retained in raw and match the per-ID CPU SHA. This analyzer does not decode or invent timestamp text.',
                                      'same': 'Original decoded RGB uint8 pixels, frame order, source geometry, scientific text and generation settings.',
                                      'different': 'Native template/token expansion, temporal packing, video timestamps and modality-specific preprocessing; not only input list syntax.'},
               'limits': ['Eight physical sources; the two cold comparisons share those sources and are not independent replications.',
                          'Actual GPU ingress is deeply observed for the first ID of each arm/cold worker only (4/32).',
                          'For the remaining 28 calls, all source pixels/readers, logical inputs and CPU-expanded prompts are verified; per-ID GPU hashes are unobserved.',
                          'Logical 180 images, unique payloads, observed GPU payloads and visual forward calls are separate counts.',
                          'Device normalization may be fused. An unobserved internal normalized tensor has no invented dtype.',
                          'Timing separates deep probes from ordinary requests; nested reader/processor times are not additive.',
                          'Cold refers to independent engines. Source-file page caches were warmed before timed requests.',
                          'No independent-trial CI, psychophysical threshold, or historical-C matching claim is computed.']}
    timing = timing_table(records, workers)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'summary.json', summary)
    atomic_json(output / 'accepted-records.json', records)
    write_csv(output / 'paired.csv', detailed)
    write_csv(output / 'timing.csv', timing)
    write_csv(output / 'accuracy.csv', accuracy)
    if render:
        render_figures(summary, output / 'figures')
    return {'status': status, 'effective_calls': 32, 'summary': descriptor(output / 'summary.json'),
            'paired': descriptor(output / 'paired.csv'), 'timing': descriptor(output / 'timing.csv')}


def render_figures(summary, output):
    if isinstance(summary, (str, Path)):
        summary = read(summary)
    output = Path(output).resolve()
    synthetic = summary['status'] == 'SYNTHETIC_TEST_ONLY'
    require(summary['schema'] == RESULT_SCHEMA and summary['effective_calls'] == 32 and
            (summary['status'] == 'COMPLETE_FINITE_T1_STUDY' or
             (synthetic and output.is_relative_to(Path(tempfile.gettempdir()).resolve()))), 'Only complete accepted counts may be plotted')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    from matplotlib.patches import Patch
    accuracy = {(r['arm'], r['cold']): r for r in summary['accuracy'] if r['sf'] is None}
    contrasts = [r for r in summary['paired_comparisons'] if r['sf'] is None]
    require(set(accuracy) == {(a, c) for a in ARMS for c in (1, 2)} and
            all(r['n'] == 8 for r in accuracy.values()) and len(contrasts) == 4 and
            all(r['n'] == 8 for r in contrasts), 'Figure denominators incomplete')
    style = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False,
             'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42}
    colors = {'video': '#0072B2', 'images': '#D55E00'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
        fig.subplots_adjust(left=.065, right=.985, bottom=.16, top=.69, wspace=.30)
        fig.suptitle(('SYNTHETIC TEST ONLY | ' if synthetic else '') + 'T1 | Video and 180 ordered images',
                     x=.065, y=.98, ha='left', fontsize=18, fontweight='bold')
        fig.text(.065, .87, summary['variant'] + ' | 8 paired sources: 4 SF4 + 4 SF8', fontsize=11)
        titles = ('Correct answers', 'Cold-repeat answer changes', 'Video-to-images answer changes')
        for ax, title in zip(axes, titles):
            ax.set_title(title, loc='left', fontsize=12, fontweight='bold')
            ax.set_ylim(0, 110)
            ax.set_yticks((0, 50, 100))
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.grid(axis='y', color='#DDDDDD', linewidth=.7)
            ax.set_axisbelow(True)
        for x, arm in enumerate(ARMS):
            for cold, offset in ((1, -.17), (2, .17)):
                n = accuracy[(arm, cold)]['correct']; value = 100 * n / 8
                axes[0].bar(x + offset, value, width=.29, edgecolor=colors[arm],
                            color=colors[arm] if cold == 1 else 'white', linewidth=1.7)
                axes[0].text(x + offset, value + 3, f'{n}/8', ha='center', fontsize=10)
        axes[0].set_xticks((0, 1), ('Video', '180 images'))
        axes[0].legend(handles=[Patch(facecolor='#555555', label='Cold 1'),
                               Patch(facecolor='white', edgecolor='#555555', label='Cold 2')],
                       frameon=False, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.30), ncol=2)
        repeat = {r['before_arm']: r for r in contrasts if r['contrast'] == 'cold_repeat'}
        cross = {r['before_cold']: r for r in contrasts if r['contrast'] == 'video_to_images'}
        for ax, keys, rows, labels in ((axes[1], ARMS, repeat, ('Video', '180 images')),
                                      (axes[2], (1, 2), cross, ('Cold 1', 'Cold 2'))):
            for x, key in enumerate(keys):
                n = rows[key]['answer_changes']; value = 100 * n / 8
                ax.bar(x, value, color=colors[key] if key in colors else '#555555', width=.58)
                ax.text(x, value + 3, f'{n}/8', ha='center', fontsize=11)
            ax.set_xticks((0, 1), labels)
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for ext in ('png', 'pdf'):
            path = output / ('T1-route-comparison.' + ext)
            fig.savefig(path, dpi=240, bbox_inches='tight', pad_inches=.13)
            paths[ext] = descriptor(path)
        plt.close(fig)
    return {'status': 'SYNTHETIC_TEST_ONLY' if synthetic else 'OBSERVED_FIGURES_EXPORTED', **paths}


def preflight(root, manifest_path):
    archive = Archive(root)
    plan = read(manifest_path)
    ids, _, _, _ = refs(archive, plan)
    missing = []
    observed_raw_calls = 0
    for spec in plan['workers']:
        profile = archive.ref(spec['profile'])
        out = archive.path(profile['output'])
        for path in [out / n for n in ('identity.json', 'raw.jsonl', 'events.jsonl', 'request-timings.jsonl',
                                      'deep-observations.jsonl', 'result.json', 'exit.json', 'lease.json',
                                      'engine-owner.json', 'runtime.json', 'public-settings.json', 'load.json', 'versions.json', 'cache-policy.json')] + [out.with_suffix('.wait.json'), out.with_suffix('.launch.json')]:
            if not path.exists():
                missing.append(str(path))
        if (out / 'raw.jsonl').exists():
            observed_raw_calls += len(journal(out / 'raw.jsonl'))
        for sid in ids:
            path = archive.path(profile['claim_root']) / (sid + '.json')
            if not path.exists():
                missing.append(str(path))
    return {'status': 'PREPARED_NOT_ACCEPTED', 'expected_calls': 32, 'observed_raw_calls': observed_raw_calls,
            'missing_paths': missing, 'scientific_completion_claimed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    result = preflight(args.root, args.manifest) if args.preflight else analyze(args.root, args.manifest, render=args.render)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
