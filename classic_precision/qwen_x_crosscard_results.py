"""Read-only, finite T2/T3 cross-card comparison; never issues model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import itertools
import json
from pathlib import Path
import statistics

from .official_x_results import (Archive, audit_worker, basic, descriptor, pair_stats,
                                 read, require, write_csv)
from .storage import atomic_json, digest, file_sha256

NAME = 'qwen-x-crosscard-t23-20261029'
CARDS = ('fourier0', 'fourier1', 'radon0')
OUTPUT_FIELDS = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason')
IDENTITY_FIELDS = ('task', 'truth', 'source_media', 'prompt_token_sha256')


def historical_comparison(archive, bindings, records, output):
    """Compare preserved A/C and earlier X without changing canonical selection."""
    def csv_rows(ref):
        path = archive.path(ref['path'])
        require(file_sha256(path) == ref['sha256'], 'Historical CSV identity changed')
        with path.open(newline='') as stream:
            return list(csv.DictReader(stream))

    selected = {r['sample_id'] for r in records}
    historical = {}
    historical_table_sha = {}
    for ref in bindings['A_tables']:
        for row in csv_rows(ref):
            if row['model'] == 'qwen' and row['sample_id'] in selected:
                require(row['sample_id'] not in historical, 'Duplicate historical A identity')
                historical[row['sample_id']] = row
                historical_table_sha[row['sample_id']] = ref['sha256']
    official = {r['sample_id']: r for r in csv_rows(bindings['C_pairs'])
                if r['model'] == 'qwen' and r['sample_id'] in selected}
    require(set(historical) == set(official) == selected, 'Historical coverage differs')
    center_summary = archive.ref(bindings['center_summary'])
    center_pairs = {(r['model'], r['sample_id']): r
                    for r in csv_rows(bindings['center_pairs'])}
    cached = {}
    old = {}
    for sid in selected:
        a, c = historical[sid], official[sid]
        path = archive.path(c['new_raw_file'])
        if path not in cached:
            cached[path] = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        raw = cached[path][int(c['new_raw_ordinal'])]
        require(raw['sample_id'] == sid and raw['arm'] == 'C'
                and digest(raw) == c['B_raw_digest'], 'Historical C is not the bound native arm')
        if c['historical_raw_digest']:
            require(a['raw_sha256'] == c['historical_raw_digest'], 'Historical A raw digest differs')
        else:
            # The earlier center-point export omitted this column. Recover its
            # existing, accepted binding via the center summary and paired row;
            # never treat an arbitrary missing digest as acceptable.
            require(c['point_role'] == 'center' and c['task'] == 'T2'
                    and float(c['point']) == .06, 'Unexpected missing historical digest')
            require(center_summary['bindings']['historical.csv'] == historical_table_sha[sid],
                    'Center historical table is not the accepted table')
            entry = center_pairs['qwen', sid]
            require(entry['historical_answer'] == a['answer']
                    and entry['C_answer'] == raw['raw_answer'].strip()
                    and entry['truth'] == a['truth']
                    and entry['source_video_sha256'] == a['source_video_sha256'],
                    'Center historical/source binding differs')
        require(a['answer'] == c['A_historical'] and a['truth'] == raw['truth'] == c['truth'],
                'A/C paired answer/GT differs')
        require(a['source_video_sha256'] == raw['source']['video_sha256']
                and a['source_stream_sha256'] == raw['source']['stream_sha256'], 'A/C source bytes differ')
        old[sid] = {
            'A': {'answer': a['answer'], 'raw_answer': a['answer'], 'correct': a['correct'] == '1',
                  'truth': a['truth'], 'output_token_ids': json.loads(a['output_token_ids']),
                  'finish_reason': a['finish_reason'], 'stop_reason': json.loads(a['stop_reason'])},
            'C': {'answer': raw['raw_answer'].strip(), 'raw_answer': raw['raw_answer'],
                  'correct': raw['correct'], 'truth': raw['truth'],
                  **{k: raw[k] for k in OUTPUT_FIELDS}},
            'source_sha256': a['source_video_sha256']}
    pairs, summaries = [], []
    for card in CARDS:
        for task in ('T2', 'T3'):
            current = [r for r in records if r['card'] == card and r['task'] == task]
            for protocol in ('A', 'C'):
                values = []
                for r in current:
                    h = old[r['sample_id']]
                    require(r['source_media'] == [('video', h['source_sha256'])], 'Historical/X source differs')
                    prior = h[protocol]
                    values.append((prior, r))
                    pairs.append({'card': card, 'task': task, 'sample_id': r['sample_id'],
                                  'before_protocol': protocol, 'before_answer': prior['answer'],
                                  'X_answer': r['answer'], 'before_correct': prior['correct'],
                                  'X_correct': r['correct'], 'answer_changed': prior['answer'] != r['answer'],
                                  'tokens_changed': prior['output_token_ids'] != r['output_token_ids']})
                summaries.append({'card': card, 'task': task, 'before_protocol': protocol,
                                  'after_protocol': 'X', **pair_stats(values)})
    previous = archive.ref(bindings['previous_X_records'])
    previous = [r for r in previous if r['model'] == 'qwen' and r['variant'] == 'X_QSA_NO_AUTOTUNE'
                and r['phase'] == 'validation' and r['sample_id'] in selected]
    require(Counter(r['replicate'] for r in previous) == {1: 48, 2: 48}
            and all(r['task'] == 'T2' for r in previous), 'Unexpected previous X overlap')
    lookup = {(r['card'], r['sample_id']): r for r in records}
    repeats = []
    for r in previous:
        for card in CARDS:
            now = lookup[card, r['sample_id']]
            require(r['truth'] == now['truth'] and r['source_media'] == [list(x) for x in now['source_media']],
                    'Earlier X source/GT differs')
            changed = [k for k in (*OUTPUT_FIELDS, 'prompt_token_sha256') if r[k] != now[k]]
            repeats.append({'card': card, 'sample_id': r['sample_id'], 'old_cold': r['replicate'],
                            'full_output_and_prompt_equal': not changed, 'changed_fields': changed})
    write_csv(output / 'historical-paired.csv', pairs)
    write_csv(output / 'historical-comparison.csv', summaries)
    write_csv(output / 'previous-X-repeat.csv', repeats)
    return {'bindings': bindings, 'same_unique_inputs': len(selected), 'comparisons': summaries,
            'previous_X_T2_pair_count': len(repeats),
            'previous_X_full_output_and_prompt_differences': sum(not x['full_output_and_prompt_equal'] for x in repeats),
            'attribution': 'A/C/X differ in input route and recorded software/backend settings; not a pure bit-depth effect.'}


def analyze(root, manifest_path):
    """Require three explicit SHA-bound sealed profiles and their real closure."""
    from .qwen_x_crosscard import science, validate_runtime
    root = Path(root)
    arc = Archive(root)
    manifest = read(manifest_path)
    require(manifest['schema'] == 'qwen-X-crosscard-analysis-v1', 'Wrong analysis manifest')
    require(manifest['maximum_new_model_calls'] == 432, 'Finite quota differs')
    plan = arc.ref(manifest['plan'])
    require(plan['schema'] == 'qwen-X-crosscard-t23-v1'
            and plan['maximum_new_model_calls'] == 432
            and plan['task_counts_per_card'] == {'T2': 48, 'T3': 96}, 'Wrong finite plan scope')
    require(len(manifest['profiles']) == 3, 'Exactly three physical workers required')
    records, proofs, seen, counts = [], [], set(), Counter()
    shared_ids = None
    science_settings = None
    recorded_versions = None
    for selected in manifest['profiles']:
        card = selected['card']
        require(card in CARDS and card not in seen, 'Duplicate/foreign card')
        seen.add(card)
        profile = arc.ref(selected['profile'])
        science(profile, resolve=arc.ref)
        require(profile['card'] == card and profile['phase'] == 'crosscard', 'Card/phase changed')
        require(len(profile['assigned_ids']) == len(set(profile['assigned_ids'])) == 144,
                'Per-card sample scope changed')
        require(profile['maximum_model_calls'] == 144 and profile['automatic_retry'] is False,
                'Call quota/retry policy changed')
        require(profile['prepared_plan_sha256'] == manifest['plan']['sha256'], 'Plan binding differs')
        require(profile['assigned_ids'] == plan['assigned_ids'], 'Selected IDs differ from frozen plan')
        auth = arc.ref(profile['execution_authorization'])
        require(auth['status'] == 'AUTHORIZED' and auth['plan_sha256'] == manifest['plan']['sha256']
                and auth.get('user_message'), 'Human authorization binding differs')
        settings = dict(profile['engine_kwargs'])
        settings['model'] = Path(settings['model']).name
        settings = {'engine': settings, 'request': profile['request_mm_processor_kwargs'],
                    'vision': profile['vision_observation'],
                    'selector_sha': profile['selector_source']['sha256']}
        if science_settings is None:
            science_settings = settings
            shared_ids = profile['assigned_ids']
        require(settings == science_settings and profile['assigned_ids'] == shared_ids,
                'Scientific settings or input order differ between cards')
        out = arc.path(profile['output'])
        require(not (out / 'failure.json').exists(), 'Failed/uncertain worker cannot be admitted')
        rows, proof = audit_worker(arc, {'profile': selected['profile'], 'phase': 'crosscard',
                                      'variant': 'X_QSA_NO_AUTOTUNE', 'replicate': 1})
        # Profiles inherit different extra startup assertions. Compare actual
        # recorded versions, while audit_worker retains every profile's pins.
        actual_versions = proof['recorded_package_versions']
        if recorded_versions is None:
            recorded_versions = actual_versions
        require(actual_versions == recorded_versions, 'Actual recorded package versions differ between cards')
        for value in proof['runtime']:
            validate_runtime(value)
        require(Counter(row['task'] for row in rows) == {'T2': 48, 'T3': 96}, 'Cell sizes differ')
        records.extend({**row, 'card': card} for row in rows)
        proofs.append({'card': card, **proof})
        counts[card] += len(rows)
    require(seen == set(CARDS) and sum(counts.values()) == 432, 'Three-card quota incomplete')
    lookup = {(row['card'], row['sample_id']): row for row in records}
    comparisons = []
    pair_summaries = []
    for left, right in itertools.combinations(CARDS, 2):
        pairs = [(lookup[left, sid], lookup[right, sid]) for sid in shared_ids]
        for a, b in pairs:
            identity_differences = [k for k in IDENTITY_FIELDS if a[k] != b[k]]
            require(not identity_differences, 'Cross-card source/GT/prompt identity differs')
            changed = [k for k in OUTPUT_FIELDS if a[k] != b[k]]
            comparisons.append({'left_card': left, 'right_card': right, 'task': a['task'],
                                'sample_id': a['sample_id'], 'left_answer': a['answer'],
                                'right_answer': b['answer'], 'left_correct': a['correct'],
                                'right_correct': b['correct'], 'full_output_equal': not changed,
                                'changed_fields': changed})
        for task in ('T2', 'T3'):
            subset = [(a, b) for a, b in pairs if a['task'] == task]
            pair_summaries.append({'left_card': left, 'right_card': right, 'task': task,
                                   **pair_stats(subset),
                                   'full_output_differences': sum(any(a[k] != b[k] for k in OUTPUT_FIELDS)
                                                                  for a, b in subset)})
    timings = []
    for card in CARDS:
        for task in ('T2', 'T3'):
            rows = [r for r in records if r['card'] == card and r['task'] == task and not r['deep']]
            values = sorted(r['timing']['end_to_end_durable_seconds'] for r in rows)
            require(values, 'Missing ordinary request timing')
            timings.append({'card': card, 'task': task, 'ordinary_n': len(values),
                            'mean_seconds': statistics.mean(values), 'median_seconds': statistics.median(values),
                            'p90_seconds': values[min(len(values)-1, int(.9*(len(values)-1)))],
                            'request_videos_per_hour': 3600/statistics.mean(values)})
    different = [r for r in comparisons if not r['full_output_equal']]
    out = root / 'runs/reproducibility' / NAME / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / 'canonical-records.json', records)
    write_csv(out / 'paired.csv', comparisons)
    write_csv(out / 'timing.csv', timings)
    write_csv(out / 'comparison.csv', pair_summaries)
    result = {'schema': 'qwen-X-crosscard-results-v1',
              'status': 'PASS_FIXED144_THREE_CARDS' if not different else 'COMPLETE_WITH_CROSSCARD_DIFFERENCES',
              'technical_status': 'COMPLETE_PASS', 'new_analysis_model_calls': 0,
              'executed_model_calls': 432, 'unique_video_ids': 144,
              'independent_cold_starts_per_card_this_round': 1,
              'plan': manifest['plan'], 'analysis_manifest': descriptor(manifest_path),
              'cards': {card: {task: basic(r for r in records if r['card'] == card and r['task'] == task)
                               for task in ('T2', 'T3')} for card in CARDS},
              'pair_summaries': pair_summaries, 'full_output_differences': different,
              'workers': proofs,
              'limits': ['Three specific cards, one fresh engine each; not arbitrary hardware or a new two-cold test.',
                         'Same X answers across cards do not imply equality to A or C.',
                         'Historical ablations justify retaining the controls for this stack; they do not prove universal necessity.',
                         'Ordinary request timings exclude model load and inter-request scheduler/claim overhead.']}
    if manifest.get('historical_comparison'):
        result['historical_comparison'] = historical_comparison(arc, manifest['historical_comparison'], records, out)
    result['outputs'] = {p.name: descriptor(p) for p in out.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(out / 'summary.json', result)
    return {k: result[k] for k in ('status', 'technical_status', 'executed_model_calls', 'unique_video_ids', 'cards')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--manifest', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.manifest)), flush=True)


if __name__ == '__main__':
    main()
