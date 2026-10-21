"""Recompute complete denominators from durable outputs, never from explanations."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

from .manifest import atomic_json, digest, file_sha, validate_manifest
from .storage import classify_output, digest


def journal(path):
    """Read complete records only; an active writer's final partial line is pending."""
    path = Path(path)
    if not path.exists():
        return [], False
    payload = path.read_bytes()
    lines = payload.splitlines(keepends=True)
    tail = bool(lines and not lines[-1].endswith(b'\n'))
    return [json.loads(line) for line in (lines[:-1] if tail else lines) if line.strip()], tail


def model_snapshot(directory):
    """Take a consistent short read while another local shard may append."""
    import contextlib
    import fcntl
    directory = Path(directory)
    lock = directory/'.model-state.lock'
    with lock.open('rb') if lock.exists() else contextlib.nullcontext(None) as stream:
        if stream is not None:
            fcntl.flock(stream, fcntl.LOCK_SH)
        identity = json.loads((directory/'identity.json').read_text())
        raw, raw_tail = journal(directory/'outputs.jsonl')
        events, event_tail = journal(directory/'events.jsonl')
        return identity, raw, events, raw_tail, event_tail


def technical_passed(rows, events):
    """Return IDs whose durable attempt has a current successful input receipt."""
    by_id = {r['sample_id']: r for r in rows}
    verdicts = {}
    for event in events:
        raw = by_id.get(event.get('sample_id'))
        if raw is None:
            continue
        attempt = event.get('attempt_id') or raw['attempt_id']
        if attempt != raw['attempt_id']:
            continue
        kind = event['event']
        if kind not in ('TECHNICAL_PASS', 'TECHNICAL_PASS_RECOVERED', 'POST_OUTPUT_VALIDATION_FAILED'):
            continue
        failed = kind == 'POST_OUTPUT_VALIDATION_FAILED'
        stamp = (event.get('time_unix', 0), failed)
        sid = raw['sample_id']
        if sid not in verdicts or stamp >= verdicts[sid][0]:
            verdicts[sid] = (stamp, not failed)
    return {sid for sid, (_, passed) in verdicts.items() if passed}


def cell_key(row):
    p = row['source_parameters']
    return (('T2', p['background_Y'], p['offset_frame']-p['onset_frame'], p['delta_Y'])
            if row['task'] == 'T2' else ('T3', p['angle_deg']))


def attempt_metrics(rows, events):
    """Keep unique answers separate from attempts, including unresolved crashes."""
    unique = {digest(event): event for event in events}
    attempts = {}
    terminals = {}
    for event in unique.values():
        aid = event.get('attempt_id')
        if not aid:
            continue
        if event['event'] == 'INFLIGHT':
            if aid in attempts and attempts[aid] != event:
                raise ValueError('Conflicting attempt identity')
            attempts[aid] = event
        elif event['event'] in ('FAILED', 'INTERRUPTED'):
            if aid not in terminals or event.get('time_unix', 0) >= terminals[aid].get('time_unix', 0):
                terminals[aid] = event
    answered = {row['attempt_id'] for row in rows}
    counts = Counter(event['sample_id'] for event in attempts.values())
    if any(count > 2 for count in counts.values()):
        raise ValueError('More than two attempts for a scientific row')
    no_output = set(attempts)-answered
    failed = [terminals[a] for a in no_output if a in terminals]
    times = [e['elapsed_seconds'] for e in failed if e.get('elapsed_seconds') is not None]
    complete_ids = {row['sample_id'] for row in rows}
    exhausted = sorted({e['sample_id'] for e in unique.values() if e['event'] == 'EXHAUSTED'}-complete_ids)
    return {'started_attempts': len(attempts), 'retry_attempts': sum(max(0, n-1) for n in counts.values()),
            'retried_sample_ids': sorted(s for s, n in counts.items() if n > 1),
            'completed_unique_answers': len(rows), 'answers_missing_attempt_event': len(answered-set(attempts)),
            'no_output_attempts': len(no_output),
            'no_output_failed': sum(e['event'] == 'FAILED' for e in failed),
            'no_output_interrupted': sum(e['event'] == 'INTERRUPTED' for e in failed),
            'no_output_inflight_or_unclosed': len(no_output-set(terminals)),
            'known_failed_attempt_seconds': sum(times),
            'no_output_attempts_without_elapsed_time': len(no_output)-len(times),
            'exhausted_sample_ids': exhausted, 'max_attempts_per_sample': 2}


def metrics(expected, observed, technical_pass_ids=None):
    predictions = [classify_output(r) for r in observed]
    n, total = len(observed), len(expected)
    correct = sum(r['correct'] for r in predictions)
    by_id = {r['sample_id']: r for r in predictions}
    groups = defaultdict(list)
    direction_pairs = defaultdict(list)
    for row in expected:
        p = row['source_parameters']
        groups[p['onset_index']].append(row['sample_id'])
        key = cell_key(row) + (p['onset_index'],) + ((p['tilt_sign'],) if row['task'] == 'T3' else ())
        direction_pairs[key].append(row['sample_id'])
    paired = [all(by_id[x]['correct'] for x in ids) for ids in groups.values()
              if all(x in by_id for x in ids)]
    interval = None
    unresolved = sorted(r['sample_id'] for r in observed
                        if technical_pass_ids is not None and r['sample_id'] not in technical_pass_ids)
    if n == total and total and not unresolved:
        import numpy as np
        # The resampling unit is an onset block, including both sides and signs.
        values = np.array([sum(by_id[x]['correct'] for x in ids)/len(ids)
                           for ids in groups.values()])
        rng = np.random.default_rng(0)
        estimates = values[rng.integers(0, len(values), (4000, len(values)))].mean(axis=1)
        interval = np.quantile(estimates, [.025, .975]).tolist()
    pairs = [ids for ids in direction_pairs.values() if all(x in by_id for x in ids)]
    return {'expected': total, 'completed': n, 'missing': total-n, 'correct': correct,
            'technical_unresolved_ids': unresolved,
            'strict_correct': sum(r['correct'] and r['raw_exact_choice'] for r in predictions),
            'accuracy': correct/total if total and n == total and not unresolved else None,
            'observed_accuracy': correct/n if n else None,
            'prediction_counts': dict(Counter(r['answer'] or 'INVALID' for r in predictions)),
            'finish_counts': dict(Counter(r['finish_reason'] for r in predictions)),
            'raw_exact_choice': sum(r['raw_exact_choice'] for r in predictions),
            'format_issues': sum(not r['raw_exact_choice'] for r in predictions),
            'invalid_or_length': sum(r['answer'] is None for r in predictions),
            'truth_counts': dict(Counter(r['truth'] for r in expected)),
            'correct_by_truth': {side: sum(r['correct'] and r['truth'] == side for r in predictions)
                                 for side in ('LEFT', 'RIGHT')},
            'onset_blocks': len(groups), 'fully_correct_onset_blocks': sum(paired),
            'completed_onset_blocks': len(paired), 'pointwise_95pct_onset_bootstrap': interval,
            'direction_pairs_expected': len(direction_pairs), 'direction_pairs_completed': len(pairs),
            'direction_pairs_both_correct': sum(all(by_id[x]['correct'] for x in ids) for ids in pairs)}


def summarize(manifest_path, output_dirs, destination, require_complete=False):
    manifest = validate_manifest(json.loads(Path(manifest_path).read_text()))
    manifest_sha = file_sha(manifest_path)
    lookup = {r['sample_id']: r for r in manifest['rows']}
    records, events, sources, incomplete_tails, system_signatures = {}, [], [], [], {}
    input_bindings = {}
    for root in dict.fromkeys(str(Path(x).resolve()) for x in output_dirs):
        for family in manifest['models']:
            directory = Path(root)/family
            if not directory.exists():
                continue
            identity, raw, local_events, raw_tail, event_tail = model_snapshot(directory)
            from .inference import GENERATION
            from .ready import INPUT_BINDING, validate_record
            if (identity['manifest_sha256'] != manifest_sha
                    or identity['protocol_id'] != manifest['protocol_id'] or identity['model_key'] != family
                    or identity['generation'] != GENERATION or identity['model'] != manifest['models'][family]
                    or identity.get('schema_version') != 2 or identity.get('input_binding') != INPUT_BINDING):
                raise ValueError('Output identity differs from frozen protocol: '+str(directory))
            binding = ('manifest_sha256', 'input_binding', 'ready_source_sha256', 'calibration_identity_sha256',
                       'model', 'generation', 'effective_resources',
                       'checkpoint_assets', 'cosmos_package_sources', 'runtime_versions',
                       'runner_sha256', 'storage_sha256', 'all_row_ids_sha256')
            signature = {k: identity[k] for k in binding}
            if family in system_signatures and signature != system_signatures[family]:
                raise ValueError('Different runner/SDK/checkpoint/input bindings cannot share one model curve')
            system_signatures[family] = signature
            if len({r['sample_id'] for r in raw}) != len(raw):
                raise ValueError('Duplicate output ID within one authoritative raw journal')
            if raw_tail:
                incomplete_tails.append(str(directory/'outputs.jsonl'))
            for r in raw:
                row = lookup.get(r['sample_id'])
                if row is None or any(r.get(k) != row[k] for k in ('task', 'truth', 'source_parameters')):
                    raise ValueError('Output source/GT differs from the scientific row')
                sid = r['sample_id']
                row_binding = r.get('row_identity')
                if not isinstance(row_binding, str) or len(row_binding) != 64:
                    raise ValueError('Raw output lacks a frozen per-row input identity')
                if sid in input_bindings and input_bindings[sid] != row_binding:
                    raise ValueError('Different models/hosts consumed different ready videos for the same scientific row')
                input_bindings[sid] = row_binding
                receipt = r.get('ready_record')
                if not isinstance(receipt, dict):
                    raise ValueError('Version2 raw output requires its full immutable ready receipt')
                validate_record(row, receipt)
                if row_binding != digest({'row': row, 'ready': receipt}) or r.get('ready_identity') != digest(receipt):
                    raise ValueError('Raw per-row ready receipt differs from its persisted identity')
                key = family, sid
                if key in records and records[key] != r:
                    raise ValueError('Two distinct durable answers for one model/ID')
                records[key] = r
            if event_tail:
                incomplete_tails.append(str(directory/'events.jsonl'))
            events.extend((family, event) for event in local_events)
            sources.append({'directory': str(directory), 'completed': len(raw), 'identity': identity})
    valid_ids = {family: technical_passed([r for (model, _), r in records.items() if model == family],
                                          [e for model, e in events if model == family])
                 for family in manifest['models']}
    cells, results = [], []
    model_results = {}
    for family in manifest['models']:
        observed = {sid: r for (model, sid), r in records.items() if model == family}
        model_results[family] = metrics(manifest['rows'], list(observed.values()), valid_ids[family])
        groups = defaultdict(list)
        for row in manifest['rows']:
            groups[cell_key(row)].append(row)
            result = {'model': family, **row, 'completed': row['sample_id'] in observed,
                      'technical_pass': row['sample_id'] in valid_ids[family]}
            if row['sample_id'] in observed:
                raw = observed[row['sample_id']]
                result.update(classify_output(raw), seconds=raw['seconds'],
                              input_tokens=raw['actual_input_tokens'],
                              output_tokens=len(raw['output_token_ids']))
            results.append(result)
        for key, expected in groups.items():
            cells.append({'model': family, 'cell': list(key),
                          **metrics(expected, [observed[r['sample_id']] for r in expected
                                               if r['sample_id'] in observed], valid_ids[family])})
    complete = all(m['missing'] == 0 and not m['technical_unresolved_ids'] for m in model_results.values()) and not incomplete_tails
    if require_complete and not complete:
        raise ValueError('The four-model panel is incomplete; no final report was written')
    # Same-onset, same-side paired duration comparison, not independent accuracy pools.
    duration_pairs = {}
    for family in manifest['models']:
        groups = defaultdict(dict)
        for row in manifest['rows']:
            p = row['source_parameters']
            if row['task'] == 'T2' and p['delta_Y'] in manifest['protocol']['T2']['control_10frame_delta_Y']:
                raw = records.get((family, row['sample_id']))
                if raw and row['sample_id'] in valid_ids[family]:
                    key = (p['background_Y'], p['delta_Y'], p['onset_index'], row['truth'])
                    groups[key][p['offset_frame']-p['onset_frame']] = classify_output(raw)['correct']
        paired = [v for v in groups.values() if set(v) == {10, 30}]
        duration_pairs[family] = {'expected_pairs': 432, 'completed_pairs': len(paired),
            '500ms_wins': sum(v[30] and not v[10] for v in paired),
            '500ms_losses': sum(v[10] and not v[30] for v in paired),
            'ties': sum(v[10] == v[30] for v in paired)}
    # Deduplicate copies of events fetched from remote roots; never sum snapshots twice.
    unique_events = {(family, digest(event)): event for family, event in events}
    loads = defaultdict(float)
    for (family, _), event in unique_events.items():
        if event['event'] == 'MODEL_READY':
            loads[family] += event['load_seconds']
    costs = {}
    attempts = {}
    for family in manifest['models']:
        calls = [r for (model, _), r in records.items() if model == family]
        attempts[family] = attempt_metrics(calls, [e for (model, _), e in unique_events.items() if model == family])
        seconds = sum(r['seconds'] for r in calls)
        costs[family] = {'completed_call_seconds': seconds, 'model_load_seconds': loads[family],
                         'model_load_plus_completed_calls_GPUh': (seconds+loads[family])/3600,
                         'output_tokens': sum(len(r['output_token_ids']) for r in calls)}
    summary = {'status': 'COMPLETE' if complete else 'INCOMPLETE',
               'protocol_id': manifest['protocol_id'], 'models': model_results, 'cells': cells,
               'T2_500ms_vs167ms': duration_pairs, 'cost': costs, 'attempts': attempts,
               'cost_definition': 'Same-process monotonic model-load + returned-call durations. '
                    'External decode/prepare, failed calls without output, inflight and idle gaps excluded. '
                    'Processing inside the engine generate call is included; not allocation wall time.',
               'interval_definition': 'Pointwise percentile bootstrap over 24 onset blocks (4000 resamples, seed0). '
                    'Paired sides/signs stay together. Fixed contents/onsets are not a random population sample; '
                    'a 100% block-bootstrap interval is not proof of zero population error.',
               'incomplete_journal_tails': incomplete_tails, 'sources': sources}
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    atomic_json(dest/'summary.json', summary)
    for name, rows in [('results.jsonl', results), ('cells.jsonl', cells)]:
        from .manifest import canonical
        temporary = dest/(name+'.partial')
        with temporary.open('wb') as stream:
            for row in rows:
                stream.write(canonical(row)+b'\n')
        temporary.replace(dest/name)
    return summary


def plot(summary, destination):
    """Raw measured points only; no forced-monotone curve."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest = Path(destination)
    for task in ('T2', 'T3'):
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True, constrained_layout=True)
        for ax, family in zip(axes, summary['models']):
            groups = defaultdict(list)
            for cell in summary['cells']:
                if cell['model'] != family or cell['cell'][0] != task or cell['accuracy'] is None:
                    continue
                key = cell['cell']
                groups[(key[1], key[2]) if task == 'T2' else ('angles',)].append(cell)
            for key, cells in sorted(groups.items()):
                cells.sort(key=lambda c: c['cell'][-1])
                label = f'BG {key[0]:g}, {key[1]/60*1000:.0f} ms' if task == 'T2' else '500 ms'
                ax.plot([c['cell'][-1] for c in cells], [100*c['accuracy'] for c in cells],
                        marker='o', ms=3, lw=1.3, label=label)
            ax.axhline(50, color='#adb5bd', ls='--', lw=.7)
            ax.set(title=family, ylim=(0, 103), xlabel='Light-dot increment ΔY' if task == 'T2' else 'Angle difference (°)')
            ax.spines[['top', 'right']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
            if groups:
                ax.legend(fontsize=7, frameon=False)
        axes[0].set_ylabel('Accuracy (%)')
        fig.savefig(dest/(task.lower()+'_curve.png'), dpi=180)
        plt.close(fig)
