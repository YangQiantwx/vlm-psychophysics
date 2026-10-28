"""Read-only reconciliation of the finite Gemma X speed study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics

from .storage import atomic_json, digest, file_sha256


def read(p):
    return json.loads(p.read_text())


def rows(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s]


def stats(xs):
    xs = sorted(xs)
    if not xs:
        return {'n': 0}
    i = (len(xs) - 1) * .9; lo = int(i)
    return {'n': len(xs), 'mean_seconds': statistics.mean(xs),
            'median_seconds': statistics.median(xs),
            'p90_seconds': xs[lo] + (xs[min(lo+1, len(xs)-1)] - xs[lo]) * (i-lo),
            'videos_per_hour': 3600 * len(xs) / sum(xs)}


def analyze(root, partial=False):
    root = Path(root); plan_dir = root/'plans/gemma-x-speed-20261028'
    out = root/'runs/reproducibility/gemma-x-speed-20261028'
    plan = read(plan_dir/'plan.json'); expected = plan['assigned_ids']
    results = {}; records = {}; errors = []; physical_calls = 0; profiles = {}
    allocation_path = plan_dir/'final-arm-allocation-v2.json'
    pointer = plan_dir/'active-allocation.json'
    if pointer.exists():
        ref = read(pointer); allocation_path = plan_dir/Path(ref['path']).name
        assert file_sha256(allocation_path) == ref['sha256']
    allocation = read(allocation_path) if allocation_path.exists() else None
    profile_paths = ([plan_dir/x for x in allocation['profile_files']] if allocation
                     else sorted(plan_dir.glob('*-profile.json')))
    if allocation:
        for name in allocation.get('retired_unlaunched_profiles', []):
            retired = read(plan_dir/name)
            assert not (out/Path(retired['output']).name).exists()
            assert not (out/Path(retired['claim_root']).name).exists()
    for pf in profile_paths:
        p = read(pf); arm = p['optimization_arm']; d = out/Path(p['output']).name
        profiles[arm] = p
        raw = rows(d/'raw.jsonl') if (d/'raw.jsonl').exists() else []
        physical_calls += len(raw)
        wf = d.with_suffix('.wait.json')
        if not wf.exists():
            errors.append({'arm': arm, 'status': 'NOT_CLOSED', 'raw': len(raw)})
            continue
        w = read(wf)
        if w['returncode'] != 0 or w['terminal_status'] != 'COMPLETE':
            errors.append({'arm': arm, 'status': 'FAILED_RETAINED', 'wait': w, 'raw': len(raw)})
            continue
        assert w['real_wait_observed'] and w['owned_engines_exited'] and w['original_physical_lease_released']
        assert w['profile_sha256'] == digest(p)
        assert [r['sample_id'] for r in raw] == p['assigned_ids'] == expected
        passes = {e['sample_id']: e for e in rows(d/'events.jsonl') if e['event'] == 'TECHNICAL_PASS'}
        timing = rows(d/'request-timings.jsonl'); assert [r['sample_id'] for r in timing] == expected
        claim_dir = out/Path(p['claim_root']).name
        for r in raw:
            sid = r['sample_id']; assert passes[sid]['raw_digest'] == digest(r)
            assert r['profile_sha256'] == digest(p)
            claim = read(claim_dir/(sid+'.json')); assert claim['profile_sha256'] == digest(p)
            assert r['finish_reason'] == 'stop'
            assert len(r['native_reader']) == 1
            reader = r['native_reader'][0]
            assert reader['shape'][0] == 180 and reader['metadata']['frames_indices'] == list(range(180))
        task_stats = {}
        for t in sorted({r['task'] for r in raw}):
            task_raw = [r for r in raw if r['task'] == t]
            tt = [r for r in timing if r['task'] == t]
            task_stats[t] = {'correct': sum(r['correct'] is True for r in task_raw), 'n': len(task_raw),
                'ordinary_e2e': stats([r['end_to_end_durable_seconds'] for r in tt if not r['deep']]),
                'cold_observation_e2e': [r['end_to_end_durable_seconds'] for r in tt if r['deep']]}
        results[arm] = {'status': 'COMPLETE_PASS', 'n': len(raw),
            'ordinary_e2e': stats([r['end_to_end_durable_seconds'] for r in timing if not r['deep']]),
            'ordinary_engine_inclusive': stats([r['engine_request_inclusive_seconds'] for r in timing if not r['deep']]),
            'tasks': task_stats, 'wait': w, 'profile_sha256': file_sha256(pf),
            'raw_sha256': file_sha256(d/'raw.jsonl'), 'timings_sha256': file_sha256(d/'request-timings.jsonl')}
        records[arm] = {r['sample_id']: r for r in raw}
    maximum_calls = allocation['maximum_calls'] if allocation else plan['maximum_scientific_calls']
    assert physical_calls <= maximum_calls
    baseline = records.get('baseline_eager', {})
    for arm, rr in records.items():
        if not baseline:
            break
        changes = []; full_changes = []; prompt_changes = []
        original = profiles['baseline_eager']
        for key in ('checkpoint','checkpoint_assets','gpu_uuid','physical_lock','source_catalog','environment'):
            assert profiles[arm][key] == original[key], (arm, key)
        for sid, r in rr.items():
            a = baseline[sid]
            assert r['source'] == a['source'] and r['question'] == a['question'] and r['generation'] == a['generation']
            assert r['native_reader'][0]['shape'] == a['native_reader'][0]['shape']
            if r['raw_answer'] != a['raw_answer']: changes.append(sid)
            if any(r[k] != a[k] for k in ('output_token_ids','finish_reason','stop_reason')): full_changes.append(sid)
            if r['actual_prompt_token_ids'] != a['actual_prompt_token_ids']: prompt_changes.append(sid)
        results[arm]['versus_baseline'] = {'answer_changes': changes, 'full_output_changes': full_changes,
            'prompt_changes': prompt_changes,
            'ordinary_e2e_reduction_percent': 100*(1-results[arm]['ordinary_e2e']['mean_seconds']/results['baseline_eager']['ordinary_e2e']['mean_seconds'])}
    status = 'COMPLETE_FINITE_SPEED_STUDY' if not errors else 'PARTIAL'
    if not partial:
        assert not any(e['status'] == 'NOT_CLOSED' for e in errors), errors
        allocation = allocation or read(plan_dir/'final-arm-allocation.json')
        expected_arms = {x.replace('baseline-eager','baseline_eager') for x in allocation['arms']}
        assert set(profiles) == expected_arms
        if errors:
            status = 'COMPLETE_STUDY_WITH_FAILED_CANDIDATES'
    value = {'schema': 'gemma-x-speed-summary-v1', 'status': status, 'raw_calls': physical_calls,
        'maximum_calls': maximum_calls, 'arms': results, 'exceptions': errors,
        'limits': ['Fixed small 12-video same-card panel, not whole benchmark.',
                   'Ordinary timings exclude first observation per task; each arm has a fresh engine.',
                   'Engine time includes embedded processor/IPC; not GPU-only.',
                   'Model correctness is not used to choose input samples or retry calls.']}
    atomic_json(out/('partial-summary.json' if partial else 'summary.json'), value)
    return value


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--partial', action='store_true')
    a = p.parse_args(); v = analyze(a.root, a.partial)
    print(json.dumps({'status': v['status'], 'raw_calls': v['raw_calls'],
        'arms': {k: {'ordinary': x['ordinary_e2e'], 'comparison': x.get('versus_baseline')}
                 for k, x in v['arms'].items()}, 'exceptions': v['exceptions']}, indent=2))


if __name__ == '__main__':
    main()
