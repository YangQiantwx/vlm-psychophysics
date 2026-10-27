"""CPU fixtures for exact scientific partition and source-bound durable evidence."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pytest

from classic_precision import qwen_t2t3_five_pass as entry
from classic_precision.manifest import selected_rows
from classic_precision.summary import cell_key
from classic_precision.storage import atomic_json, digest

B = Path('/Users/qianyang/Desktop/cvpr2027_benchmark')
D = B/'runs/reproducibility'
OP = D/'remote-fourier/operator'


def config(gpu='0'):
    source = entry.read(OP/f'early-stop-qsa-v3-fourier-gpu{gpu}-plan.json')
    value = {'panels': {}, 'baseline_environment': source['profile']['baseline_environment'],
        'host': 'cirplab-fourier', 'gpu': gpu, 'gpu_uuid': source['gpu_uuid'], 'wave_id': 'CPU-five-pass-fixture',
        'stop_path': '/CPU-five-pass/STOP.json', 'model_root': '/home/yang/cvpr2027_benchmark/runs/inference_models_20261003',
        'gpu_lock_root': '/home/yang/cvpr2027_benchmark/runs/t2_t3_classic_rgb16_production_20261020/outputs',
        'admission': {'report': str(OP/'early-stop-qsa-v3-fourier-comparison.json'),
            'plans': [str(OP/f'early-stop-qsa-v3-fourier-gpu{i}-plan.json') for i in (0, 1)]},
        'launch_environment': source['launch_environment']}
    for name, folder in [('original', 't2-t3-rgb16'), ('supplement', 't3-threshold-refinement')]:
        root = B/'runs'/folder
        value['panels'][name] = {'manifest': str(root/'manifest.json'), 'ready': str(root/'remote-fourier/ready.jsonl'),
            'asset_root': str(root/'remote-fourier'), 'calibration_root': str(root/'remote-fourier/calibration'),
            'old_outputs': [str(root/f'remote-{host}/outputs/qwen') for host in ('fourier', 'radon')]}
        if name == 'supplement': value['panels'][name]['compat'] = str(root/'remote-fourier/operator/t3_generation_compat.py')
    return value


def test_two_onset_shards_exactly_cover_4080_and_balance_every_cell():
    plans = [entry.build_plan(config(str(i))) for i in (0, 1)]
    assert plans[0]['wave_sha256'] == plans[1]['wave_sha256']
    assert [p['max_model_calls'] for p in plans] == [10200, 10200]
    assert all(p['global_max_model_calls'] == 20400 and p['profile']['torch_policy'] == {} for p in plans)
    for name, count in entry.COUNTS.items():
        manifest = entry.read(plans[0]['panels'][name]['manifest']['path'])
        ids = [set(p['panels'][name]['sample_ids']) for p in plans]
        assert not ids[0] & ids[1] and ids[0] | ids[1] == {r['sample_id'] for r in manifest['rows']}
        assert list(map(len, ids)) == [count//2, count//2]
        counters = []
        for gpu in (0, 1):
            selected = selected_rows(manifest, gpu, 2)
            assert Counter(r['truth'] for r in selected) == {'LEFT': count//4, 'RIGHT': count//4}
            counter = Counter((cell_key(r), r['truth'], r['source_parameters'].get('tilt_sign')) for r in selected)
            counters.append(counter)
            for cell in {cell_key(r) for r in selected}:
                group = [r for r in selected if cell_key(r) == cell]
                sides = Counter(r['truth'] for r in group)
                assert sides['LEFT'] == sides['RIGHT']
                signs = Counter(r['source_parameters']['tilt_sign'] for r in group if r['task'] == 'T3')
                if cell == ('T3', 90): assert signs == {1: 24}  # native manifest deduplicates equivalent +/-90 axes
                else: assert signs[-1] == signs[1]
        assert counters[0] == counters[1]


def test_real_old_cross_host_union_and_plan_roundtrip(tmp_path):
    plan = entry.build_plan(config())
    path = tmp_path/'plan.json'; atomic_json(path, plan)
    assert entry.load_plan(path) == plan
    assert len(plan['panels']['original']['old_stores']) == len(plan['panels']['supplement']['old_stores']) == 2
    assert plan['panels']['supplement']['compat']['sha256'] == entry.COMPAT_SHA
    plan['profile']['torch_policy'] = {'deterministic_algorithms': True}; atomic_json(path, plan)
    with pytest.raises(ValueError): entry.load_plan(path)


@pytest.fixture
def saved_call(tmp_path):
    """Project one real complete GPU call; no fabricated runtime/profile fields."""
    source = entry.read(OP/'early-stop-qsa-v3-fourier-gpu0-plan.json')
    original = D/'remote-fourier/early-stop-qsa-v3/fourier-gpu0/repeat-0/qwen'
    event_lines = (original/'events.jsonl').read_bytes().splitlines(keepends=True)
    first = next(i for i, line in enumerate(event_lines) if json.loads(line)['event'] == 'INFLIGHT')
    inflight = json.loads(event_lines[first]); sid = inflight['sample_id']
    last = next(i for i in range(first, len(event_lines)) if json.loads(event_lines[i]).get('event') == 'EARLY_CALL_COMPLETE'
        and json.loads(event_lines[i]).get('sample_id') == sid)
    prefix, events = b''.join(event_lines[:first]), b''.join(event_lines[first:last+1])
    raw = next(json.loads(line) for line in (original/'outputs.jsonl').read_text().splitlines() if json.loads(line)['sample_id'] == sid)
    plan = {'profile': source['profile'], 'gpu': '0', 'gpu_uuid': source['gpu_uuid'],
        'implementation': entry.implementation(), 'panels': {'original': {'manifest': {'path': source['manifest']}, 'sample_ids': [sid]}}}
    # The source manifest descriptor in the old remote plan relocates to this exact local mirror.
    plan['panels']['original']['manifest']['path'] = str(B/'runs/t2-t3-rgb16/manifest.json')
    root = entry.directory(tmp_path, 0, 'original'); store = root/'repeat-0/qwen'; store.mkdir(parents=True)
    identity = entry.read(original/'identity.json'); identity['purpose'] = entry.SCHEMA
    identity['reproducibility'] = {'plan_sha256': digest(plan), 'profile': plan['profile'], 'arm': 'original',
        'repeat': 0, 'sample_ids': [sid], 'implementation': plan['implementation']}
    atomic_json(store/'identity.json', identity)
    raw['reproducibility'] = {'plan_sha256': digest(plan), 'arm': 'original', 'repeat': 0}
    (store/'outputs.jsonl').write_text(json.dumps(raw)+'\n'); (store/'events.jsonl').write_bytes(prefix+events)
    owner = next(json.loads(line)['owner'] for line in event_lines if json.loads(line)['event'] == 'WORKER_START')
    atomic_json(root/'claim.json', {'plan_sha256': digest(plan), 'panel': 'original', 'cold': 0, 'owner': owner, 'run_id': 'real-saved-call-projection'})
    row = next(r for r in entry.read(plan['panels']['original']['manifest']['path'])['rows'] if r['sample_id'] == sid)
    sources = {sid: (row, raw['ready_record'])}
    proof = entry.proof_for(root, sid, len(prefix), 0, len(prefix))
    return tmp_path, plan, sid, sources, proof, store


def test_actual_saved_gpu_evidence_passes_source_bound_range_reader(saved_call, monkeypatch):
    root, plan, sid, sources, proof, _ = saved_call
    from classic_precision import summary
    monkeypatch.setattr(summary, 'model_snapshot', lambda *a: (_ for _ in ()).throw(AssertionError('No full-journal reread')))
    value = entry.call_evidence(root, plan, 'original', 0, sid, sources, proof)
    assert value['sample_id'] == sid and value['raw']['output_token_ids'] and value['runtime']['gpu_uuid']
    assert value['native_identity']['purpose'] == entry.SCHEMA
    assert entry.evidence_reader() is not entry.early and entry.early.SCHEMA != entry.SCHEMA


def test_range_evidence_tamper_rejected_and_unrelated_append_not_reparsed(saved_call):
    root, plan, sid, sources, proof, store = saved_call
    with (store/'events.jsonl').open('ab') as stream: stream.write(b'{"later_unrelated":true}\n')
    assert entry.call_evidence(root, plan, 'original', 0, sid, sources, proof)['sample_id'] == sid
    payload = (store/'outputs.jsonl').read_bytes(); (store/'outputs.jsonl').write_bytes(payload.replace(b'LEFT', b'RIGH', 1))
    with pytest.raises(ValueError, match='bytes changed'): entry.call_evidence(root, plan, 'original', 0, sid, sources, proof)


def test_independent_same_gpu_comparison_preserves_different_vs_invalid(saved_call):
    root, plan, sid, sources, proof, _ = saved_call
    left = entry.call_evidence(root, plan, 'original', 0, sid, sources, proof); right = deepcopy(left)
    right['owner']['pid'] += 1; right['owner']['process_start_ticks'] += '-new'; right['runtime']['pid'] += 1; right['run_id'] += '-new'
    assert entry.compare_cold(left, right)['comparison_status'] == 'MATCH'
    right['fingerprint']['output_token_ids'] = [999999]
    assert entry.compare_cold(left, right)['comparison_status'] == 'DIFFERENT'
    right['fingerprint']['actual_input_tokens'] += 1
    with pytest.raises(ValueError, match='CPU/GPU input'): entry.compare_cold(left, right)


def test_same_owner_or_different_physical_gpu_cannot_pass_cold_comparison(saved_call):
    root, plan, sid, sources, proof, _ = saved_call
    left = entry.call_evidence(root, plan, 'original', 0, sid, sources, proof)
    with pytest.raises(ValueError, match='Independent cold'): entry.compare_cold(left, deepcopy(left))
    right = deepcopy(left); right['owner']['pid'] += 1; right['run_id'] += '-new'; right['runtime']['pid'] += 1
    right['runtime']['gpu_uuid'] = '94d2f97e-b0db-db79-5423-467860a66e2f'
    with pytest.raises(ValueError, match='fixed-GPU'): entry.compare_cold(left, right)
