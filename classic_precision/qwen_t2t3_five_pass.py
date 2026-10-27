"""Five independent native Qwen T2/T3 video passes; stop on the first difference."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import uuid

from . import qsa_only_reproducibility as qsa, early_stop_reproducibility as early, reproducibility as observed
from .storage import RunStore, PersistenceError, atomic_json, digest, file_sha256, owner_identity, _append

SCHEMA = 'classic_precision_qwen_t2t3_five_pass/v1'
COUNTS = {'original': 3504, 'supplement': 576}
MANIFESTS = {'original': 'e6df00283da944806dd4c4932d86df63cb80bd43bf990c6ba0f676b1bd4e6ed7',
             'supplement': '9e03ddc1ce2be9470d757717ab982fd45c43173f00ffd173c391bae40530a4b4'}
COMPAT_SHA = '83a3c55a3f404e9b96ad2d20cf0e45d74ced273c747094a7be11ddebb5104ab5'
EARLY_SHA = '8a6e3d0869f53d4474adc5934fcce823b925ffdbd8fa64240b3bd4911b3fcddf'
require = observed._require
_reader = None


def read(path): return json.loads(Path(path).read_text())


def descriptor(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': file_sha256(path), 'bytes': path.stat().st_size}


def verified(item):
    path = Path(item['path'])
    require(path.is_absolute() and path.is_file() and descriptor(path) == item, 'Source/manifest/receipt drift: ' + str(path))
    return path


def validate_admission(report, plans, profile, gpu, gpu_uuid):
    require(len(plans) == 2 and report.get('schema') == early.SCHEMA and report.get('comparison_status') == 'MATCH'
        and report.get('evidence_status') == 'COMPLETE' and report.get('expected_ids') == len(set(report.get('checked_ids', []))) == 49
        and {report.get('plan_sha256'), report.get('peer_plan_sha256')} == {digest(p) for p in plans}
        and all(p.get('schema') == early.SCHEMA and p.get('profile') == profile and set(p['sample_ids']) == set(report['checked_ids']) for p in plans)
        and any(p['gpu'] == gpu and early._normalized_gpu_uuid(p['gpu_uuid']) == early._normalized_gpu_uuid(gpu_uuid) for p in plans), 'Native49 x4 complete QSA admission differs')


def implementation():
    from . import inference, calibration, ready, manifest, storage, summary
    require(file_sha256(early.__file__) == EARLY_SHA and file_sha256(qsa.__file__) == early.FROZEN_QSA_ENTRY_SHA256
        and file_sha256(qsa.runtime.__file__) == early.FROZEN_QSA_RUNTIME_SHA256, 'Reviewed QSA/evidence sources changed')
    return {str(Path(p).resolve()): file_sha256(p) for p in (__file__, early.__file__, qsa.__file__, qsa.runtime.__file__,
        inference.__file__, calibration.__file__, ready.__file__, manifest.__file__, storage.__file__, summary.__file__)}


def evidence_reader():
    global _reader
    if _reader is None:
        require(file_sha256(early.__file__) == EARLY_SHA, 'Reviewed evidence reader changed')
        spec = importlib.util.spec_from_file_location('classic_precision._t2t3_five_pass_reader', early.__file__)
        _reader = importlib.util.module_from_spec(spec); spec.loader.exec_module(_reader)
        _reader.SCHEMA = SCHEMA  # private purpose tag; original reader/code/guards remain unchanged
        source = inspect.getsource(early._call_evidence)
        before = 'from .summary import technical_passed, model_snapshot'
        require(source.count(before) == 1, 'Evidence reader snapshot seam changed')
        source = source.replace(before, 'from .summary import technical_passed\n    model_snapshot = _five_pass_snapshot')
        local = {}; exec(compile(source, __file__+'::source_bound_evidence_reader', 'exec'), _reader.__dict__, local)
        _reader._call_evidence = local['_call_evidence']
    return _reader


def build_plan(config):
    require(set(config['panels']) == set(COUNTS), 'Exactly original3504 and supplement576 required')
    profile = qsa.build_profile('qwen_qsa_only', config['baseline_environment'])
    require(profile['torch_policy'] == {} and profile['baseline_environment'].get('CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB') in (None, '24'), 'Native Fourier24 QSA-only profile required')
    require(config['host'] == 'cirplab-fourier' and str(config['gpu']) in ('0', '1'), 'This finite wave is Fourier GPUs0/1 only')
    require(isinstance(config.get('wave_id'), str) and config['wave_id'] and Path(config['stop_path']).is_absolute(), 'Explicit finite wave/stop identity required')
    early._normalized_gpu_uuid(config['gpu_uuid'])
    report = read(config['admission']['report'])
    admission_plans = [read(p) for p in config['admission']['plans']]
    validate_admission(report, admission_plans, profile, str(config['gpu']), config['gpu_uuid'])
    plan = {'schema': SCHEMA, 'model': 'qwen', 'passes': 5, 'shard_count': 2, 'sharding_contract': 'native selected_rows: onset_index modulo2', 'max_model_calls': 10200, 'global_max_model_calls': 20400, 'automatic_retry': False,
        'host': config['host'], 'gpu': str(config['gpu']), 'gpu_uuid': config['gpu_uuid'], 'profile': profile,
        'wave_id': config['wave_id'], 'stop_path': config['stop_path'],
        'output_root': str(Path(config['stop_path']).resolve().parent), 'first_pass_barrier_timeout_seconds': 86400,
        'launch_environment': config.get('launch_environment', {}), 'model_root': str(Path(config['model_root']).resolve()),
        'gpu_lock_root': str(Path(config['gpu_lock_root']).resolve()), 'panels': {}, 'implementation': implementation(),
        'admission': {'report': descriptor(config['admission']['report']), 'plans': [descriptor(p) for p in config['admission']['plans']]},
        'comparison_contract': 'same host/physical GPU and actual input/runtime; independent cold owner/run_id; first valid full output/token/finish difference stops; changed controls INVALID_EVIDENCE; no retry',
        'scientific_claim': 'First complete pass is a new curve; full-panel five-pass repeatability requires all five complete MATCH passes.'}
    allowed = {'CLASSIC_PRECISION_KV_COMPAT', 'CLASSIC_PRECISION_NEMO_DUMMY_CONFIG_COMPAT', 'CLASSIC_PRECISION_DEFER_MODELS',
               'VLLM_ALLOW_INSECURE_SERIALIZATION', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'TOKENIZERS_PARALLELISM',
               'PATH', 'PYTHONPATH', 'PYTHONDONTWRITEBYTECODE', 'PYTHONUNBUFFERED'}
    require(set(plan['launch_environment']) <= allowed | set(profile['environment'])
        and all(isinstance(v, str) and (k not in profile['environment'] or v == profile['environment'][k]) for k, v in plan['launch_environment'].items())
        and not set(plan['launch_environment']) & set(profile['cleared_environment']), 'Undeclared launch intervention')
    for name, count in COUNTS.items():
        source = config['panels'][name]; manifest = read(source['manifest']); rows = manifest['rows']
        require(file_sha256(source['manifest']) == MANIFESTS[name] and len(rows) == len({r['sample_id'] for r in rows}) == count, 'Frozen scientific panel differs')
        from .inference import qualification_order
        from .manifest import selected_rows
        ordered, qualification = qualification_order(selected_rows(manifest, int(plan['gpu']), 2))
        require([r['task'] for r in qualification] == (['T2', 'T3'] if name == 'original' else ['T3']), 'Original qualification changed')
        item = {key: descriptor(source[key]) for key in ('manifest', 'ready')}
        item.update(asset_root=str(Path(source['asset_root']).resolve()), calibration_root=str(Path(source['calibration_root']).resolve()),
            calibration=descriptor(Path(source['calibration_root'])/'qwen/summary.json'), all_sample_ids=[r['sample_id'] for r in rows], sample_ids=[r['sample_id'] for r in ordered],
            qualification_ids=[r['sample_id'] for r in qualification], old_stores=[])
        require(source.get('old_outputs'), 'Explicit original large-scale output stores required')
        old_ids = set()
        from .summary import model_snapshot, technical_passed
        for old in source['old_outputs']:
            bundle = {key: descriptor(Path(old)/key) for key in ('identity.json', 'outputs.jsonl', 'events.jsonl')}
            identity, raw, events, rt, et = model_snapshot(old)
            require(identity.get('model_key') == 'qwen' and identity.get('manifest_sha256') == MANIFESTS[name]
                and identity.get('model') == manifest['models']['qwen'] and not rt and not et, 'Original baseline model/manifest/journal differs')
            seen = {r['sample_id'] for r in raw}
            require(len(seen) == len(raw) and not old_ids & seen and seen <= technical_passed(raw, events), 'Old baseline duplicate IDs or unresolved technical receipts')
            old_ids.update(seen); item['old_stores'].append(bundle)
            if 'checkpoint_assets' not in plan: plan['checkpoint_assets'] = identity['checkpoint_assets']
            require(identity['checkpoint_assets'] == plan['checkpoint_assets'], 'Historical pinned checkpoint identity differs')
        require(old_ids == {r['sample_id'] for r in rows}, 'Historical stores do not exactly cover the frozen full panel')
        if name == 'supplement':
            item['compat'] = descriptor(source['compat'])
            require(item['compat']['sha256'] == COMPAT_SHA, 'Original T3 compatibility source changed')
        plan['panels'][name] = item
    plan['wave_sha256'] = wave_digest(plan)
    return plan


def wave_digest(plan):
    common = deepcopy(plan)
    for key in ('gpu', 'gpu_uuid', 'wave_sha256'): common.pop(key, None)
    for item in common['panels'].values():
        item.pop('sample_ids'); item.pop('qualification_ids')
    return digest(common)


def load_plan(path):
    plan = read(path)
    require(plan.get('schema') == SCHEMA and plan.get('implementation') == implementation()
        and plan.get('passes') == 5 and plan.get('shard_count') == 2 and plan.get('sharding_contract') == 'native selected_rows: onset_index modulo2' and plan.get('max_model_calls') == 10200
        and plan.get('global_max_model_calls') == 20400 and plan.get('automatic_retry') is False, 'Five-pass plan/source identity differs')
    require(plan['profile'] == qsa.build_profile('qwen_qsa_only', plan['profile']['baseline_environment']), 'Only QSA selector intervention allowed')
    require(plan['host'] == 'cirplab-fourier' and plan['gpu'] in ('0', '1') and plan['wave_sha256'] == wave_digest(plan), 'Fixed Fourier two-arm wave required')
    require(plan['output_root'] == str(Path(plan['stop_path']).resolve().parent)
        and plan['first_pass_barrier_timeout_seconds'] == 86400, 'Sealed first-pass output/barrier differs')
    early._normalized_gpu_uuid(plan['gpu_uuid'])
    require(set(plan['panels']) == set(COUNTS), 'Both fixed panels required')
    for name, item in plan['panels'].items():
        for key in ('manifest', 'ready', 'calibration'): verified(item[key])
        rows = read(item['manifest']['path'])['rows']
        from .inference import qualification_order
        from .manifest import selected_rows
        ordered, qualification = qualification_order(selected_rows({'rows': rows}, int(plan['gpu']), 2))
        require(item['manifest']['sha256'] == MANIFESTS[name] and len(rows) == COUNTS[name]
            and item['all_sample_ids'] == [r['sample_id'] for r in rows] and len(set(item['all_sample_ids'])) == COUNTS[name]
            and item['sample_ids'] == [r['sample_id'] for r in ordered] and item['qualification_ids'] == [r['sample_id'] for r in qualification], 'Frozen two-way panel order/qualification differs')
        if name == 'supplement': require(verified(item['compat']) and item['compat']['sha256'] == COMPAT_SHA, 'T3 compatibility identity differs')
        for old in item['old_stores']:
            for value in old.values(): verified(value)
    verified(plan['admission']['report'])
    for value in plan['admission']['plans']: verified(value)
    validate_admission(read(plan['admission']['report']['path']), [read(p['path']) for p in plan['admission']['plans']], plan['profile'], plan['gpu'], plan['gpu_uuid'])
    return plan


def panel_view(plan, name):
    item = plan['panels'][name]
    return {**plan, 'arm': name, 'manifest': item['manifest']['path'], 'sample_ids': item['sample_ids']}


def directory(output, cold, name): return Path(output)/('cold-'+str(cold))/name


def shared_stop(plan):
    path = Path(plan['stop_path'])
    if not path.exists(): return None
    result = read(path); claim = read(result['claim_path'])
    require(result.get('schema') == SCHEMA and result.get('wave_sha256') == claim.get('wave_sha256') == plan['wave_sha256']
        and result.get('plan_sha256') == claim.get('plan_sha256') and result.get('gpu') in ('0', '1')
        and early._owner_key(result['owner']) == early._owner_key(claim['owner'])
        and result.get('comparison_status') in ('DIFFERENT', 'INVALID_EVIDENCE', 'STOPPED'), 'Shared stop identity differs')
    if result['comparison_status'] == 'DIFFERENT':
        require(result.get('evidence_status') == 'COMPLETE' and result.get('first_difference', {}).get('comparison_status') == 'DIFFERENT', 'Shared counterexample incomplete')
    return result


def peer_first_pass_ready(plan):
    peer_gpu = str(1-int(plan['gpu'])); output = Path(plan['output_root'])/('gpu'+peer_gpu)
    for name in COUNTS:
        path = directory(output, 0, name)/'result.json'
        if not path.exists(): return False
        value = read(path); claim = read(path.with_name('claim.json'))
        require(value.get('schema') == SCHEMA and value.get('wave_sha256') == claim.get('wave_sha256') == plan['wave_sha256']
            and value.get('plan_sha256') == claim.get('plan_sha256') and value.get('gpu') == peer_gpu
            and value.get('panel') == name and value.get('cold') == 0
            and early._owner_key(value['owner']) == early._owner_key(claim['owner']) and value['run_id'] == claim['run_id'], 'Peer first-pass identity differs')
        require(value.get('status') == value.get('evidence_status') == 'COMPLETE'
            and value.get('comparison_status') == 'REFERENCE_COMPLETE' and value.get('model_calls') == COUNTS[name]//2, 'Peer first pass did not complete')
        from .manifest import selected_rows
        from .inference import qualification_order
        manifest = read(plan['panels'][name]['manifest']['path'])
        ordered, _ = qualification_order(selected_rows(manifest, int(peer_gpu), 2))
        with path.with_name('calls.jsonl').open() as stream: ids = [json.loads(line)['sample_id'] for line in stream]
        require(ids == [r['sample_id'] for r in ordered], 'Peer first-pass proof denominator/order differs')
    return True


def compare_cold(left, right):
    require(left['sample_id'] == right['sample_id'] and left['native_identity'] == right['native_identity'], 'Native scientific identity differs')
    a, b = deepcopy(left['runtime']), deepcopy(right['runtime'])
    require(early._owner_key(left['owner']) != early._owner_key(right['owner'])
        and left['owner']['host'] == right['owner']['host'] and left['owner']['boot_id'] == right['owner']['boot_id']
        and left['run_id'] != right['run_id'], 'Independent cold process/run identity missing')
    a.pop('pid'); b.pop('pid')
    require(a == b, 'Actual fixed-GPU runtime differs')
    controls = lambda c: {k: v for k, v in c['fingerprint'].items() if k not in early._OUTPUT_FIELDS}
    require(controls(left) == controls(right), 'Actual CPU/GPU input, prompt or generation differs')
    changed = [k for k in sorted(early._OUTPUT_FIELDS) if left['fingerprint'][k] != right['fingerprint'][k]]
    return {'comparison_status': 'DIFFERENT' if changed else 'MATCH', 'evidence_status': 'COMPLETE', 'sample_id': left['sample_id'],
        'different_fields': changed, 'left_raw_sha256': left['raw_sha256'], 'right_raw_sha256': right['raw_sha256']}


def _range(path, start, end):
    require(type(start) is type(end) is int and 0 <= start <= end <= Path(path).stat().st_size, 'Durable journal range invalid')
    with Path(path).open('rb') as stream: stream.seek(start); value = stream.read(end-start)
    require(not value or value.endswith(b'\n'), 'Incomplete durable journal range')
    return value


def proof_for(root, sid, prefix_end, raw_start, event_start):
    store = Path(root)/'repeat-0/qwen'; raw_end = (store/'outputs.jsonl').stat().st_size; event_end = (store/'events.jsonl').stat().st_size
    prefix = _range(store/'events.jsonl', 0, prefix_end)
    raw = _range(store/'outputs.jsonl', raw_start, raw_end); events = _range(store/'events.jsonl', event_start, event_end)
    return {'sample_id': sid, 'identity_sha256': file_sha256(store/'identity.json'), 'prefix_end': prefix_end,
        'raw_start': raw_start, 'raw_end': raw_end, 'event_start': event_start, 'event_end': event_end,
        'prefix_sha256': hashlib.sha256(prefix).hexdigest(), 'raw_sha256': hashlib.sha256(raw).hexdigest(), 'events_sha256': hashlib.sha256(events).hexdigest()}


def call_evidence(output, plan, name, cold, sid, sources, proof):
    root = directory(output, cold, name); claim = read(root/'claim.json')
    require(claim['plan_sha256'] == digest(plan) and claim['panel'] == name and claim['cold'] == cold, 'Cold claim differs')
    store = root/'repeat-0/qwen'
    require(proof['sample_id'] == sid and file_sha256(store/'identity.json') == proof['identity_sha256'], 'Call proof/identity differs')
    prefix = _range(store/'events.jsonl', 0, proof['prefix_end'])
    raw = _range(store/'outputs.jsonl', proof['raw_start'], proof['raw_end'])
    events = _range(store/'events.jsonl', proof['event_start'], proof['event_end'])
    require(all(hashlib.sha256(value).hexdigest() == proof[key] for value, key in
        ((prefix, 'prefix_sha256'), (raw, 'raw_sha256'), (events, 'events_sha256'))), 'Durable call evidence bytes changed')
    parse = lambda payload: [json.loads(line) for line in payload.splitlines() if line]
    def snapshot(requested):
        require(Path(requested).resolve() == store.resolve(), 'Evidence reader requested another journal')
        return read(store/'identity.json'), parse(raw), parse(prefix)+parse(events), False, False
    reader = evidence_reader(); reader._five_pass_snapshot = snapshot
    try: value = reader._call_evidence(store, panel_view(plan, name), digest(plan), 0, sid, sources)
    finally: reader._five_pass_snapshot = None
    require(early._owner_key(claim['owner']) == early._owner_key(value['owner']), 'Claim/native owner differs')
    value['run_id'] = claim['run_id']
    return value


def run(plan_path, output, name, cold):
    from . import inference, calibration
    require(name in COUNTS and cold in range(5) and 'torch' not in sys.modules and 'vllm' not in sys.modules, 'Fresh native Qwen process/panel/cold required')
    plan = load_plan(plan_path); plan_hash = digest(plan); plan_file_hash = file_sha256(plan_path); item = plan['panels'][name]
    require(socket.gethostname() == plan['host'], 'Actual host differs')
    require(all(os.environ.get(k) == v for k, v in {**plan['profile']['environment'], **plan['launch_environment']}.items())
        and all(k not in os.environ for k in plan['profile']['cleared_environment']), 'Native launch/profile environment differs')
    output = Path(output).resolve()
    require(output == Path(plan['output_root'])/('gpu'+plan['gpu']), 'Output must use the sealed arm directory')
    for panel in plan['panels'].values():
        source = Path(panel['asset_root']).resolve()
        require(not output.is_relative_to(source) and not source.is_relative_to(output), 'Protect original source/raw tree')
    require(shared_stop(plan) is None, 'Earlier finite job stopped')
    previous_owners, previous_runs = [], []
    for prior in range(cold):
        receipt = read(directory(output, prior, name)/'result.json')
        require(receipt.get('status') == 'COMPLETE' and receipt.get('comparison_status') in ('REFERENCE_COMPLETE', 'MATCH')
            and receipt.get('plan_sha256') == plan_hash and receipt.get('model_calls') == len(item['sample_ids']), 'Earlier complete cold panel required')
        previous_owners.append(early._owner_key(receipt['owner'])); previous_runs.append(receipt['run_id'])
    os.environ['CUDA_VISIBLE_DEVICES'] = plan['gpu']
    if name == 'supplement':
        os.environ['CLASSIC_PRECISION_T3_ONLY_GENERATION_COMPAT'] = '1'
        spec = importlib.util.spec_from_file_location('_five_pass_original_t3_compat', verified(item['compat']))
        compat = importlib.util.module_from_spec(spec); spec.loader.exec_module(compat)
        require(compat.install(), 'Original T3 compatibility inactive')
    else: require(os.environ.get('CLASSIC_PRECISION_T3_ONLY_GENERATION_COMPAT') is None, 'No T3 compat in original panel')
    manifest, rows, ready, assets, ready_path = inference.load_run_inputs(item['manifest']['path'], item['ready']['path'], item['asset_root'])
    lookup = {r['sample_id']: r for r in rows}; sources = {sid: (lookup[sid], ready[sid]) for sid in item['sample_ids']}
    require(set(ready) >= set(lookup), 'Full original ready catalog required')
    expected = calibration.calibration_identity(item['manifest']['path'], manifest, 'qwen', plan['model_root'], None)
    require(calibration.calibration_ready(item['calibration_root'], 'qwen', expected, manifest, ready), 'Original panel-specific CPU calibration incomplete')
    native, model_path = inference.build_run_identity(item['manifest']['path'], ready_path, manifest, 'qwen', plan['model_root'])
    require(native['checkpoint_assets'] == plan['checkpoint_assets'], 'Sealed checkpoint/code/weight-shard identity differs')
    qsa.runtime.activate('qwen', 'qwen_qsa_only', plan['profile']['baseline_environment'])
    qsa._validate_profile(qsa.runtime.status(), 'qwen', require_applied=False)
    import torch
    root = directory(output, cold, name); root.mkdir(parents=True, exist_ok=True)
    owner, run_id = owner_identity(), uuid.uuid4().hex
    early._owner_key(owner)
    require(early._owner_key(owner) not in previous_owners and run_id not in previous_runs, 'Cold process/run was reused')
    early._publish_claim(root/'claim.json', {'plan_sha256': plan_hash, 'wave_sha256': plan['wave_sha256'], 'panel': name, 'cold': cold, 'owner': owner, 'run_id': run_id})
    stop = early._Stop(); handlers = {s: signal.signal(s, stop.request) for s in (signal.SIGINT, signal.SIGTERM)}
    began, calls = time.monotonic(), 0
    def finish(status, **extra):
        result = {'schema': SCHEMA, 'status': 'COMPLETE' if status in ('REFERENCE_COMPLETE', 'MATCH', 'DIFFERENT') else status,
            'comparison_status': status, 'evidence_status': 'COMPLETE' if status in ('REFERENCE_COMPLETE', 'MATCH', 'DIFFERENT') else 'INVALID_OR_INCOMPLETE',
            'plan_sha256': plan_hash, 'panel': name, 'cold': cold, 'owner': owner, 'run_id': run_id, 'model_calls': calls,
            'wave_sha256': plan['wave_sha256'], 'gpu': plan['gpu'], 'claim_path': str(root/'claim.json'),
            'expected_calls': len(item['sample_ids']), 'elapsed_seconds': time.monotonic()-began, **extra}
        atomic_json(root/'result.json', result)
        if status in ('DIFFERENT', 'INVALID_EVIDENCE', 'STOPPED'):
            try: early._publish_claim(plan['stop_path'], result)
            except FileExistsError: pass
        return result
    try:
        with inference._gpu_lock(Path(plan['gpu_lock_root']), torch), ExitStack() as cleanup:
            require(early._normalized_gpu_uuid(str(torch.cuda.get_device_properties(0).uuid)) == early._normalized_gpu_uuid(plan['gpu_uuid']), 'Physical GPU differs')
            resources = manifest['models']['qwen']['resources']; admission = inference.resource_snapshot(torch, resources)
            require(admission['admitted'], 'Original resource admission failed')
            identity = {**native, 'purpose': SCHEMA, 'reproducibility': {'plan_sha256': plan_hash, 'profile': plan['profile'],
                'arm': name, 'repeat': 0, 'sample_ids': item['sample_ids'], 'implementation': plan['implementation']}}
            store = cleanup.enter_context(RunStore(root/'repeat-0', identity, 'qwen', 'five-pass', item['sample_ids']))
            require(not store.outputs and not any(e.get('event') == 'INFLIGHT' for e in store.events), 'No retry/resume')
            store.event('REPRO_RESOURCE_ADMISSION', gpu_selection=plan['gpu'], **admission)
            sink = observed._EventSink(store)
            session = inference.ModelSession('qwen', model_path, sink, None, resources); cleanup.callback(session.close)
            qsa.runtime.prepare_native('qwen'); snapshot = qsa.runtime.runtime_snapshot()
            evidence_reader()._snapshot_identity(snapshot, panel_view(plan, name)); store.event('REPRO_RUNTIME', snapshot=snapshot)
            prefix_end = (store.directory/'events.jsonl').stat().st_size
            reference_stream = cleanup.enter_context((directory(output, 0, name)/'calls.jsonl').open()) if cold else None
            qualification = [lookup[sid] for sid in item['qualification_ids']]
            qualified = False
            for sid in item['sample_ids']:
                if stop.reason or shared_stop(plan): return finish('STOPPED', reason=stop.reason or 'earlier finite job stopped')
                require(file_sha256(plan_path) == plan_file_hash and implementation() == plan['implementation'], 'Sealed plan/source changed')
                require(qualified or sid in item['qualification_ids'], 'Native qualification must pass before further requests')
                raw_path = store.directory/'outputs.jsonl'; raw_start = raw_path.stat().st_size if raw_path.exists() else 0
                event_start = (store.directory/'events.jsonl').stat().st_size
                store.begin(sid, digest({'row': lookup[sid], 'ready': ready[sid]})); started = time.monotonic(); calls += 1
                session.context['deep'] = not qualified
                try:
                    def durable(raw):
                        return store.commit_output(sid, {**raw, 'parent_sample_id': sid,
                            'reproducibility': {'plan_sha256': plan_hash, 'arm': name, 'repeat': 0}})
                    session.generate(lookup[sid], ready[sid], assets, durable)
                    store.event('EARLY_CALL_COMPLETE', sample_id=sid, attempt_id=store.outputs[sid]['attempt_id'], elapsed_seconds=time.monotonic()-started)
                    proof = proof_for(root, sid, prefix_end, raw_start, event_start)
                    current = call_evidence(output, plan, name, cold, sid, sources, proof)
                    _append(root/'calls.jsonl', proof)
                except BaseException as error:
                    if sid in store.outputs: store.event('POST_OUTPUT_VALIDATION_FAILED', sample_id=sid, attempt_id=store.outputs[sid]['attempt_id'], error_type=type(error).__name__, error=str(error))
                    elif not isinstance(error, PersistenceError): store.fail(sid, error, interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)), elapsed_seconds=time.monotonic()-started)
                    raise
                finally: session.context.update(active=False, codes=None)
                if not qualified: qualified = inference._qualification_marker(store, native, 'qwen', 'five-pass', qualification)
                if cold:
                    reference_proof = json.loads(reference_stream.readline())
                    reference = call_evidence(output, plan, name, 0, sid, sources, reference_proof)
                    receipt = compare_cold(reference, current)
                    if receipt['comparison_status'] == 'DIFFERENT': return finish('DIFFERENT', first_difference=receipt)
                if shared_stop(plan): return finish('STOPPED', reason='peer stopped after current durable call')
                if stop.reason: return finish('STOPPED', reason=stop.reason)
            store.event('FIVE_PASS_PANEL_COMPLETE', sample_ids=item['sample_ids'])
            if reference_stream: require(not reference_stream.readline(), 'Reference has extra calls')
            return finish('REFERENCE_COMPLETE' if cold == 0 else 'MATCH')
    except BaseException as error:
        return finish('INVALID_EVIDENCE', error_type=type(error).__name__, error=str(error))
    finally:
        for number, handler in handlers.items(): signal.signal(number, handler)


def compare_old(plan_path, output, name, cold):
    from .summary import model_snapshot, technical_passed, cell_key
    plan = load_plan(plan_path); item = plan['panels'][name]; ids = set(item['sample_ids']); all_ids = set(item['all_sample_ids'])
    done = read(directory(output, cold, name)/'result.json')
    require(done['plan_sha256'] == digest(plan), 'New-run result identity differs')
    _, new, events, rt, et = model_snapshot(directory(output, cold, name)/'repeat-0/qwen')
    require(not rt and not et and len(new) == len({r['sample_id'] for r in new}) and {r['sample_id'] for r in new} <= ids, 'New raw duplicate/extra/tail')
    passed = technical_passed(new, events); old = {}
    for bundle in item['old_stores']:
        identity, rows, history, a, b = model_snapshot(Path(bundle['identity.json']['path']).parent)
        require(identity.get('model_key') == 'qwen' and identity.get('manifest_sha256') == item['manifest']['sha256'] and not a and not b, 'Old native model/manifest/journal identity differs')
        old_passed = technical_passed(rows, history)
        for raw in rows:
            sid = raw['sample_id']; require(sid in all_ids and sid not in old and sid in old_passed, 'Old raw missing technical PASS, duplicated or extra')
            old[sid] = raw
    require(set(old) == all_ids, 'Original large-scale reference denominator incomplete')
    lookup = {r['sample_id']: r for r in read(item['manifest']['path'])['rows']}
    details, cells = [], defaultdict(lambda: {'n': 0, 'old_correct': 0, 'new_correct': 0})
    for raw in new:
        sid = raw['sample_id']; prior = old[sid]; row = lookup[sid]
        require(sid in passed and raw['truth'] == prior['truth'] == row['truth'] and raw['source_parameters'] == prior['source_parameters'] == row['source_parameters'], 'Old/new scientific identity or technical PASS differs')
        changed = [k for k in sorted(early._OUTPUT_FIELDS) if raw.get(k) != prior.get(k)]
        old_normal = prior.get('finish_reason') == 'stop' and prior.get('raw_answer') in ('LEFT', 'RIGHT')
        new_normal = raw.get('finish_reason') == 'stop' and raw.get('raw_answer') in ('LEFT', 'RIGHT')
        before, after = old_normal and prior['raw_answer'] == row['truth'], new_normal and raw['raw_answer'] == row['truth']
        details.append({'sample_id': sid, 'task': row['task'], 'different_fields': changed, 'old_correct': before, 'new_correct': after,
            'old_raw_sha256': digest(prior), 'new_raw_sha256': digest(raw), 'old_answer': prior['raw_answer'], 'new_answer': raw['raw_answer'],
            'old_normal': old_normal, 'new_normal': new_normal, 'historical_ready_record_equal': prior.get('ready_record') == raw.get('ready_record'),
            'effective_generation_equal': prior.get('effective_generation') == raw.get('effective_generation')})
        key = (row['task'], *cell_key(row)[1:]); cell = cells[key]; cell['n'] += 1; cell['old_correct'] += before; cell['new_correct'] += after
    return {'schema': SCHEMA, 'plan_sha256': digest(plan), 'wave_sha256': plan['wave_sha256'], 'gpu': plan['gpu'], 'panel': name, 'cold': cold, 'expected': len(ids), 'full_panel_expected': COUNTS[name], 'observed': len(new),
        'status': 'COMPLETE' if len(new) == len(ids) and done['comparison_status'] in ('REFERENCE_COMPLETE', 'MATCH') else 'PARTIAL',
        'repeatability_status': done['comparison_status'], 'all_five_passes_proven': False,
        'old_new_comparison_scope': 'same scientific IDs; historical runtime/input hashes are not asserted equal to the new controlled rerun',
        'changed_outputs': sum(bool(r['different_fields']) for r in details), 'wrong_to_right': sum(not r['old_correct'] and r['new_correct'] for r in details),
        'right_to_wrong': sum(r['old_correct'] and not r['new_correct'] for r in details), 'by_task': dict(Counter(r['task'] for r in details)),
        'historical_ready_record_mismatches': sum(not r['historical_ready_record_equal'] for r in details),
        'historical_generation_mismatches': sum(not r['effective_generation_equal'] for r in details),
        'cells': [{'cell': list(k), **v} for k, v in sorted(cells.items())], 'records': details}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare'); prep.add_argument('--config', required=True); prep.add_argument('--plan', required=True)
    for command in ('run', 'compare-old'):
        p = sub.add_parser(command)
        for key in ('plan', 'output', 'panel', 'cold'): p.add_argument('--'+key, required=True, type=int if key == 'cold' else str)
        if command == 'compare-old': p.add_argument('--report', required=True)
    args = parser.parse_args(argv)
    if args.command == 'prepare': early._publish_claim(args.plan, build_plan(read(args.config))); return 0
    if args.command == 'run': result = run(args.plan, args.output, args.panel, args.cold)
    else:
        result = compare_old(args.plan, args.output, args.panel, args.cold); early._publish_claim(args.report, result)
    print(json.dumps(result, sort_keys=True))
    return int(result.get('comparison_status') in ('INVALID_EVIDENCE', 'STOPPED'))


if __name__ == '__main__': raise SystemExit(main())
