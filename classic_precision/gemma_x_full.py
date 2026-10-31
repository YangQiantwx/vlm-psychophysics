"""Finite Gemma X classic9456 preparation and request-boundary lifecycle.

Preparation is CPU-only. Sealing, activation and execution require the explicit
three-card/two-cold qualification and physical SSD admission. Original claims,
wait receipts and physical-lease inodes are retained across clean pauses.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile

from . import nemo_x_full as lifecycle
from . import official_x_runner as base
from .official_x_results import Archive as BaseArchive
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity, _append

NAME = 'gemma-x-full-20261030'
MODULE = 'classic_precision.gemma_x_full'
CARDS = ('fourier0', 'fourier1', 'radon0')
COUNTS = {'T1': 3840, 'T2': 2880, 'T3': 2736}
QUALIFICATION_SCHEMA = 'gemma-X-native-compact-qualification-v1'
QUALIFICATION_PASS = 'PASS_GEMMA_X_THREE_CARDS_TWO_COLD'
ROOTS = ('/home/yang/cvpr2027_benchmark', '/mnt/sda/q/cvpr2027_benchmark')
MAC_ROOT = '/Users/qianyang/Desktop/cvpr2027_benchmark'
DEPENDENCIES = ('official_x_runner', 'official_x_gemma_adapter', 'gemma_x_native_input',
                'gemma_x_native_compact', 'gemma_x_full_input', 'gemma_x_speed',
                'nemo_x_full', 'refinement_worker', 'storage', 'inference',
                'gemma_speed', 'gemma_x_full_results')
OPERATIONAL = {
    'schema', 'worker', 'worker_token', 'phase', 'cold', 'panel', 'assigned_ids',
    'maximum_model_calls', 'source_catalog', 'sources', 'output', 'claim_root',
    'pause_file', 'control_file', 'control_generation', 'status', 'implementation',
    'media_files', 'operational_wrapper', 'full_dependencies', 'qualified_profile',
    'qualification_plan', 'qualification_summary', 'experiment_plan',
    'execution_authorization', 'canonical_catalog', 'continuation_parent',
    'continuation_wait', 'variant', 'qualification_gate', 'prepared_plan_sha256',
}
ORIGINAL_CLOSED = lifecycle.closed
ORIGINAL_IDENTITY = base.worker_identity


class Archive(BaseArchive):
    """Map mirrored immutable descriptors onto the current execution/archive root."""
    def path(self, path):
        value = str(path)
        if value.startswith(MAC_ROOT + '/'):
            return self.root / value[len(MAC_ROOT) + 1:]
        return super().path(path)


def validate_runtime(value):
    require(value['model_class'] == 'vllm.model_executor.models.gemma4_mm.Gemma4ForConditionalGeneration'
            and value['vllm_version'] == '0.33.0rc1.dev175+gf4fd1ab68'
            and value['torch_version'].split('+')[0] == '2.14.0' and value['cuda_version'] == '13.0',
            'Different qualified Gemma implementation/runtime')
    counts = value['method_kernel_counts']
    require(counts.get('kernel:vllm.model_executor.kernels.linear.nvfp4.flashinfer.FlashInferCutlassNvFp4LinearKernel') == 120
            and counts.get('method:vllm.model_executor.layers.quantization.modelopt.ModelOptLinearMethod') == 120
            and counts.get('method:vllm.model_executor.layers.quantization.modelopt.ModelOptKVCacheMethod') == 60
            and value['actual_backends'] == {} and value['mm_device_do_normalize'] is False,
            'Actual Gemma official mixed NVFP4 kernels/input path changed')
    return value


def worker_identity(worker, model):
    require(model == 'gemma_x_full', 'Wrong Gemma full worker model identity')
    return validate_runtime(ORIGINAL_IDENTITY(worker, model))


def frozen_settings(profile):
    return {k: v for k, v in profile.items() if k not in OPERATIONAL}


def dependency_sources():
    # Read bytes without importing the adapter or any GPU/runtime dependency.
    directory = Path(__file__).parent
    return {name: descriptor(directory / (name + '.py')) for name in DEPENDENCIES}


def verify_dependencies(profile):
    actual = dependency_sources()
    require(set(profile['full_dependencies']) == set(actual), 'Incomplete frozen dependency set')
    for name, value in actual.items():
        require(profile['full_dependencies'][name]['sha256'] == value['sha256'],
                'Frozen Gemma dependency changed: ' + name)
    require(profile['operational_wrapper']['sha256'] == file_sha256(__file__),
            'Frozen full lifecycle changed')


def qualification_gate(plan_ref, summary_ref, resolve=verified):
    plan = resolve(plan_ref); summary = resolve(summary_ref)
    require(plan['schema'] == QUALIFICATION_SCHEMA and set(plan['cards']) == set(CARDS),
            'Wrong Gemma qualification plan')
    require(summary['status'] == QUALIFICATION_PASS
            and summary['plan_sha256'] == plan_ref['sha256']
            and summary['unique_video_ids'] == 96 and summary['executed_model_calls'] == 576
            and summary['tasks'] == {'T3': 96}
            and summary['closed_workers'] == 6
            and summary['full_output_differences'] == [] and summary['prompt_token_differences'] == []
            and summary['input_admission_status'] == 'PASS_ALL_CARDS_TASKS',
            'Gemma three-card/two-cold qualification is not PASS')
    selection = resolve(plan['selection'])
    require(selection['task'] == 'T3' and selection['angle_deg'] == 1.5
            and selection['n'] == 96 and selection['correct'] == 67
            and selection['accuracy'] == 67 / 96 and len(set(selection['assigned_ids'])) == 96
            and selection['selection_uses_X_answers'] is False and selection['source_identity_verified'] is True
            and summary['selection_sha256'] == plan['selection']['sha256'],
            'Qualification does not bind the complete preselected historical A condition')
    workers = plan['workers']
    require(len(workers) == 6 and Counter((x['card'], x['cold']) for x in workers)
            == {(c, k): 1 for c in CARDS for k in (1, 2)}, 'Missing or duplicate cold worker')
    for card in CARDS:
        second = next(x['profile'] for x in workers if x['card'] == card and x['cold'] == 2)
        require(plan['cards'][card]['qualified_profile'] == second,
                'Formal template is not this card second cold profile')
    for worker in workers:
        profile = resolve(worker['profile'])
        require(profile['card'] == worker['card'] and profile['cold'] == worker['cold']
                and profile['assigned_ids'] == selection['assigned_ids'], 'Qualification worker condition/order changed')
    return plan, summary


def rebase(value, remote):
    if isinstance(value, str):
        for old in ROOTS:
            if value.startswith(old + '/'):
                return remote + value[len(old):]
        return value
    if isinstance(value, list): return [rebase(x, remote) for x in value]
    if isinstance(value, dict): return {k: rebase(v, remote) for k, v in value.items()}
    return value


def remote_ref(path, root, remote):
    return {'path': str(Path(remote) / Path(path).relative_to(root)), 'sha256': file_sha256(path)}


def source_check(item, original):
    """Accepted source/GT/Slack prompt identity, independent of model answers."""
    def semantic(value):
        value = copy.deepcopy(value)
        for part in value['content']:
            if part['type'] in ('video', 'image'): part.pop('path', None)
        return value
    require(semantic(item) == semantic(original), 'Canonical classic source changed')
    row = item['row']; public = item['public_index']; params = row['source_parameters']
    require(item['task'] == row['task'] == public['task'] and row['accuracy_eligible']
            and row['choices'] == ['LEFT', 'RIGHT'], 'Wrong task/denominator/choices')
    require(item['video_contract'] == {'fps': 60, 'frames': 180, 'height': 512, 'width': 768},
            'Classic source geometry differs')
    require(all(item[k] == public[k] for k in
                ('source_row_sha256', 'source_ready_sha256', 'source_audit_identity'))
            and item['ready']['video_sha256'] == public['video_sha256'] == item['content'][0]['sha256']
            and item['ready']['stream_sha256'] == public['stream_sha256'] == item['source_stream_sha256']
            and params == public['nominal_parameters'] and row['truth'] == public['truth']
            and row['prompt'] == public['prompt']
            and item['content'][1] == {'type': 'text', 'text': public['prompt']},
            'Accepted source/GT/prompt bindings differ')
    if item['task'] == 'T1': require(row['truth'] == params['direction'], 'T1 direction differs')
    elif item['task'] == 'T2':
        require(params['offset_frame'] - params['onset_frame'] == 30
                and row['truth'] == params['target_side'], 'T2 must be 500ms with accepted GT')
    elif item['task'] == 'T3':
        require(params['angle_deg'] not in (1.75, 2.25) and row['truth'] == params['reference_side'],
                'Excluded T3 condition or wrong GT')
    else: raise ValueError('Only classic T1/T2/T3 are authorized')


def science(profile, resolve=None, require_gate=True):
    p = profile; live = resolve is None
    if live: resolve = Archive(p['workspace_root']).ref
    template = resolve(p['qualified_profile'])
    require(frozen_settings(p) == frozen_settings(template), 'Qualified Gemma settings changed')
    require(p['model'] == 'gemma_x_full' and p['base_model'] == 'gemma'
            and p['phase'] == 'full' and not p.get('bootstrap')
            and p['automatic_retry'] is False and p['formal_auto_resume'] is False,
            'Wrong Gemma full phase or automatic retry')
    require(all(p['expected_packages'].get(k) == v for k, v in {
        'vllm': '0.33.0rc1.dev175+gf4fd1ab68', 'torch': '2.14.0',
        'transformers': '5.17.0', 'flashinfer-python': '0.7.2.post1'}.items()),
        'Pinned Gemma SDK packages changed')
    require(p['request_adapter']['module'] == 'classic_precision.gemma_x_full_input'
            and p['request_adapter'].get('factory', p['request_adapter'].get('function')) == 'create'
            and p['vision_validation_hook']['module'] == 'classic_precision.gemma_x_full_input'
            and p['vision_validation_hook']['function'] == 'validate_vision', 'Wrong qualified Gemma input bridge')
    plan = resolve(p['experiment_plan']); allocation = plan['cards'][p['card']]
    require(plan['schema'] == NAME + '/execution-v1' and plan['maximum_full_calls'] == 9456
            and plan['task_counts'] == COUNTS and set(plan['cards']) == set(CARDS), 'Wrong finite classic scope')
    ids = p['assigned_ids']; full = allocation['assigned_ids']
    require(len(ids) == len(set(ids)) == p['maximum_model_calls'] and set(ids) <= set(full),
            'Duplicate or foreign full input')
    require(p['source_catalog'] == allocation['source_catalog']
            and p['canonical_catalog']['sha256'] == plan['canonical_catalog']['sha256']
            and p['qualification_plan']['sha256'] == plan['qualification_plan']['sha256'],
            'Full catalog is not bound to plan')
    root = Path(p['workspace_root'])
    require(Path(p['output']).parent == root / 'runs/reproducibility' / NAME
            and Path(p['claim_root']) == root / 'runs/reproducibility' / NAME / 'claims' / p['card']
            and Path(p['control_file']) == root / 'plans' / NAME / (p['card'] + '-control.json'),
            'Full namespace differs')
    if p.get('continuation_parent'):
        parent = resolve(p['continuation_parent'])
        require(p['control_generation'] == parent['control_generation'] + 1
                and frozen_settings(parent) == frozen_settings(p), 'Continuation settings/generation changed')
        if require_gate and live:
            _, previous, raw = closed(p['continuation_parent']['path'])
            require(previous['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD')
                    and ids == [s for s in parent['assigned_ids'] if s not in raw],
                    'Continuation is not the original unattempted suffix')
    else:
        require(ids == full and p['control_generation'] == 0, 'Initial allocation changed')
    canonical = resolve(p['canonical_catalog'])['items']; catalog = resolve(p['source_catalog'])['items']
    for sid in ids:
        require(catalog[sid] == rebase(canonical[sid], p['workspace_root']), 'Source differs from accepted9456')
        source_check(catalog[sid], canonical[sid])
    if require_gate:
        qplan, _ = qualification_gate(p['qualification_plan'], p['qualification_summary'], resolve)
        require(qplan['cards'][p['card']]['qualified_profile'] == p['qualified_profile'],
                'Formal template differs from qualified card')
        auth = resolve(p['execution_authorization'])
        require(auth['status'] == 'AUTHORIZED' and auth['maximum_full_calls'] == 9456
                and set(auth['cards']) == set(CARDS)
                and auth['qualification_selection']['sha256'] == qplan['selection']['sha256']
                and isinstance(auth.get('user_message'), str) and auth['user_message'].strip(),
                'Missing bound finite authorization')


def block_devices(device, sys_root='/sys/dev/block', seen=None):
    """Resolve Linux partition/mapper leaves; reject RAM/network/unknown storage."""
    seen = set() if seen is None else seen
    key = str(device); require(key not in seen, 'Cyclic physical block topology'); seen.add(key)
    node = (Path(sys_root) / key).resolve(strict=True)
    slaves = list((node / 'slaves').iterdir()) if (node / 'slaves').exists() else []
    if slaves:
        return [v for slave in slaves for v in block_devices((slave.resolve() / 'dev').read_text().strip(), sys_root, seen.copy())]
    while not (node / 'queue/rotational').exists():
        require(node.parent != node, 'No physical block queue'); node = node.parent
    require((node / 'queue/rotational').read_text().strip() == '0', 'Rotating disk is forbidden for current inference')
    require(not node.name.startswith(('loop', 'ram', 'zram', 'nbd')), 'Nonphysical storage is forbidden')
    return [{'physical_device': node.name, 'sysfs': str(node), 'rotational': 0}]


def storage_admission(profile):
    """CPU-only live physical gate, covering weight symlinks, every source, output."""
    require(sys.platform.startswith('linux'), 'Physical storage admission must run on the Linux execution host')
    driver = Path('/proc/driver/nvidia/version').read_text()
    require(re.search(r'\b595\.99\.02\b', driver), 'Qualified NVIDIA driver changed')
    p = profile; catalog = verified(p['source_catalog'])['items']; paths = []
    for name in p['checkpoint_assets']['weight_shard_bytes']:
        paths.append(('weight', Path(p['checkpoint']) / name))
    require(paths, 'No qualified checkpoint shards')
    for sid in p['assigned_ids']:
        paths.extend(('source', Path(x['path'])) for x in catalog[sid]['content'] if x['type'] in ('video', 'image'))
    target = Path(p['output']).resolve(strict=False); ancestor = target
    while not ancestor.exists(): ancestor = ancestor.parent
    paths.append(('run', ancestor))
    devices = {}; roots = {}; counts = Counter()
    for kind, path in paths:
        real = path.resolve(strict=True); st = real.stat(); key = str(os.major(st.st_dev)) + ':' + str(os.minor(st.st_dev))
        if key not in devices: devices[key] = block_devices(key)
        counts[kind] += 1; roots.setdefault(kind, set()).add((str(real.parent if real.is_file() else real), key))
    return {'status': 'PASS_PHYSICAL_SSD_WEIGHTS_SOURCES_RUN', 'profile_sha256': digest(p),
            'checkpoint_resolved': str(Path(p['checkpoint']).resolve(strict=True)), 'output_resolved': str(target),
            'counts': dict(counts), 'devices': devices,
            'driver_version': '595.99.02',
            'resolved_directories': {k: [list(x) for x in sorted(v)] for k, v in roots.items()},
            'source_bytes_rehashed': False, 'model_calls': 0}


class Boundary(lifecycle.Boundary):
    pass


def seal(draft, output):
    p = read(draft); science(p); storage_admission(p)
    p['operational_wrapper'] = descriptor(__file__); p['full_dependencies'] = dependency_sources()
    # Parallel immutable-byte admission, then reuse the canonical sealer. Its
    # per-file SHA checks consume only these stat-bound verified digests.
    catalog = verified(p['source_catalog'])['items']; files = {}
    for sid in p['assigned_ids']:
        for entry in catalog[sid]['content']:
            if entry['type'] in ('video', 'image'): files[entry['path']] = entry['sha256']
    def hashed(entry):
        path, expected = entry; before = base.stat(path); value = file_sha256(path)
        require(value == expected and base.stat(path) == before, 'Source changed during parallel sealing')
        return path, (value, before)
    with ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 2))) as pool:
        admitted = dict(pool.map(hashed, files.items()))
    native_sha = base.file_sha256
    def admitted_sha(path):
        key = str(path)
        if key not in admitted: return native_sha(path)
        value, identity = admitted[key]; require(base.stat(path) == identity, 'Source changed after hash admission')
        return value
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temporary = f.name
    try:
        base.file_sha256 = admitted_sha
        return base.seal(temporary, output)
    finally:
        base.file_sha256 = native_sha; Path(temporary).unlink()


def run(profile):
    p = read(profile); science(p); verify_dependencies(p)
    control = read(p['control_file'])
    if not control['paused'] and control['generation'] == p['control_generation']:
        admission = storage_admission(p)
        write_new(Path(p['output']).with_suffix('.storage.json'), admission)
    previous = base.Boundary; original_identity = base.worker_identity
    base.Boundary = Boundary; base.worker_identity = worker_identity
    try: return base.run(profile)
    finally: base.Boundary = previous; base.worker_identity = original_identity


def supervise(profile):
    from . import refinement_worker as worker
    p = read(profile); science(p); verify_dependencies(p)
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.refinement_worker', MODULE) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try: return worker.supervise(profile)
    finally: worker.subprocess.Popen = native


def closed(profile, allowed=('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD')):
    p, result, raw = ORIGINAL_CLOSED(profile, allowed)
    science(p); verify_dependencies(p)
    require(result['completed_ids'] == p['assigned_ids'][:len(raw)]
            and read(Path(p['output']).with_suffix('.wait.json'))['completed_ids'] == list(raw),
            'Completion is not a clean ordered prefix')
    require(result.get('unattempted_ids', p['assigned_ids'][len(raw):]) == p['assigned_ids'][len(raw):],
            'Unattempted suffix differs')
    if raw:
        from .gemma_x_full_results import audit_segment
        audit_segment(p['workspace_root'], profile)
    return p, result, raw


@contextmanager
def queue_context():
    native = lifecycle.subprocess.Popen; native_closed = lifecycle.closed
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.nemo_x_full', MODULE) for x in argv], **kwargs)
    lifecycle.subprocess.Popen = popen; lifecycle.closed = closed
    try: yield
    finally: lifecycle.subprocess.Popen = native; lifecycle.closed = native_closed


def queue(plan):
    q = read(plan)
    for ref in q['profiles']:
        p = verified(ref); science(p); verify_dependencies(p)
    with queue_context(): return lifecycle.queue(plan)


def activate(plan):
    q = read(plan); require(q['status'] == 'SEALED' and q['automatic_retry'] is False, 'Only sealed finite queue can activate')
    with lifecycle.controller_lock(q['controller_lock']):
        require(not Path(q['state_path']).exists(), 'Attempted queue cannot reactivate')
        for ref in q['profiles']:
            p = verified(ref); science(p); verify_dependencies(p); storage_admission(p); c = read(p['control_file'])
            require(c['generation'] == p['control_generation'], 'Stale/fenced continuation')
            out = Path(p['output'])
            require(not out.exists() and not out.with_suffix('.launch.json').exists()
                    and not out.with_suffix('.wait.json').exists()
                    and not any((Path(p['claim_root']) / (s + '.json')).exists() for s in p['assigned_ids']),
                    'Attempted or claimed worker cannot activate')
        for ref in q['profiles']:
            p = verified(ref); c = read(p['control_file']); c.update(paused=False, activated_queue=descriptor(plan))
            atomic_json(p['control_file'], c)
    return {'status': 'ACTIVATED_NOT_LAUNCHED', 'queue': descriptor(plan), 'model_calls': 0}


def prepare_resume(profile, output):
    p = read(profile)
    with lifecycle.controller_lock(Path(p['control_file']).with_suffix('.resume.lock')):
        return _prepare_resume_locked(profile, output)


def _prepare_resume_locked(profile, output):
    p, result, raw = closed(profile); control = read(p['control_file'])
    require(control['paused'] and control['generation'] == p['control_generation'], 'Explicit unfenced pause required')
    ids = lifecycle.remaining(p, result, raw); require(ids, 'Nothing unattempted remains')
    q = copy.deepcopy(p); generation = control['generation'] + 1
    for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper', 'full_dependencies'): q.pop(key, None)
    worker = p['card'] + '-full-segment' + str(generation)
    q.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
             output=str(Path(p['output']).parent / worker), pause_file=str(Path(output).with_suffix('.PAUSE.json')),
             continuation_parent=descriptor(profile), continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(q, f); temporary = f.name
    try: answer = seal(temporary, output)
    finally: Path(temporary).unlink()
    queue_path = Path(output).with_name(worker + '-queue.json')
    write_new(queue_path, {'status': 'SEALED', 'automatic_retry': False, 'profiles': [descriptor(output)],
                         'state_path': str(Path(p['output']).parent / (worker + '-queue-state.json')),
                         'controller_lock': str(queue_path.with_suffix('.lock'))})
    control.update(generation=generation, paused=True, continuation_profile=descriptor(output)); atomic_json(p['control_file'], control)
    _append(Path(p['control_file']).with_suffix('.events.jsonl'), {'event': 'CONTINUATION_PREPARED_PAUSED',
            'parent': descriptor(profile), 'next': descriptor(output), 'remaining': len(ids), 'owner': owner_identity()})
    return {**answer, 'status': 'CONTINUATION_PREPARED_PAUSED', 'queue': descriptor(queue_path), 'model_calls': 0}


def prepare(root, qualification_plan, qualification_summary=None, allocation=None):
    """Create drafts only, even when qualification has not yet executed."""
    root = Path(root).resolve(); plans = root / 'plans' / NAME; arc = Archive(root)
    require(not (plans / 'plan.json').exists(), 'Full execution plan already exists; refuse overwrite')
    qpath = Path(qualification_plan).resolve(); qp = read(qpath)
    require(qp['schema'] == QUALIFICATION_SCHEMA and set(qp['cards']) == set(CARDS), 'Wrong qualification plan')
    prior_path = root / 'plans/nemo-x-full-20261029/full-allocation.json'; prior = read(prior_path)
    require(prior['task_counts'] == COUNTS and prior['maximum_new_full_calls'] == 9456, 'Accepted9456 scope differs')
    items = {}; bindings = []
    for card in ('fourier0', 'radon0'):
        ref = prior['cards'][card]['source_catalog']; catalog = arc.ref(ref); bindings.append(ref)
        require(not set(items) & set(catalog['items']), 'Canonical source catalogs overlap')
        items.update(catalog['items'])
    require(Counter(x['task'] for x in items.values()) == COUNTS, 'Canonical9456 scope differs')
    for item in items.values(): source_check(item, item)
    selected = read(allocation)['cards'] if allocation else prior['cards']
    assigned = {card: selected[card]['assigned_ids'] for card in CARDS}
    flat = [s for ids in assigned.values() for s in ids]
    require(len(flat) == len(set(flat)) == 9456 and set(flat) == set(items), 'Allocation does not partition canonical9456')
    templates = {card: arc.ref(qp['cards'][card]['qualified_profile']) for card in CARDS}
    for card, p in templates.items():
        require(p['card'] == card and p['model'] == 'gemma_x_full' and p['base_model'] == 'gemma', 'Qualified template/card differs')
    plans.mkdir(parents=True, exist_ok=True)
    write_new(plans / 'canonical-sources.json', {'schema': NAME + '/canonical-v1', 'items': items, 'accepted_catalogs': bindings})
    cards = {}; catalog_paths = {}
    for card in CARDS:
        remote = templates[card]['workspace_root']; host = 'radon' if card == 'radon0' else 'fourier'
        path = plans / (host + '-sources.json')
        if host not in catalog_paths:
            ids = [sid for c in CARDS if templates[c]['workspace_root'] == remote for sid in assigned[c]]
            write_new(path, {'schema': NAME + '/sources-v1', 'items': {s: rebase(items[s], remote) for s in ids}})
            catalog_paths[host] = path
        cards[card] = {'assigned_ids': assigned[card], 'count': len(assigned[card]),
                       'source_catalog': remote_ref(path, root, remote)}
    summary_path = (Path(qualification_summary) if qualification_summary else root / 'runs/reproducibility' / NAME / 'qualification-summary.json').resolve(strict=False)
    plan = {'schema': NAME + '/execution-v1', 'status': 'PREPARED_NOT_LAUNCHED', 'maximum_full_calls': 9456,
            'task_counts': COUNTS, 'cards': cards, 'canonical_catalog': descriptor(plans / 'canonical-sources.json'),
            'qualification_plan': descriptor(qpath), 'qualification_summary_path': str(summary_path),
            'qualification_summary_relative': str(summary_path.relative_to(root)),
            'accepted_allocation': descriptor(prior_path), 'custom_allocation': descriptor(allocation) if allocation else None,
            'automatic_retry': False, 'qualified_outputs_in_full_curve': False, 'selection_uses_model_answers': False}
    write_new(plans / 'plan.json', plan)
    write_new(plans / 'full-allocation.json', {'schema': NAME + '/allocation-v1', 'cards': cards, 'task_counts': COUNTS,
              'maximum_new_full_calls': 9456, 'experiment_plan': descriptor(plans / 'plan.json')})
    for card in CARDS:
        p = copy.deepcopy(templates[card]); remote = p['workspace_root']
        for key in OPERATIONAL: p.pop(key, None)
        worker = card + '-full-segment0'; control = plans / (card + '-control.json')
        write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
        p.update(worker=worker, phase='full', assigned_ids=assigned[card], maximum_model_calls=len(assigned[card]),
                 source_catalog=cards[card]['source_catalog'], qualified_profile=qp['cards'][card]['qualified_profile'],
                 canonical_catalog=remote_ref(plans / 'canonical-sources.json', root, remote),
                 qualification_plan=remote_ref(qpath, root, remote), experiment_plan=remote_ref(plans / 'plan.json', root, remote),
                 output=str(Path(remote) / 'runs/reproducibility' / NAME / worker),
                 claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / 'claims' / card),
                 pause_file=str(Path(remote) / 'plans' / NAME / (worker + '-PAUSE.json')),
                 control_file=str(Path(remote) / 'plans' / NAME / control.name), control_generation=0,
                 automatic_retry=False, formal_auto_resume=False, variant='X_NATIVE_COMPACT')
        science(p, arc.ref, require_gate=False); write_new(plans / (worker + '-draft.json'), p)
    return {'status': 'PREPARED_NOT_LAUNCHED', 'plan': descriptor(plans / 'plan.json'),
            'cards': {c: len(v) for c, v in assigned.items()}, 'new_model_calls': 0, 'qualification_pass_claimed': False}


def stage(plan, card, authorization, qualification_summary=None):
    plan_path = Path(plan); spec = read(plan_path); require(card in CARDS, 'Unknown card')
    draft = plan_path.parent / (card + '-full-segment0-draft.json'); p = read(draft)
    require(p['hostname'] == socket.gethostname(), 'Wrong staging host')
    summary = qualification_summary or str(Path(p['workspace_root']) / spec['qualification_summary_relative'])
    p.update(qualification_summary=descriptor(summary), execution_authorization=descriptor(authorization))
    output = draft.with_name(card + '-full-segment0-profile.json')
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temporary = f.name
    try: sealed = seal(temporary, output)
    finally: Path(temporary).unlink()
    queue_path = plan_path.parent / (card + '-full-queue.json')
    write_new(queue_path, {'status': 'SEALED', 'automatic_retry': False, 'profiles': [descriptor(output)],
                          'state_path': str(Path(p['output']).parent / (card + '-full-queue-state.json')),
                          'controller_lock': str(queue_path.with_suffix('.lock'))})
    return {**sealed, 'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(queue_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'stage', 'seal', 'activate', 'run', 'supervise',
                                           'queue', 'pause', 'prepare-resume', 'audit-closed'))
    for name in ('root', 'qualification-plan', 'qualification-summary', 'allocation', 'plan', 'card',
                 'authorization', 'draft', 'output', 'profile', 'control'): parser.add_argument('--' + name)
    a = parser.parse_args()
    if a.command == 'prepare': result = prepare(a.root, a.qualification_plan, a.qualification_summary, a.allocation)
    elif a.command == 'stage': result = stage(a.plan, a.card, a.authorization, a.qualification_summary)
    elif a.command == 'seal': result = seal(a.draft, a.output)
    elif a.command == 'activate': result = activate(a.plan)
    elif a.command == 'run': result = run(a.profile)
    elif a.command == 'supervise': result = supervise(a.profile)
    elif a.command == 'queue': result = queue(a.plan)
    elif a.command == 'pause': result = lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': result = prepare_resume(a.profile, a.output)
    else:
        _, terminal, raw = closed(a.profile)
        result = {'status': 'CLOSED_PASS', 'worker_status': terminal['status'], 'completed': len(raw), 'new_model_calls': 0}
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
