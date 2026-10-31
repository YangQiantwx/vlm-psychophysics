"""Finite Radon-only RAM placement qualification using official vLLM UVA.

Only the two unspent96 cold starts are admitted. Scientific input, checkpoint,
mixed NVFP4 kernels, generation, context and BF16 KV policy remain unchanged.
No GPU is initialized by prepare/stage or by importing this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import ctypes
import json
import os
from pathlib import Path
import socket
import tempfile

from . import gemma_x_full as full
from . import gemma_x_qualification as qualification
from . import gemma_x_provisional as provisional
from . import official_x_runner as base
from . import refinement_worker as lifecycle_worker
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256, atomic_json, _append

NAME = 'gemma-x-ram-qualification-20261031'
MODULE = 'classic_precision.gemma_x_ram_qualification'
FILTER = ['self_attn.qkv_proj.weight']
GIB = 1024**3
EXPECTED_BYTES = 4404019200  # 20 sliding + 4 full-attention QKV parameters.
RESOURCES = {'min_free_gpu_gib': 77, 'min_available_ram_gib': 28}
RAM_FIELDS = {'ram_adapter', 'ram_deployment_authorization', 'original_qualification_profile',
              'fourier_output_reference'}
OUTPUT_KEYS = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason', 'truth', 'question')
_SCIENCE = qualification.science
_VERIFY = qualification.verify_code
_SEAL = qualification.seal
_IDENTITY = full.worker_identity
_READ_PROBE = base.read_probe
_SUPERVISE = lifecycle_worker.supervise
_SUMMARIZE = qualification.summarize


def expected_parameters():
    return {f'language_model.model.layers.{i}.self_attn.qkv_proj.weight':
            [20480 if i % 6 == 5 else 16384, 5376] for i in range(24)}


def validate_placement(receipt, phase):
    expected = expected_parameters(); actual = receipt['offloaded_parameters']
    require(receipt['schema'] == NAME + '/placement-v1' and receipt['phase'] == phase
            and receipt['offload_backend'] == 'official_vllm_UVA'
            and set(actual) == set(expected)
            and receipt['actual_offloaded_parameter_bytes'] == EXPECTED_BYTES,
            'Actual RAM residency receipt is incomplete')
    for name, shape in expected.items():
        entry = actual[name]
        require(entry['dtype'] == 'torch.bfloat16' and entry['shape'] == shape
                and entry['bytes'] == shape[0] * shape[1] * 2
                and entry['torch_device'].startswith('cuda')
                and entry['cuda_memory_type'] == 1 and entry['host_pointer_nonzero'],
                'Actual mapped-host BF16 parameter differs: ' + name)
    require(sum(x['bytes'] for x in actual.values()) == EXPECTED_BYTES
            and receipt['gpu_free_plus_torch_reusable_bytes'] >= GIB
            and receipt['available_ram_bytes'] >= 20 * GIB,
            'RAM placement has insufficient verified memory margin')


def science(p, resolve=verified):
    original = resolve(p['original_qualification_profile'])
    deployment = resolve(p['ram_deployment_authorization'])
    require(p['card'] == original['card'] == 'radon0' and p['cold'] == original['cold'] in (1, 2)
            and p['maximum_model_calls'] == 96 and p['assigned_ids'] == original['assigned_ids'],
            'Only the two original unspent Radon96 cold starts are admitted')
    require(deployment['status'] == 'AUTHORIZED_REMAINING_RADON_QUALIFICATION_ONLY'
            and deployment['maximum_remaining_model_calls'] == 192
            and deployment['original_total_qualification_budget'] == 576
            and deployment['resource_budget'] == RESOURCES
            and deployment['offload_expected_parameter_bytes'] == EXPECTED_BYTES,
            'Missing finite RAM deployment authorization')
    require(p['ram_adapter']['sha256'] == file_sha256(__file__), 'RAM deployment adapter changed')
    e = p['engine_kwargs']
    require(e['cpu_offload_gb'] == 4 and e['cpu_offload_params'] == FILTER
            and e['offload_backend'] == 'uva' and p['resources'] == RESOURCES,
            'Changed explicit RAM placement/budget')
    require(p['environment'] == original['environment']
            and not p.get('bootstrap') and p['automatic_retry'] is False
            and p['formal_auto_resume'] is False, 'Unexpected runtime/input override or automatic retry')
    require(not any(k in p['environment'] for k in (
        'VLLM_WEIGHT_OFFLOADING_DISABLE_UVA', 'VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY')),
        'Official default UVA/pinned-memory behavior required')
    normalized = copy.deepcopy(p)
    for k in RAM_FIELDS: normalized.pop(k, None)
    normalized['resources'] = original['resources']
    normalized['engine_kwargs']['cpu_offload_gb'] = 0
    normalized['engine_kwargs'].pop('cpu_offload_params')
    normalized['engine_kwargs'].pop('offload_backend')
    require(full.frozen_settings(normalized) == full.frozen_settings(original),
            'A setting other than declared RAM placement/resources changed')
    _SCIENCE(normalized, resolve)
    reference = resolve(p['fourier_output_reference'])
    require(reference['status'] == 'PASS_FOURIER_FOUR_COLD_OUTPUT_REFERENCE'
            and reference['selection_sha256'] == p['qualification_selection']['sha256']
            and reference['assigned_ids'] == p['assigned_ids'], 'Fourier reference differs')


def verify_code(p):
    _VERIFY(p)
    require(p['ram_adapter']['sha256'] == file_sha256(__file__), 'RAM adapter source changed')


def _pointer_attributes(pointer):
    """Driver metadata distinguishes mapped host RAM from ordinary CUDA storage."""
    driver = ctypes.CDLL('libcuda.so.1')
    query = driver.cuPointerGetAttribute
    query.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64)
    query.restype = ctypes.c_int
    memory_type = ctypes.c_uint(); host = ctypes.c_uint64(); device = ctypes.c_uint64()
    for attribute, output in ((2, memory_type), (4, host), (3, device)):
        status = query(ctypes.byref(output), attribute, pointer)
        require(status == 0, f'CUDA pointer attribute {attribute} failed: {status}')
    return {'cuda_memory_type': memory_type.value, 'host_pointer': hex(host.value),
            'device_pointer': hex(device.value), 'host_pointer_nonzero': bool(host.value)}


def ram_worker_identity(worker, model):
    """Runs only inside the already authorized, loaded qualification engine."""
    import torch
    from vllm.model_executor.offloader import get_offloader
    from .inference import available_ram_bytes
    value = _IDENTITY(worker, model)
    phase = 'after_first_budgeted_request' if getattr(worker, '_ram_placement_seen', False) else 'after_load'
    actual = {}; expected = expected_parameters(); placement = Counter()
    for name, parameter in worker.get_model().named_parameters():
        size = parameter.numel() * parameter.element_size()
        is_uva = bool(getattr(parameter, '_vllm_is_uva_offloaded', False))
        placement['uva' if is_uva else parameter.device.type] += size
        if not is_uva and parameter.device.type != 'cpu': continue
        require(name in expected and parameter.dtype == torch.bfloat16
                and list(parameter.shape) == expected[name], 'Unexpected offloaded parameter: ' + name)
        attrs = _pointer_attributes(parameter.data_ptr())
        require(is_uva and parameter.device.type == 'cuda'
                and attrs['cuda_memory_type'] == 1 and attrs['host_pointer_nonzero'],
                'Offload marker is not backed by actual mapped host RAM: ' + name)
        actual[name] = {'dtype': str(parameter.dtype), 'shape': list(parameter.shape),
                        'bytes': size, 'torch_device': str(parameter.device), **attrs}
    offloader = get_offloader(); size = sum(x['bytes'] for x in actual.values())
    require(set(actual) == set(expected) and size == EXPECTED_BYTES
            and offloader.cpu_offload_bytes == size and offloader.uva_offloading is True,
            'Actual post-load RAM placement does not match the4.1015625GiB budget')
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    available = available_ram_bytes()
    receipt = {'schema': NAME + '/placement-v1', 'phase': phase,
        'offload_backend': 'official_vllm_UVA', 'offloaded_parameters': actual,
        'actual_offloaded_parameter_bytes': size, 'placement_parameter_bytes': dict(placement),
        'gpu_free_bytes': free, 'gpu_total_bytes': total,
        'torch_allocated_bytes': allocated, 'torch_reserved_bytes': reserved,
        'torch_reusable_cached_bytes': reserved - allocated,
        'gpu_free_plus_torch_reusable_bytes': free + reserved - allocated,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'available_ram_bytes': available, 'minimum_effective_gpu_headroom_bytes': GIB,
        'minimum_remaining_ram_bytes': 20 * GIB,
        'actual_kernel_counts_unchanged': True, 'scientific_input_or_generation_changed': False}
    validate_placement(receipt, phase)
    value['ram_placement'] = receipt
    if phase == 'after_load':
        worker._ram_placement_seen = True
        torch.cuda.reset_peak_memory_stats()
    return value


def ram_read_probe(worker, model):
    value = _READ_PROBE(worker, model)
    value['runtime'] = ram_worker_identity(worker, model)
    value.pop('runtime_error', None)
    return value


@contextmanager
def adaptation():
    changes = [(qualification, 'science', science), (qualification, 'verify_code', verify_code),
        (qualification, 'supervise', supervise), (qualification, 'NAME', NAME),
        (full, 'worker_identity', ram_worker_identity), (base, 'read_probe', ram_read_probe)]
    prior = [(obj, key, getattr(obj, key)) for obj, key, _ in changes]
    original_identity = base.worker_identity; original_boundary = base.Boundary
    try:
        for obj, key, value in changes: setattr(obj, key, value)
        yield
    finally:
        base.worker_identity = original_identity; base.Boundary = original_boundary
        for obj, key, value in reversed(prior): setattr(obj, key, value)


def seal(draft, output):
    p = read(draft); p['ram_adapter'] = descriptor(__file__); science(p)
    full.storage_admission(p)
    catalog = verified(p['source_catalog'])['items']
    files = {x['path']: x['sha256'] for sid in p['assigned_ids']
             for x in catalog[sid]['content'] if x['type'] in ('video', 'image')}
    def hashed(item):
        path, expected = item; before = base.stat(path); value = file_sha256(path)
        require(value == expected and base.stat(path) == before, 'Source changed during parallel RAM sealing')
        return path, (value, before)
    with ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 2))) as pool:
        admitted = dict(pool.map(hashed, files.items()))
    original_sha = base.file_sha256
    def cached_sha(path):
        if str(path) not in admitted: return original_sha(path)
        value, stat = admitted[str(path)]
        require(base.stat(path) == stat, 'Source changed after hashing')
        return value
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temporary = stream.name
    try:
        with adaptation():
            base.file_sha256 = cached_sha
            return _SEAL(temporary, output)
    finally:
        base.file_sha256 = original_sha; Path(temporary).unlink()


def _unspent(original, arc):
    out = arc.path(original['output'])
    require(read(arc.path(original['control_file']))['paused'], 'Original Radon qualification remains activated')
    require(not any((arc.path(original['claim_root']) / (s + '.json')).exists()
                    for s in original['assigned_ids']), 'Original Radon qualification has a scientific claim')
    for name in ('raw.jsonl', 'events.jsonl'):
        path = out / name
        require(not path.exists() or not list(full.lifecycle.rows(path)),
                'Original Radon qualification already attempted inputs')
    if out.exists():
        result = read(out / 'result.json'); wait = read(out.with_suffix('.wait.json'))
        require(result['status'] == 'PAUSED_BEFORE_LOAD' and not result['completed_ids']
                and not result['UNCERTAIN_ids'] and result['model_load_started'] is False
                and wait['returncode'] == 0 and wait['real_wait_observed']
                and wait['owned_engines_exited'] and wait['original_physical_lease_released'],
                'Original zero-call Radon launch is not cleanly closed')
    else:
        require(not out.with_suffix('.launch.json').exists(), 'Original Radon launch remains unresolved')


def prepare(root, qualification_plan):
    root = Path(root).resolve(); arc = full.Archive(root); qp = read(qualification_plan)
    plans = root / 'plans' / NAME
    require(not plans.exists(), 'Refuse to overwrite a RAM qualification preparation')
    # The existing complete384 proof is revalidated without rerunning its CPU/GPU references.
    proof_path = root / 'plans' / provisional.NAME / 'fourier-gate.json'
    provisional.gate(descriptor(qualification_plan), descriptor(proof_path), arc.ref)
    selection = arc.ref(qp['selection']); ids = selection['assigned_ids']
    anchor = next(w for w in qp['workers'] if w['card'] == 'fourier0' and w['cold'] == 1)
    anchor_p = arc.ref(anchor['profile']); raw = list(full.lifecycle.rows(arc.path(anchor_p['output']) / 'raw.jsonl'))
    require([x['sample_id'] for x in raw] == ids, 'Fourier reference order differs')
    prompt = raw[0]['actual_prompt_token_ids']
    require(all(x['actual_prompt_token_ids'] == prompt for x in raw), 'T3 prompt varies across original reference')
    reference = {'status': 'PASS_FOURIER_FOUR_COLD_OUTPUT_REFERENCE',
        'selection_sha256': qp['selection']['sha256'], 'fourier_gate': descriptor(proof_path),
        'anchor_profile': anchor['profile'], 'anchor_raw': descriptor(arc.path(anchor_p['output']) / 'raw.jsonl'),
        'assigned_ids': ids, 'actual_prompt_token_ids': prompt,
        'outputs': {x['sample_id']: {k: x[k] for k in OUTPUT_KEYS} for x in raw}}
    write_new(plans / 'fourier-output-reference.json', reference)
    deployment = {'status': 'AUTHORIZED_REMAINING_RADON_QUALIFICATION_ONLY',
        'authorization_basis': 'User requested RAM placement; use only the already authorized unspent Radon192 qualification calls',
        'original_total_qualification_budget': 576, 'completed_fourier_calls': 384,
        'maximum_remaining_model_calls': 192, 'cards': ['radon0'], 'cold_starts': [1, 2],
        'original_qualification_plan': descriptor(qualification_plan), 'fourier_gate': descriptor(proof_path),
        'resource_budget': RESOURCES, 'offload_expected_parameter_bytes': EXPECTED_BYTES,
        'budget_derivation': {'original_minimum_gpu_gib': 80,
            'expected_saved_parameter_gib': EXPECTED_BYTES / GIB, 'additional_margin_gib': 1,
            'derived_minimum_gpu_gib': 80 - EXPECTED_BYTES / GIB + 1,
            'rounded_gpu_admission_gib': 77, 'original_ram_minimum_gib': 20,
            'estimated_pinned_allocator_gib': 6, 'extra_host_margin_gib': 2,
            'gpu_admission_is_not_a_runtime_fit_guarantee': True},
        'expected_parameters': expected_parameters(),
        'engine_delta': {'cpu_offload_gb': 4, 'cpu_offload_params': FILTER, 'offload_backend': 'uva'},
        'stop_fourier_required_on_output_or_prompt_mismatch': True,
        'automatic_retry': False, 'formal_radon_launch_authorized_by_this_file': False,
        'official_reference': 'https://docs.vllm.ai/en/stable/api/vllm/config/offload/'}
    write_new(plans / 'ram-deployment-authorization.json', deployment)
    drafts = []
    for w in qp['workers']:
        if w['card'] != 'radon0': continue
        original = arc.ref(w['profile']); p = copy.deepcopy(original); remote = p['workspace_root']
        for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper', 'full_dependencies'):
            p.pop(key, None)
        worker = f'radon0-ram-qualification-cold{w["cold"]}'
        control = plans / (worker + '-control.json')
        write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
        p.update(worker=worker, control_generation=0,
            output=str(Path(remote) / 'runs/reproducibility' / NAME / worker),
            claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / 'claims' / worker),
            control_file=str(Path(remote) / 'plans' / NAME / control.name),
            pause_file=str(Path(remote) / 'plans' / NAME / (worker + '.PAUSE.json')),
            resources=dict(RESOURCES), original_qualification_profile=w['profile'],
            ram_adapter=descriptor(__file__),
            ram_deployment_authorization=full.remote_ref(plans / 'ram-deployment-authorization.json', root, remote),
            fourier_output_reference=full.remote_ref(plans / 'fourier-output-reference.json', root, remote))
        p['engine_kwargs'].update(cpu_offload_gb=4, cpu_offload_params=FILTER, offload_backend='uva')
        science(p, arc.ref)
        draft = plans / (worker + '-draft.json'); write_new(draft, p)
        drafts.append({'cold': w['cold'], 'draft': descriptor(draft)})
    value = {'schema': NAME + '/preparation-v1', 'status': 'PREPARED_NOT_LAUNCHED',
        'original_qualification_plan': descriptor(qualification_plan), 'drafts': drafts,
        'remaining_model_calls': 192, 'resource_budget': RESOURCES,
        'radon_formal_launch_authorized': False, 'automatic_retry': False}
    write_new(plans / 'radon-qualification-plan.json', value)
    return {**value, 'plan': descriptor(plans / 'radon-qualification-plan.json'), 'new_model_calls': 0}


def stage(plan_path):
    plan_path = Path(plan_path); spec = read(plan_path)
    require(spec['status'] == 'PREPARED_NOT_LAUNCHED' and spec['remaining_model_calls'] == 192,
            'Wrong finite RAM preparation')
    require(len(spec['drafts']) == 2 and {x['cold'] for x in spec['drafts']} == {1, 2},
            'Two distinct fresh Radon cold starts required')
    profiles = []
    for entry in sorted(spec['drafts'], key=lambda x: x['cold']):
        draft = plan_path.parent / Path(entry['draft']['path']).name
        require(file_sha256(draft) == entry['draft']['sha256'], 'RAM draft changed')
        p = read(draft); require(p['hostname'] == socket.gethostname(), 'Wrong staging host')
        arc = full.Archive(p['workspace_root']); _unspent(arc.ref(p['original_qualification_profile']), arc)
        output = draft.with_name(draft.name.replace('-draft.json', '-profile.json'))
        seal(draft, output); profiles.append(descriptor(output))
    first = read(profiles[0]['path']); arc = full.Archive(first['workspace_root'])
    qp = arc.ref(spec['original_qualification_plan']); six = copy.deepcopy(qp)
    for w in six['workers']:
        if w['card'] == 'radon0': w['profile'] = profiles[w['cold'] - 1]
    six['cards']['radon0']['qualified_profile'] = profiles[1]
    six.update(status='SEALED_RADON_RAM_PENDING_REAL_QUALIFICATION',
               ram_deployment_adapter=descriptor(__file__), original_qualification_plan=spec['original_qualification_plan'])
    six_path = plan_path.parent / 'six-worker-plan.json'; write_new(six_path, six)
    queue_path = plan_path.parent / 'radon-qualification-queue.json'
    write_new(queue_path, {'status': 'SEALED', 'automatic_retry': False, 'profiles': profiles,
                          'six_worker_plan': descriptor(six_path), 'maximum_model_calls': 192})
    return {'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(queue_path),
            'six_worker_plan': descriptor(six_path), 'maximum_model_calls': 192, 'new_model_calls': 0}


def run(profile):
    p = read(profile); science(p); verify_code(p)
    reference = verified(p['fourier_output_reference']); original_append = base._append
    def observed_append(path, value):
        original_append(path, value)
        if Path(path) != Path(p['output']) / 'raw.jsonl': return
        sid = value['sample_id']; expected = reference['outputs'][sid]
        fields = [k for k in OUTPUT_KEYS if value[k] != expected[k]]
        prompt_equal = value['actual_prompt_token_ids'] == reference['actual_prompt_token_ids']
        receipt = {'sample_id': sid, 'raw_digest': digest(value), 'differing_output_fields': fields,
            'complete_prompt_token_ids_equal': prompt_equal, 'match': not fields and prompt_equal,
            'actual_output': {k: value[k] for k in OUTPUT_KEYS}, 'reference_output': expected,
            'actual_prompt_sha256': digest(value['actual_prompt_token_ids']),
            'reference_prompt_sha256': digest(reference['actual_prompt_token_ids'])}
        original_append(Path(p['output']) / 'ram-equivalence.jsonl', receipt)
        if not receipt['match']:
            atomic_json(Path(p['output']) / 'reproduction-failure.json', {
                **receipt, 'status': 'PAUSE_FOURIER_REQUIRED', 'original_scientific_output_preserved': True,
                'automatic_retry': False, 'new_model_calls_by_comparison': 0})
            raise ValueError('Radon RAM output/prompt reproduction mismatch; pause Fourier now; raw preserved')
    try:
        with adaptation():
            base._append = observed_append
            return qualification.run(profile)
    finally: base._append = original_append


def supervise(profile):
    original = lifecycle_worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([x.replace('classic_precision.refinement_worker', MODULE) for x in argv], **kwargs)
    try:
        lifecycle_worker.subprocess.Popen = popen
        return _SUPERVISE(profile)
    finally: lifecycle_worker.subprocess.Popen = original


def _science_dispatch(p, resolve=verified):
    return science(p, resolve) if p.get('ram_adapter') else _SCIENCE(p, resolve)


def _verify_dispatch(p):
    return verify_code(p) if p.get('ram_adapter') else _VERIFY(p)


def summarize(root, plan, output):
    # The unchanged six-ledger auditor checks every output and prompt token.
    with adaptation():
        qualification.science = _science_dispatch; qualification.verify_code = _verify_dispatch
        value = _SUMMARIZE(root, plan)
    path = Path(root) / 'runs/reproducibility' / NAME / 'qualification-summary.json'
    summary = read(path)
    arc = full.Archive(root); ram_proofs = []
    for w in read(plan)['workers']:
        if w['card'] != 'radon0': continue
        p = arc.ref(w['profile']); science(p, arc.ref); out = arc.path(p['output'])
        for runtime in read(out / 'runtime.json'):
            validate_placement(runtime['ram_placement'], 'after_load')
        deep = list(full.lifecycle.rows(out / 'deep-observations.jsonl'))
        require(len(deep) == 1, 'Missing first budgeted-request RAM residency proof')
        for observed in deep[0]['observations']:
            validate_placement(observed['runtime']['ram_placement'], 'after_first_budgeted_request')
        comparisons = list(full.lifecycle.rows(out / 'ram-equivalence.jsonl'))
        raw = list(full.lifecycle.rows(out / 'raw.jsonl'))
        require([x['sample_id'] for x in comparisons] == p['assigned_ids']
                and all(x['match'] and x['complete_prompt_token_ids_equal']
                        and not x['differing_output_fields'] and x['raw_digest'] == digest(y)
                        for x, y in zip(comparisons, raw))
                and not (out / 'reproduction-failure.json').exists(),
                'Per-request complete Fourier/Radon comparison proof differs')
        ram_proofs.append({'card': w['card'], 'cold': w['cold'], 'profile': w['profile'],
            'runtime': descriptor(out / 'runtime.json'), 'first_request': descriptor(out / 'deep-observations.jsonl'),
            'comparisons': descriptor(out / 'ram-equivalence.jsonl'), 'actual_offloaded_parameter_bytes': EXPECTED_BYTES})
    require(len(ram_proofs) == 2, 'Both Radon RAM cold starts must have actual residency evidence')
    summary['ram_deployment_proofs'] = ram_proofs
    summary['official_runtime_kernel_and_scientific_input_settings_preserved'] = True
    atomic_json(path, summary)
    if output: write_new(output, summary)
    return {**value, 'summary': descriptor(output or path),
            'pause_fourier_required': summary['status'] != full.QUALIFICATION_PASS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'stage', 'seal', 'activate', 'queue',
                                            'run', 'supervise', 'summarize', 'audit-profile'))
    for name in ('root', 'qualification-plan', 'plan', 'profile', 'draft', 'output'):
        parser.add_argument('--' + name)
    a = parser.parse_args()
    if a.command == 'prepare': value = prepare(a.root, a.qualification_plan)
    elif a.command == 'stage': value = stage(a.plan)
    elif a.command == 'seal': value = seal(a.draft, a.output)
    elif a.command == 'run': value = run(a.profile)
    elif a.command == 'supervise': value = supervise(a.profile)
    elif a.command == 'summarize': value = summarize(a.root, a.plan, a.output)
    elif a.command == 'audit-profile':
        p = read(a.profile); science(p, full.Archive(a.root or p['workspace_root']).ref)
        value = {'status': 'CPU_PROFILE_PASS', 'profile': descriptor(a.profile), 'new_model_calls': 0}
    else:
        with adaptation(): value = getattr(qualification, a.command)(a.plan)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
