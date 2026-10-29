"""One finite 144-input Qwen X T2/T3 pass on each of three cards.

Only lifecycle and admission checks are added. The official X-QSA scientific
runner, reader, selector mathematics, model and processor remain unchanged.
Preparation is CPU-only; staging, activation and launch are separate commands.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import socket
import tempfile

from . import nemo_x_full as lifecycle
from . import official_x_runner_qsa as qsa
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

base = qsa.base
NAME = 'qwen-x-crosscard-t23-20261029'
CARDS = ('fourier0', 'fourier1', 'radon0')
ROOTS = {'fourier': '/home/yang/cvpr2027_benchmark', 'radon': '/mnt/sda/q/cvpr2027_benchmark'}
MM = {'do_resize': False, 'do_sample_frames': False}
QSA_SOURCE = 'd69e0cfff54ec41a3402bba2a333f3d76d042efd9186304d9e6c5c9d83b1712c'


def fixed_implementation():
    expected = {base.__file__: '67fce752e98e6783b1536ae0bf12bb47d4382ba2d2d2c89d19f1c158dc10aac8',
        qsa.v2.__file__: '42be6592437675a213f8a50c62641d002e76b54bf83b435b9aabf45320870fff',
        qsa.__file__: '1323adaef38d46b873ce39f206a401a270f6f11683cf047653f7b070f8680c95',
        qsa.selector.__file__: QSA_SOURCE,
        str(Path(__file__).resolve().parents[1]/'runtime_compat/classic_precision_qsa_determinism.py'): qsa.selector.HELPER_SOURCE_SHA}
    for path, sha in expected.items():
        require(file_sha256(path) == sha, 'Qualified scientific implementation changed: '+path)
    return expected


def validate_runtime(value):
    require(value['model_class'] == 'vllm.models.qwen4_exp.nvidia.model.Qwen4ExpForConditionalGeneration', 'Different Qwen implementation')
    require(value['method_kernel_counts'].get('method:vllm.model_executor.layers.quantization.modelopt.ModelOptNvFp4FusedMoE') == 48, 'Actual quantized MoE differs')
    require(value['method_kernel_counts'].get('moe_kernel:vllm.model_executor.layers.fused_moe.modular_kernel.FusedMoEKernel') == 48, 'Actual MoE kernel differs')
    require(value['actual_backends'] == {'nvfp4_backend:NvFp4MoeBackend.FLASHINFER_CUTLASS': 48}, 'Actual Qwen backend differs')
    require('enable_flashinfer_autotune=False' in value['kernel_config'] and "moe_backend='auto'" in value['kernel_config'] and "linear_backend='auto'" in value['kernel_config'], 'Official kernel policy differs')
    require(value['mm_device_do_normalize'] is True, 'Official GPU normalization differs')
    r = value['custom_qsa_selector']
    require(r['status'] == 'APPLIED_X_QSA' and r['actual_function_is_canonical_wrapper']
            and r['native_function_verified_before_replacement'] and r['selector_source_sha256'] == QSA_SOURCE
            and r['mathematics_source_sha256'] == qsa.selector.HELPER_SOURCE_SHA
            and r['helper_status_after_import'] is None and not r['helper_import_added_meta_path_hook']
            and not r['SDK_files_modified'] and not r['forward_processor_or_weights_changed'], 'Actual QSA application differs')
    return value


def worker_identity(worker, model):
    require(model == 'qwen_x_qsa', 'Explicit X-QSA identity required')
    return validate_runtime(qsa.worker_identity(worker, model))


def science(p, resolve=verified):
    require(p['model'] == 'qwen_x_qsa' and p['base_model'] == 'qwen' and not p.get('bootstrap') and not p.get('request_adapter'), 'Unexpected model/input patch')
    require(p['phase'] == 'crosscard' and p['resource_group'] == 'T23' and p['card'] in CARDS and p['cold'] == 1, 'Wrong finite experiment')
    e = p['engine_kwargs']
    expected = {'max_model_len': 65536, 'kv_cache_memory_bytes': 4*1024**3,
        'max_num_batched_tokens': 49152, 'cpu_offload_gb': 24, 'cpu_offload_params': ['experts'],
        'engram_config': {'cpu_offload': True}, 'max_num_seqs': 1, 'tensor_parallel_size': 1,
        'enforce_eager': True, 'compilation_config': {'mode': 0}, 'enable_prefix_caching': False,
        'mm_processor_cache_gb': 0, 'enable_chunked_prefill': True, 'kv_cache_dtype': 'bfloat16',
        'quantization': 'modelopt', 'seed': 0, 'trust_remote_code': True,
        'disable_custom_all_reduce': True, 'gpu_memory_utilization': 0.78,
        'kernel_config': {'enable_flashinfer_autotune': False}, 'mm_processor_kwargs': MM,
        'worker_extension_cls': 'classic_precision.official_x_qsa_selector.XQSAWorkerExtension',
        'limit_mm_per_prompt': {'audio': 0, 'image': 1, 'video': 1}}
    require(set(e) == set(expected)|{'model'} and all(e.get(k) == v for k, v in expected.items()), 'Tested science/resource configuration differs')
    require(p['request_mm_processor_kwargs'] == MM and p.get('media_io_kwargs', base.MEDIA) == base.MEDIA, 'All-frame input differs')
    require(p['minimum_kv_capacity'] == 65536 and p['vision_observation'] == {
        'allow_official_fused_normalization': True, 'feature_dim': 1536, 'normalized_paths': ['visual.patch_embed'],
        'outer_paths': ['visual'], 'patch_size': 16, 'rule': 'flat_patches', 'temporal_patch_size': 2}, 'Input admission differs')
    require(Path(p['checkpoint']).name == 'fc694b54fb0174e0913e6adf86691ef85a4ead47' and e['model'] == p['checkpoint'], 'Checkpoint differs')
    require(p['automatic_retry'] is False and p['formal_auto_resume'] is False, 'Automatic retry/resume prohibited')
    panel = resolve(p['panel'])
    require(p['assigned_ids'] == panel['assigned_ids'] and len(p['assigned_ids']) == len(set(p['assigned_ids'])) == 144
            and p['maximum_model_calls'] == 144 and panel['task_counts'] == {'T2': 48, 'T3': 96}, 'Finite common panel differs')
    catalog = resolve(p['source_catalog'])
    require(set(catalog['items']) == set(p['assigned_ids']), 'Source catalog contains foreign/missing IDs')
    for sid in p['assigned_ids']:
        item = catalog['items'][sid]; saved = panel['items'][sid]; row = item['row']
        params = row['source_parameters']
        require(item['task'] == saved['task'] and row['truth'] == saved['truth']
                and digest(row['prompt']) == saved['prompt_sha256'] and params == saved['source_parameters']
                and item['source_stream_sha256'] == saved['source_stream_sha256'], 'Panel source/GT/prompt differs')
        media = [m for m in item['content'] if m['type'] == 'video']
        require(len(media) == 1 and media[0]['sha256'] == saved['source_video_sha256']
                and item['video_contract'] == {'fps': 60, 'frames': 180, 'height': 512, 'width': 768}, 'Video identity differs')
        if item['task'] == 'T2':
            require(params['background_Y'] == .4 and params['delta_Y'] == .06 and params['offset_frame']-params['onset_frame'] == 30, 'T2 point differs')
        else:
            require(item['task'] == 'T3' and params['angle_deg'] == .8, 'T3 point differs')


def authorized(p):
    a = verified(p['execution_authorization'])
    require(a['status'] == 'AUTHORIZED' and a['plan_sha256'] == p['prepared_plan_sha256']
            and a['phases'] == ['crosscard'] and a['maximum_new_model_calls'] == 432
            and a['maximum_calls_per_card'] == 144 and a['cards'] == list(CARDS)
            and isinstance(a.get('user_message'), str) and a['user_message'].strip(), 'Missing bound finite human authorization')


class Boundary(lifecycle.Boundary):
    pass


def seal(draft, output):
    fixed_implementation()
    p = read(draft); science(p); authorized(p)
    p['operational_wrapper'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temp = f.name
    try: return qsa.seal(temp, output)
    finally: Path(temp).unlink()


def run(profile):
    fixed_implementation()
    p = read(profile); science(p); authorized(p)
    for key, path in [('operational_wrapper', __file__), ('qsa_entry', qsa.__file__), ('observer_entry', qsa.v2.__file__), ('selector_source', qsa.selector.__file__)]:
        require(file_sha256(path) == p[key]['sha256'], 'Frozen implementation changed: '+key)
    base.worker_identity = worker_identity; base.Boundary = Boundary
    return qsa.v2.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try: return worker.supervise(profile)
    finally: worker.subprocess.Popen = native


def closed(profile_path, allowed=('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD')):
    p, result, raw = lifecycle.closed(profile_path, allowed)
    science(p); authorized(p)
    require(result['completed_ids'] == p['assigned_ids'][:len(raw)]
            and read(Path(p['output']).with_suffix('.wait.json'))['completed_ids'] == list(raw), 'Not an exact clean ordered prefix')
    return p, result, raw


def stage(plan_path, card, authorization_path):
    plan = read(plan_path)
    require(plan['schema'] == 'qwen-X-crosscard-t23-v1' and card in CARDS
            and plan['status'] == 'PREPARED_NOT_LAUNCHED' and plan['maximum_new_model_calls'] == 432, 'Wrong finite plan')
    require(plan['runtime_entry']['sha256'] == file_sha256(__file__), 'Prepared wrapper changed')
    p = verified(plan['cards'][card]['draft'])
    require(p['hostname'] == socket.gethostname(), 'Wrong staging host')
    p.update(prepared_plan_sha256=file_sha256(plan_path), execution_authorization=descriptor(authorization_path))
    p.pop('status', None)
    target = Path(plan['cards'][card]['draft']['path'].replace('-draft.json', '-profile.json'))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temp = f.name
    try: answer = seal(temp, target)
    finally: Path(temp).unlink()
    return {**answer, 'status': 'SEALED_PAUSED_NOT_LAUNCHED'}


def activate(profile_path):
    p = read(profile_path); science(p); authorized(p)
    c = read(p['control_file'])
    require(c['generation'] == p['control_generation'] and not Path(p['output']).exists()
            and not Path(p['output']).with_suffix('.launch.json').exists()
            and not any((Path(p['claim_root'])/(sid+'.json')).exists() for sid in p['assigned_ids']), 'Attempted/stale worker cannot reactivate')
    c.update(paused=False, activated_profile=descriptor(profile_path)); atomic_json(p['control_file'], c)
    return {'status': 'ACTIVATED_NOT_LAUNCHED', 'profile': descriptor(profile_path), 'model_calls': 0}


def remote_ref(path, root, remote):
    return {'path': str(Path(remote)/Path(path).relative_to(root)), 'sha256': file_sha256(path)}


def prepare(root):
    from collections import Counter
    from .official_x_results import Archive
    root = Path(root).resolve(); out = root/'plans'/NAME
    require(not out.exists(), 'Refuse overwrite of prepared finite experiment')
    arc = Archive(root); old = root/'plans/official-x-20261028'
    template = read(old/'qwen-xqsa-noautotune-smoke12-cold1-profile.json')
    rest = read(old/'qwen-xqsa-noautotune-validation-rest132-cold1-profile.json')
    t2ids = [s for s in template['assigned_ids']+rest['assigned_ids'] if s.startswith('t2_')]
    allocation = read(root/'plans/qwen-x-full-20261029/full-allocation.json')
    items = {}
    for card in ('fourier0', 'radon0'): items.update(arc.ref(allocation['cards'][card]['source_catalog'])['items'])
    t3ids = sorted(s for s, i in items.items() if i['task'] == 'T3' and i['row']['source_parameters']['angle_deg'] == .8)
    ids = t2ids+t3ids
    require(len(t2ids) == 48 and len(t3ids) == 96 and len(set(ids)) == 144, 'Selected panel count differs')
    panel = {'schema': 'qwen-X-crosscard-fixed144-v1', 'assigned_ids': ids, 'task_counts': {'T2': 48, 'T3': 96},
        'selection': 'T2 BG=.4/Delta=.06/exposure30: prior X38/48; T3 .8deg: prior C77/96; no X .8 results used.', 'items': {}}
    for sid in ids:
        i = items[sid]; row = i['row']; media = [m for m in i['content'] if m['type'] == 'video']
        panel['items'][sid] = {'sample_id': sid, 'task': i['task'], 'truth': row['truth'], 'source_parameters': row['source_parameters'],
            'prompt_sha256': digest(row['prompt']), 'source_video_sha256': media[0]['sha256'], 'source_stream_sha256': i['source_stream_sha256']}
    out.mkdir(parents=True); write_new(out/'panel.json', panel)
    keep = ('model', 'base_model', 'checkpoint', 'checkpoint_assets', 'engine_kwargs', 'expected_packages', 'cleared_environment',
        'request_mm_processor_kwargs', 'vision_observation', 'minimum_kv_capacity', 'choices_by_task', 'torch_num_threads')
    cards = {}
    for card in CARDS:
        host = 'radon' if card == 'radon0' else 'fourier'; remote = ROOTS[host]
        def rebase(value):
            if isinstance(value, str): return value.replace(ROOTS['fourier'], remote)
            if isinstance(value, list): return [rebase(x) for x in value]
            if isinstance(value, dict): return {k: rebase(v) for k, v in value.items()}
            return value
        catalog = out/(host+'-sources.json')
        if not catalog.exists(): write_new(catalog, {'schema': 'qwen-X-crosscard-sources-v1', 'items': {s: rebase(items[s]) for s in ids}})
        hardware = read(root/'plans/nemo-x-full-20261029'/(card+'-qualification-cold1-profile.json'))
        native = read(root/'plans/qwen-noqsa-cosmos-speed-20261028/radon/qwen-radon0-noqsa-noautotune-cold1-profile.json') if host == 'radon' else template
        p = rebase({k: copy.deepcopy(template[k]) for k in keep})
        worker = card+'-t23-cold1'; control = out/(card+'-control.json')
        write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
        env = dict(native['environment']); env['CUDA_VISIBLE_DEVICES'] = hardware['environment']['CUDA_VISIBLE_DEVICES']
        p.update(worker=worker, card=card, resource_group='T23', phase='crosscard', cold=1, workspace_root=remote,
            python=native['python'], hostname=hardware['hostname'], checkpoint=native['checkpoint'],
            engine_kwargs={**p['engine_kwargs'], 'model': native['checkpoint']}, environment=env,
            expected_packages=native['expected_packages'], gpu_uuid=hardware['gpu_uuid'], physical_lock=hardware['physical_lock'],
            resources={'min_available_ram_gib': 64, 'min_free_gpu_gib': 72}, assigned_ids=ids, maximum_model_calls=144,
            source_catalog=remote_ref(catalog, root, remote), sources={}, panel=remote_ref(out/'panel.json', root, remote),
            output=str(Path(remote)/'runs/reproducibility'/NAME/worker),
            claim_root=str(Path(remote)/'runs/reproducibility'/NAME/'claims'/card/'attempt1'),
            pause_file=str(Path(remote)/'plans'/NAME/(worker+'-PAUSE.json')),
            control_file=str(Path(remote)/'plans'/NAME/control.name), control_generation=0,
            automatic_retry=False, formal_auto_resume=False, status='PREPARED_NOT_LAUNCHED', variant='X_QSA_NO_FLASHINFER_AUTOTUNE')
        path = out/(worker+'-draft.json'); write_new(path, p)
        cards[card] = {'draft': remote_ref(path, root, remote), 'count': 144, 'task_counts': {'T2': 48, 'T3': 96}}
    plan = {'schema': 'qwen-X-crosscard-t23-v1', 'status': 'PREPARED_NOT_LAUNCHED', 'automatic_retry': False,
        'maximum_new_model_calls': 432, 'maximum_calls_per_card': 144, 'repeats_per_card': 1, 'assigned_ids': ids,
        'task_counts_per_card': {'T2': 48, 'T3': 96}, 'cards': cards,
        'panel': remote_ref(out/'panel.json', root, ROOTS['fourier']),
        'runtime_entry': {'path': str(Path(ROOTS['fourier'])/'code/vlm-psychophysics/classic_precision/qwen_x_crosscard.py'), 'sha256': file_sha256(__file__)},
        'authorization_path': 'plans/'+NAME+'/authorization.json',
        'scientific_acceptance': 'Per-ID technical admission required; accuracy is measured, never an admission gate.',
        'comparison': 'Compare three new same-ID X passes. Prior T2 X and T3 C remain historical, with no required equality.',
        'no_automatic_successor': True}
    write_new(out/'plan.json', plan)
    return {'status': plan['status'], 'plan': descriptor(out/'plan.json'), 'new_model_calls': 0, 'budget': 432}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'stage', 'activate', 'seal', 'run', 'supervise', 'pause', 'audit-closed'])
    for name in ('root', 'plan', 'card', 'authorization', 'draft', 'output', 'profile', 'control'): parser.add_argument('--'+name)
    a = parser.parse_args()
    if a.command == 'prepare': value = prepare(a.root)
    elif a.command == 'stage': value = stage(a.plan, a.card, a.authorization)
    elif a.command == 'activate': value = activate(a.profile)
    elif a.command == 'seal': value = seal(a.draft, a.output)
    elif a.command == 'run': value = run(a.profile)
    elif a.command == 'supervise': value = supervise(a.profile)
    elif a.command == 'pause': value = lifecycle.pause(a.control)
    else:
        _, result, raw = closed(a.profile); value = {'status': 'CLOSED_PASS', 'worker_status': result['status'], 'completed': len(raw)}
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
