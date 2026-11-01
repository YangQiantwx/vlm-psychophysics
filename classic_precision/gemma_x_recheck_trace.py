"""Observations for the finite Gemma X recheck; never changes an input or result.

Installed after the original engine's load/autotune. Scientific modules and
sampling arguments remain frozen. GPU copies/profiling add observation overhead.
"""
from contextlib import contextmanager
import functools
from collections import Counter
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import re
import time
import zipfile

from .refinement_worker import descriptor, require, verified
from .storage import _append, atomic_json, digest
from .gemma_x_full_input import PATCHES


def tensor_blob(t):
    import torch
    cpu = t.detach().to('cpu').contiguous()
    data = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
    info = {'shape': list(t.shape), 'dtype': str(t.dtype),
            'device': str(t.device), 'stride': list(t.stride()),
            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    if t.ndim == 3 and list(t.shape[1:]) in ([PATCHES,768], [PATCHES,2]):
        width = len(data) // t.shape[0]
        info['per_frame_sha256'] = [hashlib.sha256(memoryview(data)[i*width:(i+1)*width]).hexdigest()
                                    for i in range(t.shape[0])]
    return info, data


def tensors(value, prefix=''):
    import torch
    if isinstance(value, torch.Tensor):
        yield prefix or 'tensor', value
    elif isinstance(value, dict):
        for k, v in value.items(): yield from tensors(v, prefix + '/' + str(k))
    elif isinstance(value, (tuple, list)):
        for k, v in enumerate(value): yield from tensors(v, prefix + '/' + str(k))


def safe(value):
    if value is None or isinstance(value, (str, int, float, bool)): return value
    if isinstance(value, (tuple, list)): return [safe(v) for v in value]
    if isinstance(value, dict): return {str(k): safe(v) for k, v in value.items()}
    return {'type': type(value).__module__ + '.' + type(value).__qualname__, 'repr': repr(value)}


def cache_snapshot():
    from flashinfer.autotuner import AutoTuner
    tuner = AutoTuner.get()
    return {'identity': safe(tuner._winner_cache_identity()),
            'entries': [{'key': safe(k), 'tactic': safe(v[0]),
                         'opt_shapes': safe(v[1].get_opt_shapes())}
                        for k, v in tuner._winner_cache().items()]}


def weight_fingerprint(network):
    rows = []
    for kind, values in [('parameter', network.named_parameters()), ('buffer', network.named_buffers())]:
        for name, tensor in values:
            info, data = tensor_blob(tensor)
            rows.append({'kind': kind, 'name': name, **info})
            del data
    stable = [{k: v for k, v in x.items() if k != 'device'} for x in rows]
    return {'tensors': rows, 'logical_bytes': sum(x['bytes'] for x in rows),
            'content_manifest_sha256': digest(stable)}


def _store(state, label, tensor, *, binary=False, **metadata):
    info, data = tensor_blob(tensor)
    row = {'kind': label, 'forward': state['forward'], 'layer': state['layer'],
           'sample_step': state['sample_step'], **metadata, **info}
    if binary:
        name = str(len(state['records'])) + '.bin'
        state['bundle'].writestr(name, data)
        row['binary_entry'] = name
    state['records'].append(row)
    return row


def observe(state, function, *args, **kwargs):
    """An observation failure must never suppress the original model output."""
    try: return function(*args, **kwargs)
    except Exception as error:
        state['records'].append({'kind':'observation_error', 'function':function.__name__, 'error':repr(error)})
        return None


def install(worker, output):
    import torch
    from flashinfer.autotuner import AutoTuner
    network = worker.get_model()
    require(not hasattr(worker, '_gemma_recheck_trace'), 'Trace already installed')
    root = Path(output) / 'diagnostics'; root.mkdir()
    state = {'active': False, 'root': root, 'handles': [], 'network': network,
             'records': [], 'forward': -1, 'layer': None, 'sample_step': -1}
    worker._gemma_recheck_trace = state
    sampler = worker.model_runner.sampler
    source = descriptor(inspect.getsourcefile(type(sampler)))
    require(source['sha256'] == '7d3d001b8b6e1f25f02572df29609a4ab5c50abb023b48f5ebdb4752a2b5f7c5',
            'Sampler source changed; inspect before recording')

    def sampler_pre(module, args, kwargs):
        if state['active']:
            state['sample_step'] += 1
            logits = args[0] if args else kwargs['logits']
            observe(state, _store, state, 'raw_logits', logits, binary=True)
    state['handles'].append(sampler.register_forward_pre_hook(sampler_pre, with_kwargs=True))
    original_greedy = sampler.greedy_sample

    def greedy(logits):
        result = original_greedy(logits)
        if state['active']:
            def record():
                row = _store(state, 'actual_greedy_logits', logits, binary=True)
                cpu = logits.detach().to('cpu').float()
                vals, ids = cpu.topk(min(8, cpu.shape[-1]), dim=-1)
                row.update(top_token_ids=ids.tolist(), top_logits=vals.tolist(),
                           chosen_token_ids=result.detach().cpu().tolist())
            observe(state, record)
        return result
    sampler.greedy_sample = greedy
    state['sampler'] = sampler; state['original_greedy'] = original_greedy

    def vision_pre(module, args, kwargs):
        if state['active']:
            for key, tensor in tensors((args, kwargs)):
                observe(state, _store, state, 'actual_GPU_vision_input', tensor, argument=key)
    vision = network.vision_tower.patch_embedder
    state['handles'].append(vision.register_forward_pre_hook(vision_pre, with_kwargs=True))
    def vision_output(module,args,result):
        if state['active']:
            for key,tensor in tensors(result):
                observe(state,_store,state,'vision_tower_output',tensor,result_component=key)
    state['handles'].append(network.vision_tower.register_forward_hook(vision_output))
    layers = [(name, layer) for name, layer in network.named_modules()
              if re.search(r'(?:^|\.)language_model\.model\.layers\.\d+$', name)]
    require(len(layers) == 60 and {int(n.rsplit('.', 1)[-1]) for n, _ in layers} == set(range(60)),
            'Expected all 60 decoder layer observation points')
    for name, layer in layers:
        index = int(name.rsplit('.', 1)[-1])
        def before(module, args, kwargs, index=index):
            if state['active']:
                if index == 0: state['forward'] += 1
                state['layer'] = index
                if index == 0:
                    for key, tensor in tensors((args,kwargs)):
                        observe(state,_store,state,'decoder0_full_input',tensor,
                                binary=tensor.ndim==1,argument=key)
        def after(module, args, result, index=index):
            if state['active']:
                for key, tensor in tensors(result):
                    if tensor.ndim == 2:
                        observe(state, _store, state, 'decoder_last_token', tensor[-1], binary=True,
                               result_component=key, full_shape=list(tensor.shape))
                state['layer'] = None
        state['handles'].extend([layer.register_forward_pre_hook(before,with_kwargs=True), layer.register_forward_hook(after)])
    original_choose = AutoTuner.choose_one
    def choose(tuner, custom_op, runners, tuning_config, inputs, **kwargs):
        result = original_choose(tuner, custom_op, runners, tuning_config, inputs, **kwargs)
        if state['active']:
            state['records'].append({'kind': 'actual_autotune_selection',
                'forward': state['forward'], 'layer': state['layer'], 'custom_op': custom_op,
                'runner': type(result[0]).__module__ + '.' + type(result[0]).__qualname__,
                'tactic': safe(result[1]), 'input_shapes': [list(x.shape) if hasattr(x,'shape') else None for x in inputs],
                'input_dtypes': [str(x.dtype) if hasattr(x,'dtype') else None for x in inputs]})
        return result
    AutoTuner.choose_one = choose
    state['original_choose'] = original_choose
    proof = {'status': 'INSTALLED_AFTER_ORIGINAL_LOAD_AND_AUTOTUNE',
             'sampler_source': source, 'trace_implementation': descriptor(__file__),
             'decoder_layers': [name for name, _ in layers],
             'math_or_sampling_arguments_changed': False,
             'observation_overhead_expected': True,
             'weights': weight_fingerprint(network), 'autotune': cache_snapshot(),
             'torch_flags': {'float32_matmul_precision': torch.get_float32_matmul_precision(),
                 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                 'cudnn_benchmark': torch.backends.cudnn.benchmark,
                 'cudnn_deterministic': torch.backends.cudnn.deterministic,
                 'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32}}
    atomic_json(root / 'after-load.json', proof)
    return {'status': proof['status'], 'proof': descriptor(root / 'after-load.json')}


def resource_observation(p):
    import subprocess
    import os
    row={'monotonic_seconds':time.monotonic()}
    command=['nvidia-smi','-i',p['gpu_uuid'],
        '--query-gpu=uuid,name,driver_version,memory.total,memory.free,utilization.gpu,temperature.gpu,clocks.sm,clocks.mem,pstate,power.draw',
        '--format=csv,noheader,nounits']
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=10)
        row.update(command=command,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
        row['process_affinity']=sorted(os.sched_getaffinity(0))
        row['meminfo']=Path('/proc/meminfo').read_text()
    except Exception as error:row['error']=repr(error)
    return row


def begin(worker, sid):
    import torch
    s = worker._gemma_recheck_trace
    require(not s['active'], 'Previous diagnostic request still active')
    s.update(active=True, sid=sid, records=[], forward=-1, layer=None, sample_step=-1,
             started=time.monotonic(), profiler=None, profiler_error=None)
    path = s['root'] / (sid + '.zip')
    s['bundle_path'] = path
    s['bundle'] = zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1)
    try:
        profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                     torch.profiler.ProfilerActivity.CUDA])
        profiler.__enter__(); s['profiler'] = profiler
    except Exception as error:
        s['profiler_error'] = repr(error)
    return {'sample_id': sid, 'CUDA_kernel_trace_started': s['profiler'] is not None,
            'profiler_error': s['profiler_error']}


def finish(worker):
    s = worker._gemma_recheck_trace
    require(s['active'], 'No active diagnostic request')
    s['active'] = False
    profiler = s.pop('profiler', None)
    trace = None; kernels = Counter()
    if profiler is not None:
        try:
            profiler.__exit__(None, None, None)
            temporary = s['root'] / (s['sid'] + '.chrome.json')
            profiler.export_chrome_trace(str(temporary))
            with temporary.open() as handle:
                for event in json.load(handle).get('traceEvents',[]):
                    if event.get('cat')=='kernel':kernels[event.get('name','UNKNOWN')]+=1
            target = temporary.with_suffix('.json.gz')
            with temporary.open('rb') as src, gzip.open(target, 'wb', compresslevel=1) as dst:
                import shutil
                shutil.copyfileobj(src, dst, 8 * 1024**2)
            temporary.unlink(); trace = descriptor(target)
        except Exception as error:
            s['profiler_error'] = repr(error)
    proof = {'sample_id': s['sid'], 'records': s['records'],
             'autotune_after_request': cache_snapshot(), 'CUDA_kernel_trace': trace,
             'actual_CUDA_kernels':dict(kernels), 'actual_CUDA_kernel_events':sum(kernels.values()),
             'profiler_error': s['profiler_error'], 'elapsed_with_observations': time.monotonic() - s['started']}
    s['bundle'].writestr('manifest.json', json.dumps(proof, sort_keys=True))
    s['bundle'].close()
    atomic_json(s['root'] / (s['sid'] + '.json'), proof)
    return {'sample_id': s['sid'], 'proof': descriptor(s['root'] / (s['sid'] + '.json')),
            'tensor_bundle': descriptor(s['bundle_path']), 'CUDA_kernel_trace': trace,
            'profiler_error': s['profiler_error'],
            'actual_CUDA_kernel_events':sum(kernels.values()),
            'vision_chunks': sum(x['kind'] == 'actual_GPU_vision_input' for x in s['records']),
            'greedy_steps': sum(x['kind'] == 'actual_greedy_logits' for x in s['records']),
            'observation_errors': [x for x in s['records'] if x['kind']=='observation_error'],
            'GPU_pixel_frame_hashes': [h for x in s['records']
                if x['kind']=='actual_GPU_vision_input' and x['shape'][1:]==[PATCHES,768]
                for h in x.get('per_frame_sha256',[])]}


def close(worker):
    from flashinfer.autotuner import AutoTuner
    s = worker._gemma_recheck_trace
    if s['active']: finish(worker)
    atomic_json(s['root'] / 'before-exit.json', {
        'weights': weight_fingerprint(s['network']), 'autotune': cache_snapshot()})
    for handle in s['handles']: handle.remove()
    s['sampler'].greedy_sample = s['original_greedy']
    AutoTuner.choose_one = s['original_choose']
    del worker._gemma_recheck_trace
    return {'before_exit': descriptor(s['root'] / 'before-exit.json')}


class AdapterObservation:
    def __init__(self, original, p, engine, state):
        self.original, self.p, self.engine, self.state = original, p, engine, state
        self.root = Path(p['output']); self.active = False
        self.reference = verified(p['recheck']['reference'])
        self.installation = engine.collective_rpc(install, args=(p['output'],))
        atomic_json(self.root / 'diagnostic-installation.json', self.installation)
        prior = original.native_class._apply_hf_processor_main
        def observed(native, mm_items, hf_kwargs):
            result = prior(native, mm_items, hf_kwargs)
            if self.active:
                rows = []
                for name in ['pixel_values_videos', 'pixel_position_ids_videos']:
                    for key, tensor in tensors(result.get(name), name):
                        info, data = tensor_blob(tensor); del data
                        rows.append({'field': key, **info})
                self.cpu_tensors.extend(rows)
                _append(self.root / 'diagnostic-native-CPU.jsonl', {'sample_id': self.sid, 'tensors': rows})
            return result
        original.native_class._apply_hf_processor_main = observed

    def prepare(self, item, content):
        from .qwen_kv_shared_smoke import shared_observation
        self.sid = item['sample_id']; self.active = True; self.cpu_tensors = []
        _append(self.root / 'shared-gpu-observations.jsonl', shared_observation(self.p, self.sid))
        _append(self.root / 'diagnostic-resources.jsonl', {'sample_id':self.sid,**resource_observation(self.p)})
        self.engine.collective_rpc(begin, args=(self.sid,))
        result = self.original.prepare(item, content)
        frames = result['request']['multi_modal_data']['video'][0][0]
        _append(self.root / 'diagnostic-submitted-frames.jsonl', {
            'sample_id': self.sid, 'shape': list(frames.shape), 'dtype': str(frames.dtype),
            'sha256': hashlib.sha256(memoryview(frames).cast('B')).hexdigest(),
            'receipt': result['receipt']})
        return result

    def validate(self, answer):
        result = self.original.validate(answer)
        evidence = self.engine.collective_rpc(finish)
        self.active = False
        _append(self.root / 'diagnostic-requests.jsonl', {'sample_id': self.sid, 'workers': evidence})
        cpu_hashes = [h for x in self.cpu_tensors if x['shape'][1:]==[PATCHES,768]
                      for h in x.get('per_frame_sha256',[])]
        gpu_hashes = [h for x in evidence for h in x['GPU_pixel_frame_hashes']]
        byte_match = len(cpu_hashes)==180 and cpu_hashes==gpu_hashes
        _append(self.root/'diagnostic-input-equivalence.jsonl', {'sample_id':self.sid,
            'actual_CPU_GPU_per_frame_bytes_equal':byte_match, 'CPU_frames':len(cpu_hashes),
            'GPU_frames':len(gpu_hashes)})
        require(byte_match, 'Actual CPU/GPU input bytes differ; raw and diagnostics retained')
        require(all(x['vision_chunks'] > 0 and x['greedy_steps'] > 0 and not x['observation_errors'] for x in evidence),
                'Missing actual GPU input or decision logits; raw retained')
        raw = self.state['raw']; expected = self.reference['by_id'][self.sid]
        keys = ['raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason', 'truth', 'question', 'generation']
        differences = [k for k in keys if raw[k] != expected[k]]
        if raw['actual_prompt_token_ids'] != self.reference['actual_prompt_token_ids']:
            differences.append('actual_prompt_token_ids')
        equal = not differences
        row = {'sample_id': self.sid, 'full_output_prompt_equal': equal,
               'different_fields': differences, 'raw_digest': digest(raw)}
        _append(self.root / 'output-equivalence.jsonl', row)
        if not equal:
            atomic_json(self.root / 'reproduction-failure.json', {
                **row, 'raw_answer': raw['raw_answer'], 'reference_answer': expected['raw_answer'],
                'raw_persisted_before_comparison': True, 'automatic_retry': False})
        require(equal, 'REPRODUCTION_DIFFERENCE_RAW_AND_DIAGNOSTICS_RETAINED')
        return result

    def clear(self):
        self.original.clear(); self.state.pop('raw', None)

    def close(self):
        try:
            evidence = self.engine.collective_rpc(close)
            atomic_json(self.root / 'diagnostic-close.json', evidence)
        finally:
            self.original.close()


@contextmanager
def observing(p):
    from . import gemma_x_full_input as inputs
    from . import official_x_runner as runner
    state = {}; original_create = inputs.create; original_append = runner._append
    def create(profile, engine):
        return AdapterObservation(original_create(profile, engine), profile, engine, state)
    def append(path, value):
        original_append(path, value)
        if Path(path) == Path(p['output']) / 'raw.jsonl': state['raw'] = value
    inputs.create = create; runner._append = append
    try: yield
    finally:
        inputs.create = original_create; runner._append = original_append
