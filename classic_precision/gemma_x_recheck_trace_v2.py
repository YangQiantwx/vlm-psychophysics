"""Observe the installed vLLM V2 sampler without replacing its native decision.

The original recorder targeted the legacy nn.Module sampler. This version binds
both actual V2 source files, records the original native kernel arguments/results,
and delegates every computation exactly once with unchanged arguments.
"""
import inspect
from pathlib import Path
import re
from . import gemma_x_recheck_trace as base
from .gemma_x_recheck_trace import (require, descriptor, atomic_json, observe,
    _store, tensors, safe, weight_fingerprint, cache_snapshot)

SAMPLER_SHA = '7f9941f9d7d04a7481ec393925ed5d000d4d77ee6db2ca6a8d9547c211d00a09'
GUMBEL_SHA = '12f51f9151fe53a04901990391aa5b2443318ba411fa59ff6f306da7232e7c99'


def validate_sources(sampler_class, sampler_source, gumbel_source):
    require(sampler_class == 'vllm.v1.worker.gpu.sample.sampler.Sampler'
            and sampler_source['sha256'] == SAMPLER_SHA
            and gumbel_source['sha256'] == GUMBEL_SHA,
            'V2 sampler source differs from inspected installed SDK')


def attach_sampler(state, sampler, module, capture):
    original_sample, original_gumbel = sampler.sample, module.gumbel_sample
    def sample(*args, **kwargs):
        if state['active']:
            state['sample_step'] += 1
            observe(state, capture, 'raw_sampler_logits', args[0], None, args, kwargs)
        return original_sample(*args, **kwargs)
    def gumbel(*args, **kwargs):
        # Original function does not mutate logits; still copy the true input
        # before executing the native kernel so provenance is unambiguous.
        row = None
        if state['active']:
            row = observe(state, capture, 'actual_greedy_logits', args[0], None, args, kwargs)
        result = original_gumbel(*args, **kwargs)
        if state['active']:
            observe(state, capture, 'actual_greedy_result', result, row, args, kwargs)
        return result
    sampler.sample, module.gumbel_sample = sample, gumbel
    def restore():
        sampler.sample, module.gumbel_sample = original_sample, original_gumbel
    state['restore_sampler'] = restore


def install(worker, output):
    import torch
    from flashinfer.autotuner import AutoTuner
    from vllm.v1.worker.gpu.sample import sampler as sampler_module
    network = worker.get_model()
    require(not hasattr(worker, '_gemma_recheck_trace'), 'Trace already installed')
    sampler = worker.model_runner.sampler
    source = descriptor(inspect.getsourcefile(type(sampler)))
    gumbel_source = descriptor(inspect.getsourcefile(sampler_module.gumbel_sample))
    validate_sources(type(sampler).__module__ + '.' + type(sampler).__qualname__, source, gumbel_source)
    root = Path(output) / 'diagnostics'; root.mkdir()
    state = {'active': False, 'root': root, 'handles': [], 'network': network,
             'records': [], 'forward': -1, 'layer': None, 'sample_step': -1}
    worker._gemma_recheck_trace = state
    def capture(kind, tensor, row, args, kwargs):
        if kind == 'actual_greedy_result':
            tokens = tensor.detach().to('cpu').tolist()
            require(row is not None, 'Native greedy input recording missing')
            row['chosen_token_ids'] = tokens
            return row
        row = _store(state, kind, tensor, binary=True,
                     sampler_path='vLLM V2 original sample/gumbel_sample')
        if kind == 'actual_greedy_logits':
            idx = args[1].detach().to('cpu').long()
            temperatures = args[2].detach().to('cpu')[idx].tolist()
            row.update(temperatures=temperatures,
                       seeds=args[3].detach().to('cpu')[idx].tolist(),
                       positions=args[4].detach().to('cpu').tolist(),
                       native_kernel_kwargs=safe(kwargs))
            require(all(t == 0 for t in temperatures), 'Expected original temperature-zero greedy branch')
            values, indices = tensor.detach().float().to('cpu').topk(8, dim=-1)
            row.update(top8_values=values.tolist(), top8_ids=indices.tolist())
        return row
    attach_sampler(state, sampler, sampler_module, capture)
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
             'sampler_source': source, 'gumbel_source': gumbel_source,
             'sampler_path': 'V2 native gumbel_sample; temperature-zero greedy branch', 'trace_implementation': descriptor(__file__),
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


def close(worker):
    from flashinfer.autotuner import AutoTuner
    s = worker._gemma_recheck_trace
    if s['active']: base.finish(worker)
    atomic_json(s['root'] / 'before-exit.json', {
        'weights': weight_fingerprint(s['network']), 'autotune': cache_snapshot()})
    for handle in s['handles']: handle.remove()
    s['restore_sampler']()
    AutoTuner.choose_one = s['original_choose']
    del worker._gemma_recheck_trace
    return {'before_exit': descriptor(s['root'] / 'before-exit.json')}
