"""Observe official selective prefetch placement; never patch model forwards.

Import and receipt validation are CPU-only. Runtime observation is called only
inside the authorized engine, after loading and after its first budgeted request.
"""
from __future__ import annotations

from . import gemma_x_full as full
from . import official_x_runner as base
from .refinement_worker import require

NAME = 'gemma-x-prefetch-qualification-20261031'
FILTER = ['self_attn.qkv_proj.weight', 'self_attn.o_proj.weight']
GIB = 1024**3
EXPECTED_BYTES = 17175674880
EXPECTED_POOL_BYTES = 1321205760
ENGINE_DELTA = {
    'offload_backend': 'prefetch', 'cpu_offload_gb': 0,
    'cpu_offload_params': [], 'offload_group_size': 1,
    'offload_num_in_group': 1, 'offload_prefetch_step': 2,
    'offload_params': FILTER.copy(), 'kv_cache_memory_bytes': 44 * GIB,
    'gpu_memory_utilization': 0.8,
}
RESOURCES = {'min_free_gpu_gib': 77, 'min_available_ram_gib': 48}
KERNEL_COUNTS = {
    'kernel:vllm.model_executor.kernels.linear.nvfp4.flashinfer.FlashInferCutlassNvFp4LinearKernel': 120,
    'method:vllm.model_executor.layers.linear.UnquantizedLinearMethod': 312,
    'method:vllm.model_executor.layers.quantization.modelopt.ModelOptKVCacheMethod': 60,
    'method:vllm.model_executor.layers.quantization.modelopt.ModelOptLinearMethod': 120,
    'method:vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod': 1,
}
_IDENTITY = full.worker_identity
_READ_PROBE = base.read_probe


def expected_parameters():
    result = {}
    for index in range(60):
        prefix = f'language_model.model.layers.{index}.self_attn.'
        result[prefix + 'qkv_proj.weight'] = [20480 if index % 6 == 5 else 16384, 5376]
        result[prefix + 'o_proj.weight'] = [5376, 16384 if index % 6 == 5 else 8192]
    return result


def _buffer_specs():
    return {('self_attn.' + name.split('.self_attn.', 1)[1], tuple(shape),
             (shape[1], 1), 'torch.bfloat16')
            for name, shape in expected_parameters().items()}


def _cuda_memory_type(pointer):
    import ctypes
    driver = ctypes.CDLL('libcuda.so.1')
    query = driver.cuPointerGetAttribute
    query.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64)
    query.restype = ctypes.c_int
    result = ctypes.c_uint()
    status = query(ctypes.byref(result), 2, pointer)
    require(status == 0, f'CUDA pointer memory-type query failed: {status}')
    return result.value


def validate_placement(receipt, phase):
    require(phase in ('after_load', 'after_first_budgeted_request'), 'Unknown prefetch observation phase')
    expected = expected_parameters()
    require(receipt['schema'] == NAME + '/placement-v1' and receipt['phase'] == phase
            and receipt['offload_backend'] == 'official_vllm_PrefetchOffloader'
            and receipt['offloader_class'] == 'vllm.model_executor.offloader.prefetch.PrefetchOffloader'
            and receipt['group_size'] == receipt['num_in_group'] == 1
            and receipt['prefetch_step'] == receipt['slot_capacity'] == 2
            and receipt['mode'] == 'cpu' and set(receipt['offload_params']) == set(FILTER)
            and receipt['offloaded_module_count'] == 60
            and receipt['actual_offloaded_parameter_bytes'] == EXPECTED_BYTES
            and receipt['actual_gpu_static_buffer_bytes'] == EXPECTED_POOL_BYTES,
            'Actual official prefetch backend, selection or buffer budget differs')
    actual = receipt['offloaded_parameters']
    require(set(actual) == set(expected), 'Actual prefetch parameter set is incomplete')
    cpu_pointers = set()
    for name, shape in expected.items():
        entry = actual[name]
        layer = int(name.split('.layers.', 1)[1].split('.', 1)[0])
        require(entry['dtype'] == entry['cpu_dtype'] == 'torch.bfloat16'
                and entry['shape'] == entry['cpu_shape'] == shape
                and entry['stride'] == entry['cpu_stride'] == [shape[1], 1]
                and entry['bytes'] == shape[0] * shape[1] * 2
                and entry['torch_device'].startswith('cuda') and entry['cpu_device'] == 'cpu'
                and entry['cpu_is_pinned'] is True and entry['is_uva_offloaded'] is False
                and entry['module_index'] == layer and entry['buffer_slot'] == layer % 2
                and entry['parameter_pointer'] == entry['gpu_buffer_pointer']
                and entry['pool_buffer_matches'] is True and entry['parameter_object_matches'] is True
                and int(entry['cpu_pointer'], 16) > 0,
                'Actual pinned CPU storage/static GPU mapping differs: ' + name)
        cpu_pointers.add(entry['cpu_pointer'])
    buffers = receipt['static_buffers']
    specs = {(x['name'], tuple(x['shape']), tuple(x['stride']), x['dtype']) for x in buffers}
    require(len(cpu_pointers) == 120 and sum(x['bytes'] for x in actual.values()) == EXPECTED_BYTES
            and len(buffers) == 8 and specs == _buffer_specs()
            and len({x['pointer'] for x in buffers}) == 8
            and sum(x['bytes'] for x in buffers) == EXPECTED_POOL_BYTES,
            'Actual CPU storages or four-shape/two-slot GPU pool differs')
    for spec in specs:
        matches = [x for x in buffers
                   if (x['name'], tuple(x['shape']), tuple(x['stride']), x['dtype']) == spec]
        require({x['slot'] for x in matches} == {0, 1}
                and all(x['device'].startswith('cuda') and int(x['pointer'], 16) > 0
                        and x['cuda_memory_type'] == 2
                        and x['bytes'] == x['shape'][0] * x['shape'][1] * 2 for x in matches),
                'GPU buffer slots, device or physical sizes differ')
    require(receipt['method_kernel_counts'] == KERNEL_COUNTS
            and receipt['actual_kernel_counts_unchanged'] is True
            and receipt['unexpected_cpu_or_uva_parameters'] == []
            and receipt['actual_kv_capacity'] >= 65536
            and receipt['kv_cache_dtype'] == 'bfloat16'
            and receipt['kv_cache_memory_bytes'] == 44 * GIB
            and receipt['max_model_len'] == 65536
            and receipt['max_num_batched_tokens'] == 49152 and receipt['max_num_seqs'] == 1,
            'Qualified mixed NVFP4 kernels, BF16 KV or context/input capacity changed')
    require(receipt['gpu_free_plus_torch_reusable_bytes'] >= GIB
            and receipt['available_ram_bytes'] >= 20 * GIB,
            'Actual prefetch placement has insufficient GPU/RAM margin')


def worker_identity(worker, model):
    """Read actual offloader storage, without adding a forward/math hook."""
    import torch
    from vllm.model_executor.offloader import get_offloader
    from vllm.model_executor.offloader.prefetch import PrefetchOffloader
    from .inference import available_ram_bytes

    value = _IDENTITY(worker, model)
    phase = ('after_first_budgeted_request'
             if getattr(worker, '_prefetch_placement_seen', False) else 'after_load')
    offloader = get_offloader()
    require(type(offloader) is PrefetchOffloader and offloader.buffer_pool is not None,
            'Actual engine did not construct the official PrefetchOffloader/static pool')
    pool = offloader.buffer_pool
    expected = expected_parameters()
    parameters = dict(worker.get_model().named_parameters())
    modules = dict(worker.get_model().named_modules())
    actual = {}
    unexpected = [name for name, p in parameters.items()
                  if p.device.type == 'cpu' or getattr(p, '_vllm_is_uva_offloaded', False)]
    for index, module_offloader in enumerate(offloader.module_offloaders):
        prefix = f'language_model.model.layers.{index}'
        require(module_offloader.module is modules.get(prefix)
                and module_offloader.layer_idx == index
                and module_offloader._buffer_pool is pool
                and module_offloader._buffer_slot_idx == index % 2
                and set(module_offloader._param_offloaders) == set(FILTER),
                'Actual decoder/module offloader binding differs: ' + prefix)
        for suffix, param_offloader in module_offloader._param_offloaders.items():
            name = prefix + '.' + suffix
            require(name in expected, 'Unexpected prefetch parameter: ' + name)
            parameter = parameters[name]
            cpu = param_offloader._cpu_storage
            gpu = param_offloader._gpu_buffer
            require(cpu is not None and gpu is not None, 'Missing actual CPU/GPU backing: ' + name)
            key = (suffix, tuple(cpu.shape), tuple(cpu.stride()), cpu.dtype)
            buffers = pool._buffers.get(key, [])
            actual[name] = {
                'dtype': str(parameter.dtype), 'shape': list(parameter.shape),
                'stride': list(parameter.stride()), 'bytes': parameter.numel() * parameter.element_size(),
                'torch_device': str(parameter.device),
                'is_uva_offloaded': bool(getattr(parameter, '_vllm_is_uva_offloaded', False)),
                'cpu_dtype': str(cpu.dtype), 'cpu_shape': list(cpu.shape), 'cpu_stride': list(cpu.stride()),
                'cpu_device': str(cpu.device), 'cpu_is_pinned': cpu.is_pinned(),
                'cpu_pointer': hex(cpu.data_ptr()), 'parameter_pointer': hex(parameter.data_ptr()),
                'gpu_buffer_pointer': hex(gpu.data_ptr()), 'module_index': index, 'buffer_slot': index % 2,
                'pool_buffer_matches': len(buffers) == 2 and gpu is buffers[index % 2],
                'parameter_object_matches': param_offloader._param is parameter,
            }
    static_buffers = []
    for key, buffers in pool._buffers.items():
        for slot, tensor in enumerate(buffers):
            static_buffers.append({
                'name': key[0], 'shape': list(tensor.shape), 'stride': list(tensor.stride()),
                'dtype': str(tensor.dtype), 'device': str(tensor.device), 'slot': slot,
                'pointer': hex(tensor.data_ptr()), 'bytes': tensor.numel() * tensor.element_size(),
                'cuda_memory_type': _cuda_memory_type(tensor.data_ptr()),
            })
    require(pool.total_bytes == sum(x['bytes'] for x in static_buffers), 'GPU buffer pool accounting differs')
    require(offloader.total_offloaded_bytes == sum(x['bytes'] for x in actual.values()),
            'Official prefetch CPU-storage byte accounting differs')
    free, total = torch.cuda.mem_get_info()
    allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
    config = worker.vllm_config
    receipt = {
        'schema': NAME + '/placement-v1', 'phase': phase,
        'offload_backend': 'official_vllm_PrefetchOffloader',
        'offloader_class': type(offloader).__module__ + '.' + type(offloader).__name__,
        'group_size': offloader.group_size, 'num_in_group': offloader.num_in_group,
        'prefetch_step': offloader.prefetch_step, 'offload_params': sorted(offloader.offload_params),
        'mode': offloader.mode, 'slot_capacity': pool.slot_capacity,
        'offloaded_module_count': len(offloader.module_offloaders),
        'actual_offloaded_parameter_bytes': offloader.total_offloaded_bytes,
        'actual_gpu_static_buffer_bytes': pool.total_bytes,
        'offloaded_parameters': actual, 'static_buffers': static_buffers,
        'unexpected_cpu_or_uva_parameters': unexpected,
        'method_kernel_counts': value['method_kernel_counts'],
        'actual_kernel_counts_unchanged': value['method_kernel_counts'] == KERNEL_COUNTS,
        'actual_kv_capacity': value['actual_kv_capacity'], 'kv_concurrency': value['kv_concurrency'],
        'kv_num_blocks': worker.model_runner.kv_cache_config.num_blocks,
        'kv_group_count': len(worker.model_runner.kv_cache_config.kv_cache_groups),
        'kv_cache_dtype': config.cache_config.cache_dtype,
        'kv_cache_memory_bytes': config.cache_config.kv_cache_memory_bytes,
        'max_model_len': config.model_config.max_model_len,
        'max_num_batched_tokens': config.scheduler_config.max_num_batched_tokens,
        'max_num_seqs': config.scheduler_config.max_num_seqs,
        'gpu_free_bytes': free, 'gpu_total_bytes': total,
        'torch_allocated_bytes': allocated, 'torch_reserved_bytes': reserved,
        'torch_reusable_cached_bytes': reserved - allocated,
        'gpu_free_plus_torch_reusable_bytes': free + reserved - allocated,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'available_ram_bytes': available_ram_bytes(),
        'minimum_effective_gpu_headroom_bytes': GIB, 'minimum_remaining_ram_bytes': 20 * GIB,
        'scientific_input_or_generation_changed': False,
    }
    validate_placement(receipt, phase)
    value['prefetch_placement'] = receipt
    value['ram_placement'] = receipt
    if phase == 'after_load':
        worker._prefetch_placement_seen = True
        torch.cuda.reset_peak_memory_stats()
    return value


def read_probe(worker, model):
    value = _READ_PROBE(worker, model)
    runtime = value.get('runtime', {})
    receipt = runtime.get('prefetch_placement', {})
    if receipt.get('phase') != 'after_first_budgeted_request':
        runtime = worker_identity(worker, model)
    value['runtime'] = runtime
    value.pop('runtime_error', None)
    return value
