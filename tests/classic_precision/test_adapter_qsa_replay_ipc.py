"""Real CPU spawn reproduces the native dynamic-module RPC transport failure."""
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import cloudpickle
import pytest

from classic_precision import adapter_qsa_replay_ipc as ipc

B = Path('/Users/qianyang/Desktop/cvpr2027_benchmark')
T1 = B/'runs/t1-rgb16/evaluate.py'
T4 = B/'runs/t4-rgb16/endpoint_evaluate.py'


def _worker(payload, output, kind):
    """Fresh interpreter has no _adapter_replay_* modules or parent aliases."""
    try:
        operations = cloudpickle.loads(payload)
        if kind == 'set_deep':
            worker = SimpleNamespace()
            result = operations(worker, True)
            output.put({'result': result, 'deep': worker._t1_deep})
            return
        import torch
        class Hook:
            def __init__(self): self.removed = False
            def remove(self): self.removed = True
        class Target:
            def register_forward_pre_hook(self, callback, with_kwargs=False):
                self.callback, self.with_kwargs, self.handle = callback, with_kwargs, Hook()
                return self.handle
        visual, projection = Target(), Target()
        visual.patch_embed = SimpleNamespace(proj=projection)
        model = SimpleNamespace(visual=visual)
        worker = SimpleNamespace(get_model=lambda: model, _t1_deep=True)
        install, read = operations
        result = install(worker) if kind == 'strong_probe' else install(worker, True)
        pixels = torch.arange(6, dtype=torch.float32).reshape(2, 3).to(torch.bfloat16)
        visual.callback(visual, (pixels,), {'grid_thw': torch.tensor([[1, 1, 2]])})
        if kind == 'strong_probe': projection.callback(projection, (pixels,))
        receipts = read(worker)
        output.put({'result': result, 'receipts': receipts, 'hook_removed': visual.handle.removed,
                    'CUDA_initialized': torch.cuda.is_initialized()})
    except BaseException as error:
        output.put({'error': type(error).__name__, 'message': str(error)})


def spawned(payload, kind):
    context = multiprocessing.get_context('spawn')
    channel = context.Queue()
    child = context.Process(target=_worker, args=(payload, channel, kind))
    child.start()
    try:
        result = channel.get(timeout=30)
        child.join(timeout=30)
        assert child.exitcode == 0
        return result
    finally:
        if child.is_alive(): child.terminate(); child.join(timeout=10)  # CPU fixture only
        channel.close(); channel.join_thread()


@pytest.fixture
def loaded(monkeypatch):
    modules = []
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
        return module
    yield load
    for module in modules:
        if module.__name__ in cloudpickle.list_registry_pickle_by_value(): cloudpickle.unregister_pickle_by_value(module)


def test_original_set_deep_by_reference_fails_in_real_spawn(loaded):
    module = loaded('_adapter_replay_video', T1)
    result = spawned(cloudpickle.dumps(module._set_deep), 'set_deep')
    assert result['error'] == 'ModuleNotFoundError' and '_adapter_replay_video' in result['message']


def test_original_set_deep_by_value_succeeds_without_function_change(loaded):
    module = loaded('_adapter_replay_video', T1)
    callback, code = module._set_deep, module._set_deep.__code__
    ipc.register_native_module(module, ipc.file_sha256(T1))
    assert module._set_deep is callback and callback.__code__ is code
    assert spawned(cloudpickle.dumps(callback), 'set_deep') == {'result': True, 'deep': True}


def test_original_native_strong_probe_roundtrip_in_spawn(loaded):
    module = loaded('_adapter_replay_video', T1)
    ipc.register_native_module(module, ipc.file_sha256(T1))
    result = spawned(cloudpickle.dumps((module._install_strong_probe, module._read_strong_probe)), 'strong_probe')
    assert result['result'] is True and result['hook_removed'] and result['CUDA_initialized'] is False
    record = result['receipts'][0]
    assert record['shape'] == [2, 3] and record['dtype'] == 'torch.bfloat16'
    assert record['t1_hashes']['deep'] is True and len(record['t1_hashes']['input_sha256']) == 64
    assert record['t1_hashes']['input_sha256'] == record['t1_hashes']['projection_sha256']


def test_nested_reviewed_t4_helper_callbacks_cross_spawn(loaded):
    helper = loaded('reviewed_t4_endpoint', T4)
    module = loaded('_adapter_replay_multi', B/'runs/t1-rgb16/static/evaluate.py')
    # The native source imports another exact helper instance without sys.modules.
    sys.modules.pop('reviewed_t4_endpoint')
    records = ipc.register_native_module(module, ipc.file_sha256(module.__file__))
    assert records[-1] == {'module': 'reviewed_t4_endpoint', 'source_sha256': module.T4_SHA}
    result = spawned(cloudpickle.dumps((module.t4.install_probe, module.t4.read_probe)), 'pair_probe')
    assert result['result'] is True and result['hook_removed'] and result['CUDA_initialized'] is False
    assert len(result['receipts'][0]['tensor_sha256']) == 64
    cloudpickle.unregister_pickle_by_value(module.t4)
    sys.modules['reviewed_t4_endpoint'] = helper


def test_module_source_drift_and_name_collision_are_rejected(loaded, monkeypatch):
    module = loaded('_adapter_replay_video', T1)
    with pytest.raises(RuntimeError, match='Unqualified'): ipc.register_native_module(module, '0' * 64)
    monkeypatch.setitem(sys.modules, module.__name__, ModuleType(module.__name__))
    with pytest.raises(RuntimeError, match='another instance'): ipc.register_native_module(module, ipc.file_sha256(T1))


def test_new_plan_identity_is_private_and_preserves_native_qsa_policy():
    from classic_precision import adapter_qsa_replay as old, adapter_replay as native
    loader, old_impl = native._load, old.implementation()
    entry = ipc._entry()
    assert entry is not old and entry.native is not native and native._load is loader
    assert old.implementation() == old_impl and old.SCHEMA != ipc.SCHEMA
    source = json.loads((B/'runs/reproducibility/operator/adapter-qsa-fourier-plan.json').read_text())
    baselines = {r: source['profiles'][r]['baseline_environment'] for r in source['routes']}
    plan = ipc.build_plan(source, baselines, source['routes'], '/CPU-fixture/new-wave/STOP.json')
    assert plan['schema'] == ipc.SCHEMA and plan['implementation']['entry_sha256'] == ipc.file_sha256(ipc.__file__)
    assert plan['implementation']['parent_entry_sha256'] == ipc.PARENT_ENTRY_SHA256
    assert plan['implementation']['selector_sha256'] == old_impl['selector_sha256']
    assert all(p['torch_policy'] == {} for p in plan['profiles'].values())
    assert plan['max_model_calls'] == 40 and plan['automatic_retry'] is False
