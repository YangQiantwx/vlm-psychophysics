import sys
from types import SimpleNamespace

import pytest

from classic_precision import nemo_canonical_restore as r
from classic_precision import inference, gemma_speed


def test_frozen_formal_partition_is_disjoint_and_complete():
    ids = [str(i) for i in range(836)]
    parts = r.split_formal(ids)
    assert [len(x) for x in parts.values()] == [418, 418]
    assert not set(parts['fourier0']) & set(parts['fourier1'])
    assert set(parts['fourier0']) | set(parts['fourier1']) == set(ids)
    with pytest.raises(ValueError): r.split_formal(ids[:-1])
    with pytest.raises(ValueError): r.split_formal(ids[:-1] + [ids[0]])


def test_actual_backend_gate_precedes_original_session_generation(monkeypatch, tmp_path):
    calls = []
    class Engine:
        def collective_rpc(self, function):
            if function is gemma_speed._worker_identity: return [{'worker': {}}]
            return [{'actual_kernel_config': dict(r.CONTRACT),
                     'classes': {'vllm.FlashInferCutlassNvFp4LinearKernel': 1}, 'backends': []}]
    def llm(**settings):
        assert settings == {'model': 'fixed', 'kernel_config': r.CONTRACT}
        calls.append('load'); return Engine()
    fake = SimpleNamespace(LLM=llm)
    monkeypatch.setitem(sys.modules, 'vllm', fake)
    monkeypatch.setattr(gemma_speed, 'shutdown_owned_engine', lambda *args: calls.append('shutdown'))
    def original(self, name):
        self.engine = fake.LLM(model='fixed')
        calls.append('would_generate')
    monkeypatch.setattr(inference.ModelSession, '__init__', original)
    with r.legacy_session(tmp_path):
        with pytest.raises(ValueError, match='Actual mixed'):
            inference.ModelSession('nemo')
    assert calls == ['load', 'shutdown']
    assert fake.LLM is llm and inference.ModelSession.__init__ is original


def test_context_preserves_original_settings_and_sampling(monkeypatch, tmp_path):
    seen = []; sampling = object()
    class Engine:
        def collective_rpc(self, function):
            if function is gemma_speed._worker_identity: return [{'worker': {}}]
            return [{'actual_kernel_config': dict(r.CONTRACT),
                     'classes': {'vllm.CutlassNvFp4LinearKernel': 1,
                                 'vllm.CutlassFP8ScaledMMLinearKernel': 1},
                     'backends': ['nvfp4_backend=NvFp4MoeBackend.VLLM_CUTLASS']}]
    def llm(**settings): seen.append(settings); return Engine()
    fake = SimpleNamespace(LLM=llm); monkeypatch.setitem(sys.modules, 'vllm', fake)
    def original(self, name, checkpoint, **kwargs):
        assert (name, checkpoint, kwargs) == ('nemo', 'frozen', {'resources': {'fixed': True}})
        self.engine = fake.LLM(model=checkpoint, temperature_unchanged='marker')
        self.sampling = sampling
    monkeypatch.setattr(inference.ModelSession, '__init__', original)
    from classic_precision import reader_route_smoke
    def receipt(value):
        assert value is sampling
        return {'status': 'same_sampling_object'}
    monkeypatch.setattr(reader_route_smoke, 'generation_receipt', receipt)
    with r.legacy_session(tmp_path): inference.ModelSession('nemo', 'frozen', resources={'fixed': True})
    assert seen == [{'model': 'frozen', 'temperature_unchanged': 'marker', 'kernel_config': r.CONTRACT}]
    assert fake.LLM is llm and inference.ModelSession.__init__ is original


def test_relocation_preserves_scientific_values():
    old = {'path': r.ROOTS[2]+'/source', 'values': [180, 768, 512], 'dtype': 'float32'}
    assert r.relocate(old, '/new') == {'path': '/new/source', 'values': [180, 768, 512], 'dtype': 'float32'}


def test_audit_rejects_configuration_drift_in_completed_repair(monkeypatch, tmp_path):
    from classic_precision import nemo_restoration_results as results
    parent={'output':'/mnt/sda/q/cvpr2027_benchmark/runs/reproducibility/smoke/old',
            'generation':{'temperature':0},'checkpoint_assets':{'frozen':True}}
    new={**parent,'output':'/mnt/sda/q/cvpr2027_benchmark/runs/reproducibility/smoke/new',
         'worker':'new','assigned_ids':['id'],'generation':{'temperature':.1},
         'restoration':{'authorization':{},'block_index':0,'preserved_claim_sha256':{},'sealed_allocation':{}}}
    monkeypatch.setattr(results,'load_bound',lambda root,ref:new if ref.get('profile') else {})
    monkeypatch.setattr(results,'preserved',lambda *args:(parent,[],{}))
    with pytest.raises(ValueError,match='Scientific|scientific'):
        results.profile_contract(tmp_path,{'profile':True},{},{'blocks':[{}]})
