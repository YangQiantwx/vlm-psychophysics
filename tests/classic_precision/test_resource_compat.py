"""Resource opt-in must preserve the pinned scientific loader and its receipts."""
import ast
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from classic_precision import inference
from runtime_compat import classic_precision_vllm_resource_compat as compat


ROOT = Path(__file__).resolve().parents[2]


def _process(tmp_path, code, value=None, kv=False, package_root=ROOT):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT / "runtime_compat"), str(package_root)])}
    env.pop(compat.ENVIRONMENT_VARIABLE, None)
    env.pop("CLASSIC_PRECISION_KV_COMPAT", None)
    if value is not None:
        env[compat.ENVIRONMENT_VARIABLE] = value
    if kv:
        env["CLASSIC_PRECISION_KV_COMPAT"] = "1"
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def _exercise_loader(monkeypatch, loader, family):
    """Use stdlib stand-ins to observe every setting without Torch or vLLM."""
    class Processor:
        def __call__(self, *args, **kwargs):
            return {}

    class LLM:
        def __init__(self, **settings):
            self.settings = deepcopy(settings)

        def collective_rpc(self, callback):
            if callback is inference._kv_capacity:
                return [{"capacity_tokens": 65536, "fixture": "native_kv"}]
            assert callback is inference._weight_receipt
            return [{"torch.float8_e4m3fn": 1234}]

    class SamplingParams(SimpleNamespace):
        pass

    class StructuredOutputsParams(SimpleNamespace):
        pass

    vllm = ModuleType("vllm")
    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    sampling = ModuleType("vllm.sampling_params")
    sampling.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", sampling)
    events = []
    session = inference.ModelSession.__new__(inference.ModelSession)
    session.family, session.model_path, session.resources = family, Path("/fixture/model"), {}
    session.torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (90 * 2**30, 96 * 2**30)))
    session.store = SimpleNamespace(event=lambda name, **fields: events.append((name, fields)))
    session.processor = Processor()
    session.native = SimpleNamespace(NanoNemotronVLProcessor=Processor)
    session.context, session._restore = {"active": False}, []
    loader(session)
    return session.engine.settings, session.effective_generation, events


def test_resource_helper_import_and_absent_opt_in_are_inert(monkeypatch):
    monkeypatch.delenv(compat.ENVIRONMENT_VARIABLE, raising=False)
    original = inference.ModelSession._load_vllm
    capacity = inference._kv_capacity
    assert compat.install() is None
    assert inference.ModelSession._load_vllm is original
    assert inference._kv_capacity is capacity
    assert hashlib.sha256(Path(inference.__file__).read_bytes()).hexdigest() == compat.BASE_RUNNER_SHA256


@pytest.mark.parametrize("value", [12, 16, 24])
def test_resource_override_changes_only_qwen_literal_and_reports_effective_settings(monkeypatch, value):
    original = inference.ModelSession._load_vllm
    capacity = inference._kv_capacity
    monkeypatch.setattr(inference.ModelSession, "_load_vllm", original)
    module_before, class_before = vars(inference).copy(), vars(inference.ModelSession).copy()
    baseline, generation, base_events = _exercise_loader(monkeypatch, original, "qwen")
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, str(value))
    evidence = compat.install()
    installed = inference.ModelSession._load_vllm
    actual, actual_generation, events = _exercise_loader(monkeypatch, installed, "qwen")
    expected = {**baseline, "cpu_offload_gb": value}
    assert actual == expected
    assert actual["cpu_offload_params"] == {"experts"}
    assert actual["engram_config"] == {"cpu_offload": True}
    assert actual_generation == generation
    expected_event = deepcopy(base_events[0])
    expected_event[1]["settings"]["cpu_offload_gb"] = value
    assert events[0] == expected_event
    assert events[0][0] == "MODEL_LOADED"
    assert events[0][1]["settings"]["cpu_offload_gb"] == value
    assert events[1] == ("RUNTIME_RESOURCE_OVERRIDE", evidence)
    assert len(events) == 2
    assert evidence["base_cpu_offload_gb"] == 24
    assert evidence["requested_cpu_offload_gb"] == evidence["effective_cpu_offload_gb"] == value
    assert evidence["base_runner_sha256"] == compat.BASE_RUNNER_SHA256
    assert evidence["helper_source_sha256"] == hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest()
    assert inference._kv_capacity is capacity
    assert vars(inference).keys() == module_before.keys()
    assert all(vars(inference)[key] is item for key, item in module_before.items())
    assert [key for key, item in class_before.items() if vars(inference.ModelSession)[key] is not item] == ["_load_vllm"]
    assert compat.install() == evidence
    assert inference.ModelSession._load_vllm is installed
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, "12" if value != 12 else "16")
    with pytest.raises(RuntimeError, match="cannot change within a process"):
        compat.install()


def test_resource_override_leaves_nemo_loader_and_events_identical(monkeypatch):
    original = inference.ModelSession._load_vllm
    monkeypatch.setattr(inference.ModelSession, "_load_vllm", original)
    baseline = _exercise_loader(monkeypatch, original, "nemo")
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, "16")
    compat.install()
    assert _exercise_loader(monkeypatch, inference.ModelSession._load_vllm, "nemo") == baseline


def test_resource_ast_evidence_is_exact_single_constant_change():
    source = ast.parse(Path(inference.__file__).read_text())
    model = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ModelSession")
    original = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_load_vllm")
    expected = deepcopy(original)
    keyword, = [node for node in ast.walk(expected) if isinstance(node, ast.keyword) and node.arg == "cpu_offload_gb"]
    assert keyword.value.value == 24
    keyword.value.value = 16
    _, evidence = compat._loader(inference, 16)
    for key, tree in (("base_loader_ast_sha256", original), ("effective_loader_ast_sha256", expected)):
        assert evidence[key] == hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


@pytest.mark.parametrize("value", ["", "0", "8", "16.0", "016", " 16", "16 ", "-16", "True"])
def test_resource_process_rejects_invalid_values_before_main(tmp_path, value):
    result = _process(tmp_path, "print('SHOULD_NOT_RUN')", value)
    assert result.returncode == 78
    assert "must be exactly 12, 16, or 24" in result.stderr
    assert "SHOULD_NOT_RUN" not in result.stdout


@pytest.mark.parametrize("value,kv", [(None, False), ("12", False), ("16", False), ("24", False), ("16", True)])
def test_resource_process_opt_in_is_independent_and_gpu_import_free(tmp_path, value, kv):
    code = f"""
import sys
from classic_precision import inference
method = inference.ModelSession._load_vllm
assert hasattr(method, '_classic_precision_resource_override') == {value is not None!r}
assert ('classic_precision_vllm_resource_compat' in sys.modules) == {value is not None!r}
assert inference._kv_capacity.__module__ == {'classic_precision_vllm_kv_compat' if kv else 'classic_precision.inference'!r}
assert 'vllm' not in sys.modules and 'torch' not in sys.modules
"""
    result = _process(tmp_path, code, value, kv)
    assert result.returncode == 0, result.stderr


def test_resource_process_refuses_base_source_drift_before_import(tmp_path):
    package = tmp_path / "classic_precision"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "inference.py").write_text("raise AssertionError('UNVERIFIED_IMPORT')\n")
    result = _process(tmp_path, "print('SHOULD_NOT_RUN')", "16", package_root=tmp_path)
    assert result.returncode == 78
    assert "base runner SHA mismatch" in result.stderr
    assert "UNVERIFIED_IMPORT" not in result.stderr
    assert "SHOULD_NOT_RUN" not in result.stdout
