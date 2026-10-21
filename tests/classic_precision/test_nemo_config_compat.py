"""Only vLLM's exact native dummy probe bypasses the original Nemo geometry."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from classic_precision import inference
from runtime_compat import classic_precision_nemo_config_compat as compat


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def native(monkeypatch):
    module = ModuleType("transformers.configuration_utils")
    class PreTrainedConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
    module.PreTrainedConfig = PreTrainedConfig
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return PreTrainedConfig


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(inference, "_nemo_geometry", inference._nemo_geometry)
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, "1")
    return compat.install()


def _process(tmp_path, code, value=None, others=False, package_root=ROOT):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT / "runtime_compat"), str(package_root)])}
    for name in (compat.ENVIRONMENT_VARIABLE, "CLASSIC_PRECISION_KV_COMPAT",
                 "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB", "CLASSIC_PRECISION_DEFER_MODELS"):
        env.pop(name, None)
    if value is not None:
        env[compat.ENVIRONMENT_VARIABLE] = value
    if others:
        env.update(CLASSIC_PRECISION_KV_COMPAT="1", CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB="16",
                   CLASSIC_PRECISION_DEFER_MODELS="gemma")
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def test_absent_flag_is_inert_and_original_dummy_fails(monkeypatch, native, tmp_path):
    monkeypatch.delenv(compat.ENVIRONMENT_VARIABLE, raising=False)
    original = inference._nemo_geometry
    assert compat.install() is None and inference._nemo_geometry is original
    with pytest.raises(AttributeError, match="vision_config"):
        original(native(architectures=[""], model_type="dummy_nemotron_h_omni"))
    result = _process(tmp_path, """
import sys
assert 'classic_precision_nemo_config_compat' not in sys.modules
from classic_precision import inference
assert not hasattr(inference._nemo_geometry, '_classic_precision_nemo_config_override')
""")
    assert result.returncode == 0, result.stderr


def test_exact_native_dummy_is_returned_unchanged_and_install_is_idempotent(installed, native):
    dummy = native(architectures=[""], model_type="dummy_nemotron_h_omni")
    before = deepcopy(vars(dummy))
    assert inference._nemo_geometry(dummy) is dummy
    assert vars(dummy) == before
    geometry = inference._nemo_geometry
    assert compat.install() == installed and inference._nemo_geometry is geometry


@pytest.mark.parametrize("vision", [{"other": "preserved"}, SimpleNamespace(other="preserved")])
def test_real_geometry_is_identical_to_original(installed, native, vision):
    original = inference._nemo_geometry.__wrapped__
    expected = native(architectures=["NemotronH_Omni_Reasoning_V3"],
                      model_type="nemotron_h_omni", vision_config=deepcopy(vision))
    actual = deepcopy(expected)
    assert original(expected) is expected
    assert inference._nemo_geometry(actual) is actual
    assert vars(actual) == vars(expected)
    values = actual.vision_config if isinstance(vision, dict) else vars(actual.vision_config)
    assert values == {"other": "preserved", "video_target_num_patches": 1536,
                      "video_maintain_aspect_ratio": True}


@pytest.mark.parametrize("model_type,architectures", [
    ("nemotron_h_omni", [""]), ("dummy_other_model", [""]),
    ("dummy_nemotron_h_omni", ["NemotronH_Omni_Reasoning_V3"]),
    ("dummy_nemotron_h_omni", None),
])
def test_malformed_real_or_other_dummy_is_not_silently_accepted(installed, native, model_type, architectures):
    with pytest.raises(AttributeError, match="vision_config"):
        inference._nemo_geometry(native(model_type=model_type, architectures=architectures))


def test_native_class_identity_and_absent_vision_are_required(installed, native):
    fields = {"model_type": "dummy_nemotron_h_omni", "architectures": [""]}
    class Derived(native):
        pass
    for obj in (SimpleNamespace(**fields), Derived(**fields)):
        with pytest.raises(AttributeError, match="vision_config"):
            inference._nemo_geometry(obj)
    obj = native(**fields, vision_config={})
    assert inference._nemo_geometry(obj).vision_config == {
        "video_target_num_patches": 1536, "video_maintain_aspect_ratio": True}


@pytest.mark.parametrize("others", [False, True])
def test_process_opt_in_is_gpu_import_free_and_coexists_with_other_hooks(tmp_path, others):
    code = f"""
import sys
from classic_precision import inference, scheduler
assert hasattr(inference._nemo_geometry, '_classic_precision_nemo_config_override')
assert hasattr(scheduler.choose_model, '_classic_precision_queue_override') == {others!r}
assert hasattr(inference.ModelSession._load_vllm, '_classic_precision_resource_override') == {others!r}
assert inference._kv_capacity.__module__ == {'classic_precision_vllm_kv_compat' if others else 'classic_precision.inference'!r}
assert not any(name in sys.modules for name in ('torch', 'vllm', 'transformers'))
"""
    result = _process(tmp_path, code, "1", others)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ["", "0", "true", "01", "1 "])
def test_invalid_flag_refuses_before_main(tmp_path, value):
    result = _process(tmp_path, "print('SHOULD_NOT_RUN')", value)
    assert result.returncode == 78 and "must be exactly 1" in result.stderr
    assert "SHOULD_NOT_RUN" not in result.stdout


def test_base_source_drift_refuses_before_import(tmp_path):
    package = tmp_path / "classic_precision"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "inference.py").write_text("raise AssertionError('UNVERIFIED_IMPORT')\n")
    result = _process(tmp_path, "print('SHOULD_NOT_RUN')", "1", package_root=tmp_path)
    assert result.returncode == 78 and "base runner SHA mismatch" in result.stderr
    assert "UNVERIFIED_IMPORT" not in result.stderr and "SHOULD_NOT_RUN" not in result.stdout


def test_only_geometry_is_patched_and_runner_bytes_stay_pinned(installed):
    assert hashlib.sha256(Path(inference.__file__).read_bytes()).hexdigest() == compat.BASE_RUNNER_SHA256
    assert installed["helper_source_sha256"] == hashlib.sha256(Path(compat.__file__).read_bytes()).hexdigest()
    assert inference._nemo_geometry.__wrapped__.__module__ == "classic_precision.inference"
