"""Tests at the opt-in boundary and the actual bytes used for input matching."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "runtime_compat/classic_precision_determinism.py"
spec = importlib.util.spec_from_file_location("determinism_profile_test", HELPER)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_fingerprint_matches_unconverted_storage(dtype):
    x = torch.arange(24).reshape(4, 6).to(dtype).t()  # noncontiguous logical input
    actual = profile.tensor_fingerprint(x, "input")
    raw = x.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
    assert actual == {"name": "input", "shape": [6, 4], "dtype": str(dtype),
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def test_fingerprint_detects_one_bf16_bit_and_preserves_signed_zero():
    x = torch.tensor([0., -0., 1.], dtype=torch.bfloat16)
    y = x.clone()
    y.view(torch.int16)[0] ^= 1
    assert profile.tensor_fingerprint(x)["sha256"] != profile.tensor_fingerprint(y)["sha256"]
    assert profile.tensor_fingerprint(x)["sha256"] != profile.tensor_fingerprint(x.abs())["sha256"]


@pytest.mark.parametrize("x", [torch.tensor(1., dtype=torch.bfloat16), torch.empty(0)])
def test_fingerprint_scalar_and_empty_storage(x):
    raw = x.reshape(-1).view(torch.uint8).numpy().tobytes()
    assert profile.tensor_fingerprint(x)["sha256"] == hashlib.sha256(raw).hexdigest()


def test_module_import_is_cuda_free_and_configuration_is_explicit():
    code = f"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('profile', {str(HELPER)!r})
p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert p.status() == {{}}
try:
 p.activate('qwen')
except RuntimeError as e:
 assert 'launch_environment' in str(e)
else:
 raise AssertionError('unconfigured activation was accepted')
assert 'torch' not in sys.modules
print(json.dumps(p.launch_environment()))
"""
    env = dict(os.environ)
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    result = subprocess.run([sys.executable, "-B", "-c", code], env=env,
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_late_activation_refuses_already_imported_torch(monkeypatch):
    for key, value in profile.launch_environment().items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="before importing Torch"):
        profile.activate("cosmos")


def test_unsupported_family_is_not_silently_accepted():
    with pytest.raises(ValueError, match="Unknown model"):
        profile.activate("another-model")


@pytest.mark.parametrize("family,expected_fill", [("nemo", True), ("qwen", False),
                                                ("cosmos", True), ("gemma", True)])
def test_fill_is_set_explicitly_with_other_strict_flags_retained(monkeypatch, family, expected_fill):
    calls = []
    precision = lambda: SimpleNamespace(fp32_precision="tf32")
    fake = SimpleNamespace(
        use_deterministic_algorithms=lambda *a, **k: calls.append((a, k)),
        manual_seed=lambda seed: calls.append(("seed", seed)),
        utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=not expected_fill)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=precision()),
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, conv=precision(), rnn=precision())))
    monkeypatch.setattr(profile, "_state", {"family": family})
    profile._configure_torch(fake)
    assert fake.utils.deterministic.fill_uninitialized_memory is expected_fill
    assert calls[0] == ((True,), {"warn_only": False})
    assert fake.backends.cudnn.deterministic is True and fake.backends.cudnn.benchmark is False
    assert fake.backends.cuda.matmul.fp32_precision == "ieee"
    assert fake.backends.cudnn.conv.fp32_precision == fake.backends.cudnn.rnn.fp32_precision == "ieee"
