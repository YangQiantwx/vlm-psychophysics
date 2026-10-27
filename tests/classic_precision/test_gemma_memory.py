"""Memory-overlay seams; reuse the frozen worker's already-tested pipeline."""
from argparse import Namespace
from contextlib import ExitStack
from copy import deepcopy
import fcntl
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import gemma_memory as m


def baseline():
    checkpoint = "/fixture/" + m.g.REVISION
    return {"schema": m.g.SCHEMA, "name": "fixture-memory28",
        "runtime": {"checkpoint": checkpoint}, "variant": "kernel",
        "engine_settings": m._BASE_ENGINE_SETTINGS(checkpoint),
        "implementation": {"gemma_speed": m.BASE_SHA, "inference": m.INFERENCE_SHA},
        "maximum_model_calls": 20, "first_deep_calls": 2, "automatic_retry": False,
        "panel": {"items": [{"sample_id": f"fixture-{i}"} for i in range(20)]},
        "kernel": {"range_lookup": "binary", "prefill_block_m": 32,
                   "expected_native_sha256": m.g.NATIVE_SHA}}


def memory_profile():
    original = baseline()
    profile = deepcopy(original)
    profile["engine_settings"] = m.engine_settings(profile["runtime"]["checkpoint"])
    profile["execution_overlay"] = m.overlay_record(original)
    return profile


def test_only_two_engine_keys_change_and_all_scientific_settings_survive():
    original = m._BASE_ENGINE_SETTINGS("checkpoint")
    memory = m.engine_settings("checkpoint")
    changed = {k for k in original if original[k] != memory[k]}
    assert set(original) == set(memory)
    assert changed == {"cpu_offload_gb", "gpu_memory_utilization"}
    assert memory["cpu_offload_gb"] == 28 and memory["gpu_memory_utilization"] == .80
    assert memory["cpu_offload_params"] == ["gate_up_proj"]
    assert memory["dtype"] == memory["kv_cache_dtype"] == "bfloat16"
    assert memory["kv_cache_memory_bytes"] == 32 * 2**30
    assert memory["max_model_len"] == 65536
    assert memory["limit_mm_per_prompt"]["video"]["num_frames"] == 180
    assert m.RESOURCE_BUDGETS == {"min_free_gpu_gib": 78, "min_available_ram_gib": 64}


def test_profile_keeps_truthful_base_identity_and_binds_overlay():
    profile = memory_profile()
    assert m.validate_profile(profile) is profile
    assert profile["implementation"] == {"gemma_speed": m.BASE_SHA, "inference": m.INFERENCE_SHA}
    overlay = profile["execution_overlay"]
    assert overlay["source_sha256"] == m.g.file_sha256(m.__file__)
    assert overlay["baseline_profile_sha256"] == m.g.digest(baseline())
    assert overlay["reference_file_sha256"] == m.REFERENCE_SHA[20]
    assert overlay["rebound_parent_functions"] == ["engine_settings", "physical_lease"]


def test_wrong_overlay_or_base_source_sha_rejected(monkeypatch):
    profile = memory_profile()
    profile["execution_overlay"]["source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="overlay source SHA"):
        m.validate_profile(profile)
    monkeypatch.setattr(m, "BASE_SHA", "wrong")
    with pytest.raises(ValueError, match="Frozen Gemma/inference source differs"):
        with m.installed_overlay():
            pytest.fail("An overlay must not be installed against wrong base bytes")


def test_no_precision_context_kv_or_offload_scope_drift_is_accepted():
    for key, value in {"dtype": "float16", "kv_cache_dtype": "fp8", "max_model_len": 32768,
        "kv_cache_memory_bytes": 16 * 2**30, "cpu_offload_params": ["down_proj"], "seed": 1}.items():
        profile = memory_profile()
        profile["engine_settings"][key] = value
        with pytest.raises(ValueError, match="two authorized placement keys"):
            m.validate_profile(profile)
    profile = memory_profile()
    profile["execution_overlay"]["resource_budgets"]["min_free_gpu_gib"] = 1
    with pytest.raises(ValueError, match="declaration"):
        m.validate_profile(profile)


def test_seal_uses_temporary_baseline_and_never_rewrites_existing_profile(tmp_path, monkeypatch):
    called = []
    target = tmp_path / "memory28.json"
    def original_seal(args):
        called.append(Path(args.output))
        assert Path(args.output) != target
        assert args.panel_key == "panel"
        m.g.seal(args.output, baseline())
    monkeypatch.setattr(m.g, "seal_profile", original_seal)
    result = m.seal_profile(Namespace(output=str(target), panel="normalized48.json"))
    assert result["status"] == "SEALED" and result["GPU_calls"] == 0
    assert not called[0].exists() and not called[0].parent.exists()
    assert m.validate_profile(m.g.read(target))["kernel"]["prefill_block_m"] == 32
    before = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        m.seal_profile(Namespace(output=str(target), panel="normalized48.json"))
    assert target.read_bytes() == before and len(called) == 1


def test_process_overlay_rebinds_exactly_two_functions_and_restores():
    previous = dict(vars(m.g))
    with m.installed_overlay():
        changed = {k for k, value in previous.items() if getattr(m.g, k) is not value}
        assert changed == {"engine_settings", "physical_lease"}
        assert m.g.engine_settings("checkpoint")["cpu_offload_gb"] == 28
    assert m.g.engine_settings is m._BASE_ENGINE_SETTINGS
    assert m.g.physical_lease is m._BASE_PHYSICAL_LEASE


def lock_fixture(tmp_path, monkeypatch, create=True, admitted=True):
    root = tmp_path / "t1/evaluation/outputs"
    root.mkdir(parents=True)
    device = "fixture-existing-native-gpu"
    path = root / (".gpu-" + m.g.digest({"host": socket.gethostname(), "device": device}) + ".lock")
    if create:
        path.write_bytes(b"existing-native-lock")
    runtime = {"lease_root": str(root), "asset_roots": {"T1": str(tmp_path / "t1")}}
    torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 1,
        get_device_properties=lambda index: SimpleNamespace(uuid=device)))
    observed = []
    def snapshot(torch_arg, budgets):
        assert torch_arg is torch
        observed.append(dict(budgets))
        return {"admitted": admitted, "budgets": budgets}
    monkeypatch.setattr(m.inference, "resource_snapshot", snapshot)
    return runtime, torch, path, observed


def assert_locked(path):
    with path.open("r+b") as contender:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def assert_unlocked(path):
    with path.open("r+b") as contender:
        fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_uses_same_existing_lock_path_and_inode(tmp_path, monkeypatch):
    runtime, torch, path, observed = lock_fixture(tmp_path, monkeypatch)
    before = path.stat()
    with m.physical_lease(runtime, torch) as receipt:
        assert receipt["path"] == str(path)
        assert (receipt["st_dev"], receipt["st_ino"]) == (before.st_dev, before.st_ino)
        assert_locked(path)
    assert_unlocked(path)
    assert path.stat().st_ino == before.st_ino
    assert path.read_bytes() == b"existing-native-lock"
    assert observed == [{"min_free_gpu_gib": 78, "min_available_ram_gib": 64}]


def test_missing_native_lock_is_never_created(tmp_path, monkeypatch):
    runtime, torch, path, observed = lock_fixture(tmp_path, monkeypatch, create=False)
    with pytest.raises(ValueError, match="Existing native GPU lock"):
        with m.physical_lease(runtime, torch):
            pytest.fail("No independent replacement lock is allowed")
    assert not path.exists() and observed == []


def test_insufficient_current_resources_release_lock_without_launch(tmp_path, monkeypatch):
    runtime, torch, path, observed = lock_fixture(tmp_path, monkeypatch, admitted=False)
    with pytest.raises(ValueError, match="not currently available"):
        with m.physical_lease(runtime, torch):
            pytest.fail("No model work admitted")
    assert_unlocked(path)


def test_run_delegates_shutdown_order_and_restores_process_overlay(tmp_path, monkeypatch):
    runtime, torch, lock, observed = lock_fixture(tmp_path, monkeypatch)
    reference = tmp_path / "existing-reference.json"
    reference.write_text("already-qualified-reference")
    monkeypatch.setitem(m.REFERENCE_SHA, 20, m.g.file_sha256(reference))
    profile_path = tmp_path / "memory-profile.json"
    m.g.seal(profile_path, memory_profile())
    args = Namespace(profile=str(profile_path), reference=str(reference), admission="existing-admission", output="new-output")
    order = []
    def delegated_run(actual):
        assert actual is args
        assert m.g.engine_settings is m.engine_settings and m.g.physical_lease is m.physical_lease
        with m.g.physical_lease(runtime, torch), ExitStack() as cleanup:
            def engine_shutdown():
                assert_locked(lock)
                order.append("owned-engine-exited")
            cleanup.callback(engine_shutdown)
            order.append("generation-finished")
        assert_unlocked(lock)
        order.append("native-lease-released")
        return {"status": "COMPLETE"}
    monkeypatch.setattr(m.g, "run", delegated_run)
    assert m.run(args) == {"status": "COMPLETE"}
    assert order == ["generation-finished", "owned-engine-exited", "native-lease-released"]
    assert m.g.engine_settings is m._BASE_ENGINE_SETTINGS
    assert m.g.physical_lease is m._BASE_PHYSICAL_LEASE


def test_reference_mismatch_stops_before_delegated_runner(tmp_path, monkeypatch):
    path, reference = tmp_path / "profile.json", tmp_path / "wrong-reference.json"
    m.g.seal(path, memory_profile())
    reference.write_text("not the already-qualified reference")
    monkeypatch.setattr(m.g, "run", lambda args: pytest.fail("Reference mismatch must precede base run"))
    with pytest.raises(ValueError, match="existing frozen CPU reference"):
        m.run(Namespace(profile=str(path), reference=str(reference)))
    assert m.g.engine_settings is m._BASE_ENGINE_SETTINGS


def test_new_cli_import_remains_cpu_only():
    result = subprocess.run([sys.executable, "-B", "-c",
        "import classic_precision.gemma_memory,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
