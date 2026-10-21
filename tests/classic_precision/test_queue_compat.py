"""Queue deferral changes admission order, never scientific work or denominators."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import scheduler
from classic_precision.manifest import make_manifest
from runtime_compat import classic_precision_queue_compat as compat


ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_SOURCES = {
    "inference.py": "2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3",
    "scheduler.py": compat.BASE_SCHEDULER_SHA256,
    "pipeline.py": "a2537dd42cfe1660bbd767b665c93c2444f37b099553c08c3ed7952e6cffc34e",
    "protocol.json": "188762caf2bae60b6dd52e8be6046387ca10d1eba6045f3f6624a4985d45827c",
}


@pytest.fixture
def installed(monkeypatch):
    monkeypatch.setattr(scheduler, "choose_model", scheduler.choose_model)
    monkeypatch.setattr(scheduler, "atomic_json", scheduler.atomic_json)
    monkeypatch.setenv(compat.ENVIRONMENT_VARIABLE, "gemma")
    return compat.install()


def _process(tmp_path, code, value=None, others=False, package_root=ROOT):
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONPATH": os.pathsep.join([str(ROOT / "runtime_compat"), str(package_root)])}
    for name in (compat.ENVIRONMENT_VARIABLE, "CLASSIC_PRECISION_KV_COMPAT",
                 "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB"):
        env.pop(name, None)
    if value is not None:
        env[compat.ENVIRONMENT_VARIABLE] = value
    if others:
        env.update(CLASSIC_PRECISION_KV_COMPAT="1", CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB="16")
    return subprocess.run([sys.executable, "-B", "-c", code], cwd=tmp_path, env=env,
                          capture_output=True, text=True)


def test_absent_flag_leaves_original_functions_and_choices(monkeypatch, tmp_path):
    monkeypatch.delenv(compat.ENVIRONMENT_VARIABLE, raising=False)
    original, original_json = scheduler.choose_model, scheduler.atomic_json
    assert compat.install() is None
    assert scheduler.choose_model is original and scheduler.atomic_json is original_json
    specs = make_manifest()["models"]
    assert original(list(specs), specs, {"free_gpu_gib": 96, "available_ram_gib": 100}) == "qwen"
    result = _process(tmp_path, """
import sys
assert 'classic_precision_queue_compat' not in sys.modules
assert 'classic_precision.scheduler' not in sys.modules
from classic_precision import scheduler
assert not hasattr(scheduler.choose_model, '_classic_precision_queue_override')
""")
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("pending,gpu,ram,expected", [
    (["gemma", "qwen", "nemo", "cosmos"], 96, 100, "cosmos"),
    (["qwen", "nemo"], 96, 100, "nemo"),
    (["gemma", "qwen", "nemo", "cosmos"], 80, 19, "nemo"),
    (["gemma", "qwen", "nemo", "cosmos"], 78, 100, "qwen"),
    (["gemma", "qwen", "nemo", "cosmos"], 72, 64, "qwen"),
    (["gemma", "qwen", "nemo", "cosmos"], 71, 100, None),
    (["gemma", "qwen", "nemo", "cosmos"], 96, 15, None),
    (["gemma"], 96, 100, None),
    ([], 96, 100, None),
])
def test_filter_priority_and_original_capacity_gates(installed, pending, gpu, ram, expected):
    specs = make_manifest()["models"]
    before = list(pending), json.dumps(specs, sort_keys=True)
    assert scheduler.choose_model(pending, specs, {"free_gpu_gib": gpu,
           "available_ram_gib": ram}) == expected
    assert (pending, json.dumps(specs, sort_keys=True)) == before


def test_install_is_idempotent_and_only_supervisor_snapshots_gain_metadata(installed, tmp_path):
    choose, write = scheduler.choose_model, scheduler.atomic_json
    assert compat.install() == installed
    assert scheduler.choose_model is choose and scheduler.atomic_json is write
    value = {"status": "WAITING_FOR_FITTING_MODEL", "expected_per_model": 1168,
             "completed": {"cosmos": 1168, "nemo": 1168, "qwen": 1168, "gemma": 40}}
    for name in ("supervisor-shard-0.json", "identity.json", "pipeline.json", "summary.json",
                 "supervisor-shard-invalid.json"):
        scheduler.atomic_json(tmp_path / name, value)
        actual = json.loads((tmp_path / name).read_text())
        if name == "supervisor-shard-0.json":
            assert actual == {**value, "deferred_models": ["gemma"],
                              "selection_scope": ["cosmos", "nemo", "qwen"]}
        else:
            assert actual == value
    assert "deferred_models" not in value


def test_deferred_remaining_work_keeps_supervisor_waiting(installed, monkeypatch, tmp_path):
    from classic_precision import calibration, inference, ready
    manifest = make_manifest()
    model_root = tmp_path / "models"
    for family, spec in manifest["models"].items():
        (model_root / spec["checkpoint_relative"]).mkdir(parents=True)
        marker = tmp_path / "calibration" / family / "summary.json"
        marker.parent.mkdir(parents=True)
        marker.write_text("{}")
    ready_rows = {row["sample_id"]: {} for row in manifest["rows"]}
    monkeypatch.setattr(inference, "load_run_inputs", lambda *a, **k:
        (manifest, None, ready_rows, None, tmp_path / "ready.jsonl",
         SimpleNamespace(refresh=lambda: ready_rows)))
    monkeypatch.setattr(inference, "build_run_identity", lambda *a, **k: ({}, None))
    monkeypatch.setattr(calibration, "calibration_identity", lambda *a, **k: {})
    monkeypatch.setattr(calibration, "calibration_ready", lambda *a, **k: True)
    monkeypatch.setattr(ready, "start_ready", lambda *a, **k: True)
    monkeypatch.setattr(scheduler, "progress", lambda output, family, ids, *a, **k:
        {"complete": family != "gemma", "completed": len(ids) if family != "gemma" else 40,
         "needs_recovery": False, "exhausted_ids": [], "held_ids": []})
    monkeypatch.setattr(scheduler, "capacity", lambda gpu:
        {"free_gpu_gib": 96, "available_ram_gib": 100})
    monkeypatch.setattr(scheduler.subprocess, "run", lambda *a, **k: pytest.fail("worker launched"))
    class EndFixture(Exception):
        pass
    monkeypatch.setattr(scheduler.time, "sleep", lambda seconds: (_ for _ in ()).throw(EndFixture()))
    with pytest.raises(EndFixture):
        scheduler.supervise(tmp_path / "manifest.json", tmp_path / "outputs", model_root, "0", 0)
    status = json.loads((tmp_path / "outputs" / "supervisor-shard-0.json").read_text())
    assert status["status"] == "WAITING_FOR_FITTING_MODEL"
    assert status["model"] is None and status["expected_per_model"] == 1168
    assert status["completed"] == {"cosmos": 1168, "nemo": 1168, "qwen": 1168, "gemma": 40}
    assert status["deferred_models"] == ["gemma"]


@pytest.mark.parametrize("others", [False, True])
def test_process_opt_in_preserves_other_hooks_and_cpu_entry_is_gpu_import_free(tmp_path, others):
    code = f"""
import sys
from classic_precision import scheduler, pipeline, inference
assert hasattr(scheduler.choose_model, '_classic_precision_queue_override')
assert hasattr(inference.ModelSession._load_vllm, '_classic_precision_resource_override') == {others!r}
assert inference._kv_capacity.__module__ == {'classic_precision_vllm_kv_compat' if others else 'classic_precision.inference'!r}
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
"""
    result = _process(tmp_path, code, "gemma", others)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("value", ["", "Gemma", "gemma,qwen", " gemma", "gemma "])
def test_invalid_opt_in_refuses_before_main(tmp_path, value):
    result = _process(tmp_path, "print('SHOULD_NOT_RUN')", value)
    assert result.returncode == 78 and "must be exactly gemma" in result.stderr
    assert "SHOULD_NOT_RUN" not in result.stdout


def test_scheduler_source_drift_refuses_before_import(tmp_path):
    package = tmp_path / "classic_precision"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "scheduler.py").write_text("raise AssertionError('UNVERIFIED_IMPORT')\n")
    result = _process(tmp_path, "print('SHOULD_NOT_RUN')", "gemma", package_root=tmp_path)
    assert result.returncode == 78 and "base scheduler SHA mismatch" in result.stderr
    assert "UNVERIFIED_IMPORT" not in result.stderr and "SHOULD_NOT_RUN" not in result.stdout


def test_scientific_sources_and_nemo_checkpoint_are_unchanged():
    for name, expected in SCIENTIFIC_SOURCES.items():
        assert hashlib.sha256((ROOT / "classic_precision" / name).read_bytes()).hexdigest() == expected
    nemo = make_manifest()["models"]["nemo"]
    assert nemo["repo"] == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4"
    assert nemo["revision"] == "82d93699acd3be9e4617c53360fe5b3f5dbeb1b3"
