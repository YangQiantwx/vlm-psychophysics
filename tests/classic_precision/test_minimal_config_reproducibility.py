"""CPU evidence and intervention checks; no model loading, SSH or CUDA calls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import minimal_config_reproducibility as minimal
from classic_precision import reproducibility as original
from classic_precision.inference import GENERATION
from classic_precision.storage import RunStore, atomic_json, digest, file_sha256

runtime = minimal.runtime


def _helpers(name):
    spec = importlib.util.spec_from_file_location("minimal_test_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(mode):
    plan = minimal.build_profile(mode)
    result = {key: deepcopy(plan[key]) for key in ("mode", "baseline_environment", "environment_interventions", "environment",
        "cleared_environment", "application_phase", "torch_policy", "certification")}
    result.update(profile_id=plan["id"], family=runtime.family_for_mode(mode), sdk_files_modified=False,
        startup_environment={**plan["baseline_environment"], **plan["environment_interventions"]},
        base_runner_sha256=runtime.BASE_RUNNER_SHA256, helper_sha256=file_sha256(runtime.__file__),
        native_guard_helper_sha256=file_sha256(runtime.native.__file__), observer_helper_sha256=file_sha256(runtime.native.observers.__file__))
    if result["family"] == "qwen":
        result["qsa_installation"] = _helpers("test_config_only_reproducibility")._profile("qwen")["qsa_installation"]
    return result


def _flags():
    return {"seed": 0, "deterministic_algorithms": False, "deterministic_warn_only": False,
        "fill_uninitialized_memory": True, "cudnn_benchmark": True, "cudnn_deterministic": False,
        "fp32_precision": {"allow_tf32_matmul": True, "allow_tf32_cudnn": True}}


def _receipt(mode):
    before, after = _flags(), _flags()
    if mode == "qwen_tf32_off_only":
        after["fp32_precision"] = {key: False for key in after["fp32_precision"]}
    elif mode == "qwen_torch_deterministic":
        after.update(deterministic_algorithms=True, deterministic_warn_only=False, fill_uninitialized_memory=False)
    return {"phase": runtime.APPLICATION_PHASE, "model_loaded_before_application": True, "mode": mode,
        "declared_torch_policy": deepcopy(runtime.TORCH_POLICIES[mode]), "before": before, "after": after}


def _snapshot(mode):
    profile = _profile(mode)
    value = {"profile": profile, "packages": {"torch": "pinned", "transformers": "pinned", "numpy": "pinned", "av": "pinned"},
        "gpu_name": "test GPU", "compute_capability": [10, 0], "cuda": "13", "driver": "pinned",
        "inference_sha256": runtime.BASE_RUNNER_SHA256, "effective_environment": profile["environment"],
        "observed_environment": {**profile["baseline_environment"], **profile["environment_interventions"]},
        "native_postload_environment": {**profile["baseline_environment"], **profile["environment_interventions"]},
        "qsa_replacement_module_imported": False, "qsa_replacement_import_hook": False, "application_receipt": _receipt(mode),
        **deepcopy(_receipt(mode)["after"]), "worker_runtime": []}
    if profile["family"] == "qwen":
        worker = deepcopy(_helpers("test_config_only_reproducibility")._snapshot("qwen")["comparison_identity"]["worker_runtime"][0])
        worker.update(deepcopy(_receipt(mode)["after"]), application_receipt=_receipt(mode))
        worker.update(native_postload_environment=deepcopy(value["native_postload_environment"]),
            observed_environment=deepcopy(value["observed_environment"]))
        value["worker_runtime"] = [worker]
    return {"comparison_identity": value}


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_exact_profile_seals_native_baseline_without_hidden_optimizations(mode):
    profile = minimal.build_profile(mode)
    assert profile["mode"] == mode
    assert profile["environment_interventions"] == ({"VLLM_ENABLE_V1_MULTIPROCESSING": "0"} if mode == "qwen_schedule_only" else {})
    for key in runtime.native.BASELINE_CLEARED_ENVIRONMENT:
        assert profile["baseline_environment"][key] is None
        assert key in profile["cleared_environment"] or key in profile["environment_interventions"]
    assert minimal.validate_runtime_policy(_snapshot(mode), runtime.family_for_mode(mode))


def test_explicit_observed_baseline_is_reported_and_sealed():
    profile = minimal.build_profile("qwen_tf32_off_only", {"NUMPY_MADVISE_HUGEPAGE": "0", "OPENBLAS_NUM_THREADS": "1"})
    assert profile["baseline_environment"]["NUMPY_MADVISE_HUGEPAGE"] == "0"
    assert profile["environment_interventions"] == {}
    assert profile["environment"]["OPENBLAS_NUM_THREADS"] == "1"


def _fake_torch():
    state = {"deterministic": False, "warn": False, "calls": []}
    def deterministic(enabled, warn_only=False):
        state.update(deterministic=enabled, warn=warn_only)
        state["calls"].append((enabled, warn_only))
    precision = lambda: SimpleNamespace(fp32_precision="tf32")
    torch = SimpleNamespace(initial_seed=lambda: 0, are_deterministic_algorithms_enabled=lambda: state["deterministic"],
        is_deterministic_algorithms_warn_only_enabled=lambda: state["warn"], use_deterministic_algorithms=deterministic,
        utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=precision()),
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, conv=precision(), rnn=precision())))
    return torch, state


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_policy_never_calls_a_seed_or_changes_undeclared_flags(mode):
    torch, state = _fake_torch()  # deliberately has no seed or thread setter
    receipt = runtime._apply(torch, mode)
    value = {**receipt["after"], "application_receipt": receipt}
    minimal._validate_application(value, mode)
    assert state["calls"] == ([(True, False)] if mode == "qwen_torch_deterministic" else [])
    assert torch.backends.cudnn.benchmark is True and torch.backends.cudnn.deterministic is False
    if mode in ("native_cosmos", "qwen_schedule_only"):
        assert receipt["before"] == receipt["after"]


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
@pytest.mark.parametrize("change", ["undeclared_cudnn", "unapplied", "load_phase", "baseline_missing", "environment", "source"])
def test_parent_policy_or_unrecorded_baseline_cannot_certify_runtime(mode, change):
    snapshot = _snapshot(mode)
    value = snapshot["comparison_identity"]
    if change == "undeclared_cudnn":
        value["cudnn_benchmark"] = False
        value["application_receipt"]["after"]["cudnn_benchmark"] = False
    elif change == "unapplied": value["fill_uninitialized_memory"] = not value["fill_uninitialized_memory"]
    elif change == "load_phase": value["application_receipt"]["model_loaded_before_application"] = False
    elif change == "baseline_missing": value["application_receipt"]["before"].pop("fp32_precision")
    elif change == "environment": value["observed_environment"]["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    else: value["profile"]["helper_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, runtime.family_for_mode(mode))


@pytest.mark.parametrize("change", ["flags", "qsa", "engine"])
def test_actual_worker_is_checked_independently_of_correct_parent(change):
    snapshot = _snapshot("qwen_torch_deterministic")
    worker = snapshot["comparison_identity"]["worker_runtime"][0]
    if change == "flags": worker["fill_uninitialized_memory"] = True
    elif change == "qsa": worker["qsa_native_identity"]["replacement_enabled"] = True
    else: worker["engine_config"]["model_config"]["seed"] = 1
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, "qwen")


@pytest.mark.parametrize("scope", ["parent", "worker"])
@pytest.mark.parametrize("key,accepted", [("VLLM_WORKER_MULTIPROC_METHOD", True), ("CUBLAS_WORKSPACE_CONFIG", False)])
def test_only_documented_native_spawn_environment_mutation_is_allowed(scope, key, accepted):
    snapshot = _snapshot("qwen_tf32_off_only")
    value = snapshot["comparison_identity"]
    target = value if scope == "parent" else value["worker_runtime"][0]
    target["native_postload_environment"][key] = "spawn" if accepted else ":4096:8"
    target["observed_environment"][key] = target["native_postload_environment"][key]
    if accepted:
        minimal.validate_runtime_policy(snapshot, "qwen")
        target["observed_environment"][key] = "fork"
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, "qwen")


def test_worker_rpc_applies_after_load_and_keeps_native_qsa_guard(monkeypatch):
    torch, state = _fake_torch()
    monkeypatch.setitem(sys.modules, "torch", torch)
    calls = []
    monkeypatch.setattr(runtime.native, "verify_native_qsa", lambda: calls.append("native_qsa"))
    monkeypatch.setattr(runtime.native, "_worker_runtime", lambda worker: runtime.native._torch_flags(torch))
    worker = SimpleNamespace()
    first = runtime._worker_apply(worker, "qwen_torch_deterministic")
    second = runtime._worker_apply(worker, "qwen_torch_deterministic")
    assert first == second and len(state["calls"]) == 1 and calls == ["native_qsa", "native_qsa"]
    assert first["application_receipt"]["model_loaded_before_application"] is True


@pytest.fixture
def panel(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rows = [{"sample_id": "t2-" + str(i), "task": "T2", "truth": "RIGHT", "source_parameters": {
        "delta_Y": .11, "background_Y": .05, "fps": 60, "onset_frame": 18, "offset_frame": 48}} for i in range(48)]
    rows += [{"sample_id": "t3-qualification", "task": "T3", "truth": "LEFT", "source_parameters": {}}]
    manifest, ready = source / "manifest.json", source / "ready.jsonl"
    atomic_json(manifest, {"rows": rows})
    ready.write_text("".join(json.dumps({"sample_id": row["sample_id"], "video_sha256": "e" * 64}) + "\n" for row in rows))
    plan = minimal.prepare_plan({"source_run": str(source), "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "ready": str(ready), "asset_root": str(source), "sample_ids": [row["sample_id"] for row in rows],
        "scientific_ids": [row["sample_id"] for row in rows[:48]], "auxiliary_T3_qualification_id": rows[-1]["sample_id"]})
    path = tmp_path / "minimal.json"
    atomic_json(path, plan)
    return path, plan, tmp_path / "outputs"


def _populate(panel):
    _, plan, output = panel
    mode, family = plan["profile"]["mode"], plan["models"][0]
    rows = json.loads(Path(plan["manifest"]).read_text())["rows"]
    ready = {row["sample_id"]: row for row in original._read_jsonl(plan["ready"])}
    tensors = _helpers("test_config_only_reproducibility")._tensors
    for cold in range(plan["cold_passes"]):
        for repeat in range(plan["repeats"]):
            identity = {"model_key": family, "purpose": minimal.SCHEMA, "manifest_sha256": file_sha256(plan["manifest"]),
                "reproducibility": {"plan_sha256": digest(plan), "profile": plan["profile"], "profile_status": _profile(mode),
                    "cold_pass": cold, "repeat": repeat, "sample_ids": plan["sample_ids"], "harness_sha256": file_sha256(original.__file__)}}
            with RunStore(original._repeat_directory(output, cold, repeat), identity, family, "repro", plan["sample_ids"]) as store:
                events_path = store.directory / "events.jsonl"
                event = json.loads(events_path.read_text())
                event["owner"].update(pid=100+cold, process_start_ticks=str(10+cold), boot_id="fixture")
                events_path.write_text(json.dumps(event) + "\n")
                snapshot = _snapshot(mode)
                value = original.runtime_identity(snapshot)
                store.event("REPRO_RUNTIME", snapshot=snapshot, comparison_identity=value, comparison_sha256=digest(value))
                for row in rows:
                    sid = row["sample_id"]
                    store.begin(sid, digest(row))
                    for stage in ("cpu_processor", "gpu_forward"):
                        store.event("REPRO_INPUT_HASH", sample_id=sid, stage=stage, tensors=tensors(stage, family))
                    for stage in ("before", "after"):
                        store.event("REPRO_CALL_RUNTIME", sample_id=sid, stage=stage, snapshot=snapshot)
                    result = store.commit_output(sid, {"task": row["task"], "truth": row["truth"],
                        "source_parameters": row["source_parameters"], "row_identity": digest({"row": row, "ready": ready[sid]}),
                        "ready_record": ready[sid], "ready_identity": digest(ready[sid]), "raw_answer": "LEFT", "output_token_ids": [10, 11],
                        "finish_reason": "stop", "actual_input_tokens": 100, "input_receipt": {"chat_template_sha256": "f" * 64},
                        "effective_generation": deepcopy(GENERATION), "parent_sample_id": sid,
                        "reproducibility": {"plan_sha256": digest(plan), "cold_pass": cold, "repeat": repeat}})
                    store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=result["attempt_id"])
                store.event("REPRO_REPEAT_COMPLETE", status="COMPLETE")
        atomic_json(output / ("cold-" + str(cold)) / (family + ".json"), {"status": "COMPLETE", "plan_sha256": digest(plan)})


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_complete_four_store_evidence_has_correct_independent_process_layout(panel, mode):
    path, source, output = panel
    plan = minimal.prepare_plan(source, mode)
    atomic_json(path, plan)
    assert minimal.load_plan(path)[0] == plan
    assert (plan["cold_passes"], plan["repeats"]) == ((4, 1) if mode == "native_cosmos" else (2, 2))
    _populate((path, plan, output))
    report = minimal.compare(path, output)
    value = report["models"][plan["models"][0]]
    assert report["status"] == "PASS" and report["evidence_status"] == "COMPLETE"
    assert value["complete_evidence_stores"] == 4 and value["samples_per_repeat"] == 49


@pytest.mark.parametrize("change,expected", [("tokens", "DIFFERENT"), ("input_hash", "INVALID_EVIDENCE"),
    ("runtime_call", "INVALID_EVIDENCE"), ("owner", "INVALID_EVIDENCE"), ("owner_missing", "INVALID_EVIDENCE"), ("completion", "INVALID_EVIDENCE")])
def test_output_difference_is_valid_but_changed_controls_fail_closed(panel, change, expected):
    _populate(panel)
    directory = original._repeat_directory(panel[2], 3, 0) / "cosmos"
    if change == "completion":
        (panel[2] / "cold-3/cosmos.json").unlink()
    else:
        path = directory / ("outputs.jsonl" if change == "tokens" else "events.jsonl")
        rows = original._read_jsonl(path)
        if change == "tokens": rows[0]["output_token_ids"] = [10, 12]
        elif change == "input_hash": next(row for row in rows if row.get("stage") == "gpu_forward")["tensors"][0]["sha256"] = "0" * 64
        elif change == "runtime_call": next(row for row in rows if row.get("event") == "REPRO_CALL_RUNTIME")["snapshot"]["comparison_identity"]["cudnn_benchmark"] = False
        elif change == "owner_missing": next(row for row in rows if row.get("event") == "WORKER_START")["owner"].pop("boot_id")
        else: next(row for row in rows if row.get("event") == "WORKER_START")["owner"].update(pid=100, process_start_ticks="10")
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = minimal.compare(panel[0], panel[2])
    assert report["status"] == "FAIL" and report["models"]["cosmos"]["comparison_status"] == expected
    assert report["evidence_status"] == ("COMPLETE" if change == "tokens" else "INVALID_OR_INCOMPLETE")


@pytest.mark.parametrize("change", ["layout", "retry", "source", "implementation", "family"])
def test_plan_tampering_is_rejected(panel, change):
    path, plan, _ = panel
    if change == "layout": plan.update(cold_passes=2, repeats=2)
    elif change == "retry": plan["automatic_retry"] = True
    elif change == "source": plan["manifest_sha256"] = "0" * 64
    elif change == "family": plan["models"] = ["qwen"]
    else: plan["profile"]["implementation"]["entry_sha256"] = "0" * 64
    atomic_json(path, plan)
    with pytest.raises(ValueError):
        minimal.load_plan(path)


@pytest.mark.parametrize("calibration_ok", [True, False])
def test_native_runner_keeps_calibration_lease_resource_and_no_retry_gates(monkeypatch, panel, calibration_ok):
    helpers = _helpers("test_reproducibility")
    state = helpers._mock_run(monkeypatch, panel, calibration_ok=calibration_ok)
    for key in panel[1]["profile"]["cleared_environment"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in panel[1]["profile"]["environment"].items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(runtime, "activate", lambda *a: state["guards"].append("minimal_activate"))
    monkeypatch.setattr(runtime, "status", lambda: _profile("native_cosmos"))
    monkeypatch.setattr(runtime, "runtime_snapshot", lambda: _snapshot("native_cosmos"))
    monkeypatch.setattr(runtime, "prepare_native", lambda family: None)
    if not calibration_ok:
        with pytest.raises(ValueError, match="128-state"):
            minimal.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
        assert state["loads"] == 0 and "minimal_activate" not in state["guards"]
    else:
        result = minimal.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
        assert result["status"] == "COMPLETE" and len(state["calls"]) == 49
        assert state["loads"] == state["closes"] == 1
        assert "resources" in state["guards"] and any(isinstance(v, tuple) and v[0] == "lock" for v in state["guards"])
        with pytest.raises(FileExistsError):
            minimal.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])


def test_import_and_harness_creation_leave_frozen_modules_and_torch_untouched():
    code = """
import sys
from classic_precision import reproducibility as original
from classic_precision import config_only_reproducibility as config
before=(original.SCHEMA, original._validate_profile, original.importlib, config.SCHEMA)
from classic_precision import minimal_config_reproducibility as minimal
minimal._new_harness()
assert before == (original.SCHEMA, original._validate_profile, original.importlib, config.SCHEMA)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert not any(n.endswith('classic_precision_qsa_determinism') for n in sys.modules)
assert minimal.runtime.status() == {}
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(minimal.__file__).parents[1], check=True)
