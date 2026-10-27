"""CPU-only profile, native-operator and unchanged-harness gate checks."""
from copy import deepcopy
import functools
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace, FunctionType

import pytest

from classic_precision import config_only_reproducibility as config
from classic_precision import reproducibility as strict
from classic_precision.inference import GENERATION
from classic_precision.storage import RunStore, atomic_json, digest, file_sha256

runtime = config.runtime


def _profile(family="cosmos", mode="config_only", environment=None):
    result = {"profile_id": config.PROFILE_ID, "family": family, "mode": "config_only",
        "base_runner_sha256": runtime.BASE_RUNNER_SHA256,
        "helper_sha256": file_sha256(runtime.__file__), "observer_helper_sha256": file_sha256(runtime.observers.__file__),
        "environment": runtime.launch_environment(), "sdk_files_modified": False,
        "certification": "CONFIG_ONLY_UNVALIDATED_UNTIL_REPEAT_SUITE_PASSES"}
    if family == "qwen":
        result["qsa_installation"] = {"target_module": runtime.QSA_TARGET_MODULE, "function_module": runtime.QSA_TARGET_MODULE,
            "function_qualname": "_topk", "expected_native_source_sha256": runtime.QSA_NATIVE_SOURCE_SHA256,
            "native_source_sha256": runtime.QSA_NATIVE_SOURCE_SHA256, "native_code_sha256": "b" * 64,
            "replacement_enabled": False, "observer_only": True, "state": "native_verified"}
    if mode == "native_nemo":
        result.update(profile_id=runtime.BASELINE_PROFILE_ID, mode=mode,
            environment=dict(environment or runtime.BASELINE_ENVIRONMENT),
            certification="NATIVE_NEMO_OBSERVED_RETEST_NOT_HISTORICAL_FLAG_RESTORATION",
            startup_cleared_environment={key: None for key in runtime.BASELINE_CLEARED_ENVIRONMENT})
    return result


def _snapshot(family="cosmos", mode="config_only", environment=None):
    profile = _profile(family, mode, environment)
    result = {"comparison_identity": {"profile": profile, "packages": {"torch": "pinned", "transformers": "pinned",
        "numpy": "pinned", "av": "pinned"}, "gpu_name": "test GPU", "compute_capability": [10, 0],
        "cuda": "13", "driver": "pinned", "inference_sha256": runtime.BASE_RUNNER_SHA256, "seed": 0,
        "effective_environment": runtime.launch_environment(), "deterministic_algorithms": True,
        "deterministic_warn_only": False, "cudnn_benchmark": False, "cudnn_deterministic": True,
        "fill_uninitialized_memory": False, "qsa_replacement_module_imported": False, "qsa_replacement_import_hook": False,
        "fp32_precision": {"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"}}}
    flags = result["comparison_identity"]
    if mode == "native_nemo":
        flags.update(deterministic_algorithms=False, cudnn_deterministic=False, fill_uninitialized_memory=True,
            fp32_precision={"allow_tf32_matmul": True, "allow_tf32_cudnn": True},
            effective_environment=profile["environment"], startup_cleared_environment=profile["startup_cleared_environment"],
            effective_cleared_environment={key: None for key in runtime.BASELINE_CLEARED_ENVIRONMENT})
    keys = ("seed", "deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic", "fill_uninitialized_memory", "fp32_precision")
    engine = {"model_config": {"seed": 0, "enforce_eager": True, "max_model_len": 65536, "dtype": "torch.bfloat16"},
        "parallel_config": {"tensor_parallel_size": 1, "pipeline_parallel_size": 1, "data_parallel_size": 1, "distributed_executor_backend": "native"},
        "scheduler_config": {"max_num_seqs": 1, "max_num_batched_tokens": 4096 if family == "nemo" else 49152, "enable_chunked_prefill": True},
        "cache_config": {"cache_dtype": "bfloat16", "enable_prefix_caching": False}}
    flags["worker_runtime"] = [{**{key: deepcopy(flags[key]) for key in keys}, "worker_class": "test.Worker",
        "model_class": "native.Qwen4ExpForConditionalGeneration" if family == "qwen" else "native.Nemotron",
        "engine_config": engine, "qsa_native_identity": deepcopy(profile.get("qsa_installation")),
        "gpu_name": flags["gpu_name"]}] if family in ("nemo", "qwen") else []
    return result


@pytest.mark.parametrize("family", runtime.FAMILIES)
def test_all_families_require_same_fill_false_and_policy(monkeypatch, family):
    snapshot = _snapshot(family)
    assert config.validate_runtime_policy(snapshot, family)["fill_uninitialized_memory"] is False
    snapshot["comparison_identity"]["fill_uninitialized_memory"] = True
    with pytest.raises(ValueError, match="fill_uninitialized_memory"):
        config.validate_runtime_policy(snapshot, family)
    calls = []
    precision = lambda: SimpleNamespace(fp32_precision="tf32")
    torch = SimpleNamespace(use_deterministic_algorithms=lambda *a, **k: calls.append((a, k)),
        manual_seed=lambda seed: calls.append(seed),
        utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=precision()),
            cudnn=SimpleNamespace(benchmark=True, deterministic=False, conv=precision(), rnn=precision())))
    runtime._configure_torch(torch)
    assert torch.utils.deterministic.fill_uninitialized_memory is False
    assert calls == [((True,), {"warn_only": False}), 0]
    assert torch.backends.cudnn.benchmark is False and torch.backends.cudnn.deterministic is True
    assert torch.backends.cuda.matmul.fp32_precision == "ieee"


@pytest.mark.parametrize("field,value", [("deterministic_algorithms", False), ("deterministic_warn_only", True),
    ("cudnn_benchmark", True), ("cudnn_deterministic", False), ("seed", 1), ("qsa_replacement_module_imported", True),
    ("qsa_replacement_import_hook", True), ("fp32_precision", {"matmul": "tf32"})])
def test_relaxed_runtime_is_rejected(field, value):
    snapshot = _snapshot()
    snapshot["comparison_identity"][field] = value
    with pytest.raises(ValueError):
        config.validate_runtime_policy(snapshot, "cosmos")


@pytest.fixture
def native_qsa(tmp_path, monkeypatch):
    path = tmp_path / "qsa_indexer.py"
    path.write_text("def _topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):\n    return None\n")
    monkeypatch.setattr(runtime, "QSA_NATIVE_SOURCE_SHA256", file_sha256(path))
    monkeypatch.setattr(runtime, "_native_function", None)
    monkeypatch.setenv("CLASSIC_PRECISION_QSA_DETERMINISM", "0")
    module = ModuleType(runtime.QSA_TARGET_MODULE)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(module))
    return module


def test_native_qsa_source_code_qualname_and_original_object_verified(native_qsa):
    original = native_qsa._topk
    evidence = runtime.verify_native_qsa(native_qsa)
    assert native_qsa._topk is original
    assert evidence["native_source_sha256"] == file_sha256(native_qsa.__file__)
    assert evidence["function_module"] == runtime.QSA_TARGET_MODULE and evidence["function_qualname"] == "_topk"
    assert len(evidence["native_code_sha256"]) == 64 and evidence["replacement_enabled"] is False
    runtime.verify_native_qsa(native_qsa)
    native_qsa._topk = FunctionType(original.__code__, vars(native_qsa), "_topk")
    with pytest.raises(RuntimeError, match="object changed"):
        runtime.verify_native_qsa(native_qsa)


@pytest.mark.parametrize("change", ["wrapped", "code", "globals", "source", "qualname"])
def test_native_qsa_hook_or_source_change_fails_closed(native_qsa, change):
    function = native_qsa._topk
    if change == "wrapped":
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            return function(*args, **kwargs)
        native_qsa._topk = wrapper
    elif change == "code":
        function.__code__ = function.__code__.replace(co_consts=(None, 1))
    elif change == "globals":
        native_qsa._topk = FunctionType(function.__code__, dict(vars(native_qsa)), "_topk")
    elif change == "source":
        Path(native_qsa.__file__).write_text("# changed\n" + Path(native_qsa.__file__).read_text())
    else:
        function.__qualname__ = "other"
    with pytest.raises(RuntimeError, match="QSA"):
        runtime.verify_native_qsa(native_qsa)


def test_replacement_import_env_and_hook_are_forbidden(monkeypatch):
    monkeypatch.setenv("CLASSIC_PRECISION_QSA_DETERMINISM", "1")
    with pytest.raises(RuntimeError, match="=0"):
        runtime.assert_no_qsa_replacement()
    monkeypatch.setenv("CLASSIC_PRECISION_QSA_DETERMINISM", "0")
    monkeypatch.setitem(sys.modules, "classic_precision_qsa_determinism", ModuleType("classic_precision_qsa_determinism"))
    with pytest.raises(RuntimeError, match="module was imported"):
        runtime.assert_no_qsa_replacement()
    monkeypatch.delitem(sys.modules, "classic_precision_qsa_determinism")
    finder = type("Finder", (), {"__module__": "classic_precision_qsa_determinism"})()
    monkeypatch.setattr(sys, "meta_path", [*sys.meta_path, finder])
    with pytest.raises(RuntimeError, match="import hook"):
        runtime.assert_no_qsa_replacement()


def test_import_is_cpu_free_does_not_import_replacement_or_mutate_strict():
    code = """
import sys
from classic_precision import reproducibility as strict
before=(strict.SCHEMA, strict._validate_profile, strict.importlib)
from classic_precision import config_only_reproducibility as config
config._new_harness()
assert before == (strict.SCHEMA, strict._validate_profile, strict.importlib)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert not any(n.endswith('classic_precision_qsa_determinism') for n in sys.modules)
assert config.runtime.status() == {}
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(config.__file__).parents[1], check=True)


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
    source_plan = {"source_run": str(source), "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
        "ready": str(ready), "asset_root": str(source), "operational_ids": [row["sample_id"] for row in rows],
        "scientific_ids": [row["sample_id"] for row in rows[:48]], "auxiliary_T3_qualification_id": rows[-1]["sample_id"]}
    plan = config.prepare_plan(source_plan)
    path = tmp_path / "config-only.json"
    atomic_json(path, plan)
    return path, plan, tmp_path / "outputs"


def _tensors(stage, family="cosmos"):
    import math
    def tensor(name, shape, dtype, size):
        return {"name": name, "shape": shape, "dtype": dtype, "bytes": math.prod(shape) * size, "sha256": "b" * 64}
    if stage == "cpu_processor":
        if family == "nemo":
            return [tensor("input_ids", [1, 100], "torch.int64", 8), tensor("pixel_values_flat_video", [180, 3, 512, 768], "torch.bfloat16", 2)]
        return [tensor("input_ids", [1, 100], "torch.int64", 8), tensor("pixel_values_videos", [138240, 1536], "torch.float32", 4)]
    if family == "nemo":
        return [tensor("pixels", [180, 3, 512, 768], "torch.bfloat16", 2)]
    return [tensor("pixels", [138240, 1536], "torch.bfloat16", 2)]


def _populate(panel, mutation=None, family="cosmos"):
    _, plan, output = panel
    rows = {row["sample_id"]: row for row in json.loads(Path(plan["manifest"]).read_text())["rows"]}
    ready = {row["sample_id"]: row for row in strict._read_jsonl(plan["ready"])}
    for cold in range(2):
        for repeat in range(2):
            mode = config._mode(plan["profile"])
            identity = {"model_key": family, "purpose": plan["schema"], "manifest_sha256": file_sha256(plan["manifest"]),
                "reproducibility": {"plan_sha256": digest(plan), "profile": plan["profile"], "profile_status": _profile(family, mode, plan["profile"]["environment"]),
                    "cold_pass": cold, "repeat": repeat, "sample_ids": plan["sample_ids"], "harness_sha256": file_sha256(strict.__file__)}}
            directory = strict._repeat_directory(output, cold, repeat)
            with RunStore(directory, identity, family, "repro", plan["sample_ids"]) as store:
                events_path = store.directory / "events.jsonl"
                event = json.loads(events_path.read_text())
                event["owner"].update(pid=100+cold, process_start_ticks=str(cold), boot_id="fixture")
                events_path.write_text(json.dumps(event) + "\n")
                snapshot = _snapshot(family, mode, plan["profile"]["environment"])
                value = strict.runtime_identity(snapshot)
                store.event("REPRO_RUNTIME", snapshot=snapshot, comparison_identity=value, comparison_sha256=digest(value))
                for sid, row in rows.items():
                    store.begin(sid, digest(row))
                    for stage in ("cpu_processor", "gpu_forward"):
                        store.event("REPRO_INPUT_HASH", sample_id=sid, stage=stage, tensors=_tensors(stage, family))
                    if family in ("nemo", "qwen"):
                        for stage in ("before", "after"):
                            store.event("REPRO_WORKER_RUNTIME", sample_id=sid, stage=stage, workers=value["worker_runtime"])
                    result = store.commit_output(sid, {"task": row["task"], "truth": row["truth"],
                        "source_parameters": row["source_parameters"], "row_identity": digest({"row": row, "ready": ready[sid]}),
                        "ready_record": ready[sid], "ready_identity": digest(ready[sid]), "raw_answer": "LEFT",
                        "output_token_ids": [10, 11], "finish_reason": "stop", "actual_input_tokens": 100,
                        "input_receipt": {"chat_template_sha256": "f" * 64}, "effective_generation": deepcopy(GENERATION),
                        "parent_sample_id": sid, "reproducibility": {"plan_sha256": digest(plan), "cold_pass": cold, "repeat": repeat}})
                    store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=result["attempt_id"])
                store.event("REPRO_REPEAT_COMPLETE", status="COMPLETE")
                if mutation and cold == 1 and repeat == 1:
                    mutation(store.directory)
        atomic_json(output / ("cold-" + str(cold)) / (family + ".json"), {"status": "COMPLETE", "plan_sha256": digest(plan)})


def test_four_stores_normal_completion_exact_hash_and_full_token_pass(panel):
    _populate(panel)
    report = config.compare(panel[0], panel[2], models=["cosmos"])
    assert report["status"] == "PASS" and report["schema"] == config.SCHEMA
    assert report["models"]["cosmos"]["checked_repeat_stores"] == 4
    assert report["models"]["cosmos"]["samples_per_repeat"] == 49


@pytest.mark.parametrize("change", ["tokens", "input_hash", "technical", "completion", "runtime", "binding", "source"])
def test_unchanged_harness_gates_fail_closed(panel, change):
    def mutate(directory):
        if change == "completion":
            (directory.parent.parent / "cosmos.json").unlink(missing_ok=True)
            return
        filename = "events.jsonl" if change in ("input_hash", "technical", "runtime") else "outputs.jsonl"
        path = directory / filename
        rows = strict._read_jsonl(path)
        if change == "tokens": rows[0]["output_token_ids"] = [10, 12]
        elif change == "input_hash": next(r for r in rows if r.get("stage") == "gpu_forward")["tensors"][0]["sha256"] = "0" * 64
        elif change == "technical": rows[:] = [r for r in rows if r.get("event") != "TECHNICAL_PASS"]
        elif change == "runtime": next(r for r in rows if r.get("event") == "REPRO_RUNTIME")["snapshot"]["comparison_identity"]["fill_uninitialized_memory"] = True
        elif change == "binding": rows[0]["reproducibility"]["repeat"] = 0
        elif change == "source": rows[0]["row_identity"] = "0" * 64
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _populate(panel, mutate)
    if change == "completion":
        (panel[2] / "cold-1" / "cosmos.json").unlink()
    assert config.compare(panel[0], panel[2], models=["cosmos"])["status"] == "FAIL"


@pytest.mark.parametrize("change", ["strict_profile", "gemma", "fill", "count", "source_sha"])
def test_wrong_profile_gemma_or_unfrozen_design_rejected(panel, change):
    path, plan, _ = panel
    if change == "strict_profile": plan["profile"]["id"] = "classic-precision-strict-v2"
    elif change == "gemma": plan["models"].append("gemma")
    elif change == "fill": plan["profile"]["torch_policy"]["fill_uninitialized_memory"] = True
    elif change == "count": plan["cold_passes"] = 3
    else: plan["manifest_sha256"] = "0" * 64
    atomic_json(path, plan)
    with pytest.raises(ValueError):
        config.load_plan(path)


def test_strict_plan_loader_rejects_configuration_only_and_gemma_runtime(panel):
    with pytest.raises(ValueError, match="schema"):
        strict.load_plan(panel[0])
    with pytest.raises(ValueError, match="Gemma"):
        runtime.activate("gemma")


def _baseline_panel(panel):
    path, plan, output = panel
    baseline = config.prepare_plan(plan, mode="native_nemo")
    atomic_json(path, baseline)
    return path, baseline, output


def test_native_nemo_observed_flags_are_distinct_and_four_stores_pass(panel):
    panel = _baseline_panel(panel)
    _populate(panel, family="nemo")
    report = config.compare(panel[0], panel[2])
    assert report["status"] == "PASS" and report["schema"] == config.BASELINE_SCHEMA
    assert report["evidence_status"] == "COMPLETE"
    assert report["models"]["nemo"]["complete_evidence_stores"] == 4
    snapshot = _snapshot("nemo", "native_nemo")
    assert config.validate_runtime_policy(snapshot, "nemo")["fill_uninitialized_memory"] is True
    snapshot["comparison_identity"]["worker_runtime"][0]["fill_uninitialized_memory"] = False
    # Parent and worker flags are separate observations, with no forced defaults.
    config.validate_runtime_policy(snapshot, "nemo")


def test_native_nemo_worker_flags_must_exist_and_match_per_call(panel):
    panel = _baseline_panel(panel)
    def mutate(directory):
        path = directory / "events.jsonl"
        rows = strict._read_jsonl(path)
        next(row for row in rows if row.get("event") == "REPRO_WORKER_RUNTIME")["workers"][0]["fill_uninitialized_memory"] = False
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _populate(panel, mutate, family="nemo")
    result = config.compare(panel[0], panel[2])
    assert result["status"] == "FAIL" and result["evidence_status"] == "INVALID_OR_INCOMPLETE"
    assert result["models"]["nemo"]["comparison_status"] == "INVALID_EVIDENCE"


def test_complete_but_different_tokens_is_a_valid_diagnostic_result(panel):
    def mutate(directory):
        path = directory / "outputs.jsonl"
        rows = strict._read_jsonl(path)
        rows[0]["output_token_ids"] = [10, 12]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _populate(panel, mutate)
    report = config.compare(panel[0], panel[2], models=["cosmos"])
    assert report["status"] == "FAIL" and report["evidence_status"] == "COMPLETE"
    assert report["models"]["cosmos"]["comparison_status"] == "DIFFERENT"
    assert report["models"]["cosmos"]["complete_evidence_stores"] == 4


@pytest.mark.parametrize("change", ["gpu_input_hash", "same_cold_owner", "runtime_package"])
def test_complete_individual_stores_do_not_hide_invalid_cross_store_controls(panel, change):
    def mutate(directory):
        path = directory / "events.jsonl"
        rows = strict._read_jsonl(path)
        if change == "gpu_input_hash":
            next(row for row in rows if row.get("stage") == "gpu_forward")["tensors"][0]["sha256"] = "0" * 64
        elif change == "same_cold_owner":
            # Both repeats of cold 1 are rewritten after fixture population so
            # this tests cold independence, not a within-cold owner mismatch.
            return
        else:
            row = next(row for row in rows if row.get("event") == "REPRO_RUNTIME")
            row["snapshot"]["comparison_identity"]["packages"]["torch"] = "different"
            row["comparison_identity"] = row["snapshot"]["comparison_identity"]
            row["comparison_sha256"] = digest(row["comparison_identity"])
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _populate(panel, mutate)
    if change == "same_cold_owner":
        for repeat in range(2):
            path = strict._repeat_directory(panel[2], 1, repeat) / "cosmos/events.jsonl"
            rows = strict._read_jsonl(path)
            next(row for row in rows if row.get("event") == "WORKER_START")["owner"].update(pid=100, process_start_ticks="0")
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = config.compare(panel[0], panel[2], models=["cosmos"])
    value = report["models"]["cosmos"]
    assert report["status"] == "FAIL" and report["evidence_status"] == "INVALID_OR_INCOMPLETE"
    assert value["complete_evidence_stores"] == 4
    assert value["comparison_status"] == "INVALID_EVIDENCE" and value["control_error"]


@pytest.mark.parametrize("family", ["qwen", "cosmos"])
def test_native_baseline_refuses_other_families(family):
    with pytest.raises(ValueError, match="restricted to Nemo"):
        runtime.activate(family, "native_nemo")


def test_native_baseline_does_not_call_config_setter_and_preserves_native_flags():
    code = """
import os, sys
from types import SimpleNamespace
from runtime_compat import classic_precision_config_only as p
from classic_precision import inference
for key in p.BASELINE_CLEARED_ENVIRONMENT:
    os.environ.pop(key, None)
os.environ.update(p.BASELINE_ENVIRONMENT)
def native(self, *a, **k):
    self.store=SimpleNamespace(event=lambda *a, **k: None)
inference.ModelSession._initialize_processor=native
p._configure_torch=lambda *a: (_ for _ in ()).throw(AssertionError('configuration setter called'))
p.activate('nemo', 'native_nemo')
fake=SimpleNamespace(unchanged='sentinel')
sys.modules['torch']=fake
session=SimpleNamespace()
inference.ModelSession._initialize_processor(session, 'nemo', None, None, None)
assert vars(fake) == {'unchanged': 'sentinel'}
assert p.status()['mode'] == 'native_nemo'
assert p.status()['profile_id'] == p.BASELINE_PROFILE_ID
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(config.__file__).parents[1], check=True)


def test_native_baseline_cannot_inherit_strict_launch_environment():
    with pytest.raises(ValueError, match="clear strict"):
        config.build_profile("native_nemo", {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    snapshot = _snapshot("nemo", "native_nemo")
    snapshot["comparison_identity"]["startup_cleared_environment"]["PYTHONHASHSEED"] = "0"
    with pytest.raises(ValueError, match="inherited"):
        config.validate_runtime_policy(snapshot, "nemo")


@pytest.mark.parametrize("calibration_ok", [True, False])
def test_run_keeps_original_identity_calibration_resources_and_store_rotation(monkeypatch, panel, calibration_ok):
    # Reuse the native harness test's fake ModelSession rather than maintaining
    # another approximation of its request/commit/technical implementation.
    path = Path(__file__).with_name("test_reproducibility.py")
    spec = importlib.util.spec_from_file_location("base_harness_fixture", path)
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    state = helpers._mock_run(monkeypatch, panel, calibration_ok=calibration_ok)
    monkeypatch.setattr(runtime, "activate", lambda *a: state["guards"].append("config_activate"))
    monkeypatch.setattr(runtime, "status", _profile)
    monkeypatch.setattr(runtime, "runtime_snapshot", _snapshot)
    monkeypatch.setattr(runtime, "prepare_native", lambda family: None)
    if not calibration_ok:
        with pytest.raises(ValueError, match="128-state"):
            config.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
        assert state["loads"] == 0 and "config_activate" not in state["guards"]
        return
    result = config.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
    assert result["status"] == "COMPLETE" and len(state["calls"]) == 98
    assert state["loads"] == state["closes"] == 1
    assert state["guards"][:3] == ["calibration_identity", "calibration_ready", "config_activate"]
    assert "resources" in state["guards"]
    assert all((strict._repeat_directory(panel[2], 0, repeat) / "cosmos/identity.json").is_file() for repeat in range(2))


def test_compare_cli_returns_one_for_complete_valid_difference_and_saves_report(panel):
    def mutate(directory):
        path = directory / "outputs.jsonl"
        rows = strict._read_jsonl(path)
        rows[0]["output_token_ids"] = [10, 12]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _populate(panel, mutate)
    report_path = panel[2] / "comparison.json"
    code = config.main(["compare", "--plan", str(panel[0]), "--output", str(panel[2]), "--model", "cosmos", "--report", str(report_path)])
    assert code == 1
    report = json.loads(report_path.read_text())
    assert report["schema"] == config.SCHEMA and report["evidence_status"] == "COMPLETE"


@pytest.mark.parametrize("change", ["fill", "model_seed", "scheduler", "qsa"])
def test_correct_parent_does_not_mask_wrong_actual_qwen_worker(change):
    snapshot = _snapshot("qwen")
    worker = snapshot["comparison_identity"]["worker_runtime"][0]
    if change == "fill": worker["fill_uninitialized_memory"] = True
    elif change == "model_seed": worker["engine_config"]["model_config"]["seed"] = 1
    elif change == "scheduler": worker["engine_config"]["scheduler_config"]["max_num_seqs"] = 2
    else: worker["qsa_native_identity"]["replacement_enabled"] = True
    with pytest.raises(ValueError):
        config.validate_runtime_policy(snapshot, "qwen")


def test_worker_rpc_observes_flags_and_engine_without_any_setter(monkeypatch):
    expected = _snapshot("nemo", "native_nemo")["comparison_identity"]["worker_runtime"][0]
    engine = expected["engine_config"]
    torch = SimpleNamespace(initial_seed=lambda: 0, are_deterministic_algorithms_enabled=lambda: False,
        is_deterministic_algorithms_warn_only_enabled=lambda: False,
        utils=SimpleNamespace(deterministic=SimpleNamespace(fill_uninitialized_memory=True)),
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
            cudnn=SimpleNamespace(benchmark=False, deterministic=False, allow_tf32=True)),
        cuda=SimpleNamespace(get_device_properties=lambda index: SimpleNamespace(name="test GPU")))
    monkeypatch.setitem(sys.modules, "torch", torch)
    worker = SimpleNamespace(get_model=lambda: SimpleNamespace(),
        vllm_config=SimpleNamespace(**{key: SimpleNamespace(**value) for key, value in engine.items()}))
    observed = runtime._worker_runtime(worker)
    assert observed["engine_config"] == engine and observed["fill_uninitialized_memory"] is True
    assert observed["deterministic_algorithms"] is False
    assert observed["fp32_precision"] == {"allow_tf32_matmul": True, "allow_tf32_cudnn": True}
