"""CPU regression for native loading plus only the reviewed QSA selector."""
from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from classic_precision import qsa_only_reproducibility as minimal
from classic_precision.storage import atomic_json, file_sha256, digest

runtime = minimal.runtime
BASELINE = {"NUMPY_MADVISE_HUGEPAGE": "0", "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB": "16", "PATH": "/qy/bin:/usr/bin:/bin"}
OPERATOR = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/operator/qsa_only_control_queue.py")


def _old():
    path = Path(__file__).with_name("test_batch_invariant_reproducibility.py")
    spec = importlib.util.spec_from_file_location("qsa_only_native_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _proof():
    return {"selector_source_sha256": runtime.QSA_HELPER_SHA256, "native_source_sha256": runtime.native.QSA_NATIVE_SOURCE_SHA256,
        "application_phase": runtime.APPLICATION_PHASE, "wrapper_code_exact": True, "original_code_exact": True,
        "selector_code_exact": True, "shape_checks_code_exact": True, "original_object_stable": True,
        "replacement_object_stable": True, "selector_object_stable": True, "reviewed_hook_only": True}


def _profile(mode):
    profile = _old()._profile("native_qwen")
    plan = minimal.build_profile(mode, BASELINE)
    original = deepcopy(profile["qsa_installation"])
    profile.update({key: deepcopy(value) for key, value in plan.items() if key not in ("id", "implementation")})
    profile.update(profile_id=plan["id"], startup_environment={**plan["baseline_environment"], **plan["environment_interventions"]},
        helper_sha256=file_sha256(runtime.__file__))
    if mode == "qwen_qsa_only":
        profile["qsa_installation"] = runtime.selector_receipt(original, "applied")
    return profile


def _snapshot(mode="qwen_qsa_only"):
    snapshot = _old()._snapshot("native_qwen")
    value, profile = snapshot["comparison_identity"], _profile(mode)
    value.update(profile=profile, effective_environment=profile["environment"],
        qsa_replacement_module_imported=mode == "qwen_qsa_only", qsa_replacement_import_hook=mode == "qwen_qsa_only")
    for scope, target in (("parent", value), ("worker", value["worker_runtime"][0])):
        environment = {**profile["startup_environment"], "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
        if scope == "worker": environment.update(OMP_NUM_THREADS="6", VLLM_OMP_NUM_THREADS_SET_BY_VLLM="1")
        target.update(native_postload_environment=deepcopy(environment), observed_environment=deepcopy(environment),
            qsa_live_observation=_proof() if mode == "qwen_qsa_only" else None)
        target["application_receipt"]["mode"] = mode
    value["worker_runtime"][0]["qsa_native_identity"] = deepcopy(profile["qsa_installation"])
    return snapshot


def test_selector_only_changes_qsa_flag_preserves_explicit_host_baseline_and_native_mp():
    native = minimal.build_profile("native_qwen", BASELINE)
    exact = minimal.build_profile("qwen_qsa_only", BASELINE)
    assert native["baseline_environment"] == exact["baseline_environment"]
    assert exact["environment_interventions"] == {runtime.FLAG: "1"}
    assert native["environment_interventions"] == {} and native["torch_policy"] == exact["torch_policy"] == {}
    for key, value in BASELINE.items(): assert exact["environment"][key] == value
    for key in ("OMP_NUM_THREADS", "VLLM_ENABLE_V1_MULTIPROCESSING", "CUBLAS_WORKSPACE_CONFIG", "VLLM_BATCH_INVARIANT"):
        assert exact["baseline_environment"][key] is None and key in exact["cleared_environment"]
    with pytest.raises(ValueError, match="explicit NUMPY"):
        minimal.build_profile("qwen_qsa_only")
    with pytest.raises(ValueError, match="absent"):
        minimal.build_profile("qwen_qsa_only", {**BASELINE, "VLLM_BATCH_INVARIANT": "1"})


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_no_torch_setters_and_exact_runtime_receipts(mode):
    torch, state = _old()._helpers()._fake_torch()
    receipt = runtime._base._apply(torch, mode)
    assert receipt["before"] == receipt["after"] and state["calls"] == []
    minimal.validate_runtime_policy(_snapshot(mode), "qwen")


def test_pending_profile_does_not_claim_live_application_and_keeps_same_source_identity():
    applied = _profile("qwen_qsa_only")
    pending = deepcopy(applied)
    pending["qsa_installation"]["state"] = "pending_target_import"
    minimal._validate_profile(pending, "qwen", require_applied=False)
    with pytest.raises(ValueError, match="not applied"):
        minimal._validate_profile(pending, "qwen", require_applied=True)
    assert minimal._base.original._profile_version(pending) == minimal._base.original._profile_version(applied)


@pytest.mark.parametrize("scope", ["parent", "worker"])
@pytest.mark.parametrize("change", ["proof", "source", "original_code", "selector_contract", "torch", "env", "mp", "bi"])
def test_missing_or_spoofed_selector_and_changed_controls_fail_closed(scope, change):
    snapshot = _snapshot()
    value = snapshot["comparison_identity"]
    target = value if scope == "parent" else value["worker_runtime"][0]
    if change == "proof": target.pop("qsa_live_observation")
    elif change == "source": target["qsa_live_observation"]["selector_source_sha256"] = "0" * 64
    elif change == "original_code": target["qsa_live_observation"]["original_code_exact"] = False
    elif change == "selector_contract": value["profile"]["qsa_installation"]["selector_contract"]["max_sort_elements"] = 99
    elif change == "torch": target["application_receipt"]["after"]["deterministic_algorithms"] = True
    elif change == "env": target["observed_environment"]["NUMPY_MADVISE_HUGEPAGE"] = None
    elif change == "mp": value["worker_runtime"][0]["engine_config"]["parallel_config"]["distributed_executor_backend"] = "mp"
    else: target["batch_invariant_observation"]["effective_value"] = True
    with pytest.raises(ValueError): minimal.validate_runtime_policy(snapshot, "qwen")


@pytest.fixture
def live_selector(monkeypatch, tmp_path):
    name = "runtime_compat.classic_precision_qsa_determinism"
    before = sys.modules.get(name)
    selector = runtime._selector()
    monkeypatch.setattr(selector, "_finder", None)
    monkeypatch.setattr(selector, "_state", None)
    path = tmp_path / "qsa_indexer.py"
    path.write_text("def _topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):\n    return None\n")
    module = ModuleType(runtime.native.QSA_TARGET_MODULE)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec", dont_inherit=True), vars(module))
    monkeypatch.setitem(sys.modules, module.__name__, module)
    sha = file_sha256(path)
    monkeypatch.setattr(runtime.native, "QSA_NATIVE_SOURCE_SHA256", sha)
    monkeypatch.setattr(runtime.native, "_native_source_path", lambda: path)
    monkeypatch.setattr(selector, "EXPECTED_SOURCE_SHA256", sha)
    for field in ("_original_function", "_replacement_function", "_selector_function", "_last_proof"):
        monkeypatch.setattr(runtime, field, None)
    monkeypatch.setenv(runtime.FLAG, "1")
    monkeypatch.delenv("VLLM_BATCH_INVARIANT", raising=False)
    try:
        yield selector, module
    finally:
        if selector._finder in sys.meta_path: sys.meta_path.remove(selector._finder)
        if before is None: sys.modules.pop(name, None)


def test_real_reviewed_leaf_install_has_exact_wrapper_original_and_stable_objects(live_selector):
    selector, module = live_selector
    original = module._topk
    receipt = runtime._verify_qsa(module)
    assert receipt["state"] == "applied" and receipt["selector_contract"] == runtime.selector_contract()
    assert module._topk.__wrapped__ is original and runtime._last_proof == _proof()
    wrapped = module._topk
    assert runtime._verify_qsa(module) == receipt and module._topk is wrapped
    assert selector.status() == {**runtime.selector_contract(), "state": "applied"}


@pytest.mark.parametrize("change", ["wrapper_code", "wrapper_object", "original_code", "original_defaults", "algorithm", "shape_checks", "marker", "source", "hook"])
def test_actual_live_selector_tampering_is_rejected(live_selector, monkeypatch, change):
    selector, module = live_selector
    runtime._verify_qsa(module)
    if change == "wrapper_code": monkeypatch.setattr(module._topk, "__code__", module._topk.__code__.replace(co_firstlineno=1))
    elif change == "wrapper_object":
        from types import FunctionType
        clone = FunctionType(module._topk.__code__, module._topk.__globals__)
        clone.__dict__.update(module._topk.__dict__)
        clone.__module__, clone.__qualname__ = module._topk.__module__, module._topk.__qualname__
        module._topk = clone
    elif change == "original_code": monkeypatch.setattr(module._topk.__wrapped__, "__code__", module._topk.__wrapped__.__code__.replace(co_filename="spoof.py"))
    elif change == "original_defaults": monkeypatch.setattr(module._topk.__wrapped__, "__defaults__", (1,))
    elif change == "algorithm": monkeypatch.setattr(selector, "exact_qsa_topk", lambda *args: None)
    elif change == "shape_checks": monkeypatch.setattr(selector, "_shape_checks", lambda *args: None)
    elif change == "marker": monkeypatch.setattr(module._topk, selector._MARKER, {"state": "applied"})
    elif change == "source": monkeypatch.setattr(runtime, "QSA_HELPER_SHA256", "0" * 64)
    else: sys.meta_path.remove(selector._finder)
    with pytest.raises(RuntimeError): runtime._verify_qsa(module)


@pytest.fixture
def panel(tmp_path):
    native = _old()
    helpers = native._helpers()
    path, old, output = helpers.panel.__wrapped__(tmp_path)
    plan = minimal.prepare_plan(old, baseline_environment=BASELINE)
    atomic_json(path, plan)
    helpers.minimal, helpers._profile, helpers._snapshot = minimal, _profile, _snapshot
    helpers._populate((path, plan, output))
    for cold in range(2):
        for repeat in range(2):
            path_events = minimal._base.original._repeat_directory(output, cold, repeat) / "qwen/events.jsonl"
            events = minimal._base.original._read_jsonl(path_events)
            bound = next(i for i, event in enumerate(events) if event.get("event") == "REPRO_RUNTIME")
            additions = [] if repeat else [{"event": "REPRO_ENGINE_LAUNCH_PARENT", "receipt": native._helpers("_v3")._launch()}, {"event": "MODEL_READY"}]
            additions += [{"event": "REPRO_RUNTIME_OBSERVED", "evidence_contract": runtime.EVIDENCE_CONTRACT, "snapshot": events[bound]["snapshot"]}]
            events[bound:bound] = additions
            path_events.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path, plan, output


@pytest.mark.parametrize("change,expected", [(None, "PASS"), ("tokens", "DIFFERENT"), ("input", "INVALID_EVIDENCE"), ("worker", "INVALID_EVIDENCE")])
def test_full_196_call_compare_retains_valid_difference_and_invalid_control_distinction(panel, change, expected):
    directory = minimal._base.original._repeat_directory(panel[2], 1, 0) / "qwen"
    path = directory / ("outputs.jsonl" if change == "tokens" else "events.jsonl")
    rows = minimal._base.original._read_jsonl(path)
    if change == "tokens": rows[0]["output_token_ids"] = [10, 12]
    elif change == "input": next(row for row in rows if row.get("event") == "REPRO_INPUT_HASH")["tensors"][0]["sha256"] = "0" * 64
    elif change == "worker": next(row for row in rows if row.get("event") == "REPRO_CALL_RUNTIME")["snapshot"]["comparison_identity"]["worker_runtime"][0].pop("qsa_live_observation")
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert minimal.compare(panel[0], panel[2])["models"]["qwen"]["comparison_status"] == expected


def test_actual_parent_and_worker_use_reviewed_leaf_after_loading(monkeypatch, tmp_path, live_selector):
    # Real leaf installation above is exercised independently; this check drives
    # actual native observer callbacks, with mocked Torch/model metadata only.
    selector, module = live_selector
    v3 = runtime._v3
    profile = _profile("qwen_qsa_only")
    for key in profile["cleared_environment"]: monkeypatch.delenv(key, raising=False)
    for key, value in profile["environment"].items(): monkeypatch.setenv(key, value)
    torch, calls = _old()._helpers()._fake_torch()
    torch.get_num_threads = lambda: 6
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(v3.os, "sched_getaffinity", lambda pid: set(range(48)), raising=False)
    monkeypatch.setattr(v3, "_engine_startup_source", _old()._helpers("_v3")._source)
    for field in ("_launch_parent", "_parent_thread_baseline"): monkeypatch.setattr(v3, field, {})
    monkeypatch.setattr(v3, "_persisted_stores", set())
    for field in ("_application", "_postload_environment"): monkeypatch.setattr(runtime._base, field, {})
    monkeypatch.setattr(runtime._base, "_state", profile)
    events = []
    session = SimpleNamespace(family="qwen", store=SimpleNamespace(directory=tmp_path, event=lambda name, **fields: events.append((name, deepcopy(fields)))))
    monkeypatch.setattr(runtime._base, "_session", session)
    monkeypatch.setattr(v3, "_thread_observation", lambda scope: _old()._helpers("_v3")._threads(scope, {
        "OMP_NUM_THREADS": runtime.os.environ.get("OMP_NUM_THREADS"), "VLLM_OMP_NUM_THREADS_SET_BY_VLLM": runtime.os.environ.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM")}))
    runtime.status()
    assert not hasattr(module._topk, "__wrapped__") and runtime._last_proof is None
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    worker = SimpleNamespace()
    class Engine:
        def collective_rpc(self, function, args=()):
            assert function.__module__ == runtime.__name__
            with monkeypatch.context() as environment:
                environment.setenv("OMP_NUM_THREADS", "6")
                environment.setenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "1")
                return [function(worker, *args)]
    session.engine = Engine()
    def native_worker(target):
        runtime._verify_qsa(module)
        result = _snapshot()["comparison_identity"]["worker_runtime"][0]
        result.update(runtime.native._torch_flags(torch), qsa_native_identity=runtime._verify_qsa(module))
        return result
    monkeypatch.setattr(runtime.native, "_worker_runtime", native_worker)
    runtime.prepare_native("qwen")
    assert hasattr(module._topk, "__wrapped__") and runtime._last_proof == _proof() and calls["calls"] == []
    assert worker._classic_precision_minimal_application["mode"] == "qwen_qsa_only"
    assert events[0][0] == "REPRO_ENGINE_LAUNCH_PARENT"


def test_isolated_import_and_worker_mode_tables_do_not_apply_strict_policy():
    code = '''
import sys
from classic_precision import batch_invariant_reproducibility as old
before = (old.SCHEMA, old.runtime.PROFILE_MODES, old.runtime._base.native)
from classic_precision import qsa_only_reproducibility as qsa
r=qsa.runtime
assert before == (old.SCHEMA, old.runtime.PROFILE_MODES, old.runtime._base.native)
assert r._v3._worker_apply is r._worker_apply
assert r._base.PROFILE_MODES == ('native_qwen','qwen_qsa_only')
assert r._base.TORCH_POLICIES == {'native_qwen':{},'qwen_qsa_only':{}}
assert not any(n.rsplit('.',1)[-1] == 'classic_precision_qsa_determinism' for n in sys.modules)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
'''
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(minimal.__file__).parents[1], check=True)


def test_finite_controller_uses_new_module_and_preserves_valid_different_classification():
    spec = importlib.util.spec_from_file_location("qsa_only_queue_test", OPERATOR)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    assert controller.queue.MODULE == "classic_precision.qsa_only_reproducibility"
    assert controller.queue.PLAN_SCHEMAS == {minimal.SCHEMA}
    assert "runtime_compat/classic_precision_qsa_determinism.py" in controller.queue.CODE_FILES
    native_test = _old()
    native_test._controller = lambda: controller
    for outcome in ("PASS", "DIFFERENT", "INVALID_EVIDENCE"):
        native_test.test_controller_accepts_complete_different_and_rejects_invalid(outcome)


def _predecessor_fixture(tmp_path, outcome="DIFFERENT"):
    spec = importlib.util.spec_from_file_location("qsa_only_predecessor_test", OPERATOR)
    controller = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(controller)
    native = _old()
    _, old, _ = native._helpers().panel.__wrapped__(tmp_path)
    native_plan = native.minimal.prepare_plan(old, "native_qwen", BASELINE)
    native_plan.update(launch_environment=native_plan["profile"]["environment"], launch_cleared_environment=native_plan["profile"]["cleared_environment"])
    native_path, native_spec, report_path, state_path = [tmp_path / name for name in ("native-plan.json", "native-spec.json", "native-report.json", "native-state.json")]
    atomic_json(native_path, native_plan)
    atomic_json(native_spec, {"jobs": [{"model": "qwen", "plan": str(native_path), "report": str(report_path)}]})
    status = "PASS" if outcome == "PASS" else "FAIL"
    report = {"schema": native_plan["schema"], "plan_sha256": digest(native_plan), "status": status, "evidence_status": "COMPLETE",
        "profile_mode": "native_qwen", "models": {"qwen": {"status": status, "evidence_status": "COMPLETE",
            "comparison_status": outcome, "control_error": None, "complete_evidence_stores": 4, "samples_per_repeat": 49, "expected_repeat_stores": 4}}}
    atomic_json(report_path, report)
    predecessor = {"pid": 101, "startticks": "12", "model": "qwen", "require_status": "COMPLETE", "plan": str(native_path),
        "plan_sha256": digest(native_plan), "spec_path": str(native_spec), "spec_sha256": file_sha256(native_spec),
        "state_file": str(state_path), "report_path": str(report_path)}
    previous = {"schema": "classic_precision_finite_native_qwen_batch_invariant/v1", "owner": {"pid": 101, "startticks": "12"},
        "status": "COMPLETE", "spec": str(native_spec), "spec_sha256": file_sha256(native_spec),
        "controller_sha256": file_sha256(controller.queue.__file__), "bound_files_sha256": {
            str(native_path): file_sha256(native_path), str(native_spec): file_sha256(native_spec)},
        "jobs": [{"plan": str(native_path), "report": str(report_path), "status": "COMPLETE", "comparison_status": outcome,
            "report_sha256": file_sha256(report_path)}]}
    atomic_json(state_path, previous)
    return controller, predecessor, previous, report, old


@pytest.mark.parametrize("outcome", ["PASS", "DIFFERENT"])
@pytest.mark.parametrize("reused_pid", [False, True])
def test_predecessor_accepts_only_normal_exit_and_complete_valid_native_evidence(monkeypatch, tmp_path, outcome, reused_pid):
    controller, predecessor, _, _, _ = _predecessor_fixture(tmp_path, outcome)
    monkeypatch.setattr(controller.queue, "process_identity", lambda pid: {"startticks": "99", "process_state": "S"} if reused_pid else None)
    state = {}
    controller.wait_predecessor(predecessor, state, lambda: None, controller.time.monotonic() + 30)
    assert state["predecessor"]["comparison_status"] == outcome
    assert state["predecessor"]["original_process_exited"] is True
    assert state["predecessor"]["pid_reused"] is reused_pid


@pytest.mark.parametrize("change", ["owner", "spec", "status", "bound_source", "invalid", "stores", "denominator", "profile", "report_binding"])
def test_predecessor_failure_invalid_or_changed_identity_never_passes(monkeypatch, tmp_path, change):
    controller, predecessor, previous, report, _ = _predecessor_fixture(tmp_path)
    if change == "owner": previous["owner"]["startticks"] = "13"
    elif change == "spec": previous["spec_sha256"] = "0" * 64
    elif change == "status": previous["status"] = "FAILED"
    elif change == "bound_source": previous["bound_files_sha256"][predecessor["plan"]] = "0" * 64
    elif change == "invalid": report["models"]["qwen"]["comparison_status"] = "INVALID_EVIDENCE"
    elif change == "stores": report["models"]["qwen"]["complete_evidence_stores"] = 3
    elif change == "denominator": report["models"]["qwen"]["samples_per_repeat"] = 48
    elif change == "profile": report["profile_mode"] = "qwen_batch_invariant"
    else: previous["jobs"][0]["report_sha256"] = "0" * 64
    atomic_json(predecessor["state_file"], previous)
    atomic_json(predecessor["report_path"], report)
    monkeypatch.setattr(controller.queue, "process_identity", lambda pid: None)
    with pytest.raises(RuntimeError):
        controller.wait_predecessor(predecessor, {}, lambda: None, controller.time.monotonic() + 30)


def test_predecessor_waits_for_original_process_natural_exit(monkeypatch, tmp_path):
    controller, predecessor, _, _, _ = _predecessor_fixture(tmp_path)
    identities = iter([{"startticks": "12", "process_state": "S"}, None])
    monkeypatch.setattr(controller.queue, "process_identity", lambda pid: next(identities))
    waits = []
    monkeypatch.setattr(controller.time, "sleep", lambda duration: waits.append(duration))
    controller.wait_predecessor(predecessor, {}, lambda: None, controller.time.monotonic() + 60)
    assert len(waits) == 1 and 0 < waits[0] <= 30


def test_predecessor_spec_checks_same_inputs_and_only_qsa_launch_change(tmp_path):
    controller, predecessor, _, _, old = _predecessor_fixture(tmp_path)
    plan = minimal.prepare_plan(old, baseline_environment=BASELINE)
    plan.update(launch_environment=plan["profile"]["environment"], launch_cleared_environment=plan["profile"]["cleared_environment"])
    path = tmp_path / "qsa-plan.json"
    atomic_json(path, plan)
    job = {"model": "qwen", "plan": str(path), "output": str(tmp_path / "qsa-output"), "report": str(tmp_path / "qsa-report.json"),
        "env": plan["launch_environment"], "clear_env": plan["launch_cleared_environment"]}
    spec = {"python": sys.executable, "code": str(Path(minimal.__file__).parents[1]), "cwd": str(tmp_path),
        "model_root": str(tmp_path), "gpu": "0", "jobs": [job], "predecessor": predecessor}
    controller.validate_spec(spec)
    assert predecessor["plan"] in controller.queue.CODE_FILES and predecessor["spec_path"] in controller.queue.CODE_FILES
    plan["launch_environment"] = {**plan["launch_environment"], "PATH": "/different"}
    atomic_json(path, plan)
    job["env"] = plan["launch_environment"]
    with pytest.raises(RuntimeError, match="operational launch baseline differs"):
        controller.validate_spec(spec)
