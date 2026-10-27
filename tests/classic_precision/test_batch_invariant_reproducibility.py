"""Focused CPU checks for the single official BI switch; no model or GPU load."""
from copy import deepcopy
import functools
import importlib.util
import json
import marshal
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from classic_precision import batch_invariant_reproducibility as minimal
from classic_precision.storage import atomic_json, digest, file_sha256

runtime = minimal.runtime
OPERATOR = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/operator/batch_invariant_control_queue.py")


def _helpers(version=""):
    path = Path(__file__).with_name("test_minimal_config_reproducibility" + version + ".py")
    spec = importlib.util.spec_from_file_location("bi_test_helpers" + version, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile(mode):
    plan = minimal.build_profile(mode)
    result = {key: deepcopy(value) for key, value in plan.items() if key not in ("id", "implementation")}
    result.update(profile_id=plan["id"], family="qwen", sdk_files_modified=False,
        startup_environment={**plan["baseline_environment"], **plan["environment_interventions"]},
        base_runner_sha256=runtime.BASE_RUNNER_SHA256, helper_sha256=file_sha256(runtime.__file__),
        native_guard_helper_sha256=file_sha256(runtime.native.__file__),
        observer_helper_sha256=file_sha256(runtime.native.observers.__file__),
        frozen_policy_helper_sha256=runtime.FROZEN_HELPER_SHA256,
        frozen_observer_helper_sha256=runtime.FROZEN_V2_HELPER_SHA256,
        qsa_installation=_helpers("_v3")._profile("qwen_tf32_off_only")["qsa_installation"])
    return result


def _batch(scope, mode):
    return {"scope": scope, "environment_value": "1" if mode == "qwen_batch_invariant" else None,
        "module": "vllm.envs", "expected_source_sha256": runtime.VLLM_ENVS_SHA256,
        "actual_source_sha256": runtime.VLLM_ENVS_SHA256, "parser_native": True, "getter_native": True,
        "parsed_value": mode == "qwen_batch_invariant", "effective_value": mode == "qwen_batch_invariant"}


def _snapshot(mode):
    value = _helpers("_v3")._snapshot("qwen_tf32_off_only")["comparison_identity"]
    profile = _profile(mode)
    parent = {**profile["startup_environment"], "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    flags = _helpers()._flags()
    receipt = {"phase": runtime.APPLICATION_PHASE, "model_loaded_before_application": True, "mode": mode,
        "declared_torch_policy": {}, "before": deepcopy(flags), "after": deepcopy(flags)}
    value.update(profile=profile, effective_environment=profile["environment"])
    for scope, target in (("parent", value), ("worker", value["worker_runtime"][0])):
        environment = dict(parent)
        if scope == "worker":
            environment.update(OMP_NUM_THREADS="6", VLLM_OMP_NUM_THREADS_SET_BY_VLLM="1")
        target.update(deepcopy(flags), application_receipt=deepcopy(receipt),
            native_postload_environment=deepcopy(environment), observed_environment=deepcopy(environment),
            batch_invariant_observation=_batch(scope, mode))
    return {"comparison_identity": value}


@pytest.mark.parametrize("numpy_value", [None, "0", "1"])
def test_exactly_one_environment_change_and_preserved_numpy_baseline(numpy_value):
    baseline = {"NUMPY_MADVISE_HUGEPAGE": numpy_value}
    native = minimal.build_profile("native_qwen", baseline)
    bi = minimal.build_profile("qwen_batch_invariant", baseline)
    assert native["baseline_environment"] == bi["baseline_environment"]
    assert native["environment_interventions"] == {} and bi["environment_interventions"] == {runtime.FLAG: "1"}
    assert native["environment"] == {key: value for key, value in bi["environment"].items() if key != runtime.FLAG}
    assert native["cleared_environment"] == sorted(bi["cleared_environment"] + [runtime.FLAG])
    assert native["baseline_environment"]["NUMPY_MADVISE_HUGEPAGE"] == numpy_value
    assert native["torch_policy"] == bi["torch_policy"] == {}
    with pytest.raises(ValueError, match="absent"):
        minimal.build_profile("qwen_batch_invariant", {runtime.FLAG: "0"})


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_no_torch_setters_and_exact_native_runtime_accepted(mode):
    torch, state = _helpers()._fake_torch()  # no seed/thread setters exist
    receipt = runtime._base._apply(torch, mode)
    assert receipt["before"] == receipt["after"] and state["calls"] == []
    minimal._base._validate_application({**receipt["after"], "application_receipt": receipt}, mode)
    minimal.validate_runtime_policy(_snapshot(mode), "qwen")


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
@pytest.mark.parametrize("key", [runtime.FLAG, "NUMPY_MADVISE_HUGEPAGE"])
def test_prestartup_activation_rejects_environment_change_or_lost_absence(monkeypatch, mode, key):
    profile = minimal.build_profile(mode)
    for name in profile["cleared_environment"]: monkeypatch.delenv(name, raising=False)
    for name, value in profile["environment"].items(): monkeypatch.setenv(name, value)
    monkeypatch.setenv(key, "0")
    with pytest.raises(RuntimeError, match="Actual minimal launch environment differs"):
        runtime.activate("qwen", mode)


@pytest.mark.parametrize("scope", ["parent", "worker"])
@pytest.mark.parametrize("change", ["missing", "source", "parser", "getter", "flag", "raw", "error", "torch", "numpy", "engine"])
def test_correct_parent_cannot_hide_worker_or_environment_spoofs(scope, change):
    snapshot = _snapshot("qwen_batch_invariant")
    value = snapshot["comparison_identity"]
    target = value if scope == "parent" else value["worker_runtime"][0]
    observation = target["batch_invariant_observation"]
    if change == "missing": target.pop("batch_invariant_observation")
    elif change == "source": observation["actual_source_sha256"] = "0" * 64
    elif change in ("parser", "getter"): observation[change + "_native"] = False
    elif change == "flag": observation["effective_value"] = False
    elif change == "raw": observation["environment_value"] = None
    elif change == "error": observation["observation_error"] = {"type": "RuntimeError"}
    elif change == "torch": target["application_receipt"]["after"]["cudnn_benchmark"] = False
    elif change == "numpy":
        target["observed_environment"]["NUMPY_MADVISE_HUGEPAGE"] = "0"
        target["native_postload_environment"]["NUMPY_MADVISE_HUGEPAGE"] = "0"
    else:
        value["worker_runtime"][0]["engine_config"]["model_config"]["seed"] = 1
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, "qwen")


def _envs_module(monkeypatch, tmp_path):
    path = tmp_path / "envs.py"
    path.write_text('import os\nenvironment_variables = {"VLLM_BATCH_INVARIANT": lambda: bool(int(os.getenv("VLLM_BATCH_INVARIANT", "0")))}\ndef __getattr__(name):\n    return environment_variables[name]()\n')
    module = ModuleType("vllm.envs")
    module.__file__ = str(path)
    exec(compile(path.read_text(), str(path), "exec", dont_inherit=True), module.__dict__)
    monkeypatch.setitem(sys.modules, "vllm.envs", module)
    monkeypatch.setattr(runtime, "VLLM_ENVS_SHA256", file_sha256(path))
    return module


@pytest.mark.parametrize("raw", [None, "1"])
@pytest.mark.parametrize("cached", [False, True])
def test_actual_official_style_getter_parser_and_cache_are_observed(monkeypatch, tmp_path, raw, cached):
    module = _envs_module(monkeypatch, tmp_path)
    monkeypatch.delenv(runtime.FLAG, raising=False) if raw is None else monkeypatch.setenv(runtime.FLAG, raw)
    if cached:
        module.__getattr__ = functools.cache(module.__getattr__)
        module.__getattr__(runtime.FLAG)
    result = runtime._batch_observation("worker")
    assert "observation_error" not in result
    assert result["parser_native"] is result["getter_native"] is True
    assert result["environment_value"] == raw and result["parsed_value"] == result["effective_value"] == (raw == "1")


@pytest.mark.parametrize("target", ["parser", "getter"])
@pytest.mark.parametrize("cached", [False, True])
def test_equal_live_code_with_different_marshal_reference_graph_is_accepted(monkeypatch, tmp_path, target, cached):
    module = _envs_module(monkeypatch, tmp_path)
    monkeypatch.setenv(runtime.FLAG, "1")
    function = module.environment_variables[runtime.FLAG] if target == "parser" else module.__getattr__
    original = function.__code__
    reloaded = marshal.loads(marshal.dumps(original))
    assert original == reloaded
    assert marshal.dumps(original) != marshal.dumps(reloaded)
    assert runtime._same_code(original, reloaded)
    function.__code__ = reloaded
    if cached:
        module.__getattr__ = functools.cache(module.__getattr__)
        module.__getattr__(runtime.FLAG)
    result = runtime._batch_observation("worker")
    assert "observation_error" not in result
    assert result["parser_native"] is result["getter_native"] is result["effective_value"] is True


@pytest.mark.parametrize("target", ["parser", "getter"])
@pytest.mark.parametrize("field", ["co_filename", "co_firstlineno", "co_name", "co_qualname", "co_code", "co_consts", "co_names", "co_linetable"])
def test_structural_comparison_still_rejects_live_code_and_metadata_tampering(monkeypatch, tmp_path, target, field):
    module = _envs_module(monkeypatch, tmp_path)
    monkeypatch.setenv(runtime.FLAG, "1")
    function = module.environment_variables[runtime.FLAG] if target == "parser" else module.__getattr__
    original = function.__code__
    if field in ("co_filename", "co_name", "co_qualname"):
        replacement = getattr(original, field) + "_tampered"
    elif field == "co_firstlineno":
        replacement = original.co_firstlineno + 1
    elif field == "co_consts":
        replacement = original.co_consts + (False,)
    elif field == "co_names":
        replacement = original.co_names + ("tampered",)
    elif field == "co_code":
        replacement = original.co_code + bytes([9, 0])  # extra NOP, same return value
    else:
        replacement = original.co_linetable + b"\x00"
    function.__code__ = original.replace(**{field: replacement})
    assert not runtime._same_code(original, function.__code__)
    result = runtime._batch_observation("worker")
    assert "observation_error" in result


def test_recursive_code_constant_types_and_nested_metadata_are_exact():
    outer = compile("lambda: (lambda: 1)", "native.py", "eval").co_consts[0]
    nested = next(code for code in outer.co_consts if hasattr(code, "co_code"))
    altered = nested.replace(co_consts=tuple(True if type(value) is int and value == 1 else value for value in nested.co_consts))
    assert not runtime._same_code(nested, altered)  # bool and int compare equal in Python
    assert not runtime._same_code(outer, outer.replace(co_consts=tuple(altered if value is nested else value for value in outer.co_consts)))
    altered = nested.replace(co_filename="spoof.py", co_firstlineno=nested.co_firstlineno + 1)
    assert not runtime._same_code(outer, outer.replace(co_consts=tuple(altered if value is nested else value for value in outer.co_consts)))
    assert not runtime._same_constant(0.0, -0.0)
    assert not runtime._same_constant(frozenset({True}), frozenset({1}))


@pytest.mark.parametrize("change", ["parser", "getter", "shadow", "source", "stale_cache"])
def test_live_official_observation_rejects_spoof_or_exposes_stale_cache(monkeypatch, tmp_path, change):
    module = _envs_module(monkeypatch, tmp_path)
    monkeypatch.setenv(runtime.FLAG, "1")
    if change == "parser": module.environment_variables[runtime.FLAG] = lambda: True
    elif change == "getter": module.__getattr__ = lambda name: True
    elif change == "shadow": setattr(module, runtime.FLAG, True)
    elif change == "source": Path(module.__file__).write_text("# changed\n")
    else:
        monkeypatch.delenv(runtime.FLAG)
        module.__getattr__ = functools.cache(module.__getattr__)
        assert getattr(module, runtime.FLAG) is False
        monkeypatch.setenv(runtime.FLAG, "1")
    observed = runtime._batch_observation("worker")
    if change == "stale_cache":
        assert observed["parsed_value"] is True and observed["effective_value"] is False
    else:
        assert "observation_error" in observed


def test_actual_worker_rpc_import_and_parent_snapshot_use_observed_not_prefilled_data(monkeypatch, tmp_path):
    mode, events = "qwen_batch_invariant", []
    v3 = runtime._v3
    profile = _profile(mode)
    for key in profile["cleared_environment"]: monkeypatch.delenv(key, raising=False)
    for key, value in profile["environment"].items(): monkeypatch.setenv(key, value)
    torch, calls = _helpers()._fake_torch()
    torch.get_num_threads = lambda: 6
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(v3.os, "sched_getaffinity", lambda pid: set(range(48)), raising=False)
    monkeypatch.setattr(v3, "_engine_startup_source", _helpers("_v3")._source)
    for name in ("_launch_parent", "_parent_thread_baseline"): monkeypatch.setattr(v3, name, {})
    monkeypatch.setattr(v3, "_persisted_stores", set())
    for name in ("_application", "_postload_environment"): monkeypatch.setattr(runtime._base, name, {})
    monkeypatch.setattr(runtime._base, "_state", profile)
    session = SimpleNamespace(family="qwen", store=SimpleNamespace(directory=tmp_path,
        event=lambda name, **fields: events.append((name, deepcopy(fields)))))
    worker = SimpleNamespace()
    monkeypatch.setattr(runtime._base, "_session", session)
    monkeypatch.setattr(v3, "_thread_observation", lambda scope: _helpers("_v3")._threads(scope, {
        "OMP_NUM_THREADS": runtime.os.environ.get("OMP_NUM_THREADS"),
        "VLLM_OMP_NUM_THREADS_SET_BY_VLLM": runtime.os.environ.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM")}))
    runtime.status()
    assert events[0][0] == "REPRO_ENGINE_LAUNCH_PARENT" and not vars(worker)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    observed_scopes = []
    def actual_batch(scope):
        observed_scopes.append(scope)
        result = _batch(scope, mode)
        result["environment_value"] = runtime.os.environ.get(runtime.FLAG)
        return result
    monkeypatch.setattr(runtime, "_batch_observation", actual_batch)
    class Engine:
        def collective_rpc(self, function, args=()):
            assert function.__module__ == runtime.__name__
            with monkeypatch.context() as environment:
                environment.setenv("OMP_NUM_THREADS", "6")
                environment.setenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "1")
                return [function(worker, *args)]
    session.engine = Engine()
    monkeypatch.setattr(runtime.native, "prepare_native", lambda family: None)
    monkeypatch.setattr(runtime.native, "verify_native_qsa", lambda: None)
    monkeypatch.setattr(runtime.native, "_worker_runtime", lambda target: {
        **_snapshot(mode)["comparison_identity"]["worker_runtime"][0], **runtime.native._torch_flags(torch)})
    runtime.prepare_native("qwen")
    def original_snapshot():
        value = _snapshot(mode)["comparison_identity"]
        value.update(runtime.native._torch_flags(torch), application_receipt=deepcopy(runtime._base._application),
            native_postload_environment=deepcopy(runtime._base._postload_environment),
            observed_environment=deepcopy(runtime._base._postload_environment),
            worker_runtime=session.engine.collective_rpc(runtime._worker_runtime))
        return {"comparison_identity": value}
    monkeypatch.setattr(runtime, "_original_snapshot", original_snapshot)
    snapshot = runtime.runtime_snapshot()
    minimal.validate_runtime_policy(snapshot, "qwen")
    assert {"parent", "worker"} <= set(observed_scopes) and calls["calls"] == []
    assert events[-1][0] == "REPRO_RUNTIME_OBSERVED"
    assert events[-1][1]["snapshot"] == snapshot


@pytest.fixture
def panel(tmp_path):
    helpers = _helpers()
    path, old, output = helpers.panel.__wrapped__(tmp_path)
    plan = minimal.prepare_plan(old)
    atomic_json(path, plan)
    helpers.minimal, helpers._profile, helpers._snapshot = minimal, _profile, _snapshot
    helpers._populate((path, plan, output))
    for cold in range(2):
        for repeat in range(2):
            path_events = minimal._base.original._repeat_directory(output, cold, repeat) / "qwen/events.jsonl"
            events = minimal._base.original._read_jsonl(path_events)
            bound = next(i for i, event in enumerate(events) if event.get("event") == "REPRO_RUNTIME")
            additions = [] if repeat else [{"event": "REPRO_ENGINE_LAUNCH_PARENT", "receipt": _helpers("_v3")._launch()}, {"event": "MODEL_READY"}]
            additions += [{"event": "REPRO_RUNTIME_OBSERVED", "evidence_contract": runtime.EVIDENCE_CONTRACT, "snapshot": events[bound]["snapshot"]}]
            events[bound:bound] = additions
            path_events.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path, plan, output


@pytest.mark.parametrize("change,expected", [(None, "PASS"), ("tokens", "DIFFERENT"), ("input_hash", "INVALID_EVIDENCE"),
    ("missing_launch", "INVALID_EVIDENCE"), ("worker_bi", "INVALID_EVIDENCE"), ("owner", "INVALID_EVIDENCE")])
def test_full_196_output_comparison_keeps_output_difference_and_control_failure_separate(panel, change, expected):
    directory = minimal._base.original._repeat_directory(panel[2], 1, 0) / "qwen"
    name = "outputs.jsonl" if change == "tokens" else "events.jsonl"
    path = directory / name
    rows = minimal._base.original._read_jsonl(path)
    if change == "tokens": rows[0]["output_token_ids"] = [10, 12]
    elif change == "input_hash":
        next(row for row in rows if row.get("event") == "REPRO_INPUT_HASH")["tensors"][0]["sha256"] = "0" * 64
    elif change == "missing_launch": rows = [row for row in rows if row.get("event") != "REPRO_ENGINE_LAUNCH_PARENT"]
    elif change == "worker_bi":
        next(row for row in rows if row.get("event") == "REPRO_CALL_RUNTIME")["snapshot"]["comparison_identity"]["worker_runtime"][0]["batch_invariant_observation"]["effective_value"] = False
    elif change == "owner": rows[0]["owner"]["pid"] = 100
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = minimal.compare(panel[0], panel[2])
    assert report["models"]["qwen"]["comparison_status"] == expected
    assert report["schema"] == minimal.SCHEMA and panel[1]["total_model_calls"] == 196


def _controller():
    spec = importlib.util.spec_from_file_location("bi_controller_tests", OPERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("outcome", ["PASS", "DIFFERENT", "INVALID_EVIDENCE"])
def test_controller_accepts_complete_different_and_rejects_invalid(outcome):
    queue = _controller().queue
    plan = {"schema": minimal.SCHEMA}
    complete = outcome != "INVALID_EVIDENCE"
    status = "PASS" if outcome == "PASS" else "FAIL"
    evidence = "COMPLETE" if complete else "INVALID_OR_INCOMPLETE"
    report = {"schema": minimal.SCHEMA, "plan_sha256": digest(plan), "status": status, "evidence_status": evidence,
        "models": {"qwen": {"status": status, "evidence_status": evidence, "complete_evidence_stores": 4,
            "control_error": None if complete else "invalid", "comparison_status": outcome}}}
    if complete:
        assert queue.classify_comparison(report, int(status != "PASS"), plan, "qwen") == outcome
    else:
        with pytest.raises(RuntimeError, match="invalid or incomplete"):
            queue.classify_comparison(report, 1, plan, "qwen")


def test_controller_checks_actual_plan_and_environment_without_rewriting_it(tmp_path):
    helpers = _helpers()
    path, old, _ = helpers.panel.__wrapped__(tmp_path)
    plan = minimal.prepare_plan(old)
    plan.update(launch_environment=plan["profile"]["environment"], launch_cleared_environment=plan["profile"]["cleared_environment"])
    atomic_json(path, plan)
    before = path.read_bytes()
    job = {"plan": str(path), "model": "qwen", "output": str(tmp_path / "new-output"), "report": str(tmp_path / "report.json"),
        "env": plan["launch_environment"], "clear_env": plan["launch_cleared_environment"]}
    spec = {"python": sys.executable, "cwd": str(tmp_path), "code": str(Path(minimal.__file__).parents[1]),
        "model_root": str(tmp_path), "gpu": "0", "jobs": [job]}
    controller = _controller()
    controller.validate_spec(spec)
    assert path.read_bytes() == before
    spec["jobs"][0]["env"] = {**job["env"], "NUMPY_MADVISE_HUGEPAGE": "0"}
    with pytest.raises(RuntimeError, match="environment differs"):
        controller.validate_spec(spec)
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid", [False, True])
def test_controller_finite_execute_preserves_different_or_stops_invalid_without_retry(monkeypatch, tmp_path, invalid):
    _, old, _ = _helpers().panel.__wrapped__(tmp_path)
    jobs = []
    for mode in ("qwen_batch_invariant", "native_qwen"):
        plan = minimal.prepare_plan(old, mode)
        plan.update(launch_environment=plan["profile"]["environment"], launch_cleared_environment=plan["profile"]["cleared_environment"])
        path = tmp_path / (mode + ".json")
        atomic_json(path, plan)
        jobs.append({"plan": str(path), "model": "qwen", "output": str(tmp_path / (mode + "-outputs")),
            "report": str(tmp_path / (mode + "-report.json")), "env": plan["launch_environment"],
            "clear_env": plan["launch_cleared_environment"]})
    spec = {"python": sys.executable, "cwd": str(tmp_path), "code": str(Path(minimal.__file__).parents[1]),
        "model_root": str(tmp_path), "gpu": "0", "jobs": jobs}
    spec_path, state_path = tmp_path / "spec.json", tmp_path / "state.json"
    atomic_json(spec_path, spec)
    queue, commands = _controller().queue, []
    monkeypatch.setattr(queue, "owner", lambda pid=None: {"pid": pid or 100, "startticks": "10", "process_state": "S"})
    class Process:
        pid = 101
        returncode = None
        def __init__(self, command, **kwargs):
            commands.append(command)
            assert command[3] == queue.MODULE
            self.phase = command[4]
            self.plan = json.loads(Path(command[command.index("--plan") + 1]).read_text())
            self.output = Path(command[command.index("--output") + 1])
            if self.phase == "run":
                self.cold = int(command[command.index("--pass-index") + 1])
            else:
                self.report = Path(command[command.index("--report") + 1])
        def wait(self, timeout):
            if self.phase == "run":
                atomic_json(self.output / f"cold-{self.cold}/qwen.json", {"status": "COMPLETE", "model": "qwen",
                    "pass_index": self.cold, "plan_sha256": digest(self.plan)})
                self.returncode = 0
            else:
                first = self.plan["profile"]["mode"] == "qwen_batch_invariant"
                comparison = "INVALID_EVIDENCE" if first and invalid else "DIFFERENT" if first else "PASS"
                evidence = "INVALID_OR_INCOMPLETE" if comparison == "INVALID_EVIDENCE" else "COMPLETE"
                status = "PASS" if comparison == "PASS" else "FAIL"
                atomic_json(self.report, {"schema": self.plan["schema"], "plan_sha256": digest(self.plan),
                    "status": status, "evidence_status": evidence, "models": {"qwen": {"status": status,
                        "evidence_status": evidence, "complete_evidence_stores": 4,
                        "control_error": "invalid" if invalid and first else None, "comparison_status": comparison}}})
                self.returncode = int(status != "PASS")
            return self.returncode
        def poll(self): return self.returncode
    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    if invalid:
        with pytest.raises(RuntimeError, match="invalid or incomplete"):
            queue.execute(spec_path, state_path)
        state = json.loads(state_path.read_text())
        assert state["status"] == "FAILED" and len(state["jobs"]) == 1 and len(commands) == 3
    else:
        state = queue.execute(spec_path, state_path)
        assert state["status"] == "COMPLETE" and len(commands) == 6
        assert [job["comparison_status"] for job in state["jobs"]] == ["DIFFERENT", "PASS"]


def test_worker_fresh_import_installs_modes_without_importing_torch_or_mutating_frozen_globals():
    code = '''
import sys
from classic_precision import minimal_config_reproducibility_v3 as v3
from runtime_compat import classic_precision_minimal_config_v3 as r3
before = (v3.SCHEMA, v3.runtime, r3.PROFILE_MODES, r3._base.PROFILE_IDS.copy(), r3._base._worker_runtime)
from classic_precision import batch_invariant_reproducibility as bi
r = bi.runtime
assert before == (v3.SCHEMA, v3.runtime, r3.PROFILE_MODES, r3._base.PROFILE_IDS.copy(), r3._base._worker_runtime)
assert r._worker_apply.__module__ == "runtime_compat.classic_precision_batch_invariant"
assert r._v3._worker_apply is r._worker_apply
assert r._base.PROFILE_MODES == ("native_qwen", "qwen_batch_invariant")
assert all(r._base.TORCH_POLICIES[mode] == {} for mode in r.PROFILE_MODES)
assert "torch" not in sys.modules and "vllm" not in sys.modules
'''
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(minimal.__file__).parents[1], check=True)
