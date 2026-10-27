"""CPU regression for the observed UniProc OMP6 path and pre-launch receipts."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import minimal_config_reproducibility_v3 as minimal
from classic_precision.storage import atomic_json, file_sha256

runtime = minimal.runtime
REAL_V2_EVENTS = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/remote-fourier/minimal-outputs-v2/qwen_tf32_off_only/cold-0/repeat-0/qwen/events.jsonl")


def _helpers(version="v2"):
    suffix = "_" + version if version else ""
    path = Path(__file__).with_name("test_minimal_config_reproducibility" + suffix + ".py")
    spec = importlib.util.spec_from_file_location("v3_readonly_helpers" + suffix, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _source():
    return {"module": runtime.ENGINE_STARTUP_MODULE, "expected_sha256": runtime.ENGINE_STARTUP_SHA256,
        "actual_sha256": runtime.ENGINE_STARTUP_SHA256, "function": "_configure_uniproc_startup_threads"}


def _profile(mode):
    value = _helpers()._profile(mode)
    value.update(profile_id=runtime.PROFILE_IDS[mode], helper_sha256=file_sha256(runtime.__file__),
        frozen_observer_helper_sha256=runtime.FROZEN_V2_HELPER_SHA256,
        evidence_contract=runtime.EVIDENCE_CONTRACT, engine_startup_source_sha256=runtime.ENGINE_STARTUP_SHA256)
    return value


def _launch(threads=6):
    return {"phase": "after_native_processor_before_engine_load", "origin": "launching_parent_process", "family": "qwen",
        "inference_sha256": runtime.BASE_RUNNER_SHA256, "engine_startup_source": _source(),
        "cpu_affinity": list(range(48)), "torch_num_threads": threads,
        "omp_num_threads": None, "vllm_omp_marker": None, "local_engine_count": 1}


def _threads(scope, environment):
    value = _helpers()._threads(scope, environment)
    value.update(cpu_affinity=list(range(48)), cgroup_cpu_limit=None)
    return value


def _snapshot(mode):
    snapshot = _helpers()._snapshot(mode)
    value = snapshot["comparison_identity"]
    profile = _profile(mode)
    parent_env = {**profile["startup_environment"], "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    worker_env = {**parent_env, "OMP_NUM_THREADS": "6", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM": "1"}
    value.update(profile=profile, effective_environment=profile["environment"], native_postload_environment=deepcopy(parent_env),
        observed_environment=deepcopy(parent_env), native_thread_baseline=_threads("parent", parent_env),
        native_thread_observation=_threads("parent", parent_env), engine_launch_parent=_launch(), engine_startup_source=_source())
    worker = value["worker_runtime"][0]
    worker.update(native_postload_environment=deepcopy(worker_env), observed_environment=deepcopy(worker_env),
        native_thread_baseline=_threads("worker", worker_env), native_thread_observation=_threads("worker", worker_env),
        engine_launch_parent=_launch(), engine_startup_source=_source())
    worker["engine_config"]["parallel_config"]["distributed_executor_backend"] = "uni"
    return snapshot


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_v3_all_runtime_validators_accept_exact_native_uniproc6_formula(mode):
    value = minimal.validate_runtime_policy(_snapshot(mode), "qwen")
    assert value["worker_runtime"][0]["observed_environment"]["OMP_NUM_THREADS"] == "6"
    profile = minimal.build_profile(mode)
    assert profile["id"].endswith("-v3") and profile["environment_interventions"] == {}
    assert profile["torch_policy"] == runtime.TORCH_POLICIES[mode]


@pytest.mark.skipif(not REAL_V2_EVENTS.is_file(), reason="Local preserved zero-call v2 evidence is not available")
def test_real_preserved_v2_snapshot_proves_formula_but_not_missing_v3_launch_receipt():
    before = file_sha256(REAL_V2_EVENTS)
    events = [json.loads(line) for line in REAL_V2_EVENTS.read_text().splitlines()]
    snapshot = next(event["snapshot"] for event in events if event.get("event") == "REPRO_RUNTIME_OBSERVED")
    value, worker = snapshot["comparison_identity"], snapshot["comparison_identity"]["worker_runtime"][0]
    assert worker["engine_config"]["parallel_config"]["distributed_executor_backend"] == "uni"
    assert value["native_thread_observation"]["torch_num_threads"] == 6
    assert worker["native_thread_observation"]["native_startup_omp_num_threads"] == 48
    assert worker["observed_environment"]["OMP_NUM_THREADS"] == str(min(6, 48))
    assert worker["native_thread_observation"]["torch_num_threads"] == 1
    # Identity/source witnesses below are synthetic test fields. None is written
    # back to the real old evidence or presented as a recovered launch receipt.
    adapted = deepcopy(snapshot)
    adapted["comparison_identity"]["profile"] = _profile("qwen_tf32_off_only")
    adapted["comparison_identity"]["profile"]["qsa_installation"] = deepcopy(value["profile"]["qsa_installation"])
    for target in (adapted["comparison_identity"], adapted["comparison_identity"]["worker_runtime"][0]):
        target["engine_startup_source"] = _source()
    with pytest.raises(ValueError, match="pre-engine parent observation missing"):
        minimal.validate_runtime_policy(adapted, "qwen")
    for target in (adapted["comparison_identity"], adapted["comparison_identity"]["worker_runtime"][0]):
        target["engine_launch_parent"] = _launch()
    minimal.validate_runtime_policy(adapted, "qwen")
    assert file_sha256(REAL_V2_EVENTS) == before


@pytest.mark.parametrize("change", ["missing_launch", "wrong_phase", "wrong_origin", "wrong_launch_threads", "source",
    "backend", "omp48", "marker", "worker_threads", "parent_context", "affinity", "cublas"])
def test_launch_provenance_formula_and_untouched_flags_fail_closed(change):
    snapshot = _snapshot("qwen_tf32_off_only")
    value, worker = snapshot["comparison_identity"], snapshot["comparison_identity"]["worker_runtime"][0]
    if change == "missing_launch": worker.pop("engine_launch_parent")
    elif change == "wrong_phase": worker["engine_launch_parent"]["phase"] = "after_load"
    elif change == "wrong_origin": worker["engine_launch_parent"]["origin"] = "worker"
    elif change == "wrong_launch_threads": worker["engine_launch_parent"]["torch_num_threads"] = 7
    elif change == "source": worker["engine_startup_source"]["actual_sha256"] = "0" * 64
    elif change == "backend": worker["engine_config"]["parallel_config"]["distributed_executor_backend"] = "mp"
    elif change in ("omp48", "marker", "cublas"):
        key, replacement = {"omp48": ("OMP_NUM_THREADS", "48"), "marker": ("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", None),
            "cublas": ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}[change]
        worker["observed_environment"][key] = worker["native_postload_environment"][key] = replacement
        for name in ("native_thread_baseline", "native_thread_observation"):
            if change == "omp48": worker[name]["omp_num_threads"] = replacement
            elif change == "marker": worker[name]["vllm_omp_marker"] = replacement
    elif change == "worker_threads":
        for name in ("native_thread_baseline", "native_thread_observation"): worker[name]["torch_num_threads"] = 6
    elif change == "parent_context":
        value["observed_environment"]["OMP_NUM_THREADS"] = value["native_postload_environment"]["OMP_NUM_THREADS"] = "6"
        for name in ("native_thread_baseline", "native_thread_observation"): value[name]["omp_num_threads"] = "6"
    else: worker["engine_launch_parent"]["cpu_affinity"] = list(range(47))
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, "qwen")


def test_worker_cannot_substitute_a_different_individually_consistent_parent_receipt():
    snapshot = _snapshot("qwen_tf32_off_only")
    worker = snapshot["comparison_identity"]["worker_runtime"][0]
    worker["engine_launch_parent"]["torch_num_threads"] = 7
    worker["observed_environment"]["OMP_NUM_THREADS"] = worker["native_postload_environment"]["OMP_NUM_THREADS"] = "7"
    for name in ("native_thread_baseline", "native_thread_observation"):
        worker[name]["omp_num_threads"] = "7"
    with pytest.raises(ValueError, match="launching-parent receipt differs"):
        minimal.validate_runtime_policy(snapshot, "qwen")


def test_actual_status_rpc_and_snapshot_bind_parent_receipt_without_fixture_worker_prefill(monkeypatch, tmp_path):
    mode, events = "qwen_tf32_off_only", []
    profile = _profile(mode)
    for key in profile["cleared_environment"]:
        monkeypatch.delenv(key, raising=False)
    for key, value in profile["environment"].items():
        monkeypatch.setenv(key, value)
    torch, _ = _helpers("")._fake_torch()
    torch.backends.cudnn.benchmark = False
    torch.get_num_threads = lambda: 6
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(range(48)), raising=False)
    monkeypatch.setattr(runtime, "_engine_startup_source", _source)
    monkeypatch.setattr(runtime, "_launch_parent", {})
    monkeypatch.setattr(runtime, "_parent_thread_baseline", {})
    monkeypatch.setattr(runtime, "_persisted_stores", set())
    for name in ("_application", "_postload_environment"):
        monkeypatch.setattr(runtime._base, name, {})
    monkeypatch.setattr(runtime._base, "_state", profile)
    sink = SimpleNamespace(directory=tmp_path, event=lambda name, **fields: events.append((name, deepcopy(fields))))
    session, worker = SimpleNamespace(family="qwen", store=sink), SimpleNamespace()
    monkeypatch.setattr(runtime._base, "_session", session)
    monkeypatch.setattr(runtime, "_thread_observation", lambda scope: _threads(scope, {
        "OMP_NUM_THREADS": runtime.os.environ.get("OMP_NUM_THREADS"),
        "VLLM_OMP_NUM_THREADS_SET_BY_VLLM": runtime.os.environ.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM")}))
    runtime.status()  # actual pre-engine hook; no worker receipt is prefilled
    assert events[0][0] == "REPRO_ENGINE_LAUNCH_PARENT" and not hasattr(worker, "_classic_precision_v3_engine_launch_parent")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    class Engine:
        def collective_rpc(self, function, args=()):
            with monkeypatch.context() as scope:
                scope.setenv("OMP_NUM_THREADS", "6")
                scope.setenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "1")
                return [function(worker, *args)]

    session.engine = Engine()
    monkeypatch.setattr(runtime.native, "prepare_native", lambda family: None)
    monkeypatch.setattr(runtime.native, "verify_native_qsa", lambda: None)
    def native_worker(target):
        value = _snapshot(mode)["comparison_identity"]["worker_runtime"][0]
        value.pop("engine_launch_parent")
        value.update(runtime.native._torch_flags(torch))
        return value
    monkeypatch.setattr(runtime.native, "_worker_runtime", native_worker)
    runtime.prepare_native("qwen")
    def original_snapshot():
        value = _snapshot(mode)["comparison_identity"]
        value.update(runtime.native._torch_flags(torch), application_receipt=deepcopy(runtime._base._application),
            native_postload_environment=deepcopy(runtime._base._postload_environment),
            observed_environment=deepcopy(runtime._base._postload_environment),
            worker_runtime=session.engine.collective_rpc(runtime._worker_runtime))
        return {"comparison_identity": value}
    monkeypatch.setattr(runtime, "_original_snapshot", original_snapshot)
    observed = runtime.runtime_snapshot()
    minimal.validate_runtime_policy(observed, "qwen")
    returned = observed["comparison_identity"]["worker_runtime"][0]["engine_launch_parent"]
    assert returned == events[0][1]["receipt"] == runtime._launch_parent
    assert returned["origin"] == "launching_parent_process"
    assert events[-1][0] == "REPRO_RUNTIME_OBSERVED"


@pytest.fixture
def panel(tmp_path):
    helpers = _helpers("")
    path, old, output = helpers.panel.__wrapped__(tmp_path)
    plan = minimal.prepare_plan(old)
    atomic_json(path, plan)
    helpers.minimal, helpers._profile, helpers._snapshot = minimal, _profile, _snapshot
    helpers._populate((path, plan, output))
    for cold in range(2):
        for repeat in range(2):
            event_path = minimal._base.original._repeat_directory(output, cold, repeat) / "qwen/events.jsonl"
            events = minimal._base.original._read_jsonl(event_path)
            bound = next(i for i, event in enumerate(events) if event.get("event") == "REPRO_RUNTIME")
            additions = [] if repeat else [{"event": "REPRO_ENGINE_LAUNCH_PARENT", "receipt": _launch()}, {"event": "MODEL_READY"}]
            additions += [{"event": "REPRO_RUNTIME_OBSERVED", "evidence_contract": runtime.EVIDENCE_CONTRACT, "snapshot": events[bound]["snapshot"]}]
            events[bound:bound] = additions
            event_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return path, plan, output


@pytest.mark.parametrize("change", [None, "tokens", "missing_launch_event", "late_launch", "late_loaded", "missing_raw"])
def test_full_four_store_comparison_requires_durable_prelaunch_and_raw_receipts(panel, change):
    if change:
        path = minimal._base.original._repeat_directory(panel[2], 1, 0) / "qwen" / ("outputs.jsonl" if change == "tokens" else "events.jsonl")
        rows = minimal._base.original._read_jsonl(path)
        if change == "tokens": rows[0]["output_token_ids"] = [10, 12]
        elif change == "missing_launch_event": rows = [row for row in rows if row.get("event") != "REPRO_ENGINE_LAUNCH_PARENT"]
        elif change == "missing_raw": rows = [row for row in rows if row.get("event") != "REPRO_RUNTIME_OBSERVED"]
        else:
            event = next(row for row in rows if row.get("event") == ("MODEL_READY" if change == "late_loaded" else "REPRO_ENGINE_LAUNCH_PARENT"))
            rows.remove(event)
            rows.append(event)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = minimal.compare(panel[0], panel[2])
    assert report["schema"] == minimal.SCHEMA
    assert report["models"]["qwen"]["comparison_status"] == ("PASS" if change is None else "DIFFERENT" if change == "tokens" else "INVALID_EVIDENCE")


def test_import_preserves_v1_v2_globals_and_does_not_import_torch_or_vllm():
    code = """
import sys
from classic_precision import minimal_config_reproducibility as v1
from classic_precision import minimal_config_reproducibility_v2 as v2
from runtime_compat import classic_precision_minimal_config_v2 as r2
before=(v1.SCHEMA,v1.runtime,v2.SCHEMA,v2.runtime,r2.PROFILE_IDS.copy(),r2._base.status)
from classic_precision import minimal_config_reproducibility_v3 as v3
assert before==(v1.SCHEMA,v1.runtime,v2.SCHEMA,v2.runtime,r2.PROFILE_IDS.copy(),r2._base.status)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert v3.runtime.status()=={} and r2.status()=={}
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(minimal.__file__).parents[1], check=True)
