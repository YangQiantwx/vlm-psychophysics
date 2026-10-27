"""CPU-only native thread provenance and version isolation checks."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import minimal_config_reproducibility_v2 as minimal
from classic_precision.storage import atomic_json, file_sha256

runtime = minimal.runtime


def _old_tests():
    path = Path(__file__).with_name("test_minimal_config_reproducibility.py")
    spec = importlib.util.spec_from_file_location("v2_frozen_test_helpers", path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def _profile(mode, baseline=None):
    old = _old_tests()._profile(mode)
    plan = minimal.build_profile(mode, baseline)
    for key in ("baseline_environment", "environment_interventions", "environment", "cleared_environment"):
        old[key] = deepcopy(plan[key])
    old.update(profile_id=plan["id"], helper_sha256=file_sha256(runtime.__file__),
        startup_environment={**plan["baseline_environment"], **plan["environment_interventions"]},
        evidence_contract=runtime.EVIDENCE_CONTRACT, frozen_policy_helper_sha256=runtime.FROZEN_HELPER_SHA256,
        native_thread_source_sha256=deepcopy(runtime.THREAD_SOURCE_SHA256))
    return old


def _threads(scope, environment, torch_threads=None):
    return {"scope": scope, "torch_num_threads": (1 if scope == "worker" else 6) if torch_threads is None else torch_threads,
        "omp_num_threads": environment["OMP_NUM_THREADS"], "vllm_omp_marker": environment["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"],
        "cpu_affinity": list(range(64)), "num_local_procs": 1, "cgroup_cpu_limit": 48.0,
        "native_available_cpu_count": 48, "native_startup_omp_num_threads": 48,
        "source_provenance": {key: {"expected_sha256": sha, "actual_sha256": sha} for key, sha in runtime.THREAD_SOURCE_SHA256.items()},
        "function_identities": {name: {"module": "vllm.utils.torch_utils", "qualname": name,
            "source_sha256": runtime.THREAD_SOURCE_SHA256["vllm.utils.torch_utils"], "wrapped": False}
            for name in ("startup_omp_num_threads", "available_cpu_count", "_cgroup_cpu_limit")}}


def _snapshot(mode, baseline=None):
    snapshot = _old_tests()._snapshot(mode)
    value = snapshot["comparison_identity"]
    profile = _profile(mode, baseline)
    env = deepcopy(profile["startup_environment"])
    if mode not in ("native_cosmos", "qwen_schedule_only") and env["OMP_NUM_THREADS"] is None:
        env.update(OMP_NUM_THREADS="48", VLLM_OMP_NUM_THREADS_SET_BY_VLLM="1", VLLM_WORKER_MULTIPROC_METHOD="spawn")
    value.update(profile=profile, effective_environment=profile["environment"],
        native_postload_environment=deepcopy(env), observed_environment=deepcopy(env))
    if mode != "native_cosmos":
        value.update(native_thread_baseline=_threads("parent", env), native_thread_observation=_threads("parent", env))
        worker = value["worker_runtime"][0]
        worker.update(native_postload_environment=deepcopy(env), observed_environment=deepcopy(env),
            native_thread_baseline=_threads("worker", env), native_thread_observation=_threads("worker", env))
    return snapshot


@pytest.mark.parametrize("mode", runtime.PROFILE_MODES)
def test_versioned_profile_keeps_numerical_interventions_and_seals_marker_absence(mode):
    profile = minimal.build_profile(mode)
    assert profile["id"].endswith("-v2") and profile["evidence_contract"] == runtime.EVIDENCE_CONTRACT
    assert profile["baseline_environment"]["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] is None
    assert profile["environment_interventions"] == ({"VLLM_ENABLE_V1_MULTIPROCESSING": "0"} if mode == "qwen_schedule_only" else {})
    assert profile["torch_policy"] == runtime.TORCH_POLICIES[mode]
    minimal.validate_runtime_policy(_snapshot(mode), runtime.family_for_mode(mode))


def test_parent_threads_are_observed_without_assuming_worker_value():
    snapshot = _snapshot("qwen_tf32_off_only")
    assert snapshot["comparison_identity"]["native_thread_observation"]["torch_num_threads"] == 6
    assert snapshot["comparison_identity"]["worker_runtime"][0]["native_thread_observation"]["torch_num_threads"] == 1
    minimal.validate_runtime_policy(snapshot, "qwen")


def test_outer_parent_may_keep_native_environment_while_engine_worker_sets_omp():
    snapshot = _snapshot("qwen_tf32_off_only")
    value = snapshot["comparison_identity"]
    env = deepcopy(value["profile"]["startup_environment"])
    value.update(native_postload_environment=deepcopy(env), observed_environment=deepcopy(env),
        native_thread_baseline=_threads("parent", env), native_thread_observation=_threads("parent", env))
    minimal.validate_runtime_policy(snapshot, "qwen")
    assert value["native_thread_observation"]["omp_num_threads"] is None
    assert value["worker_runtime"][0]["native_thread_observation"]["omp_num_threads"] == "48"


def test_actual_observer_reads_affinity_cgroup_and_native_functions_without_setters(monkeypatch, tmp_path):
    path = tmp_path / "torch_utils.py"
    path.write_text("def _cgroup_cpu_limit():\n    return 48.0\n"
        "def available_cpu_count():\n    return max(1, min(len(os.sched_getaffinity(0)), int(_cgroup_cpu_limit())))\n"
        "def startup_omp_num_threads(num_local_procs):\n    return max(1, available_cpu_count() // max(1, num_local_procs))\n")
    namespace = {"__name__": "vllm.utils.torch_utils", "os": runtime.os}
    exec(compile(path.read_bytes(), str(path), "exec"), namespace)
    modules = {name: SimpleNamespace(__file__=str(path)) for name in runtime.THREAD_SOURCE_SHA256}
    modules["vllm.utils.torch_utils"].startup_omp_num_threads = namespace["startup_omp_num_threads"]
    monkeypatch.setattr(runtime.importlib, "import_module", lambda name: modules[name])
    monkeypatch.setattr(runtime, "THREAD_SOURCE_SHA256", {name: file_sha256(path) for name in modules})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(get_num_threads=lambda: 6))
    monkeypatch.setattr(runtime.os, "sched_getaffinity", lambda pid: set(range(64)), raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "48")
    monkeypatch.setenv("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "1")
    observed = runtime._thread_observation("parent")
    assert "observation_error" not in observed
    assert observed["torch_num_threads"] == 6 and observed["native_startup_omp_num_threads"] == 48
    assert observed["cgroup_cpu_limit"] == 48.0 and observed["cpu_affinity"] == list(range(64))
    assert observed["function_identities"]["available_cpu_count"]["source_sha256"] == file_sha256(path)


@pytest.mark.parametrize("change", ["omp_one", "marker_missing", "omp_missing", "wrong_native_count", "affinity_missing",
    "source", "function_source", "worker_threads", "thread_drift", "unrelated_env", "no_receipt"])
def test_arbitrary_native_omp_or_unbound_provenance_is_rejected(change):
    snapshot = _snapshot("qwen_tf32_off_only")
    worker = snapshot["comparison_identity"]["worker_runtime"][0]
    if change in ("omp_one", "marker_missing", "omp_missing", "unrelated_env"):
        key, replacement = {"omp_one": ("OMP_NUM_THREADS", "1"), "marker_missing": ("VLLM_OMP_NUM_THREADS_SET_BY_VLLM", None),
            "omp_missing": ("OMP_NUM_THREADS", None), "unrelated_env": ("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}[change]
        worker["native_postload_environment"][key] = worker["observed_environment"][key] = replacement
        for label in ("native_thread_baseline", "native_thread_observation"):
            if key == "OMP_NUM_THREADS": worker[label]["omp_num_threads"] = replacement
            elif key == "VLLM_OMP_NUM_THREADS_SET_BY_VLLM": worker[label]["vllm_omp_marker"] = replacement
    elif change == "no_receipt": worker.pop("native_thread_baseline")
    else:
        for label in ("native_thread_baseline", "native_thread_observation"):
            observation = worker[label]
            if change == "wrong_native_count": observation["native_startup_omp_num_threads"] = 1
            elif change == "affinity_missing": observation["cpu_affinity"] = []
            elif change == "source": observation["source_provenance"]["vllm.utils.torch_utils"]["actual_sha256"] = "0" * 64
            elif change == "function_source": observation["function_identities"]["available_cpu_count"]["source_sha256"] = "0" * 64
            elif change == "worker_threads": observation["torch_num_threads"] = 6
        if change == "thread_drift": worker["native_thread_observation"]["torch_num_threads"] = 2
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, "qwen")


def test_native_omp_does_not_become_a_third_torch_deterministic_intervention():
    snapshot = _snapshot("qwen_torch_deterministic")
    minimal.validate_runtime_policy(snapshot, "qwen")
    value = snapshot["comparison_identity"]
    assert value["application_receipt"]["declared_torch_policy"] == runtime.TORCH_POLICIES["qwen_torch_deterministic"]
    value["application_receipt"]["after"]["cudnn_deterministic"] = True
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(snapshot, "qwen")


@pytest.fixture
def panel(tmp_path):
    helpers = _old_tests()
    path, old, output = helpers.panel.__wrapped__(tmp_path)
    plan = minimal.prepare_plan(old, "qwen_tf32_off_only")
    atomic_json(path, plan)
    helpers.minimal, helpers._profile, helpers._snapshot = minimal, _profile, _snapshot
    return helpers, (path, plan, output)


@pytest.mark.parametrize("different", [False, True])
def test_independent_v2_plan_four_stores_keep_complete_difference_classification(panel, different):
    helpers, values = panel
    helpers._populate(values)
    if different:
        path = minimal._base.original._repeat_directory(values[2], 1, 1) / "qwen/outputs.jsonl"
        rows = minimal._base.original._read_jsonl(path)
        rows[0]["output_token_ids"] = [10, 12]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = minimal.compare(values[0], values[2])
    assert report["schema"] == minimal.SCHEMA and report["evidence_status"] == "COMPLETE"
    assert report["models"]["qwen"]["comparison_status"] == ("DIFFERENT" if different else "PASS")


def test_raw_invalid_snapshot_is_persisted_before_unchanged_harness_validation(monkeypatch, tmp_path):
    snapshot = _snapshot("qwen_tf32_off_only")
    snapshot["comparison_identity"]["worker_runtime"][0]["observed_environment"]["OMP_NUM_THREADS"] = "wrong"
    events = []
    sink = SimpleNamespace(directory=tmp_path, event=lambda name, **fields: events.append((name, deepcopy(fields))))
    monkeypatch.setattr(runtime, "_original_snapshot", lambda: deepcopy(snapshot))
    monkeypatch.setattr(runtime, "_thread_observation", lambda scope: deepcopy(snapshot["comparison_identity"]["native_thread_observation"]))
    monkeypatch.setattr(runtime._base, "_session", SimpleNamespace(store=sink))
    monkeypatch.setattr(runtime._base, "_state", {"family": "qwen"})
    monkeypatch.setattr(runtime, "_parent_thread_baseline", snapshot["comparison_identity"]["native_thread_baseline"])
    monkeypatch.setattr(runtime, "_persisted_stores", set())
    observed = runtime.runtime_snapshot()
    assert events[0][0] == "REPRO_RUNTIME_OBSERVED" and events[0][1]["snapshot"] == observed
    with pytest.raises(ValueError):
        minimal.validate_runtime_policy(observed, "qwen")
    runtime.runtime_snapshot()
    assert len(events) == 1


def test_import_overlay_does_not_mutate_v1_globals_or_import_torch():
    code = """
import sys
from classic_precision import minimal_config_reproducibility as first
from runtime_compat import classic_precision_minimal_config as helper
before=(first.SCHEMA,first.runtime,first.build_profile,helper.PROFILE_IDS.copy(),helper.TRACKED_ENVIRONMENT,helper.prepare_native)
from classic_precision import minimal_config_reproducibility_v2 as second
assert before==(first.SCHEMA,first.runtime,first.build_profile,helper.PROFILE_IDS.copy(),helper.TRACKED_ENVIRONMENT,helper.prepare_native)
assert 'torch' not in sys.modules and 'vllm' not in sys.modules
assert second.runtime.status()=={} and helper.status()=={}
assert second.SCHEMA != first.SCHEMA
"""
    subprocess.run([sys.executable, "-B", "-c", code], cwd=Path(minimal.__file__).parents[1], check=True)
