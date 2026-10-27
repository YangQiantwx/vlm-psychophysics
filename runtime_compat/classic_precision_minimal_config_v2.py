"""Isolated v2 observer for native vLLM multiprocessing thread setup.

The frozen v1 policy is loaded in a private namespace. Only observations and
version identities change; numerical setters and native QSA guards are reused.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path

FROZEN_HELPER_SHA256 = "a2743fdf1feffc53e5a06ec178a3185efd73c704eee3ac3a794b8eb8ad078042"
THREAD_SOURCE_SHA256 = {
    "vllm.v1.executor.multiproc_executor": "11a1554380d1efa516bb481416c798798d5839f38ac3b0f379ef4b8b547f4d1b",
    "vllm.utils.torch_utils": "31e8a695f7efa52584ce20cc33edfb643b70c28ce83d273feb34326c61ab631c",
    "vllm.utils.system_utils": "9be361854412a4d58cc9eef6b999346c606ab11d388e2644fc47fb454546255d",
}
EVIDENCE_CONTRACT = "native-vllm-multiprocess-thread-observation-v2"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_frozen():
    path = Path(__file__).with_name("classic_precision_minimal_config.py")
    if _sha(path) != FROZEN_HELPER_SHA256:
        raise RuntimeError("Frozen v1 minimal runtime source changed")
    spec = importlib.util.spec_from_file_location("runtime_compat._minimal_v2_frozen_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_frozen()
native = _base.native
PROFILE_MODES, TORCH_POLICIES = _base.PROFILE_MODES, _base.TORCH_POLICIES
PROFILE_IDS = {mode: value.removesuffix("-v1") + "-v2" for mode, value in _base.PROFILE_IDS.items()}
BASE_RUNNER_SHA256, APPLICATION_PHASE = _base.BASE_RUNNER_SHA256, _base.APPLICATION_PHASE
TRACKED_ENVIRONMENT = tuple(sorted(set(_base.TRACKED_ENVIRONMENT) | {"VLLM_OMP_NUM_THREADS_SET_BY_VLLM"}))
_base.PROFILE_IDS, _base.TRACKED_ENVIRONMENT = PROFILE_IDS, TRACKED_ENVIRONMENT
family_for_mode, baseline_environment, environment_contract = _base.family_for_mode, _base.baseline_environment, _base.environment_contract
_original_snapshot, _original_status = _base.runtime_snapshot, _base.status
_parent_thread_baseline, _persisted_stores = {}, set()


def _function_identity(function):
    path = inspect.getsourcefile(function)
    return {"module": function.__module__, "qualname": function.__qualname__,
        "source_sha256": _sha(path) if path else None, "wrapped": hasattr(function, "__wrapped__")}


def _thread_observation(scope):
    """Read the actual process and native capacity calculation; never set threads."""
    import torch
    result = {"scope": scope, "torch_num_threads": torch.get_num_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "vllm_omp_marker": os.environ.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM"),
        "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "num_local_procs": 1, "source_provenance": {}}
    try:
        for name in THREAD_SOURCE_SHA256:
            module = importlib.import_module(name)
            result["source_provenance"][name] = {"expected_sha256": THREAD_SOURCE_SHA256[name], "actual_sha256": _sha(module.__file__)}
        utils = importlib.import_module("vllm.utils.torch_utils")
        startup = utils.startup_omp_num_threads
        available = startup.__globals__["available_cpu_count"]
        cgroup = available.__globals__["_cgroup_cpu_limit"]
        result.update(native_startup_omp_num_threads=startup(1), native_available_cpu_count=available(),
            cgroup_cpu_limit=cgroup(), function_identities={"startup_omp_num_threads": _function_identity(startup),
                "available_cpu_count": _function_identity(available), "_cgroup_cpu_limit": _function_identity(cgroup)})
    except Exception as error:
        # Preserve the observations before the validator rejects incomplete or
        # changed native source; do not lose a zero-call diagnostic snapshot.
        result["observation_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _worker_runtime(worker):
    value = native._worker_runtime(worker)
    baseline = getattr(worker, "_classic_precision_minimal_native_environment", {})
    value.update(application_receipt=deepcopy(getattr(worker, "_classic_precision_minimal_application", None)),
        native_postload_environment=deepcopy(baseline), observed_environment={key: os.environ.get(key) for key in baseline},
        native_thread_baseline=deepcopy(getattr(worker, "_classic_precision_v2_native_thread_baseline", None)),
        native_thread_observation=_thread_observation("worker"))
    return value


def _worker_apply(worker, mode, environment_keys=TRACKED_ENVIRONMENT):
    import torch
    native.verify_native_qsa()
    receipt = getattr(worker, "_classic_precision_minimal_application", None)
    if receipt is None:
        worker._classic_precision_minimal_native_environment = {key: os.environ.get(key) for key in environment_keys}
        worker._classic_precision_v2_native_thread_baseline = _thread_observation("worker")
        worker._classic_precision_minimal_application = _base._apply(torch, mode)
    elif receipt["mode"] != mode:
        raise RuntimeError("Worker already belongs to another minimal profile")
    return _worker_runtime(worker)


def prepare_native(family):
    native.prepare_native(family)
    if _base._application:
        return
    session = _base._session
    if session is None or not hasattr(session, "engine" if family == "qwen" else "model"):
        raise RuntimeError("Minimal v2 policy requires a fully loaded native model")
    import torch
    keys = tuple(sorted(set(_base._state["environment"]) | set(_base._state["cleared_environment"])))
    _base._postload_environment.update({key: os.environ.get(key) for key in keys})
    if family == "qwen":
        _parent_thread_baseline.update(_thread_observation("parent"))
    _base._application.update(_base._apply(torch, _base._state["mode"]))
    if family == "qwen":
        workers = session.engine.collective_rpc(_worker_apply, args=(_base._state["mode"], keys))
        if not workers or any(item.get("application_receipt") is None for item in workers):
            raise RuntimeError("Actual worker v2 application receipt is missing")


def status():
    return _original_status()


def activate(family, mode="native_cosmos", environment=None):
    _base.activate(family, mode, environment)
    _base._state.update(helper_sha256=_sha(__file__), frozen_policy_helper_sha256=FROZEN_HELPER_SHA256,
        evidence_contract=EVIDENCE_CONTRACT, native_thread_source_sha256=deepcopy(THREAD_SOURCE_SHA256))
    return status()


def runtime_snapshot():
    value = _original_snapshot()
    identity = value["comparison_identity"]
    if _base._state["family"] == "qwen":
        identity.update(native_thread_baseline=deepcopy(_parent_thread_baseline),
            native_thread_observation=_thread_observation("parent"))
    # The original harness validates before writing REPRO_RUNTIME. Save the raw
    # first snapshot of each store before handing it to that unchanged guard.
    sink = _base._session.store
    store = getattr(sink, "store", sink)
    key = str(store.directory)
    if key not in _persisted_stores:
        sink.event("REPRO_RUNTIME_OBSERVED", evidence_contract=EVIDENCE_CONTRACT, snapshot=value)
        _persisted_stores.add(key)
    return value


# These are dependencies of privately loaded v1 closures, never mutations of
# the imported/frozen v1 module or of any active process.
_base.prepare_native, _base.runtime_snapshot, _base._worker_runtime = prepare_native, runtime_snapshot, _worker_runtime
