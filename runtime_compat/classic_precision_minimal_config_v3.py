"""Observe native UniProc EngineCore thread inheritance without new setters."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

from runtime_compat import classic_precision_minimal_config_v2 as frozen_observers

FROZEN_HELPER_SHA256 = frozen_observers.FROZEN_HELPER_SHA256
FROZEN_V2_HELPER_SHA256 = "5f635ddd6cb0b015842dabff9041c68977b86f4a822d49bb04d99325aefa941f"
if frozen_observers._sha(frozen_observers.__file__) != FROZEN_V2_HELPER_SHA256:
    raise RuntimeError("Frozen v2 observer source changed")
ENGINE_STARTUP_MODULE = "vllm.v1.engine.utils"
ENGINE_STARTUP_SHA256 = "2f5b1b802ca31f19545ba00c68afc83cbbec4e55399e311e9102ce7a483254ab"
EVIDENCE_CONTRACT = "native-uniproc-enginecore-launch-thread-observation-v3"
_base = frozen_observers._load_frozen()
native = _base.native
PROFILE_MODES = ("qwen_tf32_off_only", "qwen_torch_deterministic")
PROFILE_IDS = {mode: _base.PROFILE_IDS[mode].removesuffix("-v1") + "-v3" for mode in PROFILE_MODES}
TORCH_POLICIES, THREAD_SOURCE_SHA256 = _base.TORCH_POLICIES, frozen_observers.THREAD_SOURCE_SHA256
BASE_RUNNER_SHA256, APPLICATION_PHASE = _base.BASE_RUNNER_SHA256, _base.APPLICATION_PHASE
TRACKED_ENVIRONMENT = frozen_observers.TRACKED_ENVIRONMENT
_base.PROFILE_IDS, _base.TRACKED_ENVIRONMENT = PROFILE_IDS, TRACKED_ENVIRONMENT
baseline_environment, environment_contract = _base.baseline_environment, _base.environment_contract
family_for_mode = _base.family_for_mode
_original_snapshot, _original_status = _base.runtime_snapshot, _base.status
_launch_parent, _parent_thread_baseline, _persisted_stores = {}, {}, set()
_sha, _thread_observation = frozen_observers._sha, frozen_observers._thread_observation


def _engine_startup_source():
    # PathFinder locates the pinned QSA source without importing vLLM; the
    # processor has not loaded EngineCore at the pre-launch observation point.
    path = native._native_source_path().parents[4] / "v1/engine/utils.py"
    return {"module": ENGINE_STARTUP_MODULE, "expected_sha256": ENGINE_STARTUP_SHA256,
        "actual_sha256": _sha(path) if path is not None and path.is_file() else None,
        "function": "_configure_uniproc_startup_threads"}


def status():
    session = _base._session
    # The unchanged processor closure calls status exactly after assigning its
    # initialized session and before the unchanged _initialize loads the engine.
    if session is not None and not hasattr(session, "engine") and not _launch_parent:
        import torch
        _launch_parent.update(phase="after_native_processor_before_engine_load", origin="launching_parent_process", family=session.family,
            inference_sha256=native._sha(Path(__file__).parents[1] / "classic_precision/inference.py"),
            engine_startup_source=_engine_startup_source(),
            cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            torch_num_threads=torch.get_num_threads(), omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            vllm_omp_marker=os.environ.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM"), local_engine_count=1)
        session.store.event("REPRO_ENGINE_LAUNCH_PARENT", receipt=deepcopy(_launch_parent))
    return _original_status()


def _worker_runtime(worker):
    value = native._worker_runtime(worker)
    baseline = getattr(worker, "_classic_precision_minimal_native_environment", {})
    value.update(application_receipt=deepcopy(getattr(worker, "_classic_precision_minimal_application", None)),
        native_postload_environment=deepcopy(baseline), observed_environment={key: os.environ.get(key) for key in baseline},
        native_thread_baseline=deepcopy(getattr(worker, "_classic_precision_v3_native_thread_baseline", None)),
        native_thread_observation=_thread_observation("worker"), engine_startup_source=_engine_startup_source(),
        engine_launch_parent=deepcopy(getattr(worker, "_classic_precision_v3_engine_launch_parent", None)))
    return value


def _worker_apply(worker, mode, environment_keys=TRACKED_ENVIRONMENT, engine_launch_parent=None):
    import torch
    native.verify_native_qsa()
    receipt = getattr(worker, "_classic_precision_minimal_application", None)
    if receipt is None:
        worker._classic_precision_minimal_native_environment = {key: os.environ.get(key) for key in environment_keys}
        worker._classic_precision_v3_native_thread_baseline = _thread_observation("worker")
        worker._classic_precision_v3_engine_launch_parent = deepcopy(engine_launch_parent)
        worker._classic_precision_minimal_application = _base._apply(torch, mode)
    elif receipt["mode"] != mode:
        raise RuntimeError("Worker already belongs to another minimal profile")
    return _worker_runtime(worker)


def prepare_native(family):
    native.prepare_native(family)
    if _base._application:
        return
    session = _base._session
    if family != "qwen" or session is None or not hasattr(session, "engine"):
        raise RuntimeError("Minimal v3 requires a loaded native Qwen engine")
    import torch
    keys = tuple(sorted(set(_base._state["environment"]) | set(_base._state["cleared_environment"])))
    _base._postload_environment.update({key: os.environ.get(key) for key in keys})
    _parent_thread_baseline.update(_thread_observation("parent"))
    _base._application.update(_base._apply(torch, _base._state["mode"]))
    workers = session.engine.collective_rpc(_worker_apply, args=(_base._state["mode"], keys, deepcopy(_launch_parent)))
    if not workers or any(item.get("application_receipt") is None for item in workers):
        raise RuntimeError("Actual worker v3 application receipt missing")


def activate(family, mode="qwen_tf32_off_only", environment=None):
    if mode not in PROFILE_MODES:
        raise ValueError("Minimal v3 only supports the native-EngineCore Qwen TF32/deterministic arms")
    _base.activate(family, mode, environment)
    _base._state.update(helper_sha256=_sha(__file__), frozen_policy_helper_sha256=FROZEN_HELPER_SHA256,
        frozen_observer_helper_sha256=FROZEN_V2_HELPER_SHA256, evidence_contract=EVIDENCE_CONTRACT,
        native_thread_source_sha256=deepcopy(THREAD_SOURCE_SHA256), engine_startup_source_sha256=ENGINE_STARTUP_SHA256)
    return status()


def runtime_snapshot():
    value = _original_snapshot()
    value["comparison_identity"].update(native_thread_baseline=deepcopy(_parent_thread_baseline),
        native_thread_observation=_thread_observation("parent"), engine_launch_parent=deepcopy(_launch_parent),
        engine_startup_source=_engine_startup_source())
    sink = _base._session.store
    store = getattr(sink, "store", sink)
    key = str(store.directory)
    if key not in _persisted_stores:
        sink.event("REPRO_RUNTIME_OBSERVED", evidence_contract=EVIDENCE_CONTRACT, snapshot=value)
        _persisted_stores.add(key)
    return value


_base.status, _base.prepare_native, _base.runtime_snapshot, _base._worker_runtime = status, prepare_native, runtime_snapshot, _worker_runtime
