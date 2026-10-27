"""Private native-adapter input observers with the reviewed QSA-only policy.

The frozen adapter observer supplies source guards and actual tensor capture;
its strict activation is never called. Worker callbacks are importable here.
"""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

from classic_precision import qsa_only_reproducibility as qsa
from classic_precision.storage import file_sha256

PROFILE_ID = "classic-precision-native-adapter-qsa-only-v1"
FROZEN_OBSERVER_SHA256 = "c177a34e7dee227ad9d789443ecf106786cc00ae79d0147d23a30531c376c890"
FROZEN_QSA_ENTRY_SHA256 = "b10f7824ef476ce70d54a991cabbd0f2a6eded9b7b29bd215350d5e02cfe4d0a"
FROZEN_QSA_RUNTIME_SHA256 = "3210381807e5f60eb5cf5c5b7537e462217bc6ac54771bf013111c27b2e2ac07"
_private = None
_state = {}
_validator = None
THREAD_ENVIRONMENT = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def require(condition, message):
    if not condition: raise RuntimeError(message)


def _coverage(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    frozen = value.get("frozen_strict_profile", {}).get("implementation", {})
    require(value.get("schema") == "classic_precision_input_path_coverage/v1"
        and frozen.get("base_runner_sha256") == qsa.runtime.BASE_RUNNER_SHA256
        and frozen.get("qsa_helper_sha256") == qsa.runtime.QSA_HELPER_SHA256,
        "Historical coverage native/selector source differs")
    return path, value


def _observers():
    global _private
    if _private is None:
        path = Path(__file__).with_name("classic_precision_adapter_determinism.py")
        require(file_sha256(path) == FROZEN_OBSERVER_SHA256, "Frozen adapter observer changed")
        spec = importlib.util.spec_from_file_location("runtime_compat._adapter_qsa_private_observers", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.PROFILE_ID = PROFILE_ID
        module._core = lambda: qsa.runtime.native  # pure byte hashing only
        module._coverage = _coverage
        module._assert_policy = _assert_policy
        module._install_worker_probe_native = module._install_worker_probe
        module._read_worker_probe_native = module._read_worker_probe
        module._install_worker_probe = _install_worker_probe
        module._read_worker_probe = _read_worker_probe
        _private = module
    return _private


def _install_worker_probe(worker, family):
    require(family == "qwen", "Only Qwen adapter execution is enabled")
    return _observers()._install_worker_probe_native(worker, family)


def _read_worker_probe(worker):
    return _observers()._read_worker_probe_native(worker)


def activate(coverage_path, baseline):
    require(not _state and "torch" not in sys.modules and "vllm" not in sys.modules,
            "QSA adapter activation requires a fresh independent process")
    require(file_sha256(qsa.__file__) == FROZEN_QSA_ENTRY_SHA256
        and file_sha256(qsa.runtime.__file__) == FROZEN_QSA_RUNTIME_SHA256, "Reviewed QSA-only dependency changed")
    path, coverage = _coverage(coverage_path)
    observer = _observers()
    qsa.runtime.activate("qwen", "qwen_qsa_only", baseline)
    from classic_precision import inference as base
    expected = coverage["frozen_strict_profile"]["implementation"]["base_runner_sha256"]
    require(file_sha256(base.__file__) == expected, "Original native inference changed")
    originals = {}
    for name in ("prepare", "generate"):
        wrapped = getattr(base.ModelSession, name)
        observer._source_guard(wrapped, {qsa.runtime._base.__file__: file_sha256(qsa.runtime._base.__file__)}, wrapped=True)
        original = getattr(wrapped, "__wrapped__", None)
        observer._source_guard(original, {base.__file__: expected})
        require(inspect.getclosurevars(wrapped).nonlocals.get("original_" + name) is original,
                "QSA video wrapper original binding differs")
        originals[name] = original
    for name, original in originals.items(): setattr(base.ModelSession, name, original)
    # Retain the reviewed processor/launch observer; only incompatible video
    # prepare/generate wrappers are removed before native adapter construction.
    observer._state.update(family="qwen", coverage_sha256=file_sha256(path), coverage_path=str(path),
        core=qsa.runtime, base=base, module_sha256=file_sha256(__file__))
    _state.update(coverage_sha256=file_sha256(path), baseline=deepcopy(baseline), active_sample=None)
    return status()


def snapshot():
    qsa.runtime.prepare_native("qwen")
    value = qsa.runtime.runtime_snapshot()
    identity = value["comparison_identity"]
    identity["adapter_native_environment"] = _native_environment()
    workers = qsa.runtime._base._session.engine.collective_rpc(_worker_native_environment)
    require(len(workers) == len(identity["worker_runtime"]) == 1, "Actual adapter worker environment missing")
    for observed, environment in zip(identity["worker_runtime"], workers): observed["adapter_native_environment"] = environment
    validate_snapshot(value, _state["route"], _state["import_receipt"], _observers()._state["coverage_path"])
    return value


def _native_environment():
    return {key: os.environ.get(key) for key in THREAD_ENVIRONMENT}


def _worker_native_environment(worker):
    return _native_environment()


def bind_native_import(route, receipt):
    require(route in ("video", "multi", "single", "pair"), "Unknown native adapter")
    _state.update(route=route, import_receipt=deepcopy(receipt))


def _validation_entry():
    global _validator
    if _validator is None:
        path = Path(qsa.__file__)
        require(file_sha256(path) == FROZEN_QSA_ENTRY_SHA256, "Frozen QSA validator source changed")
        spec = importlib.util.spec_from_file_location("classic_precision._adapter_qsa_private_validator", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _validator = module
    return _validator


def _static_environment(value, profile, receipt):
    require(receipt.get("static_inputs_sha256") and all(receipt.get("after_native_import", {}).get(k) == "1" for k in THREAD_ENVIRONMENT),
            "Original source-bound static import derivation missing")
    v3 = _validation_entry()._native_entry._v3
    scope = "worker" if "worker_class" in value else "parent"
    observed = v3.frozen_thread_validator._validate_thread_observation(value, scope)
    v3._validate_source(value.get("engine_startup_source"))
    launch = value.get("engine_launch_parent", {})
    require(launch.get("phase") == "after_native_processor_before_engine_load"
        and launch.get("origin") == "launching_parent_process" and launch.get("family") == "qwen"
        and launch.get("inference_sha256") == qsa.runtime.BASE_RUNNER_SHA256
        and launch.get("omp_num_threads") == "1" and launch.get("vllm_omp_marker") is None
        and type(launch.get("torch_num_threads")) is int and launch["torch_num_threads"] > 0
        and launch.get("local_engine_count") == 1 and launch.get("cpu_affinity") == observed["cpu_affinity"],
        "Actual native static pre-engine launch differs")
    v3._validate_source(launch.get("engine_startup_source"))
    startup, baseline = profile["startup_environment"], value.get("native_postload_environment", {})
    require(set(baseline) == set(startup) and value.get("observed_environment") == baseline
        and value.get("adapter_native_environment") == dict.fromkeys(THREAD_ENVIRONMENT, "1"), "Native static actual thread environment drifted")
    expected = dict(startup)
    for key in THREAD_ENVIRONMENT:
        if key in expected: expected[key] = "1"
    expected["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    require(baseline == expected and baseline.get("VLLM_OMP_NUM_THREADS_SET_BY_VLLM") is None,
        "Undeclared static native environment transition")
    require(observed["torch_num_threads"] == (1 if scope == "worker" else launch["torch_num_threads"]), "Native static parent/worker Torch thread derivation differs")
    if scope == "worker":
        require(value.get("engine_config", {}).get("parallel_config", {}).get("distributed_executor_backend") == "uni", "Static native engine executor differs")


def validate_snapshot(snapshot, route, receipt=None, coverage_path=None):
    receipt = receipt or {}
    fields = ("before_import", "after_native_import", "before_native_processor")
    require(all(isinstance(receipt.get(field), dict) and set(receipt[field]) == set(THREAD_ENVIRONMENT) for field in fields),
        "Actual native import/processor environment receipt missing")
    require(receipt["before_native_processor"] == receipt["after_native_import"], "Native import environment was changed before processor construction")
    if route == "single":
        _, coverage = _coverage(coverage_path)
        selected = next(r for r in coverage["routes"] if r["route_id"] == "single")
        source = next(r for r in selected["implementation_files"] if Path(r["path"]).name == "static_inputs.py")
        require(receipt.get("static_inputs_sha256") == source["sha256"], "Native static import source differs from original coverage")
    else:
        require(receipt["before_import"] == receipt["after_native_import"], "Undeclared nonstatic native import environment mutation")
    validator = _validation_entry()
    original = validator._base._validate_environment
    if route == "single":
        validator._base._validate_environment = lambda value, profile: _static_environment(value, profile, receipt or {})
    try:
        return validator.validate_runtime_policy(snapshot, "qwen")
    finally:
        validator._base._validate_environment = original


def _assert_policy(family, require_qsa=False):
    require(family == "qwen" and _state, "QSA-only adapter is inactive")
    require(file_sha256(__file__) == _observers()._state["module_sha256"]
        and file_sha256(qsa.__file__) == FROZEN_QSA_ENTRY_SHA256
        and file_sha256(qsa.runtime.__file__) == FROZEN_QSA_RUNTIME_SHA256,
        "Active adapter/QSA-only source changed")
    value = snapshot()
    session = qsa.runtime._base._session
    session.store.event("ADAPTER_QSA_CALL_RUNTIME", sample_id=_state["active_sample"],
        stage="after" if require_qsa else "before", snapshot=value)


def instrument_session(session, route, adapter_module, coverage_path):
    require(session is qsa.runtime._base._session and hasattr(session, "engine"),
            "Actual native adapter engine is not bound to the reviewed processor observer")
    observer = _observers()
    handle = observer.instrument_session(session, route, adapter_module, coverage_path)
    original = handle.capture
    def capture(row, operation):
        require(_state["active_sample"] is None, "Nested native adapter request")
        _state["active_sample"] = row["sample_id"]
        try: return original(row, operation)
        finally: _state["active_sample"] = None
    handle.capture = capture
    return handle


def status():
    return {"profile_id": PROFILE_ID, "helper_sha256": file_sha256(__file__),
        "frozen_observer_sha256": FROZEN_OBSERVER_SHA256, "qsa_runtime_sha256": FROZEN_QSA_RUNTIME_SHA256,
        "state": "QSA_ONLY_NATIVE_ADAPTER_ACTIVE" if _state else "INACTIVE",
        "torch_policy": {}, "sdk_files_modified": False}
