"""Minimal inference-time interventions; native loading and QSA stay unchanged.

Both the parent and actual vLLM workers observe their native post-load flags
before applying the declared intervention. No policy is applied during loading
or warmup. The native and scheduling arms never call a Torch policy setter.
"""
from __future__ import annotations

from copy import deepcopy
import functools
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys

from runtime_compat import classic_precision_config_only as native

PROFILE_MODES = ("native_cosmos", "qwen_schedule_only", "qwen_tf32_off_only", "qwen_torch_deterministic")
PROFILE_IDS = {mode: "classic-precision-minimal-" + mode.replace("_", "-") + "-v1" for mode in PROFILE_MODES}
BASE_RUNNER_SHA256 = native.BASE_RUNNER_SHA256
APPLICATION_PHASE = "after_native_model_load_before_first_diagnostic_generate"
TRACKED_ENVIRONMENT = tuple(sorted(set(native.observers.ENVIRONMENT) | set(native.BASELINE_ENVIRONMENT)
    | {"CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB"}))
TORCH_POLICIES = {
    "native_cosmos": {}, "qwen_schedule_only": {},
    "qwen_tf32_off_only": {"tf32": False},
    "qwen_torch_deterministic": {"deterministic_algorithms": True, "deterministic_warn_only": False,
        "fill_uninitialized_memory": False},
}
_state, _application, _postload_environment = {}, {}, {}
_session = None


def family_for_mode(mode):
    if mode not in PROFILE_MODES:
        raise ValueError("Unknown minimal profile mode")
    return "cosmos" if mode == "native_cosmos" else "qwen"


def baseline_environment(environment=None):
    """Seal every relevant flag, including absence; callers may supply observed values.

    The default is the original native entry's compatibility/offline environment,
    with *all* strict-added flags absent. Explicit caller values are baseline
    facts, never hidden interventions, and are included in the plan identity.
    """
    result = {key: None for key in TRACKED_ENVIRONMENT}
    result.update(native.BASELINE_ENVIRONMENT)
    if environment is not None:
        if not isinstance(environment, dict) or not all(isinstance(k, str) and k
                and (v is None or isinstance(v, str)) for k, v in environment.items()):
            raise ValueError("Baseline environment must contain string keys and string/null values")
        result.update(environment)
    if any(result.get(key) != value for key, value in native.BASELINE_ENVIRONMENT.items()):
        raise ValueError("Native compatibility/offline/QSA environment differs")
    return dict(sorted(result.items()))


def environment_contract(mode, baseline=None):
    family_for_mode(mode)
    before = baseline_environment(baseline)
    interventions = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0"} if mode == "qwen_schedule_only" else {}
    effective = {**before, **interventions}
    return before, interventions, {k: v for k, v in effective.items() if v is not None}, [k for k, v in effective.items() if v is None]


def _configure_torch(torch, mode):
    """Only declared setters; deliberately no seeds, cuDNN flags or thread calls."""
    if mode == "qwen_tf32_off_only":
        if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "ieee"
            torch.backends.cudnn.rnn.fp32_precision = "ieee"
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
    elif mode == "qwen_torch_deterministic":
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.utils.deterministic.fill_uninitialized_memory = False
    else:
        family_for_mode(mode)


def _apply(torch, mode):
    before = native._torch_flags(torch)
    _configure_torch(torch, mode)
    return {"phase": APPLICATION_PHASE, "model_loaded_before_application": True,
        "mode": mode, "declared_torch_policy": deepcopy(TORCH_POLICIES[mode]),
        "before": before, "after": native._torch_flags(torch)}


def _worker_runtime(worker):
    # The pinned native helper checks the original source, code and live object
    # inside the actual worker on every observation.
    value = native._worker_runtime(worker)
    value["application_receipt"] = deepcopy(getattr(worker, "_classic_precision_minimal_application", None))
    baseline = getattr(worker, "_classic_precision_minimal_native_environment", {})
    value["native_postload_environment"] = deepcopy(baseline)
    value["observed_environment"] = {key: os.environ.get(key) for key in baseline}
    return value


def _worker_apply(worker, mode, environment_keys=TRACKED_ENVIRONMENT):
    import torch
    native.verify_native_qsa()
    receipt = getattr(worker, "_classic_precision_minimal_application", None)
    if receipt is None:
        worker._classic_precision_minimal_native_environment = {key: os.environ.get(key) for key in environment_keys}
        worker._classic_precision_minimal_application = _apply(torch, mode)
    elif receipt["mode"] != mode:
        raise RuntimeError("Worker already belongs to another minimal profile")
    return _worker_runtime(worker)


def prepare_native(family):
    """Called after the unchanged runner loads, and before every model call."""
    native.prepare_native(family)
    if _application:
        return
    if _session is None or not hasattr(_session, "engine" if family == "qwen" else "model"):
        raise RuntimeError("Minimal policy requires a fully loaded native model")
    import torch
    keys = tuple(sorted(set(_state["environment"]) | set(_state["cleared_environment"])))
    _postload_environment.update({key: os.environ.get(key) for key in keys})
    _application.update(_apply(torch, _state["mode"]))
    if family == "qwen":
        # RPC reaches the actual worker even when native multiprocessing is on.
        workers = _session.engine.collective_rpc(_worker_apply, args=(_state["mode"], keys))
        if not workers or any(item.get("application_receipt") is None for item in workers):
            raise RuntimeError("Actual worker policy application receipt is missing")


def activate(family, mode="native_cosmos", environment=None):
    global _session
    if family != family_for_mode(mode):
        raise ValueError("Minimal profile mode/family mismatch")
    native.assert_no_qsa_replacement()
    before, interventions, required, cleared = environment_contract(mode, environment)
    if _state:
        if (_state["family"], _state["mode"], _state["baseline_environment"]) != (family, mode, before):
            raise RuntimeError("One sealed minimal profile per fresh process")
        return status()
    bad = {key: os.environ.get(key) for key, value in {**required, **dict.fromkeys(cleared)}.items()
        if os.environ.get(key) != value}
    if bad:
        raise RuntimeError("Actual minimal launch environment differs: " + json.dumps(bad))
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("Activate minimal profile before importing Torch or vLLM")
    from classic_precision import inference
    if native._sha(inference.__file__) != BASE_RUNNER_SHA256:
        raise RuntimeError("Unreviewed inference source; minimal profile refuses activation")
    methods = (inference.ModelSession._initialize_processor, inference.ModelSession.prepare, inference.ModelSession.generate)
    if native.observers._state or native._state or any(hasattr(method, "__wrapped__") for method in methods):
        raise RuntimeError("An inference runtime wrapper is already installed")
    qsa = native._qsa_source_evidence(native._native_source_path())[1] if family == "qwen" else None
    original_processor, original_prepare, original_generate = methods

    @functools.wraps(original_processor)
    def processor(session, *args, **kwargs):
        global _session
        result = original_processor(session, *args, **kwargs)
        _session = session
        session.store.event("REPRO_PROFILE", profile=status())
        return result

    @functools.wraps(original_prepare)
    def prepare(session, row, *args, **kwargs):
        result = original_prepare(session, row, *args, **kwargs)
        fingerprints = [native.tensor_fingerprint(value, key) for key, value in sorted(result[1].items())
            if isinstance(value, session.torch.Tensor)]
        session.store.event("REPRO_INPUT_HASH", sample_id=row["sample_id"], stage="cpu_processor", tensors=fingerprints)
        return result

    @functools.wraps(original_generate)
    def generate(session, row, *args, **kwargs):
        prepare_native(session.family)
        session.store.event("REPRO_CALL_RUNTIME", sample_id=row["sample_id"], stage="before", snapshot=runtime_snapshot())
        captured, hook = [], None
        if session.family == "cosmos":
            def before(module, positional, keywords):
                x = keywords.get("pixel_values_videos")
                if x is not None and not captured:
                    captured.append(native.tensor_fingerprint(x))
            hook = session.model.register_forward_pre_hook(before, with_kwargs=True)
        try:
            result = original_generate(session, row, *args, **kwargs)
            if session.family == "qwen":
                captured = [entry["tensor_fingerprint"] for group in session.engine.collective_rpc(native.observers._read_hashed_receipts) for entry in group]
            prepare_native(session.family)
            session.store.event("REPRO_CALL_RUNTIME", sample_id=row["sample_id"], stage="after", snapshot=runtime_snapshot())
            if not captured:
                raise RuntimeError("No actual GPU visual input hash")
            session.store.event("REPRO_INPUT_HASH", sample_id=row["sample_id"], stage="gpu_forward", tensors=captured)
            return result
        finally:
            if hook is not None:
                hook.remove()

    inference.ModelSession._initialize_processor = processor
    inference.ModelSession.prepare = prepare
    inference.ModelSession.generate = generate
    inference._install_vision_probe = native._install_hashed_vision_probe
    _state.update(profile_id=PROFILE_IDS[mode], family=family, mode=mode, baseline_environment=before,
        environment_interventions=interventions, environment=required, cleared_environment=cleared,
        startup_environment={key: os.environ.get(key) for key in sorted(set(required) | set(cleared))},
        application_phase=APPLICATION_PHASE, torch_policy=deepcopy(TORCH_POLICIES[mode]),
        base_runner_sha256=BASE_RUNNER_SHA256, helper_sha256=native._sha(__file__),
        native_guard_helper_sha256=native._sha(native.__file__), observer_helper_sha256=native._sha(native.observers.__file__),
        sdk_files_modified=False, certification="MINIMAL_INFERENCE_TIME_DIAGNOSTIC_NOT_COLD_LOAD_POLICY")
    if qsa is not None:
        _state["qsa_installation"] = {**qsa, "state": "pending_target_import"}
    return status()


def status():
    result = deepcopy(_state)
    if result.get("family") == "qwen":
        module = sys.modules.get(native.QSA_TARGET_MODULE)
        if module is not None:
            result["qsa_installation"] = native.verify_native_qsa(module)
    return result


def runtime_snapshot():
    import torch
    native.assert_no_qsa_replacement()
    if not _application:
        raise RuntimeError("Native post-load baseline and application have not been observed")
    packages = {}
    for name in ("torch", "vllm", "transformers", "numpy", "av", "triton", "flashinfer-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(0)
    driver = Path("/proc/driver/nvidia/version")
    from classic_precision import inference
    tracked = set(_state["environment"]) | set(_state["cleared_environment"])
    identity = {"profile": status(), "packages": packages, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor], "gpu_total_bytes": properties.total_memory,
        "driver": driver.read_text().splitlines()[0] if driver.exists() else None,
        "inference_sha256": native._sha(inference.__file__), **native._torch_flags(torch),
        "application_receipt": deepcopy(_application),
        "effective_environment": {key: os.environ.get(key) for key in _state["environment"]},
        "native_postload_environment": deepcopy(_postload_environment),
        "observed_environment": {key: os.environ.get(key) for key in sorted(tracked)},
        "qsa_replacement_module_imported": False, "qsa_replacement_import_hook": False,
        "worker_runtime": _session.engine.collective_rpc(_worker_runtime) if _state["family"] == "qwen" else []}
    return {"comparison_identity": identity, "hostname": socket.gethostname(), "pid": os.getpid(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_uuid": str(getattr(properties, "uuid", "unavailable"))}
