"""Opt-in configuration-only diagnostic; native model operators stay intact.

Only Python/Torch configuration and observational input hashes are installed.
The strict QSA replacement module must never be imported in this process.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import inspect
import json
import marshal
import os
from pathlib import Path
import random
import socket
import sys
from types import CodeType

from runtime_compat import classic_precision_determinism as observers

PROFILE_ID = "classic-precision-config-only-v1"
BASELINE_PROFILE_ID = "classic-precision-native-nemo-observed-v1"
FAMILIES = ("nemo", "qwen", "cosmos")
BASE_RUNNER_SHA256 = observers.BASE_RUNNER_SHA256
ENVIRONMENT = {**observers.ENVIRONMENT, "CLASSIC_PRECISION_QSA_DETERMINISM": "0"}
FILL_UNINITIALIZED_MEMORY = {family: False for family in FAMILIES}
BASELINE_ENVIRONMENT = {"CLASSIC_PRECISION_KV_COMPAT": "1", "CLASSIC_PRECISION_NEMO_DUMMY_CONFIG_COMPAT": "1",
    "CLASSIC_PRECISION_DEFER_MODELS": "gemma", "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CLASSIC_PRECISION_QSA_DETERMINISM": "0"}
BASELINE_CLEARED_ENVIRONMENT = tuple(key for key in observers.ENVIRONMENT
    if key not in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"))
QSA_TARGET_MODULE = "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer"
QSA_NATIVE_SOURCE_SHA256 = "4d09eb2f409e7cd71c7849c4f3bdd71d722bce6feaccb558f7ae67ac2d489a2b"
_PARAMETERS = ("logits", "visible_blocks", "token_topk", "compress_ratio", "block_indices", "topk_workspace")
_state = {}
_native_function = None
_session = None
tensor_fingerprint = observers.tensor_fingerprint


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_environment():
    return dict(ENVIRONMENT)


def assert_no_qsa_replacement():
    if os.environ.get("CLASSIC_PRECISION_QSA_DETERMINISM") != "0":
        raise RuntimeError("Configuration-only requires CLASSIC_PRECISION_QSA_DETERMINISM=0")
    if any(name.rsplit(".", 1)[-1] == "classic_precision_qsa_determinism" for name in sys.modules):
        raise RuntimeError("QSA replacement module was imported in configuration-only process")
    if any("classic_precision_qsa_determinism" in type(finder).__module__ for finder in sys.meta_path):
        raise RuntimeError("QSA replacement import hook is installed")


def _native_source_path():
    # PathFinder on the top-level package does not execute vLLM or import Torch.
    spec = importlib.machinery.PathFinder.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate native vLLM QSA source")
    paths = [Path(root) / "models/qwen4_exp/nvidia/ops/qsa_indexer.py"
             for root in spec.submodule_search_locations]
    matches = [path for path in paths if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError("Native QSA source is missing or ambiguous")
    return matches[0]


def _qsa_source_evidence(path):
    source = Path(path).read_bytes()
    if hashlib.sha256(source).hexdigest() != QSA_NATIVE_SOURCE_SHA256:
        raise RuntimeError("Native QSA source SHA mismatch")
    compiled = compile(source, str(path), "exec", dont_inherit=True)
    codes = [value for value in compiled.co_consts if isinstance(value, CodeType) and value.co_name == "_topk"]
    if len(codes) != 1:
        raise RuntimeError("Native QSA _topk definition is missing or ambiguous")
    code = codes[0]
    evidence = {"target_module": QSA_TARGET_MODULE, "function_module": QSA_TARGET_MODULE,
                "function_qualname": "_topk", "expected_native_source_sha256": QSA_NATIVE_SOURCE_SHA256,
                "native_source_sha256": hashlib.sha256(source).hexdigest(),
                "native_code_sha256": hashlib.sha256(marshal.dumps(code.replace(co_filename="<native-qsa>"))).hexdigest(),
                "replacement_enabled": False, "observer_only": True}
    return code, evidence


def verify_native_qsa(module=None):
    """Check the live function against the pinned, compiled original and object."""
    global _native_function
    assert_no_qsa_replacement()
    module = importlib.import_module(QSA_TARGET_MODULE) if module is None else module
    code, evidence = _qsa_source_evidence(module.__file__)
    function = getattr(module, "_topk", None)
    if (not inspect.isfunction(function) or function.__module__ != QSA_TARGET_MODULE
            or function.__qualname__ != "_topk" or function.__globals__ is not vars(module)
            or function.__code__.co_filename != module.__file__
            or hasattr(function, "__wrapped__") or hasattr(function, "_classic_precision_qsa_determinism")
            or tuple(inspect.signature(function).parameters) != _PARAMETERS
            or any(p.kind != inspect.Parameter.POSITIONAL_OR_KEYWORD or p.default is not inspect.Parameter.empty
                   for p in inspect.signature(function).parameters.values())
            or marshal.dumps(function.__code__) != marshal.dumps(code)):
        raise RuntimeError("Native QSA _topk function identity/code changed or was hooked")
    if _native_function is not None and function is not _native_function:
        raise RuntimeError("Native QSA _topk function object changed after initial verification")
    _native_function = function
    return {**evidence, "state": "native_verified"}


def prepare_native(family):
    assert_no_qsa_replacement()
    if family == "qwen":
        verify_native_qsa()


def _configure_torch(torch):
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.conv.fp32_precision = "ieee"
        torch.backends.cudnn.rnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    import numpy as np
    np.random.seed(0)
    random.seed(0)


def _install_hashed_vision_probe(worker):
    model = worker.get_model()
    target = getattr(model, "vision_model", None)
    if target is None:
        target = model.visual
    worker._classic_precision_receipts = []
    def before(module, args, kwargs):
        x, grid = args[0], kwargs.get("grid_thw")
        worker._classic_precision_receipts.append({"shape": list(x.shape), "dtype": str(x.dtype),
            "device": str(x.device), "grid_thw": grid.tolist() if grid is not None else None,
            "num_frames_argument": kwargs.get("num_frames"), "tensor_fingerprint": tensor_fingerprint(x)})
    worker._classic_precision_hook = target.register_forward_pre_hook(before, with_kwargs=True)
    return True


def activate(family, mode="config_only", environment=None):
    global _session
    if mode not in ("config_only", "native_nemo"):
        raise ValueError("Unknown reproducibility mode")
    if family not in FAMILIES:
        raise ValueError("Configuration-only supports only Nemo, Qwen and Cosmos; Gemma is excluded")
    if mode == "native_nemo" and family != "nemo":
        raise ValueError("Native baseline is restricted to Nemo")
    assert_no_qsa_replacement()
    if _state:
        if _state["family"] != family or _state["mode"] != mode:
            raise RuntimeError("One model family per configuration-only process")
        return status()
    required_env = dict(ENVIRONMENT if mode == "config_only" else environment or BASELINE_ENVIRONMENT)
    bad = {key: os.environ.get(key) for key, value in required_env.items() if os.environ.get(key) != value}
    if mode == "native_nemo":
        bad.update({key: os.environ[key] for key in BASELINE_CLEARED_ENVIRONMENT if key in os.environ})
    if bad:
        raise RuntimeError("Start a fresh process with launch_environment(): " + json.dumps(bad))
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("Activate configuration-only before importing Torch or vLLM")
    from classic_precision import inference
    if _sha(inference.__file__) != BASE_RUNNER_SHA256:
        raise RuntimeError("Unreviewed inference source; configuration-only refuses activation")
    methods = (inference.ModelSession._initialize_processor, inference.ModelSession.prepare, inference.ModelSession.generate)
    if observers._state or any(hasattr(method, "__wrapped__") for method in methods):
        raise RuntimeError("An inference runtime wrapper is already installed")
    qsa_evidence = _qsa_source_evidence(_native_source_path())[1] if family == "qwen" else None
    original_processor, original_prepare, original_generate = methods

    @functools.wraps(original_processor)
    def processor(session, *args, **kwargs):
        global _session
        import torch
        if mode == "config_only":
            _configure_torch(torch)
        result = original_processor(session, *args, **kwargs)
        _session = session
        session.store.event("REPRO_PROFILE", profile=status())
        return result

    @functools.wraps(original_prepare)
    def prepare(session, row, *args, **kwargs):
        result = original_prepare(session, row, *args, **kwargs)
        fingerprints = [tensor_fingerprint(value, key) for key, value in sorted(result[1].items())
                        if isinstance(value, session.torch.Tensor)]
        session.store.event("REPRO_INPUT_HASH", sample_id=row["sample_id"], stage="cpu_processor", tensors=fingerprints)
        return result

    @functools.wraps(original_generate)
    def generate(session, row, *args, **kwargs):
        prepare_native(session.family)
        if session.family in ("nemo", "qwen"):
            session.store.event("REPRO_WORKER_RUNTIME", sample_id=row["sample_id"], stage="before",
                                workers=session.engine.collective_rpc(_worker_runtime))
        captured, hook = [], None
        if session.family == "cosmos":
            def before(module, positional, keywords):
                x = keywords.get("pixel_values_videos")
                if x is not None and not captured:
                    captured.append(tensor_fingerprint(x))
            hook = session.model.register_forward_pre_hook(before, with_kwargs=True)
        try:
            result = original_generate(session, row, *args, **kwargs)
            if session.family in ("nemo", "qwen"):
                captured = [entry["tensor_fingerprint"]
                            for group in session.engine.collective_rpc(observers._read_hashed_receipts) for entry in group]
                session.store.event("REPRO_WORKER_RUNTIME", sample_id=row["sample_id"], stage="after",
                                    workers=session.engine.collective_rpc(_worker_runtime))
            prepare_native(session.family)
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
    inference._install_vision_probe = _install_hashed_vision_probe
    _state.update(profile_id=PROFILE_ID if mode == "config_only" else BASELINE_PROFILE_ID,
                  family=family, mode=mode, environment=required_env,
                  base_runner_sha256=BASE_RUNNER_SHA256, helper_sha256=_sha(__file__),
                  observer_helper_sha256=_sha(observers.__file__), sdk_files_modified=False,
                  certification="CONFIG_ONLY_UNVALIDATED_UNTIL_REPEAT_SUITE_PASSES" if mode == "config_only"
                      else "NATIVE_NEMO_OBSERVED_RETEST_NOT_HISTORICAL_FLAG_RESTORATION")
    if mode == "native_nemo":
        _state["startup_cleared_environment"] = {key: None for key in BASELINE_CLEARED_ENVIRONMENT}
    if family == "qwen":
        _state["qsa_installation"] = {**qsa_evidence, "state": "pending_target_import"}
    return status()


def status():
    result = dict(_state)
    if result.get("family") == "qwen":
        module = sys.modules.get(QSA_TARGET_MODULE)
        result["qsa_installation"] = verify_native_qsa(module) if module is not None else dict(_state["qsa_installation"])
    return result


def _torch_flags(torch):
    value = {"seed": torch.initial_seed(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        "cudnn_benchmark": torch.backends.cudnn.benchmark, "cudnn_deterministic": torch.backends.cudnn.deterministic}
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        value["fp32_precision"] = {"matmul": torch.backends.cuda.matmul.fp32_precision,
            "cudnn_conv": torch.backends.cudnn.conv.fp32_precision, "cudnn_rnn": torch.backends.cudnn.rnn.fp32_precision}
    else:
        value["fp32_precision"] = {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                  "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32}
    return value


def _worker_runtime(worker):
    """Observe inside the actual vLLM worker; never set a runtime flag."""
    import torch
    model = worker.get_model()
    config = worker.vllm_config
    fields = {"model_config": ("seed", "enforce_eager", "max_model_len", "dtype"),
        "parallel_config": ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size", "distributed_executor_backend"),
        "scheduler_config": ("max_num_seqs", "max_num_batched_tokens", "enable_chunked_prefill"),
        "cache_config": ("cache_dtype", "enable_prefix_caching")}
    engine = {}
    for section, names in fields.items():
        part = getattr(config, section)
        engine[section] = {key: (str(getattr(part, key)) if key == "dtype" else getattr(part, key)) for key in names}
    qsa = verify_native_qsa() if type(model).__name__ == "Qwen4ExpForConditionalGeneration" else None
    return {**_torch_flags(torch), "worker_class": type(worker).__module__ + "." + type(worker).__qualname__,
            "model_class": type(model).__module__ + "." + type(model).__qualname__, "engine_config": engine,
            "qsa_native_identity": qsa, "gpu_name": torch.cuda.get_device_properties(0).name}


def runtime_snapshot():
    import torch
    assert_no_qsa_replacement()
    packages = {}
    for name in ("torch", "vllm", "transformers", "numpy", "av", "triton", "flashinfer-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(0)
    driver = Path("/proc/driver/nvidia/version")
    from classic_precision import inference
    identity = {"profile": status(), "packages": packages, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "gpu_name": properties.name,
        "compute_capability": [properties.major, properties.minor], "gpu_total_bytes": properties.total_memory,
        "driver": driver.read_text().splitlines()[0] if driver.exists() else None,
        "inference_sha256": _sha(inference.__file__), **_torch_flags(torch),
        "effective_environment": {key: os.environ.get(key) for key in _state["environment"]},
        "startup_cleared_environment": _state.get("startup_cleared_environment", {}),
        "effective_cleared_environment": {key: os.environ.get(key) for key in BASELINE_CLEARED_ENVIRONMENT}
            if _state["mode"] == "native_nemo" else {},
        "qsa_replacement_module_imported": False, "qsa_replacement_import_hook": False}
    identity["worker_runtime"] = _session.engine.collective_rpc(_worker_runtime) if _state["family"] in ("nemo", "qwen") else []
    return {"comparison_identity": identity, "hostname": socket.gethostname(), "pid": os.getpid(),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu_uuid": str(getattr(properties, "uuid", "unavailable"))}
