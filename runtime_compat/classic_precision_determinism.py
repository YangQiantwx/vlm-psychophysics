"""Explicit reproducibility profile; never enabled in historical production.

This profile fixes scheduling and known selection/reduction choices. Certification
comes from ``classic_precision.reproducibility``, not from these switches alone.
"""
from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import socket
import sys

PROFILE_ID = "classic-precision-strict-v2"
FILL_UNINITIALIZED_MEMORY = {"nemo": True, "qwen": False, "cosmos": True, "gemma": True}
BASE_RUNNER_SHA256 = "2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3"
ENVIRONMENT = {
    "PYTHONHASHSEED": "0", "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "VLLM_ENABLE_V1_MULTIPROCESSING": "0", "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
    "NUMPY_MADVISE_HUGEPAGE": "0", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "6", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_state = {}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def launch_environment():
    """Set these in the *new process* environment, before starting Python."""
    return dict(ENVIRONMENT)


def _configure_torch(torch):
    torch.use_deterministic_algorithms(True, warn_only=False)
    # This is a debug poison fill, not a deterministic arithmetic algorithm.
    # Qwen's finite native/exact controls require it disabled before loading.
    torch.utils.deterministic.fill_uninitialized_memory = FILL_UNINITIALIZED_MEMORY[_state["family"]]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Do not mix the new fp32_precision API with legacy allow_tf32 setters.
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


def tensor_fingerprint(tensor, name="pixels"):
    """Hash actual storage values in logical C order, including BF16 on CUDA.

    Bounded host chunks avoid an extra whole-video CPU allocation. No reduction,
    quantization or floating point conversion is used for the fingerprint.
    """
    import torch
    value = tensor.detach().contiguous().reshape(-1).view(torch.uint8)
    digest = hashlib.sha256()
    for chunk in value.split(16 * 1024 * 1024):
        raw = chunk.cpu().numpy()
        digest.update(memoryview(raw))
    return {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": digest.hexdigest(), "bytes": tensor.numel() * tensor.element_size()}


def _install_hashed_vision_probe(worker):
    from classic_precision_determinism import tensor_fingerprint
    model = worker.get_model()
    target = getattr(model, "vision_model", None)
    if target is None:
        target = model.visual
    worker._classic_precision_receipts = []

    def before(module, args, kwargs):
        x = args[0]
        grid = kwargs.get("grid_thw")
        worker._classic_precision_receipts.append({
            "shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
            "grid_thw": grid.tolist() if grid is not None else None,
            "num_frames_argument": kwargs.get("num_frames"),
            "tensor_fingerprint": tensor_fingerprint(x),
        })
    worker._classic_precision_hook = target.register_forward_pre_hook(before, with_kwargs=True)
    return True


def _read_hashed_receipts(worker):
    return worker._classic_precision_receipts


def activate(family):
    """Install only in a fresh, explicitly configured diagnostic process.

    Shared SDK files, checkpoint tensors and the frozen inference source remain
    untouched. Unsupported deterministic PyTorch operations raise an error.
    """
    if family not in ("qwen", "nemo", "cosmos", "gemma"):
        raise ValueError("Unknown model family")
    if _state:
        if _state["family"] != family:
            raise RuntimeError("One model family per reproducibility process")
        return status()
    bad = {k: {"expected": v, "actual": os.environ.get(k)}
           for k, v in ENVIRONMENT.items() if os.environ.get(k) != v}
    if bad:
        raise RuntimeError("Start a fresh process with launch_environment(): " + json.dumps(bad))
    if "torch" in sys.modules or "vllm" in sys.modules:
        raise RuntimeError("Activate reproducibility before importing Torch or vLLM")
    from classic_precision import inference
    if _sha(inference.__file__) != BASE_RUNNER_SHA256:
        raise RuntimeError("Unreviewed inference source; profile refuses activation")
    qsa = None
    if family == "qwen":
        os.environ["CLASSIC_PRECISION_QSA_DETERMINISM"] = "1"
        import classic_precision_qsa_determinism
        qsa = classic_precision_qsa_determinism.install()
    original_processor = inference.ModelSession._initialize_processor
    original_prepare = inference.ModelSession.prepare
    original_generate = inference.ModelSession.generate

    @functools.wraps(original_processor)
    def processor(session, *args, **kwargs):
        import torch
        _configure_torch(torch)
        result = original_processor(session, *args, **kwargs)
        session.store.event("REPRO_PROFILE", profile=status())
        return result

    @functools.wraps(original_prepare)
    def prepare(session, row, *args, **kwargs):
        result = original_prepare(session, row, *args, **kwargs)
        _, batch, _, _ = result
        fingerprints = [tensor_fingerprint(value, key) for key, value in sorted(batch.items())
                        if isinstance(value, session.torch.Tensor)]
        session.store.event("REPRO_INPUT_HASH", sample_id=row["sample_id"],
                            stage="cpu_processor", tensors=fingerprints)
        return result

    @functools.wraps(original_generate)
    def generate(session, row, *args, **kwargs):
        captured = []
        hook = None
        if session.family in ("cosmos", "gemma"):
            def before(module, positional, keywords):
                x = keywords.get("pixel_values_videos")
                if x is not None and not captured:
                    captured.append(tensor_fingerprint(x))
            hook = session.model.register_forward_pre_hook(before, with_kwargs=True)
        try:
            result = original_generate(session, row, *args, **kwargs)
            if session.family in ("nemo", "qwen"):
                captured = [entry["tensor_fingerprint"]
                            for group in session.engine.collective_rpc(_read_hashed_receipts)
                            for entry in group]
            if session.family == "qwen":
                import classic_precision_qsa_determinism
                qsa_status = classic_precision_qsa_determinism.status()
                if not qsa_status or qsa_status.get("state") != "applied":
                    raise RuntimeError("Qwen QSA deterministic hook did not apply")
            if not captured:
                raise RuntimeError("No actual GPU visual input hash")
            session.store.event("REPRO_INPUT_HASH", sample_id=row["sample_id"],
                                stage="gpu_forward", tensors=captured)
            return result
        finally:
            if hook is not None:
                hook.remove()

    inference.ModelSession._initialize_processor = processor
    inference.ModelSession.prepare = prepare
    inference.ModelSession.generate = generate
    inference._install_vision_probe = _install_hashed_vision_probe
    _state.update(profile_id=PROFILE_ID, family=family, environment=dict(ENVIRONMENT),
                  base_runner_sha256=BASE_RUNNER_SHA256, helper_sha256=_sha(__file__),
                  qsa_installation=qsa, certification="UNVALIDATED_UNTIL_REPEAT_SUITE_PASSES",
                  sdk_files_modified=False)
    return status()


def status():
    result = dict(_state)
    if result.get("family") == "qwen":
        import classic_precision_qsa_determinism
        result["qsa_installation"] = classic_precision_qsa_determinism.status()
    return result


def runtime_snapshot():
    import torch
    names = ("torch", "vllm", "transformers", "numpy", "av", "triton", "flashinfer-python")
    packages = {}
    for name in names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    properties = torch.cuda.get_device_properties(0)
    driver = Path("/proc/driver/nvidia/version")
    from classic_precision import inference
    identity = {"profile": status(), "packages": packages, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(), "gpu_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "gpu_total_bytes": properties.total_memory,
                "driver": driver.read_text().splitlines()[0] if driver.exists() else None,
                "inference_sha256": _sha(inference.__file__),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "effective_environment": {key: os.environ.get(key) for key in ENVIRONMENT}}
    if _state.get("family") == "qwen":
        identity["effective_environment"]["CLASSIC_PRECISION_QSA_DETERMINISM"] = os.environ.get(
            "CLASSIC_PRECISION_QSA_DETERMINISM")
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        identity["fp32_precision"] = {"matmul": torch.backends.cuda.matmul.fp32_precision,
                                     "cudnn_conv": torch.backends.cudnn.conv.fp32_precision,
                                     "cudnn_rnn": torch.backends.cudnn.rnn.fp32_precision}
    else:
        identity["fp32_precision"] = {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                                     "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32}
    return {"comparison_identity": identity, "hostname": socket.gethostname(),
            "pid": os.getpid(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_uuid": str(getattr(properties, "uuid", "unavailable"))}
