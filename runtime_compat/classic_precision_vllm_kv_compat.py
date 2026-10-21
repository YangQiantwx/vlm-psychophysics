"""Opt-in KV capacity measurement; the scientific runner stays byte-identical."""
import hashlib
import importlib
import importlib.util
import inspect
from pathlib import Path


BASE_RUNNER_SHA256 = "2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _kv_capacity(worker):
    """Measure the initialized groups with the installed vLLM native API."""
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    from classic_precision import inference
    base_sha = _sha(inference.__file__)
    if base_sha != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    runner = getattr(worker, "model_runner", None)
    config = getattr(runner, "kv_cache_config", None)
    block_sizes = [int(group.kv_cache_spec.block_size)
                   for group in getattr(config, "kv_cache_groups", [])]
    capacity_tokens = max_concurrency = None
    if config is not None:
        capacity_tokens, max_concurrency = get_kv_cache_capacity(worker.vllm_config, config)
    return {"num_gpu_blocks": int(config.num_blocks) if config is not None else None,
            "block_sizes": block_sizes,
            "capacity_tokens": int(capacity_tokens) if capacity_tokens is not None else None,
            "max_concurrency": float(max_concurrency) if max_concurrency is not None else None,
            "validation_compat_source_sha256": _sha(__file__),
            "base_runner_sha256": base_sha,
            "native_kv_helper_source_sha256": _sha(inspect.getsourcefile(get_kv_cache_capacity))}


def install():
    """Verify the original source before replacing only its capacity reader."""
    spec = importlib.util.find_spec("classic_precision.inference")
    if spec is None or spec.origin is None or _sha(spec.origin) != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    inference = importlib.import_module("classic_precision.inference")
    if _sha(inference.__file__) != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    inference._kv_capacity = _kv_capacity
