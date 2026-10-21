"""Allow vLLM's exact Nemotron model-type probe without changing real geometry."""
import functools
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys


BASE_RUNNER_SHA256 = "2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3"
ENVIRONMENT_VARIABLE = "CLASSIC_PRECISION_NEMO_DUMMY_CONFIG_COMPAT"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install():
    """Only bypass the native PreTrainedConfig dummy; load no Transformers/GPU stack."""
    requested = os.environ.get(ENVIRONMENT_VARIABLE)
    if requested is None:
        return None
    if requested != "1":
        raise ValueError(ENVIRONMENT_VARIABLE + " must be exactly 1")
    spec = importlib.util.find_spec("classic_precision.inference")
    if spec is None or spec.origin is None or _sha(spec.origin) != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    inference = importlib.import_module("classic_precision.inference")
    if _sha(inference.__file__) != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    original = inference._nemo_geometry
    installed = getattr(original, "_classic_precision_nemo_config_override", None)
    if installed is not None:
        return dict(installed)
    evidence = {"environment_variable": ENVIRONMENT_VARIABLE,
                "dummy_model_type": "dummy_nemotron_h_omni",
                "helper_source_sha256": _sha(__file__),
                "base_runner_sha256": BASE_RUNNER_SHA256}

    @functools.wraps(original)
    def geometry(config):
        native = sys.modules.get("transformers.configuration_utils")
        if (native is not None and type(config) is getattr(native, "PreTrainedConfig", None)
                and getattr(config, "model_type", None) == "dummy_nemotron_h_omni"
                and getattr(config, "architectures", None) == [""]
                and not hasattr(config, "vision_config")):
            return config
        return original(config)

    geometry._classic_precision_nemo_config_override = evidence
    inference._nemo_geometry = geometry
    return dict(evidence)
