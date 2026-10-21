"""Explicit Qwen expert-residency override for the byte-identical runner."""
import ast
import copy
import functools
import hashlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path


BASE_RUNNER_SHA256 = "2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3"
ENVIRONMENT_VARIABLE = "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB"
ALLOWED_VALUES = {"12": 12, "16": 16, "24": 24}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _loader(inference, value):
    """Compile the pinned method with exactly one integer literal replaced."""
    source = Path(inference.__file__).read_text()
    tree = ast.parse(source)
    model = next(node for node in tree.body
                 if isinstance(node, ast.ClassDef) and node.name == "ModelSession")
    method = next(node for node in model.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_load_vllm")
    original = inference.ModelSession._load_vllm
    expected = "".join(source.splitlines(keepends=True)[method.lineno - 1:method.end_lineno])
    if inspect.getsource(original) != expected:
        raise RuntimeError("Classic precision vLLM loader was already replaced")
    replacement = copy.deepcopy(method)
    keywords = [node for node in ast.walk(replacement)
                if isinstance(node, ast.keyword) and node.arg == "cpu_offload_gb"]
    if (len(keywords) != 1 or not isinstance(keywords[0].value, ast.Constant)
            or type(keywords[0].value.value) is not int or keywords[0].value.value != 24):
        raise RuntimeError("Expected exactly one cpu_offload_gb=24 literal")
    base_ast = ast.dump(method, include_attributes=False)
    keywords[0].value.value = value
    effective_ast = ast.dump(replacement, include_attributes=False)
    keywords[0].value.value = 24
    if ast.dump(replacement, include_attributes=False) != base_ast:
        raise RuntimeError("Resource override changed another loader setting")
    keywords[0].value.value = value
    namespace = {}
    exec(compile(ast.Module(body=[replacement], type_ignores=[]), inference.__file__, "exec"),
         vars(inference), namespace)
    evidence = {"base_loader_ast_sha256": hashlib.sha256(base_ast.encode()).hexdigest(),
                "effective_loader_ast_sha256": hashlib.sha256(effective_ast.encode()).hexdigest()}
    return namespace[method.name], evidence


def install():
    """Install only for an exact allowed environment value; import no GPU stack."""
    requested = os.environ.get(ENVIRONMENT_VARIABLE)
    if requested is None:
        return None
    if requested not in ALLOWED_VALUES:
        raise ValueError(ENVIRONMENT_VARIABLE + " must be exactly 12, 16, or 24")
    value = ALLOWED_VALUES[requested]
    spec = importlib.util.find_spec("classic_precision.inference")
    if spec is None or spec.origin is None or _sha(spec.origin) != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    inference = importlib.import_module("classic_precision.inference")
    if _sha(inference.__file__) != BASE_RUNNER_SHA256:
        raise RuntimeError("Classic precision base runner SHA mismatch")
    original = inference.ModelSession._load_vllm
    installed = getattr(original, "_classic_precision_resource_override", None)
    if installed is not None:
        if installed["requested_cpu_offload_gb"] != value:
            raise RuntimeError("Qwen resource override cannot change within a process")
        return dict(installed)
    replacement, evidence = _loader(inference, value)
    evidence.update(environment_variable=ENVIRONMENT_VARIABLE, family="qwen", backend="vllm",
                    base_cpu_offload_gb=24, requested_cpu_offload_gb=value,
                    effective_cpu_offload_gb=value, cpu_offload_params=["experts"],
                    helper_source_sha256=_sha(__file__), base_runner_sha256=BASE_RUNNER_SHA256)

    @functools.wraps(original)
    def load(self):
        if self.family != "qwen":
            return original(self)
        result = replacement(self)
        self.store.event("RUNTIME_RESOURCE_OVERRIDE", **evidence)
        return result

    load._classic_precision_resource_override = evidence
    inference.ModelSession._load_vllm = load
    return dict(evidence)
