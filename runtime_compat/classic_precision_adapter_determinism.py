"""Opt-in instrumentation around source-bound native task adapters.

Call activate() in a fresh process before Torch/model imports, then construct the
unchanged native session and call instrument_session(). This keeps strict-v1's
Torch/QSA initialization, removes only its incompatible video wrappers, and uses
an independent vision hook alongside each adapter's original strong guards.

This module neither loads a model nor implements a queue/certifier. Gemma's
standalone single-image loop has no generate_image API: use handle.capture(row,
operation) with an operation returning {"raw": raw, "post": post}. The qualified
native validate_gemma is then called explicitly before recording a GPU receipt.
Every repeat must request the native deep checks. GPU execution is unvalidated
until the resulting native and REPRO_INPUT_HASH receipts pass a finite suite.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import sys
from types import CodeType


PROFILE_ID = "classic-precision-native-adapter-instrumentation-v1"
_state = {}
_PROBE_ATTRIBUTE = "_classic_precision_adapter_repro_probe"
_METHODS = {"video": ("prepare", "generate"), "multi": ("prepare_images", "generate_images"),
            "single": ("prepare_image", "generate_image"), "pair": ("prepare_pair", "generate_pair")}


def _require(condition, message):
    if not condition:
        raise RuntimeError(message)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _core():
    names = ("classic_precision_determinism", "runtime_compat.classic_precision_determinism")
    loaded = [sys.modules[n] for n in names if n in sys.modules]
    _require(not loaded or all(x is loaded[0] for x in loaded), "Multiple strict-core module instances")
    if loaded:
        return loaded[0]
    try:
        return importlib.import_module(names[0])
    except ModuleNotFoundError as error:
        if error.name != names[0]:
            raise
        return importlib.import_module(names[1])


def _coverage(path):
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    _require(value.get("schema") == "classic_precision_input_path_coverage/v1", "Unknown coverage schema")
    implementation = value.get("frozen_strict_profile", {}).get("implementation", {})
    _require(bool(implementation.get("determinism_helper_sha256")) and
             implementation["determinism_helper_sha256"] == _sha(_core().__file__), "Strict-core source SHA mismatch")
    return path, value


def _code_objects(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _code_objects(value)


def _source_guard(function, allowed, wrapped=False):
    """Check file bytes AND the live function bytecode before wrapping it."""
    function = getattr(function, "__func__", function)
    _require(inspect.isfunction(function), "Expected an inspectable native Python method")
    _require(wrapped or function.__qualname__ == function.__code__.co_qualname,
             "Native callable API identity differs from its live code")
    path = Path(inspect.getsourcefile(function) or "").resolve()
    _require(path.is_file(), "Native callable source is unavailable")
    identities = {(Path(p).name, sha) for p, sha in allowed.items()}
    _require((path.name, _sha(path)) in identities, "Native callable source SHA is unqualified: " + str(path))
    compiled = compile(path.read_bytes(), function.__code__.co_filename, "exec", dont_inherit=True)
    expected = [c for c in _code_objects(compiled) if c.co_qualname == function.__code__.co_qualname]
    # Marshal's reference/interning flags can differ for equal code loaded by
    # importlib versus freshly compiled code. Compare code objects and location
    # tables directly, rather than rejecting those serialization differences.
    _require(len(expected) == 1 and expected[0] == function.__code__
             and expected[0].co_firstlineno == function.__code__.co_firstlineno
             and expected[0].co_linetable == function.__code__.co_linetable
             and expected[0].co_exceptiontable == function.__code__.co_exceptiontable,
             "Native callable bytecode differs from its qualified source: " + function.__qualname__)


def activate(family, coverage_path, base_module=None):
    """Keep strict-v1 core policy, restoring only its video-only wrappers.

    Files and adapter methods are not modified. Session construction and model
    loading remain the caller's explicit responsibility in this fresh process.
    """
    _require(family in ("nemo", "qwen", "cosmos", "gemma"), "Unsupported family")
    path, coverage = _coverage(coverage_path)
    if _state:
        _require(_state["family"] == family and _state["coverage_sha256"] == _sha(path),
                 "Adapter instrumentation already activated with a different binding")
        return status()
    _require("torch" not in sys.modules and "vllm" not in sys.modules,
             "Activate native-adapter instrumentation before importing Torch/vLLM")
    core = _core()
    qsa_sha = None
    if family == "qwen":
        qsa_path = Path(core.__file__).with_name("classic_precision_qsa_determinism.py")
        qsa_sha = coverage["frozen_strict_profile"]["implementation"].get("qsa_helper_sha256")
        _require(qsa_path.is_file() and qsa_sha == _sha(qsa_path), "Qualified QSA helper source SHA mismatch")
    core.activate(family)
    base = base_module or importlib.import_module("classic_precision.inference")
    expected = coverage["frozen_strict_profile"]["implementation"]["base_runner_sha256"]
    _require(_sha(base.__file__) == expected, "Frozen native inference source SHA mismatch")
    # Validate both originals before either mutation. __wrapped__ alone is not
    # authority: both the wrapper's and original's live code must match source.
    originals = {}
    for name in ("prepare", "generate"):
        wrapped = getattr(base.ModelSession, name)
        _source_guard(wrapped, {core.__file__: _sha(core.__file__)}, wrapped=True)
        native = getattr(wrapped, "__wrapped__", None)
        _require(native is not None, "Strict video wrapper cannot be safely separated: " + name)
        _source_guard(native, {base.__file__: expected})
        _require(inspect.getclosurevars(wrapped).nonlocals.get("original_" + name) is native,
                 "Strict wrapper's original callable binding differs")
        originals[name] = native
    for name, native in originals.items():
        setattr(base.ModelSession, name, native)
    _state.update(family=family, coverage_sha256=_sha(path), coverage_path=str(path),
                  core=core, base=base, core_source_sha256=_sha(core.__file__),
                  base_source_sha256=expected, qsa_source_sha256=qsa_sha, module_sha256=_sha(__file__))
    return status()


def status():
    if not _state:
        return {"profile_id": PROFILE_ID, "state": "INACTIVE", "certification": "NOT_VALIDATED"}
    return {"profile_id": PROFILE_ID, "state": "CORE_ACTIVE_VIDEO_WRAPPERS_SEPARATED",
            **{k: v for k, v in _state.items() if k not in ("core", "base")},
            "core": _state["core"].status(), "certification": "NOT_VALIDATED",
            "sdk_files_modified": False}


def _assert_policy(family, require_qsa=False):
    import torch
    _require(_state and _state["family"] == family, "Native adapter core has not been activated")
    core = _state["core"]
    _require(_sha(core.__file__) == _state["core_source_sha256"], "Strict core source changed after activation")
    _require(_sha(__file__) == _state["module_sha256"] and _sha(_state["base"].__file__) == _state["base_source_sha256"],
             "Instrumentation/native source changed after activation")
    _require(all(os.environ.get(k) == v for k, v in core.launch_environment().items()),
             "Strict runtime environment changed")
    _require(torch.are_deterministic_algorithms_enabled() and
             not torch.is_deterministic_algorithms_warn_only_enabled() and
             not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic,
             "Strict Torch policy is not enforced")
    fill = getattr(core, "FILL_UNINITIALIZED_MEMORY", None)
    _require(isinstance(fill, dict) and type(fill.get(family)) is bool
             and torch.utils.deterministic.fill_uninitialized_memory is fill[family],
             "Strict family fill_uninitialized_memory policy changed")
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        _require(torch.backends.cuda.matmul.fp32_precision == "ieee" and
                 torch.backends.cudnn.conv.fp32_precision == "ieee" and
                 torch.backends.cudnn.rnn.fp32_precision == "ieee", "Strict TF32 precision policy changed")
    else:
        _require(torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False,
                 "Strict TF32 precision policy changed")
    if require_qsa and family == "qwen":
        qsa = core.status().get("qsa_installation")
        _require(isinstance(qsa, dict) and qsa.get("state") == "applied" and
                 qsa.get("helper_source_sha256") == _state["qsa_source_sha256"], "Qwen QSA hook is not applied/qualified")


def _cpu_tensors(batch):
    import torch
    _require(isinstance(batch, dict) or hasattr(batch, "items"), "Native processor did not return a tensor batch")
    tensors = []
    def visit(value, name):
        if isinstance(value, torch.Tensor):
            _require(value.device.type == "cpu", "Processor fingerprint must capture the original CPU tensor")
            tensors.append(_core().tensor_fingerprint(value, name))
        elif isinstance(value, (tuple, list)):
            for index, child in enumerate(value):
                visit(child, name + "[" + str(index) + "]")
    for key, value in sorted(batch.items()):
        visit(value, key)
    names = {t["name"] for t in tensors}
    _require("input_ids" in names and any(n.startswith("pixel_values") for n in names),
             "Native CPU input_ids/pixel tensors are missing")
    return tensors


def _vision_target(model, family):
    paths = {"nemo": ("vision_model",), "qwen": ("visual",),
             "cosmos": ("model.visual", "visual"),
             "gemma": ("model.vision_tower", "vision_tower", "model.model.vision_tower")}[family]
    for path in paths:
        target = model
        for part in path.split("."):
            target = getattr(target, part, None)
            if target is None:
                break
        if target is not None and callable(getattr(target, "register_forward_pre_hook", None)):
            return target
    raise RuntimeError("Unsupported actual vision entry for " + family)


def _gpu_tensors(module, args, kwargs, family):
    import torch
    pixels = kwargs.get("pixel_values", kwargs.get("pixel_values_videos"))
    if pixels is None and args:
        pixels = args[0]
    _require(isinstance(pixels, torch.Tensor) and pixels.device.type == "cuda",
             "Actual vision input must be a CUDA tensor")
    _require(pixels.dtype == torch.bfloat16, "Actual native vision dtype changed from BF16")
    result = [_core().tensor_fingerprint(pixels, "pixels")]
    positions = {k: v for k, v in kwargs.items() if "position" in k and isinstance(v, torch.Tensor)}
    if family == "gemma" and not positions:
        try:
            bound = inspect.signature(module.forward).bind_partial(*args, **kwargs)
            positions = {k: v for k, v in bound.arguments.items() if "position" in k and isinstance(v, torch.Tensor)}
        except (TypeError, ValueError):
            pass
    if family == "gemma":
        _require(bool(positions), "Actual Gemma vision position tensor was not observed")
    for name, tensor in sorted(positions.items()):
        _require(tensor.device.type == "cuda", "Actual vision positions are not on CUDA")
        result.append(_core().tensor_fingerprint(tensor, name))
    return result


def _install_worker_probe(worker, family):
    _require(not hasattr(worker, _PROBE_ATTRIBUTE), "Adapter vision probe is already active")
    receipts = []
    target = _vision_target(worker.get_model(), family)
    def before(module, args, kwargs):
        receipts.append(_gpu_tensors(module, args, kwargs, family))
    handle = target.register_forward_pre_hook(before, with_kwargs=True)
    setattr(worker, _PROBE_ATTRIBUTE, (handle, receipts))
    return True


def _read_worker_probe(worker):
    _require(hasattr(worker, _PROBE_ATTRIBUTE), "Adapter vision probe is missing")
    handle, receipts = getattr(worker, _PROBE_ATTRIBUTE)
    try:
        handle.remove()
        return receipts
    finally:
        delattr(worker, _PROBE_ATTRIBUTE)


class SessionInstrumentation:
    """Reversible instance wrappers; never writes or fabricates native raw output."""
    def __init__(self, session, route, adapter_module, coverage_path):
        _require(_state and route in _METHODS, "Unsupported/inactive native adapter route")
        path, coverage = _coverage(coverage_path)
        _require(_sha(path) == _state["coverage_sha256"], "Coverage changed after activation")
        entries = [r for r in coverage["routes"] if r.get("route_id") == route]
        _require(len(entries) == 1, "Route is not uniquely source-bound in coverage")
        entry = entries[0]
        allowed = {r["path"]: r["sha256"] for r in entry["implementation_files"]}
        module_path = Path(adapter_module.__file__).resolve()
        _require((module_path.name, _sha(module_path)) in {(Path(p).name, h) for p, h in allowed.items()},
                 "Adapter module source is unqualified")
        _require(getattr(session, "family", None) == _state["family"], "Session family differs from activated core")
        _require(not getattr(session, "_native_adapter_repro_instrumentation", None), "Session already instrumented")
        self.session, self.route, self.adapter = session, route, adapter_module
        self.binding = {"profile_id": PROFILE_ID, "coverage_sha256": _state["coverage_sha256"],
                        "adapter_sha256": _sha(module_path), "instrumentation_sha256": _state["module_sha256"],
                        "route": route, "family": session.family}
        self._originals, self._active, self._cpu_events = {}, None, 0
        self._active_row_sha = self._prepared_receipt = None
        self._allowed = allowed
        prepare, generate = _METHODS[route]
        native_prepare = getattr(session, prepare, None)
        _source_guard(native_prepare, allowed)
        native_generate = getattr(session, generate, None)
        # The frozen Gemma single-image loop lives in run_gemma, not a method.
        self.explicit_generation = route == "single" and session.family == "gemma" and native_generate is None
        _require(native_generate is not None or self.explicit_generation, "Native generation method is unsupported")
        if native_generate is not None:
            _source_guard(native_generate, allowed)
        self.explicit_validator = None
        if self.explicit_generation:
            self.explicit_validator = getattr(adapter_module, "validate_gemma", None)
            _source_guard(self.explicit_validator, allowed)
        if route == "video":
            # These are exactly the bindings made by the original T1 run().
            base = _state["base"]
            self._video_guard()
            bindings = {}
            for target, source in (("decode_rgb16", "decode_t1"), ("_install_vision_probe", "_install_strong_probe"),
                                   ("_read_vision_probe", "_read_strong_probe")):
                function = getattr(adapter_module, source, None)
                _source_guard(function, allowed)
                bindings[target] = function
            for target, function in bindings.items():
                setattr(base, target, function)
        self._replace(prepare, self._prepare(native_prepare))
        if native_generate is not None:
            @functools.wraps(native_generate)
            def wrapped(*args, **kwargs):
                bound = inspect.signature(native_generate).bind(*args, **kwargs)
                bound.apply_defaults()
                _require("row" in bound.arguments, "Native generation row is unavailable")
                if "deep" in bound.arguments:
                    _require(bound.arguments["deep"] is True, "Every strict repeat requires native deep=True")
                return self.capture(bound.arguments["row"], lambda: native_generate(*args, **kwargs))
            self._replace(generate, wrapped)
        session._native_adapter_repro_instrumentation = self

    def _video_guard(self):
        guard = self.session.store
        cls = getattr(self.adapter, "GuardStore", None)
        _require(cls is not None and type(guard) is cls,
                 "T1 video requires its original source-qualified GuardStore")
        _source_guard(guard.event, self._allowed)
        validator = getattr(self.adapter, "validate_post", None)
        _source_guard(validator, self._allowed)
        _require(guard.event.__func__.__globals__.get("validate_post") is validator,
                 "T1 GuardStore validator binding differs")
        _require(guard.session is self.session, "T1 GuardStore is bound to another session")
        _require(guard.base is _state["base"], "T1 GuardStore is bound to another native base")

    def _explicit_receipts(self, result, receipts):
        _require(isinstance(result, dict) and isinstance(result.get("raw"), dict)
                 and isinstance(result.get("post"), dict), "Explicit Gemma operation must return actual raw/post records")
        raw, post = result["raw"], result["post"]
        receipt = raw.get("input_receipt")
        _require(isinstance(receipt, dict) and receipt.get("deep_check") is True
                 and self._prepared_receipt is not None and _row_sha(receipt) == self._prepared_receipt,
                 "Explicit Gemma raw input receipt differs from actual deep preparation")
        forwards = post.get("actual_GPU_visual_forward")
        _require(isinstance(forwards, list) and len(forwards) == len(receipts) == 1,
                 "Explicit Gemma post does not match actual GPU captures")
        actual = {t["name"]: t for t in receipts[0]}
        pixels = actual.get("pixels")
        positions = [t for name, t in actual.items() if "position" in name]
        _require(pixels is not None and len(positions) == 1,
                 "Explicit Gemma actual pixel/position captures missing")
        forward = forwards[0]
        _require(forward.get("shape") == pixels["shape"] == receipt.get("actual_processor_tensor_shape")
                 and forward.get("dtype") == pixels["dtype"] == "torch.bfloat16"
                 and str(forward.get("device", "")).startswith("cuda")
                 and forward.get("tensor_sha256") == pixels["sha256"] == receipt.get("expected_BF16_GPU_sha256")
                 and forward.get("positions_sha256") == positions[0]["sha256"] == receipt.get("image_position_ids_sha256"),
                 "Explicit Gemma post differs from independently captured GPU tensors")
        before = _row_sha(result)
        self.explicit_validator(raw, post)
        _require(_row_sha(result) == before, "Explicit Gemma validator changed actual raw/post records")

    def _replace(self, name, value):
        self._originals[name] = (name in vars(self.session), vars(self.session).get(name))
        setattr(self.session, name, value)

    def _prepare(self, original):
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            _require(self._active is not None, "Call native preparation inside a captured generation")
            bound = inspect.signature(original).bind(*args, **kwargs)
            _require(isinstance(bound.arguments.get("row"), dict)
                     and _row_sha(bound.arguments["row"]) == self._active_row_sha,
                     "Native preparation row differs from captured scientific row")
            result = original(*args, **kwargs)  # native scientific guards execute first
            index = 0 if self.explicit_generation else 1
            _require(isinstance(result, tuple) and len(result) > index, "Unsupported native processor return API")
            receipt_index = 1 if self.explicit_generation else 3
            _require(len(result) > receipt_index and isinstance(result[receipt_index], dict), "Native input receipt missing")
            receipt = result[receipt_index]
            if "deep_check" in receipt:
                _require(receipt["deep_check"] is True, "Native processor deep qualification is disabled")
            tensors = _cpu_tensors(result[index])
            if self.explicit_generation:
                _require(receipt.get("deep_check") is True, "Explicit Gemma requires actual deep preparation")
                self._prepared_receipt = _row_sha(receipt)
            self.session.store.event("REPRO_INPUT_HASH", sample_id=self._active,
                                     row_sha256=self._active_row_sha, stage="cpu_processor", tensors=tensors, **self.binding)
            self._cpu_events += 1
            return result
        return wrapped

    def capture(self, row, operation):
        """Wrap one native operation; errors propagate and no raw output is invented.

        For the explicit Gemma loop, operation must call wrapped prepare_image
        and return a dict containing the actual raw/post records. The unchanged,
        source-qualified validate_gemma is always invoked on those records.
        """
        _require(self._active is None and isinstance(row, dict) and bool(row.get("sample_id")), "Nested capture or missing sample identity")
        _assert_policy(self.session.family)
        if self.route == "video":
            _require(getattr(self.session, "hash_deep", None) is True, "T1 video requires native hash_deep=True")
            self._video_guard()
        self._active, self._cpu_events = row["sample_id"], 0
        self._prepared_receipt = None
        self._active_row_sha = _row_sha(row)
        engine = getattr(self.session, "engine", None)
        local_hook, receipts, installed = None, [], False
        try:
            if engine is not None:
                installed = True
                results = engine.collective_rpc(_install_worker_probe, args=(self.session.family,))
                _require(results and all(x is True for x in results), "Actual worker probe installation failed")
            else:
                target = _vision_target(self.session.model, self.session.family)
                def before(module, args, kwargs):
                    receipts.append(_gpu_tensors(module, args, kwargs, self.session.family))
                local_hook = target.register_forward_pre_hook(before, with_kwargs=True)
            result = operation()
            if engine is not None:
                worker_receipts = engine.collective_rpc(_read_worker_probe)
                installed = False
                receipts = [one for worker in worker_receipts for one in worker]
            if self.explicit_validator is not None:
                self._explicit_receipts(result, receipts)
            _assert_policy(self.session.family, require_qsa=True)
            _require(_row_sha(row) == self._active_row_sha, "Scientific row changed during native generation")
            _require(self._cpu_events == 1, "Expected exactly one actual native CPU processor return")
            _require(bool(receipts) and all(bool(r) for r in receipts), "Actual GPU vision tensors were not captured")
            self.session.store.event("REPRO_INPUT_HASH", sample_id=self._active, stage="gpu_forward",
                                     row_sha256=self._active_row_sha, tensors=[t for call in receipts for t in call], **self.binding)
            return result
        finally:
            original_error = sys.exc_info()[1]
            try:
                if local_hook is not None:
                    local_hook.remove()
                if installed:
                    try:
                        engine.collective_rpc(_read_worker_probe)
                    except BaseException as cleanup_error:
                        if original_error is None:
                            raise
                        original_error.add_note("Adapter probe cleanup also failed: " + str(cleanup_error))
            finally:
                self._active = None
                self._active_row_sha = None
                self._prepared_receipt = None

    def close(self):
        _require(self._active is None, "Cannot restore a session during capture")
        for name, (owned, value) in self._originals.items():
            if owned:
                setattr(self.session, name, value)
            else:
                delattr(self.session, name)
        self._originals.clear()
        if getattr(self.session, "_native_adapter_repro_instrumentation", None) is self:
            delattr(self.session, "_native_adapter_repro_instrumentation")


def instrument_session(session, route, adapter_module, coverage_path):
    return SessionInstrumentation(session, route, adapter_module, coverage_path)
