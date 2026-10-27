"""QSA selector alone after native loading; no Torch, MP or BLAS setters."""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import sys

FROZEN_NATIVE_HELPER_SHA256 = "f9a4c6697e26ba1316c7c65ec29d8f9c1e16406c13abb2f167d65c8632c40015"
NATIVE_GUARD_SHA256 = "01b814af8f5d634885e5ba4316a063ccd607ecded6c925cd7be3f02d5f6e4f8c"
QSA_HELPER_SHA256 = "bf3f16fc3092b4edc6a8308dbad49bc84efacc327a88e2f697ae6bb3b38b7624"
FLAG = "CLASSIC_PRECISION_QSA_DETERMINISM"
PROFILE_MODES = ("native_qwen", "qwen_qsa_only")
PROFILE_IDS = {mode: "classic-precision-" + mode.replace("_", "-") + "-selector-v1" for mode in PROFILE_MODES}
TORCH_POLICIES = {mode: {} for mode in PROFILE_MODES}
EVIDENCE_CONTRACT = "native-qwen-qsa-selector-only-observation-v1"
CERTIFICATION = "FINITE_NATIVE_LOAD_QSA_SELECTOR_ONLY_DIAGNOSTIC"

from runtime_compat.classic_precision_minimal_config_v2 import _sha


def _private(name, filename, expected):
    path = Path(__file__).with_name(filename)
    if _sha(path) != expected:
        raise RuntimeError("Frozen QSA-only dependency changed: " + filename)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_observer = _private("runtime_compat._qsa_only_native_observer", "classic_precision_batch_invariant.py", FROZEN_NATIVE_HELPER_SHA256)
native = _private("runtime_compat._qsa_only_native_guard", "classic_precision_config_only.py", NATIVE_GUARD_SHA256)
_base, _v3 = _observer._base, _observer._v3
FROZEN_HELPER_SHA256, FROZEN_V2_HELPER_SHA256 = _observer.FROZEN_HELPER_SHA256, _observer.FROZEN_V2_HELPER_SHA256
FROZEN_V3_HELPER_SHA256 = _observer.FROZEN_V3_HELPER_SHA256
VLLM_ENVS_SHA256, TRACKED_ENVIRONMENT = _observer.VLLM_ENVS_SHA256, _observer.TRACKED_ENVIRONMENT
BASE_RUNNER_SHA256, APPLICATION_PHASE = _observer.BASE_RUNNER_SHA256, _observer.APPLICATION_PHASE
THREAD_SOURCE_SHA256 = _observer.THREAD_SOURCE_SHA256
ENGINE_STARTUP_MODULE, ENGINE_STARTUP_SHA256 = _observer.ENGINE_STARTUP_MODULE, _observer.ENGINE_STARTUP_SHA256
_same_code, _codes = _observer._same_code, _observer._codes
_native_source_evidence, _native_verify = native._qsa_source_evidence, native.verify_native_qsa
_native_assert, _native_probe = native.assert_no_qsa_replacement, native._install_hashed_vision_probe
_raw_snapshot = _observer._original_snapshot
_original_status = _v3._original_status
_original_function, _replacement_function, _selector_function = None, None, None
_last_proof = None


def family_for_mode(mode):
    if mode not in PROFILE_MODES:
        raise ValueError("Unknown QSA-only mode")
    return "qwen"


def environment_contract(mode, baseline=None):
    family_for_mode(mode)
    before = _observer.baseline_environment(baseline)
    if before["NUMPY_MADVISE_HUGEPAGE"] != "0":
        raise ValueError("This diagnostic requires explicit NUMPY_MADVISE_HUGEPAGE=0 baseline")
    interventions = {FLAG: "1"} if mode == "qwen_qsa_only" else {}
    effective = {**before, **interventions}
    return before, interventions, {k: v for k, v in effective.items() if v is not None}, [k for k, v in effective.items() if v is None]


baseline_environment = _observer.baseline_environment


def selector_contract():
    return {"environment_variable": FLAG, "target_module": native.QSA_TARGET_MODULE,
        "model_architecture": "Qwen4ExpForConditionalGeneration", "expected_native_source_sha256": native.QSA_NATIVE_SOURCE_SHA256,
        "helper_source_sha256": QSA_HELPER_SHA256, "algorithm": "stable_score_descending_index_ties_then_index_ascending",
        "padding": -1, "visible_nan_policy": "reject", "max_sort_elements": 1 << 20}


def selector_receipt(original, state):
    return {"state": state, "selector_contract": selector_contract(),
        "native_original": {key: value for key, value in original.items() if key != "state"},
        "replacement_enabled": True, "observer_only": False}


def _selector():
    names = ("runtime_compat.classic_precision_qsa_determinism", "classic_precision_qsa_determinism")
    loaded = [sys.modules[name] for name in names if name in sys.modules]
    if loaded and any(module is not loaded[0] for module in loaded):
        raise RuntimeError("Multiple QSA selector module instances")
    module = loaded[0] if loaded else importlib.import_module(names[0])
    if _sha(module.__file__) != QSA_HELPER_SHA256:
        raise RuntimeError("Reviewed QSA selector source changed")
    return module


def _assert_qsa_scope():
    if os.environ.get("VLLM_BATCH_INVARIANT") is not None:
        raise RuntimeError("QSA-only requires BI absent")
    if os.environ.get(FLAG) == "0":
        return _native_assert()
    if os.environ.get(FLAG) != "1":
        raise RuntimeError("QSA-only flag must be exactly 0 or 1")
    loaded = [module for name, module in sys.modules.items() if name.rsplit(".", 1)[-1] == "classic_precision_qsa_determinism"]
    if loaded:
        selector = _selector()
        if any(module is not selector for module in loaded):
            raise RuntimeError("Unreviewed QSA module present")
        finders = [finder for finder in sys.meta_path if "classic_precision_qsa_determinism" in type(finder).__module__]
        if finders and (len(finders) != 1 or finders[0] is not selector._finder or type(finders[0]) is not selector._QSAFinder):
            raise RuntimeError("Unreviewed QSA import hook present")


def _guard_function(function, module, expected):
    if (not inspect.isfunction(function) or function.__globals__ is not vars(module)
            or function.__defaults__ is not None or function.__kwdefaults__ not in (None, {})
            or hasattr(function, "__wrapped__") or not _same_code(function.__code__, expected)):
        raise RuntimeError("QSA selector live function/code changed")


def _verify_qsa(module=None):
    global _original_function, _replacement_function, _selector_function, _last_proof
    _assert_qsa_scope()
    if os.environ[FLAG] == "0":
        return _native_verify(module)
    selector = _selector()
    selector.install()  # only this reviewed leaf, after native load; no Torch policy setters
    module = importlib.import_module(native.QSA_TARGET_MODULE) if module is None else module
    expected_original, original_identity = _native_source_evidence(module.__file__)
    wrapped = getattr(module, "_topk", None)
    original = getattr(wrapped, "__wrapped__", None)
    expected = {code.co_qualname: code for code in _codes(compile(Path(selector.__file__).read_bytes(), selector.__file__, "exec", dont_inherit=True))}
    _guard_function(original, module, expected_original)
    if (original.__module__ != native.QSA_TARGET_MODULE or original.__qualname__ != "_topk"
            or tuple(inspect.signature(original).parameters) != native._PARAMETERS
            or not inspect.isfunction(wrapped) or wrapped.__globals__ is not vars(selector)
            or wrapped.__defaults__ is not None or wrapped.__kwdefaults__ not in (None, {})
            or wrapped.__module__ != native.QSA_TARGET_MODULE or wrapped.__qualname__ != "_topk"
            or not _same_code(wrapped.__code__, expected["_apply.<locals>.topk"])):
        raise RuntimeError("QSA replacement wrapper/original identity differs")
    for name in ("exact_qsa_topk", "_shape_checks"):
        _guard_function(getattr(selector, name), selector, expected[name])
    marker = {**selector_contract(), "state": "applied"}
    if (getattr(wrapped, selector._MARKER, None) != marker or selector.status() != marker
            or selector.MAX_SORT_ELEMENTS != 1 << 20 or selector._PARAMETERS != native._PARAMETERS
            or selector._finder not in sys.meta_path or type(selector._finder) is not selector._QSAFinder):
        raise RuntimeError("Reviewed QSA selector installation/algorithm contract differs")
    current = (original, wrapped, selector.exact_qsa_topk)
    previous = (_original_function, _replacement_function, _selector_function)
    if any(before is not None and before is not now for before, now in zip(previous, current)):
        raise RuntimeError("QSA live original/replacement/selector object changed")
    _original_function, _replacement_function, _selector_function = current
    _last_proof = {"selector_source_sha256": QSA_HELPER_SHA256, "native_source_sha256": native.QSA_NATIVE_SOURCE_SHA256,
        "application_phase": APPLICATION_PHASE, "wrapper_code_exact": True, "original_code_exact": True,
        "selector_code_exact": True, "shape_checks_code_exact": True, "original_object_stable": True,
        "replacement_object_stable": True, "selector_object_stable": True, "reviewed_hook_only": True}
    return selector_receipt(original_identity, "applied")


def _status_before_or_after_application():
    if _base._state.get("mode") != "qwen_qsa_only":
        return _original_status()
    value = deepcopy(_base._state)
    if _base._application:
        value["qsa_installation"] = _verify_qsa()
    else:
        original = _native_source_evidence(native._native_source_path())[1]
        value["qsa_installation"] = selector_receipt(original, "pending_target_import")
    return value


def _worker_runtime(worker):
    value = _observer._worker_runtime(worker)
    value["qsa_live_observation"] = deepcopy(_last_proof) if os.environ.get(FLAG) == "1" else None
    return value


def _worker_apply(worker, mode, environment_keys=TRACKED_ENVIRONMENT, engine_launch_parent=None):
    family_for_mode(mode)
    return _observer._worker_apply(worker, mode, environment_keys, engine_launch_parent)


def _install_vision_probe(worker):
    return _native_probe(worker)


def _snapshot():
    value = _raw_snapshot()
    identity = value["comparison_identity"]
    selector_mode = os.environ.get(FLAG) == "1"
    imported = any(name.rsplit(".", 1)[-1] == "classic_precision_qsa_determinism" for name in sys.modules)
    hooked = any("classic_precision_qsa_determinism" in type(finder).__module__ for finder in sys.meta_path)
    identity.update(qsa_replacement_module_imported=imported, qsa_replacement_import_hook=hooked,
        qsa_live_observation=deepcopy(_last_proof) if selector_mode else None)
    return value


def activate(family, mode="qwen_qsa_only", environment=None):
    _observer.activate(family, mode, environment)
    _base._state.update(helper_sha256=_sha(__file__), qsa_selector_helper_sha256=QSA_HELPER_SHA256,
        native_observer_helper_sha256=FROZEN_NATIVE_HELPER_SHA256, certification=CERTIFICATION)
    return status()


native.assert_no_qsa_replacement, native.verify_native_qsa = _assert_qsa_scope, _verify_qsa
native._install_hashed_vision_probe = _install_vision_probe
_base.native, _v3.native, _observer.native = native, native, native
_base.PROFILE_MODES, _base.PROFILE_IDS, _base.TORCH_POLICIES = PROFILE_MODES, PROFILE_IDS, TORCH_POLICIES
_base.family_for_mode, _base.environment_contract = family_for_mode, environment_contract
_v3.PROFILE_MODES, _v3.PROFILE_IDS, _v3.TORCH_POLICIES = PROFILE_MODES, PROFILE_IDS, TORCH_POLICIES
_observer.PROFILE_MODES, _observer.PROFILE_IDS = PROFILE_MODES, PROFILE_IDS
_observer.CERTIFICATION, _v3.EVIDENCE_CONTRACT = CERTIFICATION, EVIDENCE_CONTRACT
_v3._original_status, _observer._original_snapshot = _status_before_or_after_application, _snapshot
_v3._worker_apply, _base._worker_runtime = _worker_apply, _worker_runtime
status, prepare_native, runtime_snapshot = _observer.status, _observer.prepare_native, _observer.runtime_snapshot
