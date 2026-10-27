"""Native Qwen / official vLLM batch-invariant observation; no Torch setters.

Only the BI arm adds VLLM_BATCH_INVARIANT=1 before engine startup. The private
v3 observer retains native UniProc launch, QSA, input hashes and token evidence.
Importing this module in a real worker installs its two empty-policy modes.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import functools
import importlib
import importlib.util
import inspect
import os
from pathlib import Path
import struct
from types import CodeType

FROZEN_V3_HELPER_SHA256 = "7afd0941e919c6c07a7b9ed075670781c7e68422eb88f41185330af9ed8299b2"
VLLM_ENVS_SHA256 = "1d7101800f67b803a700fbf7489292cf1f0022b570085166ad407e6b1513dc15"
FLAG = "VLLM_BATCH_INVARIANT"
EVIDENCE_CONTRACT = "native-qwen-official-batch-invariant-observation-v1"
CERTIFICATION = "FINITE_NATIVE_QWEN_STARTUP_SWITCH_DIAGNOSTIC"
PROFILE_MODES = ("native_qwen", "qwen_batch_invariant")
PROFILE_IDS = {mode: "classic-precision-" + mode.replace("_", "-") + "-bi-v1" for mode in PROFILE_MODES}
TORCH_POLICIES = {mode: {} for mode in PROFILE_MODES}

from runtime_compat.classic_precision_minimal_config_v2 import _sha

_path = Path(__file__).with_name("classic_precision_minimal_config_v3.py")
if _sha(_path) != FROZEN_V3_HELPER_SHA256:
    raise RuntimeError("Frozen native UniProc helper changed")
_spec = importlib.util.spec_from_file_location("runtime_compat._batch_invariant_frozen_v3", _path)
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)
_base, native = _v3._base, _v3.native
FROZEN_HELPER_SHA256, FROZEN_V2_HELPER_SHA256 = _v3.FROZEN_HELPER_SHA256, _v3.FROZEN_V2_HELPER_SHA256
BASE_RUNNER_SHA256, APPLICATION_PHASE = _v3.BASE_RUNNER_SHA256, _v3.APPLICATION_PHASE
THREAD_SOURCE_SHA256 = _v3.THREAD_SOURCE_SHA256
ENGINE_STARTUP_MODULE, ENGINE_STARTUP_SHA256 = _v3.ENGINE_STARTUP_MODULE, _v3.ENGINE_STARTUP_SHA256
TRACKED_ENVIRONMENT = tuple(sorted(set(_v3.TRACKED_ENVIRONMENT) | {FLAG, "NUMPY_MADVISE_HUGEPAGE"}))
_original_baseline, _original_worker = _base.baseline_environment, _v3._worker_runtime
_original_snapshot = _v3._original_snapshot


def family_for_mode(mode):
    if mode not in PROFILE_MODES:
        raise ValueError("Unknown native Qwen batch-invariant diagnostic mode")
    return "qwen"


def baseline_environment(environment=None):
    before = _original_baseline(environment)
    if before[FLAG] is not None:
        raise ValueError("Native baseline requires VLLM_BATCH_INVARIANT absent")
    return before


def environment_contract(mode, baseline=None):
    family_for_mode(mode)
    before = baseline_environment(baseline)
    interventions = {FLAG: "1"} if mode == "qwen_batch_invariant" else {}
    effective = {**before, **interventions}
    return before, interventions, {k: v for k, v in effective.items() if v is not None}, [k for k, v in effective.items() if v is None]


def _codes(code):
    yield code
    for value in code.co_consts:
        if isinstance(value, CodeType):
            yield from _codes(value)


def _same_constant(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(actual, CodeType):
        return _same_code(actual, expected)
    if isinstance(actual, tuple):
        return len(actual) == len(expected) and all(_same_constant(a, b) for a, b in zip(actual, expected))
    if isinstance(actual, frozenset):
        return len(actual) == len(expected) and all(any(_same_constant(a, b) for b in expected) for a in actual)
    if isinstance(actual, float):
        return struct.pack(">d", actual) == struct.pack(">d", expected)
    if isinstance(actual, complex):
        return _same_constant(actual.real, expected.real) and _same_constant(actual.imag, expected.imag)
    return actual == expected


def _same_code(actual, expected):
    """Exact code fields, including metadata; ignore marshal reference graphs.

    A live/pyc-loaded code object and a fresh compile can have different string
    interning and marshal reference bytes despite equal code fields. Compare
    every noncallable co_* attribute and recursively type-check constants instead.
    """
    if not isinstance(actual, CodeType) or not isinstance(expected, CodeType):
        return False
    fields = lambda code: {name for name in dir(code) if name.startswith("co_") and not callable(getattr(code, name))}
    names = fields(actual)
    return names == fields(expected) and all(_same_constant(getattr(actual, name), getattr(expected, name)) for name in names)


def _batch_observation(scope):
    """Read the actual process's official live flag and pinned parser code."""
    result = {"scope": scope, "environment_value": os.environ.get(FLAG), "module": "vllm.envs",
        "expected_source_sha256": VLLM_ENVS_SHA256}
    try:
        module = importlib.import_module("vllm.envs")
        path = Path(module.__file__)
        result["actual_source_sha256"] = _sha(path)
        if result["actual_source_sha256"] != VLLM_ENVS_SHA256:
            raise RuntimeError("Official vLLM envs source SHA differs")
        source = path.read_text()
        nodes = [value for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Dict)
            for key, value in zip(node.keys, node.values)
            if isinstance(key, ast.Constant) and key.value == FLAG and isinstance(value, ast.Lambda)]
        if len(nodes) != 1:
            raise RuntimeError("Official BI environment parser is missing or ambiguous")
        compiled = [code for code in _codes(compile(source, str(path), "exec", dont_inherit=True))
            if code.co_name == "<lambda>" and code.co_firstlineno == nodes[0].lineno]
        parser = module.environment_variables[FLAG]
        if (len(compiled) != 1 or not inspect.isfunction(parser) or parser.__module__ != "vllm.envs"
                or parser.__globals__ is not vars(module) or hasattr(parser, "__wrapped__")
                or not _same_code(parser.__code__, compiled[0])):
            raise RuntimeError("Official live BI parser identity/code differs")
        getter = module.__getattr__
        if isinstance(getter, type(functools.cache(lambda: None))):
            getter = getter.__wrapped__  # official enable_envs_cache keeps the original getter
        getter_codes = [code for code in _codes(compile(source, str(path), "exec", dont_inherit=True))
            if code.co_name == "__getattr__"]
        if (len(getter_codes) != 1 or not inspect.isfunction(getter) or getter.__globals__ is not vars(module)
                or hasattr(getter, "__wrapped__") or FLAG in vars(module)
                or not _same_code(getter.__code__, getter_codes[0])):
            raise RuntimeError("Official live BI getter identity/code differs")
        result.update(parser_native=True, getter_native=True, parsed_value=parser(), effective_value=getattr(module, FLAG))
    except Exception as error:
        result["observation_error"] = {"type": type(error).__name__, "message": str(error)}
    return result


def _worker_runtime(worker):
    value = _original_worker(worker)
    value["batch_invariant_observation"] = _batch_observation("worker")
    return value


def _worker_apply(worker, mode, environment_keys=TRACKED_ENVIRONMENT, engine_launch_parent=None):
    # This module is the RPC import target, so private mode tables are installed
    # in a fresh actual worker before v3 calls its unchanged empty-policy _apply.
    family_for_mode(mode)
    return _v3_worker_apply(worker, mode, environment_keys, engine_launch_parent)


_v3_worker_apply = _v3._worker_apply


def activate(family, mode="qwen_batch_invariant", environment=None):
    _v3.activate(family, mode, environment)
    _base._state.update(helper_sha256=_sha(__file__), frozen_native_observer_sha256=FROZEN_V3_HELPER_SHA256,
        vllm_envs_source_sha256=VLLM_ENVS_SHA256, certification=CERTIFICATION)
    return status()


def _snapshot_with_batch():
    value = _original_snapshot()
    value["comparison_identity"]["batch_invariant_observation"] = _batch_observation("parent")
    return value


_base.PROFILE_MODES, _base.PROFILE_IDS, _base.TORCH_POLICIES = PROFILE_MODES, PROFILE_IDS, TORCH_POLICIES
_base.TRACKED_ENVIRONMENT = TRACKED_ENVIRONMENT
_base.family_for_mode, _base.baseline_environment, _base.environment_contract = family_for_mode, baseline_environment, environment_contract
_v3.PROFILE_MODES, _v3.PROFILE_IDS, _v3.TORCH_POLICIES = PROFILE_MODES, PROFILE_IDS, TORCH_POLICIES
_v3.TRACKED_ENVIRONMENT, _v3.EVIDENCE_CONTRACT = TRACKED_ENVIRONMENT, EVIDENCE_CONTRACT
_v3._worker_apply, _v3._worker_runtime, _base._worker_runtime = _worker_apply, _worker_runtime, _worker_runtime
_v3._original_snapshot = _snapshot_with_batch
status, prepare_native, runtime_snapshot = _v3.status, _v3.prepare_native, _v3.runtime_snapshot
