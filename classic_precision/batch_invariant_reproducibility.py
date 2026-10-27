"""Finite original 49-input native Qwen / official batch-invariant diagnostic."""
from __future__ import annotations

import importlib.util
from pathlib import Path

from .storage import file_sha256
from runtime_compat import classic_precision_batch_invariant as runtime

SCHEMA = "classic_precision_native_qwen_batch_invariant/v1"
FROZEN_V3_ENTRY_SHA256 = "807596210a910315b0e791d2381b3339cd840debec72ec0e3c5b6d906f16ee26"
_path = Path(__file__).with_name("minimal_config_reproducibility_v3.py")
if file_sha256(_path) != FROZEN_V3_ENTRY_SHA256:
    raise RuntimeError("Frozen native UniProc entry changed")
_spec = importlib.util.spec_from_file_location("classic_precision._batch_invariant_frozen_v3", _path)
_v3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v3)
_base = _v3._base
_v3.runtime, _base.runtime, _base.SCHEMA = runtime, runtime, SCHEMA
_require = _base._require


def build_profile(profile_mode="qwen_batch_invariant", baseline_environment=None):
    value = _v3.build_profile(profile_mode, baseline_environment)
    value.update(vllm_envs_source_sha256=runtime.VLLM_ENVS_SHA256,
        frozen_native_observer_sha256=runtime.FROZEN_V3_HELPER_SHA256, certification=runtime.CERTIFICATION)
    value["implementation"].update(overlay_entry_sha256=file_sha256(__file__),
        frozen_native_entry_sha256=FROZEN_V3_ENTRY_SHA256)
    return value


def _validate_profile(profile, family, require_applied):
    value = _v3._validate_profile(profile, family, require_applied)
    _require(value.get("vllm_envs_source_sha256") == runtime.VLLM_ENVS_SHA256
        and value.get("frozen_native_observer_sha256") == runtime.FROZEN_V3_HELPER_SHA256,
        "Official BI observer source identity differs")
    return value


def _validate_batch(value, profile, scope):
    observation = value.get("batch_invariant_observation")
    expected = profile["startup_environment"][runtime.FLAG]
    _require(isinstance(observation, dict) and "observation_error" not in observation
        and observation.get("scope") == scope and observation.get("module") == "vllm.envs"
        and observation.get("expected_source_sha256") == observation.get("actual_source_sha256") == runtime.VLLM_ENVS_SHA256
        and observation.get("parser_native") is observation.get("getter_native") is True,
        "Actual official BI flag/source observation missing or invalid")
    _require(observation.get("environment_value") == expected
        and type(observation.get("parsed_value")) is bool and type(observation.get("effective_value")) is bool
        and observation["parsed_value"] == observation["effective_value"] == (expected == "1"),
        "Actual process environment/parser/effective BI flag differs")


def validate_runtime_policy(snapshot, family):
    value = _v3.validate_runtime_policy(snapshot, family)
    profile = value["profile"]
    _validate_batch(value, profile, "parent")
    for worker in value["worker_runtime"]:
        _validate_batch(worker, profile, "worker")
    return value


_base.build_profile, _base._validate_profile, _base.validate_runtime_policy = build_profile, _validate_profile, validate_runtime_policy
PROFILE_MODES = runtime.PROFILE_MODES
load_plan, run, _original_compare = _base.load_plan, _base.run, _base.compare
_new_harness = _v3._new_harness


def prepare_plan(source_plan, profile_mode="qwen_batch_invariant", baseline_environment=None):
    value = _base.prepare_plan(source_plan, profile_mode, baseline_environment)
    value["purpose"] = "Finite 49-input native Qwen diagnostic; BI arm adds only the official batch-invariant flag to the explicitly sealed baseline"
    return value


def compare(plan_path, *args, **kwargs):
    report = _original_compare(plan_path, *args, **kwargs)
    report["claim"] = "49 original inputs, two cold loads x two same-engine repeats, full input hashes and token/text comparison under the sealed native or official BI startup environment; no general determinism claim."
    return report


_base.compare = compare


def main(argv=None):
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
