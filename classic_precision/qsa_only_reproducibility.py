"""Finite native Qwen versus reviewed QSA selector alone, with native loading."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

from .storage import file_sha256
from runtime_compat import classic_precision_qsa_only as runtime

SCHEMA = "classic_precision_native_qwen_qsa_only/v1"
FROZEN_NATIVE_ENTRY_SHA256 = "94845dfa20a6e2e84a43bc7b30360f34d84c4bfaf2bd362488f8bd3ba88f51c1"
_path = Path(__file__).with_name("batch_invariant_reproducibility.py")
if file_sha256(_path) != FROZEN_NATIVE_ENTRY_SHA256:
    raise RuntimeError("Frozen native Qwen diagnostic entry changed")
_spec = importlib.util.spec_from_file_location("classic_precision._qsa_only_native_entry", _path)
_native_entry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native_entry)
_base = _native_entry._base
_native_entry.runtime = _native_entry._v3.runtime = _base.runtime = runtime
_base.SCHEMA = SCHEMA
_require = _base._require


def build_profile(profile_mode="qwen_qsa_only", baseline_environment=None):
    value = _native_entry.build_profile(profile_mode, baseline_environment)
    value.update(qsa_policy="reviewed_exact_selector_only" if profile_mode == "qwen_qsa_only" else "native_original_unhooked",
        qsa_selector_helper_sha256=runtime.QSA_HELPER_SHA256,
        native_observer_helper_sha256=runtime.FROZEN_NATIVE_HELPER_SHA256)
    value["implementation"].update(overlay_entry_sha256=file_sha256(__file__),
        frozen_native_diagnostic_entry_sha256=FROZEN_NATIVE_ENTRY_SHA256,
        qsa_selector_helper_sha256=runtime.QSA_HELPER_SHA256)
    return value


def _validate_profile(profile, family, require_applied):
    _require(isinstance(profile, dict), "QSA-only runtime profile missing")
    _require(profile.get("qsa_selector_helper_sha256") == runtime.QSA_HELPER_SHA256
        and profile.get("native_observer_helper_sha256") == runtime.FROZEN_NATIVE_HELPER_SHA256,
        "QSA-only selector/native observer source differs")
    view = deepcopy(profile)
    if profile.get("mode") == "qwen_qsa_only":
        qsa = profile.get("qsa_installation", {})
        _require(qsa.get("selector_contract") == runtime.selector_contract()
            and qsa.get("replacement_enabled") is True and qsa.get("observer_only") is False,
            "Reviewed QSA-only selector contract differs")
        _require(qsa.get("state") == "applied" if require_applied else qsa.get("state") in ("pending_target_import", "applied"),
            "QSA selector was not applied")
        view["qsa_installation"] = {**qsa.get("native_original", {}),
            "state": "native_verified" if qsa.get("state") == "applied" else "pending_target_import"}
    _native_entry._validate_profile(view, family, require_applied)
    return profile


def _validate_selector_observation(value, mode):
    observation = value.get("qsa_live_observation")
    if mode == "native_qwen":
        _require(observation is None, "Native control unexpectedly has selector evidence")
        return
    expected = {"selector_source_sha256": runtime.QSA_HELPER_SHA256, "native_source_sha256": runtime.native.QSA_NATIVE_SOURCE_SHA256,
        "application_phase": runtime.APPLICATION_PHASE, "wrapper_code_exact": True, "original_code_exact": True,
        "selector_code_exact": True, "shape_checks_code_exact": True, "original_object_stable": True,
        "replacement_object_stable": True, "selector_object_stable": True, "reviewed_hook_only": True}
    _require(observation == expected, "Actual parent/worker QSA live code/object evidence differs")


def _validate_bi_absence(value, profile, scope):
    observation = value.get("batch_invariant_observation", {})
    _require(profile["startup_environment"].get("VLLM_BATCH_INVARIANT") is None
        and isinstance(observation, dict) and "observation_error" not in observation
        and observation.get("scope") == scope and observation.get("module") == "vllm.envs"
        and observation.get("expected_source_sha256") == observation.get("actual_source_sha256") == runtime.VLLM_ENVS_SHA256
        and observation.get("parser_native") is observation.get("getter_native") is True
        and observation.get("environment_value") is None
        and observation.get("parsed_value") is observation.get("effective_value") is False,
        "QSA-only requires actual parent/worker BI absent and disabled")


def validate_runtime_policy(snapshot, family):
    raw = snapshot.get("comparison_identity", {})
    mode = raw.get("profile", {}).get("mode")
    replacement = mode == "qwen_qsa_only"
    _require(all(raw.get(key) is replacement for key in ("qsa_replacement_module_imported", "qsa_replacement_import_hook")),
        "Actual QSA replacement module/hook observation differs")
    _validate_selector_observation(raw, mode)
    for worker in raw.get("worker_runtime", []):
        _validate_selector_observation(worker, mode)
    view = deepcopy(snapshot)
    # The frozen native validator's no-replacement booleans are replaced only
    # after independently validating the actual selector, original and hook.
    view["comparison_identity"].update(qsa_replacement_module_imported=False, qsa_replacement_import_hook=False)
    _native_entry.validate_runtime_policy(view, family)
    return _base.original.runtime_identity(snapshot)


_base.build_profile, _base._validate_profile, _base.validate_runtime_policy = build_profile, _validate_profile, validate_runtime_policy
_native_entry._validate_batch = _validate_bi_absence
PROFILE_MODES = runtime.PROFILE_MODES
load_plan, run = _base.load_plan, _base.run
_new_harness = _native_entry._new_harness
_original_compare = _base.compare


def prepare_plan(source_plan, profile_mode="qwen_qsa_only", baseline_environment=None):
    value = _base.prepare_plan(source_plan, profile_mode, baseline_environment)
    value["purpose"] = "49 original inputs; native model load, native MP and unmodified Torch policy; only reviewed QSA selection changes after loading, with explicitly sealed common operational baseline"
    return value


def compare(plan_path, *args, **kwargs):
    report = _original_compare(plan_path, *args, **kwargs)
    report["claim"] = "49 original inputs x two cold loads x two same-engine repeats, full input hashes and token/text comparison under the sealed native load and declared QSA-only selection; no general determinism claim."
    return report


_base.compare = compare


def main(argv=None):
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
