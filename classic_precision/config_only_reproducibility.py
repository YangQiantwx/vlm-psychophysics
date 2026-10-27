"""Finite 49-input, 2-cold x 2-repeat configuration-only diagnostic.

An isolated instance of the existing harness receives only explicit profile
dependencies. Native requests, persistence, calibration, leases, technical
gates, source bindings, input hashes and exact token comparison are reused.
Neither strict harness globals nor SDK/operator implementations are changed.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from . import reproducibility as original
from .storage import digest, file_sha256
from runtime_compat import classic_precision_config_only as runtime

SCHEMA = "classic_precision_config_only_reproducibility/v1"
BASELINE_SCHEMA = "classic_precision_native_nemo_observed_reproducibility/v1"
PROFILE_ID = runtime.PROFILE_ID
_require = original._require


def _mode(profile):
    return "native_nemo" if profile.get("id") == runtime.BASELINE_PROFILE_ID else "config_only"


def build_profile(mode="config_only", baseline_environment=None):
    """Return the complete versioned profile for a new, separately frozen plan."""
    _require(mode in ("config_only", "native_nemo"), "Unknown reproducibility mode")
    profile = {"id": PROFILE_ID, "environment": runtime.launch_environment(), "model_environment": {},
        "torch_policy": {"seed": 0, "deterministic_algorithms": True, "deterministic_warn_only": False,
            "cudnn_benchmark": False, "cudnn_deterministic": True, "tf32": False,
            "fill_uninitialized_memory": False}, "qsa_policy": "native_original_unhooked",
        "implementation": {"entry_sha256": file_sha256(__file__), "runtime_helper_sha256": file_sha256(runtime.__file__),
            "observer_helper_sha256": file_sha256(runtime.observers.__file__),
            "base_runner_sha256": runtime.BASE_RUNNER_SHA256, "harness_sha256": file_sha256(original.__file__)},
        "certification": "CONFIG_ONLY_DIAGNOSTIC_SEPARATE_FROM_STRICT"}
    if mode == "native_nemo":
        environment = {**runtime.BASELINE_ENVIRONMENT, **(baseline_environment or {})}
        _require(not (set(environment) & set(runtime.BASELINE_CLEARED_ENVIRONMENT)), "Baseline must clear strict-added environment")
        _require(all(environment.get(key) == value for key, value in runtime.BASELINE_ENVIRONMENT.items()),
                 "Baseline old-entry compatibility/offline environment differs")
        profile.update(id=runtime.BASELINE_PROFILE_ID, environment=environment,
            cleared_environment=list(runtime.BASELINE_CLEARED_ENVIRONMENT), torch_policy="observe_native_flags_without_setters",
            certification="NATIVE_NEMO_OBSERVED_RETEST_NOT_HISTORICAL_FLAG_RESTORATION")
    return profile


def _validate_plan_profile(plan, family):
    _require(family in runtime.FAMILIES, "Configuration-only excludes Gemma and unsupported models")
    mode = _mode(plan["profile"])
    _require(mode != "native_nemo" or family == "nemo", "Native baseline is restricted to Nemo")
    expected = build_profile(mode, plan["profile"].get("environment") if mode == "native_nemo" else None)
    _require(plan["profile"] == expected, "Plan diagnostic profile or implementation differs")
    return expected["environment"]


def _validate_profile(profile, family, require_applied):
    mode = "native_nemo" if isinstance(profile, dict) and profile.get("profile_id") == runtime.BASELINE_PROFILE_ID else "config_only"
    _require(family in runtime.FAMILIES and isinstance(profile, dict)
        and profile.get("profile_id") == (PROFILE_ID if mode == "config_only" else runtime.BASELINE_PROFILE_ID)
        and profile.get("mode") == mode and profile.get("family") == family,
        "Wrong configuration-only runtime profile ID/family")
    _require(profile.get("base_runner_sha256") == runtime.BASE_RUNNER_SHA256
        == file_sha256(Path(original.__file__).with_name("inference.py")), "Runtime base source identity mismatch")
    _require(profile.get("helper_sha256") == file_sha256(runtime.__file__)
        and profile.get("observer_helper_sha256") == file_sha256(runtime.observers.__file__), "Runtime helper source identity mismatch")
    expected_env = build_profile(mode, profile.get("environment") if mode == "native_nemo" else None)["environment"]
    _require(profile.get("environment") == expected_env, "Diagnostic environment differs")
    _require(profile.get("sdk_files_modified") is False, "Configuration-only must preserve SDK files")
    _require(profile.get("certification") == ("CONFIG_ONLY_UNVALIDATED_UNTIL_REPEAT_SUITE_PASSES" if mode == "config_only"
        else "NATIVE_NEMO_OBSERVED_RETEST_NOT_HISTORICAL_FLAG_RESTORATION"), "Wrong diagnostic certification")
    _require(mode != "native_nemo" or family == "nemo", "Native baseline is restricted to Nemo")
    if family == "qwen":
        qsa = profile.get("qsa_installation", {})
        _require(qsa.get("target_module") == qsa.get("function_module") == runtime.QSA_TARGET_MODULE
            and qsa.get("function_qualname") == "_topk"
            and qsa.get("expected_native_source_sha256") == qsa.get("native_source_sha256") == runtime.QSA_NATIVE_SOURCE_SHA256
            and qsa.get("replacement_enabled") is False and qsa.get("observer_only") is True,
            "Native QSA identity or no-replacement evidence differs")
        sha = qsa.get("native_code_sha256")
        _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Native QSA code identity missing")
        _require(qsa.get("state") == "native_verified" if require_applied else
            qsa.get("state") in ("pending_target_import", "native_verified"), "Native QSA live identity was not verified")
    else:
        _require(profile.get("qsa_installation") is None, "Unexpected QSA installation evidence")
    return profile


def validate_runtime_policy(snapshot, family):
    _require(isinstance(snapshot, dict) and isinstance(snapshot.get("comparison_identity"), dict),
        "Explicit runtime comparison_identity is required")
    value = snapshot["comparison_identity"]
    profile = _validate_profile(value.get("profile"), family, require_applied=True)
    mode = profile["mode"]
    for key in ("qsa_replacement_module_imported", "qsa_replacement_import_hook"):
        _require(value.get(key) is False, "QSA replacement guard evidence differs")
    _validate_flags(value, mode)
    workers = value.get("worker_runtime")
    if family in ("nemo", "qwen"):
        _require(isinstance(workers, list) and len(workers) == 1, "Actual single-GPU worker runtime is missing")
        for worker in workers:
            _validate_flags(worker, mode)
            _require(bool(worker.get("worker_class")) and worker.get("gpu_name") == value.get("gpu_name"), "Actual worker identity differs")
            engine = worker.get("engine_config", {})
            _require(engine.get("model_config", {}).get("seed") == 0
                and engine["model_config"].get("enforce_eager") is True
                and engine["model_config"].get("max_model_len") == 65536, "Actual worker model/seed/eager config differs")
            _require(all(engine.get("parallel_config", {}).get(key) == 1 for key in
                ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")), "Actual worker parallel config differs")
            _require(engine.get("scheduler_config", {}).get("max_num_seqs") == 1
                and engine["scheduler_config"].get("max_num_batched_tokens") == (4096 if family == "nemo" else 49152)
                and engine["scheduler_config"].get("enable_chunked_prefill") is True, "Actual worker scheduling config differs")
            _require(engine.get("cache_config", {}).get("cache_dtype") == "bfloat16"
                and engine["cache_config"].get("enable_prefix_caching") is False, "Actual worker cache config differs")
            if family == "qwen":
                _require(worker.get("model_class", "").endswith(".Qwen4ExpForConditionalGeneration")
                    and worker.get("qsa_native_identity") == profile.get("qsa_installation"), "Actual worker native QSA identity differs")
            else:
                _require(bool(worker.get("model_class")) and worker.get("qsa_native_identity") is None,
                    "Unexpected Nemo worker model/QSA identity")
    else:
        _require(workers == [], "Unexpected vLLM worker runtime")
    _require(value.get("inference_sha256") == profile["base_runner_sha256"], "Runtime inference source mismatch")
    _require(value.get("effective_environment") == profile["environment"], "Actual runtime environment differs")
    if mode == "native_nemo":
        expected_cleared = {key: None for key in runtime.BASELINE_CLEARED_ENVIRONMENT}
        _require(profile.get("startup_cleared_environment") == value.get("startup_cleared_environment") == expected_cleared,
            "Baseline inherited strict-added environment")
        _require(isinstance(value.get("effective_cleared_environment"), dict)
            and set(value["effective_cleared_environment"]) == set(expected_cleared)
            and all(v is None or isinstance(v, str) for v in value["effective_cleared_environment"].values()),
            "Actual native post-load environment is missing")
    _require(isinstance(value.get("packages"), dict) and all(value["packages"].get(k)
        for k in ("torch", "transformers", "numpy", "av")), "Required runtime package identities missing")
    _require(bool(value.get("gpu_name")) and isinstance(value.get("compute_capability"), list)
        and len(value["compute_capability"]) == 2 and bool(value.get("cuda")) and bool(value.get("driver")),
        "Actual CUDA/GPU/driver runtime identity missing")
    return original.runtime_identity(snapshot)


def _validate_flags(value, mode):
    if mode == "native_nemo":
        for key in ("deterministic_algorithms", "deterministic_warn_only", "cudnn_benchmark", "cudnn_deterministic", "fill_uninitialized_memory"):
            _require(type(value.get(key)) is bool, "Actual native runtime flag missing: " + key)
        _require(type(value.get("seed")) is int and value["seed"] == 0, "Native fixed generation seed differs")
        precision = value.get("fp32_precision")
        _require(isinstance(precision, dict) and (set(precision) == {"matmul", "cudnn_conv", "cudnn_rnn"}
            and all(isinstance(v, str) and bool(v) for v in precision.values())
            or set(precision) == {"allow_tf32_matmul", "allow_tf32_cudnn"} and all(type(v) is bool for v in precision.values())),
            "Actual native precision flags missing")
        return
    for key, required in (("deterministic_algorithms", True), ("deterministic_warn_only", False),
            ("cudnn_benchmark", False), ("cudnn_deterministic", True), ("fill_uninitialized_memory", False)):
        _require(value.get(key) is required, "Configuration-only runtime policy violation: " + key)
    _require(type(value.get("seed")) is int and value["seed"] == 0, "Configuration-only seed differs")
    _require(value.get("fp32_precision") in ({"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"},
        {"allow_tf32_matmul": False, "allow_tf32_cudnn": False}), "TF32 precision policy missing or relaxed")


def _new_harness(mode="config_only", baseline_environment=None):
    """Load unchanged code in an isolated namespace; inject profile dependencies."""
    spec = importlib.util.spec_from_file_location("classic_precision._config_only_harness", original.__file__)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    native_import = importlib.import_module
    harness.SCHEMA = SCHEMA if mode == "config_only" else BASELINE_SCHEMA
    harness._validate_plan_profile = _validate_plan_profile
    harness._validate_profile = _validate_profile
    harness.validate_runtime_policy = validate_runtime_policy
    harness._prepare_runtime_modules = runtime.prepare_native
    profile_runtime = SimpleNamespace(activate=lambda family: runtime.activate(family, mode, baseline_environment),
        status=runtime.status, runtime_snapshot=runtime.runtime_snapshot)
    harness.importlib = SimpleNamespace(import_module=lambda name:
        profile_runtime if name == "classic_precision_determinism" else native_import(name))
    base_evidence = harness._evidence
    harness._diagnostic_evidence_stores = []
    harness._diagnostic_evidence = {}
    def evidence(directory, plan, plan_hash, family, cold, repeat, sources):
        value = base_evidence(directory, plan, plan_hash, family, cold, repeat, sources)
        if family in ("nemo", "qwen"):
            events = original._read_jsonl(directory / "events.jsonl")
            records = [event for event in events if event.get("event") == "REPRO_WORKER_RUNTIME"]
            _require(len(records) == 2 * len(plan["sample_ids"]), "Per-call actual worker runtime receipts missing")
            for sid in plan["sample_ids"]:
                receipts = [event for event in records if event.get("sample_id") == sid]
                _require(sorted(event.get("stage") for event in receipts) == ["after", "before"], "Worker runtime receipt stages differ")
                _require(all(event.get("workers") == value["runtime"]["worker_runtime"] for event in receipts), "Actual worker runtime changed during calls")
        harness._diagnostic_evidence_stores.append((family, cold, repeat))
        harness._diagnostic_evidence[(family, cold, repeat)] = deepcopy(value)
        return value
    harness._evidence = evidence
    harness._base_load_plan = harness.load_plan
    harness.load_plan = load_plan
    return harness


def _validate_design(plan):
    _require(plan.get("cold_passes") == 2 and plan.get("repeats") == 2, "Configuration-only requires exactly 2 cold x 2 repeats")
    mode = _mode(plan["profile"])
    _require(plan.get("models") == (list(runtime.FAMILIES) if mode == "config_only" else ["nemo"]), "Diagnostic plan models differ")
    _require(len(plan["sample_ids"]) == 49 and plan.get("model_calls_per_model") == 196
        and plan.get("total_model_calls") == (588 if mode == "config_only" else 196), "Finite diagnostic call budget differs")
    _require(plan.get("manifest_sha256") == file_sha256(plan["manifest"]), "Frozen manifest SHA is required")
    sources = original._source_bindings(plan)
    t2 = [sid for sid in plan["sample_ids"] if sources[sid][0].get("task") == "T2"]
    t3 = [sid for sid in plan["sample_ids"] if sources[sid][0].get("task") == "T3"]
    _require(len(t2) == 48 and len(t3) == 1, "Plan requires 48 scientific T2 plus one T3 qualification")
    for sid in t2:
        p = sources[sid][0].get("source_parameters", {})
        _require(p.get("delta_Y") == 0.11 and p.get("background_Y") == 0.05 and p.get("fps") == 60
            and p.get("offset_frame", 0) - p.get("onset_frame", 0) == 30,
            "Frozen T2 delta/background/duration differs: " + sid)
    _require(plan.get("scientific_ids") == t2 and plan.get("auxiliary_T3_qualification_id") == t3[0],
        "Scientific/qualification denominators differ")


def load_plan(path):
    # Reuse the original parser before adding this diagnostic's finite design.
    raw = json.loads(Path(path).read_text())
    _require(raw.get("schema") in (SCHEMA, BASELINE_SCHEMA), "Unknown diagnostic plan schema")
    mode = "native_nemo" if raw["schema"] == BASELINE_SCHEMA else "config_only"
    _require(_mode(raw.get("profile", {})) == mode, "Diagnostic schema/profile mismatch")
    harness = _new_harness(mode)
    plan, plan_hash = harness._base_load_plan(path)
    _validate_design(plan)
    return plan, plan_hash


def prepare_plan(source_plan, mode="config_only", baseline_environment=None):
    """Build, without writing, the new plan from an existing reference plan dict."""
    source = json.loads(json.dumps(source_plan))
    keys = ("source_run", "manifest", "manifest_sha256", "ready", "asset_root", "calibration_root", "gpu_lock_root")
    plan = {key: source[key] for key in keys if key in source}
    plan.update(schema=SCHEMA if mode == "config_only" else BASELINE_SCHEMA,
        profile=build_profile(mode, baseline_environment), models=list(runtime.FAMILIES) if mode == "config_only" else ["nemo"], cold_passes=2, repeats=2,
        sample_ids=source.get("operational_ids", source.get("sample_ids")), scientific_ids=source["scientific_ids"],
        auxiliary_T3_qualification_id=source["auxiliary_T3_qualification_id"], model_calls_per_model=196,
        total_model_calls=588 if mode == "config_only" else 196, automatic_retry=False, production_launch=False,
        purpose="Finite native-operator diagnostic; separate from strict and production; native baseline observes current flags without reconstructing unrecorded history")
    return plan


def run(plan_path, *args, **kwargs):
    plan, _ = load_plan(plan_path)
    mode = _mode(plan["profile"])
    import os
    launch_env = plan.get("launch_environment", {})
    cleared = plan.get("launch_cleared_environment", [])
    _require(isinstance(launch_env, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in launch_env.items()),
        "Invalid frozen launch environment")
    _require(isinstance(cleared, list) and all(isinstance(k, str) for k in cleared), "Invalid frozen cleared environment")
    _require(all(os.environ.get(key) == value for key, value in launch_env.items()), "Actual launch environment differs from frozen plan")
    _require(all(key not in os.environ for key in cleared), "Actual launch inherited forbidden environment")
    return _new_harness(mode, plan["profile"]["environment"]).run(plan_path, *args, **kwargs)


def compare(plan_path, *args, **kwargs):
    plan, _ = load_plan(plan_path)
    mode = _mode(plan["profile"])
    harness = _new_harness(mode)
    report = harness.compare(plan_path, *args, **kwargs)
    for family, result in report["models"].items():
        count = sum(item[0] == family for item in harness._diagnostic_evidence_stores)
        result["complete_evidence_stores"] = count
        control_error = _control_error(harness._diagnostic_evidence, family, plan)
        result["control_error"] = control_error
        result["evidence_status"] = "COMPLETE" if control_error is None else "INVALID_OR_INCOMPLETE"
        result["comparison_status"] = "PASS" if result["status"] == "PASS" else ("DIFFERENT" if control_error is None else "INVALID_EVIDENCE")
    report["evidence_status"] = "COMPLETE" if all(r["evidence_status"] == "COMPLETE" for r in report["models"].values()) else "INVALID_OR_INCOMPLETE"
    report["claim"] = "Native-operator diagnostic: exact saved input hashes, full generated tokens/text and technical receipts on 49 inputs and four stores per model; no strict-profile, restored historical GPU flags or production claim."
    return report


def _control_error(evidence, family, plan):
    """Output differences are interpretable only under equal verified controls."""
    try:
        expected = [(family, cold, repeat) for cold in range(2) for repeat in range(2)]
        _require(all(key in evidence for key in expected), "Four complete repeat stores are required")
        baseline, owners = evidence[expected[0]], {}
        for key in expected:
            value = evidence[key]
            cold = key[1]
            owner = value["owner"]
            owner_key = tuple(owner.get(field) for field in ("host", "boot_id", "pid", "process_start_ticks"))
            if cold in owners:
                _require(owner_key == owners[cold], "Same-engine repeats came from different processes")
            else:
                _require(owner_key not in owners.values(), "Cold passes reused a process")
                owners[cold] = owner_key
            for field in ("native_identity", "profile_status", "runtime"):
                changed = original._differences(baseline[field], value[field])
                _require(not changed, field + " mismatch: " + ",".join(changed[:32]))
            for sid in plan["sample_ids"]:
                a, b = baseline["fingerprints"][sid], value["fingerprints"][sid]
                controls_a = {field: item for field, item in a.items()
                    if field not in {"raw_answer", "output_token_ids", "stop_reason"}}
                controls_b = {field: item for field, item in b.items()
                    if field not in {"raw_answer", "output_token_ids", "stop_reason"}}
                changed = original._differences(controls_a, controls_b)
                _require(not changed, "Non-output control mismatch " + sid + ": " + ",".join(changed[:32]))
        return None
    except (ValueError, KeyError, TypeError) as error:
        return str(error)


def main(argv=None):
    # Keep the existing CLI and its argument names; its globals are isolated.
    harness = _new_harness()
    harness.run, harness.compare = run, compare
    return harness.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
