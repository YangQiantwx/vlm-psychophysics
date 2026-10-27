"""Finite native-Cosmos and singleton Qwen inference-time configuration tests.

The unchanged native harness runs 49 original inputs in four independent stores
(four Cosmos cold loads, or two Qwen cold loads with two repeats). Loading remains native;
post-load baseline observations establish which flags the intervention changes.
"""
from __future__ import annotations

from copy import deepcopy
import importlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

from . import config_only_reproducibility as config
from . import reproducibility as original
from .storage import digest, file_sha256
from runtime_compat import classic_precision_minimal_config as runtime

SCHEMA = "classic_precision_minimal_config_reproducibility/v1"
PROFILE_MODES = runtime.PROFILE_MODES
_require = original._require
_FLAG_KEYS = {"seed", "deterministic_algorithms", "deterministic_warn_only", "fill_uninitialized_memory",
    "cudnn_benchmark", "cudnn_deterministic", "fp32_precision"}


def build_profile(profile_mode="native_cosmos", baseline_environment=None):
    before, interventions, environment, cleared = runtime.environment_contract(profile_mode, baseline_environment)
    return {"id": runtime.PROFILE_IDS[profile_mode], "mode": profile_mode,
        "baseline_origin": "sealed_native_launch_environment", "baseline_environment": before,
        "environment_interventions": interventions, "environment": environment, "cleared_environment": cleared,
        "model_environment": {}, "torch_policy": deepcopy(runtime.TORCH_POLICIES[profile_mode]),
        "application_phase": runtime.APPLICATION_PHASE, "qsa_policy": "native_original_unhooked",
        "native_environment_mutation_allowlist": {} if profile_mode == "native_cosmos" else
            {"VLLM_WORKER_MULTIPROC_METHOD": {"from": None, "to": "spawn"}},
        "implementation": {"entry_sha256": file_sha256(__file__), "runtime_helper_sha256": file_sha256(runtime.__file__),
            "config_validation_sha256": file_sha256(config.__file__), "native_guard_helper_sha256": file_sha256(runtime.native.__file__),
            "observer_helper_sha256": file_sha256(runtime.native.observers.__file__),
            "base_runner_sha256": runtime.BASE_RUNNER_SHA256, "harness_sha256": file_sha256(original.__file__)},
        "certification": "MINIMAL_INFERENCE_TIME_DIAGNOSTIC_NOT_COLD_LOAD_POLICY"}


def _validate_plan_profile(plan, family):
    profile = plan["profile"]
    mode = profile.get("mode")
    _require(family == runtime.family_for_mode(mode), "Minimal profile mode/family mismatch")
    _require(profile == build_profile(mode, profile.get("baseline_environment")), "Plan minimal profile or source identity differs")
    return profile["environment"]


def _validate_profile(profile, family, require_applied):
    _require(isinstance(profile, dict), "Minimal runtime profile is missing")
    mode = profile.get("mode")
    _require(family == runtime.family_for_mode(mode) and profile.get("family") == family
        and profile.get("profile_id") == runtime.PROFILE_IDS[mode], "Wrong minimal profile ID/family")
    expected = build_profile(mode, profile.get("baseline_environment"))
    for key in ("baseline_environment", "environment_interventions", "environment", "cleared_environment", "torch_policy", "application_phase", "certification"):
        _require(profile.get(key) == expected[key], "Minimal runtime profile differs: " + key)
    _require(profile.get("startup_environment") == {**profile["baseline_environment"], **profile["environment_interventions"]},
        "Actual sealed startup environment differs")
    for key, value in (("base_runner_sha256", runtime.BASE_RUNNER_SHA256), ("helper_sha256", file_sha256(runtime.__file__)),
            ("native_guard_helper_sha256", file_sha256(runtime.native.__file__)),
            ("observer_helper_sha256", file_sha256(runtime.native.observers.__file__))):
        _require(profile.get(key) == value, "Minimal runtime source identity differs: " + key)
    _require(profile["base_runner_sha256"] == file_sha256(Path(original.__file__).with_name("inference.py")), "Native inference source changed")
    _require(profile.get("sdk_files_modified") is False, "Minimal profiles must preserve SDK files")
    if family == "qwen":
        qsa = profile.get("qsa_installation", {})
        _require(qsa.get("target_module") == qsa.get("function_module") == runtime.native.QSA_TARGET_MODULE
            and qsa.get("function_qualname") == "_topk"
            and qsa.get("expected_native_source_sha256") == qsa.get("native_source_sha256") == runtime.native.QSA_NATIVE_SOURCE_SHA256
            and qsa.get("replacement_enabled") is False and qsa.get("observer_only") is True,
            "Native QSA source or no-replacement identity differs")
        sha = qsa.get("native_code_sha256")
        _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha), "Native QSA code identity missing")
        _require(qsa.get("state") == "native_verified" if require_applied else
            qsa.get("state") in ("pending_target_import", "native_verified"), "Native QSA live object was not verified")
    else:
        _require(profile.get("qsa_installation") is None, "Unexpected Cosmos QSA evidence")
    return profile


def _validate_application(value, mode):
    config._validate_flags(value, "native_nemo")  # typed native observations, without forced flag defaults
    receipt = value.get("application_receipt")
    _require(isinstance(receipt, dict) and receipt.get("phase") == runtime.APPLICATION_PHASE
        and receipt.get("model_loaded_before_application") is True and receipt.get("mode") == mode
        and receipt.get("declared_torch_policy") == runtime.TORCH_POLICIES[mode], "Post-load minimal application receipt missing")
    before, after = receipt.get("before"), receipt.get("after")
    _require(isinstance(before, dict) and set(before) == _FLAG_KEYS and isinstance(after, dict)
        and set(after) == _FLAG_KEYS, "Native baseline/application flag coverage differs")
    config._validate_flags(before, "native_nemo")
    expected = deepcopy(before)
    if mode == "qwen_tf32_off_only":
        precision = before["fp32_precision"]
        expected["fp32_precision"] = ({key: "ieee" for key in precision} if "matmul" in precision else {key: False for key in precision})
    elif mode == "qwen_torch_deterministic":
        expected.update(deterministic_algorithms=True, deterministic_warn_only=False, fill_uninitialized_memory=False)
    _require(after == expected, "Undeclared flag changed or requested intervention was not applied")
    _require({key: value.get(key) for key in _FLAG_KEYS} == after, "Actual flags changed after minimal application")


def validate_runtime_policy(snapshot, family):
    _require(isinstance(snapshot, dict) and isinstance(snapshot.get("comparison_identity"), dict), "Explicit runtime comparison_identity is required")
    value = snapshot["comparison_identity"]
    profile = _validate_profile(value.get("profile"), family, require_applied=True)
    mode = profile["mode"]
    for key in ("qsa_replacement_module_imported", "qsa_replacement_import_hook"):
        _require(value.get(key) is False, "QSA replacement guard evidence differs")
    _validate_application(value, mode)
    _validate_environment(value, profile)
    _require(value.get("effective_environment") == profile["environment"], "Actual required launch environment differs")
    _require(value.get("inference_sha256") == runtime.BASE_RUNNER_SHA256, "Runtime inference source mismatch")
    workers = value.get("worker_runtime")
    if family == "qwen":
        _require(isinstance(workers, list) and len(workers) == 1, "Actual single-GPU Qwen worker runtime missing")
        worker = workers[0]
        _validate_application(worker, mode)
        _validate_environment(worker, profile)
        _require(bool(worker.get("worker_class")) and worker.get("model_class", "").endswith(".Qwen4ExpForConditionalGeneration")
            and worker.get("gpu_name") == value.get("gpu_name")
            and worker.get("qsa_native_identity") == profile.get("qsa_installation"), "Actual worker native/QSA identity differs")
        engine = worker.get("engine_config", {})
        _require(engine.get("model_config", {}).get("seed") == 0 and engine["model_config"].get("enforce_eager") is True
            and engine["model_config"].get("max_model_len") == 65536, "Actual worker native seed/eager config differs")
        _require(all(engine.get("parallel_config", {}).get(key) == 1 for key in
            ("tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size")), "Actual worker native parallel config differs")
        _require(engine.get("scheduler_config", {}).get("max_num_seqs") == 1
            and engine["scheduler_config"].get("max_num_batched_tokens") == 49152
            and engine["scheduler_config"].get("enable_chunked_prefill") is True, "Actual worker native scheduler config differs")
        _require(engine.get("cache_config", {}).get("cache_dtype") == "bfloat16"
            and engine["cache_config"].get("enable_prefix_caching") is False, "Actual worker native cache config differs")
    else:
        _require(workers == [], "Unexpected Cosmos vLLM worker")
    _require(isinstance(value.get("packages"), dict) and all(value["packages"].get(k)
        for k in ("torch", "transformers", "numpy", "av")), "Required runtime package identities missing")
    _require(bool(value.get("gpu_name")) and isinstance(value.get("compute_capability"), list)
        and len(value["compute_capability"]) == 2 and bool(value.get("cuda")) and bool(value.get("driver")), "Actual CUDA/GPU/driver identity missing")
    return original.runtime_identity(snapshot)


def _validate_environment(value, profile):
    startup = profile["startup_environment"]
    baseline = value.get("native_postload_environment")
    _require(isinstance(baseline, dict) and set(baseline) == set(startup)
        and value.get("observed_environment") == baseline, "Environment drifted after native post-load baseline")
    # vLLM's native CUDA-initialized path selects spawn before creating workers.
    # No other inherited strict flag or arbitrary native drift is accepted.
    allowed = build_profile(profile["mode"], profile["baseline_environment"])["native_environment_mutation_allowlist"]
    for key in startup:
        if baseline[key] != startup[key]:
            _require(key in allowed and startup[key] == allowed[key]["from"] and baseline[key] == allowed[key]["to"],
                "Undeclared native post-load environment mutation: " + key)


def _new_harness(profile_mode="native_cosmos", baseline_environment=None):
    """Inject dependencies into an isolated copy of the unchanged native module."""
    spec = importlib.util.spec_from_file_location("classic_precision._minimal_config_harness", original.__file__)
    harness = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(harness)
    native_import = importlib.import_module
    harness.SCHEMA = SCHEMA
    harness._validate_plan_profile = _validate_plan_profile
    harness._validate_profile = _validate_profile
    harness.validate_runtime_policy = validate_runtime_policy
    harness._prepare_runtime_modules = runtime.prepare_native
    profile_runtime = SimpleNamespace(activate=lambda family: runtime.activate(family, profile_mode, baseline_environment),
        status=runtime.status, runtime_snapshot=runtime.runtime_snapshot)
    harness.importlib = SimpleNamespace(import_module=lambda name:
        profile_runtime if name == "classic_precision_determinism" else native_import(name))
    base_evidence = harness._evidence
    harness._diagnostic_evidence_stores, harness._diagnostic_evidence = [], {}

    def evidence(directory, plan, plan_hash, family, cold, repeat, sources):
        value = base_evidence(directory, plan, plan_hash, family, cold, repeat, sources)
        records = [event for event in original._read_jsonl(directory / "events.jsonl") if event.get("event") == "REPRO_CALL_RUNTIME"]
        _require(len(records) == 2 * len(plan["sample_ids"]), "Per-call parent/worker runtime receipts missing")
        for sid in plan["sample_ids"]:
            receipts = [event for event in records if event.get("sample_id") == sid]
            _require(sorted(event.get("stage") for event in receipts) == ["after", "before"], "Per-call runtime stages differ")
            for event in receipts:
                observed = validate_runtime_policy(event.get("snapshot"), family)
                _require(observed == value["runtime"], "Parent/worker runtime changed during diagnostic calls")
        harness._diagnostic_evidence_stores.append((family, cold, repeat))
        harness._diagnostic_evidence[(family, cold, repeat)] = deepcopy(value)
        return value

    harness._evidence = evidence
    harness._base_load_plan = harness.load_plan
    harness.load_plan = load_plan
    return harness


def _validate_design(plan):
    layout = (4, 1) if plan["profile"]["mode"] == "native_cosmos" else (2, 2)
    _require((plan.get("cold_passes"), plan.get("repeats")) == layout, "Minimal diagnostic cold/repeat layout differs")
    _require(plan.get("models") == [runtime.family_for_mode(plan["profile"]["mode"])], "Minimal diagnostic model list differs")
    _require(len(plan["sample_ids"]) == 49 and plan.get("model_calls_per_model") == plan.get("total_model_calls") == 196, "Finite minimal call budget differs")
    _require(plan.get("automatic_retry") is False and plan.get("production_launch") is False, "Minimal plan must disable retries and production launch")
    _require(plan.get("manifest_sha256") == file_sha256(plan["manifest"]), "Frozen source manifest SHA is required")
    sources = original._source_bindings(plan)
    t2 = [sid for sid in plan["sample_ids"] if sources[sid][0].get("task") == "T2"]
    t3 = [sid for sid in plan["sample_ids"] if sources[sid][0].get("task") == "T3"]
    _require(len(t2) == 48 and len(t3) == 1, "Plan requires original 48 T2 plus one T3 qualification")
    for sid in t2:
        p = sources[sid][0].get("source_parameters", {})
        _require(p.get("delta_Y") == .11 and p.get("background_Y") == .05 and p.get("fps") == 60
            and p.get("offset_frame", 0) - p.get("onset_frame", 0) == 30, "Frozen T2 condition differs: " + sid)
    _require(plan.get("scientific_ids") == t2 and plan.get("auxiliary_T3_qualification_id") == t3[0], "Scientific/qualification denominators differ")


def load_plan(path):
    # The original loader requires >=2 repeats; Cosmos deliberately uses four
    # independent cold processes. Retain its common guards explicitly here.
    plan = json.loads(Path(path).read_text())
    _require(isinstance(plan, dict) and plan.get("schema") == SCHEMA, "Unknown minimal diagnostic plan schema")
    _require(isinstance(plan.get("profile"), dict), "Minimal profile is missing")
    for name in ("source_run", "manifest", "ready", "asset_root"):
        _require(isinstance(plan.get(name), str) and Path(plan[name]).is_absolute(), name + " must be absolute")
    ids = plan.get("sample_ids")
    _require(isinstance(ids, list) and ids and all(isinstance(s, str) and s for s in ids)
        and len(ids) == len(set(ids)), "sample_ids must be a unique fixed list")
    models = plan.get("models")
    _require(isinstance(models, list) and len(models) == 1, "Minimal plan requires one model")
    _validate_plan_profile(plan, models[0])
    _require(Path(plan["manifest"]).is_file() and Path(plan["ready"]).is_file(), "Original manifest/ready is missing")
    _require(Path(plan["source_run"]).is_dir() and Path(plan["asset_root"]).is_dir(), "Original source/assets is missing")
    _validate_design(plan)
    return plan, digest(plan)


def prepare_plan(source_plan, profile_mode="native_cosmos", baseline_environment=None):
    """Return an unwritten, separately frozen 196-call plan for exactly one arm."""
    source = deepcopy(source_plan)
    keys = ("source_run", "manifest", "manifest_sha256", "ready", "asset_root", "calibration_root", "gpu_lock_root")
    plan = {key: source[key] for key in keys if key in source}
    plan.update(schema=SCHEMA, profile=build_profile(profile_mode, baseline_environment),
        models=[runtime.family_for_mode(profile_mode)], cold_passes=4 if profile_mode == "native_cosmos" else 2,
        repeats=1 if profile_mode == "native_cosmos" else 2,
        sample_ids=source.get("operational_ids", source.get("sample_ids")), scientific_ids=source["scientific_ids"],
        auxiliary_T3_qualification_id=source["auxiliary_T3_qualification_id"], model_calls_per_model=196,
        total_model_calls=196, automatic_retry=False, production_launch=False,
        purpose="Finite native-load, minimal inference-time diagnostic; no restored historical flags or strict-profile claim")
    return plan


def run(plan_path, *args, **kwargs):
    plan, _ = load_plan(plan_path)
    profile = plan["profile"]
    required = {**profile["environment"], **plan.get("launch_environment", {})}
    cleared = set(profile["cleared_environment"]) | set(plan.get("launch_cleared_environment", []))
    _require(all(isinstance(key, str) and isinstance(value, str) for key, value in required.items()), "Invalid frozen launch environment")
    _require(all(os.environ.get(key) == value for key, value in required.items()) and all(key not in os.environ for key in cleared), "Actual launch differs from sealed minimal environment")
    return _new_harness(profile["mode"], profile["baseline_environment"]).run(plan_path, *args, **kwargs)


def compare(plan_path, *args, **kwargs):
    plan, _ = load_plan(plan_path)
    harness = _new_harness(plan["profile"]["mode"])
    report = harness.compare(plan_path, *args, **kwargs)
    for family, result in report["models"].items():
        result["complete_evidence_stores"] = sum(item[0] == family for item in harness._diagnostic_evidence_stores)
        error = _control_error(harness._diagnostic_evidence, family, plan)
        result.update(control_error=error, evidence_status="COMPLETE" if error is None else "INVALID_OR_INCOMPLETE",
            comparison_status="INVALID_EVIDENCE" if error is not None else ("PASS" if result["status"] == "PASS" else "DIFFERENT"))
        if error is not None:
            result["status"] = report["status"] = "FAIL"
            if family not in report["errors"]:
                report["errors"].append(family)
    report["evidence_status"] = "COMPLETE" if all(r["evidence_status"] == "COMPLETE" for r in report["models"].values()) else "INVALID_OR_INCOMPLETE"
    report["profile_mode"] = plan["profile"]["mode"]
    report["claim"] = "49 native inputs, four independent stores and full token/text comparison under sealed native load and declared post-load inference-time interventions; no cold-load determinism or strict-profile claim."
    return report


def _control_error(evidence, family, plan):
    """Keep the validated control comparison, with this arm's explicit layout."""
    try:
        keys = [(family, cold, repeat) for cold in range(plan["cold_passes"]) for repeat in range(plan["repeats"])]
        _require(len(keys) == 4 and all(key in evidence for key in keys), "Four complete repeat stores are required")
        baseline, owners = evidence[keys[0]], {}
        for key in keys:
            value, cold = evidence[key], key[1]
            owner = value["owner"]
            owner_key = tuple(owner.get(field) for field in ("host", "boot_id", "pid", "process_start_ticks"))
            _require(all(owner_key), "Complete process owner identity is required")
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
                # Finish reason is validated terminal by the native harness.
                controls_a = {field: item for field, item in a.items() if field not in {"raw_answer", "output_token_ids", "stop_reason"}}
                controls_b = {field: item for field, item in b.items() if field not in {"raw_answer", "output_token_ids", "stop_reason"}}
                changed = original._differences(controls_a, controls_b)
                _require(not changed, "Non-output control mismatch " + sid + ": " + ",".join(changed[:32]))
        return None
    except (ValueError, KeyError, TypeError) as error:
        return str(error)


def main(argv=None):
    harness = _new_harness()
    harness.run, harness.compare = run, compare
    return harness.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
