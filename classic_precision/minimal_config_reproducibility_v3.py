"""UniProc EngineCore launch evidence overlay; numerical policies stay frozen."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path

from . import minimal_config_reproducibility_v2 as frozen_thread_validator
from .storage import file_sha256
from runtime_compat import classic_precision_minimal_config_v3 as runtime

SCHEMA = "classic_precision_minimal_config_reproducibility/v3"
FROZEN_ENTRY_SHA256 = "162551b7cb625c22cb388f931c8bcb0c89aff2e89e4d561f053526ef438fd75a"
FROZEN_V2_ENTRY_SHA256 = "7bdfa3e4ab22446d01df19b9912bf85f57e7dba0c2b190f266a69544d4e89196"
_path = Path(__file__).with_name("minimal_config_reproducibility.py")
if file_sha256(_path) != FROZEN_ENTRY_SHA256 or file_sha256(frozen_thread_validator.__file__) != FROZEN_V2_ENTRY_SHA256:
    raise RuntimeError("Frozen minimal entry/thread validator source changed")
_spec = importlib.util.spec_from_file_location("classic_precision._minimal_v3_frozen_entry", _path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.SCHEMA, _base.runtime = SCHEMA, runtime
_require = _base._require
_original_build_profile, _original_validate_profile = _base.build_profile, _base._validate_profile
_original_new_harness = _base._new_harness
_original_validate_runtime_policy = _base.validate_runtime_policy


def build_profile(profile_mode="qwen_tf32_off_only", baseline_environment=None):
    _require(profile_mode in runtime.PROFILE_MODES, "Minimal v3 supports only native EngineCore Qwen TF32/deterministic arms")
    value = _original_build_profile(profile_mode, baseline_environment)
    _require(all(value["baseline_environment"][key] is None for key in
        ("OMP_NUM_THREADS", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "VLLM_ENABLE_V1_MULTIPROCESSING")),
        "Native EngineCore launch requires original unset OMP/marker/multiprocessing flags")
    value.update(evidence_contract=runtime.EVIDENCE_CONTRACT, native_thread_source_sha256=deepcopy(runtime.THREAD_SOURCE_SHA256),
        engine_startup_source_sha256=runtime.ENGINE_STARTUP_SHA256, executor_policy="native_uni_enginecore")
    value["implementation"].update(overlay_entry_sha256=file_sha256(__file__),
        frozen_observer_helper_sha256=runtime.FROZEN_V2_HELPER_SHA256, frozen_thread_validator_sha256=FROZEN_V2_ENTRY_SHA256)
    value["native_environment_mutation_allowlist"].update(
        OMP_NUM_THREADS={"from": None, "to": "min(pre_engine_parent_torch_threads,startup_omp_num_threads(local_engine_count=1))"},
        VLLM_OMP_NUM_THREADS_SET_BY_VLLM={"from": None, "to": "1"})
    return value


def _validate_profile(profile, family, require_applied):
    value = _original_validate_profile(profile, family, require_applied)
    _require(value.get("evidence_contract") == runtime.EVIDENCE_CONTRACT
        and value.get("frozen_policy_helper_sha256") == runtime.FROZEN_HELPER_SHA256
        and value.get("frozen_observer_helper_sha256") == runtime.FROZEN_V2_HELPER_SHA256
        and value.get("native_thread_source_sha256") == runtime.THREAD_SOURCE_SHA256
        and value.get("engine_startup_source_sha256") == runtime.ENGINE_STARTUP_SHA256,
        "Native UniProc launch evidence/source identity differs")
    return value


def _validate_source(value):
    expected = {"module": runtime.ENGINE_STARTUP_MODULE, "expected_sha256": runtime.ENGINE_STARTUP_SHA256,
        "actual_sha256": runtime.ENGINE_STARTUP_SHA256, "function": "_configure_uniproc_startup_threads"}
    _require(value == expected, "Native UniProc startup source differs")


def _validate_launch_parent(value):
    receipt = value.get("engine_launch_parent")
    _require(isinstance(receipt, dict) and receipt.get("phase") == "after_native_processor_before_engine_load"
        and receipt.get("origin") == "launching_parent_process"
        and receipt.get("family") == "qwen" and receipt.get("inference_sha256") == runtime.BASE_RUNNER_SHA256
        and type(receipt.get("local_engine_count")) is int and receipt["local_engine_count"] == 1,
        "Original pre-engine parent observation missing")
    _require(type(receipt.get("torch_num_threads")) is int and receipt["torch_num_threads"] > 0
        and receipt.get("omp_num_threads") is None and receipt.get("vllm_omp_marker") is None,
        "Original pre-engine Torch/env observation differs")
    affinity = receipt.get("cpu_affinity")
    _require(isinstance(affinity, list) and affinity and affinity == sorted(set(affinity))
        and all(type(cpu) is int and cpu >= 0 for cpu in affinity), "Pre-engine CPU affinity observation missing")
    _validate_source(receipt.get("engine_startup_source"))
    return receipt


def _validate_environment(value, profile):
    startup, baseline = profile["startup_environment"], value.get("native_postload_environment")
    _require(isinstance(baseline, dict) and set(baseline) == set(startup)
        and value.get("observed_environment") == baseline, "Environment drifted after native post-load baseline")
    scope = "worker" if "worker_class" in value else "parent"
    observed = frozen_thread_validator._validate_thread_observation(value, scope)
    _validate_source(value.get("engine_startup_source"))
    launch = _validate_launch_parent(value)
    _require(launch["cpu_affinity"] == observed["cpu_affinity"], "CPU affinity changed across native engine launch")
    changed = {key for key in startup if baseline[key] != startup[key]}
    _require(changed <= {"VLLM_WORKER_MULTIPROC_METHOD", "OMP_NUM_THREADS", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM"},
        "Undeclared native post-load environment mutation")
    if "VLLM_WORKER_MULTIPROC_METHOD" in changed:
        _require(startup["VLLM_WORKER_MULTIPROC_METHOD"] is None and baseline["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn",
            "Native multiprocessing method transition differs")
    if scope == "parent":
        # The original context manager restores the outer parent's env/threads.
        _require(baseline["OMP_NUM_THREADS"] is None and baseline["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] is None
            and observed["torch_num_threads"] == launch["torch_num_threads"], "Outer parent UniProc context was not restored")
    else:
        _require(value.get("engine_config", {}).get("parallel_config", {}).get("distributed_executor_backend") == "uni",
            "Actual worker did not use the verified native UniProc path")
        expected_omp = min(launch["torch_num_threads"], observed["native_startup_omp_num_threads"])
        _require(baseline["OMP_NUM_THREADS"] == str(expected_omp)
            and baseline["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] == "1" and observed["torch_num_threads"] == 1,
            "Native UniProc inherited OMP/marker/runtime thread formula differs")


def validate_runtime_policy(snapshot, family):
    value = _original_validate_runtime_policy(snapshot, family)
    _require(all(worker.get("engine_launch_parent") == value.get("engine_launch_parent") for worker in value["worker_runtime"]),
        "Actual worker launching-parent receipt differs from outer parent receipt")
    return value


def _new_harness(profile_mode="qwen_tf32_off_only", baseline_environment=None):
    harness = _original_new_harness(profile_mode, baseline_environment)
    base_evidence = harness._evidence

    def evidence(directory, plan, plan_hash, family, cold, repeat, sources):
        events = _base.original._read_jsonl(directory / "events.jsonl")
        raw = [i for i, event in enumerate(events) if event.get("event") == "REPRO_RUNTIME_OBSERVED"]
        bound = [i for i, event in enumerate(events) if event.get("event") == "REPRO_RUNTIME"]
        _require(len(raw) == len(bound) == 1 and raw[0] < bound[0], "Durable raw runtime observation must precede policy validation")
        snapshot = events[raw[0]].get("snapshot")
        _require(events[raw[0]].get("evidence_contract") == runtime.EVIDENCE_CONTRACT
            and snapshot == events[bound[0]].get("snapshot"), "Raw/bound v3 runtime receipt differs")
        first_events = events if repeat == 0 else _base.original._read_jsonl(
            directory.parent.parent / "repeat-0" / family / "events.jsonl")
        launch = [i for i, event in enumerate(first_events) if event.get("event") == "REPRO_ENGINE_LAUNCH_PARENT"]
        loaded = [i for i, event in enumerate(first_events) if event.get("event") == "MODEL_READY"]
        first_raw = [i for i, event in enumerate(first_events) if event.get("event") == "REPRO_RUNTIME_OBSERVED"]
        _require(len(launch) == len(loaded) == len(first_raw) == 1 and launch[0] < loaded[0] < first_raw[0],
            "Pre-engine parent receipt must precede native model loading completion and raw runtime")
        _require(first_events[launch[0]].get("receipt") == snapshot["comparison_identity"].get("engine_launch_parent"),
            "Runtime pre-engine parent receipt is not bound to this cold's launch event")
        return base_evidence(directory, plan, plan_hash, family, cold, repeat, sources)

    harness._evidence = evidence
    return harness


_base.build_profile, _base._validate_profile, _base._validate_environment = build_profile, _validate_profile, _validate_environment
_base._new_harness = _new_harness
_base.validate_runtime_policy = validate_runtime_policy
PROFILE_MODES = runtime.PROFILE_MODES
load_plan, run, compare = _base.load_plan, _base.run, _base.compare


def prepare_plan(source_plan, profile_mode="qwen_tf32_off_only", baseline_environment=None):
    return _base.prepare_plan(source_plan, profile_mode, baseline_environment)


def main(argv=None):
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
