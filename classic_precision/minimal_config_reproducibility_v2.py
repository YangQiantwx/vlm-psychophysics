"""Versioned native-MP thread evidence overlay; frozen v1 runs remain intact."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import math
from pathlib import Path

from .storage import file_sha256
from runtime_compat import classic_precision_minimal_config_v2 as runtime

SCHEMA = "classic_precision_minimal_config_reproducibility/v2"
FROZEN_ENTRY_SHA256 = "162551b7cb625c22cb388f931c8bcb0c89aff2e89e4d561f053526ef438fd75a"
_path = Path(__file__).with_name("minimal_config_reproducibility.py")
if file_sha256(_path) != FROZEN_ENTRY_SHA256:
    raise RuntimeError("Frozen v1 minimal entry source changed")
_spec = importlib.util.spec_from_file_location("classic_precision._minimal_v2_frozen_entry", _path)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_base.SCHEMA, _base.runtime = SCHEMA, runtime
_require = _base._require
_original_build_profile, _original_validate_profile = _base.build_profile, _base._validate_profile


def build_profile(profile_mode="native_cosmos", baseline_environment=None):
    value = _original_build_profile(profile_mode, baseline_environment)
    value["evidence_contract"] = runtime.EVIDENCE_CONTRACT
    value["native_thread_source_sha256"] = deepcopy(runtime.THREAD_SOURCE_SHA256)
    value["implementation"].update(overlay_entry_sha256=file_sha256(__file__),
        frozen_policy_helper_sha256=runtime.FROZEN_HELPER_SHA256)
    if profile_mode != "native_cosmos":
        value["native_environment_mutation_allowlist"].update(
            OMP_NUM_THREADS={"from": None, "to": "native_startup_omp_num_threads(local_world_size=1)"},
            VLLM_OMP_NUM_THREADS_SET_BY_VLLM={"from": None, "to": "1"})
    return value


def _validate_profile(profile, family, require_applied):
    value = _original_validate_profile(profile, family, require_applied)
    _require(value.get("evidence_contract") == runtime.EVIDENCE_CONTRACT
        and value.get("frozen_policy_helper_sha256") == runtime.FROZEN_HELPER_SHA256
        and value.get("native_thread_source_sha256") == runtime.THREAD_SOURCE_SHA256,
        "Native thread evidence contract/source identity differs")
    return value


def _validate_thread_observation(value, scope):
    before, now = value.get("native_thread_baseline"), value.get("native_thread_observation")
    _require(isinstance(before, dict) and before == now and now.get("scope") == scope,
        "Actual native thread runtime drifted or baseline missing")
    _require("observation_error" not in now, "Native thread observation is incomplete: " + str(now.get("observation_error")))
    _require(type(now.get("torch_num_threads")) is int and now["torch_num_threads"] > 0, "Actual Torch thread count missing")
    affinity = now.get("cpu_affinity")
    _require(isinstance(affinity, list) and affinity and all(type(cpu) is int and cpu >= 0 for cpu in affinity)
        and affinity == sorted(set(affinity)), "Actual Linux CPU affinity missing")
    _require(now.get("num_local_procs") == 1, "Native single-GPU local world size differs")
    limit = now.get("cgroup_cpu_limit")
    _require(limit is None or (type(limit) in (int, float) and math.isfinite(limit) and limit > 0), "Actual cgroup CPU limit invalid")
    available = max(1, len(affinity) if limit is None else min(len(affinity), int(limit)))
    _require(type(now.get("native_available_cpu_count")) is int and now["native_available_cpu_count"] == available
        and type(now.get("native_startup_omp_num_threads")) is int and now["native_startup_omp_num_threads"] == available,
        "Native startup OMP value disagrees with affinity/cgroup derivation")
    expected = runtime.THREAD_SOURCE_SHA256
    provenance = now.get("source_provenance")
    _require(isinstance(provenance, dict) and set(provenance) == set(expected)
        and all(provenance[name] == {"expected_sha256": sha, "actual_sha256": sha} for name, sha in expected.items()),
        "Native thread source provenance differs")
    functions = now.get("function_identities", {})
    _require(set(functions) == {"startup_omp_num_threads", "available_cpu_count", "_cgroup_cpu_limit"}, "Native CPU function identities missing")
    for name, identity in functions.items():
        _require(identity.get("qualname") == name and identity.get("module") in expected
            and identity.get("source_sha256") == expected[identity["module"]] and identity.get("wrapped") is False,
            "Native CPU function source identity differs: " + name)
    env = value["native_postload_environment"]
    _require(now.get("omp_num_threads") == env["OMP_NUM_THREADS"]
        and now.get("vllm_omp_marker") == env["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"], "Native thread/env observation differs")
    if scope == "worker" and now["vllm_omp_marker"] == "1":
        _require(now["torch_num_threads"] == 1, "Native serving worker must use one Torch runtime thread")
    return now


def _validate_environment(value, profile):
    startup, baseline = profile["startup_environment"], value.get("native_postload_environment")
    _require(isinstance(baseline, dict) and set(baseline) == set(startup)
        and value.get("observed_environment") == baseline, "Environment drifted after native post-load baseline")
    changed = {key for key in startup if baseline[key] != startup[key]}
    if profile["family"] == "cosmos":
        _require(not changed, "Unexpected Cosmos native environment mutation")
        return
    scope = "worker" if "worker_class" in value else "parent"
    observed = _validate_thread_observation(value, scope)
    allowed = {"VLLM_WORKER_MULTIPROC_METHOD", "OMP_NUM_THREADS", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM"}
    _require(changed <= allowed, "Undeclared native post-load environment mutation: " + ",".join(sorted(changed - allowed)))
    if "VLLM_WORKER_MULTIPROC_METHOD" in changed:
        _require(startup["VLLM_WORKER_MULTIPROC_METHOD"] is None and baseline["VLLM_WORKER_MULTIPROC_METHOD"] == "spawn",
            "Native multiprocessing method transition differs")
    native_mpi_expected = (scope == "worker" and profile["mode"] != "qwen_schedule_only" and startup["OMP_NUM_THREADS"] is None
        and startup["VLLM_ENABLE_V1_MULTIPROCESSING"] != "0")
    native_omp = native_mpi_expected or "OMP_NUM_THREADS" in changed or "VLLM_OMP_NUM_THREADS_SET_BY_VLLM" in changed
    if native_omp:
        _require(profile["mode"] != "qwen_schedule_only" and startup["OMP_NUM_THREADS"] is None
            and startup["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] is None
            and baseline["OMP_NUM_THREADS"] == str(observed["native_startup_omp_num_threads"])
            and baseline["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] == "1",
            "Native MPI OMP transition lacks exact derived value/marker/provenance")


_base.build_profile, _base._validate_profile, _base._validate_environment = build_profile, _validate_profile, _validate_environment
PROFILE_MODES = runtime.PROFILE_MODES
prepare_plan, load_plan, run, compare = _base.prepare_plan, _base.load_plan, _base.run, _base.compare
validate_runtime_policy, _new_harness = _base.validate_runtime_policy, _base._new_harness


def main(argv=None):
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
