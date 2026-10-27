"""Finite, separately journaled repeats of the frozen native inference path.

Cold passes are separate CLI processes. Repeats within one cold pass reuse one
ModelSession, while each repeat has its own RunStore and immutable identity.
The comparison checks saved inputs, tokens, terminal text and technical receipts;
it never turns equal short answers into a claim of equal internal activations.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import json
import os
import math
from pathlib import Path
import sys
import time

from .storage import RunStore, PersistenceError, atomic_json, digest, file_sha256


SCHEMA = "classic_precision_reproducibility/v1"
_VOLATILE = {"time_unix", "started_unix", "pid", "process_start_ticks", "boot_id",
             "host", "hostname", "gpu_uuid", "uuid", "seconds", "free_gpu_gib",
             "available_ram_gib", "gpu_allocated_gib", "gpu_reserved_gib"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_plan(path):
    """Read a fixed experiment plan; reading this module does not import Torch."""
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    _require(isinstance(plan, dict) and plan.get("schema") == SCHEMA, "Unknown reproducibility plan schema")
    for name in ("source_run", "manifest", "ready", "asset_root"):
        _require(isinstance(plan.get(name), str) and Path(plan[name]).is_absolute(), name + " must be absolute")
    ids = plan.get("sample_ids")
    _require(isinstance(ids, list) and ids and all(isinstance(s, str) and s for s in ids)
             and len(ids) == len(set(ids)), "sample_ids must be a nonempty unique fixed list")
    _require(isinstance(plan.get("profile"), dict) and bool(plan["profile"].get("id")), "A versioned profile.id is required")
    for name in ("cold_passes", "repeats"):
        _require(type(plan.get(name)) is int and plan[name] >= 2, name + " must be an integer >=2")
    models = plan.get("models", ["nemo", "qwen", "cosmos"])
    _require(isinstance(models, list) and models and len(models) == len(set(models))
             and set(models) <= {"nemo", "qwen", "cosmos", "gemma"}, "Invalid plan models")
    for family in models:
        _validate_plan_profile(plan, family)
    _require(Path(plan["manifest"]).is_file() and Path(plan["ready"]).is_file(), "Original manifest/ready is missing")
    _require(Path(plan["source_run"]).is_dir() and Path(plan["asset_root"]).is_dir(), "Original source/assets is missing")
    if "manifest_sha256" in plan:
        _require(plan["manifest_sha256"] == file_sha256(plan["manifest"]), "Frozen source manifest changed")
    return plan, digest(plan)


def _stable(value):
    if isinstance(value, dict):
        return {k: _stable(v) for k, v in value.items() if k not in _VOLATILE}
    if isinstance(value, list):
        return [_stable(v) for v in value]
    return value


def runtime_identity(snapshot):
    """A runtime may explicitly expose its stable comparison identity."""
    _require(isinstance(snapshot, dict) and bool(snapshot), "Runtime snapshot is missing")
    value = snapshot.get("comparison_identity", snapshot)
    _require(isinstance(value, dict) and bool(value), "Stable runtime identity is missing")
    return _stable(value)


def _profile_contract():
    root = Path(__file__).resolve().parents[1] / "runtime_compat"
    def literals(path, names):
        result = {}
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in names:
                        result[target.id] = ast.literal_eval(node.value)
        _require(set(result) == set(names), "Runtime profile contract constants are missing")
        return result
    helper = root / "classic_precision_determinism.py"
    values = literals(helper, ("PROFILE_ID", "BASE_RUNNER_SHA256", "ENVIRONMENT", "FILL_UNINITIALIZED_MEMORY"))
    values["helper_sha256"] = file_sha256(helper)
    qsa = root / "classic_precision_qsa_determinism.py"
    values["qsa"] = literals(qsa, ("EXPECTED_SOURCE_SHA256", "TARGET_MODULE", "MODEL_ARCHITECTURE"))
    values["qsa_helper_sha256"] = file_sha256(qsa)
    return values


def _validate_profile(profile, family, require_applied):
    contract = _profile_contract()
    _require(isinstance(profile, dict) and profile.get("profile_id") == contract["PROFILE_ID"]
             and profile.get("family") == family, "Wrong runtime profile ID/family")
    _require(profile.get("base_runner_sha256") == contract["BASE_RUNNER_SHA256"]
             == file_sha256(Path(__file__).with_name("inference.py")), "Runtime base source identity mismatch")
    _require(profile.get("helper_sha256") == contract["helper_sha256"], "Runtime helper source identity mismatch")
    _require(profile.get("environment") == contract["ENVIRONMENT"], "Required deterministic environment differs")
    _require(profile.get("sdk_files_modified") is False, "Strict profile must preserve SDK files")
    if family == "qwen":
        qsa = profile.get("qsa_installation")
        _require(isinstance(qsa, dict), "Qwen QSA installation evidence is missing")
        expected = contract["qsa"]
        _require(qsa.get("expected_native_source_sha256") == expected["EXPECTED_SOURCE_SHA256"]
                 and qsa.get("helper_source_sha256") == contract["qsa_helper_sha256"]
                 and qsa.get("target_module") == expected["TARGET_MODULE"]
                 and qsa.get("model_architecture") == expected["MODEL_ARCHITECTURE"], "Qwen QSA source/target mismatch")
        _require(qsa.get("state") == "applied" if require_applied else
                 qsa.get("state") in ("pending_target_import", "applied"), "Qwen QSA replacement is not applied")
    return profile


def _validate_plan_profile(plan, family):
    contract = _profile_contract()
    profile = plan["profile"]
    _require(profile.get("id") == contract["PROFILE_ID"], "Plan strict profile ID differs")
    expected = {"determinism_helper_sha256": contract["helper_sha256"],
                "qsa_helper_sha256": contract["qsa_helper_sha256"],
                "base_runner_sha256": contract["BASE_RUNNER_SHA256"],
                "harness_sha256": file_sha256(__file__)}
    _require(profile.get("implementation") == expected, "Plan implementation fingerprints differ from actual profile/code")
    _require(profile.get("environment") == contract["ENVIRONMENT"], "Plan required base environment differs")
    overrides = profile.get("model_environment", {}).get(family, {})
    _require(overrides == ({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if family == "qwen" else {}),
             "Plan model-specific environment differs")
    return {**contract["ENVIRONMENT"], **overrides}


def _profile_version(profile):
    value = json.loads(json.dumps(profile))
    if isinstance(value.get("qsa_installation"), dict):
        value["qsa_installation"].pop("state", None)
    return value


def validate_runtime_policy(snapshot, family):
    """Equal snapshots certify only an actually enforced strict runtime."""
    _require(isinstance(snapshot, dict) and isinstance(snapshot.get("comparison_identity"), dict),
             "Explicit runtime comparison_identity is required; metadata fallback is forbidden")
    runtime = snapshot["comparison_identity"]
    profile = _validate_profile(runtime.get("profile"), family, require_applied=True)
    for key, required in (("deterministic_algorithms", True), ("deterministic_warn_only", False),
                          ("cudnn_benchmark", False), ("cudnn_deterministic", True)):
        _require(runtime.get(key) is required, "Strict runtime policy violation: " + key)
    _require(runtime.get("fill_uninitialized_memory") is _profile_contract()["FILL_UNINITIALIZED_MEMORY"][family],
             "Strict runtime policy violation: fill_uninitialized_memory")
    _require(runtime.get("inference_sha256") == profile["base_runner_sha256"], "Runtime inference source mismatch")
    expected_env = {**_profile_contract()["ENVIRONMENT"],
                    **({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if family == "qwen" else {})}
    _require(runtime.get("effective_environment") == expected_env, "Actual runtime environment is missing or differs")
    precision = runtime.get("fp32_precision")
    _require(precision in ({"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"},
                           {"allow_tf32_matmul": False, "allow_tf32_cudnn": False}), "TF32 precision policy missing or relaxed")
    _require(isinstance(runtime.get("packages"), dict) and all(runtime["packages"].get(k)
             for k in ("torch", "transformers", "numpy", "av")), "Required runtime package identities are missing")
    _require(bool(runtime.get("gpu_name")) and isinstance(runtime.get("compute_capability"), list)
             and len(runtime["compute_capability"]) == 2 and bool(runtime.get("cuda"))
             and bool(runtime.get("driver")), "Actual CUDA/GPU/driver runtime identity is missing")
    return runtime_identity(snapshot)


def _hash_receipts(events, sid, attempt_id, family):
    found = []
    stages = set()
    for event in events:
        if event.get("event") != "REPRO_INPUT_HASH" or event.get("sample_id") != sid:
            continue
        if event.get("attempt_id") not in (None, attempt_id):
            continue
        receipt = {k: v for k, v in event.items()
                   if k not in {"event", "shard_id", "time_unix", "attempt_id", "sample_id"}}
        stage = receipt.get("stage")
        _require(isinstance(stage, str), "Input hash receipt lacks a stage: " + sid)
        tensors = receipt.get("tensors")
        _require(isinstance(tensors, list) and bool(tensors), "Actual tensor hash list missing: " + sid)
        sizes = {"torch.bool": 1, "torch.uint8": 1, "torch.int8": 1, "torch.int16": 2,
                 "torch.int32": 4, "torch.int64": 8, "torch.float16": 2, "torch.bfloat16": 2,
                 "torch.float32": 4, "torch.float64": 8, "torch.uint16": 2, "torch.uint32": 4,
                 "torch.uint64": 8, "torch.complex64": 8, "torch.complex128": 16}
        for tensor in tensors:
            _require(isinstance(tensor, dict) and isinstance(tensor.get("name"), str) and tensor["name"]
                     and tensor.get("dtype") in sizes, "Tensor name/dtype missing or unsupported: " + sid)
            shape, sha = tensor.get("shape"), tensor.get("sha256")
            _require(isinstance(shape, list) and all(type(v) is int and v >= 0 for v in shape), "Invalid tensor shape: " + sid)
            _require(type(tensor.get("bytes")) is int and tensor["bytes"] == math.prod(shape) * sizes[tensor["dtype"]],
                     "Tensor byte count disagrees with shape/dtype: " + sid)
            _require(isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha),
                     "Tensor SHA256 missing or invalid: " + sid)
        if stage == "cpu_processor":
            names = [t["name"] for t in tensors]
            _require(len(names) == len(set(names)) and "input_ids" in names, "CPU processor input_ids hash missing: " + sid)
            pixels = "pixel_values_flat_video" if family == "nemo" else "pixel_values_videos"
            _require(pixels in names, "CPU processor pixels hash missing: " + sid)
            token = next(t for t in tensors if t["name"] == "input_ids")
            _require(token["dtype"] == "torch.int64" and len(token["shape"]) == 2
                     and token["shape"][0] == 1 and token["shape"][1] > 0, "CPU input_ids geometry/dtype changed: " + sid)
            pixel = next(t for t in tensors if t["name"] == pixels)
            shape = [180, 3, 512, 768] if family == "nemo" else ([1, 180, 2520, 768] if family == "gemma" else [138240, 1536])
            dtype = "torch.bfloat16" if family == "nemo" else "torch.float32"
            _require(pixel["shape"] == shape and pixel["dtype"] == dtype, "CPU processor pixel geometry/dtype changed: " + sid)
        elif stage == "gpu_forward":
            _require(all(t["name"] == "pixels" and t["dtype"] == "torch.bfloat16" for t in tensors), "GPU pixel name/dtype changed: " + sid)
            if family == "nemo":
                _require(all(len(t["shape"]) == 4 and t["shape"][0] > 0 and t["shape"][1:] == [3, 512, 768] for t in tensors)
                         and sum(t["shape"][0] for t in tensors) == 180, "GPU Nemo frame geometry changed: " + sid)
            else:
                shape = [1, 180, 2520, 768] if family == "gemma" else [138240, 1536]
                _require(len(tensors) == 1 and tensors[0]["shape"] == shape, "GPU pixel geometry changed: " + sid)
        else:
            raise ValueError("Unknown input hash stage: " + stage)
        stages.add(stage)
        found.append(receipt)
    _require("cpu_processor" in stages and "gpu_forward" in stages, "CPU processor/GPU forward hash is missing: " + sid)
    _require(len(found) == 2, "Each sample needs exactly one CPU and one GPU tensor hash receipt: " + sid)
    return found


class _EventSink:
    """ModelSession keeps this object while repeats rotate the durable store."""
    def __init__(self, store):
        self.store = store

    def event(self, event, **metadata):
        self.store.event(event, **metadata)


def _repeat_directory(output, pass_index, repeat_index):
    return Path(output) / ("cold-" + str(pass_index)) / ("repeat-" + str(repeat_index))



def _prepare_runtime_modules(family):
    """Load the guarded QSA leaf before pre-call certification.

    Native Qwen imports this leaf lazily in its first indexer forward. Importing
    it here runs the same source-verified hook without issuing a model request;
    the applied-hook requirement remains strict before the first request.
    """
    if family == "qwen":
        qsa = importlib.import_module("classic_precision_qsa_determinism")
        importlib.import_module(qsa.TARGET_MODULE)
        state = qsa.status()
        _require(isinstance(state, dict) and state.get("state") == "applied",
                 "Qwen QSA guarded import did not apply replacement")


def run(plan_path, output, model, pass_index, gpu, model_root, cosmos_package=None, repeats=None):
    """Run exactly one cold pass, with no failure retry or output overwriting."""
    from . import inference
    from .calibration import calibration_identity, calibration_ready
    from .summary import technical_passed
    plan, plan_hash = load_plan(plan_path)
    _require(model in plan.get("models", ["nemo", "qwen", "cosmos"]), "Model is outside this plan")
    _require(type(pass_index) is int and 0 <= pass_index < plan["cold_passes"], "Cold pass index is zero based and outside plan")
    _require(repeats in (None, plan["repeats"]), "CLI repeats differs from fixed plan")
    output = Path(output).resolve()
    source = Path(plan["source_run"]).resolve()
    _require(output != source and not source.is_relative_to(output), "Use a separate reproducibility output directory")
    _require(bool(str(gpu)) and "," not in str(gpu), "Exactly one GPU is required")
    if "torch" in sys.modules:
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "Set CUDA_VISIBLE_DEVICES before importing Torch")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest, rows, ready, assets, ready_path = inference.load_run_inputs(
        plan["manifest"], plan["ready"], plan["asset_root"])
    lookup = {row["sample_id"]: row for row in rows}
    _require(set(plan["sample_ids"]) <= set(lookup), "Plan contains unknown source sample IDs")
    selected = [lookup[sid] for sid in plan["sample_ids"]]
    ordered, qualification = inference.qualification_order(selected)
    _require(len(qualification) == 2, "Plan requires predetermined T2 and T3 qualification samples")
    _require(all(sid in ready for sid in plan["sample_ids"]), "Selected original ready videos are incomplete")
    expected_calibration = calibration_identity(plan["manifest"], manifest, model, model_root, cosmos_package)
    calibration_root = Path(plan.get("calibration_root", source / "calibration"))
    _require(calibration_ready(calibration_root, model, expected_calibration, manifest, ready),
             "Original 128-state CPU calibration is not ready")
    native_identity, model_path = inference.build_run_identity(
        plan["manifest"], ready_path, manifest, model, model_root, cosmos_package=cosmos_package)
    runtime = importlib.import_module("classic_precision_determinism")
    runtime.activate(model)
    profile_status = runtime.status()
    _require(isinstance(profile_status, dict) and bool(profile_status), "Strict runtime activation status is missing")
    _require(profile_status.get("profile_id") == plan["profile"]["id"], "Activated runtime profile differs from plan")
    _validate_profile(profile_status, model, require_applied=False)
    import torch
    cold_dir = output / ("cold-" + str(pass_index))
    cold_dir.mkdir(parents=True, exist_ok=True)
    # A failed cold pass is preserved as evidence and cannot be resumed under
    # its original identity. A genuinely new pass needs a newly frozen plan.
    claim = cold_dir / ("." + model + ".claim.json")
    with claim.open("x") as stream:
        json.dump({"plan_sha256": plan_hash, "model": model, "pass_index": pass_index}, stream)
    session, sink, reports = None, None, []
    resources = manifest["models"][model]["resources"]
    try:
        # Share the frozen source run's physical-GPU lease with native workers.
        with inference._gpu_lock(Path(plan.get("gpu_lock_root", source / "outputs")), torch), contextlib.ExitStack() as cleanup:
            admission = inference.resource_snapshot(torch, resources)
            _require(admission["admitted"], "Actual free VRAM/RAM does not meet original model budgets")
            for repeat_index in range(plan["repeats"]):
                identity = {**native_identity, "purpose": SCHEMA, "reproducibility": {
                    "plan_sha256": plan_hash, "profile": plan["profile"], "profile_status": profile_status,
                    "cold_pass": pass_index, "repeat": repeat_index,
                    "sample_ids": plan["sample_ids"], "harness_sha256": file_sha256(__file__)}}
                dest = _repeat_directory(output, pass_index, repeat_index)
                with RunStore(dest, identity, model, "repro", plan["sample_ids"]) as store:
                    _require(not store.outputs and not any(e.get("event") == "INFLIGHT" for e in store.events),
                             "Repeat directory has prior attempts; no implicit retry")
                    if sink is None:
                        sink = _EventSink(store)
                    else:
                        sink.store = store
                    store.event("REPRO_RESOURCE_ADMISSION", gpu_selection=str(gpu), **admission)
                    if session is None:
                        session = inference.ModelSession(model, model_path, sink, cosmos_package, resources)
                        cleanup.callback(session.close)
                        _prepare_runtime_modules(model)
                    snapshot = runtime.runtime_snapshot()
                    stable_runtime = validate_runtime_policy(snapshot, model)
                    store.event("REPRO_RUNTIME", snapshot=snapshot,
                                comparison_identity=stable_runtime, comparison_sha256=digest(stable_runtime))
                    qualified = False
                    for row in ordered:
                        sid = row["sample_id"]
                        store.begin(sid, digest({"row": row, "ready": ready[sid]}))
                        started = time.monotonic()
                        try:
                            def commit(record, name=sid):
                                return store.commit_output(name, {**record, "parent_sample_id": name,
                                    "reproducibility": {"plan_sha256": plan_hash, "cold_pass": pass_index,
                                                        "repeat": repeat_index}})
                            session.generate(row, ready[sid], assets, commit)
                            if not qualified and all(r["sample_id"] in store.outputs for r in qualification):
                                qualified = inference._qualification_marker(store, identity, model, "repro", qualification)
                        except PersistenceError:
                            raise
                        except BaseException as error:
                            if sid in store.outputs:
                                store.event("POST_OUTPUT_VALIDATION_FAILED", sample_id=sid,
                                    attempt_id=store.outputs[sid]["attempt_id"], error_type=type(error).__name__, error=str(error))
                            else:
                                store.fail(sid, error, interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)),
                                           elapsed_seconds=time.monotonic() - started)
                            raise
                        finally:
                            session.context.update(active=False, codes=None)
                    store.pending_ids(retry_failed=False, retry_interrupted=False)
                    _require(set(store.outputs) == set(plan["sample_ids"]), "Repeat is incomplete")
                    _require(technical_passed(list(store.outputs.values()), store.events) == set(plan["sample_ids"]),
                             "Repeat lacks native technical PASS")
                    for sid, raw in store.outputs.items():
                        _hash_receipts(store.events, sid, raw["attempt_id"], model)
                    report = {"status": "COMPLETE", "model": model, "pass_index": pass_index,
                              "repeat_index": repeat_index, "plan_sha256": plan_hash,
                              "samples": len(store.outputs), "runtime_sha256": digest(stable_runtime),
                              "model_calls": len(store.outputs),
                              "generation_seconds": sum(r.get("seconds", 0) for r in store.outputs.values()),
                              "load_seconds": next((e.get("load_seconds") for e in store.events
                                                    if e.get("event") == "MODEL_READY"), 0)}
                    store.event("REPRO_REPEAT_COMPLETE", **report)
                    atomic_json(store.directory / "reproducibility.json", report)
                    reports.append(report)
        result = {"status": "COMPLETE", "model": model, "pass_index": pass_index,
                  "plan_sha256": plan_hash, "repeats": reports}
        atomic_json(cold_dir / (model + ".json"), result)
        return result
    except BaseException as error:
        atomic_json(cold_dir / (model + ".json"), {"status": "FAILED", "model": model,
            "pass_index": pass_index, "plan_sha256": plan_hash,
            "error_type": type(error).__name__, "error": str(error), "completed_repeats": reports})
        raise


def _read_jsonl(path):
    payload = Path(path).read_bytes()
    _require(not payload or payload.endswith(b"\n"), "Incomplete journal tail: " + str(path))
    records = [json.loads(line) for line in payload.splitlines() if line.strip()]
    _require(all(isinstance(row, dict) for row in records), "Journal objects are required")
    return records


def _source_bindings(plan):
    from .ready import read_records
    manifest = json.loads(Path(plan["manifest"]).read_text())
    ids = [r["sample_id"] for r in manifest["rows"]]
    _require(len(ids) == len(set(ids)), "Duplicate source manifest IDs")
    rows = {r["sample_id"]: r for r in manifest["rows"]}
    ready, _, _ = read_records(plan["ready"])
    _require(set(plan["sample_ids"]) <= set(rows) & set(ready), "Original target source/ready is incomplete")
    return {sid: (rows[sid], ready[sid]) for sid in plan["sample_ids"]}


def _differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in
                (_differences(a[k], b[k], prefix + "." + k) if k in a and k in b else [prefix + "." + k])]
    return [] if a == b else [prefix.lstrip(".")]


def _evidence(directory, plan, plan_hash, family, cold, repeat, sources):
    from .summary import technical_passed
    identity = json.loads((directory / "identity.json").read_text())
    completion = json.loads((directory.parent.parent / (family + ".json")).read_text())
    _require(completion.get("status") == "COMPLETE" and completion.get("plan_sha256") == plan_hash,
             "Cold pass did not complete normally")
    binding = identity.get("reproducibility", {})
    _require(identity.get("purpose") == SCHEMA and identity.get("model_key") == family, "Wrong output purpose/model")
    _require(binding.get("plan_sha256") == plan_hash and binding.get("profile") == plan["profile"], "Plan/profile mismatch")
    _require(binding.get("cold_pass") == cold and binding.get("repeat") == repeat
             and binding.get("sample_ids") == plan["sample_ids"], "Cold/repeat/selected sample binding mismatch")
    _require(binding.get("harness_sha256") == file_sha256(__file__), "Harness source identity changed")
    _require(identity.get("manifest_sha256") == file_sha256(plan["manifest"]), "Source manifest mismatch")
    rows, events = _read_jsonl(directory / "outputs.jsonl"), _read_jsonl(directory / "events.jsonl")
    ids = [r.get("sample_id") for r in rows]
    _require(len(ids) == len(set(ids)) and set(ids) == set(plan["sample_ids"]), "Missing, duplicate or unexpected output IDs")
    _require(technical_passed(rows, events) == set(ids), "Native technical PASS is missing or superseded")
    starts = [e for e in events if e.get("event") == "WORKER_START"]
    _require(len(starts) == 1, "Repeat was resumed or lacks a unique process owner")
    owner = starts[0].get("owner", {})
    _require(bool(owner.get("pid")) and bool(owner.get("host")), "Process owner identity missing")
    runtime = [e for e in events if e.get("event") == "REPRO_RUNTIME"]
    _require(len(runtime) == 1, "Expected one bound runtime snapshot per repeat")
    runtime_id = runtime_identity(runtime[0].get("snapshot"))
    validate_runtime_policy(runtime[0].get("snapshot"), family)
    activated_profile = _validate_profile(binding.get("profile_status"), family, require_applied=False)
    _require(_profile_version(activated_profile) == _profile_version(runtime_id["profile"]),
             "Activated profile status differs from actual runtime version")
    _require(runtime[0].get("comparison_identity") == runtime_id
             and runtime[0].get("comparison_sha256") == digest(runtime_id), "Runtime fingerprint mismatch")
    _require(sum(e.get("event") == "REPRO_REPEAT_COMPLETE" for e in events) == 1, "Repeat completion evidence is missing")
    fingerprints = {}
    for row in rows:
        sid, aid = row["sample_id"], row["attempt_id"]
        original, ready = sources[sid]
        _require(row.get("ready_record") == ready and row.get("ready_identity") == digest(ready)
                 and row.get("row_identity") == digest({"row": original, "ready": ready}),
                 "Raw input no longer matches original source/ready: " + sid)
        _require(all(row.get(k) == original.get(k) for k in ("task", "truth", "source_parameters")),
                 "Raw condition or ground truth differs from original: " + sid)
        attempts = [e for e in events if e.get("event") == "INFLIGHT" and e.get("sample_id") == sid]
        _require(len(attempts) == 1 and attempts[0].get("attempt_id") == aid, "Implicit retry or unmatched attempt: " + sid)
        _require(row.get("parent_sample_id") == sid and row.get("reproducibility") == {
            "plan_sha256": plan_hash, "cold_pass": cold, "repeat": repeat}, "Raw repetition identity mismatch: " + sid)
        tokens = row.get("output_token_ids")
        _require(isinstance(tokens, list) and tokens and all(type(t) is int for t in tokens), "Token IDs missing: " + sid)
        _require(row.get("raw_answer") in ("LEFT", "RIGHT") and row.get("finish_reason") == "stop", "Nonterminal/invalid answer: " + sid)
        receipts = _hash_receipts(events, sid, aid, family)
        from .inference import GENERATION
        generation = row.get("effective_generation", {})
        _require(isinstance(generation, dict) and all(generation.get(k) == v for k, v in GENERATION.items()),
                 "effective_generation differs from fixed native GENERATION: " + sid)
        keys = ("task", "truth", "row_identity", "ready_identity", "ready_record", "source_parameters",
                "effective_generation", "actual_input_tokens", "raw_answer", "output_token_ids", "finish_reason", "stop_reason")
        _require(all(k in row for k in ("row_identity", "ready_identity", "ready_record", "input_receipt",
                                        "effective_generation", "actual_input_tokens")), "Source/input binding missing: " + sid)
        fingerprint = {k: row.get(k) for k in keys}
        # The first call deeply inverts pixels; later calls reuse qualified
        # geometry. Those validation-only fields differ by design, while all
        # tensor bytes, exact prompt text and effective generation must match.
        input_receipt = row["input_receipt"]
        _require(bool(input_receipt.get("chat_template_sha256")), "Exact formatted prompt identity missing: " + sid)
        fingerprint["prompt_sha256"] = input_receipt["chat_template_sha256"]
        fingerprint["input_hash_receipts"] = receipts
        fingerprints[sid] = fingerprint
    common = {k: v for k, v in identity.items() if k not in {"reproducibility", "storage_all_row_ids_sha256"}}
    return {"fingerprints": fingerprints, "native_identity": common,
            "profile_status": binding.get("profile_status"), "runtime": runtime_id, "owner": owner}


def compare(plan_path, output, models=None):
    """Fail closed on incomplete passes or any input/token/runtime mismatch."""
    plan, plan_hash = load_plan(plan_path)
    sources = _source_bindings(plan)
    models = models or plan.get("models", ["nemo", "qwen", "cosmos"])
    _require(set(models) <= set(plan.get("models", ["nemo", "qwen", "cosmos"])), "Requested model is outside plan")
    report = {"schema": SCHEMA, "plan_sha256": plan_hash, "status": "PASS", "models": {}, "errors": []}
    for family in models:
        baseline, owners, checked = None, {}, 0
        mismatches = []
        for cold in range(plan["cold_passes"]):
            for repeat in range(plan["repeats"]):
                path = _repeat_directory(output, cold, repeat) / family
                try:
                    value = _evidence(path, plan, plan_hash, family, cold, repeat, sources)
                    owner = value.pop("owner")
                    owner_key = tuple(owner.get(k) for k in ("host", "boot_id", "pid", "process_start_ticks"))
                    if cold in owners:
                        _require(owner_key == owners[cold], "Same-engine repeats came from different processes")
                    else:
                        _require(owner_key not in owners.values(), "Cold passes reused a process")
                        owners[cold] = owner_key
                    if baseline is None:
                        baseline = value
                    else:
                        for key in ("native_identity", "profile_status", "runtime"):
                            changed = _differences(baseline[key], value[key])
                            _require(not changed, key + " mismatch: " + ",".join(changed[:32]))
                        for sid in plan["sample_ids"]:
                            a, b = baseline["fingerprints"][sid], value["fingerprints"][sid]
                            different = [k for k in a if a[k] != b[k]]
                            _require(not different, "Exact evidence mismatch " + sid + ": " + ",".join(different))
                    checked += 1
                except (OSError, ValueError, KeyError, TypeError) as error:
                    mismatches.append({"cold_pass": cold, "repeat": repeat, "directory": str(path),
                                       "error_type": type(error).__name__, "error": str(error)})
        ok = not mismatches and checked == plan["cold_passes"] * plan["repeats"]
        report["models"][family] = {"status": "PASS" if ok else "FAIL", "checked_repeat_stores": checked,
            "expected_repeat_stores": plan["cold_passes"] * plan["repeats"],
            "samples_per_repeat": len(plan["sample_ids"]),
            "reference_fingerprints_sha256": digest(baseline["fingerprints"]) if baseline else None,
            "mismatches": mismatches}
        if not ok:
            report["status"] = "FAIL"
            report["errors"].append(family)
    report["claim"] = "Exact saved input hashes, generated token IDs/text and technical receipts on this fixed sample set and runtime; no untested hardware or full-curve claim."
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--plan", required=True)
    runner.add_argument("--output", required=True)
    runner.add_argument("--model", required=True)
    runner.add_argument("--pass-index", type=int, required=True)
    runner.add_argument("--gpu", required=True)
    runner.add_argument("--model-root", required=True)
    runner.add_argument("--cosmos-package")
    runner.add_argument("--repeats", type=int)
    checker = commands.add_parser("compare")
    checker.add_argument("--plan", required=True)
    checker.add_argument("--output", required=True)
    checker.add_argument("--model", action="append")
    checker.add_argument("--report")
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.plan, args.output, args.model, args.pass_index, args.gpu,
                     args.model_root, args.cosmos_package, args.repeats)
    else:
        result = compare(args.plan, args.output, args.model)
        if args.report:
            atomic_json(args.report, result)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0 if result["status"] in ("PASS", "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(main())
