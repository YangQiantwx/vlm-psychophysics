"""Prespecified adjacent repeats; durably stop at the first valid difference.

Each arm is one independently loaded native Qwen engine with only the reviewed
post-load QSA selector. Two arms may run on separately leased physical GPUs.
The first eight IDs are a fixed coverage stage, never a scored subset.
"""
from __future__ import annotations

import argparse
import contextlib
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from . import qsa_only_reproducibility as qsa
from . import reproducibility as original
from .storage import RunStore, PersistenceError, atomic_json, digest, file_sha256, owner_identity, _sync_directory

SCHEMA = "classic_precision_qsa_adjacent_early_stop/v1"
FROZEN_QSA_ENTRY_SHA256 = "b10f7824ef476ce70d54a991cabbd0f2a6eded9b7b29bd215350d5e02cfe4d0a"
FROZEN_QSA_RUNTIME_SHA256 = "3210381807e5f60eb5cf5c5b7537e462217bc6ac54771bf013111c27b2e2ac07"
_require = original._require
_OUTPUT_FIELDS = {"raw_answer", "output_token_ids", "finish_reason", "stop_reason"}
_TERMINAL_STOP = {"DIFFERENT", "INVALID_EVIDENCE", "STOPPED"}
_OWNER_FIELDS = ("host", "boot_id", "pid", "process_start_ticks")


def _implementation():
    _require(file_sha256(qsa.__file__) == FROZEN_QSA_ENTRY_SHA256, "Frozen QSA entry changed")
    _require(file_sha256(qsa.runtime.__file__) == FROZEN_QSA_RUNTIME_SHA256, "Frozen QSA runtime changed")
    return {"entry_sha256": file_sha256(__file__), "qsa_entry_sha256": FROZEN_QSA_ENTRY_SHA256,
            "qsa_runtime_sha256": FROZEN_QSA_RUNTIME_SHA256,
            "inference_sha256": qsa.runtime.BASE_RUNNER_SHA256}


def prepare_plan(source_plan, arm, gpu, gpu_uuid, prefix_ids=None, baseline_environment=None):
    base = qsa.prepare_plan(source_plan, "qwen_qsa_only", baseline_environment)
    ids = base["sample_ids"]
    t3 = base["auxiliary_T3_qualification_id"]
    prefix = list(prefix_ids) if prefix_ids is not None else [ids[0], t3] + [s for s in ids if s not in (ids[0], t3)][:6]
    _require(len(prefix) == 8 and len(set(prefix)) == 8 and set(prefix) <= set(ids) and t3 in prefix,
             "Prespecify eight unique original IDs including T3")
    _require(prefix[:2] == [source_plan["scientific_ids"][0], t3], "Prefix must start with original T2/T3 qualification IDs")
    ordered = prefix + [sid for sid in ids if sid not in prefix]
    base.update(schema=SCHEMA, arm=arm, gpu=str(gpu), gpu_uuid=gpu_uuid, cold_passes=1, repeats=2,
        sample_ids=ordered, scientific_ids=[s for s in ordered if s != t3], prefix_ids=prefix,
        model_calls_per_model=98, total_model_calls=98, implementation=_implementation(),
        purpose="Adjacent repeats on the fixed original 49 inputs; stop immediately on valid output difference or invalid controls",
        comparison_contract={"within_arm": "same physical GPU, process, engine and actual input/runtime",
            "cross_arm_allowed_differences": ["arm", "gpu", "gpu_uuid", "process_owner"],
            "pair_order": "per ID repeat0 then repeat1 before next ID", "prefix_stage_size": 8,
            "prefix_design": "prespecified targeted repeatability screen including historically variable IDs; not an unbiased accuracy estimate",
            "prefix_barrier": "both independent arms complete eight matching pairs and cross-arm controls/full outputs match before expansion",
            "peer_barrier_timeout_seconds": 1800,
            "stop": "after durable in-flight output and technical receipts; no retry or subsequent model request"})
    return base


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    _require(plan.get("schema") == SCHEMA and plan.get("implementation") == _implementation(), "Early-stop plan/source identity differs")
    _require(isinstance(plan.get("arm"), str) and bool(plan["arm"]), "Arm identity missing")
    _require(isinstance(plan.get("gpu"), str) and plan["gpu"] and "," not in plan["gpu"], "Exactly one sealed GPU required")
    _require(isinstance(plan.get("gpu_uuid"), str) and plan["gpu_uuid"].startswith("GPU-"), "Seal physical GPU UUID")
    _require((plan.get("cold_passes"), plan.get("repeats"), plan.get("total_model_calls"), plan.get("model_calls_per_model")) == (1, 2, 98, 98), "Early-stop finite budget differs")
    _require(plan.get("sample_ids", [])[:8] == plan.get("prefix_ids") and len(set(plan["prefix_ids"])) == 8
             and plan.get("auxiliary_T3_qualification_id") in plan["prefix_ids"], "Prespecified coverage prefix differs")
    expected = prepare_plan(plan, plan["arm"], plan["gpu"], plan["gpu_uuid"], plan["prefix_ids"], plan.get("profile", {}).get("baseline_environment"))
    _require(plan.get("comparison_contract") == expected["comparison_contract"], "Sealed comparison/stop contract differs")
    # Reuse every native plan/profile/49-source guard without editing its bound plan.
    view = deepcopy(plan)
    view.update(schema=qsa.SCHEMA, cold_passes=2, total_model_calls=196, model_calls_per_model=196)
    for name in ("source_run", "manifest", "ready", "asset_root"):
        _require(isinstance(plan.get(name), str) and Path(plan[name]).is_absolute(), name + " must be absolute")
    qsa._base._validate_plan_profile(view, "qwen")
    qsa._base._validate_design(view)
    _require(Path(plan["ready"]).is_file() and Path(plan["source_run"]).is_dir() and Path(plan["asset_root"]).is_dir(), "Original source/assets missing")
    return plan, digest(plan)


def _directory(output, repeat):
    return Path(output) / ("repeat-" + str(repeat)) / "qwen"


def _normalized_gpu_uuid(value):
    _require(isinstance(value, str) and bool(value), "Physical GPU UUID missing or malformed")
    raw = value.removeprefix("GPU-")
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError) as error:
        raise ValueError("Physical GPU UUID missing or malformed") from error
    _require(raw.lower() == str(parsed), "Physical GPU UUID must be canonical")
    return str(parsed)


def _snapshot_identity(snapshot, plan):
    stable = qsa.validate_runtime_policy(snapshot, "qwen")
    _require(_normalized_gpu_uuid(snapshot.get("gpu_uuid")) == _normalized_gpu_uuid(plan["gpu_uuid"])
             and snapshot.get("visible_devices") == plan["gpu"], "Actual physical GPU/visibility differs from sealed arm")
    _require(isinstance(snapshot.get("pid"), int) and snapshot["pid"] > 0 and bool(snapshot.get("hostname")), "Actual parent process identity missing")
    return {"policy": stable, "gpu_uuid": snapshot["gpu_uuid"], "visible_devices": snapshot["visible_devices"],
            "pid": snapshot["pid"], "hostname": snapshot["hostname"]}


def _owner_key(owner):
    _require(isinstance(owner, dict) and all(owner.get(k) for k in _OWNER_FIELDS), "Complete owner identity missing")
    return tuple(owner[k] for k in _OWNER_FIELDS)


def _publish_claim(path, value):
    """Publish a fully durable JSON claim atomically, refusing an existing arm."""
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".pending")
    try:
        atomic_json(temporary, value)
        os.link(temporary, path)  # atomic visibility and exclusive no-overwrite
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _sync_directory(path.parent)


def _call_evidence(directory, plan, plan_hash, repeat, sid, sources):
    from .inference import GENERATION
    from .summary import technical_passed, model_snapshot
    directory = Path(directory)
    identity, rows, events, raw_tail, event_tail = model_snapshot(directory)
    _require(not raw_tail and not event_tail, "Incomplete durable journal tail")
    binding = identity.get("reproducibility", {})
    _require(identity.get("purpose") == SCHEMA and identity.get("model_key") == "qwen"
             and identity.get("manifest_sha256") == file_sha256(plan["manifest"]), "Stored native identity/source differs")
    _require(binding == {"plan_sha256": plan_hash, "profile": plan["profile"], "arm": plan["arm"],
        "repeat": repeat, "sample_ids": plan["sample_ids"], "implementation": plan["implementation"]}, "Stored repeat identity differs")
    found = [row for row in rows if row.get("sample_id") == sid]
    _require(len(found) == 1 and len({r.get("sample_id") for r in rows}) == len(rows), "Missing/duplicate raw output")
    row = found[0]
    _require(sid in technical_passed(rows, events), "Native technical PASS missing")
    source, ready = sources[sid]
    _require(row.get("ready_record") == ready and row.get("ready_identity") == digest(ready)
             and row.get("row_identity") == digest({"row": source, "ready": ready}), "Raw input/source binding differs")
    _require(all(row.get(k) == source.get(k) for k in ("task", "truth", "source_parameters")), "Raw scientific condition differs")
    attempts = [e for e in events if e.get("event") == "INFLIGHT" and e.get("sample_id") == sid]
    _require(len(attempts) == 1 and attempts[0].get("attempt_id") == row.get("attempt_id"), "Implicit retry/unmatched attempt")
    _require(row.get("parent_sample_id") == sid and row.get("reproducibility") == {"plan_sha256": plan_hash, "arm": plan["arm"], "repeat": repeat}, "Raw arm/repeat binding differs")
    _require(isinstance(row.get("output_token_ids"), list) and row["output_token_ids"]
             and all(type(t) is int for t in row["output_token_ids"]) and row.get("raw_answer") in ("LEFT", "RIGHT")
             and row.get("finish_reason") == "stop" and "stop_reason" in row, "Incomplete/nonterminal raw answer")
    _require(type(row.get("seconds")) in (int, float) and math.isfinite(row["seconds"]) and row["seconds"] >= 0, "Actual generation elapsed missing")
    _require(all(row.get("effective_generation", {}).get(k) == v for k, v in GENERATION.items()), "Native effective generation differs")
    prompt = row.get("input_receipt", {}).get("chat_template_sha256")
    _require(isinstance(prompt, str) and len(prompt) == 64 and type(row.get("actual_input_tokens")) is int, "Actual prompt/token identity missing")
    runtime = [e for e in events if e.get("event") == "REPRO_RUNTIME"]
    _require(len(runtime) == 1, "Initial runtime receipt missing/duplicated")
    observed = [i for i, e in enumerate(events) if e.get("event") == "REPRO_RUNTIME_OBSERVED"]
    bound = [i for i, e in enumerate(events) if e.get("event") == "REPRO_RUNTIME"]
    _require(len(observed) == 1 and observed[0] < bound[0]
             and events[observed[0]].get("snapshot") == runtime[0]["snapshot"]
             and events[observed[0]].get("evidence_contract") == qsa.runtime.EVIDENCE_CONTRACT,
             "Durable native raw runtime observation must precede bound receipt")
    _, _, launch_events, launch_raw_tail, launch_event_tail = model_snapshot(_directory(directory.parents[1], 0))
    _require(not launch_raw_tail and not launch_event_tail, "Incomplete native launch journal tail")
    launch = [i for i, e in enumerate(launch_events) if e.get("event") == "REPRO_ENGINE_LAUNCH_PARENT"]
    loaded = [i for i, e in enumerate(launch_events) if e.get("event") == "MODEL_READY"]
    first_raw = [i for i, e in enumerate(launch_events) if e.get("event") == "REPRO_RUNTIME_OBSERVED"]
    _require(len(launch) == len(loaded) == len(first_raw) == 1 and launch[0] < loaded[0] < first_raw[0]
             and launch_events[launch[0]].get("receipt") == runtime[0]["snapshot"]["comparison_identity"].get("engine_launch_parent"),
             "Original native engine launch/load/raw runtime evidence differs")
    baseline = _snapshot_identity(runtime[0]["snapshot"], plan)
    calls = [e for e in events if e.get("event") == "REPRO_CALL_RUNTIME" and e.get("sample_id") == sid]
    _require([e.get("stage") for e in calls] == ["before", "after"], "Per-call runtime stages differ")
    _require(all(_snapshot_identity(e["snapshot"], plan) == baseline for e in calls), "Runtime changed around request")
    owners = [e.get("owner") for e in events if e.get("event") == "WORKER_START"]
    _require(len(owners) == 1 and all(owners[0].get(k) for k in ("host", "boot_id", "pid", "process_start_ticks")), "Unique complete process owner missing")
    _require(owners[0]["pid"] == baseline["pid"] and owners[0]["host"] == baseline["hostname"], "Store/runtime parent owner differs")
    elapsed = [e for e in events if e.get("event") == "EARLY_CALL_COMPLETE" and e.get("sample_id") == sid]
    _require(len(elapsed) == 1 and elapsed[0].get("attempt_id") == row["attempt_id"]
             and type(elapsed[0].get("elapsed_seconds")) in (int, float) and math.isfinite(elapsed[0]["elapsed_seconds"])
             and elapsed[0]["elapsed_seconds"] >= 0, "Request completion elapsed missing")
    fingerprint = {k: row.get(k) for k in ("task", "truth", "source_parameters", "ready_record", "ready_identity", "row_identity", "effective_generation", "actual_input_tokens", *_OUTPUT_FIELDS)}
    fingerprint.update(prompt_sha256=prompt, input_hash_receipts=original._hash_receipts(events, sid, row["attempt_id"], "qwen"))
    native = {k: v for k, v in identity.items() if k not in {"reproducibility", "storage_all_row_ids_sha256"}}
    return {"sample_id": sid, "fingerprint": fingerprint, "runtime": baseline, "native_identity": native,
        "owner": owners[0], "raw_sha256": digest(row), "raw": row, "elapsed_seconds": elapsed[0]["elapsed_seconds"]}


def compare_calls(left, right, cross_arm=False):
    """Controls first: changed inputs/runtime can never become valid DIFFERENT."""
    for key in ("sample_id", "native_identity"):
        _require(left[key] == right[key], key + " differs")
    a, b = deepcopy(left["runtime"]), deepcopy(right["runtime"])
    if cross_arm:
        _require(left["owner"]["host"] == right["owner"]["host"] and left["owner"]["boot_id"] == right["owner"]["boot_id"], "Cross-arm host/boot differs")
        _require(left["owner"]["pid"] != right["owner"]["pid"] and a["gpu_uuid"] != b["gpu_uuid"], "Cross-arm engines/GPU must be independent")
        for field in ("gpu_uuid", "visible_devices", "pid"):
            a.pop(field); b.pop(field)
    else:
        _require(_owner_key(left["owner"]) == _owner_key(right["owner"]), "Same-engine repeat process owner differs")
    _require(a == b, "Actual runtime identity differs")
    controls = lambda call: {k: v for k, v in call["fingerprint"].items() if k not in _OUTPUT_FIELDS}
    _require(controls(left) == controls(right), "Actual CPU/GPU input or generation controls differ")
    differences = [key for key in sorted(_OUTPUT_FIELDS) if left["fingerprint"][key] != right["fingerprint"][key]]
    return {"comparison_status": "DIFFERENT" if differences else "MATCH", "evidence_status": "COMPLETE",
            "sample_id": left["sample_id"], "different_fields": differences,
            "left_raw_sha256": left["raw_sha256"], "right_raw_sha256": right["raw_sha256"]}


def _pair(output, plan, plan_hash, sid, sources):
    calls = [_call_evidence(_directory(output, r), plan, plan_hash, r, sid, sources) for r in range(2)]
    return calls, compare_calls(*calls)


def _available_calls(output, plan, plan_hash, sid, sources):
    """A peer still generating is pending; completed receipts remain strict."""
    from .summary import model_snapshot
    found = []
    for repeat in range(2):
        directory = _directory(output, repeat)
        if not all((directory / name).is_file() for name in ("identity.json", "outputs.jsonl", "events.jsonl")):
            continue
        _, _, events, raw_tail, event_tail = model_snapshot(directory)
        if raw_tail or event_tail:
            continue
        if not any(e.get("event") == "EARLY_CALL_COMPLETE" and e.get("sample_id") == sid for e in events):
            continue
        found.append(_call_evidence(directory, plan, plan_hash, repeat, sid, sources))
    return found


class _Stop:
    def __init__(self):
        self.reason = None

    def request(self, number, frame):
        self.reason = "signal " + str(number)


def _peer_claim(peer_output, peer):
    path = Path(peer_output) / ".qwen.claim.json"
    if not path.exists(): return None
    claim = json.loads(path.read_text())
    _require(claim.get("plan_sha256") == peer[1] and claim.get("implementation") == peer[0]["implementation"], "Peer claim plan/source differs")
    _owner_key(claim.get("owner"))
    return claim


def _stop_reason(output, peer_output, stop, plan, plan_hash, owner, peer):
    if stop.reason:
        return stop.reason
    request_path = Path(output) / "STOP_REQUEST.json"
    if request_path.exists():
        request = json.loads(request_path.read_text())
        _require(request.get("plan_sha256") == plan_hash and request.get("arm") == plan["arm"]
                 and _owner_key(request.get("target_owner")) == _owner_key(owner), "External stop request target differs")
        return "external stop request"
    if peer_output:
        path = Path(peer_output) / "early-stop.json"
        if path.exists():
            result = json.loads(path.read_text())
            claim = _peer_claim(peer_output, peer)
            _require(claim is not None and result.get("plan_sha256") == peer[1]
                     and result.get("arm") == peer[0]["arm"]
                     and _owner_key(result.get("owner")) == _owner_key(claim["owner"]), "Peer stop result plan/owner differs")
            if result.get("comparison_status") in _TERMINAL_STOP:
                return "peer " + result["comparison_status"]
    return None


def _validate_peers(plan, peer):
    for key in ("profile", "sample_ids", "prefix_ids", "implementation", "manifest", "manifest_sha256", "source_run", "ready", "asset_root", "calibration_root", "gpu_lock_root", "comparison_contract", "launch_environment", "launch_cleared_environment"):
        _require(plan.get(key) == peer.get(key), "Cross-arm sealed plan differs: " + key)
    _require(plan["arm"] != peer["arm"] and plan["gpu"] != peer["gpu"] and plan["gpu_uuid"] != peer["gpu_uuid"], "Independent sealed arms/GPUs required")


def run(plan_path, output, gpu, model_root, peer_plan=None, peer_output=None):
    from . import inference
    from .calibration import calibration_identity, calibration_ready
    plan, plan_hash = load_plan(plan_path)
    _require(str(gpu) == plan["gpu"], "CLI GPU differs from sealed arm")
    _require(bool(peer_plan) == bool(peer_output), "Peer plan/output must be supplied together")
    peer = load_plan(peer_plan) if peer_plan else None
    if peer: _validate_peers(plan, peer[0])
    output = Path(output).resolve()
    source = Path(plan["source_run"]).resolve()
    _require(output != source and not source.is_relative_to(output) and not output.is_relative_to(source), "Keep original raw source separate")
    profile = plan["profile"]
    required = {**profile["environment"], **plan.get("launch_environment", {})}
    cleared = set(profile["cleared_environment"]) | set(plan.get("launch_cleared_environment", []))
    _require(all(os.environ.get(k) == v for k, v in required.items()) and all(k not in os.environ for k in cleared), "Actual launch differs from sealed environment")
    _require("torch" not in sys.modules and "vllm" not in sys.modules, "Fresh process before Torch/vLLM required")
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    manifest, rows, ready, assets, ready_path = inference.load_run_inputs(plan["manifest"], plan["ready"], plan["asset_root"])
    lookup = {r["sample_id"]: r for r in rows}
    sources = original._source_bindings(plan)
    _, qualification = inference.qualification_order([lookup[s] for s in plan["sample_ids"]])
    _require(len(qualification) == 2, "Original T2/T3 qualifications required")
    expected = calibration_identity(plan["manifest"], manifest, "qwen", model_root, None)
    _require(calibration_ready(Path(plan.get("calibration_root", source / "calibration")), "qwen", expected, manifest, ready), "Original CPU calibration not ready")
    native, model_path = inference.build_run_identity(plan["manifest"], ready_path, manifest, "qwen", model_root)
    qsa.runtime.activate("qwen", "qwen_qsa_only", profile["baseline_environment"])
    status = qsa.runtime.status()
    qsa._validate_profile(status, "qwen", require_applied=False)
    _require(status["profile_id"] == profile["id"], "Activated profile differs")
    import torch
    output.mkdir(parents=True, exist_ok=True)
    owner = owner_identity()
    _owner_key(owner)
    _publish_claim(output / ".qwen.claim.json", {"plan_sha256": plan_hash, "implementation": _implementation(), "owner": owner})
    stop, previous = _Stop(), {}
    for number in (signal.SIGTERM, signal.SIGINT):
        previous[number] = signal.signal(number, stop.request)
    began, calls, pairs, session = time.monotonic(), 0, [], None
    def stop_reason():
        return _stop_reason(output, peer_output, stop, plan, plan_hash, owner, peer)
    def finish(outcome, **extra):
        result = {"schema": SCHEMA, "plan_sha256": plan_hash, "arm": plan["arm"],
            "status": "COMPLETE" if outcome in ("MATCH", "DIFFERENT") else outcome,
            "comparison_status": outcome, "evidence_status": "COMPLETE" if outcome in ("MATCH", "DIFFERENT") else "INVALID_OR_INCOMPLETE",
            "model_calls": calls, "max_model_calls": 98, "completed_pair_ids": [p["sample_id"] for p in pairs],
            "owner": owner,
            "pairs": pairs, "elapsed_seconds": time.monotonic() - began,
            "claim": "Fixed observed prefix only; matching two calls is not convergence or general determinism", **extra}
        atomic_json(output / "early-stop.json", result)
        return result
    try:
        with inference._gpu_lock(Path(plan.get("gpu_lock_root", source / "outputs")), torch), contextlib.ExitStack() as cleanup:
            _require(_normalized_gpu_uuid(str(getattr(torch.cuda.get_device_properties(0), "uuid", "")))
                     == _normalized_gpu_uuid(plan["gpu_uuid"]), "Physical GPU UUID differs")
            admission = inference.resource_snapshot(torch, manifest["models"]["qwen"]["resources"])
            _require(admission["admitted"], "Original resource admission failed")
            stores = []
            for repeat in range(2):
                identity = {**native, "purpose": SCHEMA, "reproducibility": {"plan_sha256": plan_hash,
                    "profile": profile, "arm": plan["arm"], "repeat": repeat,
                    "sample_ids": plan["sample_ids"], "implementation": plan["implementation"]}}
                store = cleanup.enter_context(RunStore(output / ("repeat-" + str(repeat)), identity, "qwen", "repro", plan["sample_ids"]))
                _require(not store.outputs and not any(e.get("event") == "INFLIGHT" for e in store.events), "Prior attempts cannot resume")
                store.event("REPRO_RESOURCE_ADMISSION", gpu_selection=str(gpu), **admission)
                stores.append(store)
            reason = stop_reason()
            if reason: return finish("STOPPED", reason=reason)
            sink = original._EventSink(stores[0])
            session = inference.ModelSession("qwen", model_path, sink, None, manifest["models"]["qwen"]["resources"])
            cleanup.callback(session.close)
            qsa.runtime.prepare_native("qwen")
            for store in stores:
                sink.store = store
                snapshot = qsa.runtime.runtime_snapshot()
                _snapshot_identity(snapshot, plan)
                store.event("REPRO_RUNTIME", snapshot=snapshot)
            for sid in plan["sample_ids"]:
                for repeat, store in enumerate(stores):
                    reason = stop_reason()
                    if reason: return finish("STOPPED", reason=reason)
                    sink.store = store
                    store.begin(sid, digest({"row": lookup[sid], "ready": ready[sid]}))
                    started = time.monotonic()
                    calls += 1
                    try:
                        def commit(record):
                            return store.commit_output(sid, {**record, "parent_sample_id": sid,
                                "reproducibility": {"plan_sha256": plan_hash, "arm": plan["arm"], "repeat": repeat}})
                        session.generate(lookup[sid], ready[sid], assets, commit)
                        store.event("EARLY_CALL_COMPLETE", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], elapsed_seconds=time.monotonic() - started)
                        # Validate each completed call before allowing another model request.
                        call = _call_evidence(store.directory, plan, plan_hash, repeat, sid, sources)
                    except BaseException as error:
                        if sid in store.outputs:
                            store.event("POST_OUTPUT_VALIDATION_FAILED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started)
                        elif not isinstance(error, PersistenceError):
                            store.fail(sid, error, interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)), elapsed_seconds=time.monotonic() - started)
                        raise
                    finally:
                        session.context.update(active=False, codes=None)
                    if repeat == 1:
                        pair_calls, receipt = _pair(output, plan, plan_hash, sid, sources)
                        if receipt["comparison_status"] == "DIFFERENT":
                            atomic_json(output / "pairs" / (str(len(pairs)).zfill(2) + ".json"), receipt)
                            pairs.append(receipt)
                            return finish("DIFFERENT", first_difference=receipt)
                    if peer:
                        for other in _available_calls(peer_output, peer[0], peer[1], sid, sources):
                            cross = compare_calls(call, other, True)
                            if cross["comparison_status"] == "DIFFERENT":
                                return finish("DIFFERENT", first_difference=cross, comparison_scope="cross_arm_call")
                atomic_json(output / "pairs" / (str(len(pairs)).zfill(2) + ".json"), receipt)
                pairs.append(receipt)
                if peer:
                    index = len(pairs) - 1
                    if (Path(peer_output) / "pairs" / (str(index).zfill(2) + ".json")).exists():
                        peer_calls, _ = _pair(peer_output, peer[0], peer[1], sid, sources)
                        for a in pair_calls:
                            for b in peer_calls:
                                cross = compare_calls(a, b, cross_arm=True)
                                if cross["comparison_status"] == "DIFFERENT": return finish("DIFFERENT", first_difference=cross, comparison_scope="cross_arm")
                if len(pairs) == 8:
                    for store in stores: store.event("EARLY_PREFIX_MATCH", sample_ids=plan["prefix_ids"], model_calls=calls)
                    if peer:
                        deadline = time.monotonic() + plan["comparison_contract"]["peer_barrier_timeout_seconds"]
                        while not (Path(peer_output) / "pairs" / "07.json").exists():
                            reason = stop_reason()
                            if reason: return finish("STOPPED", reason=reason)
                            _require(time.monotonic() < deadline, "Peer prefix barrier timed out")
                            claim = _peer_claim(peer_output, peer)
                            if claim:
                                proc = Path("/proc") / str(claim["owner"]["pid"]) / "stat"
                                _require(proc.exists() and proc.read_text().rsplit(")", 1)[1].split()[19] == claim["owner"]["process_start_ticks"], "Peer process exited/reused before prefix barrier")
                            time.sleep(1)
                        for prefix_sid in plan["prefix_ids"]:
                            left, _ = _pair(output, plan, plan_hash, prefix_sid, sources)
                            right, _ = _pair(peer_output, peer[0], peer[1], prefix_sid, sources)
                            for a in left:
                                for b in right:
                                    cross = compare_calls(a, b, True)
                                    if cross["comparison_status"] == "DIFFERENT": return finish("DIFFERENT", first_difference=cross, comparison_scope="cross_arm_prefix")
                        for store in stores: store.event("EARLY_INDEPENDENT_PREFIX_MATCH", sample_ids=plan["prefix_ids"])
                # Qualification depends on input/format only, never correctness.
                for store in stores:
                    if all(r["sample_id"] in store.outputs for r in qualification) and not (store.directory / "qualification-repro.json").exists():
                        inference._qualification_marker(store, store.identity, "qwen", "repro", qualification)
            return finish("MATCH")
    except BaseException as error:
        finish("INVALID_EVIDENCE", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        for number, handler in previous.items(): signal.signal(number, handler)


def compare(plan_path, output, peer_plan, peer_output):
    plan, plan_hash = load_plan(plan_path)
    peer, peer_hash = load_plan(peer_plan)
    _validate_peers(plan, peer)
    sources = original._source_bindings(plan)
    checked = []
    result = {"schema": SCHEMA, "plan_sha256": plan_hash, "peer_plan_sha256": peer_hash,
              "comparison_status": "INVALID_EVIDENCE", "evidence_status": "INVALID_OR_INCOMPLETE"}
    try:
        for sid in plan["sample_ids"]:
            own = []
            missing = []
            for directory, bound_plan, bound_hash in ((output, plan, plan_hash), (peer_output, peer, peer_hash)):
                try:
                    own.append(_pair(directory, bound_plan, bound_hash, sid, sources))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    missing.append(error)
            # A complete local counterexample stays valid if its peer stopped
            # after one in-flight request. Do not invent a second peer output.
            local_difference = next((receipt for _, receipt in own if receipt["comparison_status"] == "DIFFERENT"), None)
            if local_difference:
                checked.append(sid)
                result.update(comparison_status="DIFFERENT", evidence_status="COMPLETE", first_difference=local_difference)
                break
            available_left = _available_calls(output, plan, plan_hash, sid, sources)
            available_right = _available_calls(peer_output, peer, peer_hash, sid, sources)
            first_cross = next((receipt for a in available_left for b in available_right
                if (receipt := compare_calls(a, b, True))["comparison_status"] == "DIFFERENT"), None)
            if first_cross:
                checked.append(sid)
                result.update(comparison_status="DIFFERENT", evidence_status="COMPLETE", first_difference=first_cross)
                break
            if missing: raise missing[0]
            (left, left_pair), (right, right_pair) = own
            receipts = [left_pair, right_pair] + [compare_calls(a, b, True) for a in left for b in right]
            checked.append(sid)
            different = next((r for r in receipts if r["comparison_status"] == "DIFFERENT"), None)
            if different:
                result.update(comparison_status="DIFFERENT", evidence_status="COMPLETE", first_difference=different)
                break
        else:
            result.update(comparison_status="MATCH", evidence_status="COMPLETE")
    except (OSError, ValueError, KeyError, TypeError) as error:
        result.update(error_type=type(error).__name__, error=str(error))
    result.update(checked_ids=checked, expected_ids=49,
        claim="Only validated complete pair evidence; fixed prefix differences are valid, incomplete prefixes cannot establish matching all 49")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for key in ("source-plan", "plan", "arm", "gpu", "gpu-uuid"): prepare.add_argument("--" + key, required=True)
    prepare.add_argument("--baseline-json")
    prepare.add_argument("--prefix-id", action="append")
    runner = commands.add_parser("run")
    for key in ("plan", "output", "gpu", "model-root"): runner.add_argument("--" + key, required=True)
    runner.add_argument("--peer-plan"); runner.add_argument("--peer-output")
    checker = commands.add_parser("compare")
    for key in ("plan", "output", "peer-plan", "peer-output"): checker.add_argument("--" + key, required=True)
    checker.add_argument("--report")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        source = json.loads(Path(args.source_plan).read_text())
        baseline = json.loads(Path(args.baseline_json).read_text()) if args.baseline_json else source.get("profile", {}).get("baseline_environment")
        result = prepare_plan(source, args.arm, args.gpu, args.gpu_uuid, args.prefix_id, baseline)
        _require(not Path(args.plan).exists(), "A sealed plan cannot be overwritten")
        atomic_json(args.plan, result)
        load_plan(args.plan)
    elif args.command == "run":
        result = run(args.plan, args.output, args.gpu, args.model_root, args.peer_plan, args.peer_output)
    else:
        result = compare(args.plan, args.output, args.peer_plan, args.peer_output)
        if args.report: atomic_json(args.report, result)
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("comparison_status") in (None, "MATCH", "DIFFERENT") else 1


if __name__ == "__main__":
    raise SystemExit(main())
