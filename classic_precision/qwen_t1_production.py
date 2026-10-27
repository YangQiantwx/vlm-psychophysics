"""Finite QSA-only T1 production with qualified native input guards.

Keep the first two actual calls deeply instrumented, then use the unchanged
native adapter's ordinary production checks. Completed predecessor IDs remain
immutable and are excluded by a sealed prefix manifest; no output is copied.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import sys
import time
import uuid

from . import adapter_qsa_replay_ipc as replay
from .storage import RunStore, PersistenceError, atomic_json, digest, file_sha256, owner_identity, _append

SCHEMA = "classic_precision_qwen_t1_production/v2"
EVIDENCE_MODE = "first-two-deep-then-native-production"
require = replay.require
MANIFEST_SHA = "9a4b69b3772ece295958541df67aaee9242dbd7807ee5dc7e7f0069474033b29"


def descriptor(path):
    p = Path(path).resolve()
    return {"path": str(p), "sha256": file_sha256(p), "bytes": p.stat().st_size}


def verified(item):
    require(descriptor(item["path"]) == item, "Frozen file changed: " + item["path"])
    return Path(item["path"])


def scope(rows, gpu):
    require(gpu in ("0", "1"), "One Fourier GPU required")
    chosen = [r for r in rows if (r["phase_index"] % 3 == 2) == (gpu == "1")]
    require(len(chosen) == (2560 if gpu == "0" else 1280), "Full T1 phase split differs")
    return chosen


def context(plan, assets=True):
    ref = replay.native.read(verified(plan["reference_plan"]))
    coverage = replay.validate_plan(ref, physical=assets)
    bound = replay.native.bind_route(ref, coverage, "video", "qwen", assets=assets)
    module = replay.native._native(bound, ref, "qwen")
    queue, ready = module.input_catalog()
    require(file_sha256(bound["root"] / "evaluation_manifest.json") == MANIFEST_SHA
            and len(queue["rows"]) == 3840, "Original full T1 source differs")
    chosen = module.select(scope(queue["rows"], plan["gpu"]), priority=(0, 6))
    initial = module.qualification(chosen)
    full = initial + [r for r in chosen if r not in initial]
    excluded = plan["completed_prefix_ids"]
    require(excluded == [r["sample_id"] for r in full[:len(excluded)]], "Predecessor is not an exact prefix")
    ordered = full[len(excluded):]
    first = ordered[:2]
    require(len(first) == 2, "Two new qualification calls required")
    require(plan["sample_ids"] == [r["sample_id"] for r in ordered], "Frozen remaining order differs")
    bound["rows"] = ordered
    bound["entry"] = deepcopy(bound["entry"])
    # Extend only the read-only source inventory. Each ready descriptor remains
    # tied to the original source manifest and native input_catalog validation.
    bound["entry"]["representative_samples"] = [{"sample_id": r["sample_id"], "source_assets": [{
        "ready_file": descriptor(bound["root"] / "ready" / (r["sample_id"] + ".json")),
        "expected_file_sha256": ready[r["sample_id"]]["video_sha256"]}]} for r in ordered]
    view = deepcopy(ref)
    view.update(schema=SCHEMA, production_plan_sha256=digest(plan),
                sample_ids={**ref["sample_ids"], "video": plan["sample_ids"]})
    return ref, view, bound, module, ready, first


def validate(plan):
    require(plan.get("schema") == SCHEMA and plan.get("passes") == 1
        and plan.get("automatic_retry") is False and plan.get("global_expected") == 3840
        and plan.get("source_sha256") == file_sha256(__file__)
        and plan.get("replay_implementation") == replay.implementation(), "Production scope/source differs")
    require(plan["expected"] == len(plan["sample_ids"]) == len(set(plan["sample_ids"]))
        and plan["expected"] + len(plan["completed_prefix_ids"]) == (2560 if plan["gpu"] == "0" else 1280), "Unique single-pass denominator differs")
    require(plan.get("evidence_mode") == EVIDENCE_MODE, "Undeclared evidence policy")
    require(not set(plan["completed_prefix_ids"]) & set(plan["sample_ids"]), "Duplicate predecessor call")
    for item in plan["predecessor_files"]: verified(item)
    verified(plan["reference_plan"])
    return plan



def admission(plan):
    ref = replay.native.read(verified(plan["reference_plan"]))
    state, report = (replay.native.read(plan[k]) for k in ("qualification_state", "admission_report"))
    require(state.get("status") == "COMPLETE" and state.get("comparison_status") == "MATCH"
        and state.get("plan_sha256") == report.get("plan_sha256") == digest(ref)
        and report.get("comparison_status") == "MATCH" and report.get("evidence_status") == "COMPLETE"
        and report["routes"]["video"]["checked_ids"] == ref["sample_ids"]["video"], "Current T1 qualification is not complete MATCH")
    receipt = state["reports"]["video"]
    require(Path(receipt["path"]).resolve() == Path(plan["admission_report"]).resolve()
        and receipt["sha256"] == file_sha256(plan["admission_report"])
        and len(state["jobs"]) == 2 and all(j.get("returncode") == 0 and j.get("comparison_status") == "MATCH" for j in state["jobs"]),
        "Qualification job/report closure differs")


def run(path):
    require("torch" not in sys.modules and "vllm" not in sys.modules, "Fresh worker required")
    plan = validate(replay.native.read(path)); plan_file = file_sha256(path)
    admission(plan)
    ref = replay.native.read(plan["reference_plan"]["path"]); profile = ref["profiles"]["video"]
    require(all(os.environ.get(k) == v for k, v in profile["environment"].items())
        and all(k not in os.environ for k in profile["cleared_environment"]), "Native environment differs")
    output = Path(plan["output"]); root = output / "cold-0/video"
    require(not root.exists(), "No resume/overwrite")
    os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
    helper = replay.helper
    helper.activate(replay.native.verified(ref["coverage"], ref["path_map"]), profile["baseline_environment"])
    before = {k: os.environ.get(k) for k in helper.THREAD_ENVIRONMENT}
    ref, view, bound, module, ready, first = context(plan)
    require(not output.resolve().is_relative_to(bound["root"].resolve())
        and not bound["root"].resolve().is_relative_to(output.resolve()), "Protect original source/output")
    after = {k: os.environ.get(k) for k in helper.THREAD_ENVIRONMENT}
    imports = {"adapter_sha256": replay.native._adapter_descriptor(bound["entry"])["sha256"],
        "before_import": before, "after_native_import": after, "before_native_processor": after}
    helper.bind_native_import("video", imports)
    owner, engine = owner_identity(), uuid.uuid4().hex
    root.mkdir(parents=True)
    replay._publish_claim(root / "qwen.claim.json", {"plan_sha256": digest(view), "owner": owner, "engine_id": engine,
        "production_plan_sha256": digest(plan)})
    from . import inference as base
    import torch
    proxy, stop, calls = replay.Proxy(), replay._Stop(), 0
    previous = {s: signal.signal(s, stop.request) for s in (signal.SIGINT, signal.SIGTERM)}
    began = time.monotonic()
    def finish(status, **extra):
        result = {"schema": SCHEMA, "status": status, "plan_sha256": digest(plan), "owner": owner,
            "engine_id": engine, "model_calls": calls, "expected": plan["expected"],
            "elapsed_seconds": time.monotonic() - began, **extra}
        atomic_json(root / "result.json", result)
        return result
    try:
        with replay.native.physical_lease(base, torch, ref, bound["spec"]["resources"]) as lease, ExitStack() as cleanup:
            identity = {"schema": SCHEMA, "model_key": "qwen", "model": bound["spec"],
                "checkpoint_assets": ref["model_state"]["qwen"]["video"]["checkpoint_assets"],
                "replay": {"plan_sha256": digest(view), "profile": profile, "route": "video", "cold_pass": 0,
                    "repeat": 0, "sample_ids": plan["sample_ids"], "engine_id": engine, "evidence_mode": EVIDENCE_MODE,
                    "model_spec_source": bound["model_spec_source"], "geometry_calibration": bound["calibration"], "geometry_coverage": bound["geometry"]}}
            store = cleanup.enter_context(RunStore(root / "repeat-0", identity, "qwen", "t1-single", plan["sample_ids"]))
            require(not store.outputs and not any(e.get("event") == "INFLIGHT" for e in store.events), "No duplicate production")
            proxy.current = store
            store.event("REPLAY_PHYSICAL_LEASE", lease=lease); store.event("REPLAY_NATIVE_IMPORT_ENVIRONMENT", receipt=imports)
            guard = module.GuardStore(proxy, base); guard.store = proxy
            session = module.make_session(base, "qwen", base._inside(ref["model_root"], bound["spec"]["checkpoint_relative"]),
                guard, ref["cosmos_package"], bound["spec"]["resources"])
            guard.session = session; cleanup.callback(session.close)
            handle = helper.instrument_session(session, "video", module, replay.native.verified(ref["coverage"], ref["path_map"]))
            cleanup.callback(handle.close)
            snapshot = helper.snapshot(); replay._runtime(snapshot, view, "video", imports)
            store.event("REPLAY_RUNTIME", snapshot=snapshot, engine_id=engine)
            # Keep the exact frozen validator but provide just the immutable
            # initialization prefix and this call's real journal byte ranges.
            source = inspect.getsource(replay._call)
            old = "from .summary import model_snapshot, technical_passed"
            require(source.count(old) == 1, "Frozen evidence reader seam differs")
            source = source.replace(old, "from .summary import technical_passed\n    model_snapshot = _production_snapshot")
            namespace = dict(replay._call.__globals__); local = {}
            exec(compile(source, __file__ + "::bounded_evidence_reader", "exec"), namespace, local)
            read_call = local["_call"]
            from .qwen_t2t3_five_pass import proof_for, _range
            prefix_end = (store.directory / "events.jsonl").stat().st_size
            qualified = set()
            first_ids = [r["sample_id"] for r in first]
            deep_qualification = None
            for row in bound["rows"]:
                sid = row["sample_id"]
                if stop.reason: return finish("STOPPED", reason=stop.reason)
                require(file_sha256(path) == plan_file and file_sha256(__file__) == plan["source_sha256"], "Sealed source/plan changed")
                raw_path = store.directory / "outputs.jsonl"
                raw_start = raw_path.stat().st_size if raw_path.exists() else 0
                event_start = (store.directory / "events.jsonl").stat().st_size
                source_input = {"row": row, "ready": ready[sid]}
                store.begin(sid, digest(source_input)); started = time.monotonic(); calls += 1
                def durable(raw):
                    return store.commit_output(sid, {**raw, "replay_parent_sample_id": sid,
                        "adapter_replay": {"source_input": source_input, "source_input_sha256": digest(source_input),
                            "plan_sha256": digest(view), "route": "video", "cold_pass": 0, "repeat": 0}})
                try:
                    deep = sid in first_ids
                    session.hash_deep = deep
                    if not deep:
                        require(deep_qualification is not None, "Same-worker deep qualification missing")
                        store.event("T1_GEOMETRY_QUALIFICATION_REUSED", sample_id=sid,
                            qualification=deep_qualification, evidence_mode=EVIDENCE_MODE,
                            per_id_gpu_content_hash=False, precision_inversion="reused_geometry_qualification")
                        before_call = helper.snapshot()
                        require(replay._runtime(before_call, view, "video", imports) == replay._runtime(snapshot, view, "video", imports), "Native runtime changed before call")
                    session.generate(row, ready[sid], bound["root"], durable)
                    if not deep:
                        after_call = helper.snapshot()
                        require(replay._runtime(after_call, view, "video", imports) == replay._runtime(snapshot, view, "video", imports), "Native runtime changed after call")
                        store.event("T1_NATIVE_RUNTIME_PASS", sample_id=sid,
                            comparison_identity_sha256=digest(replay._runtime(after_call, view, "video", imports)))
                    store.event("REPLAY_CALL_COMPLETED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], seconds=time.monotonic()-started)
                    proof = proof_for(root, sid, prefix_end, raw_start, event_start)
                    parse = lambda b: [json.loads(l) for l in b.splitlines() if l]
                    raw = parse(_range(raw_path, raw_start, proof["raw_end"]))
                    events = parse(_range(store.directory / "events.jsonl", 0, prefix_end)) + parse(_range(store.directory / "events.jsonl", event_start, proof["event_end"]))
                    def snap(requested):
                        require(Path(requested).resolve() == store.directory.resolve(), "Unexpected evidence path")
                        return json.loads((store.directory / "identity.json").read_text()), raw, events, False, False
                    namespace["_production_snapshot"] = snap
                    if deep:
                        read_call(output, view, bound, module, "video", 0, 0, sid)
                    else:
                        validate_ordinary(raw, events, row, ready[sid], module, base, deep_qualification)
                    _append(root / "calls.jsonl", proof)
                    qualified.add(sid)
                    if sid == first[-1]["sample_id"]:
                        require(all(r["sample_id"] in qualified for r in first), "First two qualification incomplete")
                        deep_qualification = {"sample_ids": first_ids, "owner": owner, "engine_id": engine,
                            "production_plan_sha256": digest(plan), "calls_prefix_sha256": file_sha256(root / "calls.jsonl")}
                        store.event("T1_SINGLE_WORKER_QUALIFIED", sample_ids=first_ids,
                            qualification=deep_qualification, evidence_mode=EVIDENCE_MODE)
                        handle.close()
                except BaseException as error:
                    if sid in store.outputs: store.event("POST_OUTPUT_VALIDATION_FAILED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], error_type=type(error).__name__, error=str(error))
                    elif not isinstance(error, PersistenceError): store.fail(sid, error, elapsed_seconds=time.monotonic()-started)
                    raise
                finally:
                    session.context.update(active=False, codes=None); module.CTX.clear()
            require(calls == plan["expected"] and qualified == set(plan["sample_ids"]), "Incomplete production")
            store.event("T1_SINGLE_COMPLETE", sample_ids=plan["sample_ids"])
            return finish("COMPLETE", evidence_status="COMPLETE")
    except BaseException as error:
        return finish("INVALID_EVIDENCE", error_type=type(error).__name__, error=str(error))
    finally:
        proxy.current = None
        for sig, handler in previous.items(): signal.signal(sig, handler)


def validate_ordinary(raw, events, row, ready, module, base, qualification):
    from .summary import technical_passed
    sid = row["sample_id"]
    require(len(raw) == 1 and raw[0]["sample_id"] == sid and sid in technical_passed(raw, events), "Current durable native PASS missing")
    r = raw[0]
    require(r["attempt_number"] == 1 and r["finish_reason"] == "stop" and r["raw_answer"] in ("LEFT", "RIGHT") and bool(r["output_token_ids"]), "Non-normal result is terminal; no retry")
    require(r["ready_identity"] == digest(ready) and r["row_identity"] == digest({"row": row, "ready": ready}), "Source identity differs")
    attempts = [e for e in events if e.get("event") == "INFLIGHT" and e.get("sample_id") == sid]
    require(len(attempts) == 1 and attempts[0]["attempt_id"] == r["attempt_id"], "Attempt identity differs")
    post = [e for e in events if e.get("event") == "POST_OUTPUT_INPUT_RECEIPT" and e.get("sample_id") == sid]
    require(len(post) == 1 and post[0]["attempt_id"] == r["attempt_id"] and post[0]["t1_deep_qualification"] is False, "Native ordinary post missing")
    module.validate_post(post[0], "qwen", base)
    receipt = r["input_receipt"]
    require(receipt["source_video_sha256"] == ready["video_sha256"] and receipt["source_ready_sha256"] == ready["source_ready_sha256"]
        and receipt["selected_source_indices"] == list(range(180)) and receipt["decoded_dtype"] == "uint16"
        and receipt["actual_processor_tensor_shape"] == [138240,1536]
        and receipt["actual_processor_dtype"] == "torch.float32"
        and receipt["precision_inversion"] == "reused_geometry_qualification", "Ordinary source/geometry differs")
    require(r["actual_input_tokens"] == receipt["input_tokens"] and r["actual_input_tokens"]+16384 <=65536, "Token budget differs")
    reused = [e for e in events if e.get("event") == "T1_GEOMETRY_QUALIFICATION_REUSED" and e.get("sample_id") == sid]
    runtime = [e for e in events if e.get("event") == "T1_NATIVE_RUNTIME_PASS" and e.get("sample_id") == sid]
    require(len(reused) == 1 and reused[0]["qualification"] == qualification and len(runtime) == 1, "Qualified runtime lineage missing")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["run"]); p.add_argument("--plan", required=True)
    args = p.parse_args(); result = run(args.plan)
    print(json.dumps(result, sort_keys=True)); return 0 if result["status"] == "COMPLETE" else 1

if __name__ == "__main__":
    sys.exit(main())
