"""Finite Qwen native-adapter repeats with QSA alone and immediate early stop."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
from types import SimpleNamespace
import uuid

from . import adapter_replay as native
from . import qsa_only_reproducibility as qsa
from .early_stop_reproducibility import _normalized_gpu_uuid, _publish_claim, _Stop
from .storage import RunStore, PersistenceError, atomic_json, digest, file_sha256, owner_identity
from runtime_compat import classic_precision_adapter_qsa as helper

SCHEMA = "classic_precision_native_adapter_qsa_early_stop/v1"
FROZEN_NATIVE_ENTRY_SHA256 = "4719c8b3f18d985d755454d9c1ae405e8145665990a70653609dcac404cc14f6"
FROZEN_EARLY_HELPERS_SHA256 = "8a6e3d0869f53d4474adc5934fcce823b925ffdbd8fa64240b3bd4911b3fcddf"
require = native.require
OUTPUT_FIELDS = {"raw_answer", "output_token_ids", "finish_reason", "stop_reason"}


def implementation():
    from . import early_stop_reproducibility as early
    require(file_sha256(native.__file__) == FROZEN_NATIVE_ENTRY_SHA256, "Frozen native adapter entry changed")
    require(file_sha256(early.__file__) == FROZEN_EARLY_HELPERS_SHA256, "Frozen shared early-stop utilities changed")
    require(file_sha256(qsa.__file__) == helper.FROZEN_QSA_ENTRY_SHA256
        and file_sha256(qsa.runtime.__file__) == helper.FROZEN_QSA_RUNTIME_SHA256, "Reviewed QSA-only sources changed")
    return {"entry_sha256": file_sha256(__file__), "helper_sha256": file_sha256(helper.__file__),
        "native_entry_sha256": FROZEN_NATIVE_ENTRY_SHA256, "native_observer_sha256": helper.FROZEN_OBSERVER_SHA256,
        "early_helpers_sha256": FROZEN_EARLY_HELPERS_SHA256, "qsa_entry_sha256": helper.FROZEN_QSA_ENTRY_SHA256,
        "qsa_runtime_sha256": helper.FROZEN_QSA_RUNTIME_SHA256, "selector_sha256": qsa.runtime.QSA_HELPER_SHA256}


def build_plan(source, baselines, routes, stop_path):
    require(isinstance(routes, list) and routes and len(routes) == len(set(routes)) and set(routes) <= set(native.COUNTS), "Fixed native routes required")
    require(set(baselines) == set(routes), "Explicit native baseline per selected route required")
    require(isinstance(stop_path, str) and Path(stop_path).is_absolute(), "Explicit same-host stop path required")
    fields = ("coverage", "path_map", "model_root", "cosmos_package", "gpu_lock_root", "sample_ids")
    plan = {k: deepcopy(source[k]) for k in fields}
    plan.update(schema=SCHEMA, models=["qwen"], routes=routes, cold_passes=2, repeats=2, stop_path=stop_path,
        source_plan_sha256=digest(source), implementation=implementation(),
        profiles={r: qsa.build_profile("qwen_qsa_only", baselines[r]) for r in routes},
        calibrations={r: {"qwen": deepcopy(source["calibrations"][r]["qwen"])} for r in routes},
        model_state={"qwen": {r: deepcopy(source["model_state"]["qwen"][r]) for r in routes}},
        max_model_calls=sum(native.COUNTS[r] * 4 for r in routes), automatic_retry=False,
        contract={"order": "per-ID repeat0/repeat1, cold1 only after cold0 matches",
            "stop": "first valid complete output difference; invalid controls stop separately; finish in-flight raw durably",
            "cold_control": "same route, baseline, actual inputs/runtime and physical GPU; independent process/engine",
            "native_import": "preserve source-bound native import environment, including static_inputs original OMP/BLAS=1 derivation",
            "scope": "30 original coverage representatives; no general determinism or accuracy claim"})
    for route, profile in plan["profiles"].items():
        baseline = profile["baseline_environment"]
        amount = baseline.get("CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB")
        require((route not in ("single", "pair") or amount == "16") and
                (route not in ("video", "multi") or amount in (None, "24")), "Preserve original route host offload16/24")
        for key in ("OMP_NUM_THREADS", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "VLLM_ENABLE_V1_MULTIPROCESSING", "CUBLAS_WORKSPACE_CONFIG", "VLLM_BATCH_INVARIANT"):
            require(baseline.get(key) is None, "QSA-only must not restore strict startup flag: " + key)
    return plan


def validate_plan(plan, physical=True):
    require(plan.get("schema") == SCHEMA and plan.get("models") == ["qwen"]
        and (plan.get("cold_passes"), plan.get("repeats")) == (2, 2)
        and plan.get("automatic_retry") is False and plan.get("implementation") == implementation(), "QSA adapter finite plan/source differs")
    routes = plan.get("routes", [])
    require(routes and len(set(routes)) == len(routes) and set(routes) <= set(native.COUNTS), "Unknown/duplicate routes")
    require(plan.get("max_model_calls") == sum(native.COUNTS[r] * 4 for r in routes), "Finite adapter call cap differs")
    require(isinstance(plan.get("stop_path"), str) and Path(plan["stop_path"]).is_absolute(), "Finite same-host stop path missing")
    coverage = native.read(native.verified(plan["coverage"], plan["path_map"]))
    helper._coverage(native.verified(plan["coverage"], plan["path_map"]))
    for route in routes:
        profile = plan["profiles"][route]
        require(profile == qsa.build_profile("qwen_qsa_only", profile.get("baseline_environment")), "QSA-only route profile differs")
        baseline = profile["baseline_environment"]
        require(all(baseline.get(k) is None for k in ("OMP_NUM_THREADS", "VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "VLLM_ENABLE_V1_MULTIPROCESSING", "CUBLAS_WORKSPACE_CONFIG", "VLLM_BATCH_INVARIANT")), "Strict startup policy is not QSA-only")
        require(baseline.get("CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB") in (("16",) if route in ("single", "pair") else (None, "24")), "Native route offload differs")
        require(plan["sample_ids"][route] == [r["sample_id"] for r in native._route(coverage, route)["representative_samples"]], "Original coverage IDs changed")
        bound = native.bind_route(plan, coverage, route, "qwen", assets=physical)
        native.native_state_anchor(plan["model_state"]["qwen"][route], bound, plan, "qwen")
        if physical:
            from . import inference
            model_path = inference._inside(native.relocated(plan["model_root"], plan["path_map"]), bound["spec"]["checkpoint_relative"])
            require(inference.checkpoint_asset_identity(model_path) == plan["model_state"]["qwen"][route]["checkpoint_assets"], "Actual checkpoint state changed")
    video_root = native.relocated(native._route(coverage, "video")["manifest"]["path"], plan["path_map"]).parent
    require(native.relocated(plan["gpu_lock_root"], plan["path_map"]) == video_root / "evaluation/outputs", "Original shared physical GPU lock root changed")
    return coverage


def _runtime(snapshot, plan, route, import_receipt):
    policy = helper.validate_snapshot(snapshot, route, import_receipt, native.verified(plan["coverage"], plan["path_map"]))
    expected = plan["profiles"][route]
    require(policy["profile"]["baseline_environment"] == expected["baseline_environment"]
        and policy["profile"]["torch_policy"] == {}, "Actual route native baseline differs")
    return {"policy": policy, "gpu_uuid": _normalized_gpu_uuid(snapshot.get("gpu_uuid")), "host": snapshot.get("hostname")}


class Proxy(native.StoreProxy):
    @property
    def directory(self):
        require(self.current is not None, "No active native repeat store")
        return self.current.directory

    def event(self, name, **fields):
        super().event(name, **fields)
        if name == "REPRO_ENGINE_LAUNCH_PARENT":
            self.current.event("ADAPTER_QSA_PRE_ENGINE_ENVIRONMENT", environment=helper._native_environment(), receipt=fields.get("receipt"))


def _root(output, route, cold):
    return Path(output) / ("cold-" + str(cold)) / route


def _call(output, plan, bound, native_module, route, cold, repeat, sid):
    from .summary import model_snapshot, technical_passed
    root = _root(output, route, cold)
    identity, rows, events, raw_tail, event_tail = model_snapshot(root / ("repeat-" + str(repeat)) / "qwen")
    require(not raw_tail and not event_tail, "Incomplete adapter journal tail")
    claim = native.read(root / "qwen.claim.json")
    replay = identity.get("replay", {})
    require(replay.get("plan_sha256") == claim.get("plan_sha256") == digest(plan)
        and replay.get("profile") == plan["profiles"][route] and replay.get("route") == route
        and replay.get("cold_pass") == cold and replay.get("repeat") == repeat
        and replay.get("sample_ids") == plan["sample_ids"][route] and replay.get("engine_id") == claim.get("engine_id"), "Adapter raw store identity differs")
    state = plan["model_state"]["qwen"][route]
    require(identity.get("model") == bound["spec"] and identity.get("checkpoint_assets") == state["checkpoint_assets"]
        and replay.get("model_spec_source") == bound["model_spec_source"]
        and replay.get("geometry_calibration") == bound["calibration"] and replay.get("geometry_coverage") == bound["geometry"], "Native model/geometry identity differs")
    starts = [e for e in events if e.get("event") == "WORKER_START"]
    require(len(starts) == 1 and all(native._owner(starts[0]["owner"]))
        and native._owner(starts[0]["owner"]) == native._owner(claim["owner"]), "Complete unique native process owner differs")
    raw = [r for r in rows if r.get("sample_id") == sid]
    require(len(raw) == 1 and sid in technical_passed(rows, events), "Complete raw/native technical PASS missing")
    raw = raw[0]
    attempts = [e for e in events if e.get("event") == "INFLIGHT" and e.get("sample_id") == sid]
    require(len(attempts) == 1 and attempts[0].get("attempt_id") == raw.get("attempt_id") and raw.get("attempt_number") == 1, "Implicit retry or unmatched native attempt")
    row = next(r for r in bound["rows"] if r["sample_id"] == sid)
    require(all(raw.get("adapter_replay", {}).get(k) == v for k, v in
        {"plan_sha256": digest(plan), "route": route, "cold_pass": cold, "repeat": repeat}.items()), "Raw plan/repeat binding differs")
    binding = {"profile_id": helper.PROFILE_ID, "family": "qwen", "route": route,
        "coverage_sha256": plan["coverage"]["sha256"], "adapter_sha256": native._adapter_descriptor(bound["entry"])["sha256"],
        "instrumentation_sha256": plan["implementation"]["helper_sha256"], "row_sha256": digest(row)}
    hashes = native.tensor_receipts(events, sid, binding)
    value = native._raw_fingerprint(raw, row, events, hashes, native._ready(bound, row))
    native._validate_native_post(native_module, "qwen", route, raw, value["post"], bound)
    receipt = raw["input_receipt"]
    shape = receipt.get("actual_expected_GPU_shape", receipt.get("actual_processor_tensor_shape"))
    require(shape in bound["geometry"]["shapes"], "Native input geometry is not CPU-qualified")
    if route != "video":
        pixels = [t for t in hashes["gpu_forward"] if t["name"] == "pixels"]
        require(len(pixels) == 1 and pixels[0]["shape"] == shape and pixels[0]["sha256"] == receipt.get("expected_BF16_GPU_sha256"), "Actual GPU pixels differ from native deep guard")
    runtime = [e for e in events if e.get("event") == "REPLAY_RUNTIME"]
    require(len(runtime) == 1 and runtime[0].get("engine_id") == claim["engine_id"], "Actual initial engine runtime missing")
    imports = [e["receipt"] for e in events if e.get("event") == "REPLAY_NATIVE_IMPORT_ENVIRONMENT"]
    require(len(imports) == 1 and imports[0].get("adapter_sha256") == native._adapter_descriptor(bound["entry"])["sha256"], "Native import/environment restoration receipt missing")
    baseline = _runtime(runtime[0]["snapshot"], plan, route, imports[0])
    _, _, launch_events, _, _ = model_snapshot(_root(output, route, cold) / "repeat-0/qwen")
    pre_engine = [e for e in launch_events if e.get("event") == "ADAPTER_QSA_PRE_ENGINE_ENVIRONMENT"]
    require(len(pre_engine) == 1 and pre_engine[0].get("environment") == imports[0]["before_native_processor"]
        and pre_engine[0].get("receipt") == runtime[0]["snapshot"]["comparison_identity"].get("engine_launch_parent"), "Actual native pre-engine environment differs")
    require(runtime[0]["snapshot"].get("pid") == claim["owner"]["pid"]
        and baseline["host"] == claim["owner"]["host"], "Actual runtime parent owner differs")
    calls = [e for e in events if e.get("event") == "ADAPTER_QSA_CALL_RUNTIME" and e.get("sample_id") == sid]
    require([e.get("stage") for e in calls] == ["before", "after"]
        and all(_runtime(e["snapshot"], plan, route, imports[0]) == baseline for e in calls), "Actual input-call runtime changed")
    leases = [e["lease"] for e in events if e.get("event") == "REPLAY_PHYSICAL_LEASE"]
    require(len(leases) == 1 and _normalized_gpu_uuid(leases[0]["device"]) == baseline["gpu_uuid"]
        and native._owner(leases[0]["owner"]) == native._owner(claim["owner"]), "Actual physical GPU lease/owner differs")
    lease = leases[0]
    expected_root = native.relocated(plan["gpu_lock_root"], plan["path_map"], claim["owner"]["host"])
    expected_lock = expected_root / (".gpu-" + digest({"host": claim["owner"]["host"], "device": lease["device"]}) + ".lock")
    require(lease.get("root") == str(expected_root) and lease.get("path") == str(expected_lock)
        and lease.get("st_dev") is not None and lease.get("st_ino")
        and lease["resource_snapshot"].get("admitted") is True
        and lease["resource_snapshot"].get("budgets") == bound["spec"]["resources"], "Original physical lock/resource admission differs")
    admission = lease["resource_snapshot"]
    require(admission.get("free_gpu_gib", -1) >= admission["budgets"]["min_free_gpu_gib"]
        and admission.get("available_ram_gib", -1) >= admission["budgets"]["min_available_ram_gib"], "Actual resource budgets were not met")
    completed = [e for e in events if e.get("event") == "REPLAY_CALL_COMPLETED" and e.get("sample_id") == sid]
    require(len(completed) == 1 and completed[0].get("attempt_id") == raw["attempt_id"]
        and isinstance(completed[0].get("seconds"), (float, int)) and completed[0]["seconds"] >= 0, "Actual complete request elapsed missing")
    return {"sample_id": sid, "value": value, "runtime": baseline, "owner": claim["owner"],
        "engine_id": claim["engine_id"], "native_import_environment": imports[0], "raw_sha256": digest(raw), "raw": raw}


def compare_calls(left, right, cross_cold=False):
    require(left["sample_id"] == right["sample_id"] and left["runtime"] == right["runtime"], "Actual sample/runtime control differs")
    require(left["native_import_environment"] == right["native_import_environment"], "Native import/environment controls differ")
    independent = native._owner(left["owner"]) != native._owner(right["owner"])
    require(independent is cross_cold and ((left["engine_id"] != right["engine_id"]) is cross_cold), "Same-engine/independent cold ownership differs")
    controls = lambda c: {**c["value"], "raw": {k: v for k, v in c["value"]["raw"].items() if k not in OUTPUT_FIELDS}}
    require(controls(left) == controls(right), "Actual CPU/GPU input, prompt, native post or generation differs")
    different = [k for k in sorted(OUTPUT_FIELDS) if left["value"]["raw"].get(k) != right["value"]["raw"].get(k)]
    return {"comparison_status": "DIFFERENT" if different else "MATCH", "evidence_status": "COMPLETE",
        "sample_id": left["sample_id"], "different_fields": different,
        "left_raw_sha256": left["raw_sha256"], "right_raw_sha256": right["raw_sha256"]}


def _shared_stop(plan):
    path = native.relocated(plan["stop_path"], plan["path_map"])
    if not path.exists(): return None
    result = native.read(path)
    require(result.get("schema") == SCHEMA and result.get("plan_sha256") == digest(plan)
        and result.get("route") in plan["routes"] and result.get("cold_pass") in (0, 1)
        and result.get("comparison_status") in ("DIFFERENT", "INVALID_EVIDENCE", "STOPPED")
        and all(native._owner(result.get("owner", {}))) and result["owner"]["host"] == socket.gethostname(), "Shared native stop identity differs")
    claim_path = Path(result.get("claim_path", ""))
    require(claim_path.is_absolute(), "Shared stop claim path missing")
    claim = native.read(claim_path)
    require(claim.get("plan_sha256") == digest(plan) and claim.get("engine_id") == result.get("engine_id")
        and native._owner(claim.get("owner", {})) == native._owner(result["owner"]), "Shared stop claim/owner differs")
    return result


def run(plan_path, output, route, cold_pass, gpu):
    require("torch" not in sys.modules and "vllm" not in sys.modules, "Native adapter requires a fresh process")
    plan = native.read(plan_path)
    coverage = validate_plan(plan)
    require(route in plan["routes"] and cold_pass in (0, 1) and str(gpu) and "," not in str(gpu), "Finite selected route/cold/GPU required")
    bound = native.bind_route(plan, coverage, route, "qwen")
    ready = {r["sample_id"]: native._ready(bound, r) for r in bound["rows"]}
    profile = plan["profiles"][route]
    require(all(os.environ.get(k) == v for k, v in profile["environment"].items())
        and all(k not in os.environ for k in profile["cleared_environment"]), "Actual native baseline/launch differs")
    output = Path(output).resolve()
    source_root = bound["root"].resolve()
    require(not output.is_relative_to(source_root) and not source_root.is_relative_to(output), "Keep original native source/raw tree separate")
    require(_shared_stop(plan) is None, "Earlier route already stopped this finite plan")
    if cold_pass == 1:
        previous = native.read(_root(output, route, 0) / "qwen.json")
        require(previous.get("plan_sha256") == digest(plan) and previous.get("comparison_status") == "MATCH"
            and previous.get("model_calls") == 2 * native.COUNTS[route], "Cold0 must completely match before independent cold1")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    root = _root(output, route, cold_pass)
    root.mkdir(parents=True, exist_ok=True)
    owner, engine = owner_identity(), uuid.uuid4().hex
    _publish_claim(root / "qwen.claim.json", {"plan_sha256": digest(plan), "owner": owner, "engine_id": engine})
    helper.activate(native.verified(plan["coverage"], plan["path_map"]), profile["baseline_environment"])
    from . import inference as base
    import_keys = helper.THREAD_ENVIRONMENT
    before_import = {key: os.environ.get(key) for key in import_keys}
    module = native._native(bound, plan, "qwen")
    after_import = {key: os.environ.get(key) for key in import_keys}
    import_receipt = {"adapter_sha256": native._adapter_descriptor(bound["entry"])["sha256"],
        "before_import": before_import, "after_native_import": after_import,
        "before_native_processor": {key: os.environ.get(key) for key in import_keys}}
    if route == "single":
        import_receipt["static_inputs_sha256"] = next(r["sha256"] for r in bound["entry"]["implementation_files"] if Path(r["path"]).name == "static_inputs.py")
    helper.bind_native_import(route, import_receipt)
    import torch
    proxy, calls, matched, stop = Proxy(), 0, [], _Stop()
    began = time.monotonic()
    previous_handlers = {n: signal.signal(n, stop.request) for n in (signal.SIGINT, signal.SIGTERM)}
    def finish(status, **extra):
        result = {"schema": SCHEMA, "plan_sha256": digest(plan), "status": "COMPLETE" if status in ("MATCH", "DIFFERENT") else status,
            "comparison_status": status, "evidence_status": "COMPLETE" if status in ("MATCH", "DIFFERENT") else "INVALID_OR_INCOMPLETE",
            "route": route, "model": "qwen", "cold_pass": cold_pass, "owner": owner, "engine_id": engine, "claim_path": str(root / "qwen.claim.json"),
            "model_calls": calls, "max_model_calls": native.COUNTS[route] * 2,
            "completed_pair_ids": matched, "elapsed_seconds": time.monotonic() - began, **extra}
        atomic_json(root / "qwen.json", result)
        if status in ("DIFFERENT", "INVALID_EVIDENCE", "STOPPED"):
            try: _publish_claim(native.relocated(plan["stop_path"], plan["path_map"]), result)
            except FileExistsError: _shared_stop(plan)  # retain the first complete counterexample/stop
        return result
    try:
        resources = bound["spec"]["resources"]
        with native.physical_lease(base, torch, plan, resources) as lease, ExitStack() as cleanup:
            stores = []
            for repeat in range(2):
                identity = {"schema": SCHEMA, "model_key": "qwen", "model": bound["spec"],
                    "checkpoint_assets": plan["model_state"]["qwen"][route]["checkpoint_assets"],
                    "replay": {"plan_sha256": digest(plan), "profile": profile, "route": route, "cold_pass": cold_pass,
                        "repeat": repeat, "sample_ids": plan["sample_ids"][route], "engine_id": engine,
                        "model_spec_source": bound["model_spec_source"], "geometry_calibration": bound["calibration"], "geometry_coverage": bound["geometry"]}}
                store = cleanup.enter_context(RunStore(root / ("repeat-" + str(repeat)), identity, "qwen", "qsa-adapter", plan["sample_ids"][route]))
                require(not store.outputs and not any(e.get("event") == "INFLIGHT" for e in store.events), "Native adapter cannot resume/retry")
                store.event("REPLAY_PHYSICAL_LEASE", lease=lease)
                store.event("REPLAY_NATIVE_IMPORT_ENVIRONMENT", receipt=import_receipt)
                stores.append(store)
            proxy.current = stores[0]
            args = SimpleNamespace(model="qwen", models=str(native.relocated(plan["model_root"], plan["path_map"])), cosmos_package=str(native.relocated(plan["cosmos_package"], plan["path_map"])))
            if stop.reason: return finish("STOPPED", reason=stop.reason)
            if route == "video":
                guard = module.GuardStore(proxy, base)
                guard.store = proxy  # observational directory bridge, native event/validator unchanged
                session = module.make_session(base, "qwen", base._inside(args.models, bound["spec"]["checkpoint_relative"]), guard, args.cosmos_package, resources)
                guard.session = session
                cleanup.callback(session.close)
            elif route == "single":
                session = sys.modules["static_worker"].StaticSession("qwen", event_sink=proxy)
                cleanup.callback(session.close)
                session.load(resources)
            else:
                session = module.make_session(base, args, proxy, bound["spec"])
                cleanup.callback(session.close)
                session.load(resources)
            require(native.processor_identity(base._source_identity(type(session.processor))) == native.processor_identity(plan["model_state"]["qwen"][route]["processor_source"]), "Actual processor source differs")
            handle = helper.instrument_session(session, route, module, native.verified(plan["coverage"], plan["path_map"]))
            cleanup.callback(handle.close)
            for store in stores:
                proxy.current = store
                snapshot = helper.snapshot()
                _runtime(snapshot, plan, route, import_receipt)
                store.event("REPLAY_RUNTIME", snapshot=snapshot, engine_id=engine)
            for row in bound["rows"]:
                sid = row["sample_id"]
                pair = []
                for repeat, store in enumerate(stores):
                    if stop.reason: return finish("STOPPED", reason=stop.reason)
                    peer_stop = _shared_stop(plan)
                    if peer_stop: return finish("STOPPED", reason="same-host peer " + peer_stop["comparison_status"])
                    proxy.current = store
                    source_input = {"row": row, "ready": ready[sid]}
                    store.begin(sid, digest(source_input))
                    started = time.monotonic()
                    calls += 1
                    def durable(raw):
                        return store.commit_output(sid, {**raw, "replay_parent_sample_id": sid,
                            "adapter_replay": {"source_input": source_input, "source_input_sha256": digest(source_input),
                                "plan_sha256": digest(plan), "route": route, "cold_pass": cold_pass, "repeat": repeat}})
                    try:
                        if route == "video":
                            session.hash_deep = True
                            session.generate(row, ready[sid], bound["root"], durable)
                        elif route == "single": session.generate_image(row, ready[sid], durable, deep=True)
                        else:
                            arrays, observed = module.read_images(row) if route == "multi" else module.read_pair(bound["source"][sid])
                            require(observed == ready[sid], "Original ordered native ready differs")
                            method = session.generate_images if route == "multi" else session.generate_pair
                            method(row, arrays, ready[sid], durable, deep=True)
                        store.event("REPLAY_CALL_COMPLETED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], seconds=time.monotonic() - started)
                        call = _call(output, plan, bound, module, route, cold_pass, repeat, sid)
                    except BaseException as error:
                        if sid in store.outputs: store.event("POST_OUTPUT_VALIDATION_FAILED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], error_type=type(error).__name__, error=str(error))
                        elif not isinstance(error, PersistenceError): store.fail(sid, error, interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)), elapsed_seconds=time.monotonic() - started)
                        raise
                    finally:
                        if hasattr(session, "context"): session.context.update(active=False)
                    pair.append(call)
                    if repeat == 1:
                        receipt = compare_calls(*pair)
                        if receipt["comparison_status"] == "DIFFERENT": return finish("DIFFERENT", first_difference=receipt)
                    if cold_pass == 1:
                        for reference_repeat in range(2):
                            reference = _call(output, plan, bound, module, route, 0, reference_repeat, sid)
                            receipt = compare_calls(reference, call, True)
                            if receipt["comparison_status"] == "DIFFERENT": return finish("DIFFERENT", first_difference=receipt)
                    peer_stop = _shared_stop(plan)
                    if peer_stop: return finish("STOPPED", reason="same-host peer " + peer_stop["comparison_status"])
                matched.append(sid)
            for store in stores: store.event("REPLAY_REPEAT_COMPLETE", sample_ids=plan["sample_ids"][route])
            return finish("MATCH")
    except BaseException as error:
        finish("INVALID_EVIDENCE", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        proxy.current = None
        for number, previous in previous_handlers.items(): signal.signal(number, previous)


def compare(plan_path, output, route=None, physical=True):
    plan = native.read(plan_path)
    coverage = validate_plan(plan, physical=physical)
    routes = [route] if route else plan["routes"]
    result = {"schema": SCHEMA, "plan_sha256": digest(plan), "comparison_status": "MATCH", "evidence_status": "COMPLETE", "routes": {}}
    for selected in routes:
        try:
            bound = native.bind_route(plan, coverage, selected, "qwen", assets=physical)
            module = native._native(bound, plan, "qwen")
            receipts = []
            for sid in plan["sample_ids"][selected]:
                observations, errors = [], []
                for cold in range(2):
                    pair = []
                    for repeat in range(2):
                        try: pair.append(_call(output, plan, bound, module, selected, cold, repeat, sid))
                        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error: errors.append(error)
                    if len(pair) == 2:
                        local = compare_calls(*pair)
                        if local["comparison_status"] == "DIFFERENT":
                            result.update(comparison_status="DIFFERENT", first_difference=local)
                            return result
                    observations.append(pair)
                for left in observations[0]:
                    for right in observations[1]:
                        receipt = compare_calls(left, right, True)
                        if receipt["comparison_status"] == "DIFFERENT":
                            result.update(comparison_status="DIFFERENT", first_difference=receipt)
                            return result
                if errors: raise errors[0]
                receipts.append(sid)
            result["routes"][selected] = {"comparison_status": "MATCH", "checked_ids": receipts}
        except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
            result.update(comparison_status="INVALID_EVIDENCE", evidence_status="INVALID_OR_INCOMPLETE", error_type=type(error).__name__, error=str(error))
            return result
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    for name in ("source-plan", "baselines", "plan", "stop-path"): prepare.add_argument("--" + name, required=True)
    prepare.add_argument("--route", action="append", required=True, choices=tuple(native.COUNTS))
    execute = commands.add_parser("run")
    for name in ("plan", "output", "route", "gpu"): execute.add_argument("--" + name, required=True)
    execute.add_argument("--cold-pass", required=True, type=int, choices=(0, 1))
    check = commands.add_parser("compare")
    for name in ("plan", "output"): check.add_argument("--" + name, required=True)
    check.add_argument("--route", choices=tuple(native.COUNTS)); check.add_argument("--report")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        require(not Path(args.plan).exists(), "Sealed plans cannot be overwritten")
        result = build_plan(native.read(args.source_plan), native.read(args.baselines), args.route, args.stop_path)
        atomic_json(args.plan, result)
    elif args.command == "run": result = run(args.plan, args.output, args.route, args.cold_pass, args.gpu)
    else:
        result = compare(args.plan, args.output, args.route)
        if args.report:
            require(not Path(args.report).exists(), "Comparison receipts are immutable")
            atomic_json(args.report, result)
    print(json.dumps(result, sort_keys=True))
    return int(result.get("comparison_status") in ("INVALID_EVIDENCE", "STOPPED"))


if __name__ == "__main__":
    raise SystemExit(main())
