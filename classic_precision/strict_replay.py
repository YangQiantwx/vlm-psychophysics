"""One strict, resumable pass over the original 3504-row T2/T3 panel.

The frozen native runner owns input/calibration/resource checks, first T2/T3
qualification, terminal raw and the two-attempt budget. This wrapper adds the
already-validated strict core/QSA, attempt-bound CPU/GPU hashes, runtime receipts
and a separate run identity. It never loads an old answer to choose a request.
"""
from __future__ import annotations

import argparse
import functools
import importlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

from . import inference, reference_replay as reference, reproducibility as repro
from .storage import RunStore, atomic_json, digest, file_sha256
from .summary import model_snapshot, technical_passed, attempt_metrics


SCHEMA = "classic_precision_strict_panel_replay/v1"
MODELS = ("nemo", "qwen")
NATIVE_ENVIRONMENT = ("CLASSIC_PRECISION_KV_COMPAT", "CLASSIC_PRECISION_NEMO_DUMMY_CONFIG_COMPAT",
                      "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB", "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA",
                      "VLLM_ALLOW_INSECURE_SERIALIZATION")
require = repro._require
POLICY = {"complete_raw_is_terminal": True, "max_attempts": 2, "implicit_retry": False,
          "qualification": "original first T2 and T3; technical PASS plus stop/exact LEFT or RIGHT; correctness is not a gate"}


def _read(path):
    return json.loads(Path(path).read_text())


def _descriptor(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path)}


def _verified(item):
    path = Path(item["path"])
    require(path.is_absolute() and path.is_file() and file_sha256(path) == item["sha256"],
            "Strict source SHA/path differs: " + str(path))
    return path


def _implementation():
    contract = repro._profile_contract()
    return {"wrapper_sha256": file_sha256(__file__), "reference_sha256": file_sha256(reference.__file__),
            "reproducibility_sha256": file_sha256(repro.__file__), "inference_sha256": file_sha256(inference.__file__),
            "storage_sha256": file_sha256(Path(__file__).with_name("storage.py")),
            "determinism_helper_sha256": contract["helper_sha256"], "qsa_helper_sha256": contract["qsa_helper_sha256"]}


def _inputs(plan):
    from .calibration import calibration_identity, calibration_ready
    manifest_path, ready_path = _verified(plan["manifest"]), _verified(plan["ready"])
    require(file_sha256(manifest_path) == reference.MANIFEST_SHA256,
            "Only the original frozen 3504-row panel is supported; T1/576 extensions use their own adapter")
    manifest, rows, ready, assets, _ = inference.load_run_inputs(manifest_path, ready_path, plan["asset_root"])
    ids = plan["sample_ids"]
    selected = inference.select_rows(rows, ids)
    ordered, first = inference.qualification_order(selected)
    require(ids == [r["sample_id"] for r in selected] and len(first) == 2,
            "Immutable IDs must follow original manifest order and include T2 and T3")
    require(set(ids) <= ready.keys(), "Original selected input ready records are incomplete")
    require(plan["input_binding_sha256"] == digest({r["sample_id"]: {"row": r, "ready": ready[r["sample_id"]]} for r in selected}),
            "Original scientific row/ready identity changed")
    model = plan["model"]
    identity, _ = inference.build_run_identity(manifest_path, ready_path, manifest, model, plan["model_root"])
    require(identity == plan["native_identity"], "Original checkpoint/processor/calibration/library identity changed")
    expected = calibration_identity(manifest_path, manifest, model, plan["model_root"])
    require(calibration_ready(Path(plan["calibration_root"]), model, expected, manifest, ready),
            "Original 128-state CPU calibration is not ready")
    return manifest, ordered, ready, assets, first


def prepare(manifest_path, ready_path, asset_root, model_root, calibration_root, model, output, sample_ids=None):
    require(model in MODELS, "Only Nemo/Qwen T2/T3 replay is authorized")
    manifest_path = Path(manifest_path).resolve()
    require(file_sha256(manifest_path) == reference.MANIFEST_SHA256, "Frozen original 3504 manifest required")
    manifest, rows, ready, assets, actual_ready = inference.load_run_inputs(manifest_path, ready_path, asset_root)
    selected = inference.select_rows(rows, sample_ids)
    require(len(inference.qualification_order(selected)[1]) == 2, "A first T2 and first T3 are required")
    require(all(r["sample_id"] in ready for r in selected), "Selected original inputs are incomplete")
    source, output = manifest_path.parent, Path(output).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            "Use an independent new output outside the original source run")
    identity, _ = inference.build_run_identity(manifest_path, actual_ready, manifest, model, model_root)
    contract = repro._profile_contract()
    plan = {"schema": SCHEMA, "model": model, "manifest": _descriptor(manifest_path), "ready": _descriptor(actual_ready),
            "asset_root": str(assets.resolve()), "model_root": str(Path(model_root).resolve()),
            "calibration_root": str(Path(calibration_root).resolve()), "output": str(output),
            "gpu_lock_root": str(source.parents[1] / "runs/t1-rgb16/evaluation/outputs"),
            "sample_ids": [r["sample_id"] for r in selected], "native_identity": identity,
            "input_binding_sha256": digest({r["sample_id"]: {"row": r, "ready": ready[r["sample_id"]]} for r in selected}),
            "implementation": _implementation(), "profile_id": contract["PROFILE_ID"],
            "environment": {**contract["ENVIRONMENT"], **({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if model == "qwen" else {})},
            "native_environment": {k: os.environ.get(k) for k in NATIVE_ENVIRONMENT},
            "policy": dict(POLICY)}
    _inputs(plan)
    return plan


def load_plan(path):
    plan = _read(path)
    require(plan.get("schema") == SCHEMA and plan.get("model") in MODELS, "Wrong strict panel schema/model")
    require(plan.get("implementation") == _implementation(), "Strict wrapper/core/reference source drift")
    require(plan.get("policy") == POLICY and set(plan.get("native_environment", {})) == set(NATIVE_ENVIRONMENT),
            "Strict terminal/retry/native environment contract differs")
    contract = repro._profile_contract()
    require(plan.get("profile_id") == contract["PROFILE_ID"] and plan.get("environment") ==
            {**contract["ENVIRONMENT"], **({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if plan["model"] == "qwen" else {})},
            "Strict runtime profile/environment differs")
    source, output = Path(plan["manifest"]["path"]).parent.resolve(), Path(plan["output"]).resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output), "Output overlaps original source run")
    require(Path(plan["gpu_lock_root"]).resolve() == source.parents[1] / "runs/t1-rgb16/evaluation/outputs",
            "Use the original shared physical GPU lease root")
    values = _inputs(plan)
    return plan, digest(plan), values


def _store_identity(plan):
    return {**plan["native_identity"], "purpose": SCHEMA, "strict_replay": {
        "plan_sha256": digest(plan), "wrapper_sha256": file_sha256(__file__),
        "profile_id": plan["profile_id"], "sample_ids": plan["sample_ids"]}}


def _runtimes(events, plan):
    found, baseline = {}, None
    for event in events:
        if event.get("event") != "STRICT_RUNTIME":
            continue
        require(event.get("plan_sha256") == digest(plan), "Runtime belongs to another strict plan")
        actual = repro.validate_runtime_policy(event["snapshot"], plan["model"])
        require(actual == event.get("comparison_identity"), "Stored strict runtime identity differs")
        require(baseline is None or baseline == actual, "Strict runtime differs between resumed workers")
        baseline = actual
        require(event["session_id"] not in found, "Duplicate runtime session identity")
        found[event["session_id"]] = actual
    return found


def _audit_raw(raw, events, plan, rows, ready, runtimes):
    sid = raw["sample_id"]
    require(sid in plan["sample_ids"], "Output contains an ID outside the sealed selection")
    binding = raw.get("strict_replay", {})
    require(binding.get("plan_sha256") == digest(plan) and binding.get("session_id") in runtimes,
            "Raw has no corresponding strict runtime/plan")
    reference._bind_raw(raw, rows[sid], ready[sid])
    require(isinstance(raw.get("raw_answer"), str) and isinstance(raw.get("output_token_ids"), list)
            and all(type(x) is int and x >= 0 for x in raw["output_token_ids"])
            and raw.get("finish_reason") in ("stop", "length", "abort", "error"), "Incomplete original raw/token terminal record")
    require(any(e.get("event") == "INFLIGHT" and e.get("sample_id") == sid
                and e.get("attempt_id") == raw.get("attempt_id") and e.get("row_identity") == raw["row_identity"]
                and e.get("attempt_number") == raw.get("attempt_number") in (1, 2) for e in events),
            "Raw lacks its original finite native attempt/input identity")
    require(sid in technical_passed([raw], events), "Complete raw lacks native technical PASS; never regenerate it")
    receipts = [e for e in events if e.get("event") == "REPRO_INPUT_HASH" and e.get("sample_id") == sid
                and e.get("attempt_id") in (None, raw["attempt_id"])]
    require(bool(receipts) and all(e.get("attempt_id") == raw["attempt_id"]
            and e.get("strict_session_id") == binding["session_id"] for e in receipts),
            "Actual input hashes are not bound to this attempt/session")
    hashes = repro._hash_receipts(receipts, sid, raw["attempt_id"], plan["model"])
    return hashes


def verify(plan_path, *, require_complete=True):
    plan, plan_hash, (_, ordered, ready, _, first) = load_plan(plan_path)
    directory = Path(plan["output"]) / plan["model"]
    identity, raw, events, raw_tail, event_tail = model_snapshot(directory)
    require(not raw_tail and not event_tail and identity == {**_store_identity(plan),
            "storage_all_row_ids_sha256": plan["native_identity"]["all_row_ids_sha256"]},
            "Strict journal identity/tail is incomplete")
    require(len(raw) == len({r["sample_id"] for r in raw}), "Duplicate complete raw ID")
    rows, runtimes = {r["sample_id"]: r for r in ordered}, _runtimes(events, plan)
    by_id = {}
    for event in events:
        by_id.setdefault(event.get("sample_id"), []).append(event)
    for record in raw:
        _audit_raw(record, by_id.get(record["sample_id"], []), plan, rows, ready, runtimes)
    complete = len(raw) == len(plan["sample_ids"])
    require(not require_complete or complete, "Strict panel is incomplete")
    first_ids = [r["sample_id"] for r in first]
    qualified = all(any(r["sample_id"] == sid and r["finish_reason"] == "stop" and r["raw_answer"] in ("LEFT", "RIGHT") for r in raw) for sid in first_ids)
    require(not complete or qualified, "Original first T2/T3 qualifications did not pass")
    return {"schema": SCHEMA, "status": "COMPLETE" if complete else "INCOMPLETE", "model": plan["model"],
            "plan_sha256": plan_hash, "expected": len(plan["sample_ids"]), "raw": len(raw),
            "normal": sum(r["finish_reason"] == "stop" and r["raw_answer"] in ("LEFT", "RIGHT") for r in raw),
            "qualification_ids": first_ids, "qualified": qualified, "attempts": attempt_metrics(raw, events)}


def run(plan_path, gpu, *, retry_failed=False, retry_interrupted=False):
    plan, plan_hash, (_, ordered, ready, assets, _) = load_plan(plan_path)
    require("torch" not in sys.modules and "vllm" not in sys.modules, "Start strict replay in a fresh process")
    require(str(gpu) and "," not in str(gpu), "One explicit GPU required")
    require(all(os.environ.get(k) == v for k, v in {**plan["environment"], **plan["native_environment"]}.items()),
            "Launch the new process with the sealed strict/native environment")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    runtime = importlib.import_module("classic_precision_determinism")
    runtime.activate(plan["model"])
    session_id, rows = uuid.uuid4().hex, {r["sample_id"]: r for r in ordered}
    originals = {name: getattr(inference, name) for name in ("build_run_identity", "RunStore")}
    original_init, original_generate = inference.ModelSession.__init__, inference.ModelSession.generate

    def identity(*args, **kwargs):
        native, path = originals["build_run_identity"](*args, **kwargs)
        require(native == plan["native_identity"], "Native run identity changed")
        return _store_identity(plan), path

    class StrictStore(RunStore):
        def __enter__(self):
            super().__enter__()
            try:
                history = _runtimes(self.events, plan)
                for raw in self.outputs.values():
                    _audit_raw(raw, self.events, plan, rows, ready, history)
                for event in self.events:
                    if event.get("event") == "INFLIGHT" and event.get("sample_id") not in self.outputs:
                        sid = event["sample_id"]
                        latest = next((e["event"] for e in reversed(self.events) if e.get("sample_id") == sid and e["event"] in ("INFLIGHT", "FAILED", "INTERRUPTED")), "INFLIGHT")
                        require(retry_failed if latest == "FAILED" else retry_interrupted,
                                "No-output attempt requires explicit retry flag: " + sid)
                return self
            except BaseException:
                self.close()
                raise

        def event(self, name, **fields):
            if name == "REPRO_INPUT_HASH":
                sid = fields["sample_id"]
                attempt = self._active.get(sid) or self.outputs.get(sid, {}).get("attempt_id")
                require(attempt, "Actual input hash has no native active/committed attempt")
                fields.update(attempt_id=attempt, strict_session_id=session_id)
            fields.setdefault("time_unix", time.time())
            super().event(name, **fields)
            self.events.append({"event": name, "shard_id": self.shard_id, **fields})

        def commit_output(self, sid, payload):
            return super().commit_output(sid, {**payload, "strict_replay": {"plan_sha256": plan_hash, "session_id": session_id}})

    @functools.wraps(original_init)
    def initialize(session, family, *args, **kwargs):
        require(family == plan["model"], "Wrong native model family")
        original_init(session, family, *args, **kwargs)
        try:
            repro._prepare_runtime_modules(family)
            snapshot = runtime.runtime_snapshot()
            actual = repro.validate_runtime_policy(snapshot, family)
            previous = _runtimes(session.store.events, plan)
            require(not previous or actual == next(iter(previous.values())), "Runtime changed since the prior strict worker")
            session.store.event("STRICT_RUNTIME", plan_sha256=plan_hash, session_id=session_id,
                                snapshot=snapshot, comparison_identity=actual)
        except BaseException:
            session.close()
            raise

    @functools.wraps(original_generate)
    def generate(session, row, observed_ready, asset_root, durable_output):
        require(row["sample_id"] in rows and row == rows[row["sample_id"]] and observed_ready == ready[row["sample_id"]],
                "Request differs from the sealed original scientific row/ready")
        result = original_generate(session, row, observed_ready, asset_root, durable_output)
        _audit_raw(session.store.outputs[row["sample_id"]], session.store.events, plan, rows, ready,
                   _runtimes(session.store.events, plan))
        return result

    inference.build_run_identity, inference.RunStore = identity, StrictStore
    inference.ModelSession.__init__, inference.ModelSession.generate = initialize, generate
    try:
        import torch
        with inference._gpu_lock(Path(plan["gpu_lock_root"]), torch):
            native = inference.run_manifest(plan["manifest"]["path"], ready_path=plan["ready"]["path"],
                output_dir=plan["output"], model_key=plan["model"], model_root=plan["model_root"],
                row_ids=plan["sample_ids"], gpu=str(gpu), asset_root=assets, shard_id="strict-panel",
                retry_failed=retry_failed, retry_interrupted=retry_interrupted,
                start_ready=len(plan["sample_ids"]), calibration_root=plan["calibration_root"])
            inference.build_run_identity = originals["build_run_identity"]
            result = verify(plan_path)
            result["native_summary"] = native
            atomic_json(Path(plan["output"]) / (plan["model"] + "-strict-summary.json"), result)
            return result
    finally:
        inference.build_run_identity, inference.RunStore = originals["build_run_identity"], originals["RunStore"]
        inference.ModelSession.__init__, inference.ModelSession.generate = original_init, original_generate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_cli = commands.add_parser("prepare")
    for name in ("manifest", "ready", "asset-root", "model-root", "calibration-root", "output", "plan"):
        prepare_cli.add_argument("--"+name, required=True)
    prepare_cli.add_argument("--model", required=True, choices=MODELS)
    prepare_cli.add_argument("--ids", help="Optional JSON list of original IDs; default all3504")
    run_cli = commands.add_parser("run")
    run_cli.add_argument("--plan", required=True); run_cli.add_argument("--gpu", required=True)
    run_cli.add_argument("--retry-failed", action="store_true"); run_cli.add_argument("--retry-interrupted", action="store_true")
    verify_cli = commands.add_parser("verify"); verify_cli.add_argument("--plan", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        require(not Path(args.plan).exists(), "Sealed strict plans cannot be overwritten")
        value = prepare(args.manifest, args.ready, args.asset_root, args.model_root, args.calibration_root,
                        args.model, args.output, _read(args.ids) if args.ids else None)
        Path(args.plan).parent.mkdir(parents=True, exist_ok=True)
        reference._exclusive(args.plan, value)
        result = {"status": "PREPARED_NO_MODEL_CALLS", "model": args.model, "samples": len(value["sample_ids"]), "plan_sha256": digest(value)}
    elif args.command == "run":
        result = run(args.plan, args.gpu, retry_failed=args.retry_failed, retry_interrupted=args.retry_interrupted)
    else:
        result = verify(args.plan)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
