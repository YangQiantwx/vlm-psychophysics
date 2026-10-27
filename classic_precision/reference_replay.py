"""One independent strict replay of the frozen 48-video delta-Y=.11 cell.

Each model uses the original complete manifest and native runner once: 48
scientific rows plus its existing T3 qualification row. A completed wrong,
invalid or length output remains terminal. Acceptance-suite journals are bound
to an operator-verified comparison SHA; no saved PASS string alone is a gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import functools
import importlib
import json
import os
from pathlib import Path
import sys

from .storage import atomic_json, canonical_json, classify_output, digest, file_sha256
from . import reproducibility as repro


SCHEMA = "classic_precision_reference_replay/v1"
MANIFEST_SHA256 = "e6df00283da944806dd4c4932d86df63cb80bd43bf990c6ba0f676b1bd4e6ed7"
MODELS = ("nemo", "qwen", "cosmos")
ACCEPTED_HARNESSES = {
    "nemo": "8d8a19556f30b7886b1f17ad917ec0b29e1792e5c44bcab98a1c087399ddfaa7",
    "cosmos": "8d8a19556f30b7886b1f17ad917ec0b29e1792e5c44bcab98a1c087399ddfaa7",
    "qwen": "5cc0956359bdab133df85716b4f7d9512bbadb73224a23fc6c8c10dbad68b86e",
}
_require = repro._require
_read_jsonl = repro._read_jsonl


def _relative(root, relative):
    path = Path(relative)
    _require(not path.is_absolute() and ".." not in path.parts, "Evidence path must stay relative")
    root = Path(root).resolve()
    result = (root / path).resolve()
    _require(result.is_relative_to(root), "Evidence path escapes root")
    return result


def _input(raw):
    receipt = raw["input_receipt"]
    keys = ("source_video_sha256", "source_stream_sha256", "chat_template_sha256",
            "chat_template_text", "input_tokens", "processor_source",
            "actual_processor_dtype", "actual_processor_tensor_shape")
    _require(all(key in receipt for key in keys), "Original input identity is incomplete")
    _require(receipt["chat_template_sha256"] == digest(receipt["chat_template_text"]), "Template hash mismatch")
    return {"receipt": {key: receipt[key] for key in keys},
            "effective_generation": raw["effective_generation"],
            "actual_input_tokens": raw["actual_input_tokens"]}


def _bundle(path, descriptor):
    path = _relative(Path(path).parent, descriptor["path"])
    _require(file_sha256(path) == descriptor["sha256"], "Reference raw bundle SHA mismatch")
    records = _read_jsonl(path)
    result = {}
    for item in records:
        raw, family = item["raw"], item["model"]
        key = (family, raw["sample_id"])
        _require(key not in result and item["raw_sha256"] == digest(raw), "Duplicate or altered reference raw")
        _require(item.get("technical_pass") is True, "Original technical PASS missing")
        _input(raw)
        result[key] = raw
    return result


def load_plan(path, *, verify_source=True):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    _require(plan.get("schema") == SCHEMA and plan.get("models") == list(MODELS), "Wrong reference schema/models")
    science, operational = plan.get("scientific_ids", []), plan.get("operational_ids", [])
    auxiliary = plan.get("auxiliary_T3_qualification_id")
    _require(len(science) == len(set(science)) == 48, "Exactly 48 unique scientific IDs required")
    _require(len(operational) == len(set(operational)) == 49 and operational == science + [auxiliary]
             and auxiliary not in science, "Exactly 49 operational IDs required; retain the T3 qualification")
    _require(plan.get("manifest_sha256") == MANIFEST_SHA256, "Only the original full manifest is allowed")
    _require(plan.get("scientific_condition") == {"background_Y": .05, "delta_Y": .11, "duration_frames": 30},
             "Reference scientific condition changed")
    acceptance = plan.get("acceptance_sample_ids", [])
    _require(len(acceptance) == len(set(acceptance)) == 8, "Exactly eight frozen acceptance IDs required")
    _require(plan.get("reference_script_sha256") == file_sha256(__file__), "Reference implementation changed")
    for family in MODELS:
        repro._validate_plan_profile({"profile": plan["profile"]}, family)
    originals = _bundle(path, plan["originals"])
    _require(set(originals) == {(family, sid) for family in MODELS for sid in science}, "144 full original raws required")
    ordinary = _bundle(path, plan["ordinary_qwen"])
    _require(set(ordinary) == {("qwen", sid) for sid in science}, "48 ordinary Qwen full raws required")
    for family in MODELS:
        _require(family in plan.get("qualification_inputs", {}), "Qualification input reference missing")
    if verify_source:
        for key in ("source_run", "manifest", "ready", "asset_root", "calibration_root", "gpu_lock_root"):
            _require(isinstance(plan.get(key), str) and Path(plan[key]).is_absolute(), "Absolute source paths required")
        _require(file_sha256(plan["manifest"]) == MANIFEST_SHA256, "Original manifest bytes changed")
        manifest = json.loads(Path(plan["manifest"]).read_text())
        rows = {row["sample_id"]: row for row in manifest["rows"]}
        _require(len(rows) == len(manifest["rows"]) == 3504, "Original 3504-row manifest required")
        _require(set(operational) <= rows.keys() and rows[auxiliary]["task"] == "T3", "Operational source IDs changed")
        for sid in science:
            row = rows[sid]
            p = row["source_parameters"]
            _require(row["task"] == "T2" and p["background_Y"] == .05 and p["delta_Y"] == .11
                     and p["offset_frame"] - p["onset_frame"] == 30, "Scientific source condition changed")
            for family in MODELS:
                _bind_raw(originals[family, sid], row)
            _bind_raw(ordinary["qwen", sid], row)
    return plan, digest(plan), originals, ordinary


def _bind_raw(raw, row, ready=None):
    actual_ready = raw["ready_record"]
    _require(raw.get("sample_id") == row["sample_id"]
             and all(raw.get(key) == row.get(key) for key in ("task", "truth", "source_parameters")),
             "Raw condition/truth differs from original source")
    _require(ready is None or actual_ready == ready, "Raw ready record differs from original")
    _require(raw.get("ready_identity") == digest(actual_ready)
             and raw.get("row_identity") == digest({"row": row, "ready": actual_ready}), "Raw source identity mismatch")
    receipt = raw["input_receipt"]
    _require(receipt["source_video_sha256"] == actual_ready["video_sha256"]
             and receipt["source_stream_sha256"] == actual_ready["stream_sha256"], "Source video/stream SHA mismatch")
    _require(raw["actual_input_tokens"] == receipt["input_tokens"], "Actual input token count mismatch")
    from .inference import GENERATION
    _require(all(raw["effective_generation"].get(k) == value for k, value in GENERATION.items()),
             "Generation differs from native fixed protocol")
    _input(raw)


def _native_identity(identity):
    return repro._stable({key: value for key, value in identity.items()
                          if key not in {"reproducibility", "purpose", "storage_all_row_ids_sha256"}})


def _suite(root, plan, family, lock, *, collecting=False):
    """Check the fixed suite and return the precise journal closure to freeze."""
    from .summary import model_snapshot, technical_passed, attempt_metrics
    root = Path(root).resolve()
    report_path, plan_path = (_relative(root, lock[key]) for key in ("report", "plan"))
    _require(file_sha256(report_path) == lock["report_sha256"], "Prerequisite report SHA mismatch")
    report, acceptance = json.loads(report_path.read_text()), json.loads(plan_path.read_text())
    _require(report.get("status") == "PASS" and report.get("errors") == []
             and set(report.get("models", {})) == {family}, "Prerequisite report is not a unique model PASS")
    model = report["models"][family]
    _require(model.get("status") == "PASS" and model.get("checked_repeat_stores") == 4
             and model.get("expected_repeat_stores") == 4 and model.get("samples_per_repeat") == 8
             and model.get("mismatches") == [], "Prerequisite needs four stores, eight cases, zero mismatches")
    _require(acceptance.get("cold_passes") == acceptance.get("repeats") == 2
             and acceptance.get("sample_ids") == plan["acceptance_sample_ids"]
             and acceptance.get("manifest_sha256") == MANIFEST_SHA256
             and family in acceptance.get("models", []), "Prerequisite plan/sample scope differs")
    _require(plan.get("gpu_lock_root") == acceptance.get("gpu_lock_root")
             and isinstance(plan.get("gpu_lock_root"), str) and Path(plan["gpu_lock_root"]).is_absolute(),
             "Reference GPU lease root differs from the reviewed acceptance suite")
    profile = acceptance["profile"]
    expected_profile = json.loads(json.dumps(plan["profile"]))
    expected_profile["implementation"]["harness_sha256"] = ACCEPTED_HARNESSES[family]
    _require(profile == expected_profile, "Prerequisite computation profile/source differs")
    source = _relative(root, lock["harness_source"])
    _require(file_sha256(source) == ACCEPTED_HARNESSES[family], "Prerequisite source snapshot changed")
    _require(report.get("plan_sha256") == digest(acceptance), "Prerequisite report/plan binding differs")
    paths = [report_path, plan_path, source]
    baseline, runtime, owners = None, None, {}
    output = _relative(root, lock["output"])
    ids = plan["acceptance_sample_ids"]
    for cold in range(2):
        completion = output / f"cold-{cold}" / (family + ".json")
        done = json.loads(completion.read_text())
        _require(done.get("status") == "COMPLETE" and done.get("plan_sha256") == digest(acceptance),
                 "Prerequisite cold pass incomplete")
        paths.append(completion)
        for repeat in range(2):
            directory = output / f"cold-{cold}" / f"repeat-{repeat}" / family
            identity, rows, events, raw_tail, event_tail = model_snapshot(directory)
            _require(not raw_tail and not event_tail, "Prerequisite journal tail is incomplete")
            binding = identity.get("reproducibility", {})
            _require(identity.get("manifest_sha256") == MANIFEST_SHA256 and identity.get("model_key") == family
                     and binding.get("plan_sha256") == digest(acceptance)
                     and binding.get("profile") == profile and binding.get("harness_sha256") == ACCEPTED_HARNESSES[family]
                     and binding.get("cold_pass") == cold and binding.get("repeat") == repeat
                     and binding.get("sample_ids") == ids, "Prerequisite store identity differs")
            _require(len(rows) == 8 and {r["sample_id"] for r in rows} == set(ids)
                     and technical_passed(rows, events) == set(ids), "Prerequisite missing/duplicate/extra IDs or technical PASS")
            attempts = attempt_metrics(rows, events)
            _require(attempts["started_attempts"] == 8 and attempts["retry_attempts"] == 0
                     and attempts["no_output_attempts"] == attempts["answers_missing_attempt_event"] == 0,
                     "Prerequisite attempts differ from one request per sample")
            snapshots = [e for e in events if e.get("event") == "REPRO_RUNTIME"]
            _require(len(snapshots) == 1, "Prerequisite runtime snapshot missing")
            actual_runtime = repro.validate_runtime_policy(snapshots[0]["snapshot"], family)
            runtime = actual_runtime if runtime is None else runtime
            _require(runtime == actual_runtime, "Prerequisite runtimes differ")
            starts = [e["owner"] for e in events if e.get("event") == "WORKER_START"]
            _require(len(starts) == 1, "Prerequisite was resumed")
            owner = tuple(starts[0].get(k) for k in ("host", "boot_id", "pid", "process_start_ticks"))
            _require(owner[0] and owner[2] and (cold not in owners or owners[cold] == owner), "Same-engine process differs")
            _require(cold in owners or owner not in owners.values(), "Cold suite reused its process")
            owners[cold] = owner
            fingerprints = {}
            for raw in rows:
                sid = raw["sample_id"]
                _require(raw.get("finish_reason") == "stop" and raw.get("raw_answer") in ("LEFT", "RIGHT")
                         and isinstance(raw.get("output_token_ids"), list) and raw["output_token_ids"], "Prerequisite lacks terminal tokens")
                keys = ("task", "truth", "row_identity", "ready_identity", "ready_record", "source_parameters",
                        "effective_generation", "actual_input_tokens", "raw_answer", "output_token_ids", "finish_reason", "stop_reason")
                fingerprints[sid] = {k: raw.get(k) for k in keys}
                fingerprints[sid]["prompt_sha256"] = raw["input_receipt"]["chat_template_sha256"]
                fingerprints[sid]["input_hash_receipts"] = repro._hash_receipts(events, sid, raw["attempt_id"], family)
            _require(model.get("reference_fingerprints_sha256") == digest(fingerprints), "Prerequisite saved-token/input fingerprint mismatch")
            stable = {k: v for k, v in identity.items() if k not in {"reproducibility", "storage_all_row_ids_sha256"}}
            value = {"native_identity": stable, "fingerprints": fingerprints}
            baseline = value if baseline is None else baseline
            _require(value == baseline, "Prerequisite exact input/token/native identity mismatch")
            for name in ("identity.json", "outputs.jsonl", "events.jsonl", "qualification-repro.json", "reproducibility.json"):
                paths.append(directory / name)
    closure = {str(p.relative_to(root)): file_sha256(p) for p in paths}
    result = {**lock, "plan_sha256": digest(acceptance), "files": closure, "runtime_identity": runtime,
              "native_identity": _native_identity(baseline["native_identity"])}
    if not collecting:
        _require(result == lock, "Prerequisite locked journal closure changed")
    return result


def seal_prerequisite(plan_path, family, root, comparison, expected_sha256, acceptance_plan, output, harness_source):
    """Freeze a concrete operator-verified suite; never launch any model."""
    plan, _, _, _ = load_plan(plan_path, verify_source=False)
    _require(family in MODELS, "Unknown reference model")
    lock = {"report": comparison, "report_sha256": expected_sha256,
            "plan": acceptance_plan, "output": output, "harness_source": harness_source}
    plan.setdefault("prerequisites", {})[family] = _suite(root, plan, family, lock, collecting=True)
    atomic_json(plan_path, plan)
    return {"model": family, "plan_sha256": digest(plan), "prerequisite_sha256": digest(plan["prerequisites"][family])}


def _exclusive(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def run(plan_path, output, model, gpu, model_root, prerequisite_root, cosmos_package=None):
    """Execute native run_manifest once, with runtime and original-input guards."""
    from . import inference
    plan, plan_hash, originals, _ = load_plan(plan_path)
    _require(model in MODELS, "Model is outside the three-model reference")
    lock = plan.get("prerequisites", {}).get(model)
    _require(isinstance(lock, dict) and lock.get("files"), "A sealed real prerequisite suite is required")
    _suite(prerequisite_root, plan, model, lock)
    output = Path(output).resolve()
    source = Path(plan["source_run"]).resolve()
    _require(output != source and not source.is_relative_to(output) and not output.is_relative_to(source),
             "Use an independent reference output outside the original source run")
    _require(not (output / model).exists(), "Existing model output/attempts cannot be implicitly resumed")
    _require(bool(str(gpu)) and "," not in str(gpu), "Exactly one explicit GPU required")
    _require("torch" not in sys.modules and "vllm" not in sys.modules, "Reference requires a fresh Python process")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    runtime = importlib.import_module("classic_precision_determinism")
    runtime.activate(model)
    output.mkdir(parents=True, exist_ok=True)
    claim = {"schema": SCHEMA, "plan_sha256": plan_hash, "model": model,
             "reference_script_sha256": file_sha256(__file__), "prerequisite_sha256": digest(lock)}
    _exclusive(output / ("." + model + ".reference-claim.json"), claim)
    original_init, original_prepare = inference.ModelSession.__init__, inference.ModelSession.prepare
    sessions, prepared = [], set()

    @functools.wraps(original_init)
    def initialize(session, family, *args, **kwargs):
        _require(family == model and not sessions, "One model and one native engine per reference")
        sessions.append(session)
        original_init(session, family, *args, **kwargs)
        try:
            _require(_native_identity(session.store.identity) == lock["native_identity"],
                     "Reference native model/checkpoint/calibration/resources identity differs from accepted suite")
            repro._prepare_runtime_modules(model)
            snapshot = runtime.runtime_snapshot()
            identity = repro.validate_runtime_policy(snapshot, model)
            _require(identity == lock["runtime_identity"], "Reference computation runtime differs from accepted suite")
            session.store.event("REFERENCE_RUNTIME", snapshot=snapshot,
                                comparison_identity=identity, **claim)
        except BaseException:
            session.close()
            raise

    @functools.wraps(original_prepare)
    def prepare(session, row, ready, asset_root):
        sid = row["sample_id"]
        _require(sid in plan["operational_ids"] and sid not in prepared, "Unexpected or repeated reference request")
        result = original_prepare(session, row, ready, asset_root)
        reference = (_input(originals[model, sid]) if sid in plan["scientific_ids"]
                     else plan["qualification_inputs"][model])
        actual = _input({"input_receipt": result[-1], "effective_generation": session.effective_generation,
                         "actual_input_tokens": result[-1]["input_tokens"]})
        _require(actual == reference, "Original input/template/effective generation mismatch before request: " + sid)
        prepared.add(sid)
        return result

    inference.ModelSession.__init__, inference.ModelSession.prepare = initialize, prepare
    try:
        import torch
        # Hold the reviewed physical-GPU lease through model load, every native
        # request, model close and final validation. The unchanged native runner
        # additionally holds its own independent-output lease inside this scope.
        with inference._gpu_lock(Path(plan["gpu_lock_root"]), torch):
            native = inference.run_manifest(plan["manifest"], ready_path=plan["ready"], output_dir=output,
                model_key=model, model_root=model_root, row_ids=plan["operational_ids"], gpu=str(gpu),
                asset_root=plan["asset_root"], cosmos_package=cosmos_package, shard_id="reference",
                retry_failed=False, retry_interrupted=False, start_ready=49, calibration_root=plan["calibration_root"])
            result = analyze(plan_path, output, model)
            result["native_summary"] = native
            atomic_json(output / (model + "-reference.json"), result)
            return result
    except BaseException as error:
        atomic_json(output / (model + "-reference-failure.json"), {**claim, "status": "FAILED",
                    "error_type": type(error).__name__, "error": str(error), "prepared_ids": sorted(prepared)})
        raise
    finally:
        inference.ModelSession.__init__, inference.ModelSession.prepare = original_init, original_prepare


def _outcome(raw):
    value = classify_output(raw)
    return {"answer": value["answer"], "raw_answer": raw["raw_answer"],
            "tokens": raw["output_token_ids"], "finish_reason": raw["finish_reason"],
            "stop_reason": raw.get("stop_reason"), "correct": value["correct"],
            "outcome": value["outcome"], "seconds": raw.get("seconds")}


def _counts(rows):
    result = {"denominator": len(rows), "first_correct": sum(r["first"]["correct"] for r in rows),
            "strict_correct": sum(r["strict"]["correct"] for r in rows),
            "same_answer": sum(r["first"]["raw_answer"] == r["strict"]["raw_answer"] for r in rows),
            "same_tokens": sum(r["first"]["tokens"] == r["strict"]["tokens"] for r in rows),
            "both_correct": sum(r["first"]["correct"] and r["strict"]["correct"] for r in rows),
            "wrong_to_correct": sum(r["change"] == "wrong_to_correct" for r in rows),
            "correct_to_wrong": sum(r["change"] == "correct_to_wrong" for r in rows)}
    if rows and rows[0]["ordinary_qwen"] is not None:
        result.update(ordinary_qwen_correct=sum(r["ordinary_qwen"]["correct"] for r in rows),
                      same_tokens_ordinary_qwen_vs_strict=sum(r["ordinary_qwen"]["tokens"] == r["strict"]["tokens"] for r in rows),
                      same_answer_ordinary_qwen_vs_strict=sum(r["ordinary_qwen"]["raw_answer"] == r["strict"]["raw_answer"] for r in rows))
    return result


def analyze(plan_path, output, model):
    """Compare complete technical outputs, retaining invalid/length as terminal."""
    from .summary import model_snapshot, technical_passed, attempt_metrics
    plan, plan_hash, originals, ordinary = load_plan(plan_path)
    _require(model in MODELS, "Unknown reference model")
    output = Path(output).resolve()
    claim = json.loads((output / ("." + model + ".reference-claim.json")).read_text())
    _require(claim.get("plan_sha256") == plan_hash and claim.get("reference_script_sha256") == file_sha256(__file__)
             and claim.get("prerequisite_sha256") == digest(plan["prerequisites"][model]), "Reference claim differs")
    identity, raw, events, raw_tail, event_tail = model_snapshot(output / model)
    ids = plan["operational_ids"]
    _require(not raw_tail and not event_tail and len(raw) == 49 and {r["sample_id"] for r in raw} == set(ids),
             "Reference needs exactly 49 unique complete raw outputs")
    _require(identity.get("manifest_sha256") == MANIFEST_SHA256 and identity.get("model_key") == model,
             "Reference native identity differs")
    _require(_native_identity(identity) == plan["prerequisites"][model]["native_identity"],
             "Reference native model/checkpoint/calibration/resources identity differs from accepted suite")
    attempts = attempt_metrics(raw, events)
    _require(attempts["started_attempts"] == 49 and attempts["retry_attempts"] == 0
             and attempts["no_output_attempts"] == attempts["answers_missing_attempt_event"] == 0,
             "Reference must contain exactly one attempt per operational ID")
    _require(technical_passed(raw, events) == set(ids), "Reference technical validation is unresolved")
    runtime = [e for e in events if e.get("event") == "REFERENCE_RUNTIME"]
    _require(len(runtime) == 1 and all(runtime[0].get(k) == v for k, v in claim.items()), "Reference runtime binding missing")
    actual_runtime = repro.validate_runtime_policy(runtime[0]["snapshot"], model)
    _require(actual_runtime == runtime[0]["comparison_identity"] == plan["prerequisites"][model]["runtime_identity"],
             "Reference runtime differs from accepted suite")
    manifest = json.loads(Path(plan["manifest"]).read_text())
    source = {r["sample_id"]: r for r in manifest["rows"]}
    comparisons = []
    for value in raw:
        sid = value["sample_id"]
        _bind_raw(value, source[sid])
        tokens = value.get("output_token_ids")
        _require(isinstance(tokens, list) and all(type(t) is int for t in tokens)
                 and isinstance(value.get("raw_answer"), str) and value.get("finish_reason") in ("stop", "length", "abort"),
                 "Incomplete terminal raw/token record")
        receipts = repro._hash_receipts(events, sid, value["attempt_id"], model)
        if sid not in plan["scientific_ids"]:
            continue
        first = originals[model, sid]
        _require(_input(value) == _input(first), "Strict original input/template/generation differs")
        a, b = _outcome(first), _outcome(value)
        direction = ("wrong_to_correct" if b["correct"] else "correct_to_wrong") if a["correct"] != b["correct"] else "unchanged_correctness"
        item = {"sample_id": sid, "truth": value["truth"], "onset_index": value["source_parameters"]["onset_index"],
                "first": a, "ordinary_qwen": _outcome(ordinary["qwen", sid]) if model == "qwen" else None,
                "strict": b, "change": direction, "source_video_sha256": value["input_receipt"]["source_video_sha256"],
                "source_stream_sha256": value["input_receipt"]["source_stream_sha256"],
                "chat_template_sha256": value["input_receipt"]["chat_template_sha256"],
                "strict_input_hashes": receipts, "historical_tensor_hashes": "not_recorded; byte equality to historical GPU input is unverified",
                "strict_technical_pass": True}
        comparisons.append(item)
    comparisons.sort(key=lambda r: plan["scientific_ids"].index(r["sample_id"]))
    totals = _counts(comparisons)
    totals.update(first_accuracy=totals["first_correct"] / 48, strict_accuracy=totals["strict_correct"] / 48,
                  paired_accuracy_difference=(totals["strict_correct"] - totals["first_correct"]) / 48)
    return {"schema": SCHEMA, "status": "COMPLETE", "model": model, "plan_sha256": plan_hash,
            "scientific_denominator": 48, "operational_outputs": 49, "attempts": attempts,
            "all_operational_stop_exact_choice": all(r["finish_reason"] == "stop" and r["raw_answer"] in ("LEFT", "RIGHT") for r in raw),
            "terminal_outcomes": dict(Counter(classify_output(r)["outcome"] for r in raw)),
            "summary": totals, "by_truth": {s: _counts([r for r in comparisons if r["truth"] == s]) for s in ("LEFT", "RIGHT")},
            "by_onset": {str(i): _counts([r for r in comparisons if r["onset_index"] == i]) for i in range(24)},
            "generation_seconds": sum(r.get("seconds", 0) for r in raw),
            "load_seconds": sum(e.get("load_seconds", 0) for e in events if e.get("event") == "MODEL_READY"),
            "comparisons": comparisons, "historical_outputs_replaced": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run")
    checker = commands.add_parser("analyze")
    for command in (runner, checker):
        command.add_argument("--plan", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--model", required=True, choices=MODELS)
    runner.add_argument("--gpu", required=True)
    runner.add_argument("--model-root", required=True)
    runner.add_argument("--prerequisite-root", required=True)
    runner.add_argument("--cosmos-package")
    seal = commands.add_parser("seal-prerequisite")
    for name in ("plan", "model", "root", "comparison", "expected-sha256", "acceptance-plan", "output", "harness-source"):
        seal.add_argument("--" + name, required=True)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "seal-prerequisite":
        args["plan_path"] = args.pop("plan")
        args["family"] = args.pop("model")
        result = seal_prerequisite(**args)
    else:
        args["plan_path"] = args.pop("plan")
        result = run(**args) if command == "run" else analyze(**args)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
