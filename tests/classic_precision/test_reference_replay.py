"""CPU checks of finite membership, real prerequisite evidence and request gates."""
from copy import deepcopy
import contextlib
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from classic_precision import inference
from classic_precision import reference_replay as ref
from classic_precision import reproducibility as repro
from classic_precision.storage import canonical_json, digest, file_sha256


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n")


def _journal(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(r) + "\n" for r in records))


def _raw(row, family):
    sid = row["sample_id"]
    ready = {"sample_id": sid, "video_sha256": digest([sid, "video"]),
             "stream_sha256": digest([sid, "stream"]), "source_parameters": row["source_parameters"]}
    text = "fixed native OFF " + row["task"]
    return {**row, "attempt_id": family + "-" + sid, "attempt_number": 1,
            "row_identity": digest({"row": row, "ready": ready}), "ready_identity": digest(ready),
            "ready_record": ready, "effective_generation": {**inference.GENERATION, "backend": "fixture"},
            "actual_input_tokens": 100, "input_receipt": {
                "source_video_sha256": ready["video_sha256"], "source_stream_sha256": ready["stream_sha256"],
                "chat_template_text": text, "chat_template_sha256": digest(text), "input_tokens": 100,
                "processor_source": {"module": "fixture", "source_sha256": "a" * 64},
                "actual_processor_dtype": "torch.bfloat16" if family == "nemo" else "torch.float32",
                "actual_processor_tensor_shape": [180, 3, 512, 768] if family == "nemo" else [138240, 1536]},
            "raw_answer": row["truth"], "output_token_ids": [1 if row["truth"] == "LEFT" else 2, 3],
            "finish_reason": "stop", "stop_reason": None, "seconds": 1.5}


def _runtime(family):
    contract = repro._profile_contract()
    return {"comparison_identity": {
        "profile": {"family": family, "profile_id": contract["PROFILE_ID"],
                    "base_runner_sha256": contract["BASE_RUNNER_SHA256"], "helper_sha256": contract["helper_sha256"],
                    "environment": contract["ENVIRONMENT"], "sdk_files_modified": False, "qsa_installation": None},
        "inference_sha256": contract["BASE_RUNNER_SHA256"], "effective_environment": contract["ENVIRONMENT"],
        "deterministic_algorithms": True, "deterministic_warn_only": False,
        "cudnn_benchmark": False, "cudnn_deterministic": True,
        "fill_uninitialized_memory": contract["FILL_UNINITIALIZED_MEMORY"][family],
        "fp32_precision": {"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"},
        "packages": {k: "fixture" for k in ("torch", "transformers", "numpy", "av")},
        "gpu_name": "fixture GPU", "compute_capability": [12, 0], "cuda": "13.0", "driver": "fixture"}}


def _hashes(sid, aid, family):
    def tensor(name, dtype, shape, size):
        import math
        return {"name": name, "dtype": dtype, "shape": shape, "bytes": math.prod(shape) * size,
                "sha256": digest([sid, name, dtype])}
    pixels = tensor("pixel_values_flat_video", "torch.bfloat16", [180, 3, 512, 768], 2)
    if family != "nemo":
        pixels = tensor("pixel_values_videos", "torch.float32", [138240, 1536], 4)
    gpu = tensor("pixels", "torch.bfloat16", [180, 3, 512, 768] if family == "nemo" else [138240, 1536], 2)
    return [{"event": "REPRO_INPUT_HASH", "sample_id": sid, "attempt_id": aid, "stage": "cpu_processor",
             "tensors": [tensor("input_ids", "torch.int64", [1, 100], 8), pixels]},
            {"event": "REPRO_INPUT_HASH", "sample_id": sid, "attempt_id": aid, "stage": "gpu_forward", "tensors": [gpu]}]


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    science = ["s" + str(i).zfill(2) for i in range(48)]
    aux = "qualification-t3"
    rows = [{"sample_id": sid, "task": "T2", "truth": "LEFT" if i % 2 == 0 else "RIGHT",
             "source_parameters": {"background_Y": .05, "delta_Y": .11, "onset_frame": 18 + 5 * (i // 2),
                                   "offset_frame": 48 + 5 * (i // 2), "onset_index": i // 2}}
            for i, sid in enumerate(science)]
    rows += [{"sample_id": aux, "task": "T3", "truth": "LEFT", "source_parameters": {"angle_deg": 4, "onset_index": 0}},
             {"sample_id": "other-t3", "task": "T3", "truth": "RIGHT", "source_parameters": {"angle_deg": 4, "onset_index": 0}}]
    originals = {(family, row["sample_id"]): _raw(row, family) for family in ref.MODELS for row in rows}
    manifest = {"rows": rows + [{"sample_id": f"filler-{i}"} for i in range(3504 - len(rows))]}
    source = tmp_path / "source"
    _write(source / "manifest.json", manifest)
    monkeypatch.setattr(ref, "MANIFEST_SHA256", file_sha256(source / "manifest.json"))
    root = tmp_path / "reference"
    root.mkdir()
    for name, records in [("originals.jsonl", [originals[f, sid] for f in ref.MODELS for sid in science]),
                          ("ordinary.jsonl", [originals["qwen", sid] for sid in science])]:
        _journal(root / name, [{"model": next(f for f in ref.MODELS if raw["attempt_id"].startswith(f + "-")),
                              "raw": raw, "raw_sha256": digest(raw), "technical_pass": True} for raw in records])
    contract = repro._profile_contract()
    profile = {"id": contract["PROFILE_ID"], "implementation": {
        "determinism_helper_sha256": contract["helper_sha256"], "qsa_helper_sha256": contract["qsa_helper_sha256"],
        "base_runner_sha256": contract["BASE_RUNNER_SHA256"], "harness_sha256": file_sha256(repro.__file__)},
        "environment": contract["ENVIRONMENT"], "model_environment": {"qwen": {"CLASSIC_PRECISION_QSA_DETERMINISM": "1"}}}
    plan = {"schema": ref.SCHEMA, "models": list(ref.MODELS), "scientific_ids": science,
            "operational_ids": science + [aux], "auxiliary_T3_qualification_id": aux,
            "manifest_sha256": ref.MANIFEST_SHA256, "scientific_condition": {
                "background_Y": .05, "delta_Y": .11, "duration_frames": 30},
            "reference_script_sha256": file_sha256(ref.__file__), "profile": profile,
            "acceptance_sample_ids": science[:6] + [aux, "other-t3"],
            "source_run": str(source), "manifest": str(source / "manifest.json"), "ready": str(source / "ready.jsonl"),
            "asset_root": str(source), "calibration_root": str(source / "calibration"),
            "gpu_lock_root": str(tmp_path / "reviewed-shared-gpu-lease"),
            "originals": {"path": "originals.jsonl", "sha256": file_sha256(root / "originals.jsonl")},
            "ordinary_qwen": {"path": "ordinary.jsonl", "sha256": file_sha256(root / "ordinary.jsonl")},
            "qualification_inputs": {f: ref._input(originals[f, aux]) for f in ref.MODELS}, "prerequisites": {}}
    path = root / "plan.json"
    _write(path, plan)
    return SimpleNamespace(path=path, plan=plan, root=root, raws=originals, rows={r["sample_id"]: r for r in rows})


def _suite(frozen, monkeypatch, family="nemo"):
    root = frozen.root / "prerequisite"
    source = root / "harness.py"
    source.parent.mkdir()
    source.write_text("# original reviewed fixture harness\n")
    monkeypatch.setitem(ref.ACCEPTED_HARNESSES, family, file_sha256(source))
    profile = deepcopy(frozen.plan["profile"])
    profile["implementation"]["harness_sha256"] = file_sha256(source)
    acceptance = {"profile": profile, "cold_passes": 2, "repeats": 2, "models": [family],
                  "sample_ids": frozen.plan["acceptance_sample_ids"], "manifest_sha256": ref.MANIFEST_SHA256,
                  "gpu_lock_root": frozen.plan["gpu_lock_root"]}
    _write(root / "acceptance.json", acceptance)
    all_rows = [deepcopy(frozen.raws[family, sid]) for sid in acceptance["sample_ids"]]
    for cold in range(2):
        _write(root / "outputs" / f"cold-{cold}" / (family + ".json"), {"status": "COMPLETE", "plan_sha256": digest(acceptance)})
        for repeat in range(2):
            directory = root / "outputs" / f"cold-{cold}" / f"repeat-{repeat}" / family
            identity = {"manifest_sha256": ref.MANIFEST_SHA256, "model_key": family,
                "model": {"revision": "reviewed-checkpoint"}, "checkpoint_assets": {"config": "reviewed-bytes"},
                "effective_resources": {"min_free_gpu_gib": 80}, "calibration_identity_sha256": "c" * 64,
                "cosmos_package_sources": None, "purpose": repro.SCHEMA, "storage_all_row_ids_sha256": digest(acceptance["sample_ids"]),
                "reproducibility": {
                "plan_sha256": digest(acceptance), "profile": profile, "harness_sha256": file_sha256(source),
                "cold_pass": cold, "repeat": repeat, "sample_ids": acceptance["sample_ids"]}}
            events = [{"event": "WORKER_START", "owner": {"host": "fixture", "boot_id": "fixture", "pid": cold + 1, "process_start_ticks": cold + 20}},
                      {"event": "REPRO_RUNTIME", "snapshot": _runtime(family)}]
            for raw in all_rows:
                sid, aid = raw["sample_id"], raw["attempt_id"]
                events += [{"event": "INFLIGHT", "sample_id": sid, "attempt_id": aid},
                           {"event": "TECHNICAL_PASS", "sample_id": sid, "attempt_id": aid}]
                events += _hashes(sid, aid, family)
            _write(directory / "identity.json", identity)
            _journal(directory / "outputs.jsonl", all_rows)
            _journal(directory / "events.jsonl", events)
            _write(directory / "qualification-repro.json", {"status": "PASS"})
            _write(directory / "reproducibility.json", {"status": "COMPLETE"})
    fingerprints = {}
    keys = ("task", "truth", "row_identity", "ready_identity", "ready_record", "source_parameters",
            "effective_generation", "actual_input_tokens", "raw_answer", "output_token_ids", "finish_reason", "stop_reason")
    for raw in all_rows:
        sid = raw["sample_id"]
        fingerprints[sid] = {k: raw.get(k) for k in keys}
        fingerprints[sid].update(prompt_sha256=raw["input_receipt"]["chat_template_sha256"],
                                 input_hash_receipts=repro._hash_receipts(events, sid, raw["attempt_id"], family))
    report = {"status": "PASS", "errors": [], "models": {family: {"status": "PASS", "checked_repeat_stores": 4,
        "expected_repeat_stores": 4, "samples_per_repeat": 8, "mismatches": [], "reference_fingerprints_sha256": digest(fingerprints)}},
        "plan_sha256": digest(acceptance)}
    _write(root / "comparison.json", report)
    lock = {"report": "comparison.json", "report_sha256": file_sha256(root / "comparison.json"),
            "plan": "acceptance.json", "output": "outputs", "harness_source": "harness.py"}
    return root, lock


def _seal(frozen, monkeypatch):
    root, lock = _suite(frozen, monkeypatch)
    locked = ref._suite(root, frozen.plan, "nemo", lock, collecting=True)
    frozen.plan["prerequisites"]["nemo"] = locked
    _write(frozen.path, frozen.plan)
    return root, locked


def test_reference_bundle_keeps_all_144_full_raw_records_and_scientific_denominator(frozen):
    plan, _, originals, ordinary = ref.load_plan(frozen.path)
    assert len(originals) == 144 and len(ordinary) == 48
    assert len(plan["operational_ids"]) == 49 and len(plan["scientific_ids"]) == 48
    assert originals["qwen", "s00"]["output_token_ids"] == [1, 3]


@pytest.mark.parametrize("mutation,match", [
    (lambda p: p["operational_ids"].pop(), "49 operational"),
    (lambda p: p["scientific_ids"].__setitem__(1, "s00"), "48 unique"),
    (lambda p: p.__setitem__("manifest_sha256", "0" * 64), "original full manifest"),
    (lambda p: p["models"].append("gemma"), "schema/models"),
    (lambda p: p["profile"]["environment"].__setitem__("PYTHONHASHSEED", "1"), "environment"),
])
def test_plan_rejects_membership_protocol_and_profile_drift(frozen, mutation, match):
    mutation(frozen.plan)
    _write(frozen.path, frozen.plan)
    with pytest.raises(ValueError, match=match):
        ref.load_plan(frozen.path)


def test_original_bundle_hash_and_original_input_binding_are_checked(frozen):
    path = frozen.root / "originals.jsonl"
    records = ref._read_jsonl(path)
    records[0]["raw"]["input_receipt"]["source_video_sha256"] = "0" * 64
    records[0]["raw_sha256"] = digest(records[0]["raw"])
    _journal(path, records)
    with pytest.raises(ValueError, match="bundle SHA"):
        ref.load_plan(frozen.path)
    frozen.plan["originals"]["sha256"] = file_sha256(path)
    _write(frozen.path, frozen.plan)
    with pytest.raises(ValueError, match="video/stream SHA"):
        ref.load_plan(frozen.path)


def test_sealed_suite_requires_real_four_store_tokens_inputs_and_runtime(frozen, monkeypatch):
    root, locked = _seal(frozen, monkeypatch)
    assert len(locked["files"]) == 25
    assert ref._suite(root, frozen.plan, "nemo", locked) == locked
    path = root / "outputs/cold-1/repeat-1/nemo/outputs.jsonl"
    records = ref._read_jsonl(path)
    records[0]["output_token_ids"] = [9, 9]
    _journal(path, records)
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ref._suite(root, frozen.plan, "nemo", locked)


def test_pass_json_alone_cannot_seal_prerequisite(frozen, monkeypatch):
    root, lock = _suite(frozen, monkeypatch)
    (root / "outputs/cold-1/repeat-1/nemo/outputs.jsonl").unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        ref._suite(root, frozen.plan, "nemo", lock, collecting=True)


def test_reference_must_share_the_exact_reviewed_gpu_lease_root(frozen, monkeypatch):
    root, lock = _suite(frozen, monkeypatch)
    frozen.plan["gpu_lock_root"] = str(frozen.root / "independent-wrong-lease")
    with pytest.raises(ValueError, match="GPU lease root"):
        ref._suite(root, frozen.plan, "nemo", lock, collecting=True)


@pytest.mark.parametrize("kind", ["extra_model", "extra_sample", "missing_technical", "relaxed_runtime", "journal_change"])
def test_prerequisite_negative_evidence_is_rejected(frozen, monkeypatch, kind):
    root, locked = _seal(frozen, monkeypatch)
    directory = root / "outputs/cold-1/repeat-1/nemo"
    if kind == "extra_model":
        path = root / "comparison.json"
        report = json.loads(path.read_text())
        report["models"]["cosmos"] = report["models"]["nemo"]
        _write(path, report)
        locked["report_sha256"] = file_sha256(path)
    elif kind == "extra_sample":
        path = directory / "outputs.jsonl"
        rows = ref._read_jsonl(path)
        rows.append(rows[0])
        _journal(path, rows)
    elif kind == "missing_technical":
        path = directory / "events.jsonl"
        _journal(path, [r for r in ref._read_jsonl(path) if r["event"] != "TECHNICAL_PASS"])
    elif kind == "relaxed_runtime":
        path = directory / "events.jsonl"
        events = ref._read_jsonl(path)
        next(e for e in events if e["event"] == "REPRO_RUNTIME")["snapshot"]["comparison_identity"]["deterministic_algorithms"] = False
        _journal(path, events)
    else:
        with (directory / "events.jsonl").open("a") as stream:
            stream.write('{"event":"UNBOUND_EXTRA_EVENT"}\n')
    with pytest.raises(ValueError):
        ref._suite(root, frozen.plan, "nemo", locked)


def _fake_native(frozen, monkeypatch, *, bad_runtime=False, bad_input=False, bad_native_field=None):
    calls = []
    frozen.lease_events = []
    held = False
    runtime = ModuleType("classic_precision_determinism")
    def activate(family):
        monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
        return {"profile_id": "fixture"}
    runtime.activate = activate
    runtime.runtime_snapshot = lambda: {"guard": "bad"} if bad_runtime else _runtime("nemo")
    monkeypatch.setitem(sys.modules, "classic_precision_determinism", runtime)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(repro, "_prepare_runtime_modules", lambda model: None)

    @contextlib.contextmanager
    def shared_lease(path, torch):
        nonlocal held
        assert not held and Path(path) == Path(frozen.plan["gpu_lock_root"])
        frozen.lease_events.append("acquire_reviewed_lease")
        held = True
        try:
            yield
        finally:
            held = False
            frozen.lease_events.append("release_reviewed_lease")
    monkeypatch.setattr(inference, "_gpu_lock", shared_lease)

    def initialize(session, family, model_path, store, *args):
        assert held
        frozen.lease_events.append("model_initialize")
        session.family, session.store = family, store
        session.effective_generation = deepcopy(frozen.raws[family, "s00"]["effective_generation"])
    def prepare(session, row, ready, assets):
        assert held
        receipt = deepcopy(frozen.raws[session.family, row["sample_id"]]["input_receipt"])
        if bad_input:
            receipt["source_stream_sha256"] = "0" * 64
        return None, None, receipt["chat_template_text"], receipt
    monkeypatch.setattr(inference.ModelSession, "__init__", initialize)
    monkeypatch.setattr(inference.ModelSession, "prepare", prepare)
    def close(session):
        assert held
        frozen.lease_events.append("model_close")
    monkeypatch.setattr(inference.ModelSession, "close", close)
    def native(*args, **kwargs):
        assert kwargs["row_ids"] == frozen.plan["operational_ids"]
        assert kwargs["retry_failed"] is kwargs["retry_interrupted"] is False
        assert kwargs["start_ready"] == 49
        assert held
        identity = deepcopy(frozen.plan["prerequisites"]["nemo"]["native_identity"])
        identity.update(purpose="independent-reference", storage_all_row_ids_sha256="different-allowed-membership")
        if bad_native_field:
            identity[bad_native_field] = "unreviewed-value"
        store = SimpleNamespace(event=lambda *args, **kwargs: None, identity=identity)
        session = inference.ModelSession(kwargs["model_key"], "fixture", store)
        try:
            row = frozen.rows["s00"]
            session.prepare(row, frozen.raws["nemo", "s00"]["ready_record"], "assets")
            assert held
            calls.append("MODEL_REQUEST")
            return {"status": "COMPLETE"}
        finally:
            session.close()
    monkeypatch.setattr(inference, "run_manifest", native)
    def analyze(*args):
        assert held
        frozen.lease_events.append("analyze_complete_outputs")
        return {"status": "COMPLETE"}
    monkeypatch.setattr(ref, "analyze", analyze)
    return calls


@pytest.mark.parametrize("bad_runtime,bad_input", [(True, False), (False, True)])
def test_runtime_or_original_input_guard_failure_issues_zero_model_requests(frozen, monkeypatch, bad_runtime, bad_input):
    root, _ = _seal(frozen, monkeypatch)
    calls = _fake_native(frozen, monkeypatch, bad_runtime=bad_runtime, bad_input=bad_input)
    with pytest.raises(ValueError):
        ref.run(frozen.path, frozen.root / "strict", "nemo", "0", "models", root)
    assert calls == []
    assert json.loads((frozen.root / "strict/nemo-reference-failure.json").read_text())["status"] == "FAILED"


def test_single_finite_native_run_and_claim_refuse_implicit_resume(frozen, monkeypatch):
    root, _ = _seal(frozen, monkeypatch)
    calls = _fake_native(frozen, monkeypatch)
    output = frozen.root / "strict"
    assert ref.run(frozen.path, output, "nemo", "0", "models", root)["status"] == "COMPLETE"
    assert calls == ["MODEL_REQUEST"]
    assert frozen.lease_events == ["acquire_reviewed_lease", "model_initialize", "model_close",
                                   "analyze_complete_outputs", "release_reviewed_lease"]
    monkeypatch.delitem(sys.modules, "torch")  # A second fresh diagnostic process.
    with pytest.raises(FileExistsError):
        ref.run(frozen.path, output, "nemo", "0", "models", root)
    assert calls == ["MODEL_REQUEST"]


@pytest.mark.parametrize("field", ["checkpoint_assets", "calibration_identity_sha256",
                                    "model", "effective_resources", "cosmos_package_sources"])
def test_native_model_identity_guard_fails_before_first_request(frozen, monkeypatch, field):
    root, _ = _seal(frozen, monkeypatch)
    calls = _fake_native(frozen, monkeypatch, bad_native_field=field)
    with pytest.raises(ValueError, match="native model/checkpoint/calibration/resources identity"):
        ref.run(frozen.path, frozen.root / "strict", "nemo", "0", "models", root)
    assert calls == []
    assert frozen.lease_events == ["acquire_reviewed_lease", "model_initialize", "model_close", "release_reviewed_lease"]


def _strict_store(frozen, monkeypatch):
    _, lock = _seal(frozen, monkeypatch)
    output = frozen.root / "strict"
    directory = output / "nemo"
    plan_hash = digest(frozen.plan)
    claim = {"schema": ref.SCHEMA, "plan_sha256": plan_hash, "model": "nemo",
             "reference_script_sha256": file_sha256(ref.__file__), "prerequisite_sha256": digest(lock)}
    _write(output / ".nemo.reference-claim.json", claim)
    identity = deepcopy(lock["native_identity"])
    identity.update(purpose="reference", storage_all_row_ids_sha256="native-full-manifest-membership")
    _write(directory / "identity.json", identity)
    rows = [deepcopy(frozen.raws["nemo", sid]) for sid in frozen.plan["operational_ids"]]
    events = [{"event": "REFERENCE_RUNTIME", "snapshot": _runtime("nemo"), "comparison_identity": lock["runtime_identity"], **claim}]
    for row in rows:
        sid, aid = row["sample_id"], row["attempt_id"]
        events += [{"event": "INFLIGHT", "sample_id": sid, "attempt_id": aid},
                   {"event": "TECHNICAL_PASS", "sample_id": sid, "attempt_id": aid}]
        events += _hashes(sid, aid, "nemo")
    _journal(directory / "outputs.jsonl", rows)
    _journal(directory / "events.jsonl", events)
    return output, directory, rows, events


@pytest.mark.parametrize("field", ["checkpoint_assets", "calibration_identity_sha256",
                                    "model", "effective_resources", "cosmos_package_sources"])
def test_analysis_rejects_native_model_identity_drift(frozen, monkeypatch, field):
    output, directory, _, _ = _strict_store(frozen, monkeypatch)
    identity_path = directory / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity[field] = "unreviewed-value"
    _write(identity_path, identity)
    with pytest.raises(ValueError, match="native model/checkpoint/calibration/resources identity"):
        ref.analyze(frozen.path, output, "nemo")


def test_complete_wrong_invalid_and_length_outputs_are_terminal_in_same_48_denominator(frozen, monkeypatch):
    output, directory, rows, _ = _strict_store(frozen, monkeypatch)
    rows[0].update(raw_answer="RIGHT", output_token_ids=[2, 3])
    rows[1].update(raw_answer="explanation", output_token_ids=[4, 5])
    rows[2].update(raw_answer="LEFT", finish_reason="length", output_token_ids=[1])
    _journal(directory / "outputs.jsonl", rows)
    result = ref.analyze(frozen.path, output, "nemo")
    assert result["status"] == "COMPLETE" and result["scientific_denominator"] == 48
    assert result["operational_outputs"] == 49 and result["summary"]["strict_correct"] == 45
    assert result["summary"]["correct_to_wrong"] == 3 and result["attempts"]["retry_attempts"] == 0
    assert result["all_operational_stop_exact_choice"] is False
    assert len(result["by_onset"]) == 24 and result["historical_outputs_replaced"] is False


@pytest.mark.parametrize("kind", ["missing", "duplicate", "technical", "retry", "hash_missing"])
def test_analysis_refuses_incomplete_or_unresolved_operational_evidence(frozen, monkeypatch, kind):
    output, directory, rows, events = _strict_store(frozen, monkeypatch)
    if kind == "missing": rows.pop()
    elif kind == "duplicate": rows[-1] = rows[0]
    elif kind == "technical": events = [e for e in events if not (e["event"] == "TECHNICAL_PASS" and e["sample_id"] == "s00")]
    elif kind == "retry": events.append({"event": "INFLIGHT", "sample_id": "s00", "attempt_id": "second"})
    else: events = [e for e in events if not (e["event"] == "REPRO_INPUT_HASH" and e["sample_id"] == "s00" and e["stage"] == "gpu_forward")]
    _journal(directory / "outputs.jsonl", rows)
    _journal(directory / "events.jsonl", events)
    with pytest.raises(ValueError):
        ref.analyze(frozen.path, output, "nemo")
