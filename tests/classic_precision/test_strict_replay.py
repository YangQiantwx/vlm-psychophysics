"""CPU integration of the strict wrapper with the unchanged native run/store loop.

The model/processor and CUDA resource readings are explicit fixtures. These tests
exercise persistence and policy gates; they are not GPU reproducibility evidence.
"""
from copy import deepcopy
import contextlib
import json
import math
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from classic_precision import calibration, inference, reference_replay as reference
from classic_precision import reproducibility as repro, strict_replay as strict
from classic_precision.storage import atomic_json, digest, file_sha256
from classic_precision.summary import journal


def _runtime(family):
    c = repro._profile_contract()
    profile = {"profile_id": c["PROFILE_ID"], "family": family, "base_runner_sha256": c["BASE_RUNNER_SHA256"],
               "helper_sha256": c["helper_sha256"], "environment": c["ENVIRONMENT"], "sdk_files_modified": False}
    if family == "qwen":
        profile["qsa_installation"] = {"state": "applied", "expected_native_source_sha256": c["qsa"]["EXPECTED_SOURCE_SHA256"],
            "helper_source_sha256": c["qsa_helper_sha256"], "target_module": c["qsa"]["TARGET_MODULE"],
            "model_architecture": c["qsa"]["MODEL_ARCHITECTURE"]}
    return {"comparison_identity": {"profile": profile, "inference_sha256": c["BASE_RUNNER_SHA256"],
        "effective_environment": {**c["ENVIRONMENT"], **({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if family == "qwen" else {})},
        "deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False,
        "cudnn_deterministic": True, "fill_uninitialized_memory": c["FILL_UNINITIALIZED_MEMORY"][family],
        "fp32_precision": {"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"},
        "packages": {k: "CPU fixture" for k in ("torch", "transformers", "numpy", "av")},
        "gpu_name": "CPU fixture", "compute_capability": [12, 0], "cuda": "fixture", "driver": "fixture"}}


def _hashes(sid, family):
    def tensor(name, dtype, shape, size):
        return {"name": name, "dtype": dtype, "shape": shape, "bytes": math.prod(shape)*size,
                "sha256": digest([sid, name, dtype])}
    shape = [180, 3, 512, 768] if family == "nemo" else [138240, 1536]
    pixel = tensor("pixel_values_flat_video" if family == "nemo" else "pixel_values_videos",
                   "torch.bfloat16" if family == "nemo" else "torch.float32", shape, 2 if family == "nemo" else 4)
    return [{"stage": "cpu_processor", "tensors": [tensor("input_ids", "torch.int64", [1, 100], 8), pixel]},
            {"stage": "gpu_forward", "tensors": [tensor("pixels", "torch.bfloat16", shape, 2)]}]


def _raw(row, ready):
    text = "unchanged fixture prompt " + row["task"]
    return {**row, "row_identity": digest({"row": row, "ready": ready}), "ready_identity": digest(ready),
            "ready_record": ready, "effective_generation": {**inference.GENERATION, "backend": "fixture"},
            "actual_input_tokens": 100, "input_receipt": {"source_video_sha256": ready["video_sha256"],
                "source_stream_sha256": ready["stream_sha256"], "chat_template_text": text,
                "chat_template_sha256": digest(text), "input_tokens": 100,
                "processor_source": {"source_sha256": "a"*64}, "actual_processor_dtype": "torch.bfloat16",
                "actual_processor_tensor_shape": [180, 3, 512, 768]},
            "raw_answer": "LEFT", "output_token_ids": [11, 12], "finish_reason": "stop", "stop_reason": None}


@pytest.fixture
def panel(tmp_path, monkeypatch):
    source = tmp_path / "benchmark/runs/t2-t3-rgb16"
    source.mkdir(parents=True)
    rows = [{"sample_id": sid, "task": task, "truth": "RIGHT", "source_parameters": {"fixture": i}}
            for i, (sid, task) in enumerate((("T2-a", "T2"), ("T2-b", "T2"), ("T3-a", "T3"), ("T3-b", "T3")))]
    ready = {r["sample_id"]: {"sample_id": r["sample_id"], "video_sha256": digest([r["sample_id"], "video"]),
                              "stream_sha256": digest([r["sample_id"], "stream"])} for r in rows}
    resources = {"min_free_gpu_gib": 1, "min_available_ram_gib": 1}
    manifest = {"rows": rows, "models": {m: {"resources": resources} for m in strict.MODELS}}
    atomic_json(source / "manifest.json", manifest)
    atomic_json(source / "ready.json", ready)
    model_root = tmp_path / "models"
    model_root.mkdir()
    calls, leases, trace = [], [], []
    state = SimpleNamespace(source=source, rows=rows, ready=ready, calls=calls, leases=leases, trace=trace,
                            family="nemo", fail_sid=None, bad_hash=False, bad_answer=None, stop_after=None,
                            runtime=None, qualified=True)

    def inputs(path, ready_path=None, asset_root=None, include_catalog=False):
        result = (manifest, rows, ready, source, source / "ready.json")
        return result + (SimpleNamespace(refresh=lambda: ready),) if include_catalog else result

    def identity(path, ready_path, actual_manifest, family, root, resources=None, cosmos_package=None):
        return {"manifest_sha256": file_sha256(path), "model_key": family, "all_row_ids_sha256": digest([r["sample_id"] for r in rows]),
                "checkpoint_assets": {"fixture": "unchanged"}, "effective_resources": manifest["models"][family]["resources"]}, model_root

    monkeypatch.setattr(reference, "MANIFEST_SHA256", file_sha256(source / "manifest.json"))
    monkeypatch.setattr(inference, "load_run_inputs", inputs)
    monkeypatch.setattr(inference, "build_run_identity", identity)
    monkeypatch.setattr(calibration, "calibration_identity", lambda *a, **k: {"fixture": "128 native states"})
    monkeypatch.setattr(calibration, "calibration_ready", lambda *a, **k: state.qualified)
    monkeypatch.setattr("classic_precision.ready.start_ready", lambda *a: True)
    monkeypatch.setattr(inference, "resource_snapshot", lambda *a: {"admitted": True, "budgets": resources})

    @contextlib.contextmanager
    def lease(root, torch):
        trace.append(("lease", str(root)))
        leases.append(Path(root))
        try:
            yield
        finally:
            leases.pop()

    monkeypatch.setattr(inference, "_gpu_lock", lease)

    class Session:
        def __init__(self, family, model_path, store, cosmos_package=None, resources=None):
            assert len(leases) == 2  # The real native loop is nested in the shared physical lease.
            self.family, self.store, self.context = family, store, {}
            trace.append(("load", family))

        def generate(self, row, observed, assets, durable):
            sid = row["sample_id"]
            calls.append(sid)
            if state.fail_sid == sid:
                raise RuntimeError("fixture no-output failure")
            self.store.event("REPRO_INPUT_HASH", sample_id=sid, **_hashes(sid, self.family)[0])
            raw = _raw(row, observed)
            if state.bad_answer == sid:
                raw.update(raw_answer="not a choice", finish_reason="length")
            committed = durable(raw)
            self.store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=committed["attempt_id"])
            if not state.bad_hash:
                self.store.event("REPRO_INPUT_HASH", sample_id=sid, **_hashes(sid, self.family)[1])
            if state.stop_after == sid:
                raise RuntimeError("fixture crash after durable output and all receipts")

        def close(self):
            trace.append(("close", self.family))

    monkeypatch.setattr(inference, "ModelSession", Session)
    torch = ModuleType("torch")
    runtime = ModuleType("classic_precision_determinism")
    runtime.activate = lambda family: (trace.append(("activate", family)), monkeypatch.setitem(sys.modules, "torch", torch))
    runtime.runtime_snapshot = lambda: deepcopy(state.runtime or _runtime(state.family))
    monkeypatch.setitem(sys.modules, "classic_precision_determinism", runtime)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(repro, "_prepare_runtime_modules", lambda family: trace.append(("prepare_runtime", family)))
    state.base_init, state.base_generate, state.base_identity = Session.__init__, Session.generate, identity

    def seal(family="nemo", ids=None):
        state.family = family
        state.path = tmp_path / ("plan-" + family + ".json")
        state.plan = strict.prepare(source / "manifest.json", source / "ready.json", source, model_root,
                                    source / "calibration", family, tmp_path / "new-run/outputs", ids)
        atomic_json(state.path, state.plan)
        for key, value in {**state.plan["environment"], **state.plan["native_environment"]}.items():
            monkeypatch.setenv(key, value) if value is not None else monkeypatch.delenv(key, raising=False)
        return state.path

    state.seal = seal
    state.fresh = lambda: monkeypatch.delitem(sys.modules, "torch", raising=False)
    return state


def _rewrite(path, change):
    records, _ = journal(path)
    change(records)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.mark.parametrize("family", strict.MODELS)
def test_native_loop_uses_one_engine_full_raw_strict_hashes_and_separate_identity(panel, family):
    path = panel.seal(family)
    result = strict.run(path, "0")
    assert result["status"] == "COMPLETE" and result["raw"] == 4
    assert result["normal"] == 4  # Wrong answers are accepted technically, not resampled.
    assert panel.calls == ["T2-a", "T3-a", "T2-b", "T3-b"]
    assert panel.trace.index(("prepare_runtime", family)) > panel.trace.index(("load", family))
    assert sum(e == ("load", family) for e in panel.trace) == 1 and not panel.leases
    records, _ = journal(Path(panel.plan["output"]) / family / "outputs.jsonl")
    assert all(r["raw_answer"] != r["truth"] and r["output_token_ids"] == [11, 12] for r in records)
    assert inference.ModelSession.__init__ is panel.base_init
    assert inference.ModelSession.generate is panel.base_generate
    assert inference.build_run_identity is panel.base_identity
    assert strict.verify(path)["attempts"]["retry_attempts"] == 0


def test_resume_one_remaining_inherits_fixed_two_qualifications_without_reasking(panel):
    path = panel.seal()
    panel.fail_sid = "T3-b"
    with pytest.raises(RuntimeError, match="no-output"):
        strict.run(path, "0")
    assert panel.calls == ["T2-a", "T3-a", "T2-b", "T3-b"]
    before = (Path(panel.plan["output"]) / "nemo/outputs.jsonl").read_bytes()
    panel.fresh()
    with pytest.raises(ValueError, match="explicit retry"):
        strict.run(path, "0")
    assert len(panel.calls) == 4
    panel.fail_sid = None
    panel.fresh()
    result = strict.run(path, "0", retry_failed=True)
    assert panel.calls == ["T2-a", "T3-a", "T2-b", "T3-b", "T3-b"]
    assert (Path(panel.plan["output"]) / "nemo/outputs.jsonl").read_bytes().startswith(before)
    assert result["raw"] == 4 and result["attempts"]["retry_attempts"] == 1
    panel.fresh()
    strict.run(path, "0")
    assert len(panel.calls) == 5  # Fully complete resume does not load or call a model.
    assert sum(e == ("load", "nemo") for e in panel.trace) == 2


def test_max_two_no_output_attempts_and_no_hidden_retry(panel):
    path = panel.seal()
    panel.fail_sid = "T2-a"
    for retry in (False, True):
        panel.fresh()
        with pytest.raises(RuntimeError, match="no-output"):
            strict.run(path, "0", retry_failed=retry)
    panel.fresh()
    with pytest.raises(ValueError, match="qualification is held"):
        strict.run(path, "0", retry_failed=True)
    assert panel.calls == ["T2-a", "T2-a"]


def test_missing_gpu_capture_preserves_raw_and_fails_before_next_call_or_retry(panel):
    path = panel.seal()
    panel.bad_hash = True
    with pytest.raises(ValueError, match="hash is missing"):
        strict.run(path, "0")
    directory = Path(panel.plan["output"]) / "nemo"
    raw, _ = journal(directory / "outputs.jsonl")
    assert len(raw) == 1 and raw[0]["output_token_ids"] == [11, 12]
    panel.fresh()
    with pytest.raises(ValueError, match="technical PASS"):
        strict.run(path, "0", retry_failed=True, retry_interrupted=True)
    assert panel.calls == ["T2-a"]


def test_first_two_invalid_raw_stops_but_later_invalid_raw_is_terminal(panel):
    path = panel.seal()
    panel.bad_answer = "T2-a"
    with pytest.raises(ValueError, match="Predetermined qualification"):
        strict.run(path, "0")
    assert panel.calls == ["T2-a", "T3-a"]
    panel.fresh()
    with pytest.raises(ValueError, match="technical PASS"):
        strict.run(path, "0", retry_failed=True)
    assert len(panel.calls) == 2


def test_later_invalid_length_answer_remains_in_denominator(panel):
    path = panel.seal()
    panel.bad_answer = "T2-b"
    result = strict.run(path, "0")
    assert result["raw"] == 4 and result["normal"] == 3
    panel.fresh()
    strict.run(path, "0")
    assert len(panel.calls) == 4


@pytest.mark.parametrize("change,match", [
    (lambda p: p["implementation"].update(wrapper_sha256="0"*64), "source drift"),
    (lambda p: p["native_identity"]["checkpoint_assets"].update(fixture="drift"), "checkpoint"),
    (lambda p: p.update(input_binding_sha256="0"*64), "scientific row/ready"),
    (lambda p: p["environment"].update(PYTHONHASHSEED="1"), "profile/environment"),
    (lambda p: p["policy"].update(max_attempts=3), "terminal/retry"),
    (lambda p: p.update(gpu_lock_root=p["output"]), "shared physical GPU lease"),
    (lambda p: p.update(output=str(Path(p["manifest"]["path"]).parent / "outputs")), "overlaps"),
    (lambda p: p.update(sample_ids=["T2-a", "T2-b"]), "include T2 and T3"),
    (lambda p: p.update(model="cosmos"), "schema/model"),
])
def test_source_profile_subset_retry_and_lease_drift_fail_before_any_call(panel, change, match):
    path = panel.seal()
    change(panel.plan)
    atomic_json(path, panel.plan)
    with pytest.raises(ValueError, match=match):
        strict.run(path, "0")
    assert not panel.calls


def test_original_calibration_guard_is_rechecked_before_loading(panel):
    path = panel.seal()
    panel.qualified = False
    with pytest.raises(ValueError, match="128-state CPU calibration"):
        strict.run(path, "0")
    assert not panel.calls and not panel.trace


def test_actual_ready_file_sha_drift_fails_before_model_loading(panel):
    path = panel.seal()
    (panel.source / "ready.json").write_text("{}\n")
    with pytest.raises(ValueError, match="source SHA/path"):
        strict.run(path, "0")
    assert not panel.calls and not panel.trace


def test_physical_lease_denial_and_resource_rejection_prevent_model_load(panel, monkeypatch):
    path = panel.seal()
    @contextlib.contextmanager
    def denied(root, torch):
        raise BlockingIOError("CPU fixture shared GPU lease is owned")
        yield
    monkeypatch.setattr(inference, "_gpu_lock", denied)
    with pytest.raises(BlockingIOError, match="lease is owned"):
        strict.run(path, "0")
    assert not panel.calls and ("load", "nemo") not in panel.trace


def test_native_resource_admission_rejection_prevents_model_load(panel, monkeypatch):
    path = panel.seal()
    monkeypatch.setattr(inference, "resource_snapshot", lambda *a: {"admitted": False})
    with pytest.raises(ValueError, match="Actual free VRAM/RAM"):
        strict.run(path, "0")
    assert not panel.calls and ("load", "nemo") not in panel.trace and not panel.leases


@pytest.mark.parametrize("field", ["cuda", "packages", "fill_uninitialized_memory"])
def test_actual_runtime_policy_is_required_even_when_all_rows_would_agree(panel, field):
    path = panel.seal()
    panel.runtime = _runtime("nemo")
    panel.runtime["comparison_identity"].pop(field)
    with pytest.raises(ValueError):
        strict.run(path, "0")
    assert not panel.calls and ("close", "nemo") in panel.trace


def test_qwen_runtime_preparation_precedes_required_applied_validation(panel, monkeypatch):
    path = panel.seal("qwen")
    panel.runtime = _runtime("qwen")
    panel.runtime["comparison_identity"]["profile"]["qsa_installation"]["state"] = "pending_target_import"
    def prepare(family):
        panel.trace.append(("prepare_runtime", family))
        panel.runtime["comparison_identity"]["profile"]["qsa_installation"]["state"] = "applied"
    monkeypatch.setattr(repro, "_prepare_runtime_modules", prepare)
    assert strict.run(path, "0")["raw"] == 4


def test_qwen_pending_qsa_rejected_without_scientific_call(panel):
    path = panel.seal("qwen")
    panel.runtime = _runtime("qwen")
    panel.runtime["comparison_identity"]["profile"]["qsa_installation"]["state"] = "pending_target_import"
    with pytest.raises(ValueError, match="not applied"):
        strict.run(path, "0")
    assert not panel.calls


@pytest.mark.parametrize("change,match", [
    (lambda events: events.__setitem__(slice(None), [e for e in events if e.get("stage") != "gpu_forward"]), "hash is missing"),
    (lambda events: next(e for e in events if e.get("stage") == "gpu_forward").pop("attempt_id"), "attempt/session"),
    (lambda events: next(e for e in events if e.get("stage") == "gpu_forward").update(strict_session_id="other"), "attempt/session"),
    (lambda events: events.__setitem__(slice(None), [e for e in events if e["event"] != "INFLIGHT"]), "finite native attempt"),
    (lambda events: next(e for e in events if e["event"] == "STRICT_RUNTIME")["snapshot"]["comparison_identity"].update(deterministic_algorithms=False), "deterministic_algorithms"),
])
def test_saved_raw_requires_both_actual_stages_same_attempt_session_and_strict_runtime(panel, change, match):
    path = panel.seal()
    strict.run(path, "0")
    _rewrite(Path(panel.plan["output"]) / "nemo/events.jsonl", change)
    with pytest.raises(ValueError, match=match):
        strict.verify(path)


def test_original_real_manifest_ids_and_science_are_unchanged():
    path = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/t2-t3-rgb16/manifest.json")
    if not path.exists():
        pytest.skip("Local original manifest fixture is not installed")
    assert file_sha256(path) == reference.MANIFEST_SHA256
    manifest = json.loads(path.read_text())
    rows = inference.select_rows(manifest["rows"])
    assert len(rows) == 3504 and len({r["sample_id"] for r in rows}) == 3504
    ordered, first = inference.qualification_order(rows)
    assert [r["task"] for r in first] == ["T2", "T3"]
    assert {r["sample_id"] for r in ordered} == {r["sample_id"] for r in rows}
    assert [r["sample_id"] for r in inference.select_rows(rows, [first[1]["sample_id"], first[0]["sample_id"]])] == [r["sample_id"] for r in first]


def test_cli_refuses_to_overwrite_sealed_plan(panel):
    path = panel.seal()
    with pytest.raises(ValueError, match="cannot be overwritten"):
        strict.main(["prepare", "--plan", str(path), "--model", "nemo", *sum((["--"+k, "unused"] for k in
                    ("manifest", "ready", "asset-root", "model-root", "calibration-root", "output")), [])])
