"""CPU-only certification checks; no checkpoint, Torch import or GPU required."""
from copy import deepcopy
import contextlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from classic_precision import reproducibility as repro
from classic_precision.inference import GENERATION
from classic_precision.storage import RunStore, atomic_json, digest, file_sha256


def _profile(family="cosmos"):
    contract = repro._profile_contract()
    value = {"profile_id": contract["PROFILE_ID"], "family": family,
             "base_runner_sha256": contract["BASE_RUNNER_SHA256"],
             "helper_sha256": contract["helper_sha256"], "environment": contract["ENVIRONMENT"],
             "sdk_files_modified": False}
    if family == "qwen":
        qsa = contract["qsa"]
        value["qsa_installation"] = {"state": "applied", "expected_native_source_sha256": qsa["EXPECTED_SOURCE_SHA256"],
            "helper_source_sha256": contract["qsa_helper_sha256"], "target_module": qsa["TARGET_MODULE"],
            "model_architecture": qsa["MODEL_ARCHITECTURE"]}
    return value


def _snapshot(family="cosmos"):
    profile = _profile(family)
    return {"comparison_identity": {"profile": profile, "packages": {"torch": "pinned", "transformers": "pinned",
            "numpy": "pinned", "av": "pinned"}, "gpu_name": "same GPU", "compute_capability": [10, 0],
            "cuda": "13", "driver": "pinned", "inference_sha256": profile["base_runner_sha256"],
            "effective_environment": {**profile["environment"], **({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if family == "qwen" else {})},
            "deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "fill_uninitialized_memory": repro._profile_contract()["FILL_UNINITIALIZED_MEMORY"][family],
            "fp32_precision": {"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"}}}


def _tensor(name, shape, dtype, size, sha="b" * 64):
    import math
    return {"name": name, "shape": shape, "dtype": dtype, "bytes": math.prod(shape) * size, "sha256": sha}


@pytest.mark.parametrize("family", ["nemo", "qwen", "cosmos", "gemma"])
def test_actual_fill_policy_is_required_per_family(family):
    snapshot = _snapshot(family)
    repro.validate_runtime_policy(snapshot, family)
    value = snapshot["comparison_identity"]
    value["fill_uninitialized_memory"] = not value["fill_uninitialized_memory"]
    with pytest.raises(ValueError, match="fill_uninitialized_memory"):
        repro.validate_runtime_policy(snapshot, family)
    del value["fill_uninitialized_memory"]
    with pytest.raises(ValueError, match="fill_uninitialized_memory"):
        repro.validate_runtime_policy(snapshot, family)


def _tensor_receipt(stage):
    if stage == "cpu_processor":
        return [_tensor("input_ids", [1, 100], "torch.int64", 8),
                _tensor("pixel_values_videos", [138240, 1536], "torch.float32", 4)]
    return [_tensor("pixels", [138240, 1536], "torch.bfloat16", 2)]


@pytest.fixture
def panel(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    manifest = source / "manifest.json"
    rows = [{"sample_id": sid, "task": sid[:2], "truth": "RIGHT", "source_parameters": {"condition": 1}}
            for sid in ("T2-a", "T3-b")]
    atomic_json(manifest, {"rows": rows})
    ready = source / "ready.jsonl"
    ready.write_text(''.join(json.dumps({"sample_id": row["sample_id"], "video_sha256": "e" * 64}) + '\n' for row in rows))
    plan = {"schema": repro.SCHEMA, "source_run": str(source), "manifest": str(manifest),
            "manifest_sha256": file_sha256(manifest), "ready": str(ready), "asset_root": str(source),
            "profile": {"id": repro._profile_contract()["PROFILE_ID"], "definition_sha256": "a" * 64},
            "models": ["cosmos"], "sample_ids": ["T2-a", "T3-b"], "cold_passes": 2, "repeats": 2}
    contract = repro._profile_contract()
    plan["profile"].update(implementation={"determinism_helper_sha256": contract["helper_sha256"],
        "qsa_helper_sha256": contract["qsa_helper_sha256"], "base_runner_sha256": contract["BASE_RUNNER_SHA256"],
        "harness_sha256": file_sha256(repro.__file__)}, environment=contract["ENVIRONMENT"],
        model_environment={"qwen": {"CLASSIC_PRECISION_QSA_DETERMINISM": "1"}})
    path = tmp_path / "plan.json"
    atomic_json(path, plan)
    return path, plan, tmp_path / "outputs"


def _populate(panel, mutation=None):
    path, plan, output = panel
    plan_hash = digest(plan)
    originals = {r["sample_id"]: r for r in json.loads(Path(plan["manifest"]).read_text())["rows"]}
    ready = {r["sample_id"]: r for r in repro._read_jsonl(plan["ready"])}
    for cold in range(2):
        for repeat in range(2):
            identity = {"model_key": "cosmos", "purpose": repro.SCHEMA,
                "manifest_sha256": file_sha256(plan["manifest"]), "runtime_versions": {"torch": "pinned"},
                "reproducibility": {"plan_sha256": plan_hash, "profile": plan["profile"],
                    "profile_status": _profile(),
                    "cold_pass": cold, "repeat": repeat, "sample_ids": plan["sample_ids"],
                    "harness_sha256": file_sha256(repro.__file__)}}
            dest = repro._repeat_directory(output, cold, repeat)
            with RunStore(dest, identity, "cosmos", "repro", plan["sample_ids"]) as store:
                # Synthetic owners distinguish cold starts without starting
                # processes, while repeats of one cold pass share an owner.
                events = store.directory / "events.jsonl"
                first = json.loads(events.read_text())
                first["owner"].update(pid=100 + cold, process_start_ticks=str(cold), boot_id="fixture")
                events.write_text(json.dumps(first) + "\n")
                snapshot = {**_snapshot(), "pid": 100 + cold, "gpu_uuid": str(cold)}
                value = repro.runtime_identity(snapshot)
                store.event("REPRO_RUNTIME", snapshot=snapshot, comparison_identity=value,
                            comparison_sha256=digest(value))
                for sid in plan["sample_ids"]:
                    store.begin(sid, "row")
                    for stage in ("cpu_processor", "gpu_forward"):
                        store.event("REPRO_INPUT_HASH", sample_id=sid, stage=stage,
                                    tensors=_tensor_receipt(stage))
                    record = {"raw_answer": "LEFT", "output_token_ids": [10, 11], "finish_reason": "stop",
                        "task": sid[:2], "truth": "RIGHT", "row_identity": digest({"row": originals[sid], "ready": ready[sid]}),
                        "ready_identity": digest(ready[sid]), "ready_record": ready[sid],
                        "source_parameters": {"condition": 1}, "effective_generation": deepcopy(GENERATION),
                        "actual_input_tokens": 100, "input_receipt": {"chat_template_sha256": "f" * 64},
                        "parent_sample_id": sid, "reproducibility": {"plan_sha256": plan_hash,
                            "cold_pass": cold, "repeat": repeat}}
                    result = store.commit_output(sid, record)
                    store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=result["attempt_id"])
                store.event("REPRO_REPEAT_COMPLETE", status="COMPLETE")
                if mutation is not None and cold == 1 and repeat == 1:
                    mutation(store.directory)
        atomic_json(output / ("cold-" + str(cold)) / "cosmos.json",
                    {"status": "COMPLETE", "plan_sha256": plan_hash})


def _rewrite(path, change):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    change(records)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


def test_four_stores_exact_inputs_tokens_and_runtime_pass(panel):
    _populate(panel)
    report = repro.compare(panel[0], panel[2])
    assert report["status"] == "PASS"
    assert report["models"]["cosmos"]["checked_repeat_stores"] == 4


@pytest.mark.parametrize("field,value", [("output_token_ids", [10, 12]), ("raw_answer", "RIGHT"),
                                         ("actual_input_tokens", 101), ("effective_generation", {"seed": 1})])
def test_same_short_answer_does_not_hide_other_mismatches(panel, field, value):
    def change(directory):
        _rewrite(directory / "outputs.jsonl", lambda rows: rows[0].update({field: value}))
    _populate(panel, change)
    report = repro.compare(panel[0], panel[2])
    assert report["status"] == "FAIL"
    assert field in report["models"]["cosmos"]["mismatches"][0]["error"]


@pytest.mark.parametrize("kind", ["hash", "missing_gpu", "technical_failure", "runtime", "duplicate", "tail", "same_process"])
def test_incomplete_or_inconsistent_evidence_fails_closed(panel, kind):
    def change(directory):
        if kind == "duplicate":
            path = directory / "outputs.jsonl"
            path.write_bytes(path.read_bytes() + path.read_bytes().splitlines(keepends=True)[0])
        elif kind == "tail":
            path = directory / "outputs.jsonl"
            path.write_bytes(path.read_bytes() + b'{"sample_id":')
        else:
            def mutate(events):
                if kind == "hash":
                    next(e for e in events if e.get("event") == "REPRO_INPUT_HASH")["tensors"][0]["sha256"] = "0" * 64
                elif kind == "missing_gpu":
                    events[:] = [e for e in events if e.get("stage") != "gpu_forward"]
                elif kind == "technical_failure":
                    event = deepcopy(next(e for e in events if e.get("event") == "TECHNICAL_PASS"))
                    event.update(event="POST_OUTPUT_VALIDATION_FAILED", time_unix=10 ** 12)
                    events.append(event)
                elif kind == "runtime":
                    event = next(e for e in events if e.get("event") == "REPRO_RUNTIME")
                    event["snapshot"]["comparison_identity"]["packages"]["torch"] = "changed"
                    event["comparison_identity"] = event["snapshot"]["comparison_identity"]
                    event["comparison_sha256"] = digest(event["comparison_identity"])
                elif kind == "same_process":
                    next(e for e in events if e.get("event") == "WORKER_START")["owner"]["pid"] = 999
            _rewrite(directory / "events.jsonl", mutate)
    _populate(panel, change)
    assert repro.compare(panel[0], panel[2])["status"] == "FAIL"


def test_missing_cold_pass_and_changed_plan_cannot_be_complete(panel):
    _populate(panel)
    (panel[2] / "cold-1" / "cosmos.json").unlink()
    assert repro.compare(panel[0], panel[2])["status"] == "FAIL"
    plan = deepcopy(panel[1])
    plan["profile"]["definition_sha256"] = "0" * 64
    atomic_json(panel[0], plan)
    assert repro.compare(panel[0], panel[2])["status"] == "FAIL"


def test_plan_requires_unique_targets_and_two_by_two_design(panel):
    path, plan, _ = panel
    for changes in ({"sample_ids": ["a", "a"]}, {"repeats": 1}, {"cold_passes": 1}):
        atomic_json(path, {**plan, **changes})
        with pytest.raises(ValueError):
            repro.load_plan(path)


def test_snapshot_ignores_only_explicit_process_location(panel):
    a = {"comparison_identity": {"gpu_name": "B200", "packages": {"torch": "2.13"}}, "pid": 1}
    b = {**a, "pid": 2, "gpu_uuid": "another"}
    assert repro.runtime_identity(a) == repro.runtime_identity(b)
    b = deepcopy(b)
    b["comparison_identity"]["gpu_name"] = "H200"
    assert repro.runtime_identity(a) != repro.runtime_identity(b)


def _mock_run(monkeypatch, panel, fail_sample=None, calibration_ok=True):
    from classic_precision import inference, calibration
    _, plan, _ = panel
    state = {"loads": 0, "closes": 0, "calls": [], "guards": []}
    rows = json.loads(Path(plan["manifest"]).read_text())["rows"]
    ready = {r["sample_id"]: r for r in repro._read_jsonl(plan["ready"])}
    resources = {"min_free_gpu_gib": 80, "min_available_ram_gib": 20}
    manifest = {"rows": rows, "models": {"cosmos": {"resources": resources}}}
    monkeypatch.setattr(inference, "load_run_inputs", lambda *a: (manifest, rows, ready,
                        Path(plan["asset_root"]), Path(plan["ready"])))
    monkeypatch.setattr(calibration, "calibration_identity", lambda *a: state["guards"].append("calibration_identity") or {})
    monkeypatch.setattr(calibration, "calibration_ready", lambda *a: state["guards"].append("calibration_ready") or calibration_ok)
    monkeypatch.setattr(inference, "build_run_identity", lambda *a, **k: ({"model_key": "cosmos",
                        "manifest_sha256": file_sha256(plan["manifest"])}, Path(plan["source_run"])))

    @contextlib.contextmanager
    def lock(path, torch):
        state["guards"].append(("lock", str(path)))
        yield

    monkeypatch.setattr(inference, "_gpu_lock", lock)
    monkeypatch.setattr(inference, "resource_snapshot", lambda *a: state["guards"].append("resources") or {"admitted": True})
    monkeypatch.setattr(inference, "_qualification_marker", lambda *a: True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    runtime = SimpleNamespace(activate=lambda family: state["guards"].append("activate"),
        status=lambda: _profile(), runtime_snapshot=lambda: _snapshot())
    monkeypatch.setitem(sys.modules, "classic_precision_determinism", runtime)

    class Session:
        def __init__(self, family, model_path, sink, cosmos_package, resources):
            state["loads"] += 1
            self.store, self.context = sink, {}
            sink.event("MODEL_READY", load_seconds=1.0)

        def generate(self, row, ready, assets, commit):
            sid = row["sample_id"]
            state["calls"].append(sid)
            if sid == fail_sample:
                raise RuntimeError("unsupported deterministic kernel")
            for stage in ("cpu_processor", "gpu_forward"):
                self.store.event("REPRO_INPUT_HASH", sample_id=sid, stage=stage,
                                 tensors=_tensor_receipt(stage))
            result = commit({"task": row["task"], "truth": row["truth"],
                "source_parameters": row["source_parameters"],
                "row_identity": digest({"row": row, "ready": ready}), "ready_identity": digest(ready),
                "ready_record": ready, "raw_answer": "LEFT", "output_token_ids": [4], "finish_reason": "stop",
                "actual_input_tokens": 100, "input_receipt": {"chat_template_sha256": "b" * 64},
                "effective_generation": deepcopy(GENERATION), "seconds": 0.1})
            self.store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=result["attempt_id"])

        def close(self):
            state["closes"] += 1

    monkeypatch.setattr(inference, "ModelSession", Session)
    return state


def test_run_reuses_one_engine_rotates_stores_and_guards_source(monkeypatch, panel):
    state = _mock_run(monkeypatch, panel)
    result = repro.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
    assert result["status"] == "COMPLETE"
    assert state["loads"] == state["closes"] == 1
    assert state["calls"] == ["T2-a", "T3-b", "T2-a", "T3-b"]
    assert state["guards"][:3] == ["calibration_identity", "calibration_ready", "activate"]
    assert ("lock", str(Path(panel[1]["source_run"]) / "outputs")) in state["guards"]
    for repeat in range(2):
        path = repro._repeat_directory(panel[2], 0, repeat) / "cosmos"
        records = repro._read_jsonl(path / "outputs.jsonl")
        assert len(records) == 2
        assert {r["reproducibility"]["repeat"] for r in records} == {repeat}
    with pytest.raises(FileExistsError):
        repro.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
    assert len(state["calls"]) == 4


def test_runtime_failure_preserves_partial_pass_and_never_retries(monkeypatch, panel):
    state = _mock_run(monkeypatch, panel, fail_sample="T3-b")
    with pytest.raises(RuntimeError, match="unsupported deterministic"):
        repro.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
    assert state["calls"] == ["T2-a", "T3-b"] and state["closes"] == 1
    result = json.loads((panel[2] / "cold-0" / "cosmos.json").read_text())
    assert result["status"] == "FAILED"
    assert not repro._repeat_directory(panel[2], 0, 1).exists()


def test_missing_original_calibration_prevents_activation_and_model_load(monkeypatch, panel):
    state = _mock_run(monkeypatch, panel, calibration_ok=False)
    with pytest.raises(ValueError, match="128-state"):
        repro.run(panel[0], panel[2], "cosmos", 0, "0", panel[1]["source_run"])
    assert state["loads"] == 0 and "activate" not in state["guards"]


@pytest.mark.parametrize("change", ["determinism_off", "warn_only", "benchmark", "cudnn", "tf32", "missing_env", "changed_env", "wrong_helper", "fallback", "pending_qsa"])
def test_equal_but_relaxed_runtime_is_rejected(change):
    family = "qwen" if change == "pending_qsa" else "cosmos"
    snapshot = _snapshot(family)
    runtime = snapshot["comparison_identity"]
    if change == "determinism_off": runtime["deterministic_algorithms"] = False
    elif change == "warn_only": runtime["deterministic_warn_only"] = True
    elif change == "benchmark": runtime["cudnn_benchmark"] = True
    elif change == "cudnn": runtime["cudnn_deterministic"] = False
    elif change == "tf32": runtime["fp32_precision"]["matmul"] = "tf32"
    elif change == "missing_env": runtime.pop("effective_environment")
    elif change == "changed_env": runtime["effective_environment"]["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    elif change == "wrong_helper": runtime["profile"]["helper_sha256"] = "0" * 64
    elif change == "fallback": snapshot = runtime
    elif change == "pending_qsa": runtime["profile"]["qsa_installation"]["state"] = "pending_target_import"
    with pytest.raises(ValueError):
        repro.validate_runtime_policy(snapshot, family)


@pytest.mark.parametrize("change", ["metadata_sha", "empty", "bad_bytes", "bad_dtype", "bad_shape", "no_input_ids", "gpu_wrong_pixels"])
def test_metadata_and_malformed_tensor_hashes_cannot_certify_input(change):
    cpu = {"event": "REPRO_INPUT_HASH", "sample_id": "a", "stage": "cpu_processor", "tensors": _tensor_receipt("cpu_processor")}
    gpu = {"event": "REPRO_INPUT_HASH", "sample_id": "a", "stage": "gpu_forward", "tensors": _tensor_receipt("gpu_forward")}
    if change == "metadata_sha":
        cpu.pop("tensors")
        cpu["not_a_tensor_sha256"] = "a" * 64
    elif change == "empty": cpu["tensors"] = []
    elif change == "bad_bytes": cpu["tensors"][0]["bytes"] += 1
    elif change == "bad_dtype": cpu["tensors"][0]["dtype"] = "torch.not_a_dtype"
    elif change == "bad_shape": cpu["tensors"][0]["shape"] = [1, -1]
    elif change == "no_input_ids": cpu["tensors"].pop(0)
    elif change == "gpu_wrong_pixels": gpu["tensors"][0]["name"] = "metadata"
    with pytest.raises(ValueError):
        repro._hash_receipts([cpu, gpu], "a", "unused", "cosmos")


def test_plan_cannot_claim_another_implementation(panel):
    path, plan, _ = panel
    plan["profile"]["implementation"]["harness_sha256"] = "0" * 64
    atomic_json(path, plan)
    with pytest.raises(ValueError, match="implementation"):
        repro.load_plan(path)


def test_identically_wrong_generation_in_every_repeat_still_fails(panel):
    _populate(panel)
    for cold in range(2):
        for repeat in range(2):
            path = repro._repeat_directory(panel[2], cold, repeat) / "cosmos" / "outputs.jsonl"
            _rewrite(path, lambda rows: [r["effective_generation"].update(seed=99) for r in rows])
    report = repro.compare(panel[0], panel[2])
    assert report["status"] == "FAIL"
    assert all("GENERATION" in m["error"] for m in report["models"]["cosmos"]["mismatches"])


def test_activation_status_cannot_be_an_arbitrary_equal_dict(panel):
    _populate(panel)
    for cold in range(2):
        for repeat in range(2):
            path = repro._repeat_directory(panel[2], cold, repeat) / "cosmos" / "identity.json"
            identity = json.loads(path.read_text())
            identity["reproducibility"]["profile_status"] = {"profile_id": panel[1]["profile"]["id"]}
            atomic_json(path, identity)
    assert repro.compare(panel[0], panel[2])["status"] == "FAIL"


def test_qwen_lazy_module_is_guarded_before_pre_call_policy(monkeypatch):
    calls = []
    state = {"state": "pending_target_import"}
    helper = SimpleNamespace(TARGET_MODULE="native.qsa.leaf", status=lambda: dict(state))
    def importer(name):
        calls.append(name)
        if name == "classic_precision_qsa_determinism":
            return helper
        assert name == "native.qsa.leaf"
        state["state"] = "applied"
        return SimpleNamespace()
    monkeypatch.setattr(repro.importlib, "import_module", importer)
    repro._prepare_runtime_modules("qwen")
    assert calls == ["classic_precision_qsa_determinism", "native.qsa.leaf"]


@pytest.mark.parametrize("state", [None, {}, {"state": "pending_target_import"}])
def test_qwen_guarded_import_cannot_bypass_applied_gate(monkeypatch, state):
    helper = SimpleNamespace(TARGET_MODULE="native.qsa.leaf", status=lambda: state)
    monkeypatch.setattr(repro.importlib, "import_module", lambda name: helper)
    with pytest.raises(ValueError, match="did not apply"):
        repro._prepare_runtime_modules("qwen")


@pytest.mark.parametrize("family", ["nemo", "cosmos", "gemma"])
def test_guarded_qsa_preimport_is_qwen_only(monkeypatch, family):
    def unexpected(name):
        raise AssertionError("QSA import must not affect other models")
    monkeypatch.setattr(repro.importlib, "import_module", unexpected)
    repro._prepare_runtime_modules(family)


def test_native_source_guard_error_is_not_hidden(monkeypatch):
    helper = SimpleNamespace(TARGET_MODULE="native.qsa.leaf", status=lambda: {"state": "applied"})
    def importer(name):
        if name == "classic_precision_qsa_determinism":
            return helper
        raise RuntimeError("native source identity mismatch")
    monkeypatch.setattr(repro.importlib, "import_module", importer)
    with pytest.raises(RuntimeError, match="native source identity mismatch"):
        repro._prepare_runtime_modules("qwen")
