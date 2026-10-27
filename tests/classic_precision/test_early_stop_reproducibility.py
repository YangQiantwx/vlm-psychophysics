"""CPU-only behavioral checks for the bounded adjacent-repeat entry."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
import shutil
import sys
from types import SimpleNamespace

import pytest

from classic_precision import early_stop_reproducibility as early
from classic_precision.inference import GENERATION
from classic_precision.storage import atomic_json, digest, file_sha256


def _helpers():
    path = Path(__file__).with_name("test_qsa_only_reproducibility.py")
    spec = importlib.util.spec_from_file_location("early_qsa_test_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def panel(tmp_path):
    helper = _helpers()
    _, source, _ = helper._old()._helpers().panel.__wrapped__(tmp_path)
    # Match Fourier's existing native baseline, including unset CPU offload.
    baseline = {"NUMPY_MADVISE_HUGEPAGE": "0", "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB": None,
                "VLLM_BATCH_INVARIANT": None, "PATH": "/qy/bin:/usr/bin:/bin"}
    helper.BASELINE = baseline
    plan = early.prepare_plan(source, "gpu0", "0", "GPU-c89c0c8c-8081-5ba9-c73f-d08af5875193", baseline_environment=baseline)
    path = tmp_path / "early.json"
    atomic_json(path, plan)
    return helper, path, plan, tmp_path / "output"


def test_exact_source_plan_and_49_sample_contract(panel):
    helper, path, plan, output = panel
    assert early.load_plan(path) == (plan, digest(plan))
    assert plan["profile"]["torch_policy"] == {}
    assert plan["profile"]["environment_interventions"] == {early.qsa.runtime.FLAG: "1"}
    assert plan["profile"]["baseline_environment"]["CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB"] is None
    assert plan["sample_ids"][:8] == plan["prefix_ids"] and len(plan["sample_ids"]) == 49
    assert plan["auxiliary_T3_qualification_id"] in plan["prefix_ids"]
    assert plan["total_model_calls"] == 98
    snapshot = helper._snapshot()
    snapshot.update(gpu_uuid="GPU-c89c0c8c-8081-5ba9-c73f-d08af5875193", visible_devices="0", pid=100, hostname="fixture")
    assert early._snapshot_identity(snapshot, plan)["policy"]["profile"]["torch_policy"] == {}


@pytest.mark.parametrize("change", ["source", "budget", "prefix", "profile", "barrier"])
def test_tampered_plan_fails_closed(panel, change):
    _, path, plan, _ = panel
    if change == "source": plan["implementation"]["entry_sha256"] = "0" * 64
    elif change == "budget": plan["total_model_calls"] = 196
    elif change == "prefix": plan["prefix_ids"][0] = "missing"
    elif change == "profile": plan["profile"]["torch_policy"] = {"deterministic_algorithms": True}
    else: plan["comparison_contract"]["prefix_barrier"] = "skip"
    atomic_json(path, plan)
    with pytest.raises(ValueError): early.load_plan(path)


def test_claim_is_complete_at_first_visibility_and_existing_claim_is_preserved(monkeypatch, tmp_path):
    path = tmp_path / ".qwen.claim.json"
    value = {"plan_sha256": "f" * 64, "owner": {"pid": 101}}
    real_link = early.os.link
    observed = []
    def publish(source, destination):
        assert not path.exists()
        assert json.loads(Path(source).read_text()) == value
        real_link(source, destination)
        observed.append(json.loads(path.read_text()))
    monkeypatch.setattr(early.os, "link", publish)
    early._publish_claim(path, value)
    assert observed == [value] and list(tmp_path.iterdir()) == [path]
    monkeypatch.setattr(early.os, "link", real_link)
    with pytest.raises(FileExistsError): early._publish_claim(path, {"changed": True})
    assert json.loads(path.read_text()) == value and list(tmp_path.iterdir()) == [path]


def _fake_runtime(monkeypatch, panel, behavior=None, observed_uuid="GPU-c89c0c8c-8081-5ba9-c73f-d08af5875193"):
    helper, path, plan, output = panel
    from classic_precision import inference, calibration, storage
    owner = {"host": "fixture", "boot_id": "fixture-boot", "pid": 101, "process_start_ticks": "12", "started_unix": 1}
    monkeypatch.setattr(early, "owner_identity", lambda: deepcopy(owner))
    monkeypatch.setattr(storage, "owner_identity", lambda: deepcopy(owner))
    rows = json.loads(Path(plan["manifest"]).read_text())["rows"]
    ready = {row["sample_id"]: row for row in early.original._read_jsonl(plan["ready"])}
    manifest = {"models": {"qwen": {"resources": {"min_free_gpu_gib": 1, "min_available_ram_gib": 1}}}}
    monkeypatch.setattr(inference, "load_run_inputs", lambda *a: (manifest, rows, ready, Path(plan["asset_root"]), Path(plan["ready"])))
    native = {"model_key": "qwen", "manifest_sha256": file_sha256(plan["manifest"])}
    monkeypatch.setattr(inference, "build_run_identity", lambda *a, **kw: (native, Path("/fixture/model")))
    monkeypatch.setattr(calibration, "calibration_identity", lambda *a: {})
    monkeypatch.setattr(calibration, "calibration_ready", lambda *a: True)
    monkeypatch.setattr(inference, "_gpu_lock", lambda *a: __import__("contextlib").nullcontext())
    monkeypatch.setattr(inference, "resource_snapshot", lambda *a: {"admitted": True})
    monkeypatch.setattr(inference, "_qualification_marker", lambda *a: True)
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda _: SimpleNamespace(uuid=observed_uuid)))
    # run requires a fresh process; install the fake only after activation.
    def activate(*a): monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(early.qsa.runtime, "activate", activate)
    monkeypatch.setattr(early.qsa.runtime, "status", lambda: helper._profile("qwen_qsa_only"))
    monkeypatch.setattr(early.qsa.runtime, "prepare_native", lambda *a: None)
    seen, signals, instances = set(), {}, []
    monkeypatch.setattr(early.signal, "signal", lambda number, handler: signals.setdefault(number, handler))
    calls = []
    tensors = helper._old()._helpers()._helpers("test_config_only_reproducibility")._tensors
    def snapshot():
        value = helper._snapshot()
        value.update(gpu_uuid=observed_uuid, visible_devices="0", pid=owner["pid"], hostname=owner["host"])
        if instances:
            store = instances[0].store.store
            if str(store.directory) not in seen:
                store.event("REPRO_RUNTIME_OBSERVED", evidence_contract=early.qsa.runtime.EVIDENCE_CONTRACT, snapshot=value)
                seen.add(str(store.directory))
        return value
    monkeypatch.setattr(early.qsa.runtime, "runtime_snapshot", snapshot)
    class Session:
        def __init__(self, family, model, sink, package, resources):
            self.store, self.context = sink, {}
            instances.append(self)
            self.store.event("REPRO_ENGINE_LAUNCH_PARENT", receipt=helper._snapshot()["comparison_identity"]["engine_launch_parent"])
            self.store.event("MODEL_READY", load_seconds=.01)
        def generate(self, row, ready, assets, commit):
            sid = row["sample_id"]
            repeat = int(self.store.store.directory.parent.name[-1])
            calls.append((sid, repeat))
            self.store.event("REPRO_CALL_RUNTIME", sample_id=sid, stage="before", snapshot=snapshot())
            self.store.event("REPRO_INPUT_HASH", sample_id=sid, stage="cpu_processor", tensors=tensors("cpu_processor", "qwen"))
            record = {"task": row["task"], "truth": row["truth"], "source_parameters": row["source_parameters"],
                "ready_record": ready, "ready_identity": digest(ready), "row_identity": digest({"row": row, "ready": ready}),
                "input_receipt": {"chat_template_sha256": "f" * 64}, "actual_input_tokens": 100,
                "effective_generation": deepcopy(GENERATION), "raw_answer": "LEFT", "output_token_ids": [10, 11],
                "finish_reason": "stop", "stop_reason": None, "seconds": .01}
            if behavior: behavior("before_commit", calls, record, self, signals)
            committed = commit(record)
            self.store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=committed["attempt_id"])
            self.store.event("REPRO_CALL_RUNTIME", sample_id=sid, stage="after", snapshot=snapshot())
            gpu = tensors("gpu_forward", "qwen")
            if behavior: behavior("after_commit", calls, gpu, self, signals)
            self.store.event("REPRO_INPUT_HASH", sample_id=sid, stage="gpu_forward", tensors=gpu)
        def close(self): pass
    monkeypatch.setattr(inference, "ModelSession", Session)
    for key in plan["profile"]["cleared_environment"]: monkeypatch.delenv(key, raising=False)
    for key, value in plan["profile"]["environment"].items(): monkeypatch.setenv(key, value)
    return calls, instances, signals


def test_first_full_output_difference_stops_exactly_after_two_requests(monkeypatch, panel):
    def behavior(stage, calls, payload, session, signals):
        if stage == "before_commit" and len(calls) == 2: payload["output_token_ids"] = [10, 12]
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    _, path, plan, output = panel
    result = early.run(path, output, "0", "/fixture")
    assert result["comparison_status"] == "DIFFERENT" and result["evidence_status"] == "COMPLETE"
    assert calls == [(plan["sample_ids"][0], 0), (plan["sample_ids"][0], 1)]
    assert result["model_calls"] == 2 and result["first_difference"]["different_fields"] == ["output_token_ids"]
    assert json.loads((output / "early-stop.json").read_text()) == result
    assert all((early._directory(output, r) / "outputs.jsonl").is_file() for r in range(2))


def test_changed_actual_input_stops_invalid_without_third_request(monkeypatch, panel):
    def behavior(stage, calls, payload, session, signals):
        if stage == "after_commit" and len(calls) == 2: payload[0]["sha256"] = "0" * 64
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    _, path, _, output = panel
    with pytest.raises(ValueError, match="input"):
        early.run(path, output, "0", "/fixture")
    assert len(calls) == 2
    assert json.loads((output / "early-stop.json").read_text())["comparison_status"] == "INVALID_EVIDENCE"


def test_per_call_actual_runtime_drift_stops_before_another_request(monkeypatch, panel):
    def behavior(stage, calls, payload, session, signals):
        if stage == "after_commit":
            path = session.store.store.directory / "events.jsonl"
            events = early.original._read_jsonl(path)
            event = next(e for e in events if e.get("event") == "REPRO_CALL_RUNTIME" and e.get("stage") == "after")
            event["snapshot"]["comparison_identity"]["qsa_live_observation"]["wrapper_code_exact"] = False
            path.write_text("".join(json.dumps(e) + "\n" for e in events))
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    _, path, _, output = panel
    with pytest.raises(ValueError, match="live code"):
        early.run(path, output, "0", "/fixture")
    assert len(calls) == 1
    assert json.loads((output / "early-stop.json").read_text())["comparison_status"] == "INVALID_EVIDENCE"


def test_signal_finishes_inflight_raw_and_technical_before_stop(monkeypatch, panel):
    def behavior(stage, calls, payload, session, signals):
        if stage == "before_commit": signals[signal.SIGTERM](signal.SIGTERM, None)
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    _, path, _, output = panel
    result = early.run(path, output, "0", "/fixture")
    assert len(calls) == 1 and result["comparison_status"] == "STOPPED" and result["elapsed_seconds"] >= 0
    events = early.original._read_jsonl(early._directory(output, 0) / "events.jsonl")
    assert any(e["event"] == "TECHNICAL_PASS" for e in events)
    assert any(e["event"] == "EARLY_CALL_COMPLETE" and e["elapsed_seconds"] >= 0 for e in events)
    assert len(early.original._read_jsonl(early._directory(output, 0) / "outputs.jsonl")) == 1


def test_stable_arm_cannot_exceed_98_calls_and_order_is_adjacent(monkeypatch, panel):
    calls, _, _ = _fake_runtime(monkeypatch, panel)
    _, path, plan, output = panel
    result = early.run(path, output, "0", "/fixture")
    assert result["comparison_status"] == "MATCH" and result["model_calls"] == 98
    assert calls == [(sid, repeat) for sid in plan["sample_ids"] for repeat in range(2)]
    with pytest.raises((FileExistsError, ValueError)):
        early.run(path, output, "0", "/fixture")


def test_cross_arm_only_allows_declared_hardware_process_differences(monkeypatch, panel):
    calls, _, _ = _fake_runtime(monkeypatch, panel)
    _, path, plan, output = panel
    early.run(path, output, "0", "/fixture")
    sources = early.original._source_bindings(plan)
    left = early._call_evidence(early._directory(output, 0), plan, digest(plan), 0, plan["sample_ids"][0], sources)
    right = deepcopy(left)
    right["owner"].update(pid=202, process_start_ticks="23")
    right["runtime"].update(pid=202, gpu_uuid="GPU-94d2f97e-b0db-db79-5423-467860a66e2f", visible_devices="1")
    assert early.compare_calls(left, right, True)["comparison_status"] == "MATCH"
    right["runtime"]["policy"]["native_thread_observation"]["torch_num_threads"] += 1
    with pytest.raises(ValueError, match="runtime"):
        early.compare_calls(left, right, True)


def test_peer_stop_must_bind_exact_plan_and_owner(panel):
    _, _, plan, output = panel
    peer_plan = deepcopy(plan)
    peer_plan.update(arm="gpu1", gpu="1", gpu_uuid="GPU-94d2f97e-b0db-db79-5423-467860a66e2f")
    peer = (peer_plan, digest(peer_plan))
    owner = {"host": "fixture", "boot_id": "b", "pid": 1, "process_start_ticks": "2"}
    peer_output = output.parent / "peer"
    atomic_json(peer_output / ".qwen.claim.json", {"plan_sha256": peer[1], "implementation": plan["implementation"], "owner": owner})
    atomic_json(peer_output / "early-stop.json", {"plan_sha256": "wrong", "arm": "gpu1", "owner": owner, "comparison_status": "DIFFERENT"})
    with pytest.raises(ValueError, match="Peer stop"):
        early._stop_reason(output, peer_output, early._Stop(), plan, digest(plan), owner, peer)
    atomic_json(peer_output / "early-stop.json", {"plan_sha256": peer[1], "arm": "gpu1", "owner": owner, "comparison_status": "DIFFERENT"})
    assert early._stop_reason(output, peer_output, early._Stop(), plan, digest(plan), owner, peer) == "peer DIFFERENT"


def _peer_plan(panel):
    _, _, plan, output = panel
    peer = deepcopy(plan)
    peer.update(arm="gpu1", gpu="1", gpu_uuid="GPU-94d2f97e-b0db-db79-5423-467860a66e2f")
    path = output.parent / "peer-plan.json"
    atomic_json(path, peer)
    return path, peer, output.parent / "peer"


def test_prefix_barrier_prevents_request_17_when_peer_has_not_completed(monkeypatch, panel):
    peer_path, peer, peer_output = _peer_plan(panel)
    owner = {"host": "fixture", "boot_id": "fixture-boot", "pid": 202, "process_start_ticks": "23"}
    atomic_json(peer_output / ".qwen.claim.json", {"plan_sha256": digest(peer), "implementation": peer["implementation"], "owner": owner})
    calls, _, _ = _fake_runtime(monkeypatch, panel)
    waits = []
    def wait(seconds):
        waits.append(seconds)
        atomic_json(peer_output / "early-stop.json", {"plan_sha256": digest(peer), "arm": peer["arm"], "owner": owner, "comparison_status": "STOPPED"})
    # Bypass only the OS liveness lookup of this synthetic CPU process.
    original_exists = Path.exists
    original_read = Path.read_text
    monkeypatch.setattr(Path, "exists", lambda p: True if str(p) == "/proc/202/stat" else original_exists(p))
    monkeypatch.setattr(Path, "read_text", lambda p, *a, **kw: "202 (fixture) S " + " ".join(["0"] * 18 + ["23"]) if str(p) == "/proc/202/stat" else original_read(p, *a, **kw))
    monkeypatch.setattr(early.time, "sleep", wait)
    _, path, plan, output = panel
    result = early.run(path, output, "0", "/fixture", peer_path, peer_output)
    assert result["comparison_status"] == "STOPPED" and len(calls) == 16 and waits == [1]
    assert result["completed_pair_ids"] == plan["prefix_ids"]


def _copy_peer_prefix(output, peer_output, peer, repeat_indices=(0, 1), prefix_ready=True):
    """Independent owner/GPU fixture with otherwise identical real receipts."""
    for repeat in repeat_indices:
        source, dest = early._directory(output, repeat), early._directory(peer_output, repeat)
        shutil.copytree(source, dest)
        identity = json.loads((dest / "identity.json").read_text())
        identity["reproducibility"].update(plan_sha256=digest(peer), arm=peer["arm"])
        atomic_json(dest / "identity.json", identity)
        rows = early.original._read_jsonl(dest / "outputs.jsonl")
        for row in rows: row["reproducibility"].update(plan_sha256=digest(peer), arm=peer["arm"])
        (dest / "outputs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        events = early.original._read_jsonl(dest / "events.jsonl")
        for event in events:
            if event.get("owner"): event["owner"].update(pid=202, process_start_ticks="23")
            if event.get("snapshot"): event["snapshot"].update(pid=202, visible_devices="1", gpu_uuid="GPU-94d2f97e-b0db-db79-5423-467860a66e2f")
        (dest / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    if prefix_ready: atomic_json(peer_output / "pairs" / "07.json", {"comparison_status": "MATCH"})


def test_remaining_ids_expand_only_after_independent_prefix_matches(monkeypatch, panel):
    peer_path, peer, peer_output = _peer_plan(panel)
    def behavior(stage, calls, payload, session, signals):
        if stage == "before_commit" and len(calls) == 18: payload["raw_answer"] = "RIGHT"
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    _, path, plan, output = panel
    waits = []
    def wait(seconds):
        waits.append(seconds)
        _copy_peer_prefix(output, peer_output, peer)
    monkeypatch.setattr(early.time, "sleep", wait)
    result = early.run(path, output, "0", "/fixture", peer_path, peer_output)
    assert result["comparison_status"] == "DIFFERENT" and len(calls) == 18 and waits == [1]
    events = early.original._read_jsonl(early._directory(output, 0) / "events.jsonl")
    assert any(e["event"] == "EARLY_INDEPENDENT_PREFIX_MATCH" for e in events)


def test_compare_preserves_valid_local_difference_if_peer_stops_half_pair(monkeypatch, panel):
    peer_path, peer, peer_output = _peer_plan(panel)
    def behavior(stage, calls, payload, session, signals):
        if stage == "before_commit" and len(calls) == 2: payload["output_token_ids"] = [10, 12]
    _fake_runtime(monkeypatch, panel, behavior)
    _, path, _, output = panel
    early.run(path, output, "0", "/fixture")
    report = early.compare(path, output, peer_path, peer_output)
    assert report["comparison_status"] == "DIFFERENT" and report["evidence_status"] == "COMPLETE"


def test_one_completed_call_on_each_gpu_can_stop_before_own_repeat1(monkeypatch, panel):
    peer_path, peer, peer_output = _peer_plan(panel)
    calls, _, _ = _fake_runtime(monkeypatch, panel)
    _, path, plan, output = panel
    original_available = early._available_calls
    copied = []
    def available(directory, bound, bound_hash, sid, sources):
        if directory == peer_output and not copied:
            # The peer completes repeat0 while this arm finishes repeat0.
            _copy_peer_prefix(output, peer_output, peer, (0,), False)
            raw_path = early._directory(peer_output, 0) / "outputs.jsonl"
            rows = early.original._read_jsonl(raw_path)
            rows[0].update(raw_answer="RIGHT", output_token_ids=[20, 21])
            raw_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            copied.append(True)
        return original_available(directory, bound, bound_hash, sid, sources)
    monkeypatch.setattr(early, "_available_calls", available)
    result = early.run(path, output, "0", "/fixture", peer_path, peer_output)
    assert len(calls) == 1 and result["comparison_status"] == "DIFFERENT"
    assert result["comparison_scope"] == "cross_arm_call" and result["model_calls"] == 1
    report = early.compare(path, output, peer_path, peer_output)
    assert report["comparison_status"] == "DIFFERENT" and report["evidence_status"] == "COMPLETE"


def test_peer_raw_without_completion_receipt_is_pending(monkeypatch, panel):
    peer_path, peer, peer_output = _peer_plan(panel)
    def behavior(stage, calls, payload, session, signals):
        if stage == "before_commit" and len(calls) == 1:
            # This peer has a durable raw and technical pass, but the wrapper
            # has not finished actual GPU hashes/runtime/elapsed yet.
            directory = early._directory(peer_output, 0)
            directory.mkdir(parents=True)
            atomic_json(directory / "identity.json", {})
            (directory / "outputs.jsonl").write_text('{}\n')
            (directory / "events.jsonl").write_text('{"event":"TECHNICAL_PASS"}\n')
        if stage == "before_commit" and len(calls) == 2:
            payload["output_token_ids"] = [10, 12]
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    _, path, _, output = panel
    result = early.run(path, output, "0", "/fixture", peer_path, peer_output)
    assert len(calls) == 2 and result["comparison_status"] == "DIFFERENT"
    assert result["evidence_status"] == "COMPLETE"


def test_valid_local_counterexample_survives_invalid_completed_peer(monkeypatch, panel):
    peer_path, peer, peer_output = _peer_plan(panel)
    _, path, plan, output = panel
    def behavior(stage, calls, payload, session, signals):
        if stage == "before_commit" and len(calls) == 2:
            payload["output_token_ids"] = [10, 12]
        if stage == "after_commit" and len(calls) == 2:
            # Peer becomes complete but invalid while our repeat1 finishes.
            _copy_peer_prefix(output, peer_output, peer, (0,), False)
            event_path = early._directory(peer_output, 0) / "events.jsonl"
            events = early.original._read_jsonl(event_path)
            next(e for e in events if e.get("event") == "REPRO_CALL_RUNTIME")["snapshot"]["comparison_identity"]["qsa_live_observation"]["wrapper_code_exact"] = False
            event_path.write_text("".join(json.dumps(e) + "\n" for e in events))
    calls, _, _ = _fake_runtime(monkeypatch, panel, behavior)
    result = early.run(path, output, "0", "/fixture", peer_path, peer_output)
    assert len(calls) == 2 and result["comparison_status"] == "DIFFERENT"
    report = early.compare(path, output, peer_path, peer_output)
    assert report["comparison_status"] == "DIFFERENT" and report["evidence_status"] == "COMPLETE"


@pytest.mark.parametrize("actual", ["c89c0c8c-8081-5ba9-c73f-d08af5875193", "GPU-c89c0c8c-8081-5ba9-c73f-d08af5875193"])
def test_uuid_real_torch_representation_matches_plan_and_saved_snapshot_stays_original(monkeypatch, panel, actual):
    def stop_after_raw(stage, calls, payload, session, signals):
        if stage == "before_commit": signals[signal.SIGTERM](signal.SIGTERM, None)
    calls, _, _ = _fake_runtime(monkeypatch, panel, stop_after_raw, actual)
    _, path, _, output = panel
    result = early.run(path, output, "0", "/fixture")
    assert len(calls) == 1 and result["comparison_status"] == "STOPPED"
    events = early.original._read_jsonl(early._directory(output, 0) / "events.jsonl")
    assert next(e for e in events if e.get("event") == "REPRO_RUNTIME")["snapshot"]["gpu_uuid"] == actual


@pytest.mark.parametrize("actual", ["94d2f97e-b0db-db79-5423-467860a66e2f", "", None, "unavailable", "GPU-c89c0c8c-8081-5ba9-c73f-d08af5875193-extra"])
def test_uuid_startup_rejects_wrong_missing_or_malformed_before_model_call(monkeypatch, panel, actual):
    calls, _, _ = _fake_runtime(monkeypatch, panel, observed_uuid=actual)
    _, path, _, output = panel
    with pytest.raises(ValueError, match="GPU UUID"):
        early.run(path, output, "0", "/fixture")
    assert calls == [] and not list(output.rglob("outputs.jsonl"))
    assert json.loads((output / "early-stop.json").read_text())["comparison_status"] == "INVALID_EVIDENCE"


@pytest.mark.parametrize("actual", ["94d2f97e-b0db-db79-5423-467860a66e2f", "", None, "unavailable", "prefix-c89c0c8c-8081-5ba9-c73f-d08af5875193"])
def test_uuid_snapshot_rejects_wrong_missing_or_malformed(panel, actual):
    helper, _, plan, _ = panel
    snapshot = helper._snapshot()
    snapshot.update(gpu_uuid=actual, visible_devices="0", pid=100, hostname="fixture")
    with pytest.raises(ValueError, match="GPU|physical GPU"):
        early._snapshot_identity(snapshot, plan)
