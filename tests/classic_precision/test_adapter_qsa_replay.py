"""Targeted CPU fixtures for QSA-only native-adapter routing and early stops."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import signal
import socket
import sys
from types import SimpleNamespace

import pytest

from classic_precision import adapter_qsa_replay as replay
from classic_precision import inference, storage
from runtime_compat import classic_precision_adapter_qsa as helper

GPU = "c89c0c8c-8081-5ba9-c73f-d08af5875193"


def _load_test(name):
    path = Path(__file__).with_name(name + ".py")
    spec = importlib.util.spec_from_file_location("qsa_adapter_" + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def panel(tmp_path, monkeypatch):
    original = _load_test("test_adapter_replay")
    suite = original.suite.__wrapped__(tmp_path, monkeypatch)
    baselines = {r: {"NUMPY_MADVISE_HUGEPAGE": "0", "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB": "16" if r in ("single", "pair") else None,
        "PATH": "/qy/no-nvcc/bin:/usr/bin:/bin" if r in ("single", "pair") else "/qy/bin:/usr/bin:/bin"} for r in replay.native.COUNTS}
    plan = replay.build_plan(suite.plan, baselines, list(replay.native.COUNTS), str(suite.output / "ADAPTER_QSA_STOP.json"))
    storage.atomic_json(suite.path, plan)
    qsa = _load_test("test_qsa_only_reproducibility")
    def snapshot(route, cold):
        qsa.BASELINE = plan["profiles"][route]["baseline_environment"]
        value = qsa._snapshot()
        value.update(gpu_uuid=GPU, hostname=socket.gethostname(), pid=100 + cold, visible_devices="0")
        identity = value["comparison_identity"]
        for target in (identity, *identity["worker_runtime"]):
            environment = dict.fromkeys(helper.THREAD_ENVIRONMENT)
            environment["OMP_NUM_THREADS"] = target["native_postload_environment"]["OMP_NUM_THREADS"]
            if route == "single":
                environment = dict.fromkeys(helper.THREAD_ENVIRONMENT, "1")
                for key in helper.THREAD_ENVIRONMENT:
                    if key in target["native_postload_environment"]:
                        target["native_postload_environment"][key] = target["observed_environment"][key] = "1"
                target["native_postload_environment"]["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] = None
                target["observed_environment"]["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] = None
                for field in ("native_thread_baseline", "native_thread_observation"):
                    target[field].update(omp_num_threads="1", vllm_omp_marker=None)
                target["engine_launch_parent"]["omp_num_threads"] = "1"
            target["adapter_native_environment"] = environment
        return value
    def import_receipt(route):
        entry = replay.native._route(suite.coverage, route)
        before = dict.fromkeys(helper.THREAD_ENVIRONMENT)
        after = dict.fromkeys(helper.THREAD_ENVIRONMENT, "1") if route == "single" else dict(before)
        receipt = {"adapter_sha256": replay.native._adapter_descriptor(entry)["sha256"], "before_import": before,
            "after_native_import": after, "before_native_processor": dict(after)}
        if route == "single": receipt["static_inputs_sha256"] = next(r["sha256"] for r in entry["implementation_files"] if Path(r["path"]).name == "static_inputs.py")
        return receipt
    return SimpleNamespace(source=suite, plan=plan, path=suite.path, output=suite.output, baselines=baselines, snapshot=snapshot, import_receipt=import_receipt, old=original)


def test_qwen_only_120_call_plan_preserves_native_route_offload_and_path(panel):
    plan = panel.plan
    assert replay.validate_plan(plan, physical=False) == panel.source.coverage
    assert plan["models"] == ["qwen"] and plan["max_model_calls"] == 120
    assert {r: len(plan["sample_ids"][r]) for r in plan["routes"]} == {"video": 4, "multi": 6, "single": 12, "pair": 8}
    for route, profile in plan["profiles"].items():
        assert profile["torch_policy"] == {} and profile["environment_interventions"] == {replay.qsa.runtime.FLAG: "1"}
        assert profile["baseline_environment"]["CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB"] == ("16" if route in ("single", "pair") else None)
        assert replay._runtime(panel.snapshot(route, 0), plan, route, panel.import_receipt(route))["gpu_uuid"] == GPU
    assert "no-nvcc" in plan["profiles"]["single"]["environment"]["PATH"]
    assert "no-nvcc" not in plan["profiles"]["video"]["environment"]["PATH"]


def test_real_saved_coverage_and_host_plans_rebuild_without_strict_flags():
    root = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility")
    coverage = replay.native.read(root / "coverage-adapter-v3.json")
    for host, routes in (("fourier", ["video", "multi"]), ("radon", ["single", "pair"])):
        source = replay.native.read(root / ("remote-" + host) / ("adapter-v3-plan-" + host + ".json"))
        baselines = {r: {"NUMPY_MADVISE_HUGEPAGE": "0", "CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB": "16" if host == "radon" else None,
            "PATH": "/qy/no-nvcc/bin" if host == "radon" else "/qy/bin"} for r in routes}
        plan = replay.build_plan(source, baselines, routes, str(root / ("CPU-fixture-" + host + "-stop.json")))
        assert plan["source_plan_sha256"] == storage.digest(source)
        for route in routes:
            assert plan["sample_ids"][route] == [r["sample_id"] for r in replay.native._route(coverage, route)["representative_samples"]]
            assert plan["model_state"]["qwen"][route] == source["model_state"]["qwen"][route]
            assert plan["calibrations"][route]["qwen"] == source["calibrations"][route]["qwen"]
            assert plan["profiles"][route]["baseline_environment"]["VLLM_ENABLE_V1_MULTIPROCESSING"] is None


def _stores(panel, monkeypatch, route, cold):
    plan = panel.plan
    bound = replay.native.bind_route(plan, panel.source.coverage, route, "qwen", assets=False)
    root = replay._root(panel.output, route, cold)
    owner = {"host": socket.gethostname(), "boot_id": "CPU-fixture", "pid": 100 + cold, "process_start_ticks": str(1000 + cold), "started_unix": 1}
    monkeypatch.setattr(storage, "owner_identity", lambda: deepcopy(owner))
    claim = {"owner": owner, "engine_id": "fixture-engine-" + str(cold), "plan_sha256": storage.digest(plan)}
    storage.atomic_json(root / "qwen.claim.json", claim)
    stores = []
    for repeat in range(2):
        identity = {"schema": replay.SCHEMA, "model_key": "qwen", "model": bound["spec"],
            "checkpoint_assets": plan["model_state"]["qwen"][route]["checkpoint_assets"],
            "replay": {"plan_sha256": storage.digest(plan), "profile": plan["profiles"][route], "route": route, "cold_pass": cold,
                "repeat": repeat, "sample_ids": plan["sample_ids"][route], "engine_id": claim["engine_id"],
                "model_spec_source": bound["model_spec_source"], "geometry_calibration": bound["calibration"], "geometry_coverage": bound["geometry"]}}
        store = storage.RunStore(root / ("repeat-" + str(repeat)), identity, "qwen", "fixture", plan["sample_ids"][route]).__enter__()
        lease_root = replay.native.relocated(plan["gpu_lock_root"], plan["path_map"])
        lock = lease_root / (".gpu-" + storage.digest({"host": owner["host"], "device": GPU}) + ".lock")
        store.event("REPLAY_PHYSICAL_LEASE", lease={"root": str(lease_root), "path": str(lock), "st_dev": 1, "st_ino": 2,
            "device": GPU, "owner": owner, "host": owner["host"], "resource_snapshot": {"admitted": True,
                "free_gpu_gib": 2, "available_ram_gib": 2, "budgets": bound["spec"]["resources"]}})
        if repeat == 0:
            store.event("ADAPTER_QSA_PRE_ENGINE_ENVIRONMENT", environment=panel.import_receipt(route)["before_native_processor"],
                receipt=panel.snapshot(route, cold)["comparison_identity"]["engine_launch_parent"])
        store.event("REPLAY_RUNTIME", snapshot=panel.snapshot(route, cold), engine_id=claim["engine_id"])
        store.event("REPLAY_NATIVE_IMPORT_ENVIRONMENT", receipt=panel.import_receipt(route))
        stores.append(store)
    return bound, stores


def _emit(panel, bound, store, route, cold, repeat, row, answer="RIGHT", tokens=None, durable=None):
    sid = row["sample_id"]
    source_input = {"row": row, "ready": replay.native._ready(bound, row)}
    attempt = store.begin(sid, storage.digest(source_input)) if durable is None else store._active[sid]
    receipt = {"deep_check": True, "input_tokens": 10, "chat_template_text": "FULL UNCHANGED NATIVE TEMPLATE",
        "chat_template_sha256": storage.digest("FULL UNCHANGED NATIVE TEMPLATE"), "actual_expected_GPU_shape": [1, 2], "expected_BF16_GPU_sha256": "a" * 64}
    store.event("ADAPTER_QSA_CALL_RUNTIME", sample_id=sid, stage="before", snapshot=panel.snapshot(route, cold))
    raw = {"task": row["task"], "truth": row["truth"], "source_parameters": row["source_parameters"], "input_receipt": receipt,
        "actual_input_tokens": 10, "raw_answer": answer, "output_token_ids": tokens or [5, 9], "finish_reason": "stop", "stop_reason": 9,
        "seconds": .01, "effective_generation": inference.GENERATION, "replay_parent_sample_id": sid,
        "adapter_replay": {"source_input": source_input, "source_input_sha256": storage.digest(source_input),
            "plan_sha256": storage.digest(panel.plan), "route": route, "cold_pass": cold, "repeat": repeat}}
    if durable is None: store.commit_output(sid, raw)
    else: durable(raw)
    store.event("POST_OUTPUT_INPUT_RECEIPT", sample_id=sid, attempt_id=attempt, native_guard=True,
        **({"t1_deep_qualification": True, "t1_expected_hashes": {"input_sha256": "a" * 64, "projection_sha256": "c" * 64}} if route == "video" else {}))
    store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=attempt)
    binding = {"profile_id": helper.PROFILE_ID, "family": "qwen", "route": route,
        "coverage_sha256": panel.plan["coverage"]["sha256"], "adapter_sha256": replay.native._adapter_descriptor(bound["entry"])["sha256"],
        "instrumentation_sha256": panel.plan["implementation"]["helper_sha256"], "row_sha256": storage.digest(row)}
    for stage, tensors in (("cpu_processor", [panel.old.fingerprint("input_ids", "torch.int64"), panel.old.fingerprint("pixel_values", "torch.float32")]),
        ("gpu_forward", [panel.old.fingerprint("pixels")])):
        store.event("REPRO_INPUT_HASH", sample_id=sid, stage=stage, tensors=tensors, **binding)
    store.event("ADAPTER_QSA_CALL_RUNTIME", sample_id=sid, stage="after", snapshot=panel.snapshot(route, cold))
    if durable is None: store.event("REPLAY_CALL_COMPLETED", sample_id=sid, attempt_id=attempt, seconds=.02)


@pytest.mark.parametrize("route", list(replay.native.COUNTS))
def test_four_native_shapes_keep_wrong_answers_and_full_raw_evidence(panel, monkeypatch, route):
    calls = []
    for cold in range(2):
        bound, stores = _stores(panel, monkeypatch, route, cold)
        try:
            row = bound["rows"][0]
            for repeat, store in enumerate(stores):
                _emit(panel, bound, store, route, cold, repeat, row)
                calls.append(replay._call(panel.output, panel.plan, bound, replay.native._native(bound, panel.plan, "qwen"), route, cold, repeat, row["sample_id"]))
        finally:
            for store in stores: store.close()
    assert replay.compare_calls(calls[0], calls[1])["comparison_status"] == "MATCH"
    assert replay.compare_calls(calls[0], calls[2], True)["comparison_status"] == "MATCH"
    assert calls[0]["raw"]["raw_answer"] == "RIGHT" and calls[0]["raw"]["truth"] == "LEFT"
    different = deepcopy(calls[1])
    different["value"]["raw"]["output_token_ids"] = [5, 10]
    assert replay.compare_calls(calls[0], different)["comparison_status"] == "DIFFERENT"
    different["value"]["hashes"]["gpu_forward"][0]["sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="input"):
        replay.compare_calls(calls[0], different)


def test_private_adapter_observer_does_not_mutate_or_activate_shared_strict_module():
    from runtime_compat import classic_precision_adapter_determinism as old
    before = (old.PROFILE_ID, old._core, old._coverage, old._assert_policy, deepcopy(old._state))
    observer = helper._observers()
    assert observer is not old and observer.PROFILE_ID == helper.PROFILE_ID
    assert before == (old.PROFILE_ID, old._core, old._coverage, old._assert_policy, old._state)
    assert observer._install_worker_probe is helper._install_worker_probe
    assert observer._read_worker_probe is helper._read_worker_probe
    assert helper._install_worker_probe.__module__ == "runtime_compat.classic_precision_adapter_qsa"


def _fake_execution(panel, monkeypatch, route, different_call=2, cold=0, bad_input_call=None, interrupt_call=None, peer_stop_call=None):
    from contextlib import contextmanager
    plan = panel.plan
    original_validate = replay.validate_plan
    monkeypatch.setattr(replay, "validate_plan", lambda p, **kw: original_validate(p, physical=False))
    bound = replay.native.bind_route(plan, panel.source.coverage, route, "qwen", assets=False)
    module = replay.native._native(bound, plan, "qwen")
    if route == "single":
        original_native = replay.native._native
        def native_import(bound, plan, family):
            for key in helper.THREAD_ENVIRONMENT: replay.os.environ[key] = "1"
            return original_native(bound, plan, family)
        monkeypatch.setattr(replay.native, "_native", native_import)
    owner = {"host": socket.gethostname(), "boot_id": "CPU-fixture", "pid": 100 + cold, "process_start_ticks": str(1000 + cold), "started_unix": 1}
    monkeypatch.setattr(replay, "owner_identity", lambda: deepcopy(owner))
    monkeypatch.setattr(storage, "owner_identity", lambda: deepcopy(owner))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(helper, "activate", lambda *a: monkeypatch.setitem(sys.modules, "torch", SimpleNamespace()))
    monkeypatch.setattr(helper, "instrument_session", lambda *a: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(helper, "snapshot", lambda: panel.snapshot(route, cold))
    monkeypatch.setattr(inference, "_source_identity", lambda *a: plan["model_state"]["qwen"][route]["processor_source"])
    lease_root = replay.native.relocated(plan["gpu_lock_root"], plan["path_map"])
    @contextmanager
    def lease(*args):
        yield {"root": str(lease_root), "path": str(lease_root / (".gpu-" + storage.digest({"host": owner["host"], "device": GPU}) + ".lock")),
            "device": GPU, "owner": owner, "host": owner["host"], "st_dev": 1, "st_ino": 2,
            "resource_snapshot": {"admitted": True, "free_gpu_gib": 2, "available_ram_gib": 2, "budgets": bound["spec"]["resources"]}}
    monkeypatch.setattr(replay.native, "physical_lease", lease)
    calls = []
    class Session:
        def __init__(self, proxy):
            self.proxy, self.processor, self.context = proxy, object(), {}
            proxy.current.event("ADAPTER_QSA_PRE_ENGINE_ENVIRONMENT", environment=helper._native_environment(), receipt=panel.snapshot(route, cold)["comparison_identity"]["engine_launch_parent"])
        def load(self, *args): pass
        def close(self): pass
        def generate(self, row, ready, root, durable): self.finish(row, durable)
        def generate_image(self, row, ready, durable, deep):
            assert deep is True
            self.finish(row, durable)
        def generate_images(self, row, arrays, ready, durable, deep):
            assert deep is True
            self.finish(row, durable)
        generate_pair = generate_images
        def finish(self, row, durable):
            store = self.proxy.current
            repeat = int(store.directory.parent.name[-1])
            calls.append((row["sample_id"], repeat))
            if len(calls) == interrupt_call: signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            _emit(panel, bound, store, route, cold, repeat, row, tokens=[5, 10] if len(calls) == different_call else [5, 9], durable=durable)
            if len(calls) == peer_stop_call:
                peer_owner = {**owner, "pid": 200, "process_start_ticks": "2000"}
                peer_claim = replay._root(panel.output, "video", 0) / "qwen.claim.json"
                storage.atomic_json(peer_claim, {"plan_sha256": storage.digest(plan), "owner": peer_owner, "engine_id": "peer-engine"})
                replay._publish_claim(plan["stop_path"], {"schema": replay.SCHEMA, "plan_sha256": storage.digest(plan),
                    "route": "video", "cold_pass": 0, "comparison_status": "DIFFERENT", "owner": peer_owner,
                    "engine_id": "peer-engine", "claim_path": str(peer_claim)})
            if len(calls) == bad_input_call:
                path = store.directory / "events.jsonl"
                events = [json.loads(line) for line in path.read_text().splitlines()]
                next(e for e in events if e.get("event") == "REPRO_INPUT_HASH" and e.get("stage") == "gpu_forward")["tensors"][0]["sha256"] = "b" * 64
                path.write_text("".join(json.dumps(e) + "\n" for e in events))
    class Guard:
        def __init__(self, proxy, base): self.raw = proxy
    if route == "video":
        monkeypatch.setattr(module, "GuardStore", Guard, raising=False)
        monkeypatch.setattr(module, "make_session", lambda base, family, model, guard, package, resources: Session(guard.raw), raising=False)
    elif route == "single":
        monkeypatch.setitem(sys.modules, "static_worker", SimpleNamespace(StaticSession=lambda family, event_sink: Session(event_sink)))
    else:
        monkeypatch.setattr(module, "make_session", lambda base, args, proxy, spec: Session(proxy), raising=False)
        ready = {r["sample_id"]: replay.native._ready(bound, r) for r in bound["rows"]}
        monkeypatch.setattr(module, "read_images" if route == "multi" else "read_pair", lambda row: ([], ready[row["sample_id"]]), raising=False)
    profile = plan["profiles"][route]
    for key in helper.THREAD_ENVIRONMENT:
        original_value = replay.os.environ.get(key)
        monkeypatch.setenv(key, original_value if original_value is not None else "fixture-before-import")
        if original_value is None: monkeypatch.delenv(key)
    for key in profile["cleared_environment"]: monkeypatch.delenv(key, raising=False)
    for key, value in profile["environment"].items(): monkeypatch.setenv(key, value)
    return calls


@pytest.mark.parametrize("route", list(replay.native.COUNTS))
def test_native_route_api_first_valid_difference_stops_after_two_requests(panel, monkeypatch, route):
    calls = _fake_execution(panel, monkeypatch, route)
    result = replay.run(panel.path, panel.output, route, 0, "0")
    assert result["comparison_status"] == "DIFFERENT" and result["evidence_status"] == "COMPLETE"
    assert len(calls) == result["model_calls"] == 2 and calls[0][0] == calls[1][0]
    assert calls[0][1:] == (0,) and calls[1][1:] == (1,)
    assert replay.native.read(panel.output / "ADAPTER_QSA_STOP.json")["comparison_status"] == "DIFFERENT"
    assert replay.compare(panel.path, panel.output, route, physical=False)["comparison_status"] == "DIFFERENT"


def test_real_qsa_activation_separates_only_video_wrappers_without_strict_or_torch_import(tmp_path, monkeypatch):
    qsa_test = _load_test("test_qsa_only_reproducibility")
    generator = qsa_test.live_selector.__wrapped__(monkeypatch, tmp_path)
    next(generator)
    runtime = replay.qsa.runtime
    baseline = {"NUMPY_MADVISE_HUGEPAGE": "0"}
    profile = replay.qsa.build_profile("qwen_qsa_only", baseline)
    for key in profile["cleared_environment"]: monkeypatch.delenv(key, raising=False)
    for key, value in profile["environment"].items(): monkeypatch.setenv(key, value)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(runtime._base, "_state", {})
    monkeypatch.setattr(runtime._base, "_application", {})
    monkeypatch.setattr(runtime._base, "_postload_environment", {})
    monkeypatch.setattr(runtime._base, "_session", None)
    monkeypatch.setattr(runtime.native.observers, "_state", {})
    monkeypatch.setattr(helper, "_state", {})
    monkeypatch.setattr(helper, "_private", None)
    originals = {name: getattr(inference.ModelSession, name) for name in ("_initialize_processor", "prepare", "generate")}
    for name, value in originals.items(): monkeypatch.setattr(inference.ModelSession, name, value)
    monkeypatch.setattr(inference, "_install_vision_probe", inference._install_vision_probe)
    coverage = {"schema": "classic_precision_input_path_coverage/v1", "frozen_strict_profile": {"implementation": {
        "base_runner_sha256": runtime.BASE_RUNNER_SHA256, "qsa_helper_sha256": runtime.QSA_HELPER_SHA256}}}
    path = tmp_path / "activation-coverage.json"
    storage.atomic_json(path, coverage)
    try:
        helper.activate(path, baseline)
        assert "torch" not in sys.modules and "vllm" not in sys.modules
        assert inference.ModelSession.prepare is originals["prepare"]
        assert inference.ModelSession.generate is originals["generate"]
        assert inference.ModelSession._initialize_processor.__wrapped__ is originals["_initialize_processor"]
        assert runtime._base._state["torch_policy"] == {} and helper.status()["torch_policy"] == {}
        assert helper._observers()._install_worker_probe is helper._install_worker_probe
    finally:
        with pytest.raises(StopIteration): next(generator)


def test_cold0_finishes_finite_ids_before_cold1_first_call_counterexample(panel, monkeypatch):
    calls0 = _fake_execution(panel, monkeypatch, "video", different_call=None)
    result0 = replay.run(panel.path, panel.output, "video", 0, "0")
    assert result0["comparison_status"] == "MATCH" and len(calls0) == 8
    assert calls0 == [(sid, repeat) for sid in panel.plan["sample_ids"]["video"] for repeat in range(2)]
    calls1 = _fake_execution(panel, monkeypatch, "video", different_call=1, cold=1)
    result1 = replay.run(panel.path, panel.output, "video", 1, "0")
    assert result1["comparison_status"] == "DIFFERENT" and result1["model_calls"] == len(calls1) == 1
    assert replay.compare(panel.path, panel.output, "video", physical=False)["comparison_status"] == "DIFFERENT"


def test_invalid_actual_gpu_input_stops_after_one_durable_raw(panel, monkeypatch):
    calls = _fake_execution(panel, monkeypatch, "multi", different_call=None, bad_input_call=1)
    with pytest.raises(RuntimeError, match="Actual GPU pixels"):
        replay.run(panel.path, panel.output, "multi", 0, "0")
    result = replay.native.read(replay._root(panel.output, "multi", 0) / "qwen.json")
    assert result["comparison_status"] == "INVALID_EVIDENCE" and len(calls) == 1
    assert len((replay._root(panel.output, "multi", 0) / "repeat-0/qwen/outputs.jsonl").read_text().splitlines()) == 1


def test_external_stop_finishes_current_raw_technical_and_elapsed(panel, monkeypatch):
    calls = _fake_execution(panel, monkeypatch, "single", different_call=None, interrupt_call=1)
    result = replay.run(panel.path, panel.output, "single", 0, "0")
    assert result["comparison_status"] == "STOPPED" and len(calls) == 1 and result["elapsed_seconds"] >= 0
    events = [json.loads(line) for line in (replay._root(panel.output, "single", 0) / "repeat-0/qwen/events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "TECHNICAL_PASS" for e in events)
    assert any(e.get("event") == "REPLAY_CALL_COMPLETED" and e["seconds"] >= 0 for e in events)


def test_importable_worker_bridge_installs_reads_and_removes_native_probe(monkeypatch):
    observer = helper._observers()
    captured = [{"name": "pixels", "sha256": "a" * 64}]
    monkeypatch.setattr(observer, "_gpu_tensors", lambda *args: captured)
    class Vision:
        def register_forward_pre_hook(self, callback, with_kwargs):
            assert with_kwargs is True
            self.callback, self.removed = callback, False
            return SimpleNamespace(remove=lambda: setattr(self, "removed", True))
    vision = Vision()
    worker = SimpleNamespace(get_model=lambda: SimpleNamespace(visual=vision))
    assert helper._install_worker_probe(worker, "qwen") is True
    vision.callback(vision, (), {})
    assert helper._read_worker_probe(worker) == [captured] and vision.removed is True
    assert not hasattr(worker, observer._PROBE_ATTRIBUTE)


@pytest.mark.parametrize("fault", [None, "parent_env", "worker_env", "marker", "source", "import_source", "processor_env", "worker_threads"])
def test_static_native_thread_derivation_uses_actual_snapshot_and_private_validator(panel, fault):
    snapshot = panel.snapshot("single", 0)
    receipt = panel.import_receipt("single")
    identity = snapshot["comparison_identity"]
    worker = identity["worker_runtime"][0]
    if fault == "parent_env": identity["adapter_native_environment"]["MKL_NUM_THREADS"] = "2"
    elif fault == "worker_env": worker["adapter_native_environment"].pop("NUMEXPR_NUM_THREADS")
    elif fault == "marker":
        worker["native_postload_environment"]["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] = "1"
        worker["observed_environment"]["VLLM_OMP_NUM_THREADS_SET_BY_VLLM"] = "1"
        for field in ("native_thread_baseline", "native_thread_observation"): worker[field]["vllm_omp_marker"] = "1"
    elif fault == "source": worker["engine_startup_source"]["actual_sha256"] = "0" * 64
    elif fault == "import_source": receipt["static_inputs_sha256"] = "0" * 64
    elif fault == "processor_env": receipt["before_native_processor"]["OMP_NUM_THREADS"] = None
    elif fault == "worker_threads":
        for field in ("native_thread_baseline", "native_thread_observation"): worker[field]["torch_num_threads"] = 6
    before, public_validator = deepcopy(snapshot), replay.qsa._base._validate_environment
    if fault is None:
        result = replay._runtime(snapshot, panel.plan, "single", receipt)
        assert result["policy"]["native_postload_environment"]["OMP_NUM_THREADS"] == "1"
    else:
        with pytest.raises((RuntimeError, ValueError)): replay._runtime(snapshot, panel.plan, "single", receipt)
    assert snapshot == before and replay.qsa._base._validate_environment is public_validator


def test_shared_host_stop_finishes_inflight_and_cannot_overwrite_first_counterexample(panel, monkeypatch):
    calls = _fake_execution(panel, monkeypatch, "multi", different_call=None, peer_stop_call=1)
    result = replay.run(panel.path, panel.output, "multi", 0, "0")
    assert len(calls) == 1 and result["comparison_status"] == "STOPPED"
    first = replay.native.read(panel.plan["stop_path"])
    assert first["comparison_status"] == "DIFFERENT" and first["route"] == "video" and first["engine_id"] == "peer-engine"
    events = [json.loads(line) for line in (replay._root(panel.output, "multi", 0) / "repeat-0/qwen/events.jsonl").read_text().splitlines()]
    assert any(e.get("event") == "TECHNICAL_PASS" for e in events)
    assert any(e.get("event") == "REPLAY_CALL_COMPLETED" for e in events)
    assert replay._shared_stop(panel.plan) == first
    bad = deepcopy(first)
    bad["owner"]["process_start_ticks"] = "reused"
    storage.atomic_json(panel.plan["stop_path"], bad)
    with pytest.raises(RuntimeError, match="claim/owner"): replay._shared_stop(panel.plan)
