"""CPU-only replay closure tests; synthetic journals never certify a model."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import pytest

from classic_precision import adapter_replay as replay, inference, storage
from runtime_compat import classic_precision_adapter_determinism as helper


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return replay.descriptor(path)


def fingerprint(name, dtype="torch.bfloat16", shape=None, sha="a" * 64):
    shape = [1, 2] if shape is None else shape
    sizes = {"torch.bfloat16": 2, "torch.int64": 8, "torch.float32": 4}
    import math
    return {"name": name, "dtype": dtype, "shape": shape, "bytes": math.prod(shape)*sizes[dtype], "sha256": sha}


def snapshot(plan, family, env):
    from classic_precision import reproducibility
    core = helper._core()
    contract = reproducibility._profile_contract()
    profile = {"profile_id": core.PROFILE_ID, "family": family,
        "base_runner_sha256": contract["BASE_RUNNER_SHA256"], "environment": core.launch_environment(), "sdk_files_modified": False,
        "helper_sha256": plan["profile"]["implementation"]["determinism_helper_sha256"],
        "qsa_installation": {"state": "applied", "helper_source_sha256": plan["profile"]["implementation"]["qsa_helper_sha256"],
            "expected_native_source_sha256": contract["qsa"]["EXPECTED_SOURCE_SHA256"], "target_module": contract["qsa"]["TARGET_MODULE"],
            "model_architecture": contract["qsa"]["MODEL_ARCHITECTURE"]}}
    return {"comparison_identity": {"profile": profile, "packages": {k: "CPU-FIXTURE" for k in ("torch", "transformers", "numpy", "av")},
        "cuda": "CPU-FIXTURE",
        "gpu_name": "CPU-FIXTURE", "gpu_total_bytes": 100, "compute_capability": [1, 0], "driver": "CPU-FIXTURE",
        "inference_sha256": plan["profile"]["implementation"]["base_runner_sha256"],
        "deterministic_algorithms": True, "deterministic_warn_only": False, "cudnn_benchmark": False,
        "cudnn_deterministic": True, "fill_uninitialized_memory": core.FILL_UNINITIALIZED_MEMORY[family],
        "fp32_precision": {"matmul": "ieee", "cudnn_conv": "ieee", "cudnn_rnn": "ieee"},
        "effective_environment": {**plan["profile"]["environment"], **({"CLASSIC_PRECISION_QSA_DETERMINISM": "1"} if family == "qwen" else {})}}, "gpu_uuid": "GPU-CPU-FIXTURE", "adapter_environment": env}


@pytest.fixture
def suite(tmp_path, monkeypatch):
    coverage = {"schema": "classic_precision_input_path_coverage/v1", "routes": [],
        "frozen_strict_profile": {"implementation": replay._implementation()}}
    native = {}
    calibrations, all_rows, states = {}, {}, {f: {} for f in replay.ACTIVE}
    token_path = None
    for route, count in replay.COUNTS.items():
        root = tmp_path / route
        name = {"video": "evaluate.py", "multi": "evaluate.py", "single": "static_extension.py", "pair": "endpoint_evaluate.py"}[route]
        source = root / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text('''def validate_post(*args):
    post = args[0] if len(args) == 3 and isinstance(args[1], str) else args[1]
    if post.get("native_guard") is not True:
        raise RuntimeError("original native guard rejected receipt")
def validation_compat(raw, post, family):
    return validate_post(raw, post, family)
''')
        module_name = "adapter_replay_cpu_" + route
        spec = importlib.util.spec_from_file_location(module_name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, module_name, module)
        spec.loader.exec_module(module)
        native[route] = module
        implementations = [replay.descriptor(source)]
        if route == "single":
            for script in ("static_inputs.py", "static_worker.py"):
                path = root / script
                path.write_text("# source-qualified CPU fixture\n")
                implementations.append(replay.descriptor(path))
        rows, representatives, original_rows = [], [], []
        for index in range(count):
            row = {"sample_id": route + str(index), "task": "T2" if route == "single" and index < 6 else "T3" if route == "single" else "T4" if route == "pair" else "T1",
                "truth": "LEFT" if index % 2 == 0 else "RIGHT", "prompt": "unchanged original fixture prompt",
                "source_parameters": {"fixture_index": index}}
            if route == "single":
                row.update(image_id="image"+str(index), prompt_id="P"+str(index % 3))
            elif route == "video":
                row["source_parameters"]["original_parameters"] = {"fixture_index": index}
            elif route in ("multi", "pair"):
                row["frame_assets"] = []
                if route == "multi":
                    row.update(source_sequence_sha256="b"*64, parent_video_sha256="c"*64, parent_ready_sha256="d"*64)
                else:
                    row["source_pair_sha256"] = "b"*64
            rows.append(row)
            assets = []
            if route == "video":
                video = root / "media" / (row["sample_id"] + ".mkv")
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes(b"CPU schema fixture only")
                stat = video.stat()
                original_rows.append({"sample_id": row["sample_id"], "correct_answer": row["truth"], "prompt": row["prompt"],
                    "params": row["source_parameters"]["original_parameters"], "scientific_spec_sha256": "d"*64})
                ready = write(root / "ready" / (row["sample_id"] + ".json"), {
                    "sample_id": row["sample_id"], "video_path": str(video.relative_to(root)), "video_sha256": replay.file_sha256(video),
                    "video_bytes": stat.st_size, "video_mtime_ns": stat.st_mtime_ns, "scientific_spec_sha256": "d"*64,
                    "technical_verdict": "PASS", "validation": {"decoded_frames": 180, "max_code_error": 0, "FP32_roundtrip_max_code_error": 0}})
                assets = [{"path": str(video), "expected_file_sha256": replay.file_sha256(video), "ready_file": ready}]
            else:
                original_rows.append(row)
            representatives.append({"sample_id": row["sample_id"], "task": row["task"], "truth": row["truth"],
                "manifest_row_sha256": storage.digest(row), "prompt_utf8_sha256": hashlib.sha256(row["prompt"].encode()).hexdigest(), "source_assets": assets})
        models = {f: {"checkpoint_relative": f, "resources": {"min_free_gpu_gib": 1, "min_available_ram_gib": 1}} for f in replay.ACTIVE}
        manifest = {"rows": rows, "models": models}
        protocol = write(root / "protocol.json", {"generation": inference.GENERATION, "models": models})
        if route == "single":
            manifest["images"] = [{"image_id": r["image_id"], "pixel_sha256": "e"*64} for r in rows]
            manifest.pop("models")  # exact original static execution schema
            manifest["protocol_sha256"] = protocol["sha256"]
        execution = write(root / "manifest.json", manifest)
        original = write(root / "source.json", {"counts": count} if route == "single" else {"rows": original_rows})
        entry = {"route_id": route, "manifest": execution, "source_manifest": original, "protocols": [protocol],
                 "implementation_files": implementations, "representative_samples": representatives}
        all_rows[route], calibrations[route] = rows, {}
        for family in replay.ACTIVE:
            records = []
            for i, row in enumerate(rows):
                receipt = {"deep_check": True, "expected_BF16_GPU_sha256": "a"*64,
                           "actual_expected_GPU_shape": [1, 2], "actual_processor_tensor_shape": [1, 2]}
                if route == "multi":
                    receipt["images_per_context"] = 2 if i == 0 else 4
                records.append({"sample_id": row["sample_id"], "input_receipt": receipt,
                    "technical_CPU_status": "PASS", "expected_GPU_hashes": {"input_sha256": "a"*64, "projection_sha256": "c"*64}})
            qualification = {"status": "COMPLETE" if route == "video" else "PASS", "model": family,
                "model_calls": 0, "weights_loaded": False, "CUDA_initialized": False, "records": records,
                "adapter_sha256": replay.descriptor(source)["sha256"]}
            key = {"video": "evaluation_manifest_sha256", "single": "manifest_sha256"}.get(route, "execution_manifest_sha256")
            qualification[key] = execution["sha256"]
            if route == "single":
                qualification["source_sha256"] = next(i["sha256"] for i in implementations if i["path"].endswith("static_inputs.py"))
            calibrations[route][family] = write(root / ("cpu-"+family+".json"), qualification)
            states[family][route] = {"checkpoint_assets": {"weight_shard_bytes": {"fixture.safetensors": 1}},
                "processor_source": {"qualname": "CPUFixtureProcessor", "source_sha256": "f"*64}}
            identity = {"model_key": family, "model": models[family], "checkpoint_assets": states[family][route]["checkpoint_assets"],
                        key: execution["sha256"], "adapter_sha256": replay.descriptor(source)["sha256"]}
            states[family][route]["native_identity"] = write(root / (family+"-native-identity.json"), identity)
            event = {"event": "PROCESSOR_IDENTITY", "processor": states[family][route]["processor_source"]}
            event_path = root / (family+"-native-events.jsonl")
            event_path.write_text(storage.canonical_json(event)+"\n")
            states[family][route]["processor_receipt"] = {"path": str(event_path), "line_number": 1, "record_sha256": storage.digest(event)}
        coverage["routes"].append(entry)
        if route == "single":
            token_path = write(root / "native-image-token-preflight.json", {"status": "PASS", "extension_sha256": replay.descriptor(source)["sha256"],
                "weights_loaded": False, "new_model_calls": 0, "features": 384, "replacement_tokens": 386,
                "placeholder_tokens": 1, "added_tokens": 385, "native_get_image_repl_source_sha256": "d"*64,
                "examples": [{"sample_id": r["sample_id"]} for r in rows[:3]+rows[6:9]]})
    core = helper._core()
    coverage_path = write(tmp_path / "coverage.json", coverage)
    plan = {"schema": replay.SCHEMA, "models": list(replay.ACTIVE), "cold_passes": 2, "repeats": 2,
        "coverage": coverage_path, "path_map": [], "profile": {"id": core.PROFILE_ID,
            "implementation": replay._implementation(), "environment": core.launch_environment(),
            "fill_uninitialized_memory": core.FILL_UNINITIALIZED_MEMORY},
        "model_root": str(tmp_path / "models"), "cosmos_package": str(tmp_path / "cosmos"), "model_state": states,
        "gpu_lock_root": str(tmp_path / "video/evaluation/outputs"), "calibrations": calibrations,
        "route_environment": {"single": {"qwen": {"CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB": "16"}}},
        "auxiliary": {"single_nemo_token_preflight": token_path}, "sample_ids": {k: [r["sample_id"] for r in v] for k, v in all_rows.items()}}
    plan_path = tmp_path / "plan.json"
    write(plan_path, plan)
    monkeypatch.setattr(replay, "_native", lambda bound, plan, family: native[bound["entry"]["route_id"]])
    output = tmp_path / "outputs"
    def materialize():
        for family in replay.ACTIVE:
            for route in replay.COUNTS:
                bound = replay.bind_route(plan, coverage, route, family, assets=False)
                for cold in range(2):
                    root = output / f"cold-{cold}" / route
                    owner = {"host": socket.gethostname(), "boot_id": "CPU-fixture", "pid": 100+list(replay.COUNTS).index(route)*2+cold,
                             "process_start_ticks": str(1000+cold), "started_unix": 1.0}
                    monkeypatch.setattr(storage, "owner_identity", lambda: owner)
                    engine = family + route + str(cold)
                    done = {"status": "COMPLETE", "plan_sha256": storage.digest(plan), "engine_id": engine, "owner": owner}
                    write(root / (family+".json"), done); write(root / (family+".claim.json"), done)
                    for repeat in range(2):
                        ids = plan["sample_ids"][route]
                        identity = {"model": bound["spec"], "checkpoint_assets": states[family][route]["checkpoint_assets"], "replay": {
                            "plan_sha256": storage.digest(plan), "profile": plan["profile"], "route": route,
                            "cold_pass": cold, "repeat": repeat, "sample_ids": ids, "engine_id": engine,
                            "model_spec_source": bound["model_spec_source"],
                            "geometry_calibration": bound["calibration"], "geometry_coverage": bound["geometry"]}}
                        env = replay._environment(plan, family, route)
                        with storage.RunStore(root / f"repeat-{repeat}", identity, family, "fixture", ids) as store:
                            device = "GPU-CPU-FIXTURE"
                            lock = Path(plan["gpu_lock_root"]) / (".gpu-"+storage.digest({"host": owner["host"], "device": device})+".lock")
                            store.event("REPLAY_PHYSICAL_LEASE", lease={"root": plan["gpu_lock_root"], "path": str(lock), "st_dev": 1, "st_ino": 1,
                                "host": owner["host"], "device": device, "owner": owner,
                                "resource_snapshot": {"admitted": True, "free_gpu_gib": 2, "available_ram_gib": 2, "budgets": bound["spec"]["resources"]}})
                            store.event("REPLAY_RUNTIME", snapshot=snapshot(plan, family, env), engine_id=engine, environment=env, model_state=states[family][route])
                            for row in all_rows[route]:
                                sid = row["sample_id"]
                                source_input = {"row": row, "ready": replay._ready(bound, row)}
                                attempt = store.begin(sid, storage.digest(source_input))
                                receipt = {"deep_check": True, "input_tokens": 10, "chat_template_text": "ORIGINAL FULL TEMPLATE",
                                    "chat_template_sha256": storage.digest("ORIGINAL FULL TEMPLATE"), "actual_expected_GPU_shape": [1, 2],
                                    "expected_BF16_GPU_sha256": "a"*64}
                                raw = {"task": row["task"], "truth": row["truth"], "source_parameters": row["source_parameters"],
                                    "input_receipt": receipt, "actual_input_tokens": 10, "raw_answer": "RIGHT", "output_token_ids": [5, 9],
                                    "finish_reason": "stop", "stop_reason": 9, "seconds": cold+repeat+0.1,
                                    "effective_generation": inference.GENERATION, "replay_parent_sample_id": sid,
                                    "adapter_replay": {"source_input": source_input, "source_input_sha256": storage.digest(source_input),
                                        "plan_sha256": storage.digest(plan), "route": route, "cold_pass": cold, "repeat": repeat}}
                                store.commit_output(sid, raw)
                                store.event("POST_OUTPUT_INPUT_RECEIPT", sample_id=sid, attempt_id=attempt, native_guard=True,
                                    **({"t1_deep_qualification": True, "t1_expected_hashes": {"input_sha256": "a"*64, "projection_sha256": "c"*64}} if route == "video" else {}))
                                store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=attempt)
                                store.event("REPLAY_CALL_COMPLETED", sample_id=sid, attempt_id=attempt, seconds=1.0)
                                binding = {"profile_id": helper.PROFILE_ID, "family": family, "route": route,
                                    "coverage_sha256": plan["coverage"]["sha256"], "adapter_sha256": replay._adapter_descriptor(bound["entry"])["sha256"],
                                    "instrumentation_sha256": plan["profile"]["implementation"]["adapter_helper_sha256"], "row_sha256": storage.digest(row)}
                                store.event("REPRO_INPUT_HASH", sample_id=sid, stage="cpu_processor", tensors=[fingerprint("input_ids", "torch.int64"), fingerprint("pixel_values", "torch.float32")], **binding)
                                store.event("REPRO_INPUT_HASH", sample_id=sid, stage="gpu_forward", tensors=[fingerprint("pixels")], **binding)
                            store.event("REPLAY_REPEAT_COMPLETE", sample_ids=ids)
    return SimpleNamespace(plan=plan, coverage=coverage, path=plan_path, output=output, rows=all_rows, materialize=materialize, root=tmp_path)


def rewrite_journal(path, change):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    change(rows)
    path.write_text("".join(storage.canonical_json(r)+"\n" for r in rows))


def test_all_30_inputs_require_four_exact_runs_and_keep_wrong_answers(suite):
    suite.materialize()
    result = replay.compare(suite.path, suite.output, verify_physical=False)
    assert result["status"] == "PASS", result["errors"][:3]
    assert all(m["checked_stores"] == 16 for m in result["models"].values())
    assert result["models"]["nemo"]["correct_outputs"] == 60
    assert result["remaining_models"] == [] and result["all_plan_models_certified"] is True
    assert sum(replay.COUNTS.values()) == 30


@pytest.mark.parametrize("missing", [False, True])
def test_model_subset_still_requires_all_sixteen_stores_and_labels_remaining_models(suite, missing):
    suite.materialize()
    if missing:
        path = suite.output / "cold-1/pair/repeat-1/nemo/events.jsonl"
        rewrite_journal(path, lambda events: events.__setitem__(slice(None), [e for e in events if e.get("stage") != "gpu_forward"]))
    result = replay.compare(suite.path, suite.output, model="nemo", verify_physical=False)
    assert set(result["models"]) == {"nemo"} and result["requested_models"] == ["nemo"]
    assert result["remaining_models"] == ["qwen", "cosmos"] and result["all_plan_models_certified"] is False
    assert result["models"]["nemo"]["expected_stores"] == 16
    assert result["models"]["nemo"]["checked_stores"] == (15 if missing else 16)
    assert result["status"] == ("FAIL" if missing else "PASS")


@pytest.mark.parametrize("fault", ["missing_gpu", "metadata_sha", "token_cold", "raw_length", "lease", "native_post", "pending_qsa", "wrong_fill", "missing_completion"])
def test_fail_closed_for_missing_or_changed_evidence(suite, fault):
    suite.materialize()
    route = "single" if fault == "pending_qsa" else "pair"
    family = "qwen" if fault == "pending_qsa" else "nemo"
    directory = suite.output / "cold-1" / route / "repeat-1" / family
    raw_path, event_path = directory / "outputs.jsonl", directory / "events.jsonl"
    if fault in ("token_cold", "raw_length"):
        def change(rows):
            rows[0]["output_token_ids"] = [5, 10]
            if fault == "raw_length":
                rows[0]["finish_reason"] = "length"
        rewrite_journal(raw_path, change)
    elif fault == "missing_completion":
        (directory.parents[1] / (family+".json")).unlink()
    else:
        def change(events):
            if fault == "missing_gpu":
                events[:] = [e for e in events if not (e["event"] == "REPRO_INPUT_HASH" and e.get("stage") == "gpu_forward")]
            elif fault == "metadata_sha":
                next(e for e in events if e["event"] == "REPRO_INPUT_HASH" and e["stage"] == "gpu_forward")["tensors"] = [{"not_a_tensor_sha256": "a"*64}]
            elif fault == "lease":
                next(e for e in events if e["event"] == "REPLAY_PHYSICAL_LEASE")["lease"]["path"] = "/tmp/a-different-gpu.lock"
            elif fault == "native_post":
                next(e for e in events if e["event"] == "POST_OUTPUT_INPUT_RECEIPT")["native_guard"] = False
            else:
                snap = next(e for e in events if e["event"] == "REPLAY_RUNTIME")["snapshot"]["comparison_identity"]
                if fault == "pending_qsa":
                    snap["profile"]["qsa_installation"]["state"] = "pending_target_import"
                else:
                    snap["fill_uninitialized_memory"] = False
        rewrite_journal(event_path, change)
    result = replay.compare(suite.path, suite.output, verify_physical=False)
    assert result["status"] == "FAIL"
    assert result["models"][family]["errors"]
    if fault == "raw_length":
        assert json.loads(raw_path.read_text().splitlines()[0])["finish_reason"] == "length"


def test_source_drift_rejected_before_replay(suite):
    entry = replay._route(suite.coverage, "pair")
    path = Path(entry["implementation_files"][0]["path"])
    path.write_text(path.read_text()+"# drift\n")
    with pytest.raises(RuntimeError, match="Source SHA/size drift"):
        replay.bind_route(suite.plan, suite.coverage, "pair", "cosmos", assets=False)


def test_equal_but_unidentified_packages_and_missing_cuda_all_repeats_fail(suite):
    suite.materialize()
    for path in suite.output.glob("cold-*/*/repeat-*/*/events.jsonl"):
        def change(events):
            snap = next(e for e in events if e["event"] == "REPLAY_RUNTIME")["snapshot"]["comparison_identity"]
            snap["packages"] = {"unrelated": "present"}
            snap.pop("cuda")
        rewrite_journal(path, change)
    result = replay.compare(suite.path, suite.output, verify_physical=False)
    assert result["status"] == "FAIL"
    assert all(m["checked_stores"] == 0 for m in result["models"].values())


@pytest.mark.parametrize("missing", [False, True])
def test_equal_video_posts_cannot_disable_or_omit_native_deep_qualification(suite, missing):
    suite.materialize()
    for path in suite.output.glob("cold-*/video/repeat-*/*/events.jsonl"):
        def change(events):
            for event in events:
                if event["event"] == "POST_OUTPUT_INPUT_RECEIPT":
                    if missing:
                        event.pop("t1_deep_qualification")
                    else:
                        event["t1_deep_qualification"] = False
        rewrite_journal(path, change)
    result = replay.compare(suite.path, suite.output, verify_physical=False)
    assert result["status"] == "FAIL"
    assert all(any("native deep qualification" in e for e in m["errors"]) for m in result["models"].values())


def test_original_geometry_is_scoped_and_does_not_claim_each_selected_row(suite):
    bound = replay.bind_route(suite.plan, suite.coverage, "multi", "cosmos", assets=False)
    assert bound["geometry"]["scope"] == "ORIGINAL_NATIVE_GEOMETRY_NOT_30_INDIVIDUAL_CALIBRATIONS"
    value = replay.read(bound["calibration"]["path"])
    for record in value["records"]:
        record["input_receipt"]["images_per_context"] = 2
    with pytest.raises(RuntimeError, match="two/four-image"):
        replay.validate_geometry(value, bound["entry"], {r["sample_id"]: r for r in bound["rows"]}, "cosmos")


def test_longest_host_specific_mapping_keeps_radon_static_scripts_together(tmp_path):
    mappings = [{"source": "/B", "target": "/mnt/B", "host": "Radon"},
        {"source": "/B/runs/t2-t3-static", "target": "/mnt/B/runs/t2_t3_static_v1", "host": "Radon"},
        {"source": "/B/runs/t2-t3-static/code", "target": "/mnt/B/runs/t2_t3_static_v1", "host": "Radon"}]
    assert replay.relocated("/B/runs/t2-t3-static/code/static_worker.py", mappings, "Radon") == Path("/mnt/B/runs/t2_t3_static_v1/static_worker.py")
    assert replay.relocated("/B/runs/t2-t3-static/execution-manifest.json", mappings, "Radon") == Path("/mnt/B/runs/t2_t3_static_v1/execution-manifest.json")


def test_physical_lease_uses_native_flock_and_admission_before_operation(tmp_path, monkeypatch):
    plan = {"gpu_lock_root": str(tmp_path / "original/evaluation/outputs"), "path_map": []}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: SimpleNamespace(uuid="CPU-LEASE-ONLY")))
    entered = []
    @contextmanager
    def lock(root, tensor_module):
        entered.append("lock")
        path = Path(root) / (".gpu-"+storage.digest({"host": socket.gethostname(), "device": "CPU-LEASE-ONLY"})+".lock")
        path.parent.mkdir(parents=True)
        path.touch()
        try:
            yield
        finally:
            entered.append("unlock")
    def resources(torch, budgets):
        assert entered == ["lock"]
        return {"admitted": True, "budgets": budgets}
    base = SimpleNamespace(_gpu_lock=lock, resource_snapshot=resources)
    with replay.physical_lease(base, torch, plan, {"min_free_gpu_gib": 1}) as lease:
        assert entered == ["lock"] and lease["st_ino"]
    assert entered == ["lock", "unlock"]


def test_resource_rejection_cannot_enter_model_operation(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: SimpleNamespace(uuid="CPU")))
    @contextmanager
    def lock(*args):
        yield
    base = SimpleNamespace(_gpu_lock=lock, resource_snapshot=lambda *args: {"admitted": False})
    with pytest.raises(RuntimeError, match="resource admission"):
        with replay.physical_lease(base, torch, {"gpu_lock_root": str(tmp_path), "path_map": []}, {}):
            pytest.fail("model operation entered")


def test_actual_native_gpu_flock_excludes_a_second_owner_without_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    torch = SimpleNamespace(cuda=SimpleNamespace(get_device_properties=lambda i: SimpleNamespace(uuid="CPU-FLOCK-ONLY")))
    base = SimpleNamespace(_gpu_lock=inference._gpu_lock, resource_snapshot=lambda *a: {"admitted": True})
    plan = {"gpu_lock_root": str(tmp_path), "path_map": []}
    with replay.physical_lease(base, torch, plan, {}) as lease:
        assert Path(lease["path"]).stat().st_ino == lease["st_ino"]
        with pytest.raises(BlockingIOError):
            with replay.physical_lease(base, torch, plan, {}):
                pytest.fail("Native physical GPU lock was not exclusive")


def test_incomplete_claim_cannot_resume_or_call_a_model(suite, monkeypatch):
    root = suite.output / "cold-0/pair"
    write(root / "cosmos.claim.json", {"status": "interrupted"})
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(replay, "validate_plan", lambda *a, **k: suite.coverage)
    monkeypatch.setattr(replay, "_native", lambda *a: pytest.fail("Model code should not be reached"))
    for key, value in replay._environment(suite.plan, "cosmos", "pair").items():
        monkeypatch.setenv(key, value)
    with pytest.raises(FileExistsError):
        replay.run(suite.path, suite.output, "cosmos", "pair", 0, "0")


def test_gemma_execution_and_nonfresh_process_are_rejected_without_loading(suite):
    with pytest.raises(RuntimeError, match="Gemma"):
        replay.run(suite.path, suite.output, "gemma", "pair", 0, "0")
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "torch", object())
        with pytest.raises(RuntimeError, match="fresh process"):
            replay.run(suite.path, suite.output, "cosmos", "pair", 0, "0")


def test_missing_model_state_and_fake_lock_root_are_rejected(suite):
    value = copy.deepcopy(suite.plan)
    value["model_state"]["cosmos"]["pair"].pop("processor_source")
    with pytest.raises(RuntimeError, match="CPU model-state"):
        replay.validate_plan(value, physical=False)
    value = copy.deepcopy(suite.plan)
    value["gpu_lock_root"] = str(suite.root / "isolated-fake-lock")
    with pytest.raises(RuntimeError, match="original T1 shared lock"):
        replay.validate_plan(value, physical=False)


def test_native_processor_anchor_ignores_import_label_but_requires_actual_source_api(suite):
    bound = replay.bind_route(suite.plan, suite.coverage, "pair", "cosmos", assets=False)
    state = copy.deepcopy(suite.plan["model_state"]["cosmos"]["pair"])
    state["processor_source"]["module"] = "__main__"
    replay.native_state_anchor(state, bound, suite.plan, "cosmos")
    state["processor_source"]["qualname"] = "ChangedProcessor"
    with pytest.raises(RuntimeError, match="processor event"):
        replay.native_state_anchor(state, bound, suite.plan, "cosmos")


@pytest.mark.parametrize("tamper", [False, True])
def test_portable_native_anchor_bundle_preserves_original_identity_bytes(suite, tamper):
    state = copy.deepcopy(suite.plan["model_state"]["cosmos"]["pair"])
    text = Path(state["native_identity"]["path"]).read_text()
    event = json.loads(Path(state["processor_receipt"]["path"]).read_text())
    if tamper:
        text += " "
    bundle = {"records": {"pair/cosmos": {"identity_source_utf8": text,
        "identity_origin": state.pop("native_identity"), "processor_event": event,
        "processor_origin": state.pop("processor_receipt")}}}
    state["anchor_bundle"] = write(suite.root / "portable-anchors.json", bundle)
    state["anchor_key"] = "pair/cosmos"
    bound = replay.bind_route(suite.plan, suite.coverage, "pair", "cosmos", assets=False)
    if tamper:
        with pytest.raises(RuntimeError, match="original identity bytes"):
            replay.native_state_anchor(state, bound, suite.plan, "cosmos")
    else:
        replay.native_state_anchor(state, bound, suite.plan, "cosmos")


def test_lazy_qsa_is_applied_before_first_runtime_policy_snapshot(suite, monkeypatch):
    from classic_precision import reproducibility
    order, state = [], {"state": "pending_target_import"}
    qsa = SimpleNamespace(TARGET_MODULE="CPU_QSA_LEAF", status=lambda: state)
    def guarded_import(name):
        if name == "classic_precision_qsa_determinism":
            order.append("qsa_pending")
            assert state["state"] == "pending_target_import"
            return qsa
        assert name == qsa.TARGET_MODULE
        order.append("target_import")
        state["state"] = "applied"
    monkeypatch.setattr(reproducibility, "importlib", SimpleNamespace(import_module=guarded_import))
    env = replay._environment(suite.plan, "qwen", "single")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    def actual_snapshot():
        order.append("runtime_snapshot")
        assert state["state"] == "applied"
        return snapshot(suite.plan, "qwen", env)
    monkeypatch.setattr(helper._core(), "runtime_snapshot", actual_snapshot)
    value = replay.pre_call_runtime(suite.plan, "qwen", env)
    assert order == ["qsa_pending", "target_import", "runtime_snapshot"]
    assert value["comparison_identity"]["profile"]["qsa_installation"]["state"] == "applied"


def test_actual_original_npz_routes_have_distinct_pixels_and_native_ready_bindings(monkeypatch):
    path = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/coverage.json")
    if not path.exists():
        pytest.skip("Source benchmark checkout absent")
    coverage = replay.read(path)
    plan = {"path_map": [], "calibrations": {r["route_id"]: {q["model"]: q["file"] for q in r["existing_cpu_qualification"]} for r in coverage["routes"]}}
    for route in ("multi", "pair"):
        bound = replay.bind_route(plan, coverage, route, "cosmos", assets=True)
        name = "actual_replay_cpu_"+route
        spec = importlib.util.spec_from_file_location(name, bound["adapter"])
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        for row in bound["rows"]:
            arrays, ready = module.read_images(row) if route == "multi" else module.read_pair(bound["source"][row["sample_id"]])
            assert ready == replay._ready(bound, row)
            assert len({hashlib.sha256(a.tobytes()).hexdigest() for a in arrays}) == len(arrays)


@pytest.mark.parametrize("family", ["nemo", "qwen", "cosmos", "gemma"])
def test_actual_fixed_static_manifest_uses_original_protocol_or_gemma_panel(family):
    path = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/coverage.json")
    if not path.exists():
        pytest.skip("Source benchmark checkout absent")
    coverage = replay.read(path)
    entry = replay._route(coverage, "single")
    manifest = replay.read(replay.verified(entry["manifest"]))
    assert "models" not in manifest
    spec, source = replay.native_model_spec(entry, manifest, family, [])
    expected_name = "four-model-panel.json" if family == "gemma" else "protocol.json"
    assert Path(source["file"]["path"]).name == expected_name
    assert spec == replay.read(replay.verified(source["file"]))["models"][family]
    assert "checkpoint_relative" in spec and set(spec["resources"]) >= {"min_free_gpu_gib", "min_available_ram_gib"}
    if family in ("qwen", "cosmos"):
        plan = {"path_map": [], "calibrations": {"single": {q["model"]: q["file"] for q in entry["existing_cpu_qualification"]}}}
        bound = replay.bind_route(plan, coverage, "single", family, assets=False)
        assert bound["spec"] == spec and len(bound["rows"]) == 12
        assert bound["model_spec_source"] == source


def test_static_spec_never_falls_back_to_an_invented_manifest_model(suite):
    entry = replay._route(suite.coverage, "single")
    manifest = replay.read(entry["manifest"]["path"])
    manifest["models"] = {"nemo": {"checkpoint_relative": "invented"}}
    spec, source = replay.native_model_spec(entry, manifest, "nemo", [])
    assert spec["checkpoint_relative"] == "nemo"
    entry = copy.deepcopy(entry)
    entry["protocols"] = []
    with pytest.raises(RuntimeError, match="source descriptor missing"):
        replay.native_model_spec(entry, manifest, "nemo", [])


def test_static_spec_protocol_sha_drift_and_manifest_link_mismatch_are_rejected(suite):
    entry = replay._route(suite.coverage, "single")
    manifest = replay.read(entry["manifest"]["path"])
    changed = copy.deepcopy(manifest)
    changed["protocol_sha256"] = "0"*64
    with pytest.raises(RuntimeError, match="protocol binding differs"):
        replay.native_model_spec(entry, changed, "cosmos", [])
    path = Path(entry["protocols"][0]["path"])
    path.write_text(path.read_text()+"\n")
    with pytest.raises(RuntimeError, match="Source SHA/size drift"):
        replay.native_model_spec(entry, manifest, "cosmos", [])


def test_actual_video_ready_matches_frozen_native_input_catalog_projection():
    path = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/coverage.json")
    if not path.exists():
        pytest.skip("Source benchmark checkout absent")
    entry = replay._route(replay.read(path), "video")
    manifest_path = replay.verified(entry["manifest"])
    rows = replay._rows(replay.read(manifest_path))
    source = replay._rows(replay.read(replay.verified(entry["source_manifest"])))
    adapter = replay.verified(replay._adapter_descriptor(entry))
    tree = ast.parse(adapter.read_text())
    catalog = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "input_catalog")
    # Execute only the original source's dictionary expression: no model,
    # video decode, filesystem mutation or weakened native require() calls.
    assignment = next(n for n in ast.walk(catalog) if isinstance(n, ast.Assign) and
        any(isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id == "ready" for t in n.targets))
    expression = compile(ast.Expression(assignment.value), str(adapter), "eval")
    for representative in entry["representative_samples"]:
        sid = representative["sample_id"]
        descriptor = representative["source_assets"][0]["ready_file"]
        raw = replay.read(replay.verified(descriptor))
        assert "video_path" in raw and "video_relative" not in raw
        expected = eval(expression, {"ROOT": manifest_path.parent, "sha": replay.file_sha256},
                        {"sid": sid, "r": raw, "old": source[sid], "row": rows[sid]})
        actual = replay.native_video_ready(rows[sid], source[sid], raw, descriptor["sha256"])
        assert actual == expected
        assert actual["video_relative"] == raw["video_path"]
        assert actual["source_ready_sha256"] == descriptor["sha256"]


@pytest.mark.parametrize("fault", ["technical", "scientific", "media_stat", "ready_sha"])
def test_video_ready_projection_keeps_original_source_guards(suite, fault):
    bound = replay.bind_route(suite.plan, suite.coverage, "video", "cosmos", assets=False)
    row = bound["rows"][0]
    representative = bound["entry"]["representative_samples"][0]
    asset = representative["source_assets"][0]
    raw = replay.read(asset["ready_file"]["path"])
    if fault == "technical":
        raw["technical_verdict"] = "FAIL"
    elif fault == "scientific":
        raw["scientific_spec_sha256"] = "0"*64
    elif fault == "media_stat":
        Path(asset["path"]).write_bytes(b"changed media stat")
        with pytest.raises(RuntimeError, match="media stat changed"):
            replay._ready(bound, row)
        return
    else:
        Path(asset["ready_file"]["path"]).write_text("{}")
        with pytest.raises(RuntimeError, match="ready source SHA/size drift"):
            replay._ready(bound, row)
        return
    with pytest.raises(RuntimeError, match="source ready identity failed"):
        replay.native_video_ready(row, bound["source"][row["sample_id"]], raw, asset["ready_file"]["sha256"])


def test_video_native_ready_failure_happens_before_engine_or_new_claim(suite, monkeypatch):
    entry = replay._route(suite.coverage, "video")
    descriptor = entry["representative_samples"][0]["source_assets"][0]["ready_file"]
    raw = replay.read(descriptor["path"]); raw["technical_verdict"] = "FAIL"
    descriptor.update(write(Path(descriptor["path"]), raw))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delitem(sys.modules, "vllm", raising=False)
    monkeypatch.setattr(replay, "validate_plan", lambda *a, **k: suite.coverage)
    monkeypatch.setattr(replay, "_native", lambda *a: pytest.fail("Engine must not be reached"))
    for key, value in replay._environment(suite.plan, "cosmos", "video").items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match="source ready identity failed"):
        replay.run(suite.path, suite.output, "cosmos", "video", 0, "0")
    assert not list(suite.output.glob("**/*.claim.json"))


def test_actual_single_ready_is_original_manifest_image_and_call_shapes_match():
    path = Path("/Users/qianyang/Desktop/cvpr2027_benchmark/runs/reproducibility/coverage.json")
    if not path.exists():
        pytest.skip("Source benchmark checkout absent")
    coverage = replay.read(path)
    entry = replay._route(coverage, "single")
    plan = {"path_map": [], "calibrations": {"single": {q["model"]: q["file"] for q in entry["existing_cpu_qualification"]}}}
    bound = replay.bind_route(plan, coverage, "single", "cosmos", assets=False)
    images = {i["image_id"]: i for i in bound["manifest"]["images"]}
    for row in bound["rows"]:
        assert replay._ready(bound, row) is images[row["image_id"]]
        assert {"image_relative", "file_sha256", "pixel_sha256"} <= images[row["image_id"]].keys()
    for route, method, expected in [
        ("single", "generate_image", ["self", "row", "image", "durable_output", "deep"]),
        ("multi", "generate_images", ["self", "row", "arrays", "ready", "durable", "deep"]),
        ("pair", "generate_pair", ["self", "row", "arrays", "ready", "durable", "deep"])]:
        entry = replay._route(coverage, route)
        name = "static_worker.py" if route == "single" else Path(replay._adapter_descriptor(entry)["path"]).name
        source = next(i for i in entry["implementation_files"] if Path(i["path"]).name == name)
        tree = ast.parse(replay.verified(source).read_text())
        native = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method)
        assert [arg.arg for arg in native.args.args] == expected
