from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from classic_precision import gemma_production as p


def test_explicit_common_C3_speed_tradeoff_binds_exact_three_results():
    timings = [{"ordinary_n": 18, "p50_seconds": x, "p90_seconds": y}
               for x, y in [(24.427, 24.576), (24.898, 25.077), (28.459, 28.595)]]
    hashes = [str(i) * 64 for i in (1, 2, 3)]
    auth = {"schema": "gemma-C3-speed-acceptance-v1",
            "decision": "PROCEED_WITH_COMMON_C3_AND_STRICT_REPLAY",
            "max_card_p50_seconds": 29, "max_card_p90_seconds": 30,
            "max_effective_seconds": 26.5, "qualified_result_sha256": hashes}
    with pytest.raises(ValueError):
        p.speed_acceptance(timings, hashes)
    p.speed_acceptance(timings, hashes, auth)
    for key, value in [("qualified_result_sha256", hashes[:2]), ("max_card_p50_seconds", 40),
                       ("decision", "ACCEPT_DIFFERENT_OUTPUTS")]:
        with pytest.raises(ValueError):
            p.speed_acceptance(timings, hashes, {**auth, key: value})
    bad = deepcopy(timings); bad[-1]["p90_seconds"] = 30.01
    with pytest.raises(ValueError):
        p.speed_acceptance(bad, hashes, auth)


def fixtures(count=4):
    expected = {"T1": count, "T2": count, "T3": count}
    panels, qa, reference_rows, qualified = [], [], {}, []
    for task_index, task in enumerate(expected):
        panel = {"key": task.lower(), "manifest_sha256": p.g.digest(task),
                 "expected_counts": {task: count}, "rows": [], "readies": []}
        for index in range(count):
            sid = f"{task}-{index}"
            stream = p.g.digest([sid, "stream"])
            row = {"sample_id": sid, "task": task, "prompt": f"exact {task} prompt"}
            ready = {"sample_id": sid, "video_relative": f"videos/{sid}.mkv",
                     "video_sha256": p.g.digest([sid, "video"]), "stream_sha256": stream}
            panel["rows"].append(row); panel["readies"].append(ready)
            qa.append({"sample_id": sid, "task": task, "status": "PASS", "panel": panel["key"],
                "manifest_sha256": panel["manifest_sha256"], "row_sha256": p.g.digest(row),
                "ready_sha256": p.g.digest(ready), "video_sha256": ready["video_sha256"],
                "decoded_stream_sha256": stream, "dimensions_wh": [768, 512],
                "decoded_frames": 180, "codec": "ffv1", "pixel_format": "gbrp16le",
                "fps": 60, "max_code_error": 0, "max_PTS_error_s": .000333,
                "identity": p.g.digest("existing completed source QA")})
            if index < 2:
                qualified.append({"sample_id": sid, "task": task, "row": deepcopy(row),
                    "ready": deepcopy(ready), "row_sha256": p.g.digest(row),
                    "ready_sha256": p.g.digest(ready)})
                reference_rows[sid] = {"input_token_ids": [10, task_index, 20],
                    "input_token_ids_sha256": p.g.digest([10, task_index, 20]),
                    "prompt_sha256": p.g.digest(task), "position_sha256": p.g.digest("positions"),
                    "position_shape": [180, 2520, 2],
                    "HF_control_tensors": {"positions": {"sha256": p.g.digest("control")}},
                    "source_stream_sha256": stream, "source_video_sha256": ready["video_sha256"],
                    "input_sha256": p.g.digest([sid, "existing deep CPU bytes"])}
        panels.append(panel)
    qpanel = {"items": qualified, "selection": "fixed existing qualification IDs"}
    reference = {"panel_sha256": p.g.digest(qpanel), "rows": reference_rows}
    completion = {"status": "COMPLETE_PASS", "failed_IDs": [], "counts": expected,
                  "unique_IDs": 3 * count, "decoded_frames": 540 * count}
    return panels, qa, completion, reference, qpanel, expected


def catalog():
    data = fixtures()
    return p.build_catalog(*data[:-1], expected_counts=data[-1]), data[3]


def test_catalog_joins_source_families_and_templates_without_model_or_processor():
    value, reference = catalog()
    assert value["counts"] == {"T1": 4, "T2": 4, "T3": 4}
    assert len(value["items"]) == 12 and len(value["templates"]) == 3
    assert value["CPU_processor_calls"] == value["GPU_model_calls"] == 0
    assert not value["production_runner_implemented"]
    refs = p.CatalogReferences(value, reference)
    assert set(refs) == set(value["items"])
    assert refs["T2-2"]["source_stream_sha256"] != refs["T2-0"]["source_stream_sha256"]
    assert refs["T2-2"]["HF_control_tensors"] == refs["T2-0"]["HF_control_tensors"]
    refs["T2-2"]["input_token_ids"].append(999)
    assert refs["T2-0"]["input_token_ids"] == [10, 1, 20]


@pytest.mark.parametrize("change", ["row", "ready", "manifest", "stream", "shape", "frames", "codec", "code_error", "PTS"])
def test_changed_identity_or_geometry_is_refused(change):
    panels, audit, completion, ref, panel, expected = fixtures()
    qa = audit[-1]
    if change in ("row", "ready", "manifest"):
        qa[change + "_sha256"] = "0" * 64
    elif change == "stream": qa["decoded_stream_sha256"] = "1" * 64
    elif change == "shape": qa["dimensions_wh"] = [512, 768]
    elif change == "frames": qa["decoded_frames"] = 179
    elif change == "codec": qa["codec"] = "h264"
    elif change == "code_error": qa["max_code_error"] = 1
    else: qa["max_PTS_error_s"] = .01
    with pytest.raises(ValueError):
        p.build_catalog(panels, audit, completion, ref, panel, expected_counts=expected)


def test_unknown_exact_prompt_cannot_borrow_task_template():
    panels, audit, completion, ref, panel, expected = fixtures()
    row = panels[-1]["rows"][-1]
    row["prompt"] += " unqualified change"
    audit[-1]["row_sha256"] = p.g.digest(row)
    with pytest.raises(ValueError, match="Exact prompt"):
        p.build_catalog(panels, audit, completion, ref, panel, expected_counts=expected)


def test_missing_duplicate_or_incomplete_coverage_is_refused():
    data = fixtures()
    data[1].append(deepcopy(data[1][0]))
    with pytest.raises(ValueError, match="audit IDs"):
        p.build_catalog(*data[:-1], expected_counts=data[-1])
    data = fixtures(); data[0][1]["readies"].pop()
    with pytest.raises(ValueError, match="Ready and manifest"):
        p.build_catalog(*data[:-1], expected_counts=data[-1])
    data = fixtures(); data[2]["failed_IDs"] = ["T3-3"]
    with pytest.raises(ValueError, match="not complete"):
        p.build_catalog(*data[:-1], expected_counts=data[-1])


def test_family_roots_and_safe_paths(tmp_path):
    value, reference = catalog(); refs = p.CatalogReferences(value, reference)
    roots = {task.lower(): str(tmp_path / task) for task in ("T1", "T2", "T3")}
    assert refs.roots_for("T2-3", roots) == {"T2": str(tmp_path / "T2")}
    value["items"]["T2-3"]["ready"]["video_relative"] = "../../another/source.mkv"
    with pytest.raises(ValueError, match="escaped"):
        refs.roots_for("T2-3", roots)


def test_only_existing_uncompleted_ids_can_supply_first_two_deep():
    value, reference = catalog(); refs = p.CatalogReferences(value, reference)
    ids = ["T3-3", "T3-2", "T3-1", "T3-0"]
    assert refs.initial_order("T3", ids) == ["T3-0", "T3-1", "T3-3", "T3-2"]
    assert refs.deep("T3-0")["input_sha256"] == reference["rows"]["T3-0"]["input_sha256"]
    with pytest.raises(ValueError, match="No existing deep"):
        refs.deep("T3-3")
    with pytest.raises(ValueError, match="two uncompleted"):
        refs.initial_order("T3", ["T3-3", "T3-2"])
    with pytest.raises(ValueError, match="wrong or duplicate"):
        refs.initial_order("T3", ["T3-0", "T1-1"])


def test_deep_reference_bytes_cannot_drift_after_catalog_sealing():
    value, reference = catalog()
    reference["rows"]["T1-0"]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CPU reference changed"):
        p.CatalogReferences(value, reference)


def test_dispatch_prevents_cross_worker_duplicate_and_uncertain_reissue(tmp_path):
    value, _ = catalog(); path = tmp_path / "dispatch.json"
    a, b = p.DispatchLedger(path, value), p.DispatchLedger(path, value)
    first = a.claim("fourier0", ["T1-0", "T1-1"], p.g.digest("runtime"))
    assert json.loads(path.read_text())["assignments"]["T1-0"]["status"] == "ASSIGNED"
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already owned"):
        b.claim("radon0", ["T1-1", "T1-2"], p.g.digest("runtime"))
    assert path.read_bytes() == before
    a.record("T1-0", "fourier0", first["batch_id"], "UNCERTAIN", p.g.digest("intent exists"))
    with pytest.raises(ValueError, match="already owned"):
        b.claim("radon0", ["T1-0"], p.g.digest("runtime"))
    second = b.claim("radon0", ["T1-2", "T1-3"], p.g.digest("runtime"))
    assert first["batch_id"] != second["batch_id"]


def test_dispatch_rejects_wrong_outcome_owner_and_catalog_change(tmp_path):
    value, _ = catalog(); path = tmp_path / "dispatch.json"
    ledger = p.DispatchLedger(path, value)
    batch = ledger.claim("fourier0", ["T1-0"], p.g.digest("runtime"))
    with pytest.raises(ValueError, match="ownership"):
        ledger.record("T1-0", "fourier1", batch["batch_id"], "COMPLETE", p.g.digest("proof"))
    ledger.record("T1-0", "fourier0", batch["batch_id"], "COMPLETE", p.g.digest("proof"))
    with pytest.raises(ValueError, match="ownership"):
        ledger.record("T1-0", "fourier0", batch["batch_id"], "COMPLETE", p.g.digest("proof"))
    other = deepcopy(value); other["items"]["T2-0"]["template_key"] = "changed"
    with pytest.raises(ValueError, match="catalog changed"):
        p.DispatchLedger(path, other).claim("radon0", ["T3-0"], p.g.digest("runtime"))


def test_interrupted_transaction_evidence_is_not_overwritten(tmp_path):
    value, _ = catalog(); path = tmp_path / "dispatch.json"
    evidence = tmp_path / "dispatch.json.transaction-tmp"
    evidence.write_text("partial interrupted transaction")
    with pytest.raises(FileExistsError):
        p.DispatchLedger(path, value).claim("fourier0", ["T1-0"], p.g.digest("runtime"))
    assert not path.exists() and evidence.read_text() == "partial interrupted transaction"


def test_result_batch_is_atomic_and_invalid_signatures_are_rejected(tmp_path):
    value, _ = catalog(); path = tmp_path / "dispatch.json"
    ledger = p.DispatchLedger(path, value)
    with pytest.raises(ValueError, match="identity"):
        ledger.claim("fourier0", ["T1-0"], "x" * 64)
    batch = ledger.claim("fourier0", ["T1-0", "T1-1"], p.g.digest("runtime"))
    before = path.read_bytes()
    outcomes = [{"sample_id": sid, "status": "COMPLETE", "evidence_sha256": p.g.digest(sid)}
                for sid in ["T1-0", "T1-2"]]
    with pytest.raises(ValueError, match="ownership"):
        ledger.record_many("fourier0", batch["batch_id"], outcomes)
    assert path.read_bytes() == before
    outcomes[1]["sample_id"] = "T1-1"
    ledger.record_many("fourier0", batch["batch_id"], outcomes)
    assert all(v["status"] == "COMPLETE" for v in json.loads(path.read_text())["assignments"].values())


def test_sealed_catalog_is_never_overwritten(tmp_path):
    path = tmp_path / "sealed.json"
    p._write_new(path, {"identity": "first"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError): p._write_new(path, {"identity": "different"})
    assert path.read_bytes() == before


def test_import_and_CLI_are_CPU_only():
    result = subprocess.run([sys.executable, "-B", "-c", "import classic_precision.gemma_production,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-B", "-m", "classic_precision.gemma_production", "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "seal-inputs" in result.stdout


# New entry tests use CPU doubles; no processor, weights or GPU is loaded.
from argparse import Namespace
from contextlib import contextmanager
from types import SimpleNamespace
import ast
import classic_precision
from classic_precision import gemma_cpu_copy as cpu
from test_gemma_cpu_copy import base_profile, SDK


@pytest.fixture
def worker_case(tmp_path, monkeypatch):
    data = fixtures(16)
    value = p.build_catalog(*data[:-1], expected_counts=data[-1])
    items = list(value["items"].values())
    panels = {str(n): {"items": items[:n], "scope": "CPU fixtures"} for n in (20, 48)}
    reference_rows = {}
    for item in items:
        sid = item["sample_id"]
        controls = deepcopy(value["templates"][item["template_key"]]["controls"])
        controls.update(source_stream_sha256=item["source_stream_sha256"], source_video_sha256=item["ready"]["video_sha256"])
        for key in ("processor_input_FP32_sha256", "pixel_FP32_sha256", "input_sha256", "projection_sha256"):
            controls[key] = p.g.digest([sid, key])
        reference_rows[sid] = controls
    refs = {str(n): {"panel_sha256": p.g.digest(panels[str(n)]), "rows": {x["sample_id"]: reference_rows[x["sample_id"]] for x in items[:n]}} for n in (20, 48)}
    value["CPU_reference_value_sha256"] = p.g.digest(refs["20"])
    paths = {}
    for key, obj in {"catalog": value, "ref20": refs["20"], "ref48": refs["48"], "panel20": panels["20"], "panel48": panels["48"]}.items():
        paths[key] = tmp_path / (key + ".json"); p._write_new(paths[key], obj)
    monkeypatch.setattr(p, "CATALOG_SHA", p.g.file_sha256(paths["catalog"]))
    monkeypatch.setattr(cpu.m, "REFERENCE_SHA", {n: p.g.file_sha256(paths["ref" + str(n)]) for n in (20, 48)})
    q = base_profile()
    q["plan_sha256"] = p.g.digest("CPU fixture plan")
    q["runtime"].update(environment={}, asset_roots={"T1": str(tmp_path / "t1")}, lease_root=str(tmp_path / "t1/evaluation/outputs"))
    q["cpu_execution_overlay"] = cpu.overlay_record(deepcopy(q), "baseline", 8, cpu.processor_candidate(SDK)[1])
    paths["runtime"] = tmp_path / "runtime.json"; p._write_new(paths["runtime"], q)
    roots = {key: str(tmp_path / key) for key in value["source_families"]}
    paths["roots"] = tmp_path / "roots.json"; p._write_new(paths["roots"], roots)
    admission = {"plan_sha256": q["plan_sha256"], "qwen_t1_complete": 3840, "all_owned_engines_exited": True, "physical_gpu_locks_released": True}
    paths["admission"] = tmp_path / "admission.json"; p._write_new(paths["admission"], admission)
    def seal(index=0, n=20):
        output = tmp_path / f"worker-{index}.json"
        p.seal_worker(Namespace(runtime_profile=str(paths["runtime"]), catalog=str(paths["catalog"]), reference20=str(paths["ref20"]),
            reference48=str(paths["ref48"]), source_roots=str(paths["roots"]), name=f"worker-{index}", worker=f"worker-{index}",
            inbox=str(tmp_path / f"inbox-{index}"), initial_task="T1", output=str(output), mode=str(n),
            panel=str(paths["panel" + str(n)]), inbox_timeout_seconds=.15))
        return output
    return SimpleNamespace(root=tmp_path, catalog=value, q=q, roots=roots, paths=paths, refs=refs, rows=reference_rows, panels=panels, seal=seal)


def publish(case, profile_path, batches):
    profile = p.g.read(profile_path)
    ledger = p.DispatchLedger(case.root / (profile["worker"] + "-ledger.json"), case.catalog)
    ledger.register(profile)
    directory = Path(profile["inbox"]); directory.mkdir()
    for ids in batches:
        batch = ledger.claim(profile["worker"], ids, profile["execution_signature"])
        p._write_new(directory / f"batch-{batch['worker_sequence']:06d}.json", batch)
    p._write_new(directory / "DONE.json", ledger.close_worker(profile))
    return ledger


@pytest.fixture
def fake_runtime(worker_case, monkeypatch):
    case = worker_case
    history, generated, observations = [], [], []
    state = {"index": 0, "host": "fourier", "device": "GPU0", "fail_at": None, "sid": None}
    class Session:
        def __init__(self, *args): history.append("session_create")
        def __enter__(self): return self
        def __exit__(self, *args): history.append("session_exit")
    class Engine:
        def __init__(self, **settings):
            assert settings == case.q["engine_settings"]
            history.append("engine_create")
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(shutdown=lambda: history.append("engine_shutdown")))
        def collective_rpc(self, fn, args=()):
            name = fn.__name__
            owner = {"pid": 1000 + state["index"], "start_ticks": 500 + state["index"], "boot_id": state["host"]}
            if name == "_worker_identity": return [{"worker": owner}]
            if name == "_placement": return [{"worker": owner, "torch_policy": {"BF16": True}}]
            if name == "_install_kernel":
                return [{"configuration": args[0], "observed_native_sha256": p.g.NATIVE_SHA,
                    "derived_helpers_sha256": "32a56e3cb9501974cf648a3524691e84bd4a8c6dd7ead6c2ecd4b683c7b1ac4f",
                    "derived_launcher_sha256": "a9c0fd9df26aae3919a024de045a6bb59e67011c297f2de87f584acf2bfc855e"}]
            if name == "_install_observer": return [{}]
            if name == "_set_deep": state["deep"] = args[0]; return [None]
            if name == "_read_observer":
                receipt = case.rows[state["sid"]]
                video = {"shape": [180, 2520, 768], "dtype": "torch.bfloat16", "device": "cuda:0", "frames": [180],
                    "embedding_shapes": [[31680, 768]], **{k: receipt[k] for k in ("input_sha256", "position_sha256", "projection_sha256")}}
                return [{"video": [video], "native_masks": [[1, 5]], "CUDA_seconds": {"fixture": 0.}}]
            raise AssertionError(name)
        def generate(self, request, sampling, use_tqdm):
            sid = request["multi_modal_data"]["video"][0]
            generated.append(sid)
            if len(generated) == state["fail_at"]: raise RuntimeError("injected generate failure")
            p.g.CTX["adapter_calls"] = 1
            return [SimpleNamespace(outputs=[SimpleNamespace(text="LEFT", token_ids=[7, 8], finish_reason="stop", stop_reason=None)],
                                    prompt_token_ids=case.rows[sid]["input_token_ids"])]
    def prepare(session, item, roots, streamlined, deep):
        sid = item["sample_id"]; state["sid"] = sid
        observations.append((sid, deep, roots))
        return sid, {}, item["row"]["prompt"], deepcopy(case.rows[sid])
    @contextmanager
    def lease(runtime, torch):
        history.append("lease_enter")
        try: yield {"device": state["device"], "path": "/same-existing-lock", "st_dev": 1, "st_ino": 99}
        finally: history.append("lease_exit")
    original_numerical = p.numerical_context
    @contextmanager
    def numerical(profile, installed_sdk=False, install=False):
        with original_numerical(profile, False, False):
            if not install:
                yield cpu
                return
            @contextmanager
            def overlay(profile):
                old_prepare, old_gc = p.g.prepare, p.g.gc
                gc = cpu.GCCadence(8)
                p.g.prepare, p.g.gc = prepare, gc
                try: yield gc
                finally:
                    p.g.prepare, p.g.gc = old_prepare, old_gc
                    gc.finish()
            yield SimpleNamespace(installed_overlay=overlay)
    from classic_precision import inference
    fake_inference = SimpleNamespace(__file__=inference.__file__, ProcessorSession=Session,
        checkpoint_asset_identity=lambda checkpoint: case.q["checkpoint_assets"])
    monkeypatch.setattr(classic_precision, "inference", fake_inference)
    monkeypatch.setitem(sys.modules, "classic_precision.inference", fake_inference)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(backends=SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
        cudnn=SimpleNamespace(allow_tf32=False)), get_float32_matmul_precision=lambda: "highest"))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", SimpleNamespace(StructuredOutputsParams=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setattr(p.g, "sys", SimpleNamespace(modules={}, argv=["CPU fixture"], executable="CPU fixture"))
    monkeypatch.setattr(p.g, "socket", SimpleNamespace(gethostname=lambda: state["host"]))
    monkeypatch.setattr(p.g, "runtime_check", lambda runtime: {"CPU_fixture": True})
    monkeypatch.setattr(p.g, "runtime_sources", lambda: case.q["runtime_sources"])
    monkeypatch.setattr(p.g, "physical_lease", lease)
    monkeypatch.setattr(p.g, "process_identity", lambda: {"pid": 2000 + state["index"], "start_ticks": 600 + state["index"], "boot_id": state["host"]})
    monkeypatch.setattr(p.g, "owned_worker_alive", lambda owner: False)
    monkeypatch.setattr(p.g, "_install_adapter", lambda *args: history.append("adapter"))
    monkeypatch.setattr(p, "numerical_context", numerical)
    monkeypatch.setattr(cpu, "_GC", SimpleNamespace(collect=lambda *a, **k: 0))
    monkeypatch.setattr(cpu, "memory_observation", lambda: {"rss_bytes": 10})
    def execute(index=0, n=20, fail_at=None):
        state.update(index=index, host="fourier" if index < 4 else "radon", device=f"GPU{index // 2}" if index < 4 else "GPU0", fail_at=fail_at)
        generated.clear(); observations.clear(); history.clear()
        profile = case.seal(index, n)
        ids = [x["sample_id"] for x in case.panels[str(n)]["items"]]
        publish(case, profile, [ids[:1], ids[1:4], ids[4:17], ids[17:]])
        out = case.root / f"run-{index}"
        result = p.run_worker(Namespace(profile=str(profile), admission=str(case.paths["admission"]), output=str(out)))
        return out, result
    return SimpleNamespace(execute=execute, history=history, generated=generated, observations=observations, state=state)


def test_derived_entry_preserves_generation_timers_and_real_GC_site():
    derived = p.derive_runner()
    assert len(derived["changes"]) == 17 and derived["base_sha256"] == p.BASE_SHA
    tree = ast.parse(derived["source"])
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "gc.collect"]
    assert len(calls) == 1 and calls[0].lineno == derived["gc_line"] != 716
    assert len([n for n in ast.walk(tree) if isinstance(n, ast.Call) and ast.unparse(n.func) == "engine.generate"]) == 1


def test_persistent_engine_first_two_deep_across_batches_and_tasks(worker_case, fake_runtime):
    before_prepare, before_gc = p.g.prepare, p.g.gc
    out, result = fake_runtime.execute()
    assert result["status"] == "COMPLETE" and result["calls"] == 20
    assert fake_runtime.history.count("engine_create") == fake_runtime.history.count("session_create") == 1
    assert fake_runtime.history.index("engine_shutdown") < fake_runtime.history.index("lease_exit")
    assert [deep for _, deep, _ in fake_runtime.observations] == [True, True] + [False] * 18
    for sid, _, roots in fake_runtime.observations:
        item = worker_case.catalog["items"][sid]
        assert roots == {item["task"]: worker_case.roots[item["source_family"]]}
    assert p.g.prepare is before_prepare and p.g.gc is before_gc
    verified = p.verify_worker(out)
    assert verified["complete"] and len(verified["proofs"]) == 20
    receipt = p.g.read(out / "entry-receipt.json")
    assert receipt["gc"]["bound_to_actual_runner"] and receipt["gc"]["line"] == p.derive_runner()["gc_line"]
    assert receipt["gc"]["maximum_pending_ordinary"] == 8 and receipt["generation_attempts"] == 20
    original = p.g.read(out / "result.json")["rows"]
    service = receipt["through_durable_proof"]
    assert service["timing"]["ordinary_n"] == 18
    assert all(a["sample_id"] == b["sample_id"] and a["full_request_seconds"] >= b["full_request_seconds"]
               for a, b in zip(service["rows"], original, strict=True))
    assert receipt["worker_wall_seconds"] >= sum(row["full_request_seconds"] for row in service["rows"])


def test_generate_failure_is_uncertain_not_retried(worker_case, fake_runtime):
    before_prepare, before_gc = p.g.prepare, p.g.gc
    with pytest.raises(RuntimeError, match="injected generate"):
        fake_runtime.execute(fail_at=3)
    assert len(fake_runtime.generated) == 3
    out = worker_case.root / "run-0"
    verified = p.verify_worker(out, allow_incomplete=True)
    assert len(verified["proofs"]) == 2 and len(verified["uncertain"]) == 1
    assert verified["uncertain"][0]["sample_id"] == "T1-2"
    assert p.g.prepare is before_prepare and p.g.gc is before_gc
    assert "engine_shutdown" in fake_runtime.history and "lease_exit" in fake_runtime.history
    with pytest.raises(ValueError, match="incomplete"): p.verify_worker(out)


def test_hard_exit_retains_intent_without_inventing_final_receipt(worker_case, fake_runtime):
    with pytest.raises(RuntimeError, match="injected generate"):
        fake_runtime.execute(fail_at=3)
    out = worker_case.root / "run-0"
    for name in ("entry-closure.json", "entry-receipt.json", "exit.json"):
        (out / name).unlink()
    with (out / "calls.jsonl").open("ab") as f: f.write(b'{"partial":')
    with (out / "events.jsonl").open("ab") as f: f.write(b'{"partial":')
    report = p.verify_worker(out, allow_incomplete=True)
    assert not report["complete"] and not report["final_closure_present"]
    assert report["closure_sha256"] is None and len(report["proofs"]) == 2
    assert report["uncertain_requires_exit_confirmation"] is True
    assert [row["sample_id"] for row in report["uncertain"]] == ["T1-2"]
    assert len(fake_runtime.generated) == 3 and not (out / "exit.json").exists()
    with pytest.raises(ValueError, match="no final closure"): p.verify_worker(out)


def test_call_byte_proof_detects_tampering_even_if_closure_rehashed(worker_case, fake_runtime):
    out, _ = fake_runtime.execute()
    raw = out / "raw.jsonl"; raw.write_bytes(raw.read_bytes().replace(b'"LEFT"', b'"RIGH"', 1))
    closure = p.g.read(out / "entry-closure.json"); closure["files"]["raw.jsonl"] = p.g.file_sha256(raw)
    (out / "entry-closure.json").write_text(json.dumps(closure))
    with pytest.raises(ValueError, match="byte proof"): p.verify_worker(out)


def test_worker_owner_and_closed_authority_cannot_be_reused(worker_case, fake_runtime):
    fake_runtime.execute()
    profile_path = worker_case.root / "worker-0.json"
    with pytest.raises(FileExistsError):
        p.run_worker(Namespace(profile=str(profile_path), admission=str(worker_case.paths["admission"]), output=str(worker_case.root / "another")))
    profile = p.g.read(profile_path)
    ledger = p.DispatchLedger(worker_case.root / "worker-0-ledger.json", worker_case.catalog)
    with pytest.raises(ValueError, match="closed"):
        ledger.claim(profile["worker"], ["T2-5"], profile["execution_signature"])
    changed = deepcopy(profile); changed["worker_token"] = "another owner"
    with pytest.raises(ValueError, match="another owner"): ledger.register(changed)


def test_inbox_rejects_wrong_owner_and_finite_wait(worker_case):
    path = worker_case.seal(); profile = p.g.read(path)
    publish(worker_case, path, [[x["sample_id"] for x in worker_case.panels["20"]["items"]]])
    refs = p.WorkerReferences(profile, worker_case.catalog, worker_case.refs["20"], worker_case.refs["48"])
    batch_path = Path(profile["inbox"]) / "batch-000000.json"
    batch = p.g.read(batch_path); batch["worker_token"] = "wrong"; batch_path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match="ownership"): list(p.Inbox(profile, worker_case.catalog, refs))
    batch_path.unlink(); (Path(profile["inbox"]) / "DONE.json").unlink()
    with pytest.raises(ValueError, match="wait expired"): list(p.Inbox(profile, worker_case.catalog, refs))


def test_template_resolver_keeps_independent_48_ID_oracle(worker_case):
    path = worker_case.seal(n=48); profile = p.g.read(path)
    refs = p.WorkerReferences(profile, worker_case.catalog, worker_case.refs["20"], worker_case.refs["48"])
    actual = deepcopy(worker_case.rows["T3-10"])
    refs.verify(actual, "T3-10", False)
    refs.control["rows"]["T3-10"]["position_sha256"] = "wrong"
    with pytest.raises(ValueError, match="position_sha256"): refs.verify(actual, "T3-10", False)


def test_six_cold_comparison_uses_same_final_entry_and_GC_signature(worker_case, fake_runtime):
    paths = [fake_runtime.execute(i, 48)[0] for i in range(6)]
    assert all(p.g.read(path / "identity.json")["reference_sha256"] == p.g.file_sha256(worker_case.paths["ref48"])
               for path in paths)
    report = p.compare_workers(paths, "consistency")
    assert report["status"] == "MATCH" and report["six_observation_consistency_pass"]
    receipt = p.g.read(paths[-1] / "entry-receipt.json")
    receipt["gc"]["code_sha256"] = "another derived code"
    (paths[-1] / "entry-receipt.json").write_text(json.dumps(receipt))
    closure = p.g.read(paths[-1] / "entry-closure.json")
    closure["files"]["entry-receipt.json"] = p.g.file_sha256(paths[-1] / "entry-receipt.json")
    (paths[-1] / "entry-closure.json").write_text(json.dumps(closure))
    with pytest.raises(ValueError, match="signatures differ"): p.compare_workers(paths, "consistency")


def test_same_entry_output_difference_disqualifies(worker_case, fake_runtime, monkeypatch):
    first, _ = fake_runtime.execute(0)
    second, _ = fake_runtime.execute(1)
    original = p.g.compare_runs
    monkeypatch.setattr(p.g, "compare_runs", lambda *args: {**original(*args), "status": "PROFILE_OUTPUT_DIFFERENCE"})
    assert p.compare_workers([first, second])["status"] == "DIFFERENT"


def test_production_admission_requires_real_reports_not_catalog_or_worker_PASS(worker_case):
    profile = p.g.read(worker_case.seal())
    profile["mode"] = "production"
    with pytest.raises(KeyError): p.admission_gate(profile, p.g.read(worker_case.paths["admission"]))



def test_real_CPU_overlay_context_is_bound_into_derived_globals(worker_case, monkeypatch):
    profile = p.g.read(worker_case.seal())
    worker = p.Worker(profile, worker_case.catalog, worker_case.refs["20"], worker_case.refs["48"], {})
    class Video:
        def _preprocess(self, videos): return videos
    original_method = Video._preprocess
    source, receipt = cpu.processor_candidate(SDK)
    receipt = {**receipt, "file_sha256": p.g.file_sha256(__file__)}
    seen = []
    def prepared(session, item, roots, streamlined, deep):
        seen.append((deep, p.g.decode is cpu.direct_decode, Video._preprocess is original_method))
        return None, None, None, {}
    monkeypatch.setattr(cpu, "installed_processor_source", lambda: (source, receipt))
    monkeypatch.setattr(cpu, "_PREPARE", prepared)
    monkeypatch.setattr(p.g, "prepare", prepared)
    before = (p.g.prepare, p.g.decode, p.g.gc, p.g.run)
    overlay_profile = {"cpu_execution_overlay": {"processor_source": receipt, "gc_cadence": 8}}
    with pytest.raises(RuntimeError, match="cleanup fixture"):
        with cpu.installed_overlay(overlay_profile) as gc:
            runner, derived = worker.compile()
            gc.code, gc.line = runner.__code__, derived["gc_line"]
            assert runner.__globals__["gc"] is gc
            assert runner.__globals__["prepare"] is p.g.prepare
            assert runner.__globals__["CTX"] is p.g.CTX
            assert runner.__globals__["_install_kernel"] is p.g._install_kernel
            session = SimpleNamespace(processor=SimpleNamespace(video_processor=Video()))
            item = worker_case.catalog["items"]["T1-0"]
            runner.__globals__["_entry_prepare"](session, item, True, True)
            runner.__globals__["_entry_prepare"](session, item, True, False)
            assert gc.code is runner.__code__ and gc.line != 716
            raise RuntimeError("cleanup fixture")
    assert seen == [(True, False, True), (False, True, False)]
    assert (p.g.prepare, p.g.decode, p.g.gc, p.g.run) == before and Video._preprocess is original_method


def test_executable_code_fingerprint_ignores_only_marshal_alias_encoding():
    source = p.derive_runner()["source"]
    fingerprints = []
    for _ in range(12):
        env = {}
        exec(compile(source, "<gemma-persistent-entry-v1>", "exec"), env)
        fingerprints.append(p.code_fingerprint(env["run"].__code__))
    assert len(set(fingerprints)) == 1
    env = {}; exec(compile(source.replace("max_tokens=16384", "max_tokens=8192"), "<gemma-persistent-entry-v1>", "exec"), env)
    assert p.code_fingerprint(env["run"].__code__) != fingerprints[0]


def test_inbox_publication_is_atomic_and_never_replaces(tmp_path, monkeypatch):
    target = tmp_path / "batch.json"
    original = p.os.link
    def link(source, destination):
        assert p.g.read(source) == {"sample_ids": ["first"]} and not Path(destination).exists()
        original(source, destination)
    monkeypatch.setattr(p.os, "link", link)
    p.publish_message(target, {"sample_ids": ["first"]})
    assert p.g.read(target) == {"sample_ids": ["first"]}
    monkeypatch.setattr(p.os, "link", original)
    with pytest.raises(FileExistsError): p.publish_message(target, {"sample_ids": ["changed"]})
    assert p.g.read(target) == {"sample_ids": ["first"]} and list(tmp_path.glob("*.publish-*")) == []


def test_global_reconcile_requires_every_authority_ID(worker_case, monkeypatch):
    case = worker_case
    monkeypatch.setattr(p, "EXPECTED", case.catalog["counts"])
    ledger_path = case.root / "global-ledger.json"
    ledger = p.DispatchLedger(ledger_path, case.catalog)
    ids = list(case.catalog["items"])
    batch = ledger.claim("worker", ids, p.g.digest("one runtime"))
    all_proofs = [{"sample_id": sid, "batch_id": batch["batch_id"]} for sid in ids]
    report = {"profile": {"worker": "worker", "mode": "production", "execution_signature": "same"},
        "proofs": all_proofs[:-1], "uncertain": [], "complete": True}
    monkeypatch.setattr(p, "verify_worker", lambda *a, **k: report)
    outputs = case.root / "outputs.json"; p._write_new(outputs, ["verified CPU fixture"])
    args = Namespace(catalog=str(case.paths["catalog"]), ledger=str(ledger_path), outputs=str(outputs), output=str(case.root / "incomplete.json"))
    assert p.reconcile(args)["status"] == "INCOMPLETE"
    report["proofs"] = all_proofs; args.output = str(case.root / "complete.json")
    assert p.reconcile(args)["status"] == "COMPLETE"
    state = p.g.read(ledger_path)
    assert len(state["assignments"]) == len(ids) and {x["status"] for x in state["assignments"].values()} == {"COMPLETE"}


def test_reconcile_does_not_terminalize_an_unclosed_live_or_crashed_intent(worker_case, monkeypatch):
    ledger_path = worker_case.root / "ledger.json"
    ledger = p.DispatchLedger(ledger_path, worker_case.catalog)
    ledger.claim("worker", ["T1-0"], p.g.digest("one runtime"))
    before = ledger_path.read_bytes()
    monkeypatch.setattr(p, "verify_worker", lambda *a, **k: {"uncertain_requires_exit_confirmation": True})
    outputs = worker_case.root / "unclosed.json"; p._write_new(outputs, ["CPU unclosed fixture"])
    with pytest.raises(ValueError, match="needs controller confirmation"):
        p.reconcile(Namespace(catalog=str(worker_case.paths["catalog"]), ledger=str(ledger_path),
            outputs=str(outputs), output=str(worker_case.root / "must-not-exist.json")))
    assert ledger_path.read_bytes() == before
