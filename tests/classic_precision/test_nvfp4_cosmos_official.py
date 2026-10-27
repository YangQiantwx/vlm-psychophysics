"""Bounded CPU contracts; no processor, weights, CUDA, or network access."""
from contextlib import contextmanager
from copy import deepcopy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import nvfp4_cosmos_official as n
from classic_precision.storage import RunStore, PersistenceError


def fixture(tmp_path, count=3):
    items = []
    for i in range(count):
        sid = f"sample-{i}"
        row = {"sample_id": sid, "task": "T2", "truth": "LEFT", "prompt": "LEFT or RIGHT", "source_parameters": {"i": i}}
        ready = {"sample_id": sid, "video_relative": sid + ".mkv", "stream_sha256": "s"}
        items.append({"sample_id": sid, "row": row, "ready": ready, "source_stream_sha256": "s",
                      "source_family": "original", "row_sha256": n.digest(row), "ready_sha256": n.digest(ready), "selected_cell_id": "cell"})
    ids = [x["sample_id"] for x in items]
    lease = {"path": str(tmp_path / "existing.lock"), "st_dev": 1, "st_ino": 2}
    runtime = {"asset_roots": {"original": str(tmp_path)}, "physical_locks": {"0": lease}}
    plan = {"runtime": runtime, "panel": {"items": items}, "assignments": [ids, []], "prior_failed_nvfp4_attempts": {ids[0]: 1}}
    return plan, ids, lease


def store_at(tmp_path, ids):
    return RunStore(tmp_path, {"frozen": True}, n.MODEL_KEY, "worker0", ids, ids)


def raw():
    return {"task": "T2", "truth": "LEFT", "raw_answer": "LEFT", "output_token_ids": [22397, 151645],
            "finish_reason": "stop", "stop_reason": 151645, "seconds": 1.}


def mark_pass(store, sid, committed):
    store.event("TECHNICAL_PASS", sample_id=sid, attempt_id=committed["attempt_id"], raw_sha256=committed["raw_record_sha256"])


def fake_runtime(monkeypatch, lease, life):
    @contextmanager
    def owned(*args):
        life.append("acquire")
        try:
            yield lease
        finally:
            life.append("release")
    monkeypatch.setattr(n.native, "physical_lease", owned)
    return SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False))


def test_cpu_entry_has_no_heavy_import_or_model_side_effects():
    code = "from classic_precision import nvfp4_cosmos_official; import sys; assert not {'torch','vllm','transformers','av','kernels'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True)
    result = subprocess.run([sys.executable, "-m", "classic_precision.nvfp4_cosmos_official", "--help"], capture_output=True, text=True, check=True)
    assert all(command in result.stdout for command in ("seal", "check", "run", "compare"))


def test_official_engine_preserves_finite_single_gpu_input_and_precision_contract():
    s = n.engine_settings({"checkpoint": "/official", "original_checkpoint": "/original"})
    assert (s["model"], s["tokenizer"]) == ("/official", "/original")
    assert s["quantization"] == "modelopt_fp4" and s["linear_backend"] == "flashinfer_cutlass"
    assert s["dtype"] == s["kv_cache_dtype"] == "bfloat16"
    assert s["kv_cache_memory_bytes"] == 17 * 2**30 and s["max_model_len"] == 65536
    assert s["tensor_parallel_size"] == s["max_num_seqs"] == 1
    assert s["max_num_batched_tokens"] >= 35435 and s["disable_chunked_mm_input"] is True
    assert s["mm_processor_kwargs"] == {"do_resize": False, "do_sample_frames": False}
    assert s["video_pruning_rate"] == s["cpu_offload_gb"] == 0
    assert s["limit_mm_per_prompt"] == {"video": 1, "image": 0, "audio": 0}
    assert not {"speculative_config", "offload_group_size"} & s.keys()


def engine_receipt():
    return [{"model_class": "Qwen3VLForConditionalGeneration", "kernels": {"FlashInferCutlassNvFp4LinearKernel": 256},
             "excluded_bytes_by_dtype": {"torch.bfloat16": 4_000_000}, "attention_kv_dtypes": {"bfloat16": 64},
             "actual_cache_dtypes": ["torch.bfloat16"], "capacity": {"capacity_tokens": 65536},
             "torch_policy": deepcopy(n.native.NATIVE_TORCH_POLICY), "torch_threads": 6}]


def test_actual_runtime_receipt_accepts_only_observed_full_capacity():
    n.validate_engine_receipt(engine_receipt())


@pytest.mark.parametrize("field,value", [
    ("kernels", {"MarlinLinearKernel": 256}), ("kernels", {"FlashInferCutlassNvFp4LinearKernel": 255}),
    ("excluded_bytes_by_dtype", {"torch.float8_e4m3fn": 4_000_000}),
    ("attention_kv_dtypes", {"fp8_e4m3": 64}), ("actual_cache_dtypes", ["torch.uint8"]),
    ("capacity", {"capacity_tokens": 65535}), ("torch_threads", 1),
    ("model_class", "OtherModel"),
])
def test_actual_runtime_fallback_and_capacity_errors_fail_closed(field, value):
    receipt = engine_receipt(); receipt[0][field] = value
    with pytest.raises(ValueError): n.validate_engine_receipt(receipt)


def test_technical_pass_survives_reopen_and_resumes_only_unattempted_suffix(tmp_path):
    _, ids, _ = fixture(tmp_path)
    with store_at(tmp_path, ids) as store:
        store.begin(ids[0], "row"); committed = store.commit_output(ids[0], raw()); mark_pass(store, ids[0], committed)
    with store_at(tmp_path, ids) as store:
        assert n.select_pending(store, ids, 3) == ids[1:]


@pytest.mark.parametrize("fault", ["raw_without_pass", "wrong_hash", "interrupted", "out_of_order"])
def test_no_hidden_retry_after_any_uncertain_attempt(tmp_path, fault):
    _, ids, _ = fixture(tmp_path)
    with store_at(tmp_path, ids) as store:
        sid = ids[1] if fault == "out_of_order" else ids[0]
        store.begin(sid, "row")
        if fault != "interrupted":
            committed = store.commit_output(sid, raw())
            if fault == "wrong_hash": committed["raw_record_sha256"] = "wrong"
            if fault in ("wrong_hash", "out_of_order"): mark_pass(store, sid, committed)
    with store_at(tmp_path, ids) as store:
        with pytest.raises(ValueError): n.select_pending(store, ids, 3)


@pytest.mark.parametrize("limit", [0, -1, 4, True])
def test_segment_budget_is_explicit_positive_and_bounded(tmp_path, limit):
    _, ids, _ = fixture(tmp_path)
    with store_at(tmp_path, ids) as store:
        with pytest.raises(ValueError): n.select_pending(store, ids, limit)


def test_one_engine_first_two_deep_then_ordinary_and_close_before_release(monkeypatch, tmp_path):
    plan, ids, lease = fixture(tmp_path); life = []
    torch = fake_runtime(monkeypatch, lease, life)
    class Session:
        def __init__(self, runtime, sink):
            life.append("load"); self.sink = sink; self.context = {}
        def generate(self, row, ready, root, commit):
            life.append((row["sample_id"], self.context["deep"]))
            value = commit(raw()); mark_pass(self.sink, row["sample_id"], value)
        def close(self): life.append("close")
    with store_at(tmp_path, ids) as store:
        assert n.execute_segment(plan, "sealed", 0, 3, store, torch, "execution", Session) == ids
        records = list(store.outputs.values())
    assert life == ["acquire", "load", (ids[0], True), (ids[1], True), (ids[2], False), "close", "release"]
    assert [r["overall_nvfp4_attempt_number"] for r in records] == [2, 1, 1]
    assert all(r["plan_sha256"] == "sealed" and r["output_token_ids"] == [22397, 151645] for r in records)
    with store_at(tmp_path, ids) as store:
        with pytest.raises(ValueError, match="already complete"): n.select_pending(store, ids, 3)


@pytest.mark.parametrize("fault", ["after_raw", "missing_pass", "constructor", "bad_choice"])
def test_fault_stops_calls_keeps_durable_evidence_and_releases_lease(monkeypatch, tmp_path, fault):
    plan, ids, lease = fixture(tmp_path); life = []
    torch = fake_runtime(monkeypatch, lease, life)
    class Session:
        def __init__(self, runtime, sink):
            self.context = {}; self.sink = sink; life.append("load")
            if fault == "constructor": raise RuntimeError("load failed")
        def generate(self, row, ready, root, commit):
            life.append("generate")
            value = raw()
            if fault == "bad_choice": value.update(raw_answer="explanation", finish_reason="length")
            committed = commit(value)
            if fault == "after_raw": raise RuntimeError("post-input guard")
            if fault == "bad_choice": mark_pass(self.sink, row["sample_id"], committed)
        def close(self): life.append("close")
    with store_at(tmp_path, ids) as store:
        with pytest.raises((ValueError, RuntimeError)):
            n.execute_segment(plan, "sealed", 0, 3, store, torch, "execution", Session)
        assert len(store.outputs) == (0 if fault == "constructor" else 1)
    assert life[-1] == "release" and life.count("generate") <= 1
    if fault != "constructor": assert life[-2] == "close"
    events = [json.loads(s) for s in (tmp_path / n.MODEL_KEY / "events.jsonl").read_text().splitlines()]
    assert any(e["event"] == "LEASE_RELEASED" for e in events)
    assert not any(e["event"] == "FIRST_TWO_QUALIFIED" for e in events)
    if fault in ("after_raw", "missing_pass"):
        with store_at(tmp_path, ids) as store:
            with pytest.raises(ValueError): n.select_pending(store, ids, 3)


def test_wrong_physical_inode_prevents_model_load(monkeypatch, tmp_path):
    plan, ids, lease = fixture(tmp_path); life = []
    torch = fake_runtime(monkeypatch, {**lease, "st_ino": 3}, life)
    def forbidden(*args): raise AssertionError("must not load")
    with store_at(tmp_path, ids) as store:
        with pytest.raises(ValueError, match="lease identity"):
            n.execute_segment(plan, "sealed", 0, 3, store, torch, "execution", forbidden)
    assert life == ["acquire", "release"]


def paired_records():
    records, old = [], {}
    for i in range(432):
        sid = str(i)
        old[sid] = {"truth": "LEFT", "selected_source": {"row_sha256": "r", "ready_sha256": "q", "source_video_sha256": "v", "source_stream_sha256": "s"},
            "raw_answer": "LEFT", "output_token_ids": [22397, 151645], "finish_reason": "stop", "correct": True,
            "historical_host": "cirplab-fourier" if i < 272 else "cirplab-radon", "seconds": 10., "actual_GPU_content_hash_available": False}
        records.append({**raw(), "sample_id": sid, "selected_cell_id": "cell", "worker": 0, "deep": i < 2,
            "catalog_row_sha256": "r", "catalog_ready_sha256": "q", "input_receipt": {"source_video_sha256": "v", "source_stream_sha256": "s"},
            "actual_prompt_token_ids_sha256": "prompt", "engine_generate_seconds": .9, "full_request_seconds": 2.})
    return {"items": old, "timing_definition": {"metric": "post_prepare"}}, records


def test_precision_changed_answers_and_tokens_remain_analysis_results():
    oracle, records = paired_records()
    records[7].update(raw_answer="RIGHT", output_token_ids=[11572, 151645])
    result = n.compare_records(oracle, records)
    assert result["all"]["N"] == 432 and result["all"]["answers_changed"] == 1
    assert result["all"]["tokens_changed"] == 1 and result["all"]["nvfp4_correct"] == 431
    assert result["rows"][7]["bf16_tokens"] == [22397, 151645]
    assert result["rows"][7]["nvfp4_tokens"] == [11572, 151645]
    assert result["timing_scope_mismatch"] is True
    assert not any(r["historical_GPU_content_hash_available"] for r in result["rows"])


@pytest.mark.parametrize("fault", ["duplicate", "missing", "source", "historical_prompt"])
def test_comparison_requires_exact_scientific_denominator_and_input_identity(fault):
    oracle, records = paired_records()
    if fault == "duplicate": records[-1] = records[0]
    elif fault == "missing": records.pop()
    elif fault == "source": records[0]["input_receipt"]["source_stream_sha256"] = "changed"
    else: oracle["items"]["0"]["actual_prompt_token_ids_sha256"] = "different"
    with pytest.raises(ValueError): n.compare_records(oracle, records)
