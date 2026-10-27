"""CPU seams only: no real processor, model weights, CUDA or remote execution."""
from argparse import Namespace
import ast
from contextlib import ExitStack
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace as NS, ModuleType

import numpy as np
import pytest

from classic_precision import gemma_parallel as p


def intervals(offset=0):
    return [[offset + i * 180 + 1, offset + i * 180 + 176] for i in range(180)]


def sampling():
    return NS(n=1, temperature=0., top_p=1., seed=0, max_tokens=16384, repetition_penalty=1.,
              structured_outputs=NS(choice=["LEFT", "RIGHT"], grammar=None, _backend=None, _backend_was_auto=False,
                  json=None, regex=None, json_object=None, structural_tag=None, disable_any_whitespace=False,
                  disable_additional_properties=False, whitespace_pattern=None))


class Tensor:
    """Small CPU ownership object; it makes no claim to be a GPU tensor."""
    def __init__(self, token):
        self.token, self.shape, self.dtype, self.device = token, (180, 2520, 768), "BF16", "cpu"
    def stride(self): return (2520 * 768, 768, 1)
    def data_ptr(self): return id(self)


def case():
    items, refs, baseline = [], {}, {}
    for index, task in enumerate(["T1"] * 2 + ["T2"] * 4 + ["T3"] * 2):
        sid = f"{task}-{index}"
        row = {"sample_id": sid, "prompt": task + " exact prompt"}
        ready = {"sample_id": sid, "video_relative": sid + ".mkv", "video_sha256": p.g.digest(sid)}
        items.append({"sample_id": sid, "task": task, "row": row, "ready": ready,
            "row_sha256": p.g.digest(row), "ready_sha256": p.g.digest(ready)})
        ids = [1, index, 3]
        refs[sid] = {"input_token_ids": ids, "input_token_ids_sha256": p.g.digest(ids), "prompt_sha256": p.g.digest(row["prompt"]),
            "source_stream_sha256": p.g.digest([sid, "stream"]), "position_sha256": p.g.digest("positions"),
            "position_shape": [180, 2520, 2], "HF_control_tensors": {"mask": "same"},
            **{key: p.g.digest([sid, key]) for key in ("input_sha256", "projection_sha256", "processor_input_FP32_sha256", "pixel_FP32_sha256")}}
        baseline[sid] = {"raw_answer": "LEFT", "output_token_ids": [8, 9], "finish_reason": "stop", "stop_reason": None,
            "native_ranges": intervals(index * 3)}
    profile = {"items": items, "orders": p.schedule(items), "baseline": baseline,
        "runtime": {"asset_roots": {task: "/cpu-fixture/" + task for task in ("T1", "T2", "T3")}}}
    return profile, {"rows": refs}


def pair_entries(profile, reference):
    entries, expected = [], {}
    for i, item in enumerate(profile["items"][:2]):
        sid = item["sample_id"]
        entry = {"item": item, "deep": True, "payload": {key: Tensor(sid + key) for key in ("pixel_values_videos", "pixel_position_ids_videos")}}
        entries.append(entry)
        expected[f"internal-{i}"] = {"sample_id": sid, "external_id": str(i), "deep": True,
            "input_token_ids": reference["rows"][sid]["input_token_ids"], "native_ranges": profile["baseline"][sid]["native_ranges"],
            "mm_identifier": "mm-" + sid}
    return entries, expected


def observations(expected, reference):
    result = []
    for rank in (0, 1):
        video = {}
        for rid, binding in expected.items():
            ref = reference["rows"][binding["sample_id"]]
            video[rid] = {"request_id": rid, "sample_id": binding["sample_id"], "frames": 180,
                "shape": [180, 2520, 768], "dtype": "torch.bfloat16", "embedding_shape": [31680, 768],
                "position_sha256": ref["position_sha256"], "GPU_prompt_token_ids_sha256": ref["input_token_ids_sha256"],
                "input_sha256": ref["input_sha256"], "projection_sha256": ref["projection_sha256"]}
        seq = {rid: {"native_ranges": row["native_ranges"]} for rid, row in expected.items()}
        result.append({"rank": rank, "device": f"GPU-{rank}", "video": video,
            "sequences": [{"sequences": seq}], "positions": [{"request_id": rid} for rid in expected]})
    return result


def test_only_declared_placement_and_scheduler_settings_change():
    before, after = p.g.engine_settings("checkpoint"), p.engine_settings("checkpoint")
    assert {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)} == {
        "tensor_parallel_size", "distributed_executor_backend", "max_num_seqs", "max_num_batched_tokens", "cpu_offload_gb"}
    assert after["max_model_len"] == 65536 and after["kv_cache_memory_bytes"] == 32 * 2**30
    assert after["dtype"] == after["kv_cache_dtype"] == "bfloat16" and after["cpu_offload_params"] == ["gate_up_proj"]
    source = p.placement_source()
    assert '>= 2 * 65536' in source and 'offload_bytes"] == 0' in source


def test_fixed_selection_rotates_companion_and_position():
    profile, _ = case(); selected = p.selected_items({"items": profile["items"]})
    orders = p.schedule(selected)
    assert orders[1] == orders[0][1:] + orders[0][:1]
    pairs = [set(frozenset(order[i:i+2]) for i in range(0, 8, 2)) for order in orders]
    assert not pairs[0] & pairs[1]
    assert all(orders[0].index(sid) % 2 != orders[1].index(sid) % 2 for sid in orders[0])


def test_projection_chunks_cross_video_boundary_without_hash_mixing():
    data = np.arange(360 * 6, dtype=np.uint16).reshape(360, 2, 3)
    hashes, cursor = [hashlib.sha256(), hashlib.sha256()], 0
    for length in (100, 130, 130):
        chunk = data[cursor:cursor+length]
        for index, lo, hi in p.frame_intersections([180, 180], cursor, length):
            hashes[index].update(memoryview(chunk[lo:hi]).cast("B"))
        cursor += length
    assert [h.hexdigest() for h in hashes] == [hashlib.sha256(memoryview(data[i:i+180]).cast("B")).hexdigest() for i in (0, 180)]


@pytest.mark.parametrize("counts,start,length", [([360], 0, 360), ([180, 179], 0, 359), ([180, 180], 359, 2), ([180], 0, 0)])
def test_incorrect_frame_or_projection_bounds_are_rejected(counts, start, length):
    with pytest.raises(ValueError): p.frame_intersections(counts, start, length)


def test_native_encoder_order_and_reordered_sequences_use_real_request_IDs():
    profile, ref = case(); _, expected = pair_entries(profile, ref)
    ledger = p.EncoderLedger(); ledger.arm(expected)
    ids = list(reversed(expected))
    actual = {rid: NS(prompt_token_ids=expected[rid]["input_token_ids"]) for rid in ids}
    ledger.scheduled([expected[rid]["mm_identifier"] for rid in ids], ["video", "video"], [(rid, object()) for rid in ids], actual)
    assert ledger.consume([180], 180) == ids[:1]
    assert ledger.consume([180], 180) == ids[1:] and not ledger.pending
    records = p.sequence_records(ids, [0, 2, 5], [4, 6], {i: expected[rid]["native_ranges"] for i, rid in enumerate(ids)}, expected)
    assert records[ids[0]]["absolute_position_start"] == 2
    receipts = p.validate_text_positions([2, 3, 3, 4, 5], records)
    assert [r["request_id"] for r in receipts] == ids
    with pytest.raises(ValueError, match="positions cross"): p.validate_text_positions([3, 4, 2, 3, 4], records)


@pytest.mark.parametrize("fault", ["duplicate", "foreign", "mixed_tokens", "wrong_mm", "missing_frame"])
def test_encoder_context_faults_are_rejected(fault):
    profile, ref = case(); _, expected = pair_entries(profile, ref)
    ids = list(expected); hashes = [expected[rid]["mm_identifier"] for rid in ids]
    actual = {rid: NS(prompt_token_ids=list(expected[rid]["input_token_ids"])) for rid in ids}
    ledger = p.EncoderLedger(); ledger.arm(expected)
    if fault == "duplicate": ids[1] = ids[0]
    elif fault == "foreign": ids[1] = "foreign"
    elif fault == "mixed_tokens": actual[ids[0]].prompt_token_ids = actual[ids[1]].prompt_token_ids
    elif fault == "wrong_mm": hashes.reverse()
    with pytest.raises(ValueError):
        ledger.scheduled(hashes, ["video", "video"], [(rid, object()) for rid in ids], actual)
        ledger.consume([180, 180], 359)


@pytest.mark.parametrize("fault", ["swapped_masks", "overlap", "missing_mask", "foreign"])
def test_actual_sequence_mask_and_context_faults_are_rejected(fault):
    profile, ref = case(); _, expected = pair_entries(profile, ref)
    ids, starts = list(expected), [0, 2, 5]
    ranges = {i: expected[rid]["native_ranges"] for i, rid in enumerate(ids)}
    if fault == "swapped_masks": ranges[0], ranges[1] = ranges[1], ranges[0]
    elif fault == "overlap": starts = [0, 4, 3]
    elif fault == "missing_mask": del ranges[1]
    else: ids[0] = "foreign"
    with pytest.raises(ValueError): p.sequence_records(ids, starts, [4, 6], ranges, expected)


def test_registry_does_not_accept_copies_aliases_or_changed_metadata():
    registry = p.PreparedRegistry()
    a = np.broadcast_to(np.zeros((1, 1, 1, 3), np.float32), (180, 528, 768, 3))
    b = np.broadcast_to(np.ones((1, 1, 1, 3), np.float32), a.shape)
    registry.entries = [{"frames": a, "payload": {"id": "A"}, "adapter_calls": 0},
                        {"frames": b, "payload": {"id": "B"}, "adapter_calls": 0}]
    metadata = {"frames_indices": list(range(180)), "fps": 60., "total_num_frames": 180, "do_sample_frames": False}
    assert registry.match(b, metadata) == {"id": "B"}
    with pytest.raises(ValueError): registry.match(b[:179], metadata)
    with pytest.raises(ValueError): registry.match(a, {**metadata, "fps": 30.})
    registry.entries.append(registry.entries[0])
    with pytest.raises(ValueError): registry.match(a, metadata)


def request_for(entry, index, reference):
    sid = entry["item"]["sample_id"]
    return NS(request_id=f"internal-{index}", external_req_id=str(index), prompt_token_ids=reference["rows"][sid]["input_token_ids"],
        sampling_params=sampling(), mm_features=[NS(modality="video", identifier="mm-" + sid,
            data={key: NS(data=value) for key, value in entry["payload"].items()})])


def test_capture_binds_actual_external_internal_IDs_and_restores_on_exception():
    profile, ref = case(); entries, _ = pair_entries(profile, ref)
    submitted = []; core = NS(add_request=submitted.append); engine = NS(llm_engine=NS(engine_core=core))
    native, expected = core.add_request, {}
    with pytest.raises(RuntimeError):
        with p.capture_requests(engine, entries, expected, ref, profile) as captured:
            core.add_request(request_for(entries[0], 0, ref)); core.add_request(request_for(entries[1], 1, ref))
            assert captured == ["internal-0", "internal-1"]
            assert expected["internal-1"]["sample_id"] == entries[1]["item"]["sample_id"]
            raise RuntimeError("seam")
    assert core.add_request is native and len(submitted) == 2


@pytest.mark.parametrize("fault", ["tensor", "prompt", "sampling", "merged_video"])
def test_native_request_capture_rejects_changed_input_or_effective_generation(fault):
    profile, ref = case(); entries, _ = pair_entries(profile, ref)
    core = NS(add_request=lambda request: None); engine = NS(llm_engine=NS(engine_core=core))
    request = request_for(entries[0], 0, ref)
    if fault == "tensor": request.mm_features[0].data["pixel_values_videos"].data = entries[1]["payload"]["pixel_values_videos"]
    elif fault == "prompt": request.prompt_token_ids = [999]
    elif fault == "sampling": request.sampling_params.max_tokens = 8192
    else: request.mm_features *= 2
    with p.capture_requests(engine, entries, {}, ref, profile):
        with pytest.raises(ValueError): core.add_request(request)


def test_returned_output_order_is_not_used_as_identity():
    profile, ref = case(); _, expected = pair_entries(profile, ref)
    outputs = [NS(request_id=b["external_id"], prompt_token_ids=b["input_token_ids"], outputs=[object()]) for b in expected.values()]
    mapped = p.map_outputs(outputs[::-1], expected)
    assert mapped["internal-0"] is outputs[0]
    with pytest.raises(ValueError): p.map_outputs([outputs[0], outputs[0]], expected)
    outputs[1].prompt_token_ids = outputs[0].prompt_token_ids
    with pytest.raises(ValueError): p.map_outputs(outputs, expected)


@pytest.mark.parametrize("fault", ["none", "total360", "swap_pixels", "swap_projection", "wrong_position", "missing_rank", "missing_mask", "wrong_ID"])
def test_per_ID_two_rank_tensor_proofs_are_not_batch_totals(fault):
    profile, ref = case(); _, expected = pair_entries(profile, ref); obs = observations(expected, ref)
    a, b = list(expected)
    if fault == "none": p.validate_observations(obs, expected, ref); return
    if fault == "total360": obs[0]["video"][a]["frames"] = 360
    elif fault == "swap_pixels": obs[0]["video"][a]["input_sha256"] = obs[0]["video"][b]["input_sha256"]
    elif fault == "swap_projection": obs[1]["video"][a]["projection_sha256"] = obs[1]["video"][b]["projection_sha256"]
    elif fault == "wrong_position": obs[0]["video"][a]["position_sha256"] = "wrong"
    elif fault == "missing_rank": obs.pop()
    elif fault == "missing_mask": obs[0]["sequences"] = []
    else: obs[0]["video"][a]["sample_id"] = expected[b]["sample_id"]
    with pytest.raises(ValueError): p.validate_observations(obs, expected, ref)


@pytest.fixture
def batch_fixture(tmp_path, monkeypatch):
    profile, reference = case(); history, deep_calls = [], []
    registry = p.PreparedRegistry()
    def prepare(session, item, roots, streamlined, deep):
        sid = item["sample_id"]; deep_calls.append((sid, deep))
        return sid, {key: Tensor(sid + key) for key in ("pixel_values_videos", "pixel_position_ids_videos")}, item["row"]["prompt"], deepcopy(reference["rows"][sid])
    monkeypatch.setattr(p.g, "prepare", prepare)
    fake_vllm = ModuleType("vllm"); fake_vllm.SamplingParams = lambda **kwargs: NS(**kwargs)
    fake_params = ModuleType("vllm.sampling_params"); fake_params.StructuredOutputsParams = lambda **kwargs: NS(**kwargs)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm); monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_params)
    class Engine:
        def __init__(self):
            self.counter, self.expected, self.pause, self.mutate = 0, None, False, False
            self.llm_engine = NS(engine_core=NS(add_request=lambda req: history.append(("add", req.request_id))))
        def sleep(self, level, mode): assert level == 0 and mode == "keep"; self.pause = True; history.append(("pause",))
        def enqueue(self, requests, params, use_tqdm):
            assert self.pause and len(requests) == 2
            ids = []
            for entry in registry.entries:
                entry["adapter_calls"] += 1
                req = request_for(entry, self.counter, reference); self.counter += 1
                self.llm_engine.engine_core.add_request(req); ids.append(req.request_id)
            return ids
        def collective_rpc(self, fn, args=()):
            if fn is p._arm:
                assert self.pause; self.expected = args[0]; history.append(("arm",)); return [{"rank": 0}, {"rank": 1}]
            assert fn is p._observe
            return observations(self.expected, reference)
        def wake_up(self, tags):
            assert tags == ["scheduling"] and self.pause and self.expected
            intents = [x for x in p.read_lines(tmp_path / "events.jsonl") if x["event"] == "GENERATE_INTENT"]
            assert len(intents) == self.counter
            self.pause = False; history.append(("wake",))
        def wait_for_completion(self, use_tqdm):
            history.append(("wait",))
            return [NS(request_id=b["external_id"], prompt_token_ids=b["input_token_ids"],
                outputs=[NS(text="LEFT", token_ids=[8, 999] if self.mutate else [8, 9], finish_reason="stop", stop_reason=None)])
                for b in reversed(list(self.expected.values()))]
    state = p.c.GCCadence(8, p.run_batches.__code__, p._gc_line())
    monkeypatch.setattr(p.g, "gc", state)
    return NS(profile=profile, ref=reference, root=tmp_path, engine=Engine(), registry=registry, state=state,
              history=history, deep_calls=deep_calls)


def test_two_pass_real_batch_loop_has_16_independent_outputs_and_durable_proofs(batch_fixture):
    f = batch_fixture
    result = p.run_batches(f.engine, object(), f.registry, f.profile, f.ref, f.root)
    assert result["calls"] == 16 and len(result["batches"]) == 8
    assert f.engine.counter == 16 and [d for _, d in f.deep_calls] == [True, True] + [False] * 14
    raw = p.read_lines(f.root / "raw.jsonl")
    assert [[r["sample_id"] for r in raw if r["pass"] == i] for i in (0, 1)] == f.profile["orders"]
    assert len({r["attempt_id"] for r in raw}) == 16
    assert len(p.read_lines(f.root / "calls.jsonl")) == 8
    assert len(f.state.rows) == 16 and f.state.maximum_pending == 8
    assert [x[0] for x in f.history[:6]] == ["pause", "add", "add", "arm", "wake", "wait"]
    f.state.finish(); assert f.state.final["status"] == "COMPLETE"


def test_identical_answer_with_changed_full_tokens_stops_and_preserves_failure_proof(batch_fixture):
    f = batch_fixture; f.engine.mutate = True
    with pytest.raises(p.OutputDifference): p.run_batches(f.engine, object(), f.registry, f.profile, f.ref, f.root)
    assert f.engine.counter == 2 and len(p.read_lines(f.root / "raw.jsonl")) == 2
    p.close_failed_attempt(f.root, "DIFFERENT")
    proof = p.read_lines(f.root / "calls.jsonl")[0]
    assert proof["status"] == "DIFFERENT" and proof["generation_attempts"] == 2 and proof["raw"]["end"] > 0


def test_zero_output_failure_keeps_two_intents_without_invented_elapsed(tmp_path):
    for sid in ("a", "b"):
        p.g.append(tmp_path / "events.jsonl", {"event": "GENERATE_INTENT", "sample_id": sid, "attempt_id": sid + "-attempt"})
    p.close_failed_attempt(tmp_path, "STOPPED")
    proof = p.read_lines(tmp_path / "calls.jsonl")[0]
    assert proof["status"] == "UNCERTAIN" and proof["generation_attempts"] == 2
    assert proof["raw"]["start"] == proof["raw"]["end"] == 0 and "elapsed" not in proof


def test_unassignable_native_outputs_are_preserved_without_invented_sample_ID(batch_fixture):
    f = batch_fixture; original = f.engine.wait_for_completion
    def wrong_id(use_tqdm):
        outputs = original(use_tqdm); outputs[0].request_id = "unknown-native-id"
        return outputs
    f.engine.wait_for_completion = wrong_id
    with pytest.raises(ValueError, match="another/duplicate context"):
        p.run_batches(f.engine, object(), f.registry, f.profile, f.ref, f.root)
    p.close_failed_attempt(f.root, "STOPPED")
    saved = [x for x in p.read_lines(f.root / "events.jsonl") if x["event"] == "UNATTRIBUTED_NATIVE_OUTPUTS"][0]
    assert len(saved["outputs"]) == 2 and saved["outputs"][0]["request_id"] == "unknown-native-id"
    assert saved["outputs"][0]["choices"][0]["token_ids"] == [8, 9]
    assert all("sample_id" not in row for row in saved["outputs"])
    assert not (f.root / "raw.jsonl").exists() and f.engine.counter == 2


def test_both_existing_native_inodes_stay_locked_until_all_rank_shutdown(tmp_path, monkeypatch):
    from classic_precision import inference
    roots = tmp_path / "t1"; lock_root = roots / "evaluation/outputs"; lock_root.mkdir(parents=True)
    runtime = {"lease_root": str(lock_root), "asset_roots": {"T1": str(roots)}}
    class Device:
        def __enter__(self): return self
        def __exit__(self, *args): pass
    torch = NS(cuda=NS(device_count=lambda: 2, get_device_properties=lambda i: NS(uuid=f"GPU-{i}"), device=lambda i: Device()))
    monkeypatch.setattr(inference, "resource_snapshot", lambda *a: {"admitted": True})
    paths = []
    for device in ("GPU-0", "GPU-1"):
        path = lock_root / (".gpu-" + p.g.digest({"host": socket.gethostname(), "device": device}) + ".lock")
        path.write_bytes(b"existing"); paths.append(path)
    inodes = [path.stat().st_ino for path in paths]
    alive = [True, True]
    monkeypatch.setattr(p.g, "owned_worker_alive", lambda identity: alive[identity["rank"]])
    def shutdown():
        for path in paths:
            with path.open("r+b") as stream:
                with pytest.raises(BlockingIOError): fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        alive[:] = [False, False]
    engine = NS(llm_engine=NS(engine_core=NS(shutdown=shutdown)))
    with p.physical_pair(runtime, torch, {"min_free_gpu_gib_per_device": 90, "min_available_ram_gib": 64}) as leases:
        with ExitStack() as cleanup:
            cleanup.callback(p.g.shutdown_owned_engine, engine, [{"worker": {"rank": i}} for i in (0, 1)], tmp_path)
            assert {x["st_ino"] for x in leases} == set(inodes)
    for path in paths:
        with path.open("r+b") as stream: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert [path.stat().st_ino for path in paths] == inodes


def test_source_drift_is_rejected_before_runtime(monkeypatch):
    original = p.g.file_sha256
    monkeypatch.setattr(p.g, "file_sha256", lambda path: "wrong" if str(path) == p.c.__file__ else original(path))
    with pytest.raises(ValueError, match="source contract"): p.validate_profile({"schema": p.SCHEMA, "execution": {}})


def test_import_and_cli_do_not_import_GPU_libraries():
    import subprocess
    result = subprocess.run([sys.executable, "-B", "-c",
        "import sys; from classic_precision import gemma_parallel; assert not any(x in sys.modules for x in ('torch','vllm','transformers')); gemma_parallel.main()", "--help"],
        capture_output=True, text=True)
    assert result.returncode == 0 and "seal-profile" in result.stdout and "compare" in result.stdout


def closed_batch_fixture(f, monkeypatch):
    result = p.run_batches(f.engine, object(), f.registry, f.profile, f.ref, f.root)
    result["status"] = "COMPLETE"; f.state.finish()
    f.profile["qualified_C3"] = {"kernel": {"range_lookup": "binary", "prefill_block_m": 32, "expected_native_sha256": p.g.NATIVE_SHA}}
    f.profile["baseline_torch_policy"] = {"unchanged": True}
    reference = f.root / "reference.json"; p.g.seal(reference, f.ref)
    monkeypatch.setattr(p, "REFERENCE_SHA", p.g.file_sha256(reference))
    monkeypatch.setattr(p, "validate_profile", lambda value, *args: value)
    ranks = []
    for i in range(2):
        ranks.append({"identity": {"rank": i, "device": f"GPU-{i}", "worker": {"rank": i}}, "actual_engine": p.ACTUAL_ENGINE,
            "kernel": {"configuration": f.profile["qualified_C3"]["kernel"], "observed_native_sha256": p.g.NATIVE_SHA,
                "derived_helpers_sha256": "32a56e3cb9501974cf648a3524691e84bd4a8c6dd7ead6c2ecd4b683c7b1ac4f",
                "derived_launcher_sha256": "a9c0fd9df26aae3919a024de045a6bb59e67011c297f2de87f584acf2bfc855e"},
            "placement": {"kv": {"capacity_tokens": 132170}, "offload_bytes": 0, "offloaded_names": [],
                          "torch_policy": f.profile["baseline_torch_policy"]},
            "observer": {"source_sha256": p.g.file_sha256(p.__file__), "extra_sources": p.EXTRA_SOURCES,
                "integration": p.observer_interface(), "actual_types": p.V2_TYPES}})
    files = {"identity.json": {"profile": f.profile, "profile_sha256": p.g.digest(f.profile),
                "reference_sha256": p.REFERENCE_SHA, "leases": [{"device": f"GPU-{i}"} for i in range(2)]},
        "result.json": result, "load.json": {"ranks": ranks}, "startup-children.json": {"status": "COMPLETE"},
        "exit.json": {"status": "COMPLETE", "owned_workers": [{"rank": 0}, {"rank": 1}], "owned_engine_workers_exited": True,
                      "lease_released_after_engine_exit": True},
        "cpu-copy.json": {"status": "COMPLETE", "bound_to_run_batches": True, "gc_line": p._gc_line(), "cadence": 8,
                          "maximum_pending_ordinary": f.state.maximum_pending, "final": f.state.final, "rows": f.state.rows}}
    for name, value in files.items(): p.g.seal(f.root / name, value)
    def close():
        p.g.atomic_json(f.root / "closure.json", {"status": "COMPLETE", "files": {
            path.name: p.g.file_sha256(path) for path in f.root.iterdir() if path.is_file() and path.name != "closure.json"}})
    close()
    return reference, close


def test_comparer_reads_real_byte_proofs_rank_controls_and_two_output_passes(batch_fixture, monkeypatch):
    reference, _ = closed_batch_fixture(batch_fixture, monkeypatch)
    report = p.compare(batch_fixture.root, reference)
    assert report["status"] == "MATCH" and report["calls"] == 16 and report["unique_IDs"] == 8
    assert not report["production_qualified"]


def test_comparer_rejects_changed_bytes_even_when_outer_closure_is_rehashed(batch_fixture, monkeypatch):
    reference, close = closed_batch_fixture(batch_fixture, monkeypatch)
    raw = batch_fixture.root / "raw.jsonl"
    raw.write_bytes(raw.read_bytes().replace(b'"LEFT"', b'"RIGH"', 1)); close()
    with pytest.raises(ValueError, match="byte proof"): p.compare(batch_fixture.root, reference)


def test_comparer_rejects_inadequate_actual_KV_capacity(batch_fixture, monkeypatch):
    reference, close = closed_batch_fixture(batch_fixture, monkeypatch)
    path = batch_fixture.root / "load.json"; load = p.g.read(path)
    load["ranks"][1]["placement"]["kv"]["capacity_tokens"] = 66085
    p.g.atomic_json(path, load); close()
    with pytest.raises(ValueError, match="placement"): p.compare(batch_fixture.root, reference)


def test_startup_failure_waits_only_recorded_owned_identities_without_signals(tmp_path, monkeypatch):
    watcher = p.StartupChildren(tmp_path)
    identity = {"host": "CPU fixture", "pid": 123, "startticks": "456", "boot_id": "boot"}
    watcher.owned[(123, "456")] = identity
    observed = []
    def alive(value):
        assert value == identity; observed.append(value)
        return len(observed) == 1
    monkeypatch.setattr(p.g, "owned_worker_alive", alive)
    monkeypatch.setattr(p.time, "sleep", lambda seconds: None)
    watcher.wait()
    receipt = p.g.read(tmp_path / "startup-children.json")
    assert receipt["status"] == "COMPLETE" and receipt["owned"] == [identity] and len(observed) == 2


def profile_fixture():
    partial, _ = case(); items = deepcopy(partial["items"])
    for task in ("T1", "T2", "T3"):
        original = next(x for x in items if x["task"] == task)
        for i in range(4):
            item = deepcopy(original); sid = task + "-extra-" + str(i)
            item["sample_id"] = item["row"]["sample_id"] = item["ready"]["sample_id"] = sid
            item["row_sha256"], item["ready_sha256"] = p.g.digest(item["row"]), p.g.digest(item["ready"])
            items.append(item)
    source = "class Gemma4VideoProcessor:\n    def _preprocess(self, videos):\n        pixel_values = torch.stack(pixel_values, dim=0)\n        position_ids = torch.stack(position_ids, dim=0)\n        return pixel_values, position_ids\n"
    _, processor = p.c.processor_candidate(source)
    runtime = {"checkpoint": "/models/" + p.g.REVISION, "environment": {"CUDA_VISIBLE_DEVICES": "0"}}
    q = {"schema": p.g.SCHEMA, "name": "historical CPU fixture", "variant": "kernel", "runtime": runtime,
        "panel": {"items": items}, "maximum_model_calls": 20, "first_deep_calls": 2,
        "engine_settings": p.g.engine_settings(runtime["checkpoint"]), "kernel": {
            "range_lookup": "binary", "prefill_block_m": 32, "expected_native_sha256": p.g.NATIVE_SHA},
        "implementation": {"gemma_speed": p.c.m.BASE_SHA, "inference": p.c.m.INFERENCE_SHA},
        "runtime_sources": {p.c.SDK_FILE: processor["file_sha256"]}}
    q["cpu_execution_overlay"] = p.c.overlay_record(deepcopy(q), "baseline", 8, processor)
    both = deepcopy(runtime); both["environment"]["CUDA_VISIBLE_DEVICES"] = "0,1"
    profile = {"schema": p.SCHEMA, "execution": p.contract(), "qualified_C3": q, "items": p.selected_items(q["panel"]),
        "orders": p.schedule(p.selected_items(q["panel"])), "runtime": both, "devices": "0,1",
        "communication": p.communication_settings(), "kv_cache_gib": 32,
        "engine_settings": p.engine_settings(runtime["checkpoint"]), "extra_sources": p.EXTRA_SOURCES,
        "reference_sha256": p.REFERENCE_SHA, "resource_budget": deepcopy(p.RESOURCE_BUDGET)}
    return profile, source, processor


def test_new_profile_explicitly_binds_sources_two_rank_policy_and_level_zero():
    profile, _, _ = profile_fixture()
    assert p.validate_profile(profile) is profile
    assert profile["execution"]["scheduling"] == {"sleep_level": 0, "sleep_mode": "keep", "wake_tags": ["scheduling"],
        "weights_and_KV_remain_resident": True, "unsupported_installed_source": "reject before LLM construction"}
    drift = deepcopy(profile); drift["execution"]["placement_source_sha256"] = "wrong"
    with pytest.raises(ValueError, match="source contract"): p.validate_profile(drift)
    drift = deepcopy(profile); drift["engine_settings"]["kv_cache_memory_bytes"] -= 1
    with pytest.raises(ValueError, match="settings"): p.validate_profile(drift)


def test_installed_scheduler_source_drift_refuses_before_model_loading(monkeypatch):
    profile, source, processor = profile_fixture()
    monkeypatch.setattr(p.c, "installed_processor_source", lambda: (source, processor))
    monkeypatch.setattr(p, "extra_sources", lambda: {**p.EXTRA_SOURCES, "vllm/v1/engine/core.py": "drift"})
    with pytest.raises(ValueError, match="Installed SDK"): p.validate_profile(profile, True)
    assert "torch" not in sys.modules and "vllm" not in sys.modules


@pytest.mark.parametrize("disabled", [False, True])
def test_communication_sealing_defaults_and_explicit_disable_are_process_local(disabled):
    profile, _, _ = profile_fixture()
    original = deepcopy(profile["qualified_C3"]["runtime"])
    profile["communication"] = p.communication_settings(disabled)
    profile["runtime"] = p.parallel_runtime(original, "0,1", profile["communication"])
    assert p.validate_profile(profile) is profile
    expected = {"NCCL_P2P_DISABLE": "1", "NCCL_DEBUG": "INFO"} if disabled else {}
    assert profile["communication"]["environment"] == expected
    assert {key: value for key, value in profile["runtime"]["environment"].items() if key.startswith("NCCL_")} == expected
    assert profile["qualified_C3"]["runtime"] == original
    assert profile["engine_settings"] == p.engine_settings(original["checkpoint"])


@pytest.mark.parametrize("fault", ["wrong_disable", "wrong_debug", "extra_override", "flag_mismatch", "non_boolean"])
def test_sealed_communication_tampering_is_rejected(fault):
    profile, _, _ = profile_fixture()
    profile["communication"] = p.communication_settings(True)
    profile["runtime"] = p.parallel_runtime(profile["qualified_C3"]["runtime"], "0,1", profile["communication"])
    if fault == "wrong_disable": profile["runtime"]["environment"]["NCCL_P2P_DISABLE"] = "0"
    elif fault == "wrong_debug": profile["runtime"]["environment"]["NCCL_DEBUG"] = "WARN"
    elif fault == "extra_override": profile["runtime"]["environment"]["NCCL_SHM_DISABLE"] = "1"
    elif fault == "flag_mismatch": profile["communication"]["disable_p2p"] = False
    else: profile["communication"]["disable_p2p"] = 1
    with pytest.raises(ValueError): p.validate_profile(profile)


@pytest.mark.parametrize("disabled", [False, True])
def test_actual_transport_environment_is_exact_and_original_runtime_checks_still_run(disabled, monkeypatch):
    profile, _, _ = profile_fixture()
    profile["communication"] = p.communication_settings(disabled)
    profile["runtime"] = p.parallel_runtime(profile["qualified_C3"]["runtime"], "0,1", profile["communication"])
    for key in p.COMMUNICATION_ENV:
        if disabled: monkeypatch.setenv(key, p.COMMUNICATION_ENV[key])
        else: monkeypatch.delenv(key, raising=False)
    seen = []
    monkeypatch.setattr(p.g, "runtime_check", lambda runtime: seen.append(deepcopy(runtime)) or {"pinned": "versions"})
    assert p.runtime_check(profile) == {"pinned": "versions"}
    expected = deepcopy(profile["runtime"])
    for key in p.COMMUNICATION_ENV: expected["environment"].pop(key, None)
    assert seen == [expected] and profile["runtime"]["environment"].get("NCCL_P2P_DISABLE") == ("1" if disabled else None)
    monkeypatch.setenv("NCCL_P2P_DISABLE", "0")
    with pytest.raises(ValueError, match="Actual communication environment"): p.runtime_check(profile)
    assert len(seen) == 1


def test_transport_flag_does_not_bypass_original_runtime_rejection(monkeypatch):
    profile, _, _ = profile_fixture()
    for key in p.COMMUNICATION_ENV: monkeypatch.delenv(key, raising=False)
    def reject(runtime): raise ValueError("original pinned runtime mismatch")
    monkeypatch.setattr(p.g, "runtime_check", reject)
    with pytest.raises(ValueError, match="original pinned runtime"): p.runtime_check(profile)


@pytest.mark.parametrize("flag,disabled", [([], False), (["--disable-p2p"], True)])
def test_seal_CLI_passes_only_explicit_transport_opt_in(flag, disabled, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(p, "seal_profile", lambda args: observed.append(args.disable_p2p) or {"GPU_calls": 0})
    monkeypatch.setattr(sys, "argv", ["gemma_parallel", "seal-profile", "--baseline-run", "baseline",
        "--reference", "reference", "--name", "fixture", "--output", "fresh-sealed-profile", *flag])
    p.main()
    assert observed == [disabled] and json.loads(capsys.readouterr().out)["GPU_calls"] == 0


def test_KV_memory_variant_changes_exactly_one_engine_setting():
    default = p.engine_settings("checkpoint")
    explicit32 = p.engine_settings("checkpoint", 32)
    explicit56 = p.engine_settings("checkpoint", 56)
    assert default == explicit32
    assert {key for key in explicit32.keys() | explicit56.keys() if explicit32.get(key) != explicit56.get(key)} == {"kv_cache_memory_bytes"}
    assert explicit32["kv_cache_memory_bytes"] == 32 * 2**30
    assert explicit56["kv_cache_memory_bytes"] == 56 * 2**30
    assert p.RESOURCE_BUDGET == {"min_free_gpu_gib_per_device": 90, "min_available_ram_gib": 64,
        "kind": "admission budget, not measured peak", "KV_capacity_tokens_per_rank_minimum": 131072}
    assert '>= 2 * 65536' in p.placement_source()


@pytest.mark.parametrize("gib", [32, 56])
def test_KV_profile_binds_explicit_allocation_with_unchanged_runtime(gib):
    profile, _, _ = profile_fixture()
    before = deepcopy(profile)
    profile["kv_cache_gib"] = gib
    profile["engine_settings"] = p.engine_settings(profile["runtime"]["checkpoint"], gib)
    assert p.validate_profile(profile) is profile
    assert {k: v for k, v in profile.items() if k not in ("kv_cache_gib", "engine_settings")} == {
        k: v for k, v in before.items() if k not in ("kv_cache_gib", "engine_settings")}


@pytest.mark.parametrize("fault", ["unbound_bytes", "unbound_gib", "unsupported", "float", "string", "boolean"])
def test_KV_profile_rejects_tampering_and_non_integer_choices(fault):
    profile, _, _ = profile_fixture()
    if fault == "unbound_bytes": profile["engine_settings"]["kv_cache_memory_bytes"] = 56 * 2**30
    elif fault == "unbound_gib": profile["kv_cache_gib"] = 56
    else: profile["kv_cache_gib"] = {"unsupported": 48, "float": 32., "string": "32", "boolean": True}[fault]
    with pytest.raises(ValueError): p.validate_profile(profile)


@pytest.mark.parametrize("flag,gib", [([], 32), (["--kv-cache-gib", "32"], 32), (["--kv-cache-gib", "56"], 56)])
def test_seal_CLI_passes_explicit_KV_allocation(flag, gib, monkeypatch, capsys):
    observed = []
    monkeypatch.setattr(p, "seal_profile", lambda args: observed.append((args.kv_cache_gib, args.disable_p2p)) or {"GPU_calls": 0})
    monkeypatch.setattr(sys, "argv", ["gemma_parallel", "seal-profile", "--baseline-run", "baseline",
        "--reference", "reference", "--name", "fixture", "--output", "fresh-sealed-profile", "--disable-p2p", *flag])
    p.main()
    assert observed == [(gib, True)] and json.loads(capsys.readouterr().out)["GPU_calls"] == 0


def test_seal_CLI_rejects_unsupported_KV_choice_before_sealing(monkeypatch):
    def never(args): pytest.fail("Unsupported KV choice reached sealing")
    monkeypatch.setattr(p, "seal_profile", never)
    monkeypatch.setattr(sys, "argv", ["gemma_parallel", "seal-profile", "--baseline-run", "baseline",
        "--reference", "reference", "--name", "fixture", "--output", "fresh-sealed-profile", "--kv-cache-gib", "48"])
    with pytest.raises(SystemExit) as error: p.main()
    assert error.value.code == 2


@pytest.fixture
def v2_sources():
    """Use actual SDK bytes copied read-only, or the installed distribution; never import it."""
    root = os.environ.get("GEMMA_V2_SOURCE_ROOT")
    if root:
        locate = lambda name: Path(root) / name
    else:
        try: distribution = p.importlib.metadata.distribution("vllm")
        except p.importlib.metadata.PackageNotFoundError: pytest.skip("Actual V2 SDK sources not present; set GEMMA_V2_SOURCE_ROOT")
        locate = distribution.locate_file
    return {name: Path(locate(name)).read_text() for name in p.observer_interface()["sources"]}


def actual_V2_method(sources, file, cls, method, namespace):
    tree = ast.parse(sources[file])
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == cls)
    fn = deepcopy(next(node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name == method))
    fn.decorator_list = []  # Inference decorators require torch; method body/arguments remain native.
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual-installed-V2-CPU-seam>", "exec"), namespace)
    return namespace[method]


def test_actual_V2_installed_AST_chain_is_compatible_without_importing_GPU(v2_sources):
    assert p.observer_compatibility(v2_sources) == p.observer_interface()
    assert "vllm/v1/worker/gpu_model_runner.py" not in p.EXTRA_SOURCES
    assert not any(name in sys.modules for name in ("torch", "vllm", "transformers"))


@pytest.mark.parametrize("fault", ["legacy_only", "changed_source", "missing_source"])
def test_actual_V2_AST_preflight_rejects_unused_legacy_or_drift(v2_sources, fault):
    sources = dict(v2_sources)
    path = "vllm/v1/worker/gpu/mm/encoder_runner.py"
    if fault == "legacy_only": sources = {"vllm/v1/worker/gpu_model_runner.py": "class GPUModelRunner: pass"}
    elif fault == "changed_source": sources[path] = sources[path].replace("def prepare_mm_inputs(", "def obsolete_mm_inputs(", 1)
    else: del sources[path]
    with pytest.raises(ValueError, match="V2 observer source"): p.observer_compatibility(sources)


class CPUNativeTensor:
    def __init__(self, data): self.data = np.asarray(data)
    def __getitem__(self, index): return CPUNativeTensor(self.data[index])
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.data.tolist()


@pytest.fixture
def v2_hook_fixture(v2_sources, monkeypatch):
    profile, reference = case(); _, expected = pair_entries(profile, reference)
    objects = {}
    for key, qualified in p.V2_TYPES.items():
        module, name = qualified.rsplit(".", 1)
        objects[key] = type(name, (), {"__module__": module})()
    runner, model_state, encoder, requests = (objects[key] for key in ("runner", "model_state", "encoder", "request_state"))
    ids = list(expected)
    cache = NS(mm_features={rid: [NS(identifier=expected[rid]["mm_identifier"], modality="video", data={"owner": rid},
        mm_position=object())] for rid in ids}, encoder_outputs={})
    runner.req_states = requests
    runner.model_state = model_state
    runner.encoder_cache = model_state.encoder_cache = encoder.encoder_cache = cache
    model_state.encoder_runner = encoder
    requests.req_id_to_index = {rid: len(ids) - i - 1 for i, rid in enumerate(ids)}
    requests.index_to_req_id = {i: rid for rid, i in requests.req_id_to_index.items()}
    requests.prompt_len = NS(np=np.array([3, 3]))
    requests.all_token_ids = NS(gpu=CPUNativeTensor([expected[rid]["input_token_ids"] for rid in ids[::-1]]))
    type(encoder).prepare_mm_inputs = actual_V2_method(v2_sources,
        "vllm/v1/worker/gpu/mm/encoder_runner.py", "EncoderRunner", "prepare_mm_inputs", {})
    fake_torch = ModuleType("torch")
    fake_torch.from_numpy = lambda values: CPUNativeTensor(values)
    class Event:
        def __init__(self, **kwargs): pass
        def record(self): pass
    fake_torch.cuda = NS(Event=Event)
    fake_torch.bfloat16 = "torch.bfloat16"
    class Builder:
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata
    backend = ModuleType("vllm.v1.attention.backends.triton_attn")
    backend.TritonAttentionMetadataBuilder = Builder
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm.v1.attention.backends.triton_attn", backend)
    current = {}
    def compute_ranges(req_ids, mm_features, sliding_window):
        return {i: expected[rid]["native_ranges"] for i, rid in enumerate(req_ids)}
    def build_metadata(**kwargs):
        current["metadata_kwargs"] = kwargs
        if current.get("raise"): raise RuntimeError("native V2 builder failure")
        metadata = NS(query_start_loc=kwargs["query_start_loc_gpu"], seq_lens=kwargs["seq_lens"],
            num_actual_tokens=kwargs["num_tokens"], mm_prefix_range=kwargs["mm_req_doc_ranges"], causal=True)
        return {"layer": Builder().build(0, metadata)}
    type(model_state).prepare_attn = actual_V2_method(v2_sources,
        "vllm/v1/worker/gpu/model_states/default.py", "DefaultModelState", "prepare_attn",
        {"torch": fake_torch, "CUDAGraphMode": NS(FULL="FULL"), "compute_mm_prefix_ranges": compute_ranges,
         "build_attn_metadata": build_metadata})
    model_state.supports_mm_inputs = True
    model_state.model_config = NS(is_mm_prefix_lm=True, get_sliding_window=lambda: 1024)
    projection = NS(register_forward_pre_hook=lambda callback: current.update(projection_hook=callback))
    def forward(input_ids, positions, inputs_embeds=None, intermediate_tensors=None): return "unchanged-forward-result"
    model = NS(_process_video_input=lambda inputs: None, forward=forward,
        vision_tower=NS(patch_embedder=NS(input_proj=projection)))
    worker = NS(model_runner=runner, get_model=lambda: model)
    monkeypatch.setattr(p, "extra_sources", lambda: dict(p.EXTRA_SOURCES))
    monkeypatch.setattr(p, "observer_compatibility", lambda: p.observer_interface())
    receipt = p._install_pair_observer(worker)
    state = worker._gemma_pair; state["ledger"].arm(expected); state["active"] = True
    # The batch order differs from request slot order and encoder order.
    batch = NS(req_ids=ids, num_reqs=2, num_tokens=5, num_reqs_after_padding=2, num_tokens_after_padding=5,
        idx_mapping_np=np.array([requests.req_id_to_index[rid] for rid in ids]),
        query_start_loc=CPUNativeTensor([0, 2, 5]), query_start_loc_np=np.array([0, 2, 5]),
        seq_lens=CPUNativeTensor([4, 6]), seq_lens_cpu_upper_bound=np.array([4, 6]), max_query_len=None,
        num_scheduled_tokens=np.array([2, 3]), dcp_local_seq_lens=None, dcp_local_seq_lens_cpu_upper_bound=None,
        positions=CPUNativeTensor([2, 3, 3, 4, 5]), is_prefilling_np=np.array([True, True]), prompt_lens=None, fast_prefill=None)
    return NS(runner=runner, model_state=model_state, encoder=encoder, requests=requests, cache=cache, receipt=receipt,
        model=model, state=state, expected=expected, batch=batch, current=current, ids=ids)


def test_actual_V2_encoder_method_and_bound_hooks_preserve_native_owner_and_return(v2_hook_fixture):
    f = v2_hook_fixture
    scheduled = {rid: [0] for rid in f.ids[::-1]}
    result = f.encoder.prepare_mm_inputs(scheduled)
    assert result[0] == [f.expected[rid]["mm_identifier"] for rid in f.ids[::-1]]
    assert all(data is f.cache.mm_features[rid][0].data for (_, data), rid in zip(result[1], f.ids[::-1]))
    assert f.state["ledger"].consume([180, 180], 360) == f.ids[::-1]
    assert not hasattr(f.runner, "requests") and not hasattr(f.runner, "input_batch")
    assert f.receipt["actual_types"] == p.V2_TYPES and f.receipt["integration"] == p.observer_interface()


def test_actual_V2_attention_method_supplies_batch_identity_to_mask_and_positions(v2_hook_fixture):
    f = v2_hook_fixture
    result = f.model_state.prepare_attn(f.batch, "NONE", (), None, [], None)
    assert "layer" in result and f.state["last_sequences"]["request_ids"] == f.ids
    actual = f.current["metadata_kwargs"]["query_start_loc_gpu"]
    assert actual.tolist() == f.batch.query_start_loc.tolist() and np.shares_memory(actual.data, f.batch.query_start_loc.data)
    assert "input_batch" not in f.state
    assert f.model.forward(input_ids=None, positions=f.batch.positions) == "unchanged-forward-result"
    assert {row["request_id"] for row in f.state["positions"]} == set(f.ids)


@pytest.mark.parametrize("fault", ["slot_owner", "prompt_tokens", "foreign_feature", "reused_data"])
def test_actual_V2_owner_hooks_reject_mixed_request_state(v2_hook_fixture, fault):
    f = v2_hook_fixture; a, b = f.ids
    if fault == "slot_owner": f.requests.index_to_req_id[f.requests.req_id_to_index[a]] = b
    elif fault == "prompt_tokens": f.requests.all_token_ids.gpu.data[f.requests.req_id_to_index[a], 1] = 999
    elif fault == "foreign_feature": f.cache.mm_features[a][0].identifier = "foreign"
    else: f.cache.mm_features[b] = f.cache.mm_features[a]
    with pytest.raises(ValueError): f.encoder.prepare_mm_inputs({rid: [0] for rid in f.ids})


def test_actual_V2_attention_context_is_removed_on_native_exception(v2_hook_fixture):
    f = v2_hook_fixture; f.current["raise"] = True
    with pytest.raises(RuntimeError, match="native V2 builder failure"):
        f.model_state.prepare_attn(f.batch, "NONE", (), None, [], None)
    assert "input_batch" not in f.state and f.state["active"]


def test_actual_V2_attention_rejects_wrong_native_slot_mapping(v2_hook_fixture):
    f = v2_hook_fixture; f.batch.idx_mapping_np = f.batch.idx_mapping_np[::-1]
    with pytest.raises(ValueError, match="request slots"):
        f.model_state.prepare_attn(f.batch, "NONE", (), None, [], None)
    assert "input_batch" not in f.state


def test_installed_V2_type_rejection_precedes_hook_mutation(v2_hook_fixture):
    f = v2_hook_fixture
    encoder_hook, attn_hook = f.encoder.prepare_mm_inputs, f.model_state.prepare_attn
    f.runner.__class__ = type("GPUModelRunner", (), {"__module__": "vllm.v1.worker.gpu_model_runner"})
    with pytest.raises(ValueError, match="anchored V2 interface"):
        p.install_v2_context(f.runner, {"active": False})
    assert f.encoder.prepare_mm_inputs is encoder_hook and f.model_state.prepare_attn is attn_hook


@pytest.mark.parametrize("source", ["def outer(x):\n    def inner(y): return y\n    return inner(x)\n",
    "async def sample(x): return x\n", "class Sample:\n    def run(self): return 1\n"])
def test_portable_AST_ignores_only_empty_Python312_type_parameters_recursively(source):
    old, new = ast.parse(source), ast.parse(source)
    for node in ast.walk(old):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node._fields = tuple(name for name in node._fields if name != "type_params")
    for node in ast.walk(new):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node._fields = tuple(dict.fromkeys((*node._fields, "type_params")))
            node.type_params = []
    assert p.portable_ast_sha256(old) == p.portable_ast_sha256(new)
    generic = next(node for node in ast.walk(new) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
    generic.type_params = [ast.Name(id="T", ctx=ast.Load())]
    assert p.portable_ast_sha256(old) != p.portable_ast_sha256(new)


@pytest.mark.parametrize("changed", ["def sample(x): return x + 2\n", "def sample(y): return y + 1\n",
    "@decorator\ndef sample(x): return x + 1\n"])
def test_portable_AST_still_fingerprints_body_signature_and_decorators(changed):
    assert p.portable_ast_sha256(ast.parse("def sample(x): return x + 1\n")) != p.portable_ast_sha256(ast.parse(changed))


def test_portable_AST_keeps_non_type_parameter_empty_fields_and_constants():
    node = ast.parse("def sample(): return (..., b'\\x00')\n").body[0]
    canonical = p.canonical_ast(node)
    assert canonical["fields"]["decorator_list"] == [] and canonical["fields"]["type_comment"] is None
    value = canonical["fields"]["body"][0]["fields"]["value"]["fields"]["elts"]
    assert value[0]["fields"]["value"] == {"constant": "Ellipsis"}
    assert value[1]["fields"]["value"] == {"bytes": "00"}


@pytest.mark.parametrize("normalized", [False, True])
def test_actual_sampling_preserves_native_representation_and_canonical_choices(normalized):
    params = sampling()
    if normalized:
        params.structured_outputs.choice = None
        params.structured_outputs.grammar = p.CHOICE_GRAMMAR
        params.structured_outputs._backend = "xgrammar"
        params.structured_outputs._backend_was_auto = True
    value = p.actual_sampling(params)
    assert {key: value[key] for key in p.SAMPLING_SCALARS} == p.SAMPLING_SCALARS
    assert value["canonical_choices"] == ["LEFT", "RIGHT"]
    assert value["native_structured_outputs"]["choice"] == (None if normalized else ["LEFT", "RIGHT"])
    assert value["native_structured_outputs"]["grammar"] == (p.CHOICE_GRAMMAR if normalized else None)
    assert value["native_structured_outputs"]["_backend_was_auto"] is normalized
    params.structured_outputs.choice = ["CHANGED"]
    assert value["native_structured_outputs"]["choice"] == (None if normalized else ["LEFT", "RIGHT"])


@pytest.mark.parametrize("key,value", [("n", 2), ("temperature", .1), ("top_p", .9), ("seed", 1),
    ("max_tokens", 8192), ("repetition_penalty", 1.1)])
def test_native_sampling_scalar_changes_are_rejected(key, value):
    params = sampling(); setattr(params, key, value)
    with pytest.raises(ValueError, match="sampling scalars"): p.actual_sampling(params)


@pytest.mark.parametrize("key,value", [("grammar", 'root ::= "LEFT" | "RIGHT" | "OTHER"'),
    ("grammar", 'root ::= "LEFT"'), ("grammar", 'root ::= "LEFT" | "RIGHT"\n'),
    ("choice", ["LEFT", "RIGHT"]), ("json", {}), ("regex", "LEFT|RIGHT"), ("json_object", True),
    ("structural_tag", "{}"), ("disable_any_whitespace", True), ("disable_additional_properties", True),
    ("whitespace_pattern", " *"), ("_backend", "other")])
def test_native_sampling_rejects_broader_grammar_and_any_additional_constraint(key, value):
    params = sampling(); params.structured_outputs.choice = None; params.structured_outputs.grammar = p.CHOICE_GRAMMAR
    setattr(params.structured_outputs, key, value)
    with pytest.raises(ValueError): p.actual_sampling(params)


def test_normalized_native_sampling_is_persisted_in_real_pair_proof_and_compared(batch_fixture, monkeypatch):
    original = sampling
    def normalized():
        params = original(); params.structured_outputs.choice = None; params.structured_outputs.grammar = p.CHOICE_GRAMMAR
        params.structured_outputs._backend = "xgrammar"
        return params
    monkeypatch.setattr(sys.modules[__name__], "sampling", normalized)
    reference, _ = closed_batch_fixture(batch_fixture, monkeypatch)
    intents = [row for row in p.read_lines(batch_fixture.root / "events.jsonl") if row["event"] == "GENERATE_INTENT"]
    assert len(intents) == 16
    assert all(row["actual_sampling"]["canonical_choices"] == ["LEFT", "RIGHT"]
        and row["actual_sampling"]["native_structured_outputs"]["choice"] is None
        and row["actual_sampling"]["native_structured_outputs"]["grammar"] == p.CHOICE_GRAMMAR for row in intents)
    assert p.compare(batch_fixture.root, reference)["status"] == "MATCH"


def test_comparer_rejects_changed_native_choices_even_with_rehashed_byte_proofs(batch_fixture, monkeypatch):
    reference, close = closed_batch_fixture(batch_fixture, monkeypatch)
    path = batch_fixture.root / "events.jsonl"; data = path.read_bytes()
    changed = data.replace(b'"choice":["LEFT","RIGHT"]', b'"choice":["WRNG","RIGHT"]', 1)
    assert changed != data and len(changed) == len(data)
    path.write_bytes(changed)
    calls = p.read_lines(batch_fixture.root / "calls.jsonl")
    for proof in calls:
        event = proof["events"]; event["sha256"] = hashlib.sha256(changed[event["start"]:event["end"]]).hexdigest()
    (batch_fixture.root / "calls.jsonl").write_text("".join(p.g.canonical_json(proof) + "\n" for proof in calls))
    close()
    with pytest.raises(ValueError, match="fixed two-choice language"): p.compare(batch_fixture.root, reference)


def test_actual_installed_native_choice_validator_CPU_roundtrip():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "": pytest.skip("Native validator smoke requires explicit CPU-only visibility")
    try: p.importlib.metadata.distribution("vllm")
    except p.importlib.metadata.PackageNotFoundError: pytest.skip("Native validator SDK is not installed")
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from vllm.v1.structured_output.backend_xgrammar import validate_xgrammar_grammar
    assert not torch.cuda.is_initialized()
    params = SamplingParams(n=1, temperature=0., top_p=1., repetition_penalty=1., seed=0,
        max_tokens=16384, structured_outputs=StructuredOutputsParams(choice=["LEFT", "RIGHT"]))
    before = p.actual_sampling(params)
    validate_xgrammar_grammar(params)
    after = p.actual_sampling(params)
    assert before["canonical_choices"] == after["canonical_choices"] == ["LEFT", "RIGHT"]
    assert before["native_structured_outputs"]["choice"] == ["LEFT", "RIGHT"]
    assert after["native_structured_outputs"]["choice"] is None
    assert after["native_structured_outputs"]["grammar"] == 'root ::= "LEFT" | "RIGHT"'
    assert all(before[key] == after[key] for key in p.SAMPLING_SCALARS)
    assert not torch.cuda.is_initialized()
