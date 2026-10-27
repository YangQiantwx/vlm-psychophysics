"""CPU seam checks; these do not assert GPU speed or numerical equivalence."""
import ast
import json
from pathlib import Path
import random

import numpy as np
import pytest

from classic_precision import gemma_speed as g


def ranges():
    return [(37 + 184 * i, 37 + 184 * i + 175) for i in range(180)]


def native_mask(q, k, intervals, window):
    visible = k <= q
    if window > 0:
        visible &= q - k < window
    for a, b in intervals:
        mm = a < b and a <= q <= b and a <= k <= b
        if window > 0:
            mm &= q - k < window
        visible |= mm
    return visible


@pytest.mark.parametrize("window", [0, 1, 1024])
def test_indexed_mask_all_frame_boundaries_and_causal_window_edges(window):
    intervals = ranges()
    g.validate_ranges(intervals)
    qs = {-1, 0, 36, 37, intervals[-1][1], intervals[-1][1] + 1, 65535}
    for a, b in intervals:
        qs.update((a - 1, a, a + 1, b - 1, b, b + 1))
    for q in qs:
        keys = {0, 36, 37, q - 1024, q - 1023, q - 1, q, q + 1, q + 175, 65535}
        for a, b in intervals:
            if abs(q - a) < 190 or abs(q - b) < 190:
                keys.update((a - 1, a, a + 1, b - 1, b, b + 1))
        for k in keys:
            assert g.indexed_mask(q, k, intervals, window) == native_mask(q, k, intervals, window)


def test_empty_intervals_follow_causal_window_cpu_spec():
    for q in (-1, 0, 1023, 1024, 65535):
        for k in (-1, 0, 1023, 1024, 65535):
            assert g.indexed_mask(q, k, [], 1024) == native_mask(q, k, [], 1024)
    with pytest.raises(ValueError, match="180"):
        g.validate_ranges([])


def test_random_full_range_mask_pairs():
    rng = random.Random(0)
    intervals = ranges()
    for _ in range(4000):
        q, k = rng.randrange(65536), rng.randrange(65536)
        assert g.indexed_mask(q, k, intervals) == native_mask(q, k, intervals, 1024)


@pytest.mark.parametrize("mutation", ["overlap", "unsorted", "empty", "too_long"])
def test_rejects_non_equivalent_interval_layout(mutation):
    rs = ranges()
    if mutation == "overlap":
        rs[1] = (rs[0][1], rs[1][1])
    elif mutation == "unsorted":
        rs[1], rs[2] = rs[2], rs[1]
    elif mutation == "empty":
        rs[1] = (221, 221)
    else:
        rs[-1] = (rs[-1][0], rs[-1][0] + 1024)
    with pytest.raises(ValueError):
        g.validate_ranges(rs)


def test_fp32_conversion_preserves_every_possible_uint16_value():
    codes = np.arange(65536, dtype=np.uint16)
    original = codes.astype(np.float32) * (255. / 65535.)
    candidate = codes.astype(np.float32)
    np.multiply(candidate, 255. / 65535., out=candidate)
    assert candidate.dtype == original.dtype == np.float32
    assert np.array_equal(original.view(np.uint32), candidate.view(np.uint32))


HELPERS = '''
@triton.jit
def compute_kv_seq_mask(query_abs_pos, seq_offset, MAX_MM_RANGES):
    seq_mask = seq_offset <= query_abs_pos
    if USE_MM_PREFIX:
        for i in range(MAX_MM_RANGES):
            seq_mask |= original_native_mask(i)
    return seq_mask
@triton.jit
def compute_tile_loop_bounds(MAX_MM_RANGES):
    if USE_MM_PREFIX and MM_PREFIX_CLAMP_SW:
        for i in range(MAX_MM_RANGES):
            first_allowed_key = native_first(i)
            last_allowed_key = native_last(i)
    return first_allowed_key, last_allowed_key
'''
LAUNCHER = '''
def unified_attention(q, max_seqlen_q):
    head_size = q.shape[2]
    num_queries_per_kv = 8
    BLOCK_M = 16 if num_queries_per_kv <= 16 else 32
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    if head_size == 256:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
    return BLOCK_M, BLOCK_Q
'''


@pytest.mark.parametrize("binary,block", [(False, 0), (True, 0), (False, 32), (True, 32)])
def test_kernel_switches_are_independent_and_change_no_native_math(binary, block):
    helpers, launcher = g.candidate_sources(HELPERS, LAUNCHER, binary, block)
    ast.parse(helpers)
    namespace = {}
    exec(launcher, namespace)
    def q(shape):
        return type("Q", (), {"shape": shape})()
    assert namespace["unified_attention"](q((33000, 32, 512)), 33000) == ((32, 4) if block else (16, 2))
    assert namespace["unified_attention"](q((1, 32, 512)), 1) == (16, 2)
    assert namespace["unified_attention"](q((33000, 32, 256)), 33000) == (32, 4)
    assert ("range(9)" in helpers) == binary
    assert ("native_first" in helpers) == (not binary)


def test_source_transform_refuses_missing_native_anchor():
    with pytest.raises(ValueError, match="enumeration"):
        g.candidate_sources(HELPERS.replace("range(MAX_MM_RANGES)", "new_native_iterator()"), LAUNCHER, True, 0)


def test_nine_step_binary_algorithm_for_all_query_positions():
    rs = ranges()
    starts, ends = np.array(rs).T
    q = np.arange(-1, 65537)
    lo, hi = np.zeros_like(q), np.full_like(q, len(rs))
    for _ in range(9):
        mid = (lo + hi) // 2
        active = lo < hi
        probe = starts[np.minimum(mid, len(rs) - 1)]
        advance = (mid < len(rs)) & (probe <= q)
        lo = np.where(active & advance, mid + 1, lo)
        hi = np.where(active & ~advance, mid, hi)
    assert np.array_equal(lo, np.searchsorted(starts, q, side="right"))


def test_vector_bounds_preserve_native_tile_union():
    rs = np.array(ranges())
    rng = random.Random(1)
    for _ in range(1000):
        qlo = rng.randrange(65536)
        qhi = qlo + rng.randrange(1, 17)
        first, last = qlo - 1023, qhi
        expected_first, expected_last = first, last
        for start, end in rs:
            if start < end and start <= qhi and end >= qlo:
                expected_first = min(expected_first, max(start, qlo - 1023))
                expected_last = max(expected_last, end)
        intersects = (rs[:, 0] < rs[:, 1]) & (rs[:, 0] <= qhi) & (rs[:, 1] >= qlo)
        actual_first = min(first, np.min(np.where(intersects, np.maximum(rs[:, 0], qlo - 1023), first)))
        actual_last = max(last, np.max(np.where(intersects, rs[:, 1], last)))
        assert (actual_first, actual_last) == (expected_first, expected_last)


def test_timing_excludes_only_first_two_qualification_calls():
    rows = [{"deep": i < 2, "status": "PASS", "full_request_seconds": 100 if i < 2 else 24} for i in range(20)]
    result = g.timing_summary(rows)
    assert result["ordinary_n"] == 18 and result["deep_qualification_n"] == 2
    assert result["speed_threshold_met"] and result["max_seconds"] == 24
    for row in rows[2:]:
        row["full_request_seconds"] = 28
    assert not g.timing_summary(rows)["speed_threshold_met"]


def test_changed_control_stops_before_model_call():
    data = {k: "same" for k in ("input_token_ids", "input_token_ids_sha256", "prompt_sha256", "source_stream_sha256",
        "position_sha256", "position_shape", "HF_control_tensors", "processor_input_FP32_sha256", "pixel_FP32_sha256",
        "input_sha256", "projection_sha256")}
    g.verify_input(data, data, True)
    changed = dict(data, projection_sha256="wrong")
    with pytest.raises(ValueError, match="projection"):
        g.verify_input(changed, data, True)


def test_no_torch_import_in_cpu_plan_module():
    # This module intentionally exposes its plan/source-transformation seams
    # without importing model runtimes; test in a fresh interpreter.
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-B", "-c",
        "import classic_precision.gemma_speed,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"],
        check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def write_observation(parent, index):
    """Small on-disk journal fixture, with 48 IDs and a distinct cold engine."""
    root = parent / f"observation-{index}"
    root.mkdir()
    ids = [f"fixture-{i:02d}" for i in range(48)]
    mask = [{"sha256": "native-mask", "ranges": {"0": [list(r) for r in ranges()]}}]
    rows = [{"sample_id": sid, "status": "PASS", "deep": i < 2,
        "full_request_seconds": 24., "native_masks": mask} for i, sid in enumerate(ids)]
    profile = {"panel": {"items": [{"sample_id": sid} for sid in ids]}, "variant": "cpu",
        "kernel": {"range_lookup": "native", "prefill_block_m": 0},
        "implementation": {"module": "frozen"}, "runtime_sources": {"sdk": "frozen"},
        "checkpoint_assets": {"revision": g.REVISION},
        "engine_settings": {"model": "/fixture/checkpoint", "dtype": "bfloat16"}}
    host, device = (("fourier", "gpu0") if index < 2 else
                    ("fourier", "gpu1") if index < 4 else ("radon", "gpu0"))
    identity = {"profile": profile, "engine_id": f"cold-engine-{index}",
        "host": host, "lease": {"device": device}}
    load = {"placement": [{"worker": {"host": host, "pid": 1000 + index,
        "startticks": str(10000 + index), "boot_id": "fixture-boot"},
        "torch_policy": {"TF32_matmul": False, "matmul_precision": "highest"}}]}
    for name, content in {"identity.json": identity, "load.json": load,
        "result.json": {"status": "COMPLETE", "calls": 48, "panel_sha256": "common-panel",
                        "rows": rows, "timing": g.timing_summary(rows)},
        "exit.json": {"status": "COMPLETE", "owned_engine_workers_exited": True,
                      "lease_released_after_engine_exit": True}}.items():
        (root / name).write_text(json.dumps(content))
    raw = [{"sample_id": sid, "raw_answer": "LEFT", "output_token_ids": [31884, 106],
        "finish_reason": "stop", "stop_reason": None, "actual_input_token_ids": [2, 123, 456],
        "input_receipt": {"source_stream_sha256": sid + "-source", "prompt_sha256": "prompt",
            "position_sha256": "positions", "HF_control_tensors": {"mask": "identical"}}} for sid in ids]
    (root / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    return root


def alter_raw(root, field, value):
    raw = [json.loads(s) for s in (root / "raw.jsonl").read_text().splitlines()]
    if field.startswith("input_receipt."):
        raw[17]["input_receipt"][field.split(".")[1]] = value
    else:
        raw[17][field] = value
    (root / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))


def test_six_cold_observations_on_three_cards_pass_complete_tokens(tmp_path):
    paths = [write_observation(tmp_path, i) for i in range(6)]
    result = g.compare_runs(paths, purpose="consistency")
    assert result["status"] == "MATCH"
    assert result["observations"] == 6 and result["unique_ids"] == 48
    assert result["six_observation_consistency_pass"]
    assert not result["production_authorized_by_this_command"]


def test_same_answer_but_different_full_tokens_fails_consistency(tmp_path):
    paths = [write_observation(tmp_path, i) for i in range(6)]
    alter_raw(paths[-1], "output_token_ids", [31884, 107])
    result = g.compare_runs(paths, purpose="consistency")
    assert result["status"] == "DIFFERENT"
    assert result["output_differences"][0]["fields"] == ["output_token_ids"]
    assert not result["six_observation_consistency_pass"]


@pytest.mark.parametrize("problem", ["repeated_engine", "missing_exit", "input_drift"])
def test_six_observation_gate_rejects_incomplete_or_different_evidence(tmp_path, problem):
    paths = [write_observation(tmp_path, i) for i in range(6)]
    if problem == "repeated_engine":
        identity = json.loads((paths[-1] / "identity.json").read_text())
        identity["engine_id"] = "cold-engine-0"
        (paths[-1] / "identity.json").write_text(json.dumps(identity))
        with pytest.raises(ValueError, match="Independent cold"):
            g.compare_runs(paths, purpose="consistency")
    elif problem == "missing_exit":
        (paths[-1] / "exit.json").unlink()
        with pytest.raises(FileNotFoundError):
            g.compare_runs(paths, purpose="consistency")
    else:
        alter_raw(paths[-1], "input_receipt.position_sha256", "different-position-tensor")
        result = g.compare_runs(paths, purpose="consistency")
        assert result["status"] == "INVALID_EVIDENCE"
        assert not result["six_observation_consistency_pass"]


def test_kernel_profile_difference_is_separate_from_same_runtime_replay(tmp_path):
    a, b = write_observation(tmp_path, 0), write_observation(tmp_path, 1)
    identity = json.loads((b / "identity.json").read_text())
    identity["profile"]["variant"] = "kernel"
    identity["profile"]["kernel"]["prefill_block_m"] = 32
    (b / "identity.json").write_text(json.dumps(identity))
    alter_raw(b, "output_token_ids", [31884, 107])
    result = g.compare_runs([a, b], purpose="candidate")
    assert result["status"] == "PROFILE_OUTPUT_DIFFERENCE"
    assert not result["input_differences"]
    assert not result["six_observation_consistency_pass"]
