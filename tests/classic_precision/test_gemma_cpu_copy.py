"""CPU overlay guards, exact GC targeting, failure cleanup and strict comparison.

The six real CPU preparations already establish the array-copy equivalence;
these tests exercise the newly added control/lifecycle seams without inference.
"""
from argparse import Namespace
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace

import pytest

from classic_precision import gemma_cpu_copy as c


SDK = '''class Gemma4VideoProcessor:
    def _preprocess(self, videos):
        pixel_values = list(videos)
        position_ids = list(videos)
        pixel_values = torch.stack(pixel_values, dim=0)
        position_ids = torch.stack(position_ids, dim=0)
        return pixel_values, position_ids
'''


@pytest.fixture
def sdk(monkeypatch):
    value = c.processor_candidate(SDK)
    monkeypatch.setattr(c, "installed_processor_source", lambda: value)
    return value


def base_profile(n=20, placement="baseline"):
    checkpoint = "/fixture/" + c.g.REVISION
    p = {"schema": c.g.SCHEMA, "name": "cpu-fixture", "runtime": {"checkpoint": checkpoint},
        "variant": "kernel", "engine_settings": c.m._BASE_ENGINE_SETTINGS(checkpoint),
        "implementation": {"gemma_speed": c.m.BASE_SHA, "inference": c.m.INFERENCE_SHA},
        "runtime_sources": {c.SDK_FILE: c.processor_candidate(SDK)[1]["file_sha256"]},
        "checkpoint_assets": {"revision": c.g.REVISION}, "first_deep_calls": 2,
        "maximum_model_calls": n, "automatic_retry": False,
        "panel": {"items": [{"sample_id": f"sample-{i}"} for i in range(n)]},
        "kernel": {"range_lookup": "binary", "prefill_block_m": 32, "expected_native_sha256": c.g.NATIVE_SHA}}
    if placement == "memory28":
        original = deepcopy(p)
        p["engine_settings"] = c.m.engine_settings(checkpoint)
        p["execution_overlay"] = c.m.overlay_record(original)
    return p


def profile(n=20, placement="baseline", cadence=8):
    p = base_profile(n, placement)
    p["cpu_execution_overlay"] = c.overlay_record(deepcopy(p), placement, cadence, c.processor_candidate(SDK)[1])
    return p


@pytest.mark.parametrize("placement", ["baseline", "memory28"])
def test_metadata_layers_keep_engine_and_base_identity(sdk, placement):
    p = profile(placement=placement)
    assert c.validate_profile(p) is p
    assert p["implementation"] == base_profile()["implementation"]
    assert p["engine_settings"] == base_profile(placement=placement)["engine_settings"]
    assert ("execution_overlay" in p) == (placement == "memory28")
    assert p["cpu_execution_overlay"]["processor_source"] == sdk[1]
    assert p["cpu_execution_overlay"]["reference_file_sha256"] == c.m.REFERENCE_SHA[20]


@pytest.mark.parametrize("mutation", ["source", "gc", "derived", "deep", "engine"])
def test_profile_drift_refused_before_execution(sdk, mutation):
    p = profile()
    if mutation == "source": p["cpu_execution_overlay"]["source_sha256"] = "wrong"
    elif mutation == "gc": p["cpu_execution_overlay"]["gc_cadence"] = 4
    elif mutation == "derived": p["cpu_execution_overlay"]["processor_source"]["derived_function_sha256"] = "wrong"
    elif mutation == "deep": p["first_deep_calls"] = 0
    else: p["engine_settings"]["kv_cache_memory_bytes"] = 16 * 2**30
    with pytest.raises(ValueError): c.validate_profile(p)


def test_native_anchor_change_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="stack anchors"):
        c.processor_candidate(SDK.replace("torch.stack(pixel_values, dim=0)", "torch.stack(pixel_values, dim=1)"))
    code, receipt = c.processor_candidate(SDK)
    assert code.count("unsqueeze(0)") == 2
    assert code.count("else torch.stack") == 2
    assert receipt["changed_assignments"] == ["pixel_values", "position_ids"]


def test_seal_uses_temporary_base_preserves_existing_profiles(sdk, tmp_path, monkeypatch):
    called = []
    def base_seal(args):
        called.append(Path(args.output))
        assert args.panel_key == "panel"
        c.g.seal(args.output, base_profile())
    monkeypatch.setattr(c.g, "seal_profile", base_seal)
    target = tmp_path / "new-profile.json"
    args = Namespace(output=str(target), placement="baseline", gc_cadence=8)
    assert c.seal_profile(args)["GPU_calls"] == 0
    assert not called[0].exists()
    before = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"): c.seal_profile(args)
    assert target.read_bytes() == before and len(called) == 1


class FakeGC:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def collect(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("injected GC failure")
        return 7


def drive(proxy, flags):
    proxy.collect()  # startup: same caller code, different source line
    for i, deep in enumerate(flags):
        sid = f"sample-{i}"
        proxy.collect()  # the only intercepted call site
    proxy.collect()  # cleanup: must remain native


def drive_site():
    lines, start = inspect.getsourcelines(drive)
    calls = sorted(n.lineno for n in ast.walk(ast.parse(textwrap.dedent("".join(lines))))
                   if isinstance(n, ast.Call) and ast.unparse(n.func) == "proxy.collect")
    return start + calls[1] - 1


@pytest.mark.parametrize("cadence,expected_calls,loop_calls", [(0, 23, 20), (8, 7, 4)])
def test_gc_targets_only_loop_and_bounds_pending(monkeypatch, cadence, expected_calls, loop_calls):
    native = FakeGC()
    monkeypatch.setattr(c, "_GC", native)
    proxy = c.GCCadence(cadence, drive.__code__, drive_site())
    drive(proxy, [True, True] + [False] * 18)
    assert len(proxy.rows) == 20
    assert sum(r["collected"] for r in proxy.rows) == loop_calls
    assert proxy.maximum_pending == cadence
    proxy.finish()
    assert len(native.calls) == expected_calls
    assert proxy.pending == 0 and proxy.final["status"] == "COMPLETE"
    assert all(r["before"]["process_peak_rss_bytes"] > 0 for r in proxy.rows)


def test_foreign_caller_always_uses_original_gc(monkeypatch):
    native = FakeGC()
    monkeypatch.setattr(c, "_GC", native)
    proxy = c.GCCadence(8, drive.__code__, drive_site())
    assert proxy.collect(1) == 7
    assert native.calls == [((1,), {})] and proxy.rows == []


def test_ordinary_method_and_decode_restore_on_prepare_failure(sdk, tmp_path, monkeypatch):
    path = tmp_path / "sdk.py"
    path.write_text(SDK)
    namespace = {"__name__": "fixture_gemma_sdk"}
    exec(compile(SDK, str(path), "exec"), namespace)
    cls = namespace["Gemma4VideoProcessor"]
    original_method = cls._preprocess
    session = SimpleNamespace(processor=SimpleNamespace(video_processor=cls()))
    calls = []
    def original_prepare(session, item, roots, streamlined, deep):
        calls.append(deep)
        if deep:
            assert c.g.decode is c._DECODE and cls._preprocess is original_method
            return "deep-original"
        assert c.g.decode is c.direct_decode and cls._preprocess is not original_method
        raise RuntimeError("injected prepare failure")
    monkeypatch.setattr(c, "_PREPARE", original_prepare)
    monkeypatch.setattr(c.g, "prepare", original_prepare)
    with pytest.raises(RuntimeError, match="prepare failure"):
        with c.installed_overlay(profile()) as state:
            assert c.g.prepare(session, {}, {}, True, True) == "deep-original"
            c.g.prepare(session, {}, {}, True, False)
    assert calls == [True, False]
    assert cls._preprocess is original_method and c.g.decode is c._DECODE
    assert c.g.prepare is original_prepare and c.g.gc is c._GC
    assert state.final["status"] == "COMPLETE"


def test_failed_eighth_collect_still_restores_and_finally_drains(sdk, monkeypatch):
    native = FakeGC(fail_on=4)  # startup, two deep collects, eighth ordinary
    monkeypatch.setattr(c, "_GC", native)
    monkeypatch.setattr(c.g, "gc", native)
    with pytest.raises(RuntimeError, match="GC failure"):
        with c.installed_overlay(profile()) as state:
            state.code, state.line = drive.__code__, drive_site()
            drive(c.g.gc, [True, True] + [False] * 18)
    assert state.maximum_pending == 8 and state.pending == 0
    assert state.final["status"] == "COMPLETE"
    assert c.g.prepare is c._PREPARE and c.g.gc is native


def test_final_collect_failure_restores_bindings(sdk, monkeypatch):
    native = FakeGC(fail_on=1)
    monkeypatch.setattr(c, "_GC", native)
    monkeypatch.setattr(c.g, "gc", native)
    with pytest.raises(RuntimeError, match="GC failure"):
        with c.installed_overlay(profile()) as state:
            pass
    assert c.g.prepare is c._PREPARE and c.g.gc is native
    assert state.final["status"] == "FAILED"


def make_run(tmp_path, index, cpu=True, cadence=8, n=20, host="fourier", device="0", placement="baseline"):
    root = tmp_path / f"run-{index}"
    root.mkdir()
    p = profile(n, placement, cadence) if cpu else base_profile(n, placement)
    ids = [r["sample_id"] for r in p["panel"]["items"]]
    def write(name, value): c.g.seal(root / name, value)
    write("identity.json", {"profile": p, "profile_sha256": c.g.digest(p), "engine_id": f"engine-{index}",
        "host": host, "lease": {"device": device}})
    write("load.json", {"placement": [{"worker": {"host": host, "pid": index + 1, "startticks": str(index + 1), "boot_id": "boot"},
        "torch_policy": {"TF32_matmul": False, "TF32_cudnn": True, "matmul_precision": "highest"}}]})
    write("exit.json", {"status": "COMPLETE", "owned_engine_workers_exited": True, "lease_released_after_engine_exit": True})
    write("result.json", {"status": "COMPLETE", "calls": n, "panel_sha256": c.g.digest(p["panel"]), "timing": {},
        "rows": [{"sample_id": sid, "status": "PASS", "native_masks": []} for sid in ids]})
    raw = [{"sample_id": sid, "output_token_ids": [11, 12], "raw_answer": "LEFT", "finish_reason": "stop",
        "stop_reason": None, "actual_input_token_ids": [1, 2], "input_receipt": {"source_stream_sha256": sid,
            "prompt_sha256": "prompt", "position_sha256": "position", "HF_control_tensors": {}}} for sid in ids]
    (root / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in raw))
    if cpu:
        write("cpu-copy.json", {"status": "COMPLETE", "profile_sha256": c.g.digest(p),
            "source_sha256": c.g.file_sha256(c.__file__), "gc_cadence": cadence, "maximum_pending_ordinary": cadence,
            "final_gc": {"status": "COMPLETE"}, "rows": [{"sample_id": sid, "deep": i < 2} for i, sid in enumerate(ids)]})
    return root


def test_cpu_candidate_matches_b2_and_token_change_is_different(tmp_path):
    a, b = make_run(tmp_path, 0, cpu=False), make_run(tmp_path, 1)
    assert c.compare_runs([a, b])["status"] == "MATCH"
    rows = [json.loads(s) for s in (b / "raw.jsonl").read_text().splitlines()]
    rows[0]["output_token_ids"] = [11, 99]  # same final answer, different full tokens
    (b / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert c.compare_runs([a, b])["status"] == "DIFFERENT"


def test_different_b0_kernel_cannot_excuse_cpu_output_difference(tmp_path):
    a, b = make_run(tmp_path, 0, cpu=False), make_run(tmp_path, 1)
    value = c.g.read(a / "identity.json")
    value["profile"]["kernel"]["range_lookup"] = "native"
    value["profile_sha256"] = c.g.digest(value["profile"])
    (a / "identity.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="identical placement, engine and kernel"):
        c.compare_runs([a, b])


def test_different_cpu_cadence_is_not_one_execution_profile(tmp_path):
    a, b = make_run(tmp_path, 0), make_run(tmp_path, 1, cadence=0)
    with pytest.raises(ValueError, match="algorithm, parameters or source"):
        c.compare_runs([a, b])


def test_six_cold_consistency_delegates_all_original_checks(tmp_path):
    paths = [make_run(tmp_path, i, n=48, host="fourier" if i < 4 else "radon", device=str(i // 2) if i < 4 else "0") for i in range(6)]
    assert c.compare_runs(paths, "consistency")["six_observation_consistency_pass"]
    value = c.g.read(paths[-1] / "identity.json")
    value["engine_id"] = "engine-0"
    (paths[-1] / "identity.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Independent cold engine"):
        c.compare_runs(paths, "consistency")


def test_missing_final_gc_receipt_rejects_otherwise_complete_run(tmp_path):
    a, b = make_run(tmp_path, 0, cpu=False), make_run(tmp_path, 1)
    value = c.g.read(b / "cpu-copy.json")
    value["final_gc"]["status"] = "FAILED"
    (b / "cpu-copy.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completion/final GC"):
        c.compare_runs([a, b])


def test_run_delegates_and_writes_cpu_receipt_after_engine_exit(sdk, tmp_path, monkeypatch):
    reference = tmp_path / "reference.json"
    reference.write_text("existing-reference")
    monkeypatch.setitem(c.m.REFERENCE_SHA, 20, c.g.file_sha256(reference))
    target, output = tmp_path / "profile.json", tmp_path / "output"
    c.g.seal(target, profile(placement="memory28"))
    order = []
    def delegated(args):
        assert c.g.engine_settings is c.m.engine_settings and c.g.physical_lease is c.m.physical_lease
        assert c.g.gc is not c._GC and c.g.prepare is not c._PREPARE
        output.mkdir()
        order.extend(["generation", "owned-engine-exit", "lease-release"])
        assert not (output / "cpu-copy.json").exists()
        return {"status": "COMPLETE"}
    monkeypatch.setattr(c.g, "run", delegated)
    result = c.run(Namespace(profile=str(target), reference=str(reference), output=str(output)))
    assert result["status"] == "COMPLETE" and order[-1] == "lease-release"
    assert c.g.read(output / "cpu-copy.json")["final_gc"]["status"] == "COMPLETE"
    assert c.g.engine_settings is c.m._BASE_ENGINE_SETTINGS and c.g.physical_lease is c.m._BASE_PHYSICAL_LEASE
    assert c.g.prepare is c._PREPARE and c.g.gc is c._GC


def test_import_and_help_do_not_load_gpu_packages():
    done = subprocess.run([sys.executable, "-B", "-c", "import classic_precision.gemma_cpu_copy,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
