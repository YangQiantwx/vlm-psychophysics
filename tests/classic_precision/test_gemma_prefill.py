"""New launcher/profile seams only; no GPU or repeated CPU input preparation."""
from argparse import Namespace
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import gemma_prefill as p
from test_gemma_cpu_copy import SDK, make_run as make_cpu_run, profile as cpu_profile


HELPERS = '''@triton.jit
def compute_kv_seq_mask():
    for i in range(MAX_MM_RANGES):
        pass
@triton.jit
def compute_tile_loop_bounds():
    for i in range(MAX_MM_RANGES):
        pass
'''
LAUNCHER = '''def unified_attention(head_size, max_seqlen_q, num_queries_per_kv, current_platform):
    BLOCK_M = 16 if num_queries_per_kv <= 16 else 1 << (num_queries_per_kv - 1).bit_length()
    BLOCK_Q = BLOCK_M // num_queries_per_kv
    launch_num_warps = None
    launch_num_stages = None
    tuned_large_head = head_size == 256 and max_seqlen_q > 1 and num_queries_per_kv <= 16 and current_platform.is_device_capability_family(100)
    if tuned_large_head:
        BLOCK_M = 32
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        launch_num_warps = 8
        launch_num_stages = 2
    TILE_SIZE_PREFILL = 32
    if tuned_large_head:
        TILE_SIZE_PREFILL = 128
    return BLOCK_M, BLOCK_Q, TILE_SIZE_PREFILL, launch_num_warps, launch_num_stages
'''


def artifacts():
    helper, base = p.g.candidate_sources(HELPERS, LAUNCHER, True, 32)
    derived = p.derive_launcher(base)
    return {"helper_source": helper, "base_source": base, "derived_source": derived,
        "receipt": {"native_source_sha256": p.g.NATIVE_SHA, "binary_helper_sha256": p.sha(helper),
            "base_launcher_sha256": p.sha(base), "derived_launcher_sha256": p.sha(derived),
            "only_added_statement": f"if {p._LOCAL_TEST}: BLOCK_M = 32"}}


@pytest.fixture
def source_seam(monkeypatch):
    a = artifacts()
    monkeypatch.setattr(p, "kernel_artifacts", lambda sources=None: a)
    monkeypatch.setattr(p.c, "installed_processor_source", lambda: p.c.processor_candidate(SDK))
    return a


def prefill_profile(n=20):
    original = cpu_profile(n=n)
    value = deepcopy(original)
    value["kernel"]["local256_prefill"] = p.local_parameters()
    value["kernel_execution_overlay"] = p.kernel_overlay_record(original, artifacts()["receipt"])
    p._rebind_cpu(value)
    return value


class Platform:
    def is_device_capability_family(self, value):
        return value == 120


@pytest.mark.parametrize("head,queries,ratio,expected_m", [
    (256, 100, 2, 32), (512, 100, 2, 32), (256, 1, 2, 16),
    (512, 1, 2, 16), (128, 100, 2, 16), (256, 100, 32, 32)])
def test_dispatch_changes_only_local_prefill(head, queries, ratio, expected_m):
    a = artifacts()
    base = p._compile_launcher(a["base_source"], {}, "<gemma-native-query-tiling>")
    candidate = p._compile_launcher(a["derived_source"], {}, "<fixture-prefill>")
    before, after = base(head, queries, ratio, Platform()), candidate(head, queries, ratio, Platform())
    assert after == (expected_m, expected_m // ratio, 32, None, None)
    if head != 256 or queries == 1 or ratio > 16:
        assert after == before
    else:
        assert before[0] == 16 and after[0] == 32
        assert after[2:] == before[2:]


def test_changed_source_anchors_and_unknown_native_sha_are_refused():
    a = artifacts()
    with pytest.raises(ValueError, match="head512 M32 anchor"):
        p.derive_launcher(a["base_source"].replace("head_size == 512", "head_size == 384"))
    with pytest.raises(ValueError, match="assignment order"):
        p.derive_launcher(a["base_source"].replace("BLOCK_Q = BLOCK_M // num_queries_per_kv", "BLOCK_Q = 1", 1))
    with pytest.raises(ValueError, match="already present"):
        p.derive_launcher(a["derived_source"])
    with pytest.raises(ValueError, match="native kernel source"):
        p.kernel_artifacts({name: "changed" for name in p.g.NATIVE_SHA})


def test_profile_kernel_visibility_and_cpu_binding(source_seam):
    value = prefill_profile()
    assert p.validate_profile(value) is value
    assert p.c.validate_profile(value) is value
    assert value["kernel"]["local256_prefill"]["source_sha256"] == p.g.file_sha256(p.__file__)
    assert value["cpu_execution_overlay"]["source_sha256"] == p.CPU_SHA
    assert value["implementation"] == cpu_profile()["implementation"]
    assert value["engine_settings"] == cpu_profile()["engine_settings"]
    assert value["kernel_execution_overlay"]["qualified_C3_profile_sha256"] == p.g.digest(cpu_profile())


@pytest.mark.parametrize("drift", ["local_tile", "CPU_source", "native_source", "memory"])
def test_scope_and_source_drift_rejected(source_seam, drift):
    value = prefill_profile()
    if drift == "local_tile": value["kernel"]["local256_prefill"]["prefill_block_m"] = 64
    elif drift == "CPU_source": value["cpu_execution_overlay"]["source_sha256"] = "wrong"
    elif drift == "native_source": value["kernel_execution_overlay"]["artifacts"]["native_source_sha256"] = {}
    else: value["engine_settings"]["cpu_offload_gb"] = 28
    with pytest.raises(ValueError): p.validate_profile(value)


def test_seal_uses_transient_cpu_profile_and_preserves_frozen_file(source_seam, tmp_path, monkeypatch):
    calls = []
    def seal_cpu(args):
        assert (args.placement, args.gc_cadence, args.range_lookup, args.prefill_block_m) == ("baseline", 8, "binary", 32)
        calls.append(Path(args.output))
        p.g.seal(args.output, cpu_profile())
    monkeypatch.setattr(p.c, "seal_profile", seal_cpu)
    target = tmp_path / "new-profile.json"
    args = Namespace(output=str(target))
    assert p.seal_profile(args)["GPU_calls"] == 0
    assert not calls[0].exists()
    before = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"): p.seal_profile(args)
    assert target.read_bytes() == before and len(calls) == 1


def dispatchers(a):
    fn = p._compile_launcher(a["base_source"], {}, "<gemma-native-query-tiling>")
    return SimpleNamespace(unified_attention=fn, current_platform=Platform()), SimpleNamespace(unified_attention=fn)


def test_binding_verifies_actual_base_and_restores_on_failure():
    a = artifacts()
    native, backend = dispatchers(a)
    original = native.unified_attention
    class FailingBackend:
        def __init__(self): object.__setattr__(self, "unified_attention", original)
        def __setattr__(self, name, value):
            if value is not original: raise RuntimeError("injected bind failure")
            object.__setattr__(self, name, value)
    with pytest.raises(RuntimeError, match="bind failure"):
        p.bind_dispatch(native, FailingBackend(), a, "fixture.py")
    assert native.unified_attention is original
    bad = p._compile_launcher(a["base_source"].replace("TILE_SIZE_PREFILL = 32", "TILE_SIZE_PREFILL = 64"), {}, "<gemma-native-query-tiling>")
    native.unified_attention = backend.unified_attention = bad
    with pytest.raises(ValueError, match="Actual dispatcher"):
        p.bind_dispatch(native, backend, a, "fixture.py")


def base_receipt(config, a):
    return {"configuration": config, "observed_native_sha256": p.g.NATIVE_SHA,
        "derived_helpers_sha256": a["receipt"]["binary_helper_sha256"],
        "derived_launcher_sha256": a["receipt"]["base_launcher_sha256"], "changed_kernels": []}


def test_RPC_records_capability_and_both_bound_references(source_seam, tmp_path, monkeypatch):
    a = source_seam
    native, backend = dispatchers(a)
    calls = []
    def install(worker, config, output):
        calls.append(config)
        (Path(output) / "native_range_helpers.py").write_text(a["helper_source"])
        return base_receipt(config, a)
    monkeypatch.setattr(p, "_INSTALL", install)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (12, 0))))
    monkeypatch.setattr(p.importlib, "import_module", lambda name: backend if name.endswith("triton_attn") else native)
    config = {**p.base_kernel(), "local256_prefill": p.local_parameters()}
    receipt = p.install_prefill(None, config, str(tmp_path))
    assert len(calls) == 1 and native.unified_attention is backend.unified_attention
    assert native.unified_attention(256, 100, 2, Platform()) == (32, 16, 32, None, None)
    assert receipt["local256_prefill"]["actual_device_capability"] == [12, 0]
    assert receipt["local256_prefill"]["B200_preset_enabled"] is False
    assert p.g.file_sha256(tmp_path / p.SOURCE_FILE) == a["receipt"]["derived_launcher_sha256"]


def test_unsupported_device_fails_before_installer(source_seam, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda: (10, 0))))
    monkeypatch.setattr(p, "_INSTALL", lambda *args: pytest.fail("Unsupported GPU must not install anything"))
    with pytest.raises(ValueError, match="SM120; no fallback"):
        p.install_prefill(None, {**p.base_kernel(), "local256_prefill": p.local_parameters()}, "unused")


def test_run_only_overrides_installer_and_restores_after_delegated_failure(source_seam, tmp_path, monkeypatch):
    target = tmp_path / "profile.json"
    p.g.seal(target, prefill_profile())
    before = dict(vars(p.g))
    def delegated(args):
        assert {k for k, v in before.items() if getattr(p.g, k) is not v} == {"_install_kernel"}
        assert p.g._install_kernel is p.install_prefill
        raise RuntimeError("owned runner failure")
    monkeypatch.setattr(p.c, "run", delegated)
    with pytest.raises(RuntimeError, match="owned runner failure"):
        p.run(Namespace(profile=str(target)))
    assert p.g._install_kernel is p._INSTALL


def make_run(tmp_path, index, new=True, n=20, host="fourier", device="0"):
    root = make_cpu_run(tmp_path, index, n=n, host=host, device=device)
    if not new: return root
    value, a = prefill_profile(n), artifacts()
    identity = p.g.read(root / "identity.json")
    identity.update(profile=value, profile_sha256=p.g.digest(value))
    (root / "identity.json").write_text(json.dumps(identity))
    cpu = p.g.read(root / "cpu-copy.json")
    cpu["profile_sha256"] = p.g.digest(value)
    (root / "cpu-copy.json").write_text(json.dumps(cpu))
    receipt = base_receipt(value["kernel"], a)
    receipt["local256_prefill"] = {"configuration": value["kernel"]["local256_prefill"],
        "actual_device_capability": [12, 0], "artifacts": a["receipt"], "derived_source_file": p.SOURCE_FILE,
        "derived_source_file_sha256": a["receipt"]["derived_launcher_sha256"], "B200_preset_enabled": False,
        "no_silent_fallback": True, "base_dispatch_verified": True, "both_dispatch_references_bound": True}
    load = p.g.read(root / "load.json")
    load["kernel"] = [receipt]
    (root / "load.json").write_text(json.dumps(load))
    (root / p.SOURCE_FILE).write_text(a["derived_source"])
    (root / "native_range_helpers.py").write_text(a["helper_source"])
    return root


@pytest.mark.parametrize("change,status", [("none", "MATCH"), ("tokens", "PROFILE_OUTPUT_DIFFERENCE"), ("input", "INVALID_EVIDENCE")])
def test_candidate_preserves_kernel_difference_and_input_invalid(tmp_path, change, status):
    old, new = make_run(tmp_path, 0, new=False), make_run(tmp_path, 1)
    rows = [json.loads(s) for s in (new / "raw.jsonl").read_text().splitlines()]
    if change == "tokens": rows[0]["output_token_ids"] = [11, 99]
    if change == "input": rows[0]["actual_input_token_ids"] = [1, 99]
    (new / "raw.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = p.compare_runs([old, new])
    assert result["status"] == status and not result["production_authorized_by_this_command"]


def test_changed_GPU_source_receipt_is_rejected(tmp_path):
    old, new = make_run(tmp_path, 0, new=False), make_run(tmp_path, 1)
    (new / p.SOURCE_FILE).write_text("changed source")
    with pytest.raises(ValueError, match="source evidence"):
        p.compare_runs([old, new])


def test_consistency_keeps_six_cold_and_kernel_signature_checks(tmp_path):
    paths = [make_run(tmp_path, i, n=48, host="fourier" if i < 4 else "radon", device=str(i // 2) if i < 4 else "0") for i in range(6)]
    assert p.compare_runs(paths, "consistency")["six_observation_consistency_pass"]
    identity = p.g.read(paths[-1] / "identity.json")
    identity["engine_id"] = "engine-0"
    (paths[-1] / "identity.json").write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="Independent cold"):
        p.compare_runs(paths, "consistency")


def test_import_is_CPU_only():
    result = subprocess.run([sys.executable, "-B", "-c", "import classic_precision.gemma_prefill,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
