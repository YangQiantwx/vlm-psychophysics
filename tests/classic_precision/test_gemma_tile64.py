"""M64 delta only: no GPU, weights, processor preparation, or old test reruns."""
from argparse import Namespace
from copy import deepcopy
import ast
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from classic_precision import gemma_tile64 as t
import test_gemma_prefill as old


LAUNCHER = old.LAUNCHER.replace("    return BLOCK_M,", """    launch_kwargs: dict[str, int] = {}
    if launch_num_warps is not None:
        launch_kwargs['num_warps'] = launch_num_warps
    if launch_num_stages is not None:
        launch_kwargs['num_stages'] = launch_num_stages
    kernel_unified_attention[(1,)](HEAD_SIZE=head_size, BLOCK_M=BLOCK_M,
        TILE_SIZE=TILE_SIZE_PREFILL, num_warps=launch_num_warps, num_stages=launch_num_stages)
    return BLOCK_M,""")
API = {"version": "fixture", "source_sha256": {"triton/runtime/jit.py": "jit", "triton/compiler/compiler.py": "compiler"},
    "JITFunction_run_returns_kernel": True, "inspection": "fixture"}


@pytest.fixture(autouse=True)
def sources(monkeypatch):
    monkeypatch.setattr(old, "LAUNCHER", LAUNCHER)
    base = old.artifacts()
    monkeypatch.setattr(t, "_ARTIFACTS", lambda sources=None: base)
    monkeypatch.setattr(t.p, "kernel_artifacts", lambda sources=None: base)
    monkeypatch.setattr(t, "K4_LAUNCHER_SHA", t.p.sha(base["derived_source"]))
    monkeypatch.setattr(t, "DERIVED_SHA", t.p.sha(t.derive_launcher(base["derived_source"])))
    monkeypatch.setattr(t, "WARP8_DERIVED_SHA", t.p.sha(t.derive_launcher(base["derived_source"], 8)))
    monkeypatch.setattr(t, "resource_api", lambda: deepcopy(API))
    monkeypatch.setattr(t.c, "installed_processor_source", lambda: t.c.processor_candidate(old.SDK))
    monkeypatch.setattr(t, "_RESOURCE", None)
    return base


def profile(n=20, warps=None):
    original = old.cpu_profile(n=n)
    value = deepcopy(original)
    with t.specialization(warps):
        value["kernel"]["local256_prefill"] = t.local_parameters()
        value["kernel_execution_overlay"] = t.overlay_record(original, t.kernel_artifacts()["receipt"])
        t.p._rebind_cpu(value)
    return value


def compiled(shared=49152, spills=0, warps=4, stages=3):
    return SimpleNamespace(metadata=SimpleNamespace(shared=shared, num_warps=warps, num_stages=stages),
        _run=SimpleNamespace(shared=shared), n_regs=96, n_spills=spills)


class Kernel:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.calls = result, error, []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid, kwargs))
            if self.error: raise self.error
            return self.result
        return launch


def bind(tmp_path, a, kernel, warps=None):
    t._RESOURCE = t.Resources(tmp_path, API, 99 * 1024, warps)
    before = t.p._compile_launcher(a["base_source"], {"kernel_unified_attention": kernel}, "<gemma-native-query-tiling>")
    native = SimpleNamespace(unified_attention=before, current_platform=old.Platform())
    backend = SimpleNamespace(unified_attention=before)
    with t.specialization(warps):
        t.p.bind_dispatch(native, backend, t.kernel_artifacts(), str(tmp_path / t.SOURCE_FILE))
    return native, backend, before


@pytest.mark.parametrize("head,length,ratio,m", [
    (256, 100, 2, 64), (256, 100, 16, 64), (512, 100, 2, 32),
    (256, 1, 2, 16), (512, 1, 2, 16), (128, 100, 2, 16), (256, 100, 32, 32)])
def test_M64_only_local_prefill_and_same_single_launch(tmp_path, sources, head, length, ratio, m):
    kernel = Kernel(compiled())
    native, backend, before = bind(tmp_path, sources, kernel)
    actual = native.unified_attention(head, length, ratio, old.Platform())
    assert native.unified_attention is backend.unified_attention
    assert actual == (m, m // ratio, 32, None, None)
    assert len(kernel.calls) == 1
    assert kernel.calls[0][1] == {"HEAD_SIZE": head, "BLOCK_M": m, "TILE_SIZE": 32, "num_warps": None, "num_stages": None}
    if head != 256 or length == 1 or ratio > 16:
        assert not (tmp_path / t.RESOURCE_FILE).exists()
        assert before(head, length, ratio, old.Platform()) == actual
    else:
        rows = [json.loads(line) for line in (tmp_path / t.RESOURCE_FILE).read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["n_regs"] == 96 and rows[0]["n_spills"] == 0


def test_AST_preserves_original_launch_arguments_and_refuses_bad_anchors(sources):
    original = sources["derived_source"]
    derived = t.derive_launcher(original)
    def launches(source):
        return [ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript)
            and ast.unparse(n.func.value) == "kernel_unified_attention"]
    assert launches(derived) == launches(original)
    with pytest.raises(ValueError, match="M32 anchor"):
        t.derive_launcher(original.replace("head_size == 256", "head_size == 384"))
    with pytest.raises(ValueError, match="M32 anchor"): t.derive_launcher(derived)
    with pytest.raises(ValueError, match="launch anchor"):
        t.derive_launcher(original.replace("kernel_unified_attention", "different_kernel"))


def test_binding_rejects_actual_dispatcher_drift_and_restores_failure(tmp_path, sources):
    fn = t.p._compile_launcher(sources["base_source"], {}, "<gemma-native-query-tiling>")
    native = SimpleNamespace(unified_attention=fn)
    class Failing:
        def __init__(self): object.__setattr__(self, "unified_attention", fn)
        def __setattr__(self, key, value):
            if value is not fn: raise RuntimeError("bind failed")
            object.__setattr__(self, key, value)
    with t.specialization(), pytest.raises(RuntimeError, match="bind failed"):
        t.p.bind_dispatch(native, Failing(), t.kernel_artifacts(), "derived.py")
    assert native.unified_attention is fn
    bad = t.p._compile_launcher(sources["base_source"].replace("TILE_SIZE_PREFILL = 32", "TILE_SIZE_PREFILL = 64"), {}, "<gemma-native-query-tiling>")
    native.unified_attention = bad
    with t.specialization(), pytest.raises(ValueError, match="Actual dispatcher"):
        t.p.bind_dispatch(native, SimpleNamespace(unified_attention=bad), t.kernel_artifacts(), "derived.py")


def test_resources_once_per_returned_object_no_extra_launch(tmp_path, sources):
    kernel = Kernel(compiled())
    native, _, _ = bind(tmp_path, sources, kernel)
    for _ in range(3): native.unified_attention(256, 100, 2, old.Platform())
    assert len(kernel.calls) == 3
    assert len((tmp_path / t.RESOURCE_FILE).read_text().splitlines()) == 1


@pytest.mark.parametrize("kind", ["spill", "shared", "missing", "failed"])
def test_resources_preserve_unknowns_and_record_failures(tmp_path, sources, kind):
    result = compiled(spills=1) if kind == "spill" else (compiled(shared=200000) if kind == "shared" else None)
    kernel = Kernel(result, RuntimeError("out of resources") if kind == "failed" else None)
    native, _, _ = bind(tmp_path, sources, kernel)
    if kind == "missing": native.unified_attention(256, 100, 2, old.Platform())
    else:
        with pytest.raises((ValueError, RuntimeError)):
            native.unified_attention(256, 100, 2, old.Platform())
    assert len(kernel.calls) == 1
    row = json.loads((tmp_path / t.RESOURCE_FILE).read_text().splitlines()[0])
    expected = "PARTIAL" if kind == "missing" else ("LAUNCH_FAILED" if kind == "failed" else "RESOURCE_REJECTED")
    assert row["status"] == expected
    if kind in ("missing", "failed"):
        assert row["shared_bytes"] is row["n_regs"] is row["n_spills"] is None


@pytest.mark.parametrize("drift", ["module", "tile", "GPU_source", "CPU_source", "placement"])
def test_profile_kernel_and_frozen_scope_drift(drift):
    value = profile()
    with t.specialization(): assert t.p.validate_profile(value) is value
    assert value["implementation"] == old.cpu_profile()["implementation"]
    assert value["engine_settings"] == old.cpu_profile()["engine_settings"]
    if drift == "module": value["kernel"]["local256_prefill"]["source_sha256"] = "bad"
    elif drift == "tile": value["kernel"]["local256_prefill"]["prefill_block_m"] = 32
    elif drift == "GPU_source": value["kernel"]["local256_prefill"]["derived_launcher_sha256"] = "bad"
    elif drift == "CPU_source": value["cpu_execution_overlay"]["source_sha256"] = "bad"
    else: value["engine_settings"]["cpu_offload_gb"] = 28
    with t.specialization(), pytest.raises(ValueError): t.p.validate_profile(value)


def test_K4_module_and_launcher_source_drift_rejected(monkeypatch, sources):
    monkeypatch.setattr(t, "K4_LAUNCHER_SHA", "bad")
    with pytest.raises(ValueError, match="K4 launcher"): t.kernel_artifacts()
    monkeypatch.setattr(t, "PREFILL_SHA", "bad")
    with pytest.raises(ValueError, match="K4 source"): t.validate_sources()


def test_seal_temp_profile_no_overwrite_and_run_cleanup(tmp_path, monkeypatch):
    temporary = []
    def seal_cpu(args):
        temporary.append(Path(args.output)); t.g.seal(args.output, old.cpu_profile())
    monkeypatch.setattr(t.c, "seal_profile", seal_cpu)
    target = tmp_path / "profile.json"
    assert t.seal_profile(Namespace(output=str(target)))["GPU_calls"] == 0
    assert not temporary[0].exists()
    original = target.read_bytes()
    with pytest.raises(ValueError, match="overwrite"): t.seal_profile(Namespace(output=str(target)))
    assert target.read_bytes() == original
    p_before = {k: getattr(t.p, k) for k in ("local_parameters", "kernel_artifacts", "install_prefill", "SOURCE_FILE")}
    def run(args):
        assert t.g._install_kernel is t.install_tile64
        assert t.c._RUN is t.g.run and t.c._RUN_GC_LINE == 716
        raise RuntimeError("owned run failure")
    monkeypatch.setattr(t.c, "run", run)
    with pytest.raises(RuntimeError, match="owned run failure"): t.run(Namespace(profile=str(target)))
    assert t.g._install_kernel is t.p._INSTALL
    assert {k: getattr(t.p, k) for k in p_before} == p_before


@pytest.mark.parametrize("warps", [None, 8])
def test_RPC_uses_real_dispatch_binding_and_resource_source_receipt(tmp_path, sources, monkeypatch, warps):
    kernel = Kernel(compiled(warps=warps or 4))
    fn = t.p._compile_launcher(sources["base_source"], {"kernel_unified_attention": kernel}, "<gemma-native-query-tiling>")
    native = SimpleNamespace(unified_attention=fn, current_platform=old.Platform())
    backend = SimpleNamespace(unified_attention=fn)
    monkeypatch.setattr(t.p, "_INSTALL", lambda worker, config, output: old.base_receipt(config, sources))
    monkeypatch.setattr(t.p.importlib, "import_module", lambda name: backend if name.endswith("triton_attn") else native)
    cuda = SimpleNamespace(get_device_capability=lambda: (12, 0), current_device=lambda: 0,
        get_device_properties=lambda device: SimpleNamespace(shared_memory_per_block_optin=99 * 1024))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    value = profile(warps=warps)
    receipt = t.install_tile64(None, value["kernel"], str(tmp_path))
    assert receipt["local256_prefill"]["resource_receipt"]["no_extra_kernel_call"]
    assert native.unified_attention is backend.unified_attention
    native.unified_attention(256, 100, 2, old.Platform())
    assert len(kernel.calls) == 1
    assert t._LOCAL_NUM_WARPS is None
    assert kernel.calls[0][1]["num_warps"] == warps
    assert t.verify_resources(tmp_path, value, {"kernel": [receipt]}) == API


def make_run(tmp_path, index, n=20, new=True, host="fourier", device="0", warps=None):
    root = old.make_run(tmp_path, index, n=n, host=host, device=device)
    if not new: return root
    value = profile(n, warps)
    with t.specialization(warps): a = t.kernel_artifacts()
    identity = t.g.read(root / "identity.json")
    identity.update(profile=value, profile_sha256=t.g.digest(value))
    (root / "identity.json").write_text(json.dumps(identity))
    cpu = t.g.read(root / "cpu-copy.json"); cpu["profile_sha256"] = t.g.digest(value)
    (root / "cpu-copy.json").write_text(json.dumps(cpu))
    load = t.g.read(root / "load.json")
    receipt = old.base_receipt(value["kernel"], a)
    receipt["local256_prefill"] = {"configuration": value["kernel"]["local256_prefill"],
        "actual_device_capability": [12, 0], "artifacts": a["receipt"], "derived_source_file": t.SOURCE_FILE,
        "derived_source_file_sha256": t.derived_sha(warps), "B200_preset_enabled": False,
        "no_silent_fallback": True, "base_dispatch_verified": True, "both_dispatch_references_bound": True,
        "resource_receipt": {"file": t.RESOURCE_FILE, "triton_api": API, "shared_limit_bytes": 99 * 1024, "local_num_warps": warps,
            "capture_started": True, "no_extra_kernel_call": True}}
    load["kernel"] = [receipt]; (root / "load.json").write_text(json.dumps(load))
    (root / t.SOURCE_FILE).write_text(a["derived_source"])
    t.Resources(root, API, 99 * 1024, warps).record(compiled(warps=warps or 4), 64, 32, warps, None)
    return root


@pytest.mark.parametrize("change,status", [("none", "MATCH"), ("tokens", "PROFILE_OUTPUT_DIFFERENCE"), ("input", "INVALID_EVIDENCE")])
def test_K4_M32_vs_M64_comparison(tmp_path, change, status):
    before, after = make_run(tmp_path, 0, new=False), make_run(tmp_path, 1)
    rows = [json.loads(line) for line in (after / "raw.jsonl").read_text().splitlines()]
    if change == "tokens": rows[0]["output_token_ids"] = [11, 99]
    if change == "input": rows[0]["actual_input_token_ids"] = [1, 99]
    (after / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert t.compare_runs([before, after])["status"] == status


def test_consistency_six_cold_same_source_and_resources(tmp_path):
    paths = [make_run(tmp_path, i, n=48, host="fourier" if i < 4 else "radon",
        device=str(i // 2) if i < 4 else "0", warps=8) for i in range(6)]
    assert t.compare_runs(paths, "consistency")["six_observation_consistency_pass"]
    (paths[-1] / t.SOURCE_FILE).write_text("source drift")
    with pytest.raises(ValueError, match="source evidence"): t.compare_runs(paths, "consistency")


def test_missing_or_rejected_resources_refuse_acceptance(tmp_path):
    before, after = make_run(tmp_path, 0, new=False), make_run(tmp_path, 1)
    path = after / t.RESOURCE_FILE
    row = json.loads(path.read_text()); row["n_spills"] = 2
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="compiled resource"): t.compare_runs([before, after])
    path.write_text("")
    with pytest.raises(ValueError, match="actually invoked"): t.compare_runs([before, after])


def test_import_and_help_are_CPU_only():
    result = subprocess.run([sys.executable, "-B", "-c", "import classic_precision.gemma_tile64,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("head,length,ratio,warps", [
    (256, 100, 2, 8), (256, 1, 2, None), (512, 100, 2, None),
    (128, 100, 2, None), (256, 100, 32, None)])
def test_eight_warps_only_local256_prefill_after_native_initialization(tmp_path, sources, head, length, ratio, warps):
    kernel = Kernel(compiled(warps=8))
    native, _, _ = bind(tmp_path, sources, kernel, 8)
    result = native.unified_attention(head, length, ratio, old.Platform())
    assert len(kernel.calls) == 1
    assert result[2:] == (32, warps, None)
    assert kernel.calls[0][1]["num_warps"] == warps and kernel.calls[0][1]["num_stages"] is None
    if warps == 8:
        row = json.loads((tmp_path / t.RESOURCE_FILE).read_text())
        assert row["compiled_num_warps"] == 8 and row["compiled_num_stages"] == 3
        assert row["derived_launcher_sha256"] == t.WARP8_DERIVED_SHA
    else:
        assert not (tmp_path / t.RESOURCE_FILE).exists()


def test_eight_warp_anchor_cannot_precede_later_native_reset(sources):
    source = sources["derived_source"]
    changed = source.replace("launch_kwargs: dict[str, int] = {}", "launch_kwargs: dict[str, int] = {}\n    launch_num_warps = None")
    with pytest.raises(ValueError, match="initialization order"): t.derive_launcher(changed, 8)
    with pytest.raises(ValueError, match="kwargs anchor"):
        t.derive_launcher(source.replace("launch_kwargs:", "different_kwargs:"), 8)
    native, explicit = ast.parse(t.derive_launcher(source)), ast.parse(t.derive_launcher(source, 8))
    added = [n for n in explicit.body[0].body if isinstance(n, ast.If)
        and len(n.body) == 1 and ast.unparse(n.body[0]) == "launch_num_warps = 8"]
    assert len(added) == 1
    explicit.body[0].body.remove(added[0])
    assert ast.dump(native, include_attributes=False) == ast.dump(explicit, include_attributes=False)


@pytest.mark.parametrize("warps,stages,launch_warps", [(4, 3, 8), (8, 2, 8), (8, 3, None)])
def test_resource_receipt_rejects_ignored_warps_or_changed_stages(tmp_path, warps, stages, launch_warps):
    recorder = t.Resources(tmp_path, API, 99 * 1024, 8)
    with pytest.raises(ValueError, match="resource rejection"):
        recorder.record(compiled(warps=warps, stages=stages), 64, 32, launch_warps, None)
    row = json.loads((tmp_path / t.RESOURCE_FILE).read_text())
    assert row["status"] == "RESOURCE_REJECTED" and row["violations"]


def test_sealed_warp_parameter_binding_and_run_restoration(tmp_path, monkeypatch):
    monkeypatch.setattr(t.c, "seal_profile", lambda args: t.g.seal(args.output, old.cpu_profile()))
    path = tmp_path / "warp8-profile.json"
    t.seal_profile(Namespace(output=str(path), local_num_warps=8))
    value = t.g.read(path)
    assert value["kernel"]["local256_prefill"]["local_num_warps"] == 8
    assert value["kernel"]["local256_prefill"]["derived_launcher_sha256"] == t.WARP8_DERIVED_SHA
    assert value["engine_settings"] == old.cpu_profile()["engine_settings"]
    with t.specialization(8): assert t.p.validate_profile(value) is value
    with t.specialization(), pytest.raises(ValueError): t.p.validate_profile(value)
    def delegated(args):
        assert t._LOCAL_NUM_WARPS == 8 and t.g._install_kernel is t.install_tile64
        raise RuntimeError("controlled failure")
    monkeypatch.setattr(t.c, "run", delegated)
    with pytest.raises(RuntimeError, match="controlled failure"): t.run(Namespace(profile=str(path)))
    assert t._LOCAL_NUM_WARPS is None and t.g._install_kernel is t.p._INSTALL


@pytest.mark.parametrize("control_warps,change,status", [
    (None, "tokens", "PROFILE_OUTPUT_DIFFERENCE"), (8, "tokens", "DIFFERENT"), (None, "input", "INVALID_EVIDENCE")])
def test_warp_profiles_keep_numerical_vs_same_runtime_comparison(tmp_path, control_warps, change, status):
    before, after = make_run(tmp_path, 0, warps=control_warps), make_run(tmp_path, 1, warps=8)
    rows = [json.loads(line) for line in (after / "raw.jsonl").read_text().splitlines()]
    rows[0]["output_token_ids" if change == "tokens" else "actual_input_token_ids"] = [7, 99]
    (after / "raw.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    assert t.compare_runs([before, after])["status"] == status


def test_six_cold_cannot_mix_native_and_eight_warp_profiles(tmp_path):
    paths = [make_run(tmp_path, i, n=48, host="fourier" if i < 4 else "radon",
        device=str(i // 2) if i < 4 else "0", warps=8 if i < 5 else None) for i in range(6)]
    with pytest.raises(ValueError, match="identical placement, engine and kernel"):
        t.compare_runs(paths, "consistency")


@pytest.mark.parametrize("args,selected", [([], None), (["--local-num-warps", "8"], 8)])
def test_CLI_seal_threads_explicit_parameter(monkeypatch, args, selected):
    seen = []
    monkeypatch.setattr(t, "seal_profile", lambda parsed: seen.append(parsed.local_num_warps) or {"status": "CPU_FIXTURE"})
    monkeypatch.setattr(sys, "argv", ["gemma_tile64", "seal-profile", "--plan", "p", "--runtime", "r",
        "--name", "n", "--output", "o", *args])
    t.main()
    assert seen == [selected]
