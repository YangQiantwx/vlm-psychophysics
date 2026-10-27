"""One SM120 launcher change: local head256 prefill query tile 16 -> 32.

The frozen binary helpers, head512 M32, CPU-copy/GC8 runner, timers, engine and
lease lifecycle remain delegated. This module never edits installed SDK files.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from copy import copy, deepcopy
import hashlib
import importlib
import importlib.metadata
import json
import marshal
from pathlib import Path
import tempfile

from . import gemma_cpu_copy as c

g = c.g
SCHEMA = "gemma-local256-prefill-v1"
CPU_SHA = "e3b93f048645d6145f362430513f99b60531c8e585e598027d2abb11ac2900e3"
SOURCE_FILE = "head256_prefill_launcher.py"
_INSTALL = g._install_kernel
_GLOBAL_TEST = "head_size == 512 and max_seqlen_q > 1 and num_queries_per_kv <= 16"
_LOCAL_TEST = "head_size == 256 and max_seqlen_q > 1 and num_queries_per_kv <= 16"


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate_sources():
    c.validate_sources()
    g.require(g.file_sha256(c.__file__) == CPU_SHA, "Frozen CPU-copy source differs")


def base_kernel():
    return {"range_lookup": "binary", "prefill_block_m": 32, "expected_native_sha256": g.NATIVE_SHA}


def local_parameters():
    return {"module": "classic_precision.gemma_prefill", "source_sha256": g.file_sha256(__file__),
        "head_size": 256, "prefill_block_m": 32, "max_num_queries_per_kv": 16,
        "prefill_only": True, "device_capability": [12, 0],
        "kv_tile_prefill": 32, "warps_and_stages": "unchanged native defaults",
        "head512_prefill_block_m": 32, "binary_ranges": "unchanged"}


def derive_launcher(base_source):
    """Insert one query-tile branch before native BLOCK_Q is calculated."""
    tree = ast.parse(base_source)
    g.require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        and tree.body[0].name == "unified_attention", "Base launcher interface differs")
    fn = tree.body[0]
    wanted = ast.dump(ast.parse(_GLOBAL_TEST, mode="eval").body, include_attributes=False)
    hits = [i for i, node in enumerate(fn.body) if isinstance(node, ast.If)
        and ast.dump(node.test, include_attributes=False) == wanted]
    g.require(len(hits) == 1, "Qualified head512 M32 anchor differs")
    i = hits[0]
    local_test = ast.dump(ast.parse(_LOCAL_TEST, mode="eval").body, include_attributes=False)
    g.require(not any(isinstance(n, ast.If) and ast.dump(n.test, include_attributes=False) == local_test
                      for n in ast.walk(tree)), "Local head256 override already present")
    g.require(ast.unparse(fn.body[i].body[0]) == "BLOCK_M = 32" and len(fn.body[i].body) == 1
        and not fn.body[i].orelse and i + 1 < len(fn.body)
        and ast.unparse(fn.body[i + 1]) == "BLOCK_Q = BLOCK_M // num_queries_per_kv",
        "Native query-tile assignment order differs")
    fn.body.insert(i + 1, ast.parse(f"if {_LOCAL_TEST}:\n    BLOCK_M = 32\n").body[0])
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def kernel_artifacts(sources=None):
    if sources is None:
        distribution = importlib.metadata.distribution("vllm")
        sources = {name: Path(distribution.locate_file(name.replace(".", "/") + ".py")).read_text()
                   for name in g.NATIVE_SHA}
    observed = {name: sha(source) for name, source in sources.items()}
    g.require(observed == g.NATIVE_SHA, "Installed native kernel source differs")
    helper, base = g.candidate_sources(sources["vllm.v1.attention.ops.triton_attention_helpers"],
        sources["vllm.v1.attention.ops.triton_unified_attention"], True, 32)
    derived = derive_launcher(base)
    return {"helper_source": helper, "base_source": base, "derived_source": derived,
        "receipt": {"native_source_sha256": observed, "binary_helper_sha256": sha(helper),
            "base_launcher_sha256": sha(base), "derived_launcher_sha256": sha(derived),
            "only_added_statement": f"if {_LOCAL_TEST}: BLOCK_M = 32"}}


def _qualified_cpu(profile, installed_sdk):
    c.validate_profile(profile, installed_sdk=installed_sdk)
    overlay = profile["cpu_execution_overlay"]
    g.require(overlay["placement"] == "baseline" and overlay["gc_cadence"] == 8
        and "execution_overlay" not in profile and profile["variant"] == "kernel",
        "This candidate keeps C3 copy8/.90, CPU-copy and GC8; memory placement cannot change")


def _rebind_cpu(profile):
    overlay = profile.pop("cpu_execution_overlay")
    profile["cpu_execution_overlay"] = c.overlay_record(profile, overlay["placement"],
        overlay["gc_cadence"], overlay["processor_source"])


def kernel_overlay_record(original_cpu_profile, artifacts):
    return {"schema": SCHEMA, "source_sha256": g.file_sha256(__file__), "CPU_copy_source_sha256": CPU_SHA,
        "qualified_C3_profile_sha256": g.digest(original_cpu_profile), "artifacts": artifacts,
        "scope": "Only head256 multi-query prefill BLOCK_M; native KV tile/warps/stages/math/masks unchanged"}


def validate_profile(profile, installed_sdk=True):
    validate_sources()
    _qualified_cpu(profile, installed_sdk)
    g.require(profile["kernel"] == {**base_kernel(), "local256_prefill": local_parameters()},
        "Kernel configuration/source scope differs")
    overlay = profile.get("kernel_execution_overlay", {})
    artifacts = kernel_artifacts()["receipt"] if installed_sdk else overlay.get("artifacts", {})
    g.require(artifacts.get("native_source_sha256") == g.NATIVE_SHA, "Recorded native kernel source differs")
    original = deepcopy(profile)
    original.pop("kernel_execution_overlay", None)
    original["kernel"].pop("local256_prefill")
    _rebind_cpu(original)
    g.require(overlay == kernel_overlay_record(original, artifacts), "Kernel overlay/baseline binding differs")
    return profile


def _compile_launcher(source, namespace, filename):
    target = dict(namespace)
    exec(compile(source, filename, "exec"), target)
    return target["unified_attention"]


def bind_dispatch(native, backend, artifacts, source_path):
    """Verify the actual C3 dispatch before deriving and binding both references."""
    current, backend_current = native.unified_attention, backend.unified_attention
    g.require(current is backend_current, "Native/backend dispatcher references diverged")
    expected = _compile_launcher(artifacts["base_source"], current.__globals__, "<gemma-native-query-tiling>")
    g.require(marshal.dumps(current.__code__) == marshal.dumps(expected.__code__), "Actual dispatcher is not the qualified head512 M32 launcher")
    candidate = _compile_launcher(artifacts["derived_source"], current.__globals__, str(source_path))
    try:
        native.unified_attention = candidate
        backend.unified_attention = candidate
        g.require(native.unified_attention is backend.unified_attention is candidate, "New dispatcher binding failed")
    except BaseException:
        native.unified_attention, backend.unified_attention = current, backend_current
        raise
    return {"base_dispatch_verified": True, "both_dispatch_references_bound": True}


def install_prefill(worker, config, output):
    """RPC in the owned GPU worker; source checks precede the first model call."""
    import torch
    validate_sources()
    g.require(config == {**base_kernel(), "local256_prefill": local_parameters()}, "RPC kernel parameters/source differ")
    capability = list(torch.cuda.get_device_capability())
    g.require(capability == [12, 0], "This single-variable candidate requires SM120; no fallback")
    artifacts = kernel_artifacts()
    receipt = _INSTALL(worker, config, output)
    g.require(receipt["observed_native_sha256"] == artifacts["receipt"]["native_source_sha256"]
        and receipt["derived_launcher_sha256"] == artifacts["receipt"]["base_launcher_sha256"]
        and receipt["derived_helpers_sha256"] == artifacts["receipt"]["binary_helper_sha256"],
        "Qualified base installer receipt differs")
    native = importlib.import_module("vllm.v1.attention.ops.triton_unified_attention")
    backend = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    g.require(native.current_platform.is_device_capability_family(120)
        and not native.current_platform.is_device_capability_family(100), "Platform must retain the non-B200 KV-tile/warp/stage path")
    target = Path(output) / SOURCE_FILE
    g.require(not target.exists(), "Derived launcher evidence already exists")
    target.write_text(artifacts["derived_source"])
    binding = bind_dispatch(native, backend, artifacts, target)
    receipt["local256_prefill"] = {"configuration": config["local256_prefill"],
        "actual_device_capability": capability, "artifacts": artifacts["receipt"],
        "derived_source_file": SOURCE_FILE, "derived_source_file_sha256": g.file_sha256(target),
        "B200_preset_enabled": False, "no_silent_fallback": True, **binding}
    receipt["changed_kernels"].append("head256 prefill BLOCK_M=32 only; native KV tile32/warps/stages unchanged")
    return receipt


@contextmanager
def installed_overlay():
    validate_sources()
    g.require(g._install_kernel is _INSTALL, "Another GPU installer overlay is active")
    g._install_kernel = install_prefill
    try:
        yield
    finally:
        g._install_kernel = _INSTALL


def seal_profile(args):
    validate_sources()
    g.require(not Path(args.output).exists(), "Refusing to overwrite sealed evidence")
    artifacts = kernel_artifacts()["receipt"]
    with tempfile.TemporaryDirectory(prefix="gemma-prefill-profile-") as temporary:
        original_args = copy(args)
        original_args.output = str(Path(temporary) / "cpu-profile.json")
        original_args.placement, original_args.gc_cadence = "baseline", 8
        original_args.variant, original_args.range_lookup, original_args.prefill_block_m = "kernel", "binary", 32
        c.seal_profile(original_args)
        original = g.read(original_args.output)
    _qualified_cpu(original, installed_sdk=True)
    g.require(original["kernel"] == base_kernel(), "C3 kernel changed")
    profile = deepcopy(original)
    profile["kernel"]["local256_prefill"] = local_parameters()
    profile["kernel_execution_overlay"] = kernel_overlay_record(original, artifacts)
    _rebind_cpu(profile)
    validate_profile(profile)
    g.seal(args.output, profile)
    return {"status": "SEALED", "profile_sha256": g.digest(profile), "kernel": profile["kernel"], "GPU_calls": 0}


def run(args):
    validate_profile(g.read(args.profile))
    with installed_overlay():
        return c.run(args)


def _verify_kernel_receipt(root, profile, load):
    expected = profile["kernel_execution_overlay"]["artifacts"]
    receipts = load["kernel"]
    g.require(len(receipts) == 1, "Expected one owned GPU kernel receipt")
    base, receipt = receipts[0], receipts[0].get("local256_prefill", {})
    g.require(base["configuration"] == profile["kernel"] and base["observed_native_sha256"] == g.NATIVE_SHA
        and base["derived_helpers_sha256"] == expected["binary_helper_sha256"]
        and base["derived_launcher_sha256"] == expected["base_launcher_sha256"], "Base GPU kernel receipt differs")
    g.require(receipt.get("configuration") == local_parameters() and receipt.get("artifacts") == expected
        and receipt.get("actual_device_capability") == [12, 0] and receipt.get("base_dispatch_verified") is True
        and receipt.get("both_dispatch_references_bound") is True and receipt.get("B200_preset_enabled") is False
        and receipt.get("no_silent_fallback") is True, "Actual local256 GPU dispatch receipt differs")
    g.require(receipt.get("derived_source_file") == SOURCE_FILE
        and receipt.get("derived_source_file_sha256") == expected["derived_launcher_sha256"]
        and g.file_sha256(root / SOURCE_FILE) == expected["derived_launcher_sha256"]
        and g.file_sha256(root / "native_range_helpers.py") == expected["binary_helper_sha256"],
        "GPU derived source evidence missing or changed")


def _verify_cpu_receipt(root, p):
    # Same receipt requirements as frozen c.compare_runs; kernel-only comparison
    # cannot call its candidate branch because that correctly forbids new kernels.
    overlay, receipt = p["cpu_execution_overlay"], g.read(root / "cpu-copy.json")
    g.require(receipt["status"] == "COMPLETE" and receipt["profile_sha256"] == g.digest(p)
        and receipt["source_sha256"] == overlay["source_sha256"] and receipt["final_gc"]["status"] == "COMPLETE",
        "CPU overlay closure differs")
    g.require([r["sample_id"] for r in receipt["rows"]] == [x["sample_id"] for x in p["panel"]["items"]]
        and [r["deep"] for r in receipt["rows"]] == [i < 2 for i in range(p["maximum_model_calls"])]
        and receipt["gc_cadence"] == 8 and receipt["maximum_pending_ordinary"] <= 8, "CPU GC ledger differs")


def compare_runs(paths, purpose="candidate"):
    cpu_signatures, execution_signatures, changed = [], [], []
    for path in paths:
        root = Path(path)
        identity, load = g.read(root / "identity.json"), g.read(root / "load.json")
        p = identity["profile"]
        g.require(identity["profile_sha256"] == g.digest(p), "Stored profile digest differs")
        new = "local256_prefill" in p["kernel"] or "kernel_execution_overlay" in p
        changed.append(new)
        if new:
            validate_profile(p, installed_sdk=False)
            _verify_kernel_receipt(root, p, load)
        else:
            validate_sources()
            _qualified_cpu(p, installed_sdk=False)
            g.require(p["kernel"] == base_kernel(), "Only qualified C3 can be a kernel control")
        _verify_cpu_receipt(root, p)
        cpu_signatures.append(g.digest(c._without_binding(p["cpu_execution_overlay"])))
        execution_signatures.append(g.digest({"settings": {k: v for k, v in p["engine_settings"].items() if k != "model"},
            "implementation": p["implementation"], "runtime_sources": p["runtime_sources"],
            "checkpoint_assets": p["checkpoint_assets"], "variant": p["variant"],
            "torch_policy": load["placement"][0]["torch_policy"]}))
    g.require(any(changed) and len(set(cpu_signatures)) == len(set(execution_signatures)) == 1,
        "CPU algorithm, placement, engine or scientific execution differs")
    if purpose == "consistency":
        g.require(all(changed), "All six cold observations must use the new kernel")
        return c.compare_runs(paths, purpose)
    return g.compare_runs(paths, purpose)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("seal-profile")
    for name in ("plan", "runtime", "name", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--panel", help="Use the normalized 48-video panel JSON")
    p = commands.add_parser("run")
    for name in ("profile", "reference", "admission", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("compare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--purpose", choices=("candidate", "consistency"), default="candidate")
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "seal-profile":
        result = seal_profile(args)
    elif args.command == "run":
        result = run(args)
    else:
        result = compare_runs(args.runs, args.purpose)
        if args.output:
            g.seal(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
