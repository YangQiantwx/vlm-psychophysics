"""SM120 local256 prefill M64 with an optional, sealed eight-warp launch.

The default retains native warps. Read resources from the same successful launch
once per compiled object; never warm up, relaunch, or silently fall back.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path

from . import gemma_prefill as p

c, g = p.c, p.g
SCHEMA = "gemma-local256-tile64-v2"
PREFILL_SHA = "422efc179064119b0420e7536ea3e887023e9eb60d8f965af5da165b7b2fda18"
K4_LAUNCHER_SHA = "cc5014f6fcc5115b60899d7d1f93b900ffba7cb8e928ca220b6440da3d47faa0"
DERIVED_SHA = "2931d47c9994fea4eb54ea48fbf18f03e2eb953d743c6c8c49b3cb38bea51ccc"
WARP8_DERIVED_SHA = "d092ab967ca15efe42642a959893bd3e5d39608401eafb0016ac4d07e04d602f"
K5_SOURCE_SHA = "0749e794a908738c606935613bf7ddb6701870eb0566e02daae95e63859e6809"
K5_BUNDLE_SHA = "72690962c609e276d9a5c2a256c51cee1d4d68419bed4fac776a6c81c64c8bbf"
SOURCE_FILE = "head256_tile64_launcher.py"
RESOURCE_FILE = "head256_tile64_resources.jsonl"
_VALIDATE, _PARAMETERS, _ARTIFACTS = p.validate_sources, p.local_parameters, p.kernel_artifacts
_COMPILE, _INSTALL = p._compile_launcher, p.install_prefill
_RESOURCE = None
_LOCAL_NUM_WARPS = None


def validate_sources():
    _VALIDATE()
    g.require(g.file_sha256(p.__file__) == PREFILL_SHA, "Frozen K4 source changed")


def derived_sha(local_num_warps=None):
    g.require(local_num_warps is None or type(local_num_warps) is int and local_num_warps == 8,
        "Only native or explicitly sealed eight local prefill warps are supported")
    return DERIVED_SHA if local_num_warps is None else WARP8_DERIVED_SHA


def local_parameters():
    return {**_PARAMETERS(), "module": "classic_precision.gemma_tile64",
        "source_sha256": g.file_sha256(__file__), "prefill_block_m": 64,
        "qualified_K4_source_sha256": PREFILL_SHA, "derived_launcher_sha256": derived_sha(_LOCAL_NUM_WARPS),
        "local_num_warps": _LOCAL_NUM_WARPS, "expected_compiled_num_warps": _LOCAL_NUM_WARPS or 4,
        "expected_compiled_num_stages": 3,
        "warps_and_stages": "native defaults" if _LOCAL_NUM_WARPS is None else "local256 prefill warps=8; native stages unchanged",
        "rejected_native_M64_source_sha256": K5_SOURCE_SHA, "rejected_native_M64_bundle_sha256": K5_BUNDLE_SHA,
        "resource_capture": "same launch return; first use per compiled object",
        "resource_policy": "stop on known spill/over-limit or dispatch mismatch; unavailable is null, never zero"}


def derive_launcher(source, local_num_warps=None):
    derived_sha(local_num_warps)
    tree = ast.parse(source)
    g.require(len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
        and tree.body[0].name == "unified_attention", "K4 dispatcher interface differs")
    body = tree.body[0].body
    target = ast.dump(ast.parse(p._LOCAL_TEST, mode="eval").body, include_attributes=False)
    branches = [n for n in body if isinstance(n, ast.If)
        and ast.dump(n.test, include_attributes=False) == target]
    g.require(len(branches) == 1 and len(branches[0].body) == 1 and not branches[0].orelse
        and ast.unparse(branches[0].body[0]) == "BLOCK_M = 32", "Unique K4 local256 M32 anchor differs")
    branches[0].body = ast.parse("BLOCK_M = 64").body
    launches = [n for n in body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Subscript)
        and ast.unparse(n.value.func.value) == "kernel_unified_attention"]
    g.require(len(launches) == 1, "Unique native Triton launch anchor differs")
    launch = launches[0]
    if local_num_warps is not None:
        anchors = [i for i, node in enumerate(body) if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name) and node.target.id == "launch_kwargs"
            and isinstance(node.value, ast.Dict) and not node.value.keys]
        g.require(len(anchors) == 1 and anchors[0] < body.index(launch), "Native launch kwargs anchor differs")
        anchor = anchors[0]
        stores = lambda nodes: {n.id for node in nodes for n in ast.walk(node)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        names = {"launch_num_warps", "launch_num_stages"}
        g.require(names <= stores(body[:anchor]) and not names & stores(body[anchor:]),
            "Native warp/stage initialization order differs")
        body[anchor:anchor] = ast.parse(f"if {p._LOCAL_TEST}:\n    launch_num_warps = 8\n").body
    # Keep every original launch argument and the launch count unchanged.
    replacement = ast.parse(f"""try:
    _tile64_compiled = PLACEHOLDER()
except BaseException as _tile64_error:
    if {p._LOCAL_TEST}:
        _tile64_failure(_tile64_error)
    raise
if {p._LOCAL_TEST}:
    _tile64_record(_tile64_compiled, BLOCK_M, TILE_SIZE_PREFILL, launch_num_warps, launch_num_stages)
""").body
    replacement[0].body[0].value = launch.value
    i = body.index(launch)
    body[i:i + 1] = replacement
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def resource_api():
    dist = importlib.metadata.distribution("triton")
    paths = ("triton/runtime/jit.py", "triton/compiler/compiler.py")
    sources = {name: Path(dist.locate_file(name)).read_text() for name in paths}
    tree = ast.parse(sources[paths[0]])
    runs = [f for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == "JITFunction"
        for f in cls.body if isinstance(f, ast.FunctionDef) and f.name == "run"]
    returns_kernel = any(isinstance(n, ast.Return) and isinstance(n.value, ast.Name)
        and n.value.id == "kernel" for f in runs for n in ast.walk(f))
    return {"version": dist.version, "source_sha256": {k: p.sha(v) for k, v in sources.items()},
        "JITFunction_run_returns_kernel": returns_kernel,
        "inspection": "returned object metadata.shared/_run.shared/n_regs/n_spills; no cache mutation"}


def kernel_artifacts(sources=None):
    base = _ARTIFACTS(sources)
    g.require(base["receipt"]["derived_launcher_sha256"] == K4_LAUNCHER_SHA, "Qualified K4 launcher differs")
    derived = derive_launcher(base["derived_source"], _LOCAL_NUM_WARPS)
    expected_sha = derived_sha(_LOCAL_NUM_WARPS)
    g.require(p.sha(derived) == expected_sha, "M64 derived launcher differs")
    return {**base, "derived_source": derived, "receipt": {**base["receipt"],
        "qualified_K4_launcher_sha256": K4_LAUNCHER_SHA, "derived_launcher_sha256": expected_sha,
        "local_num_warps": _LOCAL_NUM_WARPS,
        "only_added_statement": "local256 prefill M64; local warps=" + str(_LOCAL_NUM_WARPS) + "; observe existing launch return",
        "triton_api": resource_api()}}


class Resources:
    def __init__(self, output, api, shared_limit, local_num_warps=None):
        self.path, self.api, self.shared_limit = Path(output) / RESOURCE_FILE, api, shared_limit
        self.local_num_warps, self.derived_sha = local_num_warps, derived_sha(local_num_warps)
        self.seen = {}

    def write(self, row):
        g.append(self.path, {"schema": SCHEMA, "source_sha256": g.file_sha256(__file__),
            "derived_launcher_sha256": self.derived_sha, "local_num_warps": self.local_num_warps, "triton_api": self.api,
            "shared_limit_bytes": self.shared_limit, **row})

    def failure(self, error):
        self.write({"status": "LAUNCH_FAILED", "error_type": type(error).__name__,
            "error": str(error), "shared_bytes": None, "n_regs": None, "n_spills": None,
            "unavailable": "launch did not return a compiled kernel; no fallback"})

    def record(self, kernel, block_m, tile, warps, stages):
        if id(kernel) in self.seen:
            return
        self.seen[id(kernel)] = kernel
        missing = {}
        def get(obj, field):
            try:
                value = getattr(obj, field, None)
            except Exception as error:
                missing[field] = type(error).__name__
                return None
            if type(value) is not int or value < 0:
                missing[field] = "not exposed as a nonnegative integer"
                return None
            return value
        metadata = getattr(kernel, "metadata", None)
        shared = get(getattr(kernel, "_run", None), "shared")
        metadata_shared = get(metadata, "shared")
        shared_source = "compiled._run.shared" if shared is not None else "compiled.metadata.shared"
        if shared is None:
            shared = metadata_shared
        if shared is not None:
            missing.pop("shared", None)
        registers, spills = get(kernel, "n_regs"), get(kernel, "n_spills")
        num_warps, num_stages = get(metadata, "num_warps"), get(metadata, "num_stages")
        violations = []
        if shared is not None and self.shared_limit is not None and shared > self.shared_limit:
            violations.append("shared memory exceeds actual device opt-in limit")
        if spills is not None and spills > 0:
            violations.append("compiled kernel spills")
        if block_m != 64 or tile != 32 or warps != self.local_num_warps or stages is not None:
            violations.append("actual launch parameters differ from sealed local configuration")
        if num_warps is not None and num_warps != (self.local_num_warps or 4):
            violations.append("compiled warp count differs")
        if num_stages is not None and num_stages != 3:
            violations.append("native compiled stage count differs")
        row = {"status": "RESOURCE_REJECTED" if violations else ("PARTIAL" if missing else "OBSERVED"),
            "head_size": 256, "max_seqlen_q_gt_one": True, "BLOCK_M": block_m, "TILE_SIZE": tile,
            "explicit_num_warps": warps, "explicit_num_stages": stages,
            "shared_bytes": shared, "shared_source": shared_source if shared is not None else None,
            "metadata_shared_bytes": metadata_shared, "n_regs": registers, "n_spills": spills,
            "compiled_num_warps": num_warps, "compiled_num_stages": num_stages,
            "unavailable": missing, "violations": violations,
            "observation": "after original launch returned; once per compiled object"}
        self.write(row)
        g.require(not violations, "M64 resource rejection: " + "; ".join(violations))


def _compile_launcher(source, namespace, filename):
    extra = {} if _RESOURCE is None else {
        "_tile64_record": _RESOURCE.record, "_tile64_failure": _RESOURCE.failure}
    return _COMPILE(source, {**namespace, **extra}, filename)


def overlay_record(original, artifacts):
    return {"schema": SCHEMA, "source_sha256": g.file_sha256(__file__),
        "qualified_K4_source_sha256": PREFILL_SHA, "CPU_copy_source_sha256": p.CPU_SHA,
        "qualified_C3_profile_sha256": g.digest(original), "artifacts": artifacts,
        "scope": "local256 prefill M64, local warps=" + str(_LOCAL_NUM_WARPS) + "; native stages, other math/tiles unchanged"}


@contextmanager
def specialization(local_num_warps=None):
    global _LOCAL_NUM_WARPS
    validate_sources()
    derived_sha(local_num_warps)
    previous_warps, _LOCAL_NUM_WARPS = _LOCAL_NUM_WARPS, local_num_warps
    replacements = {"validate_sources": validate_sources, "local_parameters": local_parameters,
        "kernel_artifacts": kernel_artifacts, "kernel_overlay_record": overlay_record,
        "_compile_launcher": _compile_launcher, "install_prefill": install_tile64, "SOURCE_FILE": SOURCE_FILE}
    before = {name: getattr(p, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(p, name, value)
        yield
    finally:
        for name, value in before.items():
            setattr(p, name, value)
        _LOCAL_NUM_WARPS = previous_warps


def install_tile64(worker, config, output):
    import torch
    global _RESOURCE
    local_num_warps = config["local256_prefill"].get("local_num_warps")
    with specialization(local_num_warps):
        api = resource_api()
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        limit = getattr(properties, "shared_memory_per_block_optin", None)
        limit = limit if type(limit) is int and limit > 0 else None
        g.require(_RESOURCE is None, "M64 resource recorder already installed")
        _RESOURCE = Resources(output, api, limit, local_num_warps)
        try:
            receipt = _INSTALL(worker, config, output)
        except BaseException:
            _RESOURCE = None
            raise
        receipt["changed_kernels"][-1] = "head256 prefill M64, local warps=" + str(local_num_warps) + "; native stages; same-launch resources"
        receipt["local256_prefill"]["resource_receipt"] = {"file": RESOURCE_FILE, "triton_api": api,
            "shared_limit_bytes": limit, "local_num_warps": local_num_warps,
            "capture_started": True, "no_extra_kernel_call": True}
        return receipt


def seal_profile(args):
    with specialization(getattr(args, "local_num_warps", None)):
        return p.seal_profile(args)


def run(args):
    local_num_warps = g.read(args.profile)["kernel"]["local256_prefill"].get("local_num_warps")
    with specialization(local_num_warps):
        return p.run(args)


def verify_resources(root, profile, load):
    expected = profile["kernel_execution_overlay"]["artifacts"]["triton_api"]
    warps = profile["kernel"]["local256_prefill"].get("local_num_warps")
    receipt = load["kernel"][0]["local256_prefill"].get("resource_receipt", {})
    g.require(receipt.get("file") == RESOURCE_FILE and receipt.get("triton_api") == expected
        and receipt.get("local_num_warps") == warps
        and receipt.get("capture_started") is True and receipt.get("no_extra_kernel_call") is True,
        "M64 resource installation receipt differs")
    rows = [json.loads(line) for line in (Path(root) / RESOURCE_FILE).read_text().splitlines()]
    g.require(bool(rows), "No actually invoked local256 M64 resource observation")
    for row in rows:
        g.require(row.get("source_sha256") == g.file_sha256(__file__)
            and row.get("derived_launcher_sha256") == derived_sha(warps) and row.get("triton_api") == expected
            and row.get("status") in ("OBSERVED", "PARTIAL") and not row.get("violations")
            and row.get("BLOCK_M") == 64 and row.get("TILE_SIZE") == 32
            and row.get("local_num_warps") == row.get("explicit_num_warps") == warps
            and row.get("explicit_num_stages") is None
            and row.get("compiled_num_warps") in (None, warps or 4) and row.get("compiled_num_stages") in (None, 3)
            and row.get("shared_limit_bytes") == receipt.get("shared_limit_bytes")
            and row.get("n_spills") in (None, 0), "Invalid M64 compiled resource observation")
    return expected


def compare_runs(paths, purpose="candidate"):
    cpu, executions, new, apis = [], [], [], []
    for path in paths:
        root = Path(path)
        identity, load = g.read(root / "identity.json"), g.read(root / "load.json")
        profile = identity["profile"]
        g.require(identity["profile_sha256"] == g.digest(profile), "Profile digest differs")
        is_new = profile["kernel"].get("local256_prefill", {}).get("module") == "classic_precision.gemma_tile64"
        new.append(is_new)
        warps = profile["kernel"].get("local256_prefill", {}).get("local_num_warps")
        with specialization(warps) if is_new else _k4_validation():
            p.validate_profile(profile, installed_sdk=False)
            p._verify_kernel_receipt(root, profile, load)
        if is_new:
            apis.append(g.digest(verify_resources(root, profile, load)))
        p._verify_cpu_receipt(root, profile)
        cpu.append(g.digest(c._without_binding(profile["cpu_execution_overlay"])))
        executions.append(g.digest({"settings": {k: v for k, v in profile["engine_settings"].items() if k != "model"},
            "implementation": profile["implementation"], "runtime_sources": profile["runtime_sources"],
            "checkpoint_assets": profile["checkpoint_assets"], "variant": profile["variant"],
            "torch_policy": load["placement"][0]["torch_policy"]}))
    g.require(any(new) and len(set(cpu)) == len(set(executions)) == len(set(apis)) == 1,
        "CPU, placement, engine, source or resource API differs")
    if purpose == "consistency":
        g.require(all(new), "All six cold observations must use M64")
        return c.compare_runs(paths, purpose)
    return g.compare_runs(paths, purpose)


@contextmanager
def _k4_validation():
    validate_sources()
    yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command, fields in (("seal-profile", ("plan", "runtime", "name", "output")),
                            ("run", ("profile", "reference", "admission", "output"))):
        child = commands.add_parser(command)
        for field in fields:
            child.add_argument("--" + field, required=True)
        if command == "seal-profile":
            child.add_argument("--panel")
            child.add_argument("--local-num-warps", type=int, choices=(8,),
                help="Explicit local256 M64 prefill override; omit to retain native four warps")
    child = commands.add_parser("compare")
    child.add_argument("runs", nargs="+")
    child.add_argument("--purpose", choices=("candidate", "consistency"), default="candidate")
    child.add_argument("--output")
    args = parser.parse_args()
    result = seal_profile(args) if args.command == "seal-profile" else (run(args) if args.command == "run"
        else compare_runs(args.runs, args.purpose))
    if args.command == "compare" and args.output: g.seal(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
