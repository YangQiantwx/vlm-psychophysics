"""Source-bound ordinary CPU copies and bounded GC for the frozen Gemma runner.

The two deep calls, model execution, journals, timers and shutdown stay in the
original worker. Only ordinary preparation and its explicit loop GC are wrapped.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, nullcontext
from copy import copy, deepcopy
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

from . import gemma_memory as m
from . import gemma_speed as g

SCHEMA = "gemma-cpu-copy-overlay-v1"
MEMORY_SHA = "8687b4649d69410c87c163572f62af8bc371489141c9582799ee961532fcfcbd"
SDK_FILE = "transformers/models/gemma4/video_processing_gemma4.py"
_PREPARE, _DECODE, _GC, _RUN = g.prepare, g.decode, g.gc, g.run


def _run_gc_line():
    lines, start = inspect.getsourcelines(_RUN)
    calls = [n for n in ast.walk(ast.parse("".join(lines)))
             if isinstance(n, ast.Call) and ast.unparse(n.func) == "gc.collect"]
    g.require(len(calls) == 1, "Frozen run GC call site differs")
    return start + calls[0].lineno - 1


_RUN_GC_LINE = _run_gc_line()


def validate_sources():
    m.validate_sources()
    g.require(g.file_sha256(m.__file__) == MEMORY_SHA, "Memory overlay source differs")


def processor_candidate(source):
    """Derive just the two singleton stack views, without importing the SDK."""
    tree = ast.parse(source)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Gemma4VideoProcessor"]
    functions = [n for cls in classes for n in cls.body
                 if isinstance(n, ast.FunctionDef) and n.name == "_preprocess"]
    g.require(len(functions) == 1, "Native Gemma video processor interface differs")
    native = functions[0]
    candidate, hits = deepcopy(native), []
    g.require(not candidate.decorator_list, "Native processor decorators changed")
    for node in ast.walk(candidate):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in ("pixel_values", "position_ids") and ast.unparse(node.value) == f"torch.stack({name}, dim=0)":
            node.value = ast.parse(f"{name}[0].unsqueeze(0) if len({name}) == 1 else torch.stack({name}, dim=0)", mode="eval").body
            hits.append(name)
    g.require(sorted(hits) == ["pixel_values", "position_ids"], "Native singleton stack anchors differ")
    code = ast.unparse(ast.fix_missing_locations(ast.Module(body=[candidate], type_ignores=[]))) + "\n"
    return code, {"file": SDK_FILE, "file_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "native_function_ast_sha256": hashlib.sha256(ast.dump(native, include_attributes=False).encode()).hexdigest(),
        "derived_function_sha256": hashlib.sha256(code.encode()).hexdigest(), "changed_assignments": sorted(hits)}


def installed_processor_source():
    path = importlib.metadata.distribution("transformers").locate_file(SDK_FILE)
    return processor_candidate(Path(path).read_text())


def overlay_record(base, placement, cadence, processor):
    g.require(placement in ("baseline", "memory28") and type(cadence) is int and cadence in (0, 8), "Unsupported CPU execution flags")
    g.require(base["variant"] in ("cpu", "kernel") and base["first_deep_calls"] == 2, "Keep the two original deep calls and streamlined ordinary path")
    calls = base["maximum_model_calls"]
    g.require(calls in m.REFERENCE_SHA, "Only the frozen 20/48 references are supported")
    g.require(processor["file"] == SDK_FILE and processor["file_sha256"] == base["runtime_sources"][SDK_FILE], "Processor SDK source differs")
    return {"schema": SCHEMA, "module": "classic_precision.gemma_cpu_copy", "source_sha256": g.file_sha256(__file__),
        "base_implementation": {"gemma_speed": m.BASE_SHA, "inference": m.INFERENCE_SHA},
        "base_profile_sha256": g.digest(base), "placement": placement,
        "memory_overlay_source_sha256": MEMORY_SHA if placement == "memory28" else None,
        "reference_file_sha256": m.REFERENCE_SHA[calls], "processor_source": processor,
        "ordinary_direct_FP32_decode": True, "singleton_processor_stack_views": True,
        "deep_prepare": "original unchanged", "gc_cadence": cadence,
        "gc_target": {"function": "gemma_speed.run", "source_line": _RUN_GC_LINE,
            "match": "caller f_code identity and exact explicit loop call line",
            "other_calls": "original", "final_collect": True},
        "generation_timing_shutdown": "delegated unchanged to frozen gemma_speed"}


def validate_profile(profile, installed_sdk=True):
    validate_sources()
    base = deepcopy(profile)
    overlay = base.pop("cpu_execution_overlay", None)
    g.require(isinstance(overlay, dict), "Missing CPU execution overlay")
    g.require(base["implementation"] == {"gemma_speed": m.BASE_SHA, "inference": m.INFERENCE_SHA}, "Base implementation identity changed")
    if overlay.get("placement") == "memory28":
        m.validate_profile(base)
    else:
        g.require("execution_overlay" not in base and base["engine_settings"] == m._BASE_ENGINE_SETTINGS(base["runtime"]["checkpoint"]), "Baseline placement changed")
    processor = installed_processor_source()[1] if installed_sdk else overlay["processor_source"]
    g.require(overlay == overlay_record(base, overlay.get("placement"), overlay.get("gc_cadence"), processor), "CPU overlay source, flags or base binding differs")
    return profile


def direct_decode(item, asset_root, streamlined):
    """Ordinary-only: identical FP32 casts/multiplies, without full uint16 codes."""
    import av
    import numpy as np
    g.require(streamlined, "Direct decode requires the ordinary streamlined path")
    ready = item["ready"]
    path = Path(asset_root) / ready.get("video_relative", ready.get("video_path"))
    g.require(path.resolve().is_relative_to(Path(asset_root).resolve()), "Source escaped asset root")
    g.require(g.file_sha256(path) == ready["video_sha256"], "Source video hash differs")
    values = np.empty((180, 528, 768, 3), np.float32)
    pts, stream_hash = [], hashlib.sha256()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        g.require(stream.codec_context.name == "ffv1" and stream.codec_context.format.name == "gbrp16le", "Wrong RGB16 source codec")
        for i, frame in enumerate(container.decode(video=0)):
            g.require(i < 180, "Unexpected extra frame")
            data = frame.to_ndarray(format="rgb48le")
            g.require(data.dtype == np.uint16 and data.shape == (512, 768, 3), "Source frame shape/dtype differs")
            stream_hash.update(memoryview(data).cast("B"))
            if i == 0:
                padding = data[0, 0].astype(np.float32)
            values[i, :512], values[i, 512:] = data, padding
            np.multiply(values[i], 255. / 65535., out=values[i])
            pts.append(float(frame.pts * frame.time_base))
    g.require(len(pts) == 180 and max(abs(t - i / 60) for i, t in enumerate(pts)) <= .001, "Source frames/timestamps differ")
    stream_sha = stream_hash.hexdigest()
    if ready.get("stream_sha256"):
        g.require(stream_sha == ready["stream_sha256"], "Decoded source hash differs")
    return None, values, {"source_video_sha256": ready["video_sha256"], "source_stream_sha256": stream_sha,
        "submitted_shape": list(values.shape), "submitted_dtype": str(values.dtype), "frames": 180,
        "padding_bottom_rows": 16, "source_pts_s": pts}


def memory_observation():
    statm = Path("/proc/self/statm")
    rss = int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") if statm.exists() else None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
    return {"rss_bytes": rss, "process_peak_rss_bytes": peak}


class GCCadence:
    """A proxy for g.gc only; other modules retain the real gc module."""

    def __init__(self, cadence, code=None, line=None):
        g.require(type(cadence) is int and cadence in (0, 8), "Unsupported GC cadence")
        self.cadence, self.code, self.line = cadence, code or _RUN.__code__, line or _RUN_GC_LINE
        self.rows, self.pending, self.maximum_pending = [], 0, 0
        self.final = None

    def __getattr__(self, name):
        return getattr(_GC, name)

    def collect(self, *args, **kwargs):
        caller = sys._getframe(1)
        if caller.f_code is not self.code or caller.f_lineno != self.line:
            return _GC.collect(*args, **kwargs)
        deep, sid = caller.f_locals.get("deep"), caller.f_locals.get("sid")
        g.require(type(deep) is bool and isinstance(sid, str), "Run GC request identity missing")
        del caller
        collect = deep or self.cadence == 0
        if not collect:
            self.pending += 1
            self.maximum_pending = max(self.maximum_pending, self.pending)
            collect = self.pending == self.cadence
        before, started = memory_observation(), time.monotonic()
        reclaimed = _GC.collect(*args, **kwargs) if collect else 0
        if collect:
            self.pending = 0
        self.rows.append({"sample_id": sid, "deep": deep, "collected": collect,
            "pending_ordinary": self.pending, "collected_objects": reclaimed,
            "collect_seconds": time.monotonic() - started, "before": before, "after": memory_observation()})
        return reclaimed

    def finish(self):
        before = memory_observation()
        self.final = {"status": "STARTED", "pending_ordinary": self.pending, "before": before}
        try:
            reclaimed = _GC.collect()
        except BaseException as exc:
            self.final.update(status="FAILED", error_type=type(exc).__name__, error=str(exc))
            raise
        self.pending = 0
        self.final.update(status="COMPLETE", collected_objects=reclaimed, after=memory_observation())


@contextmanager
def installed_overlay(profile):
    validate_sources()
    g.require(g.prepare is _PREPARE and g.decode is _DECODE and g.gc is _GC, "Another CPU overlay is already installed")
    code, processor_receipt = installed_processor_source()
    overlay = profile["cpu_execution_overlay"]
    g.require(processor_receipt == overlay["processor_source"], "Actual processor source changed")
    state = GCCadence(overlay["gc_cadence"])
    compiled = {}

    def prepare(session, item, roots, streamlined, deep):
        if deep:
            return _PREPARE(session, item, roots, streamlined, deep)
        cls = type(session.processor.video_processor)
        original = cls._preprocess
        if cls not in compiled:
            g.require(g.file_sha256(inspect.getsourcefile(original)) == processor_receipt["file_sha256"], "Actual processor method source differs")
            namespace = dict(original.__globals__)
            exec(compile(code, "<gemma-ordinary-singleton-views>", "exec"), namespace)
            compiled[cls] = original, namespace["_preprocess"]
        g.require(original is compiled[cls][0], "Processor method was rebound by another overlay")
        g.decode, cls._preprocess = direct_decode, compiled[cls][1]
        try:
            return _PREPARE(session, item, roots, streamlined, deep)
        finally:
            g.decode, cls._preprocess = _DECODE, original

    g.prepare, g.gc = prepare, state
    failure = None
    try:
        yield state
    except BaseException as exc:
        failure = exc
        raise
    finally:
        g.prepare, g.gc = _PREPARE, _GC
        try:
            state.finish()
        except BaseException:
            if failure is None:
                raise


def seal_profile(args):
    validate_sources()
    g.require(not Path(args.output).exists(), "Refusing to overwrite an existing sealed profile")
    with tempfile.TemporaryDirectory(prefix="gemma-cpu-profile-") as temporary:
        base_args = copy(args)
        base_args.output, base_args.panel_key = str(Path(temporary) / "base.json"), "panel"
        (m.seal_profile if args.placement == "memory28" else g.seal_profile)(base_args)
        base = g.read(base_args.output)
    profile = deepcopy(base)
    profile["cpu_execution_overlay"] = overlay_record(base, args.placement, args.gc_cadence, installed_processor_source()[1])
    validate_profile(profile)
    g.seal(args.output, profile)
    return {"status": "SEALED", "profile_sha256": g.digest(profile), "CPU_overlay": profile["cpu_execution_overlay"], "GPU_calls": 0}


def run(args):
    profile = validate_profile(g.read(args.profile))
    g.require(g.file_sha256(args.reference) == profile["cpu_execution_overlay"]["reference_file_sha256"], "Use the existing frozen CPU reference")
    output = Path(args.output)
    g.require(not output.exists(), "No implicit resume or overwrite")
    state, completed = None, False
    placement = m.installed_overlay() if profile["cpu_execution_overlay"]["placement"] == "memory28" else nullcontext()
    try:
        with placement, installed_overlay(profile) as state:
            result = g.run(args)
        completed = True
        return result
    finally:
        if state is not None and output.is_dir():
            g.seal(output / "cpu-copy.json", {"schema": SCHEMA, "status": "COMPLETE" if completed else "STOPPED",
                "profile_sha256": g.digest(profile), "source_sha256": g.file_sha256(__file__),
                "gc_cadence": state.cadence, "maximum_pending_ordinary": state.maximum_pending,
                "rows": state.rows, "final_gc": state.final,
                "engine_and_lease_shutdown": "delegated unchanged; use original exit.json"})


def _without_binding(overlay):
    return {k: v for k, v in overlay.items() if k not in ("base_profile_sha256", "baseline_profile_sha256")}


def compare_runs(paths, purpose="candidate"):
    profiles, signatures, cpu, memories = [], [], [], []
    for path in paths:
        root = Path(path)
        identity, load = g.read(root / "identity.json"), g.read(root / "load.json")
        p = identity["profile"]
        g.require(identity["profile_sha256"] == g.digest(p), "Stored profile digest differs")
        overlay = p.get("cpu_execution_overlay")
        if overlay:
            validate_profile(p, installed_sdk=False)
            receipt = g.read(root / "cpu-copy.json")
            g.require(receipt["status"] == "COMPLETE" and receipt["profile_sha256"] == g.digest(p)
                and receipt["source_sha256"] == overlay["source_sha256"] and receipt["final_gc"]["status"] == "COMPLETE",
                "CPU overlay completion/final GC receipt missing")
            g.require([r["sample_id"] for r in receipt["rows"]] == [x["sample_id"] for x in p["panel"]["items"]]
                and [r["deep"] for r in receipt["rows"]] == [i < 2 for i in range(p["maximum_model_calls"])]
                and receipt["gc_cadence"] == overlay["gc_cadence"]
                and receipt["maximum_pending_ordinary"] <= overlay["gc_cadence"], "CPU GC request ledger differs")
            cpu.append(g.digest(_without_binding(overlay)))
        else:
            g.require(purpose == "candidate", "Every consistency observation needs the same CPU overlay")
        profiles.append(p)
        memories.append(g.digest(_without_binding(p.get("execution_overlay", {}))))
        signatures.append(g.digest({"kernel": p["kernel"], "implementation": p["implementation"],
            "runtime_sources": p["runtime_sources"], "checkpoint_assets": p["checkpoint_assets"],
            "settings": {k: v for k, v in p["engine_settings"].items() if k != "model"},
            "torch_policy": load["placement"][0]["torch_policy"]}))
    g.require(cpu and len(set(cpu)) == 1, "CPU algorithm, parameters or source differ")
    g.require(len(set(memories)) == len(set(signatures)) == 1, "CPU comparison requires identical placement, engine and kernel profiles")
    result = g.compare_runs(paths, purpose)
    if result["status"] == "PROFILE_OUTPUT_DIFFERENCE":
        result["status"] = "DIFFERENT"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("seal-profile")
    for name in ("plan", "runtime", "name", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--panel", help="Use the normalized 48-video panel JSON")
    p.add_argument("--placement", choices=("baseline", "memory28"), required=True)
    p.add_argument("--gc-cadence", type=int, choices=(0, 8), required=True)
    p.add_argument("--variant", choices=("cpu", "kernel"), default="kernel")
    p.add_argument("--range-lookup", choices=("native", "binary"), default="binary")
    p.add_argument("--prefill-block-m", type=int, choices=(0, 32), default=32)
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
