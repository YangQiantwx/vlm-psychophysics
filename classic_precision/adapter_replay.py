"""Finite native-adapter replay, with two engine repeats and two cold processes.

prepare only reads source/calibration/checkpoint metadata. run handles exactly
one family/route/cold process; start it separately for each cold pass and route.
It never dispatches a queue or retries a request. compare reads all four routes.
Original geometry qualifications are prerequisites, not new per-row calibration.
Every new request additionally executes the native deep guards and captures real
CPU/GPU tensor bytes through classic_precision_adapter_determinism.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
from types import SimpleNamespace
import uuid

from .storage import RunStore, atomic_json, canonical_json, digest, file_sha256, owner_identity


SCHEMA = "classic_precision_adapter_replay/v1"
COUNTS = {"video": 4, "multi": 6, "single": 12, "pair": 8}
ACTIVE = ("nemo", "qwen", "cosmos")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def relocated(path, mapping, host=None):
    path = Path(path)
    require(path.is_absolute(), "Source path must be absolute: " + str(path))
    host = host or socket.gethostname()
    candidates = [(Path(x["source"]), Path(x["target"])) for x in mapping if x.get("host", host) == host]
    require(all(a.is_absolute() and b.is_absolute() for a, b in candidates), "Path mappings must be absolute")
    candidates.sort(key=lambda pair: len(pair[0].parts), reverse=True)
    for source, target in candidates:
        if path.is_relative_to(source):
            return (target / path.relative_to(source)).resolve()
    return path.resolve()


def descriptor(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def verified(item, mapping=()):
    path = relocated(item["path"], mapping)
    require(path.is_file(), "Required source/qualification file is missing: " + str(path))
    require(file_sha256(path) == item["sha256"] and path.stat().st_size == item["bytes"],
            "Source SHA/size drift: " + str(path))
    return path


def _implementation():
    from runtime_compat import classic_precision_adapter_determinism as helper
    core = helper._core()
    return {"harness_sha256": file_sha256(__file__), "adapter_helper_sha256": file_sha256(helper.__file__),
            "determinism_helper_sha256": file_sha256(core.__file__),
            "qsa_helper_sha256": file_sha256(Path(core.__file__).with_name("classic_precision_qsa_determinism.py")),
            "base_runner_sha256": file_sha256(Path(__file__).with_name("inference.py")),
            "runtime_preparer_sha256": file_sha256(Path(__file__).with_name("reproducibility.py")),
            "storage_sha256": file_sha256(Path(__file__).with_name("storage.py"))}


def _route(coverage, route):
    entries = [r for r in coverage["routes"] if r["route_id"] == route]
    require(len(entries) == 1, "Coverage route must be unique: " + route)
    value = entries[0]
    ids = [r["sample_id"] for r in value["representative_samples"]]
    require(len(ids) == len(set(ids)) == COUNTS[route], "Finite representative count differs: " + route)
    return value


def _rows(value):
    rows = value.get("rows", value.get("samples"))
    require(isinstance(rows, list), "Original manifest rows missing")
    result = {r["sample_id"]: r for r in rows}
    require(len(result) == len(rows), "Duplicate original source sample ID")
    return result


def _adapter_descriptor(entry):
    name = {"video": "evaluate.py", "multi": "evaluate.py", "single": "static_extension.py", "pair": "endpoint_evaluate.py"}[entry["route_id"]]
    matches = [v for v in entry["implementation_files"] if Path(v["path"]).name == name]
    require(len(matches) == 1, "Native adapter source is ambiguous")
    return matches[0]


def native_model_spec(entry, manifest, family, mapping):
    """Read model specs from the exact files used by the native dispatchers."""
    route = entry["route_id"]
    source, value = entry["manifest"], manifest
    api = "native evaluation manifest.models[family]"
    if route == "single":
        name = "four-model-panel.json" if family == "gemma" else "protocol.json"
        candidates = [item for item in entry["protocols"] if Path(item["path"]).name == name]
        require(len(candidates) == 1, "Original static model-spec source descriptor missing: " + name)
        source = candidates[0]
        path = verified(source, mapping)
        require(path == verified(entry["manifest"], mapping).parent / name,
                "Original static model-spec file must be beside its execution manifest")
        value = read(path)
        if family == "gemma":
            require(value.get("base_execution_manifest_sha256") == entry["manifest"]["sha256"],
                    "Original Gemma panel execution-manifest binding differs")
            api = "static_extension.GemmaSession.__init__:panel.models[gemma]"
        else:
            require(manifest.get("protocol_sha256") == source["sha256"],
                    "Original static execution-manifest protocol binding differs")
            api = "static_worker.run/ImageProcessor.__init__:protocol.models[family]"
    spec = value.get("models", {}).get(family)
    require(isinstance(spec, dict), "Original native model-spec source lacks coverage: " + route + "/" + family)
    return spec, {"file": source, "native_api": api}


def bind_route(plan, coverage, route, family, *, assets=True):
    """Read-only physical source closure, selected rows and original CPU geometry."""
    entry, mapping = _route(coverage, route), plan["path_map"]
    for item in [entry["manifest"], entry["source_manifest"], *entry["protocols"], *entry["implementation_files"]]:
        verified(item, mapping)
    manifest_path = verified(entry["manifest"], mapping)
    manifest, source = read(manifest_path), read(verified(entry["source_manifest"], mapping))
    rows = _rows(manifest)
    source_rows = {} if route == "single" else _rows(source)
    selected = []
    for representative in entry["representative_samples"]:
        sid = representative["sample_id"]
        require(sid in rows and digest(rows[sid]) == representative["manifest_row_sha256"],
                "Selected scientific row changed: " + sid)
        row = rows[sid]
        require(row["task"] == representative["task"] and row["truth"] == representative["truth"]
                and __import__("hashlib").sha256(row["prompt"].encode()).hexdigest() == representative["prompt_utf8_sha256"],
                "Scientific prompt/task/truth changed: " + sid)
        # Static single has images as original source; other source manifests
        # identify every execution row. Their immutable file SHA closes linkage.
        if route != "single":
            require(sid in source_rows, "Execution row absent from original source: " + sid)
        if assets:
            for asset in representative["source_assets"]:
                path = relocated(asset["path"], mapping)
                expected = asset.get("expected_file_sha256", asset.get("expected_archive_sha256"))
                require(path.is_file(), "Selected native asset is missing: " + str(path))
                require(file_sha256(path) == expected, "Selected native asset SHA drift: " + str(path))
                if "ready_file" in asset:
                    verified(asset["ready_file"], mapping)
                if "expected_pixel_sha256" in asset:
                    import numpy as np
                    if "npz_key" in asset:
                        with np.load(path, allow_pickle=False) as archive:
                            values = archive[asset["npz_key"]]
                    else:
                        values = np.load(path, allow_pickle=False)
                    require(values.dtype == np.dtype("uint16") and values.shape == (512, 768, 3),
                            "Original RGB16 asset geometry changed: " + str(path))
                    actual = __import__("hashlib").sha256(values.tobytes(order="C")).hexdigest()
                    require(actual == asset["expected_pixel_sha256"], "Actual original pixel SHA drift: " + str(path))
        selected.append(row)
    model_spec, model_spec_source = native_model_spec(entry, manifest, family, mapping)
    calibration = plan["calibrations"][route][family]
    qualification = read(verified(calibration, mapping))
    geometry = validate_geometry(qualification, entry, rows, family)
    if route == "single" and family == "nemo":
        token = plan.get("auxiliary", {}).get("single_nemo_token_preflight")
        require(isinstance(token, dict), "Missing native Nemo single-image token preflight file binding")
        path = verified(token, mapping)
        require(path == manifest_path.parent / "native-image-token-preflight.json", "Native token preflight must be beside the static manifest")
        proof = read(path)
        extension = _adapter_descriptor(entry)["sha256"]
        require(proof.get("status") == "PASS" and proof.get("extension_sha256") == extension
                and proof.get("weights_loaded") is False and proof.get("new_model_calls") == 0
                and proof.get("features") == 384 and proof.get("replacement_tokens") == 386
                and proof.get("placeholder_tokens") == 1 and proof.get("added_tokens") == 385
                and _HEX.fullmatch(proof.get("native_get_image_repl_source_sha256", "")), "Native Nemo token preflight differs")
        examples = proof.get("examples", [])
        require({(rows[x["sample_id"]]["task"], rows[x["sample_id"]]["prompt_id"]) for x in examples} ==
                {(t, p) for t in ("T2", "T3") for p in ("P0", "P1", "P2")}, "Native token preflight prompt/task coverage missing")
    return {"entry": entry, "manifest": manifest, "rows": selected, "source": source_rows,
            "root": manifest_path.parent, "spec": model_spec, "model_spec_source": model_spec_source, "geometry": geometry,
            "calibration": calibration, "adapter": verified(_adapter_descriptor(entry), mapping)}


def validate_geometry(value, entry, all_rows, family):
    route = entry["route_id"]
    require(value.get("status") == ("COMPLETE" if route == "video" else "PASS")
            and value.get("weights_loaded") is False
            and value.get("model_calls", value.get("new_model_calls")) == 0
            and value.get("CUDA_initialized", False) is False,
            "Original CPU geometry qualification is not a zero-model-call completion")
    require(value.get("model", value.get("family", family)) == family, "CPU geometry model differs")
    manifest_key = {"video": "evaluation_manifest_sha256", "single": "manifest_sha256"}.get(route, "execution_manifest_sha256")
    require(value.get(manifest_key) == entry["manifest"]["sha256"], "CPU geometry manifest binding differs")
    if route == "single":
        native = next(v for v in entry["implementation_files"] if Path(v["path"]).name == "static_inputs.py")
        require(value.get("source_sha256") == native["sha256"], "CPU geometry static source binding differs")
    else:
        require(value.get("adapter_sha256") == _adapter_descriptor(entry)["sha256"], "CPU geometry adapter binding differs")
    records = value.get("rows", value.get("records", []))
    require(records and len({r["sample_id"] for r in records}) == len(records), "CPU geometry duplicate/missing records")
    shapes, tasks, counts = set(), set(), set()
    for record in records:
        require(record["sample_id"] in all_rows, "CPU geometry references unknown original source ID")
        row = all_rows[record["sample_id"]]
        receipt = record.get("input_receipt", record.get("receipt"))
        require(isinstance(receipt, dict), "CPU geometry receipt missing")
        if route == "video":
            require(record.get("technical_CPU_status") == "PASS" and bool(record.get("expected_GPU_hashes")),
                    "Original video expected first-projection qualification missing")
        else:
            require(receipt.get("deep_check") is True and _HEX.fullmatch(receipt.get("expected_BF16_GPU_sha256", "")),
                    "Original CPU expected BF16 qualification missing")
        shape = receipt.get("actual_expected_GPU_shape", receipt.get("actual_processor_tensor_shape"))
        require(isinstance(shape, list) and shape and all(type(n) is int and n > 0 for n in shape), "CPU geometry shape missing")
        shapes.add(tuple(shape)); tasks.add(row["task"])
        if route == "multi":
            counts.add(receipt.get("images_per_context"))
    require(tasks >= ({"T2", "T3"} if route == "single" else {"T4"} if route == "pair" else {"T1"}),
            "Original CPU geometry task coverage missing")
    require(route != "multi" or counts >= {2, 4}, "Original CPU geometry lacks two/four-image coverage")
    return {"scope": "ORIGINAL_NATIVE_GEOMETRY_NOT_30_INDIVIDUAL_CALIBRATIONS", "sample_ids": [r["sample_id"] for r in records],
            "shapes": [list(s) for s in sorted(shapes)], "tasks": sorted(tasks), "image_counts": sorted(counts)}


def processor_identity(value):
    require(isinstance(value, dict) and bool(value.get("qualname")) and _HEX.fullmatch(value.get("source_sha256", "")),
            "Processor source SHA/qualified API identity missing")
    return {k: value[k] for k in ("qualname", "source_sha256")}


def native_state_anchor(state, bound, plan, family):
    """Bind old native identity and an actual CPU processor event, without loading."""
    bundle = state.get("anchor_bundle")
    if bundle is not None:
        record = read(verified(bundle, plan["path_map"]))["records"][state["anchor_key"]]
        text, origin = record["identity_source_utf8"], record["identity_origin"]
        require(len(text.encode()) == origin["bytes"] and __import__("hashlib").sha256(text.encode()).hexdigest() == origin["sha256"],
                "Bundled original identity bytes differ from read-only source capture")
        original = json.loads(text)
        event, proof = record["processor_event"], record["processor_origin"]
    else:
        anchor = state.get("native_identity")
        require(isinstance(anchor, dict), "Missing original native model-state anchor")
        original = read(verified(anchor, plan["path_map"]))
        proof = state.get("processor_receipt", {})
        require(type(proof.get("line_number")) is int and 1 <= proof["line_number"] <= 256,
                "Original processor event needs a bounded source line anchor")
        path = relocated(proof["path"], plan["path_map"])
        import itertools
        with path.open() as stream:
            line = next(itertools.islice(stream, proof["line_number"]-1, proof["line_number"]), None)
        require(line is not None, "Original processor event source line missing")
        event = json.loads(line)
    require(original.get("model_key") == family and original.get("checkpoint_assets") == state["checkpoint_assets"]
            and original.get("cosmos_package_sources") == state.get("cosmos_package_sources"), "Original native checkpoint/package anchor differs")
    require(original.get("model") == bound["spec"], "Original native model spec differs from its bound dispatcher source")
    route = bound["entry"]["route_id"]
    key = {"video": "evaluation_manifest_sha256", "single": "manifest_sha256"}.get(route, "execution_manifest_sha256")
    require(original.get(key) == bound["entry"]["manifest"]["sha256"], "Original model-state anchor manifest differs")
    if route != "single":
        require(original.get("adapter_sha256") == _adapter_descriptor(bound["entry"])["sha256"], "Original model-state adapter source differs")
    require(event.get("event") == "PROCESSOR_IDENTITY" and digest(event) == proof.get("record_sha256")
            and processor_identity(event.get("processor")) == processor_identity(state["processor_source"]),
            "Original actual CPU processor event source binding differs")
    qualification = read(verified(bound["calibration"], plan["path_map"]))
    sources = [qualification.get("processor_source")]
    sources += [r.get("input_receipt", r.get("receipt", {})).get("processor_source")
                for r in qualification.get("rows", qualification.get("records", []))]
    require(all(processor_identity(s) == processor_identity(state["processor_source"]) for s in sources if s is not None),
            "Original measured CPU geometry used another processor source")
    return original


def build_plan(coverage_path, config):
    """Prepare on the execution host; config supplies pinned CPU model state."""
    from runtime_compat import classic_precision_adapter_determinism as helper
    core = helper._core()
    coverage_path = Path(coverage_path).resolve()
    coverage, implementation = read(coverage_path), _implementation()
    require(coverage.get("schema") == "classic_precision_input_path_coverage/v1", "Wrong coverage schema")
    frozen = coverage["frozen_strict_profile"]["implementation"]
    require(all(frozen.get(k) == implementation[k] for k in ("base_runner_sha256", "determinism_helper_sha256", "qsa_helper_sha256")),
            "Coverage is historical; supply a new version bound to the current strict core")
    require(config.get("models") == list(ACTIVE), "Finite suite requires the three active models; Gemma remains prepared")
    plan = {"schema": SCHEMA, "models": list(ACTIVE), "prepared_models": ["gemma"], "cold_passes": 2, "repeats": 2,
            "coverage": descriptor(coverage_path), "path_map": config.get("path_map", []),
            "profile": {"id": core.PROFILE_ID, "implementation": implementation, "environment": core.launch_environment(),
                        "fill_uninitialized_memory": core.FILL_UNINITIALIZED_MEMORY},
            "model_root": str(Path(config["model_root"]).resolve()),
            "cosmos_package": str(Path(config["cosmos_package"]).resolve()),
            "model_state": config["model_state"], "route_environment": config.get("route_environment", {}),
            "auxiliary": config.get("auxiliary", {}), "calibrations": {}, "sample_ids": {}}
    video_root = relocated(_route(coverage, "video")["manifest"]["path"], plan["path_map"]).parent
    plan["gpu_lock_root"] = str(Path(_route(coverage, "video")["manifest"]["path"]).parent / "evaluation/outputs")
    for route in COUNTS:
        entry = _route(coverage, route)
        plan["sample_ids"][route] = [r["sample_id"] for r in entry["representative_samples"]]
        plan["calibrations"][route] = {}
        for family in ACTIVE:
            candidates = [q["file"] for q in entry["existing_cpu_qualification"] if q["model"] == family]
            require(len(candidates) == 1, "Missing original CPU geometry: " + route + "/" + family)
            plan["calibrations"][route][family] = config.get("calibrations", {}).get(route, {}).get(family, candidates[0])
    # Preparation verifies actual data and checkpoint shards, not only JSON IDs.
    validate_plan(plan, coverage, physical=True)
    return plan


def validate_plan(plan, coverage=None, *, physical=True):
    require(plan.get("schema") == SCHEMA and plan.get("models") == list(ACTIVE)
            and plan.get("cold_passes") == plan.get("repeats") == 2, "Wrong finite replay plan")
    require(plan.get("profile", {}).get("implementation") == _implementation(), "Replay implementation drift")
    from runtime_compat import classic_precision_adapter_determinism as helper
    core = helper._core()
    require(plan["profile"]["id"] == core.PROFILE_ID and plan["profile"]["environment"] == core.launch_environment()
            and plan["profile"]["fill_uninitialized_memory"] == core.FILL_UNINITIALIZED_MEMORY, "Strict policy binding drift")
    coverage_path = verified(plan["coverage"], plan["path_map"])
    coverage = read(coverage_path) if coverage is None else coverage
    require(digest(coverage) == digest(read(coverage_path)), "Coverage object differs from pinned bytes")
    frozen = coverage["frozen_strict_profile"]["implementation"]
    require(all(frozen.get(k) == plan["profile"]["implementation"][k]
                for k in ("base_runner_sha256", "determinism_helper_sha256", "qsa_helper_sha256")), "Coverage strict core differs")
    video_root = relocated(_route(coverage, "video")["manifest"]["path"], plan["path_map"]).parent
    require(relocated(plan["gpu_lock_root"], plan["path_map"]) == video_root / "evaluation/outputs",
            "Physical GPU lease must use the original T1 shared lock root")
    for route in COUNTS:
        require(plan["sample_ids"][route] == [r["sample_id"] for r in _route(coverage, route)["representative_samples"]],
                "Finite selected IDs differ: " + route)
        for family in ACTIVE:
            state = plan["model_state"].get(family, {}).get(route, {})
            require(isinstance(state.get("checkpoint_assets"), dict) and state["checkpoint_assets"].get("weight_shard_bytes")
                    and isinstance(state.get("processor_source"), dict) and _HEX.fullmatch(state["processor_source"].get("source_sha256", "")),
                    "Missing pinned CPU model-state/processor coverage: " + family + "/" + route)
            if physical:
                bound = bind_route(plan, coverage, route, family)
                native_state_anchor(state, bound, plan, family)
                from . import inference as base
                path = base._inside(relocated(plan["model_root"], plan["path_map"]), bound["spec"]["checkpoint_relative"])
                require(base.checkpoint_asset_identity(path) == state["checkpoint_assets"], "Checkpoint metadata/shard size drift: " + str(path))
                if family == "cosmos":
                    require(base._cosmos_package_identity(relocated(plan["cosmos_package"], plan["path_map"])) == state.get("cosmos_package_sources"),
                            "Cosmos package model-state drift")
    for item in plan.get("auxiliary", {}).values():
        verified(item, plan["path_map"])
    return coverage


def _load(name, path):
    require(name not in sys.modules, "Native route must run in a fresh independent process: " + name)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _native(bound, plan, family):
    """Load the exact source adapters, preserving their native implementation."""
    route = bound["entry"]["route_id"]
    if route != "single":
        return _load("_adapter_replay_" + route, bound["adapter"])
    # Radon retains these three original scripts beside its execution manifest.
    root = bound["root"]
    for name in ("static_inputs", "static_worker", "static_extension"):
        item = next(v for v in bound["entry"]["implementation_files"] if Path(v["path"]).stem == name)
        require(verified(item, plan["path_map"]) == root / (name + ".py"),
                "Static native scripts must map beside runs/t2_t3_static_v1/execution-manifest.json")
        _load(name, root / (name + ".py"))
    si, sw, extension = (sys.modules[n] for n in ("static_inputs", "static_worker", "static_extension"))
    si.MODELS = relocated(plan["model_root"], plan["path_map"])
    si.COSMOS = relocated(plan["cosmos_package"], plan["path_map"])
    # Run the native wrapper installer, but its production dispatcher is replaced
    # with a sentinel. The native Nemo historical recovery sees an absent root.
    old_run, old_root = sw.run, extension.RUN
    absent = root / (".adapter-replay-no-history-" + uuid.uuid4().hex)
    require(not absent.exists(), "Unexpected diagnostic history path")
    calls = []
    sw.run = lambda selected: calls.append(selected)
    extension.RUN = absent
    try:
        extension.run_original_family(family)
    finally:
        sw.run, extension.RUN = old_run, old_root
    require(calls == [family], "Native wrapper installer dispatched unexpectedly")
    # This source-native *args wrapper forwards unchanged. Supply the original
    # public signature for inspect.bind; neither code nor scientific guards move.
    sw.StaticSession.prepare_image.__signature__ = inspect.signature(si.ImageProcessor.prepare_image)
    return extension


class StoreProxy:
    def __init__(self):
        self.current = None
    def event(self, name, **fields):
        require(self.current is not None, "Native event has no active repeat store")
        return self.current.event(name, **fields)


@contextmanager
def physical_lease(base, torch, plan, resources):
    """Use the actual native per-GPU flock; resource admission is inside it."""
    root = relocated(plan["gpu_lock_root"], plan["path_map"])
    properties = torch.cuda.get_device_properties(0)
    device = str(getattr(properties, "uuid", None) or os.environ["CUDA_VISIBLE_DEVICES"])
    lock = root / (".gpu-" + digest({"host": socket.gethostname(), "device": device}) + ".lock")
    with base._gpu_lock(root, torch):
        state = base.resource_snapshot(torch, resources)
        require(state.get("admitted") is True, "Physical resource admission rejected model loading")
        require(lock.is_file(), "Actual physical GPU lock file is missing")
        stat = lock.stat()
        yield {"path": str(lock), "root": str(root), "st_dev": stat.st_dev, "st_ino": stat.st_ino,
               "host": socket.gethostname(), "device": device, "owner": owner_identity(), "resource_snapshot": state}


def runtime_policy(snapshot, plan, family, environment):
    """Reuse the reviewed strict validator, then add adapter/physical bindings."""
    from . import reproducibility
    from runtime_compat import classic_precision_adapter_determinism as helper
    helper._source_guard(reproducibility.validate_runtime_policy,
        {reproducibility.__file__: plan["profile"]["implementation"]["runtime_preparer_sha256"]})
    identity = reproducibility.validate_runtime_policy(snapshot, family)
    require(snapshot.get("adapter_environment") == environment,
            "Actual native adapter environment is missing/changed")
    require(identity["profile"]["helper_sha256"] == plan["profile"]["implementation"]["determinism_helper_sha256"]
            and identity["inference_sha256"] == plan["profile"]["implementation"]["base_runner_sha256"]
            and identity.get("gpu_total_bytes") and snapshot.get("gpu_uuid") not in (None, "", "unavailable"),
            "Runtime source/physical GPU identity differs from the replay plan")
    return identity


def _environment(plan, family, route):
    value = {**plan["profile"]["environment"], **plan.get("route_environment", {}).get(route, {}).get(family, {})}
    require(all(value.get(k) == v for k, v in plan["profile"]["environment"].items()), "Route cannot override strict environment")
    if family == "qwen":
        value["CLASSIC_PRECISION_QSA_DETERMINISM"] = "1"
        require(route != "single" or value.get("CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB") == "16", "Original static Qwen offload16 must be explicit")
    return value


def pre_call_runtime(plan, family, environment):
    """Apply the source-guarded lazy QSA leaf before requiring applied status."""
    from . import reproducibility
    from runtime_compat import classic_precision_adapter_determinism as helper
    helper._source_guard(reproducibility._prepare_runtime_modules,
        {reproducibility.__file__: plan["profile"]["implementation"]["runtime_preparer_sha256"]})
    reproducibility._prepare_runtime_modules(family)
    snapshot = helper._core().runtime_snapshot()
    snapshot["adapter_environment"] = {key: os.environ.get(key) for key in environment}
    runtime_policy(snapshot, plan, family, environment)
    return snapshot


def native_video_ready(row, original, record, ready_sha256):
    """The selected-row ready projection made by T1 input_catalog()."""
    require(row["task"] == "T1" and row["truth"] == original["correct_answer"]
            and row["prompt"] == original["prompt"]
            and row["source_parameters"]["original_parameters"] == original["params"],
            "T1 queue identity or literal prompt differs from original source")
    require(record["scientific_spec_sha256"] == original["scientific_spec_sha256"]
            and record["technical_verdict"] == "PASS", "T1 source ready identity failed")
    return {"sample_id": row["sample_id"], "video_relative": record["video_path"],
            "video_sha256": record["video_sha256"], "source_ready_sha256": ready_sha256,
            "source_parameters": row["source_parameters"],
            "source_scientific_spec_sha256": original["scientific_spec_sha256"],
            "original_source_validation": {"decoded_frames": record["validation"]["decoded_frames"],
                "max_code_error": record["validation"]["max_code_error"],
                "FP32_roundtrip_max_code_error": record["validation"]["FP32_roundtrip_max_code_error"]}}


def _ready(bound, row):
    route, root = bound["entry"]["route_id"], bound["root"]
    if route == "video":
        item = next(r for r in bound["entry"]["representative_samples"] if r["sample_id"] == row["sample_id"])
        asset = item["source_assets"][0]
        expected = asset["ready_file"]
        path = root / "ready" / Path(expected["path"]).name
        require(file_sha256(path) == expected["sha256"] and path.stat().st_size == expected["bytes"],
                "T1 original ready source SHA/size drift")
        record = read(path)
        relative = Path(record["video_path"])
        video = (root / relative).resolve()
        require(not relative.is_absolute() and ".." not in relative.parts and video.is_relative_to(root.resolve()),
                "T1 source video escapes native asset root")
        require(record["video_sha256"] == asset["expected_file_sha256"], "T1 ready video SHA differs from selected original asset")
        stat = video.stat()
        require(stat.st_size == record["video_bytes"] and stat.st_mtime_ns == record["video_mtime_ns"],
                "T1 source media stat changed")
        return native_video_ready(row, bound["source"][row["sample_id"]], record, expected["sha256"])
    if route == "single":
        return next(x for x in bound["manifest"]["images"] if x["image_id"] == row["image_id"])
    if route == "multi":
        return {"sample_id": row["sample_id"], "source_sequence_sha256": row["source_sequence_sha256"],
            "ordered_assets": row["frame_assets"], "source_parameters": row["source_parameters"],
            "parent_video_sha256": row["parent_video_sha256"], "parent_ready_sha256": row["parent_ready_sha256"]}
    return {"sample_id": row["sample_id"], "source_pair_sha256": row["source_pair_sha256"],
            "ordered_assets": row["frame_assets"], "source_parameters": row["source_parameters"]}


def run(plan_path, output, family, route, cold_pass, gpu):
    require(family in ACTIVE, "Gemma is code-prepared and GPU execution remains disabled")
    require(route in COUNTS and cold_pass in (0, 1) and str(gpu) and "," not in str(gpu), "Explicit finite route/cold/single GPU required")
    require("torch" not in sys.modules and "vllm" not in sys.modules, "Replay requires a fresh process before Torch/vLLM")
    plan, output = read(plan_path), Path(output).resolve()
    coverage = validate_plan(plan, physical=True)
    environment = _environment(plan, family, route)
    require(all(os.environ.get(k) == v for k, v in environment.items()), "Set the plan environment in the new process before launching replay")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    bound = bind_route(plan, coverage, route, family)
    # Validate the native call's actual ready/image argument before loading any
    # engine; keep the original producer receipt separate from this projection.
    ready_records = {row["sample_id"]: _ready(bound, row) for row in bound["rows"]}
    directory = output / f"cold-{cold_pass}" / route
    directory.mkdir(parents=True, exist_ok=True)
    claim = directory / (family + ".claim.json")
    owner, engine_id = owner_identity(), uuid.uuid4().hex
    # O_EXCL prevents retrying or resuming even an incomplete prior cold pass.
    with claim.open("x") as stream:
        stream.write(canonical_json({"owner": owner, "plan_sha256": digest(plan), "engine_id": engine_id}) + "\n")
        stream.flush(); os.fsync(stream.fileno())
    from runtime_compat import classic_precision_adapter_determinism as helper
    helper.activate(family, verified(plan["coverage"], plan["path_map"]))
    from . import inference as base
    native = _native(bound, plan, family)
    # static_inputs sets OMP=1 at import. Restore the already-pinned strict
    # process environment before processor/model construction, and record it.
    for key, value in environment.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    cosmos_package = str(relocated(plan["cosmos_package"], plan["path_map"]))
    model_root = str(relocated(plan["model_root"], plan["path_map"]))
    if route == "video" and family == "cosmos":
        item = next(i for i in bound["entry"]["implementation_files"] if Path(i["path"]).name == "cosmos_runtime.py")
        compat = _load("_adapter_replay_cosmos_runtime", verified(item, plan["path_map"]))
        compat.install(cosmos_package, Path(base.__file__).resolve().parents[1])
    import torch
    resources = dict(bound["spec"]["resources"])
    proxy, session, handle = StoreProxy(), None, None
    args = SimpleNamespace(model=family, models=model_root, cosmos_package=cosmos_package)
    model_path = base._inside(model_root, bound["spec"]["checkpoint_relative"])
    state = plan["model_state"][family][route]
    done = {"status": "FAILED", "plan_sha256": digest(plan), "route": route, "model": family,
            "cold_pass": cold_pass, "owner": owner, "engine_id": engine_id}
    try:
        with physical_lease(base, torch, plan, resources) as lease, ExitStack() as cleanup:
            for repeat in range(2):
                identity = {"schema": SCHEMA, "model_key": family, "model": bound["spec"], "generation": base.GENERATION,
                    "manifest_sha256": bound["entry"]["manifest"]["sha256"], "checkpoint_assets": state["checkpoint_assets"],
                    "replay": {"plan_sha256": digest(plan), "profile": plan["profile"], "route": route, "cold_pass": cold_pass,
                        "repeat": repeat, "sample_ids": plan["sample_ids"][route], "engine_id": engine_id,
                        "model_spec_source": bound["model_spec_source"],
                        "geometry_calibration": bound["calibration"], "geometry_coverage": bound["geometry"]}}
                ids = plan["sample_ids"][route]
                with RunStore(directory / f"repeat-{repeat}", identity, family, "finite-adapter", ids) as store:
                    proxy.current = store
                    store.event("REPLAY_PHYSICAL_LEASE", lease=lease)
                    if session is None:
                        began = time.monotonic()
                        if route == "video":
                            guard = native.GuardStore(proxy, base)
                            session = native.make_session(base, family, model_path, guard, cosmos_package, resources)
                            guard.session = session
                        elif route == "single":
                            sw, si = sys.modules["static_worker"], sys.modules["static_inputs"]
                            session = sw.StaticSession(family, event_sink=proxy)
                            cleanup.callback(session.close)
                            session.load(resources)
                        else:
                            session = native.make_session(base, args, proxy, bound["spec"])
                            cleanup.callback(session.close)
                            session.load(resources)
                        if route == "video":
                            cleanup.callback(session.close)
                        require(processor_identity(base._source_identity(type(session.processor))) == processor_identity(state["processor_source"]), "Actual processor source differs from pinned CPU model-state")
                        handle = helper.instrument_session(session, route, native, verified(plan["coverage"], plan["path_map"]))
                        cleanup.callback(handle.close)
                        load_seconds = time.monotonic() - began
                    else:
                        load_seconds = 0.0
                    snapshot = pre_call_runtime(plan, family, environment)
                    store.event("REPLAY_RUNTIME", snapshot=snapshot, engine_id=engine_id, environment=environment,
                                model_state=state, load_seconds=load_seconds, instrumentation=helper.status())
                    for row in bound["rows"]:
                        sid = row["sample_id"]
                        ready = ready_records[sid]
                        require(sid not in store.outputs, "Replay cannot re-request durable raw")
                        binding = {"row": row, "ready": ready}
                        store.begin(sid, digest(binding))
                        began = time.monotonic()
                        def durable(raw):
                            require(raw.get("truth") == row["truth"] and raw.get("source_parameters") == row["source_parameters"], "Native raw scientific identity differs")
                            return store.commit_output(sid, {**raw, "replay_parent_sample_id": sid,
                                "adapter_replay": {"source_input": binding, "source_input_sha256": digest(binding),
                                    "plan_sha256": digest(plan), "route": route, "cold_pass": cold_pass, "repeat": repeat}})
                        try:
                            if route == "video":
                                session.hash_deep = True
                                session.generate(row, ready, bound["root"], durable)
                            elif route == "single":
                                session.generate_image(row, ready, durable, deep=True)
                            else:
                                arrays, observed = native.read_images(row) if route == "multi" else native.read_pair(bound["source"][sid])
                                require(observed == ready, "Actual native ordered source differs")
                                method = session.generate_images if route == "multi" else session.generate_pair
                                method(row, arrays, ready, durable, deep=True)
                            require(sid in store.outputs, "Native generation returned without durable raw")
                            raw = store.outputs[sid]
                            require(raw.get("finish_reason") == "stop" and raw.get("raw_answer") in ("LEFT", "RIGHT"),
                                    "Nonnormal raw is retained as terminal; replay will not retry or continue")
                            store.event("REPLAY_CALL_COMPLETED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], seconds=time.monotonic()-began)
                        except BaseException as error:
                            if sid in store.outputs:
                                store.event("POST_OUTPUT_VALIDATION_FAILED", sample_id=sid, attempt_id=store.outputs[sid]["attempt_id"], error_type=type(error).__name__, error=str(error))
                            else:
                                store.fail(sid, error, interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)), elapsed_seconds=time.monotonic()-began)
                            raise
                        finally:
                            if hasattr(session, "context"):
                                session.context.update(active=False)
                    store.event("REPLAY_REPEAT_COMPLETE", sample_ids=ids)
            done["status"] = "COMPLETE"
    except BaseException as error:
        done.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        proxy.current = None
        atomic_json(directory / (family + ".json"), done)
    return done


def tensor_receipts(events, sid, binding):
    sizes = {"torch.bool": 1, "torch.uint8": 1, "torch.int8": 1, "torch.int16": 2,
             "torch.int32": 4, "torch.int64": 8, "torch.float16": 2, "torch.bfloat16": 2,
             "torch.float32": 4, "torch.float64": 8, "torch.complex64": 8, "torch.complex128": 16}
    result = {}
    for stage in ("cpu_processor", "gpu_forward"):
        matches = [e for e in events if e.get("event") == "REPRO_INPUT_HASH" and e.get("sample_id") == sid and e.get("stage") == stage]
        require(len(matches) == 1, "Exactly one actual CPU/GPU receipt required: " + sid + "/" + stage)
        event = matches[0]
        require(all(event.get(k) == v for k, v in binding.items()), "Actual hash receipt source/instrumentation binding differs")
        tensors = event.get("tensors")
        require(isinstance(tensors, list) and tensors, "Actual tensor list missing")
        for tensor in tensors:
            shape, dtype = tensor.get("shape"), tensor.get("dtype")
            require(isinstance(tensor.get("name"), str) and tensor["name"] and dtype in sizes
                    and isinstance(shape, list) and all(type(n) is int and n >= 0 for n in shape)
                    and tensor.get("bytes") == math.prod(shape) * sizes[dtype]
                    and _HEX.fullmatch(tensor.get("sha256", "")), "Malformed actual tensor fingerprint")
        names = [t["name"] for t in tensors]
        require(stage != "cpu_processor" or ("input_ids" in names and any(n.startswith("pixel_values") for n in names)), "CPU actual pixels/input_ids absent")
        require(stage != "gpu_forward" or any(t["name"] == "pixels" and t["dtype"] == "torch.bfloat16" for t in tensors), "Actual GPU BF16 pixels absent")
        result[stage] = tensors
    return result


def _raw_fingerprint(raw, row, events, hashes, expected_ready=None):
    require(raw.get("replay_parent_sample_id") == row["sample_id"] and raw.get("truth") == row["truth"]
            and raw.get("source_parameters") == row["source_parameters"], "Raw original scientific identity differs")
    tokens = raw.get("output_token_ids")
    require(isinstance(tokens, list) and bool(tokens) and all(type(t) is int and t >= 0 for t in tokens), "Full original output tokens missing")
    require(isinstance(raw.get("raw_answer"), str) and raw.get("finish_reason") == "stop"
            and raw["raw_answer"] in ("LEFT", "RIGHT"), "Nonnormal/invalid raw remains terminal and fails certification")
    receipt = raw.get("input_receipt")
    require(isinstance(receipt, dict) and receipt.get("deep_check", True) is True, "Native deep input receipt missing")
    require(raw.get("actual_input_tokens") == receipt.get("input_tokens") and type(receipt.get("input_tokens")) is int
            and receipt["input_tokens"] + 16384 <= 65536, "Native actual full context receipt differs")
    require(receipt.get("chat_template_sha256") == digest(receipt.get("chat_template_text")), "Native full prompt template SHA differs")
    from .inference import GENERATION
    generation = raw.get("effective_generation", {})
    for key, expected in GENERATION.items():
        alias = {"output_choice": "choice", "context_limit": "max_context_tokens"}.get(key, key)
        require(generation.get(key, generation.get(alias)) == expected, "Native generation parameter changed: " + key)
    replay = raw.get("adapter_replay", {})
    require(replay.get("source_input", {}).get("row") == row and replay.get("source_input_sha256") == digest(replay.get("source_input")), "Durable raw source input binding differs")
    require(expected_ready is None or replay["source_input"].get("ready") == expected_ready, "Raw asset/ready identity differs from physical source")
    sid, attempt = raw["sample_id"], raw["attempt_id"]
    posts = [e for e in events if e.get("event") == "POST_OUTPUT_INPUT_RECEIPT" and e.get("sample_id") == sid and e.get("attempt_id") == attempt]
    require(len(posts) == 1, "Original native post receipt missing/duplicated")
    # Preserve every scientific/native input field; drop only request timing and
    # storage ownership before exact comparison across independent processes.
    omitted = {"seconds", "prepare_to_raw_seconds", "timestamp", "attempt_id", "attempt_number", "adapter_replay", "row_identity"}
    science = {k: v for k, v in raw.items() if k not in omitted}
    post = {k: v for k, v in posts[0].items() if k not in {"event", "timestamp", "time", "time_unix", "attempt_id", "owner"}}
    return {"raw": science, "post": post, "hashes": hashes}


def _validate_native_post(native, family, route, raw, post, bound):
    from runtime_compat import classic_precision_adapter_determinism as helper
    from . import inference as base
    allowed = {r["path"]: r["sha256"] for r in bound["entry"]["implementation_files"]}
    validator = native.validation_compat if route == "single" else native.validate_post
    helper._source_guard(validator, allowed)
    if route == "video":
        require(post.get("t1_deep_qualification") is True,
                "T1 video comparison requires actual native deep qualification")
        expected = post.get("t1_expected_hashes", {})
        require(isinstance(expected, dict) and all(_HEX.fullmatch(expected.get(k, ""))
                for k in ("input_sha256", "projection_sha256")),
                "T1 expected input/first projection hash qualification missing")
        validator(post, family, base)
    else:
        require(raw["input_receipt"].get("deep_check") is True, "Native image comparison requires actual deep qualification")
        validator(raw, post, family)


def _owner(value):
    return tuple(value.get(k) for k in ("host", "boot_id", "pid", "process_start_ticks"))


def compare(plan_path, output, report=None, *, model=None, verify_physical=True):
    from .summary import model_snapshot, technical_passed
    plan, output = read(plan_path), Path(output).resolve()
    coverage = validate_plan(plan, physical=verify_physical)
    require(model is None or model in plan["models"], "Selected comparison model is absent from the plan")
    selected = plan["models"] if model is None else [model]
    remaining = [family for family in plan["models"] if family not in selected]
    result = {"schema": SCHEMA, "status": "PASS", "plan_sha256": digest(plan), "models": {},
              "scope": "ONLY_THE_SELECTED_MODELS_30_INPUTS_AND_PINNED_RUNTIME", "requested_models": selected,
              "remaining_models": remaining, "all_plan_models_certified": False, "errors": []}
    for family in selected:
        model = {"status": "PASS", "checked_stores": 0, "expected_stores": 16, "correct_outputs": 0, "errors": []}
        baseline, runtime, cold_owners = {}, {}, {}
        for route in COUNTS:
            bound = bind_route(plan, coverage, route, family, assets=verify_physical)
            native_state_anchor(plan["model_state"][family][route], bound, plan, family)
            name = "static_extension" if route == "single" else "_adapter_replay_" + route
            native = sys.modules.get(name) or _native(bound, plan, family)
            ids, environment = plan["sample_ids"][route], _environment(plan, family, route)
            for cold in range(2):
                for repeat in range(2):
                    label = f"{family}/{route}/cold-{cold}/repeat-{repeat}"
                    try:
                        root = output / f"cold-{cold}" / route
                        done, claim = read(root / (family + ".json")), read(root / (family + ".claim.json"))
                        require(done.get("status") == "COMPLETE" and done.get("plan_sha256") == claim.get("plan_sha256") == digest(plan), "Cold pass is missing/incomplete or plan changed")
                        identity, rows, events, raw_tail, event_tail = model_snapshot(root / f"repeat-{repeat}" / family)
                        require(not raw_tail and not event_tail, "Journal has an incomplete tail")
                        replay = identity.get("replay", {})
                        require(replay.get("plan_sha256") == digest(plan) and replay.get("profile") == plan["profile"]
                                and replay.get("route") == route and replay.get("cold_pass") == cold and replay.get("repeat") == repeat
                                and replay.get("sample_ids") == ids and replay.get("engine_id") == done.get("engine_id") == claim.get("engine_id")
                                and replay.get("geometry_calibration") == bound["calibration"] and replay.get("geometry_coverage") == bound["geometry"], "Repeat identity/geometry binding differs")
                        require(identity.get("model") == bound["spec"] and replay.get("model_spec_source") == bound["model_spec_source"],
                                "Repeat model spec differs from original native dispatcher source")
                        require(identity.get("checkpoint_assets") == plan["model_state"][family][route]["checkpoint_assets"], "Stored checkpoint state differs")
                        require(len(rows) == len(ids) and {r["sample_id"] for r in rows} == set(ids) and technical_passed(rows, events) == set(ids), "Missing/duplicate/extra raw or original technical PASS")
                        starts = [e for e in events if e.get("event") == "WORKER_START"]
                        require(len(starts) == 1 and _owner(starts[0]["owner"]) == _owner(done["owner"]) == _owner(claim["owner"]), "Repeat process was resumed or owner binding differs")
                        owner = _owner(done["owner"])
                        require(all(owner), "Independent Linux process identity is incomplete")
                        key = (route, cold)
                        require(key not in cold_owners or cold_owners[key] == owner, "Same-engine repeats used different processes")
                        require(key in cold_owners or owner not in cold_owners.values(), "Cold/route process was reused")
                        cold_owners[key] = owner
                        leases = [e["lease"] for e in events if e.get("event") == "REPLAY_PHYSICAL_LEASE"]
                        require(len(leases) == 1, "Actual physical GPU lease receipt missing")
                        lease = leases[0]
                        lease_root = relocated(plan["gpu_lock_root"], plan["path_map"], owner[0])
                        lease_path = lease_root / (".gpu-" + digest({"host": owner[0], "device": lease.get("device")}) + ".lock")
                        require(lease.get("root") == str(lease_root) and Path(lease["path"]) == lease_path
                                and lease.get("st_dev") is not None and lease.get("st_ino")
                                and _owner(lease["owner"]) == owner and lease.get("host") == owner[0], "Lease path/process binding differs")
                        resources = lease.get("resource_snapshot", {})
                        require(resources.get("admitted") is True and resources.get("budgets") == bound["spec"]["resources"]
                                and resources.get("free_gpu_gib", -1) >= resources["budgets"]["min_free_gpu_gib"]
                                and resources.get("available_ram_gib", -1) >= resources["budgets"]["min_available_ram_gib"], "Physical resource admission receipt differs")
                        snapshots = [e for e in events if e.get("event") == "REPLAY_RUNTIME"]
                        require(len(snapshots) == 1 and snapshots[0].get("engine_id") == claim["engine_id"]
                                and snapshots[0].get("environment") == environment and snapshots[0].get("model_state") == plan["model_state"][family][route], "Runtime/engine identity differs")
                        snap = snapshots[0]["snapshot"]
                        actual_runtime = runtime_policy(snap, plan, family, environment)
                        require(snap["gpu_uuid"] == lease["device"], "Runtime GPU differs from physical lease")
                        if route in runtime:
                            require(runtime[route] == actual_runtime, "Package/source/hardware/runtime mismatch between cold passes")
                        runtime[route] = actual_runtime
                        attempts = [e for e in events if e.get("event") == "INFLIGHT"]
                        require(len(attempts) == len(ids) and len({e.get("sample_id") for e in attempts}) == len(ids)
                                and all(r.get("attempt_number") == 1 for r in rows), "Missing/repeated attempts; replay never retries")
                        require(sum(e.get("event") == "REPLAY_REPEAT_COMPLETE" for e in events) == 1, "Repeat completion missing")
                        fingerprint = {}
                        for raw in rows:
                            row = next(r for r in bound["rows"] if r["sample_id"] == raw["sample_id"])
                            binding = {"profile_id": "classic-precision-native-adapter-instrumentation-v1", "family": family, "route": route,
                                "coverage_sha256": plan["coverage"]["sha256"], "adapter_sha256": _adapter_descriptor(bound["entry"])["sha256"],
                                "instrumentation_sha256": plan["profile"]["implementation"]["adapter_helper_sha256"], "row_sha256": digest(row)}
                            hashes = tensor_receipts(events, raw["sample_id"], binding)
                            raw_replay = raw.get("adapter_replay", {})
                            require(all(raw_replay.get(k) == v for k, v in {"plan_sha256": digest(plan), "route": route,
                                    "cold_pass": cold, "repeat": repeat}.items()), "Raw repeat/plan identity differs")
                            value = _raw_fingerprint(raw, row, events, hashes, _ready(bound, row))
                            _validate_native_post(native, family, route, raw, value["post"], bound)
                            # A native expected shape/hash must match the separate
                            # actual GPU captures; no metadata-only SHA fallback.
                            receipt, gpu = raw["input_receipt"], hashes["gpu_forward"]
                            shape = receipt.get("actual_expected_GPU_shape", receipt.get("actual_processor_tensor_shape"))
                            require(shape in bound["geometry"]["shapes"], "This native input geometry lacks original CPU qualification")
                            pixels = [t for t in gpu if t["name"] == "pixels"]
                            if route != "video":
                                expected_shape = receipt.get("actual_expected_GPU_shape", receipt.get("actual_processor_tensor_shape"))
                                require(len(pixels) == 1 and pixels[0]["shape"] == expected_shape
                                        and pixels[0]["sha256"] == receipt.get("expected_BF16_GPU_sha256"), "Independent GPU captures differ from native strong input guard")
                            fingerprint[raw["sample_id"]] = value
                            model["correct_outputs"] += raw["raw_answer"] == row["truth"]
                        require(route not in baseline or baseline[route] == fingerprint, "Full raw/output tokens/native post/actual input hashes differ across repeat/cold")
                        baseline[route] = fingerprint
                        model["checked_stores"] += 1
                    except (RuntimeError, OSError, ValueError, KeyError, TypeError) as error:
                        model["errors"].append(label + ": " + str(error))
        if model["errors"] or model["checked_stores"] != 16:
            model["status"] = result["status"] = "FAIL"
        result["models"][family] = model
        result["errors"].extend(model["errors"])
    result["all_plan_models_certified"] = not remaining and result["status"] == "PASS"
    if report:
        require(not Path(report).exists(), "Comparison reports are immutable; choose a new path")
        atomic_json(report, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--coverage", required=True); prepare.add_argument("--config", required=True); prepare.add_argument("--plan", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--plan", required=True); execute.add_argument("--output", required=True)
    execute.add_argument("--model", required=True, choices=(*ACTIVE, "gemma")); execute.add_argument("--route", required=True, choices=tuple(COUNTS))
    execute.add_argument("--cold-pass", "--pass-index", dest="cold", required=True, type=int, choices=(0, 1)); execute.add_argument("--gpu", required=True)
    check = commands.add_parser("compare")
    check.add_argument("--plan", required=True); check.add_argument("--output", required=True); check.add_argument("--report", required=True)
    check.add_argument("--model", choices=ACTIVE)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        require(not Path(args.plan).exists(), "Replay plans are immutable; choose a new path")
        plan = build_plan(args.coverage, read(args.config)); atomic_json(args.plan, plan)
        print(canonical_json({"status": "PREPARED_NOT_GPU_VALIDATED", "plan_sha256": digest(plan), "selected_inputs": 30}))
    elif args.command == "run":
        print(canonical_json(run(args.plan, args.output, args.model, args.route, args.cold, args.gpu)))
    else:
        result = compare(args.plan, args.output, args.report, model=args.model); print(canonical_json(result))
        return int(result["status"] != "PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
