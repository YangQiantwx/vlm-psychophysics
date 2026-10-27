"""Finite Gemma workers with one engine, explicit ownership, and byte proofs.

Sealing and dispatch only join metadata. Inference is an explicit run command;
production additionally requires the final entry's speed/equivalence/six-cold gate.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from collections.abc import Mapping
from contextlib import contextmanager, ExitStack, nullcontext
from copy import deepcopy, copy
import fcntl
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid
from types import CodeType

from . import gemma_speed as g

SCHEMA = "gemma-production-input-catalog-v1"
EXPECTED = {"T1": 3840, "T2": 3744, "T3": 1776}
ENTRY_SCHEMA = "gemma-persistent-entry-v1"
BASE_SHA = "8f69560e111f78d515699c9dcb80921908e8610eb7eab5ce4fd25c835275ddeb"
CATALOG_SHA = "6340069fffca228be81ebc910ba89a5f6774f9e26d78149a4a2cb0d07fbd0acd"
DERIVED_FILE = "persistent_runner.py"
REFERENCE_POLICY = "catalog-stream-plus-exact-control-template; first-two-existing-deep; control-panel-per-ID-oracle-v1"
CONTROL_KEYS = ("input_token_ids", "input_token_ids_sha256", "prompt_sha256",
                "position_sha256", "position_shape", "HF_control_tensors")


def _lines(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_new(path, value):
    """Do not overwrite sealed scientific evidence."""
    path = Path(path)
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, ensure_ascii=False, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def publish_message(path, value):
    """A polling worker must never see a partially written inbox message."""
    path = Path(path)
    temporary = path.with_name(path.name + ".publish-" + uuid.uuid4().hex)
    try:
        _write_new(temporary, value)
        os.link(temporary, path)  # Atomic visibility and no replacement.
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if temporary.exists(): temporary.unlink()


def _template_key(task, prompt):
    return g.digest({"task": task, "exact_prompt": prompt})


def build_catalog(panels, audit_rows, completion, cpu_reference, qualified_panel,
                  expected_counts=None):
    """Join existing metadata; never decode videos or repeat source measurements.

    ``panels`` contain row/ready dictionaries and the actual source-family key.
    The caller binds the files supplying these dictionaries when sealing.
    """
    expected = EXPECTED if expected_counts is None else expected_counts
    g.require(completion["status"] == "COMPLETE_PASS"
              and completion["failed_IDs"] == []
              and completion["counts"] == expected
              and completion["unique_IDs"] == sum(expected.values())
              and completion["decoded_frames"] == 180 * sum(expected.values()),
              "Source audit is not complete")
    g.require(cpu_reference["panel_sha256"] == g.digest(qualified_panel),
              "Qualified CPU reference/panel differs")
    g.validate_items(qualified_panel["items"])
    audit = {x["sample_id"]: x for x in audit_rows}
    g.require(len(audit) == len(audit_rows) == sum(expected.values()),
              "Source audit IDs are missing or duplicated")
    templates, deep_ids = {}, {task: [] for task in expected}
    for item in qualified_panel["items"]:
        sid, task = item["sample_id"], item["task"]
        row, ready, qa = item["row"], item["ready"], audit[sid]
        ref = cpu_reference["rows"][sid]
        g.require(qa["row_sha256"] == g.digest(row)
                  and qa["ready_sha256"] == g.digest(ready)
                  and qa["video_sha256"] == ready["video_sha256"] == ref["source_video_sha256"]
                  and qa["decoded_stream_sha256"] == ref["source_stream_sha256"],
                  "Qualified input no longer matches source audit")
        key = _template_key(task, row["prompt"])
        value = {k: deepcopy(ref[k]) for k in CONTROL_KEYS}
        if key in templates:
            g.require(templates[key]["controls"] == value,
                      "Same prompt/geometry has different processor controls")
        else:
            templates[key] = {"task": task, "prompt": row["prompt"], "controls": value}
        deep_ids[task].append(sid)
    items, source_families = {}, {}
    for panel in panels:
        family = panel["key"]
        g.require(family not in source_families, "Duplicate source-family key")
        source_families[family] = {"manifest_sha256": panel["manifest_sha256"],
                                   "expected_counts": panel["expected_counts"]}
        readies = {x["sample_id"]: x for x in panel["readies"]}
        rows = panel["rows"]
        g.require(len(readies) == len(panel["readies"]) == len(rows)
                  and set(readies) == {x["sample_id"] for x in rows},
                  "Ready and manifest IDs differ")
        g.require(dict(Counter(x["task"] for x in rows)) == panel["expected_counts"],
                  "Source-family denominators differ")
        for row in rows:
            sid, task = row["sample_id"], row["task"]
            g.require(sid not in items and sid in audit, "Duplicate or unaudited ID")
            ready, qa = readies[sid], audit[sid]
            item = {"sample_id": sid, "task": task, "row": row, "ready": ready,
                    "row_sha256": g.digest(row), "ready_sha256": g.digest(ready)}
            g.validate_items([item])
            g.require(qa["status"] == "PASS" and qa["panel"] == family
                      and qa["task"] == task
                      and qa["manifest_sha256"] == panel["manifest_sha256"]
                      and qa["row_sha256"] == item["row_sha256"]
                      and qa["ready_sha256"] == item["ready_sha256"]
                      and qa["video_sha256"] == ready["video_sha256"],
                      "Per-ID source/row/ready binding differs")
            g.require(qa["dimensions_wh"] == [768, 512] and qa["decoded_frames"] == 180
                      and qa["codec"] == "ffv1" and qa["pixel_format"] == "gbrp16le"
                      and qa["fps"] == 60 and qa["max_code_error"] == 0
                      and qa["max_PTS_error_s"] <= .001, "Unsupported source geometry")
            key = _template_key(task, row["prompt"])
            g.require(key in templates, "Exact prompt has no qualified control template")
            stream = qa["decoded_stream_sha256"]
            g.require(len(stream) == 64 and ready.get("stream_sha256", stream) == stream,
                      "Decoded stream identity differs")
            item.update(source_family=family, template_key=key, source_stream_sha256=stream,
                        source_audit_identity=qa["identity"])
            items[sid] = item
    g.require(set(items) == set(audit)
              and dict(Counter(x["task"] for x in items.values())) == expected,
              "Global ID coverage differs")
    g.require(all(len(deep_ids[t]) >= 2 for t in expected),
              "Each task needs at least two already qualified deep IDs")
    return {"schema": SCHEMA, "counts": expected, "items": items,
            "templates": templates, "qualified_deep_ids": deep_ids,
            "CPU_reference_value_sha256": g.digest(cpu_reference),
            "source_families": source_families,
            "source_audit_completion": {"unique_IDs": completion["unique_IDs"],
                "decoded_frames": completion["decoded_frames"]},
            "CPU_processor_calls": 0, "GPU_model_calls": 0,
            "production_runner_implemented": False}


class CatalogReferences(Mapping):
    """Small templates plus per-ID stream identity, without duplicated token lists."""

    def __init__(self, catalog, cpu_reference):
        g.require(catalog["schema"] == SCHEMA, "Catalog schema differs")
        g.require(catalog["CPU_reference_value_sha256"] == g.digest(cpu_reference),
                  "Qualified CPU reference changed")
        self.catalog, self.reference = catalog, cpu_reference

    def __iter__(self):
        return iter(self.catalog["items"])

    def __len__(self):
        return len(self.catalog["items"])

    def __getitem__(self, sid):
        item = self.catalog["items"][sid]
        # Return separate containers: generation/receipt code must not mutate a
        # template shared by later trials.
        result = deepcopy(self.catalog["templates"][item["template_key"]]["controls"])
        result["source_stream_sha256"] = item["source_stream_sha256"]
        return result

    def deep(self, sid):
        g.require(sid in self.reference["rows"], "No existing deep reference for ID")
        result = deepcopy(self.reference["rows"][sid])
        g.require(all(result[k] == v for k, v in self[sid].items()),
                  "Existing deep reference does not match production catalog")
        return result

    def initial_order(self, task, incomplete_ids):
        ids = list(incomplete_ids)
        g.require(len(ids) == len(set(ids))
                  and all(self.catalog["items"][sid]["task"] == task for sid in ids),
                  "Task allocation contains wrong or duplicate IDs")
        first = [sid for sid in self.catalog["qualified_deep_ids"][task] if sid in ids][:2]
        g.require(len(first) == 2, "New engine needs two uncompleted qualified IDs")
        return first + [sid for sid in ids if sid not in first]

    def roots_for(self, sid, roots_by_family):
        item = self.catalog["items"][sid]
        root = Path(roots_by_family[item["source_family"]]).resolve()
        relative = item["ready"].get("video_relative", item["ready"].get("video_path"))
        g.require((root / relative).resolve().is_relative_to(root), "Source escaped family root")
        return {item["task"]: str(root)}


class DispatchLedger:
    """Control-host authority only. Claimed/uncertain IDs are never auto-reissued.

    This CPU transaction lock is independent of, and cannot replace, the actual
    native physical GPU lease required by the eventual worker.
    """

    def __init__(self, path, catalog):
        self.path = Path(path)
        self.lock = self.path.with_suffix(self.path.suffix + ".lock")
        self.catalog = catalog
        self.catalog_sha256 = g.digest(catalog)

    @contextmanager
    def _state(self):
        with self.lock.open("a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                state = g.read(self.path)
                g.require(state["catalog_sha256"] == self.catalog_sha256, "Ledger catalog changed")
            else:
                state = {"schema": "gemma-production-dispatch-v1",
                         "catalog_sha256": self.catalog_sha256, "revision": 0,
                         "assignments": {}, "batches": {}}
            yield state
            # The parent process alone publishes batches after this fsynced
            # transaction succeeds. No remote worker can allocate new IDs.
            state["revision"] += 1
            temporary = self.path.with_name(self.path.name + ".transaction-tmp")
            with temporary.open("x") as f:
                json.dump(state, f, sort_keys=True, separators=(",", ":"))
                f.write("\n"); f.flush(); os.fsync(f.fileno())
            os.replace(temporary, self.path)
            directory = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)

    def claim(self, worker, ids, runtime_sha256):
        g.require(re.fullmatch(r"[A-Za-z0-9_.-]+", worker or "")
                  and re.fullmatch(r"[0-9a-f]{64}", runtime_sha256 or ""),
                  "Worker/runtime identity is invalid")
        ids = list(ids)
        g.require(ids and len(ids) == len(set(ids)), "Empty or duplicate batch")
        with self._state() as state:
            registration = state.get("workers", {}).get(worker)
            if registration:
                g.require(not registration.get("closed") and registration["runtime_sha256"] == runtime_sha256,
                          "Worker is closed or runtime changed")
            g.require(all(sid in self.catalog["items"] and sid not in state["assignments"]
                          for sid in ids), "ID unknown or already owned; no implicit retry")
            batch = {"worker": worker, "runtime_sha256": runtime_sha256,
                     "catalog_sha256": state["catalog_sha256"],
                     "sequence": len(state["batches"]), "sample_ids": ids,
                     "worker_sequence": sum(b["worker"] == worker for b in state["batches"].values()),
                     "worker_token": registration["worker_token"] if registration else None,
                     "profile_sha256": registration["profile_sha256"] if registration else None}
            batch_id = g.digest(batch)
            state["batches"][batch_id] = batch
            for sid in ids:
                state["assignments"][sid] = {"worker": worker, "batch_id": batch_id,
                                              "status": "ASSIGNED"}
            return {"batch_id": batch_id, **batch}

    def record(self, sid, worker, batch_id, status, evidence_sha256):
        return self.record_many(worker, batch_id, [{"sample_id": sid, "status": status,
                                                   "evidence_sha256": evidence_sha256}])

    def record_many(self, worker, batch_id, outcomes):
        """Import one verified result batch in one durable authority transaction."""
        g.require(outcomes and len({x["sample_id"] for x in outcomes}) == len(outcomes),
                  "Empty or duplicate outcome batch")
        with self._state() as state:
            for outcome in outcomes:
                g.require(outcome["status"] in ("COMPLETE", "UNCERTAIN")
                          and re.fullmatch(r"[0-9a-f]{64}", outcome["evidence_sha256"] or ""),
                          "Invalid outcome evidence")
                item = state["assignments"].get(outcome["sample_id"])
                g.require(item and item["worker"] == worker and item["batch_id"] == batch_id
                          and item["status"] == "ASSIGNED", "Outcome ownership/state differs")
            for outcome in outcomes:
                state["assignments"][outcome["sample_id"]].update(
                    status=outcome["status"], evidence_sha256=outcome["evidence_sha256"])

    def register(self, profile):
        value = {"runtime_sha256": profile["execution_signature"], "profile_sha256": g.digest(profile),
                 "worker_token": profile["worker_token"], "closed": False}
        with self._state() as state:
            workers = state.setdefault("workers", {})
            old = workers.get(profile["worker"])
            g.require(old is None or {**old, "closed": False} == value, "Worker name already has another owner/profile")
            if old is None: workers[profile["worker"]] = value

    def close_worker(self, profile):
        with self._state() as state:
            worker = state.get("workers", {}).get(profile["worker"])
            g.require(worker and worker["profile_sha256"] == g.digest(profile), "Unknown worker owner")
            worker["closed"] = True
            batches = sorted(((key, b) for key, b in state["batches"].items() if b["worker"] == profile["worker"]),
                             key=lambda pair: pair[1]["worker_sequence"])
            return {"event": "DONE", "worker": profile["worker"], "worker_token": profile["worker_token"],
                    "profile_sha256": g.digest(profile), "runtime_sha256": profile["execution_signature"],
                    "catalog_sha256": state["catalog_sha256"], "batch_ids": [key for key, _ in batches],
                    "sample_ids": [sid for _, batch in batches for sid in batch["sample_ids"]]}


def seal_inputs(args):
    config, completion = g.read(args.panels), g.read(args.completion)
    audit_path, ref_path, identity_path = map(Path, (args.audit, args.reference, args.identity))
    g.require(g.file_sha256(audit_path) == completion["files"]["per-id.jsonl"]["sha256"],
              "Completed source audit journal changed")
    panels = []
    for p in config["panels"]:
        root = Path(p["root"])
        g.require(g.file_sha256(root / "manifest.json") == p["manifest_sha256"],
                  "Frozen manifest changed")
        manifest = g.read(root / "manifest.json")
        rows = manifest["samples"] if p["task_family"] == "T1" else manifest["rows"]
        readies = ([g.read(v) for v in sorted((root / "ready").glob("*.json"))]
                   if p["task_family"] == "T1" else _lines(root / "ready.jsonl"))
        panels.append({**p, "rows": rows, "readies": readies})
    result = build_catalog(panels, _lines(audit_path), completion, g.read(ref_path),
                           g.read(identity_path)["profile"]["panel"])
    result["evidence_files"] = {"source_audit": g.file_sha256(audit_path),
        "source_audit_completion": g.file_sha256(args.completion),
        "CPU_reference": g.file_sha256(ref_path), "qualified_identity": g.file_sha256(identity_path)}
    _write_new(args.output, result)
    return {"status": "INPUT_CATALOG_SEALED_CPU_ONLY", "counts": result["counts"],
            "catalog_sha256": g.digest(result), "file_sha256": g.file_sha256(args.output),
            "templates": len(result["templates"]), "CPU_processor_calls": 0,
            "GPU_model_calls": 0, "production_runner_implemented": False}


def descriptor(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": g.file_sha256(path)}


def verified(value):
    g.require(g.file_sha256(value["path"]) == value["sha256"], "Sealed evidence file changed")
    return g.read(value["path"])


def derive_runner():
    """Auditably derive the frozen loop; never maintain a copied second loop."""
    g.require(g.file_sha256(g.__file__) == BASE_SHA, "Frozen runner source changed")
    original = inspect.getsource(g.run)
    source, changes = original, []
    replacements = [
        ('profile, reference, admission = read(args.profile), read(args.reference), read(args.admission)',
         'profile, reference, admission = _entry_inputs(args)'),
        ('profile["schema"] == SCHEMA', 'profile["schema"] == ENTRY_SCHEMA'),
        ('items = validate_items(profile["panel"]["items"])', 'items = _entry_items(profile)'),
        ('require(len(items) == profile["maximum_model_calls"] in (20, 48), "Finite request budget changed")',
         '_entry_budget(profile, items)'),
        ('require(reference["panel_sha256"] == digest(profile["panel"])\n        and set(reference["rows"]) == {x["sample_id"] for x in items}, "CPU reference panel differs")',
         '_entry_reference_gate(profile, reference)'),
        ('output.mkdir(parents=True)', 'output.mkdir(parents=True)\n    _entry_output(output)'),
        ('atomic_json(output / "load.json", load)', 'atomic_json(output / "load.json", load)\n            _entry_loaded(output)'),
        ('prepare(session, item, runtime["asset_roots"], streamlined, deep)',
         '_entry_prepare(session, item, streamlined, deep)'),
        ('verify_input(receipt, reference["rows"][sid], deep)', '_entry_verify_input(receipt, sid, deep)'),
        ('                prepared = time.monotonic()', '                _entry_intent(item)\n                prepared = time.monotonic()'),
        ('append(output / "raw.jsonl", raw)', 'raw = _entry_raw(raw)\n                append(output / "raw.jsonl", raw)'),
        ('validate_gpu(observation, reference["rows"][sid], deep)', 'validate_gpu(observation, _entry_reference(sid, deep), deep)'),
        ('append(output / "events.jsonl", {"event": "TECHNICAL_PASS", **item_result})',
         'append(output / "events.jsonl", {"event": "TECHNICAL_PASS", **item_result})\n                _entry_commit(item_result, started)'),
        ('            atomic_json(output / "result.json", {"status": "GENERATION_COMPLETE"',
         '            _entry_final(records)\n            atomic_json(output / "result.json", {"status": "GENERATION_COMPLETE"'),
        ('completed["status"] = "COMPLETE"', 'completed["status"] = _entry_status()'),
        ('    except BaseException as exc:\n        atomic_json(output / "failure.json"',
         '    except BaseException as exc:\n        _entry_failure(exc)\n        atomic_json(output / "failure.json"'),
        ('return {"status": "COMPLETE", "calls": len(records), "output": str(output)}',
         'return {"status": _entry_status(), "calls": len(records), "output": str(output)}'),
    ]
    for old, new in replacements:
        g.require(source.count(old) == 1, "Persistent runner anchor differs: " + old[:70])
        source = source.replace(old, new)
        changes.append({"before": old, "after": new})
    before, after = ast.parse(original), ast.parse(source)
    protected = {"engine.generate", "SamplingParams", "StructuredOutputsParams", "LLM", "ProcessorSession",
                 "physical_lease", "cleanup.callback", "time.monotonic", "_install_adapter"}
    calls = lambda tree: [ast.dump(n, include_attributes=False) for n in ast.walk(tree)
                         if isinstance(n, ast.Call) and ast.unparse(n.func) in protected]
    g.require(calls(before) == calls(after), "Generation, timers, or lifecycle changed")
    gc_calls = [n for n in ast.walk(after) if isinstance(n, ast.Call) and ast.unparse(n.func) == "gc.collect"]
    g.require(len(gc_calls) == 1, "Unique derived GC call missing")
    return {"source": source, "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "gc_line": gc_calls[0].lineno, "changes": changes, "base_sha256": BASE_SHA}


def code_fingerprint(code):
    """Marshal reference flags vary with object interning; hash code fields."""
    def encode(value):
        if isinstance(value, CodeType):
            fields = ("co_code", "co_consts", "co_names", "co_varnames", "co_freevars", "co_cellvars", "co_argcount",
                      "co_posonlyargcount", "co_kwonlyargcount", "co_nlocals", "co_stacksize", "co_flags",
                      "co_filename", "co_name", "co_qualname", "co_firstlineno", "co_linetable", "co_exceptiontable")
            return {name: encode(getattr(value, name)) for name in fields}
        if isinstance(value, bytes): return {"bytes": value.hex()}
        if isinstance(value, tuple): return {"tuple": [encode(v) for v in value]}
        if isinstance(value, frozenset): return {"frozenset": sorted((encode(v) for v in value), key=g.digest)}
        g.require(value is None or type(value) in (str, int, float, bool), "Unsupported executable constant")
        return {"type": type(value).__name__, "value": value}
    return g.digest({"python": list(sys.version_info[:3]), "code": encode(code)})


@contextmanager
def numerical_context(profile, installed_sdk=False, install=False):
    """Validate the historical numerical profile; only optionally install it."""
    from . import gemma_cpu_copy as cpu
    from . import gemma_prefill as prefill
    local = profile["kernel"].get("local256_prefill", {})
    module = local.get("module")
    with ExitStack() as stack:
        if module == "classic_precision.gemma_tile64":
            from . import gemma_tile64 as tile
            stack.enter_context(tile.specialization(local.get("local_num_warps")))
            prefill.validate_profile(profile, installed_sdk=installed_sdk)
        elif module == "classic_precision.gemma_prefill":
            prefill.validate_profile(profile, installed_sdk=installed_sdk)
        else:
            g.require(not local, "Unsupported selected kernel module")
            cpu.validate_profile(profile, installed_sdk=installed_sdk)
            g.require(profile["kernel"] == prefill.base_kernel(), "Only qualified binary/global512 M32 is supported")
        if install and local: stack.enter_context(prefill.installed_overlay())
        if install and profile["cpu_execution_overlay"]["placement"] == "memory28":
            stack.enter_context(cpu.m.installed_overlay())
        yield cpu


def execution_signature(profile):
    q = profile["qualified_runtime"]
    cpu = {k: v for k, v in q["cpu_execution_overlay"].items()
           if k not in ("base_profile_sha256", "baseline_profile_sha256", "gc_target", "generation_timing_shutdown")}
    return g.digest({"entry": profile["entry"], "cpu_algorithm": cpu, "kernel": q["kernel"],
        "settings": {k: v for k, v in q["engine_settings"].items() if k != "model"},
        "implementation": q["implementation"], "runtime_sources": q["runtime_sources"],
        "checkpoint_assets": q["checkpoint_assets"], "variant": q["variant"],
        "catalog": profile["catalog"]["sha256"], "reference_policy": REFERENCE_POLICY,
        "references": {k: v["sha256"] for k, v in profile["references"].items()}})


def validate_profile(profile, installed_sdk=False):
    from . import gemma_memory as memory
    derived = derive_runner()
    g.require(profile["schema"] == ENTRY_SCHEMA and profile["entry"] == {
        "source_sha256": g.file_sha256(__file__), "derived_sha256": derived["sha256"],
        "base_sha256": BASE_SHA, "gc_line": derived["gc_line"], "gc_cadence": 8,
        "gc_target": "derived runner f_code identity plus its unique explicit GC line",
        "reference_policy": REFERENCE_POLICY}, "Persistent entry source/GC/reference policy changed")
    q = profile["qualified_runtime"]
    with numerical_context(q, installed_sdk): pass
    for key in ("variant", "plan_sha256", "runtime", "engine_settings", "kernel", "implementation", "runtime_sources", "checkpoint_assets"):
        g.require(profile[key] == q[key], "Numerical execution differs from selected profile: " + key)
    g.require(q["cpu_execution_overlay"]["gc_cadence"] == 8 and profile["first_deep_calls"] == 2
        and profile["automatic_retry"] is False and profile["execution_signature"] == execution_signature(profile),
        "Execution signature or retry/deep policy differs")
    g.require(profile["mode"] in ("20", "48", "production")
        and profile["maximum_model_calls"] == (9360 if profile["mode"] == "production" else int(profile["mode"]))
        and 0 < profile["inbox_timeout_seconds"] <= 600, "Finite worker scope differs")
    g.require(profile["catalog"]["sha256"] == CATALOG_SHA
        and {k: v["sha256"] for k, v in profile["references"].items()} == {str(k): v for k, v in memory.REFERENCE_SHA.items()},
        "Use the existing sealed catalog and both CPU references")
    return profile


def seal_worker(args):
    from . import gemma_memory as memory
    q, catalog = g.read(args.runtime_profile), g.read(args.catalog)
    g.require(g.file_sha256(args.catalog) == CATALOG_SHA, "Existing production catalog changed")
    refs = {"20": descriptor(args.reference20), "48": descriptor(args.reference48)}
    g.require(all(refs[k]["sha256"] == memory.REFERENCE_SHA[int(k)] for k in refs), "Use the existing frozen CPU references")
    catalog_refs = CatalogReferences(catalog, g.read(args.reference20))
    roots = g.read(args.source_roots)
    g.require(set(roots) == set(catalog["source_families"]) and all(Path(r).is_absolute() for r in roots.values()), "Source-family roots differ")
    if args.mode == "production":
        g.require(catalog["counts"] == EXPECTED and len(catalog_refs) == 9360, "Full production universe differs")
        panel = {"items": [], "scope": "finite catalog universe; actual per-worker order is the authoritative inbox"}
    else:
        panel = g.read(args.panel) if args.panel else q["panel"]
        reference = g.read(refs[args.mode]["path"])
        g.require(len(g.validate_items(panel["items"])) == int(args.mode)
            and g.digest(panel) == reference["panel_sha256"], "Fixed validation panel differs")
        for item in panel["items"]:
            source = catalog["items"][item["sample_id"]]
            g.require(item["row_sha256"] == source["row_sha256"] and item["ready_sha256"] == source["ready_sha256"], "Control input differs from catalog")
    d = derive_runner()
    profile = {key: deepcopy(q[key]) for key in ("variant", "plan_sha256", "runtime", "engine_settings", "kernel", "implementation", "runtime_sources", "checkpoint_assets")}
    profile.update(schema=ENTRY_SCHEMA, name=args.name, mode=args.mode, panel=panel,
        qualified_runtime=q, qualified_runtime_role="historical numerical/CPU algorithm anchor; this entry and GC binding are explicit below",
        qualified_runtime_file=descriptor(args.runtime_profile), catalog=descriptor(args.catalog), references=refs,
        source_roots=roots, worker=args.worker, worker_token=uuid.uuid4().hex, inbox=str(Path(args.inbox).resolve()),
        initial_task=args.initial_task, inbox_timeout_seconds=args.inbox_timeout_seconds,
        maximum_model_calls=9360 if args.mode == "production" else int(args.mode),
        first_deep_calls=2, automatic_retry=False,
        entry={"source_sha256": g.file_sha256(__file__), "derived_sha256": d["sha256"],
               "base_sha256": BASE_SHA, "gc_line": d["gc_line"], "gc_cadence": 8,
               "gc_target": "derived runner f_code identity plus its unique explicit GC line", "reference_policy": REFERENCE_POLICY})
    g.require(re.fullmatch(r"[A-Za-z0-9_.-]+", profile["worker"] or "") and profile["initial_task"] in EXPECTED, "Worker/task identity invalid")
    profile["execution_signature"] = execution_signature(profile)
    validate_profile(profile)
    _write_new(args.output, profile)
    return {"status": "WORKER_PROFILE_SEALED", "mode": args.mode, "execution_signature": profile["execution_signature"],
            "profile_sha256": g.digest(profile), "GPU_model_calls": 0, "production_authorized": False}


class Inbox:
    def __init__(self, profile, catalog, refs):
        self.profile, self.catalog, self.refs = profile, catalog, refs
        self.directory = Path(profile["inbox"])
        self.catalog_sha, self.profile_sha = g.digest(catalog), g.digest(profile)
        self.ids, self.batches, self.by_id, self.done = [], [], {}, None

    def __len__(self): return self.profile["maximum_model_calls"]

    def check_owner(self, value):
        for key, expected in (("worker", self.profile["worker"]), ("worker_token", self.profile["worker_token"]),
                              ("profile_sha256", self.profile_sha), ("runtime_sha256", self.profile["execution_signature"]),
                              ("catalog_sha256", self.catalog_sha)):
            g.require(value.get(key) == expected, "Inbox ownership/signature differs: " + key)

    def __iter__(self):
        deadline = time.monotonic() + self.profile["inbox_timeout_seconds"]
        fixed = [item["sample_id"] for item in self.profile["panel"]["items"]]
        while True:
            path = self.directory / f"batch-{len(self.batches):06d}.json"
            if path.exists():
                batch = g.read(path); self.check_owner(batch)
                bid = batch["batch_id"]
                g.require(bid == g.digest({k: v for k, v in batch.items() if k != "batch_id"})
                    and batch["worker_sequence"] == len(self.batches) and bid not in self.batches,
                    "Inbox batch identity/order differs")
                ids = batch["sample_ids"]
                g.require(ids and len(ids) == len(set(ids)) and not set(ids) & set(self.ids)
                    and set(ids) <= set(self.catalog["items"]) and len(self.ids) + len(ids) <= len(self), "Unknown, duplicate or over-budget IDs")
                if fixed: g.require(ids == fixed[len(self.ids):len(self.ids) + len(ids)], "Validation order differs")
                self.batches.append(bid)
                for sid in ids:
                    if len(self.ids) < 2:
                        g.require(self.catalog["items"][sid]["task"] == self.profile["initial_task"], "Initial task differs")
                        self.refs.expected(sid, True)
                    self.ids.append(sid); self.by_id[sid] = bid
                    yield self.catalog["items"][sid]
                deadline = time.monotonic() + self.profile["inbox_timeout_seconds"]
                continue
            elif (self.directory / "DONE.json").exists():
                done = g.read(self.directory / "DONE.json"); self.check_owner(done)
                g.require(done.get("event") == "DONE", "Invalid inbox termination")
                if len(done["batch_ids"]) == len(self.batches):
                    g.require(done["batch_ids"] == self.batches and done["sample_ids"] == self.ids
                        and (not fixed or self.ids == fixed), "DONE omitted/reordered allocated IDs")
                    self.done = done
                    return
                g.require(len(done["batch_ids"]) > len(self.batches), "DONE precedes consumed work")
            g.require(time.monotonic() < deadline, "Finite inbox wait expired; no implicit retry")
            time.sleep(.1)


class WorkerReferences:
    def __init__(self, profile, catalog, reference20, reference48):
        self.profile, self.catalog = profile, CatalogReferences(catalog, reference20)
        self.control = {"20": reference20, "48": reference48}.get(profile["mode"])
        self.cached = None

    def expected(self, sid, deep):
        if self.cached and self.cached[:2] == (sid, deep): return self.cached[2]
        ordinary = self.catalog[sid]
        if deep:
            result = deepcopy(self.control["rows"][sid]) if self.control else self.catalog.deep(sid)
            g.require(all(result[k] == v for k, v in ordinary.items()), "Deep reference/template differs")
        else: result = ordinary
        self.cached = (sid, deep, result)
        return result

    def verify(self, receipt, sid, deep):
        g.verify_input(receipt, self.expected(sid, deep), deep)
        if self.control:
            g.verify_input(receipt, self.control["rows"][sid], deep)


def byte_range(path, start, end):
    g.require(type(start) is type(end) is int and 0 <= start <= end <= Path(path).stat().st_size, "Invalid proof byte range")
    with Path(path).open("rb") as f: f.seek(start); value = f.read(end - start)
    g.require(not value or value.endswith(b"\n"), "Partial journal record")
    return value


class Worker:
    def __init__(self, profile, catalog, reference20, reference48, admission):
        self.profile, self.catalog, self.admission = profile, catalog, admission
        self.refs = WorkerReferences(profile, catalog, reference20, reference48)
        self.inbox = Inbox(profile, catalog, self.refs)
        self.output, self.pending, self.proofs = None, None, []
        self.service_times = []

    def inputs(self, args): return self.profile, {}, self.admission

    def budget(self, profile, items):
        g.require(len(items) == profile["maximum_model_calls"] and items is self.inbox, "Finite worker budget changed")

    def reference_gate(self, profile, reference):
        g.require(profile["entry"]["reference_policy"] == REFERENCE_POLICY, "Reference policy changed")

    def start_output(self, output):
        self.output = Path(output)
        d = derive_runner()
        with (self.output / DERIVED_FILE).open("x") as f: f.write(d["source"])
        _write_new(self.output / "entry-source.json", {k: v for k, v in d.items() if k != "source"})

    def loaded(self, output):
        self.identity_sha = g.file_sha256(self.output / "identity.json")
        self.load_sha = g.file_sha256(self.output / "load.json")
        g.append(self.output / "events.jsonl", {"event": "ENTRY_LOADED", "execution_signature": self.profile["execution_signature"],
            "worker": self.profile["worker"], "worker_token": self.profile["worker_token"]})
        self.prefix_end = (self.output / "events.jsonl").stat().st_size
        self.prefix_sha = hashlib.sha256(byte_range(self.output / "events.jsonl", 0, self.prefix_end)).hexdigest()

    def prepare(self, session, item, streamlined, deep):
        return g.prepare(session, item, self.refs.catalog.roots_for(item["sample_id"], self.profile["source_roots"]), streamlined, deep)

    def intent(self, item):
        g.require(self.pending is None, "Previous attempt has no terminal proof")
        raw = self.output / "raw.jsonl"
        sid = item["sample_id"]
        self.pending = {"sample_id": sid, "attempt_id": uuid.uuid4().hex, "worker": self.profile["worker"],
            "batch_id": self.inbox.by_id[sid], "ordinal": len(self.proofs), "row_sha256": item["row_sha256"],
            "ready_sha256": item["ready_sha256"], "raw_start": raw.stat().st_size if raw.exists() else 0,
            "event_start": (self.output / "events.jsonl").stat().st_size}
        g.append(self.output / "events.jsonl", {"event": "GENERATE_INTENT", **self.pending})

    def raw(self, value):
        g.require(self.pending and value["sample_id"] == self.pending["sample_id"], "Output attempt identity differs")
        return {**value, "entry_attempt_id": self.pending["attempt_id"], "measurement_role": self.profile["mode"]}

    def commit(self, result, started):
        g.require(self.pending and result["sample_id"] == self.pending["sample_id"], "Completion attempt differs")
        proof = {**self.pending, "status": "COMPLETE", "identity_sha256": self.identity_sha, "load_sha256": self.load_sha,
            "prefix_end": self.prefix_end, "prefix_sha256": self.prefix_sha}
        for stem, filename in (("raw", "raw.jsonl"), ("event", "events.jsonl")):
            path = self.output / filename; proof[stem + "_end"] = path.stat().st_size
            proof[stem + "_sha256"] = hashlib.sha256(byte_range(path, proof[stem + "_start"], proof[stem + "_end"])).hexdigest()
        g.append(self.output / "calls.jsonl", proof)
        self.service_times.append({"sample_id": result["sample_id"], "deep": result["deep"], "status": "PASS",
            "full_request_seconds": time.monotonic() - started})
        self.proofs.append(proof); self.pending = None

    def final(self, records):
        g.require(self.inbox.done and len(records) >= 2 and [r["sample_id"] for r in records] == self.inbox.ids
            == [p["sample_id"] for p in self.proofs] and self.pending is None, "Worker/DONE/call proof coverage incomplete")

    def status(self): return "WORKER_COMPLETE" if self.profile["mode"] == "production" else "COMPLETE"

    def failure(self, error):
        if self.output and self.pending:
            g.append(self.output / "events.jsonl", {"event": "ATTEMPT_UNCERTAIN", **self.pending,
                "status": "UNCERTAIN", "error_type": type(error).__name__, "error": str(error), "automatic_retry": False})

    def compile(self, namespace=None):
        d = derive_runner(); env = dict(vars(g)); env.update(namespace or {})
        env.update(ENTRY_SCHEMA=ENTRY_SCHEMA, _entry_inputs=self.inputs, _entry_items=lambda profile: self.inbox,
            _entry_budget=self.budget, _entry_reference_gate=self.reference_gate, _entry_output=self.start_output,
            _entry_loaded=self.loaded, _entry_prepare=self.prepare, _entry_verify_input=self.refs.verify,
            _entry_reference=self.refs.expected, _entry_intent=self.intent, _entry_raw=self.raw,
            _entry_commit=self.commit, _entry_final=self.final, _entry_status=self.status, _entry_failure=self.failure)
        exec(compile(d["source"], "<gemma-persistent-entry-v1>", "exec"), env)
        return env["run"], d


def _profile_inputs(profile):
    return verified(profile["catalog"]), verified(profile["references"]["20"]), verified(profile["references"]["48"])


def speed_acceptance(timings, result_hashes, authorization=None):
    """A sealed speed tradeoff changes admission only, never model execution."""
    g.require(len(timings) == 3 and all(t["ordinary_n"] == 18 for t in timings),
              "Three complete common-panel speed observations required")
    if authorization is None:
        g.require(all(t["p50_seconds"] <= 25 and t["p90_seconds"] <= 27.5 for t in timings),
                  "Original speed threshold not met; explicit revised authorization required")
        return
    g.require(authorization["schema"] == "gemma-C3-speed-acceptance-v1"
              and authorization["decision"] == "PROCEED_WITH_COMMON_C3_AND_STRICT_REPLAY"
              and authorization["max_card_p50_seconds"] == 29
              and authorization["max_card_p90_seconds"] == 30
              and authorization["max_effective_seconds"] == 26.5
              and len(set(result_hashes)) == 3
              and set(authorization["qualified_result_sha256"]) == set(result_hashes),
              "Speed authorization or its exact observed results differ")
    g.require(all(0 < t["p50_seconds"] <= 29 and 0 < t["p90_seconds"] <= 30 for t in timings)
              and 3 / sum(1 / t["p50_seconds"] for t in timings) <= 26.5,
              "Explicit common-C3 speed envelope not met")


def admission_gate(profile, admission):
    if profile["mode"] != "production": return
    qualification = verified(admission["formal_qualification"])
    g.require(qualification["status"] == "MATCH" and qualification["six_observation_consistency_pass"]
        and qualification["execution_signature"] == profile["execution_signature"]
        and qualification["source_sha256"] == g.file_sha256(__file__), "Final entry six-cold qualification missing")
    entry = verified(admission["entry_invariance"])
    g.require(entry["status"] == "MATCH" and entry["unique_ids"] == 20
        and entry["execution_signature"] == profile["execution_signature"]
        and len(entry["entry_runs"]) == 1 and entry["entry_runs"][0]["mode"] == "20",
        "One final-entry20 exact-output bridge is required")
    speed_runs = admission["qualified_speed_runs"]
    g.require(len(speed_runs) == 3, "Existing three-card speed qualification is required")
    cards, policies, timings, result_hashes = set(), set(), [], []
    for files in speed_runs:
        identity, load, result, exited = (verified(files[key]) for key in ("identity", "load", "result", "exit"))
        q = identity["profile"]
        with numerical_context(q): pass
        g.require(_numerical_signature(q) == _numerical_signature(profile["qualified_runtime"])
            and identity["profile_sha256"] == result["profile_sha256"] == g.digest(q)
            and result["status"] == "COMPLETE" and result["calls"] == 20
            and exited["status"] == "COMPLETE" and exited["owned_engine_workers_exited"] and exited["lease_released_after_engine_exit"],
            "Existing speed runtime or engine closure differs")
        timing = result["timing"]
        timings.append(timing)
        result_hashes.append(files["result"]["sha256"])
        cards.add((identity["host"], identity["lease"]["device"]))
        policies.add(g.digest(load["placement"][0]["torch_policy"]))
    g.require(len(cards) == 3 and len(policies) == 1, "Three distinct cards with the same Torch policy required")
    speed_acceptance(timings, result_hashes,
        verified(admission["speed_acceptance"]) if admission.get("speed_acceptance") else None)
    runtime_match = verified(admission["runtime_equivalence"])
    g.require(runtime_match["status"] == "MATCH" and runtime_match["unique_ids"] == 20 and runtime_match["observations"] >= 3,
        "Three-card runtime full-token equivalence is missing")
    for report in [qualification, entry]:
        for run in report["entry_runs"]:
            closure = admission["qualification_closures"][run["closure_sha256"]]
            g.require(closure["sha256"] == run["closure_sha256"] and verified(closure)["status"] == "COMPLETE", "Qualified observation closure changed")


def run_worker(args):
    worker_started = time.monotonic()
    profile = validate_profile(g.read(args.profile), installed_sdk=True)
    catalog, ref20, ref48 = _profile_inputs(profile)
    admission = g.read(args.admission); admission_gate(profile, admission)
    inbox = Path(profile["inbox"]); g.require(inbox.is_dir(), "Controller inbox does not exist")
    _write_new(inbox / "worker-owner.json", {"profile_sha256": g.digest(profile), "worker_token": profile["worker_token"],
        "owner": g.process_identity(), "automatic_resume": False})
    worker = Worker(profile, catalog, ref20, ref48, admission)
    call_args = copy(args)
    call_args.reference = profile["references"]["48" if profile["mode"] == "48" else "20"]["path"]
    state, runner, succeeded = None, None, False
    try:
        with numerical_context(profile["qualified_runtime"], installed_sdk=True, install=True) as cpu:
            with cpu.installed_overlay(profile["qualified_runtime"]) as state:
                runner, derived = worker.compile()
                state.code, state.line = runner.__code__, derived["gc_line"]
                result = runner(call_args)
        succeeded = True
        return result
    except BaseException as error:
        if worker.output:
            g.atomic_json(worker.output / "entry-failure.json", {"status": "STOPPED", "error_type": type(error).__name__,
                "error": str(error), "automatic_retry": False})
        raise
    finally:
        if worker.output:
            receipt = {"schema": ENTRY_SCHEMA, "status": "COMPLETE" if succeeded else "STOPPED",
                "execution_signature": profile["execution_signature"], "profile_sha256": g.digest(profile),
                "source_sha256": g.file_sha256(__file__), "done": worker.inbox.done,
                "assigned_ids": worker.inbox.ids, "generation_attempts": len(worker.proofs) + int(worker.pending is not None),
                "worker_wall_seconds": time.monotonic() - worker_started,
                "through_durable_proof": {"rows": worker.service_times, "timing": g.timing_summary(worker.service_times),
                    "scope": "Original start through fsynced calls proof; original full_request_seconds stays unchanged"},
                "gc": None if state is None else {"cadence": state.cadence, "line": state.line,
                    "code_sha256": code_fingerprint(state.code),
                    "bound_to_actual_runner": runner is not None and state.code is runner.__code__,
                    "maximum_pending_ordinary": state.maximum_pending, "rows": state.rows, "final": state.final}}
            _write_new(worker.output / "entry-receipt.json", receipt)
            files = ("identity.json", "load.json", "raw.jsonl", "events.jsonl", "calls.jsonl", "exit.json", "result.json", "entry-receipt.json", DERIVED_FILE)
            _write_new(worker.output / "entry-closure.json", {"status": receipt["status"],
                "files": {name: g.file_sha256(worker.output / name) for name in files if (worker.output / name).is_file()}})


def _complete_journal_records(blob, allow_incomplete=False):
    """A hard exit may leave one partial tail; retain it, never infer success."""
    lines = blob.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        g.require(allow_incomplete, "Partial journal record")
        lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def verify_worker(path, allow_incomplete=False):
    root = Path(path); identity = g.read(root / "identity.json")
    profile = validate_profile(identity["profile"])
    g.require(identity["profile_sha256"] == g.digest(profile), "Worker profile digest changed")
    has_closure = (root / "entry-closure.json").is_file() and (root / "entry-receipt.json").is_file()
    g.require(has_closure or allow_incomplete, "Worker has no final closure")
    if has_closure:
        closure, receipt = g.read(root / "entry-closure.json"), g.read(root / "entry-receipt.json")
    else:
        # These are current observed hashes, not a fabricated exit receipt.
        closure = {"status": "NO_FINAL_CLOSURE", "files": {
            name: g.file_sha256(root / name) for name in ("identity.json", "load.json")}}
        receipt = None
    for name, sha in closure["files"].items():
        g.require(name in {"identity.json", "load.json", "raw.jsonl", "events.jsonl", "calls.jsonl", "exit.json", "result.json", "entry-receipt.json", DERIVED_FILE}
            and g.file_sha256(root / name) == sha, "Worker closure bytes changed")
    g.require(g.file_sha256(root / DERIVED_FILE) == profile["entry"]["derived_sha256"], "Entry provenance differs")
    if receipt:
        g.require(receipt["profile_sha256"] == g.digest(profile)
            and receipt["execution_signature"] == profile["execution_signature"], "Entry provenance differs")
    proofs = _complete_journal_records((root / "calls.jsonl").read_bytes(), allow_incomplete) if (root / "calls.jsonl").exists() else []
    ids, attempts, previous_raw, previous_event = [], set(), 0, None
    for proof in proofs:
        sid = proof["sample_id"]
        g.require(sid not in ids and proof["attempt_id"] not in attempts and proof["ordinal"] == len(ids)
            and proof["worker"] == profile["worker"] and proof["status"] == "COMPLETE"
            and proof["identity_sha256"] == closure["files"]["identity.json"] and proof["load_sha256"] == closure["files"]["load.json"], "Duplicate/invalid attempt owner or order")
        prefix = byte_range(root / "events.jsonl", 0, proof["prefix_end"])
        g.require(hashlib.sha256(prefix).hexdigest() == proof["prefix_sha256"], "Event prefix proof changed")
        g.require(proof["raw_start"] == previous_raw and proof["event_start"] == (previous_event if previous_event is not None else proof["prefix_end"]), "Journal proof has a gap/overlap")
        blobs = {}
        for stem, filename in (("raw", "raw.jsonl"), ("event", "events.jsonl")):
            blob = byte_range(root / filename, proof[stem + "_start"], proof[stem + "_end"])
            g.require(hashlib.sha256(blob).hexdigest() == proof[stem + "_sha256"], "Call byte proof changed")
            blobs[stem] = [json.loads(line) for line in blob.splitlines() if line]
        raw = blobs["raw"]
        intents = [e for e in blobs["event"] if e.get("event") == "GENERATE_INTENT"]
        passed = [e for e in blobs["event"] if e.get("event") == "TECHNICAL_PASS"]
        g.require(len(raw) == len(intents) == len(passed) == 1 and raw[0]["sample_id"] == intents[0]["sample_id"] == passed[0]["sample_id"] == sid
            and raw[0]["entry_attempt_id"] == intents[0]["attempt_id"] == proof["attempt_id"]
            and all(intents[0].get(key) == proof[key] for key in ("worker", "batch_id", "ordinal", "row_sha256", "ready_sha256", "raw_start", "event_start"))
            and raw[0]["finish_reason"] == "stop" and raw[0]["raw_answer"] in ("LEFT", "RIGHT")
            and raw[0]["output_token_ids"] and passed[0]["status"] == "PASS", "Attempt/output/technical proof differs")
        ids.append(sid); attempts.add(proof["attempt_id"])
        previous_raw, previous_event = proof["raw_end"], proof["event_end"]
    complete = receipt is not None and closure["status"] == receipt["status"] == "COMPLETE"
    if complete:
        result, exit_receipt, gc = g.read(root / "result.json"), g.read(root / "exit.json"), receipt["gc"]
        g.require(not (root / "failure.json").exists() and not (root / "entry-failure.json").exists()
            and result["status"] == ("WORKER_COMPLETE" if profile["mode"] == "production" else "COMPLETE")
            and exit_receipt["status"] == "COMPLETE" and exit_receipt["owned_engine_workers_exited"]
            and exit_receipt["lease_released_after_engine_exit"], "Actual engine/lease completion missing")
        g.require(ids == receipt["assigned_ids"] == receipt["done"]["sample_ids"] == [r["sample_id"] for r in result["rows"]]
            and result["calls"] == receipt["generation_attempts"] == len(ids) >= 2
            and previous_raw == (root / "raw.jsonl").stat().st_size and previous_event == (root / "events.jsonl").stat().st_size,
            "Worker finite coverage or journal tail differs")
        g.require(gc["bound_to_actual_runner"] is True and gc["line"] == profile["entry"]["gc_line"]
            and gc["cadence"] == 8 and gc["maximum_pending_ordinary"] <= 8 and gc["final"]["status"] == "COMPLETE"
            and [r["sample_id"] for r in gc["rows"]] == ids and [r["deep"] for r in gc["rows"]] == [i < 2 for i in range(len(ids))], "Derived GC binding/lifecycle differs")
    g.require(complete or allow_incomplete, "Worker is incomplete/uncertain")
    uncertain = []
    if not complete and (root / "events.jsonl").exists():
        with (root / "events.jsonl").open("rb") as f:
            f.seek(previous_event or 0); tail = f.read()
        intents = [row for row in _complete_journal_records(tail, True) if row.get("event") == "GENERATE_INTENT"]
        g.require(len(intents) <= 1, "Multiple unclosed attempts")
        uncertain = [{"sample_id": x["sample_id"], "batch_id": x["batch_id"], "status": "UNCERTAIN",
                      "evidence_sha256": g.digest(x)} for x in intents]
    return {"profile": profile, "identity": identity, "proofs": proofs, "uncertain": uncertain, "complete": complete,
            "closure_sha256": g.file_sha256(root / "entry-closure.json") if has_closure else None,
            "final_closure_present": has_closure,
            "uncertain_requires_exit_confirmation": bool(uncertain) and not has_closure}


def _verify_numerical_receipts(root, profile, formal):
    from . import gemma_prefill as prefill
    q = profile["qualified_runtime"] if formal else profile
    load = g.read(root / "load.json")
    with numerical_context(q):
        if q["kernel"].get("local256_prefill"):
            prefill._verify_kernel_receipt(root, q, load)
        else:
            base = load["kernel"]
            g.require(len(base) == 1 and base[0]["configuration"] == q["kernel"]
                and base[0]["observed_native_sha256"] == g.NATIVE_SHA
                and base[0]["derived_helpers_sha256"] == "32a56e3cb9501974cf648a3524691e84bd4a8c6dd7ead6c2ecd4b683c7b1ac4f"
                and base[0]["derived_launcher_sha256"] == "a9c0fd9df26aae3919a024de045a6bb59e67011c297f2de87f584acf2bfc855e",
                "C3 actual binary/global512 kernel receipt differs")
        if q["kernel"].get("local256_prefill", {}).get("module") == "classic_precision.gemma_tile64":
            from . import gemma_tile64 as tile
            tile.verify_resources(root, q, load)
    if not formal: prefill._verify_cpu_receipt(root, q)


def _numerical_signature(q):
    return g.digest({"kernel": q["kernel"], "CPU": {k: v for k, v in q["cpu_execution_overlay"].items()
        if k not in ("base_profile_sha256", "baseline_profile_sha256")},
        "settings": {k: v for k, v in q["engine_settings"].items() if k != "model"},
        "implementation": q["implementation"], "runtime_sources": q["runtime_sources"],
        "checkpoint_assets": q["checkpoint_assets"], "variant": q["variant"]})


def compare_workers(paths, purpose="candidate"):
    signatures, code_signatures, entry_runs, baseline_signatures = [], [], [], []
    for path in paths:
        root = Path(path); identity = g.read(root / "identity.json"); profile = identity["profile"]
        formal = profile.get("schema") == ENTRY_SCHEMA
        g.require(formal or purpose == "candidate", "Every cold observation must use the final entry")
        _verify_numerical_receipts(root, profile, formal)
        q = profile["qualified_runtime"] if formal else profile
        baseline_signatures.append(g.digest({"runtime": _numerical_signature(q),
            "torch_policy": g.read(root / "load.json")["placement"][0]["torch_policy"]}))
        if formal:
            verified_run = verify_worker(root)
            g.require(profile["mode"] in ("20", "48"), "Production shards are not a replay panel")
            signatures.append(profile["execution_signature"])
            code_signatures.append(g.read(root / "entry-receipt.json")["gc"]["code_sha256"])
            entry_runs.append({"path": str(root.resolve()), "mode": profile["mode"], "host": identity["host"], "device": identity["lease"]["device"],
                "closure_sha256": verified_run["closure_sha256"], "timing": g.read(root / "result.json")["timing"]})
    g.require(signatures and len(set(signatures)) == len(set(code_signatures)) == len(set(baseline_signatures)) == 1,
        "Final entry, CPU/kernel/placement/source or derived code signatures differ")
    result = g.compare_runs(paths, purpose)
    if result["status"] == "PROFILE_OUTPUT_DIFFERENCE": result["status"] = "DIFFERENT"
    return {**result, "schema": ENTRY_SCHEMA, "source_sha256": g.file_sha256(__file__),
        "execution_signature": signatures[0], "entry_runs": entry_runs, "any_output_difference_disqualifies": True}


def dispatch(args):
    profile = validate_profile(g.read(args.profile)); catalog = verified(profile["catalog"])
    ledger = DispatchLedger(args.ledger, catalog); ledger.register(profile)
    directory = Path(getattr(args, "publish_directory", None) or profile["inbox"])
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == "done":
        value, name = ledger.close_worker(profile), "DONE.json"
    else:
        ids = g.read(args.ids)
        value = ledger.claim(profile["worker"], ids, profile["execution_signature"])
        name = f"batch-{value['worker_sequence']:06d}.json"
    publish_message(directory / name, value)
    return {"status": "PUBLISHED_AFTER_DURABLE_AUTHORITY_COMMIT", "file": str(directory / name), "GPU_model_calls": 0}


def reconcile(args):
    g.require(g.file_sha256(args.catalog) == CATALOG_SHA, "Existing global catalog changed")
    catalog = g.read(args.catalog); ledger = DispatchLedger(args.ledger, catalog)
    reports = [verify_worker(path, allow_incomplete=True) for path in g.read(args.outputs)]
    g.require(not any(r.get("uncertain_requires_exit_confirmation") for r in reports),
        "Unclosed intent without final exit evidence: live/crashed state needs controller confirmation; authority stays unchanged")
    workers = [r["profile"]["worker"] for r in reports]
    g.require(len(workers) == len(set(workers)) and all(r["profile"]["mode"] == "production" for r in reports), "Repeated worker or non-production output")
    g.require(len({r["profile"]["execution_signature"] for r in reports}) == 1, "Production runtime signatures differ")
    outcomes, ids = [], set()
    for report in reports:
        for proof in report["proofs"]:
            outcomes.append((report["profile"]["worker"], {"sample_id": proof["sample_id"], "batch_id": proof["batch_id"],
                "status": "COMPLETE", "evidence_sha256": g.digest(proof)}))
        outcomes.extend((report["profile"]["worker"], x) for x in report["uncertain"])
    state = g.read(args.ledger); pending = {}
    for worker, row in outcomes:
        g.require(row["sample_id"] not in ids, "Duplicate production ID across workers")
        ids.add(row["sample_id"])
        current = state["assignments"].get(row["sample_id"])
        g.require(current and current["worker"] == worker and current["batch_id"] == row["batch_id"], "Authority/output owner differs")
        if current["status"] == "ASSIGNED":
            pending.setdefault((worker, row["batch_id"]), []).append(row)
        else:
            g.require(current["status"] == row["status"] and current["evidence_sha256"] == row["evidence_sha256"], "Terminal evidence cannot be replaced")
    for (worker, bid), rows in pending.items(): ledger.record_many(worker, bid, rows)
    state = g.read(args.ledger)
    complete = catalog["counts"] == EXPECTED and set(state["assignments"]) == set(catalog["items"]) == ids
    complete = complete and all(x["status"] == "COMPLETE" for x in state["assignments"].values()) and all(r["complete"] for r in reports)
    result = {"status": "COMPLETE" if complete else "INCOMPLETE", "expected": EXPECTED,
        "unique_terminal_IDs": len(ids), "statuses": dict(Counter(x["status"] for x in state["assignments"].values())),
        "ledger_sha256": g.file_sha256(args.ledger), "automatic_retry": False}
    _write_new(args.output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("seal-inputs")
    for name in ("panels", "audit", "completion", "reference", "identity", "output"):
        command.add_argument("--" + name, required=True)
    command = commands.add_parser("seal-worker")
    for name in ("runtime-profile", "catalog", "reference20", "reference48", "source-roots", "name", "worker", "inbox", "initial-task", "output"):
        command.add_argument("--" + name, required=True)
    command.add_argument("--mode", choices=("20", "48", "production"), required=True)
    command.add_argument("--panel")
    command.add_argument("--inbox-timeout-seconds", type=float, default=60.)
    for name in ("dispatch", "done"):
        command = commands.add_parser(name)
        command.add_argument("--ledger", required=True); command.add_argument("--profile", required=True)
        command.add_argument("--publish-directory", help="Control-host outbox; external transport must atomically publish the exact bytes to the sealed worker inbox")
        if name == "dispatch": command.add_argument("--ids", required=True)
    command = commands.add_parser("run")
    for name in ("profile", "admission", "output"): command.add_argument("--" + name, required=True)
    command = commands.add_parser("compare")
    command.add_argument("runs", nargs="+"); command.add_argument("--purpose", choices=("candidate", "consistency"), default="candidate")
    command.add_argument("--output", required=True)
    command = commands.add_parser("reconcile")
    for name in ("catalog", "ledger", "outputs", "output"): command.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.command == "seal-inputs": result = seal_inputs(args)
    elif args.command == "seal-worker": result = seal_worker(args)
    elif args.command in ("dispatch", "done"): result = dispatch(args)
    elif args.command == "run": result = run_worker(args)
    elif args.command == "compare":
        result = compare_workers(args.runs, args.purpose); _write_new(args.output, result)
    else: result = reconcile(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
