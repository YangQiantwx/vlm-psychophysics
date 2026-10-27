"""Finite T3 extension using the already qualified official Cosmos session unchanged.

Only allocation, sealing and worker orchestration differ from the 432-ID pilot.
The original OfficialSession, engine settings, input checks and physical lease run
directly; existing pilot T3 answers are references and never dispatched again.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import time

from . import nvfp4_cosmos_official as o
from .storage import RunStore, canonical_json, digest, file_sha256, owner_identity

SCHEMA = "cosmos-official-nvfp4-t3-extension-v1"
BASE_SHA = "04cfb6e0f5bd581f8791d674c43aadaaeeb6da536256826f5c1b5a901359145e"
BASE_PLAN_SHA = "5cafb6898c4c24c044af745df4ec743599c0e71255680fea9272f9deeced1a2e"
require, read, write_new = o.require, o.read, o.write_new


def validate_allocation(a, catalog):
    require(a["schema"] == SCHEMA + "/allocation", "Wrong allocation schema")
    t3 = {sid for sid, x in catalog["items"].items() if x["task"] == "T3"}
    reused = a["reused_ids"]
    groups = a["new_assignments"]
    require(set(groups) == {"fourier0", "fourier1", "radon0"}, "Three explicit shards required")
    require(len(reused) == len(set(reused)) == 144, "Exactly144 existing T3 IDs required")
    require(all(len(x) == len(set(x)) == 544 for x in groups.values()), "Each new shard must have544 unique IDs")
    new = [sid for ids in groups.values() for sid in ids]
    require(len(set(new)) == 1632 and not set(new) & set(reused), "Duplicate or reused ID dispatched")
    require(len(t3) == 1776 and set(new + reused) == t3, "Incomplete T3 coverage")
    require(a["new_bf16_calls"] == 0 and a["automatic_retry"] is False, "No new BF16 or automatic retry")
    return t3


def validate_contract(plan, base, allocation, catalog):
    validate_allocation(allocation, catalog)
    require(plan["schema"] == SCHEMA and plan["host"] in ("fourier", "radon"), "Invalid host/schema")
    names = ["fourier0", "fourier1"] if plan["host"] == "fourier" else ["radon0"]
    require(plan["assignments"] == [allocation["new_assignments"][k] for k in names], "Host assignment changed")
    require(plan["shard_names"] == names, "Shard identity differs")
    require(plan["generation"] == base["generation"] == o.inference.GENERATION, "Generation controls changed")
    old_settings, new_settings = dict(base["engine_settings"]), dict(plan["engine_settings"])
    for key in ("model", "tokenizer"):
        old_settings.pop(key); new_settings.pop(key)
    require(old_settings == new_settings and plan["engine_settings"] == o.engine_settings(plan["runtime"]),
            "Qualified engine settings changed")
    require(plan["max_new_calls"] == len(names) * 544 and plan["max_new_calls_per_id"] == 1
            and plan["max_total_attempts_per_id"] == 1 and plan["automatic_retry"] is False
            and plan["new_bf16_calls"] == 0 and plan["deep_calls_per_fresh_worker"] == 2
            and plan["prior_failed_nvfp4_attempts"] == {}, "Finite one-attempt contract changed")
    ids = [sid for group in plan["assignments"] for sid in group]
    items = plan["panel"]["items"]
    require([x["sample_id"] for x in items] == ids, "Panel order differs")
    for item in items:
        original = dict(item); cell = original.pop("selected_cell_id")
        require(original == catalog["items"][item["sample_id"]], "Catalog input changed")
        p = item["row"]["source_parameters"]
        require(cell == "T3/angle=" + str(p["angle_deg"]) + "/sign=" + str(p["tilt_sign"]), "Cell identity differs")
    resources = plan["runtime"]["resources"]
    require(resources == base["runtime"]["resources"], "Qualified resource margins changed")
    require(set(plan["runtime"]["physical_locks"]) == {str(i) for i in range(len(names))}, "Physical leases missing")


def validate(plan, physical=True):
    refs = plan["references"]
    require(refs["base_plan"]["sha256"] == BASE_PLAN_SHA and refs["catalog"]["sha256"] == o.native.CATALOG_SHA,
            "Qualified reference identity changed")
    base, allocation, catalog = [read(o.verified(refs[k])) for k in ("base_plan", "allocation", "catalog")]
    validate_contract(plan, base, allocation, catalog)
    require(file_sha256(o.__file__) == BASE_SHA and file_sha256(__file__) == plan["extension_source_sha256"], "Runner source changed")
    if physical:
        for lock in plan["runtime"]["physical_locks"].values():
            state = o.file_state(lock["path"])
            require(all(state[k] == lock[k] for k in ("st_dev", "st_ino")), "Physical lock inode changed")
        current = o.runtime_identity(plan["runtime"])
        require(current == plan["runtime_identity"], "Sealed runtime changed")
        for key in ("packages", "sdk_sources", "implementation", "environment", "cosmos_package", "native_t1_sha256"):
            require(current[key] == base["runtime_identity"][key], "Pilot runtime differs: " + key)


def seal(args):
    base, allocation, catalog = read(args.base_plan), read(args.allocation), read(args.catalog)
    runtime = read(args.runtime)
    names = ["fourier0", "fourier1"] if args.host == "fourier" else ["radon0"]
    groups = [allocation["new_assignments"][k] for k in names]
    items = []
    for sid in sum(groups, []):
        item = dict(catalog["items"][sid]); p = item["row"]["source_parameters"]
        item["selected_cell_id"] = "T3/angle=" + str(p["angle_deg"]) + "/sign=" + str(p["tilt_sign"])
        items.append(item)
    plan = {"schema": SCHEMA, "host": args.host, "shard_names": names,
            "references": {k: o.descriptor(v) for k, v in
                           (("base_plan", args.base_plan), ("allocation", args.allocation), ("catalog", args.catalog))},
            "runtime": runtime, "runtime_identity": o.runtime_identity(runtime),
            "extension_source_sha256": file_sha256(__file__), "panel": {"items": items},
            "assignments": groups, "generation": base["generation"], "engine_settings": o.engine_settings(runtime),
            "max_new_calls": len(names) * 544, "max_new_calls_per_id": 1, "max_total_attempts_per_id": 1,
            "automatic_retry": False, "new_bf16_calls": 0, "deep_calls_per_fresh_worker": 2,
            "prior_failed_nvfp4_attempts": {}}
    validate(plan)
    write_new(args.output, plan)
    return {"status": "SEALED", "sha256": file_sha256(args.output), "new_calls": plan["max_new_calls"]}


def run(args):
    plan = read(args.plan); validate(plan)
    require(0 <= args.worker < len(plan["assignments"]) and re.fullmatch(r"[A-Za-z0-9_.-]+", args.execution), "Invalid worker")
    require(args.limit == 544, "Run exactly the finite assigned shard")
    admission = read(args.admission); plan_sha = file_sha256(args.plan)
    require(admission["plan_sha256"] == plan_sha and all(admission[k] is True for k in
            ("production_held", "owned_engines_exited", "physical_locks_released")), "Missing actual launch admission")
    require(admission["workers"][str(args.worker)]["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES"), "Wrong physical GPU")
    root = Path(plan["runtime"]["output_root"]); run_dir = root / "executions" / args.execution
    run_dir.mkdir(parents=True, exist_ok=False)
    binding = {"schema": SCHEMA, "plan_sha256": plan_sha, "worker": args.worker,
               "shard": plan["shard_names"][args.worker], "execution": args.execution, "owner": owner_identity(),
               "admission_sha256": file_sha256(args.admission), "limit": args.limit}
    write_new(run_dir / "binding.json", binding)
    identity = {"schema": SCHEMA, "plan_sha256": plan_sha, "allocation_sha256": plan["references"]["allocation"]["sha256"]}
    status, completed, started = "FAILED", [], time.monotonic()
    try:
        with RunStore(root, identity, o.MODEL_KEY, "worker" + str(args.worker),
                      plan["assignments"][args.worker], sum(plan["assignments"], [])) as store, o._signals():
            o.select_pending(store, plan["assignments"][args.worker], args.limit)
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
            import torch
            require(torch.version.cuda == "13.0", "Qualified CUDA13.0 required")
            o._configure_torch(torch)
            completed = o.execute_segment(plan, plan_sha, args.worker, args.limit, store, torch, args.execution)
            status = "SEGMENT_COMPLETE"
            store.event(status, execution=args.execution, IDs=completed, elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        write_new(run_dir / "failed.json", {**binding, "status": "FAILED", "error_type": type(exc).__name__,
                  "error": str(exc), "automatic_retry": False, "elapsed_seconds": time.monotonic() - started})
        raise
    finally:
        write_new(run_dir / "worker_return.json", {**binding, "status": status, "process_exit_observed": False})
    write_new(run_dir / "complete.json", {**binding, "status": status, "IDs": completed, "new_model_calls": len(completed)})
    return {"status": status, "new_model_calls": len(completed)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seal")
    for k in ("base-plan", "allocation", "catalog", "runtime", "output"): p.add_argument("--" + k, required=True)
    p.add_argument("--host", choices=("fourier", "radon"), required=True)
    p = sub.add_parser("check"); p.add_argument("--plan", required=True)
    p = sub.add_parser("run")
    for k in ("plan", "admission", "execution"): p.add_argument("--" + k, required=True)
    p.add_argument("--worker", type=int, required=True); p.add_argument("--limit", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "check": validate(read(args.plan)); result = {"status": "CPU_METADATA_PASS", "model_calls": 0}
    else: result = {"seal": seal, "run": run}[args.command](args)
    print(canonical_json(result)); return result


if __name__ == "__main__":
    main()
