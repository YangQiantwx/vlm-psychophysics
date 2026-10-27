"""One finite QSA-only pass over the frozen 1,440-row duration extension.

Reuse the verified native worker and durable call reader in a private module.
The production plan has its own strict validator; no old five-pass plan is
modified, accepted, or resumed. This module cannot execute cold repeats.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

from .storage import atomic_json, digest, file_sha256

SCHEMA = "classic_precision_qwen_t2_extension_single/v1"
WORKER_SHA = "44b831ec9fc6e990a0e99572545aacd06c879d9473419bb2b2c1e9fec0b56f4c"
MANIFEST_SHA = "675552daf04449aca1b8c0c7b3934dbed3e4d2e1f2d04c3e0b35a0c8bdd3bce8"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


_path = Path(__file__).with_name("qwen_t2t3_five_pass.py")
require(file_sha256(_path) == WORKER_SHA, "Frozen worker source changed")
_spec = importlib.util.spec_from_file_location("classic_precision._duration_single_worker", _path)
worker = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(worker)
worker.SCHEMA = SCHEMA
worker.COUNTS = {"original": 1440}
_native_implementation = worker.implementation
_compat = None


def install_compat(path, expected):
    global _compat
    require(file_sha256(path) == expected, "Frozen duration adapter changed")
    if _compat is None:
        import os
        require(os.environ.get("CLASSIC_PRECISION_T2_DURATION_EXTENSION_COMPAT") == "1",
                "Explicit duration compatibility environment required")
        spec = importlib.util.spec_from_file_location("_duration_single_compat", path)
        _compat = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_compat)
        require(_compat.install(), "Duration adapter not installed")
    require(Path(_compat.__file__).resolve() == Path(path).resolve(), "Another duration adapter loaded")


def implementation():
    return {**_native_implementation(), str(Path(__file__).resolve()): file_sha256(__file__)}


def validate(plan):
    require(plan.get("schema") == SCHEMA and plan.get("passes") == 1
            and plan.get("max_model_calls") == plan.get("global_max_model_calls") == 1440
            and plan.get("automatic_retry") is False and plan.get("implementation") == implementation(),
            "Single-pass scope or implementation differs")
    require(plan["host"] == "cirplab-fourier" and plan["gpu"] in ("0", "1")
            and set(plan["panels"]) == {"original"}, "Unexpected host/GPU/panel")
    require(plan["profile"] == worker.qsa.build_profile("qwen_qsa_only", plan["profile"]["baseline_environment"])
            and plan["profile"]["torch_policy"] == {}, "QSA-only profile differs")
    require(plan["output_root"] == str(Path(plan["stop_path"]).resolve().parent), "Output/stop binding differs")
    require(plan["launch_environment"] == {"CLASSIC_PRECISION_T2_DURATION_EXTENSION_COMPAT": "1"},
            "Unexpected launch intervention")
    compat = plan["compat"]
    worker.verified(compat)
    install_compat(compat["path"], compat["sha256"])
    item = plan["panels"]["original"]
    for key in ("manifest", "ready", "calibration"):
        worker.verified(item[key])
    manifest = worker.read(item["manifest"]["path"])
    rows = manifest["rows"]
    from .inference import qualification_order
    ordered, qualification = qualification_order(rows)
    ids = [r["sample_id"] for r in rows]
    require(item["manifest"]["sha256"] == MANIFEST_SHA and len(ids) == len(set(ids)) == 1440
            and {r["task"] for r in rows} == {"T2"}
            and item["all_sample_ids"] == ids and item["sample_ids"] == [r["sample_id"] for r in ordered]
            and item["qualification_ids"] == [r["sample_id"] for r in qualification]
            and len(qualification) == 2, "Frozen duration IDs/order/qualification differ")
    require(item["old_stores"] == [], "Extension must not reuse original responses")
    worker.verified(plan["reference_plan"])
    original = worker.read(plan["reference_plan"]["path"])
    require(original["profile"] == plan["profile"] and original["checkpoint_assets"] == plan["checkpoint_assets"]
            and original["admission"] == plan["admission"], "Qualified original profile/checkpoint differs")
    worker.verified(plan["admission"]["report"])
    for value in plan["admission"]["plans"]:
        worker.verified(value)
    worker.validate_admission(worker.read(plan["admission"]["report"]["path"]),
        [worker.read(x["path"]) for x in plan["admission"]["plans"]], plan["profile"], plan["gpu"], plan["gpu_uuid"])
    require(plan["wave_sha256"] == worker.wave_digest(plan), "Single-pass wave changed")
    return plan


def load_plan(path):
    return validate(worker.read(path))


worker.implementation = implementation
worker.load_plan = load_plan


def prepare(args):
    root = Path(args.root).resolve()
    source = worker.read(args.reference_plan)
    ext = root / "runs/t2-duration-extension"
    compat = worker.descriptor(ext / "operator/t2_duration_compat.py")
    install_compat(compat["path"], compat["sha256"])
    manifest = worker.read(ext / "manifest.json")
    from .inference import qualification_order
    ordered, qualification = qualification_order(manifest["rows"])
    item = {k: worker.descriptor(ext / p) for k, p in
            (("manifest", "manifest.json"), ("ready", "ready.jsonl"), ("calibration", "calibration/qwen/summary.json"))}
    item.update(asset_root=str(ext), calibration_root=str(ext / "calibration"),
        all_sample_ids=[r["sample_id"] for r in manifest["rows"]],
        sample_ids=[r["sample_id"] for r in ordered],
        qualification_ids=[r["sample_id"] for r in qualification], old_stores=[])
    output = Path(args.output_root).resolve()
    plan = {"schema": SCHEMA, "model": "qwen", "passes": 1, "max_model_calls": 1440,
        "global_max_model_calls": 1440, "automatic_retry": False, "host": source["host"],
        "gpu": source["gpu"], "gpu_uuid": source["gpu_uuid"], "profile": source["profile"],
        "checkpoint_assets": source["checkpoint_assets"], "model_root": source["model_root"],
        "gpu_lock_root": source["gpu_lock_root"], "admission": source["admission"],
        "reference_plan": worker.descriptor(args.reference_plan), "compat": compat,
        "implementation": implementation(), "wave_id": "qwen-t2-duration-single-20261024",
        "output_root": str(output), "stop_path": str(output / "STOP.json"),
        "launch_environment": {"CLASSIC_PRECISION_T2_DURATION_EXTENSION_COMPAT": "1"},
        "panels": {"original": item},
        "scientific_claim": "One new measurement of each extension video; not a full repeatability test."}
    plan["wave_sha256"] = worker.wave_digest(plan)
    validate(plan)
    require(not Path(args.plan).exists(), "Cannot overwrite sealed plan")
    atomic_json(args.plan, plan)
    return {"status": "PREPARED", "plan_sha256": digest(plan), "calls": 1440}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    for key in ("root", "reference-plan", "output-root", "plan"):
        prep.add_argument("--" + key, required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    else:
        plan = load_plan(args.plan)
        output = Path(plan["output_root"]) / ("gpu" + plan["gpu"])
        require(not worker.directory(output, 0, "original").exists(), "No resume or duplicate launch")
        result = worker.run(args.plan, output, "original", 0)
    import json
    print(json.dumps(result, sort_keys=True))
    return 0 if result.get("status") in ("PREPARED", "COMPLETE") else 1


if __name__ == "__main__":
    sys.exit(main())
