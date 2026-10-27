"""Explicit memory placement overlay for the frozen Gemma speed worker.

Only two parent-process functions are rebound. The original input preparation,
model observer, generation, journals and shutdown ordering remain in gemma_speed.
This module never edits the base worker, installed SDK, or an existing profile.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import copy, deepcopy
import json
import os
from pathlib import Path
import socket
import tempfile

from . import gemma_speed as g
from . import inference

SCHEMA = "gemma-memory-overlay-v1"
BASE_SHA = "8f69560e111f78d515699c9dcb80921908e8610eb7eab5ce4fd25c835275ddeb"
INFERENCE_SHA = "2f5ca1c92f4c9931bf25271a1ad9db812403f5a00467faf872a50628c57d66d3"
ENGINE_CHANGES = {
    "cpu_offload_gb": {"from": 8, "to": 28},
    "gpu_memory_utilization": {"from": .90, "to": .80},
}
RESOURCE_BUDGETS = {"min_free_gpu_gib": 78, "min_available_ram_gib": 64}
REFERENCE_SHA = {
    20: "738d823872b884aeefef1f514b6bec09535fa74ebb1115e60333e3bc1ca967f9",
    48: "4513e79c763807e4586d46224be59774970f11cd9723df573d6022b2ccf2903c",
}
_BASE_ENGINE_SETTINGS = g.engine_settings
_BASE_PHYSICAL_LEASE = g.physical_lease


def validate_sources():
    g.require(g.file_sha256(g.__file__) == BASE_SHA
        and g.file_sha256(inference.__file__) == INFERENCE_SHA,
        "Frozen Gemma/inference source differs; no memory overlay installed")


def engine_settings(checkpoint):
    settings = _BASE_ENGINE_SETTINGS(checkpoint)
    for key, change in ENGINE_CHANGES.items():
        g.require(settings[key] == change["from"], "Original placement setting changed: " + key)
        settings[key] = change["to"]
    return settings


@contextmanager
def physical_lease(runtime, torch):
    """The same existing native lock, with an explicitly different budget."""
    import fcntl
    g.require(torch.cuda.device_count() == 1, "Exactly one visible GPU per worker")
    root = Path(runtime["lease_root"]).resolve()
    g.require(root == Path(runtime["asset_roots"]["T1"]).resolve() / "evaluation/outputs",
        "Native physical lock root changed")
    device = str(getattr(torch.cuda.get_device_properties(0), "uuid", None)
        or os.environ["CUDA_VISIBLE_DEVICES"])
    path = root / (".gpu-" + g.digest({"host": socket.gethostname(), "device": device}) + ".lock")
    g.require(path.is_file() and not path.is_symlink(),
        "Existing native GPU lock required; never create another lock")
    expected = path.stat()
    with path.open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st, current = os.fstat(stream.fileno()), path.stat()
        g.require((st.st_dev, st.st_ino) == (expected.st_dev, expected.st_ino)
            == (current.st_dev, current.st_ino), "Native physical lock inode changed")
        state = inference.resource_snapshot(torch, dict(RESOURCE_BUDGETS))
        g.require(state["admitted"], "Memory profile resource budget is not currently available")
        # These thresholds are controller-selected admission budgets. Neither
        # the thresholds nor this CPU overlay establish the actual GPU peak.
        yield {"path": str(path), "st_dev": st.st_dev, "st_ino": st.st_ino,
            "device": device, "resources": state, "resource_budget_profile": SCHEMA}
        current = path.stat()
        g.require((st.st_dev, st.st_ino) == (current.st_dev, current.st_ino),
            "Native physical lock inode changed while held")


def overlay_record(baseline):
    calls = baseline["maximum_model_calls"]
    g.require(calls in REFERENCE_SHA, "Only the existing 20/48 CPU references are supported")
    return {"schema": SCHEMA, "module": "classic_precision.gemma_memory",
        "source_sha256": g.file_sha256(__file__),
        "base_implementation": {"gemma_speed": BASE_SHA, "inference": INFERENCE_SHA},
        "baseline_profile_sha256": g.digest(baseline),
        "engine_setting_changes": deepcopy(ENGINE_CHANGES),
        "resource_budgets": dict(RESOURCE_BUDGETS),
        "rebound_parent_functions": ["engine_settings", "physical_lease"],
        "reference_file_sha256": REFERENCE_SHA[calls],
        "input_generation_and_shutdown": "delegated unchanged to frozen gemma_speed",
        "weight_placement": "native BF16 gate_up_proj-only CPU offload; actual placement measured by base observer"}


def validate_profile(profile):
    validate_sources()
    expected_implementation = {"gemma_speed": BASE_SHA, "inference": INFERENCE_SHA}
    g.require(profile.get("implementation") == expected_implementation,
        "Base implementation identity must remain explicit and unchanged")
    overlay = profile.get("execution_overlay", {})
    g.require(overlay.get("source_sha256") == g.file_sha256(__file__), "Memory overlay source SHA differs")
    g.require(profile["engine_settings"] == engine_settings(profile["runtime"]["checkpoint"]),
        "Memory profile changed a setting beyond the two authorized placement keys")
    baseline = deepcopy(profile)
    baseline.pop("execution_overlay", None)
    baseline["engine_settings"] = _BASE_ENGINE_SETTINGS(profile["runtime"]["checkpoint"])
    g.require(overlay == overlay_record(baseline), "Memory overlay declaration or baseline binding differs")
    return profile


@contextmanager
def installed_overlay():
    validate_sources()
    g.require(g.engine_settings is _BASE_ENGINE_SETTINGS and g.physical_lease is _BASE_PHYSICAL_LEASE,
        "Another execution overlay is already installed in this process")
    g.engine_settings, g.physical_lease = engine_settings, physical_lease
    try:
        yield
    finally:
        g.engine_settings, g.physical_lease = _BASE_ENGINE_SETTINGS, _BASE_PHYSICAL_LEASE


def seal_profile(args):
    validate_sources()
    g.require(not Path(args.output).exists(), "Refusing to overwrite an existing sealed profile")
    # g.seal_profile performs all original runtime/source/input validation.
    # Its baseline output is transient and is never an existing scientific file.
    with tempfile.TemporaryDirectory(prefix="gemma-memory-profile-") as temporary:
        baseline_args = copy(args)
        baseline_args.output = str(Path(temporary) / "baseline.json")
        baseline_args.panel_key = "panel"  # 48 uses the explicitly normalized --panel file.
        g.seal_profile(baseline_args)
        baseline = g.read(baseline_args.output)
    profile = deepcopy(baseline)
    profile["engine_settings"] = engine_settings(profile["runtime"]["checkpoint"])
    profile["execution_overlay"] = overlay_record(baseline)
    validate_profile(profile)
    g.seal(args.output, profile)
    return {"status": "SEALED", "profile_sha256": g.digest(profile),
        "execution_overlay": profile["execution_overlay"], "GPU_calls": 0}


def run(args):
    profile = validate_profile(g.read(args.profile))
    g.require(g.file_sha256(args.reference) == profile["execution_overlay"]["reference_file_sha256"],
        "Use the existing frozen CPU reference; do not rebuild or replace it")
    # The original run keeps its physical_lease outside its engine ExitStack.
    # Its shutdown observer, owned-PID checks, failure and exit receipts therefore
    # finish before this replacement lease exits; no lifecycle is reimplemented.
    with installed_overlay():
        return g.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("seal-profile")
    for name in ("plan", "runtime", "name", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--panel", help="Use the normalized 48-video panel JSON for consistency runs")
    p.add_argument("--variant", choices=("baseline", "cpu", "kernel"), default="cpu")
    p.add_argument("--range-lookup", choices=("native", "binary"), default="native")
    p.add_argument("--prefill-block-m", type=int, choices=(0, 32), default=0)
    p = commands.add_parser("run")
    for name in ("profile", "reference", "admission", "output"):
        p.add_argument("--" + name, required=True)
    args = parser.parse_args()
    print(json.dumps(seal_profile(args) if args.command == "seal-profile" else run(args),
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
