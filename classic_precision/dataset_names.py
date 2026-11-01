"""Stable readable filenames derived from the existing T1-T3 metadata.

This changes exported paths only. Sample IDs, source content and inference
profiles remain independent of the display filename.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import math
from pathlib import Path

from .dataset import load_index, validate_index
from .manifest import atomic_json, digest

POLICY = "classic-readable-v1"


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Expected a finite numeric condition")
    return format(value, ".12g")


def side(value):
    if value not in ("LEFT", "RIGHT"):
        raise ValueError("Expected LEFT or RIGHT")
    return value


def trial(index):
    if type(index) is not int or not 0 <= index < 24:
        raise ValueError("The existing source bank has trial indices 0..23")
    return f"trial{index + 1:02d}"


def filename(entry):
    task, p = entry["task"], entry["nominal_parameters"]
    if task == "T1":
        k = p["bank_start_index"]
        label = trial(k)
        if not math.isclose(p["phase_degrees"], 7.5 + 15*k, abs_tol=1e-10):
            raise ValueError("Phase/trial binding differs from the frozen source bank")
        direction = side(p["direction"])
        if entry["truth"] != direction:
            raise ValueError("Motion direction and truth disagree")
        return (f"T1_{direction.lower()}_sf{number(p['sf_cycles_per_aperture'])}"
                f"_contrast{number(p['modulation_depth'])}_{label}.mkv")
    if task not in ("T2", "T3"):
        raise ValueError("Naming policy only covers T1-T3")
    k = p["onset_index"]
    label = trial(k)
    if p["onset_frame"] != 18 + 5*k:
        raise ValueError("Onset/trial binding differs from the frozen source bank")
    duration = p["offset_frame"] - p["onset_frame"]
    if task == "T2":
        target = side(p["target_side"])
        if entry["truth"] != target or duration not in (10, 30):
            raise ValueError("Unexpected brightness target or duration")
        return (f"T2_{target.lower()}_bg{number(p['background_Y'])}"
                f"_delta{number(p['delta_Y'])}_{duration}frames_{label}.mkv")
    reference = side(p["reference_side"])
    angle, sign = p["angle_deg"], p["tilt_sign"]
    if not 0 < angle <= 90 or type(sign) is not int or sign not in (-1, 1):
        raise ValueError("Invalid orientation condition")
    if entry["truth"] != reference or duration != 30:
        raise ValueError("Unexpected orientation reference or duration")
    if angle == 90:
        if sign != 1:
            raise ValueError("The canonical horizontal source is deduplicated")
        tilt = "horizontal"
    else:
        # raster.vertices uses image y downward: positive angle leans the
        # lower end right, so its upper end leans left.
        tilt = "tiltLeft" if sign == 1 else "tiltRight"
    return f"T3_vertical{reference.title()}_{tilt}_{number(angle)}deg_{label}.mkv"


def relabel_index(index, layout="tasks", identity_registry=None):
    validate_index(index)
    if layout not in ("tasks", "flat"):
        raise ValueError("Layout must be tasks or flat")
    if identity_registry is not None and layout != "tasks":
        raise ValueError("The versioned identity registry fixes the canonical task-folder layout")
    result, mapping, paths = deepcopy(index), [], set()
    for sid, original in index["items"].items():
        path = (original["task"] if layout == "tasks" else "videos") + "/" + filename(original)
        if path in paths:
            raise ValueError("Two source IDs map to the same readable filename: " + path)
        paths.add(path)
        result["items"][sid]["video_relative"] = path
        mapping.append({"sample_id": sid, "task": original["task"],
                        "old_path": original["video_relative"], "new_path": path,
                        "video_sha256": original["video_sha256"]})
    result["filename_policy"] = POLICY
    result["layout_policy"] = "task-folders-v1" if layout == "tasks" else "flat-v1"
    result["index_id"] = digest({k: v for k, v in result.items() if k != "index_id"})
    validate_index(result)
    if identity_registry is not None:
        result = identity_registry.project_index(result)
        for row in mapping:
            row["legacy_sample_id"] = row["sample_id"]
            row["sample_id"] = identity_registry.canonical_id(row["sample_id"], domain="classic")
    return result, sorted(mapping, key=lambda r: r["sample_id"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--layout", choices=("tasks", "flat"), default="tasks")
    p.add_argument("--identity-registry", type=Path,
                   help="Opt-in versioned alias CSV; frozen generator IDs remain unchanged")
    args = p.parse_args()
    from .dataset_identity import IdentityRegistry
    registry = IdentityRegistry.from_csv(args.identity_registry) if args.identity_registry else None
    result, mapping = relabel_index(load_index(args.index), args.layout, registry)
    atomic_json(args.output, result)
    with args.mapping.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    print(f"{len(mapping)} unique source IDs and paths; no video files read or modified")


if __name__ == "__main__":
    main()
