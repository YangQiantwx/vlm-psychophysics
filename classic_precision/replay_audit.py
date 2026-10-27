"""Seal a finite, stratified cell audit after final reference results exist.

CPU only: this command selects complete cells; it never launches inference or
certifies model repeatability. Model-specific launch/evidence gates are separate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

SCHEMA = "classic_precision_cell_audit/v1"
DEFAULT_SEED = "cvpr2027-postrun-cells-20261023-v1"
GROUP_FIELDS = ("model", "task", "modality", "protocol_sha256", "profile_sha256")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def verified(root, descriptor):
    relative = Path(descriptor["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, "Use relative evidence paths")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), "Evidence path escapes inventory directory")
    payload = path.read_bytes()
    require(sha(payload) == descriptor["sha256"], "Evidence SHA mismatch: " + str(relative))
    return payload


def cell_reference(root, cell, group, source):
    """Cross-check the entire cell against its sealed source-row and raw bundles."""
    rows = json.loads(verified(root, cell["source_rows"]))
    raws = [json.loads(line) for line in verified(root, cell["reference_raws"]).splitlines() if line.strip()]
    require(isinstance(rows, list) and rows, "Empty source cell")
    ids = [row["sample_id"] for row in rows]
    require(len(ids) == len(set(ids)), "Duplicate source ID")
    by_id = {raw["sample_id"]: raw for raw in raws}
    require(len(by_id) == len(raws) and set(by_id) == set(ids), "Incomplete/duplicate/extra reference ID")
    correct = 0
    references = []
    for row in rows:
        sid = row["sample_id"]
        raw = by_id[sid]
        require(sid in source and row == source[sid], "Cell source differs from full manifest")
        require(row["task"] == group["task"] and row["truth"] in ("LEFT", "RIGHT"), "Source task/truth differs")
        require(raw["task"] == row["task"] and raw["truth"] == row["truth"]
                and raw["source_parameters"] == row["source_parameters"], "Reference row identity differs")
        require(raw.get("technical_pass") is True, "Reference lacks verified current technical PASS")
        require(isinstance(raw.get("output_token_ids"), list) and raw["output_token_ids"]
                and isinstance(raw.get("raw_answer"), str) and raw.get("finish_reason") is not None,
                "Reference lacks full terminal output")
        # Complete invalid/length/wrong outputs remain in the denominator.
        correct += raw["finish_reason"] == "stop" and raw["raw_answer"] == row["truth"]
        references.append({"sample_id": sid, "source_row_sha256": sha(canonical(row).encode()),
                           "reference_raw_sha256": sha(canonical(raw).encode())})
    return {"config": cell["config"], "source_rows": cell["source_rows"],
            "reference_raws": cell["reference_raws"], "expected": len(ids), "correct": correct,
            "references": references}


def select(inventory_path, seed=DEFAULT_SEED):
    path = Path(inventory_path).resolve()
    payload = path.read_bytes()
    inventory = json.loads(payload)
    require(inventory.get("schema") == SCHEMA and inventory.get("final_references_sealed") is True,
            "Wait for sealed final reference results")
    require(isinstance(seed, str) and bool(seed), "Explicit fixed seed required")
    groups = inventory["groups"]
    require(isinstance(groups, list) and groups, "Model/task coverage inventory required")
    selected, gaps, seen = [], [], set()
    for group in groups:
        identity = {field: group[field] for field in GROUP_FIELDS}
        key = canonical(identity)
        require(all(isinstance(v, str) and v for v in identity.values()), "Incomplete group identity")
        require(key not in seen, "Duplicate model/task/profile group")
        seen.add(key)
        if group.get("status") != "COMPLETE":
            require(group.get("reason"), "Missing reference needs an explicit reason")
            gaps.append({**identity, "status": group["status"], "reason": group["reason"]})
            continue
        source = {}
        for descriptor in group["source_manifests"]:
            manifest = json.loads(verified(path.parent, descriptor))
            for row in manifest["rows"]:
                if row["task"] == group["task"]:
                    require(row["sample_id"] not in source, "Overlapping source manifests")
                    source[row["sample_id"]] = row
        require(source, "Full task source manifest required")
        cells = [cell_reference(path.parent, cell, group, source) for cell in group["cells"]]
        configs = [canonical(c["config"]) for c in cells]
        require(len(configs) == len(set(configs)), "Duplicate configuration cell")
        ids = [r["sample_id"] for c in cells for r in c["references"]]
        require(len(ids) == len(set(ids)), "Overlapping cell IDs within one group")
        require(set(ids) == set(source), "Cell inventory omits/adds source manifest IDs")
        if len(cells) < 3:
            gaps.append({**identity, "status": "INSUFFICIENT_CELLS", "count": len(cells)})
            continue
        def random_key(label, cell):
            return sha(canonical([seed, identity, label, cell["config"]]).encode())
        # Exact fractions avoid ranking cells by rounded displayed accuracy.
        cells.sort(key=lambda c: (Fraction(c["correct"], c["expected"]), random_key("tie", c)))
        n = len(cells)
        boundaries = (0, n // 3, 2 * n // 3, n)
        for i, label in enumerate(("low", "middle", "high")):
            pool = cells[boundaries[i]:boundaries[i+1]]
            chosen = min(pool, key=lambda c: random_key(label, c))
            selected.append({**identity, "stratum": label, "candidate_count": len(pool),
                "stratum_accuracy_range": [float(Fraction(c["correct"], c["expected"])) for c in (pool[0], pool[-1])],
                "accuracy_ties_present": len({Fraction(c["correct"], c["expected"]) for c in cells}) < n,
                **chosen})
    return {"schema": SCHEMA, "status": "SELECTION_SEALED_WITH_GAPS" if gaps else "SELECTION_SEALED",
        "inventory_sha256": sha(payload), "selector_sha256": sha(Path(__file__).read_bytes()),
        "seed": seed, "sampling": "rank thirds; seeded tie ordering; seeded uniform hash choice within each third",
        "selected_cells": selected, "coverage_gaps": gaps,
        "scientific_calls": sum(c["expected"] for c in selected),
        "repeats_per_selected_id": 1, "gpu_calls_started": 0,
        "claim": "Selection only. Selected complete cells require an independent model rerun and ID-bound input/runtime/full-output comparison; no full-dataset repeatability claim."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    result = select(args.inventory, args.seed)
    # A different reference snapshot or new draw must never overwrite a plan.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({key: result[key] for key in ("status", "scientific_calls", "gpu_calls_started")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
