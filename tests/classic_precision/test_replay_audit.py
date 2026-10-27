"""Selection must be reproducible, complete-cell based and honest about gaps."""
from copy import deepcopy
import json

import pytest

from classic_precision.replay_audit import SCHEMA, canonical, sha, select, main


def fixture_inventory(tmp_path):
    cells, all_rows = [], []
    for k, correct in enumerate((0, 1, 2, 3, 4, 5, 6, 7, 8)):
        rows = [{"sample_id": f"cell{k}-id{i}", "task": "T2", "truth": "LEFT",
                 "source_parameters": {"delta_Y": k / 10}} for i in range(8)]
        all_rows.extend(rows)
        raws = [{**row, "raw_answer": "LEFT" if i < correct else "RIGHT", "output_token_ids": [i+1],
                 "finish_reason": "stop", "technical_pass": True} for i, row in enumerate(rows)]
        cell = {"config": {"delta_Y": k / 10}}
        for field, suffix, payload in (("source_rows", "json", canonical(rows).encode()),
                ("reference_raws", "jsonl", ("\n".join(canonical(r) for r in raws)+"\n").encode())):
            name = f"{k}-{field}.{suffix}"
            (tmp_path / name).write_bytes(payload)
            cell[field] = {"path": name, "sha256": sha(payload)}
        cells.append(cell)
    payload = canonical({"rows": all_rows}).encode()
    (tmp_path / "manifest.json").write_bytes(payload)
    group = {"model": "qwen", "task": "T2", "modality": "video", "protocol_sha256": "a"*64,
             "source_manifests": [{"path": "manifest.json", "sha256": sha(payload)}],
             "profile_sha256": "b"*64, "status": "COMPLETE", "cells": cells}
    value = {"schema": SCHEMA, "final_references_sealed": True, "groups": [group]}
    path = tmp_path / "inventory.json"
    path.write_text(canonical(value))
    return path, value


def test_selection_is_order_independent_and_uses_all_ids_in_three_strata(tmp_path):
    path, value = fixture_inventory(tmp_path)
    a = select(path)
    value["groups"][0]["cells"].reverse()
    path.write_text(canonical(value))
    b = select(path)
    assert a["selected_cells"] == b["selected_cells"]
    assert a["scientific_calls"] == 24 and a["gpu_calls_started"] == 0
    for c, bounds in zip(a["selected_cells"], ((0, 2), (3, 5), (6, 8)), strict=True):
        assert bounds[0] <= c["correct"] <= bounds[1]
        assert len(c["references"]) == c["expected"] == 8


def test_refuses_in_progress_reference_and_changed_raw_bytes(tmp_path):
    path, value = fixture_inventory(tmp_path)
    value["final_references_sealed"] = False
    path.write_text(canonical(value))
    with pytest.raises(ValueError, match="sealed final"):
        select(path)
    value["final_references_sealed"] = True
    path.write_text(canonical(value))
    raw = tmp_path / value["groups"][0]["cells"][0]["reference_raws"]["path"]
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA mismatch"):
        select(path)


@pytest.mark.parametrize("mutation", ("duplicate_id", "missing_id", "missing_technical", "missing_tokens"))
def test_refuses_incomplete_or_unverified_cell_even_with_valid_bundle_sha(tmp_path, mutation):
    path, value = fixture_inventory(tmp_path)
    descriptor = value["groups"][0]["cells"][0]["reference_raws"]
    raw_path = tmp_path / descriptor["path"]
    records = [json.loads(line) for line in raw_path.read_text().splitlines()]
    if mutation == "duplicate_id": records.append(deepcopy(records[0]))
    elif mutation == "missing_id": records.pop()
    elif mutation == "missing_technical": records[0]["technical_pass"] = False
    else: records[0]["output_token_ids"] = []
    payload = ("\n".join(canonical(r) for r in records)+"\n").encode()
    raw_path.write_bytes(payload)
    descriptor["sha256"] = sha(payload)
    path.write_text(canonical(value))
    with pytest.raises(ValueError): select(path)


def test_explicit_unavailable_task_not_silently_passed_and_plan_never_overwritten(tmp_path):
    path, value = fixture_inventory(tmp_path)
    value["groups"].append({**{k: value["groups"][0][k] for k in
        ("model", "modality", "protocol_sha256", "profile_sha256")}, "task": "T5",
        "status": "NO_REFERENCE", "reason": "Library only; no formal inference panel"})
    path.write_text(canonical(value))
    result = select(path)
    assert result["status"] == "SELECTION_SEALED_WITH_GAPS"
    assert result["coverage_gaps"][0]["task"] == "T5"
    output = tmp_path / "plan.json"
    main(["--inventory", str(path), "--output", str(output)])
    before = output.read_bytes()
    with pytest.raises(FileExistsError): main(["--inventory", str(path), "--output", str(output)])
    assert output.read_bytes() == before


def test_cannot_hide_a_cell_from_the_sampling_population(tmp_path):
    path, value = fixture_inventory(tmp_path)
    value["groups"][0]["cells"].pop()
    path.write_text(canonical(value))
    with pytest.raises(ValueError, match="omits/adds"):
        select(path)
