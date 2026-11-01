"""Private, append-only identity migration checks. Never runs models or publishes."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .dataset_identity import IdentityRegistry, canonical_json, restore_record
from .dataset_migration_preflight import (
    CURRENT, digest, extract_points, file_identity, freeze_current_records, read_json,
    source_catalog, write_csv,
)


def read_csv(path):
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def csv_value(value):
    return "" if value is None else str(value)


def compare_csv_rows(expected, actual, label):
    if len(expected) != len(actual):
        raise ValueError(f"{label} count differs: {len(expected)} != {len(actual)}")
    for ordinal, (before, after) in enumerate(zip(expected, actual), 1):
        normalized = {key: csv_value(value) for key, value in after.items()}
        if before != normalized:
            fields = sorted(key for key in set(before) | set(normalized) if before.get(key) != normalized.get(key))
            raise ValueError(f"{label} row {ordinal} differs in {fields}")


def scientific_fields(record):
    return {key: value for key, value in record.items()
            if key not in ("sample_id", "legacy_sample_id", "dataset_revision", "source_record_sha256")}


def audit_consumer_candidates(preparation, code_root, output):
    """Disposition every preflight candidate; retain frozen scientific consumers.

    This inventory is evidence of file ownership and data flow, not a claim that
    grep proves runtime behavior. The alias/media/record tests supply that part.
    """
    preparation, code_root, output = Path(preparation), Path(code_root), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a consumer review snapshot")
    manifest = read_json(preparation / "preparation.json")
    original_code = Path(manifest["code_root"])
    rows = []
    for candidate in manifest["consumer_candidates"]:
        original = Path(candidate["path"])
        if original.is_relative_to(original_code):
            relative = original.relative_to(original_code)
            review_path = code_root / relative
            if not review_path.is_file():
                review_path = original
            name = relative.as_posix()
        else:
            review_path = original
            name = original.relative_to(Path(manifest["root"])).as_posix()
        text = review_path.read_text(errors="replace")
        functions, imports, paths, dynamic = [], [], [], []
        if review_path.suffix == ".py":
            tree = ast.parse(text)
            functions = [node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
            imports = sorted({node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module})
            paths = sorted({node.value for node in ast.walk(tree) if isinstance(node, ast.Constant)
                and isinstance(node.value, str) and len(node.value) < 220
                and any(term in node.value for term in (".json", ".csv", ".jsonl", "manifest", "profile", "dataset-index"))})
        for ordinal, line in enumerate(text.splitlines(), 1):
            if "sample_id" in line and any(term in line for term in ("sorted(", "digest(", "seed", "startswith(", "split(", "re.")):
                dynamic.append({"line": ordinal, "code": line.strip()[:240]})
        if name in ("classic_precision/dataset.py", "classic_precision/dataset_names.py", "classic_precision/cosmos_x.py"):
            disposition, rationale, bridge = "UPDATED_PORTABLE_READER_AND_EXPORT", "Versioned registry resolves canonical/legacy ID/path; original scientific rows are preserved.", "dataset_identity.IdentityRegistry; dataset.resolve_entry/read_sample; dataset_names optional identity registry"
        elif name in ("README.md", "EXPERIMENT_PLAN.md", "classic_precision/source_release.py", "site/library.html"):
            disposition, rationale, bridge = "PUBLIC_RELEASE_OWNER", "Current source-only release uses dataset_release and download_dataset; historical release recipe retained.", "dataset_release source catalog + aliases.csv; download_dataset"
        elif name.startswith("tests/"):
            disposition, rationale, bridge = "PRESERVED_EXISTING_REGRESSION", "Existing tests retain fixtures and frozen scientific contracts; new registry/reader tests cover migration.", "tests/test_dataset_identity.py and existing dataset/name tests"
        elif name.startswith("runtime_compat/"):
            disposition, rationale, bridge = "PRESERVED_QUALIFIED_RUNTIME", "Runtime patch identities participate in frozen deployments; no inference/runtime replacement in this migration.", "legacy frozen profile/queue; opt-in identity views outside runtime"
        elif name.startswith("results/") or name == "report.md":
            disposition, rationale, bridge = "PRESERVED_SCIENTIFIC_HISTORY", "Original result and negative outcome content stays byte-identical; private canonical view is reversible.", "dataset_migration_validation full-record/point comparison"
        elif name.startswith(("site/", "web_label/")):
            disposition, rationale, bridge = "PRESERVED_RESULT_VIEW_OR_EXTERNAL_PUBLIC_ENTRY", "Website result payloads retain their frozen IDs; no new private model outputs are published. Public dataset links are owned by release task.", "legacy result payload unchanged; source catalog/aliases for new download entries"
        elif "gemma" in Path(name).name or name in ("classic_precision/inference.py", "classic_precision/storage.py", "classic_precision/gpu_telemetry.py"):
            disposition, rationale, bridge = "PRESERVED_ACTIVE_OR_SHARED_RUNTIME", "Current Gemma and frozen successors must retain IDs, paths, seeds, digests and code. No live entry is switched.", "IdentityRegistry outside raw/claim/queue/profile; final append-only baseline pending owner closure"
        elif any(term in Path(name).stem for term in ("results", "figures", "analysis", "audit", "summary", "history", "restore", "repair")):
            disposition, rationale, bridge = "PRESERVED_BOUND_ANALYSIS_OR_EVIDENCE", "Consumer joins original row/raw/profile IDs and scientific digests; its historical input remains available.", "reversible IdentityRegistry.project_record; original digest and per-cell answer distribution audit"
        elif Path(name).name in ("dataset_migration_preflight.py", "dataset_names.py"):
            disposition, rationale, bridge = "PRESERVED_BASELINE_PRODUCER", "Existing preparation is immutable; no prepare rerun or original baseline replacement.", "verify + dataset_migration_validation append-only snapshots"
        elif any(term in Path(name).stem for term in ("stimuli", "manifest", "refinement", "calibration", "pipeline", "source", "natural_t1_rgb", "natural_spoon_revision", "ready", "cpu")):
            disposition, rationale, bridge = "PRESERVED_SCIENTIFIC_GENERATOR_OR_SELECTION", "Frozen generation and selection keys remain legacy to preserve seeds/order/dedup/source digests. Naming is an exported view.", "IdentityRegistry.project_index/select_rows/source_sample_id; dataset_release"
        else:
            disposition, rationale, bridge = "PRESERVED_FROZEN_EXPERIMENT_OR_SMOKE", "Explicit existing experiment/qualification inputs retain legacy ID/path and profile identity; no new experiment is launched.", "old indexes/paths retained; opt-in registry resolves external canonical requests"
        now = file_identity(review_path)
        rows.append({"candidate_path": str(original), "reviewed_path": str(review_path), "relative_path": name,
            "candidate_sha256": candidate["sha256"], "reviewed_sha256": now["sha256"],
            "candidate_content_changed": candidate["sha256"] != now["sha256"],
            "disposition": disposition, "reason": rationale, "compatibility_or_preservation": bridge,
            "function_names_json": canonical_json(functions), "imports_json": canonical_json(imports),
            "input_output_reference_literals_json": canonical_json(paths), "dynamic_ID_usage_json": canonical_json(dynamic),
            "original_candidate_match_lines": canonical_json([m["line"] for m in candidate["matches"]]),
            "scientific_history_rewritten": False, "model_calls": 0})
    output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(output, rows)
    return {"status": "ALL_CANDIDATES_DISPOSITIONED", "files": len(rows),
        "dispositions": dict(Counter(row["disposition"] for row in rows)),
        "dynamic_ID_usage_files": sum(row["dynamic_ID_usage_json"] != "[]" for row in rows),
        "candidate_content_changed_files": [row["relative_path"] for row in rows if row["candidate_content_changed"]],
        "artifact": file_identity(output), "model_calls": 0}


def audit_results(root, preparation, output):
    """Compare every frozen full record and point; canonical exports are views."""
    root, preparation, output = Path(root), Path(preparation), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite a migration snapshot: " + str(output))
    registry = IdentityRegistry.from_csv(preparation / "assets-proposed.csv")
    assets, lookup, _ = source_catalog(root)
    compare_csv_rows(read_csv(preparation / "assets-proposed.csv"), assets, "current source bindings")
    current, cohorts, cells = freeze_current_records(root, lookup)
    baseline = read_csv(preparation / "baseline-records.csv")
    serialization_notes = []
    # The original private CSV did not expose the derived T4/T5 cell key.
    # Its full source-record digest is still checked below; retain that original
    # blank export field instead of changing the baseline or scientific data.
    for before, after in zip(baseline, current):
        if after["task"] in ("T4", "T5") and before["cell"] == "" and after["cell"]:
            serialization_notes.append({"cohort": after["cohort"], "sample_id": after["legacy_sample_id"],
                "field": "cell", "frozen_export": "", "newly_derived_key": after["cell"]})
            after["cell"] = ""
    compare_csv_rows(baseline, current, "completed current records")
    compare_csv_rows(read_csv(preparation / "baseline-current-cells.csv"), cells, "current cells")
    baseline_points = read_csv(preparation / "baseline-points.csv")
    # Keep preflight ordering exactly, without including newly arriving points.
    point_paths = sorted({Path(row["source_file"]) for row in baseline_points})
    points = extract_points(point_paths)
    compare_csv_rows(baseline_points, points, "historical/current points")

    baseline_by_key = {(row["cohort"], row["legacy_sample_id"]): row for row in baseline}
    field_hashes = defaultdict(lambda: defaultdict(lambda: [hashlib.sha256(), hashlib.sha256()]))
    answer_before, answer_after = defaultdict(Counter), defaultdict(Counter)
    records_out, unchanged, changed, non_T123 = [], 0, 0, 0
    for cohort, (directory, filename, _, _) in CURRENT.items():
        original = read_json(root / directory / filename)
        for record in original:
            row = baseline_by_key[(cohort, record["sample_id"])]
            if digest(record) != row["source_record_sha256"]:
                raise ValueError("Full original record changed: " + record["sample_id"])
            domain = "natural" if "natural" in cohort else "classic"
            if record["task"] in ("T1", "T2", "T3"):
                binding = registry.resolve(record["sample_id"], domain=domain)
                projected = registry.project_record(record, domain=domain)
                if restore_record(projected) != record:
                    raise ValueError("Full scientific record round trip changed")
                if ["video", binding.video_sha256] not in record["source_media"]:
                    raise ValueError("Full source-media SHA does not match")
                after = scientific_fields(projected)
                canonical_id = binding.canonical_id
                changed += int(canonical_id != record["sample_id"])
                unchanged += int(canonical_id == record["sample_id"])
            else:
                # T4/T5 controls remain local and retain missing GT/unscored labels.
                after, canonical_id = scientific_fields(record), record["sample_id"]
                non_T123 += 1
            before = scientific_fields(record)
            if before != after:
                raise ValueError("A scientific field differs after identity projection")
            for name in before:
                for hasher, value in zip(field_hashes[cohort][name], (before[name], after[name])):
                    hasher.update(canonical_json([record["sample_id"], value]).encode() + b"\n")
            cell = row["cell"]
            if not cell and record["task"] in ("T4", "T5"):
                p = record["parameters"]
                subgroup = "all" if record["task"] == "T4" else ("target_absent" if p["scenario"].startswith("target_absent") else p["scenario"])
                cell = canonical_json([record["domain"], record["task"], str(p["coherence"]) if record["task"] == "T4" else p["mask"], subgroup])
            key = (cohort, cell)
            answer_before[key].update([canonical_json([record["answer"], record["raw_answer"], record["truth"], record["eligible"], record["correct"]])])
            answer_after[key].update([canonical_json([after["answer"], after["raw_answer"], after["truth"], after["eligible"], after["correct"]])])
            records_out.append({"cohort": cohort, "task": record["task"], "legacy_sample_id": record["sample_id"],
                "canonical_sample_id": canonical_id, "source_record_sha256": digest(record),
                "scientific_before_sha256": digest(before), "scientific_after_sha256": digest(after),
                "video_media_sha256_json": canonical_json(record["source_media"]),
                "answer": record["answer"], "truth": record["truth"], "correct": record["correct"],
                "eligible": record["eligible"], "finish_reason": record.get("finish_reason"),
                "stop_reason_json": canonical_json(record.get("stop_reason")),
                "full_output_tokens_sha256": digest(record["output_token_ids"]),
                "prompt_token_sha256": record["prompt_token_sha256"], "raw_digest": record.get("raw_digest", ""),
                "comparison": "ALL_SCIENTIFIC_FIELDS_EXACT"})
    if answer_before != answer_after:
        raise ValueError("Per-cell full-answer/GT/eligibility distribution differs")
    distributions = [{"cohort": cohort, "cell": cell, "n": sum(counts.values()),
        "before_answers_GT_eligibility_correctness_json": canonical_json(dict(sorted(counts.items()))),
        "after_answers_GT_eligibility_correctness_json": canonical_json(dict(sorted(answer_after[(cohort, cell)].items()))),
        "exact": True} for (cohort, cell), counts in sorted(answer_before.items())]
    hashes = {cohort: {name: {"before": h[0].hexdigest(), "after": h[1].hexdigest(),
        "exact": h[0].digest() == h[1].digest()} for name, h in sorted(fields.items())}
        for cohort, fields in sorted(field_hashes.items())}
    output.mkdir(parents=True)
    write_csv(output / "records-comparison.csv", records_out)
    write_csv(output / "cell-answer-distributions.csv", distributions)
    if serialization_notes:
        write_csv(output / "preserved-export-serialization.csv", serialization_notes)
    # Retain exact per-point fields, including failure/unscored labels and source pointers.
    write_csv(output / "points-comparison.csv", [{"source_file": row["source_file"],
        "source_record": row["source_record"], "before_record_sha256": row["record_sha256"],
        "after_record_sha256": row["record_sha256"], "all_fields_exact": True} for row in points])
    report = {"schema": "dataset-migration-result-equivalence-v1", "status": "PASS_ALL_SCIENTIFIC_FIELDS_EXACT",
        "captured_UTC": datetime.now(timezone.utc).isoformat(), "private_results_no_upload": True,
        "model_calls": 0, "mutated_original_records": 0, "source_bindings": len(assets),
        "records": len(records_out), "classic_T1_ID_views_changed": changed,
        "other_T123_ID_views_unchanged": unchanged, "preserved_outside_scope_T45": non_T123,
        "current_cells": len(cells), "cells_with_answer_distributions": len(distributions),
        "point_records": len(points), "point_sources": len(point_paths),
        "baseline_all_CSV_fields_exact": True, "full_original_record_digest_exact": True,
        "preserved_export_serialization": {"rows": len(serialization_notes),
            "reason": "Frozen T45 baseline cell was blank; retain blank, independently recompute current cells and full record digests"},
        "full_projected_record_roundtrip_exact": True, "per_field_scientific_digests": hashes,
        "cohorts": cohorts, "active_Gemma_additions": "NOT_IN_THIS_FROZEN_BASELINE; append after original owner closes runs",
        "code": [file_identity(Path(__file__)), file_identity(Path(__file__).with_name("dataset_identity.py"))],
        "artifacts": [file_identity(output / name) for name in
                      ("records-comparison.csv", "cell-answer-distributions.csv", "points-comparison.csv")]}
    (output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_results(args.root, args.preparation, args.output)
    print(canonical_json({key: report[key] for key in ("status", "records", "current_cells", "point_records", "model_calls")}))


if __name__ == "__main__":
    main()
