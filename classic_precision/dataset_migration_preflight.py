"""Freeze private evidence before a dataset naming migration. No media mutations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
from zoneinfo import ZoneInfo


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def file_identity(path):
    path = Path(path)
    before = path.stat()
    with path.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
    if identity(before) != identity(after):
        raise ValueError(f"File changed during snapshot: {path}")
    return {"path": str(path), "bytes": after.st_size, "sha256": sha,
            "device": after.st_dev, "inode": after.st_ino, "mtime_ns": after.st_mtime_ns}


def rg_files(root, globs):
    command = ["rg", "--files", str(root), "-g", "!node_modules/**", "-g", "!.git/**"]
    for pattern in globs:
        command += ["-g", pattern]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeError(result.stderr)
    return [Path(line) for line in result.stdout.splitlines() if line]


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value):
    return format(float(value), ".12g")


def asset_alias(value):
    match = re.fullmatch(r"(photo|gpt)_(bg|obj)_(\d+)", value)
    if not match:
        raise ValueError(f"Unknown asset identity: {value}")
    return ("na" if match[1] == "photo" else "gpt") + "_" + match[3].zfill(3)


def physical_tilt_sign(domain, angle, source_sign):
    """Positive means the upper end leans left; zero denotes horizontal."""
    if float(angle) == 90:
        return 0
    if domain not in ("classic", "natural") or source_sign not in (-1, 1):
        raise ValueError((domain, angle, source_sign))
    return source_sign if domain == "classic" else -source_sign


def source_catalog(root):
    classic_path = root / "plans/classic-refinement/dataset-index.json"
    t1_path = root / "plans/natural-t1-rgb-20261030/source-index.json"
    release_path = root / "plans/video-library-20261030/release-manifest.json"
    t3_path = root / "plans/natural-t3-spoon-20261030/canonical-source-index.json"
    classic = read_json(classic_path)["items"]
    release = read_json(release_path)["datasets"]
    natural = read_json(t1_path)["rows"] + next(x["rows"] for x in release if x["task"] == "T2") + read_json(t3_path)["rows"]
    assert read_json(t3_path)["rows"] == next(x["rows"] for x in release if x["task"] == "T3")
    assert len(classic) == 12528 and len(natural) == 10416
    rows, lookup = [], {}
    for domain, records in (("classic", list(classic.values())), ("natural", natural)):
        for record in records:
            sid, task, sha = record["sample_id"], record["task"], record["video_sha256"]
            assert task in ("T1", "T2", "T3") and re.fullmatch(r"[0-9a-f]{64}", sha)
            old_path = record.get("video_root_relative", record["video_relative"] if "video_relative" in record else "")
            assert old_path
            new_sid, new_path, full_identity = sid, old_path, ""
            params = record.get("nominal_parameters", record)
            primary = True
            if domain == "classic":
                if task == "T1":
                    full_identity = digest({"schema": "sample-identity-v1", "domain": domain, "task": task, "video_sha256": sha})
                    new_sid = "t1_" + full_identity[:20]
                    phase = int(re.search(r"_i(\d+)_", sid)[1])
                    condition = [task, number(params["sf_cycles_per_aperture"]), number(params["modulation_depth"]), phase, record["truth"]]
                elif task == "T2":
                    duration = params["offset_frame"] - params["onset_frame"]
                    primary = duration == 30
                    condition = [task, number(params["delta_Y"]), params["onset_frame"], duration, record["truth"]]
                else:
                    primary = params["angle_deg"] not in (1.75, 2.25)
                    condition = [task, number(params["angle_deg"]), physical_tilt_sign(domain, params["angle_deg"], params["tilt_sign"]), params["onset_frame"], record["truth"]]
            else:
                bg = asset_alias(record["background_id"])
                if task == "T1":
                    name = f'T1_{record["truth"].lower()}_bg-{bg}_sf{number(record["sf"])}_contrast{number(record["m"])}_trial{record["phase_index"]+1:02d}.mp4'
                    condition = [task, number(record["sf"]), number(record["m"]), record["phase_index"], record["truth"]]
                elif task == "T2":
                    obj = asset_alias(record["object_id"])
                    duration = record["offset_frame"] - record["onset_frame"]
                    assert (record["onset_frame"] - 18) % 5 == 0
                    trial = (record["onset_frame"] - 18) // 5 + 1
                    name = f'T2_{record["truth"].lower()}_bg-{bg}_obj-{obj}_delta{number(record["delta_Y"])}_{duration}frames_trial{trial:02d}.mp4'
                    condition = [task, number(record["delta_Y"]), record["onset_frame"], duration, record["truth"]]
                else:
                    obj = asset_alias(record["object_id"])
                    sign = physical_tilt_sign(domain, record["angle"], record["tilt_sign"])
                    tilt = "horizontal" if sign == 0 else ("tiltRight" if sign < 0 else "tiltLeft")
                    name = f'T3_vertical{record["truth"].title()}_{tilt}_{number(record["angle"])}deg_bg-{bg}_obj-{obj}_trial{record["onset_index"]+1:02d}.mp4'
                    condition = [task, number(record["angle"]), sign, record["onset"], record["truth"]]
                new_path = task + "/" + name
            row = {"domain": domain, "task": task, "legacy_sample_id": sid, "proposed_sample_id": new_sid,
                   "legacy_path": old_path, "proposed_public_path": new_path, "video_sha256": sha,
                   "decoded_stream_sha256": record.get("stream_sha256", record.get("RGB_stream_sha256", "")),
                   "truth": record["truth"], "primary_T123_scope": primary,
                   "source_record_sha256": digest(record), "new_identity_full_sha256": full_identity,
                   "condition_key": canonical(condition), "background_id": record.get("background_id", ""),
                   "source_tilt_sign": params.get("tilt_sign", ""),
                   "physical_upper_end_direction": ("horizontal" if condition[2] == 0 else ("left" if condition[2] > 0 else "right")) if task == "T3" else "",
                   "object_id": record.get("object_id", ""), "parent_sample_id": record.get("parent_sample_id", ""),
                   "policy": "PROPOSED_ONLY_MEDIA_UNCHANGED"}
            rows.append(row)
            assert sid not in lookup
            lookup[sid] = row
    assert len({r["proposed_sample_id"] for r in rows}) == len(rows)
    assert len({r["video_sha256"] for r in rows}) == len(rows)
    assert len({(r["domain"], r["proposed_public_path"]) for r in rows}) == len(rows)
    assert sum(r["primary_T123_scope"] for r in rows if r["domain"] == "classic") == 9456
    return rows, lookup, [classic_path, t1_path, release_path, t3_path]


def point_like(record):
    if not isinstance(record, dict) or any(k in record for k in ("sample_id", "before_id", "classic_id", "short_id", "case_id")):
        return False
    return any(("accura" in k.lower() or "correct" in k.lower()) and not isinstance(v, (dict, list)) for k, v in record.items())


def point_row(path, where, record):
    def pick(*keys):
        value = next((record[k] for k in keys if k in record), "")
        return canonical(value) if isinstance(value, (dict, list)) else value
    return {"source_file": str(path), "source_record": where, "task": pick("task"), "model": pick("model"),
            "protocol": pick("variant", "protocol", "phase"), "cell": pick("cell", "cell_json", "config", "point"),
            "group": pick("group", "condition", "cohort", "domain"), "n": pick("n", "completed", "paired_n", "pairs"),
            "scored_n": pick("accuracy_n", "scored_n"), "correct": pick("correct"), "accuracy": pick("accuracy"),
            "original_record_json": canonical(record), "record_sha256": digest(record)}


def extract_points(paths):
    output = []
    def walk(value, path, pointer):
        if point_like(value):
            output.append(point_row(path, pointer, value))
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, path, pointer + "/" + key.replace("~", "~0").replace("/", "~1"))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path, pointer + "/" + str(index))
    for path in paths:
        if path.suffix == ".csv":
            with path.open(newline="") as stream:
                for ordinal, row in enumerate(csv.DictReader(stream), 1):
                    if point_like(row):
                        output.append(point_row(path, f"row:{ordinal}", row))
        elif path.name == "cells.jsonl":
            for ordinal, line in enumerate(path.read_text().splitlines(), 1):
                walk(json.loads(line), path, f"line:{ordinal}")
        elif path.suffix == ".json" and "summary" in path.name:
            walk(read_json(path), path, "")
    return output


CURRENT = {
    "cosmos_classic_X": ("runs/reproducibility/cosmos-x-full-20261028/analysis", "canonical-records.json", "cells.csv", 9456),
    "nemo_classic_X": ("runs/reproducibility/nemo-x-full-20261029/analysis", "canonical-records.json", "cells.csv", 9456),
    "qwen_classic_X": ("runs/reproducibility/qwen-x-remaining-t13-20261030/classic-analysis-complete-retry1", "canonical-records.json", "cells.csv", 9456),
    "qwen_natural_T2_X": ("runs/reproducibility/natural-t2-spoon-20261030", "canonical-records.json", "cell-results.csv", 3840),
    "cosmos_T45_X_preserved_not_published": ("runs/reproducibility/cosmos-x-t45-20261030/analysis", "records.json", "cells.csv", 1200),
}


def freeze_current_records(root, sources):
    output, checks, point_output = [], {}, []
    for label, (directory, filename, cellfile, expected) in CURRENT.items():
        records = read_json(root / directory / filename)
        assert len(records) == expected and len({x["sample_id"] for x in records}) == expected
        cells, tasks = defaultdict(Counter), defaultdict(Counter)
        for record in records:
            sid, task = record["sample_id"], record["task"]
            truth, answer, eligible = record["truth"], record["answer"], record["eligible"]
            assert record["correct"] == (answer == truth if eligible else None)
            source = sources.get(sid)
            if task in ("T1", "T2", "T3"):
                assert source and source["truth"] == truth
                assert ["video", source["video_sha256"]] in record["source_media"]
                assert source["primary_T123_scope"]
            if label == "qwen_natural_T2_X":
                key = canonical([record["parameters"]["group"], number(record["parameters"]["delta_Y"])])
            elif task in ("T1", "T2", "T3"):
                key = canonical(record["cell"])
            else:
                p = record["parameters"]
                subgroup = "all" if task == "T4" else ("target_absent" if p["scenario"].startswith("target_absent") else p["scenario"])
                key = canonical([record["domain"], task, str(p["coherence"]) if task == "T4" else p["mask"], subgroup])
            cells[key].update({"n": 1, "correct": int(record["correct"] is True), "accuracy_n": int(eligible)})
            tasks[task].update({"n": 1, "correct": int(record["correct"] is True), "accuracy_n": int(eligible)})
            output.append({"cohort": label, "model": record["model"], "protocol": record["variant"],
                           "task": task, "legacy_sample_id": sid, "proposed_sample_id": source["proposed_sample_id"] if source else sid,
                           "truth": truth, "answer": answer, "correct": record["correct"], "eligible": eligible,
                           "cell": key, "source_media_json": canonical(record["source_media"]),
                           "raw_answer": record["raw_answer"], "output_token_ids_json": canonical(record["output_token_ids"]),
                           "prompt_token_sha256": record["prompt_token_sha256"], "raw_digest": record.get("raw_digest", ""),
                           "source_record_sha256": digest(record), "source_file": str(root / directory / filename)})
        checked_cells = 0
        with (root / directory / cellfile).open(newline="") as stream:
            for row in csv.DictReader(stream):
                if "T45" in label:
                    key = canonical([row["domain"], row["task"], row["level"], row["subgroup"]])
                else:
                    key = canonical([row["group"], number(row["delta_Y"])]) if label == "qwen_natural_T2_X" else canonical(json.loads(row["cell"]))
                counts = cells[key]
                for name in ("n", "correct", "accuracy_n"):
                    source_name = "scored_n" if "T45" in label and name == "accuracy_n" else name
                    assert int(row[source_name]) == counts[name], (label, key, name)
                if counts["accuracy_n"]:
                    assert abs(float(row["accuracy"]) - counts["correct"] / counts["accuracy_n"]) < 1e-14
                else:
                    assert row["accuracy"] == ""
                checked_cells += 1
                point_output.append({"cohort": label, "model": records[0]["model"], "variant": records[0]["variant"],
                                     "task": row["task"] if "task" in row else ("T2" if label == "qwen_natural_T2_X" else json.loads(row["cell"])[0]),
                                     "cell": key, "n": counts["n"], "scored_n": counts["accuracy_n"], "correct": counts["correct"],
                                     "accuracy": row["accuracy"], "recomputed_matches": True, "source_file": str(root / directory / cellfile)})
        assert checked_cells == len(cells)
        checks[label] = {"records": len(records), "per_task": dict(tasks), "cells_recomputed_from_records": checked_cells,
                         "correctness_recomputed": True, "source_SHA_GT_matched": "NOT_REAUDITED_T45_OUTSIDE_RENAMING_SCOPE" if "T45" in label else True,
                         "status": "LOCAL_IMMUTABLE_COMPLETED_SELECTION"}
    return output, checks, point_output


def consumers(code, root):
    pattern = re.compile(r"sample_id|t1rgb16_|video_relative|video_path|dataset-index|canonical-records|natural-t[123]|huggingface|hf_hub_download|snapshot_download|read_csv|read_json")
    paths = rg_files(code, ["*.py", "*.md", "*.json", "*.sh", "*.js", "*.ts", "*.yml", "*.yaml"])
    paths += rg_files(root / "site", ["*.py", "*.md", "*.js", "*.ts", "*.html", "!**/node_modules/**"])
    result = []
    for path in sorted(set(paths)):
        if "node_modules" in path.parts or path.is_symlink() or path.stat().st_size > 5_000_000:
            continue
        matches = []
        for line, text in enumerate(path.read_text(errors="replace").splitlines(), 1):
            found = sorted(set(pattern.findall(text)))
            if found:
                matches.append({"line": line, "keys": found})
        if matches:
            result.append({**file_identity(path), "matches": matches, "review": "STATIC_REFERENCE_CANDIDATE_NOT_RUNTIME_VERIFIED"})
    return result


def prepare(root, code, out):
    if out.exists():
        raise ValueError(f"Refusing to overwrite a baseline: {out}")
    assets, sources, source_paths = source_catalog(root)
    evidence = rg_files(root / "runs", ["*.csv", "*summary*.json", "*records*.json", "cells.jsonl", "raw.jsonl", "results.jsonl", "outputs.jsonl"])
    evidence = sorted(set(p for p in evidence if not p.is_symlink()))
    points = extract_points(evidence)
    records, checks, current_points = freeze_current_records(root, sources)
    print(canonical({"phase": "hashing_evidence", "files": len(evidence), "bytes": sum(p.stat().st_size for p in evidence)}), flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        identities = list(pool.map(file_identity, evidence + source_paths))
    references = consumers(code, root)
    now = datetime.now(ZoneInfo("Pacific/Honolulu")).isoformat()
    current_runs = root / "runs/reproducibility"
    mutable = {}
    for rel in ("gemma-x-three-card-recheck-20261031/preparation-current.json", "gemma-x-three-card-recheck-20261031/on-confirm1/preparation-current.json", "gemma-x-provisional-20261031/status-fourier-latest.json"):
        path = current_runs / rel
        mutable[str(path)] = {"file_at_capture": file_identity(path), "snapshot": read_json(path), "policy": "MUTABLE_MONITOR_STATE_NOT_A_RESULT_BASELINE"}
    git_status = subprocess.check_output(["/usr/bin/git", "-C", str(code), "status", "--porcelain=v1"], text=True)
    manifest = {"schema": "dataset-naming-migration-preflight-v1", "status": "PREPARED_NOT_APPLIED",
                "captured_HST": now, "root": str(root), "code_root": str(code), "private_results_no_upload": True,
                "media_renamed": 0, "model_calls": 0, "remote_poll_count": 0,
                "source_counts": dict(Counter(r["domain"] + "/" + r["task"] for r in assets)),
                "classic_primary": 9456, "natural_primary": 10416, "proposed_T1_ID_changes": 3840,
                "all_current_source_bindings": len(assets), "point_metric_records": len(points),
                "point_sources": len({r["source_file"] for r in points}),
                "completed_current_trial_records": len(records), "completed_current_audits": checks,
                "immutable_evidence_files": identities, "consumer_candidates": references,
                "mutable_run_snapshots": mutable, "git_status_before": git_status,
                "git_HEAD": subprocess.check_output(["/usr/bin/git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip(),
                "source_catalog_policy": "Only current T1-T3 sources; historical variants and T45 retained unchanged outside public migration.",
                "identity_policy": "Only classic T1 becomes t1_ + first20 hex of SHA256 canonical JSON(schema=sample-identity-v1,domain=classic,task=T1,video_sha256=fullSHA). Existing hashed IDs retained. Full digest stored. Names are not identity inputs.",
                "results_policy": "Each point source/row preserved separately, never pooled across protocol/version/replicate. Current completed canonical records verified independently against stored cells. Historical CSV/JSON/JSONL evidence hashed and metric records frozen; not every historical result recomputed.",
                "active_run_policy": "Gemma H/U/L and ON-confirm1 stay on old IDs/paths/code. Live remote outputs newer than local snapshots are NOT included. Before cutover, capture final segments, merge a new explicitly timestamped baseline, verify true closure; never overwrite this baseline.",
                "consumer_policy": "Static candidates require triage. Add compatibility readers; never regex-rewrite immutable raw/claim/queue/audit records. Old source digests continue referring to original source records.",
                "source_precision_policy": "No media pixel/encoding changes. Existing per-domain full-frame QA is inherited only after actual destination bytes/SHA match. Metadata checks are not a new decode audit.",
                "deletion_policy": "No deletion in preparation. Future cleanup requires per-file identity/consumer/retained-copy whitelist; preserve all raw, negative outcomes and active assets.",
                "publication_policy": "New HF release contains only T1-T3 sources, provenance, source QA, ID/path mapping, downloader and metadata-only demo CSV. Do not upload model result tables or this private baseline. Preserve historical commits."}
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="migration-preflight-", dir=out.parent) as temporary:
        stage = Path(temporary)
        write_csv(stage / "assets-proposed.csv", assets)
        write_csv(stage / "baseline-points.csv", points)
        write_csv(stage / "baseline-records.csv", records)
        write_csv(stage / "baseline-current-cells.csv", current_points)
        manifest["artifacts"] = [{**file_identity(stage / name), "path": name} for name in ("assets-proposed.csv", "baseline-points.csv", "baseline-records.csv", "baseline-current-cells.csv")]
        (stage / "preparation.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        out.mkdir()
        for path in stage.iterdir():
            path.rename(out / path.name)
    print(canonical({"status": manifest["status"], "out": str(out), "assets": len(assets), "points": len(points), "current_records": len(records), "consumer_candidates": len(references), "checks": checks}), flush=True)


def verify(out):
    manifest = read_json(out / "preparation.json")
    expected = manifest["immutable_evidence_files"] + [{**row, "path": str(out / row["path"])} for row in manifest["artifacts"]]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(file_identity, [row["path"] for row in expected]))
    failures = [row["path"] for row, now in zip(expected, actual) if (row["bytes"], row["sha256"]) != (now["bytes"], now["sha256"])]
    if failures:
        raise ValueError(canonical({"changed": failures}))
    print(canonical({"status": "PASS_PRE_MIGRATION_BASELINE_UNCHANGED", "files": len(actual), "mutable_live_snapshots_excluded": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, help="Benchmark root; required when creating a new baseline")
    parser.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.root is None:
            parser.error("prepare requires --root")
        prepare(args.root.resolve(), args.code_root.resolve(), args.out.resolve())
    else:
        verify(args.out.resolve())


if __name__ == "__main__":
    main()
