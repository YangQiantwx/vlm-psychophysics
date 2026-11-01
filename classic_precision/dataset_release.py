"""Build an explicit, source-only T1–T3 release. This module never uploads."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil

from .dataset_identity import IdentityRegistry

REVISION = "dataset-naming-v1"
REPOS = {("classic", None): "DrQY/vlm-psychophysics-classic-t1-t3",
         **{("natural", t): f"DrQY/vlm-psychophysics-natural-{t.lower()}"
            for t in ("T1", "T2", "T3")}}
PARAMETERS = ("sf", "m", "phase_index", "delta_Y", "onset_frame", "offset_frame",
              "angle", "tilt_sign", "onset_index", "onset", "group", "background_source",
              "object_source", "alpha_area_px2", "baseline_object_Y", "white_blend_k")
PROVENANCE = ("category", "source_kind", "family", "semantic_id", "label", "asset_id",
              "source_group", "capture_group_status", "url", "source_url", "api_info_url",
              "api_files_url", "download_url", "license", "license_url", "author",
              "publisher_description", "source_dimensions", "source_sha256", "publisher_md5",
              "publisher_max_resolution", "date_taken", "backplate_id", "kind", "yaw_deg",
              "pitch_deg", "generated", "color_note", "source_provenance_note", "master_dimensions",
              "master_sha256", "sha256", "SHA", "transform", "recommended_input_transform",
              "input_dimensions", "input_sha256", "input_transform", "dimensions", "provider",
              "provenance", "alpha_mode", "raw_sha256", "mask_sha256", "segmentation", "concept",
              "generation_prompt", "generation_model", "generation_method", "source_revision", "prompt", "full_prompt")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def lfs_sha256(value):
    """Support Hub clients returning either a mapping or a typed LFS record."""
    if isinstance(value, dict):
        return value.get("sha256")
    return getattr(value, "sha256", None)


def json_file(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def json_lines(path, rows):
    path.write_text("".join(canonical(r) + "\n" for r in rows))


def safe_relative(value):
    p = PurePosixPath(value)
    if not value or p.is_absolute() or ".." in p.parts or "\\" in value:
        raise ValueError(f"Unsafe public relative path: {value}")
    return p.as_posix()


def public_value(value):
    """Reject workstation paths and private storage references, including nested values."""
    if isinstance(value, dict):
        return {k: public_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [public_value(v) for v in value]
    if isinstance(value, str) and any(s in value for s in
            ("/Users/", "/home/", "/mnt/", "gs://", "100.65.75.63", "100.106.240.56")):
        raise ValueError("Selected public field contains a private workspace reference")
    return value


def source_records(root):
    def load(relative):
        return json.loads((root / relative).read_text())
    classic = load("plans/classic-refinement/dataset-index.json")
    t1 = load("plans/natural-t1-rgb-20261030/source-index.json")["rows"]
    release = load("plans/video-library-20261030/release-manifest.json")["datasets"]
    natural = t1 + next(d["rows"] for d in release if d["task"] == "T2")
    natural += load("plans/natural-t3-spoon-20261030/canonical-source-index.json")["rows"]
    return classic, {(r["sample_id"]): r for r in list(classic["items"].values()) + natural}


def project_rows(mapping, originals, prompts=None):
    rows = []
    for m in mapping:
        r = originals[m["legacy_sample_id"]]
        if hashlib.sha256(canonical(r).encode()).hexdigest() != m["source_record_sha256"]:
            raise ValueError(f"Frozen source record changed: {m['legacy_sample_id']}")
        if r["video_sha256"] != m["video_sha256"] or r["truth"] != m["truth"]:
            raise ValueError("Source media/GT differs from preparation")
        row = {"dataset_revision": REVISION, "domain": m["domain"], "task": m["task"],
               "sample_id": m["proposed_sample_id"], "legacy_sample_id": m["legacy_sample_id"],
               "path": safe_relative(m["proposed_public_path"]), "legacy_path": safe_relative(m["legacy_path"]),
               "video_sha256": m["video_sha256"], "decoded_stream_sha256": m["decoded_stream_sha256"],
               "full_identity_sha256": m["new_identity_full_sha256"],
               "source_record_sha256": m["source_record_sha256"], "truth": m["truth"],
               "condition_key": json.loads(m["condition_key"]),
               "primary_T123_scope": m["primary_T123_scope"] == "True",
               "width": r.get("width", 768), "height": r.get("height", 512),
               "frames": r.get("frames", 180), "fps": r.get("fps", 60),
               "parameters": r.get("nominal_parameters", {k:r[k] for k in PARAMETERS if k in r})}
        for key in ("background_id", "object_id", "parent_sample_id", "physical_upper_end_direction"):
            if m[key]:
                row[key] = m[key]
        if "prompt" in r:
            row["prompt"] = r["prompt"]
        elif prompts and (m["domain"], m["task"]) in prompts:
            row["prompt"] = prompts[m["domain"], m["task"]]
            if "prompt_sha256" in r and hashlib.sha256(row["prompt"].encode()).hexdigest() != r["prompt_sha256"]:
                raise ValueError("Prompt file bytes differ from the frozen source prompt")
        rows.append(public_value(row))
    IdentityRegistry.from_rows(rows, dataset_revision=REVISION)
    return rows


def source_assets(root, task, rows):
    release = json.loads((root / "plans/video-library-20261030/release-manifest.json").read_text())
    supplements = {a["id"]:a for d in release["datasets"] if d["task"] == "T2" for a in d["assets"]}
    if task == "T1":
        catalog = json.loads((root / "plans/natural-t1-rgb-20261030/catalog.json").read_text())
    else:
        assets = next(d["assets"] for d in release["datasets"] if d["task"] == task)
        catalog = {"backgrounds":[a for a in assets if "_bg_" in a["id"]],
                   "objects":[a for a in assets if "_obj_" in a["id"]]}
    used = {r[k] for r in rows for k in ("background_id", "object_id") if k in r}
    result = []
    for kind in ("backgrounds", "objects"):
        for asset in catalog.get(kind, []):
            if asset["id"] not in used:
                continue
            supplement = supplements.get(asset["id"])
            supplement_digest = None
            if task == "T1" and supplement and asset.get("sha256") == supplement.get("sha256") and asset.get("sha256"):
                supplement_digest = hashlib.sha256(canonical(supplement).encode()).hexdigest()
                asset = {**supplement, **{k:v for k,v in asset.items() if v not in (None, "")}}
            ident = asset["id"]
            prefix, _, number = ident.split("_")
            row = {"source_id": ident, "filename_alias": ("na" if prefix == "photo" else "gpt") + "_" + number.zfill(3),
                   "kind": "background" if kind == "backgrounds" else "object"}
            if supplement_digest:
                row["provenance_supplement"] = {"source_record":"video-library-20261030:T2.assets",
                    "source_record_sha256":supplement_digest,"join":"exact source_id and source asset SHA256"}
            for key in PROVENANCE:
                if key in asset:
                    try:
                        row[key] = public_value(asset[key])
                    except ValueError:
                        # Provenance identities and license fields are required below;
                        # workstation implementation paths remain in private evidence.
                        if key in ("license", "license_url", "source_url", "author"):
                            raise
            result.append(row)
    if {r["source_id"] for r in result} != used:
        raise ValueError("Source catalog does not cover the used source IDs")
    return result


def card(domain, task, rows, repo_id):
    counts = Counter(r["task"] for r in rows)
    title = "Classic T1–T3" if domain == "classic" else f"Natural {task}"
    codec = "lossless RGB16 FFV1/MKV" if domain == "classic" else "lossless RGB8 H.264/MP4"
    command = "--domain classic" if domain == "classic" else f"--domain natural --task {task}"
    dest = "data/classic" if domain == "classic" else f"data/natural/{task}"
    lines = ["---", "language:", "- en", "task_categories:", "- video-classification", "---", "",
             f"# {title}: visual psychophysics source videos", "",
             f"{len(rows):,} source videos: " + ", ".join(f"{t} {n:,}" for t,n in sorted(counts.items())) + ".",
             f"Each video is 768×512, 180 frames at 60 fps (3 s), encoded as {codec}.", "",
             "T1 measures left/right motion; T2 identifies the side that briefly brightens; T3 identifies the vertical object.", "",
             "`catalog.jsonl` and `metadata.jsonl` contain source IDs, paths, ground truth, conditions and SHA256 identities. "
             "`aliases.csv` resolves the previous IDs and paths; `metadata-demo.csv` is a small metadata-only example. "
             "T1 classic uses SHA-bound canonical IDs; the other IDs are unchanged. Files are renamed without re-encoding.", "",
             "Download a complete task folder at a fixed commit (replace `COMMIT_SHA` with the revision shown under Files and versions):", "",
             "```bash", "python -m pip install -r requirements-download.txt",
             f"python download_dataset.py {command} --revision COMMIT_SHA --output data --workers 8",
             f"python read_example.py --root {dest}", "```", "",
             "For classic data, add `--task T1`, `--task T2` or `--task T3`; `--folder T3` copies that entire folder. "
             "For natural data, each task is a separate repository. Downloads record the resolved commit and verify source-video SHA256 values.", "",
             "Filenames and ground truth reveal labels. Pass only the video and task question to models or participants.", "",
             "T3 `verticalLeft_tiltRight` means the left object is vertical and the right object's upper end leans right; "
             "90° is `horizontal`. Source rotation signs are retained separately from this physical description.", "",
             "The full source set is retained. `primary_T123_scope` identifies the existing 9,456-video classic selection "
             "(all T1; 30-frame T2; T3 excluding 1.75° and 2.25°); it does not remove auxiliary sources."]
    if domain == "natural":
        lines += ["", "`source-assets.json`, existing asset files, attribution and license evidence retain the fixed source-number mapping. "
                  "`bg-na_001`/`obj-na_001` denote photographic sources and `gpt` denotes generated sources. "
                  "Consult each upstream license; digital RGB quantities are not calibrated display luminance."]
        if task == "T1":
            lines += ["", "T1 applies a moving sinusoidal modulation to a stationary linear-RGB background: "
                      "I = 0.5 B [1 + m W cos(φ)]. Spatial frequency is measured per 448-pixel aperture; "
                      "m is modulation depth. The corrected RGB source version is used."]
        if task in ("T2", "T3"):
            lines += ["", "The current spoon replacements retain their own source identities. `parent_sample_id` records lineage, "
                      "and does not make a retired source an alias of the replacement."]
    lines += ["", "Original source QA and generation inputs remain with their source-version names. "
              "Frozen generators reproduce legacy source paths; the current catalog and aliases provide the new reading view. "
              "`validation/source-content.jsonl` records the per-video decoded-content/name checks for this release.", ""]
    return "\n".join(lines)


def build(root, mapping_path, output, inventory_path):
    root, output = Path(root), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refuse to overwrite existing public staging")
    with Path(mapping_path).open() as stream:
        mapping = list(csv.DictReader(stream))
    classic, originals = source_records(root)
    prompts = {("natural", "T1"): (root / "plans/natural-t1-rgb-20261030/T1-prompt.txt").read_text(),
               ("natural", "T3"): (root / "plans/natural-t13-20261029/T3-prompt.txt").read_text()}
    rows = project_rows(mapping, originals, prompts)
    expected_counts = {("classic","T1"):3840,("classic","T2"):5760,("classic","T3"):2928,
                       ("natural","T1"):3840,("natural","T2"):3840,("natural","T3"):2736}
    if Counter((r["domain"],r["task"]) for r in rows) != expected_counts:
        raise ValueError("Current source denominators differ from the authorized release")
    inventories = {r["repo"]:r for r in json.loads(Path(inventory_path).read_text())}
    output.mkdir(parents=True, exist_ok=True)
    plan = {"schema": "source-release-plan-v1", "dataset_revision": REVISION,
            "publishable": False, "gate": "PENDING_ALL_VIDEO_DECODED_CONTENT_ACCEPTANCE", "repositories": []}
    here = Path(__file__).parent
    for (domain, task), repo_id in REPOS.items():
        selected = [r for r in rows if r["domain"] == domain and (task is None or r["task"] == task)]
        inventory = inventories[repo_id]
        old_media = [p for p in inventory["files"] if p.endswith((".mp4", ".mkv", ".mov"))]
        old_media_set = set(old_media)
        old_by_stem = {PurePosixPath(p).stem:p for p in old_media}
        if len(old_by_stem) != len(old_media):
            raise ValueError("Ambiguous public legacy media filename")
        for row in selected:
            old = row["path"] if row["path"] in old_media_set else old_by_stem.get(row["legacy_sample_id"])
            if not old:
                raise ValueError(f"Missing exact legacy public video: {row['legacy_sample_id']}")
            row["legacy_public_path"] = old
            row["alias_paths"] = [old] if old not in (row["path"],row["legacy_path"]) else []
        IdentityRegistry.from_rows(selected, dataset_revision=REVISION)
        folder = output / repo_id.split("/")[1]
        folder.mkdir()
        json_lines(folder / "catalog.jsonl", selected)
        json_lines(folder / "metadata.jsonl", selected)
        fields = ("dataset_revision", "domain", "task", "sample_id", "legacy_sample_id", "path", "legacy_path",
                  "legacy_public_path", "alias_paths", "video_sha256", "full_identity_sha256")
        with (folder / "aliases.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows({**r,"alias_paths":canonical(r["alias_paths"])} for r in selected)
        # Include both sides of every task represented, without any model outputs.
        demo = []
        for t in sorted({r["task"] for r in selected}):
            for truth in ("LEFT", "RIGHT"):
                demo.append(next(r for r in selected if r["task"] == t and r["truth"] == truth))
        with (folder / "metadata-demo.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=(*fields, "truth", "parameters", "condition_key"), extrasaction="ignore")
            writer.writeheader()
            writer.writerows({**r,"alias_paths":canonical(r["alias_paths"]),"parameters":canonical(r["parameters"]),"condition_key":canonical(r["condition_key"])} for r in demo)
        shutil.copyfile(here / "download_dataset.py", folder / "download_dataset.py")
        shutil.copyfile(here / "read_dataset_example.py", folder / "read_example.py")
        (folder / "requirements-download.txt").write_text("huggingface-hub>=0.21\nav>=14\n")
        (folder / "README.md").write_text(card(domain, task, selected, repo_id))
        if domain == "classic":
            # Frozen dataset-index/recipe remain unchanged. The reader and the
            # single-file runner accept new aliases while retaining source keys.
            for name in ("dataset.py", "dataset_identity.py", "dataset_names.py", "manifest.py"):
                target = folder / "reader/classic_precision" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(here / name, target)
            target = folder / "reproduction/cosmos_x.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(here / "cosmos_x.py", target)
        else:
            previous = Path(inventory_path).parent / repo_id.split("/")[1]
            target = folder / "reproduction/classic_precision/dataset_identity.py"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(here / "dataset_identity.py", target)
            for license_path in ("LICENSE", "LICENSES/GFDL-1.2.txt"):
                source = previous / license_path
                if source.is_file():
                    target = folder / license_path
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
            source = previous / "reproduction/src/stimuli/t3_precision_reference.py"
            if source.is_file():
                text = source.read_text()
                old_host = "'execution_host':'Fourier /home/yang/cvpr2027_benchmark'"
                if text.count(old_host) != 1:
                    raise ValueError("Unexpected public reference-code host field")
                target = folder / "reproduction/src/stimuli/t3_precision_reference.py"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text.replace(old_host, "'execution_host':'local'"))
        if domain == "natural":
            json_file(folder / "source-assets.json", {"source_number_policy":"fixed-original-source-suffix-v1", "assets":source_assets(root, task, selected)})
        desired_media = {r["path"] for r in selected}
        if len(old_media) != len(selected):
            raise ValueError(f"Current public media denominator differs: {repo_id}")
        copies = []
        metadata = inventory.get("file_metadata", {})
        for row in selected:
            old = row["legacy_public_path"]
            meta = metadata.get(old, {})
            remote_sha = (meta.get("lfs") or {}).get("sha256")
            if remote_sha != row["video_sha256"]:
                raise ValueError(f"Public source SHA256 differs: {old}")
            if old != row["path"]:
                copies.append({"src_path_in_repo":old, "path_in_repo":row["path"],
                               "src_revision":inventory["commit"], "video_sha256":row["video_sha256"],
                               "bytes":meta.get("bytes")})
        deleted = [p for p in inventory["files"] if p.startswith("results/") or (p in old_media and p not in desired_media)]
        retained = [p for p in inventory["files"] if p not in deleted and p not in old_media]
        new_files = [p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()]
        plan["repositories"].append({"repo_id":repo_id, "parent_commit":inventory["commit"], "source_videos":len(selected),
            "staging_folder":folder.name,"upload_files":sorted(new_files), "delete_files":sorted(deleted),
            "copy_files":copies,
            "delete_result_identities":[{"path_in_repo":p,"bytes":metadata[p]["bytes"],
                "blob_id":metadata[p]["blob_id"],"sha256":(metadata[p].get("lfs") or {}).get("sha256")}
                for p in sorted(deleted) if p.startswith("results/")],
            "retain_existing_files":sorted(p for p in retained if p not in new_files),
            "media_files":sorted(desired_media), "history_preserved_at":inventory["commit"]})
    return plan


def finalize(output, plan, acceptance_path):
    """Admit measured records only after exact whole-release identity coverage."""
    output = Path(output)
    with Path(acceptance_path).open() as stream:
        checks = [json.loads(line) for line in stream]
    by_id = {}
    for check in checks:
        key = check["domain"], check["sample_id"]
        if key in by_id:
            raise ValueError("Duplicate decoded-content receipt")
        if check.get("status") != "PASS" or not check.get("measured") or not check.get("method") or not check.get("tolerances"):
            raise ValueError("Incomplete or failing decoded-content receipt")
        comparisons = check.get("comparison")
        if not isinstance(comparisons,dict) or not comparisons or any(v is not True for v in comparisons.values()):
            raise ValueError("Missing or failing decoded-content comparisons")
        if not re.fullmatch(r"[0-9a-f]{64}",check.get("code_sha256","")):
            raise ValueError("Missing decoded-content measurement code identity")
        by_id[key] = public_value(check)
    required = set()
    bundles = []
    for repo in plan["repositories"]:
        folder = output / repo["staging_folder"]
        rows = [json.loads(line) for line in (folder / "catalog.jsonl").read_text().splitlines()]
        selected = []
        for r in rows:
            key = r["domain"], r["sample_id"]
            check = by_id.get(key)
            if not check or any(check.get(k) != r[k] for k in ("task", "video_sha256", "legacy_sample_id", "path",
                    "source_record_sha256", "decoded_stream_sha256")) or check.get("source_sha256") != r["video_sha256"]:
                raise ValueError(f"Decoded-content coverage/identity differs: {key}")
            measured_side = check["measured"].get("vertical_side" if r["task"] == "T3" else "side")
            if measured_side != r["truth"]:
                raise ValueError(f"Decoded-content measured side differs from GT: {key}")
            required.add(key); selected.append(check)
        bundles.append((repo, folder, rows, selected))
    if required != set(by_id):
        raise ValueError("Decoded-content receipts include unexpected source IDs")
    for repo, folder, rows, selected in bundles:
        (folder / "validation").mkdir(exist_ok=True)
        json_lines(folder / "validation/source-content.jsonl", selected)
        release = {"schema":"source-release-v1", "dataset_revision":REVISION,"source_videos":len(rows),
                   "catalog_sha256":sha256(folder / "catalog.jsonl"),
                   "source_content_qa_sha256":sha256(folder / "validation/source-content.jsonl"),
                   "previous_commit":repo["parent_commit"], "source_media_reencoded":False}
        json_file(folder / "release.json", release)
        repo["upload_files"] = sorted(set(repo["upload_files"]) | {"validation/source-content.jsonl", "release.json"})
        repo["upload_file_identities"] = [{"path_in_repo":p,"bytes":(folder/p).stat().st_size,
                                           "sha256":sha256(folder/p)} for p in repo["upload_files"]]
    plan.update(publishable=True, gate="PASS_ALL_VIDEO_DECODED_CONTENT_ACCEPTANCE")
    return plan


def verify_results_preservation(plan, preservation, archive_root=None):
    """Match every removed result to the SHA-verified retained old revision."""
    for repo in plan["repositories"]:
        removed = repo.get("delete_result_identities", [])
        if not removed:
            continue
        if preservation.get("status") != "PASS_ALL_59_PUBLIC_FILES_RETAINED" or preservation.get("repo_id") != repo["repo_id"] or preservation.get("commit") != repo["parent_commit"]:
            raise ValueError("Result-preservation receipt does not bind the publication parent")
        retained = {r["path_in_repo"]:r for r in preservation["files"]}
        if set(retained) != {r["path_in_repo"] for r in removed}:
            raise ValueError("Result-preservation receipt differs from the exact removal list")
        for old in removed:
            kept = retained[old["path_in_repo"]]
            if kept["status"] != "PASS" or kept["bytes"] != old["bytes"] or kept["remote_blob_id"] != old["blob_id"] or (old["sha256"] and kept["sha256"] != old["sha256"]):
                raise ValueError("Removed result differs from its retained identity")
            if archive_root is not None:
                path = Path(archive_root) / safe_relative(kept["retained_relative_path"])
                if not path.is_file() or path.stat().st_size != kept["bytes"] or sha256(path) != kept["sha256"]:
                    raise ValueError("Retained result file is absent or changed")


def publish_source_releases(output, plan, preservation, *, api=None, on_commit=None, archive_root=None):
    """Commit a finalized explicit whitelist, using existing host credentials.

    This callable performs network writes only when explicitly invoked. Media
    moves use server-side CommitOperationCopy from the frozen parent revision.
    Callers should persist each on_commit receipt because four repositories
    cannot share one atomic transaction. No token is accepted or copied here.
    """
    if not plan.get("publishable") or plan.get("gate") != "PASS_ALL_VIDEO_DECODED_CONTENT_ACCEPTANCE":
        raise ValueError("Publication blocked: decoded-content acceptance is incomplete")
    verify_results_preservation(plan, preservation, archive_root)
    output = Path(output)
    # Check every local file before any network mutation.
    for repo in plan["repositories"]:
        identities = repo.get("upload_file_identities", [])
        if {i["path_in_repo"] for i in identities} != set(repo["upload_files"]):
            raise ValueError("Unsealed upload whitelist")
        for ident in identities:
            path = output / repo["staging_folder"] / safe_relative(ident["path_in_repo"])
            if path.stat().st_size != ident["bytes"] or sha256(path) != ident["sha256"]:
                raise ValueError("Finalized public staging file changed")
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    for repo in plan["repositories"]:
        actual = api.repo_info(repo["repo_id"], repo_type="dataset", revision="main").sha
        if actual != repo["parent_commit"]:
            raise ValueError(f"Public HEAD changed: {repo['repo_id']} expected {repo['parent_commit']} got {actual}")
    receipts = []
    from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete
    for repo in plan["repositories"]:
        folder = output / repo["staging_folder"]
        operations = [CommitOperationCopy(src_path_in_repo=safe_relative(c["src_path_in_repo"]),
                         path_in_repo=safe_relative(c["path_in_repo"]),src_revision=c["src_revision"])
                      for c in repo["copy_files"]]
        operations += [CommitOperationDelete(path_in_repo=safe_relative(p)) for p in repo["delete_files"]]
        operations += [CommitOperationAdd(path_in_repo=safe_relative(p),path_or_fileobj=str(folder/p))
                       for p in repo["upload_files"]]
        result = api.create_commit(repo_id=repo["repo_id"], repo_type="dataset", revision="main",
            parent_commit=repo["parent_commit"], operations=operations, num_threads=8,
            commit_message="Publish SHA-bound T1-T3 identities, readable paths and source QA")
        receipt = {"repo_id":repo["repo_id"],"parent_commit":repo["parent_commit"],
                   "commit":result.oid,"copied_media":len(repo["copy_files"]),
                   "deleted_paths":len(repo["delete_files"]),"uploaded_files":len(repo["upload_files"])}
        receipts.append(receipt)
        if on_commit:
            on_commit(receipt)
    return receipts


def verify_published_sources(output, plan, receipts, *, api=None):
    """Read back each actual commit tree and verify all media and uploaded bytes."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    commits = {r["repo_id"]:r["commit"] for r in receipts}
    verified = []
    for repo in plan["repositories"]:
        commit = commits[repo["repo_id"]]
        info = api.repo_info(repo["repo_id"],repo_type="dataset",revision=commit,files_metadata=True)
        if info.sha != commit:
            raise ValueError("Publication readback did not resolve the requested commit")
        files = {s.rfilename:s for s in info.siblings}
        expected = set(repo["retain_existing_files"]) | set(repo["upload_files"]) | set(repo["media_files"])
        if set(files) != expected:
            raise ValueError(f"Publication tree differs: missing={len(expected-set(files))}, extra={len(set(files)-expected)}")
        folder = Path(output) / repo["staging_folder"]
        rows = [json.loads(line) for line in (folder / "catalog.jsonl").read_text().splitlines()]
        for row in rows:
            meta = files[row["path"]]
            if lfs_sha256(meta.lfs) != row["video_sha256"]:
                raise ValueError("Published media source SHA256 differs: " + row["sample_id"])
        for ident in repo["upload_file_identities"]:
            data = (folder / ident["path_in_repo"]).read_bytes()
            actual_sha = hashlib.sha256(data).hexdigest()
            if actual_sha != ident["sha256"]:
                raise ValueError("Local finalized upload changed during publication")
            meta = files[ident["path_in_repo"]]
            expected_blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if meta.size != len(data) or (lfs_sha256(meta.lfs) != actual_sha if meta.lfs else meta.blob_id != expected_blob):
                raise ValueError("Published upload content differs: " + ident["path_in_repo"])
        verified.append({"repo_id":repo["repo_id"],"commit":commit,"status":"PASS_EXACT_TREE_AND_CONTENT_IDENTITIES",
                         "source_videos":len(rows),"uploaded_files_verified":len(repo["upload_file_identities"]),"files":len(files)})
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "finalize"))
    parser.add_argument("--root", type=Path)
    parser.add_argument("--mapping", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="Private plan, outside public staging")
    parser.add_argument("--acceptance", type=Path)
    args = parser.parse_args(argv)
    if args.output.resolve() in args.plan.resolve().parents:
        parser.error("Private release plan must stay outside public staging")
    if args.command == "build":
        if not all((args.root, args.mapping, args.inventory)):
            parser.error("build needs --root, --mapping and --inventory")
        plan = build(args.root,args.mapping,args.output,args.inventory)
    else:
        if not args.acceptance:
            parser.error("finalize needs --acceptance")
        plan = finalize(args.output, json.loads(args.plan.read_text()), args.acceptance)
    json_file(args.plan, plan)
    print(canonical({"repositories":len(plan["repositories"]),"publishable":plan["publishable"],"gate":plan["gate"]}))


if __name__ == "__main__":
    main()
