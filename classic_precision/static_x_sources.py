"""Plan and CPU-export the current X static T1/T2/T3 inputs.

``plan`` reads metadata only. ``prepare --panel`` establishes equivalence to
VideoMediaIO's official OpenCV uint8 reader; ``prepare --all`` reuses that
qualification and exports only the 4/1/1 required frames. No model is loaded.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import socket
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path


NAME = "static-x-t123-20261104"
FROOT = str(Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve())
PROTOCOL = f"plans/{NAME}/protocol.json"
VERIFY = "runs/stimulus-preparation/current-t123-reverification-20261103"
CLASSIC = "plans/dataset-naming-migration/public-staging/vlm-psychophysics-classic-t1-t3/catalog.jsonl"
NATURAL = "plans/natural-t3-lr-paired-20261102/catalog-fourier.json"
CHOICES = ["LEFT", "RIGHT"]
CONTRACT = {"frames": 180, "fps": 60, "width": 768, "height": 512}
EXPECTED = {"classic/T1": 3840, "classic/T2": 120, "classic/T3": 114,
            "natural/T1": 3840, "natural/T2": 3840, "natural/T3": 2736}


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".static-x-", delete=False) as f:
        pending = Path(f.name)
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write("\n")
    os.replace(pending, path)


def logical_path(path, benchmark_root=FROOT):
    """Keep the configured workspace path; never adopt a ledger's storage realpath."""
    path = str(path)
    token = "/cvpr2027_benchmark/"
    require(token in path, f"Not a benchmark logical path: {path}")
    return str(Path(benchmark_root) / path.split(token, 1)[1])


def replaced_prompt(protocol, task, domain, original):
    spec = protocol["tasks"][task]
    require(original == spec["source_video_prompt_by_domain"][domain],
            f"Source prompt bytes differ: {domain}/{task}")
    result = original
    for rule in spec["minimal_replacements"]:
        require(result.count(rule["old"]) == rule["expected_occurrences"],
                f"Unexpected prompt replacement count: {domain}/{task}")
        result = result.replace(rule["old"], rule["new"])
    require(result == spec["prompt_by_domain"][domain], "Static prompt bytes differ")
    require(sha_bytes(result.encode()) == spec["prompt_sha256_by_domain"][domain],
            "Static prompt SHA differs")
    return result


def pair_parameters(row):
    params = {k: v for k, v in row["filename"].items() if k not in ("side", "trial")}
    if row["task"] == "T1":
        params["phase_index"] = row["condition_key"][3]
    if row["task"] == "T3" and float(params["angle"]) == 90:
        params["direction"] = "axis90"
    return {"domain": row["domain"], "task": row["task"], "visible": params}


def validate_pairs(items):
    groups = defaultdict(list)
    for item in items.values():
        groups[item["pair_key"]].append(item)
    bad = [k for k, v in groups.items()
           if len(v) != 2 or {x["row"]["truth"] for x in v} != set(CHOICES)]
    require(not bad, f"LR pair identity/denominator failed: {bad[:5]}")
    return len(groups)


def select_panel(items):
    selected = []
    for task in ("T1", "T2", "T3"):
        buckets = defaultdict(list)
        for case_id, item in items.items():
            if item["task"] != task:
                continue
            if item["domain"] == "classic":
                group = "classic"
            elif task == "T1":
                group = item["source_parameters"]["group"] + "/" + item["row"]["truth"]
            else:
                group = item["source_parameters"]["group"]
            buckets[group].append(case_id)
        require(len(buckets) == 5, f"Expected five fixed panel strata for {task}")
        for group in sorted(buckets):
            candidates = buckets[group]
            key = lambda cid: digest([NAME, "metadata-only-panel-v1", task, group, cid,
                                      items[cid]["source_parameters"],
                                      items[cid]["original_source"]["sha256"]])
            cid = min(candidates, key=key)
            selected.append({"case_id": cid, "task": task, "stratum": group,
                             "selection_sha256": key(cid), "pair_key": items[cid]["pair_key"]})
    require(len(selected) == 15 and len({x["case_id"] for x in selected}) == 15,
            "Panel must contain 15 unique fixed cases")
    return selected


def build_manifest(root, benchmark_root=FROOT):
    """Metadata-only construction; deliberately does not open any video."""
    root = Path(root)
    protocol_path = root / PROTOCOL
    protocol = read(protocol_path)
    complete_path = root / VERIFY / "complete-receipt.json"
    complete = read(complete_path)
    require(complete["status"] == "PASS_ALL_19872_CURRENT_VIDEOS"
            and complete["videos"] == 19872 and not complete["failed_ids"], "Full source QA not PASS")
    classic_path, natural_path = root / CLASSIC, root / NATURAL
    require(file_sha(classic_path) == complete["source_input_sha256"][CLASSIC],
            "Classic catalog is not the complete-receipt input")
    classic_rows = read_jsonl(classic_path)
    classic = {r["sample_id"]: r for r in classic_rows}
    require(len(classic) == len(classic_rows), "Duplicate classic catalog IDs")
    natural = read(natural_path)["items"]
    require(len(natural) == 10416, "Canonical natural catalog count differs")
    evidence = [{"path": logical_path(str(protocol_path), benchmark_root), "sha256": file_sha(protocol_path)},
                {"path": logical_path(str(complete_path), benchmark_root), "sha256": file_sha(complete_path)},
                {"path": logical_path(str(classic_path), benchmark_root), "sha256": file_sha(classic_path)},
                {"path": logical_path(str(natural_path), benchmark_root), "sha256": file_sha(natural_path)}]
    all_rows, grouped = [], defaultdict(list)
    for domain in ("classic", "natural"):
        path = root / VERIFY / f"full-{domain}" / "per-video.jsonl"
        binding = next(x for x in complete["journals"] if f"full-{domain}/" in x["path"])
        require(file_sha(path) == binding["sha256"], f"Unbound {domain} journal")
        evidence.append({"path": logical_path(str(path), benchmark_root), "sha256": binding["sha256"]})
        rows = read_jsonl(path)
        require(len({x["sample_id"] for x in rows}) == len(rows), "Duplicate ledger IDs")
        require(Counter(x["task"] for x in rows) ==
                {t: complete["counts"][domain + "/" + t] for t in ("T1", "T2", "T3")},
                "Ledger task counts differ")
        for row in rows:
            sid, task = row["sample_id"], row["task"]
            require(row["domain"] == domain and row["status"] == "PASS"
                    and row["decoded_frames"] == 180
                    and all(row[k] == v for k, v in CONTRACT.items() if k != "frames"),
                    f"Source contract failed: {sid}")
            if domain == "classic":
                source = classic[sid]
                original = source["prompt"]
                require(source["video_sha256"] == row["sha256"]
                        and source["truth"] == row["truth"] and source["task"] == task
                        and source["parameters"] == row["catalog_parameters"]
                        and source["decoded_stream_sha256"] == row["decoded_stream_sha256"],
                        f"Classic source/catalog mismatch: {sid}")
            else:
                source = natural[sid]
                original = source["row"]["prompt"]
                videos = [x for x in source["content"] if x["type"] == "video"]
                texts = [x["text"] for x in source["content"] if x["type"] == "text"]
                require(source["sample_id"] == source["row"]["sample_id"] == sid
                        and len(videos) == 1 and videos[0]["sha256"] == row["sha256"]
                        and logical_path(videos[0]["path"], benchmark_root)
                        == logical_path(row["path"], benchmark_root)
                        and source["row"]["truth"] == row["truth"] and source["task"] == task
                        and source["row"]["choices"] == CHOICES and texts == [original]
                        and source["row"]["source_parameters"] == row["catalog_parameters"]
                        and source["video_contract"] == CONTRACT,
                        f"Natural source/catalog mismatch: {sid}")
            prompt = replaced_prompt(protocol, task, domain, original)
            if task != "T1":
                onset, offset = row["event_frame_interval_half_open"]
                require(offset - onset == row["event_frame_count"] == 30
                        and row["event_frame_sha256"] != row["baseline_frame_sha256"]
                        and 0 <= onset < onset + 15 < offset <= 180, f"Event contract differs: {sid}")
                selector = row["event_frame_sha256"]
            else:
                selector = sid  # Never collapse ordered T1 questions by an unordered frame set.
            key = (domain, task, selector, prompt, tuple(CHOICES))
            grouped[key].append(row)
            all_rows.append(row)
    items = {}
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: r["sample_id"])
        row = rows[0]
        require(len({r["truth"] for r in rows}) == 1, f"HOLD: source-group GT conflict: {key[:3]}")
        pp = pair_parameters(row)
        require(all(pair_parameters(r) == pp for r in rows), "Alias visible parameters differ")
        domain, task, _, prompt, _ = key
        cid = "sx_" + task.lower() + "_" + digest(list(key))[:24]
        aliases = []
        for alias in rows:
            a = {"sample_id": alias["sample_id"], "original_video_ID": alias["sample_id"],
                 "domain": domain, "task": task,
                 "path": logical_path(alias["path"], benchmark_root), "sha256": alias["sha256"],
                 "truth": alias["truth"], "source_parameters": alias["catalog_parameters"],
                 "condition_key": alias["condition_key"],
                 "catalog_record_sha256": alias["catalog_record_sha256"]}
            if task != "T1":
                a["onset_frame"] = alias["event_frame_interval_half_open"][0]
            aliases.append(a)
        item = {"sample_id": cid, "case_id": cid, "domain": domain, "task": task,
                "original_video_ID": row["sample_id"], "original_video_aliases": aliases,
                "original_source": {"path": aliases[0]["path"], "sha256": row["sha256"]},
                "source_parameters": row["catalog_parameters"],
                "pair_key": "sxp_" + digest(pp)[:24], "pair_parameters": pp,
                "row": {"sample_id": cid, "task": task, "truth": row["truth"],
                        "prompt": prompt, "choices": CHOICES,
                        "source_parameters": row["catalog_parameters"]},
                "video_contract": CONTRACT,
                "image_contract": {"count": 4 if task == "T1" else 1, "width": 768, "height": 512},
                "target_frame_indices": [0, 1, 2, 3] if task == "T1" else
                                        [row["event_frame_interval_half_open"][0] + 15],
                "source_ledger": {"pixel_hash_format": "RGB48le" if domain == "classic" else "RGB24",
                                  "decoded_stream_sha256": row["decoded_stream_sha256"],
                                  "event_frame_sha256": row.get("event_frame_sha256"),
                                  "catalog_record_sha256": row["catalog_record_sha256"]}}
        if task != "T1":
            item["onset_frame"] = row["event_frame_interval_half_open"][0]
        items[cid] = item
    counts = Counter(x["domain"] + "/" + x["task"] for x in items.values())
    require(counts == EXPECTED and len(all_rows) == 19872, f"Initial manifest counts differ: {counts}")
    pairs = validate_pairs(items)
    panel = select_panel(items)
    return {"schema": "static-x-case-manifest-v1", "source_catalog": "static-x-catalog-v1",
            "protocol_id": NAME, "status": "METADATA_PASS_RGB8_PROJECTION_PENDING",
            "benchmark_root": str(benchmark_root), "source_evidence": evidence,
            "counts": dict(counts), "source_video_count": len(all_rows), "pair_count": pairs,
            "items": items, "panel": panel, "panel_case_ids": [x["case_id"] for x in panel],
            "panel_selection": "SHA256 of fixed seed, stratum, case ID, scientific parameters and source SHA; no model results",
            "preprojection_grouping": "T1 identity; T23 complete ledger event hash; final grouping requires actual ordered RGB8 image hashes + exact prompt + choices",
            "model_calls": 0}


def runtime_identity():
    import cv2
    import numpy as np
    import PIL
    vllm = importlib.util.find_spec("vllm")
    require(vllm is not None and vllm.submodule_search_locations, "Official vllm reader is required")
    package = Path(next(iter(vllm.submodule_search_locations)))
    reader_tree = {str(p.relative_to(package)): file_sha(p)
                   for p in sorted((package / "multimodal").rglob("*.py"))}
    require("multimodal/media/video.py" in reader_tree
            and "multimodal/media/image.py" in reader_tree, "Official reader sources unavailable")
    return {"python": platform.python_version(), "opencv": cv2.__version__,
            "numpy": np.__version__, "PIL": PIL.__version__,
            "vllm": importlib.metadata.version("vllm"),
            "vllm_multimodal_source_sha256": digest(reader_tree),
            "vllm_multimodal_source_files": len(reader_tree),
            "VideoMediaIO_source_sha256": reader_tree["multimodal/media/video.py"],
            "ImageMediaIO_source_sha256": reader_tree["multimodal/media/image.py"],
            "opencv_build_sha256": sha_bytes(cv2.getBuildInformation().encode()),
            "implementation_sha256": file_sha(__file__)}


def machine_identity():
    boot = Path("/proc/sys/kernel/random/boot_id")
    return {"host": socket.gethostname(), "boot_id": boot.read_text().strip() if boot.exists() else ""}


def stat_identity(path):
    s = Path(path).stat()
    return {k: getattr(s, "st_" + k) for k in ("dev", "ino", "size", "mtime_ns", "ctime_ns")}


def verify_source(source, out, machine):
    path = Path(source["path"])
    before = stat_identity(path)
    cache_path = out / "source-cache" / (digest(str(path)) + ".json")
    expected = {"path": str(path), "sha256": source["sha256"], "stat": before, "machine": machine}
    if cache_path.exists() and read(cache_path) == expected:
        return "same_host_stat_bound_sha256"
    require(file_sha(path) == source["sha256"], f"Source SHA differs: {path}")
    require(stat_identity(path) == before, f"Source changed while hashing: {path}")
    write(cache_path, expected)
    return "fresh_file_sha256"


def read_keyframes(item):
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    source = item["original_source"]["path"]
    indices = item["target_frame_indices"]
    cap = cv2.VideoCapture(source)
    require(cap.isOpened(), f"OpenCV cannot open {source}")
    frames, positions = [], []
    try:
        require(int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))) == 180
                and abs(cap.get(cv2.CAP_PROP_FPS) - 60) < 1e-6
                and int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 768
                and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 512, "OpenCV video contract differs")
        for index in indices:
            if item["task"] != "T1":
                require(cap.set(cv2.CAP_PROP_POS_FRAMES, index), f"Frame seek failed: {source}:{index}")
            before = cap.get(cv2.CAP_PROP_POS_FRAMES)
            ok, bgr = cap.read()
            after = cap.get(cv2.CAP_PROP_POS_FRAMES)
            require(ok and abs(before - index) < 1e-6 and abs(after - index - 1) < 1e-6,
                    f"Frame seek/read position differs: {source}:{index}, {before}, {after}")
            require(bgr.dtype == np.uint8 and bgr.shape == (512, 768, 3), "Reader changed uint8 geometry")
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if item["domain"] == "natural" and item["task"] != "T1":
                require(sha_bytes(rgb.tobytes()) == item["source_ledger"]["event_frame_sha256"],
                        f"Natural event RGB8 differs from full-frame source QA: {source}")
            frames.append(rgb)
            positions.append({"frame_index": index, "before": before, "after": after})
    finally:
        cap.release()
    return frames, positions


def check_png(path, pixel_sha):
    import numpy as np
    from PIL import Image
    with Image.open(path) as im:
        require(im.format == "PNG" and im.mode == "RGB" and im.size == (768, 512) and not im.info,
                f"PNG geometry/mode/metadata differs: {path}")
        require(sha_bytes(np.asarray(im).tobytes()) == pixel_sha, f"PNG roundtrip pixels differ: {path}")
    return file_sha(path)


def cas_png(frame, out):
    from PIL import Image
    pixel_sha = sha_bytes(frame.tobytes())
    path = out / "images" / pixel_sha[:2] / (pixel_sha + ".png")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd, name = tempfile.mkstemp(prefix=".static-x-", suffix=".png", dir=path.parent)
        os.close(fd)
        pending = Path(name)
        try:
            Image.fromarray(frame, mode="RGB").save(pending, format="PNG")
            check_png(pending, pixel_sha)
            os.replace(pending, path)
        finally:
            pending.unlink(missing_ok=True)
    return {"type": "image", "path": str(path), "sha256": check_png(path, pixel_sha),
            "pixel_sha256": pixel_sha}


def prepare_case(item, out, runtime, machine):
    start = time.monotonic()
    checks = Counter(verify_source(alias, out, machine) for alias in item["original_video_aliases"])
    cache_path = out / "case-cache" / (item["case_id"] + ".json")
    binding = {"manifest_item_sha256": digest(item), "runtime": runtime}
    if cache_path.exists():
        cached = read(cache_path)
        if cached.get("binding") == binding:
            result = cached["item"]
            for image in result["content"]:
                if image["type"] == "image":
                    require(check_png(image["path"], image["pixel_sha256"]) == image["sha256"], "Cached PNG SHA differs")
            return result, {"case_id": item["case_id"], "domain": item["domain"], "task": item["task"], "cache_reused": True,
                            "source_sha_checks": dict(checks), "seconds": time.monotonic() - start}
    source_path = item["original_source"]["path"]
    source_before = read(out / "source-cache" / (digest(source_path) + ".json"))["stat"]
    require(stat_identity(source_path) == source_before, "Source changed after SHA verification")
    frames, positions = read_keyframes(item)
    require(stat_identity(source_path) == source_before, "Source changed during decode")
    content = []
    for index, frame in zip(item["target_frame_indices"], frames):
        image = cas_png(frame, out)
        image["frame_index"] = index
        content.append(image)
    content.append({"type": "text", "text": item["row"]["prompt"]})
    result = copy.deepcopy(item)
    result.update(content=content, input_modality="native_image_list", accuracy_eligible=True,
                  source_qa="PASS", reader="OpenCV uint8 BGR->RGB; no resize/crop",
                  decoded_frame_positions=positions,
                  source_frame_top_left_rgb=[int(x) for x in frames[0][0, 0]])
    write(cache_path, {"binding": binding, "item": result})
    return result, {"case_id": item["case_id"], "domain": item["domain"], "task": item["task"], "cache_reused": False,
                    "source_sha_checks": dict(checks), "seconds": time.monotonic() - start}


def qualify_reader(manifest, prepared, out, runtime, manifest_sha):
    import numpy as np
    import torch
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO
    require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
    torch.set_num_threads(1)
    loader = VideoMediaIO(ImageMediaIO(), video_backend="opencv", num_frames=180, fps=-1)
    records = []
    for cid in manifest["panel_case_ids"]:
        item = prepared[cid]
        before = stat_identity(item["original_source"]["path"])
        frames, meta = loader.load_file(Path(item["original_source"]["path"])).media
        require(frames.dtype == np.uint8 and frames.shape == (180, 512, 768, 3)
                and list(meta["frames_indices"]) == list(range(180)), "Official reader contract differs")
        require(stat_identity(item["original_source"]["path"]) == before, "Source changed in official reader")
        image_hashes = [x["pixel_sha256"] for x in item["content"] if x["type"] == "image"]
        official = [sha_bytes(np.ascontiguousarray(frames[i]).tobytes()) for i in item["target_frame_indices"]]
        require(image_hashes == official, f"Keyframe path != official uint8 reader: {cid}")
        top_left = [int(x) for x in frames[0, 0, 0]]
        require(all([int(x) for x in frames[i, 0, 0]] == top_left for i in item["target_frame_indices"]),
                f"Target top-left differs from video frame0; Gemma padding needs review: {cid}")
        records.append({"case_id": cid, "original_source": item["original_source"],
                        "target_frame_indices": item["target_frame_indices"],
                        "official_target_pixel_sha256": official, "video_frame0_top_left_rgb": top_left})
        del frames
    require(not torch.cuda.is_initialized(), "Official CPU reader initialized CUDA")
    value = {"schema": "static-x-reader-qualification-v1", "status": "PASS",
             "manifest_sha256": manifest_sha, "runtime": runtime, "records": records,
             "scope": "All 15 fixed panel cases, every selected target vs official VideoMediaIO OpenCV RGB8; classic ledger RGB48 is not asserted RGB8-equal",
             "model_calls": 0}
    write(out / "reader-qualification.json", value)
    return value


def deduplicate(prepared, panel_ids):
    """Group by actual model input; truth is checked, never used to hide conflict."""
    groups = defaultdict(list)
    for cid, item in prepared.items():
        images = [x["pixel_sha256"] for x in item["content"] if x["type"] == "image"]
        key = canonical([images, item["row"]["prompt"], item["row"]["choices"]])
        groups[key].append(cid)
    final, remap, collapsed, conflicts = {}, {}, [], []
    panel_set = set(panel_ids)
    for key, ids in groups.items():
        if len({prepared[cid]["row"]["truth"] for cid in ids}) != 1:
            conflicts.append({"input_sha256": sha_bytes(key.encode()), "case_ids": sorted(ids),
                              "truths": {cid: prepared[cid]["row"]["truth"] for cid in ids}})
            continue
        pinned = sorted(panel_set.intersection(ids))
        require(len(pinned) <= 1, f"HOLD: fixed panel cases collapse after RGB8 projection: {pinned}")
        keep = pinned[0] if pinned else min(ids)
        item = copy.deepcopy(prepared[keep])
        item["model_input_sha256"] = sha_bytes(key.encode())
        item["model_input_signature"] = {
            "ordered_image_pixel_sha256": [x["pixel_sha256"] for x in item["content"] if x["type"] == "image"],
            "prompt_utf8_sha256": sha_bytes(item["row"]["prompt"].encode("utf-8")),
            "choices": item["row"]["choices"]}
        item["preprojection_case_aliases"] = sorted(ids)
        item["pair_keys"] = sorted({prepared[cid]["pair_key"] for cid in ids})
        aliases = []
        for cid in sorted(ids):
            remap[cid] = keep
            for original_alias in prepared[cid]["original_video_aliases"]:
                alias = copy.deepcopy(original_alias)
                alias.update(preprojection_case_id=cid, pair_key=prepared[cid]["pair_key"],
                             pair_parameters=copy.deepcopy(prepared[cid]["pair_parameters"]))
                aliases.append(alias)
        item["original_video_aliases"] = sorted(aliases, key=lambda x: x["sample_id"])
        final[keep] = item
        if len(ids) > 1:
            collapsed.append({"retained_case_id": keep, "case_ids": sorted(ids),
                              "model_input_sha256": item["model_input_sha256"]})
    return dict(sorted(final.items())), remap, collapsed, conflicts


def prepare(manifest_path, output_root, panel=False, workers=2):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for CPU preparation")
    require(1 <= workers <= 4, "Use 1-4 CPU workers to protect the active Gemma queue")
    manifest_path, out = Path(manifest_path), Path(output_root)
    require(out.is_absolute(), "Output root must be an explicit absolute SSD path")
    manifest = read(manifest_path)
    require(manifest["schema"] == "static-x-case-manifest-v1"
            and manifest["counts"] == EXPECTED, "Unknown/unsealed manifest")
    runtime, machine, msha = runtime_identity(), machine_identity(), file_sha(manifest_path)
    out.mkdir(parents=True, exist_ok=True)
    selected = manifest["panel_case_ids"] if panel else list(manifest["items"])
    qualification_path = out / "reader-qualification.json"
    if not panel:
        require(qualification_path.exists(), "Run prepare --panel and inspect its throughput before --all")
        q = read(qualification_path)
        require(q["status"] == "PASS" and q["manifest_sha256"] == msha and q["runtime"] == runtime
                and [r["case_id"] for r in q["records"]] == manifest["panel_case_ids"],
                "Official reader qualification is not bound to this manifest/runtime")
    prepared, timings = {}, []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = {pool.submit(prepare_case, manifest["items"][cid], out, runtime, machine): cid for cid in selected}
        for future in concurrent.futures.as_completed(tasks):
            item, timing = future.result()
            prepared[item["case_id"]] = item
            timings.append(timing)
            done = len(prepared)
            if done == 1 or done % 100 == 0 or done == len(selected):
                elapsed = time.monotonic() - started
                print(canonical({"phase": "CPU_PNG_QA", "done": done, "total": len(selected),
                                 "seconds": round(elapsed, 3), "cases_per_second": round(done / elapsed, 3),
                                 "ETA_remaining_s": round((len(selected) - done) * elapsed / done)}), flush=True)
    export_seconds = time.monotonic() - started
    if panel:
        qualification = qualify_reader(manifest, prepared, out, runtime, msha)
    else:
        qualification = read(qualification_path)
        for record in qualification["records"]:
            item = prepared[record["case_id"]]
            require([x["pixel_sha256"] for x in item["content"] if x["type"] == "image"]
                    == record["official_target_pixel_sha256"], "Formal panel pixels changed after qualification")
    suffix = "panel" if panel else "all"
    try:
        final, remap, collapse, conflicts = deduplicate(prepared, manifest["panel_case_ids"])
    except ValueError as error:
        write(out / f"receipt-{suffix}.json", {"schema": "static-x-prepare-receipt-v1",
              "status": "HOLD", "error": str(error), "manifest_sha256": msha,
              "runtime": runtime, "model_calls": 0})
        raise
    source_checks = Counter()
    strata = defaultdict(list)
    for timing in timings:
        source_checks.update(timing["source_sha_checks"])
        strata[timing["domain"] + "/" + timing["task"]].append(timing["seconds"])
    serial_estimate = sum(sum(values) / len(values) * EXPECTED[stratum]
                          for stratum, values in strata.items())
    measured_parallel_factor = export_seconds / sum(x["seconds"] for x in timings)
    receipt = {"schema": "static-x-prepare-receipt-v1", "status": "HOLD_GT_CONFLICT" if conflicts else "PASS",
               "manifest_sha256": msha, "runtime": runtime, "machine": machine,
               "output_root": str(out), "resolved_output_root": str(out.resolve()),
               "source_evidence": manifest["source_evidence"],
               "selection": suffix, "workers": workers, "preprojection_cases": len(prepared),
               "original_scientific_video_aliases": sum(len(x["original_video_aliases"]) for x in prepared.values()),
               "final_unique_cases": len(final), "additional_RGB8_input_collapses": collapse,
               "GT_conflicts": conflicts, "case_id_mapping": remap,
               "counts": dict(Counter(x["domain"] + "/" + x["task"] for x in final.values())),
               "source_sha_checks": dict(source_checks), "PNG_export_seconds": export_seconds,
               "seconds_including_reader_qualification": time.monotonic() - started,
               "panel_based_all_export_ETA_s": round(serial_estimate * measured_parallel_factor) if panel else None,
               "measured_seconds_per_case_by_stratum": {k: sum(v) / len(v) for k, v in strata.items()},
               "cache_reused_cases": sum(x["cache_reused"] for x in timings),
               "ETA_scope": "Stratum-weighted fixed-panel estimate includes every alias source SHA, exact target decode and PNG roundtrip with measured worker overlap; cache/codec cost may differ",
               "reader_qualification_sha256": file_sha(qualification_path), "model_calls": 0}
    write(out / f"receipt-{suffix}.json", receipt)
    require(not conflicts, "HOLD: GT conflicts; inspect receipt, no model-ready catalog was written")
    if not panel:
        if not collapse:
            require(receipt["counts"] == EXPECTED and validate_pairs(final) == manifest["pair_count"],
                    "Final denominator/pair counts differ")
        # Legal additional input collapse lowers the real denominator. All original
        # pair memberships survive in pair_keys for operations' connected allocation.
    catalog = {"schema": "static-x-catalog-v1", "source_catalog": "static-x-catalog-v1",
               "protocol_id": NAME, "status": "SOURCE_QA_PASS_MODEL_INPUT_PENDING", "items": final,
               "assigned_ids": list(final), "panel_case_ids": [remap[x] for x in manifest["panel_case_ids"]],
               "manifest_sha256": msha, "reader_qualification_sha256": file_sha(qualification_path),
               "prepare_receipt_sha256": file_sha(out / f"receipt-{suffix}.json"), "model_calls": 0}
    write(out / f"catalog-{suffix}.json", catalog)
    print(canonical({"phase": "COMPLETE", "catalog": str(out / f"catalog-{suffix}.json"),
                     "receipt": str(out / f"receipt-{suffix}.json"), "cases": len(final),
                     "panel_based_all_export_ETA_s": receipt["panel_based_all_export_ETA_s"]}), flush=True)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Metadata only: seal manifest and fixed 15-case panel")
    p.add_argument("--root", required=True, help="Local benchmark metadata root")
    p.add_argument("--benchmark-root", default=FROOT, help="Logical source root on the execution host")
    p.add_argument("--output", required=True, help="Manifest JSON output path")
    p = sub.add_parser("prepare", help="CPU official-reader qualification / target PNG export")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-root", required=True, help="Explicit execution-host SSD output directory")
    p.add_argument("--workers", type=int, default=2)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--panel", action="store_true")
    which.add_argument("--all", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        result = build_manifest(args.root, args.benchmark_root)
        write(args.output, result)
        print(canonical({"manifest": args.output, "sha256": file_sha(args.output), "counts": result["counts"],
                         "pair_count": result["pair_count"], "panel": result["panel"]}), flush=True)
    else:
        prepare(args.manifest, args.output_root, panel=args.panel, workers=args.workers)


if __name__ == "__main__":
    main()
