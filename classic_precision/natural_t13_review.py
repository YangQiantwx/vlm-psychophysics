"""Read-only design/source audit and local paired-video review for natural T1/T3.

The generation module is never edited by this module. No model results are read.
Run `design` during CPU construction and `finish` only after every pair is closed.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from . import natural_t13_stimuli as s


def csv_write(path, rows):
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def cv_axis(alpha):
    m = cv2.moments(np.asarray(alpha, dtype=np.float64))
    # Analytic orientation of the major eigenvector of the central-moment matrix.
    return float(np.degrees(0.5 * np.arctan2(2 * m["mu11"], m["mu20"] - m["mu02"])))


def angle_separation(root):
    """Measure the actual difference between reference and tilted mask axes."""
    root = Path(root)
    p, d, r = s.paths(root)
    rows = []
    for block in s.read(p / "allocation.json")["blocks"]:
        if block["task"] != "T3":
            continue
        q = s.read(d / "qa" / (block["block_id"] + ".json"))
        zero, tilt = q["patches"]
        actual = abs(s.angle_diff(tilt["measured_axis_deg"], zero["measured_axis_deg"]))
        assert abs(s.angle_diff(zero["measured_axis_deg"], 90)) < abs(
            s.angle_diff(tilt["measured_axis_deg"], 90)
        )
        rows.append(
            {
                "block_id": block["block_id"],
                "angle": block["angle"],
                "actual_mask_axis_difference_deg": actual,
                "error_deg": actual - block["angle"],
                "closer_to_vertical_GT_correct": True,
            }
        )
    assert len(rows) == 1368
    csv_write(r / "angle-separation.csv", rows)
    result = {
        "T3_max_actual_tilt_error_deg": max(abs(x["error_deg"]) for x in rows),
        "T3_closer_to_vertical_GT_pairs_verified": 1368,
        "angle_separation_csv_sha256": s.sha(r / "angle-separation.csv"),
    }
    if (r / "source-audit.json").exists():
        audit = s.read(r / "source-audit.json")
        audit.update(result)
        s.write(r / "source-audit.json", audit)
    return result


def design(root):
    root = Path(root)
    p, d, r = s.paths(root)
    alloc = s.read(p / "allocation.json")
    catalog = s.read(p / "catalog.json")
    blocks = alloc["blocks"]
    assert alloc["criteria_sha256"] == s.sha(p / "criteria.json") and alloc[
        "catalog_sha256"
    ] == s.sha(p / "catalog.json")
    assert alloc["model_calls"] == 0 and not alloc["sample_selection_uses_answers"]
    assert len(blocks) == 3288 and len({b["block_id"] for b in blocks}) == 3288
    samples = [c for b in blocks for c in b["cases"]]
    assert len({c["sample_id"] for c in samples}) == 6576
    for b in blocks:
        assert Counter(c["truth"] for c in b["cases"]) == {"LEFT": 1, "RIGHT": 1}
        assert all(re.fullmatch(r"n[0-9a-f]{24}", c["sample_id"]) for c in b["cases"])
    cells = []
    for task, ncells in [("T1", 80), ("T3", 29)]:
        taskblocks = [b for b in blocks if b["task"] == task]
        assert len({b["background_id"] for b in taskblocks}) == 200
        groups = ["photo", "gpt"] if task == "T1" else ["_".join(g) for g in s.GROUPS]
        for ci in range(ncells):
            cell = [b for b in taskblocks if b["cell"] == ci]
            if task == "T1":
                assert sorted(b["phase_index"] for b in cell) == list(range(24))
            for group in groups:
                rows = [b for b in cell if b["group"] == group]
                n = 6 if task == "T3" and ci == 28 else 12
                assert len(rows) == n
                assert len({b["background_id"] for b in rows}) == n
                if task == "T3":
                    assert len({b["object_id"] for b in rows}) == n
                    assert len({b["onset_index"] for b in rows}) == n
                    if ci < 28:
                        assert Counter(b["tilt_sign"] for b in rows) == {-1: 6, 1: 6}
                cells.append(
                    {
                        "task": task,
                        "cell": ci,
                        "group": group,
                        "SF": rows[0].get("sf", ""),
                        "m": rows[0].get("m", ""),
                        "angle": rows[0].get("angle", ""),
                        "videos": 2 * n,
                        "LEFT": n,
                        "RIGHT": n,
                        "unique_backgrounds": n,
                        "unique_objects": n if task == "T3" else 0,
                        "distinct_phase_or_onset": len(
                            {b.get("phase_index", b.get("onset_index")) for b in rows}
                        ),
                    }
                )
    non90 = [
        b
        for b in blocks
        if b["task"] == "T3" and b["angle"] != 90 and b["group"] == "photo_photo"
    ]
    assert Counter(b["onset_index"] for b in non90) == {i: 14 for i in range(24)}
    for onset in range(24):
        assert Counter(b["tilt_sign"] for b in non90 if b["onset_index"] == onset) == {
            -1: 7,
            1: 7,
        }
    eligibility = s.read(p / "object-eligibility.json")
    pairids = eligibility["eligible_pairs"]
    bf = {a["asset_id"]: a for a in s.read(p / "background-features.json")}
    of = {a["asset_id"]: a for a in eligibility["objects"]}
    assets = {a["id"]: a for a in catalog["backgrounds"] + catalog["objects"]}
    metrics = {}
    for name, task, field, ids, get_features in [
        (
            "T1_bg",
            "T1",
            "background_id",
            sorted({a["pair_id"] for a in catalog["backgrounds"]}),
            lambda k: bf["photo_" + k]["T1"] + bf["gpt_" + k]["T1"],
        ),
        (
            "T3_bg",
            "T3",
            "background_id",
            sorted({a["pair_id"] for a in catalog["backgrounds"]}),
            lambda k: bf["photo_" + k]["T3"] + bf["gpt_" + k]["T3"],
        ),
        (
            "T3_obj",
            "T3",
            "object_id",
            pairids,
            lambda k: of["photo_" + k]["features"] + of["gpt_" + k]["features"],
        ),
    ]:
        pool = np.array([get_features(k) for k in ids])
        z = (pool - pool.mean(0)) / np.maximum(pool.std(0), 1e-10)
        lookup = {k: i for i, k in enumerate(ids)}
        fam = [
            assets["gpt_" + k].get(
                "family",
                assets["photo_" + k].get(
                    "family", "kitchen" if k == "obj_051" else "workbench"
                ),
            )
            for k in ids
        ]
        rates = {k: v / len(ids) for k, v in Counter(fam).items()}
        shifts = []
        deviations = []
        used = []
        for ci in range(80 if task == "T1" else 29):
            rows = [
                b
                for b in blocks
                if b["task"] == task
                and b["cell"] == ci
                and b["group"] == ("photo" if task == "T1" else "photo_photo")
            ]
            ks = [b[field].split("_", 1)[1] for b in rows]
            indices = [lookup[k] for k in ks]
            used += ks
            shifts.append(z[indices].mean(0))
            counts = Counter(fam[i] for i in indices)
            deviations.extend(
                abs(counts[k] - len(rows) * rate) for k, rate in rates.items()
            )
        counts = Counter(used)
        assert (
            set(counts) == set(ids) and max(counts.values()) - min(counts.values()) <= 1
        )
        maximum = float(np.max(abs(np.array(shifts))))
        assert maximum <= 0.5 + 1e-12 and max(deviations) <= 2 + 1e-12
        assert abs(maximum - alloc["diagnostics"][name]["max_mean_shift_SD"]) < 1e-10
        metrics[name] = {
            "max_mean_shift_SD": maximum,
            "max_family_count_deviation": max(deviations),
            "reuse_min": min(counts.values()),
            "reuse_max": max(counts.values()),
            "semantic_pairs": len(ids),
        }
    source_records = []
    manual_masks = []
    usedobjects = {b["object_id"] for b in blocks if b["task"] == "T3"}
    for a in catalog["backgrounds"] + [
        x for x in catalog["objects"] if x["id"] in usedobjects
    ]:
        source = root / "assets/library" / a["path"]
        assert s.sha(source) == a["sha256"], a["id"]
        if a["id"].startswith("photo_"):
            assert a.get("author") and a.get("license") and a.get("source_url"), a["id"]
        else:
            assert a.get("prompt") or a.get("full_prompt"), a["id"]
        if "manual_source_polygon" in json.dumps(a):
            manual_masks.append(a["id"])
        source_records.append(
            {
                "id": a["id"],
                "source_sha256": a["sha256"],
                "source_url": a.get("source_url"),
                "author": a.get("author"),
                "license": a.get("license"),
                "generation_prompt": a.get("prompt", a.get("full_prompt")),
            }
        )
    correction = (
        root
        / "assets/library/extensions/t2-natural-20261029/provenance-correction.json"
    )
    if correction.exists():
        shutil.copy2(correction, p / "provenance-correction.json")
    result = {
        "status": "PASS_DESIGN_AND_SOURCE_IDENTITY",
        "model_calls": 0,
        "counts": {"T1": 3840, "T3": 2736},
        "backgrounds_each_task": 200,
        "T3_objects": len(usedobjects),
        "T3_semantic_pairs": len(pairids),
        "metrics": metrics,
        "allocation_sha256": s.sha(p / "allocation.json"),
        "original_assets_checked": len(source_records),
        "legacy_manual_polygon_masks": manual_masks,
        "mask_scope": "Original masks are frozen inputs with recorded provenance; uniform PCA/affine/compositing applies to this construction, not a claim that every historical mask was automatically segmented.",
        "sampling_scope": "T1 source groups each cover12 of24 phases per cell, alternating parity across SF/m. T3 each group covers12 of24 onsets per non90 angle; onset/sign assignments counterbalance across28 angles. A source need not occur at every level.",
    }
    s.write(r / "design-audit.json", result)
    csv_write(r / "cell-design.csv", cells)
    s.write(p / "attribution.json", source_records)
    return result


def finish(root):
    root = Path(root)
    p, d, r = s.paths(root)
    allocation = s.read(p / "allocation.json")
    rows = []
    receipts = []
    usedpatches = {}
    des = (
        s.read(r / "design-audit.json")
        if (r / "design-audit.json").exists()
        else design(root)
    )
    assert des["status"] == "PASS_DESIGN_AND_SOURCE_IDENTITY" and des[
        "allocation_sha256"
    ] == s.sha(p / "allocation.json")
    for ident in {b["background_id"] for b in allocation["blocks"]}:
        path = d / "backgrounds" / (ident + ".npy")
        assert s.sha(path) == s.read(path.with_suffix(".json"))["prepared_sha256"]
    for ident in {b["object_id"] for b in allocation["blocks"] if b["task"] == "T3"}:
        receipt = s.read(d / "objects" / (ident + ".json"))
        assert (
            s.sha(d / "objects" / (ident + "-source.npy"))
            == receipt["source_array_sha256"]
        )
    prompt_sha = {t: s.sha(p / f"{t}-prompt.txt") for t in ["T1", "T3"]}
    for b in allocation["blocks"]:
        q = d / "qa" / (b["block_id"] + ".json")
        rec = s.read(q)
        assert rec["status"] == "PASS"
        assert rec["criteria_sha256"] == s.sha(p / "criteria.json")
        assert {k: v for k, v in b.items() if k != "cases"} == {
            k: rec[k] for k in b if k != "cases"
        }
        assert [(c["sample_id"], c["truth"]) for c in b["cases"]] == [
            (c["sample_id"], c["truth"]) for c in rec["cases"]
        ]
        for x in rec["patches"]:
            usedpatches[(x["asset_id"], x["theta"])] = x
        for c in rec["cases"]:
            assert (
                c["frames_byte_equal"] == c["timestamps_checked"] == 180
                and c["parameter_measurement"]["passed"]
            )
            video = d / "videos" / b["task"] / (c["sample_id"] + ".mp4")
            assert s.sha(video) == c["video_sha256"]
            rows.append(
                {
                    **{k: v for k, v in b.items() if k != "cases"},
                    **c,
                    "video_relative": str(video.relative_to(root)),
                    "prompt_file": f'plans/{s.NAME}/{b["task"]}-prompt.txt',
                    "prompt_sha256": prompt_sha[b["task"]],
                    "qa_sha256": s.sha(q),
                }
            )
        receipts.append({"block_id": b["block_id"], "sha256": s.sha(q)})
    assert Counter(x["task"] for x in rows) == {"T1": 3840, "T3": 2736}
    for field in ["sample_id", "video_sha256", "RGB_stream_sha256"]:
        assert len({x[field] for x in rows}) == 6576, field
    patch_records = []
    for (ident, angle), metadata in usedpatches.items():
        path = d / "patches" / f"{ident}_{angle:+g}.npy"
        assert s.sha(path) == metadata["patch_sha256"]
        a = np.load(path, mmap_mode="r")
        mask = a[:, :, 3]
        measured = cv_axis(mask)
        error = s.angle_diff(measured, 90 + angle)
        assert abs(error) <= 0.01 and abs(mask.sum() / 3800 - 1) <= 0.001
        assert (
            max(mask[0].max(), mask[-1].max(), mask[:, 0].max(), mask[:, -1].max()) == 0
        )
        assert abs(s.angle_diff(measured, metadata["measured_axis_deg"])) < 1e-8
        patch_records.append(
            {
                "asset_id": ident,
                "angle": angle,
                "opencv_axis_deg": measured,
                "axis_error_deg": error,
                "alpha_area": float(mask.sum()),
                "matrix": metadata["matrix"],
            }
        )
    areas = [x["alpha_area"] for x in patch_records]
    result = {
        "status": "COMPLETE_CPU_SOURCE_VALIDATION",
        "model_calls": 0,
        "counts": {"T1": 3840, "T3": 2736},
        "frames_verified": 6576 * 180,
        "timestamps_verified": 6576 * 180,
        "unique_video_and_RGB_streams": 6576,
        "video_bytes": sum(x["bytes"] for x in rows),
        "T1_max_abs_parameter_errors": {
            k: max(
                abs(x["parameter_measurement"][k]) for x in rows if x["task"] == "T1"
            )
            for k in ["sf_error", "m_error", "phase_error_rad"]
        },
        "T3_independent_OpenCV_patches": len(patch_records),
        "T3_max_axis_error_deg": max(abs(x["axis_error_deg"]) for x in patch_records),
        "T3_area_range": [min(areas), max(areas)],
        "design_audit_sha256": s.sha(r / "design-audit.json"),
        "criteria_sha256": s.sha(p / "criteria.json"),
        "allocation_sha256": s.sha(p / "allocation.json"),
        "prompt_sha256": prompt_sha,
        "receipts": receipts,
        "scope": "Every encoded RGB frame and timestamp checked during construction. Independent OpenCV moments check all used T3 masks. Does not measure human thresholds or processor behavior.",
    }
    s.write(r / "source-audit.json", result)
    s.write(r / "geometry-audit.json", {"patches": patch_records})
    s.write(p / "source-index.json", {"counts": result["counts"], "rows": rows})
    flat = [
        {
            k: x.get(k, "")
            for k in [
                "sample_id",
                "task",
                "group",
                "sf",
                "m",
                "phase_index",
                "angle",
                "tilt_sign",
                "onset",
                "background_id",
                "object_id",
                "truth",
                "video_relative",
                "video_sha256",
                "RGB_stream_sha256",
            ]
        }
        for x in rows
    ]
    csv_write(r / "source-index.csv", flat)
    versions = {
        name: importlib.metadata.version(name)
        for name in ["numpy", "scipy", "Pillow", "opencv-python", "av"]
    }
    s.write(
        p / "environment.json",
        {
            "python": sys.version,
            "packages": versions,
            "av_libraries": {
                k: list(v) for k, v in __import__("av").library_versions.items()
            },
            "model_calls": 0,
        },
    )
    result.update(angle_separation(root))
    preview(root)
    return {k: v for k, v in result.items() if k != "receipts"}


def preview_pair(job):
    import av

    root, block = job
    root = Path(root)
    p, d, r = s.paths(root)
    dest = r / "previews"
    q = d / "qa" / (block["block_id"] + ".json")
    if not q.exists():
        return None
    rec = s.read(q)
    receipt = dest / (block["block_id"] + "-display.json")
    if receipt.exists():
        previous = s.read(receipt)
        assert previous["qa_sha256"] == s.sha(q)
        return previous["sample"]
    seen = {}
    items = []
    for case in rec["cases"]:
        source = d / "videos" / block["task"] / (case["sample_id"] + ".mp4")
        target = dest / (case["sample_id"] + "-browser.mp4")
        temp = target.with_suffix(".partial.mp4")
        frame_images = []
        source_rgb = hashlib.sha256()
        out = av.open(str(temp), "w")
        stream = out.add_stream("libx264", rate=60)
        stream.width = 768
        stream.height = 512
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "preset": "ultrafast",
            "crf": "18",
            "x264-params": "ref=12:scenecut=0",
        }
        stream.codec_context.thread_count = 1
        stream.codec_context.colorspace = 1
        stream.codec_context.color_primaries = 1
        stream.codec_context.color_trc = 13
        stream.codec_context.color_range = 1
        with av.open(str(source)) as container:
            container.streams.video[0].codec_context.thread_count = 1
            for n, frame in enumerate(container.decode(video=0)):
                rgb = frame.to_ndarray(format="rgb24")
                raw = rgb.tobytes()
                source_rgb.update(raw)
                key = hashlib.sha256(raw).hexdigest()
                if key not in seen:
                    png = dest / (block["block_id"] + "_" + key[:20] + ".png")
                    Image.fromarray(rgb).save(png, compress_level=1)
                    assert np.array_equal(np.asarray(Image.open(png)), rgb)
                    seen[key] = {
                        "url": "previews/" + png.name,
                        "RGB_sha256": key,
                        "PNG_sha256": s.sha(png),
                    }
                frame_images.append(seen[key]["url"])
                converted = frame.reformat(
                    format="yuv420p",
                    dst_colorspace="ITU709",
                    src_color_range="JPEG",
                    dst_color_range="MPEG",
                    dst_color_trc=13,
                    dst_color_primaries=1,
                    threads=1,
                )
                converted.pts = n
                converted.time_base = Fraction(1, 60)
                for packet in stream.encode(converted):
                    out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
        out.close()
        assert (
            len(frame_images) == 180
            and source_rgb.hexdigest() == case["RGB_stream_sha256"]
        )
        temp.replace(target)
        items.append(
            {
                "side": case["truth"],
                "url": "previews/" + target.name,
                "preview_sha256": s.sha(target),
                "source_sha256": case["video_sha256"],
                "frame_images": frame_images,
            }
        )
    sample = {**{k: v for k, v in block.items() if k != "cases"}, "videos": items}
    s.write(
        receipt,
        {
            "qa_sha256": s.sha(q),
            "sample": sample,
            "exact_source_PNGs": list(seen.values()),
            "display_policy": "YUV420 playback derivative only. Every paused/selected frame displays its exact decoded source RGB PNG. Scientific source videos are unchanged.",
        },
    )
    return sample


def preview(root):
    root = Path(root)
    p, d, r = s.paths(root)
    blocks = s.read(p / "allocation.json")["blocks"]
    dest = r / "previews"
    dest.mkdir(exist_ok=True)
    chosen = {}
    for b in blocks:
        chosen.setdefault((b["task"], b["cell"], b["group"]), b)
    with ProcessPoolExecutor(3) as pool:
        manifest = [
            m
            for m in pool.map(preview_pair, [(str(root), b) for b in chosen.values()])
            if m is not None
        ]
    s.write(
        r / "preview-manifest.json",
        {
            "samples": manifest,
            "selection": "first allocated LR pair per task/cell/source group, before model outcomes",
            "complete": (r / "source-audit.json").exists(),
        },
    )
    (r / "index.html").write_text(HTML)
    return {"preview_pairs": len(manifest)}


HTML = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Natural T1 · T3</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f4ef;color:#182b24;font:16px system-ui}main{max-width:1400px;margin:auto;padding:32px}h1{font-size:36px;margin:8px 0}header{display:flex;justify-content:space-between;align-items:center;gap:16px}.stats{display:flex;gap:20px;margin:24px 0}.stats div{background:white;padding:14px 20px;border-radius:12px}.stats b{display:block;font-size:24px}.controls{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}select,button,input{font:inherit}select,button{padding:10px 16px;border:1px solid #c5cfc8;border-radius:8px;background:white}button{cursor:pointer}button.primary{background:#245b45;color:white}.pair{display:grid;grid-template-columns:1fr 1fr;gap:18px}.panel{background:white;border-radius:14px;padding:12px}.panel b{display:block;margin:4px 0 10px}video{display:block;width:100%;aspect-ratio:3/2;background:#ddd}.stage{position:relative}.stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}.time{display:flex;gap:12px;align-items:center;margin-top:20px}.time input{flex:1}.tag{display:inline-block;background:#e0e9e2;padding:7px 12px;border-radius:30px;margin:0 6px 8px 0}a{color:#245b45}details{margin-top:24px;line-height:1.6} @media(max-width:800px){main{padding:16px}.pair{grid-template-columns:1fr}.stats{flex-wrap:wrap}h1{font-size:28px}}
</style><main><header><div><span class="tag">CPU construction · no model inference</span><h1>Natural backgrounds. Controlled signals.</h1></div><a href="source-audit.json">Audit ↗</a></header><div class="stats"><div><b>3,840</b>T1 videos</div><div><b>2,736</b>T3 videos</div><div><b>200</b>backgrounds / task</div><div><b id="status">Building</b>source validation</div></div><div class="controls"><select id="task"><option>T1</option><option>T3</option></select><select id="source"></select><select id="axis"></select><select id="depth"></select><button id="previous">← Previous</button><button id="next">Next →</button></div><div id="tags"></div><span class="tag" id="renderMode">Original RGB frame</span><div class="pair"><div class="panel"><b id="lefttitle">LEFT</b><div class="stage"><video id="left" muted playsinline preload="auto"></video><img id="leftFrame" alt="Exact source RGB frame"></div></div><div class="panel"><b id="righttitle">RIGHT</b><div class="stage"><video id="right" muted playsinline preload="auto"></video><img id="rightFrame" alt="Exact source RGB frame"></div></div></div><div class="time"><button class="primary" id="play">Play pair</button><button id="key">Key frame</button><input id="frame" type="range" min="0" max="179" value="0"><span id="counter">0 / 179</span></div><details><summary>Design & source records</summary><p>T1: matched LEFT/RIGHT motion, 24 phases per SF × m cell. Signal opacity is 0.5; m describes the added grating, not the final scene contrast.</p><p>T3: the same silhouette PCA, fixed area and one affine transform. Four background/object source groups. Each angle has balanced LEFT/RIGHT and tilt signs; 90° has one unique axial sign.</p><p><a href="design-audit.json">Balance</a> · <a href="geometry-audit.json">Geometry</a> · <a href="source-index.csv">Every video</a></p></details></main><script>
let data=[],current=null,currentFrame=0;const $=x=>document.getElementById(x),vs=[$('left'),$('right')],imgs=[$('leftFrame'),$('rightFrame')];
function options(id,values,label=x=>x){const old=$(id).value;$(id).innerHTML='';for(const x of values){const o=document.createElement('option');o.value=x;o.textContent=label(x);$(id).append(o)}if(values.map(String).includes(old))$(id).value=old}
function filter(){const task=$('task').value,rows=data.filter(x=>x.task===task);options('source',[...new Set(rows.map(x=>x.group))],x=>x.replaceAll('photo','Photo').replaceAll('gpt','Generated').replace('_',' BG × ') +(task==='T3'?' OBJ':''));const selected=rows.filter(x=>x.group===$('source').value);options('axis',[...new Set(selected.map(x=>task==='T1'?x.sf:x.angle))],x=>task==='T1'?'SF '+x:x+'°');$('depth').hidden=task==='T3';if(task==='T1')options('depth',[...new Set(selected.filter(x=>x.sf===$('axis').value*1).map(x=>x.m))],x=>'m '+x);load()}
function exact(n){currentFrame=n;imgs.forEach((img,i)=>{img.src=current.videos[i].frame_images[n];img.hidden=false});$('frame').value=n;$('counter').textContent=n+' / 179';$('renderMode').textContent='Original RGB frame';$('play').textContent='Play pair'}
function load(){const task=$('task').value;current=data.find(x=>x.task===task&&x.group===$('source').value&&(task==='T1'?x.sf===$('axis').value*1&&x.m===$('depth').value*1:x.angle===$('axis').value*1));if(!current)return;vs.forEach((v,i)=>{v.pause();v.src=current.videos[i].url;v.onloadedmetadata=()=>{v.currentTime=(currentFrame+.001)/60}});exact(0);$('lefttitle').textContent=task==='T1'?'LEFT motion':'LEFT more vertical';$('righttitle').textContent=task==='T1'?'RIGHT motion':'RIGHT more vertical';const tags=task==='T1'?['768 × 512','180 frames · 60 fps','Phase '+(7.5+15*current.phase_index)+'°','Opacity 0.5']:['768 × 512','PCA axis · 3,800 px²','Onset '+(current.onset/60).toFixed(3)+' s','Tilt '+(current.tilt_sign>0?'+':'−')+current.angle+'°'];$('tags').innerHTML=tags.map(x=>'<span class="tag">'+x+'</span>').join('')}
function seek(n){vs.forEach(v=>{v.pause();if(v.readyState>=1)v.currentTime=(n+.001)/60});exact(n)}
$('task').onchange=filter;$('source').onchange=filter;$('axis').onchange=filter;$('depth').onchange=load;$('frame').oninput=()=>seek(+$('frame').value);$('key').onclick=()=>seek(current.task==='T3'?current.onset:3);
$('play').onclick=async()=>{if(vs[0].paused){if(currentFrame>=179)seek(0);await Promise.all(vs.map(v=>v.play()));imgs.forEach(img=>img.hidden=true);$('renderMode').textContent='Playback preview';$('play').textContent='Pause'}else{const n=Math.min(179,Math.floor(vs[0].currentTime*60));seek(n)}};
for(const [id,step] of [['previous',-1],['next',1]])$(id).onclick=()=>{const sel=current.task==='T1'?$('depth'):$('axis');sel.selectedIndex=(sel.selectedIndex+step+sel.options.length)%sel.options.length;sel.dispatchEvent(new Event('change'))};
vs[0].ontimeupdate=()=>{if(vs[0].paused)return;currentFrame=Math.min(179,Math.floor(vs[0].currentTime*60));$('frame').value=currentFrame;$('counter').textContent=currentFrame+' / 179';if(Math.abs(vs[1].currentTime-vs[0].currentTime)>.07)vs[1].currentTime=vs[0].currentTime};vs[0].onended=()=>seek(179);
fetch('preview-manifest.json').then(r=>r.json()).then(x=>{data=x.samples;$('status').textContent=x.complete?'PASS':'Building';filter()});
</script></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command", choices=["design", "finish", "preview", "angle_separation"]
    )
    ap.add_argument("--root", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(globals()[args.command](args.root), ensure_ascii=False))


if __name__ == "__main__":
    main()
