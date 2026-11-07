"""Paired single/dual-event T5 stimuli with the established T5 display geometry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import shutil
import time

import av
import numpy as np
from PIL import Image


PROTOCOL = "t5-paired-v1"
WIDTH, HEIGHT, FPS, FRAMES = 768, 512, 60, 180
CENTERS = ((160, 256), (608, 256), (384, 128), (384, 384))
POSITIONS = ("left", "right", "top", "bottom")
GROUPS = ("photo_photo", "photo_gpt", "gpt_photo", "gpt_gpt")
SEED = 20261107
PROMPT_SHA = "bd57b15607a312dc654fc6104e2db437d403275b64266d33710ade39142f8b3d"
PROMPT = (
    "The first input is a single reference image showing the target object in the orientation that counts as a match. "
    "In the video, objects appear in one or two separate events. Determine whether any event shows that object in exactly the reference orientation, "
    "comparing orientation in the image plane while ignoring position, size, and background. A rotated version does not count as a match. "
    "Judge each object during its visible presentation. If a line pattern replaces an object, this mask marks the end of that presentation; "
    "it is not another target orientation. A match still counts if it appeared earlier and is no longer visible at the end. "
    "Answer YES if any event matches, or NO if none does. Give your final answer as exactly one uppercase word: YES or NO."
)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def value_sha(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def bind(path):
    path = Path(path)
    return {"path": str(path), "sha256": file_sha(path)}


def rgb8(linear):
    linear = np.clip(linear, 0, 1)
    encoded = np.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * linear ** (1 / 2.4) - 0.055,
    )
    return np.rint(255 * encoded).astype(np.uint8)


def linearize(encoded):
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def balanced_allocation():
    """Choose one legal added event for every onset/location block, before answers."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    import scipy

    candidates = []
    for k in range(24):
        onset = 6 + 6 * k
        for position in range(4):
            for order in ("before", "after"):
                if order == "before" and onset < 60:
                    continue
                if order == "after" and onset > 90:
                    continue
                for soa in (30, 45, 60):
                    added_onset = onset + (-soa if order == "before" else soa)
                    for added_position in range(4):
                        if added_position != position:
                            candidates.append({
                                "k": k,
                                "onset": onset,
                                "position": position,
                                "group_index": (k + position) % 4,
                                "order": order,
                                "soa": soa,
                                "added_onset": added_onset,
                                "added_position": added_position,
                            })

    memberships = defaultdict(list)
    limits = {}

    def include(index, name, key, lower, upper=None):
        constraint = (name, *key)
        memberships[constraint].append(index)
        limits[constraint] = (lower, lower if upper is None else upper)

    duplicate_sequences = defaultdict(list)
    for index, row in enumerate(candidates):
        k, a, b = row["k"], row["position"], row["added_position"]
        group, order, soa = row["group_index"], row["order"], row["soa"]
        include(index, "anchor", (k, a), 1)
        include(index, "group_order_soa", (group, order, soa), 4)
        include(index, "group_positions", (group, a, b), 2)
        include(index, "onset_soa", (k, soa), 1, 2)
        if 9 <= k <= 14:
            include(index, "central_onset_order", (k, order), 2)
        sequence = tuple(sorted(((row["onset"], a), (row["added_onset"], b))))
        duplicate_sequences[sequence].append(index)
    for sequence, indices in duplicate_sequences.items():
        if len(indices) > 1:
            constraint = ("unique_absent_sequence", sequence)
            memberships[constraint] = indices
            limits[constraint] = (0, 1)

    rr, cc, vv, lower, upper = [], [], [], [], []
    for number, (key, indices) in enumerate(memberships.items()):
        rr.extend([number] * len(indices))
        cc.extend(indices)
        vv.extend([1.0] * len(indices))
        lo, hi = limits[key]
        lower.append(lo)
        upper.append(hi)
    matrix = coo_matrix(
        (vv, (rr, cc)), shape=(len(memberships), len(candidates))
    ).tocsc()
    rng = np.random.default_rng(SEED)
    result = milp(
        c=rng.random(len(candidates)),
        integrality=np.ones(len(candidates)),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 60, "mip_rel_gap": 0.05},
    )
    if result.x is None:
        raise RuntimeError(f"No feasible balanced allocation: {result.message}")
    chosen = np.rint(result.x)
    actual = matrix @ chosen
    assert np.all(actual >= np.array(lower)) and np.all(actual <= np.array(upper))
    rows = [row for row, flag in zip(candidates, chosen) if flag]
    rows.sort(key=lambda row: (row["k"], row["position"]))
    assert len(rows) == 96
    return rows, {
        "seed": SEED,
        "solver": "scipy.optimize.milp",
        "scipy_version": scipy.__version__,
        "solver_message": result.message,
        "constraints": len(memberships),
        "allocation_sha256": value_sha(rows),
        "uses_model_answers": False,
    }


def inherited_scenes(root):
    old_plan = root / "plans/t45-final-20261030/current1200-manifest.json"
    old_rows = json.loads(old_plan.read_text())["items"]
    source = root / "datasets/t45-final-20261030/natural-fit56-v2"
    catalog_path = root / "plans/t45-final-20261030/natural-fit56-v2/asset-catalog.json"
    catalog = json.loads(catalog_path.read_text())
    objects = {row["id"]: row for row in catalog["objects"]}
    backgrounds = {row["id"]: row for row in catalog["backgrounds"]}
    scenes = {}
    for row in old_rows:
        if row["task"] != "T5" or row["domain"] != "natural":
            continue
        key = (row["object_id"], row["background_id"])
        if key in scenes:
            continue
        object_id, background_id = key
        prepared = source / "prepared"
        scene = {
            "object_id": object_id,
            "background_id": background_id,
            "group": row["group"],
            "parent_sample_id": row["sample_id"],
            "object_prepared": bind(prepared / (object_id + "-T5.npy")),
            "object_receipt": bind(prepared / (object_id + ".json")),
            "object_source": bind(root / "assets/library" / objects[object_id]["path"]),
            "background_prepared": bind(prepared / (background_id + ".npy")),
            "background_receipt": bind(root / "datasets/natural-t13-20261029/backgrounds" / (background_id + ".json")),
            "background_source": bind(root / "assets/library" / backgrounds[background_id]["path"]),
            "reference_source": bind(source / "references" / (object_id + ".png")),
        }
        assert scene["object_source"]["sha256"] == row["source_asset_hashes"]["object_id"]
        assert scene["background_source"]["sha256"] == row["source_asset_hashes"]["background_id"]
        scenes[key] = scene
    grouped = {
        group: sorted(
            (scene for scene in scenes.values() if scene["group"] == group),
            key=lambda scene: (scene["object_id"], scene["background_id"]),
        )
        for group in GROUPS
    }
    assert all(len(values) == 6 for values in grouped.values())
    assert len({scene["object_id"] for scene in scenes.values()}) == 24
    return grouped, bind(old_plan), bind(catalog_path)


def initialize(root):
    root = Path(root)
    plan_path = root / "plans" / PROTOCOL / "manifest.json"
    if plan_path.exists():
        return json.loads(plan_path.read_text())
    assert hashlib.sha256(PROMPT.encode()).hexdigest() == PROMPT_SHA
    allocation, solver = balanced_allocation()
    scenes, parent, catalog = inherited_scenes(root)
    rows = []
    classic_ids = {}
    for domain in ("classic", "natural"):
        media_root = root / "datasets" / PROTOCOL / domain
        for folder in ("videos", "references", "inspection", "qa"):
            (media_root / folder).mkdir(parents=True, exist_ok=True)
        for allocation_row in allocation:
            k, position = allocation_row["k"], allocation_row["position"]
            scene = None
            if domain == "natural":
                scene = scenes[GROUPS[allocation_row["group_index"]]][k // 4]
                reference_source = scene["reference_source"]
                reference_name = scene["object_id"] + ".png"
            else:
                reference_source = bind(root / "datasets/t45-final-20261030/references/T5-classic.png")
                reference_name = "T5-classic.png"
            reference = media_root / "references" / reference_name
            if not reference.exists():
                shutil.copyfile(reference_source["path"], reference)
            assert file_sha(reference) == reference_source["sha256"]
            for truth in ("YES", "NO"):
                angle = 0 if truth == "YES" else 90
                paired_block = f"{domain}-t5p1-k{k:02d}-p{position}-{truth.lower()}"
                probe = {
                    "role": "probe",
                    "onset_frame": allocation_row["onset"],
                    "position_index": position,
                    "angle_deg": angle,
                }
                for cohort in ("single", "dual"):
                    events = [dict(probe)]
                    if cohort == "dual":
                        events.append({
                            "role": "distractor",
                            "onset_frame": allocation_row["added_onset"],
                            "position_index": allocation_row["added_position"],
                            "angle_deg": 90,
                        })
                    parameters = {
                        "protocol_id": PROTOCOL,
                        "domain": domain,
                        "cohort": cohort,
                        "k": k,
                        "onset": allocation_row["onset"],
                        "position_index": position,
                        "events": events,
                        "exposure_frames": 30,
                        "mask": "none",
                        "guide_size_px": 160,
                        "target_size_px": 128,
                        "target_stroke_px": 28 if domain == "classic" else None,
                        "reference_wh": [160, 160],
                        "centers_xy": [list(center) for center in CENTERS],
                    }
                    if scene:
                        parameters.update({
                            "object_id": scene["object_id"],
                            "background_id": scene["background_id"],
                            "group": scene["group"],
                            "object_prepared_sha256": scene["object_prepared"]["sha256"],
                            "background_prepared_sha256": scene["background_prepared"]["sha256"],
                        })
                    sample_id = ("c5p1_" if domain == "classic" else "n5p1_") + value_sha(parameters)[:24]
                    stem = f"T5_{cohort}_{truth.lower()}_onset{probe['onset_frame']:03d}_{POSITIONS[position]}"
                    if cohort == "dual":
                        stem += (
                            f"_{allocation_row['order']}_soa{allocation_row['soa']:03d}"
                            f"_{POSITIONS[allocation_row['added_position']]}"
                        )
                    row = {
                        "sample_id": sample_id,
                        "task": "T5",
                        "domain": domain,
                        "cohort": cohort,
                        "condition": cohort,
                        "paired_block": paired_block,
                        "stimulus_block": f"{domain}-t5p1-k{k:02d}-p{position}",
                        "onset": probe["onset_frame"],
                        "position_index": position,
                        "truth": truth,
                        "choices": ["YES", "NO"],
                        "accuracy_eligible": True,
                        "prompt": PROMPT,
                        "prompt_sha256": PROMPT_SHA,
                        "events": events,
                        "source_parameters": parameters,
                        "reference_path": str(reference),
                        "reference_sha256": reference_source["sha256"],
                        "reference_pixel_sha256": hashlib.sha256(
                            np.asarray(Image.open(reference).convert("RGB")).tobytes()
                        ).hexdigest(),
                        "video_path": str(media_root / "videos" / (stem + ".mp4")),
                        "friendly_filename": stem + ".mp4",
                        "scene": scene,
                        "group": scene["group"] if scene else "procedural",
                        "SOA_frames": allocation_row["soa"] if cohort == "dual" else None,
                        "ISI_frames": allocation_row["soa"] - 30 if cohort == "dual" else None,
                        "added_event_order": allocation_row["order"] if cohort == "dual" else None,
                    }
                    if domain == "classic":
                        classic_ids[(k, position, truth, cohort)] = sample_id
                    else:
                        row["parent_classic_id"] = classic_ids[(k, position, truth, cohort)]
                    rows.append(row)
    manifest = {
        "schema": "t5-paired-v1-manifest",
        "protocol_id": PROTOCOL,
        "display": {"width": WIDTH, "height": HEIGHT, "fps": FPS, "frames": FRAMES},
        "source_version": PROTOCOL,
        "parent_manifest": parent,
        "parent_asset_catalog": catalog,
        "generator": bind(__file__),
        "allocation": allocation,
        "allocation_method": solver,
        "items": rows,
    }
    assert len(rows) == len({row["sample_id"] for row in rows}) == 768
    write_json(plan_path, manifest)
    return manifest


@lru_cache(maxsize=64)
def load_array(path):
    return np.load(path)


def frame_images(row):
    """Render the fixed scene and the exact visible state of each event."""
    scene = row["scene"]
    background = np.full((HEIGHT, WIDTH, 3), 0.10) if scene is None else load_array(scene["background_prepared"]["path"])
    base = rgb8(background)
    for x, y in CENTERS:
        base[y - 80:y + 80, x - 80:x - 78] = 115
        base[y - 80:y + 80, x + 78:x + 80] = 115
        base[y - 80:y - 78, x - 80:x + 80] = 115
        base[y + 78:y + 80, x - 80:x + 80] = 115
    base[255:257, 379:390] = 170
    base[251:262, 383:385] = 170
    if scene is None:
        canonical = np.full((128, 128, 3), int(rgb8(np.array(0.10))), np.uint8)
        canonical[:, :28] = int(rgb8(np.array(0.82)))
        canonical[-28:, :] = int(rgb8(np.array(0.82)))
    else:
        canonical = load_array(scene["object_prepared"]["path"])
    images = [base]
    schedule = np.zeros(FRAMES, dtype=np.int16)
    for index, event in enumerate(row["events"], start=1):
        start = event["onset_frame"]
        assert 0 <= start and start + 30 <= FRAMES
        assert not schedule[start:start + 30].any()
        sprite = np.rot90(canonical, -event["angle_deg"] // 90)
        frame = base.copy()
        x, y = CENTERS[event["position_index"]]
        roi = frame[y - 64:y + 64, x - 64:x + 64]
        if scene is None:
            roi[:] = sprite
        else:
            back = linearize(roi.astype(np.float64) / 255)
            roi[:] = rgb8(sprite[:, :, :3] + (1 - sprite[:, :, 3, None]) * back)
        images.append(frame)
        schedule[start:start + 30] = index
    return images, schedule


def render_one(payload):
    root, row = payload
    started = time.monotonic()
    video = Path(row["video_path"])
    qa_path = video.parent.parent / "qa" / (row["sample_id"] + ".json")
    if qa_path.exists():
        qa = json.loads(qa_path.read_text())
        assert qa["row_sha256"] == value_sha(row)
        assert file_sha(video) == qa["video_sha256"]
        return qa
    assert not video.exists(), f"Unverified existing video: {video}"
    images, schedule = frame_images(row)
    hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in images]
    temporary = video.with_suffix(".partial.mp4")
    assert not temporary.exists(), f"Preserved partial video: {temporary}"
    stream_hash = hashlib.sha256()
    with av.open(str(temporary), "w") as container:
        stream = container.add_stream("libx264rgb", rate=FPS)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "rgb24"
        stream.codec_context.thread_count = 1
        stream.options = {"crf": "0", "preset": "ultrafast", "threads": "1"}
        for index, state in enumerate(schedule):
            rgb = images[state]
            stream_hash.update(rgb.tobytes())
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    pts = []
    with av.open(str(temporary)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        assert stream.codec_context.pix_fmt == "gbrp"
        assert stream.average_rate == FPS
        for index, frame in enumerate(container.decode(video=0)):
            assert index < FRAMES
            actual = frame.to_ndarray(format="rgb24")
            assert hashlib.sha256(actual.tobytes()).hexdigest() == hashes[schedule[index]]
            pts.append(float(frame.pts * frame.time_base))
            assert abs(pts[-1] - index / FPS) <= 1e-6
    assert len(pts) == FRAMES
    temporary.replace(video)
    inspection = video.parent.parent / "inspection" / row["sample_id"]
    inspection.mkdir(exist_ok=True)
    inspection_paths = [row["reference_path"]]
    for event in sorted(row["events"], key=lambda event: event["onset_frame"]):
        index = event["onset_frame"] + 15
        path = inspection / f"frame{index:03d}.png"
        Image.fromarray(images[schedule[index]]).save(path)
        inspection_paths.append(str(path))
    qa = {
        "status": "PASS_ENCODER_ALL180_RGB8_PTS",
        "sample_id": row["sample_id"],
        "row_sha256": value_sha(row),
        "video_sha256": file_sha(video),
        "stream_sha256": stream_hash.hexdigest(),
        "frame_sha256": [hashes[state] for state in schedule],
        "frames": FRAMES,
        "max_PTS_error_seconds": max(abs(value - index / FPS) for index, value in enumerate(pts)),
        "inspection_paths": inspection_paths,
        "implementation_sha256": file_sha(__file__),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(qa_path, qa)
    return qa


def make_catalog(root, manifest, domains):
    items = {}
    for row in manifest["items"]:
        path = Path(row["video_path"])
        qa_path = path.parent.parent / "qa" / (row["sample_id"] + ".json")
        if not qa_path.exists():
            continue
        qa = json.loads(qa_path.read_text())
        items[row["sample_id"]] = {
            "sample_id": row["sample_id"],
            "task": "T5",
            "row": row,
            "content": [
                {"type": "image", "path": row["reference_path"], "sha256": row["reference_sha256"], "pixel_sha256": row["reference_pixel_sha256"]},
                {"type": "video", "path": row["video_path"], "sha256": qa["video_sha256"]},
                {"type": "text", "text": row["prompt"]},
            ],
            "video_contract": {"frames": FRAMES, "fps": FPS, "width": WIDTH, "height": HEIGHT, "no_resize": True},
            "ready": {
                "stream_sha256": qa["stream_sha256"],
                "video_sha256": qa["video_sha256"],
                "frames": FRAMES,
                "fps": FPS,
                "width": WIDTH,
                "height": HEIGHT,
            },
            "source_qa_path": str(qa_path),
        }
    catalog = {
        "schema": "t5-paired-v1-ready",
        "manifest": bind(Path(root) / "plans" / PROTOCOL / "manifest.json"),
        "items": items,
        "independent_validation_required": True,
    }
    write_json(Path(root) / "plans" / PROTOCOL / "ready_catalog.json", catalog)
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initialize", action="store_true")
    parser.add_argument("--domain", choices=("classic", "natural", "all"), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = initialize(args.root)
    if args.initialize:
        print(json.dumps({"items": len(manifest["items"]), "allocation": manifest["allocation_method"]}), flush=True)
        return
    domains = ("classic", "natural") if args.domain == "all" else (args.domain,)
    rows = [row for row in manifest["items"] if row["domain"] in domains]
    if args.limit:
        rows = rows[:args.limit]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(render_one, (str(args.root), row)) for row in rows]
        for completed, future in enumerate(as_completed(futures), start=1):
            qa = future.result()
            if completed % 16 == 0 or completed == len(rows):
                print(json.dumps({"completed": completed, "expected": len(rows), "last": qa["sample_id"], "elapsed_seconds": time.monotonic() - started}), flush=True)
    catalog = make_catalog(args.root, manifest, domains)
    print(json.dumps({"ready_items": len(catalog["items"]), "elapsed_seconds": time.monotonic() - started}), flush=True)


if __name__ == "__main__":
    main()
