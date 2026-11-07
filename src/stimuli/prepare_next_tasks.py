"""CPU-only review samples for the next protocols; never dispatch inference.

Keep a small RDM pilot, two object-angle pilots, and a deterministic candidate
content allocation together. These are review inputs, not an approved full run.
"""

from __future__ import annotations
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.optimize import minimize_scalar
from scipy.integrate import quad
from build_precise_protocol_examples import circle_pixel_area

SEED = "cvpr2027-next-task-preparation-v1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(value):
    return hashlib.sha256((SEED + ":" + str(value)).encode()).hexdigest()


def linear(rgb):
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)


def encode(rgb):
    rgb = np.clip(rgb, 0, 1)
    srgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb ** (1 / 2.4) - 0.055)
    return np.rint(srgb * 65535).astype(np.uint16)


def axis(alpha):
    y, x = np.indices(alpha.shape, dtype=np.float64)
    mass = alpha.sum()
    cx = float((alpha * x).sum() / mass)
    cy = float((alpha * y).sum() / mass)
    x -= cx
    y -= cy
    cov = (
        np.array(
            [
                [(alpha * x * x).sum(), (alpha * x * y).sum()],
                [(alpha * x * y).sum(), (alpha * y * y).sum()],
            ]
        )
        / mass
    )
    values, vectors = np.linalg.eigh(cov)
    v = vectors[:, -1]
    return (
        math.degrees(math.atan2(v[1], v[0])),
        (cx, cy),
        float(values[-1] / max(values[0], 1e-12)),
    )


def angle_delta(a, b):
    return (a - b + 90) % 180 - 90


def sample_affine(source, matrix, size=256, samples=8):
    """One source-to-output affine, integrated on a uniform subpixel grid."""
    inv = np.linalg.inv(matrix)
    y, x = np.indices((size, size), dtype=np.float64)
    out = np.zeros((size, size, source.shape[2]), np.float64)
    for dy in (np.arange(samples) + 0.5) / samples - 0.5:
        for dx in (np.arange(samples) + 0.5) / samples - 0.5:
            sx = inv[0, 0] * (x + dx) + inv[0, 1] * (y + dy) + inv[0, 2]
            sy = inv[1, 0] * (x + dx) + inv[1, 1] * (y + dy) + inv[1, 2]
            for c in range(source.shape[2]):
                out[:, :, c] += map_coordinates(
                    source[:, :, c],
                    [sy, sx],
                    order=1,
                    mode="constant",
                    cval=0,
                    prefilter=False,
                )
    return out / samples**2


def matrix(scale, theta, center):
    t = math.radians(theta)
    c, s = math.cos(t), math.sin(t)
    a = scale * np.array([[c, -s], [s, c]], np.float64)
    out = np.eye(3)
    out[:2, :2] = a
    out[:2, 2] = np.array([127.5, 127.5]) - a @ center
    return out


def rgb_register(reference, target):
    """Estimate angle from observed RGB pixels; target angle is not an input."""
    # Fixed isotropic prefilter reduces raster-template interpolation bias.
    # It changes only the independent estimator, never a saved stimulus.
    reference = gaussian_filter(reference, sigma=(1.0, 1.0, 0), mode="nearest")
    target = gaussian_filter(target, sigma=(1.0, 1.0, 0), mode="nearest")
    size = reference.shape[0]
    background = np.median(
        np.concatenate(
            [reference[0], reference[-1], reference[:, 0], reference[:, -1]], axis=0
        ),
        axis=0,
    )
    y, x = np.indices((size, size), dtype=np.float64)
    x -= (size - 1) / 2
    y -= (size - 1) / 2

    def loss(deg):
        t = math.radians(deg)
        c, s = math.cos(t), math.sin(t)
        sx = c * x + s * y + (size - 1) / 2
        sy = -s * x + c * y + (size - 1) / 2
        warped = np.stack(
            [
                map_coordinates(
                    reference[:, :, k] - background[k],
                    [sy, sx],
                    order=3,
                    mode="constant",
                    cval=0,
                    prefilter=True,
                )
                + background[k]
                for k in range(3)
            ],
            axis=-1,
        )
        return float(np.mean((warped - target) ** 2))

    fit = minimize_scalar(
        loss, bounds=(-2, 2), method="bounded", options={"xatol": 1e-5}
    )
    return {
        "angle_deg": float(fit.x),
        "mean_square_error": float(fit.fun),
        "success": bool(fit.success),
        "search_bounds_deg": [-2, 2],
        "background_linear_rgb": background.tolist(),
        "estimator_prefilter_sigma_px": 1.0,
    }


def object_pilot(root, out, catalog):
    selected = ["photo_obj_021", "photo_obj_027"]
    arrays, records, tiles = {}, [], []
    for sid in selected:
        asset = next(a for a in catalog["objects"] if a["id"] == sid)
        path = (root / "assets/library" / asset["path"]).resolve()
        assert sha(path) == asset["sha256"]
        rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255
        alpha = rgba[:, :, 3]
        base_axis, center, ratio = axis(alpha)
        yy, xx = np.nonzero(alpha > 0.01)
        v = np.array(
            [math.cos(math.radians(base_axis)), math.sin(math.radians(base_axis))]
        )
        projection = np.stack([xx, yy], axis=1) @ v
        scale = 190 / (projection.max() - projection.min())
        source = np.concatenate(
            [linear(rgba[:, :, :3]) * alpha[:, :, None], alpha[:, :, None]], axis=2
        )
        item = {
            "asset_id": sid,
            "label": asset["label"],
            "source_path": str(path.relative_to(root)),
            "source_sha256": sha(path),
            "source_bits": 8,
            "source_axis_deg": base_axis,
            "source_centroid_px": center,
            "axis_variance_ratio": ratio,
            "transform": "FP64 scale+axis canonicalization+rotation+translation composed once; original premultiplied linear RGB/alpha sampled directly",
            "supersampling": 8,
            "angles": [],
        }
        observed = {}
        for theta in (0.0, -0.5, 0.5, -1.0, 1.0):
            transform = matrix(scale, 90 - base_axis + theta, center)
            sampled = sample_affine(source, transform)
            a16 = np.rint(np.clip(sampled[:, :, 3], 0, 1) * 65535).astype(np.uint16)
            rgb16 = encode(sampled[:, :, :3] + (0.1 * (1 - sampled[:, :, 3, None])))
            measured = axis(a16.astype(np.float64) / 65535)[0]
            tag = str(theta).replace("-", "neg").replace(".", "p")
            arrays[sid + "_" + tag + "_rgb16"] = rgb16
            arrays[sid + "_" + tag + "_alpha16"] = a16
            observed[theta] = (rgb16, a16, measured)
            item["angles"].append(
                {
                    "nominal_delta_deg": theta,
                    "source_to_output_matrix": transform.tolist(),
                    "measured_axis_deg": measured,
                    "alpha16_sha256": hashlib.sha256(a16.tobytes()).hexdigest(),
                    "rgb16_sha256": hashlib.sha256(rgb16.tobytes()).hexdigest(),
                }
            )
            tiles.append((f"{sid}: {theta:+.1f} deg", (rgb16 >> 8).astype(np.uint8)))
        ref = observed[0.0][0].astype(np.float64) / 65535
        ref = linear(ref)
        for row in item["angles"]:
            theta = row["nominal_delta_deg"]
            measured = angle_delta(row["measured_axis_deg"], observed[0.0][2])
            registration = rgb_register(
                ref, linear(observed[theta][0].astype(np.float64) / 65535)
            )
            row.update(
                alpha_measured_delta_deg=measured,
                alpha_error_deg=measured - theta,
                rgb_registration=registration,
                alpha_gate_pass=abs(measured - theta) <= 0.01,
                rgb_registration_gate_pass=abs(registration["angle_deg"] - theta)
                <= 0.01,
            )
        item["status"] = (
            "PIXEL_PILOT_PASS_PROCESSOR_PENDING"
            if all(
                r["alpha_gate_pass"] and r["rgb_registration_gate_pass"]
                for r in item["angles"]
            )
            else "PIXEL_REVIEW_REQUIRED"
        )
        records.append(item)
    np.savez_compressed(out / "object-angle-pilots.npz", **arrays)
    sheet = Image.new("RGB", (5 * 256, 2 * 286), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (label, pixels) in enumerate(tiles):
        x = (i % 5) * 256
        y = (i // 5) * 286
        sheet.paste(Image.fromarray(pixels), (x, y + 28))
        draw.text((x + 5, y + 5), label, fill=(20, 20, 20))
    sheet.save(out / "object-angle-review.png")
    return {
        "status": "REVIEW_PILOT_ONLY",
        "objects": records,
        "data": "object-angle-pilots.npz",
        "preview": "object-angle-review.png",
        "actual_official_processor_status": "PENDING_NEW_OBJECT_PATH",
        "GPU_calls": 0,
        "full_production_authorized": False,
    }


def rdm_tracks(coherence, seed=103):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-240, 240, (3, 18, 2))
    positions = np.full((180, 18, 2), np.nan)
    signals = np.zeros((180, 18), bool)
    wrapped = np.zeros((180, 18), bool)
    for f in range(30, 90):
        group = (f - 30) % 3
        selected = rng.random(18) < coherence
        noise = rng.uniform(-240, 240, (18, 2))
        proposed = pos[group].copy()
        proposed[selected, 0] += 15.0
        proposed[~selected] = noise[~selected]
        hit = np.any((proposed < -240) | (proposed >= 240), axis=1)
        proposed = (proposed + 240) % 480 - 240
        pos[group] = proposed
        positions[f] = proposed
        signals[f] = selected
        wrapped[f] = hit
    return positions, signals, wrapped


def draw_rdm(points, shape=(640, 640)):
    alpha = np.zeros(shape, np.float64)
    covered = {}
    for px, py in points:
        if px * px + py * py >= 240**2:
            continue
        cx, cy = 320 + px, 320 + py
        r = 1.5
        for y in range(max(0, math.floor(cy - r)), min(shape[0], math.ceil(cy + r))):
            for x in range(
                max(0, math.floor(cx - r)), min(shape[1], math.ceil(cx + r))
            ):
                value = circle_pixel_area(x - cx, x + 1 - cx, y - cy, y + 1 - cy, r)
                if value > 0:
                    covered.setdefault((y, x), []).append((cx, cy, value))
    for (y, x), disks in covered.items():
        if len(disks) == 1:
            alpha[y, x] = disks[0][2]
            continue

        # Integrate the actual union where dots overlap; do not sum their area.
        def vertical_union(q):
            intervals = []
            for cx, cy, _ in disks:
                if abs(q - cx) >= 1.5:
                    continue
                half = math.sqrt(2.25 - (q - cx) ** 2)
                lo = max(y, cy - half)
                hi = min(y + 1, cy + half)
                if hi > lo:
                    intervals.append((lo, hi))
            intervals.sort()
            total = 0.0
            last = -math.inf
            for lo, hi in intervals:
                total += max(0.0, hi - max(lo, last))
                last = max(last, hi)
            return total

        cuts = [
            q for cx, cy, _ in disks for q in (cx - 1.5, cx, cx + 1.5) if x < q < x + 1
        ]
        value, error = quad(
            vertical_union, x, x + 1, points=sorted(set(cuts)), epsabs=1e-9, limit=80
        )
        assert error < 1e-6
        alpha[y, x] = np.clip(value, 0, 1)
    return np.repeat(encode(alpha)[:, :, None], 3, axis=2)


def video_pair(out, sid, positions):
    master = out / (sid + ".mkv")
    preview = out / (sid + ".mp4")
    hashes = []
    pts = []
    with av.open(str(master), "w") as a, av.open(str(preview), "w") as b:
        sa = a.add_stream("ffv1", rate=60)
        sa.width = sa.height = 640
        sa.pix_fmt = "gbrp16le"
        sa.options = {"level": "3", "coder": "1", "context": "1", "slicecrc": "1"}
        sb = b.add_stream("libx264", rate=60)
        sb.width = sb.height = 640
        sb.pix_fmt = "yuv420p"
        sb.options = {"crf": "18", "preset": "fast"}
        for i in range(180):
            codes = (
                draw_rdm(positions[i])
                if np.isfinite(positions[i]).all()
                else np.zeros((640, 640, 3), np.uint16)
            )
            hashes.append(hashlib.sha256(codes.tobytes()).hexdigest())
            frame = av.VideoFrame.from_ndarray(codes, format="rgb48le")
            frame.pts = i
            frame.time_base = Fraction(1, 60)
            for packet in sa.encode(frame):
                a.mux(packet)
            preview_frame = av.VideoFrame.from_ndarray(
                (codes >> 8).astype(np.uint8), format="rgb24"
            )
            for packet in sb.encode(preview_frame):
                b.mux(packet)
        for packet in sa.encode():
            a.mux(packet)
        for packet in sb.encode():
            b.mux(packet)
    with av.open(str(master)) as a:
        a.streams.video[0].codec_context.thread_count = 1
        decoded = [
            (
                hashlib.sha256(f.to_ndarray(format="rgb48le").tobytes()).hexdigest(),
                float(f.pts * f.time_base),
            )
            for f in a.decode(video=0)
        ]
    assert [h for h, t in decoded] == hashes and len(decoded) == 180
    assert max(abs(t - i / 60) for i, (h, t) in enumerate(decoded)) <= 0.001
    return {
        "master": master.name,
        "master_sha256": sha(master),
        "bytes": master.stat().st_size,
        "preview": preview.name,
        "preview_bits": 8,
        "frames": 180,
        "fps": 60,
        "per_frame_sha256": hashes,
        "decoded_code_mismatches": 0,
    }


def rdm_pilot(root, out):
    records = []
    arrays = {}
    coherences = (0.0, 0.064, 0.256, 1.0)
    for coherence in coherences:
        positions, signals, wrapped = rdm_tracks(coherence)
        for direction in (("NONE",) if coherence == 0 else ("RIGHT", "LEFT")):
            pos = positions.copy()
            if direction == "LEFT":
                pos[:, :, 0] *= -1
            sid = f"rdm_c{str(coherence).replace('.','p')}_{direction.lower()}"
            arrays[sid + "_positions"] = pos
            arrays[sid + "_signals"] = signals
            arrays[sid + "_wrapped"] = wrapped
            ordinary = signals[30:90]
            visible = (pos[30:90] ** 2).sum(axis=2) < 240**2
            errors = []
            for f in range(33, 90):
                chosen = signals[f] & ~wrapped[f]
                if np.any(chosen):
                    errors.extend(
                        (
                            pos[f, chosen, 0]
                            - pos[f - 3, chosen, 0]
                            - (15 if direction != "LEFT" else -15)
                        ).tolist()
                    )
            assert max(map(abs, errors), default=0) < 1e-10
            assert np.nanmax(abs(pos)) <= 240
            rec = {
                "sample_id": sid,
                "direction_truth": None if direction == "NONE" else direction,
                "coherence_probability": coherence,
                "actual_signal_fraction": float(ordinary.mean()),
                "visible_signal_fraction": float(ordinary[visible].mean()),
                "signal_displacement_error_px": max(map(abs, errors), default=0),
                "wrapped_updates": int(wrapped.sum()),
                "display_group_interval_frames": 3,
                "source_onset_frame": 30,
                "source_offset_frame": 90,
                **video_pair(out, sid, pos),
            }
            records.append(rec)
    # The saved left/right physical points are exact mirrors, including noise.
    for c in coherences[1:]:
        prefix = f"rdm_c{str(c).replace('.','p')}_"
        l = arrays[prefix + "left_positions"]
        r = arrays[prefix + "right_positions"]
        assert np.array_equal(
            l[:, :, 0], -r[:, :, 0], equal_nan=True
        ) and np.array_equal(l[:, :, 1], r[:, :, 1], equal_nan=True)
    np.savez_compressed(out / "rdm-tracks.npz", **arrays)
    return {
        "status": "SOURCE_PILOT_PASS_PROCESSOR_PENDING",
        "clips": records,
        "source_tracks": "rdm-tracks.npz",
        "GPU_calls": 0,
        "upstream": {
            p.name: sha(p) for p in (root / "evidence/classic-rdk-kiani").glob("*.m")
        },
        "declared_variants": [
            "60Hz rather than demo75Hz; same group updates every50ms",
            "FP64 analytic area rasterization rather than integer-center PTB rasterization",
            "symmetric modulo re-insertion and exact LR mirror pairing; upstream dotsShow.m lower-bound branch uses a suspicious minus sign and is not copied literally",
            "pixel coordinate speed300px/s; no physical visual-angle claim",
        ],
        "actual_official_processor_status": "PENDING_NEW_640_SQUARE_PATH",
        "full_production_authorized": False,
    }


def allocation(catalog):
    bgs = {a["id"]: a for a in catalog["backgrounds"]}
    objects = {a["id"]: a for a in catalog["objects"]}
    pairs = []
    for i in range(1, 51):
        a, b = objects.get(f"photo_obj_{i:03}"), objects.get(f"gpt_obj_{i:03}")
        if (
            a
            and b
            and {"T2", "T3", "T4", "T5"}
            <= set(a.get("tasks", [])) & set(b.get("tasks", []))
        ):
            pairs.append(i)
    pairs = sorted(pairs, key=key)[:3]
    assert len(pairs) == 3
    objids = [f"photo_obj_{i:03}" for i in pairs] + [f"gpt_obj_{i:03}" for i in pairs]
    # Four paired background concepts; provenance is balanced within each cell.
    bgpairs = sorted(range(1, 101), key=lambda i: key("bg" + str(i)))[:4]
    bgids = [f"photo_bg_{i:03}" for i in bgpairs] + [f"gpt_bg_{i:03}" for i in bgpairs]
    assert all(i in bgs for i in bgids)
    pattern = ((0, 1, 3), (1, 2, 4), (2, 3, 5), (0, 4, 5))
    combos = []
    for i, bg in enumerate(bgids):
        for j in pattern[(i + (1 if i >= 4 else 0)) % 4]:
            combos.append({"background_id": bg, "object_id": objids[j]})
    assert all(sum(x["background_id"] == v for x in combos) == 3 for v in bgids)
    assert all(sum(x["object_id"] == v for x in combos) == 4 for v in objids)
    return {
        "status": "DETERMINISTIC_REVIEW_CANDIDATES_NOT_INFERENCE_AUTHORIZATION",
        "seed": SEED,
        "library_backgrounds": 200,
        "library_objects": 100,
        "background_ids": bgids,
        "object_ids": objids,
        "combinations": combos,
        "same_combinations_at_every_difficulty": True,
        "per_background_degree": 3,
        "per_object_degree": 4,
        "object_tasks_intersection": ["T2", "T3", "T4", "T5"],
        "selection_basis": "fixed seeded hash and existing task-eligibility tags, no model scores; final pixel suitability/human review pending",
        "T1_proposed_count_formula": "80 SF-depth cells x24 backgrounds x2directions=3840; choose T1 backgrounds independently, phase balance and same-phase classical controls required",
        "full_bg_by_obj_cross_product_generated": False,
        "GPU_calls": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    catalog_path = root / "assets/library/catalog.json"
    catalog = json.loads(catalog_path.read_text())
    result = {
        "schema": "next_task_preparation/v1",
        "status": "CPU_REVIEW_SAMPLES_ONLY",
        "source_sha256": sha(__file__),
        "catalog_sha256": sha(catalog_path),
        "model_calls": 0,
        "T4": rdm_pilot(root, out),
        "object_angle": object_pilot(root, out, catalog),
        "content_allocation": allocation(catalog),
        "T5": {
            "status": "PROTOCOL_REVIEW_CANDIDATE",
            "current_library_source_sha256": sha(
                root / "site/review/library-render.js"
            ),
            "question": "Sequential target-orientation recognition with matched single-event controls and finite masks",
            "current_library": "two events, exposure variable, ISI30frames, canonical target reference, four present/absent scenarios; masks currently persist to frame180",
            "candidate_controls": [
                "matched event1 alone",
                "matched event2 alone",
                "both events with identical target/foil images",
                "mask duration3/6frames followed by restored background",
            ],
            "exposure_frames": [10, 30, 60],
            "scenarios": ["PRESENT_FIRST", "PRESENT_SECOND", "ABSENT_A", "ABSENT_B"],
            "proposal_reason": "Separate basic visibility, sequence cost and persistent-mask interference before increasing content count",
            "formal_RGB16_source_and_processor_qualification": "PENDING",
            "GPU_calls": 0,
            "full_production_authorized": False,
        },
    }
    result["elapsed_CPU_wall_seconds"] = time.monotonic() - began
    (out / "preparation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(out),
                "T4_clips": len(result["T4"]["clips"]),
                "object_statuses": [
                    r["status"] for r in result["object_angle"]["objects"]
                ],
                "candidate_combinations": len(
                    result["content_allocation"]["combinations"]
                ),
                "model_calls": 0,
                "elapsed": result["elapsed_CPU_wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
