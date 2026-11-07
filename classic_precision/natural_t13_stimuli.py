"""CPU-only natural T1/T3 sources. No inference, answer access, or GPU imports.

Public method references and all design choices are written to criteria.json.
Every left/right pair shares assets and nuisance variables. Existing T2 remains
immutable. The renderer uses premultiplied linear RGB and a single source affine.
"""

from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import map_coordinates, sobel
from scipy.sparse import csr_matrix

NAME = "natural-t13-20261029"
SF = [0.5, 1, 2, 4, 8, 16, 32, 64]
DEPTH = [0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
ANGLES = [
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    3.0,
    3.5,
    4.0,
    8.0,
    16.0,
    32.0,
    45.0,
    90.0,
]
Y = np.array([0.2126, 0.7152, 0.0722])
GROUPS = [("photo", "photo"), ("photo", "gpt"), ("gpt", "photo"), ("gpt", "gpt")]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write(p, d):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_suffix(p.suffix + ".tmp")
    q.write_text(json.dumps(d, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    q.replace(p)


def digest(d):
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def rgb8(x):
    assert np.isfinite(x).all() and x.min() >= -1e-10 and x.max() <= 1 + 1e-10
    x = np.clip(x, 0, 1)
    return np.rint(
        255 * np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)
    ).astype(np.uint8)


def axis(a):
    yy, xx = np.indices(a.shape, dtype=float)
    mass = a.sum()
    cx = (a * xx).sum() / mass
    cy = (a * yy).sum() / mass
    xx -= cx
    yy -= cy
    cov = (
        np.array(
            [
                [(a * xx * xx).sum(), (a * xx * yy).sum()],
                [(a * xx * yy).sum(), (a * yy * yy).sum()],
            ]
        )
        / mass
    )
    val, vec = np.linalg.eigh(cov)
    v = vec[:, -1]
    return (
        float(np.degrees(np.arctan2(v[1], v[0]))),
        np.array([cx, cy]),
        float(val[-1] / val[0]),
    )


def angle_diff(a, b):
    return (a - b + 90) % 180 - 90


def affine(source, matrix, size=320, samples=8):
    """Inverse-map bilinear reconstruction integrated over an 8x8 pixel grid."""
    inv = np.linalg.inv(matrix)
    yy, xx = np.indices((size, size), dtype=float)
    out = np.zeros((size, size, 4), dtype=float)
    for dy in (np.arange(samples) + 0.5) / samples - 0.5:
        for dx in (np.arange(samples) + 0.5) / samples - 0.5:
            sx = inv[0, 0] * (xx + dx) + inv[0, 1] * (yy + dy) + inv[0, 2]
            sy = inv[1, 0] * (xx + dx) + inv[1, 1] * (yy + dy) + inv[1, 2]
            for c in range(4):
                out[:, :, c] += map_coordinates(
                    source[:, :, c],
                    [sy, sx],
                    order=1,
                    mode="constant",
                    cval=0,
                    prefilter=False,
                )
    return out / samples**2


def paths(root):
    r = Path(root)
    return (
        r / "plans" / NAME,
        r / "datasets" / NAME,
        r / "runs/stimulus-preparation" / NAME,
    )


def criteria(root):
    p, d, r = paths(root)
    out = {
        "schema": "natural-t13-source-v1",
        "model_calls": 0,
        "counts": {"T1": 3840, "T3": 2736},
        "canvas": [768, 512],
        "fps": 60,
        "frames": 180,
        "encoding": "lossless RGB8 H264, no chroma subsampling",
        "T1": {
            "SF": SF,
            "m": DEPTH,
            "opacity": 0.5,
            "mean_signal_Y": 0.31,
            "diameter": 448,
            "taper_radii": [192, 224],
            "temporal_Hz": 5,
            "phase_degrees": "7.5+15*k, k=0..23; every cell covers all24",
            "per_cell": 48,
            "per_source_per_cell": 24,
            "background_source_phase_parity": "counterbalanced by SF index plus depth index",
            "meaning": "known grating signal added within a half-opacity aperture; m is signal modulation, not final image contrast",
        },
        "T3": {
            "angles": ANGLES,
            "per_angle": 96,
            "at90": 48,
            "per_group_per_angle": 24,
            "at90_per_group": 12,
            "object_area": 3800,
            "axis": "alpha-weighted full-silhouette second-moment principal axis; all objects same rule",
            "axis_ratio_min": 4,
            "max_rotated_radius": 150,
            "axis_error_deg_max": 0.01,
            "area_relative_error_max": 0.001,
            "exposure_frames": 30,
            "onset_bank": "18+5*k, k=0..23; six evenly spread onsets per sign/group/angle; alternate even/odd subsets and sign assignments",
            "groups": GROUPS,
            "semantic_pairs": "both provenance counterparts must pass the same geometry-only gates",
            "scope": "geometric long-axis orientation, not semantic upright or 3D pose; near-isotropic silhouettes excluded with records",
        },
        "balance": {
            "method": "seeded allocation with frequency-preserving swaps; no model outcomes",
            "feature_max_mean_shift_SD": 0.5,
            "family_max_count_deviation": 2.0,
            "first_passing_seed": True,
            "T1_features": "central luminance mean/std, gradient strength and orientation, eight horizontal spectral bands, RGB means; both source types",
            "T3_features": "left/right local background luminance/texture/orientation; object luminance/texture/log elongation and compactness; paired sources",
            "global_reuse": "floor/ceil allocation per asset or semantic pair",
            "no_claim": "balance of measured covariates is not proof of absence of every confound",
        },
        "references": [
            {
                "url": "https://docs.opencv.org/4.13.0/d1/dee/tutorial_introduction_to_pca.html",
                "supports": "principal components as geometric orientation; does not prescribe our ratio gate",
            },
            {
                "url": "https://docs.opencv.org/4.13.0/da/d6e/tutorial_py_geometric_transformations.html",
                "supports": "composed affine rotation/translation/scaling and interpolation",
            },
            {
                "url": "https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html",
                "supports": "official interpolation implementation used by renderer",
            },
            {
                "url": "https://www.w3.org/TR/compositing-1/#simplealphacompositing",
                "supports": "Porter-Duff source-over/premultiplied alpha; linear-light domain is explicitly our choice",
            },
            {
                "url": "https://doi.org/10.1214/12-AOS1008",
                "supports": "pre-outcome covariate balance/rerandomization motivation; our multi-condition constrained design is an adaptation",
            },
            {
                "url": "https://doi.org/10.1037/a0028347",
                "supports": "stimulus identity matters for generalization; repeated videos are not independent scenes",
            },
        ],
        "implementation_sha256": sha(__file__),
        "source_catalog": "assets/library/extensions/t2-natural-20261029/catalog-at-preparation.json",
        "authorization": "User authorized CPU T1/T3 construction, total equal classic; no model inference or public natural-asset upload.",
    }
    q = p / "criteria.json"
    if q.exists():
        previous = read(q)
        assert {k: v for k, v in previous.items() if k != "implementation_sha256"} == {
            k: v for k, v in out.items() if k != "implementation_sha256"
        }, "Existing scientific criteria differ: create a new version"
        return previous  # retain original preparation implementation binding
    else:
        write(q, out)
    return out


def background_from_source(path, width=768, height=512):
    """Linear-light exact pixel-box overlap, identical to the frozen T2 helper."""
    image = np.asarray(Image.open(path).convert("RGB"), np.float64) / 255
    h, w = image.shape[:2]
    if w / h > width / height:
        cw = int(h * width / height)
        x = (w - cw) // 2
        crop = (x, 0, x + cw, h)
    else:
        ch = int(w * height / width)
        y = (h - ch) // 2
        crop = (0, y, w, y + ch)
    x0, y0, x1, y1 = crop
    image = linear(image[y0:y1, x0:x1])

    def weights(n, m):
        edges = np.linspace(0, n, m + 1)
        ii = []
        jj = []
        vv = []
        for i in range(m):
            for j in range(int(np.floor(edges[i])), min(n, int(np.ceil(edges[i + 1])))):
                overlap = max(0.0, min(edges[i + 1], j + 1) - max(edges[i], j))
                if overlap:
                    ii.append(i)
                    jj.append(j)
                    vv.append(overlap / (edges[i + 1] - edges[i]))
        result = csr_matrix((vv, (ii, jj)), shape=(m, n))
        assert np.max(abs(np.asarray(result.sum(axis=1)).ravel() - 1)) < 1e-12
        return result

    wy, wx = weights(image.shape[0], height), weights(image.shape[1], width)
    output = np.stack([(wx @ (wy @ image[:, :, c]).T).T for c in range(3)], axis=-1)
    return output, {
        "crop_xyxy": list(crop),
        "resize": "FP64 linear-RGB exact box overlap",
        "input_dimensions": [w, h],
        "output_dimensions": [width, height],
    }


def bg_features(a, t1=True):
    l = a @ Y
    if t1:
        regions = [l[76:436, 204:564]]
    else:
        regions = [l[144:464, 64:384], l[144:464, 384:704]]
    f = []
    for z in regions:
        gx = sobel(z, 1) / 8
        gy = sobel(z, 0) / 8
        energy = np.mean(gx * gx + gy * gy)
        f += [
            float(z.mean()),
            float(z.std()),
            float(np.sqrt(energy)),
            float(np.mean(gx * gx - gy * gy) / (energy + 1e-12)),
            float(np.mean(2 * gx * gy) / (energy + 1e-12)),
        ]
        if t1:
            xx = np.arange(z.shape[1])
            demean = z - z.mean(axis=1, keepdims=True)
            for sf in SF:
                phase = np.exp(-2j * np.pi * sf * xx / 448)
                f.append(float(np.mean(np.abs(demean @ phase / z.shape[1]) ** 2)))
    if t1:
        f += a[76:436, 204:564].mean(axis=(0, 1)).tolist()
    return f


def object_prepare(job):
    root, asset = job
    root = Path(root)
    p, d, r = paths(root)
    out = d / "objects"
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / (asset["id"] + ".json")
    if receipt.exists():
        return read(receipt)
    src = root / "assets/library" / asset["path"]
    assert sha(src) == asset["sha256"]
    rgba = np.asarray(Image.open(src).convert("RGBA"), dtype=float) / 255
    alpha = rgba[:, :, 3]
    orientation, center, ratio = axis(alpha)
    scale = math.sqrt(3800 / alpha.sum())
    yy, xx = np.where(alpha > 1e-6)
    radius = float(np.hypot(xx - center[0], yy - center[1]).max() * scale)
    record = {
        "asset_id": asset["id"],
        "pair_id": asset["pair_id"],
        "source_sha256": sha(src),
        "source_path": asset["path"],
        "source_axis_deg": orientation,
        "center": center.tolist(),
        "ratio": ratio,
        "scale": scale,
        "radius": radius,
        "accepted": ratio >= 4 and radius <= 150,
        "rule": "same alpha-moment PCA, ratio>=4 and scaled radius<=150",
        "reasons": (["unstable_or_not_elongated_axis"] if ratio < 4 else [])
        + (["rotation_footprint_exceeds_fixed_area_canvas"] if radius > 150 else []),
    }
    if record["accepted"]:
        premul = np.dstack([linear(rgba[:, :, :3]) * alpha[:, :, None], alpha])
        np.save(out / (asset["id"] + "-source.npy"), premul)
        angle = math.radians(90 - orientation)
        m = np.eye(3)
        m[:2, :2] = scale * np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]
        )
        m[:2, 2] = 159.5 - m[:2, :2] @ center
        z = affine(premul, m)
        a = z[:, :, 3]
        pm = z[:, :, :3]
        observed = axis(a)[0]
        assert abs(angle_diff(observed, 90)) <= 0.01, (asset["id"], "axis", observed)
        assert abs(a.sum() / 3800 - 1) <= 0.001, (asset["id"], "area", a.sum())
        mu = float((pm @ Y).sum() / a.sum())
        var = float(
            (
                (np.divide(pm @ Y, a, out=np.zeros_like(a), where=a > 0) - mu) ** 2 * a
            ).sum()
            / a.sum()
        )
        record.update(
            baseline_axis_deg=observed,
            area=float(a.sum()),
            features=[
                mu,
                math.sqrt(var),
                math.log(ratio),
                float(
                    a.sum()
                    / ((xx.max() - xx.min() + 1) * (yy.max() - yy.min() + 1) * scale**2)
                ),
            ],
            source_array_sha256=sha(out / (asset["id"] + "-source.npy")),
        )
        np.save(out / (asset["id"] + "-zero.npy"), z)
        Image.fromarray(rgb8(pm + 0.1 * (1 - a[:, :, None]))).save(
            out / (asset["id"] + "-zero.png")
        )
    write(receipt, record)
    return record


def prepare(root, workers):
    root = Path(root)
    p, d, r = paths(root)
    crit = criteria(root)
    d.mkdir(parents=True, exist_ok=True)
    catalog = read(root / crit["source_catalog"])
    write(p / "catalog.json", catalog)
    records = []
    with ProcessPoolExecutor(workers) as pool:
        for rec in pool.map(
            object_prepare, [(str(root), a) for a in catalog["objects"]]
        ):
            records.append(rec)
    pairs = sorted({x["pair_id"] for x in records if x["accepted"]})
    paired = [
        v
        for v in pairs
        if sum(x["pair_id"] == v and x["accepted"] for x in records) == 2
    ]
    write(
        p / "object-eligibility.json",
        {
            "objects": records,
            "eligible_pairs": paired,
            "method": "all100 measured; both counterparts must pass",
            "model_calls": 0,
        },
    )
    features = []
    sourceprep = root / "datasets/natural-t2-qwen-x-20261029/prepared"
    previous = root / "plans/natural-t2-qwen-x-20261029/asset-preparation.json"
    receipts = (
        {x["asset_id"]: x for x in read(previous)["assets"]}
        if previous.exists()
        else {}
    )
    for asset in catalog["backgrounds"]:
        target = d / "backgrounds" / (asset["id"] + ".npy")
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            entry = read(target.with_suffix(".json"))
        else:
            source = sourceprep / (asset["id"] + ".npz")
            if source.exists() and asset["id"] in receipts:
                assert sha(source) == receipts[asset["id"]]["prepared_sha256"]
                with np.load(source) as z:
                    a = z["rgb"]
                preparation = receipts[asset["id"]]
            else:
                source = root / "assets/library" / asset["path"]
                assert sha(source) == asset["sha256"]
                a, preparation = background_from_source(source)
            np.save(target, a)
            entry = {
                "asset_id": asset["id"],
                "source_sha256": asset["sha256"],
                "prepared_sha256": sha(target),
                "T1": bg_features(a, True),
                "T3": bg_features(a, False),
                "reused_preparation": preparation,
            }
            write(target.with_suffix(".json"), entry)
        features.append(entry)
    write(p / "background-features.json", features)
    print(
        json.dumps(
            {
                "status": "ASSETS_PREPARED",
                "eligible_T3_pairs": len(paired),
                "backgrounds": len(features),
            }
        ),
        flush=True,
    )


def allocation_search(features, families, cell_sizes, seed):
    """Swap assignments, preserve global counts and uniqueness in every cell."""
    rng = np.random.default_rng(seed)
    n = len(features)
    z = (features - features.mean(0)) / np.maximum(features.std(0), 1e-10)
    famnames = sorted(set(families))
    cats = np.array([[float(f == g) for g in famnames] for f in families])
    proportions = cats.mean(0)
    f = np.c_[z, cats]
    offset = np.r_[0, np.cumsum(cell_sizes)]
    cell = np.repeat(np.arange(len(cell_sizes)), cell_sizes)
    # A permutation repeated cyclically gives exact floor/ceil global frequencies.
    a = np.resize(rng.permutation(n), offset[-1])
    groups = [set(a[offset[c] : offset[c + 1]]) for c in range(len(cell_sizes))]
    assert all(len(g) == cell_sizes[c] for c, g in enumerate(groups))
    sums = np.array(
        [f[a[offset[c] : offset[c + 1]]].sum(0) for c in range(len(cell_sizes))]
    )

    def cost(s, sz):
        sd = s[: len(z[0])] / sz
        dev = s[len(z[0]) :] - sz * proportions
        return float(
            np.maximum(abs(sd) - 0.45, 0) @ np.maximum(abs(sd) - 0.45, 0) * 100
            + (sd @ sd) * 0.015
            + np.sum(np.maximum(abs(dev) - 1.8, 0) ** 2) * 5
        )

    costs = np.array([cost(s, sz) for s, sz in zip(sums, cell_sizes)])
    initial = float(costs.sum())
    accepted = 0
    for step in range(500000):
        i, j = rng.integers(0, len(a), 2)
        ci, cj = cell[i], cell[j]
        if ci == cj or a[j] in groups[ci] or a[i] in groups[cj]:
            continue
        delta = f[a[j]] - f[a[i]]
        si = sums[ci] + delta
        sj = sums[cj] - delta
        ni = cost(si, cell_sizes[ci])
        nj = cost(sj, cell_sizes[cj])
        change = ni + nj - costs[ci] - costs[cj]
        temperature = 0.002 * (1 - step / 500000)
        if change < 0 or rng.random() < math.exp(
            -min(700, change / max(temperature, 1e-10))
        ):
            groups[ci].remove(a[i])
            groups[cj].remove(a[j])
            groups[ci].add(a[j])
            groups[cj].add(a[i])
            a[i], a[j] = a[j], a[i]
            sums[ci] = si
            sums[cj] = sj
            costs[ci] = ni
            costs[cj] = nj
            accepted += 1
        if step % 5000 == 0:
            means = sums[:, : z.shape[1]] / np.array(cell_sizes)[:, None]
            deviations = (
                sums[:, z.shape[1] :] - np.array(cell_sizes)[:, None] * proportions
            )
            if abs(means).max() <= 0.5 and abs(deviations).max() <= 2:
                break
    means = sums[:, : z.shape[1]] / np.array(cell_sizes)[:, None]
    deviations = sums[:, z.shape[1] :] - np.array(cell_sizes)[:, None] * proportions
    result = {
        "seed": seed,
        "steps": step + 1,
        "accepted_swaps": accepted,
        "initial_objective": initial,
        "final_objective": float(costs.sum()),
        "max_mean_shift_SD": float(abs(means).max()),
        "max_family_count_deviation": float(abs(deviations).max()),
        "counts": dict(Counter(map(int, a))),
        "passed": bool(abs(means).max() <= 0.5 and abs(deviations).max() <= 2),
        "cell_feature_mean_shifts_SD": means.tolist(),
        "cell_family_counts": sums[:, z.shape[1] :].round(9).tolist(),
        "family_names": famnames,
    }
    return [
        a[offset[c] : offset[c + 1]].tolist() for c in range(len(cell_sizes))
    ], result


def allocate(root):
    root = Path(root)
    p, d, r = paths(root)
    crit = read(p / "criteria.json")
    catalog = read(p / "catalog.json")
    bf = {x["asset_id"]: x for x in read(p / "background-features.json")}
    oe = read(p / "object-eligibility.json")
    of = {x["asset_id"]: x for x in oe["objects"]}
    bgpairs = sorted({a["pair_id"] for a in catalog["backgrounds"]})
    opairs = oe["eligible_pairs"]
    assets = {a["id"]: a for a in catalog["backgrounds"] + catalog["objects"]}
    diagnostics = {}
    alloc = {}
    for name, ids, sizes, feat, kind in [
        (
            "T1_bg",
            bgpairs,
            [12] * 80,
            lambda v: bf["photo_" + v]["T1"] + bf["gpt_" + v]["T1"],
            "bg",
        ),
        (
            "T3_bg",
            bgpairs,
            [12] * 28 + [6],
            lambda v: bf["photo_" + v]["T3"] + bf["gpt_" + v]["T3"],
            "bg",
        ),
        (
            "T3_obj",
            opairs,
            [12] * 28 + [6],
            lambda v: of["photo_" + v]["features"] + of["gpt_" + v]["features"],
            "obj",
        ),
    ]:
        f = np.array([feat(v) for v in ids])
        families = [
            assets["gpt_" + v].get(
                "family",
                assets["photo_" + v].get(
                    "family", "kitchen" if v == "obj_051" else "workbench"
                ),
            )
            for v in ids
        ]
        for seed in range(100):
            selected, diag = allocation_search(f, families, sizes, seed)
            if diag["passed"]:
                break
        assert diag["passed"], (name, diag["max_mean_shift_SD"])
        alloc[name] = [[ids[i] for i in indices] for indices in selected]
        diagnostics[name] = diag
        print(
            json.dumps(
                {
                    "allocation": name,
                    "seed": seed,
                    "max_SD": diag["max_mean_shift_SD"],
                    "steps": diag["steps"],
                }
            ),
            flush=True,
        )
    blocks = []
    for si, sf in enumerate(SF):
        for mi, m in enumerate(DEPTH):
            ci = si * 10 + mi
            for j, bg in enumerate(alloc["T1_bg"][ci]):
                for source in ["photo", "gpt"]:
                    phase = 2 * j + ((si + mi + (source == "gpt")) % 2)
                    blocks.append(
                        {
                            "task": "T1",
                            "sf": sf,
                            "m": m,
                            "phase_index": phase,
                            "background_id": source + "_" + bg,
                            "background_source": source,
                            "cell": ci,
                            "group": source,
                            "cases": [],
                        }
                    )
    rng = np.random.default_rng(20261029)
    for ci, theta in enumerate(ANGLES):
        n = 6 if theta == 90 else 12
        order = rng.permutation(n)
        for j in range(n):
            # Distinct onsets per angle; across 28 non90 angles each onset occurs14 times.
            onset_index = (4 * j + 1) if theta == 90 else (2 * j + ci % 2)
            sign = 1 if theta == 90 else (-1 if (j + ci // 2) % 2 else 1)
            for bgsource, objsource in GROUPS:
                blocks.append(
                    {
                        "task": "T3",
                        "angle": theta,
                        "tilt_sign": sign,
                        "onset_index": onset_index,
                        "onset": 18 + 5 * onset_index,
                        "background_id": bgsource + "_" + alloc["T3_bg"][ci][j],
                        "object_id": objsource
                        + "_"
                        + alloc["T3_obj"][ci][int(order[j])],
                        "background_source": bgsource,
                        "object_source": objsource,
                        "cell": ci,
                        "group": bgsource + "_" + objsource,
                        "cases": [],
                    }
                )
    for block in blocks:
        for side in ["LEFT", "RIGHT"]:
            identity = {k: v for k, v in block.items() if k != "cases"}
            identity["truth"] = side
            block["cases"].append(
                {"sample_id": "n" + digest(identity)[:24], "truth": side}
            )
        block["block_id"] = digest({k: v for k, v in block.items() if k != "cases"})[
            :20
        ]
    counts = Counter(b["task"] for b in blocks)
    assert counts == {"T1": 1920, "T3": 1368}
    result = {
        "schema": "natural-t13-allocation-v1",
        "blocks": blocks,
        "diagnostics": diagnostics,
        "counts": {"T1": 3840, "T3": 2736},
        "criteria_sha256": sha(p / "criteria.json"),
        "catalog_sha256": sha(p / "catalog.json"),
        "model_calls": 0,
        "sample_selection_uses_answers": False,
    }
    write(p / "allocation.json", result)
    print(json.dumps({"status": "ALLOCATED", "blocks": len(blocks)}), flush=True)


@lru_cache(9)
def t1_basis(sf):
    """Same mathematics as the certified classic T1 producer (analytic+GL16)."""
    yy, xx = np.indices((512, 768))
    x = xx + 0.5 - 384
    y = yy + 0.5 - 256
    far = np.hypot(abs(x) + 0.5, abs(y) + 0.5)
    near = np.hypot(np.maximum(abs(x) - 0.5, 0), np.maximum(abs(y) - 0.5, 0))
    core = far <= 192
    taper = (far > 192) & (near < 224)
    k = 2 * np.pi * sf / 448
    c = np.zeros_like(x)
    s = np.zeros_like(x)
    c[core] = np.sinc(sf / 448) * np.cos(k * x[core])
    s[core] = np.sinc(sf / 448) * np.sin(k * x[core])
    gx, gw = np.polynomial.legendre.leggauss(16)
    gx /= 2
    gw /= 2
    xt = x[taper]
    yt = y[taper]
    cc = np.zeros_like(xt)
    ss = cc.copy()
    for dx, wx in zip(gx, gw):
        w = np.zeros_like(xt)
        for dy, wy in zip(gx, gw):
            radius = np.hypot(xt + dx, yt + dy)
            w += wy * np.where(
                radius <= 192,
                1,
                np.where(
                    radius < 224, 0.5 * (1 + np.cos(np.pi * (radius - 192) / 32)), 0
                ),
            )
        cc += wx * w * np.cos(k * (xt + dx))
        ss += wx * w * np.sin(k * (xt + dx))
    c[taper] = cc
    s[taper] = ss
    return c, s


def t1_measure(bank, indices, block, side):
    # Independent measurement from final RGB8 central scan line; no frame metadata
    # is used to obtain phase slope or direction. Expected values only compare.
    frames = np.array(
        [(linear(bank[i][256, 204:564].astype(float) / 255) @ Y) for i in indices]
    )
    t = np.arange(180) / 60
    design = np.column_stack(
        [np.ones(180), np.cos(2 * np.pi * 5 * t), np.sin(2 * np.pi * 5 * t)]
    )
    fit = np.linalg.lstsq(design, frames, rcond=None)[0]
    z = fit[1] + 1j * fit[2]
    phase = np.unwrap(np.angle(z))
    x = np.arange(204, 564) + 0.5 - 384
    slope, intercept = np.polyfit(x, phase, 1)
    sign = 1 if slope > 0 else -1
    observed_sf = abs(slope) * 448 / (2 * np.pi)
    m = float(np.median(abs(z)) / (0.5 * 0.31 * np.sinc(observed_sf / 448)))
    true_sign = -1 if side == "LEFT" else 1
    phi = math.radians(7.5 + 15 * block["phase_index"])
    phi_err = (sign * intercept - phi + np.pi) % (2 * np.pi) - np.pi
    q = {
        "measured_sf": float(observed_sf),
        "sf_error": float(observed_sf - block["sf"]),
        "measured_signal_m": m,
        "m_error": m - block["m"],
        "phase_error_rad": float(phi_err),
        "direction_sign": sign,
        "method": "RGB8 central scanline: temporal sinusoid LS then spatial complex-phase slope; static background absorbed by per-pixel intercept",
    }
    q["passed"] = bool(
        sign == true_sign
        and abs(q["sf_error"]) <= 0.015
        and abs(q["m_error"]) <= 0.003
        and abs(phi_err) <= 0.05
    )
    assert q["passed"], (block["block_id"], side, q)
    return q


def patch(root, ident, theta):
    import fcntl

    p, d, r = paths(root)
    out = d / "patches"
    out.mkdir(exist_ok=True)
    tag = f"{ident}_{theta:+g}"
    target = out / (tag + ".npy")
    record = out / (tag + ".json")
    with (out / (tag + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return np.load(target), read(record)
        base = read(d / "objects" / (ident + ".json"))
        src = np.load(d / "objects" / (ident + "-source.npy"), mmap_mode="r")
        t = math.radians(90 - base["source_axis_deg"] + theta)
        m = np.eye(3)
        m[:2, :2] = base["scale"] * np.array(
            [[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]]
        )
        m[:2, 2] = 159.5 - m[:2, :2] @ np.array(base["center"])
        attempts = []
        # Uniform numerical rule for every patch: refine the integration mesh
        # when the measured geometry misses the fixed, predeclared tolerances.
        for samples in [8, 16, 32]:
            z = affine(src, m, samples=samples)
            a = z[:, :, 3]
            measured = axis(a)[0]
            error = angle_diff(measured, 90 + theta)
            attempts.append(
                {
                    "samples_per_axis": samples,
                    "axis_error_deg": error,
                    "area_error_relative": float(a.sum() / 3800 - 1),
                }
            )
            if abs(error) <= 0.01 and abs(a.sum() / 3800 - 1) <= 0.001:
                break
        assert abs(error) <= 0.01, (ident, theta, "axis_error", error)
        assert abs(a.sum() / 3800 - 1) <= 0.001, (ident, theta, "area", a.sum())
        assert not any(
            np.any(edge) for edge in [a[0], a[-1], a[:, 0], a[:, -1]]
        ), "Object clipping"
        meta = {
            "asset_id": ident,
            "source_sha256": base["source_sha256"],
            "theta": theta,
            "matrix": m.tolist(),
            "measured_axis_deg": measured,
            "axis_error_deg": error,
            "alpha_area": float(a.sum()),
            "integration_attempts": attempts,
            "implementation_sha256": sha(__file__),
        }
        temp = target.with_name(target.name + f".{os.getpid()}.tmp")
        with temp.open("wb") as f:
            np.save(f, z)
        temp.replace(target)
        meta["patch_sha256"] = sha(target)
        write(record, meta)
        return z, meta


def stamp(bg, z, cx, cy):
    h, w = z.shape[:2]
    x = cx - w // 2
    y = cy - h // 2
    bg[y : y + h, x : x + w] = (
        z[:, :, :3] + (1 - z[:, :, 3, None]) * bg[y : y + h, x : x + w]
    )


def encode_verify(path, bank, indices):
    import av, cv2

    cv2.setNumThreads(1)
    path = Path(path)
    temp = path.with_suffix(".partial.mp4")
    out = av.open(str(temp), "w")
    stream = out.add_stream("libx264rgb", rate=60)
    stream.width = 768
    stream.height = 512
    stream.pix_fmt = "rgb24"
    stream.options = {
        "crf": "0",
        "preset": "ultrafast",
        "threads": "1",
        "x264-params": "ref=16:scenecut=0",
    }
    stream.codec_context.thread_count = 1
    expected = hashlib.sha256()
    for n, i in enumerate(indices):
        raw = bank[i]
        expected.update(raw.tobytes())
        frame = av.VideoFrame.from_ndarray(raw, format="rgb24")
        frame.pts = n
        frame.time_base = Fraction(1, 60)
        for packet in stream.encode(frame):
            out.mux(packet)
    for packet in stream.encode():
        out.mux(packet)
    out.close()
    cap = cv2.VideoCapture(str(temp), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    actual = hashlib.sha256()
    n = 0
    assert abs(cap.get(cv2.CAP_PROP_FPS) - 60) < 1e-6
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        assert n < len(indices) and np.array_equal(rgb, bank[indices[n]]), (
            path,
            n,
            "decoded RGB differs",
        )
        assert abs(cap.get(cv2.CAP_PROP_POS_MSEC) - n * 1000 / 60) < 0.001, (
            path,
            n,
            "frame timestamp differs",
        )
        actual.update(rgb.tobytes())
        n += 1
    cap.release()
    assert n == 180 and actual.hexdigest() == expected.hexdigest()
    temp.replace(path)
    return {
        "video_sha256": sha(path),
        "RGB_stream_sha256": actual.hexdigest(),
        "frames_byte_equal": n,
        "timestamps_checked": n,
        "bytes": path.stat().st_size,
    }


def render_one(job):
    root, block = job
    root = Path(root)
    p, d, r = paths(root)
    q = d / "qa" / (block["block_id"] + ".json")
    start = time.monotonic()
    if q.exists():
        old = read(q)
        assert all(
            sha(d / "videos" / block["task"] / (x["sample_id"] + ".mp4"))
            == x["video_sha256"]
            for x in old["cases"]
        )
        return {"block_id": block["block_id"], "existing": True}
    b = np.load(d / "backgrounds" / (block["background_id"] + ".npy"))
    patches = []
    cases = []
    if block["task"] == "T1":
        w, _ = t1_basis(0.0)
        c, s = t1_basis(block["sf"])
        base = (1 - 0.5 * w[:, :, None]) * b + 0.5 * w[:, :, None] * 0.31
        bank = []
        parity = block["phase_index"] % 2
        for j in range(12):
            phase = np.deg2rad(7.5 + 15 * (parity + 2 * j))
            image = (
                base
                + 0.5
                * 0.31
                * block["m"]
                * (c * np.cos(phase) - s * np.sin(phase))[:, :, None]
            )
            bank.append(rgb8(image))
        outside = w == 0
        original = rgb8(b)
        assert all(np.array_equal(x[outside], original[outside]) for x in bank)
    else:
        z0, m0 = patch(root, block["object_id"], 0.0)
        zt, mt = patch(root, block["object_id"], block["tilt_sign"] * block["angle"])
        patches = [m0, mt]
        baseline = rgb8(b)
    for case in block["cases"]:
        if block["task"] == "T1":
            sign = -1 if case["truth"] == "LEFT" else 1
            indices = [
                ((block["phase_index"] - 2 * sign * f) % 24) // 2 for f in range(180)
            ]
            metric = t1_measure(bank, indices, block, case["truth"])
        else:
            event = b.copy()
            stamp(event, z0 if case["truth"] == "LEFT" else zt, 224, 304)
            stamp(event, zt if case["truth"] == "LEFT" else z0, 544, 304)
            bank = [baseline, rgb8(event)]
            indices = [
                int(block["onset"] <= f < block["onset"] + 30) for f in range(180)
            ]
            metric = {
                "axis_errors_deg": [x["axis_error_deg"] for x in patches],
                "passed": True,
                "offset": block["onset"] + 30,
            }
            left, right = (z0, zt) if case["truth"] == "LEFT" else (zt, z0)
            roi = np.zeros((512, 768), bool)
            roi[144:464, 64:384] = left[:, :, 3] > 0
            roi[144:464, 384:704] = right[:, :, 3] > 0
            # Both alternatives use the same two source patches; no side-specific scale.
            assert np.array_equal(
                bank[0][~roi], bank[1][~roi]
            ), "Pixels outside objects changed"
        dest = d / "videos" / block["task"] / (case["sample_id"] + ".mp4")
        dest.parent.mkdir(parents=True, exist_ok=True)
        receipt = encode_verify(dest, bank, indices)
        cases.append({**case, **receipt, "parameter_measurement": metric})
        # Predeclared samples: first two blocks in each task/cell (selected in render()).
        if block.get("review", False):
            rd = d / "review"
            rd.mkdir(exist_ok=True)
            frames = (
                [0, 1, 3, 6]
                if block["task"] == "T1"
                else [
                    block["onset"] - 1,
                    block["onset"],
                    block["onset"] + 29,
                    block["onset"] + 30,
                ]
            )
            for frame in frames:
                Image.fromarray(bank[indices[frame]]).save(
                    rd / f'{case["sample_id"]}_f{frame:03}.png'
                )
    record = {
        **block,
        "cases": cases,
        "patches": patches,
        "source_background_sha256": read(
            (d / "backgrounds" / (block["background_id"] + ".json"))
        )["source_sha256"],
        "criteria_sha256": sha(p / "criteria.json"),
        "implementation_sha256": sha(__file__),
        "elapsed_seconds": time.monotonic() - start,
        "status": "PASS",
    }
    write(q, record)
    return {
        "block_id": block["block_id"],
        "seconds": record["elapsed_seconds"],
        "task": block["task"],
    }


def render(root, workers, limit):
    root = Path(root)
    p, d, r = paths(root)
    data = read(p / "allocation.json")
    blocks = data["blocks"]
    seen = Counter()
    for b in blocks:
        k = (b["task"], b["cell"])
        b["review"] = seen[k] == 0
        seen[k] += 1
    # Pilot includes both task types before committing bulk CPU work.
    if limit:
        blocks = [
            b
            for b in blocks
            if (b["task"] == "T1" and b["cell"] in [0, 79] and b["review"])
            or (b["task"] == "T3" and b["cell"] in [0, 3, 28] and b["review"])
        ][:limit]
    r.mkdir(parents=True, exist_ok=True)
    source = r / ("execution-source-" + sha(__file__)[:12] + ".py")
    if not source.exists():
        source.write_bytes(Path(__file__).read_bytes())
    policy = {
        "criteria_sha256": sha(p / "criteria.json"),
        "source_sha256": sha(__file__),
        "T1_recovery_limits": {
            "SF_absolute": 0.015,
            "signal_m_absolute": 0.003,
            "phase_radians": 0.05,
            "direction": "exact",
        },
        "video": {
            "encoder": "libx264rgb",
            "crf": 0,
            "preset": "ultrafast",
            "reference_frames": 16,
            "scenecut": 0,
            "RGB_bytes": "all180 identical to renderer",
            "timestamps": "every n/60 within .001ms",
        },
        "reason": "Reference-frame count changes lossless storage only; original pilot streams retained separately.",
    }
    write(r / ("render-policy-" + sha(__file__)[:12] + ".json"), policy)
    write(
        r / "render-start.json",
        {
            "status": "RUNNING_CPU_ONLY",
            "workers": workers,
            "blocks": len(blocks),
            "source_sha256": sha(__file__),
            "model_calls": 0,
            "pid": os.getpid(),
        },
    )
    with ProcessPoolExecutor(workers) as pool:
        for i, result in enumerate(
            pool.map(render_one, [(str(root), b) for b in blocks])
        ):
            if i % 20 == 0:
                write(
                    r / "progress.json",
                    {
                        "closed_blocks_this_dispatch": i + 1,
                        "total_blocks_this_dispatch": len(blocks),
                        "latest": result,
                        "monotonic": time.monotonic(),
                        "pid": os.getpid(),
                    },
                )
                print(
                    json.dumps(
                        {
                            "completed_blocks": i + 1,
                            "total": len(blocks),
                            "latest": result,
                        }
                    ),
                    flush=True,
                )
    write(
        r / "render-finished.json",
        {
            "status": "CPU_RENDER_CLOSED_PENDING_AUDIT",
            "blocks": len(blocks),
            "source_sha256": sha(__file__),
            "model_calls": 0,
        },
    )


def audit(root):
    root = Path(root)
    p, d, r = paths(root)
    allocation = read(p / "allocation.json")
    rows = []
    receipts = []
    for block in allocation["blocks"]:
        q = d / "qa" / (block["block_id"] + ".json")
        assert q.exists(), ("missing", block["block_id"])
        rec = read(q)
        assert rec["status"] == "PASS"
        assert {x["sample_id"] for x in rec["cases"]} == {
            x["sample_id"] for x in block["cases"]
        }
        for case in rec["cases"]:
            assert (
                case["frames_byte_equal"] == 180
                and case["parameter_measurement"]["passed"]
            )
            assert (
                sha(d / "videos" / block["task"] / (case["sample_id"] + ".mp4"))
                == case["video_sha256"]
            )
            rows.append({**{k: v for k, v in block.items() if k != "cases"}, **case})
        receipts.append({"block_id": block["block_id"], "sha256": sha(q)})
    counts = Counter(x["task"] for x in rows)
    assert counts == {"T1": 3840, "T3": 2736}
    assert len({x["sample_id"] for x in rows}) == 6576
    assert len({x["video_sha256"] for x in rows}) == 6576
    assert len({x["RGB_stream_sha256"] for x in rows}) == 6576
    summary = {
        "status": "COMPLETE_CPU_SOURCE_VALIDATION",
        "model_calls": 0,
        "counts": dict(counts),
        "video_bytes": sum(x["bytes"] for x in rows),
        "frames_verified": 180 * len(rows),
        "T1_max_abs_parameter_errors": {
            k: max(
                abs(x["parameter_measurement"][k]) for x in rows if x["task"] == "T1"
            )
            for k in ["sf_error", "m_error", "phase_error_rad"]
        },
        "T3_max_abs_axis_error_deg": max(
            abs(v)
            for x in rows
            if x["task"] == "T3"
            for v in x["parameter_measurement"]["axis_errors_deg"]
        ),
        "groups": dict(Counter(x["task"] + ":" + x["group"] for x in rows)),
        "criteria_sha256": sha(p / "criteria.json"),
        "allocation_sha256": sha(p / "allocation.json"),
        "receipts": receipts,
        "scope": "all encoded RGB frames verified against intended frames; geometric/signal-source validation, no processor or human behavioral qualification",
    }
    write(r / "source-audit.json", summary)
    write(p / "source-index.json", {"counts": dict(counts), "rows": rows})
    print(json.dumps({k: v for k, v in summary.items() if k != "receipts"}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "allocate", "render", "audit"])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.command == "prepare":
        prepare(args.root, args.workers)
    elif args.command == "allocate":
        allocate(args.root)
    elif args.command == "render":
        render(args.root, args.workers, args.limit)
    elif args.command == "audit":
        audit(args.root)


if __name__ == "__main__":
    main()
