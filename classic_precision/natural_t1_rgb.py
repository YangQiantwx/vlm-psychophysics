"""CPU-only RGB-multiplicative T1 construction and independent source checks.

Reuses the legacy make_frames T1 equation, not the retired alpha-gray renderer.
Background preparation, constrained allocation and lossless RGB I/O are shared
with natural_t13_stimuli. No inference or model outcomes are imported or read.
"""

from __future__ import annotations
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
from . import natural_t13_stimuli as s

NAME = "natural-t1-rgb-20261030"
GAIN = 0.5


def paths(root):
    root = Path(root)
    return (
        root / "plans" / NAME,
        root / "datasets" / NAME,
        root / "runs/stimulus-preparation" / NAME,
    )


def csv_write(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


@lru_cache(4)
def spatial(width=768, height=512):
    yy, xx = np.mgrid[:height, :width]
    x = xx - (width - 1) / 2
    y = yy - (height - 1) / 2
    radius = np.hypot(x, y)
    window = np.where(
        radius <= 192,
        1,
        np.where(radius < 224, 0.5 * (1 + np.cos(np.pi * (radius - 192) / 32)), 0),
    )
    return x, window


def frame_bank(background, sf, m, phase_rad, gain=GAIN):
    """Legacy point-sampled RGB multiplication, with a global headroom gain."""
    height, width = background.shape[:2]
    x, window = spatial(width, height)
    base = gain * background
    phase = 2 * np.pi * sf * x / 448 + phase_rad
    bank = []
    lo = 1.0
    hi = 0.0
    for f in range(12):
        factor = 1 + m * window * np.cos(phase - 2 * np.pi * 5 * f / 60)
        image = base * factor[:, :, None]
        lo = min(lo, float(image.min()))
        hi = max(hi, float(image.max()))
        bank.append(s.rgb8(image))
    assert lo >= -1e-12 and hi <= 1 + 1e-12
    original = s.rgb8(base)
    outside = window == 0
    assert all(np.array_equal(a[outside], original[outside]) for a in bank)
    return bank, {
        "linear_min": lo,
        "linear_max": hi,
        "outside_static": True,
        "clipped_values": 0,
    }


def legacy_audit(root):
    root = Path(root)
    p, d, r = paths(root)
    r.mkdir(parents=True, exist_ok=True)
    old = root / "src/stimuli/build_grid_smoke.py"
    text = old.read_text()
    tree = ast.parse(text)
    nodes = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in ["lin", "rgb", "make_frames"]
    ]
    assert len(nodes) == 3
    rng = np.random.default_rng(0)
    raw = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
    env = {
        "np": np,
        "Image": SimpleNamespace(open=lambda _: Image.fromarray(raw)),
        "ROOT": Path("/unused"),
        "FPS": 60,
        "FRAMES": 180,
        "Y": np.array(s.Y, dtype=np.float32),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(old), "exec"), env)
    records = []
    for sf, m in [(0.5, 0.05), (8, 0.2), (64, 0.4)]:
        for side in ["LEFT", "RIGHT"]:
            sign = 1 if side == "RIGHT" else -1
            params = {
                "sf_cycles_per_aperture": sf,
                "phase_rad": 0.1308996938995747,
                "tf_hz": 5,
                "direction_sign": sign,
                "multiplier_depth": m,
            }
            row = {
                "task_id": "T1",
                "category": "photo",
                "source_id": "audit",
                "params": params,
            }
            oldframe, *_ = env["make_frames"](row)
            bank, _ = frame_bank(
                env["lin"](Image.fromarray(raw)), sf, m, params["phase_rad"], gain=0.62
            )
            maxdiff = 0
            for f in range(180):
                a = oldframe(f)
                b = bank[(sign * f) % 12]
                maxdiff = max(
                    maxdiff, int(np.max(np.abs(a.astype(int) - b.astype(int))))
                )
                assert np.array_equal(a, b), (sf, m, side, f, maxdiff)
            records.append(
                {
                    "sf": sf,
                    "m": m,
                    "side": side,
                    "frames": 180,
                    "RGB_byte_differences": 0,
                }
            )
    result = {
        "status": "PASS_LEGACY_RGB_EQUATION",
        "legacy_source_sha256": s.sha(old),
        "legacy_function": "make_frames/T1",
        "executed_legacy_AST": True,
        "comparisons": records,
        "formula": "I_c(x,y,t)=g * B_c(x,y) * [1 + m W(x,y) cos(2pi sf x/448 + phase - direction 2pi 5t)]",
        "differences_from_legacy": {
            "canvas": "768x512 instead of 512x512; same centered 448px aperture",
            "gain": "0.5 instead of 0.62, fixed across all assets/levels because max m is now1 instead of0.4",
            "allocation": "3840 videos, all200 backgrounds,24 phases; no model results used",
        },
        "raster_rule": "Legacy pixel-center sampling, no extra spatial filtering; SF is cycles per448px aperture, not display visual degrees.",
        "range_proof": "0<=B_c<=1, 0<=W<=1, 0<=m<=1 imply 0<=0.5*B_c*(1+m*W*cos)<=1. No clipping or per-image exposure adjustment.",
        "color_proof": "The same nonnegative scalar multiplies R/G/B at every pixel; channel ratios are preserved before RGB8 quantization, except undefined ratios at black.",
        "references": [
            {
                "url": "https://www.w3.org/TR/compositing-1/#blendingmultiply",
                "supports": "Componentwise RGB multiplication; does not select our gain or experiment design",
            },
            {
                "url": "https://psychopy.github.io/psychopy-visionscience/coder/EnvelopeGrating/",
                "supports": "Separate carrier/envelope/mask and explicit modulation-depth/headroom control; our positive natural RGB carrier is not a claim of isolated second-order motion",
            },
            {
                "url": "https://doi.org/10.1214/12-AOS1008",
                "supports": "Outcome-independent rerandomization/covariate balance rationale; our frequency-preserving swaps are an adaptation",
            },
        ],
        "not_claimed": "Natural composite contains the background spectrum and product sidebands; m is modulation depth, not whole-scene Michelson contrast. Digital linear RGB is not a calibrated display measurement.",
        "model_calls": 0,
    }
    s.write(r / "legacy-implementation-audit.json", result)
    return result


def balance_phase_assignment(chosen, features, ids):
    """Within-cell swaps preserve every asset/cell count and improve phase balance."""
    arrays = []
    for source in ["photo", "gpt"]:
        f = np.array([features[source + "_" + k]["features"] for k in ids])
        arrays.append((f - f.mean(0)) / np.maximum(f.std(0), 1e-10))
    z = np.array(arrays)
    sums = np.zeros((2, 24, z.shape[-1]))
    counts = np.zeros((2, 24), int)
    for ci, selected in enumerate(chosen):
        si, mi = divmod(ci, 10)
        for j, k in enumerate(selected):
            for src in range(2):
                phase = 2 * j + (si + mi + src) % 2
                sums[src, phase] += z[src, k]
                counts[src, phase] += 1
    assert set(counts.ravel()) == {40}
    initial = float(abs(sums / 40).max())
    rng = np.random.default_rng(20261030)
    accepted = 0

    def cost(a):
        x = a / 40
        return float(
            np.sum(np.maximum(abs(x) - 0.44, 0) ** 2) * 100 + 0.001 * np.sum(x * x)
        )

    for step in range(100000):
        if abs(sums / 40).max() <= 0.45:
            break
        ci = int(rng.integers(80))
        j1, j2 = map(int, rng.choice(12, 2, replace=False))
        k1, k2 = chosen[ci][j1], chosen[ci][j2]
        si, mi = divmod(ci, 10)
        phases = np.array(
            [
                [2 * j1 + (si + mi + src) % 2, 2 * j2 + (si + mi + src) % 2]
                for src in range(2)
            ]
        )
        old = np.stack([sums[src, phases[src]] for src in range(2)])
        delta = z[:, k2] - z[:, k1]
        new = old.copy()
        new[:, 0] += delta
        new[:, 1] -= delta
        if cost(new) < cost(old):
            for src in range(2):
                sums[src, phases[src]] = new[src]
            chosen[ci][j1], chosen[ci][j2] = k2, k1
            accepted += 1
    final = float(abs(sums / 40).max())
    assert final <= 0.5
    return chosen, {
        "method": "within-cell background-pair swaps; source parity and every count unchanged",
        "seed": 20261030,
        "initial_max_phase_mean_shift_SD": initial,
        "final_max_phase_mean_shift_SD": final,
        "accepted_swaps": accepted,
        "proposals": step + 1,
        "gate": 0.5,
        "model_answers_used": False,
    }


def prepare(root):
    root = Path(root)
    p, d, r = paths(root)
    p.mkdir(parents=True, exist_ok=True)
    d.mkdir(parents=True, exist_ok=True)
    r.mkdir(parents=True, exist_ok=True)
    if (p / "allocation.json").exists():
        return {"status": "ALREADY_PREPARED"}
    audit = legacy_audit(root)
    op, od, orr = s.paths(root)
    catalog = s.read(op / "catalog.json")
    oldfeatures = {a["asset_id"]: a for a in s.read(op / "background-features.json")}
    features = {}
    _, window = spatial()
    roi = window > 0
    for a in catalog["backgrounds"]:
        q = od / "backgrounds" / (a["id"] + ".npy")
        assert s.sha(q) == oldfeatures[a["id"]]["prepared_sha256"]
        z = np.load(q)
        l = z @ s.Y
        inside = l[roi]
        extra = [float(np.quantile(inside, v)) for v in [0.1, 0.5, 0.9]] + [
            float(np.mean(inside < 0.01)),
            float(np.sqrt(np.mean((GAIN * l * window) ** 2) / 2)),
            float((z.max(2) - z.min(2))[roi].mean()),
        ]
        features[a["id"]] = {
            "id": a["id"],
            "prepared_relative": str(q.relative_to(root)),
            "prepared_sha256": s.sha(q),
            "source_sha256": a["sha256"],
            "features": oldfeatures[a["id"]]["T1"] + extra,
        }
    ids = sorted({a["pair_id"] for a in catalog["backgrounds"]})
    assets = {a["id"]: a for a in catalog["backgrounds"]}
    f = np.array(
        [
            features["photo_" + k]["features"] + features["gpt_" + k]["features"]
            for k in ids
        ]
    )
    families = [
        assets["gpt_" + k].get("family", assets["photo_" + k].get("family", "unknown"))
        for k in ids
    ]
    attempts = []
    for seed in range(100):
        chosen, diag = s.allocation_search(f, families, [12] * 80, seed)
        attempts.append(
            {
                "seed": seed,
                "passed": diag["passed"],
                "max_SD": diag["max_mean_shift_SD"],
            }
        )
        if diag["passed"]:
            break
    assert diag["passed"]
    chosen, phase_balance = balance_phase_assignment(chosen, features, ids)
    crit = {
        "schema": "natural-t1-rgb-v1",
        "count": 3840,
        "SF": s.SF,
        "m": s.DEPTH,
        "gain": GAIN,
        "canvas": [768, 512],
        "frames": 180,
        "fps": 60,
        "temporal_Hz": 5,
        "aperture": {
            "diameter_px": 448,
            "core_radius_px": 192,
            "taper_outer_radius_px": 224,
            "sampling": "pixel-center, matches legacy RGB implementation",
        },
        "formula": audit["formula"],
        "range_proof": audit["range_proof"],
        "phase": "7.5+15*k, k0..23; each source12 evenly spaced phases per cell; parity counterbalanced across SF/m",
        "allocation": "12 background pairs per cell, each photo/gpt source with matched LEFT/RIGHT; total48 per cell. Original source-pair identity is NOT same-phase photo/gpt causal pairing.",
        "balance_gate": {
            "max_cell_feature_mean_shift_SD": 0.5,
            "max_phase_marginal_feature_mean_shift_SD": 0.5,
            "max_family_count_deviation": 2,
            "global_asset_counts": "floor/ceil",
            "model_answer_selection": False,
        },
        "measurement_gate": {
            "SF_abs_error": 0.015,
            "m_abs_error": 0.003,
            "phase_rad_abs_error": 0.05,
            "direction": "exact",
            "RGB8_line_fit": "Temporal least squares and spatial phase slope on pooled central32rows; fitted DC normalizes the multiplicative amplitude",
        },
        "legacy_audit_sha256": s.sha(r / "legacy-implementation-audit.json"),
        "original_catalog_sha256": s.sha(op / "catalog.json"),
        "source_module_sha256": s.sha(__file__),
        "shared_module_sha256": s.sha(s.__file__),
        "new_model_calls": 0,
    }
    s.write(p / "criteria.json", crit)
    s.write(p / "catalog.json", {"backgrounds": catalog["backgrounds"]})
    s.write(p / "background-features.json", features)
    shutil.copy2(op / "T1-prompt.txt", p / "T1-prompt.txt")
    blocks = []
    for si, sf in enumerate(s.SF):
        for mi, m in enumerate(s.DEPTH):
            ci = si * 10 + mi
            for j, asset_index in enumerate(chosen[ci]):
                for source in ["photo", "gpt"]:
                    phase = 2 * j + ((si + mi + (source == "gpt")) % 2)
                    b = {
                        "task": "T1",
                        "protocol": NAME,
                        "sf": sf,
                        "m": m,
                        "phase_index": phase,
                        "background_id": source + "_" + ids[asset_index],
                        "background_source": source,
                        "group": source,
                        "cell": ci,
                    }
                    b["block_id"] = s.digest(b)[:20]
                    b["cases"] = [
                        {
                            "sample_id": "n" + s.digest({**b, "truth": side})[:24],
                            "truth": side,
                        }
                        for side in ["LEFT", "RIGHT"]
                    ]
                    blocks.append(b)
    s.write(
        p / "allocation.json",
        {
            "blocks": blocks,
            "counts": {"T1": 3840},
            "criteria_sha256": s.sha(p / "criteria.json"),
            "diagnostics": diag,
            "phase_balance": phase_balance,
            "seed_attempts": attempts,
            "model_calls": 0,
        },
    )
    return design(root)


def design(root):
    root = Path(root)
    p, d, r = paths(root)
    a = s.read(p / "allocation.json")
    blocks = a["blocks"]
    bf = s.read(p / "background-features.json")
    catalog = s.read(p / "catalog.json")
    assets = {x["id"]: x for x in catalog["backgrounds"]}
    assert (
        len(blocks) == 1920
        and len({c["sample_id"] for b in blocks for c in b["cases"]}) == 3840
    )
    rows = []
    levels = []
    counts = Counter()
    pairids = sorted({v.split("_", 1)[1] for v in bf})
    maxsd = 0
    maxfam = 0
    marginals = {}
    for source in ["photo", "gpt"]:
        ids = [source + "_" + k for k in pairids]
        f = np.array([bf[k]["features"] for k in ids])
        z = (f - f.mean(0)) / np.maximum(f.std(0), 1e-10)
        ix = {k: i for i, k in enumerate(ids)}
        families = [assets[k].get("family", "unknown") for k in ids]
        rates = {k: v / 100 for k, v in Counter(families).items()}
        for ci in range(80):
            bs = [b for b in blocks if b["group"] == source and b["cell"] == ci]
            chosen = [b["background_id"] for b in bs]
            assert len(bs) == len(set(chosen)) == 12
            phases = [b["phase_index"] for b in bs]
            assert sorted(np.diff(sorted(phases)).tolist()) == [2] * 11
            for b in bs:
                assert [c["truth"] for c in b["cases"]] == ["LEFT", "RIGHT"]
                counts[b["background_id"]] += 2
            shift = z[[ix[k] for k in chosen]].mean(0)
            fc = Counter(assets[k].get("family", "unknown") for k in chosen)
            dev = max(abs(fc[k] - 12 * v) for k, v in rates.items())
            maxsd = max(maxsd, float(abs(shift).max()))
            maxfam = max(maxfam, dev)
            assert abs(shift).max() <= 0.5 + 1e-10 and dev <= 2 + 1e-10
            phase_moments = [
                float(
                    abs(np.exp(1j * n * np.deg2rad(7.5 + 15 * np.array(phases))).mean())
                )
                for n in range(1, 6)
            ]
            assert max(phase_moments) < 1e-12
            rows.append(
                {
                    "group": source,
                    "cell": ci,
                    "sf": bs[0]["sf"],
                    "m": bs[0]["m"],
                    "videos": 24,
                    "LEFT": 12,
                    "RIGHT": 12,
                    "backgrounds": 12,
                    "phases": 12,
                    "max_mean_shift_SD": float(abs(shift).max()),
                    "max_family_count_deviation": dev,
                    "phase_first5_circular_moments_max": max(phase_moments),
                }
            )
            levels.append(
                {
                    "group": source,
                    "cell": ci,
                    "feature_mean_shifts_SD": shift.tolist(),
                    "family_counts": dict(fc),
                    "phases": phases,
                }
            )
        for key in ["phase_index", "sf", "m"]:
            marginal = []
            for value in sorted({b[key] for b in blocks}):
                bs = [b for b in blocks if b["group"] == source and b[key] == value]
                selected = [ix[b["background_id"]] for b in bs]
                marginal.append(
                    {
                        "value": value,
                        "videos": 2 * len(bs),
                        "max_mean_shift_SD": float(abs(z[selected].mean(0)).max()),
                    }
                )
            maximum = max(x["max_mean_shift_SD"] for x in marginal)
            assert maximum <= 0.5 + 1e-10
            marginals[source + "_" + key] = {"maximum": maximum, "levels": marginal}
    for ci in range(80):
        assert sorted(b["phase_index"] for b in blocks if b["cell"] == ci) == list(
            range(24)
        )
    assert len(counts) == 200 and set(counts.values()) == {18, 20}
    phasecounts = Counter((b["group"], b["phase_index"]) for b in blocks)
    assert set(phasecounts.values()) == {40}
    result = {
        "status": "PASS_ALLOCATION_RECALCULATED",
        "count": 3840,
        "cells": 80,
        "videos_per_cell": 48,
        "backgrounds": 200,
        "photo_videos": 1920,
        "generated_videos": 1920,
        "asset_reuse_range": [18, 20],
        "max_cell_feature_mean_shift_SD": maxsd,
        "max_family_count_deviation": maxfam,
        "features_per_source": len(next(iter(bf.values()))["features"]),
        "each_cell_all24_phases": True,
        "each_source_cell12_uniform_phases": True,
        "each_source_phase_total_videos": 80,
        "LR_every_asset_phase_pair": True,
        "background_static": True,
        "seed": a["diagnostics"]["seed"],
        "allocation_sha256": s.sha(p / "allocation.json"),
        "new_model_calls": 0,
        "scope": "Measured predeclared covariates and exact count constraints; not a guarantee of balance on every unmeasured property.",
    }
    result["max_phase_feature_mean_shift_SD"] = max(
        v["maximum"] for k, v in marginals.items() if "phase_index" in k
    )
    csv_write(r / "allocation-balance-cells.csv", rows)
    csv_write(
        r / "asset-usage.csv",
        [{"asset_id": k, "videos": v} for k, v in sorted(counts.items())],
    )
    s.write(r / "allocation-feature-details.json", levels)
    s.write(r / "marginal-balance-audit.json", marginals)
    s.write(r / "allocation-balance-audit.json", result)
    return result


def measure(bank, indices, block, side):
    # Pooled rows improve RGB8 recovery without reading the source image or GT.
    frames = np.array(
        [
            (s.linear(bank[i][240:272, 204:564].astype(float) / 255) @ s.Y).mean(0)
            for i in indices[:12]
        ]
    )
    t = np.arange(12) / 60
    mat = np.column_stack(
        [np.ones(12), np.cos(2 * np.pi * 5 * t), np.sin(2 * np.pi * 5 * t)]
    )
    fit = np.linalg.lstsq(mat, frames, rcond=None)[0]
    z = fit[1] + 1j * fit[2]
    x = np.arange(204, 564) + 0.5 - 384
    phase = np.unwrap(np.angle(z))
    weight = np.abs(z)
    slope, intercept = np.polyfit(x, phase, 1, w=weight)
    direction = 1 if slope > 0 else -1
    sf = abs(slope) * 448 / (2 * np.pi)
    valid = fit[0] > 0.003
    assert (
        valid.sum() >= 100
    ), "Insufficient decoded modulation for independent recovery"
    depth = float(np.median(np.abs(z[valid]) / fit[0, valid]))
    phi = np.deg2rad(7.5 + 15 * block["phase_index"])
    pe = float((direction * intercept - phi + np.pi) % (2 * np.pi) - np.pi)
    q = {
        "measured_sf": float(sf),
        "sf_error": float(sf - block["sf"]),
        "measured_m": depth,
        "m_error": depth - block["m"],
        "phase_error_rad": pe,
        "direction_sign": direction,
        "valid_columns": int(valid.sum()),
        "method": "RGB8 only:32row-pooled temporal LS, spatial complex-phase slope, amplitude/DC",
    }
    q["passed"] = bool(
        direction == (1 if side == "RIGHT" else -1)
        and abs(q["sf_error"]) <= 0.015
        and abs(q["m_error"]) <= 0.003
        and abs(pe) <= 0.05
    )
    assert q["passed"], (block["block_id"], q)
    return q


def render_one(job):
    root, block = job
    root = Path(root)
    p, d, r = paths(root)
    q = d / "qa" / (block["block_id"] + ".json")
    start = time.monotonic()
    if q.exists():
        old = s.read(q)
        assert old["status"] == "PASS" and old["criteria_sha256"] == s.sha(
            p / "criteria.json"
        )
        for c in old["cases"]:
            assert (
                s.sha(d / "videos/T1" / (c["sample_id"] + ".mp4")) == c["video_sha256"]
            )
        return {"block_id": block["block_id"], "existing": True}
    source = s.read(p / "background-features.json")[block["background_id"]]
    bgpath = root / source["prepared_relative"]
    assert s.sha(bgpath) == source["prepared_sha256"]
    bg = np.load(bgpath)
    bank, rangecheck = frame_bank(
        bg, block["sf"], block["m"], np.deg2rad(7.5 + 15 * block["phase_index"])
    )
    cases = []
    for c in block["cases"]:
        sign = 1 if c["truth"] == "RIGHT" else -1
        indices = [sign * f % 12 for f in range(180)]
        metric = measure(bank, indices, block, c["truth"])
        path = d / "videos/T1" / (c["sample_id"] + ".mp4")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.with_suffix(".partial.mp4").exists():
            raise RuntimeError(
                "Uncommitted media retained; inspect rather than overwrite: "
                + str(path)
            )
        rec = s.encode_verify(path, bank, indices)
        cases.append({**c, **rec, "parameter_measurement": metric})
    rec = {
        **block,
        "cases": cases,
        "patches": [],
        "status": "PASS",
        "criteria_sha256": s.sha(p / "criteria.json"),
        "implementation_sha256": s.sha(__file__),
        "background_source_sha256": source["source_sha256"],
        "prepared_sha256": source["prepared_sha256"],
        "float_range_check": rangecheck,
        "seconds": time.monotonic() - start,
    }
    s.write(q, rec)
    return {"block_id": block["block_id"], "seconds": rec["seconds"]}


def render(root, workers, pilot=False):
    root = Path(root)
    p, d, r = paths(root)
    blocks = s.read(p / "allocation.json")["blocks"]
    selected = []
    seen = set()
    if pilot:
        for b in blocks:
            key = (b["cell"], b["group"])
            if b["cell"] in [0, 9, 40, 49, 70, 79] and key not in seen:
                selected.append(b)
                seen.add(key)
        blocks = selected
    else:
        accepted = s.read(r / "pilot-acceptance.json")
        assert accepted["status"] == "PASS_RGB_MULTIPLICATIVE_PILOT"
        assert accepted["visual_review"] == "PASS_RAW_RGB_REPRESENTATIVE_INSPECTION"
        assert accepted["source_sha256"] == s.sha(__file__) and accepted[
            "criteria_sha256"
        ] == s.sha(p / "criteria.json")
    snapshot = r / ("execution-source-" + s.sha(__file__)[:12] + ".py")
    if not snapshot.exists():
        shutil.copy2(__file__, snapshot)
    prefix = "pilot" if pilot else "render"
    s.write(
        r / (prefix + "-start.json"),
        {
            "status": "RUNNING_CPU_ONLY",
            "pid": os.getpid(),
            "workers": workers,
            "blocks": len(blocks),
            "source_sha256": s.sha(__file__),
            "model_calls": 0,
        },
    )
    with ProcessPoolExecutor(workers) as pool:
        for i, v in enumerate(pool.map(render_one, [(str(root), b) for b in blocks])):
            if i % 20 == 0 or i + 1 == len(blocks):
                update = {
                    "closed_blocks": i + 1,
                    "total_blocks": len(blocks),
                    "monotonic": time.monotonic(),
                    "latest": v,
                }
                s.write(r / (prefix + "-progress.json"), update)
                print(json.dumps(update), flush=True)
    s.write(
        r / (prefix + "-finished.json"),
        {
            "status": "CLOSED_CPU_RENDER_PENDING_AUDIT",
            "blocks": len(blocks),
            "model_calls": 0,
        },
    )
    if pilot:
        qs = [s.read(d / "qa" / (b["block_id"] + ".json")) for b in blocks]
        s.write(
            r / "pilot-acceptance.json",
            {
                "status": "PASS_RGB_MULTIPLICATIVE_PILOT",
                "videos": sum(len(q["cases"]) for q in qs),
                "all180_RGB_and_timestamps": True,
                "max_errors": {
                    k: max(
                        abs(c["parameter_measurement"][k])
                        for q in qs
                        for c in q["cases"]
                    )
                    for k in ["sf_error", "m_error", "phase_error_rad"]
                },
                "source_sha256": s.sha(__file__),
                "criteria_sha256": s.sha(p / "criteria.json"),
                "visual_review": "Pending direct inspection of representative raw PNGs",
                "model_calls": 0,
            },
        )
    return {"blocks": len(blocks), "pilot": pilot}


def preview(root):
    from . import natural_t13_review as review

    root = Path(root)
    p, d, r = paths(root)
    dest = r / "previews"
    dest.mkdir(exist_ok=True)
    chosen = {}
    for b in s.read(p / "allocation.json")["blocks"]:
        if (d / "qa" / (b["block_id"] + ".json")).exists():
            chosen.setdefault((b["cell"], b["group"]), b)
    # Reuse the existing lossless-source -> original PNG/display-copy routine.
    oldname = s.NAME
    try:
        s.NAME = NAME
        samples = [review.preview_pair((str(root), b)) for b in chosen.values()]
    finally:
        s.NAME = oldname
    s.write(
        r / "preview-manifest.json",
        {
            "samples": samples,
            "complete": (r / "source-audit.json").exists(),
            "selection": "First allocated LR pair per cell/source; independent of model answers.",
        },
    )
    html = (
        review.HTML.replace("<option>T3</option>", "")
        .replace(
            "<title>Natural T1 · T3</title>",
            "<title>Natural T1 · RGB modulation</title>",
        )
        .replace(
            "Natural backgrounds. Controlled signals.", "Natural T1 · RGB modulation"
        )
        .replace("<div><b>2,736</b>T3 videos</div>", "")
        .replace("Opacity 0.5", "RGB × moving grating")
        .replace(
            "Signal opacity is 0.5; m describes the added grating, not the final scene contrast.",
            "The same moving multiplier modulates R/G/B. Fixed global gain0.5 prevents clipping; m is modulation depth, not total scene contrast.",
        )
        .replace(
            "<p>T3: the same silhouette PCA, fixed area and one affine transform. Four background/object source groups. Each angle has balanced LEFT/RIGHT and tilt signs; 90° has one unique axial sign.</p>",
            "",
        )
        .replace("design-audit.json", "allocation-balance-audit.json")
        .replace(
            '<a href="geometry-audit.json">Geometry</a>',
            '<a href="legacy-implementation-audit.json">Legacy equation</a>',
        )
        .replace(
            "CPU construction · no model inference",
            "RGB multiplication · no model inference",
        )
    )
    (r / "index.html").write_text(html)
    return {"preview_pairs": len(samples)}


def finish(root):
    root = Path(root)
    p, d, r = paths(root)
    allocation = s.read(p / "allocation.json")
    rows = []
    receipts = []
    for b in allocation["blocks"]:
        q = d / "qa" / (b["block_id"] + ".json")
        v = s.read(q)
        assert v["status"] == "PASS" and v["criteria_sha256"] == s.sha(
            p / "criteria.json"
        )
        assert all(v[k] == b[k] for k in b if k != "cases")
        assert [(c["sample_id"], c["truth"]) for c in v["cases"]] == [
            (c["sample_id"], c["truth"]) for c in b["cases"]
        ]
        for c in v["cases"]:
            path = d / "videos/T1" / (c["sample_id"] + ".mp4")
            assert (
                s.sha(path) == c["video_sha256"]
                and c["frames_byte_equal"] == c["timestamps_checked"] == 180
                and c["parameter_measurement"]["passed"]
            )
            rows.append(
                {
                    **{k: w for k, w in b.items() if k != "cases"},
                    **c,
                    "video_relative": str(path.relative_to(root)),
                    "qa_sha256": s.sha(q),
                    "prompt_sha256": s.sha(p / "T1-prompt.txt"),
                }
            )
        receipts.append({"block_id": b["block_id"], "qa_sha256": s.sha(q)})
    assert len(rows) == 3840
    for k in ["sample_id", "video_sha256", "RGB_stream_sha256"]:
        assert len({x[k] for x in rows}) == 3840, k
    summary = {
        "status": "COMPLETE_CPU_RGB_MULTIPLICATIVE_T1",
        "model_calls": 0,
        "videos": 3840,
        "cells": 80,
        "backgrounds": 200,
        "frames_and_timestamps_verified": 3840 * 180,
        "video_bytes": sum(x["bytes"] for x in rows),
        "maximum_abs_errors": {
            k: max(abs(x["parameter_measurement"][k]) for x in rows)
            for k in ["sf_error", "m_error", "phase_error_rad"]
        },
        "allocation_sha256": s.sha(p / "allocation.json"),
        "criteria_sha256": s.sha(p / "criteria.json"),
        "balance_audit_sha256": s.sha(r / "allocation-balance-audit.json"),
        "legacy_audit_sha256": s.sha(r / "legacy-implementation-audit.json"),
        "receipts": receipts,
        "scope": "Exact CPU source decoding and parameter recovery, not model or human validation.",
    }
    s.write(p / "source-index.json", {"rows": rows})
    csv_write(
        r / "source-index.csv",
        [
            {
                k: x.get(k, "")
                for k in [
                    "sample_id",
                    "cell",
                    "group",
                    "sf",
                    "m",
                    "phase_index",
                    "background_id",
                    "truth",
                    "video_relative",
                    "video_sha256",
                    "RGB_stream_sha256",
                ]
            }
            for x in rows
        ],
    )
    s.write(r / "source-audit.json", summary)
    preview(root)
    return {k: v for k, v in summary.items() if k != "receipts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "command",
        choices=[
            "prepare",
            "design",
            "legacy_audit",
            "pilot",
            "render",
            "preview",
            "finish",
        ],
    )
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.command in ["pilot", "render"]:
        result = render(args.root, args.workers, args.command == "pilot")
    else:
        result = globals()[args.command](args.root)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
