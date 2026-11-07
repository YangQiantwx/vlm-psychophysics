"""CPU measurement of the actual official processor, before and after BF16.

This is calibration, not label correction. No model weights or VLM calls.
The post-BF16 image estimators do not imply anything about learned features.
"""

from __future__ import annotations

import gc
import json
import os
from pathlib import Path

import numpy as np

from . import raster
from .manifest import atomic_json, digest, file_sha, state_key


def image_measurement(row, source_codes, restored_srgb):
    p = row["source_parameters"]
    baseline, event = 0, p["onset_frame"]
    indices = [baseline, event]
    observed_rgb = restored_srgb[indices, :512, :768].double().numpy()
    if not np.isfinite(observed_rgb).all():
        raise ValueError("Nonfinite inverse processor pixels")
    actual_y = raster.linear(observed_rgb) @ np.array([0.2126, 0.7152, 0.0722])
    from .verify import linear_y

    source_y = linear_y(source_codes[indices, :512, :768])
    result = {
        "max_sRGB_difference": float(
            np.max(abs(observed_rgb - source_codes[indices, :512, :768] / 65535.0))
        ),
        "observed_sRGB_min": float(observed_rgb.min()),
        "observed_sRGB_max": float(observed_rgb.max()),
        "task": row["task"],
    }
    if row["task"] == "T2":
        a, _ = raster.disk_coverage(area=p["disk_alpha_area_px2"])
        if p["target_side"] == "RIGHT":
            a = np.roll(a, 320, axis=1)
        core = a >= 1 - 1e-12
        actual = float((actual_y[1] - actual_y[0])[core].mean())
        source = float((source_y[1] - source_y[0])[core].mean())
        result.update(
            nominal_delta_Y=p["delta_Y"],
            source_delta_Y=source,
            BF16_delta_Y=actual,
            absolute_nominal_error=abs(actual - p["delta_Y"]),
            absolute_source_error=abs(actual - source),
        )
    else:
        axes = []
        for index, side in enumerate(("LEFT", "RIGHT")):
            cx, cy = p["centers_xy"][index]
            patch = actual_y[1, cy - 128 : cy + 128, cx - 128 : cx + 128]
            # Calibrate opaque foreground/background from observed tensor values.
            bg, fg = float(np.median(patch[:10])), float(patch.max())
            if not fg > bg:
                raise ValueError("BF16 image has no foreground/background separation")
            a = np.clip((patch - bg) / (fg - bg), 0, 1)
            pca, edge = raster.pca_axis(a), raster.edge_axis(a)
            if edge["status"] != "ok":
                raise ValueError("Post-BF16 image edge measurement failed")
            nominal = (
                0.0 if side == p["reference_side"] else p["angle_deg"] * p["tilt_sign"]
            )
            axes.append(
                {
                    "side": side,
                    "nominal_deg": nominal,
                    "PCA_deg": pca,
                    "edge_OLS_deg": edge["axis_deg"],
                    "PCA_error_deg": raster.angle_error(pca, nominal),
                    "edge_OLS_error_deg": raster.angle_error(edge["axis_deg"], nominal),
                    "observed_background_Y": bg,
                    "observed_foreground_Y": fg,
                }
            )
        result["axes"] = axes
    return result


def calibration_identity(
    manifest_path,
    manifest,
    model_key,
    model_root,
    cosmos_package=None,
    runtime_versions=None,
):
    from .inference import (
        checkpoint_asset_identity,
        _versions,
        _cosmos_package_identity,
    )
    from .ready import INPUT_BINDING, representatives

    spec = manifest["models"][model_key]
    return {
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": file_sha(manifest_path),
        "model": spec,
        "input_binding": INPUT_BINDING,
        "representative_row_ids_sha256": digest(
            [r["sample_id"] for r in representatives(manifest)]
        ),
        "checkpoint_assets": checkpoint_asset_identity(
            Path(model_root) / spec["checkpoint_relative"]
        ),
        "runtime_versions": (
            _versions() if runtime_versions is None else runtime_versions
        ),
        "cosmos_package_sources": (
            _cosmos_package_identity(cosmos_package) if model_key == "cosmos" else None
        ),
        "dependent_sources": {
            name: file_sha(Path(__file__).with_name(name))
            for name in ("manifest.py", "verify.py", "raster.py", "ready.py")
        },
        "calibration_source_sha256": file_sha(__file__),
        "processor_runner_sha256": file_sha(Path(__file__).with_name("inference.py")),
    }


def calibration_ready(output_dir, model_key, identity, manifest, ready):
    """One model's fixed128-state evidence, independent of later ready appends."""
    from .ready import representatives

    dest = Path(output_dir) / model_key
    path = dest / "summary.json"
    if not path.exists():
        return False
    summary = json.loads(path.read_text())
    if (
        summary.get("identity") != identity
        or summary.get("status") != "COMPLETE"
        or summary.get("unique_states") != 128
        or summary.get("weights_loaded") is not False
        or summary.get("new_model_calls") != 0
    ):
        raise ValueError("Calibration completion identity differs from this worker")
    rows = representatives(manifest)
    if any(row["sample_id"] not in ready for row in rows):
        return False
    records = []
    for row in rows:
        key, sid = state_key(row), row["sample_id"]
        record = json.loads((dest / (key + ".json")).read_text())
        binding = digest({"row": row, "ready": ready[sid]})
        if (
            record.get("identity") != identity
            or record.get("state_key") != key
            or record.get("sample_id") != sid
            or record.get("row_identity") != binding
            or record.get("source_parameters") != row["source_parameters"]
        ):
            raise ValueError("Calibration row/ready input identity changed")
        records.append(record)
    if summary.get("records_sha256") != digest(records):
        raise ValueError("Calibration records changed after completion")
    return True


def calibrate(
    manifest_path,
    output_dir,
    model_root,
    model_key,
    ready_path=None,
    asset_root=None,
    cosmos_package=None,
    input_poll_seconds=10,
):
    import fcntl
    import time

    if input_poll_seconds < 1:
        raise ValueError("Calibration input poll must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from .inference import ProcessorSession, load_run_inputs, restore_processor_rgb
    from .ready import representatives

    manifest, rows, ready, assets, ready_path, catalog = load_run_inputs(
        manifest_path, ready_path, asset_root, include_catalog=True
    )
    spec = manifest["models"][model_key]
    model_path = Path(model_root) / spec["checkpoint_relative"]
    dest = Path(output_dir) / model_key
    dest.mkdir(parents=True, exist_ok=True)
    identity = calibration_identity(
        manifest_path, manifest, model_key, model_root, cosmos_package
    )
    with (dest / ".calibration.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if calibration_ready(output_dir, model_key, identity, manifest, ready):
            return json.loads((dest / "summary.json").read_text())
        with ProcessorSession(model_key, model_path, cosmos_package) as session:
            records = []
            for row in representatives(manifest):
                key, sid = state_key(row), row["sample_id"]
                while sid not in ready:
                    time.sleep(input_poll_seconds)
                    ready = catalog.refresh()
                binding = digest({"row": row, "ready": ready[sid]})
                path = dest / (key + ".json")
                if path.exists():
                    record = json.loads(path.read_text())
                    if (
                        record["identity"] != identity
                        or record["sample_id"] != sid
                        or record["state_key"] != key
                        or record.get("row_identity") != binding
                        or record["source_parameters"] != row["source_parameters"]
                    ):
                        raise ValueError(
                            "Calibration cache input identity changed; preserve the old evidence"
                        )
                else:
                    prepared = session.prepare(row, ready[sid], assets)
                    pixels = prepared["pixels"].to(torch.bfloat16)
                    rgb = restore_processor_rgb(
                        pixels,
                        prepared["processor"],
                        prepared["source_codes"],
                        model_key,
                    )
                    record = {
                        "identity": identity,
                        "state_key": key,
                        "sample_id": sid,
                        "row_identity": binding,
                        "ready_identity": digest(ready[sid]),
                        "source_parameters": row["source_parameters"],
                        "input_receipt": prepared["input_receipt"],
                        "BF16_measurement": image_measurement(
                            row, prepared["source_codes"], rgb
                        ),
                        "new_model_calls": 0,
                    }
                    atomic_json(path, record)
                    del prepared, pixels, rgb
                    gc.collect()
                records.append(record)
                ready = catalog.refresh()
        t2 = [
            r["BF16_measurement"]["absolute_nominal_error"]
            for r in records
            if r["BF16_measurement"]["task"] == "T2"
        ]
        axes = [
            a
            for r in records
            if r["BF16_measurement"]["task"] == "T3"
            for a in r["BF16_measurement"]["axes"]
        ]
        result = {
            "status": "COMPLETE",
            "identity": identity,
            "unique_states": len(records),
            "records_sha256": digest(records),
            "new_model_calls": 0,
            "weights_loaded": False,
            "T2_BF16_max_absolute_nominal_delta_error": max(t2),
            "T3_BF16_max_image_axis_error_deg": max(
                max(a["PCA_error_deg"], a["edge_OLS_error_deg"]) for a in axes
            ),
            "interpretation": "Observed deployed-input measurements, not source-label corrections or learned-feature error bounds.",
        }
        atomic_json(dest / "summary.json", result)
        return result
