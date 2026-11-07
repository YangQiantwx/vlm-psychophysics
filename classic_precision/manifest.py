"""Immutable scientific rows, explicit identities, and small atomic records."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.partial")
    with temp.open("xb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def load_protocol(path=None):
    return json.loads(
        Path(path or Path(__file__).with_name("protocol.json")).read_text()
    )


def make_manifest(config=None):
    c = config or load_protocol()
    rows = []
    common = {
        "width": c["input"]["source_wh"][0],
        "height": c["input"]["source_wh"][1],
        "fps": c["input"]["fps"],
        "frames": c["input"]["frames"],
        "bits": 16,
    }

    def add(task, p, truth):
        p = {**common, **p, "task": task}
        rows.append(
            {
                "sample_id": task.lower() + "_" + digest(p)[:20],
                "task": task,
                "truth": truth,
                "prompt": c[task]["prompt"],
                "source_parameters": p,
            }
        )

    for duration, levels in [
        (30, c["T2"]["delta_Y"]),
        (10, c["T2"]["control_10frame_delta_Y"]),
    ]:
        for bg in c["T2"]["background_Y"]:
            for delta in levels:
                for index, onset in enumerate(c["onset_frames"]):
                    for side in ("LEFT", "RIGHT"):
                        add(
                            "T2",
                            {
                                "background_Y": bg,
                                "initial_dot_Y": c["T2"]["initial_dot_Y"],
                                "delta_Y": delta,
                                "target_side": side,
                                "disk_alpha_area_px2": c["T2"]["disk_alpha_area_px2"],
                                "disk_diameter_px": c["T2"]["disk_diameter_px"],
                                "centers_xy": c["T2"]["centers_xy"],
                                "onset_index": index,
                                "onset_frame": onset,
                                "offset_frame": onset + duration,
                            },
                            side,
                        )
    for angle in c["T3"]["angle_degrees"]:
        for index, onset in enumerate(c["onset_frames"]):
            for reference in ("LEFT", "RIGHT"):
                for sign in ([1] if angle == 90 else [-1, 1]):
                    add(
                        "T3",
                        {
                            "angle_deg": angle,
                            "tilt_sign": sign,
                            "reference_side": reference,
                            "bar_wh": c["T3"]["bar_wh"],
                            "centers_xy": c["T3"]["centers_xy"],
                            "background_Y": c["T3"]["background_Y"],
                            "foreground_Y": c["T3"]["foreground_Y"],
                            "onset_index": index,
                            "onset_frame": onset,
                            "offset_frame": onset + 30,
                        },
                        reference,
                    )
    counts = {task: sum(r["task"] == task for r in rows) for task in ("T2", "T3")}
    expected = c["counts"]
    if counts != {"T2": expected["T2_total"], "T3": expected["T3_total"]}:
        raise ValueError(("Manifest denominator mismatch", counts, expected))
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated scientific row identity")
    for r in rows:
        p = r["source_parameters"]
        if not 0 <= p["onset_frame"] < p["offset_frame"] <= p["frames"]:
            raise ValueError(("Invalid event window", r["sample_id"]))
    scientific = {
        "name": c["protocol_name"],
        "input": c["input"],
        "rows": rows,
        "generation": c["generation"],
    }
    return {
        "schema_version": 1,
        "protocol_id": digest(scientific),
        "protocol": c,
        "input": c["input"],
        "generation": c["generation"],
        "models": c["models"],
        "counts": counts,
        "rows": rows,
    }


def validate_manifest(manifest):
    """Rebuild GT, prompts, grid and identities from the embedded frozen protocol."""
    if "protocol" not in manifest or make_manifest(manifest["protocol"]) != manifest:
        raise ValueError("Manifest differs from its frozen scientific protocol")
    return manifest


def write_manifest(path, config=None):
    m = make_manifest(config)
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != m:
            raise ValueError("Existing manifest differs; use a new run directory")
    else:
        atomic_json(path, m)
    return m


def selected_rows(manifest, shard_index=0, shard_count=1):
    if not 0 <= shard_index < shard_count:
        raise ValueError("Invalid shard index/count")
    return [
        r
        for r in manifest["rows"]
        if r["source_parameters"]["onset_index"] % shard_count == shard_index
    ]


def state_key(row):
    p = {
        k: v
        for k, v in row["source_parameters"].items()
        if k not in {"onset_index", "onset_frame", "offset_frame"}
    }
    return digest(p)
