"""Portable, read-only RGB16 dataset access; never loads a model or uploads data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import re
from pathlib import Path, PurePosixPath

from .manifest import atomic_json, digest, file_sha


SCHEMA = "classic-precision-rgb16-index-v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _relative(value):
    _require(isinstance(value, str) and value, "Missing relative video path")
    p = PurePosixPath(value)
    _require(not p.is_absolute() and ".." not in p.parts and "\\" not in value,
             "Video path must stay relative to the dataset root")
    return p.as_posix()


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
             "Expected an explicit SHA256 identity")
    return value


def validate_index(index):
    _require(index.get("schema") == SCHEMA, "Unsupported RGB16 dataset index")
    body = {k: v for k, v in index.items() if k != "index_id"}
    _require(index.get("index_id") == digest(body), "Dataset index identity differs")
    items = index.get("items")
    _require(isinstance(items, dict) and items, "Empty dataset index")
    paths, counts = set(), {}
    for sid, entry in items.items():
        _require(entry.get("sample_id") == sid, "Sample ID differs from index key")
        _require(entry.get("task") in ("T1", "T2", "T3"), "Unknown classic task")
        _require(entry.get("truth") in ("LEFT", "RIGHT"), "Missing direction/side truth")
        _require(isinstance(entry.get("prompt"), str) and entry["prompt"], "Missing prompt")
        path = _relative(entry.get("video_relative"))
        _require(path not in paths, "Two IDs refer to the same video path")
        paths.add(path)
        for key in ("video_sha256", "stream_sha256"):
            _sha(entry.get(key))
        _require("source_audit_row_sha256" not in entry or "source_audit_identity" in entry,
                 "Legacy draft mislabels the shared audit identity; rebuild its index")
        if "source_audit_identity" in entry:
            _sha(entry["source_audit_identity"])
        for key in ("width", "height", "frames"):
            _require(type(entry.get(key)) is int and entry[key] > 0,
                     "Invalid video geometry: " + key)
        _require(isinstance(entry.get("fps"), (int, float))
                 and math.isfinite(entry["fps"]) and entry["fps"] > 0, "Invalid frame rate")
        counts[entry["task"]] = counts.get(entry["task"], 0) + 1
    _require(index.get("counts") == counts, "Dataset task denominators differ")
    return index


def index_from_catalog(catalog):
    """Export portable metadata from the already audited catalog, without decoding."""
    entries = {}
    for sid, item in catalog["items"].items():
        row, ready = item["row"], item["ready"]
        _require(sid == item["sample_id"] == row["sample_id"] == ready["sample_id"],
                 "Catalog/row/ready identities differ")
        _require(digest(row) == item["row_sha256"] and digest(ready) == item["ready_sha256"],
                 "Catalog row/ready content differs")
        family = _relative(item["source_family"])
        _require("/" not in family, "Source family must be one directory")
        parameters = row.get("source_parameters", row.get("params"))
        _require(isinstance(parameters, dict), "Missing nominal source parameters")
        rel = ready.get("video_relative", ready.get("video_path"))
        entries[sid] = {
            "sample_id": sid, "task": item["task"],
            "truth": row.get("truth", row.get("correct_answer")), "prompt": row["prompt"],
            "nominal_parameters": parameters,
            "video_relative": family + "/" + _relative(rel),
            "video_sha256": _sha(ready["video_sha256"]),
            "stream_sha256": _sha(item["source_stream_sha256"]),
            "width": row.get("width", parameters.get("width", 768)),
            "height": row.get("height", parameters.get("height", 512)),
            "frames": row.get("frames", parameters.get("frames", 180)),
            "fps": row.get("fps", parameters.get("fps", 60)),
            "source_row_sha256": item["row_sha256"],
            "source_ready_sha256": item["ready_sha256"],
            "source_audit_identity": _sha(item["source_audit_identity"]),
        }
    counts = {task: sum(x["task"] == task for x in entries.values())
              for task in ("T1", "T2", "T3") if any(x["task"] == task for x in entries.values())}
    _require(counts == catalog["counts"], "Catalog denominators differ")
    body = {
        "schema": SCHEMA, "counts": counts,
        "source_catalog_id": digest(catalog),
        "input": {"codec": "ffv1", "encoded_pixel_format": "gbrp16le",
                  "decoded_pixel_format": "rgb48le", "bits_per_channel": 16,
                  "decoded_array": "uint16 [frame, height, width, RGB]",
                  "stream_hash_bytes": "C-order little-endian uint16 RGB",
                  "transfer_function": "sRGB", "max_PTS_error_seconds": .001},
        "items": entries,
    }
    return validate_index({**body, "index_id": digest(body)})


def load_index(path):
    return validate_index(json.loads(Path(path).read_text()))


def decode_video(path, entry):
    """Return original RGB16 codes, PTS and a receipt after full-file verification.

    This does not pad, resize, crop, subsample or run an official model processor.
    Hugging Face cache symlinks are supported; SHA identities bind the target bytes.
    """
    import av
    import numpy as np

    path = Path(path)
    _require(path.is_file(), "Missing video: " + str(path))
    identity = path.stat()
    video_hash = file_sha(path)
    _require(video_hash == entry["video_sha256"], "Encoded video SHA256 differs")
    shape = (entry["frames"], entry["height"], entry["width"], 3)
    codes = np.empty(shape, dtype=np.uint16)
    timestamps, stream_hash = [], hashlib.sha256()
    with av.open(str(path)) as container:
        _require(len(container.streams.video) == 1, "Expected exactly one video stream")
        stream = container.streams.video[0]
        _require(stream.codec_context.name == "ffv1"
                 and stream.codec_context.format.name == "gbrp16le",
                 "Expected the FFV1 gbrp16le scientific master")
        for i, frame in enumerate(container.decode(video=0)):
            _require(i < shape[0], "Video contains extra frames")
            rgb = frame.to_ndarray(format="rgb48le")
            _require(rgb.dtype == np.uint16 and rgb.shape == shape[1:],
                     "Decoded RGB16 shape/dtype differs")
            _require(frame.pts is not None and frame.time_base is not None,
                     "Video frame has no timestamp")
            pts = float(frame.pts * frame.time_base)
            _require(math.isfinite(pts) and abs(pts - i / entry["fps"]) <= .001,
                     "Frame timestamp differs from its nominal source index")
            codes[i] = rgb
            stream_hash.update(rgb.astype("<u2", copy=False).tobytes(order="C"))
            timestamps.append(pts)
    _require(len(timestamps) == shape[0], "Video is missing frames")
    _require(stream_hash.hexdigest() == entry["stream_sha256"], "Decoded RGB stream SHA256 differs")
    after = path.stat()
    _require((identity.st_dev, identity.st_ino, identity.st_size, identity.st_mtime_ns)
             == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
             "Video changed while being read")
    receipt = {
        "status": "COMPLETE_PASS", "sample_id": entry["sample_id"], "task": entry["task"],
        "video_sha256": video_hash, "stream_sha256": stream_hash.hexdigest(),
        "encoded_bytes": after.st_size, "shape": list(codes.shape), "dtype": str(codes.dtype),
        "decoded_pixel_format": "rgb48le", "bits_per_channel": 16,
        "decoded_frames": len(timestamps), "first_last_PTS_seconds": timestamps[::max(1, len(timestamps)-1)],
        "max_PTS_error_seconds": max(abs(t - i / entry["fps"]) for i, t in enumerate(timestamps)),
        "nominal_parameters": entry.get("nominal_parameters", {}),
        "source_audit_identity": entry.get("source_audit_identity"),
        "new_source_numerical_audit": False, "model_processor_calls": 0, "GPU_model_calls": 0,
        "padding_bottom_rows": 0, "resize": False, "crop": False, "frame_selection": "all_in_order",
        "versions": {name: importlib.metadata.version(name) for name in ("numpy", "av")},
    }
    return codes, timestamps, receipt


def read_sample(index, sample_id, root):
    validate_index(index)
    entry = index["items"][sample_id]
    return decode_video(Path(root) / entry["video_relative"], entry)


def rgb16_to_fp32(codes):
    """Use the existing source entry's FP32 [0,255] mapping, without uint8 conversion."""
    import numpy as np
    _require(codes.dtype == np.uint16, "Expected original uint16 RGB codes")
    return codes.astype(np.float32) * (255. / 65535.)


def fetch_sample(repo_id, revision, sample_id, cache_dir=None, *, local_files_only=False):
    """Download one public sample and its index at an explicit immutable Hub commit."""
    _require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
             "Pin a full immutable Hugging Face commit, not main or a moving tag")
    from huggingface_hub import hf_hub_download
    common = dict(repo_id=repo_id, repo_type="dataset", revision=revision,
                  cache_dir=cache_dir, token=False, local_files_only=local_files_only)
    index_path = hf_hub_download(filename="dataset-index.json", **common)
    index = load_index(index_path)
    entry = index["items"][sample_id]
    video = hf_hub_download(filename=entry["video_relative"], **common)
    return decode_video(video, entry)


def public_source_qa(index, audit_rows):
    """Join immutable audited rows without decoding or remeasuring source pixels."""
    validate_index(index)
    audited = {}
    for row in audit_rows:
        sid = row["sample_id"]
        _require(sid not in audited, "Repeated source-audit sample ID")
        audited[sid] = row
    _require(set(audited) == set(index["items"]), "Index/source-audit ID sets differ")
    result = []
    for sid, entry in index["items"].items():
        qa = audited[sid]
        _require(qa["status"] == "PASS", "Source-audit row is not PASS: " + sid)
        bindings = {
            "task": "task", "truth": "truth", "video_sha256": "video_sha256",
            "stream_sha256": "decoded_stream_sha256", "source_row_sha256": "row_sha256",
            "source_ready_sha256": "ready_sha256", "source_audit_identity": "identity",
            "nominal_parameters": "parameters",
        }
        for left, right in bindings.items():
            _require(entry[left] == qa[right], "Index/audit binding differs: " + sid + ":" + left)
        _require(qa["decoded_frames"] == entry["frames"]
                 and qa["dimensions_wh"] == [entry["width"], entry["height"]]
                 and qa["fps"] == entry["fps"] and qa["max_code_error"] == 0,
                 "Audited frame/geometry/precision evidence differs: " + sid)
        _require(qa["every_decoded_RGB_code_compared"] is True
                 and qa["codec"] == "ffv1" and qa["pixel_format"] == "gbrp16le",
                 "Incomplete scientific-master source evidence: " + sid)
        result.append({
            "schema": "classic-precision-public-source-qa-v1", "sample_id": sid,
            "task": entry["task"], "truth": entry["truth"], "status": qa["status"],
            "video_relative": entry["video_relative"], "video_sha256": entry["video_sha256"],
            "decoded_stream_sha256": entry["stream_sha256"],
            "source_row_sha256": entry["source_row_sha256"],
            "source_ready_sha256": entry["source_ready_sha256"],
            "source_audit_identity": qa["identity"],
            "source_audit_row_sha256": digest(qa),
            "nominal_parameters": entry["nominal_parameters"],
            "independent_pixel_measurements": qa["independent_pixel_measurements"],
            "photometric_QA": qa.get("photometric_QA"),
            "measurement_scope": qa.get("measurement_scope", "per-video actual-pixel blind readout"),
            "source_state_key": qa.get("source_state_key"),
            "decoded_frames": qa["decoded_frames"], "dimensions_wh": qa["dimensions_wh"],
            "fps": qa["fps"], "max_decoded_code_error": qa["max_code_error"],
            "max_PTS_error_seconds": qa["max_PTS_error_s"],
            "event_frame_interval_half_open": qa.get("event_frame_interval_half_open"),
            "every_decoded_RGB_code_compared": qa["every_decoded_RGB_code_compared"],
        })
    return result


def qa_csv_row(row):
    """Flat named units for inspection, plus complete nominal parameters as JSON."""
    p, m = row["nominal_parameters"], row["independent_pixel_measurements"]
    out = {k: row[k] for k in ("sample_id", "task", "truth", "status", "video_relative",
           "video_sha256", "decoded_stream_sha256", "source_row_sha256", "source_ready_sha256",
           "source_audit_identity", "source_audit_row_sha256", "measurement_scope",
           "decoded_frames", "fps", "max_decoded_code_error", "max_PTS_error_seconds")}
    out.update(nominal_parameters_json=json.dumps(p, sort_keys=True, separators=(",", ":")),
               event_onset_frame=(row["event_frame_interval_half_open"] or [None, None])[0],
               event_offset_frame=(row["event_frame_interval_half_open"] or [None, None])[1])
    fields = ("T1_nominal_m", "T1_measured_m", "T1_m_absolute_error", "T1_nominal_SF_cycles_per_aperture",
              "T1_measured_SF_cycles_per_aperture", "T1_SF_absolute_error", "T1_nominal_TF_Hz",
              "T1_measured_TF_Hz", "T1_TF_absolute_error_Hz", "T1_measured_direction",
              "T2_nominal_delta_linear_Y", "T2_measured_core_increment_linear_Y",
              "T2_core_max_absolute_error_linear_Y", "T2_alpha_increment_linear_Y",
              "T2_alpha_absolute_error_linear_Y", "T2_changed_outside_target_pixels",
              "T3_nominal_angle_degrees")
    out.update({k: None for k in fields})
    for side in ("LEFT", "RIGHT"):
        for field in ("nominal_signed_deg", "PCA_deg", "PCA_absolute_error_deg",
                      "boundary_deg", "boundary_absolute_error_deg"):
            out["T3_" + side + "_" + field] = None
    if row["task"] == "T1":
        est, err = m["estimated"], m["errors"]
        out.update(T1_nominal_m=p["modulation_depth"], T1_measured_m=est["modulation_depth"],
                   T1_m_absolute_error=err["modulation_depth_absolute"],
                   T1_nominal_SF_cycles_per_aperture=p["sf_cycles_per_aperture"],
                   T1_measured_SF_cycles_per_aperture=est["sf_cycles_per_aperture"],
                   T1_SF_absolute_error=err["sf_cycles_per_aperture"], T1_nominal_TF_Hz=p["tf_hz"],
                   T1_measured_TF_Hz=est["temporal_frequency_hz"],
                   T1_TF_absolute_error_Hz=err["temporal_frequency_hz"],
                   T1_measured_direction=est["direction"])
    elif row["task"] == "T2":
        out.update(T2_nominal_delta_linear_Y=p["delta_Y"],
                   T2_measured_core_increment_linear_Y=m["core_increment"],
                   T2_core_max_absolute_error_linear_Y=m["core_max_error"],
                   T2_alpha_increment_linear_Y=m["alpha_regression_increment"],
                   T2_alpha_absolute_error_linear_Y=m["alpha_regression_error"],
                   T2_changed_outside_target_pixels=m["changed_outside_target"])
    else:
        out["T3_nominal_angle_degrees"] = p["angle_deg"]
        _require(len(m["axes"]) == 2 and {a["side"] for a in m["axes"]} == {"LEFT", "RIGHT"},
                 "Missing or repeated T3 axis side")
        for axis in m["axes"]:
            prefix = "T3_" + axis["side"] + "_"
            for target, source in (("nominal_signed_deg", "nominal_deg"), ("PCA_deg", "PCA_deg"),
                                   ("PCA_absolute_error_deg", "PCA_error_deg"),
                                   ("boundary_deg", "halfplane_deg"),
                                   ("boundary_absolute_error_deg", "halfplane_error_deg")):
                out[prefix + target] = axis[source]
    return out


def _new_bytes(path, data):
    path = Path(path)
    if path.exists():
        _require(path.read_bytes() == data, "Existing scientific export differs; choose a new path")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + str(os.getpid()) + ".partial")
    created_identity = None
    try:
        with temp.open("xb") as stream:
            info = os.fstat(stream.fileno())
            created_identity = (info.st_dev, info.st_ino)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        if created_identity is not None and temp.exists():
            info = temp.lstat()
            if (info.st_dev, info.st_ino) == created_identity:
                temp.unlink()


def export_source_qa(index, audit_path, jsonl_path, csv_path):
    _require(Path(jsonl_path).absolute() != Path(csv_path).absolute(), "QA export paths must differ")
    with Path(audit_path).open() as stream:
        rows = public_source_qa(index, (json.loads(line) for line in stream if line.strip()))
    data = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                    + b"\n" for r in rows)
    flat = [qa_csv_row(row) for row in rows]
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=list(flat[0]))
    writer.writeheader()
    writer.writerows(flat)
    _new_bytes(jsonl_path, data)
    _new_bytes(csv_path, text.getvalue().encode())
    return {"status": "PUBLIC_SOURCE_QA_EXPORTED", "IDs": len(rows), "counts": index["counts"],
            "index_id": index["index_id"], "original_audit_journal_sha256": file_sha(audit_path),
            "jsonl_sha256": file_sha(jsonl_path), "csv_sha256": file_sha(csv_path),
            "new_source_measurements": 0, "video_decodes": 0, "model_processor_calls": 0,
            "GPU_model_calls": 0, "uploads": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("index", help="Export portable metadata only; no media decoding/upload")
    build.add_argument("--catalog", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    qa = sub.add_parser("qa", help="Export per-ID nominal/measured source QA; no remeasurement")
    qa.add_argument("--index", type=Path, required=True)
    qa.add_argument("--audit", type=Path, required=True)
    qa.add_argument("--jsonl", type=Path, required=True)
    qa.add_argument("--csv", type=Path, required=True)
    read = sub.add_parser("read", help="Decode and verify one downloaded/local RGB16 master")
    read.add_argument("--index", type=Path, required=True)
    read.add_argument("--root", type=Path, required=True)
    read.add_argument("--id", required=True)
    fetch = sub.add_parser("fetch", help="Read one public Hub sample at a pinned commit; no upload")
    fetch.add_argument("--repo", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--id", required=True)
    fetch.add_argument("--cache-dir", type=Path)
    fetch.add_argument("--local-files-only", action="store_true")
    for command in (read, fetch):
        command.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.command == "index":
        index = index_from_catalog(json.loads(args.catalog.read_text()))
        if args.output.exists():
            _require(json.loads(args.output.read_text()) == index, "Existing index differs")
        else:
            atomic_json(args.output, index)
        result = {"status": "CPU_METADATA_EXPORTED", "index_id": index["index_id"],
                  "counts": index["counts"], "index_sha256": file_sha(args.output),
                  "video_decodes": 0, "GPU_model_calls": 0, "uploads": 0}
    elif args.command == "qa":
        result = export_source_qa(load_index(args.index), args.audit, args.jsonl, args.csv)
    else:
        if args.command == "read":
            _, _, result = read_sample(load_index(args.index), args.id, args.root)
        else:
            _, _, result = fetch_sample(args.repo, args.revision, args.id, args.cache_dir,
                                        local_files_only=args.local_files_only)
        if args.receipt:
            atomic_json(args.receipt, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
