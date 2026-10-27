from __future__ import annotations

import copy
import hashlib
import json
import sys
import types
from fractions import Fraction

import av
import numpy as np
import pytest

from classic_precision import dataset as d
from classic_precision.manifest import digest


@pytest.fixture
def master(tmp_path):
    # Every uint16 code and distinct RGB channels catch RGB/BGR and 8-bit conversion.
    plane = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    frame = np.stack((plane, np.flipud(plane), np.fliplr(plane)), axis=-1)
    codes = np.stack((frame, np.roll(frame, 3, axis=1)))
    path = tmp_path / "master.mkv"
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=60)
        stream.width, stream.height, stream.pix_fmt = 256, 256, "gbrp16le"
        stream.codec_context.thread_count = 1
        for i, rgb in enumerate(codes):
            f = av.VideoFrame.from_ndarray(rgb, format="rgb48le")
            f.pts, f.time_base = i, Fraction(1, 60)
            for packet in stream.encode(f):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    entry = {
        "sample_id": "fixture", "task": "T2", "truth": "LEFT", "prompt": "LEFT or RIGHT",
        "width": 256, "height": 256, "frames": 2, "fps": 60,
        "video_relative": "master.mkv", "video_sha256": d.file_sha(path),
        "stream_sha256": hashlib.sha256(codes.astype("<u2").tobytes()).hexdigest(),
        "nominal_parameters": {"delta_Y": .1},
    }
    return path, codes, entry


def make_index(entry):
    body = {"schema": d.SCHEMA, "counts": {entry["task"]: 1},
            "items": {entry["sample_id"]: entry}}
    return {**body, "index_id": digest(body)}


def test_full_uint16_codebook_color_order_and_timestamps(master):
    path, expected, entry = master
    actual, pts, receipt = d.decode_video(path, entry)
    assert actual.dtype == np.uint16
    assert np.array_equal(actual, expected)
    assert len(np.unique(actual)) == 65536
    assert pts[0] == 0 and abs(pts[1] - 1 / 60) <= .001
    assert receipt["status"] == "COMPLETE_PASS"
    assert receipt["model_processor_calls"] == receipt["GPU_model_calls"] == 0
    assert receipt["resize"] is receipt["crop"] is False
    assert receipt["padding_bottom_rows"] == 0


def test_fp32_mapping_keeps_every_uint16_code():
    codes = np.arange(65536, dtype=np.uint16)
    actual = d.rgb16_to_fp32(codes)
    assert actual.dtype == np.float32
    assert np.array_equal(np.rint(actual * (65535. / 255.)).astype(np.uint16), codes)
    assert np.array_equal(actual, codes.astype(np.float32) * (255. / 65535.))
    with pytest.raises(ValueError, match="original uint16"):
        d.rgb16_to_fp32(codes.astype(np.uint8))


def test_wrong_encoded_hash_rejected_before_decode(master, monkeypatch):
    path, _, entry = master
    entry = {**entry, "video_sha256": "0" * 64}
    monkeypatch.setattr(av, "open", lambda *a, **k: pytest.fail("Decoded before SHA acceptance"))
    with pytest.raises(ValueError, match="Encoded video SHA256"):
        d.decode_video(path, entry)


def test_wrong_decoded_stream_hash_rejected(master):
    path, _, entry = master
    with pytest.raises(ValueError, match="Decoded RGB stream SHA256"):
        d.decode_video(path, {**entry, "stream_sha256": "0" * 64})


@pytest.mark.parametrize("frames,message", [(1, "extra frames"), (3, "missing frames")])
def test_missing_or_extra_frames_rejected(master, frames, message):
    path, _, entry = master
    with pytest.raises(ValueError, match=message):
        d.decode_video(path, {**entry, "frames": frames})


def test_incorrect_timebase_not_silently_resampled(master):
    path, _, entry = master
    with pytest.raises(ValueError, match="timestamp differs"):
        d.decode_video(path, {**entry, "fps": 30})


def test_8bit_video_not_accepted_as_scientific_master(tmp_path, master):
    _, _, entry = master
    p = tmp_path / "8bit.mkv"
    with av.open(str(p), "w", format="matroska") as out:
        stream = out.add_stream("ffv1", rate=60)
        stream.width, stream.height, stream.pix_fmt = 256, 256, "bgr0"
        for packet in stream.encode(av.VideoFrame.from_ndarray(np.zeros((256, 256, 3), np.uint8), format="rgb24")):
            out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with pytest.raises(ValueError, match="FFV1 gbrp16le"):
        d.decode_video(p, {**entry, "video_sha256": d.file_sha(p)})


def test_hub_cache_symlink_target_supported(tmp_path, master):
    path, expected, entry = master
    cache = tmp_path / "snapshot"
    cache.mkdir()
    (cache / "master.mkv").symlink_to(path)
    result, _, _ = d.read_sample(make_index(entry), "fixture", cache)
    assert np.array_equal(result, expected)


def test_index_tamper_rejected(master):
    _, _, entry = master
    index = make_index(entry)
    index["items"]["fixture"]["truth"] = "RIGHT"
    with pytest.raises(ValueError, match="index identity"):
        d.validate_index(index)


@pytest.mark.parametrize("path", ["../secret", "/absolute.mkv", "a/../../video", "a\\b.mkv"])
def test_index_rejects_nonrelative_paths(master, path):
    _, _, entry = master
    with pytest.raises(ValueError, match="relative"):
        d.validate_index(make_index({**entry, "video_relative": path}))


def test_duplicate_video_paths_and_wrong_counts_rejected(master):
    _, _, entry = master
    body = {"schema": d.SCHEMA, "counts": {"T2": 2},
            "items": {"fixture": entry, "second": {**entry, "sample_id": "second"}}}
    with pytest.raises(ValueError, match="same video path"):
        d.validate_index({**body, "index_id": digest(body)})
    index = make_index(entry)
    index["counts"] = {"T2": 9}
    index["index_id"] = digest({k: v for k, v in index.items() if k != "index_id"})
    with pytest.raises(ValueError, match="denominators"):
        d.validate_index(index)


def test_catalog_export_preserves_task_parameters_identity_without_decode(monkeypatch):
    items = {}
    for task in ("T1", "T2", "T3"):
        sid = task + "_sample"
        params = {"m": .1} if task == "T1" else {"angle_deg": 1. if task == "T3" else .1}
        row = {"sample_id": sid, "prompt": "LEFT or RIGHT", "truth": "RIGHT",
               "source_parameters": params}
        ready = {"sample_id": sid, "video_relative": "videos/" + sid + ".mkv", "video_sha256": "a" * 64}
        items[sid] = {"sample_id": sid, "task": task, "row": row, "ready": ready,
                      "row_sha256": digest(row), "ready_sha256": digest(ready),
                      "source_family": "classic", "source_stream_sha256": "b" * 64,
                      "source_audit_identity": "c" * 64}
    catalog = {"items": items, "counts": {"T1": 1, "T2": 1, "T3": 1}}
    monkeypatch.setattr(av, "open", lambda *a, **k: pytest.fail("Catalog export decoded a video"))
    index = d.index_from_catalog(catalog)
    for sid, x in items.items():
        assert index["items"][sid]["nominal_parameters"] == x["row"]["source_parameters"]
        assert index["items"][sid]["truth"] == "RIGHT"
        assert index["items"][sid]["source_audit_identity"] == "c" * 64
        assert "source_audit_row_sha256" not in index["items"][sid]
    bad = copy.deepcopy(catalog)
    bad["items"]["T3_sample"]["row"]["source_parameters"]["angle_deg"] = 2.
    with pytest.raises(ValueError, match="content differs"):
        d.index_from_catalog(bad)


def test_fetch_pins_one_commit_two_files_and_disables_auth(master, tmp_path, monkeypatch):
    path, expected, entry = master
    index_path = tmp_path / "dataset-index.json"
    index_path.write_text(json.dumps(make_index(entry)))
    calls = []
    def fake_download(**kwargs):
        calls.append(kwargs)
        return str(index_path if kwargs["filename"] == "dataset-index.json" else path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    result, _, _ = d.fetch_sample("example/dataset", "1" * 40, "fixture")
    assert np.array_equal(result, expected)
    assert [x["filename"] for x in calls] == ["dataset-index.json", "master.mkv"]
    assert all(x["repo_type"] == "dataset" and x["revision"] == "1" * 40
               and x["token"] is False for x in calls)
    with pytest.raises(ValueError, match="immutable"):
        d.fetch_sample("example/dataset", "main", "fixture")
    assert len(calls) == 2


def test_read_cli_reports_verified_source_without_exporting_frames(master, tmp_path, capsys):
    path, _, entry = master
    index_path = tmp_path / "dataset-index.json"
    index_path.write_text(json.dumps(make_index(entry)))
    receipt_path = tmp_path / "receipt.json"
    d.main(["read", "--index", str(index_path), "--root", str(path.parent), "--id", "fixture",
            "--receipt", str(receipt_path)])
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(receipt_path.read_text())
    assert result["decoded_frames"] == 2 and result["bits_per_channel"] == 16


def qa_pair(task="T2"):
    params = {"delta_Y": .1}
    measurements = {"core_increment": .09999, "core_max_error": .00001,
                    "alpha_regression_increment": .1, "alpha_regression_error": 0.,
                    "changed_outside_target": 0}
    if task == "T1":
        params = {"modulation_depth": .1, "sf_cycles_per_aperture": 1., "tf_hz": 5.}
        measurements = {"estimated": {"modulation_depth": .100001, "sf_cycles_per_aperture": 1.,
                                       "temporal_frequency_hz": 5., "direction": "LEFT"},
                        "errors": {"modulation_depth_absolute": .000001,
                                   "sf_cycles_per_aperture": 0., "temporal_frequency_hz": 0.}}
    elif task == "T3":
        params = {"angle_deg": 1.}
        measurements = {"axes": [dict(side=side, nominal_deg=angle, PCA_deg=angle + .00001,
                                      PCA_error_deg=.00001, halfplane_deg=angle,
                                      halfplane_error_deg=0.) for side, angle in (("LEFT", 1.), ("RIGHT", 0.))]}
    entry = {"sample_id": "fixture", "task": task, "truth": "LEFT", "prompt": "LEFT or RIGHT",
             "width": 768, "height": 512, "frames": 180, "fps": 60,
             "video_relative": "family/master.mkv", "video_sha256": "a" * 64,
             "stream_sha256": "b" * 64, "nominal_parameters": params,
             "source_audit_identity": "c" * 64, "source_row_sha256": "d" * 64,
             "source_ready_sha256": "e" * 64}
    qa = {"sample_id": "fixture", "task": task, "truth": "LEFT", "status": "PASS",
          "video_sha256": "a" * 64, "decoded_stream_sha256": "b" * 64,
          "row_sha256": "d" * 64, "ready_sha256": "e" * 64, "identity": "c" * 64,
          "parameters": params, "independent_pixel_measurements": measurements,
          "decoded_frames": 180, "dimensions_wh": [768, 512], "fps": 60,
          "codec": "ffv1", "pixel_format": "gbrp16le", "max_code_error": 0,
          "every_decoded_RGB_code_compared": True, "max_PTS_error_s": .000333,
          "measurement_scope": "shared unique-state pixel measurements bound to all frame codes",
          "event_frame_interval_half_open": [18, 48],
          "source_stat": {"ino": 123, "private_host_path": "/home/private/not-for-release"}}
    return make_index(entry), qa


@pytest.mark.parametrize("task", ["T1", "T2", "T3"])
def test_public_qa_has_real_perrow_hash_nominal_measured_and_no_host_metadata(task):
    index, qa = qa_pair(task)
    row, = d.public_source_qa(index, [qa])
    assert row["source_audit_identity"] == qa["identity"]
    assert row["source_audit_row_sha256"] == digest(qa) != qa["identity"]
    assert row["nominal_parameters"] == qa["parameters"]
    assert row["independent_pixel_measurements"] == qa["independent_pixel_measurements"]
    assert "source_stat" not in row and "/home/private" not in json.dumps(row)
    flat = d.qa_csv_row(row)
    assert json.loads(flat["nominal_parameters_json"]) == qa["parameters"]
    assert flat["event_onset_frame"] == 18 and flat["event_offset_frame"] == 48
    if task == "T1":
        assert flat["T1_nominal_m"] == .1 and flat["T1_measured_m"] == .100001
    elif task == "T2":
        assert flat["T2_nominal_delta_linear_Y"] == .1
        assert flat["T2_measured_core_increment_linear_Y"] == .09999
    else:
        assert flat["T3_LEFT_nominal_signed_deg"] == 1.
        assert flat["T3_RIGHT_PCA_absolute_error_deg"] == .00001


@pytest.mark.parametrize("field,value", [("truth", "RIGHT"), ("video_sha256", "f" * 64),
                                        ("decoded_stream_sha256", "f" * 64),
                                        ("identity", "f" * 64), ("parameters", {"delta_Y": .2}),
                                        ("decoded_frames", 179), ("status", "FAILED")])
def test_public_qa_rejects_source_binding_or_incomplete_evidence(field, value):
    index, qa = qa_pair()
    qa[field] = value
    with pytest.raises(ValueError):
        d.public_source_qa(index, [qa])


def test_public_qa_rejects_duplicate_or_missing_ids():
    index, qa = qa_pair()
    with pytest.raises(ValueError, match="Repeated"):
        d.public_source_qa(index, [qa, qa])
    with pytest.raises(ValueError, match="ID sets"):
        d.public_source_qa(index, [])


def test_legacy_draft_global_identity_not_reported_as_row_sha():
    index, _ = qa_pair()
    entry = index["items"]["fixture"]
    entry["source_audit_row_sha256"] = entry.pop("source_audit_identity")
    index["index_id"] = digest({k: v for k, v in index.items() if k != "index_id"})
    with pytest.raises(ValueError, match="Legacy draft"):
        d.validate_index(index)


def test_qa_export_writes_csv_jsonl_and_preserves_existing_output(tmp_path):
    import csv
    index, qa = qa_pair()
    original = tmp_path / "original.jsonl"
    original.write_text(json.dumps(qa) + "\n")
    public, flat = tmp_path / "public.jsonl", tmp_path / "public.csv"
    receipt = d.export_source_qa(index, original, public, flat)
    row = json.loads(public.read_text())
    with flat.open(newline="") as f:
        csvrow, = list(csv.DictReader(f))
    assert receipt["IDs"] == 1 and receipt["video_decodes"] == 0
    assert csvrow["source_audit_row_sha256"] == row["source_audit_row_sha256"] == digest(qa)
    assert csvrow["T2_nominal_delta_linear_Y"] == "0.1"
    public.write_text("preserved different earlier export")
    with pytest.raises(ValueError, match="Existing scientific export differs"):
        d.export_source_qa(index, original, public, flat)
    assert public.read_text() == "preserved different earlier export"


def test_preexisting_partial_is_not_deleted_by_export(tmp_path):
    import os
    target = tmp_path / "out.jsonl"
    partial = target.with_name(target.name + "." + str(os.getpid()) + ".partial")
    partial.write_text("preserve interrupted writer")
    with pytest.raises(FileExistsError):
        d._new_bytes(target, b"new")
    assert partial.read_text() == "preserve interrupted writer"
