from copy import deepcopy
import hashlib
import json

import pytest

from classic_precision.dataset_identity import (
    IdentityRegistry, identity_digest, restore_record, source_sample_id,
)
from classic_precision import dataset
from classic_precision.manifest import digest
from classic_precision import cosmos_x
from classic_precision.dataset_names import relabel_index


def binding(payload=b"source", **changes):
    sha = hashlib.sha256(payload).hexdigest()
    full = identity_digest("classic", "T1", sha)
    return {"domain": "classic", "task": "T1", "legacy_sample_id": "t1rgb16_i00_sf4_m0p6_left",
        "proposed_sample_id": "t1_" + full[:20], "new_identity_full_sha256": full,
        "legacy_path": "frozen/videos/old.mkv", "proposed_public_path": "T1/readable.mkv",
        "video_sha256": sha, **changes}


def test_namespace_is_frozen_full_sha_not_filename_or_host():
    row = binding()
    registry = IdentityRegistry.from_rows([row])
    expected = "t1_" + hashlib.sha256(json.dumps({"schema": "sample-identity-v1",
        "domain": "classic", "task": "T1", "video_sha256": row["video_sha256"]},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:20]
    assert registry.canonical_id(row["legacy_sample_id"], domain="classic") == expected
    changed = {**row, "legacy_path": "other/old.mkv", "proposed_public_path": "T1/other.mkv"}
    assert IdentityRegistry.from_rows([changed]).bindings[0].canonical_id == expected


def test_aliases_resolve_same_exact_bytes_and_fallback(tmp_path):
    row = binding()
    registry = IdentityRegistry.from_rows([row])
    old = tmp_path / row["legacy_path"]
    old.parent.mkdir(parents=True)
    old.write_bytes(b"source")
    for key in (row["legacy_sample_id"], row["proposed_sample_id"], row["legacy_path"], row["proposed_public_path"]):
        assert registry.resolve_path(key, tmp_path, domain="classic", verify_sha=True) == old
    new = tmp_path / row["proposed_public_path"]
    new.parent.mkdir(parents=True)
    new.write_bytes(old.read_bytes())
    assert registry.resolve_path(row["legacy_path"], tmp_path, domain="classic", verify_sha=True) == new
    new.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="SHA256"):
        registry.resolve_path(row["legacy_sample_id"], tmp_path, domain="classic", verify_sha=True)


def test_no_parent_alias_for_real_source_replacement():
    row = binding(domain="natural", task="T3", proposed_sample_id="current-spoon", legacy_sample_id="current-spoon",
        parent_sample_id="old-spoon", new_identity_full_sha256="")
    registry = IdentityRegistry.from_rows([row])
    assert registry.resolve("current-spoon", domain="natural").legacy_sample_id == "current-spoon"
    with pytest.raises(KeyError):
        registry.resolve("old-spoon", domain="natural")


def test_inventory_bound_public_legacy_paths_are_aliases(tmp_path):
    row = binding(legacy_public_path="videos/published-old.mkv", alias_paths_json='["archive/other-old.mkv"]')
    registry = IdentityRegistry.from_rows([row])
    canonical = tmp_path / row["proposed_public_path"]
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"source")
    for path in (row["legacy_public_path"], "archive/other-old.mkv"):
        assert registry.resolve_path(path, tmp_path, domain="classic", verify_sha=True) == canonical
    exported = registry.rows()
    assert IdentityRegistry.from_rows(exported).resolve(row["legacy_public_path"], domain="classic") == registry.bindings[0]


def test_revision_required_when_media_changes_under_same_legacy_name():
    first = binding()
    second = binding(b"revision2")
    registry = IdentityRegistry.from_rows([{**first, "dataset_revision": "v1"}, {**second, "dataset_revision": "v2"}])
    with pytest.raises(KeyError, match="ambiguous"):
        registry.resolve(first["legacy_sample_id"], domain="classic")
    assert registry.resolve(first["legacy_sample_id"], domain="classic", dataset_revision="v2").video_sha256 == second["video_sha256"]


@pytest.mark.parametrize("changes", [
    {"proposed_sample_id": "t1_" + "0" * 20}, {"new_identity_full_sha256": "0" * 64},
    {"proposed_public_path": "../escape.mkv"}, {"legacy_path": "/absolute.mkv"},
])
def test_invalid_id_digest_path_rejected(changes):
    with pytest.raises(ValueError):
        IdentityRegistry.from_rows([binding(**changes)])


def test_duplicate_short_id_and_alias_rejected():
    first, second = binding(), binding(b"other")
    with pytest.raises(ValueError, match="Ambiguous"):
        IdentityRegistry.from_rows([first, second])
    with pytest.raises(ValueError, match="Duplicate canonical"):
        IdentityRegistry.from_rows([first, first])


def test_projection_preserves_full_answer_failure_tokens_timing_config():
    row = binding()
    registry = IdentityRegistry.from_rows([row])
    record = {"sample_id": row["legacy_sample_id"], "raw_answer": "RIGHT\n", "answer": "RIGHT",
        "truth": "LEFT", "output_token_ids": [1, 2, 3], "finish_reason": "length",
        "stop_reason": None, "failure": {"kind": "truncated"}, "eligible": False,
        "timing": {"sample_id": row["legacy_sample_id"], "seconds": 2.4},
        "generation": {"seed": 19}, "protocol": "frozen", "raw_digest": "a" * 64}
    before = deepcopy(record)
    projected = registry.project_record(record, domain="classic")
    assert record == before
    assert restore_record(projected) == before
    assert projected["timing"]["sample_id"] == row["legacy_sample_id"]
    assert source_sample_id(projected) == row["legacy_sample_id"]
    projected["output_token_ids"].append(4)
    with pytest.raises(ValueError, match="scientific record"):
        restore_record(projected)


def test_selection_seed_and_dedup_keep_frozen_order():
    first, second = binding(), binding(b"other", legacy_sample_id="other", proposed_public_path="T1/other.mkv", legacy_path="frozen/other.mkv")
    registry = IdentityRegistry.from_rows([first, second])
    rows = [{"sample_id": r["legacy_sample_id"], "seed": i, "prompt": "visual-only"} for i, r in enumerate((first, second))]
    old = registry.select_rows(rows, [second["legacy_sample_id"], first["legacy_sample_id"]], domain="classic")
    new = registry.select_rows(rows, [second["proposed_sample_id"], first["proposed_sample_id"]], domain="classic")
    assert new == old == rows
    assert [hashlib.sha256(source_sample_id(r).encode()).hexdigest() for r in new] == [hashlib.sha256(r["sample_id"].encode()).hexdigest() for r in rows]
    with pytest.raises(ValueError, match="same source twice"):
        registry.select_rows(rows, [first["legacy_sample_id"], first["proposed_sample_id"]], domain="classic")


def test_old_and_projected_index_ids_paths_read_same_decoded_media(tmp_path, monkeypatch):
    row = binding()
    registry = IdentityRegistry.from_rows([row])
    entry = {"sample_id": row["legacy_sample_id"], "task": "T1", "truth": "LEFT", "prompt": "LEFT or RIGHT?",
        "video_relative": row["legacy_path"], "video_sha256": row["video_sha256"], "stream_sha256": "b" * 64,
        "nominal_parameters": {"seed": 13}, "width": 1, "height": 1, "frames": 2, "fps": 60}
    body = {"schema": dataset.SCHEMA, "items": {entry["sample_id"]: entry}, "counts": {"T1": 1}}
    old = {**body, "index_id": digest(body)}
    new = registry.project_index(old)
    assert new["items"][row["proposed_sample_id"]]["prompt"] == entry["prompt"]
    path = tmp_path / row["legacy_path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"source")
    monkeypatch.setattr(dataset, "decode_video", lambda path, e: (path.read_bytes(), e["prompt"], e["nominal_parameters"]))
    for index in (old, new):
        for value in (row["legacy_sample_id"], row["proposed_sample_id"], row["legacy_path"]):
            assert dataset.read_sample(index, value, tmp_path) == (b"source", "LEFT or RIGHT?", {"seed": 13})


def test_standalone_cosmos_resolves_new_ids_before_frozen_selection():
    row = binding()
    entry = {"sample_id": row["legacy_sample_id"], "task": "T1", "video_relative": row["legacy_path"],
             "video_sha256": row["video_sha256"], "prompt": "visual-only", "seed": 0}
    selected = {row["legacy_sample_id"]: entry}
    for value in (row["legacy_sample_id"], row["proposed_sample_id"], row["legacy_path"]):
        assert cosmos_x.selected_entries(selected, [value]) == [(row["legacy_sample_id"], entry)]
    with pytest.raises(ValueError, match="Duplicate requested source"):
        cosmos_x.selected_entries(selected, [row["legacy_sample_id"], row["proposed_sample_id"]])


def test_registry_cannot_silently_override_requested_flat_layout():
    row = binding()
    entry = {"sample_id": row["legacy_sample_id"], "task": "T1", "truth": "LEFT", "prompt": "LEFT or RIGHT?",
        "video_relative": row["legacy_path"], "video_sha256": row["video_sha256"], "stream_sha256": "b" * 64,
        "nominal_parameters": {}, "width": 1, "height": 1, "frames": 2, "fps": 60}
    body = {"schema": dataset.SCHEMA, "items": {entry["sample_id"]: entry}, "counts": {"T1": 1}}
    index = {**body, "index_id": digest(body)}
    with pytest.raises(ValueError, match="canonical task-folder"):
        relabel_index(index, "flat", IdentityRegistry.from_rows([row]))
