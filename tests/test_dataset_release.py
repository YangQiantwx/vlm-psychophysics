"""Source-release gates and portable folder-download boundaries."""
import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from classic_precision.dataset_release import canonical, finalize, project_rows, public_value, publish_source_releases
from classic_precision.download_dataset import download, relative_folder


def test_source_projection_drops_model_fields_and_preserves_science():
    source = {"sample_id":"t2_existing", "task":"T2", "truth":"LEFT",
              "video_sha256":"a"*64, "nominal_parameters":{"delta_Y":.06},
              "prompt":"Which side brightened?", "accuracy":1, "prediction":"LEFT"}
    mapping = {"domain":"classic", "task":"T2", "legacy_sample_id":"t2_existing",
               "proposed_sample_id":"t2_existing", "legacy_path":"T2/old.mkv",
               "proposed_public_path":"T2/readable.mkv", "video_sha256":"a"*64,
               "decoded_stream_sha256":"b"*64, "new_identity_full_sha256":"",
               "source_record_sha256":hashlib.sha256(canonical(source).encode()).hexdigest(),
               "truth":"LEFT", "condition_key":'["T2","0.06",18,30,"LEFT"]',
               "primary_T123_scope":"True", "background_id":"", "object_id":"",
               "parent_sample_id":"", "physical_upper_end_direction":""}
    row = project_rows([mapping], {source["sample_id"]:source})[0]
    assert row["parameters"] == source["nominal_parameters"]
    assert row["prompt"] == source["prompt"] and row["truth"] == source["truth"]
    assert "accuracy" not in row and "prediction" not in row
    with pytest.raises(ValueError, match="changed"):
        project_rows([mapping], {source["sample_id"]:{**source,"truth":"RIGHT"}})


@pytest.mark.parametrize("path", ["../T1", "/T1", "T1/*", "results", "T4", "T1\\x"])
def test_folder_rejects_out_of_scope_or_unsafe_paths(path):
    with pytest.raises(ValueError):
        relative_folder(path)


def test_downloader_pins_commit_and_copies_entire_folder_without_results(tmp_path, monkeypatch):
    media = b"verified media bytes"
    row = {"sample_id":"one", "path":"T3/one.mkv", "video_sha256":hashlib.sha256(media).hexdigest()}
    remote = {"T3/one.mkv":media, "T3/two.mkv":b"two", "T2/other.mkv":b"other",
              "catalog.jsonl":(json.dumps(row)+"\n").encode(), "README.md":b"Source dataset",
              "results/private.csv":b"not source data"}
    calls = []
    class Api:
        def repo_info(self, *args, **kwargs):
            assert kwargs["revision"] == "tag-requested"
            return SimpleNamespace(sha="f"*40, siblings=[SimpleNamespace(rfilename=p) for p in remote])
    def snapshot(**kwargs):
        calls.append(kwargs)
        for p in kwargs["allow_patterns"]:
            path = Path(kwargs["local_dir"])/p
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(remote[p])
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(HfApi=Api,snapshot_download=snapshot))
    report = download("classic",None,"T3","tag-requested",tmp_path,4)[0]
    assert calls[0]["revision"] == "f"*40 and calls[0]["max_workers"] == 4
    assert {"T3/one.mkv","T3/two.mkv"} <= set(calls[0]["allow_patterns"])
    assert not any(p.startswith(("results/","T2/")) for p in calls[0]["allow_patterns"])
    assert report["videos_sha256_verified"] == 1


def test_finalize_requires_exact_measured_coverage_before_writing(tmp_path):
    repo = tmp_path/"repo";repo.mkdir()
    row = {"domain":"natural", "sample_id":"n_one", "legacy_sample_id":"n_one",
           "task":"T3", "video_sha256":"a"*64,"path":"T3/readable.mp4",
           "source_record_sha256":"c"*64,"decoded_stream_sha256":"d"*64,"truth":"LEFT"}
    (repo/"catalog.jsonl").write_text(json.dumps(row)+"\n")
    plan = {"publishable":False,"repositories":[{"staging_folder":"repo","parent_commit":"b"*40,"upload_files":[]}]}
    receipt = {**row,"status":"PASS","source_sha256":"a"*64,"code_sha256":"e"*64,
               "comparison":{"filename":True,"GT":True,"condition":True,"source_aliases":True},
               "measured":{"left_axis_deg":0,"right_axis_deg":.5,"vertical_side":"LEFT"},
               "method":"decoded RGB-bound alpha geometry","tolerances":{"axis_error_deg":.01}}
    qa = tmp_path/"qa.jsonl"
    qa.write_text(json.dumps({**receipt,"status":"FAIL"})+"\n")
    with pytest.raises(ValueError,match="failing"):
        finalize(tmp_path,plan,qa)
    assert not (repo/"validation").exists()
    for changed in ({"path":"T3/wrong-name.mp4"},{"source_record_sha256":"0"*64},
                    {"decoded_stream_sha256":"0"*64},
                    {"measured":{**receipt["measured"],"vertical_side":"RIGHT"}},
                    {"comparison":{**receipt["comparison"],"condition":False}}):
        qa.write_text(json.dumps({**receipt,**changed})+"\n")
        with pytest.raises(ValueError):
            finalize(tmp_path,plan,qa)
        assert not (repo/"validation").exists()
    qa.write_text(json.dumps(receipt)+"\n"+json.dumps({**receipt,"sample_id":"extra"})+"\n")
    with pytest.raises(ValueError,match="unexpected"):
        finalize(tmp_path,plan,qa)
    assert not (repo/"validation").exists()
    qa.write_text(json.dumps(receipt)+"\n")
    assert finalize(tmp_path,plan,qa)["publishable"]
    assert json.loads((repo/"release.json").read_text())["source_videos"] == 1


def test_public_nested_metadata_rejects_private_host_paths():
    with pytest.raises(ValueError,match="private workspace"):
        public_value({"source":{"path":"/Users/someone/private-source.png"}})


def test_natural_nested_folder_keeps_actual_repository_prefix(tmp_path, monkeypatch):
    calls = []
    class Api:
        def repo_info(self, repo_id, **kwargs):
            assert repo_id.endswith("natural-t3")
            return SimpleNamespace(sha="f"*40,siblings=[SimpleNamespace(rfilename="T3/sub/a.mp4"),
                SimpleNamespace(rfilename="T3/other.mp4")])
    def snapshot(**kwargs):
        calls.append(kwargs)
        for p in kwargs["allow_patterns"]:
            target=Path(kwargs["local_dir"])/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b"video")
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(HfApi=Api,snapshot_download=snapshot))
    result=download("natural",None,"T3/sub","fixed",tmp_path)
    assert calls[0]["allow_patterns"] == ["T3/sub/a.mp4"]
    assert result[0]["folder"] == "T3/sub"


def test_publisher_refuses_incomplete_gate_tampered_files_and_changed_head(tmp_path):
    # Use real operation constructors, but an API with no permitted mutation.
    with pytest.raises(ValueError,match="blocked"):
        publish_source_releases(tmp_path,{"publishable":False},{})
    stage=tmp_path/"repo";stage.mkdir();file=stage/"README.md";file.write_bytes(b"source")
    repo={"repo_id":"owner/data","parent_commit":"a"*40,"staging_folder":"repo",
          "upload_files":["README.md"],"upload_file_identities":[{"path_in_repo":"README.md",
          "bytes":6,"sha256":hashlib.sha256(b"source").hexdigest()}],"copy_files":[],"delete_files":[]}
    plan={"publishable":True,"gate":"PASS_ALL_VIDEO_DECODED_CONTENT_ACCEPTANCE","repositories":[repo]}
    class Api:
        def repo_info(self,*args,**kwargs):return SimpleNamespace(sha="b"*40)
        def create_commit(self,**kwargs):raise AssertionError("No mutation was authorized by the checks")
    with pytest.raises(ValueError,match="HEAD changed"):
        publish_source_releases(tmp_path,plan,{},api=Api())
    file.write_bytes(b"tamper")
    with pytest.raises(ValueError,match="staging file changed"):
        publish_source_releases(tmp_path,plan,{},api=Api())
