from copy import deepcopy

import pytest

from classic_precision.dataset_names import filename, relabel_index
from classic_precision.manifest import digest


def motion():
    return {"task": "T1", "truth": "LEFT", "nominal_parameters": {
        "bank_start_index": 0, "phase_degrees": 7.5, "direction": "LEFT",
        "sf_cycles_per_aperture": 4, "modulation_depth": .6}}


def brightness():
    return {"task": "T2", "truth": "RIGHT", "nominal_parameters": {
        "onset_index": 15, "onset_frame": 93, "offset_frame": 103,
        "target_side": "RIGHT", "background_Y": .05, "delta_Y": .15}}


def orientation():
    return {"task": "T3", "truth": "LEFT", "nominal_parameters": {
        "onset_index": 0, "onset_frame": 18, "offset_frame": 48,
        "reference_side": "LEFT", "angle_deg": .7, "tilt_sign": -1}}


def test_fixed_phase_bank_and_not_directory_order():
    e = motion()
    assert filename(e) == "T1_left_sf4_contrast0.6_trial01.mkv"
    e["nominal_parameters"].update(bank_start_index=23, phase_degrees=352.5)
    assert filename(e).endswith("trial24.mkv")
    e["nominal_parameters"]["phase_degrees"] = 7
    with pytest.raises(ValueError):
        filename(e)


def test_brightness_duration_and_trial_have_separate_meanings():
    e = brightness()
    assert filename(e) == "T2_right_bg0.05_delta0.15_10frames_trial16.mkv"
    e["nominal_parameters"]["offset_frame"] = 123
    assert filename(e) == "T2_right_bg0.05_delta0.15_30frames_trial16.mkv"


def test_upper_end_direction_is_separate_from_vertical_side():
    e = orientation()
    assert filename(e) == "T3_verticalLeft_tiltRight_0.7deg_trial01.mkv"
    e["nominal_parameters"]["tilt_sign"] = 1
    assert filename(e) == "T3_verticalLeft_tiltLeft_0.7deg_trial01.mkv"
    e["nominal_parameters"]["reference_side"] = "RIGHT"
    e["truth"] = "RIGHT"
    assert filename(e) == "T3_verticalRight_tiltLeft_0.7deg_trial01.mkv"


def test_horizontal_does_not_invent_two_tilt_directions():
    e = orientation()
    e["nominal_parameters"].update(angle_deg=90, tilt_sign=1)
    assert filename(e) == "T3_verticalLeft_horizontal_90deg_trial01.mkv"
    e["nominal_parameters"]["tilt_sign"] = -1
    with pytest.raises(ValueError):
        filename(e)


@pytest.mark.parametrize("e", [motion(), brightness(), orientation()])
def test_names_cannot_disagree_with_answers(e):
    e = deepcopy(e)
    e["truth"] = "RIGHT" if e["truth"] == "LEFT" else "LEFT"
    with pytest.raises(ValueError):
        filename(e)


def mini_index():
    e = {**motion(), "sample_id": "stable-id", "video_relative": "legacy/video.mkv",
         "prompt": "LEFT or RIGHT?", "video_sha256": "a"*64, "stream_sha256": "b"*64,
         "width": 768, "height": 512, "frames": 180, "fps": 60}
    body = {"schema": "classic-precision-rgb16-index-v1", "counts": {"T1": 1},
            "items": {e["sample_id"]: e}}
    return {**body, "index_id": digest(body)}


@pytest.mark.parametrize("layout,prefix", [("tasks", "T1/"), ("flat", "videos/")])
def test_layout_changes_paths_and_preserves_source_identity(layout, prefix):
    old = mini_index()
    before = deepcopy(old)
    new, mapping = relabel_index(old, layout)
    assert old == before
    assert set(new["items"]) == {"stable-id"}
    assert new["items"]["stable-id"]["video_relative"] == prefix+"T1_left_sf4_contrast0.6_trial01.mkv"
    assert {k: v for k, v in old["items"]["stable-id"].items() if k != "video_relative"} == {
        k: v for k, v in new["items"]["stable-id"].items() if k != "video_relative"}
    assert mapping[0]["old_path"] == "legacy/video.mkv"


def test_two_ids_cannot_collide_after_readable_rename():
    old = mini_index()
    e = deepcopy(old["items"]["stable-id"])
    e.update(sample_id="second-id", video_relative="legacy/second.mkv")
    old["items"]["second-id"] = e
    old["counts"]["T1"] = 2
    old["index_id"] = digest({k: v for k, v in old.items() if k != "index_id"})
    with pytest.raises(ValueError, match="same readable filename"):
        relabel_index(old)
