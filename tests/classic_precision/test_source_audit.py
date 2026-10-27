"""Failure probes for the audit, independent of the large remote dataset."""
from pathlib import Path

import numpy as np
import pytest

from classic_precision.source_audit import check_frame, fp32_codebook, pilot_items, safe_child, summarize


@pytest.fixture
def frame():
    return np.full((512, 768, 3), 30000, dtype=np.uint16)


def test_all_codes_survive_fp32_entry():
    assert fp32_codebook()['codes'] == 65536
    assert fp32_codebook()['roundtrip_max_code_error'] == 0


def test_correct_frame_and_matroska_timestamp(frame):
    assert check_frame(frame, frame[..., 0], 1, .017, 0.) == pytest.approx(1/3000)


@pytest.mark.parametrize('channel', [0, 1, 2])
def test_single_code_corruption_in_any_channel_is_rejected(frame, channel):
    expected = frame[..., 0].copy()
    frame[0, 0, channel] += 1
    with pytest.raises(ValueError, match='pixel mismatch'):
        check_frame(frame, expected, 0, 0., None)


def test_timing_and_frame_count_cannot_be_hidden(frame):
    for index, pts, previous in [(1, .018, 0.), (1, 0., 0.), (180, 3., 2.983), (0, None, None)]:
        with pytest.raises(ValueError):
            check_frame(frame, frame, index, pts, previous)


def test_8bit_decode_is_rejected(frame):
    with pytest.raises(ValueError, match='bit depth'):
        check_frame(frame.astype(np.uint8), frame, 0, 0., None)


def test_wrong_event_frame_is_rejected(frame):
    base = frame.copy()
    frame[260:340, 200:280] += 100
    with pytest.raises(ValueError, match='pixel mismatch'):
        check_frame(frame, base, 18, .3, 17/60)


def test_path_traversal_and_symlink_are_rejected(tmp_path):
    (tmp_path/'video.mkv').write_bytes(b'fixture')
    assert safe_child(tmp_path, 'video.mkv') == tmp_path/'video.mkv'
    (tmp_path/'alias.mkv').symlink_to(tmp_path/'video.mkv')
    for relative in ('../video.mkv', 'alias.mkv', str(tmp_path/'video.mkv')):
        with pytest.raises(ValueError):
            safe_child(tmp_path, relative)


def test_pilot_covers_both_ends_of_each_panel_task():
    items = [{'panel': p, 'row': {'task': t}, 'sample_id': str(i)}
             for p, t in [('a', 'T1'), ('b', 'T2'), ('b', 'T3'), ('c', 'T2'), ('d', 'T3')]
             for i in range(3)]
    assert len(pilot_items(items)) == 10


def test_partial_audit_cannot_report_complete():
    config = {'expected_counts': {'T1': 3840, 'T2': 3744, 'T3': 1776}}
    qa = {'status': 'PASS', 'seconds': 1., 'states': {}}
    report = summarize([], 9360, config, qa, 'identity', 1., 'pilot')
    assert report['status'] == 'PILOT_COMPLETE'
    assert report['completed_videos'] == 0 and report['expected_frames'] == 1684800


def test_failed_id_is_preserved_in_summary():
    records = [{'sample_id': 'bad', 'task': 'T1', 'status': 'REVIEW_REQUIRED', 'audit_seconds': 1}]
    report = summarize(records, 1, {'expected_counts': {'T1': 1}},
                       {'status': 'PASS', 'seconds': 1, 'states': {}}, 'id', 1., 'full')
    assert report['status'] == 'REVIEW_REQUIRED'
    assert report['failed_IDs'] == ['bad']
