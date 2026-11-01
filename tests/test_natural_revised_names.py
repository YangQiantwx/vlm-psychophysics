import pytest

from classic_precision.dataset_content_audit import parse_name


@pytest.mark.parametrize('task,name', [
    ('T1', 'T1_left_bg-na_071_sf0.5_contrast0.05_trial01.mp4'),
    ('T2', 'T2_right_bg-na_096_obj-na_033_delta0.25_30frames_trial15.mp4'),
    ('T3', 'T3_verticalLeft_tiltRight_0.5deg_bg-gpt_001_obj-na_025_trial01.mp4'),
])
def test_revision_suffix_preserves_condition(task, name):
    revised = name[:-4] + '_rev0123abcd.mp4'
    assert parse_name(revised, 'natural', task) == parse_name(name, 'natural', task)


@pytest.mark.parametrize('suffix', ['_rev1234', '_rev0123abcg', '_rev0123abcd_extra'])
def test_malformed_suffix_is_rejected(suffix):
    with pytest.raises(ValueError):
        parse_name('T2_left_bg-na_001_obj-na_025_delta0.08_30frames_trial01' + suffix + '.mp4', 'natural', 'T2')


def test_classic_naming_contract_unchanged():
    with pytest.raises(ValueError):
        parse_name('T2_left_bg0.2_delta0.08_30frames_trial01_rev0123abcd.mkv', 'classic', 'T2')
