"""Finite checks for the opt-in T3 follow-up generation-order adapter."""
from collections import Counter
from copy import deepcopy

import pytest

from classic_precision import ready
from classic_precision.manifest import canonical, make_manifest, state_key
from runtime_compat import classic_precision_t3_generation_compat as compat


def t3_manifest():
    protocol = deepcopy(make_manifest()['protocol'])
    protocol['protocol_name'] = 't3_threshold_refinement_rgb16_v1'
    protocol['T2']['delta_Y'] = []
    protocol['T2']['control_10frame_delta_Y'] = []
    protocol['T3']['angle_degrees'] = [.6, .7, .8, .9, 1.75, 2.25]
    protocol['counts']['T2_total'] = 0
    protocol['counts']['T3_total'] = 576
    return make_manifest(protocol)


@pytest.mark.parametrize('value', [None, '0'])
def test_no_opt_in_calls_native_unchanged(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(compat.FLAG, raising=False)
    else:
        monkeypatch.setenv(compat.FLAG, value)
    manifest = make_manifest()
    assert canonical(compat.generation_order(manifest)) == canonical(ready.generation_order(manifest))
    with pytest.raises(RuntimeError, match='StopIteration'):
        compat.generation_order(t3_manifest())
    assert compat.install() is False


def test_opt_in_keeps_original_dual_task_order_identical(monkeypatch):
    monkeypatch.setenv(compat.FLAG, '1')
    manifest = make_manifest()
    for shards in (1, 3, 6):
        expected = ready.generation_order(manifest, shards)
        actual = compat.generation_order(manifest, shards)
        assert actual == expected
        assert canonical(actual) == canonical(expected)


def test_t3_followup_has_all_576_unchanged_rows_and_24_states(monkeypatch):
    monkeypatch.setenv(compat.FLAG, '1')
    manifest = t3_manifest()
    original = deepcopy(manifest)
    rows = compat.generation_order(manifest, 3)
    assert manifest == original
    assert len(rows) == len({r['sample_id'] for r in rows}) == 576
    assert {r['sample_id'] for r in rows} == {r['sample_id'] for r in manifest['rows']}
    assert all(any(row is source for source in manifest['rows']) for row in rows)
    assert [r['source_parameters']['onset_index'] for r in rows[:3]] == [0, 1, 2]
    assert len({state_key(r) for r in rows}) == 24
    assert Counter(r['source_parameters']['angle_deg'] for r in rows) == {
        .6: 96, .7: 96, .8: 96, .9: 96, 1.75: 96, 2.25: 96}
    assert Counter((r['source_parameters']['reference_side'],
                    r['source_parameters']['tilt_sign']) for r in rows) == {
        ('LEFT', -1): 144, ('LEFT', 1): 144,
        ('RIGHT', -1): 144, ('RIGHT', 1): 144}


def test_missing_t2_and_empty_shard_do_not_invent_rows(monkeypatch):
    monkeypatch.setenv(compat.FLAG, '1')
    manifest = t3_manifest()
    rows = compat.generation_order(manifest, 25)
    assert len(rows) == len({r['sample_id'] for r in rows}) == 576
    assert {r['task'] for r in rows} == {'T3'}


@pytest.mark.parametrize('change', ['empty', 'T2_only', 'unexpected_task',
                                  'manifest_T2_nonzero', 'protocol_T2_nonzero'])
def test_unexpected_inputs_delegate_to_native(monkeypatch, change):
    monkeypatch.setenv(compat.FLAG, '1')
    manifest = t3_manifest()
    if change == 'empty':
        manifest['rows'] = []
    elif change == 'T2_only':
        manifest['rows'] = [r for r in make_manifest()['rows'] if r['task'] == 'T2']
    elif change == 'unexpected_task':
        manifest['rows'][0]['task'] = 'T1'
    elif change == 'manifest_T2_nonzero':
        manifest['counts']['T2'] = 1
    else:
        manifest['protocol']['counts']['T2_total'] = 1
    sentinel = object()
    calls = []
    def native(current, shards):
        calls.append((current, shards))
        return sentinel
    monkeypatch.setattr(compat, '_native_generation_order', native)
    assert compat.generation_order(manifest, 7) is sentinel
    assert calls == [(manifest, 7)]


def test_install_changes_only_generation_order_with_opt_in(monkeypatch):
    monkeypatch.setenv(compat.FLAG, '1')
    monkeypatch.setattr(ready, 'generation_order', ready.generation_order)
    original_representatives = ready.representatives
    assert compat.install() is True
    assert ready.generation_order is compat.generation_order
    assert ready.representatives is original_representatives
