from copy import deepcopy

import pytest

from classic_precision.dataset import SCHEMA
from classic_precision.manifest import digest, load_protocol, make_manifest
from classic_precision.refinement import build_plan, extend_index


def item(sid, path='T2/example.mkv'):
    return {'sample_id': sid, 'task': 'T2', 'truth': 'LEFT', 'prompt': 'LEFT or RIGHT',
            'video_relative': path, 'video_sha256': 'a'*64, 'stream_sha256': 'b'*64,
            'source_audit_identity': 'c'*64, 'source_ready_sha256': 'd'*64,
            'width': 768, 'height': 512, 'frames': 180, 'fps': 60}


def index(items):
    d = {'schema': SCHEMA, 'items': items, 'counts': {'T2': len(items)}}
    return {**d, 'index_id': digest(d)}


def test_append_preserves_every_original_field():
    a, b = index({'a': item('a')}), index({'b': item('b', 'T2/new.mkv')})
    original = deepcopy(a)
    out = extend_index(a, b)
    assert a == original and out['items']['a'] == original['items']['a']
    assert out['counts'] == {'T2': 2}
    assert out['parent_index_ids'] == [a['index_id'], b['index_id']]


def test_existing_id_cannot_be_rewritten():
    a = index({'a': item('a')})
    with pytest.raises(ValueError, match='overlaps'):
        extend_index(a, a)


def test_duplicate_path_rejected():
    with pytest.raises(ValueError, match='same video path'):
        extend_index(index({'a': item('a')}), index({'b': item('b')}))


def test_unverified_new_source_rejected():
    e = item('b', 'T2/new.mkv'); e.pop('source_audit_identity')
    with pytest.raises(ValueError, match='audit'):
        extend_index(index({'a': item('a')}), index({'b': e}))


def test_plan_refuses_a_different_base_panel():
    with pytest.raises(ValueError, match='9360'):
        build_plan(index({'a': item('a')}))


def test_existing_generator_has_stable_condition_ids():
    c = load_protocol(); c['T2']['delta_Y'] = [.16]; c['T2']['control_10frame_delta_Y'] = [.16]
    c['T3']['angle_degrees'] = []; c['counts'].update(T2_total=288, T3_total=0)
    a = make_manifest(c)
    c['T2']['delta_Y'] += [.17]; c['T2']['control_10frame_delta_Y'] += [.17]
    c['counts']['T2_total'] = 576
    b = make_manifest(c)
    before = {r['sample_id']: r for r in a['rows']}
    after = {r['sample_id']: r for r in b['rows']}
    assert all(after[k] == v for k, v in before.items())


def full_source_index():
    c = load_protocol()
    c['T2']['delta_Y'] = [.05,.06,.07,.08,.09,.1,.11,.12,.13,.14,.15,.2,.4]
    c['T2']['control_10frame_delta_Y'] = list(c['T2']['delta_Y'])
    c['T3']['angle_degrees'] = [.5,.6,.7,.8,.9,1,1.5,1.75,2,2.25,2.5,3,3.5,4,8,16,32,45,90]
    c['counts'].update(T2_total=3744,T3_total=1776)
    rows = make_manifest(c)['rows']
    items = {}
    for r in rows:
        sid = r['sample_id']
        items[sid] = {**item(sid, r['task']+'/'+sid+'.mkv'), 'task': r['task'],
                      'truth': r['truth'], 'prompt': r['prompt'],
                      'nominal_parameters': r['source_parameters']}
    for i in range(3840):
        sid = f't1-fixture-{i}'
        items[sid] = {**item(sid, 'T1/'+sid+'.mkv'), 'task': 'T1'}
    d = {'schema': SCHEMA, 'items': items, 'counts': {'T1':3840,'T2':3744,'T3':1776}}
    return {**d, 'index_id': digest(d)}


def test_revised_common_grid_has_complete_T2_T3_pairs_and_no_old_ids():
    base = full_source_index()
    manifest, plan, names = build_plan(base)
    assert plan['new_videos_by_task'] == {'T2':2016,'T3':1152}
    assert plan['new_videos'] == len(manifest['rows']) == len(names) == 3168
    assert plan['new_model_calls'] == 12672
    assert plan['expected_combined_videos'] == 12528
    assert .225 not in plan['delta_Y']
    assert plan['T3_regular_grid_deg'] == [i/10 for i in range(10,26)]
    assert plan['T3_existing_grid_levels_reused'] == [1,1.5,2,2.5]
    assert set(plan['angle_deg']) == set(plan['T3_regular_grid_deg']) - {1,1.5,2,2.5}
    assert not (set(base['items']) & {r['sample_id'] for r in manifest['rows']})
    from collections import Counter
    groups = Counter()
    for r in manifest['rows']:
        if r['task'] == 'T3':
            p = r['source_parameters']
            groups[(p['angle_deg'], p['tilt_sign'], p['reference_side'])] += 1
    assert len(groups) == 12*2*2 and set(groups.values()) == {24}
    assert manifest['models'] == {} and plan['GPU_calls_started'] == 0


def test_T3_fixed_geometry_change_is_rejected():
    base = full_source_index()
    for e in base['items'].values():
        if e['task'] == 'T3':
            e['nominal_parameters']['bar_wh'] = [21,190]
    base.pop('index_id');base['index_id'] = digest(base)
    with pytest.raises(ValueError, match='source/prompt control'):
        build_plan(base)
