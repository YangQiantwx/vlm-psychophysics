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
