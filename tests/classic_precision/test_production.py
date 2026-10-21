"""Scientific identities, codec fidelity, output durability and complete scoring."""
from collections import Counter
from copy import deepcopy
import json

import numpy as np
import pytest

from classic_precision.manifest import (atomic_json, canonical, make_manifest,
    selected_rows, state_key, validate_manifest, write_manifest)
from classic_precision.scheduler import choose_model
from classic_precision.storage import RunStore, classify_output


def test_frozen_grid_and_gt_tamper():
    manifest = make_manifest()
    assert Counter(r['task'] for r in manifest['rows']) == {'T2': 2304, 'T3': 1200}
    assert len({state_key(r) for r in manifest['rows']}) == 128
    partitions = [selected_rows(manifest, i, 3) for i in range(3)]
    for rows in partitions:
        assert len(rows) == 1168
        assert Counter(r['task'] for r in rows) == {'T2': 768, 'T3': 400}
        assert Counter(r['truth'] for r in rows) == {'LEFT': 584, 'RIGHT': 584}
    tampered = deepcopy(manifest)
    tampered['rows'][0]['truth'] = 'RIGHT'
    with pytest.raises(ValueError):
        validate_manifest(tampered)


def test_source_photometry_and_two_axis_algorithms():
    from classic_precision.stimuli import make_state
    m = make_manifest()
    t2 = next(r for r in m['rows'] if r['task'] == 'T2' and r['source_parameters']['delta_Y'] == .09)
    t3 = next(r for r in m['rows'] if r['task'] == 'T3' and r['source_parameters']['angle_deg'] == .5)
    for row in (t2, t3):
        base, event, receipt = make_state(row)
        assert base.dtype == event.dtype == np.uint16
        assert event.shape == (512, 768, 3)
        if row['task'] == 'T2':
            assert receipt['changed_outside_target'] == 0
            assert receipt['core_max_error'] < .0001
        else:
            assert receipt['independent_GEOS_max_coverage_difference'] < 1e-9
            assert max(a['PCA_error_deg'] for a in receipt['axes']) < .001
            assert max(a['halfplane_error_deg'] for a in receipt['axes']) < .001


def test_rgb48_codec_and_temporal_order(tmp_path):
    import av
    from fractions import Fraction
    file = tmp_path/'sentinel.mkv'
    # Distinct RGB channels, full-range codes, spatial/temporal sentinels.
    q = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    frames = [np.stack([q, np.flipud(q), np.fliplr(q)], axis=-1),
              np.stack([np.fliplr(q), q, np.flipud(q)], axis=-1)]
    with av.open(str(file), 'w') as out:
        stream = out.add_stream('ffv1', rate=60)
        stream.width = stream.height = 256
        stream.pix_fmt = 'gbrp16le'
        stream.options = {'level': '3'}
        for i, codes in enumerate(frames):
            frame = av.VideoFrame.from_ndarray(codes, format='rgb48le')
            frame.pts, frame.time_base = i, Fraction(1, 60)
            for packet in stream.encode(frame):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)
    with av.open(str(file)) as video:
        decoded = list(video.decode(video=0))
    assert len(decoded) == 2
    for expected, actual in zip(frames, decoded):
        assert np.array_equal(expected, actual.to_ndarray(format='rgb48le'))
        fp32 = expected.astype(np.float32)*(255./65535.)
        assert np.array_equal(expected, np.rint(fp32*(65535./255.)).astype(np.uint16))


def test_raw_fsync_is_terminal_even_without_derived_result(tmp_path):
    identity = {'protocol_id': 'fixture'}
    with RunStore(tmp_path, identity, 'nemo', 'one', ['a', 'b']) as store:
        store.begin('a', 'row-a')
        store.commit_output('a', {'raw_answer': 'LEFT', 'output_token_ids': [1],
                                  'finish_reason': 'stop', 'truth': 'RIGHT'})
        store.begin('b', 'row-b')
        store.fail('b', RuntimeError('input guard'))
    (tmp_path/'nemo/results.jsonl').write_text('')  # Crash after raw fsync simulation.
    with RunStore(tmp_path, identity, 'nemo', 'one', ['a', 'b']) as store:
        assert store.pending_ids(retry_failed=False) == []
        assert store.pending_ids() == ['b']
        assert store.summary()['completed_outputs'] == 1
        assert len((tmp_path/'nemo/results.jsonl').read_text().splitlines()) == 1
    assert classify_output({'sample_id': 'x', 'attempt_id': 'x', 'raw_answer': 'LEFT',
                            'truth': 'LEFT', 'finish_reason': 'length'})['correct'] is False


def test_shared_gpu_admission_is_by_actual_capacity():
    specs = make_manifest()['models']
    all_models = list(specs)
    assert choose_model(all_models, specs, {'free_gpu_gib': 74, 'available_ram_gib': 100}) == 'qwen'
    assert choose_model(all_models, specs, {'free_gpu_gib': 78, 'available_ram_gib': 48}) == 'gemma'
    assert choose_model(all_models, specs, {'free_gpu_gib': 60, 'available_ram_gib': 100}) is None


def test_checkpoint_identity_uses_exact_nested_index_shards(tmp_path):
    from classic_precision.inference import checkpoint_asset_identity
    checkpoint = tmp_path/'indexed'
    (checkpoint/'transformer').mkdir(parents=True)
    (checkpoint/'vision_encoder').mkdir()
    first = checkpoint/'transformer/part.safetensors'
    second = checkpoint/'vision_encoder/model.safetensors'
    first.write_bytes(b'first')
    second.write_bytes(b'second')
    index = checkpoint/'model.safetensors.index.json'
    index.write_text(json.dumps({'weight_map': {
        'layer.a': 'transformer/part.safetensors',
        'layer.b': 'transformer/part.safetensors',
        'vision': 'vision_encoder/model.safetensors'}}))
    assert checkpoint_asset_identity(checkpoint)['weight_shard_bytes'] == {
        'transformer/part.safetensors': 5, 'vision_encoder/model.safetensors': 6}
    second.unlink()
    with pytest.raises(ValueError, match='shard is missing'):
        checkpoint_asset_identity(checkpoint)
    for escaped in ('../outside.safetensors', '/absolute.safetensors'):
        index.write_text(json.dumps({'weight_map': {'layer': escaped}}))
        with pytest.raises(ValueError):
            checkpoint_asset_identity(checkpoint)
    fallback = tmp_path/'unindexed'
    fallback.mkdir()
    (fallback/'model.safetensors').write_bytes(b'single')
    assert checkpoint_asset_identity(fallback)['weight_shard_bytes'] == {'model.safetensors': 6}


def test_one_retry_is_persistent_and_counted_separately(tmp_path):
    from classic_precision.storage import PersistenceError
    from classic_precision.summary import attempt_metrics, model_snapshot
    with RunStore(tmp_path, {'protocol_id': 'retry-fixture'}, 'nemo', 'one', ['a', 'b']) as store:
        store.begin('a', 'a-input')
        store.fail('a', RuntimeError('interrupted call'), elapsed_seconds=3.)
        store.begin('a', 'a-input')
        store.commit_output('a', {'raw_answer': 'LEFT', 'output_token_ids': [1],
                                  'finish_reason': 'stop', 'truth': 'LEFT'})
        for _ in range(2):
            store.begin('b', 'b-input')
            store.fail('b', RuntimeError('no output'), elapsed_seconds=2.)
    with RunStore(tmp_path, {'protocol_id': 'retry-fixture'}, 'nemo', 'one', ['a', 'b']) as store:
        assert store.pending_ids() == []
        with pytest.raises(PersistenceError):
            store.begin('b', 'b-input')
        assert store.summary()['exhausted_ids'] == ['b']
    _, raw, events, _, _ = model_snapshot(tmp_path/'nemo')
    result = attempt_metrics(raw, events + events)
    assert result['started_attempts'] == 4
    assert result['retry_attempts'] == 2
    assert result['completed_unique_answers'] == 1
    assert result['no_output_failed'] == 3
    assert result['known_failed_attempt_seconds'] == 7.
    assert result['exhausted_sample_ids'] == ['b']


def test_scoring_balanced_bias_missing_and_duplicate_outputs(tmp_path):
    from classic_precision.inference import GENERATION
    from classic_precision.summary import summarize
    run = tmp_path/'run'
    m = write_manifest(run/'manifest.json')
    directory = run/'outputs/nemo'
    directory.mkdir(parents=True)
    identity = {'protocol_id': m['protocol_id'], 'model_key': 'nemo',
                'model': m['models']['nemo'], 'generation': GENERATION,
                **{k: 'fixture' for k in ('manifest_sha256', 'ready_sha256', 'effective_resources',
                     'checkpoint_assets', 'cosmos_package_sources', 'runtime_versions',
                     'runner_sha256', 'storage_sha256', 'all_row_ids_sha256')}}
    from classic_precision.manifest import file_sha
    identity['manifest_sha256'] = file_sha(run/'manifest.json')
    atomic_json(directory/'identity.json', identity)
    records = [{**r, 'attempt_id': str(i), 'raw_answer': 'LEFT', 'finish_reason': 'stop',
                'output_token_ids': [1], 'actual_input_tokens': 35000, 'seconds': 2.}
               for i, r in enumerate(m['rows'])]
    (directory/'outputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    events = [{'event': 'TECHNICAL_PASS', 'sample_id': r['sample_id'], 'attempt_id': r['attempt_id']} for r in records]
    (directory/'events.jsonl').write_bytes(b''.join(canonical(e)+b'\n' for e in events))
    result = summarize(run/'manifest.json', [run/'outputs'], tmp_path/'summary')
    assert result['models']['nemo']['correct'] == 1752
    assert result['models']['nemo']['accuracy'] == .5
    assert result['models']['qwen']['accuracy'] is None
    assert result['T2_500ms_vs167ms']['nemo']['completed_pairs'] == 432
    assert result['models']['nemo']['direction_pairs_expected'] == 1752
    assert result['cost']['nemo']['completed_call_seconds'] == 7008
    with pytest.raises(ValueError):
        summarize(run/'manifest.json', [run/'outputs'], tmp_path/'summary', require_complete=True)
    records.append({**records[0], 'attempt_id': 'duplicate'})
    (directory/'outputs.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in records))
    with pytest.raises(ValueError):
        summarize(run/'manifest.json', [run/'outputs'], tmp_path/'summary')
