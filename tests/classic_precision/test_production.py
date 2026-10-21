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
                **{k: 'fixture' for k in ('manifest_sha256', 'input_binding', 'ready_source_sha256', 'calibration_identity_sha256', 'effective_resources',
                     'checkpoint_assets', 'cosmos_package_sources', 'runtime_versions',
                     'runner_sha256', 'storage_sha256', 'all_row_ids_sha256')}}
    from classic_precision.manifest import file_sha
    identity['manifest_sha256'] = file_sha(run/'manifest.json')
    from classic_precision.ready import INPUT_BINDING
    identity.update(schema_version=2, input_binding=INPUT_BINDING)
    atomic_json(directory/'identity.json', identity)
    records = [{**r, 'attempt_id': str(i), 'raw_answer': 'LEFT', 'finish_reason': 'stop',
                'output_token_ids': [1], 'actual_input_tokens': 35000, 'seconds': 2.}
               for i, r in enumerate(m['rows'])]
    from classic_precision.storage import digest
    for record in records:
        scientific_row = next(row for row in m['rows'] if row['sample_id'] == record['sample_id'])
        receipt = _fixture_receipt(scientific_row)
        record.update(ready_record=receipt, ready_identity=digest(receipt),
                      row_identity=digest({'row': scientific_row, 'ready': receipt}))
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


def _fixture_receipt(row, data=b'fixture bytes'):
    import hashlib
    return {'sample_id': row['sample_id'], 'source_parameters': row['source_parameters'],
            'video_relative': 'videos/'+row['sample_id']+'.mkv',
            'video_sha256': hashlib.sha256(data).hexdigest(),
            'stream_sha256': hashlib.sha256(b'fixture RGB16 stream').hexdigest(),
            'frames': 180, 'master_bits_per_channel': 16,
            'max_decoded_code_error': 0, 'max_PTS_error_s': 0.}


def _publish_fixture(root, manifest, row, data=None):
    """Tiny transport bytes, not scientific video/processor acceptance evidence."""
    from classic_precision.ready import append_ready
    data = data or ('transport-fixture-'+row['sample_id']).encode()
    item = _fixture_receipt(row, data)
    path = root/item['video_relative']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    atomic_json(path.with_suffix('.ready.json'), item)
    append_ready(manifest, root, item)
    return item


def test_incremental_ready_tail_cache_mutation_and_complete_missing(tmp_path, monkeypatch):
    import classic_precision.ready as inputs
    from classic_precision.manifest import file_sha
    m = write_manifest(tmp_path/'manifest.json')
    a = _publish_fixture(tmp_path, m, m['rows'][0])
    real_hash, calls = inputs.file_sha, []
    def counting(path):
        calls.append(str(path))
        return real_hash(path)
    monkeypatch.setattr(inputs, 'file_sha', counting)
    catalog = inputs.ReadyCatalog(m, tmp_path/'ready.jsonl', tmp_path, file_sha(tmp_path/'manifest.json'))
    assert catalog.refresh() == {a['sample_id']: a}
    assert catalog.refresh() == {a['sample_id']: a}
    assert len(calls) == 1
    # A fully parseable JSON object without its newline is still unpublished.
    b = _publish_fixture(tmp_path, m, m['rows'][1])
    (tmp_path/'ready.jsonl').write_bytes(canonical(a)+b'\n'+canonical(b))
    assert set(catalog.refresh()) == {a['sample_id']}
    assert catalog.tail
    inputs.append_ready(m, tmp_path, b)
    assert set(catalog.refresh()) == {a['sample_id'], b['sample_id']}
    assert list(tmp_path.glob('ready.torn.*.bin'))
    changed = {**a, 'stream_sha256': '0'*64}
    atomic_json(tmp_path/'videos'/(a['sample_id']+'.ready.json'), changed)
    (tmp_path/'ready.jsonl').write_bytes(canonical(changed)+b'\n'+canonical(b)+b'\n')
    with pytest.raises(ValueError, match='identity changed'):
        catalog.refresh()
    atomic_json(tmp_path/'videos'/(a['sample_id']+'.ready.json'), a)
    (tmp_path/'ready.jsonl').write_bytes(canonical(a)+b'\n'+canonical(b)+b'\n')
    atomic_json(tmp_path/'generation.json', {'status': 'COMPLETE', 'protocol_id': m['protocol_id'],
        'manifest_sha256': file_sha(tmp_path/'manifest.json'), 'videos': 3504})
    with pytest.raises(ValueError, match='missing or partial'):
        catalog.refresh()
    (tmp_path/'ready.jsonl').write_bytes(canonical(a)+b'\nBROKEN\n'+canonical(b)+b'\n')
    with pytest.raises(ValueError):
        catalog.refresh()


def test_priority_buffer_and_producer_failure(tmp_path):
    from classic_precision.ready import ReadyCatalog, generation_order, representatives, start_ready
    from classic_precision.manifest import file_sha
    m = write_manifest(tmp_path/'manifest.json')
    ordered = generation_order(m)
    assert len(ordered) == len({r['sample_id'] for r in ordered}) == 3504
    assert {r['sample_id'] for r in representatives(m)} <= {r['sample_id'] for r in ordered[:134]}
    catalog = ReadyCatalog(m, tmp_path/'ready.jsonl', tmp_path, file_sha(tmp_path/'manifest.json'))
    for index in range(3):
        selected = selected_rows(m, index, 3)
        first = [next(r for r in selected if r['task'] == task) for task in ('T2', 'T3')]
        assert {r['sample_id'] for r in first} <= {r['sample_id'] for r in ordered[:6]}
        ready = {r['sample_id']: {} for r in first}
        for row in selected:
            if len(ready) == 100:
                break
            ready[row['sample_id']] = {}
        assert start_ready(catalog, selected, ready)
        ready.pop(next(sid for sid in ready if sid not in {r['sample_id'] for r in first}))
        assert not start_ready(catalog, selected, ready)
        catalog.complete = True
        assert start_ready(catalog, selected, ready)
        catalog.complete = False
    atomic_json(tmp_path/'generation.json', {'status': 'FAILED', 'protocol_id': m['protocol_id'],
        'manifest_sha256': file_sha(tmp_path/'manifest.json'), 'error': 'encoder fixture failed'})
    with pytest.raises(RuntimeError, match='producer FAILED'):
        catalog.refresh()


def test_stream_wait_loads_one_engine_and_never_repeats_raw(tmp_path, monkeypatch):
    import contextlib
    import os
    import sys
    from types import SimpleNamespace
    import classic_precision.inference as infer
    import classic_precision.calibration as calibration
    from classic_precision.storage import digest
    from classic_precision.summary import model_snapshot
    m = write_manifest(tmp_path/'manifest.json')
    chosen = [m['rows'][0], next(r for r in m['rows'] if r['task'] == 'T3'), *m['rows'][1:3]]
    _publish_fixture(tmp_path, m, chosen[0])
    output = tmp_path/'outputs'
    stats = {'created': 0, 'closed': 0, 'calls': [], 'waits': 0, 'calibrated': False}
    identity = {'protocol_id': m['protocol_id']}
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setattr(infer, 'build_run_identity', lambda *a, **k: (identity, tmp_path/'model'))
    monkeypatch.setattr(infer, 'resource_snapshot', lambda *a: {'admitted': True})
    monkeypatch.setattr(infer, '_gpu_lock', lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(calibration, 'calibration_identity', lambda *a, **k: {'fixture': '128 states'})
    monkeypatch.setattr(calibration, 'calibration_ready', lambda *a, **k: stats['calibrated'])
    class Engine:
        def __init__(self, model_key, path, store, *args):
            stats['created'] += 1
            self.store, self.context = store, {}
        def generate(self, row, ready, root, durable):
            stats['calls'].append(row['sample_id'])
            durable({**row, 'row_identity': digest({'row': row, 'ready': ready}),
                     'raw_answer': 'LEFT', 'output_token_ids': [1], 'finish_reason': 'stop',
                     'actual_input_tokens': 35000, 'input_receipt': {'input_tokens': 35000},
                     'effective_generation': infer.GENERATION, 'seconds': .01})
            self.store.event('TECHNICAL_PASS', sample_id=row['sample_id'],
                             attempt_id=self.store.outputs[row['sample_id']]['attempt_id'])
        def close(self):
            stats['closed'] += 1
    monkeypatch.setattr(infer, 'ModelSession', Engine)
    def wait(seconds):
        stats['waits'] += 1
        _, raw, events, _, _ = model_snapshot(output/'nemo')
        inflight = [e for e in events if e.get('event') == 'INFLIGHT']
        if stats['waits'] == 1:
            assert not inflight and stats['created'] == 0
            stats['calibrated'] = True
            _publish_fixture(tmp_path, m, chosen[1])
        elif stats['waits'] == 2:
            assert len(inflight) == len(raw) == 2 and stats['created'] == 1
            for row in chosen[2:]:
                _publish_fixture(tmp_path, m, row)
        else:
            raise AssertionError('Unexpected extra waiting/reloading')
    monkeypatch.setattr(infer.time, 'sleep', wait)
    kw = dict(manifest_path=tmp_path/'manifest.json', output_dir=output, model_key='nemo',
              model_root=tmp_path/'models', row_ids=[r['sample_id'] for r in chosen], start_ready=2,
              input_poll_seconds=1)
    assert infer.run_manifest(**kw)['status'] == 'COMPLETE'
    assert stats['created'] == stats['closed'] == 1
    assert len(stats['calls']) == len(set(stats['calls'])) == 4
    assert infer.run_manifest(**kw)['completed_outputs'] == 4
    assert stats['created'] == 1 and len(stats['calls']) == 4
    # A valid new receipt cannot silently change a previously observed input
    # even after constructing a fresh catalog/worker on restart.
    import classic_precision.ready as inputs
    records, _, _ = inputs.read_records(tmp_path/'ready.jsonl')
    sid = chosen[0]['sample_id']
    records[sid]['stream_sha256'] = '0'*64
    atomic_json(tmp_path/'videos'/(sid+'.ready.json'), records[sid])
    (tmp_path/'ready.jsonl').write_bytes(b''.join(canonical(r)+b'\n' for r in records.values()))
    with pytest.raises(ValueError, match='identity changed'):
        infer.run_manifest(**kw)


def test_128_calibration_and_run_identity_stay_stable_while_ready_grows(tmp_path):
    from classic_precision.calibration import calibration_identity, calibration_ready
    from classic_precision.inference import build_run_identity
    from classic_precision.ready import ReadyCatalog, representatives
    from classic_precision.manifest import file_sha
    from classic_precision.storage import digest
    m = write_manifest(tmp_path/'manifest.json')
    checkpoint = tmp_path/'models'/m['models']['nemo']['checkpoint_relative']
    checkpoint.mkdir(parents=True)
    (checkpoint/'model.safetensors').write_bytes(b'metadata-only-fixture')
    reps = representatives(m)
    for row in reps:
        _publish_fixture(tmp_path, m, row)
    catalog = ReadyCatalog(m, tmp_path/'ready.jsonl', tmp_path, file_sha(tmp_path/'manifest.json'))
    ready = catalog.refresh()
    identity = calibration_identity(tmp_path/'manifest.json', m, 'nemo', tmp_path/'models')
    before = build_run_identity(tmp_path/'manifest.json', tmp_path/'ready.jsonl', m, 'nemo', tmp_path/'models')[0]
    records, dest = [], tmp_path/'calibration/nemo'
    for row in reps:
        record = {'identity': identity, 'sample_id': row['sample_id'], 'state_key': state_key(row),
                  'source_parameters': row['source_parameters'],
                  'row_identity': digest({'row': row, 'ready': ready[row['sample_id']]})}
        records.append(record)
        atomic_json(dest/(state_key(row)+'.json'), record)
    atomic_json(dest/'summary.json', {'identity': identity, 'status': 'COMPLETE', 'unique_states': 128,
        'weights_loaded': False, 'new_model_calls': 0, 'records_sha256': digest(records)})
    assert calibration_ready(tmp_path/'calibration', 'nemo', identity, m, ready)
    extra = next(row for row in m['rows'] if row['sample_id'] not in ready)
    _publish_fixture(tmp_path, m, extra)
    ready = catalog.refresh()
    assert identity == calibration_identity(tmp_path/'manifest.json', m, 'nemo', tmp_path/'models')
    assert before == build_run_identity(tmp_path/'manifest.json', tmp_path/'ready.jsonl', m, 'nemo', tmp_path/'models')[0]
    assert calibration_ready(tmp_path/'calibration', 'nemo', identity, m, ready)
    bad = deepcopy(ready)
    bad[reps[0]['sample_id']]['stream_sha256'] = '0'*64
    with pytest.raises(ValueError, match='input identity changed'):
        calibration_ready(tmp_path/'calibration', 'nemo', identity, m, bad)


def test_incremental_replica_late_arrival_and_partial_never_published(tmp_path, monkeypatch):
    import hashlib
    import io
    import sys
    from types import SimpleNamespace
    from classic_precision.ready import ReadyCatalog, receive_video, sync_ready
    from classic_precision.manifest import file_sha
    source, dest = tmp_path/'host-a', tmp_path/'host-b'
    m = write_manifest(source/'manifest.json')
    a = _publish_fixture(source, m, m['rows'][0])
    (source/'videos/inflight.mkv.partial').write_bytes(b'incomplete unrelated bytes')
    first = sync_ready(source/'manifest.json', source, dest, once=True)
    assert first['copied_this_round'] == first['destination_ready'] == 1
    catalog = ReadyCatalog(m, dest/'ready.jsonl', dest, file_sha(dest/'manifest.json'))
    assert set(catalog.refresh()) == {a['sample_id']}
    b = _publish_fixture(source, m, m['rows'][1])
    second = sync_ready(source/'manifest.json', source, dest, once=True)
    assert second['copied_this_round'] == 1 and second['destination_ready'] == 2
    assert set(catalog.refresh()) == {a['sample_id'], b['sample_id']}
    row = m['rows'][2]
    incomplete = {**a, 'sample_id': row['sample_id'], 'source_parameters': row['source_parameters'],
                  'video_relative': 'videos/'+row['sample_id']+'.mkv'}
    monkeypatch.setattr(sys, 'stdin', SimpleNamespace(buffer=io.BytesIO(b'broken')))
    with pytest.raises(ValueError, match='Incomplete/corrupt'):
        receive_video(dest, incomplete, 999)
    assert row['sample_id'] not in catalog.refresh()
    assert not (dest/incomplete['video_relative']).exists()


def test_summary_rejects_cross_model_ready_identity_drift(tmp_path):
    from classic_precision.inference import GENERATION
    from classic_precision.manifest import file_sha
    from classic_precision.storage import digest
    from classic_precision.summary import summarize
    m = write_manifest(tmp_path/'manifest.json')
    row = m['rows'][0]
    for family in ('nemo', 'qwen'):
        directory = tmp_path/'outputs'/family
        identity = {'protocol_id': m['protocol_id'], 'model_key': family, 'model': m['models'][family],
                    'generation': GENERATION, 'manifest_sha256': file_sha(tmp_path/'manifest.json'),
                    **{k: 'fixture' for k in ('input_binding', 'ready_source_sha256', 'calibration_identity_sha256',
                        'effective_resources', 'checkpoint_assets', 'cosmos_package_sources', 'runtime_versions',
                        'runner_sha256', 'storage_sha256', 'all_row_ids_sha256')}}
        from classic_precision.ready import INPUT_BINDING
        identity.update(schema_version=2, input_binding=INPUT_BINDING)
        atomic_json(directory/'identity.json', identity)
        receipt = _fixture_receipt(row, family.encode())
        record = {**row, 'row_identity': digest({'row': row, 'ready': receipt}),
                  'ready_record': receipt, 'ready_identity': digest(receipt), 'attempt_id': family,
                  'raw_answer': 'LEFT', 'finish_reason': 'stop', 'output_token_ids': [1], 'seconds': 1.}
        (directory/'outputs.jsonl').write_bytes(canonical(record)+b'\n')
        (directory/'events.jsonl').write_bytes(canonical({'event': 'TECHNICAL_PASS',
            'sample_id': row['sample_id'], 'attempt_id': family})+b'\n')
    with pytest.raises(ValueError, match='different ready videos'):
        summarize(tmp_path/'manifest.json', [tmp_path/'outputs'], tmp_path/'summary')


def test_torn_ready_repair_never_truncates_published_prefix(tmp_path, monkeypatch):
    import classic_precision.ready as inputs
    m = write_manifest(tmp_path/'manifest.json')
    a = _publish_fixture(tmp_path, m, m['rows'][0])
    b = _publish_fixture(tmp_path, m, m['rows'][1])
    original = canonical(a)+b'\n'+canonical(b)[:100]
    path = tmp_path/'ready.jsonl'
    path.write_bytes(original)
    real_replace = inputs.os.replace
    def crash_before_publication(source, destination):
        if destination == path:
            assert path.read_bytes() == original
            raise OSError('fixture crash before atomic repair publication')
        return real_replace(source, destination)
    monkeypatch.setattr(inputs.os, 'replace', crash_before_publication)
    with pytest.raises(OSError, match='before atomic repair'):
        inputs.append_ready(m, tmp_path, b)
    assert path.read_bytes() == original
    records, tail, _ = inputs.read_records(path)
    assert records == {a['sample_id']: a} and tail


def test_source_failure_reaches_replica_and_explicit_resume(tmp_path):
    import subprocess
    from classic_precision.ready import ReadyCatalog, replica_status, sync_ready
    from classic_precision.manifest import file_sha
    source, dest = tmp_path/'source', tmp_path/'replica'
    m = write_manifest(source/'manifest.json')
    a = _publish_fixture(source, m, m['rows'][0])
    sync_ready(source/'manifest.json', source, dest, once=True)
    catalog = ReadyCatalog(m, dest/'ready.jsonl', dest, file_sha(dest/'manifest.json'))
    assert len(catalog.refresh()) == 1
    replica_status(source, 'FAILED', 'fixture encoder failed')
    with pytest.raises(subprocess.CalledProcessError):
        sync_ready(source/'manifest.json', source, dest, once=True)
    with pytest.raises(RuntimeError, match='producer FAILED'):
        catalog.refresh()
    assert json.loads((dest/'generation.failure.json').read_text())['status'] == 'FAILED'
    replica_status(source, 'RUNNING')
    b = _publish_fixture(source, m, m['rows'][1])
    resumed = sync_ready(source/'manifest.json', source, dest, once=True)
    assert resumed['copied_this_round'] == 1
    assert set(catalog.refresh()) == {a['sample_id'], b['sample_id']}
    assert json.loads((dest/'generation.json').read_text())['status'] == 'RUNNING'


def test_supervisor_existing_incomplete_model_does_not_block_calibrated_model(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import classic_precision.scheduler as scheduler
    import classic_precision.inference as inference
    import classic_precision.calibration as calibration
    m = write_manifest(tmp_path/'manifest.json')
    rows = selected_rows(m, 0, 3)
    qualification = [next(row for row in rows if row['task'] == task) for task in ('T2', 'T3')]
    selected = qualification + [row for row in rows if row not in qualification][:98]
    for row in selected:
        _publish_fixture(tmp_path, m, row)
    model_root = tmp_path/'models'
    # An unfinished downloaded directory exists, but no completed calibration.
    (model_root/m['models']['nemo']['checkpoint_relative']).mkdir(parents=True)
    (model_root/m['models']['qwen']['checkpoint_relative']).mkdir(parents=True)
    atomic_json(tmp_path/'calibration/qwen/summary.json', {'status': 'fixture complete'})
    bound, launched, finished = [], [], set()
    def bind(manifest_path, ready_path, manifest, model, *args, **kwargs):
        bound.append(model)
        assert model == 'qwen'
        return {'fixture': model}, model_root/manifest['models'][model]['checkpoint_relative']
    monkeypatch.setattr(inference, 'build_run_identity', bind)
    monkeypatch.setattr(calibration, 'calibration_identity', lambda *a, **k: {})
    monkeypatch.setattr(calibration, 'calibration_ready', lambda *a, **k: True)
    def progress(output, family, ids, *args, **kwargs):
        done = family in finished
        return {'complete': done, 'completed': len(ids) if done else 0, 'needs_recovery': False,
                'exhausted_ids': [], 'held_ids': [], 'retryable_ids': [], 'attempt_ids': {},
                'technical_unresolved_ids': []}
    monkeypatch.setattr(scheduler, 'progress', progress)
    monkeypatch.setattr(scheduler, 'capacity', lambda gpu: {'free_gpu_gib': 74, 'available_ram_gib': 100})
    def run(command, **kwargs):
        model = command[command.index('--model')+1]
        launched.append(model)
        finished.add(model)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(scheduler.subprocess, 'run', run)
    class EndFixture(Exception):
        pass
    monkeypatch.setattr(scheduler.time, 'sleep', lambda seconds: (_ for _ in ()).throw(EndFixture()))
    with pytest.raises(EndFixture):
        scheduler.supervise(tmp_path/'manifest.json', tmp_path/'outputs', model_root, '0', 0)
    assert bound == launched == ['qwen']
    status = json.loads((tmp_path/'outputs/supervisor-shard-0.json').read_text())
    assert status['status'] == 'WAITING_FOR_INPUTS_CALIBRATION_OR_CHECKPOINT'
    assert status['completed']['qwen'] == 1168
    assert status['waiting']['nemo'] == 'WAITING_FOR_128_STATE_CALIBRATION'
    assert status['waiting']['cosmos'] == status['waiting']['gemma'] == 'WAITING_FOR_CHECKPOINT'


def test_cpu_budget_reduces_inflight_without_stopping_full_buffer(monkeypatch):
    import classic_precision.cpu as cpu
    monkeypatch.setattr(cpu, 'available_cores', lambda: 18)
    monkeypatch.setattr(cpu, 'available_memory_gib', lambda: 100.)
    assert cpu.allocation()['workers'] == 16
    buffered = cpu.allocation(buffer_rows=200)
    assert 0 < buffered['workers'] == 4
    assert cpu.allocation(memory_gib=8, calibration_active=True)['workers'] == 0
    monkeypatch.setattr(cpu, 'available_memory_gib', lambda: 4.)
    assert cpu.allocation()['reason'] == 'WAITING_FOR_RAM'
    monkeypatch.setattr(cpu, 'available_memory_gib', lambda: 5.)
    assert cpu.allocation()['workers'] == 2


def test_shared_state_cache_mutex_and_empty_queue_finishes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor, Future
    import threading
    import time
    import classic_precision.stimuli as stimuli
    import classic_precision.cpu as cpu
    import classic_precision.ready as ready
    active, maximum = [0], [0]
    guard = threading.Lock()
    def prepare(manifest, work):
        with guard:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        time.sleep(.02)
        with guard:
            active[0] -= 1
        return 'fixture'
    monkeypatch.setattr(stimuli, '_prepare_states_locked', prepare)
    with ThreadPoolExecutor(2) as pool:
        assert list(pool.map(lambda x: stimuli.prepare_states({}, tmp_path/'cache'), range(2))) == ['fixture']*2
    assert maximum[0] == 1
    m = write_manifest(tmp_path/'run/manifest.json')
    row = m['rows'][0]
    monkeypatch.setattr(stimuli, 'prepare_states', lambda *a: ('unused.npy', {'states': {state_key(row): {}}}))
    monkeypatch.setattr(ready, 'generation_order', lambda *a: [row])
    plans = []
    def allocation(*a, **k):
        plans.append(1)
        return {'workers': 1 if len(plans) == 1 else 0, 'reason': 'READY_DEMAND'}
    monkeypatch.setattr(cpu, 'allocation', allocation)
    monkeypatch.setattr(stimuli, '_encode_one', lambda task: _publish_fixture(tmp_path/'run', m, row))
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def submit(self, function, task):
            future = Future()
            future.set_result(function(task))
            return future
    monkeypatch.setattr(stimuli.concurrent.futures, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(ready, 'finish_generation', lambda *a: {'fixture_finished': True})
    assert stimuli.generate(m, tmp_path/'run', workers=1)['fixture_finished']
    assert len(plans) == 1  # No extra submit/probe is needed to discover the empty queue.


def test_checkpoint_preparation_waits_for_shards_and_tokenizer(tmp_path):
    from classic_precision.pipeline import checkpoint_prepared
    atomic_json(tmp_path/'config.json', {})
    atomic_json(tmp_path/'preprocessor_config.json', {})
    atomic_json(tmp_path/'tokenizer_config.json', {})
    (tmp_path/'model-00001-of-00002.safetensors').write_bytes(b'fixture')
    assert not checkpoint_prepared(tmp_path)
    atomic_json(tmp_path/'tokenizer.json', {})
    assert not checkpoint_prepared(tmp_path)  # A first shard without its index is never a complete model.
    atomic_json(tmp_path/'model.safetensors.index.json', {'weight_map': {
        'a': 'model-00001-of-00002.safetensors', 'b': 'model-00002-of-00002.safetensors'}})
    assert not checkpoint_prepared(tmp_path)
    (tmp_path/'model-00002-of-00002.safetensors').write_bytes(b'fixture')
    assert checkpoint_prepared(tmp_path)
    (tmp_path/'config.json').write_text('broken')
    with pytest.raises(ValueError):
        checkpoint_prepared(tmp_path)


def test_pipeline_child_failure_closes_only_owned_sessions(tmp_path, monkeypatch):
    import classic_precision.pipeline as integrated
    m = write_manifest(tmp_path/'manifest.json')
    processes, stopped = {}, []
    class Process:
        def __init__(self, command, **kwargs):
            assert kwargs['start_new_session'] and kwargs['env']['OMP_NUM_THREADS'] == '1'
            self.pid, self.returncode = 90000+len(processes), None
            processes[self.pid] = self
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            return self.returncode
    monkeypatch.setattr(integrated.subprocess, 'Popen', Process)
    monkeypatch.setattr(integrated, 'allocation', lambda *a, **k: {})
    monkeypatch.setattr(integrated, 'process_owner', lambda pid: {'pid': pid, 'startticks': str(pid)})
    monkeypatch.setattr(integrated, 'owned_group_members', lambda owner:
                        [(owner['pid'], owner['pid'])] if processes[owner['pid']].returncode is None else [])
    def cleanup(owner, **kwargs):
        stopped.append(owner['pid'])
        processes[owner['pid']].returncode = -15
        return True
    monkeypatch.setattr(integrated, 'terminate_owned_session', cleanup)
    def advance(seconds):
        processes[min(processes)].returncode = 7
    monkeypatch.setattr(integrated.time, 'sleep', advance)
    with pytest.raises(RuntimeError, match='child failed'):
        integrated.pipeline(tmp_path, tmp_path/'models', [('0', 0), ('1', 1)])
    assert stopped == [max(processes)]
    status = json.loads((tmp_path/'pipeline.json').read_text())
    assert status['status'] == 'FAILED' and status['owned_sessions_stopped']
    assert sorted(p.returncode for p in processes.values()) == [-15, 7]


def test_owned_session_cleanup_after_leader_exit_preserves_foreign_session(tmp_path):
    import os
    import signal
    import subprocess
    import sys
    import time
    from classic_precision.pipeline import process_owner, terminate_owned_session, owned_group_members
    child_script = 'import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)'
    script = ('import subprocess,sys,time,os;'
              'subprocess.Popen([sys.executable,"-c",'+repr(child_script)+']);'
              'time.sleep(.2);os._exit(0)')
    leader = subprocess.Popen([sys.executable, '-c', script], start_new_session=True)
    owner = process_owner(leader.pid)
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    try:
        leader.wait(timeout=3)
        assert owned_group_members(owner)
        assert terminate_owned_session(owner, grace_seconds=.1)
        assert not owned_group_members(owner)
        assert foreign.poll() is None
    finally:
        terminate_owned_session(owner, grace_seconds=.1)
        os.killpg(foreign.pid, signal.SIGTERM)
        foreign.wait(timeout=3)
