from contextlib import contextmanager
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from classic_precision import refinement_worker as w
from classic_precision.manifest import load_protocol, make_manifest
from classic_precision.refinement import DELTA_LEVELS, ANGLE_GRID
from classic_precision.refinement_dispatch import allocation


def panel():
    c = load_protocol(); c['models'] = {}
    c['T2']['delta_Y'] = c['T2']['control_10frame_delta_Y'] = list(DELTA_LEVELS)
    c['T3']['angle_degrees'] = [a for a in ANGLE_GRID if a not in (1.,1.5,2.,2.5)]
    c['counts'].update(T2_total=2016,T3_total=1152)
    return make_manifest(c)


def test_allocation_keeps_all_lr_sign_units_and_budget():
    m = panel(); a = allocation(m); assigned = a['assignments']
    assert sum(map(len,assigned.values())) == 12672
    sets = [set(assigned['gemma-'+k]) for k in ('fourier0','fourier1','radon0')]
    assert not any(sets[i] & sets[j] for i in range(3) for j in range(i))
    assert set.union(*sets) == {r['sample_id'] for r in m['rows']}
    owners = {sid:k for k,ids in enumerate(sets) for sid in ids}
    units = {}
    for r in m['rows']:
        p = {k:v for k,v in r['source_parameters'].items() if k not in ('reference_side','target_side','tilt_sign')}
        units.setdefault(w.digest(p),set()).add(owners[r['sample_id']])
    assert all(len(v)==1 for v in units.values())


def test_split_unit_is_rejected():
    m = panel();m['rows'].pop()
    with pytest.raises(ValueError,match='contrast unit'):
        allocation(m)


def fixture_worker(tmp_path, monkeypatch, after_raw=None):
    from classic_precision import inference, gemma_speed
    rows = panel()['rows'][:3]
    items = {r['sample_id']:{'row':r,'ready':{'video_sha256':'a'*64},'source_family':'classic-refinement',
        'source_audit_identity':'b'*64,'source_stream_sha256':'c'*64,'row_sha256':w.digest(r),'ready_sha256':'d'*64} for r in rows}
    catalog=tmp_path/'catalog.json';w.write_new(catalog,{'items':items})
    profile={'schema':'classic-refinement-worker-v1','model':'nemo','worker':'test',
        'assigned_ids':list(items),'output':str(tmp_path/'output'),'checkpoint':'unused',
        'checkpoint_assets':{},'resources':{},'implementation':{'refinement_worker.py':w.file_sha256(w.__file__)},
        'source_root':str(tmp_path),'worker_token':'unique','claim_root':str(tmp_path/'claims'),
        'pause_file':str(tmp_path/'PAUSE.json'),'catalog':w.descriptor(catalog)}
    boundary=w.Boundary(profile)
    class Session:
        def __init__(self, model,path,sink,resources):
            self.context={};self.sink=sink;self.engine=SimpleNamespace(collective_rpc=lambda fn:[])
        def generate(self,row,ready,root,commit):
            result=commit({'raw_answer':'RIGHT','output_token_ids':[1,2],'truth':row['truth'],
                'finish_reason':'stop','stop_reason':None,'task':row['task']})
            self.sink.event('TECHNICAL_PASS',sample_id=row['sample_id'],attempt_id=result['attempt_id'])
            if after_raw:after_raw(boundary)
        def close(self):pass
    @contextmanager
    def lease(*_):yield {'test_only':True}
    monkeypatch.setattr(w,'physical_lease',lease)
    monkeypatch.setattr(inference,'ModelSession',Session)
    monkeypatch.setattr(inference,'checkpoint_asset_identity',lambda _: {})
    monkeypatch.setattr(gemma_speed,'shutdown_owned_engine',lambda *args:None)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace())
    return profile,boundary


def test_pause_drains_raw_and_never_claims_next_id(tmp_path,monkeypatch):
    p,b=fixture_worker(tmp_path,monkeypatch,lambda b:b.signal())
    result=w.generic_run(p,b)
    assert result['status']=='PAUSED' and result['completed_ids']==p['assigned_ids'][:1]
    assert len(list((tmp_path/'claims').glob('*.json')))==1
    assert len((tmp_path/'output/nemo/outputs.jsonl').read_text().splitlines())==1


def test_post_raw_error_keeps_answer_and_claim_no_retry(tmp_path,monkeypatch):
    def fail(_):raise RuntimeError('post raw failure')
    p,b=fixture_worker(tmp_path,monkeypatch,fail)
    with pytest.raises(RuntimeError,match='post raw'):
        w.generic_run(p,b)
    raw=json.loads((tmp_path/'output/nemo/outputs.jsonl').read_text())
    assert raw['sample_id']==p['assigned_ids'][0] and raw['attempt_number']==1
    with pytest.raises(FileExistsError):w.claim(p,p['assigned_ids'][0])
    with pytest.raises(ValueError,match='No automatic worker resume'):w.generic_run(p,b)
