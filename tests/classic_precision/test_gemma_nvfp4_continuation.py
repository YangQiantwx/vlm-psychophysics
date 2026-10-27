import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch
from classic_precision import gemma_nvfp4_continuation as n

class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.q={'pause_file':str(self.root/'PAUSE.json'),'worker_token':'a','maximum_model_calls':3,
                'assigned_ids':['x','y','z'],'assignment_sha256':'batch'}
    def test_pause_before_request(self):
        Path(self.q['pause_file']).write_text(json.dumps({'worker_token':'a','action':'PAUSE'}))
        b=n.Boundary(self.q);self.assertTrue(b.check())
    def test_foreign_pause_rejected(self):
        Path(self.q['pause_file']).write_text(json.dumps({'worker_token':'b','action':'PAUSE'}))
        with self.assertRaises(Exception):n.Boundary(self.q).check()
    def test_deadline(self):
        b=n.Boundary(dict(self.q,local_monotonic_deadline=2))
        with patch.object(n.time,'monotonic',return_value=3):self.assertTrue(b.check())
    def test_not_due(self):
        b=n.Boundary(dict(self.q,local_monotonic_deadline=4))
        with patch.object(n.time,'monotonic',return_value=3):self.assertFalse(b.check())
    def test_signal_only_sets_flag(self):
        b=n.Boundary(self.q);b.signal(signal.SIGTERM,None);self.assertTrue(b.requested)
    def items(self,b):
        class Ref:
            def expected(self,sid,deep):return {}
        return n.AssignedItems(self.q,{'items':{x:{'sample_id':x} for x in self.q['assigned_ids']}},Ref(),b)
    def test_pause_drains_current_no_next(self):
        b=n.Boundary(self.q);items=self.items(b);it=iter(items)
        self.assertEqual(next(it)['sample_id'],'x');b.signal()
        self.assertEqual(list(it),[]);self.assertEqual(items.ids,['x']);self.assertEqual(items.done['event'],'PAUSED')
    def test_complete_order(self):
        items=self.items(n.Boundary(self.q));self.assertEqual([x['sample_id'] for x in items],['x','y','z']);self.assertEqual(items.done['event'],'DONE')
    def test_no_implicit_resume_on_flag_removal(self):
        b=n.Boundary(self.q);b.signal();self.assertTrue(b.check());self.assertTrue(b.check())
    def test_frozen_sources(self):self.assertEqual(n.sources()['gemma_speed.py'],n.p.BASE_SHA)
    def test_derive_runner_protected_calls(self):self.assertGreater(len(n.p.derive_runner()['changes']),10)
    def test_baseline_no_offload_or_kv_change(self):
        s=n.settings({'checkpoint':'a','processor_checkpoint':'b'},'official-native')
        self.assertEqual(s['cpu_offload_gb'],0);self.assertEqual(s['kv_cache_dtype'],'bfloat16');self.assertEqual(s['max_model_len'],65536);self.assertEqual(s['max_num_seqs'],1)
    def test_compiled_changes_execution_only(self):
        r={'checkpoint':'a','processor_checkpoint':'b'};a=n.settings(r,'qualified-o0');b=n.settings(r,'compiled-o0')
        self.assertEqual({k for k in a if a[k]!=b[k]},{'enforce_eager'})
    def test_unknown_candidate(self):
        with self.assertRaises(Exception):n.settings({'checkpoint':'a','processor_checkpoint':'b'},'fp8-kv')
    def test_claim_duplicate_refused(self):
        p=self.root/'claim.json';n.p._write_new(p,{'ID':'x'})
        with self.assertRaises(FileExistsError):n.p._write_new(p,{'ID':'x'})
    def test_uncertain_resume_rejected(self):
        (self.root/'identity.json').write_text('{}');(self.root/'continuation.json').write_text(json.dumps({'status':'UNCERTAIN_OR_FAILED'}))
        with self.assertRaises(Exception):n.reconcile(self.root)
    def fixture(self,status='PAUSED'):
        q={'x':1};h=n.g.digest(q)
        (self.root/'identity.json').write_text(json.dumps({'profile':q,'profile_sha256':h}))
        raw={'sample_id':'x','entry_attempt_id':'a','finish_reason':'stop','raw_answer':'LEFT','output_token_ids':[1,2]}
        event=[{'event':'GENERATE_INTENT','sample_id':'x','attempt_id':'a'},
               {'event':'TECHNICAL_PASS','sample_id':'x','status':'PASS'}]
        b=(json.dumps(raw)+'\n').encode();e=('\n'.join(map(json.dumps,event))+'\n').encode()
        (self.root/'raw.jsonl').write_bytes(b);(self.root/'events.jsonl').write_bytes(e)
        call={'sample_id':'x','attempt_id':'a','ordinal':0,'raw_start':0,'raw_end':len(b),
              'raw_sha256':n.g.hashlib.sha256(b).hexdigest(),'prefix_end':0,'event_start':0,'event_end':len(e),
              'event_sha256':n.g.hashlib.sha256(e).hexdigest()}
        (self.root/'calls.jsonl').write_text(json.dumps(call)+'\n')
        (self.root/'exit.json').write_text(json.dumps({'status':'COMPLETE','owned_engine_workers_exited':True,'lease_released_after_engine_exit':True}))
        (self.root/'continuation.json').write_text(json.dumps({'status':status,'profile_sha256':h,'completed_ids':['x'],
            'uncertain_id':None,'unattempted_ids':['y'],'assigned_ids':['x','y']}))
    def test_pause_reconciles_only_unattempted(self):
        self.fixture();r=n.reconcile(self.root);self.assertEqual(r['remaining_ids'],['y']);self.assertFalse(r['automatic_resume'])
    def test_changed_raw_rejected(self):
        self.fixture();p=self.root/'raw.jsonl';p.write_bytes(p.read_bytes().replace(b'LEFT',b'RIGT'))
        with self.assertRaises(Exception):n.reconcile(self.root)
    def test_uncommitted_tail_rejected(self):
        self.fixture();p=self.root/'events.jsonl';p.write_text(p.read_text()+'{}\n')
        with self.assertRaises(Exception):n.reconcile(self.root)
    def test_live_engine_not_resumable(self):
        self.fixture();p=self.root/'exit.json';x=json.loads(p.read_text());x['owned_engine_workers_exited']=False;p.write_text(json.dumps(x))
        with self.assertRaises(Exception):n.reconcile(self.root)
    def test_claim_gap_rejected(self):
        self.fixture();p=self.root/'continuation.json';x=json.loads(p.read_text());x['unattempted_ids']=[];p.write_text(json.dumps(x))
        with self.assertRaises(Exception):n.reconcile(self.root)
    def test_cpu_import_no_model(self):self.assertNotIn('torch',sys.modules);self.assertNotIn('vllm',sys.modules)

if __name__=='__main__':unittest.main()
