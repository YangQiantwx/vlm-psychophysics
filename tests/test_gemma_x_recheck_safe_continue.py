import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

from classic_precision import gemma_x_recheck_safe_continue as m


class SafeContinuationTests(unittest.TestCase):
    def fixture(self):
        b=Path('/Users/qianyang/Desktop/cvpr2027_benchmark');r=b/'runs/reproducibility'/m.BASE
        return (json.loads((r/'autotune-off2-recorder-failure-closed.json').read_text()),
                json.loads((r/'autotune-off2-retained-first-answer.json').read_text()),
                json.loads((b/'plans'/m.BASE/'autotune-off2/radon0-recheck-cold1-profile.json').read_text()))

    def test_nonfinite_metadata_serializes_without_changing_original(self):
        value={'top8':[1.0,-math.inf,math.inf,math.nan],'binary_sha':'abc'}
        safe=m.safe.json_safe(value)
        self.assertEqual(safe['top8'],[1.0,'-Infinity','Infinity','NaN'])
        json.dumps(safe,allow_nan=False)
        self.assertEqual(value['top8'][1],-math.inf)
        self.assertEqual(safe['binary_sha'],'abc')

    def test_native_prepare_does_not_import_qwen_or_replace_runner_hooks(self):
        from classic_precision import official_x_runner as runner
        keys=['_append','worker_identity','install_probe','read_probe','validate_deep']
        before={k:getattr(runner,k) for k in keys}
        with tempfile.TemporaryDirectory() as tmp:
            frames=np.zeros((2,2,2,3),dtype=np.uint8)
            result={'request':{'multi_modal_data':{'video':[(frames,{})]}},'receipt':{}}
            o=SimpleNamespace(root=Path(tmp),p={},original=SimpleNamespace(prepare=lambda *x:result),
                engine=SimpleNamespace(collective_rpc=lambda *a,**kw:None))
            with patch.object(m.safe,'shared_observation',return_value={}),patch.object(m.safe,'resource_observation',return_value={}):
                self.assertIs(m.safe.prepare_observed(o,{'sample_id':'s'},[]),result)
            self.assertEqual(before,{k:getattr(runner,k) for k in keys})
            self.assertNotIn('classic_precision.qwen_kv_shared_smoke',sys.modules)

    def test_both_card_namespaces_keep_native_recorder_and_restore(self):
        before=m.trace.finish,m.safe.ORIGINAL_ADAPTER.prepare,m.off.cold1_reference,m.repair.name
        for card in ['radon0','fourier0','fourier1']:
            with m.namespace(card):
                self.assertIs(m.trace.finish,m.safe.finish)
                self.assertIs(m.safe.ORIGINAL_ADAPTER.prepare,m.safe.prepare_observed)
                self.assertEqual(m.core.NAME,m.name(card))
                self.assertEqual(m.core.MODULE,m.MODULE)
                self.assertIs(m.core.science,m.science)
                if card=='radon0':
                    with m.recovery.namespace():
                        self.assertEqual(m.core.MODULE,m.MODULE)
                        self.assertIs(m.core.science,m.science)
            self.assertEqual(before,(m.trace.finish,m.safe.ORIGINAL_ADAPTER.prepare,m.off.cold1_reference,m.repair.name))

    def test_exact_saved_answer_closure_and_zero_extra_budget(self):
        c,p,d=self.fixture();m.validate_prefix(c,p,d)
        for change in ['calls','claim','input','fence','wait','identity']:
            cc,pp=copy.deepcopy(c),copy.deepcopy(p)
            if change=='calls':cc['model_calls']=2
            if change=='claim':cc['workers'][1]['claims']=1
            if change=='input':pp['actual_CPU_NATIVE_GPU_180_bytes_equal']=False
            if change=='fence':cc['controls'][0]['after']['paused']=False
            if change=='wait':cc['workers'][0]['real_wait']['owned_engines_exited']=False
            if change=='identity':pp['raw']['profile_sha256']='different'
            with self.subTest(change=change),self.assertRaises(ValueError):m.validate_prefix(cc,pp,d)

    def test_remainder_excludes_the_claimed_id_and_projects_validation_only(self):
        closed,prefix,donor=self.fixture()
        for cold in [1,2]:
            p=copy.deepcopy(donor);p['cold']=cold
            p['assigned_ids']=donor['assigned_ids'][1 if cold==1 else 0:]
            p['maximum_model_calls']=len(p['assigned_ids'])
            p['recheck']['safe_observer']={'entrypoint':{'sha256':m.file_sha256(m.__file__)},
                'observer':{'sha256':m.file_sha256(m.safe.__file__)},'scientific_changes_from_donor':[],
                'donor_closed':'c','retained_prefix':'p','prefix_profile':'d'}
            resolve=lambda ref:{'c':closed,'p':prefix,'d':donor}[ref]
            original=copy.deepcopy(p)
            with patch.object(m,'SCIENCE') as science:
                m.science(p,resolve)
                self.assertEqual(science.call_args.args[0]['assigned_ids'],donor['assigned_ids'])
                self.assertEqual(science.call_args.args[0]['maximum_model_calls'],96)
            self.assertEqual(p,original)
            p['assigned_ids']=p['assigned_ids'][:-1]
            with self.assertRaises(ValueError):m.science(p,resolve)

    def test_second_group_requires_complete95_and_retained1_exactly_once(self):
        c,p,d=self.fixture();spec={'retained_prefix':'p','prefix_profile':'d','donor_closed':'c'}
        records={sid:{'sample_id':sid} for sid in d['assigned_ids'][1:]}
        with patch.object(m,'verified',side_effect=lambda x:{'c':c,'p':p,'d':d}[x]),patch.object(m,'COLD1_REFERENCE',return_value=records):
            combined=m.cold1_reference({'recheck':{'safe_observer':spec}})
            self.assertEqual(len(combined),96)
            self.assertIs(combined[d['assigned_ids'][0]],p['raw'])
            with self.assertRaises(ValueError):m.cold1_reference({'recheck':{'safe_observer':spec}})


if __name__=='__main__':unittest.main()
