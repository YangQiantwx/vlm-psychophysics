import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from classic_precision import gemma_x_recheck_recovery as recovery
from classic_precision.storage import digest, file_sha256


def closure():
    return {'status':'ZERO_CALL_STARTUP_RESOURCE_FAILURE_CLOSED_FENCED',
        'new_model_calls':0,'model_load_started':False,
        'outer_owner':{'status':'OWNER_PROCESS_EXITED','alive':False},
        'outer_real_wait':{'returncode':1,'real_wait_observed':True},
        'workers':[{'worker':f'radon0-recheck-cold{i}','claims':0,'raw':0,
                    'scientific_INTENT':0,'model_load_started':False,
                    'original_same_inode_lease_free':True,
                    'real_wait':({'returncode':1,'real_wait_observed':True,
                                 'owned_engines_exited':True,'original_physical_lease_released':True,
                                 'completed_ids':[]} if i==1 else None)} for i in [1,2]],
        'controls':[{'after':{'paused':True,'generation':1}} for _ in range(2)]}


class RecoveryTests(unittest.TestCase):
    def test_only_known_zero_call_closed_donor_is_accepted(self):
        recovery.validate_closure(closure())
        for field in ['claims','raw','scientific_INTENT','model_load_started']:
            v=closure();v['workers'][0][field]=1
            with self.subTest(field=field), self.assertRaises(ValueError):recovery.validate_closure(v)
        for mutate in [lambda v:v['outer_owner'].update(alive=True),
                       lambda v:v['outer_real_wait'].update(real_wait_observed=False),
                       lambda v:v['workers'][0]['real_wait'].update(owned_engines_exited=False),
                       lambda v:v['workers'][1].update(real_wait={}),
                       lambda v:v['controls'][0]['after'].update(generation=0)]:
            v=closure();mutate(v)
            with self.assertRaises(ValueError):recovery.validate_closure(v)

    def test_transient_resource_peak_cannot_admit_window(self):
        p={'gpu_uuid':'GPU-test','resources':{'min_free_gpu_gib':80,'min_available_ram_gib':20}}
        w={'profile_sha256':digest(p),'boot_id':'boot','samples':[
            {'monotonic':t,'gpu_uuid':'GPU-test','free_gpu_gib':94,'available_ram_gib':100}
            for t in [0,10,21]]}
        self.assertTrue(recovery.window_admitted(w,p,22,'boot'))
        for mutate in [lambda v:v['samples'][1].update(free_gpu_gib=78),
                       lambda v:v['samples'][1].update(available_ram_gib=19),
                       lambda v:v['samples'][1].update(gpu_uuid='other'),
                       lambda v:v.update(boot_id='other'),
                       lambda v:v.update(profile_sha256='other'),
                       lambda v:v['samples'][-1].update(monotonic=19)]:
            v=copy.deepcopy(w);mutate(v)
            self.assertFalse(recovery.window_admitted(v,p,22,'boot'))
        self.assertFalse(recovery.window_admitted(w,p,82,'boot'))

    def test_recovery_does_not_relax_original_science(self):
        donor={'card':'radon0','cold':1,'assigned_ids':list(range(96)),
               'engine_kwargs':{'kv_cache_memory_bytes':48*2**30,'max_model_len':65536},
               'output':'old','recheck':{}}
        p=copy.deepcopy(donor);p['output']='new';p['recheck']['startup_recovery']={
            'entrypoint':{'sha256':file_sha256(recovery.__file__)},'closure':{'key':'closed'},
            'donor_profile':{'key':'donor'}}
        resolve=lambda ref:closure() if ref['key']=='closed' else donor
        with patch.object(recovery,'ORIGINAL_SCIENCE') as original:
            recovery.science(p,resolve);original.assert_called_once_with(p,resolve)
        for mutate in [lambda v:v['engine_kwargs'].update(kv_cache_memory_bytes=44*2**30),
                       lambda v:v['engine_kwargs'].update(max_model_len=98304),
                       lambda v:v.update(assigned_ids=list(range(95))),
                       lambda v:v.update(card='fourier0')]:
            v=copy.deepcopy(p);mutate(v)
            with patch.object(recovery,'ORIGINAL_SCIENCE') as original:
                with self.assertRaises(ValueError):recovery.science(v,resolve)
                original.assert_not_called()

    def test_resource_observation_returns_original_value_and_calls_once(self):
        result={'admitted':False,'free_gpu_gib':78.6,'available_ram_gib':104}
        original=Mock(return_value=result);marker=object();budgets={'min_free_gpu_gib':80}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'resource.jsonl';wrapped=recovery.recorded_snapshot(original,p)
            self.assertIs(wrapped(marker,budgets),result)
            original.assert_called_once_with(marker,budgets)
            self.assertEqual(json.loads(p.read_text())['snapshot'],result)
            failed=Mock(side_effect=RuntimeError('source failed'))
            with self.assertRaises(RuntimeError):recovery.recorded_snapshot(failed,Path(d)/'absent')(marker,budgets)
            self.assertFalse((Path(d)/'absent').exists())

    def test_namespace_restores_frozen_core_on_failure(self):
        old=(recovery.core.NAME,recovery.core.MODULE,recovery.core.science)
        with self.assertRaises(RuntimeError):
            with recovery.namespace():
                self.assertEqual(recovery.core.NAME,recovery.NAME)
                self.assertEqual(recovery.core.MODULE,recovery.MODULE)
                self.assertIs(recovery.core.science,recovery.science)
                raise RuntimeError('stop')
        self.assertEqual((recovery.core.NAME,recovery.core.MODULE,recovery.core.science),old)


if __name__=='__main__':unittest.main()
