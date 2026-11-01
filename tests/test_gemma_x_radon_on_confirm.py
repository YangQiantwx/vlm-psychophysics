import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from classic_precision import gemma_x_radon_on_confirm as m
from classic_precision.refinement_worker import descriptor
from classic_precision.storage import digest, file_sha256

B = Path('/Users/qianyang/Desktop/cvpr2027_benchmark')


class ConfirmationTests(unittest.TestCase):
    def fixture(self):
        arc = m.core.full.Archive(B)
        source = B/'plans/gemma-x-full-20261030/radon0-qualification-cold1-profile.json'
        p = json.loads(source.read_text());root = Path(p['workspace_root']);w='radon0-recheck-cold1'
        ref=descriptor(B/'plans'/m.BASE/'reference.json')
        pred={'path':str(root/'plans'/m.BASE/'autotune-off3/radon0-recheck-queue.json'),'sha256':m.PREDECESSOR_SHA}
        auth={'status':'EXPLICIT_USER_AUTHORIZED_ON_REPEAT','independent_new_call_budget':96,'cold_runs':1,
              'configuration_changes_from_original_ON':[],'preserve_all_original_answers':True,'point_deg':1.5,
              'reference':ref,'predecessor_queue':pred,'stop_on_first_same_configuration_output_difference':True}
        ar={'path':'new-authority','sha256':digest(auth)}
        p.update(worker=w,output=str(root/'runs/reproducibility'/m.NAME/w),
                 claim_root=str(root/'runs/reproducibility'/m.NAME/'claims'/w),
                 control_file=str(root/'plans'/m.NAME/(w+'-control.json')),execution_authorization=ar)
        p['recheck']={'operations':descriptor(m.__file__),'trace':descriptor(m.trace.__file__),
            'trace_v2':descriptor(m.v2.__file__),'observer_safe':descriptor(m.observer.__file__),
            'template_profile':descriptor(source),'reference':ref,'predecessor_queue':pred}
        def resolve(value):
            if value is ar or value.get('path')=='new-authority':return auth
            path=arc.path(value['path']);self.assertEqual(file_sha256(path),value['sha256']);return json.loads(path.read_text())
        return p,auth,resolve

    def test_existing_ON_science_and_actual_new_authority_remain_intact(self):
        p,auth,resolve=self.fixture();before=copy.deepcopy(p)
        with patch.object(m.core.qual,'verify_code'):m.science(p,resolve)
        self.assertEqual(p,before);self.assertEqual(auth['independent_new_call_budget'],96)

    def test_reject_real_config_and_budget_changes(self):
        for change in ['off','kv','context','seed','count','duplicate','cold','budget','observer','namespace']:
            with self.subTest(change=change):
                p,a,resolve=self.fixture()
                if change=='off':p['engine_kwargs']['kernel_config']={'enable_flashinfer_autotune':False}
                if change=='kv':p['engine_kwargs']['kv_cache_memory_bytes']=44*1024**3
                if change=='context':p['engine_kwargs']['max_model_len']=98304
                if change=='seed':p['engine_kwargs']['seed']=42
                if change=='count':p['maximum_model_calls']=97
                if change=='duplicate':p['assigned_ids'][-1]=p['assigned_ids'][0]
                if change=='cold':p['cold']=2
                if change=='budget':a['independent_new_call_budget']=192
                if change=='observer':p['recheck']['observer_safe']['sha256']='bad'
                if change=='namespace':p['claim_root']='/old/claims'
                with patch.object(m.core.qual,'verify_code'),self.assertRaises(ValueError):m.science(p,resolve)

    def closed(self,code=0):
        def worker(suffix):
            ids=[suffix];sha='profile'+suffix
            return {'started':True,'claims':1,'raw':1,'same_inode_lease_free':True,'live_owner_count':0,
                'profile':{'sha256':'file-'+sha},'profile_digest':sha,'assigned_ids':ids,'claimed_ids':ids,'result_status':'COMPLETE',
                'wait':{'real_wait_observed':True,'returncode':0,'owned_engines_exited':True,
                    'original_physical_lease_released':True,'profile_sha256':sha,'completed_ids':ids}}
        return {'outer_wait':{'real_wait_observed':True,'returncode':code},
                'outer_owner':{'status':'OWNER_PROCESS_EXITED','alive':False},'workers':[worker('1'),worker('2')]}

    def test_complete_and_failed_but_really_closed_predecessors(self):
        m.validate_terminal(self.closed())
        d=self.closed(1);d['workers'][1]['wait']['returncode']=1;d['workers'][1]['result_status']='FAILED'
        m.validate_terminal(d)
        d['workers'][1].update(started=False,claims=0,raw=0,wait=None);m.validate_terminal(d)

    def test_reject_incomplete_or_live_closure(self):
        for field in ['outer_live','outer_no_wait','worker_live','lease','no_real_wait','profile','missing_completed','unstarted_claim','all_unstarted']:
            with self.subTest(field=field):
                d=self.closed();w=d['workers'][1]
                if field=='outer_live':d['outer_owner']['alive']=True
                if field=='outer_no_wait':d['outer_wait']['real_wait_observed']=False
                if field=='worker_live':w['live_owner_count']=1
                if field=='lease':w['same_inode_lease_free']=False
                if field=='no_real_wait':w['wait']['real_wait_observed']=False
                if field=='profile':w['wait']['profile_sha256']='bad'
                if field=='missing_completed':w['wait']['completed_ids']=[]
                if field=='unstarted_claim':w.update(started=False,wait=None)
                if field=='all_unstarted':
                    for w in d['workers']:w.update(started=False,claims=0,raw=0,wait=None)
                with self.assertRaises(ValueError):m.validate_terminal(d)

    def test_namespace_restores_functions_even_on_error(self):
        old=(m.core.NAME,m.core.MODULE,m.core.science,m.core.queue_check,m.core.qual.queue,m.trace.install,m.trace.finish)
        with self.assertRaises(ValueError):
            with m.namespace():
                self.assertEqual(m.core.MODULE,m.MODULE);self.assertIs(m.core.qual.queue,m.queue)
                self.assertIs(m.trace.install,m.v2.install);self.assertIs(m.trace.finish,m.observer.finish)
                raise ValueError('sentinel')
        self.assertEqual(old,(m.core.NAME,m.core.MODULE,m.core.science,m.core.queue_check,m.core.qual.queue,m.trace.install,m.trace.finish))

    def test_one_worker_only_and_real_supervisor_failure_stops(self):
        p={'control_file':'control'};q={'status':'SEALED','profiles':[{'path':'profile'}]}
        wait={'returncode':0,'terminal_status':'COMPLETE','real_wait_observed':True,
              'owned_engines_exited':True,'original_physical_lease_released':True}
        with patch.object(m,'queue_check',return_value=(q,[p])),patch.object(m,'read',return_value={'paused':False}),patch.object(m.core,'supervise',return_value=wait) as run:
            self.assertEqual(m.queue('q')['status'],'COMPLETE_ON_CONFIRMATION_QUEUE');run.assert_called_once_with('profile')
            wait['returncode']=1
            with self.assertRaises(ValueError):m.queue('q')

    def test_used_queue_rejected_before_predecessor_or_resource(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'intent';path.write_text('{}')
            with patch.object(m.core,'paths',return_value={'launch-intent':path}),patch.object(m,'predecessor') as predecessor,patch.object(m.recovery,'resource_window') as resource:
                with self.assertRaises(ValueError):m.launch(t)
                predecessor.assert_not_called();resource.assert_not_called()


if __name__=='__main__':unittest.main()
