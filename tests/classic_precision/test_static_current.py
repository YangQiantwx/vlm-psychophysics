import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from classic_precision import static_current as s


class CurrentStaticTests(unittest.TestCase):
    def setUp(self):
        self.manifest={'images':[{'image_id':str(i)} for i in range(152)],'rows':[
            {'sample_id':f'{i}-{p}','task':'T2' if i<78 else 'T3','prompt_id':p}
            for i in range(152) for p in ['P0','P1','P2']]}
    def test_exact_full_scope_and_first_two_tasks(self):
        rows=s.ordered_rows(self.manifest)
        self.assertEqual(len(rows),456);self.assertEqual([x['task'] for x in rows[:2]],['T2','T3'])
        self.assertEqual(set(x['sample_id'] for x in rows),set(x['sample_id'] for x in self.manifest['rows']))
    def test_duplicate_refused(self):
        self.manifest['rows'][0]=copy.deepcopy(self.manifest['rows'][1])
        with self.assertRaises(Exception):s.ordered_rows(self.manifest)
    def test_missing_refused(self):
        self.manifest['rows'].pop()
        with self.assertRaises(Exception):s.ordered_rows(self.manifest)
    def test_task_factor_refused(self):
        self.manifest['rows'][0]['task']='T1'
        with self.assertRaises(Exception):s.ordered_rows(self.manifest)
    def test_prompt_factor_refused(self):
        self.manifest['rows'][0]['prompt_id']='P1'
        with self.assertRaises(Exception):s.ordered_rows(self.manifest)
    def test_image_coverage_refused(self):
        self.manifest['images'].pop()
        with self.assertRaises(Exception):s.ordered_rows(self.manifest)
    def test_backend_settings_only_modality_changes(self):
        rt={'checkpoint':'nv4','original_checkpoint':'bf16'};a=s.o.engine_settings(rt);b=s.settings(rt)
        self.assertEqual({k for k in a if a[k]!=b[k]},{'limit_mm_per_prompt','mm_processor_kwargs'})
        self.assertEqual(b['kv_cache_dtype'],'bfloat16');self.assertEqual(b['kv_cache_memory_bytes'],17*2**30)
    def test_frozen_backend(self):self.assertEqual(s.file_sha256(s.o.__file__),s.COSMOS_SHA)


class QwenSealTests(unittest.TestCase):
    def profile(self):return {'mode':'qwen_qsa_only','qsa_policy':'reviewed_exact_selector_only',
                             'environment':{'CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB':'16'}}
    def call(self,profile):
        import tempfile,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,value in [('execution-manifest.json',{}),('protocol.json',{}),('native.json',profile),('video.json',{'runtime':{}})]:
                (root/name).write_text(json.dumps(value))
            for name in ['static_inputs.py','static_worker.py']:(root/name).write_text('source')
            (root/'coverage.json').write_text(json.dumps({'routes':[{'route_id':'single','implementation_files':[
                {'path':str(root/n),'sha256':s.file_sha256(root/n)} for n in ['static_inputs.py','static_worker.py']]}]}))
            args=SimpleNamespace(static_root=str(root),static_code=str(root),native_profile=str(root/'native.json'),
                 video_plan=str(root/'video.json'),coverage=str(root/'coverage.json'),output=str(root/'out.json'),gpu=0)
            with patch.object(s,'ordered_rows',return_value=[{'sample_id':'a'},{'sample_id':'b'}]):
                result=s.seal_qwen(args)
            return result,json.loads((root/'out.json').read_text())
    def test_source_bound_native_image_placement_retained(self):
        result,q=self.call(self.profile())
        self.assertEqual(result['GPU_calls'],0);self.assertEqual(q['maximum_calls'],456)
        self.assertFalse(q['automatic_retry']);self.assertEqual(q['first_deep_calls'],2)
        self.assertEqual(q['native_profile']['environment']['CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB'],'16')
        self.assertTrue(q['qsa_helper']['sha256'])
    def test_wrong_selector_refused(self):
        p=self.profile();p['qsa_policy']='generic'
        with self.assertRaises(Exception):self.call(p)
    def test_wrong_mode_refused(self):
        p=self.profile();p['mode']='strict_all'
        with self.assertRaises(Exception):self.call(p)
    def test_unqualified_placement_refused(self):
        p=self.profile();p['environment']['CLASSIC_PRECISION_QWEN_CPU_OFFLOAD_GB']='24'
        with self.assertRaises(Exception):self.call(p)


if __name__=='__main__':unittest.main()
