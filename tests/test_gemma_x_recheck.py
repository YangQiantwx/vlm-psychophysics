"""CPU checks of observation identity, finite admission and raw-first failure."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from classic_precision import gemma_x_recheck as op
from classic_precision import gemma_x_recheck_trace as trace
from classic_precision.storage import atomic_json, file_sha256


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.scale=torch.nn.Parameter(torch.tensor(0.999))
    def forward(self,x): return x*self.scale


class Vision(torch.nn.Module):
    def forward(self,x): return x[:,:1,:6].reshape(-1,6)


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.vision_tower=torch.nn.Module()
        self.vision_tower.patch_embedder=Vision()
        self.language_model=torch.nn.Module(); self.language_model.model=torch.nn.Module()
        self.language_model.model.layers=torch.nn.ModuleList([Block() for _ in range(60)])
    def forward(self,x):
        x=self.vision_tower.patch_embedder(x)
        for layer in self.language_model.model.layers:x=layer(x)
        return x


class Sampler(torch.nn.Module):
    def __init__(self): super().__init__();self.calls=0;self.last=None
    def greedy_sample(self,x):
        self.calls+=1;self.last=x.argmax(-1);return self.last
    def forward(self,logits):return self.greedy_sample(logits)


class Tuner:
    @classmethod
    def get(cls):return cls()
    def _winner_cache_identity(self):return None
    def _winner_cache(self):return {}
    def choose_one(self,custom_op,runners,tuning_config,inputs,**kwargs):return runners[0],-1


class Profiler:
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def export_chrome_trace(self,path):Path(path).write_text('{"traceEvents":[]}')


class RecheckTests(unittest.TestCase):
    def test_failed_final_closure_cannot_leave_a_pass_summary(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);plan=root/'plan.json';atomic_json(plan,{'workers':[{}]*6})
            old_name=op.qual.NAME;old_writer=op.qual.atomic_json
            def provisional(root_arg,plan_arg):
                op.qual.atomic_json(root/'runs/reproducibility'/op.NAME/'qualification-summary.json',
                    {'status':'PASS_PLACEHOLDER'})
            with patch.object(op.qual,'summarize',provisional):
                with self.assertRaises(Exception):op.summarize(root,plan)
            self.assertFalse((root/'runs/reproducibility'/op.NAME/'qualification-summary.json').exists())
            self.assertEqual(op.qual.NAME,old_name);self.assertIs(op.qual.atomic_json,old_writer)

    def test_tensor_bytes_are_exact_and_inputs_untouched(self):
        for dtype in [torch.bfloat16,torch.float32,torch.int64,torch.uint8]:
            x=torch.arange(36).reshape(6,6).to(dtype).T;before=x.clone()
            info,data=trace.tensor_blob(x)
            self.assertTrue(torch.equal(x,before));self.assertEqual(info['stride'],list(x.stride()))
            self.assertEqual(data,x.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        x=torch.zeros((2,trace.PATCHES,768),dtype=torch.bfloat16)
        info,data=trace.tensor_blob(x)
        self.assertEqual(len(info['per_frame_sha256']),2)
        self.assertEqual(info['per_frame_sha256'][0],hashlib.sha256(data[:len(data)//2]).hexdigest())

    def test_hooks_preserve_original_results_and_capture_decision(self):
        network=Network();sampler=Sampler();worker=SimpleNamespace(get_model=lambda:network,
            model_runner=SimpleNamespace(sampler=sampler))
        data=torch.arange(2*trace.PATCHES*768,dtype=torch.float32).reshape(2,trace.PATCHES,768)
        before=data.clone();baseline=network(data);expected=baseline.argmax(-1)
        module=ModuleType('flashinfer.autotuner');module.AutoTuner=Tuner
        parent=ModuleType('flashinfer');parent.autotuner=module
        original_descriptor=trace.descriptor
        def descriptor(path):
            if Path(path)==Path(__file__):return {'path':str(path),'sha256':'7d3d001b8b6e1f25f02572df29609a4ab5c50abb023b48f5ebdb4752a2b5f7c5'}
            return original_descriptor(path)
        with tempfile.TemporaryDirectory() as td, patch.dict('sys.modules',{'flashinfer':parent,'flashinfer.autotuner':module}),\
             patch.object(trace,'descriptor',descriptor),patch.object(torch.profiler,'profile',lambda **kw:Profiler()):
            trace.install(worker,td);trace.begin(worker,'sample')
            logits=network(data);logits_before=logits.clone();answer=sampler(logits)
            self.assertIs(answer,sampler.last);self.assertEqual(sampler.calls,1)
            self.assertTrue(torch.equal(logits,baseline));self.assertTrue(torch.equal(answer,expected))
            self.assertTrue(torch.equal(logits,logits_before));self.assertTrue(torch.equal(data,before))
            value=trace.finish(worker);self.assertEqual(value['greedy_steps'],1)
            self.assertEqual(len(value['GPU_pixel_frame_hashes']),2)
            proof=json.loads((Path(td)/'diagnostics/sample.json').read_text())
            self.assertEqual(len([x for x in proof['records'] if x['kind']=='decoder_last_token']),60)
            self.assertFalse(value['observation_errors']);trace.close(worker)
            a=json.loads((Path(td)/'diagnostics/after-load.json').read_text())
            b=json.loads((Path(td)/'diagnostics/before-exit.json').read_text())
            self.assertEqual(a['weights']['content_manifest_sha256'],b['weights']['content_manifest_sha256'])

    def test_observation_failure_is_recorded_without_propagating(self):
        state={'records':[]}
        def broken():raise MemoryError('simulated copy failure')
        self.assertIsNone(trace.observe(state,broken))
        self.assertEqual(state['records'][0]['kind'],'observation_error')

    def test_different_answer_is_retained_before_failure(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);raw={'raw_answer':'RIGHT','output_token_ids':[2], 'finish_reason':'stop',
                'stop_reason':None,'truth':'RIGHT','question':'q','generation':{'temperature':0},
                'actual_prompt_token_ids':[4,5]}
            expected={k:v for k,v in raw.items() if k!='actual_prompt_token_ids'}
            expected.update(raw_answer='LEFT',output_token_ids=[1])
            atomic_json(root/'raw.json',raw);before=(root/'raw.json').read_bytes()
            a=object.__new__(trace.AdapterObservation);a.original=SimpleNamespace(validate=lambda answer:{'pass':True})
            a.root=root;a.sid='s';a.state={'raw':raw};a.active=True
            a.reference={'by_id':{'s':expected},'actual_prompt_token_ids':[4,5]}
            a.cpu_tensors=[{'shape':[180,trace.PATCHES,768],'per_frame_sha256':['h']*180}]
            a.engine=SimpleNamespace(collective_rpc=lambda fn:[{'vision_chunks':1,'greedy_steps':1,
                'observation_errors':[],'GPU_pixel_frame_hashes':['h']*180}])
            with self.assertRaises(Exception):a.validate(None)
            self.assertEqual(before,(root/'raw.json').read_bytes())
            fail=json.loads((root/'reproduction-failure.json').read_text())
            self.assertTrue(fail['raw_persisted_before_comparison'])
            self.assertEqual(set(fail['different_fields']),{'raw_answer','output_token_ids'})

    def test_science_mutations_are_rejected(self):
        root=Path('/Users/qianyang/Desktop/cvpr2027_benchmark')
        if not root.exists():self.skipTest('Archive-bound admission exercised on preparation host')
        arc=op.full.Archive(root)
        template_ref=json.loads((root/'plans/gemma-x-full-20261030/qualification-plan-exec1.json').read_text())['workers'][0]['profile']
        template=arc.ref(template_ref);p=copy.deepcopy(template);remote=Path(p['workspace_root']);worker=p['card']+'-recheck-cold1'
        original=[json.loads(l) for l in (arc.path(template['output'])/'raw.jsonl').read_text().splitlines()]
        keys=['raw_answer','output_token_ids','finish_reason','stop_reason','truth','question','generation']
        reference={'by_id':{r['sample_id']:{k:r[k] for k in keys} for r in original},
                   'actual_prompt_token_ids':original[0]['actual_prompt_token_ids']}
        ref={'path':'memory-reference','sha256':'bound-in-test'};authref={'path':'memory-auth','sha256':'bound-in-test'}
        auth={'status':'AUTHORIZED','maximum_qualification_calls':576,'independent_new_call_budget':576,
            'qualification_selection':p['qualification_selection'],'point_deg':1.5,
            'preserve_all_original_answers':True,'configuration_changes':[],'observation_only':True,'reference':ref}
        p.update(worker=worker,output=str(remote/'runs/reproducibility'/op.NAME/worker),
            claim_root=str(remote/'runs/reproducibility'/op.NAME/'claims'/worker),
            control_file=str(remote/'plans'/op.NAME/(worker+'-control.json')),execution_authorization=authref,
            recheck={'template_profile':template_ref,'reference':ref,
                     'operations':{'sha256':file_sha256(op.__file__)},'trace':{'sha256':file_sha256(trace.__file__)}})
        def resolve(r):
            if r==ref:return reference
            if r==authref:return auth
            return arc.ref(r)
        op.science(p,resolve)
        cases=[('KV',lambda x:x['engine_kwargs'].__setitem__('kv_cache_memory_bytes',44*1024**3)),
               ('context',lambda x:x['engine_kwargs'].__setitem__('max_model_len',98304)),
               ('autotune',lambda x:x['engine_kwargs'].__setitem__('kernel_config',{'enable_flashinfer_autotune':False})),
               ('output',lambda x:x.__setitem__('output',template['output'])),
               ('claims',lambda x:x.__setitem__('claim_root',template['claim_root'])),
               ('quota',lambda x:x.__setitem__('maximum_model_calls',97)),
               ('SDK',lambda x:x['expected_packages'].__setitem__('vllm','0.31.0')),
               ('sampling input',lambda x:x['request_mm_processor_kwargs'].__setitem__('do_resize',True))]
        for name,change in cases:
            with self.subTest(name=name):
                altered=copy.deepcopy(p);change(altered)
                with self.assertRaises(Exception):op.science(altered,resolve)


if __name__=='__main__':unittest.main()
