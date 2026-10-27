import copy
import ast
import inspect
import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace
import sys
from classic_precision import gemma_static_current as m


class ImageContract(unittest.TestCase):
    def test_native_field_translation_keeps_positions_and_native_BF16_cast(self):
        pixels,converted,positions=Mock(),object(),object();pixels.to.return_value=converted
        with patch.dict(sys.modules,{'torch':SimpleNamespace(bfloat16='bf16')}):
            result=m.image_payload({'pixel_values':pixels,'image_position_ids':positions,'input_ids':[1]})
        pixels.to.assert_called_once_with(dtype='bf16')
        self.assertEqual(set(result),{'pixel_values','pixel_position_ids'})
        self.assertIs(result['pixel_values'],converted)
        self.assertIs(result['pixel_position_ids'],positions)
    def test_bridge_accepts_official_keyword_arguments(self):
        bridge=next(n for n in ast.walk(ast.parse(inspect.getsource(m.run)))
                    if isinstance(n,ast.FunctionDef) and n.name=='prepared')
        self.assertEqual([a.arg for a in bridge.args.args],['this','mm_items','hf_kwargs'])
    def setUp(self):
        self.raw={'input_receipt':{'actual_processor_tensor_shape':[1,2520,768],
            'deep_check':True,'expected_BF16_GPU_sha256':'pixels','expected_normalized_projection_BF16_sha256':'normalized','image_position_ids_sha256':'positions'},
            'actual_input_token_ids':[4,5,6],'expected_input_token_ids':[4,5,6],
            'bridge_calls':1,'raw_answer':'RIGHT','finish_reason':'stop','output_token_ids':[28589,106]}
        self.post={'actual_GPU_visual_forward':[{'shape':[1,2520,768],'dtype':'torch.bfloat16',
            'projection':{'shape':[1,2520,768],'dtype':'torch.bfloat16','device':'cuda:0','tensor_sha256':'normalized'},
            'device':'cuda:0','tensor_sha256':'pixels','positions_sha256':'positions'}]}

    def test_actual_image_and_complete_prompt(self):m.validate_output(self.raw,self.post)
    def test_prompt_difference_rejected(self):
        self.raw['actual_input_token_ids']=[4,5]
        with self.assertRaises(Exception):m.validate_output(self.raw,self.post)
    def test_pixel_difference_rejected(self):
        self.post['actual_GPU_visual_forward'][0]['tensor_sha256']='different'
        with self.assertRaises(Exception):m.validate_output(self.raw,self.post)
    def test_position_difference_rejected(self):
        self.post['actual_GPU_visual_forward'][0]['positions_sha256']='different'
        with self.assertRaises(Exception):m.validate_output(self.raw,self.post)
    def test_wrong_projection_dtype_or_extra_image_rejected(self):
        for change in ['dtype','count']:
            p=copy.deepcopy(self.post)
            if change=='dtype':p['actual_GPU_visual_forward'][0]['projection']['dtype']='torch.float32'
            else:p['actual_GPU_visual_forward']*=2
            with self.assertRaises(Exception):m.validate_output(self.raw,p)
    def test_pre_normalization_FP32_is_rejected(self):
        self.post['actual_GPU_visual_forward'][0]['dtype']='torch.float32'
        with self.assertRaises(Exception):m.validate_output(self.raw,self.post)
    def test_projection_values_rejected(self):
        self.post['actual_GPU_visual_forward'][0]['projection']['tensor_sha256']='wrong'
        with self.assertRaises(Exception):m.validate_output(self.raw,self.post)
    def test_native_eos_and_lr_required(self):
        for key,value in [('output_token_ids',[28589]),('finish_reason','length'),('raw_answer','RIGHT.'),('bridge_calls',0)]:
            r=copy.deepcopy(self.raw);r[key]=value
            with self.assertRaises(Exception):m.validate_output(r,self.post)
    def test_ordinary_keeps_shape_dtype_prompt_checks(self):
        self.raw['input_receipt']['deep_check']=False
        self.post['actual_GPU_visual_forward'][0].pop('tensor_sha256')
        self.post['actual_GPU_visual_forward'][0].pop('positions_sha256')
        m.validate_output(self.raw,self.post)
    def test_static_changes_only_modality_policy(self):
        base={'dtype':'bfloat16','kv_cache_dtype':'bfloat16','max_model_len':65536,
              'limit_mm_per_prompt':{'video':1},'enforce_eager':False}
        with patch.object(m.c,'settings',return_value=dict(base)):
            result=m.settings({'runtime':{},'candidate':'compiled-o0'})
        self.assertEqual(result['limit_mm_per_prompt'],{'video':0,'image':1,'audio':0})
        self.assertEqual(result['mm_processor_kwargs'],{'max_soft_tokens':280})
        self.assertEqual({k:result[k] for k in base if k!='limit_mm_per_prompt'},
                         {k:v for k,v in base.items() if k!='limit_mm_per_prompt'})


if __name__=='__main__':unittest.main()
