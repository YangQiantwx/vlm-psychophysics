import copy
import unittest
from unittest.mock import patch

from classic_precision import gemma_x_recheck_autotune_off as off


class AutotuneOffTests(unittest.TestCase):
    def records(self):
        expected = dict(raw_answer='LEFT', output_token_ids=[1], finish_reason='stop',
                        stop_reason=None, truth='RIGHT', question='q', generation={'temperature': 0})
        raw = dict(expected, sample_id='s', actual_prompt_token_ids=[3, 4])
        ref = {'by_id': {'s': expected}, 'actual_prompt_token_ids': [3, 4]}
        return raw, ref

    def test_on_off_answer_change_is_measured_not_reproduction_failure(self):
        raw, ref = self.records(); raw.update(raw_answer='RIGHT', output_token_ids=[2])
        result = off.compare(raw, ref)
        self.assertTrue(result['accepted']); self.assertFalse(result['ON_OFF_full_output_prompt_equal'])
        self.assertEqual(result['ON_OFF_different_fields'], ['raw_answer', 'output_token_ids'])
        self.assertIsNone(result['OFF_cold_full_output_prompt_equal'])

    def test_off_colds_must_match_every_output_and_prompt_field(self):
        raw, ref = self.records(); previous = {'s': copy.deepcopy(raw)}
        self.assertTrue(off.compare(raw, ref, previous)['accepted'])
        for key in off.FIELDS + ['actual_prompt_token_ids']:
            changed = copy.deepcopy(raw); changed[key] = 'different'
            with self.subTest(key=key): self.assertFalse(off.compare(changed, ref, previous)['accepted'])

    def test_first_cold_rejects_input_prompt_generation_changes(self):
        raw, ref = self.records()
        for key in ['truth', 'question', 'generation', 'actual_prompt_token_ids']:
            changed = copy.deepcopy(raw); changed[key] = 'different'
            with self.subTest(key=key): self.assertFalse(off.compare(changed, ref)['accepted'])

    def test_single_setting_projection_does_not_mutate_actual_profile(self):
        p = {'card':'radon0','engine_kwargs':{'max_model_len':65536,
             'kernel_config':{'enable_flashinfer_autotune':False}},
             'recheck':{'autotune_off':{'entrypoint':{'sha256':off.file_sha256(off.__file__)},
                  'authorization':'a','zero_call_closure':'c'}}}
        auth = {'status':'AUTHORIZED','maximum_Radon_calls':192,'new_budget_added':0,
                'preserve_original_answers':True,
                'only_scientific_change':{'kernel_config.enable_flashinfer_autotune':False}}
        resolver = lambda ref: auth if ref == 'a' else {}
        with patch.object(off.recovery,'validate_closure'), patch.object(off,'ORIGINAL_SCIENCE') as original:
            off.science(p,resolver)
            self.assertNotIn('kernel_config',original.call_args.args[0]['engine_kwargs'])
            self.assertFalse(p['engine_kwargs']['kernel_config']['enable_flashinfer_autotune'])
            p['engine_kwargs']['kernel_config']['extra']=1
            with self.assertRaises(ValueError):off.science(p,resolver)

    def test_actual_runtime_must_report_off(self):
        for setting in [True,False]:
            value={'kernel_config':f'KernelConfig(enable_flashinfer_autotune={setting})'}
            with patch.object(off,'ORIGINAL_IDENTITY',return_value=value):
                if setting:
                    with self.assertRaises(ValueError):off.identity(None,'gemma_x_full')
                else:self.assertIs(off.identity(None,'gemma_x_full'),value)

    def test_nested_resource_launch_keeps_off_gpu_entrypoint(self):
        original=(off.core.NAME,off.core.MODULE,off.recovery.NAME,off.recovery.MODULE)
        with off.namespace():
            with off.recovery.namespace():
                self.assertEqual(off.core.NAME,off.NAME)
                self.assertEqual(off.core.MODULE,off.MODULE)
                self.assertIs(off.core.science,off.science)
        self.assertEqual(original,(off.core.NAME,off.core.MODULE,off.recovery.NAME,off.recovery.MODULE))


if __name__ == '__main__':unittest.main()
