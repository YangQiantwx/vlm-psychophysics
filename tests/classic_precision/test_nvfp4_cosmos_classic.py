import copy
import unittest
from unittest.mock import patch

from classic_precision import nvfp4_cosmos_classic as c


class CompletionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        items, reused = {}, {}
        for task, cells in [('T1', 80), ('T2', 78)]:
            for cell in range(cells):
                for index in range(24):
                    for side in ('LEFT', 'RIGHT'):
                        sid = f'{task}_{cell}_{index}_{side}'
                        p = ({'sf_cycles_per_aperture': cell, 'modulation_depth': .1, 'phase_degrees': index}
                             if task == 'T1' else {'background_Y': cell, 'delta_Y': .1,
                                                  'onset_frame': index, 'offset_frame': index + 30,
                                                  'onset_index': index})
                        row = ({'params': p, 'correct_answer': side} if task == 'T1'
                               else {'source_parameters': p, 'truth': side})
                        item = dict(sample_id=sid, task=task, row=row, row_sha256=c.digest(row),
                                    ready_sha256=c.digest(sid))
                        items[sid] = item
                        if cell < 3:
                            raw = dict(sample_id=sid, task=task, truth=side,
                                       catalog_row_sha256=item['row_sha256'],
                                       catalog_ready_sha256=item['ready_sha256'],
                                       finish_reason='stop', raw_answer=side, output_token_ids=[5, 9],
                                       effective_generation={'native_eos_token_ids': [9]}, attempt_id=sid)
                            sha = c.digest(raw)
                            reused[sid] = dict(raw=raw, raw_sha256=sha, technical_pass_raw_sha256=sha,
                                               technical_pass_attempt_id=sid)
        cls.catalog = {'items': items}
        cls.reference = dict(schema=c.SCHEMA + '/reused-reference', status='COMPLETE_PASS', items=reused)
        cls.costs = dict(fourier0=13.58, fourier1=13.83, radon0=17.19)
        cls.allocation = c.allocate(cls.catalog, cls.reference, cls.costs, 'fixed-test-seed')
        runtime = dict(checkpoint='official-nv4', original_checkpoint='original-tokenizer',
                       resources={'min_available_ram_gib': 20, 'min_free_gpu_gib': 80},
                       physical_locks={'0': {}, '1': {}})
        cls.base = dict(generation=copy.deepcopy(c.o.inference.GENERATION),
                        engine_settings=c.o.engine_settings(runtime), runtime=copy.deepcopy(runtime))
        groups = [cls.allocation['new_assignments'][name] for name in ['fourier0', 'fourier1']]
        cls.plan = dict(schema=c.SCHEMA, host='fourier', shard_names=['fourier0', 'fourier1'],
                        assignments=groups, generation=copy.deepcopy(cls.base['generation']),
                        engine_settings=c.o.engine_settings(runtime), runtime=runtime,
                        max_new_calls=sum(map(len, groups)), max_new_calls_per_id=1,
                        max_total_attempts_per_id=1, automatic_retry=False, new_bf16_calls=0,
                        deep_calls_per_fresh_worker=2, prior_failed_nvfp4_attempts={},
                        panel={'items': [{**cls.catalog['items'][s], 'selected_cell_id': c.selected_cell(cls.catalog['items'][s])}
                                         for s in sum(groups, [])]})

    def contract(self, plan):
        c.validate_contract(plan, self.base, self.allocation, self.catalog, self.reference)

    def test_complete_balanced_paired_allocation(self):
        c.validate_allocation(self.allocation, self.catalog, self.reference)
        seconds = list(self.allocation['predicted_worker_seconds'].values())
        self.assertLess(max(seconds) - min(seconds), 40)
        self.contract(self.plan)

    def test_seed_replays_exact_order(self):
        self.assertEqual(self.allocation, c.allocate(self.catalog, self.reference, self.costs, 'fixed-test-seed'))

    def test_duplicate_rejected(self):
        a = copy.deepcopy(self.allocation); a['new_assignments']['fourier1'][2] = a['new_assignments']['fourier0'][2]
        with self.assertRaises(Exception): c.validate_allocation(a, self.catalog, self.reference)

    def test_reused_id_dispatch_rejected(self):
        a = copy.deepcopy(self.allocation); a['new_assignments']['fourier0'][2] = next(iter(self.reference['items']))
        with self.assertRaises(Exception): c.validate_allocation(a, self.catalog, self.reference)

    def test_split_lr_pair_rejected(self):
        a = copy.deepcopy(self.allocation)
        a['new_assignments']['fourier0'][2], a['new_assignments']['fourier1'][2] = a['new_assignments']['fourier1'][2], a['new_assignments']['fourier0'][2]
        with self.assertRaises(Exception): c.validate_allocation(a, self.catalog, self.reference)

    def test_reuse_raw_tamper_rejected(self):
        r = copy.deepcopy(self.reference); next(iter(r['items'].values()))['raw']['raw_answer'] = 'tampered'
        with self.assertRaises(Exception): c.validate_reuse(r, self.catalog)

    def test_reuse_source_mismatch_rejected(self):
        r = copy.deepcopy(self.reference); value = next(iter(r['items'].values()))
        value['raw']['catalog_ready_sha256'] = 'changed'; value['raw_sha256'] = c.digest(value['raw'])
        with self.assertRaises(Exception): c.validate_reuse(r, self.catalog)

    def test_invalid_costs_rejected(self):
        for bad in (0, -1, float('nan')):
            with self.subTest(bad=bad), self.assertRaises(Exception):
                c.allocate(self.catalog, self.reference, {**self.costs, 'radon0': bad}, 's')

    def test_fixed_execution_parameters_rejected_if_changed(self):
        for key, value in [('kv_cache_dtype', 'fp8'), ('max_model_len', 32768),
                           ('max_num_batched_tokens', 49152), ('cpu_offload_gb', 8),
                           ('kernel_config', {'enable_flashinfer_autotune': True})]:
            p = copy.deepcopy(self.plan); p['engine_settings'][key] = value
            with self.subTest(key=key), self.assertRaises(Exception): self.contract(p)

    def test_generation_change_rejected(self):
        p = copy.deepcopy(self.plan); p['generation']['temperature'] = .1
        with self.assertRaises(Exception): self.contract(p)

    def test_retry_or_bf16_or_extra_attempts_rejected(self):
        for key, value in [('automatic_retry', True), ('new_bf16_calls', 1),
                           ('max_new_calls_per_id', 2), ('max_total_attempts_per_id', 2)]:
            p = copy.deepcopy(self.plan); p[key] = value
            with self.subTest(key=key), self.assertRaises(Exception): self.contract(p)

    def test_catalog_mutation_rejected(self):
        p = copy.deepcopy(self.plan); p['panel']['items'][0]['row']['new_prompt'] = 'changed'
        with self.assertRaises(Exception): self.contract(p)

    def test_resource_margin_change_rejected(self):
        p = copy.deepcopy(self.plan); p['runtime']['resources']['min_free_gpu_gib'] = 1
        with self.assertRaises(Exception): self.contract(p)

    def test_direct_frozen_execution_is_used(self):
        self.assertEqual(c.o.file_sha256(c.o.__file__), c.BASE_SHA)
        self.assertNotIn('torch', c.__dict__)


if __name__ == '__main__':
    unittest.main()
