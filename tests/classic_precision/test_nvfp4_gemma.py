"""CPU contracts only: no weights, processor, CUDA, or speed claims."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from classic_precision import nvfp4_gemma as n

def parameters(offload=True):
    rows = [{'name': 'vision_tower.weight', 'dtype': 'torch.bfloat16', 'place': 'cuda', 'numel': 1024, 'bytes': 2048, 'shape': [32, 32]}]
    for layer in range(60):
        for kind, out, inp in (('gate_up_proj', 43008, 5376), ('down_proj', 5376, 21504)):
            prefix = f'language_model.model.layers.{layer}.mlp.{kind}.'
            place = 'cpu' if offload and kind == 'gate_up_proj' else 'cuda'
            for field in ('weight', 'weight_scale', *sorted(n.SCALE_NAMES)):
                numel = out * inp // (2 if field == 'weight' else 16) if field in ('weight', 'weight_scale') else 1
                dtype = 'torch.uint8' if field == 'weight' else 'torch.float8_e4m3fn' if field == 'weight_scale' else 'torch.float32'
                rows.append({'name': prefix + field, 'dtype': dtype, 'place': place, 'numel': numel, 'bytes': numel * (4 if field in n.SCALE_NAMES else 1), 'shape': [numel]})
    return rows

def quant_config():
    excludes = ['lm_head', 'model.embed_vision*', 'model.vision_tower*'] + [f'model.language_model.layers.{i}.self_attn*' for i in range(60)]
    spec = {'dynamic': False, 'num_bits': 4, 'type': 'float', 'group_size': 16}
    q = {'quant_method': 'modelopt', 'quant_algo': 'NVFP4', 'config_groups': {'group_0': {'input_activations': spec, 'weights': spec, 'targets': ['Linear']}}, 'ignore': excludes, 'kv_cache_scheme': {'dynamic': False, 'num_bits': 8, 'type': 'float'}, 'producer': {'name': 'modelopt', 'version': '0.37.0'}}
    return ({'architectures': ['Gemma4ForConditionalGeneration'], 'text_config': {'num_hidden_layers': 60, 'hidden_size': 5376, 'intermediate_size': 21504}, 'quantization_config': q}, {'producer': q['producer'], 'quantization': {'quant_algo': 'NVFP4', 'kv_cache_quant_algo': 'FP8', 'group_size': 16, 'exclude_modules': excludes}})

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj))

def profile_fixture(count, placement='copy8', name='test'):
    original = {'schema': n.g.SCHEMA, 'name': 'BF16', 'runtime': {'checkpoint': '/base/' + n.g.REVISION}, 'checkpoint_assets': {}, 'engine_settings': n._SETTINGS('/base/' + n.g.REVISION), 'kernel': {'range_lookup': 'binary', 'prefill_block_m': 32}, 'cpu_execution_overlay': {'placement': 'baseline', 'gc_cadence': 8}, 'automatic_retry': False, 'maximum_model_calls': count, 'panel': {'items': [{'sample_id': str(i)} for i in range(count)]}}
    original['runtime']['environment'] = {'CUDA_VISIBLE_DEVICES': '0'}
    p = deepcopy(original)
    p['runtime'] = dict(original['runtime'], checkpoint='/quant/' + n.REVISION, processor_checkpoint=original['runtime']['checkpoint'])
    p['runtime']['environment'] = dict(original['runtime']['environment'], **n.COMPILER_ENV)
    p.update(name=name, engine_settings=n.engine_settings(p['runtime']['checkpoint'], p['runtime']['processor_checkpoint'], placement))
    p['nvfp4_overlay'] = {'schema': n.SCHEMA, 'source_sha256': n.g.file_sha256(n.__file__), 'baseline_profile_sha256': n.g.digest(original), 'placement': placement, 'additional_execution_delta': None if placement == 'copy8' else 'offload0', 'original_fields': {k: original[k] for k in ('name', 'runtime', 'checkpoint_assets', 'engine_settings')}}
    p['nvfp4_overlay']['experiment_plan'] = {'file_sha256': n.EXPERIMENT_PLAN_SHA, 'model_calls_cap': 136}
    p['nvfp4_overlay']['compiler_execution_overlay'] = {'environment': dict(n.COMPILER_ENV)}
    return (original, p)

def run_fixture(root, profile, engine, token=1):
    root.mkdir()
    ids = [x['sample_id'] for x in profile['panel']['items']]
    rows = [{'sample_id': sid, 'status': 'PASS', 'native_masks': [{'same': True}]} for sid in ids]
    write_json(root / 'identity.json', {'profile': profile, 'profile_sha256': n.g.digest(profile), 'engine_id': str(engine), 'host': 'radon', 'lease': {'device': 'oneGPU'}})
    write_json(root / 'load.json', {'placement': [{'worker': {'pid': engine, 'startticks': engine}, 'torch_policy': {'TF32': False}}]})
    write_json(root / 'result.json', {'status': 'COMPLETE', 'panel_sha256': n.g.digest(profile['panel']), 'calls': len(ids), 'rows': rows, 'timing': {}})
    write_json(root / 'exit.json', {'status': 'COMPLETE', 'owned_engine_workers_exited': True, 'lease_released_after_engine_exit': True})
    write_json(root / 'nvfp4-execution.json', {'status': 'COMPLETE', 'profile_sha256': n.g.digest(profile), 'source_sha256': n.g.file_sha256(n.__file__), 'final_gc': {'status': 'COMPLETE'}, 'gc_cadence': 8, 'maximum_pending_ordinary': 8, 'rows': rows})
    raw = [{'sample_id': sid, 'output_token_ids': [token, 106], 'raw_answer': 'LEFT', 'finish_reason': 'stop', 'stop_reason': None, 'actual_input_token_ids': [2, 3], 'input_receipt': {'source_stream_sha256': sid, 'prompt_sha256': 'prompt', 'position_sha256': 'pos', 'HF_control_tensors': {'mask': 'same'}}} for sid in ids]
    (root / 'raw.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in raw)))
    return root
import unittest
import tempfile
from unittest import mock

class NVFP4Contracts(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='nvfp4-contract-test-')
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)
        self.enterContext(mock.patch.object(n, 'validate_sources', lambda: None))
        self.enterContext(mock.patch.object(n.c, 'validate_profile', lambda p, installed_sdk=True: p))

    def test_copy8_accepts_all_eligible_nvfp4_parameters_below_eight_gib(self):
        result = n.parameter_contract(parameters(), 8)
        assert 7 * 2 ** 30 < result['observed_cpu_parameter_bytes'] < 8 * 2 ** 30
        assert result['observed_cpu_parameter_bytes'] == result['eligible_gate_up_parameter_bytes']
        assert result['runtime_quantized_MLP_modules'] == 120

    def test_quantized_placement_rejects_wrong_precision_and_placement(self):
        for change in ['packed_bf16', 'scale_bf16', 'unexpected_float32', 'missing_module', 'cuda_subset', 'vision_cpu', 'uva']:
            rows = parameters()
            if change == 'packed_bf16':
                rows[1]['dtype'] = 'torch.bfloat16'
            if change == 'scale_bf16':
                rows[2]['dtype'] = 'torch.bfloat16'
            if change == 'unexpected_float32':
                rows[0]['dtype'] = 'torch.float32'
            if change == 'missing_module':
                rows = [r for r in rows if '.layers.59.mlp.down_proj.' not in r['name']]
            if change == 'cuda_subset':
                rows[1]['place'] = 'cuda'
            if change == 'vision_cpu':
                rows[0]['place'] = 'cpu'
            if change == 'uva':
                rows[1]['place'] = 'uva'
            with self.assertRaises(ValueError):
                n.parameter_contract(rows, 8)

    def test_full_gpu_candidate_rejects_hidden_cpu_parameters(self):
        assert n.parameter_contract(parameters(False), 0)['observed_cpu_parameter_bytes'] == 0
        with self.assertRaisesRegex(ValueError, 'Full-GPU'):
            n.parameter_contract(parameters(), 0)

    def test_settings_preserve_scientific_and_c3_configuration(self):
        a, b = (n.engine_settings('quantized', 'original', p) for p in ('copy8', 'full-gpu'))
        assert {k for k in a if a[k] != b[k]} == {'cpu_offload_gb', 'cpu_offload_params'}
        original = n._SETTINGS('quantized')
        assert {k for k in a if k not in original or a[k] != original[k]} == {'quantization', 'tokenizer', 'tensor_parallel_size'}
        assert a['kv_cache_dtype'] == 'bfloat16' and a['kv_cache_memory_bytes'] == 32 * 2 ** 30
        assert a['max_model_len'] == 65536 and a['max_num_batched_tokens'] == 34000
        assert a['tokenizer'] == 'original' and a['tensor_parallel_size'] == 1

    def test_old_admission_cannot_authorize_new_precision_experiment(self):
        _, profile = profile_fixture(20)
        with self.assertRaisesRegex(ValueError, "Fresh finite"):
            n.admission_contract({'all_owned_engines_exited': True}, profile)
        admission = {'nvfp4_authorized': True, 'nvfp4_plan_file_sha256': n.EXPERIMENT_PLAN_SHA,
            'nvfp4_profile_sha256': n.g.digest(profile), 'nvfp4_model_calls_cap': 136,
            'nvfp4_observation_role': 'O8'}
        assert n.admission_contract(admission, profile) == 'O8'
        admission['nvfp4_observation_role'] = 'cold0'
        with self.assertRaisesRegex(ValueError, "role"):
            n.admission_contract(admission, profile)

    def test_compiler_environment_requires_exact_five_keys_and_retains_base_guard(self):
        _, profile = profile_fixture(20)
        with mock.patch.dict(n.os.environ, n.COMPILER_ENV, clear=False):
            with mock.patch.object(n, '_RUNTIME', return_value={'base_checked': True}) as guard:
                assert n.runtime_check(profile['runtime']) == {'base_checked': True}
                forwarded = guard.call_args.args[0]
                assert forwarded['checkpoint'].endswith(n.g.REVISION)
                assert forwarded['environment'] == {'CUDA_VISIBLE_DEVICES': '0'}
            for key in n.COMPILER_ENV:
                changed = deepcopy(profile['runtime'])
                changed['environment'][key] = 'wrong'
                with self.assertRaisesRegex(ValueError, 'compiler environment'):
                    n.runtime_check(changed)

    def test_profile_rejects_unapproved_sixth_runtime_environment_key(self):
        _, profile = profile_fixture(20)
        profile['runtime']['environment']['FLASHINFER_EXTRA_CUDAFLAGS'] = '--use_fast_math'
        with self.assertRaisesRegex(ValueError, 'five compiler keys'):
            n.validate_profile(profile, installed_sdk=False)

    def test_owned_ninja_view_is_the_only_new_path_prefix(self):
        source, view = self.tmp_path / 'existing-ninja', self.tmp_path / 'ninja-view'
        source.write_text('#!/bin/sh\nprintf "%s\\n" "1.13.2.git.kitware.jobserver-pipe-1"\n')
        source.chmod(0o755)
        with mock.patch.object(n, 'NINJA_SOURCE', str(source)), mock.patch.object(n, 'NINJA_VIEW', str(view)), mock.patch.dict(n.COMPILER_ENV, {'PATH': str(view) + ':' + n.ORIGINAL_TOOL_PATH}):
            result = n.create_ninja_view()
            assert result['symlink_count'] == 1 and result['copied_binary_bytes'] == 0
            assert n.verify_ninja_view()['target'] == str(source)
            assert n.shutil.which('ninja', path=n.COMPILER_ENV['PATH']) == str(view / 'ninja')
            (view / 'unapproved-tool').write_text('no')
            with self.assertRaisesRegex(ValueError, 'identity'):
                n.verify_ninja_view()

    def test_native_probe_has_two_calls_at_actual_gate_up_geometry(self):
        spec = n.native_probe_spec(33000)
        assert spec['M'] == [8, 33000] and spec['K'] == 5376 and spec['N'] == 43008
        assert spec['packed_weight_shape'] == [spec['N'], spec['K'] // 2]
        assert spec['weight_scale_shape'] == [spec['N'], spec['K'] // 16]
        assert spec['maximum_synthetic_native_gemm_calls'] == 2 and spec['scientific_model_calls'] == 0
        for tokens in (0, 180, 34001):
            with self.assertRaisesRegex(ValueError, 'full-prefill'):
                n.native_probe_spec(tokens)

    def test_owned_compiler_view_links_and_binds_existing_files(self):
        root, view = self.tmp_path / 'existing', self.tmp_path / 'view'
        for name in n.TOOLCHAIN_FILES:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('existing ' + name).encode())
        driver = self.tmp_path / 'libcuda.so'
        driver.write_bytes(b'existing driver fixture')
        with mock.patch.object(n, 'CUDA13_ROOT', str(root)), mock.patch.object(n, 'CUDA13_VIEW', str(view)), mock.patch.object(n, 'CUDA_DRIVER_LINK', str(driver)):
            result = n.create_toolchain_view()
            assert result['symlink_count'] == 5 and result['copied_library_bytes'] == 0
            record = n.verify_toolchain_view()
            assert (view / 'lib64/libcudart.so').samefile(root / 'lib/libcudart.so.13')
            assert record['required_runtime_files']['bin/nvcc']['sha256'] == n.g.file_sha256(root / 'bin/nvcc')
            (view / 'lib64/libcudart.so').unlink()
            (view / 'lib64/libcudart.so').symlink_to(driver)
            with self.assertRaisesRegex(ValueError, 'retargeted'):
                n.verify_toolchain_view()

    def test_quantization_contract_records_unproven_base_commit_and_bf16_kv_override(self):
        config, separate = quant_config()
        contract = n.quantization_contract(config, separate)
        assert contract['published_base_commit'] is None and contract['experiment_KV_dtype'] == 'bfloat16'
        changed = deepcopy(config)
        changed['quantization_config']['config_groups']['group_0']['input_activations'] = None
        with self.assertRaisesRegex(ValueError, 'W4A4'):
            n.quantization_contract(changed, separate)

    def test_no_runtime_import_for_cpu_cli(self):
        result = subprocess.run([sys.executable, '-B', '-c', "import classic_precision.nvfp4_gemma,sys; assert 'torch' not in sys.modules; assert 'vllm' not in sys.modules; assert 'transformers' not in sys.modules"], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    def test_process_local_overlay_restores_original_guards_after_exception(self):
        original = n.g._placement
        profile = {'runtime': {'checkpoint': 'quantized', 'processor_checkpoint': 'original'}, 'nvfp4_overlay': {'placement': 'copy8'}, 'engine_settings': {'cpu_offload_gb': 8}}
        with self.assertRaises(RuntimeError):
            with n.installed_overlay(profile):
                assert n.g._placement is not original
                assert n.g.engine_settings('quantized')['tokenizer'] == 'original'
                raise RuntimeError('fixture failure')
        assert n.g._placement is original and n.g.runtime_check is n._RUNTIME
        assert n.inference.ProcessorSession is n._PROCESSOR

    def test_precision_difference_is_measured_and_not_false_repeat_failure(self):
        base, quant = profile_fixture(20)
        a, b = (run_fixture(self.tmp_path / 'a', base, 1), run_fixture(self.tmp_path / 'b', quant, 2, 9))
        result = n.compare_runs([a, b], 'precision')
        assert result['status'] == 'COMPLETE_PRECISION_COMPARISON' and result['precision_difference_count'] == 20
        assert not result['two_cold_full_token_pass']

    def test_repeat_requires_full_tokens_and_independent_cold_identity(self):
        _, quant = profile_fixture(48)
        a, b = (run_fixture(self.tmp_path / 'a', quant, 1), run_fixture(self.tmp_path / 'b', quant, 2))
        assert n.compare_runs([a, b], 'repeat')['two_cold_full_token_pass']
        same_engine = n.g.read(b / 'load.json')
        same_engine['placement'][0]['worker'] = {'pid': 1, 'startticks': 1}
        write_json(b / 'load.json', same_engine)
        with self.assertRaisesRegex(ValueError, 'independent cold'):
            n.compare_runs([a, b], 'repeat')

    def test_repeat_rejects_token_difference_even_when_answer_is_identical(self):
        _, quant = profile_fixture(48)
        a, b = (run_fixture(self.tmp_path / 'a', quant, 1), run_fixture(self.tmp_path / 'b', quant, 2, 9))
        result = n.compare_runs([a, b], 'repeat')
        assert result['status'] == 'DIFFERENT' and (not result['two_cold_full_token_pass'])

    def test_placement_requires_tokens_match_and_only_offload_changes(self):
        _, copy8 = profile_fixture(20, 'copy8', 'O8')
        _, gpu = profile_fixture(20, 'full-gpu', 'O0')
        a, b = (run_fixture(self.tmp_path / 'a', copy8, 1), run_fixture(self.tmp_path / 'b', gpu, 2, 9))
        result = n.compare_runs([a, b], 'placement')
        assert result['status'] == 'DIFFERENT' and (not result['placement_full_token_pass'])

    def test_input_difference_invalidates_precision_comparison(self):
        base, quant = profile_fixture(20)
        a, b = (run_fixture(self.tmp_path / 'a', base, 1), run_fixture(self.tmp_path / 'b', quant, 2))
        raw = n._records(b)
        raw[0]['input_receipt']['processor_input_FP32_sha256'] = 'changed'
        (b / 'raw.jsonl').write_text(''.join((json.dumps(r) + '\n' for r in raw)))
        assert n.compare_runs([a, b], 'precision')['status'] == 'INVALID_EVIDENCE'

    def test_precision_comparison_rejects_changed_torch_policy(self):
        base, quant = profile_fixture(20)
        a = run_fixture(self.tmp_path / 'a', base, 1)
        b = run_fixture(self.tmp_path / 'b', quant, 2)
        load = n.g.read(b / 'load.json')
        load['placement'][0]['torch_policy']['TF32'] = True
        write_json(b / 'load.json', load)
        with self.assertRaisesRegex(ValueError, 'Torch runtime'):
            n.compare_runs([a, b], 'precision')

    def _composition_fixture(self):
        source, original, output = (self.tmp_path / label / rev for label, rev in (('source', n.REVISION), ('original', n.g.REVISION), ('composed', n.REVISION)))
        source.mkdir(parents=True)
        original.mkdir(parents=True)
        config, quant = quant_config()
        baseline = {'vision_config': {'image_seq_length': 280, 'video_seq_length': 280}, 'model_type': 'gemma4'}
        write_json(source / 'config.json', config)
        write_json(source / 'hf_quant_config.json', quant)
        write_json(original / 'config.json', baseline)
        for root in (source, original):
            write_json(root / 'model.safetensors.index.json', {'weight_map': {'fixture': 'model.safetensors'}})
            (root / 'model.safetensors').write_bytes(b'fixture, not model weights')
        for filename in ('processor_config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
            write_json(original / filename, {'original': filename})
        (original / 'chat_template.jinja').write_text('original exact template')
        self.enterContext(mock.patch.object(n, 'checkpoint_contract', lambda a, b: {'fixture': True}))
        before = n.g.file_sha256(original / 'config.json')
        result = n.compose_checkpoint(SimpleNamespace(source_checkpoint=str(source), processor_checkpoint=str(original), output=str(output)))
        return source, original, output, config, baseline, before, result

    def test_composition_uses_original_config_and_inputs_without_overwriting(self):
        source, original, output, config, baseline, before, result = self._composition_fixture()
        assert result['status'] == 'COMPOSED'
        effective = n.g.read(output / 'config.json')
        assert effective.pop('quantization_config') == config['quantization_config'] and effective == baseline
        assert n.g.file_sha256(original / 'config.json') == before
        assert (output / 'chat_template.jinja').resolve() == (original / 'chat_template.jinja').resolve()
        assert not (output / 'model.safetensors').is_symlink()
        assert (output / 'model.safetensors').samefile(source / 'model.safetensors')
        assert (output / 'model.safetensors.index.json').samefile(source / 'model.safetensors.index.json')
        assert result['hardlink_count'] == 2 and result['duplicated_weight_bytes'] == 0
        assets = n.inference.checkpoint_asset_identity(output)
        assert assets['weight_shard_bytes'] == {'model.safetensors': (source / 'model.safetensors').stat().st_size}
        (output / 'tokenizer.json').unlink()
        (output / 'tokenizer.json').symlink_to(source / 'config.json')
        with self.assertRaisesRegex(ValueError, 'retargeted'):
            n.effective_checkpoint_contract(output, original)

    def test_hardlink_guard_rejects_replacement_copy_and_external_symlink(self):
        source, original, output, *_ = self._composition_fixture()
        for name in ('model.safetensors', 'model.safetensors.index.json'):
            actual, target = output / name, source / name
            content = actual.read_bytes()
            actual.unlink()
            actual.write_bytes(content)
            with self.assertRaisesRegex(ValueError, 'hardlink identity'):
                n.effective_checkpoint_contract(output, original)
            actual.unlink()
            actual.symlink_to(target)
            with self.assertRaisesRegex(ValueError, 'hardlink identity'):
                n.effective_checkpoint_contract(output, original)
            actual.unlink()
            n.os.link(target, actual)
            n.effective_checkpoint_contract(output, original)
if __name__ == '__main__':
    unittest.main()
