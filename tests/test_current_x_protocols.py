import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from classic_precision.reproduce import configuration


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('model', ['cosmos', 'qwen', 'nemo', 'gemma'])
@pytest.mark.parametrize('task', ['T1', 'T2', 'T3'])
def test_current_export_is_portable_and_binds_exact_runtime(model, task, tmp_path):
    result = configuration(model, 'current-x', tmp_path / 'model', tmp_path / 'tokenizer', tmp_path / 'data', task)
    recipe = result['recipe']
    assert result['status'] == 'CONFIG_ONLY_NOT_A_RUN'
    assert task in recipe['tasks']
    assert recipe['engine_kwargs']['model'] == str(tmp_path / 'model')
    assert recipe['expected_packages']['vllm'] == '0.33.0rc1.dev175+gf4fd1ab68'
    assert recipe['generation']['choice'] == ['LEFT', 'RIGHT']
    assert recipe['generation']['temperature'] == 0
    assert recipe['generation']['ignore_eos'] is False
    assert recipe['input']['frames'] == 180
    for relative, expected in recipe['runtime_source_sha256'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert '/home/' not in json.dumps(result)


def test_qwen_variants_and_gemma_video_adapter_are_not_conflated():
    t1 = configuration('qwen', 'current-x', task='T1')['recipe']
    t3 = configuration('qwen', 'current-x', task='T3')['recipe']
    assert (t1['engine_kwargs']['kv_cache_memory_bytes'], t1['engine_kwargs']['max_model_len']) == (8 * 2**30, 98304)
    assert (t3['engine_kwargs']['kv_cache_memory_bytes'], t3['engine_kwargs']['max_model_len']) == (4 * 2**30, 65536)
    gemma = configuration('gemma', 'current-x')['recipe']
    assert gemma['input']['profile_image_max_soft_tokens'] == 140
    assert gemma['input']['video_max_soft_tokens'] == 280
    assert gemma['input']['effective_tokens_per_frame'] == 176
    assert gemma['request_adapter']['module'] == 'classic_precision.gemma_x_full_input'


def test_export_does_not_import_model_or_gpu_libraries():
    code = "from classic_precision.reproduce import configuration; import sys; configuration('gemma', 'current-x'); assert not ({'torch','vllm','transformers'} & set(sys.modules))"
    subprocess.run([sys.executable, '-B', '-c', code], cwd=ROOT, check=True)


def test_unknown_task_rejected_and_legacy_export_retained():
    with pytest.raises(ValueError):
        configuration('qwen', 'current-x', task='T4')
    assert configuration('qwen', 'native-original')['status'] == 'CONFIG_ONLY_NOT_A_RUN'
