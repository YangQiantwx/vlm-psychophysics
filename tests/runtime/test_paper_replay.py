"""Check real request branches without importing a GPU runtime."""

import ast
from hashlib import sha256
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

from paper_core import native_final, scores
from paper_core.replay import recipe
from paper_core.runtime_recipe import (RECIPE_PATH, invocation, invoke, parse_output, prefill, prompt_sha_i32)
from paper_core.scores import read_json


CONFIG = read_json(RECIPE_PATH)
PROFILES = CONFIG['recipes']
LEFT, RIGHT, NO, YES = (4, 5, 6, 7)
OPEN, CLOSE, EFFICIENT, LOW, NEWLINE, REASON = (10, 11, 12, 13, 14, 15)


class Tokenizer:
    mapping = {LEFT:'LEFT', RIGHT:'RIGHT', NO:'NO', YES:'YES', OPEN:'<think>', CLOSE:'</think>',
               EFFICIENT:'{reasoning effort: efficient}', LOW:'Reasoning effort is set to low.', NEWLINE:'\n', REASON:'x'}

    def encode(self, text, **_):
        number, = [k for k, v in self.mapping.items() if v == text]
        return [number]

    def decode(self, values, skip_special_tokens=False):
        return ''.join(self.mapping[i] for i in values if not (skip_special_tokens and i in (OPEN, CLOSE)))


@pytest.fixture
def fake_vllm(monkeypatch):
    vllm = ModuleType('vllm')
    vllm.SamplingParams = lambda **kw: SimpleNamespace(ignore_eos=False, stop=[], stop_token_ids=[], **kw)
    sampling = ModuleType('vllm.sampling_params')
    sampling.StructuredOutputsParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, 'vllm', vllm)
    monkeypatch.setitem(sys.modules, 'vllm.sampling_params', sampling)


@pytest.mark.parametrize('task,mode,model,variant,arm,wanted', [
    ('T3','video','cosmos','axis-base','off','cosmos-axis-video'),
    ('T3','video','cosmos','axis','off','cosmos-axis-video'),
    ('T3','video','qwen','axis','off','qwen-axis-video'),
    ('T3','static','qwen','black_exact','off','qwen-axis-static'),
    ('T3','static','cosmos','objects_only','off','cosmos-axis-static'),
    ('T4','video','cosmos','original','cot','cosmos-cot-t4'),
    ('T5','video','cosmos','original','cot','cosmos-cot-t5'),
    ('T5','video','qwen','enlarged','off','qwen-enlarged'),
    ('T1','static','qwen','original','low','qwen-native-static'),
    ('T5','video','nemo','original','low','nemo-t5'),
])
def test_trial_selects_its_archived_recipe(task, mode, model, variant, arm, wanted):
    args = SimpleNamespace(mode=mode, model=model, variant=variant, arm=arm)
    assert recipe(args, task) == wanted


def test_axis_video_and_static_use_distinct_label_decoding(fake_vllm):
    tok = Tokenizer()
    for model in ('qwen','cosmos'):
        video, vcall = invocation(PROFILES[f'{model}-axis-video'], 'off', ('LEFT','RIGHT'), tok)
        assert video.max_tokens == 16384 and video.structured_outputs.choice == ['LEFT','RIGHT']
        static, scall = invocation(PROFILES[f'{model}-axis-static'], 'off', ('LEFT','RIGHT'), tok)
        assert static.max_tokens == 1 and static.allowed_token_ids == [LEFT, RIGHT]
        assert vcall['chat_template_kwargs'] == scall['chat_template_kwargs'] == {'enable_thinking':False}
        result = SimpleNamespace(text='RIGHT', token_ids=[RIGHT], finish_reason='length', stop_reason=None)
        assert parse_output(PROFILES[f'{model}-axis-static'], 'off', ('LEFT','RIGHT'), result, tok, [])['answer'] == 'RIGHT'
        assert parse_output(PROFILES[f'{model}-axis-video'], 'off', ('LEFT','RIGHT'), result, tok, [])['answer'] is None


@pytest.mark.parametrize('task,budget', [('t4',1500),('t5',3000)])
def test_cosmos_prefill_keeps_budget_and_omits_native_thinking(fake_vllm, task, budget):
    profile = PROFILES[f'cosmos-cot-{task}']
    tok = Tokenizer()
    message = [{'role':'user','content':[{'type':'text','text':'Question'}]}]
    message = prefill(message, 'Question', profile, 'cot')
    assert message[-1] == {'role':'assistant','content':'<think>'}
    assert message[0]['content'][0]['text'] == 'Question' + profile['cot_prompt_suffix']
    params, kwargs = invocation(profile, 'cot', ('LEFT','RIGHT'), tok)
    assert params.thinking_token_budget == budget and params.max_tokens == 16384
    assert kwargs['continue_final_message'] is True and kwargs['add_generation_prompt'] is False
    assert 'chat_template_kwargs' not in kwargs
    result = SimpleNamespace(text='x</think>RIGHT', token_ids=[REASON,CLOSE,RIGHT], finish_reason='stop', stop_reason=None)
    actual = parse_output(profile, 'cot', ('LEFT','RIGHT'), result, tok, [OPEN])
    assert actual['answer']=='RIGHT' and actual['reasoning_span_tokens']==1


def test_nemotron_checks_only_the_native_final_span(fake_vllm):
    profile = PROFILES['nemo-t5']
    tok = Tokenizer()
    params, kwargs = invocation(profile, 'low', ('YES','NO'), tok)
    assert params.structured_outputs.choice == ['YES','NO']
    assert kwargs == {'use_tqdm':False,'chat_template_kwargs':{'enable_thinking':True,'medium_effort':True}}
    raw = SimpleNamespace(text='NOx</think>YES', token_ids=[NO, REASON, CLOSE, YES], finish_reason='stop', stop_reason=None)
    assert parse_output(profile,'low',('YES','NO'),raw,tok,[EFFICIENT,OPEN,NEWLINE])['answer']=='YES'
    raw.finish_reason='length'
    assert parse_output(profile,'low',('YES','NO'),raw,tok,[EFFICIENT,OPEN,NEWLINE])['answer'] is None


def test_mock_engine_checks_the_t3_prompt_hash(fake_vllm):
    prompt_ids = [5,10,13]
    output = SimpleNamespace(text='LEFT',token_ids=[LEFT],finish_reason='stop',stop_reason=None)
    class Engine:
        def get_tokenizer(self):
            return Tokenizer()

        def chat(self, messages, sampling_params, **kwargs):
            assert kwargs['chat_template_kwargs']=={'enable_thinking':False}
            assert kwargs['mm_processor_kwargs']=={'do_resize':False,'do_sample_frames':False}
            assert sampling_params.max_tokens==16384 and messages[0]['content'][0]['type']=='video_url'
            return [SimpleNamespace(prompt_token_ids=prompt_ids, outputs=[output])]

    messages=[{'role':'user','content':[{'type':'video_url','video_url':{'url':'file:///fixture.mp4'}}]}]
    assert invoke(Engine(),PROFILES['cosmos-axis-video'],'off',('LEFT','RIGHT'),messages,prompt_sha_i32(prompt_ids))['answer']=='LEFT'
    with pytest.raises(ValueError, match='prompt-token digest'):
        invoke(Engine(),PROFILES['cosmos-axis-video'],'off',('LEFT','RIGHT'),messages,'0'*64)


def test_archived_parser_and_molmo_source_digests():
    raw = Path(native_final.__file__).read_text()
    parser, = [n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)]
    assert sha256(ast.get_source_segment(raw,parser).encode()).hexdigest()==CONFIG['native_parser_function_sha256']
    molmo = RECIPE_PATH.parent/'molmo'
    names = {'runner':'run_molmo2_noresize_subset.py','input_bridge':'molmo2_noresize.py','prefix_attention':'molmo2_prefix_attention.py'}
    for k, name in names.items():
        assert sha256((molmo/name).read_bytes()).hexdigest()==CONFIG['molmo']['source_hashes'][k]


@pytest.mark.parametrize('column,invalid,expected', [
    ('model', 'other', 'unknown baseline model'),
    ('modality', 'image', 'unknown baseline modality'),
    ('task', 'T2', 'wrong task file'),
])
def test_public_results_rejects_wrong_grid_cell(monkeypatch, tmp_path, column, invalid, expected):
    video = dict(task='T1', domain='classic', truth='LEFT', video_sha256='video-hash')
    static = {'images': [{'sha256': 'image-hash'}]}
    rows = [dict(video, sample_id='one', model=model, modality=mode,
                 static_image_sha256='["image-hash"]', raw_answer='LEFT', prediction='LEFT', correct='1')
            for model in scores.MODELS[:-1] for mode in ('video', 'static')]
    monkeypatch.setattr(scores, 'read_csv', lambda path: rows if path.parent.name == 't1' else [])
    args = (tmp_path, {'one': video}, {'one': static})
    assert len(scores.public_results(*args)) == len(rows) == 8
    rows[-1][column] = invalid
    with pytest.raises(ValueError, match=expected):
        scores.public_results(*args)
