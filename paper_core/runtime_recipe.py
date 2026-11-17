"""Execute the archived local request contracts against a selected vLLM engine."""

from collections import Counter
from importlib import metadata
import os
from pathlib import Path
import struct
from hashlib import sha256

from .native_final import parse_generated_response
from .scores import read_json, require

RECIPE_PATH = Path(__file__).with_name('recipes.json')
VIDEO_INPUT = {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}


def prefill(messages, prompt, profile, arm):
    if profile.get('execution') != 'cosmos_prefill' or arm != 'cot':
        return messages
    messages[0]['content'][-1]['text'] = prompt + profile['cot_prompt_suffix']
    messages.append({'role': 'assistant', 'content': profile['assistant_prefill']})
    return messages


def invocation(profile, arm, choices, tokenizer):
    """Return the worker's exact sampling settings and kwargs for LLM.chat."""
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    execution = profile['execution']
    settings = {'use_tqdm': False}
    if 'request_mm_processor_kwargs' in profile:
        settings['mm_processor_kwargs'] = profile['request_mm_processor_kwargs']
    if execution == 'axis_static':
        ids = [tokenizer.encode(label, add_special_tokens=False) for label in ('LEFT', 'RIGHT', 'NO', 'YES')]
        require(all(len(item) == 1 for item in ids) and len({x[0] for x in ids}) == 4,
                'T3 uses one distinct token per label')
        params = SamplingParams(n=1, temperature=0., seed=0, max_tokens=1,
                                allowed_token_ids=[x[0] for x in ids[:2]])
        settings['chat_template_kwargs'] = {'enable_thinking': False}
        return params, settings

    # Structured label choices and native EOS from the official X worker.
    params = SamplingParams(n=1, temperature=0., top_p=1., seed=0, repetition_penalty=1.,
                            max_tokens=16384, structured_outputs=StructuredOutputsParams(choice=list(choices)))
    require(params.ignore_eos is False and params.stop == [] and params.stop_token_ids == [],
            'Native EOS contract changed')
    if execution == 'cosmos_prefill':
        cot = arm == 'cot'
        settings.update(add_generation_prompt=not cot, continue_final_message=cot)
        if cot:
            require(profile['total_tokens'] == params.max_tokens, 'Cosmos total budget changed')
            params.thinking_token_budget = profile['thinking_tokens']
    elif execution == 'nemo_native':
        settings['chat_template_kwargs'] = profile['native_kwargs'][arm]
    else:
        settings['chat_template_kwargs'] = {'enable_thinking': arm == 'low'}
        if arm == 'low':
            settings['chat_template_kwargs']['reasoning_effort'] = 'low'
    return params, settings


def prompt_sha_i32(token_ids):
    return sha256(struct.pack(f'<{len(token_ids)}i', *token_ids)).hexdigest()


def parse_output(profile, arm, choices, output, tokenizer, prompt_ids):
    execution = profile['execution']
    tokens = list(output.token_ids)
    audit = {}
    if execution in ('qwen_native', 'nemo_native'):
        native = {'enable_thinking': arm == 'low'}
        parser = 'official_x_direct' if execution == 'nemo_native' else 'strict_label'
        audit = parse_generated_response(output, tokenizer, native, choices, legacy_parser_version=parser)
        label = audit['final_answer']
        eligible = output.finish_reason == 'stop'
        if execution == 'nemo_native' and arm == 'low':
            tail = tokenizer.decode(prompt_ids[-256:], skip_special_tokens=False)
            require(tail.endswith('<think>\n') and '{reasoning effort: efficient}' in tail,
                    'Nemotron native Efficient template changed')
        if execution == 'qwen_native':
            suffix = '<think>\n' if arm == 'low' else '<think>\n\n</think>\n\n'
            require(tokenizer.decode(prompt_ids[-20:], skip_special_tokens=False).endswith(suffix),
                    'Qwen native thinking template changed')
            if 'frame_prompt_prefix' not in profile:
                head = tokenizer.decode(prompt_ids[:96], skip_special_tokens=False)
                require(('Reasoning effort is set to low.' in head) == (arm == 'low'),
                        'Qwen native reasoning header changed')
    elif execution == 'cosmos_prefill' and arm == 'cot':
        open_id, = tokenizer.encode(profile['assistant_prefill'], add_special_tokens=False)
        close_id, = tokenizer.encode('</think>', add_special_tokens=False)
        require(prompt_ids[-1] == open_id, 'Cosmos assistant prefill token changed')
        ends = [i for i, token in enumerate(tokens) if token == close_id]
        boundary = ends[0] if len(ends) == 1 else None
        require(boundary is None or boundary <= profile['thinking_tokens'], 'Cosmos reasoning exceeded its budget')
        label = tokenizer.decode(tokens[boundary + 1:], skip_special_tokens=True) if boundary is not None else None
        eligible = output.finish_reason == 'stop'
        audit['reasoning_span_tokens'] = boundary
    else:
        label = output.text
        eligible = output.finish_reason == 'stop'
        if execution == 'axis_static':
            allowed_ids = {tuple(tokenizer.encode(x, add_special_tokens=False)) for x in choices}
            eligible = tuple(tokens) in allowed_ids
    answer = label.strip() if eligible and label is not None and label.strip() in choices else None
    return {**audit, 'answer': answer, 'finish_reason': output.finish_reason,
            'prompt_token_ids': prompt_ids, 'output_token_ids': tokens, 'raw_answer': output.text}


def invoke(engine, profile, arm, choices, messages, expected_prompt_sha=None):
    tokenizer = engine.get_tokenizer()
    params, settings = invocation(profile, arm, choices, tokenizer)
    result, = engine.chat(messages, sampling_params=params, **settings)
    output, = result.outputs
    ids = list(result.prompt_token_ids)
    if expected_prompt_sha:
        require(prompt_sha_i32(ids) == expected_prompt_sha, 'Archived T3 prompt-token digest differs')
    return parse_output(profile, arm, choices, output, tokenizer, ids)


def check_versions(expected):
    for package, version in expected.items():
        require(metadata.version(package) == version, (package, 'installed version differs', version))


def verify_model_artifacts(model, weights, tokenizer, cache):
    from vlm_psychophysics import assets

    lock = assets._lock({})
    binding = lock['bindings'][model]
    for kind, path in (('model_checkpoint', weights), ('tokenizer_checkpoint', tokenizer)):
        if path is None:
            continue
        materials = lock['materials'][binding[kind]]
        checked = assets._check_files(path, materials)
        assets._verify_weight_hashes(cache, checked, materials)


def execute(args, profile, messages, choices, expected_prompt_sha, source_video_rgb):
    require(args.weights and args.weights.is_dir() and args.weights.name == profile['checkpoint_revision'],
            'Provide the local snapshot at the checkpoint revision in recipes.json')
    check_versions(profile['package_versions'])
    tokenizer = args.tokenizer
    if 'tokenizer_revision' in profile:
        require(tokenizer and tokenizer.is_dir() and tokenizer.name == profile['tokenizer_revision'],
                'Provide the pinned tokenizer snapshot')
    verify_model_artifacts(args.model, args.weights, tokenizer, args.verification_cache)
    for name in profile['cleared_environment']:
        os.environ.pop(name, None)
    os.environ.update(profile['environment'])
    import torch
    from vllm import LLM
    from vlm_psychophysics.observers import NativeObserver, worker_identity

    if profile['torch_num_threads'] is not None:
        torch.set_num_threads(profile['torch_num_threads'])
    require(torch.cuda.device_count() == 1, 'Set CUDA_VISIBLE_DEVICES to one GPU')
    roots = [args.data.resolve(), args.study.resolve(), args.out.resolve()]
    if args.axis_images:
        roots.append(args.axis_images.resolve())
    kwargs = dict(profile['engine'], model=str(args.weights.resolve()), allowed_local_media_path=os.path.commonpath(roots))
    if 'cpu_offload_params' in kwargs:
        kwargs['cpu_offload_params'] = set(kwargs['cpu_offload_params'])
    if tokenizer:
        kwargs['tokenizer'] = str(tokenizer.resolve())
    engine = LLM(**kwargs)
    workers = engine.collective_rpc(worker_identity, args=(args.model,))
    worker, = workers
    minimum = profile['minimum_kv_capacity']
    if minimum is not None:
        require(worker['actual_kv_capacity'] >= minimum, 'Actual KV capacity differs from admission')
    if args.model == 'qwen':
        selector = worker['custom_qsa_selector']
        require(selector['status'] == 'APPLIED_X_QSA' and
                selector['selector_source_sha256'] == profile['selector_sha256'], 'Actual Qwen selector differs')
    with NativeObserver(profile) as obs:
        obs.contract = VIDEO_INPUT
        obs.deep, obs.active = True, True
        result = invoke(engine, profile, args.arm, choices, messages, expected_prompt_sha)
    require(len(obs.reads) == int(args.mode == 'video'), 'Native reader call count changed')
    if obs.reads:
        reader, = obs.reads
        require(reader['metadata']['fps'] == 60 and reader['decoded_sha256'] == source_video_rgb,
                'Actual reader RGB or video clock differs')
    if args.model in ('qwen', 'cosmos'):
        grids = [a['after_native_dtype'] for a in obs.processors]
        if args.mode == 'video':
            require(any(x.get('video_grid_thw', {}).get('values') == [[90, 32, 48]] for x in grids),
                    'Actual native video grid changed')
        image_count = Counter(c['type'] for c in messages[0]['content'])['image_url']
        if image_count:
            require(any(x.get('image_grid_thw', {}).get('values') == [[1,32,48]] * image_count for x in grids),
                    'Actual image grids changed')
    return {**result, 'native_reader': obs.reads, 'native_processor': obs.processors, 'actual_worker': worker}
