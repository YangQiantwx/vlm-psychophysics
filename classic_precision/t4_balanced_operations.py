"""Independent T4 scope on the already qualified full180 video X deployments.

Preparation/admission is CPU-only. No queue, SSH, GPU query, implicit launch or
retry is implemented here. The root's finite queue owns launch/readiness and
the global non-Gemma barrier. Native inference and input mathematics are reused.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import stat
import sys
import tempfile
import time


NAME = 't4-region-balanced-20261104'
MODULE = 'classic_precision.t4_balanced_operations'
OWNER = os.environ.get('PSYCHOPHYSICS_PARENT_QUEUE_OWNER', 'unbound-parent-queue')
MODELS = ('cosmos', 'qwen', 'nemo', 'gemma')
MODEL_LABELS = {'cosmos': 'cosmos', 'qwen': 'qwen_x_qsa', 'nemo': 'nemo', 'gemma': 'gemma_x_full'}
CONTRACT = {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}
PARENT_FIELDS = ('checkpoint', 'checkpoint_assets', 'processor_checkpoint', 'engine_kwargs', 'environment',
                 'cleared_environment', 'expected_packages', 'python', 'hostname', 'gpu_uuid', 'physical_lock',
                 'request_mm_processor_kwargs', 'request_mm_processor_kwargs_by_task', 'media_io_kwargs',
                 'vision_observation', 'minimum_kv_capacity', 'torch_num_threads', 'input_dependencies',
                 'adapter_image_max_soft_tokens', 'resources')
RUNTIME_FIELDS = ('vllm_version', 'torch_version', 'cuda_version', 'model_class', 'runner_class',
                  'kernel_config', 'actual_backends', 'method_kernel_counts', 'mm_device_do_normalize',
                  'actual_kv_capacity')
DEPENDENCIES = ('official_x_runner', 'official_x_runner_v2', 'official_x_runner_qsa', 'official_x_qsa_selector',
                'refinement_worker', 'storage', 'inference', 'gemma_speed', 'nemo_x_full', 'gemma_x_full',
                'gemma_x_full_input', 'gemma_x_native_input', 'gemma_x_native_compact', 'gemma_x_speed',
                'official_x_gemma_adapter', 'static_x_queue')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def descriptor(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'A regular metadata/source file is required: ' + str(path))
    return {'path': str(path), 'sha256': file_sha(path)}


def verified(ref):
    require(descriptor(ref['path']) == {k: ref[k] for k in ('path', 'sha256')}, 'Bound bytes changed: ' + str(ref['path']))
    return read(ref['path'])


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return descriptor(path)


def claims_gate(p, complete=False):
    """Inspect only this worker's entire own claim namespace; never follow links."""
    root = Path(p['claim_root'])
    require(not root.is_symlink(), 'Claim root is a symlink')
    if not root.exists():
        require(not complete, 'Completed claim namespace is missing')
        return
    require(root.is_dir(), 'Claim namespace is not a directory')
    entries = list(root.iterdir())
    if not complete:
        require(not entries, 'Preexisting/unknown claim entry; no model start or retry')
        return
    expected = {sid + '.json' for sid in p['assigned_ids']}
    require(len(expected) == len(p['assigned_ids']) and {path.name for path in entries} == expected,
            'Completed claim root has missing/extra/unknown entries')
    require(all(not path.is_symlink() and stat.S_ISREG(path.lstat().st_mode) for path in entries),
            'Completed claim root contains a symlink or non-regular claim')


def signature(profile):
    """Only declared post-review operational binding fields are excluded."""
    value = copy.deepcopy(profile)
    for key in ('t4_CPU_admission', 't4_final_review', 'execution_policy'):
        value.pop(key, None)
    intent = value.get('t4_pre_gemma_intent')
    if intent is not None:
        intent.pop('expected_non_gemma_profiles', None)
    return digest(value)


def draft_signature(profile):
    return signature(profile)


def dependency_refs():
    folder = Path(__file__).parent
    return [descriptor(folder / (name + '.py')) for name in DEPENDENCIES]


def validate_authority(auth):
    require(auth['schema'] == 't4-region-balanced-authorization-v1' and auth['owner_chat'] == OWNER
            and auth['domain'] == 'classic' and auth['natural_authorized'] is False
            and auth['models'] == list(MODELS) and set(auth['point_counts']) == {8, 16}, 'Wrong T4 scientific authority')
    budget = auth['proposed_budget']
    require(budget == {'per_N_scored': 384, 'per_N_zero_coherence_bias': 24, 'per_model': 816,
                       'all_four_models_cap': 3264, 'extra_qualification_calls': 0}, 'T4 finite budget changed')


def authorization(p):
    """Metadata only; allocation is globally fixed before any claim or seal."""
    auth = verified(p['t4_authorization'])
    validate_authority(auth)
    plan = verified(p['t4_allocation_plan'])
    require(plan['schema'] == 't4-balanced-global-allocation-v1' and plan['owner_chat'] == OWNER
            and all(plan[key] == p[key] for key in ('t4_authorization', 'source_catalog', 't4_protocol')),
            'T4 allocation common bindings differ')
    catalog = verified(p['source_catalog'])
    expected = set(catalog['items'])
    require(len(expected) == 816, 'T4 source universe is not the complete 816')
    allocations = plan['allocations']
    for model in MODELS:
        own = [x for x in allocations if x['model'] == model]
        ids = [sid for row in own for sid in row['assigned_ids']]
        require(len(ids) == len(set(ids)) == 816 and set(ids) == expected, 'Global ownership repeats or omits new IDs: ' + model)
    label = p['t4_model'] + '-' + p['card']
    found = [row for row in allocations if row['allocation_id'] == label]
    require(len(found) == 1, 'No unique finite T4 worker allocation')
    allocation = found[0]
    require(all(p[key] == allocation[key] for key in ('card', 'assigned_ids', 'output', 'claim_root'))
            and allocation['model'] == p['t4_model'] and p['maximum_model_calls'] == len(allocation['assigned_ids']),
            'Worker differs from global T4 ownership')
    return auth, allocation


def validate_item(sid, item, protocol, protocol_digest=None, protocol_rows=None):
    row = item['row']
    require(item['sample_id'] == sid and item['task'] == row['task'] == 'T4' and row['domain'] == 'classic', 'Foreign source identity/task/domain')
    require(item['video_contract'] == CONTRACT and row['choices'] == ['LEFT', 'RIGHT'], 'Wrong pure full180 video/answer contract')
    content = item['content']
    require([value['type'] for value in content] == ['video', 'text'] and content[1]['text'] == row['prompt'] == protocol['prompt'],
            'T4 must be a single video followed by its exact scientific question')
    require(content[0]['path'] and len(content[0]['sha256']) == 64, 'Unbound source video')
    source = row['source_parameters']
    n, coherence = source['N'], source['coherence']
    require(n in (8, 16) and coherence in [value / 8 for value in range(9)] and 0 <= source['layout_index'] < 24,
            'Foreign T4 N/coherence/layout')
    require(source['signal_count'] == n * coherence and source['diameter_px'] == 56 and source['speed_px_s'] == 32,
            'Scientific source parameters changed')
    eligible = coherence > 0
    require(row['accuracy_eligible'] is eligible and item['accuracy_eligible'] is eligible and
            ((row['truth'] == source['direction'] and row['truth'] in ('LEFT', 'RIGHT')) if eligible else
             (row['truth'] is None and source['direction'] == 'NONE')), 'Zero GT/eligibility or coherent signal direction differs')
    require(item['source_sample']['source_protocol_digest'] == (protocol_digest or digest(protocol)), 'Ready source belongs to another protocol')
    if 'rows' in protocol:
        expected = (protocol_rows or {value['sample_id']: value for value in protocol['rows']}).get(sid)
        require(expected is not None and all(source[key] == expected[key] for key in
                ('N', 'coherence', 'signal_count', 'layout_index', 'direction'))
                and row['truth'] == expected['truth'] and row['accuracy_eligible'] is expected['accuracy_eligible'],
                'Ready source row does not match its frozen protocol row')


def configuration(p):
    """Metadata/source-contract admission only; no processor, weights or CUDA."""
    catalog = verified(p['source_catalog'])
    protocol = verified(p['t4_protocol'])
    require(catalog['schema'] == 't4-balanced-native-video-catalog-v1' and catalog['t4_protocol'] == p['t4_protocol']
            and catalog['source_implementation'] == p['t4_source_implementation'], 'Wrong source version/implementation')
    require(len(catalog['items']) == 816 and len(set(p['assigned_ids'])) == p['maximum_model_calls'], 'Finite source/assignment differs')
    protocol_digest = digest(protocol)
    protocol_rows = {value['sample_id']: value for value in protocol['rows']}
    for sid in p['assigned_ids']:
        require(sid in catalog['items'], 'Foreign allocated source ID')
        validate_item(sid, catalog['items'][sid], protocol, protocol_digest, protocol_rows)
    return catalog


def parent_configuration(p):
    index = verified(p['t4_parent_index'])
    require(index['schema'] == 't4-region-balanced-video-X-parent-bindings-v1' and index['owner_chat'] == OWNER,
            'Wrong video X parent index; static image admission is not a video parent')
    parent = index['parents'][p['t4_parent_key']]
    require(parent['card'] == p['card'] and p['model'] == MODEL_LABELS[p['t4_model']], 'Wrong parent physical card/model')
    original = parent['configuration']
    for field in PARENT_FIELDS:
        require(p.get(field) == original.get(field), 'Qualified video X setting changed: ' + field)
    require(p['engine_kwargs']['limit_mm_per_prompt'].get('video') in (1, {'count': 1, 'num_frames': 180})
            and p.get('media_io_kwargs', {'video': {'video_backend': 'opencv', 'num_frames': 180, 'fps': -1}})
            == {'video': {'video_backend': 'opencv', 'num_frames': 180, 'fps': -1}}, 'Not the qualified all180 video branch')
    require(parent['proof_boundary']['profile_and_runtime_bytes_verified_on_Mac'] is True
            and parent['proof_boundary']['runtime_is_actual_recorded_execution'] is True, 'Parent has declarations without recorded execution')
    return parent


def science(p):
    authorization(p)
    configuration(p)
    parent_configuration(p)
    require(p['automatic_retry'] is False and p['formal_auto_resume'] is False, 'No retries or implicit resume')
    require(descriptor(p['t4_source_implementation']['path']) == p['t4_source_implementation'], 'Scientific source implementation changed')
    admission = verified(p['t4_source_admission'])
    require(admission['schema'] == 't4-region-balanced-source-admission-v1'
            and admission['status'] == 'PASS_T4_CLASSIC816_SOURCES' and admission['owner_chat'] == OWNER
            and admission['full_816'] is True and admission['t4_authorization'] == p['t4_authorization']
            and admission['t4_protocol'] == p['t4_protocol']
            and admission['source_implementation'] == p['t4_source_implementation'], 'Independent source admission is missing or differs')
    review = verified(admission['scientific_review'])
    require(review['schema'] == 't4-region-balanced-astra-source-review-v1'
            and review['status'] == 'PASS_SOURCE_DESIGN_AND_CPU_GENERATOR'
            and review['source']['sha256'] == p['t4_source_implementation']['sha256']
            and review['protocol']['sha256'] == p['t4_protocol']['sha256']
            and review['authorization']['sha256'] == p['t4_authorization']['sha256'],
            'Independent Astra review does not bind this exact source/protocol/authority')
    require(admission['ready_catalog'] == verified(p['source_catalog'])['ready_catalog'], 'Ready source admission/catalog differs')
    generation = verified(admission['full_generation_receipt'])
    require(generation['schema'] == 't4-full816-generation-receipt-v1'
            and generation['status'] == 'PASS_FULL816_RGB8_180_PTS_AND_SOURCE_IDENTITIES'
            and generation['ready_catalog'] == admission['ready_catalog']
            and generation['source_implementation'] == p['t4_source_implementation']
            and generation['t4_protocol'] == p['t4_protocol']
            and generation['source_count'] == 816 and generation['frame_count'] == 146880
            and generation['scored_sources'] == 768 and generation['zero_bias_sources'] == 48
            and generation['complete_LR_pairs'] == 384 and generation['paired_frame0_RGB8_equal'] is True
            and generation['all_PyAV_and_OpenCV_fullRGB8_decode_checks_PASS'] is True,
            'Full generated video/source acceptance receipt differs')
    root = Path(p['workspace_root'])
    expected = root / 'runs/reproducibility' / NAME
    require(Path(p['output']).parent == expected and Path(p['claim_root']).parent == expected
            and Path(p['output']).name == p['t4_model'] + '-' + p['card']
            and Path(p['claim_root']).name == p['t4_model'] + '-' + p['card'] + '-claims', 'Foreign output/claim namespace')


@contextmanager
def t4_task_contract():
    """Task allow-list only; finally restore it in this isolated worker process."""
    from . import gemma_x_full_input as original
    before = original.TASKS
    require(before in (('T1', 'T2', 'T3'), ('T4',)), 'Unexpected shared task guard')
    original.TASKS = ('T4',)
    try:
        yield original
    finally:
        original.TASKS = before


def create(profile, engine=None):
    # No native class/math/processor implementation is copied or replaced here.
    from .gemma_x_full_input import FullInputAdapter

    class T4InputAdapter(FullInputAdapter):
        def __init__(self, p, llm):
            with t4_task_contract():
                super().__init__(p, llm)

        def prepare(self, item, content):
            with t4_task_contract():
                return super().prepare(item, content)

    return T4InputAdapter(profile, engine)


def validate_vision(profile, item, deep):
    from .gemma_x_full_input import validate_vision as original
    return original(profile, item, deep)


def cpu_video_sample(p, item):
    """Actual unchanged native video processor, without constructing an engine."""
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Native CPU admission must have no visible CUDA devices')
    import torch
    from vllm.config import ModelConfig
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from vllm.renderers.chat_utils import parse_chat_messages
    from vllm.renderers.hf import resolve_chat_template_content_format, safe_apply_chat_template
    from . import official_x_runner as base
    require(not torch.cuda.is_initialized(), 'Native CPU admission must not initialize CUDA')
    require(sys.executable == p['python'], 'Native CPU admission must use the exact video deployment interpreter with CUDA hidden')
    for package, version in p['expected_packages'].items():
        require(importlib.metadata.version(package) == version, 'Native CPU deployment package differs: ' + package)
    if p.get('torch_num_threads'):
        torch.set_num_threads(p['torch_num_threads'])
    e = p['engine_kwargs']
    config_kwargs = {key: copy.deepcopy(e[key]) for key in ('dtype', 'trust_remote_code', 'hf_overrides',
        'max_model_len', 'limit_mm_per_prompt', 'mm_processor_cache_gb', 'mm_processor_kwargs') if key in e}
    tokenizer = e.get('tokenizer', p.get('processor_checkpoint') or p['checkpoint'])
    config = ModelConfig(model=p['checkpoint'], tokenizer=tokenizer,
                         allowed_local_media_path=e.get('allowed_local_media_path', p['workspace_root']), **config_kwargs)
    tok = cached_tokenizer_from_config(config)
    native = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tok)
    content_format = resolve_chat_template_content_format(chat_template=None, tools=None, given_format='auto',
        tokenizer=tok, model_config=config)
    with base.NativeObserver(p) as observer:
        observer.active, observer.deep, observer.contract = True, True, CONTRACT
        conversation, mm_data, _ = parse_chat_messages(base.messages(p, item['sample_id'], item), config,
            content_format=content_format, media_io_kwargs=p.get('media_io_kwargs', base.MEDIA),
            mm_processor_kwargs=p.get('request_mm_processor_kwargs', {}))
        require(isinstance(mm_data, dict) and set(mm_data) == {'video'}, 'CPU native path lost or added a modality')
        text = safe_apply_chat_template(config, tok, conversation, chat_template=None, tokenize=False,
            add_generation_prompt=True, continue_final_message=False, tools=None, enable_thinking=False)
        require(isinstance(text, str) and not text.rstrip().endswith('<think>'), 'Native CPU template/thinking differs')
        output = native.apply(ProcessorInputs(prompt=tok.encode(text, add_special_tokens=False),
            mm_data_items=native.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs=p.get('request_mm_processor_kwargs', {})), TimingContext(enabled=False))
        ids = list(output['prompt_token_ids'])
        base.validate_processor(p, item, observer.processors)
        require(len(observer.reads) == 1 and observer.reads[0]['shape'] == [180, 512, 768, 3]
                and observer.reads[0]['dtype'] == 'uint8'
                and observer.reads[0]['metadata']['frames_indices'] == list(range(180)), 'CPU native input did not retain every original video frame')
        placeholders = {kind: [{'offset': int(value.offset), 'length': int(value.length)} for value in values]
                        for kind, values in output['mm_placeholders'].items()}
        require(set(placeholders) == {'video'} and len(placeholders['video']) == 1
                and placeholders['video'][0]['length'] > 0, 'CPU native video token placement differs')
        receipt = {'sample_id': item['sample_id'], 'item_digest': digest(item),
            'native_video_contract': CONTRACT, 'source_reader': observer.reads,
            'native_processor': observer.processors, 'prompt_token_ids': ids,
            'prompt_token_ids_sha256': digest(ids), 'prompt_tokens': len(ids),
            'output_reserve': 16384, 'context': e['max_model_len'],
            'fits_context': len(ids) + 16384 <= e['max_model_len'],
            'mm_placeholders': placeholders, 'native_processor_class': type(native).__module__ + '.' + type(native).__name__,
            'native_processor_source': descriptor(inspect.getsourcefile(type(native))),
            'tokenizer_class': type(tok).__module__ + '.' + type(tok).__name__,
            'source_input_branch': 'video', 'image_inputs': 0, 'weights_loaded': False, 'CUDA_initialized': False}
    require(not torch.cuda.is_initialized(), 'Native CPU processing initialized CUDA')
    return receipt


def cpu(draft, output):
    p = read(draft)
    science(p)
    parent = parent_configuration(p)
    table = None
    native_video = None
    if p['t4_model'] == 'gemma':
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Gemma CPU reference must have no visible CUDA devices')
        require(sys.executable == p['python'], 'Gemma CPU admission must use the exact qualified deployment interpreter with CUDA hidden')
        table_path = Path(output).with_name(Path(output).stem + '-gemma-task-prompts.json')
        with t4_task_contract() as original:
            original.build_prompt_catalog(p, verified(p['source_catalog']), table_path)
        table = descriptor(table_path)
        value = verified(table)
        require(set(value['prompt_token_ids']) == {'T4'}, 'Wrong Gemma T4 prompt reference')
        native_video = {'source_input_branch': 'video', 'image_inputs': 0,
                        'prompt_tokens': len(value['prompt_token_ids']['T4']), 'output_reserve': 16384,
                        'context': p['engine_kwargs']['max_model_len'],
                        'fits_context': len(value['prompt_token_ids']['T4']) + 16384 <= p['engine_kwargs']['max_model_len']}
    else:
        catalog = configuration(p)
        native_video = cpu_video_sample(p, catalog['items'][p['assigned_ids'][0]])
    receipt = {'schema': 't4-video-cpu-admission-v1', 'status': 'PASS_T4_VIDEO_CPU_ADMISSION',
               'owner_chat': OWNER, 'draft_signature': signature(p), 't4_parent_index': p['t4_parent_index'],
               'parent_profile': parent['profile'], 'parent_runtime': parent['runtime'],
               'source_catalog': p['source_catalog'], 't4_protocol': p['t4_protocol'],
               'assigned_count': len(p['assigned_ids']), 'native_video_contract': CONTRACT,
               'reference_prompt_catalog': table, 'source_video_generation_repeated': False,
               'representative_videos_decoded': 1, 'full_source_audit_repeated': False,
               'native_video_CPU': native_video,
               'GPU_queries': 0, 'weights_loaded': False, 'new_model_calls': 0,
               'scope': 'CPU scientific/identity/config admission plus Gemma official task prompt reference; first formal request retains native actual deep validation.'}
    if not native_video['fits_context']:
        receipt['status'] = 'HOLD_T4_VIDEO_CONTEXT_OVERFLOW'
        write_new(output, receipt)
        raise ValueError('Actual native T4 prompt ' + str(native_video['prompt_tokens']) + '+16384 exceeds unchanged context '
                         + str(native_video['context']) + '; preserve this CPU failure and explicitly select another already-qualified video parent')
    return {'status': receipt['status'], 'admission': write_new(output, receipt), 'model_calls': 0}


def final_review(p, draft_signature):
    review = verified(p['t4_final_review'])
    require(review['schema'] == 't4-final-execution-review-v1' and review['status'] == 'PASS_FINAL_T4_EXECUTION_REVIEW'
            and review['owner_chat'] == OWNER, 'Missing independent final T4 execution review')
    for field in ('t4_authorization', 'source_catalog', 't4_protocol', 't4_parent_index', 't4_allocation_plan'):
        require(review[field] == p[field], 'Final review binding differs: ' + field)
    require(review['draft_signatures'][p['worker']] == draft_signature and p['t4_CPU_admission'] in review['cpu_admissions'],
            'Final review does not bind this exact CPU-admitted draft')
    refs = review['code_bundle']
    require(descriptor(__file__) in refs and p['t4_source_implementation'] in refs, 'Final review omits source/operations implementation')
    for ref in refs:
        require(descriptor(ref['path']) == ref, 'Reviewed code bundle changed')
    return review


def verify_cpu(p):
    budget = verified(p['t4_CPU_admission'])
    require(budget['status'] == 'PASS_T4_VIDEO_CPU_ADMISSION' and budget['owner_chat'] == OWNER
            and budget['assigned_count'] == len(p['assigned_ids'])
            and budget['source_catalog'] == p['source_catalog'] and budget['t4_protocol'] == p['t4_protocol']
            and budget['draft_signature'] == signature(p.get('t4_reviewed_draft', p))
            and budget['native_video_CPU']['source_input_branch'] == 'video'
            and budget['native_video_CPU']['image_inputs'] == 0 and budget['native_video_CPU']['fits_context'] is True,
            'Wrong CPU admission/scientific draft/native video context')
    return budget


def seal(draft, output):
    p = read(draft)
    science(p)
    budget = verify_cpu(p)
    require(budget['draft_signature'] == signature(p), 'CPU-admitted scientific draft changed')
    final_review(p, signature(p))
    claims_gate(p)
    from .gemma_x_full import storage_admission
    storage_admission(p)  # CPU filesystem/driver metadata only; no GPU query or lock.
    lock = p['physical_lock']
    state = Path(lock['path']).stat()
    require(not Path(lock['path']).is_symlink() and (state.st_dev, state.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original physical lease identity changed')
    reviewed_draft = copy.deepcopy(p)
    p['execution_policy'] = {'module': MODULE, 'source_sha256': file_sha(__file__)}
    p['t4_reviewed_draft_signature'] = signature(p)
    p['t4_reviewed_draft'] = reviewed_draft
    p['t4_dependencies'] = dependency_refs()
    if p['t4_model'] == 'gemma':
        require(budget['reference_prompt_catalog'], 'Gemma T4 CPU prompt reference missing')
        p['reference_prompt_catalog'] = budget['reference_prompt_catalog']
        p['request_adapter'] = {'module': MODULE, 'function': 'create', 'source_sha256': file_sha(__file__)}
        p['vision_validation_hook'] = {'module': MODULE, 'function': 'validate_vision', 'source_sha256': file_sha(__file__)}
    # The original seal hashes every exact source video; it does not load weights/CUDA.
    from . import official_x_runner_v2 as native
    if p['t4_model'] == 'qwen':
        from . import official_x_runner_qsa as qsa
        sealer = qsa.seal
    else:
        sealer = native.seal
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream)
        temporary = Path(stream.name)
    try:
        return sealer(temporary, output)
    finally:
        temporary.unlink()


def gemma_gate(p):
    intent = p['t4_pre_gemma_intent']
    require(Path(intent['path']) == Path(p['workspace_root']) / 'runs/reproducibility' / NAME /
            ('queue-' + p['card']) / 'pre-gemma-global-closure.json', 'Gemma barrier path differs from its current queue')
    gate = read(intent['path'])
    require(gate['schema'] == 't4-global-pre-gemma-closure-v1'
            and gate['status'] == 'PASS_ALL_NON_GEMMA_T4_REAL_CLOSED' and gate['owner_chat'] == OWNER
            and gate['physical_GPU_lease_held'] is False and gate['GPU_queries'] == 0, 'Global Gemma-last gate not passed')
    for field in ('t4_authorization', 'source_catalog', 't4_protocol'):
        require(intent[field] == gate[field] == p[field], 'Gemma barrier common binding differs')
    expected = intent['expected_non_gemma_profiles']
    require(intent['owner_chat'] == OWNER and len(expected) == 6 and len({x['path'] for x in expected}) == 6
            and gate['expected_non_gemma_profiles'] == expected and len(gate['closures']) == 6,
            'Global Gemma barrier must cover the exact six prior profiles')
    return descriptor(intent['path'])


def runtime_signature(value):
    result = {key: value.get(key) for key in RUNTIME_FIELDS}
    if 'custom_qsa_selector' in value:
        result['custom_qsa_selector'] = {key: value['custom_qsa_selector'].get(key) for key in
            ('status', 'selector_source_sha256', 'mathematics_source_sha256', 'actual_function_code_sha256', 'actual_math_code_sha256')}
    return result


def run(profile):
    p = read(profile)
    science(p)
    verify_cpu(p)
    require(signature(p['t4_reviewed_draft']) == p['t4_reviewed_draft_signature'], 'Reviewed draft signature changed')
    for field, value in p['t4_reviewed_draft'].items():
        if p['t4_model'] == 'gemma' and field in ('request_adapter', 'vision_validation_hook'):
            continue
        require(p.get(field) == value, 'Sealed profile differs from reviewed draft: ' + field)
    if p['t4_model'] == 'gemma':
        require(p['request_adapter'] == {'module': MODULE, 'function': 'create', 'source_sha256': file_sha(__file__)}
                and p['vision_validation_hook'] == {'module': MODULE, 'function': 'validate_vision', 'source_sha256': file_sha(__file__)},
                'Gemma task-only adapter/validation hook changed')
    final_review(p, p['t4_reviewed_draft_signature'])
    require(p['execution_policy'] == {'module': MODULE, 'source_sha256': file_sha(__file__)}, 'T4 operations changed after review/seal')
    for ref in p['t4_dependencies']:
        require(descriptor(ref['path']) == ref, 'Frozen native video dependency changed')
    claims_gate(p)
    from .gemma_x_full import storage_admission
    storage = storage_admission(p)
    if p['t4_model'] == 'gemma':
        gemma_gate(p)
    write_new(Path(p['output']).with_suffix('.storage.json'), storage)
    if p['t4_model'] == 'qwen':
        from . import official_x_runner_qsa as route
        native = route.v2
    else:
        from . import official_x_runner_v2 as native
    base = native.base
    original_identity = base.worker_identity
    if p['t4_model'] == 'nemo':
        from . import nemo_x_full
        original_identity = nemo_x_full.worker_identity
    expected = parent_configuration(p)['runtime_observed']
    expected = expected[0] if isinstance(expected, list) else expected

    def qualified_identity(worker, model):
        value = original_identity(worker, model)
        require(runtime_signature(value) == runtime_signature(expected), 'Actual runtime differs from the bound video deployment before first claim')
        return value

    previous = base.worker_identity
    base.worker_identity = qualified_identity
    try:
        return native.run(profile)
    finally:
        base.worker_identity = previous


def supervise(profile):
    from . import static_x_queue as lifecycle
    from .gemma_speed import owned_worker_alive
    p = read(profile)
    require(sys.executable == p['python'], 'Supervisor must use its sealed deployment interpreter')
    out = Path(p['output'])
    require(not out.exists() and not out.with_suffix('.launch.json').exists()
            and not out.with_suffix('.wait.json').exists(), 'Already attempted supervisor; no retry')
    claims_gate(p)
    env = dict(os.environ)
    for key in p.get('cleared_environment', []):
        env.pop(key, None)
    env.update(p['environment'])
    out.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    error, child_wait, identity, code = None, None, None, None
    try:
        with out.with_suffix('.log').open('xb') as log:
            identity, code = lifecycle.spawn_and_wait([p['python'], '-B', '-m', MODULE, 'run', '--profile', str(profile)],
                log=log, cwd=str(Path(p['implementation']['path']).parent.parent), env=env,
                child_receipt=out.with_suffix('.launch.json'),
                receipt_fields={'owner': lifecycle.process_identity(), 'profile_sha256': digest(p), 'automatic_retry': False},
                identity_key='model_child')
            child_wait = {'identity': identity, 'returncode': code, 'real_wait_observed': True}
    except BaseException as caught:
        error = caught
        child_wait = getattr(caught, 'static_queue_child_wait', None)
        if child_wait is None:
            raise  # No child was created, or the reviewed spawn helper did not return a real wait.
        identity, code = child_wait.get('identity'), child_wait['returncode']
    # A created model process was really waited even on receipt/interrupt failure.
    owned_file, exit_file = out / 'engine-owner.json', out / 'exit.json'
    owned = read(owned_file).get('owned', []) if owned_file.exists() else []
    if exit_file.exists():
        owned = [{'worker': value} for value in read(exit_file).get('owned_workers', [])] or owned
    exited = all(not owned_worker_alive(value['worker']) for value in owned)
    lock, free = p['physical_lock'], False
    lock_path = Path(lock['path'])
    require(lock_path.is_file() and not lock_path.is_symlink(), 'Original physical lease missing after child wait')
    with lock_path.open('r+b') as stream:
        state = os.fstat(stream.fileno())
        require((state.st_dev, state.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original physical lease inode changed')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            free = True
        except BlockingIOError:
            pass
    terminal = read(out / 'result.json') if (out / 'result.json').exists() else {}
    result = {'returncode': code, 'real_wait_observed': True, 'model_child': identity,
              'owned_engines_exited': exited, 'original_physical_lease_released': free,
              'profile_sha256': digest(p), 'supervisor_monotonic_seconds': time.monotonic() - started,
              'terminal_status': terminal.get('status', 'FAILED_OR_UNCERTAIN'),
              'completed_ids': terminal.get('completed_ids', []), 'automatic_retry': False,
              'supervisor_error': {'type': type(error).__name__, 'message': str(error)} if error else None}
    write_new(out.with_suffix('.wait.json'), result)
    if error is not None:
        raise error
    require(code == 0 and exited and free, 'Model child failed or owned exit/original lease release is incomplete')
    return result


def records(path):
    data = Path(path).read_bytes()
    require(data.endswith(b'\n'), 'Incomplete durable journal tail: ' + str(path))
    values = [json.loads(line) for line in data.splitlines()]
    require(all(isinstance(row, dict) for row in values), 'Non-object durable journal')
    return values


def closed(profile):
    p = read(profile)
    science(p)
    claims_gate(p, complete=True)
    out = Path(p['output'])
    wait, result, exit_record = read(out.with_suffix('.wait.json')), read(out / 'result.json'), read(out / 'exit.json')
    require(wait['profile_sha256'] == digest(p) and wait['returncode'] == 0 and wait['real_wait_observed'] is True
            and wait['owned_engines_exited'] is True and wait['original_physical_lease_released'] is True
            and result['status'] == 'COMPLETE' and result['completed_ids'] == p['assigned_ids'] and not result.get('UNCERTAIN_ids')
            and not (out / 'failure.json').exists() and exit_record['owned_engine_workers_exited'] is True
            and exit_record['shutdown_error'] is None, 'No real clean T4 worker closure; never retry unknown claims')
    raw, events, deep = records(out / 'raw.jsonl'), records(out / 'events.jsonl'), records(out / 'deep-observations.jsonl')
    require([row['sample_id'] for row in raw] == p['assigned_ids'] and
            [(row['event'], row['sample_id']) for row in events] == [(event, sid) for sid in p['assigned_ids'] for event in ('INTENT', 'TECHNICAL_PASS')]
            and len(deep) == 1 and deep[0]['sample_id'] == p['assigned_ids'][0], 'Original raw/PASS/deep exact finite journal differs')
    catalog = configuration(p)
    for ordinal, row in enumerate(raw):
        sid = row['sample_id']
        item = catalog['items'][sid]
        claim = read(Path(p['claim_root']) / (sid + '.json'))
        passed = events[ordinal * 2 + 1]
        require(row['profile_sha256'] == claim['profile_sha256'] == digest(p) and claim['attempt'] == row['attempt'] == 1
                and claim['sample_id'] == sid and claim['worker_token'] == p['worker_token'] and row['ordinal'] == ordinal
                and row['task'] == 'T4' and row['question'] == item['row']['prompt'] and row['truth'] == item['row']['truth']
                and row['accuracy_eligible'] is item['row']['accuracy_eligible']
                and row['correct'] == (row['raw_answer'].strip() == row['truth'] if row['accuracy_eligible'] else None)
                and row['raw_answer'].strip() in ('LEFT', 'RIGHT') and row['finish_reason'] == 'stop'
                and passed['raw_digest'] == digest(row) and row['deep'] is (ordinal == 0)
                and row['source']['catalog'] == p['source_catalog'] and row['source']['item_digest'] == digest(item)
                and row['source']['media'][0]['sha256'] == item['content'][0]['sha256']
                and row['generation']['max_tokens'] == 16384 and row['generation']['temperature'] == 0
                and row['generation']['seed'] == 0 and row['generation']['thinking'] is False,
                'Original T4 claim/raw/source/GT/zero/digest/generation differs')
        readers = row['native_reader']
        require(len(readers) == 1 and readers[0]['shape'] == [180, 512, 768, 3] and readers[0]['dtype'] == 'uint8'
                and readers[0]['metadata']['frames_indices'] == list(range(180))
                and readers[0]['metadata']['fps'] == 60.0 and readers[0]['metadata']['duration'] == 3.0,
                'Actual original video reader count/order/shape/time differs')
        if ordinal == 0:
            require(readers[0]['decoded_sha256'] == item['source_sample']['source_evidence']['stream_sha256'],
                    'First actual native video RGB stream differs from the independently generated source')
    require(deep[0]['raw_digest'] == digest(raw[0]), 'First in-budget deep observation is not bound to first raw')
    if p['t4_model'] in ('cosmos', 'qwen', 'gemma'):
        admissions = records(out / 'input-admissions.jsonl')
        require(len(admissions) == 1 and admissions[0]['sample_id'] == raw[0]['sample_id'], 'First native input admission missing/duplicated')
        if p['t4_model'] == 'gemma':
            proof = admissions[0]
            require(proof['status'] == 'PASS' and proof['profile_sha256'] == digest(p)
                    and proof['source_item_digest'] == digest(catalog['items'][raw[0]['sample_id']])
                    and proof['actual_GPU_chunks_equal_independent_CPU'] is True and proof['frame_count'] == 180
                    and proof['reference_tensor_submitted_to_engine'] is False, 'Stored actual Gemma native all180 proof differs')
        else:
            proof = admissions[0]
            require(proof['raw_digest'] == digest(raw[0]) and proof['observations_digest'] == digest(deep[0]['observations'])
                    and proof['proof']['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS'
                    and proof['proof']['CPU_to_GPU_patch_bytes_equal'] is True and proof['proof']['source_reader_all180'] is True,
                    'Stored actual native video CPU/GPU ingress proof differs')
    runtime = read(out / 'runtime.json')
    parent_runtime = parent_configuration(p)['runtime_observed']
    parent_runtime = parent_runtime[0] if isinstance(parent_runtime, list) else parent_runtime
    require(all(runtime_signature(value) == runtime_signature(parent_runtime) for value in runtime), 'Closed runtime deployment differs')
    return {'status': 'COMPLETE_PASS', 'profile': descriptor(profile), 'model': p['t4_model'], 'card': p['card'],
            'completed': len(raw), 'wait': descriptor(out.with_suffix('.wait.json')), 'result': descriptor(out / 'result.json'),
            'same_inode_release_recorded': True, 'model_calls': 0}


def return_manifest(profile, output):
    closure = closed(profile)
    p = read(profile)
    out = Path(p['output'])
    paths = [Path(profile), out.with_suffix('.wait.json')]
    paths += [path for path in (out.with_suffix('.launch.json'), out.with_suffix('.storage.json')) if path.is_file()]
    paths += [Path(p[field]['path']) for field in ('t4_authorization', 'source_catalog', 't4_protocol',
        't4_parent_index', 't4_allocation_plan', 't4_source_admission', 't4_CPU_admission', 't4_final_review')]
    if p.get('reference_prompt_catalog'):
        paths.append(Path(p['reference_prompt_catalog']['path']))
    paths += [path for path in out.iterdir() if path.is_file() and path.suffix in ('.json', '.jsonl')]
    paths += [Path(p['claim_root']) / (sid + '.json') for sid in p['assigned_ids']]
    result = {'schema': 't4-closed-worker-full-return-manifest-v1', 'status': 'READY_FULL_SHA_RETURN_NOT_YET_RETURNED',
              'owner_chat': OWNER, 'closure': closure, 'files': [descriptor(path) for path in sorted(set(paths))],
              'media_or_weights_included': False, 'model_calls': 0}
    return {'status': result['status'], 'manifest': write_new(output, result), 'files': len(paths), 'model_calls': 0}


def summarize(plan, output):
    plan_value = read(plan)
    refs = plan_value.get('global_profiles', plan_value.get('profiles'))
    require(isinstance(refs, list) and refs, 'Explicit global profile refs missing')
    answers, cells, closures, common = {}, defaultdict(list), [], None
    for ref in refs:
        p = verified(ref)
        closure = closed(ref['path'])
        require(not common or common == {field: p[field] for field in ('t4_authorization', 'source_catalog', 't4_protocol')}, 'Global summary mixes scientific versions')
        common = {field: p[field] for field in ('t4_authorization', 'source_catalog', 't4_protocol')}
        closures.append(closure)
        catalog = configuration(p)
        for raw in records(Path(p['output']) / 'raw.jsonl'):
            key = (p['t4_model'], raw['sample_id'])
            require(key not in answers, 'A source was attempted twice for this model')
            answers[key] = raw
            params = catalog['items'][raw['sample_id']]['row']['source_parameters']
            cells[p['t4_model'], params['N'], params['coherence']].append(raw)
    require(len(answers) == 3264 and all(sum(key[0] == model for key in answers) == 816 for model in MODELS), 'Complete four-model816 denominator missing')
    rows = []
    for (model, n, coherence), raw in sorted(cells.items()):
        eligible = [row for row in raw if row['accuracy_eligible']]
        require(len(raw) == (24 if coherence == 0 else 48) and len(eligible) == (0 if coherence == 0 else 48), 'Wrong scored/bias cell denominator')
        rows.append({'model': model, 'N': n, 'coherence': coherence, 'calls': len(raw), 'accuracy_n': len(eligible),
                     'correct': sum(row['correct'] for row in eligible), 'accuracy': sum(row['correct'] for row in eligible) / len(eligible) if eligible else None,
                     'answers': dict(Counter(row['raw_answer'].strip() for row in raw)), 'zero_is_bias_only': coherence == 0})
    result = {'schema': 't4-balanced-four-model-summary-v1', 'status': 'COMPLETE_ORIGINAL_RAW_CLOSED_AUDIT_PASS',
              'owner_chat': OWNER, **common, 'executed_model_calls': 3264, 'extra_qualification_calls': 0,
              'scored': 3072, 'zero_bias': 192, 'closures': closures, 'cells': rows,
              'full_SHA_return_required_before_publication': True, 'model_calls_during_summary': 0}
    return {'status': result['status'], 'summary': write_new(output, result), 'model_calls': 0}


def normalize_ready(ready, protocol_ref, implementation_ref):
    protocol = verified(protocol_ref)
    protocol_digest = digest(protocol)
    protocol_rows = {value['sample_id']: value for value in protocol['rows']}
    require(ready['schema'] == NAME + '-v1-ready-catalog' and ready['protocol_id'] == NAME
            and isinstance(ready['items'], dict) and ready['full_816'] is True and ready['count'] == len(ready['items']) == 816
            and ready['status'] == 'PASS_FINITE_REQUESTED_SOURCE_SET' and ready['protocol_digest'] == digest(protocol)
            and ready['implementation_sha256'] == implementation_ref['sha256'], 'Full source generation/admission is not complete')
    require(set(ready['items']) == set(protocol_rows), 'Ready source IDs differ from the frozen protocol')
    items = {}
    for sid, original in ready['items'].items():
        require(original['sample_id'] == sid and original['task'] == original['row']['task'] == 'T4', 'Duplicate or foreign ready source')
        row = copy.deepcopy(original['row'])
        parameters = row['source_parameters']
        require(parameters['domain'] == 'classic' and parameters['protocol_id'] == NAME,
                'Ready source scientific namespace/domain differs')
        row.update(domain=parameters['domain'], paired_block=parameters['paired_block'], cross_n_pair=parameters['cross_n_pair'])
        evidence = original['ready']
        require({key: evidence[key] for key in ('frames', 'fps', 'width', 'height')} == CONTRACT, 'Ready video dimensions differ')
        require(original['video_contract'] == {**CONTRACT, 'no_resize': True}
                and [value['type'] for value in original['content']] == ['video', 'text']
                and original['content'][0]['sha256'] == evidence['video_sha256'], 'Ready native video modality/SHA/guard differs')
        sample = copy.deepcopy(original)
        sample['source_protocol_digest'] = parameters['source_protocol_digest']
        sample['source_evidence'] = copy.deepcopy(evidence)
        item = {**copy.deepcopy(original), 'domain': parameters['domain'], 'row': row,
                'accuracy_eligible': row['accuracy_eligible'], 'video_contract': CONTRACT.copy(), 'source_sample': sample}
        validate_item(sid, item, protocol, protocol_digest, protocol_rows)
        items[sid] = item
    require(Counter((item['row']['source_parameters']['N'], item['row']['accuracy_eligible']) for item in items.values())
            == Counter({(8, True): 384, (8, False): 24, (16, True): 384, (16, False): 24}), 'Source N/scored/zero denominators differ')
    return items


def prepare(root, ready_catalog, protocol, source_admission, parent_index, authorization_path, cards=('fourier0', 'fourier1')):
    root = Path(root)
    require(tuple(cards) == ('fourier0', 'fourier1'), 'This initial finite plan is two Fourier cards; Radon needs explicit new ownership admission')
    folder = root / 'plans' / NAME
    auth_ref, protocol_ref = descriptor(authorization_path), descriptor(protocol)
    admission_ref, parent_ref = descriptor(source_admission), descriptor(parent_index)
    admission = verified(admission_ref)
    source_ref = admission['source_implementation']
    ready_ref = descriptor(ready_catalog)
    require(admission['ready_catalog'] == ready_ref and admission['t4_protocol'] == protocol_ref, 'Source acceptance binding differs')
    items = normalize_ready(read(ready_catalog), protocol_ref, source_ref)
    catalog_ref = write_new(folder / 'source-catalog.json', {'schema': 't4-balanced-native-video-catalog-v1',
        'ready_catalog': ready_ref, 't4_protocol': protocol_ref, 'source_implementation': source_ref, 'items': items})
    allocations = []
    for card in cards:
        ids = sorted(sid for sid, item in items.items() if item['row']['source_parameters']['layout_index'] % 2 == int(card[-1]))
        require(len(ids) == 408, 'Two-card layout ownership is not408 per model/card')
        for model in MODELS:
            label = model + '-' + card
            allocations.append({'allocation_id': label, 'model': model, 'card': card, 'assigned_ids': ids,
                'output': str(root / 'runs/reproducibility' / NAME / label),
                'claim_root': str(root / 'runs/reproducibility' / NAME / (label + '-claims'))})
    allocation_ref = write_new(folder / 'allocation.json', {'schema': 't4-balanced-global-allocation-v1', 'owner_chat': OWNER,
        't4_authorization': auth_ref, 'source_catalog': catalog_ref, 't4_protocol': protocol_ref, 'allocations': allocations})
    index = verified(parent_ref)
    drafts = []
    for allocation in allocations:
        label = allocation['allocation_id']
        parent = index['parents'][label]
        p = copy.deepcopy(parent['configuration'])
        p.update(model=MODEL_LABELS[allocation['model']], t4_model=allocation['model'], worker=label,
                 card=allocation['card'], workspace_root=str(root), phase='full', sources={},
                 assigned_ids=allocation['assigned_ids'], maximum_model_calls=len(allocation['assigned_ids']),
                 output=allocation['output'], claim_root=allocation['claim_root'], automatic_retry=False, formal_auto_resume=False,
                 pause_file=str(root / 'runs/reproducibility' / NAME / (label + '-pause.json')),
                 choices_by_task={'T4': ['LEFT', 'RIGHT']}, t4_authorization=auth_ref, t4_protocol=protocol_ref,
                 source_catalog=catalog_ref, t4_source_admission=admission_ref, t4_source_implementation=source_ref,
                 t4_parent_index=parent_ref, t4_parent_key=label, t4_allocation_plan=allocation_ref)
        if allocation['model'] == 'gemma':
            p['t4_pre_gemma_intent'] = {'path': str(root / 'runs/reproducibility' / NAME /
                ('queue-' + allocation['card']) / 'pre-gemma-global-closure.json'),
                'owner_chat': OWNER, 't4_authorization': auth_ref, 'source_catalog': catalog_ref, 't4_protocol': protocol_ref,
                'expected_non_gemma_profiles': []}
        science(p)
        drafts.append(write_new(folder / (label + '-draft.json'), p))
    plan = {'schema': 't4-balanced-draft-plan-v1', 'status': 'CPU_DRAFTS_ONLY_NOT_SEALED', 'owner_chat': OWNER,
            't4_authorization': auth_ref, 'source_catalog': catalog_ref, 't4_protocol': protocol_ref,
            't4_parent_index': parent_ref, 't4_allocation_plan': allocation_ref, 'drafts': drafts, 'model_calls': 0}
    return {'status': plan['status'], 'plan': write_new(folder / 'draft-plan.json', plan), 'drafts': drafts, 'model_calls': 0}


def self_test():
    # Validate only local data contracts and review signatures, never import a model runtime.
    p = {'science': 1, 't4_pre_gemma_intent': {'path': '/fixed', 'expected_non_gemma_profiles': []}}
    q = copy.deepcopy(p)
    q.update(t4_CPU_admission={'path': '/cpu', 'sha256': 'x'}, t4_final_review={'path': '/review', 'sha256': 'x'}, execution_policy={'module': MODULE})
    q['t4_pre_gemma_intent']['expected_non_gemma_profiles'] = [{'path': '/sealed', 'sha256': 'x'}]
    require(signature(p) == signature(q), 'Allowed post-review bindings change scientific signature')
    q['science'] = 2
    require(signature(p) != signature(q), 'Scientific mutation escaped review signature')
    protocol = {'prompt': 'coherent signal direction; not necessarily a majority'}
    sample = {'source_protocol_digest': digest(protocol)}
    row = {'sample_id': 'synthetic', 'task': 'T4', 'domain': 'classic', 'truth': None,
           'accuracy_eligible': False, 'choices': ['LEFT', 'RIGHT'], 'prompt': protocol['prompt'],
           'source_parameters': {'N': 8, 'coherence': 0, 'signal_count': 0, 'layout_index': 0,
                                 'diameter_px': 56, 'speed_px_s': 32, 'direction': 'NONE'}}
    item = {'sample_id': 'synthetic', 'task': 'T4', 'row': row, 'accuracy_eligible': False,
            'video_contract': CONTRACT, 'source_sample': sample,
            'content': [{'type': 'video', 'path': '/synthetic.mp4', 'sha256': 'a' * 64}, {'type': 'text', 'text': protocol['prompt']}]}
    validate_item('synthetic', item, protocol)
    item['row']['source_parameters'].update(coherence=1/8, signal_count=1, direction='LEFT')
    item['row'].update(accuracy_eligible=True, truth='LEFT')
    item['accuracy_eligible'] = True
    validate_item('synthetic', item, protocol)
    item['row']['truth'] = 'RIGHT'
    try:
        validate_item('synthetic', item, protocol)
    except ValueError:
        pass
    else:
        raise ValueError('Wrong coherent-signal GT was accepted')
    with tempfile.TemporaryDirectory(prefix='t4-claim-allowlist-selftest-') as temporary:
        root = Path(temporary) / 'claims'
        claim_profile = {'claim_root': str(root), 'assigned_ids': ['one']}
        claims_gate(claim_profile)
        root.mkdir()
        claims_gate(claim_profile)
        (root / 'one.json').write_text('{}\n')
        claims_gate(claim_profile, complete=True)
        for kind in ('extra-file', 'directory', 'symlink'):
            extra = root / 'unknown'
            if kind == 'extra-file':
                extra.write_text('{}\n')
            elif kind == 'directory':
                extra.mkdir()
            else:
                extra.symlink_to(root / 'one.json')
            try:
                claims_gate(claim_profile, complete=True)
            except ValueError:
                pass
            else:
                raise ValueError('Unknown claim entry accepted: ' + kind)
            extra.rmdir() if kind == 'directory' else extra.unlink()
        (root / 'one.json').unlink()
        (root / 'one.json').symlink_to(Path(temporary) / 'outside.json')
        try:
            claims_gate(claim_profile, complete=True)
        except ValueError:
            pass
        else:
            raise ValueError('Assigned-name claim symlink accepted')
    return {'status': 'PASS_CPU_CONTRACT_SELFTEST', 'tests': ['review_signature_allowed_operational_fields',
        'scientific_mutation_changes_signature', 'zero_bias_has_no_GT', 'N8_one_signal_valid_without_majority', 'wrong_signal_GT_rejected',
        'claim_root_absent_or_empty_before_start', 'closed_exact_regular_claim_allowlist', 'unknown_file_directory_symlink_rejected'],
        'GPU_queries': 0, 'model_calls': 0, 'runtime_imported': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'cpu', 'seal', 'run', 'supervise', 'closed', 'return-manifest', 'summarize', 'self-test'))
    for name in ('profile', 'draft', 'output', 'plan', 'root', 'ready-catalog', 'protocol', 'source-admission', 'parent-index', 'authorization'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.root, args.ready_catalog, args.protocol, args.source_admission, args.parent_index, args.authorization)
    elif args.command == 'cpu':
        result = cpu(args.draft, args.output)
    elif args.command == 'seal':
        result = seal(args.draft, args.output)
    elif args.command == 'run':
        result = run(args.profile)
    elif args.command == 'supervise':
        result = supervise(args.profile)
    elif args.command == 'closed':
        result = closed(args.profile)
    elif args.command == 'return-manifest':
        result = return_manifest(args.profile, args.output)
    elif args.command == 'summarize':
        result = summarize(args.plan, args.output)
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
