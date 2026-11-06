"""Finite current-X native 4/1-image admission; never a full-static launcher.

CPU builds sealed prompt/placement/patch references without loading weights.
GPU execution reuses official_x_runner's single-attempt claims, durable raw,
owned-engine shutdown and original physical lease. Only 120 authorized case
slots (15 fixed cases x four models x two independent cold loads) exist; failed claims are never retried.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile

from . import official_x_runner as base
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import digest, file_sha256

MODULE = 'classic_precision.static_x_admission'
MODELS = {'cosmos': 'cosmos', 'qwen_x_qsa': 'qwen', 'nemo': 'nemo', 'gemma_x_full': 'gemma'}
CONTRACT = {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}
_ORIGINAL_IDENTITY = base.worker_identity
_ORIGINAL_INSTALL = base.install_probe
_ORIGINAL_READ = base.read_probe
_CURRENT_ADAPTER = None
_JSON_CACHE = {}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()


def tensor_sha(value):
    import torch
    return sha(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())


def official_source(path, distribution):
    """Verify unmodified installed source against its own wheel RECORD."""
    import base64
    import importlib.metadata
    path = Path(path).resolve()
    dist = importlib.metadata.distribution(distribution)
    entry = next((x for x in dist.files if Path(dist.locate_file(x)).resolve() == path), None)
    require(entry is not None and entry.hash is not None and entry.hash.mode == 'sha256',
            'Installed official source lacks wheel RECORD: ' + str(path))
    encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).decode().rstrip('=')
    require(encoded == entry.hash.value, 'Installed official source differs from wheel RECORD')
    return {**descriptor(path), 'distribution': distribution, 'version': dist.version,
            'wheel_record_sha256': entry.hash.value}


def processor_sources(native, model):
    sources = [official_source(inspect.getsourcefile(type(native)), 'vllm')]
    if model == 'nemo':
        from vllm.transformers_utils.processors import nano_nemotron_vl
        sources.append(official_source(nano_nemotron_vl.__file__, 'vllm'))
    if model == 'gemma':
        from transformers.models.gemma4 import image_processing_gemma4, image_processing_pil_gemma4, processing_gemma4
        for module in (image_processing_gemma4, image_processing_pil_gemma4, processing_gemma4):
            sources.append(official_source(module.__file__, 'transformers'))
    return sources


def cached_verified(ref):
    key = (ref['path'], ref['sha256'])
    state = base.stat(ref['path'])
    if key not in _JSON_CACHE:
        _JSON_CACHE[key] = (state, verified(ref))
    saved_state, value = _JSON_CACHE[key]
    require(state == saved_state, 'Bound JSON changed during execution: ' + ref['path'])
    return value


def model_key(p):
    require(p['model'] in MODELS, 'Only the four bound current-X models are allowed')
    return MODELS[p['model']]


def image_entries(item):
    return [x for x in item['content'] if x['type'] == 'image']


def check_item(item, protocol):
    task, domain = item['task'], item['domain']
    require(task in ('T1', 'T2', 'T3') and domain in ('classic', 'natural'), 'Unknown task/domain')
    spec = protocol['tasks'][task]
    count = spec['images_per_question']
    require(count in (1, 4), 'Only static4/static1 input shapes are authorized')
    require([x['type'] for x in item['content']] == ['image'] * count + ['text'],
            'Image count, content order or modality differs')
    question = item['row']['prompt']
    require(question == spec['prompt_by_domain'][domain]
            and sha(question.encode()) == spec['prompt_sha256_by_domain'][domain]
            and item['content'][-1]['text'] == question, 'Scientific prompt bytes or inherited LF differ')
    original = spec['source_video_prompt_by_domain'][domain]
    require(sha(original.encode()) == spec['source_video_prompt_sha256_by_domain'][domain],
            'Source video prompt hash differs')
    transformed = original
    for replacement in spec['minimal_replacements']:
        require(transformed.count(replacement['old']) == replacement['expected_occurrences'],
                'Literal prompt replacement count differs')
        transformed = transformed.replace(replacement['old'], replacement['new'])
    require(transformed == question, 'Prompt includes edits beyond the explicit protocol replacements')
    require(item['row']['choices'] == ['LEFT', 'RIGHT'] and item['row']['truth'] in ('LEFT', 'RIGHT'),
            'Response collection or truth differs')
    require(item['video_contract'] == CONTRACT
            and item['image_contract'] == {'count': count, 'width': 768, 'height': 512},
            'Source/image geometry differs')
    indices = [x['frame_index'] for x in image_entries(item)]
    require(indices == ([0, 1, 2, 3] if task == 'T1' else [item['onset_frame'] + 15]),
            'Image chronology or event+15 selection differs')
    require(all(isinstance(i, int) and 0 <= i < 180 for i in indices), 'Invalid source frame index')
    return count


def authorization(p):
    """Bind all 120 calls before work, with common durable per-cold claims."""
    a = verified(p['static_authorization'])
    require(a['schema'] == 'static-x-120-call-authorization-v1'
            and a['maximum_model_calls'] == 120 and a['automatic_retry'] is False
            and a['full_static_authorized'] is False, 'Not the finite 120-call authorization')
    ids = a['case_ids']; allocations = a['allocations']
    require(len(ids) == len(set(ids)) == 15, 'Exactly 15 frozen case IDs required')
    require(set(a['cold_gpu_uuids']) == {'0', '1'} and len(set(a['cold_gpu_uuids'].values())) == 2
            and all(x['gpu_uuid'] == a['cold_gpu_uuids'][str(x['cold'])] for x in allocations),
            'Each cold must use its one fixed authorized Fourier GPU')
    slots = [(x['model'], x['cold'], sid) for x in allocations for sid in x['assigned_ids']]
    require(len(slots) == len(set(slots)) == 120
            and set(slots) == {(m, cold, sid) for m in MODELS.values() for cold in (0, 1) for sid in ids},
            'Authorization must contain each model/cold/case exactly once')
    require(len({(x['claim_root'], sid) for x in allocations for sid in x['assigned_ids']}) == 120,
            'Each authorized call needs a unique durable claim')
    own = [x for x in allocations if x['model'] == model_key(p) and x['cold'] == p['static_cold']
           and x['assigned_ids'] == p['assigned_ids']]
    require(len(own) == 1 and p['maximum_model_calls'] == len(p['assigned_ids']), 'Unallocated calls')
    row = own[0]
    require(row['output'] == p['output'] and row['claim_root'] == p['claim_root']
            and row['parent_profile_sha256'] == p['parent_validated_profile']['sha256']
            and row['gpu_uuid'] == p['gpu_uuid'], 'Allocation path, GPU or parent profile differs')
    return a, row


def native_kwargs(p):
    model = model_key(p)
    if model == 'gemma':
        return {'do_resize': False, 'do_sample_frames': False, 'max_soft_tokens': 280}
    if model in ('cosmos', 'qwen'):
        return {'do_resize': False, 'do_sample_frames': False}
    # Official Nemo has no resize kwarg; the bound no-resize adapter enforces it.
    return {}


def expected_engine(parent, model):
    e = copy.deepcopy(parent['engine_kwargs'])
    e['limit_mm_per_prompt'] = {'image': 4, 'video': 0, 'audio': 0}
    e['mm_processor_kwargs'] = native_kwargs({'model': next(k for k, v in MODELS.items() if v == model)})
    return e


def configuration(p):
    parent = verified(p['parent_validated_profile'])
    require(not p.get('static_cpu_template_map'), 'Qualification requires a true CPU reference for every case')
    model = model_key(p)
    require(model_key(parent) == model and p['engine_kwargs'] == expected_engine(parent, model),
            'Engine differs from bound video profile beyond declared image count/processor kwargs')
    for field in ('checkpoint', 'checkpoint_assets', 'processor_checkpoint', 'python', 'environment',
                  'cleared_environment', 'expected_packages', 'vision_observation', 'bootstrap',
                  'minimum_kv_capacity', 'media_io_kwargs', 'gpu_uuid', 'physical_lock', 'card',
                  'hostname', 'resources', 'torch_num_threads'):
        require(p.get(field) == parent.get(field), 'Inherited video setting changed: ' + field)
    require(p.get('request_mm_processor_kwargs', {}) == native_kwargs(p)
            and not p.get('request_mm_processor_kwargs_by_task'), 'Image kwargs differ')
    require(not p.get('bootstrap'), 'Unexpected bootstrap')
    adapter = p.get('request_adapter')
    if model == 'gemma':
        require(adapter == {'module': MODULE, 'function': 'create_gemma', 'source_sha256': file_sha256(__file__)},
                'Gemma image adapter must be this explicitly bound entry')
    else:
        require(not adapter, 'Native image route does not allow an input bridge')
    if model == 'qwen':
        require(p.get('selector_source') == parent.get('selector_source'), 'Bound X QSA selector changed')
    catalog = verified(p['source_catalog'])
    protocol = verified(p['static_protocol'])
    auth, allocation = authorization(p)
    require(Counter(catalog['items'][s]['task'] for s in auth['case_ids']) == {'T1': 5, 'T2': 5, 'T3': 5},
            'Exactly five fixed source cases per task are required')
    require(len(set(p['assigned_ids'])) == len(p['assigned_ids']), 'Duplicate sample IDs')
    counts = []
    for sid in p['assigned_ids']:
        item = catalog['items'][sid]
        require(item['sample_id'] == sid, 'Catalog identity mismatch')
        count = check_item(item, protocol)
        counts.append(count)
    tasks = {catalog['items'][s]['task'] for s in p['assigned_ids']}
    if model == 'qwen':
        require((tasks == {'T1'} and len(counts) == 5) or (tasks == {'T2', 'T3'} and len(counts) == 10),
                'Qwen must preserve its task-specific video resource profiles')
        context, kv = (98304, 8 * 2**30) if tasks == {'T1'} else (65536, 4 * 2**30)
        require(p['engine_kwargs']['max_model_len'] == parent['engine_kwargs']['max_model_len'] == context
                and p['engine_kwargs']['kv_cache_memory_bytes'] == parent['engine_kwargs']['kv_cache_memory_bytes'] == kv,
                'Qwen task mapped to the wrong qualified video context/KV profile')
    else:
        require(len(counts) == 15, 'Each other model cold load must cover all fifteen fixed cases')
    return catalog


class NoSpatialTransform:
    """Fail before processor resize/crop; guard is local to the CPU process."""
    def __enter__(self):
        self.saved = []
        from PIL import Image
        import torch.nn.functional as tf
        modules = [(Image.Image, ('resize', 'crop')), (tf, ('interpolate',))]
        for name in ('torchvision.transforms.functional', 'torchvision.transforms.v2.functional',
                     'transformers.image_transforms'):
            module = importlib.import_module(name)
            modules.append((module, ('resize', 'crop', 'center_crop', 'resized_crop', 'five_crop', 'ten_crop')))
        def forbidden(*args, **kwargs):
            raise ValueError('Input resize/crop/interpolation was attempted; no model call may qualify')
        for module, names in modules:
            for name in names:
                if hasattr(module, name):
                    self.saved.append((module, name, getattr(module, name)))
                    setattr(module, name, forbidden)
        return self
    def __exit__(self, *_):
        for module, name, original in reversed(self.saved):
            setattr(module, name, original)


def load_images(item, verify_video=False):
    import numpy as np
    from PIL import Image
    from vllm.multimodal.media.image import ImageMediaIO
    images, receipts = [], []
    frames = None
    if verify_video:
        from vllm.multimodal.media.video import VideoMediaIO
        source = item['original_source']
        require(file_sha256(source['path']) == source['sha256'], 'Original video bytes changed')
        frames, metadata = VideoMediaIO(ImageMediaIO(), video_backend='opencv', num_frames=180, fps=-1).load_file(Path(source['path'])).media
        require(frames.dtype == np.uint8 and list(frames.shape) == [180, 512, 768, 3]
                and metadata['frames_indices'] == list(range(180)), 'Video source decode/order differs')
    for entry in image_entries(item):
        require(file_sha256(entry['path']) == entry['sha256'], 'Source PNG bytes changed')
        with Image.open(entry['path']) as png:
            require(png.mode == 'RGB' and png.size == (768, 512) and not png.info,
                    'PNG must be untagged RGB8 at native resolution')
        image = ImageMediaIO().load_file(Path(entry['path'])).media
        pixels = np.asarray(image)
        require(image.mode == 'RGB' and pixels.dtype == np.uint8 and list(pixels.shape) == [512, 768, 3],
                'Official image reader changed mode/dtype/size')
        checksum = sha(pixels.tobytes())
        require(checksum == entry['pixel_sha256'], 'PNG reader pixels changed')
        if frames is not None:
            require(np.array_equal(pixels[0, 0], frames[0, 0, 0]),
                    'Image corner padding color differs from video frame-zero padding')
            require(np.array_equal(pixels, frames[entry['frame_index']]),
                    'PNG pixels differ from corresponding official video RGB8 frame')
        images.append(image)
        receipts.append({'path': entry['path'], 'frame_index': entry['frame_index'],
                         'shape': list(pixels.shape), 'dtype': str(pixels.dtype), 'pixel_sha256': checksum,
                         'matches_original_video_RGB8': frames is not None,
                         'video_padding_RGB': frames[0, 0, 0].tolist() if frames is not None else None})
    return images, receipts


def chunks(value, model):
    """Return complete per-image views at the documented native vision boundary."""
    shape = list(value.shape)
    if model in ('cosmos', 'qwen'):
        require(value.dtype.__str__() == 'torch.uint8' and len(shape) == 2
                and shape[1] == 1536 and shape[0] % 1536 == 0, 'Native raw image patch geometry differs')
        return list(value.split(1536, dim=0))
    if model == 'nemo':
        if shape == [3, 512, 768]:
            from vllm.transformers_utils.processors.nano_nemotron_vl import DynamicResolutionImageTiler
            return [DynamicResolutionImageTiler.stack([value], 16)[0]]
        if len(shape) == 3 and shape[0] == 1 and shape[2] == 768 and shape[1] % 1536 == 0:
            return list(value[0].split(1536, dim=0))
        require(len(shape) == 4 and shape[1:] == [3, 512, 768], 'Nemo native image tensor resized/cropped')
        from vllm.transformers_utils.processors.nano_nemotron_vl import DynamicResolutionImageTiler
        return [DynamicResolutionImageTiler.stack([image], 16)[0] for image in value]
    if shape == [1584, 768]:
        return [value]
    require(len(shape) == 3 and shape[1:] == [1584, 768], 'Gemma valid patch count differs')
    return list(value.unbind(0))


def payload_description(value, model):
    import torch
    if isinstance(value, torch.Tensor):
        d = base.tensor_description(value, True)
        if value.ndim >= 2:
            shape = list(value.shape)
            is_pixels = ((model in ('cosmos', 'qwen') and value.dtype == torch.uint8 and shape[-1] == 1536)
                         or (model == 'nemo' and (shape == [3, 512, 768] or shape[-3:] == [3, 512, 768]
                            or (len(shape) == 3 and shape[0] == 1 and shape[-1] == 768 and shape[1] % 1536 == 0)))
                         or (model == 'gemma' and shape[-2:] == [1584, 768]))
            if is_pixels:
                d['per_image_sha256'] = [tensor_sha(x) for x in chunks(value, model)]
            if model == 'gemma' and shape[-2:] == [1584, 2]:
                d['per_image_position_sha256'] = [tensor_sha(x) for x in (value.unbind(0) if value.ndim == 3 else [value])]
        return d
    if isinstance(value, (list, tuple)):
        return [payload_description(x, model) for x in value]
    if isinstance(value, dict) or hasattr(value, 'items'):
        return {str(k): payload_description(v, model) for k, v in value.items()}
    if hasattr(value, 'data'):
        return payload_description(value.data, model)
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    raise ValueError('Unknown native processor payload type: ' + str(type(value)))


def leaves(value):
    if isinstance(value, dict):
        if 'shape' in value and 'dtype' in value:
            yield value
        else:
            for x in value.values():
                yield from leaves(x)
    elif isinstance(value, list):
        for x in value:
            yield from leaves(x)


def ordered_hashes(description, field='per_image_sha256'):
    return [checksum for value in leaves(description) for checksum in value.get(field, [])]


def placeholders(value):
    result = {}
    for modality, rows in value.items():
        result[modality] = []
        for row in rows:
            entry = {'offset': int(row.offset), 'length': int(row.length)}
            if getattr(row, 'is_embed', None) is not None:
                mask = row.is_embed
                entry['is_embed'] = mask.tolist() if hasattr(mask, 'tolist') else list(mask)
            result[modality].append(entry)
    return result


def template(processor, item):
    content = [{'type': 'image'} for _ in image_entries(item)] + [{'type': 'text', 'text': item['row']['prompt']}]
    text = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    require(not text.rstrip().endswith('<think>'), 'Thinking unexpectedly enabled')
    return text


def cpu_sample(p, item):
    import torch
    from transformers import AutoProcessor
    from vllm.tokenizers import cached_tokenizer_from_config
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    require(not torch.cuda.is_initialized(), 'CPU admission must not initialize CUDA')
    e = p['engine_kwargs']; model = model_key(p)
    tokenizer = e.get('tokenizer', p.get('processor_checkpoint') or p['checkpoint'])
    config_kwargs = {k: copy.deepcopy(e[k]) for k in ('dtype', 'trust_remote_code', 'hf_overrides',
                    'max_model_len', 'limit_mm_per_prompt', 'mm_processor_cache_gb', 'mm_processor_kwargs') if k in e}
    config = ModelConfig(model=p['checkpoint'], tokenizer=tokenizer,
                         allowed_local_media_path=p['workspace_root'], **config_kwargs)
    tok = cached_tokenizer_from_config(config)
    hf = AutoProcessor.from_pretrained(tokenizer, local_files_only=True, trust_remote_code=False) if model == 'gemma' else None
    images, readers = load_images(item, verify_video=True)
    with ExitStack() as stack:
        nemo = stack.enter_context(NemoNoResize()) if model == 'nemo' else None
        if model == 'gemma':
            adapter = stack.enter_context(GemmaImageAdapter(p))
            images, padding = adapter.pad_images(images)
        else:
            padding = []
        native = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=tok)
        if model == 'gemma':
            text = template(hf, item)
        else:
            from vllm.renderers.chat_utils import parse_chat_messages
            from vllm.renderers.hf import resolve_chat_template_content_format, safe_apply_chat_template
            content_format = resolve_chat_template_content_format(chat_template=None, tools=None, given_format='auto',
                tokenizer=tok, model_config=config)
            conversation, _, _ = parse_chat_messages(base.messages(p, item['sample_id'], item), config,
                content_format=content_format, media_io_kwargs=None, mm_processor_kwargs=native_kwargs(p))
            text = safe_apply_chat_template(config, tok, conversation, chat_template=None, tokenize=False,
                add_generation_prompt=True, continue_final_message=False, tools=None, enable_thinking=False)
        require(isinstance(text, str) and not text.rstrip().endswith('<think>'), 'Native template/thinking differs')
        nemo_equivalence = nemo.check_reference(native, images) if nemo else []
        with NoSpatialTransform():
            output = native.apply(ProcessorInputs(prompt=tok.encode(text, add_special_tokens=False),
                mm_data_items=native.info.parse_mm_data({'image': images}),
                hf_processor_mm_kwargs=native_kwargs(p)), TimingContext(enabled=False))
        payload = payload_description(output['mm_kwargs'], model)
        ids = list(output['prompt_token_ids'])
        positions = placeholders(output['mm_placeholders'])
        hashes = ordered_hashes(payload)
        count = len(images)
        require(len(hashes) == count and set(positions) == {'image'} and len(positions['image']) == count,
                'Native processor lost/reordered logical image placements or patch payloads')
        require(all(x['length'] > 0 for x in positions['image']) and all(
            left['offset'] + left['length'] <= right['offset'] for left, right in zip(positions['image'], positions['image'][1:])),
            'Logical image prompt positions overlap or are out of order')
        if model in ('cosmos', 'qwen'):
            grids = [x['values'] for x in leaves(payload) if x.get('shape') == [3] and 'values' in x]
            require(grids == [[1, 32, 48]] * count, 'Native image grid differs from 768x512')
        if model == 'gemma':
            require(len(ordered_hashes(payload, 'per_image_position_sha256')) == count,
                    'Gemma image patch positions missing')
        require(len(ids) + 16384 <= e['max_model_len'], 'Input plus unchanged output budget exceeds video context')
        require(not torch.cuda.is_initialized(), 'CPU qualification initialized CUDA')
        return {'sample_id': item['sample_id'], 'item_digest': digest(item), 'model': model,
                'source_reader': readers, 'image_count': count, 'padding': padding,
                'nemo_same_size_reference': nemo_equivalence,
                'prompt_token_ids': ids, 'prompt_token_ids_sha256': sha(json_bytes(ids)),
                'prompt_tokens': len(ids), 'template_text_sha256': sha(text.encode()),
                'mm_placeholders': positions, 'native_payload': payload,
                'native_processor_class': type(native).__module__ + '.' + type(native).__name__,
                'native_processor_source': descriptor(inspect.getsourcefile(type(native))),
                'official_source_identity': processor_sources(native, model),
                'hf_processor_class': (type(hf).__module__ + '.' + type(hf).__name__) if hf else None,
                'tokenizer_class': type(tok).__module__ + '.' + type(tok).__name__,
                'per_image_patch_sha256': hashes, 'processor_resize_crop_calls': 0,
                'fits_context': True, 'CUDA_initialized': False}


def cpu(draft, output):
    import torch
    p = read(draft); catalog = configuration(p)
    require(not torch.cuda.is_initialized(), 'CPU process already initialized CUDA')
    attempts = [cpu_sample(p, catalog['items'][s]) for s in p['assigned_ids']]
    result = {'schema': 'static-x-native-cpu-admission-v1', 'status': 'PASS', 'model_calls': 0,
              'weights_loaded': False, 'CUDA_initialized': False, 'source_catalog': p['source_catalog'],
              'static_protocol': p['static_protocol'], 'parent_validated_profile': p['parent_validated_profile'],
              'engine_kwargs_digest': digest(p['engine_kwargs']), 'entry': descriptor(__file__), 'attempts': attempts}
    write_new(output, result)
    return {'status': 'CPU_PASS_GPU_PENDING', 'cpu_budget': descriptor(output), 'model_calls': 0,
            'prompt_tokens': {x['sample_id']: x['prompt_tokens'] for x in attempts}}


class NemoNoResize:
    """Skip only the official image tiler's identity-sized interpolation."""
    def __init__(self, verify_equivalence=False):
        from vllm.transformers_utils.processors import nano_nemotron_vl as module
        self.module = module
        self.original = module.DynamicResolutionImageTiler.apply_params
        self.verify_equivalence = verify_equivalence
        self.receipts = []
        owner = self
        def apply(tiler, params, dtype=None):
            import torch
            dtype = torch.float32 if dtype is None else dtype
            target = (params.patch_size[1] * tiler._patch_size, params.patch_size[0] * tiler._patch_size)
            require(params.media.mode == 'RGB' and params.media.size == (768, 512)
                    and target == (512, 768) and params.patch_size == (48, 32)
                    and tiler._patch_size == 16 and params.num_tiles == 1 and params.num_embeddings == 384,
                    'Nemo official dynamic image plan would resize/crop/tile; HOLD')
            source = module._pil_to_nhwc_tensor(params.media)
            require(source.dtype == torch.uint8 and list(source.shape) == [1, 512, 768, 3],
                    'Nemo official raw NHWC conversion differs')
            actual = module._bicubic_resize_and_normalize(source, size=None,
                norm_mean=tiler.norm_mean, norm_std=tiler.norm_std, dtype=dtype)
            equal = None
            if owner.verify_equivalence:
                # CPU-only independent check of the unmodified same-size path.
                # Run outside NoSpatialTransform, then require exact bytes.
                expected = owner.original(tiler, params, dtype=dtype)
                equal = len(expected) == 1 and torch.equal(expected[0], actual[0])
                require(equal, 'Nemo skipped identity resize changes official normalized bytes; HOLD')
            owner.receipts.append({'source_RGB8_sha256': tensor_sha(source), 'target_hw': list(target),
                'patch_grid_hw': [32, 48], 'patches': 1536, 'image_tokens': 384,
                'dtype': str(actual.dtype), 'normalized_sha256': tensor_sha(actual[0]),
                'official_same_size_interpolation_bytes_equal': equal,
                'actual_interpolation_performed': False})
            return list(actual)
        module.DynamicResolutionImageTiler.apply_params = apply

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.module.DynamicResolutionImageTiler.apply_params = self.original

    def check_reference(self, native, images):
        """Compare official identity interpolation before activating hard guard."""
        processor = native.info.get_hf_processor(**native_kwargs({'model': 'nemo'}))
        tiler = processor.dynamic_tiler
        require(tiler is not None, 'Nemo image path must use its native dynamic-resolution tiler')
        params = tiler.compute_params(images, tiler.max_num_tokens_available(1024))
        require(len(params) == len(images), 'Nemo reference plan lost an image')
        self.verify_equivalence = True
        try:
            for param in params:
                tiler.apply_params(param, dtype=processor.dtype)
        finally:
            self.verify_equivalence = False
        return list(self.receipts)


class GemmaImageAdapter:
    """Reuse X's no-resize count formula and masked-tail removal for images.

    Official native HF processing computes the pixels. This does not supply
    cached CPU reference tensors, alter vision forward, or change pooling.
    """
    def __init__(self, p, engine=None):
        global _CURRENT_ADAPTER
        import vllm.model_executor.models.gemma4_mm as gm
        from transformers import AutoProcessor
        require(_CURRENT_ADAPTER is None, 'Only one Gemma image adapter may be active')
        self.p = p; self.gm = gm; self.current = None; self.receipt = None
        self.processor = AutoProcessor.from_pretrained(p['processor_checkpoint'], local_files_only=True,
                                                       trust_remote_code=False)
        self.original_count = gm.Gemma4ProcessingInfo._compute_num_soft_tokens
        self.native_class = gm.Gemma4MultiModalProcessor
        self.original_apply = self.native_class._apply_hf_processor_main
        original_count, original_apply, owner = self.original_count, self.original_apply, self
        def count(info, image_width, image_height, max_soft_tokens=None):
            if info.ctx.get_merged_mm_kwargs({}).get('do_resize') is False:
                require(image_width % 48 == image_height % 48 == 0, 'Gemma no-resize grid misaligned')
                return image_width * image_height // (48 * 48)
            return original_count(info, image_width, image_height, max_soft_tokens)
        def compact(native, mm_items, hf_kwargs):
            import torch
            result = original_apply(native, mm_items, hf_kwargs)
            if 'pixel_values' not in result:
                return result
            require('pixel_values_videos' not in result, 'Static Gemma cannot contain video')
            pixels, positions = result['pixel_values'], result['pixel_position_ids']
            require(isinstance(pixels, torch.Tensor) and isinstance(positions, torch.Tensor)
                    and pixels.ndim == 3 and positions.ndim == 3
                    and list(pixels.shape[1:]) == [2520, 768]
                    and list(positions.shape) == [pixels.shape[0], 2520, 2],
                    'Gemma native image budget/layout differs')
            valid = (positions != -1).any(-1)
            require((valid.sum(-1) == 1584).all() and valid[:, :1584].all() and not valid[:, 1584:].any(),
                    'Gemma compaction would remove valid/noncontiguous patches')
            result['pixel_values'] = pixels[:, :1584].contiguous()
            result['pixel_position_ids'] = positions[:, :1584].contiguous()
            if owner.receipt is not None:
                owner.receipt['native_compacted_payload'] = payload_description({
                    'pixel_values': result['pixel_values'], 'pixel_position_ids': result['pixel_position_ids']}, 'gemma')
                owner.receipt['native_compaction_calls'] += 1
            return result
        gm.Gemma4ProcessingInfo._compute_num_soft_tokens = count
        self.native_class._apply_hf_processor_main = compact
        _CURRENT_ADAPTER = self

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def pad_images(self, images):
        import numpy as np
        from PIL import Image
        from .official_x_gemma_adapter import pad
        padded, receipts = [], []
        for image in images:
            source = np.asarray(image)
            pixels, receipt = pad(source)
            require(np.array_equal(pixels[:512, :768], source) and pixels.shape == (528, 768, 3)
                    and receipt['bottom_rows'] == 16 and receipt['right_columns'] == 0,
                    'Gemma padding changed effective source pixels')
            padded.append(Image.fromarray(pixels)); receipts.append(receipt)
        return padded, receipts

    def prepare(self, item, content):
        require(self.current is None, 'Previous Gemma image request was not cleared')
        images, readers = load_images(item)
        padded, padding = self.pad_images(images)
        self.current = item['sample_id']
        self.receipt = {'route_label': 'X-native-static-image-count-fix-and-masked-tail-compaction',
            'native_reader': readers, 'image_padding': padding, 'image_max_soft_tokens': 280,
            'actual_soft_tokens_per_image': 176, 'native_compaction_calls': 0,
            'shared_SDK_files_modified': False, 'precomputed_tensor_bridge': False,
            'model_forward_pooling_changed': False}
        return {'request': {'prompt': template(self.processor, item), 'multi_modal_data': {'image': padded},
                            'mm_processor_kwargs': native_kwargs(self.p)}, 'receipt': self.receipt}

    def validate(self, answer):
        expected = budget_for(self.p, self.current)
        require(self.receipt['native_compaction_calls'] > 0
                and list(answer.prompt_token_ids) == expected['prompt_token_ids'],
                'Gemma native compaction or exact prompt expansion missing')
        return {'status': 'PASS'}

    def clear(self):
        self.current = None; self.receipt = None

    def close(self):
        global _CURRENT_ADAPTER
        self.native_class._apply_hf_processor_main = self.original_apply
        self.gm.Gemma4ProcessingInfo._compute_num_soft_tokens = self.original_count
        self.clear()
        if _CURRENT_ADAPTER is self:
            _CURRENT_ADAPTER = None


def create_gemma(p, engine=None):
    return GemmaImageAdapter(p, engine)


def budget_for(p, sid):
    budget = cached_verified(p['static_cpu_budget'])
    reference_sid = p.get('static_cpu_template_map', {}).get(sid, sid)
    if reference_sid != sid:
        require(p['execution_policy']['module'] != MODULE, 'Qualification cannot reuse template-only CPU references')
        catalog = cached_verified(p['source_catalog'])['items']
        actual, reference = catalog[sid], catalog[reference_sid]
        for key in ('task', 'domain', 'image_contract'):
            require(actual[key] == reference[key], 'CPU template mapping changes task/domain/image shape')
        require(actual['row']['prompt'] == reference['row']['prompt']
                and actual['row']['choices'] == reference['row']['choices'], 'CPU template mapping changes exact prompt/choices')
        check_item(actual, cached_verified(p['static_protocol']))
    result = next(x for x in budget['attempts'] if x['sample_id'] == reference_sid)
    require(len(result['mm_placeholders']['image']) == result['image_count'], 'Mapped CPU reference image count differs')
    return result


class ImageObserver(base.NativeObserver):
    def __init__(self, p):
        super().__init__(p)
        self.model = model_key(p)
        self.applications = []
        catalog = cached_verified(p['source_catalog'])
        self.lookup = {str(Path(x['path'])): x for sid in p['assigned_ids']
                       for x in image_entries(catalog['items'][sid])}

    def clear(self):
        super().clear()
        self.applications = []

    def describe(self, data):
        return {k: payload_description(v, self.model) for k, v in data.items()
                if 'pixel' in k or 'grid' in k or 'position' in k or 'size' in k or 'patch' in k}

    def __enter__(self):
        import numpy as np
        from vllm.multimodal.media.image import ImageMediaIO
        super().__enter__()
        self.original_image_load = ImageMediaIO.load_file
        owner = self
        def load(loader, path):
            result = owner.original_image_load(loader, path)
            if owner.active:
                key = str(Path(path)); require(key in owner.lookup, 'Reader attempted unsealed image')
                entry = owner.lookup[key]; image = result.media; pixels = np.asarray(image)
                require(image.mode == 'RGB' and pixels.dtype == np.uint8 and pixels.shape == (512, 768, 3),
                        'Actual ImageMediaIO changed mode, dtype or spatial resolution')
                checksum = sha(pixels.tobytes())
                require(checksum == entry['pixel_sha256'], 'Actual reader changed source RGB8 bytes')
                owner.reads.append({'kind': 'image', 'path': key,
                    'shape': list(pixels.shape), 'dtype': str(pixels.dtype), 'pixel_sha256': checksum})
            return result
        ImageMediaIO.load_file = load
        attempt = verified(self.p['static_cpu_budget'])['attempts'][0]
        module_name, class_name = attempt['native_processor_class'].rsplit('.', 1)
        self.processor_class = getattr(importlib.import_module(module_name), class_name)
        require(file_sha256(inspect.getsourcefile(self.processor_class)) == attempt['native_processor_source']['sha256'],
                'Native processor implementation changed after CPU admission')
        self.had_own_apply = 'apply' in self.processor_class.__dict__
        self.original_apply = self.processor_class.apply
        def apply(native, *args, **kwargs):
            with NoSpatialTransform():
                result = owner.original_apply(native, *args, **kwargs)
            if owner.active:
                owner.applications.append({'prompt_token_ids': list(result['prompt_token_ids']),
                    'mm_placeholders': placeholders(result['mm_placeholders']),
                    'native_payload': payload_description(result['mm_kwargs'], owner.model)})
            return result
        self.processor_class.apply = apply
        return self

    def __exit__(self, *args):
        from vllm.multimodal.media.image import ImageMediaIO
        ImageMediaIO.load_file = self.original_image_load
        if self.had_own_apply:
            self.processor_class.apply = self.original_apply
        else:
            delattr(self.processor_class, 'apply')
        return super().__exit__(*args)


def worker_identity(worker, model):
    import torch
    result = _ORIGINAL_IDENTITY(worker, model)
    device = torch.cuda.get_device_properties(torch.cuda.current_device())
    result['gpu_uuid'] = str(device.uuid)
    result['gpu_architecture'] = {'name': device.name, 'compute_capability': [device.major, device.minor]}
    result['gpu_total_memory_bytes'] = int(device.total_memory)
    family = MODELS[model]
    if family == 'qwen':
        from . import official_x_qsa_selector as selector
        result['custom_qsa_selector'] = selector.evidence()
        from .qwen_x_crosscard import validate_runtime
        validate_runtime(result)
    elif family == 'nemo':
        from .nemo_x_full import validate_runtime
        validate_runtime(result)
    elif family == 'gemma':
        from .gemma_x_full import validate_runtime
        validate_runtime(result)
    else:
        require(any('FlashInferCutlassNvFp4LinearKernel' in key and n > 0
                    for key, n in result['method_kernel_counts'].items()), 'Cosmos NVFP4 runtime differs')
    return result


def validate_parent_runtime(p, runtime):
    if model_key(p) != 'cosmos':
        return
    from .cosmos_t45 import RUNTIME_FIELDS
    expected = cached_verified(p['parent_runtime'])
    require(len(runtime) == len(expected) == 1, 'Cosmos TP1 runtime count differs')
    require(all(runtime[0][field] == expected[0][field] for field in RUNTIME_FIELDS),
            'Cosmos native kernel/backend/normalization differs from bound video runtime')


def install_probe(worker, spec, model):
    import torch
    result = _ORIGINAL_INSTALL(worker, spec, model)
    family = MODELS[model]
    state = worker._official_x_probe
    _, outer = base.resolve_module(worker.get_model(), spec['outer_paths'])
    state['static_images'] = []
    def observed(module, args, kwargs):
        value = base.first_tensor(args, kwargs)
        require(value is not None and value.device.type == 'cuda', 'Actual vision input missing CUDA tensor')
        grid = kwargs.get('grid_thw')
        if isinstance(grid, torch.Tensor):
            grid = grid.detach().cpu().tolist()
        record = {'tensor': payload_description(value, family), 'grid_thw': grid,
                  'arguments': payload_description({'args_after_pixels': args[1:], 'kwargs': kwargs}, family)}
        if family == 'gemma':
            require(len(args) >= 3 and isinstance(args[2], torch.Tensor), 'Gemma actual padding mask missing')
            mask = args[2]
            record['padding_mask'] = {**base.tensor_description(mask, True), 'any_true': bool(mask.any().item())}
        state['static_images'].append(record)
    state['handles'].append(outer.register_forward_pre_hook(observed, with_kwargs=True))
    if family == 'qwen':
        from . import official_x_qsa_selector as selector
        state['selector_calls_before'] = selector.evidence()['actual_calls']
    return result


def read_probe(worker, model):
    result = _ORIGINAL_READ(worker, model)
    result['runtime'] = worker_identity(worker, model)
    if MODELS[model] == 'qwen':
        from . import official_x_qsa_selector as selector
        result['actual_custom_selector_calls_during_request'] = selector.evidence()['actual_calls'] - result['selector_calls_before']
    return result


def check_gpu_order(actual, expected):
    """Permit native reuse of identical images, with first-occurrence order intact."""
    unique = list(dict.fromkeys(expected))
    require(actual == expected or actual == unique,
            'CUDA patch sequence differs from ordered source images (including native duplicate reuse)')


def validate_input(p, item, raw, deep, persist=True):
    model = model_key(p); expected = budget_for(p, item['sample_id'])
    require(expected['item_digest'] == digest(item), 'CPU input reference has different item')
    require(raw['actual_prompt_token_ids'] == expected['prompt_token_ids'], 'Actual prompt tokens differ from CPU native expansion')
    require(raw['request'] == base.messages(p, item['sample_id'], item), 'Actual ordered image/text message changed')
    entries = image_entries(item); reads = raw['native_reader']
    require(Counter((x['path'], x['pixel_sha256']) for x in reads)
            == Counter((str(Path(x['path'])), x['pixel_sha256']) for x in entries),
            'Native reader did not preserve every source image identity/frame/pixel hash')
    require(all(x['shape'] == [512, 768, 3] and x['dtype'] == 'uint8' for x in reads), 'Reader resolution/dtype changed')
    if model == 'nemo':
        check_gpu_order([x['source_RGB8_sha256'] for x in raw['nemo_no_resize']],
                        [x['pixel_sha256'] for x in entries])
        require(raw['nemo_no_resize'] and all(x['target_hw'] == [512, 768] and x['patch_grid_hw'] == [32, 48]
                        and x['patches'] == 1536 and x['image_tokens'] == 384
                        and x['actual_interpolation_performed'] is False for x in raw['nemo_no_resize']),
                'Actual Nemo no-resize native bridge was not observed')
    if model == 'gemma':
        receipt = raw['request_adapter_receipt']
        require(receipt['image_padding'] == expected['padding'] and receipt['native_compaction_calls'] > 0,
                'Gemma padding or native compaction differs from CPU')
        cpu_payloads = [receipt['native_compacted_payload']]
    else:
        cpu_payloads = [x['after_native_dtype'] for x in raw['native_processor']]
    actual_apps = raw['native_application']
    require(actual_apps and all(x['prompt_token_ids'] == expected['prompt_token_ids']
            and x['mm_placeholders'] == expected['mm_placeholders']
            and x['native_payload'] == expected['native_payload'] for x in actual_apps),
            'Actual native prompt placement/expanded tokens differs')
    cpu_payloads = [x['native_payload'] for x in actual_apps]
    expected_hashes = expected['per_image_patch_sha256']
    require(any(ordered_hashes(x) == expected_hashes for x in cpu_payloads),
            'Actual native CPU image patches differ from sealed ordered CPU source reference')
    if model == 'nemo':
        require(len(expected['nemo_same_size_reference']) == len(entries)
                and all(x['official_same_size_interpolation_bytes_equal'] is True
                        for x in expected['nemo_same_size_reference']), 'Nemo original same-size equivalence missing')
    validate_parent_runtime(p, [x['runtime'] for x in deep])
    actual, actual_positions, normalizers = [], [], []
    for observation in deep:
        require(not observation.get('runtime_error') and not observation.get('profiler_error'), 'GPU observer failed')
        runtime = observation['runtime']
        require(runtime['actual_kv_capacity'] >= p.get('minimum_kv_capacity', p['engine_kwargs']['max_model_len']),
                'Actual KV capacity below inherited profile')
        if model == 'qwen':
            evidence = runtime['custom_qsa_selector']
            require(evidence['status'] == 'APPLIED_X_QSA' and evidence['actual_function_is_canonical_wrapper']
                    and evidence['native_function_verified_before_replacement']
                    and evidence['selector_source_sha256'] == p['selector_source']['sha256']
                    and observation['actual_custom_selector_calls_during_request'] > 0, 'Actual bound X QSA missing')
        if model in ('cosmos', 'qwen'):
            from .official_x_runner_v2 import official_file
            norm = observation['normalizer']
            require(norm['class'].endswith(('.FusedMMInputNorm', '.IdentityInputNorm')) and norm.get('source'),
                    'Official native image normalizer missing')
            identity = official_file(norm['source']['path'])
            require(identity['sha256'] == norm['source']['sha256'], 'Official normalizer changed')
            normalizers.append(identity)
        for row in observation['static_images']:
            value = row['tensor']
            require(value['device'].startswith('cuda'), 'No actual CUDA vision ingress')
            required_dtype = 'torch.uint8' if model in ('cosmos', 'qwen') else 'torch.bfloat16'
            require(value['dtype'] == required_dtype, 'Actual native GPU dtype differs')
            if model in ('cosmos', 'qwen'):
                grid = row['grid_thw']
                if grid and isinstance(grid[0], int):
                    grid = [grid]
                require(grid == [[1, 32, 48]] * len(value['per_image_sha256']),
                        'Actual CUDA image grids differ from per-image 768x512/order')
            if model == 'gemma':
                require(row['padding_mask']['dtype'] == 'torch.bool'
                        and row['padding_mask']['shape'] == [len(value['per_image_sha256']), 1584]
                        and row['padding_mask']['any_true'] is False, 'Gemma CUDA mask removes effective patches')
            if model == 'nemo':
                require(row['arguments']['kwargs'].get('imgs_sizes') == [[512, 768]] * len(value['per_image_sha256']),
                        'Actual Nemo CUDA image spatial sizes/order differ')
            actual.extend(value['per_image_sha256'])
            actual_positions.extend(ordered_hashes(row['arguments'], 'per_image_position_sha256'))
    check_gpu_order(actual, expected_hashes)
    if model == 'gemma':
        expected_positions = ordered_hashes(expected['native_payload'], 'per_image_position_sha256')
        require(expected_positions and any(ordered_hashes(x, 'per_image_position_sha256') == expected_positions for x in cpu_payloads),
                'Actual native CPU Gemma positions differ')
        if actual == expected_hashes:
            encoded_indices = list(range(len(expected_hashes)))
        else:
            encoded_indices = [expected_hashes.index(value) for value in dict.fromkeys(expected_hashes)]
        require(actual_positions == [expected_positions[index] for index in encoded_indices],
                'Gemma actual position sequence differs from corresponding encoded image identities')
    proof = {'status': 'PASS_STATIC_NATIVE_INPUT', 'sample_id': item['sample_id'],
        'image_count': len(entries), 'source_frame_indices': [x['frame_index'] for x in entries],
        'source_RGB8_matches_video': True, 'effective_width': 768, 'effective_height': 512,
        'padding': expected['padding'], 'resize_crop_calls': 0, 'full_prompt_token_ids_equal_CPU': True,
        'logical_mm_placeholders': expected['mm_placeholders'],
        'actual_native_payload_and_all_grids_positions_equal_CPU': True,
        'logical_per_image_patch_sha256': expected_hashes, 'actual_GPU_per_image_patch_sha256': actual,
        'actual_GPU_position_sha256': actual_positions, 'official_normalizer_sources': normalizers,
        'internal_normalized_tensor_observed': any(x.get('normalized') for x in deep),
        'observation_boundary': 'Cosmos/Qwen prove raw uint8 CUDA patch ingress and native normalization source; Nemo/Gemma prove their actual BF16 vision entry. Logical order is checked independently of native duplicate image reuse.'}
    if persist:
        base._append(Path(p['output']) / 'input-admissions.jsonl', {'sample_id': item['sample_id'],
            'raw_digest': digest(raw), 'observations_digest': digest(deep), 'proof': proof})
    return proof


def code_equivalence_ast(path):
    """Digest the entire computation except the explicitly reviewed gate delta."""
    import ast
    tree = ast.parse(Path(path).read_text())
    excluded = {'start_gate', 'verify_cpu', 'qualification_start_gate',
                'cpu_entry_compatible', 'code_equivalence_ast'}
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name not in excluded]
    # Python 3.12 added an empty type_params field; it has no executable meaning.
    for node in ast.walk(tree):
        if hasattr(node, 'type_params') and not node.type_params:
            del node.type_params
    return sha(ast.dump(tree, include_attributes=False).encode())


def cpu_entry_compatible(p, budget):
    current_sha = file_sha256(__file__)
    if budget['entry']['sha256'] == current_sha:
        return True
    base_sha = 'aa360d87de3e43850b8f2d460252c9cfa82420caf163e54a3c9551d717c44fb7'
    require(budget['entry']['sha256'] == base_sha and p.get('static_cpu_code_equivalence'),
            'CPU entry changed without the explicit reviewed operational migration')
    review = verified(p['static_cpu_code_equivalence'])
    require(review['schema'] == 'static-x-admission-operational-equivalence-v1'
            and review['status'] == 'PASS_INPUT_COMPUTATION_UNCHANGED' and review['reviewer'] == 'astra1'
            and review['scope'] == 'GATE_ONLY_CPU_REFERENCE_MIGRATION'
            and review['changed_functions'] == ['start_gate', 'verify_cpu']
            and review['added_functions'] == ['code_equivalence_ast', 'cpu_entry_compatible', 'qualification_start_gate']
            and review['base_entry']['sha256'] == base_sha
            and review['current_entry']['sha256'] == current_sha
            and review['ast_algorithm'] == 'python-ast-dump-no-locations-empty-type-params-v1',
            'Operational equivalence review does not bind this exact old/new source delta')
    require(file_sha256(review['base_entry']['path']) == base_sha
            and code_equivalence_ast(review['base_entry']['path'])
                == review['unchanged_module_ast_sha256'] == code_equivalence_ast(__file__),
            'Previously admitted input computation changed')
    refs = review['cpu_budgets']
    require(refs and len({(x['path'], x['sha256']) for x in refs}) == len(refs)
            and p['static_cpu_budget'] in refs, 'This exact immutable CPU budget was not reviewed for migration')
    return True


def verify_cpu(p):
    budget = verified(p['static_cpu_budget'])
    require(budget['schema'] == 'static-x-native-cpu-admission-v1' and budget['status'] == 'PASS'
            and budget['model_calls'] == 0 and not budget['CUDA_initialized'] and not budget['weights_loaded']
            and cpu_entry_compatible(p, budget)
            and budget['source_catalog'] == p['source_catalog'] and budget['static_protocol'] == p['static_protocol']
            and budget['parent_validated_profile'] == p['parent_validated_profile']
            and budget['engine_kwargs_digest'] == digest(p['engine_kwargs'])
            and [x['sample_id'] for x in budget['attempts']] == p['assigned_ids'],
            'Native CPU qualification absent or bound to different input/configuration')
    for attempt in budget['attempts']:
        for ref in attempt['official_source_identity']:
            require(official_source(ref['path'], ref['distribution']) == ref, 'Official processor source changed since CPU admission')
    return budget


def dependencies():
    names = ('official_x_runner', 'official_x_runner_v2', 'refinement_worker', 'storage', 'inference',
             'gemma_speed', 'gemma_x_full', 'nemo_x_full', 'official_x_gemma_adapter', 'official_x_qsa_selector',
             'qwen_x_crosscard', 'official_x_runner_qsa', 'cosmos_t45')
    return {name: descriptor(Path(__file__).parent / (name + '.py')) for name in names}


def seal(draft, output):
    p = read(draft); configuration(p); verify_cpu(p)
    p['static_entry'] = descriptor(__file__)
    p['execution_policy'] = {'module': MODULE, 'source_sha256': file_sha256(__file__)}
    p['static_dependencies'] = dependencies()
    p['automatic_retry'] = False; p['formal_auto_resume'] = False
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); name = stream.name
    try:
        return base.seal(name, output)
    finally:
        Path(name).unlink()


def start_gate(p):
    gate = verified(p['static_start_gate'])
    if gate['schema'] == 'static-x-drained-card-qualification-gate-v2':
        return qualification_start_gate(p, gate)
    require(gate['schema'] == 'static-x-drained-card-gate-v1' and gate['status'] == 'PASS'
            and gate['gpu_uuid'] == p['gpu_uuid'] and gate['predecessor_profiles'], 'No drained-card start gate')
    complete = verified(gate['gemma_complete'])
    returned = verified(gate['gemma_full_return'])
    closure = verified(gate['gemma_remote_closure'])
    require(complete['schema'] == 'natural-x-complete-model-v1' and complete['status'] == 'COMPLETE_PASS'
            and complete['model'] == 'gemma' and complete['unique_videos'] == 10416
            and complete['counts'] == {'T1': 3840, 'T2': 3840, 'T3': 2736}
            and complete['source_version'] == 'natural-t3-lr-paired-20261102-v1'
            and complete['source_identity_sha256'] == '4fa6cf7c8f251bb91f7feeb722d97c21e423d75628828fb446b341de94f4f966',
            'Global current Gemma 10416 has not been scientifically accepted COMPLETE')
    require(returned['schema'] == 'scientific-terminal-return-receipt-v1'
            and returned['status'] == 'PASS_FULL_SHA_RETURN' and returned['all_SHA_verified'] is True
            and returned['current_model_total'] == 10416
            and returned['complete_receipt']['sha256'] == gate['gemma_complete']['sha256']
            and returned['remote_closure']['sha256'] == gate['gemma_remote_closure']['sha256'],
            'Full Gemma scientific return is incomplete or bound to different closure')
    queues = closure['queues']
    require(closure['status'] == 'PASS_REAL_CLOSURE' and len(queues) == 2
            and {x['queue']['sha256'] for x in queues} == {x['sha256'] for x in gate['current_fourier_queues']},
            'Both exact current Fourier queues must truly close')
    for queue in queues:
        require(queue['state'] == 'COMPLETE' and queue['owned_processes_exited']
                and queue['same_inode_lease_free'] and queue['outer_wait']['real_wait_observed']
                and queue['outer_wait']['returncode'] == 0
                and queue['outer_wait']['queue'] == queue['queue'], 'Current Gemma queue closure incomplete')
    require(any(p['physical_lock'] in q['original_same_inode_leases'] for q in queues),
            'Static worker does not use a released current Fourier lease')
    for ref in gate['current_fourier_queues']:
        verified(ref)
    records = []
    for ref in gate['predecessor_profiles']:
        predecessor = verified(ref); out = Path(predecessor['output'])
        wait = read(out.with_suffix('.wait.json')); result = read(out / 'result.json')
        require(predecessor['gpu_uuid'] == p['gpu_uuid']
                and wait['profile_sha256'] == digest(predecessor)
                and wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
                and wait['original_physical_lease_released'] and result['status'] == 'COMPLETE'
                and result['completed_ids'] == predecessor['assigned_ids']
                and not result.get('UNCERTAIN_ids') and not (out / 'failure.json').exists(),
                'Previous card owner has not truly drained and released its original lease')
        records.append({'profile': ref, 'wait': descriptor(out.with_suffix('.wait.json')), 'status': result['status']})
    return {'status': 'PASS', 'predecessors': records, 'gate': p['static_start_gate'],
            'global_Gemma_10416_COMPLETE_and_returned': True, 'both_current_Fourier_queues_closed': True}


def qualification_start_gate(p, gate):
    """Accept one historically drained card only for the fixed qualification.

    The shared execution core still acquires the original physical lease live.
    No raw re-audit, re-flock, global completion or full-run permission is implied.
    """
    from datetime import datetime
    require(p['execution_policy']['module'] == MODULE and type(p['static_cold']) is int
            and p['static_cold'] in (0, 1), 'Own-card gate is forbidden for formal/full execution')
    card = p['static_card']; cold = p['static_cold']
    require(card == ('fourier0', 'fourier1')[cold] and p.get('card') in (None, card)
            and gate['schema'] == 'static-x-drained-card-qualification-gate-v2' and gate['status'] == 'PASS'
            and gate['mode'] == 'OWN_CARD_QUALIFICATION_ONLY'
            and gate['card'] == card and gate['static_cold'] == cold and gate['gpu_uuid'] == p['gpu_uuid'],
            'Qualification gate card/cold/GPU differs')
    for field in ('source_catalog', 'static_cpu_budget', 'static_authorization'):
        require(gate[field] == p[field], 'Own-card gate binds a different ' + field)
    auth = verified(p['static_authorization'])
    scope = verified(gate['user_scope_authorization'])
    original_scope = verified(auth['scope_authorization'])
    root = Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve()
    scope_relative = Path('plans/static-x-t123-20261104/authorization.json')
    require(p['workspace_root'] == str(root)
            and auth['scope_authorization']['path'] == str(root / scope_relative)
            and scope['extends_scope_authorization']['path'] in
                (str(root / scope_relative), str(Path(os.environ.get('PSYCHOPHYSICS_MIRROR', root)).resolve() / scope_relative))
            and gate['current_gemma_run_set']['path'] == str(root / 'plans/natural-x-t123-20261101'
                / 'gemma-natural-three-card-20261103/radon-to-fourier-20261103/run-set-fourier.json'),
            'Supplemental scope belongs to another workspace or original run set')
    require(scope['schema'] == 'static-x-supplemental-execution-scope-v1'
            and scope['owner_chat'] == original_scope['owner_chat'] == os.environ.get('PSYCHOPHYSICS_PARENT_QUEUE_OWNER', 'unbound-parent-queue')
            and scope['per_card_qualification_after_real_release'] is True
            and scope['per_card_requires_own_gemma_full_return_and_real_closure'] is True
            and scope['formal_full_requires_global_gemma10416_complete'] is True
            and scope['scientific_inputs_and_call_caps_unchanged'] is True
            and scope['maximum_qualification_calls'] == original_scope['qualification_calls'] == 120
            and scope['maximum_new_model_calls'] == original_scope['maximum_new_model_calls'] == 58020
            and scope['extends_scope_authorization']['sha256'] == auth['scope_authorization']['sha256']
            and scope['current_gemma_run_set'] == gate['current_gemma_run_set']
            and auth['schema'] == 'static-x-120-call-authorization-v1'
            and auth['maximum_model_calls'] == 120 and auth['automatic_retry'] is False
            and auth['cold_gpu_uuids'][str(cold)] == p['gpu_uuid'],
            'Latest scheduling supplement does not extend the bound finite authorization')
    run_set = verified(gate['current_gemma_run_set'])
    current_queue = verified(gate['current_gemma_queue'])
    require(run_set['schema'] == 'natural-x-run-set-v1'
            and gate['current_gemma_queue'] in run_set['queues']
            and current_queue['schema'] == 'natural-x-finite-queue-v1' and current_queue['model'] == 'gemma'
            and current_queue['execution_authorization'] == run_set['execution_authorization'],
            'Own queue does not belong to the exact original current Gemma run set')
    closure = verified(gate['gemma_card_closure'])
    returned = verified(gate['gemma_card_full_return'])
    counts = {'T1': 328 if card == 'fourier0' else 326, 'T2': 640, 'T3': 456}
    require(closure['schema'] == 'gemma-current-takeover-card-real-closure-v1'
            and closure['status'] == 'PASS_REAL_CARD_CLOSURE' and closure['model'] == 'gemma'
            and closure['card'] == returned['card'] == card
            and closure['gpu_uuid'] == returned['gpu_uuid'] == p['gpu_uuid']
            and closure['task_counts'] == returned['task_counts'] == counts
            and closure['run_set'] == gate['current_gemma_run_set'] and len(closure['queues']) == 1
            and returned['schema'] == 'gemma-takeover-card-outer-Mac-return-v1'
            and returned['status'] == 'PASS_CURRENT_TAKEOVER_CARD_OUTER_METADATA_SHA_RETURN'
            and returned['all_SHA_verified'] is True
            and returned['closure']['sha256'] == gate['gemma_card_closure']['sha256']
            and returned['queue'] == gate['current_gemma_queue']
            and returned['task_returns'] == closure['task_returns']
            and returned['prior_task_full_returns'] == closure['prior_task_full_returns'],
            'Own card real closure and returned metadata do not match')
    queue = closure['queues'][0]
    require(queue['card'] == card and queue['gpu_uuid'] == p['gpu_uuid'] and queue['task_counts'] == counts
            and queue['queue'] == gate['current_gemma_queue'] and queue['state'] == 'COMPLETE'
            and queue['state_record']['status'] == 'COMPLETE' and queue['state_record']['plan'] == queue['queue']
            and queue['owned_processes_exited'] is True and queue['same_inode_lease_free'] is True
            and queue['outer_wait']['real_wait_observed'] is True and queue['outer_wait']['returncode'] == 0
            and queue['outer_wait']['queue'] == queue['queue']
            and queue['profiles'] == current_queue['profiles'] and len(queue['profiles']) == 3
            and queue['original_same_inode_leases'] == [p['physical_lock']]
            and queue['lease_free_scope'] == 'HISTORICAL_AT_RECORDED_CLOSURE'
            and queue['current_lease_free_claimed'] is False,
            'Own queue is not COMPLETE with real wait, owned exit and historical original-inode release')
    window = queue['closure_check_window_UTC']
    started, finished = (datetime.fromisoformat(window[k]) for k in ('started', 'finished'))
    require(started.tzinfo is not None and finished.tzinfo is not None and started <= finished
            and queue['closure_observed_UTC'] == queue['lease_free_observed_UTC'] == window['finished'],
            'Historical lease release has no complete observation window')
    for key, value in (('state_source', 'state_record'), ('outer_wait_source', 'outer_wait')):
        require(queue[key] in closure['bindings'] and verified(queue[key]) == queue[value],
                'Recorded queue state/wait differs from its original SHA-bound bytes')
    rows = closure['task_returns']; mapped = gate['task_returns']
    require([x['task'] for x in rows] == [x['task'] for x in mapped] == ['T1', 'T2', 'T3']
            and [x['profile'] for x in rows] == queue['profiles']
            and {x['task']: x['accepted'] for x in rows} == counts
            and len(closure['prior_task_full_returns']) == 3
            and {x['return_receipt']['sha256'] for x in rows}
                == {x['sha256'] for x in closure['prior_task_full_returns']},
            'Own card task-return coverage is incomplete')
    records = []
    for row, local_row in zip(rows, mapped):
        # Receipt/audit mirror paths may differ; their immutable bytes may not.
        require(set(row) == set(local_row)
                and all(local_row[k]['sha256'] == row[k]['sha256'] if k in ('return_receipt', 'science_audit')
                        else local_row[k] == row[k] for k in row), 'Task mirror changed the original proof')
        task = row['task']; task_return = verified(local_row['return_receipt'])
        audit = verified(local_row['science_audit'])
        require(task_return['status'] == 'PASS_FULL_CLOSED_TASK_SHA_RETURN' and task_return['takeover'] is True
                and task_return['task'] == task and card in task_return['cards']
                and task_return['science_audit']['sha256'] == row['science_audit']['sha256']
                and audit['status'] == 'PASS_CLOSED_' + task + '_RAW_SOURCE_CONFIG_AND_NATIVE180_AUDIT',
                'Own task raw/source full SHA return has not passed')
        proof = audit['cards'][card]; science = proof['scientific_audit']
        require(proof['profile'] == science['profile'] == row['profile']
                and proof['terminal_status'] == 'COMPLETE' and proof['completed'] == science['accepted'] == counts[task]
                and science['worker'] == row['worker'] and science['raw'] == row['raw'] and science['wait'] == row['wait']
                and science['model'] == 'gemma' and science['original_returncode'] == 0
                and science['admission'] == 'WORKER_TECHNICAL_PASS',
                'Per-card scientific audit does not match the original worker/count/raw/wait')
        predecessor = verified(row['profile']); wait = verified(row['wait'])
        require(predecessor['worker'] == row['worker'] and predecessor['card'] == card
                and predecessor['gpu_uuid'] == p['gpu_uuid'] and predecessor['physical_lock'] == p['physical_lock']
                and len(predecessor['assigned_ids']) == counts[task]
                and wait['profile_sha256'] == digest(predecessor) and wait['returncode'] == 0
                and wait['real_wait_observed'] is True and wait['owned_engines_exited'] is True
                and wait['original_physical_lease_released'] is True and wait['terminal_status'] == 'COMPLETE'
                and wait['completed_ids'] == predecessor['assigned_ids'],
                'Original per-task worker has not truly completed, waited and released its own lease')
        records.append({'task': task, 'profile': row['profile'], 'wait': row['wait'],
                        'return_receipt': local_row['return_receipt'], 'accepted': counts[task]})
    old = gate['existing_full_returns']
    require(set(old) == {'original_F', 'radon626'}, 'Both original full returns are required')
    original_f = verified(old['original_F']); radon = verified(old['radon626'])
    require(original_f['status'] == 'PASS_BOTH_ORIGINAL_QUEUES_FULLY_CLOSED_AUDITED_RETURNED_READY'
            and original_f['original_answers'] == 6944 and set(original_f['task_return_receipts']) == {'T1', 'T2', 'T3'}
            and radon['schema'] == 'gemma-radon626-original-readonly-Mac-return-v1'
            and radon['status'] == 'PASS_ORIGINAL626_FULL_RAW_CPU_AUDITED_SHA_RETURNED'
            and radon['answers'] == radon['claims'] == 626,
            'Original F6944/R626 full audited returns are missing; dispatch exclusion is insufficient')
    return {'status': 'PASS', 'mode': 'OWN_CARD_QUALIFICATION_ONLY', 'gate': p['static_start_gate'],
            'card': card, 'gpu_uuid': p['gpu_uuid'], 'predecessors': records,
            'original_F6944_and_R626_returned': True, 'own_card_takeover_returned': sum(counts.values()),
            'historical_original_inode_release_verified': True, 'current_lease_free_claimed': False,
            'global_Gemma_10416_COMPLETE_and_returned': False, 'full_static_authorized': False}


def image_independent_layout(value):
    """Keep every native layout/dtype/grid/position field; remove only pixels."""
    if isinstance(value, dict):
        if 'per_image_sha256' in value:
            return {key: item for key, item in value.items() if key not in ('sha256', 'per_image_sha256')}
        return {key: image_independent_layout(item) for key, item in value.items()}
    if isinstance(value, list):
        return [image_independent_layout(item) for item in value]
    return value


def ordinary_input(p, item, raw, expected):
    """Full-run ordinary checks; caller binds a shape/task/domain CPU reference.

    This checks exact reader identity and prompt/placement plus native processor
    geometry. It deliberately makes no per-request GPU-byte qualification claim.
    """
    model = model_key(p); count = check_item(item, cached_verified(p['static_protocol']))
    require(raw['request'] == base.messages(p, item['sample_id'], item)
            and raw['actual_prompt_token_ids'] == expected['prompt_token_ids'],
            'Ordinary exact request/prompt reference differs')
    entries = image_entries(item)
    require(Counter((x['path'], x['pixel_sha256']) for x in raw['native_reader'])
            == Counter((str(Path(x['path'])), x['pixel_sha256']) for x in entries),
            'Ordinary native reader source/count/order identities differ')
    require(all(x['shape'] == [512, 768, 3] and x['dtype'] == 'uint8' for x in raw['native_reader']),
            'Ordinary reader dimensions/dtype changed')
    apps = raw['native_application']
    require(apps and all(x['prompt_token_ids'] == expected['prompt_token_ids']
            and x['mm_placeholders'] == expected['mm_placeholders'] for x in apps),
            'Ordinary native logical placements changed')
    for app in apps:
        values = list(leaves(app['native_payload']))
        require(image_independent_layout(app['native_payload']) == image_independent_layout(expected['native_payload']),
                'Ordinary native tensor layout/dtype/grid/position differs from qualified CPU template')
        require(len(ordered_hashes(app['native_payload'])) == count, 'Ordinary processor image count differs')
        if model in ('cosmos', 'qwen'):
            require([x['values'] for x in values if x.get('shape') == [3] and 'values' in x]
                    == [[1, 32, 48]] * count, 'Ordinary CPU image grids changed')
        if model == 'gemma':
            require(ordered_hashes(app['native_payload'], 'per_image_position_sha256')
                    == ordered_hashes(expected['native_payload'], 'per_image_position_sha256'),
                    'Ordinary Gemma native positions changed')
    if model == 'nemo':
        receipts = raw['nemo_no_resize']
        check_gpu_order([x['source_RGB8_sha256'] for x in receipts], [x['pixel_sha256'] for x in entries])
        require(receipts and all(x['target_hw'] == [512, 768] and x['patches'] == 1536
                        and x['image_tokens'] == 384 and x['actual_interpolation_performed'] is False for x in receipts),
                'Ordinary Nemo identity-size interpolation skip/source order differs')
    if model == 'gemma':
        receipt = raw['request_adapter_receipt']
        require(receipt['native_compaction_calls'] > 0 and len(receipt['image_padding']) == count
                and all(x['source_wh'] == [768, 512] and x['submitted_wh'] == [768, 528]
                        and x['bottom_rows'] == 16 and x['right_columns'] == 0
                        and not x['effective_pixels_resized_or_cropped'] for x in receipt['image_padding']),
                'Ordinary Gemma padding/compaction differs')
    return {'status': 'PASS_ORDINARY_STATIC_INPUT', 'image_count': count,
            'full_prompt_token_ids_equal_CPU_reference': True, 'native_source_and_shape_checked': True,
            'per_request_GPU_bytes_checked': False, 'resize_crop_calls': 0}


def qualification_policy(p):
    catalog = configuration(p)
    verify_cpu(p)
    return catalog


def every_case_deep(p, item, ordinal):
    return True


def run(profile_path):
    return execute_bound_profile(profile_path, policy_validator=qualification_policy,
                                 input_validator=validate_input, deep_policy=every_case_deep)


def execute_bound_profile(profile_path, *, policy_validator, input_validator, deep_policy, runtime_validator=None):

    """Run a sealed policy using the existing X lease and owned-engine lifecycle."""
    import importlib.metadata
    import os
    import signal
    import socket
    import sys
    import time
    from .storage import atomic_json, owner_identity
    from . import gemma_speed as lifecycle
    from .inference import checkpoint_asset_identity
    p = read(profile_path)
    callbacks = (policy_validator, input_validator, deep_policy) + ((runtime_validator,) if runtime_validator is not None else ())
    for callback in callbacks:
        expected_module = p['execution_policy']['module']
        actual_module = callback.__module__
        if actual_module == '__main__':
            spec = callback.__globals__.get('__spec__')
            actual_module = spec.name if spec is not None else actual_module
        require(actual_module in (MODULE, expected_module), 'Unbound execution policy callback')
        expected_sha = p['static_entry']['sha256'] if actual_module == MODULE else p['execution_policy']['source_sha256']
        require(file_sha256(inspect.getsourcefile(callback)) == expected_sha, 'Execution policy source changed')
    catalog = policy_validator(p)
    require(p['static_entry']['sha256'] == file_sha256(__file__) and p['static_dependencies'] == dependencies(),
            'Frozen static admission/dependencies changed')
    require(file_sha256(base.__file__) == p['implementation']['sha256'], 'Original lifecycle implementation changed')
    require(sys.executable == p['python'] and socket.gethostname() == p['hostname'], 'Wrong python/host')
    for key, value in p['environment'].items():
        require(os.environ.get(key) == value, 'Environment differs: ' + key)
    for key in p.get('cleared_environment', []):
        require(key not in os.environ, 'Unexpected environment: ' + key)
    for package, version in p.get('expected_packages', {}).items():
        require(importlib.metadata.version(package) == version, 'Package changed: ' + package)
    out = Path(p['output']); require(not out.exists(), 'No implicit worker resume')
    require(not any((Path(p['claim_root']) / (sid + '.json')).exists() for sid in p['assigned_ids']),
            'Existing claim forbids retry')
    require(checkpoint_asset_identity(p['checkpoint']) == p['checkpoint_assets'], 'Checkpoint assets changed')
    gate = start_gate(p)
    from .gemma_x_full import storage_admission
    storage = storage_admission(p)
    out.mkdir(parents=True)
    atomic_json(out / 'identity.json', {'profile': p, 'profile_sha256': digest(p), 'owner': owner_identity(), 'python': sys.executable})
    atomic_json(out / 'start-gate.json', gate); atomic_json(out / 'storage-admission.json', storage)
    atomic_json(out / 'versions.json', {name: importlib.metadata.version(name)
        for name in ('vllm', 'torch', 'transformers', 'numpy')})
    boundary = base.Boundary(p)
    for sig in (signal.SIGUSR1, signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, boundary.signal)
    if boundary.check():
        result = {'status': 'PAUSED_BEFORE_LOAD', 'completed_ids': [], 'UNCERTAIN_ids': [], 'model_load_started': False}
        atomic_json(out / 'result.json', result); return result
    import torch
    if p.get('torch_num_threads'):
        torch.set_num_threads(p['torch_num_threads'])
    completed, pending = [], None
    with base.physical_lease(p, torch) as lease:
        atomic_json(out / 'lease.json', lease)
        with ExitStack() as stack:
            from vllm import LLM
            kwargs = base.engine_kwargs(p)
            atomic_json(out / 'public-settings.json', {'LLM_kwargs': {k: sorted(v) if isinstance(v, set) else v for k, v in kwargs.items()},
                'request_mm_processor_kwargs': native_kwargs(p), 'chat_template_kwargs': {'enable_thinking': False},
                'parent_video_profile': p['parent_validated_profile'],
                'image_specific_changes': {'limit_mm_per_prompt': kwargs['limit_mm_per_prompt'],
                    'mm_processor_kwargs': kwargs['mm_processor_kwargs'], 'Gemma_bottom_padding_rows': 16 if model_key(p) == 'gemma' else 0,
                    'Nemo_same_size_interpolation_skipped': model_key(p) == 'nemo'},
                'cold': p.get('static_cold'), 'deep_policy': deep_policy.__name__,
                'execution_policy': p['execution_policy']})
            started = time.monotonic(); engine = LLM(**kwargs)
            owned = engine.collective_rpc(lifecycle._worker_identity)
            atomic_json(out / 'engine-owner.json', {'owned': owned})
            stack.callback(lifecycle.shutdown_owned_engine, engine, owned, out)
            runtime = engine.collective_rpc(worker_identity, args=(p['model'],))
            atomic_json(out / 'runtime.json', runtime)
            validate_parent_runtime(p, runtime)
            require(all(x['actual_kv_capacity'] >= p.get('minimum_kv_capacity', p['engine_kwargs']['max_model_len'])
                        and x['gpu_uuid'].lower().removeprefix('gpu-') == p['gpu_uuid'].lower().removeprefix('gpu-')
                        for x in runtime), 'Actual GPU identity/KV capacity differs before first claim')
            if runtime_validator is not None:
                runtime_validator(p, runtime)
            atomic_json(out / 'load.json', {'seconds': time.monotonic() - started,
                                          'native_eos': engine.get_tokenizer().eos_token_id})
            nemo = stack.enter_context(NemoNoResize()) if model_key(p) == 'nemo' else None
            adapter = GemmaImageAdapter(p, engine) if model_key(p) == 'gemma' else None
            if adapter:
                stack.callback(adapter.close)
            for media in p['media_files'].values():
                require(base.stat(media['path']) == media['stat'] and file_sha256(media['path']) == media['sha256'],
                        'Sealed PNG source changed')
            observer = stack.enter_context(ImageObserver(p))
            sampling, generation = base.generation(['LEFT', 'RIGHT'])
            try:
                for ordinal, sid in enumerate(p['assigned_ids']):
                    if boundary.check():
                        break
                    item = catalog['items'][sid]; pending = sid
                    deep = bool(deep_policy(p, item, ordinal))
                    for media in image_entries(item):
                        require(base.stat(media['path']) == p['media_files'][media['path']]['stat'], 'Source changed before request')
                    write_new(Path(p['claim_root']) / (sid + '.json'), {'sample_id': sid, 'profile_sha256': digest(p),
                        'worker_token': p['worker_token'], 'attempt': 1, 'cold': p.get('static_cold'),
                        'authorization': p['static_authorization']})
                    base._append(out / 'events.jsonl', {'event': 'INTENT', 'sample_id': sid, 'ordinal': ordinal, 'deep': deep})
                    observer.clear(); observer.active = True; observer.deep = deep; observer.contract = CONTRACT
                    if deep:
                        installed = engine.collective_rpc(install_probe, args=(p['vision_observation'], p['model']))
                        base._append(out / 'deep-install.jsonl', {'sample_id': sid, 'observations': installed})
                    request = base.messages(p, sid, item); prepared = None; started = time.monotonic()
                    if nemo:
                        nemo.receipts.clear()
                    if adapter is None:
                        answer = engine.chat(request, sampling_params=sampling, use_tqdm=False,
                            chat_template_kwargs={'enable_thinking': False}, mm_processor_kwargs=native_kwargs(p))[0]
                    else:
                        prepared = adapter.prepare(item, request[0]['content'])
                        answer = engine.generate(prepared['request'], sampling, use_tqdm=False)[0]
                    output = answer.outputs[0]
                    raw = {'sample_id': sid, 'task': item['task'], 'domain': item['domain'], 'model': p['model'],
                        'variant': 'CURRENT_X_STATIC', 'cold': p.get('static_cold'), 'ordinal': ordinal, 'deep': deep, 'attempt': 1,
                        'truth': item['row']['truth'], 'raw_answer': output.text, 'correct': output.text.strip() == item['row']['truth'],
                        'output_token_ids': list(output.token_ids), 'finish_reason': output.finish_reason, 'stop_reason': output.stop_reason,
                        'actual_prompt_token_ids': list(answer.prompt_token_ids), 'question': item['row']['prompt'], 'request': request,
                        'generation': generation, 'request_mm_processor_kwargs': native_kwargs(p), 'profile_sha256': digest(p),
                        'source': {'catalog': p['source_catalog'], 'item_digest': digest(item),
                            'media': [p['media_files'][x['path']] for x in image_entries(item)]},
                        'native_reader': observer.reads, 'native_processor': observer.processors,
                        'native_application': observer.applications,
                        'nemo_no_resize': list(nemo.receipts) if nemo else None,
                        'request_adapter_receipt': (prepared or {}).get('receipt'),
                        'timing': {'engine_request_inclusive_seconds': time.monotonic() - started}}
                    base._append(out / 'raw.jsonl', raw)
                    evidence = engine.collective_rpc(read_probe, args=(p['model'],)) if deep else []
                    if deep:
                        base._append(out / 'deep-observations.jsonl', {'sample_id': sid, 'raw_digest': digest(raw), 'observations': evidence})
                    require(output.finish_reason == 'stop' and output.text.strip() in ('LEFT', 'RIGHT'),
                            'Incomplete/invalid completion retained; no retry')
                    input_validator(p, item, raw, evidence)
                    base._append(out / 'events.jsonl', {'event': 'TECHNICAL_PASS', 'sample_id': sid,
                        'ordinal': ordinal, 'raw_digest': digest(raw), 'deep': deep})
                    completed.append(sid); pending = None
                    atomic_json(out / 'progress.json', {'status': 'RUNNING', 'completed': len(completed),
                        'expected': len(p['assigned_ids']), 'last': sid})
                    if adapter:
                        adapter.clear()
                    observer.active = False
            except BaseException as error:
                atomic_json(out / 'failure.json', {'error_type': type(error).__name__, 'error': str(error),
                    'UNCERTAIN_id': pending, 'completed': len(completed), 'automatic_retry': False,
                    'native_reader': observer.reads, 'native_processor': observer.processors,
                    'native_application': observer.applications})
                raise
            finally:
                observer.active = False
                if adapter:
                    adapter.clear()
    result = {'status': 'COMPLETE' if completed == p['assigned_ids'] else 'PAUSED', 'completed_ids': completed,
              'UNCERTAIN_ids': [pending] if pending else [], 'automatic_retry': False, 'formal_auto_resume': False}
    atomic_json(out / 'result.json', result)
    return result


def supervise(profile):
    from . import refinement_worker as worker
    original = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return original([x.replace('classic_precision.refinement_worker', MODULE) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = original


def deployment_signature(p):
    # Only predeclared physical-card fields are excluded. Each profile has
    # already been checked against its own source-bound video parent.
    keys = ('model', 'checkpoint', 'checkpoint_assets', 'processor_checkpoint', 'engine_kwargs',
            'request_mm_processor_kwargs', 'request_adapter', 'vision_observation', 'expected_packages',
            'environment', 'cleared_environment', 'python', 'resources', 'torch_num_threads',
            'static_entry', 'static_dependencies', 'selector_source')
    result = {key: copy.deepcopy(p.get(key)) for key in keys}
    result['environment'] = {key: value for key, value in p['environment'].items() if key != 'CUDA_VISIBLE_DEVICES'}
    return result


def runtime_signature(value):
    keys = ('vllm_version', 'torch_version', 'cuda_version', 'model_class', 'runner_class',
            'kernel_config', 'method_kernel_counts', 'actual_backends', 'actual_kv_capacity',
            'kv_concurrency', 'mm_device_do_normalize', 'gpu_architecture', 'gpu_total_memory_bytes')
    return {key: value.get(key) for key in keys}


def strict_records(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Journal must be a regular non-symlink file')
    data = path.read_bytes()
    require(data and data.endswith(b'\n'), 'Journal is empty or lacks a durable final LF')
    rows = []
    for line in data.split(b'\n')[:-1]:
        require(line.strip() and not line.endswith(b'\r'), 'Journal has a blank or non-LF line')
        value = json.loads(line)
        require(isinstance(value, dict), 'Every journal line must be a complete JSON object')
        rows.append(value)
    return rows


def claim_ids(claim_root, allowed):
    root = Path(claim_root)
    require(root.is_dir() and not root.is_symlink(), 'Claim root must be a real directory')
    result = set()
    for path in root.iterdir():
        require(path.is_file() and not path.is_symlink() and path.suffix == '.json'
                and path.stem in allowed, 'Unknown/nonregular claim entry: ' + str(path))
        result.add(path.stem)
    return result


def closed(profile_path):
    p = read(profile_path); catalog = configuration(p); verify_cpu(p); out = Path(p['output'])
    wait = read(out.with_suffix('.wait.json')); result = read(out / 'result.json')
    require(wait['profile_sha256'] == digest(p) and wait['returncode'] == 0 and wait['real_wait_observed']
            and wait['owned_engines_exited'] and wait['original_physical_lease_released']
            and result['status'] == 'COMPLETE' and result['completed_ids'] == p['assigned_ids']
            and not result.get('UNCERTAIN_ids') and not (out / 'failure.json').exists(), 'Worker is not safely COMPLETE')
    exit_record = read(out / 'exit.json')
    require(exit_record['owned_engine_workers_exited'] and exit_record['shutdown_error'] is None,
            'Owned-engine shutdown proof incomplete')
    raws = strict_records(out / 'raw.jsonl'); proofs = strict_records(out / 'input-admissions.jsonl')
    events = strict_records(out / 'events.jsonl'); deep = strict_records(out / 'deep-observations.jsonl')
    installs = strict_records(out / 'deep-install.jsonl')
    require([(x['event'], x['sample_id']) for x in events]
            == [(event, sid) for sid in p['assigned_ids'] for event in ('INTENT', 'TECHNICAL_PASS')],
            'Incomplete, duplicate or reordered INTENT/PASS journal')
    passed = events[1::2]
    require([x['sample_id'] for x in raws] == [x['sample_id'] for x in proofs]
            == [x['sample_id'] for x in deep] == [x['sample_id'] for x in installs] == p['assigned_ids'],
            'Raw, input/deep/install proof and assignment differ')
    auth, _ = authorization(p)
    allowed_claim_ids = {sid for allocation in auth['allocations'] if allocation['claim_root'] == p['claim_root']
                         for sid in allocation['assigned_ids']}
    actual_claim_ids = claim_ids(p['claim_root'], allowed_claim_ids)
    require(set(p['assigned_ids']) <= actual_claim_ids <= allowed_claim_ids, 'Missing or unauthorized claim')
    runtime = [runtime_signature(x) for x in read(out / 'runtime.json')]
    for ordinal, (raw, proof, event, observation) in enumerate(zip(raws, proofs, passed, deep)):
        sid = raw['sample_id']; claim = read(Path(p['claim_root']) / (sid + '.json'))
        require(claim['profile_sha256'] == raw['profile_sha256'] == digest(p) and claim['attempt'] == 1
                and claim['worker_token'] == p['worker_token'] and claim['cold'] == p['static_cold']
                and claim['authorization'] == p['static_authorization']
                and raw['cold'] == p['static_cold'] and raw['ordinal'] == event['ordinal'] == ordinal
                and raw['deep'] and event['deep']
                and proof['raw_digest'] == event['raw_digest'] == observation['raw_digest'] == digest(raw)
                and proof['observations_digest'] == digest(observation['observations'])
                and proof['proof']['status'] == 'PASS_STATIC_NATIVE_INPUT', 'Durable proof/claim binding differs')
        require([runtime_signature(x['runtime']) for x in observation['observations']] == runtime,
                'Actual per-request deployment drifted within cold load')
        require(proof['proof'] == validate_input(p, catalog['items'][sid], raw, observation['observations'], persist=False),
                'Stored input proof does not reproduce from actual deep/raw evidence')
    return p, raws


def summarize(plan_path, output):
    plan = read(plan_path); auth = verified(plan['static_authorization'])
    rows, profiles, deployment, runtime = {}, [], {}, {}
    for ref in plan['profiles']:
        verified(ref); p, raw = closed(ref['path'])
        authorization(p)
        require(p['static_authorization'] == plan['static_authorization'], 'Mixed authorization')
        profiles.append(ref)
        full_runtime = read(Path(p['output']) / 'runtime.json')
        require(all(x['gpu_uuid'].lower().removeprefix('gpu-')
                    == auth['cold_gpu_uuids'][str(p['static_cold'])].lower().removeprefix('gpu-') for x in full_runtime),
                'Actual runtime GPU differs from authorized cold/card mapping')
        actual_runtime = [runtime_signature(x) for x in full_runtime]
        for row in raw:
            key = (model_key(p), p['static_cold'], row['sample_id'])
            require(key not in rows, 'Duplicate cold answer')
            rows[key] = row
            deployment[key] = deployment_signature(p)
            runtime[key] = actual_runtime
    require(len(rows) == 120, 'Exactly 120 completed single-attempt calls required')
    fields = ('actual_prompt_token_ids', 'output_token_ids', 'raw_answer', 'finish_reason', 'stop_reason')
    comparisons = []
    for model in ('cosmos', 'qwen', 'nemo', 'gemma'):
        for sid in auth['case_ids']:
            left, right = rows[model, 0, sid], rows[model, 1, sid]
            changed = [key for key in fields if left[key] != right[key]]
            deployment_changes = [key for key in deployment[model, 0, sid]
                if deployment[model, 0, sid][key] != deployment[model, 1, sid][key]]
            if deployment_changes:
                changed.append('deployment_signature')
            if runtime[model, 0, sid] != runtime[model, 1, sid]:
                changed.append('actual_runtime_signature')
            require(left['source']['item_digest'] == right['source']['item_digest'] and left['generation'] == right['generation'],
                    'Cold repeats have different source or generation settings')
            comparisons.append({'model': model, 'sample_id': sid, 'task': left['task'], 'equal': not changed,
                'different_fields': changed, 'deployment_different_fields': deployment_changes, 'cold0_raw_digest': digest(left), 'cold1_raw_digest': digest(right),
                'answer': left['raw_answer'], 'correct': left['correct']})
    status = 'PASS_STATIC_15CASE_TWO_COLD' if all(x['equal'] for x in comparisons) else 'HOLD_COLD_REPRODUCTION_DIFFERENCE'
    result = {'schema': 'static-x-15case-two-cold-summary-v1', 'status': status, 'executed_model_calls': 120,
        'static_authorization': plan['static_authorization'], 'profiles': profiles, 'comparisons': comparisons,
        'exact_repeat_pairs': sum(x['equal'] for x in comparisons), 'total_pairs': 60,
        'correctness_required_for_technical_PASS': False, 'automatic_retry': False,
        'third_cold_authorized': False, 'full_static_authorized': False,
        'scope': 'Fixed fifteen cases, five per task, four current-X deployments, two independent cold loads on the two fixed Fourier GPUs per inherited configuration. This is input and repeatability qualification, not full-static results or evidence for additional cold loads or other GPUs.'}
    write_new(output, result)
    return {'status': status, 'summary': descriptor(output), 'exact_repeat_pairs': result['exact_repeat_pairs'], 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('cpu', 'seal', 'run', 'supervise', 'summarize'))
    for name in ('draft', 'profile', 'output', 'plan'):
        parser.add_argument('--' + name)
    args = parser.parse_args()
    if args.command == 'cpu':
        result = cpu(args.draft, args.output)
    elif args.command == 'seal':
        result = seal(args.draft, args.output)
    elif args.command == 'run':
        result = run(args.profile)
    elif args.command == 'supervise':
        result = supervise(args.profile)
    else:
        result = summarize(args.plan, args.output)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
