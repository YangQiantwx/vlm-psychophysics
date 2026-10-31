"""Classic Gemma X task references and independent first-sample input admission.

The qualified native processor and masked-tail compaction remain the inference
route. Official HF tensors are CPU-only references; they are never submitted to
the engine. GPU byte admission covers the first sample of each task per engine.
"""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path

from .gemma_x_native_compact import NativeCompactAdapter
from .gemma_x_speed import validate_compact_vision
from .official_x_gemma_adapter import json_bytes, pad, require
from .refinement_worker import descriptor, read, verified, write_new
from .storage import _append, digest, file_sha256


TASKS = ('T1', 'T2', 'T3')
CONTRACT = {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}
PATCHES = 1584
SOFT_TOKENS = 176
REQUIRED_DEPENDENCIES = {
    'classic_precision.gemma_x_native_input',
    'classic_precision.gemma_x_native_compact',
    'classic_precision.official_x_gemma_adapter',
    'classic_precision.gemma_x_speed',
}
_CURRENT_ADAPTER = None


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _tensor_sha(value):
    import torch
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast('B')).hexdigest()


def _check_dependencies(profile):
    refs = profile.get('input_dependencies')
    require(isinstance(refs, list) and refs, 'Explicit Gemma input dependencies missing')
    modules = set()
    for ref in refs:
        name = ref['module']
        require(name not in modules, 'Duplicate Gemma input dependency: ' + name)
        module = importlib.import_module(name)
        require(file_sha256(module.__file__) == ref['source_sha256'],
                'Gemma input dependency changed: ' + name)
        modules.add(name)
    require(REQUIRED_DEPENDENCIES <= modules, 'Inherited Gemma input dependencies are unbound')


def _check_item(item, table=None):
    task = item['task']
    require(task in TASKS, 'Only classic T1/T2/T3 are admitted')
    require(item.get('video_contract') == CONTRACT, 'Classic video contract differs')
    content = item.get('content', [])
    require([entry['type'] for entry in content] == ['video', 'text'],
            'Classic content must be one video followed by its question')
    question = item['row']['prompt']
    require(content[1]['text'] == question and isinstance(question, str) and question,
            'Classic scientific question differs from content')
    if table is not None:
        expected = table[task]
        require(expected['question'] == question
                and expected['question_sha256'] == _sha(question.encode())
                and expected['video_contract'] == CONTRACT
                and expected['content_types'] == ['video', 'text'],
                'Sample differs from sealed task prompt contract')
    return task, question


def _check_reader(source, metadata):
    import numpy as np
    require(source.dtype == np.uint8 and list(source.shape) == [180, 512, 768, 3],
            'Official OpenCV uint8 source geometry differs')
    require(metadata.get('frames_indices') == list(range(180))
            and metadata.get('total_num_frames') == 180
            and metadata.get('fps') == 60.
            and metadata.get('duration') == 3.,
            'Official video frame order or time base differs')


def _template(processor, question):
    text = processor.apply_chat_template([{'role': 'user', 'content': [
        {'type': 'video'}, {'type': 'text', 'text': question}]}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require(not text.rstrip().endswith('<think>'), 'Gemma thinking unexpectedly enabled')
    return text


def _official_reference(processor, text, frames):
    """Run unmodified official HF processing, retaining only valid CPU patches."""
    import numpy as np
    import torch
    from transformers.video_utils import VideoMetadata
    cuda_was_initialized = torch.cuda.is_initialized()
    require(frames.dtype == np.uint8 and list(frames.shape) == [180, 528, 768, 3],
            'HF reference must use the exact padded uint8 video')
    metadata = VideoMetadata(total_num_frames=180, fps=60., duration=3.,
                             frames_indices=list(range(180)))
    data = processor(text=[text], videos=[frames], videos_kwargs={
        'do_resize': False, 'do_sample_frames': False, 'max_soft_tokens': 280,
        'video_metadata': [metadata]}, return_tensors='pt')
    positions_key = next(k for k in data if 'position_ids' in k and 'video' in k)
    pixels, positions = data['pixel_values_videos'][0], data[positions_key][0]
    require(pixels.device.type == positions.device.type == 'cpu'
            and pixels.dtype == torch.float32 and positions.dtype == torch.int64
            and list(pixels.shape) == [180, 2520, 768]
            and list(positions.shape) == [180, 2520, 2],
            'Official HF CPU patch layout or dtype differs')
    valid = (positions != -1).any(-1)
    require((valid.sum(-1) == PATCHES).all()
            and valid[:, :PATCHES].all() and not valid[:, PATCHES:].any()
            and (positions[:, :PATCHES] >= 0).all(),
            'Official HF reference has unexpected valid/masked patch positions')
    compact = pixels[:, :PATCHES].to(torch.bfloat16).contiguous()
    compact_positions = positions[:, :PATCHES].contiguous()
    prompt = data['input_ids'][0].tolist()
    receipt = {'shape': list(compact.shape), 'dtype': str(compact.dtype),
        'sha256': _tensor_sha(compact), 'positions_sha256': _tensor_sha(compact_positions),
        'positions_shape': list(compact_positions.shape),
        'original_patch_slots_per_frame': 2520, 'valid_patch_slots_per_frame': PATCHES,
        'removed_slots_all_position_minus_one': True,
        'soft_tokens_per_frame': SOFT_TOKENS, 'video_max_soft_tokens': 280,
        'dtype_conversion': 'official HF float32 to native model bfloat16',
        'timestamps': [i / 60. for i in range(180)],
        'reference_tensor_submitted_to_engine': False}
    require(torch.cuda.is_initialized() == cuda_was_initialized,
            'Official HF CPU reference changed CUDA initialization state')
    return prompt, compact, compact_positions, receipt


class _ContractReader:
    def __init__(self, original, owner):
        self.original, self.owner = original, owner

    def load_file(self, path):
        result = self.original.load_file(path)
        source, metadata = result.media
        _check_reader(source, metadata)
        self.owner._reader_metadata = dict(metadata)
        return result


class FullInputAdapter(NativeCompactAdapter):
    def __init__(self, profile, engine=None):
        global _CURRENT_ADAPTER
        require(_CURRENT_ADAPTER is None, 'Only one full Gemma input adapter may be active')
        _check_dependencies(profile)
        reference = verified(profile['reference_prompt_catalog'])
        require(reference.get('schema') == 'gemma-x-full-task-prompts-v1',
                'Full Gemma task prompt table schema differs')
        table = reference['task_contracts']
        prompts = reference['prompt_token_ids']
        require(set(table) == set(prompts) == set(TASKS), 'Exactly three task references are required')
        for task in TASKS:
            require(isinstance(prompts[task], list) and prompts[task]
                    and all(isinstance(token, int) and not isinstance(token, bool)
                            for token in prompts[task]), 'Invalid sealed task prompt tokens')
            require(reference['prompt_ids_sha256'][task] == _sha(json_bytes(prompts[task])),
                    'Sealed task token digest differs')
        catalog = verified(profile['source_catalog'])
        ids = profile['assigned_ids']
        require(ids and len(ids) == len(set(ids)), 'Explicit unique assigned IDs are required')
        assigned = {}
        for sid in ids:
            require(sid in catalog['items'], 'Assigned ID missing from sealed source catalog')
            item = catalog['items'][sid]
            _check_item(item, table)
            assigned[sid] = item
        super().__init__(profile, engine)
        self.prompts = {sid: self.prompts[item['task']] for sid, item in assigned.items()}
        self._assigned = assigned
        self._task_table = table
        self._prompt_ids_sha256 = reference['prompt_ids_sha256']
        self._profile_sha256 = digest(profile)
        self._seen_tasks = set()
        self._reader_metadata = None
        self._expected_cpu = None
        self._expected_positions = None
        self._expected_receipt = None
        self._current_task = None
        self._current_deep = False
        self._gpu_admitted = False
        self._native_cpu_observation = None
        self.reader = _ContractReader(self.reader, self)
        compact = self.native_class._apply_hf_processor_main
        owner = self

        def observed_compact(native, mm_items, hf_kwargs):
            import torch
            result = compact(native, mm_items, hf_kwargs)
            if owner.current is None or 'pixel_values_videos' not in result:
                return result
            pixels, positions = result['pixel_values_videos'], result['pixel_position_ids_videos']
            if not isinstance(pixels, torch.Tensor):
                require(len(pixels) == len(positions) == 1, 'Unexpected native compact video layout')
                pixels, positions = pixels[0], positions[0]
            require(isinstance(positions, torch.Tensor)
                    and pixels.device.type == positions.device.type == 'cpu'
                    and pixels.dtype == torch.bfloat16 and positions.dtype == torch.int64
                    and list(pixels.shape) == [180, PATCHES, 768]
                    and list(positions.shape) == [180, PATCHES, 2],
                    'Actual native compact CPU geometry or dtype differs')
            if not owner._current_deep:
                return result
            require(isinstance(positions, torch.Tensor)
                    and owner._expected_cpu is not None
                    and pixels.device.type == positions.device.type == 'cpu'
                    and pixels.dtype == owner._expected_cpu.dtype
                    and positions.dtype == owner._expected_positions.dtype
                    and torch.equal(pixels, owner._expected_cpu)
                    and torch.equal(positions, owner._expected_positions),
                    'Actual native compact CPU pixels or positions differ from independent official HF')
            owner._native_cpu_observation = {
                'actual_native_CPU_compact_sha256': _tensor_sha(pixels),
                'actual_native_positions_sha256': _tensor_sha(positions),
                'valid_pixels_positions_exact_equal': True}
            return result

        self.native_class._apply_hf_processor_main = observed_compact
        _CURRENT_ADAPTER = self

    def prepare(self, item, content):
        sid = item['sample_id']
        require(sid in self._assigned, 'Sample is outside sealed assigned allocation')
        task, question = _check_item(item, self._task_table)
        sealed = self._assigned[sid]
        require(item['content'] == sealed['content'] and item['row'] == sealed['row'],
                'Assigned scientific source item changed')
        self._reader_metadata = None
        result = super().prepare(item, content)
        self._current_task = task
        self._current_deep = task not in self._seen_tasks
        self._gpu_admitted = False
        self._native_cpu_observation = None
        frames, metadata = result['request']['multi_modal_data']['video'][0]
        require(self._reader_metadata is not None and metadata == self._reader_metadata
                and set(result['request']['multi_modal_data']) == {'video'},
                'Full native request metadata or modality differs')
        geometry = result['receipt']['video_padding']
        require(geometry['source_wh'] == [768, 512]
                and geometry['submitted_wh'] == [768, 528]
                and geometry['bottom_rows'] == 16 and geometry['right_columns'] == 0
                and not geometry['effective_pixels_resized_or_cropped'],
                'Full native padding differs')
        receipt = result['receipt']
        receipt.update(route_label='X-native-compact-full-task-reference',
            task_prompt_table=self.p['reference_prompt_catalog'],
            scientific_question_sha256=_sha(question.encode()),
            expected_prompt_ids_sha256=self._prompt_ids_sha256[task],
            expected_prompt_tokens=len(self.prompts[sid]),
            reader_metadata=self._reader_metadata,
            independent_GPU_hash_scope='first_sample_per_task_per_engine',
            independent_CPU_reference_required=self._current_deep)
        if self._current_deep:
            prompt, pixels, positions, cpu_receipt = _official_reference(
                self.processor, result['request']['prompt'], frames)
            require(prompt == self.prompts[sid],
                    'Independent official HF prompt differs from sealed task reference')
            self._expected_cpu, self._expected_positions = pixels, positions
            self._expected_receipt = cpu_receipt
            receipt['independent_CPU_reference'] = cpu_receipt
        self._seen_tasks.add(task)
        return result

    def validate(self, answer):
        result = super().validate(answer)
        require(not self._current_deep or self._gpu_admitted,
                'First task sample lacks independent actual GPU byte admission')
        return {**result, 'sealed_task_prompt_equal': True,
                'independent_GPU_byte_admission': self._gpu_admitted,
                'independent_GPU_hash_scope': 'first_sample_per_task_per_engine'}

    def clear(self):
        self._expected_cpu = None
        self._expected_positions = None
        self._expected_receipt = None
        self._reader_metadata = None
        self._current_task = None
        self._current_deep = False
        self._gpu_admitted = False
        self._native_cpu_observation = None
        self.compact_calls = 0
        super().clear()

    def close(self):
        global _CURRENT_ADAPTER
        try:
            super().close()
        finally:
            if _CURRENT_ADAPTER is self:
                _CURRENT_ADAPTER = None


def create(profile, engine=None):
    return FullInputAdapter(profile, engine)


def validate_vision(profile, item, deep):
    """Compare actual forward chunk bytes with this sample's independent HF CPU reference."""
    adapter = _CURRENT_ADAPTER
    require(adapter is not None and adapter.current is not None
            and adapter._current_deep and adapter._current_task == item['task']
            and adapter._expected_cpu is not None and adapter._native_cpu_observation
            and not adapter._gpu_admitted,
            'No matching first-task CPU input reference is active')
    require(digest(profile) == adapter._profile_sha256
            and item == adapter._assigned[adapter.current],
            'GPU admission profile or source item differs from the active request')
    validate_compact_vision(profile, item, deep)
    chunks, cursor = [], 0
    for worker_index, worker in enumerate(deep):
        require(worker.get('outer_path') == worker.get('normalized_path')
                == 'vision_tower.patch_embedder', 'Unexpected actual Gemma input observation point')
        for row in worker['normalized']:
            tensor = row['tensor']
            shape = tensor['shape']
            require(len(shape) == 3 and shape[1:] == [PATCHES, 768]
                    and tensor['dtype'] == 'torch.bfloat16'
                    and tensor['device'].startswith('cuda') and shape[0] > 0
                    and cursor + shape[0] <= 180,
                    'Unexpected actual GPU compact input chunk')
            expected_sha = _tensor_sha(adapter._expected_cpu[cursor:cursor + shape[0]])
            require(tensor.get('sha256') == expected_sha,
                    'Actual GPU compact patch bytes differ from independent official HF reference')
            chunks.append({'worker_index': worker_index, 'offset': cursor,
                'frame_count': shape[0], 'shape': shape,
                'expected_CPU_sha256': expected_sha, 'actual_GPU_sha256': tensor['sha256'],
                'outer_index': row.get('outer_index')})
            cursor += shape[0]
    require(cursor == 180 and chunks, 'Independent actual GPU admission did not cover all180 frames')
    receipt = {'schema': 'gemma-x-full-input-admission-v1', 'status': 'PASS',
        'sample_id': adapter.current, 'task': item['task'],
        'profile_sha256': adapter._profile_sha256,
        'source_item_digest': digest(adapter._assigned[adapter.current]),
        'source_catalog': adapter.p['source_catalog'],
        'source_video_sha256': adapter._assigned[adapter.current]['content'][0]['sha256'],
        'scientific_question_sha256': adapter._task_table[item['task']]['question_sha256'],
        'observed_scope': 'first_sample_per_task_per_engine',
        'prompt_table': adapter.p['reference_prompt_catalog'],
        'expected_prompt_ids_sha256': adapter._prompt_ids_sha256[item['task']],
        'expected_CPU_compact_sha256': adapter._expected_receipt['sha256'],
        'positions_sha256': adapter._expected_receipt['positions_sha256'],
        'native_CPU_observation': adapter._native_cpu_observation,
        'actual_GPU_chunks_equal_independent_CPU': True, 'frame_count': cursor,
        'patch_slots_per_frame': PATCHES, 'soft_tokens_per_frame': SOFT_TOKENS,
        'video_max_soft_tokens': 280, 'reader_metadata': adapter._reader_metadata,
        'chunks': chunks, 'reference_tensor_submitted_to_engine': False}
    _append(Path(profile['output']) / 'input-admissions.jsonl', receipt)
    adapter._gpu_admitted = True
    return receipt


def build_prompt_catalog(profile, catalog, output):
    """CPU-only: seal three official HF task prompts from real representative videos.

    ``profile`` is a profile dict (or JSON path); ``catalog`` is a source-catalog
    dict, verified descriptor, or path. No model or engine is constructed.
    """
    import torch
    from transformers import AutoProcessor
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO
    require(not torch.cuda.is_initialized(), 'Prompt catalog builder requires a CPU-only process')
    if not isinstance(profile, dict):
        profile = read(profile)
    if not isinstance(catalog, dict):
        catalog = read(catalog)
    elif 'items' not in catalog:
        catalog = verified(catalog)
    items, contracts, representatives = catalog['items'], {}, {}
    for sid, item in items.items():
        task, question = _check_item(item)
        contract = {'question': question, 'question_sha256': _sha(question.encode()),
                    'video_contract': dict(CONTRACT), 'content_types': ['video', 'text']}
        if task in contracts:
            require(contracts[task] == contract, 'Task does not have a single scientific prompt contract')
        else:
            contracts[task], representatives[task] = contract, sid
    require(set(contracts) == set(TASKS), 'Catalog must contain all three classic tasks')
    processor = AutoProcessor.from_pretrained(profile['processor_checkpoint'],
        local_files_only=True, trust_remote_code=False)
    reader = VideoMediaIO(ImageMediaIO(), video_backend='opencv', num_frames=180, fps=-1)
    prompts, receipts = {}, {}
    for task in TASKS:
        sid = representatives[task]
        item = items[sid]
        media = item['content'][0]
        require(file_sha256(media['path']) == media['sha256'], 'Representative video bytes changed')
        source, metadata = reader.load_file(Path(media['path'])).media
        _check_reader(source, metadata)
        frames, geometry = pad(source)
        text = _template(processor, contracts[task]['question'])
        prompt, pixels, positions, receipt = _official_reference(processor, text, frames)
        prompts[task] = prompt
        contracts[task]['representative_sample_id'] = sid
        receipts[task] = {'sample_id': sid, 'source_video': descriptor(media['path']),
            'reader_metadata': metadata, 'padding': geometry, 'official_HF_CPU': receipt,
            'prompt_tokens': len(prompt), 'prompt_ids_sha256': _sha(json_bytes(prompt))}
        del source, frames, pixels, positions
    result = {'schema': 'gemma-x-full-task-prompts-v1', 'prompt_token_ids': prompts,
        'prompt_ids_sha256': {task: _sha(json_bytes(prompts[task])) for task in TASKS},
        'task_contracts': contracts, 'representatives': receipts,
        'processor_checkpoint': profile['processor_checkpoint'],
        'source_catalog_digest': digest(catalog), 'model_calls': 0,
        'CUDA_initialized': torch.cuda.is_initialized(), 'weights_loaded': False}
    require(not result['CUDA_initialized'], 'Prompt catalog builder initialized CUDA')
    write_new(output, result)
    return {'status': 'PASS', 'reference_prompt_catalog': descriptor(output), 'model_calls': 0,
            'CUDA_initialized': False,
            'prompt_tokens_by_task': {task: len(prompts[task]) for task in TASKS}}
