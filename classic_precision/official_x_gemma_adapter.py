"""Gemma X-adapted input boundary: official HF processing with fixed padding.

The vLLM model, pooling, generation and quantization kernels are untouched. This
bridge supplies the official HF video tensors and their real soft-token counts,
because the pinned native video reader unconditionally budgets 70 tokens/frame.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import time


def require(value, message):
    if not value:
        raise ValueError(message)


def tensor_description(value, deep=False):
    import torch
    result = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if deep:
        result['sha256'] = hashlib.sha256(
            value.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


def pad(values, unit=48):
    """Extend bottom/right only, with the source first frame's background RGB."""
    import numpy as np
    video = values.ndim == 4
    h, w = values.shape[-3:-1]
    bottom, right = (-h) % unit, (-w) % unit
    background = values[0, 0, 0] if video else values[0, 0]
    shape = (values.shape[0], h + bottom, w + right, 3) if video else (h + bottom, w + right, 3)
    output = np.empty(shape, dtype=values.dtype)
    output[...] = background
    if video:
        output[:, :h, :w] = values
    else:
        output[:h, :w] = values
    return output, {'source_wh': [w, h], 'submitted_wh': [w + right, h + bottom],
                    'bottom_rows': bottom, 'right_columns': right,
                    'padding_RGB': background.tolist(), 'effective_pixels_resized_or_cropped': False}


class Adapter:
    def __init__(self, profile, engine=None):
        from transformers import AutoProcessor
        from transformers.feature_extraction_utils import BatchFeature
        from vllm.model_executor.models.gemma4_mm import Gemma4MultiModalProcessor
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.media.image import ImageMediaIO
        self.profile = profile
        self.processor = AutoProcessor.from_pretrained(
            profile['processor_checkpoint'], local_files_only=True, trust_remote_code=False)
        self.video_loader = VideoMediaIO(ImageMediaIO(), video_backend='opencv', num_frames=180, fps=-1)
        self.image_loader = ImageMediaIO()
        self.seen_tasks = set()
        self.current = None
        self.native_class = Gemma4MultiModalProcessor
        self.original = Gemma4MultiModalProcessor._apply_hf_processor_main
        adapter = self

        def prepared(native, mm_items, hf_kwargs):
            if adapter.current is None:
                return adapter.original(native, mm_items, hf_kwargs)
            state = adapter.current
            require(mm_items.get_all_counts() == state['counts'], 'Gemma bridge modality count changed')
            videos = mm_items['video'].get_processor_data()['videos']
            require(len(videos) == 1, 'Gemma bridge requires one complete video')
            values, metadata = videos[0]
            expected = state['frames']
            require(values.shape == expected.shape and values.dtype == expected.dtype
                    and values.strides == expected.strides
                    and values.__array_interface__['data'][0] == expected.__array_interface__['data'][0],
                    'Gemma bridge submitted video array changed')
            require(metadata['frames_indices'] == list(range(180)) and metadata['fps'] == 60.,
                    'Gemma bridge submitted frame order changed')
            if state['counts'].get('image'):
                images = mm_items['image'].get_processor_data()['images']
                require(len(images) == 1 and images[0].size == state['image'].size,
                        'Gemma reference image geometry changed')
            state['calls'] += 1
            return BatchFeature(state['payload'])

        Gemma4MultiModalProcessor._apply_hf_processor_main = prepared

    def prepare(self, item, content):
        import numpy as np
        import torch
        from PIL import Image
        from transformers.video_utils import VideoMetadata
        from urllib.parse import unquote, urlparse
        require(self.current is None, 'Previous Gemma bridge request was not cleared')
        normalized_content = []
        for entry in content:
            if entry['type'] in ('video_url', 'image_url'):
                kind = entry['type'].removesuffix('_url')
                url = urlparse(entry[entry['type']]['url'])
                require(url.scheme == 'file' and url.netloc in ('', 'localhost'),
                        'Gemma input bridge accepts sealed local files only')
                normalized_content.append({'type': kind, 'path': unquote(url.path)})
            else:
                normalized_content.append(entry)
        content = normalized_content
        deep = item['task'] not in self.seen_tasks
        self.seen_tasks.add(item['task'])
        video_entries = [x for x in content if x['type'] == 'video']
        image_entries = [x for x in content if x['type'] == 'image']
        require(len(video_entries) == 1 and len(image_entries) <= 1, 'Unsupported Gemma modality layout')
        require([x['text'] for x in content if x['type'] == 'text'] == [item['row']['prompt']],
                'Gemma canonical question changed')
        start = time.monotonic()
        source, metadata = self.video_loader.load_file(Path(video_entries[0]['path'])).media
        contract = item.get('video_contract', self.profile.get('video_contract', {}))
        require(source.dtype == np.uint8 and list(source.shape) == [180, contract['height'], contract['width'], 3]
                and metadata['frames_indices'] == list(range(180)), 'Gemma official reader contract changed')
        reader = {'path': video_entries[0]['path'], 'shape': list(source.shape), 'dtype': str(source.dtype),
                  'metadata': metadata}
        if deep:
            reader.update(decoded_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
                          per_frame_sha256=[hashlib.sha256(x.tobytes()).hexdigest() for x in source])
        read_seconds = time.monotonic() - start
        start = time.monotonic()
        frames, video_pad = pad(source)
        reference = None
        image_pad = None
        image_read = None
        if image_entries:
            original_image = self.image_loader.load_file(Path(image_entries[0]['path'])).media
            image_values = np.asarray(original_image.convert('RGB'))
            image_read = {'path': image_entries[0]['path'], 'shape': list(image_values.shape),
                          'dtype': str(image_values.dtype)}
            if deep:
                image_read['decoded_sha256'] = hashlib.sha256(image_values.tobytes()).hexdigest()
            padded_image, image_pad = pad(image_values)
            reference = Image.fromarray(padded_image)
        pad_seconds = time.monotonic() - start
        template = [{'type': x['type']} if x['type'] != 'text' else {'type': 'text', 'text': x['text']}
                    for x in content]
        text = self.processor.apply_chat_template([{'role': 'user', 'content': template}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        require(not text.rstrip().endswith('<think>'), 'Gemma thinking unexpectedly enabled')
        video_metadata = VideoMetadata(total_num_frames=180, fps=60., duration=3., frames_indices=list(range(180)))
        kwargs = dict(text=[text], videos=[frames], videos_kwargs={
            'do_resize': False, 'do_sample_frames': False, 'max_soft_tokens': 280,
            'video_metadata': [video_metadata]}, return_tensors='pt')
        image_budget = int(self.profile.get('adapter_image_max_soft_tokens', 140))
        if reference is not None:
            kwargs.update(images=[reference], images_kwargs={'do_resize': False, 'max_soft_tokens': image_budget})
        start = time.monotonic()
        data = self.processor(**kwargs)
        processor_seconds = time.monotonic() - start
        start = time.monotonic()
        positions_key = next(k for k in data if 'position_ids' in k and 'video' in k)
        pixels = data['pixel_values_videos'][0]
        positions = data[positions_key][0]
        soft = ((positions != -1).any(-1).sum(-1) // 9).tolist()
        require(len(soft) == 180 and len(set(soft)) == 1, 'Gemma HF video frame count changed')
        expected_soft = (frames.shape[1] // 48) * (frames.shape[2] // 48)
        require(set(soft) == {expected_soft} and list(pixels.shape) == [180, 2520, 768],
                'Gemma HF video geometry changed')
        native = {'pixel_values_videos': tensor_description(pixels, deep),
                  'pixel_position_ids_videos': tensor_description(positions, deep)}
        payload = {'pixel_values_videos': pixels.to(torch.bfloat16), 'pixel_position_ids_videos': positions,
                   'video_frame_counts': torch.tensor([180]), 'video_num_soft_tokens': [soft],
                   'video_timestamps': [[i / 60. for i in range(180)]]}
        image_soft = None
        if reference is not None:
            image_pixels = data['pixel_values']
            image_positions = data.get('image_position_ids', data.get('pixel_position_ids'))
            require(image_positions is not None, 'Gemma HF image positions missing')
            image_soft = ((image_positions != -1).any(-1).sum(-1) // 9).tolist()
            require(image_soft == [121] and list(image_pixels.shape) == [1, image_budget * 9, 768],
                    'Gemma HF reference image geometry changed')
            native.update(pixel_values=tensor_description(image_pixels, deep),
                          pixel_position_ids=tensor_description(image_positions, deep))
            payload.update(pixel_values=image_pixels.to(torch.bfloat16), pixel_position_ids=image_positions)
        post = {k: tensor_description(v, deep) for k, v in payload.items() if isinstance(v, torch.Tensor)}
        expected_prompt = data['input_ids'][0].tolist()
        counts = {'video': 1, **({'image': 1} if reference is not None else {})}
        receipt = {'route_label': 'X-adapted', 'native_reader': [reader], 'reference_reader': image_read,
                   'native_processor': [{'before_native_dtype': native, 'after_native_dtype': post}],
                   'video_padding': video_pad, 'reference_padding': image_pad,
                   'video_max_soft_tokens': 280, 'video_effective_tokens_per_frame': soft,
                   'image_max_soft_tokens': image_budget if reference is not None else None,
                   'image_effective_tokens': image_soft, 'processor': type(self.processor).__name__,
                   'post_processor_cast': 'torch.bfloat16', 'expected_prompt_tokens': len(expected_prompt),
                   'expected_prompt_ids_sha256': hashlib.sha256(json_bytes(expected_prompt)).hexdigest(),
                   'GPU_model_math_modified': False}
        self.current = {'frames': frames, 'image': reference, 'payload': payload, 'counts': counts,
                        'calls': 0, 'expected_prompt': expected_prompt, 'receipt': receipt}
        mm_data = {'video': [(frames, metadata)]}
        if reference is not None:
            mm_data['image'] = [reference]
        return {'request': {'prompt': text, 'multi_modal_data': mm_data}, 'receipt': receipt,
                'timing': {'official_reader_external_inclusive_seconds': read_seconds,
                           'padding_and_reference_read_external_seconds': pad_seconds,
                           'official_HF_processor_external_seconds': processor_seconds,
                           'post_processor_cast_payload_external_seconds': time.monotonic() - start}}

    def validate(self, answer):
        require(self.current is not None and self.current['calls'] >= 1, 'Gemma input bridge did not execute')
        require(list(answer.prompt_token_ids) == self.current['expected_prompt'],
                'Gemma actual prompt expansion differs from official HF processor')
        return {'status': 'PASS', 'bridge_calls': self.current['calls'],
                'prompt_tokens_equal_official_HF': True}

    def clear(self):
        self.current = None

    def close(self):
        self.clear()
        self.native_class._apply_hf_processor_main = self.original


def json_bytes(value):
    import json
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()


def create(profile, engine=None):
    return Adapter(profile, engine)


def validate_vision(profile, item, deep):
    contract = item.get('video_contract', profile.get('video_contract', {}))
    video_rows = []
    image_rows = []
    for worker in deep:
        for row in worker['normalized']:
            tensor = row['tensor']
            shape = tensor['shape']
            require(tensor['dtype'] == 'torch.bfloat16' and tensor['device'].startswith('cuda'),
                    'Gemma actual GPU patch dtype/device changed')
            require(len(shape) == 3 and shape[-1] == 768, 'Gemma actual patch feature dimension changed')
            if shape[1] == 2520:
                video_rows.append(tensor)
            elif shape[1] == int(profile.get('adapter_image_max_soft_tokens', 140)) * 9:
                image_rows.append(tensor)
            else:
                raise ValueError('Unexpected Gemma actual GPU patch budget: ' + str(shape))
    require(sum(x['shape'][0] for x in video_rows) == contract['frames'] == 180,
            'Gemma did not forward all 180 adapted source frames')
    expected_images = sum(x['type'] == 'image' for x in item.get('content', []))
    require(sum(x['shape'][0] for x in image_rows) == expected_images,
            'Gemma actual reference image count changed')
    return {'status': 'PASS', 'video_frames': 180, 'reference_images': expected_images,
            'GPU_model_math_modified': False}
