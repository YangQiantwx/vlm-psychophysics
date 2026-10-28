"""Gemma native video route with a process-local no-resize accounting fix.

Only the CPU input budget and soft-token count are corrected. The stock vLLM
HF processor, model forward, pooling, quantization and SDK files are unchanged.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from urllib.parse import unquote, urlparse

from .official_x_gemma_adapter import pad, require


class NativeAdapter:
    def __init__(self, profile, engine=None):
        import vllm.model_executor.models.gemma4_mm as gm
        from transformers import AutoProcessor
        from vllm.multimodal.media.video import VideoMediaIO
        from vllm.multimodal.media.image import ImageMediaIO
        self.p = profile
        self.processor = AutoProcessor.from_pretrained(profile['processor_checkpoint'],
            local_files_only=True, trust_remote_code=False)
        self.reader = VideoMediaIO(ImageMediaIO(), video_backend='opencv', num_frames=180, fps=-1)
        self.images = ImageMediaIO()
        self.gm = gm
        self.old_budget = gm._VIDEO_MAX_SOFT_TOKENS
        self.old_count = gm.Gemma4ProcessingInfo._compute_num_soft_tokens
        original = self.old_count

        def count(info, image_width, image_height, max_soft_tokens=None):
            if info.ctx.get_merged_mm_kwargs({}).get('do_resize') is False:
                require(image_width % 48 == image_height % 48 == 0,
                        'No-resize Gemma canvas must align to patch/pool grid')
                return image_width * image_height // (48 * 48)
            return original(info, image_width, image_height, max_soft_tokens)

        gm._VIDEO_MAX_SOFT_TOKENS = 280
        gm.Gemma4ProcessingInfo._compute_num_soft_tokens = count
        descriptor = profile['reference_prompt_catalog']
        data = Path(descriptor['path']).read_bytes()
        require(hashlib.sha256(data).hexdigest() == descriptor['sha256'],
                'Reference prompt identity differs')
        self.prompts = json.loads(data)['prompt_token_ids']
        self.current = None
        self.seen = set()

    def prepare(self, item, content):
        import numpy as np
        from PIL import Image
        require(self.current is None, 'Previous native request not cleared')
        entries = []
        for x in content:
            if x['type'] in ('video_url', 'image_url'):
                kind = x['type'].removesuffix('_url'); u = urlparse(x[x['type']]['url'])
                require(u.scheme == 'file' and u.netloc in ('', 'localhost'), 'Local sealed media required')
                entries.append({'type': kind, 'path': unquote(u.path)})
            else:
                entries.append(x)
        videos = [x for x in entries if x['type'] == 'video']
        images = [x for x in entries if x['type'] == 'image']
        require(len(videos) == 1 and len(images) <= 1, 'Unexpected modality layout')
        require([x['text'] for x in entries if x['type'] == 'text'] == [item['row']['prompt']],
                'Scientific question changed')
        t = time.monotonic(); source, metadata = self.reader.load_file(Path(videos[0]['path'])).media
        c = item.get('video_contract', self.p.get('video_contract', {}))
        require(source.dtype == np.uint8 and list(source.shape) == [180, c['height'], c['width'], 3]
                and metadata['frames_indices'] == list(range(180)), 'Reader source contract differs')
        read_seconds = time.monotonic() - t; t = time.monotonic()
        frames, geometry = pad(source); mm = {'video': [(frames, metadata)]}
        reference_geometry = None
        if images:
            img = self.images.load_file(Path(images[0]['path'])).media
            values, reference_geometry = pad(np.asarray(img.convert('RGB')))
            mm['image'] = [Image.fromarray(values)]
        template = [{'type': x['type'], **({'text': x['text']} if x['type'] == 'text' else {})}
                    for x in entries]
        text = self.processor.apply_chat_template([{'role': 'user', 'content': template}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        sid = item['sample_id']; require(sid in self.prompts, 'Unsealed prompt reference')
        self.current = sid
        receipt = {'route_label': 'X-native-input-fix', 'video_padding': geometry,
            'reference_padding': reference_geometry, 'video_frames': 180,
            'video_indices': list(range(180)), 'video_max_soft_tokens': 280,
            'video_effective_tokens_per_frame': frames.shape[1] * frames.shape[2] // 2304,
            'shared_SDK_files_modified': False, 'precomputed_tensor_bridge': False,
            'model_forward_pooling_changed': False}
        return {'request': {'prompt': text, 'multi_modal_data': mm,
                'mm_processor_kwargs': {'do_resize': False, 'do_sample_frames': False, 'max_soft_tokens': 140}},
            'receipt': receipt, 'timing': {'official_reader_external_inclusive_seconds': read_seconds,
                'padding_and_template_external_seconds': time.monotonic() - t}}

    def validate(self, answer):
        require(self.current is not None and list(answer.prompt_token_ids) == self.prompts[self.current],
                'Actual native prompt differs from sealed baseline')
        return {'status': 'PASS', 'actual_prompt_equal_sealed_baseline': True}

    def clear(self):
        self.current = None

    def close(self):
        self.clear()
        self.gm._VIDEO_MAX_SOFT_TOKENS = self.old_budget
        self.gm.Gemma4ProcessingInfo._compute_num_soft_tokens = self.old_count


def create(profile, engine=None):
    return NativeAdapter(profile, engine)
