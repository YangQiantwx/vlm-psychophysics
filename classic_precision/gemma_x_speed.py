"""Finite Gemma X speed candidates; original scientific modules stay unchanged.

The compact candidate removes only processor slots whose position IDs are -1
for every frame. It preserves every source pixel, frame, valid patch, prompt,
dtype conversion and the stock vLLM model/attention/pooling implementation.
It is a distinct, explicitly tested layout, not an assumed bitwise equivalent.
"""
from __future__ import annotations

from .official_x_gemma_adapter import Adapter, require, tensor_description


class CompactAdapter(Adapter):
    def prepare(self, item, content):
        result = super().prepare(item, content)
        payload = self.current['payload']
        positions = payload['pixel_position_ids_videos']
        pixels = payload['pixel_values_videos']
        valid = (positions != -1).any(-1)
        counts = valid.sum(-1)
        require(len(counts) == 180 and (counts == counts[0]).all(),
                'Unequal or incomplete frame patch counts')
        n = int(counts[0])
        require(n % 9 == 0 and valid[:, :n].all() and not valid[:, n:].any(),
                'Only contiguous, fully masked processor padding may be removed')
        payload['pixel_values_videos'] = pixels[:, :n].contiguous()
        payload['pixel_position_ids_videos'] = positions[:, :n].contiguous()
        require(payload['pixel_values_videos'].equal(pixels[:, :n]) and
                payload['pixel_position_ids_videos'].equal(positions[:, :n]),
                'Valid patch data changed')
        receipt = result['receipt']
        receipt['compact_processor_padding'] = {
            'original_patch_slots_per_frame': int(pixels.shape[1]),
            'submitted_patch_slots_per_frame': n,
            'removed_slots_all_position_minus_one': True,
            'valid_pixels_positions_exact_equal': True,
            'video_frames': 180,
            'model_forward_pooling_changed': False,
            'submitted_pixels': tensor_description(payload['pixel_values_videos'],
                                                   item['task'] not in getattr(self, '_compact_seen', set())),
        }
        if not hasattr(self, '_compact_seen'):
            self._compact_seen = set()
        self._compact_seen.add(item['task'])
        return result


def create_compact(profile, engine=None):
    return CompactAdapter(profile, engine)


def validate_compact_vision(profile, item, deep):
    contract = item.get('video_contract', profile.get('video_contract', {}))
    w, h = contract['width'], contract['height']
    video_patches = ((h + 47) // 48) * ((w + 47) // 48) * 9
    image_patches = int(profile.get('adapter_image_max_soft_tokens', 140)) * 9
    videos, images = [], []
    for worker in deep:
        for row in worker['normalized']:
            t = row['tensor']; shape = t['shape']
            require(t['dtype'] == 'torch.bfloat16' and t['device'].startswith('cuda'),
                    'Unexpected GPU input dtype/device')
            require(len(shape) == 3 and shape[-1] == 768, 'Unexpected patch geometry')
            if shape[1] == video_patches:
                videos.append(t)
            elif shape[1] == image_patches:
                images.append(t)
            else:
                raise ValueError('Unexpected actual GPU patch shape: ' + str(shape))
    require(sum(t['shape'][0] for t in videos) == 180, 'Actual GPU frames != 180')
    require(sum(t['shape'][0] for t in images) ==
            sum(x['type'] == 'image' for x in item.get('content', [])),
            'Actual reference image count differs')
    return {'status': 'PASS', 'actual_GPU_video_frames': 180,
            'video_patches_per_frame': video_patches, 'model_math_modified': False}
