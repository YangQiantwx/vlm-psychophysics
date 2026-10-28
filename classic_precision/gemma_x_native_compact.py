"""Combine the native input accounting fix with verified masked-slot removal."""
from __future__ import annotations

from .gemma_x_native_input import NativeAdapter
from .official_x_gemma_adapter import require


class NativeCompactAdapter(NativeAdapter):
    def __init__(self, profile, engine=None):
        super().__init__(profile, engine)
        self.native_class = self.gm.Gemma4MultiModalProcessor
        self.original_apply = self.native_class._apply_hf_processor_main
        original = self.original_apply
        owner = self

        def compact(native, mm_items, hf_kwargs):
            import torch
            result = original(native, mm_items, hf_kwargs)
            if owner.current is None or 'pixel_values_videos' not in result:
                return result
            pixels = result['pixel_values_videos']
            positions = result['pixel_position_ids_videos']
            flat = isinstance(pixels, torch.Tensor)
            if flat:
                require(isinstance(positions, torch.Tensor), 'Mixed native tensor layout')
                pixels, positions = [pixels], [positions]
            require(len(pixels) == len(positions) == 1, 'Expected one complete video')
            new_pixels, new_positions = [], []
            for v, pos in zip(pixels, positions):
                valid = (pos != -1).any(-1); counts = valid.sum(-1)
                require(len(counts) == 180 and (counts == counts[0]).all(),
                        'Native processor changed frame/patch counts')
                n = int(counts[0])
                require(n % 9 == 0 and valid[:, :n].all() and not valid[:, n:].any(),
                        'Cannot remove valid or noncontiguous patch positions')
                new_pixels.append(v[:, :n].contiguous())
                new_positions.append(pos[:, :n].contiguous())
            result['pixel_values_videos'] = new_pixels[0] if flat else new_pixels
            result['pixel_position_ids_videos'] = new_positions[0] if flat else new_positions
            owner.compact_calls += 1
            return result

        self.compact_calls = 0
        self.native_class._apply_hf_processor_main = compact

    def validate(self, answer):
        result = super().validate(answer)
        require(self.compact_calls > 0, 'Native masked-slot removal did not execute')
        self.compact_calls = 0
        return {**result, 'native_masked_slot_removal': True}

    def close(self):
        self.native_class._apply_hf_processor_main = self.original_apply
        super().close()


def create(profile, engine=None):
    return NativeCompactAdapter(profile, engine)
