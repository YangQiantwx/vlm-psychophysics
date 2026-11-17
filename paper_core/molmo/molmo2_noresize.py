"""Molmo2 input bridge for the frozen 768×512 X panel, with pad only."""

import hashlib
import importlib
from types import MethodType

import numpy as np


HEIGHT, WIDTH = 512, 768
FRAMES, FPS = 180, 60
PATCH = 14
PATCH_ROWS, PATCH_COLS = 37, 55
PAD_HEIGHT, PAD_WIDTH = PATCH_ROWS * PATCH, PATCH_COLS * PATCH
PATCHES = PATCH_ROWS * PATCH_COLS
PIXELS_PER_PATCH = PATCH * PATCH * 3
CONTEXT = 65_536
MODEL_CONTEXT = 36_864
VIDEO_GRID = (13, 19)
IMAGE_GRID = (19, 28)
VIDEO_EMBED_TOKENS = FRAMES * VIDEO_GRID[0] * VIDEO_GRID[1]
IMAGE_EMBED_TOKENS = IMAGE_GRID[0] * IMAGE_GRID[1]


def pack_pixels(frames):
    """Normalize and patchify the complete RGB8 array without rescaling coordinates."""
    if frames.dtype != np.uint8 or frames.shape[1:] not in (
        (HEIGHT, WIDTH, 3), (PAD_HEIGHT, PAD_WIDTH, 3)
    ):
        raise ValueError(f"Unexpected RGB8 dimensions: {frames.shape}, {frames.dtype}")
    height, width = frames.shape[1:3]
    out = np.empty((len(frames), PATCHES, PIXELS_PER_PATCH), dtype=np.float32)
    for start in range(0, len(frames), 12):
        block = frames[start:start + 12]
        padded = np.full((len(block), PAD_HEIGHT, PAD_WIDTH, 3), -1.0, dtype=np.float32)
        real = padded[:, :height, :width]
        np.divide(block, np.float32(255), out=real, dtype=np.float32)
        real -= np.float32(0.5)
        real *= np.float32(2)
        patches = padded.reshape(len(block), PATCH_ROWS, PATCH, PATCH_COLS, PATCH, 3)
        out[start:start + len(block)] = patches.transpose(0, 1, 3, 2, 4, 5).reshape(
            len(block), PATCHES, PIXELS_PER_PATCH
        )
    return out


def restore_pixels(patches):
    n = len(patches)
    hwc = patches.reshape(n, PATCH_ROWS, PATCH_COLS, PATCH, PATCH, 3)
    hwc = hwc.transpose(0, 1, 3, 2, 4, 5).reshape(n, PAD_HEIGHT, PAD_WIDTH, 3)
    return np.rint((hwc + np.float32(1)) * np.float32(127.5)).astype(np.uint8)


def sha(array):
    return hashlib.sha256(np.ascontiguousarray(array)).hexdigest()


def pooling_index(upstream, pool_h, pool_w):
    patch_ids = np.arange(PATCHES).reshape(PATCH_ROWS, PATCH_COLS)
    return upstream.arange_for_pooling(patch_ids, pool_h, pool_w).reshape(-1, pool_h * pool_w)


def video_preprocess(self, videos, return_tensors=None, **kwargs):
    from transformers.feature_extraction_utils import BatchFeature

    upstream = importlib.import_module(type(self).__module__)
    indices = pooling_index(upstream, 3, 3)
    if any(len(clip) != FRAMES for clip in videos):
        raise ValueError("Molmo2 X video must contain exactly 180 frames")
    if len(videos) != 1:
        raise ValueError("Molmo2 X takes a single clip per request")
    offsets = np.arange(FRAMES)[:, None, None] * PATCHES
    all_indices = np.where(indices >= 0, indices + offsets, -1).reshape(-1, 9)
    return BatchFeature(
        dict(
            pixel_values_videos=pack_pixels(videos[0]),
            video_token_pooling=all_indices,
            video_grids=np.array([[FRAMES, *VIDEO_GRID]]),
        ),
        tensor_type=return_tensors,
    )


def image_preprocess(self, images, return_tensors=None, *, video_module, **kwargs):
    from transformers.feature_extraction_utils import BatchFeature

    # Keep the high-resolution input and both special-token pairs. The resized
    # overview is omitted by giving it an empty grid.
    upstream = importlib.import_module(type(self).__module__)
    images = upstream.make_flat_list_of_images(images)
    frames = np.stack([np.asarray(upstream.convert_to_rgb(image)) for image in images])
    indices = pooling_index(video_module, 2, 2)
    return BatchFeature(
        dict(
            pixel_values=pack_pixels(frames),
            image_token_pooling=np.tile(indices, (len(images), 1)),
            image_grids=np.tile([0, 0, *IMAGE_GRID], (len(images), 1)),
            image_num_crops=np.ones(len(images), dtype=np.int64),
        ),
        tensor_type=return_tensors,
    )


def configure_processor(processor):
    image = processor.image_processor
    video = processor.video_processor
    if (image.patch_size, video.patch_size, tuple(image.pooling_size), tuple(video.pooling_size)) != (
        PATCH, PATCH, (2, 2), (3, 3)
    ):
        raise ValueError("Molmo2 checkpoint patch or pooling configuration differs")
    for part in (image, video):
        part.size = {"height": PAD_HEIGHT, "width": PAD_WIDTH}
        part.do_resize = False
    video.num_frames = FRAMES
    video.do_sample_frames = False
    video._preprocess = MethodType(video_preprocess, video)
    video_module = importlib.import_module(type(video).__module__)

    def preprocess_images(self, images, return_tensors=None, **kwargs):
        return image_preprocess(self, images, return_tensors, video_module=video_module, **kwargs)

    image.preprocess = MethodType(preprocess_images, image)
    return processor


def patch_vllm():
    from vllm.model_executor.models import molmo2

    info = molmo2.Molmo2ProcessingInfo
    original_get = info.get_hf_processor

    def get_processor(self, **kwargs):
        return configure_processor(original_get(self, **kwargs))

    def grid(self, processor):
        return (VIDEO_GRID[1], VIDEO_GRID[0]) if hasattr(processor, "do_sample_frames") else (0, 0)

    # The vLLM processor uses these sizes for placeholder expansion and for
    # its independent check of the processor's crop count.
    info.get_hf_processor = get_processor
    info.get_base_grid_size = grid
    info.select_tiling = lambda self, **kwargs: (1, 0)
    info.get_patches_grid_size = lambda self, **kwargs: (IMAGE_GRID[1], IMAGE_GRID[0])
    info.get_image_size_with_most_features = lambda self: molmo2.ImageSize(WIDTH, HEIGHT)


def hardware_config(config):
    if not hasattr(config, "text_config"):
        return config  # vLLM probes the model type with an empty config first.
    if config.text_config.max_position_embeddings not in (MODEL_CONTEXT, CONTEXT):
        raise ValueError("Unexpected Molmo2 checkpoint context")
    config.text_config.max_position_embeddings = CONTEXT
    config.vit_config.image_default_input_size = (PAD_HEIGHT, PAD_WIDTH)
    return config


def observe_vision(model):
    """Observe the real ViT arguments from the in-process TP=1 worker."""
    import torch

    vit = model.vision_backbone.image_vit
    if tuple(vit.patch_num) != (PATCH_ROWS, PATCH_COLS):
        raise ValueError(f"Model patch grid differs: {vit.patch_num}")
    model.cvpr_vision_calls = []
    model.cvpr_deep_input = False

    def capture(module, args):
        (patches,) = args
        evidence = {"shape": list(patches.shape), "dtype": str(patches.dtype),
                    "device": str(patches.device)}
        if model.cvpr_deep_input:
            # Bound CPU transfer and working memory while checking every frame.
            hashes = []
            for chunk in patches.split(12):
                rgb = restore_pixels(chunk.detach().to(device="cpu", dtype=torch.float32).numpy())
                if np.any(rgb[:, HEIGHT:]) or np.any(rgb[:, :, WIDTH:]):
                    raise ValueError("Nonzero pixels outside the original canvas")
                hashes.extend(sha(frame[:HEIGHT, :WIDTH]) for frame in rgb)
            evidence["rgb_frame_sha256"] = hashes
        model.cvpr_vision_calls.append(evidence)

    vit.register_forward_pre_hook(capture)
    return {"patch_grid_hw": list(vit.patch_num), "dtype": str(vit.patch_embedding.weight.dtype),
            "original_position_embeddings": list(vit.positional_embedding.shape)}


def reuse_equal_frame_features(model):
    """Evaluate repeated SigLIP inputs once, then restore every timed slot."""
    import torch

    vit = model.vision_backbone.image_vit
    baseline = vit.forward

    def forward(patches, *args, **kwargs):
        indices = getattr(model, "cvpr_equal_frame_indices", None)
        if len(patches) != FRAMES or indices is None:
            return baseline(patches, *args, **kwargs)
        first, inverse = indices
        if len(first) == FRAMES:
            return baseline(patches, *args, **kwargs)
        first = torch.tensor(first, dtype=torch.long, device=patches.device)
        inverse = torch.tensor(inverse, dtype=torch.long, device=patches.device)
        representatives = patches.index_select(0, first)
        if not torch.equal(representatives.index_select(0, inverse), patches):
            raise ValueError("GPU ViT inputs differ within a proposed duplicate group")
        features = baseline(representatives, *args, **kwargs)
        return [feature.index_select(0, inverse) if i in model.vision_backbone.vit_layers else feature
                for i, feature in enumerate(features)]

    vit.forward = forward
