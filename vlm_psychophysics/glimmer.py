"""Run Muse Glimmer with the benchmark's original frame and image inputs."""

from __future__ import annotations

from contextlib import nullcontext
import inspect
import re
from pathlib import Path
import time

from . import data, inputs, models, observers


MODEL_ID = "nvidia/Muse-Glimmer-30B-NVFP4"
REVISION = "47818374517751c48c55cde2621594926b1888b6"
MODEL = "glimmer"
CONTEXT = 73728
KV_BYTES = 8 * 1024**3
MIN_FREE_MIB = 76000
FRAME_COUNT = 180
FPS = 60
PATCH_SIZE = 14
PATCH_ALIGNMENT = 28
TEMPLATE_DATE = "2026-11-08"
DIRECT_PREFIX = "<|start|>assistant to=user<|message|>"
MM_KWARGS = {
    "do_resize": False,
    "do_sample_frames": False,
    "fps": FPS,
    "num_frames": FRAME_COUNT,
    "max_video_frame_tokens": 532,
    "max_image_tokens": 532,
}
GENERATION = {
    "n": 1,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 0,
    "repetition_penalty": 1.0,
    "max_tokens": 16384,
    "ignore_eos": False,
}


def engine_settings(weights):
    return {
        "model": str(weights),
        "dtype": "bfloat16",
        "trust_remote_code": False,
        "quantization": "modelopt_mixed",
        "tensor_parallel_size": 1,
        "enforce_eager": True,
        "max_model_len": CONTEXT,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8192,
        "gpu_memory_utilization": 0.76,
        "kv_cache_memory_bytes": KV_BYTES,
        "generation_config": "vllm",
        "worker_extension_cls": "vlm_psychophysics.glimmer.GlimmerWorkerHooks",
        "enable_prefix_caching": False,
        "mm_processor_cache_gb": 0,
        "limit_mm_per_prompt": {"image": 4, "video": 1},
        "mm_processor_kwargs": MM_KWARGS,
    }


def preserve_video_metadata():
    from transformers.video_utils import VideoMetadata
    from vllm.model_executor.models.muse_glimmer import (
        MuseGlimmerMultiModalProcessor, MuseGlimmerProcessingInfo,
    )

    cls = MuseGlimmerMultiModalProcessor
    original = cls._get_hf_mm_inputs
    had_override = "_get_hf_mm_inputs" in cls.__dict__

    def with_metadata(processor, items, kwargs):
        packed = original(processor, items, kwargs)
        video = items.get("video")
        if video and all(metadata is not None for metadata in video.metadata):
            packed.hf_data["video_metadata"] = [
                VideoMetadata(
                    total_num_frames=meta["total_num_frames"],
                    fps=meta["fps"],
                    duration=meta["duration"],
                    frames_indices=meta["frames_indices"],
                )
                for meta in video.metadata
            ]
        return packed

    # vLLM also generates replacement token blocks from processor attributes.
    # A per-request fps option alone leaves that path at the checkpoint's 2 fps.
    original_get_processor = MuseGlimmerProcessingInfo.get_hf_processor

    def configured_processor(info, **kwargs):
        processor = original_get_processor(info, **kwargs)
        processor.video_processor.fps = FPS
        processor.video_processor.num_frames = FRAME_COUNT
        processor.video_processor.max_video_frame_tokens = MM_KWARGS["max_video_frame_tokens"]
        processor.image_processor.max_image_tokens = MM_KWARGS["max_image_tokens"]
        return processor

    cls._get_hf_mm_inputs = with_metadata
    MuseGlimmerProcessingInfo.get_hf_processor = configured_processor

    def restore():
        MuseGlimmerProcessingInfo.get_hf_processor = original_get_processor
        if had_override:
            cls._get_hf_mm_inputs = original
        else:
            delattr(cls, "_get_hf_mm_inputs")

    return restore


def prepare(case, tokenizer):
    from PIL import Image
    import numpy as np
    from vllm.multimodal.media.image import ImageMediaIO
    from vllm.multimodal.media.video import VideoMediaIO

    padding, media, content = [], {}, []
    if case.get("images"):
        images, _ = inputs.load_images(inputs.case_item(case))
        media["image"] = []
        for image in images:
            padded, geometry = inputs.pad(np.asarray(image), PATCH_ALIGNMENT)
            media["image"].append(Image.fromarray(padded))
            padding.append({"kind": "image", **geometry})
            content.append({"type": "image"})
    if case["modality"] == "video":
        path = Path(case["video_path"])
        inputs.require(data.file_sha(path) == case["video_sha256"], "Video SHA changed")
        frames, metadata = VideoMediaIO(
            ImageMediaIO(), video_backend="opencv", num_frames=FRAME_COUNT, fps=-1
        ).load_file(path).media
        inputs.require(
            frames.dtype == np.uint8
            and frames.shape == (FRAME_COUNT, 512, 768, 3)
            and metadata["frames_indices"] == list(range(FRAME_COUNT))
            and abs(float(metadata["fps"]) - FPS) < 1e-6,
            "Video reader changed frame count, order, FPS or geometry",
        )
        padded, geometry = inputs.pad(frames, PATCH_ALIGNMENT)
        media["video"] = [(padded, metadata)]
        padding.append({"kind": "video", **geometry})
        content.append({"type": "video"})
    content.append({"type": "text", "text": case["prompt"]})
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        current_date=TEMPLATE_DATE,
        reasoning_strength="low",
    )
    inputs.require(
        text.endswith("<|start|>assistant")
        and text.count(case["prompt"]) == 1,
        "Native template or exact task prompt changed",
    )
    return {
        "prompt": text.removesuffix("<|start|>assistant") + DIRECT_PREFIX,
        "multi_modal_data": media,
        "mm_processor_kwargs": MM_KWARGS,
    }, padding


def prepared_grid(padding):
    return [
        [
            1 if row["kind"] == "image" else FRAME_COUNT // 2,
            row["submitted_wh"][1] // PATCH_SIZE,
            row["submitted_wh"][0] // PATCH_SIZE,
        ]
        for row in padding
    ]


def payload_grids(payload):
    return [
        item[f"{modality}_grid_thw"]["values"]
        for modality in ("image", "video")
        for item in payload.get(modality, [])
    ]


def payload_pixels(payload):
    return [
        item["pixel_values" if modality == "image" else "pixel_values_videos"]
        for modality in ("image", "video")
        for item in payload.get(modality, [])
    ]


def check_native(artifact, padding):
    payload = artifact["native_payload"]
    grids = payload_grids(payload)
    pixels = payload_pixels(payload)
    expected = prepared_grid(padding)
    inputs.require(grids == expected, f"Native vision grids changed: {grids} != {expected}")
    inputs.require(
        [row["shape"] for row in pixels]
        == [[t * h * w, 2 * 3 * PATCH_SIZE**2] for t, h, w in grids],
        "Native image or video patch rows changed",
    )
    modalities = artifact["mm_placeholders"]
    placements = [
        row for kind in ("image", "video") for row in modalities.get(kind, [])
    ]
    inputs.require(
        len(placements) == len(expected)
        and [len(modalities.get(kind, [])) for kind in ("image", "video")]
        == [sum(p["kind"] == kind for p in padding) for kind in ("image", "video")]
        and all(
            a["offset"] + a["length"] <= b["offset"]
            for a, b in zip(placements, placements[1:])
        ),
        "Native image/video placeholder order or count changed",
    )


def install_vision_probe(worker):
    import torch

    network = worker.get_model()
    samples = []

    def capture(_module, args, kwargs):
        pixels = args[0] if args else kwargs["pixel_values"]
        grid = args[1] if len(args) > 1 else kwargs["grid_thw"]
        inputs.require(isinstance(pixels, torch.Tensor) and pixels.is_cuda, "Vision input is not CUDA")
        grids = grid.tolist()
        sizes = grid.prod(-1).tolist()
        for row, values in zip(grids, pixels.split(sizes), strict=True):
            samples.append({
                "grid": row,
                "pixels": observers.tensor_description(values, deep=True),
            })

    worker._glimmer_vision_samples = samples
    worker._glimmer_vision_hook = network.vision_encoder.register_forward_pre_hook(
        capture, with_kwargs=True
    )


def read_vision_probe(worker):
    worker._glimmer_vision_hook.remove()
    samples = worker._glimmer_vision_samples
    del worker._glimmer_vision_hook, worker._glimmer_vision_samples
    return samples


class GlimmerWorkerHooks:
    """Read-only observations callable on the local vLLM worker."""

    def glimmer_runtime(self):
        receipt = observers.worker_identity(self, MODEL)
        receipt["torch_version"] = str(receipt["torch_version"])
        return receipt

    def glimmer_install_probe(self):
        install_vision_probe(self)

    def glimmer_read_probe(self):
        return read_vision_probe(self)


class Session:
    def __init__(self, weights, data_root, out, *, cpu_only=False):
        import importlib.metadata
        from transformers import AutoTokenizer
        from vllm.config import ModelConfig
        from vllm.multimodal import MULTIMODAL_REGISTRY

        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.settings = engine_settings(weights)
        self.restore_video_metadata = preserve_video_metadata()
        self.tokenizer = AutoTokenizer.from_pretrained(weights, local_files_only=True)
        config = ModelConfig(
            model=str(weights), dtype="bfloat16", quantization="modelopt_mixed",
            trust_remote_code=False, max_model_len=CONTEXT,
            limit_mm_per_prompt=self.settings["limit_mm_per_prompt"],
            mm_processor_cache_gb=0, mm_processor_kwargs=MM_KWARGS,
            allowed_local_media_path=str(data_root),
        )
        self.processor = MULTIMODAL_REGISTRY.create_processor(config, tokenizer=self.tokenizer)
        self.profile = {
            "model": MODEL_ID,
            "revision": REVISION,
            "engine_kwargs": self.settings,
            "generation": GENERATION,
            "assistant_prefix": DIRECT_PREFIX,
            "reasoning_strength": "low",
            "current_date": TEMPLATE_DATE,
            "source_geometry": [768, 512, FRAME_COUNT, FPS],
            "padding_alignment": PATCH_ALIGNMENT,
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("torch", "vllm", "transformers", "numpy", "pillow")
            },
            "processor_code": {
                "glimmer": data.file_sha(Path(__file__)),
                "native": data.file_sha(Path(inspect.getfile(type(self.processor)))),
            },
        }
        self.profile_sha256 = models.checksum({
            **self.profile,
            "engine_kwargs": {**self.settings, "model": MODEL_ID},
        })
        models.write_json(self.out / "resolved-model.json", {
            **self.profile, "profile_sha256": self.profile_sha256,
            "process": models.process_identity(),
        })
        self.references = {}
        self.sampling = {}
        self.engine = None
        if not cpu_only:
            from vllm import LLM

            self.engine = LLM(**self.settings)
            actual = self.engine.collective_rpc("glimmer_runtime")
            inputs.require(len(actual) == 1, "Expected one GPU worker")
            inputs.require(
                actual[0]["actual_kv_capacity"] >= CONTEXT
                and any("fp4" in key.lower() for key in actual[0]["method_kernel_counts"]),
                "Actual context capacity or NVFP4 kernel differs",
            )
            models.write_json(self.out / "runtime.json", actual)

    def cpu_reference(self, request, case):
        from vllm.multimodal.processing.context import TimingContext
        from vllm.multimodal.processing.inputs import ProcessorInputs

        with inputs.NoSpatialTransform():
            expanded = self.processor.apply(
                ProcessorInputs(
                    prompt=self.tokenizer.encode(request["prompt"], add_special_tokens=False),
                    mm_data_items=self.processor.info.parse_mm_data(request["multi_modal_data"]),
                    hf_processor_mm_kwargs=MM_KWARGS,
                ),
                TimingContext(enabled=False),
            )
        ids = list(expanded["prompt_token_ids"])
        inputs.require(
            len(ids) + GENERATION["max_tokens"] <= CONTEXT
            and ids[-len(self.tokenizer.encode(DIRECT_PREFIX, add_special_tokens=False)):]
            == self.tokenizer.encode(DIRECT_PREFIX, add_special_tokens=False),
            "Expanded prompt length or direct-answer suffix changed",
        )
        if case["modality"] == "video":
            from collections import Counter

            video_config = self.processor.info.get_hf_config()
            counts = Counter(ids)
            inputs.require(
                counts[video_config.video_token_id] == 47880,
                f"Expected 47,880 video tokens, got {counts[video_config.video_token_id]}",
            )
            timing_ids = [
                token for token in ids if token != video_config.video_token_id
            ]
            text = self.tokenizer.decode(timing_ids, skip_special_tokens=False)
            times = re.findall(r"Time: ([0-9]+\.[0-9])s", text)
            expected_times = [f"{index / FPS:.1f}" for index in range(0, FRAME_COUNT, 2)]
            inputs.require(
                times == expected_times,
                f"Native Glimmer timestamps differ: {times[:5]}…{times[-5:]}",
            )
        return {
            "prompt_token_ids": ids,
            "mm_placeholders": inputs.placeholders(expanded["mm_placeholders"]),
            "native_payload": inputs.payload_description(expanded["mm_kwargs"], MODEL),
        }

    def run(self, case, *, force_deep=False):
        started = time.monotonic()
        key = models.checksum([
            case["prompt"], case["modality"], case["choices"],
            [data.image_size(item) for item in case.get("images", [])],
        ])
        deep = force_deep or key not in self.references
        observer = observers.InputObserver(MODEL, type(self.processor), case, deep)
        with observer if self.engine else nullcontext():
            request, padding = prepare(case, self.tokenizer)
            if deep:
                reference = self.cpu_reference(request, case)
                check_native(reference, padding)
                self.references[key] = reference
                models.append_json(self.out / "input-references.jsonl", {
                    "sample_id": case["sample_id"], "reference_key": key, **reference,
                    "template_sha256": inputs.sha(request["prompt"].encode("utf-8")),
                })
                observer.applications.clear()
            reference = self.references[key]
            if not self.engine:
                return {
                    "sample_id": case["sample_id"],
                    "modality": case["modality"],
                    "input_grid": prepared_grid(padding),
                    "prompt_tokens": len(reference["prompt_token_ids"]),
                    "prompt_ids_sha256": models.checksum(reference["prompt_token_ids"]),
                }
            if deep:
                self.engine.collective_rpc("glimmer_install_probe")
            try:
                answer = self.engine.generate(request, self.sampling_for(case["choices"]), use_tqdm=False)[0]
            finally:
                ingress = self.engine.collective_rpc("glimmer_read_probe")[0] if deep else None
        result = answer.outputs[0]
        receipt = {
            "sample_id": case["sample_id"],
            "case_id": case["case_id"],
            "task": case["task"],
            "domain": case["domain"],
            "modality": case["modality"],
            "model_input_sha256": case["model_input_sha256"],
            "prompt_sha256": case["prompt_sha256"],
            "profile_sha256": self.profile_sha256,
            "raw_answer": result.text,
            "output_token_ids": list(result.token_ids),
            "prompt_ids_sha256": models.checksum(list(answer.prompt_token_ids)),
            "prompt_token_count": len(answer.prompt_token_ids),
            "finish_reason": result.finish_reason,
            "stop_reason": result.stop_reason,
            "seconds": time.monotonic() - started,
            "input_grid": prepared_grid(padding),
            "padding": padding,
            "deep_observation": deep,
        }
        try:
            inputs.require(result.finish_reason == "stop", "Native generation did not stop")
            inputs.require(
                list(answer.prompt_token_ids) == reference["prompt_token_ids"]
                and len(observer.applications) == 1,
                "Observed GPU request prompt expansion or application count changed",
            )
            actual = observer.applications[0]
            check_native(actual, padding)
            inputs.require(
                actual["mm_placeholders"] == reference["mm_placeholders"]
                and actual["prompt_token_ids"] == reference["prompt_token_ids"],
                "Native placement or prompt IDs changed during generation",
            )
            if deep:
                pixels = payload_pixels(reference["native_payload"])
                cpu = payload_pixels(actual["native_payload"])
                inputs.require(
                    [row["grid"] for row in ingress] == prepared_grid(padding),
                    f"Actual GPU vision grids differ: {[row['grid'] for row in ingress]}",
                )
                signatures = lambda rows: [(row["shape"], row["dtype"], row["sha256"]) for row in rows]
                inputs.require(
                    signatures(pixels) == signatures(cpu) == signatures([x["pixels"] for x in ingress]),
                    "Actual GPU vision pixels differ from CPU reference",
                )
        except Exception as error:
            receipt["technical_error"] = str(error)
        receipt["technical_pass"] = "technical_error" not in receipt
        if deep or not receipt["technical_pass"]:
            models.append_json(self.out / "input-observations.jsonl", {
                "sample_id": case["sample_id"], "reference_key": key,
                "reader": observer.reads, "GPU": ingress, "receipt": receipt,
            })
        models.append_json(self.out / "returned.jsonl", receipt)
        inputs.require(receipt["technical_pass"], receipt.get("technical_error", ""))
        return receipt

    def sampling_for(self, choices):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        key = tuple(choices)
        if key not in self.sampling:
            self.sampling[key] = SamplingParams(
                **GENERATION, structured_outputs=StructuredOutputsParams(choice=list(choices))
            )
        return self.sampling[key]

    def close(self):
        try:
            if self.engine is not None:
                self.engine.llm_engine.engine_core.shutdown()
        finally:
            self.restore_video_metadata()
