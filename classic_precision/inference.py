"""Finite four-model inference using the input paths qualified in v32.

Imports do not load Torch, create a CUDA context, download a checkpoint, or call a
model. Runtime paths are explicit. Source RGB16 is checked per video; expensive
processor inversion is performed once per loaded processor geometry, while every
call retains source hashes, frame/shape guards and real GPU-forward receipts.
"""
from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import inspect
import json
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path

from .storage import MAX_ATTEMPTS, PersistenceError, RunStore, atomic_json, digest, file_sha256


FAMILIES = ("nemo", "qwen", "cosmos", "gemma")
GENERATION = {"native_thinking": False, "output_choice": ["LEFT", "RIGHT"],
              "seed": 0, "max_output_tokens": 16384, "context_limit": 65536,
              "temperature": 0.0, "top_p": 1.0, "repetition_penalty": 1.0,
              "do_sample": False, "num_beams": 1}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _versions():
    names = ("torch", "transformers", "numpy", "av", "vllm", "accelerate", "safetensors")
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _source_identity(value):
    path = inspect.getsourcefile(value)
    return {"module": value.__module__, "qualname": value.__qualname__,
            "source_sha256": file_sha256(path) if path else None}


def checkpoint_asset_identity(path):
    """Bind processor/tokenizer/config/code bytes without rehashing all weights."""
    path = Path(path)
    names = {"config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
             "video_preprocessor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
             "chat_template.jinja", "special_tokens_map.json", "model.safetensors.index.json"}
    names.update(p.name for p in path.glob("*.py"))
    assets = {name: {"sha256": file_sha256(path / name), "bytes": (path / name).stat().st_size}
              for name in sorted(names) if (path / name).is_file()}
    return {"processor_tokenizer_config_and_code": assets,
            "weight_shard_bytes": {p.name: p.stat().st_size for p in sorted(path.glob("*.safetensors"))}}


def _cosmos_package_identity(package):
    package = package or os.environ.get("COSMOS_TRANSFORMERS_PACKAGE")
    _require(bool(package), "Cosmos requires the qualified official transformers-cosmos3 package directory")
    paths = sorted((Path(package) / "transformers_cosmos3").glob("*.py"))
    _require(bool(paths), "Official Cosmos package Python sources are missing")
    return {p.name: file_sha256(p) for p in paths}


def _inside(root, relative):
    relative = Path(relative)
    _require(not relative.is_absolute() and ".." not in relative.parts, "Asset paths must stay relative to asset_root")
    root = Path(root).resolve()
    path = (root / relative).resolve()
    _require(path.is_relative_to(root), "Asset path escapes asset_root")
    return path


def load_run_inputs(manifest_path, ready_path=None, asset_root=None):
    """Bind immutable scientific rows to lossless, fully qualified ready videos."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    from .manifest import validate_manifest
    validate_manifest(manifest)
    ready_path = Path(ready_path) if ready_path else manifest_path.parent / "ready.jsonl"
    source = ready_path.read_text()
    if source.lstrip().startswith("["):
        ready_records = json.loads(source)
    else:
        ready_records = [json.loads(line) for line in source.splitlines() if line.strip()]
    rows = manifest["rows"]
    _require(len(rows) == 3504, "Production manifest must contain exactly 3504 rows per model")
    _require(set(manifest["models"]) == set(FAMILIES), "All four fixed model specifications are required")
    ids = [row["sample_id"] for row in rows]
    _require(len(set(ids)) == len(ids), "Duplicate manifest sample_id")
    ready = {row["sample_id"]: row for row in ready_records}
    _require(len(ready) == len(ready_records), "Duplicate ready sample_id")
    _require(set(ready) == set(ids), "Ready ledger must exactly cover the frozen 3504 rows")
    for row in rows:
        receipt = ready[row["sample_id"]]
        _require(receipt["source_parameters"] == row["source_parameters"], "Ready source parameters differ from manifest")
        _require(row["truth"] in ("LEFT", "RIGHT"), "Unknown truth label")
        for name in ("video_sha256", "stream_sha256"):
            _require(bool(re.fullmatch(r"[0-9a-f]{64}", receipt[name])), "Invalid ready SHA256")
        path = _inside(asset_root or manifest_path.parent, receipt["video_relative"])
        _require(path.is_file(), "Ready video is missing before any model load: " + receipt["video_relative"])
    for name, expected in GENERATION.items():
        if name in manifest.get("generation", {}):
            _require(manifest["generation"][name] == expected, "Manifest generation differs from the fixed protocol: " + name)
    generation = manifest["generation"]
    _require(generation.get("choice", generation.get("output_choice")) == ["LEFT", "RIGHT"], "Exact LEFT/RIGHT choices are required")
    _require(generation.get("max_context_tokens", generation.get("context_limit")) == 65536, "Context limit must be 65536")
    input_spec = manifest["input"]
    for key, expected in {"source_wh": [768, 512], "frames": 180, "fps": 60, "master_bits": 16,
                          "do_resize": False, "do_sample_frames": False}.items():
        _require(input_spec.get(key) == expected, "Manifest video input differs from the fixed protocol: " + key)
    return manifest, rows, ready, Path(asset_root or manifest_path.parent), ready_path


def select_rows(rows, row_ids=None, shard_index=0, shard_count=1):
    _require(shard_count >= 1 and 0 <= shard_index < shard_count, "Invalid finite shard")
    lookup = {row["sample_id"]: row for row in rows}
    if row_ids is not None:
        requested = list(row_ids)
        _require(len(set(requested)) == len(requested) and set(requested) <= set(lookup), "Invalid explicit row IDs")
        _require(shard_count == 1 and shard_index == 0, "Explicit row IDs and automatic sharding are mutually exclusive")
        selected = set(requested)
        return [row for row in rows if row["sample_id"] in selected]
    if shard_count == 1:
        return list(rows)
    def onset_index(row):
        params = row["source_parameters"]
        index = params.get("onset_index", (params["onset_frame"] - 18) // 5)
        _require(params["onset_frame"] == 18 + 5 * index and 0 <= index < 24, "Unknown onset block")
        return index
    return [row for row in rows if onset_index(row) % shard_count == shard_index]


def qualification_order(rows):
    """Use the first predetermined T2 and T3 without selecting by correctness."""
    first = [next((row for row in rows if row["task"] == task), None) for task in ("T2", "T3")]
    first = [row for row in first if row is not None]
    ids = {row["sample_id"] for row in first}
    return first + [row for row in rows if row["sample_id"] not in ids], first


def available_ram_bytes():
    meminfo = Path("/proc/meminfo")
    _require(meminfo.exists(), "GPU runner requires Linux MemAvailable accounting")
    fields = {parts[0].rstrip(":"): int(parts[1]) * 1024
              for line in meminfo.read_text().splitlines() if len(parts := line.split()) >= 2}
    return fields["MemAvailable"]


def resource_snapshot(torch, resources):
    _require("min_free_gpu_gib" in resources and "min_available_ram_gib" in resources,
             "Explicit model resource budgets are required")
    free, total = torch.cuda.mem_get_info()
    ram = available_ram_bytes()
    for key in ("min_free_gpu_gib", "min_available_ram_gib"):
        _require(float(resources[key]) > 0, "Resource budgets must be positive")
    return {"free_gpu_gib": free / 2**30, "total_gpu_gib": total / 2**30,
            "available_ram_gib": ram / 2**30, "budgets": resources,
            "admitted": free >= float(resources["min_free_gpu_gib"]) * 2**30
                        and ram >= float(resources["min_available_ram_gib"]) * 2**30}


@contextlib.contextmanager
def _gpu_lock(output_dir, torch):
    """Cooperate with this runner; never kill or reject an unrelated process."""
    import fcntl
    properties = torch.cuda.get_device_properties(0)
    identity = str(getattr(properties, "uuid", None) or os.environ["CUDA_VISIBLE_DEVICES"])
    path = Path(output_dir) / (".gpu-" + digest({"host": socket.gethostname(), "device": identity}) + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
        fcntl.flock(stream, fcntl.LOCK_UN)


def decode_rgb16(asset_root, ready, family):
    import av
    import hashlib
    import numpy as np
    path = _inside(asset_root, ready["video_relative"])
    _require(file_sha256(path) == ready["video_sha256"], "Encoded video SHA256 differs from ready receipt")
    codes, timestamps = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        _require(stream.codec_context.name == "ffv1" and stream.codec_context.format.name == "gbrp16le",
                 "Expected FFV1 gbrp16le scientific master")
        for frame in container.decode(video=0):
            codes.append(frame.to_ndarray(format="rgb48le"))
            timestamps.append(float(frame.pts * frame.time_base))
    codes = np.stack(codes)
    _require(codes.dtype == np.uint16 and codes.shape == (180, 512, 768, 3), "Unexpected RGB16 video geometry")
    _require(hashlib.sha256(codes.tobytes()).hexdigest() == ready["stream_sha256"], "Decoded RGB stream differs from ready receipt")
    _require(max(abs(t - i / 60) for i, t in enumerate(timestamps)) <= .001, "Unexpected source frame timestamps")
    padding = 16 if family == "gemma" else 0
    if padding:
        bg = codes[0, 0, 0]
        padded = np.empty((180, 528, 768, 3), dtype=np.uint16)
        padded[:, :512] = codes
        padded[:, 512:] = bg
        _require(np.array_equal(padded[:, :512], codes), "Gemma padding changed original pixels")
        codes = padded
    frames = codes.astype(np.float32) * (255. / 65535.)
    return codes, frames, {"source_video_sha256": ready["video_sha256"],
        "source_stream_sha256": ready["stream_sha256"], "source_shape": [180, 512, 768, 3],
        "submitted_shape": list(frames.shape), "decoded_dtype": "uint16", "submitted_dtype": "float32",
        "master_bits_per_channel": 16, "padding_bottom_rows": padding, "resize": False,
        "selected_source_indices": list(range(180)), "source_pts_s": timestamps}


def _video_metadata():
    return {"fps": 60., "frames_indices": list(range(180)), "total_num_frames": 180,
            "duration": 3., "video_backend": "pyav_explicit_RGB48"}


def restore_processor_rgb(pixels, processor, source_codes, family):
    """Undo official patch packing/normalization into FP32 sRGB in [0,1].

    BF16 tensors are accepted without rounding/clipping their reconstruction.
    This is a measurement helper, never substituted for the official processor.
    Gemma includes its explicit 16 bottom rows; callers crop to source height512.
    """
    import torch
    if family == "nemo":
        return (pixels.float() * processor.norm_std + processor.norm_mean).permute(0, 2, 3, 1)
    h, w = source_codes.shape[1:3]
    vp = processor.video_processor
    p = vp.patch_size
    if family == "gemma":
        z = pixels[0, :, :(h // p) * (w // p)].float().reshape(180, h // p, w // p, p, p, 3)
        z = z.permute(0, 5, 1, 3, 2, 4).reshape(180, 3, h, w)
    else:
        m, t = vp.merge_size, vp.temporal_patch_size
        z = pixels.float().reshape(180 // t, h // p // m, w // p // m, m, m, 3, t, p, p)
        z = z.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(180, 3, h, w)
    if vp.do_normalize:
        z = z * torch.tensor(vp.image_std).reshape(1, 3, 1, 1) + torch.tensor(vp.image_mean).reshape(1, 3, 1, 1)
    if vp.do_rescale:
        z = z / vp.rescale_factor
    return z.permute(0, 2, 3, 1) / 255.


def _recover_codes(pixels, processor, codes, family, torch):
    restored = torch.round(restore_processor_rgb(pixels, processor, codes, family) * 65535.).clamp(0, 65535).to(torch.int32)
    error = int((restored - torch.from_numpy(codes.astype("int32"))).abs().max())
    _require(error == 0, f"Processor changed source precision/geometry: {error} code values")
    return {"pre_BF16_recovered_uint16_max_code_error": error, "uint8_intermediate_detected": False,
            "verified_frames": 180}


def _nemo_geometry(config):
    vision = config.vision_config
    values = {"video_target_num_patches": 1536, "video_maintain_aspect_ratio": True}
    if isinstance(vision, dict):
        vision.update(values)
    else:
        for key, value in values.items():
            setattr(vision, key, value)
    return config


def _install_vision_probe(worker):
    model = worker.get_model()
    target = getattr(model, "vision_model", None)
    if target is None:
        target = model.visual
    worker._classic_precision_receipts = []
    def hook(module, args, kwargs):
        x = args[0]
        grid = kwargs.get("grid_thw")
        worker._classic_precision_receipts.append({"shape": list(x.shape), "dtype": str(x.dtype),
            "device": str(x.device), "grid_thw": grid.tolist() if grid is not None else None,
            "num_frames_argument": kwargs.get("num_frames")})
    worker._classic_precision_hook = target.register_forward_pre_hook(hook, with_kwargs=True)
    return True


def _read_vision_probe(worker):
    worker._classic_precision_hook.remove()
    return worker._classic_precision_receipts


def _weight_receipt(worker):
    counts = defaultdict(int)
    for parameter in worker.get_model().parameters():
        counts[str(parameter.dtype)] += parameter.numel() * parameter.element_size()
    return dict(counts)


def _kv_capacity(worker):
    """Read actual initialized KV blocks, never infer capacity from the budget."""
    cache = getattr(worker, "cache_config", None)
    runner = getattr(worker, "model_runner", None)
    config = getattr(runner, "kv_cache_config", None)
    blocks = getattr(config, "num_blocks", None) or getattr(cache, "num_gpu_blocks", None)
    groups = getattr(config, "kv_cache_groups", None)
    block_sizes = []
    for group in groups or []:
        spec = getattr(group, "kv_cache_spec", None)
        value = getattr(spec, "block_size", None)
        if value is not None:
            block_sizes.append(int(value))
    if not block_sizes and cache is not None and getattr(cache, "block_size", None):
        block_sizes.append(int(cache.block_size))
    return {"num_gpu_blocks": int(blocks) if blocks is not None else None,
            "block_sizes": block_sizes,
            "capacity_tokens": int(blocks) * min(block_sizes) if blocks and block_sizes else None}


def _validate_forwards(family, receipts):
    _require(bool(receipts), "No real GPU-forward receipt")
    _require(all(r["dtype"] == "torch.bfloat16" and r["device"].startswith("cuda") for r in receipts),
             "Unexpected actual GPU activation dtype/device")
    if family == "nemo":
        _require(all(r["shape"][1:] == [3, 512, 768] and r["num_frames_argument"] == r["shape"][0]
                     for r in receipts), "Nemo native chunks changed geometry")
        _require(sum(r["shape"][0] for r in receipts) == 180, "Nemo did not forward all 180 source frames")
    elif family in ("qwen", "cosmos"):
        _require(len(receipts) == 1 and receipts[0]["shape"] == [138240, 1536]
                 and receipts[0].get("grid_thw") == [[90, 32, 48]], "Qwen/Cosmos native temporal-patch input changed")
    else:
        _require(len(receipts) == 1 and receipts[0]["shape"] == [1, 180, 2520, 768], "Gemma actual GPU video input changed")


class ModelSession:
    """One accepted loader and official processor, reused for a finite shard."""

    def __init__(self, family, model_path, store, cosmos_package=None, resources=None):
        self._restore = []
        self.context = {"active": False, "codes": None}
        try:
            self._initialize(family, model_path, store, cosmos_package, resources)
        except BaseException:
            self.close()
            raise

    def _initialize(self, family, model_path, store, cosmos_package, resources):
        self.resources = resources or {}
        self._initialize_processor(family, model_path, store, cosmos_package)
        self._owns_cuda = True
        self.torch.cuda.manual_seed_all(0)
        began = time.monotonic()
        if family in ("nemo", "qwen"):
            self._load_vllm()
        else:
            self._load_hf()
        self.store.event("MODEL_READY", load_seconds=time.monotonic() - began,
                         gpu_allocated_gib=self.torch.cuda.memory_allocated() / 2**30,
                         gpu_reserved_gib=self.torch.cuda.memory_reserved() / 2**30)

    def _initialize_processor(self, family, model_path, store, cosmos_package):
        import torch
        from transformers import AutoConfig, AutoProcessor, AutoTokenizer
        self.torch, self.family, self.store = torch, family, store
        self.model_path = Path(model_path)
        self._owns_cuda = False
        self._restore = []
        self.context = {"active": False, "deep": True, "norm_receipt": None, "engine_receipt": None}
        torch.set_num_threads(6)
        torch.set_grad_enabled(False)
        # CPU preflight must not even queue a CUDA seed callback.
        torch.default_generator.manual_seed(0)
        if family == "cosmos":
            package = cosmos_package or os.environ.get("COSMOS_TRANSFORMERS_PACKAGE")
            _require(bool(package), "Cosmos requires the qualified official transformers-cosmos3 package directory")
            package = Path(package).resolve()
            _require(package.is_dir(), "Cosmos package directory missing")
            sys.path.insert(0, str(package))
            import transformers_cosmos3  # noqa: F401: official registration
            _require(Path(transformers_cosmos3.__file__).resolve().is_relative_to(package),
                     "An already imported Cosmos shim came from a different package directory")
        if family == "nemo":
            from vllm.transformers_utils.processors import nano_nemotron_vl as native
            self.native = native
            config = _nemo_geometry(AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True))
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
            self.processor = native.NanoNemotronVLProcessor(config, self.tokenizer, max_model_len=65536, video_pruning_rate=0.)
            self._guard_nemo_normalizer()
        else:
            self.processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
            self.tokenizer = self.processor.tokenizer
        self.store.event("PROCESSOR_IDENTITY", processor=_source_identity(type(self.processor)),
                         tokenizer=_source_identity(type(self.tokenizer)), versions=_versions())

    def _patch(self, target, name, replacement):
        self._restore.append((target, name, getattr(target, name)))
        setattr(target, name, replacement)

    def _guard_nemo_normalizer(self):
        native, torch = self.native, self.torch
        original = native._bicubic_resize_and_normalize
        signature = inspect.signature(original)
        def checked(*args, **kwargs):
            out = original(*args, **kwargs)
            if self.context["active"]:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                values = dict(bound.arguments)
                image = next(v for v in values.values() if isinstance(v, torch.Tensor) and v.ndim == 4)
                _require(image.dtype == torch.float32 and values["size"] is None,
                         "Nemo input was cast or resized before its official normalizer")
                receipt = {"normalizer_input_dtype": str(image.dtype), "resize_argument": None,
                           "verified_frames": int(image.shape[0])}
                if self.context["deep"]:
                    values["dtype"] = torch.float32
                    pre = original(**values)
                    restored = (pre * self.processor.norm_std + self.processor.norm_mean).permute(0, 2, 3, 1)
                    recovered = torch.round(restored * 65535).clamp(0, 65535).to(torch.int32)
                    expected = torch.from_numpy(self.context["codes"].astype("int32"))
                    error = int((recovered - expected).abs().max())
                    difference = float((pre.to(out.dtype) - out).abs().max())
                    _require(error == 0 and difference == 0, "Nemo official normalization changed source codes")
                    receipt.update(pre_BF16_recovered_uint16_max_code_error=error,
                                   actual_BF16_vs_direct_normalization_max_abs_difference=difference)
                self.context["norm_receipt"] = receipt
            return out
        self._patch(native, "_bicubic_resize_and_normalize", checked)

    def _guard_pixels(self, data, processor, engine=False):
        family = self.family
        if family == "nemo":
            pixels = data["pixel_values_flat_video"]
            _require(list(pixels.shape) == [180, 3, 512, 768] and str(pixels.dtype) == "torch.bfloat16",
                     "Nemo processor shape/dtype changed")
            _require(data["frames_indices"].flatten().tolist() == list(range(180)), "Nemo sampled source frames")
            _require(self.context["norm_receipt"] is not None, "Nemo normalizer guard did not execute")
            receipt = dict(self.context["norm_receipt"])
        else:
            pixels = data["pixel_values_videos"]
            expected = [1, 180, 2520, 768] if family == "gemma" else [138240, 1536]
            _require(list(pixels.shape) == expected and str(pixels.dtype) == "torch.float32", "Official processor shape/dtype changed")
            if family != "gemma":
                _require(data["video_grid_thw"].tolist() == [[90, 32, 48]], "Official video grid changed")
            receipt = _recover_codes(pixels, processor, self.context["codes"], family, self.torch) if self.context["deep"] else {}
        receipt.update(actual_processor_tensor_shape=list(pixels.shape), actual_processor_dtype=str(pixels.dtype),
                       precision_inversion="performed" if self.context["deep"] else "reused_geometry_qualification")
        if engine:
            self.context["engine_receipt"] = receipt
        return receipt

    def _load_vllm(self):
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        family = self.family
        free, total = self.torch.cuda.mem_get_info()
        reserve = float(self.resources.get("gpu_reserve_gib", 2)) * 2**30
        utilization = min(.78, (free - reserve) / total)
        _require(0 < utilization < 1, "Insufficient free GPU memory for the explicit reserve")
        settings = dict(model=str(self.model_path), trust_remote_code=True, tensor_parallel_size=1,
            max_model_len=65536, max_num_seqs=1,
            max_num_batched_tokens=4096 if family == "nemo" else 49152,
            gpu_memory_utilization=utilization, enforce_eager=True, enable_prefix_caching=False,
            enable_chunked_prefill=True, kv_cache_dtype="bfloat16",
            kv_cache_memory_bytes=(2 if family == "nemo" else 4) * 2**30,
            limit_mm_per_prompt={"video": 1, "image": 0, "audio": 0}, mm_processor_cache_gb=0,
            disable_custom_all_reduce=True, seed=0)
        if family == "nemo":
            settings.update(hf_overrides=_nemo_geometry, video_pruning_rate=0.)
        else:
            settings.update(quantization="modelopt", compilation_config={"mode": 0},
                engram_config={"cpu_offload": True}, cpu_offload_gb=24, cpu_offload_params={"experts"},
                mm_processor_kwargs={"do_resize": False, "do_sample_frames": False})
        self.engine = LLM(**settings)
        capacity = self.engine.collective_rpc(_kv_capacity)
        _require(capacity and all(c["capacity_tokens"] is not None and c["capacity_tokens"] >= 65536 for c in capacity),
                 "Actual initialized KV capacity could not be confirmed >=65536; stop before generation")
        public_settings = {k: (sorted(v) if isinstance(v, set) else v) for k, v in settings.items() if k != "model"}
        if family == "nemo":
            public_settings["hf_overrides"] = _source_identity(_nemo_geometry)
        self.store.event("MODEL_LOADED", backend="vllm", settings=public_settings, actual_kv_capacity=capacity,
                         weight_storage_bytes_by_dtype=self.engine.collective_rpc(_weight_receipt))
        self.sampling = SamplingParams(n=1, temperature=0., top_p=1., repetition_penalty=1., seed=0,
            max_tokens=16384, structured_outputs=StructuredOutputsParams(choice=["LEFT", "RIGHT"]))
        self.effective_generation = {**GENERATION, "backend": "vllm", "constraint": "StructuredOutputsParams(choice)",
            "actual_temperature": self.sampling.temperature, "actual_top_p": self.sampling.top_p,
            "actual_max_tokens": self.sampling.max_tokens, "actual_seed": self.sampling.seed,
            "actual_repetition_penalty": self.sampling.repetition_penalty,
            "sampling_params_repr": repr(self.sampling)}
        processor_type = self.native.NanoNemotronVLProcessor if family == "nemo" else type(self.processor)
        original = processor_type.__call__
        def checked(processor, *args, **kwargs):
            data = original(processor, *args, **kwargs)
            pixel_name = "pixel_values_flat_video" if family == "nemo" else "pixel_values_videos"
            if self.context["active"] and pixel_name in data:
                self._guard_pixels(data, processor, engine=True)
            return data
        self._patch(processor_type, "__call__", checked)

    def _load_hf(self):
        torch = self.torch
        if self.family == "cosmos":
            from transformers_cosmos3 import Cosmos3ForConditionalGeneration as Model
            from transformers_cosmos3.model import KEY_MAPPING
            from transformers import AutoConfig, Qwen3VLForConditionalGeneration
            keys = json.loads((self.model_path / "model.safetensors.index.json").read_text())["weight_map"]
            exact_mapping = {}
            for key in keys:
                renamed = key
                for pattern, replacement in KEY_MAPPING.items():
                    renamed = re.sub(pattern, replacement, renamed)
                if renamed != key:
                    exact_mapping["^" + re.escape(key) + "$"] = renamed
            config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
            _require(config.text_config.hidden_size == 5120 and config.text_config.num_hidden_layers == 64,
                     "Cosmos official shim configuration changed")
            self.model, info = Qwen3VLForConditionalGeneration.from_pretrained.__func__(Model,
                self.model_path, config=config, key_mapping=exact_mapping, local_files_only=True,
                dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa", output_loading_info=True)
            _require(not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")),
                     "Cosmos weights have missing, mismatched or erroneous tensors")
            _require(not any(p.is_meta for p in self.model.parameters()), "Cosmos contains uninitialized meta weights")
            self.store.event("WEIGHT_MAPPING_PASS", mapping_count=len(exact_mapping),
                official_shim=_source_identity(Model), exact_mapping_sha256=digest(exact_mapping),
                loading_info={k: sorted(v) if isinstance(v, set) else v for k, v in info.items()})
        else:
            from transformers import AutoModelForMultimodalLM as Model
            self.model = Model.from_pretrained(self.model_path, local_files_only=True, dtype=torch.bfloat16,
                device_map={"": "cuda:0"}, attn_implementation="sdpa")
        self.model.eval()
        counts = defaultdict(int)
        for parameter in self.model.parameters():
            counts[str(parameter.dtype)] += parameter.numel() * parameter.element_size()
        self.store.event("MODEL_LOADED", backend="transformers", dtype="bfloat16", attn_implementation="sdpa",
                         weight_storage_bytes_by_dtype=dict(counts), model_class=_source_identity(type(self.model)))
        self.allowed_sequences = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("LEFT", "RIGHT")]
        _require(all(self.allowed_sequences) and len({tuple(s) for s in self.allowed_sequences}) == 2,
                 "Tokenizer cannot represent both exact choices")
        eos = self.model.generation_config.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos or [])
        _require(bool(self.eos), "Native EOS is required for exact HF choice decoding")
        self.effective_generation = {**GENERATION, "backend": "transformers", "constraint": "prefix_allowed_tokens_fn",
            "choice_token_sequences": self.allowed_sequences, "native_eos_token_ids": self.eos,
            "num_return_sequences": 1, "max_time": None,
            "inherited_native_generation_config": json.loads(self.model.generation_config.to_json_string()),
            "cache_implementation": "offloaded" if self.family == "gemma" else "native_default"}

    def prepare(self, row, ready, asset_root):
        torch, family = self.torch, self.family
        codes, frames, receipt = decode_rgb16(asset_root, ready, family)
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": row["prompt"]}]}]
        formatter = self.tokenizer if family == "nemo" else self.processor
        text = formatter.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        _require(not text.rstrip().endswith("<think>"), "Native thinking prefix remained open")
        if family == "gemma":
            _require("<|think|>" not in text and text.endswith("<|channel>thought\n<channel|>"),
                     "Gemma official OFF chat template changed")
        self.context.update(active=True, codes=codes, norm_receipt=None, engine_receipt=None)
        if family == "nemo":
            data = self.processor(text=[text], videos=[(frames, _video_metadata())], return_tensors="pt")
        else:
            from transformers.video_utils import VideoMetadata
            metadata = VideoMetadata(total_num_frames=180, fps=60., duration=3., frames_indices=list(range(180)))
            kwargs = {"do_resize": False, "do_sample_frames": False, "video_metadata": [metadata]}
            if family == "gemma":
                kwargs["max_soft_tokens"] = 280
            data = self.processor(text=[text], videos=[frames], videos_kwargs=kwargs, return_tensors="pt")
        receipt.update(self._guard_pixels(data, self.processor))
        receipt.update(input_tokens=int(data["input_ids"].shape[-1]), native_OFF_chat_kwargs={"enable_thinking": False},
                       chat_template_text=text, chat_template_sha256=digest(text),
                       processor_source=_source_identity(type(self.processor)))
        _require(receipt["input_tokens"] + 16384 <= 65536, "Actual processor input exceeds the context budget")
        return frames, data, text, receipt

    def generate(self, row, ready, asset_root, durable_output):
        """Call durable_output immediately after receiving raw text and token IDs.

        Post-generation input receipts are additional events. Their failure never
        discards a completed answer or makes it eligible for another model call.
        """
        torch, family = self.torch, self.family
        frames, data, text, receipt = self.prepare(row, ready, asset_root)
        began = time.monotonic()
        base = {"task": row["task"], "truth": row["truth"], "row_identity": digest({"row": row, "ready": ready}),
                "source_parameters": row["source_parameters"], "input_receipt": receipt,
                "effective_generation": self.effective_generation}
        if family in ("nemo", "qwen"):
            del data
            self.context["engine_receipt"] = None
            self.context["norm_receipt"] = None
            _require(all(self.engine.collective_rpc(_install_vision_probe)), "GPU probe installation failed")
            answer = self.engine.generate({"prompt": text, "multi_modal_data": {"video": [(frames, _video_metadata())]}},
                                          self.sampling, use_tqdm=False)[0]
            output = answer.outputs[0]
            actual_tokens = len(answer.prompt_token_ids)
            record = {**base, "raw_answer": output.text, "output_token_ids": list(output.token_ids),
                "finish_reason": output.finish_reason, "stop_reason": output.stop_reason,
                "actual_input_tokens": actual_tokens, "seconds": time.monotonic() - began}
            # No assertion, probe RPC or parsing is allowed before durable raw.
            committed = durable_output(record)
            forwards = self.engine.collective_rpc(_read_vision_probe)
            flattened = [item for worker in forwards for item in worker]
            engine_receipt = self.context["engine_receipt"]
            self.store.event("POST_OUTPUT_INPUT_RECEIPT", sample_id=row["sample_id"],
                             attempt_id=committed["attempt_id"], actual_GPU_visual_forward=forwards,
                             actual_engine_processor_receipt=engine_receipt)
            _require(actual_tokens == receipt["input_tokens"] and actual_tokens + 16384 <= 65536,
                     "Actual engine prompt/context budget differs from the checked input")
            _require(engine_receipt is not None, "Actual engine processor guard did not execute")
            _validate_forwards(family, flattened)
            self.store.event("TECHNICAL_PASS", sample_id=row["sample_id"], attempt_id=committed["attempt_id"])
        else:
            del frames
            data = {k: v.to("cuda:0", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda:0") for k, v in data.items()}
            prompt_length = int(data["input_ids"].shape[-1])
            forwards = []
            def before(module, args, kwargs):
                if "pixel_values_videos" in kwargs:
                    x = kwargs["pixel_values_videos"]
                    grid = kwargs.get("video_grid_thw")
                    if not forwards:
                        forwards.append({"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
                                         "grid_thw": grid.tolist() if grid is not None else None})
            hook = self.model.register_forward_pre_hook(before, with_kwargs=True)
            def allowed(batch_id, ids):
                prefix = ids[prompt_length:].tolist()
                permitted = {e for e in self.eos if prefix in self.allowed_sequences}
                for sequence in self.allowed_sequences:
                    if sequence[:len(prefix)] == prefix and len(prefix) < len(sequence):
                        permitted.add(sequence[len(prefix)])
                _require(bool(permitted), "HF prefix left the exact choice grammar")
                return sorted(permitted)
            memory = {"cache_implementation": "offloaded"} if family == "gemma" else {}
            try:
                with torch.inference_mode():
                    result = self.model.generate(**data, max_new_tokens=16384, do_sample=False,
                        num_beams=1, num_return_sequences=1, max_time=None, repetition_penalty=1.,
                        prefix_allowed_tokens_fn=allowed, **memory)
                ids = result[0, prompt_length:].tolist()
                raw = self.tokenizer.decode(ids, skip_special_tokens=True)
                committed = durable_output({**base, "raw_answer": raw, "output_token_ids": ids,
                    "finish_reason": "stop" if ids and ids[-1] in self.eos else "length",
                    "actual_input_tokens": prompt_length, "seconds": time.monotonic() - began})
                self.store.event("POST_OUTPUT_INPUT_RECEIPT", sample_id=row["sample_id"],
                                 attempt_id=committed["attempt_id"], actual_GPU_model_forward=forwards)
                _require(prompt_length + 16384 <= 65536, "Actual HF input exceeds context budget")
                _validate_forwards(family, forwards)
                self.store.event("TECHNICAL_PASS", sample_id=row["sample_id"], attempt_id=committed["attempt_id"])
            finally:
                hook.remove()
        self.context.update(active=False, codes=None, deep=False)
        del text
        gc.collect()

    def close(self):
        for target, name, original in reversed(self._restore):
            setattr(target, name, original)
        self._restore.clear()
        self.context.update(active=False, codes=None)
        engine = getattr(self, "engine", None)
        if engine is not None:
            candidates = [engine, getattr(engine, "llm_engine", None)]
            candidates.append(getattr(candidates[-1], "engine_core", None))
            for candidate in candidates:
                shutdown = getattr(candidate, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                    break
            self.engine = None
        if hasattr(self, "model"):
            self.model = None
        gc.collect()
        torch = getattr(self, "torch", None)
        if getattr(self, "_owns_cuda", False) and torch is not None and torch.cuda.is_initialized():
            torch.cuda.empty_cache()


class _DiscardEvents:
    def event(self, *args, **kwargs):
        pass


class ProcessorSession:
    """CPU-only official input processing; no checkpoint weight loader is called.

    Reuse one session for all state representatives. Each prepare deeply checks
    all180 frames before BF16 and returns the actual official processor tensors.
    The caller owns the returned arrays and releases each row before the next.
    """

    def __init__(self, family, model_path, cosmos_package=None, event_sink=None):
        _require(family in FAMILIES and Path(model_path).is_dir(), "A known family and local checkpoint directory are required")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        implementation = ModelSession.__new__(ModelSession)
        implementation._restore = []
        implementation.context = {"active": False, "codes": None}
        self._implementation = implementation
        try:
            implementation._initialize_processor(family, model_path, event_sink or _DiscardEvents(), cosmos_package)
        except BaseException:
            implementation.close()
            raise

    @property
    def processor(self):
        return self._implementation.processor

    def prepare(self, row, ready, asset_root):
        _require(row["sample_id"] == ready["sample_id"] and row["source_parameters"] == ready["source_parameters"],
                 "CPU preflight row and ready identity differ")
        implementation = self._implementation
        implementation.context["deep"] = True
        frames, data, text, receipt = implementation.prepare(row, ready, asset_root)
        key = "pixel_values_flat_video" if implementation.family == "nemo" else "pixel_values_videos"
        _require(data[key].device.type == "cpu", "Prepare-only must remain on CPU")
        codes = implementation.context["codes"]
        receipt.update(new_model_calls=0, model_weights_loaded=False, preparation_device="cpu")
        del frames
        return {"sample_id": row["sample_id"], "source_parameters": row["source_parameters"],
                "source_codes": codes, "batch": data, "pixels": data[key],
                "chat_template_text": text, "input_receipt": receipt, "processor": self.processor}

    def close(self):
        self._implementation.close()

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


def prepare_row_only(manifest_path, model_key, model_root, sample_id, ready_path=None,
                     asset_root=None, cosmos_package=None, output_path=None):
    """One-row CPU preflight. Tensors stay in memory; optional JSON is small."""
    manifest, rows, ready, asset_root, _ = load_run_inputs(manifest_path, ready_path, asset_root)
    row = next((row for row in rows if row["sample_id"] == sample_id), None)
    _require(row is not None and model_key in FAMILIES, "Unknown row or model")
    model_path = _inside(model_root, manifest["models"][model_key]["checkpoint_relative"])
    with ProcessorSession(model_key, model_path, cosmos_package) as session:
        prepared = session.prepare(row, ready[sample_id], asset_root)
    if output_path is not None:
        atomic_json(output_path, {"sample_id": sample_id, "model": manifest["models"][model_key],
                                  "input_receipt": prepared["input_receipt"]})
    return prepared


def reconcile_technical_receipts(store, family):
    """Recover a verdict from saved receipts without ever calling a model."""
    store.pending_ids()  # refresh all journals under the model state lock
    receipts = {}
    verdicts = {}
    for event in store.events:
        row_id = event.get("sample_id")
        raw = store.outputs.get(row_id)
        key = (row_id, event.get("attempt_id") or (raw["attempt_id"] if raw else None))
        if event.get("event") == "POST_OUTPUT_INPUT_RECEIPT":
            receipts[key] = event
        if event.get("event") in ("TECHNICAL_PASS", "TECHNICAL_PASS_RECOVERED", "POST_OUTPUT_VALIDATION_FAILED"):
            failed = event["event"] == "POST_OUTPUT_VALIDATION_FAILED"
            stamp = (event.get("time_unix", 0), failed)
            if key not in verdicts or stamp >= verdicts[key][0]:
                verdicts[key] = (stamp, not failed)
    unresolved = {}
    for row_id, output in store.outputs.items():
        if row_id not in store.row_ids:
            continue  # Another live shard can be between raw fsync and its receipt.
        key = (row_id, output["attempt_id"])
        verdict = verdicts.get(key)
        if verdict is not None:
            if not verdict[1]:
                unresolved[row_id] = "Latest technical verdict is a persisted validation failure"
            continue
        receipt = receipts.get(key)
        if receipt is None:
            unresolved[row_id] = "Raw answer is durable but no post-output GPU input receipt is available"
            continue
        try:
            count = output["actual_input_tokens"]
            _require(count == output["input_receipt"]["input_tokens"] and count + 16384 <= 65536,
                     "Actual prompt/context budget differs from the prechecked input")
            if family in ("nemo", "qwen"):
                _require(receipt.get("actual_engine_processor_receipt") is not None, "Actual engine processor receipt missing")
                forwards = [item for worker in receipt["actual_GPU_visual_forward"] for item in worker]
            else:
                forwards = receipt["actual_GPU_model_forward"]
            _validate_forwards(family, forwards)
        except (KeyError, TypeError, ValueError) as error:
            unresolved[row_id] = str(error)
        else:
            store.event("TECHNICAL_PASS_RECOVERED", sample_id=row_id, attempt_id=output["attempt_id"])
    return unresolved


def _qualification_marker(store, identity, family, shard_id, qualification):
    ids = [row["sample_id"] for row in qualification]
    if not all(row_id in store.outputs for row_id in ids):
        return False
    _require(all(store.outputs[row_id].get("finish_reason") == "stop"
                 and store.outputs[row_id].get("raw_answer") in ("LEFT", "RIGHT") for row_id in ids),
             "Predetermined qualification output must stop normally with exactly LEFT or RIGHT; keep its raw terminal answer, do not retry implicitly")
    unresolved = reconcile_technical_receipts(store, family)
    _require(not any(row_id in unresolved for row_id in ids), "Worker qualification lacks actual input PASS")
    marker = {"status": "PASS", "row_ids": ids, "tasks": [row["task"] for row in qualification],
              "criterion": "actual input/forward guards, normal stop and exact LEFT/RIGHT; correctness is not a gate",
              "run_identity_sha256": digest(identity)}
    atomic_json(store.directory / ("qualification-" + shard_id + ".json"), marker)
    store.event("WORKER_QUALIFIED", marker=marker)
    return True


def build_run_identity(manifest_path, ready_path, manifest, model_key, model_root,
                       resources=None, cosmos_package=None):
    """Shared worker/supervisor binding; no weight load, CUDA or model call."""
    spec = manifest["models"][model_key]
    resources = resources if resources is not None else spec.get("resources", {})
    model_path = _inside(model_root, spec["checkpoint_relative"])
    _require(model_path.is_dir(), "The explicitly selected local checkpoint directory is missing")
    identity = {"schema_version": 1, "protocol_id": manifest["protocol_id"],
        "manifest_sha256": file_sha256(manifest_path), "ready_sha256": file_sha256(ready_path),
        "model_key": model_key, "model": spec, "generation": GENERATION,
        "effective_resources": resources,
        "retry_policy": {"max_attempts_per_sample": MAX_ATTEMPTS, "durable_raw_is_terminal": True},
        "checkpoint_assets": checkpoint_asset_identity(model_path),
        "cosmos_package_sources": _cosmos_package_identity(cosmos_package) if model_key == "cosmos" else None,
        "runtime_versions": _versions(),
        "all_row_ids_sha256": digest([row["sample_id"] for row in manifest["rows"]]),
        "runner_sha256": file_sha256(__file__), "storage_sha256": file_sha256(Path(__file__).with_name("storage.py"))}
    return identity, model_path


def run_manifest(manifest_path, ready_path=None, output_dir=None, model_key=None, model_root=None,
                 row_ids=None, shard_index=0, shard_count=1, gpu="0", host=None, resources=None,
                 asset_root=None, cosmos_package=None, shard_id=None, retry_failed=True, retry_interrupted=True):
    """Run exactly the selected unfinished IDs; no wall/length-total stop rule."""
    _require(model_key in FAMILIES and output_dir is not None and model_root is not None,
             "model_key, output_dir and model_root are required")
    _require("," not in str(gpu) and bool(str(gpu)), "Exactly one explicit GPU is required")
    if "torch" in sys.modules:
        _require(os.environ.get("CUDA_VISIBLE_DEVICES") == str(gpu), "Set GPU before importing Torch")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    manifest, rows, ready, asset_root, ready_path = load_run_inputs(manifest_path, ready_path, asset_root)
    selected = select_rows(rows, row_ids, shard_index, shard_count)
    _require(bool(selected), "Selected shard is empty")
    ordered, qualification = qualification_order(selected)
    _require(len(qualification) == 2, "Each production shard must include a predetermined T2 and T3 qualification row")
    qualification_ids = {row["sample_id"] for row in qualification}
    spec = manifest["models"][model_key]
    resources = resources if resources is not None else spec.get("resources", {})
    identity, model_path = build_run_identity(manifest_path, ready_path, manifest, model_key, model_root,
                                            resources, cosmos_package)
    shard_id = shard_id or ("ids-" + digest([r["sample_id"] for r in selected])[:16] if row_ids is not None
                            else f"onset-{shard_index}-of-{shard_count}")
    with RunStore(output_dir, identity, model_key, shard_id, [r["sample_id"] for r in selected],
                  all_row_ids=[r["sample_id"] for r in rows]) as store:
        pending = set(store.pending_ids(retry_failed=retry_failed, retry_interrupted=retry_interrupted))
        unresolved = reconcile_technical_receipts(store, model_key)
        _require(not unresolved, "Durable output has unresolved technical input validation; no new calls: " + str(unresolved))
        _require(all(row_id in store.outputs or row_id in pending for row_id in qualification_ids),
                 "Worker qualification is held or failed; resolve it explicitly before ordinary calls")
        qualified = _qualification_marker(store, identity, model_key, shard_id, qualification)
        if not pending:
            return store.summary(retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        import torch
        with _gpu_lock(output_dir, torch):
            snapshot = resource_snapshot(torch, resources or {})
            store.event("RESOURCE_ADMISSION", host_label=host, gpu_selection=str(gpu), **snapshot)
            _require(snapshot["admitted"], "Actual free VRAM/RAM does not meet the explicit model budgets")
            session = None
            try:
                session = ModelSession(model_key, model_path, store, cosmos_package, resources)
                for row in ordered:
                    row_id = row["sample_id"]
                    if row_id not in pending:
                        continue
                    _require(row_id in qualification_ids or qualified,
                             "Both predetermined qualification rows must pass before ordinary calls")
                    row_identity = digest({"row": row, "ready": ready[row_id]})
                    store.begin(row_id, row_identity)
                    attempt_began = time.monotonic()
                    try:
                        session.generate(row, ready[row_id], asset_root, lambda record, name=row_id: store.commit_output(name, record))
                        if not qualified:
                            qualified = _qualification_marker(store, identity, model_key, shard_id, qualification)
                    except PersistenceError:
                        raise
                    except BaseException as error:
                        # Post-output receipt failure retains the raw terminal answer.
                        if row_id in store.outputs:
                            store.event("POST_OUTPUT_VALIDATION_FAILED", sample_id=row_id,
                                        attempt_id=store.outputs[row_id]["attempt_id"],
                                        error_type=type(error).__name__, error=str(error))
                        else:
                            store.fail(row_id, error, interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)),
                                       elapsed_seconds=time.monotonic() - attempt_began)
                        raise
                    finally:
                        session.context.update(active=False, codes=None)
                summary = store.summary(retry_failed=retry_failed, retry_interrupted=retry_interrupted)
                store.event("SHARD_FINISHED", summary=summary)
                atomic_json(store.directory / ("summary-" + shard_id + ".json"), summary)
                return summary
            except BaseException as error:
                if not isinstance(error, PersistenceError):
                    store.event("WORKER_STOPPED", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                if session is not None:
                    session.close()
