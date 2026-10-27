"""Finite, source-bound Gemma latency and replay worker.

CPU commands never import vLLM or load model weights. ``run`` requires a sealed
profile, a shared CPU reference, a controller admission receipt, and the SAME
existing physical GPU lock used by the native T1 runner. No retries or launches
are hidden here. All kernel changes are opt-in and local to this worker process.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, ExitStack
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import textwrap
import time
import types
import uuid

from .storage import atomic_json, canonical_json, digest, file_sha256

SCHEMA = "gemma-speed-profile-v1"
REVISION = "842da3794eaa0b77d5f08bae87a17459d91ff475"
NATIVE_SHA = {
    "vllm.v1.attention.ops.triton_unified_attention": "65d0fbcf33d00789df8ca688c0a1c908e710cfaef53471178747ecae11f05829",
    "vllm.v1.attention.ops.triton_attention_helpers": "d134f1919807183f54a3a57499703da44ab2127d033aff3c1509a3274f37a5f7",
    "vllm.v1.attention.backends.triton_attn": "d4089a21fe947648c253f1ede264a60ea4e7534f2ef8812a5547c1483bc27374",
}
CTX = {}
SDK_FILES = {
    "vllm": ("vllm/model_executor/models/gemma4_mm.py", "vllm/model_executor/models/gemma4.py",
        "vllm/model_executor/offloader/uva.py", "vllm/v1/core/kv_cache_utils.py",
        "vllm/v1/attention/ops/triton_unified_attention.py", "vllm/v1/attention/ops/triton_attention_helpers.py",
        "vllm/v1/attention/backends/triton_attn.py"),
    "transformers": ("transformers/models/gemma4/modeling_gemma4.py", "transformers/models/gemma4/processing_gemma4.py",
        "transformers/models/gemma4/video_processing_gemma4.py"),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def runtime_sources():
    """Read installed source bytes without importing torch/vLLM/Transformers."""
    return {relative: file_sha256(importlib.metadata.distribution(distribution).locate_file(relative))
        for distribution, names in SDK_FILES.items() for relative in names}


def process_identity():
    stat = Path("/proc/self/stat")
    boot = Path("/proc/sys/kernel/random/boot_id")
    return {"host": socket.gethostname(), "pid": os.getpid(),
        "startticks": stat.read_text().rsplit(")", 1)[1].split()[19] if stat.exists() else None,
        "boot_id": boot.read_text().strip() if boot.exists() else None}


def seal(path, value):
    require(not Path(path).exists(), "Refusing to overwrite sealed evidence: " + str(path))
    atomic_json(path, value)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_items(items):
    require(bool(items), "Empty panel")
    ids = [x["sample_id"] for x in items]
    require(len(ids) == len(set(ids)), "Duplicate sample IDs")
    for x in items:
        require(x["task"] in ("T1", "T2", "T3"), "Unsupported task")
        require(x["sample_id"] == x["row"]["sample_id"] == x["ready"]["sample_id"], "Sample identity differs")
        require(digest(x["row"]) == x["row_sha256"] and digest(x["ready"]) == x["ready_sha256"], "Frozen row/ready changed")
        require(isinstance(x["row"].get("prompt"), str) and x["row"]["prompt"], "Missing exact prompt")
        r = x["ready"]
        rel = r.get("video_relative", r.get("video_path"))
        require(rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts, "Unsafe source path")
        require(len(r["video_sha256"]) == 64, "Missing source binding")
    return items


def engine_settings(checkpoint):
    # Existing qualified copy8 profile. Tuning never lowers context or dtype.
    return dict(model=str(checkpoint), dtype="bfloat16", max_model_len=65536,
        max_num_seqs=1, max_num_batched_tokens=34000, enforce_eager=True,
        enable_prefix_caching=False, enable_chunked_prefill=True,
        kv_cache_dtype="bfloat16", kv_cache_memory_bytes=32 * 2**30,
        gpu_memory_utilization=.90, async_scheduling=False, cpu_offload_gb=8,
        cpu_offload_params=["gate_up_proj"],
        limit_mm_per_prompt={"video": {"count": 1, "num_frames": 180}, "image": 0, "audio": 0},
        mm_processor_cache_gb=0, skip_mm_profiling=True,
        disable_custom_all_reduce=True, seed=0)


def validate_ranges(ranges):
    """Native inclusive intervals; binary lookup is valid only if disjoint."""
    require(len(ranges) == 180, "Expected exactly 180 native video intervals")
    previous = -1
    for start, end in ranges:
        require(isinstance(start, int) and isinstance(end, int)
            and previous < start < end and end - start + 1 <= 1024,
            "Native intervals must be sorted, disjoint, nonempty and within the local window")
        previous = end
    return True


def indexed_mask(q, k, ranges, window=1024):
    """CPU specification of the candidate, for boundary/equivalence tests."""
    import bisect
    i = bisect.bisect_right([a for a, _ in ranges], q) - 1
    mm = i >= 0 and ranges[i][0] <= q <= ranges[i][1] and ranges[i][0] <= k <= ranges[i][1]
    return (k <= q or mm) and (window <= 0 or q - k < window)


# These replace only native interval enumeration. The native kernel, dot,
# softmax, causal/window operations, accumulators and dtypes remain unchanged.
_BINARY_MASK = """
lo = tl.full(query_abs_pos.shape, 0, tl.int32)
hi = tl.full(query_abs_pos.shape, MAX_MM_RANGES, tl.int32)
for search_step in range(9):
    mid = (lo + hi) // 2
    active = lo < hi
    probe = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + mid * 2,
                    mask=active & (mid < MAX_MM_RANGES), other=2147483647)
    advance = probe <= query_abs_pos
    lo = tl.where(active & advance, mid + 1, lo)
    hi = tl.where(active & ~advance, mid, hi)
idx = lo - 1
range_start = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + idx * 2,
                      mask=idx >= 0, other=2147483647)
range_end = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + idx * 2 + 1,
                    mask=idx >= 0, other=-1)
mm_mask = ((idx >= 0) & (range_start < range_end)
           & (query_abs_pos >= range_start) & (query_abs_pos <= range_end)
           & (seq_offset[None, :] >= range_start) & (seq_offset[None, :] <= range_end))
if MM_PREFIX_CLAMP_SW and SLIDING_WINDOW > 0:
    mm_mask = mm_mask & ((query_abs_pos - seq_offset) < SLIDING_WINDOW)
seq_mask |= mm_mask
"""
_VECTOR_BOUNDS = """
range_indices = tl.arange(0, triton.next_power_of_2(MAX_MM_RANGES))
range_start = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + range_indices * 2,
                      mask=range_indices < MAX_MM_RANGES, other=0)
range_end = tl.load(mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + range_indices * 2 + 1,
                    mask=range_indices < MAX_MM_RANGES, other=0)
intersects_query_block = ((range_indices < MAX_MM_RANGES) & (range_start < range_end)
                         & (range_start <= query_abs_hi) & (range_end >= query_abs_lo))
mm_first_allowed_key = tl.maximum(range_start, query_abs_lo - SLIDING_WINDOW + 1)
first_allowed_key = tl.minimum(first_allowed_key,
    tl.min(tl.where(intersects_query_block, mm_first_allowed_key, first_allowed_key), 0))
last_allowed_key = tl.maximum(last_allowed_key,
    tl.max(tl.where(intersects_query_block, range_end, last_allowed_key), 0))
"""


def candidate_sources(helper_source, launcher_source, binary_ranges, prefill_block_m):
    """Source transformation with unique AST anchors; usable in CPU tests."""
    require(prefill_block_m in (0, 32), "Only separately sealed native/32 query tiling is supported")
    helpers = ast.parse(helper_source)
    wanted = ("compute_kv_seq_mask", "compute_tile_loop_bounds")
    functions = [n for n in helpers.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    require(len(functions) == 2, "Native helper interface changed")
    if binary_ranges:
        for fn in functions:
            hits = 0
            for parent in ast.walk(fn):
                for field, value in ast.iter_fields(parent):
                    if not isinstance(value, list):
                        continue
                    for i, node in enumerate(value):
                        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(MAX_MM_RANGES)":
                            replacement = _BINARY_MASK if fn.name == "compute_kv_seq_mask" else _VECTOR_BOUNDS
                            value[i:i + 1] = ast.parse(replacement).body
                            hits += 1
            require(hits == 1, "Native range enumeration changed: " + fn.name)
    helper_tree = ast.Module(body=ast.parse(
        "from vllm.triton_utils import tl, triton\n"
        "from vllm.v1.attention.ops.triton_attention_helpers import cdiv_fn\n").body + functions, type_ignores=[])
    launcher = ast.parse(launcher_source)
    targets = [n for n in launcher.body if isinstance(n, ast.FunctionDef) and n.name == "unified_attention"]
    require(len(targets) == 1, "Native launcher interface changed")
    fn = targets[0]
    if prefill_block_m:
        hits = 0
        for i, node in enumerate(fn.body):
            if isinstance(node, ast.Assign) and [ast.unparse(t) for t in node.targets] == ["BLOCK_M"]:
                fn.body[i + 1:i + 1] = ast.parse(
                    "if head_size == 512 and max_seqlen_q > 1 and num_queries_per_kv <= 16:\n"
                    "    BLOCK_M = 32\n").body
                hits += 1
                break
        require(hits == 1, "Native BLOCK_M assignment changed")
    return (ast.unparse(ast.fix_missing_locations(helper_tree)) + "\n",
            ast.unparse(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))) + "\n")


def _install_kernel(worker, config, output):
    """Runs in the engine process, before its first scientific request."""
    import triton
    modules = {name: importlib.import_module(name) for name in NATIVE_SHA}
    observed = {name: file_sha256(module.__file__) for name, module in modules.items()}
    require(observed == config["expected_native_sha256"], "Installed native attention source differs")
    native = modules["vllm.v1.attention.ops.triton_unified_attention"]
    helper = modules["vllm.v1.attention.ops.triton_attention_helpers"]
    backend = modules["vllm.v1.attention.backends.triton_attn"]
    binary = config["range_lookup"] == "binary"
    block_m = config["prefill_block_m"]
    receipt = {"configuration": config, "observed_native_sha256": observed, "changed_kernels": []}
    if binary or block_m:
        helper_code, launch_code = candidate_sources(Path(helper.__file__).read_text(),
            Path(native.__file__).read_text(), binary, block_m)
        if binary:
            # A real source file lets Triton inspect/cache the two derived
            # helpers. This file is run evidence; installed SDK bytes stay put.
            target = Path(output) / "native_range_helpers.py"
            require(not target.exists(), "Duplicate kernel installation")
            target.write_text(helper_code)
            name = "_gemma_ranges_" + hashlib.sha256(helper_code.encode()).hexdigest()[:16]
            spec = importlib.util.spec_from_file_location(name, target)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
            native.compute_kv_seq_mask = module.compute_kv_seq_mask
            native.compute_tile_loop_bounds = module.compute_tile_loop_bounds
            # A fresh JITFunction recomputes dependency hashes after rebinding
            # the helpers, even if startup touched the old kernel's cache key.
            kernel = native.kernel_unified_attention
            require(inspect.getsource(kernel.fn).lstrip().startswith("@triton.jit\n"), "Native kernel decorator changed")
            native.kernel_unified_attention = triton.jit(kernel.fn)
            receipt["changed_kernels"].append("native interval enumeration only")
            receipt["derived_helpers_sha256"] = file_sha256(target)
        if block_m:
            namespace = dict(native.__dict__)
            exec(compile(launch_code, "<gemma-native-query-tiling>", "exec"), namespace)
            native.unified_attention = namespace["unified_attention"]
            backend.unified_attention = native.unified_attention
            receipt["derived_launcher_sha256"] = hashlib.sha256(launch_code.encode()).hexdigest()
            receipt["changed_kernels"].append("head512 prefill BLOCK_M=32")
    return receipt


def sha_array(array):
    import numpy as np
    require(array.flags.c_contiguous, "Hash requires a contiguous view")
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def sha_tensor(tensor):
    import torch
    return sha_array(tensor.detach().contiguous().cpu().view(torch.uint8).numpy())


def decode(item, asset_root, streamlined):
    """Same uint16 -> FP32 math; preallocation removes stack/cast temporaries."""
    import av
    import numpy as np
    ready = item["ready"]
    path = Path(asset_root) / ready.get("video_relative", ready.get("video_path"))
    require(path.resolve().is_relative_to(Path(asset_root).resolve()), "Source escaped asset root")
    require(file_sha256(path) == ready["video_sha256"], "Source video hash differs")
    codes = np.empty((180, 528, 768, 3), np.uint16) if streamlined else None
    pending, pts, stream_hash = [], [], hashlib.sha256()
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        require(stream.codec_context.name == "ffv1" and stream.codec_context.format.name == "gbrp16le", "Wrong RGB16 source codec")
        for i, frame in enumerate(container.decode(video=0)):
            require(i < 180, "Unexpected extra frame")
            data = frame.to_ndarray(format="rgb48le")
            require(data.dtype == np.uint16 and data.shape == (512, 768, 3), "Source frame shape/dtype differs")
            stream_hash.update(memoryview(data).cast("B"))
            if streamlined:
                codes[i, :512] = data
            else:
                pending.append(data)
            pts.append(float(frame.pts * frame.time_base))
    require(len(pts) == 180 and max(abs(t - i / 60) for i, t in enumerate(pts)) <= .001, "Source frames/timestamps differ")
    stream_sha = stream_hash.hexdigest()
    if ready.get("stream_sha256"):
        require(stream_sha == ready["stream_sha256"], "Decoded source hash differs")
    if not streamlined:
        stacked = np.stack(pending)
        codes = np.empty((180, 528, 768, 3), np.uint16)
        codes[:, :512] = stacked
        del stacked, pending
    codes[:, 512:] = codes[0, 0, 0]
    values = codes.astype(np.float32)
    if streamlined:
        np.multiply(values, 255. / 65535., out=values)
    else:
        values = values * (255. / 65535.)
    return codes, values, {"source_video_sha256": ready["video_sha256"], "source_stream_sha256": stream_sha,
        "submitted_shape": list(values.shape), "submitted_dtype": str(values.dtype),
        "frames": 180, "padding_bottom_rows": 16, "source_pts_s": pts}


def prepare(session, item, roots, streamlined, deep):
    import torch
    from transformers.video_utils import VideoMetadata
    codes, frames, receipt = decode(item, roots[item["task"]], streamlined)
    impl = session._implementation
    messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": item["row"]["prompt"]}]}]
    text = session.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    require("<|think|>" not in text and text.endswith("<|channel>thought\n<channel|>"), "Native OFF prompt changed")
    impl.context.update(active=True, deep=deep, codes=codes)
    metadata = VideoMetadata(total_num_frames=180, fps=60., duration=3., frames_indices=list(range(180)))
    data = session.processor(text=[text], videos=[frames], videos_kwargs={"do_resize": False,
        "do_sample_frames": False, "video_metadata": [metadata], "max_soft_tokens": 280}, return_tensors="pt")
    receipt.update(impl._guard_pixels(data, session.processor))
    positions = data[next(k for k in data if "position_ids" in k and "video" in k)][0]
    pixels = data["pixel_values_videos"][0].to(torch.bfloat16)
    soft = (((positions != -1).any(-1).sum(-1)) // session.processor.video_processor.pooling_kernel_size**2).tolist()
    require(len(soft) == 180 and set(soft) == {176}, "Native soft token geometry differs")
    ids = data["input_ids"][0].tolist()
    require(len(ids) + 16384 <= 65536, "Prompt exceeds context budget")
    payload = {"pixel_values_videos": pixels, "pixel_position_ids_videos": positions,
        "video_frame_counts": torch.tensor([180]), "video_num_soft_tokens": [soft],
        "video_timestamps": [[i / 60. for i in range(180)]]}
    checks = {"input_token_ids": ids, "input_token_ids_sha256": digest(ids), "prompt_sha256": digest(text),
        "source_stream_sha256": receipt["source_stream_sha256"],
        "position_sha256": sha_tensor(positions), "position_shape": list(positions.shape),
        "HF_control_tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": sha_tensor(v)}
            for k, v in data.items() if hasattr(v, "shape") and ("mask" in k or "position" in k)}}
    if deep:
        checks.update(processor_input_FP32_sha256=sha_array(frames), pixel_FP32_sha256=sha_tensor(data["pixel_values_videos"][0]),
            input_sha256=sha_tensor(pixels), projection_sha256=sha_tensor(2 * (pixels - .5)))
    receipt.update(checks)
    impl.context.update(active=False, codes=None)
    del codes, data
    return frames, payload, text, receipt


def _install_adapter(session, streamlined):
    from PIL import Image
    from transformers.feature_extraction_utils import BatchFeature
    from vllm.model_executor.models.gemma4_mm import Gemma4MultiModalProcessor
    original, pil = Gemma4MultiModalProcessor._apply_hf_processor_main, Image.fromarray
    def no_pil(*args, **kwargs):
        require(not CTX.get("active"), "Actual RGB16 request attempted PIL conversion")
        return pil(*args, **kwargs)
    def prepared(self, mm_items, hf_kwargs):
        if not CTX.get("active"):
            return original(self, mm_items, hf_kwargs)
        import numpy as np
        require(mm_items.get_all_counts() == {"video": 1}, "Multimodal count differs")
        values = mm_items["video"].get_processor_data()["videos"]
        require(len(values) == 1, "Video count differs")
        frames, metadata = values[0]
        expected = CTX["frames"]
        require(frames.shape == expected.shape and frames.dtype == expected.dtype
            and metadata["frames_indices"] == list(range(180)) and metadata["fps"] == 60., "Actual submitted frame metadata differs")
        identical_view = (frames.strides == expected.strides and
            frames.__array_interface__["data"][0] == expected.__array_interface__["data"][0])
        require((streamlined and identical_view) or np.array_equal(frames, expected), "Actual FP32 ndarray changed")
        CTX["adapter_calls"] += 1
        return BatchFeature(CTX["payload"])
    impl = session._implementation
    impl._patch(Image, "fromarray", no_pil)
    impl._patch(Gemma4MultiModalProcessor, "_apply_hf_processor_main", prepared)


def _install_observer(worker):
    import torch
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    model = worker.get_model()
    worker._gemma_speed = {"deep": True, "receipts": [], "masks": [], "events": []}
    state = worker._gemma_speed
    original = model._process_video_input
    def projection(module, args):
        if state["deep"]:
            state["projection"].update(memoryview(args[0].detach().contiguous().cpu().view(torch.uint8).numpy()).cast("B"))
    model.vision_tower.patch_embedder.input_proj.register_forward_pre_hook(projection)
    def video(this, inputs):
        x, pos = inputs["pixel_values_videos"], inputs["pixel_position_ids_videos"]
        item = {"shape": list(x.shape), "dtype": str(x.dtype), "device": str(x.device),
            "frames": inputs["video_frame_counts"].tolist(), "deep": state["deep"]}
        state["projection"] = hashlib.sha256()
        if state["deep"]:
            item.update(input_sha256=sha_tensor(x), position_sha256=sha_tensor(pos))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); result = original(inputs); b.record()
        state["events"].append(("vision", a, b))
        item["embedding_shapes"] = [list(t.shape) for t in result]
        if state["deep"]:
            item["projection_sha256"] = state["projection"].hexdigest()
        state["receipts"].append(item)
        return result
    model._process_video_input = types.MethodType(video, model)
    native_build = TritonAttentionMetadataBuilder.build
    def mask_build(self, *args, **kwargs):
        result = native_build(self, *args, **kwargs)
        ranges = getattr(result, "mm_prefix_range", None)
        if ranges:
            for values in ranges.values():
                validate_ranges(values)
            value = {str(k): [list(v) for v in r] for k, r in ranges.items()}
            fingerprint = digest(value)
            if fingerprint not in {x["sha256"] for x in state["masks"]}:
                state["masks"].append({"sha256": fingerprint, "ranges": value})
        return result
    TritonAttentionMetadataBuilder.build = mask_build
    native_forward = model.forward
    def forward(*args, **kwargs):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); result = native_forward(*args, **kwargs); b.record()
        state["events"].append(("language", a, b))
        return result
    model.forward = forward
    return {"observer": "first2 deep; ordinary geometry, prompt tokens and native mask intervals"}


def _set_deep(worker, value):
    worker._gemma_speed["deep"] = value
    return True


def _read_observer(worker):
    import torch
    torch.cuda.synchronize()
    state = worker._gemma_speed
    result = {"video": state["receipts"], "native_masks": state["masks"],
        "CUDA_seconds": [{"stage": n, "seconds": a.elapsed_time(b) / 1000} for n, a, b in state["events"]],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    state.update(receipts=[], masks=[], events=[])
    return result


def _placement(worker):
    import torch
    from vllm.model_executor.offloader.base import get_offloader
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    model, places, names = worker.get_model(), Counter(), []
    for name, p in model.named_parameters():
        place = "uva" if getattr(p, "_vllm_is_uva_offloaded", False) else p.device.type
        places[name.split(".")[0] + ":" + place] += p.numel() * p.element_size()
        require(p.dtype == torch.bfloat16, "Model weight precision differs")
        if place != "cuda":
            names.append(name)
    offloader = get_offloader()
    cache = worker.model_runner.kv_cache_config
    capacity, concurrency = get_kv_cache_capacity(worker.vllm_config, cache)
    result = {"kv": {"capacity_tokens": int(capacity), "max_concurrency": float(concurrency),
        "num_gpu_blocks": cache.num_blocks, "helper_source_sha256": file_sha256(inspect.getsourcefile(get_kv_cache_capacity))},
        "placement_bytes": dict(places), "offloaded_names": names, "worker": process_identity(),
        "uva": getattr(offloader, "uva_offloading", None), "offload_bytes": getattr(offloader, "cpu_offload_bytes", 0),
        "torch_policy": {"TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "TF32_cudnn": torch.backends.cudnn.allow_tf32,
            "matmul_precision": torch.get_float32_matmul_precision()}}
    require(result["uva"] is False and result["offload_bytes"] >= 8 * 2**30
        and all(n.startswith("language_model.") and ".gate_up_proj." in n for n in names), "Qualified copy8 placement differs")
    require(not any(k.startswith(("vision_tower:cpu", "vision_tower:uva", "embed_vision:cpu", "embed_vision:uva")) for k in places), "Vision weights offloaded")
    # The frozen compatibility shim uses vLLM's hybrid-group capacity helper;
    # multiplication of one block count by one block size is NOT sufficient.
    require(result["kv"]["capacity_tokens"] >= 65536, "Actual native KV capacity below full context")
    return result


def _worker_identity(worker):
    return {"worker": process_identity()}


def runtime_check(runtime):
    require(Path(runtime["checkpoint"]).name == REVISION, "Checkpoint revision changed")
    require(runtime.get("expected_packages"), "Pinned runtime packages required")
    versions = {n: importlib.metadata.version(n) for n in runtime["expected_packages"]}
    require(versions == runtime["expected_packages"], "Runtime packages differ")
    allowed_environment = {"NUMPY_MADVISE_HUGEPAGE", "CLASSIC_PRECISION_KV_COMPAT", "PYTHONPATH",
        "VLLM_ALLOW_INSECURE_SERIALIZATION", "OMP_NUM_THREADS", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE",
        "CUDA_VISIBLE_DEVICES", "VLLM_WEIGHT_OFFLOADING_DISABLE_UVA", "VLLM_WORKER_MULTIPROC_METHOD",
        "PYTHONDONTWRITEBYTECODE", "TOKENIZERS_PARALLELISM"}
    require(set(runtime["environment"]) <= allowed_environment, "Unreviewed environment fields; never dump account credentials")
    for name, value in runtime["environment"].items():
        require(os.environ.get(name) == value, "Runtime environment differs: " + name)
    require(os.environ.get("VLLM_WEIGHT_OFFLOADING_DISABLE_UVA") == "1", "Explicit text copies required")
    require(os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn", "Fresh spawned GPU worker required")
    return versions


@contextmanager
def physical_lease(runtime, torch):
    import fcntl
    from .inference import resource_snapshot
    require(torch.cuda.device_count() == 1, "Exactly one visible GPU per worker")
    root = Path(runtime["lease_root"]).resolve()
    require(root == Path(runtime["asset_roots"]["T1"]).resolve() / "evaluation/outputs", "Native physical lock root changed")
    device = str(getattr(torch.cuda.get_device_properties(0), "uuid", None) or os.environ["CUDA_VISIBLE_DEVICES"])
    path = root / (".gpu-" + digest({"host": socket.gethostname(), "device": device}) + ".lock")
    require(path.is_file() and not path.is_symlink(), "Existing native GPU lock required; never create another lock")
    with path.open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Prior copy8 observed torch peak allocation was 95,907,840,000 bytes
        # (89.32 GiB). The original T1 admission budget of 94 GiB leaves room
        # for driver/allocator overhead; 64 GiB RAM is its existing host budget.
        # These are admission budgets, not a claim about current measurements.
        state = resource_snapshot(torch, {"min_free_gpu_gib": 94, "min_available_ram_gib": 64})
        require(state["admitted"], "Insufficient resource margin")
        st = os.fstat(stream.fileno())
        require((st.st_dev, st.st_ino) == (path.stat().st_dev, path.stat().st_ino), "Physical lock inode changed")
        yield {"path": str(path), "st_dev": st.st_dev, "st_ino": st.st_ino, "device": device, "resources": state}


def seal_profile(args):
    plan, runtime = read(args.plan), read(args.runtime)
    panel = read(args.panel) if args.panel else plan[args.panel_key]
    items = validate_items(panel["items"])
    require(len(items) in (20, 48), "Only the finite 20-speed or 48-replay panel is supported")
    require(args.variant == "kernel" or (args.range_lookup == "native" and args.prefill_block_m == 0), "Kernel settings require explicit kernel variant")
    from . import inference
    runtime_check(runtime)
    profile = {"schema": SCHEMA, "name": args.name, "variant": args.variant,
        "plan_file_sha256": file_sha256(args.plan), "plan_sha256": digest(plan), "panel": panel,
        "runtime": runtime, "engine_settings": engine_settings(runtime["checkpoint"]),
        "kernel": {"range_lookup": args.range_lookup, "prefill_block_m": args.prefill_block_m,
            "expected_native_sha256": NATIVE_SHA},
        "implementation": {"gemma_speed": file_sha256(__file__), "inference": file_sha256(inference.__file__)},
        "runtime_sources": runtime_sources(), "checkpoint_assets": inference.checkpoint_asset_identity(runtime["checkpoint"]),
        "first_deep_calls": 2, "maximum_model_calls": len(items), "automatic_retry": False}
    seal(args.output, profile)
    return {"status": "SEALED", "profile_sha256": digest(profile), "calls": len(items), "GPU_calls": 0}


def cpu_reference(args):
    from .inference import ProcessorSession
    profile = read(args.profile)
    runtime_check(profile["runtime"])
    require(runtime_sources() == profile["runtime_sources"], "CPU reference SDK bytes changed")
    validate_items(profile["panel"]["items"])
    require(not Path(args.output).exists(), "CPU reference already exists")
    references = {}
    with ProcessorSession("gemma", profile["runtime"]["checkpoint"]) as session:
        for item in profile["panel"]["items"]:
            frames, payload, text, receipt = prepare(session, item, profile["runtime"]["asset_roots"], False, True)
            references[item["sample_id"]] = receipt
            del frames, payload
            gc.collect()
    result = {"schema": "gemma-shared-CPU-reference-v1", "panel_sha256": digest(profile["panel"]),
        "checkpoint_revision": REVISION, "implementation": profile["implementation"],
        "rows": references, "model_calls": 0}
    seal(args.output, result)
    return {"status": "COMPLETE", "rows": len(references), "GPU_calls": 0}


def verify_input(actual, reference, deep):
    keys = ["input_token_ids", "input_token_ids_sha256", "prompt_sha256", "source_stream_sha256",
        "position_sha256", "position_shape", "HF_control_tensors"]
    if deep:
        keys += ["processor_input_FP32_sha256", "pixel_FP32_sha256", "input_sha256", "projection_sha256"]
    for key in keys:
        require(actual[key] == reference[key], "INVALID input control: " + key)


def validate_gpu(observation, reference, deep):
    require(len(observation) == 1 and len(observation[0]["video"]) == 1, "Expected one actual GPU video forward")
    item = observation[0]["video"][0]
    require(item["shape"] == [180, 2520, 768] and item["dtype"] == "torch.bfloat16"
        and item["device"].startswith("cuda") and item["frames"] == [180]
        and item["embedding_shapes"][0][0] == 31680, "INVALID actual GPU input geometry")
    require(observation[0]["native_masks"], "Missing actual native attention interval receipt")
    if deep:
        for key in ("input_sha256", "position_sha256", "projection_sha256"):
            require(item[key] == reference[key], "INVALID actual GPU bytes: " + key)


def owned_worker_alive(identity):
    require(identity["host"] == socket.gethostname(), "Engine worker belongs to another host")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if identity["boot_id"] != boot:
        return False
    path = Path("/proc") / str(identity["pid"]) / "stat"
    try:
        # comm can contain spaces. Fields after the closing ')' begin at state.
        tail = path.read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return tail[19] == str(identity["startticks"]) and tail[0] != "Z"


def shutdown_owned_engine(engine, placements, output):
    """Keep the physical lease held until its exact owned GPU workers exit.

    An unexpected shutdown failure remains visible. We never free a live
    engine's lease, kill another process, or automatically retry inference.
    """
    workers = [x["worker"] for x in placements]
    atomic_json(output / "exit.json", {"status": "SHUTDOWN_STARTED", "owned_workers": workers})
    error = None
    try:
        engine.llm_engine.engine_core.shutdown()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    waiting_recorded = False
    while any(owned_worker_alive(x) for x in workers):
        if not waiting_recorded:
            atomic_json(output / "exit.json", {"status": "WAITING_OWNED_ENGINE_EXIT",
                "physical_lease_held": True, "owned_workers": workers, "shutdown_error": error})
            waiting_recorded = True
        time.sleep(1)
    atomic_json(output / "exit.json", {"status": "COMPLETE" if error is None else "SHUTDOWN_ERROR_WORKERS_EXITED",
        "owned_engine_workers_exited": True, "owned_workers": workers, "shutdown_error": error,
        "lease_released_after_engine_exit": False})
    require(error is None, "Engine shutdown raised; generation completion is not clean process completion")


def run(args):
    require("torch" not in sys.modules and "vllm" not in sys.modules, "A fresh process is required")
    profile, reference, admission = read(args.profile), read(args.reference), read(args.admission)
    from . import inference
    require(profile["schema"] == SCHEMA and profile["implementation"] == {
        "gemma_speed": file_sha256(__file__), "inference": file_sha256(inference.__file__)}, "Worker source changed after sealing")
    items = validate_items(profile["panel"]["items"])
    require(len(items) == profile["maximum_model_calls"] in (20, 48), "Finite request budget changed")
    require(reference["panel_sha256"] == digest(profile["panel"])
        and set(reference["rows"]) == {x["sample_id"] for x in items}, "CPU reference panel differs")
    require(admission["plan_sha256"] == profile["plan_sha256"] and admission["qwen_t1_complete"] == 3840
        and admission["all_owned_engines_exited"] is True and admission["physical_gpu_locks_released"] is True,
        "Current controller completion/release admission is missing")
    runtime = profile["runtime"]
    versions = runtime_check(runtime)
    require(runtime_sources() == profile["runtime_sources"]
        and inference.checkpoint_asset_identity(runtime["checkpoint"]) == profile["checkpoint_assets"],
        "Runtime SDK or checkpoint assets changed after sealing")
    require(profile["engine_settings"] == engine_settings(runtime["checkpoint"]), "Unsupported engine override")
    output = Path(args.output)
    require(not output.exists(), "No implicit resume or overwrite")
    output.mkdir(parents=True)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from .inference import ProcessorSession
    engine, records = None, []
    try:
        with physical_lease(runtime, torch) as lease, ExitStack() as cleanup, ProcessorSession("gemma", runtime["checkpoint"]) as session:
            streamlined = profile["variant"] != "baseline"
            _install_adapter(session, streamlined)
            atomic_json(output / "identity.json", {"profile": profile, "profile_sha256": digest(profile),
                "reference_sha256": file_sha256(args.reference), "admission_sha256": file_sha256(args.admission),
                "lease": lease, "versions": versions, "owner": process_identity(), "host": socket.gethostname(),
                "engine_id": uuid.uuid4().hex,
                "argv": sys.argv, "declared_environment": runtime["environment"],
                "python": sys.executable, "torch_TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
                "torch_TF32_cudnn": torch.backends.cudnn.allow_tf32,
                "matmul_precision": torch.get_float32_matmul_precision()})
            began = time.monotonic()
            engine = LLM(**profile["engine_settings"])
            owned = []
            # ExitStack runs before physical_lease exits, including on an input
            # mismatch. A released lock must never advertise a still-live engine.
            cleanup.callback(shutdown_owned_engine, engine, owned, output)
            owned.extend(engine.collective_rpc(_worker_identity))
            placements = engine.collective_rpc(_placement)
            load = {"seconds": time.monotonic() - began, "placement": placements,
                "kernel": engine.collective_rpc(_install_kernel, args=(profile["kernel"], str(output))),
                "observer": engine.collective_rpc(_install_observer)}
            atomic_json(output / "load.json", load)
            sampling = SamplingParams(n=1, temperature=0., top_p=1., repetition_penalty=1., seed=0,
                max_tokens=16384, structured_outputs=StructuredOutputsParams(choice=["LEFT", "RIGHT"]))
            processor = None
            if not streamlined:
                from vllm.config import ModelConfig
                from vllm.multimodal import MULTIMODAL_REGISTRY
                processor = MULTIMODAL_REGISTRY.create_processor(ModelConfig(model=runtime["checkpoint"],
                    dtype="bfloat16", max_model_len=65536, limit_mm_per_prompt={"video": 1, "image": 0, "audio": 0}, mm_processor_cache_gb=0))
            for index, item in enumerate(items):
                deep, sid = index < 2, item["sample_id"]
                started = time.monotonic()
                engine.collective_rpc(_set_deep, args=(deep,))
                frames, payload, text, receipt = prepare(session, item, runtime["asset_roots"], streamlined, deep)
                verify_input(receipt, reference["rows"][sid], deep)
                metadata = {"fps": 60., "duration": 3., "total_num_frames": 180,
                    "frames_indices": list(range(180)), "do_sample_frames": False, "video_backend": "qualified-rgb16"}
                CTX.update(active=True, frames=frames, payload=payload, adapter_calls=0)
                if processor:
                    from vllm.multimodal.processing.inputs import ProcessorInputs
                    from vllm.multimodal.processing.context import TimingContext
                    parsed = processor.info.parse_mm_data({"video": (frames, metadata)})
                    checked = processor.apply(ProcessorInputs(prompt=session.processor.tokenizer.encode(text, add_special_tokens=False),
                        mm_data_items=parsed), TimingContext(enabled=False))
                    require(checked["prompt_token_ids"] == receipt["input_token_ids"], "CPU prompt-token expansion differs")
                    del parsed, checked
                prepared = time.monotonic()
                result = engine.generate({"prompt": text, "multi_modal_data": {"video": (frames, metadata)}}, sampling, use_tqdm=False)[0]
                finished = time.monotonic()
                choice = result.outputs[0]
                raw = {"sample_id": sid, "task": item["task"], "raw_answer": choice.text,
                    "output_token_ids": list(choice.token_ids), "finish_reason": choice.finish_reason,
                    "stop_reason": choice.stop_reason, "actual_input_token_ids": list(result.prompt_token_ids),
                    "deep": deep, "model_seconds": finished - prepared, "prepare_seconds": prepared - started,
                    "to_answer_seconds": finished - started, "input_receipt": receipt}
                append(output / "raw.jsonl", raw)
                observation = engine.collective_rpc(_read_observer)
                append(output / "events.jsonl", {"sample_id": sid, "event": "ACTUAL_GPU_INPUT", "observation": observation})
                require(raw["actual_input_token_ids"] == receipt["input_token_ids"], "INVALID actual prompt tokens")
                require(CTX["adapter_calls"] >= 1 and choice.finish_reason == "stop" and choice.text in ("LEFT", "RIGHT"), "Invalid completion")
                validate_gpu(observation, reference["rows"][sid], deep)
                CTX.clear()
                del frames, payload, result, receipt
                gc.collect()
                item_result = {"sample_id": sid, "deep": deep, "status": "PASS",
                    "full_request_seconds": time.monotonic() - started, "model_seconds": raw["model_seconds"],
                    "prepare_seconds": raw["prepare_seconds"], "native_masks": observation[0]["native_masks"],
                    "CUDA_seconds": observation[0]["CUDA_seconds"]}
                append(output / "events.jsonl", {"event": "TECHNICAL_PASS", **item_result})
                records.append(item_result)
                atomic_json(output / "progress.json", {"status": "RUNNING", "completed": len(records), "expected": len(items), "last": item_result})
            atomic_json(output / "result.json", {"status": "GENERATION_COMPLETE", "profile_sha256": digest(profile),
                "panel_sha256": digest(profile["panel"]), "calls": len(records), "rows": records,
                "timing": timing_summary(records), "load_seconds": load["seconds"]})
        exit_receipt = read(output / "exit.json")
        require(exit_receipt["status"] == "COMPLETE" and exit_receipt["owned_engine_workers_exited"], "Engine exit is incomplete")
        exit_receipt["lease_released_after_engine_exit"] = True
        atomic_json(output / "exit.json", exit_receipt)
        completed = read(output / "result.json")
        completed["status"] = "COMPLETE"
        atomic_json(output / "result.json", completed)
    except BaseException as exc:
        atomic_json(output / "failure.json", {"status": "STOPPED", "error_type": type(exc).__name__,
            "error": str(exc), "completed": len(records), "automatic_retry": False})
        raise
    finally:
        CTX.clear()
    return {"status": "COMPLETE", "calls": len(records), "output": str(output)}


def timing_summary(records):
    values = sorted(r["full_request_seconds"] for r in records if not r["deep"] and r["status"] == "PASS")
    def quantile(q):
        if not values:
            return None
        p = q * (len(values) - 1)
        a = math.floor(p)
        return values[a] + (values[min(a + 1, len(values) - 1)] - values[a]) * (p - a)
    return {"ordinary_n": len(values), "p50_seconds": quantile(.5), "p90_seconds": quantile(.9),
        "max_seconds": max(values) if values else None, "deep_qualification_n": sum(r["deep"] for r in records),
        "speed_threshold_met": bool(values) and quantile(.5) <= 25 and quantile(.9) <= 27.5}


def compare_runs(paths, purpose="candidate"):
    """Compare raw tokens and technical closure; never compare only accuracy."""
    require(purpose in ("candidate", "consistency"), "Unknown comparison purpose")
    require(len({str(Path(p).resolve()) for p in paths}) == len(paths), "Repeated output directory is not an independent observation")
    runs = []
    for directory in paths:
        root = Path(directory)
        result, identity = read(root / "result.json"), read(root / "identity.json")
        require(result["status"] == "COMPLETE", "Run is incomplete")
        exit_receipt = read(root / "exit.json")
        require(not (root / "failure.json").exists() and exit_receipt["status"] == "COMPLETE"
            and exit_receipt["owned_engine_workers_exited"] and exit_receipt["lease_released_after_engine_exit"],
            "Clean engine exit and subsequent physical lease release are required")
        rows = [json.loads(s) for s in (root / "raw.jsonl").read_text().splitlines() if s]
        ids = [r["sample_id"] for r in rows]
        require(len(ids) == len(set(ids)) == result["calls"], "Raw output budget/duplicates differ")
        require(ids == [x["sample_id"] for x in identity["profile"]["panel"]["items"]], "Raw panel order differs")
        require(ids == [r["sample_id"] for r in result["rows"]]
            and all(r["status"] == "PASS" for r in result["rows"]), "Technical closure differs")
        runs.append((root, result, rows, identity, read(root / "load.json")))
    require(len(runs) >= 2, "At least two observations required")
    require(len({r[1]["panel_sha256"] for r in runs}) == 1, "Panels differ")
    differences, input_differences = [], []
    baseline = runs[0][2]
    keys = ("output_token_ids", "raw_answer", "finish_reason", "stop_reason")
    for root, result, rows, identity, load in runs[1:]:
        for i, (a, b) in enumerate(zip(baseline, rows, strict=True)):
            changed = [k for k in keys if a[k] != b[k]]
            controls = ["actual_input_token_ids"] if a["actual_input_token_ids"] != b["actual_input_token_ids"] else []
            for key in ("source_stream_sha256", "prompt_sha256", "position_sha256", "HF_control_tensors"):
                if a["input_receipt"][key] != b["input_receipt"][key]:
                    controls.append(key)
            if runs[0][1]["rows"][i]["native_masks"] != result["rows"][i]["native_masks"]:
                controls.append("native_masks")
            if controls:
                input_differences.append({"sample_id": a["sample_id"], "run": str(root), "fields": controls})
            if changed:
                differences.append({"sample_id": a["sample_id"], "run": str(root), "fields": changed})
    kernel_profiles = {digest(x[3]["profile"]["kernel"]) for x in runs}
    consistency_ready = False
    if purpose == "consistency":
        require(len(runs) == 6 and len(baseline) == 48, "Consistency requires 48 IDs x 6 observations")
        require(len({x[3]["engine_id"] for x in runs}) == 6
            and len({digest(x[4]["placement"][0]["worker"]) for x in runs}) == 6,
            "Independent cold engine identities are required")
        devices = Counter((x[3]["host"], x[3]["lease"]["device"]) for x in runs)
        require(len(devices) == 3 and set(devices.values()) == {2}, "Require two cold observations on each of three cards")
        profiles = []
        for _, _, _, identity, load in runs:
            p = identity["profile"]
            profiles.append(digest({"kernel": p["kernel"], "variant": p["variant"], "implementation": p["implementation"],
                "runtime_sources": p["runtime_sources"], "checkpoint_assets": p["checkpoint_assets"],
                "settings": {k: v for k, v in p["engine_settings"].items() if k != "model"},
                "torch_policy": load["placement"][0]["torch_policy"]}))
        require(len(set(profiles)) == 1, "Consistency observations used different execution profiles")
        consistency_ready = not differences and not input_differences
    status = "MATCH"
    if input_differences:
        status = "INVALID_EVIDENCE"
    elif differences:
        # Pure CPU changes must retain B0 tokens. A changed numerical kernel
        # is a new execution profile, whose six-observation replay is separate.
        status = "PROFILE_OUTPUT_DIFFERENCE" if purpose == "candidate" and len(kernel_profiles) > 1 else "DIFFERENT"
    return {"status": status, "purpose": purpose, "observations": len(runs),
        "unique_ids": len(baseline), "output_differences": differences, "input_differences": input_differences,
        "six_observation_consistency_pass": consistency_ready,
        "production_authorized_by_this_command": False,
        "speed": {str(root): result["timing"] for root, result, rows, identity, load in runs}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("inspect")
    p.add_argument("--plan", required=True)
    p = commands.add_parser("seal-profile")
    for name in ("plan", "runtime", "name", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--panel")
    p.add_argument("--panel-key", choices=("panel", "consistency_cell"), default="panel")
    p.add_argument("--variant", choices=("baseline", "cpu", "kernel"), required=True)
    p.add_argument("--range-lookup", choices=("native", "binary"), default="native")
    p.add_argument("--prefill-block-m", type=int, choices=(0, 32), default=0)
    p = commands.add_parser("reference")
    p.add_argument("--profile", required=True)
    p.add_argument("--output", required=True)
    p = commands.add_parser("run")
    for name in ("profile", "reference", "admission", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("compare")
    p.add_argument("runs", nargs="+")
    p.add_argument("--purpose", choices=("candidate", "consistency"), default="candidate")
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "inspect":
        plan = read(args.plan)
        items = validate_items(plan["panel"]["items"])
        result = {"status": "VALID", "plan_sha256": digest(plan), "panel_sha256": digest(plan["panel"]),
            "counts": dict(Counter(x["task"] for x in items)), "GPU_calls": 0}
    elif args.command == "seal-profile":
        result = seal_profile(args)
    elif args.command == "reference":
        result = cpu_reference(args)
    elif args.command == "run":
        result = run(args)
    else:
        result = compare_runs(args.runs, args.purpose)
        if args.output:
            seal(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
