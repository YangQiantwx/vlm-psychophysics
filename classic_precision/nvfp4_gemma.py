"""Finite Gemma NVFP4 experiment using the frozen C3 input and request path.

The published checkpoint is a new precision condition. Its outputs may differ
from BF16; two independent NVFP4 cold runs must agree in every output token.
Only process-local bindings change. Sealing/comparison never load a processor,
Torch, model weights, or CUDA; no reference-generation command is provided.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from functools import partial
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from . import gemma_speed as g
from . import gemma_cpu_copy as c
from . import inference

SCHEMA = "gemma-nvfp4-experiment-v1"
MODEL = "nvidia/Gemma-4-31B-IT-NVFP4"
REVISION = "4135a98a9b728a548947683219633b25682223ac"
CPU_COPY_SHA = "e3b93f048645d6145f362430513f99b60531c8e585e598027d2abb11ac2900e3"
EXPERIMENT_PLAN_SHA = "77329e75e5f4c789542639a1fa2b9f52b9ff8236c0147d8f8b987a779bc0368d"
CUDA13_ROOT = "/home/q/miniconda3/envs/qwen_next/lib/python3.12/site-packages/nvidia/cu13"
CUDA13_VIEW = "/mnt/sda/q/cvpr2027_benchmark/runs/reproducibility/operator/nvfp4-cuda13-toolchain-view"
CUDA_DRIVER_LINK = "/usr/lib/x86_64-linux-gnu/libcuda.so"
NINJA_SOURCE = "/home/q/miniconda3/envs/qwen_next/bin/ninja"
NINJA_VIEW = "/mnt/sda/q/cvpr2027_benchmark/runs/reproducibility/operator/nvfp4-ninja-bin"
ORIGINAL_TOOL_PATH = "/home/q/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"
COMPILER_ENV = {"CUDA_HOME": CUDA13_VIEW, "FLASHINFER_NVCC": CUDA13_ROOT + "/bin/nvcc",
                "FLASHINFER_CUDA_ARCH_LIST": "12.0f", "PATH": NINJA_VIEW + ":" + ORIGINAL_TOOL_PATH,
                "MAX_JOBS": "2"}
TOOLCHAIN_FILES = ("bin/nvcc", "bin/ptxas", "bin/fatbinary", "include/cuda.h", "include/cuda_runtime.h",
    "include/crt/host_config.h", "nvvm/bin/cicc", "nvvm/libdevice/libdevice.10.bc", "lib/libcudart.so.13")
OFFICIAL_ASSETS = {
    "config.json": "aa03a6a490fb743b8186f09c60ec39a30fdbaf7c3a18dfef8e52a62c2beb9ae4",
    "hf_quant_config.json": "7295b239347a50abc3888c4a8208b8644eb0bd3d92810117bdea1768e5f21129",
    "model.safetensors.index.json": "aff5569bed013db014afe23db3986434729bb33d9f817017cc0617cd479212b9",
    "processor_config.json": "32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c",
    "tokenizer.json": "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f",
    "tokenizer_config.json": "e467669cfe172dfb0c4e7de7bfbe7553c42bfa5de95acd71f423f58a434d80de",
    "chat_template.jinja": "94899c0f917d93f6fe81c95744d1e8ddab2d21d39228d2e4aec1fb2a25bff413",
    "generation_config.json": "d4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de",
}
SDK_FILES = (
    "vllm/model_executor/layers/quantization/modelopt.py",
    "vllm/model_executor/layers/quantization/kv_cache.py",
    "vllm/model_executor/layers/quantization/utils/nvfp4_utils.py",
    "vllm/v1/worker/gpu/model_runner.py",
    "vllm/model_executor/kernels/linear/nvfp4/base.py",
    "vllm/model_executor/kernels/linear/nvfp4/b12x.py",
    "vllm/model_executor/kernels/linear/nvfp4/cutlass.py",
    "vllm/model_executor/kernels/linear/nvfp4/flashinfer.py",
)
_RUNTIME, _SETTINGS, _PLACEMENT = g.runtime_check, g.engine_settings, g._placement
_PROCESSOR = inference.ProcessorSession
MLP = re.compile(r"language_model\.model\.layers\.(\d+)\.mlp\.(gate_up_proj|down_proj)\.([^.]*)$")
SCALE_NAMES = {"weight_global_scale", "input_global_scale", "input_global_scale_inv", "alpha"}


def validate_sources():
    c.validate_sources()
    g.require(g.file_sha256(c.__file__) == CPU_COPY_SHA, "Frozen C3 CPU-copy source differs")


def sdk_sources():
    dist = importlib.metadata.distribution("vllm")
    return {name: g.file_sha256(dist.locate_file(name)) for name in SDK_FILES}


def quantization_contract(config, separate):
    """Validate the published W4A4 algorithm and its exact exclusion policy."""
    q = config["quantization_config"]
    excludes = {"lm_head", "model.embed_vision*", "model.vision_tower*"}
    excludes.update(f"model.language_model.layers.{i}.self_attn*" for i in range(60))
    spec = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}
    g.require(q["quant_method"] == "modelopt" and q["quant_algo"] == "NVFP4"
              and q["config_groups"] == {"group_0": {"input_activations": spec,
                  "weights": spec, "targets": ["Linear"]}}
              and len(q["ignore"]) == len(excludes) and set(q["ignore"]) == excludes,
              "Official NVFP4 W4A4/exclusion contract differs")
    g.require(q["kv_cache_scheme"] == {"dynamic": False, "num_bits": 8, "type": "float"},
              "Published KV scheme differs; actual experiment still requires BF16 KV")
    g.require(separate["quantization"] == {"quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8",
                  "group_size": 16, "exclude_modules": q["ignore"]}
              and separate["producer"] == q["producer"] == {"name": "modelopt", "version": "0.37.0"},
              "Official quantization files disagree")
    g.require(config["architectures"] == ["Gemma4ForConditionalGeneration"]
              and config["text_config"]["num_hidden_layers"] == 60
              and config["text_config"]["hidden_size"] == 5376
              and config["text_config"]["intermediate_size"] == 21504,
              "Gemma architecture differs")
    return {"algorithm": "NVFP4-W4A4", "group_size": 16,
        "quantized_unfused_MLP_linears": 180, "runtime_fused_MLP_linears": 120,
        "official_KV_scheme": "FP8", "experiment_KV_dtype": "bfloat16",
        "config_producer": q["producer"], "model_card_producer_version": "0.42.0",
        "method": "PTQ according to model card; no QAD claim",
        "base_model": "google/gemma-4-31B-it", "published_base_commit": None,
        "base_commit_limit": "Publisher does not specify the exact BF16 source revision"}


def architecture_contract(original, quantized):
    """Permit only the observed serialization defaults and quantization block."""
    a, b = deepcopy(original), deepcopy(quantized)
    b.pop("quantization_config")
    defaults = {"_name_or_path": "", "architectures": None, "chunk_size_feed_forward": 0,
        "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "is_encoder_decoder": False,
        "label2id": {"LABEL_0": 0, "LABEL_1": 1}, "output_attentions": False,
        "output_hidden_states": False, "problem_type": None, "return_dict": True}
    for key, value in defaults.items():
        g.require(a["vision_config"].pop(key, None) == value and key not in b["vision_config"],
                  "Unexpected original vision serialization default: " + key)
    for key, value in {"image_seq_length": 280, "video_seq_length": 70}.items():
        g.require(b["vision_config"].pop(key) == value and key not in a["vision_config"],
                  "Unexpected quantized vision token default: " + key)
    g.require(a == b, "Architecture differs beyond the recorded serialization defaults")
    return {"matched_after_documented_serialization_defaults": True,
            "original_vision_defaults": defaults, "quantized_vision_defaults": {"image_seq_length": 280, "video_seq_length": 70}}


def checkpoint_contract(checkpoint, processor_checkpoint):
    checkpoint, original = Path(checkpoint), Path(processor_checkpoint)
    g.require(checkpoint.name == REVISION and original.name == g.REVISION,
              "Exact quantized and original processor revisions required")
    observed = {name: g.file_sha256(checkpoint / name) for name in OFFICIAL_ASSETS}
    g.require(observed == OFFICIAL_ASSETS, "Published NVFP4 metadata bytes differ")
    contract = quantization_contract(g.read(checkpoint / "config.json"), g.read(checkpoint / "hf_quant_config.json"))
    contract["architecture"] = architecture_contract(g.read(original / "config.json"), g.read(checkpoint / "config.json"))
    same = ("processor_config.json", "tokenizer.json", "generation_config.json")
    for name in same:
        g.require(g.file_sha256(original / name) == observed[name], "Original input asset differs: " + name)
    old_tokenizer, new_tokenizer = g.read(original / "tokenizer_config.json"), g.read(checkpoint / "tokenizer_config.json")
    response_templates = {"original": old_tokenizer.pop("response_template", None),
                          "quantized": new_tokenizer.pop("response_template", None)}
    g.require(old_tokenizer == new_tokenizer, "Tokenizer config differs beyond response_template")
    contract.update(published_assets=observed, identical_input_assets=list(same),
        differing_unused_assets={"response_template": response_templates,
            "chat_template_sha256": {"original": g.file_sha256(original / "chat_template.jinja"),
                                     "quantized": observed["chat_template.jinja"]}},
        processor_and_tokenizer_policy="Explicit original BF16 checkpoint for ProcessorSession and vLLM tokenizer; no NVFP4 chat template")
    return contract


def compose_checkpoint(args):
    """Create one explicit config composition; never rewrite either checkpoint."""
    source, original, output = (Path(p).resolve() for p in
        (args.source_checkpoint, args.processor_checkpoint, args.output))
    g.require(output.name == REVISION and not output.exists(), "New composition directory must use the pinned NVFP4 revision")
    contract = checkpoint_contract(source, original)
    config = g.read(original / "config.json")
    config["quantization_config"] = g.read(source / "config.json")["quantization_config"]
    input_names = ("generation_config.json", "processor_config.json", "preprocessor_config.json",
        "video_preprocessor_config.json", "tokenizer.json", "tokenizer.model", "tokenizer_config.json",
        "chat_template.jinja", "special_tokens_map.json")
    links = {name: str(original / name) for name in input_names if (original / name).is_file()}
    links["hf_quant_config.json"] = str(source / "hf_quant_config.json")
    hardlinks = {}
    weight_names = set(g.read(source / "model.safetensors.index.json")["weight_map"].values())
    for name in sorted(weight_names | {"model.safetensors.index.json"}):
        g.require(Path(name).name == name and (source / name).is_file(), "Unsafe or missing source shard")
        target = (source / name).resolve()
        stat = target.stat()
        hardlinks[name] = {"path": str(target), "st_dev": stat.st_dev,
                           "st_ino": stat.st_ino, "bytes": stat.st_size}
    manifest = {"schema": SCHEMA, "source_checkpoint": str(source), "processor_checkpoint": str(original),
        "source_contract": contract, "effective_config_sha256": g.digest(config), "symlink_targets": links,
        "hardlink_targets": hardlinks,
        "effective_config_policy": "Original BF16 config unchanged except official quantization_config"}
    output.mkdir(parents=True)
    try:
        g.seal(output / "config.json", config)
        for name, target in links.items():
            os.symlink(target, output / name)
        for name, target in hardlinks.items():
            g.require(output.stat().st_dev == target["st_dev"], "Checkpoint hardlinks require the same filesystem; no copying fallback")
            os.link(target["path"], output / name)
        g.seal(output / "nvfp4-composition.json", manifest)
        effective_checkpoint_contract(output, original)
    except BaseException:
        # Leave an incomplete, explicitly new composition for diagnosis; never
        # remove original inputs or weight links by a broad cleanup operation.
        raise
    return {"status": "COMPOSED", "checkpoint": str(output), "GPU_calls": 0,
        "effective_config_sha256": manifest["effective_config_sha256"], "symlink_count": len(links),
        "hardlink_count": len(hardlinks), "duplicated_weight_bytes": 0}


def effective_checkpoint_contract(checkpoint, processor_checkpoint):
    path, original = Path(checkpoint), Path(processor_checkpoint)
    manifest = g.read(path / "nvfp4-composition.json")
    g.require(path.name == REVISION and manifest["schema"] == SCHEMA
              and Path(manifest["processor_checkpoint"]).resolve() == original.resolve(), "Composition identity differs")
    source = Path(manifest["source_checkpoint"])
    source_contract = checkpoint_contract(source, original)
    g.require(manifest["source_contract"] == source_contract, "Official source metadata changed after composition")
    expected = g.read(original / "config.json")
    expected["quantization_config"] = g.read(source / "config.json")["quantization_config"]
    g.require(g.read(path / "config.json") == expected
              and manifest["effective_config_sha256"] == g.digest(expected), "Effective model configuration drifted")
    links, hardlinks = manifest["symlink_targets"], manifest["hardlink_targets"]
    g.require(not set(links) & set(hardlinks), "Composition link kinds overlap")
    g.require(set(p.name for p in path.iterdir()) == set(links) | set(hardlinks) | {"config.json", "nvfp4-composition.json"},
              "Composition has extra or missing assets")
    original_names = set(inference.checkpoint_asset_identity(original)["processor_tokenizer_config_and_code"])
    expected_links = {name: str((original / name).resolve()) for name in original_names
                      if name not in {"config.json", "model.safetensors.index.json"}}
    expected_links["hf_quant_config.json"] = str((source / "hf_quant_config.json").resolve())
    g.require({name: str(Path(target).resolve()) for name, target in links.items()} == expected_links,
              "Composition links differ from explicit original inputs and official weights")
    for name, target in links.items():
        g.require(Path(name).name == name and (path / name).is_symlink()
                  and (path / name).resolve() == Path(target).resolve() and (path / name).is_file(),
                  "Composition link missing or retargeted: " + name)
    expected_hardlinks = {"model.safetensors.index.json"} | set(
        g.read(source / "model.safetensors.index.json")["weight_map"].values())
    g.require(set(hardlinks) == expected_hardlinks, "Composition hardlink names differ")
    for name, binding in hardlinks.items():
        target = (source / name).resolve()
        original_stat, actual_stat = target.stat(), (path / name).stat()
        expected_binding = {"path": str(target), "st_dev": original_stat.st_dev,
            "st_ino": original_stat.st_ino, "bytes": original_stat.st_size}
        g.require(Path(name).name == name and binding == expected_binding
                  and (path / name).is_file() and not (path / name).is_symlink()
                  and (actual_stat.st_dev, actual_stat.st_ino, actual_stat.st_size) ==
                      (original_stat.st_dev, original_stat.st_ino, original_stat.st_size),
                  "Composition hardlink identity differs: " + name)
    # This is the original asset guard, including its path-containment check.
    # Hardlinked shards resolve inside this composition and share source bytes.
    inference.checkpoint_asset_identity(path)
    return {"source_contract": source_contract, "composition_manifest_sha256": g.file_sha256(path / "nvfp4-composition.json"),
        "effective_config_sha256": g.digest(expected), "source_checkpoint": str(source),
        "processor_checkpoint": str(original), "effective_config_policy": manifest["effective_config_policy"]}


def engine_settings(checkpoint, processor_checkpoint, placement):
    g.require(placement in ("copy8", "full-gpu"), "Unsupported placement candidate")
    settings = _SETTINGS(checkpoint)
    settings.update(quantization="modelopt", tokenizer=str(processor_checkpoint), tensor_parallel_size=1)
    if placement == "full-gpu":
        settings.update(cpu_offload_gb=0, cpu_offload_params=[])
    return settings


def runtime_check(runtime):
    g.require(Path(runtime["checkpoint"]).name == REVISION, "Quantized checkpoint revision differs")
    environment = dict(runtime["environment"])
    for key, value in COMPILER_ENV.items():
        g.require(environment.pop(key, None) == value and os.environ.get(key) == value,
                  "Explicit NVFP4 compiler environment differs: " + key)
    original = dict(runtime, checkpoint=runtime["processor_checkpoint"], environment=environment)
    return _RUNTIME(original)


def _toolchain_view_record():
    root = Path(CUDA13_ROOT)
    g.require(all((root / name).is_file() for name in TOOLCHAIN_FILES), "Existing CUDA13 toolkit components are missing")
    links = {name: str(root / name) for name in ("bin", "include", "nvvm")}
    links.update({"lib64/" + name: str(root / "lib/libcudart.so.13") for name in ("libcudart.so", "libcudart.so.13")})
    targets = {}
    for name, target in links.items():
        path = Path(target).resolve()
        stat = path.stat()
        targets[name] = {"target": str(path), "st_dev": stat.st_dev, "st_ino": stat.st_ino,
                         "kind": "directory" if path.is_dir() else "file"}
    files = {}
    for name in TOOLCHAIN_FILES:
        path = (root / name).resolve()
        stat = path.stat()
        files[name] = {"path": str(path), "st_dev": stat.st_dev, "st_ino": stat.st_ino,
                       "bytes": stat.st_size, "sha256": g.file_sha256(path)}
    return {"schema": "gemma-nvfp4-owned-toolchain-view-v1", "root": CUDA13_VIEW,
        "source_root": CUDA13_ROOT, "symlink_targets": targets, "required_runtime_files": files,
        "system_driver_link": CUDA_DRIVER_LINK,
        "scope": "Native build layout for existing CUDA13 compiler/runtime; no copied libraries or SDK modifications"}


def create_toolchain_view(args=None):
    """Create only the authorized bounded symlink view of the installed toolkit."""
    view = Path(CUDA13_VIEW)
    g.require(not view.exists(), "Toolchain view exists; validate it rather than overwrite")
    record = _toolchain_view_record()
    g.require(Path(record["system_driver_link"]).is_file(), "Existing system CUDA driver linker name missing")
    view.mkdir()
    (view / "lib64").mkdir()
    for name, item in record["symlink_targets"].items():
        os.symlink(item["target"], view / name)
    g.seal(view / "nvfp4-toolchain-view.json", record)
    verify_toolchain_view()
    return {"status": "CREATED", "view": str(view), "symlink_count": len(record["symlink_targets"]),
            "copied_library_bytes": 0, "GPU_calls": 0}


def verify_toolchain_view():
    view = Path(CUDA13_VIEW)
    record = g.read(view / "nvfp4-toolchain-view.json")
    g.require(record == _toolchain_view_record(), "Owned CUDA13 view source identity changed")
    g.require(not view.is_symlink() and not (view / "lib64").is_symlink()
              and {p.name for p in view.iterdir()} == {"bin", "include", "nvvm", "lib64", "nvfp4-toolchain-view.json"}
              and {p.name for p in (view / "lib64").iterdir()} == {"libcudart.so", "libcudart.so.13"},
              "Owned CUDA13 view has unexpected entries")
    for name, item in record["symlink_targets"].items():
        path = view / name
        g.require(path.is_symlink() and str(path.resolve()) == item["target"]
                  and (path.stat().st_dev, path.stat().st_ino) == (item["st_dev"], item["st_ino"]),
                  "Owned CUDA13 view link retargeted: " + name)
    return record


def compiler_identity():
    """Inspect the already installed compiler without importing CUDA runtimes."""
    root = Path(CUDA13_ROOT)
    view_record = verify_toolchain_view()
    ninja_record = verify_ninja_view()
    result = subprocess.run([COMPILER_ENV["FLASHINFER_NVCC"], "--version"],
                            capture_output=True, text=True, check=True, timeout=15)
    g.require("release 13.0, V13.0.88" in result.stdout, "Pinned CUDA13 compiler version differs")
    dist = importlib.metadata.distribution("flashinfer-python")
    sources = ("flashinfer/compilation_context.py", "flashinfer/jit/cpp_ext.py", "flashinfer/jit/gemm/core.py")
    return {"schema": "gemma-nvfp4-process-compiler-v1", "environment": dict(COMPILER_ENV),
        "nvcc_version_output": result.stdout.strip(), "nvcc_sha256": g.file_sha256(root / "bin/nvcc"),
        "toolchain_view_manifest_sha256": g.file_sha256(Path(CUDA13_VIEW) / "nvfp4-toolchain-view.json"),
        "ninja_view_manifest_sha256": g.file_sha256(Path(NINJA_VIEW) / "nvfp4-ninja-view.json"),
        "ninja": ninja_record,
        "toolkit_components": view_record["required_runtime_files"],
        "flashinfer_sources": {name: g.file_sha256(dist.locate_file(name)) for name in sources},
        "scope": "Existing toolkit and Ninja, five explicit build environment keys including MAX_JOBS=2; no package or SDK changes"}


def _ninja_record():
    path = Path(NINJA_SOURCE).resolve()
    stat = path.stat()
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=True, timeout=10)
    g.require(result.stdout.strip() == "1.13.2.git.kitware.jobserver-pipe-1", "Existing pinned Ninja version differs")
    return {"schema": "gemma-nvfp4-owned-ninja-view-v1", "view": NINJA_VIEW, "target": str(path),
        "st_dev": stat.st_dev, "st_ino": stat.st_ino, "bytes": stat.st_size, "sha256": g.file_sha256(path),
        "version": result.stdout.strip(), "original_tool_path": ORIGINAL_TOOL_PATH,
        "scope": "One existing Ninja binary, linked into an owned PATH prefix; no installation"}


def create_ninja_view(args=None):
    view = Path(NINJA_VIEW)
    g.require(not view.exists(), "Ninja view exists; validate it rather than overwrite")
    record = _ninja_record()
    view.mkdir()
    os.symlink(record["target"], view / "ninja")
    g.seal(view / "nvfp4-ninja-view.json", record)
    verify_ninja_view()
    return {"status": "CREATED", "view": str(view), "symlink_count": 1, "copied_binary_bytes": 0, "GPU_calls": 0}


def verify_ninja_view():
    view = Path(NINJA_VIEW)
    record = g.read(view / "nvfp4-ninja-view.json")
    g.require(record == _ninja_record() and not view.is_symlink()
              and {p.name for p in view.iterdir()} == {"ninja", "nvfp4-ninja-view.json"}, "Owned Ninja view identity differs")
    path = view / "ninja"
    g.require(path.is_symlink() and str(path.resolve()) == record["target"]
              and (path.stat().st_dev, path.stat().st_ino) == (record["st_dev"], record["st_ino"]),
              "Owned Ninja link retargeted")
    g.require(shutil.which("ninja", path=COMPILER_ENV["PATH"]) == str(path), "Controlled PATH does not resolve owned Ninja")
    return record


def parameter_contract(rows, offload_gb):
    """Pure CPU validation of runtime parameter metadata; no weight values read."""
    g.require(offload_gb in (0, 8), "Unsealed offload budget")
    names = [r["name"] for r in rows]
    g.require(len(names) == len(set(names)) and rows, "Missing or duplicate model parameters")
    groups, places, offloaded, eligible = {}, Counter(), [], 0
    for row in rows:
        name, dtype, place = row["name"], row["dtype"], row["place"]
        g.require(place in ("cuda", "cpu"), "UVA/meta/other parameter placement is forbidden")
        places[name.split(".")[0] + ":" + place] += row["bytes"]
        match = MLP.fullmatch(name)
        if match:
            layer, kind, field = match.groups()
            g.require(0 <= int(layer) < 60, "Unexpected MLP layer")
            fields = groups.setdefault((int(layer), kind), {})
            fields[field] = row
            expected = "torch.uint8" if field == "weight" else (
                "torch.float8_e4m3fn" if field == "weight_scale" else "torch.float32")
            g.require(field in {"weight", "weight_scale"} | SCALE_NAMES and dtype == expected,
                      "Unexpected NVFP4 parameter dtype/name: " + name)
            nout, nin = (43008, 5376) if kind == "gate_up_proj" else (5376, 21504)
            expected_numel = nout * nin // (2 if field == "weight" else 16) if field in ("weight", "weight_scale") else 1
            g.require(row["numel"] == expected_numel, "Packed/scaled NVFP4 parameter size differs: " + name)
            if kind == "gate_up_proj":
                eligible += row["bytes"]
        else:
            g.require(dtype == "torch.bfloat16", "Unquantized model parameter is not BF16: " + name)
        if place == "cpu":
            g.require(match is not None and match.group(2) == "gate_up_proj", "Only language gate_up parameters may be offloaded")
            offloaded.append(name)
    expected_groups = {(i, kind) for i in range(60) for kind in ("gate_up_proj", "down_proj")}
    g.require(set(groups) == expected_groups, "Expected all 120 quantized fused MLP modules")
    g.require(all(set(fields) == {"weight", "weight_scale"} | SCALE_NAMES for fields in groups.values()),
              "Missing native NVFP4 weight/scale parameters")
    cpu_bytes = sum(r["bytes"] for r in rows if r["place"] == "cpu")
    if offload_gb:
        g.require(cpu_bytes >= min(offload_gb * 2**30, eligible), "Eligible packed parameter offload is incomplete")
    else:
        g.require(cpu_bytes == 0, "Full-GPU candidate has CPU parameters")
    return {"placement_bytes": dict(places), "offloaded_names": offloaded,
        "observed_cpu_parameter_bytes": cpu_bytes, "eligible_gate_up_parameter_bytes": eligible,
        "parameter_dtype_counts": dict(Counter(r["dtype"] for r in rows)),
        "runtime_quantized_MLP_modules": len(groups), "parameter_metadata_sha256": g.digest(rows),
        "nvfp4_weight_and_scale_layout": [r for r in rows if MLP.fullmatch(r["name"])
            and MLP.fullmatch(r["name"]).group(3) in ("weight", "weight_scale")]}


def _placement(worker, offload_gb):
    import torch
    from vllm.model_executor.offloader.base import get_offloader
    from vllm.v1.core.kv_cache_utils import get_kv_cache_capacity
    model, runner = worker.get_model(), worker.model_runner
    rows = [{"name": name, "dtype": str(p.dtype), "shape": list(p.shape), "numel": p.numel(),
             "bytes": p.numel() * p.element_size(),
             "place": "uva" if getattr(p, "_vllm_is_uva_offloaded", False) else p.device.type}
            for name, p in model.named_parameters()]
    result = parameter_contract(rows, offload_gb)
    kernels = []
    for name, module in model.named_modules():
        if MLP.fullmatch(name + ".weight"):
            method = module.quant_method
            g.require(type(method).__name__ == "ModelOptLinearMethod" and method.ctx.group_size == 16
                      and method.spec.activation is not None, "Expected native ModelOpt W4A4 linear method")
            kernel = type(method.kernel).__name__
            g.require("NvFp4" in kernel and not any(s in kernel for s in ("W4A16", "Marlin", "Emulation", "Pytorch")),
                      "Expected native NVFP4 W4A4 kernel, got " + kernel)
            kernels.append({"module": name, "method": type(method).__name__, "kernel": kernel,
                            "weight_key": str(method.spec.weight), "activation_key": str(method.spec.activation)})
    g.require(len(kernels) == 120, "Missing actual NVFP4 execution methods")
    quant = worker.vllm_config.quant_config
    g.require(quant.quant_method == "NVFP4" and quant.group_size == 16, "Actual quantization config differs")
    g.require(runner.kv_cache_dtype == torch.bfloat16
              and worker.vllm_config.cache_config.cache_dtype == "bfloat16", "Actual KV cache must remain BF16")
    offloader = get_offloader()
    observed = getattr(offloader, "cpu_offload_bytes", 0)
    g.require(observed == result["observed_cpu_parameter_bytes"], "Offloader accounting differs from actual CPU parameters")
    g.require(not getattr(offloader, "uva_offloading", False), "UVA is forbidden")
    cache = runner.kv_cache_config
    capacity, concurrency = get_kv_cache_capacity(worker.vllm_config, cache)
    g.require(capacity >= 65536, "Actual native KV capacity below full context")
    result.update(worker=g.process_identity(), uva=getattr(offloader, "uva_offloading", None),
        offload_bytes=observed, declared_offload_gb=offload_gb, quantized_kernels=kernels,
        kv={"capacity_tokens": int(capacity), "max_concurrency": float(concurrency),
            "num_gpu_blocks": cache.num_blocks, "actual_dtype": str(runner.kv_cache_dtype),
            "helper_source_sha256": g.file_sha256(inspect.getsourcefile(get_kv_cache_capacity))},
        torch_policy={"TF32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "TF32_cudnn": torch.backends.cudnn.allow_tf32, "matmul_precision": torch.get_float32_matmul_precision()})
    return result


def _baseline(profile):
    original = deepcopy(profile)
    overlay = original.pop("nvfp4_overlay")
    original.update(deepcopy(overlay["original_fields"]))
    g.require(g.digest(original) == overlay["baseline_profile_sha256"], "Original C3 profile binding differs")
    return original


def experiment_plan_contract(path, profile):
    g.require(g.file_sha256(path) == EXPERIMENT_PLAN_SHA, "Frozen finite NVFP4 experiment plan differs")
    plan = g.read(path)
    gemma, count = plan["gemma"], profile["maximum_model_calls"]
    budget = gemma["budget"]
    g.require(gemma["nvfp4_checkpoint_revision"] == REVISION and budget["model_calls_cap"] == 136
              and budget["speed_calls"] == 40 and budget["cold_repro_calls"] == 96,
              "NVFP4 experiment budget differs")
    section = gemma["speed_panel"] if count == 20 else gemma["cold_reproducibility"]
    g.require(count in (20, 48) and section["sample_ids"] == [x["sample_id"] for x in profile["panel"]["items"]],
              "Profile differs from the finite experiment panel")
    return {"file_sha256": EXPERIMENT_PLAN_SHA, "plan_sha256": g.digest(plan),
        "model_calls_cap": 136, "speed_profiles": {"O8": 20, "O0": 20}, "cold_profiles": {"cold0": 48, "cold1": 48}}


def admission_contract(admission, profile):
    overlay = profile["nvfp4_overlay"]
    g.require(admission.get("nvfp4_authorized") is True
              and admission.get("nvfp4_plan_file_sha256") == EXPERIMENT_PLAN_SHA
              and admission.get("nvfp4_profile_sha256") == g.digest(profile)
              and admission.get("nvfp4_model_calls_cap") == 136,
              "Fresh finite NVFP4 controller admission is required")
    role = admission.get("nvfp4_observation_role")
    count = profile["maximum_model_calls"]
    if count == 20:
        g.require(role == ("O8" if overlay["placement"] == "copy8" else "O0"), "Speed observation role differs")
    else:
        g.require(count == 48 and role in ("cold0", "cold1"), "Cold observation role differs")
    return role


def validate_profile(profile, installed_sdk=True):
    validate_sources()
    original = _baseline(profile)
    c.validate_profile(original, installed_sdk=installed_sdk)
    overlay = profile["nvfp4_overlay"]
    g.require(overlay["schema"] == SCHEMA and overlay["source_sha256"] == g.file_sha256(__file__), "NVFP4 entry source changed")
    g.require(original["cpu_execution_overlay"]["placement"] == "baseline"
              and original["cpu_execution_overlay"]["gc_cadence"] == 8
              and original["kernel"]["range_lookup"] == "binary" and original["kernel"]["prefill_block_m"] == 32,
              "The starting profile must be the qualified C3 copy8/GC8/binary/M32")
    runtime = profile["runtime"]
    expected_runtime = dict(original["runtime"], checkpoint=runtime["checkpoint"], processor_checkpoint=original["runtime"]["checkpoint"])
    expected_runtime["environment"] = dict(original["runtime"]["environment"], **COMPILER_ENV)
    g.require(runtime == expected_runtime, "NVFP4 runtime changed beyond checkpoint, original processor and five compiler keys")
    g.require(overlay["compiler_execution_overlay"]["environment"] == COMPILER_ENV,
              "Compiler execution overlay differs")
    g.require(profile["engine_settings"] == engine_settings(runtime["checkpoint"], runtime["processor_checkpoint"], overlay["placement"]),
              "NVFP4 engine settings differ")
    g.require(profile["maximum_model_calls"] in (20, 48) and profile["automatic_retry"] is False,
              "Finite experiment budget differs")
    g.require(overlay["experiment_plan"]["file_sha256"] == EXPERIMENT_PLAN_SHA
              and overlay["experiment_plan"]["model_calls_cap"] == 136, "Finite experiment plan binding differs")
    if installed_sdk:
        runtime_check(runtime)
        g.require(overlay["compiler_execution_overlay"] == compiler_identity(), "Existing compiler or FlashInfer source identity changed")
        g.require(overlay["quantization_sdk_sources"] == sdk_sources(), "Quantization SDK bytes changed")
        g.require(overlay["checkpoint_contract"] == effective_checkpoint_contract(runtime["checkpoint"], runtime["processor_checkpoint"]),
                  "Checkpoint or explicit original input assets changed")
        g.require(profile["checkpoint_assets"] == inference.checkpoint_asset_identity(runtime["checkpoint"]), "Quantized assets changed")
        g.require(original["checkpoint_assets"] == inference.checkpoint_asset_identity(runtime["processor_checkpoint"]), "Original processor assets changed")
    return profile


def seal_profile(args):
    validate_sources()
    original = c.validate_profile(g.read(args.baseline_profile))
    candidate_runtime = dict(original["runtime"], checkpoint=args.checkpoint,
        processor_checkpoint=original["runtime"]["checkpoint"],
        environment=dict(original["runtime"]["environment"], **COMPILER_ENV))
    runtime_check(candidate_runtime)
    profile = deepcopy(original)
    original_fields = {key: deepcopy(original[key]) for key in ("name", "runtime", "checkpoint_assets", "engine_settings")}
    runtime = profile["runtime"]
    runtime["processor_checkpoint"], runtime["checkpoint"] = original["runtime"]["checkpoint"], str(args.checkpoint)
    runtime["environment"] = dict(runtime["environment"], **COMPILER_ENV)
    profile.update(name=args.name, checkpoint_assets=inference.checkpoint_asset_identity(args.checkpoint),
        engine_settings=engine_settings(args.checkpoint, runtime["processor_checkpoint"], args.placement))
    profile["nvfp4_overlay"] = {"schema": SCHEMA, "module": "classic_precision.nvfp4_gemma",
        "source_sha256": g.file_sha256(__file__), "model": MODEL, "revision": REVISION,
        "baseline_profile_file_sha256": g.file_sha256(args.baseline_profile),
        "baseline_profile_sha256": g.digest(original), "original_fields": original_fields,
        "placement": args.placement, "quantization_sdk_sources": sdk_sources(),
        "compiler_execution_overlay": compiler_identity(),
        "checkpoint_contract": effective_checkpoint_contract(args.checkpoint, runtime["processor_checkpoint"]),
        "comparison_policy": "BF16 differences are measured; NVFP4 cold repeats require identical full tokens",
        "additional_execution_delta": None if args.placement == "copy8" else "cpu_offload_gb 8 -> 0; no CPU gate_up copies",
        "experiment_plan": experiment_plan_contract(args.experiment_plan, profile),
        "input_reference_file_sha256": c.m.REFERENCE_SHA[profile["maximum_model_calls"]]}
    validate_profile(profile)
    g.seal(args.output, profile)
    return {"status": "SEALED", "profile_sha256": g.digest(profile), "calls": profile["maximum_model_calls"], "GPU_calls": 0}


@contextmanager
def installed_overlay(profile):
    g.require((g.runtime_check, g.engine_settings, g._placement, inference.ProcessorSession) ==
              (_RUNTIME, _SETTINGS, _PLACEMENT, _PROCESSOR), "Another precision overlay is already installed")
    runtime, overlay = profile["runtime"], profile["nvfp4_overlay"]
    def original_processor(family, model_path, *args, **kwargs):
        g.require(family == "gemma" and str(model_path) == runtime["checkpoint"], "Unexpected processor request")
        return _PROCESSOR(family, runtime["processor_checkpoint"], *args, **kwargs)
    g.runtime_check = runtime_check
    g.engine_settings = partial(engine_settings, processor_checkpoint=runtime["processor_checkpoint"], placement=overlay["placement"])
    g._placement = partial(_placement, offload_gb=profile["engine_settings"]["cpu_offload_gb"])
    inference.ProcessorSession = original_processor
    try:
        yield
    finally:
        g.runtime_check, g.engine_settings, g._placement, inference.ProcessorSession = _RUNTIME, _SETTINGS, _PLACEMENT, _PROCESSOR


def run(args):
    g.require("torch" not in sys.modules and "vllm" not in sys.modules, "A fresh process is required")
    profile = validate_profile(g.read(args.profile))
    admission_contract(g.read(args.admission), profile)
    g.require(g.file_sha256(args.reference) == profile["nvfp4_overlay"]["input_reference_file_sha256"], "Use the frozen shared CPU reference")
    output, state, completed = Path(args.output), None, False
    g.require(not output.exists(), "No implicit resume or overwrite")
    try:
        with installed_overlay(profile), c.installed_overlay(profile) as state:
            result = g.run(args)
        completed = True
        return result
    finally:
        if state is not None and output.is_dir():
            g.seal(output / "nvfp4-execution.json", {"schema": SCHEMA, "status": "COMPLETE" if completed else "STOPPED",
                "profile_sha256": g.digest(profile), "source_sha256": g.file_sha256(__file__),
                "placement": profile["nvfp4_overlay"]["placement"], "gc_cadence": state.cadence,
                "maximum_pending_ordinary": state.maximum_pending, "rows": state.rows, "final_gc": state.final,
                "unchanged_frozen_worker": g.file_sha256(g.__file__), "engine_exit_receipt": "exit.json"})


def _records(root):
    return [json.loads(line) for line in (Path(root) / "raw.jsonl").read_text().splitlines() if line.strip()]


def compare_runs(paths, purpose):
    g.require(purpose in ("precision", "placement", "repeat"), "Unknown comparison purpose")
    g.require(len(paths) == 2, "Exactly two observations are required")
    result = g.compare_runs(paths, purpose="candidate")
    identities = [g.read(Path(p) / "identity.json") for p in paths]
    profiles = [x["profile"] for x in identities]
    loads = [g.read(Path(p) / "load.json") for p in paths]
    g.require(all(identity["profile_sha256"] == g.digest(profile)
                  for identity, profile in zip(identities, profiles, strict=True)), "Stored profile digest differs")
    g.require((identities[0]["host"], identities[0]["lease"]["device"]) ==
              (identities[1]["host"], identities[1]["lease"]["device"]), "Comparison must use the same physical GPU")
    g.require(loads[0]["placement"][0]["torch_policy"] == loads[1]["placement"][0]["torch_policy"],
              "Torch runtime policies differ")
    expected = 48 if purpose == "repeat" else 20
    g.require(result["unique_ids"] == expected, "Wrong finite comparison panel")
    selected = profiles[1:] if purpose == "precision" else profiles
    for path, profile in zip(paths[1:] if purpose == "precision" else paths, selected, strict=True):
        validate_profile(profile, installed_sdk=False)
        receipt = g.read(Path(path) / "nvfp4-execution.json")
        g.require(receipt["status"] == "COMPLETE" and receipt["profile_sha256"] == g.digest(profile)
                  and receipt["source_sha256"] == profile["nvfp4_overlay"]["source_sha256"]
                  and receipt["final_gc"]["status"] == "COMPLETE" and receipt["gc_cadence"] == 8
                  and receipt["maximum_pending_ordinary"] <= 8
                  and [r["sample_id"] for r in receipt["rows"]] == [x["sample_id"] for x in profile["panel"]["items"]],
                  "NVFP4 execution/GC closure differs")
    if purpose == "repeat":
        g.require(g.digest(profiles[0]) == g.digest(profiles[1]), "Repeat profiles differ")
        g.require(identities[0]["engine_id"] != identities[1]["engine_id"]
                  and loads[0]["placement"][0]["worker"] != loads[1]["placement"][0]["worker"],
                  "Two independent cold engine identities are required")
    elif purpose == "precision":
        g.require("nvfp4_overlay" not in profiles[0]
                  and g.digest(profiles[0]) == profiles[1]["nvfp4_overlay"]["baseline_profile_sha256"],
                  "Precision comparison requires the exact recorded BF16 C3 baseline")
    else:
        g.require([p["nvfp4_overlay"]["placement"] for p in profiles] == ["copy8", "full-gpu"],
                  "Placement comparison requires ordered O8 and O0 candidates")
        g.require(g.digest(_baseline(profiles[0])) == g.digest(_baseline(profiles[1])), "Placement candidates have different BF16/input baselines")
        signatures = []
        for p in profiles:
            clean = deepcopy(p)
            clean.pop("name")
            o = clean["nvfp4_overlay"]
            for key in ("placement", "additional_execution_delta"):
                o.pop(key)
            for key in ("cpu_offload_gb", "cpu_offload_params"):
                clean["engine_settings"].pop(key)
            signatures.append(g.digest(clean))
        g.require(len(set(signatures)) == 1, "Placement candidates differ beyond offload settings and name")
    raw = [_records(p) for p in paths]
    extra_input_differences = []
    for a, b in zip(*raw, strict=True):
        fields = [key for key in ("processor_input_FP32_sha256", "pixel_FP32_sha256", "input_sha256", "projection_sha256")
                  if a["input_receipt"].get(key) != b["input_receipt"].get(key)]
        if fields:
            extra_input_differences.append({"sample_id": a["sample_id"], "fields": fields})
    result["input_differences"].extend(extra_input_differences)
    differences = []
    for a, b in zip(*raw, strict=True):
        changed = [k for k in ("output_token_ids", "raw_answer", "finish_reason", "stop_reason") if a[k] != b[k]]
        if changed:
            differences.append({"sample_id": a["sample_id"], "fields": changed,
                "first": {k: a[k] for k in changed}, "second": {k: b[k] for k in changed}})
    valid_inputs = not result["input_differences"]
    result.update(schema=SCHEMA, purpose=purpose, differences= differences,
        precision_difference_count=len(differences) if purpose == "precision" else None,
        two_cold_full_token_pass=purpose == "repeat" and valid_inputs and not differences,
        placement_full_token_pass=purpose == "placement" and valid_inputs and not differences,
        status=("INVALID_EVIDENCE" if not valid_inputs else
                ("COMPLETE_PRECISION_COMPARISON" if purpose == "precision" else ("MATCH" if not differences else "DIFFERENT"))),
        production_authorized_by_this_command=False)
    return result


def native_probe_spec(full_prefill_tokens):
    g.require(type(full_prefill_tokens) is int and 31680 <= full_prefill_tokens <= 34000,
              "Probe full-prefill M must come from the existing qualified prompt-token reference")
    return {"kernel": "FlashInferCutlassNvFp4LinearKernel", "backend": "cutlass",
        "quantization": "native NVFP4 W4A4, group16", "input_dtype": "torch.bfloat16",
        "output_dtype": "torch.bfloat16", "M": [8, full_prefill_tokens], "K": 5376, "N": 43008,
        "packed_weight_shape": [43008, 2688], "weight_scale_shape": [43008, 336],
        "maximum_synthetic_native_gemm_calls": 2, "scientific_model_calls": 0,
        "purpose": "Native GEMM initialization/compilation plumbing and two shape checks; not a model benchmark"}


def native_gemm_probe(args):
    """Explicit GPU command for the operator; never called by sealing or tests."""
    g.require("torch" not in sys.modules and "vllm" not in sys.modules, "Probe requires a fresh process")
    profile = validate_profile(g.read(args.profile))
    g.require(g.file_sha256(args.reference) == profile["nvfp4_overlay"]["input_reference_file_sha256"],
              "Use the existing frozen prompt-token reference for full-prefill M")
    reference = g.read(args.reference)
    g.require(reference["panel_sha256"] == g.digest(profile["panel"]), "Probe reference panel differs")
    sid = profile["panel"]["items"][0]["sample_id"]
    spec = native_probe_spec(len(reference["rows"][sid]["input_token_ids"]))
    output = Path(args.output)
    record = {"schema": "gemma-nvfp4-native-gemm-probe-v1", "status": "STARTED", "spec": spec,
        "source_sha256": g.file_sha256(__file__), "profile_sha256": g.digest(profile),
        "reference_sha256": g.file_sha256(args.reference), "reference_sample_id": sid,
        "owner": g.process_identity(), "compiler": profile["nvfp4_overlay"]["compiler_execution_overlay"],
        "attempted_synthetic_gemm_calls": 0, "completed_synthetic_gemm_calls": 0,
        "scientific_model_calls": 0, "processor_calls": 0, "model_weights_loaded": False, "rows": []}
    g.seal(output, record)
    started = time.monotonic()
    try:
        import torch
        torch.set_grad_enabled(False)
        with g.physical_lease(profile["runtime"], torch) as lease:
            record["lease"] = lease
            g.require(torch.cuda.get_device_capability(0) == (12, 0), "Probe is scoped to the qualified SM120 GPU")
            from vllm.model_executor.kernels.linear.nvfp4.base import NvFp4LinearLayerConfig
            from vllm.model_executor.kernels.linear.nvfp4.flashinfer import FlashInferCutlassNvFp4LinearKernel
            from flashinfer.jit.core import current_compilation_context
            record["native_arch_flags"] = current_compilation_context.get_nvcc_flags_list(supported_major_versions=[12])
            g.require(current_compilation_context.TARGET_CUDA_ARCHS == {(12, "0f")}, "Native FlashInfer target differs")
            kernel = FlashInferCutlassNvFp4LinearKernel(NvFp4LinearLayerConfig())
            layer = torch.nn.Module()
            layer.output_size_per_partition, layer.input_size_per_partition = spec["N"], spec["K"]
            layer.weight = torch.nn.Parameter(torch.full(tuple(spec["packed_weight_shape"]), 0x22,
                                                         dtype=torch.uint8, device="cuda"), requires_grad=False)
            layer.weight_scale = torch.nn.Parameter(torch.ones(tuple(spec["weight_scale_shape"]),
                dtype=torch.float32, device="cuda").to(torch.float8_e4m3fn), requires_grad=False)
            layer.input_global_scale_inv = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device="cuda"), requires_grad=False)
            layer.alpha = torch.nn.Parameter(torch.ones((), dtype=torch.float32, device="cuda"), requires_grad=False)
            kernel.process_weights_after_loading(layer)
            torch.cuda.synchronize()
            record["packed_layout"] = {"weight_shape": list(layer.weight.shape), "weight_dtype": str(layer.weight.dtype),
                "scale_shape": list(layer.weight_scale.shape), "scale_dtype": str(layer.weight_scale.dtype)}
            for i, m in enumerate(spec["M"]):
                x = torch.full((m, spec["K"]), .25, dtype=torch.bfloat16, device="cuda")
                torch.cuda.synchronize()
                began = time.monotonic()
                record["attempted_synthetic_gemm_calls"] += 1
                g.atomic_json(output, record)
                y = kernel.apply_weights(layer, x)
                torch.cuda.synchronize()
                elapsed = time.monotonic() - began
                g.require(y.shape == (m, spec["N"]) and y.dtype == torch.bfloat16
                          and bool(torch.isfinite(y).all().item()) and bool((y != 0).any().item()),
                          "Native W4A4 GEMM did not produce a finite nonzero BF16 matrix")
                # Rows are equal for these constant inputs. Small checksums
                # document execution without copying the multi-GB output to CPU.
                sample = torch.stack((y[0, :128], y[-1, -128:]))
                record["rows"].append({"M": m, "N": spec["N"], "K": spec["K"], "status": "PASS",
                    "wall_seconds": elapsed, "timing_scope": "cold initialization/JIT plus GEMM" if i == 0 else "cached module, full-prefill GEMM",
                    "output_shape": list(y.shape), "output_dtype": str(y.dtype), "sample_sha256": g.sha_tensor(sample),
                    "sample_min": float(sample.min().item()), "sample_max": float(sample.max().item()),
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
                record["completed_synthetic_gemm_calls"] += 1
                g.atomic_json(output, record)
                del x, y, sample
            del layer, kernel
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        record.update(status="COMPLETE", lease_released=True, total_wall_seconds=time.monotonic() - started)
        g.atomic_json(output, record)
        return {"status": "COMPLETE", "synthetic_native_gemm_calls": 2, "scientific_model_calls": 0, "output": str(output)}
    except BaseException as exc:
        record.update(status="STOPPED", error_type=type(exc).__name__, error=str(exc),
                      total_wall_seconds=time.monotonic() - started, automatic_retry=False)
        g.atomic_json(output, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("create-toolchain-view")
    commands.add_parser("create-ninja-view")
    p = commands.add_parser("native-gemm-probe")
    for name in ("profile", "reference", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("compose-checkpoint")
    for name in ("source-checkpoint", "processor-checkpoint", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("seal-profile")
    for name in ("baseline-profile", "checkpoint", "experiment-plan", "name", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--placement", choices=("copy8", "full-gpu"), default="copy8")
    p = commands.add_parser("run")
    for name in ("profile", "reference", "admission", "output"):
        p.add_argument("--" + name, required=True)
    p = commands.add_parser("compare")
    p.add_argument("runs", nargs=2)
    p.add_argument("--purpose", choices=("precision", "placement", "repeat"), required=True)
    p.add_argument("--output")
    args = parser.parse_args()
    if args.command == "create-toolchain-view":
        result = create_toolchain_view(args)
    elif args.command == "create-ninja-view":
        result = create_ninja_view(args)
    elif args.command == "native-gemm-probe":
        result = native_gemm_probe(args)
    elif args.command == "compose-checkpoint":
        result = compose_checkpoint(args)
    elif args.command == "seal-profile":
        result = seal_profile(args)
    elif args.command == "run":
        result = run(args)
    else:
        result = compare_runs(args.runs, args.purpose)
        if args.output:
            g.seal(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
