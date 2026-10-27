"""Finite TP2/two-request Gemma experiment; never a production qualification.

Two passes over eight fixed videos (16 requests), with a one-place rotation in
pass two. Native request IDs, encoder ordering and sequence metadata provide
the attribution; tensor totals are never treated as per-video evidence.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import contextmanager, ExitStack
from copy import deepcopy
import fcntl
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
import types
import uuid

from . import gemma_speed as g
from . import gemma_cpu_copy as c

SCHEMA = "gemma-tp2-pair-v1"
CPU_SHA = "e3b93f048645d6145f362430513f99b60531c8e585e598027d2abb11ac2900e3"
REFERENCE_SHA = "738d823872b884aeefef1f514b6bec09535fa74ebb1115e60333e3bc1ca967f9"
# Exact installed Fourier sources; V2 hooks are also checked by AST preflight.
EXTRA_SOURCES = {
    "vllm/v1/worker/gpu_worker.py": "43bd01925cd84f7d8a1d4a2ec30e289014a485fe4492a5343cbc893e8ed2f518",
    "vllm/v1/worker/gpu/model_runner.py": "725c47114e148ade3a26b4cc9c5f7295588fdd387afc3cb5fdb2d16513a1bb5c",
    "vllm/v1/worker/gpu/mm/encoder_runner.py": "0981e352d70301b7154a217a48665ffe35f215d053ec5740c2515a3e1e11b4e6",
    "vllm/v1/worker/gpu/mm/encoder_cache.py": "a27a52c2c52686a574b24910d38f7bd81508fa75d9d757f1b8d6e169575aa385",
    "vllm/v1/worker/gpu/model_states/__init__.py": "07815d0b788fc88185772f581bbf892a675259f9288375571f29887622f8d93c",
    "vllm/v1/worker/gpu/model_states/default.py": "f1d34d5c8c03be6e5afec8c2a24480ec259ca7c87392462f4e52d6c4eb775733",
    "vllm/v1/worker/gpu/model_states/interface.py": "4f4fdd3e0a2611836a79ed0392b843b627fd65382138530be1dd5ae33c2ab4b3",
    "vllm/v1/worker/gpu/input_batch.py": "96163c486368f2cf0e539fe68eaa06f4df3ac814da8f7a92af2b04242e01bc6d",
    "vllm/v1/worker/gpu/states.py": "8c899b8ef63d95f450876d3f01e56772f9534938a95383a46659fddf85573fa3",
    "vllm/v1/worker/gpu/buffer_utils.py": "6597f701a30261bf250e9ad5a29737264624a1caa20527aeea0dad630325b093",
    "vllm/v1/worker/gpu/attn_utils.py": "0f9df8b0153bd0a2720ee957f6cd1600b9d865eb12b7bf432501ffaa12fb6a9c",
    "vllm/entrypoints/llm.py": "daa6aeb5e971a66ccf4cba24390a3ba6c450e7bd9048942ac1c39f0dd54f237f",
    "vllm/entrypoints/offline_utils.py": "582149546697619665c5cf93a0e54ab367cac07830c7efb1f4412cb91f6bd65e",
    "vllm/v1/engine/llm_engine.py": "fd838313336aff1734a435f606bbdc705ee7efac94c479e26c2cc87eee0f82a8",
    "vllm/v1/engine/input_processor.py": "d79601e4a053a7bacb0d6a23e273348435e4ab70c6a315dcccb2d428ed57d96b",
    "vllm/multimodal/utils.py": "24a7f1f52cd609ac55da83490dd90192b6ef0b4d58a0c51101d2c10b79eb882e",
    "vllm/v1/engine/core_client.py": "24eecc78e8fc4849d4eac948c7586ee349ebfd503d07e80f05ec6c859da06fdd",
    "vllm/v1/engine/core.py": "f116962c35d9d312b20b0b76c34a8cdaa6d3f1d31f2036db4e123957d2d1555b",
    "vllm/v1/structured_output/backend_xgrammar.py": "7fab895e89f98f2a38e1007ec2c48557d77603a6de9c2c33df1b3432cbbe1876",
    "vllm/v1/structured_output/utils.py": "4c16c4f5faa22b2d53f0712a2f2de647d36de9e908fcba4114397eaeebdba4fc",
}
OUTPUT_KEYS = ("output_token_ids", "raw_answer", "finish_reason", "stop_reason")
RESOURCE_BUDGET = {"min_free_gpu_gib_per_device": 90, "min_available_ram_gib": 64,
    "kind": "admission budget, not measured peak", "KV_capacity_tokens_per_rank_minimum": 131072}
ACTUAL_ENGINE = {"tensor_parallel_size": 2, "max_num_seqs": 2, "max_num_batched_tokens": 68000,
    "max_model_len": 65536, "dtype": "torch.bfloat16", "kv_cache_dtype": "bfloat16"}
COMMUNICATION_ENV = {"NCCL_P2P_DISABLE": "1", "NCCL_DEBUG": "INFO"}
SAMPLING_SCALARS = {"n": 1, "temperature": 0., "top_p": 1., "seed": 0, "max_tokens": 16384, "repetition_penalty": 1.}
CHOICES = ["LEFT", "RIGHT"]
CHOICE_GRAMMAR = 'root ::= "LEFT" | "RIGHT"'
STRUCTURED_DEFAULTS = {"json": None, "regex": None, "json_object": None, "structural_tag": None,
    "disable_any_whitespace": False, "disable_additional_properties": False, "whitespace_pattern": None}
V2_TYPES = {"runner": "vllm.v1.worker.gpu.model_runner.GPUModelRunner",
    "model_state": "vllm.v1.worker.gpu.model_states.default.DefaultModelState",
    "encoder": "vllm.v1.worker.gpu.mm.encoder_runner.EncoderRunner",
    "request_state": "vllm.v1.worker.gpu.states.RequestState"}
V2_HOOKS = {
    "vllm/v1/worker/gpu/model_runner.py": {
        "GPUModelRunner.add_requests": ("self", "scheduler_output"),
        "GPUModelRunner.prepare_inputs": ("self", "scheduler_output", "batch_req_state", "batch_desc", "num_active_loras"),
        "GPUModelRunner.execute_model": ("self", "scheduler_output", "intermediate_tensors", "dummy_run", "skip_attn_for_dummy_run", "is_profile", "context_len", "valid_dummy_state_slots", "randomize_inputs")},
    "vllm/v1/worker/gpu/mm/encoder_runner.py": {
        "EncoderRunner.prepare_mm_inputs": ("self", "scheduled_encoder_inputs"),
        "EncoderRunner.execute_mm_encoder": ("self", "mm_kwargs")},
    "vllm/v1/worker/gpu/model_states/default.py": {
        "DefaultModelState.prepare_inputs_embeds": ("self", "scheduled_encoder_inputs", "input_batch", "req_states"),
        "DefaultModelState.prepare_attn": ("self", "input_batch", "cudagraph_mode", "block_tables", "slot_mappings", "attn_groups", "kv_cache_config", "for_capture", "ubatch_idx", "model_specific_attn_metadata")},
    "vllm/v1/worker/gpu/model_states/interface.py": {
        "ModelState.execute_mm_encoder": ("self", "scheduled_encoder_inputs")}}
V2_HOOK_AST_SHA = {
    "GPUModelRunner.add_requests": "623557aa6e25c9f565cf20e2fa4970b1fc30db2fa3f0403a18e99f6986dbe4a0",
    "GPUModelRunner.prepare_inputs": "225b64b94c0d7a31e62521f2882d11609f64203803de2abbc0c0d2d5d96158a9",
    "GPUModelRunner.execute_model": "3b8a3314d1d171d46621d430698ecdbfeede8d29f6b71c81c9bf67641fc33f20",
    "EncoderRunner.prepare_mm_inputs": "ed6fb053efe4131d1da84a190c6f87b6da0bf841d46f6813d20d5d7e06c9a3dd",
    "EncoderRunner.execute_mm_encoder": "17a9045ae5e3bf1d71af72870e642682d85960f9f69cc443450e44892dc94f40",
    "DefaultModelState.prepare_inputs_embeds": "9037a43a96cb6fcfae870b008e714e3d19e97982cae4c3d8393510ff9aadc2f6",
    "DefaultModelState.prepare_attn": "e124600adcaa59004ca8ae987043f64c7ed39c644b6acbd0f1eb5c8c1a1efcb6",
    "ModelState.execute_mm_encoder": "f3b486b9d6ee6e76895cf06fa365f72353049c97dbf255ae01694696c570384c"}


class OutputDifference(ValueError):
    pass


def extra_sources():
    distribution = importlib.metadata.distribution("vllm")
    return {name: g.file_sha256(distribution.locate_file(name)) for name in EXTRA_SOURCES}


def canonical_ast(value):
    """Version-portable AST fields; only Python 3.12's empty type_params is omitted."""
    if isinstance(value, ast.AST):
        return {"node": type(value).__name__, "fields": {name: canonical_ast(field)
            for name, field in ast.iter_fields(value)
            if not (name == "type_params" and isinstance(value, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and type(field) is list and not field)}}
    if isinstance(value, list): return [canonical_ast(item) for item in value]
    if isinstance(value, tuple): return {"tuple": [canonical_ast(item) for item in value]}
    if isinstance(value, bytes): return {"bytes": value.hex()}
    if value is Ellipsis: return {"constant": "Ellipsis"}
    return value


def portable_ast_sha256(node):
    return hashlib.sha256(json.dumps(canonical_ast(node), sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def observer_interface():
    return {"status": "PASS", "types": V2_TYPES, "hooks": {name: list(methods) for name, methods in V2_HOOKS.items()},
        "hook_AST_sha256": V2_HOOK_AST_SHA,
        "hook_AST_format": "recursive sorted JSON; omit only empty FunctionDef/AsyncFunctionDef/ClassDef.type_params",
        "sources": {name: sha for name, sha in EXTRA_SOURCES.items() if "/worker/gpu" in name},
        "request_tokens": "req_states.all_token_ids.gpu[req_id_to_index, :prompt_len]",
        "encoder_owner": "scheduled request ID + cached MM feature identity + returned native kwargs object",
        "sequence_owner": "InputBatch passed to DefaultModelState.prepare_attn"}


def observer_compatibility(sources=None):
    """Read/parse installed V2 source only; no torch/vLLM import or model load."""
    names = observer_interface()["sources"]
    if sources is None:
        distribution = importlib.metadata.distribution("vllm")
        sources = {name: Path(distribution.locate_file(name)).read_text() for name in names}
    g.require(all(name in sources and hashlib.sha256(sources[name].encode()).hexdigest() == sha
        for name, sha in names.items()), "Installed V2 observer source differs")
    methods = {}
    for name, hooks in V2_HOOKS.items():
        tree = ast.parse(sources[name])
        found = {cls.name + "." + fn.name: fn for cls in tree.body if isinstance(cls, ast.ClassDef)
            for fn in cls.body if isinstance(fn, ast.FunctionDef)}
        for key, args in hooks.items():
            fn = found.get(key)
            g.require(fn is not None and tuple(a.arg for a in fn.args.args) == args, "V2 observer hook signature differs: " + key)
            g.require(portable_ast_sha256(fn) == V2_HOOK_AST_SHA[key],
                "V2 observer hook AST differs: " + key)
            methods[key] = fn
    required_calls = {
        "GPUModelRunner.add_requests": ("self.req_states.add_request", "self.encoder_cache.add_request", "self.req_states.apply_staged_writes"),
        "GPUModelRunner.execute_model": ("self.add_requests", "self.prepare_inputs", "self.model_state.prepare_attn", "self.model_state.prepare_inputs_embeds"),
        "DefaultModelState.prepare_inputs_embeds": ("self.execute_mm_encoder",),
        "DefaultModelState.prepare_attn": ("compute_mm_prefix_ranges", "build_attn_metadata"),
        "ModelState.execute_mm_encoder": ("self.encoder_runner.prepare_mm_inputs", "self.encoder_runner.execute_mm_encoder"),
        "EncoderRunner.execute_mm_encoder": ("group_and_batch_mm_kwargs", "self.model.embed_multimodal")}
    for key, required in required_calls.items():
        calls = {ast.unparse(node.func) for node in ast.walk(methods[key]) if isinstance(node, ast.Call)}
        g.require(set(required) <= calls, "V2 native hook call chain differs: " + key)
    returned = [ast.unparse(node.value) for node in ast.walk(methods["EncoderRunner.prepare_mm_inputs"]) if isinstance(node, ast.Return)]
    g.require(returned == ["(mm_hashes, mm_kwargs)"], "V2 encoder return structure differs")
    worker = ast.parse(sources["vllm/v1/worker/gpu_worker.py"])
    g.require(any(isinstance(node, ast.ImportFrom) and node.module == "vllm.v1.worker.gpu.model_runner"
        and any(alias.name == "GPUModelRunner" and alias.asname == "GPUModelRunnerV2" for alias in node.names)
        for node in ast.walk(worker)), "Installed worker does not select anchored V2 runner")
    return observer_interface()


def engine_settings(checkpoint, kv_cache_gib=32):
    g.require(type(kv_cache_gib) is int and kv_cache_gib in (32, 56), "KV cache must be explicitly 32 or 56 GiB per rank")
    result = g.engine_settings(checkpoint)
    result.update(tensor_parallel_size=2, distributed_executor_backend="mp",
        max_num_seqs=2, max_num_batched_tokens=68000, cpu_offload_gb=0,
        kv_cache_memory_bytes=kv_cache_gib * 2**30)
    return result


def communication_settings(disable_p2p=False):
    g.require(type(disable_p2p) is bool, "Communication flag must be explicit boolean")
    return {"disable_p2p": disable_p2p, "environment": dict(COMMUNICATION_ENV) if disable_p2p else {},
        "scope": "current process and its spawned ranks only", "automatic_fallback": False}


def parallel_runtime(qualified_runtime, devices, communication):
    g.require(communication == communication_settings(communication["disable_p2p"]), "Communication contract differs")
    runtime = deepcopy(qualified_runtime)
    g.require(not set(COMMUNICATION_ENV) & set(runtime["environment"]), "Historical C3 declared unexpected communication overrides")
    runtime["environment"].update(CUDA_VISIBLE_DEVICES=devices, **communication["environment"])
    return runtime


def runtime_check(profile):
    """Validate the two opt-in variables, then delegate every original check."""
    communication = profile["communication"]
    g.require(communication == communication_settings(communication["disable_p2p"]), "Communication contract differs")
    runtime = deepcopy(profile["runtime"])
    for key in COMMUNICATION_ENV:
        expected = communication["environment"].get(key)
        g.require(runtime["environment"].get(key) == expected and os.environ.get(key) == expected,
            "Actual communication environment differs: " + key)
        runtime["environment"].pop(key, None)
    # gemma_speed's original whitelist remains unchanged. Its package,
    # checkpoint, declared environment, spawn and UVA checks all still run.
    return g.runtime_check(runtime)


def selected_items(panel):
    remaining = {"T1": 2, "T2": 4, "T3": 2}
    result = []
    for item in g.validate_items(panel["items"]):
        if remaining[item["task"]]:
            result.append(item); remaining[item["task"]] -= 1
    g.require(not any(remaining.values()) and len(result) == 8, "Missing fixed eight-video scope")
    return result


def schedule(items):
    ids = [item["sample_id"] for item in items]
    g.require(len(ids) == len(set(ids)) == 8, "Eight unique IDs required")
    return [ids, ids[1:] + ids[:1]]


def canonical_ranges(masks):
    values = []
    for record in masks:
        for ranges in record["ranges"].values():
            g.validate_ranges(ranges)
            if ranges not in values: values.append(ranges)
    g.require(len(values) == 1, "One native video interval template required per ID")
    return values[0]


def placement_source():
    c.validate_sources()
    source = inspect.getsource(g._placement)
    old = 'result["uva"] is False and result["offload_bytes"] >= 8 * 2**30\n        and all(n.startswith("language_model.") and ".gate_up_proj." in n for n in names)'
    new = 'result["uva"] in (False, None) and result["offload_bytes"] == 0 and not names'
    g.require(source.count(old) == 1 and source.count('result["kv"]["capacity_tokens"] >= 65536') == 1,
        "Frozen placement anchors changed")
    source = source.replace(old, new).replace('result["kv"]["capacity_tokens"] >= 65536',
        'result["kv"]["capacity_tokens"] >= 2 * 65536')
    return source


def _gc_line():
    lines, start = inspect.getsourcelines(run_batches)
    calls = [node for node in ast.walk(ast.parse("".join(lines)))
             if isinstance(node, ast.Call) and ast.unparse(node.func) == "g.gc.collect"]
    g.require(len(calls) == 1, "Unique pair-loop GC call changed")
    return start + calls[0].lineno - 1


def contract():
    return {"source_sha256": g.file_sha256(__file__), "CPU_source_sha256": CPU_SHA,
        "placement_source_sha256": hashlib.sha256(placement_source().encode()).hexdigest(),
        "gc_target": {"function": "gemma_parallel.run_batches", "line": _gc_line(), "cadence": 8},
        "run_bytecode_sha256": hashlib.sha256(run_batches.__code__.co_code).hexdigest(),
        "schedule_barrier": "native sleep(level=0,mode=keep); enqueue; arm; durable intents; wake_up(scheduling)",
        "scheduling": {"sleep_level": 0, "sleep_mode": "keep", "wake_tags": ["scheduling"],
            "weights_and_KV_remain_resident": True, "unsupported_installed_source": "reject before LLM construction"},
        "observer": "native internal/external request IDs; scheduler video order; per180frame slices; chunk intersections; actual sequence boundaries",
        "observer_interface": observer_interface(),
        "sampling_guard": {"scalars": SAMPLING_SCALARS, "canonical_choices": CHOICES, "native_normalized_grammar": CHOICE_GRAMMAR,
            "other_constraints": STRUCTURED_DEFAULTS, "normalizer_sources": {name: sha for name, sha in EXTRA_SOURCES.items()
                if "/structured_output/" in name}},
        "maximum_requests": 16, "maximum_concurrency": 2, "automatic_retry": False,
        "production_qualification": False}


def validate_profile(profile, installed=False):
    c.validate_sources()
    g.require(g.file_sha256(c.__file__) == CPU_SHA and profile["schema"] == SCHEMA
        and profile["execution"] == contract(), "Pair entry/CPU/source contract changed")
    q = c.validate_profile(profile["qualified_C3"], installed_sdk=installed)
    g.require(q["kernel"] == {"range_lookup": "binary", "prefill_block_m": 32, "expected_native_sha256": g.NATIVE_SHA}
        and q["cpu_execution_overlay"]["placement"] == "baseline" and q["cpu_execution_overlay"]["gc_cadence"] == 8,
        "Only unchanged C3 CPU/kernel algorithm is supported")
    g.require(q["maximum_model_calls"] == len(q["panel"]["items"]) == 20
        and q["cpu_execution_overlay"]["reference_file_sha256"] == REFERENCE_SHA, "Existing common20 binding required")
    g.require(profile["items"] == selected_items(q["panel"]) and profile["orders"] == schedule(profile["items"])
        and profile["engine_settings"] == engine_settings(q["runtime"]["checkpoint"], profile["kv_cache_gib"])
        and profile["extra_sources"] == EXTRA_SOURCES and profile["reference_sha256"] == REFERENCE_SHA
        and profile["resource_budget"] == RESOURCE_BUDGET,
        "Finite scope, full-context settings or source binding differs")
    runtime = parallel_runtime(q["runtime"], profile["devices"], profile["communication"])
    devices = profile["devices"].split(",")
    g.require(len(devices) == len(set(devices)) == 2 and all(d.strip() == d and d for d in devices)
        and profile["runtime"] == runtime, "Two-device runtime differs")
    if installed:
        g.require(extra_sources() == profile["extra_sources"] and g.runtime_sources() == q["runtime_sources"], "Installed SDK bytes differ")
        observer_compatibility()
    return profile


def seal_profile(args):
    root = Path(args.baseline_run)
    identity, result, exited = (g.read(root / name) for name in ("identity.json", "result.json", "exit.json"))
    q = c.validate_profile(identity["profile"], installed_sdk=True)
    from .gemma_prefill import _verify_cpu_receipt
    _verify_cpu_receipt(root, q)
    g.require(result["status"] == exited["status"] == "COMPLETE" and result["calls"] == 20
        and exited["owned_engine_workers_exited"] and exited["lease_released_after_engine_exit"]
        and identity["profile_sha256"] == result["profile_sha256"] == g.digest(q)
        and not (root / "failure.json").exists(), "C3 baseline is not cleanly complete")
    reference = g.read(args.reference)
    g.require(g.file_sha256(args.reference) == REFERENCE_SHA and reference["panel_sha256"] == g.digest(q["panel"]), "Use existing common20 CPU reference")
    items = selected_items(q["panel"])
    if args.panel: g.require(g.read(args.panel)["items"] == items, "Explicit eight-ID panel differs")
    raw = {row["sample_id"]: row for row in read_lines(root / "raw.jsonl")}
    technical = {row["sample_id"]: row for row in result["rows"]}
    g.require(len(raw) == len(technical) == 20, "Incomplete baseline outputs")
    baseline = {}
    for item in items:
        sid = item["sample_id"]; row = raw[sid]
        g.verify_input(row["input_receipt"], reference["rows"][sid], row["deep"])
        g.require(row["actual_input_token_ids"] == reference["rows"][sid]["input_token_ids"]
            and row["finish_reason"] == "stop" and row["raw_answer"] in ("LEFT", "RIGHT"), "Baseline controls differ")
        baseline[sid] = {**{key: row[key] for key in OUTPUT_KEYS}, "native_ranges": canonical_ranges(technical[sid]["native_masks"])}
    communication = communication_settings(getattr(args, "disable_p2p", False))
    runtime = parallel_runtime(q["runtime"], args.devices, communication)
    kv_cache_gib = getattr(args, "kv_cache_gib", 32)
    profile = {"schema": SCHEMA, "name": args.name, "qualified_C3": q, "items": items, "orders": schedule(items),
        "devices": args.devices, "runtime": runtime, "communication": communication,
        "kv_cache_gib": kv_cache_gib, "engine_settings": engine_settings(runtime["checkpoint"], kv_cache_gib),
        "reference_sha256": REFERENCE_SHA, "extra_sources": extra_sources(), "execution": contract(), "baseline": baseline,
        "baseline_files": {name: g.file_sha256(root / name) for name in ("identity.json", "load.json", "result.json", "raw.jsonl", "exit.json", "cpu-copy.json")},
        "baseline_torch_policy": g.read(root / "load.json")["placement"][0]["torch_policy"],
        "resource_budget": deepcopy(RESOURCE_BUDGET)}
    validate_profile(profile, True)
    g.seal(args.output, profile)
    return {"status": "SEALED", "profile_sha256": g.digest(profile), "maximum_requests": 16, "GPU_calls": 0}


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def frame_intersections(counts, start, length):
    g.require(counts and all(type(n) is int and n == 180 for n in counts)
        and type(start) is int and type(length) is int and length > 0
        and 0 <= start < start + length <= sum(counts), "Invalid frame/chunk boundaries")
    result, offset = [], 0
    for index, count in enumerate(counts):
        lo, hi = max(start, offset), min(start + length, offset + count)
        if lo < hi: result.append((index, lo - start, hi - start))
        offset += count
    return result


def sequence_records(req_ids, starts, lengths, ranges, expected):
    g.require(len(req_ids) == len(set(req_ids)) == len(lengths) and len(starts) == len(req_ids) + 1
        and starts[0] == 0 and set(req_ids) <= set(expected), "Invalid/foreign sequence boundaries")
    result = {}
    for i, rid in enumerate(req_ids):
        count = starts[i + 1] - starts[i]
        g.require(0 < count <= lengths[i] <= 65536, "Overlapping/empty/out-of-context query segment")
        value = ranges.get(i)
        g.require(value is not None, "Missing per-sequence native video intervals")
        value = [list(pair) for pair in value]; g.validate_ranges(value)
        g.require(value == expected[rid]["native_ranges"], "Native mask belongs to another context")
        result[rid] = {"query_start": starts[i], "query_end": starts[i + 1], "sequence_length": lengths[i],
            "absolute_position_start": lengths[i] - count, "native_ranges": value}
    return result


def validate_text_positions(positions, records):
    receipts = []
    for rid, rec in records.items():
        actual = positions[rec["query_start"]:rec["query_end"]]
        g.require(actual == list(range(rec["absolute_position_start"], rec["sequence_length"])), "Actual text positions cross sequence boundary")
        receipts.append({"request_id": rid, "sha256": g.digest(actual), "count": len(actual),
            "start": rec["absolute_position_start"], "end": rec["sequence_length"]})
    return receipts


class EncoderLedger:
    def __init__(self):
        self.expected, self.pending, self.seen = {}, [], set()

    def arm(self, expected):
        g.require(not self.pending and len(expected) == 2
            and len({v["sample_id"] for v in expected.values()}) == 2, "Pair ownership changed/duplicated")
        self.expected, self.seen = expected, set()

    def scheduled(self, hashes, modalities, refs, actual_requests):
        g.require(not self.pending and len(hashes) == len(modalities) == len(refs), "Encoder grouping/order changed")
        ids = [rid for rid, _ in refs]
        g.require(len(ids) == len(set(ids)) and set(ids) <= set(self.expected) and not set(ids) & self.seen
            and all(mode == "video" for mode in modalities), "Mixed, duplicate or unknown encoder context")
        for rid, mm_hash in zip(ids, hashes, strict=True):
            g.require(mm_hash == self.expected[rid]["mm_identifier"]
                and list(actual_requests[rid].prompt_token_ids) == self.expected[rid]["input_token_ids"], "Actual GPU request identity/prompt differs")
        self.pending = ids

    def consume(self, counts, total_frames):
        g.require(counts and all(n == 180 for n in counts) and sum(counts) == total_frames
            and len(counts) <= len(self.pending), "360 total frames are not one 180-frame video")
        ids, self.pending = self.pending[:len(counts)], self.pending[len(counts):]
        g.require(not set(ids) & self.seen, "Video was encoded twice")
        self.seen.update(ids)
        return ids


def v2_prompt_tokens(runner, request_id):
    requests = runner.req_states
    g.require(request_id in requests.req_id_to_index, "V2 request is absent from native slot map")
    index = requests.req_id_to_index[request_id]
    g.require(requests.index_to_req_id.get(index) == request_id, "V2 request slot ownership differs")
    length = int(requests.prompt_len.np[index])
    g.require(0 < length <= 65536, "V2 native prompt length differs")
    return requests.all_token_ids.gpu[index, :length].detach().cpu().tolist()


def v2_encoder_owners(runner, scheduled, result):
    hashes, kwargs = result
    g.require(len(hashes) == len(kwargs), "V2 encoder output association differs")
    candidates = []
    for rid, indices in scheduled.items():
        features = runner.encoder_cache.mm_features[rid]
        for index in indices:
            feature = features[index]
            if feature.data is not None and feature.identifier not in runner.encoder_cache.encoder_outputs:
                candidates.append((rid, feature))
    refs, actual = [], {}
    for mm_hash, (modality, data) in zip(hashes, kwargs, strict=True):
        matches = [(rid, feature) for rid, feature in candidates if feature.identifier == mm_hash
            and feature.modality == modality and feature.data is data]
        g.require(len(matches) == 1 and modality == "video", "V2 encoder kwargs have ambiguous/foreign native ownership")
        rid, feature = matches[0]
        g.require(rid not in actual, "V2 request contains multiple encoder objects")
        refs.append((rid, feature.mm_position))
        actual[rid] = types.SimpleNamespace(prompt_token_ids=v2_prompt_tokens(runner, rid))
    return hashes, [mode for mode, _ in kwargs], refs, actual


def install_v2_context(runner, state):
    """Observe native V2 objects; return/arguments and model operations are untouched."""
    model_state, encoder = runner.model_state, runner.model_state.encoder_runner
    objects = {"runner": runner, "model_state": model_state, "encoder": encoder, "request_state": runner.req_states}
    actual = {name: type(value).__module__ + "." + type(value).__qualname__ for name, value in objects.items()}
    g.require(actual == V2_TYPES and encoder.encoder_cache is runner.encoder_cache is model_state.encoder_cache,
        "Actual runner/encoder/cache is not the anchored V2 interface")
    native_batch, native_attn = encoder.prepare_mm_inputs, model_state.prepare_attn
    g.require(tuple(inspect.signature(native_batch).parameters) == ("scheduled_encoder_inputs",)
        and tuple(inspect.signature(native_attn).parameters) == V2_HOOKS[
            "vllm/v1/worker/gpu/model_states/default.py"]["DefaultModelState.prepare_attn"][1:], "Actual V2 bound hook signature differs")
    attn_signature = inspect.signature(native_attn)
    def batch(this, scheduled_encoder_inputs):
        result = native_batch(scheduled_encoder_inputs)
        if state["active"]:
            state["ledger"].scheduled(*v2_encoder_owners(runner, scheduled_encoder_inputs, result))
        return result
    def attn(this, *args, **kwargs):
        if not state["active"]: return native_attn(*args, **kwargs)
        bound = attn_signature.bind(*args, **kwargs)
        g.require(not bound.arguments.get("for_capture", False) and "input_batch" not in state,
            "V2 observer received capture/nested attention preparation")
        state["input_batch"] = bound.arguments["input_batch"]
        try: return native_attn(*args, **kwargs)
        finally: del state["input_batch"]
    encoder.prepare_mm_inputs = types.MethodType(batch, encoder)
    model_state.prepare_attn = types.MethodType(attn, model_state)
    return actual


def _rank_identity(worker):
    import torch
    return {"worker": g.process_identity(), "rank": torch.distributed.get_rank(),
        "world_size": torch.distributed.get_world_size(), "device": str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)}


def _install_rank(worker, kernel, output):
    import torch
    identity = _rank_identity(worker)
    g.require(identity["world_size"] == 2 and identity["rank"] in (0, 1), "Actual TP world differs")
    folder = Path(output) / f"rank-{identity['rank']}"; folder.mkdir()
    receipt = g._install_kernel(worker, kernel, str(folder))
    def placement_require(ok, message):
        if not ok:
            actual = sys._getframe(1).f_locals.get("result")
            g.atomic_json(folder / "placement-rejected.json", {"identity": identity, "error": message, "actual": actual})
        g.require(ok, message)
    namespace = {**vars(g), "require": placement_require}
    exec(compile(placement_source(), "<gemma-tp2-placement>", "exec"), namespace)
    placement = namespace["_placement"](worker)
    placement.update(identity, peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    config = worker.vllm_config
    actual = {"tensor_parallel_size": config.parallel_config.tensor_parallel_size,
        "max_num_seqs": config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": config.scheduler_config.max_num_batched_tokens,
        "max_model_len": config.model_config.max_model_len, "dtype": str(config.model_config.dtype),
        "kv_cache_dtype": config.cache_config.cache_dtype}
    g.require(actual == ACTUAL_ENGINE, "Actual native engine configuration differs")
    return {"identity": identity, "kernel": receipt, "placement": placement, "actual_engine": actual, "observer": _install_pair_observer(worker)}


def _install_pair_observer(worker):
    import torch
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadataBuilder
    g.require(extra_sources() == EXTRA_SOURCES, "Worker installed request/encoder SDK source differs")
    integration = observer_compatibility()
    runner, model = worker.model_runner, worker.get_model()
    state = {"ledger": EncoderLedger(), "video": {}, "sequences": [], "positions": [], "events": [], "active": False}
    worker._gemma_pair = state
    actual_types = install_v2_context(runner, state)
    video_native = model._process_video_input
    def projection(module, args):
        if not state["active"]: return
        values = args[0]; current = state["projection"]
        g.require(list(values.shape[1:]) == [2520, 768] and values.dtype == torch.bfloat16, "Projection chunk geometry/precision differs")
        for i, lo, hi in frame_intersections(current["counts"], current["cursor"], values.shape[0]):
            rid = current["ids"][i]
            if state["ledger"].expected[rid]["deep"]:
                current["hashes"][rid].update(memoryview(values[lo:hi].detach().contiguous().cpu().view(torch.uint8).numpy()).cast("B"))
        current["cursor"] += values.shape[0]
    model.vision_tower.patch_embedder.input_proj.register_forward_pre_hook(projection)
    def video(this, inputs):
        if not state["active"]: return video_native(inputs)
        x, positions = inputs["pixel_values_videos"], inputs["pixel_position_ids_videos"]
        counts = inputs["video_frame_counts"].tolist()
        ids = state["ledger"].consume(counts, x.shape[0])
        g.require(list(x.shape[1:]) == [2520, 768] and list(positions.shape) == [sum(counts), 2520, 2]
            and x.dtype == torch.bfloat16 and x.device.type == positions.device.type == "cuda", "Actual encoder tensor geometry differs")
        state["projection"] = {"counts": counts, "ids": ids, "cursor": 0, "hashes": {rid: hashlib.sha256() for rid in ids}}
        for i, rid in enumerate(ids):
            expected = state["ledger"].expected[rid]
            receipt = {"request_id": rid, "sample_id": expected["sample_id"], "frames": 180, "frame_offset": i * 180,
                "encoding_calls_on_this_rank": 1, "encoder_group_total_frames": sum(counts),
                "rank_frame_semantics": "one full 180-frame input per TP rank; rank copies are not additional logical frames",
                "shape": [180, 2520, 768], "dtype": str(x.dtype), "device": str(x.device), "deep": expected["deep"],
                "position_sha256": g.sha_tensor(positions[i * 180:(i + 1) * 180]),
                "GPU_prompt_token_ids_sha256": g.digest(v2_prompt_tokens(runner, rid))}
            if expected["deep"]: receipt["input_sha256"] = g.sha_tensor(x[i * 180:(i + 1) * 180])
            state["video"][rid] = receipt
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); result = video_native(inputs); b.record()
        state["events"].append(("vision_group", ids, a, b))
        g.require(len(result) == len(ids) and state["projection"]["cursor"] == sum(counts), "Encoder/projection boundary coverage differs")
        for rid, embeddings in zip(ids, result, strict=True):
            state["video"][rid]["embedding_shape"] = list(embeddings.shape)
            if state["ledger"].expected[rid]["deep"]:
                state["video"][rid]["projection_sha256"] = state["projection"]["hashes"][rid].hexdigest()
        del state["projection"]
        return result
    model._process_video_input = types.MethodType(video, model)
    mask_native = TritonAttentionMetadataBuilder.build
    def mask(this, *args, **kwargs):
        result = mask_native(this, *args, **kwargs)
        if state["active"]:
            starts, lengths = result.query_start_loc.cpu().tolist(), result.seq_lens.cpu().tolist()
            input_batch = state["input_batch"]
            ids = list(input_batch.req_ids)
            g.require(len(ids) == input_batch.num_reqs and list(input_batch.idx_mapping_np) == [
                runner.req_states.req_id_to_index[rid] for rid in ids], "V2 InputBatch request slots differ")
            g.require(len(ids) == len(lengths) and starts[-1] == result.num_actual_tokens, "Scheduler sequence padding/order differs")
            records = sequence_records(ids, starts, lengths, result.mm_prefix_range or {}, state["ledger"].expected)
            causal = result.causal.tolist() if hasattr(result.causal, "tolist") else result.causal
            g.require(causal is True or causal == [True] * len(ids), "Causal policy differs")
            entry = {"request_ids": ids, "sequences": records, "causal": causal}
            state["sequences"].append(entry); state["last_sequences"] = entry
        return result
    TritonAttentionMetadataBuilder.build = mask
    forward_native, signature = model.forward, inspect.signature(model.forward)
    def forward(*args, **kwargs):
        if not state["active"]: return forward_native(*args, **kwargs)
        positions = signature.bind(*args, **kwargs).arguments["positions"].detach().cpu().tolist()
        entry = state["last_sequences"]
        state["positions"].extend(validate_text_positions(positions, entry["sequences"]))
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); result = forward_native(*args, **kwargs); b.record()
        state["events"].append(("language_batch", entry["request_ids"], a, b))
        return result
    model.forward = forward
    return {"source_sha256": g.file_sha256(__file__), "extra_sources": extra_sources(), "per_request_encoder_boundaries": True,
        "integration": integration, "actual_types": actual_types}


def _arm(worker, expected):
    state = worker._gemma_pair
    g.require(not state["active"], "Previous pair observer has not closed")
    state["ledger"].arm(expected); state["active"] = True
    state.update(video={}, sequences=[], positions=[], events=[])
    return _rank_identity(worker)


def _observe(worker):
    import torch
    state = worker._gemma_pair; torch.cuda.synchronize()
    g.require(state["active"] and not state["ledger"].pending
        and state["ledger"].seen == set(state["ledger"].expected), "Incomplete per-video GPU coverage")
    state["active"] = False
    return {**_rank_identity(worker), "video": state["video"], "sequences": state["sequences"], "positions": state["positions"],
        "CUDA_groups": [{"stage": name, "request_ids": ids, "seconds": a.elapsed_time(b) / 1000} for name, ids, a, b in state["events"]],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "driver_free_total_bytes_at_observation": list(torch.cuda.mem_get_info())}


class PreparedRegistry:
    def __init__(self): self.entries = []

    def match(self, frames, metadata):
        candidates = [entry for entry in self.entries if frames is entry["frames"] or (
            frames.shape == entry["frames"].shape and frames.dtype == entry["frames"].dtype
            and frames.strides == entry["frames"].strides
            and frames.__array_interface__["data"][0] == entry["frames"].__array_interface__["data"][0])]
        g.require(len(candidates) == 1 and list(frames.shape) == [180, 528, 768, 3]
            and metadata["frames_indices"] == list(range(180)) and metadata["fps"] == 60.
            and metadata["total_num_frames"] == 180 and metadata["do_sample_frames"] is False,
            "Unknown/aliased video or changed frame metadata")
        candidates[0]["adapter_calls"] += 1
        return candidates[0]["payload"]


def install_adapter(session, registry):
    from PIL import Image
    from transformers.feature_extraction_utils import BatchFeature
    from vllm.model_executor.models.gemma4_mm import Gemma4MultiModalProcessor
    original, pil = Gemma4MultiModalProcessor._apply_hf_processor_main, Image.fromarray
    def no_pil(*args, **kwargs):
        g.require(not registry.entries, "Actual RGB16 request attempted PIL conversion")
        return pil(*args, **kwargs)
    def prepared(self, mm_items, hf_kwargs):
        if not registry.entries: return original(self, mm_items, hf_kwargs)
        g.require(mm_items.get_all_counts() == {"video": 1}, "Each request must contain exactly one video")
        videos = mm_items["video"].get_processor_data()["videos"]
        g.require(len(videos) == 1, "Multiple videos were merged into one request")
        return BatchFeature(registry.match(*videos[0]))
    session._implementation._patch(Image, "fromarray", no_pil)
    session._implementation._patch(Gemma4MultiModalProcessor, "_apply_hf_processor_main", prepared)


def same_tensor_view(actual, expected):
    return (actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device
        and actual.stride() == expected.stride() and actual.data_ptr() == expected.data_ptr())


def validate_sampling_receipt(value):
    g.require(set(value) == set(SAMPLING_SCALARS) | {"canonical_choices", "native_structured_outputs"}
        and all(type(value[key]) is type(expected) and value[key] == expected for key, expected in SAMPLING_SCALARS.items())
        and value["canonical_choices"] == CHOICES, "Actual per-request sampling scalars/canonical choices differ")
    native = value["native_structured_outputs"]
    g.require(set(native) == set(STRUCTURED_DEFAULTS) | {"choice", "grammar", "_backend", "_backend_was_auto"}
        and all(type(native[key]) is type(expected) and native[key] == expected for key, expected in STRUCTURED_DEFAULTS.items())
        and native["_backend"] in (None, "xgrammar") and type(native["_backend_was_auto"]) is bool,
        "Actual structured-output constraints/backend differ")
    g.require((native["choice"] == CHOICES and native["grammar"] is None)
        or (native["choice"] is None and native["grammar"] == CHOICE_GRAMMAR),
        "Actual native choice/grammar differs from the fixed two-choice language")
    return value


def actual_sampling(params):
    names = (*STRUCTURED_DEFAULTS, "choice", "grammar", "_backend", "_backend_was_auto")
    g.require(params.structured_outputs is not None and all(hasattr(params.structured_outputs, key) for key in names),
        "Actual structured-output API fields are missing")
    value = {key: getattr(params, key) for key in SAMPLING_SCALARS}
    value.update(canonical_choices=list(CHOICES), native_structured_outputs={key: deepcopy(getattr(params.structured_outputs, key)) for key in names})
    return validate_sampling_receipt(value)


@contextmanager
def capture_requests(engine, entries, expected, reference, profile):
    core = engine.llm_engine.engine_core
    native, captured = core.add_request, []
    def add(request):
        g.require(len(captured) < len(entries), "Unexpected extra native request")
        entry = entries[len(captured)]; sid = entry["item"]["sample_id"]
        rid, external = request.request_id, request.external_req_id
        g.require(isinstance(rid, str) and isinstance(external, str) and rid not in expected
            and list(request.prompt_token_ids) == reference["rows"][sid]["input_token_ids"], "Native request/prompt association differs")
        features = request.mm_features
        g.require(len(features) == 1 and features[0].modality == "video", "Request multimodal context is not independent")
        for key in ("pixel_values_videos", "pixel_position_ids_videos"):
            g.require(same_tensor_view(features[0].data[key].data, entry["payload"][key]), "Native request tensor no longer belongs to its video")
        expected[rid] = {"sample_id": sid, "external_id": external, "deep": entry["deep"],
            "mm_identifier": features[0].identifier, "input_token_ids": reference["rows"][sid]["input_token_ids"],
            "native_ranges": profile["baseline"][sid]["native_ranges"], "actual_sampling": actual_sampling(request.sampling_params)}
        captured.append(rid)
        return native(request)
    core.add_request = add
    try: yield captured
    finally: core.add_request = native


def map_outputs(outputs, expected):
    external = {value["external_id"]: rid for rid, value in expected.items()}
    g.require(len(external) == len(expected) == len(outputs), "Missing/duplicated request outputs")
    mapped = {}
    for output in outputs:
        rid = external.get(output.request_id)
        g.require(rid is not None and rid not in mapped and len(output.outputs) == 1, "Returned output belongs to another/duplicate context")
        g.require(list(output.prompt_token_ids) == expected[rid]["input_token_ids"], "Returned prompt differs")
        mapped[rid] = output
    return mapped


def validate_observations(observations, expected, reference):
    g.require(len(observations) == 2 and {x["rank"] for x in observations} == {0, 1}
        and len({x["device"] for x in observations}) == 2, "Two physical TP rank observations required")
    for observed in observations:
        g.require(set(observed["video"]) == set(expected), "Missing or merged GPU video receipt")
        for rid, binding in expected.items():
            value, ref = observed["video"][rid], reference["rows"][binding["sample_id"]]
            g.require(value["request_id"] == rid and value["sample_id"] == binding["sample_id"]
                and value["frames"] == 180 and value["shape"] == [180, 2520, 768]
                and value["dtype"] == "torch.bfloat16" and value["embedding_shape"][0] == 31680
                and value["position_sha256"] == ref["position_sha256"]
                and value["GPU_prompt_token_ids_sha256"] == ref["input_token_ids_sha256"], "Per-ID GPU controls differ")
            if binding["deep"]:
                g.require(all(value[key] == ref[key] for key in ("input_sha256", "projection_sha256")), "Per-ID GPU deep tensor differs")
            sequences = [entry["sequences"][rid] for entry in observed["sequences"] if rid in entry["sequences"]]
            g.require(sequences and all(x["native_ranges"] == binding["native_ranges"] for x in sequences)
                and any(x["request_id"] == rid for x in observed["positions"]), "Per-ID sequence/mask/position coverage missing")


@contextmanager
def physical_pair(runtime, torch, budget):
    from .inference import resource_snapshot
    g.require(torch.cuda.device_count() == 2, "Exactly two visible physical GPUs required")
    root = Path(runtime["lease_root"]).resolve()
    g.require(root == Path(runtime["asset_roots"]["T1"]).resolve() / "evaluation/outputs", "Native physical lock root changed")
    devices = [(str(torch.cuda.get_device_properties(i).uuid), i) for i in range(2)]
    g.require(len({device for device, _ in devices}) == 2, "Aliased physical GPUs")
    receipts = []
    with ExitStack() as stack:
        for device, index in sorted(devices):
            path = root / (".gpu-" + g.digest({"host": socket.gethostname(), "device": device}) + ".lock")
            g.require(path.is_file() and not path.is_symlink(), "Existing native GPU lock required")
            stream = stack.enter_context(path.open("r+b")); fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            st = os.fstat(stream.fileno())
            g.require((st.st_dev, st.st_ino) == (path.stat().st_dev, path.stat().st_ino), "Physical lock inode changed")
            with torch.cuda.device(index):
                resources = resource_snapshot(torch, {"min_free_gpu_gib": budget["min_free_gpu_gib_per_device"],
                    "min_available_ram_gib": budget["min_available_ram_gib"]})
            g.require(resources["admitted"], "Two-card resource budget not currently available")
            receipts.append({"device": device, "index": index, "path": str(path), "st_dev": st.st_dev, "st_ino": st.st_ino, "resources": resources})
        yield receipts


def range_proof(path, start):
    with Path(path).open("rb") as f: f.seek(start); data = f.read()
    g.require(data.endswith(b"\n"), "Partial call evidence")
    return {"start": start, "end": start + len(data), "sha256": hashlib.sha256(data).hexdigest()}


def close_failed_attempt(output, status):
    """Preserve a failed/uncertain tail, including zero outputs and partial bytes."""
    event_path = output / "events.jsonl"
    if not event_path.exists(): return
    previous = read_lines(output / "calls.jsonl") if (output / "calls.jsonl").exists() else []
    evidence = {}
    for key, filename in (("raw", "raw.jsonl"), ("events", "events.jsonl")):
        path = output / filename
        start = previous[-1][key]["end"] if previous else 0
        data = path.read_bytes()[start:] if path.exists() else b""
        evidence[key] = {"start": start, "end": start + len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "newline_complete": not data or data.endswith(b"\n")}
    if all(value["start"] == value["end"] for value in evidence.values()): return
    tail = event_path.read_bytes()[evidence["events"]["start"]:]
    rows = [json.loads(line) for line in tail.splitlines(keepends=True) if line.endswith(b"\n")]
    intents = [row for row in rows if row.get("event") == "GENERATE_INTENT"]
    g.append(output / "calls.jsonl", {"status": "DIFFERENT" if status == "DIFFERENT" else "UNCERTAIN",
        "generation_attempts": len(intents), "sample_ids": [x["sample_id"] for x in intents],
        "attempt_ids": [x["attempt_id"] for x in intents], "automatic_retry": False, **evidence})


class StartupChildren:
    """Track only our newly spawned descendants; retain leases through cleanup.

The native constructor can fail before collective_rpc exposes TP identities.
This closes that lifecycle gap without inspecting or signaling other jobs.
"""
    def __init__(self, output):
        self.output, self.owned, self.stop = Path(output), {}, threading.Event()
        self.baseline = self._descendants()

    def _descendants(self):
        todo, found = [os.getpid()], set()
        while todo:
            pid = todo.pop()
            try: children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            except FileNotFoundError: continue
            for child in map(int, children):
                if child not in found: found.add(child); todo.append(child)
        return found

    def scan(self):
        from multiprocessing import resource_tracker, forkserver
        infrastructure = {resource_tracker._resource_tracker._pid, forkserver._forkserver._forkserver_pid}
        for pid in self._descendants() - self.baseline - infrastructure:
            try:
                ticks = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
                boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
            except FileNotFoundError: continue
            self.owned[(pid, ticks)] = {"host": socket.gethostname(), "pid": pid, "startticks": ticks, "boot_id": boot}

    def __enter__(self):
        def watch():
            while not self.stop.wait(.02): self.scan()
        self.thread = threading.Thread(target=watch, daemon=True); self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set(); self.thread.join(); self.scan()
        g.atomic_json(self.output / "startup-children.json", {"status": "IDENTIFIED", "owned": list(self.owned.values())})

    def wait(self):
        identities = list(self.owned.values())
        g.atomic_json(self.output / "startup-children.json", {"status": "WAITING_OWNED_STARTUP_EXIT", "owned": identities,
            "physical_leases_held": True})
        while any(g.owned_worker_alive(identity) for identity in identities): time.sleep(1)
        g.atomic_json(self.output / "startup-children.json", {"status": "COMPLETE", "owned": identities, "all_observed_owned_descendants_exited": True})


def run_batches(engine, session, registry, profile, reference, output):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    sampling = SamplingParams(n=1, temperature=0., top_p=1., repetition_penalty=1., seed=0,
        max_tokens=16384, structured_outputs=StructuredOutputsParams(choice=["LEFT", "RIGHT"]))
    items = {item["sample_id"]: item for item in profile["items"]}
    summaries, results, calls = [], [], 0
    run_started, ordinary_started = time.monotonic(), None
    for pass_index, order in enumerate(profile["orders"]):
        for offset in range(0, 8, 2):
            began = time.monotonic(); entries, requests, expected = [], [], {}
            if calls == 2: ordinary_started = began
            batch_id = f"pass{pass_index}-pair{offset // 2}"
            starts = {name: (output / name).stat().st_size if (output / name).exists() else 0 for name in ("raw.jsonl", "events.jsonl")}
            for sid in order[offset:offset + 2]:
                item, deep = items[sid], calls < 2
                frames, payload, text, receipt = g.prepare(session, item, profile["runtime"]["asset_roots"], True, deep)
                g.verify_input(receipt, reference["rows"][sid], deep)
                entries.append({"item": item, "deep": deep, "frames": frames, "payload": payload, "receipt": receipt, "adapter_calls": 0})
                metadata = {"fps": 60., "duration": 3., "total_num_frames": 180, "frames_indices": list(range(180)),
                    "do_sample_frames": False, "video_backend": "qualified-rgb16"}
                requests.append({"prompt": text, "multi_modal_data": {"video": (frames, metadata)}})
                calls += 1
            del frames, payload, receipt
            registry.entries = entries
            # Level zero only pauses scheduling: it does not unload weights/KV.
            engine.sleep(level=0, mode="keep")
            g.append(output / "events.jsonl", {"event": "SUBMIT_INTENT", "batch_id": batch_id, "sample_ids": order[offset:offset + 2]})
            with capture_requests(engine, entries, expected, reference, profile) as captured:
                enqueued = engine.enqueue(requests, sampling, use_tqdm=False)
            g.require(enqueued == captured and len(enqueued) == 2 and all(entry["adapter_calls"] >= 1 for entry in entries), "Enqueue/adapter ownership differs")
            engine.collective_rpc(_arm, args=(expected,))
            attempt_ids = {rid: uuid.uuid4().hex for rid in expected}
            for rid, binding in expected.items():
                g.append(output / "events.jsonl", {"event": "GENERATE_INTENT", "batch_id": batch_id,
                    "pass": pass_index, "attempt_id": attempt_ids[rid], "internal_request_id": rid, **binding})
            prepared = time.monotonic()
            engine.wake_up(tags=["scheduling"])
            returned = engine.wait_for_completion(use_tqdm=False)
            finished = time.monotonic()
            try:
                mapped = map_outputs(returned, expected)
            except BaseException:
                # Invalid association must not erase the model's actual outputs
                # or invent a sample ID for an unassignable native response.
                g.append(output / "events.jsonl", {"event": "UNATTRIBUTED_NATIVE_OUTPUTS", "batch_id": batch_id,
                    "outputs": [{"request_id": value.request_id, "prompt_token_ids": value.prompt_token_ids,
                        "choices": [{"text": choice.text, "token_ids": list(choice.token_ids),
                            "finish_reason": choice.finish_reason, "stop_reason": choice.stop_reason} for choice in value.outputs]}
                        for value in returned]})
                raise
            raw_rows = []
            for rid, binding in expected.items():
                result, entry = mapped[rid], next(x for x in entries if x["item"]["sample_id"] == binding["sample_id"])
                choice = result.outputs[0]
                raw = {"batch_id": batch_id, "pass": pass_index, "attempt_id": attempt_ids[rid], "internal_request_id": rid,
                    "external_request_id": binding["external_id"], "sample_id": binding["sample_id"], "deep": binding["deep"],
                    "raw_answer": choice.text, "output_token_ids": list(choice.token_ids), "finish_reason": choice.finish_reason,
                    "stop_reason": choice.stop_reason, "actual_input_token_ids": list(result.prompt_token_ids), "input_receipt": entry["receipt"]}
                g.append(output / "raw.jsonl", raw); raw_rows.append(raw)
            observations = engine.collective_rpc(_observe)
            g.append(output / "events.jsonl", {"event": "ACTUAL_PAIR_GPU_INPUT", "batch_id": batch_id, "observations": observations})
            validate_observations(observations, expected, reference)
            for raw in raw_rows:
                g.require(raw["raw_answer"] in ("LEFT", "RIGHT") and raw["finish_reason"] == "stop", "Invalid completion")
                changed = [key for key in OUTPUT_KEYS if raw[key] != profile["baseline"][raw["sample_id"]][key]]
                if changed:
                    g.append(output / "events.jsonl", {"event": "OUTPUT_DIFFERENCE", "sample_id": raw["sample_id"], "fields": changed})
                    raise OutputDifference("Full output differs from C3: " + raw["sample_id"] + " " + ",".join(changed))
            registry.entries = []; requests.clear(); del returned, mapped, result, entry
            for entry in entries:
                sid, deep = entry["item"]["sample_id"], entry["deep"]
                del entry["frames"], entry["payload"]
                g.gc.collect()
            g.append(output / "events.jsonl", {"event": "PAIR_TECHNICAL_PASS", "batch_id": batch_id, "sample_ids": order[offset:offset + 2]})
            proof = {"batch_id": batch_id, "pass": pass_index, "sample_ids": order[offset:offset + 2],
                "attempt_ids": list(attempt_ids.values()), "status": "COMPLETE",
                "raw": range_proof(output / "raw.jsonl", starts["raw.jsonl"]),
                "events": range_proof(output / "events.jsonl", starts["events.jsonl"])}
            g.append(output / "calls.jsonl", proof)
            wall = time.monotonic() - began
            summaries.append({"batch_id": batch_id, "pass": pass_index, "sample_ids": proof["sample_ids"], "deep": pass_index == 0 and offset == 0,
                "prepare_enqueue_seconds": prepared - began, "wake_and_wait_seconds": finished - prepared,
                "end_to_end_pair_seconds": wall, "requests_per_second": 2 / wall,
                "measurement": "pair wall includes prepare, controls, observer, GC and durable proof; not per-request latency"})
            results.extend(raw_rows)
            g.atomic_json(output / "progress.json", {"completed_requests": len(results), "maximum_requests": 16, "last_batch": summaries[-1]})
    ended = time.monotonic()
    return {"status": "GENERATION_COMPLETE", "calls": len(results), "batches": summaries,
        "whole_run_wall_seconds": ended - run_started, "whole_run_requests_per_second": 16 / (ended - run_started),
        "ordinary_wall_seconds": ended - ordinary_started, "ordinary_wall_requests_per_second": 14 / (ended - ordinary_started),
        "wall_scope": "includes preparation, scheduling barrier, GPU work, observers, GC, proofs and between-pair progress journals; load separate",
        "production_qualified": False, "new_numerical_runtime": "TP2; cannot join TP1 six-cold profile"}


def run(args):
    g.require("torch" not in sys.modules and "vllm" not in sys.modules, "Fresh worker process required")
    profile = validate_profile(g.read(args.profile), True)
    reference, admission = g.read(args.reference), g.read(args.admission)
    g.require(g.file_sha256(args.reference) == REFERENCE_SHA
        and reference["panel_sha256"] == g.digest(profile["qualified_C3"]["panel"]), "Existing CPU reference changed")
    g.require(admission["profile_sha256"] == g.digest(profile) and admission["all_owned_engines_exited"] is True
        and admission["physical_gpu_locks_released"] is True and admission["maximum_requests"] == 16,
        "Explicit two-card 16-request admission missing")
    runtime = profile["runtime"]; versions = runtime_check(profile)
    from .inference import ProcessorSession, checkpoint_asset_identity
    g.require(checkpoint_asset_identity(runtime["checkpoint"]) == profile["qualified_C3"]["checkpoint_assets"], "Checkpoint assets changed")
    output = Path(args.output); g.require(not output.exists(), "No overwrite or implicit resume")
    output.mkdir(parents=True)
    import torch
    from vllm import LLM
    state, complete = None, False
    try:
        with physical_pair(runtime, torch, profile["resource_budget"]) as leases:
            g.atomic_json(output / "identity.json", {"profile": profile, "profile_sha256": g.digest(profile), "leases": leases,
                "reference_sha256": g.file_sha256(args.reference), "admission_sha256": g.file_sha256(args.admission),
                "owner": g.process_identity(), "engine_id": uuid.uuid4().hex, "host": socket.gethostname(), "versions": versions,
                "argv": sys.argv, "declared_environment": runtime["environment"], "python": sys.executable})
            with ExitStack() as cleanup, ProcessorSession("gemma", runtime["checkpoint"]) as session:
                registry = PreparedRegistry(); install_adapter(session, registry)
                began = time.monotonic()
                with StartupChildren(output) as startup:
                    cleanup.callback(startup.wait)
                    engine = LLM(**profile["engine_settings"])
                    owned = []; cleanup.callback(g.shutdown_owned_engine, engine, owned, output)
                    owned.extend(engine.collective_rpc(g._worker_identity))
                    for value in owned:
                        worker_id = value["worker"]
                        startup.owned[(worker_id["pid"], worker_id["startticks"])] = worker_id
                installed = engine.collective_rpc(_install_rank, args=(profile["qualified_C3"]["kernel"], str(output)))
                g.atomic_json(output / "load.json", {"seconds": time.monotonic() - began, "ranks": installed})
                g.require(len(owned) == len(installed) == 2 and {x["identity"]["rank"] for x in installed} == {0, 1}, "Missing TP ranks")
                g.require(all(x["placement"]["torch_policy"] == profile["baseline_torch_policy"] for x in installed), "Native Torch policy changed")
                with c.installed_overlay(profile["qualified_C3"]) as state:
                    state.code, state.line = run_batches.__code__, _gc_line()
                    result = run_batches(engine, session, registry, profile, reference, output)
                    g.atomic_json(output / "result.json", result)
        exited = g.read(output / "exit.json")
        g.require(exited["status"] == "COMPLETE" and exited["owned_engine_workers_exited"], "Actual rank shutdown incomplete")
        exited["lease_released_after_engine_exit"] = True; g.atomic_json(output / "exit.json", exited)
        result["status"] = "COMPLETE"; g.atomic_json(output / "result.json", result)
        complete = True
        return {"status": "COMPLETE", "calls": 16, "output": str(output), "production_qualified": False}
    except BaseException as error:
        status = "DIFFERENT" if isinstance(error, OutputDifference) else "STOPPED"
        close_failed_attempt(output, status)
        g.atomic_json(output / "failure.json", {"status": status, "error_type": type(error).__name__, "error": str(error),
            "unclosed_intents": "UNCERTAIN; inspect durable journals, never retry automatically", "automatic_retry": False})
        raise
    finally:
        if state is not None:
            g.seal(output / "cpu-copy.json", {"status": "COMPLETE" if complete else "STOPPED", "cadence": state.cadence,
                "gc_line": state.line, "bound_to_run_batches": state.code is run_batches.__code__,
                "rows": state.rows, "maximum_pending_ordinary": state.maximum_pending, "final": state.final})
        g.seal(output / "closure.json", {"status": "COMPLETE" if complete else "STOPPED", "files": {
            str(path.relative_to(output)): g.file_sha256(path) for path in output.rglob("*") if path.is_file() and path.name != "closure.json"}})


def compare(path, reference_path):
    root = Path(path); identity, closure = g.read(root / "identity.json"), g.read(root / "closure.json")
    profile = validate_profile(identity["profile"])
    reference = g.read(reference_path)
    g.require(g.file_sha256(reference_path) == REFERENCE_SHA == identity["reference_sha256"], "Existing per-ID CPU reference differs")
    g.require(identity["profile_sha256"] == g.digest(profile) and closure["status"] == "COMPLETE", "Run is not complete")
    for name, sha in closure["files"].items():
        g.require(g.file_sha256(root / name) == sha, "Closed evidence bytes changed")
    exited, result, cpu = (g.read(root / name) for name in ("exit.json", "result.json", "cpu-copy.json"))
    ranks = g.read(root / "load.json")["ranks"]
    g.require(len(ranks) == len(exited["owned_workers"]) == 2
        and {r["identity"]["rank"] for r in ranks} == {0, 1}
        and {r["identity"]["device"] for r in ranks} == {lease["device"] for lease in identity["leases"]}
        and {g.digest(r["identity"]["worker"]) for r in ranks} == {g.digest(worker) for worker in exited["owned_workers"]}
        and g.read(root / "startup-children.json")["status"] == "COMPLETE", "Actual TP rank/physical lease identities or startup cleanup differ")
    for rank in ranks:
        kernel = rank["kernel"]
        g.require(kernel["configuration"] == profile["qualified_C3"]["kernel"] and kernel["observed_native_sha256"] == g.NATIVE_SHA
            and kernel["derived_helpers_sha256"] == "32a56e3cb9501974cf648a3524691e84bd4a8c6dd7ead6c2ecd4b683c7b1ac4f"
            and kernel["derived_launcher_sha256"] == "a9c0fd9df26aae3919a024de045a6bb59e67011c297f2de87f584acf2bfc855e"
            and rank["placement"]["kv"]["capacity_tokens"] >= 131072
            and rank["placement"]["offload_bytes"] == 0 and not rank["placement"]["offloaded_names"]
            and rank["placement"]["torch_policy"] == profile["baseline_torch_policy"]
            and rank["actual_engine"] == ACTUAL_ENGINE
            and rank["observer"]["source_sha256"] == g.file_sha256(__file__) and rank["observer"]["extra_sources"] == EXTRA_SOURCES
            and rank["observer"]["integration"] == observer_interface() and rank["observer"]["actual_types"] == V2_TYPES,
            "Actual kernel/placement/observer differs")
    g.require(not (root / "failure.json").exists() and exited["status"] == result["status"] == cpu["status"] == "COMPLETE"
        and exited["owned_engine_workers_exited"] and exited["lease_released_after_engine_exit"], "Actual two-rank exit required")
    raw, calls = read_lines(root / "raw.jsonl"), read_lines(root / "calls.jsonl")
    g.require(len(raw) == result["calls"] == 16 and len(calls) == 8
        and [[r["sample_id"] for r in raw if r["pass"] == i] for i in range(2)] == profile["orders"], "Finite two-pass scope differs")
    previous = {"raw": 0, "events": 0}; attempts = set()
    for proof in calls:
        g.require(proof["status"] == "COMPLETE", "Unclosed/uncertain pair proof")
        blobs = {}
        for key, filename in (("raw", "raw.jsonl"), ("events", "events.jsonl")):
            value = proof[key]; g.require(value["start"] == previous[key], "Call proof gap/overlap")
            with (root / filename).open("rb") as f: f.seek(value["start"]); data = f.read(value["end"] - value["start"])
            g.require(hashlib.sha256(data).hexdigest() == value["sha256"] and data.endswith(b"\n"), "Call byte proof differs")
            blobs[key] = [json.loads(line) for line in data.splitlines()]; previous[key] = value["end"]
        intents = [row for row in blobs["events"] if row["event"] == "GENERATE_INTENT"]
        g.require(len(intents) == len(blobs["raw"]) == 2 and [x["sample_id"] for x in intents] == proof["sample_ids"]
            == [x["sample_id"] for x in blobs["raw"]]
            and [x["attempt_id"] for x in intents] == proof["attempt_ids"] == [x["attempt_id"] for x in blobs["raw"]]
            and not attempts.intersection(proof["attempt_ids"]), "Intent/output identity differs")
        attempts.update(proof["attempt_ids"])
        expected = {row["internal_request_id"]: row for row in intents}
        gpu = [row for row in blobs["events"] if row["event"] == "ACTUAL_PAIR_GPU_INPUT"]
        g.require(len(expected) == len({row["external_id"] for row in intents}) == 2 and len(gpu) == 1
            and len([row for row in blobs["events"] if row["event"] == "PAIR_TECHNICAL_PASS"]) == 1, "Missing batch technical proof")
        for intent in intents:
            validate_sampling_receipt(intent["actual_sampling"])
        validate_observations(gpu[0]["observations"], expected, reference)
        for row in blobs["raw"]:
            ref = reference["rows"][row["sample_id"]]
            g.verify_input(row["input_receipt"], ref, row["deep"])
            g.require(row["actual_input_token_ids"] == ref["input_token_ids"]
                and expected[row["internal_request_id"]]["external_id"] == row["external_request_id"], "Per-ID raw control association differs")
    g.require(previous["raw"] == (root / "raw.jsonl").stat().st_size and previous["events"] == (root / "events.jsonl").stat().st_size,
        "Unclosed journal tail")
    g.require(cpu["bound_to_run_batches"] and cpu["gc_line"] == _gc_line() and cpu["cadence"] == 8
        and cpu["maximum_pending_ordinary"] <= 8 and cpu["final"]["status"] == "COMPLETE"
        and [x["deep"] for x in cpu["rows"]] == [i < 2 for i in range(16)]
        and [x["sample_id"] for x in cpu["rows"]] == [x["sample_id"] for x in raw], "Pair GC lifecycle differs")
    differences = [{"pass": row["pass"], "sample_id": row["sample_id"], "fields": [k for k in OUTPUT_KEYS if row[k] != profile["baseline"][row["sample_id"]][k]]}
        for row in raw if any(row[k] != profile["baseline"][row["sample_id"]][k] for k in OUTPUT_KEYS)]
    return {"status": "DIFFERENT" if differences else "MATCH", "unique_IDs": 8, "observations_per_ID": 2, "calls": 16,
        "output_differences": differences, "batch_timing": result["batches"], "production_qualified": False,
        "remaining": "Separate selected TP2 cold48 and cross-profile/companion invariance; this is not TP1 three-card six-cold PASS"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-profile")
    for name in ("baseline-run", "reference", "name", "output"): seal.add_argument("--" + name, required=True)
    seal.add_argument("--devices", default="0,1"); seal.add_argument("--panel")
    seal.add_argument("--disable-p2p", action="store_true",
        help="Seal process-local NCCL_P2P_DISABLE=1 and NCCL_DEBUG=INFO; no automatic fallback")
    seal.add_argument("--kv-cache-gib", type=int, choices=(32, 56), default=32,
        help="Explicit KV cache GiB per rank; actual capacity must still reach 131072 tokens")
    run_cli = sub.add_parser("run")
    for name in ("profile", "reference", "admission", "output"): run_cli.add_argument("--" + name, required=True)
    report = sub.add_parser("compare"); report.add_argument("run"); report.add_argument("--reference", required=True); report.add_argument("--output")
    args = parser.parse_args()
    if args.command == "seal-profile": result = seal_profile(args)
    elif args.command == "run": result = run(args)
    else:
        result = compare(args.run, args.reference)
        if args.output: g.seal(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__": main()
