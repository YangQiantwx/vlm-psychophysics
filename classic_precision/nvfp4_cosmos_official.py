"""Finite official Cosmos NVFP4/vLLM run against the sealed historical BF16 oracle.

Imports and seal/check/compare are CPU metadata only. Run is an explicit GPU
operation. The original RGB16 processor, T1 probes, IPC transport, physical lease
and RunStore own their existing responsibilities; no checkpoint or SDK is edited.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import gc
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import time

from . import inference, nvfp4_cosmos as native
from .adapter_qsa_replay_ipc import register_native_module
from .storage import RunStore, canonical_json, digest, file_sha256, owner_identity

SCHEMA = "cosmos-official-nvfp4-historical-v1"
MODEL_KEY = "cosmos-official-nvfp4"
PROTOCOL_SHA = "77329e75e5f4c789542639a1fa2b9f52b9ff8236c0147d8f8b987a779bc0368d"
ORACLE_SHA = "2a9c6838f66918a403f6af82a2b71662f803bb99dc74241331dff9dcea71a9b0"
LINEAGE_SHA = "0f753e1e740f9b847d762047487d3325badc1cf005b730365e253c14e11c460d"
ASSET_SHA = "7bbfe34d27a6464169e14d38432761e22cfc1c7350b96ccffe0d5e7484c27302"
T1_SHA = "8a9e93a9e97cbed08fedc4d346def08af74fb4250fe8b08e0725d8ff0082430d"
require, read, write_new = native.require, native.read, native.write_new
SDK_SOURCES = {
    "vllm": ("model_executor/models/qwen3_vl.py", "model_executor/layers/quantization/modelopt.py",
             "model_executor/kernels/linear/nvfp4/flashinfer.py", "multimodal/processing/context.py",
             "multimodal/processing/processor.py", "model_executor/layers/quantization/kv_cache.py"),
    "transformers": ("models/qwen3_vl/processing_qwen3_vl.py", "models/qwen3_vl/video_processing_qwen3_vl.py"),
}
ENV_KEYS = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG",
            "VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING", "OMP_NUM_THREADS",
            "VLLM_OMP_NUM_THREADS_SET_BY_VLLM", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
DELTA = {
    "weights": "Official calibrated ModelOpt NVFP4 W4A4; vision/lm_head and other excluded tensors BF16",
    "backend": "Historical HF Transformers SDPA -> native vLLM Qwen3VL / FlashInfer CUTLASS",
    "kv": "Published FP8 KV explicitly overridden to BF16",
    "history": "432 historical BF16 outputs; mixed hosts/load and different timing scopes prevent a pure quantization speed causal claim",
    "lineage_limit": "610 unquantized tensors byte-identical; quantized payload derivation/calibration not independently reproduced",
}


def descriptor(path, expected=None):
    path = Path(path).resolve()
    observed = file_sha256(path)
    require(expected is None or observed == expected, "Source digest differs: " + str(path))
    return {"path": str(path), "sha256": observed}


def verified(ref):
    require(descriptor(ref["path"])["sha256"] == ref["sha256"], "Bound file changed")
    return Path(ref["path"])


def file_state(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Expected regular checkpoint/lock file: " + str(path))
    st = path.stat()
    return {"st_dev": st.st_dev, "st_ino": st.st_ino, "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def checkpoint_identity(runtime):
    root = Path(runtime["checkpoint"])
    manifest = read(root / "NGC_ASSET_MANIFEST.json")
    require(manifest["asset_manifest_sha256"] == ASSET_SHA and len(manifest["files"]) == 17,
            "Official NGC asset inventory differs")
    result = {}
    for entry in manifest["files"]:
        name = entry["path"]
        require(Path(name).name == name, "Checkpoint filename escapes root")
        state = file_state(root / name)
        require(state["bytes"] == entry["bytes"], "Checkpoint size changed: " + name)
        if not name.endswith(".safetensors"):
            require(file_sha256(root / name) == entry["sha256"], "Checkpoint metadata changed: " + name)
        result[name] = {**entry, "stat": state}
    return result


def runtime_identity(runtime):
    versions = {n: importlib.metadata.version(n) for n in
                ("torch", "transformers", "vllm", "numpy", "av", "safetensors", "cloudpickle", "flashinfer-python")}
    require(versions["torch"].split("+")[0] == "2.13.0" and versions["transformers"] == "5.17.0"
            and versions["vllm"].split("+")[0] == "0.31.0", "Qualified shared framework changed")
    sources = {}
    for package, names in SDK_SOURCES.items():
        root = Path(importlib.metadata.distribution(package).locate_file(package))
        sources[package] = {n: file_sha256(root / n) for n in names}
    for package in ("flashinfer-cubin", "flashinfer-jit-cache"):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    require(Path(runtime["original_checkpoint"]).name == native.REVISION, "Original f543 processor required")
    require(file_sha256(runtime["native_t1_adapter"]) == T1_SHA, "Original T1 probe source changed")
    return {"packages": versions, "sdk_sources": sources,
            "implementation": {str(Path(p).name): file_sha256(p) for p in
                (__file__, native.__file__, inference.__file__, Path(__file__).with_name("storage.py"),
                 Path(__file__).with_name("adapter_qsa_replay_ipc.py"))},
            "checkpoint": checkpoint_identity(runtime),
            "original_checkpoint": inference.checkpoint_asset_identity(runtime["original_checkpoint"]),
            "cosmos_package": inference._cosmos_package_identity(runtime["cosmos_package"]),
            "native_t1_sha256": T1_SHA, "environment": {k: os.environ.get(k) for k in ENV_KEYS}}


def engine_settings(runtime):
    return dict(model=runtime["checkpoint"], tokenizer=runtime["original_checkpoint"],
        dtype="bfloat16", quantization="modelopt_fp4", linear_backend="flashinfer_cutlass",
        kernel_config={"enable_flashinfer_autotune": False},
        trust_remote_code=False, tensor_parallel_size=1, max_model_len=65536,
        max_num_seqs=1, max_num_batched_tokens=36000, kv_cache_dtype="bfloat16",
        kv_cache_memory_bytes=17 * 2**30, gpu_memory_utilization=.80,
        enforce_eager=True, compilation_config={"mode": 0}, enable_prefix_caching=False,
        enable_chunked_prefill=True, disable_chunked_mm_input=True, async_scheduling=False,
        cpu_offload_gb=0, skip_mm_profiling=True, disable_custom_all_reduce=True, seed=0,
        limit_mm_per_prompt={"video": 1, "image": 0, "audio": 0}, mm_processor_cache_gb=0,
        video_pruning_rate=0., mm_processor_kwargs={"do_resize": False, "do_sample_frames": False})


def validate_oracle(oracle, panel):
    items = {x["sample_id"]: x for x in panel["items"]}
    require(set(oracle["items"]) == set(items) and len(items) == 432, "Historical oracle lacks exact432 coverage")
    require(oracle["checkpoint"]["revision"] == native.REVISION, "Historical checkpoint changed")
    for sid, item in items.items():
        old = oracle["items"][sid]
        require(old["truth"] == native.truth(item) and old["task"] == item["task"], "Oracle truth/task mismatch")
        for field in ("row_sha256", "ready_sha256", "source_stream_sha256"):
            require(old["selected_source"][field] == item[field], "Oracle input binding mismatch: " + field)
        require(isinstance(old["output_token_ids"], list) and isinstance(old["raw_answer"], str), "Incomplete historical output")
    return items


def seal(args):
    refs = {name: descriptor(getattr(args, name), expected) for name, expected in
            (("catalog", native.CATALOG_SHA), ("protocol", PROTOCOL_SHA), ("oracle", ORACLE_SHA),
             ("lineage", LINEAGE_SHA), ("download_receipt", None))}
    protocol, catalog = read(args.protocol), read(args.catalog)
    panel = native.extract_panel(protocol)
    native.validate_panel(panel, catalog)
    validate_oracle(read(args.oracle), panel)
    runtime = read(args.runtime)
    require(runtime["asset_roots"] == protocol["source_roots"]["fourier"], "Frozen asset roots changed")
    require(Path(runtime["output_root"]).is_absolute(), "Explicit unique absolute output root required")
    require(runtime["resources"]["min_free_gpu_gib"] >= 80 and runtime["resources"]["min_available_ram_gib"] >= 20,
            "Original resource margins must not be lowered")
    for w in ("0", "1"):
        lock = runtime["physical_locks"][w]
        state = file_state(lock["path"])
        require(all(state[k] == lock[k] for k in ("st_dev", "st_ino")), "Original physical lock inode changed")
    lineage, downloaded = read(args.lineage), read(args.download_receipt)
    require(lineage["status"] == "UNQUANTIZED_BYTES_AND_LOGICAL_SHAPES_MATCH"
            and lineage["new_checkpoint"] == runtime["checkpoint"]
            and lineage["old_checkpoint"] == runtime["original_checkpoint"], "Lineage receipt/checkpoints differ")
    require(downloaded["asset_manifest_sha256"] == ASSET_SHA and downloaded["complete_files"] == 17
            and not downloaded["errors"] and downloaded["bytes_received"] == 21869085423, "Official download incomplete")
    assets = checkpoint_identity(runtime)
    require(all(downloaded["files"][k]["sha256"] == v["sha256"] and downloaded["files"][k]["status"] == "COMPLETE"
                for k, v in assets.items()), "Download vendor hashes differ")
    workers = native.assignments(panel["items"], panel["selected_cells"])
    plan = {"schema": SCHEMA, "references": refs, "runtime": runtime, "runtime_identity": runtime_identity(runtime),
            "panel": panel, "assignments": workers, "generation": inference.GENERATION,
            "engine_settings": engine_settings(runtime), "max_new_calls": 432, "max_new_calls_per_id": 1,
            "prior_failed_nvfp4_attempts": {workers[0][0]: 1}, "max_total_attempts_per_id": 2,
            "new_bf16_calls": 0, "automatic_retry": False, "deep_calls_per_fresh_worker": 2, "deltas": DELTA}
    write_new(args.output, plan)
    return {"path": str(args.output), "sha256": file_sha256(args.output), "max_new_calls": 432}


def validate_plan(plan, physical=True):
    require(plan["schema"] == SCHEMA and plan["max_new_calls"] == 432 and plan["max_new_calls_per_id"] == 1
            and plan["max_total_attempts_per_id"] == 2 and plan["new_bf16_calls"] == 0
            and plan["automatic_retry"] is False and plan["deep_calls_per_fresh_worker"] == 2, "Finite contract differs")
    require(plan["generation"] == inference.GENERATION and plan["engine_settings"] == engine_settings(plan["runtime"])
            and plan["deltas"] == DELTA, "Scientific/backend controls differ")
    require(plan["assignments"] == native.assignments(plan["panel"]["items"], plan["panel"]["selected_cells"]), "Worker assignment differs")
    require(plan["prior_failed_nvfp4_attempts"] == {plan["assignments"][0][0]: 1}, "Historical failed attempt accounting differs")
    for name, expected in (("catalog", native.CATALOG_SHA), ("protocol", PROTOCOL_SHA), ("oracle", ORACLE_SHA), ("lineage", LINEAGE_SHA)):
        require(plan["references"][name]["sha256"] == expected, "Frozen scientific reference differs")
    if physical:
        for ref in plan["references"].values(): verified(ref)
        protocol = read(plan["references"]["protocol"]["path"])
        require(plan["panel"] == native.extract_panel(protocol), "Frozen panel content differs")
        validate_oracle(read(plan["references"]["oracle"]["path"]), plan["panel"])
        require(plan["runtime_identity"] == runtime_identity(plan["runtime"]), "Sealed runtime/source/file state changed")


def _configure_torch(torch):
    torch.set_num_threads(6)
    torch.default_generator.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    require(native.torch_policy(torch) == native.NATIVE_TORCH_POLICY, "Original Cosmos Torch policy differs")


def _engine_receipt(worker):
    import torch
    _configure_torch(torch)
    model = worker.get_model()
    kernels, excluded, attention = Counter(), Counter(), []
    for name, module in model.named_modules():
        method = getattr(module, "quant_method", None)
        kernel = getattr(method, "kernel", None)
        if kernel is not None: kernels[type(kernel).__name__] += 1
        if hasattr(module, "kv_cache_dtype"): attention.append(str(module.kv_cache_dtype))
    for name, value in model.named_parameters():
        if name.startswith(("visual.", "language_model.lm_head.")):
            excluded[str(value.dtype)] += value.numel() * value.element_size()
    caches = getattr(worker.model_runner, "kv_caches", None)
    def dtypes(value):
        if isinstance(value, torch.Tensor): return [str(value.dtype)]
        if isinstance(value, dict): return [x for v in value.values() for x in dtypes(v)]
        if isinstance(value, (list, tuple)): return [x for v in value for x in dtypes(v)]
        return []
    return {"model_class": type(model).__name__, "kernels": dict(kernels), "excluded_bytes_by_dtype": dict(excluded),
            "attention_kv_dtypes": dict(Counter(attention)), "actual_cache_dtypes": sorted(set(dtypes(caches))),
            "capacity": inference._kv_capacity(worker), "weights": inference._weight_receipt(worker),
            "torch_policy": native.torch_policy(torch), "torch_threads": torch.get_num_threads()}


def validate_engine_receipt(records):
    require(len(records) == 1, "Exactly one TP1 worker required")
    r = records[0]
    require(r["model_class"] == "Qwen3VLForConditionalGeneration", "Unexpected model implementation")
    require(r["kernels"] == {"FlashInferCutlassNvFp4LinearKernel": 256}, "Native W4A4 kernel/coverage differs; no fallback")
    require(set(r["excluded_bytes_by_dtype"]) == {"torch.bfloat16"} and r["excluded_bytes_by_dtype"]["torch.bfloat16"] > 0,
            "Vision/lm_head precision changed")
    require(r["attention_kv_dtypes"] == {"bfloat16": 64} and r["actual_cache_dtypes"] == ["torch.bfloat16"], "Actual KV is not entirely BF16")
    require((r["capacity"]["capacity_tokens"] or 0) >= 65536, "Actual KV capacity below65536")
    require(r["torch_policy"] == native.NATIVE_TORCH_POLICY and r["torch_threads"] == 6, "Worker numerical policy differs")


def _native_probe(runtime):
    name = "_cosmos_official_original_t1_probe"
    require(name not in sys.modules and file_sha256(runtime["native_t1_adapter"]) == T1_SHA, "T1 probe identity differs")
    spec = importlib.util.spec_from_file_location(name, runtime["native_t1_adapter"])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    register_native_module(module, T1_SHA)
    return module


class OfficialSession(native.CosmosSession):
    """Reuse the original processor and deep prepare; replace only model execution."""
    def __init__(self, runtime, store):
        self.runtime = runtime
        self.probe = _native_probe(runtime)
        inference.ModelSession.__init__(self, "cosmos", runtime["original_checkpoint"], store,
            cosmos_package=runtime["cosmos_package"], resources=runtime["resources"])

    def _load_hf(self):
        # The inherited dispatcher calls this for family=cosmos. No HF model is loaded.
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        _configure_torch(self.torch)
        self.engine = LLM(**engine_settings(self.runtime))
        actual = self.engine.collective_rpc(_engine_receipt)
        validate_engine_receipt(actual)
        self.store.event("MODEL_LOADED", backend="vllm", actual=actual, settings=engine_settings(self.runtime), deltas=DELTA)
        self.sampling = SamplingParams(n=1, temperature=0., top_p=1., repetition_penalty=1., seed=0,
            max_tokens=16384, structured_outputs=StructuredOutputsParams(choice=["LEFT", "RIGHT"]))
        choices = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("LEFT", "RIGHT")]
        eos = read(Path(self.runtime["original_checkpoint"]) / "generation_config.json")["eos_token_id"]
        require(choices == [[22397], [11572]] and eos == [151645, 151643], "Original choice/EOS tokenizer contract changed")
        self.effective_generation = {**inference.GENERATION, "backend": "vllm",
            "constraint": "StructuredOutputsParams(choice)", "choice_token_sequences": choices,
            "native_eos_token_ids": eos, "sampling_params_repr": repr(self.sampling)}
        original = type(self.processor).__call__
        def checked(processor, *args, **kwargs):
            started = time.monotonic()
            data = original(processor, *args, **kwargs)
            if self.context.get("engine_active") and "pixel_values_videos" in data:
                self._guard_pixels(data, processor, engine=True)
                self.context["engine_processor_seconds"] += time.monotonic() - started
            return data
        self._patch(type(self.processor), "__call__", checked)

    def generate(self, row, ready, asset_root, durable_output):
        began_all = time.monotonic()
        frames, data, text, receipt = self.prepare(row, ready, asset_root)
        expected_ids = data["input_ids"][0].tolist()
        expected = self.probe.expected_hashes({"pixels": data["pixel_values_videos"], "batch": data}, "cosmos") if self.context["deep"] else None
        del data
        began = time.monotonic()
        self.engine.collective_rpc(self.probe._set_deep, args=(self.context["deep"],))
        require(all(self.engine.collective_rpc(self.probe._install_strong_probe)), "Native GPU probe installation failed")
        self.context.update(engine_active=True, engine_receipt=None, engine_processor_seconds=0.)
        infer_start = time.monotonic()
        probe_setup_seconds = infer_start - began
        answer = self.engine.generate({"prompt": text, "multi_modal_data": {"video": [(frames, inference._video_metadata())]}},
                                      self.sampling, use_tqdm=False)[0]
        infer_seconds = time.monotonic() - infer_start
        output = answer.outputs[0]
        actual_ids = list(answer.prompt_token_ids)
        post_prepare_seconds = time.monotonic() - began
        record = {"task": row["task"], "truth": row["truth"], "raw_answer": output.text,
            "output_token_ids": list(output.token_ids), "finish_reason": output.finish_reason, "stop_reason": output.stop_reason,
            "input_receipt": receipt, "effective_generation": self.effective_generation,
            "actual_input_tokens": len(actual_ids), "actual_prompt_token_ids_sha256": digest(actual_ids),
            "all_prompt_token_ids_equal": actual_ids == expected_ids,
            "seconds": post_prepare_seconds, "post_prepare_to_decoded_output_seconds": post_prepare_seconds,
            "engine_generate_seconds": infer_seconds, "probe_setup_seconds": probe_setup_seconds,
            "engine_processor_seconds": self.context["engine_processor_seconds"],
            "engine_generate_includes": ["vLLM processor and its guards", "IPC", "H2D", "model generation", "detokenization"],
            "post_prepare_less_processor_seconds": post_prepare_seconds - self.context["engine_processor_seconds"],
            "prepare_to_raw_seconds": time.monotonic() - began_all,
            "timing_scope": "post_prepare_to_decoded_output_including_vllm_processor_and_probe_setup"}
        committed = durable_output(record)  # Raw must survive every subsequent check.
        forwards = [x for worker in self.engine.collective_rpc(self.probe._read_strong_probe) for x in worker]
        self.store.event("POST_OUTPUT_INPUT_RECEIPT", sample_id=row["sample_id"], attempt_id=committed["attempt_id"],
            actual_GPU_visual_forward=forwards, actual_engine_processor_receipt=self.context["engine_receipt"],
            expected_BF16_hashes=expected, all_prompt_token_ids_equal=actual_ids == expected_ids)
        require(actual_ids == expected_ids and len(actual_ids) + 16384 <= 65536, "Actual full prompt/context differs")
        require(self.context["engine_receipt"] is not None, "Actual native processor guard did not run")
        inference._validate_forwards("cosmos", forwards)
        if expected:
            observed = forwards[0]["t1_hashes"]
            require(observed["deep"] and all(observed[k] == expected[k] for k in ("input_sha256", "projection_sha256")),
                    "Actual BF16 vision/projection input differs from original processor")
        self.store.event("TECHNICAL_PASS", sample_id=row["sample_id"], attempt_id=committed["attempt_id"],
                         raw_sha256=committed["raw_record_sha256"])
        self.context.update(active=False, engine_active=False, codes=None)
        del frames
        gc.collect()


class _EventSink:
    """Track only this execution's durable technical verdicts in memory."""
    def __init__(self, store, execution):
        self.store, self.execution, self.passed = store, execution, set()

    def event(self, name, **fields):
        self.store.event(name, execution=self.execution, **fields)
        if name == "TECHNICAL_PASS":
            self.passed.add((fields["sample_id"], fields["attempt_id"], fields["raw_sha256"]))


def select_pending(store, ids, limit):
    require(type(limit) is int and 0 < limit <= len(ids), "Positive bounded segment limit required")
    pending = store.pending_ids(retry_failed=False, retry_interrupted=False)
    attempted = {e["sample_id"] for e in store.events if e["event"] == "INFLIGHT"}
    passes = {(e["sample_id"], e.get("attempt_id"), e.get("raw_sha256")) for e in store.events if e["event"] == "TECHNICAL_PASS"}
    completed = []
    for sid in ids:
        if sid in store.outputs:
            r = store.outputs[sid]
            require((sid, r["attempt_id"], digest(r)) in passes, "Existing raw lacks bound technical pass; never retry")
            completed.append(sid)
        else:
            require(sid not in attempted, "Prior failed/interrupted request held; automatic retry forbidden")
    require(completed == ids[:len(completed)], "Completed outputs are not the assigned prefix")
    require(pending == ids[len(completed):], "Pending scientific order changed")
    require(bool(pending), "Worker already complete; no model should be loaded")
    return pending[:limit]


@contextmanager
def _signals():
    def stop(signum, frame): raise InterruptedError("Controller stop signal " + str(signum))
    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try: yield
    finally:
        for s, handler in previous.items(): signal.signal(s, handler)


def execute_segment(plan, plan_sha, worker, limit, store, torch, execution, session_factory=OfficialSession):
    ids = select_pending(store, plan["assignments"][worker], limit)
    lookup = {x["sample_id"]: x for x in plan["panel"]["items"]}
    runtime = plan["runtime"]
    completed, session, acquired = [], None, False
    sink = _EventSink(store, execution)
    try:
      with native.physical_lease(runtime, torch) as lease:
        acquired = True
        expected = runtime["physical_locks"][str(worker)]
        require(all(lease[k] == expected[k] for k in ("path", "st_dev", "st_ino")), "Original physical lease identity differs")
        store.event("LEASE_ACQUIRED", execution=execution, lease=lease, owner=store.owner)
        try:
            session = session_factory(runtime, sink)
            for ordinal, sid in enumerate(ids):
                item = lookup[sid]
                row, ready = native.normalized_input(item)
                attempt = store.begin(sid, digest({"row": row, "ready": ready}))
                call_start = time.monotonic()
                session.context["deep"] = ordinal < 2
                def commit(record):
                    return store.commit_output(sid, {**record, "worker": worker, "execution": execution,
                        "deep": ordinal < 2, "plan_sha256": plan_sha, "selected_cell_id": item["selected_cell_id"],
                        "catalog_row_sha256": item["row_sha256"], "catalog_ready_sha256": item["ready_sha256"],
                        "source_family": item["source_family"], "overall_nvfp4_attempt_number": 1 + plan["prior_failed_nvfp4_attempts"].get(sid, 0),
                        "full_request_seconds": time.monotonic() - call_start})
                session.generate(row, ready, Path(runtime["asset_roots"][item["source_family"]]), commit)
                require(sid in store.outputs, "Session returned without durable raw")
                raw = store.outputs[sid]
                require((sid, raw["attempt_id"], digest(raw)) in sink.passed, "Session lacks durable raw-bound technical pass")
                if ordinal < 2:
                    require(raw["finish_reason"] == "stop" and raw["raw_answer"] in ("LEFT", "RIGHT"),
                            "Qualification did not stop with an exact choice; raw retained, no retry")
                store.event("ITERATION_COST", sample_id=sid, attempt_id=attempt, execution=execution,
                    seconds=time.monotonic() - call_start, raw_sha256=digest(store.outputs[sid]))
                completed.append(sid)
                if ordinal == 1: store.event("FIRST_TWO_QUALIFIED", execution=execution, IDs=ids[:2])
        except BaseException as exc:
            store.event("FAILED_NO_RETRY", execution=execution, error_type=type(exc).__name__, error=str(exc), completed=completed)
            raise
        finally:
            if session is not None: session.close()
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            store.event("MODEL_CLOSED_BEFORE_LEASE_RELEASE", execution=execution)
    finally:
        if acquired: store.event("LEASE_RELEASED", execution=execution)
    return completed


def run(args):
    plan = read(args.plan)
    validate_plan(plan)
    require(args.worker in (0, 1) and re.fullmatch(r"[A-Za-z0-9_.-]+", args.execution), "Invalid worker/execution")
    admission = read(args.admission)
    plan_sha = file_sha256(args.plan)
    require(admission["plan_sha256"] == plan_sha and all(admission[k] is True for k in
        ("production_held", "owned_engines_exited", "physical_locks_released")), "Controller admission missing")
    require(admission["workers"][str(args.worker)]["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES"), "Wrong physical GPU")
    root = Path(plan["runtime"]["output_root"])
    run_dir = root / "executions" / args.execution
    run_dir.mkdir(parents=True, exist_ok=False)
    binding = {"schema": SCHEMA, "plan_sha256": plan_sha, "worker": args.worker, "execution": args.execution,
               "owner": owner_identity(), "admission_sha256": file_sha256(args.admission), "limit": args.limit}
    write_new(run_dir / "binding.json", binding)
    identity = {"schema": SCHEMA, "plan_sha256": plan_sha, "deltas": DELTA}
    all_ids = plan["assignments"][0] + plan["assignments"][1]
    status, completed, started = "FAILED", [], time.monotonic()
    try:
        with RunStore(root, identity, MODEL_KEY, "worker" + str(args.worker), plan["assignments"][args.worker], all_ids) as store, _signals():
            # Refuse held/complete work before importing Torch or acquiring a GPU.
            select_pending(store, plan["assignments"][args.worker], args.limit)
            os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
            import torch
            require(torch.version.cuda == "13.0", "Qualified CUDA13.0 build required")
            _configure_torch(torch)
            completed = execute_segment(plan, plan_sha, args.worker, args.limit, store, torch, args.execution)
            status = "SEGMENT_COMPLETE"
            store.event(status, execution=args.execution, IDs=completed, elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        write_new(run_dir / "failed.json", {**binding, "status": "FAILED", "error_type": type(exc).__name__, "error": str(exc),
            "elapsed_seconds": time.monotonic() - started, "automatic_retry": False})
        raise
    finally:
        write_new(run_dir / "worker_return.json", {**binding, "status": status, "returned_unix": time.time(),
            "process_exit_observed": False, "note": "Controller must record actual wait status and descendant closure; this is not exit proof."})
    write_new(run_dir / "complete.json", {**binding, "status": status, "IDs": completed, "new_model_calls": len(completed)})
    return {"status": status, "IDs": completed, "new_model_calls": len(completed), "execution_dir": str(run_dir)}


def compare_records(oracle, records):
    old = oracle["items"]
    require(len(records) == 432 and len({r["sample_id"] for r in records}) == 432 and {r["sample_id"] for r in records} == set(old),
            "Exactly432 complete unique outputs required")
    rows = []
    for r in records:
        a = old[r["sample_id"]]
        require(r["truth"] == a["truth"], "Truth changed")
        for key, source in (("catalog_row_sha256", "row_sha256"), ("catalog_ready_sha256", "ready_sha256")):
            require(r[key] == a["selected_source"][source], "Historical source binding changed")
        for key in ("source_video_sha256", "source_stream_sha256"):
            require(r["input_receipt"][key] == a["selected_source"][key], "Historical video identity differs")
        token_hash = a.get("actual_prompt_token_ids_sha256")
        if token_hash: require(token_hash == r["actual_prompt_token_ids_sha256"], "Observed historical prompt differs")
        valid = r["finish_reason"] == "stop" and r["raw_answer"].strip() in ("LEFT", "RIGHT")
        rows.append({"sample_id": r["sample_id"], "task": r["task"], "cell": r["selected_cell_id"], "truth": r["truth"],
            "historical_host": a["historical_host"], "new_worker": r["worker"], "deep": r["deep"],
            "bf16_answer": a["raw_answer"], "nvfp4_answer": r["raw_answer"], "answer_equal": a["raw_answer"] == r["raw_answer"],
            "bf16_tokens": a["output_token_ids"], "nvfp4_tokens": r["output_token_ids"], "tokens_equal": a["output_token_ids"] == r["output_token_ids"],
            "bf16_finish": a["finish_reason"], "nvfp4_finish": r["finish_reason"], "nvfp4_stop_reason": r["stop_reason"],
            "bf16_correct": a["correct"], "nvfp4_correct": valid and r["raw_answer"].strip() == r["truth"], "nvfp4_valid": valid,
            "bf16_seconds": a["seconds"], "nvfp4_seconds": r["seconds"], "nvfp4_engine_generate_seconds": r["engine_generate_seconds"],
            "nvfp4_full_request_seconds": r["full_request_seconds"],
            "historical_to_new_latency_ratio": a["seconds"] / r["seconds"],
            "historical_prompt_hash_available": bool(token_hash), "historical_GPU_content_hash_available": a["actual_GPU_content_hash_available"]})
    def summary(rs):
        return {"N": len(rs), "answers_changed": sum(not x["answer_equal"] for x in rs), "tokens_changed": sum(not x["tokens_equal"] for x in rs),
            "bf16_correct": sum(x["bf16_correct"] for x in rs), "nvfp4_correct": sum(x["nvfp4_correct"] for x in rs),
            "bf16_accuracy": statistics.mean(x["bf16_correct"] for x in rs), "nvfp4_accuracy": statistics.mean(x["nvfp4_correct"] for x in rs),
            "nvfp4_invalid": sum(not x["nvfp4_valid"] for x in rs),
            "historical_to_new_seconds_ratio": sum(x["bf16_seconds"] for x in rs) / sum(x["nvfp4_seconds"] for x in rs),
            "bf16_seconds_median": statistics.median(x["bf16_seconds"] for x in rs), "nvfp4_seconds_median": statistics.median(x["nvfp4_seconds"] for x in rs)}
    steady = [r for r in rows if not r["deep"]]
    return {"schema": SCHEMA + "/comparison", "rows": rows, "all": summary(rows), "ordinary": summary(steady) if steady else None,
        "tasks": {t: summary([r for r in rows if r["task"] == t]) for t in sorted({r["task"] for r in rows})},
        "deltas": DELTA, "timing_scope_mismatch": True, "historical_timing_definition": oracle["timing_definition"],
        "interpretation": "Paired-ID output/accuracy comparison; historical speed ratios include backend, processor timing and host/load confounds. Output differences are results, not failed runs."}


def compare(args):
    plan = read(args.plan)
    validate_plan(plan, physical=False)
    oracle = read(verified(plan["references"]["oracle"]))
    root = Path(plan["runtime"]["output_root"]) / MODEL_KEY
    records = [json.loads(s) for s in (root / "outputs.jsonl").read_text().splitlines()]
    events = [json.loads(s) for s in (root / "events.jsonl").read_text().splitlines()]
    passed = {(e.get("sample_id"), e.get("attempt_id"), e.get("raw_sha256")) for e in events if e["event"] == "TECHNICAL_PASS"}
    require(all((r["sample_id"], r["attempt_id"], digest(r)) in passed for r in records), "Technical/raw binding incomplete")
    executions = {r["execution"] for r in records}
    for execution in executions:
        require(any(e["event"] == "LEASE_RELEASED" and e.get("execution") == execution for e in events), "Execution lease closure missing")
    report = compare_records(oracle, records)
    report.update(plan_sha256=file_sha256(args.plan), model_ready_events=[e for e in events if e["event"] == "MODEL_READY"])
    write_new(args.output, report)
    return {k: v for k, v in report.items() if k not in ("rows", "model_ready_events")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seal")
    for key in ("catalog", "protocol", "oracle", "runtime", "lineage", "download-receipt", "output"):
        p.add_argument("--" + key, required=True)
    p = sub.add_parser("check"); p.add_argument("--plan", required=True)
    p = sub.add_parser("run")
    for key in ("plan", "admission", "execution"): p.add_argument("--" + key, required=True)
    p.add_argument("--worker", type=int, choices=(0, 1), required=True)
    p.add_argument("--limit", type=int, required=True)
    p = sub.add_parser("compare")
    p.add_argument("--plan", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    result = {"seal": seal, "run": run, "compare": compare}.get(args.command)
    if result is None:
        validate_plan(read(args.plan)); value = {"status": "CPU_METADATA_CHECK_PASS", "model_calls": 0}
    else: value = result(args)
    print(canonical_json(value))
    return value


if __name__ == "__main__":
    main()
