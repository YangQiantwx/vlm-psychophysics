"""Finite paired Cosmos BF16/NVFP4 experiment, using native HF and SDPA.

``seal`` is metadata-only. ``run`` is explicit, one physical GPU, one attempt per
ID, and never downloads. Native HF NVFP4 quantizes the original floating-point
checkpoint at load time; output differences are measured, not rejected. The
controller owns package/kernel provisioning, GPU admission, and the two workers.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket
import statistics
import time

from . import inference
from .storage import canonical_json, digest, file_sha256

SCHEMA = "cosmos-native-nvfp4-v1"
REVISION = "f543c56225b2e04d0ad141e29655be3a45d9c455"
CATALOG_SHA = "6340069fffca228be81ebc910ba89a5f6774f9e26d78149a4a2cb0d07fbd0acd"
KERNEL_REVISION = "d9b0c745a36a5506240fa059b97d0222bf66b644"
KERNEL_VARIANT = "torch-stable-abi29-cu130-x86_64-linux"
KERNEL_METADATA_SHA = "983bfa830e94a3d265f6c583a9ad4d8edd588c1586ef0eae432af54efa59f0e9"
BF16_WEIGHT_BYTES = 66_714_780_128
NATIVE_TORCH_POLICY = {"float32_matmul_precision": "highest", "matmul_allow_tf32": False,
    "cudnn_allow_tf32": True, "cudnn_benchmark": False, "cudnn_deterministic": False,
    "deterministic_algorithms": False, "deterministic_warn_only": False}
SDK_FILES = ("quantizers/quantizer_nvfp4.py", "integrations/nvfp4.py",
             "models/qwen3_vl/modeling_qwen3_vl.py", "models/qwen3_vl/processing_qwen3_vl.py",
             "models/qwen3_vl/video_processing_qwen3_vl.py", "generation/utils.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Exclusive publication: evidence is never replaced, even after a crash."""
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def append(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def cell_key(item):
    row, task = item["row"], item["task"]
    p = row.get("source_parameters", row.get("params", {}))
    if task == "T1":
        return (task, p["sf_cycles_per_aperture"], p.get("modulation_depth", p.get("multiplier_depth")))
    if task == "T2":
        return (task, p["offset_frame"] - p["onset_frame"], p["background_Y"], p["delta_Y"])
    return (task, p["angle_deg"], p["tilt_sign"])


def validate_panel(panel, catalog):
    items = panel["items"]
    require(len(items) == 432 and len({x["sample_id"] for x in items}) == 432, "Exactly432 unique videos required")
    require(Counter(x["task"] for x in items) == {"T1": 144, "T2": 144, "T3": 144}, "Three48-video cells per task required")
    require(panel.get("catalog_file_sha256", panel.get("catalog_sha256")) == CATALOG_SHA, "Panel must bind the current catalog bytes")
    for item in items:
        sid = item["sample_id"]
        require(sid in catalog["items"], "Unknown catalog ID")
        expected = catalog["items"][sid]
        for key in ("row", "ready", "row_sha256", "ready_sha256", "source_family", "source_stream_sha256"):
            require(item[key] == expected[key], "Catalog identity changed: " + sid + "/" + key)
        require(digest(item["row"]) == item["row_sha256"] and digest(item["ready"]) == item["ready_sha256"], "Row/ready hash differs")
    # Cell definition is scientific (no phase/onset/side in the key).
    groups = Counter(cell_key(x) for x in items)
    require(len(groups) == 9 and set(groups.values()) == {48}, "Panel must contain nine COMPLETE48-video cells")
    for key in groups:
        expected_ids = {sid for sid, x in catalog["items"].items() if cell_key(x) == key}
        require(expected_ids == {x["sample_id"] for x in items if cell_key(x) == key}, "Incomplete scientific cell")
    return items


def truth(item):
    return item["row"].get("truth", item["row"].get("correct_answer"))


def assignments(items, selected_cells=None):
    """Preserve the frozen whole-cell device assignment in both precision arms."""
    lookup = {x["sample_id"]: x for x in items}
    if selected_cells is None:
        groups = defaultdict(list)
        for item in items:
            groups[cell_key(item)].append(item["sample_id"])
        selected_cells = [{"sample_ids": ids, "assigned_device": f"Fourier GPU{i % 2}"}
                          for i, (_, ids) in enumerate(sorted(groups.items()))]
    workers = [[], []]
    for cell in selected_cells:
        require(cell["assigned_device"] in ("Fourier GPU0", "Fourier GPU1"), "Unknown frozen device")
        worker = int(cell["assigned_device"][-1])
        require(len(cell["sample_ids"]) == 48, "Whole48-video cell required")
        require(len({cell_key(lookup[sid]) for sid in cell["sample_ids"]}) == 1, "Mixed scientific cell")
        require(Counter(truth(lookup[sid]) for sid in cell["sample_ids"]) == {"LEFT": 24, "RIGHT": 24}, "Cell truth imbalance")
        workers[worker].extend(cell["sample_ids"])
    require([len(w) for w in workers] == [240, 192], "Frozen5/4cell assignment must be240/192")
    require(len(set(workers[0] + workers[1])) == 432 and set(workers[0] + workers[1]) == set(lookup), "Duplicate/missing worker IDs")
    for i, worker in enumerate(workers):
        first = [next(sid for sid in worker if lookup[sid]["task"] == t) for t in ("T2", "T3")]
        workers[i] = first + [sid for sid in worker if sid not in first]
    return workers


def extract_panel(protocol):
    if "selection" not in protocol:
        return protocol
    selection = protocol["selection"]
    require(digest(selection["selected_ID_order"]) == selection["selected_ID_order_sha256"], "Frozen selection order changed")
    return {"items": [selection["selected_items"][sid] for sid in selection["selected_ID_order"]],
            "catalog_file_sha256": protocol["source_catalog"]["sha256"],
            "selected_cells": selection["selected_cells"],
            "selection_sha256": digest(selection), "seed": selection["seed"]}


def normalized_input(item):
    """Adapter of field names only; original catalog dictionaries are preserved."""
    row, ready = deepcopy(item["row"]), deepcopy(item["ready"])
    row.setdefault("truth", row.get("correct_answer"))
    row.setdefault("source_parameters", row.get("params"))
    ready.setdefault("video_relative", ready.get("video_path"))
    ready.setdefault("stream_sha256", item["source_stream_sha256"])
    ready.setdefault("source_parameters", row["source_parameters"])
    require(row["sample_id"] == ready["sample_id"], "Source ID mismatch")
    require(ready["stream_sha256"] == item["source_stream_sha256"], "Source stream mismatch")
    return row, ready


def kernel_identity(build):
    build = Path(build)
    require(build.name == KERNEL_VARIANT, "Wrong kernel variant")
    metadata = build / "metadata.json"
    require(file_sha256(metadata) == KERNEL_METADATA_SHA, "Pinned native kernel metadata differs")
    meta = read(metadata)
    require(meta["version"] == 1 and "12.0a" in meta["backend"]["archs"], "Missing native SM120 support")
    result = {}
    for name, expected in meta["digest"]["files"].items():
        require(Path(name).name == name, "Kernel metadata path escapes build")
        observed = file_sha256(build / name)
        require(base64.b64encode(bytes.fromhex(observed)).decode() == expected, "Kernel file changed: " + name)
        result[name] = observed
    return {"repo": "kernels-community/nvfp4-gemm", "revision": KERNEL_REVISION,
            "variant": KERNEL_VARIANT, "metadata_sha256": KERNEL_METADATA_SHA, "files": result}


def runtime_identity(runtime):
    """Metadata and source reads only: no Torch import, model or processor call."""
    require(Path(runtime["checkpoint"]).name == REVISION, "Wrong Cosmos base revision")
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "numpy", "accelerate", "safetensors", "kernels")}
    require(versions["torch"].split("+")[0] == "2.13.0" and versions["transformers"] == "5.17.0", "Qualified framework changed")
    require(versions["kernels"].startswith("0.16."), "Transformers5.17 requires kernels>=0.16,<0.17")
    base = Path(importlib.metadata.distribution("transformers").locate_file("transformers"))
    return {"packages": versions, "transformers_sources": {f: file_sha256(base / f) for f in SDK_FILES},
            "runner_sha256": file_sha256(__file__), "inference_sha256": file_sha256(inference.__file__),
            "storage_sha256": file_sha256(Path(__file__).with_name("storage.py")),
            "checkpoint": inference.checkpoint_asset_identity(runtime["checkpoint"]),
            "official_cosmos": inference._cosmos_package_identity(runtime["cosmos_package"]),
            "kernel": kernel_identity(runtime["kernel_build"]),
            "numerical_environment": {k: os.environ.get(k) for k in
                ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "CUBLAS_WORKSPACE_CONFIG")}}


def seal(args):
    require(file_sha256(args.catalog) == CATALOG_SHA, "Catalog file changed")
    catalog, protocol, runtime = read(args.catalog), read(args.panel), read(args.runtime)
    panel = extract_panel(protocol)
    if "source_roots" in protocol:
        require(runtime["asset_roots"] == protocol["source_roots"]["fourier"], "Frozen source roots changed")
    items = validate_panel(panel, catalog)
    identity = runtime_identity(runtime)
    plan = {"schema": SCHEMA, "panel": panel, "panel_file_sha256": file_sha256(args.panel),
            "catalog_file_sha256": CATALOG_SHA, "runtime": runtime, "runtime_identity": identity,
            "assignments": assignments(items, panel.get("selected_cells")), "modes": ["bf16", "nvfp4"],
            "generation": inference.GENERATION, "max_scientific_calls": 864,
            "calls_per_worker_per_mode": {"0": 240, "1": 192}, "retries": 0,
            "qualification": "first2 planned IDs per worker/mode; no additional calls",
            "precision_delta": "HF NVFP4 native defaults: eligible bias-free16-aligned linears; W4A4prefill/W4A16decode; all other interfaces unchanged"}
    write_new(args.output, plan)
    return {"plan": str(args.output), "sha256": file_sha256(args.output), "model_calls": 0}


@contextmanager
def physical_lease(runtime, torch):
    import fcntl
    require(torch.cuda.device_count() == 1, "TP1 worker requires exactly one visible physical GPU")
    require(tuple(torch.cuda.get_device_capability(0)) == (12, 0), "Qualified Fourier SM120 required")
    root = Path(runtime["lease_root"]).resolve()
    require(root == Path(runtime["asset_roots"]["t1"]).resolve() / "evaluation/outputs", "Must use existing native T1 lock root")
    device = str(getattr(torch.cuda.get_device_properties(0), "uuid", None) or os.environ["CUDA_VISIBLE_DEVICES"])
    path = root / (".gpu-" + digest({"host": socket.gethostname(), "device": device}) + ".lock")
    require(path.is_file() and not path.is_symlink(), "Existing physical GPU lock required")
    with path.open("r+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        st = os.fstat(stream.fileno())
        require((st.st_dev, st.st_ino) == (path.stat().st_dev, path.stat().st_ino), "Physical lock inode changed")
        state = inference.resource_snapshot(torch, runtime["resources"])
        require(state["admitted"], "Insufficient GPU/RAM margin")
        yield {"path": str(path), "st_dev": st.st_dev, "st_ino": st.st_ino, "device": device, "resources": state}
        require((st.st_dev, st.st_ino) == (path.stat().st_dev, path.stat().st_ino), "Physical lock changed while owned")


class Journal:
    def __init__(self, path, mode, worker, plan_sha):
        self.path, self.mode, self.worker, self.plan_sha = Path(path), mode, worker, plan_sha
        self.passed = set()
        self.events = []

    def event(self, name, **fields):
        record = {"event": name, "monotonic": time.monotonic(), "unix": time.time(),
                  "mode": self.mode, "worker": self.worker, **fields}
        append(self.path / "events.jsonl", record)
        self.events.append(record)
        if name == "TECHNICAL_PASS":
            self.passed.add(fields["sample_id"])


def architecture(model):
    modules = []
    for name, module in model.named_modules():
        kind = type(module).__module__ + "." + type(module).__qualname__
        if type(module).__name__ == "NVFP4Linear":
            kind = "torch.nn.modules.linear.Linear"
        detail = {"name": name, "class": kind}
        for field in ("in_features", "out_features", "num_embeddings", "embedding_dim"):
            if hasattr(module, field):
                detail[field] = getattr(module, field)
        modules.append(detail)
    return {"modules": modules, "sha256": digest(modules)}


def torch_policy(torch):
    return {"float32_matmul_precision": torch.get_float32_matmul_precision(),
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


class CosmosSession(inference.ModelSession):
    """Only native NVFP4 loader arguments differ from the qualified HF path."""
    def __init__(self, mode, runtime, store):
        self.precision, self.runtime = mode, runtime
        super().__init__("cosmos", runtime["checkpoint"], store,
                         cosmos_package=runtime["cosmos_package"], resources=runtime["resources"])

    def _load_hf(self):
        from transformers import AutoConfig, NVFP4Config, Qwen3VLForConditionalGeneration
        from transformers_cosmos3 import Cosmos3ForConditionalGeneration as Model
        from transformers_cosmos3.model import KEY_MAPPING
        torch = self.torch
        policy = torch_policy(torch)
        require(policy == NATIVE_TORCH_POLICY and torch.get_num_threads() == 6, "Original native Torch policy/threads changed")
        self.store.event("TORCH_POLICY", policy=policy, num_threads=torch.get_num_threads(), seed=0)
        keys = read(self.model_path / "model.safetensors.index.json")["weight_map"]
        exact_mapping = {}
        for key in keys:
            renamed = key
            for pattern, replacement in KEY_MAPPING.items():
                renamed = re.sub(pattern, replacement, renamed)
            if renamed != key:
                exact_mapping["^" + re.escape(key) + "$"] = renamed
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        require(config.text_config.hidden_size == 5120 and config.text_config.num_hidden_layers == 64, "Cosmos architecture changed")
        with torch.device("meta"):
            skeleton = Model(config)
        expected = architecture(skeleton)
        del skeleton
        kwargs = {}
        if self.precision == "nvfp4":
            from kernels import get_local_kernel
            import transformers.integrations.nvfp4 as integration
            kernel = get_local_kernel(Path(self.runtime["kernel_build"]))
            # Resolve the exact official native kernel locally; no Hub/auth/network.
            original = integration.lazy_load_kernel
            self._patch(integration, "lazy_load_kernel", lambda name: kernel if name == "nvfp4" else original(name))
            integration.load_nvfp4_kernel.cache_clear()
            kwargs["quantization_config"] = NVFP4Config()
        self.model, info = Qwen3VLForConditionalGeneration.from_pretrained.__func__(Model,
            self.model_path, config=config, key_mapping=exact_mapping, local_files_only=True,
            dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation="sdpa",
            output_loading_info=True, **kwargs)
        require(not any(info.get(k) for k in ("missing_keys", "mismatched_keys", "error_msgs")), "Cosmos weight loading failed")
        require(not any(t.is_meta for t in list(self.model.parameters()) + list(self.model.buffers())), "Uninitialized model tensor")
        actual = architecture(self.model)
        require(actual == expected, "NVFP4 changed logical model architecture beyond linear precision")
        self.model.eval()
        converted = [{"name": n, "in_features": m.in_features, "out_features": m.out_features}
                     for n, m in self.model.named_modules() if type(m).__name__ == "NVFP4Linear"]
        require(bool(converted) == (self.precision == "nvfp4"), "Actual precision differs from requested mode")
        for name, module in self.model.named_modules():
            if type(module).__name__ != "NVFP4Linear":
                continue
            require(module.weight.dtype == torch.uint8 and list(module.weight.shape) == [module.out_features, module.in_features // 2], "NVFP4 packed weight dtype/shape differs: " + name)
            require(module.weight_sf.dtype == torch.int32 and module.weight_sf_rowmajor.dtype == torch.uint8
                    and module.weight_global_scale.dtype == torch.float32, "NVFP4 scale dtypes differ: " + name)
        retained = [{"name": name, "in_features": module.in_features, "out_features": module.out_features,
                     "bias": module.bias is not None, "weight_dtype": str(module.weight.dtype)}
                    for name, module in self.model.named_modules() if isinstance(module, torch.nn.Linear)]
        exclusions = getattr(getattr(self.model, "hf_quantizer", None), "modules_to_not_convert", None)
        counts = defaultdict(int)
        for kind, tensors in (("parameters", self.model.parameters()), ("buffers", self.model.buffers())):
            for tensor in tensors:
                counts[kind + ":" + str(tensor.dtype)] += tensor.numel() * tensor.element_size()
        if self.precision == "bf16":
            require(counts["parameters:torch.bfloat16"] == BF16_WEIGHT_BYTES
                    and sum(v for k, v in counts.items() if k.startswith("parameters:")) == BF16_WEIGHT_BYTES,
                    "BF16 logical checkpoint weight inventory differs")
        self.store.event("WEIGHT_MAPPING_PASS", mapping_count=len(exact_mapping),
            exact_mapping_sha256=digest(exact_mapping), official_shim=inference._source_identity(Model),
            loading_info={k: sorted(v) if isinstance(v, set) else v for k, v in info.items()})
        self.store.event("MODEL_LOADED", backend="transformers", mode=self.precision,
            attn_implementation="sdpa", architecture=actual, converted_linears=converted,
            converted_linear_count=len(converted), retained_linears=retained, native_exclusion_patterns=exclusions,
            quantized_weight_dtype="torch.uint8 packedE2M1" if converted else None,
            scale_dtypes={"swizzled": "torch.int32 packedFP8", "rowmajor": "torch.uint8 packedFP8", "global": "torch.float32"} if converted else None,
            storage_bytes_by_kind_dtype=dict(counts), quantization_config=kwargs["quantization_config"].to_dict() if kwargs else None,
            model_class=inference._source_identity(type(self.model)))
        self.allowed_sequences = [self.tokenizer.encode(s, add_special_tokens=False) for s in ("LEFT", "RIGHT")]
        require(all(self.allowed_sequences) and len({tuple(s) for s in self.allowed_sequences}) == 2, "Choice tokenizer differs")
        eos = self.model.generation_config.eos_token_id
        self.eos = [eos] if isinstance(eos, int) else list(eos or [])
        require(bool(self.eos), "Native EOS required")
        self.effective_generation = {**inference.GENERATION, "backend": "transformers", "constraint": "prefix_allowed_tokens_fn",
            "choice_token_sequences": self.allowed_sequences, "native_eos_token_ids": self.eos,
            "num_return_sequences": 1, "max_time": None,
            "inherited_native_generation_config": json.loads(self.model.generation_config.to_json_string()),
            "cache_implementation": "native_default"}

    def prepare(self, row, ready, asset_root):
        started = time.monotonic()
        frames, data, text, receipt = super().prepare(row, ready, asset_root)
        input_ids = data["input_ids"].flatten().tolist()
        receipt["input_token_ids_sha256"] = digest(input_ids)
        if self.context["deep"]:
            receipt["input_token_ids"] = input_ids
            pixels = data["pixel_values_videos"].contiguous()
            receipt["FP32_processor_tensor_sha256"] = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
            receipt["BF16_pretransfer_tensor_sha256"] = hashlib.sha256(pixels.to(self.torch.bfloat16).view(self.torch.uint8).numpy().tobytes()).hexdigest()
        receipt["prepare_seconds"] = time.monotonic() - started
        return frames, data, text, receipt


def run(args):
    plan = read(args.plan)
    require(plan["schema"] == SCHEMA and plan["max_scientific_calls"] == 864 and plan["retries"] == 0, "Invalid finite plan")
    runtime, mode, worker = plan["runtime"], args.mode, args.worker
    require(mode in plan["modes"] and worker in (0, 1), "Invalid worker")
    require(runtime_identity(runtime) == plan["runtime_identity"], "Sealed source/runtime changed")
    admission = read(args.admission)
    require(admission["plan_sha256"] == file_sha256(args.plan) and admission["production_held"] is True
            and admission["owned_engines_exited"] is True and admission["physical_locks_released"] is True,
            "Controller admission missing")
    require(admission["workers"][str(worker)]["cuda_visible_devices"] == os.environ.get("CUDA_VISIBLE_DEVICES"), "Wrong physical worker assignment")
    ids = plan["assignments"][worker]
    require(len(ids) == len(set(ids)) == {0: 240, 1: 192}[worker], "Wrong finite worker budget")
    lookup = {x["sample_id"]: x for x in plan["panel"]["items"]}
    require(set(ids) <= set(lookup), "Unknown worker ID")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output, mode, worker, file_sha256(args.plan))
    write_new(output / "binding.json", {"plan_sha256": file_sha256(args.plan), "mode": mode, "worker": worker,
              "IDs": ids, "admission_sha256": file_sha256(args.admission), "pid": os.getpid(), "host": socket.gethostname()})
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1")
    import torch
    require(torch.version.cuda == "13.0", "Qualified CUDA13.0 torch build required")
    completed, begun = [], time.monotonic()
    lease_acquired = False
    try:
        with physical_lease(runtime, torch) as lease:
            lease_acquired = True
            journal.event("LEASE_ACQUIRED", lease=lease)
            session = None
            try:
                session = CosmosSession(mode, runtime, journal)
                for index, sid in enumerate(ids):
                    item = lookup[sid]
                    row, ready = normalized_input(item)
                    session.context["deep"] = index < 2
                    attempt_id = f"{mode}-w{worker}-{index:04d}"
                    journal.event("ATTEMPT_STARTED", sample_id=sid, attempt_id=attempt_id, ordinal=index, deep=index < 2)
                    call_start = time.monotonic()
                    committed_record = {}
                    def commit(record):
                        record = {**record, "sample_id": sid, "attempt_id": attempt_id, "mode": mode, "worker": worker,
                                  "ordinal": index, "deep": index < 2, "selected_cell_id": item.get("selected_cell_id", canonical_json(cell_key(item))),
                                  "catalog_row_sha256": item["row_sha256"],
                                  "catalog_ready_sha256": item["ready_sha256"], "source_family": item["source_family"],
                                  "full_request_seconds": time.monotonic() - call_start}
                        append(output / "raw.jsonl", record)
                        committed_record.update(record)
                        completed.append(sid)
                        return record
                    session.generate(row, ready, Path(runtime["asset_roots"][item["source_family"]]), commit)
                    require(sid in journal.passed, "Completed raw output lacks technical pass")
                    append(output / "technical.jsonl", {"sample_id": sid, "attempt_id": attempt_id, "status": "PASS",
                        "raw_sha256": digest(committed_record), "complete_request_seconds": time.monotonic() - call_start})
                    if index == 1:
                        journal.event("FIRST_TWO_QUALIFIED", IDs=ids[:2])
                require(completed == ids and journal.passed == set(ids), "Incomplete finite worker")
                journal.event("FINITE_CALLS_COMPLETE", completed=len(completed), worker_seconds=time.monotonic() - begun)
            except BaseException as exc:
                journal.event("FAILED_NO_RETRY", error_type=type(exc).__name__, error=str(exc), completed=len(completed))
                raise
            finally:
                if session is not None:
                    session.close()
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                journal.event("MODEL_CLOSED_BEFORE_LEASE_RELEASE")
    finally:
        if lease_acquired:
            journal.event("LEASE_RELEASED")
    write_new(output / "complete.json", {"status": "COMPLETE", "model_calls": len(completed), "technical_passes": len(journal.passed),
        "mode": mode, "worker": worker, "IDs": completed, "plan_sha256": file_sha256(args.plan)})
    return {"status": "COMPLETE", "model_calls": len(completed), "output": str(output)}


def compare_records(bf16, nvfp4):
    left = {x["sample_id"]: x for x in bf16}
    right = {x["sample_id"]: x for x in nvfp4}
    require(len(left) == len(bf16) and len(right) == len(nvfp4) and set(left) == set(right), "Paired IDs missing/duplicated")
    rows = []
    for sid in left:
        a, b = left[sid], right[sid]
        for key in ("catalog_row_sha256", "catalog_ready_sha256", "source_family", "worker", "effective_generation"):
            require(a[key] == b[key], "Nonprecision experimental delta: " + key)
        for key in ("source_video_sha256", "source_stream_sha256", "chat_template_sha256", "input_token_ids_sha256",
                    "actual_processor_tensor_shape", "actual_processor_dtype"):
            require(a["input_receipt"][key] == b["input_receipt"][key], "Input differs: " + key)
        for key in ("FP32_processor_tensor_sha256", "BF16_pretransfer_tensor_sha256"):
            if key in a["input_receipt"] or key in b["input_receipt"]:
                require(a["input_receipt"].get(key) == b["input_receipt"].get(key), "Deep input differs")
        rows.append({"sample_id": sid, "task": a["task"], "truth": a["truth"],
            "tokens_equal": a["output_token_ids"] == b["output_token_ids"], "answer_equal": a["raw_answer"] == b["raw_answer"],
            "finish_equal": a["finish_reason"] == b["finish_reason"],
            "bf16_tokens": a["output_token_ids"], "nvfp4_tokens": b["output_token_ids"],
            "worker": a["worker"], "cell_id": a.get("selected_cell_id", a["task"]),
            "bf16_finish": a["finish_reason"], "nvfp4_finish": b["finish_reason"],
            "bf16_valid": a["finish_reason"] == "stop" and a["raw_answer"].strip() in ("LEFT", "RIGHT"),
            "nvfp4_valid": b["finish_reason"] == "stop" and b["raw_answer"].strip() in ("LEFT", "RIGHT"),
            "bf16_answer": a["raw_answer"], "nvfp4_answer": b["raw_answer"],
            "bf16_seconds": a["seconds"], "nvfp4_seconds": b["seconds"],
            "bf16_full_request_seconds": a["full_request_seconds"], "nvfp4_full_request_seconds": b["full_request_seconds"],
            "bf16_complete_request_seconds": a.get("complete_request_seconds", a["full_request_seconds"]),
            "nvfp4_complete_request_seconds": b.get("complete_request_seconds", b["full_request_seconds"]),
            "steady": not a["deep"] and not b["deep"]})
    steady = [x for x in rows if x["steady"]]
    def summary(part):
        return {"N": len(part), "tokens_equal": sum(x["tokens_equal"] for x in part),
            "answers_equal": sum(x["answer_equal"] for x in part), "finish_equal": sum(x["finish_equal"] for x in part),
            "bf16_correct": sum(x["bf16_valid"] and x["bf16_answer"].strip() == x["truth"] for x in part),
            "nvfp4_correct": sum(x["nvfp4_valid"] and x["nvfp4_answer"].strip() == x["truth"] for x in part),
            "bf16_invalid_or_length": sum(not x["bf16_valid"] for x in part),
            "nvfp4_invalid_or_length": sum(not x["nvfp4_valid"] for x in part),
            "answer_transitions": dict(Counter(x["bf16_answer"] + " -> " + x["nvfp4_answer"] for x in part)),
            "bf16_seconds_mean": statistics.mean(x["bf16_seconds"] for x in part),
            "nvfp4_seconds_mean": statistics.mean(x["nvfp4_seconds"] for x in part),
            "bf16_complete_request_seconds_median": statistics.median(x["bf16_complete_request_seconds"] for x in part),
            "nvfp4_complete_request_seconds_median": statistics.median(x["nvfp4_complete_request_seconds"] for x in part),
            "bf16_complete_request_seconds_mean": statistics.mean(x["bf16_complete_request_seconds"] for x in part),
            "nvfp4_complete_request_seconds_mean": statistics.mean(x["nvfp4_complete_request_seconds"] for x in part),
            "complete_request_speedup": sum(x["bf16_complete_request_seconds"] for x in part) / sum(x["nvfp4_complete_request_seconds"] for x in part),
            "paired_latency_speedup": sum(x["bf16_seconds"] for x in part) / sum(x["nvfp4_seconds"] for x in part)}
    return {"status": "PAIRED_COMPARISON_COMPLETE", "all": summary(rows),
            "steady": summary(steady) if steady else None,
            "tasks": {t: summary([x for x in rows if x["task"] == t]) for t in sorted({x["task"] for x in rows})},
            "cells": {c: summary([x for x in rows if x["cell_id"] == c]) for c in sorted({x["cell_id"] for x in rows})},
            "workers": {str(w): summary([x for x in rows if x["worker"] == w]) for w in sorted({x["worker"] for x in rows})},
            "rows": rows, "interpretation": "Token/answer changes are quantization outcomes; latency is paired same-worker HF/SDPA measurement, not a BF16-equivalence gate."}


def compare(args):
    all_rows, loads = {"bf16": [], "nvfp4": []}, {}
    plan_shas, architectures, seen = set(), set(), set()
    for folder in args.runs:
        folder = Path(folder)
        binding, complete = read(folder / "binding.json"), read(folder / "complete.json")
        key = (binding["mode"], binding["worker"])
        require(key not in seen and key[0] in all_rows, "Duplicate/unknown precision worker")
        seen.add(key)
        require(complete["status"] == "COMPLETE" and complete["model_calls"] == complete["technical_passes"] == {0: 240, 1: 192}[key[1]], "Incomplete run")
        raw = [json.loads(x) for x in (folder / "raw.jsonl").read_text().splitlines()]
        require([x["sample_id"] for x in raw] == binding["IDs"] == complete["IDs"], "Worker output ID order differs")
        events = [json.loads(x) for x in (folder / "events.jsonl").read_text().splitlines()]
        require({x["sample_id"] for x in events if x["event"] == "TECHNICAL_PASS"} == set(binding["IDs"]), "Technical passes missing")
        require(events[-1]["event"] == "LEASE_RELEASED", "Lease closure missing")
        technical = [json.loads(x) for x in (folder / "technical.jsonl").read_text().splitlines()]
        require([x["sample_id"] for x in technical] == binding["IDs"], "Technical journal coverage differs")
        for record, receipt in zip(raw, technical):
            require(receipt["status"] == "PASS" and receipt["raw_sha256"] == digest(record), "Technical/raw binding differs")
            record["complete_request_seconds"] = receipt["complete_request_seconds"]
        all_rows[binding["mode"]].extend(raw)
        plan_shas.add(binding["plan_sha256"])
        architectures.update(x["architecture"]["sha256"] for x in events if x["event"] == "MODEL_LOADED")
        loads[str(key)] = next(x["load_seconds"] for x in events if x["event"] == "MODEL_READY")
    require(seen == {(m, w) for m in all_rows for w in (0, 1)} and len(plan_shas) == len(architectures) == 1, "Exactly four same-plan/logical-architecture runs required")
    report = compare_records(all_rows["bf16"], all_rows["nvfp4"])
    report.update(load_seconds=loads, plan_sha256=next(iter(plan_shas)), model_calls=864)
    write_new(args.output, report)
    return {k: v for k, v in report.items() if k != "rows"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seal")
    for field in ("catalog", "panel", "runtime", "output"):
        p.add_argument("--" + field, required=True)
    p = sub.add_parser("run")
    for field in ("plan", "admission", "output"):
        p.add_argument("--" + field, required=True)
    p.add_argument("--mode", choices=("bf16", "nvfp4"), required=True)
    p.add_argument("--worker", type=int, choices=(0, 1), required=True)
    p = sub.add_parser("compare")
    p.add_argument("--runs", nargs=4, required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    print(canonical_json({"seal": seal, "run": run, "compare": compare}[args.command](args)))


if __name__ == "__main__":
    main()
