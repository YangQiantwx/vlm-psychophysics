"""Portable Cosmos X file-video inference; show/check never load model weights.

The adjacent recipe binds the measured X route. This entry uses public LLM.chat,
without the production scheduler, observers, QSA, or model/processor patches.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recipe():
    value = json.loads(Path(__file__).with_suffix(".json").read_text())
    require(value["schema"] == "cosmos-x-public-v1", "Unknown Cosmos X recipe")
    return value


def is_primary(entry):
    task, parameters = entry["task"], entry["nominal_parameters"]
    if task == "T1":
        return True
    if task == "T2":
        return parameters["offset_frame"] - parameters["onset_frame"] == 30
    return task == "T3" and parameters["angle_deg"] not in (1.75, 2.25)


def checked_index(data_root, spec):
    root = Path(data_root).expanduser().resolve(strict=True)
    index = json.loads((root / "dataset-index.json").read_text())
    require(index["schema"] == "classic-precision-rgb16-index-v1", "Unknown dataset index")
    require(index["index_id"] == digest({k: v for k, v in index.items() if k != "index_id"}),
            "Dataset index identity differs")
    require(digest(index["items"]) == spec["dataset"]["items_sha256"],
            "Need the complete 12,528-source release; the old v0.3 index is insufficient")
    require(index["counts"] == spec["dataset"]["source_counts"], "Source denominators differ")
    require(index["input"]["bits_per_channel"] == 16 and index["input"]["codec"] == "ffv1",
            "Expected retained lossless 16-bit FFV1 sources")
    selected = {sid: item for sid, item in index["items"].items() if is_primary(item)}
    require(dict(Counter(x["task"] for x in selected.values())) == spec["scope"]["counts"],
            "Current X scope differs")
    require(digest(sorted(selected)) == spec["scope"]["sorted_ids_sha256"],
            "Current X ID set differs")
    for sid, item in selected.items():
        require(item["sample_id"] == sid, "Sample identity differs")
        require(item["prompt"] == spec["prompts"][item["task"]], "Task prompt differs")
        require([item[k] for k in ("frames", "height", "width", "fps")] == [180, 512, 768, 60],
                "Source geometry or timing differs")
    return root, index, selected


def checked_assets(model_dir, tokenizer_dir, spec):
    roots = {"checkpoint": Path(model_dir).expanduser().resolve(strict=True),
             "tokenizer": Path(tokenizer_dir).expanduser().resolve(strict=True)}
    for kind, root in roots.items():
        manifest = spec["assets"][kind]
        require(manifest.get("files"), "Recipe lacks qualified " + kind + " identities")
        for name, identity in manifest["files"].items():
            require(Path(name).name == name, "Asset path escapes its directory")
            path = root / name
            require(path.is_file() and path.stat().st_size == identity["bytes"],
                    "Asset missing or size differs: " + kind + "/" + name)
            require(file_sha(path) == identity["sha256"],
                    "Asset SHA256 differs: " + kind + "/" + name)
        for name in manifest.get("absent_files", []):
            require(not (root / name).exists(), "Unexpected asset changes template/processor selection: " + name)
    return roots


def checked_catalog(data_root, catalog_path, expected_sha):
    """A separately identified T4/T5 release, never substituted for classic T1–T3."""
    root = Path(data_root).expanduser().resolve(strict=True)
    catalog = Path(catalog_path).expanduser().resolve(strict=True)
    require(expected_sha and file_sha(catalog) == expected_sha, "Published catalog SHA256 is required and must match")
    rows = [json.loads(line) for line in catalog.read_text().splitlines() if line.strip()]
    require(rows and len({r['sample_id'] for r in rows}) == len(rows), "Empty or duplicate catalog IDs")
    selected = {}
    for row in rows:
        task = row['task']; choices = ['LEFT', 'RIGHT'] if task == 'T4' else ['YES', 'NO']
        require(task in ('T4', 'T5') and row['domain'] in ('classic', 'natural'), "Unsupported catalog condition")
        require(row['choices'] == choices and row['prompt'], "Missing or invalid task response contract")
        require([row[k] for k in ('frames', 'height', 'width', 'fps')] == [180, 512, 768, 60], "Catalog source geometry differs")
        require(row['accuracy_eligible'] == (row['truth'] is not None), "Eligibility differs from label presence")
        require(row['truth'] is None or row['truth'] in choices, "Invalid label")
        require(task == 'T4' or (row.get('reference_path') and row.get('reference_sha256')), "T5 reference identity is required")
        selected[row['sample_id']] = dict(row, video_relative=row['video_path'])
    return root, {'items': selected, 'catalog_sha256': expected_sha}, selected


def selected_entries(selected, ids):
    # Keep this file standalone in the source release. Resolve external naming
    # aliases to frozen source keys before selection; seeds/order/raw keys and
    # the pinned items/scope digests remain those of the qualified recipe.
    aliases = {}
    for sid, item in selected.items():
        values = {sid, item.get('legacy_sample_id', sid), item.get('video_relative'), item.get('legacy_path')}
        if item['task'] == 'T1' and sid.startswith('t1rgb16_'):
            values.add('t1_' + digest({'schema': 'sample-identity-v1', 'domain': 'classic',
                'task': 'T1', 'video_sha256': item['video_sha256']})[:20])
        for value in values - {None}:
            require(value not in aliases or aliases[value] == sid, 'Ambiguous source ID/path alias')
            aliases[value] = sid
    require(ids is None or all(value in aliases for value in ids),
            "Requested ID is outside the selected source release")
    chosen = [aliases[value] for value in ids] if ids else sorted(selected)
    require(len(chosen) == len(set(chosen)), "Duplicate requested source ID or alias")
    return [(sid, selected[sid]) for sid in chosen]


def checked_media(root, entries):
    paths = {}
    for sid, item in entries:
        relative = Path(item["video_relative"])
        require(not relative.is_absolute() and ".." not in relative.parts, "Invalid relative video path")
        path = (root / relative).resolve(strict=True)
        # HF snapshot files may resolve to the shared blobs directory.
        # Relative names and exact file hashes bind these cache symlinks too.
        require(file_sha(path) == item["video_sha256"], "Source file SHA256 differs: " + sid)
        paths[sid] = path
        if item.get('reference_path'):
            relative = Path(item['reference_path'])
            require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid relative reference path')
            reference = (root / relative).resolve(strict=True)
            require(file_sha(reference) == item['reference_sha256'], 'Reference SHA256 differs: ' + sid)
    return paths


def request_messages(root, path, item):
    content = []
    if item.get('reference_path'):
        reference = (root / item['reference_path']).resolve(strict=True)
        content.append({'type': 'image_url', 'image_url': {'url': reference.as_uri()}})
    content.extend([{'type': 'video_url', 'video_url': {'url': path.as_uri()}},
                    {'type': 'text', 'text': item['prompt']}])
    return [{'role': 'user', 'content': content}]


def runtime_versions(spec):
    result = {name: importlib.metadata.version(name) for name in spec["software_pins"]}
    require(result == spec["software_pins"], "Pinned runtime packages differ: " + str(result))
    for relative, expected in spec["runtime_source_sha256"].items():
        package = relative.split("/", 1)[0]
        path = importlib.metadata.distribution(package).locate_file(relative)
        require(file_sha(path) == expected, "Pinned runtime source differs: " + relative)
    return result


def environment(spec, device):
    for key in spec["cleared_environment"]:
        os.environ.pop(key, None)
    os.environ.update(spec["environment"])
    os.environ["CUDA_VISIBLE_DEVICES"] = device
    cuda_home = os.environ.get("CUDA_HOME")
    require(cuda_home and (Path(cuda_home) / "bin/nvcc").is_file(),
            "Set CUDA_HOME to the preinstalled CUDA 13.0 toolchain")
    os.environ["FLASHINFER_NVCC"] = str(Path(cuda_home) / "bin/nvcc")
    compiler = subprocess.run([os.environ["FLASHINFER_NVCC"], "--version"],
                              capture_output=True, text=True, check=True, timeout=10)
    require("release 13.0, V13.0.88" in compiler.stdout, "Expected the qualified CUDA 13.0.88 compiler")


def engine_kwargs(spec, roots, root, memory_fraction, reference_admission=False):
    value = dict(spec["LLM_kwargs"])
    value.update(model=str(roots["checkpoint"]), tokenizer=str(roots["tokenizer"]),
                 allowed_local_media_path=str(root), gpu_memory_utilization=memory_fraction)
    if reference_admission:
        value['limit_mm_per_prompt'] = {'audio': 0, 'image': 1, 'video': 1}
    return value


def append(stream, value):
    stream.write(canonical(value).decode() + "\n")
    stream.flush()
    os.fsync(stream.fileno())


def run(args, spec, root, roots, entries, paths, versions):
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    media_paths = [str(path.parent) for path in paths.values()]
    media_paths += [str((root / item['reference_path']).resolve(strict=True).parent)
                    for _, item in entries if item.get('reference_path')]
    media_root = Path(os.path.commonpath(media_paths))
    kwargs = engine_kwargs(spec, roots, media_root, args.gpu_memory_utilization,
                           reference_admission=bool(getattr(args, 'catalog', None)))
    with (output / "binding.json").open("x") as stream:
        append(stream, {"recipe_sha256": file_sha(Path(__file__).with_suffix(".json")),
                        "software": versions, "LLM_kwargs": kwargs,
                        "sampling": spec["sampling"], "chat_template_kwargs": spec["chat_template_kwargs"],
                        "request_mm_processor_kwargs": spec["request_mm_processor_kwargs"],
                        "assigned_ids": [sid for sid, _ in entries], "automatic_retry": False,
                        "catalog_sha256": getattr(args, 'catalog_sha256', None),
                        "production_GPU_observer_included": False})
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    import torch
    require(torch.__version__ == "2.14.0+cu130" and torch.version.cuda == "13.0",
            "Expected the qualified Torch CUDA 13.0 build")
    torch.set_num_threads(6)
    samplings = {}
    for choices in {tuple(item.get('choices', ['LEFT', 'RIGHT'])) for _, item in entries}:
        sampling = SamplingParams(**spec['sampling'], structured_outputs=StructuredOutputsParams(choice=list(choices)))
        require(not sampling.ignore_eos and sampling.stop == [] and sampling.stop_token_ids == [], 'Native EOS policy changed')
        samplings[choices] = sampling
    completed, pending = [], None
    started = time.monotonic()
    try:
        engine = LLM(**kwargs)
        with (output / "intents.jsonl").open("x") as intents, (output / "raw.jsonl").open("x") as raw:
            for sid, item in entries:
                pending = sid
                append(intents, {"sample_id": sid, "task": item["task"],
                                 "source_sha256": item["video_sha256"]})
                choices = tuple(item.get('choices', ['LEFT', 'RIGHT']))
                sampling = samplings[choices]
                messages = request_messages(root, paths[sid], item)
                request_start = time.monotonic()
                result = engine.chat(messages, sampling_params=sampling, use_tqdm=False,
                    mm_processor_kwargs=spec["request_mm_processor_kwargs"],
                    chat_template_kwargs=spec["chat_template_kwargs"])[0]
                answer = result.outputs[0]
                append(raw, {"sample_id": sid, "task": item["task"], "truth": item["truth"],
                             "raw_answer": answer.text, "output_token_ids": list(answer.token_ids),
                             "prompt_token_ids": list(result.prompt_token_ids),
                             "finish_reason": answer.finish_reason, "stop_reason": answer.stop_reason,
                             "source_sha256": item["video_sha256"],
                             "reference_sha256": item.get('reference_sha256'),
                             "accuracy_eligible": item.get('accuracy_eligible', True),
                             "request_seconds": time.monotonic() - request_start})
                require(answer.finish_reason == "stop", "Incomplete output retained for " + sid)
                require(answer.text.strip() in choices, "Invalid answer retained for " + sid)
                require(len(result.prompt_token_ids) + spec["sampling"]["max_tokens"]
                        <= spec["LLM_kwargs"]["max_model_len"], "Context budget exceeded for " + sid)
                completed.append(sid)
                pending = None
    finally:
        with (output / "result.json").open("x") as stream:
            append(stream, {"status": "REQUESTS_COMPLETE" if len(completed) == len(entries) else "INTERRUPTED",
                            "completed_ids": completed, "uncertain_id": pending,
                            "assigned": len(entries), "elapsed_seconds": time.monotonic() - started,
                            "automatic_retry": False, "GPU_input_qualification_claimed": False})
    return {"status": "REQUESTS_COMPLETE", "completed": len(completed), "output": str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["show", "install-command", "check-index", "check", "run"])
    parser.add_argument("--data-root")
    parser.add_argument('--catalog', help='Optional published T4/T5 metadata.jsonl; separate from classic T1–T3 index')
    parser.add_argument('--catalog-sha256', help='SHA256 published with the selected catalog release')
    parser.add_argument("--model-dir")
    parser.add_argument("--tokenizer-dir")
    parser.add_argument("--id", action="append", dest="ids", help="Optional finite subset of primary IDs")
    parser.add_argument("--output", help="New output directory; existing directories are refused")
    parser.add_argument("--device", default="0", help="One CUDA device index or UUID")
    parser.add_argument("--gpu-memory-utilization", type=float, choices=[0.9, 0.78], default=0.9)
    args = parser.parse_args()
    spec = recipe()
    if args.command == "show":
        print(json.dumps(spec, indent=2))
        return
    if args.command == "install-command":
        wheel = spec["installation"]["vllm_official_wheel"]
        torch_wheel = spec["installation"]["torch_public_cp312_wheel"]
        packages = [name + "==" + version for name, version in spec["software_pins"].items()
                    if name not in ("vllm", "torch")]
        print(shlex.join([sys.executable, "-m", "pip", "install",
                         wheel["url"] + "#sha256=" + wheel["sha256"],
                         torch_wheel["url"] + "#sha256=" + torch_wheel["sha256"], *packages]))
        return
    require(args.data_root, "--data-root is required")
    root, index, selected = (checked_catalog(args.data_root, args.catalog, args.catalog_sha256)
                             if args.catalog else checked_index(args.data_root, spec))
    entries = selected_entries(selected, args.ids)
    if args.command == "check-index":
        print(json.dumps({"status": "PASS", "source_videos": len(index["items"]),
                          "primary_counts": dict(Counter(x['task'] for x in selected.values())), "selected": len(entries),
                          "model_calls": 0, "runtime_imported": False}))
        return
    require(args.model_dir and args.tokenizer_dir, "--model-dir and --tokenizer-dir are required")
    roots = checked_assets(args.model_dir, args.tokenizer_dir, spec)
    paths = checked_media(root, entries)
    versions = runtime_versions(spec)
    if args.command == "check":
        print(json.dumps({"status": "PASS", "selected": len(entries), "software": versions,
                          "model_calls": 0, "weights_loaded": False, "runtime_imported": False}))
        return
    require(args.output, "--output is required for run")
    environment(spec, args.device)
    print(json.dumps(run(args, spec, root, roots, entries, paths, versions)))


if __name__ == "__main__":
    main()
