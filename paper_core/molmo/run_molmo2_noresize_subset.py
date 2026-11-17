"""Measure the frozen X 60 fps paired panel with Molmo2-8B pad-only inputs."""

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import time

import molmo2_noresize as bridge


MODEL = "allenai/Molmo2-8B"
MODEL_REVISION = "e28fa28597e5ec5e0cca2201dd8ab33d48bc4a1b"
NVFP4_MODEL = "tollea1234/Molmo2-8B-NVFP4"
NVFP4_REVISION = "3d880baa6cf74b3390a9fa74bbfb4bddfb467563"
DATASET_REVISION = "f562d61b4703b744c2f00f66a7ebabeacc4e0a0a"
GENERATION = dict(temperature=0, top_p=1, seed=0, n=1, max_tokens=16_384,
                  ignore_eos=False, stop=[], stop_token_ids=[])


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open() as stream:
        return [json.loads(line) for line in stream]


def append_records(path, rows):
    with path.open("a", buffering=1) as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def cases_for(root):
    selected = read_csv(root / "subset/selection.csv")
    aliases = read_csv(root / "subset/static-aliases.csv")
    metadata, static = {}, {}
    for folder in sorted(root.glob("*/*/videos")):
        for row in read_jsonl(folder.parent / "metadata.jsonl"):
            metadata[row["sample_id"]] = row
        for row in read_jsonl(folder.parent / "static/metadata.jsonl"):
            static[row["static_case_id"]] = row

    case_list = []
    required_static = {alias["static_case_id"] for alias in aliases}
    for entry in selected:
        row = metadata[entry["sample_id"]]
        if (entry["truth"] != row["truth"] or entry["video_sha256"] != row["video_sha256"]
                or sha_text(row["prompt"]) != entry["video_prompt_sha256"]
                or (row["frames"], row["fps"]) != (bridge.FRAMES, bridge.FPS)):
            raise ValueError(f"Video identity differs for {entry['sample_id']}")
        references = []
        if entry["reference_path"]:
            references.append({"path": entry["reference_path"], "sha256": entry["reference_sha256"],
                               "role": "reference"})
        case_list.append(dict(
            case_id=entry["sample_id"], modality="video", task=entry["task"], domain=entry["domain"],
            prompt=row["prompt"], truth=row["truth"], images=references,
            choices=row.get("choices", ["YES", "NO"] if entry["task"] == "T5" else ["LEFT", "RIGHT"]),
            video_path=entry["video_path"], video_sha256=entry["video_sha256"],
            source_rgb_sha256=row.get("decoded_stream_sha256"),
        ))
    for case_id in sorted(required_static):
        row = static[case_id]
        case_list.append(dict(
            case_id=case_id, modality="static", task=row["task"].upper(), domain=row["domain"],
            prompt=row["prompt"], truth=row["truth"], choices=row["choices"], images=row["images"],
            stimulus_input_sha256=row["model_input_sha256"],
        ))

    videos = {case["case_id"]: case for case in case_list if case["modality"] == "video"}
    if not (len(videos) == len(selected) == len(aliases) == 1480 and len(required_static) == 1346
            and {a["sample_id"] for a in aliases} == set(videos)):
        raise ValueError(f"Unexpected frozen panel: {len(videos)} videos, {len(required_static)} static, "
                         f"{len(aliases)} aliases")
    for alias in aliases:
        row = static[alias["static_case_id"]]
        video = videos[alias["sample_id"]]
        if (row["model_input_sha256"] != alias["model_input_sha256"] or row["truth"] != video["truth"]
                or f'{video["domain"]}/{video["task"].lower()}' != alias["scope"]):
            raise ValueError(f"Static alias differs for {alias['sample_id']}")
    return case_list, aliases


def case_key(case):
    # T4 static cases reuse the paired video's sample ID.
    return case["modality"], case["case_id"]


def canary_ids(cases):
    first = {}
    for case in cases:
        key = (case["modality"], case["task"], case["domain"], len(case["images"]))
        first.setdefault(key, case_key(case))
    return set(first.values())


def read_shard(path, cases):
    keys = [tuple(key) for key in json.loads(path.read_text())]
    if not keys or len(keys) != len(set(keys)) or set(keys) - cases.keys():
        raise ValueError(f"Invalid Molmo2 shard: {path}")
    return keys


def verified_path(root, part):
    path = root / part["path"]
    if file_sha(path) != part["sha256"]:
        raise ValueError(f"Media SHA differs: {part['path']}")
    return path


def model_request(root, case, processor, deep):
    import numpy as np
    from PIL import Image
    from vlm_psychophysics.media import opencv_rgb8, verify_timeline

    media, contents, hashes = {}, [], []
    if case["images"]:
        images = []
        for item in case["images"]:
            with Image.open(verified_path(root, item)) as im:
                rgb = im.convert("RGB")
            array = np.asarray(rgb)
            sha = bridge.sha(array)
            if (array.shape != (bridge.HEIGHT, bridge.WIDTH, 3)
                    or (item.get("pixel_sha256") and item["pixel_sha256"] != sha)):
                raise ValueError(f"Image shape or RGB SHA differs: {item['path']}")
            images.append(rgb)
            hashes.append(sha)
            contents.append({"type": "image"})
        media["image"] = images
    if case["modality"] == "video":
        path = verified_path(root, {"path": case["video_path"], "sha256": case["video_sha256"]})
        frames = opencv_rgb8(path)
        frame_hash = bridge.sha(frames)
        if case["source_rgb_sha256"] and frame_hash != case["source_rgb_sha256"]:
            raise ValueError(f"Frozen decoded video SHA differs: {case['case_id']}")
        metadata = dict(fps=bridge.FPS, duration=3.0, total_num_frames=bridge.FRAMES,
                        frames_indices=list(range(bridge.FRAMES)), video_backend="opencv",
                        do_sample_frames=False, height=bridge.HEIGHT, width=bridge.WIDTH)
        if deep:
            verify_timeline(path, master=False)
        media["video"] = (frames, metadata)
        contents.append({"type": "video"})
    contents.append({"type": "text", "text": case["prompt"]})
    prompt = processor.apply_chat_template([{"role": "user", "content": contents}], tokenize=False,
                                           add_generation_prompt=True)
    if case["prompt"] not in prompt:
        raise ValueError(f"Molmo2 template changed the frozen question: {case['case_id']}")
    request = {"prompt": prompt, "multi_modal_data": media,
               "mm_processor_kwargs": {"do_sample_frames": False}}
    evidence = {"image_rgb_sha256": hashes}
    if case["modality"] == "video":
        evidence.update(video_rgb_sha256=frame_hash, media_metadata=metadata)
        if deep:
            evidence["source_frame_sha256"] = [bridge.sha(frame) for frame in frames]
    return request, evidence


def engine_kwargs(checkpoint):
    return dict(model=str(checkpoint), tokenizer=str(checkpoint), trust_remote_code=True,
                tensor_parallel_size=1, dtype="bfloat16", max_model_len=bridge.CONTEXT,
                max_num_seqs=1, enforce_eager=True, enable_prefix_caching=False,
                enable_chunked_prefill=True, disable_chunked_mm_input=True, max_num_batched_tokens=49_152,
                kv_cache_memory_bytes=11 * (1 << 30), mm_processor_cache_gb=0,
                limit_mm_per_prompt={"image": 4, "video": 1}, skip_mm_profiling=True,
                enable_mm_embeds=False, seed=0)


def equal_frame_indices(frames):
    first, inverse, by_hash = [], [], {}
    for i, frame in enumerate(frames):
        digest = bridge.sha(frame)
        if digest not in by_hash:
            by_hash[digest] = len(first)
            first.append(i)
        inverse.append(by_hash[digest])
    return first, inverse


def measure(engine, worker, processor, sampler, root, case, deep, profile_sha):
    import torch
    from vlm_psychophysics.results import score

    start = time.monotonic()
    request, input_evidence = model_request(root, case, processor, deep)
    if worker.cvpr_reuse_frames and case["modality"] == "video":
        worker.cvpr_equal_frame_indices = equal_frame_indices(request["multi_modal_data"]["video"][0])
    else:
        worker.cvpr_equal_frame_indices = None
    prepared = time.monotonic()
    worker.cvpr_vision_calls.clear()
    worker.cvpr_deep_input = deep
    result, = engine.generate(request, sampler, use_tqdm=False)
    output, = result.outputs
    n_images = len(case["images"])
    is_video = case["modality"] == "video"
    expected_shapes = []
    if n_images:
        expected_shapes.append([n_images, bridge.PATCHES, bridge.PIXELS_PER_PATCH])
    if is_video:
        expected_shapes.append([bridge.FRAMES, bridge.PATCHES, bridge.PIXELS_PER_PATCH])
    vision = worker.cvpr_vision_calls.copy()
    if sorted(v["shape"] for v in vision) != sorted(expected_shapes) or any(
        v["dtype"] != "torch.bfloat16" or not v["device"].startswith("cuda") for v in vision
    ):
        raise ValueError(f"Actual GPU ViT input differs: {case['case_id']} {vision}")
    if deep:
        expected = [input_evidence["image_rgb_sha256"]] if n_images else []
        if is_video:
            expected.append(input_evidence["source_frame_sha256"])
        if sorted(v["rgb_frame_sha256"] for v in vision) != sorted(expected):
            raise ValueError(f"Actual GPU RGB/frame order differs: {case['case_id']}")
    ids = list(result.prompt_token_ids)
    expected_visual = int(is_video) * bridge.VIDEO_EMBED_TOKENS + n_images * bridge.IMAGE_EMBED_TOKENS
    if (ids.count(worker.config.image_patch_id) != expected_visual
            or len(ids) + GENERATION["max_tokens"] > bridge.CONTEXT):
        raise ValueError(f"Expanded Molmo2 token grid or context differs for {case['case_id']}: {len(ids)}")
    ok = output.finish_reason == "stop"
    prediction, status, correct = score(output.text, case["choices"], case["truth"], True, "RETURNED", ok)
    return dict(
        case=case, profile_sha256=profile_sha, input=input_evidence, vision=vision,
        rendered_prompt=request["prompt"], prompt_token_ids=ids,
        prompt_token_ids_sha256=sha_text(json.dumps(ids, separators=(",", ":"))),
        prompt_tokens=len(ids), visual_tokens=expected_visual, raw_answer=output.text,
        output_token_ids=list(output.token_ids), finish_reason=output.finish_reason,
        stop_reason=output.stop_reason, prediction=prediction, status=status, correct=correct,
        media_seconds=round(prepared - start, 6), total_seconds=round(time.monotonic() - start, 6),
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if deep else None,
        distinct_frames=len(worker.cvpr_equal_frame_indices[0]) if worker.cvpr_equal_frame_indices else None,
    )


def report(out, answers, aliases):
    groups = defaultdict(Counter)
    pairs = []
    for row in answers.values():
        case = row["case"]
        for key in ("all", case["task"], f'{case["task"]}/{case["domain"]}'):
            counter = groups[f'{case["modality"]}/{key}']
            counter.update(n=1, correct=int(row["correct"]), legal=int(row["status"] == "SCORED"))
    for alias in aliases:
        video = answers[("video", alias["sample_id"])]
        static = answers[("static", alias["static_case_id"])]
        case = video["case"]
        for key in ("all", case["task"], f'{case["task"]}/{case["domain"]}'):
            groups[f"linked_static/{key}"].update(
                n=1, correct=int(static["correct"]), legal=int(static["status"] == "SCORED")
            )
        pairs.append(dict(sample_id=alias["sample_id"], static_case_id=alias["static_case_id"],
                          task=case["task"], domain=case["domain"], truth=case["truth"],
                          video=video["prediction"], static=static["prediction"],
                          video_status=video["status"], static_status=static["status"],
                          video_correct=int(video["correct"]), static_correct=int(static["correct"])))
    if len(pairs) != 1480 or groups["video/all"]["n"] != 1480 or groups["static/all"]["n"] != 1346:
        raise ValueError("Final Molmo2 denominator differs")
    with (out / "paired-answers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    write_json(out / "summary.json", {"status": "COMPLETE", "groups": dict(groups),
                                      "answers_sha256": file_sha(out / "answers.jsonl"),
                                      "pairs_sha256": file_sha(out / "paired-answers.csv")})


def main(args):
    from vllm import LLM, SamplingParams
    from vllm.transformers_utils.processor import get_processor

    out = args.output
    with (out / ".run.lock").open("a") as run_lock, args.gpu_lock.open("a") as gpu_lock:
        fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run(args, LLM, SamplingParams, get_processor)
        except Exception as error:
            existing = out / "status.json"
            state = json.loads(existing.read_text()) if existing.exists() else {}
            state.update(status="FAILED", error=f"{type(error).__name__}: {str(error)[:240]}")
            write_json(existing, state)
            raise


def run(args, LLM, SamplingParams, get_processor):
    out = args.output
    cases, aliases = cases_for(args.dataset)
    selected = canary_ids(cases)
    args_for_engine = engine_kwargs(args.checkpoint)
    if args.fast_prefix:
        args_for_engine["attention_backend"] = "TRITON_ATTN"
    checkpoint_receipt = out / "checkpoint-verification.json"
    checkpoint = json.loads(checkpoint_receipt.read_text())
    expected = (NVFP4_MODEL, NVFP4_REVISION, 3) if args.nvfp4 else (MODEL, MODEL_REVISION, 8)
    if (checkpoint["model"], checkpoint["revision"], len(checkpoint["weights"])) != expected:
        raise ValueError("Unexpected Molmo2 checkpoint or missing weight verification")
    files = sorted(p for p in args.checkpoint.iterdir() if p.is_file() and p.suffix != ".safetensors")
    profile = dict(
        model=checkpoint["model"], checkpoint=checkpoint, processor={p.name: file_sha(p) for p in files},
        runner=file_sha(Path(__file__)), input_bridge=file_sha(Path(bridge.__file__)),
        dataset_revision=DATASET_REVISION, selection_sha256=file_sha(args.dataset / "subset/selection.csv"),
        static_aliases_sha256=file_sha(args.dataset / "subset/static-aliases.csv"),
        metadata={str(p.relative_to(args.dataset)): file_sha(p) for p in args.dataset.glob("*/*/**/metadata.jsonl")},
        generation=GENERATION, engine=args_for_engine, position_extrapolation="unscaled RoPE",
        source_size_wh=[bridge.WIDTH, bridge.HEIGHT], padded_size_wh=[bridge.PAD_WIDTH, bridge.PAD_HEIGHT],
        patch_grid_hw=[bridge.PATCH_ROWS, bridge.PATCH_COLS], vision_position_interpolation="bicubic",
        video_pool_hw=[3, 3], still_pool_hw=[2, 2], still_overview="omitted",
        backend="vLLM native Molmo2", thinking="no separate channel", hostname=socket.gethostname(),
        gpu=os.environ["CUDA_VISIBLE_DEVICES"], gpu_lock=str(args.gpu_lock), python=sys.executable,
        packages={name: importlib.metadata.version(name) for name in ("torch", "vllm", "transformers", "numpy")},
    )
    if args.fast_prefix:
        import molmo2_prefix_attention as fast

        mask_source, kernel_source = fast.install(bridge.CONTEXT)
        profile["prefix_attention"] = {"source": file_sha(Path(fast.__file__)),
                                       "mask": sha_text(mask_source), "kernel": sha_text(kernel_source)}
    if args.flash_prefill:
        import molmo2_prefix_attention as fast

        fast.install_flash_prefill()
        profile["flash_prefill"] = file_sha(Path(fast.__file__))
    if args.reuse_frames:
        profile["duplicate_frames"] = "exact SigLIP input only, every timestamp and feature slot restored"
    if args.shard:
        profile["shard_sha256"] = file_sha(args.shard)
    profile_path = out / "profile.json"
    if profile_path.exists() and json.loads(profile_path.read_text()) != profile:
        raise ValueError("Existing Molmo2 results are bound to a different profile")
    write_json(profile_path, profile)
    profile_sha = file_sha(profile_path)
    write_json(out / "status.json", {"status": "LOADING", "pid": os.getpid(), "profile_sha256": profile_sha})
    startticks = int(Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19])
    write_json(out / "dispatch.json", {"pid": os.getpid(), "startticks": startticks,
                                       "profile_sha256": profile_sha})
    bridge.patch_vllm()
    processor = bridge.configure_processor(get_processor(str(args.checkpoint), trust_remote_code=True))
    template = args.checkpoint / "chat_template.jinja"
    processor.chat_template = template.read_text()
    engine = LLM(hf_overrides=bridge.hardware_config, **args_for_engine)
    worker, = engine.apply_model(lambda model: model)
    observation = bridge.observe_vision(worker)
    worker.cvpr_reuse_frames = args.reuse_frames
    if args.reuse_frames:
        bridge.reuse_equal_frame_features(worker)
    write_json(out / "vision.json", observation)
    sampler = SamplingParams(**GENERATION)

    raw = out / "answers.jsonl"
    saved = read_jsonl(raw)
    answers = {case_key(row["case"]): row for row in saved}
    all_cases = {case_key(case): case for case in cases}
    if len(cases) != len(all_cases):
        raise ValueError("Duplicate Molmo2 case within one modality")
    if len(saved) != len(answers) or any(
        row["profile_sha256"] != profile_sha or row["case"] != all_cases[key]
        for key, row in answers.items()
    ):
        raise ValueError("Prior Molmo2 records differ from the fixed panel or profile")

    if args.timing_cases:
        keys = [tuple(k) for k in json.loads(args.timing_cases.read_text())]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate timing case")
        for key in keys:
            if key not in answers:
                row = measure(engine, worker, processor, sampler, args.dataset, all_cases[key], True, profile_sha)
                append_records(raw, [row])
                answers[key] = row
        write_json(out / "status.json", {"status": "TIMING_COMPLETE", "done": len(answers),
                                         "profile_sha256": profile_sha})
        return

    incoming = sorted(cases, key=lambda c: (case_key(c) not in selected, c["modality"] != "video"))
    assigned = list(all_cases)
    if args.shard:
        assigned = read_shard(args.shard, all_cases)
        incoming = [all_cases[key] for key in assigned]
    audit = []
    if args.compare_raw:
        baseline = read_jsonl(args.compare_raw)
        if len(baseline) < 2:
            raise ValueError("At least two independent original-kernel answers are needed")
        for old in baseline:
            key = case_key(old["case"])
            case = all_cases[key]
            new = answers.get(key)
            if new is None:
                new = measure(engine, worker, processor, sampler, args.dataset, case, True, profile_sha)
                append_records(raw, [new])
                answers[key] = new
            fields = ("case", "input", "vision", "prompt_token_ids", "raw_answer", "output_token_ids",
                      "finish_reason", "stop_reason")
            identical = all(new[field] == old[field] for field in fields)
            audit.append({"case_id": case["case_id"], "modality": case["modality"], "exact": identical,
                          "old_profile_sha256": old["profile_sha256"], "seconds_old": old["total_seconds"],
                          "seconds_new": new["total_seconds"], "peak_cuda_bytes": new["peak_cuda_bytes"]})
            write_json(out / "backend-comparison.json", audit)
            if not identical:
                write_json(out / "backend-mismatch.json", {"original": old, "lookup": new})
                raise ValueError(f"Optimized exact-mask path changed {case['case_id']}")
        if not (out / "backend-mismatch.json").exists():
            write_json(out / "backend-admission.json", {"profile_sha256": profile_sha, "comparison": audit})
    if args.audit_only:
        write_json(out / "status.json", {"status": "COMPARISON_COMPLETE", "done": len(answers),
                                         "profile_sha256": profile_sha})
        return
    if args.shard and not answers.keys() <= set(assigned) | {case_key(old["case"]) for old in baseline}:
        raise ValueError("Existing answers are outside the Molmo2 shard and its qualification panel")
    write_json(out / "status.json", {"status": "RUNNING", "done": len(answers), "pid": os.getpid(),
                                     "expected": len(assigned), "profile_sha256": profile_sha})
    with raw.open("a", buffering=1) as stream:
        for case in incoming:
            key = case_key(case)
            deep = key in selected
            if key in answers or (args.canary_only and not deep):
                continue
            row = measure(engine, worker, processor, sampler, args.dataset, case, deep, profile_sha)
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            answers[key] = row
            if row["status"] == "TECHNICAL_FAILURE" and deep:
                raise ValueError(f"Molmo2 canary incomplete: {case['case_id']} {row['finish_reason']}")
            if deep or len(answers) % 20 == 0:
                counts = Counter(r["case"]["modality"] for r in answers.values())
                status = {"status": "RUNNING", "pid": os.getpid(), "done": len(answers),
                          "counts": dict(counts), "expected": len(assigned),
                          "profile_sha256": profile_sha, "last_seconds": row["total_seconds"]}
                write_json(out / "status.json", status)
                print(json.dumps(status), flush=True)
            if selected.issubset(answers):
                canary = out / "admission.json"
                if not canary.exists():
                    groups = defaultdict(list)
                    for key in sorted(selected):
                        result = answers[key]
                        groups[result["case"]["modality"]].append(dict(
                            case_id=key, prompt_tokens=result["prompt_tokens"], seconds=result["total_seconds"],
                            answer=result["raw_answer"], status=result["status"], vision=result["vision"],
                        ))
                    if any(not any(r["status"] == "SCORED" for r in group) for group in groups.values()):
                        raise ValueError("A modality could not produce any legal Molmo2 canary answer")
                    write_json(canary, {"profile_sha256": profile_sha, "canaries": groups})
            if (out / "pause-after-request").exists():
                write_json(out / "status.json", {"status": "PAUSED_AT_REQUEST_BOUNDARY",
                                                 "done": len(answers), "profile_sha256": profile_sha})
                return
    if args.canary_only:
        write_json(out / "status.json", {"status": "CANARIES_COMPLETE", "done": len(answers)})
    elif args.shard:
        if set(assigned) - answers.keys():
            raise ValueError("Incomplete Molmo2 shard")
        write_json(out / "status.json", {"status": "SHARD_COMPLETE", "done": len(assigned),
                                         "profile_sha256": profile_sha, "answers_sha256": file_sha(raw)})
    else:
        if set(answers) != set(all_cases):
            raise ValueError("Incomplete frozen panel")
        report(out, answers, aliases)
        write_json(out / "status.json", {"status": "COMPLETE", "done": len(answers),
                                         "summary_sha256": file_sha(out / "summary.json")})


def merge(args):
    cases, aliases = cases_for(args.dataset)
    panel = {case_key(case): case for case in cases}
    combined, profiles, source_rows = {}, [], {}
    common = None
    for shard in args.merge_shard:
        profile = json.loads((shard / "profile.json").read_text())
        state = json.loads((shard / "status.json").read_text())
        profile_sha = file_sha(shard / "profile.json")
        raw = shard / "answers.jsonl"
        assigned = read_shard(shard / "shard.json", panel)
        if (state["status"] != "SHARD_COMPLETE" or state["done"] != len(assigned)
                or state["profile_sha256"] != profile_sha or state["answers_sha256"] != file_sha(raw)
                or profile["shard_sha256"] != file_sha(shard / "shard.json")):
            raise ValueError(f"Molmo2 shard receipt differs: {shard}")
        immutable = {k: v for k, v in profile.items() if k not in {"gpu", "gpu_lock", "shard_sha256"}}
        if common is not None and immutable != common:
            raise ValueError("Molmo2 shard configurations differ")
        common = immutable
        saved = read_jsonl(raw)
        by_key = {case_key(row["case"]): row for row in saved}
        if len(saved) != len(by_key) or any(
            row["profile_sha256"] != profile_sha or row["case"] != panel[key]
            for key, row in by_key.items()
        ):
            raise ValueError(f"Molmo2 shard contains duplicate or mismatched answers: {shard}")
        profiles.append({"path": str(shard), "sha256": profile_sha, "gpu": profile["gpu"],
                         "answers_sha256": state["answers_sha256"], "assigned": len(assigned)})
        source_rows[shard] = by_key
        for key in assigned:
            if key in combined:
                raise ValueError(f"Molmo2 shards overlap at {key}")
            combined[key] = by_key[key]
    if combined.keys() != panel.keys():
        raise ValueError("Molmo2 shards do not cover the frozen panel")
    for by_key in source_rows.values():
        for key, row in by_key.items():
            fields = ("input", "vision", "prompt_token_ids", "raw_answer", "output_token_ids", "finish_reason")
            if any(row[field] != combined[key][field] for field in fields):
                raise ValueError(f"A shared qualification answer differs at {key}")
    target = args.output / "answers.jsonl"
    pending = target.with_suffix(".jsonl.tmp")
    with pending.open("w") as stream:
        for key in panel:
            stream.write(json.dumps(combined[key]) + "\n")
    pending.replace(target)
    write_json(args.output / "profiles.json", profiles)
    report(args.output, combined, aliases)
    write_json(args.output / "status.json", {"status": "COMPLETE", "done": len(combined),
                                             "summary_sha256": file_sha(args.output / "summary.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-lock", type=Path)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--fast-prefix", action="store_true")
    parser.add_argument("--flash-prefill", action="store_true")
    parser.add_argument("--reuse-frames", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--timing-cases", type=Path)
    parser.add_argument("--compare-raw", type=Path)
    parser.add_argument("--nvfp4", action="store_true")
    parser.add_argument("--shard", type=Path)
    parser.add_argument("--merge-shard", type=Path, action="append")
    args = parser.parse_args()
    if args.dataset.name != DATASET_REVISION:
        parser.error("Use the exact dataset snapshot")
    if args.merge_shard:
        merge(args)
        sys.exit(0)
    if args.shard and not args.compare_raw:
        parser.error("A sharded run requires shared qualification inputs")
    if not args.gpu_lock or not args.checkpoint or args.checkpoint.name != (NVFP4_REVISION if args.nvfp4 else MODEL_REVISION):
        parser.error("Use the exact checkpoint and dataset snapshots")
    main(args)
