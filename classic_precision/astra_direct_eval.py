"""Astra visual-only Responses comparison; preparation is offline by default.

Reuses the completed 20261027 samples and PNG bytes. No agent loop, decoding,
tools, secret-file reads, automatic inference retries, or scientific resampling.
Live standard calls require --execute, an explicit count, and a USD budget.
Batch export only writes local JSONL files; it never submits them.
After maintenance, refresh-preparation versions the unsubmitted plan and checks
its same three prepared request bytes. It never rewrites the original proof.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import struct
import time
import urllib.error
import urllib.request

NAME = "astra-direct-20261030"
OLD = "astra-path-only-20261027"
MODEL = "gpt-6-astra"
API = "https://api.openai.com/v1/responses"
CELLS = (
    [["T1", sf, m] for sf in (8, 32, 64) for m in (0.1, 0.4, 1)]
    + [["T2", 0.4, d, 30] for d in (0.05, 0.06, 0.07)]
    + [["T3", a] for a in (0.5, 0.7, 1)]
)
DOCS = {
    "benefit": "https://developers.openai.com/api/docs/guides/chatgpt-api-credits",
    "pricing": "https://developers.openai.com/api/docs/pricing",
    "model": "https://developers.openai.com/api/docs/models/gpt-6-astra",
    "vision": "https://developers.openai.com/api/docs/guides/images-vision",
    "tools": "https://developers.openai.com/api/docs/guides/tools",
    "cache": "https://developers.openai.com/api/docs/guides/prompt-caching",
    "batch": "https://developers.openai.com/api/docs/guides/batch",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def packed(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def paths(root):
    return Path(root) / "plans" / NAME, Path(root) / "runs/reproducibility" / NAME


def cost(input_tokens, output_tokens, factor=1.0):
    # Frozen 2026-10-30 short-context Standard reservation, without caching.
    require(
        type(input_tokens) is int and 0 <= input_tokens <= 272000,
        "Long-context/invalid input usage requires a new budget",
    )
    require(type(output_tokens) is int and output_tokens >= 0, "Invalid output usage")
    return factor * (input_tokens * 10 + output_tokens * 50) / 1e6


def usage_cost(usage):
    """Account for KV-cache reuse without treating it as missing visual input.

    Reads, writes and ordinary input are disjoint. Reasoning is already part of
    output_tokens. Missing write-lifetime details use the higher published rate
    as an upper bound; the raw provider usage is retained in every result.
    """
    inputs, outputs = usage["input_tokens"], usage["output_tokens"]
    cost(inputs, outputs)  # Validate the short-context envelope and token types.
    details = usage.get("input_tokens_details") or {}
    cached, written = details.get("cached_tokens", 0), details.get(
        "cache_write_tokens", 0
    )
    require(
        all(type(x) is int and x >= 0 for x in (cached, written))
        and cached + written <= inputs,
        "Invalid/disjoint cache token counts",
    )
    lifetimes = details.get("cache_write_tokens_details")
    basis = "published_short_context_standard_rates"
    if lifetimes is None:
        short, long = 0, written
        if written:
            basis = "upper_bound_unknown_write_lifetime_at_12h_rate"
    else:
        short, long = lifetimes.get("ttl_30m_tokens", 0), lifetimes.get(
            "ttl_12h_tokens", 0
        )
        require(
            all(type(x) is int and x >= 0 for x in (short, long))
            and short + long == written,
            "Invalid cache write lifetime counts",
        )
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens", 0)
    require(
        type(reasoning) is int and 0 <= reasoning <= outputs,
        "Invalid reasoning token count",
    )
    if "total_tokens" in usage:
        require(
            type(usage["total_tokens"]) is int
            and usage["total_tokens"] == inputs + outputs,
            "Invalid total token count",
        )
    ordinary = inputs - cached - written
    amount = (
        ordinary * 10 + cached * 1 + short * 12.5 + long * 25 + outputs * 50
    ) / 1e6
    return amount, {
        "basis": basis,
        "ordinary_input_tokens": ordinary,
        "cached_input_tokens": cached,
        "cache_write_tokens": written,
        "write_30m_tokens": short,
        "write_12h_tokens": long,
        "cache_reads_do_not_prove_or_disprove_visual_attention": True,
    }


def plan_file(p):
    maintained = p / "plan-maintained.json"
    return maintained if maintained.exists() else p / "plan.json"


@contextmanager
def execution_lock(out):
    """One local executor protects aggregate reservations and exclusive claims."""
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError(
                "Another executor holds the preparation budget lock"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def maintenance_guard(p, out, plan):
    if "maintenance" in plan:
        require(
            sha(p / "plan.json") == plan["maintenance"]["original_plan_sha256"],
            "Original preparation plan changed",
        )
        for name, expected in plan["maintenance"]["unchanged_input_bindings"].items():
            require(
                sha(out / name) == expected, "Original preparation evidence changed"
            )
        receipt = read(out / "maintenance-admission.json")
        require(
            receipt["status"] == "PASS_CPU_MAINTENANCE_NOT_SUBMITTED"
            and receipt["plan_sha256"] == sha(plan_file(p))
            and receipt["implementation_sha256"] == sha(Path(__file__)),
            "Maintenance admission changed/missing",
        )


def refresh_preparation(root):
    """Version code binding for the same unsubmitted60; never rewrite old proof."""
    p, out = paths(root)
    with execution_lock(out):
        require(
            not list((out / "claims").glob("*.json"))
            and not list((out / "cases").glob("*"))
            and not (out / "batch").exists(),
            "Existing attempts/export; preserve the plan and review without resending",
        )
        original = p / "plan.json"
        plan = read(original)
        rows = read(p / "samples.json")
        require(
            len(rows) == len({r["case_id"] for r in rows}) == 60
            and plan["max_scientific_calls"] == 60,
            "Maintenance must preserve the fixed60",
        )
        counts = Counter(tuple(r["cell"]) for r in rows)
        require(
            set(counts) == {tuple(c) for c in CELLS} and set(counts.values()) == {4},
            "Selected cells changed",
        )
        pilot, batch = plan["standard_pilot_ids"], plan["batch_ids"]
        require(
            len(pilot) == len(set(pilot)) == 3
            and len(batch) == len(set(batch)) == 57
            and not set(pilot) & set(batch)
            and set(pilot + batch) == {r["case_id"] for r in rows},
            "Pilot/Batch partition changed",
        )
        require(sha(p / "samples.json") == plan["samples_sha256"], "Selection changed")
        summary = read(out / "preparation-summary.json")
        require(
            summary["model_calls"] == 0
            and summary["actual_api_cost_usd"] == 0
            and summary["plan_sha256"] == sha(original),
            "Original preparation identity changed",
        )
        for name, key in [
            ("preflight.json", "preflight_sha256"),
            ("payload-admission.json", "payload_admission_sha256"),
            ("cost-estimate.json", "cost_estimate_sha256"),
        ]:
            require(
                sha(out / name) == summary[key], "Original preparation evidence changed"
            )
        preflight = read(out / "preflight.json")
        require(
            preflight["samples_sha256"] == plan["samples_sha256"]
            and preflight["frame_catalog_sha256"] == sha(out / "frame-catalog.json"),
            "Original preflight changed",
        )
        bindings = {
            name: sha(out / name)
            for name in (
                "preflight.json",
                "frame-catalog.json",
                "payload-admission.json",
            )
        }
        prior_requests = read(out / "payload-admission.json")["requests"]
        updated = dict(plan, implementation_sha256=sha(Path(__file__)))
        updated["maintenance"] = {
            "original_plan_sha256": sha(original),
            "original_implementation_sha256": plan["implementation_sha256"],
            "unchanged_input_bindings": bindings,
            "changes": ["cache usage accounting", "serial budget/claim lock"],
            "model_calls": 0,
        }
        write(p / "plan-maintained.json", updated)
        # Only the3 previously prepared payloads are reread; no decoding or resampling.
        requests = [build_request(root, cid)[1] for cid in pilot]
        require(
            {r["case_id"]: r["request_sha256"] for r in requests}
            == {r["case_id"]: r["request_sha256"] for r in prior_requests},
            "Prepared request bytes changed",
        )
        receipt = {
            "status": "PASS_CPU_MAINTENANCE_NOT_SUBMITTED",
            "model_calls": 0,
            "implementation_sha256": sha(Path(__file__)),
            "plan_sha256": sha(plan_file(p)),
            "original_plan_sha256": sha(original),
            "samples_sha256": plan["samples_sha256"],
            "frames_reread": 540,
            "new_decodes": 0,
            "requests": requests,
            "original_preparation_bindings": bindings,
            "official_sources": DOCS,
            "api_acceptance": "NOT_TESTED",
            "auth_reads": 0,
            "batch_uploads": 0,
        }
        write(out / "maintenance-admission.json", receipt)
        return {k: v for k, v in receipt.items() if k != "requests"}


def estimate(n):
    image_tokens = 180 * math.ceil(math.ceil(768 / 32) * math.ceil(512 / 32) * 1.2)
    return {
        "videos": n,
        "frames": 180 * n,
        "image_tokens_each_estimate": image_tokens,
        "text_tokens_each_assumption": 2500,
        "output_tokens_each_assumption_range": [1000, 4000],
        "standard_usd_range": [
            round(n * cost(image_tokens + 2500, x), 3) for x in (1000, 4000)
        ],
        "batch_usd_range": [
            round(n * cost(image_tokens + 2500, x, 0.5), 3) for x in (1000, 4000)
        ],
        "not_a_bill_or_provider_guarantee": True,
    }


def prepare(root):
    p, out = paths(root)
    oldp = Path(root) / "plans" / OLD
    oldr = Path(root) / "runs/reproducibility" / OLD
    sample_path = oldp / "samples.json"
    canonical_path = oldr / "repairs/capacity-retry1/canonical-primary-records.json"
    samples = read(sample_path)
    previous = read(canonical_path)
    require(
        len(samples) == 924 and len(previous) == 916, "Historical denominator changed"
    )
    by_sample = {x["sample_id"]: x for x in samples}
    require(len(by_sample) == len(samples), "Duplicate historical source ID")
    result_paths = {x.parent.name: x for x in oldr.rglob("result.json")}
    records = []
    for old in previous:
        if old["cell"] not in CELLS:
            continue
        s = by_sample[old["sample_id"]]
        require(
            old["visual_only_trace_status"] == "PASS_OBSERVED", "Unqualified baseline"
        )
        require(
            old["cell"] == s["cell"] and old["truth"] == s["truth"],
            "Historical join mismatch",
        )
        result_path = result_paths[old["case_id"]]
        require(
            sha(result_path) == old["identity"]["result_sha256"],
            "Historical result changed",
        )
        tokens = shlex.split(old["resume_command"])
        require(
            tokens[0] == "cd" and tokens[2:4] == ["&&", "codex"],
            "Unexpected historical working directory",
        )
        cwd = Path(tokens[1])
        require(
            cwd.parent == Path("/tmp") and re.fullmatch("[a-f0-9]{32}", cwd.name),
            "Non-neutral historical directory",
        )
        records.append(
            {
                "case_id": old["case_id"],
                "ordinal": s["ordinal"],
                "sample_id": old["sample_id"],
                "task": s["task"],
                "cell": s["cell"],
                "truth": s["truth"],
                "video_relative": s["source_row"]["video_relative"],
                "source_sha256": s["source_sha256"],
                "frames_directory": str(cwd),
                "historical_case_relative": str(result_path.parent.relative_to(oldr)),
                "historical_result_sha256": old["identity"]["result_sha256"],
                "historical_rollout_sha256": old["identity"]["rollout_sha256"],
                "historical_answer": old["answer"],
                "historical_correct": old["correct"],
                "historical_returned_images": old["returned_images"],
                "historical_protocol": old["protocol_status"],
                "historical_session_id": old["session_id"],
            }
        )
    records.sort(key=lambda x: x["ordinal"])
    counts = Counter(tuple(x["cell"]) for x in records)
    require(
        len(records) == 60 and len(counts) == 15 and set(counts.values()) == {4},
        "Expected 15 complete four-video cells",
    )
    for cell in CELLS:
        require(
            Counter(x["truth"] for x in records if x["cell"] == cell)
            == {"LEFT": 2, "RIGHT": 2},
            "Side balance",
        )
    pilot_cells = [["T1", 64, 1], ["T2", 0.4, 0.06, 30], ["T3", 0.7]]
    pilot = [
        next(x["case_id"] for x in records if x["cell"] == cell) for cell in pilot_cells
    ]
    for task in ("T1", "T2", "T3"):
        source = oldp / f"{task}-prompt.txt"
        (p / "prompts").mkdir(parents=True, exist_ok=True)
        target = p / "prompts" / source.name
        with target.open("xb") as f:
            f.write(source.read_bytes())
    write(p / "samples.json", records)
    usage = {}
    for task in ("T1", "T2", "T3"):
        rr = [r for r in previous if r["task"] == task]
        u = Counter()
        for r in rr:
            for v in r["usage"]:
                u.update(v)
        usage[task] = {
            "n": len(rr),
            "correct": sum(x["correct"] for x in rr),
            "full180": sum(x["returned_images"] == 180 for x in rr),
            "usage": dict(u),
        }
    proposed = estimate(60)
    proposed["three_standard_57_batch_usd_range"] = [
        round(x * (3 + 57 / 2) / 60, 3) for x in proposed["standard_usd_range"]
    ]
    proposed["three_standard_57_batch_reserved_usd"] = round(
        cost(100000, 4096) * (3 + 57 / 2), 4
    )
    budget = {
        "user_reported_credits_usd": 40,
        "user_reported_expiry": "2026-11-04; exact timezone/time unverified",
        "spending_ceiling_usd": 38,
        "buffer_usd": 2,
        "price_date": "2026-10-30",
        "official_sources": DOCS,
        "recommended_focus60": proposed,
        "alternatives": {
            k: estimate(n)
            for k, n in [
                ("T1_all80cells", 320),
                ("current_scope", 676),
                ("historical_primary", 916),
            ]
        },
        "credit_conditions": "Activated and linked API organization; first grant prorated; no Codex API-key usage; expires at billing-cycle end.",
        "assumptions": "Medium effort; original PNG; no cache; output allowance includes reasoning. Actual usage must be recorded; rates may change.",
        "historical_cli_usage_not_an_api_bill": usage,
    }
    write(out / "cost-estimate.json", budget)
    plan = {
        "schema": "astra-direct-v1",
        "status": "PREPARED_NOT_SUBMITTED",
        "model": MODEL,
        "implementation_sha256": sha(Path(__file__)),
        "reasoning_effort": "medium",
        "frames_per_request": 180,
        "detail": "original",
        "tools": [],
        "tool_choice": "none",
        "prompt_cache_options": {"mode": "explicit"},
        "max_output_tokens": 4096,
        "input_tokens_reservation": 100000,
        "max_scientific_calls": 60,
        "standard_pilot_ids": pilot,
        "batch_ids": [x["case_id"] for x in records if x["case_id"] not in pilot],
        "selection_cells": CELLS,
        "selection": "Fixed complete historical cells, no new-answer-dependent selection. Exploratory paired comparison, not full-grid accuracy.",
        "samples_sha256": sha(p / "samples.json"),
        "source_samples_sha256": sha(sample_path),
        "canonical_primary_sha256": sha(canonical_path),
        "prompts": {
            t: sha(p / "prompts" / f"{t}-prompt.txt") for t in ("T1", "T2", "T3")
        },
        "budget": {
            "credits_usd": 40,
            "ceiling_usd": 38,
            "reserved_usd": proposed["three_standard_57_batch_reserved_usd"],
        },
        "limits": [
            "Old path-only CLI could stop early; direct input supplies all180 at once. This changes exposure, harness, and conversation structure, not only tools.",
            "No temperature/seed claim; Astra medium is not a guarantee of repeatability or an immutable backend snapshot.",
            "PNG byte identity is client evidence, not proof of internal attention, precision, or processing.",
            "Provider compatibility, organization rate limits, and actual billing remain untested until explicitly authorized pilot.",
            "T2 uses only500ms; historical167ms and T3 auxiliary1.75/2.25 are excluded.",
        ],
    }
    write(p / "plan.json", plan)
    return {
        "status": plan["status"],
        "selected": len(records),
        "tasks": dict(Counter(x["task"] for x in records)),
        "budget": budget["recommended_focus60"],
    }


def load(root):
    p, out = paths(root)
    plan = read(plan_file(p))
    require(
        sha(Path(__file__)) == plan["implementation_sha256"],
        "Implementation changed; version the plan before inference",
    )
    require(sha(p / "samples.json") == plan["samples_sha256"], "Selection changed")
    for task, expected in plan["prompts"].items():
        require(sha(p / "prompts" / f"{task}-prompt.txt") == expected, "Prompt changed")
    return p, out, plan, read(p / "samples.json")


def frame_files(row):
    files = list(Path(row["frames_directory"]).rglob("*.png"))

    def index(p):
        match = re.search(r"(\d+)$", p.stem)
        require(match is not None, "Non-indexed PNG")
        return int(match[1])

    files.sort(key=index)
    require(
        [index(x) for x in files] in (list(range(180)), list(range(1, 181))),
        "Missing/extra/out-of-order frames",
    )
    return files


def png_shape(data):
    require(
        data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR",
        "Invalid PNG signature",
    )
    w, h, depth, color = struct.unpack(">IIBB", data[16:26])
    require(
        (w, h) == (768, 512) and depth in (8, 16) and color in (0, 2, 4, 6),
        "PNG geometry/format changed",
    )
    return depth, color


def observed_hashes(rollout):
    result = []
    with Path(rollout).open() as f:
        for line in f:
            event = json.loads(line)
            if event.get("type") != "response_item":
                continue
            output = event.get("payload", {}).get("output")
            if not isinstance(output, list):
                continue
            for item in output:
                if item.get("type") == "input_image":
                    data = item.get("image_url", "")
                    require(
                        data.startswith("data:image/png;base64,"),
                        "Historical image format changed",
                    )
                    result.append(
                        digest(base64.b64decode(data.split(",", 1)[1], validate=True))
                    )
    return result


def preflight(root):
    p, out, plan, rows = load(root)
    require(
        str(Path(root).resolve()).startswith("/home/yang/"),
        "Frame preflight runs on Fourier; no mass copying/decoding on Mac",
    )
    catalog = {}
    for row in rows:
        video = Path(root) / "datasets/classic-t1-t3-v0.4" / row["video_relative"]
        require(sha(video) == row["source_sha256"], "Source bytes changed")
        case = (
            Path(root) / "runs/reproducibility" / OLD / row["historical_case_relative"]
        )
        rollout = case / "rollout.jsonl"
        require(
            sha(rollout) == row["historical_rollout_sha256"],
            "Historical rollout changed",
        )
        frames = []
        for i, path in enumerate(frame_files(row)):
            data = path.read_bytes()
            depth, color = png_shape(data)
            frames.append(
                {
                    "index": i,
                    "path": str(path),
                    "sha256": digest(data),
                    "bytes": len(data),
                    "png_bit_depth": depth,
                    "png_color_type": color,
                }
            )
        old_hashes = observed_hashes(rollout)
        require(
            len(old_hashes) == row["historical_returned_images"],
            "Historical returned-image denominator",
        )
        require(
            old_hashes == [x["sha256"] for x in frames[: len(old_hashes)]],
            "Previously returned PNG bytes differ",
        )
        catalog[row["case_id"]] = frames
    write(out / "frame-catalog.json", catalog)
    receipt = {
        "status": "PASS_OFFLINE_INPUT_PREFLIGHT",
        "model_calls": 0,
        "cases": len(rows),
        "frames": 180 * len(rows),
        "all_sources_sha_match": True,
        "historically_returned_png_bytes_match": sum(
            r["historical_returned_images"] for r in rows
        ),
        "frame_catalog_sha256": sha(out / "frame-catalog.json"),
        "samples_sha256": plan["samples_sha256"],
        "no_crop_resize_reencode_decode": True,
        "API_acceptance": "NOT_TESTED",
    }
    # Build and inspect real payloads for all3 tasks without contacting a provider.
    write(out / "preflight.json", receipt)
    payload_checks = []
    for cid in plan["standard_pilot_ids"]:
        body, evidence = build_request(root, cid)
        payload_checks.append(evidence)
    write(
        out / "payload-admission.json",
        {
            "status": "PASS_CPU_THREE_TASK_PAYLOADS",
            "model_calls": 0,
            "requests": payload_checks,
        },
    )
    return receipt


def build_request(root, case_id):
    p, out, plan, rows = load(root)
    check = read(out / "preflight.json")
    require(check["samples_sha256"] == plan["samples_sha256"], "Stale preflight")
    require(
        sha(out / "frame-catalog.json") == check["frame_catalog_sha256"],
        "Frame catalog changed",
    )
    row = next(r for r in rows if r["case_id"] == case_id)
    catalog = read(out / "frame-catalog.json")[case_id]
    require(
        len(catalog) == 180 and [x["index"] for x in catalog] == list(range(180)),
        "Frame order/count",
    )
    prompt = (p / "prompts" / f'{row["task"]}-prompt.txt').read_text().rstrip()
    content = [
        {
            "type": "input_text",
            "text": prompt
            + "\n\nThe following full frames are supplied in chronological order at 60 fps, indices0 through179. Frame times are relative to the video start.",
        }
    ]
    for frame in catalog:
        data = Path(frame["path"]).read_bytes()
        require(digest(data) == frame["sha256"], "Frame bytes changed since preflight")
        png_shape(data)
        content += [
            {
                "type": "input_text",
                "text": f'Frame {frame["index"]:03d}; t={frame["index"] / 60:.6f} s',
            },
            {
                "type": "input_image",
                "detail": "original",
                "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
            },
        ]
    body = {
        "model": MODEL,
        "reasoning": {"effort": "medium"},
        "input": [{"role": "user", "content": content}],
        "tools": [],
        "tool_choice": "none",
        "prompt_cache_options": {"mode": "explicit"},
        "max_output_tokens": plan["max_output_tokens"],
        "truncation": "disabled",
        "store": False,
        "stream": False,
        "service_tier": "default",
    }
    text = "\n".join(x["text"] for x in content if x["type"] == "input_text")
    # UTF8 byte count is a conservative text-token bound, with image estimate and4096 headroom.
    tokens_upper = 180 * 461 + len(text.encode()) + 4096
    require(
        tokens_upper <= plan["input_tokens_reservation"],
        "Payload exceeds reserved input envelope",
    )
    for forbidden in (
        row["sample_id"],
        row["frames_directory"],
        row["video_relative"],
        "ground_truth",
        "source_sha256",
    ):
        require(forbidden not in text, "Private label/path leaked into prompt")
    wire = packed(body)
    require(
        len(wire) < 190_000_000, "Single request too large for prepared batch shard"
    )
    receipt = {
        "case_id": case_id,
        "request_sha256": digest(wire),
        "bytes": len(wire),
        "frames": 180,
        "input_tokens_estimated_upper_with_headroom": tokens_upper,
        "tools": [],
        "tool_choice": "none",
        "detail": "original",
        "no_source_paths_or_labels": True,
        "implementation_sha256": plan["implementation_sha256"],
        "plan_sha256": sha(plan_file(p)),
        "png_hashes": [x["sha256"] for x in catalog],
    }
    return body, receipt


def parse_response(response):
    require(
        response.get("status") == "completed",
        "Incomplete provider response; retain without retry",
    )
    require(response.get("model") == MODEL, "Served model differs from frozen plan")
    require(
        response.get("service_tier", "default") in ("default", "batch"),
        "Unexpected paid service tier",
    )
    output = response.get("output", [])
    require(
        all(x.get("type") in ("message", "reasoning") for x in output),
        "Unexpected tool/other output; no tool is executed",
    )
    parts = [
        c for x in output if x.get("type") == "message" for c in x.get("content", [])
    ]
    require(
        all(x.get("type") == "output_text" for x in parts), "Refusal/nontext response"
    )
    answer = "".join(x["text"] for x in parts).strip()
    usage = response.get("usage")
    require(usage is not None, "Missing usage")
    amount, billing = usage_cost(usage)
    return {
        "answer_raw": answer,
        "answer": answer if answer in ("LEFT", "RIGHT") else None,
        "strict_format": answer in ("LEFT", "RIGHT"),
        "usage": usage,
        "standard_cost_usd": amount,
        "billing": billing,
        "response_id": response.get("id"),
        "zero_tool_calls": True,
    }


def run_standard(root, ids, budget, execute):
    require(execute, "Preparation only: --execute required for live requests")
    p, out, plan, rows = load(root)
    require(
        0 < budget <= plan["budget"]["ceiling_usd"],
        "Explicit budget must be within prepared ceiling",
    )
    require(
        ids
        and len(ids) == len(set(ids))
        and set(ids) <= set(plan["standard_pilot_ids"]),
        "Only the3 pilot IDs are Standard eligible",
    )
    require(not (out / "PAUSE").exists(), "Paused before launch")
    reserve = cost(plan["input_tokens_reservation"], plan["max_output_tokens"])
    maintenance_guard(p, out, plan)
    with execution_lock(out):
        # Serial execution and exclusive claims; unresolved attempts remain reserved.
        require(
            not list((out / "cases").glob("*/failure.json")),
            "Existing failed/uncertain attempt; review without retry",
        )
        for cid in ids:
            require(
                not (out / "cases" / cid / "failure.json").exists(),
                "Existing failed/uncertain attempt; review without retry",
            )
            if (out / "claims" / f"{cid}.json").exists():
                old_result = out / "cases" / cid / "result.json"
                require(
                    old_result.exists() and read(old_result)["strict_format"],
                    "Existing unresolved claim; never resend",
                )
                continue
            claims = (
                list((out / "claims").glob("*.json"))
                if (out / "claims").exists()
                else []
            )
            held = sum(read(x)["reserved_usd"] for x in claims)
            require(held + reserve <= budget + 1e-9, "Budget reservation exhausted")
            if (out / "PAUSE").exists():
                return {"status": "PAUSED_BETWEEN_REQUESTS"}
            body, evidence = build_request(root, cid)
            key = os.environ.get("OPENAI_API_KEY")
            require(
                bool(key),
                "Set the linked-organization OPENAI_API_KEY securely on Fourier; do not put it in files or prompts",
            )
            write(
                out / "claims" / f"{cid}.json",
                {
                    "case_id": cid,
                    "reserved_usd": reserve,
                    "request_sha256": evidence["request_sha256"],
                },
            )
            case = out / "cases" / cid
            write(case / "request-receipt.json", evidence)
            headers = {
                "Content-Type": "application/json",
                "Authorization": "Bearer " + key,
            }
            if os.environ.get("OPENAI_PROJECT_ID"):
                headers["OpenAI-Project"] = os.environ["OPENAI_PROJECT_ID"]
            request = urllib.request.Request(
                API, data=packed(body), headers=headers, method="POST"
            )
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=900) as res:
                    raw = res.read()
                    req_id = res.headers.get("x-request-id")
                case.mkdir(parents=True, exist_ok=True)
                with (case / "response.json").open("xb") as f:
                    f.write(raw)
                answer = parse_response(json.loads(raw))
                row = next(r for r in rows if r["case_id"] == cid)
                answer.update(
                    {
                        "case_id": cid,
                        "sample_id": row["sample_id"],
                        "correct": answer["answer"] == row["truth"],
                        "old_answer": row["historical_answer"],
                        "answer_changed": answer["answer"] != row["historical_answer"],
                        "wall_seconds": time.monotonic() - started,
                        "request_id": req_id,
                        "response_sha256": digest(raw),
                        "request_sha256": evidence["request_sha256"],
                    }
                )
                write(case / "result.json", answer)
                require(answer["strict_format"], "Format failure retained; no retry")
                require(
                    answer["standard_cost_usd"] <= reserve,
                    "Actual cost exceeds reservation; stop",
                )
            except Exception as exc:
                # No credential/header or unrestricted exception dumps.
                error = {
                    "status": "HOLD_NO_RETRY",
                    "type": type(exc).__name__,
                    "wall_seconds": time.monotonic() - started,
                }
                if isinstance(exc, urllib.error.HTTPError):
                    error["http_status"] = exc.code
                    payload = exc.read().replace(key.encode(), b"[REDACTED]")
                    with (case / "provider-error.json").open("xb") as f:
                        f.write(payload)
                write(case / "failure.json", error)
                raise RuntimeError(
                    f"{cid}: HOLD_NO_RETRY; inspect saved evidence"
                ) from None
        return {"status": "STANDARD_PILOT_COMPLETED", "count": len(ids)}


def export_batch(root):
    p, out, plan, rows = load(root)
    maintenance_guard(p, out, plan)
    pilot, batch = plan["standard_pilot_ids"], plan["batch_ids"]
    require(
        len(pilot) == len(set(pilot))
        and len(batch) == len(set(batch))
        and not set(pilot) & set(batch)
        and set(pilot + batch) == {r["case_id"] for r in rows},
        "Duplicate/missing pilot/Batch ID",
    )
    reserve = cost(plan["input_tokens_reservation"], plan["max_output_tokens"]) * (
        len(pilot) + len(batch) / 2
    )
    require(
        reserve <= plan["budget"]["ceiling_usd"],
        "Complete pilot/Batch reservation exceeds ceiling",
    )
    with execution_lock(out):
        # Do not queue the remaining budget until real three-task pilot is accepted.
        for cid in plan["standard_pilot_ids"]:
            require(
                not (out / "cases" / cid / "failure.json").exists(),
                "Pilot has a recorded failure; do not export",
            )
            r = read(out / "cases" / cid / "result.json")
            require(
                r["strict_format"] and r["zero_tool_calls"], "Pilot acceptance required"
            )
            require(
                r["standard_cost_usd"]
                <= cost(plan["input_tokens_reservation"], plan["max_output_tokens"]),
                "Pilot cost exceeds reserved budget",
            )
        folder = out / "batch"
        require(not folder.exists(), "Batch already prepared; never duplicate blindly")
        folder.mkdir()
        shards = []
        handle = None
        size = 0
        file = None
        try:
            for cid in plan["batch_ids"]:
                body, evidence = build_request(root, cid)
                body.pop(
                    "service_tier"
                )  # Batch service controls tier; no priority request.
                line = (
                    packed(
                        {
                            "custom_id": cid,
                            "method": "POST",
                            "url": "/v1/responses",
                            "body": body,
                        }
                    )
                    + b"\n"
                )
                if handle is None or size + len(line) > 190_000_000:
                    if handle:
                        handle.close()
                        shards[-1]["sha256"] = sha(file)
                    file = folder / f"input-{len(shards):03d}.jsonl"
                    handle = file.open("xb")
                    shards.append({"path": str(file), "count": 0, "bytes": 0})
                    size = 0
                handle.write(line)
                size += len(line)
                shards[-1]["count"] += 1
                shards[-1]["bytes"] = size
            if handle:
                handle.close()
                handle = None
                shards[-1]["sha256"] = sha(file)
        finally:
            if handle:
                handle.close()
        receipt = {
            "status": "EXPORTED_NOT_UPLOADED_OR_SUBMITTED",
            "model_calls": 0,
            "requests": len(batch),
            "reserved_usd": reserve,
            "batch_ids": batch,
            "excluded_standard_pilot_ids": pilot,
            "endpoint": "/v1/responses",
            "completion_window": "24h",
            "shards": shards,
            "warning": "Upload/submit exactly once after budget approval. Official per-file limit200MB; these shards stay under190MB.",
        }
        write(folder / "export.json", receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "preflight",
            "refresh-preparation",
            "run-pilot",
            "export-batch",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--budget-usd", type=float, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        value = prepare(args.root)
    elif args.command == "refresh-preparation":
        value = refresh_preparation(args.root)
    elif args.command == "preflight":
        value = preflight(args.root)
    elif args.command == "export-batch":
        value = export_batch(args.root)
    else:
        value = run_standard(
            args.root,
            load(args.root)[2]["standard_pilot_ids"],
            args.budget_usd,
            args.execute,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
