"""Plan, pilot, resume and package the frozen benchmark on hosted APIs."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
import fcntl
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import sqlite3
from tempfile import TemporaryDirectory
import time
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zipfile import ZIP_DEFLATED, ZipFile

from . import jinwei_release as release
from . import native_video_api
from .provider_api import ROUTES, parameters

SCHEMA = "jinwei-provider-v1"
PROVIDER = "nvidia-internal"
BASE_URL, KEY_ENV, _ = ROUTES[PROVIDER]
TASK_SCOPES = {
    f"{task}-{domain}": scope
    for scope in release.SCOPES
    for domain, task in [scope.split("/")]
}
RETRIES = {"SOURCE_ERROR", "CLAIMED", "HTTP_ERROR"}
UNCERTAIN = {"SENT", "TRANSPORT_UNKNOWN"}
RETURN_FILES = (
    "plan.json",
    "source-plan.jsonl",
    "config.json",
    "attempts.jsonl",
    "results.jsonl",
    "summary.json",
)


def atomic(path, data):
    temp = path.with_name("." + path.name + ".partial")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def file_sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_rows(raw):
    return [json.loads(line) for line in raw.splitlines()]


def load_plan(out):
    plan = json.loads((out / "plan.json").read_bytes())
    release.require(
        plan["schema"] == SCHEMA
        and release.known_public_release(plan["revision"], plan["index_sha256"]),
        "Unsupported run plan",
    )
    raw = (out / "source-plan.jsonl").read_bytes()
    rows = load_rows(release.checked(raw, plan["source_plan_sha256"], "source plan"))
    release.require(release.counts(rows) == plan["profiles"], "Plan counts differ")
    return plan, rows


def build_plan(out, metadata_root=None, private=None):
    rows = release.public_plan(metadata_root)
    if private:
        rows.extend(private.plan(rows))
    rows.sort(key=lambda r: (r["scope"], r["input"], r["sample_id"]))
    raw = b"".join(map(release.encoded, rows))
    plan = dict(
        schema=SCHEMA,
        repo=release.REPO,
        revision=release.REVISION,
        index_sha256=release.INDEX_SHA,
        private_index_sha256=release.PRIVATE_SHA if private else None,
        source_plan_sha256=release.sha(raw),
        profiles=release.counts(rows),
        logical_rows=len(rows),
        unique_requests=len(release.unique_requests(rows)),
        static_policy="one provider call per frozen static_case_id; project to matching video IDs",
        t5_static_policy="reference first, then event midpoints in chronological order",
    )
    if (out / "plan.json").exists():
        release.require(
            load_plan(out)[0] == plan, "This output directory already has another plan"
        )
    else:
        atomic(out / "source-plan.jsonl", raw)
        atomic(out / "plan.json", release.encoded(plan))
    return plan, rows


def configured(out, plan, model, max_tokens, provider=PROVIDER, all_frames=False):
    release.require(
        bool(model) and bool(re.fullmatch(r"[A-Za-z0-9_.:/-]+", model)),
        "Set --model to the exact provider model ID",
    )
    hosted = native_video_api.HOSTED.get(provider)
    endpoint = BASE_URL
    if hosted:
        endpoint = native_video_api.endpoint(provider)
    elif provider == "vllm":
        endpoint = native_video_api.vllm_endpoint(os.environ.get("VLLM_BASE_URL", ""))
    config = dict(
        schema=SCHEMA,
        provider=provider,
        endpoint=endpoint,
        requested_model=model,
        max_completion_tokens=max_tokens,
        source_plan_sha256=plan["source_plan_sha256"],
        code_sha256={
            p.name: file_sha(p)
            for p in (
                Path(__file__),
                Path(release.__file__),
                Path(__file__).with_name("provider_api.py"),
            )
        },
        generation="provider defaults apart from token limit; no tools; no schema constraint",
        transport="inline MIME-correct data URLs, image/reference then video as applicable",
        processor_visibility="not exposed by provider",
        answer_parser="single-label-v1",
    )
    if hosted:
        config.update(video_fps=hosted[1], transport="inline video bytes")
        config["code_sha256"]["native_video_api.py"] = file_sha(
            Path(native_video_api.__file__)
        )
        native_video_api.validate_config(config)
    if all_frames:
        config["frame_settings"] = native_video_api.FULL_FRAME_FIELDS
        config["code_sha256"]["native_video_api.py"] = file_sha(
            Path(native_video_api.__file__)
        )
    path = out / "config.json"
    if path.exists():
        release.require(
            json.loads(path.read_bytes()) == config,
            "Model, request parameters, release, or runner changed. Use a new output directory.",
        )
    else:
        atomic(path, release.encoded(config))
    return config


def open_db(out):
    db = sqlite3.connect(out / "attempts.sqlite3", timeout=30)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute(
        """CREATE TABLE IF NOT EXISTS attempts (
        request_id TEXT NOT NULL, attempt INTEGER NOT NULL, details BLOB NOT NULL, body BLOB,
        PRIMARY KEY (request_id, attempt))"""
    )
    return db


def save(db, receipt, body=None, new=False):
    if new:
        db.execute(
            "INSERT INTO attempts VALUES (?, ?, ?, ?)",
            (receipt["request_id"], receipt["attempt"], release.encoded(receipt), body),
        )
    else:
        changed = db.execute(
            "UPDATE attempts SET details=?, body=? WHERE request_id=? AND attempt=?",
            (release.encoded(receipt), body, receipt["request_id"], receipt["attempt"]),
        )
        release.require(changed.rowcount == 1, "Attempt disappeared")
    db.commit()


def records(db):
    rows = defaultdict(list)
    for (data,) in db.execute(
        "SELECT details FROM attempts ORDER BY request_id, attempt"
    ):
        row = json.loads(data)
        rows[row["request_id"]].append(row)
    return rows


def validate_frame_report(report, attempt, config):
    indices = list(range(180))
    release.require(
        attempt["phase"] == "pilot"
        and attempt.get("modality_admitted")
        and report["endpoint"] == config["endpoint"]
        and report["model"] == attempt["returned_model"]
        and report["request_sha256"] == attempt["request_sha256"]
        and isinstance(report["deployment"], str)
        and bool(report["deployment"].strip())
        and report["decoded_indices"] == report["processor_indices"] == indices,
        "The server report must bind the successful pilot and show decoded and processor indices 0–179 in order",
    )


def attach_frame_reports(path, db, history, config):
    reports = json.loads(path.read_bytes())
    release.require(
        isinstance(reports, list) and bool(reports),
        "Expected a nonempty list of server reports",
    )
    pilots = {
        a["request_sha256"]: a
        for attempts in history.values()
        for a in attempts
        if a["phase"] == "pilot" and a.get("modality_admitted")
    }
    staged = {}
    deployments = {
        a["frame_report"]["deployment"] for a in pilots.values() if "frame_report" in a
    }
    for report in reports:
        release.require(isinstance(report, dict), "Expected a server report object")
        attempt = pilots.get(report["request_sha256"])
        release.require(
            attempt is not None,
            "Server report does not match a successful pilot in this run",
        )
        validate_frame_report(report, attempt, config)
        evidence = {
            key: report[key]
            for key in (
                "endpoint",
                "model",
                "deployment",
                "request_sha256",
                "decoded_indices",
                "processor_indices",
            )
        }
        prior = attempt.get(
            "frame_report", staged.get(attempt["request_sha256"], evidence)
        )
        release.require(
            prior == evidence, "Server frame evidence differs from the accepted report"
        )
        staged[attempt["request_sha256"]] = evidence
        deployments.add(evidence["deployment"])
    release.require(
        len(deployments) == 1,
        "Use a separate output directory for each server deployment",
    )
    with db:
        for digest, report in staged.items():
            attempt = pilots[digest]
            attempt["frame_report"] = report
            db.execute(
                "UPDATE attempts SET details=? WHERE request_id=? AND attempt=?",
                (release.encoded(attempt), attempt["request_id"], attempt["attempt"]),
            )


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def transport(payload, credential, config=None):
    endpoint = config["endpoint"] if config else BASE_URL
    if config and config["provider"] == "gemini":
        endpoint += "/models/" + config["requested_model"] + ":generateContent"
        auth = {"x-goog-api-key": credential}
    else:
        endpoint += "/chat/completions"
        auth = {"Authorization": "Bearer " + credential}
    request = Request(
        endpoint,
        data=payload,
        headers={**auth, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = build_opener(NoRedirect).open(request, timeout=300)
    except HTTPError as response:
        with response:
            return response.code, response.headers, response.read()
    with response:
        return response.status, response.headers, response.read()


def response_fields(body, status, choices, expected_model, provider=PROVIDER):
    if not 200 <= status < 300:
        return {"status": "HTTP_ERROR"}
    try:
        document = json.loads(body)
        if provider == "gemini":
            document = native_video_api.gemini_completion(document)
        options = document["choices"]
        if len(options) != 1:
            raise ValueError("Expected one completion choice")
        option = options[0]
        message = option["message"]
        raw = message.get("content")
        model = document["model"]
        finish = option.get("finish_reason")
        tools = bool(message.get("tool_calls") or message.get("function_call"))
        if not isinstance(model, str) or not model:
            raise ValueError("Missing returned model identity")
    except (KeyError, IndexError, TypeError, AttributeError, ValueError):
        return {"status": "RESPONSE_ENVELOPE_ERROR"}
    complete = (
        isinstance(raw, str) and bool(raw.strip()) and finish == "stop" and not tools
    )
    label = re.sub(r"[.!]$", "", raw.strip().upper()) if complete else None
    if label not in choices:
        label = None
    status_name = (
        "MODEL_ID_CHANGED"
        if expected_model and model != expected_model
        else (
            "TRUNCATED"
            if finish == "length"
            else "ANSWERED" if label is not None else "FORMAT_OR_TOOL_FAILURE"
        )
    )
    return dict(
        status=status_name,
        returned_model=model,
        response_id=document.get("id"),
        system_fingerprint=document.get("system_fingerprint"),
        usage=document.get("usage"),
        finish_reason=finish,
        tool_calls_returned=tools,
        raw_answer=raw,
        normalized_answer=label if status_name == "ANSWERED" else None,
        strict_format_valid=complete and raw in choices and status_name == "ANSWERED",
        modality_admitted=status_name == "ANSWERED",
    )


def request_body(row, config, media_root, private):
    provider = config.get("provider", PROVIDER)
    media = []
    verified = []
    for item in row["media"]:
        local = media_root / item["path"] if media_root is not None else None
        if (
            row["source"] == "private"
            and (local is None or not local.is_file())
            and private
        ):
            data = private.media_bytes(item)
        elif local is not None:
            data = local.read_bytes()
        elif row["source"] == "public":
            data = release.public_media(item)
        else:
            raise ValueError("Authorized private media is missing")
        release.checked(data, item["sha256"], item["path"])
        media.append((item["mime"], base64.b64encode(data).decode("ascii")))
        verified.append({**item, "bytes": len(data)})
    if provider in native_video_api.HOSTED:
        body = native_video_api.request(
            provider,
            config["requested_model"],
            row["prompt"],
            media,
            config["max_completion_tokens"],
            config["video_fps"],
        )
    else:
        body = parameters(
            PROVIDER,
            config["requested_model"],
            row["prompt"],
            "<media>",
            config["max_completion_tokens"],
        )
        parts = []
        for mime, data in media:
            kind = "image_url" if mime == "image/png" else "video_url"
            parts.append({"type": kind, kind: {"url": f"data:{mime};base64,{data}"}})
        body["messages"][0]["content"] = parts + [
            {"type": "text", "text": row["prompt"]}
        ]
        body.update(config.get("frame_settings", {}))
    payload = release.encoded(body)
    if provider == "gemini":
        release.require(
            len(payload) <= 20_000_000, "Gemini inline request exceeds 20 MB"
        )
    return payload, verified


def invoke(out, receipt, row, config, media_root, private, credential, expected_model):
    with closing(open_db(out)) as db:
        started = time.monotonic()
        try:
            payload, verified = request_body(row, config, media_root, private)
        except Exception as error:
            # Exception messages from network clients can contain signed URLs.
            receipt.update(
                status="SOURCE_ERROR",
                error_type=type(error).__name__,
                source_error=str(error) if isinstance(error, ValueError) else None,
            )
            receipt.update(
                finished_unix=time.time(), elapsed_seconds=time.monotonic() - started
            )
            save(db, receipt)
            return receipt
        receipt.update(
            status="SENT",
            request_sha256=release.sha(payload),
            verified_media=verified,
            sent_unix=time.time(),
        )
        save(db, receipt)
        raw = None
        try:
            http_status, headers, raw = transport(payload, credential, config)
            receipt.update(
                http_status=http_status,
                response_body_sha256=release.sha(raw),
                response_bytes=len(raw),
            )
            if credential.encode() in raw:
                receipt.update(
                    status="SENSITIVE_RESPONSE_WITHHELD",
                    response_body_withheld="credential_echo",
                )
                raw = None
            elif re.search(
                rb"data:(?:image|video)/[a-z0-9.-]+;base64,", raw, re.IGNORECASE
            ) or (
                config["provider"] == "gemini"
                and re.search(rb'"inline_?data"\s*:', raw, re.IGNORECASE)
            ):
                receipt.update(
                    status="SENSITIVE_RESPONSE_WITHHELD",
                    response_body_withheld="media_echo",
                )
            else:
                receipt.update(
                    provider_request_id=headers.get("x-request-id")
                    or headers.get("request-id"),
                    retry_after=headers.get("retry-after"),
                    **response_fields(
                        raw,
                        http_status,
                        row["choices"],
                        expected_model,
                        config["provider"],
                    ),
                )
        except (OSError, HTTPException) as error:
            receipt.update(status="TRANSPORT_UNKNOWN", error_type=type(error).__name__)
        receipt.update(
            finished_unix=time.time(), elapsed_seconds=time.monotonic() - started
        )
        save(db, receipt, raw)
        return receipt


def selected_requests(rows, args):
    return {
        key: row
        for key, row in release.unique_requests(rows).items()
        if (not args.scope or row["scope"] in args.scope)
        and (not args.input or row["input"] == args.input)
    }


def execution(out, plan, rows, args, private, pilot):
    release.require(
        (plan["revision"], plan["index_sha256"])
        == (release.REVISION, release.INDEX_SHA),
        "Resume an earlier release with the checkout that created the run plan.",
    )
    selected = selected_requests(rows, args)
    release.require(bool(selected), "The selection contains no callable trials")
    release.require(
        args.execute and args.max_requests > 0,
        "Set --execute and a positive --max-requests",
    )
    hosted = native_video_api.HOSTED.get(args.provider)
    if args.all_frames:
        release.require(
            not hosted,
            "This provider samples video. Choose a full-frame deployment or pass --sampled-video.",
        )
        release.require(
            all(
                r["input"] == "video"
                and (
                    r["source_video_format"]["frames"],
                    r["source_video_format"]["fps"],
                )
                == (180, 60)
                for r in selected.values()
            ),
            "Select a 180-frame, 60 fps video task for the all-frame run",
        )
    if hosted:
        release.require(
            all(
                r["source"] == "public"
                and r["input"] == "video"
                and len(r["media"]) == 1
                and r["media"][0]["mime"] in hosted[2]
                for r in selected.values()
            ),
            "Select supported public single-video scopes with --scope and --input video",
        )
    if any(r["source"] == "private" for r in selected.values()):
        release.require(
            args.allow_private_media_transfer
            and (args.media_root is not None or private is not None),
            "Private inference requires authorized source access and --allow-private-media-transfer",
        )
    if private and not private.locations:
        public_rows = [row for row in rows if row["source"] == "public"]
        release.require(
            private.plan(public_rows)
            == sorted(
                (row for row in rows if row["source"] == "private"),
                key=lambda r: (r["scope"], r["sample_id"]),
            ),
            "Private metadata no longer matches the run plan",
        )
    config = configured(
        out, plan, args.model, args.max_tokens, args.provider, args.all_frames
    )
    with closing(open_db(out)) as db:
        history = records(db)
        if args.frame_report and not pilot:
            release.require(args.all_frames, "--frame-report requires an all-frame run")
            attach_frame_reports(args.frame_report, db, history, config)
        profiles = defaultdict(list)
        for key, row in selected.items():
            profiles[row["profile"]].append(key)
        admitted = {
            r["profile"]: r["returned_model"]
            for attempts in history.values()
            for r in attempts
            if r["phase"] == "pilot"
            and r.get("modality_admitted")
            and (pilot or not args.all_frames or r.get("frame_report"))
        }
        models = set(admitted.values())
        release.require(len(models) <= 1, "Pilots returned different model identities")
        expected_model = next(iter(models), None)
        blocked = sorted(profiles.keys() - admitted.keys())
        if pilot:
            overrides = {}
            for key in args.pilot_trial or []:
                release.require(
                    key in selected and key not in history,
                    f"Pilot trial {key} is unknown or already asked",
                )
                profile = selected[key]["profile"]
                release.require(
                    profile in blocked and profile not in overrides,
                    f"Pilot profile {profile} is already admitted or has two overrides",
                )
                overrides[profile] = key
            keys = [
                overrides.get(profile, min(profiles[profile])) for profile in blocked
            ]
        else:
            keys = [key for key, row in selected.items() if row["profile"] in admitted]
        retry = set(args.retry or [])
        release.require(
            retry <= selected.keys(), "A retry ID is outside the callable selection"
        )
        eligible = []
        for key in keys:
            attempts = history.get(key, [])
            if not attempts:
                eligible.append(key)
            elif key in retry:
                previous = attempts[-1]
                can_retry = previous["status"] in RETRIES or (
                    args.retry_uncertain and previous["status"] in UNCERTAIN
                )
                release.require(
                    len(attempts) == 1 and can_retry,
                    f"Trial {key} cannot be retried under the current flags",
                )
                eligible.append(key)
        release.require(
            retry <= set(eligible),
            "A requested retry is outside this pilot or admitted run",
        )
        if not eligible:
            return dict(status="NO_NEW_REQUESTS", started=0, blocked_profiles=blocked)
        key_env = (
            hosted[0]
            if hosted
            else "VLLM_API_KEY" if args.provider == "vllm" else KEY_ENV
        )
        credential = os.environ.get(key_env)
        release.require(bool(credential), f"Set {key_env} in the operator shell")
        queue = iter(eligible[: args.max_requests])
        started, recent_failures = 0, 0
        stop = None
        statuses = Counter()
        workers = 1 if pilot else args.workers
        with ThreadPoolExecutor(max_workers=workers) as pool:
            running = set()

            def schedule():
                nonlocal started
                key = next(queue, None)
                if key is None:
                    return
                row = selected[key]
                receipt = dict(
                    request_id=key,
                    attempt=len(history.get(key, [])) + 1,
                    profile=row["profile"],
                    sample_id=row["sample_id"],
                    source=row["source"],
                    status="CLAIMED",
                    phase="pilot" if pilot else "run",
                    started_unix=time.time(),
                )
                save(db, receipt, new=True)
                running.add(
                    pool.submit(
                        invoke,
                        out,
                        receipt,
                        row,
                        config,
                        args.media_root,
                        private,
                        credential,
                        expected_model,
                    )
                )
                started += 1

            for _ in range(workers):
                schedule()
            while running:
                finished, running = wait(running, return_when=FIRST_COMPLETED)
                for job in finished:
                    item = job.result()
                    status = item["status"]
                    statuses[status] += 1
                    if pilot and item.get("modality_admitted"):
                        admitted[item["profile"]] = item["returned_model"]
                        expected_model = item["returned_model"]
                    failed = status in {
                        "SOURCE_ERROR",
                        "HTTP_ERROR",
                        "TRANSPORT_UNKNOWN",
                        "RESPONSE_ENVELOPE_ERROR",
                    }
                    recent_failures = recent_failures + 1 if failed else 0
                    if status in {
                        "MODEL_ID_CHANGED",
                        "SENSITIVE_RESPONSE_WITHHELD",
                    } or item.get("http_status") in (401, 403, 429):
                        stop = f'Stopped on {item.get("http_status") or status}'
                    elif recent_failures >= 3 and not pilot:
                        stop = "Stopped after three consecutive source or provider failures"
                if not stop:
                    for _ in finished:
                        schedule()
                if started and (sum(statuses.values()) % 100 == 0 or not running):
                    print(
                        json.dumps(
                            {
                                "phase": "pilot" if pilot else "run",
                                "completed_this_phase": sum(statuses.values()),
                                "started_this_phase": started,
                                "statuses": dict(statuses),
                            }
                        ),
                        flush=True,
                    )
    return dict(
        status="STOPPED" if stop else "EXECUTED",
        started=started,
        stop_reason=stop,
        statuses=dict(statuses),
        blocked_profiles=sorted(profiles.keys() - admitted.keys()),
    )


def result_rows(rows, history, config):
    all_frames = bool((config or {}).get("frame_settings"))
    admitted = {
        a["profile"]
        for attempts in history.values()
        for a in attempts
        if a["phase"] == "pilot"
        and a.get("modality_admitted")
        and (not all_frames or a.get("frame_report"))
    }
    for row in rows:
        key = row["request_id"]
        attempts = history.get(key, [])
        final = attempts[-1] if attempts else {}
        status = (
            final["status"]
            if attempts
            else (
                "INSPECTION_ONLY"
                if key is None
                else "NOT_ATTEMPTED" if row["profile"] in admitted else "PILOT_REQUIRED"
            )
        )
        answer = final.get("normalized_answer")
        frame_verified = (
            not all_frames
            or final.get("phase") != "pilot"
            or bool(final.get("frame_report"))
        )
        fields = {
            k: row[k]
            for k in (
                "scope",
                "domain",
                "task",
                "input",
                "sample_id",
                "original_video_ID",
                "legacy_sample_id",
                "source",
                "source_manifest_sha256",
                "protocol",
                "source_version",
                "condition_key",
                "parameters",
                "mask",
                "background_id",
                "object_id",
                "object_ids_in_point_order",
                "truth",
                "accuracy_eligible",
                "choices",
                "prompt_sha256",
                "static_case_id",
                "model_input_sha256",
                "frame_indices",
                "request_id",
                "profile",
            )
        }
        yield dict(
            **fields,
            source_video_sha256=row["source_video"]["sha256"],
            requested_model=(config or {}).get("requested_model"),
            returned_model=final.get("returned_model"),
            attempts=len(attempts),
            attempt_sample_id=final.get("sample_id"),
            status=status,
            finish_reason=final.get("finish_reason"),
            raw_answer=final.get("raw_answer"),
            normalized_answer=answer,
            strict_format_valid=final.get("strict_format_valid", False),
            correct=(
                (answer == row["truth"])
                if answer is not None and row["accuracy_eligible"] and frame_verified
                else None
            ),
            http_status=final.get("http_status"),
            response_body_sha256=final.get("response_body_sha256"),
        )


def summarize(results):
    groups = defaultdict(list)
    for row in results:
        groups[row["profile"]].append(row)
    groups["all"] = [row for rows in groups.values() for row in rows]
    report = {}
    for name, rows in sorted(groups.items()):
        unique = {r["request_id"]: r for r in rows if r["request_id"] is not None}
        sections = {}
        for unit, values in (
            ("logical_rows", rows),
            ("independent_requests", list(unique.values())),
        ):
            planned = sum(r["accuracy_eligible"] for r in values)
            scorable = sum(r["correct"] is not None for r in values)
            correct = sum(r["correct"] is True for r in values)
            sections[unit] = dict(
                total=len(values),
                planned_accuracy_eligible=planned,
                observed_scorable=scorable,
                correct=correct,
                strict_format_valid=sum(r["strict_format_valid"] for r in values),
                correct_over_planned=correct / planned if planned else None,
                accuracy_among_scorable=correct / scorable if scorable else None,
                status_counts=dict(Counter(r["status"] for r in values)),
            )
        report[name] = sections
    return report


def export(out, archive):
    plan, rows = load_plan(out)
    config_path = out / "config.json"
    config = json.loads(config_path.read_bytes()) if config_path.exists() else None
    with (
        closing(open_db(out)) as db,
        TemporaryDirectory(dir=out, prefix=".return-") as scratch,
    ):
        history = records(db)
        stage = Path(scratch)
        normalized = list(result_rows(rows, history, config))
        summary = dict(
            schema=SCHEMA,
            source_plan_sha256=plan["source_plan_sha256"],
            logical_rows=len(rows),
            unique_requests=plan["unique_requests"],
            request_attempts=sum(map(len, history.values())),
            profiles=summarize(normalized),
        )
        with (stage / "attempts.jsonl").open("wb") as dest:
            for raw_receipt, raw_body in db.execute(
                "SELECT details, body FROM attempts ORDER BY request_id, attempt"
            ):
                receipt = json.loads(raw_receipt)
                receipt["response_body_b64"] = (
                    base64.b64encode(raw_body).decode("ascii")
                    if raw_body is not None
                    and not receipt.get("response_body_withheld")
                    else None
                )
                dest.write(release.encoded(receipt))
        with (stage / "results.jsonl").open("wb") as dest:
            for row in normalized:
                dest.write(release.encoded(row))
        (stage / "config.json").write_bytes(release.encoded(config))
        (stage / "summary.json").write_bytes(release.encoded(summary))
        paths = {
            name: (out if name in ("plan.json", "source-plan.jsonl") else stage) / name
            for name in RETURN_FILES
        }
        checksums = {name: file_sha(path) for name, path in paths.items()}
        archive.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=archive.parent, prefix=".jinwei-zip-") as temp:
            target = Path(temp) / "return.zip"
            with ZipFile(target, "w", ZIP_DEFLATED) as output:
                for name, path in paths.items():
                    output.write(path, name)
                output.writestr("SHA256SUMS.json", release.encoded(checksums))
            target.replace(archive)
    checksum = file_sha(archive)
    atomic(
        archive.with_suffix(archive.suffix + ".sha256"),
        f"{checksum}  {archive.name}\n".encode(),
    )
    return dict(
        status="EXPORTED",
        archive=str(archive),
        sha256=checksum,
        logical_rows=summary["logical_rows"],
        unique_requests=summary["unique_requests"],
        request_attempts=summary["request_attempts"],
    )


def verify(archive):
    checksum = file_sha(archive)
    sidecar = archive.with_suffix(archive.suffix + ".sha256")
    if sidecar.exists():
        release.require(
            sidecar.read_text().split()[0] == checksum, "Return ZIP SHA-256 mismatch"
        )
    with ZipFile(archive) as bundle:
        release.require(
            sorted(bundle.namelist()) == sorted((*RETURN_FILES, "SHA256SUMS.json")),
            "Unexpected archive members",
        )
        checksums = json.loads(bundle.read("SHA256SUMS.json"))
        release.require(set(checksums) == set(RETURN_FILES), "Missing member SHA-256")
        for name, expected in checksums.items():
            with bundle.open(name) as stream:
                release.require(
                    hashlib.file_digest(stream, "sha256").hexdigest() == expected,
                    f"Archive SHA mismatch: {name}",
                )
        plan = json.loads(bundle.read("plan.json"))
        config = json.loads(bundle.read("config.json"))
        rows = load_rows(bundle.read("source-plan.jsonl"))
        selected = release.unique_requests(rows)
        release.require(
            plan["schema"] == SCHEMA
            and release.known_public_release(plan["revision"], plan["index_sha256"])
            and plan["source_plan_sha256"] == checksums["source-plan.jsonl"]
            and plan["profiles"] == release.counts(rows)
            and plan["logical_rows"] == len(rows)
            and plan["unique_requests"] == len(selected),
            "Archive plan does not reconcile",
        )
        provider = PROVIDER
        if config:
            provider = config["provider"]
            release.require(
                config["source_plan_sha256"] == plan["source_plan_sha256"],
                "Run configuration differs from plan",
            )
            if provider in native_video_api.HOSTED:
                native_video_api.validate_config(config)
            elif provider == "vllm":
                native_video_api.vllm_endpoint(config["endpoint"])
            else:
                release.require(
                    provider == PROVIDER and config["endpoint"] == BASE_URL,
                    "Run configuration differs from provider",
                )
            if "frame_settings" in config:
                release.require(
                    provider in (PROVIDER, "vllm")
                    and config["frame_settings"] == native_video_api.FULL_FRAME_FIELDS,
                    "All-frame configuration differs",
                )
        history = defaultdict(list)
        changed_models = set()
        admitted_models = set()
        full_frame_profiles = set()
        full_frame_runs = set()
        frame_deployments = set()
        with bundle.open("attempts.jsonl") as stream:
            for line in stream:
                entry = json.loads(line)
                key = entry["request_id"]
                body = entry.pop("response_body_b64")
                release.require(
                    key in selected
                    and entry["attempt"] == len(history[key]) + 1
                    and entry["attempt"] <= 2
                    and entry["sample_id"] == selected[key]["sample_id"]
                    and entry["profile"] == selected[key]["profile"],
                    "Invalid attempt identity",
                )
                if "request_sha256" in entry:
                    source = selected[key]["media"]
                    verified = entry["verified_media"]
                    release.require(
                        bool(re.fullmatch(r"[0-9a-f]{64}", entry["request_sha256"]))
                        and len(source) == len(verified)
                        and all(
                            {k: v for k, v in actual.items() if k != "bytes"} == planned
                            and actual["bytes"] > 0
                            for planned, actual in zip(source, verified)
                        ),
                        "Sent media receipt differs from frozen input",
                    )
                if body is not None:
                    content = base64.b64decode(body, validate=True)
                    release.require(
                        entry["response_body_sha256"] == release.sha(content)
                        and entry["response_bytes"] == len(content),
                        "Raw response identity differs",
                    )
                    parsed = response_fields(
                        content,
                        entry["http_status"],
                        selected[key]["choices"],
                        None,
                        provider,
                    )
                    if entry["status"] == "MODEL_ID_CHANGED":
                        changed_models.add(parsed["returned_model"])
                        parsed.update(
                            status="MODEL_ID_CHANGED",
                            normalized_answer=None,
                            strict_format_valid=False,
                            modality_admitted=False,
                        )
                    release.require(
                        all(entry.get(k) == v for k, v in parsed.items()),
                        "Recorded answer differs from original response",
                    )
                else:
                    withheld = entry.get("response_body_withheld")
                    release.require(
                        "response_body_sha256" not in entry
                        or (
                            entry["status"] == "SENSITIVE_RESPONSE_WITHHELD"
                            and withheld in {"credential_echo", "media_echo"}
                        ),
                        "Original response body is missing",
                    )
                if entry["phase"] == "pilot" and entry.get("modality_admitted"):
                    admitted_models.add(entry["returned_model"])
                if "frame_report" in entry:
                    release.require(
                        bool((config or {}).get("frame_settings")),
                        "Unexpected server frame report",
                    )
                    validate_frame_report(entry["frame_report"], entry, config)
                    full_frame_profiles.add(entry["profile"])
                    frame_deployments.add(entry["frame_report"]["deployment"])
                if entry["phase"] == "run" and (config or {}).get("frame_settings"):
                    full_frame_runs.add(entry["profile"])
                history[key].append(entry)
        release.require(
            full_frame_runs <= full_frame_profiles and len(frame_deployments) <= 1,
            "All-frame batch evidence does not reconcile",
        )
        release.require(
            (not changed_models or bool(admitted_models))
            and not changed_models & admitted_models
            and len(admitted_models) <= 1,
            "Provider model identities do not reconcile",
        )
        release.require(
            not history or config is not None, "Attempts have no configuration"
        )
        expected = list(result_rows(rows, history, config))
        with bundle.open("results.jsonl") as stream:
            for recorded, computed in zip(stream, expected, strict=True):
                release.require(
                    json.loads(recorded) == computed,
                    "Normalized row differs from attempts",
                )
        summary = json.loads(bundle.read("summary.json"))
        release.require(
            summary["profiles"] == summarize(expected)
            and summary["logical_rows"] == len(rows)
            and summary["unique_requests"] == len(selected)
            and summary["request_attempts"] == sum(map(len, history.values())),
            "Result counts do not reconcile",
        )
    return dict(
        status="VERIFIED",
        sha256=checksum,
        logical_rows=len(rows),
        unique_requests=len(selected),
        request_attempts=summary["request_attempts"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("plan", "pilot", "run", "evaluate", "export", "verify")
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--metadata-root", type=Path, help="Offline root of the pinned public manifests"
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        help="Local media root instead of on-demand downloads",
    )
    parser.add_argument(
        "--private-index",
        help="Authorized generation-pinned GCS URL for the private T5 supplement",
    )
    parser.add_argument(
        "--private-metadata-root",
        type=Path,
        help="Local copy of the pinned private index and T5 manifests",
    )
    parser.add_argument("--allow-private-media-transfer", action="store_true")
    parser.add_argument("--model", default=os.environ.get("JINWEI_MODEL"))
    parser.add_argument(
        "--provider",
        default=PROVIDER,
        choices=(PROVIDER, "vllm", *native_video_api.HOSTED),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="Maximum new attempts in this invocation, including pilots",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scope", action="append", choices=release.SCOPES)
    parser.add_argument(
        "--task",
        action="append",
        choices=TASK_SCOPES,
        help="Select a task and domain, e.g. t1-classic; defaults to all-frame video",
    )
    parser.add_argument("--input", choices=("video", "static"))
    frames = parser.add_mutually_exclusive_group()
    frames.add_argument(
        "--all-frames",
        action="store_true",
        help="Request original frames and gate bulk on a server report",
    )
    frames.add_argument(
        "--sampled-video",
        action="store_true",
        help="Use the provider sampling path as a separate comparison",
    )
    parser.add_argument(
        "--frame-report",
        type=Path,
        help="Server decoder and processor evidence from the pilot",
    )
    parser.add_argument(
        "--retry",
        action="append",
        help="Explicit request ID to try once more; never re-asks a returned answer",
    )
    parser.add_argument(
        "--retry-uncertain",
        action="store_true",
        help="A previous request may already have reached the provider",
    )
    parser.add_argument(
        "--pilot-trial",
        action="append",
        help="Choose another previously unasked request ID for a blocked profile",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Authorize real calls from this shell"
    )
    args = parser.parse_args()
    if args.task:
        args.scope = list(
            dict.fromkeys(
                [*(args.scope or []), *(TASK_SCOPES[task] for task in args.task)]
            )
        )
        args.input = args.input or "video"
        args.all_frames = args.all_frames or (
            args.input == "video" and not args.sampled_video
        )
    try:
        if args.command == "verify":
            release.require(args.archive, "Set --archive for verification")
            result = verify(args.archive)
        else:
            release.require(args.out, "Set --out to a run directory on local storage")
            release.require(
                1 <= args.workers <= 16 and args.max_tokens > 0,
                "Invalid workers or token limit",
            )
            args.out.mkdir(parents=True, exist_ok=True, mode=0o700)
            with (args.out / ".run.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = command(args)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
    if result.get("stop_reason") or result.get("status") == "PARTIAL":
        raise SystemExit(2)


def command(args):
    archive = args.archive or args.out / "return.zip"
    if args.command == "export":
        return export(args.out, archive)
    if args.command != "plan":
        release.require(
            args.execute and args.max_requests > 0,
            "Set --execute and a positive --max-requests",
        )
    if args.command == "evaluate":
        release.require(not args.retry, "Use pilot or run for an explicit retry")
    if args.command != "pilot":
        release.require(not args.pilot_trial, "--pilot-trial applies to pilot only")
    private = None
    if args.private_index or args.private_metadata_root:
        private = release.PrivateDelta(args.private_index, args.private_metadata_root)
    if args.command == "plan" or (
        args.command == "evaluate" and not (args.out / "plan.json").exists()
    ):
        plan, rows = build_plan(args.out, args.metadata_root, private)
    else:
        plan, rows = load_plan(args.out)
        release.require(
            not private or plan["private_index_sha256"],
            "Use a new output directory for the private plan",
        )
    if args.command == "plan":
        return {
            "status": "PLANNED_NO_MODEL_CALLS",
            **plan,
            "selected_requests": len(selected_requests(rows, args)),
        }
    if args.command in ("pilot", "run"):
        return execution(
            args.out, plan, rows, args, private, pilot=args.command == "pilot"
        )
    pilot = execution(args.out, plan, rows, args, private, pilot=True)
    args.max_requests -= pilot["started"]
    run = None
    if args.max_requests > 0 and not pilot.get("stop_reason"):
        run = execution(args.out, plan, rows, args, private, pilot=False)
    packaged = export(args.out, archive)
    with closing(open_db(args.out)) as db:
        histories = records(db)
    selected = selected_requests(rows, args)
    finished = sum(
        1
        for key in selected
        if histories.get(key)
        and histories[key][-1]["status"]
        in {
            "ANSWERED",
            "FORMAT_OR_TOOL_FAILURE",
            "TRUNCATED",
        }
        and (
            not args.all_frames
            or histories[key][-1]["phase"] != "pilot"
            or histories[key][-1].get("frame_report")
        )
    )
    return {
        **packaged,
        "status": "COMPLETE" if finished == len(selected) else "PARTIAL",
        "selected_requests": len(selected),
        "finished_selected_requests": finished,
        "pilot": pilot,
        "run": run,
    }


if __name__ == "__main__":
    main()
