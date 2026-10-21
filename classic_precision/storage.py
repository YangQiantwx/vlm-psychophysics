"""Durable finite-run journals; durable outputs are never generated twice.

Raw outputs are the authoritative terminal records. Results are a rebuildable
view. Locks protect model journals and claim immutable row positions, so disjoint
shards can run together while overlapping shards cannot call the same row.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import socket
import time
import uuid
from pathlib import Path


MAX_ATTEMPTS = 2


class PersistenceError(RuntimeError):
    """Stop generation after any uncertain journal write."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path):
    path = Path(path)
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _sync_directory(created.parent)


def atomic_json(path, value):
    path = Path(path)
    _mkdir_durable(path.parent)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _append(path, value):
    path = Path(path)
    existed = path.exists()
    encoded = (canonical_json(value) + "\n").encode("utf-8")
    try:
        with Path(path).open("ab", buffering=0) as stream:
            view = memoryview(encoded)
            while view:
                written = stream.write(view)
                if not written:
                    raise OSError("Journal write made no progress")
                view = view[written:]
            os.fsync(stream.fileno())
        if not existed:
            _sync_directory(path.parent)
    except BaseException as error:
        raise PersistenceError(f"Journal append uncertain: {Path(path).name}") from error


def _read_journal(path):
    """Read complete JSON objects; preserve and repair only a torn final line.

    An invalid interior line is corruption, never silently skipped. A valid final
    object lacking its newline is already durable output and gets a delimiter.
    The caller holds the model state lock throughout recovery.
    """
    path = Path(path)
    if not path.exists():
        return []
    contents = path.read_bytes()
    pieces = contents.splitlines(keepends=True)
    records, offset = [], 0
    for index, piece in enumerate(pieces):
        try:
            record = json.loads(piece)
            if not isinstance(record, dict):
                raise ValueError("Journal entries must be objects")
        except (ValueError, UnicodeError) as error:
            if index != len(pieces) - 1 or piece.endswith(b"\n"):
                raise PersistenceError(f"Corrupt interior journal line: {path.name}:{index + 1}") from error
            tail = path.with_name(path.name + ".torn-" + uuid.uuid4().hex + ".bin")
            with tail.open("xb") as stream:
                stream.write(piece)
                stream.flush()
                os.fsync(stream.fileno())
            with path.open("r+b") as stream:
                stream.truncate(offset)
                stream.flush()
                os.fsync(stream.fileno())
            _sync_directory(path.parent)
            break
        records.append(record)
        offset += len(piece)
        if index == len(pieces) - 1 and not piece.endswith(b"\n"):
            with path.open("ab", buffering=0) as stream:
                stream.write(b"\n")
                os.fsync(stream.fileno())
    # A killed process can leave a complete line in the page cache before its
    # own fsync. Promote observed complete records before rebuilding results.
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    _sync_directory(path.parent)
    return records


def owner_identity():
    proc = Path("/proc/self/stat")
    start = proc.read_text().rsplit(")", 1)[1].split()[19] if proc.exists() else None
    boot = Path("/proc/sys/kernel/random/boot_id")
    return {"pid": os.getpid(), "process_start_ticks": start,
            "boot_id": boot.read_text().strip() if boot.exists() else None,
            "host": socket.gethostname(), "started_unix": time.time()}


def classify_output(record):
    raw = record.get("raw_answer", "")
    reason = record.get("finish_reason")
    # Length/abort outputs remain in the denominator but are never parsed into
    # a semantic answer, even if their truncated text happens to be a label.
    label = raw.strip() if isinstance(raw, str) and reason == "stop" else None
    valid = label in ("LEFT", "RIGHT")
    exact = raw in ("LEFT", "RIGHT") if isinstance(raw, str) else False
    return {"sample_id": record["sample_id"], "attempt_id": record["attempt_id"],
            "status": "completed", "answer": label if valid else None,
            "format_valid": valid, "finish_reason": reason,
            "raw_exact_choice": exact, "strict_format_valid": exact,
            "normalization": "strip_boundary_whitespace_only",
            "outcome": "length" if reason == "length" else ("valid" if valid else "invalid"),
            "truth": record.get("truth"),
            "correct": label == record.get("truth") if valid else False,
            "raw_record_sha256": digest(record)}


class RunStore:
    """One finite model/shard lease with model-wide durable output recovery."""

    def __init__(self, output_dir, run_identity, model_key, shard_id,
                 allowed_row_ids, all_row_ids=None):
        for value in (model_key, shard_id):
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise ValueError("Model and shard identifiers must be portable names")
        self.directory = Path(output_dir) / model_key
        _mkdir_durable(self.directory)
        self.model_key, self.shard_id = model_key, shard_id
        self.row_ids = tuple(allowed_row_ids)
        self.all_row_ids = tuple(all_row_ids or self.row_ids)
        self._row_id_set = set(self.row_ids)
        self._all_row_id_set = set(self.all_row_ids)
        self.identity = {**run_identity, "storage_all_row_ids_sha256": digest(self.all_row_ids)}
        if len(set(self.row_ids)) != len(self.row_ids) or len(set(self.all_row_ids)) != len(self.all_row_ids):
            raise ValueError("Duplicate immutable row IDs")
        if not set(self.row_ids) <= set(self.all_row_ids):
            raise ValueError("Shard contains an unknown row ID")
        self.owner = owner_identity()
        self._handles = []
        self._active = {}
        self._active_numbers = {}

    @contextlib.contextmanager
    def _state_lock(self):
        with (self.directory / ".model-state.lock").open("a+b") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield
            fcntl.flock(stream, fcntl.LOCK_UN)

    def __enter__(self):
        try:
            shard_lock = (self.directory / (".shard-" + self.shard_id + ".lock")).open("a+b")
            self._handles.append(shard_lock)
            fcntl.flock(shard_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            claims = (self.directory / ".row-claims.lock").open("a+b")
            self._handles.append(claims)
            positions = {name: i for i, name in enumerate(self.all_row_ids)}
            for row_id in self.row_ids:
                fcntl.lockf(claims, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, positions[row_id], os.SEEK_SET)
            with self._state_lock():
                path = self.directory / "identity.json"
                if path.exists():
                    if json.loads(path.read_text()) != self.identity:
                        raise ValueError("Run/model identity differs; use a new output directory")
                else:
                    atomic_json(path, self.identity)
                shard_path = self.directory / ("shard-" + self.shard_id + ".json")
                selected = {"row_ids": list(self.row_ids), "all_row_ids_sha256": digest(self.all_row_ids)}
                if shard_path.exists() and json.loads(shard_path.read_text()) != selected:
                    raise ValueError("Shard membership changed; choose a new shard identifier")
                if not shard_path.exists():
                    atomic_json(shard_path, selected)
                self._recover_locked()
                _append(self.directory / "events.jsonl", {"event": "WORKER_START", "owner": self.owner,
                        "shard_id": self.shard_id, "time_unix": time.time()})
            return self
        except BaseException:
            self.close()
            raise

    def _recover_locked(self):
        raw = _read_journal(self.directory / "outputs.jsonl")
        results = _read_journal(self.directory / "results.jsonl")
        events = _read_journal(self.directory / "events.jsonl")
        outputs = {}
        for record in raw:
            row_id = record["sample_id"]
            if row_id not in self._all_row_id_set or row_id in outputs:
                raise PersistenceError("Unknown or duplicate durable raw output ID")
            outputs[row_id] = record
        result_ids = {}
        for result in results:
            row_id = result["sample_id"]
            if row_id in result_ids or row_id not in outputs or result != classify_output(outputs[row_id]):
                raise PersistenceError("Result does not match authoritative raw output")
            result_ids[row_id] = result
        for row_id, record in outputs.items():
            if row_id not in result_ids:
                _append(self.directory / "results.jsonl", classify_output(record))
        self.outputs = outputs
        self.events = events

    def _attempt_state_locked(self):
        self.events = _read_journal(self.directory / "events.jsonl")
        attempts, latest = {}, {}
        for event in self.events:
            row_id = event.get("sample_id")
            if row_id not in self._all_row_id_set:
                continue
            if event.get("event") == "INFLIGHT":
                attempts.setdefault(row_id, []).append(event)
            if event.get("event") in ("INFLIGHT", "FAILED", "INTERRUPTED"):
                latest[row_id] = event["event"]
        if any(len(history) > MAX_ATTEMPTS for history in attempts.values()):
            raise PersistenceError("Persisted attempt count exceeds the fixed two-attempt budget")
        return attempts, latest

    def _exhausted_locked(self, row_id, history):
        if any(event.get("event") == "EXHAUSTED" and event.get("sample_id") == row_id
               for event in self.events):
            return
        event = {"event": "EXHAUSTED", "sample_id": row_id,
                 "attempt_id": history[-1]["attempt_id"], "attempt_count": len(history),
                 "max_attempts": MAX_ATTEMPTS, "shard_id": self.shard_id, "time_unix": time.time(),
                 "reason": "No durable complete raw output after the initial attempt and one retry"}
        _append(self.directory / "events.jsonl", event)
        self.events.append(event)

    def pending_ids(self, retry_failed=True, retry_interrupted=True):
        with self._state_lock():
            self._recover_locked()
            attempts, latest = self._attempt_state_locked()
            for row_id in self.row_ids:
                if row_id not in self.outputs and row_id not in self._active and len(attempts.get(row_id, [])) >= MAX_ATTEMPTS:
                    self._exhausted_locked(row_id, attempts[row_id])
            return [row_id for row_id in self.row_ids if row_id not in self.outputs and row_id not in self._active
                    and len(attempts.get(row_id, [])) < MAX_ATTEMPTS
                    and (retry_failed or latest.get(row_id) != "FAILED")
                    and (retry_interrupted or latest.get(row_id) not in ("INFLIGHT", "INTERRUPTED"))]

    def event(self, event, **metadata):
        with self._state_lock():
            _append(self.directory / "events.jsonl", {"event": event, "shard_id": self.shard_id,
                    "time_unix": time.time(), **metadata})

    def begin(self, row_id, row_identity):
        if row_id not in self._row_id_set:
            raise ValueError("Row outside this shard")
        with self._state_lock():
            if row_id in self.outputs:
                raise PersistenceError("Refuse to generate a durable terminal row again")
            if row_id in self._active:
                raise PersistenceError("A row already has an active attempt in this worker")
            attempts, _ = self._attempt_state_locked()
            history = attempts.get(row_id, [])
            if any(item.get("row_identity") != row_identity for item in history):
                raise ValueError("Retry input identity differs from its persisted original attempt")
            if len(history) >= MAX_ATTEMPTS:
                self._exhausted_locked(row_id, history)
                raise PersistenceError("The initial attempt and one retry are exhausted; refuse a third call")
            attempt = uuid.uuid4().hex
            event = {"event": "INFLIGHT", "sample_id": row_id,
                    "row_identity": row_identity, "attempt_id": attempt, "owner": self.owner,
                    "attempt_number": len(history) + 1, "max_attempts": MAX_ATTEMPTS,
                    "retry_of_attempt_id": history[-1]["attempt_id"] if history else None,
                    "shard_id": self.shard_id, "time_unix": time.time()}
            _append(self.directory / "events.jsonl", event)
            self.events.append(event)
            self._active[row_id] = attempt
            self._active_numbers[row_id] = len(history) + 1
            return attempt

    def commit_output(self, row_id, payload):
        """Commit raw first. Failure here must stop all further generation."""
        attempt = self._active[row_id]
        record = {**payload, "sample_id": row_id, "attempt_id": attempt,
                  "attempt_number": self._active_numbers[row_id]}
        if not isinstance(record.get("output_token_ids"), list) or "raw_answer" not in record:
            raise ValueError("Raw text and token IDs are mandatory")
        with self._state_lock():
            if row_id in self.outputs:
                raise PersistenceError("Duplicate raw output refused")
            _append(self.directory / "outputs.jsonl", record)
            self.outputs[row_id] = record
            # Any failure after the first fsync is recoverable without a model call.
            _append(self.directory / "results.jsonl", classify_output(record))
        self._active.pop(row_id, None)
        self._active_numbers.pop(row_id, None)
        return classify_output(record)

    def fail(self, row_id, error, interrupted=False, elapsed_seconds=None):
        attempt = self._active.pop(row_id, None)
        self._active_numbers.pop(row_id, None)
        with self._state_lock():
            event = {"event": "INTERRUPTED" if interrupted else "FAILED", "sample_id": row_id,
                     "attempt_id": attempt, "shard_id": self.shard_id, "time_unix": time.time(),
                     "error_type": type(error).__name__, "error": str(error),
                     "elapsed_seconds": elapsed_seconds}
            _append(self.directory / "events.jsonl", event)
            attempts, _ = self._attempt_state_locked()
            if row_id not in self.outputs and len(attempts.get(row_id, [])) >= MAX_ATTEMPTS:
                self._exhausted_locked(row_id, attempts[row_id])

    def summary(self, retry_failed=True, retry_interrupted=True):
        pending = self.pending_ids(retry_failed=retry_failed, retry_interrupted=retry_interrupted)
        terminal = [x for x in self.row_ids if x in self.outputs]
        failed = {}
        for event in self.events:
            if event.get("sample_id") in self.row_ids:
                if event.get("event") in ("INFLIGHT", "FAILED", "INTERRUPTED"):
                    failed[event["sample_id"]] = event.get("event")
        attempts = dict.fromkeys(self.row_ids, 0)
        for event in self.events:
            if event.get("event") == "INFLIGHT" and event.get("sample_id") in self._row_id_set:
                attempts[event["sample_id"]] += 1
        failures = [x for x in self.row_ids if x not in self.outputs and failed.get(x) == "FAILED"]
        held = [x for x in self.row_ids if x not in self.outputs and x not in pending]
        exhausted = [x for x in self.row_ids if x not in self.outputs and attempts[x] >= MAX_ATTEMPTS
                     and x not in self._active]
        return {"expected": len(self.row_ids), "completed_outputs": len(terminal),
                "failed_ids": failures, "held_ids": held, "pending_ids": pending,
                "exhausted_ids": exhausted, "attempt_counts": attempts, "max_attempts": MAX_ATTEMPTS,
                "status": "COMPLETE" if len(terminal) == len(self.row_ids) else "INCOMPLETE"}

    def close(self):
        for handle in reversed(self._handles):
            handle.close()
        self._handles = []

    def __exit__(self, error_type, error, traceback):
        try:
            if self._active and not isinstance(error, PersistenceError):
                for row_id in list(self._active):
                    self.fail(row_id, error or RuntimeError("Worker left with an inflight row"), interrupted=True)
        finally:
            self.close()
