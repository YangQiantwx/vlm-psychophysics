"""Append-only qualified inputs, stable row identities and finite input copying."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from .manifest import atomic_json, canonical, digest, file_sha, selected_rows, state_key


INPUT_BINDING = {
    "schema_version": 2,
    "manifest": "frozen complete manifest",
    "ready": "immutable per-row receipt; append-only publication",
}


def inside(root, relative):
    relative = Path(relative)
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not path.is_relative_to(root)
    ):
        raise ValueError("Input path escapes its explicit asset root")
    return path


def representatives(manifest):
    unique = {}
    for row in manifest["rows"]:
        unique.setdefault(state_key(row), row)
    if len(unique) != 128:
        raise ValueError("Expected exactly128 frozen source states")
    return list(unique.values())


def generation_order(manifest, shard_count=3):
    """Qualification and calibration first, then distribute the initial buffer."""
    priority = []
    shards = [
        selected_rows(manifest, index, shard_count) for index in range(shard_count)
    ]
    for shard in shards:
        priority.extend(
            next(row for row in shard if row["task"] == task) for task in ("T2", "T3")
        )
    priority.extend(representatives(manifest))
    for index in range(max(map(len, shards))):
        priority.extend(shard[index] for shard in shards if index < len(shard))
    seen = set()
    return [
        row
        for row in priority
        if not (row["sample_id"] in seen or seen.add(row["sample_id"]))
    ]


@contextlib.contextmanager
def publication_lock(path, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_name("." + path.name + ".publication.lock").open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def read_records(path):
    """Only newline-terminated records are published; readers never repair."""
    path = Path(path)
    data = path.read_bytes() if path.exists() else b""
    lines = data.split(b"\n")
    tail = lines.pop()
    records = {}
    for line in lines:
        if not line:
            raise ValueError("Blank/malformed complete ready journal line")
        item = json.loads(line)
        sid = item["sample_id"]
        if sid in records:
            raise ValueError("Repeated or changed ready sample identity: " + sid)
        records[sid] = item
    return records, bool(tail), data


def validate_record(row, item):
    if (
        item["sample_id"] != row["sample_id"]
        or item["source_parameters"] != row["source_parameters"]
    ):
        raise ValueError("Ready source parameters differ from the immutable manifest")
    for field in ("video_sha256", "stream_sha256"):
        if not isinstance(item.get(field), str) or not re.fullmatch(
            "[0-9a-f]{64}", item[field]
        ):
            raise ValueError("Invalid ready SHA256: " + field)
    if (
        item.get("frames") != 180
        or item.get("master_bits_per_channel") != 16
        or item.get("max_decoded_code_error") != 0
        or not 0 <= item.get("max_PTS_error_s", float("inf")) <= 0.001
        or Path(item["video_relative"]).suffix != ".mkv"
    ):
        raise ValueError("Ready lacks the qualified full RGB16 video receipt")


def fingerprint(path):
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


class ReadyCatalog:
    """A growing subset of one fixed panel; unchanged media are not rehashed."""

    def __init__(
        self, manifest, ready_path, asset_root, manifest_sha256, allow_failed=False
    ):
        self.manifest, self.path, self.root = (
            manifest,
            Path(ready_path),
            Path(asset_root),
        )
        self.manifest_sha256 = manifest_sha256
        self.rows = {row["sample_id"]: row for row in manifest["rows"]}
        self.known, self.verified = {}, {}
        self.complete, self.tail = False, False
        self.allow_failed = allow_failed

    def refresh(self):
        with publication_lock(self.path):
            records, tail, _ = read_records(self.path)
            marker = self.path.parent / "generation.json"
            generation = json.loads(marker.read_text()) if marker.exists() else {}
        if set(records) - set(self.rows):
            raise ValueError("Ready contains IDs outside the immutable manifest")
        if set(self.known) - set(records):
            raise ValueError("Previously published ready IDs disappeared")
        for sid, item in records.items():
            if sid in self.known and self.known[sid] != item:
                raise ValueError("Previously observed ready identity changed: " + sid)
            validate_record(self.rows[sid], item)
            path = inside(self.root, item["video_relative"])
            sidecar = path.with_suffix(".ready.json")
            if not path.is_file() or not sidecar.is_file():
                raise ValueError("Published ready video/sidecar is missing: " + sid)
            stamp = (fingerprint(path), fingerprint(sidecar))
            if self.verified.get(sid) != stamp:
                if (
                    json.loads(sidecar.read_text()) != item
                    or file_sha(path) != item["video_sha256"]
                ):
                    raise ValueError(
                        "Published input media or sidecar differs from its ready identity: "
                        + sid
                    )
                self.verified[sid] = stamp
        self.complete = generation.get("status") == "COMPLETE"
        if generation and (
            generation.get("protocol_id") != self.manifest["protocol_id"]
            or generation.get("manifest_sha256") != self.manifest_sha256
        ):
            raise ValueError(
                "Generation completion/source identity differs from frozen manifest"
            )
        if self.complete and (
            tail
            or set(records) != set(self.rows)
            or generation.get("videos") != len(self.rows)
        ):
            raise ValueError("Generation COMPLETE has missing or partial ready records")
        if generation.get("status") == "FAILED" and not self.allow_failed:
            raise RuntimeError(
                "Input producer FAILED: "
                + generation.get("error", "unknown source failure")
            )
        self.known, self.tail = records, tail
        return dict(records)


def append_ready(manifest, root, item, ready_path=None, verified=False):
    """Publish only after the exact media and atomic sidecar are durable."""
    root = Path(root)
    path = Path(ready_path) if ready_path else root / "ready.jsonl"
    row = next(
        (row for row in manifest["rows"] if row["sample_id"] == item["sample_id"]), None
    )
    if row is None:
        raise ValueError("Unknown ready scientific row")
    validate_record(row, item)
    video = inside(root, item["video_relative"])
    if (
        not video.is_file()
        or json.loads(video.with_suffix(".ready.json").read_text()) != item
    ):
        raise ValueError(
            "Ready publication requires its complete media and matching sidecar"
        )
    if not verified and file_sha(video) != item["video_sha256"]:
        raise ValueError("Ready publication video SHA mismatch")
    with publication_lock(path, exclusive=True):
        records, tail, data = read_records(path)
        if tail:
            backup = path.with_name(
                "ready.torn." + hashlib.sha256(data).hexdigest()[:16] + ".bin"
            )
            if not backup.exists():
                with backup.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
            # Only an unpublished tail is removed; durable sidecars recover it.
            temp = path.with_name(path.name + f".{os.getpid()}.recovered.partial")
            with temp.open("xb") as stream:
                stream.write(
                    b"".join(canonical(row) + b"\n" for row in records.values())
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
        sid = item["sample_id"]
        if sid in records:
            if records[sid] != item:
                raise ValueError("Refuse to change a published ready identity: " + sid)
            return
        created = not path.exists()
        with path.open("ab") as stream:
            stream.write(canonical(item) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        if created or tail:
            fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)


def finish_generation(manifest, root, ready_path=None, **metadata):
    root = Path(root)
    path = Path(ready_path) if ready_path else root / "ready.jsonl"
    with publication_lock(path, exclusive=True):
        records, tail, _ = read_records(path)
        if tail or set(records) != {row["sample_id"] for row in manifest["rows"]}:
            raise ValueError(
                "Cannot mark generation complete with missing/partial ready rows"
            )
        receipt = {
            **metadata,
            "status": "COMPLETE",
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": file_sha(root / "manifest.json"),
            "videos": len(records),
            "unique_source_states": 128,
            "new_VLM_calls": 0,
        }
        atomic_json(path.parent / "generation.json", receipt)
    return receipt


def start_ready(catalog, selected, ready, minimum=100):
    if minimum < 1:
        raise ValueError("The initial per-shard ready buffer must be positive")
    first = [
        next(row for row in selected if row["task"] == task)["sample_id"]
        for task in ("T2", "T3")
    ]
    count = sum(row["sample_id"] in ready for row in selected)
    return all(sid in ready for sid in first) and (count >= minimum or catalog.complete)


def replica_initialize(root, manifest_base64, expected_sha):
    """Install only an exact small manifest; never replace a different one."""
    import base64
    import sys
    from .manifest import validate_manifest

    data = (
        base64.b64decode(manifest_base64)
        if manifest_base64 is not None
        else sys.stdin.buffer.read()
    )
    manifest = validate_manifest(json.loads(data))
    if hashlib.sha256(data).hexdigest() != expected_sha:
        raise ValueError("Replica manifest SHA mismatch")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    with publication_lock(root / "ready.jsonl", exclusive=True):
        if path.exists():
            if path.read_bytes() != data:
                raise ValueError("Refuse to replace a different replica manifest")
        else:
            temp = path.with_name("manifest.sync.partial")
            with temp.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, path)
            fd = os.open(root, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    marker = root / "generation.json"
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    if previous.get("status") != "COMPLETE":
        replica_status(root, "RUNNING")
    return {"protocol_id": manifest["protocol_id"], "manifest_sha256": expected_sha}


def replica_snapshot(root, expected_sha):
    """Small source inventory; file SHA is checked by the destination receiver.

    Repeated inventories stat published media instead of rereading every video.
    The consumer independently validates its first observation and each decode.
    """
    from .manifest import validate_manifest

    root = Path(root)
    if file_sha(root / "manifest.json") != expected_sha:
        raise ValueError("Replica source manifest differs")
    manifest = validate_manifest(json.loads((root / "manifest.json").read_text()))
    rows = {row["sample_id"]: row for row in manifest["rows"]}
    with publication_lock(root / "ready.jsonl"):
        records, tail, _ = read_records(root / "ready.jsonl")
        marker = root / "generation.json"
        generation = json.loads(marker.read_text()) if marker.exists() else {}
    stats = {}
    for sid, item in records.items():
        if sid not in rows:
            raise ValueError("Replica ready has an unknown row")
        validate_record(rows[sid], item)
        path = inside(root, item["video_relative"])
        if json.loads(path.with_suffix(".ready.json").read_text()) != item:
            raise ValueError("Replica source sidecar changed")
        stats[sid] = list(fingerprint(path))
    complete = generation.get("status") == "COMPLETE"
    if generation and (
        generation.get("protocol_id") != manifest["protocol_id"]
        or generation.get("manifest_sha256") != expected_sha
    ):
        raise ValueError("Replica generation marker differs from manifest")
    if complete and (
        tail or set(records) != set(rows) or generation.get("videos") != len(rows)
    ):
        raise ValueError("Replica generation COMPLETE has missing/partial rows")
    if generation.get("status") == "FAILED":
        raise RuntimeError(
            "Input producer FAILED: "
            + generation.get("error", "unknown source failure")
        )
    return {
        "records": records,
        "stats": stats,
        "complete": complete,
        "partial_tail": tail,
    }


def send_video(root, item):
    import shutil
    import sys

    with inside(root, item["video_relative"]).open("rb") as stream:
        shutil.copyfileobj(stream, sys.stdout.buffer, length=1024 * 1024)


def receive_video(root, item, expected_bytes):
    """Bounded-memory stdin receiver: SHA, fsync, atomic video, sidecar, ready."""
    import sys
    from .manifest import validate_manifest

    root = Path(root)
    manifest = validate_manifest(json.loads((root / "manifest.json").read_text()))
    row = next(row for row in manifest["rows"] if row["sample_id"] == item["sample_id"])
    validate_record(row, item)
    video = inside(root, item["video_relative"])
    video.parent.mkdir(parents=True, exist_ok=True)
    partial = video.with_name(video.name + f".{os.getpid()}.sync.partial")
    with (root / ".input-sync.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        sha, size = hashlib.sha256(), 0
        with partial.open("xb") as stream:
            for block in iter(lambda: sys.stdin.buffer.read(1024 * 1024), b""):
                size += len(block)
                if size > expected_bytes:
                    raise ValueError("Replica source exceeded its exact video size")
                sha.update(block)
                stream.write(block)
            if size != expected_bytes or sha.hexdigest() != item["video_sha256"]:
                raise ValueError(
                    "Incomplete/corrupt replica video; no ready publication"
                )
            stream.flush()
            os.fsync(stream.fileno())
        if video.exists():
            if file_sha(video) != item["video_sha256"]:
                raise ValueError("Refuse to replace a different existing replica video")
            partial.unlink()
        else:
            # The input-sync lock serializes the only receiver writers.
            os.link(partial, video)
            partial.unlink()
        sidecar = video.with_suffix(".ready.json")
        if sidecar.exists() and json.loads(sidecar.read_text()) != item:
            raise ValueError("Refuse to replace an existing replica ready identity")
        atomic_json(sidecar, item)
        append_ready(manifest, root, item, verified=True)
    return {"sample_id": item["sample_id"], "verified_bytes": size, "status": "READY"}


def replica_finish(root):
    from .manifest import validate_manifest

    root = Path(root)
    return finish_generation(
        validate_manifest(json.loads((root / "manifest.json").read_text())), root
    )


def replica_status(root, status, error=None):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    receipt = {
        "status": status,
        "protocol_id": manifest["protocol_id"],
        "manifest_sha256": file_sha(root / "manifest.json"),
        "new_VLM_calls": 0,
    }
    if error is not None:
        receipt["error"] = error
    with publication_lock(root / "ready.jsonl", exclusive=True):
        if status == "FAILED":
            atomic_json(root / "generation.failure.json", receipt)
        atomic_json(root / "generation.json", receipt)
    return receipt


def sync_ready(
    manifest_path,
    source_root,
    destination_root,
    source_ssh=None,
    destination_ssh=None,
    source_python=None,
    destination_python=None,
    source_code_root=None,
    destination_code_root=None,
    poll_seconds=10,
    once=False,
):
    """One ordinary process; local or SSH endpoints, streamed with no local copy.

    A remote endpoint takes a complete existing SSH invocation (including host),
    its existing Python and deployed code root. No peer SSH trust is assumed.
    """
    import base64
    import shlex
    import subprocess
    import sys
    import time
    from .manifest import validate_manifest

    if poll_seconds < 1:
        raise ValueError("Input synchronization poll interval must be positive")
    data = Path(manifest_path).read_bytes()
    manifest = validate_manifest(json.loads(data))
    expected_sha = hashlib.sha256(data).hexdigest()

    def endpoint(ssh, python, code):
        if ssh and (not python or not code):
            raise ValueError(
                "Remote inputs require their existing Python and deployed code root"
            )
        prefix = shlex.split(ssh) if isinstance(ssh, str) else list(ssh or [])
        python = python or sys.executable
        code = str(code or Path(__file__).resolve().parent.parent)

        def command(function, args):
            script = (
                "import sys,json;sys.path.insert(0," + repr(code) + ");"
                "from classic_precision import ready;result=ready."
                + function
                + "(*"
                + repr(args)
                + ");"
                "print(json.dumps(result))"
                if function != "send_video"
                else "import sys;sys.path.insert(0,"
                + repr(code)
                + ");from classic_precision import ready;"
                "ready.send_video(*" + repr(args) + ")"
            )
            argv = [python, "-B", "-c", script]
            return prefix + [shlex.join(argv)] if prefix else argv

        return command

    source = endpoint(source_ssh, source_python, source_code_root)
    destination = endpoint(destination_ssh, destination_python, destination_code_root)
    source_root, destination_root = str(source_root), str(destination_root)

    def call(command, function, args, data=None):
        result = subprocess.run(
            command(function, args),
            check=True,
            capture_output=True,
            input=data,
            timeout=60,
        )
        return json.loads(result.stdout)

    call(
        destination, "replica_initialize", [destination_root, None, expected_sha], data
    )
    observed = {"source": {}, "destination": {}}
    try:
        while True:
            inventories = {
                "source": call(source, "replica_snapshot", [source_root, expected_sha]),
                "destination": call(
                    destination, "replica_snapshot", [destination_root, expected_sha]
                ),
            }
            for name, inventory in inventories.items():
                current = {
                    sid: (item, inventory["stats"][sid])
                    for sid, item in inventory["records"].items()
                }
                if any(
                    current.get(sid) != binding
                    for sid, binding in observed[name].items()
                ):
                    raise ValueError(
                        "Previously observed synchronized ready identity/media changed: "
                        + name
                    )
                observed[name] = current
            src, dest = inventories["source"], inventories["destination"]
            for sid, item in dest["records"].items():
                if src["records"].get(sid) != item:
                    raise ValueError(
                        "Replica has an input identity absent/different at source"
                    )
            copied = 0
            for row in generation_order(manifest):
                sid = row["sample_id"]
                if sid not in src["records"] or sid in dest["records"]:
                    continue
                item = src["records"][sid]
                sender = subprocess.Popen(
                    source("send_video", [source_root, item]), stdout=subprocess.PIPE
                )
                receiver = None
                try:
                    receiver = subprocess.Popen(
                        destination(
                            "receive_video",
                            [destination_root, item, src["stats"][sid][2]],
                        ),
                        stdin=sender.stdout,
                        stdout=subprocess.PIPE,
                    )
                    sender.stdout.close()
                    output, _ = receiver.communicate(timeout=300)
                    if receiver.returncode or sender.wait(timeout=30):
                        raise RuntimeError(
                            "Input transfer failed; partial destination was not published"
                        )
                    if json.loads(output)["status"] != "READY":
                        raise ValueError("Replica publication did not return READY")
                finally:
                    for process in (receiver, sender):
                        if process is not None and process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                dest["records"][sid] = item
                copied += 1
            if src["complete"] and len(dest["records"]) == len(manifest["rows"]):
                return call(destination, "replica_finish", [destination_root])
            status = {
                "status": "SYNCING",
                "source_ready": len(src["records"]),
                "destination_ready": len(dest["records"]),
                "copied_this_round": copied,
                "expected": len(manifest["rows"]),
                "new_VLM_calls": 0,
            }
            if once:
                return status
            time.sleep(poll_seconds)
    except BaseException as error:
        try:
            call(
                destination,
                "replica_status",
                [
                    destination_root,
                    "FAILED",
                    type(error).__name__ + ": " + str(error)[:1000],
                ],
            )
        except Exception:
            # A disconnected destination cannot receive a failure marker. The
            # controller still exits nonzero; its normal process status is the
            # evidence until connectivity returns.
            pass
        raise
