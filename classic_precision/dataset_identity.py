"""Versioned source identities and opt-in compatibility views.

Names are locators, never hash inputs. Frozen manifests, queues, original answers
and generator rows keep their original IDs; exported views carry both IDs.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


IDENTITY_SCHEMA = "sample-identity-v1"
REGISTRY_SCHEMA = "dataset-identity-registry-v1"
DEFAULT_REVISION = "dataset-naming-v1"


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def identity_digest(domain, task, video_sha256):
    if domain not in ("classic", "natural") or task not in ("T1", "T2", "T3"):
        raise ValueError("Unknown identity domain/task")
    if not isinstance(video_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", video_sha256):
        raise ValueError("A full video SHA256 is required")
    return hashlib.sha256(canonical_json({"schema": IDENTITY_SCHEMA, "domain": domain,
        "task": task, "video_sha256": video_sha256}).encode()).hexdigest()


def source_sample_id(row):
    """Stable frozen key for selection, random seeds, order and result joins."""
    return row.get("legacy_sample_id", row["sample_id"])


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("Expected a nonempty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError("Media paths must stay relative to the dataset root")
    return value


@dataclass(frozen=True)
class IdentityBinding:
    dataset_revision: str
    domain: str
    task: str
    canonical_id: str
    legacy_sample_id: str
    path: str
    legacy_path: str
    video_sha256: str
    full_identity_sha256: str = ""
    alias_paths: tuple[str, ...] = ()


class IdentityRegistry:
    """Resolve IDs/full relative paths within an explicit dataset revision.

    Parent IDs of real source replacements are deliberately not aliases. A
    caller must select their historical registry to resolve historical media.
    """

    def __init__(self, bindings):
        self.bindings = tuple(bindings)
        self._lookup, self._canonical, self._unversioned = {}, {}, {}
        media = set()
        for binding in self.bindings:
            expected = identity_digest(binding.domain, binding.task, binding.video_sha256)
            relative_path(binding.path)
            relative_path(binding.legacy_path)
            for alias_path in binding.alias_paths:
                relative_path(alias_path)
            if not binding.dataset_revision or not binding.canonical_id or not binding.legacy_sample_id:
                raise ValueError("Incomplete versioned identity")
            changed = binding.canonical_id != binding.legacy_sample_id
            if changed and (binding.domain, binding.task) != ("classic", "T1"):
                raise ValueError("Only classic T1 IDs change in identity v1")
            if changed or binding.full_identity_sha256:
                if binding.full_identity_sha256 != expected:
                    raise ValueError("Full source identity digest differs")
                if changed and binding.canonical_id != "t1_" + expected[:20]:
                    raise ValueError("Canonical T1 ID differs from its full digest")
            key = (binding.dataset_revision, binding.domain, binding.canonical_id)
            if key in self._canonical:
                raise ValueError("Duplicate canonical ID or short-digest collision")
            self._canonical[key] = binding
            media_key = (binding.dataset_revision, binding.domain, binding.video_sha256)
            if media_key in media:
                raise ValueError("Duplicate source video SHA in one dataset revision")
            media.add(media_key)
            for alias in {binding.canonical_id, binding.legacy_sample_id, binding.path, binding.legacy_path, *binding.alias_paths}:
                alias_key = (binding.dataset_revision, binding.domain, alias)
                if alias_key in self._lookup and self._lookup[alias_key] != binding:
                    raise ValueError("Ambiguous identity/path alias: " + alias)
                self._lookup[alias_key] = binding
                self._unversioned.setdefault((binding.domain, alias), []).append(binding)

    @classmethod
    def from_rows(cls, rows, dataset_revision=DEFAULT_REVISION):
        bindings = []
        for row in rows:
            sid = row.get("canonical_id") or row.get("proposed_sample_id") or row["sample_id"]
            legacy = row.get("legacy_sample_id") or row.get("legacy_id") or sid
            path = row.get("path") or row.get("proposed_public_path") or row.get("video_relative")
            aliases = row.get("alias_paths") or row.get("alias_paths_json") or ()
            if isinstance(aliases, str):
                aliases = json.loads(aliases)
            if not isinstance(aliases, (list, tuple)) or not all(isinstance(p, str) for p in aliases):
                raise ValueError("alias_paths must be a list of complete relative paths")
            aliases = tuple(dict.fromkeys([*aliases, *([row["legacy_public_path"]] if row.get("legacy_public_path") else [])]))
            bindings.append(IdentityBinding(
                dataset_revision=row.get("dataset_revision") or dataset_revision,
                domain=row["domain"], task=row["task"], canonical_id=sid,
                legacy_sample_id=legacy, path=path,
                legacy_path=row.get("legacy_path") or path, video_sha256=row["video_sha256"],
                full_identity_sha256=row.get("full_identity_sha256") or row.get("new_identity_full_sha256") or "",
                alias_paths=aliases))
        return cls(bindings)

    @classmethod
    def from_csv(cls, path, dataset_revision=DEFAULT_REVISION):
        with Path(path).open(newline="") as stream:
            return cls.from_rows(csv.DictReader(stream), dataset_revision)

    @classmethod
    def from_jsonl(cls, path, dataset_revision=DEFAULT_REVISION):
        with Path(path).open() as stream:
            return cls.from_rows((json.loads(line) for line in stream if line.strip()), dataset_revision)

    @classmethod
    def from_index(cls, index, *, domain="classic"):
        revision = index.get("dataset_revision", DEFAULT_REVISION)
        rows = []
        for row in index["items"].values():
            item = {**row, "domain": row.get("domain", domain)}
            if (item["domain"], item["task"]) == ("classic", "T1") and item["sample_id"].startswith("t1rgb16_"):
                full = identity_digest("classic", "T1", item["video_sha256"])
                item.update(canonical_id="t1_" + full[:20], legacy_sample_id=item["sample_id"],
                            full_identity_sha256=full)
            rows.append(item)
        return cls.from_rows(rows, revision)

    def resolve(self, value, *, domain, dataset_revision=None):
        if dataset_revision is not None:
            try:
                return self._lookup[(dataset_revision, domain, value)]
            except KeyError:
                raise KeyError(f"Unknown {domain}/{dataset_revision} identity or relative path: {value}") from None
        matches = self._unversioned.get((domain, value), ())
        if len(matches) != 1:
            raise KeyError("Unknown or revision-ambiguous identity/path: " + value)
        return matches[0]

    def canonical_id(self, value, *, domain, dataset_revision=None):
        return self.resolve(value, domain=domain, dataset_revision=dataset_revision).canonical_id

    def legacy_id(self, value, *, domain, dataset_revision=None):
        return self.resolve(value, domain=domain, dataset_revision=dataset_revision).legacy_sample_id

    def resolve_path(self, value, root, *, domain, dataset_revision=None, verify_sha=False, prefer="canonical"):
        binding = self.resolve(value, domain=domain, dataset_revision=dataset_revision)
        if prefer not in ("canonical", "legacy"):
            raise ValueError("prefer must be canonical or legacy")
        candidates = [binding.path, binding.legacy_path, *binding.alias_paths]
        if prefer == "legacy":
            candidates = [binding.legacy_path, *binding.alias_paths, binding.path]
        root = Path(root).resolve()
        for relative in dict.fromkeys(candidates):
            path = root / relative
            if not path.is_file():
                continue
            if not path.resolve().is_relative_to(root):
                # Hugging Face snapshot cache symlinks may point to sibling blobs;
                # accepting them requires the full expected content hash below.
                if not verify_sha:
                    raise ValueError("Media symlink leaves root; use verify_sha=True")
            if verify_sha:
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != binding.video_sha256:
                        raise ValueError("Resolved media SHA256 differs: " + str(path))
            return path
        raise FileNotFoundError("No current or legacy media path exists for " + binding.canonical_id)

    def rows(self):
        return [asdict(binding) for binding in self.bindings]

    def select_rows(self, rows, requested_ids=None, *, domain, dataset_revision=None):
        """Resolve requests, retaining frozen input order and duplicate semantics."""
        if requested_ids is None:
            return list(rows)
        selected = [self.legacy_id(value, domain=domain, dataset_revision=dataset_revision)
                    for value in requested_ids]
        if len(selected) != len(set(selected)):
            raise ValueError("Canonical/legacy requests select the same source twice")
        requested = set(selected)
        available = {source_sample_id(row) for row in rows}
        if not requested <= available:
            raise ValueError("Requested source is absent from the frozen row selection")
        return [row for row in rows if source_sample_id(row) in requested]

    def project_record(self, record, *, domain, dataset_revision=None):
        """A reversible analysis view; nested raw/timing/protocol records stay exact."""
        binding = self.resolve(record["sample_id"], domain=domain, dataset_revision=dataset_revision)
        if any(key in record for key in ("legacy_sample_id", "dataset_revision", "source_record_sha256")):
            raise ValueError("Record is already projected or has reserved identity fields")
        if record.get("video_sha256", binding.video_sha256) != binding.video_sha256:
            raise ValueError("Result/source media SHA differs")
        result = deepcopy(record)
        result.update(sample_id=binding.canonical_id, legacy_sample_id=binding.legacy_sample_id,
                      dataset_revision=binding.dataset_revision,
                      source_record_sha256=hashlib.sha256(canonical_json(record).encode()).hexdigest())
        return result

    def project_index(self, index, *, domain="classic", dataset_revision=None):
        """Preserve item order, source fields and scientific parameters in a new view."""
        from .manifest import digest
        from .dataset import validate_index
        validate_index(index)
        result = deepcopy(index)
        items = {}
        revisions = set()
        for sid, entry in index["items"].items():
            binding = self.resolve(sid, domain=domain, dataset_revision=dataset_revision)
            if entry["video_sha256"] != binding.video_sha256:
                raise ValueError("Index/source media SHA differs")
            projected = deepcopy(entry)
            projected.update(sample_id=binding.canonical_id, legacy_sample_id=binding.legacy_sample_id,
                video_relative=binding.path, legacy_path=binding.legacy_path,
                full_identity_sha256=binding.full_identity_sha256,
                alias_paths=list(binding.alias_paths),
                legacy_entry_sha256=digest(entry))
            items[binding.canonical_id] = projected
            revisions.add(binding.dataset_revision)
        if len(revisions) != 1:
            raise ValueError("An index view requires exactly one dataset revision")
        result.update(items=items, dataset_revision=revisions.pop(), identity_schema=REGISTRY_SCHEMA,
                      legacy_index_id=index["index_id"])
        result["index_id"] = digest({k: v for k, v in result.items() if k != "index_id"})
        return validate_index(result)


def restore_record(record):
    """Invert project_record and verify the exact original record digest."""
    result = deepcopy(record)
    original_digest = result.pop("source_record_sha256")
    result["sample_id"] = result.pop("legacy_sample_id")
    result.pop("dataset_revision")
    if hashlib.sha256(canonical_json(result).encode()).hexdigest() != original_digest:
        raise ValueError("Projected scientific record differs from its source")
    return result
