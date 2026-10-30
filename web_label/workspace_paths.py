"""Resolve preserved research references after local folder organization.

Original manifests and answers retain their literal paths and hashes. Only local
file access is redirected; this module is not used by production inference.
"""
from functools import lru_cache
import json
import os
import hashlib
import tempfile

from pathlib import Path as NativePath

WORKSPACE = NativePath(os.environ.get('CVPR_WORKSPACE', str(NativePath(__file__).resolve().parents[1])))


@lru_cache(maxsize=1)
def _aliases():
    index = WORKSPACE / 'archive/index.json'
    mapping = json.loads(index.read_text()).get('local_layout', {}).get('aliases', {}) if index.is_file() else {}
    return sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True)


def relocate(value):
    text = os.fspath(value) if not isinstance(value, Path) else str(value)
    absolute = os.path.abspath(text)
    seen = set()
    while absolute not in seen:
        seen.add(absolute)
        for old, new in _aliases():
            if absolute == old or absolute.startswith(old + os.sep):
                absolute = new + absolute[len(old):]
                break
        else:
            return absolute if len(seen) > 1 else text
    raise ValueError('Cyclic layout alias: ' + text)


class Path(type(NativePath())):
    """Keep original reference strings while opening their canonical local files."""
    def __fspath__(self):
        value = relocate(str(self))
        file = NativePath(value).absolute()
        root = WORKSPACE / 'configs'
        if file.parent == root and file.suffix == '.json' and not file.exists():
            return _profile_file(file.name)
        return value


_profile_temp = None


def profile_bytes(record):
    """Reconstruct an original protocol JSON without changing fields or bytes."""
    fmt = record['serialization']
    options = {k: fmt[k] for k in ('indent', 'ensure_ascii')}
    if fmt['separators'] is not None:
        options['separators'] = fmt['separators']
    raw = (json.dumps(record['parameters'], **options) + ('\n' if fmt['trailing_newline'] else '')).encode()
    if len(raw) != record['original_bytes'] or hashlib.sha256(raw).hexdigest() != record['sha256']:
        raise ValueError('Historical profile checksum differs')
    return raw


@lru_cache(maxsize=None)
def _profile_file(name):
    """Read a frozen profile through its original filename, with identical bytes."""
    global _profile_temp
    registry = WORKSPACE / 'configs/protocols.json'
    record = json.loads(registry.read_text())['profiles'].get(name)
    if record is None:
        return str(registry.parent / name)
    raw = profile_bytes(record)
    if _profile_temp is None:
        _profile_temp = tempfile.TemporaryDirectory(prefix='cvpr-frozen-profiles-')
    target = NativePath(_profile_temp.name) / name
    target.write_bytes(raw)
    return str(target)
