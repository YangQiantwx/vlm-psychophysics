"""Publication staging must never alias files consumed by live inference."""
import hashlib
import os

import pytest

from classic_precision.source_release import link


def test_staged_copy_cannot_write_source(tmp_path):
    source, staged = tmp_path / 'source', tmp_path / 'staged'
    source.write_bytes(b'frozen scientific input')
    before = source.stat()
    link(source, staged, hashlib.sha256(source.read_bytes()).hexdigest())
    assert source.stat().st_ino != staged.stat().st_ino
    staged.write_bytes(b'changed publication copy')
    assert source.read_bytes() == b'frozen scientific input'
    assert source.stat().st_mtime_ns == before.st_mtime_ns


def test_existing_shared_inode_is_rejected(tmp_path):
    source, staged = tmp_path / 'source', tmp_path / 'staged'
    source.write_bytes(b'frozen scientific input')
    os.link(source, staged)
    with pytest.raises(AssertionError, match='independent'):
        link(source, staged)


def test_existing_symlink_is_rejected(tmp_path):
    source, staged = tmp_path / 'source', tmp_path / 'staged'
    source.write_bytes(b'frozen scientific input')
    staged.symlink_to(source)
    with pytest.raises(AssertionError, match='alias'):
        link(source, staged)
