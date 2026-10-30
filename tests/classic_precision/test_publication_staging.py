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


def test_storage_control_recovers_network_and_expired_existing_token(monkeypatch):
    import requests
    from classic_precision.gcs_source_upload import ControlRequests
    monkeypatch.setattr('classic_precision.gcs_source_upload.time.sleep', lambda _: None)
    tokens = iter(['expired-test-token', 'refreshed-test-token'])
    observed = []
    outcomes = iter([requests.ConnectionError('transient DNS'), 401, 200])

    class Session:
        def request(self, method, url, **kwargs):
            observed.append(kwargs['headers']['Authorization'])
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return type('Response', (), {'status_code': outcome})()

    control = ControlRequests(lambda: next(tokens), session_factory=Session)
    assert control.request('GET', 'https://storage.googleapis.com/test').status_code == 200
    assert observed == ['Bearer expired-test-token', 'Bearer expired-test-token', 'Bearer refreshed-test-token']


def test_storage_control_retry_is_bounded_and_does_not_log_secret(monkeypatch):
    import requests
    from classic_precision.gcs_source_upload import ControlRequests
    monkeypatch.setattr('classic_precision.gcs_source_upload.time.sleep', lambda _: None)
    calls = []

    class Session:
        def request(self, *args, **kwargs):
            calls.append(1)
            raise requests.ConnectionError('dummy-sensitive-value')

    control = ControlRequests(lambda: 'test-token', session_factory=Session)
    with pytest.raises(RuntimeError) as error:
        control.request('POST', 'https://storage.googleapis.com/test')
    assert len(calls) == 5
    assert 'dummy-sensitive-value' not in str(error.value)
