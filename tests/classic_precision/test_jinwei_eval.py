import base64
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError
from zipfile import ZipFile

import pytest

from classic_precision import jinwei_eval as runner
from classic_precision import jinwei_release as release


def _fixture(tmp_path):
    def media(name, content):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return name, release.sha(content)

    def source(scope, sid, truth, path, digest, mask=None):
        domain, task = scope.split('/')
        return dict(sample_id=sid, domain=domain, task=task.upper(), truth=truth,
                    prompt='Frozen video question.', parameters={'strength': 0.5, 'mask': mask},
                    video_path=path, video_sha256=digest, frames=180, fps=60, width=768, height=512)

    scope = 'classic/t2'
    still, still_sha = media(scope + '/static/still.png', b'static fixture bytes')
    sources = []
    for sid in ('source-a', 'source-b'):
        path, digest = media(f'{scope}/{sid}.mkv', sid.encode())
        video = source(scope, sid, 'LEFT', path, digest)
        static = dict(sample_id=sid, truth='LEFT', video_sha256=digest, video_path=path,
                      static_case_id='shared-static-case', model_input_sha256=still_sha,
                      frame_indices=[24], choices=['LEFT', 'RIGHT'], prompt='Frozen static question.',
                      images=[dict(path=still, sha256=still_sha, frame_index=24)])
        sources.extend((release.trial(scope, video, 'fixture', 'video-sha'),
                        release.trial(scope, video, 'fixture/static', 'static-sha', static=static)))
    scope = 'natural/t5'
    path, digest = media(scope + '/video.mp4', b't5 video fixture')
    ref, ref_sha = media(scope + '/reference.png', b't5 reference fixture')
    video = source(scope, 'source-c', 'YES', path, digest, 'none')
    video.update(reference_path=ref, reference_sha256=ref_sha)
    inspection = dict(sample_id='source-c', truth='YES', video_sha256=digest, video_path=path,
                      purpose='stimulus inspection', model_evaluated=False, images=[
                          dict(path=ref, sha256=ref_sha, role='reference'),
                          dict(path=ref, sha256=ref_sha, role='event'),
                          dict(path=ref, sha256=ref_sha, role='event')])
    sources.extend((release.trial(scope, video, 'fixture', 'video-sha'),
                    release.trial(scope, video, 'fixture/inspection', 'static-sha', static=inspection)))
    return sources


def _response(text, *, finish='stop'):
    return json.dumps({'id': 'original-response-id', 'model': 'returned-version', 'usage': {'total_tokens': 20},
                       'choices': [{'finish_reason': finish, 'message': {'content': text}}]},
                      indent=2).encode() + b'\n'


def _args(root, **changes):
    values = dict(scope=None, input=None, execute=True, max_requests=20, allow_private_media_transfer=False,
                  media_root=root, model='provider/company/model-name', max_tokens=4096, workers=2,
                  retry=None, retry_uncertain=False, pilot_trial=None)
    return SimpleNamespace(**{**values, **changes})


def test_fixture_end_to_end(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    sources = _fixture(tmp_path)
    monkeypatch.setattr(release, 'public_plan', lambda _: sources.copy())
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential-unused')
    plan, rows = runner.build_plan(out)
    assert plan['logical_rows'] == 6
    assert plan['unique_requests'] == 4
    sent = []
    original = []

    def fake_transport(payload, credential):
        assert credential == 'fixture-credential-unused'
        request = json.loads(payload)
        assert set(request) == {'model', 'messages', 'stream', 'max_completion_tokens'}
        assert request['model'] == 'provider/company/model-name'
        parts = request['messages'][0]['content']
        media = [part[part['type']]['url'] for part in parts[:-1]]
        sent.append((media, parts[-1]['text']))
        if media[-1].startswith('data:video/x-matroska'):
            decoded = base64.b64decode(media[-1].split(',')[1])
            result = _response(' LEFT! ' if decoded == b'source-a' else 'RIGHT')
        elif media[-1].startswith('data:video/mp4'):
            assert [part['type'] for part in parts] == ['image_url', 'video_url', 'text']
            result = _response('YES')
        else:
            assert parts[-1]['text'] == 'Frozen static question.'
            result = _response('LEFT')
        original.append(result)
        return 200, {'x-request-id': 'fixture-id'}, result

    monkeypatch.setattr(runner, 'transport', fake_transport)
    pilot = runner.execution(out, plan, rows, _args(tmp_path), None, pilot=True)
    assert pilot['started'] == 3 and not pilot['blocked_profiles']
    run = runner.execution(out, plan, rows, _args(tmp_path), None, pilot=False)
    assert run['started'] == 1
    assert runner.execution(out, plan, rows, _args(tmp_path), None, pilot=False)['started'] == 0
    assert len(sent) == 4
    archive = out / 'return.zip'
    assert runner.export(out, archive)['request_attempts'] == 4
    assert runner.verify(archive)['status'] == 'VERIFIED'
    with ZipFile(archive) as package:
        actual = [base64.b64decode(r['response_body_b64'])
                  for r in runner.load_rows(package.read('attempts.jsonl'))]
        assert sorted(actual) == sorted(original)
        summary = json.loads(package.read('summary.json'))['profiles']['all']
        assert summary['logical_rows']['planned_accuracy_eligible'] == 5
        assert summary['logical_rows']['observed_scorable'] == 5
        assert summary['independent_requests']['observed_scorable'] == 4
        assert summary['independent_requests']['correct'] == 3
        assert summary['independent_requests']['strict_format_valid'] == 3
        assert summary['logical_rows']['status_counts']['INSPECTION_ONLY'] == 1
        assert b'fixture-credential-unused' not in package.read('attempts.jsonl')
        assert b't5 video fixture' not in package.read('attempts.jsonl')
    with pytest.raises(ValueError, match='Model, request parameters'):
        runner.execution(out, plan, rows, _args(tmp_path, max_tokens=40), None, pilot=False)


def test_explicit_retry_preserves_ambiguous_attempt(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    sources = _fixture(tmp_path)
    monkeypatch.setattr(release, 'public_plan', lambda _: sources.copy())
    monkeypatch.setenv('NVIDIA_API_KEY', 'fixture-credential-unused')
    plan, rows = runner.build_plan(out)

    def timeout(*_):
        raise URLError('ambiguous network failure')

    monkeypatch.setattr(runner, 'transport', timeout)
    args = _args(tmp_path, scope=['natural/t5'], input='video')
    result = runner.execution(out, plan, rows, args, None, pilot=True)
    assert result['statuses'] == {'TRANSPORT_UNKNOWN': 1}
    assert runner.execution(out, plan, rows, args, None, pilot=True)['started'] == 0
    assert runner.execution(out, plan, rows, args, None, pilot=False)['started'] == 0
    key = next(r['request_id'] for r in rows if r['scope'] == 'natural/t5' and r['input'] == 'video')
    args.retry = [key]
    with pytest.raises(ValueError, match='cannot be retried'):
        runner.execution(out, plan, rows, args, None, pilot=True)
    args.retry_uncertain = True
    monkeypatch.setattr(runner, 'transport', lambda *_: (200, {}, _response('YES')))
    assert runner.execution(out, plan, rows, args, None, pilot=True)['statuses'] == {'ANSWERED': 1}
    archive = out / 'return.zip'
    runner.export(out, archive)
    assert runner.verify(archive)['request_attempts'] == 2
    with ZipFile(archive) as package:
        old, new = runner.load_rows(package.read('attempts.jsonl'))
        assert old['status'] == 'TRANSPORT_UNKNOWN' and old['request_sha256'] == new['request_sha256']
        assert new['attempt'] == 2 and new['normalized_answer'] == 'YES'


@pytest.mark.parametrize(('body', 'status', 'finish'), [
    ('LEFT!.', 'FORMAT_OR_TOOL_FAILURE', 'stop'),
    ('LEFT or RIGHT', 'FORMAT_OR_TOOL_FAILURE', 'stop'),
    ('LEFT', 'TRUNCATED', 'length'),
])
def test_unscorable_originals(body, status, finish):
    result = runner.response_fields(_response(body, finish=finish), 200, ['LEFT', 'RIGHT'], None)
    assert result['status'] == status
    assert result['raw_answer'] == body and result['normalized_answer'] is None
    assert not result['modality_admitted']


def test_credential_and_media_echo_not_exported(tmp_path, monkeypatch):
    out = tmp_path / 'run'
    out.mkdir()
    monkeypatch.setattr(release, 'public_plan', lambda _: _fixture(tmp_path))
    monkeypatch.setenv('NVIDIA_API_KEY', 'credential-must-stay-secret')
    plan, rows = runner.build_plan(out)
    for scope, original, withheld in (
        ('classic/t2', b'credential-must-stay-secret', 'credential_echo'),
        ('natural/t5', b'data:video/mp4;base64,YQ==', 'media_echo'),
    ):
        monkeypatch.setattr(runner, 'transport', lambda *_: (400, {}, original))
        assert runner.execution(out, plan, rows, _args(tmp_path, scope=[scope], input='video'),
                                None, pilot=True)['statuses'] == {'SENSITIVE_RESPONSE_WITHHELD': 1}
    archive = out / 'return.zip'
    runner.export(out, archive)
    assert runner.verify(archive)['request_attempts'] == 2
    with ZipFile(archive) as package:
        attempts = runner.load_rows(package.read('attempts.jsonl'))
        assert {r['response_body_withheld'] for r in attempts} == {'credential_echo', 'media_echo'}
        assert all(r['response_body_b64'] is None for r in attempts)
