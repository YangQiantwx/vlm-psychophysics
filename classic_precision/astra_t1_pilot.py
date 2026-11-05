"""Prepare or submit a finite GPT-6 Astra T1 video manifest."""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import getpass
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

from . import astra_direct_eval as evidence

FRAME_COUNT = 180
FPS = 60
WIDTH, HEIGHT = 768, 512
MAX_INPUT_TOKENS = 100_000
MAX_OUTPUT_TOKENS = 4096
MAX_REQUEST_BYTES = 300_000_000
RESERVE_USD = Decimal(str(evidence.cost(MAX_INPUT_TOKENS, MAX_OUTPUT_TOKENS)))
PROMPT = (
    'The images are consecutive frames from a video, presented in chronological order.\n\n'
    'Watch the video and focus on the bright and dark stripe pattern inside the central circular aperture. '
    'The background image and aperture remain stationary. Your task is to judge the horizontal direction '
    'in which the stripe pattern travels relative to that fixed background. Follow the pattern across '
    'successive moments throughout the video, rather than comparing only the first and last images. '
    'Stripe spacing and visibility may vary, but the judgment is always about the stripes, not the '
    'depicted scene, the aperture boundary, or a change in overall brightness. Decide whether the '
    'ongoing stripe motion is toward the left or toward the right. Give your final answer as exactly '
    'one uppercase word: LEFT or RIGHT.'
)
REQUEST_SETTINGS = {
    'model': evidence.MODEL,
    'reasoning': {'effort': 'low', 'summary': 'auto'},
    'tools': [], 'tool_choice': 'none', 'max_output_tokens': MAX_OUTPUT_TOKENS,
    'truncation': 'disabled', 'store': False, 'stream': False, 'service_tier': 'default',
    'prompt_cache_options': {'mode': 'explicit'},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def save_bytes(path, data):
    with path.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def read_manifest(path):
    data = path.read_bytes()
    rows = json.loads(data)
    evidence.require(isinstance(rows, list) and rows, 'Manifest must be a nonempty JSON array')
    ids = set()
    cases = []
    for index, row in enumerate(rows, 1):
        evidence.require(isinstance(row, dict) and row.keys() <= {'id', 'video', 'truth'},
                         f'Invalid manifest row {index}')
        case_id, video, truth = row.get('id'), row.get('video'), row.get('truth')
        evidence.require(isinstance(case_id, str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,63}', case_id)
                         and case_id.casefold() not in ids, f'Invalid or duplicate id in row {index}')
        evidence.require(isinstance(video, str) and video and truth in (None, 'LEFT', 'RIGHT'),
                         f'Expected a video path and optional LEFT/RIGHT truth in row {index}')
        ids.add(case_id.casefold())
        cases.append({'id': case_id, 'video': str((path.parent / video).resolve()), 'truth': truth})
    return evidence.digest(data), cases


def inspect_stream(ffprobe, video):
    probe = subprocess.check_output([
        ffprobe, '-v', 'error', '-select_streams', 'v:0', '-show_streams', '-show_frames',
        '-show_entries', 'stream=width,height,pix_fmt,codec_name,time_base,r_frame_rate:frame=best_effort_timestamp,best_effort_timestamp_time',
        '-of', 'json', str(video),
    ], timeout=60)
    decoded = json.loads(probe)
    (stream,) = decoded['streams']
    frames = decoded['frames']
    pts = [int(frame['best_effort_timestamp']) for frame in frames]
    ticks = Fraction(stream['time_base'])
    evidence.require((stream['width'], stream['height']) == (WIDTH, HEIGHT)
                     and Fraction(stream['r_frame_rate']) == FPS and len(pts) == FRAME_COUNT
                     and ticks > 0 and all(a < b for a, b in zip(pts, pts[1:]))
                     and all(abs((p - pts[0]) * ticks - Fraction(i, FPS)) <= ticks
                             for i, p in enumerate(pts)),
                     f'{video.name}: expected 768×512, 180 ordered frames at 60 fps')
    return probe, frames


def prepare_case(cv2, ffprobe, case, directory):
    directory.mkdir(parents=True)
    video = Path(case['video'])
    source_sha256 = evidence.sha(video)
    probe, timestamps = inspect_stream(ffprobe, video)
    save_bytes(directory / 'ffprobe.json', probe)
    content = [{'type': 'input_text', 'text': PROMPT}]
    frames = []
    capture = cv2.VideoCapture(str(video), cv2.CAP_FFMPEG)
    try:
        for index, pts in enumerate(timestamps):
            ok, bgr = capture.read()
            evidence.require(ok and bgr.shape == (HEIGHT, WIDTH, 3) and bgr.dtype.name == 'uint8',
                             f'{case["id"]}: invalid RGB8 decode at frame {index}')
            ok, png = cv2.imencode('.png', bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            evidence.require(ok and cv2.imdecode(png, cv2.IMREAD_COLOR).tobytes() == bgr.tobytes(),
                             f'{case["id"]}: PNG round trip failed at frame {index}')
            data = png.tobytes()
            frames.append({'index': index, 'pts': pts, 'png_sha256': evidence.digest(data),
                           'rgb8_sha256': evidence.digest(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).tobytes())})
            content.append({'type': 'input_image', 'detail': 'original',
                            'image_url': 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')})
        evidence.require(not capture.read()[0], f'{case["id"]}: extra decoded frame')
    finally:
        capture.release()
    evidence.require(source_sha256 == evidence.sha(video), f'{case["id"]}: source changed during decode')
    request = evidence.packed(REQUEST_SETTINGS | {'input': [{'role': 'user', 'content': content}]})
    evidence.require(len(request) < MAX_REQUEST_BYTES, f'{case["id"]}: request exceeds 300 MB')
    save_bytes(directory / 'request.json', request)
    evidence.write(directory / 'frames.json', frames)
    return {**case, 'source_sha256': source_sha256, 'frames': len(frames),
            'request_sha256': evidence.digest(request), 'request_bytes': len(request)}


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def submit(opener, directory, body, key, project, truth):
    headers = {'Content-Type': 'application/json', 'X-Client-Request-Id': str(uuid.uuid4())}
    if project:
        headers['OpenAI-Project'] = project
    request = urllib.request.Request(evidence.API, body, headers | {'Authorization': f'Bearer {key}'},
                                     method='POST')
    evidence.write(directory / 'attempt.json', {
        'url': evidence.API, 'method': 'POST', 'headers': headers,
        'request_sha256': evidence.digest(body), 'started_at_utc': utc_now(),
    })
    started = time.monotonic()
    secret = key.encode()
    try:
        try:
            response = opener.open(request, timeout=240)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status = response.code
            safe_headers = [(name, value.replace(key, '[REDACTED]')) for name, value in response.headers.items()
                            if name.lower() not in {'set-cookie', 'authorization', 'proxy-authorization'}]
            evidence.write(directory / 'response-http.json', {'status': status, 'headers': safe_headers})
            response_body = response.read()
    except (OSError, EOFError, http.client.HTTPException) as error:
        partial = getattr(error, 'partial', b'')
        if isinstance(partial, bytes) and partial:
            save_bytes(directory / 'response-partial.bin', partial.replace(secret, b'[REDACTED]'))
        return {'status': 'transport_outcome_unknown', 'error_type': type(error).__name__,
                'finished_at_utc': utc_now(), 'elapsed_seconds': time.monotonic() - started}

    saved = response_body.replace(secret, b'[REDACTED]')
    save_bytes(directory / 'response.raw.json', saved)
    result = {'status': 'http_error' if status != 200 else 'unaccepted_response',
              'http_status': status, 'finished_at_utc': utc_now(),
              'elapsed_seconds': time.monotonic() - started, 'response_sha256': evidence.digest(saved),
              'credential_redacted_from_body': saved != response_body}
    try:
        decoded = json.loads(saved)
        usage = decoded.get('usage')
        if usage is not None:
            result['usage_priced_usd'], _ = evidence.usage_cost(usage)
        if status == 200:
            evidence.require(decoded.get('reasoning', {}).get('effort') == 'low' and
                             decoded.get('service_tier') == 'default', 'Unexpected served protocol')
            scored = evidence.parse_response(decoded)
            result.update(scored, status='completed', truth=truth,
                          correct=None if truth is None else scored['answer'] == truth)
            if (scored['usage']['input_tokens'] > MAX_INPUT_TOKENS or
                    scored['usage']['output_tokens'] > MAX_OUTPUT_TOKENS or
                    Decimal(str(scored['standard_cost_usd'])) > RESERVE_USD):
                result['status'] = 'reservation_exceeded'
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        result['validation_error_type'] = type(error).__name__
    return result


def usd(text):
    try:
        value = Decimal(text)
    except InvalidOperation:
        raise argparse.ArgumentTypeError('Expected a USD amount') from None
    if not value.is_finite() or value < 0:
        raise argparse.ArgumentTypeError('Budget must be finite and nonnegative')
    return value


def implementation_identity():
    return {'entry_sha256': evidence.sha(Path(__file__)),
            'evidence_helper_sha256': evidence.sha(Path(evidence.__file__))}


def prepare(output, cases, manifest_sha, budget):
    try:
        import cv2
    except ImportError:
        raise ValueError('Install opencv-python-headless in your existing Python environment') from None
    ffprobe = shutil.which('ffprobe')
    evidence.require(ffprobe, 'Install FFmpeg and put ffprobe on PATH')
    evidence.require(not (output / 'cases').exists(), 'Use a fresh output directory for dry-run')
    cv2.setNumThreads(1)
    items = [prepare_case(cv2, ffprobe, row, output / 'cases' / row['id']) for row in cases]
    receipt = {'manifest_sha256': manifest_sha, 'implementation': implementation_identity(),
               'opencv': cv2.__version__, 'prepared_at_utc': utc_now(), 'request_settings': REQUEST_SETTINGS,
               'cases': items, 'budget_usd': str(budget), 'per_request_reserve_usd': str(RESERVE_USD),
               'all_requests_reserve_usd': str(len(items) * RESERVE_USD), 'api_attempts': 0}
    evidence.write(output / 'preflight.json', receipt)
    print(f'Dry-run: {len(items)} videos prepared, 0 API calls. Reserve ${RESERVE_USD} per request.')
    print(f'All requests reserved at ${len(items) * RESERVE_USD}. Budget: ${budget}. See {output / "preflight.json"}.')


def execute(output, cases, manifest_sha, budget):
    preflight = evidence.read(output / 'preflight.json')
    evidence.require(preflight['manifest_sha256'] == manifest_sha and Decimal(preflight['budget_usd']) == budget
                     and preflight['implementation'] == implementation_identity(), 'Dry-run identity or budget changed')
    evidence.require(not (output / 'start.json').exists() and not (output / 'complete.json').exists(),
                     'Output already submitted. Use a new manifest/output for any unattempted cases.')
    pairs = list(zip(cases, preflight['cases'], strict=True))
    for case, prepared in pairs:
        directory = output / 'cases' / case['id']
        evidence.require(case['id'] == prepared['id'] and not (directory / 'attempt.json').exists()
                         and evidence.sha(directory / 'request.json') == prepared['request_sha256'],
                         f'{case["id"]}: request changed or was previously attempted')
    evidence.require(budget >= RESERVE_USD, f'Budget cannot reserve one ${RESERVE_USD} request')
    key = os.environ.get('OPENAI_API_KEY', '').strip()
    if not key and sys.stdin.isatty():
        key = getpass.getpass('OpenAI API key: ').strip()
    evidence.require(re.fullmatch(r'sk-[A-Za-z0-9_-]+', key),
                     'Supply your OpenAI API key via OPENAI_API_KEY or the hidden prompt')
    project = os.environ.get('OPENAI_PROJECT_ID')
    evidence.require(not project or re.fullmatch(r'proj_[A-Za-z0-9]+', project), 'Invalid OPENAI_PROJECT_ID')
    evidence.write(output / 'start.json', {'started_at_utc': utc_now(), 'manifest_sha256': manifest_sha,
                                           'preflight_sha256': evidence.sha(output / 'preflight.json')})
    opener = urllib.request.build_opener(NoRedirect())
    results = []
    spent = Decimal(0)
    stop = 'all_cases_completed'
    for case, prepared in pairs:
        if spent + RESERVE_USD > budget:
            stop = 'budget_reserve_stop'
            break
        directory = output / 'cases' / case['id']
        body = (directory / 'request.json').read_bytes()
        evidence.require(evidence.digest(body) == prepared['request_sha256'], 'Prepared request changed')
        result = submit(opener, directory, body, key, project, case['truth'])
        evidence.write(directory / 'result.json', result)
        results.append({'id': case['id'], **result})
        spent += Decimal(str(result.get('usage_priced_usd', 0)))
        print(f'{case["id"]}: {result["status"]} {result.get("answer") or ""}'.rstrip(), flush=True)
        if result['status'] != 'completed':
            stop = result['status']
            break
    evidence.write(output / 'complete.json', {
        'status': stop, 'finished_at_utc': utc_now(), 'planned': len(cases),
        'attempted': len(results), 'completed': sum(row['status'] == 'completed' for row in results),
        'usage_priced_usd': str(spent), 'budget_usd': str(budget),
        'unattempted_ids': [row['id'] for row in cases[len(results):]], 'results': results,
    })
    print(f'{stop}: {len(results)} attempts, ${spent} estimated from returned usage.')
    return int(stop not in {'all_cases_completed', 'budget_reserve_stop'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True, help='JSON array of id, video, and optional truth')
    parser.add_argument('--output', type=Path, required=True, help='Directory for requests and response evidence')
    parser.add_argument('--budget-usd', type=usd, required=True, help='Maximum estimated usage spend')
    parser.add_argument('--execute', action='store_true', help='Submit requests from a successful dry-run')
    args = parser.parse_args()
    os.umask(0o077)
    try:
        manifest_sha, cases = read_manifest(args.manifest.resolve())
        output = args.output.resolve()
        with evidence.execution_lock(output):
            if args.execute:
                return execute(output, cases, manifest_sha, args.budget_usd)
            prepare(output, cases, manifest_sha, args.budget_usd)
    except (ValueError, OSError, subprocess.SubprocessError, KeyError, TypeError, EOFError) as error:
        parser.exit(1, f'{parser.prog}: {error}\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
