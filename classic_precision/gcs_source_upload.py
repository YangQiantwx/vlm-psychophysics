"""Delegate object-scoped GCS upload sessions without copying OAuth credentials.

The controller runs on the authenticated Mac; the worker sends original bytes
directly from the workstation. Session URLs stay in process memory and stdin.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import requests


def work(item: dict) -> dict:
    result = {'object': item['object'], 'sha256': item['sha256'], 'bytes': item['bytes']}
    url = item['session']
    try:
        # Dataset v0.4 intentionally aliases unchanged v0.3 masters. Resolve the
        # declared source once, then verify its exact bytes before upload.
        p = Path(item['path']).resolve(strict=True)
        before = p.stat()
        assert before.st_size == item['bytes'] and p.is_file()
        h, m = hashlib.sha256(), hashlib.md5()
        with p.open('rb') as f:
            for chunk in iter(lambda: f.read(8 << 20), b''):
                h.update(chunk)
                m.update(chunk)
        assert h.hexdigest() == item['sha256']
        md5 = base64.b64encode(m.digest()).decode()
        with p.open('rb') as f:
            response = requests.put(url, data=f, headers={
                'Content-Length': str(item['bytes']), 'Content-Type': item['content_type'],
                'Content-MD5': md5}, timeout=600)
        if response.status_code not in (200, 201):
            raise RuntimeError(f'Upload HTTP {response.status_code}')
        v = response.json()
        assert int(v['size']) == item['bytes'] and v['md5Hash'] == md5
        assert v['metadata']['sha256'] == item['sha256']
        after = p.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        result.update(status='UPLOADED_SOURCE_SHA_AND_GCS_MD5_VERIFIED', md5=md5,
                      generation=v['generation'])
    except Exception as exc:
        # Never stringify a requests exception: it can contain the secret session URL.
        result.update(status='UPLOAD_FAILED', error_type=type(exc).__name__)
        try:
            requests.delete(url, timeout=30)
        except Exception:
            pass
    return result


def worker(workers: int):
    lock = threading.Lock()
    def run(x):
        r = work(x)
        with lock:
            print(json.dumps(r), flush=True)
    # Input is bounded by the controller; URLs are neither logged nor persisted.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = []
        for line in sys.stdin:
            pending.append(pool.submit(run, json.loads(line)))
        for future in pending:
            future.result()


def controller(args):
    items = json.loads(args.files.read_text())['items']
    assert items and len({x['object'] for x in items}) == len(items)
    assert args.object_prefix in ('originals/', 'browser/', 'references/')
    assert all(x['object'].startswith(args.object_prefix) for x in items)
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env['CLOUDSDK_PYTHON'] = sys.executable
    token = subprocess.check_output([args.gcloud, 'auth', 'print-access-token'], env=env, text=True).strip()
    # No OAuth access token or auth file is sent to the remote worker.
    command = ['ssh', '-i', str(args.ssh_key), '-o', 'BatchMode=yes', args.remote,
               f'{args.remote_python} -B {args.remote_script} --worker --workers {args.workers}']
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    completed, errors, sessions = [], [], []
    worker_completed = [0]
    write_lock = threading.Lock()
    auth = {'Authorization': 'Bearer ' + token}
    existing = {}
    cursor = None
    while True:
        params = {'prefix': args.object_prefix, 'maxResults': 1000,
                  'fields': 'nextPageToken,items(name,size,metadata,md5Hash,generation)'}
        if cursor:
            params['pageToken'] = cursor
        response = requests.get(f'https://storage.googleapis.com/storage/v1/b/{args.bucket}/o',
                                params=params, headers=auth, timeout=90)
        response.raise_for_status()
        listing = response.json()
        existing.update({r['name']: r for r in listing.get('items', [])})
        cursor = listing.get('nextPageToken')
        if not cursor:
            break
    # Read worker output continuously so the SSH pipe cannot deadlock.
    def receive():
        with (args.out / 'objects.jsonl').open('a') as log:
            for line in process.stdout:
                record = json.loads(line)
                log.write(json.dumps(record) + '\n')
                log.flush()
                completed.append(record)
                worker_completed[0] += 1
                if record['status'] == 'UPLOAD_FAILED':
                    errors.append(record)
                if len(completed) % 100 == 0:
                    print(json.dumps({'completed': len(completed), 'failures': len(errors)}), flush=True)
    receiver = threading.Thread(target=receive)
    receiver.start()
    # Bound outstanding upload capabilities to limit memory and exposure.
    def begin(item):
        if item['object'] in existing:
            x = existing[item['object']]
            assert int(x['size']) == item['bytes'] and x.get('metadata', {}).get('sha256') == item['sha256']
            r = {k: item[k] for k in ('object', 'sha256', 'bytes')}
            r.update(status='EXISTING_IDENTITY_VERIFIED', generation=x['generation'], md5=x['md5Hash'])
            with write_lock:
                completed.append(r)
                with (args.out / 'objects.jsonl').open('a') as f:
                    f.write(json.dumps(r) + '\n')
            return
        while len(sessions) - worker_completed[0] >= args.workers * 3:
            time.sleep(.02)
        ct = item.get('content_type', 'application/octet-stream')
        meta = {'name': item['object'], 'contentType': ct,
                'metadata': {'sha256': item['sha256'], 'dataset': item.get('dataset', '')}}
        if item.get('source_sha256'):
            meta['metadata']['source_sha256'] = item['source_sha256']
        response = requests.post(f'https://storage.googleapis.com/upload/storage/v1/b/{args.bucket}/o',
             params={'uploadType': 'resumable', 'ifGenerationMatch': 0}, headers={**auth,
             'X-Upload-Content-Type': ct, 'X-Upload-Content-Length': str(item['bytes'])},
             json=meta, timeout=90)
        if response.status_code == 412:
            v = requests.get(f'https://storage.googleapis.com/storage/v1/b/{args.bucket}/o/' +
                 requests.utils.quote(item['object'], safe=''), headers=auth, timeout=60)
            v.raise_for_status()
            x = v.json()
            assert int(x['size']) == item['bytes'] and x['metadata']['sha256'] == item['sha256']
            r = {k: item[k] for k in ('object', 'sha256', 'bytes')}
            r.update(status='EXISTING_IDENTITY_VERIFIED', generation=x['generation'], md5=x['md5Hash'])
            with write_lock:
                completed.append(r)
                with (args.out / 'objects.jsonl').open('a') as f:
                    f.write(json.dumps(r) + '\n')
            return
        if response.status_code != 200:
            raise RuntimeError(f'Session initiation HTTP {response.status_code}')
        session = response.headers['Location']
        assert session.startswith('https://storage.googleapis.com/')
        # Secret session capability is sent only through the encrypted SSH stdin.
        with write_lock:
            process.stdin.write(json.dumps({**item, 'session': session, 'content_type': ct}) + '\n')
            process.stdin.flush()
            sessions.append(item['object'])
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(begin, item) for item in items]):
                future.result()
    finally:
        process.stdin.close()
    code = process.wait()
    receiver.join()
    success = code == 0 and len(completed) == len(items) and not errors
    summary = {'status': 'ALL_ORIGINALS_UPLOADED_AND_IDENTITIES_VERIFIED' if success else 'INCOMPLETE',
               'bucket': args.bucket, 'expected': len(items), 'completed': len(completed),
               'failed': len(errors), 'bytes': sum(r['bytes'] for r in completed if r['status'] != 'UPLOAD_FAILED'),
               'OAuth_credentials_copied': False, 'session_URLs_persisted': False,
               'worker_exit': code, 'model_calls': 0}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary), flush=True)
    if not success:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--files', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--bucket')
    ap.add_argument('--remote')
    ap.add_argument('--remote-python')
    ap.add_argument('--remote-script')
    ap.add_argument('--ssh-key', type=Path)
    ap.add_argument('--gcloud', default='/usr/local/bin/gcloud')
    ap.add_argument('--object-prefix', default='originals/')
    args = ap.parse_args()
    if args.worker:
        worker(args.workers)
    else:
        controller(args)


if __name__ == '__main__':
    main()
