"""Upload an explicit media manifest to R2 using the existing Wrangler login."""
import argparse
import os
import concurrent.futures
import hashlib
import json
import threading
import time
import tomllib
from pathlib import Path as _FilesystemPath
import sys as _sys
_sys.path.insert(0, str(_FilesystemPath(__file__).resolve().parent))
from workspace_paths import Path
from urllib.parse import quote

import requests

ACCOUNT = os.environ.get('CLOUDFLARE_ACCOUNT_ID', '')
BUCKET = os.environ.get('MEDIA_BUCKET', 'vision-tasks-media')


def verify_remote(rows, token):
    """Bound each list to a known directory; compare bytes to local SHA/MD5."""
    session = requests.Session()
    session.headers['Authorization'] = 'Bearer '+token
    endpoint = f'https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/r2/buckets/{BUCKET}/objects'
    by_key = {r['key']:r for r in rows}
    remote = {}
    for prefix in sorted({r['key'].rsplit('/',1)[0]+'/' for r in rows}):
        response = session.get(endpoint, params={'prefix':prefix,'per_page':1000}, timeout=(20,60))
        response.raise_for_status()
        data = response.json()
        if not data.get('success'): raise ValueError('R2 list failed')
        for item in data['result']:
            if item['key'] in by_key: remote[item['key']] = item
        time.sleep(.31)
    missing = sorted(set(by_key)-set(remote))
    if missing: raise ValueError(f'{len(missing)} missing remote objects: {missing[:3]}')
    for key,row in by_key.items():
        if row.get('remote_existing'):
            actual=remote[key]
            if int(actual['size'])!=row['size'] or actual['etag'].strip('"')!=row['etag']:
                raise ValueError('Published-only object changed: '+key)
            continue
        sha,md5=hashlib.sha256(),hashlib.md5()
        with Path(row['local_path']).open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                sha.update(block);md5.update(block)
        actual=remote[key]
        if (int(actual['size'])!=row['size'] or actual['etag'].strip('"')!=md5.hexdigest()
                or sha.hexdigest()!=row['sha256']):
            raise ValueError('Remote/source verification failed: '+key)
    print(json.dumps({'verified_objects':len(rows),'verified_bytes':sum(r['size']for r in rows),
                      'verification':'local SHA256 where present; original upload SHA/MD5 for published-only media; remote byte size and ETag for all objects'}),flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=Path('/tmp/cvpr-web-publish/upload-manifest.json'))
    p.add_argument('--receipt', type=Path, default=Path(__file__).with_name('upload-receipt.json'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int)
    p.add_argument('--verify', action='store_true', help='Read back remote size and ETag for every manifest object; no writes')
    args = p.parse_args()
    rows = json.loads(args.manifest.read_text())
    assert sum(r['size'] for r in rows) < 9_000_000_000, 'Exceeds this release storage budget'
    cfg = tomllib.loads((Path.home()/'Library/Preferences/.wrangler/config/default.toml').read_text())
    token = cfg['oauth_token']  # Never print or persist credentials in the project.
    if args.verify:
        verify_remote(rows, token)
        return
    records = json.loads(args.receipt.read_text()).get('objects', {}) if args.receipt.exists() else {}
    todo = [r for r in rows if records.get(r['key'], {}).get('sha256') != r['sha256']]
    if any(r.get('remote_existing') for r in todo):
        raise ValueError('Published-only media requires its matching original upload receipt; restore that receipt before uploading.')
    todo.sort(key=lambda r: ('/originals/' in r['key'], -r['size']))
    if args.limit: todo = todo[:args.limit]
    if not todo:
        print(json.dumps({'queued':0,'completed_objects_total':len(records),'message':'No changed media; existing receipt preserved'}))
        return
    started = time.monotonic()
    lock = threading.Lock()
    next_request = [0.0]
    counters = {'requests': 0, 'retries': 0, 'uploaded_bytes': 0, 'uploaded_objects': 0}
    errors = []
    thread_local = threading.local()

    def paced():
        with lock:
            when = max(time.monotonic(), next_request[0])
            next_request[0] = when + 1/3.3
        time.sleep(max(0, when-time.monotonic()))

    def session():
        if not hasattr(thread_local, 'session'):
            thread_local.session = requests.Session()
            thread_local.session.headers.update({'Authorization': 'Bearer '+token})
        return thread_local.session

    def upload(row):
        path = Path(row['local_path'])
        with path.open('rb') as stream:
            sha, md5 = hashlib.sha256(), hashlib.md5()
            for block in iter(lambda: stream.read(1024*1024), b''):
                sha.update(block); md5.update(block)
        if path.stat().st_size != row['size'] or sha.hexdigest() != row['sha256']:
            raise ValueError('Source changed: '+row['key'])
        url = f'https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/r2/buckets/{BUCKET}/objects/'+quote(row['key'], safe='/')
        for attempt in range(5):
            paced()
            with lock:
                counters['requests'] += 1
                if attempt: counters['retries'] += 1
            try:
                with path.open('rb') as stream:
                    response = session().put(url, data=stream, headers={
                        'Content-Type': row['content_type'], 'Content-Length': str(row['size']),
                        'cf-r2-storage-class': 'Standard'}, timeout=(20, 600))
                if response.status_code == 429 or response.status_code >= 500:
                    time.sleep(min(40, float(response.headers.get('Retry-After', 2**(attempt+1)))))
                    continue
                response.raise_for_status()
                data = response.json()
                if not data.get('success'): raise ValueError('R2 rejected upload: '+str(data.get('errors')))
                result = data.get('result') or {}
                etag = str(result.get('etag', '')).strip('"')
                if etag and '-' not in etag and len(etag)==32 and etag != md5.hexdigest():
                    raise ValueError('Remote ETag differs: '+row['key'])
                if result.get('size') is not None and int(result['size']) != row['size']:
                    raise ValueError('Remote size differs: '+row['key'])
                return {**{k:row[k] for k in ('key','size','sha256')},'etag':etag,'source_md5':md5.hexdigest()}
            except requests.RequestException:
                if attempt == 4: raise
                time.sleep(min(30, 2**(attempt+1)))
        raise RuntimeError('Retries exhausted: '+row['key'])

    def save():
        elapsed = time.monotonic()-started
        summary = {**counters, 'elapsed_seconds':round(elapsed, 2),
                   'average_MB_s':round(counters['uploaded_bytes']/max(elapsed, .001)/1e6, 3),
                   'completed_objects_total':len(records),'expected_objects':len(rows),
                   'expected_bytes':sum(r['size'] for r in rows),'errors':errors}
        temp = args.receipt.with_suffix('.tmp')
        temp.write_text(json.dumps({'summary':summary,'objects':records},separators=(',',':')))
        temp.replace(args.receipt)
        print(json.dumps(summary), flush=True)

    print(json.dumps({'queued':len(todo),'bytes':sum(r['size'] for r in todo),'workers':args.workers}),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(upload, row):row for row in todo}
        last_report = 0
        for future in concurrent.futures.as_completed(futures):
            row = futures[future]
            try:
                result = future.result(); records[row['key']] = result
                counters['uploaded_objects'] += 1; counters['uploaded_bytes'] += row['size']
            except Exception as exc:
                errors.append({'key':row['key'],'error':str(exc)[:240]})
            if time.monotonic()-last_report>25:
                save(); last_report=time.monotonic()
    save()
    if errors: raise SystemExit(1)


if __name__ == '__main__': main()
