"""Blind, one-play annotation API. Durable units are independent GCS objects."""
import hashlib
import json
import os
import random
import re
import secrets
import time
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from flask import Flask, Response, jsonify, request
from google.api_core.exceptions import NotFound, PreconditionFailed
from google.cloud import storage

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 65536
BUCKET = os.environ.get('LABEL_BUCKET', 'cvpr2027-label-access-cis230202-570037-us-west1')
ORIGINS = set(os.environ.get('LABEL_ORIGINS', 'https://vision-tasks.pages.dev,http://localhost:18795,http://127.0.0.1:18795').split(','))
HUMAN_ENABLED = os.environ.get('HUMAN_ENABLED', '0') == '1'
client = None
BATCH_SIZE = 20
ORDER_ALGORITHM = 'stratum_round_robin_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def bucket():
    global client
    if client is None:
        client = storage.Client()
    return client.bucket(BUCKET)


def read(key):
    b = bucket().blob(key)
    try:
        raw = b.download_as_bytes()
        return json.loads(raw), int(b.generation)
    except NotFound:
        return None, 0


def write(key, value, generation=0):
    bucket().blob(key).upload_from_string(json.dumps(value, sort_keys=True, separators=(',', ':')), content_type='application/json', if_generation_match=generation)


def immutable(key, value):
    try:
        write(key, value)
        return value, True
    except PreconditionFailed:
        return read(key)[0], False


def fail(message, code=400):
    return jsonify(error=message), code


@app.after_request
def headers(response):
    origin = request.headers.get('Origin', '')
    if origin in ORIGINS:
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Vary'] = 'Origin'
    response.headers['Access-Control-Allow-Headers'] = 'Authorization,Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    response.headers['Cache-Control'] = 'no-store'
    response.headers['X-Content-Type-Options'] = 'nosniff'
    return response


@app.before_request
def gate():
    if request.method == 'OPTIONS':
        return Response(status=204)
    if request.method == 'POST' and request.headers.get('Origin') and request.headers['Origin'] not in ORIGINS:
        return fail('Origin is not allowed', 403)


@app.errorhandler(413)
def too_large(_):
    return fail('Request too large', 413)


def identity():
    token = request.headers.get('Authorization', '').removeprefix('Bearer ')
    if not re.fullmatch(r'[A-Za-z0-9_-]{40,80}', token):
        return None
    session, _ = read('sessions/' + hashlib.sha256(token.encode()).hexdigest() + '.json')
    if not session or time.time() > session['expires_epoch']:
        return None
    return session


def base(session):
    return 'records/' + session['mode'] + '/' + session['participant_id'] + '/'


def collection_policy():
    return read('manifests/collection-policy.json')[0] or {'disabled_tasks': []}


def disabled_task(task):
    return task in collection_policy().get('disabled_tasks', [])


def state_for(session):
    return read(base(session) + 'state.json')


@lru_cache(maxsize=16)
def cached_manifest(key):
    return read(key)[0]


def refresh_preview_release(session):
    # Preview inventories evolve during source QA. Human sequences remain pinned.
    if session['mode'] != 'test':
        return
    pointer, _ = read('manifests/current-test.json')
    if not pointer:
        return
    for _ in range(8):
        state, generation = state_for(session)
        if not state or state['manifest_object'] == pointer['object']:
            return
        manifest = cached_manifest(pointer['object'])
        seen = {tid: set(ids) for tid, ids in state.get('seen_ids', {}).items()}
        for tid, task in state['tasks'].items():
            seen.setdefault(tid, set()).update(task['order'][:task['cursor']])
        active = state.get('active')
        if active:
            for i, source_id in enumerate(active['items']):
                if read(event_key(session, active, i, 'start'))[0] or read(event_key(session, active, i, 'answer'))[0]:
                    seen.setdefault(active['task'], set()).add(source_id)
        immutable(base(session) + 'releases/' + state['manifest_version'] + '/state-' + str(generation) + '.json', {**state, 'archived_reason': 'preview_manifest_updated', 'archived_at': now()})
        updated = {**state, 'manifest_object': pointer['object'], 'manifest_version': manifest['version'], 'release_at': now(), 'order_algorithm': ORDER_ALGORITHM, 'active': None, 'seen_ids': {tid: sorted(ids) for tid, ids in seen.items()}, 'tasks': {}}
        for tid, task in manifest['tasks'].items():
            unseen = [item for item in task['items'] if item['id'] not in seen.get(tid, set())]
            updated['tasks'][tid] = {'order': ordered(unseen, state['seed'] + tid + manifest['version']), 'cursor': 0, 'allocation_seed': state['seed'] + tid + manifest['version']}
        try:
            write(base(session) + 'state.json', updated, generation)
            return
        except PreconditionFailed:
            continue
    raise RuntimeError('Preview release is busy; retry')


def manifest_for(state):
    return cached_manifest(state['manifest_object'])


def ordered(items, seed):
    # Shuffle complete condition groups, then round-robin them. No difficulty ranking.
    rng = random.Random(seed)
    groups = {}
    for item in items:
        group = json.dumps(item.get('stratum', {}), sort_keys=True)
        groups.setdefault(group, []).append(item['id'])
    values = list(groups.values())
    for group in values:
        rng.shuffle(group)
    order = []
    while values:
        rng.shuffle(values)
        order.extend(group.pop() for group in values)
        values = [group for group in values if group]
    return order


def allocation_seed(state, tid):
    return state['tasks'][tid].get('allocation_seed', state['seed'] + tid + (state['manifest_version'] if state.get('release_at') else ''))


def event_key(session, batch, index, kind):
    return base(session) + 'batches/' + batch['id'] + '/' + f'{index:03d}' + '-' + kind + '.json'


def current_trial(session, data):
    state, generation = state_for(session)
    batch = (state or {}).get('active')
    if not batch or batch['id'] != data.get('batch_id'):
        return None, None, None, None
    index = data.get('index')
    if not isinstance(index, int) or index < 0 or index >= len(batch['items']):
        return None, None, None, None
    # Only first unanswered trial may play or submit; prefetch is separate.
    for i in range(index):
        if not read(event_key(session, batch, i, 'answer'))[0]:
            return None, None, None, None
    return state, generation, batch, index


def prior_event(session, data, kind):
    bid, index = data.get('batch_id'), data.get('index')
    if not isinstance(bid, str) or not re.fullmatch(r'[a-f0-9]{32}', bid) or not isinstance(index, int) or not 0 <= index < BATCH_SIZE:
        return None
    return read(base(session) + 'batches/' + bid + '/' + f'{index:03d}' + '-' + kind + '.json')[0]


def public_batch(session, state):
    batch = state.get('active')
    if not batch:
        return None
    manifest = manifest_for(state)
    task = manifest['tasks'][batch['task']]
    items = {x['id']: x for x in task['items']}
    trials = []
    keys = [(i, kind) for i in range(len(batch['items'])) for kind in ('start', 'playback', 'answer')]
    with ThreadPoolExecutor(max_workers=12) as pool:
        events = dict(zip(keys, pool.map(lambda entry: bool(read(event_key(session, batch, *entry))[0]), keys)))
    for i, source_id in enumerate(batch['items']):
        start = events[(i, 'start')]
        end = events[(i, 'playback')]
        answer = events[(i, 'answer')]
        item = items[source_id]
        trials.append({'index': i, 'duration': item.get('duration', 3), 'bytes': item.get('bytes'),
                       'sha256': item['sha256'], 'width': item['width'], 'height': item['height'],
                       'started': bool(start), 'played': bool(end), 'answered': bool(answer),
                       'media': '/api/media/' + batch['id'] + '/' + str(i),
                       'reference': ({'media': '/api/reference/' + batch['id'] + '/' + str(i), 'sha256': item['reference']['sha256'], 'width': 160, 'height': 160} if item.get('reference') else None)})
    return {'id': batch['id'], 'task': batch['task'], 'batch_size': BATCH_SIZE, 'total': len(trials),
            'offset': batch['offset'], 'prompt': task['prompt'], 'choices': task['choices'],
            'trials': trials, 'complete': all(x['answered'] for x in trials), 'mode': session['mode'],
            'reference': ({'media': '/api/reference/' + batch['id'], 'sha256': task['reference']['sha256']} if task.get('reference') else None)}


@app.get('/health')
def health():
    return jsonify(status='ok', persistence='GCS-generation-CAS', human_enabled=HUMAN_ENABLED)


@app.post('/api/session')
def session_create():
    data = request.get_json(silent=True) or {}
    email = str(data.get('email', '')).strip().lower()
    mode = data.get('mode', 'test')
    if not re.fullmatch(r'[^\s@]{1,64}@[^\s@]{1,189}\.[^\s@]{2,}', email) or len(email) > 254:
        return fail('Enter an email address')
    if mode not in ('test', 'human') or (mode == 'human' and not HUMAN_ENABLED):
        return fail('Participant collection is not open', 403)
    email_key = hashlib.sha256(email.encode()).hexdigest()
    record, _ = immutable('identities/' + mode + '/' + email_key + '.json', {'email': email, 'participant_id': secrets.token_hex(16), 'created_at': now(), 'mode': mode})
    participant_id = record['participant_id']
    pointer, _ = read('manifests/current-' + mode + '.json')
    if not pointer:
        return fail('No qualified video inventory is available', 503)
    manifest, _ = read(pointer['object'])
    initial = {'participant_id': participant_id, 'mode': mode, 'manifest_object': pointer['object'], 'manifest_version': manifest['version'], 'seed': secrets.token_hex(16), 'order_algorithm': ORDER_ALGORITHM, 'created_at': now(), 'tasks': {}, 'active': None}
    for tid, task in manifest['tasks'].items():
        initial['tasks'][tid] = {'order': ordered(task['items'], initial['seed'] + tid), 'cursor': 0, 'allocation_seed': initial['seed'] + tid}
    immutable('records/' + mode + '/' + participant_id + '/state.json', initial)
    token = secrets.token_urlsafe(40)
    write('sessions/' + hashlib.sha256(token.encode()).hexdigest() + '.json', {'participant_id': participant_id, 'mode': mode, 'expires_epoch': time.time() + 30 * 86400})
    return jsonify(token=token, mode=mode)


@app.get('/api/state')
def state():
    session = identity()
    if not session:
        return fail('Session expired', 401)
    refresh_preview_release(session)
    state, _ = state_for(session)
    m = manifest_for(state)
    return jsonify(mode=session['mode'], tasks=[{'id': tid, 'title': task['title'], 'remaining': len(state['tasks'][tid]['order']) - state['tasks'][tid]['cursor'], 'disabled': disabled_task(tid)} for tid, task in m['tasks'].items()], batch=public_batch(session, state))


@app.post('/api/claim')
def claim():
    session = identity()
    if not session:
        return fail('Session expired', 401)
    refresh_preview_release(session)
    tid = (request.get_json(silent=True) or {}).get('task')
    if disabled_task(tid):
        return fail('Source update in progress; this task is temporarily closed', 409)
    for _ in range(8):
        state, generation = state_for(session)
        if state['active']:
            return jsonify(batch=public_batch(session, state), resumed=True)
        if tid not in state['tasks']:
            return fail('Unknown task')
        task = state['tasks'][tid]
        ids = task['order'][task['cursor']:task['cursor'] + BATCH_SIZE]
        if not ids:
            return jsonify(batch=None, exhausted=True)
        state['active'] = {'id': secrets.token_hex(16), 'task': tid, 'items': ids, 'offset': task['cursor'], 'created_at': now(), 'allocation_seed': allocation_seed(state, tid), 'order_algorithm': ORDER_ALGORITHM}
        try:
            write(base(session) + 'state.json', state, generation)
            return jsonify(batch=public_batch(session, state), resumed=False)
        except PreconditionFailed:
            continue
    return fail('Assignment is busy; retry', 409)


@app.post('/api/start')
def start():
    session = identity()
    if not session:
        return fail('Session expired', 401)
    data = request.get_json(silent=True) or {}
    state, _, batch, index = current_trial(session, data)
    if not batch:
        return fail('Trial is not current', 409)
    if disabled_task(batch['task']):
        return fail('Source update in progress; playback is temporarily closed', 409)
    record, created = immutable(event_key(session, batch, index, 'start'), {'server_at': now(), 'server_epoch': time.time(), 'client_at': data.get('client_at'), 'display': data.get('display'), 'manifest_version': state['manifest_version'], 'source_id': batch['items'][index], 'participant_seed': state['seed'], 'allocation_seed': allocation_seed(state, batch['task']), 'order_algorithm': ORDER_ALGORITHM, 'trial_position': batch['offset'] + index})
    if not created:
        return fail('This trial has already started; replay is disabled', 409)
    return jsonify(started=True)


@app.post('/api/playback')
def playback():
    session = identity()
    if not session:
        return fail('Session expired', 401)
    data = request.get_json(silent=True) or {}
    state, _, batch, index = current_trial(session, data)
    if not batch:
        if prior_event(session, data, 'playback'):
            return jsonify(saved=True, duplicate=True)
        return fail('Trial is not current', 409)
    started, _ = read(event_key(session, batch, index, 'start'))
    if not started:
        return fail('Playback was not started', 409)
    item = next(x for x in manifest_for(state)['tasks'][batch['task']]['items'] if x['id'] == batch['items'][index])
    evidence = data.get('evidence') or {}
    elapsed = time.time() - started['server_epoch']
    duration = item.get('duration', 3)
    if elapsed < duration - .25 or evidence.get('ended') is not True or evidence.get('rate') != 1 or evidence.get('media_time', 0) < duration - .12:
        return fail('Complete one uninterrupted playback before answering', 409)
    record, created = immutable(event_key(session, batch, index, 'playback'), {'server_at': now(), 'elapsed_server_s': elapsed, 'evidence': evidence, 'source_id': batch['items'][index], 'scope': 'browser-reported playback evidence; not a gaze or perceptual verification'})
    return jsonify(saved=True, duplicate=not created)


@app.post('/api/answer')
def answer():
    session = identity()
    if not session:
        return fail('Session expired', 401)
    data = request.get_json(silent=True) or {}
    state, _, batch, index = current_trial(session, data)
    if not batch:
        old = prior_event(session, data, 'answer')
        if old and old['answer'] == data.get('answer') and old.get('reason') == data.get('reason'):
            return jsonify(saved=True, duplicate=True)
        return fail('Trial is not current', 409)
    played, _ = read(event_key(session, batch, index, 'playback'))
    started, _ = read(event_key(session, batch, index, 'start'))
    answer = data.get('answer')
    skipped = data.get('reason') == 'playback_interrupted'
    task_config = manifest_for(state)['tasks'][batch['task']]
    choices = task_config['choices']
    source_item = next(x for x in task_config['items'] if x['id'] == batch['items'][index])
    if not started or (not played and not skipped) or (skipped and played) or (not skipped and answer not in choices) or (skipped and answer is not None):
        return fail('Response is not valid for this playback state', 409)
    record = {'answer': answer, 'reason': data.get('reason'), 'client_at': data.get('client_at'), 'reaction_ms': data.get('reaction_ms'), 'server_at': now(), 'source_id': batch['items'][index], 'task': batch['task'], 'batch_id': batch['id'], 'batch_index': index, 'trial_position': batch['offset'] + index, 'manifest_version': state['manifest_version'], 'participant_seed': state['seed'], 'allocation_seed': allocation_seed(state, batch['task']), 'order_algorithm': ORDER_ALGORITHM, 'mode': session['mode'], 'test': session['mode'] == 'test', 'research_collection': session['mode'] == 'human', 'consent_status': 'not_requested_preview' if session['mode'] == 'test' else 'managed_by_study_protocol', 'playback_complete': bool(played), 'accuracy_eligible': source_item.get('accuracy_eligible', True), 'reference_sha256': (source_item.get('reference') or {}).get('sha256')}
    old, created = immutable(event_key(session, batch, index, 'answer'), record)
    if not created and (old['answer'] != answer or old.get('reason') != data.get('reason')):
        return fail('An immutable response already exists', 409)
    return jsonify(saved=True, duplicate=not created)


@app.post('/api/complete')
def complete():
    session = identity()
    if not session:
        return fail('Session expired', 401)
    bid = (request.get_json(silent=True) or {}).get('batch_id')
    key = base(session) + 'completed/' + str(bid) + '.json'
    for _ in range(8):
        state, generation = state_for(session)
        batch = state.get('active')
        if not batch or batch['id'] != bid:
            if read(key)[0]:
                return jsonify(completed=True, duplicate=True)
            return fail('Batch is not current', 409)
        if not all(read(event_key(session, batch, i, 'answer'))[0] for i in range(len(batch['items']))):
            return fail('Finish every trial first', 409)
        # Completion marker precedes cursor advancement. A retry repairs a crash between them.
        immutable(key, {**batch, 'completed_at': now(), 'manifest_version': state['manifest_version'], 'mode': session['mode']})
        state['tasks'][batch['task']]['cursor'] = batch['offset'] + len(batch['items'])
        state['active'] = None
        try:
            write(base(session) + 'state.json', state, generation)
            return jsonify(completed=True, duplicate=False)
        except PreconditionFailed:
            continue
    return fail('Completion is busy; retry', 409)


def streamed_media(object_name, kind, filename):
    # A generator keeps large T1 files out of process memory and enables HTTP
    # chunked transfer beyond Cloud Run's non-streamed 32 MiB response limit.
    # The client still awaits the complete blob and verifies its SHA before Play.
    blob = bucket().blob(object_name)
    def chunks():
        with blob.open('rb', chunk_size=4 * 1024 * 1024) as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                yield chunk
    return Response(chunks(), mimetype=kind, headers={'Content-Disposition': 'inline; filename="' + filename + '"'})


@app.get('/api/media/<bid>/<int:index>')
def media(bid, index):
    session = identity()
    if not session:
        return fail('Session expired', 401)
    state, _ = state_for(session)
    batch = state.get('active')
    if not batch or batch['id'] != bid or index < 0 or index >= len(batch['items']):
        return fail('Media is not assigned', 404)
    if read(event_key(session, batch, index, 'start'))[0]:
        return fail('Media is closed after playback starts', 410)
    task = manifest_for(state)['tasks'][batch['task']]
    item = next(x for x in task['items'] if x['id'] == batch['items'][index])
    return streamed_media(item['object'], 'video/webm', 'clip.webm')


@app.get('/api/reference/<bid>')
@app.get('/api/reference/<bid>/<int:index>')
def reference(bid, index=None):
    session = identity()
    if not session:
        return fail('Session expired', 401)
    state, _ = state_for(session)
    batch = state.get('active')
    if not batch or batch['id'] != bid:
        return fail('Reference is not assigned', 404)
    task = manifest_for(state)['tasks'][batch['task']]
    if index is None:
        ref = task.get('reference')
    elif 0 <= index < len(batch['items']):
        ref = next(x for x in task['items'] if x['id'] == batch['items'][index]).get('reference')
    else:
        ref = None
    if not ref:
        return fail('No reference for this task', 404)
    raw = bucket().blob(ref['object']).download_as_bytes()
    return Response(raw, mimetype='image/png', headers={'Content-Length': str(len(raw)), 'Content-Disposition': 'inline; filename="reference.png"'})


def examples_manifest():
    pointer, _ = read('manifests/site-current.json')
    return cached_manifest(pointer['object']) if pointer else {'version': 'pending', 'items': []}


@app.get('/api/examples')
def examples():
    manifest = examples_manifest()
    fields = ('alias', 'task', 'domain', 'seed', 'coherence', 'direction', 'k', 'position_pair', 'scenario', 'mask', 'source_id', 'sha256', 'reference_alias', 'status')
    return jsonify(version=manifest['version'], items=[{k: v for k, v in item.items() if k in fields} for item in manifest['items']])


@app.get('/api/example-media/<alias>')
def example_media(alias):
    if not re.fullmatch(r'[a-f0-9]{16,64}', alias):
        return fail('Not found', 404)
    item = next((x for x in examples_manifest()['items'] if x['alias'] == alias), None)
    if not item or item.get('status') != 'current':
        return fail('This source is not current', 404)
    if not item['object'].startswith(('browser/', 'references/')):
        return fail('Not found', 404)
    kind = 'image/png' if item['object'].endswith('.png') else 'video/webm'
    return streamed_media(item['object'], kind, 'sample.' + ('png' if kind == 'image/png' else 'webm'))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')))
