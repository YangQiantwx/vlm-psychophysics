import {Outbox} from './outbox.js';
import {MediaPool} from './media-pool.js?v=8ef852e1a776';
const $ = id => document.getElementById(id);
const config = await fetch(new URL('./config.json', import.meta.url)).then(r => r.json());
const storageKey = 'cvpr-label-test-v1';
let token = localStorage.getItem(storageKey), batch = null, current = 0, lastTask = null;
let playing = false, starting = false, ready = false, saving = false, answeredAt = 0;
let evidence = null, pool = null, outbox = null, view = 0, callbackId = null;
let retryTimer = null, retryDelay = 1000, delivery = null, restoredReference = null;
let restorePending = false;

function status(message = '', error = false) {
  if (message === 'Identity changed') return;
  $('status').textContent = message; $('status').classList.toggle('error', error);
}
function show(id) {
  for (const name of ['identity', 'choose', 'trial', 'done']) $(name).hidden = name !== id;
  $('change-identity').hidden = !token;
}
function busy(message = '') {
  $('connection-progress').hidden = !message;
  $('connection-label').textContent = message;
}
async function api(path, body, raw = false, signal) {
  const identityToken = token;
  const timeout = AbortSignal.timeout(raw ? 300000 : 30000);
  const r = await fetch(config.api + path, {
    method: body === undefined ? 'GET' : 'POST', signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    headers: {...(identityToken ? {Authorization: 'Bearer ' + identityToken} : {}), ...(body === undefined ? {} : {'Content-Type': 'application/json'})},
    body: body === undefined ? undefined : JSON.stringify(body)
  });
  if (!r.ok) { const x = await r.json().catch(() => ({})); const error = Error(x.error || 'Connection interrupted. Try again.'); error.status = r.status; throw error; }
  const value = raw ? r : await r.json();
  if (identityToken !== token) throw Error('Identity changed');
  return value;
}
function syncState(count, error) {
  $('sync-status').textContent = count ? (error ? 'Saved on this device · reconnecting' : 'Saving…') : 'All responses saved';
  $('sync-status').dataset.pending = String(count);
}
function connectOutbox() {
  outbox = new Outbox(localStorage, token, count => syncState(count));
  syncState(outbox.entries().length);
}
async function flush() { if (outbox) await outbox.drain((path, body) => api(path, body)); }
function sync() {
  const identityToken = token, journal = outbox;
  if (!journal) return;
  clearTimeout(retryTimer);
  journal.drain((path, body) => api(path, body)).then(() => {
    if (token !== identityToken) return;
    retryDelay = 1000; syncState(0);
    if (restorePending) state().catch(error => status(error.message, true));
  }).catch(error => {
    if (token !== identityToken) return;
    syncState(journal.entries().length, error);
    if (error.status && error.status < 500 && error.status !== 429) {
      status('Your records are kept on this device. ' + error.message, true); return;
    }
    retryTimer = setTimeout(sync, retryDelay); retryDelay = Math.min(15000, retryDelay * 2);
  });
}
function clearMedia() {
  view++; pool?.close(); pool = null;
  ready = false; playing = false; $('trial').dataset.ready = 'false'; $('play').disabled = true;
  $('loading-details').hidden = true;
  if (restoredReference) URL.revokeObjectURL(restoredReference);
  restoredReference = null; $('reference').hidden = true; $('reference-image').removeAttribute('src');
  const v = $('video'); v.pause(); v.removeAttribute('src'); v.load();
}
function size() {
  const r = $('video').getBoundingClientRect();
  $('display-size').textContent = `768 × 512 source · CSS ${Math.round(r.width)} × ${Math.round(r.height)} · ${Math.round(r.width / 768 * 100)}% · DPR ${devicePixelRatio}`;
  return {native: [768, 512], css: [r.width, r.height], dpr: devicePixelRatio, css_rect: [r.left, r.top, r.width, r.height], viewport: [innerWidth, innerHeight], screen: [screen.width, screen.height], user_agent: navigator.userAgent,
    reference_css: $('reference').hidden ? null : [$('reference-image').getBoundingClientRect().width, $('reference-image').getBoundingClientRect().height]};
}
function bufferState() {
  const count = pool ? [...pool.entries.values()].filter(e => e.index > current && e.ready).length : 0;
  $('buffer-status').textContent = !batch || current + 1 >= batch.total ? '' : count ? 'Next video ready' : 'Preparing next video…';
  $('buffer-status').dataset.ready = String(count);
  const entry = pool?.entries.get(current);
  if (!entry || ready || playing || starting || batch?.trials[current]?.started) return;
  $('loading-details').hidden = false;
  const bar = $('load-progress');
  if (entry.total) bar.value = Math.min(1, entry.loaded / entry.total);
  else bar.removeAttribute('value');
  const phases = {queued:'Connecting…', downloading:'Downloading video…', verifying:'Checking video…', decoding:'Preparing playback…', reference:'Preparing reference…', ready:'Ready'};
  $('screen-label').textContent = phases[entry.phase] || 'Preparing video…';
  $('load-amount').textContent = entry.total ? `${Math.floor(Math.min(100, entry.loaded / entry.total * 100))}% · ${(entry.loaded / 1e6).toFixed(1)} / ${(entry.total / 1e6).toFixed(1)} MB` : 'Waiting for video data…';
}
function mount(entry) {
  const old = $('video'), v = entry.video; v.id = 'video';
  if (old !== v) { old.removeAttribute('id'); old.replaceWith(v); }
  bindVideo(v);
  $('reference').hidden = !entry.reference;
  if (entry.reference) $('reference-image').src = entry.referenceURL;
}
async function restoreReference(ref) {
  $('reference').hidden = !ref;
  if (!ref) return;
  const blob = await pool.blob(ref.media, ref.sha256);
  restoredReference = URL.createObjectURL(blob); $('reference-image').src = restoredReference;
  await $('reference-image').decode();
}
async function displayTrial() {
  const generation = ++view, index = current, began = performance.now();
  status(); show('trial'); playing = false; ready = false; saving = false;
  $('play').hidden = false; $('play').disabled = true; $('play').textContent = 'Play once';
  $('answers').hidden = true; $('skip').hidden = true; $('retry-save').hidden = true;
  $('screen').classList.remove('hidden'); $('screen-label').textContent = 'Preparing video…';
  $('loading-details').hidden = true;
  $('reference').hidden = true;
  $('task-title').textContent = batch.task; $('progress').textContent = `${current + 1} / ${batch.total}`;
  $('progress-fill').style.width = `${current / batch.total * 100}%`; $('prompt').textContent = batch.prompt;
  $('trial').dataset.index = String(index); $('trial').dataset.ready = 'false';
  const trial = batch.trials[index];
  if (config.disabled_tasks?.includes(batch.task)) {
    $('play').hidden = true; $('screen-label').textContent = 'Source update in progress';
    status('This batch is paused. Saved responses are retained.'); return;
  }
  pool.window(index); bufferState();
  if (trial.started) {
    $('loading-details').hidden = true;
    $('play').hidden = true; $('screen-label').textContent = trial.played ? 'Choose your response' : 'Playback interrupted';
    if (trial.played) { answeredAt = performance.now(); $('answers').hidden = false; }
    else $('skip').hidden = false;
    try { await restoreReference(trial.reference || batch.reference); } catch (error) { status(error.message, true); }
    return;
  }
  const wasReady = Boolean(pool.entries.get(index)?.ready);
  try {
    const entry = await pool.get(index);
    if (generation !== view) return;
    mount(entry); size(); ready = true;
    delivery = {prefetched: wasReady, selection_to_ready_ms: performance.now() - began, download_verify_ms: entry.download_verify_ms, decode_ms: entry.decode_ms, bytes: entry.bytes};
    $('play').disabled = false; $('screen-label').textContent = 'Ready'; $('trial').dataset.ready = 'true';
    $('loading-details').hidden = true;
    $('trial').dataset.readyMs = delivery.selection_to_ready_ms.toFixed(2); bufferState();
    document.dispatchEvent(new CustomEvent('label-ready', {detail: {...delivery, index}}));
  } catch (error) {
    if (generation !== view) return;
    status(error.message, true); $('screen-label').textContent = 'Video unavailable';
    $('loading-details').hidden = true;
    $('play').textContent = 'Retry loading'; $('play').disabled = false;
  }
}
function buildChoices() {
  $('answers').replaceChildren();
  for (const choice of batch.choices) {
    const button = document.createElement('button'); button.textContent = choice;
    button.onclick = () => saveAnswer(choice); $('answers').append(button);
  }
}
async function useBatch(value) {
  clearMedia(); batch = value; lastTask = batch.task; buildChoices();
  pool = new MediaPool(batch, (path, signal) => api(path, undefined, true, signal), bufferState);
  current = batch.trials.findIndex(t => !t.answered);
  if (current < 0) { await finish(); return; }
  await displayTrial();
}
async function state() {
  busy('Restoring your session…');
  let data;
  try { await flush(); data = await api('/api/state'); }
  finally { busy(); }
  restorePending = false;
  if (data.batch) { await useBatch(data.batch); return; }
  clearMedia(); batch = null; show('choose'); $('tasks').replaceChildren();
  for (const task of data.tasks) {
    const button = document.createElement('button'); button.textContent = `${task.id} · ${task.title}${task.disabled || config.disabled_tasks?.includes(task.id) ? ' · Source update' : ''}`;
    button.disabled = task.remaining === 0 || task.disabled || config.disabled_tasks?.includes(task.id);
    button.onclick = () => claim(task.id); $('tasks').append(button);
  }
  status();
}
async function claim(task) {
  if (config.disabled_tasks?.includes(task)) { status('Source update in progress'); return; }
  $('next').disabled = true;
  try {
    busy('Preparing your batch…'); await flush(); status(); const result = await api('/api/claim', {task}); busy();
    if (result.batch) await useBatch(result.batch); else await state();
  } catch (error) { status(error.message, true); }
  finally { busy(); $('next').disabled = false; }
}
async function finish() {
  const identityToken = token, bid = batch.id;
  show('done'); $('next').disabled = true; $('other').disabled = true;
  $('done-title').textContent = 'Batch complete.'; $('done-message').textContent = 'Responses saved on this device · finishing sync…';
  clearMedia();
  try {
    await flush(); await api('/api/complete', {batch_id: bid});
    if (token !== identityToken || batch.id !== bid) return;
    $('done-title').textContent = 'Batch saved.'; $('done-message').textContent = `${batch.total} responses saved · Preview records`;
    $('next').disabled = false; $('other').disabled = false; status(); syncState(0);
  } catch (error) {
    if (token !== identityToken) return;
    status('Your responses are kept on this device. Reconnecting…', true);
    clearTimeout(retryTimer); retryTimer = setTimeout(() => finish(), 5000);
  }
}
function saveAnswer(answer, reason) {
  if (saving) return;
  saving = true;
  const payload = {batch_id: batch.id, index: current, answer, ...(reason ? {reason} : {}), client_at: new Date().toISOString(), reaction_ms: Math.round(performance.now() - answeredAt)};
  try {
    outbox.add('/api/answer', payload); // durable before either upload or UI advance
    batch.trials[current].answered = true; current++; sync();
    if (current === batch.total) finish(); else displayTrial();
  } catch (error) { saving = false; status(error.message + ' Please keep this page open and retry.', true); }
}
async function startPlayback() {
  if (!ready && !playing && !starting) { await displayTrial(); return; }
  if (!ready || playing || starting || batch.trials[current].started) return;
  starting = true; $('play').disabled = true; $('change-identity').disabled = true;
  const clickTime = performance.now(), v = $('video');
  $('screen-label').textContent = 'Starting…';
  try {
    // Upload may proceed while the next video is visible and ready. The existing
    // server order/single-play gate must still acknowledge before playback.
    await flush();
    await api('/api/start', {batch_id: batch.id, index: current, client_at: new Date().toISOString(), display: size()});
    batch.trials[current].started = true; playing = true;
    v.defaultPlaybackRate = 1; v.playbackRate = 1;
    evidence = {ended: false, rate: 1, started_perf: performance.now(), frame_callbacks: 0, first_media_time: null, last_media_time: null, events: [], visibility_at_start: document.visibilityState, focused_at_start: document.hasFocus(), started_client: new Date().toISOString(), delivery: {...delivery, click_to_start_ack_ms: performance.now() - clickTime}};
    const frame = (_, metadata) => {
      evidence.frame_callbacks++;
      if (evidence.first_media_time === null) { evidence.first_media_time = metadata.mediaTime; evidence.delivery.click_to_first_frame_ms = performance.now() - clickTime; }
      evidence.last_media_time = metadata.mediaTime;
      if (playing) callbackId = v.requestVideoFrameCallback(frame);
    };
    if (v.requestVideoFrameCallback) callbackId = v.requestVideoFrameCallback(frame);
    $('screen').classList.add('hidden'); $('play').hidden = true; await v.play();
  } catch (error) {
    playing = false; status(error.message, true);
    if (batch.trials[current].started || error.status === 409) {
      try { await state(); } catch (recovery) { status(recovery.message, true); }
    } else { $('play').disabled = false; $('screen-label').textContent = 'Ready · reconnect to play'; sync(); }
  } finally { starting = false; $('change-identity').disabled = playing; }
}
function ended(v) {
  if (!playing || !evidence) return;
  playing = false; $('change-identity').disabled = false; answeredAt = performance.now();
  if (callbackId && v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(callbackId);
  evidence.ended = true; evidence.client_elapsed_ms = performance.now() - evidence.started_perf; delete evidence.started_perf;
  evidence.played_ranges = Array.from({length: v.played.length}, (_, i) => [v.played.start(i), v.played.end(i)]);
  evidence.rate = v.playbackRate; evidence.media_time = v.currentTime; evidence.duration = v.duration;
  const quality = v.getVideoPlaybackQuality?.(); evidence.quality = quality ? {total: quality.totalVideoFrames, dropped: quality.droppedVideoFrames} : null;
  evidence.ended_client = new Date().toISOString(); evidence.display_at_end = size();
  $('screen').classList.remove('hidden'); $('screen-label').textContent = 'Choose your response';
  v.removeAttribute('src'); v.load();
  try {
    outbox.add('/api/playback', {batch_id: batch.id, index: current, evidence});
    batch.trials[current].played = true; $('answers').hidden = false; sync();
  } catch (error) { status(error.message + ' Keep this page open and retry saving.', true); $('retry-save').hidden = false; }
}
function bindVideo(v) {
  v.onended = () => ended(v); v.oncontextmenu = e => e.preventDefault();
  for (const event of ['waiting', 'stalled', 'pause', 'seeking', 'ratechange']) v.addEventListener(event, () => {
    if (playing && evidence.events.length < 128) evidence.events.push({event, at: performance.now(), media_time: v.currentTime});
  });
  v.onratechange = () => { if (v.playbackRate !== 1) v.playbackRate = 1; };
}
new ResizeObserver(() => { size(); if (playing && evidence.events.length < 128) evidence.events.push({event: 'resize', at: performance.now()}); }).observe($('stage'));
document.addEventListener('visibilitychange', () => {
  if (playing && evidence.events.length < 128) evidence.events.push({event: document.hidden ? 'hidden' : 'visible', at: performance.now()});
  if (!document.hidden && token) sync();
});
window.addEventListener('blur', () => { if (playing && evidence.events.length < 128) evidence.events.push({event: 'blur', at: performance.now()}); });
window.addEventListener('online', sync);
window.addEventListener('beforeunload', event => { if (outbox?.entries().length) { event.preventDefault(); event.returnValue = ''; } });
$('identity-form').onsubmit = async event => {
  event.preventDefault(); const button = $('identity-form').querySelector('button'); button.disabled = true;
  try {
    busy('Connecting to your saved identity…');
    const data = await api('/api/session', {email: $('email').value, mode: config.mode});
    token = data.token; localStorage.setItem(storageKey, token); connectOutbox(); await state();
  } catch (error) { status(error.message, true); } finally { busy(); button.disabled = false; }
};
$('play').onclick = startPlayback;
$('retry-save').onclick = () => {
  try { outbox.add('/api/playback', {batch_id: batch.id, index: current, evidence}); batch.trials[current].played = true; $('retry-save').hidden = true; $('answers').hidden = false; status(); sync(); }
  catch (error) { status(error.message, true); }
};
$('skip').onclick = () => saveAnswer(null, 'playback_interrupted');
$('next').onclick = () => claim(lastTask); $('other').onclick = () => state().catch(error => status(error.message, true));
$('change-identity').onclick = async () => {
  try { await flush(); clearTimeout(retryTimer); localStorage.removeItem(storageKey); token = null; outbox = null; clearMedia(); batch = null; show('identity'); status(); $('sync-status').textContent = ''; }
  catch (error) { status('Your records are saved on this device. Reconnect before exiting.', true); }
};
if (token) {
  show(null); restorePending = true; status('Restoring session…');
  try { connectOutbox(); await state(); }
  catch (error) {
    if (error.status === 401 && !outbox?.entries().length) { localStorage.removeItem(storageKey); token = null; show('identity'); }
    status(error.message, true); if (outbox?.entries().length) sync();
  }
}
