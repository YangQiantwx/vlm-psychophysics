// Complete, SHA-verified blobs and paused decoded video elements travel together.
// Moving the ready element into the stage avoids resetting its decoder on Next.
export class MediaPool {
  constructor(batch, fetchMedia, changed = () => {}) {
    this.batch = batch; this.fetchMedia = fetchMedia; this.changed = changed;
    this.entries = new Map(); this.queue = []; this.active = 0; this.closed = false;
    this.current = 0; this.maxBytes = 384 * 1024 * 1024;
  }
  window(index) {
    this.current = index;
    for (const [i, entry] of this.entries) if (i < index) this.release(i, entry);
    const current = this.batch.trials[index];
    // Give the visible clip the connection first. Future downloads must never
    // hold the current Play button behind a large lookahead clip.
    if (current && !current.started && !current.answered) {
      this.get(index).catch(() => {});
      if (!this.entries.get(index)?.ready) return;
    }
    let reserved = 0;
    for (let i = index; i < Math.min(this.batch.total, index + 8); i++) {
      const trial = this.batch.trials[i];
      if (trial.started || trial.answered) continue;
      const bytes = trial.bytes || 80 * 1024 * 1024;
      if (i > index && reserved + bytes > this.maxBytes) break;
      reserved += bytes;
      this.get(i).catch(() => {});
    }
  }
  get(index) {
    if (this.closed) return Promise.reject(Error('Media window closed'));
    if (this.entries.has(index)) return this.entries.get(index).promise;
    const entry = {index, controller: new AbortController(), cancelled: false, ready: false,
      phase: 'queued', loaded: 0, total: this.batch.trials[index].bytes || 0};
    entry.promise = new Promise((resolve, reject) => { entry.resolve = resolve; entry.reject = reject; });
    this.entries.set(index, entry); this.queue.push(entry); this.pump();
    return entry.promise;
  }
  pump() {
    this.queue.sort((a, b) => a.index - b.index);
    while (!this.closed && this.active < 3 && this.queue.length) {
      const entry = this.queue.shift();
      if (entry.cancelled) continue;
      this.active++;
      this.prepare(entry).then(() => entry.resolve(entry), error => {
        this.release(entry.index, entry); entry.reject(error);
      }).finally(() => { this.active--; this.changed(); this.pump(); });
    }
  }
  async blob(media, sha, signal, expectedBytes = 0, progress = () => {}) {
    progress('downloading', 0, expectedBytes);
    const response = await this.fetchMedia(media, signal);
    const total = expectedBytes || Number(response.headers.get('Content-Length')) || 0;
    let blob;
    if (response.body?.getReader) {
      const reader = response.body.getReader(), chunks = [];
      let loaded = 0;
      try {
        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          loaded += value.byteLength;
          if (expectedBytes && loaded > expectedBytes) throw Error('Video size check failed');
          chunks.push(value); progress('downloading', loaded, total);
        }
      } catch (error) { await reader.cancel().catch(() => {}); throw error; }
      finally { reader.releaseLock(); }
      blob = new Blob(chunks, {type: response.headers.get('Content-Type') || 'application/octet-stream'});
    } else { blob = await response.blob(); }
    if (expectedBytes && blob.size !== expectedBytes) throw Error('Video size check failed');
    progress('verifying', blob.size, total || blob.size);
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', await blob.arrayBuffer())), x => x.toString(16).padStart(2, '0')).join('');
    if (hash !== sha) throw Error('Media integrity check failed');
    return blob;
  }
  async prepare(entry) {
    const trial = this.batch.trials[entry.index], signal = entry.controller.signal;
    const began = performance.now();
    const blob = await this.blob(trial.media, trial.sha256, signal, trial.bytes, (phase, loaded, total) => {
      entry.phase = phase; entry.loaded = loaded; entry.total = total;
      this.changed();
    });
    if (trial.bytes && blob.size !== trial.bytes) throw Error('Video size check failed');
    if (entry.cancelled) throw Error('Media window closed');
    entry.bytes = blob.size; entry.download_verify_ms = performance.now() - began;
    entry.url = URL.createObjectURL(blob);
    const v = document.createElement('video'); entry.video = v;
    v.width = trial.width; v.height = trial.height;
    v.preload = 'auto'; v.muted = true; v.playsInline = true;
    v.disablePictureInPicture = true; v.disableRemotePlayback = true;
    v.tabIndex = -1; v.setAttribute('aria-hidden', 'true');
    const decodeStart = performance.now();
    entry.phase = 'decoding'; this.changed();
    await new Promise((resolve, reject) => {
      const finish = error => {
        clearTimeout(timer); v.oncanplaythrough = v.onerror = null;
        signal.removeEventListener('abort', aborted);
        error ? reject(error) : resolve();
      };
      const aborted = () => finish(Error('Media window closed'));
      const timer = setTimeout(() => finish(Error('Video decoding timed out. Retry loading.')), 30000);
      v.oncanplaythrough = () => finish();
      v.onerror = () => finish(Error('This browser cannot decode the video.'));
      signal.addEventListener('abort', aborted, {once: true});
      v.src = entry.url; v.load();
      if (v.readyState >= 4) finish();
    });
    if (v.videoWidth !== trial.width || v.videoHeight !== trial.height || Math.abs(v.duration - trial.duration) > .12) throw Error('Video dimensions or duration differ from the assigned source');
    entry.decode_ms = performance.now() - decodeStart;
    const ref = trial.reference || this.batch.reference;
    if (ref) {
      entry.phase = 'reference'; this.changed();
      const imageBlob = await this.blob(ref.media, ref.sha256, signal);
      if (entry.cancelled) throw Error('Media window closed');
      entry.referenceURL = URL.createObjectURL(imageBlob);
      entry.reference = new Image(); entry.reference.src = entry.referenceURL;
      await entry.reference.decode();
      if (entry.reference.naturalWidth !== 160 || entry.reference.naturalHeight !== 160) throw Error('Reference dimensions differ from 160 × 160');
    }
    if (entry.cancelled) throw Error('Media window closed');
    entry.ready = true; entry.phase = 'ready'; entry.ready_at = performance.now();
    this.window(this.current); this.changed();
  }
  release(index, entry = this.entries.get(index)) {
    if (!entry) return;
    entry.cancelled = true; entry.controller.abort();
    if (entry.video) { entry.video.pause(); entry.video.removeAttribute('src'); entry.video.load(); }
    for (const url of [entry.url, entry.referenceURL]) if (url) URL.revokeObjectURL(url);
    if (this.entries.get(index) === entry) this.entries.delete(index);
    entry.reject?.(Error('Media window closed'));
  }
  close() { this.closed = true; for (const [i, entry] of this.entries) this.release(i, entry); this.queue = []; }
}
