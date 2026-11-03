// Independent durable records: an acknowledgement may delete only its own entry.
// The server remains authoritative and rejects conflicting/repeated playbacks.
export class Outbox {
  constructor(storage, token, changed = () => {}) {
    this.storage = storage;
    this.prefix = `cvpr-label-events-v2:${token}:`;
    this.changed = changed;
    this.running = null;
    for (const [suffix, path] of [[':playback', '/api/playback'], ['', '/api/answer']]) {
      const key = `cvpr-label-outbox:${token}${suffix}`;
      const raw = storage.getItem(key);
      if (raw) { this.add(path, JSON.parse(raw)); storage.removeItem(key); }
    }
  }
  entries() {
    const rows = [];
    for (let i = 0; i < this.storage.length; i++) {
      const key = this.storage.key(i);
      if (key?.startsWith(this.prefix)) rows.push({key, ...JSON.parse(this.storage.getItem(key))});
    }
    return rows.sort((a, b) => a.created - b.created || a.payload.index - b.payload.index || a.path.localeCompare(b.path) * -1);
  }
  add(path, payload) {
    const key = `${this.prefix}${payload.batch_id}:${payload.index}:${path}`;
    const old = this.storage.getItem(key);
    if (old && JSON.stringify(JSON.parse(old).payload) !== JSON.stringify(payload)) throw Error('A different response is already saved on this device.');
    if (!old) this.storage.setItem(key, JSON.stringify({path, payload, created: Date.now()}));
    // setItem is synchronous and may throw (quota/private-storage failure).
    // Never advance a trial unless this exact record can be read back.
    if (!this.storage.getItem(key)) throw Error('This browser could not save your response.');
    this.changed(this.entries().length);
  }
  drain(send) {
    if (this.running) return this.running;
    this.running = (async () => {
      for (;;) {
        const entry = this.entries()[0];
        if (!entry) return;
        const response = await send(entry.path, entry.payload);
        if (response.saved !== true) throw Error('The server has not acknowledged this record.');
        this.storage.removeItem(entry.key);
        this.changed(this.entries().length);
      }
    })().finally(() => { this.running = null; });
    return this.running;
  }
}
