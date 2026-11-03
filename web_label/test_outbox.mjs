import test from 'node:test';
import assert from 'node:assert/strict';
import {Outbox} from './frontend/label/outbox.js';

class Storage {
  data = new Map();
  get length() { return this.data.size; }
  key(index) { return [...this.data.keys()][index] ?? null; }
  getItem(key) { return this.data.get(key) ?? null; }
  setItem(key, value) { this.data.set(key, value); }
  removeItem(key) { this.data.delete(key); }
}
const payload = (index, extra = {}) => ({batch_id: 'batch', index, ...extra});
test('offline records survive reload, upload in order, and tolerate an acknowledged retry', async () => {
  const storage = new Storage(), a = new Outbox(storage, 'identity');
  a.add('/api/playback', payload(0, {evidence: {ended: true}}));
  a.add('/api/answer', payload(0, {answer: 'LEFT'}));
  await assert.rejects(a.drain(async () => { throw Error('offline'); }));
  assert.equal(a.entries().length, 2);
  const restored = new Outbox(storage, 'identity'), sent = [];
  await restored.drain(async (path, body) => { sent.push([path, body.index]); return {saved: true, duplicate: true}; });
  assert.deepEqual(sent, [['/api/playback', 0], ['/api/answer', 0]]);
  assert.equal(restored.entries().length, 0);
});
test('a slow acknowledgement cannot erase a later answer or a second identity', async () => {
  const storage = new Storage(), a = new Outbox(storage, 'one'), b = new Outbox(storage, 'two');
  a.add('/api/playback', payload(0)); b.add('/api/answer', payload(0, {answer: 'RIGHT'}));
  let release;
  const wait = new Promise(resolve => { release = resolve; }), sent = [];
  const task = a.drain(async (path, body) => { sent.push([path, body.index]); if (sent.length === 1) await wait; return {saved: true}; });
  a.add('/api/answer', payload(0, {answer: 'LEFT'}));
  a.add('/api/playback', payload(1)); a.add('/api/answer', payload(1, {answer: 'RIGHT'}));
  assert.equal(a.drain(() => { throw Error('parallel drain'); }), task);
  release(); await task;
  assert.equal(sent.length, 4); assert.equal(a.entries().length, 0); assert.equal(b.entries().length, 1);
});
test('failed durable storage and conflicting local answers fail closed', () => {
  const storage = new Storage(), a = new Outbox(storage, 'one');
  a.add('/api/answer', payload(0, {answer: 'LEFT'}));
  assert.throws(() => a.add('/api/answer', payload(0, {answer: 'RIGHT'})), /different response/);
  storage.setItem = () => { throw Error('quota'); };
  assert.throws(() => a.add('/api/answer', payload(1, {answer: 'RIGHT'})), /quota/);
  assert.equal(a.entries().length, 1);
});
test('migrate both legacy pending events without losing them if offline', async () => {
  const storage = new Storage();
  storage.setItem('cvpr-label-outbox:old:playback', JSON.stringify(payload(0)));
  storage.setItem('cvpr-label-outbox:old', JSON.stringify(payload(0, {answer: 'LEFT'})));
  const a = new Outbox(storage, 'old');
  await assert.rejects(a.drain(async () => ({saved: false})));
  assert.equal(a.entries().length, 2); assert.equal(storage.getItem('cvpr-label-outbox:old'), null);
});
