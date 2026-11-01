"""Isolated recorder plumbing: no Qwen imports or numerical model changes."""
import hashlib
import math
import os
from pathlib import Path
import pwd
import subprocess
import time
from . import gemma_x_recheck_trace as trace
from .storage import _append, owner_identity

ORIGINAL_FINISH = trace.finish
ORIGINAL_ADAPTER = trace.AdapterObservation
begin, resource_observation = trace.begin, trace.resource_observation


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return '-Infinity' if value == -math.inf else 'Infinity' if value == math.inf else 'NaN'
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def finish(worker):
    # Preserve exact binary tensor bytes. Only encode nonfinite human-readable
    # metadata as named strings so strict JSON can represent masked logits.
    state = worker._gemma_recheck_trace
    state['records'] = json_safe(state['records'])
    return ORIGINAL_FINISH(worker)


def shared_observation(p, label):
    """Bounded read-only telemetry; no context creation or GPU work submission."""
    result = {'label': label, 'owner': owner_identity(), 'monotonic_seconds': time.monotonic(),
              'unix_seconds': time.time(), 'gpu_uuid': p['gpu_uuid'], 'other_compute': [], 'errors': []}
    try:
        obs = subprocess.run(['nvidia-smi', 'pmon', '-i', p['environment']['CUDA_VISIBLE_DEVICES'],
                              '-s', 'um', '-c', '1'], capture_output=True, text=True, timeout=12)
        result.update(returncode=obs.returncode, stdout=obs.stdout, stderr=obs.stderr)
        for line in obs.stdout.splitlines():
            cells = line.split()
            if len(cells) < 5 or not cells[0].isdigit() or cells[2] != 'C' or not cells[1].isdigit(): continue
            pid = int(cells[1]); proc = Path('/proc') / str(pid)
            try:
                uid = proc.stat().st_uid; ticks = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
                row = {'pid': pid, 'uid': uid, 'user': pwd.getpwuid(uid).pw_name,
                       'process_start_ticks': ticks, 'sm_percent': int(cells[3]) if cells[3].isdigit() else None,
                       'memory_percent': cells[4]}
                if uid != os.getuid(): result['other_compute'].append(row)
            except (OSError, KeyError): pass
    except Exception as error:
        result['errors'].append(str(error))
    result['other_account_compute_active'] = any((x['sm_percent'] or 0) > 0 for x in result['other_compute'])
    return result


def prepare_observed(self, item, content):
    self.sid = item['sample_id']; self.active = True; self.cpu_tensors = []
    _append(self.root / 'shared-gpu-observations.jsonl', shared_observation(self.p, self.sid))
    _append(self.root / 'diagnostic-resources.jsonl', {'sample_id':self.sid,**resource_observation(self.p)})
    self.engine.collective_rpc(begin, args=(self.sid,))
    result = self.original.prepare(item, content)
    frames = result['request']['multi_modal_data']['video'][0][0]
    _append(self.root / 'diagnostic-submitted-frames.jsonl', {
        'sample_id': self.sid, 'shape': list(frames.shape), 'dtype': str(frames.dtype),
        'sha256': hashlib.sha256(memoryview(frames).cast('B')).hexdigest(),
        'receipt': result['receipt']})
    return result

