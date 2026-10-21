"""Small CPU/RAM admission budgets for generation, checking and calibration."""
from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess


def single_thread_environment():
    return {name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}


def configure_threads():
    os.environ.update(single_thread_environment())


def available_memory_gib():
    path = Path('/proc/meminfo')
    if path.exists():
        return next(int(line.split()[1])/1024**2 for line in path.read_text().splitlines()
                    if line.startswith('MemAvailable:'))
    # CPU-only macOS use. Linux workers use MemAvailable above.
    text = subprocess.run(['vm_stat'], check=True, capture_output=True, text=True).stdout
    size = int(re.search(r'page size of (\d+) bytes', text)[1])
    pages = sum(int(re.search(r'^'+re.escape(name)+r':\s+(\d+)', text, re.M)[1])
                for name in ('Pages free', 'Pages inactive', 'Pages speculative'))
    return pages*size/2**30


def available_cores():
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count() or 1


def allocation(max_workers=None, memory_gib=32, reserve_gib=4, calibration_active=False,
               buffer_rows=0, high_water=200, disk_free_gib=None, disk_floor_gib=10):
    """Admission estimates, not RSS guarantees; no process is killed to fit them."""
    if memory_gib <= 0 or reserve_gib < 0 or (max_workers is not None and max_workers < 1):
        raise ValueError('Positive CPU memory/worker budgets and nonnegative reserve required')
    cores = min(max_workers or max(1, available_cores()-2), available_cores())
    free = available_memory_gib()
    calibration = 8. if calibration_active else 0.
    # Retain a CPU-job allowance for the single active calibrator, and a free
    # RAM floor. GPU expert/KV offload is already reflected in MemAvailable and
    # checked separately by model resource admission; do not subtract it twice.
    budget = min(memory_gib-calibration, free-reserve_gib)
    workers = min(cores, max(0, int(budget/.5)))
    reason = 'READY_DEMAND' if buffer_rows < high_water else 'BUFFER_SUFFICIENT'
    if buffer_rows >= high_water:
        workers = min(workers, max(1, cores//4))  # Always continue finite production.
    if disk_free_gib is not None and disk_free_gib <= disk_floor_gib+2:
        workers = min(workers, 1)
        reason = 'DISK_PRESSURE'
    if disk_free_gib is not None and disk_free_gib <= disk_floor_gib:
        workers, reason = 0, 'DISK_RESERVE_REACHED'
    if budget < .5:
        workers, reason = 0, 'WAITING_FOR_RAM'
    return {'workers': workers, 'core_limit': cores, 'available_ram_gib': free,
            'cpu_memory_budget_gib': memory_gib, 'free_ram_reserve_gib': reserve_gib,
            'calibration_allowance_gib': calibration, 'per_encoding_worker_budget_gib': .5,
            'buffer_rows': buffer_rows, 'reason': reason, 'threads_per_child': 1}


def ready_buffer(manifest, run_dir, shard_count=3):
    """Produced minus fastest locally observed consumer, per frozen shard."""
    import json
    from .ready import read_records
    run_dir = Path(run_dir)
    ready, _, _ = read_records(run_dir/'ready.jsonl')
    ids = {row['sample_id']: row['source_parameters']['onset_index'] % shard_count
           for row in manifest['rows']}
    counts = [sum(ids[sid] == shard for sid in ready) for shard in range(shard_count)]
    consumed = [0]*shard_count
    for path in (run_dir/'outputs').glob('*/outputs.jsonl'):
        current = [0]*shard_count
        for line in path.read_bytes().split(b'\n')[:-1]:
            row = json.loads(line)
            if row['sample_id'] in ids:
                current[ids[row['sample_id']]] += 1
        consumed = [max(a, b) for a, b in zip(consumed, current)]
    return min(max(0, a-b) for a, b in zip(counts, consumed))
