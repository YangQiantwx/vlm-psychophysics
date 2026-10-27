"""Explicit operator entry point for generation of the T3-only follow-up panel.

The native generator requires both tasks and 128 representative source states.
Only its row ordering is adapted here; rendering, admission, encoding, checking,
publication and completion remain the native implementation.
"""
from __future__ import annotations

import os

from classic_precision import ready
from classic_precision.manifest import selected_rows, state_key


FLAG = 'CLASSIC_PRECISION_T3_ONLY_GENERATION_COMPAT'
_native_generation_order = ready.generation_order


def generation_order(manifest, shard_count=3):
    """Keep native ordering unless this process opts into a T3-only panel."""
    if (os.environ.get(FLAG) != '1'
            or {row['task'] for row in manifest['rows']} != {'T3'}
            or manifest.get('counts', {}).get('T2') != 0
            or manifest.get('protocol', {}).get('counts', {}).get('T2_total') != 0):
        return _native_generation_order(manifest, shard_count)

    shards = [selected_rows(manifest, index, shard_count) for index in range(shard_count)]
    priority = []
    for shard in shards:
        for task in ('T2', 'T3'):
            first = next((row for row in shard if row['task'] == task), None)
            if first is not None:
                priority.append(first)
    # This is native representatives()'s insertion order with the panel's
    # actual source-state count, rather than the original 128-state assertion.
    unique = {}
    for row in manifest['rows']:
        unique.setdefault(state_key(row), row)
    priority.extend(unique.values())
    for index in range(max(map(len, shards))):
        priority.extend(shard[index] for shard in shards if index < len(shard))
    seen = set()
    return [row for row in priority
            if not (row['sample_id'] in seen or seen.add(row['sample_id']))]


def install():
    if os.environ.get(FLAG) != '1':
        return False
    ready.generation_order = generation_order
    return True


def main(argv=None):
    install()
    from classic_precision.__main__ import main as native_main
    return native_main(argv)


if __name__ == '__main__':
    main()
