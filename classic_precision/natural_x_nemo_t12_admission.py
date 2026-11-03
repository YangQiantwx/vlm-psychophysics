"""Explicit finite Nemotron T1/T2 priority using the frozen natural-X operator.

Only the model-order gate changes. Qualified science, resources, original lease
identity, request-boundary pause, real waits and no-retry behavior stay intact.
"""
from collections import Counter
from pathlib import Path

from . import natural_x_operations as op
from .refinement_worker import descriptor, read, require, verified
from .storage import digest

MODULE = 'classic_precision.natural_x_nemo_t12_admission'
OWNER = '01a196a0-b6db-7080-bad8-d62df752cd4a'
SOURCE = '40d578e0359fd05235286e547e2215ea051310077cd3e9066295f928337e27e3'


def gates(queue):
    require(queue['model'] == 'nemo', 'Only Nemotron is authorized here')
    require(queue['admission_adapter'] == descriptor(__file__), 'Admission adapter changed')
    auth = verified(queue['early_t12_authorization'])
    require(auth['schema'] == 'natural-nemo-t12-priority-authorization-v1'
            and auth['status'] == 'EXPLICIT_HUMAN_AUTHORIZED'
            and auth['owner_chat'] == OWNER
            and auth['source_identity_sha256'] == SOURCE
            and auth['tasks'] == ['T1', 'T2']
            and auth['new_counts'] == {'T1': 3840, 'T2': 3840}
            and auth['maximum_new_model_calls'] == 7680
            and auth['T3_dispatch'] is False
            and auth['automatic_retry'] is False
            and auth['admission_adapter'] == queue['admission_adapter'], 'Foreign authorization')
    fence = read(queue['fence'])
    require(not fence['paused'] and fence['generation'] == queue['fence_generation'], 'Fenced/stale queue')
    profiles = [verified(ref) for ref in queue['profiles']]
    card = profiles[0]['card']
    require(card in ('fourier0', 'fourier1')
            and queue['profiles'] == auth['queue_profiles'][card]
            and queue['source_approval'] == auth['source_approval']
            and queue['fence'] == auth['fence'], 'Queue escaped authorized card/profiles')
    approval, catalog, _ = op.runner.admission(queue['source_approval'], profiles[0]['source_catalog'])
    require(approval['computed_source_identity_sha256'] == SOURCE, 'Qualified source changed')
    current = verified(auth['current_catalog'])['items']
    exact = {sid for sid, item in catalog['items'].items() if item['task'] in ('T1', 'T2')}
    require(all(current.get(sid) == catalog['items'][sid] for sid in exact)
            and digest({sid: catalog['items'][sid] for sid in sorted(exact)}) == auth['T12_projection_sha256'],
            'Current T1/T2 source projection changed')
    all_profiles = [verified(ref) for refs in auth['queue_profiles'].values() for ref in refs]
    ids = [sid for profile in all_profiles for sid in profile['assigned_ids']]
    require(len(ids) == len(set(ids)) == 7680 and set(ids) == exact
            and Counter(catalog['items'][sid]['task'] for sid in ids) == {'T1': 3840, 'T2': 3840},
            'Scope is not the exact disjoint current T1/T2 universe')
    require(queue['physical_predecessor_closure']['path'] == auth['predecessor_closure_paths'][card],
            'Wrong card closure evidence path')
    closure = verified(queue['physical_predecessor_closure'])
    require(closure['status'] == 'PASS_PAUSED_ORIGINAL_ANSWERS_TRUE_WAITS_OWNERS_EXITED_SAME_INODE_LEASES_FREE'
            and closure['no_unknown_claims'] is True, 'Missing scientific predecessor closure')
    require(queue['physical_predecessor_return']['path'] == auth['predecessor_return_paths'][card]
            and verified(queue['physical_predecessor_return'])['status'] == 'PASS_ALL_RETURNED_SHA256',
            'Missing exact-card original answer/evidence SHA return')
    predecessor = auth['physical_predecessor_queues'][card]
    require(queue['predecessors'] == [{'queue': predecessor}], 'Wrong physical-card predecessor')
    arc = op.Archive(queue['workspace_root'])
    old, old_profiles, state = op.closed_queue(arc, predecessor['path'], live=True)
    require(descriptor(predecessor['path']) == predecessor
            and state['status'] == auth['predecessor_required_states'][card]
            and any(entry['queue'] == predecessor and entry['state'] == state['status'] for entry in closure['queues'])
            and all(p['card'] == card and p['physical_lock'] == profiles[0]['physical_lock'] for p in old_profiles),
            'Exact physical-card predecessor has not genuinely closed')
    expected = auth['predecessor_completed_counts'][card]
    require(len(expected) == len(old_profiles), 'Predecessor worker list changed')
    for profile, count in zip(old_profiles, expected):
        terminal = read(Path(profile['output']) / 'result.json')
        require(len(terminal['completed_ids']) == count and not terminal['UNCERTAIN_ids']
                and terminal['status'] in ('PAUSED', 'COMPLETE'), 'Predecessor prefix changed')
    arc.unchanged()


def main():
    op.MODULE = MODULE
    op.gates = gates
    op.main()


if __name__ == '__main__':
    main()
