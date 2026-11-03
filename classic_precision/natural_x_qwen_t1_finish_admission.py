"""Exact seven-item Qwen T1 continuation after the explicit pause correction."""
from pathlib import Path

from . import natural_x_operations as op
from .refinement_worker import descriptor, read, require, verified

MODULE = 'classic_precision.natural_x_qwen_t1_finish_admission'
OWNER = '01a196a0-b6db-7080-bad8-d62df752cd4a'


def gates(queue):
    require(queue['model'] == 'qwen', 'Only Qwen T1 continuation is authorized')
    require(queue['admission_adapter'] == descriptor(__file__), 'Admission adapter changed')
    auth = verified(queue['finish_authorization'])
    require(auth['schema'] == 'natural-qwen-t1-finish7-authorization-v1'
            and auth['status'] == 'EXPLICIT_HUMAN_AUTHORIZED'
            and auth['owner_chat'] == OWNER and auth['maximum_new_model_calls'] == 7
            and auth['tasks'] == ['T1'] and auth['automatic_retry'] is False
            and auth['admission_adapter'] == queue['admission_adapter']
            and queue['profiles'] == auth['profiles']
            and queue['source_approval'] == auth['source_approval']
            and queue['resume_receipt'] == auth['resume_receipt'], 'Foreign finish authorization')
    fence = read(queue['fence'])
    require(not fence['paused'] and fence['generation'] == queue['fence_generation'] == 1, 'Fenced/stale continuation')
    require(len(queue['profiles']) == 1, 'Only one exact continuation')
    p = verified(queue['profiles'][0]); parent = verified(auth['parent_profile'])
    require(p['card'] == parent['card'] == 'fourier1'
            and p['natural_x']['parent'] == auth['parent_profile']
            and p['assigned_ids'] == parent['assigned_ids'][1913:]
            and len(p['assigned_ids']) == p['maximum_model_calls'] == 7
            and op.runner.settings(p) == op.runner.settings(parent)
            and p['physical_lock'] == parent['physical_lock'], 'Continuation escaped exact seven-item suffix')
    _, catalog, _ = op.runner.admission(queue['source_approval'], p['source_catalog'])
    require(all(catalog['items'][sid]['task'] == 'T1' for sid in p['assigned_ids']), 'T3/foreign task is forbidden')
    predecessor = auth['predecessor_queue']
    require(queue['predecessors'] == [{'queue': predecessor}], 'Wrong actual predecessor')
    arc = op.Archive(queue['workspace_root'])
    _, _, state = op.closed_queue(arc, predecessor['path'], live=True)
    terminal = read(Path(parent['output']) / 'result.json')
    require(descriptor(predecessor['path']) == predecessor and state['status'] == 'PAUSED'
            and terminal['status'] == 'PAUSED' and len(terminal['completed_ids']) == 1913
            and not terminal['UNCERTAIN_ids'], 'Parent has not safely closed at the exact prefix')
    arc.unchanged()


def main():
    op.MODULE = MODULE
    op.gates = gates
    op.main()


if __name__ == '__main__':
    main()
