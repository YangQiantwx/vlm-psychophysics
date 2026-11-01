"""Authorized Radon-only autotune-OFF comparison within the unused U192 budget.

Original ON runs and sealed modules remain immutable. OFF versus ON changes are
measured; only an OFF cold-to-cold difference or input/config error stops replay.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

from . import gemma_x_recheck as core
from . import gemma_x_recheck_recovery as recovery
from . import gemma_x_recheck_trace as trace
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, _append

BASE = core.NAME
NAME = BASE + '/autotune-off1'
MODULE = 'classic_precision.gemma_x_recheck_autotune_off'
ORIGINAL_SCIENCE = core.science
ORIGINAL_RUN = core.run
ORIGINAL_IDENTITY = core.full.worker_identity
FIELDS = ['raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason',
          'truth', 'question', 'generation']


def science(p, resolve=verified):
    require(p['card'] == 'radon0', 'OFF experiment is Radon-only')
    spec = p['recheck']['autotune_off']
    require(spec['entrypoint']['sha256'] == file_sha256(__file__), 'OFF entrypoint changed')
    auth = resolve(spec['authorization'])
    require(auth['status'] == 'AUTHORIZED' and auth['maximum_Radon_calls'] == 192
            and auth['new_budget_added'] == 0 and auth['preserve_original_answers']
            and auth['only_scientific_change'] == {'kernel_config.enable_flashinfer_autotune': False},
            'OFF authorization differs')
    recovery.validate_closure(resolve(spec['zero_call_closure']))
    require(p['engine_kwargs'].get('kernel_config') == {'enable_flashinfer_autotune': False},
            'OFF must be the sole engine change')
    projected = copy.deepcopy(p)
    projected['engine_kwargs'].pop('kernel_config')
    # Validate every original scientific/input/SDK field against the immutable ON
    # template, then separately enforce the explicit one-setting amendment above.
    ORIGINAL_SCIENCE(projected, resolve)


def identity(worker, model):
    value = ORIGINAL_IDENTITY(worker, model)
    require('enable_flashinfer_autotune=False' in value['kernel_config'],
            'Actual engine did not disable FlashInfer autotuning')
    return value


def compare(raw, on_reference, off_reference=None):
    sid = raw['sample_id']; expected = on_reference['by_id'][sid]
    on_diff = [k for k in FIELDS if raw[k] != expected[k]]
    if raw['actual_prompt_token_ids'] != on_reference['actual_prompt_token_ids']:
        on_diff.append('actual_prompt_token_ids')
    invariants = [k for k in ['truth', 'question', 'generation', 'actual_prompt_token_ids'] if k in on_diff]
    off_diff = None
    if off_reference is not None:
        previous = off_reference[sid]
        off_diff = [k for k in FIELDS + ['actual_prompt_token_ids'] if raw[k] != previous[k]]
    return {'sample_id': sid, 'ON_OFF_full_output_prompt_equal': not on_diff,
            'ON_OFF_different_fields': on_diff, 'invariant_differences': invariants,
            'OFF_cold_pair_checked': off_diff is not None,
            'OFF_cold_full_output_prompt_equal': None if off_diff is None else not off_diff,
            'OFF_cold_different_fields': off_diff,
            'accepted': not invariants and not off_diff, 'raw_digest': digest(raw)}


def cold1_reference(p):
    root = Path(p['workspace_root'])
    queue = read(root/'plans'/NAME/'radon0-recheck-queue.json')
    previous = verified(queue['profiles'][0]); science(previous)
    out = Path(previous['output']); wait = read(out.with_suffix('.wait.json'))
    require(read(out/'result.json')['status'] == 'COMPLETE' and wait['returncode'] == 0
            and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released']
            and wait['profile_sha256'] == digest(previous)
            and wait['completed_ids'] == previous['assigned_ids'], 'OFF cold1 is not truly closed')
    rows = [json.loads(x) for x in (out/'raw.jsonl').read_text().splitlines()]
    comparisons = [json.loads(x) for x in (out/'output-equivalence.jsonl').read_text().splitlines()]
    require([r['sample_id'] for r in rows] == [r['sample_id'] for r in comparisons]
            == previous['assigned_ids'] and all(x['accepted'] for x in comparisons),
            'OFF cold1 raw/input comparison is incomplete')
    return {r['sample_id']: r for r in rows}


class Observation(trace.AdapterObservation):
    def __init__(self, original, p, engine, state):
        super().__init__(original, p, engine, state)
        self.off_reference = cold1_reference(p) if p['cold'] == 2 else None

    def validate(self, answer):
        result = self.original.validate(answer)
        evidence = self.engine.collective_rpc(trace.finish); self.active = False
        _append(self.root/'diagnostic-requests.jsonl', {'sample_id': self.sid, 'workers': evidence})
        cpu = [h for x in self.cpu_tensors if x['shape'][1:] == [trace.PATCHES, 768]
               for h in x.get('per_frame_sha256', [])]
        gpu = [h for x in evidence for h in x['GPU_pixel_frame_hashes']]
        match = len(cpu) == 180 and cpu == gpu
        _append(self.root/'diagnostic-input-equivalence.jsonl', {'sample_id': self.sid,
            'actual_CPU_GPU_per_frame_bytes_equal': match, 'CPU_frames': len(cpu), 'GPU_frames': len(gpu)})
        require(match and all(x['vision_chunks'] > 0 and x['greedy_steps'] > 0
                and not x['observation_errors'] for x in evidence),
                'Missing or mismatched actual GPU input/decision observation; raw retained')
        raw = self.state['raw']  # Original runner persisted this before validate().
        row = compare(raw, self.reference, self.off_reference)
        _append(self.root/'output-equivalence.jsonl', row)
        if not row['accepted']:
            atomic_json(self.root/'reproduction-failure.json', {**row,
                'raw_persisted_before_comparison': True, 'automatic_retry': False})
        require(row['accepted'], 'OFF_COLD_OR_INPUT_DIFFERENCE_RAW_RETAINED')
        return result


def run(profile):
    p = read(profile); science(p)
    if p['cold'] == 2: cold1_reference(p)
    from . import inference
    original = inference.resource_snapshot
    inference.resource_snapshot = recovery.recorded_snapshot(original,
        Path(p['output']).with_suffix('.resource-observations.jsonl'))
    try: return ORIGINAL_RUN(profile)
    finally: inference.resource_snapshot = original


@contextmanager
def namespace():
    old = core.NAME, core.MODULE, core.science, core.run, core.full.worker_identity, trace.AdapterObservation
    old_recovery = recovery.NAME, recovery.MODULE, recovery.science
    core.NAME, core.MODULE, core.science, core.run = NAME, MODULE, science, run
    core.full.worker_identity, trace.AdapterObservation = identity, Observation
    recovery.NAME, recovery.MODULE, recovery.science = NAME, MODULE, science
    try: yield
    finally:
        core.NAME, core.MODULE, core.science, core.run, core.full.worker_identity, trace.AdapterObservation = old
        recovery.NAME, recovery.MODULE, recovery.science = old_recovery


def prepare(root):
    root = Path(root); dest = root/'plans'/NAME; parent = root/'plans'/BASE
    auth_path = parent/'autotune-off-authorization.json'; auth = read(auth_path)
    require(auth['status'] == 'AUTHORIZED' and auth['maximum_Radon_calls'] == 192,
            'Explicit OFF authorization missing')
    require(not dest.exists(), 'OFF plan already prepared; never reseal')
    closure = root/'runs/reproducibility'/BASE/'radon0-zero-call-startup-closed.json'
    recovery.validate_closure(read(closure))
    # Both ON deployments are retired; the second was sealed but never launched.
    donor_queue = parent/'radon0-recheck-queue.json'
    require(file_sha256(donor_queue) == recovery.DONOR_QUEUE_SHA, 'Original donor queue changed')
    repair = root/'plans'/recovery.NAME
    require(recovery.NAME != NAME, 'Preparation must run outside OFF namespace')
    require(all(x.exists() for x in core.paths(root,'radon0').values()),
            'Original deployment history missing')
    repair_run = root/'runs/reproducibility'/recovery.NAME
    require(not any(repair_run.glob('radon0-recheck-queue.launch*.json')), 'ON repair was launched')
    for qpath in (donor_queue, repair/'radon0-recheck-queue.json'):
        for ref in read(qpath)['profiles']:
            p = verified(ref); control = read(p['control_file'])
            require(control['paused'] and control['generation'] == 1
                    and not any(Path(p['claim_root']).glob('*.json')),
                    'ON donor is not fenced with zero claims')
    dest.mkdir(parents=True)
    for ref in read(donor_queue)['profiles']:
        p = copy.deepcopy(verified(ref)); worker = p['worker']
        p['engine_kwargs']['kernel_config'] = {'enable_flashinfer_autotune': False}
        p.update(output=str(root/'runs/reproducibility'/NAME/worker),
                 claim_root=str(root/'runs/reproducibility'/NAME/'claims'/worker),
                 control_file=str(dest/(worker+'-control.json')), pause_file=str(dest/(worker+'.pause')))
        p.pop('worker_token', None)
        p['recheck']['autotune_off'] = {'entrypoint': descriptor(__file__),
            'authorization': descriptor(auth_path), 'zero_call_closure': descriptor(closure),
            'comparison_policy': 'OFF cold1 versus original ON measured; OFF cold2 must match OFF cold1'}
        write_new(dest/(worker+'-draft.json'), p)
    (dest/'context-predecessor-summary-ref.json').write_bytes((parent/'context-predecessor-summary-ref.json').read_bytes())
    with namespace(): return core.stage(root, 'radon0')


def main():
    command = sys.argv[1]
    if command in ('prepare', 'resource-window', 'launch'):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('command'); parser.add_argument('--root', required=True); parser.add_argument('--window')
        a = parser.parse_args()
        if command == 'prepare': value = prepare(a.root)
        else:
            with namespace():
                value = recovery.resource_window(a.root) if command == 'resource-window' else recovery.launch(a.root,a.window)
        print(json.dumps(value)); return
    require(command in ('queue-supervisor', 'queue', 'supervise', 'run'), 'Unsupported OFF command')
    with namespace(): core.main()


if __name__ == '__main__':
    main()
