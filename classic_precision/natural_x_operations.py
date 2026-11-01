"""Finite natural-X queues and request-boundary pause / exact no-retry resume.

All mutations are confined to the explicitly selected new natural run set.
No classic Gemma controls or historical raw/claims/waits are ever rewritten.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import copy
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

from . import natural_x_runner as runner
from . import official_x_results as evidence
from .gemma_x_full import Archive as BaseArchive
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity

MODULE = 'classic_precision.natural_x_operations'
WORKER = 'classic_precision.natural_x_runner'


class Archive(BaseArchive):
    def __init__(self, root):
        super().__init__(Path(root).resolve()); self.bindings = {}
    def capture(self, path, sha=None):
        path = self.path(path)
        require(path.is_file() and not path.is_symlink(), 'Missing/symlink evidence: ' + str(path))
        actual = file_sha256(path); require(sha is None or actual == sha, 'Evidence SHA changed: ' + str(path))
        require(str(path) not in self.bindings or self.bindings[str(path)] == actual, 'Evidence changed during reconciliation')
        self.bindings[str(path)] = actual
        return {'path': str(path), 'sha256': actual}
    def ref(self, ref):
        self.capture(ref['path'], ref['sha256']); return read(self.path(ref['path']))
    def unchanged(self):
        for path, sha in self.bindings.items(): require(file_sha256(path) == sha, 'Reconciliation evidence changed')


def journal(path):
    path = Path(path)
    if not path.exists(): return []
    rows = []
    with path.open('rb') as stream:
        for line in stream:
            require(line.endswith(b'\n') and line.strip(), 'Dirty/partial journal: ' + str(path))
            row = json.loads(line); require(isinstance(row, dict), 'Non-object journal row'); rows.append(row)
    return rows


def paths(queue_path, queue):
    base = Path(queue['output'])
    return {key: base / (Path(queue_path).stem + '.' + key + '.json')
            for key in ('launch-intent', 'launch', 'owner', 'wait')}


def live_owner(owner):
    from .gemma_x_fourier_operations import observe_owner
    return observe_owner(owner)


def lease_free(profile):
    lock = profile['physical_lock']
    with open(lock['path'], 'rb') as stream:
        st = os.fstat(stream.fileno())
        require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original physical lease inode changed')
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(stream, fcntl.LOCK_UN)


def same_owner(left, right):
    return all(left.get(key) is not None and str(left[key]) == str(right.get(key))
               for key in ('pid', 'host', 'boot_id', 'process_start_ticks'))


def queue_spec(path, resolve=verified):
    path = Path(path); q = read(path)
    require(q['schema'] == 'natural-x-finite-queue-v1' and q['status'] == 'SEALED'
            and q['automatic_retry'] is False and q['profiles']
            and q['operator']['sha256'] == file_sha256(__file__), 'Not a finite natural queue or operator changed')
    profiles = [resolve(ref) for ref in q['profiles']]
    require(all(p['natural_x']['source_approval'] == q['source_approval']
                and p['natural_x']['model'] == q['model'] for p in profiles), 'Queue/approval/model differs')
    require(len({p['card'] for p in profiles}) == 1, 'One finite queue per physical card')
    ids = [sid for p in profiles for sid in p['assigned_ids']]
    require(len(ids) == len(set(ids))
            and all(len({p[key] for p in profiles}) == len(profiles) for key in ('output','claim_root','control_file','pause_file')),
            'Duplicate queue IDs or execution identities')
    for p in profiles: runner.science(p, resolve)
    root = Path(os.path.abspath(q['workspace_root'])); ns = profiles[0]['natural_x']['run_namespace']
    require(all(p['workspace_root'] == q['workspace_root'] and p['natural_x']['run_namespace'] == ns for p in profiles),
            'Queue mixed host/run namespaces')
    for key in ('fence', 'dispatch_lock', 'controller_lock'):
        require(Path(os.path.abspath(q[key])).is_relative_to(root / 'plans' / ns), 'Foreign queue control path')
    require(Path(os.path.abspath(q['output'])).is_relative_to(root / 'runs/reproducibility' / ns)
            and Path(os.path.abspath(q['state_path'])).is_relative_to(Path(os.path.abspath(q['output']))), 'Foreign queue state path')
    return q, profiles


@contextmanager
def locked(path, nonblocking=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (fcntl.LOCK_NB if nonblocking else 0)); yield


def check_fresh(profile):
    out = Path(profile['output'])
    require(not out.exists() and not any(out.with_suffix(s).exists() for s in ('.launch.json', '.wait.json', '.log'))
            and not list(Path(profile['claim_root']).glob('*')), 'Already attempted worker; no implicit retry')


def reconcile_profile(arc, ref):
    p = arc.ref(ref); runner.science(p, arc.ref); out = arc.path(p['output'])
    wait = arc.ref(arc.capture(out.with_suffix('.wait.json')))
    terminal = arc.ref(arc.capture(out / 'result.json')); status = terminal['status']
    require(wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released'] and wait['profile_sha256'] == digest(p)
            and status == wait['terminal_status'] and status in ('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD')
            and not terminal['UNCERTAIN_ids'] and not (out / 'failure.json').exists()
            and not (out / 'reproduction-failure.json').exists(), 'Failure/uncertain/unsafe wait; refuse resume')
    raw = journal(out / 'raw.jsonl'); events = journal(out / 'events.jsonl')
    ids = [row['sample_id'] for row in raw]
    require(len(ids) == len(set(ids)) and ids == p['assigned_ids'][:len(ids)]
            == terminal['completed_ids'] == wait['completed_ids']
            and terminal.get('unattempted_ids', p['assigned_ids'][len(ids):]) == p['assigned_ids'][len(ids):], 'Not an exact ordered complete prefix')
    require(status != 'COMPLETE' or ids == p['assigned_ids'], 'Incomplete COMPLETE')
    require(not any(any(word in row.get('event', '').upper() for word in
                ('FAIL', 'UNCERTAIN', 'RETRY', 'ERROR', 'INTERRUPT')) for row in events), 'Dirty scientific ledger')
    require([row['sample_id'] for row in events if row['event'] == 'INTENT'] == ids
            == [row['sample_id'] for row in events if row['event'] == 'TECHNICAL_PASS'], 'INTENT/PASS/raw differs')
    claims = list(arc.path(p['claim_root']).iterdir()) if arc.path(p['claim_root']).exists() else []
    require(not arc.path(p['claim_root']).is_symlink()
            and all(x.is_file() and not x.is_symlink() and x.suffix == '.json' for x in claims)
            and {x.stem for x in claims} == set(ids), 'Unknown or unresolved claim; refuse resume')
    passes = {row['sample_id']: row['raw_digest'] for row in events if row['event'] == 'TECHNICAL_PASS'}
    catalog = arc.ref(p['source_catalog'])
    approval, _, identities = runner.admission(p['natural_x']['source_approval'], p['source_catalog'], arc.ref)
    for ordinal, row in enumerate(raw):
        sid = row['sample_id']; claim = arc.ref(arc.capture(arc.path(p['claim_root']) / (sid + '.json')))
        require(row['profile_sha256'] == claim['profile_sha256'] == digest(p)
                and claim['sample_id'] == sid and claim['worker_token'] == p['worker_token']
                and passes[sid] == digest(row) and row['ordinal'] == ordinal, 'Raw/PASS/claim binding differs')
        current = runner.source_identity(catalog['items'][sid], identities, approval['dataset_revision'])
        videos = [part['sha256'] for part in row['source']['media'] if part['type'] == 'video']
        require(videos == [current['video_sha256']] and row['truth'] == current['truth']
                and row['question'] == current['prompt'], 'Raw source/GT/prompt differs')
    proof = None
    if ids:
        # The generic audit already supports strict paused prefixes. Scope its
        # dispatcher to this exact natural profile, without fake wait/claims.
        admission = nullcontext()
        if p['natural_x']['model'] == 'gemma':
            from . import gemma_x_full_results as gemma
            admission = gemma.gemma_admission()
        with admission:
            original = evidence.closed_prefix_selection
            def prefix(archive, spec, profile, waited, result, path):
                require(profile == p and waited == wait and result == terminal and status == 'PAUSED', 'Foreign prefix audit')
                return ids
            evidence.closed_prefix_selection = prefix
            try:
                spec = {'profile': ref, 'phase': 'full'}
                if status == 'PAUSED': spec['paused_prefix_successor'] = True
                records, proof = evidence.audit_worker(arc, spec)
            finally: evidence.closed_prefix_selection = original
        require([row['sample_id'] for row in records] == ids, 'Scientific auditor changed prefix')
        runner.validate_runtime(p, proof['runtime'], arc.ref)
        if p['natural_x']['model'] == 'gemma':
            deep = evidence.unique(journal(out / 'deep-observations.jsonl'))
            admitted = evidence.unique(journal(out / 'input-admissions.jsonl')); table = arc.ref(p['reference_prompt_catalog'])
            for row in raw: gemma.native_input_admission(p, catalog['items'][row['sample_id']], row,
                deep.get(row['sample_id']), admitted.get(row['sample_id']), table)
    else:
        require(not raw and not events and not claims, 'Empty pause contains an attempt')
        if status == 'PAUSED_BEFORE_LOAD':
            require(terminal['model_load_started'] is False and not (out / 'load.json').exists(), 'Pre-load pause loaded a model')
        else:
            require(status == 'PAUSED' and (out / 'load.json').is_file(), 'Unknown empty closure')
            runner.validate_runtime(p, read(out / 'runtime.json'), arc.ref)
            public = read(out / 'public-settings.json')['LLM_kwargs']
            require(all(public[key] == (sorted(value) if key == 'cpu_offload_params' else value)
                        for key, value in p['engine_kwargs'].items()), 'Empty loaded pause changed actual settings')
            exited = read(out / 'exit.json')
            require(exited['owned_engine_workers_exited'] and exited['shutdown_error'] is None, 'Empty loaded worker did not safely exit')
    files = [x for x in out.rglob('*') if x.is_file()] + [out.with_suffix('.wait.json'), out.with_suffix('.launch.json')]
    with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(arc.capture, files))
    return p, terminal, raw, proof


def continuation(arc, profile, seen=None):
    """Validate every distinct ancestor claim namespace; never filter unknowns."""
    parent_ref = profile['natural_x'].get('parent')
    if parent_ref is None: return []
    seen = set() if seen is None else seen
    key = (parent_ref['path'], parent_ref['sha256']); require(key not in seen, 'Cyclic continuation lineage'); seen.add(key)
    parent = arc.ref(parent_ref); prefix = continuation(arc, parent, seen)
    require(parent['source_catalog'] == profile['source_catalog']
            and parent['natural_x']['source_approval'] == profile['natural_x']['source_approval']
            and parent['natural_x']['allocation'] == profile['natural_x']['allocation']
            and parent['natural_x']['shard'] == profile['natural_x']['shard']
            and runner.settings(parent) == runner.settings(profile)
            and profile['control_generation'] == parent['control_generation'] + 1
            and profile['claim_root'] != parent['claim_root'] and profile['output'] != parent['output'],
            'Continuation changed source/science/generation or reused its identity')
    if arc.path(parent['output']).exists():
        parent, terminal, raws, _ = reconcile_profile(arc, parent_ref)
        require(terminal['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD')
                and arc.ref(profile['natural_x']['parent_wait']) == read(arc.path(parent['output']).with_suffix('.wait.json')),
                'Parent was not a clean paused real wait0')
        remaining = parent['assigned_ids'][len(raws):]; prefix.extend(raws)
    else:
        check_fresh(parent); remaining = parent['assigned_ids']
        require('parent_wait' not in profile['natural_x'], 'Unattempted parent has a fabricated wait')
    require(profile['assigned_ids'] == remaining and profile['maximum_model_calls'] == len(remaining),
            'Continuation repeats accepted calls or changes the exact suffix')
    require(len({row['sample_id'] for row in prefix}) == len(prefix), 'Overlapping accepted ancestor records')
    return prefix


def reusable(approval_ref, catalog_ref, model, root):
    """Retain current matching original raw; changed video/GT/prompt is not reused."""
    arc = Archive(root); approval, catalog, identities = runner.admission(approval_ref, catalog_ref, arc.ref)
    require(model in runner.ORDER, 'Unknown model')
    signatures = {digest(runner.settings(arc.ref(ref))) for ref in approval['templates'][model]}
    accepted = {}; excluded = []
    for receipt_ref in approval.get('reuse', {}).get(model, []):
        receipt = arc.ref(receipt_ref)
        require(receipt['schema'] == 'natural-x-existing-raw-reuse-v1'
                and receipt['status'] == 'ROOT_ACCEPTED_ORIGINAL_RAW_AUDIT' and receipt['model'] == model,
                'Original result reuse was not accepted by root')
        arc.ref(receipt['original_audit'])
        for segment in receipt['segments']:
            p = arc.ref(segment['profile'])
            require(digest(runner.settings(p)) in signatures, 'Original configuration not approved for reuse; hold instead of rerunning it')
            raw_path = arc.path(segment['raw']['path']); arc.capture(raw_path, segment['raw']['sha256'])
            raws = evidence.unique(journal(raw_path)); requested = segment['record_digests']
            for old_id, expected in requested.items():
                raw = raws[old_id]; require(digest(raw) == expected and raw['profile_sha256'] == digest(p), 'Original complete raw changed')
                require(raw['actual_prompt_token_ids'] and raw['output_token_ids']
                        and raw['finish_reason'] == 'stop' and raw['raw_answer'].strip() in ('LEFT','RIGHT'), 'Unqualified saved answer')
                try: binding = identities.resolve(old_id, domain='natural', dataset_revision=approval['dataset_revision'])
                except KeyError:
                    excluded.append({'sample_id': old_id, 'reason': 'NOT_CURRENT_ID_NO_PARENT_ALIAS'}); continue
                sid = binding.canonical_id; current = runner.source_identity(catalog['items'][sid], identities, approval['dataset_revision'])
                media = [x['sha256'] for x in raw['source']['media'] if x['type'] == 'video']
                if media != [current['video_sha256']] or raw['truth'] != current['truth'] or raw['question'] != current['prompt']:
                    excluded.append({'sample_id': old_id, 'reason': 'VIDEO_GT_OR_PROMPT_CHANGED'}); continue
                require(sid not in accepted, 'Two original answers claim the same current identity')
                accepted[sid] = {**current, 'raw_digest': expected, 'raw': segment['raw'], 'profile': segment['profile']}
    arc.unchanged()
    return {'schema': 'natural-x-current-reuse-v1', 'status': 'PASS_EXACT_CURRENT_SOURCE_REUSE_CPU',
            'model': model, 'source_approval': approval_ref, 'catalog': catalog_ref,
            'source_version': approval['source_version'], 'source_identity_sha256': approval['computed_source_identity_sha256'],
            'accepted': accepted, 'accepted_ids': [sid for sid in catalog['items'] if sid in accepted],
            'remaining_ids': [sid for sid in catalog['items'] if sid not in accepted], 'excluded_old_source': excluded,
            'current_denominator': sum(runner.COUNTS.values()), 'original_raw_rewritten': False,
            'bindings': [{'path': p, 'sha256': sha} for p, sha in sorted(arc.bindings.items())], 'model_calls': 0}


def closed_queue(arc, path, live=False):
    q, profiles = queue_spec(arc.path(path), arc.ref); ps = paths(arc.path(path), q)
    waited = arc.ref(arc.capture(ps['wait'])); owner = arc.ref(arc.capture(ps['owner']))
    launch = arc.ref(arc.capture(ps['launch'])); state = arc.ref(arc.capture(q['state_path']))
    require(waited['returncode'] == 0 and waited['real_wait_observed']
            and same_ref(arc, waited['queue'], descriptor(arc.path(path))) and same_owner(waited['owner'], owner)
            and launch['pid'] == owner['pid'] and launch['boot_id'] == owner['boot_id']
            and str(launch['process_start_ticks']) == str(owner['process_start_ticks'])
            and state['owner']['pid'] == waited['child_pid']
            and same_ref(arc, state['plan'], descriptor(arc.path(path)))
            and state['status'] in ('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_SUCCESSOR'), 'Queue is not genuinely closed')
    entries = state['workers']; require([x['profile'] for x in entries] == q['profiles'][:len(entries)]
            and all(x['returncode'] == 0 and x['real_supervisor_wait']
                    and x['terminal_status'] in ('COMPLETE','PAUSED','PAUSED_BEFORE_LOAD') for x in entries),
            'Inner queue worker/wait mapping differs')
    require(state['status'] != 'COMPLETE' or len(entries) == len(profiles)
            and all(x['terminal_status'] == 'COMPLETE' for x in entries), 'Incomplete queue COMPLETE')
    attempted = {ref['path'] for ref, p in zip(q['profiles'], profiles) if arc.path(p['output']).exists()}
    require(attempted == {x['profile']['path'] for x in entries}, 'Attempted profile missing real supervisor wait')
    if live:
        require(live_owner(owner)['alive'] is False and live_owner(state['owner'])['alive'] is False, 'Queue still owns a process')
        for p in profiles: lease_free(p)
    return q, profiles, state


def same_ref(arc, left, right):
    return left['sha256'] == right['sha256'] and arc.path(left['path']) == arc.path(right['path'])


def pause(run_set_path, reason):
    run_set = read(run_set_path)
    require(run_set['schema'] == 'natural-x-run-set-v1', 'Wrong run set')
    with locked(run_set['dispatch_lock']):
        fence = read(run_set['fence'])
        if fence['paused']: return {'status': 'PAUSE_ALREADY_REQUESTED', 'generation': fence['generation']}
        queues = [queue_spec(verified_ref_path(ref))[0] for ref in run_set['queues']]
        require(all(q['fence'] == run_set['fence'] and q['dispatch_lock'] == run_set['dispatch_lock'] for q in queues),
                'Run set must fence every selected successor with the same dispatch lock')
        profiles = [verified(ref) for q in queues for ref in q['profiles']]
        require(all(p['natural_x']['source_approval'] == run_set['source_approval'] for p in profiles), 'Foreign profile in pause set')
        for p in profiles:
            read(p['control_file'])
            if Path(p['pause_file']).exists():
                require(read(p['pause_file']) == {'worker_token': p['worker_token'], 'action': 'PAUSE'}, 'Foreign pause token')
        generation = fence['generation'] + 1
        # This lock also surrounds each dispatch. Fence every not-yet-executed
        # successor before returning, then the current request finishes intact.
        atomic_json(run_set['fence'], {**fence, 'generation': generation, 'paused': True, 'reason': reason})
        for p in profiles:
            c = read(p['control_file']); atomic_json(p['control_file'], {**c, 'generation': c['generation'] + 1,
                'paused': True, 'reason': reason})
            target = Path(p['pause_file']); value = {'worker_token': p['worker_token'], 'action': 'PAUSE'}
            if target.exists(): require(read(target) == value, 'Foreign pause token')
            else: write_new(target, value)
    return {'status': 'PAUSE_REQUESTED_AT_REQUEST_BOUNDARY', 'generation': generation,
            'fenced_profiles': len(profiles), 'lease_released': 'Only after real worker+outer wait closure', 'signals_sent': 0}


def fresh_suffix(profile, completed, output_root, plan_root, generation):
    ids = profile['assigned_ids'][len(completed):]; require(ids, 'Nothing remains')
    name = profile['worker'] + '-resume' + str(generation)
    p = copy.deepcopy(profile)
    p.update(worker=name, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
        output=str(Path(output_root) / name), claim_root=str(Path(output_root) / 'claims' / name),
        control_file=str(Path(plan_root) / (name + '-control.json')), pause_file=str(Path(plan_root) / (name + '.pause')))
    for key in ('worker_token', 'media_files', 'implementation'): p.pop(key, None)
    return p


def resume(queue_path, output_path):
    q = read(queue_path); arc = Archive(q['workspace_root'])
    arc.capture(__file__)
    with locked(q['dispatch_lock']):
        ps = paths(queue_path, q)
        if any(x.exists() for x in ps.values()) or Path(q['state_path']).exists():
            old, profiles, state = closed_queue(arc, queue_path, live=True)
            require(state['status'] in ('PAUSED', 'PAUSED_BEFORE_SUCCESSOR', 'COMPLETE'), 'Unclean queue terminal')
        else:
            old, profiles = queue_spec(queue_path, arc.ref)
            for p in profiles: check_fresh(p); lease_free(p)
        require(read(q['fence'])['paused'], 'Only an explicitly fenced, truly closed pause can resume')
        candidates = []; accepted = []; reconciliation = []
        # Audit all lineage bytes before generating ANY new profile or control.
        for ref, p in zip(old['profiles'], profiles):
            ancestry = continuation(arc, p)
            accepted.extend(row['sample_id'] for row in ancestry)
            out = arc.path(p['output'])
            if out.exists():
                p, terminal, raw, proof = reconcile_profile(arc, ref)
                accepted.extend(row['sample_id'] for row in raw); reconciliation.append({'profile': ref, 'proof': proof})
                if terminal['status'] == 'COMPLETE': continue
                completed = terminal['completed_ids']
            else:
                check_fresh(p); completed = []
            require(read(p['control_file'])['paused'], 'Worker not fenced')
            new = fresh_suffix(p, completed, Path(p['output']).parent / ('resume-generation-' + str(read(p['control_file'])['generation'])), Path(output_path).parent,
                               read(p['control_file'])['generation'])
            new['natural_x']['parent'] = ref
            new['natural_x'].pop('parent_wait', None)
            if out.exists(): new['natural_x']['parent_wait'] = arc.capture(out.with_suffix('.wait.json'))
            candidates.append(new)
        require(len(accepted) == len(set(accepted)), 'Duplicate accepted ID in queue')
        arc.unchanged()
        require(not Path(output_path).exists(), 'Resume receipt already exists; no branch')
        receipt = {'schema': 'natural-x-resume-candidates-v1', 'status': 'CPU_RECONCILED_DRAFTS_NOT_SEALED_NOT_LAUNCHED',
            'operator': descriptor(__file__), 'fence_generation': read(q['fence'])['generation'],
            'predecessor_queue': arc.capture(queue_path), 'accepted_ids': accepted, 'uncertain_claims': 0,
            'draft_profiles': candidates, 'reconciliation': reconciliation,
            'bindings': [{'path': path, 'sha256': sha} for path, sha in sorted(arc.bindings.items())], 'model_calls': 0}
        write_new(output_path, receipt)
    return receipt


def seal_resume(receipt_path, output_queue):
    receipt = read(receipt_path)
    require(receipt['schema'] == 'natural-x-resume-candidates-v1' and receipt['uncertain_claims'] == 0
            and receipt['operator']['sha256'] == file_sha256(__file__) and receipt['draft_profiles'], 'No exact resumable suffix')
    old = verified(receipt['predecessor_queue']); arc = Archive(old['workspace_root'])
    with locked(old['dispatch_lock']):
        require(read(old['fence'])['paused'] and read(old['fence'])['generation'] == receipt['fence_generation'], 'Pause fence changed')
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda ref: arc.capture(ref['path'], ref['sha256']), receipt['bindings']))
        for p in receipt['draft_profiles']: continuation(arc, p); check_fresh(p); lease_free(p)
        require(not Path(output_queue).exists(), 'Successor queue exists; no branch/retry')
        consumed = Path(receipt['predecessor_queue']['path']).with_suffix('.resume-once.json')
        write_new(consumed, {'receipt': descriptor(receipt_path), 'successor_queue': str(output_queue)})
        refs = []
        for p in receipt['draft_profiles']:
            target = Path(output_queue).parent / (p['worker'] + '-profile.json')
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', dir=Path(output_queue).parent, delete=False) as stream:
                json.dump(p, stream); draft_path = stream.name
            try: runner.seal(draft_path, target)
            finally: Path(draft_path).unlink()
            refs.append(descriptor(target))
            write_new(p['control_file'], {'generation': p['control_generation'], 'paused': True})
        q = copy.deepcopy(old); out = Path(old['output']) / ('resume-' + str(receipt['fence_generation']))
        q.update(profiles=refs, output=str(out), state_path=str(out / 'queue-state.json'),
            controller_lock=str(Path(output_queue).with_suffix('.controller.lock')),
            fence_generation=receipt['fence_generation'], resume_receipt=descriptor(receipt_path))
        write_new(output_queue, q)
    return {'status': 'SEALED_SUCCESSOR_PAUSED_NOT_LAUNCHED', 'queue': descriptor(output_queue), 'model_calls': 0}


def unpause(run_set_path):
    run_set = read(run_set_path)
    require(run_set['schema'] == 'natural-x-run-set-v1', 'Wrong natural run set')
    with locked(run_set['dispatch_lock']):
        fence = read(run_set['fence']); require(fence['paused'], 'Run set is already active')
        for ref in run_set['queues']:
            q, profiles = queue_spec(verified_ref_path(ref))
            require(q['fence_generation'] == fence['generation']
                    and q['source_approval'] == run_set['source_approval'], 'Old generation cannot reactivate')
            require(not any(x.exists() for x in paths(ref['path'], q).values()), 'Already attempted successor queue')
            for p in profiles:
                check_fresh(p); lease_free(p)
                require(read(p['control_file'])['generation'] == p['control_generation'] == fence['generation'], 'Fenced profile')
        atomic_json(run_set['fence'], {**fence, 'paused': False})
    return {'status': 'NEW_GENERATION_READY_ORIGINAL_RESOURCE_GATES_REMAIN', 'model_calls': 0}


def verified_ref_path(ref):
    require(file_sha256(ref['path']) == ref['sha256'], 'Queue descriptor changed'); return ref['path']


def gates(q):
    fence = read(q['fence'])
    require(not fence['paused'] and fence['generation'] == q['fence_generation'], 'Natural run set is fenced/stale')
    require(q['predecessors'], 'Model order predecessor must be explicit')
    prior = {'cosmos': ('gemma', 9456), 'qwen': ('cosmos', 10416), 'nemo': ('qwen', 10416), 'gemma': ('nemo', 10416)}[q['model']]
    matched = False
    approval = verified(q['source_approval'])
    if q['model'] != 'cosmos':
        profile = verified(q['profiles'][0])
        approval, _, _ = runner.admission(q['source_approval'], profile['source_catalog'])
    for entry in q['predecessors']:
        actual = verified(entry['receipt'])
        require(all(actual.get(key) == value for key, value in entry['required'].items()), 'Predecessor incomplete')
        if actual.get('status') == 'COMPLETE_PASS' and (actual.get('model'), actual.get('unique_videos')) == prior:
            same_source = actual.get('source_approval') == q['source_approval'] or (
                actual.get('source_version') == approval['source_version']
                and actual.get('source_identity_sha256') == approval.get('computed_source_identity_sha256'))
            require(q['model'] == 'cosmos' or same_source, 'Prior model used another source version')
            matched = True
    require(matched, 'Required Cosmos→Qwen→Nemo→Gemma predecessor has not completed')


def reserve_allocation(profile_ref, profile):
    """One initial owner / one continuation chain per root-approved shard."""
    spec = profile['natural_x']; root_ref = profile_ref; ancestors = []
    current = profile
    while current['natural_x'].get('parent'):
        ref = current['natural_x']['parent']; require(ref not in ancestors, 'Cyclic shard lineage')
        ancestors.append(ref); root_ref = ref; current = verified(ref)
    token = digest({'allocation': spec['allocation'], 'shard': spec['shard']})
    target = Path(profile['workspace_root'])/'plans'/spec['run_namespace']/'allocation-owners'/(token+'.json')
    value = {'initial_profile': root_ref, 'current_profile': profile_ref, 'generation': profile['control_generation']}
    if target.exists():
        old = read(target)
        require(old['initial_profile'] == root_ref and old['current_profile'] in ancestors,
                'Allocation already dispatched or belongs to another branch')
        continuation(Archive(profile['workspace_root']), profile)
        atomic_json(target, value)
    else: write_new(target, value)


def execute(queue_path):
    q, _ = queue_spec(queue_path)
    with locked(q['controller_lock'], nonblocking=True): return _execute_owned(queue_path)


def _execute_owned(queue_path):
    q, profiles = queue_spec(queue_path); state_path = Path(q['state_path'])
    require(not state_path.exists(), 'Queue already attempted; refuse retry')
    require(read(paths(queue_path, q)['launch-intent'])['queue'] == descriptor(queue_path), 'Missing exact launch intent')
    state = {'status': 'RUNNING', 'plan': descriptor(queue_path), 'owner': owner_identity(), 'workers': []}
    atomic_json(state_path, state)
    try:
        for ref, p in zip(q['profiles'], profiles):
            with locked(q['dispatch_lock']):
                c = read(p['control_file'])
                if read(q['fence'])['paused'] or c['paused'] or c['generation'] != p['control_generation']:
                    state['status'] = 'PAUSED_BEFORE_SUCCESSOR'; break
                gates(q); check_fresh(p); env = dict(os.environ)
                reserve_allocation(ref, p)
                for key in p['cleared_environment']: env.pop(key, None)
                env.update(p['environment']); code = Path(p['workspace_root']) / 'code/vlm-psychophysics'
                state['current_profile'] = ref; atomic_json(state_path, state)
                child = subprocess.Popen([p['python'], '-B', '-m', WORKER, 'supervise', '--profile', ref['path']],
                    env=env, cwd=code, stdin=subprocess.DEVNULL)
            rc = child.wait(); entry = {'profile': ref, 'pid': child.pid, 'returncode': rc, 'real_supervisor_wait': True}
            state['workers'].append(entry); require(rc == 0, 'Worker failure; hold all successors without retry')
            arc = Archive(q['workspace_root']); _, terminal, _, _ = reconcile_profile(arc, ref)
            entry['terminal_status'] = terminal['status']; atomic_json(state_path, state)
            if terminal['status'] != 'COMPLETE': state['status'] = 'PAUSED'; break
        else: state['status'] = 'COMPLETE'
    except BaseException as error:
        state.update(status='HELD_NO_RETRY', error=str(error)); raise
    finally: atomic_json(state_path, state)
    return {'status': state['status'], 'workers': state['workers']}


def launch(queue_path):
    q, profiles = queue_spec(queue_path); ps = paths(queue_path, q)
    require(Path(sys.executable).resolve() == Path(q['cpu_python']).resolve(), 'Use the host existing CPU qy')
    require(all(p['hostname'] == socket.gethostname() for p in profiles), 'Wrong host')
    with locked(q['dispatch_lock']):
        require(not any(x.exists() for x in ps.values()) and not Path(q['state_path']).exists(), 'Queue already attempted')
        gates(q)
        for p in profiles: check_fresh(p); lease_free(p)
        from .gemma_x_fourier_operations import resource_preflight
        resource_preflight(profiles[0])
        write_new(ps['launch-intent'], {'queue': descriptor(queue_path), 'owner': owner_identity(), 'automatic_retry': False})
        for p in profiles:
            c = read(p['control_file']); require(c['generation'] == p['control_generation'], 'Fenced generation')
            atomic_json(p['control_file'], {**c, 'paused': False})
        code = Path(q['workspace_root']) / 'code/vlm-psychophysics'; env = dict(os.environ)
        for key in profiles[0]['cleared_environment']: env.pop(key, None)
        env.update(profiles[0]['environment'])
        with ps['launch'].with_suffix('.log').open('xb') as log:
            child = subprocess.Popen([q['cpu_python'], '-B', '-m', MODULE, 'queue-supervisor', '--queue', str(queue_path)],
                env=env, cwd=code, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19]
        write_new(ps['launch'], {'queue': descriptor(queue_path), 'pid': child.pid,
            'process_start_ticks': ticks, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()})
    return {'status': 'LAUNCHED_ONCE', 'pid': child.pid}


def queue_supervisor(queue_path):
    q, _ = queue_spec(queue_path); ps = paths(queue_path, q)
    require(Path(sys.executable).resolve() == Path(q['cpu_python']).resolve(), 'Outer supervisor must use host CPU qy')
    require(read(ps['launch-intent'])['queue'] == descriptor(queue_path), 'Intent changed')
    write_new(ps['owner'], owner_identity())
    child = subprocess.Popen([q['cpu_python'], '-B', '-m', MODULE, 'queue', '--queue', str(queue_path)])
    rc = child.wait(); write_new(ps['wait'], {'returncode': rc, 'real_wait_observed': True,
        'owner': owner_identity(), 'child_pid': child.pid, 'queue': descriptor(queue_path)})
    return rc


def finalize(run_set_path, approval_ref, catalog_ref, model, output, root):
    run_set = read(run_set_path)
    arc = Archive(root); reused = reusable(approval_ref, catalog_ref, model, root)
    require(run_set['schema'] == 'natural-x-run-set-v1', 'Wrong finalization run set')
    approval, catalog, _ = runner.admission(approval_ref, catalog_ref, arc.ref)
    arc.capture(__file__)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda ref: arc.capture(ref['path'], ref['sha256']), reused['bindings']))
    selected = set(reused['accepted_ids']); workers = []
    for qref in run_set['queues']:
        q = arc.ref(qref)
        if q['model'] != model: continue
        for ref in q['profiles']:
            profile = arc.ref(ref)
            source, _, _ = runner.admission(q['source_approval'], profile['source_catalog'], arc.ref)
            require(source['source_version'] == approval['source_version']
                    and source['computed_source_identity_sha256'] == approval['computed_source_identity_sha256'],
                    'Model queue used a different approved source revision')
        q, profiles, state = closed_queue(arc, qref['path'], live=False)
        require(state['status'] == 'COMPLETE', 'Current model queue is not COMPLETE')
        for ref, p in zip(q['profiles'], profiles):
            raws = continuation(arc, p)
            _, terminal, current, proof = reconcile_profile(arc, ref)
            require(terminal['status'] == 'COMPLETE', 'Final leaf is not COMPLETE');raws += current
            ids = [row['sample_id'] for row in raws]
            require(len(ids) == len(set(ids)) and not selected.intersection(ids), 'Duplicate accepted current-source call')
            selected.update(ids);workers.append({'profile': ref, 'accepted': len(ids), 'scientific_audit': proof})
    approval, catalog, _ = runner.admission(approval_ref, catalog_ref, arc.ref)
    require(selected == set(catalog['items']) and len(selected) == sum(runner.COUNTS.values()), 'Natural model is not the exact10416 universe')
    arc.unchanged()
    result = {'schema': 'natural-x-complete-model-v1', 'status': 'COMPLETE_PASS', 'model': model,
        'source_approval': approval_ref, 'unique_videos': len(selected), 'counts': runner.COUNTS,
        'source_version': approval['source_version'], 'source_identity_sha256': approval['computed_source_identity_sha256'],
        'reused': len(reused['accepted_ids']), 'workers': workers, 'new_model_calls': 0,
        'live_host_owner_recheck': False, 'next_launch_requires_actual_host_same_inode_lease_free': True,
        'bindings': [{'path':p,'sha256':sha} for p,sha in sorted(arc.bindings.items())]}
    write_new(output,result);return result


def status(queue_path):
    q, profiles = queue_spec(queue_path); ps = paths(queue_path, q); summaries = []
    for p in profiles:
        out = Path(p['output']); partial=[]
        def snapshot(name):
            path=out/name
            if not path.exists():return []
            data=path.read_bytes().splitlines(keepends=True);rows=[]
            for line in data:
                if not line.endswith(b'\n'):
                    partial.append(name);break
                try:rows.append(json.loads(line))
                except (ValueError,UnicodeError):partial.append(name);break
            return rows
        rows=snapshot('raw.jsonl');events=snapshot('events.jsonl')
        passed = [x['sample_id'] for x in events if x.get('event') == 'TECHNICAL_PASS']
        claimed = [x.stem for x in Path(p['claim_root']).glob('*')]
        summaries.append({'worker': p['worker'], 'expected': len(p['assigned_ids']), 'raw': len(rows),
            'passed': len(passed), 'claimed': len(claimed), 'unresolved_claims': sorted(set(claimed)-set(passed)),
            'partial_journal_tails':partial,
            'wait': {key:value for key,value in read(out.with_suffix('.wait.json')).items() if key!='completed_ids'}
                    if out.with_suffix('.wait.json').exists() else None})
    return {'status': 'READ_ONLY_SNAPSHOT', 'profiles': summaries, 'fence': read(q['fence']),
        'outer_wait': read(ps['wait']) if ps['wait'].exists() else None,
        'outer_owner': live_owner(read(ps['owner'])) if ps['owner'].exists() else None, 'GPU_queries': 0, 'model_calls': 0}


def main():
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('command', choices=('pause','unpause','status','resume','seal-resume','reuse','finalize','launch','queue-supervisor','queue'))
    p.add_argument('--queue', type=Path);p.add_argument('--run-set', type=Path);p.add_argument('--output', type=Path);p.add_argument('--reason', default='User requested safe pause')
    p.add_argument('--receipt', type=Path);p.add_argument('--approval', type=Path);p.add_argument('--catalog', type=Path);p.add_argument('--model', choices=runner.ORDER);p.add_argument('--root', type=Path)
    a = p.parse_args()
    if a.command == 'pause': result = pause(a.run_set, a.reason)
    elif a.command == 'unpause': result = unpause(a.run_set)
    elif a.command == 'status': result = status(a.queue)
    elif a.command == 'resume': result = resume(a.queue, a.output)
    elif a.command == 'seal-resume': result = seal_resume(a.receipt, a.output)
    elif a.command == 'reuse':
        result = reusable(descriptor(a.approval), descriptor(a.catalog), a.model, a.root)
        if a.output: write_new(a.output, result)
    elif a.command == 'finalize': result = finalize(a.run_set, descriptor(a.approval), descriptor(a.catalog), a.model, a.output, a.root)
    elif a.command == 'launch': result = launch(a.queue)
    elif a.command == 'queue-supervisor': sys.exit(queue_supervisor(a.queue))
    else: result = execute(a.queue)
    if a.command not in ('status','pause','unpause','launch'):
        result={key:result[key] for key in ('status','model','model_calls','new_model_calls','queue','unique_videos','uncertain_claims') if key in result}
        if a.output:result['output']=str(a.output)
    print(json.dumps(result))


if __name__ == '__main__': main()
