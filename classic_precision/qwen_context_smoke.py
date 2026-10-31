"""One authorized 48-ID T1 context comparison, after Radon's Gemma queue closes.

Reuse the completed KV4/context98304 arm and frozen Qwen runner. The only
engine change is max_model_len=65536. Independent claims; no automatic retry.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from . import qwen_kv_shared_smoke as prior
from . import qwen_remaining_t13 as q
from . import qwen_remaining_t13_archive as archive
from .cosmos_x_full_status import queue_owner_observation
from .official_x_results import Archive, journal, unique
from .refinement_worker import read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity

NAME = 'qwen-x-context-smoke-20261031'
MODULE = 'classic_precision.qwen_context_smoke'
GEMMA = 'gemma-x-full-20261030'
REMOTE = q.ROOTS['radon']
ARM = 'ctx65536'
USER_MESSAGE = '也就是gpu一起跑不影响输出结果，kv改了也不影响？\n\nconttext也smokle一下吧，以防万一'
BASELINE_SUMMARY_SHA = '70357cc180e28a938f7a644400ccbdf4ff7258a0b4687c94d5d4c42877458a62'
GEMMA_QUEUE_SHA = '40e0402c937d00aff567677ffb35e37a8d45d027a8a208d6fa8a4e08e2c77713'
descriptor, ref_for = prior.descriptor, prior.ref_for


def science(p, resolve=None):
    resolve = archive.portable_resolver(p, resolve)
    entry = p['context_smoke']; plan = resolve(entry['plan'])
    require(entry['implementation']['sha256'] == plan['implementation']['sha256'] == file_sha256(__file__)
            and plan['baseline_adapter']['sha256'] == file_sha256(prior.__file__), 'Context adapter binding changed')
    require(plan['schema'] == NAME + '/v1' and plan['maximum_new_model_calls'] == 48
            and plan['only_changed_setting'] == 'engine_kwargs.max_model_len'
            and plan['baseline_context'] == 98304 and plan['target_context'] == 65536
            and plan['fixed_kv_bytes'] == 4 * 1024**3 and plan['automatic_retry'] is False,
            'Finite context comparison changed')
    auth = resolve(plan['authorization'])
    require(auth['status'] == 'AUTHORIZED' and auth['user_message'] == USER_MESSAGE
            and auth['maximum_new_model_calls'] == 48 and auth['cards'] == ['radon0']
            and auth['only_changed_setting'] == plan['only_changed_setting']
            and auth['original_results_preserved'] is True, 'Authorization differs')
    summary = resolve(plan['baseline_summary'])
    require(plan['baseline_summary']['sha256'] == BASELINE_SUMMARY_SHA
            and summary['full_output_and_prompt_equal'] and summary['kv8_vs_kv4_equal']
            and summary['new_model_calls'] == 96, 'Accepted baseline receipt differs')
    template = resolve(plan['baseline_profile']); prior.science(template, resolve)
    require(template['kv_shared_smoke']['arm'] == 'kv4'
            and template['engine_kwargs']['max_model_len'] == 98304
            and template['engine_kwargs']['kv_cache_memory_bytes'] == 4 * 1024**3, 'Wrong baseline arm')
    engine = copy.deepcopy(template['engine_kwargs']); engine['max_model_len'] = 65536
    require(p['engine_kwargs'] == engine, 'Only context may change from the accepted KV4 arm')
    operational = {'assigned_ids', 'maximum_model_calls', 'worker', 'output', 'claim_root',
                   'control_file', 'pause_file', 'source_catalog', 'phase', 'kv_shared_smoke',
                   'context_smoke', 'implementation', 'worker_token', 'media_files',
                   'operational_wrapper', 'engine_kwargs'}
    require('kv_shared_smoke' not in p and all(p.get(k) == template.get(k)
            for k in p.keys() | template.keys() if k not in operational), 'Other original configuration changed')
    previous_plan = resolve(template['kv_shared_smoke']['plan'])
    require(plan['reference'] == previous_plan['reference'] and plan['source_catalog'] == template['source_catalog'],
            'Existing full-output reference or sources changed')
    reference = resolve(plan['reference']); ids = plan['ids']
    require(ids == template['assigned_ids'] == reference['ids'] == p['assigned_ids']
            and len(ids) == len(set(ids)) == p['maximum_model_calls'] == 48,
            'Finite 48-ID allocation changed')
    require(len(reference['actual_prompt_token_ids']) == 35421
            and all(len(reference['actual_prompt_token_ids']) + row['generation']['max_tokens'] < 65536
                    for row in reference['outputs'].values()),
            'Actual input/output no longer fits the target context')
    root = Path(p['workspace_root']); run_root = root / 'runs/reproducibility' / NAME
    require(entry['arm'] == ARM and p['worker'] == 'radon0-T1-' + ARM
            and p['phase'] == 'deployment_smoke' and p['control_generation'] == 0
            and Path(p['output']) == run_root / p['worker']
            and Path(p['claim_root']) == run_root / 'claims' / ARM
            and Path(p['control_file']) == root / 'plans' / NAME / (ARM + '-control.json')
            and Path(p['pause_file']) == root / 'plans' / NAME / (ARM + '-PAUSE.json')
            and p['source_catalog'] == plan['source_catalog'], 'Independent attempt namespace changed')
    require(plan['predecessor_queue'] == {'path': str(Path(REMOTE) / 'plans' / GEMMA / 'radon0-qualification-queue.json'),
                                        'sha256': GEMMA_QUEUE_SHA}, 'Gemma predecessor binding changed')


@contextmanager
def context(execution=False):
    old_science, old_closed, native = q.science, q.closed, subprocess.Popen
    def popen(argv, **kwargs):
        if isinstance(argv, (list, tuple)):
            argv = [MODULE if x == 'classic_precision.qwen_remaining_t13' else x for x in argv]
        return native(argv, **kwargs)
    q.science, q.closed = science, closed
    if execution: subprocess.Popen = popen
    try: yield
    finally: q.science, q.closed, subprocess.Popen = old_science, old_closed, native


def seal(draft, output):
    p = read(draft); science(p)
    require(p['hostname'] == socket.gethostname(), 'Seal on Radon only')
    run_root = Path(p['output']).parent.resolve(strict=True)
    require(str(run_root) == '/home/q/cvpr2027_runs_ssd/' + NAME, 'Result run is not on the selected SSD')
    media = next(iter(verified(p['source_catalog'])['items'].values()))['content'][0]['path']
    require({run_root.stat().st_dev, Path(media).stat().st_dev, Path(p['checkpoint']).stat().st_dev} == {66306},
            'Weights/data/run must all remain on NVMe')
    with context(): return q.seal(draft, output)


def run(profile):
    p = read(profile); science(p); plan = verified(p['context_smoke']['plan'])
    reference = verified(plan['reference']); original_append = q.base._append; pending = {}; seen = set()
    def observed_append(path, value):
        out = Path(p['output'])
        if Path(path) == out / 'events.jsonl' and value.get('event') == 'TECHNICAL_PASS':
            receipt = pending.pop(value['sample_id']); original_append(out / 'output-equivalence.jsonl', receipt)
            if not receipt['all_equal']:
                atomic_json(out / 'reproduction-failure.json', {**receipt, 'status': 'STOP_CONTEXT_SMOKE_NO_RETRY',
                    'original_outputs_preserved': True, 'Fourier_Gemma_action': 'CONTINUE_UNCHANGED'})
                raise ValueError('Context smoke output differs; original raw retained; no automatic retry')
        original_append(path, value)
        if Path(path) == out / 'events.jsonl' and value.get('event') == 'INTENT':
            original_append(out / 'shared-gpu-observations.jsonl', prior.shared_observation(p, value['sample_id']))
        if Path(path) != out / 'raw.jsonl': return
        require(value['sample_id'] not in seen, 'Duplicate raw identity')
        seen.add(value['sample_id']); pending[value['sample_id']] = prior.compare_raw(reference, value)
    try:
        with context():
            q.base._append = observed_append
            return q.run(profile)
    finally: q.base._append = original_append


def supervise(profile):
    science(read(profile))
    with context(execution=True): return q.supervise(profile)


def closed(profile):
    p, terminal, raw = prior.ORIGINAL_CLOSED(profile)
    require(terminal['status'] == 'COMPLETE', 'Context arm must close COMPLETE')
    reference = verified(verified(p['context_smoke']['plan'])['reference']); out = Path(p['output'])
    rows = journal(out / 'raw.jsonl'); comparisons = journal(out / 'output-equivalence.jsonl')
    require(comparisons == [prior.compare_raw(reference, row) for row in rows]
            and len(comparisons) == 48 and all(x['all_equal'] for x in comparisons), 'Incomplete or unequal context arm')
    require([x['label'] for x in journal(out / 'shared-gpu-observations.jsonl')] == p['assigned_ids'],
            'Per-request activity records missing')
    return p, terminal, raw


def queue(plan_path):
    queue_plan = read(plan_path)
    require(queue_plan['schema'] == NAME + '/queue-v1' and len(queue_plan['profiles']) == 1
            and queue_plan['maximum_model_calls'] == 48 and queue_plan['automatic_retry'] is False,
            'Exactly one finite context arm required')
    science(verified(queue_plan['profiles'][0]))
    with context(execution=True): return q.queue(plan_path)


def predecessor_closed(plan, p):
    """Read actual G closure; do not start a model or inspect GPU resources here."""
    queue_plan = verified(plan['predecessor_queue']); root = Path(p['workspace_root'])
    run_root = root / 'runs/reproducibility' / GEMMA; prefix = run_root / 'radon0-qualification-queue'
    outer_wait = Path(str(prefix) + '.wait.json')
    if not outer_wait.exists(): return {'ready': False, 'reason': 'GEMMA_OUTER_WAIT_PENDING'}
    wait = read(outer_wait)
    require(wait['returncode'] == 0 and wait['real_wait_observed']
            and wait['queue_sha256'] == plan['predecessor_queue']['sha256']
            and wait['queue'] == plan['predecessor_queue']['path'], 'Gemma outer queue did not close successfully')
    owner = read(Path(str(prefix) + '.owner.json')); boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(queue_owner_observation(owner, boot)['alive'] is False, 'Gemma outer owner is still present')
    proofs = []
    for ref in queue_plan['profiles']:
        gp = verified(ref); out = Path(gp['output']); wp = out.with_suffix('.wait.json')
        if not wp.exists(): return {'ready': False, 'reason': 'GEMMA_WORKER_WAIT_PENDING'}
        w = read(wp); result = read(out / 'result.json')
        require(w['returncode'] == 0 and w['real_wait_observed'] and w['owned_engines_exited']
                and w['original_physical_lease_released'] and w['profile_sha256'] == digest(gp)
                and result['status'] == 'COMPLETE' and result['completed_ids'] == gp['assigned_ids']
                and not result.get('UNCERTAIN_ids') and not (out / 'failure.json').exists(),
                'Gemma worker closure/identity differs')
        require(gp['physical_lock'] == p['physical_lock'], 'Different physical GPU lease')
        proofs.append({'profile': ref, 'wait': descriptor(wp), 'result': descriptor(out / 'result.json')})
    lock = p['physical_lock']; fd = os.open(lock['path'], os.O_RDWR)
    try:
        st = os.fstat(fd); require((st.st_dev, st.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original lease inode changed')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN)
    finally: os.close(fd)
    return {'ready': True, 'outer_wait': descriptor(outer_wait), 'workers': proofs, 'same_inode_lease_free': True}


def queue_supervisor(plan_path):
    queue_plan = read(plan_path); p = verified(queue_plan['profiles'][0]); run_root = Path(p['output']).parent
    intent = read(run_root / 'queue-launch-intent.json')
    require(intent['queue'] == descriptor(plan_path) and sys.executable == p['python'], 'Launch/python binding differs')
    write_new(run_root / 'queue-owner.json', owner_identity()); started = time.monotonic()
    with (run_root / 'queue-child.log').open('xb') as log:
        child = subprocess.Popen([p['python'], '-B', '-m', MODULE, 'queue', '--plan', plan_path],
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    write_new(run_root / 'queue-wait.json', {'returncode': code, 'real_wait_observed': True,
        'child_pid': child.pid, 'queue': descriptor(plan_path), 'elapsed_monotonic_seconds': time.monotonic()-started,
        'automatic_retry': False})
    require(code == 0, 'Context smoke queue failed; no retry')
    return read(queue_plan['state_path'])


def launch(plan_path):
    queue_plan = read(plan_path)
    require(queue_plan['schema'] == NAME + '/queue-v1' and queue_plan['status'] == 'SEALED'
            and len(queue_plan['profiles']) == 1 and queue_plan['maximum_model_calls'] == 48
            and queue_plan['automatic_retry'] is False, 'One sealed 48-call arm only')
    p = verified(queue_plan['profiles'][0]); science(p); q.fresh(p); run_root = Path(p['output']).parent
    require(socket.gethostname() == p['hostname'], 'Launch on Radon only')
    require(not any((run_root / name).exists() for name in
                    ['queue-launch-intent.json', 'queue-launch.json', 'queue-owner.json', 'queue-wait.json'])
            and not Path(queue_plan['state_path']).exists(), 'Queue already attempted')
    closure = predecessor_closed(verified(p['context_smoke']['plan']), p)
    if not closure['ready']: return {'status': 'HOLD_GEMMA_CLOSURE', 'closure': closure, 'model_calls': 0}
    from .gemma_x_fourier_operations import resource_preflight
    resources = resource_preflight(p)
    observation = prior.shared_observation(p, 'before_launch')
    write_new(run_root / 'launch-shared-gpu-observation.json', observation)
    write_new(run_root / 'queue-launch-intent.json', {'queue': descriptor(plan_path), 'owner': owner_identity(),
        'maximum_new_model_calls': 48, 'automatic_retry': False, 'predecessor_closure': closure,
        'resource_admission': resources})
    with context(): q.activate(plan_path)
    env = dict(os.environ)
    for key in p['cleared_environment']: env.pop(key, None)
    env.update(p['environment']); env['PYTHONPATH'] = str(Path(p['workspace_root']) / 'code/vlm-psychophysics')
    with (run_root / 'queue-supervisor.log').open('xb') as log:
        child = subprocess.Popen([p['python'], '-B', '-m', MODULE, 'queue-supervisor', '--plan', plan_path],
            cwd=env['PYTHONPATH'], env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True)
    answer = {'status': 'LAUNCHED_ONCE', 'queue': descriptor(plan_path), 'pid': child.pid,
        'process_start_ticks': Path('/proc', str(child.pid), 'stat').read_text().rsplit(')', 1)[1].split()[19],
        'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(), 'owner': owner_identity(),
        'automatic_retry': False}
    write_new(run_root / 'queue-launch.json', answer); return answer


def summarize(root, plan_path, output):
    from .qwen_remaining_t13_results import audit_segment
    arc = Archive(root); queue_plan = read(plan_path); require(len(queue_plan['profiles']) == 1, 'One arm only')
    p = arc.ref(queue_plan['profiles'][0]); plan = arc.ref(p['context_smoke']['plan']); science(p, arc.ref)
    with context(): records, proof = audit_segment(root, arc.path(queue_plan['profiles'][0]['path']))
    out = arc.path(p['output']); rows = unique(journal(out / 'raw.jsonl')); expected = arc.ref(plan['reference'])
    pairs = [prior.compare_raw(expected, rows[sid]) for sid in p['assigned_ids']]
    require(len(records) == len(pairs) == 48 and all(x['all_equal'] for x in pairs)
            and pairs == journal(out / 'output-equivalence.jsonl'), 'Incomplete context output equivalence')
    baseline = arc.ref(plan['baseline_profile']); baseline_raw = arc.path(baseline['output']) / 'raw.jsonl'
    require(file_sha256(baseline_raw) == plan['baseline_raw']['sha256'], 'Baseline raw binding changed')
    old = unique(journal(baseline_raw))
    require(list(old) == p['assigned_ids'] and all(prior.compare_raw(expected, old[sid])['all_equal']
            for sid in p['assigned_ids']), 'Prior context98304 arm differs from paired reference')
    outer = read(out.parent / 'queue-wait.json')
    require(outer['returncode'] == 0 and outer['real_wait_observed'], 'Outer queue has not closed successfully')
    observations = journal(out / 'shared-gpu-observations.jsonl')
    require([x['label'] for x in observations] == p['assigned_ids'], 'Activity records missing')
    result = {'status': 'PASS_QWEN_T1_CONTEXT_POINT', 'new_model_calls': 48, 'unique_ids': 48,
        'baseline_context': 98304, 'target_context': 65536, 'fixed_kv_gib': 4,
        'only_changed_setting': 'engine_kwargs.max_model_len', 'full_output_and_prompt_equal': True,
        'correct': sum(x['correct'] for x in rows.values()), 'paired': pairs, 'worker': proof,
        'baseline_raw': plan['baseline_raw'], 'active_other_account_observations':
            sum(x['other_account_compute_active'] for x in observations),
        'included_in_formal_results': False, 'original_formal_results_unchanged': True}
    write_new(output, result); return {'status': result['status'], 'summary': descriptor(output)}


def prepare(root):
    root = Path(root); plans = root / 'plans' / NAME; require(not plans.exists(), 'Context experiment already prepared')
    old_plans = root / 'plans' / prior.NAME; old_run = root / 'runs/reproducibility' / prior.NAME
    baseline_path = old_plans / 'kv4-profile.json'; template = read(baseline_path); prior.science(template, Archive(root).ref)
    summary_path = old_run / 'summary.json'; require(file_sha256(summary_path) == BASELINE_SUMMARY_SHA, 'Baseline receipt differs')
    old_plan = read(old_plans / 'plan.json'); reference = read(old_plans / 'reference.json')
    require(len(reference['ids']) == 48 and sum(x['correct'] for x in reference['outputs'].values()) == 27,
            'Original complete point differs')
    plans.mkdir(parents=True); (plans / 'baseline-summary.json').write_bytes(summary_path.read_bytes())
    implementation = {'path': str(Path(REMOTE) / 'code/vlm-psychophysics/classic_precision' / Path(__file__).name),
                      'sha256': file_sha256(__file__)}
    write_new(plans / 'authorization.json', {'status': 'AUTHORIZED', 'cards': ['radon0'], 'user_message': USER_MESSAGE,
        'maximum_new_model_calls': 48, 'only_changed_setting': 'engine_kwargs.max_model_len',
        'original_results_preserved': True, 'automatic_retry': False,
        'recorded_HST': datetime.now(ZoneInfo('Pacific/Honolulu')).isoformat()})
    plan = {'schema': NAME + '/v1', 'ids': reference['ids'], 'maximum_new_model_calls': 48, 'automatic_retry': False,
        'only_changed_setting': 'engine_kwargs.max_model_len', 'baseline_context': 98304, 'target_context': 65536,
        'fixed_kv_bytes': 4 * 1024**3, 'reference': old_plan['reference'], 'source_catalog': template['source_catalog'],
        'baseline_profile': ref_for(root, baseline_path), 'baseline_summary': ref_for(root, plans / 'baseline-summary.json'),
        'baseline_raw': ref_for(root, old_run / 'radon0-T1-kv4/raw.jsonl'), 'implementation': implementation,
        'baseline_adapter': {'path': str(Path(REMOTE) / 'code/vlm-psychophysics/classic_precision' / Path(prior.__file__).name),
                             'sha256': file_sha256(prior.__file__)},
        'authorization': ref_for(root, plans / 'authorization.json'),
        'predecessor_queue': {'path': str(Path(REMOTE) / 'plans' / GEMMA / 'radon0-qualification-queue.json'),
                              'sha256': GEMMA_QUEUE_SHA}}
    write_new(plans / 'plan.json', plan); p = copy.deepcopy(template)
    for key in ('implementation', 'media_files', 'worker_token', 'operational_wrapper', 'kv_shared_smoke'): p.pop(key, None)
    rp = Path(REMOTE) / 'plans' / NAME; rr = Path(REMOTE) / 'runs/reproducibility' / NAME
    p.update(worker='radon0-T1-' + ARM, assigned_ids=reference['ids'], maximum_model_calls=48,
        output=str(rr / ('radon0-T1-' + ARM)), claim_root=str(rr / 'claims' / ARM),
        control_file=str(rp / (ARM + '-control.json')), pause_file=str(rp / (ARM + '-PAUSE.json')),
        context_smoke={'arm': ARM, 'plan': ref_for(root, plans / 'plan.json'), 'implementation': implementation})
    p['engine_kwargs']['max_model_len'] = 65536; science(p, Archive(root).ref)
    write_new(plans / (ARM + '-draft.json'), p)
    write_new(plans / (ARM + '-control.json'), {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
    return {'status': 'PREPARED_NOT_LAUNCHED', 'plan': descriptor(plans / 'plan.json'), 'maximum_new_calls': 48}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'seal', 'launch', 'queue-supervisor', 'queue', 'supervise', 'run', 'summarize', 'pause'))
    for key in ('root', 'draft', 'output', 'plan', 'profile', 'control'): parser.add_argument('--' + key)
    a = parser.parse_args()
    if a.command == 'prepare': answer = prepare(a.root)
    elif a.command == 'seal': answer = seal(a.draft, a.output)
    elif a.command == 'launch': answer = launch(a.plan)
    elif a.command == 'queue-supervisor': answer = queue_supervisor(a.plan)
    elif a.command == 'queue': answer = queue(a.plan)
    elif a.command == 'supervise': answer = supervise(a.profile)
    elif a.command == 'run': answer = run(a.profile)
    elif a.command == 'summarize': answer = summarize(a.root, a.plan, a.output)
    else: answer = q.lifecycle.pause(a.control)
    print(json.dumps(answer), flush=True)


if __name__ == '__main__': main()
