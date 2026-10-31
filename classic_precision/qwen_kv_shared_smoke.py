"""Authorized 48-ID Radon T1 replay: original KV8, then KV4 only.

The frozen scientific runner is reused unchanged. This module binds the two
allocations, observes other-account GPU activity, compares durable outputs,
and supplies a one-shot queue with real process waits. No automatic retries.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

from . import qwen_remaining_t13 as q
from . import qwen_remaining_t13_archive as archive
from .official_x_results import Archive, journal, unique
from .refinement_worker import read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity, _append

NAME = 'qwen-x-kv-shared-smoke-20261031'
MODULE = 'classic_precision.qwen_kv_shared_smoke'
REMOTE = q.ROOTS['radon']
FIELDS = ('raw_answer', 'output_token_ids', 'finish_reason', 'stop_reason',
          'truth', 'question', 'generation')
ARMS = {'kv8': 8, 'kv4': 4}
ORIGINAL_SCIENCE, ORIGINAL_CLOSED = q.science, q.closed


def descriptor(path):
    return {'path': os.path.abspath(os.fspath(path)), 'sha256': file_sha256(path)}


def ref_for(root, path):
    return {'path': str(Path(REMOTE) / Path(path).relative_to(root)), 'sha256': file_sha256(path)}


def science(p, resolve=None):
    resolve = archive.portable_resolver(p, resolve)
    require('kv_shared_smoke' in p, 'Not an authorized KV smoke profile')
    entry = p['kv_shared_smoke']; plan = resolve(entry['plan'])
    require(entry['implementation']['sha256'] == file_sha256(__file__)
            and plan['implementation']['sha256'] == file_sha256(__file__), 'Smoke adapter changed')
    require(plan['schema'] == NAME + '/v1' and plan['maximum_new_model_calls'] == 96
            and plan['arms'] == ARMS and not plan['automatic_retry'], 'Finite experiment changed')
    auth = resolve(plan['authorization'])
    require(auth['status'] == 'AUTHORIZED' and auth['maximum_new_model_calls'] == 96
            and auth['cards'] == ['radon0'] and auth['reference_sha256'] == plan['reference']['sha256']
            and auth['user_message'] == '开始在randon测试解答我的疑惑吧。配置不能变哦'
            and auth['only_scientific_change_between_arms'] == 'engine_kwargs.kv_cache_memory_bytes',
            'Exact finite authorization differs')
    template = resolve(plan['runtime_profile'])
    q.check_implementation(template['implementation_sources'])
    original = copy.deepcopy(template)
    require(original['engine_kwargs'].pop('allowed_local_media_path') == '/home/q/cvpr2027_datasets_ssd',
            'Original SSD media allowance differs')
    q.validate_template(original, 'T1')
    require(template['card'] == 'radon0' and template['phase'] == 'full'
            and template['resource_group'] == 'T1', 'Wrong original T1 deployment')
    arm = entry['arm']; require(arm in ARMS, 'Unknown arm')
    engine = copy.deepcopy(template['engine_kwargs'])
    engine['kv_cache_memory_bytes'] = ARMS[arm] * 1024**3
    require(p['engine_kwargs'] == engine, 'Only the authorized KV capacity may differ')
    operational = {'assigned_ids', 'maximum_model_calls', 'worker', 'output', 'claim_root',
                   'control_file', 'pause_file', 'source_catalog', 'phase', 'kv_shared_smoke',
                   'implementation', 'worker_token', 'media_files', 'operational_wrapper', 'engine_kwargs'}
    require(all(p.get(k) == template.get(k) for k in p.keys() | template.keys() if k not in operational),
            'Original scientific/deployment field changed')
    reference = resolve(plan['reference']); ids = plan['ids']
    require(ids == reference['ids'] == p['assigned_ids'] and len(ids) == len(set(ids)) == 48
            and p['maximum_model_calls'] == 48 and p['phase'] == 'deployment_smoke'
            and p['control_generation'] == 0, 'Smoke quota or phase changed')
    root = Path(p['workspace_root']); run = root / 'runs/reproducibility' / NAME
    worker = 'radon0-T1-' + arm
    require(p['worker'] == worker and Path(p['output']) == run / worker
            and Path(p['claim_root']) == run / 'claims' / arm
            and Path(p['control_file']) == root / 'plans' / NAME / (arm + '-control.json')
            and Path(p['pause_file']) == root / 'plans' / NAME / (arm + '-PAUSE.json')
            and p['source_catalog'] == plan['source_catalog'], 'Independent attempt namespace changed')
    canonical = resolve(p['canonical_catalog'])['items']; catalog = resolve(p['source_catalog'])['items']
    require(set(catalog) == set(ids), 'Foreign or missing source')
    for sid in ids:
        item = catalog[sid]; q.validate_source(sid, item, 'T1')
        require(item == q.rebase(canonical[sid], str(root)), 'Original input/GT/prompt changed')
        params = item['row']['source_parameters']
        require(params['sf_cycles_per_aperture'] == 4 and params['modulation_depth'] == .9,
                'Different selected point')
        r = reference['outputs'][sid]
        require(r['truth'] == item['row']['truth'] and r['question'] == item['row']['prompt']
                and r['source_media'] == [[m['type'], m['sha256']] for m in item['content'] if 'sha256' in m],
                'Reference source/GT/prompt changed')


def compare_raw(reference, raw):
    expected = reference['outputs'][raw['sample_id']]
    equal = {k: raw[k] == expected[k] for k in FIELDS}
    equal['actual_prompt_token_ids'] = raw['actual_prompt_token_ids'] == reference['actual_prompt_token_ids']
    equal['source_media'] = [[m['type'], m['sha256']] for m in raw['source']['media']] == expected['source_media']
    return {'sample_id': raw['sample_id'], 'raw_digest': digest(raw), 'equal': equal,
            'all_equal': all(equal.values()), 'differing_fields': [k for k, v in equal.items() if not v]}


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
    run = Path(p['output']).parent.resolve(strict=True)
    require(str(run) == '/home/q/cvpr2027_runs_ssd/' + NAME, 'Smoke run must be on the selected SSD')
    sample = next(iter(verified(p['source_catalog'])['items'].values()))['content'][0]['path']
    devices = {run.stat().st_dev, Path(sample).stat().st_dev, Path(p['checkpoint']).stat().st_dev}
    require(devices == {66306}, 'Weights/data/run must remain on the verified NVMe device')
    with context(): return q.seal(draft, output)


def run(profile):
    p = read(profile); science(p); plan = verified(p['kv_shared_smoke']['plan'])
    reference = verified(plan['reference']); original_append = q.base._append; seen = set(); pending = {}
    def observed_append(path, value):
        out = Path(p['output'])
        if Path(path) == out / 'events.jsonl' and value.get('event') == 'TECHNICAL_PASS':
            # Let the frozen runner collect its first-request GPU evidence and
            # complete input checks before deciding output equivalence.
            receipt = pending.pop(value['sample_id'])
            original_append(out / 'output-equivalence.jsonl', receipt)
            if not receipt['all_equal']:
                atomic_json(out / 'reproduction-failure.json', {**receipt, 'status': 'STOP_QWEN_SMOKE_NO_RETRY',
                    'arm': p['kv_shared_smoke']['arm'], 'original_outputs_preserved': True,
                    'Fourier_Gemma_action': 'CONTINUE_UNCHANGED'})
                raise ValueError('Qwen smoke output differs; raw retained; no further arm or automatic retry')
        original_append(path, value)  # Scientific raw is durable before comparison.
        if Path(path) == out / 'events.jsonl' and value.get('event') == 'INTENT':
            observation = shared_observation(p, value['sample_id'])
            original_append(out / 'shared-gpu-observations.jsonl', observation)
        if Path(path) != out / 'raw.jsonl': return
        require(value['sample_id'] not in seen, 'Duplicate raw identity')
        seen.add(value['sample_id']); pending[value['sample_id']] = compare_raw(reference, value)
    try:
        with context():
            q.base._append = observed_append
            return q.run(profile)
    finally: q.base._append = original_append


def supervise(profile):
    science(read(profile))
    with context(execution=True): return q.supervise(profile)


def closed(profile):
    p, terminal, raw = ORIGINAL_CLOSED(profile)
    require(terminal['status'] == 'COMPLETE', 'Arm must close COMPLETE before its successor')
    ref = verified(verified(p['kv_shared_smoke']['plan'])['reference'])
    actual = journal(Path(p['output']) / 'raw.jsonl')
    comparisons = journal(Path(p['output']) / 'output-equivalence.jsonl')
    require(comparisons == [compare_raw(ref, x) for x in actual]
            and len(comparisons) == 48 and all(x['all_equal'] for x in comparisons),
            'Incomplete or unequal arm; no successor')
    observations = journal(Path(p['output']) / 'shared-gpu-observations.jsonl')
    require([x['label'] for x in observations] == p['assigned_ids'], 'Missing per-request sharing record')
    return p, terminal, raw


def queue(plan_path):
    plan = read(plan_path)
    require(len(plan['profiles']) == 2 and plan['automatic_retry'] is False, 'Exactly two finite arms required')
    for arm, ref in zip(ARMS, plan['profiles']):
        p = verified(ref); science(p); require(p['kv_shared_smoke']['arm'] == arm, 'Arm order changed')
    with context(execution=True): return q.queue(plan_path)


def queue_supervisor(plan_path):
    plan = read(plan_path); p = verified(plan['profiles'][0]); run = Path(p['output']).parent
    intent = read(run / 'queue-launch-intent.json')
    require(intent['queue'] == descriptor(plan_path) and sys.executable == p['python'], 'Launch binding/python differs')
    write_new(run / 'queue-owner.json', owner_identity())
    started = time.monotonic()
    with (run / 'queue-child.log').open('xb') as log:
        child = subprocess.Popen([p['python'], '-B', '-m', MODULE, 'queue', '--plan', plan_path],
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        code = child.wait()
    write_new(run / 'queue-wait.json', {'returncode': code, 'real_wait_observed': True,
        'child_pid': child.pid, 'queue': descriptor(plan_path), 'elapsed_monotonic_seconds': time.monotonic()-started,
        'automatic_retry': False})
    require(code == 0, 'Finite smoke queue failed; no retry')
    return read(plan['state_path'])


def launch(plan_path):
    plan = read(plan_path); p = verified(plan['profiles'][0]); run = Path(p['output']).parent
    require(socket.gethostname() == p['hostname'], 'Launch only on the execution host')
    for ref in plan['profiles']: science(verified(ref)); q.fresh(verified(ref))
    require(not (run / 'queue-launch-intent.json').exists(), 'Queue already has a launch intent')
    observation = shared_observation(p, 'before_launch')
    write_new(run / 'launch-shared-gpu-observation.json', observation)
    require(observation['other_account_compute_active'], 'No active other-account compute; hold before model calls')
    write_new(run / 'queue-launch-intent.json', {'queue': descriptor(plan_path), 'owner': owner_identity(),
        'maximum_new_model_calls': 96, 'automatic_retry': False})
    with context(): q.activate(plan_path)
    env = dict(os.environ)
    for k in p['cleared_environment']: env.pop(k, None)
    env.update(p['environment'])
    with (run / 'queue-supervisor.log').open('xb') as log:
        child = subprocess.Popen([p['python'], '-B', '-m', MODULE, 'queue-supervisor', '--plan', plan_path],
            cwd=str(Path(p['workspace_root']) / 'code/vlm-psychophysics'), env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    ticks = Path('/proc/' + str(child.pid) + '/stat').read_text().rsplit(')', 1)[1].split()[19]
    answer = {'status': 'LAUNCHED_ONCE', 'queue': descriptor(plan_path), 'pid': child.pid,
        'process_start_ticks': ticks, 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'owner': owner_identity(), 'automatic_retry': False}
    write_new(run / 'queue-launch.json', answer)
    return answer


def summarize(root, plan_path, output):
    arc = Archive(root); plan = read(plan_path); proofs = []; raws = {}; active = {}
    from .qwen_remaining_t13_results import audit_segment
    with context():
        for ref in plan['profiles']:
            path = arc.path(ref['path']); p = arc.ref(ref); arm = p['kv_shared_smoke']['arm']
            science(p, arc.ref); records, proof = audit_segment(root, path)
            require(len(records) == 48, 'Incomplete arm')
            source = arc.path(p['output']); rows = unique(journal(source / 'raw.jsonl'))
            expected = arc.ref(arc.ref(p['kv_shared_smoke']['plan'])['reference'])
            comparisons = journal(source / 'output-equivalence.jsonl')
            require(comparisons == [compare_raw(expected, rows[s]) for s in p['assigned_ids']]
                    and all(x['all_equal'] for x in comparisons), 'Output equivalence incomplete')
            observations = journal(source / 'shared-gpu-observations.jsonl')
            require([x['label'] for x in observations] == p['assigned_ids'], 'Sharing coverage incomplete')
            active[arm] = sum(x['other_account_compute_active'] for x in observations)
            raws[arm] = rows; proofs.append(proof)
    require(list(raws) == list(ARMS), 'Missing or repeated arm')
    pairs = [{**compare_raw({'outputs': {sid: {**raws['kv8'][sid], 'source_media':
                [[x['type'], x['sha256']] for x in raws['kv8'][sid]['source']['media']]}},
                'actual_prompt_token_ids': raws['kv8'][sid]['actual_prompt_token_ids']}, row)}
             for sid, row in raws['kv4'].items()]
    require(len(pairs) == 48 and all(x['all_equal'] for x in pairs), 'Paired KV comparison differs')
    outer = read(arc.path(arc.ref(plan['profiles'][0])['output']).parent / 'queue-wait.json')
    require(outer['returncode'] == 0 and outer['real_wait_observed'], 'Outer queue not cleanly closed')
    result = {'status': 'PASS_QWEN_T1_KV_SHARED_POINT' if all(n == 48 for n in active.values())
              else 'PASS_OUTPUT_EQUIVALENCE_SHARING_PARTIAL', 'new_model_calls': 96, 'unique_ids': 48,
        'full_output_and_prompt_equal': True, 'kv8_vs_kv4_equal': True, 'active_other_account_observations': active,
        'task': 'T1', 'cell': {'sf_cycles_per_aperture': 4, 'modulation_depth': .9},
        'context_both': 98304, 'correct_each': sum(x['correct'] for x in raws['kv8'].values()),
        'workers': proofs, 'paired': pairs, 'included_in_formal_results': False, 'original_formal_results_unchanged': True}
    write_new(output, result); return {'status': result['status'], 'summary': descriptor(output)}


def prepare(root):
    root = Path(root); plans = root / 'plans' / NAME; require(not plans.exists(), 'Prepared smoke already exists')
    audit = read(root / 'runs/reproducibility/x-shared-gpu-audit-20261031/summary.json')
    proposal = audit['focused_qwen_full_cell_proposal']; refs = proposal['reference_records']
    require(len(refs) == 48 and sum(x['correct'] for x in refs) == 27, 'Frozen 48-ID point differs')
    catalog_path = root / 'plans' / q.NAME / 'canonical-sources.json'; catalog = read(catalog_path)['items']
    old_template = root / 'plans' / q.NAME / 'handoff1-reboot1/radon0-full-T1-handoff1-reboot1-segment0-profile.json'
    template = read(old_template); q.check_implementation(template['implementation_sources'])
    targets = {x['sample_id']: x for x in refs}; saved = {}; prompt = None; raw_bindings = {}
    for worker in sorted({x['worker'] for x in refs}):
        path = root / 'runs/reproducibility' / q.NAME / worker / 'raw.jsonl'
        with path.open() as f:
            for line in f:
                row = json.loads(line); sid = row['sample_id']
                if sid not in targets or targets[sid]['worker'] != worker: continue
                require(sid not in saved and digest(row) == targets[sid]['raw_digest'], 'Reference raw changed or repeated')
                require(digest(row['actual_prompt_token_ids']) == targets[sid]['prompt_token_sha256'], 'Reference prompt changed')
                if prompt is None: prompt = row['actual_prompt_token_ids']
                require(row['actual_prompt_token_ids'] == prompt, 'Reference prompts are not identical')
                saved[sid] = {**{k: row[k] for k in FIELDS}, 'source_media': targets[sid]['source_media'],
                    'original_raw_digest': digest(row), 'original_worker': worker, 'correct': row['correct']}
        raw_bindings[worker] = descriptor(path)
    require(set(saved) == set(targets), 'Missing full original outputs')
    ids = sorted(saved); plans.mkdir(parents=True)
    (plans / 'baseline-runtime-profile.json').write_bytes(old_template.read_bytes())
    write_new(plans / 'reference.json', {'ids': ids, 'actual_prompt_token_ids': prompt, 'outputs': saved,
        'raw_bindings': raw_bindings, 'canonical_selection': proposal['reference']})
    write_new(plans / 'sources.json', {'items': {sid: q.rebase(catalog[sid], REMOTE) for sid in ids}})
    write_new(plans / 'authorization.json', {'status': 'AUTHORIZED', 'cards': ['radon0'],
        'user_message': '开始在randon测试解答我的疑惑吧。配置不能变哦', 'maximum_new_model_calls': 96,
        'reference_sha256': file_sha256(plans / 'reference.json'),
        'only_scientific_change_between_arms': 'engine_kwargs.kv_cache_memory_bytes',
        'context_both': 98304, 'original_records_preserved': True, 'automatic_retry': False,
        'recorded_HST': datetime.now(ZoneInfo('Pacific/Honolulu')).isoformat()})
    module_ref = {'path': str(Path(REMOTE) / 'code/vlm-psychophysics/classic_precision' / Path(__file__).name),
                  'sha256': file_sha256(__file__)}
    plan = {'schema': NAME + '/v1', 'ids': ids, 'maximum_new_model_calls': 96, 'automatic_retry': False,
        'arms': ARMS, 'reference': ref_for(root, plans / 'reference.json'),
        'authorization': ref_for(root, plans / 'authorization.json'),
        'runtime_profile': ref_for(root, plans / 'baseline-runtime-profile.json'),
        'source_catalog': ref_for(root, plans / 'sources.json'), 'implementation': module_ref}
    write_new(plans / 'plan.json', plan)
    for arm, kv in ARMS.items():
        p = copy.deepcopy(template)
        for k in ('implementation', 'media_files', 'worker_token', 'operational_wrapper'): p.pop(k, None)
        worker = 'radon0-T1-' + arm; remote_plans = Path(REMOTE) / 'plans' / NAME
        remote_run = Path(REMOTE) / 'runs/reproducibility' / NAME
        p.update(worker=worker, phase='deployment_smoke', assigned_ids=ids, maximum_model_calls=48,
            output=str(remote_run / worker), claim_root=str(remote_run / 'claims' / arm),
            control_file=str(remote_plans / (arm + '-control.json')), pause_file=str(remote_plans / (arm + '-PAUSE.json')),
            source_catalog=plan['source_catalog'], kv_shared_smoke={'arm': arm, 'plan': ref_for(root, plans / 'plan.json'),
                'implementation': module_ref})
        p['engine_kwargs']['kv_cache_memory_bytes'] = kv * 1024**3
        science(p, Archive(root).ref)
        write_new(plans / (arm + '-draft.json'), p)
        write_new(plans / (arm + '-control.json'), {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
    return {'status': 'PREPARED_NOT_LAUNCHED', 'plan': descriptor(plans / 'plan.json'), 'maximum_new_calls': 96}


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
