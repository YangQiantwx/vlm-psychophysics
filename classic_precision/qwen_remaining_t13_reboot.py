"""Explicit reboot-prefix admission and finite never-claimed Qwen recovery.

An interrupted worker never receives a synthetic result or wait. Its durable
technical-PASS prefix, unresolved claims and never-claimed inputs are distinct.
New work uses fresh claims/controls and the unchanged qualified science runner.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from . import official_x_results as evidence
from . import qwen_remaining_t13 as q
from . import qwen_remaining_t13_storage as storage
from .qwen_remaining_t13_archive import Archive as BaseArchive
from .qwen_remaining_t13_results import validate_gate
from .refinement_worker import read, require, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity

NAME = q.NAME
BATCH = 'reboot1'
MODULE = 'classic_precision.qwen_remaining_t13_reboot'
OLD_BOOT = 'd965a02f-aed8-4af7-9bd6-3c252410767a'
NEW_BOOT = 'bb9cf0ec-41bd-4937-a6a6-6efdb08077ad'
UNCERTAIN = 't3_2e29ac8a4e90ee6c0f7f'
SCHEMA = NAME + '/reboot-interrupted-prefix-v1'
OPERATIONAL = {'schema', 'worker', 'output', 'claim_root', 'pause_file', 'control_file',
               'control_generation', 'assigned_ids', 'maximum_model_calls', 'implementation',
               'worker_token', 'media_files', 'operational_wrapper', 'reboot_entry',
               'continuation_parent', 'continuation_wait', 'retained_prefix_length'}


class Archive(BaseArchive):
    def path(self, value):
        value = str(value); mac = '/Users/qianyang/Desktop/cvpr2027_benchmark'
        if value.startswith(mac + '/'): return self.root / value[len(mac) + 1:]
        return super().path(value)


def descriptor(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def probe(profile, old_boot=OLD_BOOT, new_boot=NEW_BOOT):
    """Read-only Linux probe: no CUDA/NVML import, process kill or file writes."""
    p = read(profile); require(sys.platform.startswith('linux') and socket.gethostname() == p['hostname'], 'Probe on original host')
    current = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    require(current == new_boot and old_boot != current, 'Expected reboot not observed')
    entries = []; unreadable = []; uid = os.getuid()
    markers = (NAME, 'qwen_remaining_t13', 'vllm', 'EngineCore', 'VLLM::Worker')
    for node in Path('/proc').iterdir():
        if not node.name.isdigit(): continue
        try:
            if node.stat().st_uid != uid: continue
            command = (node / 'cmdline').read_bytes(); comm = (node / 'comm').read_text().strip()
            stat = (node / 'stat').read_text(); ticks = stat[stat.rfind(')') + 2:].split()[19]
        except FileNotFoundError: continue
        except PermissionError:
            unreadable.append(int(node.name)); continue
        # The probe and its transport are expected observers, not old engines.
        text = command.decode(errors='replace'); matched = [s for s in markers if s in text or s in comm]
        observer = MODULE in text and 'probe' in text
        entries.append({'pid': int(node.name), 'startticks': ticks, 'comm': comm,
                        'cmdline_sha256': hashlib.sha256(command).hexdigest(),
                        'relevant_markers': matched, 'this_probe_observer': observer})
    relevant = [x for x in entries if x['relevant_markers'] and not x['this_probe_observer']]
    lock = p['physical_lock']; path = Path(lock['path'])
    require(path.is_file() and not path.is_symlink(), 'Original physical lock missing/replaced by symlink')
    free = False
    with path.open('r+b') as stream:
        state = os.fstat(stream.fileno()); same = (state.st_dev, state.st_ino) == (lock['st_dev'], lock['st_ino'])
        if same:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB); free = True
                fcntl.flock(stream, fcntl.LOCK_UN)
            except BlockingIOError: pass
    return {'schema': NAME + '/reboot-host-probe-v1', 'status': 'PASS' if same and free and not relevant and not unreadable else 'HOLD',
            'implementation': descriptor(__file__),
            'hostname': socket.gethostname(), 'old_boot_id': old_boot, 'new_boot_id': current,
            'owner': owner_identity(), 'same_account_uid': uid, 'same_account_process_inventory': entries,
            'relevant_processes': relevant, 'unreadable_same_account_pids': unreadable,
            'physical_lock': {**lock, 'observed_st_dev': state.st_dev, 'observed_st_ino': state.st_ino,
                              'original_inode_matches': same, 'nonblocking_flock_acquired_and_released': free},
            'original_process_wait_observed': False, 'original_returncode': None,
            'GPU_or_NVML_calls': 0, 'mutated_files': 0}


def journal(path):
    """Preserve an interrupted last fragment; never trim or rewrite a journal."""
    path = Path(path)
    if not path.exists(): return [], None
    rows = []; fragment = None
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                fragment = {'bytes': len(line), 'sha256': hashlib.sha256(line).hexdigest()}; break
            if line.strip(): rows.append(json.loads(line))
    return rows, fragment


def unique(rows):
    result = {x['sample_id']: x for x in rows}; require(len(rows) == len(result), 'Duplicate journal ID'); return result


def audit(root, t3_profile, t1_profile, closure_evidence, output):
    """Audit original bytes; emit a separate admission, never an original wait."""
    root = Path(root); arc = Archive(root); p = read(arc.path(t3_profile)); p1 = read(arc.path(t1_profile))
    for profile in (p, p1): storage.science(profile, arc.ref)
    require(p['card'] == p1['card'] == 'radon0' and p['resource_group'] == 'T3' and p1['resource_group'] == 'T1'
            and len(p['assigned_ids']) == 738 and len(p1['assigned_ids']) == 1038, 'Wrong interrupted finite scope')
    proof = read(closure_evidence); lock = proof['physical_lock']
    require(proof['schema'] == NAME + '/reboot-host-probe-v1' and proof['status'] == 'PASS'
            and proof['old_boot_id'] == OLD_BOOT and proof['new_boot_id'] == NEW_BOOT
            and proof['hostname'] == p['hostname'] and proof['relevant_processes'] == []
            and proof['implementation']['sha256'] == file_sha256(__file__)
            and proof['owner']['boot_id'] == NEW_BOOT
            and any(x['this_probe_observer'] for x in proof['same_account_process_inventory'])
            and not any(x['relevant_markers'] and not x['this_probe_observer'] for x in proof['same_account_process_inventory'])
            and proof['unreadable_same_account_pids'] == []
            and lock['original_inode_matches'] and lock['nonblocking_flock_acquired_and_released']
            and all(lock[k] == p['physical_lock'][k] for k in ('path', 'st_dev', 'st_ino'))
            and proof['original_process_wait_observed'] is False and proof['original_returncode'] is None,
            'Genuine changed-boot/process/inode/free evidence required')
    out = arc.path(p['output']); old_wait = out.with_suffix('.wait.json')
    target = Path(output).resolve(strict=False)
    require(target != old_wait.resolve(strict=False) and not target.is_relative_to(out.resolve())
            and not target.name.endswith(('.wait.json', '.launch.json')), 'Admission must remain separate from original lifecycle files')
    require(not old_wait.exists() and not (out / 'result.json').exists()
            and not (out / 'failure.json').exists(), 'This admission only covers the interrupted, missing-wait worker')
    identity = read(out / 'identity.json'); owned = read(out / 'engine-owner.json')['owned']; lease = read(out / 'lease.json')
    require(identity['profile'] == p and identity['profile_sha256'] == digest(p)
            and identity['owner']['boot_id'] == OLD_BOOT and owned
            and all(x['worker']['boot_id'] == OLD_BOOT for x in owned)
            and all(lease[k] == p['physical_lock'][k] for k in ('path', 'st_dev', 'st_ino')),
            'Original identities/lease do not bind the old boot and profile')
    tables = {}; fragments = {}; bindings = {}
    names = ('raw.jsonl', 'events.jsonl', 'request-timings.jsonl', 'deep-observations.jsonl',
             'input-admissions.jsonl', 'custom-selector-admissions.jsonl')
    for name in names:
        values, fragment = journal(out / name); tables[name] = values
        if fragment: fragments[name] = fragment
        if (out / name).exists(): bindings[name] = descriptor(out / name)
    raw = unique(tables['raw.jsonl']); events = tables['events.jsonl']
    intents = unique([x for x in events if x['event'] == 'INTENT']); passed = unique([x for x in events if x['event'] == 'TECHNICAL_PASS'])
    timings = unique(tables['request-timings.jsonl']); deep = unique(tables['deep-observations.jsonl'])
    admissions = unique(tables['input-admissions.jsonl']); selectors = unique(tables['custom-selector-admissions.jsonl'])
    ids = [x['sample_id'] for x in tables['raw.jsonl'] if x['sample_id'] in passed]
    require(ids == p['assigned_ids'][:len(ids)] and len(ids) == len(passed) == 228
            and set(ids) <= set(intents) and set(timings) == set(ids), 'Durable PASS/raw/timing prefix differs')
    claim_root = arc.path(p['claim_root']); claims = {}; claim_bindings = {}
    for path in sorted(claim_root.glob('*.json')):
        require(path.is_file() and not path.is_symlink(), 'Original claim is not a regular file')
        claim = read(path); sid = path.stem
        require(claim['sample_id'] == sid and claim['profile_sha256'] == digest(p)
                and claim['worker_token'] == p['worker_token'] and claim['attempt'] == 1
                and sid in p['assigned_ids'], 'Original claim identity differs')
        claims[sid] = claim; claim_bindings[sid] = descriptor(path)
    uncertain = [s for s in p['assigned_ids'] if s in claims and s not in passed]
    require(uncertain == [UNCERTAIN] and set(claims) == set(ids) | set(uncertain)
            and set(intents) == set(claims) and set(raw) <= set(claims)
            and p['assigned_ids'][228] == UNCERTAIN, 'Unresolved claim or source partition differs')
    versions = read(out / 'versions.json'); runtime = read(out / 'runtime.json'); public = read(out / 'public-settings.json')
    require(all(versions[k] == v for k, v in p['expected_packages'].items() if k in versions), 'Recorded runtime package differs')
    require(all(k in versions for k in ('vllm', 'torch', 'transformers', 'numpy')), 'Core version receipt incomplete')
    for value in runtime: q.qualified.validate_runtime(value)
    require(all(value['actual_kv_capacity'] >= p['minimum_kv_capacity'] for value in runtime), 'Actual KV too small')
    for k, value in p['engine_kwargs'].items():
        require(public['LLM_kwargs'][k] == (sorted(value) if k == 'cpu_offload_params' else value), 'Actual engine differs: ' + k)
    require(public['request_mm_processor_kwargs'] == {'do_resize': False, 'do_sample_frames': False}
            and not public.get('request_adapter') and not public.get('request_mm_processor_kwargs_by_task'), 'Actual request policy differs')
    validate_gate(p, arc.ref); catalog = arc.ref(p['source_catalog'])['items']; records = []
    for ordinal, sid in enumerate(ids):
        value = raw[sid]; item = catalog[sid]; evidence.common_raw_checks(p, item, value, ordinal)
        require(passed[sid]['raw_digest'] == digest(value), 'Original PASS does not bind original raw')
        timing = timings[sid]
        require(timing['end_to_end_durable_seconds'] >= timing['engine_request_inclusive_seconds'] > 0, 'Invalid durable timing')
        if value['deep']:
            d = deep[sid]; a = admissions[sid]; s = selectors[sid]
            require(d['raw_digest'] == a['raw_digest'] == s['raw_digest'] == digest(value)
                    and a['observations_digest'] == digest(d['observations']), 'Deep/raw/admission binding differs')
            require(a['proof']['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS'
                    and a['proof']['CPU_to_GPU_patch_bytes_equal'] and a['proof']['source_frame_indices'] == list(range(180)),
                    'All-frame official CPU/GPU input proof missing')
            require(s['variant'] == 'X_QSA' and s['pure_official_config'] is False
                    and s['actual_request_calls'] and all(n > 0 for n in s['actual_request_calls']), 'Actual QSA proof missing')
            for worker in d['observations']: q.qualified.validate_runtime(worker['runtime'])
            for selected in s['selector_receipts']:
                require(selected['status'] == 'APPLIED_X_QSA' and selected['selector_source_sha256'] == p['selector_source']['sha256']
                        and selected['actual_function_is_canonical_wrapper'] and selected['native_function_verified_before_replacement']
                        and not selected['SDK_files_modified'] and not selected['forward_processor_or_weights_changed'], 'Selector source/scope differs')
        records.append({'sample_id': sid, 'task': value['task'], 'truth': value['truth'], 'answer': value['raw_answer'].strip(),
                        'raw_answer': value['raw_answer'], 'correct': value['correct'], 'eligible': True,
                        'parameters': item['row']['source_parameters'], 'source_media': evidence.receipt_media(value),
                        'output_token_ids': value['output_token_ids'], 'finish_reason': value['finish_reason'], 'stop_reason': value['stop_reason'],
                        'prompt_token_sha256': digest(value['actual_prompt_token_ids']), 'raw_digest': digest(value),
                        'profile_sha256': digest(p), 'worker': p['worker'], 'deep': value['deep'], 'timing': timing,
                        'admission': 'REBOOT_INTERRUPTED_DURABLE_TECHNICAL_PASS_PREFIX'})
    require([x['sample_id'] for x in records if x['deep']] == [ids[0]], 'First-task GPU evidence missing/duplicated')
    require(set(deep) - set(uncertain) == set(admissions) - set(uncertain) == set(selectors) - set(uncertain) == {ids[0]}, 'Deep ledger scope differs')
    out1 = arc.path(p1['output']); claims1 = arc.path(p1['claim_root'])
    require(not out1.exists() and not out1.with_suffix('.launch.json').exists()
            and not out1.with_suffix('.wait.json').exists() and not list(claims1.glob('*.json')), 'T1 is not entirely unattempted')
    for name in ('identity.json', 'engine-owner.json', 'lease.json', 'runtime.json', 'versions.json', 'public-settings.json', 'load.json'):
        bindings[name] = descriptor(out / name)
    remaining = {'T3': [s for s in p['assigned_ids'] if s not in claims], 'T1': p1['assigned_ids']}
    require({k: len(v) for k, v in remaining.items()} == {'T3': 509, 'T1': 1038}, 'Never-claimed count differs')
    result = {'schema': SCHEMA, 'status': 'ACCEPTED_REBOOT_INTERRUPTED_PREFIX', 'batch': BATCH,
              'original_profiles': {'T3': descriptor(arc.path(t3_profile)), 'T1': descriptor(arc.path(t1_profile))},
              'closure_evidence': descriptor(closure_evidence), 'old_boot_id': OLD_BOOT, 'new_boot_id': NEW_BOOT,
              'original_wait': None, 'original_returncode': None, 'real_original_wait_observed': False,
              'old_owned_identities_terminated_by_boot_change': True, 'original_physical_lease_free_on_new_boot': True,
              'accepted_prefix_ids': ids, 'UNCERTAIN_ids': uncertain, 'never_claimed': remaining,
              'maximum_new_never_claimed_calls': 1547, 'automatic_uncertain_retry': False,
              'original_claims': claim_bindings, 'original_evidence': bindings, 'preserved_partial_tails': fragments,
              'original_controls': {g: read(arc.path(v['control_file'])) for g, v in [('T3', p), ('T1', p1)]},
              'accepted_records': records, 'new_model_calls': 0,
              'completion_boundary': '228 accepted original answers; 1 unresolved original claim. This is not full6576 completion or a clean-wait certificate.'}
    write_new(output, result)
    return {'status': result['status'], 'accepted': len(ids), 'UNCERTAIN_ids': uncertain,
            'never_claimed': {k: len(v) for k, v in remaining.items()}, 'receipt': descriptor(output), 'new_model_calls': 0}


def verify_snapshot(receipt, arc):
    require(receipt['schema'] == SCHEMA and receipt['status'] == 'ACCEPTED_REBOOT_INTERRUPTED_PREFIX'
            and receipt['real_original_wait_observed'] is False and receipt['original_returncode'] is None
            and receipt['UNCERTAIN_ids'] == [UNCERTAIN] and receipt['automatic_uncertain_retry'] is False,
            'Wrong interrupted-prefix admission')
    closure = arc.ref(receipt['closure_evidence'])
    require(closure['status'] == 'PASS' and closure['old_boot_id'] == OLD_BOOT and closure['new_boot_id'] == NEW_BOOT
            and closure['implementation']['sha256'] == file_sha256(__file__), 'Bound reboot closure evidence changed')
    for ref in receipt['original_evidence'].values(): arc.ref(ref) if ref['path'].endswith('.json') else require(file_sha256(arc.path(ref['path'])) == ref['sha256'], 'Original journal bytes changed')
    p = arc.ref(receipt['original_profiles']['T3']); root = arc.path(p['claim_root'])
    require({x.stem for x in root.glob('*.json')} == set(receipt['original_claims']), 'Original claims changed')
    for ref in receipt['original_claims'].values(): arc.ref(ref)
    require(not arc.path(p['output']).with_suffix('.wait.json').exists()
            and not (arc.path(p['output']) / 'result.json').exists(), 'Original missing wait/result was replaced')
    p1 = arc.ref(receipt['original_profiles']['T1'])
    require(not arc.path(p1['output']).exists() and not list(arc.path(p1['claim_root']).glob('*.json')), 'Retired original T1 was attempted')


def science(p, resolve=None):
    if 'reboot_entry' not in p: return storage.science(p, resolve or q.verified)
    arc = Archive(p['workspace_root']) if resolve is None else None
    resolve = resolve or arc.ref; entry = p['reboot_entry']; receipt = resolve(entry['admission'])
    require(entry['implementation']['sha256'] == file_sha256(__file__) and entry['batch'] == BATCH, 'Reboot wrapper changed')
    group = p['resource_group']; original = resolve(entry['original_profile'])
    storage.science(original, resolve)
    require(entry['original_profile']['sha256'] == receipt['original_profiles'][group]['sha256'], 'Wrong original recovery template')
    require(receipt['schema'] == SCHEMA and receipt['status'] == 'ACCEPTED_REBOOT_INTERRUPTED_PREFIX'
            and receipt['UNCERTAIN_ids'] == [UNCERTAIN] and not receipt['automatic_uncertain_retry'], 'Wrong reboot scope')
    require(all(p.get(k) == original.get(k) for k in p.keys() | original.keys() if k not in OPERATIONAL), 'Recovery changed frozen science/resource configuration')
    expected = receipt['never_claimed'][group]
    ids = p['assigned_ids']
    require(ids and len(ids) == len(set(ids)) == p['maximum_model_calls'] and set(ids) <= set(expected)
            and UNCERTAIN not in ids and not set(ids) & set(receipt['accepted_prefix_ids']), 'Recovery reissues answered/claimed work')
    if p.get('continuation_parent'):
        parent = resolve(p['continuation_parent'])
        require(parent['reboot_entry'] == entry and p['control_generation'] == parent['control_generation'] + 1
                and ids == parent['assigned_ids'][p['retained_prefix_length']:], 'Recovery continuation is not exact suffix')
        require(all(p.get(k) == parent.get(k) for k in p.keys() | parent.keys() if k not in OPERATIONAL), 'Recovery continuation changed science')
    else: require(ids == expected and p['control_generation'] == 0, 'Initial never-claimed allocation changed')
    root = Path(p['workspace_root']); worker = 'radon0-full-' + group + '-' + BATCH + '-segment' + str(p['control_generation'])
    require(p['worker'] == worker and p['card'] == 'radon0'
            and Path(p['output']) == root / 'runs/reproducibility' / NAME / worker
            and Path(p['claim_root']) == root / 'runs/reproducibility' / NAME / ('claims-' + BATCH) / 'radon0' / group
            and Path(p['control_file']) == root / 'plans' / NAME / BATCH / ('radon0-' + group + '-control.json'), 'Recovery namespace differs')


def fence(root, admission):
    arc = Archive(root); receipt = read(admission); verify_snapshot(receipt, arc)
    p = arc.ref(receipt['original_profiles']['T3'])
    require(sys.platform.startswith('linux') and socket.gethostname() == p['hostname']
            and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == NEW_BOOT, 'Fence on the actual recovered host')
    for group in ('T3', 'T1'):
        p = arc.ref(receipt['original_profiles'][group]); path = arc.path(p['control_file']); value = read(path)
        require(value['generation'] in (p['control_generation'], p['control_generation'] + 1), 'Unexpected original generation')
        value.update(paused=True, generation=p['control_generation'] + 1, fenced_by_reboot_admission=descriptor(admission))
        atomic_json(path, value)
    return {'status': 'ORIGINAL_CONTROLS_FENCED', 'model_calls': 0}


def check_fence(p):
    arc = Archive(p['workspace_root']); receipt = arc.ref(p['reboot_entry']['admission']); verify_snapshot(receipt, arc)
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == receipt['new_boot_id'], 'Another reboot requires a new closure audit')
    for group in ('T3', 'T1'):
        original = arc.ref(receipt['original_profiles'][group]); c = read(arc.path(original['control_file']))
        require(c['paused'] and c['generation'] == original['control_generation'] + 1
                and c['fenced_by_reboot_admission']['sha256'] == p['reboot_entry']['admission']['sha256'], 'Original queue is not fenced')


def prepare(root, admission, output):
    root = Path(root).resolve(); arc = Archive(root); receipt = read(admission); verify_snapshot(receipt, arc)
    plans = Path(output); require(not plans.exists(), 'Recovery preparation exists'); plans.mkdir(parents=True)
    profiles = []
    for group in ('T3', 'T1'):
        original = arc.ref(receipt['original_profiles'][group]); p = copy.deepcopy(original); remote = p['workspace_root']
        for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper'): p.pop(key, None)
        worker = 'radon0-full-' + group + '-' + BATCH + '-segment0'; control = plans / ('radon0-' + group + '-control.json')
        write_new(control, {'generation': 0, 'paused': True, 'status': 'PREPARED_NOT_LAUNCHED'})
        p.update(worker=worker, assigned_ids=receipt['never_claimed'][group], maximum_model_calls=len(receipt['never_claimed'][group]),
                 output=str(Path(remote) / 'runs/reproducibility' / NAME / worker),
                 claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / ('claims-' + BATCH) / 'radon0' / group),
                 control_file=str(Path(remote) / control.relative_to(root)), control_generation=0,
                 pause_file=str(Path(remote) / plans.relative_to(root) / (worker + '-PAUSE.json')),
                 reboot_entry={'batch': BATCH, 'admission': q.remote_ref(Path(admission), root, remote),
                    'original_profile': receipt['original_profiles'][group],
                    'implementation': {'path': str(Path(remote) / 'code/vlm-psychophysics/classic_precision/qwen_remaining_t13_reboot.py'), 'sha256': file_sha256(__file__)}})
        science(p, arc.ref); path = plans / (worker + '-draft.json'); write_new(path, p); profiles.append(descriptor(path))
    return {'status': 'PREPARED_PAUSED_NOT_LAUNCHED', 'drafts': profiles, 'maximum_new_model_calls': 1547,
            'UNCERTAIN_ids_not_reissued': receipt['UNCERTAIN_ids'], 'new_model_calls': 0}


@contextmanager
def execution_context():
    old_science = q.science; old_closed = q.closed; native = subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.qwen_remaining_t13', MODULE) for x in argv], **kwargs)
    q.science = science; subprocess.Popen = popen
    try: yield
    finally: q.science = old_science; q.closed = old_closed; subprocess.Popen = native


def source_admission(p):
    check_fence(p); items = q.verified(p['source_catalog'])['items']
    storage.check_media_roots([m['path'] for sid in p['assigned_ids'] for m in items[sid]['content']
                              if m['type'] in ('image', 'video')], storage.ALLOWED_ROOT)


def seal(draft, output):
    p = read(draft); science(p); source_admission(p)
    require(p['hostname'] == socket.gethostname(), 'Seal on original execution host')
    catalog = q.verified(p['source_catalog'])['items']; files = {}
    for sid in p['assigned_ids']:
        for entry in catalog[sid]['content']:
            if entry['type'] in ('video', 'image'): files[entry['path']] = entry['sha256']
    def hashed(pair):
        path, expected = pair; before = q.base.stat(path); value = file_sha256(path)
        require(value == expected and q.base.stat(path) == before, 'Source changed during parallel seal')
        return path, (value, before)
    with ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 2))) as pool:
        verified_hashes = dict(pool.map(hashed, files.items()))
    original_sha = q.base.file_sha256
    def cached_sha(path):
        key = str(path)
        if key not in verified_hashes: return original_sha(path)
        value, before = verified_hashes[key]; require(q.base.stat(path) == before, 'Source changed after parallel seal')
        return value
    try:
        q.base.file_sha256 = cached_sha
        with execution_context(): return q.seal(draft, output)
    finally: q.base.file_sha256 = original_sha


def run(profile):
    p = read(profile); science(p); source_admission(p)
    with execution_context(): return q.run(profile)


def supervise(profile):
    science(read(profile))
    with execution_context(): return q.supervise(profile)


def queue(plan):
    for ref in read(plan)['profiles']: source_admission(q.verified(ref))
    with execution_context(): return q.queue(plan)


def activate(plan):
    for ref in read(plan)['profiles']: source_admission(q.verified(ref))
    with execution_context(): return q.activate(plan)


def closed(profile):
    with execution_context(): return q.closed(profile)


def prepare_resume(profile, output):
    p = read(profile)
    with q.lifecycle.controller_lock(str(Path(p['control_file']).with_suffix('.lock'))):
        p, terminal, raw = closed(profile); c = read(p['control_file'])
        require(c['paused'] and c['generation'] == p['control_generation'], 'Explicit unbranched recovery pause required')
        ids = q.lifecycle.remaining(p, terminal, raw); require(ids, 'Nothing never-attempted remains')
        require(not any((Path(p['claim_root']) / (s + '.json')).exists() for s in ids), 'Unresolved recovery claim; no retry')
        value = copy.deepcopy(p); generation = c['generation'] + 1
        for key in ('implementation', 'worker_token', 'media_files', 'operational_wrapper'): value.pop(key, None)
        worker = 'radon0-full-' + p['resource_group'] + '-' + BATCH + '-segment' + str(generation)
        value.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
                     output=str(Path(p['output']).parent / worker), pause_file=str(Path(output).with_suffix('.PAUSE.json')),
                     continuation_parent=descriptor(profile), continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')),
                     retained_prefix_length=len(raw))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
            json.dump(value, stream); temporary = stream.name
        try: result = seal(temporary, output)
        finally: Path(temporary).unlink()
        queue_path = Path(output).with_name(worker + '-queue.json')
        q.write_queue(queue_path, [descriptor(output)], p['workspace_root'])
        c.update(generation=generation, paused=True, continuation_profile=descriptor(output)); atomic_json(p['control_file'], c)
    return {**result, 'status': 'CONTINUATION_PREPARED_PAUSED', 'queue': descriptor(queue_path), 'model_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('probe', 'audit', 'prepare', 'fence', 'seal', 'run', 'supervise', 'queue', 'activate', 'pause', 'prepare-resume', 'audit-closed'))
    for name in ('root', 'profile', 't3-profile', 't1-profile', 'closure-evidence', 'admission', 'draft', 'output', 'plan', 'control'): parser.add_argument('--' + name)
    parser.add_argument('--old-boot-id', default=OLD_BOOT); parser.add_argument('--new-boot-id', default=NEW_BOOT); a = parser.parse_args()
    if a.command == 'probe': value = probe(a.profile, a.old_boot_id, a.new_boot_id)
    elif a.command == 'audit': value = audit(a.root, a.t3_profile, a.t1_profile, a.closure_evidence, a.output)
    elif a.command == 'prepare': value = prepare(a.root, a.admission, a.output)
    elif a.command == 'fence': value = fence(a.root, a.admission)
    elif a.command == 'seal': value = seal(a.draft, a.output)
    elif a.command == 'run': value = run(a.profile)
    elif a.command == 'supervise': value = supervise(a.profile)
    elif a.command == 'queue': value = queue(a.plan)
    elif a.command == 'activate': value = activate(a.plan)
    elif a.command == 'prepare-resume': value = prepare_resume(a.profile, a.output)
    elif a.command == 'audit-closed':
        _, terminal, rows = closed(a.profile); value = {'status': 'CLOSED_PASS', 'terminal': terminal['status'], 'n': len(rows)}
    else: value = q.lifecycle.pause(a.control)
    print(json.dumps(value), flush=True)


if __name__ == '__main__': main()
