"""Forty source-revision T2 calls on the unchanged, qualified natural Qwen X stack.

Only admission, lifecycle, and a new all-or-none 3840-record overlay change.
Historical source, calls, claims, and results are never rewritten.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import tempfile
import subprocess
import time

from . import nemo_x_full as lifecycle
from . import qwen_x_crosscard as qualified
from . import official_x_results as results
from .refinement_worker import descriptor, read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256, owner_identity

NAME = 'natural-t2-spoon-20261030'
PARENT = 'natural-t2-qwen-x-20261029'
CLASSIC = 'qwen-x-classic-t2-20261029'
SOURCE = 'natural-spoon-replacement-20261030'
CARDS = ('fourier0', 'fourier1', 'radon0')
COUNTS = {'fourier0': 12, 'fourier1': 8, 'radon0': 20}
ROOTS = qualified.ROOTS
USER_MESSAGE = '还有真实木勺有少量红色背景残边这个要不要换个新的，就是在素材网上，然后有关的视频替换重跑一下？'
PARAMETERS = ('block', 'group', 'background_id', 'object_id', 'delta_Y', 'onset_frame', 'offset_frame', 'truth')
SETTINGS = ('model', 'base_model', 'checkpoint', 'checkpoint_assets', 'engine_kwargs', 'expected_packages',
    'cleared_environment', 'environment', 'request_mm_processor_kwargs', 'vision_observation',
    'minimum_kv_capacity', 'torch_num_threads', 'gpu_uuid', 'physical_lock', 'python', 'resources')
qsa = qualified.qsa
base = qsa.base


def root_ref(path, root):
    return {'path':str(Path(path).resolve().relative_to(Path(root).resolve())), 'sha256':file_sha256(path)}


def local_path(root, path):
    return results.Archive(root).path(path)


def source_audit(root, source_asset=None, recipe=None, verify_media=False):
    """Independently admit the exact affected parent set, before any answers exist."""
    root = Path(root); plans = root/'plans'/NAME; oldplans = root/'plans'/PARENT
    allocation = read(plans/'allocation.json'); old = read(oldplans/'allocation.json')
    quality = read(oldplans/'source-quality-policy.json')
    oldcases = {c['sample_id']: (c, b) for b in old['blocks'] for c in b['cases']}
    parents = {s for s, (c, _) in oldcases.items() if c['object_id'] == 'photo_obj_021'}
    require(len(oldcases) == 3840 and len(parents) == 40, 'Accepted parent scope differs')
    require(len(allocation['blocks']) == len(old['blocks']) == 480, 'Block universe differs')
    revision = read(root/'runs/stimulus-preparation'/SOURCE/'source-audit.json')
    stored = read(root/'runs/stimulus-preparation'/SOURCE/'source-records.json')['T2']
    require(revision['status'] == 'COMPLETE_SPOON_SOURCE_REVISION', 'Source revision not complete')
    accepted = read(plans/'source-acceptance.json') if (plans/'source-acceptance.json').exists() else None
    asset_ref = root_ref(source_asset, root) if source_asset else accepted['source_asset']
    recipe_ref = root_ref(recipe, root) if recipe else accepted['stimulus_recipe']
    require(file_sha256(local_path(root, asset_ref['path'])) == asset_ref['sha256'] and file_sha256(local_path(root, recipe_ref['path'])) == recipe_ref['sha256'],
            'Actual replacement asset/recipe identity changed')
    old_asset = next(a for a in read(oldplans/'asset-preparation.json')['assets'] if a['asset_id'] == 'photo_obj_021')
    require(asset_ref['sha256'] != old_asset['source_sha256'], 'Replacement still uses historical asset')
    records = {}; seen_parents = set(); proofs = []
    for b, ob in zip(allocation['blocks'], old['blocks']):
        require({k: b[k] for k in ob if k != 'cases'} == {k: v for k, v in ob.items() if k != 'cases'},
                'Parent block condition or original card changed')
        expected = [c for c in ob['cases'] if c['sample_id'] in parents]
        require(len(b['cases']) == len(expected), 'Affected block subset differs')
        if not expected:
            continue
        path = root/'datasets'/NAME/'qa'/('block-%03d.json' % b['block']); qa = read(path)
        require(qa['status'] == 'FULL180_LOSSLESS_RGB_QA_PASS' and qa['block'] == b['block'], 'Source block QA incomplete')
        require([x['sample_id'] for x in qa['cases']] == [x['sample_id'] for x in b['cases']], 'QA order differs')
        old_qa = {x['sample_id']:x for x in read(root/'datasets'/PARENT/'qa'/path.name)['cases']}
        for c, previous, x in zip(b['cases'], expected, qa['cases']):
            sid = c['sample_id']; parent = c['parent_sample_id']
            require(parent == previous['sample_id'] and parent not in seen_parents and sid not in oldcases and sid not in records,
                    'Repeated, foreign, or reused parent/new ID')
            require(all(c[k] == previous[k] for k in PARAMETERS), 'Scientific condition changed')
            require(c['source_asset_sha256'] == asset_ref['sha256'] and c['stimulus_recipe_sha256'] == recipe_ref['sha256'],
                    'Asset/recipe binding differs')
            require(all(x[k] == v for k, v in c.items()), 'QA/allocation binding differs')
            require(x['video_sha256'] != old_qa[parent]['video_sha256'] and x['stream_sha256'] != old_qa[parent]['stream_sha256'],
                    'Replacement retains historical video/RGB stream')
            require(x['decoded_frames_exact'] == 180 and x['outside_target_identical'] is True
                    and x['offset_frame'] - x['onset_frame'] == 30, 'Frame/target/event differs')
            require(x['render_implementation_sha256'] in quality['accepted_generator_sha256'], 'Original mathematical generator changed')
            require(abs(x['float64_delta_Y'] - x['delta_Y']) < 1e-10
                    and abs(x['decoded_delta_Y'] - x['delta_Y']) < min(.002, .02*x['delta_Y'])
                    and 0 <= x['white_blend_k'] <= 1, 'Original luminance QA fails')
            for key in ('video_sha256', 'stream_sha256', 'baseline_rgb8_sha256', 'event_rgb8_sha256'):
                require(len(x[key]) == 64, 'Missing source identity: '+key)
            if verify_media:
                require(file_sha256(root/'datasets'/NAME/'videos'/(sid+'.mp4')) == x['video_sha256'], 'Video bytes changed')
            records[sid] = {**x, 'card': b['card']}; seen_parents.add(parent)
        for group in {x['group'] for x in qa['cases']}:
            pair = [x for x in qa['cases'] if x['group'] == group]
            require(len(pair) == 2 and {x['truth'] for x in pair} == {'LEFT', 'RIGHT'}
                    and pair[0]['baseline_rgb8_sha256'] == pair[1]['baseline_rgb8_sha256'], 'LR pairing differs')
        proofs.append(root_ref(path, root))
    require(seen_parents == parents and len(records) == 40 and Counter(x['card'] for x in records.values()) == COUNTS,
            'Full40 parent set or original card assignment differs')
    require(records == stored, 'Independent source records differ')
    summary = {'status': 'COMPLETE_REPLACEMENT_SOURCE_QA_PASS', 'videos': 40, 'decoded_frames': 7200,
        'affected_blocks': len(proofs), 'source_records_sha256': digest(records), 'source_asset': asset_ref,
        'stimulus_recipe': recipe_ref, 'source_block_receipts': proofs, 'new_model_calls': 0,
        'source_revision': root_ref(root/'runs/stimulus-preparation'/SOURCE/'source-audit.json', root),
        'source_records': root_ref(root/'runs/stimulus-preparation'/SOURCE/'source-records.json', root),
        'allocation': root_ref(plans/'allocation.json', root), 'parent_allocation': root_ref(oldplans/'allocation.json', root),
        'quality_policy': root_ref(oldplans/'source-quality-policy.json', root), 'parent_asset_preparation': root_ref(oldplans/'asset-preparation.json', root),
        'selection_uses_model_answers': False, 'all40_replaced_regardless_of_answers': True,
        'direct_video_SHA_check': verify_media}
    return summary, records


def science(p, resolve=verified):
    qualified.fixed_implementation()
    live = resolve is verified
    if live:
        resolve = lambda ref: verified({**ref, 'path':str(local_path(p['workspace_root'], ref['path']))})
    template = resolve(p['qualified_profile'])
    for key in SETTINGS:
        require(p[key] == template[key], 'Qualified natural Qwen X setting changed: '+key)
    require(p['model'] == 'qwen_x_qsa' and p['base_model'] == 'qwen' and not p.get('bootstrap') and not p.get('request_adapter'),
            'Unexpected inference adaptation')
    require(p.get('media_io_kwargs', base.MEDIA) == base.MEDIA and p['phase'] == 'natural_T2'
            and p['resource_group'] == 'natural_T2' and p['card'] in CARDS
            and p['automatic_retry'] is False and p['formal_auto_resume'] is False, 'Wrong route/retry/scope')
    plan = resolve(p['experiment_plan']); auth = resolve(p['execution_authorization']); acceptance = resolve(p['source_acceptance'])
    require(plan['schema'] == 'natural-T2-spoon-Qwen-X-execution-v1' and plan['maximum_new_model_calls'] == 40
            and plan['all40_replaced_regardless_of_answers'] is True and plan['selection_uses_model_answers'] is False,
            'Wrong finite40 scope')
    require(auth['status'] == 'AUTHORIZED' and auth['experiment_plan_sha256'] == p['experiment_plan']['sha256']
            and auth['maximum_new_model_calls'] == 40 and auth['user_message'] == USER_MESSAGE,
            'Missing independent bound40 authorization')
    gate = resolve(p['qualification_summary'])
    require(gate['status'] == 'PASS_FIXED144_THREE_CARDS' and gate['technical_status'] == 'COMPLETE_PASS'
            and gate['executed_model_calls'] == 432 and gate['unique_video_ids'] == 144
            and gate['full_output_differences'] == [] and p['qualification_summary'] == plan['qualification_summary'],
            'Existing432 qualification differs')
    require(p['source_acceptance'] == plan['source_acceptance'] and acceptance['videos'] == 40
            and acceptance['status'] == 'COMPLETE_REPLACEMENT_SOURCE_QA_PASS', 'Source acceptance not bound')
    for key in ('source_revision', 'source_records', 'allocation', 'parent_allocation', 'quality_policy', 'parent_asset_preparation'):
        resolve(acceptance[key])
    all_records = resolve(acceptance['source_records'])['T2']
    parent_allocation = resolve(acceptance['parent_allocation'])
    expected_parents = {c['sample_id']:b['card'] for b in parent_allocation['blocks'] for c in b['cases'] if c['object_id'] == 'photo_obj_021'}
    require(len(expected_parents) == 40 and set(plan['parent_ids']) == set(expected_parents)
            and all(q['card'] == expected_parents.get(q['parent_sample_id']) for q in all_records.values()), 'Parent40/card assignment differs')
    require(resolve(plan['parent_summary'])['status'] == 'COMPLETE_NATURAL_T2_X', 'Accepted parent result changed')
    old_records = resolve(plan['parent_records'])
    require(len(old_records) == len({r['sample_id'] for r in old_records}) == 3840, 'Parent3840 result universe differs')
    asset = acceptance['source_asset']; recipe = acceptance['stimulus_recipe']
    if live:
        require(file_sha256(local_path(p['workspace_root'],asset['path'])) == asset['sha256'] and file_sha256(local_path(p['workspace_root'],recipe['path'])) == recipe['sha256'],
                'Actual source asset/recipe changed')
    all_ids = [s for c in CARDS for s in plan['cards'][c]['assigned_ids']]
    require(len(all_ids) == len(set(all_ids)) == 40 and set(all_ids) == set(all_records)
            and {c:len(plan['cards'][c]['assigned_ids']) for c in CARDS} == COUNTS, 'Global40 universe changed')
    full = plan['cards'][p['card']]['assigned_ids']; ids = p['assigned_ids']
    require(len(ids) == len(set(ids)) == p['maximum_model_calls'] and set(ids) <= set(full)
            and p['source_catalog'] == plan['cards'][p['card']]['source_catalog'], 'Repeated/foreign ID or catalog')
    if p.get('continuation_parent'):
        parent = resolve(p['continuation_parent'])
        require(p['control_generation'] == parent['control_generation']+1, 'Resume generation differs')
        if live:
            parent, terminal, raw = closed(p['continuation_parent']['path'], allow_successor=True)
            require(ids == terminal['unattempted_ids'] == [s for s in parent['assigned_ids'] if s not in raw]
                    and verified(p['continuation_wait'])['profile_sha256'] == digest(parent), 'Resume is not never-claimed suffix')
    else:
        require(ids == full and p['control_generation'] == 0, 'Initial finite allocation changed')
    root = Path(p['workspace_root'])
    require(Path(p['output']).parent == root/'runs/reproducibility'/NAME
            and Path(p['claim_root']) == root/'runs/reproducibility'/NAME/'claims'/p['card']
            and Path(p['control_file']) == root/'plans'/NAME/(p['card']+'-control.json'), 'Run namespace differs')
    parent_catalog = resolve(p['parent_catalog'])['items']; catalog = resolve(p['source_catalog'])['items']
    require(p['parent_catalog'] == plan['cards'][p['card']]['parent_catalog'], 'Parent source catalog differs')
    parents = set()
    for sid in all_ids:
        q = all_records[sid]; parent = q['parent_sample_id']
        require(parent not in parents and sid != parent and q['object_id'] == 'photo_obj_021', 'Parent mapping differs')
        parents.add(parent)
    require(parents == set(plan['parent_ids']), '40 parent universe changed')
    for sid in ids:
        item = catalog[sid]; q = item['natural_qa']; old = parent_catalog[q['parent_sample_id']]; row = item['row']
        require(q == all_records[sid] and q['sample_id'] == sid and q['card'] == p['card'] and q['source_asset_sha256'] == asset['sha256']
                and q['stimulus_recipe_sha256'] == recipe['sha256'], 'Actual source/parent assignment differs')
        require(all(q[k] == old['natural_qa'][k] for k in PARAMETERS), 'Parent scientific parameters changed')
        require(row['truth'] == q['truth'] and row['prompt'] == old['row']['prompt'] == plan['prompt']
                and row['choices'] == ['LEFT', 'RIGHT'] and row['accuracy_eligible'] is True
                and row['task'] == item['task'] == 'T2', 'Question/GT/task differs')
        require(item['content'][0]['type'] == 'video' and item['content'][0]['path'] == str(root/'datasets'/NAME/'videos'/(sid+'.mp4')), 'Video namespace differs')
        require(digest(row) == item['source_row_sha256'] and digest(q) == item['source_audit_identity']
                and q['video_sha256'] == item['content'][0]['sha256']
                and item['content'][1] == {'type':'text', 'text':plan['prompt']}, 'Runtime source binding differs')
        require(q['decoded_frames_exact'] == 180 and q['outside_target_identical'] is True
                and q['render_implementation_sha256'] in plan['accepted_stimulus_implementation_sha256']
                and abs(q['float64_delta_Y']-q['delta_Y']) < 1e-10
                and abs(q['decoded_delta_Y']-q['delta_Y']) < min(.002,.02*q['delta_Y'])
                and q['offset_frame']-q['onset_frame'] == 30
                and item['video_contract'] == {'fps':60,'frames':180,'height':512,'width':768}, 'Source QA differs')


class Boundary(lifecycle.Boundary):
    pass


def predecessor_closed(p):
    """Check actual predecessor terminal receipts and original-inode lock freedom."""
    root = Path(p['workspace_root']); folder = root/'plans'/CLASSIC
    plan = read(folder/'plan.json'); paths = sorted(folder.glob(p['card']+'-segment*-profile.json'), key=lambda x:read(x)['control_generation'])
    require(paths, 'Classic predecessor profiles absent')
    done = []
    for n, path in enumerate(paths):
        old, terminal, raw = closed(path, allow_successor=n < len(paths)-1)
        require(old['card'] == p['card'] and old['physical_lock'] == p['physical_lock'] and old['control_generation'] == n,
                'Classic predecessor identity differs')
        require(terminal['status'] == ('COMPLETE' if n == len(paths)-1 else 'PAUSED'), 'Classic predecessor not fully closed')
        require(terminal['completed_ids'] == old['assigned_ids'][:len(raw)], 'Classic predecessor is not ordered prefix')
        done += list(raw)
    require(done == plan['cards'][p['card']]['assigned_ids']
            and {x.stem for x in Path(old['claim_root']).glob('*.json')} == set(done), 'Classic predecessor finite allocation/claim union incomplete')
    lock = p['physical_lock']; path = Path(lock['path'])
    require(path.is_file() and not path.is_symlink(), 'Original physical lease absent')
    with path.open('r+b') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stat = path.stat()
        require((stat.st_dev, stat.st_ino) == (lock['st_dev'], lock['st_ino']), 'Original lease inode changed')
    return {'status':'CLASSIC_PREDECESSOR_CLOSED_PASS', 'profiles':[descriptor(x) for x in paths], 'completed':len(done), 'same_inode_lock_free':True}


def seal(draft, output):
    p = read(draft); science(p); p['operational_wrapper'] = descriptor(__file__)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(p, f); temp = f.name
    try:
        return qsa.seal(temp, output)
    finally:
        Path(temp).unlink()


def run(profile):
    p = read(profile); science(p); predecessor_closed(p)
    for key, path in [('operational_wrapper',__file__), ('qsa_entry',qsa.__file__), ('observer_entry',qsa.v2.__file__), ('selector_source',qsa.selector.__file__)]:
        require(file_sha256(path) == p[key]['sha256'], 'Frozen execution source changed: '+key)
    base.worker_identity = qualified.worker_identity; base.Boundary = Boundary
    return qsa.v2.run(profile)


def supervise(profile):
    from . import refinement_worker as worker
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        return native([x.replace('classic_precision.refinement_worker', __spec__.name) for x in argv], **kwargs)
    worker.subprocess.Popen = popen
    try:
        return worker.supervise(profile)
    finally:
        worker.subprocess.Popen = native


def closed(profile, allow_successor=False):
    p = read(profile); out = Path(p['output']); terminal = read(out/'result.json'); wait = read(out.with_suffix('.wait.json'))
    require(wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released'] and wait['profile_sha256'] == digest(p)
            and wait['terminal_status'] == terminal['status']
            and terminal['status'] in ('COMPLETE','PAUSED','PAUSED_BEFORE_LOAD')
            and not terminal.get('UNCERTAIN_ids') and not (out/'failure.json').exists(), 'No safe finite worker closure')
    raw = results.unique(results.journal(out/'raw.jsonl')) if (out/'raw.jsonl').exists() else {}
    events = results.journal(out/'events.jsonl') if (out/'events.jsonl').exists() else []
    passed = results.unique([x for x in events if x['event'] == 'TECHNICAL_PASS'])
    intents = results.unique([x for x in events if x['event'] == 'INTENT'])
    require(list(raw) == terminal['completed_ids'] == wait['completed_ids'] == p['assigned_ids'][:len(raw)]
            and set(raw) == set(passed) == set(intents), 'Raw/PASS/INTENT/terminal prefix differs')
    for sid, value in raw.items():
        claim = read(Path(p['claim_root'])/(sid+'.json'))
        require(value['profile_sha256'] == digest(p) and passed[sid]['raw_digest'] == digest(value)
                and claim['sample_id'] == sid and claim['profile_sha256'] == digest(p)
                and claim['worker_token'] == p['worker_token'], 'Unqualified raw or claim')
    claimed = {s for s in p['assigned_ids'] if (Path(p['claim_root'])/(s+'.json')).exists()}
    if not allow_successor:
        require(claimed == set(raw), 'Unresolved claim; refuse retry')
    require(terminal['unattempted_ids'] == p['assigned_ids'][len(raw):], 'Unattempted suffix differs')
    if terminal['status'] == 'COMPLETE':
        require(list(raw) == p['assigned_ids'], 'Incomplete COMPLETE')
    return p, terminal, raw


def activate(profile):
    p = read(profile); science(p); proof = predecessor_closed(p); c = read(p['control_file'])
    require(c['generation'] == p['control_generation'] and not Path(p['output']).exists()
            and not Path(p['output']).with_suffix('.launch.json').exists(), 'Attempted/stale worker')
    require(not any((Path(p['claim_root'])/(s+'.json')).exists() for s in p['assigned_ids']), 'Already claimed')
    c.update(paused=False, activated_profile=descriptor(profile), predecessor_closure=proof)
    atomic_json(p['control_file'], c)
    return {'status':'ACTIVATED_NOT_LAUNCHED', 'profile':descriptor(profile), 'predecessor':proof}


def resume(profile, output):
    p, terminal, raw = closed(profile); science(p); c = read(p['control_file'])
    require(c['paused'], 'Explicit pause required'); ids = lifecycle.remaining(p, terminal, raw)
    require(ids, 'No never-claimed inputs remain')
    q = copy.deepcopy(p); generation = c['generation']+1; worker = p['card']+'-segment'+str(generation)
    for key in ('implementation','worker_token','media_files','operational_wrapper'):
        q.pop(key, None)
    q.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
        output=str(Path(p['output']).parent/worker), pause_file=str(Path(output).with_suffix('.PAUSE.json')),
        continuation_parent=descriptor(profile), continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')))
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(q, f); temp = f.name
    try:
        answer = seal(temp, output)
    finally:
        Path(temp).unlink()
    c.update(generation=generation, paused=True, continuation_profile=descriptor(output)); atomic_json(p['control_file'], c)
    return {**answer, 'status':'SEALED_PAUSED_NOT_LAUNCHED'}


def prepare(root, source_asset, recipe):
    root = Path(root).resolve(); plans = root/'plans'/NAME; arc = results.Archive(root)
    require(not (plans/'plan.json').exists(), 'Finite package already prepared')
    acceptance, qa = source_audit(root, source_asset, recipe); write_new(plans/'source-acceptance.json', acceptance)
    cards = {}; prompts = set()
    for host, remote in ROOTS.items():
        oldpath = root/'plans'/PARENT/(host+'-sources.json'); old = read(oldpath)['items']; items = {}
        for sid, q in qa.items():
            if (q['card'] == 'radon0') != (host == 'radon'):
                continue
            parent = old[q['parent_sample_id']]; prompt = parent['row']['prompt']; prompts.add(prompt)
            row = {'sample_id':sid, 'parent_sample_id':q['parent_sample_id'], 'task':'T2', 'truth':q['truth'],
                'choices':['LEFT','RIGHT'], 'accuracy_eligible':True, 'prompt':prompt,
                'source_parameters':{k:q[k] for k in PARAMETERS+('card',)}, 'source_origin':'natural-T2-spoon-source-revision'}
            ready = {k:q[k] for k in ('video_sha256','stream_sha256','baseline_rgb8_sha256','event_rgb8_sha256')}
            items[sid] = {'sample_id':sid, 'task':'T2', 'row':row, 'natural_qa':q, 'ready':ready,
                'content':[{'type':'video', 'path':str(Path(remote)/'datasets'/NAME/'videos'/(sid+'.mp4')), 'sha256':q['video_sha256']},
                    {'type':'text','text':prompt}], 'source_row_sha256':digest(row), 'source_ready_sha256':digest(ready),
                'source_stream_sha256':q['stream_sha256'], 'source_audit_identity':digest(q),
                'video_contract':{'fps':60,'frames':180,'height':512,'width':768}}
        path = plans/(host+'-sources.json'); write_new(path, {'schema':'natural-T2-spoon-sources-v1', 'items':items})
        for card in CARDS:
            if (card == 'radon0') != (host == 'radon'):
                continue
            ids = [s for s, q in qa.items() if q['card'] == card]
            cards[card] = {'assigned_ids':ids, 'count':len(ids), 'source_catalog':qualified.remote_ref(path,root,remote),
                'parent_catalog':qualified.remote_ref(oldpath,root,remote)}
            (plans/(card+'-video-files.txt')).write_text(''.join(s+'.mp4\n' for s in ids))
    require(len(prompts) == 1, 'Parent prompts differ')
    qualification = root/'runs/reproducibility/qwen-x-crosscard-t23-20261029/analysis/summary.json'
    # refs use each host's prefix but byte-identical shared package files.
    plan = {'schema':'natural-T2-spoon-Qwen-X-execution-v1', 'status':'PREPARED_NOT_LAUNCHED',
        'maximum_new_model_calls':40, 'cards':cards, 'prompt':prompts.pop(),
        'source_acceptance':root_ref(plans/'source-acceptance.json',root), 'qualification_summary':root_ref(qualification,root),
        'parent_ids':sorted(q['parent_sample_id'] for q in qa.values()),
        'parent_summary':root_ref(root/'runs/reproducibility'/PARENT/'summary.json',root),
        'parent_records':root_ref(root/'runs/reproducibility'/PARENT/'canonical-records.json',root),
        'accepted_stimulus_implementation_sha256':read(root/'plans'/PARENT/'source-quality-policy.json')['accepted_generator_sha256'],
        'automatic_retry':False, 'no_automatic_successor':True, 'selection_uses_model_answers':False,
        'all40_replaced_regardless_of_answers':True, 'overlay_denominator':3840, 'historical_records_preserved':True}
    write_new(plans/'plan.json', plan)
    write_new(plans/'authorization.json', {'status':'AUTHORIZED', 'maximum_new_model_calls':40,
        'experiment_plan_sha256':file_sha256(plans/'plan.json'), 'user_message':USER_MESSAGE,
        'scope':'Replace all40 completed natural T2 wood-spoon sources; new IDs; same original card/condition; no T3/classic/qualification/retry calls.'})
    for card in CARDS:
        host = 'radon' if card == 'radon0' else 'fourier'; remote = ROOTS[host]
        original = root/'plans'/PARENT/(card+'-segment0-profile.json'); p = read(original)
        for key in ('implementation','worker_token','media_files','operational_wrapper','continuation_parent','continuation_wait'):
            p.pop(key, None)
        worker = card+'-segment0'; control = plans/(card+'-control.json')
        write_new(control, {'generation':0, 'paused':True, 'status':'PREPARED_NOT_LAUNCHED'})
        # Keep shared refs local-to-package relative so identical plan bytes work on both hosts.
        p.update(worker=worker, phase='natural_T2', resource_group='natural_T2', workspace_root=remote,
            assigned_ids=cards[card]['assigned_ids'], maximum_model_calls=COUNTS[card],
            source_catalog=cards[card]['source_catalog'], parent_catalog=cards[card]['parent_catalog'],
            qualified_profile=qualified.remote_ref(original,root,remote),
            source_acceptance=root_ref(plans/'source-acceptance.json',root), qualification_summary=root_ref(qualification,root),
            experiment_plan=qualified.remote_ref(plans/'plan.json',root,remote), execution_authorization=qualified.remote_ref(plans/'authorization.json',root,remote),
            output=str(Path(remote)/'runs/reproducibility'/NAME/worker), claim_root=str(Path(remote)/'runs/reproducibility'/NAME/'claims'/card),
            pause_file=str(Path(remote)/'plans'/NAME/(worker+'-PAUSE.json')), control_file=str(Path(remote)/'plans'/NAME/control.name),
            control_generation=0, automatic_retry=False, formal_auto_resume=False)
        science(p, arc.ref); write_new(plans/(worker+'-draft.json'), p)
    return {'status':'PREPARED_NOT_LAUNCHED', 'maximum_new_model_calls':40, 'cards':COUNTS, 'new_qualification_calls':0}


def queue(profile, expected_sha256, timeout_seconds=7200):
    """One explicitly requested queue; exact predecessor wait-only polling, no retries."""
    require(expected_sha256 and file_sha256(profile) == expected_sha256, 'Queue profile SHA differs')
    require(0 < timeout_seconds <= 7200, 'Queue deadline must be at most two hours')
    p = read(profile); science(p)
    require(p['operational_wrapper']['sha256'] == file_sha256(__file__), 'Queue wrapper changed')
    out = Path(p['output']); launch = out.with_suffix('.queue-launch.json'); final = out.with_suffix('.queue-wait.json')
    require(not out.exists() and not out.with_suffix('.launch.json').exists() and not final.exists(), 'Attempted worker/queue')
    control_path = Path(p['control_file']); control = read(control_path)
    require(control['paused'] is True and control['generation'] == p['control_generation'], 'Queue requires current prepared-paused profile')
    predecessor_dir = Path(p['workspace_root'])/'plans'/CLASSIC
    predecessors = sorted(predecessor_dir.glob(p['card']+'-segment*-profile.json'), key=lambda path:read(path)['control_generation'])
    require(predecessors, 'Exact classic predecessor profiles absent')
    bindings = [descriptor(path) for path in predecessors]
    waits = [Path(read(path)['output']).with_suffix('.wait.json') for path in predecessors]
    state = {'status':'QUEUED_WAITING_PREDECESSOR', 'profile':descriptor(profile), 'owner':owner_identity(),
        'maximum_new_model_calls':p['maximum_model_calls'], 'experiment_maximum_new_model_calls':40,
        'predecessor_profiles':bindings, 'predecessor_wait_paths':[str(path) for path in waits],
        'poll_interval_seconds':30, 'timeout_seconds':timeout_seconds, 'automatic_retry':False,
        'real_supervisor_wait':False, 'supervisor_pid':None, 'supervisor_returncode':None}
    events = control_path.with_suffix('.events.jsonl')
    def event_identity():
        if not events.exists(): return None
        value = events.stat(); return [value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns]
    def unchanged():
        require(file_sha256(profile) == expected_sha256, 'Queued sealed profile changed')
        for ref in bindings:
            require(file_sha256(ref['path']) == ref['sha256'], 'Exact queued predecessor profile changed')
        require(sorted(str(path.resolve()) for path in predecessor_dir.glob(p['card']+'-segment*-profile.json'))
                == sorted(ref['path'] for ref in bindings), 'Classic predecessor versions changed while queued')
        if file_sha256(control_path) != control_sha or event_identity() != event_stamp:
            return False
        pause = Path(p['pause_file'])
        if pause.exists():
            require(read(pause) == {'worker_token':p['worker_token'],'action':'PAUSE'}, 'Foreign queued pause file')
            return False
        return True
    with lifecycle.controller_lock(control_path.with_suffix('.queue.lock')):
        write_new(launch, state)
        control.update(status='QUEUED_WAITING_PREDECESSOR', queued_profile=descriptor(profile), queued_owner=state['owner'])
        atomic_json(control_path,control); control_sha=file_sha256(control_path); event_stamp=event_identity()
        started=time.monotonic(); child=None; error=None
        try:
            while True:
                if not unchanged():
                    state['status']='PAUSED_NOT_LAUNCHED';break
                if time.monotonic()-started >= timeout_seconds:
                    state['status']='TIMED_OUT_NOT_LAUNCHED';break
                if all(path.is_file() for path in waits):
                    state['predecessor_closure']=predecessor_closed(p)
                    if time.monotonic()-started >= timeout_seconds:
                        state['status']='TIMED_OUT_NOT_LAUNCHED';break
                    if not unchanged():
                        state['status']='PAUSED_NOT_LAUNCHED';break
                    state['activation']=activate(profile)
                    current=read(control_path)
                    if current['paused'] or current['generation'] != p['control_generation'] or event_identity() != event_stamp or Path(p['pause_file']).exists():
                        lifecycle.pause(p['control_file']);state['status']='PAUSED_NOT_LAUNCHED';break
                    if time.monotonic()-started >= timeout_seconds:
                        lifecycle.pause(p['control_file']);state['status']='TIMED_OUT_NOT_LAUNCHED';break
                    require(file_sha256(profile) == expected_sha256, 'Activated profile SHA changed')
                    state['status']='SUPERVISOR_LAUNCHING';atomic_json(launch,state)
                    env=dict(os.environ)
                    for key in p.get('cleared_environment',[]): env.pop(key,None)
                    env.update(p['environment'])
                    child=subprocess.Popen([p['python'],'-B','-m',__spec__.name,'supervise','--profile',str(Path(profile).resolve())],
                        env=env,stdin=subprocess.DEVNULL)
                    state.update(status='SUPERVISOR_RUNNING',supervisor_pid=child.pid);atomic_json(launch,state)
                    code=child.wait();state.update(real_supervisor_wait=True,supervisor_returncode=code)
                    require(file_sha256(profile) == expected_sha256, 'Executed queue profile SHA changed')
                    worker_wait=out.with_suffix('.wait.json')
                    if worker_wait.exists():
                        w=read(worker_wait);state.update(worker_wait=descriptor(worker_wait),
                            worker_real_wait=w.get('real_wait_observed',False),owned_engines_exited=w.get('owned_engines_exited',False),
                            original_physical_lease_released=w.get('original_physical_lease_released',False))
                    require(code == 0, 'Supervisor failed; no automatic retry')
                    _,terminal,raw=closed(profile)
                    state.update(status='COMPLETE' if terminal['status']=='COMPLETE' else 'PAUSED',
                        worker_terminal_status=terminal['status'],completed_ids=list(raw))
                    break
                remaining=timeout_seconds-(time.monotonic()-started)
                if remaining <= 0:
                    state['status']='TIMED_OUT_NOT_LAUNCHED';break
                time.sleep(min(30,remaining))
        except BaseException as exc:
            state.update(status='HELD_NO_RETRY',error_type=type(exc).__name__,error=str(exc));error=exc
            if child is not None and not state['real_supervisor_wait']:
                lifecycle.pause(p['control_file'])
                state.update(supervisor_returncode=child.wait(),real_supervisor_wait=True)
        finally:
            worker_wait=out.with_suffix('.wait.json')
            if worker_wait.exists():
                w=read(worker_wait);state.update(worker_wait=descriptor(worker_wait),
                    worker_real_wait=w.get('real_wait_observed',False),owned_engines_exited=w.get('owned_engines_exited',False),
                    original_physical_lease_released=w.get('original_physical_lease_released',False))
            state['elapsed_seconds']=time.monotonic()-started
            state['observed_claims']=[sid for sid in p['assigned_ids'] if (Path(p['claim_root'])/(sid+'.json')).exists()]
            if child is None:
                state.update(worker_real_wait=False,owned_engines_exited=False,original_physical_lease_released=False)
            atomic_json(final,state)
        if error is not None: raise error
    return {'status':state['status'],'queue_launch':descriptor(launch),'queue_wait':descriptor(final),
        'real_supervisor_wait':state['real_supervisor_wait'],'supervisor_returncode':state['supervisor_returncode'],
        'completed':len(state.get('completed_ids',[]))}


def status(root):
    from .natural_t2_results import status as shared_status
    return shared_status(root, NAME)


def paused_segment(arc, spec, successor_ref):
    """Read-only paused-prefix admission; shared native raw/input checks remain exact."""
    p = arc.ref(spec['profile']); successor = arc.ref(successor_ref); out = arc.path(p['output'])
    terminal = read(out/'result.json'); wait_path = out.with_suffix('.wait.json'); wait = read(wait_path)
    selected = terminal['completed_ids']
    require(p['phase'] == successor['phase'] == 'natural_T2' and p['model'] == 'qwen_x_qsa', 'Unexpected paused protocol')
    require(wait['profile_sha256'] == digest(p) and wait['returncode'] == 0
            and wait['real_wait_observed'] and wait['owned_engines_exited'] and wait['original_physical_lease_released']
            and wait['terminal_status'] == terminal['status'] == 'PAUSED' and not terminal['UNCERTAIN_ids'], 'Paused worker not safely closed')
    require(selected == p['assigned_ids'][:len(selected)] and terminal['unattempted_ids'] == p['assigned_ids'][len(selected):]
            and wait['completed_ids'] == selected, 'Paused prefix ledger differs')
    require(arc.ref(successor['continuation_parent']) == p and arc.ref(successor['continuation_wait']) == wait
            and successor['assigned_ids'] == terminal['unattempted_ids']
            and successor['control_generation'] == p['control_generation']+1
            and successor['worker_token'] != p['worker_token'] and successor['output'] != p['output'], 'Paused successor binding differs')
    operational = {'assigned_ids','media_files','maximum_model_calls','pause_file','worker_token','continuation_parent',
        'continuation_wait','control_generation','output','worker'}
    require(all(p.get(k) == successor.get(k) for k in p.keys() | successor.keys() if k not in operational), 'Paused successor changed science')
    require(not (out/'failure.json').exists(), 'Retained paused failure cannot be retried')
    raw = results.unique(results.journal(out/'raw.jsonl')); events = results.journal(out/'events.jsonl')
    intents = results.unique([e for e in events if e['event'] == 'INTENT'])
    passes = results.unique([e for e in events if e['event'] == 'TECHNICAL_PASS'])
    timings = results.unique(results.journal(out/'request-timings.jsonl'))
    require(set(raw) == set(intents) == set(passes) == set(timings) == set(selected), 'Paused per-ID ledger differs')
    identity = read(out/'identity.json'); require(identity['profile'] == p and identity['profile_sha256'] == digest(p), 'Identity differs')
    versions = read(out/'versions.json'); runtime = read(out/'runtime.json'); public = read(out/'public-settings.json')
    require(all(k in versions for k in ('vllm','torch','transformers','numpy'))
            and all(versions[k] == v for k,v in p['expected_packages'].items() if k in versions), 'Runtime packages differ')
    require(runtime, 'Runtime receipt absent')
    for value in runtime:
        qualified.validate_runtime(value)
        require(value['vllm_version'] == versions['vllm'] and value['torch_version'].split('+')[0] == versions['torch'].split('+')[0]
                and value['actual_kv_capacity'] >= p['minimum_kv_capacity'], 'Runtime capacity/version differs')
    for key,value in p['engine_kwargs'].items():
        require(public['LLM_kwargs'][key] == (sorted(value) if key == 'cpu_offload_params' else value), 'Public engine setting differs: '+key)
    deep = results.unique(results.journal(out/'deep-observations.jsonl'))
    admissions = results.unique(results.journal(out/'input-admissions.jsonl'))
    selectors = results.unique(results.journal(out/'custom-selector-admissions.jsonl'))
    catalog = arc.ref(p['source_catalog'])['items']; rows = []
    for sid in selected:
        r = raw[sid]; item = catalog[sid]; eligible = results.common_raw_checks(p,item,r,p['assigned_ids'].index(sid))
        claim = read(arc.path(p['claim_root'])/(sid+'.json'))
        require(claim['sample_id'] == sid and claim['profile_sha256'] == digest(p) and claim['worker_token'] == p['worker_token']
                and passes[sid]['raw_digest'] == digest(r), 'Claim/raw/PASS differs')
        timing = timings[sid]
        require(math.isfinite(timing['end_to_end_durable_seconds']) and timing['end_to_end_durable_seconds'] > 0
                and timing['engine_request_inclusive_seconds'] <= timing['end_to_end_durable_seconds'], 'Durable timing differs')
        if r['deep']:
            observations = deep[sid]['observations']; a = admissions[sid]; q = selectors[sid]
            require(deep[sid]['raw_digest'] == a['raw_digest'] == q['raw_digest'] == digest(r)
                    and a['observations_digest'] == digest(observations) and observations
                    and all('runtime_error' not in x and 'profiler_error' not in x for x in observations), 'Deep proof differs')
            require(a['proof']['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS' and a['proof']['CPU_to_GPU_patch_bytes_equal']
                    and a['proof']['source_frame_indices'] == list(range(180)), 'GPU input admission differs')
            require(q['variant'] == 'X_QSA' and q['pure_official_config'] is False
                    and q['actual_request_calls'] and all(n > 0 for n in q['actual_request_calls']), 'Actual QSA absent')
            for receipt in q['selector_receipts']:
                qualified.validate_runtime({**runtime[0], 'custom_qsa_selector':receipt})
        rows.append({'model':'qwen','variant':spec['variant'],'phase':spec['phase'],'replicate':None,
            'sample_id':sid,'task':r['task'],'truth':r['truth'],'answer':r['raw_answer'].strip(),'raw_answer':r['raw_answer'],
            'correct':r['correct'],'eligible':eligible,'parameters':item['row']['source_parameters'],
            'analysis_group':item.get('analysis_group','main'),'include_in_main_accuracy_curve':item.get('include_in_main_accuracy_curve',True),
            'worker':p['worker'],'profile_sha256':digest(p),'raw_digest':digest(r),'source_media':results.receipt_media(r),
            'output_token_ids':r['output_token_ids'],'finish_reason':r['finish_reason'],'stop_reason':r['stop_reason'],
            'prompt_token_sha256':digest(r['actual_prompt_token_ids']),'prompt_tokens':len(r['actual_prompt_token_ids']),
            'deep':r['deep'],'timing':timing,'raw_timing':r['timing'],'adapter_timing':r.get('request_adapter_timing'),
            'admission':'PAUSED_PREFIX_TECHNICAL_PASS'})
    return rows, {'worker':p['worker'],'accepted':len(rows),'profile':spec['profile'],'raw':descriptor(out/'raw.jsonl'),
        'wait':descriptor(wait_path),'original_returncode':0,'admission':'PAUSED_PREFIX_TECHNICAL_PASS','successor':successor_ref,
        'recorded_package_versions':versions,'runtime':runtime,'load':read(out/'load.json'),'public_settings':public}


def full(root):
    root = Path(root); plans = root/'plans'/NAME; out = root/'runs/reproducibility'/NAME
    arc = results.Archive(root); plan = read(plans/'plan.json'); _, source = source_audit(root)
    records = []; proofs = []
    for card in CARDS:
        paths = sorted(plans.glob(card+'-segment*-profile.json'), key=lambda x:read(x)['control_generation'])
        require(paths, 'Missing sealed card: '+card); card_rows = []
        for n, path in enumerate(paths):
            p = read(path); science(p, arc.ref); require(p['control_generation'] == n, 'Segment gap')
            terminal = read(arc.path(p['output'])/'result.json')
            spec = {'profile':descriptor(path), 'phase':'natural_T2', 'variant':'X_QSA_NO_FLASHINFER_AUTOTUNE'}
            if n < len(paths)-1:
                rows, evidence = paused_segment(arc, spec, descriptor(paths[n+1]))
            else:
                require(terminal['status'] == 'COMPLETE', 'Final segment incomplete')
                rows, evidence = results.audit_worker(arc, spec)
            card_rows += rows; proofs.append(evidence)
        require([r['sample_id'] for r in card_rows] == plan['cards'][card]['assigned_ids'], 'Card order/union differs')
        require({x.stem for x in arc.path(p['claim_root']).glob('*.json')} == {r['sample_id'] for r in card_rows}, 'Final claim universe differs')
        records += card_rows
    require(len(records) == len({r['sample_id'] for r in records}) == 40 and {r['sample_id'] for r in records} == set(source), 'Finite40 union differs')
    parent_folder = root/'runs/reproducibility'/PARENT
    old_summary = arc.ref(plan['parent_summary']); old = arc.ref(plan['parent_records'])
    require(old_summary['status'] == 'COMPLETE_NATURAL_T2_X' and len(old) == len({r['sample_id'] for r in old}) == 3840,
            'Accepted parent3840 absent')
    parent_ids = set(plan['parent_ids']); old_by = {r['sample_id']:r for r in old}; replacements = {}
    for r in records:
        q = source[r['sample_id']]; r['natural_qa'] = q; parent = q['parent_sample_id']; previous = old_by[parent]
        require(r['truth'] == previous['truth'] and all(r['parameters'][k] == previous['parameters'][k] for k in PARAMETERS), 'Overlay condition differs')
        replacements[parent] = r
    require(set(replacements) == parent_ids, 'All40 substitution required regardless of score')
    overlay = [replacements.get(r['sample_id'], r) for r in old]
    require(len(overlay) == len({r['sample_id'] for r in overlay}) == 3840, 'Overlay denominator differs')
    groups = defaultdict(list)
    for r in overlay:
        q = r['natural_qa']; groups[q['group'],q['delta_Y']].append(r)
    require(len(groups) == 80 and {len(v) for v in groups.values()} == {48}, 'Overlay cells unbalanced')
    cells = [{'group':g,'delta_Y':d,**results.basic(v)} for (g,d),v in sorted(groups.items())]
    summary = {'status':'COMPLETE_NATURAL_T2_SPOON_X_OVERLAY', 'model':'Qwen', 'new_scientific_calls':40,
        'overall':results.basic(overlay), 'replacement_only':results.basic(records), 'parent_overall':old_summary['overall'],
        'replaced_parent_ids':sorted(parent_ids), 'preserved_historical_record_count':3840, 'overlay_denominator':3840,
        'all40_replaced_regardless_of_answers':True, 'selection_uses_model_answers':False, 'cells':cells,
        'worker_evidence':proofs, 'plan':descriptor(plans/'plan.json'), 'source_acceptance':descriptor(plans/'source-acceptance.json'),
        'parent_summary':descriptor(parent_folder/'summary.json'), 'parent_records':descriptor(parent_folder/'canonical-records.json')}
    atomic_json(out/'replacement-records.json', records); atomic_json(out/'canonical-records.json', overlay)
    atomic_json(out/'summary.json', summary); results.write_csv(out/'cell-results.csv', cells)
    results.write_csv(out/'replacement-answers.csv', [{k:r[k] for k in ('sample_id','truth','answer','correct','worker')}|
        {'parent_sample_id':r['natural_qa']['parent_sample_id']}|r['parameters'] for r in records])
    results.write_csv(out/'replacement-timing-summary.csv', results.timing_tables(records))
    return {'status':summary['status'], 'new_calls':40, 'n':3840, 'correct':summary['overall']['correct'], 'summary':descriptor(out/'summary.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare','science','seal','activate','run','supervise','pause','prepare-resume','audit-closed','queue','status','audit','results'])
    for key in ('root','draft','output','profile','control','source-asset','recipe','profile-sha256'):
        parser.add_argument('--'+key)
    parser.add_argument('--timeout-seconds',type=float,default=7200)
    a = parser.parse_args()
    if a.command == 'prepare': value = prepare(a.root, a.source_asset, a.recipe)
    elif a.command == 'science': science(read(a.profile)); value = {'status':'SCIENCE_PASS','model_calls':0}
    elif a.command == 'seal': value = seal(a.draft,a.output)
    elif a.command == 'activate': value = activate(a.profile)
    elif a.command == 'run': value = run(a.profile)
    elif a.command == 'supervise': value = supervise(a.profile)
    elif a.command == 'pause': value = lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': value = resume(a.profile,a.output)
    elif a.command == 'queue': value = queue(a.profile,a.profile_sha256,a.timeout_seconds)
    elif a.command == 'status': value = status(a.root)
    elif a.command in ('audit','results'): value = full(a.root)
    else:
        p, terminal, raw = closed(a.profile); science(p)
        value = {'status':'CLOSED_PASS','worker_status':terminal['status'],'completed':len(raw)}
    print(json.dumps(value), flush=True)


if __name__ == '__main__':
    main()
