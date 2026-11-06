"""Prepare, admit, dispatch and return the finite T1 m=.5 and strict Classic T4 X runs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import copy
import csv
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve()
NAME = 't1-m05-t4-classic'
CODE = ROOT / 'code/vlm-psychophysics'
SOURCE = ROOT / 'runs/stimulus-preparation' / NAME
RUN = ROOT / 'runs/reproducibility' / NAME
MODELS = {'cosmos': 'cosmos', 'nemo': 'nemo', 'qwen': 'qwen_x_qsa', 'gemma': 'gemma_x_full'}
CONTRACT = {'frames': 180, 'fps': 60, 'width': 768, 'height': 512}
COUNTS = {'t1_classic': 384, 't1_natural': 384, 't4_classic': 768}
T1_CARD = {'gemma': 'fourier0', 'nemo': 'fourier0', 'cosmos': 'fourier1', 'qwen': 'fourier1'}
CARDS = ('fourier0', 'fourier1')
CLASSIC = {
    'cosmos': 'cosmos-x-full-20261028/{card}-profile.json',
    'nemo': 'nemo-x-full-20261029/{card}-full-segment0-profile.json',
    'qwen': 'qwen-x-remaining-t13-20261030/handoff1-reboot1/{card}-full-T1-handoff1-reboot1-segment0-profile.json',
    'gemma': 'gemma-x-fourier-takeover-20261031/{card}-full-segment0-profile.json',
}
NATURAL = {
    'cosmos': 'cosmos-{card}-segment0-profile.json',
    'nemo': 'nemo-t12-20261102/nemo-{card}-T12-20261102-segment0-profile.json',
    'qwen': 'qwen-t12-20261102/qwen-{card}-T1-20261102-segment0-profile.json',
    'gemma': 'gemma-natural-three-card-20261103/gemma-{card}-T1-natural-segment0-profile.json',
}
OLD_T4 = 't4-region-balanced-20261104'
COPY_FIELDS = ('checkpoint', 'checkpoint_assets', 'engine_kwargs', 'environment', 'cleared_environment',
               'expected_packages', 'python', 'hostname', 'gpu_uuid', 'physical_lock', 'workspace_root',
               'request_mm_processor_kwargs', 'request_mm_processor_kwargs_by_task', 'media_io_kwargs',
               'vision_observation', 'minimum_kv_capacity', 'torch_num_threads', 'resources',
               'processor_checkpoint', 'input_dependencies', 'adapter_image_max_soft_tokens',
               'request_adapter', 'vision_validation_hook', 'bootstrap')
SEAL_FIELDS = {'schema', 'implementation', 'media_files', 'worker_token', 'observer_entry',
               'qsa_entry', 'selector_source', 'native_CPU', 'reference_prompt_catalog',
               'draft_digest'}

# The execution host supplies this existing, immutable code tree. A worktree may
# be supplied by PYTHONPATH for offline structural checks on the Mac.
sys.path.append(str(CODE))
from classic_precision import t4_balanced_operations as old
from classic_precision.refinement_worker import require, read, verified
from classic_precision.t4_balanced_operations import write_new
from classic_precision.storage import digest, file_sha256


def descriptor(path):
    p = Path(path).absolute()
    require(p.is_file() and not p.is_symlink(), 'Expected regular file: ' + str(p))
    return {'path': str(p), 'sha256': file_sha256(p)}


def arms(phase):
    if phase == 't1':
        return [(source, model, card) for card in CARDS for model, owner in T1_CARD.items() if owner == card
                for source in ('t1_classic', 't1_natural')]
    require(phase == 't4', 'Unknown phase')
    return [('t4_classic', model, card) for card in CARDS for model in MODELS]


def key(source, model, card):
    return '-'.join((source, model, card))


def plan_sources():
    plans = {task: descriptor(HERE / f'{task}-plan.json') for task in ('t1', 't4')}
    t1, t4 = (verified(plans[task]) for task in ('t1', 't4'))
    rows = {}
    for domain, cases in (
        ('t1_classic', [(r['sample_id'], r['correct_answer'], {'sf': r['params']['sf_cycles_per_aperture'],
            'm': r['params']['modulation_depth'], 'phase_index': r['phase_index']}) for r in t1['classic']]),
        ('t1_natural', [(c['sample_id'], c['truth'], {'sf': r['sf'], 'm': r['m'],
            'phase_index': r['phase_index'], 'background_id': r['background_id'], 'background_group': r['group']})
            for r in t1['natural'] for c in r['cases']]),
        ('t4_classic', [(r['sample_id'], r['truth'], {f: r[f] for f in
            ('N', 'coherence', 'layout', 'baseline_id')}) for r in t4['clips']]),
    ):
        require(len(cases) == COUNTS[domain], 'Source count differs: ' + domain)
        require(len({sid for sid, _, _ in cases}) == COUNTS[domain], 'Source IDs duplicated')
        for sid, truth, params in cases:
            require(sid not in rows and truth in ('LEFT', 'RIGHT'), 'Bad or colliding source ID')
            if domain.startswith('t1'):
                require(params['m'] == 0.5, 'Unexpected T1 m')
                rel = Path('t1') / domain.split('_')[1] / sid
            else:
                require(params['N'] in (8, 16) and params['coherence'] > 0, 'Unexpected T4 condition')
                rel = Path('t4') / sid
            rows[sid] = {'source': domain, 'truth': truth, 'parameters': params, 'directory': str(rel)}
    return plans, rows


def donor_path(source, model, card):
    if source == 't4_classic':
        relative = f'{OLD_T4}/{model}-{card}-profile.json'
    elif source == 't1_classic':
        relative = CLASSIC[model].format(card=card)
    else:
        relative = 'natural-x-t123-20261101/' + NATURAL[model].format(card=card)
    return ROOT / 'plans' / relative


def donor(source, model, card):
    profile = descriptor(donor_path(source, model, card))
    p = verified(profile)
    require(p['model'] == MODELS[model] and p['environment']['CUDA_VISIBLE_DEVICES'] == card[-1]
            and Path(p['workspace_root']) == ROOT and p['hostname'] == socket.gethostname(),
            'Accepted source profile must match the actual card/host: ' + profile['path'])
    quantization = p['engine_kwargs'].get('quantization')
    require(quantization in ('modelopt', 'modelopt_fp4')
            or (model == 'nemo' and quantization is None and '-NVFP4/82d93699' in p['checkpoint']),
            'Expected official NVFP4 deployment')
    if model == 'qwen':
        e = p['engine_kwargs']
        expected = (65536, 4) if source == 't4_classic' else (98304, 8)
        require((e['max_model_len'], e['kv_cache_memory_bytes']) == (expected[0], expected[1] * 1024**3),
                'Qwen T1/T4 resource configuration differs')
    task = source[:2].upper()
    out = Path(p['output'])
    prompt_parent, prompt_out = p, out
    if source == 't1_natural' and model == 'cosmos' and card == 'fourier1':
        # F1 ran natural T2/T3 while F0 ran natural T1. Reuse the precise
        # T1 text/tokens only after checking the two native processor settings.
        prompt_parent = read(donor_path(source, model, 'fourier0'))
        fields = ('checkpoint', 'checkpoint_assets', 'engine_kwargs', 'request_mm_processor_kwargs',
                  'request_mm_processor_kwargs_by_task', 'media_io_kwargs', 'expected_packages')
        require(all(p.get(f) == prompt_parent.get(f) for f in fields), 'Natural Cosmos processor settings differ across cards')
        prompt_out = Path(prompt_parent['output'])
    with (prompt_out / 'raw.jsonl').open() as f:
        original = next((row for line in f if (row := json.loads(line))['task'] == task), None)
    require(original is not None and original['profile_sha256'] == digest(prompt_parent)
            and original['finish_reason'] == 'stop'
            and original['sample_id'] in prompt_parent['assigned_ids'] and original['actual_prompt_token_ids'],
            'Original native prompt evidence has different lineage: ' + str(out))
    prompt = original['question']
    if source.startswith('t1'):
        require(prompt.endswith('\n') == (source == 't1_natural'), 'T1 domain prompt bytes differ')
    return p, {'profile': profile, 'runtime': descriptor(out / 'runtime.json'),
               'reference_prompt_catalog': p.get('reference_prompt_catalog'),
               'prompt': prompt, 'prompt_ids': original['actual_prompt_token_ids'],
               'original_prompt_sample_id': original['sample_id'], 'original_raw_record_digest': digest(original)}


def source_row(sid, spec, plans, source_root, prompt, complete=None):
    case = source_root / spec['directory']
    receipt_ref = descriptor(case / 'receipt.json')
    r = verified(receipt_ref)
    phase = spec['source'][:2]
    extension = '.mp4' if spec['source'] == 't1_natural' else '.mkv'
    media = case / (sid + extension)
    require(r['sample_id'] == sid and r['file'] == media.name and r['frames'] == 180
            and r['plan_sha256'] == plans[phase]['sha256'] and media.is_file() and not media.is_symlink(),
            'Per-video source receipt differs: ' + sid)
    if spec['source'] == 't1_classic':
        require(r['measurement']['pass'] and r['measurement']['decoded_frames'] == 180,
                'Classic decoded measurement failed: ' + sid)
    elif spec['source'] == 't1_natural':
        require(r['measurement']['passed'] and r['frames_byte_equal'] == r['timestamps_checked'] == 180,
                'Natural decoded measurement failed: ' + sid)
    else:
        require(r['baseline_id'] == spec['parameters']['baseline_id'] and len(r['rgb16_frame_sha256']) == 180
                and r['minimum_edge_gap_px'] > 0 and r['minimum_continuous_center_distance_px'] > 56,
                'Strict T4 geometry or full-frame source evidence missing: ' + sid)
    if complete is not None:
        expected = {'path': str(media.relative_to(source_root)), 'video_sha256': r['video_sha256'],
                    'receipt_sha256': receipt_ref['sha256'], 'bytes': media.stat().st_size}
        require(all(complete.get(k) == v for k, v in expected.items()),
                'Formal source ledger differs: ' + sid)
    return {'sample_id': sid, 'task': phase.upper(), 'video_contract': CONTRACT, 'accuracy_eligible': True,
            'source_receipt': receipt_ref, 'source_version': spec['source'], 'parameters': spec['parameters'],
            'row': {'sample_id': sid, 'task': phase.upper(), 'truth': spec['truth'], 'prompt': prompt,
                    'choices': ['LEFT', 'RIGHT'], 'accuracy_eligible': True},
            'content': [{'type': 'video', 'path': str(media), 'sha256': r['video_sha256']},
                        {'type': 'text', 'text': prompt}]}


def prepare(phase, source_root, out, owner, pilot=False):
    """Build CPU drafts. A pilot cannot be sealed or dispatched."""
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    plans, planned = plan_sources()
    refs = {}
    if not pilot:
        require(source_root == SOURCE, 'Formal drafts must bind the canonical formal source directory')
        refs['source_receipt'] = descriptor(source_root / 'source-receipt.json')
        complete = verified(refs['source_receipt'])
        require(complete['status'] == 'PASS_COMPLETE_SOURCE' and complete['counts'] == COUNTS
                and complete['frames'] == 180 * sum(COUNTS.values()) and set(complete['items']) == set(planned)
                and all(complete[f'{task}_plan_sha256'] == plans[task]['sha256'] for task in plans),
                'Formal combined source acceptance differs')
    require(not out.exists(), 'Fresh draft directory required')
    refs.update(plans=plans, code=descriptor(__file__), dependencies=old.dependency_refs())
    out.mkdir(parents=True)
    all_drafts, catalogs = {}, {}
    for source, model, card in arms(phase):
        p, original = donor(source, model, card)
        candidates = [sid for sid, r in planned.items() if r['source'] == source]
        if pilot:
            candidates = [sid for sid in candidates if (source_root / planned[sid]['directory'] / 'receipt.json').is_file()][:1]
            require(len(candidates) == 1, 'Missing generated pilot sample: ' + source)
        if source not in catalogs:
            items = {sid: source_row(sid, planned[sid], plans, source_root, original['prompt'],
                                    None if pilot else complete['items'][sid]) for sid in candidates}
            catalogs[source] = write_new(out / f'{source}-catalog.json', {'items': items})
        else:
            items = verified(catalogs[source])['items']
            require(next(iter(items.values()))['row']['prompt'] == original['prompt'], 'Same-domain model prompts differ')
        ids = [sid for sid in candidates if pilot or phase != 't4' or
               planned[sid]['parameters']['layout'] % 2 == int(card[-1])]
        name = key(source, model, card)
        draft = {field: copy.deepcopy(p[field]) for field in COPY_FIELDS if field in p}
        draft.update(owner_chat=owner, purpose='PILOT_CPU_ONLY' if pilot else 'FORMAL_FINITE_X',
            bindings=refs, donor=original, source_catalog=catalogs[source], source_version=source,
            assigned_ids=ids, maximum_model_calls=len(ids), choices_by_task={phase.upper(): ['LEFT', 'RIGHT']},
            model=MODELS[model], card=card, worker=name, phase=phase, automatic_retry=False, formal_auto_resume=False,
            output=str(RUN / phase / name), claim_root=str(RUN / phase / f'{name}-claims'),
            pause_file=str(RUN / phase / f'{name}-pause.json'))
        all_drafts[name] = write_new(out / f'{name}-draft.json', draft)
    return write_new(out / 'drafts.json', {'phase': phase, 'owner_chat': owner, 'pilot': pilot, 'drafts': all_drafts})


def checked(profile):
    """Revalidate exact inherited settings and immutable artifacts."""
    require(descriptor(__file__) == profile['bindings']['code'], 'Execution adapter changed')
    for ref in profile['bindings']['dependencies']:
        require(descriptor(ref['path']) == ref, 'Qualified dependency changed: ' + ref['path'])
    parent = verified(profile['donor']['profile'])
    for field in COPY_FIELDS:
        require(profile.get(field) == parent.get(field), 'Inherited setting changed: ' + field)
    require(profile['model'] == parent['model'] and socket.gethostname() == profile['hostname']
            and profile['environment']['CUDA_VISIBLE_DEVICES'] == profile['card'][-1], 'Wrong deployment host or card')
    c = verified(profile['source_catalog'])
    ids = profile['assigned_ids']
    require(len(ids) == len(set(ids)) == profile['maximum_model_calls'] and ids and set(ids) <= set(c['items']),
            'Allocated source scope differs')
    return c


def native_frames(reader):
    require(reader['shape'] == [180, 512, 768, 3] and reader['dtype'] == 'uint8'
            and reader['metadata']['frames_indices'] == list(range(180))
            and reader['metadata']['fps'] == 60 and reader['metadata']['duration'] == 3.0,
            'Native reader did not retain all ordered full-size frames and source clock')


def cpu(draft_ref, output):
    p = verified(draft_ref)
    c = checked(p)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.executable == p['python'],
            'CPU admission requires the qualified interpreter and hidden CUDA')
    packages = {n: importlib.metadata.version(n) for n in p['expected_packages']}
    require(packages == p['expected_packages'], 'Qualified package versions differ')
    item = c['items'][p['assigned_ids'][0]]
    gemma_prompts = None
    if p['model'] == MODELS['gemma']:
        from classic_precision import gemma_x_full_input as gi
        from classic_precision import official_x_runner as base
        from vllm.multimodal.media.image import ImageMediaIO
        from vllm.multimodal.media.video import VideoMediaIO
        task = item['task']
        prompt_path = Path(output).with_suffix('.gemma-prompts.json')
        # Only narrows the table builder's task allow-list in this CPU process.
        original = gi.TASKS
        try:
            gi.TASKS = (task,)
            gi.build_prompt_catalog(p, {'items': {item['sample_id']: item}}, prompt_path)
        finally:
            gi.TASKS = original
        gemma_prompts = descriptor(prompt_path)
        table = verified(gemma_prompts)
        with base.NativeObserver(p) as observer:
            observer.active, observer.deep, observer.contract = True, True, CONTRACT
            video = item['content'][0]
            VideoMediaIO(ImageMediaIO(), **base.MEDIA['video']).load_file(Path(video['path']))
        reader = observer.reads[0]
        ids = table['prompt_token_ids'][task]
        processor = table['representatives'][task]
    else:
        proof = old.cpu_video_sample(p, item)
        require(proof['fits_context'] and not proof['CUDA_initialized'] and not proof['weights_loaded'],
                'Native prompt budget or CPU isolation failed')
        reader, ids = proof['source_reader'][0], proof['prompt_token_ids']
        processor = proof
    native_frames(reader)
    require(ids == p['donor']['prompt_ids'] and len(ids) + 16384 <= p['engine_kwargs']['max_model_len'],
            'Actual native CPU prompt tokens differ from the original qualified run')
    return write_new(output, {'status': 'PASS_NATIVE_CPU_NO_MODEL', 'draft': draft_ref,
        'sample_id': item['sample_id'], 'item_digest': digest(item), 'source_sha256': item['content'][0]['sha256'],
        'native_reader': reader, 'prompt_token_ids': ids, 'processor': processor, 'gemma_prompts': gemma_prompts,
        'actual_packages': packages, 'CUDA_initialized': False, 'weights_loaded': False, 'model_calls': 0})


def draft_digest(p):
    return digest({k: v for k, v in p.items() if k not in SEAL_FIELDS})


def cpu_proof(p):
    proof = verified(p['native_CPU'])
    draft = verified(proof['draft'])
    require(draft_digest(p) == draft_digest(draft) == p['draft_digest']
            and proof['status'] == 'PASS_NATIVE_CPU_NO_MODEL' and proof['model_calls'] == 0
            and proof['sample_id'] == p['assigned_ids'][0] and proof['prompt_token_ids'] == p['donor']['prompt_ids']
            and proof['CUDA_initialized'] is False and proof['weights_loaded'] is False,
            'CPU admission or sealed draft identity differs')
    item = verified(p['source_catalog'])['items'][proof['sample_id']]
    require(proof['item_digest'] == digest(item) and proof['source_sha256'] == item['content'][0]['sha256'],
            'CPU evidence is for a different video or prompt')
    native_frames(proof['native_reader'])
    return proof


def seal(draft_ref, cpu_ref, output):
    p = verified(draft_ref)
    checked(p)
    require(p['purpose'] == 'FORMAL_FINITE_X' and os.environ.get('CUDA_VISIBLE_DEVICES') == ''
            and sys.executable == p['python'], 'Seal requires formal draft, qualified interpreter and hidden CUDA')
    p.update(draft_digest=draft_digest(p), native_CPU=cpu_ref)
    proof = cpu_proof(p)
    if proof['gemma_prompts']:
        p['reference_prompt_catalog'] = (proof['gemma_prompts'] if p['phase'] == 't4'
                                         else p['donor']['reference_prompt_catalog'])
        reference = verified(p['reference_prompt_catalog'])
        require(reference['prompt_token_ids'][p['phase'].upper()] == proof['prompt_token_ids'],
                'Qualified Gemma prompt table differs from actual CPU')
    old.claims_gate(p)
    from classic_precision.gemma_x_full import storage_admission
    storage_admission(p)
    if p['model'] == MODELS['qwen']:
        from classic_precision import official_x_runner_qsa as route
    else:
        from classic_precision import official_x_runner_v2 as route
    with tempfile.TemporaryDirectory(prefix='t1m05-t4-seal-') as temporary:
        path = Path(temporary) / 'draft.json'
        write_new(path, p)
        result = route.seal(path, output)
    require(draft_digest(read(output)) == p['draft_digest'], 'Native seal changed inherited settings')
    return result


def arm(drafts_ref, profiles_dir, owner, previous=None):
    """Validate the entire phase before writing its one-use card intents."""
    stage = verified(drafts_ref)
    phase = stage['phase']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and stage['pilot'] is False
            and stage['owner_chat'] == owner and owner, 'Wrong CPU arm scope or owner')
    require(set(stage['drafts']) == {key(*a) for a in arms(phase)}, 'Incomplete staged scope')
    if phase == 't4':
        require(previous, 'T4 requires the complete T1 terminal receipt')
        t1 = verified(previous)
        require(t1['status'] == 'COMPLETE_PASS' and t1['phase'] == 't1'
                and t1['owner_chat'] == owner and t1['completed'] == 3072
                and t1['source_receipt'] == verified(next(iter(stage['drafts'].values())))['bindings']['source_receipt'],
                'T4 predecessor must close the same batch T1 allocation')
    else:
        require(previous is None, 'Unexpected T1 predecessor')
    _, planned = plan_sources()
    expected = {(model, sid) for model in MODELS.values() for sid, item in planned.items()
                if item['source'].startswith(phase)}
    seen = set()
    profiles = {card: [] for card in CARDS}
    for source, model, card in arms(phase):
        name = key(source, model, card)
        ref = descriptor(Path(profiles_dir) / f'{name}-profile.json')
        p = verified(ref)
        checked(p)
        proof = cpu_proof(p)
        old.claims_gate(p)
        require(p['purpose'] == 'FORMAL_FINITE_X' and p['owner_chat'] == owner
                and p['worker'] == name and p['card'] == card and p['model'] == MODELS[model]
                and p['source_version'] == source and proof['draft'] == stage['drafts'][name]
                and p['output'] == str(RUN / phase / name) and not Path(p['output']).exists()
                and p['claim_root'] == str(RUN / phase / f'{name}-claims'),
                'Profile differs from one-use phase assignment: ' + name)
        pairs = {(p['model'], sid) for sid in p['assigned_ids']}
        require(not seen & pairs, 'Duplicate model/source assignment')
        seen.update(pairs)
        profiles[card].append(ref)
    require(seen == expected, 'Missing model/source pairs or unexpected IDs')
    intents = {card: RUN / phase / f'{card}-arm.json' for card in CARDS}
    require(not any(path.exists() for path in intents.values()), 'Phase already armed')
    return {card: write_new(intents[card], {'status': 'ROOT_ARMED_ONCE', 'owner_chat': owner, 'phase': phase,
            'card': card, 'drafts': drafts_ref, 'profiles': profiles[card], 'execution': descriptor(__file__),
            't1_terminal': previous, 'automatic_retry': False}) for card in CARDS}


def admission(profile, arm_ref):
    p, a = verified(profile), verified(arm_ref)
    require(a['status'] == 'ROOT_ARMED_ONCE' and a['execution'] == descriptor(__file__) and profile in a['profiles']
            and a['card'] == p['card'] and a['phase'] == p['phase'] and a['owner_chat'] == p['owner_chat'],
            'No matching root arm for this exact profile')
    if a['phase'] == 't4':
        require(verified(a['t1_terminal'])['status'] == 'COMPLETE_PASS', 'T1 terminal proof changed')
    return p


def run(profile, arm_ref):
    p = admission(profile, arm_ref)
    checked(p)
    cpu_proof(p)
    old.claims_gate(p)
    require(sys.executable == p['python'], 'Wrong native worker interpreter')
    from classic_precision.gemma_x_full import storage_admission
    write_new(Path(p['output']).with_suffix('.storage.json'), storage_admission(p))
    if p['model'] == MODELS['qwen']:
        from classic_precision import official_x_runner_qsa as qsa
        require(descriptor(qsa.__file__) == p['qsa_entry'] and descriptor(qsa.selector.__file__) == p['selector_source'],
                'Qualified QSA implementation changed')
        native = qsa.v2
    else:
        from classic_precision import official_x_runner_v2 as native
    identity = native.base.worker_identity
    if p['model'] == MODELS['nemo']:
        from classic_precision.nemo_x_full import worker_identity as identity
    elif p['model'] == MODELS['gemma']:
        from classic_precision.gemma_x_full import worker_identity as identity
    expected = verified(p['donor']['runtime'])[0]

    def checked_runtime(worker, model):
        actual = identity(worker, model)
        require(old.runtime_signature(actual) == old.runtime_signature(expected),
                'Actual runtime differs from the qualified deployment before first claim')
        return actual

    native.base.worker_identity = checked_runtime
    return native.run(profile['path'])


def supervisor(profile, arm_ref):
    p = admission(profile, arm_ref)
    out = Path(p['output'])
    old.claims_gate(p)
    from classic_precision.gemma_speed import owned_worker_alive
    from classic_precision.static_x_queue import process_identity, spawn_and_wait
    env = os.environ.copy()
    for field in p.get('cleared_environment', []):
        env.pop(field, None)
    env.update(p['environment'])
    args = [p['python'], '-B', str(Path(__file__).resolve()), 'run', '--input', profile['path'],
            '--arm', arm_ref['path'], '--arm-sha', arm_ref['sha256'], '--profile-sha', profile['sha256']]
    before = time.monotonic()
    caught = None
    with out.with_suffix('.log').open('xb') as log:
        try:
            child, code = spawn_and_wait(args, log=log, cwd=str(CODE), env=env,
                child_receipt=out.with_suffix('.launch.json'), identity_key='model_child',
                receipt_fields={'supervisor': process_identity(), 'profile': profile, 'arm': arm_ref})
        except BaseException as error:
            caught = error
            receipt = getattr(error, 'static_queue_child_wait', None)
            if not receipt:
                raise
            child, code = receipt['identity'], receipt['returncode']
    exit_path = out / 'exit.json'
    owned = read(exit_path)['owned_workers'] if exit_path.is_file() else []
    if not owned and (out / 'engine-owner.json').is_file():
        owned = [value['worker'] for value in read(out / 'engine-owner.json')['owned']]
    engines_exited = all(not owned_worker_alive(value) for value in owned)
    lock = p['physical_lock']
    lock_path = Path(lock['path'])
    require(lock_path.is_file() and not lock_path.is_symlink(), 'Original physical lock missing')
    with lock_path.open('r+b') as stream:
        stat = os.fstat(stream.fileno())
        require((stat.st_dev, stat.st_ino) == (lock['st_dev'], lock['st_ino']), 'Physical lease identity changed')
        released = False
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            released = True
        except BlockingIOError:
            pass
    ref = write_new(out.with_suffix('.wait.json'), {'profile': profile, 'arm': arm_ref, 'model_child': child,
        'returncode': code, 'real_wait_observed': True, 'owned_engines_exited': engines_exited,
        'original_physical_lease_released': released, 'seconds': time.monotonic() - before,
        'automatic_retry': False, 'supervisor_error': str(caught) if caught else None})
    if caught:
        raise caught
    require(code == 0 and engines_exited and released, 'Worker or lease did not close cleanly')
    return ref


def queue(arm_ref):
    a = verified(arm_ref)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Queue controller must have CUDA hidden')
    from classic_precision.static_x_queue import process_identity
    out = RUN / a['phase'] / a['card']
    write_new(out.with_suffix('.queue-launch.json'), {'owner': process_identity(), 'arm': arm_ref, 'automatic_retry': False})
    closures = []
    for ref in a['profiles']:
        supervisor(ref, arm_ref)
        result, _ = close_worker(ref, arm_ref)
        closures.append(result)
    return write_new(out.with_suffix('.queue-result.json'),
                     {'status': 'COMPLETE_PASS', 'arm': arm_ref, 'closures': closures, 'automatic_retry': False})


def close_worker(profile, arm_ref):
    p = admission(profile, arm_ref)
    c, proof = checked(p), cpu_proof(p)
    old.claims_gate(p, complete=True)
    out = Path(p['output'])
    wait_ref = descriptor(out.with_suffix('.wait.json'))
    wait = verified(wait_ref)
    launch = read(out.with_suffix('.launch.json'))
    result, exited = read(out / 'result.json'), read(out / 'exit.json')
    ids = p['assigned_ids']
    require(wait['profile'] == launch['profile'] == profile and wait['arm'] == launch['arm'] == arm_ref
            and wait['model_child'] == launch['model_child']
            and wait['returncode'] == 0 and wait['real_wait_observed'] is True and wait['supervisor_error'] is None
            and wait['owned_engines_exited'] is True and wait['original_physical_lease_released'] is True
            and result['status'] == 'COMPLETE' and result['completed_ids'] == ids and not result['UNCERTAIN_ids']
            and exited['owned_engine_workers_exited'] is True and exited['shutdown_error'] is None
            and not (out / 'failure.json').exists(), 'No real clean worker closure')
    raw, events, deep = (old.records(out / f'{name}.jsonl') for name in
                         ('raw', 'events', 'deep-observations'))
    require([row['sample_id'] for row in raw] == ids
            and [(x['event'], x['sample_id']) for x in events] == [(e, sid) for sid in ids for e in ('INTENT', 'TECHNICAL_PASS')]
            and len(deep) == 1 and deep[0]['sample_id'] == ids[0] and deep[0]['raw_digest'] == digest(raw[0]),
            'Complete ordered raw, event or ingress journals differ')
    runtime = read(out / 'runtime.json')
    require(len(runtime) == 1 and old.runtime_signature(runtime[0]) == old.runtime_signature(verified(p['donor']['runtime'])[0]),
            'Actual closed runtime is unqualified')
    scored, profile_digest = [], digest(p)
    for i, row in enumerate(raw):
        sid = row['sample_id']
        item = c['items'][sid]
        r = item['row']
        claim = read(Path(p['claim_root']) / (sid + '.json'))
        g = row['generation']
        answer, raw_digest = row['raw_answer'].strip(), digest(row)
        require(claim == {'sample_id': sid, 'profile_sha256': profile_digest, 'worker_token': p['worker_token'], 'attempt': 1}
                and row['profile_sha256'] == profile_digest and row['attempt'] == 1 and row['ordinal'] == i
                and row['model'] == p['model'] and row['task'] == item['task'] and row['deep'] is (i == 0)
                and row['truth'] == r['truth'] and row['question'] == r['prompt'] and row['accuracy_eligible'] is True
                and row['finish_reason'] == 'stop' and answer in r['choices']
                and row['correct'] is (answer == r['truth']) and events[2*i+1]['raw_digest'] == raw_digest
                and row['source']['catalog'] == p['source_catalog'] and row['source']['item_digest'] == digest(item)
                and row['source']['media'] == [p['media_files'][item['content'][0]['path']]]
                and row['actual_prompt_token_ids'] == proof['prompt_token_ids']
                and (g['n'], g['temperature'], g['top_p'], g['seed'], g['max_tokens'], g['thinking'],
                     g['native_EOS_override'], g['choice']) == (1, 0, 1, 0, 16384, False, False, r['choices']),
                'Raw answer, source, prompt IDs, decode or claim differs: ' + sid)
        require(len(row['native_reader']) == 1, 'Unexpected reader count: ' + sid)
        native_frames(row['native_reader'][0])
        scored.append({'model': p['model'], 'card': p['card'], 'sample_id': sid,
            'source_version': p['source_version'], **item['parameters'], 'truth': r['truth'],
            'raw_answer': row['raw_answer'], 'correct': row['correct'], 'strict_format': row['raw_answer'] in r['choices'],
            'source_sha256': item['content'][0]['sha256'], 'raw_digest': raw_digest})
    require(raw[0]['native_reader'][0]['decoded_sha256'] == proof['native_reader']['decoded_sha256'],
            'First GPU request read a different actual uint8 input than CPU admission')
    if p['model'] != MODELS['nemo']:
        admissions = old.records(out / 'input-admissions.jsonl')
        require(len(admissions) == 1 and admissions[0]['sample_id'] == ids[0], 'Actual ingress proof missing')
        ingress = admissions[0]
        if p['model'] == MODELS['gemma']:
            require(ingress['status'] == 'PASS' and ingress['actual_GPU_chunks_equal_independent_CPU'] is True
                    and ingress['frame_count'] == 180 and ingress['profile_sha256'] == profile_digest
                    and ingress['source_item_digest'] == digest(c['items'][ids[0]]), 'Gemma ingress proof differs')
        else:
            require(ingress['proof']['status'] == 'PASS_OFFICIAL_RAW_GPU_INGRESS'
                    and ingress['raw_digest'] == digest(raw[0])
                    and ingress['proof']['CPU_to_GPU_patch_bytes_equal'] is True
                    and ingress['proof']['source_reader_all180'] is True, 'Official CUDA input proof differs')
    return {'status': 'COMPLETE_PASS', 'profile': profile, 'wait': wait_ref, 'completed': len(ids),
            'source_receipt': p['bindings']['source_receipt'], 'raw': descriptor(out / 'raw.jsonl')}, scored


def cell_table(phase, scored):
    conditions = ('sf', 'm') if phase == 't1' else ('N', 'coherence')
    groups = defaultdict(list)
    for row in scored:
        groups[(row['model'], row['source_version'], *(row[k] for k in conditions))].append(row)
    expected_cells = 4 * 8 * 2 if phase == 't1' else 4 * 16
    require(len(groups) == expected_cells, 'Unexpected condition coverage')
    cells = []
    for (model, source, *values), rows in sorted(groups.items()):
        truth = Counter(r['truth'] for r in rows)
        require(truth == {'LEFT': 24, 'RIGHT': 24} and len(rows) == 48,
                'Unexpected condition denominator or balance')
        cells.append({'model': model, 'source_version': source, **dict(zip(conditions, values)),
                      'n': len(rows), 'correct': sum(r['correct'] for r in rows),
                      'strict_format': sum(r['strict_format'] for r in rows),
                      'answers_left': sum(r['raw_answer'].strip() == 'LEFT' for r in rows)})
    return cells


def finish(phase, owner, output):
    """Prove exact phase coverage, then manifest only owned results and evidence."""
    output = Path(output).resolve()
    require(output.is_relative_to(RUN / phase) and not output.exists(), 'Use fresh in-phase terminal path')
    refs, files, scored = [], {descriptor(__file__)['path']}, []
    for card in CARDS:
        prefix = RUN / phase / card
        intent = descriptor(prefix.with_name(card + '-arm.json'))
        arm_value = verified(intent)
        queued = prefix.with_suffix('.queue-result.json')
        receipt = read(queued)
        require(owner == arm_value['owner_chat'] and receipt['status'] == 'COMPLETE_PASS'
                and receipt['arm'] == intent, 'Queue did not close under this owner')
        files.update((intent['path'], str(queued), str(prefix.with_suffix('.queue-launch.json'))))
        for ref in arm_value['profiles']:
            closure, rows = close_worker(ref, intent)
            require(closure in receipt['closures'], 'Queue result and live terminal evidence differ')
            refs.append(closure)
            scored.extend(rows)
            p = verified(ref)
            out = Path(p['output'])
            proof = verified(p['native_CPU'])
            files.update(x['path'] for x in (ref, p['source_catalog'], p['native_CPU'],
                                             p['bindings']['source_receipt'], proof['draft']))
            if proof['gemma_prompts']:
                files.add(proof['gemma_prompts']['path'])
            files.update(str(path) for path in out.iterdir() if path.is_file() and path.suffix in ('.json', '.jsonl'))
            files.update(str(out.with_suffix(suffix)) for suffix in ('.launch.json', '.wait.json', '.storage.json'))
            files.update(str(Path(p['claim_root']) / (sid + '.json')) for sid in p['assigned_ids'])
    _, plan = plan_sources()
    expected = {(model, sid) for model in MODELS.values() for sid, item in plan.items() if item['source'].startswith(phase)}
    require(len(scored) == len(expected) == 3072 and {(x['model'], x['sample_id']) for x in scored} == expected,
            'Formal phase is missing IDs or contains duplicates')
    cells = cell_table(phase, scored)
    output.parent.mkdir(parents=True, exist_ok=True)
    scored_path, cells_path = output.with_suffix('.trials.jsonl'), output.with_suffix('.cells.csv')
    with scored_path.open('x') as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    with cells_path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    files.update((str(scored_path), str(cells_path)))
    receipts = {r['source_receipt']['sha256']: r['source_receipt'] for r in refs}
    require(len(receipts) == 1, 'Phase mixed source ledgers')
    return write_new(output, {'status': 'COMPLETE_PASS', 'owner_chat': owner, 'phase': phase, 'completed': len(scored),
        'cells': len(cells), 'source_receipt': next(iter(receipts.values())), 'worker_closures': refs,
        'files': [descriptor(path) for path in sorted(files)], 'media_or_weights_included': False})


def verify_return(manifest, local_root):
    """Verify a byte-preserved copy under the local benchmark root after transfer."""
    receipt = read(manifest)
    remote_root = Path(os.environ.get('PSYCHOPHYSICS_WORKSPACE', Path.cwd())).resolve()
    require(receipt['status'] == 'COMPLETE_PASS' and receipt['completed'] == 3072, 'Wrong terminal manifest')
    files = receipt['files']
    require(len({r['path'] for r in files}) == len(files), 'Duplicate returned path')
    for ref in files:
        remote = Path(ref['path'])
        require(remote.is_relative_to(remote_root) and '..' not in remote.parts, 'Return path escapes benchmark')
        local = Path(local_root) / remote.relative_to(remote_root)
        require(descriptor(local)['sha256'] == ref['sha256'], 'Returned bytes differ: ' + str(local))
    return {'status': 'PASS_ALL_RETURN_SHA', 'phase': receipt['phase'], 'files_verified': len(files),
            'completed': receipt['completed'], 'cells': receipt['cells']}

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('command', choices=('references', 'prepare', 'cpu', 'seal', 'arm', 'queue', 'run', 'finish', 'verify-return'))
    cli.add_argument('--phase', choices=('t1', 't4'))
    cli.add_argument('--source', type=Path, default=SOURCE)
    cli.add_argument('--pilot', action='store_true')
    cli.add_argument('--owner')
    cli.add_argument('--input', type=Path)
    cli.add_argument('--output', type=Path)
    cli.add_argument('--profiles', type=Path)
    cli.add_argument('--previous', type=Path)
    cli.add_argument('--arm', type=Path)
    cli.add_argument('--arm-sha')
    cli.add_argument('--profile-sha')
    cli.add_argument('--local-root', type=Path)
    args = cli.parse_args()
    if args.command == 'references':
        rows = {}
        for scope, model, card in arms(args.phase):
            _, p = donor(scope, model, card)
            rows[key(scope, model, card)] = {'profile': p['profile'], 'runtime': p['runtime'],
                'prompt_token_count': len(p['prompt_ids']), 'prompt_sha256': hashlib.sha256(p['prompt'].encode()).hexdigest()}
        result = rows
    elif args.command == 'prepare':
        require(args.owner, 'Explicit root owner identity required')
        result = prepare(args.phase, args.source, args.output, args.owner, args.pilot)
    elif args.command == 'cpu':
        result = cpu(descriptor(args.input), args.output)
    elif args.command == 'seal':
        result = seal(descriptor(args.input), descriptor(args.previous), args.output)
    elif args.command == 'arm':
        result = arm(descriptor(args.input), args.profiles, args.owner,
                     descriptor(args.previous) if args.previous else None)
    elif args.command == 'queue':
        result = queue(descriptor(args.input))
    elif args.command == 'run':
        result = run({'path': str(args.input), 'sha256': args.profile_sha},
                     {'path': str(args.arm), 'sha256': args.arm_sha})
    elif args.command == 'finish':
        result = finish(args.phase, args.owner, args.output)
    else:
        result = verify_return(args.input, args.local_root)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
