"""Natural T1--T3 admission around the existing qualified X scientific runners.

No stimulus generation, new qualification, scientific retry or model patch.
An unfrozen source version is never executable. Operational continuations use
separate claims/output/control identities; original scientific files stay put.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from . import official_x_runner as base
from .dataset_identity import IdentityRegistry
from .refinement_worker import Boundary as RequestBoundary, descriptor, read, require, verified
from .storage import digest, file_sha256

COUNTS = {'T1': 3840, 'T2': 3840, 'T3': 2736}
ORDER = ('cosmos', 'qwen', 'nemo', 'gemma')
MODELS = {'cosmos': 'cosmos', 'qwen': 'qwen_x_qsa', 'nemo': 'nemo', 'gemma': 'gemma_x_full'}
SETTINGS = ('model', 'base_model', 'checkpoint', 'checkpoint_assets', 'engine_kwargs',
    'expected_packages', 'request_mm_processor_kwargs', 'request_mm_processor_kwargs_by_task',
    'vision_observation', 'minimum_kv_capacity', 'torch_num_threads', 'media_io_kwargs',
    'bootstrap', 'request_adapter', 'vision_validation_hook', 'full_dependencies',
    'input_dependencies', 'operational_wrapper', 'qsa_entry', 'observer_entry', 'selector_source')
_NATIVE = {key: getattr(base, key) for key in ('worker_identity', 'install_probe',
    'read_probe', 'validate_deep', '_append', 'Boundary')}


def settings(profile):
    return {key: copy.deepcopy(profile.get(key)) for key in SETTINGS}


def catalog_from_publications(publication_refs, media_roots):
    """Project the approved publication rows, never the frozen render catalog.

    A metadata-only builder: paths and row layout change; IDs/videoSHA/GT/prompt
    are copied literally. No decoding, encoding, stimulus generation or model.
    Root must still approve/SHA-lock the resulting execution catalog.
    """
    items = {}; identity_rows = []
    for task, ref in publication_refs.items():
        require(task in COUNTS and file_sha256(ref['path']) == ref['sha256'], 'Publication projection changed')
        with Path(ref['path']).open() as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        for row in rows:
            sid = row.get('canonical_id') or row['sample_id']
            require(row['domain'] == 'natural' and row['task'] == task and sid not in items, 'Foreign/duplicate publication identity')
            path = Path(row['path']); require(not path.is_absolute() and '..' not in path.parts, 'Publication path escapes media root')
            identity_rows.append(row)
            item = {'sample_id': sid, 'task': task, 'row': {'sample_id': sid, 'task': task,
                'truth': row['truth'], 'prompt': row['prompt'], 'choices': ['LEFT','RIGHT'],
                'accuracy_eligible': True, 'source_parameters': copy.deepcopy(row['parameters'])},
                'video_contract': {key: row[key] for key in ('fps','frames','height','width')},
                'content': [{'type': 'video', 'path': str(Path(media_roots[task]) / path), 'sha256': row['video_sha256']},
                            {'type': 'text', 'text': row['prompt']}], 'publication_row_sha256': digest(row)}
            items[sid] = item
    require(Counter(item['task'] for item in items.values()) == COUNTS, 'Publication projection must be the current10416')
    identities = IdentityRegistry.from_rows(identity_rows)
    for row in identity_rows: source_identity(items[row.get('canonical_id') or row['sample_id']], identities, row['dataset_revision'])
    return {'schema': 'natural-x-publication-execution-catalog-v1', 'items': items,
            'publication_refs': publication_refs, 'model_calls': 0}


def registry(ref, resolve=verified):
    # Resolve and hash the descriptor even for CSV/JSONL registries.
    owner = getattr(resolve, '__self__', None)
    path = owner.path(ref['path']) if owner is not None and hasattr(owner, 'path') else Path(ref['path'])
    require(file_sha256(path) == ref['sha256'], 'Identity registry changed')
    if owner is not None and hasattr(owner, 'capture'): owner.capture(path, ref['sha256'])
    if path.suffix == '.csv': return IdentityRegistry.from_csv(path)
    if path.suffix == '.jsonl': return IdentityRegistry.from_jsonl(path)
    value = resolve(ref)
    return IdentityRegistry.from_rows(value.get('rows', value.get('bindings', [])))


def source_identity(item, identities, revision):
    sid = item['sample_id']; binding = identities.resolve(sid, domain='natural', dataset_revision=revision)
    row = item['row']; media = [part for part in item['content'] if part['type'] == 'video']
    require(binding.canonical_id == sid and item['task'] == row['task'] == binding.task
            and row['sample_id'] in (sid, binding.legacy_sample_id)
            and len(media) == 1 and len(item['content']) == 2
            and media[0]['sha256'] == binding.video_sha256
            and [part.get('text') for part in item['content'] if part['type'] == 'text'] == [row['prompt']]
            and row['truth'] in ('LEFT', 'RIGHT') and row['choices'] == ['LEFT', 'RIGHT']
            and row.get('accuracy_eligible', True) is True
            and item['video_contract'] == {'fps': 60, 'frames': 180, 'height': 512, 'width': 768},
            'Canonical natural video/GT/prompt/180-frame identity differs: ' + sid)
    return {'sample_id': sid, 'task': item['task'], 'legacy_sample_id': binding.legacy_sample_id,
            'video_sha256': binding.video_sha256, 'truth': row['truth'], 'prompt': row['prompt']}


def admission(approval_ref, catalog_ref, resolve=verified):
    approval = resolve(approval_ref)
    require(approval['schema'] == 'natural-x-source-approval-v1'
            and approval['status'] == 'ROOT_APPROVED_NATURAL_X_SOURCE_VERSION'
            and approval['source_version'] and approval['counts'] == COUNTS
            and tuple(approval['model_order']) == ORDER
            and catalog_ref in approval['catalogs'],
            'Natural source version is not root-approved and SHA-bound')
    catalog = resolve(catalog_ref); identities = registry(approval['registry'], resolve)
    items = catalog['items']; require(len(items) == sum(COUNTS.values()), 'Natural universe must contain10416 IDs')
    actual = Counter()
    identity_rows = []
    for sid, item in items.items():
        require(item['sample_id'] == sid, 'Catalog key/identity differs')
        identity_rows.append(source_identity(item, identities, approval['dataset_revision'])); actual[item['task']] += 1
    require(actual == COUNTS, 'Natural task denominators differ')
    signature = digest(sorted(identity_rows, key=lambda row: row['sample_id']))
    require(approval.get('source_identity_sha256', signature) == signature, 'Approved cross-host source identity differs')
    approval['computed_source_identity_sha256'] = signature
    return approval, catalog, identities


def science(profile, resolve=verified):
    spec = profile['natural_x']; model = spec['model']
    approval, catalog, identities = admission(spec['source_approval'], profile['source_catalog'], resolve)
    require(model in MODELS and profile['model'] == MODELS[model]
            and profile['phase'] == 'full' and profile['automatic_retry'] is False
            and profile['formal_auto_resume'] is False
            and spec['qualified_template'] in approval['templates'][model], 'Unapproved model/template or retry')
    template = resolve(spec['qualified_template'])
    require(settings(profile) == settings(template), 'Qualified X scientific settings changed')
    # Runtime location/card settings also come from the actual tested template.
    for key in ('python', 'gpu_uuid', 'physical_lock', 'environment', 'cleared_environment'):
        default = [] if key == 'cleared_environment' else None
        require(profile.get(key, default) == template.get(key, default), 'Tested deployment changed: ' + key)
    require((template.get('card') is None or profile['card'] == template['card'])
            and profile['hostname'] == template['hostname'] and profile['workspace_root'] == template['workspace_root']
            and (spec['shard'] == profile['card'] or any(spec['shard'].startswith(profile['card'] + separator)
                 for separator in ('/', ':', '-'))), 'Shard belongs to another tested card/host')
    namespace = spec['run_namespace']; root = Path(os.path.abspath(profile['workspace_root']))
    require(namespace.startswith('natural-x-') and Path(namespace).name == namespace,
            'Only a new natural execution namespace is owned')
    for key in ('output', 'claim_root'):
        require(Path(os.path.abspath(profile[key])).is_relative_to(root / 'runs/reproducibility' / namespace), 'Foreign natural run path')
    for key in ('control_file', 'pause_file'):
        require(Path(os.path.abspath(profile[key])).is_relative_to(root / 'plans' / namespace), 'Foreign control/pause path')
    require(Path(os.path.abspath(spec['fence'])).is_relative_to(root / 'plans' / namespace), 'Foreign run fence')
    ids = profile['assigned_ids']
    require(ids and len(ids) == len(set(ids)) == profile['maximum_model_calls']
            and set(ids) <= set(catalog['items']), 'Foreign/duplicate natural allocation')
    allocation = resolve(spec['allocation'])
    require(allocation['schema'] == 'natural-x-approved-allocation-v1'
            and allocation['status'] == 'ROOT_APPROVED_EXACT_REMAINING'
            and allocation['source_approval'] == spec['source_approval']
            and allocation['model'] == model and allocation['catalog'] == profile['source_catalog'],
            'Remaining denominator/allocation has not been root-approved')
    reused = allocation['accepted_reuse_ids']; remaining = allocation['remaining_ids']
    flat = [sid for shard in allocation['shards'].values() for sid in shard]
    require(len(reused) == len(set(reused)) and len(flat) == len(set(flat))
            and flat == remaining and not set(reused).intersection(flat)
            and set(reused) | set(flat) == set(catalog['items']), 'Reuse/remaining universe differs')
    reuse = resolve(allocation['reuse_receipt'])
    same_source = reuse['source_approval'] == spec['source_approval'] and reuse['catalog'] == profile['source_catalog']
    same_source = same_source or (reuse.get('source_version') == approval['source_version']
        and reuse.get('source_identity_sha256') == approval['computed_source_identity_sha256'])
    require(reuse['status'] == 'PASS_EXACT_CURRENT_SOURCE_REUSE_CPU' and reuse['model'] == model and same_source
            and reuse['accepted_ids'] == reused and reuse['remaining_ids'] == remaining,
            'Approved allocation does not match exact current-source accepted reuse')
    shard = allocation['shards'][spec['shard']]
    require(set(ids) <= set(shard) and (spec.get('parent') or ids == shard), 'Profile escaped original remaining shard')
    for sid in ids: source_identity(catalog['items'][sid], identities, approval['dataset_revision'])
    if model == 'gemma':
        table = resolve(profile['reference_prompt_catalog'])
        require(profile['reference_prompt_catalog'] == template['reference_prompt_catalog']
                or profile['reference_prompt_catalog'] in approval.get('prompt_catalogs', []), 'Unapproved prompt token catalog')
        for sid in ids:
            item = catalog['items'][sid]
            require(table['task_contracts'][item['task']]['question'] == item['row']['prompt'], 'Frozen Gemma task prompt differs')
    if model == 'cosmos':
        runtime = spec.get('qualified_runtime') or template.get('qualified_runtime')
        require(runtime is not None and runtime == approval['runtime_references'][spec['qualified_template']['sha256']],
                'Existing qualified Cosmos runtime reference is required')
        resolve(runtime)
    return approval, catalog


def validate_runtime(profile, observed, resolve=verified):
    require(observed, 'Missing actual runtime')
    model = profile['natural_x']['model']
    if model == 'cosmos':
        from .cosmos_t45 import RUNTIME_FIELDS
        spec = profile['natural_x']; template = resolve(spec['qualified_template'])
        expected = resolve(spec.get('qualified_runtime') or template['qualified_runtime'])
        require([{key: row[key] for key in RUNTIME_FIELDS} for row in observed]
                == [{key: row[key] for key in RUNTIME_FIELDS} for row in expected], 'Actual Cosmos runtime differs')
    else:
        if model == 'gemma':
            from .gemma_x_full import validate_runtime as check
        elif model == 'nemo':
            from .nemo_x_full import validate_runtime as check
        else:
            from .qwen_x_crosscard import validate_runtime as check
        for row in observed: check(row)


def draft(template_ref, approval_ref, catalog_ref, allocation_ref, shard, worker, output, claims, control, pause, prompt_catalog=None, card=None):
    template = verified(template_ref); approval = verified(approval_ref); allocation = verified(allocation_ref)
    model = allocation['model']; p = copy.deepcopy(template)
    require(card is not None or template.get('card') is not None, 'Existing template has no operation card label; supply the already bound physical card label')
    p['card'] = card if card is not None else template['card']; p.setdefault('cleared_environment', [])
    for key in ('implementation', 'worker_token', 'media_files', 'continuation_parent',
                'continuation_wait', 'control_generation', 'natural_x'):
        p.pop(key, None)
    ids = allocation['shards'][shard]
    p.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), phase='full',
        source_catalog=catalog_ref, output=str(output), claim_root=str(claims), control_file=str(control),
        pause_file=str(pause), control_generation=0, automatic_retry=False, formal_auto_resume=False,
        natural_x={'model': model, 'source_approval': approval_ref, 'qualified_template': template_ref,
            'allocation': allocation_ref, 'shard': shard, 'runner': descriptor(__file__),
            'run_namespace': approval['run_namespace'],
            'fence': str(Path(p['workspace_root']) / 'plans' / approval['run_namespace'] / 'fence.json')})
    runtime = approval.get('runtime_references', {}).get(template_ref['sha256'])
    if runtime: p['natural_x']['qualified_runtime'] = runtime
    if prompt_catalog is not None: p['reference_prompt_catalog'] = prompt_catalog
    science(p)
    return p


class Boundary(RequestBoundary):
    def check(self):
        fence = read(self.profile['natural_x']['fence'])
        if fence['paused'] or fence['generation'] != self.profile['control_generation']:
            self.requested = True
            return True
        control = read(self.profile['control_file'])
        if control['paused'] or control['generation'] != self.profile['control_generation']:
            self.requested = True
        return super().check()


@contextmanager
def scientific_context(profile):
    """Restore global admission hooks; delegate all numerical computation."""
    previous = {key: getattr(base, key) for key in _NATIVE}
    try:
        for key, value in _NATIVE.items(): setattr(base, key, value)
        model = profile['natural_x']['model']
        if model == 'qwen':
            from . import qwen_x_crosscard as qualified
            qsa = qualified.qsa
            qualified.fixed_implementation()
            for key in ('install_probe', 'read_probe', 'validate_deep'):
                setattr(base, key, getattr(qsa, key))
            base.worker_identity = qualified.worker_identity; base._append = qsa.append
        elif model == 'nemo':
            from . import nemo_x_full
            base.worker_identity = nemo_x_full.worker_identity
        elif model == 'gemma':
            from . import gemma_x_full
            gemma_x_full.verify_dependencies(profile)
            base.worker_identity = gemma_x_full.worker_identity
        elif model == 'cosmos':
            from . import official_x_runner_v2 as v2
            from . import cosmos_t45
            original = v2.validate_deep
            def validate(p, item, deep):
                ref = p['natural_x'].get('qualified_runtime') or verified(p['natural_x']['qualified_template'])['qualified_runtime']
                expected = verified(ref)
                fields = cosmos_t45.RUNTIME_FIELDS
                require([{k: part['runtime'][k] for k in fields} for part in deep]
                        == [{k: part[k] for k in fields} for part in expected], 'Actual Cosmos runtime differs')
                return original(p, item, deep)
            base.validate_deep = validate
        else: raise ValueError('Unknown natural model')
        base.Boundary = Boundary
        yield
    finally:
        for key, value in previous.items(): setattr(base, key, value)


def seal(draft, output):
    profile = read(draft); science(profile)
    require(profile['natural_x']['runner']['sha256'] == file_sha256(__file__), 'Natural operational runner changed')
    from .gemma_x_full import storage_admission
    storage_admission(profile)
    catalog = verified(profile['source_catalog'])['items']; files = {}
    for sid in profile['assigned_ids']:
        for entry in catalog[sid]['content']:
            if entry['type'] in ('video', 'image'): files[entry['path']] = entry['sha256']
    def hashed(pair):
        path, expected = pair; before = base.stat(path); actual = file_sha256(path)
        require(actual == expected and base.stat(path) == before, 'Source changed during parallel sealing')
        return path, (actual, before)
    with ThreadPoolExecutor(max_workers=min(16, max(2, os.cpu_count() or 2))) as pool:
        hashes = dict(pool.map(hashed, files.items()))
    native_sha = base.file_sha256
    def admitted(path):
        key = str(path)
        if key not in hashes: return native_sha(path)
        value, before = hashes[key]; require(base.stat(path) == before, 'Source changed after hash admission'); return value
    try:
      base.file_sha256 = admitted
      with scientific_context(profile):
        if profile['natural_x']['model'] == 'qwen':
            from .official_x_runner_qsa import seal as original_seal
        else: original_seal = base.seal
        return original_seal(draft, output)
    finally: base.file_sha256 = native_sha


def run(profile_path):
    profile = read(profile_path); science(profile)
    require(profile['natural_x']['runner']['sha256'] == file_sha256(__file__), 'Natural runner changed')
    if profile['natural_x'].get('parent'):
        from .natural_x_operations import Archive, continuation
        continuation(Archive(profile['workspace_root']), profile)
    from .gemma_x_full import storage_admission
    from .refinement_worker import write_new
    write_new(Path(profile['output']).with_suffix('.storage.json'), storage_admission(profile))
    with scientific_context(profile): return base.run(profile_path)


def supervise(profile_path):
    from . import refinement_worker as worker
    profile = read(profile_path); science(profile)
    native = worker.subprocess.Popen
    def popen(argv, **kwargs):
        require(argv[3:5] == ['classic_precision.refinement_worker', 'run'], 'Unknown worker constructor')
        argv = list(argv); argv[3] = __spec__.name
        return native(argv, **kwargs)
    worker.subprocess.Popen = popen
    try: return worker.supervise(profile_path)
    finally: worker.subprocess.Popen = native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('seal', 'run', 'supervise', 'check'))
    parser.add_argument('--profile'); parser.add_argument('--draft'); parser.add_argument('--output')
    args = parser.parse_args()
    if args.command == 'check': science(read(args.profile)); result = {'status': 'PASS_SOURCE_TEMPLATE_ADMISSION', 'model_calls': 0}
    elif args.command == 'seal': result = seal(args.draft, args.output)
    else: result = run(args.profile) if args.command == 'run' else supervise(args.profile)
    print(json.dumps(result))


if __name__ == '__main__': main()
