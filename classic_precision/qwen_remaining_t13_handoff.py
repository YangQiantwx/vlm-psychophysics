"""One finite reassignment of closed Radon work after both Fourier queues finish.

Preparation is CPU-only and never pauses or launches a worker. Assignment and
deployment change explicitly; model/input/generation settings and the original
qualification gate are retained. Existing records remain selected and immutable.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
import copy
import json
from pathlib import Path
import socket
import statistics
import subprocess
import tempfile

from . import qwen_remaining_t13 as q
from . import qwen_remaining_t13_storage as storage
from . import qwen_remaining_t13_archive as archive
from .official_x_results import journal
from .refinement_worker import read, require, verified, write_new
from .storage import atomic_json, digest, file_sha256

NAME = q.NAME
BATCH = 'handoff1-reboot1'
MODULE = 'classic_precision.qwen_remaining_t13_handoff'
CARDS = ('fourier0', 'fourier1', 'radon0')
GROUPS = ('T3', 'T1')
REL = Path('plans') / NAME / BATCH


def descriptor(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def relative_ref(root, path):
    return {'path': str(Path(path).relative_to(root)), 'sha256': file_sha256(path)}


def absolute_ref(root, ref):
    return {'path': str(Path(root) / ref['path']), 'sha256': ref['sha256']}


def portable_ref(root, ref, card='fourier0'):
    path = archive.Archive(root).path(ref['path']).relative_to(root)
    remote = q.ROOTS['radon' if card == 'radon0' else 'fourier']
    return {'path': str(Path(remote) / path), 'sha256': ref['sha256']}


def worker_name(card, group, generation=0):
    return card + '-full-' + group + '-' + BATCH + '-segment' + str(generation)


def authorization(value, manifest_sha):
    require(value['schema'] == NAME + '/handoff1-authorization-v1' and value['status'] == 'AUTHORIZED'
            and value['batch'] == BATCH and value['source_card'] == 'radon0'
            and value['cards'] == list(CARDS) and value['source_manifest_sha256'] == manifest_sha
            and value['maximum_new_model_calls'] == 1547 and value['automatic_retry'] is False
            and value['retain_completed'] is True and isinstance(value.get('user_message'), str)
            and value['user_message'].strip(), 'Finite handoff authorization differs')


def lineage_key(p):
    kind = BATCH if 'handoff_entry' in p else ('reboot1' if 'reboot_entry' in p else 'baseline')
    return (kind, p['card'], p['resource_group'])


def validate_partition(receipt, plan, resolve=verified):
    require(receipt['plan_sha256'] and receipt['implementation_sha256'] == file_sha256(__file__), 'Receipt source identity changed')
    from . import qwen_remaining_t13_reboot as reboot
    admitted = resolve(receipt['original_reboot_admission'])
    require(admitted['schema'] == reboot.SCHEMA and admitted['status'] == 'ACCEPTED_REBOOT_INTERRUPTED_PREFIX'
            and admitted['UNCERTAIN_ids'] == [reboot.UNCERTAIN]
            and admitted['automatic_uncertain_retry'] is False
            and admitted['real_original_wait_observed'] is False and admitted['original_returncode'] is None
            and admitted['maximum_new_never_claimed_calls'] == 1547,
            'Handoff requires the explicit interrupted-prefix admission')
    original = plan['cards']['radon0']['groups']
    old = admitted['accepted_prefix_ids']; excluded = admitted['UNCERTAIN_ids']
    require(len(old) == 228 and old == original['T3'][:228]
            and excluded == original['T3'][228:229]
            and admitted['never_claimed'] == {'T3': original['T3'][229:], 'T1': original['T1']},
            'Interrupted-prefix/uncertain/never-claimed source partition differs')
    require(receipt['batch'] == BATCH and receipt['UNCERTAIN_ids_not_reissued'] == excluded
            and receipt['expected_unique_total'] == q.TOTAL
            and receipt['maximum_accepted_without_uncertain_retry'] == q.TOTAL - len(excluded),
            'Handoff lost the original total or unresolved request boundary')
    for group in GROUPS:
        expected = admitted['never_claimed'][group]
        retained = receipt['retained_reboot_ids'][group]; remaining = receipt['remaining_ids'][group]
        require(retained == expected[:len(retained)] and remaining == expected[len(retained):]
                and receipt['donor_claim_ids'][group] == sorted(retained)
                and set(receipt['donor_controls'][group]['claims']) == set(retained), 'Retained/never-claimed boundary differs')
        require(receipt['retained_radon_ids'][group] == (old if group == 'T3' else []) + retained,
                'Retained Radon answers do not preserve both admitted boot lineages')
        assigned = [sid for c in CARDS for sid in receipt['allocation'][c][group]]
        require(len(assigned) == len(set(assigned)) == len(remaining) and set(assigned) == set(remaining),
                'Reallocated IDs are not an exact disjoint remaining cover')
        require(not set(assigned) & (set(old) | set(excluded)), 'Handoff reissues an answered or unresolved old claim')
    require(receipt['maximum_new_model_calls'] == sum(len(x) for x in receipt['remaining_ids'].values()) <= 1547,
            'Finite reboot handoff budget exceeded')


def science(p, resolve=None):
    resolve = archive.portable_resolver(p, resolve)
    if 'handoff_entry' not in p:
        if 'reboot_entry' in p:
            from . import qwen_remaining_t13_reboot as reboot
            return reboot.science(p, resolve)
        return storage.science(p, resolve)
    entry = p['handoff_entry']; root = Path(p['workspace_root'])
    require(set(entry) in ({'implementation', 'receipt', 'runtime_profile'}, {'implementation', 'receipt', 'runtime_profile', 'fence'})
            and entry['implementation']['sha256'] == file_sha256(__file__), 'Handoff entry changed')
    receipt = resolve(entry['receipt'])
    require(receipt['schema'] == NAME + '/handoff1-receipt-v1' and receipt['status'] == 'SEALED_NEVER_CLAIMED'
            and receipt['implementation_sha256'] == file_sha256(__file__)
            and receipt['plan_sha256'] == p['experiment_plan']['sha256'], 'Reallocation receipt differs')
    authorization(resolve(absolute_ref(root, receipt['authorization'])), receipt['source_manifest_sha256'])
    validate_partition(receipt, resolve(p['experiment_plan']),
                       lambda ref: resolve(q.rebase(ref, str(root))))
    if 'implementation' in p:
        require('fence' in entry, 'Sealed handoff profile lacks a bound donor fence')
    if 'fence' in entry:
        fence = resolve(entry['fence'])
        require(fence['schema'] == NAME + '/handoff1-fence-v1' and fence['status'] == 'DONOR_FENCED_PAUSED'
                and fence['receipt_sha256'] == entry['receipt']['sha256'], 'Bound donor fence differs')
    q.check_implementation(p['implementation_sources']); card = p['card']; group = p['resource_group']
    require(card in CARDS and group in GROUPS and p['phase'] == 'full'
            and not p['automatic_retry'] and not p['formal_auto_resume']
            and 'storage_entry' not in p and p['sources'] == {} and all(k not in p for k in q.FORBIDDEN),
            'Unexpected model/input/retry entry')
    ref = absolute_ref(root, receipt['runtime_profiles'][card][group])
    require(entry['runtime_profile'] == ref, 'Runtime profile is not the bound deployment template')
    template = resolve(ref)
    # The copied reference is the original, SHA-bound complete profile. Verify
    # its original science contract before comparing every deployment setting.
    storage.science(template, resolve)
    require(all(p.get(k) == template.get(k) for k in q.SETTING_KEYS)
            and p['hostname'] == template['hostname'] and p['resources'] == template['resources'],
            'Deployment science/resource setting changed')
    for key in ('implementation_sources', 'driver', 'canonical_catalog', 'qualification_gate',
                'experiment_plan', 'execution_authorization', 'reference_panel'):
        require(p[key] == template[key], 'Original source/gate binding changed: ' + key)
    q.authorized(p, resolve)
    require(p['source_catalog'] == absolute_ref(root, receipt['catalogs'][card]), 'Unbound reallocation catalog')
    expected = receipt['allocation'][card][group]; ids = p['assigned_ids']
    require(ids and len(ids) == len(set(ids)) == p['maximum_model_calls'], 'Duplicate/empty allocation')
    if p.get('continuation_parent'):
        parent = resolve(p['continuation_parent'])
        require(parent.get('handoff_entry') == entry and p['control_generation'] == parent['control_generation'] + 1
                and ids == parent['assigned_ids'][p['retained_prefix_length']:], 'Invalid handoff continuation')
        operational = {'implementation', 'operational_wrapper', 'assigned_ids', 'maximum_model_calls',
            'media_files', 'worker_token', 'worker', 'output', 'pause_file', 'control_generation',
            'continuation_parent', 'continuation_wait', 'retained_prefix_length'}
        require(all(parent.get(k) == p.get(k) for k in parent.keys() | p.keys() if k not in operational),
                'Continuation changed scientific configuration')
    else:
        require(ids == expected and p['control_generation'] == 0, 'Initial handoff allocation changed')
    require(set(ids) <= set(expected), 'Foreign handoff ID')
    worker = worker_name(card, group, p['control_generation'])
    require(p['worker'] == worker and Path(p['output']) == root / 'runs/reproducibility' / NAME / worker
            and Path(p['claim_root']) == root / 'runs/reproducibility' / NAME / ('claims-' + BATCH) / card / group
            and Path(p['control_file']) == root / REL / (card + '-' + group + '-control.json')
            and Path(p['pause_file']) == root / REL / (worker + '-PAUSE.json'), 'Handoff namespace differs')
    canonical = resolve(p['canonical_catalog'])['items']; catalog = resolve(p['source_catalog'])['items']
    for sid in ids:
        require(catalog[sid] == q.rebase(canonical[sid], str(root)), 'Transferred source/GT/prompt differs')
        q.validate_source(sid, catalog[sid], group)


@contextmanager
def science_context():
    previous = q.science; q.science = science
    try:
        yield
    finally:
        q.science = previous


def queue_terminal(arc, ref, allowed, postclose_admission=None):
    value = arc.ref(ref); state_path = arc.path(value['state_path'])
    require(state_path.exists(), 'Source queue has no state')
    state = read(state_path)
    if postclose_admission is not None:
        admitted = archive.verify_postclose_queue_admission(arc, ref, postclose_admission)
        require(admitted['effective_terminal_status'] in allowed, 'Supplement does not cover an allowed terminal boundary')
        return value, state, descriptor(state_path)
    require(state['status'] in allowed, 'Source queue is not safely terminal')
    require([w['profile'] for w in state['workers']] == value['profiles'][:len(state['workers'])], 'Queue supervisor sequence differs')
    if state['status'] == 'COMPLETE':
        require(len(state['workers']) == len(value['profiles']) and all(w['terminal_status'] == 'COMPLETE' for w in state['workers']),
                'Queue marked complete before every worker closed')
    for worker in state['workers']:
        require(worker['real_supervisor_wait'] and worker['returncode'] == 0
                and worker['terminal_status'] in ('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD'), 'Source supervisor did not close')
        p = arc.ref(worker['profile']); wait = read(arc.path(p['output']).with_suffix('.wait.json'))
        require(wait['profile_sha256'] == digest(p) and wait['returncode'] == 0 and wait['real_wait_observed']
                and wait['owned_engines_exited'] and wait['original_physical_lease_released']
                and wait['terminal_status'] == worker['terminal_status'], 'Source worker/lease is not closed')
    return value, state, descriptor(state_path)


def ready(root, request):
    arc = archive.Archive(root); request = read(request) if not isinstance(request, dict) else request
    status = {}
    for card in CARDS[:2]:
        queue = arc.ref(request['queues'][card]); path = arc.path(queue['state_path'])
        state = read(path).get('status') if path.exists() else 'NOT_STARTED'; status[card] = state
        if state == 'COMPLETE':
            queue_terminal(arc, request['queues'][card], ('COMPLETE',))
    result = 'READY_BOTH_FOURIER_COMPLETE' if set(status.values()) == {'COMPLETE'} else (
        'HOLD_FOURIER_FAILURE' if any(v.startswith(('HELD', 'FAILED')) for v in status.values()) else 'WAIT_FOURIER')
    return {'status': result, 'cards': status}


def allocate(remaining, canonical, seconds):
    """Keep available LR pairs intact; leave a claimed-mate orphan on Radon."""
    output = {card: {group: [] for group in GROUPS} for card in CARDS}; orphans = []
    for group in GROUPS:
        pairs = defaultdict(dict)
        for sid in remaining[group]:
            row = canonical[sid]['row']; v = row['source_parameters']
            key = (v['angle_deg'], v['tilt_sign'], v['onset_index']) if group == 'T3' else (
                v['sf_cycles_per_aperture'], v['modulation_depth'], v['bank_start_index'])
            require(row['truth'] not in pairs[key], 'Duplicate pair member')
            pairs[key][row['truth']] = sid
        units = []
        for key, pair in sorted(pairs.items()):
            unit = [pair[x] for x in ('LEFT', 'RIGHT') if x in pair]
            if len(unit) == 1:
                output['radon0'][group] += unit; orphans += unit
            else:
                units.append(unit)
        for unit in units:
            card = min(CARDS, key=lambda c: ((len(output[c][group]) + len(unit)) * seconds[c][group], CARDS.index(c)))
            output[card][group] += unit
    flattened = [sid for groups in output.values() for ids in groups.values() for sid in ids]
    require(len(flattened) == len(set(flattened)) and set(flattened) == {s for ids in remaining.values() for s in ids},
            'Reallocation is not an exact disjoint cover')
    return output, orphans


def prepare(root, request_path, output):
    root = Path(root); request = read(request_path); arc = archive.Archive(root); output = Path(output)
    require(output == root / REL and not output.exists(), 'Use one new reboot handoff directory; never overwrite')
    require(request['schema'] == NAME + '/handoff1-request-v1', 'Wrong handoff request')
    authorization(arc.ref(request['authorization']), request['manifest']['sha256'])
    require(ready(root, request)['status'] == 'READY_BOTH_FOURIER_COMPLETE', 'Both Fourier queues must finish first')
    postclose = request.get('postclose_admissions', {})
    require(set(postclose) <= {'radon0'}, 'Postclose supplementation is only for the exact Radon queue')
    queues = {c: queue_terminal(arc, request['queues'][c], ('COMPLETE',) if c != 'radon0' else
              ('PAUSED', 'PAUSED_BEFORE_SUCCESSOR', 'COMPLETE'), postclose_admission=postclose.get(c)) for c in CARDS}
    spec = arc.ref(request['manifest']); completed = {c: {g: [] for g in GROUPS} for c in CARDS}
    from . import qwen_remaining_t13_reboot as reboot
    require(len(spec.get('reboot_admissions', [])) == 1 and not spec.get('reallocations'),
            'One explicit reboot admission and no previous handoff are required')
    admission_ref = spec['reboot_admissions'][0]; admitted = arc.ref(admission_ref)
    reboot.verify_snapshot(admitted, arc)
    require(spec['retired_profiles'] == {
        'interrupted': [admitted['original_profiles']['T3']],
        'never_started': [admitted['original_profiles']['T1']]}, 'Original interrupted/unstarted owners differ')
    runtimes = {}; retained_refs = []; proofs = []; times = defaultdict(list); donor_controls = {}; source_claims = {}; donors = {}
    for record in admitted['accepted_records']:
        if not record['deep']:
            times['radon0', 'T3'].append(record['timing'])
    with archive.installed():
        _, _, owners, groups = archive.manifest_profiles(root, spec)
        archive.claim_ledger(arc, owners, groups)
        for key, identities in groups.items():
            card, group = key[-2:]; ordered = sorted((owners[i] for i in identities), key=lambda x: x['profile']['control_generation'])
            require(card in CARDS and group in GROUPS and all('handoff_entry' not in x['profile'] for x in ordered), 'Only one handoff is allowed')
            for owner in ordered:
                p = owner['profile']; out = arc.path(p['output']); ref = owner['reference']
                if card == 'radon0':
                    require('reboot_entry' in p and p['reboot_entry']['admission']['sha256'] == admission_ref['sha256'],
                            'Only the explicitly recovered clean Radon donor may be selected')
                    template_ref = p['reboot_entry']['original_profile']
                    template = arc.ref(template_ref); storage.science(template, arc.ref)
                    runtimes[card, group] = (template, template_ref)
                else:
                    require('reboot_entry' not in p, 'Fourier has an unexpected recovery lineage')
                    runtimes[card, group] = (p, ref)
                if out.exists():
                    records, proof = archive.audit_segment(root, ref['path']); proofs.append(proof)
                    retained_refs.append(portable_ref(root, ref, card))
                    completed[card][group] += [x['sample_id'] for x in records]
                    for t in journal(out / 'request-timings.jsonl') if records else []:
                        if not t['deep']: times[card, group].append(t)
                else:
                    require(card == 'radon0' and not out.with_suffix('.launch.json').exists()
                            and not out.with_suffix('.wait.json').exists(), 'Unclosed or missing source output')
            latest = ordered[-1]['profile']; claims = {f.stem for f in arc.path(latest['claim_root']).glob('*.json')}
            require(claims == set(completed[card][group]), 'Source group has an unresolved claim')
            source_claims[card, group] = sorted(claims)
            if card == 'radon0':
                donors[group] = ordered[-1]['reference']
                control = read(arc.path(latest['control_file']))
                require(control['paused'] and control['generation'] == latest['control_generation'], 'Pause both Radon controls before snapshot')
                donor_controls[group] = {'path': str(Path(latest['control_file']).relative_to(latest['workspace_root'])),
                    'generation': latest['control_generation'], 'snapshot': control,
                    'claim_root': str(Path(latest['claim_root']).relative_to(latest['workspace_root'])),
                    'claims': {f.stem: file_sha256(f) for f in arc.path(latest['claim_root']).glob('*.json')}}
    plan = read(root / 'plans' / NAME / 'plan.json')
    for c in CARDS[:2]:
        require(all(completed[c][g] == plan['cards'][c]['groups'][g] for g in GROUPS), 'Fourier source work is not complete')
    remaining = {}
    for group in GROUPS:
        expected = admitted['never_claimed'][group]; done = completed['radon0'][group]
        require(done == expected[:len(done)], 'Recovered Radon answers are not a clean prefix of never-claimed work')
        remaining[group] = expected[len(done):]
    if not any(remaining.values()):
        return {'status': 'NO_HANDOFF_RADON_ALREADY_COMPLETE', 'model_calls': 0}
    seconds = {c: {} for c in CARDS}; rate_proofs = {c: {} for c in CARDS}
    for card in CARDS:
        for group in GROUPS:
            measured_group = group if times[card, group] else next((g for g in GROUPS if times[card, g]), None)
            require(measured_group is not None, 'No existing ordinary timing for participant')
            selected = times[card, measured_group][-20:]
            value = statistics.mean(x['end_to_end_durable_seconds'] for x in selected)
            require(value > 0, 'Invalid ordinary request rate'); seconds[card][group] = value
            rate_proofs[card][group] = {'seconds_per_request': value, 'ordinary_n': len(selected),
                'sample_ids': [x['sample_id'] for x in selected], 'measured_task': measured_group,
                'proxy_for_unmeasured_task': measured_group != group, 'nested_timers_not_added': True}
    canonical = arc.ref(runtimes['radon0', 'T3'][0]['canonical_catalog'])['items']
    allocation, orphans = allocate(remaining, canonical, seconds)
    output.mkdir(parents=True)
    for source, name in ((arc.path(request['authorization']['path']), 'authorization.json'),
                         (arc.path(request['manifest']['path']), 'source-manifest.json')):
        (output / name).write_bytes(source.read_bytes())
    receipt = {'schema': NAME + '/handoff1-receipt-v1', 'status': 'SEALED_NEVER_CLAIMED',
        'batch': BATCH, 'original_reboot_admission': admission_ref,
        'retained_reboot_ids': completed['radon0'],
        'UNCERTAIN_ids_not_reissued': admitted['UNCERTAIN_ids'],
        'expected_unique_total': q.TOTAL,
        'maximum_accepted_without_uncertain_retry': q.TOTAL - len(admitted['UNCERTAIN_ids']),
        'maximum_new_model_calls': sum(map(len, remaining.values())),
        'implementation_sha256': file_sha256(__file__), 'plan_sha256': spec['plan_sha256'],
        'source_manifest_sha256': request['manifest']['sha256'], 'authorization': relative_ref(root, output / 'authorization.json'),
        'postclose_admissions': copy.deepcopy(postclose),
        'source_manifest': relative_ref(root, output / 'source-manifest.json'), 'retained_profiles': retained_refs,
        'retained_radon_ids': {g: (admitted['accepted_prefix_ids'] if g == 'T3' else []) + completed['radon0'][g]
                              for g in GROUPS}, 'remaining_ids': remaining, 'allocation': allocation,
        'ordinary_rates': rate_proofs, 'orphan_single_sides_retained_on_radon': orphans,
        'source_audits': proofs, 'source_queue_states': {c: {**queues[c][2],
            'relative_path': str(Path(queues[c][2]['path']).relative_to(root))} for c in CARDS},
        'donor_controls': donor_controls, 'donor_claim_ids': {g: source_claims['radon0', g] for g in GROUPS},
        'runtime_profiles': {c: {} for c in CARDS}, 'donor_profiles': {}, 'catalogs': {},
        'scope': 'One closed-boundary reassignment of never-claimed Radon IDs. No new qualification or model parameter change.'}
    require(sum(len(x) for x in receipt['retained_reboot_ids'].values()) + receipt['maximum_new_model_calls'] == 1547,
            'Recovery retained/remaining accounting differs from1547 never-claimed IDs')
    for card in CARDS:
        remote = q.ROOTS['radon' if card == 'radon0' else 'fourier']
        catalog = output / (card + '-sources.json')
        write_new(catalog, {'items': {sid: q.rebase(canonical[sid], remote)
            for group in GROUPS for sid in allocation[card][group]}})
        receipt['catalogs'][card] = relative_ref(root, catalog)
        for group in GROUPS:
            path = output / (card + '-' + group + '-runtime.json')
            path.write_bytes(arc.path(runtimes[card, group][1]['path']).read_bytes())
            receipt['runtime_profiles'][card][group] = relative_ref(root, path)
    for group in GROUPS:
        path = output / ('radon0-' + group + '-donor.json')
        path.write_bytes(arc.path(donors[group]['path']).read_bytes())
        receipt['donor_profiles'][group] = relative_ref(root, path)
    validate_partition(receipt, plan, arc.ref)
    write_new(output / 'receipt.json', receipt); drafts = {c: [] for c in CARDS}
    for card in CARDS:
        remote = q.ROOTS['radon' if card == 'radon0' else 'fourier']
        for group in GROUPS:
            ids = allocation[card][group]
            if not ids: continue
            p = copy.deepcopy(runtimes[card, group][0])
            for key in ('implementation', 'operational_wrapper', 'worker_token', 'media_files', 'storage_entry',
                        'reboot_entry', 'continuation_parent', 'continuation_wait', 'retained_prefix_length'):
                p.pop(key, None)
            worker = worker_name(card, group)
            p.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=0,
                output=str(Path(remote) / 'runs/reproducibility' / NAME / worker),
                claim_root=str(Path(remote) / 'runs/reproducibility' / NAME / ('claims-' + BATCH) / card / group),
                control_file=str(Path(remote) / REL / (card + '-' + group + '-control.json')),
                pause_file=str(Path(remote) / REL / (worker + '-PAUSE.json')),
                source_catalog=absolute_ref(remote, receipt['catalogs'][card]),
                handoff_entry={'implementation': {'path': str(Path(remote) / 'code/vlm-psychophysics/classic_precision' / Path(__file__).name), 'sha256': file_sha256(__file__)},
                    'receipt': absolute_ref(remote, relative_ref(root, output / 'receipt.json')),
                    'runtime_profile': absolute_ref(remote, receipt['runtime_profiles'][card][group])})
            science(p, arc.ref)
            control = output / (card + '-' + group + '-control.json'); write_new(control, {'generation': 0, 'paused': True})
            path = output / (worker + '-draft.json'); write_new(path, p); drafts[card].append(relative_ref(root, path))
    write_new(output / 'plan.json', {'schema': NAME + '/handoff1-plan-v1', 'batch': BATCH, 'status': 'PREPARED_NOT_LAUNCHED',
        'receipt': relative_ref(root, output / 'receipt.json'), 'drafts': drafts})
    return {'status': 'PREPARED_NOT_LAUNCHED', 'receipt': descriptor(output / 'receipt.json'),
        'counts': {c: {g: len(allocation[c][g]) for g in GROUPS} for c in CARDS}, 'model_calls': 0}


def donor_snapshot_unchanged(root, receipt):
    root = Path(root)
    from . import qwen_remaining_t13_reboot as reboot
    arc = archive.Archive(root)
    reboot.verify_snapshot(arc.ref(receipt['original_reboot_admission']), arc)
    for info in receipt['donor_controls'].values():
        path = root / info['claim_root']
        require({f.stem: file_sha256(f) for f in path.glob('*.json')} == info['claims'], 'Donor claim ledger changed since snapshot')
    state = receipt['source_queue_states']['radon0']
    require(file_sha256(root / state['relative_path']) == state['sha256'], 'Donor queue changed since closure')
    for card, ref in receipt.get('postclose_admissions', {}).items():
        require(card == 'radon0', 'Foreign postclose donor')
        admitted = arc.ref(ref)
        archive.verify_postclose_queue_admission(arc, admitted['original_queue'], ref)


def fence_donor(root, receipt_path):
    """Explicit post-closure fence; never invoked by prepare or readiness polling."""
    root = Path(root); receipt = read(receipt_path)
    require(str(root) == q.ROOTS['radon'] and Path(receipt_path) == root / REL / 'receipt.json'
            and receipt['schema'] == NAME + '/handoff1-receipt-v1' and receipt['status'] == 'SEALED_NEVER_CLAIMED'
            and receipt['implementation_sha256'] == file_sha256(__file__), 'Fence only the prepared original Radon handoff')
    authorization(verified(absolute_ref(root, receipt['authorization'])), receipt['source_manifest_sha256'])
    require(receipt['source_manifest']['sha256'] == receipt['source_manifest_sha256'], 'Source manifest binding differs')
    verified(absolute_ref(root, receipt['source_manifest']))
    validate_partition(receipt, read(root / 'plans' / NAME / 'plan.json'), archive.Archive(root).ref)
    for group in GROUPS:
        template = verified(absolute_ref(root, receipt['donor_profiles'][group]))
        science(template)
        info = receipt['donor_controls'][group]
        require(template['card'] == 'radon0' and template['resource_group'] == group
                and Path(template['control_file']) == root / info['path']
                and Path(template['claim_root']) == root / info['claim_root']
                and template['control_generation'] == info['generation'], 'Donor fence escaped its bound source profile')
    target = root / REL / 'fence.json'
    with q.lifecycle.controller_lock(root / REL / 'fence.lock'), ExitStack() as locks:
        for group in GROUPS:
            control_path = root / receipt['donor_controls'][group]['path']
            locks.enter_context(q.lifecycle.controller_lock(control_path.with_suffix('.lock')))
        donor_snapshot_unchanged(root, receipt)
        for info in receipt['donor_controls'].values():
            path = root / info['path']; control = read(path)
            require(control['paused'], 'Donor must remain paused')
            if control['generation'] == info['generation']:
                control.update(generation=info['generation'] + 1, reallocated_receipt_sha256=file_sha256(receipt_path))
                atomic_json(path, control)
            else:
                require(control['generation'] == info['generation'] + 1
                        and control.get('reallocated_receipt_sha256') == file_sha256(receipt_path), 'Foreign donor generation change')
        value = {'schema': NAME + '/handoff1-fence-v1', 'receipt_sha256': file_sha256(receipt_path),
            'status': 'DONOR_FENCED_PAUSED', 'controls': {g: descriptor(root / v['path']) for g, v in receipt['donor_controls'].items()}}
        if target.exists():
            require(read(target) == value, 'Existing fence differs')
        else:
            write_new(target, value)
    return {'status': value['status'], 'fence': descriptor(target), 'model_calls': 0}


def check_fence(p):
    root = Path(p['workspace_root']); receipt = verified(p['handoff_entry']['receipt'])
    fence_path = root / REL / 'fence.json'; fence = read(fence_path)
    require(fence['schema'] == NAME + '/handoff1-fence-v1' and fence['status'] == 'DONOR_FENCED_PAUSED'
            and fence['receipt_sha256'] == p['handoff_entry']['receipt']['sha256'], 'Missing/mismatched donor fence')
    if 'fence' in p['handoff_entry']:
        require(file_sha256(fence_path) == p['handoff_entry']['fence']['sha256'], 'Sealed donor fence changed')
    if p['card'] == 'radon0':
        donor_snapshot_unchanged(root, receipt)
        for info in receipt['donor_controls'].values():
            control = read(root / info['path'])
            require(control['paused'] and control['generation'] == info['generation'] + 1
                    and control.get('reallocated_receipt_sha256') == fence['receipt_sha256'], 'Donor fence is no longer active')
    else:
        for card in CARDS[:2]:
            state = receipt['source_queue_states'][card]; path = root / state['relative_path']
            require(file_sha256(path) == state['sha256'] and read(path)['status'] == 'COMPLETE', 'Fourier source queue is not complete')
    return descriptor(fence_path)


def source_admission(p):
    check_fence(p)
    if p['card'] == 'radon0':
        items = verified(p['source_catalog'])['items']
        storage.check_media_roots([m['path'] for sid in p['assigned_ids']
            for m in q.base.media_entries(p, sid, items[sid]) if m['type'] in ('image', 'video')], storage.ALLOWED_ROOT)


@contextmanager
def execution_context():
    native = subprocess.Popen
    def popen(argv, **kwargs):
        if isinstance(argv, (list, tuple)):
            argv = list(argv)
            if '-m' in argv:
                index = argv.index('-m') + 1
                if index < len(argv) and argv[index] == 'classic_precision.qwen_remaining_t13': argv[index] = MODULE
        return native(argv, **kwargs)
    with science_context():
        subprocess.Popen = popen
        try: yield
        finally: subprocess.Popen = native


def seal(draft, output):
    p = read(draft); science(p); source_admission(p)
    require(p['hostname'] == socket.gethostname(), 'Seal on the bound deployment host')
    p['handoff_entry']['fence'] = check_fence(p)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
        json.dump(p, stream); temp = stream.name
    try:
        with science_context(): return q.seal(temp, output)
    finally: Path(temp).unlink()


def run(profile):
    p = read(profile); science(p); source_admission(p)
    with science_context(): return q.run(profile)


def supervise(profile):
    science(read(profile))
    with execution_context(): return q.supervise(profile)


def activate(queue_path):
    for ref in read(queue_path)['profiles']:
        p = verified(ref); science(p); source_admission(p)
    with science_context(): return q.activate(queue_path)


def queue(queue_path):
    for ref in read(queue_path)['profiles']:
        p = verified(ref); science(p); require('handoff_entry' in p, 'Nonhandoff profile in queue')
    with execution_context(): return q.queue(queue_path)


def closed(profile):
    with science_context(): return q.closed(profile)


def prepare_resume(profile, output):
    p = read(profile); science(p)
    with q.lifecycle.controller_lock(str(Path(p['control_file']).with_suffix('.lock'))):
        p, result, raw = closed(profile); control = read(p['control_file'])
        require(control['paused'] and control['generation'] == p['control_generation'], 'Pause the unbranched handoff worker first')
        ids = q.lifecycle.remaining(p, result, raw); require(ids, 'Nothing remains')
        value = copy.deepcopy(p)
        for key in ('implementation', 'operational_wrapper', 'worker_token', 'media_files'): value.pop(key, None)
        generation = control['generation'] + 1; worker = worker_name(p['card'], p['resource_group'], generation)
        value.update(worker=worker, assigned_ids=ids, maximum_model_calls=len(ids), control_generation=generation,
            output=str(Path(p['output']).parent / worker), pause_file=str(Path(p['workspace_root']) / REL / (worker + '-PAUSE.json')),
            continuation_parent=descriptor(profile), continuation_wait=descriptor(Path(p['output']).with_suffix('.wait.json')),
            retained_prefix_length=len(raw))
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as stream:
            json.dump(value, stream); temp = stream.name
        try: answer = seal(temp, output)
        finally: Path(temp).unlink()
        target = Path(output).with_name(worker + '-queue.json'); q.write_queue(target, [descriptor(output)], p['workspace_root'])
        control.update(generation=generation, paused=True, continuation_profile=descriptor(output)); atomic_json(p['control_file'], control)
    return {**answer, 'status': 'CONTINUATION_PREPARED_PAUSED', 'queue': descriptor(target), 'model_calls': 0}


def prepare_resume_queue(queue_path, output):
    previous = read(queue_path); state = read(previous['state_path']); profiles = []
    require(state['status'] in ('PAUSED', 'PAUSED_BEFORE_SUCCESSOR') and not Path(output).exists(), 'Only a closed paused queue can continue')
    with q.lifecycle.controller_lock(previous['controller_lock']):
        for ref in previous['profiles']:
            p = verified(ref); science(p)
            require('handoff_entry' in p and read(p['control_file'])['paused'], 'Pause both handoff resource groups first')
            if Path(p['output']).exists():
                _, terminal, _ = closed(ref['path'])
                if terminal['status'] == 'COMPLETE': continue
                target = Path(ref['path']).with_name(worker_name(p['card'], p['resource_group'], p['control_generation'] + 1) + '-profile.json')
                result = prepare_resume(ref['path'], target); profiles.extend(verified(result['queue'])['profiles'])
            else:
                q.fresh(p); profiles.append(ref)
        require(profiles, 'Nothing remains'); q.write_queue(output, profiles, read(profiles[0]['path'])['workspace_root'])
    return {'status': 'CONTINUATION_QUEUE_PREPARED_PAUSED', 'queue': descriptor(output), 'model_calls': 0}


def build_queue(root, card, output):
    root = Path(root); plan = read(root / REL / 'plan.json'); profiles = []
    for ref in plan['drafts'][card]:
        path = root / ref['path'].replace('-draft.json', '-profile.json')
        p = read(path); science(p); require(p['card'] == card, 'Wrong deployment card'); profiles.append(descriptor(path))
    require(profiles, 'No work for this card'); q.write_queue(output, profiles, root)
    return {'status': 'SEALED_PAUSED_NOT_LAUNCHED', 'queue': descriptor(output)}


def seal_card(root, card):
    root = Path(root); require(card in CARDS, 'Unknown deployment card')
    plan = read(root / REL / 'plan.json'); profiles = []
    for ref in plan['drafts'][card]:
        draft = root / ref['path']; require(file_sha256(draft) == ref['sha256'], 'Prepared draft changed')
        target = Path(str(draft).replace('-draft.json', '-profile.json'))
        seal(draft, target); profiles.append(descriptor(target))
    target = root / REL / (card + '-' + BATCH + '-queue.json')
    if profiles:
        q.write_queue(target, profiles, root)
    value = {'status': 'SEALED_PAUSED_NOT_LAUNCHED' if profiles else 'NO_WORK', 'card': card,
        'profiles': profiles, 'queue': descriptor(target) if profiles else None, 'model_calls': 0}
    write_new(root / REL / (card + '-seal-receipt.json'), value)
    return value


def build_manifest(root, receipt_path, output, segments=None):
    root = Path(root); receipt = read(receipt_path); plan = read(root / REL / 'plan.json')
    refs = receipt['retained_profiles'][:]
    for card in CARDS:
        for ref in plan['drafts'][card]:
            path = root / ref['path'].replace('-draft.json', '-profile.json'); p = read(path)
            science(p, archive.Archive(root).ref); refs.append(portable_ref(root, descriptor(path), p['card']))
    if segments:
        for ref in read(segments)['profiles']:
            p = archive.Archive(root).ref(ref)
            science(p, archive.Archive(root).ref); require(p.get('continuation_parent'), 'Extra profile is not a continuation')
            refs.append(portable_ref(root, ref, p['card']))
    source = archive.Archive(root).ref(absolute_ref(root, receipt['source_manifest']))
    value = {k: copy.deepcopy(v) for k, v in source.items() if k not in ('profiles', 'reallocations')}
    value.update(schema=NAME + '/segments-v1', plan_sha256=receipt['plan_sha256'],
                 profiles=refs, reallocations=[portable_ref(root, descriptor(receipt_path))])
    write_new(output, value)
    return {'status': 'MANIFEST_PREPARED_NOT_INSTALLED', 'manifest': descriptor(output)}


def snapshot_files(root, request_path):
    """Return a bounded transfer list; no SSH, scan or source-media copying."""
    root = Path(root); arc = archive.Archive(root); request = read(request_path)
    output = {'fourier': set(), 'radon': set()}
    for card, ref in request['queues'].items():
        queue = arc.ref(ref); host = 'radon' if card == 'radon0' else 'fourier'
        output[host].update((str(arc.path(ref['path']).relative_to(root)), str(arc.path(queue['state_path']).relative_to(root))))
        if card == 'radon0':
            stem = arc.path(ref['path']).stem; parent = arc.path(queue['state_path']).parent
            output[host].update(str((parent / (stem + suffix)).relative_to(root))
                                for suffix in ('.wait.json', '.log', '.launch.json'))
    for card, ref in request.get('postclose_admissions', {}).items():
        require(card == 'radon0', 'Foreign supplementary queue admission')
        admitted = archive.verify_postclose_queue_admission(arc, request['queues'][card], ref)
        output['radon'].add(str(arc.path(ref['path']).relative_to(root)))
        for bound in [*admitted['bindings'].values(), *admitted['claims'].values()]:
            output['radon'].add(str(arc.path(bound['path']).relative_to(root)))
    manifest = arc.ref(request['manifest'])
    for ref in manifest['profiles']:
        p = arc.ref(ref); host = 'radon' if p['card'] == 'radon0' else 'fourier'
        for path in (ref['path'], p['output'], p['claim_root'], p['control_file'],
                     str(Path(p['output']).with_suffix('.wait.json')), str(Path(p['output']).with_suffix('.launch.json'))):
            output[host].add(str(arc.path(path).relative_to(root)))
    for ref in manifest.get('reboot_admissions', []):
        admitted = arc.ref(ref)
        refs = [ref, admitted['closure_evidence'], *admitted['original_profiles'].values(),
                *admitted['original_evidence'].values(), *admitted['original_claims'].values()]
        for bound in refs:
            output['radon'].add(str(arc.path(bound['path']).relative_to(root)))
        for original in admitted['original_profiles'].values():
            p = arc.ref(original)
            for path in (p['output'], p['claim_root'], p['control_file']):
                output['radon'].add(str(arc.path(path).relative_to(root)))
    return {'status': 'EXPLICIT_SOURCE_SNAPSHOT_PATHS', 'relative_paths_by_host': {k: sorted(v) for k, v in output.items()},
        'scope': 'Only listed source metadata/output/claim directories and admitted original reboot evidence. An unstarted Radon successor has no output/wait/launch; the interrupted original has no invented wait/result.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('ready', 'snapshot-files', 'prepare', 'fence-donor', 'seal', 'seal-card', 'build-queue',
        'build-manifest', 'activate', 'queue', 'run', 'supervise', 'pause', 'prepare-resume', 'prepare-resume-queue', 'audit-closed'))
    for key in ('root', 'request', 'receipt', 'output', 'draft', 'profile', 'plan', 'card', 'control', 'segments'): parser.add_argument('--' + key)
    a = parser.parse_args()
    if a.command == 'ready': result = ready(a.root, a.request)
    elif a.command == 'snapshot-files': result = snapshot_files(a.root, a.request)
    elif a.command == 'prepare': result = prepare(a.root, a.request, a.output)
    elif a.command == 'fence-donor': result = fence_donor(a.root, a.receipt)
    elif a.command == 'seal': result = seal(a.draft, a.output)
    elif a.command == 'seal-card': result = seal_card(a.root, a.card)
    elif a.command == 'build-queue': result = build_queue(a.root, a.card, a.output)
    elif a.command == 'build-manifest': result = build_manifest(a.root, a.receipt, a.output, a.segments)
    elif a.command == 'activate': result = activate(a.plan)
    elif a.command == 'queue': result = queue(a.plan)
    elif a.command == 'run': result = run(a.profile)
    elif a.command == 'supervise': result = supervise(a.profile)
    elif a.command == 'pause': result = q.lifecycle.pause(a.control)
    elif a.command == 'prepare-resume': result = prepare_resume(a.profile, a.output)
    elif a.command == 'prepare-resume-queue': result = prepare_resume_queue(a.plan, a.output)
    else:
        _, terminal, raw = closed(a.profile); result = {'status': 'CLOSED_PASS', 'worker_status': terminal['status'], 'completed': len(raw)}
    print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
