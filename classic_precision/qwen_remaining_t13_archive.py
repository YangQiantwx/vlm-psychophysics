"""Read-only archive adapter for shared-claim Qwen T1/T3 segments.

The frozen scientific worker audit is reused unchanged. Claims are scoped to
their profile. A separately authorized handoff changes deployment accounting,
while preserving the complete original source-ID universe. No model is run.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import json
from pathlib import Path

from . import official_x_results as shared
from . import qwen_remaining_t13_results as frozen
from .refinement_worker import read, require
from .storage import atomic_json, digest, file_sha256

NAME = 'qwen-x-remaining-t13-20261030'
SSD_RUN = '/home/q/cvpr2027_runs_ssd/' + NAME
_original_audit_segment = frozen.audit_segment


class Archive(shared.Archive):
    """Also map already-recorded Radon SSD receipt paths into a local archive."""
    def path(self, value):
        text = str(value)
        mac = '/Users/qianyang/Desktop/cvpr2027_benchmark'
        if text.startswith(mac + '/'):
            return self.root / text[len(mac) + 1:]
        if text == SSD_RUN or text.startswith(SSD_RUN + '/'):
            relative = text[len(SSD_RUN):].lstrip('/')
            return self.root / 'runs/reproducibility' / NAME / relative
        return super().path(value)


def descriptor(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def adapter_identity():
    from . import qwen_remaining_t13_storage as storage
    from . import qwen_remaining_t13_handoff as handoff
    from . import qwen_remaining_t13_reboot as reboot
    return {'schema': NAME + '/archive-adapter-v1', 'adapter': descriptor(__file__),
        'frozen_audit': descriptor(frozen.__file__), 'shared_scientific_audit': descriptor(shared.__file__),
        'storage_profile_validator': descriptor(storage.__file__),
        'handoff_profile_validator': descriptor(handoff.__file__),
        'reboot_prefix_validator': descriptor(reboot.__file__),
        'change': 'Claims are scoped by profile_sha256. Explicit reboot admissions preserve original interrupted PASS records and unresolved claims without a synthetic wait. Deployment accounting never changes the expected6576-ID universe.',
        'scientific_model_calls': 0, 'original_records_modified': False}


def science(profile, resolve=None):
    resolve = resolve or Archive(profile['workspace_root']).ref
    if 'handoff_entry' in profile:
        from .qwen_remaining_t13_handoff import science as validate
    elif 'reboot_entry' in profile:
        from .qwen_remaining_t13_reboot import science as validate
    else:
        from .qwen_remaining_t13_storage import science as validate
    return validate(profile, resolve)


def lineage_key(profile):
    if 'handoff_entry' in profile:
        from .qwen_remaining_t13_handoff import lineage_key as key
        return key(profile)
    return ('reboot1' if 'reboot_entry' in profile else 'baseline', profile['card'], profile['resource_group'])


def manifest_profiles(root, manifest):
    """An explicit manifest selects owners; no latest-file or owner discovery."""
    arc = Archive(root)
    spec = manifest if isinstance(manifest, dict) else read(manifest)
    plan_path = Path(root) / 'plans' / NAME / 'plan.json'
    require(spec['schema'] == NAME + '/segments-v1'
            and spec['plan_sha256'] == file_sha256(plan_path), 'Foreign segment manifest')
    owners = {}; generations = set(); groups = defaultdict(list)
    for ref in spec['profiles']:
        p = arc.ref(ref); science(p, arc.ref)
        require(p['phase'] == 'full' and p['experiment_plan']['sha256'] == spec['plan_sha256'],
                'Foreign/nonfull claim owner')
        lineage = lineage_key(p); key = (*lineage, p['control_generation']); identity = digest(p)
        require(identity not in owners and key not in generations, 'Duplicate profile/generation in manifest')
        generations.add(key); owners[identity] = {'profile': p, 'reference': ref}
        groups[lineage].append(identity)
    require(owners, 'Empty segment manifest')
    for identities in groups.values():
        ordered = sorted(identities, key=lambda key: owners[key]['profile']['control_generation'])
        for generation, identity in enumerate(ordered):
            p = owners[identity]['profile']
            require(p['control_generation'] == generation, 'Missing ancestor segment')
            if generation:
                require(arc.ref(p['continuation_parent']) == owners[ordered[generation-1]]['profile'],
                        'Segment ancestor differs from manifest')
    return arc, spec, owners, groups


def claim_ledger(arc, owners, groups):
    """Reject undeclared owners, misplaced/duplicate claims and known uncertainty."""
    seen = set(); counts = Counter(); by_group = {}
    for group, identities in groups.items():
        profiles = [owners[x]['profile'] for x in identities]
        roots = {str(arc.path(p['claim_root'])) for p in profiles}
        require(len(roots) == 1, 'Segments do not share one canonical group claim root')
        root = Path(next(iter(roots))); selected = set()
        for identity in identities:
            p = owners[identity]['profile']; out = arc.path(p['output'])
            if (out / 'result.json').exists():
                require(not read(out / 'result.json').get('UNCERTAIN_ids'), 'Uncertain owner in group ledger')
            require(not (out / 'failure.json').exists(), 'Failed owner in group ledger')
        for path in sorted(root.glob('*.json')):
            require(path.is_file() and not path.is_symlink(), 'Claim must be a regular original file')
            claim = read(path); sid = path.stem; identity = claim.get('profile_sha256')
            require(identity in identities, 'Unknown/foreign profile claim in group: ' + str(path))
            p = owners[identity]['profile']
            require(claim.get('sample_id') == sid and sid in p['assigned_ids']
                    and claim.get('worker_token') == p['worker_token'] and claim.get('attempt') == 1,
                    'Claim content/owner/allocation differs')
            require(sid not in seen, 'Duplicate claim sample ID across groups')
            seen.add(sid); selected.add(sid); counts[identity] += 1
        by_group['/'.join(group)] = {'claim_n': len(selected), 'ids': sorted(selected)}
    return {'total_claims': len(seen), 'by_profile_digest': dict(counts), 'groups': by_group}


class _OwnedClaim:
    def __init__(self, path, identity):
        self.path, self.identity = path, identity

    def __fspath__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    def exists(self):
        return self.path.exists() and read(self.path).get('profile_sha256') == self.identity


class _ClaimRoot:
    def __init__(self, path, identity):
        self.path, self.identity = path, identity

    def __fspath__(self):
        return str(self.path)

    def __str__(self):
        return str(self.path)

    def __truediv__(self, filename):
        return _OwnedClaim(self.path / filename, self.identity)

    def glob(self, pattern):
        # The final group union must always see every physical claim.
        return self.path.glob(pattern)


def audit_segment(root, profile_path):
    """Delegate every original check; change only claim-existence ownership."""
    arc = Archive(root); path = arc.path(profile_path); p = read(path)
    identity = digest(p); claim_root = arc.path(p['claim_root'])

    class SegmentArchive(Archive):
        def path(self, value):
            real = super().path(value)
            return _ClaimRoot(real, identity) if real == claim_root else real

    from . import qwen_remaining_t13 as runner
    previous = frozen.Archive, runner.science
    frozen.Archive = SegmentArchive
    runner.science = science
    try:
        return _original_audit_segment(root, profile_path)
    finally:
        frozen.Archive, runner.science = previous


@contextmanager
def installed():
    from . import qwen_remaining_t13 as runner
    previous = frozen.audit_segment, frozen.Archive, runner.science
    frozen.audit_segment, frozen.Archive = audit_segment, Archive
    runner.science = science
    try:
        yield
    finally:
        frozen.audit_segment, frozen.Archive, runner.science = previous


def reboot_admissions(root, spec, owners):
    """Reconcile the admitted original228 records against their unchanged raw.

    Qualification/runtime/input proofs were already audited by the SHA-bound
    reboot admission. Here only the affected receipt, immutable snapshot and
    derived canonical records are reconciled; original waits remain absent.
    """
    from . import qwen_remaining_t13_reboot as reboot
    arc = Archive(root); refs = spec.get('reboot_admissions', [])
    require(len(refs) == 1, 'Exactly one explicit reboot admission is supported')
    receipt = arc.ref(refs[0]); reboot.verify_snapshot(receipt, arc)
    old = {g: arc.ref(ref) for g, ref in receipt['original_profiles'].items()}
    require(set(old) == {'T3', 'T1'} and receipt['old_boot_id'] == reboot.OLD_BOOT
            and receipt['new_boot_id'] == reboot.NEW_BOOT
            and receipt['original_wait'] is None and receipt['original_returncode'] is None
            and receipt['real_original_wait_observed'] is False
            and receipt['old_owned_identities_terminated_by_boot_change'] is True
            and receipt['original_physical_lease_free_on_new_boot'] is True,
            'Reboot admission does not preserve the unknown original wait')
    require(all(p['card'] == 'radon0' and p['phase'] == 'full'
                and p['experiment_plan']['sha256'] == spec['plan_sha256']
                for p in old.values()), 'Retired original profiles belong to another experiment')
    retired = spec.get('retired_profiles', {})
    require(set(retired) == {'interrupted', 'never_started'}
            and len(retired['interrupted']) == len(retired['never_started']) == 1
            and retired['interrupted'][0]['sha256'] == receipt['original_profiles']['T3']['sha256']
            and retired['never_started'][0]['sha256'] == receipt['original_profiles']['T1']['sha256'],
            'Both retired original Radon profiles must be explicitly bound')
    require(not {digest(x) for x in old.values()} & set(owners), 'Interrupted/never-started original cannot enter clean-wait owners')
    ids = receipt['accepted_prefix_ids']; uncertain = receipt['UNCERTAIN_ids']; p = old['T3']
    require(len(ids) == 228 and ids == p['assigned_ids'][:228] and uncertain == [reboot.UNCERTAIN]
            and p['assigned_ids'][228] == reboot.UNCERTAIN
            and receipt['never_claimed']['T3'] == p['assigned_ids'][229:]
            and receipt['never_claimed']['T1'] == old['T1']['assigned_ids']
            and {g: len(v) for g, v in receipt['never_claimed'].items()} == {'T3': 509, 'T1': 1038},
            'Original accepted/quarantined/never-claimed partition differs')
    out = arc.path(p['output']); catalog = arc.ref(p['source_catalog'])['items']
    raws = reboot.unique(reboot.journal(out / 'raw.jsonl')[0])
    times = reboot.unique(reboot.journal(out / 'request-timings.jsonl')[0])
    selected = receipt['accepted_records']
    require([r['sample_id'] for r in selected] == ids and len({r['sample_id'] for r in selected}) == 228,
            'Admitted canonical records duplicate or reorder the original prefix')
    for record in selected:
        sid = record['sample_id']; raw = raws[sid]; item = catalog[sid]
        expected = {'sample_id': sid, 'task': raw['task'], 'truth': raw['truth'], 'answer': raw['raw_answer'].strip(),
            'raw_answer': raw['raw_answer'], 'correct': raw['correct'], 'eligible': True,
            'parameters': item['row']['source_parameters'], 'source_media': [list(x) for x in shared.receipt_media(raw)],
            'output_token_ids': raw['output_token_ids'], 'finish_reason': raw['finish_reason'], 'stop_reason': raw['stop_reason'],
            'prompt_token_sha256': digest(raw['actual_prompt_token_ids']), 'raw_digest': digest(raw),
            'profile_sha256': digest(p), 'worker': p['worker'], 'deep': raw['deep'], 'timing': times[sid],
            'admission': 'REBOOT_INTERRUPTED_DURABLE_TECHNICAL_PASS_PREFIX'}
        require(record == expected, 'Admitted canonical record differs from original raw: ' + sid)
    return {'reference': refs[0], 'receipt': receipt, 'records': selected, 'original_profiles': old,
            'quarantined': [{'sample_id': sid, 'original_claim': receipt['original_claims'][sid],
                             'original_profile': receipt['original_profiles']['T3'], 'retry_authorized': False} for sid in uncertain],
            'proof': {'admission': refs[0], 'accepted': 228, 'original_wait': None, 'original_returncode': None,
                      'original_wait_observed': False, 'closure': receipt['closure_evidence'],
                      'status': 'REBOOT_INTERRUPTED_PREFIX_ACCEPTED', 'quarantined_ids': uncertain}}


def analyze(root, manifest, profile=None):
    with installed():
        arc, spec, owners, groups = manifest_profiles(root, manifest)
        before = claim_ledger(arc, owners, groups)
        identity = adapter_identity()
        recovery = reboot_admissions(root, spec, owners) if spec.get('reboot_admissions') else None
        if profile:
            path = arc.path(profile); p = read(path)
            if digest(p) in owners:
                records, evidence = audit_segment(root, path)
                result = {'status': 'CLOSED_PROFILE_PASS', 'accepted': len(records),
                    'profile': descriptor(path), 'original_scientific_evidence': evidence,
                    'scope': 'Selected closed segment only; this does not certify completion of its group.'}
            else:
                require(recovery and digest(p) in {digest(x) for x in recovery['original_profiles'].values()},
                        'Requested profile is not explicitly selected or retired')
                interrupted = p['resource_group'] == 'T3'
                result = {'status': 'REBOOT_PREFIX_ACCEPTED' if interrupted else 'RETIRED_NEVER_STARTED',
                          'accepted': len(recovery['records']) if interrupted else 0,
                          'profile': descriptor(path), 'reboot_admission': recovery['reference'],
                          'original_wait_observed': False, 'original_returncode': None,
                          'UNCERTAIN_ids': recovery['receipt']['UNCERTAIN_ids'] if interrupted else []}
        else:
            # All original full counts, segment lineage, raw/PASS/claim/token/
            # source checks and the unfiltered group claim union still execute.
            if recovery:
                result = reboot_full(root, manifest, arc, spec, owners, groups, recovery)
            else:
                result = reallocated_full(root, manifest, arc, spec, owners, groups) if spec.get('reallocations') else frozen.full(root, manifest)
    after = claim_ledger(arc, owners, groups)
    require(before == after, 'Claim ledger changed during read-only audit')
    result.update(archive_adapter=identity, selected_group_claim_ledger=after)
    if recovery:
        from .qwen_remaining_t13_reboot import verify_snapshot
        verify_snapshot(recovery['receipt'], arc)
        result['quarantined_original_claims'] = recovery['quarantined']
    if not profile:
        summary_path = Path(root) / 'runs/reproducibility' / NAME / 'analysis/summary.json'
        summary = read(summary_path); summary['archive_adapter'] = identity
        summary['selected_group_claim_ledger'] = after
        if recovery: summary['quarantined_original_claims'] = recovery['quarantined']
        atomic_json(summary_path, summary); result['summary'] = descriptor(summary_path)
    return result


def reboot_full(root, manifest, arc, spec, owners, groups, recovery):
    """Audit the authorized6575 cover while keeping the expected6576 universe."""
    from . import qwen_remaining_t13_handoff as handoff
    from .qwen_remaining_t13 import COUNTS, TOTAL
    root = Path(root); plan_path = root / 'plans' / NAME / 'plan.json'; plan = read(plan_path)
    require(spec.get('expected_unique_total', TOTAL) == TOTAL
            and spec.get('maximum_accepted_without_uncertain_retry', TOTAL - 1) == TOTAL - 1,
            'Manifest changes the expected or currently authorized denominator')
    quarantine = recovery['receipt']['UNCERTAIN_ids']
    require(spec.get('unresolved_ids', quarantine) == quarantine, 'Manifest changes the quarantined ID')
    reallocations = spec.get('reallocations', []); transfer = None
    if reallocations:
        require(len(reallocations) == 1, 'Only the explicitly bound reboot handoff is supported')
        transfer = arc.ref(reallocations[0])
        require(transfer['original_reboot_admission']['sha256'] == recovery['reference']['sha256']
                and transfer['plan_sha256'] == spec['plan_sha256'],
                'Handoff binds another reboot admission')
        handoff.validate_partition(transfer, plan, arc.ref)
        handoff.donor_snapshot_unchanged(root, transfer)
        retained = {digest(arc.ref(ref)) for ref in transfer['retained_profiles']}
        baseline = {i for i, owner in owners.items() if 'handoff_entry' not in owner['profile']}
        require(baseline == retained, 'Retained pre-handoff profile selection differs')
    records = [{**row, 'deployment_card': 'radon0', 'logical_source_card': 'radon0'} for row in recovery['records']]
    proofs = [recovery['proof']]; seen = set()
    for lineage, identities in groups.items():
        kind, card, group = lineage; combined = []
        require(kind in ('baseline', 'reboot1', handoff.BATCH), 'Unknown deployment lineage')
        if kind == 'baseline':
            require(card in ('fourier0', 'fourier1'), 'Original Radon is retired, not a clean-wait owner')
            expected = plan['cards'][card]['groups'][group]
        elif kind == 'reboot1':
            require(card == 'radon0', 'Reboot1 must remain on the original qualified card')
            expected = transfer['retained_reboot_ids'][group] if transfer else recovery['receipt']['never_claimed'][group]
        else:
            require(transfer is not None, 'Unbound transferred work')
            expected = transfer['allocation'][card][group]
        ordered = sorted(identities, key=lambda i: owners[i]['profile']['control_generation'])
        for index, identity in enumerate(ordered):
            p = owners[identity]['profile']; ref = owners[identity]['reference']
            if kind == 'reboot1':
                require(p['reboot_entry']['admission']['sha256'] == recovery['reference']['sha256'], 'Foreign reboot lineage')
            elif kind == handoff.BATCH:
                require(p['handoff_entry']['receipt']['sha256'] == reallocations[0]['sha256'], 'Foreign transfer lineage')
            rows, proof = audit_segment(root, ref['path']); proofs.append(proof)
            terminal = read(arc.path(p['output']) / 'result.json')
            allowed = ('PAUSED', 'PAUSED_BEFORE_LOAD') if index < len(ordered) - 1 else (
                ('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD') if kind == 'reboot1' and transfer else ('COMPLETE',))
            require(terminal['status'] in allowed, 'Selected recovery/handoff segment is not appropriately closed')
            combined += [{**row, 'deployment_card': card,
                          'logical_source_card': 'radon0' if kind != 'baseline' else card} for row in rows]
        require([r['sample_id'] for r in combined] == expected, 'Lineage does not reconstruct its exact allocated prefix')
        require({f.stem for f in arc.path(p['claim_root']).glob('*.json')} == set(expected), 'Lineage has foreign/unresolved claims')
        records += combined; seen.add(lineage)
    for card in ('fourier0', 'fourier1'):
        for group in ('T3', 'T1'):
            require(('baseline', card, group) in seen, 'Missing original completed Fourier lineage')
    for group in ('T3', 'T1'):
        if not transfer:
            require(('reboot1', 'radon0', group) in seen, 'Missing complete never-claimed Radon group')
        elif transfer['retained_reboot_ids'][group]:
            require(('reboot1', 'radon0', group) in seen, 'Missing retained reboot donor prefix')
        elif ('reboot1', 'radon0', group) not in seen:
            # An unlaunched successor has no worker wait. The explicit donor
            # snapshot, source profile and empty original namespace retire it.
            ref = handoff.absolute_ref(root, transfer['donor_profiles'][group]); p = arc.ref(ref)
            require(p.get('reboot_entry') and p['reboot_entry']['admission']['sha256'] == recovery['reference']['sha256'],
                    'Missing never-started recovery donor binding')
            out = arc.path(p['output'])
            require(not out.exists() and not out.with_suffix('.launch.json').exists()
                    and not out.with_suffix('.wait.json').exists()
                    and not list(arc.path(p['claim_root']).glob('*.json')), 'Retired empty donor was actually attempted')
            proofs.append({'profile': ref, 'accepted': 0, 'status': 'RETIRED_NEVER_STARTED_HANDOFF_DONOR',
                           'original_wait_observed': False, 'original_returncode': None})
    if transfer:
        for card, tasks in transfer['allocation'].items():
            for group, ids in tasks.items():
                require(not ids or (handoff.BATCH, card, group) in seen, 'Missing allocated handoff work')
    expected_ids = {s for card in plan['cards'].values() for ids in card['groups'].values() for s in ids}
    observed = [r['sample_id'] for r in records]
    require(len(expected_ids) == TOTAL and len(observed) == len(set(observed)) == TOTAL - len(quarantine)
            and set(observed) == expected_ids - set(quarantine), 'Authorized never-claimed cover is missing, duplicated or foreign')
    counts = Counter(r['task'] for r in records); expected_counts = dict(COUNTS); expected_counts['T3'] -= len(quarantine)
    require(counts == expected_counts, 'Observed task denominator differs from the explicit unresolved source')
    output = root / 'runs/reproducibility' / NAME / 'analysis'; output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'canonical-records.json', records); shared.write_csv(output / 'predictions.csv', records)
    summary = {'schema': NAME + '/results-v1', 'status': 'PARTIAL_CLASSIC_QWEN_T13_X_UNRESOLVED_CLAIM',
        'completion_status': 'PARTIAL', 'technical_status': 'COMPLETE_FOR_AUTHORIZED_NEVER_CLAIMED_SCOPE',
        'unique_videos': len(records), 'expected_unique_videos': TOTAL, 'missing_ids': quarantine,
        'UNCERTAIN_ids': quarantine, 'quarantined_claims_scored_as_wrong': False,
        'tasks': {t: shared.basic(r for r in records if r['task'] == t) for t in COUNTS},
        'expected_task_counts': COUNTS, 'workers': proofs, 'segment_manifest': descriptor(manifest),
        'plan': descriptor(plan_path), 'reboot_admissions': spec['reboot_admissions'],
        'retired_profiles': spec['retired_profiles'], 'reallocation': reallocations[0] if transfer else None,
        'qualification_outputs_included': False, 'natural_outputs_included': False,
        'new_model_calls': 0, 'automatic_uncertain_retry': False}
    summary['outputs'] = {p.name: descriptor(p) for p in output.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(output / 'summary.json', summary)
    return {k: summary[k] for k in ('status', 'completion_status', 'unique_videos', 'expected_unique_videos', 'missing_ids', 'tasks')}


def reallocated_full(root, manifest, arc, spec, owners, groups):
    """Retain original science audits; replace only the retired card allocation."""
    from . import qwen_remaining_t13_handoff as handoff
    from .qwen_remaining_t13 import COUNTS, TOTAL
    require(len(spec['reallocations']) == 1, 'Only the single authorized handoff is supported')
    receipt = arc.ref(spec['reallocations'][0]); plan = read(Path(root) / 'plans' / NAME / 'plan.json')
    handoff.validate_partition(receipt, plan)
    handoff.donor_snapshot_unchanged(root, receipt)
    retained = {digest(arc.ref(ref)) for ref in receipt['retained_profiles']}
    baseline = {identity for identity, owner in owners.items() if 'handoff_entry' not in owner['profile']}
    require(baseline == retained, 'Retained source profile selection changed')
    records = []; proofs = []; seen_lineages = set()
    for lineage, identities in groups.items():
        kind, card, group = lineage; combined = []
        ordered = sorted(identities, key=lambda i: owners[i]['profile']['control_generation'])
        for index, identity in enumerate(ordered):
            p = owners[identity]['profile']; ref = owners[identity]['reference']
            if kind == 'handoff1':
                require(p['handoff_entry']['receipt']['sha256'] == spec['reallocations'][0]['sha256'], 'Foreign handoff receipt')
            rows, proof = audit_segment(root, ref['path']); proofs.append(proof)
            terminal = read(arc.path(p['output']) / 'result.json')
            if index < len(ordered) - 1:
                require(terminal['status'] in ('PAUSED', 'PAUSED_BEFORE_LOAD'), 'Nonfinal segment was not cleanly paused')
            elif kind == 'handoff1' or card != 'radon0':
                require(terminal['status'] == 'COMPLETE', 'Final assigned work is not complete')
            else:
                require(terminal['status'] in ('COMPLETE', 'PAUSED', 'PAUSED_BEFORE_LOAD'), 'Retired Radon prefix is not closed')
            combined += [{**row, 'deployment_card': card,
                'logical_source_card': 'radon0' if kind == 'handoff1' else card} for row in rows]
        expected = receipt['allocation'][card][group] if kind == 'handoff1' else (
            receipt['retained_radon_ids'][group] if card == 'radon0' else plan['cards'][card]['groups'][group])
        require([row['sample_id'] for row in combined] == expected, 'Lineage union/order differs from finite allocation')
        require({f.stem for f in arc.path(p['claim_root']).glob('*.json')} == set(expected), 'Lineage has foreign/unresolved claims')
        records += combined; seen_lineages.add(lineage)
    for card in ('fourier0', 'fourier1'):
        for group in ('T3', 'T1'):
            require(('baseline', card, group) in seen_lineages, 'Missing completed Fourier source lineage')
    for card, tasks in receipt['allocation'].items():
        for group, ids in tasks.items():
            require(not ids or ('handoff1', card, group) in seen_lineages, 'Missing reassigned work')
    expected_ids = {sid for item in plan['cards'].values() for ids in item['groups'].values() for sid in ids}
    require(len(records) == len({r['sample_id'] for r in records}) == TOTAL
            and {r['sample_id'] for r in records} == expected_ids
            and Counter(r['task'] for r in records) == COUNTS, 'The original6576 unique source universe changed')
    output = Path(root) / 'runs/reproducibility' / NAME / 'analysis'; output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / 'canonical-records.json', records); shared.write_csv(output / 'predictions.csv', records)
    summary = {'schema': NAME + '/results-v1', 'status': 'COMPLETE_CLASSIC_QWEN_T13_X', 'unique_videos': TOTAL,
        'tasks': {t: shared.basic(r for r in records if r['task'] == t) for t in COUNTS}, 'workers': proofs,
        'segment_manifest': descriptor(manifest), 'plan': descriptor(Path(root) / 'plans' / NAME / 'plan.json'),
        'reallocation': spec['reallocations'][0], 'qualification_outputs_included': False, 'natural_outputs_included': False}
    summary['outputs'] = {p.name: descriptor(p) for p in output.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(output / 'summary.json', summary)
    return {'status': summary['status'], 'unique_videos': TOTAL, 'tasks': summary['tasks'], 'summary': descriptor(output / 'summary.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--profile')
    args = parser.parse_args()
    print(json.dumps(analyze(args.root, args.manifest, args.profile)), flush=True)


if __name__ == '__main__':
    main()
