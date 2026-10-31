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
        if text == SSD_RUN or text.startswith(SSD_RUN + '/'):
            relative = text[len(SSD_RUN):].lstrip('/')
            return self.root / 'runs/reproducibility' / NAME / relative
        return super().path(value)


def descriptor(path):
    return {'path': str(path), 'sha256': file_sha256(path)}


def adapter_identity():
    from . import qwen_remaining_t13_storage as storage
    from . import qwen_remaining_t13_handoff as handoff
    return {'schema': NAME + '/archive-adapter-v1', 'adapter': descriptor(__file__),
        'frozen_audit': descriptor(frozen.__file__), 'shared_scientific_audit': descriptor(shared.__file__),
        'storage_profile_validator': descriptor(storage.__file__),
        'handoff_profile_validator': descriptor(handoff.__file__),
        'change': 'Claim exists() is scoped by profile_sha256. An explicit handoff receipt changes only deployment/lineage accounting; the original6576-ID universe and frozen scientific worker checks remain mandatory.',
        'scientific_model_calls': 0, 'original_records_modified': False}


def manifest_profiles(root, manifest):
    """An explicit manifest selects owners; no latest-file or owner discovery."""
    from .qwen_remaining_t13 import science
    from .qwen_remaining_t13_handoff import lineage_key
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

    previous = frozen.Archive
    frozen.Archive = SegmentArchive
    try:
        return _original_audit_segment(root, profile_path)
    finally:
        frozen.Archive = previous


@contextmanager
def installed():
    from .qwen_remaining_t13_handoff import science_context
    previous = frozen.audit_segment, frozen.Archive
    frozen.audit_segment, frozen.Archive = audit_segment, Archive
    try:
        with science_context():
            yield
    finally:
        frozen.audit_segment, frozen.Archive = previous


def analyze(root, manifest, profile=None):
    with installed():
        arc, spec, owners, groups = manifest_profiles(root, manifest)
        before = claim_ledger(arc, owners, groups)
        identity = adapter_identity()
        if profile:
            path = arc.path(profile); p = read(path)
            require(digest(p) in owners, 'Requested profile is not explicitly selected')
            records, evidence = audit_segment(root, path)
            result = {'status': 'CLOSED_PROFILE_PASS', 'accepted': len(records),
                'profile': descriptor(path), 'original_scientific_evidence': evidence,
                'scope': 'Selected closed segment only; this does not certify completion of its group.'}
        else:
            # All original full counts, segment lineage, raw/PASS/claim/token/
            # source checks and the unfiltered group claim union still execute.
            result = reallocated_full(root, manifest, arc, spec, owners, groups) if spec.get('reallocations') else frozen.full(root, manifest)
    after = claim_ledger(arc, owners, groups)
    require(before == after, 'Claim ledger changed during read-only audit')
    result.update(archive_adapter=identity, selected_group_claim_ledger=after)
    if not profile:
        summary_path = Path(root) / 'runs/reproducibility' / NAME / 'analysis/summary.json'
        summary = read(summary_path); summary['archive_adapter'] = identity
        summary['selected_group_claim_ledger'] = after
        atomic_json(summary_path, summary); result['summary'] = descriptor(summary_path)
    return result


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
