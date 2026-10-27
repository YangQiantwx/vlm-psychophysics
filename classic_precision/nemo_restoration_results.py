"""Read-only closure/identity audit of the five finite Nemo restoration workers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .nemo_canonical_restore import AUTHORIZATION, MUTABLE, relocate, load_bound, preserved, records
from .nemo_kernel_replay import validate_inventory
from .refinement_worker import require, read, descriptor
from .storage import atomic_json, digest, file_sha256

NAME = 'reader-route-smoke-20261026'


def profile_contract(root, path, auth, intent):
    p = load_bound(root, path); restoration = p['restoration']
    require(load_bound(root, restoration['authorization']) == auth, 'Profile authorization differs')
    parent, old, claims = preserved(root, intent['blocks'][restoration['block_index']])
    require({k:v for k,v in p.items() if k not in MUTABLE} ==
            {k:v for k,v in relocate(parent, Path(p['output']).parents[3]).items() if k not in MUTABLE},
            'Restoration changed scientific fields')
    require({sid:ref['sha256'] for sid,ref in claims.items()} == restoration['preserved_claim_sha256'],
            'Retained claim changed')
    allocation = load_bound(root, restoration['sealed_allocation'])
    require(p['assigned_ids'] == allocation['assignments'][p['worker']], 'Restored allocation changed')
    return p, parent, {r['sample_id']:r for r in old}


def audit(root, partial=False):
    root=Path(root);plan=root/'plans'/NAME
    auth_path=plan/'nemo-restore1-authorization.json';auth=read(auth_path)
    require(auth['human_message']==AUTHORIZATION and auth['maximum_additional_calls']==1268
            and not auth['automatic_retry'], 'Explicit1268 authority missing')
    intent=load_bound(root,auth['intent']);reports=[];replacements={};cards={};seen=set();total=0
    variants=[];formal=[];history={}
    import csv
    for row in csv.DictReader((plan/'historical.csv').open()):
        if row['model']=='nemo':history[row['sample_id']]=row
    replay=[]
    for host in ['fourier','radon']:
        manifest_path=plan/f'nemo-restore1-{host}-manifest.json'
        require(manifest_path.exists() or partial,'Missing host restoration manifest')
        if not manifest_path.exists():continue
        m=read(manifest_path)
        require(load_bound(root,m['authorization'])==auth,'Manifest authorization differs')
        for card,qp in m['queues'].items():
            qlocal=Path(relocate(qp,root));q=read(qlocal)
            cards[card]={'queue_path':qp,'queue_status':q['status'],'queue_sha256':file_sha256(qlocal)}
            require(partial or q['status']=='COMPLETE','Restoration queue not complete')
        for worker,ref in m['profiles'].items():
            p,parent,old=profile_contract(root,ref['profile'],auth,intent)
            out=Path(relocate(p['output'],root));smoke='arm' in p
            raw_path=out/'raw.jsonl' if smoke else out/'nemo/outputs.jsonl'
            rows=records(raw_path) if raw_path.exists() else []
            events_path=out/'events.jsonl' if smoke else out/'nemo/events.jsonl'
            events=records(events_path) if events_path.exists() else []
            passes={e['sample_id']:e for e in events if e.get('event')=='TECHNICAL_PASS'}
            if rows:
                validate_inventory(read(out/'restored-kernels.json'))
                generation=read(out/'restored-generation.json')
                require(generation['temperature']==0 and generation['top_p']==1
                        and generation['seed']==0 and generation['thinking'] is False
                        and generation['ignore_eos'] is False,'Actual generation policy differs')
            require(len(rows)==len({r['sample_id']for r in rows}),'Duplicate restored ID')
            claimed={f.stem:f for f in Path(relocate(p['claim_root'],root)).glob('*.json')}
            for row in rows:
                sid=row['sample_id'];key=('smoke',p['arm'],sid) if smoke else ('formal',sid)
                require(key not in seen and sid in p['assigned_ids'],'Unexpected/duplicate restored tuple');seen.add(key)
                require(row['profile_sha256']==digest(p),'Raw/profile identity differs')
                require(sid in claimed and read(claimed[sid])['profile_sha256']==digest(p),'Raw lacks bound restoration claim')
                proof=passes.get(sid);require(proof and
                    proof.get('raw_digest',proof.get('raw_sha256'))==digest(row),'Raw lacks exact technical PASS')
                oldrow=old[sid]
                require(row['truth']==oldrow['truth'],'GT changed')
                require((row['source']['video_sha256'] if smoke else row['source_video_sha256'])==
                        (oldrow['source']['video_sha256'] if smoke else oldrow['source_video_sha256']), 'Source changed')
                require(row['finish_reason']=='stop' and row['raw_answer'].strip() in ['LEFT','RIGHT'],'Invalid completion')
                if smoke and p['arm']=='A':
                    h=history[sid]
                    replay.append({'sample_id':sid,'task':row['task'],
                        'answer_matches':row['raw_answer'].strip()==h['answer'].strip(),
                        'full_tokens_match':row['output_token_ids']==json.loads(h['output_token_ids']),
                        'finish_matches':row['finish_reason']==h['finish_reason'],
                        'stop_matches':row['stop_reason']==json.loads(h['stop_reason'])})
                total+=1
            wait=out.with_suffix('.wait.json');w=read(wait) if wait.exists() else None
            if not partial:
                require(len(rows)==len(p['assigned_ids']) and set(passes)==set(p['assigned_ids'])
                        and set(claimed)&set(p['assigned_ids'])==set(p['assigned_ids']), 'Restored raw/PASS/ID ledger incomplete')
                require(w and w['profile_sha256']==digest(p) and w['returncode']==0 and w['real_wait_observed']
                        and w['owned_engines_exited'] and w['original_physical_lease_released']
                        and w['terminal_status']=='COMPLETE' and w['completed_ids']==p['assigned_ids'], 'Restoration not safely closed')
            report={'worker':worker,'completed':len(rows),'expected':len(p['assigned_ids']),
                    'profile':ref['profile'],'actual_wait':w,'output':p['output']}
            reports.append(report)
            if smoke:
                replacements['nemo-radon0-'+p['arm']]=worker
                original_out=Path(relocate(parent['output'],root))
                variants.append({'worker':parent['worker'],'raw':descriptor(original_out/'raw.jsonl'),
                                 'actual_wait':read(original_out.with_suffix('.wait.json')),
                                 'reason':'Preserved FlashInfer runtime variant; excluded from canonical smoke'})
            else:formal.append(report)
    if not partial:require(total==1268 and len(reports)==5 and len(formal)==2 and len(replacements)==3,
                           'Five workers/1268 canonical restoration calls required')
    result={'schema':'nemo-canonical-restoration-audit-v1','status':'PARTIAL' if partial else 'COMPLETE_PASS',
            'maximum_additional_calls':1268,'technical_pass':total,'workers':reports,
            'logical_smoke_replacements':replacements,'card_completion':cards,
            'retained_smoke_runtime_variants':variants,'canonical_formal_workers':formal,
            'Nemo_A_historical_replay':{'paired':len(replay),
                'answer_changes':sum(not x['answer_matches']for x in replay),
                'full_token_changes':sum(not x['full_tokens_match']for x in replay),
                'per_ID':replay},'authorization':descriptor(auth_path),
            'analysis_source_sha256':file_sha256(__file__),'automatic_retry':False}
    dest=root/'runs/reproducibility/nemo-canonical-restoration-20261027'
    dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/('partial-audit.json' if partial else 'canonical-audit.json'),result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--partial',action='store_true');a=p.parse_args();r=audit(a.root,a.partial)
    print(json.dumps({k:v for k,v in r.items() if k in ['status','technical_pass','maximum_additional_calls']},sort_keys=True))


if __name__=='__main__':main()
