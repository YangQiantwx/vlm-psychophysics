"""Read-only monitor and terminal audit for the finite classic Qwen X T2 scan."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path
from .qwen_classic_t2 import NAME,science
from .natural_t2_results import status as shared_status
from .official_x_results import Archive,audit_worker,basic,write_csv,timing_tables
from .refinement_worker import read,require,descriptor
from .storage import atomic_json

def full(root):
    root=Path(root);plans=root/'plans'/NAME;out=root/'runs/reproducibility'/NAME
    plan=read(plans/'plan.json');arc=Archive(root);records=[];proofs=[]
    for card,allocation in plan['cards'].items():
        paths=sorted(plans.glob(card+'-segment*-profile.json'),key=lambda x:read(x)['control_generation'])
        require(paths,'Missing card '+card);card_rows=[]
        for n,path in enumerate(paths):
            p=read(path);science(p,arc.ref);require(p['control_generation']==n,'Segment gap')
            terminal=read(arc.path(p['output'])/'result.json')
            spec={'profile':descriptor(path),'phase':'classic_T2','variant':'X_QSA_NO_FLASHINFER_AUTOTUNE'}
            if n<len(paths)-1:
                require(terminal['status']=='PAUSED','Nonfinal segment not paused')
                successor=read(paths[n+1])
                require(arc.ref(successor['continuation_parent'])==p
                        and arc.ref(successor['continuation_wait'])==read(arc.path(p['output']).with_suffix('.wait.json'))
                        and successor['assigned_ids']==terminal['unattempted_ids'],'Successor does not bind closed prefix')
                spec['closed_prefix_only']=True
            else:require(terminal['status']=='COMPLETE','Final segment incomplete')
            rows,evidence=audit_worker(arc,spec);card_rows+=rows;proofs.append(evidence)
        require([r['sample_id'] for r in card_rows]==allocation['assigned_ids'],'Card union/order differs')
        require({x.stem for x in arc.path(p['claim_root']).glob('*.json')}=={r['sample_id'] for r in card_rows},'Claim union differs')
        records+=card_rows
    expected={s for c in plan['cards'].values() for s in c['assigned_ids']}
    require(len(records)==len({r['sample_id'] for r in records})==2880 and {r['sample_id'] for r in records}==expected,'Full source union differs')
    groups=defaultdict(list)
    for r in records:groups[r['parameters']['background_Y'],r['parameters']['delta_Y']].append(r)
    require(len(groups)==60 and {len(v) for v in groups.values()}=={48},'Condition coverage differs')
    cells=[{'background_Y':b,'delta_Y':d,**basic(v),'left_correct':sum(r['correct'] for r in v if r['truth']=='LEFT'),
            'right_correct':sum(r['correct'] for r in v if r['truth']=='RIGHT')} for (b,d),v in sorted(groups.items())]
    summary={'status':'COMPLETE_CLASSIC_QWEN_T2_X','model':'Qwen','overall':basic(records),
             'new_scientific_calls':2880,'qualification_outputs_included':False,'cells':cells,
             'worker_evidence':proofs,'plan':descriptor(plans/'plan.json')}
    atomic_json(out/'canonical-records.json',records);atomic_json(out/'summary.json',summary)
    write_csv(out/'cell-results.csv',cells);write_csv(out/'timing-summary.csv',timing_tables(records))
    write_csv(out/'answers.csv',[{k:r[k] for k in ('sample_id','truth','answer','correct','worker')}|r['parameters'] for r in records])
    return {'status':summary['status'],'n':2880,'correct':summary['overall']['correct'],'summary':descriptor(out/'summary.json')}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--status',action='store_true');a=p.parse_args()
    print(json.dumps(shared_status(a.root,NAME) if a.status else full(a.root)),flush=True)

if __name__=='__main__':main()
