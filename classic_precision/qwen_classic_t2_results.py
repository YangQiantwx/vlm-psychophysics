"""Read-only monitor and terminal audit for the finite classic Qwen X T2 scan."""
from __future__ import annotations
import argparse
import json
import itertools
import socket
from collections import defaultdict
from pathlib import Path
from .qwen_classic_t2 import NAME,science
from .natural_t2_results import status as shared_status
from .official_x_results import Archive,audit_worker,basic,write_csv,timing_tables,common_raw_checks
from .refinement_worker import read,require,descriptor,verified
from .storage import atomic_json,digest

def startup(root):
    from .qwen_x_crosscard import validate_runtime
    root=Path(root);plans=root/'plans'/NAME;results=[]
    def first(path):
        if not path.exists():return []
        rows=[]
        with path.open() as f:
            for line in itertools.islice(f,2):
                if line.endswith('\n'):rows.append(json.loads(line))
        return rows
    for path in sorted(plans.glob('*-segment0-profile.json')):
        p=read(path)
        if p['hostname']!=socket.gethostname():continue
        out=Path(p['output']);raw=first(out/'raw.jsonl')
        admissions={x['sample_id']:x for x in first(out/'input-admissions.jsonl')}
        selectors={x['sample_id']:x for x in first(out/'custom-selector-admissions.jsonl')}
        deep=[r for r in raw if r['deep']]
        item={'card':p['card'],'first_raw_n':len(raw),'profile':descriptor(path),'status':'FIRST_INPUT_PENDING'}
        if (out/'progress.json').exists():
            progress=read(out/'progress.json');item['completed']=progress['completed'];item['expected']=progress['expected']
            from .cosmos_x_full_status import timing_window
            item['recent_ordinary']=timing_window(out/'request-timings.jsonl')
        if (out/'failure.json').exists():item['failure']=read(out/'failure.json')
        if deep and all(r['sample_id'] in admissions and r['sample_id'] in selectors for r in deep):
            for value in read(out/'runtime.json'):validate_runtime(value)
            catalog=verified(p['source_catalog'])['items'];evidence=[]
            for i,r in enumerate(raw):common_raw_checks(p,catalog[r['sample_id']],r,i)
            for r in deep:
                a=admissions[r['sample_id']];q=selectors[r['sample_id']];proof=a['proof']
                require(a['raw_digest']==q['raw_digest']==digest(r),'Input evidence identity differs')
                require(proof['status']=='PASS_OFFICIAL_RAW_GPU_INGRESS' and proof['CPU_to_GPU_patch_bytes_equal']
                        and proof['source_frame_indices']==list(range(180)),'Actual GPU input differs')
                require(q['actual_request_calls'] and all(n>0 for n in q['actual_request_calls']),'QSA not actually exercised')
                evidence.append({'sample_id':r['sample_id'],'raw_digest':digest(r),'frames':180,
                                 'CPU_GPU_patch_equal':True,'actual_QSA_calls':q['actual_request_calls']})
            item.update(status='FIRST_INPUT_ACTUAL_GPU_PASS',deep_n=len(deep),inputs=evidence,runtime=descriptor(out/'runtime.json'))
        results.append(item)
    require(results,'No local sealed profiles')
    host='radon' if results[0]['card']=='radon0' else 'fourier'
    atomic_json(root/'runs/reproducibility'/NAME/('startup-'+host+'-latest.json'),results)
    return results

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
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--status',action='store_true');p.add_argument('--startup',action='store_true');a=p.parse_args()
    result=shared_status(a.root,NAME) if a.status else startup(a.root) if a.startup else full(a.root)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
