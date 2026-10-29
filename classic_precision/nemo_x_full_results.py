"""Read-only three-card Nemo X qualification. No model requests or retries."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter
import socket
import time
from datetime import datetime, timezone
from .official_x_results import Archive, audit_worker, descriptor, read, require, journal, write_csv, common_raw_checks
from .storage import atomic_json, digest
from .nemo_x_full import NAME, validate_runtime, science
from . import cosmos_x_full_status as monitor


def qualification(root):
    root=Path(root);plans=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME;arc=Archive(root)
    plan=read(plans/'plan.json');ids=plan['qualification_ids'];require(len(set(ids))==12,'Wrong preselected panel')
    all_records={};evidence=[];rows=[];source_payloads={}
    for card in ('fourier0','fourier1','radon0'):
        queue=read(run/(card+'-qualification-queue-state.json'))
        require(queue['status']=='COMPLETE' and len(queue['workers'])==2,'Qualification queue did not close')
        for cold in (1,2):
            path=plans/(card+'-qualification-cold'+str(cold)+'-profile.json')
            p=read(path);science(p);require(p['assigned_ids']==ids,'Qualification IDs/order changed')
            records,proof=audit_worker(arc,{'profile':descriptor(path),'phase':'qualification','variant':'NEMO_X_NO_AUTOTUNE','replicate':cold})
            for v in proof['runtime']:validate_runtime(v)
            for entry in queue['workers']:
                require(entry['real_supervisor_wait'] and entry['returncode']==0,'Outer supervisor not closed')
            out=arc.path(p['output']);deep=journal(out/'deep-observations.jsonl')
            require(Counter(x['task'] for x in records if x['deep'])=={'T1':1,'T2':1,'T3':1},'First-task actual GPU capture missing')
            for row in deep:
                observed=sum(z['tensor']['shape'][0] for d in row['observations'] for z in d['normalized'] if z['tensor']['shape'][1:]==[3,512,768])
                require(observed==180,'Actual GPU did not receive all180 original-size frames')
            all_records[(card,cold)]={x['sample_id']:x for x in records};evidence.append(proof)
            source_payloads[(card,cold)]={x['sample_id']:[(v['decoded_sha256'],tuple(v['per_frame_sha256'])) for v in x['native_reader']] for x in journal(out/'raw.jsonl') if x['deep']}
    reference=all_records[('radon0',1)];fields=['raw_answer','output_token_ids','finish_reason','stop_reason','prompt_token_sha256','source_media']
    for (card,cold),records in all_records.items():
        for sid in ids:
            x=records[sid];same={k:x[k]==reference[sid][k] for k in fields}
            rows.append({'card':card,'cold':cold,'sample_id':sid,'task':x['task'],'answer':x['answer'],'correct':x['correct'],**same,'all_equal':all(same.values())})
    require(all(v==source_payloads[('radon0',1)] for v in source_payloads.values()),'First-task decoded pixels differ across cards/colds')
    stable=all(x['all_equal'] for x in rows)
    result={'status':'PASS_ALL_THREE_CARDS' if stable else 'HOLD_REPRODUCIBILITY_DIFFERENCE','calls':len(rows),'unique_panel_ids':12,
        'same_card_full_output_matches':{card:sum(all(all_records[(card,1)][sid][k]==all_records[(card,2)][sid][k] for k in fields) for sid in ids) for card in ('fourier0','fourier1','radon0')},
        'matches_against_R0_cold1':{card+'-cold'+str(cold):sum(row['all_equal'] for row in rows if row['card']==card and row['cold']==cold) for card,cold in all_records},
        'actual_GPU_admissions':18,'first_task_decoded_pixel_hashes_equal':True,'limits':'Fixed12 inputs, two cold starts per card; not a universal or full9456 repeatability proof. Qualification is separate from formal scores.',
        'automatic_retry':False,'workers':evidence,'differences':[r for r in rows if not r['all_equal']],
        'plan':descriptor(plans/'plan.json'),'input_profile_sha256':[e['profile']['sha256'] for e in evidence]}
    atomic_json(run/'qualification-summary.json',result);write_csv(run/'qualification-paired.csv',rows)
    atomic_json(run/'qualification-records.json',[{'card':card,'cold':cold,**r} for (card,cold),records in all_records.items() for r in records.values()])
    return result


def status(root):
    root=Path(root);plans=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME
    local=[(p,read(p)) for p in plans.glob('*-profile.json') if read(p)['hostname']==socket.gethostname()]
    full=[(p,j) for p,j in local if j['phase']=='full'];workers=[];active=False
    if not full:
        workers=[monitor.inspect_profile(p) for p,j in local]
        active=any(not x['wait'] and x.get('status')!='NOT_STARTED' for x in workers)
    else:
        allocation=read(plans/'full-allocation.json')
        for card in sorted({j['card'] for p,j in full}):
            segments=sorted([(p,j) for p,j in full if j['card']==card],key=lambda x:x[1]['control_generation'])
            allowed=set(allocation['cards'][card]['assigned_ids']);passed=[];receipts=[]
            for path,p in segments:
                out=Path(p['output']);events,partial=monitor.journal(out/'events.jsonl')
                done=[x['sample_id'] for x in events if x['event']=='TECHNICAL_PASS'];passed+=done
                wait=read(out.with_suffix('.wait.json')) if out.with_suffix('.wait.json').exists() else None
                result=read(out/'result.json') if (out/'result.json').exists() else None
                receipts.append({'worker':p['worker'],'profile':descriptor(path),'completed':len(done),'wait':wait,'result':result,'failure':read(out/'failure.json') if (out/'failure.json').exists() else None,'partial_event_tail':partial})
            latest=segments[-1][1];out=Path(latest['output']);claims={p.stem for p in Path(latest['claim_root']).glob('*.json')}
            issues=[]
            if len(passed)!=len(set(passed)):issues.append('DUPLICATE_PASS_ACROSS_SEGMENTS')
            if not (set(passed)|claims)<=allowed:issues.append('FOREIGN_ID')
            if not set(passed)<=claims:issues.append('PASS_WITHOUT_CLAIM')
            active=active or not receipts[-1]['wait']
            workers.append({'card':card,'expected':len(allowed),'completed':len(passed),'claimed':len(claims),'claimed_without_pass':sorted(claims-set(passed)),'unattempted':len(allowed-claims),'control':read(latest['control_file']),'issues':issues,'segments':receipts,'recent_ordinary':monitor.timing_window(out/'request-timings.jsonl')})
    return {'status':'LIVE_SNAPSHOT_NOT_FINAL_AUDIT','host':socket.gethostname(),'observed_UTC':datetime.now(timezone.utc).isoformat(),'observed_monotonic_seconds':time.monotonic(),'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'workers':workers,'resources':monitor.resources(root) if active else {'skipped':'all local workers closed'},'model_calls':0,'mutations':0}


def startup(root, profile):
    arc=Archive(root);path=Path(profile);p=read(path);science(p);out=arc.path(p['output'])
    ident=read(out/'identity.json');require(ident['profile']==p and ident['profile_sha256']==digest(p),'Startup identity differs')
    runtime=read(out/'runtime.json')
    for r in runtime:validate_runtime(r);require(r['actual_kv_capacity']>=p['minimum_kv_capacity'],'Actual KV too small')
    public=read(out/'public-settings.json')
    require(all(public['LLM_kwargs'][k]==v for k,v in p['engine_kwargs'].items()),'Actual engine setting differs')
    versions=read(out/'versions.json');require(all(versions[k]==v for k,v in p['expected_packages'].items() if k in versions),'Worker version differs')
    catalog=arc.ref(p['source_catalog']);raw={r['sample_id']:r for r in monitor.journal(out/'raw.jsonl')[0]}
    passed={r['sample_id']:r for r in monitor.journal(out/'events.jsonl')[0] if r['event']=='TECHNICAL_PASS'}
    deeps={r['sample_id']:r for r in monitor.journal(out/'deep-observations.jsonl')[0]};ids=p['assigned_ids'][:3];proof=[]
    for ordinal,sid in enumerate(ids):
        x=raw[sid];common_raw_checks(p,catalog['items'][sid],x,ordinal)
        require(passed[sid]['raw_digest']==digest(x),'No technical PASS binding')
        claim=read(arc.path(p['claim_root'])/(sid+'.json'));require(claim['profile_sha256']==digest(p) and claim['worker_token']==p['worker_token'],'Claim differs')
        if x['deep']:
            d=deeps[sid];require(d['raw_digest']==digest(x),'Actual observation not bound')
            n=sum(z['tensor']['shape'][0] for item in d['observations'] for z in item['normalized'] if z['tensor']['shape'][1:]==[3,512,768])
            require(n==180,'Not all180 original-size GPU frames')
        proof.append({'sample_id':sid,'task':x['task'],'deep':x['deep'],'raw_digest':digest(x),'technical_pass':True})
    if p['control_generation']==0:require({x['task'] for x in proof}=={'T1','T2','T3'} and all(x['deep'] for x in proof),'Missing first-task captures')
    result={'status':'PASS_STARTUP_ONLY_NOT_FULL_COMPLETION','profile':descriptor(path),'actual_runtime':runtime,'inputs':proof,'no_new_model_calls_for_audit':True}
    atomic_json(out/'startup-acceptance.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--status',action='store_true');a=p.parse_args();r=status(a.root) if a.status else qualification(a.root)
    import json
    print(json.dumps(r if a.status else {k:v for k,v in r.items() if k not in ('workers','input_profile_sha256')},ensure_ascii=False))
