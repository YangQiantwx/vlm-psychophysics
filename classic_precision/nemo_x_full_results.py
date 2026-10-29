"""Read-only three-card Nemo X qualification. No model requests or retries."""
from __future__ import annotations
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import socket
import time
from datetime import datetime, timezone
from .official_x_results import Archive, audit_worker, descriptor, read, require, journal, write_csv, common_raw_checks, basic, timing_tables
from .storage import atomic_json, digest, file_sha256
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


def full(root):
    """Audit all immutable segments, retaining the original clean paused prefix."""
    from .cosmos_x_full_results import cell_key, pair_stats, EXPECTED
    from zoneinfo import ZoneInfo
    root=Path(root);arc=Archive(root);plans=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME
    plan=read(plans/'plan.json');allocation=read(plans/'full-allocation.json')
    require(plan['maximum_full_calls']==9456 and plan['primary_counts']==EXPECTED
            and allocation['maximum_new_full_calls']==9456 and allocation['task_counts']==EXPECTED,'Full scope changed')
    gate=run/'qualification-summary.json'
    require(file_sha256(gate)==allocation['qualification_gate_sha256']
            and read(gate)['status']=='PASS_ALL_THREE_CARDS','Qualification gate changed')
    records=[];proofs=[];claim_scopes=defaultdict(set);allocated=[]
    for card in ('fourier0','fourier1','radon0'):
        assigned=allocation['cards'][card]['assigned_ids'];allocated+=assigned
        prior=read(allocation['cards'][card]['prior_allocation_profile'])
        require(prior['assigned_ids']==assigned,'Allocation changed from Cosmos X')
        segments=sorted(plans.glob(card+'-full-segment*-profile.json'),key=lambda p:read(p)['control_generation'])
        require(segments and read(segments[0])['assigned_ids']==assigned,'Initial allocation differs')
        card_records=[]
        for ordinal,path in enumerate(segments):
            p=read(path);science(p);require(p['control_generation']==ordinal,'Segment generation gap')
            require(p['source_catalog']==allocation['cards'][card]['source_catalog'],'Catalog/allocation differs')
            require(arc.ref(p['qualification_gate'])==read(gate),'Profile qualification binding differs')
            out=arc.path(p['output']);terminal=read(out/'result.json')
            require(not (out/'failure.json').exists(),'Failed attempt cannot be admitted')
            spec={'profile':descriptor(path),'phase':'full','variant':'X'}
            if ordinal<len(segments)-1:
                spec['paused_prefix_successor']=descriptor(segments[ordinal+1])
                require(terminal['status']=='PAUSED','Nonfinal segment is not preserved PAUSED')
            else:require(terminal['status']=='COMPLETE','Last segment not COMPLETE')
            queue_path=run/(p['worker']+'-queue-state.json');queue=read(queue_path)
            require(queue['status']==terminal['status'] and len(queue['workers'])==1,'Queue not closed')
            qw=queue['workers'][0]
            require(qw['real_supervisor_wait'] and qw['returncode']==0 and qw['terminal_status']==terminal['status']
                    and arc.ref(qw['profile'])==p,'Outer queue real wait/profile differs')
            rows,proof=audit_worker(arc,spec)
            for v in proof['runtime']:validate_runtime(v)
            deep=journal(out/'deep-observations.jsonl')
            for d in deep:
                n=sum(z['tensor']['shape'][0] for item in d['observations'] for z in item['normalized'] if z['tensor']['shape'][1:]==[3,512,768])
                require(n==180,'Actual GPU deep input is not all180 original-size frames')
            proof['queue']=descriptor(queue_path);proof['terminal_status']=terminal['status']
            proof['paused_prefix_successor']=spec.get('paused_prefix_successor')
            proofs.append(proof);card_records+=rows
            claim_scopes[arc.path(p['claim_root'])].update(r['sample_id'] for r in rows)
            print(json.dumps({'worker':p['worker'],'accepted':len(rows),'terminal':terminal['status']}),flush=True)
        require([r['sample_id'] for r in card_records]==assigned,'Card union reordered, missing or repeated')
        records+=card_records
    for folder,ids in claim_scopes.items():
        require({p.stem for p in folder.glob('*.json')}==ids,'Claim without accepted answer or foreign claim')
    require(len(allocated)==len(set(allocated))==len(records)==9456
            and {r['sample_id'] for r in records}==set(allocated),'Full union incomplete/duplicate')
    require(Counter(r['task'] for r in records)==EXPECTED
            and all(r['eligible'] and r['include_in_main_accuracy_curve'] for r in records),'Task/primary scope differs')
    cells=defaultdict(list);strata=defaultdict(list)
    for r in records:
        r['cell']=list(cell_key(r));cells[tuple(r['cell'])].append(r)
        strata[(r['task'],'truth',r['truth'])].append(r)
        for key in ('phase_degrees','onset_index','tilt_sign'):
            if key in r['parameters']:strata[(r['task'],key,r['parameters'][key])].append(r)
    require(Counter(k[0] for k in cells)=={'T1':80,'T2':60,'T3':29},'Configuration grid incomplete')
    for key,rows in cells.items():
        n=96 if key[0]=='T3' and key[1]!=90 else 48
        require(len(rows)==n and Counter(r['truth'] for r in rows)=={'LEFT':n//2,'RIGHT':n//2},'Cell unbalanced/incomplete')
    a_dir=root/'runs/reproducibility/current-nvfp4';a_path=a_dir/'video-predictions.csv';a_summary=read(a_dir/'summary.json')
    require(a_summary['status']=='COMPLETE' and file_sha256(a_path)==a_summary['video_csv_sha256'],'Historical A export changed')
    with a_path.open() as f:a_rows=[r for r in csv.DictReader(f) if r['model']=='nemo']
    a_lookup={r['sample_id']:r for r in a_rows};require(len(a_rows)==len(a_lookup)==9360,'Historical Nemo ledger differs')
    paired=[]
    for x in records:
        a=a_lookup.get(x['sample_id'])
        if a is None:continue
        require(a['protocol']=='controlled-rgb16' and a['task']==x['task'] and a['truth']==x['truth']
                and a['source_video_sha256'] in [sha for kind,sha in x['source_media'] if kind=='video'],'Historical A identity differs')
        ac=a['answer']==a['truth'];require(ac==bool(int(a['correct'])),'Historical A score differs')
        paired.append({'sample_id':x['sample_id'],'task':x['task'],'cell':x['cell'],'truth':x['truth'],'A':a['answer'],'X':x['answer'],
                       'A_correct':ac,'X_correct':x['correct'],'output_tokens_changed':json.loads(a['output_token_ids'])!=x['output_token_ids'],
                       'A_raw_sha256':a['raw_sha256'],'X_raw_digest':x['raw_digest'],'source_video_sha256':a['source_video_sha256']})
    cosdir=root/'runs/reproducibility/cosmos-x-full-20261028/analysis';cs=read(cosdir/'summary.json');cp=cosdir/'canonical-records.json'
    require(cs['status']=='COMPLETE_PASS' and file_sha256(cp)==cs['outputs'][cp.name]['sha256'],'Cosmos X accepted records changed')
    cosmos={r['sample_id']:r for r in read(cp)};require(set(cosmos)==set(allocated),'Cosmos comparison scope differs')
    between=[]
    for n in records:
        c=cosmos[n['sample_id']];require(list(map(tuple,c['source_media']))==list(map(tuple,n['source_media'])) and c['truth']==n['truth'] and c['task']==n['task'],'Cosmos/Nemo source or GT differs')
        between.append({'sample_id':n['sample_id'],'task':n['task'],'cell':n['cell'],'truth':n['truth'],'Cosmos':c['answer'],'Nemo':n['answer'],
                        'Cosmos_correct':c['correct'],'Nemo_correct':n['correct'],'answers_differ':c['answer']!=n['answer']})
    full_lookup={r['sample_id']:r for r in records};qualification_rows=read(run/'qualification-records.json')
    overlap=[]
    for q in qualification_rows:
        x=full_lookup[q['sample_id']];require(list(map(tuple,q['source_media']))==list(map(tuple,x['source_media'])) and q['truth']==x['truth'],'Qualification source mismatch')
        keys=['raw_answer','output_token_ids','finish_reason','stop_reason','prompt_token_sha256']
        overlap.append({'sample_id':q['sample_id'],'card':q['card'],'cold':q['cold'],'task':q['task'],**{k+'_equal':q[k]==x[k] for k in keys}})
    out=run/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'canonical-records.json',records);write_csv(out/'predictions.csv',records)
    write_csv(out/'cells.csv',[{'cell':list(k),**basic(rows)} for k,rows in sorted(cells.items())])
    write_csv(out/'paired-A-X.csv',paired)
    task_pairs=[{'task':t,**pair_stats([r for r in paired if r['task']==t])} for t in EXPECTED]
    write_csv(out/'comparison-A-X.csv',task_pairs)
    write_csv(out/'cells-A-X.csv',[{'cell':list(k),**pair_stats([r for r in paired if tuple(r['cell'])==k])} for k in sorted({tuple(r['cell']) for r in paired})])
    write_csv(out/'paired-Cosmos-Nemo-X.csv',between)
    write_csv(out/'strata.csv',[{'task':k[0],'stratum':k[1],'value':k[2],**basic(rows)} for k,rows in sorted(strata.items(),key=lambda x:str(x[0]))])
    write_csv(out/'timing.csv',[{'worker':worker,**t} for worker in sorted({r['worker'] for r in records}) for t in timing_tables([r for r in records if r['worker']==worker])])
    write_csv(out/'qualification-full-overlap.csv',overlap)
    result={'schema':'nemo-X-full-results-v1','status':'COMPLETE_PASS','completed_HST':datetime.now(ZoneInfo('Pacific/Honolulu')).isoformat(),
            'model':'nemo','precision':'official-mixed-NVFP4','protocol':'X','unique_videos':9456,'configs':169,'new_model_calls':0,
            'tasks':{t:basic(r for r in records if r['task']==t) for t in EXPECTED},'A_X_paired':task_pairs,'A_unavailable_in_X_scope':9456-len(paired),
            'Cosmos_X_paired':{t:{'n':sum(r['task']==t for r in between),'Cosmos_correct':sum(r['Cosmos_correct'] for r in between if r['task']==t),
                'Nemo_correct':sum(r['Nemo_correct'] for r in between if r['task']==t),'answer_disagreements':sum(r['answers_differ'] for r in between if r['task']==t)} for t in EXPECTED},
            'qualification_full_overlap':{'comparisons':len(overlap),'unique_ids':len({r['sample_id'] for r in overlap}),
                'all_fields_equal':sum(all(v for k,v in r.items() if k.endswith('_equal')) for r in overlap),'differences':[r for r in overlap if not all(v for k,v in r.items() if k.endswith('_equal'))]},
            'workers':proofs,'plan':descriptor(plans/'plan.json'),'allocation':descriptor(plans/'full-allocation.json'),'qualification':descriptor(gate),
            'historical_A':descriptor(a_path),'Cosmos_X':descriptor(cp),'input':{'frames':180,'effective_width':768,'effective_height':512,'reader':'official OpenCV uint8','actual_GPU_deep_admissions':sum(r['deep'] for r in records)},
            'limitations':['One full run; finite cold-start and cross-card panels do not guarantee every input/hardware.',
                'A/X differs in reader/input, runtime stack and actual backend policy; not a bit-depth-only causal comparison.',
                'Original clean paused prefix and all original answers/claims remain intact; qualification excluded from main scores.']}
    result['outputs']={p.name:descriptor(p) for p in out.iterdir() if p.is_file() and p.name!='summary.json'}
    atomic_json(out/'summary.json',result)
    return {k:result[k] for k in ['status','unique_videos','configs','tasks','A_X_paired','Cosmos_X_paired','qualification_full_overlap']}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--status',action='store_true');p.add_argument('--full',action='store_true');a=p.parse_args();r=status(a.root) if a.status else full(a.root) if a.full else qualification(a.root)
    import json
    print(json.dumps(r if a.status else {k:v for k,v in r.items() if k not in ('workers','input_profile_sha256')},ensure_ascii=False))
