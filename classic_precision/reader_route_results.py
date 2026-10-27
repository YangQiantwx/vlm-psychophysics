"""Read-only reconciliation and paired statistics for reader_route_smoke.

Never launches, retries, or selects samples. Complete mode requires every bound
raw/PASS/actual-wait/owned-exit/original-lease receipt for the fixed 1728 tuples.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import time

from .reader_route_smoke import NAME
from .refinement_worker import require, read
from .storage import digest, atomic_json, file_sha256


def records(path):
    if not path.exists(): return []
    return [json.loads(line) for line in path.open()]


def csv_write(path, rows):
    if not rows: return
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def stats(values):
    values=sorted(values)
    if not values:return {'n':0,'median':None,'mean':None,'p90':None}
    q=(len(values)-1)*.9;lo=int(q);hi=min(lo+1,len(values)-1)
    return {'n':len(values),'median':statistics.median(values),'mean':statistics.mean(values),
            'p90':values[lo]+(values[hi]-values[lo])*(q-lo)}


def status(root):
    """One bounded host snapshot. No tails, retries, launches or model imports."""
    root=Path(root);plan=root/'plans'/NAME;reports=[];all_sources={}
    for qpath in sorted(plan.glob('*-queue.json')):
        q=read(qpath)
        # Ignore a copied queue belonging to the other host.
        if not q.get('profiles') or not Path(q['profiles'][0]).exists():continue
        for path in q['profiles']:
            p=read(path)
            if p['hostname']!=socket.gethostname():continue
            all_sources.update(p['sources']);out=Path(p['output'])
            progress=read(out/'progress.json')if(out/'progress.json').exists()else{}
            terminal=read(out/'result.json')if(out/'result.json').exists()else{}
            failure=read(out/'failure.json')if(out/'failure.json').exists()else None
            timings=records(out/'request-timings.jsonl')
            ordinary=[t for t in timings if not t['observed_or_cold']][-20:]
            sample=stats([t['end_to_end_durable_seconds']for t in ordinary])
            completed=len(terminal.get('completed_ids',[]))if terminal else progress.get('completed',0)
            claimed={x.stem for x in Path(p['claim_root']).glob('*.json')} & set(p['assigned_ids'])
            inflight=sorted(claimed-{t['sample_id']for t in timings})
            wait=out.with_suffix('.wait.json')
            reports.append({'worker':p['worker'],'queue_status':q['status'],
                'status':terminal.get('status',progress.get('status','LOADING'if out.exists()else'QUEUED')),
                'completed':completed,'expected':len(p['assigned_ids']),'claimed':len(claimed),
                'inflight_or_unresolved':inflight,'failure':failure,
                'recent_20_ordinary_e2e_seconds':sample,
                'current_arm_remaining_seconds_at_recent_mean':(len(p['assigned_ids'])-completed)*sample['mean']if sample['n']else None,
                'ETA_scope':'current assigned arm only; excludes other arms/models and future loading',
                'actual_wait':{k:v for k,v in read(wait).items()if k!='completed_ids'}if wait.exists()else None})
    changed=[s for s,r in all_sources.items() if not Path(r['path']).exists()or
        [Path(r['path']).stat().st_dev,Path(r['path']).stat().st_ino,Path(r['path']).stat().st_size,
         Path(r['path']).stat().st_mtime_ns]!=r['stat']]
    mem={line.split(':')[0]:line.split(':')[1].strip()for line in Path('/proc/meminfo').read_text().splitlines()}
    def cpu():return list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
    a=cpu();time.sleep(.2);b=cpu();delta=[y-x for x,y in zip(a,b)]
    pressure={name:(Path('/proc/pressure')/name).read_text().strip()for name in ['cpu','memory','io']}
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,temperature.gpu',
        '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    disk=os.statvfs(root)
    return {'hostname':socket.gethostname(),'profiles':reports,
        'source_identity':{'unique_sources':len(all_sources),'changed_stat_ids':changed,
            'policy':'reused accepted all-frame source QA; immutable stat binding checked, no repeated full-video hash'},
        'resources':{'GPU_csv':gpu.stdout.strip(),'GPU_error':gpu.stderr.strip(),
            'MemAvailable':mem.get('MemAvailable'),'MemTotal':mem.get('MemTotal'),
            'SwapFree':mem.get('SwapFree'),'CPU_busy_percent_200ms':100*(1-(delta[3]+delta[4])/sum(delta))if sum(delta)else None,
            'PSI':pressure,'disk_available_GiB':disk.f_bavail*disk.f_frsize/2**30},
        'formal_auto_resume':False}


def reconcile(root, partial=False):
    root=Path(root);plan=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME
    alloc=read(plan/'allocation.json');sources=read(plan/'sources.json')['items']
    data={};timings={};closures=[];loads=[];inputs=[];expected=set();missing=[]
    for card,segments in alloc['cards'].items():
        for segment in segments:
            model=segment['model']
            for arm in segment['arms']:
                worker=model+'-'+card+'-'+arm;out=run/worker
                ids=segment['sample_ids'];expected.update((model,arm,s) for s in ids)
                raw=records(out/'raw.jsonl');events=records(out/'events.jsonl')
                passes={e['sample_id']:e for e in events if e.get('event')=='TECHNICAL_PASS'}
                require(len({r['sample_id']for r in raw})==len(raw),'Duplicate raw: '+worker)
                ppath=plan/(worker+'-profile.json')
                profile=read(ppath) if ppath.exists() else None
                if (out/'load.json').exists():
                    loads.append({'worker':worker,**read(out/'load.json')})
                for r in raw:
                    sid=r['sample_id'];key=(model,arm,sid)
                    require(key not in data and sid in ids and r['model']==model and r['arm']==arm,'Unexpected tuple')
                    require(r['truth']==sources[sid]['row']['truth'],'GT changed')
                    require(r['source']['video_sha256']==sources[sid]['ready']['video_sha256'],'Source changed')
                    require(profile and r['profile_sha256']==digest(profile),'Profile changed')
                    proof=passes.get(sid)
                    if proof is None:
                        if partial:continue
                        raise ValueError('Raw retained without technical PASS: '+str(key))
                    require(proof['raw_digest']==digest(r),'Raw/PASS digest mismatch')
                    require(r['finish_reason']=='stop' and r['raw_answer'].strip() in ['LEFT','RIGHT'],'Invalid completion retained')
                    data[key]=r
                    if r.get('observed_or_cold'):
                        inputs.append({'worker':worker,'sample_id':sid,'task':r['task'],
                            'reader':r.get('reader'),'native_reader':r.get('native_reader'),
                            'native_processor':r.get('native_processor'),
                            'Gemma_input_controls':{k:v for k,v in r.get('input_controls',{}).items()if k!='expected_prompt_token_ids'},
                            'actual_prompt_token_count':len(r['actual_prompt_token_ids']),
                            'actual_prompt_token_ids_sha256':digest(r['actual_prompt_token_ids']),
                            'actual_GPU_input':proof.get('actual_GPU_input')})
                ts=records(out/'request-timings.jsonl')
                for t in ts:
                    k=(model,arm,t['sample_id']);require(k not in timings,'Duplicate timing')
                    timings[k]=t
                wait=out.with_suffix('.wait.json')
                if wait.exists():
                    w=read(wait);closures.append({'worker':worker,**w})
                    if not partial:
                        require(w['returncode']==0 and w['real_wait_observed'] and w['owned_engines_exited']
                            and w['original_physical_lease_released'] and w['terminal_status']=='COMPLETE'
                            and w['completed_ids']==ids,'Actual terminal closure incomplete: '+worker)
                elif not partial:raise ValueError('Missing actual wait: '+worker)
    missing=sorted(expected-set(data))
    if not partial:
        require(len(expected)==1728 and not missing and set(timings)==expected,'1728 finite ledger incomplete')
        for card in alloc['cards']:
            require(read(plan/(card+'-queue.json'))['status']=='COMPLETE','Queue not terminal')
    historical={}
    for r in csv.DictReader((plan/'historical.csv').open()):
        key=(r['model'],r['sample_id']);require(key not in historical,'Duplicate history');historical[key]=r
    require(len(historical)==576,'Historical fixed panel changed')
    paired=[];matrix=[];transitions=[];strata=[];timing_rows=[]
    for model in ['qwen','cosmos','gemma','nemo']:
        for task in ['T2','T3']:
            ids=[s for s,x in sources.items() if x['task']==task]
            row={'model':model,'task':task,'denominator':len(ids),
                 'historical_correct':sum(historical[(model,s)]['answer'].strip()==sources[s]['row']['truth'] for s in ids)}
            for arm in 'ABC':
                available=[data[(model,arm,s)]for s in ids if(model,arm,s)in data]
                row[arm+'_complete']=len(available)
                row[arm+'_correct']=sum(r['raw_answer'].strip()==r['truth'] for r in available) if available else None
                tt=[timings[(model,arm,s)]for s in ids if(model,arm,s)in timings
                    and not timings[(model,arm,s)]['observed_or_cold']]
                end=stats([t['end_to_end_durable_seconds'] for t in tt])
                row[arm+'_ordinary_e2e_median_s']=end['median']
                row[arm+'_ordinary_e2e_mean_s']=end['mean']
                row[arm+'_ordinary_e2e_p90_s']=end['p90']
                row[arm+'_ordinary_answers_per_hour']=3600*len(tt)/sum(t['end_to_end_durable_seconds']for t in tt) if tt else None
                fields=sorted({k for t in tt for k,v in t.items() if k.endswith('_seconds') and isinstance(v,(float,int))})
                for field in fields:
                    timing_rows.append({'model':model,'task':task,'arm':arm,'stage':field,
                        'scope':('same observed external read already in read_decode_seconds; do not add'
                            if field=='native_reader_nested_seconds' and (arm=='B' or(model=='gemma'and arm=='C')) else
                            'nested within engine request; do not add') if '_nested_' in field else
                            ('inclusive; not GPU-only' if field=='engine_request_inclusive_seconds' else 'external measured worker stage'),
                        **stats([t[field]for t in tt if field in t])})
            for a,b in itertools.combinations('ABC',2):
                pairs=[(data[(model,a,s)],data[(model,b,s)])for s in ids if(model,a,s)in data and(model,b,s)in data]
                changed=sum(x['raw_answer'].strip()!=y['raw_answer'].strip() for x,y in pairs)
                ctw=sum(x['raw_answer'].strip()==x['truth'] and y['raw_answer'].strip()!=y['truth']for x,y in pairs)
                wtc=sum(x['raw_answer'].strip()!=x['truth'] and y['raw_answer'].strip()==y['truth']for x,y in pairs)
                row[a+b+'_paired_n']=len(pairs)
                row[a+b+'_answer_changes']=changed if pairs else None
                row[a+b+'_correct_to_wrong']=ctw if pairs else None;row[a+b+'_wrong_to_correct']=wtc if pairs else None
                transitions.append({'model':model,'task':task,'from':a,'to':b,'paired_n':len(pairs),
                    'answer_changes':changed if pairs else None,'correct_to_wrong':ctw if pairs else None,'wrong_to_correct':wtc if pairs else None,
                    'prompt_token_layout_changes':sum(x['actual_prompt_token_ids']!=y['actual_prompt_token_ids']for x,y in pairs) if pairs else None})
            matrix.append(row)
            for sid in ids:
                item=sources[sid];p=item['row']['source_parameters'];truth=item['row']['truth']
                rec={'model':model,'task':task,'sample_id':sid,'truth':truth,'onset_index':p['onset_index'],
                     'onset_frame':p['onset_frame'],'tilt_sign':p.get('tilt_sign'),
                     'historical_answer':historical[(model,sid)]['answer'],
                     'source_video_sha256':item['ready']['video_sha256']}
                for arm in 'ABC':
                    r=data.get((model,arm,sid));rec[arm+'_answer']=r['raw_answer'].strip() if r else None
                    rec[arm+'_correct']=r['raw_answer'].strip()==truth if r else None
                paired.append(rec)
            relevant=[r for r in paired if r['model']==model and r['task']==task]
            for field in ['truth','onset_index']+(['tilt_sign']if task=='T3'else[]):
                for value in sorted({r[field]for r in relevant},key=str):
                    ss=[r for r in relevant if r[field]==value]
                    st={'model':model,'task':task,'stratifier':field,'value':value,'expected_n':len(ss)}
                    for arm in 'ABC':
                        st[arm+'_complete']=sum(r[arm+'_answer']is not None for r in ss)
                        st[arm+'_correct']=sum(r[arm+'_correct']is True for r in ss) if st[arm+'_complete'] else None
                    for a,b in itertools.combinations('ABC',2):
                        pp=[r for r in ss if r[a+'_answer']is not None and r[b+'_answer']is not None]
                        st[a+b+'_paired_n']=len(pp)
                        st[a+b+'_changes']=sum(r[a+'_answer']!=r[b+'_answer']for r in pp) if pp else None
                        st[a+b+'_correct_to_wrong']=sum(r[a+'_correct']and not r[b+'_correct']for r in pp) if pp else None
                        st[a+b+'_wrong_to_correct']=sum(not r[a+'_correct']and r[b+'_correct']for r in pp) if pp else None
                    strata.append(st)
    status='COMPLETE_PASS' if not partial else 'PARTIAL_NO_FINAL_CLAIM'
    report={'status':status,'expected':1728,'technical_pass':len(data),'missing_tuples':missing,
        'matrix':matrix,'paired_transitions':transitions,'timing':timing_rows,
        'closures':closures,'loading_separate_from_requests':loads,'observed_actual_inputs':inputs,
        'observation_cold_requests_separate':[{'model':m,'arm':a,**t}for(m,a,s),t in timings.items()if t.get('observed_or_cold')],
        'formal_auto_resume':False,
        'timing_interpretation':'ordinary same-host/same-GPU paired arms only; B and Gemma C observed native-reader counter duplicates the external read_decode stage; other C native reader and native processor are inside engine request. Never add overlapping counters; bridge/IPC not independently separable, never equate remainder to GPU',
        'analysis_source_sha256':file_sha256(__file__),
        'Gemma_C':'official OpenCV uint8 -> native HF video processor, 280 spatial budget -> post-processor BF16 -> shared spatial/input adapter; not unmodified vLLM native file video',
        'historical_timing':'not used for causal speed comparison',
        'bindings':{n:file_sha256(plan/n)for n in ['plan.json','sources.json','allocation.json','historical.csv']}}
    # A single existing result namespace; summaries are rebuildable from original raw.
    dest=run/('partial' if partial else 'comparison');dest.mkdir(parents=True,exist_ok=True)
    atomic_json(dest/'summary.json',report)
    csv_write(dest/'comparison.csv',matrix);csv_write(dest/'paired.csv',paired)
    csv_write(dest/'strata.csv',strata);csv_write(dest/'timing.csv',timing_rows)
    return {'status':status,'technical_pass':len(data),'expected':1728,'output':str(dest)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--partial',action='store_true');p.add_argument('--status',action='store_true')
    a=p.parse_args();print(json.dumps(status(a.root)if a.status else reconcile(a.root,a.partial),sort_keys=True))


if __name__=='__main__':main()
