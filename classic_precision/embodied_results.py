"""Read-only reconciliation and timing budgets for finite embodied admissions."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
from .embodied_admission import require, read
from .storage import atomic_json, digest, file_sha256

NAME='embodied-points-20261027'
COUNTS={'T1':3840,'T2':5760,'T3':2928}


def lines(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def csv_write(path,rows):
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def reconcile_model(root,model):
    runs=['embodied-admission-20261027',NAME] if model=='rynn' else [NAME]
    combined={1:{},2:{}};all_times=defaultdict(list);ordinary=defaultdict(list);loads=[];receipts=[]
    for name in runs:
        base=root/'runs/reproducibility'/name
        plan=read(root/'plans'/name/('rx-plan.json' if model=='rx' else 'plan.json'))
        samples={s['sample_id']:s for s in plan['samples']}
        for cold in [1,2]:
            folder=base/f'{model}-cold{cold}';wait=read(base/f'{model}-cold{cold}.wait.json')
            require(wait['returncode']==0 and wait['real_wait'] and wait['owned_hf_process_exited']
                    and wait['original_same_inode_lease_free'],'Worker not fully closed')
            rows=lines(folder/'raw.jsonl');events=lines(folder/'events.jsonl');times=lines(folder/'timings.jsonl')
            require(len(rows)==len(samples) and {r['sample_id'] for r in rows}==set(samples),'Missing or duplicate raw ID')
            passes={e['sample_id']:e for e in events if e['event']=='TECHNICAL_PASS'}
            timing={t['sample_id']:t for t in times}
            require(len(passes)==len(timing)==len(rows),'Missing PASS/timing')
            result=read(folder/'result.json');require(result['status']=='COMPLETE_PASS' and set(result['completed_ids'])==set(samples),'Incomplete result')
            seen_tasks=set()
            for row in rows:
                sid=row['sample_id'];s=samples[sid];claim=read(folder/(sid+'.claim.json'))
                require(sid not in combined[cold],'Duplicate case across runs')
                require(row['source_sha256']==s['source_video_sha256'] and row['truth']==s['truth']
                    and row['plan_sha256']==digest(plan) and claim['plan_sha256']==digest(plan),'Source/GT/config drift')
                require(passes[sid]['raw_sha256']==digest(row),'PASS does not bind raw')
                require(row['input']['reader_shape']==[180,512,768,3] and row['input']['frame_indices']==list(range(180)), 'Input geometry/order drift')
                require(row['finish_reason']=='stop' and row['answer'] in ['LEFT','RIGHT'],'Invalid raw answer')
                combined[cold][sid]=row
                task=row['task'];sec=timing[sid]['end_to_durable_raw_and_pass_seconds'];all_times[task].append(sec)
                if task not in seen_tasks:seen_tasks.add(task)
                else:ordinary[task].append(sec)
            load=read(folder/'load.json');loads.append(load['seconds']);receipts.append(file_sha256(base/f'{model}-cold{cold}.wait.json'))
    require(len(combined[1])==len(combined[2])==18,'Expected18 paired IDs')
    paired=[];cells=defaultdict(lambda:{'n':0,'cold1_correct':0,'cold2_correct':0,'matches':0})
    cells_by_id={s['sample_id']:s['cell_json'] for s in read(root/'plans'/NAME/'rx-plan.json')['samples']}
    for sid,a in combined[1].items():
        b=combined[2][sid]
        match=all(a[k]==b[k] for k in ['answer','output_token_ids','finish_reason','stop_reason'])
        inputs=all(a['input'][k]==b['input'][k] for k in ['processor_sha256','prompt_token_ids','decoded_sha256'])
        row={'model':model,'sample_id':sid,'task':a['task'],'cell':cells_by_id[sid],'truth':a['truth'],
             'cold1_answer':a['answer'],'cold2_answer':b['answer'],'full_output_match':match,'input_match':inputs}
        paired.append(row);c=cells[row['cell']];c['n']+=1;c['cold1_correct']+=a['answer']==a['truth'];c['cold2_correct']+=b['answer']==b['truth'];c['matches']+=match and inputs
    summary={'status':'COMPLETE_SAME_CARD_COLD_REPLAY' if all(p['full_output_match'] and p['input_match'] for p in paired) else 'COMPLETE_REPLAY_GAP',
        'model_calls':36,'unique_videos':18,'complete_matches':sum(p['full_output_match'] and p['input_match'] for p in paired),
        'cold_load_seconds':loads,'worker_wait_sha256':receipts,'cells':dict(cells),
        'ordinary_E2E_seconds':{t:{'n':len(v),'mean':statistics.mean(v),'median':statistics.median(v),
            'p90':sorted(v)[min(len(v)-1,int(.9*len(v)))]} for t,v in ordinary.items()},
        'timing_limit':'All requests include input admission/hash/vision observation. First request per task per cold worker excluded; these are small-pilot throughput estimates, not optimized full-run benchmarks.'}
    return summary,paired


def analyze(root):
    root=Path(root);out=root/'runs/reproducibility'/NAME
    models={};paired=[]
    for model in ['rynn','rx']:
        summary,rows=reconcile_model(root,model);models[model]=summary;paired+=rows
    result={'status':'COMPLETE_FINITE_ADMISSION','models':models,'total_calls':72,
            'limits':['18 fixed videos per model, two same-card cold starts; no full curves, no cross-host claim.',
                      'Rynn uses video pairs; Rx uses official VQA ordered images. Both preserve180 source frames.']}
    atomic_json(out/'embodied-summary.json',result);csv_write(out/'embodied-paired.csv',paired)
    old_path=root/'runs/reproducibility/cosmos-t3-neighbors-20261027/gpu-hours-estimate.json';old=read(old_path)
    rates={m:v['seconds_per_video'] for m,v in old['models'].items()}
    rates.update({m:{t:v['mean'] for t,v in s['ordinary_E2E_seconds'].items()} for m,s in models.items()})
    per_model={m:{'seconds_per_video':v,'T1_T3_GPU_hours':sum(v[t]*n/3600 for t,n in COUNTS.items())} for m,v in rates.items()}
    scenarios={}
    scopes={'classic_T1_T3':COUNTS,'classic_T1_T5':{**COUNTS,'T4':312,'T5':384},
        'classic_plus_one_natural_T1_T5':{'T1':7680,'T2':7680,'T3':5856,'T4':624,'T5':768}}
    for name,counts in scopes.items():
        hours=sum(rates[m][t]*counts[t]/3600 for m in rates for t in COUNTS)
        extra_low=extra_high=0
        for m,v in rates.items():
            low,high=(15,30) if m in old['models'] else (.75*statistics.mean(v.values()),1.5*statistics.mean(v.values()))
            extra_low+=low*(counts.get('T4',0)+counts.get('T5',0))/3600
            extra_high+=high*(counts.get('T4',0)+counts.get('T5',0))/3600
        total=[hours+extra_low,hours+extra_high];margin=[total[0]*1.15,total[1]*1.25]
        scenarios[name]={'videos_per_model':sum(counts.values()),'six_model_calls':6*sum(counts.values()),
            'task_counts':counts,'GPU_slot_hours_central_or_proxy_range':total,
            'GPU_slot_hours_planning_15_to_25pct':margin,'three_card_ideal_planning_days':[x/72 for x in margin]}
    budget={'status':'PLANNING_ONLY_NO_FULL_RUN_AUTHORIZED','previous_four_model_estimate_sha256':file_sha256(old_path),
        'new_finite_summary_sha256':file_sha256(out/'embodied-summary.json'),'models':per_model,'scenarios':scenarios,
        'limits':['Four-model T1 remains own-model T2/T3 proxy; new-model T1 uses small finite pilot actual E2E.',
                  'No T4/T5 accepted speed: existing four use15–30s; new two use0.75–1.5 times own T1–T3 mean as scheduling proxy.',
                  'Natural set costs assume same frames/geometry/output budget; no natural six-model throughput measured.',
                  'Three-card days assume all three available and balanced. Excludes media preparation, human collection, writing, model transfers.',
                  '15–25percent margins are scheduling allowances, not statistical confidence bounds.']}
    atomic_json(out/'six-model-gpu-hours.json',budget)
    print(json.dumps({'status':result['status'],'matches':{m:s['complete_matches'] for m,s in models.items()},'budgets':scenarios}))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();analyze(a.root)

if __name__=='__main__':main()
