"""Seal and audit the authorized 384-call Cosmos neighboring-angle comparison.

Execution is the unchanged reader_route_smoke worker. User route B maps to its
native C arm; this module never changes processors, model math or sampling.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
import fcntl
import json
import os
from pathlib import Path
import socket
import statistics
import sys

from .refinement_worker import read, require, write_new, descriptor, verified
from .storage import digest, file_sha256, atomic_json

NAME = 'cosmos-t3-neighbors-20261027'
ROOTS = ('/home/yang/cvpr2027_benchmark', '/mnt/sda/q/cvpr2027_benchmark')


def local_path(root, path):
    for prefix in ROOTS:
        if str(path).startswith(prefix + '/'):
            return Path(root) / str(path)[len(prefix)+1:]
    return Path(path)


def seal(root, host):
    root = Path(root); plan = root/'plans'/NAME
    experiment = read(plan/'plan.json'); catalog = read(plan/'sources.json')
    allocation = read(plan/'allocation.json')
    require(experiment['maximum_new_scientific_calls'] == 384, 'Changed quota')
    require(experiment['fixed_angles_deg'] == [.6, .8], 'Changed fixed cells')
    for name, sha in experiment['files'].items():
        require(file_sha256(plan/name) == sha, 'Changed sealed input: '+name)
    cards = ['fourier0', 'fourier1'] if host == 'fourier' else ['radon0']
    base_name = 'cosmos-fourier1-A-profile.json' if host == 'fourier' else 'cosmos-radon0-A-profile.json'
    base_path = root/'plans/reader-route-smoke-20261026'/base_name
    base = read(base_path)
    from . import reader_route_smoke as worker
    files = {}
    for card in cards:
        ids = allocation['cards'][card]['sample_ids']
        require(len(ids) == 64 and len(set(ids)) == 64, 'Wrong allocation')
        for sid in ids:
            item = catalog['items'][sid]
            p = item['row']['source_parameters']
            require(item['task'] == 'T3' and p['angle_deg'] in [.6, .8], 'Wrong cell')
            path = Path(base['runtime']['asset_roots'][item['source_family']])/item['ready']['video_relative']
            require(file_sha256(path) == item['ready']['video_sha256'], 'Changed source')
            files[sid] = {'path':str(path), 'stat':worker.state(path)}
    result = []
    for card in cards:
        gpu = read(root/'plans/classic-refinement'/('gemma-'+card+'-profile.json'))
        lock = gpu['physical_lock']
        with Path(lock['path']).open('r+b') as f:
            st = os.fstat(f.fileno())
            require((st.st_dev,st.st_ino)==(lock['st_dev'],lock['st_ino']), 'Lease inode changed')
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        profiles = []
        for arm in ['A','C']:
            p = deepcopy(base); ids = allocation['cards'][card]['sample_ids']
            worker_name = 'cosmos-'+card+'-'+arm+'-neighbors'
            out = root/'runs/reproducibility'/NAME/worker_name
            p.update(worker=worker_name, assigned_ids=ids, maximum_model_calls=64,
                arm=arm, user_route='A' if arm=='A' else 'B',
                route_label=('controlled RGB16/PyAV -> FP32, legacy downstream' if arm=='A' else
                    'official OpenCV uint8 + native processor, no resize/no secondary sampling; internal C'),
                output=str(out), claim_root=str(out.parent/'claims'/'cosmos'/arm),
                pause_file=str(plan/(worker_name+'-PAUSE.json')),
                worker_token=digest({'allocation':descriptor(plan/'allocation.json'),'worker':worker_name}),
                source_catalog=descriptor(plan/'sources.json'), allocation=descriptor(plan/'allocation.json'),
                experiment=descriptor(plan/'plan.json'), sources={sid:files[sid] for sid in ids},
                implementation=descriptor(worker.__file__), gpu_uuid=gpu['gpu_uuid'],
                physical_lock=lock, hostname=socket.gethostname(),
                qualified_neighbor_parent=descriptor(base_path), automatic_retry=False, formal_auto_resume=False)
            p['environment']['CUDA_VISIBLE_DEVICES'] = card[-1]
            require(not out.exists(), 'Existing output; no implicit resume')
            path = plan/(worker_name+'-profile.json');write_new(path,p);profiles.append(str(path))
        q = plan/(card+'-queue.json')
        write_new(q,{'status':'SEALED','profiles':profiles,'automatic_retry':False,'formal_auto_resume':False})
        result.append(str(q))
    return {'status':'SEALED','queues':result,'sources_verified':len(files),'GPU_calls':0}


def cpu(profile):
    """Observe the installed native C processor before any GPU request."""
    p=read(profile);require(p['arm']=='C' and p['model']=='cosmos','Wrong CPU route')
    for key in p.get('cleared_environment',[]):os.environ.pop(key,None)
    os.environ.update(p['environment']);os.environ['CUDA_VISIBLE_DEVICES']=''
    sys.path.insert(0,p['runtime']['cosmos_package'])
    import transformers_cosmos3  # noqa: F401
    import torch
    torch.set_num_threads(2);torch.set_grad_enabled(False)
    from transformers import AutoProcessor
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.inputs import ProcessorInputs
    from vllm.multimodal.processing.context import TimingContext
    from .reader_route_smoke import decode, NativeObservation
    processor=AutoProcessor.from_pretrained(p['checkpoint'],local_files_only=True,trust_remote_code=False)
    config=ModelConfig(model=p['checkpoint'],tokenizer=p['runtime']['original_checkpoint'],dtype='bfloat16',
        max_model_len=65536,limit_mm_per_prompt={'video':1,'image':0,'audio':0},mm_processor_cache_gb=0,
        mm_processor_kwargs={'do_resize':False,'do_sample_frames':False})
    native=MULTIMODAL_REGISTRY.create_processor(config,tokenizer=processor.tokenizer)
    catalog=verified(p['source_catalog']); seen=set(); rows=[]
    for sid in p['assigned_ids']:
        item=catalog['items'][sid];angle=item['row']['source_parameters']['angle_deg']
        if angle in seen:continue
        seen.add(angle);frames,receipt,_=decode(p['sources'][sid]['path'],'C')
        text=processor.apply_chat_template([{'role':'user','content':[{'type':'video'},
            {'type':'text','text':item['row']['prompt']}]}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        parsed=native.info.parse_mm_data({'video':(frames,receipt['official_metadata'])})
        with NativeObservation('cosmos') as obs:
            obs.active=True
            result=native.apply(ProcessorInputs(prompt=processor.tokenizer.encode(text,add_special_tokens=False),
                mm_data_items=parsed,hf_processor_mm_kwargs={'do_resize':False,'do_sample_frames':False}),TimingContext(enabled=False))
            require(obs.processors,'No processor observation')
            rows.append({'sample_id':sid,'angle_deg':angle,'reader':receipt,'actual_processor':obs.processors,
                'prompt_token_count':len(result['prompt_token_ids']),'prompt_token_sha256':digest(result['prompt_token_ids'])})
        del frames,result
    require(seen=={.6,.8} and not torch.cuda.is_initialized(),'Incomplete CPU check or CUDA initialized')
    record={'status':'PASS','model_calls':0,'CUDA_initialized':False,'profile_sha256':digest(p),'rows':rows}
    write_new(Path(profile).with_suffix('.cpu.json'),record)
    return {'status':'PASS','worker':p['worker'],'points':len(rows),'model_calls':0}


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def analyze(root):
    root=Path(root);plan=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME
    catalog=read(plan/'sources.json');hist={r['sample_id']:r for r in csv.DictReader((plan/'historical.csv').open())}
    all_rows={};timings=[];closures=[];profiles=[]
    for card in ['fourier0','fourier1','radon0']:
        queue=read(plan/(card+'-queue.json'));require(queue['status']=='COMPLETE','Queue not complete: '+card)
        for remote in queue['profiles']:
            pp=local_path(root,remote);p=read(pp);out=local_path(root,p['output']);arm=p['arm'];profiles.append(p)
            wait=read(out.with_suffix('.wait.json'))
            require(wait['returncode']==0 and wait['real_wait_observed'] and wait['owned_engines_exited'] and
                wait['original_physical_lease_released'] and wait['terminal_status']=='COMPLETE','Unclosed worker')
            require(wait['profile_sha256']==digest(p) and wait['completed_ids']==p['assigned_ids'],'Closure identity')
            raw=records(out/'raw.jsonl');events=records(out/'events.jsonl');times=records(out/'request-timings.jsonl')
            pass_events=[e for e in events if e['event']=='TECHNICAL_PASS']
            passed={e['sample_id']:e for e in pass_events}
            require([r['sample_id'] for r in raw]==[t['sample_id'] for t in times]==p['assigned_ids'] and
                len(passed)==len(pass_events)==len(raw)==len(times)==64,'ID ledger')
            for r in raw:
                sid=r['sample_id'];key=(sid,arm);source=catalog['items'][sid]
                require(key not in all_rows and r['profile_sha256']==digest(p) and passed[sid]['raw_digest']==digest(r),'Raw identity')
                require(r['truth']==hist[sid]['truth'] and r['source']['video_sha256']==source['ready']['video_sha256'],'Source/GT')
                require(r['source']['row_sha256']==source['row_sha256'] and
                    r['source']['stream_sha256']==source['source_stream_sha256'],'Source parameters/stream identity')
                claim=read(local_path(root,p['claim_root'])/(sid+'.json'))
                require(claim['profile_sha256']==digest(p) and claim['attempt']==1,'Claim mismatch')
                all_rows[key]=r
            timings.extend([{**t,'arm':arm,'card':card,'angle_deg':catalog['items'][t['sample_id']]['row']['source_parameters']['angle_deg']} for t in times])
            closures.append({'worker':p['worker'],**wait})
    require(len(all_rows)==384 and len(hist)==192,'Incomplete fixed budget')
    for arm in ['A','C']:
        require({p.stem for p in (run/'claims/cosmos'/arm).glob('*.json')}==set(hist),'Unclosed/extra/missing claim')
    paired=[]
    for sid in sorted(hist):
        old=hist[sid];a=all_rows[sid,'A'];b=all_rows[sid,'C'];sp=catalog['items'][sid]['row']['source_parameters']
        require(a['source']['path']==b['source']['path'],'Unpaired source/card')
        require(a['actual_prompt_token_ids']==b['actual_prompt_token_ids'],'Prompt layout changed')
        ha=old['answer'];aa=a['raw_answer'].strip();ba=b['raw_answer'].strip();truth=old['truth']
        paired.append({'sample_id':sid,'angle_deg':sp['angle_deg'],'truth':truth,'tilt_sign':sp['tilt_sign'],
            'onset_index':sp['onset_index'],'history':ha,'A':aa,'B':ba,'history_correct':ha==truth,
            'A_correct':aa==truth,'B_correct':ba==truth,'history_A_changed':ha!=aa,'A_B_changed':aa!=ba,
            'A_B_correct_to_wrong':aa==truth and ba!=truth,'A_B_wrong_to_correct':aa!=truth and ba==truth,
            'history_A_tokens_changed':json.loads(old['output_token_ids'])!=a['output_token_ids'],
            'history_A_finish_changed':old['finish_reason']!=a['finish_reason'],
            'A_B_tokens_changed':a['output_token_ids']!=b['output_token_ids'],
            'A_B_finish_stop_changed':(a['finish_reason'],a['stop_reason'])!=(b['finish_reason'],b['stop_reason'])})
    def stats(vals):
        vals=sorted(vals);n=len(vals)
        return {'n':n,'mean':statistics.mean(vals),'median':statistics.median(vals),
            'p90':vals[min(n-1,int(.9*n))],'throughput_per_hour':3600*n/sum(vals)} if vals else {'n':0}
    cells=[]
    for angle in [.6,.8]:
        rr=[r for r in paired if r['angle_deg']==angle]
        cell={'angle_deg':angle,'n':len(rr),**{k:sum(r[k] for r in rr) for k in (
            'history_correct','A_correct','B_correct','history_A_changed','history_A_tokens_changed',
            'A_B_changed','A_B_correct_to_wrong','A_B_wrong_to_correct')}}
        cell['ordinary_E2E_seconds']={arm:stats([t['end_to_end_durable_seconds'] for t in timings
            if t['angle_deg']==angle and t['arm']==internal and not t['observed_or_cold']]) for arm,internal in [('A','A'),('B','C')]}
        cells.append(cell)
    strata=[]
    for angle in [.6,.8]:
        for field in ['truth','tilt_sign','onset_index']:
            for value in sorted({r[field] for r in paired}):
                rr=[r for r in paired if r['angle_deg']==angle and r[field]==value]
                strata.append({'angle_deg':angle,'stratum':field,'value':value,'n':len(rr),
                    **{k:sum(r[k] for r in rr) for k in ['history_correct','A_correct','B_correct','A_B_changed','A_B_correct_to_wrong','A_B_wrong_to_correct']}})
    result={'status':'COMPLETE_PASS','calls':384,'unique_sources':192,'cells':cells,'closures':closures,
        'protocol':'user B = prior C, official uint8/native/no resize,180 frames; A/B source/card/prompt matched',
        'causal_limit':'actual reader plus native route comparison, not ideal scalar quantization-only',
        'source_catalog_sha256':file_sha256(plan/'sources.json'),'historical_sha256':file_sha256(plan/'historical.csv')}
    run.mkdir(exist_ok=True,parents=True)
    atomic_json(run/'summary.json',result)
    for name,rows in [('paired.csv',paired),('strata.csv',strata),('timing.csv',timings)]:
        fields=sorted({k for r in rows for k in r})
        with (run/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    return {'status':result['status'],'calls':384,'cells':cells}


def pixel_summary(root):
    """Reconcile the four CPU journals and retain task-unit measurements."""
    from collections import defaultdict
    root=Path(root);run=root/'runs/reproducibility'/NAME
    plan=read(root/'plans'/NAME/'pixel-plan.json'); measured=[]
    for shard in range(4):
        out=run/('pixel-'+str(shard));identity=read(out/'identity.json');end=read(out/'completion.json')
        rows=records(out/'measurements.jsonl')
        require(end['status']=='COMPLETE_PASS' and end['CUDA_initialized'] is False and
            [r['sample_id'] for r in rows]==identity['assigned_ids']==end['ids'],'Incomplete CPU shard')
        require(identity['plan_sha256']==file_sha256(root/'plans'/NAME/'pixel-plan.json'),'CPU plan changed')
        measured.extend(rows)
    require(len(measured)==len({r['sample_id'] for r in measured})==682,'Incomplete CPU coverage')
    require({r['sample_id'] for r in measured}==set(plan['items']),'CPU ID coverage')
    long=[]
    for r in measured:
        p=r['nominal'];task=r['task']
        if task=='T1':
            specs=[('modulation_depth','fraction',p['modulation_depth']),
                ('contrast_normalized_measured_mean','fraction',p['modulation_depth']),
                ('sf_cycles_per_aperture','cycles/aperture',p['sf_cycles_per_aperture']),
                ('temporal_frequency_hz','Hz',p['tf_hz']),('mean_linear_Y','relative_linear_Y',p['classic_mean_Y'])]
            values={route:{**v['estimated'],'contrast_normalized_measured_mean':
                v['estimated']['modulation_depth']*p['classic_mean_Y']/v['estimated']['mean_linear_Y']}
                for route,v in r['routes'].items()}
        elif task=='T2':
            specs=[('delta_Y','relative_linear_Y',p['delta_Y']),('base_Y','relative_linear_Y',p['initial_dot_Y'])]
            values={route:v[p['target_side']] for route,v in r['routes'].items()}
        else:
            specs=[('pca_deg','degree',p['angle_deg']),('edge_deg','degree',p['angle_deg'])]
            tilted='RIGHT' if p['reference_side']=='LEFT' else 'LEFT'
            values={route:{k:abs(v[tilted][k]) for k in ['pca_deg','edge_deg']} for route,v in r['routes'].items()}
        for metric,unit,nominal in specs:
            for route in ['source16','fp32','uint8']:
                value=values[route][metric];err=value-nominal
                long.append({'sample_id':r['sample_id'],'task':task,'cell_json':json.dumps(r['cell']),
                    'route':route,'metric':metric,'unit':unit,'nominal':nominal,'measured':value,
                    'error':err,'relative_error_pct':100*err/nominal if nominal else None,
                    'difference_from_fp32':value-values['fp32'][metric]})
    grouped=defaultdict(list)
    for r in long:grouped[(r['cell_json'],r['route'],r['metric'])].append(r)
    cells=[]
    for key,rows in sorted(grouped.items()):
        first=rows[0];cells.append({k:first[k] for k in ['task','cell_json','route','metric','unit','nominal']}|
            {'n_representative_videos':len(rows),'measured_mean':statistics.mean(r['measured'] for r in rows),
             'measured_min':min(r['measured'] for r in rows),'measured_max':max(r['measured'] for r in rows),
             'max_abs_error':max(abs(r['error']) for r in rows),
             'max_abs_relative_error_pct':max(abs(r['relative_error_pct']) for r in rows if r['relative_error_pct'] is not None),
             'max_abs_difference_from_fp32':max(abs(r['difference_from_fp32']) for r in rows)})
    bytask={}
    for task in ['T1','T2','T3']:
        bytask[task]={}
        for route in ['source16','fp32','uint8']:
            bytask[task][route]={}
            for metric in sorted({r['metric'] for r in long if r['task']==task}):
                rr=[r for r in long if r['task']==task and r['route']==route and r['metric']==metric]
                worst=max(rr,key=lambda r:abs(r['error']))
                bytask[task][route][metric]={'unit':worst['unit'],'n':len(rr),
                    'max_abs_error':abs(worst['error']),'max_abs_difference_from_fp32':max(abs(r['difference_from_fp32']) for r in rr),
                    'max_abs_relative_error_pct':max(abs(r['relative_error_pct']) for r in rr),
                    'worst_absolute_error_sample':worst['sample_id'],'worst_absolute_error_cell':json.loads(worst['cell_json']),
                    'nominal_at_worst_absolute':worst['nominal'],'measured_at_worst_absolute':worst['measured']}
    source_t1=read(root/'runs/reproducibility/operator/gemma-production-input-catalog-20261024.json')['items']
    parity=[]
    for r in measured:
        if r['task']=='T1':
            expected=source_t1[r['sample_id']]['ready']['validation']['independent_readout']['estimated']
            actual=r['routes']['source16']['estimated']
            parity.append(max(abs(actual[k]-expected[k]) for k in actual if isinstance(actual[k],(float,int))))
    require(max(parity)<1e-7,'Frozen source estimator no longer matches existing source QA')
    summary={'schema':'all-grid-reader-pixel-summary-v1','status':'COMPLETE_PASS','model_calls':0,'processor_calls':0,
        'representative_videos':682,'counts':dict(Counter(r['task'] for r in measured)),
        'full_grid_videos_represented':12528,'nominal_cells':{t:len({json.dumps(r['cell']) for r in measured if r['task']==t}) for t in bytask},
        'by_task_route':bytask,'source_T1_readout_vs_existing_QA_max_difference':max(parity),
        'fp32_recovered_code_mismatches':sum(r['pixel_errors']['fp32_code_mismatches'] for r in measured),
        'T1_direction_mismatches':{route:sum(r['routes'][route]['estimated']['direction']!=r['truth'] for r in measured if r['task']=='T1') for route in ['source16','fp32','uint8']},
        'limits':['All nominal points, not all12528 files. T2/T3 onset0 covers unique spatial states; T1 phase parity0/1 and both directions cover the24 source bank states.',
            'Digital inverse-sRGB Rec709 relative luminance; not calibrated screen luminance or human thresholds.',
            'T1 modulation_depth follows the frozen estimator: fitted fundamental amplitude divided by nominal mean0.31 and pixel sinc; contrast_normalized_measured_mean additionally divides by observed mean. Both are harmonic estimates, not extrema-based Michelson measurements.',
            'T3 PCA and edge estimators have different antialiasing sensitivity; both retained, neither fitted to GT.',
            'Actual official OpenCV decoding includes channel/color conversion effects; not ideal nearest-uint8 quantization.',
            'Pixel fidelity does not alone establish model accuracy; new inference is confined to Cosmos192 pairs.']}
    atomic_json(run/'pixel-summary.json',summary)
    for name,rows in [('pixel-per-video.csv',long),('pixel-per-cell.csv',cells)]:
        with (run/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return summary


def cost(root):
    """Planning GPU-slot occupancy; never claims a measured all-grid runtime."""
    root=Path(root);run=root/'runs/reproducibility'/NAME
    timing_path=root/'runs/reproducibility/reader-route-smoke-20261026/partial/timing.csv'
    timing=list(csv.DictReader(timing_path.open()));counts={'T1':3840,'T2':5760,'T3':2928};models={}
    old=read(root/'runs/reproducibility/smoke-results/t1-resize-summary.json')
    extension=root/'runs/reproducibility/reader-neighbors-extension-20261027'
    current_B=None
    if (extension/'summary.json').exists():
        accepted=read(extension/'summary.json')
        require(accepted['status']=='COMPLETE_DESCRIPTIVE_COMPARISON' and accepted['B_answers']==1152,'Invalid complete neighbor comparison')
        current_B=list(csv.DictReader((extension/'timing.csv').open()))
    for model in ['qwen','cosmos','gemma','nemo']:
        means={task:float(next(r['mean'] for r in timing if r['model']==model and r['task']==task and
            r['arm']=='C' and r['stage']=='end_to_end_durable_seconds')) for task in ['T2','T3']}
        previous_t3=means['T3'];basis='recent finite native C E2E; fixed two cells'
        if model=='cosmos' and (run/'summary.json').exists():
            summary=read(run/'summary.json');require(summary['status']=='COMPLETE_PASS','Invalid new timing source')
            rows=list(csv.DictReader((run/'timing.csv').open()))
            current=[float(r['end_to_end_durable_seconds']) for r in rows if r['arm']=='C' and r['observed_or_cold']=='False']
            means['T3']=statistics.mean(current);basis='new0.6/0.8 native route, two Fourier/one Radon equal ID allocation; concurrent low-priority CPU audit recorded'
        if current_B is not None:
            means={task:statistics.mean([float(r['end_to_end_durable_seconds']) for r in current_B
                if r['model']==model and r['task']==task and r['observed_or_cold']=='False']) for task in ['T2','T3']}
            basis='complete1152 B neighbors; T2 .05/.07, T3 .6/.8; assigned hosts retained, not a cross-host speedup claim'
        means['T1']=statistics.mean([means['T2'],means['T3']])
        hours={task:counts[task]*means[task]/3600 for task in counts}
        models[model]={'task_counts':counts,'seconds_per_video':means,'task_gpu_hours':hours,
            'total_gpu_hours_central':sum(hours.values()),'T3_basis':basis,'previous_smoke_T3_E2E_seconds':previous_t3,
            'T1_basis':'proxy from this model native T2/T3 E2E mean, NOT direct new-route T1 observation',
            'legacy_T1_chat_only_seconds_reference':old.get('models',{}).get(model,{}).get('speed',{}).get('no_resize',{}).get('request_mean_seconds')}
    total=sum(m['total_gpu_hours_central'] for m in models.values())
    result={'schema':'official-no-resize-all-grid-budget-v1','status':'ESTIMATE_NOT_RUN_AUTHORIZATION',
        'counts_per_model':counts,'total_per_model':12528,'total_calls':50112,'models':models,
        'total_gpu_hours_central':total,'planning_gpu_hours_with_15_to_25pct_margin':[total*1.15,total*1.25],
        'three_card_ideal_wall_hours':total/3,'three_card_planning_wall_hours':[total*1.15/3,total*1.25/3],
        'source_timing_sha256':file_sha256(timing_path),'neighbor_summary_sha256':file_sha256(run/'summary.json') if (run/'summary.json').exists() else None,
        'complete_B_timing_sha256':file_sha256(extension/'timing.csv') if current_B is not None else None,
        'complete_B_summary_sha256':file_sha256(extension/'summary.json') if current_B is not None else None,
        'scope':'GPU-slot occupancy from ordinary end-to-end time; not CUDA-active kernel seconds. Margin is a planning allowance, not confidence interval.',
        'limits':['T1 is a planning proxy; historical chat-only numbers retained separately and not pooled as E2E.',
            'Timing extrapolates fixed-cell observations to all nominal cells; model loading, scheduling and imbalance require margin.',
            'Gemma includes bottom16 padding and shared input adapter; not untouched vLLM file70.',
            'Nemo requires explicit legacy CUTLASS and a qualified hardware/runtime contract; unresolved cross-host reproduction remains independent.',
            'No full rerun is launched or authorized by this estimate.']}
    atomic_json(run/'gpu-hours-estimate.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['seal','cpu','analyze','pixel-summary','cost']);parser.add_argument('--root')
    parser.add_argument('--host');parser.add_argument('--profile');args=parser.parse_args()
    value=(seal(args.root,args.host) if args.command=='seal' else cpu(args.profile) if args.command=='cpu'
        else pixel_summary(args.root) if args.command=='pixel-summary' else cost(args.root) if args.command=='cost' else analyze(args.root))
    print(json.dumps(value,sort_keys=True))


if __name__=='__main__':main()
