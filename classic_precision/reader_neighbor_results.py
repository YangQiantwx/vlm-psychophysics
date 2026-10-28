"""Read-only audit of 1152 official-route neighbor answers against frozen history.

Does not launch models, replay claims, resolve the old Nemo hold, or treat
historical controlled answers as newly matched runtime measurements.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .cosmos_neighbor_smoke import local_path, records, analyze as cosmos_audit
from .reader_route_results import stats
from .refinement_worker import read, require
from .storage import atomic_json, digest, file_sha256

NAME = 'reader-neighbors-extension-20261027'
PREVIOUS = 'cosmos-t3-neighbors-20261027'
COUNTS = ('A_historical_correct', 'B_correct', 'changed', 'correct_to_wrong', 'wrong_to_correct', 'tokens_changed')


def bound(root, ref):
    marker = '/code/vlm-psychophysics/'
    path = (Path(__file__).parents[1]/ref['path'].split(marker, 1)[1]
            if marker in ref['path'] else local_path(root, ref['path']))
    require(file_sha256(path) == ref['sha256'], 'Bound file changed: '+str(path))
    return read(path) if path.suffix == '.json' else path


def csv_save(path, rows):
    if not rows: return
    fields = sorted({k for row in rows for k in row})
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def summary_rows(rows):
    return {'n': len(rows), **{key: sum(r[key] for r in rows) for key in COUNTS}}


def audit_worker(root, profile_path, history, catalog, extension):
    p = read(profile_path); model = p['model']; out = local_path(root, p['output'])
    require(p['arm'] == 'C' and p['user_route'] == 'B', 'Not the authorized B route')
    require(not p['automatic_retry'] and not p['formal_auto_resume'], 'Unexpected retry/resume')
    for name in ('source_catalog', 'allocation', 'experiment', 'implementation'):
        bound(root, p[name])
    if extension:
        from .reader_neighbor_extension import SCIENCE
        parent = bound(root, p['qualified_neighbor_parent']); bound(root, p['neighbor_entry'])
        require({k:p[k] for k in SCIENCE if k in p} == {k:parent[k] for k in SCIENCE if k in parent}, 'Scientific parent drift')
        if model == 'nemo':
            from .nemo_kernel_replay import validate_inventory
            for ref in p['legacy_kernel_context'].values(): bound(root, ref)
            validate_inventory(read(out/'restored-kernels.json'))
    wait = read(out.with_suffix('.wait.json')); terminal = read(out/'result.json')
    require(wait['returncode'] == 0 and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released'] and wait['terminal_status'] == 'COMPLETE', 'Incomplete worker closure')
    ids = p['assigned_ids']; require(len(ids) == len(set(ids)) == p['maximum_model_calls'], 'Quota mismatch')
    require(wait['profile_sha256'] == digest(p) and wait['completed_ids'] == ids and terminal['completed_ids'] == ids
            and not terminal['UNCERTAIN_ids'] and not terminal['unattempted_ids'], 'Terminal identity mismatch')
    require(read(out/'identity.json')['profile_sha256'] == digest(p), 'Execution identity mismatch')
    raw = records(out/'raw.jsonl'); events = records(out/'events.jsonl'); times = records(out/'request-timings.jsonl')
    passes = [r for r in events if r['event'] == 'TECHNICAL_PASS']
    require([r['sample_id'] for r in raw] == [r['sample_id'] for r in times] == [r['sample_id'] for r in passes] == ids, 'ID ledger mismatch')
    by_pass = {r['sample_id']: r for r in passes}; paired = []; observations = []
    for r in raw:
        sid = r['sample_id']; source = catalog['items'][sid]; old = history[model, sid]
        require(r['profile_sha256'] == digest(p) and by_pass[sid]['raw_digest'] == digest(r), 'Raw/PASS identity')
        require(r['model'] == model and r['arm'] == 'C' and r['attempt'] == 1, 'Unexpected model/arm/attempt')
        require(r['truth'] == old['truth'] == source['row']['truth'] and r['question'] == source['row']['prompt'], 'GT or question drift')
        require(r['source']['video_sha256'] == old['source_video_sha256'] == source['ready']['video_sha256']
                and r['source']['row_sha256'] == source['row_sha256']
                and r['source']['stream_sha256'] == source['source_stream_sha256'], 'Source identity drift')
        claim = read(local_path(root, p['claim_root'])/(sid+'.json'))
        require(claim['profile_sha256'] == digest(p) and claim['attempt'] == 1 and claim['worker'] == p['worker'], 'Claim identity')
        g = r['generation']
        require(g['temperature'] == 0 and g['top_p'] == 1 and g['seed'] == 0 and not g['thinking']
                and not g['ignore_eos'] and not g['native_EOS_override'] and not g['stop'] and not g['stop_token_ids']
                and g['choice'] == ['LEFT','RIGHT'] and g['max_tokens'] == 16384, 'Generation drift')
        answer = r['raw_answer'].strip(); truth = r['truth']; a = old['answer']; sp = source['row']['source_parameters']
        require(answer in ['LEFT','RIGHT'] and r['finish_reason'] == 'stop', 'Invalid answer completion')
        point = sp['delta_Y'] if r['task'] == 'T2' else sp['angle_deg']
        paired.append({'model':model, 'task':r['task'], 'point':point, 'sample_id':sid, 'truth':truth,
            'tilt_sign':sp.get('tilt_sign',''), 'onset_index':sp['onset_index'], 'A_historical':a, 'B':answer,
            'A_historical_correct':a == truth, 'B_correct':answer == truth, 'changed':a != answer,
            'correct_to_wrong':a == truth and answer != truth, 'wrong_to_correct':a != truth and answer == truth,
            'tokens_changed':json.loads(old['output_token_ids']) != r['output_token_ids'],
            'finish_changed':old['finish_reason'] != r['finish_reason'],
            'historical_stop_recorded':old.get('stop_reason','') != '',
            'historical_raw_digest':old['raw_sha256'], 'B_raw_digest':digest(r),
            'worker':p['worker'], 'new_raw_file':str(out/'raw.jsonl'), 'new_raw_ordinal':r['ordinal'],
            'historical_runtime_matched_this_call':False,
            'Nemo_historical_runtime_gap_deferred':model == 'nemo'})
        if r['observed_or_cold']:
            require(by_pass[sid]['actual_GPU_input'], 'Missing first-task GPU observation')
            observations.append({'worker':p['worker'],'sample_id':sid,'task':r['task'], 'reader':r['reader'],
                'native_reader':r['native_reader'],'native_processor':r['native_processor'], 'input_controls':r['input_controls'],
                'actual_GPU_input':by_pass[sid]['actual_GPU_input'], 'prompt_tokens':len(r['actual_prompt_token_ids']),
                'prompt_token_sha256':digest(r['actual_prompt_token_ids'])})
    require({r['task'] for r in observations} == {r['task'] for r in raw}, 'Missing task observation')
    timing_rows = [{**t,'model':model,'worker':p['worker'], 'point':catalog['items'][t['sample_id']]['row']['source_parameters'][
        'delta_Y' if t['task'] == 'T2' else 'angle_deg']} for t in times]
    return paired, timing_rows, observations, {'worker':p['worker'], **wait, 'load':read(out/'load.json')}


def analyze(root):
    root = Path(root); plan = root/'plans'/NAME; out = root/'runs/reproducibility'/NAME
    exp = read(plan/'plan.json'); require(exp['maximum_new_scientific_calls'] == 960 and exp['historical_A_reuse'], 'Wrong scope')
    for name, sha in exp['files'].items(): require(file_sha256(plan/name) == sha, 'Frozen input changed')
    allocation = read(plan/'allocation.json'); catalog = read(plan/'sources.json')
    old_rows = list(csv.DictReader((plan/'historical.csv').open())); history = {(r['model'],r['sample_id']):r for r in old_rows}
    require(len(old_rows) == len(history) == 1152, 'Historical fixed-ID scope')
    cosmos_audit(root)  # Independently closes the already-run 192 A + 192 B, preserving all of them.
    paired = []; timings = []; observations = []; closures = []; claim_sets = {}
    for dirname in [PREVIOUS, NAME]:
        for card in ['fourier0','fourier1','radon0']:
            queue = read(root/'plans'/dirname/(card+'-queue.json'))
            require(queue['status'] == 'COMPLETE', 'Queue unfinished: '+card+'/'+dirname)
            for remote in queue['profiles']:
                profile = local_path(root,remote); p = read(profile)
                if p['arm'] != 'C': continue
                if dirname == NAME:
                    segment = next(x for x in allocation['cards'][card] if x['model'] == p['model'])
                    require(segment['arms'] == ['C'] and segment['sample_ids'] == p['assigned_ids'], 'Allocation drift')
                    gate = read(plan/(card+'-predecessor-accepted.json'))
                    require(gate['status'] == 'PASS', 'Missing predecessor acceptance')
                rows, ts, obs, closure = audit_worker(root, profile, history, catalog, dirname == NAME)
                paired.extend(rows); timings.extend(ts); observations.extend(obs); closures.append(closure)
                # Host roots merge into one local journal directory. Union the
                # disjoint host assignments before checking for extra claims.
                claim_sets.setdefault(str(local_path(root,p['claim_root'])),set()).update(p['assigned_ids'])
    for path, ids in claim_sets.items():
        require({p.stem for p in local_path(root,path).glob('*.json')} == ids, 'Extra/missing/unclosed claims')
    require(len(paired) == len({(r['model'],r['sample_id']) for r in paired}) == 1152
            and {(r['model'],r['sample_id']) for r in paired} == set(history), 'Incomplete/duplicate B scope')
    cells = []; strata = []; timing_stats = []
    for model in ['qwen','cosmos','gemma','nemo']:
        for task,points in [('T2',[.05,.07]),('T3',[.6,.8])]:
            for point in points:
                rr = [r for r in paired if (r['model'],r['task'],r['point']) == (model,task,point)]
                require(len(rr) == (48 if task == 'T2' else 96), 'Cell denominator')
                ordinary = [t for t in timings if (t['model'],t['task'],t['point']) == (model,task,point) and not t['observed_or_cold']]
                e2e = stats([t['end_to_end_durable_seconds'] for t in ordinary])
                cells.append({'model':model,'task':task,'point':point,**summary_rows(rr),
                    'B_ordinary_E2E_seconds':e2e, 'B_throughput_per_hour':3600/e2e['mean'] if e2e['n'] else None})
                for field in ['truth','tilt_sign','onset_index']:
                    if field == 'tilt_sign' and task == 'T2': continue
                    for value in sorted({r[field] for r in rr}):
                        sr = [r for r in rr if r[field] == value]
                        strata.append({'model':model,'task':task,'point':point,'stratum':field,'value':value,**summary_rows(sr)})
                for stage in sorted({k for t in ordinary for k,v in t.items() if k.endswith('_seconds') and isinstance(v,(int,float))}):
                    timing_stats.append({'model':model,'task':task,'point':point,'stage':stage,
                        'nested_not_additive':stage.startswith('native_'),**stats([t[stage] for t in ordinary if isinstance(t.get(stage),(int,float))])})
    result = {'status':'COMPLETE_DESCRIPTIVE_COMPARISON','B_answers':1152,'new_extension_calls':960,
        'preserved_Cosmos_A_calls':192,'cells':cells,'totals':summary_rows(paired),'B_worker_closures':closures,
        'history_sha256':file_sha256(plan/'historical.csv'),'plan_sha256':file_sha256(plan/'plan.json'),
        'interpretation':'Frozen historical A versus new official-route B. A was not rerun for this extension; Nemo cross-host gap explicitly deferred. Not a uint8-only causal experiment or strict all-history reproduction claim.',
        'timing_scope':'B same-worker durable E2E only; native nested stages not additive. No new A/B speedup claim from historical timing.',
        'formal_auto_resume':False,'full_rerun_authorized':False}
    atomic_json(out/'summary.json',result);atomic_json(out/'input-observations.json',observations)
    flat = [{k:v for k,v in c.items() if k != 'B_ordinary_E2E_seconds'} | {
        'B_mean_seconds':c['B_ordinary_E2E_seconds']['mean'],'B_median_seconds':c['B_ordinary_E2E_seconds']['median'],
        'B_p90_seconds':c['B_ordinary_E2E_seconds']['p90']} for c in cells]
    for name,rows in [('comparison.csv',flat),('paired.csv',paired),('strata.csv',strata),('timing.csv',timings),('timing-summary.csv',timing_stats)]:
        csv_save(out/name,rows)
    return {'status':result['status'],'B_answers':1152,'cells':flat}


def status(root):
    from . import reader_route_results as reporter
    previous = reporter.NAME; reporter.NAME = NAME
    try: return reporter.status(root)
    finally: reporter.NAME = previous


def three_point(root):
    """Add accepted center-point C answers as user B; no scientific calls."""
    root=Path(root);out=root/'runs/reproducibility'/NAME
    accepted=read(out/'summary.json')
    require(accepted['status']=='COMPLETE_DESCRIPTIVE_COMPARISON' and accepted['B_answers']==1152,'Neighbors not accepted')
    center=root/'runs/reproducibility/reader-route-smoke-20261026'
    old=read(center/'partial/summary.json')
    require(old['technical_pass']==1728 and old['canonical_runtime_restoration']['status']=='COMPLETE_PASS','Center technical acceptance missing')
    cplan=root/'plans/reader-route-smoke-20261026'
    for name,sha in old['bindings'].items():require(file_sha256(cplan/name)==sha,'Center binding changed')
    sources=read(cplan/'sources.json')['items']
    history={(r['model'],r['sample_id']):r for r in csv.DictReader((cplan/'historical.csv').open())}
    center_pairs=list(csv.DictReader((center/'partial/paired.csv').open()))
    center_by_key={(r['model'],r['sample_id']):r for r in center_pairs}
    require(len(center_pairs)==len(center_by_key)==576,'Wrong center scope')
    raw_by_key={};raw_files={}
    for logical in ['qwen-fourier1-C','cosmos-fourier1-C','cosmos-radon0-C','gemma-fourier0-C','nemo-radon0-C']:
        worker=old['logical_arm_replacements'].get(logical,logical);path=center/worker/'raw.jsonl'
        wait=read((center/worker).with_suffix('.wait.json'))
        require(wait['returncode']==0 and wait['real_wait_observed'] and wait['owned_engines_exited']
            and wait['original_physical_lease_released'],'Center worker not closed')
        for raw in records(path):
            key=(raw['model'],raw['sample_id']);require(key not in raw_by_key and raw['arm']=='C','Duplicate/non-C center')
            raw_by_key[key]=raw;raw_files[key]=str(path)
    require(set(raw_by_key)==set(center_by_key),'Center raw coverage')
    rows=[]
    for row in csv.DictReader((out/'paired.csv').open()):
        row=dict(row);row['point']=float(row['point']);row['onset_index']=int(row['onset_index'])
        for key in COUNTS:row[key]=row[key]=='True'
        row['point_role']='neighbor';rows.append(row)
    for key,entry in center_by_key.items():
        model,sid=key;raw=raw_by_key[key];source=sources[sid];sp=source['row']['source_parameters'];h=history[key]
        task=entry['task'];point=.06 if task=='T2' else .7
        require((sp['delta_Y']==.06 and sp['background_Y']==.4 and sp['offset_frame']-sp['onset_frame']==30)
            if task=='T2' else sp['angle_deg']==.7,'Wrong midpoint')
        require(raw['raw_answer'].strip()==entry['C_answer'] and entry['historical_answer']==h['answer']
            and raw['truth']==entry['truth']==h['truth'],'Center answer/GT mapping')
        require(raw['source']['video_sha256']==entry['source_video_sha256']==source['ready']['video_sha256'],'Center source mapping')
        a=h['answer'];b=entry['C_answer'];truth=entry['truth']
        rows.append({'model':model,'task':task,'point':point,'point_role':'center','sample_id':sid,'truth':truth,
            'tilt_sign':entry['tilt_sign'],'onset_index':int(entry['onset_index']),'A_historical':a,'B':b,
            'A_historical_correct':a==truth,'B_correct':b==truth,'changed':a!=b,
            'correct_to_wrong':a==truth and b!=truth,'wrong_to_correct':a!=truth and b==truth,
            'tokens_changed':json.loads(h['output_token_ids'])!=raw['output_token_ids'],
            'B_raw_digest':digest(raw),'new_raw_file':raw_files[key],'new_raw_ordinal':raw['ordinal'],
            'historical_runtime_matched_this_call':False,'Nemo_historical_runtime_gap_deferred':model=='nemo'})
    require(len(rows)==len({(r['model'],r['sample_id'])for r in rows})==1728,'Combined paired scope')
    cells=[]
    for model in ['qwen','cosmos','gemma','nemo']:
        for task,points in [('T2',[.05,.06,.07]),('T3',[.6,.7,.8])]:
            for point in points:
                rr=[r for r in rows if (r['model'],r['task'],r['point'])==(model,task,point)]
                require(len(rr)==(48 if task=='T2' else 96),'Three-point denominator')
                cells.append({'model':model,'task':task,'point':point,'point_role':rr[0]['point_role'],**summary_rows(rr)})
    result={'status':'COMPLETE_DESCRIPTIVE_THREE_POINT_VIEW','new_model_calls':0,'paired_rows':1728,
        'cells':cells,'totals':summary_rows(rows),'neighbor_summary_sha256':file_sha256(out/'summary.json'),
        'center_summary_sha256':file_sha256(center/'partial/summary.json'),
        'center_paired_sha256':file_sha256(center/'partial/paired.csv'),
        'route_mapping':'A=historical controlled throughout; B=official native route, original center internal C. Original center reader-only B is NOT used.',
        'Nemo_limit':'Historical replay gap remains deferred; original center controlled A27 is not substituted for historical28.',
        'formal_auto_resume':False}
    csv_save(out/'three-point-comparison.csv',cells);csv_save(out/'three-point-paired.csv',rows)
    atomic_json(out/'three-point-summary.json',result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--status',action='store_true');mode.add_argument('--with-midpoints',action='store_true')
    args = parser.parse_args();print(json.dumps(status(args.root) if args.status else three_point(args.root) if args.with_midpoints else analyze(args.root),sort_keys=True))


if __name__ == '__main__': main()
