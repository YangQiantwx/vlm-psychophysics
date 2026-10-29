"""Read-only, fail-closed reconciliation of the sealed Cosmos X full run.

Reuses the scientific checks of official_x_results; never imports inference.
Historical A is joined only where source bytes, task and ground truth match.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path

from .official_x_results import (Archive, audit_worker, basic, descriptor, read,
                                 require, timing_tables, write_csv)
from .storage import atomic_json, file_sha256, digest

RUN = 'cosmos-x-full-20261028'
EXPECTED = {'T1': 3840, 'T2': 2880, 'T3': 2736}


def cell_key(row):
    p, task = row['parameters'], row['task']
    if task == 'T1':
        return (task, p['sf_cycles_per_aperture'], p['modulation_depth'])
    if task == 'T2':
        duration = p['offset_frame'] - p['onset_frame']
        require(duration == 30, 'Excluded T2 duration entered main scope')
        return (task, p['background_Y'], duration, p['delta_Y'])
    require(task == 'T3' and p['angle_deg'] not in (1.75, 2.25), 'Excluded task/angle')
    return (task, p['angle_deg'])


def pair_stats(rows):
    n = len(rows)
    return {'n': n, 'A_correct': sum(r['A_correct'] for r in rows),
            'X_correct': sum(r['X_correct'] for r in rows),
            'changed': sum(r['A'] != r['X'] for r in rows),
            'correct_to_wrong': sum(r['A_correct'] and not r['X_correct'] for r in rows),
            'wrong_to_correct': sum(not r['A_correct'] and r['X_correct'] for r in rows),
            'full_output_token_changes': sum(r['output_tokens_changed'] for r in rows)}


def analyze(root):
    root = Path(root)
    archive = Archive(root)
    plans = root / 'plans' / RUN
    output = root / 'runs/reproducibility' / RUN / 'analysis'
    plan = read(plans / 'plan.json')
    require(plan['counts'] == EXPECTED and plan['maximum_new_model_calls'] == 9456,
            'Unexpected authorized full-run scope')
    records, evidence, assigned = [], [], []
    for card in ('fourier0', 'fourier1', 'radon0'):
        profile_path = plans / (card + '-profile.json')
        p = read(profile_path)
        require(p['assigned_ids'] == plan['allocation'][card]['ids'], 'Profile/allocation mismatch')
        assigned.extend(p['assigned_ids'])
        spec = {'profile': descriptor(profile_path), 'phase': 'full', 'variant': 'X'}
        rows, proof = audit_worker(archive, spec)
        print(json.dumps({'worker': p['worker'], 'accepted': len(rows)}), flush=True)
        records.extend(rows)
        evidence.append(proof)
    require(len(assigned) == len(set(assigned)) == 9456, 'Overlapping allocation')
    require(len(records) == 9456 and {r['sample_id'] for r in records} == set(assigned),
            'Full union has missing or duplicate IDs')
    require(dict(Counter(r['task'] for r in records)) == EXPECTED, 'Task counts differ')
    require(all(r['eligible'] and r['include_in_main_accuracy_curve'] for r in records),
            'Nonprimary record included')
    cells, strata, task_rows = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        r['cell'] = list(cell_key(r))
        cells[tuple(r['cell'])].append(r)
        task_rows[r['task']].append(r)
        strata[(r['task'], 'truth', r['truth'])].append(r)
        for key in ('phase_degrees', 'onset_index', 'tilt_sign'):
            if key in r['parameters']:
                strata[(r['task'], key, r['parameters'][key])].append(r)
    require(len(cells) == 169 and Counter(k[0] for k in cells) == {'T1':80, 'T2':60, 'T3':29},
            'Incomplete main configuration grid')
    cell_rows = []
    for key, rows in sorted(cells.items()):
        n = 96 if key[0] == 'T3' and key[1] != 90 else 48
        require(len(rows) == n and Counter(r['truth'] for r in rows) == {'LEFT':n//2,'RIGHT':n//2},
                'Unbalanced or incomplete cell: ' + str(key))
        cell_rows.append({'cell':list(key), **basic(rows)})

    a_folder = root / 'runs/reproducibility/current-nvfp4'
    a_summary = read(a_folder / 'summary.json')
    a_path = a_folder / 'video-predictions.csv'
    require(a_summary['status'] == 'COMPLETE' and file_sha256(a_path) == a_summary['video_csv_sha256'],
            'Historical A export identity differs')
    with a_path.open() as f:
        a_rows = [r for r in csv.DictReader(f) if r['model'] == 'cosmos']
    a_lookup = {r['sample_id']:r for r in a_rows}
    require(len(a_rows) == len(a_lookup) == 9360, 'Historical Cosmos A ledger differs')
    paired = []
    for x in records:
        a = a_lookup.get(x['sample_id'])
        if a is None:
            continue
        require(a['protocol'] == 'controlled-rgb16' and a['task'] == x['task'] and a['truth'] == x['truth'],
                'Historical A task/GT/protocol differs')
        require(a['source_video_sha256'] in [sha for kind,sha in x['source_media'] if kind=='video'],
                'Historical A source video differs')
        a_correct = a['answer'] == a['truth']
        require(a_correct == bool(int(a['correct'])), 'Historical A score differs from label')
        paired.append({'sample_id':x['sample_id'], 'task':x['task'], 'cell':x['cell'],
                       'truth':x['truth'], 'A':a['answer'], 'X':x['answer'],
                       'A_correct':a_correct, 'X_correct':x['correct'],
                       'output_tokens_changed':json.loads(a['output_token_ids']) != x['output_token_ids'],
                       'source_video_sha256':a['source_video_sha256'], 'A_raw_sha256':a['raw_sha256'],
                       'X_raw_digest':x['raw_digest'], 'A_reference_file':a['reference_file'],
                       'A_reference_row':a['reference_row'], 'X_worker':x['worker']})
    paired_cells = defaultdict(list)
    for r in paired:
        paired_cells[tuple(r['cell'])].append(r)
    comparisons = [{'task':task, **pair_stats([r for r in paired if r['task']==task])}
                   for task in EXPECTED]
    timing = []
    for worker in sorted({r['worker'] for r in records}):
        timing.extend({'worker':worker, **r} for r in timing_tables([r for r in records if r['worker']==worker]))
    summary = {'schema':'cosmos-X-full-results-v1', 'status':'COMPLETE_PASS',
               'completed_HST':datetime.now().astimezone().isoformat(),
               'model':'cosmos', 'precision':'official-mixed-NVFP4', 'protocol':'X',
               'unique_videos':len(records), 'configs':len(cells), 'new_model_calls':0,
               'tasks':{task:basic(rows) for task,rows in task_rows.items()},
               'A_X_paired':comparisons, 'A_unavailable_in_X_scope':len(records)-len(paired),
               'workers':evidence, 'plan':descriptor(plans/'plan.json'),
               'historical_A':descriptor(a_path),
               'input':{'frames':180,'effective_width':768,'effective_height':512,
                        'reader':'official OpenCV uint8','resize':False,'secondary_sampling':False,
                        'actual_GPU_deep_admissions':sum(r['deep'] for r in records)},
               'limitations':['Historical A and X differ in runtime stack and configured backend policy as well as input route.',
                              'A/X accuracy is paired only on shared source IDs; incremental cells without A are not imputed.',
                              'Timing is per worker; native reader and processor are nested inside engine-inclusive time.',
                              'Single full run is not a repeated full-run or universal cross-hardware reproducibility test.']}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output/'canonical-records.json', records)
    write_csv(output/'predictions.csv', records)
    write_csv(output/'cells.csv', cell_rows)
    write_csv(output/'paired-A-X.csv', paired)
    write_csv(output/'comparison-A-X.csv', comparisons)
    write_csv(output/'cells-A-X.csv', [{'cell':list(k), **pair_stats(rows)} for k,rows in sorted(paired_cells.items())])
    write_csv(output/'strata.csv', [{'task':k[0],'stratum':k[1],'value':k[2],**basic(rows)} for k,rows in sorted(strata.items(),key=lambda x:str(x[0]))])
    write_csv(output/'timing.csv', timing)
    summary['outputs'] = {p.name:descriptor(p) for p in output.iterdir() if p.name != 'summary.json' and p.is_file()}
    atomic_json(output/'summary.json', summary)
    return {k:summary[k] for k in ('status','unique_videos','configs','tasks','A_X_paired','A_unavailable_in_X_scope')}


def compare_prior_x(root):
    """Descriptive overlap check; differences never trigger exclusion or retries."""
    root = Path(root)
    out = root / 'runs/reproducibility' / RUN / 'analysis'
    current = read(out/'summary.json')
    require(current['status']=='COMPLETE_PASS', 'Full audit required')
    full_path = out/'canonical-records.json'
    require(descriptor(full_path)==current['outputs']['canonical-records.json'], 'Full records changed')
    full = {r['sample_id']:r for r in read(full_path)}
    prior_dir = root/'runs/reproducibility/official-x-20261028/analysis'
    prior_summary = read(prior_dir/'summary.json')
    require(prior_summary['status']=='COMPLETE_FINITE_STUDY', 'Prior finite study not accepted')
    prior = [r for r in read(prior_dir/'canonical-records.json')
             if r['model']=='cosmos' and r['phase']=='validation' and r['replicate']==1]
    require(len(prior)==len({r['sample_id'] for r in prior})==144, 'Prior scope differs')
    proofs = {r['worker']:r for r in prior_summary['workers']}
    archive = Archive(root)
    for worker in {r['worker'] for r in prior}:
        desc = proofs[worker]['raw'];path=archive.path(desc['path'])
        require(file_sha256(path)==desc['sha256'], 'Prior raw changed')
        with path.open() as f:raws={r['sample_id']:r for r in map(json.loads,f)}
        for r in (x for x in prior if x['worker']==worker):
            raw=raws[r['sample_id']]
            require(digest(raw)==r['raw_digest'] and raw['output_token_ids']==r['output_token_ids']
                    and raw['actual_prompt_token_ids'] and digest(raw['actual_prompt_token_ids'])==r['prompt_token_sha256']
                    and raw['raw_answer'].strip()==r['answer'], 'Prior exported answer differs from raw')
    rows=[]
    for p in prior:
        x=full[p['sample_id']]
        require(p['source_media']==x['source_media'] and p['truth']==x['truth'] and p['task']==x['task'],
                'Prior/current source or ground truth differs')
        rows.append({'sample_id':p['sample_id'],'task':p['task'],'prior_worker':p['worker'],
                     'full_worker':x['worker'],'prior_answer':p['answer'],'full_answer':x['answer'],
                     'answer_equal':p['answer']==x['answer'],
                     'output_tokens_equal':p['output_token_ids']==x['output_token_ids'],
                     'prompt_tokens_equal':p['prompt_token_sha256']==x['prompt_token_sha256'],
                     'finish_stop_equal':(p['finish_reason'],p['stop_reason'])==(x['finish_reason'],x['stop_reason'])})
    result={'schema':'cosmos-X-full-prior-overlap-v1','n':144,'new_model_calls':0,
            **{k:sum(r[k] for r in rows) for k in ['answer_equal','output_tokens_equal','prompt_tokens_equal','finish_stop_equal']},
            'prior_summary':descriptor(prior_dir/'summary.json'),'prior_records':descriptor(prior_dir/'canonical-records.json'),
            'scope':'Previously selected 144 Cosmos X validation sources compared with their full-run answers; cards may differ.',
            'differences':[r for r in rows if not all(r[k] for k in ['answer_equal','output_tokens_equal','prompt_tokens_equal','finish_stop_equal'])]}
    atomic_json(out/'prior-X-overlap.json',result)
    write_csv(out/'prior-X-overlap.csv',rows)
    current['prior_X_overlap']={k:v for k,v in result.items() if k!='differences'}
    for name in ['prior-X-overlap.json','prior-X-overlap.csv']:current['outputs'][name]=descriptor(out/name)
    atomic_json(out/'summary.json',current)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True)
    p.add_argument('--compare-prior-x', action='store_true')
    args = p.parse_args()
    print(json.dumps(compare_prior_x(args.root) if args.compare_prior_x else analyze(args.root),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
