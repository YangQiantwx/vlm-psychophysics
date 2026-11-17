"""Recompute the paper's tables and plotted model counts from frozen answers."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
from hashlib import file_digest
import json
from pathlib import Path


TASKS = ('T1', 'T2', 'T3', 'T4', 'T5')
SCENES = ('classic', 'natural')
MODELS = ('cosmos', 'nemo', 'qwen', 'gemma', 'glimmer')
CHOICES = {task: {'YES', 'NO'} if task == 'T5' else {'LEFT', 'RIGHT'} for task in TASKS}


def read_json(path):
    return json.loads(path.read_bytes())


def sha256(path):
    with path.open('rb') as source:
        return file_digest(source, 'sha256').hexdigest()


def read_csv(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf8', newline='') as source:
        yield from csv.DictReader(source)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dataset(root, expected_sha):
    path = root / 'dataset-index.json'
    require(sha256(path) == expected_sha, f'Wrong dataset index at {path}')
    catalog = read_json(path)
    videos, images = {}, {}
    for scope, spec in catalog['scopes'].items():
        for file_key, lookup in (('metadata', videos), ('static_metadata', images)):
            path = root / spec[file_key]
            require(sha256(path) == spec[f'{file_key}_sha256'], f'Manifest changed at {path}')
            with path.open(encoding='utf8') as source:
                for line in source:
                    row = json.loads(line)
                    sid = row['sample_id']
                    require(sid not in lookup, (scope, sid, 'duplicate trial'))
                    lookup[sid] = row
    require(videos.keys() == images.keys(), 'Missing video or static associations')
    return videos, images


def factors(row):
    parameters = row.get('parameters', row.get('source_parameters'))
    task = row['task']
    if task == 'T1':
        return (Decimal(str(parameters.get('sf_cycles_per_aperture', parameters.get('sf')))),
                Decimal(str(parameters.get('modulation_depth', parameters.get('m')))))
    if task == 'T2':
        return (Decimal(str(parameters['delta_Y'])),)
    if task == 'T3':
        return (Decimal(str(parameters.get('angle_deg', parameters.get('angle')))),)
    if task == 'T4':
        return (Decimal(str(parameters['coherence'])),)
    return (row['cohort'],)


def public_results(root, videos, images):
    """Parse single-label answers and bind every row to its published trial."""
    scores = {}
    for task in TASKS:
        for row in read_csv(root / 'results' / task.lower() / 'results.csv.gz'):
            sid = row['sample_id']
            v, s = videos[sid], images[sid]
            key = row['model'], row['modality'], sid
            require(row['task'] == task, (key, 'wrong task file'))
            require(row['model'] in MODELS[:-1], (key, 'unknown baseline model'))
            require(row['modality'] in ('video', 'static'), (key, 'unknown baseline modality'))
            require(key not in scores, (key, 'duplicate association'))
            require((row['task'], row['domain'], row['truth'], row['video_sha256']) ==
                    (v['task'], v['domain'], v['truth'], v['video_sha256']), (key, 'video metadata'))
            if row['modality'] == 'static':
                require(json.loads(row['static_image_sha256']) == [i['sha256'] for i in s['images']],
                        (key, 'ordered static images'))
            raw = row['raw_answer'].strip()
            answer = raw if raw in CHOICES[task] else ''
            require(answer == row['prediction'] and int(row['correct']) == (answer == v['truth']),
                    (key, 'scoring mismatch'))
            scores[key] = answer
    require(len(scores) == 8 * len(videos), 'Incomplete four-model result grid')
    return scores


def add_delta(counters, key, correct):
    term = counters[key]
    term[0] += 1
    term[1] += correct


def cell_counts(scores, videos):
    bins = defaultdict(lambda: [0, 0])
    totals = defaultdict(lambda: [0, 0])
    for (model, sid), answer in scores.items():
        row = videos[sid]
        task, scene, truth = row['task'], row['domain'], row['truth']
        key = model, scene, task
        delta = answer == truth
        add_delta(totals, key, delta)
        add_delta(bins, (*key, factors(row)[-1]), delta)
    plots = defaultdict(dict)
    for (m, d, t, level), v in bins.items():
        plots[m, d, t][level] = v
    for key, value in totals.items():
        plots[key]['all'] = value
    return totals, plots


def sustained75(points):
    threshold = None
    for level in sorted((k for k in points if k != 'all'), reverse=True):
        n, correct = points[level]
        if 4 * correct < 3 * n:
            break
        threshold = level
    return threshold


def add_counts(cells):
    return [sum(p[i] for p in cells) for i in (0, 1)]


def equal_weight(cells, model, tasks):
    return sum(cells[model, d, t][1] / cells[model, d, t][0] for d in SCENES for t in tasks) / (2 * len(tasks))


def panel_answers(release, videos, images):
    stills = {s['static_case_id']: s for s in images.values()}
    require(len(stills) == 1346 and len(videos) == 1480, 'Comparison panel denominator')
    answers = {}
    cells = defaultdict(lambda: [0, 0, 0])
    for row in read_csv(release / 'panel-results.csv.gz'):
        model, setting, mode, sid = (row[k] for k in ('model', 'setting', 'modality', 'case_id'))
        r = (videos if mode == 'video' else stills)[sid]
        task, truth = r['task'], r['truth']
        prediction = row['prediction']
        require((row['task'],row['domain'],row['truth']) == (task,r['domain'],truth), (model, sid, 'panel identity'))
        require(prediction in CHOICES[task] | {''}, (model, sid, 'answer choice'))
        require(bool(prediction) == (row['status'] == 'scored'), (model, sid, 'scoring status'))
        key = model, setting, mode, sid
        require(key not in answers, (key, 'duplicate result'))
        answers[key] = prediction
        for slice_ in (task, 'all'):
            cell = cells[model, setting, mode, slice_]
            cell[0] += 1
            cell[1] += prediction == truth
            cell[2] += bool(prediction)
    for (model, setting, mode, task), (n, _, _) in cells.items():
        if task == 'all':
            require(n == (1480 if mode == 'video' else 1346), (model, setting, mode, 'denominator'))
    return answers, cells


def evaluate(base, panel, study):
    manifest = read_json(study / 'manifest.json')
    for path, item in manifest['files'].items():
        candidate = study / path
        if candidate.is_file():
            require(sha256(candidate) == item['sha256'], f'Changed release file {path}')
        else:
            require(not item['required_for_tables'], f'Missing release table input {path}')
    v, s = dataset(base, manifest['base']['index_sha256'])
    p, ps = dataset(panel, manifest['panel']['index_sha256'])
    require((len(v),len(p)) == (25440,1480), 'Unexpected video counts')
    require(p.keys() <= v.keys(), 'Unrecognized subset member')
    require(all(r['video_sha256'] == v[sid]['video_sha256'] and r['truth'] == v[sid]['truth']
                for sid,r in p.items()), 'Panel source video mismatch')
    require(len({x['static_case_id'] for x in s.values()}) == 18795, 'Wrong full independent static count')
    base_answers = public_results(base, v, s)
    subset_answers = public_results(panel, p, ps)
    require(all(base_answers[k] == value for k,value in subset_answers.items()), 'Panel labels differ from the full run')
    all_video = {(m,sid):a for (m,mode,sid),a in base_answers.items() if mode=='video'}
    subset_video = {(m,sid):a for (m,mode,sid),a in subset_answers.items() if mode=='video'}
    muse_calls = set()
    for r in read_csv(study / 'muse-results.csv.gz'):
        mode, sid, gt, answer = (r[k] for k in ('modality','case_id','truth','prediction'))
        require((mode,sid) not in muse_calls, (mode,sid,'duplicate Muse call'))
        muse_calls.add((mode,sid))
        if mode == 'video':
            require((v[sid]['task'],v[sid]['truth']) == (r['task'],gt), (sid,'wrong Muse trial'))
            all_video['glimmer',sid] = answer
            if sid in p:
                subset_video['glimmer',sid] = answer
    require(Counter(mode for mode, _ in muse_calls) == {'video':25440,'static':18795}, 'Incomplete Muse results')
    ff, plots = cell_counts(all_video, v)
    sp, _ = cell_counts(subset_video, p)
    tables = {'table1': {scene:{t:sum(x['task']==t and x['domain']==scene for x in v.values()) for t in TASKS}
                         for scene in SCENES}}
    tables['table2'] = {}
    for m in MODELS[:-1]:
        full = equal_weight(ff,m,TASKS)
        subset = equal_weight(sp,m,TASKS)
        worst = max(abs(equal_weight(sp,m,[t])-equal_weight(ff,m,[t])) for t in TASKS)
        tables['table2'][m] = dict(full=full,subset=subset,delta=subset-full,max_task_delta=worst)
    tables['table3'] = {scene:{m:{t:ff[m,scene,t] for t in TASKS} | {
            'pooled':add_counts([ff[m,scene,t] for t in TASKS]),
            'sustained_75':{t:(float(x) if (x := sustained75(plots[m,scene,t])) is not None else None)
                            for t in ('T2','T3','T4')}}
            for m in MODELS} for scene in SCENES}
    panel_predictions, panel_cells = panel_answers(study,p,ps)
    model_configs = (('astra','low'), ('opus-5.5','low'), ('kimi-k3.1-pro','low'), ('gemini-4-argon','low'),
                     ('qwen','png-low'), ('molmo','direct'))
    tables['table4'] = {}
    for m in MODELS:
        vals = {t:add_counts([sp[m,d,t] for d in SCENES]) for t in TASKS}
        tables['table4'][m] = vals | {'pooled':add_counts(list(vals.values()))}
    for m, arm in model_configs:
        tables['table4'][f'{m}:{arm}'] = {t:panel_cells[m,arm,'video',t][:2] for t in TASKS} | {
            'pooled':panel_cells[m,arm,'video','all'][:2]}
    human = read_json(study / 'human-figure6.json')
    require(human['clips']==1478 and human['observers']==4 and len(human['cells'])==134, 'Human aggregates')
    observer_counts = defaultdict(lambda:[0,0])
    for r in human['totals']:
        observer_counts[r['task']][0] += r['clips'] * human['observers']
        observer_counts[r['task']][1] += r['human_correct_sum']
    tables['table4']['human'] = dict(observer_counts) | {'pooled':add_counts(list(observer_counts.values()))}
    tables['qwen_video_reasoning'] = {arm:{t:panel_cells['qwen',arm,'video',t] for t in (*TASKS,'all')}
                                     for arm in ('off','low','png-off','png-low')}
    tables['qwen_static_reasoning'] = {arm:{t:panel_cells['qwen',arm,'static',t] for t in (*TASKS,'all')}
                                      for arm in ('off','low')}
    return tables, plots, p, panel_predictions


def main():
    ap = ArgumentParser(description=__doc__)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--panel', type=Path, required=True)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--figures', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    tables, plots, panel, predictions = evaluate(args.base,args.panel,args.study)
    (args.out/'tables.json').write_text(json.dumps(tables, indent=2) + '\n')
    if args.figures:
        from .figures import render
        render(args, plots, panel, predictions)
    print(f'Recomputed Tables 1–4 from {args.base}, {args.panel} and {args.study}.')


if __name__ == '__main__':
    main()
