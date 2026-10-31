"""Read-only release of audited Qwen classic X records; retain missing claims.

This never launches inference, changes a scientific ledger, or admits a raw run.
Its inputs are the already accepted T13 archive and the completed T2 archive.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .cosmos_x_full_results import cell_key, pair_stats
from .official_x_results import basic, timing_tables, write_csv
from .refinement_worker import descriptor, digest, file_sha256, read, require
from .storage import atomic_json

NAME = 'qwen-x-remaining-t13-20261030'
MISSING = 't3_2e29ac8a4e90ee6c0f7f'


def bound_records(folder):
    summary = read(folder / 'summary.json')
    path = folder / 'canonical-records.json'
    rows = read(path)
    if 'outputs' in summary:
        require(file_sha256(path) == summary['outputs'][path.name]['sha256'],
                'Accepted canonical records changed: ' + str(path))
    else:
        # The earlier T2 summary binds original raw journals, not its export.
        # Check that export against those bindings without rerunning input,
        # source-video, lifecycle or GPU audits. Stream one request at a time.
        require(summary['status'] == 'COMPLETE_CLASSIC_QWEN_T2_X', 'Unknown archive schema')
        def projection(proof):
            source = Path(proof['raw']['path'])
            require(file_sha256(source) == proof['raw']['sha256'], 'Accepted T2 journal changed')
            result = {}
            with source.open() as f:
                for line in f:
                    raw = json.loads(line)
                    sid = raw['sample_id']
                    require(sid not in result, 'Repeated raw ID')
                    result[sid] = (digest(raw), raw['task'], raw['truth'], raw['raw_answer'],
                                   raw['output_token_ids'], raw['finish_reason'], raw['stop_reason'], raw['correct'])
            return result
        with ThreadPoolExecutor(max_workers=3) as pool:
            maps = list(pool.map(projection, summary['worker_evidence']))
        original = {k: v for m in maps for k, v in m.items()}
        require(len(original) == sum(map(len, maps)) == len(rows)
                and set(original) == {r['sample_id'] for r in rows}, 'T2 export scope differs')
        for r in rows:
            require((r['raw_digest'], r['task'], r['truth'], r['raw_answer'], r['output_token_ids'],
                     r['finish_reason'], r['stop_reason'], r['correct']) == original[r['sample_id']],
                    'T2 canonical projection differs')
        require(basic(rows) == summary['overall'], 'T2 accepted totals differ')
    return summary, rows


def identity(a, b):
    require(a['task'] == b['task'] and a['truth'] == b['truth']
            and a['parameters'] == b['parameters']
            and list(map(tuple, a['source_media'])) == list(map(tuple, b['source_media'])),
            'Comparison source/task/GT mismatch: ' + a['sample_id'])


def render(cells, pair_cells, between, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sf = sorted({key[1] for key in cells if key[0] == 'T1'})
    bg = sorted({key[1] for key in cells if key[0] == 'T2'})
    panels = [('T1', v) for v in sf] + [('T2', v) for v in bg] + [('T3', None)]
    require(len(panels) == 12, 'Unexpected overview layout')
    colors = {'A': '#dc6a32', 'X': '#2867ac', 'Cosmos': '#31916e',
              'Nemo': '#9a5ab6', 'Qwen': '#2867ac'}
    for mode, title in [('paired-A-X', 'Qwen: A and X on the same video IDs'),
                        ('shared-three-model-X', 'Protocol X: 9,455 shared videos')]:
        fig, axes = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
        for ax, (task, cond) in zip(axes.flat, panels):
            keys = sorted(k for k in cells if k[0] == task and (task == 'T3' or k[1] == cond))
            series = ['A', 'X'] if mode == 'paired-A-X' else ['Cosmos', 'Nemo', 'Qwen']
            for name in series:
                x, y = [], []
                for key in keys:
                    rows = pair_cells.get(key, []) if mode == 'paired-A-X' else between[key]
                    if not rows:
                        continue
                    x.append(key[-1])
                    y.append(100 * sum(r[name + '_correct'] for r in rows) / len(rows))
                ax.plot(x, y, 'o-', ms=3, lw=1.5, color=colors[name], label=name)
            ax.axhline(50, color='#aaa', ls=':', lw=1)
            ax.set_ylim(0, 103)
            ax.grid(alpha=.16)
            ax.set_title(('T1 · SF ' + str(cond)) if task == 'T1' else
                         ('T2 · BG ' + str(cond)) if task == 'T2' else 'T3 · orientation')
            ax.set_xlabel({'T1': 'Modulation depth', 'T2': 'Brightness increment',
                           'T3': 'Angle (degrees)'}[task])
            ax.set_ylabel('Accuracy (%)')
            if task == 'T3':
                ax.set_xscale('log')
        axes.flat[0].legend(frameon=False)
        fig.suptitle(title + '\nOne reboot-interrupted Qwen video remains unscored', fontsize=16)
        for ext in ('png', 'pdf'):
            fig.savefig(out / (mode + '.' + ext), dpi=180)
        plt.close(fig)


def analyze(root):
    root = Path(root)
    runs = root / 'runs/reproducibility'
    t13dir = runs / NAME / 'analysis'
    t2dir = runs / 'qwen-x-classic-t2-20261029'
    s13, r13 = bound_records(t13dir)
    s2, r2 = bound_records(t2dir)
    require(s13['status'] == 'PARTIAL_CLASSIC_QWEN_T13_X_UNRESOLVED_CLAIM'
            and s13['missing_ids'] == [MISSING] and len(r13) == 6575,
            'This release requires the explicit one-claim partial archive')
    require(s2['status'] == 'COMPLETE_CLASSIC_QWEN_T2_X' and len(r2) == 2880,
            'T2 archive is incomplete')
    records = [dict(r, model='qwen', variant='X') for r in r13 + r2]
    for row in records:
        row.setdefault('phase', 'classic_' + row['task'])
        row.setdefault('replicate', None)
    lookup = {r['sample_id']: r for r in records}
    require(len(lookup) == len(records) == 9455 and MISSING not in lookup,
            'Duplicate or missing scope differs')
    require(Counter(r['task'] for r in records) == {'T1': 3840, 'T2': 2880, 'T3': 2735},
            'Task denominators differ')
    require(all(r['eligible'] and r['correct'] == (r['answer'] == r['truth']) for r in records),
            'Accepted record score differs')
    cells = defaultdict(list)
    for row in records:
        row['cell'] = list(cell_key(row))
        cells[tuple(row['cell'])].append(row)
    require(Counter(k[0] for k in cells) == {'T1': 80, 'T2': 60, 'T3': 29}, 'Grid differs')
    a_dir = runs / 'current-nvfp4'
    a_path = a_dir / 'video-predictions.csv'
    a_summary = read(a_dir / 'summary.json')
    require(a_summary['status'] == 'COMPLETE' and file_sha256(a_path) == a_summary['video_csv_sha256'],
            'Historical A export changed')
    with a_path.open() as f:
        a_rows = [r for r in csv.DictReader(f) if r['model'] == 'qwen']
    amap = {r['sample_id']: r for r in a_rows}
    require(len(amap) == len(a_rows), 'Repeated historical A ID')
    paired = []
    pair_cells = defaultdict(list)
    for x in records:
        a = amap.get(x['sample_id'])
        if a is None:
            continue
        require(a['protocol'] == 'controlled-rgb16' and a['task'] == x['task']
                and a['truth'] == x['truth'] and a['source_video_sha256'] in
                [sha for kind, sha in x['source_media'] if kind == 'video'], 'Historical A identity differs')
        ac = a['answer'] == a['truth']
        require(ac == bool(int(a['correct'])), 'Historical A score differs')
        r = {'sample_id': x['sample_id'], 'task': x['task'], 'cell': x['cell'],
             'truth': x['truth'], 'A': a['answer'], 'X': x['answer'], 'A_correct': ac,
             'X_correct': x['correct'], 'output_tokens_changed':
             json.loads(a['output_token_ids']) != x['output_token_ids'],
             'A_raw_sha256': a['raw_sha256'], 'X_raw_digest': x['raw_digest']}
        paired.append(r)
        pair_cells[tuple(x['cell'])].append(r)
    peers = {}
    peer_bindings = {}
    for model, name in [('Cosmos', 'cosmos-x-full-20261028'), ('Nemo', 'nemo-x-full-20261029')]:
        folder = runs / name / 'analysis'
        summary, rows = bound_records(folder)
        peers[model] = {r['sample_id']: r for r in rows}
        require(summary['status'] == 'COMPLETE_PASS' and len(rows) == 9456
                and set(peers[model]) == set(lookup) | {MISSING}, 'Peer model scope differs')
        peer_bindings[model] = descriptor(folder / 'summary.json')
    between = defaultdict(list)
    for x in records:
        r = {'sample_id': x['sample_id'], 'task': x['task'], 'cell': x['cell'],
             'truth': x['truth'], 'Qwen': x['answer'], 'Qwen_correct': x['correct']}
        for model, rows in peers.items():
            peer = rows[x['sample_id']]
            identity(peer, x)
            r[model] = peer['answer']
            r[model + '_correct'] = peer['correct']
        between[tuple(x['cell'])].append(r)
    out = runs / NAME / 'classic-analysis'
    out.mkdir(exist_ok=True)
    atomic_json(out / 'canonical-records.json', records)
    write_csv(out / 'predictions.csv', records)
    cell_rows = [{'cell': list(k), 'expected_n': 96 if k[0] == 'T3' and k[1] != 90 else 48,
                  'truth_counts': dict(Counter(r['truth'] for r in rows)), **basic(rows)}
                 for k, rows in sorted(cells.items())]
    require(sum(r['expected_n'] - r['n'] for r in cell_rows) == 1, 'Missing cell budget differs')
    write_csv(out / 'cells.csv', cell_rows)
    write_csv(out / 'paired-A-X.csv', paired)
    write_csv(out / 'cells-A-X.csv', [{'cell': list(k), **pair_stats(rows)} for k, rows in sorted(pair_cells.items())])
    write_csv(out / 'shared-three-model-predictions.csv', [r for rows in between.values() for r in rows])
    write_csv(out / 'shared-three-model-cells.csv', [dict(cell=list(k), n=len(rows),
              **{m + '_correct': sum(r[m + '_correct'] for r in rows) for m in ['Cosmos', 'Nemo', 'Qwen']})
              for k, rows in sorted(between.items())])
    write_csv(out / 'timing.csv', [dict(worker=w, **t) for w in sorted({r['worker'] for r in records})
                                 for t in timing_tables([r for r in records if r['worker'] == w])])
    render(cells, pair_cells, between, out)
    summary = {'schema': 'qwen-classic-X-accepted-analysis-v1', 'status': 'PARTIAL_PASS_ONE_UNRESOLVED_CLAIM',
               'generated_HST': datetime.now(ZoneInfo('Pacific/Honolulu')).isoformat(),
               'unique_videos': 9455, 'expected_unique_videos': 9456, 'configs': 169,
               'missing_ids': [MISSING], 'missing_scored_as_wrong': False, 'new_model_calls': 0,
               'tasks': {t: basic(r for r in records if r['task'] == t) for t in ['T1', 'T2', 'T3']},
               'A_X_paired': [{'task': t, **pair_stats([r for r in paired if r['task'] == t])}
                              for t in ['T1', 'T2', 'T3']],
               'accepted_parent_archives': [descriptor(t13dir / 'summary.json'), descriptor(t2dir / 'summary.json')],
               'historical_A': descriptor(a_path), 'peer_X_archives': peer_bindings,
               'analysis_code': descriptor(__file__),
               'limitations': ['One unresolved reboot claim is excluded and listed, never scored as wrong.',
                               'A/X is a protocol-package comparison, not a single-factor load or bit-depth effect.',
                               'Cosmos/Nemo common-ID curves exclude the same one ID; their full original results remain unchanged.']}
    summary['outputs'] = {p.name: descriptor(p) for p in out.iterdir() if p.is_file() and p.name != 'summary.json'}
    atomic_json(out / 'summary.json', summary)
    return {k: summary[k] for k in ['status', 'unique_videos', 'configs', 'tasks', 'A_X_paired']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.root), ensure_ascii=False))
