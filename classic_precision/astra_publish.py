"""Export verified Astra primary scalar results for the website; never infer."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cells_from_records(records):
    groups = defaultdict(list)
    seen = set()
    for row in records:
        assert row['sample_id'] not in seen, 'Repeated primary sample'
        seen.add(row['sample_id'])
        assert row['strict_format'] and row['visual_only_trace_status'] == 'PASS_OBSERVED'
        assert row['model_effort_verified']
        assert row['answer'] in ('LEFT', 'RIGHT')
        assert row['correct'] == (row['answer'] == row['truth'])
        cell = tuple(row['cell'])
        assert cell[0] == row['task']
        assert not (row['task'] == 'T3' and cell[1] in (1.75, 2.25)), 'Auxiliary angle in primary results'
        assert 0 < row['returned_images'] <= 180
        assert (row['protocol_status'] == 'FULL180_TRACE_PASS') == (row['returned_images'] == 180)
        groups[cell].append(row)
    cells = []
    for cell, rows in sorted(groups.items()):
        assert len(rows) == 4, 'Every primary point must have four answers'
        cells.append({'cell': list(cell), 'correct': sum(r['correct'] for r in rows), 'n': 4,
                      'right': sum(r['answer'] == 'RIGHT' for r in rows),
                      'full180': sum(r['protocol_status'] == 'FULL180_TRACE_PASS' for r in rows),
                      'observed_frames': {'min': min(r['returned_images'] for r in rows),
                                          'max': max(r['returned_images'] for r in rows),
                                          'total': sum(r['returned_images'] for r in rows)}})
    return cells


def aggregate(cells):
    return {'correct': sum(c['correct'] for c in cells), 'n': sum(c['n'] for c in cells),
            'full180': sum(c['full180'] for c in cells), 'cells': len(cells)}


def export(root, output):
    run = root / 'runs/reproducibility/astra-path-only-20261027'
    final = run / 'repairs/capacity-retry1'
    files = {'canonical': final / 'canonical-primary-records.json', 'summary': final / 'summary.json',
             'answers': final / 'primary-answers.csv', 'cell_review': run / 'figures/cell-review.csv',
             'sample_review': run / 'figures/sample-review.json'}
    summary = json.loads(files['summary'].read_text())
    assert summary['status'] == 'COMPLETE_PRIMARY_VISUAL_ANSWERS' and summary['terminal_closure_verified']
    records = json.loads(files['canonical'].read_text())
    assert len(records) == summary['primary_valid'] == summary['primary_planned'] == 916
    assert summary['primary_missing'] == 0
    review = json.loads(files['sample_review'].read_text())
    assert review['source_canonical_sha256'] == sha(files['canonical'])
    assert review['source_summary_sha256'] == sha(files['summary'])
    answers = {r['sample_id']: r for r in csv.DictReader(files['answers'].open())}
    assert len(answers) == len(records)
    for row in records:
        answer = answers[row['sample_id']]
        assert json.loads(answer['cell']) == row['cell'] and answer['answer'] == row['answer']
        assert (answer['correct'] == 'True') == row['correct']
        assert int(answer['returned_images']) == row['returned_images']
        assert answer['protocol_status'] == row['protocol_status']
    cells = cells_from_records(records)
    assert len(cells) == 229
    checked = {tuple(json.loads(r['cell'])): r for r in csv.DictReader(files['cell_review'].open())}
    assert len(checked) == len(cells)
    for cell in cells:
        row = checked[tuple(cell['cell'])]
        assert (int(row['n']), int(row['correct']), int(row['full180_n'])) == (4, cell['correct'], cell['full180'])
    tasks = {}
    for tid in ('T1', 'T2', 'T3'):
        selected = [cell for cell in cells if cell['cell'][0] == tid]
        total = aggregate(selected)
        assert total['n'] == summary['by_task'][tid]['valid'] and total['correct'] == summary['by_task'][tid]['correct']
        tasks[tid] = {'cells': selected, 'all_primary': total}
        if tid == 'T2':
            tasks[tid]['by_exposure_frames'] = {str(frames): aggregate([c for c in selected if c['cell'][3] == frames]) for frames in (10, 30)}
    assert sum(t['all_primary']['full180'] for t in tasks.values()) == summary['full180'] == 215
    default_cells = [c for c in cells if c['cell'][0] != 'T2' or c['cell'][3] == 30]
    figure_assets = {}
    for stem in ('astra-T1', 'astra-T1-direction', 'astra-T2', 'astra-T3'):
        for suffix in ('.png', '.pdf'):
            path = run / 'figures' / (stem + suffix)
            figure_assets[path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    payload = {'schema': 'astra-observed-primary-site-v1', 'status': 'COMPLETE_PRIMARY_VISUAL_ANSWERS',
               'label': 'Astra', 'input_system': 'CLI · observed frames', 'reasoning_effort': 'medium',
               'generation': {'temperature': 'not specified', 'seed': 'not specified'},
               'all_primary': aggregate(cells), 'default_overview_500ms': aggregate(default_cells),
               'tasks': tasks, 'source_sha256': {name: sha(path) for name, path in files.items()},
               'scope': {'client_input': 'Original-size decoded PNGs viewed chronologically by CLI tools; not native video input.',
                         'observed': '215 answers viewed all180 frames;701 answered early. Counts in each UI view follow its filters.',
                         'sampling': 'Four answers per condition from two paired phase/onset placements; T3 side×sign strata can have n1.',
                         'inference': 'Exploratory observed points; no fitted threshold or population confidence interval.',
                         'excluded_primary_angles_deg': [1.75, 2.25],
                         'full180_subset': 'Behavior-selected; not shown as a primary curve or an unbiased comparison.',
                         'internal_processing': 'Client PNG identity does not establish RGB16 preservation, internal attention, or zero server preprocessing.'},
               'figures': {tid: {'png': 'astra/' + 'astra-' + tid + '.png', 'pdf': 'astra/' + 'astra-' + tid + '.pdf'} for tid in tasks},
               'figure_assets': figure_assets}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({'output': str(output), 'all_primary': payload['all_primary'], 'default_500ms': payload['default_overview_500ms'],
                      'by_task': {tid: item['all_primary'] for tid, item in tasks.items()}, 'source_sha256': payload['source_sha256']}))
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    export(args.root, args.output)


if __name__ == '__main__':
    main()
