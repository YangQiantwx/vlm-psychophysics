"""Repeatability figure helper for official_x_results.render_figures.

This module only plots accepted counts. It never imports a model or a runner.
The existing results renderer remains the single entry point for T4/T5 figures.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import tempfile

MODELS = ('qwen', 'cosmos', 'nemo', 'gemma')
NAMES = {'qwen': 'Qwen3.8 Flash Next', 'cosmos': 'Cosmos3 Super',
         'nemo': 'Nemotron 3.5 Super VL', 'gemma': 'Gemma 4 31B'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def repeat_cells(rows, *, allow_partial=False):
    """Keep each explicit model/configuration/task identity separate."""
    cells = {}
    partial = False
    expected = {'T2': 48, 'T3': 96}
    for row in rows:
        model, task = row['model'], row['task']
        version = row.get('variant', row.get('configuration'))
        require(isinstance(version, str) and version.strip(),
                'Repeat rows need explicit variant; default/qualified cannot be inferred')
        version = version.strip()
        require(model in MODELS and task in expected, 'Unsupported repeat model/task')
        n, changed = row['n'], row['answer_changes']
        require(type(n) is int and type(changed) is int and n > 0 and 0 <= changed <= n,
                'Invalid repeat count/paired denominator')
        key = (model, version, task)
        require(key not in cells, f'Duplicate repeat cell: {key}')
        cells[key] = row
        partial |= n != expected[task]
    require(cells, 'No repeatability data')
    partial |= {m for m, _, _ in cells} != set(MODELS)
    for model, version in {(m, v) for m, v, _ in cells}:
        partial |= any((model, version, task) not in cells for task in expected)
    require(allow_partial or not partial,
            'Incomplete four-model repeat data; allow_partial must be explicit')
    return cells, partial


def render_repeatability(summary, output, *, allow_partial=False, source_summary=None):
    """Export one PNG/PDF, with count/n labels and explicit configuration legend.

    Partial exports require opt-in. Synthetic checks are visibly labelled and
    confined to the OS temporary directory. No independent-trial CI is drawn.
    """
    if isinstance(summary, (str, Path)):
        path = Path(summary).resolve()
        payload = path.read_bytes()
        summary = json.loads(payload)
        source_summary = {'path': str(path), 'sha256': hashlib.sha256(payload).hexdigest()}
    elif source_summary is None:
        payload = json.dumps(summary, sort_keys=True, separators=(',', ':')).encode()
        source_summary = {'sha256': hashlib.sha256(payload).hexdigest()}
    require(summary.get('schema') == 'official-x-study-results-v1', 'Unsupported summary schema')
    output = Path(output).resolve()
    synthetic = summary.get('status') == 'SYNTHETIC_LAYOUT_CHECK_ONLY'
    require(summary.get('status') == 'COMPLETE_FINITE_STUDY' or
            (allow_partial and summary.get('status') in ('PARTIAL_FINITE_STUDY', 'PARTIAL_ACCEPTED_RESULTS')) or
            (synthetic and output.is_relative_to(Path(tempfile.gettempdir()).resolve())),
            'Repeat figures require accepted counts; synthetic checks must remain temporary')
    cells, partial = repeat_cells(summary['validation'], allow_partial=allow_partial)
    partial |= summary.get('status', '').startswith('PARTIAL_')
    models = [m for m in MODELS if any(k[0] == m for k in cells)]
    versions = sorted({k[1] for k in cells},
                      key=lambda v: ({'default': 0, 'qualified': 1}.get(v.lower(), 2), v))
    palette = ('#0072B2', '#009E73', '#D55E00', '#8856A7', '#555555')
    require(len(versions) <= len(palette), 'Too many repeat variants for a legible comparison')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    style = {'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.titlesize': 13,
             'axes.labelsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 10,
             'axes.spines.top': False, 'axes.spines.right': False,
             'axes.edgecolor': '#555555', 'axes.linewidth': .8,
             'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'}
    columns = min(2, len(models))
    rows = math.ceil(len(models) / columns)
    maximum = max(100 * r['answer_changes'] / r['n'] for r in cells.values())
    upper = min(108, max(10, math.ceil(maximum * 1.18 / 5) * 5))
    tick_step = 5 if upper <= 25 else (10 if upper <= 50 else 25)
    with plt.rc_context(style):
        fig, axes = plt.subplots(rows, columns, figsize=(6.2 * columns, 7.2 if rows > 1 else 4.6),
                                 squeeze=False, sharey=True)
        fig.subplots_adjust(left=.085, right=.985, bottom=.10, top=.80, hspace=.40, wspace=.20)
        prefix = 'SYNTHETIC LAYOUT CHECK | ' if synthetic else ('Partial | ' if partial else '')
        fig.suptitle(prefix + 'Repeated-input answer changes', x=.065, y=.985,
                     ha='left', fontsize=19, fontweight='bold')
        fig.text(.065, .985 - .43 / fig.get_figheight(),
                 'Cold run 1 vs 2 | changed / paired inputs | n/a: no paired data',
                 ha='left', va='top', fontsize=11, color='#444444')
        width = .72 / len(versions)
        for ax, model in zip(axes.flat, models):
            ax.set_title(NAMES[model], loc='left', fontweight='bold', pad=10)
            ax.set_ylim(0, upper)
            ax.set_yticks(range(0, int(min(upper, 100)) + 1, tick_step))
            ax.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#DDDDDD', linewidth=.7)
            ax.set_ylabel('Changed answers')
            for j, version in enumerate(versions):
                for x, task in enumerate(('T2', 'T3')):
                    row = cells.get((model, version, task))
                    xpos = x + (j - (len(versions) - 1) / 2) * width
                    if row is None:
                        ax.annotate('n/a', (xpos, 0), xytext=(0, 6), textcoords='offset points',
                                    ha='center', fontsize=10, color='#666666')
                        continue
                    value = 100 * row['answer_changes'] / row['n']
                    ax.bar(xpos, value, width=width * .88, color=palette[j])
                    ax.annotate(f"{row['answer_changes']}/{row['n']}", (xpos, value), xytext=(0, 6),
                                textcoords='offset points', ha='center', fontsize=11)
            ax.set_xticks((0, 1), ('T2', 'T3'))
            ax.set_xlim(-.55, 1.55)
        for ax in list(axes.flat)[len(models):]:
            ax.set_visible(False)
        labels = [v.capitalize() if v in ('default', 'qualified') else v for v in versions]
        handles = [Line2D([0], [0], color=palette[j], linewidth=7, label=label)
                   for j, label in enumerate(labels)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.55, .907),
                   ncol=min(3, len(handles)), frameon=False)
        denominators = [{'model': m, 'variant': v, 'task': t, 'n': r['n'],
                         'answer_changes': r['answer_changes']} for (m, v, t), r in cells.items()]
        provenance = {'source_summary': source_summary, 'partial': partial, 'synthetic': synthetic,
                      'denominators': denominators, 'uncertainty': 'Paired-input counts; no independent-trial CI.'}
        output.mkdir(parents=True, exist_ok=True)
        paths = {}
        for ext in ('png', 'pdf'):
            path = output / ('repeat-answer-changes.' + ext)
            metadata = ({'Description': json.dumps(provenance, sort_keys=True)} if ext == 'png' else
                        {'Subject': json.dumps(provenance, sort_keys=True), 'Creator': 'official_x_figures'})
            fig.savefig(path, dpi=240, bbox_inches='tight', pad_inches=.13, metadata=metadata)
            paths[ext] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        plt.close(fig)
    return {'name': 'repeat-answer-changes', **paths, **provenance,
            'role': 'TEMPORARY_SYNTHETIC_LAYOUT' if synthetic else 'OBSERVED_FINITE_STUDY'}
