"""Plot accepted Cosmos X cells and unchanged historical A proportions."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .official_x_results import require
from .storage import file_sha256


def render(root):
    out = Path(root) / 'runs/reproducibility/cosmos-x-full-20261028/analysis'
    summary = json.loads((out/'summary.json').read_text())
    require(summary['status']=='COMPLETE_PASS', 'Full audit required before plotting')
    data = {}
    for name in ('cells.csv','cells-A-X.csv'):
        require(file_sha256(out/name)==summary['outputs'][name]['sha256'], 'Accepted cell table changed')
        with (out/name).open() as f:
            data[name] = [dict(r, cell=json.loads(r['cell'])) for r in csv.DictReader(f)]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    def panel(ax, task, predicate, index, xlabel):
        for route, name, correct, color, marker in [('A','cells-A-X.csv','A_correct','#8b5a91','s'),
                                                   ('X','cells.csv','correct','#007e87','o')]:
            rows = sorted((r for r in data[name] if r['cell'][0]==task and predicate(r['cell'])),
                          key=lambda r:r['cell'][index])
            ax.plot([r['cell'][index] for r in rows],[100*int(r[correct])/int(r['n']) for r in rows],
                    color=color,marker=marker,ms=4,lw=1.5,label=route)
        ax.axhline(50,color='#9a9a9a',ls='--',lw=.8)
        ax.set(ylim=(-3,103),xlabel=xlabel,ylabel='Correct (%)')
        ax.grid(alpha=.15)
    fig, axs = plt.subplots(2,4,figsize=(13.2,7),layout='constrained')
    for sf,ax in zip([.5,1,2,4,8,16,32,64],axs.flat):
        panel(ax,'T1',lambda c:c[1]==sf,2,'Contrast m');ax.set_title(f'{sf:g} cycles / aperture')
    axs[0,0].legend();fig.suptitle('Cosmos · T1 motion direction · A vs X')
    figures=[('T1-A-X',fig)]
    fig,axs=plt.subplots(1,3,figsize=(12,3.9),layout='constrained')
    for bg,ax in zip([.05,.2,.4],axs):
        panel(ax,'T2',lambda c:c[1]==bg,3,'Luminance increment ΔY');ax.set_title(f'Background Y = {bg:g} · 500 ms')
    axs[0].legend();fig.suptitle('Cosmos · T2 brightness discrimination · A vs X')
    figures.append(('T2-A-X',fig))
    fig,axs=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    panel(axs[0],'T3',lambda c:c[1]<=2,1,'Angle (degrees)');axs[0].set_title('Fine-angle conditions')
    panel(axs[1],'T3',lambda c:True,1,'Angle (degrees)');axs[1].set_xscale('log');axs[1].set_title('Full primary range')
    axs[0].legend();fig.suptitle('Cosmos · T3 orientation discrimination · A vs X')
    figures.append(('T3-A-X',fig))
    target=out/'figures';target.mkdir(exist_ok=True)
    for name,fig in figures:
        for suffix in ('png','pdf'):fig.savefig(target/f'{name}.{suffix}',dpi=180)
        plt.close(fig)
    print(json.dumps({'figures':6,'directory':str(target),'new_model_calls':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    render(p.parse_args().root)
