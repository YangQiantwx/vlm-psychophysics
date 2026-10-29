"""Read-only Cosmos A/X point changes and monotonicity on matched grids."""
from __future__ import annotations
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.optimize import isotonic_regression
from .storage import atomic_json,file_sha256
from .official_x_results import read,write_csv,require


def analyze(root):
    r=Path(root)/'runs/reproducibility/cosmos-x-full-20261028/analysis';accepted=read(r/'summary.json')
    require(accepted['status']=='COMPLETE_PASS','Accepted full-X audit required')
    rows=list(csv.DictReader((r/'cells-A-X.csv').open()));full=list(csv.DictReader((r/'cells.csv').open()));pairs=list(csv.DictReader((r/'paired-A-X.csv').open()))
    require(len(pairs)==len({x['sample_id'] for x in pairs})==7296,'Pair identity scope differs')
    grouped=defaultdict(list)
    for p in pairs:grouped[p['cell']].append(p)
    for q in rows:
        ps=grouped[q['cell']];aa=[p['A_correct']=='True' for p in ps];xx=[p['X_correct']=='True' for p in ps]
        require(int(q['n'])==len(ps) and int(q['A_correct'])==sum(aa) and int(q['X_correct'])==sum(xx)
            and int(q['changed'])==sum(p['A']!=p['X'] for p in ps)
            and int(q['correct_to_wrong'])==sum(a and not x for a,x in zip(aa,xx))
            and int(q['wrong_to_correct'])==sum(not a and x for a,x in zip(aa,xx)),'Existing per-ID/cell tables differ')
    bycell={tuple(json.loads(q['cell'])):q for q in rows};points=[]
    for q in full:
        cell=tuple(json.loads(q['cell']));paired=bycell.get(cell);task=cell[0];n=int(q['n'])
        point={'task':task,'cell':list(cell),'spatial_frequency':cell[1] if task=='T1' else None,
            'background_Y':cell[1] if task=='T2' else None,'exposure_ms':500 if task=='T2' else None,
            'x':cell[-1],'x_unit':{'T1':'modulation_depth','T2':'delta_Y','T3':'degrees'}[task],
            'X_full_n':n,'X_full_correct':int(q['correct']),'X_full_pct':100*int(q['correct'])/n,
            'comparison_status':'PAIRED' if paired else 'A_UNAVAILABLE','paired_n':0 if not paired else int(paired['n'])}
        if paired:
            point.update({k:int(paired[k]) for k in ('A_correct','X_correct','changed','correct_to_wrong','wrong_to_correct')})
            point.update(A_pct=100*point['A_correct']/point['paired_n'],X_pct=100*point['X_correct']/point['paired_n'],delta_pp=100*(point['X_correct']-point['A_correct'])/point['paired_n'],changed_pct=100*point['changed']/point['paired_n'])
            require(n==point['paired_n'] and point['X_full_correct']==point['X_correct'],'Partial cell; do not silently compare different denominators')
        points.append(point)
    points.sort(key=lambda q:q['cell']);curves=defaultdict(list)
    for q in points:
        if q['comparison_status']=='PAIRED':curves[tuple(q['cell'][:-1])].append(q)
    metrics=[];tasks=[]
    for key,ps in sorted(curves.items()):
        ps.sort(key=lambda q:q['x']);weights=np.array([q['paired_n'] for q in ps],float)
        for arm in ('A','X'):
            y=np.array([q[arm+'_pct'] for q in ps])/100;d=np.diff(y);fit=isotonic_regression(y,weights=weights).x
            metrics.append({'curve':list(key),'task':key[0],'protocol':arm,'points':len(ps),'paired_n':int(weights.sum()),
                'adjacent_intervals':len(d),'downward_intervals':int((d<0).sum()),'total_downward_pp':float(100*(-d[d<0]).sum()),
                'max_downward_pp':float(100*max(0,-d.min(initial=0))),'weighted_monotone_squared_error':float(sum(weights*(y-fit)**2)),
                'isotonic_rmse_pp':float(100*np.sqrt(np.average((y-fit)**2,weights=weights))),
                'last_minus_first_pp':float(100*(y[-1]-y[0]))})
    for task in ('T1','T2','T3'):
        ps=[q for q in points if q['task']==task and q['comparison_status']=='PAIRED'];n=sum(q['paired_n'] for q in ps)
        row={'task':task,'paired_n':n,'paired_points':len(ps),'X_only_points':sum(q['task']==task and q['comparison_status']!='PAIRED' for q in points),
            **{k:sum(q[k] for q in ps) for k in ('A_correct','X_correct','changed','correct_to_wrong','wrong_to_correct')}}
        row.update(A_pct=100*row['A_correct']/n,X_pct=100*row['X_correct']/n,delta_pp=100*(row['X_correct']-row['A_correct'])/n,changed_pct=100*row['changed']/n)
        for arm in ('A','X'):
            ms=[q for q in metrics if q['task']==task and q['protocol']==arm]
            row[arm+'_downward_intervals']=sum(q['downward_intervals'] for q in ms);row[arm+'_intervals']=sum(q['adjacent_intervals'] for q in ms)
            row[arm+'_isotonic_rmse_pp']=100*np.sqrt(sum(q['weighted_monotone_squared_error'] for q in ms)/n)
        tasks.append(row)
    out=r/'curve-review';out.mkdir(exist_ok=True);write_csv(out/'point-changes.csv',points);write_csv(out/'task-changes.csv',tasks);write_csv(out/'curve-smoothness.csv',metrics)
    summary={'status':'COMPLETE_READ_ONLY','model':'Cosmos official mixed NVFP4','comparison':'Original A vs final X VIDEO; 180-image experiment is separate.',
        'model_calls':0,'full_X_points':169,'paired_points':136,'X_only_points':33,'paired_sources':7296,'X_sources_without_A':2160,
        'tasks':tasks,'largest_point_changes':sorted([q for q in points if q['comparison_status']=='PAIRED'],key=lambda q:abs(q['delta_pp']),reverse=True)[:12],
        'smoothness_methods':['Same matched x grid only; T1 split by SF, T2 by background with500ms fixed, T3 one angle curve.','Count decreases and sum their percentage-point magnitude. Weighted increasing-isotonic RMSE is a descriptive lack-of-monotonicity measure, not a fitted threshold or significance test.','A flat chance line can be smooth; monotonicity alone does not establish meaningful sensitivity. No outputs are filtered or smoothed for scoring.'],
        'judgment':'No general smoothness advantage for X. T3 both monotone on all17 matched points; T2 A has2 and X3 downward steps, both small. T1 has32/33 downward steps across8 curves; X has lower isotonic error but no clean increasing sensitivity curve.','causal_limit':'A-to-X includes decoder/input and runtime differences already documented; cannot attribute all changes to uint8 alone.',
        'inputs':{n:file_sha256(r/n) for n in ('summary.json','cells.csv','cells-A-X.csv','paired-A-X.csv')}}
    atomic_json(out/'summary.json',summary);render(points,out)
    return summary


def render(points,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    for task in ('T1','T2','T3'):
        keys=sorted({tuple(q['cell'][:-1]) for q in points if q['task']==task})
        shape=(2,4) if task=='T1' else ((1,3) if task=='T2' else (1,1));fig,axes=plt.subplots(*shape,figsize=(13,6.5) if task=='T1' else ((12,4) if task=='T2' else (7,4.5)),squeeze=False,sharey=True)
        for key,ax in zip(keys,axes.flat):
            rows=sorted([q for q in points if tuple(q['cell'][:-1])==key],key=lambda q:q['x']);ps=[q for q in rows if q['comparison_status']=='PAIRED']
            for arm,color in [('A','#345ca8'),('X','#d1772b')]:ax.plot([q['x'] for q in ps],[q[arm+'_pct'] for q in ps],'-o',label=arm+' · paired',color=color,markersize=4,lw=1.7)
            extra=[q for q in rows if q['comparison_status']!='PAIRED']
            if extra:ax.scatter([q['x'] for q in extra],[q['X_full_pct'] for q in extra],facecolors='none',edgecolors='#999',s=26,label='X only · no A pair')
            ax.set_title(f'SF {key[1]:g}' if task=='T1' else (f'Background {key[1]:g} · 500 ms' if task=='T2' else 'Angle from vertical'))
            ax.axhline(50,c='#b9b9b9',ls=':',lw=.8);ax.set_ylim(-3,104);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
            if task in ('T2','T3'):
                ax.set_xscale('log')
                ax.xaxis.set_major_locator(FixedLocator([.05,.1,.2,.4] if task=='T2' else [.5,.7,1,2,5,10,30,90]))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{v:g}'))
                ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xlabel({'T1':'Modulation depth','T2':'Luminance increment ΔY (log scale)','T3':'Angle (degrees; log scale)'}[task])
        for ax in axes[:,0]:ax.set_ylabel('Correct (%)')
        handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.97),ncol=3,frameon=False);fig.suptitle('Cosmos NVFP4 · '+task+' · A vs X video',y=1.02)
        fig.tight_layout(rect=(0,0,1,.9))
        for ext in ('png','pdf'):fig.savefig(out/(task+'-paired-curves.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();j=analyze(a.root);print(json.dumps({'status':j['status'],'paired_points':j['paired_points'],'X_only_points':j['X_only_points'],'tasks':j['tasks']},ensure_ascii=False))
