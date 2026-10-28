"""Read-only Astra primary figures and prospective sample-size review; no inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wilson(k, n):
    """95% independent-Bernoulli reference only; pairs are not independent trials."""
    z = 1.959963984540054
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n)) / d
    h = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / d
    return max(0., c-h), min(1., c+h)


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def priority(cell):
    if cell[0] == 'T1':
        return cell[1] in (32, 64) or (cell[1] in (.5, 8, 16) and cell[2] in (.05, .1, .2))
    if cell[0] == 'T2':
        return cell[2] in (.05, .06, .07)
    return cell[1] <= 1.


def analyze(root):
    root = Path(root)
    run = root/'runs/reproducibility/astra-path-only-20261027'
    canonical = run/'repairs/capacity-retry1/canonical-primary-records.json'
    summary_path = canonical.parent/'summary.json'
    sample_path = root/'plans/astra-path-only-20261027/samples.json'
    rows = json.loads(canonical.read_text())
    summary = json.loads(summary_path.read_text())
    source = {s['sample_id']: s for s in json.loads(sample_path.read_text())}
    assert summary['status'] == 'COMPLETE_PRIMARY_VISUAL_ANSWERS'
    assert len(rows) == len({r['sample_id'] for r in rows}) == 916
    assert all(r['visual_only_trace_status']=='PASS_OBSERVED' and r['strict_format'] for r in rows)
    grouped = defaultdict(list)
    for r in rows:
        assert r['cell'] == source[r['sample_id']]['cell']
        assert not (r['task']=='T3' and r['cell'][1] in (1.75, 2.25))
        grouped[tuple(r['cell'])].append(r)
    assert len(grouped) == 229 and all(len(v)==4 for v in grouped.values())
    cells, strata = [], []
    for cell, rr in sorted(grouped.items()):
        k = sum(r['correct'] for r in rr)
        lo, hi = wilson(k, len(rr))
        params = [source[r['sample_id']]['source_row']['nominal_parameters'] for r in rr]
        placement_key = 'bank_start_index' if cell[0]=='T1' else 'onset_index'
        placements = {p[placement_key] for p in params}
        assert len(placements)==2
        subgroup = defaultdict(list)
        for r, p in zip(rr, params):
            subgroup[('answer_side', r['truth'])].append(r)
            subgroup[('side_by_placement', f"{r['truth']}|{p[placement_key]}")].append(r)
            if cell[0]=='T3':
                subgroup[('side_by_sign', f"{r['truth']}|{p['tilt_sign']}")].append(r)
        for (kind, label), ss in sorted(subgroup.items()):
            strata.append(dict(task=cell[0], cell=json.dumps(cell), stratum=kind,
                               level=label, correct=sum(x['correct'] for x in ss), n=len(ss),
                               display=f"{sum(x['correct'] for x in ss)}/{len(ss)}",
                               singleton=len(ss)==1))
        full = [r for r in rr if r['protocol_status']=='FULL180_TRACE_PASS']
        cells.append(dict(task=cell[0], cell=json.dumps(cell),
                          sf=cell[1] if cell[0]=='T1' else '',
                          modulation=cell[2] if cell[0]=='T1' else '',
                          background=cell[1] if cell[0]=='T2' else '',
                          delta=cell[2] if cell[0]=='T2' else '',
                          exposure_frames=cell[3] if cell[0]=='T2' else '',
                          angle_deg=cell[1] if cell[0]=='T3' else '',
                          correct=k, n=4, accuracy=k/4,
                          wilson_reference_low=lo, wilson_reference_high=hi,
                          paired_placements=len(placements),
                          left_correct=sum(r['correct'] for r in rr if r['truth']=='LEFT'), left_n=2,
                          right_correct=sum(r['correct'] for r in rr if r['truth']=='RIGHT'), right_n=2,
                          full180_n=len(full), full180_correct=sum(r['correct'] for r in full),
                          full180_singleton=len(full)==1,
                          side_sign_min_n=min((len(v) for (kind,_),v in subgroup.items() if kind=='side_by_sign'),default=''),
                          priority='P1' if priority(cell) else 'P2', target_n=16,
                          proposed_additions=12, execution_status='PROPOSED_NOT_AUTHORIZED',
                          selected_placements=json.dumps(sorted(placements))))
    out = run/'figures'
    out.mkdir(exist_ok=True)
    save_csv(out/'cell-review.csv', cells)
    save_csv(out/'strata-review.csv', strata)
    task_review = {}
    for t in ('T1','T2','T3'):
        cc = [c for c in cells if c['task']==t]
        task_review[t] = dict(cells=len(cc), videos=sum(c['n'] for c in cc),
                             correct=sum(c['correct'] for c in cc),
                             perfect_cells=sum(c['correct']==4 for c in cc),
                             mixed_cells=sum(0<c['correct']<4 for c in cc),
                             full180_singleton_cells=sum(c['full180_singleton'] for c in cc),
                             priority_cells=sum(c['priority']=='P1' for c in cc),
                             priority_new_calls=sum(c['proposed_additions'] for c in cc if c['priority']=='P1'))
    review = dict(status='FIGURES_AND_SAMPLE_REVIEW_COMPLETE_NO_NEW_INFERENCE',
                  source_canonical_sha256=sha(canonical), source_summary_sha256=sha(summary_path),
                  source_samples_sha256=sha(sample_path), implementation_sha256=sha(__file__),
                  current_videos=916, current_cells=229, missing_answers=0,
                  task_review=task_review,
                  uncertainty={'method':'95% Wilson independent-Bernoulli reference',
                    'formula_source':'https://itl.nist.gov/div898/handbook/prc/section2/prc241.htm',
                    'limitation':'Only two paired phase/onset placements per cell. Reference bars are not calibrated cluster/population confidence intervals.',
                    'four_of_four':wilson(4,4),'one_of_one':wilson(1,1)},
                  prospective={'status':'PROPOSED_NOT_AUTHORIZED', 'target_videos_per_cell':16,
                    'focused_cells':sum(c['priority']=='P1' for c in cells),
                    'focused_new_calls':sum(c['proposed_additions'] for c in cells if c['priority']=='P1'),
                    'all_cells_new_calls':229*12,
                    'T1':'8 distinct phases x LEFT/RIGHT; preserve existing two phases, add six; spread new phases over the 24-position bank before any new answers.',
                    'T2':'8 distinct onsets x LEFT/RIGHT; preserve existing two onsets, add six across the onset bank.',
                    'T3':'For non-90deg: 4 shared onsets x LEFT/RIGHT x sign, 16 videos; include both existing onsets and complete both signs at each. At 90deg: canonical sign, 8 onsets x LEFT/RIGHT.',
                    'priority_T1':'SF32/64 all ten modulation values; SF0.5/8/16 at m=.05/.1/.2 (29 cells).',
                    'priority_T2':'Delta=.05/.06/.07 x all three backgrounds x both exposure durations (18 cells).',
                    'priority_T3':'Angles .5/.6/.7/.8/.9/1.0 deg (6 cells).',
                    'selection_warning':'Priorities use the completed pilot. Label the new wave separately; do not pool unequal sample sizes into an unweighted overall score or repeatedly sample until significance.',
                    'no_duplicate_calls':'Use distinct existing source IDs/placements. Repeating a decoded video is not a new stimulus sample.',
                    'not_precision_guarantee':'16 is a balanced follow-up allocation, not a power analysis or proof of perfect performance.'},
                  singleton_side_sign_groups=sum(s['stratum']=='side_by_sign' and s['singleton'] for s in strata),
                  full180_subset_warning='Behavior-selected subset, not randomized. Do not plot singleton percentages as reliable performance or force new full180 trials into this old protocol.',
                  scientific_calls_added=0)
    (out/'sample-review.json').write_text(json.dumps(review,indent=2)+'\n')
    return out, cells, review


BLUE = '#205b91'
ORANGE = '#c16635'
TEAL = '#238477'


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['bottom','left']].set_color('#b5bfca')
    ax.set_ylim(-.05,1.19)
    ax.set_yticks([0,.25,.5,.75,1], ['0','25','50','75','100'])
    ax.grid(axis='y',color='#e9edf1',linewidth=.7)
    ax.axhline(.5,color='#a8b0bb',ls='--',lw=.8,zorder=0)
    ax.tick_params(length=3,color='#b5bfca')


def curve(ax, cc, xkey, side=False):
    cc = sorted(cc,key=lambda c:c[xkey]);x=np.array([c[xkey] for c in cc],dtype=float)
    y=np.array([c['accuracy'] for c in cc]);low=np.array([c['wilson_reference_low'] for c in cc]);high=np.array([c['wilson_reference_high'] for c in cc])
    ax.errorbar(x,y,yerr=[np.maximum(0,y-low),np.maximum(0,high-y)],color=BLUE,
                ecolor='#aac1d8',capsize=3,lw=1.6,elinewidth=1,marker='o',ms=5,zorder=3)
    for i,(a,b,c) in enumerate(zip(x,y,cc)):
        dy = 24 if i and a-x[i-1]<.07 and b==y[i-1] else 9
        ax.annotate(f"{c['correct']}/4",(a,b),xytext=(0,dy),textcoords='offset points',ha='center',fontsize=9,color=BLUE)
    if side:
        ax.plot(x,[c['left_correct']/2 for c in cc],color=TEAL,ls=':',lw=1,marker='^',ms=3,alpha=.8)
        ax.plot(x,[c['right_correct']/2 for c in cc],color=ORANGE,ls=':',lw=1,marker='v',ms=3,alpha=.8)
    style(ax)


def finish(fig, out, name, subtitle):
    fig.text(.055,.90 if fig.get_figheight()<6 else .934,subtitle,fontsize=11,color='#435367')
    fig.text(.055,.055,'Points show k/4 videos from two paired placements. Lines connect observations; dashed line = 50% chance. No fitted threshold.',fontsize=9,color='#435367')
    fig.text(.055,.025,'Bars: 95% Wilson binomial reference assuming independent videos; pairing is not modeled. Early answers included; 1.75°/2.25° excluded.',fontsize=9,color='#435367')
    fig.savefig(out/(name+'.png'),dpi=180,facecolor='white')
    fig.savefig(out/(name+'.pdf'),facecolor='white')
    plt.close(fig)


def plot(out, cells):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.labelcolor':'#24364b','text.color':'#24364b','pdf.fonttype':42})
    t1=[c for c in cells if c['task']=='T1']
    fig,axs=plt.subplots(2,4,figsize=(15,8.4),sharex=True,sharey=True)
    fig.subplots_adjust(top=.855,bottom=.13,left=.055,right=.985,wspace=.15,hspace=.3)
    fig.suptitle('Astra · T1 stripe-motion judgments',x=.055,y=.982,ha='left',fontsize=20,fontweight='bold')
    for ax,sf in zip(axs.flat,sorted({c['sf'] for c in t1})):
        cc=[c for c in t1 if c['sf']==sf];curve(ax,cc,'modulation',True)
        ax.set_title(f'SF {sf:g} cycles/aperture',fontsize=12,pad=7)
        ax.set_xticks([.05,.2,.4,.6,.8,1],['.05','.2','.4','.6','.8','1'])
    for ax in axs[-1]:ax.set_xlabel('Modulation depth m')
    for ax in axs[:,0]:ax.set_ylabel('Correct (%)')
    fig.text(.055,.895,'Blue: all 4 videos     Green ▲: LEFT (n=2)     Orange ▼: RIGHT (n=2)',fontsize=10,color='#435367')
    finish(fig,out,'astra-T1','320 videos · 80 conditions · 278/320 correct · SF64: LEFT 19/20, RIGHT 2/20')
    t2=[c for c in cells if c['task']=='T2']
    fig,axs=plt.subplots(2,3,figsize=(15,7.7),sharex=True,sharey=True)
    fig.subplots_adjust(top=.86,bottom=.14,left=.055,right=.985,wspace=.12,hspace=.28)
    fig.suptitle('Astra · T2 brightness-event judgments',x=.055,y=.982,ha='left',fontsize=20,fontweight='bold')
    for i,d in enumerate((10,30)):
        for j,bg in enumerate((.05,.2,.4)):
            ax=axs[i,j];curve(ax,[c for c in t2 if c['background']==bg and c['exposure_frames']==d],'delta')
            ax.set_title(f'Background Y={bg:g}  |  {d} frames',fontsize=12,pad=7)
            ax.set_xticks([.05,.1,.15,.2,.25,.3,.35,.4],['.05','.10','.15','.20','.25','.30','.35','.40'])
    for ax in axs[-1]:ax.set_xlabel('Luminance increment ΔY')
    for ax in axs[:,0]:ax.set_ylabel('Correct (%)')
    # Twenty near-ceiling labels overlap in the dense low-delta section: one explicit shared denominator suffices.
    for ax in axs.flat:
        for artist in list(ax.texts):artist.remove()
        ax.text(.98,.91,'Every point: 4/4',transform=ax.transAxes,ha='right',fontsize=10,color=BLUE)
    finish(fig,out,'astra-T2','480 videos · 120 conditions · 480/480 correct · Ceiling observed; no transition or threshold established')
    t3=[c for c in cells if c['task']=='T3']
    fig,axs=plt.subplots(1,2,figsize=(15,5.7),gridspec_kw={'width_ratios':[1.35,1]})
    fig.subplots_adjust(top=.79,bottom=.21,left=.055,right=.985,wspace=.15)
    fig.suptitle('Astra · T3 vertical-axis judgments',x=.055,y=.982,ha='left',fontsize=20,fontweight='bold')
    near=[c for c in t3 if c['angle_deg']<=2.5];wide=[c for c in t3 if c['angle_deg']>=3]
    for ax,cc in zip(axs,(near,wide)):curve(ax,cc,'angle_deg');ax.set_xlabel('Tilt angle (degrees)')
    axs[0].set_xticks([.5,.7,.9,1.1,1.3,1.5,1.7,1.9,2.1,2.3,2.5]);axs[0].set_title('Fine-angle region (21 conditions)')
    axs[1].set_xscale('log');axs[1].set_xticks([3,4,8,16,32,45,90],['3','4','8','16','32','45','90']);axs[1].set_title('Larger angles (8 conditions; log scale)')
    axs[0].set_ylabel('Correct (%)')
    for ax in axs:
        for artist in list(ax.texts):artist.remove()
        ax.text(.97,.89,'Every point: 4/4',transform=ax.transAxes,ha='right',fontsize=10,color=BLUE)
    fig.text(.055,.115,'At 28 non-90° angles: LEFT/RIGHT × tilt sign has only 1 video per subgroup. At 90°: one canonical sign, 2 per side.',fontsize=10,color='#8a4b27')
    finish(fig,out,'astra-T3','116 videos · 29 conditions · 116/116 correct · 0.5° is the smallest tested angle, not an estimated discrimination threshold')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args()
    out,cells,review=analyze(args.root);plot(out,cells)
    print(json.dumps({'output':str(out),'task_review':review['task_review'],'prospective':review['prospective'],'scientific_calls_added':0},ensure_ascii=False))


if __name__=='__main__':main()
