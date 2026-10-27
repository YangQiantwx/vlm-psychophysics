"""Read-only paired T2 duration analysis from completed source-bound journals.

Match the full scientific row except offset_frame. Preserve onset/side pairs and
report a same-host sensitivity analysis because duration arms were run in stages.
This command never loads a model or recomputes already accepted source videos.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .storage import digest, file_sha256, atomic_json
from .summary import model_snapshot, technical_passed


def require(value, message):
    if not value: raise ValueError(message)


def pair_key(row):
    params = {k:v for k,v in row['source_parameters'].items() if k != 'offset_frame'}
    return digest({'params':params,'task':row['task'],'truth':row['truth'],'prompt':row['prompt']})


def read_model(root, model, manifest):
    records, evidence = {}, []
    for run in ('t2-t3-rgb16','t2-duration-extension'):
        for mirror in ('remote-fourier','remote-radon'):
            folder = root/'runs'/run/mirror/'outputs'/model
            if not (folder/'identity.json').exists(): continue
            identity, raw, events, raw_tail, event_tail = model_snapshot(folder)
            require(not raw_tail and not event_tail, 'Incomplete finished journal tail: '+str(folder))
            accepted=technical_passed(raw,events)
            owners={e['attempt_id']:e.get('owner',{}) for e in events if e.get('event')=='INFLIGHT'}
            control={k:identity[k] for k in ('model','checkpoint_assets','runner_sha256','runtime_versions')}
            evidence.append({'path':str(folder.relative_to(root)),'identity_sha256':file_sha256(folder/'identity.json'),
                             'raw_sha256':file_sha256(folder/'outputs.jsonl'),'events_sha256':file_sha256(folder/'events.jsonl')})
            for r in raw:
                if r['task'] != 'T2': continue
                sid=r['sample_id']; require(sid in accepted,'Unresolved technical reference: '+sid)
                row=manifest[sid]
                require(r['source_parameters']==row['source_parameters'] and r['truth']==row['truth'],'Raw source row differs')
                require(sid not in records,'Duplicate reference ID; do not select one silently')
                require(r['finish_reason']=='stop' and r['raw_answer'] in ('LEFT','RIGHT'),'Non-normal result needs explicit analysis handling')
                owner=owners.get(r['attempt_id'],{});require(bool(owner.get('host')),'Missing actual attempt host')
                records[sid]={'row':row,'raw':r,'host':owner['host'],'controls':control,
                              'raw_record_sha256':digest(r),'source':evidence[-1]['path']}
    return records,evidence


def summary(rows):
    require(bool(rows),'Empty paired cohort')
    n=len(rows);short=sum(r['short_correct'] for r in rows);long=sum(r['long_correct'] for r in rows)
    groups=defaultdict(list)
    for r in rows: groups[r['onset_index']].append(r)
    clusters=list(groups.values());rng=np.random.default_rng(20261024)
    numerator=np.array([sum(int(r['long_correct'])-int(r['short_correct']) for r in g) for g in clusters])
    denominator=np.array([len(g) for g in clusters]);indices=rng.integers(0,len(clusters),(5000,len(clusters)))
    samples=numerator[indices].sum(axis=1)/denominator[indices].sum(axis=1)*100
    return {'pairs':n,'short_correct':short,'long_correct':long,'short_accuracy':short/n,'long_accuracy':long/n,
        'gain_wrong_to_correct':sum(not r['short_correct'] and r['long_correct'] for r in rows),
        'loss_correct_to_wrong':sum(r['short_correct'] and not r['long_correct'] for r in rows),
        'net_percentage_points':100*(long-short)/n,
        'short_RIGHT':sum(r['short_answer']=='RIGHT' for r in rows),'long_RIGHT':sum(r['long_answer']=='RIGHT' for r in rows),
        'onset_blocks':len(clusters),'onset_resampling_interval95_pp':np.quantile(samples,[.025,.975]).tolist(),
        'interval_scope':'Descriptive resampling of tested onset placements with paired sides and shared conditions; not independent scenes or human-population uncertainty.'}


def analyze(root, output):
    manifest={};source=[]
    for run in ('t2-t3-rgb16','t2-duration-extension'):
        path=root/'runs'/run/'manifest.json';value=json.loads(path.read_text());source.append({'path':str(path.relative_to(root)),'sha256':file_sha256(path)})
        for row in value['rows']:
            if row['task']=='T2':
                require(row['sample_id'] not in manifest,'Overlapping source ID')
                manifest[row['sample_id']]=row
    require(len(manifest)==3744,'Full 39-condition x48-placement x2-duration source inventory required')
    result={'schema':'T2_duration_paired/v1','comparison':'10frames(166.667ms) vs30frames(500ms), both180-frame clips',
            'analysis_source_sha256':file_sha256(__file__),'source_manifests':source,'new_model_calls':0,'models':{},
            'limits':['Time stages/hosts are recorded; same-host sensitivity cohort reported separately.',
                      'SDK/checkpoint/generation match does not prove every undocumented runtime state is identical.',
                      'This is input-to-answer behavior; attention, event detection, sensory integration and verbal readout are not individually identified.',
                      'Qwen corrected extension is still running; it is excluded until complete.']}
    pairs=[];cells=[];onsets=[]
    for model in ('nemo','cosmos'):
        records,evidence=read_model(root,model,manifest);require(set(records)==set(manifest),'Incomplete model T2 population')
        grouped=defaultdict(dict)
        for item in records.values():
            row=item['row'];p=row['source_parameters'];duration=p['offset_frame']-p['onset_frame']
            require(duration in (10,30),'Unexpected duration');key=pair_key(row)
            require(duration not in grouped[key],'Duplicate scientific duration cell')
            grouped[key][duration]=item
        require(len(grouped)==1872 and all(set(v)=={10,30} for v in grouped.values()),'Unpaired source conditions')
        model_pairs=[]
        for values in grouped.values():
            a,b=values[10],values[30];ra,rb=a['raw'],b['raw'];p=a['row']['source_parameters']
            require(a['controls']==b['controls'],'Documented checkpoint/SDK/runner differs in a duration pair')
            require(ra['effective_generation']==rb['effective_generation'],'Generation differs in a duration pair')
            require(ra['input_receipt']['chat_template_sha256']==rb['input_receipt']['chat_template_sha256'],'Actual chat template differs')
            require(ra['actual_input_tokens']==rb['actual_input_tokens'],'Context token length differs')
            row={'model':model,'background_Y':p['background_Y'],'delta_Y':p['delta_Y'],'onset_index':p['onset_index'],
                 'onset_frame':p['onset_frame'],'truth':a['row']['truth'],'short_id':ra['sample_id'],'long_id':rb['sample_id'],
                 'short_host':a['host'],'long_host':b['host'],'same_host':a['host']==b['host'],
                 'short_answer':ra['raw_answer'],'long_answer':rb['raw_answer'],
                 'short_correct':ra['raw_answer']==ra['truth'],'long_correct':rb['raw_answer']==rb['truth'],
                 'short_raw_sha256':a['raw_record_sha256'],'long_raw_sha256':b['raw_record_sha256']}
            pairs.append(row);model_pairs.append(row)
        matched=[r for r in model_pairs if r['same_host']]
        result['models'][model]={'all_documented_controls_matched':summary(model_pairs),'same_host':summary(matched),'source_evidence':evidence}
        by_cell=defaultdict(list);by_onset=defaultdict(list)
        for r in model_pairs:by_cell[(r['background_Y'],r['delta_Y'])].append(r);by_onset[r['onset_index']].append(r)
        require(len(by_cell)==39 and all(len(v)==48 for v in by_cell.values()),'Incomplete 39 full paired cells')
        for (bg,delta),rr in sorted(by_cell.items()):
            require({r['truth'] for r in rr}=={'LEFT','RIGHT'} and len({r['onset_index'] for r in rr})==24,'Unbalanced timing/side cell')
            for cohort,selected in [('all',rr),('same_host',[r for r in rr if r['same_host']])]:
                cells.append({'model':model,'cohort':cohort,'background_Y':bg,'delta_Y':delta,**summary(selected)})
        for onset,rr in sorted(by_onset.items()):onsets.append({'model':model,'onset_index':onset,**summary(rr)})
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('pairs.csv',pairs),('cells.csv',cells),('onsets.csv',onsets)]:
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result['outputs']={'pairs':'pairs.csv','cells':'cells.csv','onsets':'onsets.csv','figure':'duration_curves.png'}
    atomic_json(output/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True,sharey=True)
    for i,model in enumerate(('nemo','cosmos')):
        for j,bg in enumerate((.05,.2,.4)):
            ax=axes[i,j];cc=[c for c in cells if c['model']==model and c['cohort']=='all' and c['background_Y']==bg]
            cc.sort(key=lambda c:c['delta_Y'])
            for arm,label,color in [('short','167 ms','#3374b5'),('long','500 ms','#b86a31')]:
                ax.plot([c['delta_Y'] for c in cc],[100*c[arm+'_accuracy'] for c in cc],'-o',ms=3,label=label,color=color)
            ax.axhline(50,color='.7',ls=':',lw=1);ax.set_ylim(0,105);ax.set_title(f'{model.capitalize()} | background Y={bg}')
            ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
            if j==0:ax.set_ylabel('Accuracy (%)')
            if i==1:ax.set_xlabel('Linear luminance increment')
    axes[0,0].legend(frameon=False);fig.suptitle('Matched T2 intensity and duration: 48 paired side/onset answers per point',fontsize=12)
    fig.tight_layout();fig.savefig(output/'duration_curves.png',dpi=180);fig.savefig(output/'duration_curves.pdf');plt.close(fig)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=analyze(args.root.resolve(),args.output.resolve())
    print(json.dumps({m:{c:result['models'][m][c] for c in ('all_documented_controls_matched','same_host')} for m in result['models']}))


if __name__=='__main__':main()
