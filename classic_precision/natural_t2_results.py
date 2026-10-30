"""Read-only source, runtime and final-ledger analysis for natural T2."""
from __future__ import annotations
import argparse
import csv
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json
from pathlib import Path
import socket
import time

from .natural_t2_stimuli import NAME
from .refinement_worker import read,require,descriptor
from .storage import digest,file_sha256,atomic_json
from .official_x_results import Archive,audit_worker,basic,write_csv,timing_tables
from . import cosmos_x_full_status as monitor

def source_audit(root,verify_media=False):
    root=Path(root);plan=root/'plans'/NAME;data=root/'datasets'/NAME
    allocation=read(plan/'allocation.json');criteria=read(plan/'allocation-criteria.json')
    quality=read(plan/'source-quality-policy.json')
    require(allocation['criteria_sha256']==file_sha256(plan/'allocation-criteria.json'),'Allocation recipe changed')
    require(allocation['asset_preparation_sha256']==file_sha256(plan/'asset-preparation.json'),'Prepared assets changed')
    expected={c['sample_id']:(c,b['card']) for b in allocation['blocks'] for c in b['cases']}
    require(len(expected)==3840 and len(allocation['blocks'])==480,'Scope differs')
    records={};proofs=[]
    for b in allocation['blocks']:
        path=data/'qa'/('block-%03d.json'%b['block']);q=read(path)
        require(q['status']=='FULL180_LOSSLESS_RGB_QA_PASS' and q['block']==b['block'],'Source block QA incomplete')
        require([x['sample_id'] for x in q['cases']]==[x['sample_id'] for x in b['cases']],'Source block IDs/order differ')
        for x in q['cases']:
            sid=x['sample_id'];target,card=expected[sid]
            require(sid not in records and all(x[k]==v for k,v in target.items()),'Duplicate/changed source condition')
            require(x['decoded_frames_exact']==180 and x['outside_target_identical'],'Frame/target check failed')
            require(x['render_implementation_sha256'] in quality['accepted_generator_sha256'],'Generator differs')
            require(abs(x['float64_delta_Y']-x['delta_Y'])<1e-10
                    and abs(x['decoded_delta_Y']-x['delta_Y'])<min(.002,.02*x['delta_Y']),'Absolute luminance increment differs')
            require(0<=x['white_blend_k']<=1 and x['offset_frame']-x['onset_frame']==30,'Invalid blend/event')
            for key in ('video_sha256','stream_sha256','baseline_rgb8_sha256','event_rgb8_sha256'):
                require(len(x[key])==64,'Missing identity '+key)
            if verify_media:require(file_sha256(data/'videos'/(sid+'.mp4'))==x['video_sha256'],'Encoded source changed')
            records[sid]={**x,'card':card}
        for group in {x['group'] for x in q['cases']}:
            pair=[x for x in q['cases'] if x['group']==group]
            require(len(pair)==2 and {x['truth'] for x in pair}=={'LEFT','RIGHT'}
                    and pair[0]['baseline_rgb8_sha256']==pair[1]['baseline_rgb8_sha256'],'LR pairing differs')
        proofs.append({'block':b['block'],'sha256':file_sha256(path)})
    require(set(records)==set(expected),'Missing/foreign source')
    cell=Counter((q['group'],q['delta_Y']) for q in records.values())
    require(len(cell)==80 and set(cell.values())=={48},'Unbalanced groups/deltas')
    require(len({q['background_id'] for q in records.values()})==200
            and len({q['object_id'] for q in records.values()})==100,'Library coverage changed')
    summary={'status':'COMPLETE_SOURCE_QA_PASS','videos':3840,'decoded_frames':3840*180,
             'source_block_receipts':proofs,'source_records_sha256':digest(records),
             'all_original_backgrounds':200,'object_sources':{'photo':50,'generated':50},
             'source_groups':4,'delta_cells':80,'videos_per_cell':48,
             'max_abs_decoded_delta_error':max(abs(x['decoded_delta_Y']-x['delta_Y']) for x in records.values()),
             'max_relative_decoded_delta_error':max(abs(x['decoded_delta_Y']/x['delta_Y']-1) for x in records.values()),
             'quality_policy':descriptor(plan/'source-quality-policy.json'),
             'baseline_Y_range':[min(x['baseline_object_Y'] for x in records.values()),max(x['baseline_object_Y'] for x in records.values())],
             'alpha_area_range':[min(x['alpha_area_px2'] for x in records.values()),max(x['alpha_area_px2'] for x in records.values())],
             'aggregate_direct_video_SHA_check':verify_media,'new_model_calls':0,
             'frame_comparison':'Every generated video decoded completely and RGB8 compared byte-for-byte to prescribed baseline/event frames.',
             'digital_units':'sRGB-decoded linear RGB Y; not calibrated display luminance.'}
    return summary,records

def status(root,name=NAME):
    root=Path(root);plans=root/'plans'/name;plan=read(plans/'plan.json');workers=[];active=False
    for card in plan['cards']:
        paths=sorted(plans.glob(card+'-segment*-profile.json'),key=lambda x:read(x)['control_generation'])
        paths=[x for x in paths if read(x)['hostname']==socket.gethostname()]
        if not paths:continue
        done=[];segments=[];issues=[]
        for path in paths:
            p=read(path);out=Path(p['output']);events,partial=monitor.journal(out/'events.jsonl')
            passed=[e['sample_id'] for e in events if e['event']=='TECHNICAL_PASS'];done+=passed
            wait=read(out.with_suffix('.wait.json')) if out.with_suffix('.wait.json').exists() else None
            failure=read(out/'failure.json') if (out/'failure.json').exists() else None
            if failure:issues.append('RETAINED_FAILURE_'+p['worker'])
            if wait and (wait['returncode']!=0 or not all(wait.get(k) for k in
                 ('real_wait_observed','owned_engines_exited','original_physical_lease_released'))):issues.append('UNCLOSED_ATTEMPT')
            if wait and wait.get('profile_sha256')!=digest(p):issues.append('WAIT_PROFILE_MISMATCH')
            segments.append({'worker':p['worker'],'profile':descriptor(path),'completed':len(passed),
                'wait':{k:wait.get(k) for k in ('returncode','terminal_status','real_wait_observed','owned_engines_exited','original_physical_lease_released')} if wait else None,
                'failure':failure,'partial_event_tail':partial})
        p=read(paths[-1]);out=Path(p['output']);allowed=set(plan['cards'][card]['assigned_ids'])
        claims={x.stem for x in Path(p['claim_root']).glob('*.json')}
        if len(done)!=len(set(done)) or not set(done)<=claims or not claims<=allowed:issues.append('CLAIM_ID_LEDGER_MISMATCH')
        if segments[-1]['wait'] and claims!=set(done):issues.append('TERMINAL_CLAIM_WITHOUT_PASS')
        if file_sha256(p['source_catalog']['path'])!=p['source_catalog']['sha256']:issues.append('SOURCE_CATALOG_CHANGED')
        active=active or not segments[-1]['wait'];window=monitor.timing_window(out/'request-timings.jsonl')
        workers.append({'card':card,'expected':len(allowed),'completed':len(done),'claimed':len(claims),
            'claimed_without_pass':sorted(claims-set(done)),'unattempted':len(allowed-claims),
            'recent_ordinary':window,'control':read(p['control_file']),'segments':segments,'issues':issues})
    return {'status':'LIVE_SNAPSHOT_NOT_FINAL_AUDIT','host':socket.gethostname(),
        'observed_UTC':datetime.now(timezone.utc).isoformat(),'observed_monotonic_seconds':time.monotonic(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'workers':workers,
        'resources':monitor.resources(root) if active else {'skipped':'all local workers closed'},'new_model_calls':0}

def full(root):
    from .natural_t2_qwen import science
    root=Path(root);plans=root/'plans'/NAME;out=root/'runs/reproducibility'/NAME;arc=Archive(root)
    plan=read(plans/'plan.json');qa,source=source_audit(root);records=[];proofs=[]
    for card in plan['cards']:
        paths=sorted(plans.glob(card+'-segment*-profile.json'),key=lambda x:read(x)['control_generation'])
        require(paths,'Missing card '+card);card_rows=[]
        for n,path in enumerate(paths):
            p=read(path);science(p,arc.ref);require(p['control_generation']==n,'Segment gap')
            spec={'profile':descriptor(path),'phase':'natural_T2','variant':'X_QSA_NO_FLASHINFER_AUTOTUNE'}
            terminal=read(arc.path(p['output'])/'result.json')
            if n<len(paths)-1:
                require(terminal['status']=='PAUSED','Nonfinal segment not preserved paused')
                spec['paused_prefix_successor']=descriptor(paths[n+1])
            else:require(terminal['status']=='COMPLETE','Final segment not complete')
            rows,evidence=audit_worker(arc,spec);card_rows+=rows;proofs.append(evidence)
        require([x['sample_id'] for x in card_rows]==plan['cards'][card]['assigned_ids'],'Card union/order differs')
        require({x.stem for x in arc.path(p['claim_root']).glob('*.json')}=={x['sample_id'] for x in card_rows},'Final claim union differs')
        records+=card_rows
    require(len(records)==len({x['sample_id'] for x in records})==3840 and {x['sample_id'] for x in records}==set(source),'Full union differs')
    groups=defaultdict(list)
    for r in records:
        q=source[r['sample_id']];r['natural_qa']=q
        groups[(q['group'],q['delta_Y'])].append(r)
    cells=[{'group':g,'delta_Y':d,**basic(rows),'left_correct':sum(r['correct'] for r in rows if r['truth']=='LEFT'),
            'right_correct':sum(r['correct'] for r in rows if r['truth']=='RIGHT')}
           for (g,d),rows in sorted(groups.items())]
    summary={'status':'COMPLETE_NATURAL_T2_X','model':'Qwen','new_scientific_calls':3840,'overall':basic(records),
             'source_QA':qa,'cells':cells,'worker_evidence':proofs,'plan':descriptor(plans/'plan.json'),
             'interpretation':'Paired appearance/background transfer; four source combinations, not a background-only effect.'}
    atomic_json(out/'canonical-records.json',records);write_csv(out/'cell-results.csv',cells)
    write_csv(out/'answers.csv',[{k:r[k] for k in ('sample_id','truth','answer','correct','worker')}|r['parameters'] for r in records])
    write_csv(out/'timing-summary.csv',timing_tables(records))
    references=reference_curves(root);write_csv(out/'classic-reference-cells.csv',references)
    summary['classic_reference_scope']='Historical A and existing qualified X are separately labeled; no new classic inference and no background-only causal claim.'
    atomic_json(out/'summary.json',summary);figures(out,cells,references)
    return {'status':summary['status'],'summary':descriptor(out/'summary.json'),'correct':summary['overall']['correct'],'n':3840}

def reference_curves(root,per_id=False):
    """Use existing controlled A, plus the one already qualified X T2 point."""
    root=Path(root);folder=root/'runs/reproducibility/current-nvfp4';path=folder/'video-predictions.csv'
    require(file_sha256(path)==read(folder/'summary.json')['video_csv_sha256'],'Historical A export changed')
    old={}
    with path.open() as f:
        for x in csv.DictReader(f):
            if x['model']!='qwen' or x['task']!='T2':continue
            cell=json.loads(x['cell_json'])
            if cell[3]!=30:continue
            require(x['protocol']=='controlled-rgb16','Not historical A')
            old[x['sample_id']]={'protocol':'A_historical','background_Y':cell[1],'delta_Y':cell[2],
                'answer':x['answer'],'truth':x['truth'],'correct':x['answer']==x['truth']}
    extra=root/'runs/reproducibility/reader-neighbors-extension-20261027/three-point-paired.csv'
    with extra.open() as f:
        for x in csv.DictReader(f):
            if x['model']!='qwen' or x['task']!='T2':continue
            row={'protocol':'A_historical','background_Y':.4,'delta_Y':float(x['point']),
                 'answer':x['A_historical'],'truth':x['truth'],'correct':x['A_historical']==x['truth']}
            if x['sample_id'] in old:require(row==old[x['sample_id']],'Historical A copies disagree')
            old[x['sample_id']]=row
    if per_id:return old
    qual=read(root/'runs/reproducibility/qwen-x-crosscard-t23-20261029/analysis/canonical-records.json')
    if isinstance(qual,dict):qual=qual['records']
    new=[x for x in qual if x['task']=='T2' and x['worker']=='fourier0-t23-cold1']
    require(len(new)==len({x['sample_id'] for x in new})==48,'Qualified X reference differs')
    rows=list(old.values())+[{'protocol':'X_qualified','background_Y':x['parameters']['background_Y'],
         'delta_Y':x['parameters']['delta_Y'],'correct':x['correct']} for x in new]
    groups=defaultdict(list)
    for x in rows:groups[(x['protocol'],x['background_Y'],x['delta_Y'])].append(x)
    return [{'protocol':p,'background_Y':b,'delta_Y':d,'n':len(v),'correct':sum(x['correct'] for x in v),
             'accuracy':sum(x['correct'] for x in v)/len(v)} for (p,b,d),v in sorted(groups.items())]

def figures(out,cells,references):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'photo_bg__photo_obj':'Photo background / Photo object','gpt_bg__gpt_obj':'Generated background / Generated object',
            'photo_bg__gpt_obj':'Photo background / Generated object','gpt_bg__photo_obj':'Generated background / Photo object'}
    colors=['#217a63','#9c4d2a','#396bb0','#9467ad']
    fig,ax=plt.subplots(figsize=(10,5.8))
    for (group,label),color in zip(labels.items(),colors):
        rows=sorted((x for x in cells if x['group']==group),key=lambda x:x['delta_Y'])
        ax.plot([x['delta_Y'] for x in rows],[x['correct']/x['n'] for x in rows],'-o',color=color,label=label,ms=4,lw=1.8)
    ax.axhline(.5,color='.5',ls=':',lw=1);ax.set(xlabel='Nominal increase in mean linear luminance',ylabel='Accuracy',ylim=(0,1.02),xlim=(.045,.405),title='Qwen X — natural T2 (48 videos per point and source group)')
    ax.legend(fontsize=9,frameon=False);ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.15)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('natural-T2-curves.'+ext),dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    for ax,((group,label),color) in zip(axes.flat,zip(labels.items(),colors)):
        rows=sorted((x for x in cells if x['group']==group),key=lambda x:x['delta_Y'])
        ax.plot([x['delta_Y'] for x in rows],[x['correct']/x['n'] for x in rows],'-o',color=color,label='Natural X',ms=4)
        for bg,style in zip(sorted({r['background_Y'] for r in references if r['protocol']=='A_historical'}),['--',':','-.']):
            rr=sorted((r for r in references if r['protocol']=='A_historical' and r['background_Y']==bg and .05<=r['delta_Y']<=.4),key=lambda r:r['delta_Y'])
            ax.plot([r['delta_Y'] for r in rr],[r['accuracy'] for r in rr],style,color='.5',lw=1.2,label=f'Classic A, background {bg:g}')
        rr=[r for r in references if r['protocol']=='X_qualified']
        ax.scatter([r['delta_Y'] for r in rr],[r['accuracy'] for r in rr],marker='D',color='black',s=35,label='Existing classic X point',zorder=4)
        ax.set_title(label,fontsize=11);ax.set(ylim=(0,1.02),xlim=(.045,.405));ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0,0].legend(fontsize=8,frameon=False);fig.supxlabel('Nominal increase in mean linear luminance');fig.supylabel('Accuracy')
    fig.suptitle('Natural X and separately labeled classic references');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('natural-vs-classic.'+ext),dpi=180)
    plt.close(fig)

def sensitivity(root):
    """Descriptive checks on fixed answers; never alter allocation or refit stimuli."""
    root=Path(root);out=root/'runs/reproducibility'/NAME
    summary=read(out/'summary.json');require(summary['status']=='COMPLETE_NATURAL_T2_X','Final audit required')
    records=read(out/'canonical-records.json');require(len(records)==3840,'Record count differs')
    axis_path=root/'datasets'/NAME/'review/axis-algorithm-review.json'
    axes={x['asset_id']:x for x in read(axis_path)['assets']}
    weights={src:sum(v['eigenvalue_ratio']>=4 for k,v in axes.items() if k.startswith(src+'_obj_'))/50
             for src in ('photo','gpt')}
    groups=defaultdict(list);cells=defaultdict(list);assets=defaultdict(list)
    for r in records:
        q=r['natural_qa'];groups[q['group']].append(r);cells[q['group'],q['delta_Y']].append(r)
        for kind in ('object','background'):assets[q['group'],kind,q[kind+'_id']].append(r)
    def acc(rows):return sum(x['correct'] for x in rows)/len(rows)
    def leave_one(rows,kind):
        base=acc(rows);values=[]
        for aid in {r['natural_qa'][kind+'_id'] for r in rows}:
            kept=[r for r in rows if r['natural_qa'][kind+'_id']!=aid]
            values.append({'removed':aid,'remaining_n':len(kept),'accuracy':acc(kept),'change_pp':100*(acc(kept)-base)})
        return max(values,key=lambda x:abs(x['change_pp']))
    cell_rows=[]
    for (group,delta),rows in sorted(cells.items()):
        strata={s:[r for r in rows if (axes[r['natural_qa']['object_id']]['eigenvalue_ratio']>=4)==s] for s in (False,True)}
        require(all(strata.values()),'Empty shape stratum; do not extrapolate')
        w=weights[rows[0]['natural_qa']['object_id'].split('_')[0]]
        adjusted=(1-w)*acc(strata[False])+w*acc(strata[True])
        cell_rows.append({'group':group,'delta_Y':delta,'n':len(rows),'accuracy':acc(rows),
            'elongated_n':len(strata[True]),'elongated_accuracy':acc(strata[True]),'other_accuracy':acc(strata[False]),
            'library_elongated_fraction':w,'shape_standardized_accuracy':adjusted,'change_pp':100*(adjusted-acc(rows)),
            'max_leave_one_object_change_pp':leave_one(rows,'object')['change_pp'],
            'max_leave_one_background_change_pp':leave_one(rows,'background')['change_pp']})
    group_rows=[{'group':g,**basic(rows),'leave_one_object':leave_one(rows,'object'),
                 'leave_one_background':leave_one(rows,'background')} for g,rows in sorted(groups.items())]
    asset_rows=[{'group':g,'asset_kind':k,'asset_id':a,**basic(rows)} for (g,k,a),rows in sorted(assets.items())]
    result={'status':'COMPLETE_DESCRIPTIVE_OUTCOME_SENSITIVITY','new_model_calls':0,'original_answers_and_allocation_unchanged':True,
        'bindings':{'summary':descriptor(out/'summary.json'),'records':descriptor(out/'canonical-records.json'),'axes':descriptor(axis_path)},
        'groups':group_rows,'shape_library_weights':weights,
        'max_cell_shape_standardization':max(cell_rows,key=lambda x:abs(x['change_pp'])),
        'attention_cells':[x for x in cell_rows if x['delta_Y'] in (.08,.18)],
        'method':'Use the prior allocation audit threshold eigenvalue ratio >=4. Reweight each observed cell to its source library proportion; retain the original raw curve. Leave one observed object or background out in turn.',
        'limits':'Descriptive sensitivity, not a causal adjustment or a test proving absence of confounding. Repeated assets and matched LR/source blocks are dependent; no independent-video significance claim.'}
    write_csv(out/'outcome-sensitivity-cells.csv',cell_rows);write_csv(out/'outcome-by-asset.csv',asset_rows)
    atomic_json(out/'outcome-sensitivity.json',result)
    return {'status':result['status'],'groups':group_rows,'max_cell_shape_standardization':result['max_cell_shape_standardization']}


def compare_classic_x(root):
    """Require both complete X experiments before drawing the requested comparison."""
    from .qwen_classic_t2 import NAME as CLASSIC
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=root/'runs/reproducibility'/NAME;cr=root/'runs/reproducibility'/CLASSIC
    natural=read(out/'summary.json');classic=read(cr/'summary.json')
    require(natural['status']=='COMPLETE_NATURAL_T2_X' and classic['status']=='COMPLETE_CLASSIC_QWEN_T2_X','Both full X audits required')
    rows=[{'condition':'classic_X','background_Y':r['background_Y'],'group':'classic','delta_Y':r['delta_Y'],'n':r['n'],'correct':r['correct'],'accuracy':r['accuracy']} for r in classic['cells']]
    rows += [{'condition':'natural_X','background_Y':None,'group':r['group'],'delta_Y':r['delta_Y'],'n':r['n'],'correct':r['correct'],'accuracy':r['accuracy']} for r in natural['cells']]
    write_csv(out/'classic-X-vs-natural-X.csv',rows)
    old=reference_curves(root,per_id=True);new=read(cr/'canonical-records.json');paired=[];groups=defaultdict(list)
    for r in new:
        sid=r['sample_id']
        if sid not in old:continue
        a=old[sid];q=r['parameters'];require(a['truth']==r['truth'] and a['background_Y']==q['background_Y'] and a['delta_Y']==q['delta_Y'],'A/X pairing differs')
        row={'sample_id':sid,'background_Y':q['background_Y'],'delta_Y':q['delta_Y'],'truth':r['truth'],'A':a['answer'],'X':r['answer'],
             'A_correct':a['correct'],'X_correct':r['correct'],'answer_changed':a['answer']!=r['answer']}
        paired.append(row);groups[q['background_Y'],q['delta_Y']].append(row)
    transitions=[]
    for (bg,delta),rr in sorted(groups.items()):
        transitions.append({'background_Y':bg,'delta_Y':delta,'paired_n':len(rr),'A_correct':sum(x['A_correct'] for x in rr),
            'X_correct':sum(x['X_correct'] for x in rr),'answer_changes':sum(x['answer_changed'] for x in rr),
            'correct_to_wrong':sum(x['A_correct'] and not x['X_correct'] for x in rr),
            'wrong_to_correct':sum(not x['A_correct'] and x['X_correct'] for x in rr)})
    write_csv(cr/'A-X-paired.csv',paired);write_csv(cr/'A-X-cells.csv',transitions)
    labels={'photo_bg__photo_obj':'Photo / photo','photo_bg__gpt_obj':'Photo / generated',
            'gpt_bg__photo_obj':'Generated / photo','gpt_bg__gpt_obj':'Generated / generated'}
    colors=['#217a63','#396bb0','#9467ad','#9c4d2a'];fig,axes=plt.subplots(1,3,figsize=(14,4.7),sharex=True,sharey=True)
    for ax,bg in zip(axes,(.05,.2,.4)):
        rr=sorted((r for r in classic['cells'] if r['background_Y']==bg),key=lambda r:r['delta_Y'])
        ax.plot([r['delta_Y'] for r in rr],[r['accuracy'] for r in rr],'-o',color='black',lw=2.4,ms=3,label='Classic X')
        for (group,label),color in zip(labels.items(),colors):
            rr=sorted((r for r in natural['cells'] if r['group']==group),key=lambda r:r['delta_Y'])
            ax.plot([r['delta_Y'] for r in rr],[r['accuracy'] for r in rr],'-o',color=color,ms=2.8,lw=1.3,label=label)
        ax.axhline(.5,color='.6',ls=':',lw=1);ax.set(title=f'Classic background {bg:g}',ylim=(0,1.03),xlabel='Absolute luminance increase')
        ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Accuracy');axes[0].legend(title='Background / object',frameon=False,fontsize=8)
    fig.suptitle('Qwen X — classic and natural T2 | 48 videos per point and condition');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('classic-X-vs-natural-X.'+ext),dpi=180)
    plt.close(fig)
    result={'status':'COMPLETE_SAME_X_CLASSIC_NATURAL_COMPARISON','new_model_calls':0,'classic_n':2880,'natural_n':3840,
        'paired_A_X_n':len(paired),'A_X_answer_changes':sum(x['answer_changed'] for x in paired),
        'bindings':{'classic':descriptor(cr/'summary.json'),'natural':descriptor(out/'summary.json')},
        'limits':'Same model/runtime and exposure; natural objects and backgrounds both differ. A only on common IDs; qualification excluded from formal X.'}
    atomic_json(out/'classic-X-comparison-summary.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--status',action='store_true');p.add_argument('--source-audit',action='store_true');p.add_argument('--verify-media',action='store_true');p.add_argument('--sensitivity',action='store_true');p.add_argument('--compare-classic-x',action='store_true')
    a=p.parse_args()
    if a.status:r=status(a.root)
    elif a.compare_classic_x:r=compare_classic_x(a.root)
    elif a.sensitivity:r=sensitivity(a.root)
    elif a.source_audit:
        r,_=source_audit(a.root,a.verify_media)
        atomic_json(Path(a.root)/'runs/reproducibility'/NAME/'source-audit.json',r)
        r={k:v for k,v in r.items() if k!='source_block_receipts'}
    else:r=full(a.root)
    print(json.dumps(r),flush=True)

if __name__=='__main__':main()
