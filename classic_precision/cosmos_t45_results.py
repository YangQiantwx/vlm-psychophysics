"""Read-only current T4/T5 Cosmos X status, exact closure and paired analysis."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import socket
import time

from . import official_x_results as audit
from .cosmos_x_full_status import inspect_profile, resources
from .storage import atomic_json, digest, file_sha256

NAME = 'cosmos-x-t45-20261030'
FIELDS = ('model_class','method_kernel_counts','actual_backends','kernel_config','mm_device_do_normalize')


def status(root):
    root = Path(root);host='radon' if str(root).startswith('/mnt/sda/') else 'fourier'
    current=root/'plans'/NAME/(host+'-reconciliation-execution-plan.json')
    if not current.exists():current=root/'plans'/NAME/(host+'-revision2-execution-plan.json')
    plan = audit.read(current if current.exists() else root/'plans'/NAME/'execution-plan.json')
    cards = ('radon0',) if str(root).startswith('/mnt/sda/') else ('fourier0','fourier1')
    values = []
    for name in plan['shards']:
        if plan['shards'][name]['card'] not in cards: continue
        if not plan['shards'][name].get('run_new_calls',True):
            values.append({'shard':name,'status':('CLOSED_PRECLAIM_REFUSAL_VALID_PREFIX' if plan['shards'][name].get('preclaim_refusal_admission') else 'REUSED_VALID_CLOSED_PREFIX'),
                'current_valid_count':len(plan['shards'][name]['assigned_ids']),
                'preserved_invalid_old_count':len(plan['shards'][name]['excluded_original_ids'])})
            continue
        files = sorted((root/'plans'/NAME).glob(name+'-segment*-profile.json'))
        if not files:
            values.append({'shard':name,'status':'WAITING_FOR_VALIDATED_SOURCE','planned':len(plan['shards'][name]['assigned_ids'])})
        else:
            for p in files:
                row={'shard':name, **inspect_profile(p)}
                qpath=root/'plans'/NAME/(plan['shards'][name]['card']+'-reconciled-handoff.json')
                if not qpath.exists():qpath=root/'plans'/NAME/(plan['shards'][name]['card']+'-current-handoff.json')
                if plan['shards'][name]['domain']=='natural' and qpath.exists():
                    queue=audit.read(qpath);state_path=Path(queue['state_path'])
                    if state_path.exists():
                        state=audit.read(state_path);row['finite_queue_status']=state['status']
                        if state['status']=='WAITING_PREDECESSOR' and row['claimed']==0:
                            row['status']='WAITING_EXACT_PREDECESSOR'
                values.append(row)
    return {'status':'BOUNDED_READ_ONLY_SNAPSHOT','hostname':socket.gethostname(),
            'observed_UTC':datetime.now(timezone.utc).isoformat(),'monotonic_seconds':time.monotonic(),
            'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'workers':values,'resources':resources(root),'model_calls_added':0}


def counts(rows):
    eligible = [r for r in rows if r['eligible']]
    return {'n':len(rows),'scored_n':len(eligible),'correct':sum(r['correct'] for r in eligible),
            'accuracy':sum(r['correct'] for r in eligible)/len(eligible) if eligible else None,
            'LEFT':sum(r['answer']=='LEFT' for r in rows),'RIGHT':sum(r['answer']=='RIGHT' for r in rows),
            'YES':sum(r['answer']=='YES' for r in rows),'NO':sum(r['answer']=='NO' for r in rows)}


def source_cue_diagnostic(root):
    """Evaluate a fixed geometric cue, independently of every model answer."""
    root=Path(root)
    manifest=root/'plans/t45-final-20261030/current1200-manifest.json'
    sources=audit.read(manifest)['items']
    rows=[]
    for r in sources:
        if r['domain']!='classic' or r['task']!='T4' or r['coherence']==0:continue
        positions,velocities=r['positions_xy_px'],r['velocity_xy_px_s']
        audit.require(len(positions)==len(velocities)==r['source_parameters']['N']==16,
                      'Unexpected current T4 geometry')
        first=math.fsum(p[0] for p in positions)/16
        last=math.fsum(p[0]+v[0]*(179/60) for p,v in zip(positions,velocities))/16
        initial_answer='RIGHT' if first>384 else 'LEFT'
        final_answer='RIGHT' if last>384 else 'LEFT'
        rows.append({'sample_id':r['sample_id'],'coherence':r['coherence'],
            'initial_centroid_x':first,'final_centroid_x':last,'truth':r['direction'],
            'initial_rule_correct':initial_answer==r['direction'],
            'final_rule_answer':final_answer,'final_rule_correct':final_answer==r['direction']})
    audit.require(len(rows)==384,'Expected all 384 nonzero-coherence classic inputs')
    cells=[]
    for coherence in sorted({r['coherence'] for r in rows}):
        group=[r for r in rows if r['coherence']==coherence]
        audit.require(len(group)==48,'Incomplete source-cue condition')
        cells.append({'coherence':coherence,'n':len(group),
            'initial_rule_correct':sum(r['initial_rule_correct'] for r in group),
            'final_rule_correct':sum(r['final_rule_correct'] for r in group)})
    result={'status':'COMPLETE_FIXED_GEOMETRIC_CUE_DIAGNOSTIC','source_manifest':audit.descriptor(manifest),
        'n':len(rows),'correct':sum(r['final_rule_correct'] for r in rows),
        'initial_rule_correct':sum(r['initial_rule_correct'] for r in rows),'model_calls':0,
        'rule':'RIGHT iff the mean of the 16 final planned x centers is >384; LEFT otherwise.',
        'fixed_protocol':{'width':768,'height':512,'fps':60,'last_frame_index':179},
        'scope':'Uses source geometry, not decoded-image measurements or a VLM. No fitting, training, result selection, or source changes. Natural pairs share these trajectories.',
        'interpretation':'Final spatial position carries answer information. T4 measures direction discrimination under these stimuli; accuracy alone does not establish temporal motion integration.',
        'cells':cells,'records':rows}
    out=root/'runs/reproducibility'/NAME/'source-cue-diagnostic.json'
    atomic_json(out,result)
    return {k:result[k] for k in ('status','n','correct','initial_rule_correct','model_calls','cells')}


def audited_revision_prefix(archive, spec, profile, wait, terminal, wait_path):
    """Audit the entire preserved prefix before selecting source-valid records."""
    plan=archive.ref(spec['retained_source_revision'])
    group=plan['shards'][spec['revision_shard']]
    audit.require(group['run_new_calls'] is False and group['previous_classic_profile']==spec['profile'],
                  'Not a bound source-revision prefix')
    ids=terminal['completed_ids']
    audit.require(wait['returncode']==0 and wait['terminal_status']==terminal['status']=='PAUSED'
                  and not terminal['UNCERTAIN_ids'] and ids==profile['assigned_ids'][:len(ids)]
                  and terminal['unattempted_ids']==profile['assigned_ids'][len(ids):], 'Unclean preserved prefix')
    claims={s for s in profile['assigned_ids'] if (archive.path(profile['claim_root'])/(s+'.json')).exists()}
    audit.require(claims==set(ids),'Unresolved original claim')
    revisions=archive.ref(plan['classic_replacement_map'])['items']
    changed={r['parent_v1_source_id'] for r in revisions if r['changed']}
    audit.require(group['assigned_ids']==[s for s in ids if s not in changed]
                  and group['excluded_original_ids']==[s for s in ids if s in changed],
                  'Retained answers were not selected only by independent source identity')
    return ids


def analyze(root):
    root=Path(root); arc=audit.Archive(root); plans=root/'plans'/NAME
    selected=audit.read(plans/'analysis-selection.json') if (plans/'analysis-selection.json').exists() else None
    audit.require(selected is not None,'Explicit current analysis-selection.json is required; historical plans are not a fallback')
    plan=arc.ref(selected['execution_plan']) if selected else audit.read(plans/'execution-plan.json')
    current_manifest=arc.ref(plan['source_id_plan'])
    current_sources=current_manifest['items'] if isinstance(current_manifest,dict) else current_manifest
    source_by_id={r['sample_id']:r for r in current_sources}
    audit.require(len(current_sources)==len(source_by_id)==1200,'Current source manifest is incomplete or duplicated')
    cue_path=root/'runs/reproducibility'/NAME/'source-cue-diagnostic.json'
    if not cue_path.exists():source_cue_diagnostic(root)
    cue=audit.read(cue_path)
    audit.require(cue['source_manifest']['sha256']==plan['source_id_plan']['sha256']
                  and cue['status']=='COMPLETE_FIXED_GEOMETRIC_CUE_DIAGNOSTIC'
                  and cue['model_calls']==0,'Source cue diagnostic is not bound to this manifest')
    if selected:
        source_audit=arc.ref(selected['final_source_audit'])
        audit.require(source_audit['status']=='PASS_FINAL1200_SOURCE_AND_INDEPENDENT_PIXELS'
                      and source_audit['sources']==1200 and source_audit['frames']==216000,
                      'Current source validation incomplete')
    records=[];retired=[];workers=[];all_items={};reader_bindings=[]
    for shard, spec in plan['shards'].items():
        ppath=arc.path(selected['profiles'][shard]['path']) if selected else plans/(shard+'-segment0-profile.json')
        if selected:audit.require(audit.descriptor(ppath)['sha256']==selected['profiles'][shard]['sha256'],'Accepted profile changed')
        # An explicit successor manifest is required if a human paused a shard.
        # A latest-file heuristic must never silently select a resume version.
        p=audit.read(ppath)
        retained=not spec.get('run_new_calls',True)
        if not retained:audit.require(p['assigned_ids']==spec['assigned_ids'],'Initial complete shard allocation differs')
        worker_spec={'profile':selected['profiles'][shard] if selected else audit.descriptor(ppath),
                     'phase':spec['domain'],'variant':'X_OFFICIAL_NATIVE_T45'}
        preclaim=spec.get('preclaim_refusal_admission')
        if preclaim:
            worker_spec['preclaim_refusal_admission']=preclaim
        elif retained:
            worker_spec.update(paused_prefix_successor=selected['execution_plan'],
                               retained_source_revision=selected['execution_plan'],revision_shard=shard)
        original=audit.closed_prefix_selection
        try:
            if retained and not preclaim:audit.closed_prefix_selection=audited_revision_prefix
            rows,evidence=audit.audit_worker(arc,worker_spec)
        finally:audit.closed_prefix_selection=original
        expected=arc.ref(p['qualified_runtime'])
        audit.require([{k:x[k] for k in FIELDS} for x in evidence['runtime']]
                      ==[{k:x[k] for k in FIELDS} for x in expected], 'Runtime changed')
        acceptance=arc.ref(p['source_acceptance']);catalog=arc.ref(p['source_catalog'])
        parent=arc.ref(p['qualified_profile'])
        audit.require(p.get('request_mm_processor_kwargs_by_task',{})==parent.get('request_mm_processor_kwargs_by_task',{}),
                      'Unqualified task-specific processor setting')
        audit.require(acceptance['assigned_ids']==p['assigned_ids']
                      and acceptance['catalog_sha256']==p['source_catalog']['sha256'], 'CPU source receipt differs')
        if selected or not retained:
            pixels=arc.ref(selected['independent_pixel_audits'][shard] if retained else acceptance['independent_pixel_audit'])
            audit.require(pixels['status']=='PASS_INDEPENDENT_PIXELS_COMPLETE_CARD','Independent source pixels not accepted')
            measured={r['sample_id']:r for r in pixels['measurements']}
            for sid in spec['assigned_ids']:
                audit.require(measured[sid]['status']=='PASS_INDEPENDENT_PIXEL_AUDIT'
                              and measured[sid]['video_sha256']==catalog['items'][sid]['ready']['video_sha256']
                              and measured[sid]['stream_sha256']==catalog['items'][sid]['ready']['stream_sha256'],
                              'Independent source pixels not bound to actual catalog')
        for raw in audit.journal(arc.path(p['output'])/'raw.jsonl'):
            item=catalog['items'][raw['sample_id']]
            actual=p.get('request_mm_processor_kwargs_by_task',{}).get(raw['task'],p['request_mm_processor_kwargs'])
            audit.require(raw['request_mm_processor_kwargs']==actual,'Effective processor settings changed')
            if raw['deep']:
                qa=audit.read(arc.path(item['source_qa_path']))
                for reader in raw['native_reader']:
                    audit.require(reader['decoded_sha256']==qa['stream_sha256']==item['ready']['stream_sha256']
                                  and reader['per_frame_sha256']==qa['decoded_RGB8_frame_hashes'],
                                  'Actual official decoder pixels disagree with all-frame CPU source QA')
                reader_bindings.append({'sample_id':raw['sample_id'],'worker':p['worker'],
                    'task':raw['task'],'stream_sha256':qa['stream_sha256'],'frames':180,
                    'included_in_current1200':raw['sample_id'] in spec['assigned_ids']})
        if retained:
            retired.extend({**r,'excluded_reason':'independently_measured_classic_raster_source_error'}
                           for r in rows if r['sample_id'] not in spec['assigned_ids'])
            rows=[r for r in rows if r['sample_id'] in spec['assigned_ids']]
        for row in rows:
            row['domain']=spec['domain'];item=catalog['items'][row['sample_id']]
            all_items[row['sample_id']]=item
            # Sealed inference catalogs deliberately omit generation lineage.
            # Pairing comes from the separately SHA-bound current source manifest.
            row['parent_classic_id']=source_by_id[row['sample_id']].get('parent_classic_id')
        records.extend(rows);workers.append(evidence)
    audit.require(len(records)==len({r['sample_id'] for r in records})==1200,'Missing or duplicate full1200')
    audit.require({r['sample_id'] for r in records}=={r['sample_id'] for r in current_sources},
                  'Results do not exactly match current source IDs')
    audit.require(len(retired)==plan.get('prior_invalid_source_calls_preserved',0),
                  'Preserved excluded-source count differs')
    audit.require(Counter((r['domain'],r['task']) for r in records)
                  =={('classic','T4'):408,('natural','T4'):408,('classic','T5'):192,('natural','T5'):192}, 'Wrong task/domain union')
    groups=defaultdict(list);zero=[]
    for row in records:
        p=row['parameters']
        if row['task']=='T4':
            coherence=p['coherence']
            audit.require(row['eligible']==(coherence>0),'Zero coherence included in accuracy')
            groups[(row['domain'],'T4',coherence,'all')].append(row)
            if not row['eligible']:zero.append(row)
        else:
            scenario=p['scenario'];scenario='target_absent' if scenario.startswith('target_absent') else scenario
            groups[(row['domain'],'T5',p['mask'],scenario)].append(row)
    cells=[]
    for (domain,task,level,subgroup),rows in groups.items():
        n=24 if task=='T4' and level==0 else 48 if task=='T4' else 48 if subgroup=='target_absent' else 24
        audit.require(len(rows)==n,'Condition denominator differs')
        cells.append({'domain':domain,'task':task,'level':level,'subgroup':subgroup,**counts(rows)})
    by_id={r['sample_id']:r for r in records};pairs=[]
    revision_map={r['parent_v1_source_id']:r['current_sample_id'] for r in arc.ref(plan['classic_replacement_map'])['items']} if 'classic_replacement_map' in plan else {}
    for row in records:
        if row['domain']!='natural':continue
        parent=revision_map.get(row['parent_classic_id'],row['parent_classic_id'])
        audit.require(parent in by_id and by_id[parent]['domain']=='classic','No fixed classic/natural pairing')
        old=by_id[parent]
        audit.require(old['task']==row['task'] and old['truth']==row['truth'] and old['eligible']==row['eligible'], 'Paired labels differ')
        left,right=source_by_id[parent],source_by_id[row['sample_id']]
        pair_fields=(('seed','coherence','direction','positions_xy_px','velocity_xy_px_s','signal_indices')
                     if row['task']=='T4' else
                     ('k','position_pair','onset','scenario','mask','angles_deg','centers_xy'))
        audit.require(all(k in left and k in right and left[k]==right[k] for k in pair_fields),
                      'Paired spatial or temporal conditions differ')
        pairs.append({'classic_id':parent,'natural_id':row['sample_id'],'task':row['task'],
            'truth':row['truth'],'eligible':row['eligible'],'classic_answer':old['answer'],'natural_answer':row['answer'],
            'changed':old['answer']!=row['answer'],'classic_correct':old['correct'],'natural_correct':row['correct']})
    audit.require(len(pairs)==len({r['classic_id'] for r in pairs})==600,'Incomplete paired600')
    paired_summary=[]
    for task in ('T4','T5'):
        pr=[r for r in pairs if r['task']==task and r['eligible']]
        paired_summary.append({'task':task,'n':len(pr),'changed':sum(r['changed'] for r in pr),
            'correct_to_wrong':sum(r['classic_correct'] and not r['natural_correct'] for r in pr),
            'wrong_to_correct':sum(not r['classic_correct'] and r['natural_correct'] for r in pr)})
    summary={'status':'COMPLETE_COSMOS_X_T45_1200','unique_model_calls':1200,
        'unique_model_calls_definition':'One accepted answer per current source ID; physical calls including excluded source revisions are counted separately.',
        'source_frames':216000,'accuracy_n':1152,'zero_coherence_bias_only_n':48,
        'preserved_invalid_source_calls':len(retired),'physical_model_calls_including_preserved_invalid_sources':1200+len(retired),
        'previous_unchanged_source_calls_reused':sum(len(s['assigned_ids']) for s in plan['shards'].values() if not s.get('run_new_calls',True) and not s.get('preclaim_refusal_admission')),
        'completed_calls_retained_after_preclaim_stat_refusal':sum(len(s['assigned_ids']) for s in plan['shards'].values() if s.get('preclaim_refusal_admission')),
        'by_task_domain':[{'domain':d,'task':t,**counts([r for r in records if r['domain']==d and r['task']==t])}
            for d in ('classic','natural') for t in ('T4','T5')],
        'cells':cells,'paired_changes':paired_summary,'workers':workers,'execution_plan':selected['execution_plan'] if selected else audit.descriptor(plans/'execution-plan.json'),
        'final_source_audit':selected['final_source_audit'] if selected else None,
        'actual_deep_reader_source_byte_checks':reader_bindings,
        'actual_deep_reader_scope':'First real request of each task per worker; all remaining requests retain reader count/order/shape and frozen source identity, not full GPU byte observations.',
        'statistical_scope':'24 layout seeds for T4; 24 onset-position blocks for T5. Paired raw curves, no threshold fit forced.',
        'T4_construct_scope':'Final position is an available direction cue; these scores alone do not establish temporal integration. See source-cue-diagnostic.json for a fixed geometry-only rule.',
        'source_cue_diagnostic':audit.descriptor(root/'runs/reproducibility'/NAME/'source-cue-diagnostic.json'),
        'inference_scope':'Official Cosmos X native video+T5 reference; no model numeric patch. Limited prior reproducibility evidence, not arbitrary-hardware determinism.'}
    out=root/'runs/reproducibility'/NAME/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'records.json',records)
    atomic_json(out/'preserved-invalid-source-records.json',retired)
    audit.write_csv(out/'cells.csv',cells);audit.write_csv(out/'classic-natural-paired.csv',pairs)
    audit.write_csv(out/'timing.csv',audit.timing_tables(records))
    render(out,cells)
    atomic_json(out/'summary.json',summary)
    return {k:summary[k] for k in ('status','unique_model_calls','accuracy_n','zero_coherence_bias_only_n','by_task_domain')}


def render(out,cells):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'classic':'#2667ad','natural':'#db7a35'}
    fig,ax=plt.subplots(figsize=(7.3,4.6))
    for domain in colors:
        rows=sorted([r for r in cells if r['task']=='T4' and r['domain']==domain and float(r['level'])>0],key=lambda r:float(r['level']))
        ax.plot([100*float(r['level']) for r in rows],[100*r['accuracy'] for r in rows],marker='o',color=colors[domain],label=domain.title())
    ax.axhline(50,color='#888',lw=1,ls='--');ax.set(xlabel='Coherence (%)',ylabel='Accuracy (%)',ylim=(0,103),title='Cosmos X · Coherent motion')
    ax.legend(frameon=False);ax.spines[['top','right']].set_visible(False);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('T4-classic-natural.'+ext),dpi=200)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(10.4,4.0),sharey=True)
    for ax,group,title in zip(axes,['target_first','target_second','target_absent'],['Target first · hit rate','Target second · hit rate','Target absent · false alarms']):
        for domain in colors:
            rows=[next(r for r in cells if r['task']=='T5' and r['domain']==domain and r['subgroup']==group and r['level']==mask) for mask in ('none','persistent')]
            rates=[r['YES']/r['n']*100 for r in rows]
            ax.plot([0,1],rates,marker='o',color=colors[domain],label=domain.title())
        ax.set(xticks=[0,1],xticklabels=['No mask','Mask'],ylim=(0,103),title=title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Responses (%)');axes[-1].legend(frameon=False);fig.suptitle('Cosmos X · Successive-event target detection');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('T5-classic-natural.'+ext),dpi=200)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    mode=p.add_mutually_exclusive_group();mode.add_argument('--status',action='store_true');mode.add_argument('--source-cues',action='store_true');a=p.parse_args()
    print(json.dumps(status(a.root) if a.status else source_cue_diagnostic(a.root) if a.source_cues else analyze(a.root)),flush=True)


if __name__=='__main__':main()
