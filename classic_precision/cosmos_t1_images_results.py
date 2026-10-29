"""Reconcile Cosmos 320 image requests and the authorized finite cold replay."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from .official_x_results import Archive, descriptor, journal, read, require, unique, timing_tables, write_csv
from .official_x_t1_results import expected_messages, processor_checks, backend_signature
from .storage import digest, file_sha256, atomic_json

NAME = 'cosmos-t1-images-20261029'


def audit_worker(archive, card, selected, *, startup=False, replay=False):
    path = archive.root / 'plans' / NAME / (card + ('-replay1' if replay else '') + '-profile.json')
    p = read(path); ids = p['assigned_ids']; ph = digest(p)
    require(p['model'] == 'cosmos' and p['input_arm'] == 'images' and p['maximum_model_calls'] == len(ids)
            and not p['automatic_retry'] and not p['formal_auto_resume'], 'Wrong finite image allocation')
    require(ids == [sid for sid, row in selected.items() if row['card'] == card], 'Sample/card order changed')
    parent = archive.ref(p['parent_validated_profile'])
    allowed = {'max_model_len':98304,'kv_cache_memory_bytes':25*2**30,
               'limit_mm_per_prompt':{'image':180,'video':0,'audio':0}}
    expected_settings = {**parent['engine_kwargs'], **allowed}
    require(p['engine_kwargs'] == expected_settings and p['environment'] == parent['environment']
            and p['checkpoint_assets'] == parent['checkpoint_assets'] and p['python'] == parent['python'],
            'Unexpected model/runtime changes beyond image capacity')
    require(p['request_mm_processor_kwargs'] == parent['request_mm_processor_kwargs'] ==
            {'do_resize':False,'do_sample_frames':False}, 'Request processing differs')
    for key in ('image_entry','t1_entry','observer_entry','implementation'):
        ref=p[key];local=Path(__file__).parent/Path(ref['path']).name
        require(file_sha256(local)==ref['sha256'], 'Frozen execution/observation source changed')
    catalog = archive.ref(p['source_catalog']); budget = archive.ref(p['t1_cpu_budget'])
    require(budget['status']=='PASS' and budget['model_calls']==0 and not budget['weights_loaded']
            and not budget['CUDA_initialized'], 'CPU prerequisite missing')
    attempts=unique(budget['attempts']);require(set(attempts)==set(ids), 'CPU expected-input coverage differs')
    out=archive.path(p['output'])
    if not startup:
        out=archive.path(p['output']); identity=read(out/'identity.json');wait=read(out.with_suffix('.wait.json'))
        launch=read(out.with_suffix('.launch.json'));end=read(out/'result.json');exit_=read(out/'exit.json')
        lease=read(out/'lease.json');owned=read(out/'engine-owner.json')['owned']
        require(identity['profile']==p and identity['profile_sha256']==ph and
                launch['profile_sha256']==wait['profile_sha256']==ph and launch['pid']==identity['owner']['pid'],
                'Profile/launch/wait binding differs')
        require(wait['returncode']==0 and wait['real_wait_observed'] and wait['owned_engines_exited']
                and wait['original_physical_lease_released'] and wait['terminal_status']=='COMPLETE', 'Worker not closed')
        outer_wait=read(out.with_suffix('.supervisor-wait.json'));outer_launch=read(out.with_suffix('.supervisor-launch.json'))
        require(outer_wait['returncode']==0 and outer_wait['real_wait_observed'] and
                outer_launch['pid']==launch['owner']['pid'], 'Persistent outer supervisor not closed')
        require(end['status']=='COMPLETE' and end['completed_ids']==wait['completed_ids']==ids
                and not end['UNCERTAIN_ids'] and not end['unattempted_ids'], 'Incomplete finite ledger')
        require(exit_['status']=='COMPLETE' and exit_['owned_engine_workers_exited'] and not exit_['shutdown_error']
                and exit_['owned_workers']==[o['worker'] for o in owned], 'Owned engine exit differs')
        require(all(lease[k]==p['physical_lock'][k] for k in ('path','st_dev','st_ino')) and lease['admission']['admitted'],
                'Original physical lease differs')
    actual=read(out/'public-settings.json'); cache=read(out/'cache-policy.json');runtime=read(out/'runtime.json')
    require(all(actual['LLM_kwargs'][k]==v for k,v in p['engine_kwargs'].items()), 'Actual engine settings differ')
    require(actual['request_mm_processor_kwargs']==p['request_mm_processor_kwargs'] and
            actual['chat_template_kwargs']=={'enable_thinking':False} and not actual['runtime_forward_patch']
            and not actual['runtime_processor_input_bridge'], 'Hidden model/input modifications')
    require(cache['fresh_engine'] and cache['source_file_page_cache_warmed'] and not cache['decoded_cache']
            and not cache['prefix_cache'] and cache['mm_processor_cache_gb']==0, 'Cache policy differs')
    versions=read(out/'versions.json')
    receipt=read(out.parent/('dependency-identity-'+('fourier' if card.startswith('fourier') else 'radon')+('-replay1' if replay else '')+'.json'))
    require(receipt['python']==p['python'] and receipt['profiles'][card]['sha256']==file_sha256(path) and receipt['model_calls']==0, 'Dependency receipt identity differs')
    require(all(receipt['versions'][k]==v for k,v in p['expected_packages'].items()) and
            all(receipt['versions'][k]==v for k,v in versions.items()), 'SDK mismatch')
    require(all(r['actual_kv_capacity']>=98304 for r in runtime), 'Insufficient actual image KV')
    raw=unique(journal(out/'raw.jsonl'));events=journal(out/'events.jsonl');timings=unique(journal(out/'request-timings.jsonl'))
    require(all(e['event'] in ('INTENT','TECHNICAL_PASS') for e in events), 'Unexpected failed event')
    intents=unique([e for e in events if e['event']=='INTENT']);passes=unique([e for e in events if e['event']=='TECHNICAL_PASS'])
    work_ids=ids[:1] if startup else ids
    if startup:
        require(set(work_ids)<=set(raw)&set(intents)&set(passes)&set(timings), 'Startup record incomplete')
    else:
        require(set(raw)==set(intents)==set(passes)==set(timings)==set(ids), 'Raw/PASS/timing/intent IDs differ')
    deeps=unique(journal(out/'deep-observations.jsonl'));admissions=unique(journal(out/'input-admissions.jsonl'))
    require(set(deeps)==set(admissions)=={ids[0]}, 'First-input GPU admission scope differs')
    records=[]
    for ordinal,sid in enumerate(work_ids):
        r=raw[sid];item=catalog['items'][sid];source=selected[sid];media=item['content'][:-1];expected=attempts[sid]
        require(item['row']['truth']==source['truth'] and item['row']['prompt']==source['prompt']
                and item['source_parameters']==source['parameters'] and item['ready']['video_sha256']==source['source_sha256'],
                'Original scientific source/GT/text mismatch')
        require(len(media)==180 and item['video_contract']=={'frames':180,'fps':60,'width':768,'height':512}, 'Source dimensions differ')
        for i,m in enumerate(media):
            require(m['type']=='image' and m['frame_index']==i and m['pixel_sha256']==item['frame_pixel_sha256'][i]
                    and Path(m['path']).name==f"{source['case_id']}_{i:03d}.png", 'Frame order/anonymous image path differs')
            require(p['media_files'][m['path']]['sha256']==m['sha256'], 'PNG sealed identity differs')
        require(r['sample_id']==sid and r['task']=='T1' and r['ordinal']==ordinal and r['profile_sha256']==ph
                and r['attempt']==1 and r['deep']==(ordinal==0), 'Raw allocation mismatch')
        require(r['source']['catalog']==p['source_catalog'] and r['source']['item_digest']==digest(item)
                and r['source']['media']==[p['media_files'][m['path']] for m in media], 'Raw source binding differs')
        require(r['request']==expected_messages(item) and r['question']==source['prompt'] and r['truth']==source['truth']
                and r['accuracy_eligible'] and r['request_mm_processor_kwargs']==p['request_mm_processor_kwargs'],
                'Actual message, anonymity, question, GT or processing changed')
        gen=r['generation'];require(all(gen[k]==v for k,v in {'n':1,'temperature':0.,'top_p':1.,'seed':0,
                 'repetition_penalty':1.,'max_tokens':16384,'ignore_eos':False,'stop':[],'stop_token_ids':[],
                 'choice':['LEFT','RIGHT'],'thinking':False,'native_EOS_override':False}.items()), 'Generation differs')
        answer=r['raw_answer'].strip();require(answer in ('LEFT','RIGHT') and r['finish_reason']=='stop'
                 and r['output_token_ids'] and r['correct']==(answer==r['truth']), 'Final answer or score invalid')
        require(len(r['actual_prompt_token_ids'])==expected['prompt_tokens'] and
                digest(r['actual_prompt_token_ids'])==expected['prompt_token_ids_sha256'], 'Actual logical image prompt differs')
        readers=r['native_reader'];require(len(readers)==180, 'Missing source reads')
        by_frame=unique(readers,'frame_index');require(set(by_frame)==set(range(180)), 'Frame read identities differ')
        for i,m in enumerate(media):
            v=by_frame[i];require(v['kind']=='image' and v['shape']==[512,768,3] and v['dtype']=='uint8'
                and v['path']==m['path'] and v['pixel_sha256']==m['pixel_sha256'], 'Image source pixel bytes differ')
        processor_checks(r,'images',expected)
        claim=read(archive.path(p['claim_root'])/(sid+'.json'))
        require(claim['sample_id']==sid and claim['attempt']==1 and claim['profile_sha256']==ph
                and claim['worker_token']==p['worker_token'], 'Claim differs')
        require(passes[sid]['raw_digest']==digest(r) and passes[sid]['ordinal']==intents[sid]['ordinal']==ordinal
                and passes[sid]['deep']==intents[sid]['deep']==r['deep'], 'PASS/intent binding differs')
        timing=timings[sid];require(timing['sample_id']==sid and timing['deep']==r['deep'] and
                0<timing['engine_request_inclusive_seconds']<=timing['end_to_end_durable_seconds'], 'Timing differs')
        if r['deep']:
            d=deeps[sid];a=admissions[sid];proof=a['proof']
            require(d['raw_digest']==a['raw_digest']==digest(r) and a['observations_digest']==digest(d['observations']), 'Deep binding differs')
            require(backend_signature([o['runtime'] for o in d['observations']])==backend_signature(runtime), 'Runtime deep identity differs')
            hashes=[z['sha256'] for z in expected['pixel_payloads']]
            require(proof['status']=='PASS_NATIVE_ORDERED_IMAGE_LIST' and proof['logical_image_count']==180
                    and proof['logical_patch_sha256_in_order']==hashes and
                    set(proof['actual_GPU_encoded_patch_sha256'])==set(hashes), 'Actual GPU image coverage differs')
        records.append({'sample_id':sid,'cell':source['cell'],'truth':r['truth'],'answer':answer,'correct':r['correct'],
            'parameters':source['parameters'],'source_video_sha256':source['source_sha256'],
            'prompt_tokens':len(r['actual_prompt_token_ids']),'prompt_token_sha256':digest(r['actual_prompt_token_ids']),
            'output_token_ids':r['output_token_ids'],'finish_reason':r['finish_reason'],'stop_reason':r['stop_reason'],
            'raw_digest':digest(r),'profile_sha256':ph,'worker':p['worker'],'card':card,'task':'T1',
            'deep':r['deep'],'timing':timing,'frame_pixel_sha256':item['frame_pixel_sha256'],
            'model':'cosmos','variant':'X-images','phase':p['phase'],'replicate':2 if replay else 1})
    if startup:
        return records,{'card':card,'scope':'FIRST_GPU_REQUEST_ONLY','logical_images':180,'actual_kv_capacity':runtime[0]['actual_kv_capacity'],'profile':descriptor(path),'model_calls_added_by_audit':0,'terminal_audited':False}
    return records,{'card':card,'worker':p['worker'],'n':len(ids),'profile':descriptor(path),
        'raw':descriptor(out/'raw.jsonl'),'wait':descriptor(out.with_suffix('.wait.json')),
        'load_seconds':read(out/'load.json')['seconds'],'worker_lifetime_seconds':wait['supervisor_monotonic_seconds']}


def analyze(root):
    archive=Archive(root);P=archive.root/'plans'/NAME;R=archive.root/'runs/reproducibility'/NAME
    plan=read(P/'plan.json');selection=archive.ref(plan['selection']);selected=unique(selection['samples'])
    require(plan['maximum_new_model_calls']==len(selected)==320 and len({tuple(s['cell']) for s in selected.values()})==80,
            'Unexpected total sample scope')
    records=[];workers=[]
    for card in ('fourier0','fourier1','radon0'):
        rows,evidence=audit_worker(archive,card,selected);records.extend(rows);workers.append(evidence)
    require(set(unique(records))==set(selected), 'Final union differs')
    old=unique(archive.ref(selection['video_baseline']));astra=unique(archive.ref(selection['astra_visual_baseline']))
    pairs=[];cells=defaultdict(list)
    for r in records:
        v=old[r['sample_id']];a=astra[r['sample_id']]
        require(v['truth']==a['truth']==r['truth'] and v['worker']=='cosmos-'+r['card']+'-X-full'
                and r['source_video_sha256'] in [h for k,h in v['source_media'] if k=='video'], 'Baseline source/card identity differs')
        pair={**{k:r[k] for k in ('sample_id','cell','truth','card')},'video':v['answer'],'images':r['answer'],
            'astra':a['answer'],'video_correct':v['correct'],'images_correct':r['correct'],'astra_correct':a['correct'],
            'changed':v['answer']!=r['answer'],'correct_to_wrong':v['correct'] and not r['correct'],
            'wrong_to_correct':not v['correct'] and r['correct'],'astra_returned_images':a['returned_images'],
            'astra_protocol_status':a['protocol_status'],'images_prompt_tokens':r['prompt_tokens'],
            'video_prompt_tokens':v['prompt_tokens']}
        pairs.append(pair);cells[tuple(r['cell'])].append(pair)
    cell_rows=[]
    for key,rows in sorted(cells.items()):
        require(len(rows)==4 and Counter(r['truth'] for r in rows)=={'LEFT':2,'RIGHT':2}, 'Cell incomplete/unbalanced')
        cell_rows.append({'cell':list(key),'n':4,**{k:sum(r[k] for r in rows) for k in
              ('video_correct','images_correct','astra_correct','changed','correct_to_wrong','wrong_to_correct')}})
    out=R/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'records.json',records);write_csv(out/'paired.csv',pairs);write_csv(out/'cells.csv',cell_rows)
    timing=[]
    for w in workers:
        timing.extend({'worker':w['worker'],**t} for t in timing_tables([r for r in records if r['worker']==w['worker']]))
    write_csv(out/'timing.csv',timing)
    summary={'status':'COMPLETE_320_IMAGE_COMPARISON','schema':'cosmos-T1-images-results-v1','n':320,'cells':80,
             **{k:sum(r[k] for r in pairs) for k in ('video_correct','images_correct','astra_correct','changed','correct_to_wrong','wrong_to_correct')},
             'image_answers':dict(Counter(r['answer'] for r in records)),'workers':workers,
             'primary_calls':320,'replay_calls':0,'same_source_same_card_video_baseline':True,
             'limitations':['Four samples per cell; 100% means only 4/4.','Native image temporal positions and patching differ from video, plus required context/KV capacity increases.',
                 'Astra is descriptive: different model, tool interface and visual-only instructions; observed frame counts retained.',
                 'Primary 320 are a single pass; the separately authorized replay panel is reported separately.'],
             'outputs':{f:descriptor(out/f) for f in ('records.json','paired.csv','cells.csv','timing.csv')}}
    atomic_json(out/'summary.json',summary)
    render(summary,cell_rows,out)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('workers','outputs')},ensure_ascii=False))
    return summary


def render(summary,cells,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    spatial=sorted({r['cell'][1] for r in cells})
    fig,axes=plt.subplots(2,4,figsize=(13,6.5),sharex=True,sharey=True)
    series=[('video_correct','Cosmos X video','#3268ac','-'),
            ('images_correct','Cosmos 180 images','#d57424','-'),
            ('astra_correct','Astra CLI (different interface)','#83858a','--')]
    for sf,ax in zip(spatial,axes.flat):
        rows=sorted([r for r in cells if r['cell'][1]==sf],key=lambda r:r['cell'][2])
        for key,label,color,style in series:
            ax.plot([r['cell'][2] for r in rows],[100*r[key]/r['n'] for r in rows],style,
                    color=color,marker='o',markersize=3,linewidth=1.7,label=label)
        ax.set_title(f'SF {sf:g} cycles/aperture',fontsize=11);ax.axhline(50,color='#cccccc',lw=.8)
        ax.set_ylim(-4,104);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.15)
        ax.spines[['top','right']].set_visible(False)
    for ax in axes[1]:ax.set_xlabel('Modulation depth')
    for ax in axes[:,0]:ax.set_ylabel('Correct (%)')
    fig.suptitle('T1 · same 320 source clips',fontsize=17,y=.99)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.95),ncol=3,frameon=False)
    fig.text(.5,.015,'4 clips per point (2 LEFT, 2 RIGHT) · 100% means 4/4 · no fitted threshold',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.04,1,.88));fig.savefig(out/'T1-video-images-Astra.png',dpi=180)
    fig.savefig(out/'T1-video-images-Astra.pdf');plt.close(fig)


def audit_replays(root,primary_summary):
    archive=Archive(root);plans=archive.root/'plans'/NAME;run=archive.root/'runs/reproducibility'/NAME
    plan=read(plans/'replay1-plan.json');selection=archive.ref(plan['selection'])
    require(plan['maximum_new_model_calls']==36 and len(plan['ids'])==len(set(plan['ids']))==12
            and not plan['automatic_retry'] and plan['same_card_pairs']==12 and plan['cross_card_pairs']==24,
            'Authorized replay scope changed')
    primary=unique(read(run/'analysis/records.json'));records=[];workers=[]
    for card in ('fourier0','fourier1','radon0'):
        chosen=unique([{**s,'origin_card':s['card'],'card':card} for s in selection['samples']])
        p=read(plans/(card+'-replay1-profile.json'));parent=read(plans/(card+'-profile.json'))
        require(p['assigned_ids']==plan['ids'] and p['cold']==2 and p['phase']=='images-replay'
                and p['engine_kwargs']==parent['engine_kwargs'] and
                p['parent_validated_profile']['sha256']==file_sha256(plans/(card+'-profile.json')),
                'Cold replay differs from primary image configuration')
        rows,evidence=audit_worker(archive,card,chosen,replay=True);records.extend(rows);workers.append(evidence)
    pairs=[];grouped=defaultdict(list)
    for r in records:
        a=primary[r['sample_id']]
        require(r['frame_pixel_sha256']==a['frame_pixel_sha256'] and r['truth']==a['truth'] and
                r['prompt_token_sha256']==a['prompt_token_sha256'], 'Replay source pixels/prompt/GT changed')
        token_same=(r['output_token_ids'],r['finish_reason'],r['stop_reason'])==(
                    a['output_token_ids'],a['finish_reason'],a['stop_reason'])
        pairs.append({'sample_id':r['sample_id'],'initial_card':a['card'],'replay_card':r['card'],
            'same_card':a['card']==r['card'],'initial_answer':a['answer'],'replay_answer':r['answer'],
            'answer_equal':a['answer']==r['answer'],'full_output_equal':token_same,
            'prompt_tokens_equal':True,'source_pixels_equal':True})
        grouped[r['sample_id']].append(r)
    require(len(pairs)==36 and sum(x['same_card'] for x in pairs)==12, 'Replay pair count differs')
    result={'status':'PASS_36_FULL_OUTPUT_MATCHES' if all(p['full_output_equal'] for p in pairs) else 'OBSERVED_OUTPUT_DIFFERENCES',
        'unique_ids':12,'new_calls':36,'same_card':{'n':12,'full_output_equal':sum(p['full_output_equal'] for p in pairs if p['same_card'])},
        'cross_card':{'n':24,'full_output_equal':sum(p['full_output_equal'] for p in pairs if not p['same_card'])},
        'all_answers_equal':sum(p['answer_equal'] for p in pairs),
        'three_new_cold_engines_equal_ids':sum(len({digest([r['output_token_ids'],r['finish_reason'],r['stop_reason']]) for r in rows})==1 for rows in grouped.values()),
        'workers':workers,'scope':'Twelve source IDs, four from each original card, now run in the same order in three fresh engines. Finite evidence, not a guarantee for every input/hardware.'}
    out=run/'analysis';atomic_json(out/'replay-records.json',records);write_csv(out/'replay-paired.csv',pairs)
    atomic_json(out/'replay-summary.json',result)
    summary={**primary_summary,'replay_calls':36,'reproduction':result}
    atomic_json(out/'summary.json',summary)
    return summary


def launch_replays(root,summary,hosts,ssh):
    """Only this explicit post-primary gate can submit the 36 authorized replays."""
    import shlex
    import subprocess
    root=Path(root);plans=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME
    plan=read(plans/'replay1-plan.json')
    require(plan['status']=='READY_WAITING_PRIMARY320' and plan['maximum_new_model_calls']==36
            and not plan['automatic_retry'] and summary['status']=='COMPLETE_320_IMAGE_COMPARISON', 'Replay gate not satisfied')
    primary=run/'analysis/primary-audit-before-replay.json'
    with primary.open('xb') as f:f.write((run/'analysis/summary.json').read_bytes())
    gate={'schema':'cosmos-image-replay-admission-v1','primary_status':summary['status'],
          'primary_audit_sha256':file_sha256(primary),'authorization_plan':descriptor(plans/'replay1-plan.json'),
          'maximum_new_model_calls':36,'automatic_retry':False,
          'profiles':{c:descriptor(plans/(c+'-replay1-profile.json')) for c in plan['cards']}}
    gate_path=run/'replay1-admission.json';atomic_json(gate_path,gate)
    for name,host,remote,py,cards in hosts:
        for local in (gate_path,primary):
            subprocess.run(['scp','-q','-i','/Users/qianyang/.ssh/cirp_codex_tmp','-o','BatchMode=yes',str(local),
                host+':'+remote+'/runs/reproducibility/'+NAME+'/'+local.name],check=True)
        script=f'''import json,hashlib,subprocess,os
from pathlib import Path
root=Path({remote!r});run=root/'runs/reproducibility/{NAME}';gate=json.loads((run/'replay1-admission.json').read_text())
assert gate['primary_status']=='COMPLETE_320_IMAGE_COMPARISON' and gate['maximum_new_model_calls']==36 and not gate['automatic_retry']
assert hashlib.sha256((run/'primary-audit-before-replay.json').read_bytes()).hexdigest()==gate['primary_audit_sha256']
for card in {cards!r}:
 path=root/'plans/{NAME}'/(card+'-replay1-profile.json');p=json.loads(path.read_text())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==gate['profiles'][card]['sha256'] and p['maximum_model_calls']==12 and not p['automatic_retry']
 old=json.loads(Path(p['parent_validated_profile']['path']).read_text());wait=json.loads(Path(old['output']).with_suffix('.wait.json').read_text())
 assert wait['returncode']==0 and wait['real_wait_observed'] and wait['owned_engines_exited'] and wait['original_physical_lease_released']
 out=Path(p['output'])
 with out.with_suffix('.submission.json').open('x') as f:json.dump({{'profile':str(path),'profile_sha256':gate['profiles'][card]['sha256'],'primary_audit_sha256':gate['primary_audit_sha256'],'maximum_model_calls':12,'automatic_retry':False}},f)
 with out.with_suffix('.controller.log').open('xb') as log:child=subprocess.Popen([{py!r},'-B',str(run/'launch_controller.py'),str(path)],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
 print(json.dumps({{'card':card,'controller_pid':child.pid,'maximum_model_calls':12}}),flush=True)
'''
        result=subprocess.run(ssh+[host,shlex.join([py,'-B','-'])],input=script,text=True,capture_output=True,timeout=30)
        (run/('replay1-submission-'+name+'.log')).write_text(result.stdout+result.stderr)
        require(result.returncode==0,'Finite replay submission failed; no retry: '+name)


# A one-shot job dependency wait. It observes only the two supervisor-completion
# files through Linux inotify; it does not poll GPU health or submit model jobs.
REMOTE_WAIT = r'''
import ctypes,json,os,select,sys,time
from pathlib import Path
root=Path(sys.argv[1]);cards=json.loads(sys.argv[2]);suffix=sys.argv[3] if len(sys.argv)>3 else '';lib=ctypes.CDLL(None,use_errno=True)
fd=lib.inotify_init1(os.O_CLOEXEC);assert fd>=0
assert lib.inotify_add_watch(fd,os.fsencode(root),0x8|0x80|0x100)>=0
deadline=time.monotonic()+14400
while True:
    result={}
    for card in cards:
        p=root/('cosmos-'+card+'-X-images'+suffix+'.supervisor-wait.json')
        try: result[card]=json.loads(p.read_text())
        except (FileNotFoundError,json.JSONDecodeError): pass
    if len(result)==len(cards):
        print(json.dumps(result),flush=True);break
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Finite image job completion not observed within four hours')
    if select.select([fd],[],[],remaining)[0]:os.read(fd,65536)
os.close(fd)
'''


def finalize(root):
    """Close primary320, then run authorized replay36, audit and render locally."""
    import concurrent.futures
    import datetime
    import os
    import shlex
    import subprocess
    import traceback
    root=Path(root);run=root/'runs/reproducibility'/NAME
    state_path=run/'finalizer-state.json'
    replay_plan=read(root/'plans'/NAME/'replay1-plan.json')
    require(replay_plan['status']=='READY_WAITING_PRIMARY320' and replay_plan['maximum_new_model_calls']==36,
            'Authorized replay is not ready')
    state={'status':'WAITING_FOR_PRIMARY320_CLOSURE','pid':os.getpid(),'primary_call_budget':320,'replay_call_budget':36,
           'recurring_schedule_changed':False,'execution_entry':descriptor(__file__)}
    with state_path.open('x') as f:json.dump(state,f,indent=2)
    hosts=[('fourier','yang@100.65.75.63','/home/yang/cvpr2027_benchmark','/home/yang/miniconda3/envs/qy/bin/python',['fourier0','fourier1']),
           ('radon','q@100.106.240.56','/mnt/sda/q/cvpr2027_benchmark','/home/q/miniconda3/envs/qwen_next/bin/python',['radon0'])]
    ssh=['ssh','-i','/Users/qianyang/.ssh/cirp_codex_tmp','-o','BatchMode=yes','-o','ServerAliveInterval=30']
    try:
        def wait_host(job,suffix=''):
            name,host,remote,py,cards=job
            args=[py,'-B','-c',REMOTE_WAIT,remote+'/runs/reproducibility/'+NAME,json.dumps(cards),suffix]
            result=subprocess.run(ssh+[host,shlex.join(args)],capture_output=True,text=True,timeout=14500)
            (run/('completion-listener-'+name+suffix+'.log')).write_text(result.stdout+result.stderr)
            require(result.returncode==0,'Completion listener failed: '+name)
            receipt=json.loads(result.stdout);atomic_json(run/('completion-listener-'+name+suffix+'.json'),receipt)
            return receipt
        def copy_closed():
            for name,host,remote,py,cards in hosts:
                for folder,exclude in [('plans','--exclude=plan.json'),('runs/reproducibility','--exclude=media/')]:
                    subprocess.run(['rsync','-a',exclude,'-e',shlex.join(ssh),
                        host+':'+remote+'/'+folder+'/'+NAME+'/',str(root/folder/NAME)+'/'],check=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(wait_host,hosts):
                print(json.dumps({'worker_closure_observed':result}),flush=True)
        state['status']='COPYING_CLOSED_EVIDENCE';atomic_json(state_path,state)
        # Closed scientific data are copied once. Reconstructible 57,600 PNGs
        # remain on their execution hosts, bound by per-frame hashes and readers.
        copy_closed()
        require(descriptor(__file__)==state['execution_entry'],'Analysis changed during one-shot wait')
        summary=analyze(root)
        launch_replays(root,summary,hosts,ssh)
        state.update(status='WAITING_FOR_AUTHORIZED_REPLAY36_CLOSURE',primary_completed=320,replay_profiles_submitted=3)
        atomic_json(state_path,state)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            for result in pool.map(lambda job:wait_host(job,'-replay1'),hosts):
                print(json.dumps({'replay_closure_observed':result}),flush=True)
        copy_closed()
        require(descriptor(__file__)==state['execution_entry'],'Analysis changed during one-shot replay wait')
        summary=audit_replays(root,summary)
        completed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-10))).isoformat()
        block=(f'**{completed}：Cosmos 180图片 T1有限比较已完整验收320/320，80点各4条。** '
               f'同ID视频X {summary["video_correct"]}/320 → 图片 {summary["images_correct"]}/320；'
               f'改答{summary["changed"]}，对转错{summary["correct_to_wrong"]}、错转对{summary["wrong_to_correct"]}。'
               f'Astra同ID {summary["astra_correct"]}/320（不同工具/提示/实际看帧数，只作描述）。\n'
               '全部原始回答、完整tokens、180图顺序/像素/processor/首GPU证据和三卡真实wait、owned退出、原锁释放通过。'
               '主X视频曲线保留；本轮只改输入形式并相应增至98304上下文/25GiB KV。\n'
               f'独立12输入三卡冷启动复验36次：同卡完整输出匹配{summary["reproduction"]["same_card"]["full_output_equal"]}/12，'
               f'跨卡匹配{summary["reproduction"]["cross_card"]["full_output_equal"]}/24。只代表此固定面板，不推广所有输入/硬件。\n'
               f'完整逐ID、80点比较和PNG/PDF：`runs/reproducibility/{NAME}/analysis/`。'
               'cvpr-4保持原暂停状态；本轮单次回收器已结束，没有继续推理。\n')
        start='<!-- cosmos-t1-images-current -->';end='<!-- /cosmos-t1-images-current -->'
        for path in (root.parent/'cvpr2027/LATEST.md',root/'EXPERIMENT_PLAN.md',root/'report.md'):
            text=path.read_text();require(start in text and end in text,'Missing dedicated current block: '+str(path))
            before,tail=text.split(start,1);_,after=tail.split(end,1);path.write_text(before+start+'\n'+block+end+after)
        registry_path=root/'runs/t2-t3-rgb16/registry.json';registry=read(registry_path)
        registry['execution']['cosmos_t1_images_20261029'].update(status='COMPLETE_320_IMAGE_COMPARISON_AND_36_REPLAY',completed=320,replay_completed=36,reproduction_status=summary['reproduction']['status'],
                updated_HST=completed,summary=descriptor(run/'analysis/summary.json'))
        atomic_json(registry_path,registry)
        state.update(status='COMPLETE_320_IMAGE_COMPARISON_AND_36_REPLAY',completed_HST=completed,primary_completed=320,
                     replay_completed=36,summary=descriptor(run/'analysis/summary.json'))
        atomic_json(state_path,state)
    except BaseException as exc:
        state.update(status='HOLD_FINALIZER_FAILED_NO_RETRY',error=repr(exc),traceback=traceback.format_exc())
        atomic_json(state_path,state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--startup',action='store_true');p.add_argument('--finalize-once',action='store_true');a=p.parse_args()
    if a.startup:
        archive=Archive(a.root);plan=read(archive.root/'plans'/NAME/'plan.json');selected=unique(archive.ref(plan['selection'])['samples']);proofs=[]
        for card in ('fourier0','fourier1','radon0'):
            records,evidence=audit_worker(archive,card,selected,startup=True);proofs.append({**evidence,'sample_id':records[0]['sample_id'],'prompt_tokens':records[0]['prompt_tokens']})
        result={'status':'PASS_THREE_FIRST_GPU_INPUTS','scope':'First authorized scientific call on each card; not terminal completion or image-route repeatability.','workers':proofs}
        atomic_json(archive.root/'runs/reproducibility'/NAME/'startup-acceptance.json',result);print(json.dumps(result))
    elif a.finalize_once:finalize(a.root)
    else:analyze(a.root)
