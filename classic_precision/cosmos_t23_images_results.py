"""Strict two-cell reconciliation and finite dependency-controlled submission.

The audit is scoped to T2/T3 task-first GPU records. It is kept separate from
an already running, frozen T1 finalizer. No scientific model implementation
is changed and no failed or completed source is retried.
"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import json
from pathlib import Path
from .official_x_results import Archive, descriptor, journal, read, require, unique, timing_tables, write_csv
from .official_x_t1_results import expected_messages, processor_checks, backend_signature
from .storage import digest, file_sha256, atomic_json
NAME='cosmos-t23-images-20261029'
PREDECESSOR='cosmos-t1-images-20261029'

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
    for key in ('image_entry','t1_entry','observer_entry','implementation','preparation_entry'):
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
    require(set(versions)=={'numpy','torch','transformers','vllm'}, 'Incomplete worker version receipt')
    # NumPy is recorded by the worker; the independent startup receipt records
    # the four profile-pinned packages, including FlashInfer instead of NumPy.
    require(all(receipt['versions'][k]==v for k,v in p['expected_packages'].items()) and
            all(receipt['versions'][k]==versions[k] for k in ('torch','transformers','vllm')), 'SDK mismatch')
    require(all(r['actual_kv_capacity']>=98304 for r in runtime), 'Insufficient actual image KV')
    raw=unique(journal(out/'raw.jsonl'));events=journal(out/'events.jsonl');timings=unique(journal(out/'request-timings.jsonl'))
    require(all(e['event'] in ('INTENT','TECHNICAL_PASS') for e in events), 'Unexpected failed event')
    intents=unique([e for e in events if e['event']=='INTENT']);passes=unique([e for e in events if e['event']=='TECHNICAL_PASS'])
    work_ids=ids[:1] if startup else ids
    if startup:
        require(set(work_ids)<=set(raw)&set(intents)&set(passes)&set(timings), 'Startup record incomplete')
    else:
        require(set(raw)==set(intents)==set(passes)==set(timings)==set(ids), 'Raw/PASS/timing/intent IDs differ')
    first_ids={next(sid for sid in ids if selected[sid]['task']==task) for task in {selected[sid]['task'] for sid in ids}}
    deeps=unique(journal(out/'deep-observations.jsonl'));admissions=unique(journal(out/'input-admissions.jsonl'))
    require(set(deeps)==set(admissions)==first_ids, 'First-input GPU admission scope differs')
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
        require(r['sample_id']==sid and r['task']==item['task']==source['task'] and r['ordinal']==ordinal and r['profile_sha256']==ph
                and r['attempt']==1 and r['deep']==(sid in first_ids), 'Raw allocation mismatch')
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
            'raw_digest':digest(r),'profile_sha256':ph,'worker':p['worker'],'card':card,'task':source['task'],
            'deep':r['deep'],'timing':timing,'frame_pixel_sha256':item['frame_pixel_sha256'],
            'model':'cosmos','variant':'X-images','phase':p['phase'],'replicate':2 if replay else 1})
    if startup:
        return records,{'card':card,'scope':'FIRST_GPU_REQUEST_ONLY','logical_images':180,'actual_kv_capacity':runtime[0]['actual_kv_capacity'],'profile':descriptor(path),'model_calls_added_by_audit':0,'terminal_audited':False}
    return records,{'card':card,'worker':p['worker'],'n':len(ids),'profile':descriptor(path),
        'raw':descriptor(out/'raw.jsonl'),'wait':descriptor(out.with_suffix('.wait.json')),
        'load_seconds':read(out/'load.json')['seconds'],'worker_lifetime_seconds':wait['supervisor_monotonic_seconds']}


def analyze(root):
    a=Archive(root);plans=a.root/'plans'/NAME;run=a.root/'runs/reproducibility'/NAME
    plan=read(plans/'plan.json');selected=unique(a.ref(plan['selection'])['samples'])
    require(plan['maximum_new_model_calls']==len(selected)==144 and plan['allocation']=={'fourier0':96,'fourier1':48}
            and Counter(x['task'] for x in selected.values())=={'T2':48,'T3':96},'Fixed two-cell allocation differs')
    records=[];workers=[]
    for card in plan['allocation']:
        rows,evidence=audit_worker(a,card,selected);records.extend(rows);workers.append(evidence)
    require(set(unique(records))==set(selected),'Missing/duplicate final sources')
    old=unique(a.ref(a.ref(plan['selection'])['video_baseline']));pairs=[];cells=defaultdict(list)
    for r in records:
        v=old[r['sample_id']]
        require(v['truth']==r['truth'] and v['cell']==r['cell'] and v['worker']=='cosmos-'+r['card']+'-X-full'
                and r['source_video_sha256'] in [h for k,h in v['source_media'] if k=='video'],'Video/source/card pairing differs')
        pair={**{k:r[k] for k in ('sample_id','task','cell','truth','card')},'video':v['answer'],'images':r['answer'],
            'video_correct':v['correct'],'images_correct':r['correct'],'changed':v['answer']!=r['answer'],
            'correct_to_wrong':v['correct'] and not r['correct'],'wrong_to_correct':not v['correct'] and r['correct']}
        pairs.append(pair);cells[tuple(r['cell'])].append(pair)
    comparison=[{'cell':list(cell),'n':len(rows),**{k:sum(r[k] for r in rows) for k in
                ('video_correct','images_correct','changed','correct_to_wrong','wrong_to_correct')}} for cell,rows in sorted(cells.items())]
    strata=[]
    for cell,rows in cells.items():
        for key in ('truth','tilt_sign','onset_frame'):
            values=defaultdict(list)
            for r in rows:
                value=r['truth'] if key=='truth' else selected[r['sample_id']]['parameters'].get(key)
                if value is not None:values[str(value)].append(r)
            for value,group in values.items():
                strata.append({'cell':list(cell),'stratum':key,'value':value,'n':len(group),**{k:sum(r[k] for r in group)
                    for k in ('video_correct','images_correct','changed','correct_to_wrong','wrong_to_correct')}})
    out=run/'analysis';out.mkdir(exist_ok=True)
    atomic_json(out/'records.json',records);write_csv(out/'paired.csv',pairs);write_csv(out/'comparison.csv',comparison);write_csv(out/'strata.csv',strata)
    matched=[]
    for r in records:
        v=old[r['sample_id']]
        if not r['deep'] and not v['deep']:
            matched.extend([r,{**v,'model':'cosmos','variant':'X-video','phase':'video-baseline','replicate':1,'card':r['card']}])
    timing=[]
    for card in plan['allocation']:
        timing.extend({'card':card,**t} for t in timing_tables([r for r in matched if r['card']==card]))
    write_csv(out/'timing.csv',timing)
    atomic_json(out/'timing-notes.json',{'scope':'Matched source IDs and original GPU; omit either-arm deep observations. Separate historical video/image runs, not concurrent randomized timings. Nested stages cannot be summed.','paired_ordinary_sources':len(matched)//2,'load_seconds_by_worker':{w['worker']:w['load_seconds'] for w in workers}})
    summary={'status':'COMPLETE_144_TWO_TASK_IMAGE_COMPARISON','unique_sources':144,'new_model_calls':144,
        'additional_replay_calls':0,'comparison':comparison,'workers':workers,'same_source_same_card':True,
        'reproduction_boundary':'T1 has its separate fixed 12-ID cold replay; T2/T3 here have one image answer per source and are not cold replays.',
        'model_math':'Unchanged frozen Cosmos X image entry; original prompts, official uint8, all180 logical images, no resize; image context98304/KV25GiB.',
        'total_changed':sum(r['changed'] for r in pairs),'selection':plan['selection']}
    atomic_json(out/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5));x=list(range(2));w=.32
    for dx,key,label,color in [(-w/2,'video_correct','X video','#176f82'),(w/2,'images_correct','X 180 images','#e49b43')]:
        bars=ax.bar([v+dx for v in x],[100*r[key]/r['n'] for r in comparison],w,label=label,color=color)
        ax.bar_label(bars,labels=[f"{r[key]}/{r['n']}" for r in comparison],padding=4)
    ax.set_xticks(x,['T2: BG .05, Δ .09, 500 ms','T3: 0.7°']);ax.set_ylim(0,108);ax.set_ylabel('Correct (%)');ax.axhline(50,color='#aaa',linestyle=':',lw=1);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,loc='lower right');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/('comparison.'+ext),dpi=180)
    plt.close(fig)
    return summary


def predecessor_gate(root):
    """Read-only dependency gate; an output mismatch is retained, not retried."""
    root=Path(root);run=root/'runs/reproducibility'/PREDECESSOR;state=read(run/'finalizer-state.json')
    require(state['status']=='COMPLETE_320_IMAGE_COMPARISON_AND_36_REPLAY','T1 predecessor not fully closed')
    require(state['primary_completed']==320 and state['replay_completed']==36,'Predecessor quota differs')
    summary=Archive(root).ref(state['summary'])
    require(summary['status']=='COMPLETE_320_IMAGE_COMPARISON' and summary['replay_calls']==36 and
            summary['reproduction']['new_calls']==36,'Predecessor audit incomplete')
    return descriptor(run/'finalizer-state.json'),state['summary']


def update_current(root,status,detail):
    import datetime
    root=Path(root);when=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=-10))).isoformat()
    start='<!-- cosmos-t23-images-current -->';end='<!-- /cosmos-t23-images-current -->'
    block=f'**{when}：{status}。** {detail}\n'
    for path in (root.parent/'cvpr2027/LATEST.md',root/'EXPERIMENT_PLAN.md',root/'report.md'):
        text=path.read_text();require(start in text and end in text,'Missing own current marker')
        before,tail=text.split(start,1);_,after=tail.split(end,1);path.write_text(before+start+'\n'+block+end+after)
    path=root/'runs/t2-t3-rgb16/registry.json';j=read(path);j['execution']['cosmos_t23_images_20261029'].update(status=status,updated_HST=when)
    atomic_json(path,j)


def finalize(root):
    import os,select,subprocess,shlex,time,traceback
    root=Path(root);plans=root/'plans'/NAME;run=root/'runs/reproducibility'/NAME
    require(read(plans/'plan.json')['status']=='READY_WAITING_T1_AND_REPLAY','T2/T3 CPU preparation not ready')
    state={'status':'WAITING_T1_320_AND_REPLAY36','pid':os.getpid(),'maximum_new_model_calls':144,'automatic_retry':False,'execution_entry':descriptor(__file__),'recurring_schedule_changed':False}
    statefile=run/'finalizer-state.json'
    with statefile.open('x') as f:json.dump(state,f,indent=2)
    host='yang@100.65.75.63';remote='/home/yang/cvpr2027_benchmark';py='/home/yang/miniconda3/envs/qy/bin/python'
    ssh=['ssh','-i','/Users/qianyang/.ssh/cirp_codex_tmp','-o','BatchMode=yes','-o','ServerAliveInterval=30']
    try:
        # Local filesystem event subscription, no GPU/host health polling.
        oldrun=root/'runs/reproducibility'/PREDECESSOR;fd=os.open(oldrun,os.O_RDONLY);watch=select.kqueue()
        watch.control([select.kevent(fd,filter=select.KQ_FILTER_VNODE,flags=select.KQ_EV_ADD|select.KQ_EV_CLEAR,fflags=select.KQ_NOTE_WRITE)],0,0)
        deadline=time.monotonic()+14400
        try:
            while True:
                previous=read(oldrun/'finalizer-state.json')
                if previous['status']=='COMPLETE_320_IMAGE_COMPARISON_AND_36_REPLAY':break
                require(not previous['status'].startswith('HOLD'),'Predecessor held; do not submit dependent scientific calls')
                require(time.monotonic()<deadline,'Predecessor did not close within four hours')
                watch.control(None,1,min(60,deadline-time.monotonic()))
        finally:watch.close();os.close(fd)
        previous,audit=predecessor_gate(root)
        require(descriptor(__file__)==state['execution_entry'],'Controller source changed during wait')
        gate={'previous':previous,'audit':audit,'maximum_new_model_calls':144,'allocation':{'fourier0':96,'fourier1':48},'profiles':{c:descriptor(plans/(c+'-profile.json')) for c in ('fourier0','fourier1')}}
        atomic_json(run/'admission.json',gate)
        # The exact closed source evidence remains in the preceding run.
        script=f'''import json,subprocess,os,hashlib
from pathlib import Path
r=Path({remote!r});n={NAME!r};run=r/'runs/reproducibility'/n;gate={gate!r}
for card,ncalls in gate['allocation'].items():
 pth=r/'plans'/n/(card+'-profile.json');p=json.loads(pth.read_text());assert hashlib.sha256(pth.read_bytes()).hexdigest()==gate['profiles'][card]['sha256'];assert p['maximum_model_calls']==ncalls and not p['automatic_retry']
 oldrun=r/'runs/reproducibility/{PREDECESSOR}'
 for suffix in ['', '-replay1']:
  old=json.loads((oldrun/('cosmos-'+card+'-X-images'+suffix+'.wait.json')).read_text())
  assert old['returncode']==0 and old['real_wait_observed'] and old['owned_engines_exited'] and old['original_physical_lease_released']
 out=Path(p['output'])
 with out.with_suffix('.submission.json').open('x') as f:json.dump({{'profile':str(pth),'profile_sha256':gate['profiles'][card]['sha256'],'predecessor_audit':gate['audit'],'maximum_model_calls':ncalls,'automatic_retry':False}},f)
 with out.with_suffix('.controller.log').open('xb') as log:child=subprocess.Popen([{py!r},'-B',str(run/'launch_controller.py'),str(pth)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 print(json.dumps({{'card':card,'controller_pid':child.pid,'quota':ncalls}}),flush=True)
'''
        result=subprocess.run(ssh+[host,shlex.join([py,'-B','-'])],input=script,capture_output=True,text=True,timeout=30)
        (run/'submission.log').write_text(result.stdout+result.stderr);require(result.returncode==0,'Finite launch failed; no retry')
        state['status']='WAITING_T23_144_CLOSURE';atomic_json(statefile,state)
        update_current(root,'RUNNING_144_NATIVE_IMAGES','T2 48＋T3 96已在原Fourier卡发出；原T1 320＋36复验先行闭合。仅本轮144，无自动重试；完整结果待实际raw/PASS/退出验收。')
        from .cosmos_t1_images_results import REMOTE_WAIT
        result=subprocess.run(ssh+[host,shlex.join([py,'-B','-c',REMOTE_WAIT,remote+'/runs/reproducibility/'+NAME,json.dumps(['fourier0','fourier1']),'-T23'])],capture_output=True,text=True,timeout=14500)
        (run/'completion-listener.log').write_text(result.stdout+result.stderr);require(result.returncode==0,'Completion listener failed')
        for folder,exclude in [('plans','--exclude=plan.json'),('runs/reproducibility','--exclude=media/')]:
            subprocess.run(['rsync','-a',exclude,'-e',shlex.join(ssh),host+':'+remote+'/'+folder+'/'+NAME+'/',str(root/folder/NAME)+'/'],check=True)
        require(descriptor(__file__)==state['execution_entry'],'Audit source changed during inference')
        summary=analyze(root);detail='；'.join(f"{r['cell'][0]} 视频{r['video_correct']}/{r['n']}→图片{r['images_correct']}/{r['n']}，改答{r['changed']}（对转错{r['correct_to_wrong']}／错转对{r['wrong_to_correct']}）" for r in summary['comparison'])
        update_current(root,summary['status'],detail+'。144唯一输入、原卡配对及两worker实际wait/退出/原锁释放通过。T2/T3没有新增冷复验，不将单次配对冒称可重复性证明。完整原答/逐ID/分层/计时及PNG/PDF在本轮analysis。')
        state.update(status=summary['status'],completed=144,summary=descriptor(run/'analysis/summary.json'));atomic_json(statefile,state)
    except BaseException as exc:
        state.update(status='HOLD_NO_RETRY',error=repr(exc),traceback=traceback.format_exc());atomic_json(statefile,state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--finalize-once',action='store_true');a=p.parse_args()
    if a.finalize_once:finalize(a.root)
    else:print(json.dumps(analyze(a.root)))
