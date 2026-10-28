"""Finite, persistent path-only Codex evaluations with a bounded dynamic pool.

The coordinator copies bytes, dispatches independent sessions, and audits logs.
It never decodes stimulus video or decides the visual answer for a session.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict, deque
import csv
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
import uuid

NAME = 'astra-path-only-20261027'
MODEL = 'gpt-6-astra'
EFFORT = 'medium'
SEED = 20261027
COUNTS = {'T1': 320, 'T2': 480, 'T3': 124}


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hst():
    return datetime.now(timezone(timedelta(hours=-10))).isoformat(timespec='seconds')


def write(path, value, *, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2) + '\n'
    if exclusive:
        with path.open('x') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
    else:
        temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
        with temp.open('x') as f:
            f.write(payload); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n'); f.flush(); os.fsync(f.fileno())


def paths(root):
    return Path(root)/'plans'/NAME, Path(root)/'runs/reproducibility'/NAME


def cell(row):
    p = row['nominal_parameters']; t = row['task']
    if t == 'T1':
        return (t, p['sf_cycles_per_aperture'], p['modulation_depth'])
    if t == 'T2':
        return (t, p['background_Y'], p['delta_Y'], p['offset_frame']-p['onset_frame'])
    return (t, p['angle_deg'])


def select(index):
    grouped = defaultdict(list)
    for row in index['items'].values():
        require(row['task'] in COUNTS, 'Unexpected task')
        grouped[cell(row)].append(row)
    selected = []
    for key, rows in sorted(grouped.items()):
        rng = random.Random(str(SEED) + json.dumps(key))
        onsets = rng.sample(range(24), 2)
        t = key[0]
        def placement(r):
            p = r['nominal_parameters']
            return p['bank_start_index'] if t == 'T1' else p['onset_index']
        picks = []
        if t == 'T3' and len({r['nominal_parameters']['tilt_sign'] for r in rows}) == 2:
            for sign, onset in zip([-1, 1], onsets):
                picks += [r for r in rows if placement(r) == onset
                          and r['nominal_parameters']['tilt_sign'] == sign]
        else:
            picks = [r for r in rows if placement(r) in onsets]
        require(len(picks) == 4 and Counter(r['truth'] for r in picks) == {'LEFT': 2, 'RIGHT': 2},
                f'Unbalanced or missing cell {key}')
        selected.extend(picks)
    require(Counter(r['task'] for r in selected) == COUNTS, 'Grid count changed')
    require(len({r['sample_id'] for r in selected}) == 924, 'Duplicate source ID')
    # Interleave tasks, independently shuffled before seeing any new answers.
    queues = {}
    for t in COUNTS:
        rr = [r for r in selected if r['task'] == t]
        random.Random(str(SEED)+t).shuffle(rr); queues[t] = deque(rr)
    ordered = []
    while any(queues.values()):
        for t in COUNTS:
            if queues[t]: ordered.append(queues[t].popleft())
    return ordered


def prepare(root):
    plan_dir, out = paths(root)
    require(not (plan_dir/'plan.json').exists(), 'Plan already exists; do not overwrite')
    index_path = Path(root)/'datasets/classic-t1-t3-v0.4/dataset-index.json'
    index = read(index_path)
    require(index['counts'] == {'T1':3840, 'T2':5760, 'T3':2928}, 'Wrong full source index')
    selected = select(index)
    plan_dir.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    t1 = (Path(root)/'plans/codex-local-visual-20261027/prompt.txt').read_text().strip()
    common = t1[t1.index('Use only direct visual inspection'):t1.index('Base your judgment on the direction')].strip()
    prompt_map = {}
    for t in COUNTS:
        bases = {r['prompt'] for r in index['items'].values() if r['task'] == t}
        require(len(bases) == 1, f'Fixed task prompt mismatch {t}')
        base = bases.pop()
        prompt = t1 if t == 'T1' else base+'\n'+common+'\nReturn exactly one uppercase word: LEFT or RIGHT.'
        require(prompt.startswith(base), 'Original scientific prompt changed')
        pp = plan_dir/(t+'-prompt.txt'); pp.write_text(prompt+'\n')
        prompt_map[t] = {'file':pp.name, 'sha256':sha(pp), 'original_task_prompt':base,
                         'change':'Original task text retained; shared visual-only restrictions appended'}
    records = []
    for ordinal, row in enumerate(selected):
        uid = uuid.uuid4().hex
        folder = Path('/tmp')/uuid.uuid4().hex
        neutral = folder/(uid+'.mkv')
        source = index_path.parent/row['video_relative']
        require(source.is_file() and sha(source) == row['video_sha256'], 'Source identity failed')
        require(row['frames'] == 180 and (row['width'],row['height']) == (768,512), 'Source geometry changed')
        folder.mkdir(mode=0o700)
        # A regular copy, never a symlink exposing an answer-bearing target path.
        shutil.copyfile(source, neutral)
        require(not neutral.is_symlink() and sha(neutral) == row['video_sha256'], 'Neutral copy changed')
        records.append({'ordinal':ordinal,'case_id':uid,'public_directory':str(folder),
                        'video_path':str(neutral),'sample_id':row['sample_id'],'task':row['task'],
                        'cell':list(cell(row)),'truth':row['truth'],'source':str(source),
                        'source_sha256':row['video_sha256'],'source_row':row})
    write(plan_dir/'samples.json', records, exclusive=True)
    model_cache = Path.home()/'.codex/models_cache.json'
    model_info = next(m for m in read(model_cache)['models'] if m['slug'] == MODEL)
    require(any(x['effort']==EFFORT for x in model_info['supported_reasoning_levels']), 'Effort unavailable')
    write(plan_dir/'model-metadata.json', model_info, exclusive=True)
    p = {'schema':'astra-path-only-v1','status':'SEALED','model':MODEL,'effort':EFFORT,
         'maximum_concurrent_sessions':20,'maximum_sessions':924,'counts':COUNTS,
         'nominal_cells':{'T1':80,'T2':120,'T3':31},'videos_per_cell':4,
         'source_index_id':index['index_id'],'source_index_sha256':sha(index_path),
         'samples_sha256':sha(plan_dir/'samples.json'),'prompts':prompt_map,'selection_seed':SEED,
         'selection':'Two LR pairs per cell; T3 both signs, except90deg one sign and two onsets; pre-answer seeded placements',
         'input':'Fixed prompt plus neutral video path only; no images attached or coordinator decoding',
         'session_policy':'One persistent Codex session per selected video; no automatic scientific retry',
         'scheduler':'Dynamic replenishment as each process truly exits, maximum20; PAUSE drains active sessions',
         'auth':'Existing workstation ChatGPT login; no API key, paid top-up or relogin',
         'sampling':'Official Codex Astra medium; temperature/top_p/seed not exposed by this CLI route; no deterministic guarantee',
         'reasoning_record':'Save CLI-exposed summaries and available rollout records; no hidden chain-of-thought extraction',
         'audit':'Decoder/tool/image coverage separately from answer; path receipt does not establish180frame inspection',
         'minimum_free_disk_GiB':40,'created_HST':hst(),
         'authorization':['每个点四个视频，测1-3task，Astra medium，一个视频一个Codex CLI',
                          '最大同时20个窗口在推理，一个好了就开下一个','直接指向video path']}
    write(plan_dir/'plan.json',p,exclusive=True)
    return {'status':'SEALED','counts':COUNTS,'total':924,'cells':231,'max_concurrent':20}


def settings():
    return {'model_provider':'openai','model_reasoning_effort':EFFORT,
            'approval_policy':'never','web_search':'disabled','project_doc_max_bytes':0,
            'project_root_markers':[], 'features.multi_agent':False,'features.apps':False,
            'features.memories':False,'features.code_mode':True,
            'features.view_image':True,'features.shell_tool':True,'features.plugins':False,
            'features.enable_request_compression':False,
            'developer_instructions':'Store any temporary files, including decoded frames, only inside the current working directory or its subdirectories. Use relative output paths. Each conversation has its own working directory; do not use shared absolute temporary paths.'}


def command(case, output, codex):
    cmd = [codex,'exec','--ignore-user-config','--ignore-rules','--strict-config',
           '--skip-git-repo-check','--sandbox','danger-full-access','--json',
           '-C',case['public_directory'],'-m',MODEL,'-o',str(output)]
    for key,value in settings().items(): cmd += ['-c',key+'='+json.dumps(value)]
    return cmd + ['-']


def prompt_for(plan_dir, plan, case):
    p = plan_dir/plan['prompts'][case['task']]['file']
    require(sha(p) == plan['prompts'][case['task']]['sha256'], 'Prompt identity changed')
    return p.read_text().rstrip()+'\n\nVideo path: '+case['video_path']+'\n'


def capture(root, codex):
    """Validate the real CLI request without contacting a model provider."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    import threading
    plan_dir,out,plan,samples=load(root);case=samples[0]
    folder=out/'cpu-admission';folder.mkdir(exist_ok=True)
    require(not (folder/'receipt.json').exists(), 'Already qualified')
    captured=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_GET(self):
            self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(json.dumps({'models':[read(plan_dir/'model-metadata.json')]}).encode())
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            texts=[]
            for item in body.get('input',[]):
                for part in item.get('content',[]):
                    require(part['type'] in ['input_text','output_text'],'Unexpected initial image')
                    texts.append(part['text'])
            joined='\n'.join(texts)
            expected=prompt_for(plan_dir,plan,case).strip()
            def names_in(items):
                result=[]
                for tool in items:
                    result.append(tool.get('name',tool.get('type')))
                    result.extend(names_in(tool.get('tools',[])))
                return result
            definitions=list(body.get('tools',[]))
            for item in body.get('input',[]):
                if item.get('type')=='additional_tools':definitions.extend(item.get('tools',[]))
            names=names_in(definitions)
            write(folder/'request-metadata.json',{'model':body.get('model'),'reasoning':body.get('reasoning'),
                  'tools':body.get('tools',[]),'service_tier':body.get('service_tier'),
                  'body_keys':list(body),'request_path':self.path})
            write(folder/'request.json',body)
            require(expected in joined,'Fixed user prompt/path absent')
            require(body['model']==MODEL and body.get('reasoning',{}).get('effort')==EFFORT,'Model/effort mismatch')
            tools_text=json.dumps(definitions)
            require('view_image' in tools_text and 'exec_command' in tools_text,'Frame tools unavailable')
            require(case['sample_id'] not in joined and 'samples.json' not in joined,'Private identity leak')
            receipt={'status':'PASS','scientific_model_calls':0,'initial_images':0,
                     'model':body['model'],'reasoning':body.get('reasoning'),
                     'exact_prompt_and_path':True,'tools':names,
                     'tool_capability':'exec_command and view_image supplied through additional_tools code-mode definitions; actual use audited separately',
                     'HST':hst()}
            captured.append(receipt)
            self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(b'{"error":{"message":"CPU qualification only; no model called","type":"cpu_admission"}}')
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    cmd=command(case,folder/'no-answer.txt',codex)
    cmd=cmd[:-1]+['--ephemeral','-c','model_provider="qualification"',
        '-c','model_providers.qualification.name="CPU admission"',
        '-c',f'model_providers.qualification.base_url="http://127.0.0.1:{server.server_port}/v1"',
        '-c','model_providers.qualification.wire_api="responses"',
        '-c','model_providers.qualification.requires_openai_auth=false',
        '-c','model_providers.qualification.supports_websockets=false',
        '-c','model_providers.qualification.request_max_retries=0',
        '-c','model_providers.qualification.stream_max_retries=0','-']
    try:
        with (folder/'events.jsonl').open('xb') as log,(folder/'stderr.log').open('xb') as err:
            child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=err,text=True)
            child.communicate(prompt_for(plan_dir,plan,case),timeout=90)
        write(folder/'wait.json',{'returncode':child.returncode,'real_wait':True,'model_calls':0},exclusive=True)
    finally:server.shutdown();server.server_close()
    require(len(captured)==1,'No qualified request; inspect CPU stderr')
    write(folder/'receipt.json',captured[0],exclusive=True)
    return captured[0]


def load(root):
    plan_dir,out=paths(root); plan=read(plan_dir/'plan.json')
    require(plan['model']==MODEL and plan['effort']==EFFORT and plan['maximum_concurrent_sessions']==20,
            'Model/effort/concurrency contract changed')
    require(sha(plan_dir/'samples.json')==plan['samples_sha256'], 'Sample manifest changed')
    samples=read(plan_dir/'samples.json')
    require(len(samples)==plan['maximum_sessions']==924, 'Budget mismatch')
    return plan_dir,out,plan,samples


def json_lines(path):
    if not Path(path).exists(): return []
    result=[]
    for line in Path(path).read_text(errors='replace').splitlines():
        try: result.append(json.loads(line))
        except ValueError: pass
    return result


def session_id(events):
    return next((r['thread_id'] for r in events if r.get('type')=='thread.started'),None)


def rollout_path(sid):
    if not sid: return None
    # Only our exact session, never read authentication or unrelated histories.
    candidates=list((Path.home()/'.codex/sessions').glob('*/*/*/*'+sid+'*.jsonl'))
    return max(candidates,key=lambda p:p.stat().st_mtime) if candidates else None


def safe_rollout_copy(source, target):
    """Retain content, tools and visible summaries; omit encrypted reasoning blobs."""
    def clean(v):
        if isinstance(v,dict):
            return {k:clean(x) for k,x in v.items() if k not in {'encrypted_content'}}
        if isinstance(v,list):return [clean(x) for x in v]
        return v
    with Path(target).open('x') as f:
        for r in json_lines(source): f.write(json.dumps(clean(r),ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())


def audit(folder,case,rc):
    events=json_lines(folder/'events.jsonl');sid=session_id(events)
    source=rollout_path(sid)
    if source and not (folder/'rollout.jsonl').exists():safe_rollout_copy(source,folder/'rollout.jsonl')
    rows=json_lines(folder/'rollout.jsonl')
    calls=[];contexts=[];visible=[];image_calls=[];image_blocks=0
    def count_images(value):
        if isinstance(value,dict):
            return int(value.get('type') in ['input_image','image','image_url'])+sum(count_images(v) for v in value.values())
        if isinstance(value,list):return sum(count_images(v) for v in value)
        return 0
    for r in rows:
        p=r.get('payload',{})
        image_blocks+=count_images(p)
        if r.get('type')=='turn_context':contexts.append(p)
        if r.get('type')=='response_item' and p.get('type') in ('function_call','custom_tool_call'):
            name=p.get('name','');args=p.get('arguments',p.get('input',''))
            try:parsed=json.loads(args) if isinstance(args,str) else args
            except ValueError:parsed={'raw':args}
            calls.append({'name':name,'arguments':parsed,'call_id':p.get('call_id')})
            if name.split('.')[-1]=='view_image':image_calls.append(parsed.get('path'))
        if r.get('type')=='response_item' and p.get('type')=='reasoning':
            visible.extend(x.get('text','') for x in p.get('summary',[]) if x.get('text'))
    images=[]
    for name in image_calls:
        path=Path(name) if name else Path('/nonexistent')
        if not path.is_absolute():path=Path(case['public_directory'])/path
        record={'path':str(path),'exists':path.is_file()}
        if path.is_file():
            with path.open('rb') as f: header=f.read(24)
            if header.startswith(b'\x89PNG\r\n\x1a\n'):
                record['width'],record['height']=struct.unpack('>II',header[16:24])
            record['sha256']=sha(path)
        images.append(record)
    shell=[c for c in calls if c['name'].split('.')[-1] in ('exec_command','shell_command','shell')]
    commands=[str(c['arguments'].get('cmd',c['arguments'].get('command',''))) for c in shell]
    decode_candidates=[c for c in commands if re.search(r'\bffmpeg\b|VideoCapture|av\.open',c)]
    suspect=[c for c in commands if re.search(r'optical|calcOptical|phaseCorrelate|fft|absdiff|hstack|vstack|tile=|crop=|scale=|select=|fps=|framestep|ffprobe|showinfo',c,re.I)]
    outside=[c for c in commands if any(k in c for k in ['dataset-index','samples.json','source_code','/cvpr2027_benchmark/','/runs/reproducibility/'])]
    answer_path=folder/'answer.txt'
    answer=answer_path.read_text(errors='replace') if answer_path.exists() else ''
    usage=[r.get('usage',{}) for r in events if r.get('type')=='turn.completed']
    errors=[r for r in events if r.get('type') in ('turn.failed','error')]
    auto='REQUIRES_TOOL_TRACE_REVIEW'
    if rc!=0 or errors:auto='EXECUTION_FAILED'
    elif not image_calls and not image_blocks:auto='NO_RECORDED_VISUAL_INPUT'
    elif len(set(image_calls))!=180:auto='INCOMPLETE_OR_NONSTANDARD_FRAME_COVERAGE'
    elif suspect or outside:auto='PROTOCOL_REVIEW_FLAG'
    elif any((i.get('width'),i.get('height'))!=(768,512) for i in images):auto='GEOMETRY_NOT_CONFIRMED'
    # This is evidence extraction, not a claim that unknown commands are safe.
    result={'schema':'astra-case-result-v1','case_id':case['case_id'],'session_id':sid,
            'original_rollout_path':str(source) if source else None,
            'resume_command':f'cd {case["public_directory"]} && codex resume {sid}' if sid else None,
            'returncode':rc,'real_wait':True,'answer_raw':answer,'strict_format':answer.strip() in ['LEFT','RIGHT'],
            'answer':answer.strip() if answer.strip() in ['LEFT','RIGHT'] else None,
            'protocol_status':auto,'model_effort_contexts':[{k:p.get(k) for k in ['model','effort','reasoning_effort']} for p in contexts],
            'tool_calls':len(calls),'view_image_calls':len(image_calls),'unique_viewed_paths':len(set(image_calls)),
            'image_content_blocks':image_blocks,
            'decode_command_candidates':len(decode_candidates),'suspect_commands':suspect,
            'outside_information_flags':outside,'usage':usage,'errors':errors,'finished_HST':hst()}
    write(folder/'tool-audit.json',{'calls':calls,'images_in_call_order':images,
                                   'decode_commands':decode_candidates,'limitations':'Static flags require trace review; no inference of hidden image tokens or internal resizing'})
    (folder/'visible-reasoning.txt').write_text('\n\n'.join(visible))
    write(folder/'result.json',result,exclusive=True)
    return result


async def one(root,plan,case,codex,slot,control=None):
    import psutil
    plan_dir,out=paths(root);folder=out/'cases'/case['case_id'];folder.mkdir(parents=True,exist_ok=True)
    control=Path(control) if control else out
    claim=out/'claims'/case['case_id'];claim.parent.mkdir(exist_ok=True)
    write(claim,{'case_id':case['case_id'],'claimed_HST':hst(),'attempt':1},exclusive=True)
    require(sha(case['video_path'])==case['source_sha256'],'Neutral video mutated')
    require([p.name for p in Path(case['public_directory']).iterdir()]==[Path(case['video_path']).name],
            'Public case folder is not pristine')
    prompt=prompt_for(plan_dir,plan,case);(folder/'prompt.txt').write_text(prompt)
    cmd=command(case,folder/'answer.txt',codex)
    env=os.environ.copy()
    for k in list(env):
        if k.startswith('CVPR_') or k in ['OPENAI_API_KEY','CODEX_API_KEY','OPENAI_BASE_URL','OPENAI_API_BASE']:
            env.pop(k,None)
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               TMPDIR=case['public_directory'],TEMP=case['public_directory'],TMP=case['public_directory'])
    # Bound only this owned child's CPU affinity; do not change shared host settings.
    available=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else []
    cores=available[(slot*2)%max(1,len(available)-4):(slot*2)%max(1,len(available)-4)+2]
    def child_setup():
        os.nice(10)
        if cores:os.sched_setaffinity(0,cores)
    start=time.monotonic(); descendants={}
    with (folder/'events.jsonl').open('xb') as stdout,(folder/'stderr.log').open('xb') as stderr:
        process=await asyncio.create_subprocess_exec(*cmd,stdin=asyncio.subprocess.PIPE,
            stdout=stdout,stderr=stderr,env=env,start_new_session=True,preexec_fn=child_setup)
        parent=psutil.Process(process.pid)
        write(folder/'launch.json',{'pid':process.pid,'create_time':parent.create_time(),'command':cmd,
              'model':MODEL,'effort':EFFORT,'cpu_affinity':cores,'started_HST':hst()},exclusive=True)
        stdin_error=None
        try:
            process.stdin.write(prompt.encode());await process.stdin.drain();process.stdin.close()
        except (BrokenPipeError,ConnectionResetError) as exc:
            stdin_error=repr(exc)
        waiter=asyncio.create_task(process.wait())
        while not waiter.done():
            try:
                for p in parent.children(recursive=True):descendants[p.pid]=p.create_time()
            except psutil.NoSuchProcess:pass
            await asyncio.wait({waiter},timeout=2)
        rc=await waiter
    # Do not refill a slot while owned decoder children are still running.
    while True:
        active=[]
        for pid,created in descendants.items():
            try:
                p=psutil.Process(pid)
                if p.create_time()==created and p.status()!=psutil.STATUS_ZOMBIE:active.append(pid)
            except psutil.NoSuchProcess:pass
        if not active:break
        write(folder/'waiting-owned-exit.json',{'pids':active,'CLI_returncode':rc})
        await asyncio.sleep(2)
    write(folder/'wait.json',{'returncode':rc,'real_wait':True,'owned_descendants_exited':True,
          'descendants':descendants,'stdin_error':stdin_error,
          'wall_seconds':time.monotonic()-start,'finished_HST':hst()},exclusive=True)
    result=audit(folder,case,rc)
    append(out/'completions.jsonl',{'case_id':case['case_id'],'session_id':result['session_id'],
           'returncode':rc,'answer':result['answer'],'protocol_status':result['protocol_status'],
           'wall_seconds':time.monotonic()-start,'finished_HST':hst()})
    # Quota/auth failures are not invitations to retry all claimed samples.
    error_text=(folder/'stderr.log').read_text(errors='replace')+'\n'+json.dumps(result['errors'])
    if re.search(r'usage limit|rate limit|429|quota exceeded|not authorized|authentication|insufficient_quota',error_text,re.I):
        write(control/'PAUSE.json',{'reason':'Provider quota/auth error; no automatic retries','case_id':case['case_id'],'HST':hst()})
    rollout=(folder/'rollout.jsonl').read_text(errors='replace') if (folder/'rollout.jsonl').exists() else ''
    if 'Failed RTM_NEWADDR' in rollout or 'bwrap: setting up uid map' in rollout:
        write(control/'PAUSE.json',{'reason':'Sandbox infrastructure failure; drain and inspect','case_id':case['case_id'],'HST':hst()})
    return result


async def dynamic_pool(items, limit, execute, paused=lambda:False, progress=None):
    """Real sliding window: no batch barrier and no duplicate scheduling."""
    pending=deque(items);active={};free=deque(range(limit));done=[]
    while pending or active:
        while pending and free and not paused():
            item=pending.popleft();slot=free.popleft()
            task=asyncio.create_task(execute(item,slot));active[task]=(item,slot)
        if progress:progress(len(pending),len(active),len(done))
        if not active:break
        ended,_=await asyncio.wait(active,timeout=5,return_when=asyncio.FIRST_COMPLETED)
        for task in ended:
            item,slot=active.pop(task);free.append(slot)
            try:done.append(await task)
            except Exception as exc:done.append({'error':repr(exc),'item':item})
    return done,list(pending)


def closed_ledger(out):
    records={}
    for p in sorted((out/'claims').iterdir()):
        folder=out/'cases'/p.name
        require((folder/'result.json').exists() and (folder/'wait.json').exists(), 'Unclosed claim: '+p.name)
        wait=read(folder/'wait.json')
        require(wait['real_wait'] and wait['owned_descendants_exited'], 'Owned CLI not closed')
        records[p.name]={'claim':sha(p),'result':sha(folder/'result.json'),'wait':sha(folder/'wait.json')}
    return records


def prepare_resume(root):
    plan_dir,out,plan,samples=load(root)
    parent=out/'segments/full-access1';control=out/'segments/quota-resume1'
    previous=read(parent/'scheduler-final.json');wait=read(parent/'supervisor.wait.json')
    require(previous['status']=='PAUSED_DRAINED' and previous['all_children_waited'], 'Parent not drained')
    require(wait['real_wait'] and wait['returncode']==0, 'Parent real wait missing')
    require(read(parent/'user-pause-closure.json')['status']=='USER_PAUSED_DRAINED', 'Independent closure missing')
    ledger=closed_ledger(out)
    require(len(ledger)==732, 'Paused ledger changed; inspect rather than replay')
    pending=[s for s in samples if s['case_id'] not in ledger]
    require(len(pending)==192 and all(s['task']!='T3' for s in pending), 'Unexpected remaining set')
    control.mkdir(exist_ok=False)
    manifest={'schema':'astra-quota-resume-v1','status':'SEALED','maximum_new_sessions':192,
        'preserved_prior_attempts':732,'case_ids':[s['case_id'] for s in pending],
        'prior_ledger_sha256':digest(ledger),'plan_sha256':sha(plan_dir/'plan.json'),
        'samples_sha256':sha(plan_dir/'samples.json'),'parent':str(parent),
        'model':MODEL,'effort':EFFORT,'same_video_retry':False,
        'scientific_prompt_and_selection_unchanged':True,'supplemental_angles':[1.75,2.25],
        'supplemental_policy':'Already attempted; retain separately from current primary protocol',
        'human_authorization':'让astra完成工作，让codexcliu配本地模型的也继续尝试一下个位数的',
        'created_HST':hst()}
    write(control/'continuation.json',manifest,exclusive=True)
    return {'status':'SEALED','segment':'quota-resume1','new_sessions':len(pending),'preserved':len(ledger)}


async def run_async(root,codex,segment=None):
    plan_dir,out,plan,samples=load(root);out.mkdir(parents=True,exist_ok=True)
    control=out if segment is None else out/'segments'/segment
    if segment=='quota-resume1':
        continuation=read(control/'continuation.json')
        require(continuation['status']=='SEALED' and continuation['same_video_retry'] is False, 'Continuation not sealed')
        require(sha(plan_dir/'plan.json')==continuation['plan_sha256']
                and sha(plan_dir/'samples.json')==continuation['samples_sha256'], 'Original inputs changed')
        ledger=closed_ledger(out)
        require(digest(ledger)==continuation['prior_ledger_sha256'], 'Prior answers/claims changed')
        samples=[s for s in samples if s['case_id'] in continuation['case_ids']]
        require(len(samples)==192 and not any(s['case_id'] in ledger for s in samples), 'Not exactly unattempted IDs')
    elif segment:
        require(segment=='full-access1','Unknown execution segment')
        previous=read(out/'scheduler-final.json')
        require(previous['status']=='PAUSED_DRAINED' and previous['all_children_waited'],'Previous scheduler not closed')
        require(read(out/'supervisor.wait.json')['real_wait'],'Previous supervisor wait missing')
        claims={p.name for p in (out/'claims').iterdir()}
        require(all((out/'cases'/cid/'wait.json').exists() for cid in claims),'Unclosed original claims')
        samples=[s for s in samples if s['case_id'] not in claims]
        require(len(samples)==866 and len(claims)==58,'Continuation ledger changed')
        control.mkdir(parents=True,exist_ok=True)
        write(control/'authorization.json',{'human_message':'不用沙箱，给full access就行',
              'maximum_new_sessions':866,'preserved_prior_sessions':58,'same_video_retry':False,
              'scientific_prompt_unchanged':True,'execution_settings':settings(),
              'temporary_paths':'Unique CWD and TMPDIR plus fixed developer file-placement instruction; not OS-enforced isolation',
              'HST':hst()},exclusive=True)
    lock_path=out/'scheduler.lock';lock_path.touch(exist_ok=True)
    with lock_path.open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (control/'scheduler-started.json').exists(), 'Already dispatched; inspect ledger, never replay')
        require(not (control/'PAUSE.json').exists(),'PAUSE exists')
        require(read(out/'cpu-admission/receipt.json')['status']=='PASS','Real CLI CPU qualification missing')
        version=subprocess.check_output([codex,'--version'],text=True).strip()
        write(control/'scheduler-started.json',{'pid':os.getpid(),'started_HST':hst(),'model':MODEL,
              'effort':EFFORT,'maximum_concurrent':20,'codex_version':version,
              'scheduled':len(samples),'segment':segment,'sandbox':'danger-full-access',
              'plan_sha256':sha(plan_dir/'plan.json'),'implementation_sha256':sha(__file__)},exclusive=True)
        shutil.copyfile(__file__,control/'execution-source.py')
        stopping=False
        def stop():
            nonlocal stopping
            stopping=True
            write(control/'PAUSE.json',{'reason':'Operator requested safe drain','HST':hst()})
        loop=asyncio.get_running_loop()
        for sig in [signal.SIGINT,signal.SIGTERM]:loop.add_signal_handler(sig,stop)
        def paused():
            if stopping or (control/'PAUSE.json').exists():return True
            if shutil.disk_usage('/tmp').free < plan['minimum_free_disk_GiB']*2**30:
                write(control/'PAUSE.json',{'reason':'Free disk below reserved40GiB; active sessions drain','HST':hst()});return True
            return False
        def progress(remaining,active,completed):
            write(control/'progress.json',{'remaining_unclaimed':remaining,'active_sessions':active,
                  'closed_sessions':completed,'maximum_concurrent':20,'paused':paused(),'HST':hst()})
        async def execute(case,slot):
            try:return await one(root,plan,case,codex,slot,control)
            except Exception as exc:
                write(out/'cases'/case['case_id']/'coordinator-error.json',{'error':repr(exc),'HST':hst()},exclusive=True)
                write(control/'PAUSE.json',{'reason':'Coordinator error; preserve claim, drain active','case_id':case['case_id'],'error':repr(exc),'HST':hst()})
                raise
        done,pending=await dynamic_pool(samples,20,execute,paused,progress)
        write(control/'scheduler-final.json',{'status':'COMPLETE' if not pending and len(done)==len(samples) else 'PAUSED_DRAINED',
              'closed':len(done),'unclaimed':len(pending),'all_children_waited':True,'finished_HST':hst()},exclusive=True)
        fcntl.flock(lock,fcntl.LOCK_UN)


def status(root):
    _,out,plan,samples=load(root)
    controls=[out]+[p for p in (out/'segments').glob('*') if (p/'scheduler-started.json').exists()]
    control=max(controls,key=lambda p:(p/'scheduler-started.json').stat().st_mtime if (p/'scheduler-started.json').exists() else 0)
    results=[];running=[];uncertain=[]
    for s in samples:
        d=out/'cases'/s['case_id']
        if (d/'result.json').exists():results.append(read(d/'result.json'))
        elif (out/'claims'/s['case_id']).exists():
            sid=session_id(json_lines(d/'events.jsonl'))
            running.append({'case_id':s['case_id'],'session_id':sid,'task':s['task']})
            if (d/'coordinator-error.json').exists():uncertain.append(s['case_id'])
    return {'status':'COMPLETE' if len(results)==924 else 'IN_PROGRESS',
            'maximum_sessions':924,'completed':len(results),'claimed_unclosed':len(running),
            'unclaimed':924-len(results)-len(running),'protocol_statuses':dict(Counter(r['protocol_status'] for r in results)),
            'progress':read(control/'progress.json') if (control/'progress.json').exists() else None,
            'pause':read(control/'PAUSE.json') if (control/'PAUSE.json').exists() else None,
            'active_sessions':running[:20],'uncertain':uncertain,'HST':hst()}


def report(root):
    _,out,plan,samples=load(root);rows=[]
    for s in samples:
        p=out/'cases'/s['case_id']/'result.json'
        if not p.exists():continue
        r=read(p)
        rows.append({'case_id':s['case_id'],'sample_id':s['sample_id'],'task':s['task'],
                     'cell_json':json.dumps(s['cell']),'truth':s['truth'],'answer':r['answer'],
                     'correct':r['answer']==s['truth'],'session_id':r['session_id'],
                     'protocol_status':r['protocol_status'],'view_image_calls':r['view_image_calls'],
                     'resume_command':r['resume_command']})
    if rows:
        with (out/'answers.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result=status(root);result['by_task']={t:{'answered':sum(r['answer'] is not None for r in rows if r['task']==t),
        'raw_correct_not_protocol_filtered':sum(r['correct'] for r in rows if r['task']==t)} for t in COUNTS}
    result['interpretation']='Descriptive raw answers only. No valid visual accuracy until tool traces establish compliance.'
    write(out/'summary.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','prepare-resume','capture','run','status','report'])
    p.add_argument('--root',required=True);p.add_argument('--codex',default='/home/yang/.local/bin/codex')
    p.add_argument('--segment',choices=['full-access1','quota-resume1'])
    a=p.parse_args()
    if a.action=='prepare':result=prepare(a.root)
    elif a.action=='prepare-resume':result=prepare_resume(a.root)
    elif a.action=='capture':result=capture(a.root,a.codex)
    elif a.action=='run':asyncio.run(run_async(a.root,a.codex,a.segment));result=status(a.root)
    elif a.action=='status':result=status(a.root)
    else:result=report(a.root)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
