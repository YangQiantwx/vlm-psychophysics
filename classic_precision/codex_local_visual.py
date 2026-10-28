"""Path-only Codex CLI visual experiment. The coordinator never decodes video.

The real agent receives the exact prompt and a neutral path and owns tool use.
CPU transport qualification is separate from model inference.
"""
from __future__ import annotations
import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
from .embodied_admission import require, read, new
from .storage import file_sha256
NAME='codex-local-visual-20261027'


def paths(root):
    root=Path(root)
    return root/'plans'/NAME,root/'runs/reproducibility'/NAME


def case_inputs(root,index):
    plan_dir,out=paths(root);p=read(plan_dir/'path-only-plan.json')
    require(p['model_input'].startswith('Only exact user prompt'),'Wrong path-only plan')
    require(file_sha256(plan_dir/'prompt.txt')==p['prompt_sha256'],'Prompt changed')
    case=p['samples'][index];evidence=read(out/'input-manifest.json')['records'][index]
    video=Path(case['public_directory'])/case['public_filename']
    require(str(video)==evidence['neutral_video'],'Neutral path changed')
    require(file_sha256(video)==evidence['source_sha256'],'Neutral video bytes changed')
    return case,(plan_dir/'prompt.txt').read_text()+'\nVideo path: '+str(video)+'\n'


def prepare_catalog(root):
    plan_dir,_=paths(root)
    m=dict(read(Path('/home/yang/.codex/models_cache.json'))['models'][0])
    template=m['model_messages']['instructions_template']
    m['base_instructions']=template.replace('You are Codex, an agent based on GPT-6.',
        'You are Codex, an agent using a local Cosmos model.',1)
    m.update(slug='cosmos-local',display_name='Local Cosmos NVFP4',
        description='Local model through the real Codex CLI agent',
        default_reasoning_level='none',supported_reasoning_levels=[],
        include_skills_usage_instructions=False,include_plugin_usage_instructions=False,
        include_apps_usage_instructions=False,support_verbosity=False,
        supports_image_detail_original=True,context_window=98304,max_context_window=98304,
        input_modalities=['text','image'],supports_search_tool=False,experimental_supported_tools=[],
        supports_reasoning_effort_updates=False,node_repl_disabled=True,use_responses_lite=False)
    for k in ['model_messages','tool_mode','multi_agent_version','multi_agent_reasoning_effort',
              'available_access_programs','availability_nux','comp_hash','service_tiers',
              'additional_speed_tiers','upgrade']:m.pop(k,None)
    # No replacement instructions: retain Codex's normal agent harness.
    new(plan_dir/'path-only-model-catalog.json',{'models':[m]})


def cli_command(root,case,port,output):
    plan_dir,_=paths(root)
    args=['/home/yang/.local/bin/codex','exec','--ignore-user-config','--ignore-rules',
        '--skip-git-repo-check','--ephemeral','--sandbox','workspace-write','--json',
        '-C',case['public_directory'],'-m','cosmos-local']
    settings={
        'model_provider':'cvpr_local',
        'model_providers.cvpr_local.base_url':f'http://127.0.0.1:{port}/v1',
        'model_providers.cvpr_local.name':'Local Cosmos',
        'model_providers.cvpr_local.wire_api':'responses',
        'model_providers.cvpr_local.requires_openai_auth':False,
        'model_providers.cvpr_local.supports_websockets':False,
        'model_providers.cvpr_local.request_max_retries':0,
        'model_providers.cvpr_local.stream_max_retries':0,
        'model_providers.cvpr_local.stream_idle_timeout_ms':600000,
        'model_catalog_json':str(plan_dir/'path-only-model-catalog.json'),
        'model_reasoning_effort':'none',
        'model_context_window':98304,'project_doc_max_bytes':0,'project_root_markers':[],
        'web_search':'disabled','features.shell_tool':True,
        'features.multi_agent':False,'features.enable_request_compression':False,
        'features.apps':False,'features.memories':False,'features.js_repl':False,
        'approval_policy':'never','sandbox_workspace_write.network_access':False}
    for k,v in settings.items():args+=['-c',k+'='+json.dumps(v)]
    args+=['--output-last-message',str(output)]
    require('--image' not in args and 'model_instructions_file' not in ' '.join(args),
            'Path-only must use normal CLI tools and no pre-attached images')
    return args


def effective_request(body):
    result=copy.deepcopy(body)
    # Qwen3-VL's official template has one leading system block and no developer
    # role. Preserve all Codex instruction text in Responses.instructions rather
    # than letting the model template silently omit those messages.
    instructions=[result.get('instructions','')]
    inputs=[]
    for item in result.get('input',[]):
        if item.get('role') in ['developer','system']:
            require(all(p.get('type') in ['input_text','output_text'] for p in item['content']),
                    'Unexpected media inside instruction message')
            instructions.extend(p['text'] for p in item['content'])
        else:inputs.append(item)
    result['instructions']='\n\n'.join(x for x in instructions if x)
    result['input']=inputs
    result.update(temperature=0,top_p=1,seed=0,max_output_tokens=4096,
        mm_processor_kwargs={'do_resize':False,'do_sample_frames':False},
        chat_template_kwargs={'enable_thinking':False},parallel_tool_calls=False)
    # Native vLLM supports function tools, not Codex's custom apply_patch grammar.
    removed=[t.get('name',t.get('type')) for t in result.get('tools',[])
             if t.get('type') not in ['function','namespace']]
    result['tools']=[t for t in result.get('tools',[]) if t.get('type') in ['function','namespace']]
    return result,removed


def validate_first_request(root,body,case):
    plan_dir,_=paths(root);texts=[]
    for item in body['input']:
        for part in item.get('content',[]):
            require(part['type'] in ['input_text','output_text'],'Initial request contains media')
            texts.append(part['text'])
    joined='\n'.join(texts)
    require((plan_dir/'prompt.txt').read_text().strip() in joined,'Exact prompt missing')
    require(str(Path(case['public_directory'])/case['public_filename']) in joined,'Path missing')
    for word in ['t1rgb16','_left','_right','contrast1','sf64','sf8','dataset-index']:
        require(word not in joined,'Source identity leaked: '+word)
    names=[t.get('name') for t in body.get('tools',[])]
    require('view_image' in names and any(t in names for t in ['exec_command','shell_command','shell']),
            'Normal CLI inspection tools missing')
    return names


def capture(root,port):
    plan_dir,out=paths(root)
    if not (plan_dir/'path-only-model-catalog.json').exists():prepare_catalog(root)
    case,prompt=case_inputs(root,0);captured=[]
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            new(out/'path-only-cli-captured-request.json',body)
            try:
                names=validate_first_request(root,body,case);effective,removed=effective_request(body)
                from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
                parsed=ResponsesRequest.model_validate(effective)
                require(parsed.seed==0,'Seed unsupported')
                new(out/'path-only-transport-audit.json',{'status':'PASS','model_calls':0,
                    'initial_images':0,'exact_prompt_and_neutral_path':True,'original_tools':names,
                    'removed_unsupported_tools':removed,'vllm_responses_schema':'PASS',
                    'normal_codex_instructions':True})
                new(out/'path-only-effective-request.json',effective);captured.append(True)
            except Exception as exc:
                new(out/'path-only-transport-failure.json',{'error':repr(exc),'model_calls':0});captured.append(False)
            self.send_response(400);self.send_header('Content-Type','application/json');self.end_headers()
            self.wfile.write(b'{"error":{"message":"CPU-only capture, no inference","type":"qualification_only"}}')
    server=ThreadingHTTPServer(('127.0.0.1',port),Handler);server.timeout=1
    with (out/'path-only-qualification.jsonl').open('xb') as log,(out/'path-only-qualification.stderr').open('xb') as err:
        child=subprocess.Popen(cli_command(root,case,port,out/'path-only-no-answer.txt'),
            stdin=subprocess.PIPE,stdout=log,stderr=err,text=True)
        child.stdin.write(prompt);child.stdin.close()
        import time
        deadline=time.monotonic()+60
        while child.poll() is None and not captured and time.monotonic()<deadline:
            server.handle_request()
        rc=child.wait(timeout=60)
    server.server_close()
    new(out/'path-only-qualification.wait.json',{'returncode':rc,'real_wait':True,
        'model_calls':0,'capture_success':captured})
    require(captured==[True],'Path-only qualification failed')
    print(json.dumps({'status':'PATH_ONLY_CLI_TRANSPORT_PASS','model_calls':0}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['capture','server-cpu','serve','run'])
    p.add_argument('--root',required=True);p.add_argument('--port',type=int,default=18573)
    a=p.parse_args()
    if a.command=='capture':capture(a.root,a.port)
    elif a.command=='serve':serve(a.root,a.port)
    elif a.command=='run':run_all(a.root,a.port)
    else:
        args,parent,settings,engine_args=server_arguments(a.root,a.port)
        _,out=paths(a.root)
        new(out/'server-cpu-admission.json',{'status':'PASS','settings':settings,
            'model_calls':0,'tool_parser':args.tool_call_parser,
            'same_checkpoint':settings['model']==parent['engine_settings']['model'],
            'engine_args_class':type(engine_args).__name__})
        print('SERVER_CPU_ARGUMENTS_PASS')


def server_arguments(root, port):
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.engine.arg_utils import AsyncEngineArgs
    plan_dir,out=paths(root)
    parent=read(Path(root)/'plans/reader-route-smoke-20261026/cosmos-fourier1-C-repair1-profile.json')
    settings=copy.deepcopy(parent['engine_settings'])
    settings.update(max_model_len=98304,kv_cache_memory_bytes=32*2**30,
        limit_mm_per_prompt={'image':180,'video':1,'audio':0},
        worker_cls='classic_precision.codex_local_visual_worker.LocalVisualWorker')
    parser=make_arg_parser(FlexibleArgumentParser())
    args=parser.parse_args(['--model',settings['model'],'--host','127.0.0.1','--port',str(port),
        '--served-model-name','cosmos-local','--enable-auto-tool-choice','--tool-call-parser','hermes',
        '--generation-config','vllm','--trust-request-mm-kwargs'])
    for k,v in settings.items():setattr(args,k,v)
    validate_parsed_serve_args(args)
    # This validates options/model metadata on CPU; never instantiate an engine here.
    engine_args=AsyncEngineArgs.from_cli_args(args)
    return args,parent,settings,engine_args


def serve(root,port):
    import uvloop
    from vllm.entrypoints.launchers.api_server.entry import run_server
    args,_,_,_=server_arguments(root,port)
    uvloop.run(run_server(args))


def store_payload(folder,body):
    """Losslessly retain payload images once; never decode/transform video."""
    import base64,hashlib
    blobs=folder.parent/'payload-images';blobs.mkdir(exist_ok=True)
    count=0
    def visit(v):
        nonlocal count
        if isinstance(v,str) and v.startswith('data:image/'):
            count+=1;header,encoded=v.split(',',1);raw=base64.b64decode(encoded)
            sha=hashlib.sha256(raw).hexdigest();p=blobs/sha
            if not p.exists():
                with p.open('xb') as f:f.write(raw)
            return {'data_url_header':header,'evidence_blob_sha256':sha,'bytes':len(raw)}
        if isinstance(v,list):return [visit(x) for x in v]
        if isinstance(v,dict):return {k:visit(x) for k,x in v.items()}
        return v
    saved=visit(body)
    return saved,count


def session(root,index,proxy_port,backend_port):
    import http.client,os,threading,time,signal,traceback
    from .embodied_admission import append
    plan_dir,out=paths(root);case,prompt=case_inputs(root,index)
    folder=out/('case-'+str(index+1));folder.mkdir(exist_ok=False)
    prior_transport_rejection=out/'startup-mm-policy-v2'/folder.name/'result.json'
    new(folder/'claim.json',{'case':index+1,'public_id':case['public_id'],
        'attempt':2 if prior_transport_rejection.exists() else 1,
        'prior_transport_rejection':str(prior_transport_rejection) if prior_transport_rejection.exists() else None,
        'plan_sha256':file_sha256(plan_dir/'path-only-plan.json'),'coordinator_predecode':False})
    events=[];requests=[];lock=threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            connection=None
            try:
                with lock:
                    n=len(requests)+1
                    require(n<=200,'Finite 200-request session limit reached')
                    body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    if n==1:validate_first_request(root,body,case)
                    effective,removed=effective_request(body)
                    saved,num_images=store_payload(folder,body)
                    new(folder/f'request-{n:03d}.json',saved)
                    # Store effective controls separately; the original exact text/tool outputs remain above.
                    record={'request':n,'images_in_context':num_images,'removed_tools':removed,
                        'compatibility':'developer/system text preserved in Responses.instructions',
                        'controls':{k:effective[k] for k in ['temperature','top_p','seed','max_output_tokens',
                        'mm_processor_kwargs','chat_template_kwargs','parallel_tool_calls']}}
                    requests.append(record);append(folder/'requests.jsonl',record)
                connection=http.client.HTTPConnection('127.0.0.1',backend_port,timeout=600)
                start=time.monotonic()
                connection.request('POST','/v1/responses',body=json.dumps(effective),
                    headers={'Content-Type':'application/json','Accept':'text/event-stream'})
                response=connection.getresponse()
                self.send_response(response.status)
                self.send_header('Content-Type',response.getheader('Content-Type','application/json'))
                self.send_header('Connection','close');self.end_headers()
                with (folder/f'response-{n:03d}.sse').open('xb') as log:
                    while True:
                        line=response.readline()
                        if not line:break
                        log.write(line);log.flush();self.wfile.write(line);self.wfile.flush()
                        if line.strip()==b'data: [DONE]':break
                    log.flush();os.fsync(log.fileno())
                append(folder/'request-results.jsonl',{'request':n,'http_status':response.status,
                    'engine_http_seconds':time.monotonic()-start})
            except Exception as exc:
                append(folder/'proxy-errors.jsonl',{'error':repr(exc),'traceback':traceback.format_exc()})
                try:self.send_error(500,str(exc))
                except (BrokenPipeError,ConnectionResetError):pass
            finally:
                if connection:connection.close()
                self.close_connection=True
    server=ThreadingHTTPServer(('127.0.0.1',proxy_port),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    try:
        with (folder/'cli.jsonl').open('xb') as log,(folder/'cli.stderr').open('xb') as err:
            args=cli_command(root,case,proxy_port,folder/'answer.txt')
            new(folder/'launch.json',{'command':args,'prompt':prompt,'initial_images':0,
                'session_limit_seconds':3600,'maximum_requests':200})
            child=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=log,stderr=err,text=True)
            child.stdin.write(prompt);child.stdin.close();timed_out=False
            try:rc=child.wait(timeout=3600)
            except subprocess.TimeoutExpired:
                timed_out=True;child.send_signal(signal.SIGINT);rc=child.wait(timeout=60)
        answer=(folder/'answer.txt').read_text().strip() if (folder/'answer.txt').exists() else None
        result={'case':index+1,'returncode':rc,'real_wait':True,'pid':child.pid,
            'owned_cli_exited':child.poll() is not None,'timed_out':timed_out,
            'answer':answer,'requests':len(requests),'maximum_images_in_context':
            max((r['images_in_context'] for r in requests),default=0),
            'protocol_status':'PENDING_TOOL_TRACE_REVIEW','automatic_retry':False}
        new(folder/'result.json',result);print(json.dumps(result),flush=True)
        return result
    finally:server.shutdown();thread.join(timeout=10);server.server_close()


def run_all(root,port):
    import fcntl,os,signal,time,urllib.request,psutil
    plan_dir,out=paths(root)
    require(read(out/'path-only-transport-audit.json')['status']=='PASS','CPU transport not ready')
    args,parent,settings,_=server_arguments(root,port)
    # User requested preceding finite model work first.
    rx_root=Path(root)/'runs/reproducibility/embodied-points-20261027'
    rx2=rx_root/'rx-cold2.wait.json';rx1=rx_root/'rx-cold1.wait.json'
    rx=read(rx2) if rx2.exists() else read(rx1) if rx1.exists() else {}
    require(rx and (rx2.exists() or rx['returncode']!=0)
        and rx['real_wait'] and rx['owned_hf_process_exited']
        and rx['original_same_inode_lease_free'], 'Rx finite work has not reached an audited terminal state')
    new(out/'server-launch.json',{'coordinator_pid':os.getpid(),'settings':settings,
        'parent_profile':parent['worker'],'automatic_retry':False,'preceding_rx_terminal':rx,
        'implementation_sha256':file_sha256(__file__),
        'worker_implementation_sha256':file_sha256(Path(__file__).with_name('codex_local_visual_worker.py'))})
    env=os.environ.copy()
    for k in parent['cleared_environment']:env.pop(k,None)
    env.update(parent['environment']);env['CVPR_CLI_AUDIT_ROOT']=str(out)
    lease=parent['physical_lock'];owned=set();child=None;survivors=[]
    with Path(lease['path']).open('r+b') as handle:
        st=os.fstat(handle.fileno())
        require((st.st_dev,st.st_ino)==(lease['st_dev'],lease['st_ino']),'Original lease changed')
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            usage=subprocess.run(['nvidia-smi','--id='+parent['gpu_uuid'],
                '--query-gpu=memory.free','--format=csv,noheader,nounits'],
                capture_output=True,text=True,check=True)
            require(int(usage.stdout.strip())>80*1024,'GPU headroom below80GiB')
            with (out/'server.log').open('xb') as log:
                child=subprocess.Popen([sys.executable,'-B','-m','classic_precision.codex_local_visual',
                    'serve','--root',str(root),'--port',str(port)],env=env,stdout=log,stderr=subprocess.STDOUT)
                deadline=time.monotonic()+600;ready=False
                while time.monotonic()<deadline and child.poll() is None:
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=2) as r:ready=r.status==200
                        if ready:break
                    except Exception:pass
                    time.sleep(2)
                require(ready,'Local server failed to become ready; no case submitted')
                require((out/'engine-receipt.json').exists(),'Actual kernel receipt missing')
                for proc in psutil.Process(child.pid).children(recursive=True):owned.add((proc.pid,proc.create_time()))
                results=[]
                for index in range(4):
                    result=session(root,index,18573,port);results.append(result)
                    # Capability failure is retained. Do not blindly continue scoring invisible videos.
                    if result['returncode']!=0 or result['maximum_images_in_context']==0:break
                new(out/'sessions-result.json',{'results':results,'status':'PENDING_TOOL_TRACE_REVIEW'})
        finally:
            if child:
                try:
                    for proc in psutil.Process(child.pid).children(recursive=True):
                        owned.add((proc.pid,proc.create_time()))
                except psutil.NoSuchProcess:pass
                if child.poll() is None:child.send_signal(signal.SIGINT)
                rc=child.wait(timeout=120)
                survivors=[]
                for pid,created in owned:
                    try:
                        p=psutil.Process(pid)
                        if p.create_time()==created and p.status()!=psutil.STATUS_ZOMBIE:survivors.append(pid)
                    except psutil.NoSuchProcess:pass
                new(out/'server.wait.json',{'returncode':rc,'real_wait':True,'owned_survivors':survivors,
                    'graceful_signal':'SIGINT after finite sessions','original_inode':[st.st_dev,st.st_ino]})
            fcntl.flock(handle,fcntl.LOCK_UN)
            with Path(lease['path']).open('r+b') as f:
                st=os.fstat(f.fileno())
                require((st.st_dev,st.st_ino)==(lease['st_dev'],lease['st_ino']),'Lease replaced')
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                new(out/'lease-closure.json',{'original_same_inode_free':True,'owned_survivors':survivors})


if __name__=='__main__':main()
