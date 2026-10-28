"""Read-only trace audit for path-only Astra sessions; never invokes a model."""
from __future__ import annotations
import argparse
import ast
import base64
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import struct
from .astra_video_eval import load,read,write,sha,json_lines,digest


def literal(value):
    try:return json.loads(value) if value.startswith('"') else ast.literal_eval(value)
    except (ValueError,SyntaxError):return None


def image_paths(code, args):
    """Interpret only literal paths and simple increasing loops, never execute JS."""
    if isinstance(args,dict) and 'path' in args:return [args['path']]
    if 'view_image' not in code:return []
    tokens=re.findall(r'''["']?path["']?\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`[^`]*`)''',code)
    loops=re.findall(r'for\s*\(\s*(?:let|const|var)\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*(<=|<)\s*(\d+)\s*;\s*(?:\1\+\+|\+\+\1|\1\s*\+=\s*1)\s*\)',code)
    if 'Promise.all' in code or len(loops)>1:return None
    if loops:
        var,start,op,end=loops[0];values=range(int(start),int(end)+(op=='<='))
        if len(tokens)!=1 or not tokens[0].startswith('`'):return None
        template=tokens[0][1:-1];result=[]
        for i in values:
            v=re.sub(r'\$\{String\('+re.escape(var)+r'''\)\.padStart\((\d+),\s*["']0["']\)\}''',
                     lambda m:str(i).zfill(int(m[1])),template)
            v=v.replace('${'+var+'}',str(i))
            if '${' in v:return None
            result.append(v)
        return result
    result=[]
    for t in tokens:
        if t.startswith('`'):
            if '${' in t:return None
            result.append(t[1:-1])
        else:
            v=literal(t)
            if v is None:return None
            result.append(v)
    return result or None


def audit_case(root,case):
    _,out,_,_=load(root);folder=out/'cases'/case['case_id']
    result=read(folder/'result.json');wait=read(folder/'wait.json')
    rollout=folder/'rollout.jsonl'
    key={'result_sha256':sha(folder/'result.json'),'rollout_sha256':sha(rollout) if rollout.exists() else None,
         'auditor_sha256':sha(__file__)}
    cache=out/'trace-audits'/f'{case["case_id"]}.json'
    if cache.exists():
        old=read(cache)
        if old.get('identity')==key:return old
    pending={};viewed=[];images=0;unresolved=0;image_shapes=Counter();bad_hash=[];unknown_tools=[]
    tool_names=[];runtime_errors=[];contexts=[]
    for row in json_lines(rollout):
        p=row.get('payload',{});typ=p.get('type')
        if row.get('type')=='turn_context':contexts.append(p)
        if row.get('type')!='response_item':continue
        if typ in ['custom_tool_call','function_call']:
            raw=p.get('input',p.get('arguments',''));args={}
            if isinstance(raw,str):
                try:args=json.loads(raw)
                except ValueError:pass
            name=p.get('name','');code=raw if isinstance(raw,str) else json.dumps(raw)
            pending[p['call_id']]=image_paths(code,args) if name.split('.')[-1] in ['exec','view_image'] else []
            tool_names.extend(re.findall(r'tools\.([\w]+)\s*\(',code))
            if name.split('.')[-1] not in ['exec','view_image','exec_command','write_stdin','wait']:
                unknown_tools.append(name)
        elif typ in ['custom_tool_call_output','function_call_output']:
            output=p.get('output',[])
            if not isinstance(output,list):
                if 'bwrap:' in str(output):runtime_errors.append(str(output)[:300])
                continue
            actual=[x for x in output if x.get('type')=='input_image']
            images+=len(actual)
            pp=pending.get(p.get('call_id'))
            if actual and (pp is None or len(pp)!=len(actual)):
                unresolved+=len(actual);pp=[]
            for j,item in enumerate(actual):
                value=item.get('image_url','');data=None
                if isinstance(value,str) and value.startswith('data:image/'):
                    data=base64.b64decode(value.split(',',1)[1])
                    if data.startswith(b'\x89PNG\r\n\x1a\n'):
                        size=struct.unpack('>II',data[16:24]);image_shapes[str(size)]+=1
                    else:image_shapes['non-PNG-needs-review']+=1
                if pp:
                    path=Path(pp[j]);path=path if path.is_absolute() else Path(case['public_directory'])/path
                    viewed.append(str(path))
                    if data is None or not path.is_file() or hashlib.sha256(data).hexdigest()!=sha(path):
                        bad_hash.append(str(path))
            for item in output:
                text=item.get('text','')
                if 'bwrap:' in text or 'Failed RTM_NEWADDR' in text:runtime_errors.append(text[:300])
    events=json_lines(folder/'events.jsonl')
    commands=[x['item'] for x in events if x.get('type')=='item.completed'
              and x.get('item',{}).get('type')=='command_execution']
    decode=[x for x in commands if re.search(r'\bffmpeg\b|VideoCapture|av\.open',x.get('command',''))]
    transforms=[x['command'] for x in commands if re.search(r'optical|calcOptical|phaseCorrelate|fft|absdiff|hstack|vstack|tile=|crop=|scale=|select=|fps=|framestep|ffprobe|showinfo| -ss | -t | -vf | -filter',x.get('command',''),re.I)]
    local_dir=Path(case['public_directory']);outside=[];indices=[]
    for value in viewed:
        path=Path(value)
        try:path.relative_to(local_dir)
        except ValueError:outside.append(value)
        match=re.search(r'(\d+)$',path.stem)
        indices.append(int(match[1]) if match else None)
    decoded_files=list(local_dir.rglob('*.png')) if local_dir.exists() else []
    correct_order=(indices==list(range(1,181)) or indices==list(range(180)))
    actual_model=all(p.get('model')=='gpt-6-astra' and p.get('effort')=='medium' for p in contexts) and bool(contexts)
    allowed={'exec_command','view_image','write_stdin','clock__curr_time'}
    unusual=sorted(set(tool_names)-allowed)
    if runtime_errors:status='ENVIRONMENT_FAILED_NO_VALID_VISUAL_EVALUATION'
    elif result['returncode']!=0:status='EXECUTION_FAILED'
    elif images==0:status='NO_VISUAL_INPUT'
    elif images<180:status='EARLY_ANSWER_PARTIAL_SEQUENCE'
    elif images>180:status='REPEATED_OR_EXTRA_IMAGES'
    elif unresolved or not correct_order:status='FRAME_ORDER_REQUIRES_REVIEW'
    elif transforms or outside or unusual or unknown_tools:status='TOOL_PROTOCOL_REQUIRES_REVIEW'
    elif bad_hash or image_shapes!={'(768, 512)':180}:status='IMAGE_IDENTITY_OR_GEOMETRY_UNCONFIRMED'
    elif len(decode)!=1 or decode[0].get('exit_code')!=0 or len(decoded_files)!=180:status='DECODE_REQUIRES_REVIEW'
    elif not actual_model or not wait.get('real_wait') or not wait.get('owned_descendants_exited'):status='EXECUTION_IDENTITY_UNCONFIRMED'
    else:status='FULL180_TRACE_PASS'
    value={'schema':'astra-trace-audit-v1','identity':key,'case_id':case['case_id'],'sample_id':case['sample_id'],
           'task':case['task'],'cell':case['cell'],'truth':case['truth'],'answer':result['answer'],
           'correct':result['answer']==case['truth'],'strict_format':result['strict_format'],
           'session_id':result['session_id'],'resume_command':result['resume_command'],
           'protocol_status':status,'returned_images':images,'resolved_frame_paths':len(viewed),
           'unresolved_image_paths':unresolved,'frame_indices_in_order':indices,'image_dimensions':dict(image_shapes),
           'one_successful_decode':len(decode)==1 and decode[0].get('exit_code')==0,'decoded_png_files':len(decoded_files),
           'image_identity_mismatches':bad_hash,'outside_case_paths':outside,'transform_flags':transforms,
           'unusual_tools':unusual+unknown_tools,'runtime_errors':runtime_errors,'model_effort_verified':actual_model,
           'wall_seconds':wait['wall_seconds'],'usage':result['usage'],
           'limits':'Tool-return evidence establishes ordered full-size images made available, not internal attention or hidden server resizing.'}
    write(cache,value)
    return value


def run(root):
    _,out,plan,samples=load(root);rows=[]
    for s in samples:
        if (out/'cases'/s['case_id']/'result.json').exists():rows.append(audit_case(root,s))
    fields=['case_id','sample_id','task','cell_json','truth','answer','correct','strict_format',
            'session_id','resume_command','protocol_status','returned_images','wall_seconds']
    with (out/'trace-answers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:(json.dumps(r['cell']) if k=='cell_json' else r[k]) for k in fields})
    grouped=defaultdict(list)
    for r in rows:grouped[json.dumps(r['cell'])].append(r)
    cells=[]
    for k,rr in sorted(grouped.items()):
        full=[x for x in rr if x['protocol_status']=='FULL180_TRACE_PASS']
        cells.append({'cell_json':k,'task':rr[0]['task'],'expected':4,'closed':len(rr),
                      'raw_correct':sum(x['correct'] for x in rr),'full180_n':len(full),
                      'full180_correct':sum(x['correct'] for x in full)})
    with (out/'trace-cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cell_json','task','expected','closed','raw_correct','full180_n','full180_correct'])
        w.writeheader();w.writerows(cells)
    tasks={}
    for t in ['T1','T2','T3']:
        rr=[r for r in rows if r['task']==t];full=[r for r in rr if r['protocol_status']=='FULL180_TRACE_PASS']
        tasks[t]={'planned':plan['counts'][t],'closed':len(rr),'raw_correct':sum(r['correct'] for r in rr),
                  'full180_n':len(full),'full180_correct':sum(r['correct'] for r in full),
                  'status_counts':dict(Counter(r['protocol_status'] for r in rr))}
    result={'status':'COMPLETE_TRACE_AUDIT' if len(rows)==924 else 'PARTIAL_TRACE_AUDIT',
            'closed':len(rows),'planned':924,'by_task':tasks,'status_counts':dict(Counter(r['protocol_status'] for r in rows)),
            'full180_total':sum(r['protocol_status']=='FULL180_TRACE_PASS' for r in rows),
            'limits':['Raw accuracy includes environment failures and partial inspection; never present it as full180 accuracy.',
                      'Full180 subset is selected by observed agent behavior, not randomized; report selection bias.',
                      'Four videos per cell is a finite pilot, not a well-estimated psychometric threshold.']}
    write(out/'trace-summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(run(a.root),ensure_ascii=False,indent=2))
