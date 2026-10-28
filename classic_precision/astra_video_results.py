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
import shlex
from .astra_video_eval import load,read,write,sha,json_lines,digest


def approved_shell(command, case):
    """Fail closed: recognize decoding and frame-file bookkeeping only."""
    try:
        outer=shlex.split(command)
        if len(outer)==3 and Path(outer[0]).name in ('bash','sh') and outer[1] in ('-lc','-c'):
            command=outer[2]
        lexer=shlex.shlex(command,posix=True,punctuation_chars=';&|<>\n')
        lexer.whitespace=' \t\r';lexer.whitespace_split=True
        tokens=list(lexer)
        chunks=[];part=[]
        for token in tokens:
            if token in ('&&',';','|','\n'):
                if part:chunks.append(part);part=[]
            elif token in ('>','>>','<','<<','&','||'):return False,'shell redirection/background/conditional'
            else:part.append(token)
        if part:chunks.append(part)
        base=Path(case['public_directory']).resolve();cwd=base
        source=Path(case['video_path']).resolve()
        def path(value):
            if any(x in value for x in ('$','`','*','?')):raise ValueError('dynamic path')
            p=Path(value);p=(p if p.is_absolute() else cwd/p).resolve()
            p.relative_to(base);return p
        for args in chunks:
            name=Path(args[0]).name;a=args[1:]
            if name=='cd' and len(a)==1:cwd=path(a[0]);continue
            if name=='mkdir' and a and a[0]=='-p' and len(a)>1:
                for value in a[1:]:path(value)
                continue
            if name=='ffmpeg':
                i=0;inputs=[];outputs=[]
                values={'-v':('error','warning'),'-loglevel':('error','warning'),
                        '-vsync':('0',),'-fps_mode':('passthrough',)}
                while i<len(a):
                    value=a[i]
                    if value in ('-hide_banner','-y','-nostdin','-an','-sn','-dn'):i+=1;continue
                    if value=='-i':inputs.append(path(a[i+1]));i+=2;continue
                    if value in values:
                        if a[i+1] not in values[value]:return False,'decoder option value'
                        i+=2;continue
                    if value.startswith('-'):return False,'unreviewed decoder option '+value
                    outputs.append(path(value));i+=1
                if inputs!=[source] or len(outputs)!=1 or outputs[0].suffix!='.png' or '%' not in outputs[0].name:
                    return False,'decoder input/output differs'
                continue
            if name in ('ls','find'):
                allowed={'-1','-type','f','-maxdepth','1'}
                for value in a:
                    if value in allowed:continue
                    if value.startswith('-'):return False,'unreviewed listing option'
                    path(value)
                continue
            if name=='wc' and a==['-l']:continue
            if name=='sort' and not a:continue
            if name=='pwd' and not a:continue
            return False,'unreviewed command '+name
        return bool(chunks),'approved decode/frame bookkeeping'
    except (ValueError,IndexError):return False,'unparsed shell syntax/path'


def exec_review(code):
    # Do not execute agent JavaScript. Unknown functions/indirection require
    # inspection even when a keyword blacklist would have missed them.
    stripped=re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|`[^`]*`','""',code)
    calls=re.findall(r'([\w.]+)\s*\(',stripped)
    allowed={'for','if','catch','text','image','String','JSON.stringify',
             'tools.exec_command','tools.view_image','tools.write_stdin','tools.clock__curr_time'}
    issues=sorted(set(calls)-allowed)
    if re.search(r'\btools\s*\[|\beval\b|\bFunction\b|\bfetch\b|\batob\b|\bBuffer\b',stripped):
        issues.append('dynamic execution/data inspection')
    if re.search(r'\.image_url\s*(?:\.|\[)|\b(?:pixel|rgba|base64|charCodeAt|DataView|Uint8Array|ArrayBuffer)\b',stripped):
        issues.append('image data inspection outside direct image return')
    if 'chars' in code and not re.search(r'chars\s*:\s*(?:""|\'\')',code):
        issues.append('nonempty/unresolved stdin write')
    return issues


def literal(value):
    try:return json.loads(value) if value.startswith('"') else ast.literal_eval(value)
    except (ValueError,SyntaxError):return None


def image_paths(code, args):
    """Interpret only literal paths and simple increasing loops, never execute JS."""
    if isinstance(args,dict) and 'path' in args:return [args['path']]
    if 'view_image' not in code:return []
    tokens=re.findall(r'''["']?path["']?\s*:\s*("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`[^`]*`)''',code)
    loops=re.findall(r'for\s*\(\s*(?:let|const|var)\s+(\w+)\s*=\s*(\d+)\s*;\s*\1\s*(<=|<)\s*(\d+)\s*;\s*(?:\1\+\+|\+\+\1|\1\s*\+=\s*1)\s*\)',code)
    array_loops=re.findall(r'for\s*\(\s*(?:let|const|var)\s+(\w+)\s+of\s+(\[[^\[\]\n]*\])\s*\)',code)
    if 'Promise.all' in code or len(loops)+len(array_loops)>1:return None
    if loops or array_loops:
        if loops:
            var,start,op,end=loops[0];values=range(int(start),int(end)+(op=='<='))
        else:
            var,raw=array_loops[0];values=literal(raw)
            if not isinstance(values,list) or any(not(type(i) is int or
                    isinstance(i,str) and re.fullmatch(r'\d+',i)) for i in values):return None
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
    pending={};viewed=[];images=0;unresolved=0;image_shapes=Counter();bad_hash=[];unknown_tools=[];eof_probes=0
    tool_names=[];runtime_errors=[];contexts=[];exec_issues=[]
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
            if name.split('.')[-1]=='exec':
                issues=exec_review(code)
                if issues:exec_issues.append({'call_id':p['call_id'],'issues':issues})
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
            if actual and pp and len(pp)>len(actual):
                def exists(value):
                    pth=Path(value)
                    return (pth if pth.is_absolute() else Path(case['public_directory'])/pth).is_file()
                # A recorded sequential loop can stop at the first missing EOF
                # file, either caught by the agent or returned as a tool error.
                # Verify every returned prefix hash below; never infer
                # omitted frames merely from a requested loop range.
                if all(exists(v) for v in pp[:len(actual)]) and not any(exists(v) for v in pp[len(actual):]):
                    eof_probes+=1;pp=pp[:len(actual)]
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
    command_starts=[x['item'] for x in events if x.get('type')=='item.started'
                    and x.get('item',{}).get('type')=='command_execution']
    ended_ids={x.get('id') for x in commands}
    unclosed_commands=[x.get('id') for x in command_starts if x.get('id') not in ended_ids]
    label_output=[]
    for x in commands:
        output=x.get('aggregated_output','')
        if any(v and v in output for v in [case['sample_id'],Path(case['source']).name]) or re.search(
                r'ground.?truth|target_side|reference_side|angle_deg|modulation_depth|sf_cycles|delta_y',output,re.I):
            label_output.append(x.get('id'))
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
    shell_reviews=[{'command':x.get('command',''),'reason':reason}
        for x in commands for ok,reason in [approved_shell(x.get('command',''),case)] if not ok]
    prefix_order=(indices==list(range(1,images+1)) or indices==list(range(images)))
    prompt_path=folder/'prompt.txt'
    from .astra_video_eval import prompt_for
    plan_dir,_,plan,_=load(root)
    prompt_matches=prompt_path.exists() and prompt_path.read_text()==prompt_for(plan_dir,plan,case)
    neutral=bool(re.fullmatch(r'[0-9a-f]{32}',Path(case['video_path']).stem)
                 and re.fullmatch(r'[0-9a-f]{32}',Path(case['public_directory']).name))
    proof_checks={'neutral_id_path':neutral,'exact_frozen_prompt':prompt_matches,
        'actual_astra_medium':actual_model,'one_successful_full_decode':len(decode)==1 and decode[0].get('exit_code')==0 and len(decoded_files)==180,
        'all_recorded_shell_commands_approved':bool(commands) and not shell_reviews,
        'all_started_shell_commands_closed':not unclosed_commands,
        'no_source_labels_in_command_outputs':not label_output,
        'no_unreviewed_exec_or_tools':not(exec_issues or unusual or unknown_tools),
        'only_case_frame_paths':not outside,'returned_images_match_original_png':images>0 and not(bad_hash or unresolved),
        'all_returned_images_768x512':image_shapes=={'(768, 512)':images} and images>0,
        'chronological_without_revisit':prefix_order and images>0 and images<=180,
        'real_wait_owned_exit':bool(wait.get('real_wait') and wait.get('owned_descendants_exited'))}
    visual_proof='PASS_OBSERVED' if all(proof_checks.values()) and not runtime_errors and result['returncode']==0 else 'REQUIRES_REVIEW_OR_INVALID'
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
           'end_of_sequence_missing_file_probes':eof_probes,
           'one_successful_decode':len(decode)==1 and decode[0].get('exit_code')==0,'decoded_png_files':len(decoded_files),
           'image_identity_mismatches':bad_hash,'outside_case_paths':outside,'transform_flags':transforms,
           'unusual_tools':unusual+unknown_tools,'runtime_errors':runtime_errors,'model_effort_verified':actual_model,
           'visual_only_trace_status':visual_proof,'visual_only_checks':proof_checks,
           'shell_commands_requiring_review':shell_reviews,'exec_calls_requiring_review':exec_issues,
           'unclosed_command_items':unclosed_commands,'source_label_output_items':label_output,
           'wall_seconds':wait['wall_seconds'],'usage':result['usage'],
           'limits':'Tool-return evidence establishes ordered full-size images made available, not internal attention or hidden server resizing.'}
    write(cache,value)
    return value


def run(root):
    _,out,plan,samples=load(root);rows=[]
    for s in samples:
        if (out/'cases'/s['case_id']/'result.json').exists():rows.append(audit_case(root,s))
    fields=['case_id','sample_id','task','cell_json','truth','answer','correct','strict_format',
            'visual_only_trace_status',
            'session_id','resume_command','protocol_status','returned_images','wall_seconds']
    with (out/'trace-answers.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:(json.dumps(r['cell']) if k=='cell_json' else r[k]) for k in fields})
    grouped=defaultdict(list)
    for r in rows:grouped[json.dumps(r['cell'])].append(r)
    cells=[]
    for k,rr in sorted(grouped.items()):
        full=[x for x in rr if x['protocol_status']=='FULL180_TRACE_PASS']
        visual=[x for x in rr if x['returned_images']>0 and x['protocol_status']!='EXECUTION_FAILED']
        cells.append({'cell_json':k,'task':rr[0]['task'],'expected':4,'closed':len(rr),
                      'raw_correct':sum(x['correct'] for x in rr),
                      'observed_visual_n':len(visual),'observed_visual_correct':sum(x['correct'] for x in visual),
                      'full180_n':len(full),
                      'full180_correct':sum(x['correct'] for x in full)})
    with (out/'trace-cells.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['cell_json','task','expected','closed','raw_correct',
            'observed_visual_n','observed_visual_correct','full180_n','full180_correct'])
        w.writeheader();w.writerows(cells)
    tasks={}
    for t in ['T1','T2','T3']:
        rr=[r for r in rows if r['task']==t];full=[r for r in rr if r['protocol_status']=='FULL180_TRACE_PASS']
        visual=[r for r in rr if r['returned_images']>0 and r['protocol_status']!='EXECUTION_FAILED']
        clean=[r for r in rr if r['visual_only_trace_status']=='PASS_OBSERVED']
        tasks[t]={'planned':plan['counts'][t],'closed':len(rr),'raw_correct':sum(r['correct'] for r in rr),
                  'observed_visual_n':len(visual),'observed_visual_correct':sum(r['correct'] for r in visual),
                  'visual_only_trace_n':len(clean),'visual_only_trace_correct':sum(r['correct'] for r in clean),
                  'full180_n':len(full),'full180_correct':sum(r['correct'] for r in full),
                  'status_counts':dict(Counter(r['protocol_status'] for r in rr))}
    result={'status':'COMPLETE_TRACE_AUDIT' if len(rows)==924 else 'PARTIAL_TRACE_AUDIT',
            'closed':len(rows),'planned':924,'by_task':tasks,'status_counts':dict(Counter(r['protocol_status'] for r in rows)),
            'full180_total':sum(r['protocol_status']=='FULL180_TRACE_PASS' for r in rows),
            'visual_only_trace_counts':dict(Counter(r['visual_only_trace_status'] for r in rows)),
            'scope':{'primary_planned':916,'supplemental_planned':8,
                'supplemental_angles_deg':[1.75,2.25],
                'primary_closed':sum(not(r['task']=='T3' and r['cell'][1] in [1.75,2.25]) for r in rows),
                'supplemental_closed':sum(r['task']=='T3' and r['cell'][1] in [1.75,2.25] for r in rows)},
            'limits':['Raw accuracy includes environment failures and partial inspection; never present it as full180 accuracy.',
                      'Observed-visual accuracy excludes zero-image and failed executions but includes early answers and unresolved trace reviews; it is not full180 protocol acceptance.',
                      'Full180 subset is selected by observed agent behavior, not randomized; report selection bias.',
                      'Four videos per cell is a finite pilot, not a well-estimated psychometric threshold.']}
    write(out/'trace-summary.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
    print(json.dumps(run(a.root),ensure_ascii=False,indent=2))
