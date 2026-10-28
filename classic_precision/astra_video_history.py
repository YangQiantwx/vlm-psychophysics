"""Read-only, visible conversation export; no hidden reasoning or model calls."""
from __future__ import annotations
import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path
from . import astra_video_eval as base
from .astra_video_results import image_paths


def export_one(case,folder,audit,preview_dir,preview=False):
    pending={};operations=[];raw_sha=hashlib.sha256();commands=[]
    with (folder/'rollout.jsonl').open('rb') as stream:
        for raw in stream:
            raw_sha.update(raw);event=json.loads(raw);p=event.get('payload',{});typ=p.get('type')
            if event.get('type')!='response_item':continue
            if typ=='message' and p.get('role') in ['user','assistant'] and p.get('channel')!='analysis':
                text='\n'.join(x.get('text','') for x in p.get('content',[]) if x.get('type') in ['input_text','output_text'])
                if text:operations.append({'kind':p['role'],'text':text})
            elif typ in ['custom_tool_call','function_call']:
                value=p.get('input',p.get('arguments',''));code=value if isinstance(value,str) else json.dumps(value)
                try:args=json.loads(code)
                except ValueError:args={}
                pending[p['call_id']]=image_paths(code,args)
                operations.append({'kind':'tool_call','name':p.get('name'),'call_id':p['call_id'],'text':code})
            elif typ in ['custom_tool_call_output','function_call_output']:
                output=p.get('output',[]);call=p.get('call_id');paths=pending.get(call) or []
                if not isinstance(output,list):operations.append({'kind':'tool_output','call_id':call,'text':str(output)});continue
                images=[x for x in output if x.get('type')=='input_image'];previews=[]
                if preview:
                    for x in images:
                        value=x.get('image_url','')
                        if not value.startswith('data:image/png;'):continue
                        data=base64.b64decode(value.split(',',1)[1]);digest=hashlib.sha256(data).hexdigest()
                        destination=preview_dir/(digest+'.png')
                        if not destination.exists():destination.write_bytes(data)
                        previews.append('history-images/'+destination.name)
                operations.append({'kind':'tool_output','call_id':call,
                    'text':'\n'.join(x.get('text','') for x in output if x.get('text')),
                    'images':len(images),'paths':paths[:len(images)],'previews':previews})
    base.require(raw_sha.hexdigest()==audit['identity']['rollout_sha256'],'Archived conversation changed')
    for event in base.json_lines(folder/'events.jsonl'):
        item=event.get('item',{})
        if event.get('type')=='item.completed' and item.get('type')=='command_execution':
            commands.append({k:item.get(k) for k in ['id','command','exit_code','aggregated_output']})
    original=base.read(folder/'result.json')
    return {'case_id':case['case_id'],'sample_id':case['sample_id'],'task':case['task'],'cell':case['cell'],
        'attempt':case.get('attempt',1),'prior_case_id':case.get('prior_case_id'),
        'scope':'auxiliary' if case['task']=='T3' and case['cell'][1] in [1.75,2.25] else 'primary',
        'session_id':audit['session_id'],'resume_command':audit['resume_command'],
        'status':audit['protocol_status'],'visual_only_trace_status':audit['visual_only_trace_status'],
        'frames':audit['returned_images'],'answer':audit['answer'],'operations':operations,'commands':commands,
        'raw_rollout_path':str(folder/'rollout.jsonl'),'original_persistent_path':original['original_rollout_path'],
        'raw_sha256':raw_sha.hexdigest(),'persistent_original_exists':Path(original['original_rollout_path']).is_file(),
        'preview_images_included':preview,'hidden_reasoning_exported':False}


HTML=r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Astra 会话历史与工具证据</title>
<style>body{margin:0;background:#f5f6f8;color:#18202c;font:15px system-ui}header{padding:20px 28px;background:white;border-bottom:1px solid #ddd}h1{font-size:23px;margin:0 0 8px}p{line-height:1.6}input,select{font:inherit;padding:8px;border:1px solid #ccd0d5;border-radius:6px;margin-right:8px}main{display:grid;grid-template-columns:360px 1fr;height:calc(100vh - 165px)}nav{overflow:auto;padding:12px;border-right:1px solid #ddd}nav button{display:block;text-align:left;width:100%;padding:10px;margin:4px 0;border:0;border-radius:7px;background:white;cursor:pointer}nav button.selected{outline:2px solid #3265a8}article{overflow:auto;padding:22px 28px}details,.card{padding:12px 16px;background:white;border-radius:8px;margin:12px 0;border:1px solid #e3e5e8}summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere;line-height:1.5;font:13px ui-monospace,monospace}.meta{color:#586271;font-size:13px}.images{display:flex;flex-wrap:wrap;gap:8px}.images img{width:192px;height:128px;object-fit:contain;background:#ddd}code{overflow-wrap:anywhere}@media(max-width:800px){main{display:block;height:auto}nav{max-height:220px}article{overflow:visible}}</style>
<header><h1>Astra：逐条可见对话与工具操作</h1><p id="stats"></p><select id="task"><option value="">全部任务</option><option>T1</option><option>T2</option><option>T3</option></select><input id="query" placeholder="搜索 config / case / session"><span class="meta">只读导出 · 不含隐藏思维链 · 原始记录保留</span></header>
<main><nav id="list"></nav><article id="detail"></article></main>
<script id="data" type="application/json">DATA_PLACEHOLDER</script><script>
const data=JSON.parse(document.getElementById('data').textContent);let selected=null;
const el=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
document.getElementById('stats').textContent=`${data.length} 条历史记录；按真实调用展示命令、看图批次和最终回答。只有代表样例附本地原PNG，其余完整图像保留于Fourier原rollout。`;
function show(r){selected=r.case_id;const d=document.getElementById('detail');d.replaceChildren();d.append(el('h2',r.cell.join(' · ')+` / attempt ${r.attempt}`));d.append(el('p',`最终回答：${r.answer??'无有效回答'} ｜ 返回图像 ${r.frames} ｜ ${r.visual_only_trace_status}`));d.append(el('p',`Session: ${r.session_id}`,'meta'));d.append(el('pre',r.resume_command));
const provenance=el('details');provenance.append(el('summary','原始记录、SHA与状态'));provenance.append(el('pre',JSON.stringify({case_id:r.case_id,prior_case_id:r.prior_case_id,status:r.status,scope:r.scope,raw_rollout_path:r.raw_rollout_path,original_persistent_path:r.original_persistent_path,sha256:r.raw_sha256},null,2)));d.append(provenance);
const commands=el('details');commands.append(el('summary',`实际执行命令：${r.commands.length} 条`));for(const c of r.commands){commands.append(el('pre',c.command));commands.append(el('pre',`exit=${c.exit_code}\n${c.aggregated_output??''}`));}d.append(commands);
for(const op of r.operations){if(op.kind==='user'||op.kind==='assistant'){const card=el('div',undefined,'card');card.append(el('strong',op.kind==='user'?'输入':'助手可见消息'));card.append(el('pre',op.text));d.append(card);continue;}
 const item=el('details');item.append(el('summary',op.images?`Viewed ${op.images} images`:(op.kind==='tool_call'?`工具调用：${op.name}`:'工具返回')));if(op.text)item.append(el('pre',op.text));if(op.paths?.length)item.append(el('pre',op.paths.join('\n')));if(op.previews?.length){const imgs=el('div',undefined,'images');for(const src of op.previews){const im=el('img');im.src=src;im.loading='lazy';im.alt='原始工具返回PNG';imgs.append(im);}item.append(imgs);}d.append(item);}
render(false);}
function render(choose=true){const q=document.getElementById('query').value.toLowerCase(),task=document.getElementById('task').value;const rows=data.filter(r=>(!task||r.task===task)&&JSON.stringify([r.cell,r.case_id,r.session_id,r.sample_id]).toLowerCase().includes(q));const list=document.getElementById('list');list.replaceChildren();for(const r of rows){const b=el('button',r.cell.join(' · ')+` | #${r.attempt} | ${r.frames}帧 | ${r.answer??'失败'}`);if(r.case_id===selected)b.className='selected';b.onclick=()=>show(r);list.append(b);}if(choose&&rows.length)show(rows.find(r=>r.preview_images_included)||rows[0]);}
document.getElementById('query').oninput=()=>render();document.getElementById('task').onchange=()=>render();render();
</script></html>'''


def run(root):
    _,out,_,samples=base.load(root);preview_dir=out/'history-images';preview_dir.mkdir(exist_ok=True)
    proof=base.read(out/'visual-proof-summary.json');previews={x['case_id'] for x in proof['examples']};records=[]
    candidates=[(s,out) for s in samples]
    repair_plans=Path(root)/'plans'/base.NAME
    for directory in sorted(repair_plans.iterdir(),key=lambda p:(p.name!='primary-repair1',p.name)):
        if directory.name!='primary-repair1' and not directory.name.startswith('capacity-retry'):continue
        manifest=directory/'samples.json'
        if manifest.is_file():candidates.extend((s,out/'repairs'/directory.name) for s in base.read(manifest))
    for case,parent in candidates:
        folder=parent/'cases'/case['case_id'];ap=parent/'trace-audits'/(case['case_id']+'.json')
        if not ap.exists():continue
        records.append(export_one(case,folder,base.read(ap),preview_dir,case['case_id'] in previews))
    with (out/'conversation-history.jsonl').open('w') as f:
        for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    payload=json.dumps(records,ensure_ascii=False,separators=(',',':')).replace('<',chr(92)+'u003c')
    (out/'conversation-history.html').write_text(HTML.replace('DATA_PLACEHOLDER',payload))
    result={'status':'VISIBLE_HISTORY_RETRIEVED','records':len(records),'original_attempts':sum(x['attempt']==1 for x in records),
        'repair_attempts':sum(x['attempt']>=2 for x in records),'all_original_persistent_files_present':all(x['persistent_original_exists'] for x in records),
        'all_archive_shas_verified':True,'hidden_reasoning_exported':False,'scientific_calls_added':0,
        'exact_png_preview_files':len(list(preview_dir.glob('*.png'))),'jsonl_sha256':base.sha(out/'conversation-history.jsonl'),
        'html_sha256':base.sha(out/'conversation-history.html')}
    base.write(out/'conversation-history-summary.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();print(json.dumps(run(a.root),ensure_ascii=False))
