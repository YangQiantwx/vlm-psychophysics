#!/usr/bin/env python3
"""Build a bounded public snapshot; originals stay local, R2 media are not copied.
Run with qy and --root BENCHMARK_WORKSPACE --site-root BENCHMARK_WORKSPACE/site.
"""
import argparse
import csv
import datetime
import hashlib
import html
import json
import mimetypes
import io
from PIL import Image
import re
from pathlib import Path as _FilesystemPath
import sys as _sys
_sys.path.insert(0, str(_FilesystemPath(__file__).resolve().parent))
try:
    from workspace_paths import Path
except ModuleNotFoundError:
    from pathlib import Path

SITE_ROOT = Path(__file__).parent
from urllib.parse import urljoin, urlsplit, unquote

DROP = {'meeting_alignment','meeting','source_locations','local_path','remote_verification',
        'browser_qa','coordinator_qa','coordinator_review','independent_qa','verification',
        'reproduction_sources','natural_frames','classic_frames','keyframes','visual_qa',
        'incremental_rebuild','compatibility_recheck','previous_protocol'}
PRIVATE = re.compile(r'/Users/[^\s"\'<>]+|/home/[^\s"\'<>]+|file://[^\s"\'<>]+')
MEDIA_FIELDS = ('path','source_path','input_path','master_path','raw_path','original_path','preview_path')
HISTORY = ('T1_bark_sf8_LEFT','T2_d0.022_LEFT','T3_a16_LEFT','T4_c0.6_RIGHT','T5_gap280_PRESENT')



PUBLIC_NAV_CSS = """
/* One task navigation across all public pages. */
.task-navigation{display:flex!important;justify-content:space-between;align-items:center;gap:24px;min-height:80px;border-bottom:1px solid #e1e7ea;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
.task-navigation .site-brand{font-size:17px!important;font-weight:650;white-space:nowrap;text-decoration:none!important;color:inherit}
.task-navigation .task-links{display:flex;align-items:center;gap:6px}
.task-navigation .task-links a{padding:10px 12px!important;display:flex;align-items:center;gap:7px;font-size:14px!important;white-space:nowrap;border-radius:6px;text-decoration:none!important}
.task-navigation .task-links a b{font-size:12px;font-weight:650}
.task-navigation .task-links a[aria-current]{background:#202b31!important;color:#fff!important;border:0!important}
.task-navigation .task-links a:after{display:none!important}
.construction-options{margin:20px 0;font-size:14px}.construction-options summary{cursor:pointer;color:#69767e}.construction-options>div{display:flex;gap:20px;flex-wrap:wrap;padding:16px 0}
@media(max-width:1000px){.task-navigation .task-links a span{display:none}.task-navigation .task-links a b{font-size:14px}.task-navigation .task-links a{padding:10px 8px!important}}
@media(max-width:600px){.task-navigation{gap:8px;min-height:70px}.task-navigation .site-brand{font-size:14px!important}.task-navigation .task-links{gap:2px}.task-navigation .task-links a{padding:9px 6px!important}.task-navigation .task-links a b{font-size:13px}}
"""
PUBLIC_NAV_CSS += "\n.library-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 28px}.library-tabs a{padding:9px 16px!important;border:1px solid #dce4e8;border-radius:6px;text-decoration:none;font:14px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}.library-tabs a[aria-current]{background:#202b31;color:#fff}.library-heading{margin:34px 0 22px!important}.library-heading h1{font-size:34px!important;margin:0!important}header:has(+ #display),header:has(+ #status){margin:24px 0!important}#principle,.principle,#pool-summary,.prompt-note:empty,#classic-controls{display:none!important}.library-link{font-size:14px!important;font-weight:600}.task-heading h2{font-size:24px}\n@media(max-width:600px){.task-navigation .site-brand{font-size:0!important}.task-navigation .site-brand:before{content:'Vision';font-size:14px}.task-navigation .task-links{gap:1px}.task-navigation .task-links a{padding:9px 5px!important}.task-navigation .library-link{font-size:12px!important}.library-tabs{gap:6px}.library-tabs a{padding:8px 10px!important;font-size:13px}.library-heading h1{font-size:28px!important}}\n"
PUBLIC_NAV_CSS += "\n.recorded-sample{padding:24px;border:1px solid #dce4e8;border-radius:10px;margin-bottom:36px}.recorded-sample h2{margin:0 0 16px;font-size:23px}.recorded-sample label{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.recorded-sample select{max-width:100%;padding:8px;font-size:14px}.recorded-media{display:flex;align-items:center;gap:20px;margin-top:18px}.recorded-media video{display:block;width:100%;max-width:680px;max-height:440px;background:#101113}.recorded-media figure{margin:0}.sample-note,#recorded-status{font-size:13px;color:#64727b;max-width:800px}.explorer-heading{font-size:24px;margin:32px 0 22px}@media(max-width:600px){.recorded-sample{padding:16px}.recorded-media{flex-direction:column;align-items:flex-start}.recorded-media img{max-height:100px}}\n"

def public_navigation(text, prefix):
    """Shared task/results navigation, with one Library entry and five library tabs."""
    if not prefix.startswith('/samples/'):
        return text
    match=re.search(r'/samples/t([1-5])/',prefix)
    current='T'+match[1] if match else None
    if '/convolution/' in prefix:return '<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/samples/t1/"><title>Library · T1</title><a href="/samples/t1/">Current T1 library</a>'
    tasks=(('T1','Motion'),('T2','Brightness'),('T3','Orientation'),('T4','Global motion'),('T5','Detection'))
    links=''.join('<a href="/experiments/?task='+tid+'"><b>'+tid+'</b><span>'+label+'</span></a>' for tid,label in tasks)
    nav='<nav class="project-nav task-navigation" aria-label="Website"><a class="site-brand" href="/experiments/?task=T1">Visual perception</a><div class="task-links">'+links+'<a class="library-link" href="/library/" aria-current="page">Library</a></div></nav>'
    text=re.sub(r'<nav\b[^>]*>.*?</nav>',lambda _:nav,text,count=1,flags=re.S)
    text=re.sub(r'<details class="construction-options".*?</details>','',text,flags=re.S)
    text=re.sub(r'<nav class="scheme-tabs"[^>]*>.*?</nav>','',text,flags=re.S)
    if current:
        buttons=''.join('<a role="button" data-task="'+tid+'" href="/samples/'+tid.lower()+'/"'+(' aria-current="page"' if current==tid else '')+'>'+tid+' · '+label+'</a>' for tid,label in tasks)
        tabs='<div class="library-tabs" role="group" aria-label="Library task">'+buttons+'</div>'
        if '<h1>Library</h1>' not in text:
            text=text.replace('</nav>','</nav><header class="library-heading"><h1><a href="/library/" style="color:inherit;text-decoration:none">Library</a></h1></header>'+tabs,1)
            text=re.sub(r'<h1(\s+[^>]*)?>(.*?)</h1>',lambda m:m[0] if 'Library' in m[2] else '<h2'+(m[1] or '')+'>'+m[2]+'</h2>',text,flags=re.S)
        text=text.replace(tabs,tabs+recorded_sample(current),1)
    return text


def recorded_sample(task):
    """Playable, retained video examples; canvas explorers remain separate."""
    review=SITE_ROOT/'review'
    if task in ('T1','T2','T3'):
        row=json.loads((review/'previews.json').read_text())['tasks'][task]
        options=[(row['video'],row['caption'],row.get('poster',''))]
        note='Browser preview of a verified evaluation video. Download the lossless 16-bit source from the dataset.'
    elif task=='T4':
        clips=json.loads((review.parents[1]/'runs/stimulus-preparation/preparation.json').read_text())['T4']['clips']
        options=[('/library/media/'+r['preview'],f"{r['coherence_probability']*100:g}% coherence · {r['direction_truth'] or 'no signal'}",'/library/media/'+Path(r['preview']).stem+'.jpg') for r in clips]
        note='Prepared random-dot motion samples: 640 × 640, 180 frames at 60 fps. These full-video samples have not been scored.'
    else:
        row=next(r for r in json.loads((review/'plan.json').read_text())['tasks'] if r['id']=='T5')['examples']['classic']
        options=[(row['video'],'Two-event detection · earlier mask protocol',row.get('poster',''))]
        note='Earlier two-event sample with its target reference. The simplified single / dual-event design is still proposed.'
    opts=''.join('<option value="'+html.escape(url,quote=True)+'" data-poster="'+html.escape(poster,quote=True)+'">'+html.escape(label)+'</option>' for url,label,poster in options)
    reference=('<figure><img src="'+row['reference']+'" alt="Canonical target reference" style="max-width:140px"><figcaption>Target reference</figcaption></figure>') if task=='T5' else ''
    download=('<a href="https://huggingface.co/datasets/DrQY/vlm-psychophysics-classic-t1-t3/tree/main/'+task+'">Download '+task+' sources ↗</a>') if task in ('T1','T2','T3') else ''
    return '<section class="recorded-sample" aria-label="Recorded video"><h2>Video sample</h2><label>Example <select id="recorded-choice">'+opts+'</select></label><div class="recorded-media">'+reference+'<video id="recorded-video" controls playsinline muted loop preload="metadata" src="'+options[0][0]+'" poster="'+options[0][2]+'"></video></div><p id="recorded-status" role="status"></p><p class="sample-note">'+note+' '+download+'</p></section><script type="module" src="/library/player.js"></script>'+task_card(task)+'<h2 class="explorer-heading">Interactive preview</h2>'


def task_card(task):
    data=json.loads((SITE_ROOT/'review/task-cards.json').read_text())
    card=data['tasks'][task];card=card.get('library',card)
    fields=lambda rows:'<dl>'+''.join('<dt>'+html.escape(k)+'</dt><dd>'+html.escape(v)+'</dd>' for k,v in rows)+'</dl>'
    shared=('<h3>Evaluation input and output</h3>'+fields(data['shared'])) if task in ('T1','T2','T3') else ''
    return '<link rel="stylesheet" href="/library/task-cards.css"><details class="task-card"><summary>Task card · '+task+' · '+html.escape(card['title'])+'<span>'+html.escape(card['status'])+'</span></summary>'+fields(card['fields'])+''.join('<p>'+html.escape(note)+'</p>' for note in card['notes'])+shared+'</details>'


def current_library_code(text, prefix, name, review):
    """Use retained source/object pickers with only the current public protocol preview."""
    match=re.search(r'/samples/t([1-5])/',prefix)
    if not match or '/convolution/' in prefix:return text
    task='T'+match[1]
    config=json.loads((review/'library.json').read_text())
    if name=='index.html' and task in ('T4','T5'):
        return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><meta http-equiv="refresh" content="0;url=/experiments/?task='+task+'"><title>'+task+' · Visual perception</title><a href="/experiments/?task='+task+'">Current '+task+' source comparison</a></html>'
    if name=='index.html':
        text=re.sub(r'<label[^>]*>\s*<input[^>]*id="(?:earlier-enabled|earlier)"[^>]*>.*?</label>',lambda m:'<input type="checkbox" id="'+('earlier-enabled' if task=='T1' else 'earlier')+'" hidden>',text,flags=re.S)
        if task=='T1':
            text=re.sub(r'<title>.*?</title>','<title>Library · T1</title>',text)
            text=text.replace('Matched grating','Classic').replace('Current control','Classic')
            text=re.sub(r'<select id="phase">.*?</select>','<select id="phase">'+''.join('<option value="'+str(__import__('math').pi/24+i*__import__('math').pi/12)+'">'+str(7.5+15*i)+'°</option>' for i in range(24))+'</select>',text,flags=re.S)
            text=text.replace('max="5" step="1" value="2"','max="7" step="1" value="3"').replace('max="4" step="1" value="3"','max="9" step="1" value="4"')
            text=re.sub(r'(<pre id="model-prompt">).*?(</pre>)',lambda m:m[1]+html.escape(config['prompts']['T1'])+m[2],text,flags=re.S)
            text=re.sub(r'<p class="prompt-note">.*?</p>','',text,flags=re.S)
            # Keep existing source and comparison controls; retire outdated method text.
            text=re.sub(r'<details class="support"><summary>Method &amp; references</summary>.*?</details>','<details class="support"><summary>Method</summary><p>3 seconds · 60 fps · 180 source frames. Fixed 5 Hz; 448 px cosine-edge aperture. Natural sources use RGB multiplication at gain .5; the classic grating uses fixed mean linear Y .31. The 24 phases are 7.5° + 15° × k.</p><p>Interactive display: 768 × 512, 8-bit RGB. Square background textures fill this canvas by equal-scale center cropping.</p><p id="runtime-description"></p></details>',text,flags=re.S)
            text=re.sub(r'<p class="support-note">Source panels.*?</p>','',text,flags=re.S)
            text=text.replace('<summary>Sources &amp; actual parameters</summary>','<summary>Sources</summary>')
        else:
            text=text.replace('max="71"','max="179"').replace('Matched classic','Classic')
            text=text.replace('<option value="silhouette">Object silhouette</option>','')
        return text
    if task=='T1' and name=='calibration.js':
        return """export const LEVELS={sf:[.5,1,2,4,8,16,32,64],m:[.05,.1,.2,.3,.4,.6,.7,.8,.9,1],tf:[5],duration:[3]};
export const DEFAULTS={sf:4,m:.4,tf:5,duration:3,direction:'RIGHT',phase:Math.PI/24};
export function calibrate(source,p){return {dc:.31,k:.31*p.m,mu:.31,calibrationVersion:'fixed_classic_mean_Y_v18'};}
export function phaseText(p){return (p.phase*180/Math.PI).toFixed(1)+'°';}
"""
    if task=='T1' and name=='style.css':
        return text+"\n/* Current preview canvas: 768 x 512, square pixels. */\n.image-slot{aspect-ratio:3/2}.column-title{min-height:2.6em}.fit-window .stimulus figcaption{height:30px}@media(max-width:650px){.column-title{min-height:0}}\n"
    if task=='T1' and name=='renderer.js':
        text=text.replace('frame/120','frame/60').replace('duration*120','duration*60').replace('.62*linearRGB(raw)','.5*linearRGB(raw)')
        # The retained background textures are square. Cover the 3:2 preview by
        # an equal-scale center crop, while defining the aperture in pixels.
        text=text.replace('texture2D(source,uv).rgb','texture2D(source,vec2(uv.x,(uv.y-.5)*(2./3.)+.5)).rgb')
        text=text.replace('vec2 pos=(uv-.5)*512.','vec2 pos=(uv-.5)*vec2(768.,512.)')
        text=text.replace('gl.RGBA,512,512,0','gl.RGBA,768,512,0').replace('g.viewport(0,0,512,512)','g.viewport(0,0,768,512)')
        text=text.replace('new Uint8Array(512*512*4)','new Uint8Array(768*512*4)').replace('readPixels(0,0,512,512,','readPixels(0,0,768,512,')
        text=text.replace('The 512 px rendering buffer','The 768 by 512 rendering buffer')
        return text
    if name!='app.js':return text
    if task=='T1':
        a=text.index('async function loadCalibration(');z=text.index('let drawStats=',a);text=text[:a]+'async function loadCalibration(source){}\n'+text[z:]
        a=text.index('function measurements(){');z=text.index('function saveURL()',a);text=text[:a]+"function measurements(){$('measurements').replaceChildren();}\n"+text[z:]
        text=text.replace('Matched grating','Classic grating');text=text.replace('earlierEnabled:true','earlierEnabled:false').replace("state.earlierEnabled=q.get('earlier')!=='0'",'state.earlierEnabled=false')
        text=text.replace("s.master_dimensions:[512,512]","s.master_dimensions:[768,512]")
        text=text.replace("mode==='classic'?' matched to':''","mode==='classic'?'':''")
        text=text.replace('side=Math.max(140,Math.min(400,(width-gap*(columns-1))/columns,height-48))','side=Math.max(140,Math.min(400,(width-gap*(columns-1))/columns,(height-48)*1.5))')
        text=text.replace('[0,Math.PI/2].some','Array.from({length:24},(_,i)=>Math.PI/24+i*Math.PI/12).some')
        text=text.replace('*120','*60').replace('/120','/60').replace('reference_fps:120','reference_fps:60').replace('maxFrames:360','maxFrames:179')
        text=text.replace('Math.round(p.duration*60)))','Math.round(p.duration*60)))-1')
        text=text.replace('(state.maxFrames/60).toFixed(3)','((state.maxFrames+1)/60).toFixed(3)')
        text=text.replace("mode==='classic'?'Current control'","mode==='classic'?'Classic'")
        text=text.replace('`Matched to ${s.label} · ${sourceFamily(s)}`',"'Fixed mean Y .31'")
        a=text.index('async function boot()');text=text[:a]+"""async function boot(){try{
 renderer=new StimulusRenderer($('renderer'));const response=await fetch(asset('sources.json'));if(!response.ok)throw Error('Sources unavailable');const data=await response.json();
 state.sources=data.sources.map(s=>({...s,family:provenanceFamily(s),inputURL:asset(s.input_path)}));state.metrics=[];
 populateInputs();restoreURL();buildSourcePicker();sourceTable();axisControls();bind();await rebuild();
}catch(error){showError(error);$('empty').hidden=false;$('empty').textContent='The selected inputs could not be loaded.';}}
boot();
"""
        text=text.replace("./calibration.js?v=direction-pooled-v1","./calibration.js?v=current-library").replace("./renderer.js?v=source-master-v1","./renderer.js?v=current-library")
        return text
    # Scene pages retain the same independent background/object combination selector.
    digest=hashlib.sha256((review/'library-render.js').read_bytes()).hexdigest()[:16]
    text,replacements=re.subn(r"(['\"])(?:\./)?media/render\.js\1",lambda _:f"'/experiments/library-render.js?v={digest}'",text,count=1)
    if replacements!=1:raise ValueError(f'Missing scene renderer import: {prefix}')
    text,replacements=re.subn(r"\bmedia=p=>['\"](?:\./)?media/?['\"]\+p","media=p=>new URL('media/'+p,location.href).href",text,count=1)
    if replacements!=1:raise ValueError(f'Missing scene media URL resolver: {prefix}')
    needle='manifest.stimulus_revision=review.stimulus_revision;'
    text=text.replace(needle,needle+"\nconst currentLibrary=await fetch('/experiments/library.json').then(r=>{if(!r.ok)throw Error('Current library unavailable');return r.json();});manifest.tasks[TASK]={...manifest.tasks[TASK],...currentLibrary.tasks[TASK]};manifest.config=currentLibrary.config;manifest.library_layouts=currentLibrary.T4;manifest.library_mask=currentLibrary.T5.mask_source;\n")
    text=text.replace("if(TASK==='T4')task.parameters.speed_px_s={levels:[16,32],default:32,label:'Speed · px/s'};",'')
    text=text.replace("if(TASK==='T3')defaults.tilt_deg=16;",'')
    text=text.replace("earlier:q.get('earlier')!=='0'",'earlier:false').replace("control:q.get('control')==='silhouette'?'silhouette':'classic'","control:'classic'")
    a=text.index('const earlierData=');z=text.index('\n',a);text=text[:a]+'const earlier=null;'+text[z:]
    text=text.replace("T2:['scene_gain','delta_Y','label'],T3:['tilt_deg','label'],T4:['N','coherence','direction','speed_px_s'],T5:['soa','scenario']","T2:['delta_Y','bright_frames','onset_frame','label','baseline'],T3:['tilt_deg','label','sign','onset_frame'],T4:['coherence','direction','seed'],T5:['exposure_frames','scenario','position']")
    text=text.replace("function paramLabel(k,v){","function paramLabel(k,v){if(k==='bright_frames'||k==='exposure_frames')return (v/60*1000).toFixed(v===10?1:0)+' ms';if(k==='onset_frame')return (v/60).toFixed(3)+' s';if(k==='baseline')return 'Background '+[.05,.2,.4][v]+' · dot 0.50';if(k==='seed'&&TASK==='T4')return 'Layout '+(v+1);")
    a=text.index('function updateModelPrompt(){');z=text.index('\nfunction ',a+10)
    text=text[:a]+"""function updateModelPrompt(){
 const list=$('#model-prompt-list');list.replaceChildren();const pre=document.createElement('pre');pre.textContent=currentLibrary.prompts[TASK];list.append(pre);
 $('#model-input-format').textContent=TASK==='T5'?'Single canonical target reference + 180-frame video → YES / NO':'180-frame video → LEFT / RIGHT';$('#model-prompt-note').textContent='';
}
"""+text[z:]
    text=text.replace('im.src=media(p)',"im.src=p.startsWith('/')?p:media(p)")
    text=text.replace('nav a[href="../${TASK.toLowerCase()}/"]','.library-tabs a[data-task="${TASK}"]')
    text=text.replace('nav a[href="/experiments/?task=${TASK}"]','.library-tabs a[data-task="${TASK}"]')
    text=text.replace('canvas.width=reference.width/4','canvas.width=reference.width').replace('Matched classic','Classic')
    text=text.replace('document.title=`${TASK} · Scene review`','document.title=`Library · ${TASK}`')
    text=text.replace("principles[TASK]","''")
    text=re.sub(r"\$\('#summary'\)\.textContent=.*?;\n",lambda _:"$('#summary').textContent='3 seconds · 60 fps · 180 source frames. Interactive 8-bit display preview; natural compositions are exploratory. Use the recorded samples and source dataset for evaluated inputs.';\n",text,count=1)
    return text


def compute_cost_snapshot(root):
    """Measured call time by task; mixed-run loading is retained once per run."""
    runs=[]
    for run_id,label in [('classic_t1_t2_t4_20261019_v18','Nemotron 3.5 Super'),
                         ('t1_next_classic_20261019_v19','Qwen3.8 Next'),
                         ('t3_classic_20261019_v22','Nemotron 3.5 Super')]:
        directory=root/'runs'/run_id
        summary=json.loads((directory/'summary.json').read_text())
        tasks={}
        with (directory/'results.csv').open() as handle:
            for row in csv.DictReader(handle):
                task=tasks.setdefault(row['task'],{'videos':0,'completed_call_seconds':0.})
                task['videos']+=1;task['completed_call_seconds']+=float(row['call_seconds'])
        cost=summary['cost']
        assert abs(sum(t['completed_call_seconds'] for t in tasks.values())-cost['completed_call_seconds'])<1e-5
        runs.append({'run_id':run_id,'model':label,'tasks':tasks,
            'load_seconds':cost['load_seconds'],'completed_call_seconds':cost['completed_call_seconds'],
            'load_plus_completed_call_GPUh':cost['load_plus_completed_call_GPUh'],
            'shared_load':len(tasks)>1})
    return {'definition':'Sum of same-host monotonic model loading and completed-call times across GPUs; not elapsed wall time or full allocation time.',
            'runs':runs}


def refresh_t3(root, review):
    """Export T3 labels and the two full response examples requested by the user."""
    import importlib.util
    source=root/'analysis/summarize_classic_panel.py'
    module_spec=importlib.util.spec_from_file_location('public_t3_answer_parser',source)
    analysis=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(analysis)
    run=root/'runs/t3_classic_20261019_v22'
    registry=json.loads((run/'model_selection.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    specs={s['sample_id']:s for s in manifest['samples']}
    angles=[1,2,4,8,12,16,24,32,45,90]
    encode=lambda value:{'LEFT':0,'RIGHT':1}.get(value,-1)
    trials=[]; seen=set(); source_digests={}; response_examples=[]
    example_ids={'v22_t3_i21_a4_sp1_left':'numbered_conclusion',
                 'v22_t3_i01_a1_sp1_left':'extra_boundary'}
    for shard in ('fourier0','fourier1','radon0'):
        path=run/f'nemotron35_classic_{shard}_off/results.jsonl'
        source_digests[shard]=hashlib.sha256(path.read_bytes()).hexdigest()
        for line in path.read_text().splitlines():
            row=json.loads(line); sid=row['sample_id']; spec=specs[sid]
            assert sid not in seen
            seen.add(sid)
            for key in ('prompt','correct_answer','scientific_spec_sha256','instance_index'):
                assert row[key]==spec[key], (sid,key)
            assert row['generation']==registry['generation'] and row['native_chat_kwargs']['enable_thinking'] is False
            valid=row['finish_reason']=='stop' and not row['output_truncated'] and row['format_valid'] and row['final_answer'] in ('LEFT','RIGHT')
            strict_prediction=encode(row['final_answer']) if valid else -1
            answer,_=analysis.reported_answer(row)
            prediction=encode(answer)
            truth=encode(row['correct_answer'])
            assert (strict_prediction==truth)==bool(row['correct'])
            trials.append([angles.index(spec['params']['tilt_deg']),spec['instance_index'],spec['params']['sign'],truth,prediction,int(valid),strict_prediction])
            if sid in example_ids:
                assert not valid and answer in ('LEFT','RIGHT') and row['finish_reason']=='stop'
                raw=row['raw_answer']
                if example_ids[sid]=='numbered_conclusion':
                    match=re.search(r'(?m)^5\.\s+\*\*Conclusion:\*\*',raw)
                    assert match and '<think>' not in raw and '</think>' not in raw
                    assert prediction==truth
                else:
                    assert raw.count('</think>')==1
                    assert prediction!=truth
                assert raw.strip().endswith(answer)
                response_examples.append({'sample_id':sid,'angle_deg':spec['params']['tilt_deg'],
                    'onset_seconds':spec['params']['onset_frame']/60,
                    'truth':row['correct_answer'],'answer':answer,'correct':prediction==truth,
                    'format_valid':False,'thinking':False,'output_tokens':row['output_tokens'],
                    'output':raw,'output_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
                    'source_row_sha256':hashlib.sha256(line.encode()).hexdigest()})
    summary_path=run/'summary.json'
    summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
    complete=len(trials)==912 and summary.get('status')=='COMPLETE_912_EXACT_IDS'
    if complete: assert sum(r[3]==r[4] for r in trials)==summary['answer_scoring']['correct']
    payload=json.loads((review/'classic.json').read_text())
    payload['T3']={'status':'complete' if complete else 'partial','n':len(trials),'planned_n':912,
        'label':'Nemotron 3.5 Super','model':registry['model'],'generation':registry['generation'],
        'angles':angles,'onset_frames':[18+5*k for k in range(24)],'exposure_frames':30,
        'prompt':registry['prompt_G']['T3'],'trial_columns':['angle_index','onset_index','tilt_sign','truth_side','final_answer_side','single_label_format_valid','strict_predicted_side'],
        'side_encoding':{'LEFT':0,'RIGHT':1,'INVALID':-1},'trials':sorted(trials),
        'source_raw_sha256':source_digests,'curves':summary.get('answer_scoring',{}).get('curves',[]) if complete else [],
        'score_definition':'Explicit final answer scored regardless of extra text; strict-format issues reported separately. No inference rerun.',
        'answer_parser_version':analysis.ANSWER_PARSER_VERSION,
        'answer_parser_source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'strict_format_correct':summary.get('tasks',{}).get('T3',{}).get('correct'),
        'format_invalid':sum(r[5]==0 for r in trials),
        'tilt_sign_convention':{'-1':'clockwise','1':'counterclockwise'},
        'response_examples':sorted(response_examples,key=lambda e:not e['correct']),
        'exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    assert len(response_examples)==2
    payload['compute_costs']=compute_cost_snapshot(root)
    (review/'classic.json').write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'T3_observed':len(trials),'T3_planned':912,'T3_status':payload['T3']['status']}))


def refresh_classic(root, review):
    """Export completed T1/T2 scalar trials only; no private runtime or reasoning."""
    import importlib.util
    import numpy as np
    runs=root/'runs'
    nemo=runs/'classic_t1_t2_t4_20261019_v18'
    nxt=runs/'t1_next_classic_20261019_v19'
    registry=json.loads((nemo/'model_selection.json').read_text())
    scientific=registry['scientific_config']
    sf=scientific['T1']['SF_cycles_per_448px_aperture']
    contrasts=scientific['T1']['multiplier_depth']
    t2=scientific['T2']; baselines=t2['background_linear_Y']; increments=t2['increment_delta_Y']
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    previous=json.loads((review/'classic.json').read_text()) if (review/'classic.json').exists() else {}
    payload={'version':'classic-20261019','exported_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'shared':{'seconds':3,'fps':60,'source_frames':180,'thinking':'OFF','output_budget':16384,'context':65536},
        'T1':{'sf':sf,'contrasts':contrasts,'phase_degrees':[7.5+15*k for k in range(24)],
              'frequency_hz':5,'aperture_px':448,'mean_linear_Y':.31,'prompt':registry['prompt_G']['T1'],
              'trial_columns':['sf_index','contrast_index','phase_index','truth_side','predicted_side'],
              'side_encoding':{'LEFT':0,'RIGHT':1,'INVALID':-1},'models':[]},
        'T2':{'baselines':baselines,'pedestals':t2['dot_pedestal_linear_Y'],'increments':increments,
              'durations_frames':[10,30],'onset_frames':[18+5*k for k in range(24)],'prompt':registry['prompt_G']['T2'],
              'trial_columns':['baseline_index','increment_index','duration_index','onset_index','truth_side','predicted_side'],
              'refinement':{'status':'Videos prepared; not evaluated','main_increments':[.02*k for k in range(1,11)],'ceiling_increment':.5,'total_videos':3168}}}
    payload['compute_costs']=compute_cost_snapshot(root)
    encode=lambda value:{'LEFT':0,'RIGHT':1}.get(value,-1)
    for directory,key,label in [(nxt,'next','Qwen3.8 Next'),(nemo,'nemo','Nemotron 3.5 Super')]:
        summary=json.loads((directory/'summary.json').read_text())
        assert summary['status'].startswith('COMPLETE_') and summary['missing']==0 and summary['duplicate']==0
        rows=list(csv.DictReader((directory/'results.csv').open()))
        t1rows=[r for r in rows if r['task']=='T1']; assert len(t1rows)==3840
        trials=[[sf.index(float(r['SF'])),contrasts.index(float(r['contrast'])),int(r['instance']),encode(r['truth']),encode(r['prediction'])] for r in t1rows]
        assert len({tuple(r[:4]) for r in trials})==3840
        assert sum(r[3]==r[4] for r in trials)==summary['tasks']['T1']['correct']
        payload['T1']['models'].append({'id':key,'label':label,'model':summary['model'],'generation':summary['generation'],'n':3840,
            'correct':summary['tasks']['T1']['correct'],'right_answers':sum(r[4]==1 for r in trials),
            'trials':trials,'source_csv_sha256':digest(directory/'results.csv'),
            'source_summary_sha256':digest(directory/'summary.json')})
        if key=='nemo':
            rows=[r for r in rows if r['task']=='T2']; assert len(rows)==2880
            trials=[[baselines.index(float(r['background_Y'])),increments.index(float(r['delta_Y'])),[10,30].index(int(r['bright_frames'])),int(r['instance']),encode(r['truth']),encode(r['prediction'])] for r in rows]
            assert len({tuple(r[:5]) for r in trials})==2880
            assert sum(r[4]==r[5] for r in trials)==1958
            payload['T2'].update({'model':summary['model'],'generation':summary['generation'],'label':label,'trials':trials,
                'paired_durations':summary['T2_500ms_vs167ms'],'source_csv_sha256':digest(directory/'results.csv')})
    # Exact frozen source RGB bank, not a new stimulus implementation or inference.
    source=root/'src/stimuli/build_classic_panel.py'
    spec=importlib.util.spec_from_file_location('public_classic_bank',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    frames=module.t1_bank(4,1); atlas=Image.new('RGB',(512*6,512*4))
    for i,frame in enumerate(frames): atlas.paste(Image.fromarray(frame),((i%6)*512,(i//6)*512))
    atlas_path=review/'figures/classic_sf4_c1_frames.png';atlas_path.parent.mkdir(parents=True,exist_ok=True);atlas.save(atlas_path,optimize=True)
    decoded=np.asarray(Image.open(atlas_path))
    for i,frame in enumerate(frames): assert np.array_equal(decoded[(i//6)*512:(i//6+1)*512,(i%6)*512:(i%6+1)*512],frame)
    payload['T1']['frame_bank']={'url':'/experiments/figures/classic_sf4_c1_frames.png','sf':4,'contrast':1,
        'columns':6,'tile_px':512,'phase_indices':[0,6,12,18],'generator_sha256':digest(source),
        'RGB_sha256':[hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames],
        'frame_index_formula':'(initial_phase_index - direction_sign * 2 * frame) modulo 24',
        'note':'Lossless source RGB reconstructed with the unchanged, frozen generator; not processor frames.'}
    if 'T3' in previous: payload['T3']=previous['T3']
    target=review/'classic.json';target.write_text(json.dumps(payload,separators=(',',':'),allow_nan=False)+'\n')
    print(json.dumps({'snapshot':str(target),'bytes':target.stat().st_size,'atlas_bytes':atlas_path.stat().st_size,
        'T1_trials_per_model':3840,'T2_trials':2880,'frame_bank_lossless_tiles_verified':24}))


def sanitize(value):
    if isinstance(value, dict):
        out = {k:sanitize(v) for k,v in value.items() if k not in DROP and not (isinstance(v,str) and PRIVATE.search(v))}
        if out.get('id') == 'photo_obj_010':
            out['author'] = 'Fir0002/Flagstaffotos'
        if out.get('id') == 'photo_obj_002' and not out.get('license_url'):
            out['license_url'] = 'https://creativecommons.org/licenses/by/2.0/'
        return out
    if isinstance(value,list): return [sanitize(v) for v in value]
    if isinstance(value,str): return PRIVATE.sub('[local provenance omitted]', value)
    return value


def current_results(root, code_root):
    """Export measured scalar cells only; preserve every historical run and curve."""
    def read(run):
        return json.loads((root/'runs'/run/'analysis/summary.json').read_text())
    active=('nemo','qwen','cosmos')
    t1,video,refinement,static,t4=map(read,('t1-rgb16','t2-t3-rgb16','t3-threshold-refinement','t2-t3-static','t4-rgb16'))
    def cell(c):
        return {k:c[k] for k in ('cell','model','correct','completed','expected','prediction_counts',
            'correct_by_truth','truth_counts','direction_pairs_both_correct','direction_pairs_completed',
            'pointwise_95pct_onset_bootstrap') if k in c}
    result={'schema':'current-rgb16-results-v1','models':[
        {'id':'nemo','label':'Nemotron 3.5 Super VL','color':'#a75b19'},
        {'id':'qwen','label':'Qwen3.8 Flash Next','color':'#3569ba'},
        {'id':'cosmos','label':'Cosmos3 Super','color':'#147d73'}],
        'T1':{},'video_cells':[cell(c) for c in video['cells']+refinement['cells'] if c['model'] in active],
        'static':{k:{'normal':m['normal'],'cells':m['cells'],'task_prompt':m['task_prompt']} for k,m in static['models'].items()},
        'T4':{k:{x:m[x] for x in ('normal','n','correct','answers','cells')} for k,m in t4['models'].items()},
        'sources':['t1-rgb16','t2-t3-rgb16','t3-threshold-refinement','t2-t3-static','t4-rgb16']}
    for k,m in t1['models'].items():
        result['T1'][k]={'video_normal':m['video']['normal'],'video_expected':m['video']['expected'],
            'main_panel':m['main_panel_video'],'static_normal':m['static']['normal'],
            'static_cells':m['static']['cells'],
            'video':{field:m['video'][field] for field in ('complete','n','correct','answers','cells')}}
        assert m['static']['complete'] and m['static']['normal']==960
    assert all(video['models'][k]['completed']==3504 and refinement['models'][k]['completed']==576 for k in active)
    assert all(m['normal']==456 for m in static['models'].values())
    assert all(m['normal']==480 and m['complete'] for m in t4['models'].values())
    qsa=json.loads((root/'runs/reproducibility/operator/early-stop-qsa-v3-analysis.json').read_text())
    assert qsa['status']=='COMPLETE_MATCH_ALL_49' and qsa['normal']==196
    result['qsa']={k:qsa[k] for k in ('operational_inputs','scientific_inputs','observations_per_input','normal',
        'original_scientific_correct','production_pairing','official_config_only','historical_outputs_replaced')}
    # Bind the highlighted condition to its actual IDs, GT and current static receipts.
    from collections import Counter
    _sys.path.insert(0,str(code_root))
    from classic_precision.summary import model_snapshot, technical_passed
    manifest_path=root/'runs/t2-t3-rgb16/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    def focus_row(r):
        p=r['source_parameters']
        return r['task']=='T2' and p['background_Y']==.05 and p['delta_Y']==.11
    rows=[r for r in manifest['rows'] if focus_row(r) and
          r['source_parameters']['offset_frame']-r['source_parameters']['onset_frame']==30]
    assert len(rows)==len({r['sample_id'] for r in rows})==48
    assert Counter(r['truth'] for r in rows)=={'LEFT':24,'RIGHT':24}
    assert all(r['truth']==r['source_parameters']['target_side'] for r in rows)
    input_spec=manifest['protocol']['input']
    source=rows[0]['source_parameters']
    qsa_rows={r['sample_id']:r for r in qsa['per_id'] if r['scientific']}
    assert set(qsa_rows)=={r['sample_id'] for r in rows}
    assert all(r['all_actual_input_hashes_equal'] and r['all_full_outputs_equal'] for r in qsa_rows.values())
    focus={'background_Y':.05,'delta_Y':.11,'exposure_frames':30,
        'initial_dot_Y':source['initial_dot_Y'],'target_event_Y':source['initial_dot_Y']+.11,
        'truth_counts':dict(Counter(r['truth'] for r in rows)),
        'source':{k:input_spec[k] for k in ('source_wh','fps','frames','seconds','master_bits','master_codec','master_pixel_format')},
        'onset_frames':sorted({r['source_parameters']['onset_frame'] for r in rows}),
        'video':{k:next(cell(c) for c in video['cells'] if c['model']==k and c['cell']==['T2',.05,30,.11]) for k in active},
        'static':{},'qwen_qsa':{'correct':sum(r['correct'] for r in qsa_rows.values()),'n':48,
            'observations_per_input':qsa['observations_per_input'],'comparison_sha256':qsa['comparison_sha256']},
        'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'samples':[{'sample_id':r['sample_id'],'truth':r['truth'],
            'onset_frame':r['source_parameters']['onset_frame'],
            'original_qwen_answer':qsa_rows[r['sample_id']]['original']['raw_answer'],
            'qsa_qwen_answer':qsa_rows[r['sample_id']]['current']['raw_answer']} for r in rows]}
    static_run=root/'runs/t2-t3-static'
    static_manifest_path=static_run/'execution-manifest.json'
    assert hashlib.sha256(static_manifest_path.read_bytes()).hexdigest()==static['manifest_sha256']
    static_manifest=json.loads(static_manifest_path.read_text())
    targets={r['sample_id']:r for r in static_manifest['rows'] if focus_row(r) and r['prompt_id']=='P1'}
    assert len(targets)==2 and Counter(r['truth'] for r in targets.values())=={'LEFT':1,'RIGHT':1}
    for k in static['models']:
        _,raw,events,raw_tail,event_tail=model_snapshot(static_run/'outputs'/k)
        assert not raw_tail and not event_tail
        passed=technical_passed(raw,events)
        selected=[r for r in raw if r['sample_id'] in targets]
        assert len(selected)==len({r['sample_id'] for r in selected})==2
        for r in selected:
            assert r['sample_id'] in passed and r['finish_reason']=='stop' and r['raw_answer'] in ('LEFT','RIGHT')
            assert all(r[f]==targets[r['sample_id']][f] for f in ('task','truth','source_parameters','prompt_id','prompt','image_id'))
        focus['static'][k]={'correct':sum(r['raw_answer']==r['truth'] for r in selected),'n':2,
            'prompt':'P1','sample_ids':[r['sample_id'] for r in selected]}
    result['T2_focus']=focus
    # Publish a complete task from the ID-paired analysis; never mix a partial
    # corrected task into the original cells or overwrite a historical answer.
    paired_path=root/'runs/reproducibility/operator/qwen-t2t3-single-pass-paired-analysis.json'
    paired=json.loads(paired_path.read_text())
    refinement_path=root/'runs/t3-threshold-refinement/manifest.json'
    refinement_manifest=json.loads(refinement_path.read_text())
    assert not paired['pending'] and len(paired['records'])==4080
    for task,expected_count in (('T2',2304),('T3',1776)):
        corrected=[r for r in paired['records'] if r['task']==task]
        expected={r['sample_id'] for r in manifest['rows']+refinement_manifest['rows'] if r['task']==task}
        assert len(corrected)==len({r['sample_id'] for r in corrected})==expected_count
        assert {r['sample_id'] for r in corrected}==expected
    for saved in paired['source_files']:
        assert hashlib.sha256(_FilesystemPath(saved['path']).read_bytes()).hexdigest()==saved['sha256']
    result['corrected_video_cells']=[{
        'cell':c['cell'],'model':'qwen','correct':c['new_correct'],
        'completed':c['n'],'expected':c['expected'],
        'prediction_counts':{'RIGHT':c['new_RIGHT'],'LEFT':c['n']-c['new_RIGHT']}}
        for c in paired['cells'] if c['cell'][0] in ('T2','T3') and c['complete']]
    assert Counter(c['cell'][0] for c in result['corrected_video_cells'])=={'T2':48,'T3':19}
    result['qwen_current_scope']={'T2':'COMPLETE','T3':'COMPLETE',
        'T2_completed':2304,'T3_completed':1776,
        'analysis_sha256':hashlib.sha256(paired_path.read_bytes()).hexdigest(),
        'refinement_manifest_sha256':hashlib.sha256(refinement_path.read_bytes()).hexdigest()}
    # Keep the archived original T1 separate from the complete corrected run.
    t1_path=root/'runs/reproducibility/operator/qwen-t1-corrected-complete-20261024-analysis.json'
    t1_ids=root/'runs/reproducibility/operator/qwen-t1-corrected-complete-20261024-per-id.csv'
    t1_current=json.loads(t1_path.read_text())
    assert t1_current['status']=='COMPLETE' and t1_current['normal']==3840 and t1_current['correct']==1924
    with t1_ids.open() as handle:
        t1_rows=list(csv.DictReader(handle))
    t1_manifest=json.loads((root/'runs/t1-rgb16/manifest.json').read_text())
    assert len(t1_rows)==len({r['sample_id'] for r in t1_rows})==3840
    assert {r['sample_id'] for r in t1_rows}=={r['sample_id'] for r in t1_manifest['samples']}
    t1_cells=[]
    for saved in t1_current['cells']:
        rows=[r for r in t1_rows if float(r['sf'])==saved['sf'] and float(r['m'])==saved['m']]
        assert len(rows)==48 and Counter(r['truth'] for r in rows)=={'LEFT':24,'RIGHT':24}
        correct=sum(r['truth']==r['answer'] for r in rows)
        assert correct==saved['correct'] and saved['expected']==48
        t1_cells.append({'sf_cycles_per_aperture':saved['sf'],'modulation_depth':saved['m'],
            'arm':'video','expected_n':48,'n':48,'correct':correct,'answers':dict(Counter(r['answer'] for r in rows))})
    assert len(t1_cells)==80 and sum(c['correct'] for c in t1_cells)==1924
    result['corrected_T1']={'complete':True,'n':3840,'correct':1924,'cells':t1_cells,
        'answers':dict(Counter(r['answer'] for r in t1_rows)),
        'analysis_sha256':hashlib.sha256(t1_path.read_bytes()).hexdigest(),
        'per_id_csv_sha256':hashlib.sha256(t1_ids.read_bytes()).hexdigest()}
    # Completed duration-paired analysis already covers both native models.
    # Export its all-onset cohort, retaining the original 2304-video panel above.
    duration_path=root/'runs/t2-duration-extension/analysis/duration-paired/cells.csv'
    with duration_path.open() as handle:
        duration_rows=[r for r in csv.DictReader(handle) if r['cohort']=='all']
    expanded=[]
    for model in ('nemo','cosmos'):
        rows=[r for r in duration_rows if r['model']==model]
        assert len(rows)==39 and all(int(r['pairs'])==48 for r in rows)
        for r in rows:
            for duration,prefix in ((10,'short'),(30,'long')):
                right=int(r[prefix+'_RIGHT'])
                expanded.append({'cell':['T2',float(r['background_Y']),duration,float(r['delta_Y'])],
                    'model':model,'correct':int(r[prefix+'_correct']),'completed':48,'expected':48,
                    'prediction_counts':{'RIGHT':right,'LEFT':48-right}})
    registry=json.loads((root/'runs/t2-t3-rgb16/registry.json').read_text())
    final=registry['execution']['native_completion_20261023']['T2_extension']['Qwen_extension']['final_analysis']
    assert final['status']=='COMPLETE' and final['parent_corrected_analysis']['sha256']==hashlib.sha256(paired_path.read_bytes()).hexdigest()
    merged=final['merged_corrected_task']
    assert merged['expected']==3744 and merged['correct']==3005 and merged['cells']==78
    expanded += [dict(c,model='qwen') for c in merged['cells_data']]
    expected_cells={tuple(c['cell']) for c in expanded if c['model']=='qwen'}
    for model in active:
        cells=[c for c in expanded if c['model']==model]
        assert len(cells)==78 and {tuple(c['cell']) for c in cells}==expected_cells
        assert sum(c['completed'] for c in cells)==3744
        assert Counter(c['cell'][2] for c in cells)=={10:39,30:39}
    assert sum(c['correct'] for c in expanded if c['model']=='qwen')==3005
    result['expanded_video_cells']=expanded
    result['qwen_current_scope'].update(T1='COMPLETE',T1_completed=3840,T2_completed=3744,
        T2_original_completed=2304,T2_extension_completed=1440,
        duration_cells_sha256=hashlib.sha256(duration_path.read_bytes()).hexdigest())
    # Reuse the completed full source audit; publishing never reruns stimulus QA.
    audit_root=root/'runs/reproducibility/operator/classic-source-reverification-20261024'
    summary_path=audit_root/'summary.json'
    qa_path=audit_root/'source-qa.json'
    audit=json.loads(summary_path.read_text())
    qa=json.loads(qa_path.read_text())
    assert audit['status']=='COMPLETE' and audit['source_qa_status']==qa['status']=='PASS'
    assert audit['completed_counts']=={'T1':3840,'T2':3744,'T3':1776}
    assert audit['completed_videos']==9360 and audit['decoded_frames']==1684800 and not audit['failed_IDs']
    assert audit['max_decoded_code_error']==0 and qa['t1']['pass']
    measurements=audit['numeric_measurement_summary']
    result['precision']={
        'source_wh':[768,512],'fps':60,'frames':180,'seconds':3,'bits_per_channel':16,
        'decoded_videos':9360,'decoded_frames':1684800,'max_decoded_code_error':0,
        'task_video_counts':audit['completed_counts'],'max_PTS_error_ms':1000*audit['max_PTS_error_s'],
        'T2_states':len({(r['source_parameters']['background_Y'],r['source_parameters']['delta_Y'])
            for r in qa['states'].values() if r['source_parameters']['task']=='T2'}),
        'T2_max_linear_luminance_error':measurements['T2_max_core_linear_Y_error'],
        'T2_outside_target_changed_pixels':measurements['T2_changed_outside_target'],
        'T3_geometries':qa['state_panel']['T3_unique_geometries'],
        'T3_max_PCA_error_deg':measurements['T3_max_PCA_error_deg'],
        'T3_max_blind_edge_error_deg':measurements['T3_max_halfplane_error_deg'],
        'T1_integral_checked_pixels':sum(r['pixels'] for r in qa['t1']['pixel_integrals']),
        'T1_max_integral_linear_Y_error':max(r['max_possible_linear_Y_error_at_m1'] for r in qa['t1']['pixel_integrals']),
        'T1_readout_checks':3840,'T1_readout_max_errors':measurements['T1_max_blind_error'],
        'evidence_sha256':{'full_summary':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            'source_qa':hashlib.sha256(qa_path.read_bytes()).hexdigest(),
            'per_id':hashlib.sha256((audit_root/'per-id.csv').read_bytes()).hexdigest()}}
    return current_snapshot(root, result)


def current_snapshot(root, result):
    result=add_gemma_video(root,result)
    result=add_current_video(root,result)
    result=add_cosmos_x_full(root,result)
    result=add_qwen_t2_x(root,result)
    path=root/'runs/reproducibility/operator/current-static-final-20261026-summary.json'
    if path.exists() and not all(k in result for k in ('static_current','T1_static_current','current_static')):
        current=json.loads(path.read_text());assert current['status']=='COMPLETE_PASS' and current['single_counts']=={k:456 for k in ['qwen','cosmos','gemma','nemo']}
        result['static_current']=current['static_current'];result['T1_static_current']=current['T1_static_current']
        result['current_static']={'model_versions':current['model_versions'],'source_geometry':current['source_geometry'],'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'COMPLETE_PASS'}
    result['task_cards']=json.loads((SITE_ROOT/'review/task-cards.json').read_text())
    return result


def add_qwen_t2_x(root, result):
    """Export only fully audited classic/natural X cells, with immutable result bindings."""
    runs=root/'runs/reproducibility'
    classic=runs/'qwen-x-classic-t2-20261029'
    natural=runs/'natural-t2-spoon-20261030'
    if not (natural/'classic-X-comparison-summary.json').exists():return result
    load=lambda p:json.loads(p.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    comparison=load(natural/'classic-X-comparison-summary.json')
    assert comparison['status']=='COMPLETE_SAME_X_CLASSIC_NATURAL_COMPARISON'
    assert comparison['all40_replaced_regardless_of_answers'] is True
    for key,folder in [('classic',classic),('natural',natural)]:
        assert digest(folder/'summary.json')==comparison['bindings'][key]['sha256']
    c,n=load(classic/'summary.json'),load(natural/'summary.json')
    assert c['status']=='COMPLETE_CLASSIC_QWEN_T2_X' and n['status']=='COMPLETE_NATURAL_T2_SPOON_X_OVERLAY'
    assert c['overall']['n']==2880 and n['overall']['n']==3840
    assert len(c['cells'])==60 and len(n['cells'])==80
    assert all(r['n']==r['accuracy_n']==48 and r['invalid_n']==0 for r in c['cells']+n['cells'])
    assert {r['delta_Y'] for r in c['cells']}=={r['delta_Y'] for r in n['cells']}
    parent=runs/'natural-t2-qwen-x-20261029'/'summary.json'
    assert digest(parent)==n['parent_summary']['sha256']
    old=load(parent)
    result['qwen_t2_x']={'status':'COMPLETE_PASS','version':'X','precision':'NVFP4',
        'unique_videos':2880,'configs':60,'summary_sha256':digest(classic/'summary.json'),
        'video_cells':[{'cell':['T2',r['background_Y'],30,r['delta_Y']],'model':'qwen',
                        'completed':48,'expected':48,'correct':r['correct'],
                        'prediction_counts':r['answers'],'truth_counts':{'LEFT':24,'RIGHT':24}} for r in c['cells']]}
    result['qwen_t2_transfer']={'status':comparison['status'],'classic':c['cells'],'natural':n['cells'],
        'natural_original':old['cells'],'classic_overall':c['overall'],'natural_overall':n['overall'],
        'spoon_transition':{k:comparison['spoon_paired_transition'][k] for k in
                            ['n','answer_changes','correct_to_wrong','wrong_to_correct','before_correct','after_correct']},
        'bindings':{k:comparison['bindings'][k]['sha256'] for k in ['classic','natural','parent_summary']},
        'paired_A_X_n':comparison['paired_A_X_n'],'A_X_answer_changes':comparison['A_X_answer_changes'],
        'scope':'Same Qwen X runtime, 180 frames, 500 ms; natural changes both objects and backgrounds.'}
    return result


def add_cosmos_x_full(root, result):
    """Expose accepted full X releases separately from all unchanged A curves."""
    for model,run_name,release in [('cosmos','cosmos-x-full-20261028','cosmos-X-20261028'),('nemo','nemo-x-full-20261029','nemo-X-20261029')]:
        folder=root/'runs/reproducibility'/run_name/'analysis'
        path=folder/'summary.json'
        if not path.exists():continue
        summary=json.loads(path.read_text())
        assert summary['status']=='COMPLETE_PASS' and summary['unique_videos']==9456 and summary['configs']==169
        cells_path=folder/'cells.csv'
        assert hashlib.sha256(cells_path.read_bytes()).hexdigest()==summary['outputs']['cells.csv']['sha256']
        rows=list(csv.DictReader(cells_path.open()))
        t1=[];video=[]
        for row in rows:
            key=json.loads(row['cell']);n=int(row['n']);correct=int(row['correct']);answers=json.loads(row['answers'])
            if key[0]=='T1':
                t1.append({'sf_cycles_per_aperture':key[1],'modulation_depth':key[2],
                           'arm':'video','expected_n':n,'n':n,'correct':correct,'accuracy':correct/n,'answers':answers})
            else:
                video.append({'cell':key,'model':model,'correct':correct,'completed':n,'expected':n,
                              'prediction_counts':answers,'truth_counts':{'LEFT':n//2,'RIGHT':n//2}})
        assert len(t1)==80 and len(video)==89
        result[model+'_x_full']={'status':'COMPLETE_PASS','version':'X','precision':'NVFP4',
                                'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                'unique_videos':9456,'configs':169,'tasks':summary['tasks'],
                                'T1':{'complete':True,**summary['tasks']['T1'],'cells':t1},
                                'video_cells':video,'A_X_paired':summary['A_X_paired'],
                                'results_url':'https://huggingface.co/datasets/DrQY/vlm-psychophysics-classic-t1-t3/tree/main/results/'+release}
    return result


def add_current_video(root, result):
    """Export closed A/NVFP4 answers separately from all historical panels."""
    from collections import Counter, defaultdict
    folder=root/'runs/reproducibility/current-nvfp4'
    summary_path=folder/'summary.json'
    if not summary_path.exists():
        return result
    summary=json.loads(summary_path.read_text())
    source=folder/'video-predictions.csv'
    assert summary['status']=='COMPLETE' and summary['video_rows']==37440
    assert hashlib.sha256(source.read_bytes()).hexdigest()==summary['video_csv_sha256']
    expected={'T1':3840,'T2':3744,'T3':1776}
    cells=defaultdict(lambda:{'n':0,'correct':0,'answers':Counter(),'truths':Counter()})
    seen=set();identities={};totals=defaultdict(lambda:{'n':0,'correct':0})
    with source.open(newline='') as handle:
        for row in csv.DictReader(handle):
            model,task,sid=row['model'],row['task'],row['sample_id']
            assert model in summary['models'] and task in expected and row['protocol']=='controlled-rgb16'
            assert (model,sid) not in seen
            seen.add((model,sid))
            identity=(task,row['truth'],row['source_video_sha256'],row['source_stream_sha256'])
            assert identities.setdefault(sid,identity)==identity
            correct=int(row['correct'])
            assert row['answer'] in ('LEFT','RIGHT') and correct==int(row['answer']==row['truth'])
            key=json.loads(row['cell_json']);assert key[0]==task
            if task=='T2':
                # The per-ID export stores background, increment, duration;
                # the public plot schema stores background, duration, increment.
                assert len(key)==4 and key[3] in (10,30)
                key=[key[0],key[1],key[3],key[2]]
            # Tilt signs are paired at the same displayed angle, as in the historical plots.
            if task=='T3':
                assert len(key)==3 and key[2] in (-1,1)
                key=key[:2]
            c=cells[(model,tuple(key))]
            c['n']+=1;c['correct']+=correct;c['answers'][row['answer']]+=1;c['truths'][row['truth']]+=1
            total=totals[(model,task)];total['n']+=1;total['correct']+=correct
    assert len(seen)==37440 and len(identities)==9360
    output=[];t1={}
    for model in summary['models']:
        for task,n in expected.items():
            assert totals[(model,task)]==summary['models'][model][task]
            assert totals[(model,task)]['n']==n
        t1_cells=[]
        for (mid,key),c in sorted(cells.items()):
            if mid!=model:continue
            n=c['n'];task=key[0]
            assert n==(48 if task!='T3' or key[1]==90 else 96)
            assert c['truths']=={'LEFT':n//2,'RIGHT':n//2}
            if task=='T1':
                t1_cells.append({'sf_cycles_per_aperture':key[1],'modulation_depth':key[2],
                    'arm':'video','expected_n':n,'n':n,'correct':c['correct'],
                    'accuracy':c['correct']/n,'answers':dict(c['answers'])})
            else:
                output.append({'cell':list(key),'model':model,'correct':c['correct'],
                    'completed':n,'expected':n,'prediction_counts':dict(c['answers']),
                    'truth_counts':dict(c['truths'])})
        assert len(t1_cells)==80
        t1[model]={'complete':True,**totals[(model,'T1')],
            'answers':dict(sum((Counter(c['answers']) for c in t1_cells),Counter())),'cells':t1_cells}
    result['current_video']={'status':'COMPLETE','version':'A','precision':'NVFP4',
        'protocol':summary['protocol'],'dataset_index_id':summary['dataset_index_id'],
        'precision_scope':summary['precision_scope'],'sources':summary['sources'],
        'video_csv_sha256':summary['video_csv_sha256'],
        'summary_sha256':hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        'models':summary['models'],'T1':t1,'video_cells':output,
        'T3_auxiliary_angles':[1.75,2.25]}
    return result


def add_gemma_video(root, result):
    """Attach the separately closed NVFP4 video version to the scalar snapshot."""
    path = root/'runs/reproducibility/operator/gemma-nvfp4-final-20261026-summary.json'
    if not path.exists():
        return result
    final = json.loads(path.read_text())
    assert final['status'] == 'COMPLETE_PASS' and final['total'] == final['unique_ids'] == 9360
    assert final['counts'] == {'T1':3840,'T2':3744,'T3':1776}
    assert final['execution_signature'] == '16300995ef35a0291f016c3bf758ef8411026b80a26fa6d5917f2fcdeefba830'
    result['models'] = [m for m in result['models'] if m['id'] != 'gemma'] + [
        {'id':'gemma','label':'Gemma 4 31B · NVFP4 video','color':'#8b5a91'}]
    result['T1']['gemma'] = {'video':final['website']['T1_video'],'static_cells':[],
        'video_normal':3840,'video_expected':3840,'static_normal':0}
    result['gemma_video_cells'] = final['website']['video_cells']
    result['gemma_video'] = {'status':'COMPLETE_PASS','counts':final['counts'],
        'tasks':final['tasks'],'execution_signature':final['execution_signature'],
        'checkpoint_revision':final['checkpoint_revision'],'protocol':final['protocol'],
        'summary_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'scope':'Independent complete NVFP4 video version; existing static and other-model versions retained'}
    return result


def refresh_previews(root):
    """Derive browser-only previews from identified RGB16 evaluation sources."""
    import subprocess
    import numpy as np
    import av
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    folder=SITE_ROOT/'review/previews';folder.mkdir(exist_ok=True)
    digest=lambda p:hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
    payload={'schema':'evaluated-stimulus-previews-v1','tasks':{},'assets':[],
        'display_encoding':'8-bit H.264 MP4 / RGB8 PNG; display derivatives, not model inputs',
        'derivation':{'encoder':'libx264','CRF':14,'pixel_format':'yuv420p','fps':60,
            'frames':180,'PNG8_from_RGB16':'round(code / 257)',
            'ffmpeg_version':imageio_ffmpeg.get_ffmpeg_version()},
        'new_model_calls':0}
    def probe(path):
        with av.open(str(path)) as container:
            stream=container.streams.video[0];stream.thread_count=1
            rate=stream.average_rate
            info={'width':stream.width,'height':stream.height,
                'nb_read_frames':str(sum(1 for _ in container.decode(stream))),
                'avg_frame_rate':f'{rate.numerator}/{rate.denominator}'}
        assert (info['width'],info['height'],int(info['nb_read_frames']))==(768,512,180)
        assert info['avg_frame_rate']=='60/1'
        return info
    def png(name,codes):
        assert codes.shape==(512,768,3) and codes.dtype==np.uint16
        target=folder/name
        image=((codes.astype(np.uint32)+128)//257).astype(np.uint8)
        Image.fromarray(image).save(target,optimize=True)
        assert np.array_equal(np.asarray(Image.open(target)),image)
        return '/experiments/previews/'+name
    t1=json.loads((root/'runs/t1-rgb16/manifest.json').read_text())
    row=next(r for r in t1['samples'] if r['phase_index']==0 and r['params']['sf_cycles_per_aperture']==4
        and r['params']['modulation_depth']==.4 and r['correct_answer']=='RIGHT')
    ready_path=root/'runs/t1-rgb16/examples'/f'{row["sample_id"]}-ready.json'
    ready=json.loads(ready_path.read_text())
    assert ready['technical_verdict']=='PASS' and ready['validation']['max_code_error']==0
    sources=[('T1',row['sample_id'],root/'runs/t1-rgb16/examples'/f'{row["sample_id"]}.mkv',
              ready['video_sha256'],[0,1,2,3],0,'SF 4 cycles / aperture · contrast .4 · 5 Hz · RIGHT')]
    manifest=json.loads((root/'runs/t2-t3-rgb16/manifest.json').read_text())
    receipts={r['sample_id']:r for r in (json.loads(line) for line in
        (root/'runs/t2-t3-rgb16/remote-fourier/ready.jsonl').read_text().splitlines())}
    selected={'T2':'t2_7359c6a16fa78c3bda54','T3':'t3_1e7fad0670c4705aafdd'}
    captions={'T2':'ΔY .10 · background .05 · 500 ms · RIGHT',
              'T3':'1° tilt · 500 ms · vertical bar LEFT'}
    for task,sid in selected.items():
        r=next(r for r in manifest['rows'] if r['sample_id']==sid)
        assert r['task']==task and r['source_parameters']['onset_frame']==18
        assert receipts[sid]['max_decoded_code_error']==0
        sources.append((task,sid,root/'runs/t2-t3-rgb16/examples'/f'{sid}.mkv',
            receipts[sid]['video_sha256'],[18],18,captions[task]))
    for task,sid,source,sha,frames,poster,caption in sources:
        assert digest(source)==sha
        probe(source)
        target=folder/f'{task}.mp4';staging=folder/f'{task}.partial.mp4'
        command=[ffmpeg,'-v','error','-y','-threads','1','-i',str(source),
            '-an','-c:v','libx264','-preset','medium','-crf','14','-pix_fmt','yuv420p',
            '-r','60','-frames:v','180','-threads','1','-movflags','+faststart',str(staging)]
        subprocess.run(command,check=True,capture_output=True,timeout=120)
        info=probe(staging);staging.replace(target)
        selector='+'.join(f'eq(n\\,{f})' for f in frames)
        command=[ffmpeg,'-v','error','-threads','1','-i',str(source),
            '-vf','select='+selector,'-fps_mode','passthrough','-frames:v',str(len(frames)),
            '-pix_fmt','rgb48le','-f','rawvideo','-threads','1','-']
        raw=subprocess.run(command,check=True,capture_output=True,timeout=90).stdout
        codes=np.frombuffer(raw,dtype='<u2').reshape((len(frames),512,768,3))
        images={str(f):png(f'{task}-frame-{f}.png',c) for f,c in zip(frames,codes)}
        payload['tasks'][task]={'sample_id':sid,'kind':'video','video':f'/experiments/previews/{task}.mp4',
            'images':images,'poster':images[str(poster)],'caption':caption,
            'source_video_sha256':sha,'source_codec':'FFV1 RGB16','preview':info,
            'preview_video_sha256':digest(target)}
    t4=json.loads((root/'runs/t4-rgb16/endpoint_evaluation_manifest.json').read_text())
    variants={}
    for n in [15,25]:
        r=next(r for r in t4['rows'] if r['N']==n and r['coherence']==1 and r['truth']=='RIGHT' and r['seed']==2026102301)
        images={}
        for asset in r['frame_assets']:
            archive=root/'runs/t4-rgb16'/asset['archive_relative']
            assert digest(archive)==asset['archive_sha256']
            with np.load(archive,allow_pickle=False) as arrays:codes=arrays[asset['npz_key']].copy()
            assert hashlib.sha256(np.ascontiguousarray(codes,dtype='<u2').tobytes()).hexdigest()==asset['source_rgb16_sha256']
            f=asset['frame_index'];images[str(f)]=png(f'T4-n{n}-frame-{f}.png',codes)
        variants[str(n)]={'sample_id':r['sample_id'],'images':images,
            'source_pair_sha256':r['source_pair_sha256'],'caption':f'{n} dots · 100% coherence · RIGHT'}
    payload['tasks']['T4']={'kind':'endpoint-images','variants':variants}
    for path in sorted(folder.iterdir()):
        assert path.is_file() and not path.is_symlink() and path.suffix in ('.mp4','.png')
        payload['assets'].append({'name':path.name,'sha256':digest(path),'bytes':path.stat().st_size})
    (folder.parent/'previews.json').write_text(json.dumps(payload,separators=(',',':'))+'\n')
    print(json.dumps({'preview_tasks':4,'video_clips':3,'assets':len(payload['assets']),
        'display_bytes':sum(a['bytes'] for a in payload['assets']),'source_SHA_verified':True,'new_model_calls':0}))


def t45_payloads(root, review, require_complete=False):
    """Read only the coordinator's sealed four-file release, never trial output."""
    analysis=root/'runs/reproducibility/cosmos-x-t45-20261030/analysis'
    summary=analysis/'summary.json'
    complete=summary.is_file() and json.loads(summary.read_text()).get('status') in ('COMPLETE','COMPLETE_COSMOS_X_T45_1200')
    if not complete:
        if require_complete:raise ValueError('Final T45 analysis is not COMPLETE')
        return None
    required=('summary.json','cells.csv','T4-classic-natural.png','T5-classic-natural.png')
    if not all((analysis/name).is_file() for name in required):
        raise ValueError('COMPLETE T45 summary is missing a final figure or condition table')
    payloads={}
    for name in required:
        raw=(analysis/name).read_bytes()
        if name.endswith('.png'):
            with Image.open(io.BytesIO(raw)) as image:
                if image.format!='PNG':raise ValueError('Final T45 figure is not PNG')
                image.verify()
        if name.endswith('.csv') and PRIVATE.search(raw.decode()):
            raise ValueError('Private filesystem path in T45 public condition data')
        if name.endswith('.json'):
            raw=(json.dumps(sanitize(json.loads(raw)),ensure_ascii=False,separators=(',',':'))+'\n').encode()
        payloads['experiments/t45-results/'+name]=raw
    config=json.loads((review/'t45-final.json').read_text())
    final=json.loads(summary.read_text())
    for tid in ('T4','T5'):
        config['tasks'][tid]['result']['status']='complete'
        config['tasks'][tid]['inference_status']='Inference complete'
        config['tasks'][tid]['source_status']='Source QA passed'
        rows=[]
        for row in final.get('by_task_domain',[]):
            if row['task']!=tid:continue
            entry={key:row[key] for key in ('domain','correct','scored_n','n','accuracy','RIGHT','NO')}
            if tid=='T5':
                cells=[cell for cell in final['cells'] if cell['task']=='T5' and cell['domain']==row['domain']]
                present=[cell for cell in cells if cell['subgroup'] in ('target_first','target_second')]
                absent=[cell for cell in cells if cell['subgroup']=='target_absent']
                entry.update(hit=sum(cell['YES'] for cell in present),present_n=sum(cell['n'] for cell in present),false_alarm=sum(cell['YES'] for cell in absent),absent_n=sum(cell['n'] for cell in absent))
            rows.append(entry)
        config['tasks'][tid]['result']['summary_rows']=rows
        config['tasks'][tid]['result']['summary_source_sha256']=hashlib.sha256(summary.read_bytes()).hexdigest()
    config['source_release']='current'
    payloads['experiments/t45-final.json']=(json.dumps(sanitize(config),ensure_ascii=False,separators=(',',':'))+'\n').encode()
    cue=review/'t45-source-cue.json'
    if cue.is_file():payloads['experiments/t45-source-cue.json']=(json.dumps(sanitize(json.loads(cue.read_text())),ensure_ascii=False,separators=(',',':'))+'\n').encode()
    return payloads


def refresh_t45(root, review, stage):
    payloads=t45_payloads(root,review,require_complete=True)
    if not stage.is_dir():raise ValueError('Existing public stage is required; build it first')
    for relative,raw in payloads.items():
        target=stage/relative;target.parent.mkdir(parents=True,exist_ok=True)
        pending=target.with_name(target.name+'.pending');pending.write_bytes(raw);pending.replace(target)
    print(json.dumps({'status':'COMPLETE_T45_PUBLIC_STAGE','files':{name:hashlib.sha256(raw).hexdigest() for name,raw in payloads.items()}}))


def main():
    global SITE_ROOT
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--code-root',type=Path,default=Path.home()/'Documents/GitHub/vlm-psychophysics',
                   help='Existing benchmark code checkout used to validate current technical receipts')
    p.add_argument('--output', type=Path, default=Path('/tmp/cvpr-web-publish'))
    p.add_argument('--site-root', type=Path, default=SITE_ROOT, help='Site source/configuration inputs')
    p.add_argument('--refresh-t45', action='store_true', help='Stage only the sealed COMPLETE T45 figures, table and configuration')
    p.add_argument('--stage', type=Path, help='Direct existing stage directory for --refresh-t45')
    p.add_argument('--refresh-classic',action='store_true',help='Only refresh the completed T1/T2 interactive public snapshot')
    p.add_argument('--refresh-t3',action='store_true',help='Refresh observed T3 scalar trials; mark partial panels explicitly')
    p.add_argument('--refresh-current',action='store_true',help='Refresh the current RGB16 scalar snapshot without publishing')
    p.add_argument('--refresh-previews',action='store_true',help='Build browser derivatives from retained evaluation examples; no inference')
    p.add_argument('--keep-results',action='store_true',help='Reuse the existing scalar snapshot for a presentation-only deployment')
    args=p.parse_args(); root=args.root.resolve(); SITE_ROOT=args.site_root.resolve()
    if args.refresh_t45:
        refresh_t45(root,SITE_ROOT/'review',args.stage.resolve() if args.stage else args.output.resolve()/'stage');return
    if args.refresh_previews:
        refresh_previews(root);return
    if args.refresh_current:
        target=SITE_ROOT/'review/current.json'
        target.write_text(json.dumps(current_results(root,args.code_root),separators=(',',':'),allow_nan=False)+'\n')
        print(json.dumps({'current_results':str(target),'bytes':target.stat().st_size})); return
    if args.refresh_t3:
        refresh_t3(root,SITE_ROOT/'review'); return
    if args.refresh_classic:
        refresh_classic(root,SITE_ROOT/'review'); return
    out=args.output.resolve(); stage=out/'stage'; generated=out/'generated'
    stage.mkdir(parents=True,exist_ok=True); generated.mkdir(parents=True,exist_ok=True)
    routes={}; entries={}; files={}; owned=set(); sanitizations=[]
    previous=json.loads((SITE_ROOT/'upload-receipt.json').read_text())
    remote_only=previous.get('remote_only_media',{}).get('objects',{})
    published_by_sha={r['sha256']:r for r in previous.get('objects',{}).values()}
    review=SITE_ROOT/'review'
    review_media=review/'media'
    runs=root/'runs'; t1=runs/'t1_review_20261012'; conv=runs/'t1_convolution_review'; scene=runs/'scene_review_natural'; lib=runs/'asset_library'
    allowed=(review_media.resolve(),t1.resolve(),conv.resolve(),scene.resolve(),lib.resolve(),(runs/'scene_review_20261012/assets').resolve(),(runs/'meeting_samples_20261007/site/media').resolve(),(runs/'tasks_review_20261012/assets').resolve(),(runs/'site_psychophysics_20261005/assets/generated/ball.png').resolve(),(runs/'showcase_natural_20261004_v1/assets_generated/objects_original.png').resolve())
    def read(path): return json.loads(path.read_text())
    def write(url,data):
        target=stage/url.lstrip('/');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);owned.add(str(target));files[url]=len(data)
    def media(path,url):
        path=path.resolve(strict=False)
        if not any(path.is_relative_to(a) for a in allowed): raise ValueError(f'Outside public media allowlist: {path}')
        if path not in entries:
            if path.is_file():
                digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
                old=published_by_sha.get(digest)
                key=old['key'] if old and old['size']==path.stat().st_size else 'media/'+str(path.relative_to(root))
                entries[path]={'local_path':str(path),'key':key,'size':path.stat().st_size,'sha256':digest,'content_type':mimetypes.guess_type(path.name)[0] or 'application/octet-stream'}
            else:
                old=remote_only.get(str(path))
                if old is None: raise ValueError(f'No local file or verified published object: {path}')
                entries[path]={k:old[k] for k in ('key','size','sha256','etag','source_md5')}
                entries[path].update(local_path=None,remote_existing=True,content_type=mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        key=entries[path]['key']
        if url in routes and routes[url]!=key: raise ValueError(f'Conflicting URL {url}')
        routes[url]=key
    def config(path,url,value=None):
        before=read(path) if value is None else value
        data=(json.dumps(sanitize(before),ensure_ascii=False,separators=(',',':'))+'\n').encode()
        if len(data)<=25*1024*1024:write(url,data)
        else:
            target=generated/(hashlib.sha256(url.encode()).hexdigest()+'.json'); target.write_bytes(data);owned.add(str(target))
            key='config/'+hashlib.sha256(data).hexdigest()+'.json';routes[url]=key
            entries[target]={'local_path':str(target),'key':key,'size':len(data),'sha256':hashlib.sha256(data).hexdigest(),'content_type':'application/json'}
        sanitizations.append(url)
    def asset_record(record,base,prefix):
        for k in MEDIA_FIELDS:
            value=record.get(k)
            if not value or not isinstance(value,str) or value.startswith(('/', 'http:', 'https:')):continue
            candidate=base/value
            if candidate.is_file():media(candidate,urljoin(prefix,value))
            elif k in ('path','preview_path') and base==t1 and (lib/value).is_file():
                media(lib/value,urljoin(prefix,value))
            elif k in ('path','input_path','raw_path','preview_path'):raise FileNotFoundError(candidate)
        if record.get('placement_view'):asset_record(record['placement_view'],base,prefix)
    def pairs(data,base,prefix):
        for row in data.get('pairs',[]):
            for k in ('natural','classic'):
                if isinstance(row.get(k),str):media(base/row[k],urljoin(prefix,row[k]))
    def code(base,prefix,names):
        for name in names:
            text=(base/name).read_text()
            if name=='index.html':
                text=public_navigation(text,prefix)
                text=text.replace('<a href="../#T1">Human reference &amp; protocol differences</a>','<a href="/attributions.html">Source attribution &amp; licenses</a>')
                if base!=review:
                    text=text.replace('</main>','<p class="muted"><a href="/attributions.html">Source attribution &amp; licenses</a></p></main>')
            text=current_library_code(text,prefix,name,review)
            if prefix.startswith('/samples/') and name=='style.css':text+='\n'+PUBLIC_NAV_CSS
            if prefix.startswith('/samples/') and name=='app.js':
                text=text.replace('nav a[href="../${TASK.toLowerCase()}/"]','nav a[href="/experiments/?task=${TASK}"]')
            if base==review and name=='app.js':
                for module in ('classic.js','current.js'):
                    digest=hashlib.sha256((base/module).read_bytes()).hexdigest()[:16]
                    text=text.replace(f"import('./{module}')",f"import('./{module}?v={digest}')")
            if PRIVATE.search(text):raise ValueError(f'Private path in code: {base/name}')
            write(prefix+name,text.encode())
    code(t1/'web','/samples/t1/',('index.html','app.js','renderer.js','calibration.js','style.css'))
    source_data=read(t1/'sources.json')
    catalog=read(lib/'catalog.json')
    source_previews={r.get('sha256'):r.get('preview_path') for r in catalog['backgrounds'] if r.get('sha256') and r.get('preview_path')}
    for row in source_data['sources']:
        preview=row.get('preview_path') or source_previews.get(row.get('master_sha256'))
        if preview:
            row['preview_path']='/samples/library/media/'+preview
        else:
            # Picker-only thumbnail; the renderer still loads its exact 512px input.
            with Image.open(t1/row['input_path']) as image:
                image=image.convert('RGB');image.thumbnail((200,200),Image.Resampling.LANCZOS)
                output=io.BytesIO();image.save(output,format='WEBP',quality=82)
            data=output.getvalue();url='/samples/t1/picker/'+hashlib.sha256(data).hexdigest()[:20]+'.webp'
            write(url,data);row['preview_path']=url
    config(t1/'sources.json','/samples/t1/media/sources.json',source_data)
    for name in ('calibration.json','protocol.json','metrics.json'):config(t1/name,'/samples/t1/media/'+name)
    # Keep every calibration value intact, but do not make two selected sources
    # wait for all 100. Pair photo/generated names before making ten-source shards.
    quantized=read(t1/'quantized_calibration.json')
    rows=sorted(quantized['sources'],key=lambda row:(re.sub(r'^(photo|gpt)_','',row['id']),row['id']))
    index={k:v for k,v in quantized.items() if k!='sources'}; index['sources']=[]
    for start in range(0,len(rows),10):
        group=rows[start:start+10]
        payload={'calibration_version':quantized['calibration_version'],'sources':group}
        data=(json.dumps(sanitize(payload),ensure_ascii=False,separators=(',',':'))+'\n').encode()
        relative='calibration-parts/'+hashlib.sha256(data).hexdigest()[:20]+'.json'
        write('/samples/t1/media/'+relative,data)
        index['sources'].extend({'id':r['id'],'input_sha256':r['input_sha256'],'conditions_path':relative} for r in group)
    config(t1/'quantized_calibration.json','/samples/t1/media/quantized_calibration.json',index)
    for row in read(t1/'sources.json')['sources']:asset_record(row,t1,'/samples/t1/media/')
    for row in read(t1/'metrics.json')['videos']:
        for kind in ('natural','classic'):media(t1/row[kind]['path'],'/samples/t1/media/'+row[kind]['path'])
    code(conv/'web','/samples/t1/convolution/',('index.html','app.js','style.css'))
    for name in ('manifest.json','protocol.json','foreground/manifest.json'):config(conv/name,'/samples/t1/convolution/media/'+name)
    cm=read(conv/'manifest.json')
    for row in cm['sources']:media(conv/row['atlas_path'],'/samples/t1/convolution/media/'+row['atlas_path'])
    for row in cm['videos']:media(conv/row['video_path'],'/samples/t1/convolution/media/'+row['video_path'])
    for row in read(conv/'foreground/manifest.json')['videos']:media(conv/'foreground'/row['path'],'/samples/t1/convolution/media/foreground/'+row['path'])
    catalog=read(lib/'catalog.json')
    library_prefixes=['/samples/library/media/','/samples/t1/convolution/sources/asset_library/']+['/samples/t'+str(n)+'/media/library/' for n in range(2,6)]
    for prefix in library_prefixes:
        config(lib/'catalog.json',prefix+'catalog.json')
        for row in catalog['backgrounds']+catalog['objects']:asset_record(row,lib,prefix)
    code(lib/'web','/samples/library/',('index.html','app.js','style.css'))
    design='\n'.join(line for line in (lib/'design.txt').read_text().splitlines() if not re.search(r'George|Jinwei|meeting|Oct12|03:|04:|07:|Browser acceptance|desktop\.png|Local/remote|coordinator',line,re.I))
    design=PRIVATE.sub('[local implementation path omitted]',design)
    for prefix in library_prefixes:
        write(prefix+'design.txt',design.encode())
        for name in ('review_shared_manifest.json','review_manifest.json'):
            data=read(lib/name);config(lib/name,prefix+name);pairs(data,lib,prefix)
    sm=read(scene/'manifest.json')
    for n in range(2,6):
        prefix=f'/samples/t{n}/';code(scene/'web',prefix,('index.html','app.js','style.css'))
        write(prefix+'media/render.js',(scene/'render.js').read_bytes());config(scene/'manifest.json',prefix+'media/manifest.json')
        pairs(sm,scene,prefix+'media/')
        for row in sm['sources']['backgrounds']+sm['sources']['objects']:asset_record(row,scene,prefix+'media/')
    # Compact experiment specification and this run's results are separate from legacy samples.
    if review.is_dir():
        code(review,'/experiments/',('index.html','app.js','style.css','classic.js','classic.css','current.js','t45-final.js','t45-final.css'))
        config(review/'current.json','/experiments/current.json',current_snapshot(root,read(review/'current.json')) if args.keep_results else current_results(root,args.code_root))
        comparison=root/'runs/reproducibility/natural-t2-spoon-20261030'
        if (comparison/'classic-X-comparison-summary.json').exists():
            for name in ('classic-X-vs-natural-X.csv','classic-X-vs-natural-X.png','classic-X-vs-natural-X.pdf','A-X-paired.csv','A-X-cells.csv','spoon-old-new-paired.csv'):
                write('/experiments/t2-transfer/'+name,(comparison/name).read_bytes())
            pooled=comparison/'pooled-comparison-summary.json'
            if pooled.exists():
                accepted=read(pooled)
                assert accepted['status']=='COMPLETE_POOLED_SAME_X_COMPARISON'
                assert hashlib.sha256((comparison/'classic-X-comparison-summary.json').read_bytes()).hexdigest()==accepted['comparison']['sha256']
                for name in ('classic-X-vs-natural-X-pooled.csv','classic-X-vs-natural-X-pooled.png','classic-X-vs-natural-X-pooled.pdf'):
                    raw=(comparison/name).read_bytes()
                    assert hashlib.sha256(raw).hexdigest()==accepted['outputs'][name]['sha256']
                    write('/experiments/t2-transfer/'+name,raw)
        write('/library/task-cards.css',(review/'task-cards.css').read_bytes())
        previews=read(review/'previews.json')
        config(review/'previews.json','/experiments/previews.json',previews)
        for asset in previews['assets']:
            name=asset['name'];assert re.fullmatch(r'T[1-4](?:-n(?:15|25))?(?:-frame-\d+)?\.(?:mp4|png)',name)
            source=review/'previews'/name
            raw=source.read_bytes();assert hashlib.sha256(raw).hexdigest()==asset['sha256']
            assert len(raw)==asset['bytes'] and len(raw)<25*1024*1024
            write('/experiments/previews/'+name,raw)
        write('/smoke/index.html',b'<!doctype html><meta charset="utf-8"><title>Visual perception</title><script>const t=new URLSearchParams(location.search).get("task");location.replace("/experiments/?task="+(/^T[1-5]$/.test(t)?t:"T1"));</script><a href="/experiments/?task=T1">Open tasks</a>')
        t45_release=t45_payloads(root,review)
        if t45_release:
            for name,raw in t45_release.items():write('/'+name,raw)
        else:config(review/'t45-final.json','/experiments/t45-final.json')
        config(review/'task-cards.json','/experiments/task-cards.json')
        if (review/'astra.json').is_file():
            astra=read(review/'astra.json')
            config(review/'astra.json','/experiments/astra.json')
            for name,identity in astra['figure_assets'].items():
                if not re.fullmatch(r'astra-T[123](?:-direction)?\.(?:png|pdf)',name):raise ValueError('Unexpected Astra public figure')
                raw=(root/'runs/reproducibility/astra-path-only-20261027/figures'/name).read_bytes()
                if len(raw)!=identity['bytes'] or hashlib.sha256(raw).hexdigest()!=identity['sha256']:raise ValueError('Astra figure identity changed')
                write('/experiments/astra/'+name,raw)
        if (review/'t45-source-cue.json').is_file():config(review/'t45-source-cue.json','/experiments/t45-source-cue.json')
        config(review/'plan.json','/experiments/plan.json')
        config(review/'library.json','/experiments/library.json')
        write('/experiments/library-render.js',(review/'library-render.js').read_bytes())
        write('/experiments/figures/library_t4_disks.png',(review/'figures/library_t4_disks.png').read_bytes())
        write('/experiments/figures/t5_psytoolkit_mask.png',(review/'figures/t5_psytoolkit_mask.png').read_bytes())
        write('/library/index.html',(SITE_ROOT/'library.html').read_bytes())
        write('/library/player.js',(review/'library-player.js').read_bytes())
        for clip in read(root/'runs/stimulus-preparation/preparation.json')['T4']['clips']:
            preview=root/'runs/stimulus-preparation'/clip['preview']
            if preview.suffix!='.mp4' or preview.stat().st_size>25*1024*1024:raise ValueError('Unexpected RDM browser preview')
            write('/library/media/'+preview.name,preview.read_bytes())
            poster=review/'previews'/(preview.stem+'.jpg')
            write('/library/media/'+poster.name,poster.read_bytes())
        if (review/'classic.json').exists():
            classic=read(review/'classic.json');classic['T1']['frame_bank']['url']='/experiments/figures/classic_sf4_c1_frames.png'
            config(review/'classic.json','/experiments/classic.json',classic)
            write('/experiments/figures/classic_sf4_c1_frames.png',(review/'figures/classic_sf4_c1_frames.png').read_bytes())
        public_plan=read(review/'plan.json')
        for task in public_plan['tasks']:
            for example in task.get('examples',{}).values():
                for field in ('video','reference','poster'):
                    url=example.get(field)
                    if url:
                        if not url.startswith('/experiments/media/'):raise ValueError(f'Unexpected review media URL: {url}')
                        media(review_media/url.removeprefix('/experiments/media/'),url)
    history=read(runs/'meeting_samples_20261007/site/data.json')
    history['rows']=[{k:v for k,v in row.items() if k not in ('natural','sources')} for row in history['rows'] if row['id'] in HISTORY]
    if len(history['rows'])!=5:raise ValueError('Missing historical reference rows')
    config(runs/'meeting_samples_20261007/site/data.json','/data.json',history)
    for row in history['rows']:media(runs/'meeting_samples_20261007/site'/row['classic'],'/'+row['classic'])
    landing='<!doctype html><html lang="en"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/experiments/?task=overview"><title>Natural video psychophysics</title><a href="/experiments/?task=overview">Open results</a></html>'
    write('/index.html',landing.encode())
    write('/samples/index.html',landing.replace('/experiments/?task=overview','/samples/t1/').encode())
    # License details are visible, not buried in downloaded provenance JSON.
    cards=[]
    for row in catalog['backgrounds']+catalog['objects']:
        author=sanitize(row).get('author','Not specified');author=', '.join(author) if isinstance(author,dict) else str(author)
        lic=row.get('license','Generated for this project');link=row.get('license_url') or ('https://creativecommons.org/licenses/by/2.0/' if row.get('id')=='photo_obj_002' else '')
        transform=row.get('transform',{});transform=json.dumps(transform,ensure_ascii=False) if isinstance(transform,dict) else str(transform)
        cards.append('<article><h2>'+html.escape(row.get('label',row['id']))+'</h2><p>'+html.escape(row['id']+' — '+author)+' · '+('<a href="'+html.escape(link,quote=True)+'">'+html.escape(lic)+'</a>' if link else html.escape(lic))+'</p><p><a href="/samples/library/media/'+html.escape(row.get('raw_path') or row['path'],quote=True)+'">Original</a> · <a href="'+html.escape(row.get('source_url') or '/samples/library/',quote=True)+'">Publisher / provenance</a></p><p>Preparation: '+html.escape(transform or 'See source manifest for preparation metadata.')+'</p>'+( '<p>Prepared derivative retains the source GFDL license. <a href="https://www.gnu.org/licenses/old-licenses/fdl-1.2.html">Full GNU Free Documentation License 1.2</a>.</p>' if 'GFDL' in lic else '')+'</article>')
    write('/attributions.html',('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Source attribution and licenses</title><main style="max-width:900px;margin:2rem auto;font:16px/1.6 system-ui"><a href="/samples/library/">Source library</a><h1>Source attribution and licenses</h1><p>Sources retain their individual licenses. Generated images and programmatic preparation are identified in the manifest.</p>'+''.join(cards)+'</main></html>').encode())
    # Blind annotation is separate; private manifests/records never enter this snapshot.
    for name in ('index.html','app.js','style.css','config.json'):
        write('/label/'+name,(SITE_ROOT/'label'/name).read_bytes())
    # Advanced-mode Worker and its exact URL allowlist remain part of every rebuild.
    write('/_worker.js',(SITE_ROOT/'worker.js').read_bytes())
    write('/asset-routes.json',(json.dumps(routes,sort_keys=True,separators=(',',':'))+'\n').encode())
    media_mounts=['/experiments/media/*','/media/*','/samples/t1/media/*','/samples/t1/convolution/media/*',
                  '/samples/t1/convolution/sources/*','/samples/library/media/*']
    media_mounts += [f'/samples/t{n}/media/*' for n in range(2,6)]
    # JSON/configuration and render code are Pages assets, not R2 Worker calls.
    static_media=sorted(url for url in files if any(url.startswith(mount[:-1]) for mount in media_mounts))
    if len(media_mounts)+len(static_media)>100:raise ValueError('Pages route rule limit exceeded')
    write('/_routes.json',(json.dumps({'version':1,'include':media_mounts,'exclude':static_media},indent=2)+'\n').encode())
    write('/_headers',b'/*\n  X-Content-Type-Options: nosniff\n  Referrer-Policy: strict-origin-when-cross-origin\n  X-Frame-Options: SAMEORIGIN\n  Permissions-Policy: camera=(), microphone=(), geolocation=()\n')
    with (stage/'_headers').open('ab') as headers:
        headers.write(b'/samples/t1/media/calibration-parts/*\n  Cache-Control: public, max-age=31536000, immutable\n/samples/t1/picker/*\n  Cache-Control: public, max-age=31536000, immutable\n')
    files['/_headers']=(stage/'_headers').stat().st_size
    # Version entry scripts so an updated index never runs an older cached loader.
    for url in list(files):
        if url.endswith('/index.html'):
            app_url=url.removesuffix('index.html')+'app.js'
            if app_url in files:
                digest=hashlib.sha256((stage/app_url.lstrip('/')).read_bytes()).hexdigest()[:16]
                page=(stage/url.lstrip('/')).read_text()
                page=re.sub(r'(src=[\"\'](?:\./)?app\.js)(?:\?[^\"\']*)?',lambda m:m[1]+'?v='+digest,page)
                write(url,page.encode())
    # Check literal JS fetch/import targets plus the finite current fetch families.
    js_dependencies=set()
    for url in list(files):
        if not url.endswith('.js') or url=='/_worker.js': continue
        text=(stage/url.lstrip('/')).read_text()
        imports=re.findall(r"\b(?:from\s*|import\s*(?:\(\s*)?)['\"]([^'\"]+)['\"]",text)
        for target in imports:
            if not target.startswith(('./','../','/')) and not urlsplit(target).scheme:
                raise ValueError(f'Bare browser module specifier in {url}: {target}')
        patterns=(r"fetch\(\s*['\"]([^'\"]+)['\"]",r"(?:from|import)\s*['\"]([^'\"]+)['\"]")
        for pattern in patterns:
            for target in re.findall(pattern,text):
                if not urlsplit(target).scheme:js_dependencies.add(unquote(urlsplit(urljoin(url,target)).path))
    js_dependencies.update('/samples/t1/media/'+name for name in ('sources.json','calibration.json','protocol.json','metrics.json','quantized_calibration.json'))
    js_dependencies.update(('/samples/t1/convolution/sources/asset_library/catalog.json','/samples/t1/convolution/media/manifest.json','/samples/t1/convolution/media/foreground/manifest.json'))
    for n in range(2,6):
        js_dependencies.update(f'/samples/t{n}/media/'+name for name in ('manifest.json','library/catalog.json','library/review_shared_manifest.json'))
    missing=sorted(url for url in js_dependencies if url not in routes and url not in files)
    if missing: raise ValueError(f'Missing JavaScript dependencies: {missing}')
    # Only previous builder-owned output files may be retired on a repeated build.
    ledger=out/'build-owned.json'
    if ledger.exists():
        for old in json.loads(ledger.read_text()):
            path=Path(old)
            if old not in owned and path.is_relative_to(out) and path.is_file() and not path.is_symlink():path.unlink()
    ledger.write_text(json.dumps(sorted(owned),indent=2))
    (out/'routes.json').write_text(json.dumps(routes,sort_keys=True,separators=(',',':'))+'\n')
    manifest=sorted(entries.values(),key=lambda x:x['key'])
    (out/'upload-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'asset-inventory.json').write_text(json.dumps([{k:v for k,v in entry.items() if k!='local_path'} for entry in manifest],indent=2)+'\n')
    summary={'static_files':len(files),'static_bytes':sum(files.values()),'r2_objects':len(manifest),'r2_bytes':sum(e['size'] for e in manifest),'r2_url_aliases':len(routes),'current_t1_sources':len(read(t1/'sources.json')['sources']),'library_backgrounds':len(catalog['backgrounds']),'library_objects':len(catalog['objects']),'sanitized_config_urls':len(sanitizations),'stage':str(stage),'validated_js_dependencies':len(js_dependencies)}
    (out/'build-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
