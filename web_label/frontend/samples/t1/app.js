import {LEVELS,DEFAULTS,calibrate,phaseText} from './calibration.js?v=current-library';
import {StimulusRenderer} from './renderer.js?v=current-library';
const $=id=>document.getElementById(id);
const state={sources:[],metrics:[],selected:new Set(),classicEnabled:false,earlierEnabled:false,classicSource:null,params:{...DEFAULTS},columns:2,layout:'fit',view:'backgrounds',family:'all',mode:'processed',axis:'sf',axisValues:new Set(LEVELS.sf),custom:[],panels:[],frame:0,maxFrames:179,playing:false,loop:false,start:0,offset:0,ready:false,buildToken:0,refreshes:0};
let renderer,raf=0,drawPending=false,lastPaint=0,refreshIntervals=[],earlierColumn;
const calibrationLoads=new Map();
const pickerObserver=new IntersectionObserver(entries=>{for(const entry of entries){if(!entry.isIntersecting)continue;const image=entry.target;image.src=image.dataset.preview;pickerObserver.unobserve(image);}},{rootMargin:'120px 0px'});
async function loadCalibration(source){}
let drawStats={timestamps:[],frames:[],visiblePanels:[]};
function runtimeStats(){const intervals=drawStats.timestamps.slice(1).map((t,i)=>t-drawStats.timestamps[i]),mean=intervals.length?intervals.reduce((a,b)=>a+b,0)/intervals.length:null;return {reference_fps:60,draw_callbacks:drawStats.timestamps.length,mean_draw_interval_ms:mean,estimated_draw_callbacks_per_second:mean?1000/mean:null,skipped_reference_indices:drawStats.frames.slice(1).reduce((sum,f,i)=>sum+Math.max(0,f-drawStats.frames[i]-1),0),max_visible_panels:Math.max(0,...drawStats.visiblePanels),note:"Browser draw callbacks; not calibrated display refresh or model input."};}
const labels={sf:'SF',m:'Stripe strength',tf:'Hz',duration:'Exposure'};
function conditionText(p){return `${p.sf} cyc/ap · Stripe strength ${p.m} · ${p.tf} Hz · ${p.duration} s · ${p.direction==='LEFT'?'←':'→'}`;}
function node(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!==undefined)e.textContent=text;return e;}
function asset(path,source){if(!path)return '';if(/^(https?:)?\/\//.test(path)||path.startsWith('/'))return new URL(path,location.href).href;if(source?.asset_base){const base=asset(source.asset_base.replace(/\/?$/,'/'));return new URL(path,base).href;}return new URL('media/'+path.replace(/^\/+/,''),location.href).href;}
function provenanceFamily(s){return [s.provenance,s.family,s.kind].find(v=>['photo','gpt'].includes(v))||(s.id.startsWith('gpt_')?'gpt':s.id.startsWith('photo_')?'photo':null);}
function sourceOrder(a,b){return String(a.pair_id||a.semantic_id||a.category||a.id).localeCompare(String(b.pair_id||b.semantic_id||b.category||b.id))||a.id.localeCompare(b.id);}
function sourceFamily(s){return s.family==='gpt'?'GPT':'Photo';}
function layoutPanels(){
 const fit=state.layout==='fit',stage=$('stage');document.body.classList.toggle('fit-window',fit);document.body.classList.toggle('floating-playback',!fit&&(state.panels.length>2||(innerWidth<650&&state.panels.length>1)));$('columns').disabled=fit;
 stage.classList.remove('dense');stage.style.setProperty('--columns',state.columns);
 for(const key of ['--fit-columns','--fit-side','--fit-height','--fit-gap'])stage.style.removeProperty(key);
 if(fit&&state.panels.length){
  // Use document coordinates so scrolling down to configuration cannot enlarge the comparison.
  const top=$('display').getBoundingClientRect().top+scrollY,width=stage.getBoundingClientRect().width,height=Math.max(220,Math.min(470,innerHeight-top-document.querySelector('.playback').getBoundingClientRect().height-28));
  const gap=20,columns=width<620?1:Math.max(1,stage.querySelectorAll('.display-column').length),rows=Math.max(1,...[...stage.querySelectorAll('.display-column')].map(c=>c.querySelectorAll('figure').length)),side=Math.max(140,Math.min(400,(width-gap*(columns-1))/columns,(height-48)*1.5));const grid={columns,rows,side};
  stage.style.setProperty('--fit-columns',columns);stage.style.setProperty('--fit-side',Math.floor(side)+'px');stage.style.setProperty('--fit-height','auto');stage.style.setProperty('--fit-gap',gap+'px');
  stage.dataset.columns=grid.columns;stage.dataset.rows=grid.rows;stage.dataset.panelPixels=Math.max(1,Math.floor(grid.side));
 }else{delete stage.dataset.columns;delete stage.dataset.rows;delete stage.dataset.panelPixels;}
 requestDraw();
}
function selectedSources(){return state.sources.filter(s=>state.selected.has(s.id)&&(state.family==='all'||state.family==='classic'||s.family===state.family));}
function conditions(){if(state.view==='backgrounds')return [{...state.params}];if(state.axis==='custom')return state.custom.map(p=>({...p}));return LEVELS[state.axis].filter(v=>state.axisValues.has(v)).map(v=>({...state.params,[state.axis]:v}));}
function modes(){if(state.family==='classic')return [];return {original:['original'],processed:['processed'],classic:[],paired:['processed'],all:['original','processed']}[state.mode];}
function classicReference(){return state.sources.find(s=>s.id===state.classicSource)||selectedSources()[0]||state.sources[0];}
function paramsFromControls(){for(const key of ['sf','m','tf','duration','phase'])state.params[key]=Number($(key).value);state.params.direction=$('direction').value;}
function showError(error){$('load-status').textContent=error.message||String(error);$('load-status').classList.add('error');}
function setFrame(frame){state.frame=Math.max(0,Math.min(state.maxFrames,Math.floor(frame)));$('seek').value=state.frame;$('time').textContent=`${(state.frame/60).toFixed(3)} / ${((state.maxFrames+1)/60).toFixed(3)} s`;for(const p of state.panels){p.slot.dataset.referenceFrame=state.frame;p.exposure.textContent=p.mode==='original'?'':state.frame>=Math.round(p.params.duration*60)?'Exposure ended':'';}requestDraw();}
function requestDraw(){if(drawPending)return;drawPending=true;requestAnimationFrame(timestamp=>{drawPending=false;if(renderer){const drawn=renderer.draw(state.panels,state.frame);if(state.playing){drawStats.timestamps.push(timestamp);drawStats.frames.push(state.frame);drawStats.visiblePanels.push(drawn);}}});}
function pause(){state.playing=false;cancelAnimationFrame(raf);$('play').textContent='Play current';$('play').setAttribute('aria-pressed','false');requestDraw();}
function tick(now){if(!state.playing)return;const elapsed=state.offset+(now-state.start)/1000;let frame=Math.floor(elapsed*60);if(frame>=state.maxFrames){if(state.loop){const rest=elapsed%(state.maxFrames/60);state.offset=rest;state.start=now;frame=Math.floor(rest*60);}else{setFrame(state.maxFrames);pause();return;}}if(lastPaint)refreshIntervals.push(now-lastPaint);lastPaint=now;if(refreshIntervals.length>120)refreshIntervals.shift();setFrame(frame);raf=requestAnimationFrame(tick);}
function play(){if(!state.panels.length||!state.ready)return;if(state.frame>=state.maxFrames)setFrame(0);state.offset=state.frame/60;state.start=performance.now();lastPaint=0;drawStats={timestamps:[],frames:[],visiblePanels:[]};state.playing=true;$('play').textContent='Pause';$('play').setAttribute('aria-pressed','true');raf=requestAnimationFrame(tick);}
function restoreParams(){for(const [key,value] of Object.entries(state.params))$(key).value=String(value);updateSliders();}
function updateSliders(){for(const key of ['sf','m']){$(key+'-slider').value=LEVELS[key].indexOf(state.params[key]);$(key+'-slider').setAttribute('aria-valuetext',String(state.params[key])+(key==='sf'?' cycles per aperture':' stripe strength'));}}
function populateInputs(){for(const key of ['sf','m','tf','duration']){for(const value of LEVELS[key]){const o=node('option','',String(value));o.value=String(value);$(key).append(o);}}restoreParams();}
function axisControls(){
 $('condition-controls').hidden=state.view!=='conditions';const custom=state.axis==='custom';$('axis-values').hidden=custom;$('custom-conditions').hidden=!custom;$('add-condition').hidden=!custom;$('clear-conditions').hidden=!custom;$('axis-values').replaceChildren();
 if(!custom){for(const value of LEVELS[state.axis]){const label=node('label'),input=node('input');input.type='checkbox';input.checked=state.axisValues.has(value);input.value=value;input.addEventListener('change',()=>{input.checked?state.axisValues.add(value):state.axisValues.delete(value);rebuild();});label.append(input,document.createTextNode(String(value)));$('axis-values').append(label);}}
 $('custom-conditions').replaceChildren();state.custom.forEach((p,i)=>{const chip=node('div','custom-item',conditionText(p)+' · φ '+phaseText(p)),remove=node('button','', '×');remove.setAttribute('aria-label',`Remove condition ${i+1}`);remove.addEventListener('click',()=>{state.custom.splice(i,1);axisControls();rebuild();});chip.append(remove);$('custom-conditions').append(chip);});
}
function buildSourcePicker(){
 pickerObserver.disconnect();$('source-picker').replaceChildren();
 $('classic-reference').replaceChildren();for(const s of state.sources){const option=node('option','',`${s.label} · ${sourceFamily(s)}`);option.value=s.id;$('classic-reference').append(option);}
 for(const family of ['photo','gpt']){
  const entries=state.sources.filter(s=>s.family===family).sort(sourceOrder);
  const group=node('section','source-family'),heading=node('div','source-family-heading'),title=node('h3','',`${family==='photo'?'Photographic':'GPT-generated'} · ${entries.length}`);group.dataset.family=family;title.dataset.familyCount=family;heading.append(title);
  for(const [name,choose] of [['All',()=>true],['Half',(_,i)=>i%2===0],['Clear',()=>false]]){const b=node('button','',name);b.setAttribute('aria-label',`${name} ${family==='photo'?'Photographic':'GPT-generated'} sources`);b.onclick=()=>{entries.forEach((s,i)=>choose(s,i)?state.selected.add(s.id):state.selected.delete(s.id));state.family='all';$('family').value='all';rebuild();};heading.append(b);}
  const grid=node('div','source-family-grid');
  for(const s of entries){const label=node('label','source-choice'),input=node('input'),im=node('img'),caption=node('span','',s.label);input.type='checkbox';input.value=s.id;input.dataset.sourceId=s.id;input.checked=state.selected.has(s.id);input.setAttribute('aria-label',`${sourceFamily(s)}: ${s.label}`);im.dataset.preview=s.preview_path?asset(s.preview_path.startsWith('previews/')?'sources/library/'+s.preview_path:s.preview_path):s.inputURL;im.alt='';im.decoding='async';pickerObserver.observe(im);label.append(input,im,caption);label.dataset.sourceId=s.id;input.addEventListener('change',()=>{input.checked?state.selected.add(s.id):state.selected.delete(s.id);rebuild();});grid.append(label);}
  group.append(heading,grid);$('source-picker').append(group);
 }
 $('classic-enabled').checked=state.classicEnabled;$('classic-enabled').onchange=e=>{state.classicEnabled=e.target.checked;rebuild();};
 $('select-all').textContent=`All ${state.sources.length}`;
}
function updateSourceChecks(){for(const family of ['photo','gpt']){const entries=state.sources.filter(s=>s.family===family),heading=$('source-picker').querySelector(`[data-family-count="${family}"]`);if(heading)heading.textContent=`${family==='photo'?'Photographic':'GPT-generated'} · ${entries.filter(s=>state.selected.has(s.id)).length} / ${entries.length} selected`;}for(const box of $('source-picker').querySelectorAll('input[data-source-id]')){box.checked=state.selected.has(box.value);const s=state.sources.find(s=>s.id===box.value);box.closest('label').classList.toggle('filtered',state.family==='classic'||state.mode==='classic'||(state.family!=='all'&&s.family!==state.family));}$('classic-enabled').checked=state.classicEnabled;$('classic-controls').hidden=!state.classicEnabled;$('classic-reference').value=classicReference().id;}
function sourceTable(){
 $('source-table').replaceChildren();for(const s of state.sources){const tr=node('tr');const n=node('td','',s.label+' · '+sourceFamily(s));if(s.scene_family||s.category)n.append(node('small','',[s.scene_family,s.category].filter(Boolean).filter((x,i,a)=>a.indexOf(x)===i).join(' · ')));const prov=node('td','',s.source_provenance_note||s.source_provenance_detail||s.source_kind?.replaceAll('_',' ')||(s.family==='gpt'?'Text-only image generation':'Photographic source'));if(s.family==='gpt')prov.append(node('small','',s.model||'Tool-managed model; exact version not exposed.'));
 const size=node('td','',s.source_dimensions.join(' × ')+' px');if(s.master_dimensions?.join()!==s.source_dimensions?.join())size.append(node('small','',`Master ${s.master_dimensions.join(' × ')} px`));
 const transform=node('td','',typeof s.transform==='string'?s.transform:s.transform?.operation||'See source manifest');transform.append(node('small','', '512 × 512 stimulus input; processor input unmeasured'));
 const links=node('td');for(const [name,path] of [['Original',s.raw_path||s.original_path],['Master',s.master_path],['512 input',s.input_path]]){if(!path)continue;const a=node('a','',name);a.href=asset(path,path===s.input_path?null:s);a.target='_blank';a.rel='noopener';links.append(a,document.createTextNode(' · '));}if(s.url||s.source_url){const a=node('a','',s.license||'Source');a.href=s.url||s.source_url;a.target='_blank';a.rel='noopener';links.append(a);}else links.append(document.createTextNode('Generated'));tr.append(n,prov,size,transform,links);$('source-table').append(tr);
 }
}
function measured(source,params){return state.metrics.find(v=>{const p=v.parameters;return v.id===source.id&&p.sf_cycles_per_aperture===params.sf&&p.multiplier_depth===params.m&&p.tf_hz===params.tf&&p.duration_s===params.duration&&p.direction===params.direction&&Math.abs(p.phase_rad-params.phase)<1e-12;});}
function measurements(){$('measurements').replaceChildren();}
function saveURL(){const q=new URLSearchParams();q.set('sources',[...state.selected].join(','));q.set('family',state.family);q.set('show',state.mode);q.set('classic',state.classicEnabled?'1':'0');q.set('earlier',state.earlierEnabled?'1':'0');q.set('classic_source',classicReference().id);q.set('view',state.view);if(state.columns!==2)q.set('columns',state.columns);if(state.layout!=='fit')q.set('layout',state.layout);for(const [key,v]of Object.entries(state.params))if(v!==DEFAULTS[key])q.set(key,v);if(state.view==='conditions'){q.set('axis',state.axis);if(state.axis==='custom')q.set('conditions',JSON.stringify(state.custom));else q.set('values',[...state.axisValues].join(','));}history.replaceState(null,'',location.pathname+'?'+q.toString()+location.hash);}
async function rebuild(){
 const token=++state.buildToken;pause();$('stage').style.setProperty('--columns',state.columns);state.ready=false;setFrame(0);$('play').disabled=true;$('restart').disabled=true;$('seek').disabled=true;$('stage').replaceChildren();state.panels=[];$('load-status').classList.remove('error');
 const sources=selectedSources(),cs=conditions(),types=modes(),specs=[];
 for(const s of sources)for(const p of cs)for(const mode of types)specs.push({source:s,params:p,mode});
 if(state.classicEnabled)for(const p of cs)specs.push({source:classicReference(),params:p,mode:'classic'});
 const loadSources=[...new Map(specs.map(p=>[p.source.id,p.source])).values()];state.maxFrames=Math.max(1,...cs.map(p=>Math.round(p.duration*60)))-1;$('seek').max=state.maxFrames;setFrame(0);$('empty').hidden=specs.length>0;$('empty').textContent=cs.length?'No stimuli selected. Choose Classic or backgrounds below.':'No conditions selected. Choose values below.';
 $('selection-count').textContent=`${types.length?sources.length:0} backgrounds${state.classicEnabled?' + 1 classic':''} · ${cs.length} condition${cs.length===1?'':'s'} · ${specs.length} panel${specs.length===1?'':'s'}`;
 $('parameter-summary').textContent=state.view==='backgrounds'?conditionText(state.params):`${cs.length} conditions · same reference clock`;$('stage').classList.toggle('single',specs.length===1);$('display-mode').disabled=state.family==='classic';updateSourceChecks();saveURL();
 try{
  $("load-status").textContent=loadSources.length?"Loading selected inputs…":"";
  // Fetch only the selected calibration shards, concurrently with image inputs.
  const inputLoads=Promise.allSettled(loadSources.map(s=>renderer.load(s)));
  await Promise.all(loadSources.map(loadCalibration));if(token!==state.buildToken)return;
  const masterLoads=[],columns=new Map();
  for(const [id,title] of [['photo','Photographic scenes'],['gpt','GPT-generated scenes'],['classic','Classic grating']]){
   if(!specs.some(p=>(p.mode==='classic'?'classic':p.source.family)===id))continue;
   const col=node('section','display-column');col.dataset.family=id;col.append(node('h2','column-title',title));columns.set(id,col);$('stage').append(col);
  }

  for(const {source:s,params:p,mode} of specs){const fig=node('figure','stimulus'),caption=node('figcaption'),a=node('a','',mode==='classic'?'Classic':s.label);a.href=asset(s.master_path,s);a.target='_blank';a.rel='noopener';const name={original:'Source master',processed:'Moving stripes on image',classic:'Classic'}[mode],dimensions=mode==='original'?s.master_dimensions:[768,512];caption.append(a,node('span','descriptor',mode==='classic'?'Fixed mean Y .31':`${name} · ${dimensions.join(' × ')}`));const slot=node('div','image-slot not-ready');slot.setAttribute('role','img');slot.setAttribute('aria-label',`${name}${mode==='classic'?'':''} ${s.label}, ${sourceFamily(s)}, ${dimensions.join(' by ')} pixels${mode==='original'?'':', '+conditionText(p)}`);slot.dataset.source=s.id;slot.dataset.mode=mode;slot.dataset.sf=p.sf;slot.dataset.tf=p.tf;slot.dataset.duration=p.duration;
   if(mode==='original'){const im=node('img','source-master');im.alt='';im.width=dimensions[0];im.height=dimensions[1];im.src=asset(s.master_path,s);slot.append(im);masterLoads.push(im.decode().then(()=>{if(im.naturalWidth!==dimensions[0]||im.naturalHeight!==dimensions[1])throw new Error(`${s.label}: source master dimensions do not match the manifest.`);slot.classList.remove('not-ready');}).catch(error=>{slot.classList.add('failed');throw error;}));}
   const note=node('div','panel-note'),exposure=node('span','exposure-state');note.append(node('span','',state.view==='conditions'?conditionText(p):''),exposure);fig.title=`${s.label} · ${sourceFamily(s)} · ${name} · ${dimensions.join(' × ')} px${mode==='original'?'':' · '+conditionText(p)}`;fig.append(caption,slot,note);columns.get(mode==='classic'?'classic':s.family).append(fig);state.panels.push({source:s,params:p,mode,slot,exposure,calibration:calibrate(s,p)});
  }
  if(state.earlierEnabled&&!earlierColumn){
   const col=node('section','display-column earlier-column'),fig=node('figure','stimulus'),v=node('video');col.dataset.family='earlier';col.append(node('h2','column-title','Earlier classic'));
   v.src='/media/T1_bark_sf8_LEFT_classic.mp4';v.controls=true;v.muted=true;v.playsInline=true;v.preload='metadata';v.setAttribute('aria-label','Earlier classical sample, independent playback');
   fig.append(node('figcaption','','7 Oct · fixed reference'),v,node('p','historical-parameters','8 cyc/ap · RMS 0.09 · 3 s · ←'));col.append(fig);earlierColumn=col;
  }
  if(state.earlierEnabled)$('stage').append(earlierColumn);
  layoutPanels();
  $('load-status').textContent=loadSources.length?'Loading selected inputs…':'';
  const [loaded,masters]=await Promise.all([inputLoads,Promise.allSettled(masterLoads)]);if(token!==state.buildToken)return;
  loaded.forEach((r,i)=>{for(const p of state.panels.filter(p=>p.source.id===loadSources[i].id&&p.mode!=='original')){p.slot.classList.remove('not-ready');if(r.status==='rejected'){p.slot.classList.add('failed','not-ready');}}});const failures=[...loaded,...masters].filter(r=>r.status==='rejected');if(failures.length)throw failures[0].reason;
  state.ready=state.panels.length>0;$('load-status').textContent='';$('play').disabled=!state.ready;$('restart').disabled=!state.ready;$('seek').disabled=!state.ready;measurements();setFrame(0);layoutPanels();
 }catch(error){showError(error);requestDraw();}
}
function validateParams(p){for(const k of ['sf','m','tf','duration'])if(!LEVELS[k].includes(Number(p[k])))return false;return ['LEFT','RIGHT'].includes(p.direction)&&Array.from({length:24},(_,i)=>Math.PI/24+i*Math.PI/12).some(x=>Math.abs(x-Number(p.phase))<1e-12);}
function restoreURL(){const q=new URLSearchParams(location.search);state.earlierEnabled=false;$('earlier-enabled').checked=state.earlierEnabled;if(q.get('layout')==='large')state.layout='large';$('layout').value=state.layout;if([1,2,3,4,5].includes(Number(q.get('columns'))))state.columns=Number(q.get('columns'));$('columns').value=state.columns;if(q.has('sources'))state.selected=new Set(q.get('sources').split(',').filter(id=>state.sources.some(s=>s.id===id)));else state.selected=new Set(['photo_forest','gpt_forest'].filter(id=>state.sources.some(s=>s.id===id)));if(!q.has('sources')&&!state.selected.size)state.selected=new Set(state.sources.slice(0,1).map(s=>s.id));if(['all','photo','gpt','classic'].includes(q.get('family')))state.family=q.get('family');if(['original','processed','classic','paired','all'].includes(q.get('show')))state.mode=q.get('show');state.classicEnabled=q.has('classic')?q.get('classic')==='1':true;if(state.mode==='paired')state.mode='processed';state.classicSource=state.sources.some(s=>s.id===q.get('classic_source'))?q.get('classic_source'):(selectedSources()[0]||state.sources[0]).id;if(q.get('view')==='conditions')state.view='conditions';const p={...state.params};for(const key of ['sf','m','tf','duration','phase'])if(q.has(key))p[key]=Number(q.get(key));if(q.has('direction'))p.direction=q.get('direction');if(validateParams(p))state.params=p;if([...Object.keys(LEVELS),'custom'].includes(q.get('axis')))state.axis=q.get('axis');state.axisValues=new Set(LEVELS[state.axis]||[]);if(q.has('values')&&state.axis!=='custom')state.axisValues=new Set(q.get('values').split(',').map(Number).filter(v=>LEVELS[state.axis].includes(v)));try{if(q.has('conditions'))state.custom=JSON.parse(q.get('conditions')).filter(validateParams);}catch{}$('family').value=state.family;$('display-mode').value=state.mode;document.querySelector(`input[name=view][value=${state.view}]`).checked=true;$('axis').value=state.axis;restoreParams();}
function bind(){
 $('earlier-enabled').onchange=e=>{state.earlierEnabled=e.target.checked;rebuild();};
 $('layout').addEventListener('change',e=>{state.layout=e.target.value;saveURL();layoutPanels();});
 $('columns').addEventListener('change',e=>{state.columns=Number(e.target.value);saveURL();layoutPanels();});
 for(const key of ['sf','m'])$(key+'-slider').addEventListener('input',e=>{state.params[key]=LEVELS[key][Number(e.target.value)];$(key).value=String(state.params[key]);updateSliders();rebuild();});
 for(const key of ['sf','m','tf','duration','direction','phase'])$(key).addEventListener('change',()=>{paramsFromControls();updateSliders();rebuild();});
 document.querySelectorAll('input[name=view]').forEach(e=>e.addEventListener('change',()=>{state.view=e.value;axisControls();rebuild();}));
 $('family').addEventListener('change',e=>{state.family=e.target.value;if(state.family==='classic')state.classicEnabled=true;rebuild();});$('display-mode').addEventListener('change',e=>{state.mode=e.target.value;if(state.mode==='classic')state.classicEnabled=true;rebuild();});$('classic-reference').addEventListener('change',e=>{state.classicSource=e.target.value;rebuild();});$('axis').addEventListener('change',e=>{state.axis=e.target.value;if(state.axis==='custom'&&!state.custom.length)state.custom.push({...state.params});state.axisValues=new Set(LEVELS[state.axis]||[]);axisControls();rebuild();});
 $('add-condition').addEventListener('click',()=>{const p={...state.params};if(!state.custom.some(c=>JSON.stringify(c)===JSON.stringify(p)))state.custom.push(p);axisControls();rebuild();});$('clear-conditions').addEventListener('click',()=>{state.custom=[];axisControls();rebuild();});
 for(const [id,family]of [['select-all','all'],['select-photo','photo'],['select-gpt','gpt'],['select-none','none']])$(id).addEventListener('click',()=>{state.selected=new Set(state.sources.filter(s=>family==='all'||s.family===family).map(s=>s.id));if(family==='none')state.classicEnabled=false;if(family!=='none'&&state.family!=='classic'){state.family='all';$('family').value='all';}rebuild();});
 $('play').addEventListener('click',()=>state.playing?pause():play());$('restart').addEventListener('click',()=>{pause();setFrame(0);play();});$('loop').addEventListener('change',e=>state.loop=e.target.checked);$('seek').addEventListener('input',()=>{pause();setFrame(Number($('seek').value));});
 addEventListener('resize',layoutPanels);addEventListener('scroll',requestDraw,{passive:true});const layoutObserver=new ResizeObserver(layoutPanels);for(const element of [document.querySelector('header'),document.querySelector('.project-nav'),document.querySelector('.playback')])layoutObserver.observe(element);document.addEventListener('visibilitychange',()=>{if(document.hidden)pause();});document.addEventListener('keydown',e=>{if(['INPUT','SELECT','TEXTAREA','BUTTON','A','SUMMARY'].includes(document.activeElement.tagName))return;if(e.code==='Space'){e.preventDefault();state.playing?pause():play();}else if(e.code==='ArrowLeft'||e.code==='ArrowRight'){e.preventDefault();pause();setFrame(state.frame+(e.code==='ArrowRight'?1:-1));}});
 document.querySelectorAll('details').forEach(d=>d.addEventListener('toggle',()=>{requestDraw();const average=refreshIntervals.length?refreshIntervals.reduce((a,b)=>a+b,0)/refreshIntervals.length:0;$('runtime-description').textContent=average?`Observed browser animation callbacks in the last playback: about ${(1000/average).toFixed(1)} per second. This is a browser observation, not stimulus delivery calibration.`:'Browser refresh is not yet observed; use Play all to inspect motion.';}));
}
async function boot(){try{
 renderer=new StimulusRenderer($('renderer'));const response=await fetch(asset('sources.json'));if(!response.ok)throw Error('Sources unavailable');const data=await response.json();
 state.sources=data.sources.map(s=>({...s,family:provenanceFamily(s),inputURL:asset(s.input_path)}));state.metrics=[];
 populateInputs();restoreURL();buildSourcePicker();sourceTable();axisControls();bind();await rebuild();
}catch(error){showError(error);$('empty').hidden=false;$('empty').textContent='The selected inputs could not be loaded.';}}
boot();
