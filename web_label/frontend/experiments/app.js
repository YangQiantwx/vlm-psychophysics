const query=new URLSearchParams(location.search);
const requested=query.get('task')||'overview';
const task=['overview','precision','T1','T2','T3','T4','T5'].includes(requested)?requested:'overview';
if(location.pathname.startsWith('/smoke')){
 location.replace('/experiments/?task='+task);
}else if(['T4','T5'].includes(task)&&query.get('protocol')!=='historical'){
 await import('./t45-final.js');
}else if(['overview','precision','T1','T2','T3','T4'].includes(task)){
 await import('./current.js');
}else{
 const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 const base=location.pathname.startsWith('/experiments')?'/experiments/':new URL('.',import.meta.url).pathname;
 document.body.classList.add('classic-dashboard');
 const stylesheet=document.createElement('link');stylesheet.rel='stylesheet';stylesheet.href=new URL('classic.css',import.meta.url);document.head.append(stylesheet);
 try{
  const response=await fetch(base+'plan.json');if(!response.ok)throw Error('Protocol unavailable');
  const plan=await response.json(),t=plan.tasks.find(x=>x.id===task);
  const names=[['T1','Motion'],['T2','Brightness'],['T3','Orientation'],['T4','Global motion'],['T5','Detection']];
  const title=names.find(x=>x[0]===task)[1];document.title=task+' · '+title+' · Visual perception';
  const categories=[['classic','Classic'],['photo','Photographic'],['gpt','GPT-generated']];
  const fixed=t.display.fixed,fields=[...fixed,...t.display.sweeps.map(([label,values,unit])=>[label,values.join(' · '),unit])];
  document.querySelector('main').innerHTML=`<nav class="task-navigation" aria-label="Tasks"><a class="site-brand" href="/experiments/?task=T1">Visual perception</a><div class="task-links">${names.map(([id,label])=>`<a href="/experiments/?task=${id}" ${task===id?'aria-current="page"':''}><b>${id}</b><span>${label}</span></a>`).join('')}<a class="library-link" href="/library/">Library</a></div></nav><header class="page-heading"><h1><span>${task}</span>${title}</h1></header><section class="library-launch"><a href="/samples/${task.toLowerCase()}/">Open ${task} examples in Library →</a></section><details class="protocol-box"><summary>Protocol & prompt</summary><dl class="protocol-grid">${fields.map(([label,value,unit])=>`<div><dt>${esc(label)}</dt><dd>${esc(value)} ${esc(unit||'')}</dd></div>`).join('')}</dl><details class="prompt-details"><summary>Long prompt</summary><blockquote>${esc(t.prompt)}</blockquote></details></details><footer class="page-footer"><a href="/samples/${task.toLowerCase()}/">Stimulus explorer</a><a href="/samples/library/">Source assets</a><details class="method-details"><summary>References</summary><p>${t.references.map(x=>`<a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.label)}</a>`).join(' · ')}</p></details><a href="/attributions.html">Source credits</a></footer>`;
 }catch(error){document.querySelector('main').innerHTML='<h1>Visual perception</h1><p>The page could not be loaded. Please reload.</p>';console.error(error);}
}
