import {SOURCES,MODELS,HARDWARE,DEFAULT_STATE,derived,fmt,compact,gb,sanitizeState,roofline,layerKind} from './core.mjs';
import {buildScene} from './diagrams.mjs';

const $=q=>document.querySelector(q), $$=q=>[...document.querySelectorAll(q)];
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const svgIcon=(paths)=>`<svg viewBox="0 0 24 24" aria-hidden="true">${paths}</svg>`;
const ICONS={
 model:svgIcon('<rect x="8" y="2" width="8" height="5" rx="1"/><rect x="2" y="16" width="8" height="5" rx="1"/><rect x="14" y="16" width="8" height="5" rx="1"/><path d="M12 7v5H6v4m6-4h6v4"/>'),
 training:svgIcon('<path d="M3 6h18M3 12h18M3 18h18"/><rect x="6" y="3" width="5" height="6" rx="1"/><rect x="13" y="9" width="5" height="6" rx="1"/><rect x="8" y="15" width="5" height="6" rx="1"/>'),
 serving:svgIcon('<path d="m4 5 6 7-6 7m7-14 6 7-6 7M20 4v16"/>'),
 hardware:svgIcon('<rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M8 1v4m8-4v4M8 19v4m8-4v4M1 8h4m-4 8h4m14-8h4m-4 8h4"/>'),
 network:svgIcon('<rect x="8" y="2" width="8" height="5" rx="1"/><path d="M12 7v6M4 13h16M4 13v4m8-4v4m8-4v4"/><rect x="1" y="17" width="6" height="5" rx="1"/><rect x="9" y="17" width="6" height="5" rx="1"/><rect x="17" y="17" width="6" height="5" rx="1"/>'),
 sun:svgIcon('<circle cx="12" cy="12" r="4"/><path d="M12 1v2m0 18v2M1 12h2m18 0h2M4 4l2 2m12 12 2 2M4 20l2-2M18 6l2-2"/>'),
 moon:svgIcon('<path d="M20 15A9 9 0 0 1 9 4a9 9 0 1 0 11 11Z"/>'),
 expand:svgIcon('<path d="M3 9V3h6m6 0h6v6M3 15v6h6m6 0h6v-6"/>'),
 export:svgIcon('<path d="M12 3v12m-5-5 5 5 5-5M4 15v6h16v-6"/>')
};
const VIEWS=[
 {id:'model',label:'Model',eyebrow:'01 / THE COMPUTATIONAL GRAPH',title:'Three axes of information flow.',subtitle:'Sequence through KDA and MLA. Depth through AttnRes. Channels through latent experts.',canvas:'Model schematic',badge:'DOCUMENTED TOPOLOGY'},
 {id:'training',label:'Training',eyebrow:'02 / STATE, GRADIENTS & EXECUTION',title:'Training, co-designed with the model.',subtitle:'Explore the reported pretraining stack and the state lifecycles of million-token agentic RL.',canvas:'Training anatomy',badge:'REPORTED MECHANISMS'},
 {id:'serving',label:'Serving',eyebrow:'03 / THE LIFE OF A REQUEST',title:'A prefix is more than a KV cache.',subtitle:'Growing MLA history and fixed KDA state must agree on the same continuation boundary.',canvas:'Serving lifecycle',badge:'LOGICAL PIPELINE'},
 {id:'hardware',label:'Hardware',eyebrow:'04 / THE PHYSICAL SUBSTRATE',title:'A node is more than eight GPUs.',subtitle:'HBM, Tensor Cores, NVLink and PCIe impose different limits on the same computation.',canvas:'Inside the accelerator node',badge:'DOCUMENTED HARDWARE'},
 {id:'network',label:'Network',eyebrow:'05 / TOPOLOGY & COLLECTIVES',title:'Move tensors. Not just packets.',subtitle:'Price latent expert vectors, context-parallel fragments and complete cache handoffs separately.',canvas:'Compute-fabric schematic',badge:'ANALYTIC SCENARIO'}
];
let initial={};try{initial=Object.fromEntries(new URLSearchParams(location.hash.slice(1)));if(!initial.theme)initial.theme=localStorage.getItem('stratum-theme')||'dark';}catch{}
let state=sanitizeState({...DEFAULT_STATE,...initial}),scene=null,toastTimer;
const sourceById=id=>SOURCES.find(s=>s.id===id);
function persist(){const p=new URLSearchParams();for(const[k,v]of Object.entries(state))if(!['selected','zoom'].includes(k))p.set(k,v);try{history.replaceState(null,'','#'+p.toString());localStorage.setItem('stratum-theme',state.theme);}catch{}}
function notify(text){$('#toast').textContent=text;$('#toast').classList.add('show');clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').classList.remove('show'),2800);}
function setState(patch,{resetZoom=false}={}){
 Object.assign(state,patch);
 if(patch.layer)state.selected=null;
 if(patch.view){state.selected=null;state.zoom=1;}
 if(resetZoom)state.zoom=1;
 render();persist();
}
function optionHTML(values,current){return values.map(v=>{const value=Array.isArray(v)?v[0]:v,label=Array.isArray(v)?v[1]:v;return `<option value="${esc(value)}" ${String(value)===String(current)?'selected':''}>${esc(label)}</option>`;}).join('');}
function selectControl(label,key,values){return `<label class="control"><span class="control-label">${esc(label)}</span><select data-key="${key}" aria-label="${esc(label)}">${optionHTML(values,state[key])}</select></label>`;}
function segments(label,key,values){return `<div class="control"><span class="control-label">${esc(label)}</span><div class="segments" role="group" aria-label="${esc(label)}">${values.map(([value,text])=>`<button data-key="${key}" data-value="${value}" class="${state[key]===value?'active':''}" aria-pressed="${state[key]===value}">${esc(text)}</button>`).join('')}</div></div>`;}
function renderControls(){
 let c='';
 if(state.view==='model'){
  c+=selectControl('Decoder layer','layer',Array.from({length:93},(_,i)=>[i+1,`${String(i+1).padStart(2,'0')} · ${layerKind(i+1).toUpperCase()} · ${i===0?'Dense':'MoE'}`]));
  c+=segments('Detail','detail',[['operators','Operators'],['shapes','Tensor shapes']]);
  c+='<span class="control"><span class="control-label">Layer 1 dense · 92 latent MoE · Block AttnRes size 12</span></span>';
 }
 if(state.view==='training')c+=segments('Lifecycle','trainingMode',[['pretrain','Pretraining'],['agentic','Agentic RL']])+'<span class="control"><span class="control-label">Reported mechanisms · no invented degrees or timings</span></span>';
 const contexts=[[1024,'1k'],[4096,'4k'],[8192,'8k'],[32768,'32k'],[131072,'128k'],[262144,'256k'],[524288,'512k'],[1048576,'1M']];
 if(state.view==='serving'){
  c+=segments('Placement','serveMode',[['disaggregated','Split P/D'],['colocated','Colocated']]);
  c+=selectControl('Context','context',contexts)+selectControl('Batch','batch',[1,4,8,16,32,64,128]);
  c+=selectControl('GPUs / replica','replicaGPUs',[8,16,32,64,128]);
  c+=selectControl('MLA cache','cacheMode',[['compressed','Latent · 576'],['expanded','Expanded K/V']]);
  c+=selectControl('KV bytes','kvPrecision',[[2,'BF16 · 2'],[1,'8-bit · 1 (A)']]);
  c+=selectControl('KDA state bytes','statePrecision',[[4,'FP32 · 4 (A)'],[2,'BF16 · 2 (A)']]);
  c+=selectControl('Weights','weightMode',[['mixed','Mixed MXFP4 (A)'],['bf16','BF16 (A)']]);
  c+=selectControl('Prefix match example','match',[[512,'512'],[2048,'2048'],[2800,'2800'],[4096,'4096'],[6144,'6144']]);
 }
 if(state.view==='hardware')c+=selectControl('Arithmetic intensity','intensity',[[1,'1 FLOP/B'],[4,'4 FLOP/B'],[16,'16 FLOP/B'],[64,'64 FLOP/B'],[256,'256 FLOP/B'],[1024,'1024 FLOP/B'],[4096,'4096 FLOP/B']])+'<span class="control"><span class="control-label">Hopper example, not a reconstruction of Moonshot’s cluster</span></span>';
 if(state.view==='network'){
  c+=segments('Payload','trafficMode',[['moe','Latent MoE'],['kcp','KCP (A, B)'],['cache','Cache handoff']]);
  if(state.trafficMode!=='cache')c+=selectControl('Participating ranks','nodes',[1,2,4,8,16,32,64]);
  if(state.trafficMode==='moe')c+=selectControl('Tokens / source rank','tokens',[256,1024,4096,16384,65536]);
  if(state.trafficMode==='kcp')c+=selectControl('Fragment bytes','statePrecision',[[4,'FP32 · 4 (A)'],[2,'BF16 · 2 (A)']]);
  if(state.trafficMode==='cache')c+=selectControl('Context / request','context',contexts);
  c+=selectControl('Oversub.','oversub',[[1,'1:1'],[2,'2:1'],[4,'4:1']]);
  c+=`<label class="control"><span class="control-label">Assumed efficiency</span><input type="range" min="10" max="100" step="5" value="${state.utilization}" data-key="utilization" aria-label="Assumed link utilization"><span>${state.utilization}%</span></label>`;
 }
 $('#controls').innerHTML=c;
}
function renderMetrics(){
 const d=derived(state),{m,p,h}=d;let metrics=[];
 const M=(label,value,unit,detail,prov='C',tone='')=>({label,value,unit,detail,prov,tone});
 if(state.view==='model')metrics=[M('TOTAL PARAMETERS','2.78','T','Reported rounded count; not a manifest sum','D','accent'),M('ACTIVE PARAMETERS','104.2','B','Publisher convention · per token','D'),M('TOKEN MIXERS','69 / 24','KDA / MLA','23 × (3 KDA + 1 MLA), then final MLA','D','state'),M('ROUTED EXPERTS','16 / 896','selected','3584-wide latent · 2 full-width shared','D')];
 if(state.view==='training')metrics=state.trainingMode==='pretrain'?[M('OPTIMIZER','Muon','','Per-head updates; owner-directed P2P','D','accent'),M('PARALLELISM','PP · EP · CP','','Virtual stages + ZeRO-1 / Pipeline ZeRO-2','D'),M('DEPTH BLOCKS','8 + 1','','8 decoder groups + embedding source','C','state'),M('CONTEXT LIMIT','1,048,576','tokens','Released max_position_embeddings','D')]:[M('ROLLOUT','Partial','','Unfinished trajectories survive iterations','D','accent'),M('PREFIX POOL','DRAM','','Write-back on eviction, joint KDA / MLA','D','state'),M('TRAINING STATE','NVMe','','Phase offload frees DRAM for rollout','D'),M('ENVIRONMENT','microVM','','AgentENV pause / resume / fork','D')];
 if(state.view==='serving')metrics=[M('WEIGHT PAYLOAD ESTIMATE',fmt(d.weights/1e12,2),'TB','Rounded remainder; not checkpoint file size','A','accent'),M('HYBRID RUNNING CACHE',fmt(d.cache.total/(d.cache.total>=1e12?1e12:1e9),2),d.cache.total>=1e12?'TB':'GB',`${state.batch} requests × ${fmt(state.context)} prompt tokens`,'C','state'),M('IDEAL AVERAGE / GPU',fmt(d.perGPU/1e9,2),'GB','Weights + running cache; not a fit guarantee','A',d.perGPU>h.memoryGB*1e9?'coral':''),M('MLA INCREMENT / TOKEN',fmt(d.cache.mlaPerToken/1024,1),'KiB','24 global layers; KDA state stays fixed','C')];
 if(state.view==='hardware')metrics=[M('NOMINAL CAPACITY',h.memoryGB,'GB',`${h.memoryType}; less is allocatable`,'D','state'),M('HBM BANDWIDTH',fmt(h.hbmGBs/1000,2),'TB/s','Advertised peak memory bandwidth','D'),M('DENSE BF16 CEILING',fmt(h.bf16TF,1),'TFLOP/s','Not MXFP4 throughput; not 2:4 sparsity','C','accent'),M('NVLINK / GPU','900','GB/s','Aggregate bidirectional, not a single link','D','comm')];
 if(state.view==='network')metrics=[M('NIC / DIRECTION','50','GB/s','400 Gb/s line rate ÷ 8','C'),M('SCENARIO BANDWIDTH',fmt(d.networkGBs,2),'GB/s',`${state.utilization}% efficiency · ${state.oversub}:1 oversubscription`,'A','comm'),M(state.trafficMode==='cache'?'FULL REQUEST PAYLOAD':'SENT / RANK / LAYER',fmt(d.traffic/(d.traffic>=1024**3?1024**3:1024**2),2),d.traffic>=1024**3?'GiB':'MiB','No headers, duplication or extra messages','C'),M('BANDWIDTH COMPONENT',fmt(d.networkMs/(d.networkMs>=1000?1000:1),2),d.networkMs>=1000?'s':'ms','Not measured latency or a speedup claim','A','accent')];
 $('#metrics').innerHTML=metrics.map(v=>`<div class="metric ${v.tone}"><div class="metric-label">${esc(v.label)}<span class="provenance" title="${v.prov==='D'?'Documented':v.prov==='C'?'Calculated':'Assumed scenario'}">${v.prov}</span></div><div class="metric-value">${esc(v.value)}<span class="unit">${esc(v.unit)}</span></div><div class="metric-detail">${esc(v.detail)}</div></div>`).join('');
}
function drawNode(n){
 const cx=n.x+n.w/2,cy=n.y+n.h/2;
 let body=n.kind==='sum'?`<circle cx="${cx}" cy="${cy}" r="${n.w/2}"/>`:`<rect x="${n.x}" y="${n.y}" width="${n.w}" height="${n.h}" rx="${n.kind==='pill'?14:n.kind==='bar'?0:5}"/>`;
 if(n.kind==='gpu'){
  body+=`<rect class="chip-grid" x="${n.x+17}" y="${n.y+20}" width="43" height="43" rx="3"/><rect class="chip-core" x="${n.x+24}" y="${n.y+27}" width="29" height="29" rx="1"/>`;
  for(let i=0;i<4;i++){const v=n.x+23+i*10,u=n.y+26+i*10;body+=`<path class="chip-grid" d="M${v} ${n.y+13}v7 M${v} ${n.y+63}v7 M${n.x+10} ${u}h7 M${n.x+60} ${u}h7"/>`;}
  body+=`<text class="node-title" x="${n.x+78}" y="${n.y+35}">${esc(n.name)}</text><text class="node-sub" x="${n.x+78}" y="${n.y+56}">${esc(n.sub)}</text>`;
 }else if(['sum','rank','page','schedule'].includes(n.kind))body+=`<text class="node-title" text-anchor="middle" x="${cx}" y="${cy+4.5}">${esc(n.name)}</text>`;
 else if(n.kind!=='bar'){
  body+=`<text class="node-title" x="${n.x+13}" y="${cy+(n.sub?-5:4.5)}">${esc(n.name)}</text>`;
  if(n.sub)body+=`<text class="node-sub" x="${n.x+13}" y="${cy+13}">${esc(n.sub)}</text>`;
 }
 if(n.id==='nv-fabric')for(let i=0;i<4;i++){const x=n.x+n.w-274+i*64;body+=`<rect x="${x}" y="${n.y+15}" width="50" height="32" rx="3" style="fill:var(--panel);stroke:var(--comm);stroke-opacity:.5"/><text class="node-sub" text-anchor="middle" x="${x+25}" y="${n.y+35}">SW${i}</text>`;}
 return `<g class="node ${n.kind} ${n.tone} ${state.selected===n.id?'selected':''}" data-node="${esc(n.id)}" role="button" tabindex="0" aria-label="${esc(n.name||n.id)}${n.sub?': '+esc(n.sub):''}" aria-pressed="${state.selected===n.id}"><title>${esc(n.info?.title||n.name)}</title>${body}</g>`;
}
function drawScene(){
 const types={flow:'var(--wire)',residual:'var(--wire)',state:'var(--state)',comm:'var(--comm)',metadata:'var(--comm)',host:'var(--blue)',nvlink:'var(--comm)'};
 const defs=`<defs>${Object.entries(types).map(([key,color])=>`<marker id="arrow-${key}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6.5" markerHeight="6.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="${color}"/></marker>`).join('')}</defs>`;
 let inner=defs;
 for(const b of scene.groups){inner+=`<g class="group" data-group="${b.id}"><rect class="region ${b.id==='expanded'?'outer':''}" x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="8"/>`;if(b.title)inner+=`<text class="region-title" x="${b.x+(b.titleOffset||14)}" y="${b.y+23}">${esc(b.title)}</text>`;if(b.subtitle)inner+=`<text class="region-sub" x="${b.x+b.w-14}" y="${b.y+23}" text-anchor="end">${esc(b.subtitle)}</text>`;inner+='</g>';}
 for(const e of scene.edges){const path=e.points.map((p,i)=>`${i?'L':'M'}${p[0]},${p[1]}`).join(' ');inner+=`<path class="wire ${e.kind} ${e.delayed?'delayed':''}" data-edge="${e.id}" data-source="${esc(e.source)}" data-target="${esc(e.target)}" d="${path}" ${e.arrow&&types[e.kind]?`marker-end="url(#arrow-${e.kind})"`:''}/>`;if(e.junctionStart)inner+=`<circle class="junction" cx="${e.points[0][0]}" cy="${e.points[0][1]}" r="2.2"/>`;if(e.junctionEnd){const p=e.points[e.points.length-1];inner+=`<circle class="junction" cx="${p[0]}" cy="${p[1]}" r="2.2"/>`;}}
 for(const n of scene.nodes)inner+=drawNode(n);
 for(const l of scene.labels)inner+=`<text class="annotation ${l.tone} ${l.small?'small':''} ${l.section?'section':''} ${l.display?'display':''}" data-label="${l.id}" x="${l.x}" y="${l.y}" ${l.anchor?`text-anchor="${l.anchor}"`:''}>${esc(l.text)}</text>`;
 for(const d of scene.decorations)if(d.type==='capacity')inner+=`<path class="capacity" d="M${d.x},${d.y}v${d.h}"/>`;
 $('#diagram-host').innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" class="schematic" viewBox="0 0 ${scene.width} ${scene.height}" role="group" aria-labelledby="diagram-title diagram-desc"><title id="diagram-title">${esc(VIEWS.find(v=>v.id===state.view).canvas)} — ${esc(MODELS[state.model].name)}</title><desc id="diagram-desc">${esc(scene.caption)} Select an operator to inspect its equation and sources.</desc>${inner}</svg>`;
 updateZoom();
}
function genericInfo(n){
 const m=MODELS[state.model];
 const norm=n.name?.includes('RMSNorm'),sum=n.kind==='sum',swiglu=n.name==='SwiGLU';
 return {category:sum?'MODEL / EXPLICIT ADDITION':'MODEL / OPERATOR',title:sum?'Combine the two branches':n.name||n.id,body:norm?'RMSNorm rescales a token along the feature dimension. It does not change the tensor width.':sum?'A sum node means addition of shape-compatible tensors, not concatenation or an ambiguous crossing.':swiglu?'The gate is passed through SiLU and multiplied elementwise by the up-projection output.':'This operation sits on the selected model’s data path. Projection labels omit leading batch and sequence axes; see the linked definition for the full operator semantics.',formula:norm?'y = γ ⊙ x / √(mean(x²) + ε)':sum?'y = a + b':swiglu?'h = SiLU(g) ⊙ u':n.sub||'',facts:[],sources:m.sources,links:[]};
}
function renderInspector(){
 const node=scene.nodes.find(n=>n.id===state.selected)||scene.nodes.find(n=>n.id===scene.defaultSelected)||scene.nodes[0];
 const info=node.info||genericInfo(node),index=scene.nodes.indexOf(node);
 const srcs=(info.sources||[]).map(sourceById).filter(Boolean);
 const factsHTML=(info.facts||[]).length?`<dl class="facts">${info.facts.map(([k,v])=>`<div class="fact"><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('')}</dl>`:'';
 const sourcesHTML=srcs.map(s=>`<a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer"><span>${esc(s.title)}</span><span class="ext">↗</span></a>`).join('');
 const linksHTML=(info.links||[]).length?`<div class="inspector-label">FOLLOW THE SYSTEM</div>${info.links.map(([view,text])=>`<button class="deep-link" data-view="${view}">${esc(text)} <span>→</span></button>`).join('')}`:'';
 $('#inspector').innerHTML=`
  <div class="inspector-top"><span>COMPONENT INSPECTOR</span><span>⌖ ${String(index+1).padStart(2,'0')}</span></div>
  <div class="inspector-content">
   <div class="inspect-primary"><div class="inspector-eyebrow">${esc(info.category)}</div><h3>${esc(info.title)}</h3><p>${esc(info.body)}</p>${info.formula?`<div class="equation">${esc(info.formula)}</div>`:''}</div>
   <div>${factsHTML}<div class="inspector-label">SOURCE OF TRUTH</div><div class="source-links">${sourcesHTML}</div>${linksHTML}</div>
  </div>
  <div class="inspector-foot"><b>D</b> documented &nbsp; <b>C</b> calculated &nbsp; <b>A</b> assumed<br>Specs are not measurements. Bounds are not benchmarks.</div>`;
}
function selectNode(id){
 const n=scene.nodes.find(n=>n.id===id);if(!n)return;
 if(n.action?.startsWith('layer:')){state.layer=Number(n.action.split(':')[1]);state.selected=null;render();persist();return;}
 state.selected=id;
 $$('.schematic .node').forEach(el=>{const active=el.dataset.node===id;el.classList.toggle('selected',active);el.setAttribute('aria-pressed',String(active));});
 renderInspector();
}
function hoverNode(id){
 const connected=new Set([id]);for(const e of scene.edges)if(e.source===id||e.target===id){connected.add(e.source);connected.add(e.target);}
 $$('.schematic .wire').forEach(el=>el.classList.toggle('related',!!id&&(el.dataset.source===id||el.dataset.target===id)));
}
function modelCompanion(d){
 const {p}=d,routedPct=100*p.routed/p.total;
 return `<article class="companion-card"><div class="eyebrow">C / ROUTED WEIGHT ANATOMY</div><h3>The latent experts dominate residency.</h3><div class="mini-bar"><span style="width:${routedPct}%;background:var(--accent)"></span><span style="width:${100-routedPct}%;background:var(--state)"></span></div><div class="mini-legend"><span>Routed matrices <b>${compact(p.routed,3)}</b></span><span>Reported total <b>2.78T</b></span></div><p style="margin-top:16px">92 MoE layers × 896 experts × 3 × 3584 × 3072. This exact matrix count is distinct from the publisher’s rounded whole-model count. Top-16 routing reduces activated computation, not the number of weights that need a residency plan.</p></article><article class="companion-card"><div class="eyebrow">THREE AXES / THREE DIFFERENT MECHANISMS</div><h3>Sequence. Depth. Channels.</h3><p>KDA evolves a fixed recurrent matrix over token time. MLA keeps global content-addressable history. AttnRes retrieves across depth, while Stable LatentMoE routes through half-width expert vectors. The first decoder layer is dense; layer 93 is always global MLA.</p><button class="cta" data-view="serving">Follow the two attention states →</button></article>`;
}
function rooflineSVG(d){
 const W=540,H=221,x0=48,x1=514,y0=179,y1=29,x=i=>x0+Math.log2(i)/12*(x1-x0),y=tf=>y0-Math.log10(Math.max(1,tf))/Math.log10(2000)*(y0-y1);
 const pts=[];for(let i=0;i<=96;i++){const ai=2**(i/8);pts.push([x(ai),y(roofline(d.h,ai))]);}
 let str=`<svg class="roofline" viewBox="0 0 ${W} ${H}" role="img" aria-label="Dense BF16 roofline: the lesser of compute ceiling and arithmetic intensity times HBM bandwidth"><text x="${x0}" y="12" style="fill:var(--faint);font:9px var(--mono)">DENSE BF16 · TFLOP/s</text>`;
 for(const tf of [1,10,100,1000])str+=`<path d="M${x0} ${y(tf)}H${x1}" stroke="var(--line)" fill="none"/><text x="${x0-9}" y="${y(tf)+3}" text-anchor="end" style="fill:var(--faint);font:9px var(--mono)">${tf}</text>`;
 for(const ai of [1,4,16,64,256,1024,4096])str+=`<text x="${x(ai)}" y="${y0+17}" text-anchor="middle" style="fill:var(--faint);font:8px var(--mono)">${ai}</text>`;
 str+=`<path d="${pts.map((p,i)=>(i?'L':'M')+p.join(',')).join(' ')}" fill="none" stroke="var(--state)" stroke-width="2"/><path d="M${x(d.ridge)} ${y0}V${y(d.h.bf16TF)}" stroke="var(--faint)" stroke-dasharray="3 4"/><circle cx="${x(state.intensity)}" cy="${y(d.roofTF)}" r="4" fill="var(--accent)" stroke="var(--canvas)" stroke-width="2"/><text x="${Math.min(x1-2,x(state.intensity)+12)}" y="${y(d.roofTF)-10}" text-anchor="${state.intensity>=1024?'end':'start'}" style="fill:var(--accent);font:10px var(--mono)">${fmt(d.roofTF,1)} TFLOP/s</text><text x="${x0}" y="${H-3}" style="fill:var(--faint);font:9px var(--mono)">ARITHMETIC INTENSITY · FLOP / BYTE (log scale)</text></svg>`;
 return str;
}
function renderCompanion(){
 const d=derived(state),{h}=d;let text='';
 const card=(eyebrow,title,body)=>`<article class="companion-card"><div class="eyebrow">${eyebrow}</div><h3>${title}</h3>${body}</article>`;
 if(state.view==='model')text=modelCompanion(d);
 if(state.view==='training')text=card('D / REPORTED, NOT RECONSTRUCTED','No fabricated training-memory total.',`<p>K3 combines Per-Head Muon, expert placement, ZeRO-1 optimizer sharding, Pipeline ZeRO-2 gradients, virtual pipeline stages and tensor-level activation policies. The report does not reduce this to a single uniform 16-byte Adam state. Exact parallelism degrees and live buffer sizes remain outside this atlas.</p>`)+card('STATE / KEEP THE LIFECYCLES DISTINCT',state.trainingMode==='agentic'?'A trajectory includes an environment.':'A tensor has an owner and a lifetime.',`<p>${state.trainingMode==='agentic'?'KDA/MLA prefixes, policy/optimizer weights and microVM snapshots are different kinds of state. Restoring a conversation without its environment, or a recurrent matrix without the matching prefix, is not a valid continuation.':'A replicated shared expert, a migrated routed expert, a completed depth block and an offloaded activation do not have the same ownership. Read the inspector for the contract attached to each transfer and reuse boundary.'}</p><button class="cta" data-view="network">Inspect the communication contracts →</button>`);
 if(state.view==='serving')text=card('C / STATE DECOMPOSITION','Fixed state plus growing history.',`<div class="mini-bar">${[[d.cache.mla,'var(--state)'],[d.cache.recurrent,'var(--comm)'],[d.cache.conv,'var(--blue)']].map(([n,c])=>`<span style="width:${100*n/d.cache.total}%;background:${c}"></span>`).join('')}</div><div class="mini-legend"><span>MLA <b>${gb(d.cache.mla)}</b></span><span>KDA <b>${gb(d.cache.recurrent)}</b></span><span>Conv <b>${gb(d.cache.conv)}</b></span></div><p style="margin-top:16px">${state.cacheMode==='compressed'?'NoPE still retains 512 + 64 = 576 cache elements per global layer.':'The reference expanded layout keeps 96 × (192 + 128) K/V elements per global layer.'} Running-state totals exclude retained prefix snapshots, physical-page rounding, extra TP copies and generated-token reserve.</p>`)+card('A / PAYLOAD DOES NOT CERTIFY FIT','Mixed precision is not four bits everywhere.',`<p>MXFP4 routed experts are budgeted at 4 bits plus one byte of scale per 32 values. The rounded non-routed remainder is budgeted at BF16. ${state.serveMode==='disaggregated'?`Split P/D uses two independent weight replicas: ${state.replicaGPUs*2} GPU slots at the selected ${state.replicaGPUs} per pool.`:'Colocation shares one replica but prompt work competes with decoding.'} Arithmetic averaging does not model expert imbalance, KV replication, dequantization workspace or engine support.</p><button class="cta" data-view="hardware">Inspect the selected hardware →</button>`);
 if(state.view==='hardware')text=card('C / THEORETICAL DENSE BF16 ROOFLINE','A compute bound, not a serving benchmark.',rooflineSVG(d))+card('HOPPER EXAMPLE / NOT THE PRODUCTION CLUSTER','Storage format is not compute throughput.',`<p>At ${state.intensity} FLOP/byte the ideal dense BF16 roof is ${fmt(d.roofTF,1)} TFLOP/s. MXFP4 packed weight storage does not turn a Hopper GPU into native FP4 Tensor Cores. Kernel support, conversion and memory traffic must be validated for a chosen engine. The DGX H100/H200 topology is a documented illustrative substrate, not an assertion about Moonshot’s deployment.</p><button class="cta" data-view="network">Cross the node boundary →</button>`);
 if(state.view==='network')text=card('C / THE SIZE OF THE MESSAGE IS MODEL-SPECIFIC',state.trafficMode==='moe'?'Half-width vectors. Not half the model.':state.trafficMode==='kcp'?'Affine transitions, not additive fragments.':'Restore one coherent continuation.',`<p>${state.trafficMode==='moe'?`At ${fmt(state.tokens)} source tokens, logical dispatch is ${gb(d.dispatch.logical)} before removing local destinations. Full-width vectors would be ${gb(d.dispatch.fullWidth)}. This is vector-byte accounting, not a measured 2× latency improvement.`:state.trafficMode==='kcp'?`Each rank contributes ${fmt(d.kcp.local/1024**2,2)} MiB for the 96-head (A, B) pair at the selected precision. All-gather send volume grows with rank count but not token count. Layer count and head TP are not silently multiplied in.`:`A full request transfer is ${gb(d.handoff.total)} at the selected context, cache representation and precision. Real transfer can involve page alignment, checkpoint metadata and head re-layout for asymmetric TP.`}</p>`)+card('A / SCENARIO, NOT SIMULATOR','A line rate is not a promise.',`<p>400 Gb/s is 50 GB/s in one direction. Eight NICs can inject an ideal 400 GB/s per direction in aggregate; that is distinct from 900 GB/s bidirectional NVLink per GPU. The estimate omits startup, congestion, scheduling, overlap and extra migration traffic. A single participating rank sends zero collective payload.</p><button class="cta" id="method-cta">Read the assumptions and sources ↗</button>`);
 $('#companion').innerHTML=text;
}
function updateZoom(){
 const host=$('#diagram-host'),viewport=$('#canvas-viewport');
 const minimum=parseFloat(getComputedStyle(host).minWidth)||0;
 const base=Math.max(viewport.clientWidth-2,minimum);
 host.style.width=state.zoom===1?'100%':`${base*state.zoom}px`;
 $('#zoom-fit').textContent=state.zoom===1?(minimum>viewport.clientWidth?'1×':'Fit'):`${Math.round(state.zoom*100)}%`;
 $('#zoom-out').disabled=state.zoom<=1;$('#zoom-in').disabled=state.zoom>=2.5;
}
function render(){
 document.documentElement.dataset.theme=state.theme;
 $('#theme-button').innerHTML=state.theme==='dark'?ICONS.sun:ICONS.moon;
 $('#theme-button').setAttribute('aria-label',`Switch to ${state.theme==='dark'?'light':'dark'} theme`);
 $('#model-select').innerHTML=optionHTML(Object.entries(MODELS).map(([id,m])=>[id,m.name]),state.model);
 $('#hardware-select').innerHTML=optionHTML(Object.entries(HARDWARE).map(([id,h])=>[id,h.name]),state.hardware);
 $('.view-tabs').innerHTML=VIEWS.map((v,i)=>`<button class="view-tab" id="tab-${v.id}" role="tab" aria-selected="${state.view===v.id}" aria-controls="workspace" data-view="${v.id}" tabindex="${state.view===v.id?0:-1}"><span class="num">0${i+1}</span>${ICONS[v.id]}${v.label}</button>`).join('');
 const v=VIEWS.find(v=>v.id===state.view);
 $('#view-eyebrow').textContent=v.eyebrow;$('#view-title').textContent=v.title;$('#view-subtitle').textContent=v.subtitle;
 $('#canvas-title').textContent=v.canvas;$('#canvas-badge').textContent=v.badge;
 renderMetrics();renderControls();
 scene=buildScene(state);if(!state.selected||!scene.nodes.some(n=>n.id===state.selected))state.selected=scene.defaultSelected;
 drawScene();renderInspector();renderCompanion();$('#figure-caption').textContent=scene.caption;
 document.title=`${v.label} · ${MODELS[state.model].name} — STRATUM`;
}
function openSources(){if(!$('#sources-dialog').open)$('#sources-dialog').showModal();}
function openSearch(){if(!$('#search-dialog').open)$('#search-dialog').showModal();$('#search-input').value='';updateSearch();setTimeout(()=>$('#search-input').focus(),0);}
function updateSearch(){
 const q=$('#search-input').value.toLowerCase().trim();
 const results=[...VIEWS.map(v=>({label:v.label,kind:'Lens',view:v.id})),...scene.nodes.filter(n=>n.name&&n.kind!=='bar').map(n=>({label:n.info?.title||n.name,kind:n.kind==='op'?'Operation':n.kind,node:n.id}))].filter(v=>`${v.label} ${v.kind}`.toLowerCase().includes(q)).slice(0,24);
 $('#search-results').innerHTML=results.length?results.map(r=>`<button class="search-result" ${r.view?`data-search-view="${r.view}"`:`data-search-node="${esc(r.node)}"`}>${esc(r.label)}<span>${esc(r.kind)} ↗</span></button>`).join(''):'<div class="empty-search">No matching operation in this view.</div>';
}
function downloadText(text,name,type){const url=URL.createObjectURL(new Blob([text],{type}));const a=document.createElement('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),5000);}
function exportSVG(){
 const original=$('.schematic'),copy=original.cloneNode(true),rootStyles=getComputedStyle(document.documentElement);
 copy.setAttribute('width',scene.width);copy.setAttribute('height',scene.height);copy.setAttribute('xmlns','http://www.w3.org/2000/svg');
 const colors=['canvas','panel','surface','raised','text','muted','faint','line','line-strong','accent','accent-dim','accent-bg','state','state-bg','comm','comm-bg','blue','blue-bg','coral','coral-bg','wire','sans','mono'];
 let css=`svg{${colors.map(k=>`--${k}:${rootStyles.getPropertyValue('--'+k)}`).join(';')};background:var(--canvas);font-family:var(--sans)}`;
 for(const sheet of document.styleSheets){try{for(const rule of sheet.cssRules)if(rule.cssText.includes('.schematic'))css+='\n'+rule.cssText;}catch{}}
 const style=document.createElementNS('http://www.w3.org/2000/svg','style');style.textContent=css;copy.prepend(style);
 copy.querySelectorAll('[tabindex]').forEach(e=>e.removeAttribute('tabindex'));
 const bg=document.createElementNS('http://www.w3.org/2000/svg','rect');bg.setAttribute('width',scene.width);bg.setAttribute('height',scene.height);bg.setAttribute('fill',rootStyles.getPropertyValue('--canvas').trim());copy.insertBefore(bg,copy.querySelector('defs'));
 downloadText('<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(copy),`stratum-${state.model}-${state.view}.svg`,'image/svg+xml');notify('Vector schematic exported.');
}
function exportConfig(){
 const config={schema:'stratum-kimi-k3/v1',generated:new Date().toISOString(),state:{...state,selected:undefined,zoom:undefined},model:MODELS[state.model],hardware:HARDWARE[state.hardware],units:{memory:'decimal GB unless explicitly marked GiB/MiB',network:'one direction unless explicitly marked bidirectional'},scope:'Analytic educational atlas; no measured performance or deployment-fit guarantee.',sources:SOURCES};
 downloadText(JSON.stringify(config,null,2),'stratum-kimi-k3-configuration.json','application/json');notify('Configuration and source provenance exported.');
}
/** Browser geometry audit. Public debug API; reports defects rather than masking them. */
function geometryReport(){
 const texts=[...$('.schematic').querySelectorAll('text')].filter(t=>t.textContent.trim()).map(t=>{const b=t.getBBox();return {id:t.closest('[data-node]')?.dataset.node||t.dataset.label||t.classList[0],node:t.closest('[data-node]')?.dataset.node,text:t.textContent,x:b.x,y:b.y,w:b.width,h:b.height};});
 const overlap=(a,b,pad=0)=>a.x+pad<b.x+b.w&&a.x+a.w-pad>b.x&&a.y+pad<b.y+b.h&&a.y+a.h-pad>b.y;
 const segmentBox=(a,b,r)=>{const p=.8,x0=r.x+p,x1=r.x+r.w-p,y0=r.y+p,y1=r.y+r.h-p;if(a[0]===b[0])return a[0]>x0&&a[0]<x1&&Math.max(a[1],b[1])>y0&&Math.min(a[1],b[1])<y1;return a[1]>y0&&a[1]<y1&&Math.max(a[0],b[0])>x0&&Math.min(a[0],b[0])<x1;};
 const report={textOverflow:[],textOverlap:[],nodeOverlap:[],edgeText:[],edgeNode:[],canvasOverflow:[]};
 for(const t of texts)if(t.x<-.5||t.y<-.5||t.x+t.w>scene.width+.5||t.y+t.h>scene.height+.5)report.canvasOverflow.push({text:t.text});
 for(const t of texts)if(t.node){const n=scene.nodes.find(n=>n.id===t.node);if(t.x<n.x-1||t.x+t.w>n.x+n.w+1||t.y<n.y-1||t.y+t.h>n.y+n.h+1)report.textOverflow.push({node:n.id,text:t.text,width:t.w,available:n.w});}
 for(let i=0;i<texts.length;i++)for(let j=i+1;j<texts.length;j++)if(overlap(texts[i],texts[j],.8))report.textOverlap.push([texts[i].text,texts[j].text]);
 for(let i=0;i<scene.nodes.length;i++)for(let j=i+1;j<scene.nodes.length;j++)if(overlap(scene.nodes[i],scene.nodes[j],.8))report.nodeOverlap.push([scene.nodes[i].id,scene.nodes[j].id]);
 for(const edge of scene.edges){for(let i=1;i<edge.points.length;i++){
  for(const text of texts)if(segmentBox(edge.points[i-1],edge.points[i],text))report.edgeText.push({edge:edge.id,text:text.text});
  for(const n of scene.nodes)if(n.id!==edge.source&&n.id!==edge.target&&segmentBox(edge.points[i-1],edge.points[i],n))report.edgeNode.push({edge:edge.id,node:n.id});
 }}
 return report;
}
$('#focus-button').innerHTML=ICONS.expand;$('#export-button').innerHTML=ICONS.export;
$('#source-list').innerHTML=SOURCES.map((s,i)=>`<div class="source-row"><span class="source-num">${String(i+1).padStart(2,'0')}</span><div><a href="${esc(s.url)}" target="_blank" rel="noopener noreferrer">${esc(s.title)} ↗</a><small>${esc(s.author)}</small><p>${esc(s.note)}</p></div></div>`).join('');
$('#model-select').addEventListener('change',e=>setState({model:e.target.value}));
$('#hardware-select').addEventListener('change',e=>setState({hardware:e.target.value}));
$('#theme-button').addEventListener('click',()=>setState({theme:state.theme==='dark'?'light':'dark'}));
$('#sources-button').addEventListener('click',openSources);$('.dialog-close').addEventListener('click',()=>$('#sources-dialog').close());
$('#search-button').addEventListener('click',openSearch);$('#search-input').addEventListener('input',updateSearch);
$('#zoom-in').addEventListener('click',()=>{state.zoom=Math.min(2.5,state.zoom+.25);updateZoom();});
$('#zoom-out').addEventListener('click',()=>{state.zoom=Math.max(1,state.zoom-.25);updateZoom();});
$('#zoom-fit').addEventListener('click',()=>{state.zoom=1;updateZoom();$('#canvas-viewport').scrollTo({left:0,top:0});});
$('#focus-button').addEventListener('click',()=>{document.body.classList.toggle('focus-mode');$('#focus-button').setAttribute('aria-label',document.body.classList.contains('focus-mode')?'Restore inspector':'Expand canvas');updateZoom();});
$('#export-button').addEventListener('click',exportSVG);$('#export-config').addEventListener('click',exportConfig);
$('#controls').addEventListener('change',e=>{const k=e.target.dataset.key;if(k)setState({[k]:typeof DEFAULT_STATE[k]==='number'?Number(e.target.value):e.target.value});});
$('#controls').addEventListener('click',e=>{const b=e.target.closest('button[data-key]');if(b)setState({[b.dataset.key]:b.dataset.value});});
$('#diagram-host').addEventListener('click',e=>{const n=e.target.closest('[data-node]');if(n)selectNode(n.dataset.node);});
$('#diagram-host').addEventListener('mouseover',e=>{const n=e.target.closest('[data-node]');hoverNode(n?n.dataset.node:null);});
$('#diagram-host').addEventListener('mouseleave',()=>hoverNode(null));
$('#diagram-host').addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)&&e.target.dataset.node){e.preventDefault();selectNode(e.target.dataset.node);}});
$('#search-results').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;$('#search-dialog').close();if(b.dataset.searchView)setState({view:b.dataset.searchView});else if(b.dataset.searchNode){selectNode(b.dataset.searchNode);const node=$(`[data-node="${b.dataset.searchNode}"]`);node?.focus({preventScroll:true});}});
for(const dialog of $$('dialog'))dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});
document.addEventListener('click',e=>{const b=e.target.closest('[data-view]');if(b)setState({view:b.dataset.view});if(e.target.closest('#method-cta'))openSources();});
document.addEventListener('keydown',e=>{
 if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();return;}
 if(e.key==='Escape'){if($('#search-dialog').open)$('#search-dialog').close();if($('#sources-dialog').open)$('#sources-dialog').close();return;}
 if(e.target.closest('input,select,textarea,dialog'))return;
 if(e.key==='/'){e.preventDefault();openSearch();}
 if(/^[1-5]$/.test(e.key))setState({view:VIEWS[Number(e.key)-1].id});
 if(e.target.matches('.view-tab')&&['ArrowRight','ArrowLeft','Home','End'].includes(e.key)){e.preventDefault();const i=VIEWS.findIndex(v=>v.id===state.view);const next=e.key==='Home'?0:e.key==='End'?4:(i+(e.key==='ArrowRight'?1:4))%5;setState({view:VIEWS[next].id});$(`#tab-${VIEWS[next].id}`).focus();}
 if(e.key==='0'){state.zoom=1;updateZoom();}
});
window.addEventListener('resize',updateZoom);
window.addEventListener('hashchange',()=>{try{state=sanitizeState({...DEFAULT_STATE,...Object.fromEntries(new URLSearchParams(location.hash.slice(1)))});render();}catch{notify('Invalid configuration in URL.');}});
window.STRATUM={getState:()=>({...state}),getScene:()=>scene,setState:patch=>{const safe=sanitizeState({...state,...patch});setState(safe);},geometryReport,exportSVG,exportConfig};
render();
