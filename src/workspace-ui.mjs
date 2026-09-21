import {renderMechanismChapter} from './mechanism-visuals.mjs';
/** The connected reading workspace. Model facts and executable code stay in their owners. */
import {MODELS,focalStages,getModel} from './core.mjs';
import {PAGES,MODEL_VIEWS,sectionHref} from './pages.mjs';
import {DEEP_DIVE_TITLES,chapterContent} from './deepdives.mjs';
import {moduleExample,moduleStage} from './pytorch.mjs';
import {workedHTML,tensorText} from './worked-examples.mjs';
import {implementationReferences} from './implementation-sources.mjs';
const $=id=>document.getElementById(id);
const LABELS={chapter:'Chapter',schematic:'Schematic',maths:'Maths',pytorch:'PyTorch',worked:'Worked example',sources:'Sources'};
const CHAPTERS={block:'Layer architecture',ced:'Overview',encoder:'Causal encoder',decoder:'Decoder',inputs:'Inputs & vision',residual:'Residual streams',ple:'N-gram memory',...DEEP_DIVE_TITLES};
export function closeNavigation(){const open=document.body.classList.contains('navigation-open');document.body.classList.remove('navigation-open');$('navigation-toggle').setAttribute('aria-expanded','false');if(open)$('navigation-toggle').focus();return open;}
export function toggleNavigation(){if(closeNavigation())return;document.body.classList.add('navigation-open');$('navigation-toggle').setAttribute('aria-expanded','true');$('chapter-nav').querySelector('button,a')?.focus();}
function sourceGroups(state,info,esc){
 const groups=implementationReferences(state.model,state.view,info);
 const titles={baseline:'Publisher baseline & Transformers',serving:'Serving implementations',training:'Training mechanisms'};
 const order=state.view==='training'?['training','baseline']:state.view==='serving'?['serving','baseline']:['baseline','serving','training'];
 return order.map(key=>`<section class="reference-group"><h4>${titles[key]}</h4>${groups[key].map(r=>`<article class="implementation-reference"><a href="${esc(r.url)}" target="_blank" rel="noopener noreferrer">${esc(r.title)} ↗</a><p class="reference-scope">${esc(r.scope)} · ${esc(r.provider)}</p><code>${esc(r.symbols.join(' · '))}</code><p>${esc(r.explanation)}</p><details><summary>Revision & evidence status</summary><code>${esc(r.revision||'Unpinned')}</code><p>Source mapping; independent acceptance and runtime execution remain pending.</p></details></article>`).join('')||'<p>No implementation mapping is available for this operation.</p>'}</section>`).join('');
}
export function renderModuleInspector({state,scene,workspace,esc,numberText}){
 const n=workspace.node;if(!n){$('inspector').innerHTML='';return;}
 const info=n.info||{},index=scene.nodes.findIndex(x=>x.id===n.id),meta=workspace.metadata;
 const list=(title,nodes)=>nodes?.length?`<div class="inspector-section"><h4>${title}</h4><ul class="operator-contract">${nodes.map(x=>`<li><button data-operator="${esc(x.id)}">${esc(x.name)}</button>${x.sub?`<code>${esc(x.sub)}</code>`:''}</li>`).join('')}</ul></div>`:'';
 $('inspector').innerHTML=`${state.panel==='chapter'?'<button class="button close-inspector" data-close-inspector="true">Close operation ×</button>':''}<div class="operator-navigation"><span>Operator <code>${index+1} / ${scene.nodes.length}</code></span><button class="icon-button" data-operator="${esc(scene.nodes[Math.max(0,index-1)].id)}" aria-label="Previous operator" ${index===0?'disabled':''}>‹</button><button class="icon-button" data-operator="${esc(scene.nodes[Math.min(scene.nodes.length-1,index+1)].id)}" aria-label="Next operator" ${index===scene.nodes.length-1?'disabled':''}>›</button></div><div class="inspector-top"><span class="eyebrow">${esc(info.category||'Operation')}</span><h3>${esc(info.title||n.name)}</h3><p>${numberText(info.body||'')}</p></div>${list('Inputs from',meta.inputs)}${list('Outputs feed',meta.outputs)}${list('Computed here',meta.computed)}${list('Retained-state inputs',meta.retainedInputs||meta.reused)}${info.formula?`<div class="inspector-section"><h4>Mathematical contract</h4><pre class="formula">${esc(info.formula)}</pre></div>`:''}${info.facts?.length?`<div class="inspector-section"><h4>Tensor & ownership contract</h4><dl class="facts">${info.facts.map(([k,v])=>`<div class="fact"><dt>${esc(k)}</dt><dd>${numberText(v)}</dd></div>`).join('')}</dl></div>`:''}<div class="inspector-section"><span class="status-pill">Reference mapped · review pending</span><p>Inspect the selected operation’s sources and implementation scope.</p>${workspace.panels.includes('pytorch')?'<button class="button" data-panel="pytorch">Canonical PyTorch →</button>':''}${workspace.panels.includes('sources')?'<button class="button" data-panel="sources">Sources & implementations →</button>':''}</div>`;
}
export function renderWorkspaceUI({state,scene,workspace,pageInfo,esc,numberText,pytorchExamples,sourceCard}){
 document.body.classList.toggle('reading-chapter',state.panel==='chapter');
 const shared=Boolean(pageInfo?.view);document.body.classList.add('connected-workspace');
 const model=getModel(state.model),stages=(model.encoderLayers?['ced','encoder','decoder','engram','csa2','dspark','inputs','residual']:['block',...focalStages(model)]).filter(k=>!moduleExample(k)&&!(model.encoderLayers&&k==='block'));
 if(stages.includes('ced'))stages.splice(stages.indexOf('ced'),1),stages.unshift('ced');
 const lenses=$('lens-tabs');
 $('chapter-nav').innerHTML=`<div class="sidebar-heading">${esc(pageInfo?.view?pageInfo.title:pageInfo?.title||model.name)}</div>${!shared?`<nav aria-label="Architecture chapters">${stages.map(stage=>`<button data-setting="stage" data-value="${stage}" data-chapter="${stage}" class="chapter-link ${state.view==='model'&&state.stage===stage?'active':''}" aria-current="${state.view==='model'&&state.stage===stage?'page':'false'}">${esc(CHAPTERS[stage]||stage)}</button>`).join('')}</nav>`:''}<div id="sidebar-lenses"></div><nav class="sidebar-sections" aria-label="Shared sections"><a href="index.html">Home</a>${PAGES.filter(p=>p.view).map(p=>`<a href="${sectionHref(p.view,state)}">${esc(p.title)}</a>`).join('')}<a href="data-representation.html">Data Representation</a>${shared?'<button class="chapter-link" data-view="model">Return to model</button><button class="chapter-link" data-view="evidence">Model evidence</button>':''}</nav><p class="sidebar-note">Publisher baseline<br>Framework execution<br>Explicit evidence</p>`;
 // Keep the existing lens controls and keyboard seam; change their location, not their behavior.
 $('sidebar-lenses').append(lenses);
 $('site-nav').hidden=true;$('model-strip').hidden=true;
 const existing=$('checkpoint-picker');existing?.remove();
 if(pageInfo?.models.length>1){const label=document.createElement('label');label.id='checkpoint-picker';label.className='checkpoint-picker';label.innerHTML=`<span class="sr-only">Exact checkpoint</span><select aria-label="Exact checkpoint" data-key="model">${pageInfo.models.map(id=>`<option value="${id}" ${id===state.model?'selected':''}>${esc(MODELS[id].name)}</option>`).join('')}</select>`;document.querySelector('.global-actions').before(label);}
 $('model-title').textContent=shared?pageInfo.title:model.name;
 $('model-eyebrow').textContent=state.view==='model'?'Architecture / '+(CHAPTERS[state.stage]||moduleExample(state.stage)?.title||'Module'):state.view;
 $('model-subtitle').textContent=state.view==='model'?(chapterContent(model,state)?.question||scene.caption):$('view-description').textContent;
 $('view-heading').textContent=state.view==='model'?(CHAPTERS[state.stage]||moduleExample(state.stage)?.title||'Architecture'):$('view-heading').textContent;
 $('identity-metrics').hidden=state.panel!=='schematic'||state.view!=='model'||!['ced','block'].includes(state.stage);
 // Chapter selection lives only in the sidebar; module dropdown remains available.
 document.querySelectorAll('#controls .control').forEach(c=>{if(c.querySelector('[data-setting="stage"]'))c.hidden=true;});
 const tabs=workspace.panels;
 $('workspace-tabs').hidden=tabs.length<=1;
 $('workspace-tabs').innerHTML=tabs.map(panel=>`<button class="workspace-tab ${state.panel===panel?'active':''}" data-panel="${panel}" aria-pressed="${state.panel===panel}">${LABELS[panel]}</button>`).join('');
 const panel=$('module-panel');panel.hidden=state.panel==='schematic';
 document.querySelector('.canvas-frame').hidden=!panel.hidden;
 $('controls').hidden=state.panel==='chapter';
 const info=workspace.info||{};
 if(!panel.hidden){
  let content='';
  if(state.panel==='chapter')content=renderMechanismChapter(state,model);
  if(state.panel==='maths')content=(workspace.mathematics||[info]).map(i=>`<section class="module-mathematics"><h4>${esc(i.title||'Mathematical contract')}</h4>${i.formula?`<pre class="formula">${esc(i.formula)}</pre>`:''}<p class="workspace-copy">${numberText(i.body||'')}</p><dl class="facts">${(i.facts||[]).map(([k,v])=>`<div class="fact"><dt>${esc(k)}</dt><dd>${numberText(v)}</dd></div>`).join('')}</dl></section>`).join('');
  if(state.panel==='pytorch')content=workspace.pythonExamples.map(example=>pytorchExamples(moduleStage(example.id))).join('');
  if(state.panel==='worked'){
   const fixture=workspace.fixture,rows=fixture?[...fixture.inputs,...fixture.steps]:[];
   const step=workspace.step||rows[0];
   content=(step?`<label class="step-picker">Inspect a computed tensor<select data-key="example" aria-label="CPU tensor step">${rows.map(row=>`<option value="${esc(workspace.fixtureId+'--'+row.id)}" ${step.id===row.id?'selected':''}>${esc(row.title)}</option>`).join('')}</select></label><section class="cpu-step"><h4>${esc(step.title)}</h4><p class="workspace-copy">${esc(step.operation)}</p><code>${esc(step.axes.join(' × '))} · ${esc(tensorText(step.shape))} · ${esc(step.dtype)}</code><pre class="formula" data-cpu-step="${esc(step.id)}">${esc(tensorText(step.values))}</pre></section>`:'')+`<p class="workspace-copy">These values were computed by the checked-in CPU PyTorch program. Shapes are reduced; the implementation and intermediate tensors remain inspectable.</p>`+workedHTML(moduleStage(workspace.example.id));
  }
  if(state.panel==='sources')content=(shared?'':sourceGroups(state,{...info,operatorId:workspace.node?.id},esc))+`<details class="complete-references"><summary>Operation reference chain</summary>${[...new Set([...(info.sources||[]),...(workspace.example?.sources||[])])].map(sourceCard).join('')}</details>`;
  panel.innerHTML=state.panel==='chapter'?content:`<div class="module-panel-heading"><span class="eyebrow">${LABELS[state.panel]}</span><h3>${esc(info.title||workspace.node?.name||'Module')}</h3></div>${content}`;
 }else panel.innerHTML='';
 if(state.panel==='pytorch')installCodeNavigator(panel,workspace.example,esc,state);
 // Keep all authored chapter explanations reachable, without making every view a long appendix.
 const analysis=$('analysis-panel');analysis.hidden=state.panel!=='schematic';
 if(state.view==='model'&&analysis.innerHTML.trim())analysis.innerHTML=`<details class="chapter-reference"><summary>Chapter explanation, mathematics & complete examples</summary>${analysis.innerHTML}</details>`;
 $('principles').hidden=state.view==='model';
}

function installCodeNavigator(panel,example,esc,state){
 if(!example)return;
 const pre=panel.querySelector(`[data-example="${example.id}"] .pytorch-code`);if(!pre)return;
 const symbols=example.code.split('\n').flatMap((line,index)=>{const match=line.match(/^(?:class|def) (\w+)/);return match?[{name:match[1],line:index}]:[];});
 if(!symbols.length)return;
 const label=document.createElement('label');label.className='code-symbol-picker';label.innerHTML=`Entry point<select aria-label="Python entry point">${symbols.map(symbol=>`<option value="${symbol.line}">${esc(symbol.name)}</option>`).join('')}</select>`;pre.before(label);
 const preferred=state.example.startsWith('chap--residual--')?'sinkhorn_trace':{'attention-mixer':'HyperConnectionMixer','ffn-mixer':'HyperConnectionMixer','attention-read':'collapse_residual_streams','ffn-read':'collapse_residual_streams','attention-write':'combine_residual_streams','ffn-write':'combine_residual_streams'}[state.module];
 const select=label.querySelector('select');select.value=String((symbols.find(symbol=>symbol.name===preferred)||symbols.at(-1)).line);
 const focus=()=>{const lines=pre.querySelectorAll('.code-line');for(const line of lines)line.classList.remove('code-entry');const line=lines[Number(select.value)];if(line){line.classList.add('code-entry');pre.scrollTop=line.offsetTop-pre.offsetTop-16;}};
 select.addEventListener('change',focus);requestAnimationFrame(focus);
}
