import test from 'node:test';import assert from 'node:assert/strict';
import {MODELS,HARDWARE,VIEWS,DEFAULT_STATE,layerKind,sanitizeState,focalStages} from '../src/core.mjs';import {buildScene} from '../src/diagrams.mjs';import {buildCED} from '../src/diagram-meta.mjs';import {attentionRenderer} from '../src/diagram-model.mjs';import {structuralReport} from '../src/scene.mjs';
for(const id of Object.keys(MODELS))test(`${id}: every decoder layer has a contained, orthogonal scene`,()=>{for(let layer=1;layer<=MODELS[id].layers;layer++){const g=buildScene({...DEFAULT_STATE,model:id,layer});assert.deepEqual(structuralReport(g),[],`layer ${layer}`);assert.ok(g.nodes.every(n=>n.info?.sources.length));}});
// The lens axis is `VIEWS`, not a copy of it. Models and hardware in the body below were already
// derived; this one literal meant a new lens got no structural scene test at all -- and a test
// that never runs for a lens reads exactly like a lens that passes. 'model' is excluded because
// the per-layer sweep above already covers it for every layer of every checkpoint.
for(const view of VIEWS.filter(v=>v!=='model'))test(`${view}: all model/hardware identities have valid scenes`,()=>{for(const model of Object.keys(MODELS))for(const hardware of Object.keys(HARDWARE)){const g=buildScene({...DEFAULT_STATE,model,hardware,view});assert.deepEqual(structuralReport(g),[]);}});
test('Kimi global layers do not insert rotary encoding',()=>{const g=buildScene({...DEFAULT_STATE,model:'kimi3',layer:4});assert.ok(!g.nodes.some(n=>/RoPE/.test(n.name)&&!/NoPE/.test(n.name)));assert.ok(g.nodes.some(n=>n.id==='latent-combine'));assert.ok(g.edges.some(e=>e.source==='latent-combine'&&e.target==='latent-up'));});
test('Qwen3 has Q/K head norms but no shared expert',()=>{const g=buildScene({...DEFAULT_STATE,model:'q3235',layer:4});assert.ok(g.nodes.some(n=>n.id==='q-norm'));assert.ok(g.nodes.some(n=>n.id==='k-norm'));assert.ok(!g.nodes.some(n=>n.id==='shared-up'));});
test('QSA has separate full history and block-selection paths',()=>{const g=buildScene({...DEFAULT_STATE,model:'q38flash',layer:4});assert.ok(g.nodes.some(n=>n.id==='kv-cache'&&n.name==='Full K/V history'));assert.ok(g.edges.some(e=>e.source==='index-select'&&e.target==='attention'));});
test('recurrence is explicitly time-delayed',()=>{const g=buildScene({...DEFAULT_STATE,model:'q3827',layer:1});assert.ok(g.edges.some(e=>e.source==='attention'&&e.target==='state-prev'&&e.delay));});
test('shared expert consumes hidden input, never routing scores',()=>{for(const model of ['ds2','ds3','kimi3','q38flash']){const g=buildScene({...DEFAULT_STATE,model,layer:4});assert.ok(g.edges.some(e=>e.source==='norm2'&&e.target==='shared-up'));assert.ok(!g.edges.some(e=>e.source==='router'&&e.target==='shared-up'));}});
test('nonstandard residuals are not rendered as ordinary residual sums',()=>{for(const model of ['kimi3','q38flash']){const g=buildScene({...DEFAULT_STATE,model});assert.notEqual(g.nodes.find(n=>n.id==='merge-attn').kind,'sum');}});
test('input / residual / PLE focus stages are real distinct scenes',()=>{for(const [model,stage]of[['kimi3','inputs'],['kimi3','residual'],['q38flash','ple'],['q38flash','residual']]){const g=buildScene({...DEFAULT_STATE,model,stage});assert.deepEqual(structuralReport(g),[]);assert.ok(!g.nodes.some(n=>n.id==='overview-block'));}});
test('training modes change actual collective semantics',()=>{let f=buildScene({...DEFAULT_STATE,view:'training',trainMode:'fsdp'}),d=buildScene({...DEFAULT_STATE,view:'training',trainMode:'ddp'});assert.match(f.nodes.find(n=>n.id==='gradient-sync').name,/reduce-scatter/);assert.match(d.nodes.find(n=>n.id==='gradient-sync').name,/all-reduce/);});
test('two-line operator boxes reserve at least 32px for fractional-DPR text bounds',()=>{for(const model of Object.keys(MODELS)){const g=buildScene({...DEFAULT_STATE,model,layer:2});for(const n of g.nodes){if(n.sub)assert.ok(n.h>=32,`${model}/${n.id}: ${n.h}px`);}}});
test('the residual stream is labelled as an input above both normalization entries',()=>{for(const id of Object.keys(MODELS)){const g=buildScene({...DEFAULT_STATE,model:id});const norm1=g.nodes.find(n=>n.id==='norm1'),norm2=g.nodes.find(n=>n.id==='norm2');const streams=g.labels.filter(l=>/^residual stream · /.test(l.text)).sort((p,q)=>p.x-q.x);assert.equal(streams.length,2,`${id}: each normalization entry needs a named residual-stream input`);const[a,b]=streams;assert.ok(a.x>=g.groups.find(x=>x.id==='block-region').x&&a.x<norm1.x+norm1.w&&a.y<norm1.y,`${id}: attention entry label is not above the arrival point`);assert.ok(b.x>=norm2.x&&b.x<norm2.x+norm2.w&&b.y<norm2.y,`${id}: feed-forward entry label is not above norm2`);}});
// Revised once the intent was clarified: the stream must read as arriving from OUTSIDE the
// block, not merely as a longer stub. It enters on the block-region boundary and splits there.
test('the residual stream arrives from outside the block and splits into attention and the mainline',()=>{for(const id of Object.keys(MODELS)){const g=buildScene({...DEFAULT_STATE,model:id});const block=g.groups.find(x=>x.id==='block-region');const entry=g.edges.find(e=>!e.source&&e.target==='norm1');assert.ok(entry,`${id}: no residual entry wire into norm1`);assert.equal(entry.points[0][0],block.x,`${id}: entry starts at x=${entry.points[0][0]}, not on the block boundary ${block.x}`);const rail=g.edges.find(e=>!e.source&&e.target==='merge-attn'&&e.kind==='residual');assert.ok(rail,`${id}: no residual mainline`);assert.equal(rail.points[0][1],entry.points[0][1],`${id}: the mainline does not branch from the arrival point`);assert.ok(rail.points[0][0]>block.x&&rail.points[0][0]<entry.points[1][0],`${id}: the mainline branch is not between the boundary and the attention column`);}});
test('every dsv41 layer draws a contract-clean CSA2 block',()=>{for(let layer=1;layer<=MODELS.dsv41.layers;layer++){const g=buildScene({...DEFAULT_STATE,model:'dsv41',view:'model',stage:'block',layer});assert.deepEqual(structuralReport(g),[],`layer ${layer} geometry`);assert.ok(!g.nodes.some(n=>['decay-gate','short-conv','state-prev','gate-functions','head-norm'].includes(n.id)),`layer ${layer} renders recurrence it does not have`);}});
test('dsv41 attention shows two KV paths merging into one call',()=>{const full=buildScene({...DEFAULT_STATE,model:'dsv41',layer:3});const ids=new Set(full.nodes.map(n=>n.id));assert.equal(layerKind('dsv41',3),'csa2-full');assert.ok(ids.has('window-kv'),'the sliding window is on every layer');assert.ok(ids.has('compress-kv'),'a Full-mode layer owns the compressed cache');assert.ok(ids.has('attention'),'one attention call, not two');assert.equal(full.nodes.filter(n=>/^attention/.test(n.id)&&n.id!=='attn-sink'&&n.tone==='compute').length,1,'exactly one attention operator');const intoAttention=full.edges.filter(e=>e.target==='attention');assert.ok(intoAttention.length>=3,'query, window KV and compressed KV all reach it');});
test('an SWA-only dsv41 layer draws no compressed path',()=>{const swa=buildScene({...DEFAULT_STATE,model:'dsv41',layer:1});const ids=new Set(swa.nodes.map(n=>n.id));assert.equal(layerKind('dsv41',1),'swa');assert.ok(ids.has('window-kv'));assert.ok(!ids.has('compress-kv'),'layers 1-2 have compression ratio 0');assert.ok(!ids.has('index-proj'));assert.ok(!ids.has('index-select'));});
test('a dsv41 reuse layer reads a selection instead of computing one',()=>{const g=buildScene({...DEFAULT_STATE,model:'dsv41',layer:4});const ids=new Set(g.nodes.map(n=>n.id));assert.equal(layerKind('dsv41',4),'csa2-reuse');assert.ok(ids.has('compress-kv'));assert.ok(ids.has('index-select'));assert.ok(!ids.has('index-proj'),'a reuse layer runs no indexer projection');assert.match(g.nodes.find(n=>n.id==='index-select').name,/Reuse/);const reindex=buildScene({...DEFAULT_STATE,model:'dsv41',layer:25});assert.equal(layerKind('dsv41',25),'csa2-reindex');assert.ok(reindex.nodes.some(n=>n.id==='index-proj'),'a reindex layer does re-score');});
test('dsv41 states its single latent KV rather than implying per-head KV',()=>{const scene=buildScene({...DEFAULT_STATE,model:'dsv41',layer:3});const kv=scene.nodes.find(n=>n.id==='window-kv');assert.match(kv.sub+kv.name,/1 KV|one latent|single latent/i);assert.equal(MODELS.dsv41.kvHeads,1);});
test('dsv41 names the grouped output projection rather than a dense one',()=>{const o=buildScene({...DEFAULT_STATE,model:'dsv41',layer:3}).nodes.find(n=>n.id==='out-proj');assert.match(o.sub,/8 groups/);assert.match(o.sub,/1,024/);});
test('the mHC residual is not drawn as an ordinary sum',()=>{const scene=buildScene({...DEFAULT_STATE,model:'dsv41',stage:'residual'});assert.deepEqual(structuralReport(scene),[]);const text=JSON.stringify(scene);assert.match(text,/4 streams|four streams/i);assert.match(text,/Sinkhorn/i,'the doubly stochastic mix is the distinguishing property');const block=buildScene({...DEFAULT_STATE,model:'dsv41',layer:3});assert.notEqual(block.nodes.find(n=>n.id==='merge-attn').kind,'sum');assert.match(JSON.stringify(block.labels),/residual stream · 4 streams/);});
test('an unknown layer kind cannot silently render as recurrence',()=>{assert.throws(()=>attentionRenderer('mamba'),RangeError);assert.throws(()=>attentionRenderer(undefined),RangeError);assert.throws(()=>attentionRenderer('csa2-invented'),RangeError);for(const kind of ['mla','gqa','qsa','kda','delta','swa','csa2-full','csa2-reindex','csa2-reuse'])assert.equal(typeof attentionRenderer(kind),'function',kind);});
test('every layer kind core.mjs can produce has a renderer',()=>{for(const id of Object.keys(MODELS))for(let layer=1;layer<=MODELS[id].layers;layer++)assert.equal(typeof attentionRenderer(layerKind(id,layer)),'function',`${id}/${layer}`);});
test('the hierarchical candidate pool is drawn only where the decoder has one',()=>{const m=MODELS.dsv41;for(let layer=1;layer<=m.layers;layer++){const g=buildScene({...DEFAULT_STATE,model:'dsv41',layer});const kind=layerKind('dsv41',layer);const scores=kind==='csa2-full'||kind==='csa2-reindex';const pooled=scores&&layer-1>=m.candidateSource;assert.equal(g.nodes.some(n=>n.id==='index-blocks'),pooled,`layer ${layer} candidate pool`);const text=JSON.stringify(g);
  // Naming the pool to deny it is fine; drawing on it is not. So the assertion is that no
  // unpooled layer states the pool's shape or says a source layer supplies it.
  if(!pooled){assert.ok(!new RegExp(`layer ${m.candidateSource+1} (?:maintains|establishes)`).test(text),`layer ${layer} claims a candidate set supplies it`);assert.ok(!/2,048 × 8/.test(text),`layer ${layer} states the candidate-pool shape`);}
  if(scores)assert.deepEqual(g.nodes.find(n=>n.id==='index-proj').info.facts[0],['Scope',pooled?'candidate pool':'full causal range'],`layer ${layer} indexer scope`);}
 // Encoder Full layers 3, 9 and 15 precede the candidate source and score the full range.
 for(const layer of [3,9,15]){const g=buildScene({...DEFAULT_STATE,model:'dsv41',layer});assert.ok(g.nodes.some(n=>n.id==='index-proj'),`layer ${layer} still scores`);assert.ok(g.edges.some(e=>e.source==='index-proj'&&e.target==='index-select'),`layer ${layer} indexer must reach the selection directly`);assert.match(JSON.stringify(g.nodes.find(n=>n.id==='index-proj').info),/decoder-only mechanism/);}});
test('the CED page is contract-clean and only offered where it applies',()=>{const scene=buildScene({...DEFAULT_STATE,model:'dsv41',view:'model',stage:'ced'});assert.deepEqual(structuralReport(scene),[]);
 // A model without an encoder/decoder split falls back rather than rendering a foreign page.
 const fallback=buildScene({...DEFAULT_STATE,model:'ds3',view:'model',stage:'ced'});assert.notDeepEqual(fallback.nodes.map(n=>n.id),scene.nodes.map(n=>n.id));assert.deepEqual(structuralReport(fallback),[]);
 // This assertion used to read: sanitizeState must return 'ced' for *every* model, because the
 // dispatch and not the clamp decides who renders it. Revisited deliberately. The dispatch does
 // still decide -- it is now an exhaustive table over `focalStages` -- but leaving the stage
 // un-clamped meant `?model=ds3&stage=ced` kept 'ced' in the state while the page drew the
 // decoder block, so `exportConfig` and `getState` recorded a stage the drawing is not. On a
 // branch whose recurring defect is a label measured on the wrong page, that is the same defect
 // in the state object. sanitizeState clamps every other field to what the model admits; the
 // stage is now one of them, which is also what makes the dispatch safe to make total.
 for(const id of Object.keys(MODELS))assert.equal(sanitizeState({model:id,view:'model',stage:'ced'}).stage,MODELS[id].encoderLayers?'ced':'block',`${id} must not hold a stage it cannot render`);
 // The substance the old assertion protected: no model without the split ever gets the page.
 for(const id of Object.keys(MODELS)){const g=buildScene({...DEFAULT_STATE,model:id,view:'model',stage:'ced'});assert.equal(g.nodes.some(n=>n.id==='enc-out'),Boolean(MODELS[id].encoderLayers),`${id} CED page`);}});

test('every focal stage has exactly one builder, and an unoffered page throws rather than drawing another',()=>{
 // The latent defect this closes: `app.mjs` and `diagrams.mjs` gated the N-gram page on
 // `m.id==='q38flash'` while the exporters and two gates derived it from `pleLayers`. A tenth
 // checkpoint with pleLayers and a different id would have been offered no tab while the gates
 // generated `<mid>/model/ple/4`, buildScene fell through to buildModel, and a signature
 // labelled as one page was baselined on another.
 const offered=new Set(Object.keys(MODELS).flatMap(id=>focalStages(id)));
 assert.deepEqual([...offered].sort(),['ced','csa2','decoder','dspark','encoder','engram','inputs','ple','residual'],'every focal stage a model offers');
 for(const id of Object.keys(MODELS)){
  const stages=focalStages(id);
  assert.deepEqual(stages,[...new Set(stages)],`${id} lists a stage twice`);
  for(const stage of stages)assert.deepEqual(structuralReport(buildScene({...DEFAULT_STATE,model:id,view:'model',stage})),[],`${id}/${stage}`);
  // An unoffered stage now cannot survive into the state at all, so the drawing and the state
  // agree instead of the state naming a page the drawing is not. The dispatch's RangeError is
  // defence behind this, unreachable through any caller that sanitizes -- which is every caller,
  // since buildScene sanitizes its own input. "Impossible" rather than "brief", per T2g.
  const block=JSON.stringify(buildScene({...DEFAULT_STATE,model:id,view:'model',stage:'block'}));
  for(const stage of ['inputs','residual','ple','ced'].filter(x=>!stages.includes(x))){
   const s=sanitizeState({model:id,view:'model',stage});
   assert.equal(s.stage,'block',`${id} kept a stage it cannot render`);
   assert.equal(JSON.stringify(buildScene({...DEFAULT_STATE,model:id,view:'model',stage})),block,`${id}/${stage}`);
  }
  // Every offered stage is a genuinely different page, so no builder is a silent alias for the block.
  for(const stage of stages)assert.notEqual(JSON.stringify(buildScene({...DEFAULT_STATE,model:id,view:'model',stage})),block,`${id}/${stage} is the decoder block under another name`);
 }});

test('the N-gram page is offered exactly where the model has n-gram layers',()=>{
 for(const id of Object.keys(MODELS))assert.equal(focalStages(id).includes('ple'),MODELS[id].pleLayers.length>0,id);});
test('the CED page shows a one-way handoff, not a shared stack',()=>{const scene=buildScene({...DEFAULT_STATE,model:'dsv41',view:'model',stage:'ced'});const ids=new Set(scene.nodes.map(n=>n.id));
 for(const id of ['enc-out','dec-full','engram','candidate-pool','dspark'])assert.ok(ids.has(id),`${id} is part of the architecture`);
 const ced=scene.edges.filter(e=>e.source==='enc-out'&&e.target==='dec-full');assert.equal(ced.length,1,'exactly one CED handoff edge');
 const back=scene.edges.filter(e=>/^dec-/.test(e.source||'')&&/^enc-/.test(e.target||''));assert.deepEqual(back,[],'a decoder-to-encoder edge would make this an ordinary shared-KV Transformer');});
// Band spans exactly as the page draws them: the id carries the span, except `dec-full`, which
// keeps a stable id for the handoff edge and states its span in its title instead.
function cedBands(scene){return scene.nodes.filter(n=>/-band-\d+-\d+$/.test(n.id)||n.id==='dec-full')
 .map(n=>({id:n.id,side:n.id.slice(0,3),span:n.id==='dec-full'?n.name.match(/L(\d+)–(\d+)/).slice(1,3).map(Number):n.id.split('-').slice(-2).map(Number)}))
 .sort((a,b)=>a.span[0]-b.span[0]);}
test('the CED bands tile the declared depth, with no layer invented or dropped',()=>{const m=MODELS.dsv41,scene=buildScene({...DEFAULT_STATE,model:'dsv41',view:'model',stage:'ced'});
 const bands=cedBands(scene);assert.ok(bands.some(b=>b.id==='dec-full'),'the decoder Full band is the CED edge target and keeps a stable id');
 const covered=side=>bands.filter(b=>b.side===side).reduce((n,b)=>n+b.span[1]-b.span[0]+1,0);
 assert.equal(covered('enc'),m.encoderLayers,'the encoder bands must cover exactly encoderLayers layers');
 assert.equal(covered('dec'),m.layers-m.encoderLayers,'the decoder bands must cover the rest');
 let next=1;for(const {span:[lo,hi]} of bands){assert.equal(lo,next,`band starts at ${lo}, expected ${next}`);next=hi+1;}
 assert.equal(next,m.layers+1,'the two halves must tile 1..layers with no gap and no overlap');
 // A non-head layer belongs to its band only by reusing what the head published, or by being
 // another layer of the same window-only kind. Nothing else may be folded in.
 for(const {span:[lo,hi]} of bands){const head=layerKind('dsv41',lo);for(let l=lo+1;l<=hi;l++){const k=layerKind('dsv41',l);assert.ok(k==='csa2-reuse'||(k===head&&head==='swa'),`layer ${l} (${k}) is folded into a ${head} band it does not belong to`);}}});
test('the CED fan-outs are derived from the model, not from literals',()=>{const m=MODELS.dsv41,scene=buildScene({...DEFAULT_STATE,model:'dsv41',view:'model',stage:'ced'});
 const out=id=>scene.edges.filter(e=>e.source===id);
 assert.equal(out('engram').length,m.engramLayers.length,'one Engram edge per declared injection site');
 const reindex=m.indexSourceLayers.filter(l=>l>m.candidateSource).length;
 assert.equal(scene.nodes.filter(n=>n.id.startsWith('dec-band-')).length,reindex,'one Reindex band per decoder-band index source');
 assert.equal(out('candidate-pool').length,reindex,'the pool must feed every Reindex band and nothing else');
 assert.equal(scene.edges.filter(e=>e.source==='dec-full'&&e.target==='candidate-pool').length,1,'the pool is sourced from the decoder Full layer');
 // How many bands DSpark reads is the model's business too: one per band holding a target.
 const holding=layers=>new Set(layers.map(l=>cedBands(scene).find(b=>l>=b.span[0]&&l<=b.span[1])?.id));
 const targets=holding(m.dsparkTargets.map(l=>l+1));
 assert.ok(!targets.has(undefined),'every declared DSpark target must land in a drawn band');
 assert.deepEqual(new Set(scene.edges.filter(e=>e.target==='dspark').map(e=>e.source)),targets,'DSpark must read exactly the bands holding its declared target layers');
 const dspark=scene.nodes.find(n=>n.id==='dspark');for(const l of m.dsparkTargets)assert.match(dspark.sub+dspark.info.body,new RegExp(`\\b${l+1}\\b`),`DSpark must name target layer ${l+1}`);
 const engram=scene.nodes.find(n=>n.id==='engram');for(const l of m.engramLayers)assert.match(engram.sub+engram.info.body,new RegExp(`\\b${l+1}\\b`),`Engram must name injection layer ${l+1}`);});
// Both fan-outs search both halves. dsv41 puts every Engram site in the encoder, so without a
// preset that does otherwise, "searches both halves" is an untested claim; buildCED is exported,
// so the declaration can be perturbed directly instead of being trusted.
test('a CED page cannot silently drop a declared site in the other half',()=>{const decoderSite={...MODELS.dsv41,engramLayers:[25]};
 const g=buildCED({...DEFAULT_STATE,model:decoderSite,view:'model',stage:'ced'});
 assert.deepEqual(structuralReport(g),[]);
 assert.deepEqual(g.edges.filter(e=>e.source==='engram').map(e=>e.target),['dec-band-25-28'],'a decoder-band injection site must be drawn, not dropped');
 // The draft head has no honest route to an encoder band, so that is raised rather than drawn.
 assert.throws(()=>buildCED({...DEFAULT_STATE,model:{...MODELS.dsv41,dsparkTargets:[2]},view:'model',stage:'ced'}),RangeError);});
