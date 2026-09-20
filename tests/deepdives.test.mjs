import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS,DEFAULT_STATE,focalStages,sanitizeState} from '../src/core.mjs';
import {chapterContent} from '../src/deepdives.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {structuralReport} from '../src/scene.mjs';
const chapters=['encoder','decoder','engram','csa2','dspark'];
const scene=stage=>buildScene({...DEFAULT_STATE,model:'dsv41',view:'model',stage});
test('V4.1 offers independent mechanism chapters and other checkpoints reject them',()=>{
 for(const stage of chapters){assert.ok(focalStages('dsv41').includes(stage),stage);assert.equal(sanitizeState({model:'dsv41',stage}).stage,stage);
  for(const model of Object.keys(MODELS).filter(m=>m!=='dsv41'))assert.equal(sanitizeState({model,stage}).stage,'block');
 }
});
test('every deep dive is distinct, connected, inspectable and contract-clean',()=>{
 const signatures=new Set();for(const stage of chapters){const g=scene(stage);assert.deepEqual(structuralReport(g),[]);assert.ok(g.nodes.length>=6);assert.ok(g.edges.length>=5);signatures.add(g.nodes.map(n=>n.id).join(','));
  for(const n of g.nodes){assert.ok(n.info?.body,n.id);assert.ok(n.info.sources.length,n.id);}
 }assert.equal(signatures.size,chapters.length);
});
test('encoder and decoder distinguish token causality and compressed-state ownership',()=>{
 const encoder=scene('encoder'),decoder=scene('decoder');
 for(const g of [encoder,decoder])for(const id of ['norm1','attention','router','out-proj','merge-ffn'])assert.ok(g.nodes.some(n=>n.id===id));
 assert.deepEqual(encoder.decorations.filter(d=>d.kind==='layer').map(d=>d.layer),Array.from({length:MODELS.dsv41.encoderLayers},(_,i)=>i+1));
 assert.ok(decoder.decorations.every(d=>d.kind!=='layer'||d.layer>MODELS.dsv41.encoderLayers));
 assert.match(decoder.caption,/reference|report/i);
});
test('Engram separates token lookup from contextual gating',()=>{
 const g=scene('engram');for(const [a,b] of [['engram-history','engram-hash'],['engram-hash','engram-table'],['engram-table','engram-project'],['engram-context','engram-gate'],['engram-project','engram-gate'],['engram-gate','engram-write']])assert.ok(g.edges.some(e=>e.source===a&&e.target===b),`${a} -> ${b}`);
});
test('CSA2 exposes distinct write, rescore and reuse responsibilities',()=>{
 const g=scene('csa2');for(const id of ['csa2-full','csa2-reindex','csa2-reuse','csa2-window','csa2-union'])assert.ok(g.nodes.some(n=>n.id===id),id);
 assert.ok(g.edges.some(e=>e.source==='csa2-window'&&e.target==='csa2-union'));
});
test('dSpark keeps Pro architecture, Flash mapping and DeepSpec verification distinct',()=>{
 const g=scene('dspark');for(const id of ['dspark-features','dspark-seed','dspark-draft','dspark-markov','dspark-confidence','dspark-verification','dspark-predecessors'])assert.ok(g.nodes.some(n=>n.id===id),id);
 const boundary=g.nodes.find(n=>n.id==='dspark-verification');assert.match(boundary.info.body,/DeepSpec/);
 const c=chapterContent('dsv41',{...DEFAULT_STATE,stage:'dspark'});
 for(const id of ['dspark-paper','deepspec-verification','deepspec-draft','dsv4-dspark-card'])assert.ok(c.sources.includes(id),id);
 assert.ok(c.trace.some(([title,body])=>title.includes('retains')&&body.includes('crops speculative state')));
 assert.match(g.caption,/Pro.*Flash/);assert.ok(!g.nodes.some(n=>/speedup|acceptance.rate/i.test(n.name)));
 assert.ok(!g.edges.some(e=>e.source==='dspark-sample-5'&&e.target==='dspark-confidence'));
});
