import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS,DEFAULT_STATE,focalStages,sanitizeState} from '../src/core.mjs';
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
 const encoder=scene('encoder'),decoder=scene('decoder');assert.ok(encoder.nodes.some(n=>n.id==='encoder-causal'));
 assert.ok(decoder.edges.some(e=>e.source==='decoder-handoff'&&e.target==='decoder-producer'));
 assert.ok(decoder.edges.some(e=>e.source==='decoder-producer'&&e.target==='decoder-readers'));
 assert.match(decoder.caption,/reference|report/i);
});
test('Engram separates token lookup from contextual gating',()=>{
 const g=scene('engram');for(const [a,b] of [['engram-history','engram-hash'],['engram-hash','engram-table'],['engram-table','engram-project'],['engram-context','engram-gate'],['engram-project','engram-gate'],['engram-gate','engram-write']])assert.ok(g.edges.some(e=>e.source===a&&e.target===b),`${a} -> ${b}`);
});
test('CSA2 exposes distinct write, rescore and reuse responsibilities',()=>{
 const g=scene('csa2');for(const id of ['csa2-full','csa2-reindex','csa2-reuse','csa2-window','csa2-union'])assert.ok(g.nodes.some(n=>n.id===id),id);
 assert.ok(g.edges.some(e=>e.source==='csa2-window'&&e.target==='csa2-union'));
});
test('dSpark exposes draft feature inputs and the missing coordinator boundary',()=>{
 const g=scene('dspark');for(const id of ['dspark-features','dspark-seed','dspark-draft','dspark-markov','dspark-confidence','dspark-verification'])assert.ok(g.nodes.some(n=>n.id===id),id);
 const boundary=g.nodes.find(n=>n.id==='dspark-verification');assert.match(boundary.info.body,/not (provided|implemented)|unavailable|missing/i);
 assert.match(g.caption,/unpublished|excluded/i);assert.ok(!g.nodes.some(n=>/speedup|acceptance.rate/i.test(n.name)));
});
