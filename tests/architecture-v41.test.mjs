import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {structuralReport} from '../src/scene.mjs';

const facts=n=>Object.fromEntries(n.info.facts);
test('every V4.1 block restores the attention output position basis before projection',()=>{
 for(let layer=1;layer<=MODELS.dsv41.layers;layer++){
  const scene=buildScene({model:'dsv41',layer});
  assert.deepEqual(structuralReport(scene),[]);
  const inverse=scene.nodes.find(n=>n.id==='inverse-rope');
  assert.ok(inverse,'shared post-RoPE KV requires inverse RoPE on the output');
  assert.equal(facts(inverse)['Rotary channels'],String(MODELS.dsv41.ropeDim));
  assert.ok(scene.edges.some(e=>e.source==='attention'&&e.target==='inverse-rope'));
  assert.ok(scene.edges.some(e=>e.source==='inverse-rope'&&e.target==='out-proj'));
  assert.ok(!scene.edges.some(e=>e.source==='attention'&&e.target==='out-proj'),'projection cannot bypass inverse RoPE');
 }
});

test('mHC read collapses the stream axis with a vector and write retains separate gates',()=>{
 const m=MODELS.dsv41,scene=buildScene({model:m.id,stage:'residual'});
 assert.deepEqual(structuralReport(scene),[]);
 const read=facts(scene.nodes.find(n=>n.id==='depth-read'));
 assert.equal(read['Read coefficients'],`[B, S, ${m.streams}]`);
 assert.equal(read['Output'],`[B, S, ${m.d}]`);
 const write=facts(scene.nodes.find(n=>n.id==='depth-write'));
 assert.equal(write['Residual mix'],`[B, S, ${m.streams}, ${m.streams}]`);
 assert.equal(write['Output gate'],`[B, S, ${m.streams}]`);
 assert.equal(write['Coefficient owner'],'current sublayer');
});

test('mHC block inspectors distinguish incoming read vector from current write coefficients',()=>{
 const scene=buildScene({model:'dsv41',layer:3});
 for(const id of ['norm1','norm2'])assert.equal(facts(scene.nodes.find(n=>n.id===id))['Read coefficients'],'previous sublayer vector');
 for(const id of ['merge-attn','merge-ffn']){
  const f=facts(scene.nodes.find(n=>n.id===id));
  assert.equal(f['Output gate'],'current sublayer vector');
  assert.equal(f['Residual mix'],'current sublayer matrix');
 }
});

test('comparison preserves mHC identity on either side of the same-scenario comparison',()=>{
 for(const side of ['a','b']){
  const scene=buildScene({view:'compare',model:side==='a'?'dsv41':'ds3',compare:side==='b'?'dsv41':'ds3'});
  const residual=scene.nodes.find(n=>n.id===`${side}-residual`);
  assert.ok(residual);
  assert.equal(residual.sub,`${MODELS.dsv41.streams}-stream mHC`);
  assert.deepEqual(structuralReport(scene),[]);
 }
});
