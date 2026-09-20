import test from 'node:test';
import assert from 'node:assert/strict';
import {PYTORCH_EXAMPLES,moduleStage,moduleExample} from '../src/pytorch.mjs';
import {focalStages,sanitizeState} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {structuralReport} from '../src/scene.mjs';

test('each executable example has its own reachable source-linked schematic',()=>{
 const scenes=[];
 for(const example of PYTORCH_EXAMPLES){
  const stage=moduleStage(example.id);
  assert.ok(focalStages('dsv41').includes(stage),example.id);
  assert.equal(moduleExample(stage),example);
  assert.equal(sanitizeState({model:'dsv41',stage}).stage,stage);
  const scene=buildScene({model:'dsv41',stage});
  assert.deepEqual(structuralReport(scene),[],example.id);
  assert.ok(scene.nodes.length>=5,example.id);
  assert.ok(scene.nodes.every(n=>n.info?.sources.length),example.id);
  assert.ok(scene.nodes.some(n=>n.info.formula),example.id);
  scenes.push(JSON.stringify(scene));
  assert.equal(sanitizeState({model:'ds3',stage}).stage,'block');
 }
 assert.equal(new Set(scenes).size,PYTORCH_EXAMPLES.length);
});
test('unknown module cannot masquerade as a supported schematic',()=>{
 assert.equal(moduleExample('module-does-not-exist'),null);
 assert.equal(sanitizeState({model:'dsv41',stage:'module-does-not-exist'}).stage,'block');
 assert.throws(()=>moduleStage('does-not-exist'),RangeError);
});
