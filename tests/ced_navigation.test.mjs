import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS,sanitizeState,layerNumbers} from '../src/core.mjs';
test('CED chapters restrict absolute layers and legacy block links preserve layer identity',()=>{
 const m=MODELS.dsv41;
 for(let layer=1;layer<=m.layers;layer++){
  const s=sanitizeState({model:m.id,view:'model',stage:'block',layer});
  assert.equal(s.layer,layer);assert.equal(s.stage,layer<=m.encoderLayers?'encoder':'decoder');
 }
 assert.deepEqual(layerNumbers(m,'encoder'),Array.from({length:m.encoderLayers},(_,i)=>i+1));
 assert.deepEqual(layerNumbers(m,'decoder'),Array.from({length:m.layers-m.encoderLayers},(_,i)=>m.encoderLayers+i+1));
 assert.equal(sanitizeState({model:m.id,stage:'decoder',layer:3}).layer,m.encoderLayers+1);
 assert.equal(sanitizeState({model:m.id,stage:'encoder',layer:m.layers}).layer,m.encoderLayers);
 assert.equal(sanitizeState({model:'ds3',stage:'block',layer:3}).stage,'block');
});
