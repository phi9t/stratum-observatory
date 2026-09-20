import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_STATE,getModel,layerKind} from '../src/core.mjs';
import {buildEncoder,buildDecoder} from '../src/diagram-deepdives.mjs';
import {buildModel} from '../src/diagram-model.mjs';
import {structuralReport} from '../src/scene.mjs';

const m=getModel('dsv41');
const state=(stage,layer)=>({...DEFAULT_STATE,model:m.id,stage,layer});
const nodes=g=>new Map(g.nodes.map(n=>[n.id,n]));
for(const [stage,build,first,last] of [['encoder',buildEncoder,1,m.encoderLayers],['decoder',buildDecoder,m.encoderLayers+1,m.layers]]){
 test(`${stage} chapter preserves the detailed operator graph at every absolute layer`,()=>{
  for(let layer=first;layer<=last;layer++){
   const s=state(stage,layer),g=build(s),reference=buildModel(s),ids=nodes(g);
   for(const id of ['norm1','qkv-proj','window-kv','attention','inverse-rope','out-proj','merge-attn','norm2','router','top-k','dispatch','expert-up','expert-down','shared-up','shared-down','routed-shared','merge-ffn'])assert.deepEqual(ids.get(id),nodes(reference).get(id),`${stage} L${layer}: ${id}`);
   assert.ok(ids.has('attention'));
   for(const [source,target] of [['qkv-proj','qk-position'],['window-kv','attention'],['attention','inverse-rope'],['out-proj','merge-attn'],['merge-attn','norm2'],['norm2','router'],['top-k','dispatch'],['norm2','shared-up'],['shared-down','routed-shared'],['routed-shared','merge-ffn']])assert.ok(g.edges.some(e=>e.source===source&&e.target===target),`${stage} L${layer}: ${source} → ${target}`);
   assert.ok(!g.edges.some(e=>['router','top-k'].includes(e.source)&&e.target==='shared-up'));
   assert.deepEqual(g.edges,reference.edges);
   assert.deepEqual(structuralReport(g),[]);
   assert.deepEqual(g.decorations.filter(d=>d.kind==='layer').map(d=>d.layer),Array.from({length:last-first+1},(_,i)=>first+i));
   assert.equal(ids.get('overview-block').name,`${stage==='encoder'?'Encoder':'Decoder'} stack`);
   assert.match(ids.get('overview-block').sub,new RegExp(`^${last-first+1} layers`));
   assert.deepEqual(g.decorations.filter(d=>d.active).map(d=>d.layer),[layer]);
   if(layerKind(m,layer).startsWith('csa2-')){
    const owner=m.kvSourceLayers.filter(l=>l<layer).at(-1)+1;
    assert.ok(ids.get('compress-kv').info.facts.some(([key,value])=>key==='Owned by'&&value===`layer ${owner}`));
   }
  }
 });
}
test('encoder ends at the decoder boundary and decoder begins at encoder output',()=>{
 const encoder=nodes(buildEncoder(state('encoder',m.encoderLayers)));
 assert.ok(encoder.has('text-input'));assert.ok(encoder.has('vision-input'));
 assert.ok(encoder.has('encoder-handoff'));
 assert.ok(!encoder.has('final-norm'));assert.ok(!encoder.has('lm-head'));
 const decoder=nodes(buildDecoder(state('decoder',m.encoderLayers+1)));
 assert.equal(decoder.get('decoder-handoff')?.name,'Encoder output');
 for(const id of ['text-input','vision-input','embedding','encoder-handoff'])assert.ok(!decoder.has(id),id);
 assert.ok(decoder.has('final-norm'));assert.ok(decoder.has('lm-head'));
});
test('Engram modifies the input residual only at configured absolute layers',()=>{
 for(let layer=1;layer<=m.encoderLayers;layer++){
  const g=buildEncoder(state('encoder',layer)),ids=nodes(g),active=m.engramLayers.includes(layer-1);
  assert.equal(ids.has('engram-input'),active,`L${layer}`);
  if(!active)continue;
  for(const [source,target] of [['engram-history','engram-input'],['engram-residual','engram-input'],['engram-input','norm1'],['engram-input','merge-attn']])assert.ok(g.edges.some(e=>e.source===source&&e.target===target),`${source} → ${target}`);
  assert.match(ids.get('engram-history').info.body,/token history/i);
  assert.match(ids.get('engram-input').info.body,/gate/i);
  assert.match(ids.get('engram-input').info.formula,/gate.*value/);
  assert.ok(!g.edges.some(e=>e.source==='engram-residual'&&e.target==='engram-history'));
  assert.deepEqual(structuralReport(g),[]);
 }
});
test('decoder collapses residual copies before final normalization and vocabulary projection',()=>{
 const g=buildDecoder(state('decoder',m.layers)),ids=nodes(g);
 for(const [source,target] of [['overview-block','residual-collapse'],['residual-collapse','final-norm'],['final-norm','lm-head']])assert.ok(g.edges.some(e=>e.source===source&&e.target===target),`${source} → ${target}`);
 assert.match(ids.get('residual-collapse').info.formula,/pre/);
 assert.doesNotMatch(ids.get('residual-collapse').info.formula,/mean|\/\s*4/);
 assert.ok(!nodes(buildEncoder(state('encoder',m.encoderLayers))).has('residual-collapse'));
});
