import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS,DEFAULT_STATE,focalStages} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {chapterContent,DEEP_DIVE_TITLES} from '../src/deepdives.mjs';

test('tensor dimensions never contain thousands separators',()=>{
 const strings=value=>typeof value==='string'?[value]:Array.isArray(value)?value.flatMap(strings):value&&typeof value==='object'?Object.values(value).flatMap(strings):[];
 const check=value=>{for(const text of strings(value))for(const bracket of text.matchAll(/\[[^\]\n]*\]/g))assert.doesNotMatch(bracket[0],/\d,\d{3}(?=\D|$)/,text);};
 for(const model of Object.keys(MODELS))for(const stage of ['block',...focalStages(model)])check(buildScene({...DEFAULT_STATE,model,stage}));
 for(const stage of Object.keys(DEEP_DIVE_TITLES))check(chapterContent('dsv41',{...DEFAULT_STATE,model:'dsv41',stage}));
});

test('V4.1 chapter tiles are grouped by producer and reuse bundle',()=>{
 for(const [stage,ranges] of Object.entries({encoder:[[1,2],[3,8],[9,14],[15,20]],decoder:[[21,24],[25,28],[29,32],[33,36],[37,40]]})){
  for(const selected of ranges.flatMap(([lo,hi])=>Array.from({length:hi-lo+1},(_,i)=>lo+i))){
   const scene=buildScene({...DEFAULT_STATE,model:'dsv41',stage,layer:selected});
   const groups=scene.groups.filter(g=>g.id.startsWith('layer-bundle-'));
   assert.equal(groups.length,ranges.length);
   for(const [i,[lo,hi]] of ranges.entries()){
    const group=groups[i];assert.equal(group.id,`layer-bundle-${lo}-${hi}`);
    const tiles=scene.decorations.filter(d=>d.kind==='layer'&&d.x>=group.x&&d.y>=group.y&&d.x+d.w<=group.x+group.w&&d.y+d.h<=group.y+group.h);
    assert.deepEqual(tiles.map(t=>t.layer),Array.from({length:hi-lo+1},(_,n)=>lo+n));
   }
   assert.deepEqual(scene.decorations.filter(t=>t.active).map(t=>t.layer),[selected]);
  }
 }
});
