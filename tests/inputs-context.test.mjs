import test from 'node:test';
import assert from 'node:assert/strict';
import {buildScene} from '../src/diagrams.mjs';
import {DEFAULT_STATE,getModel} from '../src/core.mjs';
import {structuralReport} from '../src/scene.mjs';
import {chapterContent} from '../src/deepdives.mjs';
test('V4.1 inputs exposes vision block internals and distinct token/Engram paths',()=>{
 const m=getModel('dsv41'),g=buildScene({...DEFAULT_STATE,model:'dsv41',stage:'inputs'});
 assert.deepEqual(structuralReport(g),[]);
 const n=Object.fromEntries(g.nodes.map(n=>[n.id,n]));
 for(const id of ['tokenizer','text-embeddings','vision-qkv','vision-rope','vision-attention','vision-ffn','vision-merge','interleave','engram-remap','engram-hash','engram-inject','encoder-entry'])assert.ok(n[id],id);
 assert.match(n['vision-merge'].sub,/9216/);
 for(const [a,b] of [['tokenizer','expanded-ids'],['expanded-ids','text-embeddings'],['expanded-ids','engram-remap'],['encoder-entry','engram-inject'],['vision-merge','interleave'],['engram-inject','encoder-entry']])assert.ok(g.edges.some(e=>e.source===a&&e.target===b),`${a} → ${b}`);
 assert.match(n['encoder-entry'].sub,/1–20/);
 assert.match(g.groups.find(x=>x.id==='vision-block').title,new RegExp(String(m.visionLayers)));
 assert.ok(chapterContent('dsv41',{...DEFAULT_STATE,stage:'inputs'}).math.length>=3);
});
