import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_STATE} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {chapterContent} from '../src/deepdives.mjs';
import {structuralReport} from '../src/scene.mjs';
const state=stage=>({...DEFAULT_STATE,model:'dsv41',view:'model',stage});
test('decoder local-only illustration preserves the full token-by-layer dependency cone',()=>{
 const g=buildScene(state('decoder'));
 for(const [a,b] of [['0-0','1-1'],['0-1','1-1'],['0-1','1-2'],['0-2','1-2'],['1-1','2-2'],['1-2','2-2']])
  assert.ok(g.edges.some(e=>e.source===`replay-${a}`&&e.target===`replay-${b}`),`${a} → ${b}`);
 assert.deepEqual(structuralReport(g),[]);
 const boundary=g.nodes.find(n=>n.id==='replay-truncation');
 assert.match(boundary.info.body,/t − 2/);
 assert.match(boundary.info.formula,/D.*W.*1/);
 assert.ok(!g.edges.some(e=>e.source==='decoder-producer'&&e.target.startsWith('replay-')),'global KV is not a missing local-state reconstruction edge');
 const c=chapterContent('dsv41',state('decoder'));
 assert.ok(c.trace.some(([title,body])=>/replay/i.test(title)&&/approximate/.test(body)&&/local-only/.test(body)));
});
test('CSA2 candidate selection distinguishes position scores, block maxima and final position selection',()=>{
 const g=buildScene(state('csa2'));
 const candidate=g.nodes.find(n=>n.id==='csa2-candidates');
 assert.match(candidate.info.body,/maximum/);
 assert.match(candidate.info.formula,/position.*max.*candidate/i);
 assert.match(g.nodes.find(n=>n.id==='csa2-select').info.body,/position scores/);
 assert.deepEqual(structuralReport(g),[]);
});
