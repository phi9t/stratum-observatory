import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_STATE} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {chapterContent} from '../src/deepdives.mjs';
import {structuralReport} from '../src/scene.mjs';
const state=stage=>({...DEFAULT_STATE,model:'dsv41',view:'model',stage});
test('decoder retains the bounded-replay mathematical explanation alongside the detailed block',()=>{
 const c=chapterContent('dsv41',state('decoder'));
 assert.ok(c.trace.some(([title,body])=>/replay/i.test(title)&&/approximate/.test(body)&&/local-only/.test(body)&&/W − 1/.test(body)));
});
test('CSA2 candidate selection distinguishes position scores, block maxima and final position selection',()=>{
 const g=buildScene(state('csa2'));
 const candidate=g.nodes.find(n=>n.id==='csa2-candidates');
 assert.match(candidate.info.body,/maximum/);
 assert.match(candidate.info.formula,/position.*max.*candidate/i);
 assert.match(g.nodes.find(n=>n.id==='csa2-select').info.body,/position scores/);
 assert.deepEqual(structuralReport(g),[]);
});
