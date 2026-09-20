import test from 'node:test';
import assert from 'node:assert/strict';
import {buildScene} from '../src/diagrams.mjs';
import {DEFAULT_STATE} from '../src/core.mjs';
import {chapterContent} from '../src/deepdives.mjs';
import {structuralReport} from '../src/scene.mjs';
test('dSpark expands the released Pro block and separates parallel slots from sequential token decisions',()=>{
 const g=buildScene({...DEFAULT_STATE,model:'dsv41',stage:'dspark'}),n=Object.fromEntries(g.nodes.map(n=>[n.id,n]));
 assert.deepEqual(structuralReport(g),[]);
 for(const id of ['dspark-context','dspark-expand','dspark-query','dspark-history','dspark-attention','dspark-output','dspark-moe','dspark-collapse','dspark-base','dspark-markov','dspark-confidence'])assert.ok(n[id],id);
 assert.match(n['dspark-query'].sub,/128.*512/);
 assert.match(n['dspark-attention'].info.body,/every draft slot/);
 assert.match(n['dspark-context'].info.body,/after/);
 for(const[a,b]of [['dspark-context','dspark-history'],['dspark-seed','dspark-markov'],['dspark-collapse','dspark-confidence']])assert.ok(g.edges.some(e=>e.source===a&&e.target===b),`${a} → ${b}`);
 const c=chapterContent('dsv41',{...DEFAULT_STATE,stage:'dspark'});
 assert.match(c.summary,/V4-Pro/);assert.ok(c.math.length>=5);assert.ok(c.sources.includes('dspark-paper'));assert.ok(c.sources.includes('dsv4-dspark-config'));
});
