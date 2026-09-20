import test from 'node:test';
import assert from 'node:assert/strict';
import {MEMORY_SCHEMATICS} from '../src/diagram-pytorch-memory.mjs';
import {MEMORY_EXAMPLES} from '../src/pytorch-memory.mjs';
import {MODELS} from '../src/models.mjs';
import {structuralReport} from '../src/scene.mjs';
const scene=id=>MEMORY_SCHEMATICS[id](MODELS.dsv41,MEMORY_EXAMPLES.find(e=>e.id===id));
const edge=(g,a,b)=>g.edges.some(e=>e.source===a&&e.target===b);
for(const example of MEMORY_EXAMPLES)test(`${example.id}: contained graph with source-bound inspectors`,()=>{
 const g=scene(example.id);assert.deepEqual(structuralReport(g),[]);assert.equal(g.width,1440);
 assert.ok(g.nodes.length>=10);assert.ok(g.nodes.every(n=>n.w>=300&&n.info.formula&&n.info.facts.length&&n.info.sources===example.sources));
});
test('Engram keeps shared value outside the stream gate and bypasses the residual',()=>{
 const g=scene('engram-write');assert.ok(edge(g,'project','keys')&&edge(g,'project','value'));
 assert.ok(edge(g,'value','write')&&edge(g,'residual','write'));
 assert.ok(!edge(g,'value','score'));assert.ok(edge(g,'token-mask','gate'));
 assert.match(g.nodes.find(n=>n.id==='score').info.formula,/sum_D/);
});
test('compression direct path has no pooling gate; unfinished state bypasses normalization',()=>{
 const g=scene('csa2-compression');assert.ok(edge(g,'direct-project','direct-norm'));
 assert.ok(!edge(g,'direct-project','pool'));assert.ok(edge(g,'complete','next-state'));
 assert.ok(!edge(g,'next-state','group-norm'));assert.ok(g.edges.some(e=>e.source==='next-state'&&e.target==='previous-state'&&e.kind==='delay'&&e.delay));
 assert.match(g.nodes.find(n=>n.id==='pool').info.formula,/softmax_r/);
});
test('selection preserves candidate block maximum, newest block and invalid sentinel',()=>{
 const g=scene('csa2-selection');assert.ok(edge(g,'block-max','newest-block')&&edge(g,'candidate-mask','allowed'));
 assert.ok(edge(g,'scores','allowed'));assert.match(g.nodes.find(n=>n.id==='selected').info.formula,/-1/);
 assert.ok(!g.nodes.some(n=>/softmax/i.test(n.name)));
});
test('attention has one normalization over both caches and no sink value edge',()=>{
 const g=scene('csa2-attention');assert.ok(edge(g,'local-cache','cache')&&edge(g,'compressed-cache','cache'));
 assert.ok(edge(g,'sink','softmax')&&!edge(g,'sink','weighted-sum'));
 assert.ok(edge(g,'gather','weighted-sum'));assert.match(g.nodes.find(n=>n.id==='softmax').info.formula,/sink/);
 assert.equal(g.nodes.filter(n=>/softmax/i.test(n.name)).length,1);
});
test('attention index inputs and validity outputs occupy separate wire lanes',()=>{
 const g=scene('csa2-attention');
 const input=g.edges.find(e=>e.source==='compressed-indices'&&e.target==='indices');
 const output=g.edges.find(e=>e.source==='indices'&&e.target==='logits');
 for(let i=1;i<input.points.length;i++)for(let j=1;j<output.points.length;j++){
  const [a,b]=input.points.slice(i-1,i+1),[c,d]=output.points.slice(j-1,j+1);
  for(const axis of [0,1]){
   const along=1-axis;
   if(a[axis]!==b[axis]||c[axis]!==d[axis]||a[axis]!==c[axis])continue;
   const overlap=Math.min(Math.max(a[along],b[along]),Math.max(c[along],d[along]))-Math.max(Math.min(a[along],b[along]),Math.min(c[along],d[along]));
   assert.ok(overlap<=0,'incoming compressed indices must not overlap outgoing validity');
  }
 }
});
