import test from 'node:test';
import assert from 'node:assert/strict';
import {execFileSync} from 'node:child_process';
import {DSPARK_SCHEMATICS} from '../src/diagram-pytorch-dspark.mjs';
import {DSPARK_EXAMPLES} from '../src/pytorch-dspark.mjs';
import {MODELS} from '../src/models.mjs';
import {structuralReport} from '../src/scene.mjs';
const scene=id=>DSPARK_SCHEMATICS[id](MODELS.dsv41,DSPARK_EXAMPLES.find(e=>e.id===id));
const edge=(g,a,b)=>g.edges.some(e=>e.source===a&&e.target===b);
for(const example of DSPARK_EXAMPLES.filter(e=>e.id!=='dspark-pro-block'))test(`${example.id}: inspectable orthogonal module schematic`,()=>{
 const g=scene(example.id);assert.deepEqual(structuralReport(g),[]);assert.equal(g.width,1440);
 assert.ok(g.nodes.length>=10);assert.ok(g.nodes.every(n=>n.w>=300&&n.info.formula&&n.info.facts.length&&n.info.sources.length));
 for(const symbol of example.symbols)assert.ok(g.nodes.some(n=>n.info.body.includes(symbol)),symbol);
});
test('draft attention sink normalizes but never supplies values; no causal triangle',()=>{
 const g=scene('dspark-attention');assert.ok(edge(g,'sink','normalize'));assert.ok(edge(g,'key-values','weighted-values'));
 assert.ok(!edge(g,'sink','weighted-values'));assert.match(g.nodes.find(n=>n.id==='visibility').info.body,/every draft slot/);
 assert.ok(edge(g,'history','key-values')&&edge(g,'draft-kv','key-values'));
});
test('Markov sampling feeds the next position; confidence bypasses final norm',()=>{
 const g=scene('dspark-heads');assert.ok(g.edges.some(e=>e.source==='sample'&&e.target==='previous-token'&&e.delay));
 assert.ok(edge(g,'hidden','confidence-features'));assert.ok(edge(g,'previous-embedding','embedding-stack'));
 assert.ok(!edge(g,'probabilities','raw-confidence'));assert.ok(!g.nodes.some(n=>/sigmoid/i.test(n.name)));
});
test('DeepSpec explicitly separates rejection, bonus and cache ownership',()=>{
 const g=scene('deepspec-verifier');assert.match(g.caption,/offline.*Qwen3.*Gemma4/);
 for(const[a,b]of[['prefix','branch'],['branch','residual'],['branch','bonus'],['residual','fallback'],['fallback','residual-normalize'],['residual-normalize','next-token'],['bonus','next-token'],['accepted','target-crop'],['accepted','feature-crop']])assert.ok(edge(g,a,b),`${a} -> ${b}`);
 assert.ok(!edge(g,'next-token','target-crop'));assert.match(g.nodes.find(n=>n.id==='draft-crop').info.formula,/start/);
 assert.match(g.nodes.find(n=>n.id==='target-crop').info.formula,/start \+ accepted \+ 1/);
});
test('routing rails do not pass through unrelated operators',()=>{
 for(const example of DSPARK_EXAMPLES.filter(e=>e.id!=='dspark-pro-block')){const g=scene(example.id);
  for(const edge of g.edges)for(const node of g.nodes){
   if([edge.source,edge.target].includes(node.id))continue;
   for(let i=1;i<edge.points.length;i++){
    const[a,b]=[edge.points[i-1],edge.points[i]];
    const intersects=a[0]===b[0]
     ? a[0]>node.x&&a[0]<node.x+node.w&&Math.min(a[1],b[1])<node.y+node.h&&Math.max(a[1],b[1])>node.y
     : a[1]>node.y&&a[1]<node.y+node.h&&Math.min(a[0],b[0])<node.x+node.w&&Math.max(a[0],b[0])>node.x;
    assert.ok(!intersects,`${example.id}: ${edge.source} → ${edge.target} intersects ${node.id}`);
   }
  }
 }
});

test('DeepSpec stays in its own offline category and distinguishes ephemeral inputs',()=>{
 const g=scene('deepspec-verifier');
 assert.ok(g.nodes.every(n=>n.info.category==='DeepSpec offline evaluator'));
 for(const id of ['confidence','uniforms'])assert.equal(g.nodes.find(n=>n.id===id).tone,'neutral');
 assert.equal(g.nodes.find(n=>n.id==='draft-forward').tone,'compute');
});
test('DeepSpec passes the production lane and crossing contracts',()=>{
 const script=`import json,sys
from verification.aesthetics import count_crossings
from verification.schematics import judge_lanes
scene=json.load(sys.stdin)
assert count_crossings(scene['edges']) <= 8
assert all(f.status == 'PASS' for f in judge_lanes(scene))`;
 execFileSync('uv',['run','python','-c',script],{input:JSON.stringify(scene('deepspec-verifier')),stdio:['pipe','pipe','pipe']});
});
