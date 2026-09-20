import test from 'node:test';
import assert from 'node:assert/strict';
import {MODELS} from '../src/models.mjs';
import {BACKBONE_EXAMPLES} from '../src/pytorch-backbone.mjs';
import {BACKBONE_SCHEMATICS} from '../src/diagram-pytorch-backbone.mjs';
import {structuralReport} from '../src/scene.mjs';
const scene=id=>BACKBONE_SCHEMATICS[id](MODELS.dsv41,BACKBONE_EXAMPLES.find(e=>e.id===id));
const edge=(g,a,b)=>g.edges.some(e=>e.source===a&&e.target===b);
for(const example of BACKBONE_EXAMPLES)test(`${example.id} has contained, inspectable branching geometry`,()=>{
 const g=scene(example.id);
 assert.equal(g.width,1440);assert.deepEqual(structuralReport(g),[]);
 for(const n of g.nodes){assert.ok(n.w>=300);assert.ok(n.info.formula);assert.ok(n.info.facts.length);assert.deepEqual(n.info.sources,example.sources);}
 assert.ok(g.nodes.some(n=>g.edges.filter(e=>e.source===n.id).length>=3));
 for(const e of g.edges)for(const [x,y]of e.points){assert.ok(x>=0&&x<=g.width);assert.ok(y>=0&&y<=g.height);}
});
test('mHC reads previous coefficients, preserves input/output matrix axes and observes a flat boundary',()=>{
 const g=scene('backbone-mhc');
 assert.ok(edge(g,'initial-premix','attn-collapse'));
 assert.ok(edge(g,'attn-next','ffn-collapse'));
 assert.ok(!edge(g,'attn-next','attn-collapse'));
 assert.ok(edge(g,'ffn-next','final-collapse'));
 assert.ok(edge(g,'attn-residual','attn-carried'));
 assert.ok(edge(g,'attn-matrix','attn-carried'));
 assert.match(g.nodes.find(n=>n.id==='attn-carried').info.formula,/btij,btid.*btjd/);
 assert.match(g.nodes.find(n=>n.id==='boundary').info.body,/observation/);
 assert.ok(!g.nodes.some(n=>/cross.attention/i.test(n.name)));
 assert.ok(edge(g,'boundary','return'));assert.ok(edge(g,'output-norm','return'));
 assert.match(g.nodes.find(n=>n.id==='return').info.formula,/normalized, boundary/);
});
test('MoE selection bias cannot reach mixture weights without selected indices; shared bypasses routing',()=>{
 const g=scene('backbone-moe');
 assert.ok(edge(g,'selection-bias','select'));
 assert.ok(!edge(g,'selection-bias','weights'));
 assert.ok(edge(g,'scores','weights'));assert.ok(edge(g,'select','weights'));
 assert.ok(edge(g,'tokens','shared-input'));assert.ok(!edge(g,'select','shared-input'));
 assert.ok(edge(g,'weights','routed-product'));assert.ok(!edge(g,'weights','shared-product'));
 assert.ok(edge(g,'routed-product','routed-down'));assert.ok(edge(g,'routed-down','scatter'));
 assert.ok(edge(g,'shared-down','output'));assert.ok(edge(g,'scatter','output'));
});
for(const example of BACKBONE_EXAMPLES)test(`${example.id} routes avoid unrelated operations`,()=>{
 const g=scene(example.id);
 for(const [index,n]of g.nodes.entries())for(const other of g.nodes.slice(index+1)){
  assert.ok(n.x+n.w<=other.x||other.x+other.w<=n.x||n.y+n.h<=other.y||other.y+other.h<=n.y,`${n.id} overlaps ${other.id}`);
 }
 for(const w of g.edges)for(let i=1;i<w.points.length;i++){
  const a=w.points[i-1],b=w.points[i];
  for(const n of g.nodes){
   if(n.id===w.source||n.id===w.target)continue;
   const hit=a[0]===b[0]
    ?a[0]>n.x&&a[0]<n.x+n.w&&Math.max(a[1],b[1])>n.y&&Math.min(a[1],b[1])<n.y+n.h
    :a[1]>n.y&&a[1]<n.y+n.h&&Math.max(a[0],b[0])>n.x&&Math.min(a[0],b[0])<n.x+n.w;
   assert.ok(!hit,`${w.source} -> ${w.target} crosses ${n.id}`);
  }
 }
});
test('backbone inspector dimensions follow the supplied model',()=>{
 const m={...MODELS.dsv41,d:37,streams:7,experts:13,topK:3,expertFF:19};
 const mhc=BACKBONE_SCHEMATICS['backbone-mhc'](m,BACKBONE_EXAMPLES[0]);
 const moe=BACKBONE_SCHEMATICS['backbone-moe'](m,BACKBONE_EXAMPLES[1]);
 const fact=(g,id,key)=>g.nodes.find(n=>n.id===id).info.facts.find(([k])=>k===key)[1];
 assert.equal(fact(mhc,'embedded','hidden_dim'),'37');
 assert.equal(fact(mhc,'attn-mixer','Coefficient count'),'63');
 assert.equal(fact(moe,'scores','experts'),'13');assert.equal(fact(moe,'select','topk'),'3');
 assert.equal(fact(moe,'routed-gate','hidden_width'),'19');
});
