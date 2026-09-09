import test from 'node:test';
import assert from 'node:assert/strict';
import {DEFAULT_STATE,layerKind,SOURCES} from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
const make=(p={})=>buildScene({...DEFAULT_STATE,...p});
test('KDA scene has actual recurrence and a depth-source bank',()=>{const ids=make().nodes.map(n=>n.id);for(const id of ['depth-bank','kda-core','kda-state','attnres-a','attnres-f','latent-down','routed-norm','latent-up'])assert.ok(ids.includes(id),id);});
test('MLA is NoPE and the actual final decoder layer is global',()=>{const g=make({layer:93});assert.ok(g.nodes.some(n=>n.id==='mla-core'));assert.ok(g.nodes.some(n=>n.name.includes('NoPE')));assert.ok(!g.nodes.some(n=>n.name==='RoPE'));});
test('only the first layer uses dense SiTU FFN',()=>{assert.ok(make({layer:1}).nodes.some(n=>n.id==='dense-up'));assert.ok(!make({layer:2}).nodes.some(n=>n.id==='dense-up'));});
test('expert weighting precedes aggregate norm and latent lift',()=>{const g=make();const connected=(a,b)=>g.edges.some(e=>e.source===a&&e.target===b);assert.ok(connected('expert-down','weighted-sum'));assert.ok(connected('weighted-sum','routed-norm'));assert.ok(connected('routed-norm','latent-up'));assert.ok(connected('norm-f','shared-up'));assert.ok(!connected('topk','shared-up'));});
test('reported training has Muon and MoonEP, not inherited Adam/FSDP',()=>{const t=make({view:'training'});assert.ok(t.nodes.some(n=>n.id==='muon'));assert.ok(t.nodes.some(n=>n.id==='moonep'));assert.ok(!t.nodes.some(n=>n.name.includes('FSDP')||n.name.includes('Adam')));});
test('serving draws both recurrent state and growing MLA history',()=>{const ids=make({view:'serving'}).nodes.map(n=>n.id);for(const id of ['kda-cache','mla-cache','prefix-match','restore'])assert.ok(ids.includes(id));});
for(const view of ['model','training','serving','hardware','network'])test(`${view}: finite bounds, unique IDs, orthogonal edges, known sources`,()=>{
 const g=make({view}),ids=new Set();for(const n of g.nodes){assert.ok(!ids.has(n.id));ids.add(n.id);for(const k of ['x','y','w','h'])assert.ok(Number.isFinite(n[k]));assert.ok(n.x>=0&&n.y>=0&&n.x+n.w<=g.width+.01&&n.y+n.h<=g.height+.01,n.id);}
 for(const n of g.nodes)for(const id of n.info?.sources||[])assert.ok(SOURCES.some(s=>s.id===id),`${n.id}: missing source ${id}`);
 for(const e of g.edges)for(let i=1;i<e.points.length;i++)assert.ok(e.points[i][0]===e.points[i-1][0]||e.points[i][1]===e.points[i-1][1]);
});
