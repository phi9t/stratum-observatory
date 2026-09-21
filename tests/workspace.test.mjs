import test from 'node:test';
import assert from 'node:assert/strict';
import {buildScene} from '../src/diagrams.mjs';
import {resolveWorkspace,nodeModule} from '../src/workspace.mjs';

test('CSA2 reindex connects its exact operation to selection Python and CPU fixture',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2',module:'csa2-reindex',panel:'pytorch'};
 const scene=buildScene(state),w=resolveWorkspace(state,scene);
 assert.equal(w.nodeId,'csa2-reindex');assert.equal(w.example.id,'csa2-selection');
 assert.deepEqual(w.workedIds,['selection']);assert.equal(w.panel,'pytorch');
 assert.equal(w.info,scene.nodes.find(n=>n.id===w.nodeId).info);
 for(const panel of w.panels)assert.equal(resolveWorkspace({...state,panel},scene).nodeId,w.nodeId);
});
test('wrong models and system views cannot inherit Python or worked support',()=>{
 for(const state of [{model:'q38flash',view:'model',stage:'block'},{model:'dsv41',view:'serving',stage:'csa2'}]){
 const scene=buildScene(state),w=resolveWorkspace({...state,panel:'pytorch'},scene);
 assert.equal(w.example,null);assert.deepEqual(w.pythonExamples,[]);assert.deepEqual(w.workedIds,[]);assert.equal(w.panel,'schematic');
 }
});
test('invalid node and panel IDs recover to actual scene records',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2',module:'not-real',panel:'not-real'};
 const scene=buildScene(state),w=resolveWorkspace(state,scene);
 assert.ok(scene.nodes.some(n=>n.id===w.nodeId));assert.equal(w.panel,'schematic');
 assert.equal(resolveWorkspace(state,{nodes:[],edges:[]}).nodeId,null);
});
test('exact node wins over module family and fallback identity remains node-specific',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2'},scene=buildScene(state);
 for(const id of ['csa2-reindex','csa2-select'])assert.equal(resolveWorkspace({...state,module:id},scene).nodeId,id);
 assert.equal(nodeModule({model:'kimi3',view:'model',stage:'block'},{id:'index-select'}),'index-select');
});
test('structured metadata only contains original scene records and original explanations',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2',module:'csa2-reindex'};
 const scene=buildScene(state),w=resolveWorkspace(state,scene);
 for(const rows of Object.values(w.metadata))for(const row of rows)assert.equal(row.info,scene.nodes.find(n=>n.id===row.id).info);
});
test('unclassified nodes cannot gain implementation by impersonating an example ID',()=>{
 const w=resolveWorkspace({model:'dsv41',view:'model',stage:'ced',module:'csa2-attention',panel:'pytorch'},{nodes:[{id:'csa2-attention'}],edges:[]});
 assert.equal(w.example,null);assert.equal(w.panel,'schematic');
});
test('explicit module stages preserve operation selection and the canonical example',()=>{
 const state={model:'dsv41',view:'model',stage:'module-csa2-selection'};
 const scene=buildScene(state);
 for(const node of scene.nodes){const w=resolveWorkspace({...state,module:node.id,panel:'worked'},scene);assert.equal(w.nodeId,node.id);assert.equal(w.example.id,'csa2-selection');assert.equal(w.panel,'worked');}
});
test('related module mathematics enables reindex maths without inventing its formula',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2',module:'csa2-reindex',panel:'maths'},scene=buildScene(state);
 const w=resolveWorkspace(state,scene);
 assert.equal(w.panel,'maths');assert.ok(w.mathematics.length);
 for(const info of w.mathematics)assert.ok(scene.nodes.some(n=>n.info===info&&nodeModule(state,n)===w.moduleKey));
 assert.ok(w.mathematics.every(info=>info.title&&(info.formula||info.facts.length)));
});
test('retained input classification never claims that state was reused',()=>{
 const scene={nodes:[{id:'local-kv',tone:'state'},{id:'attention',tone:'compute'}],edges:[{source:'local-kv',target:'attention'}]};
 const w=resolveWorkspace({model:'dsv41',view:'model',stage:'decoder',module:'attention'},scene);
 assert.deepEqual(w.metadata.retainedInputs,[scene.nodes[0]]);assert.deepEqual(w.metadata.reused,[]);
});
test('CPU fixture selection is bounded to the selected module',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2',module:'csa2-reindex'},scene=buildScene(state);
 assert.equal(resolveWorkspace({...state,example:'query'},scene).workedId,'selection');
 assert.equal(resolveWorkspace({...state,example:'selection'},scene).workedId,'selection');
 assert.equal(resolveWorkspace({...state,module:'csa2-full',example:'selection'},scene).workedId,null);
});
test('actual residual, vision and Pro operations map only to matching implementations',()=>{
 for(const [stage,id,example] of [['residual','handoff','backbone-mhc'],['residual','attention-compute','backbone-mhc'],['inputs','engram-lookup','engram-write'],['inputs','vision-rope','vision-inputs'],['dspark','dspark-query','dspark-pro-block'],['dspark','dspark-router','dspark-pro-block'],['dspark','dspark-sample-2','dspark-heads']]){
 const state={model:'dsv41',view:'model',stage,module:id};assert.equal(resolveWorkspace(state,buildScene(state)).example?.id,example);
 }
 for(const [stage,id] of [['inputs','tokenizer'],['inputs','engram-hash'],['dspark','dspark-context'],['dspark','dspark-collapse']]){
 const state={model:'dsv41',view:'model',stage,module:id};assert.equal(resolveWorkspace(state,buildScene(state)).example,null);
 }
});
test('CPU step navigation validates fixture and row identity against saved tensors',()=>{
 const state={model:'dsv41',view:'model',stage:'csa2',module:'csa2-reindex'},scene=buildScene(state);
 const w=resolveWorkspace({...state,example:'selection--selector_scores'},scene);
 assert.equal(w.exampleId,'selection--selector_scores');assert.equal(w.fixtureId,'selection');
 assert.equal(w.step,w.fixture.steps.find(row=>row.id==='selector_scores'));
 for(const example of ['selection--missing','query--outputs','selection--selected_indices--extra']){
 const bad=resolveWorkspace({...state,example},scene);assert.equal(bad.exampleId,'selection');assert.equal(bad.step,null);
 }
});
