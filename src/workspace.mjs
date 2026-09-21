/** Navigation bindings only. Tensor contracts and explanations remain in their source records. */
import {PYTORCH_EXAMPLES,moduleExample} from './pytorch.mjs';
import {WORKED_RESULTS} from './worked-results.mjs';

const EXAMPLES=new Map(PYTORCH_EXAMPLES.map(example=>[example.id,example]));
const FIXTURES={
 'csa2-selection':['selection'],'csa2-attention':['query'],
 'backbone-mhc':['residual'],'vision-inputs':['inputs'],
 'dspark-attention':['dspark'],'dspark-heads':['dspark'],
};
const CSA2={
 'csa2-full':'csa2-compression','csa2-owners':'csa2-compression',
 'csa2-reindex':'csa2-selection','csa2-reuse':'csa2-selection',
 'csa2-candidates':'csa2-selection','csa2-select':'csa2-selection',
 'csa2-window':'csa2-attention','csa2-union':'csa2-attention',
};
const BLOCK={
 'index-proj':'csa2-selection','index-blocks':'csa2-selection','index-select':'csa2-selection',
 'attention':'csa2-attention','attn-sink':'csa2-attention',
 'router':'backbone-moe','top-k':'backbone-moe','expert-up':'backbone-moe',
 'expert-activation':'backbone-moe','expert-down':'backbone-moe','combine':'backbone-moe',
 'shared-up':'backbone-moe','shared-activation':'backbone-moe','shared-down':'backbone-moe',
};
const INPUTS=new Set(['patches','vision-norm1','vision-qkv','vision-rope','vision-attention','vision-add1','vision-ffn','vision-add2','vision-merge']);
const ENGRAM=new Set(['engram-table','engram-project','engram-gate','engram-write']);
const RESIDUAL=new Set(['encoder','decoder','handoff','attention-compute','ffn-compute','expand','collapse','incoming','attention-read','attention-mixer','attention-write','ffn-read','ffn-mixer','ffn-write','outgoing']);

/** Semantic identity is model- and chapter-specific; unclassified nodes retain their own ID. */
function exampleKey(state,node){
 if(!node)return null;
 if(state.model!=='dsv41'||(state.view&&state.view!=='model'))return null;
 const example=moduleExample(state.stage);
 if(example)return example.id;
 if(state.stage==='csa2')return CSA2[node.id]||null;
 if(['encoder','decoder','block'].includes(state.stage))return BLOCK[node.id]||null;
 if(state.stage==='inputs'&&INPUTS.has(node.id))return 'vision-inputs';
 if(state.stage==='inputs'&&['engram-lookup','engram-inject'].includes(node.id))return 'engram-write';
 if(state.stage==='engram'&&ENGRAM.has(node.id))return 'engram-write';
 if(state.stage==='residual'&&RESIDUAL.has(node.id))return 'backbone-mhc';
 if(state.stage==='ced'&&node.id==='candidate-pool')return 'csa2-selection';
 if(state.stage==='dspark'){
  if(node.id==='dspark-attention')return 'dspark-attention';
  if(['dspark-read','dspark-query','dspark-history','dspark-draft-kv','dspark-output','dspark-attention-write','dspark-ffn-read','dspark-router','dspark-routed','dspark-shared','dspark-moe','dspark-ffn-write','dspark-draft'].includes(node.id))return 'dspark-pro-block';
  if(/^dspark-sample-[0-9]+$/.test(node.id))return 'dspark-heads';
  if(['dspark-markov','dspark-confidence','dspark-parallel-logits','dspark-predecessors'].includes(node.id))return 'dspark-heads';
  if(node.id==='dspark-verification')return 'deepspec-verifier';
 }
 return null;
}
export function nodeModule(state,node){return exampleKey(state,node)||node?.id||null;}

/** Resolve URL module to an actual operation, then expose only its supported reading panels. */
export function resolveWorkspace(state,scene){
 const nodes=scene.nodes||[],edges=scene.edges||[];
 const node=nodes.find(n=>n.id===state.module)
  ||nodes.find(n=>nodeModule(state,n)===state.module)
  ||nodes.find(n=>n.id===scene.defaultSelected)||nodes[0]||null;
 const key=nodeModule(state,node);
 // Identity alone cannot grant implementation support: the model and explicit mapping must agree.
 const example=EXAMPLES.get(exampleKey(state,node))||null;
 const workedIds=example?(FIXTURES[example.id]||[]).filter(id=>Object.hasOwn(WORKED_RESULTS,id)):[];
 const [requestedFixture,requestedStep]=String(state.example||'').split('--');
 const fixtureId=workedIds.includes(requestedFixture)?requestedFixture:workedIds[0]||null;
 const fixture=fixtureId?WORKED_RESULTS[fixtureId]:null;
 const step=fixture&&requestedFixture===fixtureId?[...fixture.inputs,...fixture.steps].find(row=>row.id===requestedStep&&state.example===`${fixtureId}--${row.id}`)||null:null;
 const exampleId=fixtureId?(step?`${fixtureId}--${step.id}`:fixtureId):'';
 const info=node?.info||null;
 const mathematics=nodes.filter(n=>n===node||(example&&exampleKey(state,n)===example.id)).map(n=>n.info).filter(info=>info&&(info.formula||info.facts?.length));
 const panels=['schematic'];
 if(mathematics.length)panels.push('maths');
 if(example)panels.push('pytorch');
 if(workedIds.length)panels.push('worked');
 if(info?.sources?.length||example?.sources?.length)panels.push('sources');
 const neighbors=direction=>nodes.filter(n=>edges.some(e=>direction==='in'?e.target===node?.id&&e.source===n.id:e.source===node?.id&&e.target===n.id));
 const inputs=neighbors('in'),outputs=neighbors('out');
 return {module:node?.id||null,moduleKey:key,nodeId:node?.id||null,node,example,
  panels,panel:panels.includes(state.panel)?state.panel:'schematic',
  pythonExamples:example?[example]:[],workedIds,workedId:fixtureId,fixtureId,fixture,step,exampleId,info,mathematics,
  metadata:{inputs,outputs,computed:node&&node.tone!=='state'&&info?.formula?[node]:[],retainedInputs:inputs.filter(n=>n.tone==='state'),reused:[]},
 };
}
