/** Live multi-model inventory. Digests identify local consumers, NOT source truth. */
import * as core from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {readFileSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
const sha=v=>createHash('sha256').update(v).digest('hex');
const stable=v=>JSON.stringify(v,(_,x)=>x&&!Array.isArray(x)&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])):x);
const fields={},functions={},scenes={};
function walk(value,path){if(value&&typeof value==='object'&&!Array.isArray(value)){for(const k of Object.keys(value).sort())walk(value[k],`${path}.${k}`);}else fields[path]=value;}
for(const[name,v]of Object.entries(core))if(typeof v==='function')functions[`function.${name}`]={sha256:sha(v.toString())};else if(name!=='SOURCES')walk(v,name);
for(const file of [...readdirSync(new URL('../src/',import.meta.url)).filter(x=>/\.(mjs|css)$/.test(x)).map(x=>'src/'+x),'build.mjs','index.html'])functions[`file.${file}`]={sha256:sha(readFileSync(new URL('../'+file,import.meta.url)))};
const cases=[];
for(const [model,m]of Object.entries(core.MODELS)){
 for(let layer=1;layer<=m.layers;layer++)for(const detail of ['shapes','operators'])cases.push({model,view:'model',stage:'block',layer,detail});
 for(const view of core.VIEWS)cases.push({model,view,layer:Math.min(4,m.layers)});
 if(m.vision)cases.push({model,view:'model',stage:'inputs'});
 if(m.residual!=='standard')cases.push({model,view:'model',stage:'residual'});
 if(m.pleLayers.length)cases.push({model,view:'model',stage:'ple'});
 for(const trainMode of ['fsdp','ddp','rl'])cases.push({model,view:'training',trainMode});
 for(const placement of ['split','colocated'])cases.push({model,view:'serving',placement});
 for(const compare of Object.keys(core.MODELS))cases.push({model,view:'compare',compare});
}
for(const hardware of Object.keys(core.HARDWARE))cases.push({view:'hardware',hardware});
for(const fabric of ['ib','roce'])for(const library of ['nccl','nvshmem','nixl'])for(const traffic of ['expert','allreduce','handoff'])cases.push({view:'network',fabric,library,traffic});
for(const patch of cases){const state=core.sanitizeState({...core.DEFAULT_STATE,...patch}),g=buildScene(state);
 for(const n of g.nodes){if(!n.info)continue;const content={name:n.name,sub:n.sub,info:n.info},signature=sha(stable(content));scenes[`scene.${state.model}.${state.view}.${state.stage}.${n.id}.${signature.slice(0,16)}`]=signature;}
 const content={caption:g.caption,groups:g.groups.map(x=>[x.id,x.title,x.subtitle]),labels:g.labels.map(x=>x.text),edges:g.edges.map(x=>[x.source,x.target,x.kind,Boolean(x.delay||x.delayed)])},signature=sha(stable(content));
 scenes[`scene.${state.model}.${state.view}.${state.stage}.graph.${signature.slice(0,16)}`]=signature;
}
console.log(JSON.stringify({schema:'stratum-inventory/v1',fields,functions,scenes,models:core.MODELS,hardware:core.HARDWARE,samples:{decoder_layers:Object.values(core.MODELS).reduce((n,m)=>n+m.layers,0),scenario_cases:cases.length,partitions:Object.fromEntries(Object.keys(core.MODELS).map(k=>[k,core.layerPartition(k)])),logical_state:Object.fromEntries(Object.keys(core.MODELS).map(k=>[k,core.cacheLedger(k,core.DEFAULT_STATE)]))}},null,2));
