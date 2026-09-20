/** Live multi-model inventory. Digests identify local consumers, NOT source truth. */
import * as core from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';
import {innovationRows} from '../src/innovations.mjs';
import {PAGES} from '../src/pages.mjs';
import {DATA_TOPICS} from '../src/landing.mjs';
import {readFileSync,readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';
const sha=v=>createHash('sha256').update(v).digest('hex');
const stable=v=>JSON.stringify(v,(_,x)=>x&&!Array.isArray(x)&&typeof x==='object'?Object.fromEntries(Object.keys(x).sort().map(k=>[k,x[k]])):x);
const fields={},functions={},scenes={};
for(const topic of DATA_TOPICS)fields[`data.topic.${topic.id}`]=topic;
const accounting=JSON.parse(readFileSync(new URL('../verification/manifests/accounting-methods.json',import.meta.url),'utf8'));
for(const method of accounting.methods)fields[`accounting.method.${method.id}`]=method;
for(const example of accounting.examples.filter(x=>x.sources?.length))fields[`accounting.example.${example.id}`]=example;
function walk(value,path){if(value&&typeof value==='object'&&!Array.isArray(value)){for(const k of Object.keys(value).sort())walk(value[k],`${path}.${k}`);}else fields[path]=value;}
for(const[name,v]of Object.entries(core))if(typeof v==='function')functions[`function.${name}`]={sha256:sha(v.toString())};else if(name!=='SOURCES')walk(v,name);
for(const file of [...readdirSync(new URL('../src/',import.meta.url)).filter(x=>/\.(mjs|css)$/.test(x)).map(x=>'src/'+x),'build.mjs','index.html'])functions[`file.${file}`]={sha256:sha(readFileSync(new URL('../'+file,import.meta.url)))};
const params=m=>{const p=core.parameterLedger(m);return{main:p.main,resident:p.resident,ngram:p.ngram,embedding:p.embedding,routed:p.routed,shared:p.shared,attention:p.attention,approximate:p.approximate};};
const cases=[];
for(const [model,m]of Object.entries(core.MODELS)){
 for(let layer=1;layer<=m.layers;layer++)for(const detail of ['shapes','operators'])cases.push({model,view:'model',stage:'block',layer,detail});
 for(const view of core.VIEWS)cases.push({model,view,layer:Math.min(4,m.layers)});
 for(const stage of core.focalStages(m))cases.push({model,view:'model',stage});
 for(const trainMode of ['fsdp','ddp','rl'])cases.push({model,view:'training',trainMode});
 for(const placement of ['split','colocated'])for(const phase of m.encoderLayers?['prefill','decode']:['decode'])cases.push({model,view:'serving',placement,phase});
 for(const compare of Object.keys(core.MODELS))cases.push({model,view:'compare',compare});
}
for(const hardware of Object.keys(core.HARDWARE))cases.push({view:'hardware',hardware});
for(const fabric of ['ib','roce'])for(const library of ['nccl','nvshmem','nixl'])for(const traffic of ['expert','allreduce','handoff'])cases.push({view:'network',fabric,library,traffic});
for(const patch of cases){const state=core.sanitizeState({...core.DEFAULT_STATE,...patch}),g=buildScene(state);
 for(const n of g.nodes){if(!n.info)continue;const content={name:n.name,sub:n.sub,info:n.info},signature=sha(stable(content));scenes[`scene.${state.model}.${state.view}.${state.stage}.${n.id}.${signature.slice(0,16)}`]=signature;}
 const content={caption:g.caption,groups:g.groups.map(x=>[x.id,x.title,x.subtitle]),labels:g.labels.map(x=>x.text),edges:g.edges.map(x=>[x.source,x.target,x.kind,Boolean(x.delay||x.delayed)])},signature=sha(stable(content));
 scenes[`scene.${state.model}.${state.view}.${state.stage}.graph.${signature.slice(0,16)}`]=signature;
}
console.log(JSON.stringify({schema:'stratum-inventory/v1',fields,functions,scenes,models:core.MODELS,hardware:core.HARDWARE,samples:{pages:PAGES,serving_phases:Object.fromEntries(Object.values(core.MODELS).filter(m=>m.encoderLayers).map(m=>[m.id,Object.fromEntries(['prefill','decode'].map(phase=>[phase,core.servingLedger(m,{...core.DEFAULT_STATE,phase})]))])),decoder_layers:Object.values(core.MODELS).reduce((n,m)=>n+m.layers,0),scenario_cases:cases.length,partitions:Object.fromEntries(Object.keys(core.MODELS).map(k=>[k,core.layerPartition(k)])),logical_state:Object.fromEntries(Object.keys(core.MODELS).map(k=>[k,core.cacheLedger(k,core.DEFAULT_STATE)])),logical_state_far:Object.fromEntries(Object.keys(core.MODELS).map(k=>[k,core.cacheLedger(k,{...core.DEFAULT_STATE,context:131072})])),parameters:Object.fromEntries(Object.values(core.MODELS).map(m=>[m.id,params(m)])),parameters_unit_ratio:Object.fromEntries(Object.values(core.MODELS).map(m=>[m.id,params({...m,compressEncoder:1,compressDecoder:1})])),innovations:Object.fromEntries(Object.values(core.MODELS).filter(m=>m.encoderLayers).map(m=>[m.id,innovationRows(m).map(r=>({id:r.id,title:r.title,replaces:r.replaces,mechanism:r.mechanism,effect:r.effect,sources:r.sources,citations:r.sources.map(s=>core.SOURCES[s]&&core.SOURCES[s].url).filter(Boolean)}))])),source_ids:Object.keys(core.SOURCES),focal_stages:Object.fromEntries(Object.keys(core.MODELS).map(k=>[k,core.focalStages(k)]))}},null,2));
