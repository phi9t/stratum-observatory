/** Actual consumer values only. Independent expected arithmetic lives in Python. */
import {createHash} from 'node:crypto';
import {readFileSync} from 'node:fs';
import {MODELS,HARDWARE,DEFAULT_STATE,sanitizeState,parameterLedger,cacheLedger,trainingLedger,networkLedger,roofline} from '../src/core.mjs';
const hash=x=>createHash('sha256').update(x).digest('hex');
const unique=values=>[...new Set(values)];
const models=Object.values(MODELS),hardware=Object.values(HARDWARE);
const ranks=unique([1,DEFAULT_STATE.ranks,...hardware.map(h=>h.gpus)]);
const axes={cache:unique([1,DEFAULT_STATE.context,...models.map(m=>m.maxContext)]).flatMap(context=>unique([1,DEFAULT_STATE.batch,2]).flatMap(batch=>unique([1,2,4].map(kvBytes=>sanitizeState({kvBytes}).kvBytes)).map(kvBytes=>({context,batch,kvBytes})))),training:ranks.flatMap(ranks=>['ddp','fsdp','rl'].map(trainMode=>({ranks,trainMode}))),network:ranks.map(ranks=>({ranks,traffic:'allreduce'})),intensity:unique([sanitizeState({intensity:1}).intensity,DEFAULT_STATE.intensity,sanitizeState({intensity:Number.MAX_SAFE_INTEGER}).intensity])};
const cases=[];
for(const m of models){
 cases.push({id:`book-gqa-parameters/${m.id}`,method:'book-gqa-parameters',model:m.id,settings:{},actual:parameterLedger(m)});
 for(const [i,s]of axes.cache.entries())cases.push({id:`book-gqa-cache/${m.id}/${i}`,method:'book-gqa-cache',model:m.id,settings:s,actual:cacheLedger(m,s)});
 for(const [i,s]of axes.training.entries())cases.push({id:`book-training-state/${m.id}/${i}`,method:'book-training-state',model:m.id,settings:s,actual:trainingLedger(m,s)});
 for(const [i,s]of axes.network.entries())cases.push({id:`book-ring-allreduce/${m.id}/${i}`,method:'book-ring-allreduce',model:m.id,settings:s,actual:networkLedger(m,s)});
}
for(const h of hardware)for(const [i,intensity]of axes.intensity.entries())cases.push({id:`book-roofline/${h.id}/${i}`,method:'book-roofline',hardware:h.id,settings:{intensity},actual:roofline(h.id,intensity)});
const function_sha256=Object.fromEntries(Object.entries({sanitizeState,parameterLedger,cacheLedger,trainingLedger,networkLedger,roofline}).map(([k,v])=>[k,hash(v.toString())]));
const consumer_sha256=hash(JSON.stringify(Object.fromEntries(['src/models.mjs','src/core.mjs'].map(p=>[p,hash(readFileSync(p))]))));
console.log(JSON.stringify({schema:'stratum-accounting-export/v1',models,hardware,defaults:DEFAULT_STATE,axes,cases,function_sha256,consumer_sha256}));
