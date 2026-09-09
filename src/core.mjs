import {MODELS,HARDWARE,SOURCES,VIEWS,VIEW_NAMES} from './models.mjs';
export {MODELS,HARDWARE,SOURCES,VIEWS,VIEW_NAMES};
export const DEFAULT_STATE={model:'ds3',view:'model',stage:'block',layer:4,hardware:'h200',context:32768,batch:1,kvBytes:2,stateBytes:4,weightBytes:2,ranks:8,placement:'split',trainMode:'fsdp',traffic:'expert',fabric:'ib',library:'nccl',utilization:70,oversub:1,intensity:128,compare:'kimi3',theme:'dark',detail:'shapes',zoom:1,trace:false};
export function getModel(id){if(typeof id==='object'&&id?.id)return id;const m=MODELS[id];if(!m)throw new RangeError(`Unknown model: ${id}`);return m;}
export function sanitizeState(input={}){
 const s={...DEFAULT_STATE};
 const choices={model:Object.keys(MODELS),compare:Object.keys(MODELS),view:VIEWS,hardware:Object.keys(HARDWARE),stage:['block','inputs','residual','ple'],placement:['split','colocated'],trainMode:['fsdp','ddp','rl'],traffic:['expert','allreduce','handoff'],fabric:['ib','roce'],library:['nccl','nvshmem','nixl'],theme:['dark','light'],detail:['shapes','operators']};
 for(const [k,opts]of Object.entries(choices))if(opts.includes(input[k]))s[k]=input[k];
 const numeric={context:[1,1048576],batch:[1,256],ranks:[1,4096],utilization:[1,100],oversub:[1,16],intensity:[1,2048],layer:[1,MODELS[s.model].layers]};
 for(const[k,[lo,hi]]of Object.entries(numeric)){const x=Number(input[k]);if(Number.isFinite(x)&&x>=lo)s[k]=Math.min(hi,Math.floor(x));}
 for(const k of ['kvBytes','weightBytes'])if([1,2,4].includes(Number(input[k])))s[k]=Number(input[k]);
 if([2,4].includes(Number(input.stateBytes)))s.stateBytes=Number(input.stateBytes);
 if(Number.isFinite(Number(input.zoom))&&Number(input.zoom)>=.75)s.zoom=Math.min(2,Number(input.zoom));
 if(typeof input.trace==='boolean')s.trace=input.trace;
 s.layer=Math.min(s.layer,MODELS[s.model].layers);return s;
}
export function layerKind(id,layer){const m=getModel(id);if(!Number.isInteger(layer)||layer<1||layer>m.layers)throw new RangeError('Layer is one-indexed and must exist.');if(m.architecture==='hybrid-kda')return layer%4===0||layer===m.layers?'mla':'kda';if(m.architecture.startsWith('hybrid'))return layer%m.globalEvery===0?(m.architecture==='hybrid-qsa'?'qsa':'gqa'):'delta';return m.architecture;}
export function layerPartition(id){const m=getModel(id),r={mla:0,gqa:0,kda:0,delta:0,qsa:0};for(let l=1;l<=m.layers;l++)r[layerKind(m,l)]++;return r;}
export function ffnKind(id,layer){const m=getModel(id);return !m.experts||layer<=m.denseLayers?'dense':m.latent?'latent-moe':'moe';}
export function parameterLedger(id){
 const m=getModel(id);let main=m.total,approximate=true;
 const embedding=2*m.vocab*m.d;
 const routed=(m.layers-m.denseLayers)*m.experts*3*(m.latent||m.d)*m.expertFF;
 const shared=(m.layers-m.denseLayers)*m.shared*3*m.d*m.expertFF;
 let attention=0;
 if(m.architecture==='gqa'){
   attention=m.layers*(2*m.d*m.heads*m.headDim+2*m.d*m.kvHeads*m.headDim+2*m.headDim);
   const ffn=m.experts?routed+m.layers*m.d*m.experts: m.layers*3*m.d*m.ff;
   main=embedding+attention+ffn+(2*m.layers+1)*m.d;approximate=false;
 }else if(m.architecture==='mla'){
   const a=m.d*m.qRank+m.qRank*m.heads*(m.nopeDim+m.ropeDim)+m.d*(m.kvRank+m.ropeDim)+m.kvRank*m.heads*(m.nopeDim+m.vDim)+m.heads*m.vDim*m.d+m.qRank+m.kvRank;
   attention=m.layers*a;
   main=embedding+attention+(2*m.layers+1)*m.d+routed+shared+m.denseLayers*3*m.d*m.ff+(m.layers-m.denseLayers)*(m.d*m.experts+(m.id==='ds3'?m.experts:0));approximate=false;
 }
 const ngram=m.ngram||0;
 return {main,resident:main+ngram,ngram,mtp:m.id==='q38flash'?m.mtp:0,hasMtp:!!m.mtp,active:m.active,approximate,embedding,routed,shared,attention,
  scope:m.id==='q38flash'?'Rounded 125B backbone + 51B n-gram tables; excludes 4B MTP and separately unaccounted auxiliary/vision weights.':approximate?'Publisher-rounded scale; not an exact tensor-manifest count. Vision/MTP scope is not independently reconciled.':'Calculated base text decoder, untied vocabulary and norms; excludes MTP, vision and quantization metadata.'};
}
export function cacheLedger(id,settings={}){
 const m=getModel(id),s=sanitizeState({...DEFAULT_STATE,...settings,model:m.id});const p=layerPartition(m),B=s.batch,S=s.context;
 const linear=p.kda+p.delta,global=p.mla+p.gqa+p.qsa;
 const mla=p.mla*B*S*(m.kvRank+m.ropeDim)*s.kvBytes;
 const gqa=(p.gqa+p.qsa)*B*S*2*m.kvHeads*m.headDim*s.kvBytes;
 const recurrent=linear*B*m.linearHeads*m.linearHeadDim*m.linearHeadDim*s.stateBytes;
 const conv=linear*B*(2*m.linearKeyHeads+m.linearHeads)*m.linearHeadDim*m.convKernel*2;
 const indexer=p.qsa ? p.qsa*B*Math.ceil(S/m.indexerCompression)*m.indexer*s.kvBytes : 0;
 const history=mla+gqa;const total=history+recurrent+conv+indexer;
 const exclusions=['Page rounding, scale/metadata storage, fragmentation and temporary workspace','Output-token reserve, distributed replication and retained prefix snapshots'];
 if(m.pleLayers.length)exclusions.push('PLE convolution, n-gram context and incomplete indexer-block bookkeeping');
 return {history,mla,gqa,recurrent,conv,indexer,total,global,linear,B,S,exclusions,partial:true,
   formula:p.mla?`${p.mla} × B × S × (${m.kvRank} + ${m.ropeDim}) × b_KV`:`${global} × B × S × 2 × ${m.kvHeads} × ${m.headDim} × b_KV`};
}
export function servingLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),p=parameterLedger(m),cache=cacheLedger(m,s);const weights=p.resident*s.weightBytes;return{...cache,weights,perRank:(weights+cache.total)/s.ranks,replicas:s.placement==='split'?2:1,fleetWeights:weights*(s.placement==='split'?2:1),scope:p.scope};}
export function trainingLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),p=parameterLedger(m).resident;
 const weights=2*p,gradients=2*p,master=4*p,moments=8*p,total=weights+gradients+master+moments;return{weights,gradients,master,moments,total,perRank:total/(s.trainMode==='ddp'?1:s.ranks),ranks:s.ranks,bytesPerParameter:16,approximate:parameterLedger(m).approximate};}
export function networkLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),H=HARDWARE[s.hardware];const n=s.ranks,off=(n-1)/n;let sent=0,formula='',scope='';
 if(s.traffic==='expert'){sent=m.experts?s.batch*s.context*m.topK*(m.latent||m.d)*s.kvBytes*off:0;formula='T × k × d_dispatch × b × (N − 1) / N';scope='One dispatch phase; uniform expert placement; tokens originate at each rank. Combine, metadata and imbalance excluded.';}
 if(s.traffic==='allreduce'){const M=parameterLedger(m).resident*2;sent=2*off*M;formula='2 × (N − 1) / N × M';scope='Ring all-reduce: bytes sent per rank for a full BF16 gradient buffer. NCCL algorithm selection is not simulated.';}
 if(s.traffic==='handoff'){sent=cacheLedger(m,s).total;formula='Logical KV history + current recurrent/conv state';scope='One full prefix handoff between independent pools. PLE bookkeeping, page metadata and replicas excluded.';}
 const rate=H.nicGbps*1e9/8*s.utilization/100/s.oversub;return{sent,rate,seconds:sent/rate,formula,scope,width:m.latent||m.d,off,linkGbps:H.nicGbps};}
export function roofline(hardware,intensity){const h=HARDWARE[hardware];if(!h||!Number.isFinite(intensity)||intensity<=0)throw new RangeError('Invalid roofline inputs');const memory=h.hbmTBps*intensity;return{memory,compute:h.peakTF,attainable:Math.min(memory,h.peakTF),ridge:h.peakTF/h.hbmTBps,bound:memory<h.peakTF?'HBM bandwidth':'Tensor compute'};}
export function formatNum(x,dec=0){return Number(x).toLocaleString('en-US',{maximumFractionDigits:dec});}
export function formatParams(x){return x>=1e12?(x/1e12).toFixed(2)+'T':x>=1e9?(x/1e9).toFixed(x>=100e9?1:2)+'B':x>=1e6?(x/1e6).toFixed(1)+'M':formatNum(x);}
export function formatBytes(x){if(!Number.isFinite(x))return'—';if(x===0)return'0 B';const units=['B','KiB','MiB','GiB','TiB','PiB'];let i=0;while(x>=1024&&i<5){x/=1024;i++;}return x.toFixed(i?2:0)+' '+units[i];}
export function formatTime(sec){if(sec===0)return'0';return sec>=1?sec.toFixed(2)+' s':sec>=.001?(sec*1000).toFixed(2)+' ms':(sec*1e6).toFixed(1)+' μs';}
export function evidenceSummary(){return{referenceRecords:Object.keys(SOURCES).length,snapshots:0,independentReviews:0,publication:'BLOCKED',runtime:'NOT_RUN',reason:'References inspected; immutable source bytes, exact semantic anchors and independent signatures are not accepted in this artifact.'};}
