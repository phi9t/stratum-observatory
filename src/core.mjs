import {MODELS,HARDWARE,SOURCES,VIEWS,VIEW_NAMES,DSPARK_PRO} from './models.mjs';
import {PYTORCH_EXAMPLES,moduleStage} from './pytorch.mjs';
export {MODELS,HARDWARE,SOURCES,VIEWS,VIEW_NAMES,DSPARK_PRO};
export const DEFAULT_STATE={model:'ds3',view:'model',stage:'block',layer:4,hardware:'h200',context:32768,batch:1,kvBytes:2,stateBytes:4,weightBytes:2,ranks:8,placement:'split',trainMode:'fsdp',traffic:'expert',fabric:'ib',library:'nccl',utilization:70,oversub:1,intensity:128,compare:'kimi3',theme:'dark',detail:'shapes',zoom:1,trace:false,phase:'decode'};
export function getModel(id){if(typeof id==='object'&&id?.id)return id;const m=MODELS[id];if(!m)throw new RangeError(`Unknown model: ${id}`);return m;}
export function sanitizeState(input={}){
 const s={...DEFAULT_STATE};
 const choices={model:Object.keys(MODELS),compare:Object.keys(MODELS),view:VIEWS,hardware:Object.keys(HARDWARE),stage:['block',...new Set(Object.keys(MODELS).flatMap(focalStages))],placement:['split','colocated'],trainMode:['fsdp','ddp','rl'],traffic:['expert','allreduce','handoff'],fabric:['ib','roce'],library:['nccl','nvshmem','nixl'],theme:['dark','light'],detail:['shapes','operators'],phase:['prefill','decode']};
 for(const [k,opts]of Object.entries(choices))if(opts.includes(input[k]))s[k]=input[k];
 const numeric={context:[1,1048576],batch:[1,256],ranks:[1,4096],utilization:[1,100],oversub:[1,16],intensity:[1,2048],layer:[1,MODELS[s.model].layers]};
 for(const[k,[lo,hi]]of Object.entries(numeric)){const x=Number(input[k]);if(Number.isFinite(x)&&x>=lo)s[k]=Math.min(hi,Math.floor(x));}
 for(const k of ['kvBytes','weightBytes'])if([1,2,4].includes(Number(input[k])))s[k]=Number(input[k]);
 if([2,4].includes(Number(input.stateBytes)))s.stateBytes=Number(input.stateBytes);
 if(Number.isFinite(Number(input.zoom))&&Number(input.zoom)>=.75)s.zoom=Math.min(2,Number(input.zoom));
 if(typeof input.trace==='boolean')s.trace=input.trace;
 s.layer=Math.min(s.layer,MODELS[s.model].layers);
 // Clamp the stage to what this checkpoint offers, so an unreachable focal page cannot arrive
 // from a URL fragment. Without this the whitelist above admits `ds3&stage=ple`, buildScene fell
 // through to the decoder block, and the exported scenario recorded a stage the drawing is not.
 // Clamping here is also what lets the stage dispatch be exhaustive-or-throw rather than
 // fail-open (ruling T2g).
 if(s.stage!=='block'&&!focalStages(s.model).includes(s.stage))s.stage='block';
 if(s.view==='model'&&MODELS[s.model].encoderLayers&&s.stage==='block')s.stage=s.layer<=MODELS[s.model].encoderLayers?'encoder':'decoder';
 const layers=layerNumbers(s.model,s.stage);s.layer=Math.max(layers[0],Math.min(layers.at(-1),s.layer));
 return s;
}
/** The focal pages a checkpoint actually offers, in the order the control strip shows them.
 *
 * One derivation, seven former copies. `app.mjs` gated the N-gram page on `m.id==='q38flash'`
 * twice over -- once in its control strip, once in its model-switch guard -- and `diagrams.mjs`
 * gated it a third time on the same literal, while the exporters, the browser matrix and the
 * aesthetics gate derived the same stage from `pleLayers`, so the predicates differed in form: a
 * tenth checkpoint with `pleLayers` and a different id would be offered no N-gram tab while three
 * gates generated the page, `buildScene` fell through to `buildModel`, and a signature labelled
 * as one page was measured on another. Deriving it is also what closes the half-migration on the
 * other side -- `tests/capture.py` and `verification/presentations.py` each read the stage list
 * and then threw it away for the literal `'ced'`, so the mHC residual page was depicted in no
 * preview and measured by the presentations gate in no state. */
export function focalStages(id){const m=getModel(id),out=[];if(m.vision)out.push('inputs');if(m.residual!=='standard')out.push('residual');if(m.pleLayers.length)out.push('ple');if(m.encoderLayers)out.push('ced');if(m.architecture==='ced-csa2')out.push('encoder','decoder','engram','csa2','dspark',...PYTORCH_EXAMPLES.map(x=>moduleStage(x.id)));return out;}
/** Absolute layer IDs offered by a chapter; the checkpoint still owns the full stack. */
export function layerNumbers(id,stage){const m=getModel(id),first=m.encoderLayers&&stage==='decoder'?m.encoderLayers+1:1,last=m.encoderLayers&&stage==='encoder'?m.encoderLayers:m.layers;return Array.from({length:last-first+1},(_,i)=>first+i);}
export function layerKind(id,layer){const m=getModel(id);if(!Number.isInteger(layer)||layer<1||layer>m.layers)throw new RangeError('Layer is one-indexed and must exist.');if(m.architecture==='ced-csa2'){const l=layer-1;if(l<m.swaLayers)return'swa';if(m.kvSourceLayers.includes(l))return'csa2-full';if(m.indexSourceLayers.includes(l))return'csa2-reindex';return'csa2-reuse';}if(m.architecture==='hybrid-kda')return layer%4===0||layer===m.layers?'mla':'kda';if(m.architecture.startsWith('hybrid'))return layer%m.globalEvery===0?(m.architecture==='hybrid-qsa'?'qsa':'gqa'):'delta';return m.architecture;}
/** Bands of consecutive layers: one head layer plus the reuse layers that read what it published.
 *
 * Derived from layerKind over the requested one-indexed span, never from a written schedule, so a
 * band cannot exist that the model's own layer pattern does not produce. A 'csa2-reuse' layer joins
 * the band above it; an 'swa' layer joins only another 'swa'. Every other kind opens a band.
 */
export function layerBands(m,lo,hi){const out=[];
 for(let l=lo;l<=hi;l++){const kind=layerKind(m,l),last=out[out.length-1];
  if(last&&(kind==='csa2-reuse'||(kind==='swa'&&last.kind==='swa'))){last.hi=l;if(kind==='csa2-reuse')last.reuse++;continue;}
  out.push({kind,lo:l,hi:l,reuse:0});}
 return out;}
export function layerPartition(id){const m=getModel(id),r={mla:0,gqa:0,kda:0,delta:0,qsa:0,swa:0,'csa2-full':0,'csa2-reindex':0,'csa2-reuse':0};for(let l=1;l<=m.layers;l++)r[layerKind(m,l)]++;return r;}
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
 }else if(m.architecture==='ced-csa2'){
   const a=m.d*m.qRank+m.qRank+m.qRank*m.heads*m.headDim+m.d*m.headDim+m.headDim+m.oRank*m.heads*m.headDim+m.oGroups*m.oRank*m.d+m.heads;
   const mixHC=(2+m.streams)*m.streams;
   const mhc=m.layers*(2*mixHC*m.streams*m.d+2*mixHC+6);
   const gate=m.layers*(m.d*m.experts+m.experts*(m.vision?2:1));
   const engram=m.engramLayers.length*((m.engramNgram-1)*m.engramHeads*m.engramHeadDim*m.d*(m.streams+1)+2*m.streams*m.d);
   // Compressor (model.py:437-457): wkv and norm (RMSNorm(head_dim)) on every Full-mode layer;
   // wgate only where compress_ratio > 1.
   const compressor=m.kvSourceLayers.reduce((sum,l)=>{const ratio=l<m.encoderLayers?m.compressEncoder:m.compressDecoder;return sum+m.d*m.headDim+m.headDim+(ratio>1?m.d*m.headDim:0);},0);
   // Indexer (model.py:496-526): wq_b + weights_proj on every index-source layer; wk and k_norm
   // (RMSNorm(index_head_dim)) only where a Full layer also owns fresh KV.
   const indexer=m.indexSourceLayers.length*(m.qRank*m.indexerHeads*m.indexer+m.d*m.indexerHeads)+m.kvSourceLayers.length*(m.headDim*m.indexer+m.indexer);
   attention=m.layers*a;
   main=embedding+attention+routed+shared+gate+mhc+engram+compressor+indexer+(2*m.layers+1)*m.d;approximate=false;
 }
 const ngram=m.ngram||0;
 return {main,resident:main+ngram,ngram,mtp:m.id==='q38flash'?m.mtp:0,hasMtp:!!m.mtp,active:m.active,approximate,embedding,routed,shared,attention,
  scope:m.architecture==='ced-csa2'?'Calculated text backbone: attention, MoE, gating, mHC and Engram projections, untied vocabulary and norms. Engram tables are carried separately as n-gram residency. Excludes the vision encoder, the DSpark draft layers — whose parameter count is unpublished — and quantization metadata.':m.id==='q38flash'?'Rounded 125B backbone + 51B n-gram tables; excludes 4B MTP and separately unaccounted auxiliary/vision weights.':approximate?'Publisher-rounded scale; not an exact tensor-manifest count. Vision/MTP scope is not independently reconciled.':'Calculated base text decoder, untied vocabulary and norms; excludes MTP, vision and quantization metadata.'};
}
export function cacheLedger(id,settings={}){
 const m=getModel(id),s=sanitizeState({...DEFAULT_STATE,...settings,model:m.id});const p=layerPartition(m),B=s.batch,S=s.context;
 const csa2=p['csa2-full']+p['csa2-reindex']+p['csa2-reuse'];
 const linear=p.kda+p.delta,global=p.mla+p.gqa+p.qsa+csa2;
 const mla=p.mla*B*S*(m.kvRank+m.ropeDim)*s.kvBytes;
 const gqa=(p.gqa+p.qsa)*B*S*2*m.kvHeads*m.headDim*s.kvBytes;
 const recurrent=linear*B*m.linearHeads*m.linearHeadDim*m.linearHeadDim*s.stateBytes;
 const conv=linear*B*(2*m.linearKeyHeads+m.linearHeads)*m.linearHeadDim*m.convKernel*2;
 let indexer=p.qsa ? p.qsa*B*Math.ceil(S/m.indexerCompression)*m.indexer*s.kvBytes : 0;
 // FP4 E2M1 plus one E4M3 scale per 16 channels. The precision is trained in (QAT), so it is
 // an architectural fact and s.kvBytes deliberately does not apply. SWA KV is retained at FP8.
 let mainKV=0,windowKV=0;
 if(m.architecture==='ced-csa2'){
  for(const l of m.kvSourceLayers){const ratio=l<m.encoderLayers?m.compressEncoder:m.compressDecoder;mainKV+=B*S*(m.headDim/ratio)*(0.5+1/16);indexer+=B*S*(m.indexer/ratio)*(0.5+1/m.indexerScaleGroup);}
  windowKV=B*m.window*m.layers*m.headDim*1;
 }
 const history=mla+gqa+mainKV;
 const contextLinear=history+indexer;
 const bounded=recurrent+conv+windowKV;
 const total=contextLinear+bounded;
 const exclusions=[m.architecture==='ced-csa2'?'Page rounding, additional metadata, fragmentation and temporary workspace':'Page rounding, scale/metadata storage, fragmentation and temporary workspace','Output-token reserve, distributed replication and retained prefix snapshots'];
 if(m.pleLayers.length)exclusions.push('PLE convolution, n-gram context and incomplete indexer-block bookkeeping');
 if(m.architecture==='ced-csa2')exclusions.push('Incomplete compression pairs, candidate-pool bookkeeping and SWA Bounded Replay recomputation');
 return {history,mla,gqa,recurrent,conv,indexer,mainKV,windowKV,contextLinear,bounded,total,global,linear,B,S,exclusions,partial:true,
   formula:m.architecture==='ced-csa2'?`${m.kvSourceLayers.length} Full-mode layers × B × S × (${m.headDim} / m) × b_FP4  +  B × ${m.window} × ${m.layers} × ${m.headDim} × b_FP8`:p.mla?`${p.mla} × B × S × (${m.kvRank} + ${m.ropeDim}) × b_KV`:`${global} × B × S × 2 × ${m.kvHeads} × ${m.headDim} × b_KV`};
}
export function servingLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),p=parameterLedger(m),cache=cacheLedger(m,s);const weights=p.resident*s.weightBytes;
 const phaseLayers=s.phase==='prefill'&&m.encoderLayers?m.encoderLayers:m.layers;
 const phaseActive=s.phase==='prefill'?m.active:(m.activeDecode||m.active);
 return{...cache,weights,perRank:(weights+cache.total)/s.ranks,replicas:s.placement==='split'?2:1,fleetWeights:weights*(s.placement==='split'?2:1),phase:s.phase,phaseLayers,phaseActive,scope:p.scope};}
export function trainingLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),p=parameterLedger(m).resident;
 const weights=2*p,gradients=2*p,master=4*p,moments=8*p,total=weights+gradients+master+moments;return{weights,gradients,master,moments,total,perRank:total/(s.trainMode==='ddp'?1:s.ranks),ranks:s.ranks,bytesPerParameter:16,approximate:parameterLedger(m).approximate};}
export function networkLedger(id,settings={}){const m=getModel(id),s=sanitizeState({...settings,model:m.id}),H=HARDWARE[s.hardware];const n=s.ranks,off=(n-1)/n;let sent=0,formula='',scope='';
 if(s.traffic==='expert'){sent=m.experts?s.batch*s.context*m.topK*(m.latent||m.d)*s.kvBytes*off:0;formula='T × k × d_dispatch × b × (N − 1) / N';scope='One dispatch phase; uniform expert placement; tokens originate at each rank. Combine, metadata and imbalance excluded.';}
 if(s.traffic==='allreduce'){const M=parameterLedger(m).resident*2;sent=2*off*M;formula='2 × (N − 1) / N × M';scope='Ring all-reduce: bytes sent per rank for a full BF16 gradient buffer. NCCL algorithm selection is not simulated.';}
 if(s.traffic==='handoff'){sent=cacheLedger(m,s).total;formula='Logical KV history + current recurrent/conv state';scope='One full prefix handoff between independent pools. PLE bookkeeping, page metadata and replicas excluded.';}
 const rate=H.nicGbps*1e9/8*s.utilization/100/s.oversub;return{sent,rate,seconds:sent/rate,formula,scope,width:m.latent||m.d,off,linkGbps:H.nicGbps};}
export function roofline(hardware,intensity){const h=HARDWARE[hardware];if(!h||!Number.isFinite(intensity)||intensity<=0)throw new RangeError('Invalid roofline inputs');const memory=h.hbmTBps*intensity;return{memory,compute:h.peakTF,attainable:Math.min(memory,h.peakTF),ridge:h.peakTF/h.hbmTBps,bound:memory<h.peakTF?'HBM bandwidth':'Tensor compute'};}
export function formatNum(x,dec=0){return Number(x).toLocaleString('en-US',{maximumFractionDigits:dec,useGrouping:false});}
export function formatParams(x){return x>=1e12?(x/1e12).toFixed(2)+'T':x>=1e9?(x/1e9).toFixed(x>=100e9?1:2)+'B':x>=1e6?(x/1e6).toFixed(1)+'M':formatNum(x);}
export function formatBytes(x){if(!Number.isFinite(x))return'—';if(x===0)return'0 B';const units=['B','KiB','MiB','GiB','TiB','PiB'];let i=0;while(x>=1024&&i<5){x/=1024;i++;}return x.toFixed(i?2:0)+' '+units[i];}
export function formatTime(sec){if(sec===0)return'0';return sec>=1?sec.toFixed(2)+' s':sec>=.001?(sec*1000).toFixed(2)+' ms':(sec*1e6).toFixed(1)+' μs';}
export function evidenceSummary(){return{referenceRecords:Object.keys(SOURCES).length,snapshots:0,independentReviews:0,publication:'BLOCKED',runtime:'NOT_RUN',reason:'References inspected; immutable source bytes, exact semantic anchors and independent signatures are not accepted in this artifact.'};}

/** Reader-facing method assumptions. Arithmetic audit results belong to their sampled report. */
export function accountingAssumptions(id,settings={}){
 const m=getModel(id),s=sanitizeState({...settings,model:m.id});
 const row=(method,title,body,scope,applicability,sources,values={})=>({method,title,body,scope,applicability,sources,values});
 if(s.view==='training'){
  const l=trainingLedger(m,s),parameters=parameterLedger(m).resident;
  return [row('book-training-state','Persistent training state',
   `The selected baseline retains BF16 weights and gradients, FP32 master weights and Adam moments: ${l.bytesPerParameter} bytes per parameter (${formatBytes(l.total)} globally). An additional FP32 accumulation buffer adds ${formatBytes(l.master)}, giving ${(l.total+l.master)/parameters} bytes per parameter (${formatBytes(l.total+l.master)}). The hosted playbook prose mentions that extra copy; its displayed equations omit it.`,
   'Logical persistent learner state. Activations, temporary gathers, allocator overhead and other RL models are excluded. Uniform division is an accounting assumption, not a verified sharding plan.',
   'applicable',['book-ultrascale','book-scaling-training'],{baseline:l.total,accumulation:l.master,withAccumulation:l.total+l.master,bytesPerParameter:l.bytesPerParameter,alternativeBytesPerParameter:(l.total+l.master)/parameters})];
 }
 if(s.view==='hardware'){
  const h=HARDWARE[s.hardware],l=roofline(s.hardware,s.intensity);
  return [row('book-roofline','Matching roofline units',
   `The selected per-GPU dense FP8 ceiling is ${formatNum(l.compute)} TFLOP/s. Decimal bandwidth ${h.hbmTBps} TB/s × ${s.intensity} FLOP/byte gives ${formatNum(l.memory,2)} TFLOP/s; the lower ceiling is ${formatNum(l.attainable,2)} TFLOP/s. The ridge is ${formatNum(l.ridge,2)} FLOP/byte.`,
   'The intensity must describe actual HBM reads and writes at the same precision and device scope. This bound excludes communication, launch costs and measured utilization.',
   'applicable',['book-scaling-roofline'],l)];
 }
 if(s.view==='network'){
  const l=networkLedger(m,s),ring=s.traffic==='allreduce';
  return [row('book-ring-allreduce','Ring traffic and direction',
   ring?`For ${s.ranks} ranks, reduce-scatter plus all-gather sends 2 × (N − 1) / N × M bytes per rank: ${formatBytes(l.sent)} for the full BF16 gradient buffer. Each rank receives the same amount separately. Adding both directions would double this sent-byte figure.`:
   `The selected ${s.traffic==='expert'?'expert dispatch':'cache handoff'} payload is outside the ring all-reduce method. STRATUM derives ${formatBytes(l.sent)} under its own operation-specific ownership assumptions.`,
   `${l.scope} Network rates use decimal bytes per second and one-way NIC capacity; payload divided by useful rate is a bandwidth-only lower bound.`,
   ring?'applicable':'specialized',['book-ultrascale'],{sent:l.sent,rate:l.rate,seconds:l.seconds})];
 }
 const cache=cacheLedger(m,s),params=parameterLedger(m),gqa=m.architecture==='gqa';
 const notes=[row('book-gqa-cache','Cache storage and ownership',
  gqa?`The GQA method stores both K and V: 2 × batch × context × layers × KV heads × head dimension × bytes per element. The selected ${m.name} scenario stores ${formatBytes(cache.total)}. KV heads, rather than query heads, determine this payload.`:
  m.architecture==='ced-csa2'?`The generic GQA formula does not cover CSA2. STRATUM counts compressed cache storage at ${m.kvSourceLayers.length} unique source layers, plus the indexer and sliding-window state: ${formatBytes(cache.total)}. Shared cache can be read multiple times during execution; repeated reads consume bandwidth without duplicating retained storage.`:
  `The generic GQA formula does not cover the complete ${m.name} cache. STRATUM separately accounts for its latent, recurrent or compressed mechanisms: ${formatBytes(cache.total)}. These terms require architecture-specific implementation evidence.`,
  `${cache.exclusions.join('. ')}. Byte displays use binary KiB/GiB; book decimal GB examples are converted explicitly in the audit.`,
  gqa?'applicable':'specialized',['book-scaling-inference'],{total:cache.total})];
 if(s.view!=='serving')notes.push(row('book-gqa-parameters','Parameter scope',
  gqa?`The book projection method counts Q, K, V and output matrices from their dimensions. STRATUM also includes scoped embeddings, feed-forward matrices and normalization parameters, giving ${formatParams(params.resident)} resident parameters for ${m.name}.`:
  `The ordinary Transformer projection formula cannot validate the complete ${m.name} architecture. STRATUM's ${formatParams(params.resident)} resident-parameter scope still depends on its model-specific evidence.`,
  `${params.scope} Method references supplement publisher code, independent implementation review and runtime evidence; they cannot replace those requirements.`,
  gqa?'applicable':'specialized',['book-scaling-transformers','book-scaling-inference'],{resident:params.resident}));
 return notes;
}
