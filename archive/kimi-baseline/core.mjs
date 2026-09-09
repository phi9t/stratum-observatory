/** Kimi K3 scientific model. Byte units are bytes; reported counts are not exact checkpoint counts. */
export const SOURCES = [
 {id:'k3-config',title:'Kimi K3 · released configuration',author:'MoonshotAI · Hugging Face',url:'https://huggingface.co/moonshotai/Kimi-K3/blob/main/config.json',note:'93 layers, explicit KDA/MLA indices, 7168/3584 widths, 896/16/2 experts, 96 heads, NoPE, SiTU, vision and mixed MXFP4 exclusions. The released base text config sets num_nextn_predict_layers=0.'},
 {id:'k3-code',title:'Kimi K3 · attention and decoder implementation',author:'MoonshotAI · reference model',url:'https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py',note:'Projection dimensions, 64 shared-key channels retained under NoPE, depth-bank update order, short-convolution state and reference materialized KV. Kernel storage orientation can differ from the logical equations.'},
 {id:'k3-paper',title:'Kimi K3 · technical report',author:'Kimi Team · arXiv:2607.24653',url:'https://arxiv.org/html/2607.24653v1',note:'§2 architecture; Table 1 rounded 2.78T / 104.2B; §5 training, rollout, kernels and production serving. The report describes one MTP layer; do not silently add it to the released base configuration.'},
 {id:'k3-moe',title:'Stable LatentMoE · equations 11–13',author:'Kimi K3 report · §2.3',url:'https://arxiv.org/html/2607.24653v1#S2.SS3',note:'Shared experts operate at full width. Routed experts operate at latent width; RMSNorm is after weighted aggregation and before latent up-projection. Quantile-balancing bias affects selection, not mixture weights.'},
 {id:'k3-infra',title:'Kimi K3 · infrastructure and serving',author:'Kimi K3 report · §5',url:'https://arxiv.org/html/2607.24653v1#S5',note:'PP/VP, EP, ZeRO-1 and Pipeline ZeRO-2, KCP, MoonEP, Muon P2P, activation policy, joint prefix cache and resumable RL. Diagrams show dependencies, not measured timing or exact physical placement.'},
 {id:'flashkda',title:'FlashKDA · chunkwise attention kernels',author:'MoonshotAI',url:'https://github.com/MoonshotAI/FlashKDA',note:'CUTLASS-based KDA training/prefill implementation. A recurrent token mixer requires different kernels for prefill and decode.'},
 {id:'h100',title:'H100 SXM · product specifications',author:'NVIDIA',url:'https://www.nvidia.com/en-us/data-center/h100/',note:'80 GB, 3.35 TB/s HBM, 900 GB/s aggregate bidirectional NVLink; dense BF16 peak is half the advertised structured-sparse value.'},
 {id:'h200',title:'H200 SXM · product specifications',author:'NVIDIA',url:'https://www.nvidia.com/en-us/data-center/h200/',note:'141 GB, 4.8 TB/s HBM. Advertised capacity is not measured allocatable memory.'},
 {id:'dgx',title:'DGX H100/H200 · system user guide',author:'NVIDIA',url:'https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html',note:'Eight GPUs, four NVSwitches and eight 400 Gb/s compute adapters. This is an illustrative Hopper substrate, not a claim about Moonshot’s physical cluster.'},
 {id:'fabric',title:'NVSwitch fabric topology',author:'NVIDIA Fabric Manager',url:'https://docs.nvidia.com/datacenter/tesla/fabric-manager-user-guide/index.html',note:'Logical connectivity rather than a physical cabling plan. Scale-up NVLink and scale-out NIC rates are different quantities.'},
 {id:'nccl',title:'Collective operation semantics',author:'NVIDIA NCCL',url:'https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html',note:'Payload formulas are explicitly selected analytic algorithms, not a guarantee of NCCL’s runtime choice.'},
 {id:'design',title:'Architecture schematic inspiration',author:'Edward Z. Yang',url:'https://deepseek-v3.ezyang.com/studies/01-deepseek-diagram.html',note:'Inspired by hierarchical operators, orthogonal wires and reserved label lanes. All schematics and application code in this atlas are original.'}
];
export const MODELS={kimi3:{id:'kimi3',name:'Kimi K3',subtitle:'Hybrid KDA–MLA · Attention Residuals · Stable LatentMoE',family:'kimi3',layers:93,kdaLayers:69,mlaLayers:24,d:7168,latent:3584,vocab:163840,ff:33792,heads:96,headDim:128,vDim:128,qRank:1536,kvRank:512,sharedKeyDim:64,nopeDim:128,experts:896,shared:2,topK:16,expertFF:3072,denseLayers:1,attnResBlock:12,convKernel:4,maxContext:1048576,reportedTotal:2.78e12,reportedActive:104.2e9,mtpInBase:0,sources:['k3-config','k3-code','k3-paper']}};
export const HARDWARE={
 h100:{id:'h100',name:'H100 SXM',memoryGB:80,memoryType:'HBM3',hbmGBs:3350,bf16TF:989.5,nvlinkGBsBidirectional:900,pcieGBsBidirectional:128,tdpW:700,sources:['h100','dgx','fabric']},
 h200:{id:'h200',name:'H200 SXM',memoryGB:141,memoryType:'HBM3e',hbmGBs:4800,bf16TF:989.5,nvlinkGBsBidirectional:900,pcieGBsBidirectional:128,tdpW:700,sources:['h200','dgx','fabric']}
};
export const DEFAULT_STATE={view:'model',model:'kimi3',hardware:'h200',theme:'dark',layer:2,detail:'shapes',trainingMode:'pretrain',context:32768,batch:1,kvPrecision:2,statePrecision:4,weightMode:'mixed',cacheMode:'compressed',replicaGPUs:32,nodes:8,serveMode:'disaggregated',trafficMode:'moe',tokens:4096,utilization:70,oversub:1,rail:0,intensity:64,match:2800,zoom:1,selected:null};
function count(n,name,zero=false){if(!Number.isSafeInteger(n)||(zero?n<0:n<=0))throw new RangeError(`${name} must be a ${zero?'non-negative':'positive'} safe integer`);}
function precision(n){if(![1,2,4].includes(n))throw new RangeError('Unsupported byte precision');}
export function layerKind(layer){count(layer,'layer');if(layer>93)throw new RangeError('layer > 93');return layer%4===0||layer===93?'mla':'kda';}
export function parameterLedger(m=MODELS.kimi3){
 const routed=(m.layers-1)*m.experts*3*m.latent*m.expertFF;
 const shared=(m.layers-1)*m.shared*3*m.d*m.expertFF;
 const latent=(m.layers-1)*2*m.d*m.latent;
 const routers=(m.layers-1)*(m.d*m.experts+m.experts);
 const embedding=m.d*m.vocab,dense=3*m.d*m.ff;
 return {total:m.reportedTotal,active:m.reportedActive,routed,shared,latent,routers,embedding,head:embedding,dense,rounded:true,expertPerLayer:3*m.latent*m.expertFF};
}
export function weightBudget(mode='mixed'){
 const p=parameterLedger();
 if(mode==='bf16')return {routed:p.routed*2,other:(p.total-p.routed)*2,total:p.total*2,approximate:true};
 if(mode!=='mixed')throw new RangeError('Unknown weight mode');
 // Only routed experts are budgeted at MXFP4: 4 bits + one uint8 scale per 32 values.
 // Remaining weights are a BF16 scenario using a rounded reported total, not a manifest sum.
 const routed=p.routed*(.5+1/32),other=(p.total-p.routed)*2;
 return {routed,other,total:routed+other,approximate:true};
}
export function cacheBudget(context,batch,kvPrecision=2,statePrecision=4,mode='compressed'){
 count(context,'context',true);count(batch,'batch',true);precision(kvPrecision);precision(statePrecision);
 if(!['compressed','expanded'].includes(mode))throw new RangeError('Unknown cache mode');
 const m=MODELS.kimi3,mlaElements=mode==='compressed'?m.kvRank+m.sharedKeyDim:m.heads*(m.nopeDim+m.sharedKeyDim+m.vDim);
 const mla=m.mlaLayers*context*batch*mlaElements*kvPrecision;
 const recurrent=m.kdaLayers*m.heads*m.headDim*m.vDim*batch*statePrecision;
 // Reference-shaped streaming convolution buffers: 3 projections, kernel width 4, BF16.
 const conv=m.kdaLayers*3*m.heads*m.headDim*m.convKernel*batch*2;
 return {mla,recurrent,conv,total:mla+recurrent+conv,mlaElements,mlaPerToken:m.mlaLayers*mlaElements*kvPrecision};
}
export function moeTraffic(tokens,ranks,bytes=2){count(tokens,'tokens',true);count(ranks,'ranks');precision(bytes);const logical=tokens*16*3584*bytes;return {logical,fullWidth:logical*2,sent:logical*(ranks-1)/ranks};}
export function kcpTraffic(ranks,bytes=4){count(ranks,'ranks');precision(bytes);const local=96*(128*128+128*128)*bytes;return {local,sent:(ranks-1)*local};}
export function prefixHit(matched,checkpointGroups,hash=512){count(matched,'matched',true);count(hash,'hash');if(!Array.isArray(checkpointGroups)||!checkpointGroups.length)return 0;const bound=Math.floor(matched/hash)*hash;return Math.max(0,...checkpointGroups[0].filter(v=>Number.isSafeInteger(v)&&v>0&&v<=bound&&v%hash===0&&checkpointGroups.every(a=>a.includes(v))));}
export function roofline(h,intensity){if(!Number.isFinite(intensity)||intensity<=0)throw new RangeError('Invalid intensity');return Math.min(h.bf16TF,h.hbmGBs*intensity/1000);}
export function derived(s){
 const m=MODELS.kimi3,h=HARDWARE[s.hardware],p=parameterLedger(m),cache=cacheBudget(s.context,s.batch,s.kvPrecision,s.statePrecision,s.cacheMode),weight=weightBudget(s.weightMode);
 const dispatch=moeTraffic(s.tokens,s.nodes,2),kcp=kcpTraffic(s.nodes,s.statePrecision),handoff=cacheBudget(s.context,1,s.kvPrecision,s.statePrecision,s.cacheMode);
 const traffic=s.trafficMode==='moe'?dispatch.sent:s.trafficMode==='kcp'?kcp.sent:handoff.total;
 const networkGBs=50*s.utilization/100/s.oversub;
 return {m,h,p,cache,weight,weights:weight.total,kv:cache.total,kvPerToken:cache.mlaPerToken,world:s.nodes*8,perGPU:(cache.total+weight.total)/s.replicaGPUs,dispatch,kcp,handoff,traffic,networkGBs,networkMs:traffic/(networkGBs*1e9)*1000,transferMs:handoff.total/(networkGBs*1e9)*1000,ridge:h.bf16TF*1000/h.hbmGBs,roofTF:roofline(h,s.intensity),hit:prefixHit(s.match,[[512,1536,2560,4096,6144],[512,2560,4096,6144]])};
}
export function fmt(n,d=1){return new Intl.NumberFormat('en-US',{maximumFractionDigits:d}).format(n);}
export function compact(n,d=2){return n>=1e12?`${fmt(n/1e12,d)}T`:n>=1e9?`${fmt(n/1e9,d)}B`:n>=1e6?`${fmt(n/1e6,d)}M`:n>=1e3?`${fmt(n/1e3,d)}k`:fmt(n,d);}
export function gb(n){return `${fmt(n/1e9,2)} GB`;}
export function sanitizeState(input={}){
 const out={...DEFAULT_STATE};
 const enums={view:['model','training','serving','hardware','network'],model:['kimi3'],hardware:['h100','h200'],theme:['dark','light'],detail:['operators','shapes'],trainingMode:['pretrain','agentic'],serveMode:['disaggregated','colocated'],weightMode:['mixed','bf16'],cacheMode:['compressed','expanded'],trafficMode:['moe','kcp','cache']};
 for(const [k,values] of Object.entries(enums))if(values.includes(input[k]))out[k]=input[k];
 const choices={context:[1024,4096,8192,32768,131072,262144,524288,1048576],batch:[1,4,8,16,32,64,128],nodes:[1,2,4,8,16,32,64],replicaGPUs:[8,16,32,64,128],kvPrecision:[1,2],statePrecision:[2,4],tokens:[256,1024,4096,16384,65536],oversub:[1,2,4],rail:[0,1,2,3,4,5,6,7],intensity:[1,4,16,64,256,1024,4096],match:[512,2048,2800,4096,6144]};
 for(const [k,values] of Object.entries(choices))if(values.includes(Number(input[k])))out[k]=Number(input[k]);
 if(Number.isInteger(Number(input.layer))&&input.layer>=1&&input.layer<=93)out.layer=Number(input.layer);
 if(Number.isFinite(Number(input.utilization)))out.utilization=Math.min(100,Math.max(10,Math.round(Number(input.utilization))));
 return out;
}
