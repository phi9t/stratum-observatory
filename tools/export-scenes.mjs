/** Scene IR plus the model truths each scene must agree with.
 *
 * The scene comes from the diagram builders; `expect` comes from core.mjs, the same
 * functions that produce the ledgers. A verifier comparing them is checking that the
 * picture agrees with the mathematics, not that the picture is self-consistent.
 */
import * as core from '../src/core.mjs';
import {buildScene} from '../src/diagrams.mjs';

const compactInfo=info=>info?{category:info.category,sources:(info.sources||[]).length,facts:(info.facts||[]).length}:null;
const compact=g=>({
 width:g.width,height:g.height,caption:g.caption,defaultSelected:g.defaultSelected,
 nodes:g.nodes.map(n=>({id:n.id,x:n.x,y:n.y,w:n.w,h:n.h,name:n.name,sub:n.sub,kind:n.kind,tone:n.tone,small:!!n.small,info:compactInfo(n.info)})),
 edges:g.edges.map(e=>({id:e.id,source:e.source,target:e.target,kind:e.kind,arrow:e.arrow,delay:!!e.delay,points:e.points})),
 groups:g.groups.map(x=>({id:x.id,x:x.x,y:x.y,w:x.w,h:x.h,title:x.title,subtitle:x.subtitle})),
 labels:g.labels.map(l=>({id:l.id,text:l.text,x:l.x,y:l.y,tone:l.tone,size:l.size??null})),
 decorations:g.decorations.map(d=>({kind:d.kind,layer:d.layer,id:d.id,x:d.x,y:d.y,w:d.w,h:d.h,attention:d.attention,active:!!d.active,dense:!!d.dense})),
});

/** What core.mjs says is true for this state, independent of how it is drawn. */
function expectations(state){
 const m=core.getModel(state.model);
 const out={model:m.id,layers:m.layers,experts:m.experts,topK:m.topK,shared:m.shared,denseLayers:m.denseLayers,
  vision:!!m.vision,residual:m.residual,latent:!!m.latent,architecture:m.architecture,
  partition:core.layerPartition(m.id)};
 if(state.view==='model'&&state.stage==='block'){
  out.layerKind=core.layerKind(m.id,state.layer);
  out.ffnKind=core.ffnKind(m.id,state.layer);
 }
 return out;
}

const scenes=[];
function emit(key,patch){
 const state=core.sanitizeState({...core.DEFAULT_STATE,...patch});
 scenes.push({key,state,scene:compact(buildScene(state)),expect:expectations(state)});
}

for(const [model,m] of Object.entries(core.MODELS)){
 for(let layer=1;layer<=m.layers;layer++)emit(`${model}/model/block/${layer}`,{model,view:'model',stage:'block',layer});
 for(const view of core.VIEWS)if(view!=='model')emit(`${model}/${view}/block/4`,{model,view,layer:4});
 if(m.vision)emit(`${model}/model/inputs/4`,{model,view:'model',stage:'inputs'});
 if(m.residual!=='standard')emit(`${model}/model/residual/4`,{model,view:'model',stage:'residual'});
 if(m.pleLayers.length)emit(`${model}/model/ple/4`,{model,view:'model',stage:'ple'});
 for(const compare of Object.keys(core.MODELS))emit(`${model}/compare/${compare}/4`,{model,view:'compare',compare});
 for(const trainMode of ['fsdp','ddp','rl'])emit(`${model}/training/${trainMode}/4`,{model,view:'training',trainMode});
 for(const placement of ['split','colocated'])emit(`${model}/serving/${placement}/4`,{model,view:'serving',placement});
}
for(const hardware of Object.keys(core.HARDWARE))emit(`hw/${hardware}/block/4`,{view:'hardware',hardware});
for(const fabric of ['ib','roce'])for(const library of ['nccl','nvshmem','nixl'])for(const traffic of ['expert','allreduce','handoff'])
 emit(`net/${fabric}-${library}-${traffic}/block/4`,{view:'network',fabric,library,traffic});

process.stdout.write(JSON.stringify({schema:'stratum-scenes/v1',count:scenes.length,scenes}));
