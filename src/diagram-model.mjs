import {getModel,layerKind,layerPartition,ffnKind,parameterLedger,formatNum,formatParams} from './core.mjs';
import {makeScene,addNode,addGroup,addLabel,addWire,addSum,makeInfo} from './scene.mjs';
const F=formatNum;
function modelInfo(m,title,body,formula='',facts=[],links=[]){return makeInfo('Model architecture',title,body,formula,facts,m.sources,links);}
function labelShape(g,s,text,x,y,tone='state'){if(s.detail==='shapes')addLabel(g,text,x,y,tone);}
function node(g,m,id,x,y,w,h,title,sub,body,formula='',facts=[],options={}){return addNode(g,id,x,y,w,h,title,sub,{info:modelInfo(m,title,body,formula,facts),...options});}
function stackOverview(g,m,s){
 addLabel(g,'MODEL TOPOLOGY',20,34,'muted',{size:11,tracking:1.5});
 if(m.vision){
  node(g,m,'text-input',20,70,74,42,'Text','tokens','The tokenizer and embedding lookup produce residual-width text embeddings. Tokenizer revision and special tokens are separate evidence obligations.','E[token_id]',[],{small:true});
  node(g,m,'vision-input',112,70,74,42,'Vision','patches','A native vision tower maps patches to visual representations before projection and interleaving with text. Use the Inputs control for the expanded path.','patches → ViT → connector',[],{small:true,tone:'compute'});
  addWire(g,[[57,112],[57,138],[50,138],[50,164]],'text-input','overview-block');
  addWire(g,[[149,112],[149,138],[132,138],[132,164]],'vision-input','overview-block');
 }else{
  node(g,m,'embedding',20,70,166,52,'Embedding',`${F(m.vocab)} × ${F(m.d)}`,'Untied vocabulary embedding. The complete weight table is resident even though a token performs a lookup.','x₀ = E[token]',[['Width',F(m.d)]]);
  addWire(g,[[40,122],[40,164]],'embedding','overview-block');
 }
 node(g,m,'overview-block',20,164,166,58,'Decoder stack',`${m.layers} layers · d = ${F(m.d)}`,'Select a layer in the grid. Attention and feed-forward motifs are derived from the selected checkpoint, not from a family name.','xₗ₊₁ = Blockₗ(xₗ)',Object.entries(layerPartition(m)).filter(x=>x[1]).map(([k,v])=>[k.toUpperCase(),String(v)]),{tone:'compute'});
 addLabel(g,'SELECT A LAYER',20,253,'muted',{size:10.5,tracking:1});
 for(let l=1;l<=m.layers;l++){const i=l-1;g.decorations.push({kind:'layer',layer:l,id:`layer-${l}`,x:20+(i%6)*28,y:274+Math.floor(i/6)*27,w:23,h:21,attention:layerKind(m,l),active:l===s.layer,dense:ffnKind(m,l)==='dense'});}
 const gy=274+Math.ceil(m.layers/6)*27;
 addLabel(g,'Numbers are one-indexed.',20,gy+20,'muted',{size:10.5});
 addLabel(g,'Amber = recurrent layer',20,gy+39,'state',{size:10.5});
 const oy=Math.max(gy+75,755);
 node(g,m,'final-norm',20,oy,166,34,'Final RMSNorm', '', 'Normalize the final decoder representation before the output vocabulary projection.','RMSNorm(x)',[],{kind:'pill'});
 node(g,m,'lm-head',20,oy+69,166,52,'LM head',`${F(m.d)} → ${F(m.vocab)}`,'Untied output vocabulary projection. Softmax/sampling behavior does not change the block architecture.','logits = W_lm x',[],{});
 addWire(g,[[40,oy+34],[40,oy+69]],'final-norm','lm-head');
 addLabel(g,'logits / sampling',20,oy+154,'muted',{size:11});
}
function residuals(g,m,s){
 const special=m.residual!=='standard';
 const normTitle=m.residual==='attnres'?'Depth read · RMSNorm':m.residual==='hyper'?'Stream read · RMSNorm':m.residual==='mhc'?'Mixed stream read · RMSNorm':'RMSNorm';
 const desc=m.residual==='attnres'?'Block Attention Residuals retrieve across depth sources per token. The current partial block accumulator is updated and sealed at block boundaries; this is not an ordinary uniform residual sum.':m.residual==='hyper'?'The checkpoint carries four residual streams. A learned input mixer reads into one computation stream; an output gate writes back. It is not a single ordinary residual addition.':m.residual==='mhc'?`The checkpoint carries ${m.streams} residual streams mixed by manifold hyper-connections. The coefficients that mix the streams entering this sublayer were computed by the previous sublayer, so the matrix reading into attention came out of the previous block's feed-forward. It is made doubly stochastic by ${m.sinkhornIters} Sinkhorn iterations, which redistributes the streams without rescaling them. It is not a single ordinary residual addition.`:'Pre-normalization leaves a separate residual bypass untouched. After the attention output projection the update is added to the residual.';
 node(g,m,'norm1',270,112,468,40,normTitle,`${F(m.d)} channels`,desc,m.residual==='attnres'?'h = Σᵢ softmax(qᵀ norm(bᵢ))ᵢ bᵢ':m.residual==='hyper'?'h = H_pre(x₁,…,x₄)':m.residual==='mhc'?'M = Sinkhorn(W_prev)\nh = RMSNorm(Σⱼ Mᵢⱼ xⱼ)':'h = RMSNorm(x)',[],{kind:'pill'});
 node(g,m,'norm2',886,112,486,40,normTitle,`${F(m.d)} channels`,desc,'h₂ = normalize(read(x₁))',[],{kind:'pill'});
 const stream=m.residual==='hyper'?'4 streams':m.residual==='mhc'?`${m.streams} streams`:m.residual==='attnres'?'depth bank + partial sum':'x';
 const stream2=m.residual==='hyper'?'4 streams':m.residual==='mhc'?`${m.streams} streams`:m.residual==='attnres'?'updated depth bank':'x₁';
 addLabel(g,`residual stream · ${stream}`,250,80,'muted',{size:12});
 addLabel(g,`residual stream · ${stream2}`,920,80,'muted',{size:12});
 // The stream arrives from outside the block and splits: normalized into attention, and
 // straight down the left rail as the mainline that bypasses it.
 addWire(g,[[206,86],[286,86],[286,112]],'', 'norm1');
 addWire(g,[[238,86],[238,899],[270,899]],'', 'merge-attn',{kind:'residual'});
 if(special){
  const writeTitle=m.residual==='hyper'?'Gated stream write':m.residual==='mhc'?'Mixed stream write':'Update block accumulator';
  node(g,m,'merge-attn',270,884,230,30,writeTitle,'',desc,m.residual==='mhc'?'xᵢ′ = Σⱼ Mᵢⱼ xⱼ + Δ;  W_next = f(h)':'update residual state',[],{kind:'pill',tone:'state'});
  node(g,m,'merge-ffn',886,934,230,30,writeTitle,'',desc,m.residual==='mhc'?'xᵢ′ = Σⱼ Mᵢⱼ xⱼ + Δ;  W_next = f(h)':'update residual state',[],{kind:'pill',tone:'state'});
 }else{
  addSum(g,'merge-attn',286,899,{info:modelInfo(m,'Attention residual add','Combine the attention output with the unchanged residual stream.','x₁ = x + Attention(RMSNorm(x))')});
  addLabel(g,'residual add',312,904,'muted',{size:12});
  addSum(g,'merge-ffn',902,949,{info:modelInfo(m,'FFN residual add','Combine the feed-forward output with the residual from the attention sublayer.','x₂ = x₁ + FFN(RMSNorm(x₁))')});
  addLabel(g,'residual add',928,954,'muted',{size:12});
 }
 const ay=special?914:911;
 addWire(g,[[286,866],[286,special?884:887]],'out-proj','merge-attn');
 addWire(g,[[286,ay],[286,925],[806,925],[806,82],[902,82],[902,112]],'merge-attn','norm2',{kind:'residual'});
 addWire(g,[[806,925],[806,949],[special?886:890,949]],'merge-attn','merge-ffn',{kind:'residual'});
 addWire(g,[[902,special?964:961],[902,986]],'merge-ffn','');
 addLabel(g,'x₂ · block output',923,985,'muted',{size:12});
}
function mla(g,m,s){
 const xq=286,xk=532;
 node(g,m,'q-down',270,236,222,48,'Q down-projection',`${F(m.d)} → ${F(m.qRank)}`,'Compress the query projection before expanding per-head queries. Query latent states are not the autoregressive KV cache.','cᑫ = W_DQ h');
 node(g,m,'kv-down',516,236,222,48,'KV down-projection',`${F(m.d)} → ${m.kvRank} + ${m.ropeDim}`,'Produce a shared compressed content latent plus a separate 64-channel key component. Kimi K3 keeps that component even though its MLA uses NoPE.','[cᴷⱽ, k_shared] = W_DKV h');
 addWire(g,[[286,152],[286,218],[286,236]],'norm1','q-down');addWire(g,[[286,218],[532,218],[532,236]],'norm1','kv-down');
 for(const[id,x,w,rank]of[['q',270,222,m.qRank],['kv',516,222,m.kvRank]]){
  node(g,m,`${id}-latent-norm`,x,332,w,32,'Latent RMSNorm',String(rank),'Normalize the low-rank latent before the learned up-projection.','ĉ = RMSNorm(c)',[],{kind:'pill'});
  addWire(g,[[x+16,284],[x+16,332]],`${id}-down`,`${id}-latent-norm`);
  labelShape(g,s,`${id} latent · ${rank}`,x+32,311);
 }
 node(g,m,'q-up',270,410,222,48,'Q up-projection',`${m.qRank} → ${m.heads} × ${m.nopeDim+m.ropeDim}`,'Expand query heads. Per-head query width includes content and the additional positional/shared-key subspace.','Q = reshape(W_UQ ĉᑫ)');
 node(g,m,'kv-up',516,410,222,48,'KV up-projection',`${m.kvRank} → ${m.heads} × ${m.nopeDim+m.vDim}`,'Conceptual expanded K/V heads. Optimized decode can absorb these projections and retain the compressed cache instead.','[K_content,V] = reshape(W_UKV ĉᴷⱽ)');
 addWire(g,[[xq,364],[xq,410]],'q-latent-norm','q-up');addWire(g,[[xk,364],[xk,410]],'kv-latent-norm','kv-up');
 labelShape(g,s,'normalized query latent',302,391);labelShape(g,s,'normalized KV latent',548,391);
 node(g,m,'q-position',270,510,222,32,m.id==='kimi3'?'NoPE query heads':'Partial RoPE',`${m.nopeDim} + ${m.ropeDim} per head`,m.id==='kimi3'?'No rotary operation is inserted. The extra key component is a learned shared subspace, not deleted storage.':'Apply rotary position encoding to the decoupled positional part, not to the values.','Q = [Q_content, Q_shared]',[],{kind:'pill'});
 node(g,m,'kv-build',516,510,222,32,m.id==='kimi3'?'Build K/V · NoPE':'RoPE + build K/V','shared key + head content','Concatenate the shared key component with each head’s content keys. Values are not rotated.','K = [K_content, k_shared]',[],{kind:'pill'});
 addWire(g,[[xq,458],[xq,510]],'q-up','q-position');addWire(g,[[xk,458],[xk,510]],'kv-up','kv-build');
 addWire(g,[[738,260],[784,260],[784,526],[738,526]],'kv-down','kv-build',{kind:'state'});
 labelShape(g,s,`${m.heads} query heads`,302,488);labelShape(g,s,`${m.heads} K/V heads`,548,488);labelShape(g,s,'key · 64',656,311);
 node(g,m,'attention',270,600,468,58,'Causal attention',`${m.heads} heads · Q/K ${m.nopeDim+m.ropeDim} · V ${m.vDim}`,'This is the mathematical expanded-head view. A compressed MLA decode implementation performs an equivalent factorization; it does not store these full heads by necessity.','O = softmax(QKᵀ / √d_k + causal mask) V',[['Compressed cache per token/layer',`${m.kvRank+m.ropeDim} elements`]],{tone:'compute'});
 addWire(g,[[xq,542],[xq,600]],'q-position','attention');addWire(g,[[xk,542],[xk,600]],'kv-build','attention');
 labelShape(g,s,`Q · ${m.heads} × ${m.nopeDim+m.ropeDim}`,302,579);labelShape(g,s,`K,V · ${m.heads} heads`,548,579);
 finishAttention(g,m,s,658,m.heads*m.vDim);
}
function finishAttention(g,m,s,fromY,width){
 if(m.gate){
  node(g,m,'attention-gate',270,746,468,32,'Output gate',`head output · ${F(width)}`,'An input-dependent gate modulates attention output before the output projection. The gate pathway comes from the normalized layer input.','O_g = O ⊙ gate(h)',[],{kind:'pill'});
  addWire(g,[[286,fromY],[286,746]],'attention','attention-gate');
  addWire(g,[[738,132],[766,132],[766,762],[738,762]],'norm1','attention-gate',{kind:'state'});
  addWire(g,[[286,778],[286,818]],'attention-gate','out-proj');
 }else addWire(g,[[286,fromY],[286,818]],'attention','out-proj');
 labelShape(g,s,`attention output · ${F(width)}`,302,718);
 node(g,m,'out-proj',270,818,468,48,'Attention output projection',`${F(width)} → ${F(m.d)}`,'Return the concatenated head outputs to residual-stream width. Head-space width need not equal residual width.','Δx = W_O concat(O₁,…,O_H)');
}
function gqa(g,m,s){
 for(const[id,x,heads]of[['q',270,m.heads],['k',434,m.kvHeads],['v',598,m.kvHeads]]){
  node(g,m,`${id}-proj`,x,236,140,48,`${id.toUpperCase()} projection`,`${heads} × ${m.headDim}`,'Project normalized hidden states into head space. Queries and key/value heads are different tensor contracts.','projection(h)',[['Input width',F(m.d)],['Head count',String(heads)],['Head width',String(m.headDim)]]);
  addWire(g,[[286,152],[286,218],[x+16,218],[x+16,236]],'norm1',`${id}-proj`);
  if(id!=='v'){
   node(g,m,`${id}-norm`,x,332,140,32,'Head RMSNorm',`${m.headDim} channels`,'Qwen applies per-head RMS normalization to queries and keys before position encoding. Values bypass this normalization.','q̂ = RMSNorm(q); k̂ = RMSNorm(k)',[],{kind:'pill'});
   node(g,m,`${id}-rope`,x,410,140,32,m.ropeDim?'Partial RoPE':'RoPE',m.ropeDim?`${m.ropeDim} rotary dims`:`${m.headDim} rotary dims`,'Position encoding acts on Q and K; values are not rotated. Multimodal position conventions are checkpoint-specific.','rotate(Q,K; position)',[],{kind:'pill'});
   addWire(g,[[x+16,284],[x+16,332]],`${id}-proj`,`${id}-norm`);addWire(g,[[x+16,364],[x+16,410]],`${id}-norm`,`${id}-rope`);
  }
 }
 node(g,m,'kv-cache',434,510,304,40,'K/V history',`${m.kvHeads} heads · append one position`,'Each global-attention layer retains key and value history. GQA reduces cache heads, not the number of query heads.','bytes = 2 × H_KV × D × B × S × b',[],{tone:'state',kind:'pill'});
 addWire(g,[[450,442],[450,510]],'k-rope','kv-cache');addWire(g,[[614,284],[614,510]],'v-proj','kv-cache');
 labelShape(g,s,`V · ${m.kvHeads} × ${m.headDim}`,632,476);
 node(g,m,'attention',270,600,468,58,'Grouped-query attention',`${m.heads} query heads / ${m.kvHeads} KV heads`,'Each key/value head is shared by a group of query heads. The softmax remains causal over the available key history.','O = softmax(QKᵀ / √d_h + mask) V',[['Query heads per KV head',String(m.heads/m.kvHeads)]],{tone:'compute'});
 addWire(g,[[286,442],[286,600]],'q-rope','attention');addWire(g,[[450,550],[450,580],[532,580],[532,600]],'kv-cache','attention');
 labelShape(g,s,`Q · ${m.heads} × ${m.headDim}`,302,575);
 finishAttention(g,m,s,658,m.heads*m.headDim);
}
function recurrent(g,m,s,kind){
 const D=m.linearHeadDim,H=m.linearHeads,K=m.linearKeyHeads;
 node(g,m,'qkv-proj',270,236,282,48,'Q/K/V projections',`Q,K ${K} × ${D} · V ${H} × ${D}`,'Project queries, keys and values for recurrent attention. Grouped heads are expanded according to the implementation before the state update.','[q,k,v] = W_qkv h');
 node(g,m,'decay-gate',588,236,150,48,'Decay / write',kind==='kda'?'vector α · scalar β':'scalar α · scalar β',kind==='kda'?'KDA has channel-wise decay, unlike scalar-decay Gated DeltaNet. The write strength β controls the delta correction.':'Gated DeltaNet combines scalar head-wise forgetting and a delta-rule write strength.','α, β = gates(h)',[],{small:true});
 addWire(g,[[286,152],[286,218],[286,236]],'norm1','qkv-proj');addWire(g,[[286,218],[604,218],[604,236]],'norm1','decay-gate');
 node(g,m,'short-conv',270,332,282,42,'Causal short convolution',`kernel ${m.convKernel} · SiLU`,'A short causal convolution precedes the recurrence. Streaming execution keeps convolution history separate from the recurrent matrix.','q,k,v = SiLU(ShortConv(W h))',[],{kind:'pill'});
 node(g,m,'qk-norm',270,430,282,32,'L2-normalize Q and K','V passes through','Normalize queries and keys for the delta update. This is L2 normalization, not the RMSNorm used by global Qwen attention.','q̂ = q / ||q||₂; k̂ = k / ||k||₂',[],{kind:'pill'});
 addWire(g,[[286,284],[286,332]],'qkv-proj','short-conv');addWire(g,[[286,374],[286,430]],'short-conv','qk-norm');
 labelShape(g,s,`${H} value heads · d = ${D}`,302,408);
 node(g,m,'gate-functions',588,364,150,48,'Gate transforms',kind==='kda'?'α ∈ ℝᵈ · β ∈ ℝ':'α, β per head','Forgetting must be applied before predicting and correcting the new value. Gate nonlinearities and parameterization are implementation-specific.','decay then delta-write',[],{small:true,kind:'pill'});
 addWire(g,[[604,284],[604,364]],'decay-gate','gate-functions');
 node(g,m,'state-prev',588,480,150,42,'S[t − 1]',`${H} × ${D} × ${D}`,'Persistent per-sequence recurrent matrix. The dashed feedback crosses a token-time boundary; it is not a same-step graph cycle.','S_prev → recurrent update',[],{tone:'state',small:true});
 node(g,m,'attention',270,570,468,64,kind==='kda'?'Kimi Delta Attention':'Gated DeltaNet',`${H} recurrent heads · no token-axis KV history`,kind==='kda'?'Apply vector decay to the previous state; predict the current value from the decayed state; write the delta correction; read the result with the query.':'Apply scalar head-wise decay to the previous state, then the delta correction. Prefill may use a chunked parallel algorithm equivalent to this recurrence.',kind==='kda'?'S̄ = S[t−1] diag(α)\nS[t] = S̄ + β (v − S̄k) kᵀ\no = S[t]q':'S̄ = α S[t−1]\nS[t] = S̄ + β (v − S̄k) kᵀ\no = S[t]q',[['State orientation','value × key'],['State elements / layer',F(H*D*D)],['Decode context dependence','Fixed current matrix; snapshots add storage']],{tone:'compute'});
 addWire(g,[[286,462],[286,570]],'qk-norm','attention');
 addWire(g,[[604,412],[604,448],[570,448],[570,546],[532,546],[532,570]],'gate-functions','attention');
 addWire(g,[[604,522],[604,552],[660,552],[660,570]],'state-prev','attention',{kind:'state'});
 addWire(g,[[738,602],[784,602],[784,501],[738,501]],'attention','state-prev',{kind:'delay',delay:true});
 node(g,m,'head-norm',270,686,468,32,'Per-head RMSNorm',`${H} × ${D}`,'Normalize the recurrent attention output per head before its output gate. State storage and output activation precision are separate choices.','û = RMSNorm_head(o)',[],{kind:'pill'});
 addWire(g,[[286,634],[286,686]],'attention','head-norm');
 node(g,m,'attention-gate',270,746,468,32,`${m.linearGate} output gate`,`${H} × ${D}`,'The gate derives from normalized hidden input and modulates the normalized recurrent output before the output projection.','o_g = RMSNorm_head(o) ⊙ gate(W_z h)',[],{kind:'pill'});
 addWire(g,[[286,718],[286,746]],'head-norm','attention-gate');
 addWire(g,[[738,132],[766,132],[766,762],[738,762]],'norm1','attention-gate',{kind:'state'});
 node(g,m,'out-proj',270,818,468,48,'Attention output projection',`${F(H*D)} → ${F(m.d)}`,'Project concatenated recurrent heads back to residual width.','Δx = W_O concat(o_g)');
 addWire(g,[[286,778],[286,818]],'attention-gate','out-proj');
 labelShape(g,s,`Q/K normalized · V unchanged`,302,538);labelShape(g,s,`gated output · ${F(H*D)}`,302,800);
}
function sparseAttention(g,m,s){
 node(g,m,'qkv-proj',270,236,282,48,'Q/K/V projections',`${m.heads} Q heads · ${m.kvHeads} KV heads`,'Global sparse attention retains full key/value history; sparsity selects which past positions participate in attention for each query.','[Q,K,V] = projections(h)');
 node(g,m,'index-proj',588,236,150,48,'QSA indexer',`${m.indexerHeads} Q · 1 K · ${m.indexer}`,'An inexpensive separate indexer scores compressed key blocks. This is not the full attention Q/K projection.','q_I, k_I = W_I h',[],{small:true});
 addWire(g,[[286,152],[286,218],[286,236]],'norm1','qkv-proj');addWire(g,[[286,218],[604,218],[604,236]],'norm1','index-proj');
 node(g,m,'qk-position',270,332,282,32,'QK norm + partial RoPE',`${m.ropeDim} rotary channels per head`,'Normalize and position-encode query/key heads. The query path bypasses the KV cache write.','Q,K = RoPE(RMSNorm(Q,K))',[],{kind:'pill'});
 addWire(g,[[286,284],[286,332]],'qkv-proj','qk-position');
 node(g,m,'index-compress',588,332,150,42,'Block-average K',`${m.indexerCompression} tokens / block`,'Compressed index keys summarize completed microblocks. An incomplete current block needs separate handling.','k_block = mean(k_I)',[],{small:true,kind:'pill'});
 node(g,m,'index-select',588,430,150,48,'Select blocks',`${m.indexerBudget} token budget`,'The budget is in selected tokens from complete compressed blocks, not the total cache capacity. Preserve causal handling of the current incomplete block.','top blocks → token mask',[],{tone:'state',small:true});
 addWire(g,[[604,284],[604,332]],'index-proj','index-compress');addWire(g,[[604,374],[604,430]],'index-compress','index-select');
 node(g,m,'kv-cache',270,474,282,56,'Full K/V history','grows with sequence length','Sparse attention does not justify truncating resident KV history to the indexer budget. A separate compressed index cache also grows with context.','KV bytes ∝ S; selected attention reads ≤ budget + tail',[],{tone:'state'});
 addWire(g,[[286,364],[286,412],[350,412],[350,474]],'qk-position','kv-cache');
 node(g,m,'attention',270,600,468,58,'Qwen Sparse Attention',`${m.heads} Q / ${m.kvHeads} KV · selected causal positions`,'Evaluate attention only over selected valid positions. This drawing distinguishes query, resident KV and selection-mask paths.','O = softmax(Q K_selectedᵀ / √d_h) V_selected',[],{tone:'compute'});
 addWire(g,[[286,412],[250,412],[250,629],[270,629]],'qk-position','attention');
 addWire(g,[[350,530],[350,600]],'kv-cache','attention');
 addWire(g,[[604,478],[604,572],[660,572],[660,600]],'index-select','attention',{kind:'state'});
 finishAttention(g,m,s,658,m.heads*m.headDim);
}
function csa2Attention(g,m,s,kind){
 const sparse=kind!=='swa',scores=kind==='csa2-full'||kind==='csa2-reindex',l=s.layer-1;
 // The hierarchical candidate pool is a decoder mechanism: the report's Hierarchical Sparse
 // Indexer "is used only in the decoder of CED". Encoder scoring layers -- one-indexed 3, 9 and
 // 15 -- precede the candidate source entirely and score the whole causally visible range, so
 // they get no candidate-pool node and their prose must not name one.
 const pooled=scores&&l>=m.candidateSource;
 const kvOwner=m.kvSourceLayers.filter(x=>x<=l).pop(),idxOwner=m.indexSourceLayers.filter(x=>x<=l).pop();
 const qWidth=m.nopeDim+m.ropeDim,headWidth=m.heads*m.vDim;
 node(g,m,'qkv-proj',270,236,282,48,'Q/K/V projections',`${m.heads} Q heads · one latent KV × ${m.vDim}`,`Queries pass through a rank-${F(m.qRank)} bottleneck and expand to ${m.heads} heads of ${m.nopeDim} content plus ${m.ropeDim} rotary channels. Keys and values are a single ${m.vDim}-wide latent per position — kvHeads is ${m.kvHeads} — so one latent KV serves all ${m.heads} query heads. This is not one key/value pair per head.`,'[Q, k, v] = projections(h)',[['Query heads',String(m.heads)],['KV heads',String(m.kvHeads)],['Latent KV width',String(m.vDim)],['Query rank',F(m.qRank)]]);
 addWire(g,[[286,152],[286,218],[286,236]],'norm1','qkv-proj');
 node(g,m,'qk-position',270,332,282,32,'Partial RoPE',`${m.ropeDim} of ${qWidth} channels per head`,'Rotary position encoding acts on the decoupled positional channels of the query and the shared key. Content channels and the value latent are not rotated.','Q,k = RoPE(Q,k; position)',[],{kind:'pill'});
 addWire(g,[[286,284],[286,332]],'qkv-proj','qk-position');
 node(g,m,'window-kv',270,474,282,56,'Sliding-window KV',`${m.window} positions · one latent KV × ${m.vDim}`,`Every layer keeps a ${m.window}-position sliding window of the single ${m.vDim}-wide latent KV, written from this layer's own hidden state and stored as ${m.swaKvFormat}. One latent KV serves all ${m.heads} query heads, so the window is not ${m.heads} cached key/value pairs.`,'K_w, V_w = window(k, v)',[['Window',`${m.window} positions`],['KV heads',String(m.kvHeads)],['Stored as',m.swaKvFormat]],{tone:'state'});
 addWire(g,[[286,364],[286,412],[350,412],[350,474]],'qk-position','window-kv');
 addWire(g,[[286,412],[250,412],[250,629],[270,629]],'qk-position','attention');
 addWire(g,[[350,530],[350,600]],'window-kv','attention');
 labelShape(g,s,`Q · ${m.heads} × ${qWidth}`,366,452);
 if(sparse){
  if(scores){
   node(g,m,'index-proj',588,236,150,48,'CSA2 indexer',`${m.indexerHeads} Q · 1 K · ${m.indexer}`,`A separate inexpensive indexer scores the compressed cache. It is not the full attention Q/K projection, and its output is a selection rather than a value. ${pooled?`This is a decoder layer, so it scores the hierarchical candidate pool layer ${m.candidateSource+1} maintains.`:'This is an encoder layer: the hierarchical candidate pool is a decoder-only mechanism, so the pass runs over the whole causally visible range and no candidate set is involved.'}`,'q_I, k_I = W_I h',[['Scope',pooled?'candidate pool':'full causal range']],{small:true});
   addWire(g,[[286,218],[604,218],[604,236]],'norm1','index-proj');
   if(pooled){
    node(g,m,'index-blocks',588,332,150,42,'Candidate blocks',`${F(m.candidateBlocks)} × ${m.candidateBlockSize}`,`Indexer scores are taken over ${F(m.candidateBlocks)} candidate blocks of ${m.candidateBlockSize} positions, drawn from the candidate set layer ${m.candidateSource+1} establishes. Scoring a block is not reading it.`,'score(q_I, k_block)',[],{small:true,kind:'pill'});
    addWire(g,[[604,284],[604,332]],'index-proj','index-blocks');
   }
  }
  node(g,m,'index-select',588,430,150,48,scores?'Select blocks':'Reuse selection',scores?`${m.indexerBudget} token budget`:`computed at layer ${idxOwner+1}`,scores?`Top-scoring positions become a token mask of at most ${m.indexerBudget} selected positions, ${pooled?`taken from the candidate pool layer ${m.candidateSource+1} maintains`:'ranked over the whole causally visible range because this encoder layer has no candidate pool'}. The budget counts selected tokens, not resident cache capacity, and the incomplete current block still needs causal handling.`:`This layer runs no indexer: it reads the token selection layer ${idxOwner+1} already computed. Drawing an indexer projection here would assert a computation this layer does not perform.`,scores?'top blocks → token mask':`mask ← layer ${idxOwner+1}`,[['Mode',scores?'re-scored here':'reused'],['Scope',scores?(pooled?'candidate pool':'full causal range'):`layer ${idxOwner+1}`],['Budget',`${m.indexerBudget} tokens`]],{tone:'state',small:true});
  if(pooled)addWire(g,[[604,374],[604,430]],'index-blocks','index-select');
  else if(scores)addWire(g,[[604,284],[604,430]],'index-proj','index-select');
  node(g,m,'compress-kv',588,510,150,56,'Compressed KV cache',kind==='csa2-full'?'written here':`owned by layer ${kvOwner+1}`,`The compressed KV cache holds one ${m.vDim}-wide latent per compressed position and is shared across layers instead of being rebuilt per layer. Layer ${kvOwner+1} compresses and owns it. ${kind==='csa2-full'?'This layer is that owner: it writes the cache from its own normalized hidden state.':kind==='csa2-reindex'?'This layer does not write it; it re-scores the same stored bytes with its own indexer, so the selection differs while the cache does not.':'This layer neither writes it nor re-scores it; it reads a selection made earlier in the stack.'} Stored as ${m.kvFormat}.`,'K_c, V_c = compress(h_owner)',[['Owned by',`layer ${kvOwner+1}`],['This layer',kind==='csa2-full'?'owns the cache':kind==='csa2-reindex'?'re-scores the cache':'reads a prior selection'],['Stored as',m.kvFormat]],{tone:'state',small:true});
  addWire(g,[[604,478],[604,510]],'index-select','compress-kv',{kind:'control'});
  addWire(g,[[604,566],[604,600]],'compress-kv','attention');
  if(kind==='csa2-full')addWire(g,[[738,132],[784,132],[784,538],[738,538]],'norm1','compress-kv',{kind:'state'});
  labelShape(g,s,`selected · ≤ ${F(m.indexerBudget)}`,620,592);
 }
 node(g,m,'attention',270,600,468,58,sparse?'CSA2 sparse attention':'Sliding-window attention',sparse?`${m.heads} Q heads · window ${m.window} + ≤ ${F(m.indexerBudget)} selected`:`${m.heads} Q heads · ${m.window}-position window only`,sparse?`Attention.forward concatenates the sliding-window KV and the selected compressed KV and runs one sparse_attn call over their union, with the per-head sink in the same softmax. Two cache paths, one attention: drawing two attention operations whose outputs are added would misrepresent the mechanism.`:`The first ${m.swaLayers} layers have no compressed path at all — their compression ratio is zero — so attention is the sliding window alone. The per-head sink joins the same softmax.`,sparse?'O = sparse_attn(Q, [K_w; K_sel], [V_w; V_sel], sink)':'O = sparse_attn(Q, K_w, V_w, sink)',[['Attention calls','1'],['KV paths',sparse?'window + compressed':'window only'],['KV heads',String(m.kvHeads)]],{tone:'compute'});
 node(g,m,'attn-sink',562,690,176,44,'Per-head attention sink',`${m.heads} learned logits`,'Each head carries one learned logit that enters the softmax denominator with no value attached, which lets a head attend to nothing. It is a parameter, not a cached position, and it sits inside the same softmax rather than being a second attention.','softmax([qᵀk …, sink_h])',[],{tone:'state',small:true});
 addWire(g,[[660,690],[660,658]],'attn-sink','attention',{kind:'state'});
 node(g,m,'out-proj',270,818,468,48,'Grouped output projection',`${m.oGroups} groups · rank ${F(m.oRank)} · ${F(headWidth)} → ${F(m.d)}`,`The output projection is block-diagonal in ${m.oGroups} groups: each group maps its own slice of the concatenated head outputs through a rank-${F(m.oRank)} factor. It is one projection with a constrained weight, not a dense ${F(headWidth)} × ${F(m.d)} matrix and not two stacked projections.`,'Δx = Σ_g W_O,g concat(O_g)',[['Groups',String(m.oGroups)],['Rank per group',F(m.oRank)],['Head-space width',F(headWidth)]]);
 addWire(g,[[286,658],[286,818]],'attention','out-proj');
 labelShape(g,s,`attention output · ${F(headWidth)}`,302,718);
}
function feedForward(g,m,s){
 const kind=ffnKind(m,s.layer),latent=kind==='latent-moe',D=m.latent||m.d;
 if(kind==='dense'){
  node(g,m,'ffn-up',886,308,486,56,'Gate + up projections',`${F(m.d)} → 2 × ${F(m.ff)}`,'Two learned projections form the gated feed-forward intermediate. Dense layers execute this full FFN for every token.','u = W_up h; g = W_gate h');
  node(g,m,'ffn-activation',886,472,486,36,m.id==='kimi3'?'SiTU activation':m.swigluLimit?`Clamped SwiGLU · ±${m.swigluLimit}`:'SwiGLU',`${F(m.ff)} channels`,m.id==='kimi3'?'Kimi uses its source-defined capped SiTU activation rather than ordinary SwiGLU. Caps are part of the numerical contract.':m.swigluLimit?`Apply SiLU to the gate branch and multiply by the up branch, then clamp the product to ±${m.swigluLimit}. The clamp is part of the numerical contract, not an optimization, and an unclamped SwiGLU is a different function.`:'Apply SiLU to the gate branch and multiply by the up branch.','a = activation(g) ⊙ u',[],{kind:'pill'});
  node(g,m,'ffn-down',886,684,486,56,'Down projection',`${F(m.ff)} → ${F(m.d)}`,'Return the FFN activation to residual width before the residual-state update.','Δx = W_down a');
  addWire(g,[[902,152],[902,308]],'norm2','ffn-up');addWire(g,[[902,364],[902,472]],'ffn-up','ffn-activation');addWire(g,[[902,508],[902,684]],'ffn-activation','ffn-down');addWire(g,[[902,740],[902,m.residual==='standard'?937:934]],'ffn-down','merge-ffn');
  labelShape(g,s,`gate / up · 2 × ${F(m.ff)}`,918,420);labelShape(g,s,`activation · ${F(m.ff)}`,918,603);labelShape(g,s,`FFN output · ${F(m.d)}`,918,860);
  return;
 }
 const shared=m.shared>0,x=886,w=shared?282:486;
 node(g,m,'router',x,236,w,48,'Router',`${F(m.d)} → ${m.experts} scores`,m.id==='ds2'?'Softmax routing with grouped selection; V2 differs from V3’s sigmoid scoring and bias-corrected selection.':m.scoring==='sqrtsoftplus'?`Router scores are sqrt-softplus of the linear score, scaled by ${m.routeScale}. A per-expert correction bias is added for selection only: it decides which experts run, while the mixture weights applied to their outputs come from the uncorrected scores. Ranking and weighting are deliberately different quantities.`:m.scoring==='sigmoid'?'Router scores come from the full-width hidden vector. Correction bias, where present, changes selection; selected mixture weights follow the source implementation, not the corrected ranking scores.':'Softmax scores select the top-k routed experts. No shared expert is added unless the checkpoint explicitly defines one.','scores = router(h)',[['Experts',String(m.experts)],['Selected / token',String(m.topK)]]);
 node(g,m,'top-k',x,318,w,32,'Top-k selection',`${m.topK} of ${m.experts} experts`,`Expert count and experts selected per token are separate axes. Routing is dynamic; the diagram shows grouped work rather than drawing hundreds of expert boxes.${m.scoring==='sqrtsoftplus'?' Selection reads the bias-corrected scores; the weights that reach the expert outputs do not.':''}`,'I = topk(scores)',[],{kind:'pill'});
 addWire(g,[[902,152],[902,236]],'norm2','router');addWire(g,[[902,284],[902,318]],'router','top-k');
 let dispatchSource='top-k',outY=350;
 if(latent){node(g,m,'latent-down',x,392,w,48,'Latent down-projection',`${F(m.d)} → ${F(m.latent)}`,'Routed experts operate on a latent-width vector; shared experts still receive full-width normalized input. Router logits never become expert activations.','z = W_D h');addWire(g,[[1168,334],[1180,334],[1180,488],[1168,488]],'top-k','dispatch',{kind:'control'});dispatchSource='latent-down';outY=440;addWire(g,[[1372,132],[1402,132],[1402,416],[1168,416]],'norm2','latent-down',{kind:'state'});}
 const dy=latent?472:392;
 node(g,m,'dispatch',x,dy,w,32,'Expert dispatch',`${m.topK} × ${F(D)} per token`,'Optional distributed implementation: permutation plus expert-parallel transfer. Local experts incur no off-rank payload. Tokens, not router logits, are dispatched.','dispatch(h or z, expert indices)',[],{kind:'comm',tone:'comm'});
 addWire(g,[[902,outY],[902,dy]],dispatchSource,'dispatch',{kind:latent?'flow':'control'});
 if(!latent)addWire(g,[[1372,132],[1402,132],[1402,dy+16],[x+w,dy+16]],'norm2','dispatch',{kind:'state'});
 const gy=latent?546:488,ay=gy+82,downY=gy+142;
 node(g,m,'expert-up',x,gy,w,48,'Expert gate / up',`${F(D)} → 2 × ${F(m.expertFF)}`,'Grouped execution of the selected experts. Every expert has distinct parameters; only the selected subset contributes to a token.','uᵢ = W_up,i z; gᵢ = W_gate,i z');
 node(g,m,'expert-activation',x,ay,w,32,m.id==='kimi3'?'SiTU · routed experts':m.swigluLimit?`Clamped SwiGLU · ±${m.swigluLimit}`:'SwiGLU · routed experts',`${m.topK} selected experts`,`Apply the checkpoint’s expert activation independently per selected expert. Mixture weights are applied to expert outputs, not to routing logits.${m.swigluLimit?` The SwiGLU product is clamped to ±${m.swigluLimit} inside every expert; the clamp is part of the numerical contract.`:''}`,'aᵢ = activation(gᵢ) ⊙ uᵢ',[],{kind:'pill'});
 node(g,m,'expert-down',x,downY,w,48,'Expert down-projection',`${F(m.expertFF)} → ${F(D)}`,'Map each selected expert’s intermediate back to the routed-path width.','eᵢ = W_down,i aᵢ');
 addWire(g,[[902,dy+32],[902,gy]],'dispatch','expert-up');addWire(g,[[902,gy+48],[902,ay]],'expert-up','expert-activation');addWire(g,[[902,ay+32],[902,downY]],'expert-activation','expert-down');
 let combineY=latent?800:736;
 if(latent){
  node(g,m,'latent-combine',x,758,w,34,'Weighted sum → RMSNorm','normalize once after aggregation','Kimi’s routed expert outputs are first weighted and summed, normalized once, then lifted to residual width. Normalizing each expert separately would change the model.','z_out = RMSNorm(Σᵢ pᵢ eᵢ)',[],{kind:'pill'});
  addWire(g,[[902,downY+48],[902,758]],'expert-down','latent-combine');
  node(g,m,'latent-up',x,812,w,40,'Latent up-projection',`${F(m.latent)} → ${F(m.d)}`,'Lift the aggregate routed contribution to the full residual width. Shared experts are added after this lift.','routed = W_U z_out');
  addWire(g,[[902,792],[902,812]],'latent-combine','latent-up');
 }else{
  node(g,m,'combine',x,combineY,w,40,'Combine expert outputs',`${m.topK} → 1 weighted output`,'Gather/return expert outputs and form the selected mixture. Distributed all-to-all is an implementation mapping, not an additional model layer.','routed = Σᵢ pᵢ eᵢ',[],{kind:'comm',tone:'comm'});
  addWire(g,[[902,downY+48],[902,combineY]],'expert-down','combine');
 }
 addWire(g,[[x+w,334],[x+w+12,334],[x+w+12,latent?775:756],[x+w,latent?775:756]],'top-k',latent?'latent-combine':'combine',{kind:'control'});
 const routedLast=latent?'latent-up':'combine';const lastY=latent?852:776;
 if(shared){
  const sx=1196,sw=176;
  node(g,m,'shared-up',sx,gy,sw,48,'Shared gate / up',`${F(m.d)} → 2 × ${F(m.expertFF)}`,'The shared expert branch consumes the full-width hidden input, independently of the selected routed experts.','u_s,g_s = W_s h',[['Shared experts',String(m.shared)]],{small:true});
  node(g,m,'shared-activation',sx,ay,sw,32,m.id==='kimi3'?'SiTU':m.swigluLimit?`Clamped SwiGLU · ±${m.swigluLimit}`:'SwiGLU',`${m.shared} always-on`, `Shared expert activations follow the checkpoint. Qwen Flash-Next additionally gates its shared expert contribution.${m.swigluLimit?` The shared branch uses the same clamp at ±${m.swigluLimit} as the routed experts.`:''}`,'a_s = act(g_s) ⊙ u_s',[],{kind:'pill',small:true});
  node(g,m,'shared-down',sx,downY,sw,48,'Shared down',`${F(m.expertFF)} → ${F(m.d)}`,'Return full-width shared-expert outputs. Where multiple shared experts exist, their contribution is always active.','shared = Σ W_down,s a_s',[],{small:true});
  addWire(g,[[902,152],[902,214],[1212,214],[1212,gy]],'norm2','shared-up',{kind:'residual'});
  addWire(g,[[1212,gy+48],[1212,ay]],'shared-up','shared-activation');addWire(g,[[1212,ay+32],[1212,downY]],'shared-activation','shared-down');
  let sharedLast='shared-down',sharedBottom=downY+48;
  if(m.id==='q38flash'){node(g,m,'shared-gate',sx,736,sw,32,'Shared expert gate','sigmoid(h · w)','Flash-Next includes an input-dependent shared expert gate; it is not identical to the ungated shared experts in DeepSeek.','shared = σ(wᵀh) · e_shared',[],{kind:'pill',small:true});addWire(g,[[1212,sharedBottom],[1212,736]],'shared-down','shared-gate');sharedLast='shared-gate';sharedBottom=768;}
  addSum(g,'routed-shared',902,895,{info:modelInfo(m,'Routed + shared contribution','The routed mixture and full-width shared contribution are combined before the residual-state update.','Δx = routed + shared')});
  addWire(g,[[902,lastY],[902,883]],routedLast,'routed-shared');
  addWire(g,[[1212,sharedBottom],[1212,895],[914,895]],sharedLast,'routed-shared');
  addWire(g,[[902,907],[902,m.residual==='standard'?937:934]],'routed-shared','merge-ffn');
  addLabel(g,'routed + shared',932,882,'muted',{size:12});
 }else addWire(g,[[902,lastY],[902,937]],routedLast,'merge-ffn');
}
/** Exhaustive attention dispatch.
 *
 * Until 3.0.0 this was `else recurrent(g,m,s,kind)`, so every layer kind except three
 * silently rendered as gated recurrence -- no error, no warning, a confident wrong diagram.
 * An unknown kind is now a RangeError: a missing renderer is a defect to fix, never a
 * mechanism to borrow from another architecture.
 */
export function attentionRenderer(kind){
 if(kind==='mla')return mla;
 if(kind==='gqa')return gqa;
 if(kind==='qsa')return sparseAttention;
 if(kind==='kda'||kind==='delta')return (g,m,s)=>recurrent(g,m,s,kind);
 if(kind==='swa'||kind==='csa2-full'||kind==='csa2-reindex'||kind==='csa2-reuse')return (g,m,s)=>csa2Attention(g,m,s,kind);
 throw new RangeError(`No attention renderer for layer kind '${kind}'. Add a renderer for it; do not fall through to another mechanism.`);
}
export function buildModel(s){
 const m=getModel(s.model),kind=layerKind(m,s.layer),ff=ffnKind(m,s.layer),g=makeScene();
 stackOverview(g,m,s);
 addGroup(g,'block-region',206,16,1216,986,'','');
 addLabel(g,`LAYER ${String(s.layer).padStart(2,'0')}`,238,48,'compute',{size:12,tracking:1.8});
 addLabel(g,`${kind==='kda'?'KDA':kind==='delta'?'GATED DELTANET':kind==='qsa'?'QSA':kind.startsWith('csa2-')?`CSA2 · ${kind.slice(5).toUpperCase()}`:kind.toUpperCase()}  /  ${ff==='dense'?'DENSE FFN':ff==='latent-moe'?'LATENT MOE':'MOE'}`,386,48,'muted',{size:12,tracking:1});
 addGroup(g,'attention-region',258,174,498,704,'01  ATTENTION','',{headerInset:50});
 addGroup(g,'ffn-region',874,174,516,702,`02  ${ff==='dense'?'DENSE FFN':ff==='latent-moe'?'STABLE LATENT MOE':'MIXTURE OF EXPERTS'}`,'',{headerInset:50});
 residuals(g,m,s);
 attentionRenderer(kind)(g,m,s);
 feedForward(g,m,s);
 if(m.pleLayers.includes(s.layer)){
   // An explicit input-side modifier, not a fabricated standalone residual sum.
   addLabel(g,'PLE n-gram lookup at this layer → input mixer',466,81,'state',{size:11.5});
   g.nodes.find(n=>n.id==='norm1').info.body+=' This is a PLE layer: bigram/trigram lookup, contextual gating and a causal convolution augment its input. PLE tables and caches are distinct from attention state.';
 }
 g.defaultSelected='attention';g.caption='Follow attention first, then the feed-forward panel. The bottom return rail carries the attention-updated stream to the second normalization. Wires are logical dependencies, not a kernel timeline.';return g;
}
