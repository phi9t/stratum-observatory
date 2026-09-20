import {WORKED_RESULTS} from './worked-results.mjs';
import {makeScene,addNode,addGroup,addLabel,addWire,makeInfo} from './scene.mjs';

// These are operation graphs of the adjacent readable programs, not fused execution traces.
function memoryScene(example,height){
 const g=makeScene(1440,height);
 g.caption=`${example.summary} ${example.limitations} Publisher mapping is not production certification.`;
 addLabel(g,example.title,60,36,'muted',{size:14});
 addLabel(g,'FP32 reference arithmetic · symbolic batch / token axes · select an operation for scope',60,62,'muted',{size:12});
 g.defaultSelected=null;return g;
}
function op(g,e,id,col,y,name,shape,body,formula,facts=[],tone='compute'){
 const n=addNode(g,id,60+col*480,y,360,76,name,shape,{tone,info:makeInfo('Readable PyTorch schematic',name,body,formula,[['Tensor contract',shape],...facts],e.sources)});
 if(!g.defaultSelected)g.defaultSelected=id;return n;
}
function wire(g,a,b,points,options={}){addWire(g,points,a,b,options);}
function down(g,a,b,options={}){
 const s=g.nodes.find(n=>n.id===a),t=g.nodes.find(n=>n.id===b),sx=s.x+s.w/2,tx=t.x+t.w/2;
 const mid=(s.y+s.h+t.y)/2;
 wire(g,a,b,[[sx,s.y+s.h],[sx,mid],[tx,mid],[tx,t.y]],options);
}
function across(g,a,b,options={}){
 const s=g.nodes.find(n=>n.id===a),t=g.nodes.find(n=>n.id===b),right=t.x>s.x;
 wire(g,a,b,[[s.x+(right?s.w:0),s.y+s.h/2],[t.x+(right?0:t.w),t.y+t.h/2]],options);
}

function engram(m,e){
 const g=memoryScene(e,1460),cols=(m.engramNgram-1)*m.engramHeads;
 addGroup(g,'lookup-region',510,90,420,570,'HASH MEMORY','External global row IDs; hashing is outside this program');
 op(g,e,'residual',0,130,'Residual copies','x [B,T,S,D]', 'EngramMemory.forward / Engram.forward receive the residual copies. NgramHashState supplies global table IDs upstream; this example does not hash tokens.','S = m.streams; D = m.d',[['Checkpoint streams',String(m.streams)],['Checkpoint width',String(m.d)]],'state');
 op(g,e,'hash-ids',1,150,'Global hash row IDs','ids [B,T,C]', 'EngramMemory.forward receives IDs already laid out by NgramHashState. C is the number of lookup columns; row identities must not be interpreted as token IDs.','C = (max_ngram_size - 1) × n_heads',[['Checkpoint columns',String(cols)]],'neutral');
 op(g,e,'lookup',1,300,'Embedding lookup · flatten','[B,T,C,R] → [B,T,C×R]', 'EngramMemory.table adapts ParallelEngramEmbedding.forward with a plain embedding. Publisher row sharding and FP8 row dequantization are omitted.','embedded = table(ids).flatten(-2).float()',[['Checkpoint row width',String(m.engramHeadDim)]]);
 op(g,e,'project',1,450,'Project keys and shared value','[B,T,(S+1)×D]', 'EngramMemory.project corresponds to Engram.wkv. FP32 F.linear accepts FP32 or BF16 parameters in the example.','projected = embedded × W_projectᵀ');
 op(g,e,'keys',1,590,'Split · reshape keys','keys [B,T,S,D]', 'EngramMemory.forward splits the first S×D channels, then restores the stream axis as Engram.forward does.','keys = projected[..., :S×D].unflatten(-1,(S,D))');
 op(g,e,'value',2,590,'Split shared value','value [B,T,D]', 'EngramMemory.forward keeps a shared value, corresponding to Engram.forward value. The stream axis is introduced only when writing.','value = projected[..., S×D:]');
 op(g,e,'normalize',0,450,'Residual RMS statistics','inverse RMS [B,T,S]', 'EngramMemory.forward / Engram.forward normalize over D separately within each residual stream; epsilon comes from config.norm_epsilon.','r_x = rsqrt(mean_D(x²) + eps)');
 op(g,e,'score',1,760,'Per-stream normalized match','score [B,T,S]', 'EngramMemory.forward follows Engram.forward: learned query_scale and key_scale enter as a product; key RMS is independent of residual RMS.','score = sum_D(x × q_scale × k_scale × keys) × r_x × rsqrt(mean_D(keys²)+eps) / sqrt(D)');
 op(g,e,'signed-root',1,910,'Signed square root','[B,T,S]', 'EngramMemory.forward / Engram.forward preserve the score sign while clamping its absolute magnitude before square root.','signed_root = copysign(sqrt(max(abs(score),1e-6)),score)');
 op(g,e,'token-mask',0,1060,'Caller write permission','mask [B,T] · boolean', 'EngramMemory.forward requires a boolean mask; Engram.forward allows an optional token mask. False positions must preserve the residual unchanged.','mask = token_mask[...,None]',[],'neutral');
 op(g,e,'gate',1,1060,'Sigmoid · mask write gate','gate [B,T,S]', 'EngramMemory.forward / Engram.forward zero the gate after sigmoid for forbidden tokens.','gate = sigmoid(signed_root).masked_fill(~mask,0)');
 op(g,e,'write',1,1210,'Broadcast value · residual add','[B,T,S,D]', 'EngramMemory.forward / Engram.forward multiply the shared value by each stream gate, then add the untouched residual in FP32.','updated = x.float() + gate[...,None] × value[...,None,:]',[],'state');
 op(g,e,'output',1,1360,'Restore residual dtype','[B,T,S,D]', 'EngramMemory.forward casts the sum to the incoming residual dtype; this is not a checkpoint loading or quantized serving claim.','return updated.to(residual.dtype)',[],'compute');
 down(g,'hash-ids','lookup');down(g,'lookup','project');down(g,'project','keys');
 wire(g,'project','value',[[900,488],[1200,488],[1200,590]]);
 down(g,'residual','normalize');down(g,'keys','score');
 wire(g,'normalize','score',[[240,526],[240,798],[540,798]]);
 wire(g,'residual','score',[[420,168],[460,168],[460,730],[600,730],[600,760]]);
 down(g,'score','signed-root');down(g,'signed-root','gate');across(g,'token-mask','gate');down(g,'gate','write');down(g,'write','output');
 wire(g,'value','write',[[1200,666],[1200,1248],[900,1248]]);
 wire(g,'residual','write',[[60,168],[30,168],[30,1248],[540,1248]],{kind:'residual'});
 return g;
}

function compression(m,e){
 const g=memoryScene(e,1320);
 addGroup(g,'direct-region',40,240,400,470,'TokenCompressor · r = 1','Projection only; no gate and no pooled state');
 addGroup(g,'group-region',520,240,880,1030,'GroupedCompressor · r > 1','Caller owns FP32 tails; reset each new sequence');
 op(g,e,'tokens',1,110,'Input tokens','tokens [B,T,D]', 'TokenCompressor.forward and GroupedCompressor.forward adapt Compressor.forward with explicit caller-owned state.','D = m.d; K = m.headDim',[['Checkpoint D / K',`${m.d} / ${m.headDim}`],['Encoder / decoder ratios',`${m.compressEncoder} / ${m.compressDecoder}`]],'neutral');
 op(g,e,'direct-project',0,310,'Direct FP32 projection','[B,T,D] → [B,T,K]', 'TokenCompressor.forward adapts the ratio-one Compressor.wkv branch. The example promotes arithmetic to FP32; the publisher direct branch stays BF16.','projected = linear(tokens.float(),W_project.float())');
 op(g,e,'direct-norm',0,470,'CompressionNorm','[B,T,K]', 'CompressionNorm.forward adapts RMSNorm.forward. Statistics reduce only the final latent axis; epsilon comes from config.norm_epsilon.','norm(v) = v × rsqrt(mean_K(v²)+eps) × weight');
 op(g,e,'direct-output',0,620,'Per-token latents','[B,T,K] · input dtype', 'TokenCompressor.forward returns every token, before rotary encoding or quantization.','return norm(projected).to(tokens.dtype)',[],'compute');
 op(g,e,'values',1,310,'Project latent values','values [B,T,K]', 'GroupedCompressor.project corresponds to Compressor.wkv on the grouped branch.','values_new = linear(tokens.float(),W_project.float())');
 op(g,e,'scores',2,310,'Project pooling scores','scores [B,T,K]', 'GroupedCompressor.gate corresponds to Compressor.wgate; a separate score exists for each token and latent channel.','scores_new = linear(tokens.float(),W_gate.float())');
 op(g,e,'concat',1,510,'Prepend previous tails','values / scores [B,P+T,K]', 'GroupedCompressor.forward concatenates each previous tail with its new projection. This explicit state replaces Compressor.kv_state and score_state decode buffers.','values = cat(previous_values,values_new); scores = cat(previous_scores,scores_new)');
 op(g,e,'previous-state',2,510,'empty_state / prior tails','two [B,P,K] FP32 · P < r', 'GroupedCompressor.empty_state creates an empty tensor and clone on the token device. forward validates matching shapes, FP32 dtype and an unfinished group.','new sequence: P=0; continuation: state=(previous_values,previous_scores)',[],'state');
 op(g,e,'complete',1,710,'Split completed groups','[B,G,r,K] · G = floor((P+T)/r)', 'GroupedCompressor.forward pools only completed consecutive groups; unfinished tokens do not become cache entries.','complete = floor((P+T)/r)×r; grouped = values[:,:complete].unflatten(1,(G,r))');
 op(g,e,'next-state',2,710,'Clone unfinished tails','two [B,(P+T) mod r,K]', 'GroupedCompressor.forward clones values and scores after complete. This avoids retaining a large allocation; autograd history remains unless the caller detaches.','next_state = (values[:,complete:].clone(),scores[:,complete:].clone())',[],'state');
 op(g,e,'pool',1,910,'Channel-wise softmax pooling','pooled [B,G,K]', 'GroupedCompressor.forward / Compressor.forward normalize over the r token slots independently for each latent channel, then sum values over that same axis.','pooled = sum_r(grouped_values × softmax_r(grouped_scores))');
 op(g,e,'group-norm',1,1070,'CompressionNorm','normalized [B,G,K]', 'CompressionNorm.forward adapts RMSNorm.forward with FP32 statistics. The example normalizes FP32 pooled values before restoring the incoming dtype.','normalized = pooled × rsqrt(mean_K(pooled²)+eps) × weight');
 op(g,e,'group-output',1,1210,'Completed-group latents','[B,G,K] · input dtype', 'GroupedCompressor.forward returns an empty group axis when none completes, together with next_state. Publisher Compressor.forward instead returns None at an incomplete decode step.','return normalized.to(tokens.dtype), next_state',[],'compute');
 wire(g,'tokens','direct-project',[[540,148],[20,148],[20,348],[60,348]]);
 wire(g,'tokens','values',[[820,186],[820,210],[950,210],[950,348],[900,348]]);
 wire(g,'tokens','scores',[[900,148],[1200,148],[1200,310]]);
 down(g,'direct-project','direct-norm');down(g,'direct-norm','direct-output');down(g,'values','concat');
 wire(g,'scores','concat',[[1200,386],[1200,430],[820,430],[820,510]]);
 across(g,'previous-state','concat');down(g,'concat','complete');across(g,'complete','next-state');down(g,'complete','pool');down(g,'pool','group-norm');down(g,'group-norm','group-output');
 wire(g,'next-state','previous-state',[[1380,748],[1410,748],[1410,548],[1380,548]],{kind:'delay',delay:true});
 return g;
}

function selection(m,e){
 const g=memoryScene(e,1950);
 op(g,e,'query',0,110,'Projected / rotated queries','query [B,T,H,Dᵢ]', 'compute_index_scores isolates the contraction in Indexer.forward; query projection, rotary encoding and FP4 quantization are outside this example.','H = indexerHeads; Dᵢ = indexer',[['Checkpoint H / Dᵢ',`${m.indexerHeads} / ${m.indexer}`]],'neutral');
 op(g,e,'keys',1,110,'Shared index keys','keys [B,C,Dᵢ]', 'compute_index_scores reads already projected / rotated keys corresponding to Indexer.forward shared_attn.index_k.','C = number of published compressed positions',[],'state');
 op(g,e,'products',0,300,'Contract width · ReLU','products [B,T,H,C]', 'compute_index_scores / Indexer.forward rectify each head contraction before combining heads.','products = relu(einsum(bthd,bcd→bthc,query,keys))');
 op(g,e,'weights',1,300,'Learned head weights','weights [B,T,H]', 'compute_index_scores receives weights already projected by the caller, corresponding to Indexer.weights_proj.','weights = head_weights / sqrt(Dᵢ×H)');
 op(g,e,'scores',0,480,'Weighted head reduction','scores [B,T,C]', 'compute_index_scores / Indexer.forward sum weighted rectified products. Distributed reduction is omitted in this example.','scores = sum_H(products × weights[...,None])');
 op(g,e,'causal',1,660,'Mask unreachable positions','masked [B,T,C]', 'select_candidate_blocks masks any compressed position not completed at the current query, as Indexer.forward does before candidate selection.','masked = scores.masked_fill(arange(C) >= visible,-inf)');
 op(g,e,'visible',2,660,'Completed causal group count','visible [B,T,1]', 'select_candidate_blocks and select_sparse_indices receive visible from the caller; Indexer.forward computes it from absolute query position.','visible = floor((query_position+1)/ratio)',[],'neutral');
 op(g,e,'block-max',1,830,'Pad · maximum within block','block_scores [B,T,ceil(C/b)]', 'select_candidate_blocks pads with negative infinity and scores each block by its best compressed position.','block_scores = amax_b(pad(masked,-inf).unflatten(-1,(-1,b)))',[['Checkpoint block size',String(m.candidateBlockSize)]]);
 op(g,e,'newest-block',1,1000,'Pin newest reachable block','block_scores [B,T,ceil(C/b)]', 'select_candidate_blocks promotes the newest reachable block, even when older blocks have higher ordinary scores.','block_scores[block_id == floor((visible-1)/b)] = +inf');
 op(g,e,'block-topk',1,1170,'Select reachable blocks','keep [B,T,ceil(C/b)] · bool', 'select_candidate_blocks rejects negative-infinity top-k filler. The candidate budget is separate from the sparse position budget.','top = topk(block_scores,min(top_blocks,ceil(C/b))); keep.scatter(top.indices,top.values > -inf)',[['Checkpoint block budget',String(m.candidateBlocks)]]);
 op(g,e,'candidate-mask',1,1340,'Expand candidate mask','candidates [B,T,C] · bool', 'select_candidate_blocks repeats each selected block over its positions, trims padding and intersects causal visibility. Reindex layers receive the published mask.','candidates = repeat_interleave(keep,b)[...,:C] & causal',[],'state');
 op(g,e,'allowed',2,1510,'Candidate and causal intersection','allowed [B,T,C] · bool', 'select_sparse_indices adapts the candidate-mask branch of Indexer.forward. Scores are this indexer’s own; the candidate mask can come from an earlier producer.','allowed = candidates & (arange(C) < visible)');
 op(g,e,'position-topk',2,1680,'Top-k scores · position sort','indices [B,T,min(k,C)]', 'select_sparse_indices ranks allowed positions by score, then sorts the selected indices into position order. This is selection, not attention softmax.','indices = topk(scores.masked_fill(~allowed,-inf),min(k,C)).indices.sort(-1)',[['Checkpoint sparse budget',String(m.indexerBudget)]]);
 op(g,e,'selected',2,1850,'Invalid selections become −1','selected [B,T,min(k,C)]', 'select_sparse_indices checks gathered validity after top-k. Full/Reindex publish indices; Reuse layers consume them unchanged. Cache offsets belong to the attention gather.','selected = indices.masked_fill(~allowed.gather(-1,indices),-1)',[],'state');
 down(g,'query','products');wire(g,'keys','products',[[720,186],[720,240],[320,240],[320,300]]);down(g,'products','scores');
 wire(g,'weights','scores',[[720,376],[720,418],[360,418],[360,480]]);
 down(g,'scores','causal');across(g,'visible','causal');down(g,'causal','block-max');down(g,'block-max','newest-block');down(g,'newest-block','block-topk');down(g,'block-topk','candidate-mask');down(g,'candidate-mask','allowed');
 wire(g,'scores','allowed',[[240,556],[240,1548],[1020,1548]]);
 wire(g,'visible','allowed',[[1380,698],[1410,698],[1410,1480],[1200,1480],[1200,1510]]);
 wire(g,'visible','newest-block',[[1200,736],[1200,1038],[900,1038]]);
 wire(g,'allowed','selected',[[1380,1548],[1410,1548],[1410,1888],[1380,1888]]);
 down(g,'allowed','position-topk');down(g,'position-topk','selected');return g;
}

function attention(m,e){
 const experiment=WORKED_RESULTS.query,settings=experiment.settings,values=Object.fromEntries(experiment.inputs.map(row=>[row.id,row.values]));
 const queryShape=`[${settings.batch},${values.queries.length},${settings.heads},${settings.head_dim}]`;
 const g=memoryScene(e,1660);
 op(g,e,'local-cache',0,110,'Local shared KV history','local_kv [B,L,D]', 'sparse_attention adapts Attention.forward and sparse_attn_kernel with absolute history instead of a ring buffer. The same latent supplies key and value.','D = m.headDim',[['Checkpoint latent width',String(m.headDim)]],'state');
 op(g,e,'compressed-cache',1,110,'Compressed shared KV','compressed_kv [B,C,D]', 'sparse_attention receives published, already projected and rotated latents. Compression and cache publication are outside this program.','cache_compressed = compressed_kv.float()',[],'state');
 op(g,e,'cache',0,300,'Concatenate shared KV','cache [B,L+C,D]', 'sparse_attention concatenates local and compressed latents, corresponding to the shared cache passed by Attention.forward into sparse_attn_kernel.','cache = cat(local_kv.float(),compressed_kv.float(),dim=1)');
 op(g,e,'positions',1,300,'Absolute query positions','positions [T]', 'sparse_attention uses absolute positions for the local window; positions identify the current query in the full local history.','offsets = arange(window-1,-1,-1)',[['Checkpoint window',String(m.window)]],'neutral');
 op(g,e,'selected',2,300,'Published sparse selection','selected [B,T,K]', 'sparse_attention consumes already causal compressed indices or -1. Full/Reindex publish them; Reuse consumes them unchanged.','valid_compressed = (selected >= 0) & (selected < C)',[],'state');
 op(g,e,'local-indices',1,490,'Build causal local window','indices / valid [B,T,W]', 'sparse_attention subtracts window offsets from each query position. Out-of-history local positions are invalid.','local_indices = positions[:,None]-offsets; valid_local = (0 <= local_indices) & (local_indices < L)');
 op(g,e,'compressed-indices',2,490,'Offset compressed positions','indices / valid [B,T,K]', 'sparse_attention shifts compressed indices by local history length so they address the concatenated cache. The -1 sentinel remains invalid via its separate mask.','compressed_indices = selected + L');
 op(g,e,'indices',1,680,'Join indices and validity','[B,T,W+K]', 'sparse_attention joins local and compressed paths before gathering; no independent local/compressed attention normalization is introduced.','indices = cat(local_indices,compressed_indices); valid = cat(valid_local,valid_compressed)');
 op(g,e,'gather',0,870,'Batch-aware shared KV gather','gathered [B,T,W+K,D]', 'sparse_attention safely substitutes index zero for invalid lanes before batch-aware indexing. Invalid lanes are removed from logits, not treated as valid copies of row zero.','gathered = cache[batch_indices,indices.masked_fill(~valid,0)]');
 op(g,e,'query',1,870,'Attention queries','query [B,T,H,D]', 'sparse_attention isolates the sparse_attn_kernel core after query projection and rotary encoding; these preprocessing operations are not reimplemented here.','H = m.heads; D = m.headDim',[['Checkpoint heads / width',`${m.heads} / ${m.headDim}`]],'neutral');
 op(g,e,'logits',0,1060,'Scaled dot · invalid mask','logits [B,T,H,W+K]', 'sparse_attention / sparse_attn_kernel contract width and set invalid gathered positions to negative infinity before normalization.','logits = einsum(bthd,btkd→bthk,query,gathered)/sqrt(D); logits[~valid] = -inf');
 op(g,e,'sink',1,1230,'Learned sink logit','sink [H] · finite', 'sparse_attention / sparse_attn_kernel include a learned denominator-only sink. It has no value vector and never enters the value contraction.','sink_logits = broadcast(sink,[B,T,H,1])',[],'state');
 op(g,e,'softmax',0,1230,'Joint softmax · drop sink lane','probability [B,T,H,W+K]', 'sparse_attention normalizes local and compressed logits together with the sink. sparse_attn_kernel implements the equivalent denominator addition through online normalization.','probability = softmax(cat(logits,sink_logits),-1)[...,:-1]');
 op(g,e,'weighted-sum',0,1400,'Weighted shared-value sum','output [B,T,H,D]', 'sparse_attention / sparse_attn_kernel use the gathered latents as values. The sink contributes denominator mass only; all-invalid selections yield zero with a finite sink.','output = einsum(bthk,btkd→bthd,probability,gathered)');
 op(g,e,'output',0,1560,'Restore query dtype','[B,T,H,D]', 'sparse_attention returns query dtype. Inverse output RoPE, grouped output projections and quantization remain outside this readable program.','return output.to(query.dtype)',[],'compute');
 addGroup(g,'query-experiment',1000,990,400,560,'FIXED-DICTIONARY EXPERIMENT','Alternative queries at the same position');
 op(g,e,'fixed-dictionary',2,1060,'Hold selected KV fixed',`local [${settings.batch},${settings.local_positions},${settings.head_dim}] · compressed [${settings.batch},${settings.compressed_positions},${settings.head_dim}]`, 'run_query_example keeps local KV, selected compressed KV, positions and the illustrative sink logit unchanged. These are CPU fixture vectors, not checkpoint activations. The same dictionary supplies keys and values.',`M = ${JSON.stringify(values.dictionary)}; sink = ${values.sink[0]}`,[],'state');
 op(g,e,'alternative-queries',2,1230,'Read with alternative queries',`Q ${queryShape} · same position`, `run_query_example calls sparse_attention with Q₁=${JSON.stringify(values.queries[0])} and Q₂=${JSON.stringify(values.queries[1])} at the same absolute position. Its token axis represents counterfactual alternatives, not successive tokens or layers. Each query has its own joint normalization over the unchanged dictionary and zero-valued sink.`,'pᵢ = softmax([QᵢMᵀ/√D, sink])');
 op(g,e,'query-readouts',2,1400,'Different weights · readouts',`O ${queryShape} · unchanged KV`, 'The CPU example exposes the weights for inspection and returns outputs from the actual sparse_attention core. Real Attention layers still produce fresh local KV and queries and apply output projections; this experiment does not describe a query-only model.','Oᵢ = Σⱼ pᵢⱼ Mⱼ; sink value = 0',[['Normalization','Local + compressed + sink share the denominator']]);
 down(g,'fixed-dictionary','alternative-queries');down(g,'alternative-queries','query-readouts');
 down(g,'local-cache','cache');wire(g,'compressed-cache','cache',[[720,186],[720,240],[320,240],[320,300]]);
 down(g,'positions','local-indices');down(g,'selected','compressed-indices');down(g,'local-indices','indices');
 wire(g,'compressed-indices','indices',[[1200,566],[1200,718],[900,718]]);
 down(g,'cache','gather');wire(g,'indices','gather',[[720,756],[720,812],[320,812],[320,870]]);
 down(g,'gather','logits');wire(g,'query','logits',[[720,946],[720,998],[320,998],[320,1060]]);
 down(g,'logits','softmax');across(g,'sink','softmax');down(g,'softmax','weighted-sum');down(g,'weighted-sum','output');
 wire(g,'gather','weighted-sum',[[60,908],[30,908],[30,1438],[60,1438]]);
 wire(g,'indices','logits',[[900,742],[960,742],[960,1098],[420,1098]]);
 return g;
}

export const MEMORY_SCHEMATICS={'engram-write':engram,'csa2-compression':compression,'csa2-selection':selection,'csa2-attention':attention};
