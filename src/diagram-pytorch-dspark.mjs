import {makeScene,addNode,addGroup,addWire,makeInfo} from './scene.mjs';

// Named tensor operators, with independent input lanes and explicit state ownership.
// B=batch, K=draft positions, T=retained history, H=heads, W=head width,
// D=hidden width, R=Markov rank, V=vocabulary. These are caller dimensions.
function canvas(example,height,caption,offline=false){
 const g=makeScene(1440,height);g.caption=caption;g.defaultSelected=offline?'prefix':'queries';
 addGroup(g,'module',30,30,1380,height-60,example.title,offline?'DeepSpec offline evaluator · independent of V4.1 serving':'Source-linked PyTorch adaptation · inspect operators');
 g.moduleCategory=offline?'DeepSpec offline evaluator':'V4.1 draft module';return g;
}
function op(g,e,id,col,row,title,shape,body,formula,tone='compute'){
 return addNode(g,id,70+480*col,110+150*row,340,80,title,shape,{tone,info:makeInfo(g.moduleCategory,title,body,formula,[['Tensor contract',shape],['Scope',e.limitations]],e.sources)});
}
function wire(g,a,b,options={}){
 const s=g.nodes.find(n=>n.id===a),t=g.nodes.find(n=>n.id===b);
 const x=s.x+s.w/2,tx=t.x+t.w/2,y=s.y+s.h,ty=t.y;
 let points;
 if(s.y===t.y){const right=t.x>s.x;points=[[right?s.x+s.w:s.x,s.y+s.h/2],[right?t.x:t.x+t.w,t.y+t.h/2]];}
 else if(s.x===t.x&&ty>y)points=[[x,y],[tx,ty]];
 else{const lane=options.lane??(ty>y?y+35:s.y-30);points=[[x,y],[x,lane],[tx,lane],[tx,ty]];}
 const{lane,...rest}=options;addWire(g,points,a,b,rest);
}
function attention(m,e){
 const g=canvas(e,1390,'V4.1 draft contraction: captured target history and every current draft slot are visible; the learned sink carries probability mass only.');
 op(g,e,'queries',0,0,'Projected queries','Q [B,K,H,W]','DSparkAttention.forward supplies queries after projection, query normalization and RoPE. This helper begins at that boundary.','Q = rotated_projected_queries');
 op(g,e,'history',1,0,'Captured target history','history [B,T,W]','DSparkAttention.forward captures target features as shared key/value vectors. Only valid captured slots enter this helper; ring writes are outside its scope.','history = valid_captured_KV','state');
 op(g,e,'draft-kv',2,0,'Current draft block','draft [B,K,W]','DSparkAttention.forward projects and rotates the current block. Each latent is both its key and value.','draft = projected_rotated_KV');
 op(g,e,'guards',0,1,'Validate input axes','B, K, H, W; configured H, W','dspark_projected_attention checks tensor ranks, matching batch and width axes, and the configured attention head count and width.','shape(Q)=[B,K,H,W]');
 op(g,e,'key-values',1,1,'Concatenate shared K/V','KV [B,T+K,W] · FP32','dspark_projected_attention concatenates history_key_values and draft_key_values along the position axis, then casts to FP32 with autocast disabled.','KV = cat(history, draft; position)');
 op(g,e,'visibility',2,2,'Block visibility','all retained history + all draft slots','get_dspark_topk_idxs broadcasts the same valid window and every draft slot to each query. No triangular mask: the first draft query sees the final draft slot. The helper has already removed invalid ring-buffer slots.','visible(k) = history ∪ draft');
 op(g,e,'dot-products',0,2,'Query × shared keys','scores [B,K,H,T+K]','sparse_attn and sparse_attn_kernel compute the contraction over width. The dense helper uses einsum with FP32 accumulation.','scores[b,k,h,n] = Σw Q[b,k,h,w] KV[b,n,w]');
 op(g,e,'scale',0,3,'Scale attention scores','scores [B,K,H,T+K]','dspark_projected_attention scales key scores by the inverse square root of config.head_dim.','scores ← scores / √head_dim');
 op(g,e,'sink',2,3,'Learned sink per head','sink [H] → [B,K,H,1]','sparse_attn_kernel includes attn_sink in the normalization. Broadcasting adds one score per head, with no associated value vector.','sink_scores = broadcast(sink)','state');
 op(g,e,'normalize',1,4,'Joint softmax with sink','probabilities [B,K,H,T+K+1]','dspark_projected_attention concatenates scores and sink_scores before softmax. Dropping the sink before normalization would change every attention weight.','P = softmax(cat(scores, sink_scores))');
 op(g,e,'drop-sink',1,5,'Exclude sink from value read','P_read [B,K,H,T+K]','sparse_attn_kernel leaves sink mass in the denominator while assigning it no value. The helper removes only the final probability column.','P_read = P[..., :-1]');
 op(g,e,'weighted-values',1,6,'Weighted shared-value sum','output [B,K,H,W] · FP32','dspark_projected_attention contracts the retained key-position axis with the same key_values tensor used for scoring.','O[b,k,h,w] = Σn P_read[b,k,h,n] KV[b,n,w]');
 op(g,e,'output',1,7,'Restore query dtype','output [B,K,H,W]','dspark_projected_attention returns output.to(queries.dtype). DSparkAttention.forward additionally performs inverse output RoPE and grouped output projections outside this isolated helper.','return O.to(Q.dtype)');
 wire(g,'queries','guards');wire(g,'guards','dot-products');wire(g,'history','key-values');wire(g,'draft-kv','key-values');wire(g,'key-values','dot-products');wire(g,'visibility','dot-products');wire(g,'dot-products','scale');wire(g,'scale','normalize');wire(g,'sink','normalize');wire(g,'normalize','drop-sink');wire(g,'drop-sink','weighted-values');
 // Shared values travel in a reserved rail, bypassing score normalization.
 addWire(g,[[890,300],[950,300],[950,1050],[890,1050]],'key-values','weighted-values');wire(g,'weighted-values','output');return g;
}
function heads(m,e){
 const g=canvas(e,1720,'V4.1 output heads: parallel backbone features feed sequential token sampling; confidence reads the unnormalized residual and preceding-token embeddings.');g.defaultSelected='previous-token';
 op(g,e,'hidden',0,0,'Collapsed residual input','hidden [B,K,D] · before final norm','DSparkBlock.forward_head collapses streams with hc_pre. DSparkOutputHeads receives this unnormalized residual; it must not reuse the normalized vocabulary input for confidence.','hidden = hc_pre(x, pre_mix)');
 op(g,e,'base-logits',1,0,'Base vocabulary logits','base_logits [B,K,V]','DSparkBlock.forward_head applies final RMSNorm and the shared vocabulary head. DSparkOutputHeads takes these precomputed logits as a separate input.','base_logits = head(norm(hidden))');
 op(g,e,'seed',2,0,'Preceding seed token','seed [B] · int64','DSparkOutputHeads validates the seed vocabulary range. The seed is preserved as the first returned token, preceding all K samples.','token_ids = [seed]','state');
 op(g,e,'guards',0,1,'Validate heads inputs','D,R,V > 0; K ≥ 1; temperature > 0','DSparkOutputHeads validates constructor dimensions, hidden/base-logit shapes and a finite positive temperature before sampling.','shape(base_logits)=[B,K,V]');
 op(g,e,'previous-token',2,1,'Position i: previous token','token_ids[-1] [B]','DSparkBlock.forward_head samples in position order. Position zero reads the seed; each later position reads the preceding sample.','previous = seed if i=0 else token[i−1]','state');
 op(g,e,'previous-embedding',2,2,'Markov token embedding','previous_embedding [B,R]','DSparkMarkovHead.embed looks up the preceding token. The same embedding branches to vocabulary correction and later confidence.','E_prev = markov_embed(previous)');
 op(g,e,'markov-bias',1,2,'Low-rank vocabulary bias','markov_bias [B,V] · FP32','DSparkMarkovHead.head projects preceding-token features to vocabulary logits. DSparkOutputHeads disables autocast and explicitly casts the embedding and weight to float.','bias = linear(E_prev.float(), W_markov.float())');
 op(g,e,'corrected',1,3,'Add Markov correction','position_logits [B,V] · FP32','DSparkBlock.forward_head adds the position-specific Markov bias to the base logits. Corrected logits are retained before temperature scaling.','L_i = base_logits[:,i].float() + bias');
 op(g,e,'probabilities',1,4,'Temperature and softmax','probabilities [B,V]','DSparkOutputHeads uses a finite positive temperature and a vocabulary-axis softmax. This is the distribution used to sample, not the confidence output.','p_i = softmax(L_i / temperature)');
 op(g,e,'sample',2,4,'Sample current token','sample [B] · multinomial','DSparkBlock.forward_head advances the previous-token dependency after each sample. The adaptation uses torch.multinomial instead of the publisher sampler.','token[i] ∼ Categorical(p_i)');
 op(g,e,'embedding-stack',2,5,'Retain preceding embeddings','markov_embeddings [B,K,R]','DSparkOutputHeads stacks previous_embeddings in draft-position order after sampling. These embeddings refer to preceding tokens, not the token sampled at the same position.','E = stack(previous_embeddings, dim=1)');
 op(g,e,'confidence-features',0,6,'Concatenate confidence features','confidence_input [B,K,D+R]','DSparkConfidenceHead concatenates raw collapsed residuals with Markov embeddings across features, not positions. Both branches cast to FP32.','C = cat(hidden.float(), E.float(); features)');
 op(g,e,'raw-confidence',0,7,'FP32 confidence projection','raw_confidence [B,K]','DSparkConfidenceHead.proj emits raw logits. No sigmoid or threshold is applied by DSparkOutputHeads; that policy belongs to a separate consumer.','raw = linear(C, W_confidence.float()).squeeze(-1)');
 op(g,e,'logits-stack',1,7,'Stack corrected logits','corrected_logits [B,K,V]','DSparkOutputHeads returns logits before temperature scaling; these are not the normalized sampling probabilities.','logits = stack(corrected_logits, dim=1)');
 op(g,e,'tokens-stack',2,7,'Seed and sampled tokens','tokens [B,K+1]','DSparkOutputHeads preserves the seed and appends each sampled token in order.','tokens = stack(token_ids, dim=1)');
 op(g,e,'return',1,9,'Return head outputs','tokens, corrected logits, raw confidence','DSparkBlock.forward_head returns the ordered token block, Markov-corrected vocabulary logits and raw confidence. These are reference computations, not a production scheduling policy.','return tokens, logits, raw');
 wire(g,'hidden','guards');wire(g,'seed','previous-token');wire(g,'previous-token','previous-embedding');wire(g,'previous-embedding','markov-bias');wire(g,'markov-bias','corrected');
 addWire(g,[[720,190],[470,190],[470,600],[550,600]],'base-logits','corrected');wire(g,'corrected','probabilities');wire(g,'probabilities','sample');
 addWire(g,[[1370,750],[1390,750],[1390,300],[1370,300]],'sample','previous-token',{kind:'delay',delay:true});
 addWire(g,[[1370,450],[1380,450],[1380,900],[1370,900]],'previous-embedding','embedding-stack');
 addWire(g,[[70,150],[50,150],[50,1050],[70,1050]],'hidden','confidence-features');wire(g,'embedding-stack','confidence-features');wire(g,'confidence-features','raw-confidence');
 addWire(g,[[890,600],[940,600],[940,1200],[890,1200]],'corrected','logits-stack');addWire(g,[[1370,770],[1400,770],[1400,1200],[1370,1200]],'sample','tokens-stack');wire(g,'raw-confidence','return');wire(g,'logits-stack','return');wire(g,'tokens-stack','return');return g;
}
function verifier(m,e){
 const g=canvas(e,2390,'DeepSpec offline Qwen3 / Gemma4 evaluator, separate from V4.1 serving runtime. Nonterminal round: no EOS or output-budget truncation.',true);g.defaultSelected='prefix';
 op(g,e,'confidence',0,0,'Draft raw confidence','confidence [K_full]','_confident_prefix_length consumes confidence logits. The adaptation deepspec_confident_prefix supports a finite positive fixed threshold, not the production dynamic scheduler.','0 < threshold ≤ 1','neutral');
 op(g,e,'confidence-prefix',0,1,'Contiguous confident prefix','proposal count K ≤ K_full','deepspec_confident_prefix checks finite vector input, applies sigmoid, then cumprod so a low-confidence position excludes all later positions. Upstream _confident_prefix_length also supports a disabled-threshold mode omitted here.','K = sum(cumprod(sigmoid(confidence) ≥ threshold))');
 op(g,e,'draft-forward',2,0,'Draft forward boundary','draft cache [cached_positions,…]','forward_dspark_draft_block executes the offline draft backbone with is_causal=False, then immediately crops its DynamicCache to the old start. The helper shows the crop result using dense token-axis-first tensors.','draft forward → crop(start)','compute');
 op(g,e,'draft-crop',2,1,'Release draft speculative tail','retained draft [start,…]','forward_dspark_draft_block crops immediately after draft forward, independently of target acceptance. deepspec_reuse_state clones the slice to release speculative backing storage.','draft = draft_cache[:start].clone()','state');
 op(g,e,'proposal',0,2,'Proposal token vector','proposed [K] · int64','verify_draft_tokens verifies a single sequence. Empty proposals are allowed. The helper validates proposal shape, integer dtype and vocabulary bounds.','proposed = draft_tokens[:K]');
 op(g,e,'draft-probs',1,2,'Normalized draft probabilities','draft [K,V]','_deepspec_check_distribution requires finite, nonnegative rows of unit mass and a nonempty vocabulary. Input probabilities already include the chosen temperature.','∀i: Σv draft[i,v] = 1');
 op(g,e,'target-probs',2,2,'Target probabilities plus bonus','target [K+1,V]','verify_draft_tokens obtains a target forward for seed plus proposals. The helper starts at the normalized probabilities; row i predicts proposal i and the final row supplies a bonus. _deepspec_check_distribution checks each row.','∀i: Σv target[i,v] = 1');
 op(g,e,'gather',1,3,'Gather proposed-token mass','p_selected, q_selected [K]','verify_draft_tokens gathers paired row and token indices. deepspec_verify_probabilities clamps selected draft mass away from zero.','p_i=target[i,proposed[i]]; q_i=max(draft[i,proposed[i]],ε)');
 op(g,e,'uniforms',2,4,'Uniform acceptance draws','uniforms [K] · finite in [0,1)','deepspec_verify_probabilities rejects invalid draws or a vector length that differs from K. Draws are independent input data, not confidence thresholds.','u_i ∈ [0,1)','neutral');
 op(g,e,'acceptance',1,4,'Per-position acceptance','accept mask [K]','verify_draft_tokens compares uniform draws with the clipped target/draft probability ratio. Acceptance is distinct from proposal confidence.','accept_i = u_i < min(1, p_i/q_i)');
 op(g,e,'prefix',1,5,'Stop at first rejection','prefix_mask [K]','verify_draft_tokens uses cumulative product: after the first rejection, every later position is excluded even if its individual acceptance test passes.','prefix_mask = cumprod(accept_mask)');
 op(g,e,'accepted',2,5,'Count accepted prefix','accepted = A · scalar','deepspec_verify_probabilities sums the contiguous prefix mask. With K=0 the sum is zero and the bonus branch remains valid.','A = sum(prefix_mask)');
 op(g,e,'branch',1,6,'Any rejected proposal?','A < K → correction; A = K → bonus','verify_draft_tokens selects residual correction only when a proposal was rejected; full acceptance and empty proposals select the final target row.','branch = (accepted < proposal_count)');
 op(g,e,'residual',0,7,'Rejection: positive residual','residual [V]','sample_residual subtracts draft mass from target mass at the first rejected position and clamps negative entries to zero. The helper validates matching vector shapes and distributions.','r = max(target[A] − draft[A], 0)');
 op(g,e,'bonus',2,7,'All accepted: target bonus','bonus probabilities [V]','verify_draft_tokens uses the final target row when every proposal passed. No draft subtraction occurs on this branch, including K=0.','next_probabilities = target[-1]');
 op(g,e,'fallback',0,8,'Vanishing residual fallback','residual [V] · ε = 1e−8','sample_residual uses target probabilities if residual mass is at most epsilon. This is a real branch, not a zero distribution.','r ← target[A] if sum(r) ≤ ε else r');
 op(g,e,'residual-normalize',0,9,'Normalize correction mass','next_probabilities [V]','sample_residual renormalizes the positive residual or its target fallback before categorical sampling.','next_probabilities = r / max(sum(r), ε)');
 op(g,e,'next-token',1,10,'Sample correction or bonus','next_token [1]','deepspec_verify_probabilities samples exactly one token from the selected distribution. This new token has not yet entered the target cache.','next_token ∼ Categorical(next_probabilities)');
 op(g,e,'commit',1,11,'Commit accepted prefix + token','committed [A+1]','verify_draft_tokens concatenates only the accepted proposed prefix with the correction or bonus token. Rejected speculative tokens are discarded.','committed = cat(proposed[:A], next_token)');
 op(g,e,'state-guards',0,13,'Validate retained state lengths','start,A ≥ 0 · token axis first','deepspec_reuse_state requires integer nonnegative boundaries, leading token axes and sufficient target/draft/feature storage. It never silently truncates unavailable retained state.','target_length = start + accepted + 1');
 op(g,e,'target-crop',2,12,'Retain target seed + prefix','target [start+A+1,…]','generate_decoding_sample crops the target cache to the old start plus the seed and accepted proposals. The new correction/bonus token is excluded until the next target forward.','target = target_cache[:start + accepted + 1].clone()','state');
 op(g,e,'feature-crop',0,12,'Retain verified target features','features [A+1,…]','Qwen3DSparkEvaluator._update retains target features for the seed and accepted prefix. deepspec_reuse_state clones the matching dense tensor slice; rejected proposal features are discarded.','features = verified_features[:accepted + 1].clone()','state');
 op(g,e,'next-round',1,14,'Next nonterminal round','committed tokens + retained state','generate_decoding_sample continues with the correction/bonus as the next seed, the cropped target cache, old-start draft cache and accepted target features. Stop tokens and output budgets are outside this schematic.','next_seed = committed[-1]','state');
 wire(g,'confidence','confidence-prefix');wire(g,'confidence-prefix','proposal');wire(g,'draft-forward','draft-crop');wire(g,'proposal','gather');wire(g,'draft-probs','gather');wire(g,'target-probs','gather');wire(g,'gather','acceptance');wire(g,'uniforms','acceptance');wire(g,'acceptance','prefix');wire(g,'prefix','accepted');wire(g,'prefix','branch');wire(g,'branch','residual');wire(g,'branch','bonus');wire(g,'residual','fallback');wire(g,'fallback','residual-normalize');wire(g,'residual-normalize','next-token');
 addWire(g,[[1370,1200],[1390,1200],[1390,1575],[720,1575],[720,1610]],'bonus','next-token');wire(g,'next-token','commit');
 addWire(g,[[1370,900],[1400,900],[1400,1950],[1370,1950]],'accepted','target-crop');
 addWire(g,[[1370,900],[1400,900],[1400,2340],[50,2340],[50,1950],[70,1950]],'accepted','feature-crop');
 addWire(g,[[550,450],[450,450],[450,1200],[410,1200]],'draft-probs','residual');
 addWire(g,[[1200,490],[1200,525],[1010,525],[1010,1140],[920,1140],[920,1200],[410,1200]],'target-probs','residual');
 addWire(g,[[1370,900],[1400,900],[1400,1800],[890,1800]],'accepted','commit');
 addWire(g,[[70,450],[60,450],[60,1740],[680,1740],[680,1760]],'proposal','commit');
 addWire(g,[[1370,300],[1430,300],[1430,2250],[890,2250]],'draft-crop','next-round');
 addWire(g,[[240,2140],[240,2360],[1420,2360],[1420,1950],[1370,1950]],'state-guards','target-crop');addWire(g,[[100,2060],[100,1990]],'state-guards','feature-crop');wire(g,'feature-crop','next-round');wire(g,'target-crop','next-round');wire(g,'commit','next-round');
 // Reserve a separate input row for acceptance randomness. Gathered probabilities
 // use the left gutter; neither input crosses the retained probability rails.
 g.height+=120;g.groups[0].h+=120;
 for(const n of g.nodes)if(n.y>=710)n.y+=120;
 for(const edge of g.edges)for(const point of edge.points)if(point[1]>=710)point[1]+=120;
 const uniforms=g.nodes.find(n=>n.id==='uniforms');uniforms.x=550;uniforms.y=710;
 g.edges.find(w=>w.source==='uniforms'&&w.target==='acceptance').points=[[720,790],[720,830]];
 g.edges.find(w=>w.source==='gather'&&w.target==='acceptance').points=[[550,600],[500,600],[500,870],[550,870]];
 return g;
}
export const DSPARK_SCHEMATICS = {'dspark-attention':attention,'dspark-heads':heads,'deepspec-verifier':verifier};
