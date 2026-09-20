import {makeScene,addNode,addGroup,addLabel,addWire,makeInfo} from './scene.mjs';

// Each route is authored in a reserved gap; helper functions only attach scene metadata.
function operation(g,e,id,x,y,name,shape,body,formula,facts=[],tone='compute'){
 return addNode(g,id,x,y,360,68,name,shape,{tone,info:makeInfo('PyTorch reference implementation',name,body,formula,[['Tensor shape',shape],...facts],e.sources)});
}
function wire(g,a,b,points,kind='flow'){addWire(g,points,a,b,{kind});}
function heading(g,e,title){
 addLabel(g,title,60,38,'muted',{size:16});
 addLabel(g,'REFERENCE IMPLEMENTATION · select an operation for shapes, equations and sources',60,68,'muted',{size:11});
 g.caption=`${e.summary} ${e.limitations}`;
}
function mhc(m,e){
 const g=makeScene(1440,1970);
 heading(g,e,'Single-pass mHC · residual ownership through the flat stack');
 operation(g,e,'embedded',60,110,'Caller embeddings','[batch, tokens, hidden_dim]','The caller supplies embeddings. The reference does not load weights or embed token IDs.','embedded ∈ ℝ^[batch,tokens,hidden_dim]',[['hidden_dim',String(m.d)]]);
 operation(g,e,'initial-premix',540,110,'One-hot initial premix','[batch, tokens, streams]','The first sublayer reads the first residual copy. Later reads use the previous sublayer’s predicted weights.','p[...,0] = 1; p[...,1:] = 0',[['streams',String(m.streams)]],'state');
 operation(g,e,'expand-streams',1020,110,'Duplicate residual streams','[batch, tokens, streams, hidden_dim]','Insert the residual-copy axis; streams are not attention heads.','X = embedded.unsqueeze(-2).repeat(1,1,streams,1)',[['streams',String(m.streams)],['hidden_dim',String(m.d)]],'state');
 wire(g,'embedded','expand-streams',[[240,178],[240,202],[1080,202],[1080,178]]);
 const sublayer=(prefix,y,label)=>{
  addGroup(g,`${prefix}-region`,40,y-36,1360,624,label+' · TransformerBlock', '',{headerInset:20});
  operation(g,e,`${prefix}-residual`,1020,y,'Retained residual X','[batch, tokens, streams, hidden_dim]','Keep this sublayer’s incoming residual for the carried branch and the input collapse. The mixer reads this same tensor.','X = residual',[['Ownership','current sublayer input']],'state');
  operation(g,e,`${prefix}-mixer`,540,y,'Flatten · project · RMS scale','[batch, tokens, streams × hidden_dim]','HyperConnectionMixer uses a single RMS statistic across all residual copies and hidden channels. The projection and statistic accumulate in float32 with autocast disabled.','z = flatten(X); c = linear(z,W) / √(mean(z²)+ε_norm)',[['Projection output','[batch, tokens, streams × (streams + 2)]'],['Coefficient count',String(m.streams*(m.streams+2))]]);
  operation(g,e,`${prefix}-collapse`,60,y+120,'Collapse residual streams','[batch, tokens, hidden_dim]','collapse_residual_streams broadcasts the previous sublayer’s input weights across hidden channels and sums only the streams axis. Return to the residual dtype.','h[b,t,d] = Σᵢ p_prev[b,t,i] X[b,t,i,d]',[['Read ownership',prefix==='attn'?'previous block FFN; initially one-hot':'current block attention premix']]);
  operation(g,e,`${prefix}-split`,540,y+120,'Split mixing coefficients','[batch, tokens, streams × (streams + 2)]','Apply learned scales and offsets to the input-vector, output-vector and residual-matrix slices. Only the matrix uses Sinkhorn.','c → (c[:R], c[R:2R], reshape(c[2R:],[R,R]))',[['R','streams'],['Scales','[coefficient_groups=3]'],['Offsets','[streams × (streams + 2)]']]);
  operation(g,e,`${prefix}-next`,1020,y+120,'Next sublayer input weights','[batch, tokens, streams]','Predict the input weights consumed by the following sublayer. These coefficients do not collapse the current sublayer input.','p_next = sigmoid(c_input × scale_input + offset_input) + ε_mix',[['Consumer',prefix==='attn'?'FFN collapse':'next block attention or final collapse']],'state');
  operation(g,e,`${prefix}-norm`,60,y+240,'RMSNorm · learned scale','[batch, tokens, hidden_dim]','RMSNorm accumulates variance and multiplies the learned hidden-channel scale in float32, then casts back to the input dtype.','h_norm = (h.float() × rsqrt(mean(h.float()²)+ε_norm) × weight.float()).to(h.dtype)',[['Normalized axis','hidden_dim']]);
  operation(g,e,`${prefix}-post`,540,y+240,'Current output weights','[batch, tokens, streams]','Each output residual copy receives the same sublayer result multiplied by its own current output weight.','post = 2 × sigmoid(c_output × scale_output + offset_output)',[['Consumer','current sublayer expansion']]);
  operation(g,e,`${prefix}-matrix`,1020,y+240,'Residual matrix · Sinkhorn','[batch, tokens, input_streams, output_streams]','Start with row softmax plus epsilon, normalize columns, then alternate row and column normalization. Epsilon-stabilized finite iterations do not assert exact doubly stochastic sums.','C = softmax(A, -1)+ε_mix; C /= sum(C,-2)+ε_mix; repeat row/column normalization',[['Checkpoint iterations',String(m.sinkhornIters)],['Reference iterations','config.sinkhorn_iterations']],'state');
  operation(g,e,`${prefix}-operator`,60,y+360,label,'[batch, tokens, hidden_dim]','Injected torch module. Attention must be causal and owns its cache state; the residual orchestrator itself does not implement cache retention or the operator internals.','Δ = sublayer(h_norm)',[['Module',prefix==='attn'?'self.attention':'self.feed_forward']]);
  operation(g,e,`${prefix}-expanded`,540,y+360,'Expand sublayer result','[batch, tokens, streams, hidden_dim]','Broadcast the same output along the residual-copy axis, using the current output weights. Accumulate in float32.','expanded[b,t,j,d] = post[b,t,j] × Δ[b,t,d]',[['Owner','current sublayer']]);
  operation(g,e,`${prefix}-carried`,1020,y+360,'Mix retained residual','[batch, tokens, output_streams, hidden_dim]','The first matrix axis is the input copy i; sum i and retain output copy j. Do not transpose input/output ownership. Accumulate with autocast disabled.','carried = einsum("btij,btid->btjd", C.float(), X.float())',[['Summed axis','input_streams'],['Preserved axes','batch, tokens, output_streams, hidden_dim']],'state');
  operation(g,e,`${prefix}-combine`,60,y+500,'Combine residual streams','[batch, tokens, streams, hidden_dim]','combine_residual_streams adds the expanded sublayer update to the mixed retained input and casts to the sublayer output dtype.','X_next = (expanded + carried).to(Δ.dtype)',[['Result','residual for the next sublayer']],'state');
  wire(g,`${prefix}-residual`,`${prefix}-mixer`,[[1020,y+34],[900,y+34]]);
  wire(g,`${prefix}-residual`,`${prefix}-collapse`,[[1200,y+68],[1200,y+92],[360,y+92],[360,y+120]],'residual');
  wire(g,`${prefix}-mixer`,`${prefix}-split`,[[720,y+68],[720,y+120]]);
  wire(g,`${prefix}-split`,`${prefix}-next`,[[900,y+154],[1020,y+154]],'state');
  wire(g,`${prefix}-split`,`${prefix}-post`,[[720,y+188],[720,y+240]]);
  wire(g,`${prefix}-split`,`${prefix}-matrix`,[[840,y+188],[840,y+216],[1200,y+216],[1200,y+240]],'state');
  wire(g,`${prefix}-collapse`,`${prefix}-norm`,[[240,y+188],[240,y+240]]);
  wire(g,`${prefix}-norm`,`${prefix}-operator`,[[240,y+308],[240,y+360]]);
  wire(g,`${prefix}-operator`,`${prefix}-expanded`,[[420,y+394],[540,y+394]]);
  wire(g,`${prefix}-post`,`${prefix}-expanded`,[[720,y+308],[720,y+360]]);
  wire(g,`${prefix}-matrix`,`${prefix}-carried`,[[1200,y+308],[1200,y+360]],'state');
  wire(g,`${prefix}-residual`,`${prefix}-carried`,[[1380,y+34],[1400,y+34],[1400,y+394],[1380,y+394]],'residual');
  wire(g,`${prefix}-expanded`,`${prefix}-combine`,[[720,y+428],[720,y+468],[360,y+468],[360,y+500]]);
  wire(g,`${prefix}-carried`,`${prefix}-combine`,[[1200,y+428],[1200,y+486],[400,y+486],[400,y+500]],'residual');
 };
 sublayer('attn',260,'Injected causal attention');
 sublayer('ffn',920,'Injected feed-forward');
 wire(g,'expand-streams','attn-residual',[[1200,178],[1200,260]],'residual');
 wire(g,'initial-premix','attn-collapse',[[600,110],[600,90],[30,90],[30,414],[60,414]],'state');
 wire(g,'attn-combine','ffn-residual',[[240,828],[240,870],[1200,870],[1200,920]],'residual');
 wire(g,'attn-next','ffn-collapse',[[1380,414],[1420,414],[1420,1014],[400,1014],[400,1040]],'state');
 operation(g,e,'flat-stack',60,1570,'Continue the flat block stack','residual [batch, tokens, streams, hidden_dim]','CausalEncoderDecoder passes both residual and premix through every supplied block in order. The encoder boundary is an observation point in this same stack.','(X,p) ← block[index](X,p)',[['Checkpoint encoder layers',String(m.encoderLayers)],['Checkpoint decoder layers',String(m.layers-m.encoderLayers)]],'state');
 operation(g,e,'boundary',540,1570,'Clone at encoder boundary','[batch, tokens, streams, hidden_dim]','The returned boundary is a clone for observation. No inserted cross-attention consumes it; injected attention owns global-cache behavior.','if index + 1 == encoder_layers: boundary = residual.clone()',[['Owner','returned observation; no alias with later mutation']],'state');
 operation(g,e,'final-collapse',60,1730,'Final residual collapse','[batch, tokens, hidden_dim]','Read the last block’s FFN-predicted input weights. The displayed expanded block is the repeated unit; the final path uses its last occurrence.','h_final = Σᵢ p_last_FFN[...,i] × X_last[...,i,:]',[['Read ownership','last FFN premix']]);
 operation(g,e,'output-norm',540,1730,'Final RMSNorm','[batch, tokens, hidden_dim]','Normalize over hidden channels and return this representation together with the observed encoder boundary. A vocabulary head is outside the example.','normalized = RMSNorm(h_final)',[['Output dtype','embedding / injected operator dtype']]);
 wire(g,'ffn-combine','flat-stack',[[240,1488],[240,1570]],'residual');
 wire(g,'ffn-next','flat-stack',[[1380,1074],[1420,1074],[1420,1530],[360,1530],[360,1570]],'state');
 wire(g,'flat-stack','boundary',[[420,1604],[540,1604]],'state');
 wire(g,'flat-stack','final-collapse',[[240,1638],[240,1730]],'residual');
 wire(g,'ffn-next','final-collapse',[[1380,1074],[1420,1074],[1420,1690],[360,1690],[360,1730]],'state');
 wire(g,'final-collapse','output-norm',[[420,1764],[540,1764]]);
 operation(g,e,'return',1020,1840,'Return output and boundary','normalized [batch, tokens, hidden_dim]','Return the normalized final representation and the independently retained encoder boundary as a tuple. Both dependencies remain selectable in the trace.','return normalized, boundary',[['Boundary shape','[batch, tokens, streams, hidden_dim]']],'state');
 wire(g,'output-norm','return',[[900,1764],[960,1764],[960,1874],[1020,1874]]);
 wire(g,'boundary','return',[[900,1604],[980,1604],[980,1810],[1200,1810],[1200,1840]],'state');
 addLabel(g,'Depth proceeds through encoder then decoder; the token axis is preserved throughout.',60,1860,'muted',{size:12});
 g.defaultSelected='attn-collapse';return g;
}

function moe(m,e){
 const g=makeScene(1440,2140);
 addGroup(g,'routing-region',40,84,1360,620,'Router · selection and weights','',{headerInset:20});
 addGroup(g,'routed-region',40,924,800,488,'Routed SwiGLU','',{headerInset:420});
 addGroup(g,'shared-region',600,1304,800,606,'Shared SwiGLU · every token, unit weight','',{headerInset:20});
 heading(g,e,'Mixture of experts · selection, weighted routing and independent shared path');
 operation(g,e,'tokens',60,120,'Flatten token axes','[tokens, hidden_dim]','Merge leading input axes only for dispatch; remember the original shape and dtype for the final result.','tokens = inputs.reshape(-1, inputs.shape[-1])',[['hidden_dim',String(m.d)],['Original shape','[..., hidden_dim]']]);
 operation(g,e,'scores',540,120,'Router · sqrt-softplus','[tokens, experts]','Float32 router projection and scores with autocast disabled. Temperature comes from config.router_temperature.','logits = linear(tokens.float(),W.float()) / temperature; scores = √softplus(logits)',[['experts',String(m.experts)],['Routing scope','text-only reference']]);
 operation(g,e,'selection-bias',1020,120,'Correction bias','[experts]','Learned correction bias changes which experts are selected. It is never added to the mixture weights.','selection_scores = scores + selection_bias.float()', [['Weight ownership','selection only']],'state');
 operation(g,e,'select',1020,280,'Select expert indices','[tokens, topk]','Top-k is applied to biased scores. The result contains expert identities and selection slots, not mixture weights.','selected = topk(scores + selection_bias, topk).indices',[['topk',String(m.topK)],['Guard','1 < topk ≤ experts']]);
 operation(g,e,'weights',540,440,'Gather · normalize · scale','[tokens, topk]','Gather the unbiased scores at the selected indices, normalize across selected experts, then apply route_scale.','w = gather(scores,selected); w = w / (sum(w,-1)+1e-20) × route_scale',[['route_scale',String(m.routeScale)],['Normalization axis','selected experts']]);
 operation(g,e,'dispatch',60,600,'Dispatch selected token rows','token_indices, selection_slots','For each expert, find tokens selecting that expert. Keep the original token row and slot to gather its route weight and scatter the contribution back.','token_indices, selection_slots = where(selected == expert_id)',[['Expert loop',`experts = ${m.experts}`]]);
 operation(g,e,'routed-input',60,780,'Routed expert input','[selected_tokens, hidden_dim]','Each routed expert sees only its selected token rows. The weight tensor broadcasts over intermediate channels.','x_e = tokens[token_indices]; w_e = weights[token_indices,selection_slots,None]',[['Route weight shape','[selected_tokens, broadcast_hidden=1]']]);
 operation(g,e,'shared-input',780,780,'Shared expert input','[tokens, hidden_dim]','The shared expert processes every token directly, independently of the router and its top-k budget.','x_shared = tokens; weight_shared = 1.0',[['Shared experts',String(m.shared)]]);
 const expert=(prefix,left,right,y,shared)=>{
  const count=shared?'tokens':'selected_tokens';
  operation(g,e,`${prefix}-gate`,left,y,'Gate projection · upper clamp',`[${count}, hidden_width]`,'SwiGLU gate projection is clamped only above. Projection output is converted to float32 before the clamp.','gate = linear(x,W_gate).float().clamp(max=activation_limit)',[['hidden_width',String(m.expertFF)],['Checkpoint clamp',String(m.swigluLimit)]]);
  operation(g,e,`${prefix}-up`,right,y,'Up projection · signed clamp',`[${count}, hidden_width]`,'The parallel up branch uses both lower and upper clamp limits. It receives the same expert input as the gate branch.','up = linear(x,W_up).float().clamp(-activation_limit, activation_limit)',[['hidden_width',String(m.expertFF)]]);
  operation(g,e,`${prefix}-product`,left,y+180,'SiLU × up × weight',`[${count}, hidden_width]`,shared?'The shared expert uses scalar weight 1.0 and receives no routed score.':'Apply the selected route weight across intermediate channels before the down projection.','activated = silu(gate) × up × '+(shared?'1.0':'w_e'),[['Weight source',shared?'constant unit weight':'unbiased selected score']]);
  operation(g,e,`${prefix}-down`,left,y+360,'Down projection',`[${count}, hidden_dim]`,'Cast the weighted activation to the expert input dtype before the down projection. Convert its contribution to float32 for accumulation.','contribution = linear(activated.to(x.dtype), W_down).float()', [['hidden_dim',String(m.d)]]);
  wire(g,`${prefix}-input`,`${prefix}-gate`,[[left+180,848],[left+180,y]]);
  wire(g,`${prefix}-input`,`${prefix}-up`,[[left+300,848],[left+300,y-30],[right+180,y-30],[right+180,y]]);
  wire(g,`${prefix}-gate`,`${prefix}-product`,[[left+180,y+68],[left+180,y+180]]);
  wire(g,`${prefix}-up`,`${prefix}-product`,[[right+180,y+68],[right+180,y+134],[left+300,y+134],[left+300,y+180]]);
  wire(g,`${prefix}-product`,`${prefix}-down`,[[left+180,y+248],[left+180,y+360]]);
 };
 // Wider branch spacing uses two horizontal tiers so every operation retains reading width.
 expert('routed',60,460,960,false);
 operation(g,e,'shared-gate',620,1460,'Gate projection · upper clamp','[tokens, hidden_width]','The shared expert has independent projection parameters. Its gate is clamped only above, in float32.','gate = linear(tokens,W_gate).float().clamp(max=activation_limit)',[['hidden_width',String(m.expertFF)],['Checkpoint clamp',String(m.swigluLimit)]]);
 operation(g,e,'shared-up',1020,1340,'Up projection · signed clamp','[tokens, hidden_width]','The same shared token input feeds a parallel up projection, clamped on both sides.','up = linear(tokens,W_up).float().clamp(-activation_limit,activation_limit)',[['hidden_width',String(m.expertFF)]]);
 operation(g,e,'shared-product',1020,1640,'SiLU × up · unit weight','[tokens, hidden_width]','All tokens use scalar weight 1.0. No router score, correction bias or selected index reaches this branch.','activated = silu(gate) × up × 1.0',[['Weight source','constant unit weight']]);
 operation(g,e,'shared-down',1020,1820,'Shared down projection','[tokens, hidden_dim]','Cast activations to the input dtype before the down projection; accumulate the returned contribution in float32.','shared = linear(activated.to(tokens.dtype),W_down).float()',[['hidden_dim',String(m.d)]]);
 wire(g,'shared-input','shared-gate',[[960,848],[960,1420],[800,1420],[800,1460]]);
 wire(g,'shared-input','shared-up',[[1140,814],[1200,814],[1200,1340]]);
 wire(g,'shared-gate','shared-product',[[800,1528],[800,1590],[1140,1590],[1140,1640]]);
 wire(g,'shared-up','shared-product',[[1200,1408],[1200,1640]]);
 wire(g,'shared-product','shared-down',[[1200,1708],[1200,1820]]);
 operation(g,e,'scatter',60,1510,'Scatter-add routed outputs','[tokens, hidden_dim]','Float32 routed accumulator starts at zero. index_add restores each contribution to its original token row and sums across selected experts.','routed = routed.index_add(0,token_indices,contribution)',[['Accumulator dtype','float32']]);
 operation(g,e,'output',60,1970,'Routed + shared · restore shape','[..., hidden_dim]','Sum routed and shared outputs in float32, cast to original input dtype and restore all leading axes. The shared path never consumes router logits.','output = (routed + shared).to(inputs.dtype).reshape(inputs.shape)',[['Output contract','same shape and dtype as input']]);
 wire(g,'tokens','scores',[[420,154],[540,154]]);
 wire(g,'scores','select',[[900,154],[960,154],[960,250],[1140,250],[1140,280]]);
 wire(g,'selection-bias','select',[[1200,188],[1200,280]],'state');
 wire(g,'scores','weights',[[720,188],[720,440]]);
 wire(g,'select','weights',[[1200,348],[1200,474],[900,474]]);
 wire(g,'select','dispatch',[[1020,314],[480,314],[480,560],[240,560],[240,600]]);
 wire(g,'tokens','dispatch',[[240,188],[240,600]]);
 wire(g,'tokens','shared-input',[[360,188],[360,220],[1400,220],[1400,730],[960,730],[960,780]]);
 wire(g,'dispatch','routed-input',[[240,668],[240,780]]);
 wire(g,'weights','routed-input',[[720,508],[720,710],[360,710],[360,780]]);
 wire(g,'weights','routed-product',[[600,508],[600,540],[30,540],[30,1174],[60,1174]]);
 wire(g,'routed-down','scatter',[[240,1388],[240,1510]]);
 wire(g,'dispatch','scatter',[[60,634],[16,634],[16,1544],[60,1544]],'state');
 wire(g,'scatter','output',[[240,1578],[240,1970]]);
 wire(g,'shared-down','output',[[1200,1888],[1200,1920],[360,1920],[360,1970]]);
 g.edges.forEach((w,i)=>{w.id=`wire-${i}`;});
 g.defaultSelected='weights';return g;
}
export const BACKBONE_SCHEMATICS={'backbone-mhc':mhc,'backbone-moe':moe};
