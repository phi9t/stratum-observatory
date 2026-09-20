/** Residual transport through the published flat backbone. */
import {getModel} from './core.mjs';
import {makeScene,addNode,addGroup,addLabel,addWire,makeInfo} from './scene.mjs';
export function buildMHCResidual(s){
 const m=getModel(s.model),g=makeScene(1440,1230),shape=`[B, T, ${m.streams}, ${m.d}]`;
 const node=(id,x,y,w,h,title,sub,body,formula,tone='neutral')=>addNode(g,id,x,y,w,h,title,sub,{tone,info:makeInfo('Residual mathematics',title,body,formula,[],['dsv41-config','dsv41-code','dsv41-kernel'])});
 const wire=(a,b,points,kind='flow')=>addWire(g,points,a,b,{kind});
 addLabel(g,'MODEL / RESIDUAL TRANSPORT',36,38,'compute',{size:11,tracking:1.3});
 addLabel(g,'One stream axis. Carried through the entire model.',36,83,'text',{size:25});
 addLabel(g,`${m.streams} streams per token · ordinary sublayers still operate at width ${m.d}`,36,116,'muted',{size:13});
 addLabel(g,'WHOLE BACKBONE',36,170,'muted',{size:11,tracking:1.3});
 node('embedding',36,198,294,64,'Token / vision embeddings',`[B, T, ${m.d}]`,'Image features replace their embedded positions before expansion. Token time and model depth remain separate axes.','E = embed(tokens)');
 node('expand',36,304,294,76,'Expand once',shape,'Copy the embedding along the stream axis. The initial read gate selects the first copy, so the first read equals the embedding.','X_i = E; p_prev = [1, 0, 0, 0]','state');
 node('encoder',36,426,294,82,'Causal encoder',`layers 1–${m.encoderLayers}`,'Every block performs an attention read/write followed by an MoE read/write. Engram updates before its configured blocks preserve the stream axis and pending read gate.','(X, p) → Block(X, p)','compute');
 node('handoff',36,558,294,76,'Carry state and read gate',`after layer ${m.encoderLayers}`,'The boundary transfers both residual streams and the latest FFN read gate. There is no second embedding or stream expansion.','X: [B,T,R,D]; p: [B,T,R]','state');
 node('decoder',36,684,294,82,'Decoder',`layers ${m.encoderLayers+1}–${m.layers}`,'The decoder continues the same flat block list and residual recurrence. Its attention regime changes; the residual contract does not.','(X, p) → Block(X, p)','compute');
 node('collapse',36,834,294,82,'Final weighted collapse',`${m.streams} streams → ${m.d} channels`,'Use the last FFN mixer’s pending read gate. This is a weighted sum, not a mean and not a Sinkhorn operation.','h = Σ_i p_last,i X_i','state');
 node('head',36,982,294,82,'RMSNorm → vocabulary head',`[B, ${m.vocab}] last-token logits`,'Normalize the collapsed channels, then apply the vocabulary projection at the selected output position.','logits = W_head RMSNorm(h_last)');
 for(const [a,b,y0,y1] of [['embedding','expand',262,304],['expand','encoder',380,426],['encoder','handoff',508,558],['handoff','decoder',634,684],['decoder','collapse',766,834],['collapse','head',916,982]])wire(a,b,[[183,y0],[183,y1]]);
 addLabel(g,'EACH BACKBONE LAYER',410,170,'compute',{size:11,tracking:1.3});
 node('incoming',410,198,940,68,'Incoming residual and pending read gate',`${shape} + [B, T, ${m.streams}]`,'The incoming gate was predicted by the preceding sublayer. Both attention and MoE keep all residual streams.','X₀, p_prev','state');
 wire('encoder','incoming',[[330,458],[362,458],[362,232],[410,232]],'control');
 wire('decoder','incoming',[[330,718],[374,718],[374,250],[410,250]],'control');
 for(const [prefix,y,gate,input,output,title] of [['attention',310,'p_prev','X₀','X₁','01 ATTENTION'],['ffn',704,'p_attn','X₁','X₂','02 MIXTURE OF EXPERTS']]){
  addGroup(g,prefix+'-region',394,y,976,344,title,'');
  const read=node(prefix+'-read',426,y+48,424,62,'Read with the previous gate',`${gate} · ${m.streams} streams → ${m.d}`,'Sum across streams only. The current mixer’s new read gate is reserved for the next sublayer.',`h = Σ_i ${gate},i ${input},i`,'compute');
  node(prefix+'-mixer',926,y+48,424,76,'Predict current coefficients',`p_next, q, M · ${m.streams*(m.streams+2)} coefficients`,'Flatten incoming streams and channels, normalize by their joint RMS, then project. Split into next-read gates, current-write gates and a Sinkhorn residual matrix.',`(p_next, q, M) = Mixer(${input})`,'state');
  node(prefix+'-compute',426,y+160,424,58,prefix==='attention'?'RMSNorm → attention':'RMSNorm → MoE',`[B, T, ${m.d}] → [B, T, ${m.d}]`,'Ordinary learned RMSNorm acts on the collapsed channel axis. The sublayer computes one hidden-width output per token.','y = F(RMSNorm(h))','compute');
  const write=node(prefix+'-write',426,y+260,924,64,'Mix the old streams + write the sublayer output',`${output}: ${shape}`,'M has input-stream rows and output-stream columns. Each output stream receives a column-weighted sum of old streams plus its gated copy of the sublayer output.','X_next,j = Σ_i M_ij X_i + q_j y','state');
  read.info.facts=[['Read coefficients',`[B, T, ${m.streams}]`],['Output',`[B, T, ${m.d}]`]];
  write.info.facts=[['Residual mix',`[B, T, ${m.streams}, ${m.streams}]`],['Output gate',`[B, T, ${m.streams}]`],['Coefficient owner','current sublayer']];
  wire(prefix+'-read',prefix+'-compute',[[638,y+110],[638,y+160]]);
  wire(prefix+'-compute',prefix+'-write',[[638,y+218],[638,y+260]]);
  wire(prefix+'-mixer',prefix+'-write',[[1138,y+124],[1138,y+260]],'state');
  const source=prefix==='attention'?'incoming':'attention-write',sy=prefix==='attention'?266:634;
  wire(source,prefix+'-read',[[638,sy],[638,y+48]]);
  wire(source,prefix+'-mixer',[[1138,sy],[1138,y+48]]);
  wire(source,prefix+'-write',[[prefix==='attention'?410:426,sy-24],[402,sy-24],[402,y+292],[426,y+292]],'state');
 }
 wire('attention-mixer','ffn-read',[[1350,394],[1400,394],[1400,680],[878,680],[878,780],[850,780]],'control');
 addLabel(g,'p_attn → next read',1010,673,'state',{size:11});
 node('outgoing',426,1120,924,68,'Carry both into the next layer',`X₂ + p_ffn · final layer sends them to the output collapse`,'The current FFN mixer predicts the gate used by the next layer’s attention. After the final layer, that gate instead performs the output collapse.','next state = (X₂, p_ffn)','state');
 wire('ffn-write','outgoing',[[638,1028],[638,1120]]);
 wire('ffn-mixer','outgoing',[[1350,788],[1400,788],[1400,1154],[1350,1154]],'control');
 g.defaultSelected='attention-write';g.caption='Solid paths carry residuals or sublayer outputs; dashed paths carry the next read gate or point from the backbone to its expanded layer. Mixing acts across streams at each token, never across token positions. Engram updates preserve this state contract; dSpark observes separate mean-stream taps.';
 return g;
}
export function residualChapter(m){return {
 stage:'residual',title:'Residual state: read, compute, mix, carry',question:'How does one token move through the full backbone?',
 summary:`Each token retains ${m.streams} streams through all ${m.layers} layers. Attention and MoE each read a hidden-width vector, then update every stream. The read gate travels one sublayer ahead.`,
 math:[
 ['State and initialization',`X ∈ R^[B,T,${m.streams},${m.d}]\np_prev ∈ R^[B,T,${m.streams}]\nX_i = embedding; p_prev = [1, 0, 0, 0]`,'B indexes sequences, T token positions, i input streams, j output streams, and D hidden channels. The stream copies start equal and can diverge after learned gated writes.'],
 ['Read the previous gate; write current coefficients','h = Σ_i p_prev,i X_i\ny = F(RMSNorm(h))\nX_next,j = Σ_i M_ij X_i + q_j y\nnext read gate = p_next','F is attention or MoE. M has input-stream rows and output-stream columns: its contraction is MᵀX. Only the stream index is summed. q broadcasts the same sublayer output into separately gated streams.'],
 ['Predict the coefficients from incoming state',`v = flatten_streams_and_channels(X)\nz = W v / sqrt(mean(v²) + eps_norm)\np_next = sigmoid(a_pre z_pre + b_pre) + eps_hc\nq = 2 sigmoid(a_post z_post + b_post)\nL = reshape(a_mix z_mix + b_mix, [${m.streams}, ${m.streams}])`,'This RMS statistic spans all streams and channels together. The learned projection produces next-read gates, current-write gates and matrix logits. The separate sublayer RMSNorm has learned channel weights. Read and write gates are not probability distributions.'],
 ['Balance the residual matrix',`M = softmax_rows(L) + eps_hc\nM /= sum_over_input_streams(M) + eps_hc\nrepeat ${m.sinkhornIters-1} times:\n    M /= sum_over_output_streams(M) + eps_hc\n    M /= sum_over_input_streams(M) + eps_hc`,'Finite Sinkhorn rounds target approximately unit row and column sums. Epsilon and a finite iteration count prevent an exact doubly stochastic guarantee. This balancing applies only to M.'],
 ],
 trace:[['Attention','Read with the preceding FFN gate. Predict p_attn, q_attn and M_attn from the incoming residual; write with q_attn and M_attn.'],['MoE','Read the attention-updated residual with p_attn. Predict p_ffn, q_ffn and M_ffn, then update the residual again.'],['Boundary and output',`Both X and p_ffn cross layer ${m.encoderLayers}. After layer ${m.layers}, p_ffn collapses the streams before final normalization and the vocabulary head.`]],
 rows:[['Residual',`[B, T, ${m.streams}, ${m.d}]`,'Carried through depth'],['Read / write gates',`[B, T, ${m.streams}]`,'Previous read gate; current output gate'],['Mixing matrix',`[B, T, ${m.streams}, ${m.streams}]`,'Input stream × output stream'],['Mixer projection',`${m.streams*(m.streams+2)} coefficients`,'Read gate + write gate + matrix']],
 caveat:'The publisher runs a flat block list. Encoder/decoder labels partition attention regimes; they do not create separate residual networks. Engram injects before its configured blocks without replacing the pending gate. dSpark observes stream means, not this weighted read.',
 scope:'The CPU fixture below validates residual mechanics with reduced hidden width and explicit linear stand-ins for attention and MoE. It is not a checkpoint execution or a GPU performance result.',
 symbols:'Matrix axes and gate timing follow the pinned publisher implementation.',sources:['dsv41-config','dsv41-code','dsv41-kernel'],anchors:['Block.hc_mixes','Block.hc_pre','Block.hc_post','Block.forward','hc_split_sinkhorn_kernel','make_identity_pre_mix','Transformer.forward','ParallelHead.forward']
};}
