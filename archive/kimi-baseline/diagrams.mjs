import {derived,fmt,compact,gb,layerKind} from './core.mjs';
import {makeScene,addNode,addGroup,addLabel,addWire,addSum,makeInfo} from './scene.mjs';

const k3Info=(title,body,formula='',facts=[],links=[])=>makeInfo('KIMI K3 / DOCUMENTED MECHANISM',title,body,formula,facts,['k3-config','k3-code','k3-paper'],links);
const infraInfo=(title,body,formula='',facts=[],links=[])=>makeInfo('KIMI K3 / REPORTED SYSTEM',title,body,formula,facts,['k3-infra','k3-paper'],links);
function kit(g,s={}){
 return {
 N:(id,x,y,w,h,name,sub='',info=null,tone='neutral',kind='op',extra={})=>addNode(g,id,x,y,w,h,name,s.detail==='operators'||(kind==='pill'&&h<42)?'':sub,{info,tone,kind,...extra}),
 W:(points,source='',target='',options={})=>addWire(g,points,source,target,options),
 L:(text,x,y,tone='muted',options={})=>addLabel(g,text,x,y,tone,options),
 G:(...args)=>addGroup(g,...args),
 V:(x,y1,y2,a,b,kind='flow')=>addWire(g,[[x,y1],[x,y2]],a,b,{kind})
 };
}
function modelScene(s){
 const {m}=derived(s),g=makeScene(1680,1015),{N,W,L,G,V}=kit(g,s),kda=layerKind(s.layer)==='kda',dense=s.layer===1;
 const boundary=(s.layer-1)%12===0;
 g.defaultSelected=kda?'kda-core':'mla-core';
 g.caption=`Layer ${s.layer}/93: ${kda?'KDA recurrence':'Gated MLA (NoPE)'} + ${dense?'dense SiTU FFN':'Stable LatentMoE'}. Projection labels omit batch/token axes. The depth bank and local partial-sum rails implement Block AttnRes, not a conventional residual connection. Dashed state loop advances token time; it is not a same-token cycle.`;
 L('KIMI K3 / NATIVE MULTIMODAL',22,43,'faint',{small:true});
 N('embedding',22,79,210,48,'Text embedding','163840 → 7168',k3Info('Text and vision meet at the input','The vocabulary has 163,840 entries. Token embeddings and projected visual features are interleaved in a jointly trained multimodal backbone.','x_text = Embedding(token_id)'));
 N('vision',22,161,210,62,'MoonViT-V2','27 layers · 14 × 14 patches',k3Info('Native vision, not a post-hoc attachment','The report describes a 401M-parameter visual encoder. Released configuration: 27 layers, width 1024, 12 attention heads and patch size 14. Vision uses its own positional encoding; NoPE refers to the backbone MLA layers.','image / video → patches → ViT',[['Visual width','1024'],['ViT parameters','401M (reported)']]));
 N('projector',22,255,210,48,'Patch merge + projector','2 × 2 merge → 7168',k3Info('Visual-to-language projection','The release specifies a 2×2 merge kernel and a patchmergerv2 projector into the 7168-wide text stream. The drawing folds image/video tokenization and temporal pooling.','visual features → merge → projection'));
 V(42,223,255,'vision','projector');
 N('input-mix',22,343,210,48,'Interleave modalities','One causal token stream',k3Info('Joint next-token objective','Visual and text features share the causal decoder. This is a sequence merge, not addition of unrelated feature vectors.','X = interleave(X_text, X_visual)'));
 W([[232,103],[247,103],[247,365],[232,365]],'embedding','input-mix');V(42,303,343,'projector','input-mix');
 N('hybrid-stack',22,431,210,70,'93 decoder layers','[KDA ×3 → MLA] ×23 + MLA',k3Info('A precise hybrid schedule','Layers 4, 8, …, 92 and 93 use gated MLA. All other layers use KDA. Only layer 1 uses the dense FFN; layers 2–93 use Stable LatentMoE.','69 KDA + 24 MLA = 93',[['Selected layer',String(s.layer)],['Final layer','Gated MLA']]),'accent');V(42,391,431,'input-mix','hybrid-stack');
 N('final-head',22,541,210,62,'Final AttnRes + head','RMSNorm · 7168 → 163840',k3Info('Depth retrieval precedes vocabulary projection','A final attention-residual read aggregates the depth sources, then RMSNorm and the untied language-model head produce vocabulary logits.','logits = W_vocab RMSNorm(AttnRes(bank))'));V(42,501,541,'hybrid-stack','final-head');
 L('SELECT A DECODER LAYER',22,650,'faint',{small:true});
 for(let i=1;i<=93;i++){const x=22+((i-1)%12)*17,y=678+Math.floor((i-1)/12)*23;N(`layer-${i}`,x,y,12,14,'','',k3Info(`Layer ${i} · ${layerKind(i).toUpperCase()}`,`${i===1?'Dense FFN':'LatentMoE'}; attention-residual block ${Math.ceil(i/12)}.`),'', 'page',{tone:i===s.layer?'accent':layerKind(i)==='mla'?'comm':'state',action:`layer:${i}`});}
 L('KDA 69',22,889,'state',{small:true});L('MLA 24',118,889,'comm',{small:true});
 L('1 M tokens · base config',22,933,'muted',{small:true});L('MTP is not added to this base graph.',22,959,'faint',{small:true});
 G('expanded',265,18,1385,958,`DECODER LAYER ${String(s.layer).padStart(2,'0')}`,`depth block ${Math.ceil(s.layer/12)}/8 · ${boundary?'boundary':'within block'}`);
 N('depth-bank',320,78,1240,44,'Depth source bank','Embedding + completed block sums · immutable sources for this layer',k3Info('Attention Residuals: attend over depth','Learned per-sublayer pseudo-queries score RMS-normalized depth representations. Softmax runs over sources at the same token, not over sequence positions. Eight decoder blocks (size 12, final size 9) plus the embedding form up to nine terminal sources. At a boundary, the old partial sum is sealed into the bank before resetting the local accumulator.','a_i = softmax_i(wᵀ RMSNorm(b_i))\nh = Σ_i a_i b_i',[['Axis','Depth, independently per token'],['Local state','Partial sum inside the current block']]),'state');
 const arInfo=k3Info('Block AttnRes read','Retrieve from embedding/completed blocks and, inside a block, the current partial sum. Norm follows retrieval. This is not x + f(x) at the input of each sublayer.','h = AttnRes(bank, partial_sum; w_layer)');
 N('attnres-a',345,157,485,42,'AttnRes · attention input','Learned pseudo-query over depth',arInfo,'state');V(365,122,157,'depth-bank','attnres-a','state');
 N('norm-a',345,219,485,30,'RMSNorm','',k3Info('Token-wise input normalization','Feature normalization follows attention-residual mixing; it does not mix positions or depth sources.','x = γ ⊙ h / √(mean(h²)+ε)'),'neutral','pill');V(365,199,219,'attnres-a','norm-a');
 N('attnres-f',965,157,590,42,'AttnRes · FFN input','Bank + updated local partial sum',arInfo,'state');V(985,122,157,'depth-bank','attnres-f','state');
 N('norm-f',965,219,590,30,'RMSNorm','',k3Info('FFN input normalization','Both the router and shared/routed branches consume this normalized full-width vector.','x ∈ R⁷¹⁶⁸'),'neutral','pill');V(985,199,219,'attnres-f','norm-f');
 G('attention',325,275,550,546,kda?'KIMI DELTA ATTENTION':'GATED MLA','',{titleOffset:56});
 if(kda){
  N('qkv',345,315,280,50,'Q / K / V projections','3 × (7168 → 12288)',k3Info('Three full-width projections','Each projection produces 96 heads ×128 features. The three operators are grouped visually; this does not assert a particular GEMM fusion.','Q=W_qx; K=W_kx; V=W_vx'));
  V(365,249,315,'norm-a','qkv');
  N('conv',345,395,280,42,'Causal short convolution + SiLU','Kernel width 4 · Q, K and V',k3Info('Short-convolution state','Each of Q/K/V has its own causal depthwise short convolution followed by SiLU. Streaming inference retains convolution history separately from the KDA matrix state.','q′=SiLU(Conv₄(W_qx))\nk′=SiLU(Conv₄(W_kx))\nv=SiLU(Conv₄(W_vx))'));V(365,365,395,'qkv','conv');
  N('qk-norm',345,465,280,34,'L2-normalize Q and K','V is not L2-normalized',k3Info('Normalize query and key directions','Unit-length keys control the delta update. Values retain their projected scale.','q=q′/‖q′‖₂; k=k′/‖k′‖₂'),'neutral','pill');V(365,437,465,'conv','qk-norm');
  N('decay-project',680,315,150,50,'β + decay logits','96 + 96 × 128',k3Info('Input-dependent write and retention','β uses a 7168→96 projection followed by sigmoid. Decay logits use a low-rank 7168→128→12288 path plus a channel bias.','β=sigmoid(W_βx)\nz=W_b W_a x+b_α'));
  W([[705,249],[705,263],[750,263],[750,315]],'norm-a','decay-project');
  N('bounded-decay',680,395,150,42,'Bounded decay','g ∈ (−5, 0)',k3Info('Numerical range is an architectural choice','K3 changes the log-decay to a lower-bounded sigmoid mapping. Retention is exp(g), not sigmoid(g). This enables Tensor-Core-friendly chunkwise computation without the old unbounded reciprocal range.','g = −5 sigmoid(exp(A)z)\nα = exp(g) ∈ (exp(−5),1)'),'accent');V(750,365,395,'decay-project','bounded-decay');
  N('kda-core',345,555,280,60,'Delta-rule recurrence','Update Sₜ, then read with qₜ',k3Info('KDA: erase a prediction, write a correction','The channel-wise gate first decays the previous key rows. The delta rule subtracts the key-conditioned prediction and writes the new value. Output reads the updated state, including the current token. Chunkwise training is parallel within chunks; decode updates one running state.','Sₜ = (I−βₜkₜkₜᵀ) Diag(αₜ) Sₜ₋₁\n     + βₜkₜvₜᵀ\noₜ = Sₜᵀqₜ',[['Logical state / head','128 × 128'],['Sequence growth','Fixed running matrix, not token KV']],[['training','Follow chunkwise KCP'],['serving','Inspect recurrent cache']]),'accent');V(365,499,555,'qk-norm','kda-core');
  W([[750,437],[750,523],[595,523],[595,555]],'bounded-decay','kda-core');
  N('kda-state',680,560,150,56,'Sₜ₋₁ → Sₜ','96 × 128 × 128',k3Info('Persistent recurrent matrix','The box denotes the state transition between token steps. The feedback edge crosses time. Logical axes are key×value; physical kernels may transpose them. Conv history is additional.','running state: O(H dₖ dᵥ) / request'),'state');
  W([[680,581],[625,581]],'kda-state','kda-core',{kind:'state'});
  W([[625,603],[648,603],[648,646],[795,646],[795,616]],'kda-core','kda-state',{kind:'state',delayed:true});
  N('gated-output',345,689,280,44,'Head RMSNorm × output gate','96 × 128',k3Info('Head normalization and full-rank gating','A separate full-rank projection supplies a sigmoid gate. It scales normalized recurrent output before the output projection.','y = W_o[sigmoid(W_gx) ⊙ RMSNorm_head(o)]'));V(365,615,689,'kda-core','gated-output');
 }else{
  N('q-down',345,315,215,50,'Q down-proj','7168 → 1536',k3Info('Low-rank query path','The query latent is normalized and expanded into 96 heads of 192 channels.','q_latent = W_q↓ x'));
  N('kv-down',590,315,240,50,'KV down-proj + shared key','7168 → 512 + 64',k3Info('NoPE still has shared key channels','The code retains the 64-dimensional shared key branch although it disables rotation. The legacy config name qk_rope_head_dim does not imply RoPE is applied.','[c, k_shared] = W_kv↓x\nwidth = 512 + 64'));
  V(365,249,315,'norm-a','q-down');W([[705,249],[705,315]],'norm-a','kv-down');
  N('q-latent-norm',345,395,215,32,'RMSNorm · Q latent','1536',k3Info('Normalize compressed queries','RMSNorm acts before the query up-projection.','q̄=RMSNorm(q_latent)'),'neutral','pill');V(365,365,395,'q-down','q-latent-norm');
  N('kv-latent-norm',590,395,240,32,'RMSNorm · KV latent','512 only',k3Info('Normalize the content latent','The normalization is applied to the 512-dimensional content latent, not the separate 64 shared key channels.','c̄=RMSNorm(c)'),'neutral','pill');V(610,365,395,'kv-down','kv-latent-norm');
  N('q-up',345,460,215,50,'Q up-proj','1536 → 96 × 192',k3Info('Expanded queries','192 =128 content channels +64 channels paired with the shared key. Neither branch is rotated.','q = W_q↑q̄'));V(365,427,460,'q-latent-norm','q-up');
  N('kv-up',590,460,240,50,'KV up-proj','512 → 96 × (128 + 128)',k3Info('Materialized K and V','The explicit-head view reconstructs 128 content-key and 128 value channels per head. Optimized decoding can absorb these projections and cache the latent.','[k_content,v] = W_kv↑c̄'));V(610,427,460,'kv-latent-norm','kv-up');
  N('nope',590,541,240,34,'Build keys · NoPE','K = [content, shared 64]',k3Info('Global content attention without RoPE','Concatenate content keys with the shared 64-dimensional branch. KDA layers supply order-sensitive sequence mixing. No explicit positional encoding is applied to MLA Q or K.','K_h=[K_content,h, k_shared]'),'neutral','pill');V(610,510,541,'kv-up','nope');W([[830,340],[850,340],[850,558],[830,558]],'kv-down','nope');
  N('mla-core',345,610,485,50,'Causal global attention','Q,K: 96 × 192 · V: 96 × 128',k3Info('Gated multi-head latent attention','Global causal softmax attention complements local recurrent state. The explicit head tensors are not the optimized persistent cache. The latter retains 512 latent +64 shared-key elements per token and layer.','o = softmax(QKᵀ / √192 + causal_mask)V',[['Compressed cache','576 elements / layer / token'],['Expanded reference','96 × (192 + 128) elements']],[['serving','Compare the two cache representations']]),'accent');V(365,510,610,'q-up','mla-core');V(710,575,610,'nope','mla-core');
  N('gated-output',345,689,280,44,'Attention output × gate','96 × 128',k3Info('MLA also uses a full-rank output gate','A data-dependent sigmoid gate scales the raw attention output before W_o. Do not import the KDA head-output RMSNorm into the MLA formula.','y = W_o[sigmoid(W_gx) ⊙ o]'));V(365,660,689,'mla-core','gated-output');
 }
 N('output-gate',680,689,150,44,'Full-rank gate','7168 → 12288',k3Info('Not a low-rank output gate','K3 uses a full-rank 7168×12288 gate matrix for both token mixers.','g_out = sigmoid(W_g x)'));
 W([[830,234],[891,234],[891,711],[830,711]],'norm-a','output-gate');W([[680,711],[625,711]],'output-gate','gated-output');
 N('attention-out',345,760,485,44,'Attention output projection','12288 → 7168',k3Info('Return to model width','The gated multi-head output is projected back into the 7168-dimensional depth representation.','f_attn = W_o gated_output'));V(365,733,760,'gated-output','attention-out');
 const plusInfo=k3Info('Local partial-sum update','Inside an AttnRes block, add the new sublayer output to the block accumulator. The next sublayer retrieves from this accumulator plus the depth bank. At a new block the accumulator starts at zero.','p ← p + f_sublayer');
 addSum(g,'attention-add',365,870,{info:plusInfo});V(365,804,858,'attention-out','attention-add');
 L(boundary?'p = 0 after sealing':'incoming partial p',440,141,'faint',{small:true});
 W([[285,151],[285,870],[353,870]],'','attention-add',{kind:'residual'});
 W([[285,178],[345,178]],'','attnres-a',{kind:'residual'});
 W([[365,882],[365,906],[915,906],[915,178],[965,178]],'attention-add','attnres-f',{kind:'residual'});
 G('ffn',945,275,645,546,dense?'DENSE SiTU FFN · LAYER 1':'STABLE LATENT MoE',dense?'Full-width path':'896 routed · top 16 · 2 shared',{titleOffset:50});
 if(dense){
  N('dense-up',965,365,590,58,'Dense gate / up projections','7168 → 2 × 33792',k3Info('Only decoder layer 1 is dense','The released intermediate_size is 33792. Subsequent layers use the distinct latent-expert width.','g=W_gx; u=W_ux'));V(985,249,365,'norm-f','dense-up');
  N('dense-situ',965,493,590,48,'SiTU-GLU','Smooth caps: gate 4 · up 25',k3Info('Both multiplicative branches are bounded','SiTU preserves a SwiGLU-like small-input response while softly bounding each branch. The ≤100 bound applies before the down-projection, not to final network activations.','SiTU(g,u) = [4 tanh(g/4) sigmoid(g)]\n            ⊙ [25 tanh(u/25)]'),'accent');V(985,423,493,'dense-up','dense-situ');
  N('dense-down',965,635,590,58,'Dense down-projection','33792 → 7168',k3Info('Dense FFN output','Project the bounded activation back to model width before the local partial-sum update.','f_ffn = W_down SiTU(g,u)'));V(985,541,635,'dense-situ','dense-down');V(985,693,918,'dense-down','ffn-add');
 }else{
  N('latent-down',965,315,320,50,'Latent down-projection','7168 → 3584',k3Info('Communicate in half-width latent space','The routed path compresses full-width x before dispatch. Shared experts retain x directly. The factor-of-two vector-byte reduction is not a claim of twice the end-to-end throughput.','z = W↓x ∈ R³⁵⁸⁴',[['Per-token copies','16 selected routes']],[['network','Price the routed payload']]),'accent');V(985,249,315,'norm-f','latent-down');
  N('router',1335,315,220,50,'Sigmoid router','7168 → 896',k3Info('Route full-width input, not the latent','The router scores are sigmoid(W_r x). Quantile Balancing learns selection corrections; the expert vectors are z, not logits.','s=sigmoid(W_r x)'));V(1435,249,315,'norm-f','router');
  N('topk',1335,395,220,42,'Top-16 + unbiased weights','Select s+b; normalize selected s',k3Info('Selection bias is not a mixture weight','Add the expert bias only for Top-k selection. Gather the original sigmoid scores and normalize them across the selected experts. Quantile Balancing updates the bias during training.','T=Top16(s+b)\np_i=s_i / Σ_{j∈T}s_j',[['Expert pool','896'],['Selection','16 per token']]));V(1435,365,395,'router','topk');
  N('dispatch',965,410,320,36,'Dispatch latent tokens','16 × 3584 · distributed lowering',infraInfo('Expert dispatch','Permutation and transport place each latent vector with the selected experts. MoonEP adds planning, redundant-expert migration and balanced execution; this node is not a claim about a particular all-to-all algorithm.','logical bytes = T × 16 × 3584 × b'),'comm');V(985,365,410,'latent-down','dispatch');
  W([[1335,410],[1296,410],[1296,422],[1285,422]],'topk','dispatch',{kind:'metadata'});
  G('routed-experts',953,479,343,207,'ROUTED EXPERTS','×896',{titleOffset:50});
  N('expert-up',965,516,320,44,'Expert gate / up','3584 → 2 × 3072',k3Info('Routed experts operate at latent width','Every routed expert maps a 3584-wide latent vector through two 3072-wide branches and a 3072→3584 down-projection.','E_i(z)=W_down,i SiTU(W_gate,i z, W_up,i z)'));V(985,446,516,'dispatch','expert-up');
  N('expert-situ',965,578,320,32,'SiTU-GLU · caps 4 and 25','',k3Info('Smoothly capped gated activation','The gate is 4·tanh(g/4)·sigmoid(g); the value branch is 25·tanh(u/25). Their product has scalar magnitude ≤100 before the final linear map.','a=[4 tanh(g/4) sigmoid(g)] ⊙ [25 tanh(u/25)]'),'neutral','pill');V(985,560,578,'expert-up','expert-situ');
  N('expert-down',965,634,320,44,'Expert down-projection','3072 → 3584',k3Info('Return each expert to latent width','Apply each expert down-projection before weighted aggregation. Routing coefficients do not scale the shared path.','e_i = W_down,i a_i'));V(985,610,634,'expert-situ','expert-down');
  N('weighted-sum',965,708,320,38,'Combine · weighted sum','u = Σ pᵢ Eᵢ(z) · 3584',k3Info('Aggregate first, normalize second','The returned expert vectors are weighted and summed in the latent space. Only the aggregate u is RMS-normalized before W↑.','u = Σ_{i∈T} p_i E_i(z)'),'comm');V(985,678,708,'expert-down','weighted-sum');W([[1335,428],[1314,428],[1314,727],[1285,727]],'topk','weighted-sum',{kind:'metadata'});
  N('routed-norm',965,768,145,32,'RMSNorm','3584',k3Info('Normalization after the routed aggregate','This normalization is the stability intervention in Normalized LatentMoE. Moving it independently into every expert changes the function.','ū = RMSNorm(Σ_i p_iE_i(z))'),'neutral','pill');V(985,746,768,'weighted-sum','routed-norm');
  N('latent-up',1130,768,155,44,'Latent up-proj','3584 → 7168',k3Info('Lift the normalized routed result','After normalization, the shared W↑ lifts the latent aggregate back to model width.','r = W↑ū'));W([[1110,784],[1130,784]],'routed-norm','latent-up');
  N('shared-up',1370,516,185,44,'Shared gate / up ×2','7168 → 2 × 6144',k3Info('Two full-width shared experts','Two 3072-wide shared experts are shown as one fused 6144-wide FFN. Their inputs come directly from normalized x, not from z or router scores.','shared(x)=Σ_{j=1}² E_shared,j(x)'));
  W([[1555,234],[1612,234],[1612,538],[1555,538]],'norm-f','shared-up');
  N('shared-situ',1370,591,185,32,'SiTU-GLU','Not router-weighted',k3Info('Shared activation','Shared experts use SiTU-GLU but do not multiply by selected routing probabilities.','a_shared = SiTU(g_shared,u_shared)'),'neutral','pill');V(1390,560,591,'shared-up','shared-situ');
  N('shared-down',1370,687,185,44,'Shared down','6144 → 7168',k3Info('Sum the full-width shared experts','The fused down projection corresponds to summing two independent expert outputs into the residual-width representation.','s = W_shared,down a_shared'));V(1390,623,687,'shared-situ','shared-down');
  addSum(g,'branch-add',985,870,{info:k3Info('Routed plus shared','The two paths meet only after the routed aggregate is normalized and lifted to full model width.','f_ffn = W↑ RMSNorm(u) + Σ_j E_shared,j(x)')});
  W([[1205,812],[1205,841],[985,841],[985,858]],'latent-up','branch-add');W([[1390,731],[1390,870],[997,870]],'shared-down','branch-add');V(985,882,918,'branch-add','ffn-add');
 }
 addSum(g,'ffn-add',985,930,{info:plusInfo});W([[915,906],[915,930],[973,930]],'attention-add','ffn-add',{kind:'residual'});
 L('p ← p + attention output',390,875,'faint',{small:true});L('updated block partial → next layer',1015,935,'state',{small:true});
 V(985,942,957,'ffn-add','');
 return g;
}

function trainingScene(s){
 const g=makeScene(1400,910),{N,W,L,G,V}=kit(g),rl=s.trainingMode==='agentic';
 g.defaultSelected=rl?'external-pool':'moonep';
 g.caption=rl?'Reported agentic-RL mechanisms: colocated policy training and partial rollouts, joint KDA/MLA retention, NVMe phase offload and resumable microVMs. Arrows show dependencies and lifecycle transitions, not measured overlap or a cluster bill of materials.':'Reported pretraining design: PP/virtual stages, EP, ZeRO-1 DP, Pipeline ZeRO-2 gradients and KDA context parallelism. Columns group related mechanisms; only selected dependencies are wired. This is not a wall-clock trace or a claim about exact parallelism degrees.';
 L(rl?'LONG-HORIZON RL / STATE SURVIVES THE ITERATION':'PRETRAINING / ALGORITHM–SYSTEM CO-DESIGN',30,44,'faint',{small:true});
 L(rl?'Keep model state, prefix state and environment state on separate lifecycles.':'The training graph is not an Adam + FSDP substitute. These are the reported K3 mechanisms.',30,79,'muted');
 const col=[50,510,970],w=370,ys=[176,286,396,506,616];
 for(let i=0;i<3;i++)G(`train-group-${i}`,col[i]-20,120,410,650,rl?['POLICY / TRAIN ↔ ROLLOUT','HIERARCHICAL MODEL STATE','AGENTENV / ENVIRONMENT STATE'][i]:['DEPTH & CONTEXT PARALLELISM','MOONEP / EXPERT EXECUTION','MEMORY & OPTIMIZATION'][i]);
 function lane(items,x,sequence=false){items.forEach(([id,name,sub,info,tone='neutral'],i)=>{N(id,x,ys[i],w,62,name,sub,info,tone);if(i&&sequence)V(x+20,ys[i-1]+62,ys[i],items[i-1][0],id);});}
 if(!rl){
 lane([
 ['native-vision','Native vision + text','Dynamic image CP · gather-KV',infraInfo('Balance the multimodal encoder','Large images are split across context-parallel ranks along the patch axis. ViT work is scheduled into virtual-pipeline bubbles; some initial/final work remains exposed.','image patches → CP encoder → projected visual tokens')],
 ['pipeline','Pipeline + virtual stages','Interleaved 1F1B · PP/VP',infraInfo('Depth is split across physical and virtual stages','The paper combines pipeline and virtual pipeline stages with expert, data and context parallelism. It does not justify multiplying an arbitrary set of degrees or assuming every tensor has the same sharding.','stage i → activation / block cache → stage i+1')],
 ['kcp','KDA context parallelism','Exchange transition AND zero-state',infraInfo('Affine state composition, not a sum of states','Each segment produces a cumulative transition A and a zero-initial-state contribution B. Gather both, then compose in document order to recover incoming states. Pure head TP cannot shorten the recurrence.','S_out = A S_in + B\n(A₂,B₂) ∘ (A₁,B₁)\n  = (A₂A₁, A₂B₁+B₂)',[['Communication','Fixed-size matrices / segment']],[['network','Inspect KCP all-gather']]),'comm'],
 ['attnres-training','Block AttnRes checkpointing','Only new block sums cross PP',infraInfo('Depth-bank memory is explicitly managed','Generate each completed block representation once, keep it on GPU, and checkpoint retrieval computation. A cache-based pipeline protocol transfers only newly generated blocks.','completed block → incremental transfer → reuse'),'state'],
 ['pipeline-next','Next virtual stage','ViT work occupies available bubbles',infraInfo('Schedule overlap is a dependency property','Expert communication, shared-expert computation, activation movement and encoder work may overlap when dependencies allow. This diagram is not a measured schedule.','forward / backward / communication / I/O')]
 ],col[0]);
 lane([
 ['moonep','MoonEP online planner','Bounded redundant experts',infraInfo('Balance ranks with dynamic expert migration','For the current microbatch and layer, the GPU planner chooses redundant experts and prefetches weights. The report proves a bound of E/R redundant expert slots per rank in its balanced setting. These are additional physical copies, not additional learned experts.','reserve ≤ E/R redundant expert slots / rank',[['Logical expert count','896 remains unchanged']]),'accent'],
 ['ep-dispatch','Fused permute + dispatch','Direct remote expert-grouped placement',infraInfo('A zero-copy communication layout','Planned destinations permit tokens to land directly in their expert-grouped remote buffer positions. Fixed balanced rank shapes remove per-layer host synchronization.','z = W↓x → dispatch(z, plan)'),'comm'],
 ['ep-compute','Workload-aware expert GEMMs','Shared experts on a separate stream',infraInfo('Balanced ranks do not imply balanced SMs','Token counts may remain skewed between local experts. A workload-aware grouped-GEMM scheduler balances execution inside each rank; shared-expert GEMMs use an independent stream.','grouped gate/up → SiTU → grouped down'),'accent'],
 ['ep-combine','Combine + latent RMSNorm','Weighted aggregate → lift → shared add',infraInfo('Restore the mathematical model','The transport layout must preserve expert identity, mixture probabilities and token order. Normalization follows aggregation, then the latent result is projected up.','y=W↑ RMSNorm(Σ_i p_iE_i(z)) + shared(x)'),'comm'],
 ['ep-backward','Backward + home-rank reduction','Recompute dispatch; return replica grads',infraInfo('Redundant execution does not change ownership','Redundant-expert gradients are staged locally and reduced to the home ranks. Recompute group-GEMM input by dispatching again, rather than retaining all permuted forward activations.','replica gradients → owner gradient buffers')]
 ],col[1],true);
 lane([
 ['gradient-shards','Pipeline ZeRO-2 gradients','Double GPU buffers → CPU shards',infraInfo('Gradient sharding and CPU accumulation','Gradient reduce-scatter lands in double GPU buffers, then accumulates into CPU-resident shards. Do not interpret this as all gradients being absent from HBM.','DP reduce-scatter → GPU buffer → CPU shard'),'state'],
 ['activation-policy','Unified activation manager','FP8 · recompute · CPU / remote offload',infraInfo('Storage policy is separate from model code','Tensor-level policy combines blockwise FP8 activation quantization, recomputation and local/remote offload. Main-stream allocation in one pool reduces fragmentation.','saved tensor → selected storage backend'),'state'],
 ['remote-offload','Balance activation residency','Mooncake Transfer Engine across PP',infraInfo('Pipeline stages have uneven live activations','Interleaved 1F1B produces different resident activation counts during warmup. Remote offload uses spare memory on other pipeline ranks; prefetch overlaps computation.','busy PP HBM → spare PP HBM → onload'),'comm'],
 ['muon','Per-Head Muon + ZeRO-1','Owner-directed P2P matrix assembly',infraInfo('Gather only matrices needed by the optimizer owner','Muon orthogonalization requires full matrices, while optimizer state is sharded. Owners retrieve the needed shards via P2P rather than materializing the full model on every DP rank.','owner P2P gather → Newton–Schulz → update',[['Not represented','A universal 16-byte Adam ledger']]),'accent'],
 ['optimizer-commit','Commit optimizer update','Chunkwise communication / compute overlap',infraInfo('Different tensor families have different storage','Exact peak HBM depends on pipeline placement, expert replicas, activation policies and optimizer buffers. A single bytes-per-parameter number cannot reconstruct this training system.','update → distribute updated shards')]
 ],col[2]);
 V(70,238,286,'native-vision','pipeline');
 W([[420,317],[465,317],[465,207],[510,207]],'pipeline','moonep');
 W([[510,537],[465,537],[465,647],[420,647]],'ep-combine','pipeline-next');
 W([[880,647],[945,647],[945,207],[970,207]],'ep-backward','gradient-shards',{kind:'comm'});
 V(990,348,396,'activation-policy','remote-offload','state');
 W([[1340,207],[1374,207],[1374,537],[1340,537]],'gradient-shards','muon',{kind:'comm'});
 V(990,568,616,'muon','optimizer-commit');
 }else{
 lane([
 ['policy-step','Policy training phase','Chunkwise forward / loss / backward',infraInfo('Colocated RL','Policy training and generation share the accelerator pool across phases. Partial rollouts keep long trajectories alive between iterations.','policy update ↔ rollout iteration'),'accent'],
 ['reference-buffer','Reference forward on demand','Reuse policy gradient-buffer storage',infraInfo('No second permanent reference replica','Reference weights live on CPU and stream through the policy model’s gradient-buffer slots when those slots are safe to reuse. Two chunks permit compute/prefetch overlap.','reference weights → reusable GPU buffer')],
 ['rollout','Partial rollout generation','Resume unfinished trajectories',infraInfo('Avoid waiting for every long trajectory','Partial rollouts limit tail waiting, but make retained prefixes especially valuable at the next iteration.','resume prefix → generate → tool call'),'accent'],
 ['throttle','Auto-throttling scheduler','Active / queued requests + cache pressure',infraInfo('Concurrency should change as contexts grow','The scheduler adjusts inference admission using request counts and cache utilization rather than a fixed average-length concurrency target.','cache pressure ↑ → request admission ↓')],
 ['reward','Verifier / reward → policy update','Environment state, not self-reported success',infraInfo('Ground rewards in actual task state','Agentic workloads produce tool observations and verifier feedback. Model and environment state must be recoverable across iteration boundaries.','trajectory + verifier result → RL update')]
 ],col[0]);
 lane([
 ['gpu-working','GPU active state','Running KDA + active MLA blocks',infraInfo('Keep the active decode path resident','Active decode blocks stay on GPU. Do not make unnecessary CPU copies while they remain active.','active KDA / MLA → GPU'),'state'],
 ['external-pool','External prefix pool · CPU DRAM','Write back only upon GPU eviction',infraInfo('Keep KDA and MLA lifecycles aligned','Idle reusable prefixes move to an external DRAM pool when evicted, then prefetch before reuse. KDA states move with matching MLA blocks. This is write-back, not unconditional write-through.','evict → DRAM retention → prefetch',[['Invariant','Same token boundary for all cache groups']]),'state'],
 ['nvme-state','Training state → NVMe','After training, before rollout',infraInfo('Free DRAM for the external prefix pool','Offload training weights and optimizer states to NVMe after the training iteration. Release the external cache pool before the training phase needs its memory again.','training end → NVMe offload → rollout'),'blue'],
 ['restore-training','Release pool / restore training state','Phase ownership is explicit',infraInfo('A residency plan changes across phases','The same bytes cannot be counted simultaneously as free space for both training and rollout. Transfer and restore dependencies remain part of the critical path.','rollout end → release pool → restore state')],
 ['resume-prefix','Restore a consistent prefix','KDA snapshot + MLA + convolution history',infraInfo('Reuse means restoring complete execution state','A running recurrent matrix without matching token history and convolution state is not a valid continuation.','prefix boundary B → all required state at B'),'state']
 ],col[1]);
 lane([
 ['agentenv','AgentENV microVM','Firecracker isolation',infraInfo('Environment state is independent of model cache','The report describes microVM isolation for general agent workloads, in addition to container and GPU sandbox runtimes. This lane is not an LLM hidden-state cache.','agent tool call → isolated environment')],
 ['pause','Pause / resume','No active CPU or memory while paused',infraInfo('Pause while the agent waits for inference','Checkpoint the environment and release its running resources during idle intervals. Resume retains task progress rather than rebuilding from a clean container.','pause(env) → snapshot → resume(env)'),'state'],
 ['fork','Fork for evaluation','Preserve the original running state',infraInfo('Evaluate without mutating the original','Fork a sandbox state for judging or alternate execution while retaining the original branch.','env → copy-on-write branch')],
 ['snapshot','Incremental snapshots','Dirty pages + durable recovery points',infraInfo('Only changed state needs to be saved','Incremental checkpointing stores modified pages. Images and filesystem state use shared storage mechanisms described by the report.','dirty pages → incremental checkpoint'),'blue'],
 ['environment-result','Tool observations / final state','Verifier feeds the learning loop',infraInfo('Restore both sides of an agent trajectory','A durable trajectory includes the conversation/model prefix and the external environment state. Restoring only one can invalidate future observations.','tool output → trajectory / verifier')]
 ],col[2]);
 }
 L('READ THE CONTRACT, NOT THE DISTANCE BETWEEN BOXES.',30,819,'faint',{small:true});
 N('training-scope',30,840,1340,48,rl?'Model cache ≠ optimizer state ≠ sandbox state':'Reported mechanisms; no invented GPU count, parallel degrees or measured timeline',rl?'Three lifecycles; one reproducible trajectory.':'Hardware and network controls below are separate illustrative scenarios.',infraInfo('Scientific boundary','The paper supports the mechanisms shown. Exact cluster placement, end-to-end memory peaks and measured schedules are not reconstructed by this diagram.','documented mechanism ≠ inferred deployment'),'neutral');
 return g;
}

function servingScene(s){
 const d=derived(s),g=makeScene(1450,1055),{N,W,L,G,V}=kit(g),split=s.serveMode==='disaggregated';
 g.defaultSelected='prefix-match';
 g.caption='Optimized cache layout is an explicitly selected scenario; the reference implementation can materialize expanded K/V. Running KDA matrices, BF16 short-convolution buffers and MLA history are counted separately. Snapshots, page rounding, alignment, TP replication, draft buffers, workspace and output-token reserve are excluded.';
 L('SERVING / TWO KINDS OF MEMORY, ONE VALID PREFIX',30,44,'faint',{small:true});
 N('admission',30,83,1390,50,'Cache-affinity routing + budget-based admission','Separate long/short request cost classes; preserve useful prefixes',infraInfo('Fleet-level scheduling','The reported system uses cache-aware affinity and admission budgets to handle requests with widely different costs. The atlas does not model queues or predict TTFT.','request → affinity / budget → worker'));
 G('prefill-group',30,186,430,650,split?'PREFILL POOL':'PREFILL PHASE',split?'Independent weights':'One shared replica',{titleOffset:62});
 G('cache-group',530,186,400,650,'HYBRID CACHE','Joint lifecycle');
 G('decode-group',990,186,430,650,split?'DECODE POOL':'DECODE PHASE',split?'Independent weights':'Same replica');
 N('prefill-weights',50,247,390,52,'Weight residency scenario',`≈ ${gb(d.weights)} per full replica`,k3Info('Mixed storage, not uniformly four-bit','The released configuration targets routed expert weights with MXFP4 and excludes attention, shared experts, ordinary dense FFN projections, vocabulary head, vision and projector. The estimate uses exact routed matrix counts plus BF16 for the rounded remainder. It is not the sum of checkpoint files.','MXFP4 routed: 0.5 + 1/32 bytes / weight\nother ≈ 2 bytes / weight',[['Chosen mode',s.weightMode],['Hardware','Hopper support is backend-dependent']]),'accent');
 N('prefill-kernels',50,365,390,64,'Prefill the uncached prompt suffix','From restored B (or zero) to prompt end',infraInfo('Prefill computes all prompt states','KDA is parallel inside chunks and recurrent across them. Intra-device context parallelism can expose more work without cross-device communication; cross-device KCP is a distinct mechanism.','prompt → projected inputs → states / logits',[['Prompt tokens',fmt(s.context)]],[['training','Inspect KCP'],['hardware','Inspect memory / compute constraints']]),'accent');V(70,299,365,'prefill-weights','prefill-kernels');V(70,133,247,'admission','prefill-weights');
 N('first-token',50,535,390,50,'First-token seed','Sample from final prompt-position logits',infraInfo('Prefill seeds decode','The last prompt position predicts the first generated token. Decode then consumes generated tokens one at a time or through a separate speculative scheme.','x_next ~ sample(logits_last)'));V(70,429,535,'prefill-kernels','first-token');
 N('handoff',50,706,390,68,split?'Export a complete continuation':'Reuse the in-process continuation',split?'KDA state + conv history + MLA KV':'No inter-pool cache transfer',infraInfo('A state transfer must be complete','KDA states and matching MLA cache move together. Convolution history must also be restored. Head-aligned transfer can support different prefill/decode TP degrees; the drawing does not assume uniform physical cache sharding.','continuation = {S_KDA, conv, KV_MLA, token}'),'comm');V(70,585,706,'first-token','handoff');
 N('kda-cache',555,247,350,64,'KDA running state · 69 layers',`${gb(d.cache.recurrent)} · independent of context`,k3Info('The fixed part of serving memory','Each KDA layer has 96 heads with a 128×128 recurrent matrix per request. The selected 2- or 4-byte state precision is an accounting assumption, not a guarantee of stable numerical accuracy.','B_KDA = B × 69 × 96 × 128² × b_state',[['State bytes',String(s.statePrecision)],['Snapshots','Not included in running-state total']]),'state');
 N('conv-cache',555,353,350,52,'Short-convolution buffers',`${gb(d.cache.conv)} · BF16 history`,k3Info('Small state is still state','The budget includes three width-12288 convolution buffers of kernel width four per KDA layer and request, at two bytes each. Runtime buffering may differ.','B_conv = B × 69 × 3 × 12288 × 4 × 2'),'state');
 N('mla-cache',555,451,350,70,'MLA history · 24 global layers',`${gb(d.cache.mla)} · grows with context`,k3Info('Global attention retains token history','Compressed mode counts 512 normalized latent plus64 unrotated shared-key channels. Expanded reference mode stores head-specific K and V. NoPE means no rotation; it does not remove the shared branch.','compressed: B × S × 24 × 576 × b_KV\nexpanded: B × S × 24 × 96 × 320 × b_KV',[['Mode',s.cacheMode],['Increment / token',`${fmt(d.cache.mlaPerToken/1024,1)} KiB`]]),'state');
 W([[440,382],[489,382],[489,279],[555,279]],'prefill-kernels','kda-cache',{kind:'state'});
 W([[440,411],[509,411],[509,486],[555,486]],'prefill-kernels','mla-cache',{kind:'state'});
 N('prefix-match',555,616,350,64,'Prefix checkpoint lookup',`MLA match ${fmt(s.match)} → resume ${fmt(d.hit)}`,infraInfo('A matching hash is necessary, not sufficient','Before a new request prefills, look up previously retained prefixes. A reusable boundary must have matching MLA entries and a persisted checkpoint in every KDA cache group. The example uses 512-token hashes inside 6144-token physical blocks; these are example granularities, not universal constants.','B_hit = max { b ≤ matched :\n  MLA(b) ∧ every KDA group has checkpoint(b) }',[['Example match',fmt(s.match)],['Common checkpoints','512, 2560, 4096, 6144'],['Selected hit',fmt(d.hit)]]),'accent');
 W([[905,279],[951,279],[951,632],[905,632]],'kda-cache','prefix-match',{kind:'state'});
 W([[905,486],[941,486],[941,661],[905,661]],'mla-cache','prefix-match',{kind:'state'});
 N('restore',555,750,350,50,'Restore private prefix at B','Then finish the uncached prompt suffix',infraInfo('Prefix consistency is an execution contract','Pin hit blocks across all groups before allocation. Copy into private running state, COW a partial MLA page, and exclude unready blocks from matching. Evicting one group atomically invalidates sibling checkpoints.','shared snapshot at B → private continuation at B'),'state');V(575,680,750,'prefix-match','restore');
 N('decode-weights',1015,247,380,52,split?'Decode weight replica':'Shared weight replica',`≈ ${gb(d.weights)} · ${s.replicaGPUs} GPUs assumed`,k3Info('Sparse activation is not sparse residency','All expert weights need a placement or offload plan. Split P/D normally has two independent model replicas. The per-GPU division is an idealized budget, not a fit certificate.','ideal average = (weights + logical cache) / GPUs'),'accent');
 N('decode-kernel',1015,365,380,64,'Fused recurrent decode','ShortConv → normalize → KDA → gate',infraInfo('Decode touches state, not the entire KDA past','Dedicated KDA kernels fuse state update operations. MLA still reads its global history. Optional speculative decoding replays cached projected inputs to rebuild accepted states rather than writing a full state snapshot for every draft token.','running state + token → next state / logits',[['Optional MTP','Separate draft path; not added to base config']]),'accent');V(1035,299,365,'decode-weights','decode-kernel');
 N('decode-sample',1015,535,380,50,'Sample next token','Append MLA history; advance KDA state',infraInfo('The loop is autoregressive','Each generated token updates both parts of the hybrid state. The next token cannot reuse a stale KDA state with newer MLA history.','token_t → state_t → logits_t → token_(t+1)'));V(1035,429,535,'decode-kernel','decode-sample');
 W([[1395,560],[1430,560],[1430,338],[1190,338],[1190,365]],'decode-sample','decode-kernel',{kind:'residual'});
 N('idle-prefix',1015,641,380,52,'Idle prefix / eviction','Retain synchronized reusable state',infraInfo('Serving and agentic retention differ by phase','The serving cache manages reusable immutable boundaries. The reported agentic RL system also has a DRAM write-back pool between partial-rollout iterations.','active → checkpoint / retain → evict / restore'),'state');V(1035,585,641,'decode-sample','idle-prefix');
 N('resume',1015,750,380,50,'Decode from the complete prompt','All uncached prompt tokens processed',infraInfo('Resume after transfer dependencies complete','A target worker may resume only after all required state is ready and compatible with its placement. Transport payload divided by line rate is not request latency.','wait(state ready) → next forward pass'),'comm');W([[555,775],[495,775],[495,445],[470,445],[470,397],[440,397]],'restore','prefill-kernels',{kind:'state'});
 W([[1395,775],[1413,775],[1413,404],[1395,404]],'resume','decode-kernel',{kind:'comm'});
 W([[70,774],[70,858],[1215,858],[1215,800]],'handoff','resume',{kind:split?'comm':'state'});
 L('PREFIX EXAMPLE / HASH BLOCKS ARE NOT PHYSICAL ALLOCATION BLOCKS',30,893,'faint',{small:true});
 for(let i=1;i<=12;i++){const x=30+(i-1)*115,end=i*512;N(`hash-${i}`,x,922,104,40,String(end),'',infraInfo(`Hash boundary ${end}`,end<=d.hit?'This endpoint lies in the reused prefix.':'This endpoint is not reused under the current common-boundary lookup.','hash granularity: 512 tokens'),end<=d.hit?'state':'neutral','page');}
 L(`One physical block in this example: 6144 tokens · prompt matches ${s.match} · common resumable prefix ${d.hit}`,30,1002,'muted');
 return g;
}

function hardwareScene(s){
 const {h,world,roofTF,ridge}=derived(s),g=makeScene(1320,790);
 const I=(title,body,formula,facts=[])=>makeInfo('HARDWARE / DOCUMENTED SPECIFICATION',title,body,formula,facts,h.sources);
 const N=(id,x,y,w,ht,name,sub,info,tone='neutral',kind='op')=>addNode(g,id,x,y,w,ht,name,sub,{info,tone,kind});
 g.defaultSelected='nv-fabric';
 g.caption='Logical DGX H100/H200 node; placement is schematic, not a board/cabling drawing. NVLink connectivity is bundled through four switches. Compute, storage and management networks are distinct planes.';
 addLabel(g,'INSIDE ONE 8-GPU NODE',28,43,'faint',{small:true});
 addLabel(g,`${h.name} · ${h.memoryType} · four third-generation NVSwitches`,28,74,'muted');
 addGroup(g,'board',250,111,1036,481,'NVLINK DOMAIN','logical connectivity · not to physical scale');
 const gpuInfo=I(h.name,`Each accelerator has ${h.memoryGB} GB advertised ${h.memoryType} and ${h.hbmGBs/1000} TB/s peak HBM bandwidth. A selected model is not automatically guaranteed to fit. Allocatable memory is lower and placement matters.`,`dense BF16 ceiling ≈ ${fmt(h.bf16TF)} TFLOP/s`,[['Nominal memory',`${h.memoryGB} GB`],['HBM bandwidth',`${h.hbmGBs/1000} TB/s`],['NVLink / GPU','900 GB/s, both directions'],['Max GPU TDP','700 W']]);
 for(let i=0;i<8;i++){
  const x=278+(i%4)*249,y=i<4?169:440;
  N(`gpu-${i}`,x,y,220,87,`GPU ${i.toString().padStart(2,'0')}`,`${h.name} · ${h.memoryGB} GB`,gpuInfo,i===0?'accent':'neutral','gpu');
  // Decorative memory stacks represent packages abstractly, not an exact stack count.
  if(i<4)addWire(g,[[x+110,y+87],[x+110,316]],`gpu-${i}`,'nv-fabric',{kind:'nvlink',arrow:false});
  else addWire(g,[[x+110,379],[x+110,y]],'nv-fabric',`gpu-${i}`,{kind:'nvlink',arrow:false});
 }
 N('nv-fabric',278,316,967,63,'NVSwitch fabric','4 switches · all eight GPUs connected',I('A switched fabric, not an eight-GPU mesh','Each H100 GPU has 18 fourth-generation NVLinks spread across four third-generation NVSwitches. The 900 GB/s number sums both directions per GPU; it is neither one directional link nor an inter-node NIC bandwidth.','18 links / GPU → 4 NVSwitches',[['Per-GPU aggregate','900 GB/s bidirectional'],['Directional sum','450 GB/s ideal per direction'],['Fabric generation','NVLink 4 / NVSwitch 3']]),'comm','fabric');
 addLabel(g,'NVLink · scale up',286,290,'comm',{small:true});
 addLabel(g,'PCIe / host attachments are collapsed below',278,568,'faint',{small:true});
 N('cpu',28,224,188,66,'Dual host CPUs','2 TB system memory',I('Host and control plane','DGX H100/H200 uses two Intel Xeon Platinum 8480C CPUs and 2 TB system memory. The host handles orchestration and I/O; GPU collectives need not stage their payload through host DRAM.','host memory ≠ GPU HBM'));
 N('storage',28,409,188,65,'Local NVMe','OS + data cache',I('Local storage','The DGX specification lists 2 × 1.92 TB OS drives and 8 × 3.84 TB data-cache drives. Checkpoint storage can also be remote. These capacities do not become HBM.','NVMe → host / GPU data path'));
 N('pcie',251,624,1035,35,'PCIe Gen5 / GPU–NIC locality','',I('PCIe is a separate interconnect','The GPU–NIC and host I/O paths use PCIe. A PCIe 5.0 x16 interface has a raw encoded data rate of about 63 GB/s per direction before transaction overhead; the commonly quoted 128 GB/s is rounded bidirectional aggregate.','32 GT/s × 16 × (128/130) / 8 ≈ 63 GB/s / direction',[['Directionality','Do not compare 128 to one-way 50'],['GPU direct RDMA','Avoids host-memory payload staging']]),'blue','pill');
 addWire(g,[[216,257],[232,257],[232,641],[251,641]],'cpu','pcie',{kind:'host',arrow:false});
 addWire(g,[[216,442],[232,442]],'storage','',{kind:'host',arrow:false});
 for(let i=0;i<8;i++){
  const x=251+i*131;
  N(`nic-${i}`,x,706,117,48,`NIC ${i}`,`400 Gb/s`,I('ConnectX-7 compute NIC','The DGX node has eight single-port ConnectX-7 adapters for the compute network. Storage/management adapters are separate. 400 Gb/s translates to 50 GB/s line rate in one direction, before protocol overhead.','400 Gb/s ÷ 8 = 50 GB/s',[['Compute NICs / node','8'],['Node injection ceiling','400 GB/s per direction'],['Physical OSFP ports','4 ports serving 8 adapters']]),'state','nic');
  addWire(g,[[x+58,659],[x+58,706]],'pcie',`nic-${i}`,{kind:'host',arrow:false});
 }
 addLabel(g,'SCALE OUT →',28,733,'state',{small:true});
 addLabel(g,'8 independent compute-network adapters · storage and management adapters not shown',251,781,'faint',{small:true});
 return g;
}
function networkScene(s){
 const d=derived(s),g=makeScene(1400,900),{N,W,L,G,V}=kit(g),mode=s.trafficMode;
 g.defaultSelected='traffic-formula';
 g.caption='A traffic contract above a logical scale-out substrate. MoE: balanced off-rank payload for one layer, one direction. KCP: per-rank all-gather send volume for one layer, all 96 heads locally (no TP). Cache: one full request handoff, before replication. No protocol, startup, congestion or compute-overlap prediction.';
 L('NETWORK / THE PAYLOAD IS DETERMINED BY THE ARCHITECTURE',30,43,'faint',{small:true});
 const contracts={
 moe:{title:'Latent expert dispatch',sub:`${fmt(s.tokens)} source tokens · top 16 · BF16 vectors`,formula:'V_send = T × 16 × 3584 × 2 × (R−1)/R',body:'A rank routes T full local tokens into 16 latent-vector destinations. The off-rank fraction (R−1)/R assumes uniform balanced destinations. Full-width expert routing would double vector bytes. Combine has comparable logical vector volume but is not included here; expert migration, probabilities, indices and alignment are extra.',target:'Expert owner ranks',targetSub:'Rank balance ≠ per-expert uniformity'},
 kcp:{title:'KDA context-parallel fragments',sub:`One KDA layer · 96 local heads · ${s.statePrecision}-byte matrices`,formula:'V_local = 96 × (128² + 128²) × b\nV_send = (R−1) × V_local',body:'Every rank contributes both its cumulative transition A and zero-state contribution B. The all-gather payload is fixed with respect to sequence length. Ordered affine composition reconstructs the incoming state; simply summing B is wrong. The count is for one KDA layer, not all69, and assumes all96 heads are local.',target:'Context-parallel ranks',targetSub:'Ordered scan · reset at document boundaries'},
 cache:{title:'Prefill → decode continuation',sub:`One request · ${fmt(s.context)} tokens · ${s.cacheMode} MLA`,formula:'V = B_MLA(S) + B_KDA + B_conv',body:'Transfer a complete request continuation: growing MLA history, fixed KDA state and convolution history. Source and destination must agree on the prefix boundary and precision. The count is a logical full-request transfer, not per-rank traffic under a specific TP layout. The rank selector is irrelevant to this full-request payload.',target:'Decode worker / replica',targetSub:'All cache groups ready at one boundary'}
 };
 const c=contracts[mode];
 N('payload-origin',30,99,390,64,c.title,c.sub,infraInfo(c.title,c.body,c.formula),mode==='cache'?'state':'accent');
 N('traffic-formula',490,99,430,64,mode==='kcp'?'Exchange (A, B), not only state':mode==='moe'?'Communicate 3584, not 7168':'Two cache types; one boundary',`${fmt(d.traffic/1024**2,2)} MiB logical send payload`,infraInfo('Selected payload contract',c.body,c.formula,[['Scope',mode==='cache'?'One full request':'One layer / one rank'],['Ranks',mode==='cache'?'Not applied':String(s.nodes)],['Logical sent',gb(d.traffic)]]),'comm');
 N('payload-target',990,99,380,64,c.target,c.targetSub,infraInfo('Consumer contract','Collective and cache transport only preserve semantics when ownership, token order, dtype and boundary agree. A link alone does not establish correctness.','validate → transfer → synchronize → consume'));
 W([[420,131],[490,131]],'payload-origin','traffic-formula');W([[920,131],[990,131]],'traffic-formula','payload-target');
 L('SCALE UP STAYS LOCAL. SCALE OUT CROSSES THE COMPUTE FABRIC.',30,250,'faint',{small:true});
 G('source-node',30,301,410,386,'SOURCE NODE','logical Hopper example');
 G('destination-node',970,301,410,386,'DESTINATION NODE','not a rack-port map');
 const gpuInfo=makeInfo('HARDWARE / ILLUSTRATIVE PLACEMENT','GPU ownership and locality','This illustrative DGX H100/H200 substrate is not asserted to be Moonshot’s deployed cluster. An eight-GPU node has a switched NVLink domain; inter-node data uses separate compute NICs.','GPU HBM ↔ local NVLink domain',[['Selected accelerator',d.h.name]],['dgx','fabric']);
 N('source-gpu',50,365,370,64,'Local GPU / NVLink domain','Eight GPUs · four NVSwitches',gpuInfo,'accent');
 N('target-gpu',990,365,370,64,'Remote GPU / NVLink domain','Different ownership or context segment',gpuInfo,'accent');
 N('source-nic',50,555,370,54,`Compute NIC · rail ${s.rail}`,'400 Gb/s = 50 GB/s per direction',makeInfo('NETWORK / DOCUMENTED LINE RATE','One compute-network adapter','DGX H100/H200 exposes eight single-port 400Gb/s compute adapters. Eight rails can ideally inject 400GB/s per direction in aggregate; one flow does not automatically achieve that.','400 gigabits / 8 = 50 gigabytes',[['Node aggregate','8 × 50 = 400 GB/s / direction']],['dgx','nccl']),'state');
 N('target-nic',990,555,370,54,`Peer compute NIC · rail ${s.rail}`,'GPUDirect RDMA path (logical)',makeInfo('NETWORK / ILLUSTRATIVE PATH','NIC to device memory','GPUDirect RDMA can move payload without staging the data through CPU DRAM. PCIe topology and device locality still constrain the path.','NIC ↔ PCIe ↔ HBM',[],['dgx']),'state');
 V(70,429,555,'source-gpu','source-nic','host');W([[1010,555],[1010,429]],'target-nic','target-gpu',{kind:'host'});
 L('PCIe / locality',100,499,'blue',{small:true});L('PCIe / locality',1040,499,'blue',{small:true});
 N('fabric',510,414,360,74,'Leaf / spine compute fabric',`${s.oversub}:1 oversubscription assumed`,makeInfo('NETWORK / ANALYTIC SCENARIO','A logical fabric, not a wiring specification','The line-rate sensitivity model divides by an assumed oversubscription ratio and multiplies by an assumed efficiency. Real port maps, collective algorithms, routes and congestion are not simulated.','B_eff = 50 GB/s × u / oversubscription',[['Assumed efficiency',`${s.utilization}%`],['Effective rate',`${fmt(d.networkGBs)} GB/s`]],['dgx','nccl']),'comm');
 W([[420,582],[477,582],[477,451],[510,451]],'source-nic','fabric',{kind:'comm'});
 W([[870,451],[923,451],[923,582],[990,582]],'fabric','target-nic',{kind:'comm'});
 L(mode==='cache'?'Full-request handoff':`${s.nodes} participating ranks`,690,345,'muted',{anchor:'middle'});
 L('Storage and management planes are separate.',690,629,'faint',{small:true,anchor:'middle'});
 N('bandwidth-time',30,760,430,65,'Bandwidth-only transport component',`${fmt(d.networkMs,2)} ms at ${fmt(d.networkGBs,2)} GB/s`,makeInfo('CALCULATED / UNDER ASSUMPTIONS','Not latency, not tokens per second','Only payload divided by effective link rate is evaluated. Per-message startup, chunking, protocol overhead beyond the efficiency factor, route sharing, expert imbalance and overlap may change observed time.','T_bandwidth = V / B_eff',[],['nccl']),'comm');
 N('extra-traffic',490,760,430,65,mode==='moe'?'Combine and migration are additional':mode==='kcp'?'State composition is ordered':'Physical layout may increase bytes',mode==='moe'?'No fabricated end-to-end 2× speedup':mode==='kcp'?'A₂A₁, A₂B₁ + B₂':'TP replication / padding / page metadata',infraInfo('Terms not included',c.body,c.formula));
 N('network-boundary',990,760,380,65,'HBM ≠ NVLink ≠ NIC','4.8 TB/s ≠ 900 GB/s ≠ 50 GB/s',makeInfo('UNITS / DIRECTIONALITY','Never compare unlike bandwidth figures','H200 peak HBM bandwidth, per-GPU bidirectional NVLink aggregate and a one-way NIC rate refer to three different substrates. The H100 alternative has a different HBM peak.','capacity ≠ bandwidth ≠ latency',[],['h200','dgx']),'state');
 return g;
}
export function buildScene(s){
 const builders={model:modelScene,training:trainingScene,serving:servingScene,hardware:hardwareScene,network:networkScene};
 if(!builders[s.view])throw new RangeError('Unknown view');return builders[s.view](s);
}
