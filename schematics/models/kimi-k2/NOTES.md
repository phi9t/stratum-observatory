# Reading notes: Kimi-K2-Instruct

Checkpoint identity and all code revisions are in sources.json. This is the original
Instruct checkpoint, not Instruct-0905, Thinking, K2.5, or a DeepSeek checkpoint.

## Publisher reference, read in full

- DeepseekV3RMSNorm: cast activations to float32, mean square over last axis,
  rsqrt(variance + epsilon), multiply gain after casting back. No centering/bias.
- DeepseekV3RotaryEmbedding: nonpersistent frequency/cos/sin buffers, growing cache.
  LinearScaling and DynamicNTKScaling are unused (rope_scaling.type=yarn).
- DeepseekV3YarnRotaryEmbedding: interpolate/extrapolate inverse frequencies with
  correction bounds floor/ceil from beta_fast/beta_slow. Equal bounds need a small
  denominator offset. Both betas are 1, so the transition is particularly narrow.
  mscale/mscale_all_dim both 1: their amplitude ratio cancels, but attention still
  multiplies its inverse-square-root scale by yarn_get_mscale(factor,1)^2.
- apply_rotary_pos_emb: explicitly rearranges adjacent channels into two halves
  before rotate_half; merely applying NeoX half rotation to the original layout is wrong.
- DeepseekV3MLP: gate_proj and up_proj run in parallel; SiLU(gate) * up; down_proj.
  All are biasless. Dense intermediate 18432; each expert intermediate 2048.
- MoEGate: flatten B*S, float32 linear [384,7168], sigmoid; add learned correction
  bias for selection ONLY. Sum group top two, select groups, mask excluded groups,
  choose eight experts unsorted; gather weights from ORIGINAL sigmoid scores,
  renormalize with 1e-20, multiply 2.827. Exactly one group selected from one:
  group filtering is an identity for this checkpoint. noaux_tc asserts eval mode.
- DeepseekV3MoE: 384 routed MLPs plus separate always-on shared MLP. Gate sees no
  shared expert. moe_infer counts assignments, sorts/gathers tokens, executes each
  nonempty expert, restores assignment order, weighted sum, adds shared result.
  ep_size defaults to 1 in configuration; distributed all-to-all is implementation
  plumbing excluded from the clean arithmetic.
- DeepseekV3Attention: q_a [1536,7168], RMSNorm with constructor default 1e-6,
  q_b [12288,1536], reshape [B,64,S,192], split content 128 / rotary 64.
  kv_a [576,7168], split latent 512 / positional 64; RMSNorm latent (default 1e-6),
  kv_b [16384,512], reshape [B,64,S,256], split content key 128 / value 128.
  Positional key is [B,1,S,64] broadcast to all heads; YaRN rotate q/k positional
  slices, concatenate content and rotated slices. Reference DynamicCache stores
  EXPANDED key [B,64,T,192] and value [B,64,T,128], not the compressed latent.
  q*k transpose, scale, add causal/padding mask, float32 softmax, cast back,
  dropout (zero in this config), weights*values, merge [B,S,8192], o [7168,8192].
- DeepseekV3FlashAttention2: same projection and cache computation; pad value width
  up to query width for kernel then truncate; unpad/pad for variable lengths; old
  flash kernels have a causal alignment caveat. Backend alternative, not another
  architecture. The page uses eager arithmetic explicitly, not an assumed default.
- DeepseekV3DecoderLayer: input RMSNorm -> attention -> residual add -> post-attention
  RMSNorm -> FFN -> residual add. MoE iff routed experts exist AND index>=first_k_dense_replace
  AND index%moe_layer_freq==0. Actual indices: dense 0, MoE 1..60.
- DeepseekV3PreTrainedModel: initialization/framework base, no inference computation.
- DeepseekV3Model: embedding [163840,7168], positions start at usable past length,
  causal mask preparation, iterate decoder layers, final RMSNorm. No learned position
  embedding. DynamicCache legacy conversion is API plumbing.
- DeepseekV3ForCausalLM: untied biasless head [163840,7168], cast logits float32.
  Shifted training loss and generation input slicing/beam reorder excluded.
- DeepseekV3ForSequenceClassification: unused task head, architecture names CausalLM.

## Independent serving implementation, read in full

vLLM deepseek_v2.py includes unrelated DeepSeek-v1 MHA, GLM, indexer/V3.2 and Eagle
branches. Kimi config has no index_topk and nonzero MLA widths: none of those belong
on the drawing. Decoder has the same dense/MoE index predicate.
DeepseekV2MLP fuses gate/up into MergedColumnParallelLinear and SiluAndMul.
DeepseekV2MoE uses FusedMoEFactory with sigmoid, correction bias, normalization,
scaling and optional shared-expert fusion. Physical redundant expert replicas and
parallel partitions are runtime allocations, not logical checkpoint experts.
DeepseekV2MLAAttention fuses q_a+kv_a, passes MLAModules to MLA wrapper; normalization
uses config epsilon (same value as publisher defaults here). get_rope uses interleaved
rotation and translated rope_parameters; this page does not claim to test that config
translation or GPU kernels. Decoder carries residual separately and fuses add+norm;
FP16 overflow scaling branches differ from the unscaled BF16/reference diagram.
FlashAttnMLAImpl.forward_mqa explicitly splits cache into kv_lora_rank latent and
qk_rope_head_dim positional key, passes latent as value and absorbed query as q_v.
That proves compressed cache width; GPU implementation/performance is not verified.

## Parameter storage

Pinned safetensors index includes inverse frequency and FP8 weight_scale_inv tensors
in addition to learned weights. These are not trainable parameters. Reconcile by
constructing each learned shape and each scale grid using config weight_block_size;
validate scale names against the pinned index, without downloading any weights.

## Decisions

Use checkpoint publisher code over a modern family implementation. Clean TypeScript
covers eval eager full-sequence decoder to logits with no padding/cache; cached paths
are drawn and accounted separately. Numerical fixture executes unchanged pinned
module definitions via AST to avoid legacy transformers import glue; all arithmetic
comes from upstream classes. Toy dimensions change sizes, keep low-rank q/kv,
YaRN, correction bias, top-k>one, shared expert and both layer kinds. For testing,
two dense and two MoE layers are used. No exact public training forward implementation
has been established; do not substitute a family training recipe or optimizer.

Final backend scope: publisher FlashAttention pads V to the query-head width before cache update. The expanded cache calculator and CPU parity cover the eager path only. The MoonshotAI/Kimi-K2 public repository tree at 1b4022bbb7187cf4011a8bdf0b4cd10e2daa26c4 was also searched; no exact checkpoint training forward was found.
